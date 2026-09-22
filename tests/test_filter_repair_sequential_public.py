"""Actual public sequential call chain, original records and buffered observers."""

import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_controller_tf as native
from bayesfilter.inference import sequential_map_covariance as public
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_lifecycle_original import _compare_original
from tests.test_filter_repair_sequential_controller import (
    CASES,
    check_complete_original_records_and_target_order,
    fixture,
)

D = tf.float64


@pytest.mark.parametrize('case', CASES)
def test_public_original_records_and_target_order(case, request):
    check_complete_original_records_and_target_order(case, request, use_public=True)


def test_public_reuses_owner_and_buffers_progress_until_numerical_completion(request):
    raw_scalar, raw_batch, _locator, cfg, starts, scale = fixture('terminal')
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def scalar(point):
        calls.assign_add(1)
        return raw_scalar(point)

    def batched(points):
        calls.assign_add(tf.shape(points, out_type=tf.int64)[0])
        return raw_batch(points)

    records, owners, events = [], [], []
    for positions, scaling in ((starts, scale), (starts + .002, scale * 1.1)):
        calls.assign(0)
        observed = []

        def progress(event, observed=observed):
            observed.append((event, int(calls)))

        result = public.estimate_sequential_map_covariance(scalar, positions,
            batched_value_and_score_fn=batched, scale=scaling, config=cfg,
            progress_callback=progress)
        records.append(result.payload())
        owners.append(native._LAST_CONTROLLER[4])
        events.append(observed)
        assert observed[0][0]['stage'] == 'initializer_started'
        assert observed[0][1] == 0
        assert len(observed) > 1
        assert all(count == int(calls) for _, count in observed[1:])
        assert int(calls) == result.diagnostics['exact_evaluations']
    assert owners[0] is owners[1]
    assert owners[0].compiled.experimental_get_tracing_count() == 1
    assert owners[0].compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    checkpoint = FrozenCheckpoint('3582b4ac', 'sequential_public_changed_inputs')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    expected = []
    for positions, scaling in ((starts, scale), (starts + .002, scale * 1.1)):
        expected.append(original.estimate_sequential_map_covariance(raw_scalar, positions,
            batched_value_and_score_fn=raw_batch, scale=scaling, config=cfg).payload())
    report = {'scope': 'Actual public API changing-input reuse and observational progress',
        'records': records, 'original': expected, 'events_and_observed_rows': events,
        'reference': '3582b4ac', 'frozen_source_sha256': checkpoint.hashes(),
        'nonclaims': ['Buffered progress cannot enforce a live deadline.']}
    path = Path(request.config.getoption('xmlpath')).parent / 'sequential-public-reuse.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    _compare_original(records, expected)


def test_public_validation_precedes_target_use():
    calls = []

    def forbidden(point):
        calls.append(point)
        raise AssertionError('Invalid input reached target')

    cfg = public.SequentialMapCovarianceConfig(locator_policy='center_first')
    for starts, scale in (([[0., 0., 0.]], [1., 0., 1.]),
                          ([[0., 0., 0.]], [1., 1.]),
                          ([[0., 0., 0.], [1., 1., 1.]], None),
                          ([0., 0., 0.], None)):
        with pytest.raises(ValueError):
            public.estimate_sequential_map_covariance(forbidden, starts, scale=scale, config=cfg)
    assert calls == []


def test_public_unsupported_callback_has_no_eager_retry():
    _, batch, _, cfg, starts, scale = fixture('terminal')
    calls, events = [], []

    def host_only(row):
        calls.append('traced')
        row.numpy()
        return tf.constant(0., D), tf.zeros([3], D)

    with pytest.raises((AttributeError, TypeError, ValueError, NotImplementedError)):
        public.estimate_sequential_map_covariance(host_only, starts,
            batched_value_and_score_fn=batch, scale=scale, config=cfg,
            progress_callback=events.append)
    assert calls == ['traced']
    assert [row['stage'] for row in events] == ['initializer_started']


def test_public_overflow_delivers_no_partial_progress(monkeypatch):
    scalar, batched, locator, cfg, starts, scale = fixture('batched_locator')
    owner = native.SequentialController(scalar, batched, locator, 2, 3, cfg,
        cfg.search_sample_count, progress=True, device=starts.device, trace_capacity=1)
    monkeypatch.setattr(public, 'sequential_controller', lambda *args, **kwargs: owner)
    events = []
    with pytest.raises(RuntimeError, match='capacity 1 exceeded'):
        public.estimate_sequential_map_covariance(scalar, starts,
            batched_value_and_score_fn=batched, batched_locator_value_and_score_fn=locator,
            scale=scale, config=cfg, progress_callback=events.append)
    assert [row['stage'] for row in events] == ['initializer_started']


def test_public_preparation_keeps_frozen_derivative_boundary():
    scalar, batched, _locator, cfg, starts, scale = fixture('terminal')
    with tf.GradientTape() as tape:
        tape.watch((starts, scale))
        result = public.estimate_sequential_map_covariance(scalar, starts,
            batched_value_and_score_fn=batched, scale=scale, config=cfg)
        assert result.accepted
        total = (tf.reduce_sum(result.map_candidate) + tf.reduce_sum(result.precision)
            + tf.reduce_sum(result.covariance))
    assert tape.gradient(total, (starts, scale)) == (None, None)
