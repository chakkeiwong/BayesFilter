"""Complete sequential enclosure against original public records and call order."""

import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_controller_tf as native
from bayesfilter.inference import sequential_map_covariance as public
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_lifecycle_original import _compare_original
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64
CASES = ('terminal', 'terminal_reject', 'symmetric', 'recenter', 'fit_reject',
    'factor_one', 'factor_two', 'factor_two_reuse', 'stationary_budget',
    'moving_budget', 'nonfinite', 'paired', 'scaled_search', 'score_disabled',
    'resolvable', 'scalar_locator', 'batched_locator', 'locator_budget')


def fixture(case):
    dimension = 5 if 'two' in case else 3
    factor = case.startswith('factor') or case == 'stationary_budget'
    cfg = public.SequentialMapCovarianceConfig(locator_policy='center_first', max_attempts=2,
        search_sample_count=4, regression_sample_count=24, terminal_sample_count=24,
        max_exact_evaluations=256, initial_radius=.25, terminal_score_max_abs=1e-5,
        refinement_geometry_policy='factor_correlation' if factor else 'full_symmetric',
        structured_max_factors=2 if 'two' in case else 1, reuse_search_scores=case.endswith('reuse'),
        structured_holdout_score_relative_rmse=.001,
        score_holdout_relative_rmse=1e-12 if case.endswith('reject') else .35,
        record_refinement_movement_diagnostics=True, pair_disjoint_score_holdout=case == 'paired',
        dimension_scaled_search=case == 'scaled_search', orthogonal_antithetic_search=case == 'scaled_search',
        require_proposal_score_reduction=case != 'score_disabled',
        proposal_score_acceptance_policy='resolvable_decrease' if case == 'resolvable' else 'fractional')
    if factor and case != 'stationary_budget':
        scalar, batched, _ = _inputs(dimension, 4)
    else:
        def batched(rows):
            values = -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1)
            scores = -rows - .4 * rows ** 3
            if case == 'nonfinite':
                values = tf.fill(tf.shape(values), tf.constant(float('nan'), D))
            return values, scores

        def scalar(row):
            values, scores = batched(row[None])
            return values[0], scores[0]

    center = (tf.zeros([dimension], D) if case.startswith('terminal') or case == 'stationary_budget' else
        tf.linspace(tf.constant(-.5, D), tf.constant(.8, D), dimension) if case == 'recenter' else
        tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension))
    starts, locator = center[None], None
    if case in ('stationary_budget', 'moving_budget'):
        cfg = dataclasses.replace(cfg, max_exact_evaluations=1)
    if 'locator' in case:
        cfg = dataclasses.replace(cfg, locator_policy='multistart', locator_max_iterations=2,
            locator_max_line_search_iterations=5, max_exact_evaluations=1 if case == 'locator_budget' else 256)
        starts = tf.stack((center + .2, center - .3))
        locator = batched if case in ('batched_locator', 'locator_budget') else None
    return scalar, batched, locator, cfg, starts, tf.ones([dimension], D)


def _normalize_events(events):
    for event in events:
        if 'delivery_mode' in event:
            assert event.pop('delivery_mode') == 'buffered_after_compiled_locator'
    return events


@pytest.mark.parametrize('case', CASES)
def test_complete_original_records_and_target_order(case, request):
    raw_scalar, raw_batch, raw_locator, cfg, starts, scale = fixture(case)
    dimension, start_count = int(starts.shape[1]), int(starts.shape[0])
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    points = tf.Variable(tf.zeros([4096, dimension], D), trainable=False)

    def observe(rows):
        count = tf.shape(rows, out_type=tf.int64)[0]
        indices = (tf.range(count) + calls.read_value())[:, None]
        points.scatter_nd_update(indices, rows)
        calls.assign_add(count)

    def scalar(row):
        observe(row[None])
        return raw_scalar(row)

    def batched(rows):
        observe(rows)
        return raw_batch(rows)

    locator = batched if raw_locator is not None else None
    checkpoint = FrozenCheckpoint('3582b4ac', 'sequential_public_enclosure_' + case)
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    before_events = []
    before = original.estimate_sequential_map_covariance(scalar, starts,
        batched_value_and_score_fn=batched, batched_locator_value_and_score_fn=locator,
        scale=scale, config=cfg, progress_callback=before_events.append).payload()
    before_calls, before_points = int(calls), points[:int(calls)].numpy().tolist()
    calls.assign(0)
    points.assign(tf.zeros_like(points))
    search_count = public.dimension_scaled_search_count(dimension) if cfg.dimension_scaled_search else cfg.search_sample_count
    owner = native.SequentialController(scalar, batched, locator, start_count, dimension, cfg,
        search_count, progress=True, device=starts.device)
    computed = owner(starts, scale)
    after_events = []
    after = sequential_result(computed, cfg, start_count, dimension, after_events.append).payload()
    report = {'case': case, 'before': before, 'after': after,
        'before_events': before_events, 'after_events': after_events,
        'before_calls': before_calls, 'after_calls': int(calls),
        'before_points': before_points, 'after_points': points[:int(calls)].numpy().tolist(),
        'reference': '3582b4ac', 'frozen_source_sha256': checkpoint.hashes(),
        'scope': 'Complete original public endpoint versus enclosing native candidate, including final mass and reporting',
        'nonclaims': ['Deferred progress does not preserve live callback interruption.',
            'Internal endpoint comparison does not establish public wiring or consumer qualification.']}
    path = Path(request.config.getoption('xmlpath')).parent / f'sequential-controller-{case}.json'
    with path.open('x') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    _compare_original(after, before)
    _compare_original(_normalize_events(after_events), before_events)
    assert int(calls) == before_calls == after['diagnostics']['exact_evaluations']
    np.testing.assert_allclose(points[:int(calls)], before_points, atol=1e-10, rtol=1e-10)
    assert owner.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    assert owner.compiled.experimental_get_tracing_count() == 1


def test_changed_inputs_reuse_compiled_controller_and_hlo(request):
    scalar, batched, locator, cfg, starts, scale = fixture('terminal')
    owner = native.sequential_controller(scalar, batched, locator, 1, 3, cfg, cfg.search_sample_count)
    first = owner(starts, scale)
    original = FrozenCheckpoint('3582b4ac', 'sequential_changed').load('bayesfilter.inference.sequential_map_covariance')
    changed = starts + .002
    other = native.sequential_controller(scalar, batched, locator, 1, 3, cfg, cfg.search_sample_count)
    assert owner is other
    second = owner(changed, scale * 1.1)
    for arguments, result in (((starts, scale), first), ((changed, scale * 1.1), second)):
        expected = original.estimate_sequential_map_covariance(scalar, arguments[0],
            batched_value_and_score_fn=batched, scale=arguments[1], config=cfg).payload()
        _compare_original(sequential_result(result, cfg, 1, 3).payload(), expected)
    before = owner.compiled.experimental_get_compiler_ir(starts, scale)(stage='hlo')
    after = owner.compiled.experimental_get_compiler_ir(changed, scale * 1.1)(stage='hlo')
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'sequential-controller-before.hlo').write_text(before)
    (directory / 'sequential-controller-changed.hlo').write_text(after)

    def stable_hlo(text):
        # Same dummy-source uniquifiers observed in the existing lifecycle
        # tests; instructions, operands, constants and all other metadata stay.
        return re.sub(r'op_name="(zeros(?:_\d+)?)(?:/_\d+)+"(?= source_file="dummy_file_name" source_line=10)',
            r'op_name="\1/UNIQUE"', text)

    assert stable_hlo(before) == stable_hlo(after)
    assert owner.compiled.experimental_get_tracing_count() == 1


def test_report_never_recalculates_factor_admission(monkeypatch):
    from bayesfilter.inference import factor_correlation_geometry as factor

    scalar, batched, locator, cfg, starts, scale = fixture('factor_one')
    owner = native.SequentialController(scalar, batched, locator, 1, 3, cfg, cfg.search_sample_count)
    result = owner(starts, scale)

    def forbidden(*_args, **_kwargs):
        raise AssertionError('Completed factor reporting must not run numerical admission')

    monkeypatch.setattr(factor, 'factor_decisions_program', forbidden)
    record = sequential_result(result, cfg, 1, 3).payload()
    assert record['diagnostics']['history']


def test_overflow_veto_precedes_all_lifecycle_target_calls():
    scalar, batched, _locator, cfg, starts, scale = fixture('batched_locator')
    from bayesfilter.inference.sequential_batched_locator_tf import (
        BufferedBatchedLocator,
    )

    rows = tf.Variable(0, dtype=tf.int64, trainable=False)

    def counted_scalar(point):
        rows.assign_add(1)
        return scalar(point)

    def counted_batch(points):
        rows.assign_add(tf.shape(points, out_type=tf.int64)[0])
        return batched(points)

    reference = BufferedBatchedLocator(counted_scalar, counted_batch, 2, 3,
        cfg.locator_standardized_box_radius, cfg.locator_gradient_tolerance,
        cfg.locator_max_iterations, cfg.locator_max_line_search_iterations,
        cfg.locator_stopping_condition, device=starts.device, capacity=1)
    with pytest.raises(RuntimeError, match='capacity 1 exceeded'):
        reference(starts, scale)
    expected_rows = int(rows)
    rows.assign(0)
    owner = native.SequentialController(counted_scalar, counted_batch, counted_batch, 2, 3,
        cfg, cfg.search_sample_count, progress=True, device=starts.device, trace_capacity=1)
    with pytest.raises(RuntimeError, match='capacity 1 exceeded'):
        owner(starts, scale)
    assert int(rows) == expected_rows


def test_host_only_callback_has_no_eager_retry():
    _scalar, batch, _locator, cfg, starts, scale = fixture('terminal')
    events = []

    def host_only(row):
        events.append('callback')
        row.numpy()
        return tf.constant(0., D), tf.zeros([3], D)

    with pytest.raises((AttributeError, TypeError, ValueError, NotImplementedError)):
        native.SequentialController(host_only, batch, None, 1, 3, cfg, cfg.search_sample_count)(starts, scale)
    assert events == ['callback']


def test_complete_preparation_preserves_original_frozen_derivative_boundary():
    scalar, batched, locator, cfg, starts, scale = fixture('terminal')
    owner = native.SequentialController(scalar, batched, locator, 1, 3, cfg, cfg.search_sample_count)
    original = FrozenCheckpoint('3582b4ac', 'sequential_frozen_public').load(
        'bayesfilter.inference.sequential_map_covariance')
    for prior in (True, False):
        with tf.GradientTape() as tape:
            tape.watch((starts, scale))
            result = (original.estimate_sequential_map_covariance(scalar, starts,
                batched_value_and_score_fn=batched, scale=scale, config=cfg) if prior else
                sequential_result(owner(starts, scale), cfg, 1, 3))
            assert result.accepted
            value = (tf.reduce_sum(tf.convert_to_tensor(result.map_candidate, D))
                + tf.reduce_sum(tf.convert_to_tensor(result.precision, D))
                + tf.reduce_sum(tf.convert_to_tensor(result.covariance, D)))
        assert tape.gradient(value, (starts, scale)) == (None, None)
