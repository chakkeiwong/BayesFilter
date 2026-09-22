"""Independent resolution boundaries and real public no-use/diagnostic checks."""

import json
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_controller_tf as native
from bayesfilter.inference import sequential_map_covariance as public
from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
    locate_block_coordinate_center,
)
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import _sequential_config
from tests.test_filter_repair_block_center import _assert_record
from tests.test_filter_repair_block_controller import _diagnostic_json

D = tf.float64


def _host(value):
    return tf.nest.map_structure(lambda item: item.numpy().tolist(), value)


def _write(request, name, payload):
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / name).open('x') as output:
        json.dump(_diagnostic_json(payload), output, indent=2, allow_nan=False)
        output.write('\n')


def test_exact_resolution_boundaries():
    cases = []
    for before in (-1e100, -1., -float.fromhex('0x1p-1022'), -math.ulp(0.),
                   -0.0, 0.0, math.ulp(0.), float.fromhex('0x1p-1022'), 1., 1e100):
        adjacent = math.nextafter(before, math.inf)
        for after, expected in ((before, False), (adjacent, True),
                                (math.nextafter(adjacent, math.inf), False),
                                (math.nextafter(before, -math.inf), False)):
            for executed, promoted in ((True, True), (False, True), (True, False), (False, False)):
                cases.append((before, after, promoted, executed, expected and executed and promoted))
    cases.extend((a, b, True, True, False) for a, b in (
        (-math.inf, -1.), (1., math.inf), (math.nan, 1.), (1., math.nan),
        (float.fromhex('0x1.fffffffffffffp+1023'), math.inf)))
    compiled = tf.function(native._objective_resolution_flags, jit_compile=True, autograph=False,
        input_signature=[tf.TensorSpec([None], D), tf.TensorSpec([None], D),
            tf.TensorSpec([None], tf.bool), tf.TensorSpec([None], tf.bool)])
    actual = compiled(tf.constant([row[0] for row in cases], D), tf.constant([row[1] for row in cases], D),
        tf.constant([row[2] for row in cases]), tf.constant([row[3] for row in cases]))
    assert actual.numpy().tolist() == [row[4] for row in cases]


def _fixture():
    precision = tf.constant([[4., .2, .1, 1.], [.2, 5., .5, .2],
        [.1, .5, 6., .3], [1., .2, .3, 5.]], D)
    mode = tf.constant([.2, -.1, .3, -.2], D)
    center, scale = tf.fill([4], tf.constant(.03, D)), tf.fill([4], tf.constant(1.1, D))
    calls = tf.Variable(0, dtype=tf.int64)

    def scalar(point):
        calls.assign_add(1)
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def batched(points):
        calls.assign_add(tf.shape(points, out_type=tf.int64)[0])
        delta = points - mode[None]
        scores = -tf.einsum('ij,bj->bi', precision, delta)
        return .5 * tf.reduce_sum(delta * scores, axis=1), scores

    return scalar, batched, calls, center, scale


def _assert_no_geometry(result):
    assert result.status == 'objective_resolution_limited'
    assert not result.accepted
    assert result.map_candidate is result.precision is result.covariance is None
    flags = result.diagnostics['objective_resolution']['attempt_flags']
    assert any(flags)
    diagnostics = result.diagnostics['objective_resolution']
    for index, flag in enumerate(flags):
        if flag:
            before = diagnostics['selected_value'][index]
            after = diagnostics['promoted_value'][index]
            assert before < after == math.nextafter(before, math.inf)


@pytest.mark.parametrize('batched', [False, True])
def test_completed_error_preserves_lifecycle_and_skips_mass(batched, monkeypatch, request):
    scalar, batch, calls, center, scale = _fixture()
    cfg = _sequential_config()
    block = BlockCoordinateCenterBlock('wide', 1, 3, cfg)
    mass_calls = tf.Variable(0, dtype=tf.int64)
    original_factory = native.precision_program

    def counted_factory(*args, **kwargs):
        mass = original_factory(*args, **kwargs)

        def counted(*inputs):
            mass_calls.assign_add(1)
            return mass(*inputs)

        return counted

    monkeypatch.setattr(native, 'precision_program', counted_factory)
    owner = ConditionalSequentialProgram(scalar, batch if batched else None, 4, block)
    raw = owner.compiled(center, scale)
    current_calls = int(calls)
    assert int(raw['status']) == native.OBJECTIVE_RESOLUTION_LIMITED
    assert int(mass_calls) == 0
    assert not bool(tf.reduce_any(raw['precision'] != 0.))
    assert not bool(tf.reduce_any(raw['covariance'] != 0.))
    assert not bool(tf.reduce_any(raw['mass_flags']))
    events = []
    result = sequential_result(raw, cfg, 1, 2, events.append)
    _assert_no_geometry(result)
    assert current_calls == result.diagnostics['exact_evaluations'] == int(raw['lifecycle']['evaluations'])
    assert events[-1]['status'] == 'objective_resolution_limited' and not events[-1]['accepted']

    checkpoint = FrozenCheckpoint('17b56ade2', f'resolution_lifecycle_{batched}')
    old_module = checkpoint.load('bayesfilter.inference.block_conditional_tf')
    calls.assign(0)
    previous = old_module.ConditionalSequentialProgram(scalar, batch if batched else None, 4, block)
    old_raw = previous.compiled(center, scale)
    assert int(calls) == current_calls
    _assert_record(_host(raw['lifecycle']), _host(old_raw['lifecycle']))
    old_report = checkpoint.load('bayesfilter.inference.sequential_controller_report')
    old_events = []
    old_result = old_report.sequential_result(old_raw, cfg, 1, 2, old_events.append)
    _assert_record(result.diagnostics['history'], old_result.diagnostics['history'])
    _assert_record(result.diagnostics['locator'], old_result.diagnostics['locator'])
    for field in ('terminal_fit_attempts', 'terminal_max_abs_scaled_score', 'search_seed', 'terminal_seed', 'terminal_fit'):
        _assert_record(result.diagnostics[field], old_result.diagnostics[field])
    _assert_record(events[:-1], old_events[:-1] if old_events[-1]['stage'] == 'initializer_completed' else old_events)

    def conditional(point):
        value, score = scalar(tf.concat([center[:1], point, center[3:]], 0))
        return value, score[1:3]

    def conditional_batch(points):
        count = tf.shape(points)[0]
        values, scores = batch(tf.concat([tf.repeat(center[None, :1], count, axis=0), points,
            tf.repeat(center[None, 3:], count, axis=0)], axis=1))
        return values, scores[:, 1:3]

    calls.assign(0)
    public_result = public.estimate_sequential_map_covariance(conditional, center[None, 1:3],
        batched_value_and_score_fn=conditional_batch if batched else None, scale=scale[1:3], config=cfg)
    _assert_no_geometry(public_result)
    assert int(calls) == current_calls and int(mass_calls) == 0
    _assert_record(public_result.diagnostics['history'], result.diagnostics['history'])
    # Positive control for the resource probe: a healthy execution really does
    # call the same mass preparation dependency, so zero above proves a skip.
    healthy = owner.compiled(tf.zeros([4], D), tf.ones([4], D))
    assert int(healthy['status']) == 0 and int(healthy['lifecycle']['status']) == 0
    assert int(mass_calls) == 1
    assert owner.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
    _write(request, f'objective-resolution-sequential-{batched}.json', {
        'public': public_result.payload(), 'conditional': result.payload(), 'previous': old_result.payload(),
        'current_calls': current_calls, 'rejected_mass_calls': 0, 'healthy_mass_calls': int(mass_calls), 'events': events,
        'unchanged_lifecycle': True, 'previous_source_sha256': checkpoint.hashes()})


@pytest.mark.parametrize('batched', [False, True])
def test_public_block_error_prevents_replay_and_later_blocks(batched, request):
    scalar, batch, calls, center, scale = _fixture()
    cfg = _sequential_config()
    blocks = (BlockCoordinateCenterBlock('wide', 1, 3, cfg), BlockCoordinateCenterBlock('later', 0, 1, cfg))
    events = []
    result = locate_block_coordinate_center(scalar, center, blocks=blocks,
        batched_value_and_score_fn=batch if batched else None, scale=scale,
        config=BlockCoordinateCenterConfig(max_physical_target_rows=300), progress_callback=events.append)
    assert result.status == 'invalid_sequential_handoff' and not result.completed
    assert result.accepted_block_count == 0 and result.completed_block_count == 1
    assert result.private_block_records[0]['handoff_status'] == 'objective_resolution_limited'
    assert result.final_center.numpy().tolist() == center.numpy().tolist()
    assert result.physical_target_rows == int(calls) == 1 + result.sequential_exact_evaluations
    assert [row['block_index'] for row in events if row['stage'] == 'block_started'] == [0]
    assert not any(row['stage'] == 'block_completed' for row in events)
    _write(request, f'objective-resolution-public-block-{batched}.json', {
        'result': result.private_payload(), 'events': events, 'calls': int(calls)})
