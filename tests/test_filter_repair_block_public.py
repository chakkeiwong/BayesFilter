"""Actual public ordered-block original records, target order and compiled reuse."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import block_controller_tf as native
from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
    locate_block_coordinate_center,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import (
    _inject_conditional_result,
    _result,
    _sequential_config,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_block_center import _assert_record

D = tf.float64


def _diagnostic_json(value):
    if isinstance(value, dict):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_diagnostic_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': repr(value)}
    return value


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['coupled', 'record_reversal', 'partial_heterogeneous'])
def test_complete_public_block_matches_original(case, batched, request):
    dimension = 4 if case == 'partial_heterogeneous' else 2
    precision = tf.constant([[4., 3.], [3., 4.]] if dimension == 2 else
        [[4., .2, .1, 1.], [.2, 5., .5, .2], [.1, .5, 6., .3], [1., .2, .3, 5.]], D)
    mode = tf.constant([1., -1.] if dimension == 2 else [.2, -.1, .3, -.2], D)
    calls = tf.Variable(0, dtype=tf.int64)
    trace = tf.Variable(tf.zeros([300, dimension], D))

    def record(points):
        first = calls.read_value()
        count = tf.shape(points, out_type=tf.int64)[0]
        trace.scatter_nd_update((tf.range(count) + first)[:, None], points)
        calls.assign_add(count)

    def scalar(point):
        record(point[None])
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def batch(points):
        record(points)
        delta = points - mode[None]
        scores = -tf.einsum('ij,bj->bi', precision, delta)
        return .5 * tf.reduce_sum(delta * scores, axis=1), scores

    cfg = _sequential_config()
    blocks = (BlockCoordinateCenterBlock('first', 0, 1, cfg),
        BlockCoordinateCenterBlock('second', 1, 2, cfg)) if dimension == 2 else (
        BlockCoordinateCenterBlock('wide', 1, 3, cfg),
        BlockCoordinateCenterBlock('first', 0, 1, replace(cfg, terminal_sample_count=20, seed=(2026, 718))))
    sweep_cfg = BlockCoordinateCenterConfig(max_physical_target_rows=300,
        stop_on_material_reversal=case != 'record_reversal')
    native.clear_block_controller_cache()
    assert int(calls) == 0
    checkpoint = FrozenCheckpoint('3582b4ac', f'public_ordered_{case}_{batched}')
    original = checkpoint.load('bayesfilter.inference.block_coordinate_center')
    old_sequential = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    original_blocks = tuple(original.BlockCoordinateCenterBlock(block.name, block.start, block.stop,
        old_sequential.SequentialMapCovarianceConfig(**asdict(block.sequential_config))) for block in blocks)
    original_cfg = original.BlockCoordinateCenterConfig(**asdict(sweep_cfg))
    rows, hlos, owners = [], [], []
    for offset, multiplier in ((0., 1.), (.03, 1.1)):
        center = tf.fill([dimension], tf.constant(offset, D))
        scale = tf.fill([dimension], tf.constant(multiplier, D))
        calls.assign(0)
        events, observed_counts = [], []

        def progress(event, events=events, observed_counts=observed_counts):
            events.append(event)
            observed_counts.append(int(calls))

        actual = locate_block_coordinate_center(scalar, center, blocks=blocks,
            batched_value_and_score_fn=batch if batched else None, scale=scale,
            config=sweep_cfg, progress_callback=progress)
        owner = native._LAST_CONTROLLER[3]
        owners.append(owner)
        actual_calls = int(calls)
        actual_trace = trace[:actual_calls].numpy().tolist()
        assert observed_counts and all(count == actual_calls for count in observed_counts)
        calls.assign(0)
        expected_events = []
        expected = original.locate_block_coordinate_center(scalar, center, blocks=original_blocks,
            batched_value_and_score_fn=batch if batched else None, scale=scale, config=original_cfg,
            progress_callback=expected_events.append)
        rows.append({'actual': actual.private_payload(), 'expected': expected.private_payload(),
            'events': events, 'expected_events': expected_events,
            'resolution_flags': [actual.private_block_records[0]['handoff_status'] == 'objective_resolution_limited'],
            'executed': [any(event['stage'] == 'block_started' and event['block_index'] == index for event in events) for index in range(len(blocks))],
            'actual_trace': actual_trace, 'expected_trace': trace[:int(calls)].numpy().tolist(),
            'observed_counts': observed_counts,
            'actual_calls': actual_calls, 'expected_calls': int(calls)})
        hlos.append(owner.compiled.experimental_get_compiler_ir(center, scale)(stage='hlo'))
    report = {'scope': 'actual public complete ordered-block endpoint', 'case': case, 'batched': batched,
        'records': rows, 'original_source_sha256': checkpoint.hashes(),
        'trace_count': owner.compiled.experimental_get_tracing_count(),
        'hlo_unchanged': stable_hlo(hlos[0]) == stable_hlo(hlos[1])}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / f'block-public-{case}-{batched}.json').open('x') as output:
        json.dump(_diagnostic_json(report), output, indent=2, allow_nan=False)
        output.write('\n')
    (root / f'block-public-{case}-{batched}.hlo.txt').write_text(hlos[0])
    (root / f'block-public-{case}-{batched}-changed.hlo.txt').write_text(hlos[1])
    for row in rows:
        if any(row['resolution_flags']):
            # Only the owner's exact completed-boundary error can replace a
            # flagged comparison; the failed03028 records remain archived.
            assert case == 'partial_heterogeneous'
            summary = row['actual']['public_summary']
            assert not summary['completed'] and summary['status'] == 'invalid_sequential_handoff'
            assert summary['accepted_block_count'] == 0
            assert row['executed'] == [True, False]
            assert row['actual_calls'] == summary['physical_target_rows'] == 1 + summary['sequential_exact_evaluations']
            assert len(row['actual']['block_records']) == 1
            record = row['actual']['block_records'][0]
            assert record['handoff_status'] == 'objective_resolution_limited' and not record['committed']
            assert record['center_after'] == record['center_before']
            assert not any(event['stage'] == 'block_completed' for event in row['events'])
            continue
        _assert_record(row['actual'], row['expected'])
        _assert_record(row['events'], row['expected_events'])
        _assert_record(row['actual_trace'], row['expected_trace'])
        assert row['actual_calls'] == row['expected_calls'] == row['actual']['public_summary']['physical_target_rows']
    assert owners[0] is owners[1]
    assert report['trace_count'] == 1 and report['hlo_unchanged']
    graph = owner.compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert operations & {'While', 'StatelessWhile'}
    assert operations & {'Case', 'StatelessCase'}
    assert owner.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b


def test_public_block_validation_precedes_target_use():
    calls = []

    def forbidden(point):
        calls.append(point)
        raise AssertionError('Invalid configuration reached target')

    block = BlockCoordinateCenterBlock('one', 0, 1, _sequential_config())
    cases = (([], None, (block,), 300), ([0., 0.], [1.], (block,), 300),
        ([0.], [0.], (block,), 300), ([0.], [float('nan')], (block,), 300),
        ([0.], None, (), 300), ([0.], None, (block, block), 300),
        ([0.], None, (block,), 129))
    for center, scale, blocks, cap in cases:
        with pytest.raises(ValueError):
            locate_block_coordinate_center(forbidden, center, blocks=blocks, scale=scale,
                config=BlockCoordinateCenterConfig(max_physical_target_rows=cap))
    assert calls == []


def test_public_block_unsupported_callback_has_no_eager_retry():
    calls, events = [], []

    def host_only(point):
        calls.append('traced')
        point.numpy()
        return tf.constant(0., D), tf.zeros([1], D)

    block = BlockCoordinateCenterBlock('one', 0, 1, _sequential_config())
    with pytest.raises((AttributeError, TypeError, ValueError, NotImplementedError)):
        locate_block_coordinate_center(host_only, [0.], blocks=(block,),
            progress_callback=events.append)
    assert calls == ['traced'] and events == []


def test_public_block_keeps_frozen_derivative_boundary():
    def target(point):
        delta = point - tf.constant(.25, D)
        return -.5 * tf.reduce_sum(delta ** 2), -delta

    point, scale = tf.constant([0.], D), tf.constant([1.], D)
    block = BlockCoordinateCenterBlock('one', 0, 1, _sequential_config())
    with tf.GradientTape() as tape:
        tape.watch((point, scale))
        result = locate_block_coordinate_center(target, point, blocks=(block,), scale=scale)
        assert result.completed and result.accepted_block_count == 1
        total = tf.reduce_sum(result.final_center) + tf.reduce_sum(result.final_score)
    assert tape.gradient(total, (point, scale)) == (None, None)


@pytest.mark.parametrize('case', ['nonfinite_initial', 'nonfinite_replay', 'overcount', 'negative_count'])
def test_public_block_propagates_transaction_failures(case, monkeypatch):
    calls = tf.Variable(0, dtype=tf.int64)

    def target(point):
        calls.assign_add(1)
        delta = point - tf.constant(.5, D)
        value = -.5 * tf.reduce_sum(delta ** 2)
        if case == 'nonfinite_initial':
            value = tf.constant(float('nan'), D)
        elif case == 'nonfinite_replay':
            value = tf.where(point[0] == 0., value, tf.constant(float('nan'), D))
        return value, -delta

    evaluations = 129 if case == 'overcount' else -1 if case == 'negative_count' else 3
    _inject_conditional_result(monkeypatch,
        lambda: _result(tf.constant([.5], D), evaluations=evaluations))
    blocks = tuple(BlockCoordinateCenterBlock(str(index), index, index + 1, _sequential_config())
        for index in range(2))
    events = []
    if case == 'overcount':
        result = locate_block_coordinate_center(target, [0., 0.], blocks=blocks,
            progress_callback=events.append)
        assert result.status == 'sequential_row_accounting_invalid'
        assert result.physical_target_rows == 1 and result.sequential_exact_evaluations == 0
        assert not result.private_block_records and not result.completed
        assert result.final_center.numpy().tolist() == [0., 0.]
    else:
        message = 'nonnegative' if case == 'negative_count' else 'full target replay must be finite'
        with pytest.raises(ValueError, match=message):
            locate_block_coordinate_center(target, [0., 0.], blocks=blocks,
                progress_callback=events.append)
    assert int(calls) == (2 if case == 'nonfinite_replay' else 1)
    assert not any(event['stage'] == 'block_started' and event['block_index'] == 1 for event in events)
