"""Full original-source reference checks for internal ordered-block execution."""

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.block_controller_report import block_result
from bayesfilter.inference.block_controller_tf import BlockController
from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import _sequential_config
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
def test_complete_ordered_block_matches_original(case, batched, request):
    dimension = 4 if case == 'partial_heterogeneous' else 2
    precision = tf.constant([[4., 3.], [3., 4.]] if dimension == 2 else
        [[4., .2, .1, 1.], [.2, 5., .5, .2], [.1, .5, 6., .3], [1., .2, .3, 5.]], D)
    mode = tf.constant([1., -1.] if dimension == 2 else [.2, -.1, .3, -.2], D)
    calls = tf.Variable(0, dtype=tf.int64)

    def scalar(point):
        calls.assign_add(1)
        delta = point - mode
        score = -tf.linalg.matvec(precision, delta)
        return .5 * tf.reduce_sum(delta * score), score

    def batch(points):
        calls.assign_add(tf.shape(points, out_type=tf.int64)[0])
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
    owner = BlockController(scalar, batch if batched else None, dimension, blocks, sweep_cfg, progress=True)
    assert int(calls) == 0
    checkpoint = FrozenCheckpoint('3582b4ac', f'ordered_{case}_{batched}')
    original = checkpoint.load('bayesfilter.inference.block_coordinate_center')
    old_sequential = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    original_blocks = tuple(original.BlockCoordinateCenterBlock(block.name, block.start, block.stop,
        old_sequential.SequentialMapCovarianceConfig(**asdict(block.sequential_config))) for block in blocks)
    original_cfg = original.BlockCoordinateCenterConfig(**asdict(sweep_cfg))
    rows, hlos = [], []
    for offset, multiplier in ((0., 1.), (.03, 1.1)):
        center = tf.fill([dimension], tf.constant(offset, D))
        scale = tf.fill([dimension], tf.constant(multiplier, D))
        calls.assign(0)
        events = []
        raw = owner.compiled(center, scale)
        actual = block_result(raw, center, blocks, sweep_cfg, events.append)
        actual_calls = int(calls)
        calls.assign(0)
        expected_events = []
        expected = original.locate_block_coordinate_center(scalar, center, blocks=original_blocks,
            batched_value_and_score_fn=batch if batched else None, scale=scale, config=original_cfg,
            progress_callback=expected_events.append)
        rows.append({'actual': actual.private_payload(), 'expected': expected.private_payload(),
            'events': events, 'expected_events': expected_events,
            'actual_calls': actual_calls, 'expected_calls': int(calls)})
        hlos.append(owner.compiled.experimental_get_compiler_ir(center, scale)(stage='hlo'))
    report = {'scope': 'internal complete ordered-block endpoint', 'case': case, 'batched': batched,
        'records': rows, 'original_source_sha256': checkpoint.hashes(),
        'trace_count': owner.compiled.experimental_get_tracing_count(),
        'hlo_unchanged': stable_hlo(hlos[0]) == stable_hlo(hlos[1])}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / f'block-controller-{case}-{batched}.json').open('x') as output:
        json.dump(_diagnostic_json(report), output, indent=2, allow_nan=False)
        output.write('\n')
    (root / f'block-controller-{case}-{batched}.hlo.txt').write_text(hlos[0])
    (root / f'block-controller-{case}-{batched}-changed.hlo.txt').write_text(hlos[1])
    for row in rows:
        _assert_record(row['actual'], row['expected'])
        _assert_record(row['events'], row['expected_events'])
        assert row['actual_calls'] == row['expected_calls'] == row['actual']['public_summary']['physical_target_rows']
    assert report['trace_count'] == 1 and report['hlo_unchanged']
    graph = owner.compiled.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(fn.node_def for fn in graph.library.function)) for node in nodes}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert operations & {'While', 'StatelessWhile'}
    assert operations & {'Case', 'StatelessCase'}
    assert owner.compiled.get_concrete_function().function_def.attr['_XlaMustCompile'].b
