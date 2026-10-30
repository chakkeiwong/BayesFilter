"""Actual public block wiring for alternate qualified sequential policies."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
    locate_block_coordinate_center,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_center import _assert_record
from tests.test_filter_repair_block_public import _diagnostic_json
from tests.test_filter_repair_lifecycle_original import _compare_original
from tests.test_filter_repair_sequential_controller import fixture

D = tf.float64
CASES = ('scalar_locator', 'factor_one', 'factor_two_reuse', 'paired', 'scaled_search', 'score_disabled')


@pytest.mark.parametrize('case', CASES)
def test_public_block_configured_dependencies(case, request):
    raw_scalar, raw_batch, _, cfg, starts, scale = fixture(case)
    dimension = int(starts.shape[1])
    calls = tf.Variable(0, dtype=tf.int64)
    trace = tf.Variable(tf.zeros([1024, dimension], D))

    def record(points):
        count = tf.shape(points, out_type=tf.int64)[0]
        indices = (tf.range(count) + calls.read_value())[:, None]
        trace.scatter_nd_update(indices, points)
        calls.assign_add(count)

    def scalar(point):
        record(point[None])
        return raw_scalar(point)

    def batched(points):
        record(points)
        return raw_batch(points)

    center = starts[0]
    settings = BlockCoordinateCenterConfig(max_physical_target_rows=cfg.max_exact_evaluations + 2)
    blocks = (BlockCoordinateCenterBlock('all', 0, dimension, cfg),)
    events = []
    actual = locate_block_coordinate_center(scalar, center, blocks=blocks, scale=scale,
        batched_value_and_score_fn=batched, config=settings, progress_callback=events.append)
    actual_calls, actual_trace = int(calls), trace[:int(calls)].numpy().tolist()
    checkpoint = FrozenCheckpoint('3582b4ac', f'block_public_options_{case}')
    original = checkpoint.load('bayesfilter.inference.block_coordinate_center')
    sequential = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    old_blocks = (original.BlockCoordinateCenterBlock('all', 0, dimension,
        sequential.SequentialMapCovarianceConfig(**asdict(cfg))),)
    calls.assign(0)
    expected_events = []
    expected = original.locate_block_coordinate_center(scalar, center, blocks=old_blocks, scale=scale,
        batched_value_and_score_fn=batched, config=original.BlockCoordinateCenterConfig(**asdict(settings)),
        progress_callback=expected_events.append)
    expected_calls, expected_trace = int(calls), trace[:int(calls)].numpy().tolist()
    report = {'case': case, 'actual': actual.private_payload(), 'expected': expected.private_payload(),
        'events': events, 'expected_events': expected_events, 'actual_calls': actual_calls,
        'expected_calls': expected_calls, 'actual_trace': actual_trace, 'expected_trace': expected_trace,
        'original_source_sha256': checkpoint.hashes()}
    path = Path(request.config.getoption('xmlpath')).parent / f'block-public-options-{case}.json'
    with path.open('x') as out:
        json.dump(_diagnostic_json(report), out, indent=2, allow_nan=False)
        out.write('\n')
    _compare_original(report['actual'], report['expected'])
    _assert_record(events, expected_events)
    _assert_record(actual_trace, expected_trace)
    assert actual_calls == expected_calls == actual.physical_target_rows
