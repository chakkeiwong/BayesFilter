"""Diagnostic attribution of backend-dependent proposal-resolution events."""

import math
from dataclasses import asdict

import pytest
import tensorflow as tf

from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_coordinate_center import BlockCoordinateCenterBlock
from bayesfilter.inference.sequential_controller_report import sequential_result
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_block_coordinate_center import _sequential_config
from tests.test_filter_repair_block_center import _assert_record
from tests.test_filter_repair_objective_resolution import _fixture, _host, _write


def _comparison(actual, expected):
    try:
        _assert_record(actual, expected)
    except AssertionError as error:
        return {'matches': False, 'failure': str(error)}
    return {'matches': True}


@pytest.mark.parametrize('batched', [False, True])
def test_backend_resolution_attribution(batched, request):
    scalar, batch, calls, center, scale = _fixture()
    cfg = _sequential_config()
    block = BlockCoordinateCenterBlock('wide', 1, 3, cfg)
    owner = ConditionalSequentialProgram(scalar, batch if batched else None, 4, block)
    raw = owner.compiled(center, scale)
    current_calls = int(calls)
    result = sequential_result(raw, cfg, 1, 2)
    history = _host(raw['lifecycle']['history'])
    refined = history['refine']
    expected_flags = [
        event == 3 and action >= 2 and executed and promoted
        and math.isfinite(before) and math.isfinite(after)
        and before < after <= math.nextafter(before, math.inf)
        for event, action, executed, promoted, before, after in zip(
            history['event'], refined['action'], refined['attempts']['last_evaluated'],
            refined['attempts']['promoted_without_acceptance'],
            refined['selected_value'], refined['center_value'])
    ]

    preguard = FrozenCheckpoint('17b56ade2', f'resolution_backend_{batched}')
    old_conditional = preguard.load('bayesfilter.inference.block_conditional_tf')
    calls.assign(0)
    previous = old_conditional.ConditionalSequentialProgram(scalar, batch if batched else None, 4, block)
    old_raw = previous.compiled(center, scale)
    previous_calls = int(calls)

    def conditional(point):
        value, score = scalar(tf.concat([center[:1], point, center[3:]], 0))
        return value, score[1:3]

    def conditional_batch(points):
        count = tf.shape(points)[0]
        values, scores = batch(tf.concat([tf.repeat(center[None, :1], count, axis=0), points,
            tf.repeat(center[None, 3:], count, axis=0)], axis=1))
        return values, scores[:, 1:3]

    original = FrozenCheckpoint('3582b4ac', f'resolution_original_{batched}')
    old_public = original.load('bayesfilter.inference.sequential_map_covariance')
    calls.assign(0)
    reference = old_public.estimate_sequential_map_covariance(conditional, center[None, 1:3],
        batched_value_and_score_fn=conditional_batch if batched else None, scale=scale[1:3],
        config=old_public.SequentialMapCovarianceConfig(**asdict(cfg)))
    comparisons = {
        'preguard_lifecycle': _comparison(_host(raw['lifecycle']), _host(old_raw['lifecycle'])),
        'original_public': _comparison(result.payload(), reference.payload()),
    }
    observed_flags = _host(raw['objective_resolution']['attempt_flags'])
    _write(request, f'objective-resolution-backend-{batched}.json', {
        'role': 'diagnostic attribution; does not replace public qualification',
        'raw': _host(raw), 'preguard_raw': _host(old_raw),
        'public': result.payload(), 'original_public': reference.payload(),
        'current_calls': current_calls, 'preguard_calls': previous_calls,
        'original_calls': int(calls), 'independent_flags': expected_flags,
        'comparisons': comparisons, 'preguard_source': preguard.hashes(),
        'original_source': original.hashes(),
    })
    assert observed_flags == expected_flags
    assert int(raw['status']) == (4 if any(expected_flags) else 0)
    assert comparisons['preguard_lifecycle']['matches'], comparisons
    assert current_calls == previous_calls
