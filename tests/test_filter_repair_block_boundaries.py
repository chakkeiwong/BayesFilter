"""Diagnostic native-boundary injections for ordered transaction failure paths."""

import gc
import json
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import block_controller_tf as native
from bayesfilter.inference.block_conditional_tf import ConditionalSequentialProgram
from bayesfilter.inference.block_controller_report import block_result
from bayesfilter.inference.block_coordinate_center import (
    BlockCoordinateCenterBlock,
    BlockCoordinateCenterConfig,
)
from tests.test_block_coordinate_center import _sequential_config

D = tf.float64


@pytest.mark.parametrize('case', ['allowed_no_geometry', 'allowed_projection', 'decrease',
    'invalid_handoff', 'objective_resolution', 'overcount', 'negative_count', 'nonfinite_initial', 'nonfinite_replay', 'invalid_mass'])
def test_transaction_veto_stops_later_blocks(case, monkeypatch, request):
    full_calls = tf.Variable(0, dtype=tf.int64)
    block_calls = tf.Variable([0, 0], dtype=tf.int64)
    cfg = _sequential_config()
    blocks = tuple(BlockCoordinateCenterBlock(str(index), index, index + 1, cfg) for index in range(2))
    settings = BlockCoordinateCenterConfig(max_physical_target_rows=300)
    center, scale = tf.zeros([2], D), tf.ones([2], D)

    def target(point):
        full_calls.assign_add(1)
        shift = tf.constant(0. if case == 'decrease' else .5, D)
        delta = point - shift
        value = -.5 * tf.reduce_sum(delta ** 2)
        if case == 'nonfinite_initial':
            value = tf.constant(float('nan'), D)
        if case == 'nonfinite_replay':
            value = tf.where(point[0] == 0., value, tf.constant(float('nan'), D))
        return value, -delta

    # Obtain only the real endpoint's static tensor schema; no target call.
    schema_owner = ConditionalSequentialProgram(target, None, 2, blocks[0])
    schema = schema_owner.compiled.get_concrete_function().structured_outputs
    assert int(full_calls) == 0

    class Boundary:
        def __init__(self, scalar, batched, dimension, block, **kwargs):
            @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension], D)],
                         jit_compile=True, autograph=False)
            def execute(point, scaling):
                block_calls.assign_add(tf.one_hot(block.start, 2, dtype=tf.int64))
                result = tf.nest.map_structure(lambda row: tf.zeros(row.shape, row.dtype), schema)
                evaluations = -1 if case == 'negative_count' else 129 if case == 'overcount' else 3
                status = 3 if case == 'allowed_projection' else 0 if case == 'invalid_mass' else 4
                result['status'] = tf.constant(4 if case == 'objective_resolution' else 2 if case == 'invalid_handoff' else 0)
                result['locator_evaluations'] = tf.constant(1 if case == 'objective_resolution' else evaluations, tf.int64)
                result['locator']['selected']['finite_count'] = tf.constant(1)
                result['lifecycle'].update(status=tf.constant(status), evaluations=tf.constant(evaluations, tf.int64),
                    center=tf.constant([2. if case == 'decrease' else .5], D))
                result['mass_flags'] = tf.constant([False, True, True, False])
                return result

            self.compiled = execute

    monkeypatch.setattr(native, 'ConditionalSequentialProgram', Boundary)
    owner = native.BlockController(target, None, 2, blocks, settings)
    assert int(full_calls) == 0 and block_calls.numpy().tolist() == [0, 0]
    raw = owner.compiled(center, scale)
    observed = {'case': case, 'state': tf.nest.map_structure(lambda row: row.numpy().tolist(), raw['state']),
        'full_calls': int(full_calls), 'block_calls': block_calls.numpy().tolist()}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / f'block-boundary-{case}.json').open('x') as out:
        json.dump(observed, out, indent=2)
        out.write('\n')
    if case in ('negative_count', 'nonfinite_initial', 'nonfinite_replay', 'invalid_mass'):
        message = 'nonnegative' if case == 'negative_count' else 'precision must be finite' if case == 'invalid_mass' else 'full target replay must be finite'
        with pytest.raises(ValueError, match=message):
            block_result(raw, center, blocks, settings)
    else:
        result = block_result(raw, center, blocks, settings)
        expected = {'allowed_no_geometry': 'sweep_completed_with_resolvable_progress',
            'allowed_projection': 'sweep_completed_with_resolvable_progress',
            'decrease': 'transaction_objective_decrease', 'invalid_handoff': 'invalid_sequential_handoff',
            'objective_resolution': 'invalid_sequential_handoff',
            'overcount': 'sequential_row_accounting_invalid'}[case]
        assert result.status == expected
        if case.startswith('allowed'):
            assert result.accepted_block_count == 2 and result.physical_target_rows == 9
            assert result.sequential_exact_evaluations == 6
            assert result.final_center.numpy().tolist() == [.5, .5]
        else:
            assert result.accepted_block_count == 0
            assert result.final_center.numpy().tolist() == [0., 0.]
            assert result.physical_target_rows == {'decrease': 5, 'invalid_handoff': 4, 'objective_resolution': 4, 'overcount': 1}[case]
            if case == 'objective_resolution':
                assert result.private_block_records[0]['handoff_status'] == 'objective_resolution_limited'
                assert result.sequential_exact_evaluations == 3
    assert observed['block_calls'] == ([1, 1] if case.startswith('allowed') else [0, 0] if case == 'nonfinite_initial' else [1, 0])
    assert observed['full_calls'] == (3 if case.startswith('allowed') else 2 if case in ('decrease', 'nonfinite_replay') else 1)


def test_outer_owner_release_preserves_retained_compiled_handle(request):
    def targets(multiplier):
        def scalar(point):
            return -.5 * multiplier * tf.reduce_sum(point ** 2), -multiplier * point
        return scalar

    cfg = _sequential_config()
    blocks = tuple(BlockCoordinateCenterBlock(str(index), index, index + 1, cfg) for index in range(2))
    settings = BlockCoordinateCenterConfig(max_physical_target_rows=300)
    point, scale = tf.zeros([2], D), tf.ones([2], D)
    scalar = targets(1.)
    owner = native.BlockController(scalar, None, 2, blocks, settings)
    first = owner.compiled(point, scale)
    refs = {'target': weakref.ref(scalar), 'owner': weakref.ref(owner),
        'graph': weakref.ref(owner.compiled.get_concrete_function().graph), 'scope': weakref.ref(owner.dependency_scope)}
    held = owner.compiled
    del owner, scalar
    gc.collect()
    assert refs['target']() is not None
    successor = native.BlockController(targets(2.), None, 2, blocks, settings)
    changed = successor.compiled(point, scale)
    repeated = held(point, scale)
    tf.debugging.assert_near(first['sequential'][0]['precision'], repeated['sequential'][0]['precision'], atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(changed['sequential'][0]['precision'], 2. * first['sequential'][0]['precision'], atol=1e-10, rtol=1e-10)
    del held
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    root = Path(request.config.getoption('xmlpath')).parent
    with (root / 'block-outer-ownership.json').open('x') as out:
        json.dump({'released': released, 'nonclaim': 'No native executable eviction claim'}, out, indent=2)
        out.write('\n')
    assert all(released.values()), released
