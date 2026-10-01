"""Diagnostic original records and admission checks for scalar geometry targets."""

import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry_control_tf as runtime
from bayesfilter.inference.quadratic_geometry_control_report import (
    center_refinement_report,
    exact_replay_report,
)
from tests.test_filter_repair_geometry_control import (
    clean,
    fixture,
    reference,
    save,
    source,
)
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def callback(case):
    def target(point):
        value, score = -.5 * tf.reduce_sum(point * point), -point
        if case == 'raises':
            raise ValueError('deliberate callback failure')
        if case == 'arity':
            return value, score, value
        if case == 'value_shape':
            return tf.stack((value, value)), score
        if case == 'value_dtype':
            return tf.cast(value, tf.float32), score
        if case == 'score_dtype':
            return value, tf.cast(score, tf.float32)
        if case == 'score_column':
            return value, score[:, None]
        if case == 'score_short':
            return value, score[:1]
        if case == 'score_empty':
            return value, score[:0]
        return value, score
    return target


def original_replay(target, args):
    checkpoint, module = source('3582b4ac')
    candidate = checkpoint.load('bayesfilter.inference._exact_incumbent').ExactCandidate(
        args[0].numpy(), float(args[4]), args[3].numpy(), 2, 'design')
    return clean(module._canonical_replay(target, candidate, evaluation_index=7))


@pytest.mark.parametrize('case', ['raises', 'arity', 'value_shape', 'value_dtype', 'score_dtype', 'score_column'])
@pytest.mark.parametrize('role', ['proposal', 'replay'])
def test_callback_failure_records_match_original(case, role, request):
    target = callback(case)
    _, cfg, args, _, _ = fixture(3, 'unconstrained', record_calls=False)
    expected = (clean(reference(source('3582b4ac')[1], target, cfg, args)) if role == 'proposal'
                else original_replay(target, args))
    records = {'original': expected}
    for jit in (False, True):
        try:
            if role == 'proposal':
                raw = runtime.make_center_refinement_program(target, 3, cfg, jit_compile=jit)(*args)
                actual = center_refinement_report(raw, cfg, args[4], args[5])
            else:
                raw = runtime.make_exact_replay_program(target, 3, jit_compile=jit)(args[0], args[4], args[3], True)
                actual = exact_replay_report(raw, evaluation_index=7, source='design')
            records[str(jit)] = clean(actual)
        except Exception as error:  # noqa: BLE001 - archive failure before the exact-record assertion.
            records[str(jit)] = {'error': type(error).__name__, 'message': str(error)}
    save(request, f'geometry-callback-{role}-{case}.json', records)
    for jit in (False, True):
        _equal_records(records[str(jit)], expected)


@pytest.mark.parametrize('case', ['score_short', 'score_empty'])
def test_malformed_score_is_explicitly_outside_original_parity(case, request):
    target = callback(case)
    _, cfg, args, _, _ = fixture(3, 'unconstrained', record_calls=False)
    try:
        original = clean(reference(source('3582b4ac')[1], target, cfg, args))
    except Exception as error:  # noqa: BLE001 - classify original malformed-output behavior.
        original = {'error': type(error).__name__, 'message': str(error)}
    records = {}
    for jit in (False, True):
        raw = runtime.make_center_refinement_program(target, 3, cfg, jit_compile=jit)(*args)
        records[str(jit)] = clean(center_refinement_report(raw, cfg, args[4], args[5]))
        assert int(raw['status']) == 6
        assert not bool(raw['accepted'])
    save(request, f'geometry-callback-malformed-{case}.json', {
        'role': 'explicit_shape_guard_not_original_parity', 'original': original, 'results': records})


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('operation', ['Assert', 'CheckNumerics', 'EagerPyFunc'])
def test_callback_runtime_veto_cannot_be_discarded_by_xla(operation, nested, request):
    calls = tf.Variable(0, dtype=tf.int64)

    def checked(point):
        calls.assign_add(1)
        if operation == 'Assert':
            with tf.control_dependencies([tf.debugging.assert_positive(point)]):
                score = -tf.identity(point)
        elif operation == 'CheckNumerics':
            score = tf.debugging.check_numerics(-point, 'score must be finite')
        else:
            score = tf.py_function(lambda value: -value, [point], D)
            score.set_shape([3])
        return -.5 * tf.reduce_sum(point * point), score

    target = (tf.function(checked, input_signature=[tf.TensorSpec([3], D)], autograph=False)
              if nested else checked)
    _, cfg, _, _, _ = fixture(3, 'unconstrained', record_calls=False)
    with pytest.raises(ValueError, match='unsupported callback operations:.*' + operation):
        runtime.make_center_refinement_program(target, 3, cfg)
    with pytest.raises(ValueError, match='unsupported callback operations:.*' + operation):
        runtime.make_exact_replay_program(target, 3)
    assert int(calls) == 0  # Tracing is not numerical execution.
    save(request, f'geometry-callback-guard-{operation}-{nested}.json',
         {'operation': operation, 'nested': nested, 'executed_calls': int(calls),
          'decision': 'reject_native_construction_without_eager_retry'})


def test_target_resources_execute_only_on_native_attempts():
    position = tf.Variable(tf.zeros([3], D))
    with tf.device(position.device):
        calls = tf.Variable(0, dtype=tf.int64)

    def target(point):
        calls.assign_add(1)
        position.assign(point)
        return -.5 * tf.reduce_sum(point * point), -point

    _, cfg, args, _, _ = fixture(3, 'interior', record_calls=False)
    proposal = runtime.make_center_refinement_program(target, 3, cfg)
    replay = runtime.make_exact_replay_program(target, 3)
    assert int(calls) == 0
    invalid = (*args[:2], -args[2], *args[3:])
    assert not bool(proposal(*invalid)['target_called'])
    assert not bool(replay(args[0], args[4], args[3], False)['attempted'])
    assert int(calls) == 0
    result = proposal(*args)
    assert int(calls) == 1
    tf.debugging.assert_equal(position, result['refined_center'])
    replay(args[0], args[4], args[3], True)
    assert int(calls) == 2
    tf.debugging.assert_equal(position, args[0])
