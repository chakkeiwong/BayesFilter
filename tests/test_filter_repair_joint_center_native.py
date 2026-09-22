"""Complete pinned locator records, callback sequence and native state reset."""

import ast
import dataclasses
import hashlib

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.joint_center_tf import (
    joint_center_result,
    make_joint_center_program,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def original_source(label):
    checkpoint = FrozenCheckpoint('3582b4ac', label)
    module = checkpoint.load('bayesfilter.inference.joint_center')
    compatibility = None
    if tf.config.list_logical_devices('GPU'):
        path = 'bayesfilter/inference/joint_center.py'
        source = checkpoint.sources[path]
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'locate_joint_center')
        text = '\n'.join(source.splitlines()[function.lineno - 1:function.end_lineno])
        # Three bounded resource counters and the matching cap constant. TFP's
        # own optimizer, target arithmetic, predicates and replay are unchanged.
        assert text.count('tf.int32') == 4
        compatible = text.replace('tf.int32', 'tf.int64')
        exec(compile(compatible, '3582b4ac:int64_accounting_only', 'exec'), module.__dict__)  # noqa: S102
        compatibility = {'classification': 'original_with_int64_accounting_only',
            'changed_declarations': 4, 'maximum_fixture_rows': 120,
            'function_sha256': hashlib.sha256(compatible.encode()).hexdigest()}
    return checkpoint, module, compatibility


def fixture(dimension, case):
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = .14 + tf.cast(tf.range(dimension), D) * .03
    calls = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([256, dimension], D))

    def target(point):
        index = calls.assign_add(1) - 1
        update = positions.scatter_nd_update(index[None, None], point[None])
        with tf.control_dependencies([update]):
            delta = point - mode
            score = -tf.linalg.matvec(precision, delta)
            value = .5 * tf.reduce_sum(delta * score)
        if case in ('quartic', 'iterations'):
            value -= .03 * tf.reduce_sum(delta ** 4)
            score -= .12 * delta ** 3
        if case == 'constant':
            return tf.constant(1., D), tf.zeros_like(point)
        if case == 'nonfinite':
            value = tf.constant(float('nan'), D)
        return value, score

    def reset():
        calls.assign(0)
        positions.assign(tf.zeros_like(positions))

    def record():
        return {'calls': int(calls), 'positions': positions[:int(calls)].numpy().tolist()}

    config = JointCenterLocatorConfig(max_iterations=1 if case == 'iterations' else 20,
        max_objective_evaluations=1 if case == 'cap' else 120, gradient_tolerance=1e-8)
    return target, config, reset, record


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('case', ['quadratic', 'quartic', 'constant', 'nonfinite', 'cap', 'iterations'])
def test_native_joint_locator_original_records(dimension, case, request):
    checkpoint, original, compatibility = original_source('joint_locator_native_original')
    callback, config, reset, record = fixture(dimension, case)
    program = make_joint_center_program(callback, dimension, config)
    comparisons = []
    for shift in (0., .11):
        start = .6 + shift - tf.cast(tf.range(dimension), D) * .14
        scale = .8 + shift + tf.cast(tf.range(dimension), D) * .09
        reset()
        expected = original.locate_joint_center(callback, start, scale=scale,
            config=original.JointCenterLocatorConfig(**dataclasses.asdict(config)))
        expected_calls = record()
        reset()
        with program.invocation_lock:
            raw = program(start, scale)
            actual = joint_center_result(raw, construction_error=program.construction_error)
        actual_calls = record()
        comparisons.append({'actual': clean(dataclasses.asdict(actual)),
            'expected': clean(dataclasses.asdict(expected)), 'actual_calls': actual_calls,
            'expected_calls': expected_calls})
    save(request, f'joint-native-{case}-{dimension}.json',
         {'comparisons': comparisons, 'reference_sources': checkpoint.hashes(),
          'reference_gpu_compatibility': compatibility})
    for comparison in comparisons:
        _equal_records(comparison['actual'], comparison['expected'])
        _equal_records(comparison['actual_calls'], comparison['expected_calls'])
        assert comparison['actual']['physical_target_rows'] == comparison['actual_calls']['calls']
    assert program.experimental_get_tracing_count() == 1


def test_native_joint_locator_hlo_and_independent_owners(request):
    callback, config, reset, record = fixture(3, 'quartic')
    first = make_joint_center_program(callback, 3, config)
    second = make_joint_center_program(callback, 3, dataclasses.replace(config, max_objective_evaluations=1))
    point, scale = tf.constant([.6, -.4, .3], D), tf.constant([.8, 1.2, .7], D)
    reset()
    expected = clean(first(point, scale))
    expected_calls = record()
    assert bool(second(point, scale)['cap_exhausted'])
    reset()
    actual = clean(first(point, scale))
    _equal_records(actual, expected)
    _equal_records(record(), expected_calls)
    hlo = first.experimental_get_compiler_ir(point, scale)(stage='hlo')
    changed_hlo = first.experimental_get_compiler_ir(point + .1, scale * .9)(stage='hlo')
    assert hlo == changed_hlo
    assert 'HloModule' in hlo and 'while(' in hlo
    graph = first.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(f.node_def for f in graph.library.function)) for node in nodes}
    assert not operations & {'PyFunc', 'EagerPyFunc', 'PyFuncStateless'}
    assert first.experimental_get_tracing_count() == 1
    save(request, 'joint-native-hlo.json', {'hlo': hlo, 'reference': expected, 'actual': actual})


def test_native_joint_locator_construction_exception(monkeypatch, request):
    from bayesfilter.inference import joint_center_tf

    checkpoint, original, compatibility = original_source('joint_locator_exception_original')

    def fail(*args, **kwargs):
        raise RuntimeError('synthetic optimizer construction failure')

    monkeypatch.setattr(joint_center_tf.tfp.optimizer, 'lbfgs_minimize', fail)
    callback, config, reset, record = fixture(1, 'quadratic')
    point, scale = tf.constant([.6], D), tf.constant([.8], D)
    expected = original.locate_joint_center(callback, point, scale=scale,
        config=original.JointCenterLocatorConfig(**dataclasses.asdict(config)))
    reset()
    program = make_joint_center_program(callback, 1, config)
    raw = program(point, scale)
    actual = joint_center_result(raw, construction_error=program.construction_error)
    save(request, 'joint-native-exception.json', {'actual': clean(dataclasses.asdict(actual)),
        'expected': clean(dataclasses.asdict(expected)), 'reference_sources': checkpoint.hashes(),
        'reference_gpu_compatibility': compatibility})
    _equal_records(clean(dataclasses.asdict(actual)), clean(dataclasses.asdict(expected)))
    assert record()['calls'] == 1


def test_native_joint_locator_rejects_wall_guard_and_host_callback():
    callback = lambda x: (tf.reduce_sum(x), tf.ones_like(x))
    with pytest.raises(ValueError, match='parent wall deadline'):
        make_joint_center_program(callback, 1, JointCenterLocatorConfig(jit_compile=False, max_wall_seconds=1.))
    def host_callback(x):
        value = tf.py_function(lambda z: np.sum(z), [x], D)
        return value, tf.ones_like(x)
    with pytest.raises(ValueError, match='unsupported callback'):
        make_joint_center_program(host_callback, 1, JointCenterLocatorConfig())


def test_native_affine_rounding_matches_separate_binary64_operations(request):
    from bayesfilter.inference.joint_center_tf import rounded_affine_position

    rng = np.random.default_rng(9032)
    initial, scale, z = rng.normal(size=(3, 2000))
    initial[:4], scale[:4], z[:4] = [.6, .71, -0., 0.], [.8, .91, 1., 1.], [-.575, -.6263736263736265, -0., -0.]
    expected = initial + scale * z
    program = tf.function(rounded_affine_position,
        input_signature=[tf.TensorSpec([2000], D)] * 3, jit_compile=True, autograph=False)
    actual = program(tf.constant(initial), tf.constant(scale), tf.constant(z)).numpy()
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
    save(request, 'joint-affine-rounding.json', {'count': 2000, 'bitwise_equal': True,
        'seed': 9032, 'signed_zero_included': True})


@pytest.mark.parametrize('case', ['interior', 'failed', 'accounting', 'cap',
                                  'endpoint_invalid', 'sentinel', 'invalid_position'])
def test_native_joint_locator_synthetic_failures(case, monkeypatch, request):
    """Independent controlled optimizer outcomes retain exact status priority."""
    from tensorflow_probability.python.optimizer.lbfgs import LBfgsOptimizerResults

    from bayesfilter.inference import joint_center_tf

    checkpoint, original, compatibility = original_source('joint_locator_failure_original')
    calls = tf.Variable(0, dtype=tf.int64)
    points = tf.Variable(tf.zeros([16, 1], D))

    def callback(point):
        index = calls.assign_add(1) - 1
        recorded = points.scatter_nd_update(index[None, None], point[None])
        with tf.control_dependencies([recorded]):
            value = -.5 * tf.reduce_sum((point - 1.) ** 2)
            score = 1. - point
        if case == 'endpoint_invalid':
            value = tf.where(point[0] > 3., tf.constant(float('nan'), D), value)
        if case == 'sentinel':
            value = tf.where(point[0] > 3., tf.constant(-1e30, D), value)
            score = tf.where(point[0] > 3., tf.zeros_like(score), score)
        return value, score

    def optimizer(function, initial_position, **kwargs):
        first = function(initial_position)
        with tf.control_dependencies(first):
            second = function(tf.ones_like(initial_position))
        with tf.control_dependencies(second):
            value, gradient = function(tf.fill([1], tf.constant(2., D)))
        endpoint = (float('inf') if case == 'invalid_position' else
                    10. if case in ('endpoint_invalid', 'sentinel') else 2.)
        return LBfgsOptimizerResults(
            converged=tf.constant(True), failed=tf.constant(case == 'failed'),
            num_iterations=tf.constant(1),
            num_objective_evaluations=tf.constant(4 if case == 'accounting' else 3),
            position=tf.fill([1], tf.constant(endpoint, D)), objective_value=value,
            objective_gradient=gradient, position_deltas=tf.zeros([10, 1], D),
            gradient_deltas=tf.zeros([10, 1], D))

    monkeypatch.setattr(joint_center_tf.tfp.optimizer, 'lbfgs_minimize', optimizer)
    config = JointCenterLocatorConfig(max_objective_evaluations=2 if case == 'cap' else 12)
    point, scale = tf.zeros([1], D), tf.ones([1], D)
    expected = original.locate_joint_center(callback, point, scale=scale,
        config=original.JointCenterLocatorConfig(**dataclasses.asdict(config)))
    expected_calls = clean({'calls': calls.read_value(), 'positions': points[:int(calls)]})
    calls.assign(0)
    points.assign(tf.zeros_like(points))
    program = make_joint_center_program(callback, 1, config)
    actual = joint_center_result(program(point, scale))
    actual_calls = clean({'calls': calls.read_value(), 'positions': points[:int(calls)]})
    save(request, f'joint-native-failure-{case}.json',
        {'actual': clean(dataclasses.asdict(actual)), 'expected': clean(dataclasses.asdict(expected)),
         'actual_calls': actual_calls, 'expected_calls': expected_calls,
         'reference_sources': checkpoint.hashes(), 'reference_gpu_compatibility': compatibility})
    _equal_records(clean(dataclasses.asdict(actual)), clean(dataclasses.asdict(expected)))
    _equal_records(actual_calls, expected_calls)
    assert actual.best_evaluated_source == 'optimizer_callback'
    assert actual.best_evaluated_callback_index == 1
    assert actual.physical_target_rows == actual_calls['calls']
