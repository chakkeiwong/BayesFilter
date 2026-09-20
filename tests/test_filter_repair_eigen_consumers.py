"""Regression checks for the exposed sequential XLA eigensystem residuals."""

import re

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_preparation_tf as prepare
from bayesfilter.inference import sequential_score_fit_tf as fit
from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from tests.test_filter_repair_sequential_preparation import baseline as trust_baseline
from tests.test_filter_repair_sequential_score_fit import (
    _assert_payload,
)
from tests.test_filter_repair_sequential_score_fit import (
    baseline as fit_baseline,
)
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64
TRUST_PRECISION = [
    [1.0039470651832982, .015240239018094447, -.05116129653011128],
    [.015240239018094444, .6953013675081983, .02067123492643267],
    [-.05116129653011128, .02067123492643267, .7379628902598411],
]


def test_isolated_eigenpair_graph_preserves_spectral_pullback():
    program = eigenpair_program(3)
    signature = [tf.TensorSpec([3, 3], D)]

    def value_and_gradient(matrix, eigenpairs):
        with tf.GradientTape() as tape:
            tape.watch(matrix)
            values, vectors = eigenpairs(matrix)
            value = tf.reduce_sum(values ** 2) + tf.reduce_sum(vectors ** 4)
        return value, tape.gradient(value, matrix)

    reference = tf.function(lambda matrix: value_and_gradient(matrix, tf.linalg.eigh),
        input_signature=signature, jit_compile=False, autograph=False)
    enclosed = tf.function(lambda matrix: value_and_gradient(matrix, program),
        input_signature=signature, jit_compile=True, autograph=False)
    for multiplier in (1., 1.1):
        matrix = tf.constant(TRUST_PRECISION, D) * multiplier
        expected = reference(matrix)
        actual = enclosed(matrix)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
    assert not program.get_concrete_function().captured_inputs
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize('radius', [.03, 3.])
def test_dense_trust_matches_original_graph_and_stationarity(radius):
    original = trust_baseline.__wrapped__()
    matrix = tf.constant(TRUST_PRECISION, D)
    linear = tf.constant([.1, -.2, .3], D)
    expected = original._solve_trust_region_tf(matrix, linear, radius)
    actual = current._solve_trust_region_tf(matrix, linear, radius)
    assert actual['boundary_active'] == expected['boundary_active']
    np.testing.assert_allclose(actual['step'], expected['step'], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual['predicted_improvement'], expected['predicted_improvement'], atol=1e-10, rtol=1e-10)
    step = tf.convert_to_tensor(actual['step'], D)
    residual = linear - tf.linalg.matvec(matrix, step)
    if actual['boundary_active']:
        multiplier = tf.reduce_sum(residual * step) / tf.reduce_sum(step * step)
        assert float(multiplier) >= 0.
        residual -= multiplier * step
    np.testing.assert_allclose(residual, 0., atol=1e-12, rtol=0.)


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('scale_multiplier', [1., 1.1])
def test_terminal_full_fields_match_original_eager_authority(dimension, scale_multiplier):
    original = fit_baseline.__wrapped__()
    scalar, batched, _ = _inputs(dimension, 32)
    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension) * 1e-6
    score, scale = scalar(center)[1], tf.fill([dimension], tf.constant(scale_multiplier, D))
    cfg = current.SequentialMapCovarianceConfig(terminal_sample_count=24)
    kwargs = {'dimension': dimension, 'radius': .25, 'sample_count': 24, 'seed': (2026, 100719),
        'config': cfg, 'evaluations': 7, 'batched_value_and_score_fn': batched}
    expected, count = original._fit_score_curvature(scalar, center, score, scale, **kwargs)
    with tf.GradientTape() as tape:
        tape.watch(center)
        actual, actual_count = current._fit_score_curvature(scalar, center, score, scale, **kwargs)
        total = tf.reduce_sum(actual['projected_precision_z'])
    assert tape.gradient(total, center) is None
    assert count == actual_count
    _assert_payload(actual, expected)


def test_refined_numerics_keep_runtime_operands_for_changed_matrices():
    scalar, batched, _ = _inputs(5, 32)
    train, heldout = fit.partition_schema(24, .25, pair_disjoint=False)
    program = fit.score_fit_program(scalar, batched, 24, 5, train, heldout)
    args = (tf.zeros([5], D), tf.zeros([5], D), tf.ones([5], D), tf.constant(.25, D),
        tf.constant([2026, 100719]), tf.constant(1e-10, D), tf.constant(1e-8, D),
        tf.constant(1e8, D), tf.constant(.35, D))
    program(*args)
    concrete = program.get_concrete_function()
    arity = len(args) + len(concrete.captured_inputs)
    first = program.experimental_get_compiler_ir(*args)(stage='hlo')
    changed = (args[0] + .01, args[1] + .02, args[2] * 1.1, *args[3:])
    program(*changed)
    second = program.experimental_get_compiler_ir(*changed)(stage='hlo')
    for hlo in (first, second):
        entry = hlo[hlo.rfind('\nENTRY '):]
        assert len(re.findall(r'\bparameter\((\d+)\)', entry)) == arity
    assert first == second
    assert program.experimental_get_tracing_count() == 1
    solver = prepare.trust_region_program(3)
    inputs = (tf.constant(TRUST_PRECISION, D), tf.constant([.1, -.2, .3], D), tf.constant(.3, D))
    solver(*inputs)
    hlo = solver.experimental_get_compiler_ir(*inputs)(stage='hlo')
    assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 3
