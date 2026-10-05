"""Independent diagnostic tests for the compiled rank-aware solver."""

import re

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.qr_lstsq_tf import (
    complete_orthogonal_lstsq,
    condition_number,
    full_rank_lstsq,
)


@pytest.mark.parametrize("jit", [False, True])
def test_repeated_columns_preserve_rank_one_minimum_norm(jit):
    matrix = tf.ones([33, 3], tf.float64)
    rhs = tf.broadcast_to(tf.constant([[1.3175, 2.2075, 3.135]], tf.float64), [33, 3])
    solve = tf.function(complete_orthogonal_lstsq, autograph=False, jit_compile=jit,
        input_signature=[tf.TensorSpec([33, 3], tf.float64), tf.TensorSpec([33, 3], tf.float64)])
    actual = solve(matrix, rhs)
    expected = tf.linalg.lstsq(matrix, rhs, fast=False)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual, np.broadcast_to(rhs.numpy()[0] / 3., (3, 3)), atol=1e-12, rtol=1e-12)


def test_cod_retains_matrix_and_rhs_as_runtime_inputs_across_ranks():
    call = tf.function(complete_orthogonal_lstsq, autograph=False, jit_compile=True,
        input_signature=[tf.TensorSpec([4, 3], tf.float64), tf.TensorSpec([4, 1], tf.float64)])
    rhs = tf.constant([[.7], [.3], [.2], [.4]], tf.float64)
    compiler_outputs = []
    for small in (0., 1e-16, 1e-12, .05):
        matrix = tf.constant([[0., 1., .1], [1., 0., .2], [0., 0., small], [0., 0., 0.]], tf.float64)
        np.testing.assert_allclose(call(matrix, rhs), tf.linalg.lstsq(matrix, rhs, fast=False),
            atol=1e-10, rtol=1e-10)
        hlo = call.experimental_get_compiler_ir(matrix, rhs)(stage="hlo")
        entry = hlo[hlo.rfind("\nENTRY "):]
        assert len(re.findall(r"\bparameter\(\d+\)", entry)) == 2
        compiler_outputs.append(hlo)
    assert all(hlo == compiler_outputs[0] for hlo in compiler_outputs)
    assert call.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("small", [0.0, 1e-16, 1e-12, 0.05])
def test_native_cod_preserves_original_rank_truncation(jit, small):
    a = np.array([[0.0, 1.0, 0.1], [1.0, 0.0, 0.2], [0.0, 0.0, small], [0.0, 0.0, 0.0]])
    b = tf.constant([[0.7], [0.3], [0.2], [0.4]], tf.float64)
    a = tf.constant(a, tf.float64)
    call = tf.function(
        complete_orthogonal_lstsq,
        input_signature=[
            tf.TensorSpec(a.shape, a.dtype),
            tf.TensorSpec(b.shape, b.dtype),
        ],
        jit_compile=jit,
        autograph=False,
    )
    expected = tf.linalg.lstsq(a, b, fast=False)
    actual = call(a, b)
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("condition", [1.0, 1e6, 1e10, 1e14, 1e16])
def test_qr_retains_high_condition_range_without_gram_squaring(jit, condition):
    # Orthogonal columns with known spectrum isolate the condition estimator's
    # dynamic range from ambiguity at machine precision in a rotated matrix.
    a = tf.constant(
        np.vstack([np.diag([1.0, 0.1, 1.0 / condition]), np.zeros([2, 3])]), tf.float64
    )
    x = tf.constant([[0.3], [-0.2], [0.7]], tf.float64)
    b = a @ x

    @tf.function(
        input_signature=[
            tf.TensorSpec([5, 3], tf.float64),
            tf.TensorSpec([5, 1], tf.float64),
        ],
        jit_compile=jit,
        autograph=False,
    )
    def solve(a, b):
        return full_rank_lstsq(a, b), condition_number(a, jit_compile=jit)

    actual, observed = solve(a, b)
    np.testing.assert_allclose(actual, x, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(observed, max(10.0, condition), atol=0, rtol=2e-14)


@pytest.mark.parametrize("jit", [False, True])
def test_qr_primal_and_total_derivative_match_original_lstsq(jit):
    from bayesfilter.highdim.fitting import _stable_overdetermined_lstsq

    a = tf.constant(np.random.default_rng(45).normal(size=(12, 4)), tf.float64)
    b = tf.constant(np.random.default_rng(46).normal(size=(12, 1)), tf.float64)
    with tf.GradientTape() as tape:
        tape.watch((a, b))
        expected = _stable_overdetermined_lstsq(a, b)
        loss = tf.reduce_sum(expected**2)
    expected_grad = tape.gradient(loss, (a, b))

    @tf.function(
        input_signature=[
            tf.TensorSpec(a.shape, a.dtype),
            tf.TensorSpec(b.shape, b.dtype),
        ],
        jit_compile=jit,
        autograph=False,
    )
    def solve(a, b):
        with tf.GradientTape() as tape:
            tape.watch((a, b))
            value = complete_orthogonal_lstsq(a, b)
            loss = tf.reduce_sum(value**2)
        return value, tape.gradient(loss, (a, b))

    actual, gradient = solve(a, b)
    for left, right in zip(
        tf.nest.flatten((actual, gradient)), tf.nest.flatten((expected, expected_grad))
    ):
        np.testing.assert_allclose(left, right, atol=1e-12, rtol=1e-12)


def test_fixed_fit_accepts_and_reports_the_native_solver_backend():
    from bayesfilter.highdim.fitting import _solve_scaled_augmented_ridge
    from bayesfilter.ops.qr_lstsq_tf import BACKEND

    design = tf.constant([[1.0, 0.2], [-0.4, 0.9], [0.1, -0.6]], tf.float64)
    kwargs = {
        "design": design,
        "target_values": tf.constant([0.3, -0.7, 0.2], tf.float64),
        "weights": tf.constant([0.2, 0.5, 0.3], tf.float64),
        "ridge": 1e-10,
    }
    expected = _solve_scaled_augmented_ridge(**kwargs)
    actual = _solve_scaled_augmented_ridge(**kwargs, solver_backend=BACKEND)
    np.testing.assert_allclose(
        actual.solution, expected.solution, atol=1e-12, rtol=1e-12
    )
    assert actual.diagnostics["solver_backend"] == BACKEND
    assert actual.diagnostics["stabilization_policy"]["solver_backend"] == BACKEND


@pytest.mark.parametrize("jit", [False, True])
def test_underdetermined_cod_multiple_responses_preserves_minimum_norm(jit):
    matrix = tf.constant([[0.2, -0.6, 0.7, -0.1], [0.8, 0.3, -0.2, 0.9]], tf.float64)
    rhs = tf.constant([[0.2, -0.5], [0.3, 0.7]], tf.float64)
    call = tf.function(
        complete_orthogonal_lstsq,
        autograph=False,
        jit_compile=jit,
        input_signature=[
            tf.TensorSpec(matrix.shape, matrix.dtype),
            tf.TensorSpec(rhs.shape, rhs.dtype),
        ],
    )
    np.testing.assert_allclose(
        call(matrix, rhs),
        tf.linalg.lstsq(matrix, rhs, fast=False),
        atol=1e-12,
        rtol=1e-12,
    )
    with tf.GradientTape() as tape:
        tape.watch((matrix, rhs))
        objective = tf.reduce_sum(call(matrix, rhs) ** 2)
    gradient = tape.gradient(objective, (matrix, rhs))
    directions = (
        tf.constant([[0.1, -0.3, 0.2, 0.05], [-0.04, 0.1, 0.2, -0.3]], tf.float64),
        tf.constant([[0.2, -0.1], [-0.05, 0.3]], tf.float64),
    )
    step = 1e-5
    plus = tf.linalg.lstsq(
        matrix + step * directions[0], rhs + step * directions[1], fast=False
    )
    minus = tf.linalg.lstsq(
        matrix - step * directions[0], rhs - step * directions[1], fast=False
    )
    expected = tf.reduce_sum(plus**2 - minus**2) / (2 * step)
    actual = sum(tf.reduce_sum(g * d) for g, d in zip(gradient, directions))
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("jit", [False, True])
def test_dynamic_row_signature_preserves_rank_and_full_rank_pullbacks(jit):
    @tf.function(input_signature=[tf.TensorSpec([None, 3], tf.float64),
                                 tf.TensorSpec([None, 2], tf.float64)],
                 jit_compile=jit, autograph=False)
    def solve(matrix, rhs):
        with tf.GradientTape() as tape:
            tape.watch((matrix, rhs))
            value = complete_orthogonal_lstsq(matrix, rhs)
            loss = tf.reduce_sum(value**2)
        return value, tape.gradient(loss, (matrix, rhs))

    generator = np.random.default_rng(46)
    for rows in (2, 5, 9):
        matrix = tf.constant(generator.normal(size=[rows, 3]), tf.float64)
        rhs = tf.constant(generator.normal(size=[rows, 2]), tf.float64)
        direction = tf.constant(generator.normal(size=[rows, 3]), tf.float64)
        actual, (gradient, _) = solve(matrix, rhs)
        expected = tf.linalg.lstsq(matrix, rhs, fast=False)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        step = 1e-5
        upper = tf.linalg.lstsq(matrix + step * direction, rhs, fast=False)
        lower = tf.linalg.lstsq(matrix - step * direction, rhs, fast=False)
        np.testing.assert_allclose(tf.reduce_sum(gradient * direction),
                                   tf.reduce_sum(upper**2 - lower**2)/(2*step),
                                   rtol=1e-7, atol=1e-7)
    assert solve.experimental_get_tracing_count() == 1
