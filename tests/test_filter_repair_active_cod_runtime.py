"""Compact/padded COD primal, full-rank pullback and compiler-input checks."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.qr_lstsq_tf import (
    complete_orthogonal_lstsq,
    complete_orthogonal_lstsq_active_rows,
)

D = tf.float64


def _data(active=7, capacity=12, rank_deficient=False):
    index = np.arange(active * 3).reshape(active, 3)
    matrix = np.sin(.13 * index ** 2 + .2)
    if rank_deficient:
        matrix[:, 2] = matrix[:, 0] - 2 * matrix[:, 1]
    rhs = np.cos(.37 * np.arange(active * 2).reshape(active, 2))
    return matrix, rhs, np.pad(matrix, ((0, capacity - active), (0, 0)), constant_values=np.nan), np.pad(
        rhs, ((0, capacity - active), (0, 0)), constant_values=np.nan)


def _program(rows, active):
    signature = [tf.TensorSpec([rows, 3], D), tf.TensorSpec([rows, 2], D)]
    if active:
        signature.append(tf.TensorSpec([], tf.int32))

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def solve(matrix, rhs, count=None):
        with tf.GradientTape() as tape:
            tape.watch((matrix, rhs))
            solution = (complete_orthogonal_lstsq(matrix, rhs) if count is None
                else complete_orthogonal_lstsq_active_rows(matrix, rhs, count))
            objective = tf.reduce_sum(tf.sin(solution))
        gradients = tape.gradient(objective, (matrix, rhs))
        return solution, objective, *gradients
    return solve


@pytest.mark.parametrize("active", [6, 7, 12])
def test_active_cod_keeps_primal_and_full_rank_pullback(active):
    a, b, padded_a, padded_b = _data(active)
    compact = _program(active, False)(a, b)
    program = _program(12, True)
    padded = program(padded_a, padded_b, tf.constant(active))
    for index, (before, after) in enumerate(zip(compact, padded, strict=True)):
        np.testing.assert_allclose(after[:active] if index >= 2 else after,
            before, atol=1e-10, rtol=1e-10)
    for gradient in padded[2:]:
        np.testing.assert_array_equal(gradient[active:], np.zeros(gradient[active:].shape))
    direction = np.sin(np.arange(active * 3).reshape(active, 3) + .3)
    step = 1e-5
    plus = np.pad(a + step * direction, ((0, 12 - active), (0, 0)), constant_values=np.nan)
    minus = np.pad(a - step * direction, ((0, 12 - active), (0, 0)), constant_values=np.nan)
    difference = (program(plus, padded_b, tf.constant(active))[1] -
        program(minus, padded_b, tf.constant(active))[1]) / (2 * step)
    np.testing.assert_allclose(difference, tf.reduce_sum(padded[2][:active] * direction), rtol=1e-7, atol=1e-8)


def test_active_cod_rank_threshold_and_runtime_input_reuse():
    @tf.function(input_signature=[tf.TensorSpec([12, 3], D), tf.TensorSpec([12, 2], D),
        tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def solve(a, b, count):
        return complete_orthogonal_lstsq_active_rows(a, b, count)

    hlo = None
    for active, rank_deficient in ((6, False), (7, True), (12, False)):
        a, b, padded_a, padded_b = _data(active, rank_deficient=rank_deficient)
        before = complete_orthogonal_lstsq(tf.constant(a, D), tf.constant(b, D))
        arguments = (tf.constant(padded_a, D), tf.constant(padded_b, D), tf.constant(active))
        after = solve(*arguments)
        np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-10)
        current = solve.experimental_get_compiler_ir(*arguments)(stage="hlo")
        if hlo is None:
            hlo = current
        else:
            assert hlo == current
    assert solve.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("count", [5, 13])
def test_invalid_active_count_is_nonfinite(count):
    _, _, a, b = _data()
    result = _program(12, True)(a, b, tf.constant(count))[0]
    assert bool(tf.reduce_all(tf.math.is_nan(result)))
