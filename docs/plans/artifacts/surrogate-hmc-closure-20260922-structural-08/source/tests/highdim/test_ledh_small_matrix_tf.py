"""Independent library, reconstruction and derivative checks of shared kernels."""
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_small_matrix_tf as small
from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky, cholesky_differential


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("lower,adjoint", [(True, False), (True, True), (False, False), (False, True)])
def test_broadcast_solve_against_library(dtype, order, lower, adjoint):
    matrix = tf.random.stateless_normal([2, order, order], [57, order], dtype=dtype)/5
    matrix += 2*tf.eye(order, dtype=dtype)
    # An extra leading tangent axis must broadcast over the factor batch.
    rhs = tf.random.stateless_normal([3, 2, order, 4], [58, order], dtype=dtype)
    actual = small.triangular_solve(matrix, rhs, lower=lower, adjoint=adjoint)
    expected = tf.linalg.triangular_solve(matrix, rhs, lower=lower, adjoint=adjoint)
    tolerance = 2e-6 if dtype == tf.float32 else 2e-14
    tf.debugging.assert_near(actual, expected, atol=tolerance, rtol=tolerance)
    triangle = tf.linalg.band_part(matrix, -1, 0) if lower else tf.linalg.band_part(matrix, 0, -1)
    if adjoint:
        triangle = tf.linalg.matrix_transpose(triangle)
    tf.debugging.assert_near(triangle @ actual, rhs, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_factor_and_solve_scaled_covariance(dtype, order):
    raw = tf.random.stateless_normal([6, order, order], [59, order], dtype=dtype)
    scales = tf.constant([1e-20, 1e-10, 1., 1e5, 1e10, 1e20], dtype)
    matrix = (raw @ tf.linalg.matrix_transpose(raw)+tf.eye(order, dtype=dtype))*scales[:, None, None]
    valid, factor = safe_cholesky(matrix)
    assert bool(tf.reduce_all(valid))
    expected = tf.linalg.cholesky(matrix)
    tolerance = 3e-6 if dtype == tf.float32 else 3e-14
    tf.debugging.assert_near(factor/tf.sqrt(scales[:, None, None]), expected/tf.sqrt(scales[:, None, None]), atol=tolerance, rtol=tolerance)
    tf.debugging.assert_near((factor @ tf.linalg.matrix_transpose(factor))/scales[:, None, None], matrix/scales[:, None, None], atol=tolerance, rtol=tolerance)
    rhs = tf.random.stateless_normal([6, order, 3], [60, order], dtype=dtype)*scales[:, None, None]
    tf.debugging.assert_near(small.cholesky_solve(factor, rhs), tf.linalg.cholesky_solve(expected, rhs), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_invalid_pivots_and_unused_triangle_cannot_be_accepted(dtype):
    inputs = tf.constant([[[0., 0.], [0., 1.]], [[-1., 0.], [0., 1.]],
                          [[1., 2.], [2., 1.]], [[1., 1.], [1., 1.]],
                          [[1., float("nan")], [0., 1.]], [[1., 2.], [0., 1.]],
                          [[float("inf"), 0.], [0., 1.]]], dtype)
    valid, factor = safe_cholesky(inputs)
    assert not bool(tf.reduce_any(valid))
    tf.debugging.assert_equal(factor, tf.eye(2, batch_shape=[7], dtype=dtype))
    tf.debugging.assert_equal(cholesky_differential(factor, tf.ones_like(factor), valid), tf.zeros_like(factor))
    valid, _ = safe_cholesky(tf.constant([[[0.]], [[-1.]], [[float("inf")]]], dtype))
    assert not bool(tf.reduce_any(valid))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_analytical_cholesky_differential_against_finite_difference(order):
    raw = tf.random.stateless_normal([2, order, order], [61, order], dtype=tf.float64)
    matrix = raw @ tf.linalg.matrix_transpose(raw)+tf.eye(order, dtype=tf.float64)
    direction = (raw+tf.linalg.matrix_transpose(raw))/10
    valid, factor = safe_cholesky(matrix)
    eps = tf.constant(1e-5, tf.float64)
    expected = (tf.linalg.cholesky(matrix+eps*direction)-tf.linalg.cholesky(matrix-eps*direction))/(2*eps)
    tf.debugging.assert_near(cholesky_differential(factor, direction, valid), expected, atol=2e-10, rtol=2e-8)


def test_unknown_order_and_complex_keep_library_behavior():
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float64), tf.TensorSpec([None, 1], tf.float64)], autograph=False)
    def solve(matrix, rhs):
        return small.cholesky(matrix), small.triangular_solve(matrix, rhs)
    matrix = tf.constant([[2., 0.], [.3, 1.]], tf.float64)
    rhs = tf.ones([2, 1], tf.float64)
    factor, result = solve(matrix, rhs)
    tf.debugging.assert_near(factor, tf.linalg.cholesky(matrix))
    tf.debugging.assert_near(result, tf.linalg.triangular_solve(matrix, rhs))
    matrix = tf.cast(matrix, tf.complex128)+tf.constant([[0j, 0j], [.1j, 0j]], tf.complex128)
    rhs = tf.cast(rhs, tf.complex128)
    tf.debugging.assert_near(small.triangular_solve(matrix, rhs, adjoint=True), tf.linalg.triangular_solve(matrix, rhs, adjoint=True))


def test_source_closure_includes_shared_arithmetic():
    from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
    assert "bayesfilter/highdim/ledh_small_matrix_tf.py" in source_closure()
