"""Independent determinant/value/gradient authority for the XLA primitive."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.ops.slogdet_tf import slogdet_tf


@pytest.mark.parametrize("matrix", [
    [[.3]], [[-2.]], [[0., 3.], [2., .4]],
    [[.01, .3, -.2], [2., 1., .1], [.7, -.4, 1.]],
    [[1e-100, 2e-100], [0., 3e-100]], [[1., 1.], [1., 1.]],
])
def test_xla_signed_logdet_matches_lu_and_inverse_transpose(matrix):
    matrix = np.array(matrix)
    sign, log_abs = np.linalg.slogdet(matrix)
    @tf.function(jit_compile=True, autograph=False)
    def call(value):
        with tf.GradientTape() as tape:
            tape.watch(value)
            actual = slogdet_tf(value)
        return actual, tape.gradient(actual[1], value)
    actual, gradient = call(tf.constant(matrix, tf.float64))
    np.testing.assert_allclose(actual, (sign, log_abs), atol=1e-12, rtol=1e-12)
    if sign:
        inverse_transpose = np.linalg.inv(matrix).T
        # Compare in matrix-relative units: a 1e100 derivative has rounding
        # errors near 1e84 even in entries whose exact value is zero.
        scale = np.max(np.abs(inverse_transpose))
        np.testing.assert_allclose(gradient / scale, inverse_transpose / scale, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(matrix.T @ gradient, np.eye(matrix.shape[0]), atol=1e-12, rtol=1e-12)
