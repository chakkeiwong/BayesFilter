from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower


def test_batched_qr_reconstructs_and_derivative_matches_fd() -> None:
    rng = np.random.default_rng(4)
    stack = tf.constant(rng.normal(size=(2, 3, 8)), tf.float64)
    dstack = tf.constant(rng.normal(size=(2, 2, 3, 8)), tf.float64)
    factor, dfactor, diagnostics = batched_stack_qr_lower(stack, dstack)
    covariance = stack @ tf.transpose(stack, [0, 2, 1])
    reconstructed = factor @ tf.transpose(factor, [0, 2, 1])
    np.testing.assert_allclose(reconstructed, covariance, rtol=1e-12, atol=1e-12)
    assert float(tf.reduce_max(diagnostics["factor_derivative_reconstruction_residual"])) < 1e-10
    eps = 1e-6
    plus = batched_stack_qr_lower(stack + eps * dstack[:, 0], None)[0]
    minus = batched_stack_qr_lower(stack - eps * dstack[:, 0], None)[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), dfactor[:, 0], rtol=1e-6, atol=1e-8)
    assert bool(tf.reduce_all(tf.linalg.diag_part(factor) > 0))


def test_qr_batch_permutation_and_invalid_inputs() -> None:
    stack = tf.constant(np.arange(24, dtype=float).reshape(2, 3, 4) + np.eye(3, 4), tf.float64)
    factor = batched_stack_qr_lower(stack)[0]
    permuted = batched_stack_qr_lower(tf.reverse(stack, [0]))[0]
    np.testing.assert_allclose(permuted, tf.reverse(factor, [0]))
    with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
        batched_stack_qr_lower(tf.constant(np.nan, shape=[1, 1, 1], dtype=tf.float64))
    with pytest.raises(ValueError):
        batched_stack_qr_lower(tf.zeros([1, 3, 2], tf.float64))
