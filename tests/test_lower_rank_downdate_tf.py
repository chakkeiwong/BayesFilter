from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.lower_rank_downdate_tf import batched_lower_rank_downdate


def test_sequential_downdate_reconstructs_and_derivative_matches_fd() -> None:
    rng = np.random.default_rng(8)
    factor = tf.constant(np.tile(np.diag([2.0, 1.7, 1.4])[None], (2, 1, 1)), tf.float64)
    vectors = tf.constant(rng.normal(size=(2, 3, 2)) * 0.15, tf.float64)
    d_factor = tf.constant(rng.normal(size=(2, 1, 3, 3)) * 0.05, tf.float64)
    d_vectors = tf.constant(rng.normal(size=(2, 1, 3, 2)) * 0.05, tf.float64)
    result, dresult, diagnostics = batched_lower_rank_downdate(factor, vectors, d_factor, d_vectors)
    target = factor @ tf.transpose(factor, [0, 2, 1]) - vectors @ tf.transpose(vectors, [0, 2, 1])
    np.testing.assert_allclose(result @ tf.transpose(result, [0, 2, 1]), target, rtol=1e-12, atol=1e-12)
    assert float(tf.reduce_min(diagnostics["minimum_downdate_margin"])) > 0
    eps = 1e-6
    plus = batched_lower_rank_downdate(factor + eps * d_factor[:, 0], vectors + eps * d_vectors[:, 0])[0]
    minus = batched_lower_rank_downdate(factor - eps * d_factor[:, 0], vectors - eps * d_vectors[:, 0])[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), dresult[:, 0], rtol=1e-5, atol=1e-8)


def test_downdate_rejects_indefinite_and_nonfinite_inputs() -> None:
    factor = tf.eye(2, batch_shape=[1], dtype=tf.float64)
    with pytest.raises(tf.errors.InvalidArgumentError):
        batched_lower_rank_downdate(factor, tf.constant([[[2.0], [0.0]]], tf.float64))[0].numpy()
    with pytest.raises(tf.errors.InvalidArgumentError):
        batched_lower_rank_downdate(factor, tf.constant([[[np.nan], [0.0]]], tf.float64))[0].numpy()
