"""Regression tests for float64 weighted-IAF score pullbacks."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)


def test_elu_pullback_preserves_float64_dtype() -> None:
    transport = WeightedDenseIAFTransport(
        WeightedNeuTraConfig(
            dimension=5,
            hidden_layers=(7, 7),
            stages=3,
            initialization_seed=(20260813, 49001),
            initialization_scale=0.02,
        )
    )
    latent = tf.random.stateless_normal((4, 5), seed=(20260813, 49002), dtype=tf.float64)
    score = tf.random.stateless_normal((4, 5), seed=(20260813, 49003), dtype=tf.float64)
    pulled = transport.pullback_score_batch(latent, score)
    logdet_score = transport.log_abs_det_jacobian_score_batch(latent)
    assert pulled.dtype == tf.float64
    assert logdet_score.dtype == tf.float64
    tf.debugging.assert_all_finite(pulled, "weighted IAF pullback")
    tf.debugging.assert_all_finite(logdet_score, "weighted IAF logdet score")
