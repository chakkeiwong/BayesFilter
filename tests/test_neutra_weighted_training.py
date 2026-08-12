"""Analytic mechanics tests for weighted forward-KL NeuTra training."""

from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_weighted_training import (
    MatchedReverseKLNeuTraTrainer,
    WEIGHTED_NEUTRA_NONCLAIMS,
    WeightedDenseIAFTransport,
    WeightedForwardKLNeuTraTrainer,
    WeightedNeuTraConfig,
)


DTYPE = tf.float64


def _config(**overrides) -> WeightedNeuTraConfig:
    values = {
        "dimension": 2,
        "hidden_layers": (8, 8),
        "stages": 2,
        "activation": "tanh",
        "initialization_scale": 0.02,
        "initialization_seed": (20260811, 9101),
        "learning_rate": 1.0e-3,
        "jit_compile": True,
    }
    values.update(overrides)
    return WeightedNeuTraConfig(**values)


def test_initial_transport_preserves_standard_normal_density() -> None:
    transport = WeightedDenseIAFTransport(_config())
    rows = tf.constant(((0.0, 0.0), (1.0, -2.0), (-0.5, 0.25)), DTYPE)
    physical, logdet = transport.forward_and_logdet(rows)
    recovered, inverse_forward_logdet = transport.inverse_and_forward_logdet(physical)
    expected = -0.5 * (
        tf.reduce_sum(tf.square(rows), axis=1)
        + tf.constant(2.0 * math.log(2.0 * math.pi), DTYPE)
    )
    # Two identity autoregressive stages contain one fixed coordinate reversal.
    tf.debugging.assert_near(
        physical, tf.reverse(rows, axis=(-1,)), atol=1.0e-15, rtol=1.0e-15
    )
    tf.debugging.assert_near(recovered, rows, atol=1.0e-15, rtol=1.0e-15)
    tf.debugging.assert_near(logdet, tf.zeros(3, DTYPE), atol=1.0e-15, rtol=0.0)
    tf.debugging.assert_near(
        inverse_forward_logdet, tf.zeros(3, DTYPE), atol=1.0e-15, rtol=0.0
    )
    tf.debugging.assert_near(transport.log_prob(rows), expected, atol=1.0e-14)


def test_inverse_roundtrip_and_logdet_hold_after_parameter_perturbation() -> None:
    transport = WeightedDenseIAFTransport(_config(stages=3))
    for index, variable in enumerate(transport.trainable_variables):
        seed = tf.constant((20260811, 9200 + index), tf.int32)
        variable.assign(
            variable
            + tf.random.stateless_normal(variable.shape, seed=seed, dtype=DTYPE) * 0.03
        )
    latent = tf.random.stateless_normal((17, 2), seed=(20260811, 9301), dtype=DTYPE)
    physical, forward_logdet = transport.forward_and_logdet(latent)
    recovered, recovered_forward_logdet = transport.inverse_and_forward_logdet(physical)
    tf.debugging.assert_near(recovered, latent, atol=2.0e-12, rtol=2.0e-12)
    tf.debugging.assert_near(
        recovered_forward_logdet, forward_logdet, atol=2.0e-12, rtol=2.0e-12
    )


def test_weighted_loss_matches_manual_reduction_and_weight_diagnostics() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config())
    rows = tf.constant(((0.0, 0.0), (1.0, 0.0), (0.0, 2.0), (-1.0, -1.0)), DTYPE)
    log_weights = tf.math.log(tf.constant((0.1, 0.2, 0.3, 0.4), DTYPE))
    validation = trainer.validation_batch(rows, log_weights)
    normalized = tf.nn.softmax(log_weights)
    expected = tf.reduce_sum(normalized * -trainer.log_prob(rows))
    tf.debugging.assert_near(validation.loss, expected, atol=1.0e-13)
    tf.debugging.assert_near(validation.normalized_weights, normalized, atol=1.0e-14)
    tf.debugging.assert_near(
        validation.effective_sample_size,
        tf.math.reciprocal(tf.reduce_sum(tf.square(normalized))),
        atol=1.0e-13,
    )
    assert float(validation.maximum_normalized_weight.numpy()) == pytest.approx(0.4)


def test_weighted_gradient_matches_finite_difference() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config(jit_compile=False))
    rows = tf.constant(((0.3, -0.7), (1.2, 0.5), (-1.1, 0.8), (0.4, 1.3)), DTYPE)
    log_weights = tf.math.log(tf.constant((0.1, 0.2, 0.3, 0.4), DTYPE))
    variable = trainer.variables[-1]
    index = 2
    with tf.GradientTape() as tape:
        loss = trainer.validation_batch(rows, log_weights).loss
    analytic = tape.gradient(loss, variable)[index]
    original = variable.read_value()
    epsilon = tf.constant(1.0e-5, DTYPE)
    direction = tf.one_hot(index, variable.shape[0], dtype=DTYPE)
    variable.assign(original + epsilon * direction)
    plus = trainer.validation_batch(rows, log_weights).loss
    variable.assign(original - epsilon * direction)
    minus = trainer.validation_batch(rows, log_weights).loss
    variable.assign(original)
    numeric = (plus - minus) / (2.0 * epsilon)
    tf.debugging.assert_near(analytic, numeric, atol=2.0e-7, rtol=2.0e-6)


def test_xla_weighted_update_is_finite_and_replays_from_same_initialization() -> None:
    rows = tf.random.stateless_normal((32, 2), seed=(20260811, 9401), dtype=DTYPE)
    log_weights = tf.linspace(tf.constant(-2.0, DTYPE), tf.constant(1.0, DTYPE), 32)
    left = WeightedForwardKLNeuTraTrainer(_config())
    right = WeightedForwardKLNeuTraTrainer(_config())
    first = left.train_step(rows, log_weights)
    replay = right.train_step(rows, log_weights)
    tf.debugging.assert_near(first.loss, replay.loss, atol=1.0e-14, rtol=1.0e-14)
    assert int(first.step.numpy()) == 1
    assert bool(tf.math.is_finite(first.gradient_norm).numpy())
    for left_variable, right_variable in zip(left.variables, right.variables):
        tf.debugging.assert_near(
            left_variable, right_variable, atol=1.0e-14, rtol=1.0e-14
        )
    assert "weighted particles are not an unweighted posterior archive" in WEIGHTED_NEUTRA_NONCLAIMS


def test_invalid_weights_and_shapes_fail_closed() -> None:
    trainer = WeightedForwardKLNeuTraTrainer(_config(jit_compile=False))
    with pytest.raises(ValueError, match="log_weights"):
        trainer.train_step(tf.zeros((4, 2), DTYPE), tf.zeros(3, DTYPE))
    with pytest.raises(tf.errors.InvalidArgumentError):
        trainer.validation_batch(
            tf.zeros((4, 2), DTYPE),
            tf.constant((0.0, 0.0, float("nan"), 0.0), DTYPE),
        )


def test_matched_reverse_kl_comparator_uses_same_transport_and_xla() -> None:
    config = _config()

    def target(rows: tf.Tensor) -> tf.Tensor:
        return -0.5 * tf.reduce_sum(tf.square(rows - 0.5), axis=1)

    trainer = MatchedReverseKLNeuTraTrainer(config, target)
    latent = tf.random.stateless_normal((32, 2), seed=(20260811, 9501), dtype=DTYPE)
    result = trainer.train_step(latent)
    assert int(result.step.numpy()) == 1
    assert bool(tf.math.is_finite(result.loss).numpy())
    assert len(trainer.variables) == len(WeightedDenseIAFTransport(config).trainable_variables)
