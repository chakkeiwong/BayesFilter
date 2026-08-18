"""CPU reference checks for the generic frozen weighted-mixture HMC authority."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.testing.weighted_neutra_gaussian_mixture_hmc_tf import (
    AnalyticGaussianMixtureValueScoreAdapter,
    analytic_three_mode_target,
    component_aware_initial_state,
)


def _transport() -> WeightedDenseIAFTransport:
    config = WeightedNeuTraConfig(
        dimension=4,
        hidden_layers=(8, 8),
        stages=3,
        activation="tanh",
        initialization_seed=(20260812, 13001),
    )
    transport = WeightedDenseIAFTransport(config)
    transport.bind_frozen_identity(
        {
            "checkpoint_sha256": "a" * 64,
            "training_state_hash": "b" * 64,
            "transport_tensor_hash": "c" * 64,
        }
    )
    return transport


def test_three_mode_adapter_score_matches_gradient_tape_reference() -> None:
    base = AnalyticGaussianMixtureValueScoreAdapter(analytic_three_mode_target())
    points = tf.constant(
        ((-2.0, 1.0, 0.3, -0.2), (1.5, -1.2, 0.6, 0.7)), tf.float64
    )
    actual_value, actual_score = base.log_prob_and_grad(points)
    with tf.GradientTape() as tape:
        tape.watch(points)
        expected_value, _ = base.log_prob_and_grad(points)
        total = tf.reduce_sum(expected_value)
    expected_score = tape.gradient(total, points)
    tf.debugging.assert_all_finite(actual_value, "mixture values")
    tf.debugging.assert_near(actual_score, expected_score, atol=2.0e-12, rtol=2.0e-12)


def test_transformed_three_mode_score_matches_gradient_tape_reference() -> None:
    transport = _transport()
    base = AnalyticGaussianMixtureValueScoreAdapter(analytic_three_mode_target())
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope="weighted-neutra-three-mode-test",
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    latent = tf.constant(
        ((0.2, -0.4, 0.1, 0.3), (-0.3, 0.7, -0.2, 0.5)), tf.float64
    )
    actual_value, actual_score = adapter.log_prob_and_grad_batch(latent)
    with tf.GradientTape() as tape:
        tape.watch(latent)
        physical = transport.forward_batch(latent)
        physical_value, _ = base.log_prob_and_grad(physical)
        expected_value = physical_value + transport.log_abs_det_jacobian_batch(latent)
        total = tf.reduce_sum(expected_value)
    expected_score = tape.gradient(total, latent)
    tf.debugging.assert_near(actual_value, expected_value, atol=2.0e-12, rtol=2.0e-12)
    tf.debugging.assert_near(actual_score, expected_score, atol=3.0e-11, rtol=3.0e-11)


def test_component_aware_initial_states_cover_all_components() -> None:
    target = analytic_three_mode_target()
    transport = _transport()
    latent = component_aware_initial_state(transport, target, chain_count=4)
    physical = transport.forward_batch(latent)
    distances = tf.reduce_sum(
        tf.square(physical[:, tf.newaxis, :] - target["means"][tf.newaxis, :, :]),
        axis=-1,
    )
    assigned = tf.argmin(distances, axis=1, output_type=tf.int32)
    assert set(assigned.numpy().tolist()) == {0, 1, 2}
