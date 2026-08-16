from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)
from bayesfilter.testing.defensive_weighted_neutra_hmc_tf import (
    AnalyticGaussianMixtureValueScoreAdapter,
    analytic_two_mode_target,
    load_weighted_neutra_transport,
    mode_aware_initial_state,
    retained_analytic_diagnostics,
    sample_gaussian_mixture,
    stable_json_hash,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/defensive-weighted-neutra-validation-2026-08-11/"
    "r1-two-mode/capacity-depth6-width128-updates10000-confirmation-1-v1/"
    "trainer_states.json"
)


def _transport() -> WeightedDenseIAFTransport:
    config = WeightedNeuTraConfig(
        dimension=4,
        hidden_layers=(5, 5),
        stages=3,
        activation="tanh",
        initialization_seed=(20260812, 2001),
    )
    return WeightedDenseIAFTransport(config)


def test_explicit_weighted_iaf_scores_match_debug_autodiff() -> None:
    transport = _transport()
    z = tf.constant(
        ((0.2, -0.4, 0.1, 0.3), (-0.3, 0.7, -0.2, 0.5)), tf.float64
    )
    output_score = tf.constant(
        ((0.7, -1.1, 0.2, 0.4), (-0.2, 0.5, 0.3, -0.6)), tf.float64
    )
    with tf.GradientTape() as tape:
        tape.watch(z)
        objective = tf.reduce_sum(transport.forward_batch(z) * output_score)
    expected_pullback = tape.gradient(objective, z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        logdet = tf.reduce_sum(transport.log_abs_det_jacobian_batch(z))
    expected_logdet_score = tape.gradient(logdet, z)

    tf.debugging.assert_near(
        transport.pullback_score_batch(z, output_score),
        expected_pullback,
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    tf.debugging.assert_near(
        transport.log_abs_det_jacobian_score_batch(z),
        expected_logdet_score,
        atol=2.0e-12,
        rtol=2.0e-12,
    )


def test_exact_mixture_score_and_transformed_plus_logdet_match_autodiff() -> None:
    base = AnalyticGaussianMixtureValueScoreAdapter()
    transport = _transport()
    transport.bind_frozen_identity(
        {
            "checkpoint_sha256": "a" * 64,
            "training_state_hash": "b" * 64,
            "transport_tensor_hash": "c" * 64,
        }
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope="weighted-neutra-transformed-test",
        require_batch_native=True,
    )
    z = tf.constant(
        ((0.2, -0.4, 0.1, 0.3), (-0.3, 0.7, -0.2, 0.5)), tf.float64
    )
    actual_value, actual_score = adapter.log_prob_and_grad_batch(z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        theta = transport.forward_batch(z)
        physical_value, _physical_score = base.log_prob_and_grad(theta)
        expected_value = physical_value + transport.log_abs_det_jacobian_batch(z)
        total = tf.reduce_sum(expected_value)
    expected_score = tape.gradient(total, z)

    tf.debugging.assert_near(actual_value, expected_value, atol=2.0e-12, rtol=2.0e-12)
    tf.debugging.assert_near(actual_score, expected_score, atol=2.0e-11, rtol=2.0e-11)


def test_real_weighted_checkpoint_restores_hashes_and_inverse_parity() -> None:
    loaded = load_weighted_neutra_transport(CHECKPOINT)
    assert loaded.selected_step > 0
    assert loaded.state_hash == loaded.transport.manifest_payload()["frozen_identity"][
        "training_state_hash"
    ]
    z = tf.constant(
        ((-1.0, 0.5, 0.2, -0.3), (0.7, -0.2, 1.1, 0.4)), tf.float64
    )
    theta, forward_logdet = loaded.transport.forward_and_logdet(z)
    inverse, inverse_forward_logdet = loaded.transport.inverse_and_forward_logdet(theta)
    tf.debugging.assert_near(inverse, z, atol=2.0e-10, rtol=2.0e-10)
    tf.debugging.assert_near(
        inverse_forward_logdet, forward_logdet, atol=2.0e-10, rtol=2.0e-10
    )
    initial = mode_aware_initial_state(loaded.transport)
    assert initial.shape == (4, 4)
    tf.debugging.assert_all_finite(initial, "initial latent states")


def test_analytic_target_moments_and_exact_reference_diagnostics() -> None:
    target = analytic_two_mode_target()
    expected_mean = tf.reduce_sum(
        target["probabilities"][:, tf.newaxis] * target["means"], axis=0
    )
    tf.debugging.assert_near(target["true_mean"], expected_mean, atol=1.0e-14, rtol=1.0e-14)
    rows, _labels = sample_gaussian_mixture(
        40_000,
        target["probabilities"],
        target["means"],
        target["covariances"],
        seed=(20260812, 3001),
    )
    diagnostics = retained_analytic_diagnostics(
        tf.reshape(rows, (10_000, 4, 4)), reference_seed=(20260812, 3002)
    )
    assert diagnostics["gates"]["all_finite"] is True
    assert abs(diagnostics["minority_mass"] - 0.2) < 0.015
    assert diagnostics["gates"]["both_modes_observed_overall"] is True
    assert diagnostics["joint_moment_test_performed"] is False
    assert diagnostics["moment_diagnostics"]["mean_interval_total_count"] == 4
    assert diagnostics["moment_diagnostics"]["covariance_interval_total_count"] == 16
    assert diagnostics["passed_primary_screens"] is True


def test_checkpoint_state_hash_uses_training_serialization_identity() -> None:
    import json

    payload = json.loads(CHECKPOINT.read_text(encoding="utf-8"))["weighted"]
    expected = payload.pop("state_hash")
    assert stable_json_hash(payload) == expected
