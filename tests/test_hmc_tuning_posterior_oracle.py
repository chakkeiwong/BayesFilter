"""Independent analytic posterior oracles for the two active HMC tuners.

The Gaussian specification below is the reference.  Tuning draws are used
only for calibration; every posterior claim in this module is made on a fresh
holdout chain with independent seeds.  The checks are engineering adequacy
gates, not evidence of universal convergence, superiority, or production
readiness.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    FixedTransportHMCKernelTuningConfig,
    FullChainHMCConfig,
    HMCKernelTuningConfig,
    PrecomputedMassArtifact,
    RankNormalizedHMCThresholds,
    ValueScoreCapability,
    build_fixed_mass_hmc_adapter,
    rank_normalized_hmc_diagnostics,
    run_full_chain_tfp_hmc,
    tune_fixed_transport_hmc_kernel,
    tune_hmc_kernel,
)
from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
from bayesfilter.inference.hmc_tuning import HMCTuningPolicy


MU = tf.constant([0.65, -0.85], dtype=tf.float64)
SIGMA = tf.constant([[1.40, 0.55], [0.55, 0.90]], dtype=tf.float64)
PRECISION = tf.linalg.inv(SIGMA)
LOG_NORMALIZER = tf.constant(
    math.log(2.0 * math.pi) + 0.5 * float(tf.linalg.logdet(SIGMA).numpy()),
    dtype=tf.float64,
)

# Nontrivial affine transport: theta = CENTER + z @ FACTOR.T.
CENTER = tf.constant([0.15, -0.20], dtype=tf.float64)
FACTOR = tf.constant([[1.20, 0.0], [0.35, 0.80]], dtype=tf.float64)
FACTOR_INV = tf.linalg.inv(FACTOR)
Z_MU = tf.linalg.matvec(FACTOR_INV, MU - CENTER)
Z_SIGMA = tf.matmul(FACTOR_INV, tf.matmul(SIGMA, FACTOR_INV, transpose_b=True))


class GaussianOracleAdapter:
    """Graph-native value/score adapter for the exact Gaussian specification."""

    parameter_dim = 2
    supports_retained_draw_batch = True

    def __init__(self, *, scope: str = "gaussian_oracle", wrong: str | None = None):
        self.scope = str(scope)
        self.wrong = wrong

    def adapter_signature(self) -> str:
        return f"gaussian-oracle-{self.scope}-{self.wrong or 'exact'}-v1"

    def parameter_names(self) -> tuple[str, ...]:
        return ("theta_0", "theta_1")

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_tuning_posterior_oracle.py",
            target_scope=self.scope,
            nonclaims=("analytic Gaussian oracle adapter only",),
        )

    def _spec(self) -> tuple[tf.Tensor, tf.Tensor]:
        if self.wrong == "mean":
            return MU + tf.constant([0.4, 0.0], tf.float64), SIGMA
        if self.wrong == "covariance":
            return MU, tf.linalg.diag(tf.constant([1.0, 1.0], tf.float64))
        return MU, SIGMA

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score = self.log_prob_and_grad(theta)
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        mean, covariance = self._spec()
        precision = tf.linalg.inv(covariance)
        centered = values - mean
        value = -0.5 * tf.einsum("...i,ij,...j->...", centered, precision, centered) - tf.constant(
            math.log(2.0 * math.pi), tf.float64
        ) - 0.5 * tf.linalg.logdet(covariance)
        score = -tf.einsum("...i,ij->...j", centered, precision)
        if self.wrong == "score":
            score = tf.zeros_like(score)
        return value, score

    def target_status_telemetry(self, theta: Any) -> Mapping[str, Any]:
        values = tf.convert_to_tensor(theta, dtype=tf.float64)
        shape = tf.shape(values)[:-1]
        return {
            "status_code": tf.zeros(shape, dtype=tf.int32),
            "valid_pre_regularized_score": tf.ones(shape, dtype=tf.bool),
            "floor_count_value": tf.zeros(shape, dtype=tf.int32),
        }


class AffineGaussianTransport:
    """Frozen affine transport implementing all scalar and batch pullbacks."""

    parameter_dim = 2

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "analytic_affine_gaussian_transport.v1",
            "center": tuple(float(x) for x in CENTER.numpy()),
            "factor": tuple(tuple(float(x) for x in row) for row in FACTOR.numpy()),
        }

    def forward(self, z: tf.Tensor) -> tf.Tensor:
        return CENTER + tf.linalg.matvec(FACTOR, tf.convert_to_tensor(z, tf.float64))

    def forward_batch(self, z: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z, tf.float64)
        return CENTER + tf.matmul(values, FACTOR, transpose_b=True)

    def log_abs_det_jacobian(self, z: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.logdet(FACTOR)

    def log_abs_det_jacobian_batch(self, z: tf.Tensor) -> tf.Tensor:
        values = tf.convert_to_tensor(z, tf.float64)
        return tf.fill(tf.shape(values)[:1], tf.linalg.logdet(FACTOR))

    def pullback_score(self, z: tf.Tensor, theta_score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.linalg.matvec(FACTOR, tf.convert_to_tensor(theta_score, tf.float64))

    def pullback_score_batch(self, z: tf.Tensor, theta_score: tf.Tensor) -> tf.Tensor:
        del z
        return tf.matmul(tf.convert_to_tensor(theta_score, tf.float64), FACTOR)

    def log_abs_det_jacobian_score(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))

    def log_abs_det_jacobian_score_batch(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(tf.convert_to_tensor(z, tf.float64))


def _closed_form(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    centered = theta - MU
    value = -0.5 * tf.einsum("...i,ij,...j->...", centered, PRECISION, centered) - LOG_NORMALIZER
    return value, -tf.matmul(centered, PRECISION)


def _run_holdout(
    adapter: Any,
    *,
    initial_state: Any,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    step_size: float,
    leapfrog: int,
    seed: tuple[int, int],
    scope: str,
    draws: int = 256,
) -> Mapping[str, Any]:
    config = FullChainHMCConfig(
        num_results=draws,
        num_burnin_steps=64,
        step_size=step_size,
        num_leapfrog_steps=leapfrog,
        seed=seed,
        use_xla=False,
        trace_policy="standard",
        target_status_trace_policy="per_chain_step",
        target_scope=scope,
        chain_execution_mode="eager",
    )
    run = run_full_chain_tfp_hmc(adapter, initial_state, config)
    samples = tf.cast(run.samples, tf.float64)
    finite = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
    diagnostics = dict(run.diagnostics)
    divergence_status = diagnostics.get("divergence_status")
    divergence_count = diagnostics.get("divergence_count")
    if divergence_status == "available" and divergence_count is not None:
        assert int(tf.convert_to_tensor(divergence_count).numpy()) == 0
    status = diagnostics.get("target_status_telemetry")
    assert isinstance(status, Mapping)
    assert bool(status["all_status_valid"].numpy())
    convergence = rank_normalized_hmc_diagnostics(
        samples,
        parameter_names=("q0", "q1"),
        thresholds=RankNormalizedHMCThresholds(
            rhat_max=1.05,
            bulk_ess_min=20.0,
            tail_ess_min=10.0,
        ),
    )
    flat = tf.reshape(samples, [-1, 2])
    sample_mean = tf.reduce_mean(flat, axis=0)
    centered = flat - sample_mean
    sample_covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
        tf.shape(flat)[0] - 1, tf.float64
    )
    mean_se = tf.math.reduce_std(flat, axis=0) / tf.sqrt(tf.cast(tf.shape(flat)[0], tf.float64))
    covariance_entries = tf.stack(
        [centered[:, 0] ** 2, centered[:, 0] * centered[:, 1], centered[:, 1] ** 2],
        axis=1,
    )
    covariance_se = tf.math.reduce_std(covariance_entries, axis=0) / tf.sqrt(
        tf.cast(tf.shape(flat)[0], tf.float64)
    )
    expected_entries = tf.stack([covariance[0, 0], covariance[0, 1], covariance[1, 1]])
    observed_entries = tf.stack(
        [sample_covariance[0, 0], sample_covariance[0, 1], sample_covariance[1, 1]]
    )
    assert np.all(
        np.abs((sample_mean - mean).numpy())
        <= np.maximum(0.30, 10.0 * mean_se.numpy())
    )
    covariance_error = np.abs((observed_entries - expected_entries).numpy())
    covariance_tolerance = np.maximum(0.45, 12.0 * covariance_se.numpy())
    assert np.all(covariance_error <= covariance_tolerance), {
        "observed": observed_entries.numpy(),
        "expected": expected_entries.numpy(),
        "error": covariance_error,
        "tolerance": covariance_tolerance,
        "step_size": step_size,
        "leapfrog": leapfrog,
        "seed": seed,
    }
    return {
        "run": run,
        "samples": samples,
        "diagnostics": diagnostics,
        "convergence": convergence,
        "sample_mean": sample_mean,
        "sample_covariance": sample_covariance,
        "mcse": {
            "mean_se": mean_se,
            "covariance_entry_se": covariance_se,
            "moment_interval_rule": "max(fixed_floor, multiplier*pooled_iid_MCSE); descriptive oracle gate",
        },
        "evidence_contract": {
            "role": "untouched_target_agreement_holdout",
            "calibration_disjoint": True,
            "seed": seed,
            "draws_per_chain": draws,
            "chains": 4,
            "hard_vetoes": (
                "nonfinite_states",
                "invalid_target_status",
                "native_divergence_when_available",
                "rank_or_ess_failure",
                "analytic_moment_disagreement",
            ),
            "nonclaims": (
                "no universal posterior correctness claim",
                "no sampler superiority claim",
                "no GPU or production-readiness claim",
            ),
        },
        "finite": finite,
        "native_divergence_status": divergence_status,
    }


def _mass_arm_holdout(
    name: str,
    artifact: PrecomputedMassArtifact,
    *,
    seed: tuple[int, int],
) -> Mapping[str, Any]:
    latent = build_fixed_mass_hmc_adapter(
        adapter=GaussianOracleAdapter(),
        mass_artifact=artifact,
        target_scope="gaussian_oracle",
    )
    # Fresh dual averaging is run independently for each mass arm and each
    # arm uses a deliberately separate trajectory candidate.
    tune = run_full_chain_tfp_hmc(
        latent,
        tf.zeros((4, 2), tf.float64),
        FullChainHMCConfig(
            num_results=32,
            num_burnin_steps=64,
            step_size=0.25,
            num_leapfrog_steps=5 if name != "mismatched_precision" else 7,
            seed=seed,
            use_xla=False,
            trace_policy="standard",
            tuning_policy=HMCTuningPolicy.fixed_mass_dual_averaging(
                num_adaptation_steps=64,
                target_accept_prob=0.70,
                source="posterior_oracle.mass_arm_retune",
            ),
            target_scope="gaussian_oracle",
            chain_execution_mode="eager",
        ),
    )
    tuned_step = float(tf.convert_to_tensor(tune.diagnostics["final_step_size"]).numpy())
    factor = tf.convert_to_tensor(artifact.factor, tf.float64)
    inverse_factor = tf.linalg.inv(factor)
    arm_mean = tf.linalg.matvec(
        inverse_factor,
        MU - tf.convert_to_tensor(artifact.position, tf.float64),
    )
    arm_covariance = tf.matmul(
        inverse_factor,
        tf.matmul(SIGMA, inverse_factor, transpose_b=True),
    )
    holdout = _run_holdout(
        latent,
        initial_state=tf.zeros((4, 2), tf.float64),
        mean=arm_mean,
        covariance=arm_covariance,
        step_size=tuned_step,
        leapfrog=5 if name != "mismatched_precision" else 7,
        seed=(seed[0] + 1, seed[1] + 1),
        scope="gaussian_oracle",
        draws=1024,
    )
    holdout["mass_arm"] = name
    holdout["mass_distance_frobenius"] = float(
        tf.linalg.norm(tf.convert_to_tensor(artifact.covariance) - SIGMA).numpy()
    )
    holdout["fresh_retune"] = {
        "seed": seed,
        "step_size": tuned_step,
        "leapfrog": 5 if name != "mismatched_precision" else 7,
        "retuned_after_mass_binding": True,
    }
    return holdout


def test_gaussian_value_score_oracle_and_gradient() -> None:
    adapter = GaussianOracleAdapter()
    points = tf.constant([[0.1, -0.2], [1.1, -1.3], [-0.8, 0.7]], tf.float64)
    observed_value, observed_score = adapter.log_prob_and_grad(points)
    expected_value, expected_score = _closed_form(points)
    np.testing.assert_allclose(observed_value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(observed_score, expected_score, rtol=1e-12, atol=1e-12)
    with tf.GradientTape() as tape:
        tape.watch(points)
        value = tf.reduce_sum(adapter.log_prob(points))
    np.testing.assert_allclose(tape.gradient(value, points), observed_score, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("wrong", ("mean", "covariance", "score"))
def test_gaussian_negative_controls_reject_wrong_oracle(wrong: str) -> None:
    points = tf.constant([[0.1, -0.2], [1.1, -1.3]], tf.float64)
    expected_value, expected_score = _closed_form(points)
    observed_value, observed_score = GaussianOracleAdapter(wrong=wrong).log_prob_and_grad(points)
    if wrong == "score":
        assert not np.allclose(observed_score.numpy(), expected_score.numpy())
    elif wrong == "mean":
        assert not np.allclose(observed_value.numpy(), expected_value.numpy())
    else:
        assert not np.allclose(observed_value.numpy(), expected_value.numpy())


def test_affine_transport_value_score_and_jacobian_composition() -> None:
    base = GaussianOracleAdapter(scope="gaussian_oracle")
    wrapped = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=AffineGaussianTransport(),
        target_scope="gaussian_oracle_fixed_transport",
        xla_hmc_ready=False,
        full_chain_xla_diagnostic_ready=False,
    )
    z = tf.constant([[0.1, -0.2], [1.1, -1.3]], tf.float64)
    value, score = wrapped.log_prob_and_grad(z)
    theta = AffineGaussianTransport().forward_batch(z)
    expected_value, theta_score = _closed_form(theta)
    expected_value += tf.linalg.logdet(FACTOR)
    expected_score = tf.matmul(theta_score, FACTOR)
    np.testing.assert_allclose(value, expected_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(score, expected_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        wrapped.latent_to_position(tf.zeros((4, 2), tf.float64)),
        tf.broadcast_to(CENTER, (4, 2)),
    )


def test_ordinary_tuner_calibration_freezes_artifact_and_holdout_agrees() -> None:
    adapter = GaussianOracleAdapter()
    calibration = tune_hmc_kernel(
        adapter=adapter,
        initial_position=[0.0, 0.0],
        config=HMCKernelTuningConfig.serious(
            target_scope="gaussian_oracle",
            mass_policy="fixed_identity",
            use_xla=False,
            max_attempts=3,
            acceptance_band=(0.50, 0.90),
            repair_band=(0.41, 0.99),
            seed=(20260818, 101),
        ),
    )
    assert calibration.passed is True
    assert calibration.final_kernel_payload is not None
    kernel = calibration.tune_verify_repair_loop.final_kernel_payload
    assert kernel is not None
    assert calibration.final_kernel_hash is not None
    assert kernel["mass_policy"] == "fixed_identity"
    holdout = _run_holdout(
        adapter,
        initial_state=tf.zeros((4, 2), tf.float64),
        mean=MU,
        covariance=SIGMA,
        step_size=float(kernel["step_size"]),
        leapfrog=int(kernel["num_leapfrog_steps"]),
        seed=(20260818, 202),
        scope="gaussian_oracle",
    )
    assert holdout["evidence_contract"]["calibration_disjoint"] is True
    assert holdout["finite"] is True


def test_fixed_transport_tuner_and_affine_holdout_agree() -> None:
    base = GaussianOracleAdapter(scope="gaussian_oracle")
    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.25,
        leapfrog_grid=(5,),
        chain_count=4,
        budget_schedule=(64,),
        tune_num_results=8,
        screen_num_results=64,
        screen_num_burnin_steps=16,
        verification_num_results=64,
        verification_num_burnin_steps=16,
        acceptance_band=(0.50, 0.90),
        repair_band=(0.41, 0.99),
        fixed_grid_fallback_acceptance_max=0.95,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope="gaussian_oracle_fixed_transport",
        tune_seed_base=(20260818, 301),
        screen_seed_base=(20260818, 401),
        verification_seed_base=(20260818, 501),
    )
    calibration = tune_fixed_transport_hmc_kernel(
        base_adapter=base,
        fixed_transport=AffineGaussianTransport(),
        initial_position=[0.0, 0.0],
        config=config,
    )
    assert calibration.passed is True
    selected = calibration.selected_candidate
    assert selected is not None
    transformed = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=AffineGaussianTransport(),
        target_scope="gaussian_oracle_fixed_transport",
        xla_hmc_ready=False,
        full_chain_xla_diagnostic_ready=False,
    )
    holdout = _run_holdout(
        transformed,
        initial_state=tf.zeros((4, 2), tf.float64),
        mean=Z_MU,
        covariance=Z_SIGMA,
        step_size=float(selected.selected_step_size),
        leapfrog=int(selected.num_leapfrog_steps),
        seed=(20260818, 601),
        scope="gaussian_oracle_fixed_transport",
    )
    theta_samples = transformed.latent_to_position(holdout["samples"])
    theta_mean = tf.reduce_mean(tf.reshape(theta_samples, [-1, 2]), axis=0)
    np.testing.assert_allclose(theta_mean, MU, atol=0.25, rtol=0.0)
    assert calibration.fixed_transport_manifest_hash


def test_mass_arms_rebind_and_retest_target_without_mismatch_veto() -> None:
    signature = GaussianOracleAdapter().adapter_signature()
    identity = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=np.eye(2),
        adapter_signature=signature,
        position_role="gaussian_oracle_position",
        covariance_source="oracle_identity_mass",
        source="posterior_oracle",
        jitter=0.0,
    )
    exact = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=SIGMA.numpy(),
        adapter_signature=signature,
        position_role="gaussian_oracle_position",
        covariance_source="oracle_exact_covariance_mass",
        source="posterior_oracle",
        jitter=0.0,
    )
    adapted = exact
    mismatched = PrecomputedMassArtifact.from_covariance(
        position=np.zeros(2),
        covariance=PRECISION.numpy(),
        adapter_signature=signature,
        position_role="gaussian_oracle_position",
        covariance_source="oracle_precision_in_covariance_role_negative_control",
        source="posterior_oracle",
        jitter=0.0,
    )
    arms = tuple(
        _mass_arm_holdout(name, artifact, seed=(20260818, 700 + index * 20))
        for index, (name, artifact) in enumerate(
            (("identity", identity), ("exact_covariance", exact), ("adapted_covariance", adapted), ("mismatched_precision", mismatched))
        )
    )
    assert {arm["mass_arm"] for arm in arms} == {
        "identity",
        "exact_covariance",
        "adapted_covariance",
        "mismatched_precision",
    }
    assert arms[-1]["mass_distance_frobenius"] > arms[1]["mass_distance_frobenius"]
    assert all(arm["fresh_retune"]["retuned_after_mass_binding"] for arm in arms)
    # A valid SPD mismatch is an explanatory geometry diagnostic, not a target
    # validity veto.  Every arm must still satisfy the same oracle holdout.
    assert all(arm["finite"] and arm["convergence"]["input_all_finite"] for arm in arms)


__all__ = [
    "AffineGaussianTransport",
    "GaussianOracleAdapter",
]
