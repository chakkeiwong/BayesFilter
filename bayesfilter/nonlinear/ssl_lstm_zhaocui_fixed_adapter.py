"""Deterministic fixed-variant SSL-LSTM Zhao-Cui adapter.

This module implements the narrow Phase-2 ``zhaocui_fixed`` lane: a fixed
reference-sample replay with a manual first-order score.  It records
Zhao-Cui source anchors for the replay/recentering vocabulary, but the target
likelihood is a BayesFilter clean-room fixed adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_protocol import (
    SSLLSTMAdapterProtocol,
    SSLLSTMStaticConfig,
    build_expected_ssl_lstm_adapter_protocol,
    validate_ssl_lstm_value_score_artifact,
)
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    SSLLSTMConstrainedParameters,
    ssl_lstm_observation,
    ssl_lstm_observation_parameter_derivative,
    ssl_lstm_observation_state_jvp,
    ssl_lstm_transition,
    ssl_lstm_transition_parameter_derivative,
    ssl_lstm_transition_state_jvp,
    unpack_ssl_lstm_parameters,
)


@dataclass(frozen=True)
class SSLLSTMZhaoCuiFixedManifest:
    """Fixed branch choices for the deterministic replay adapter."""

    reference_sample_count: int = 7
    initial_seed: tuple[int, int] = (20260705, 2301)
    process_seed: tuple[int, int] = (20260705, 2302)
    recenter_ridge: float = 1.0e-5
    source_route_classification: str = "fixed_hmc_adaptation_with_extension_likelihood"
    reference_seed_policy: str = "stateless_required"
    resampling_policy: str = "not_used_fixed_logmeanexp_replay"
    recenter_policy: str = "computeL_style_weighted_frame_diagnostic_only"
    score_path: str = "manual_first_order_tensorflow_chain_rule"
    forbidden_authorities: tuple[str, ...] = (
        "gradient_tape_fallback",
        "tf_py_function_bridge",
        "finite_difference_target_score",
    )
    source_anchor_summary: tuple[str, ...] = (
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:21-43 sequential replay loop",
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:46-124 fixed reapproximation/recentering structure",
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m:132-135 local prior-transition-likelihood target split",
        "third_party/audit/zhao_cui_tensor_ssm_p10/source/models/computeL.m:24-47 weighted affine recentering diagnostic",
    )
    nonclaims: tuple[str, ...] = (
        "clean-room fixed adaptation only",
        "no source-faithful SSL-LSTM Zhao-Cui parity claim",
        "no TTSIRT or KR map implementation claim",
        "no HMC convergence claim",
        "no posterior correctness claim",
        "no method superiority claim",
        "no default-readiness claim",
    )

    def __post_init__(self) -> None:
        if int(self.reference_sample_count) <= 0:
            raise ValueError("reference_sample_count must be positive")
        if len(self.initial_seed) != 2 or len(self.process_seed) != 2:
            raise ValueError("stateless seeds must contain exactly two integers")
        if float(self.recenter_ridge) <= 0.0:
            raise ValueError("recenter_ridge must be positive")

    def as_dict(self) -> dict[str, object]:
        """Return stable manifest metadata for artifacts and diagnostics."""

        return {
            "reference_sample_count": int(self.reference_sample_count),
            "initial_seed": tuple(int(item) for item in self.initial_seed),
            "process_seed": tuple(int(item) for item in self.process_seed),
            "recenter_ridge": float(self.recenter_ridge),
            "source_route_classification": self.source_route_classification,
            "reference_seed_policy": self.reference_seed_policy,
            "resampling_policy": self.resampling_policy,
            "recenter_policy": self.recenter_policy,
            "score_path": self.score_path,
            "forbidden_authorities": self.forbidden_authorities,
            "source_anchor_summary": self.source_anchor_summary,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class SSLLSTMZhaoCuiFixedComponents:
    """Parameters, protocol metadata, and fixed manifest for the adapter."""

    parameters: SSLLSTMConstrainedParameters
    protocol: SSLLSTMAdapterProtocol
    manifest: SSLLSTMZhaoCuiFixedManifest


@dataclass(frozen=True)
class SSLLSTMZhaoCuiFixedScoreResult:
    """Value, score, and diagnostics from the fixed replay adapter."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    particle_log_likelihoods: tf.Tensor
    normalized_particle_weights: tf.Tensor
    filtered_means: tf.Tensor
    final_particles: tf.Tensor
    diagnostics: Mapping[str, object]
    failure: None = None


def make_ssl_lstm_zhaocui_fixed_components(
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    manifest: SSLLSTMZhaoCuiFixedManifest | None = None,
    std_floor: float = 1.0e-4,
) -> SSLLSTMZhaoCuiFixedComponents:
    """Build the fixed replay adapter components."""

    fixed_manifest = manifest or SSLLSTMZhaoCuiFixedManifest()
    params = unpack_ssl_lstm_parameters(theta, config, std_floor=std_floor)
    protocol = build_expected_ssl_lstm_adapter_protocol(
        config,
        filter_name="zhaocui_fixed",
        evidence_path=evidence_path,
        target_scope="ssl_lstm_filter_hmc:zhaocui_fixed:phase2",
        nonclaims=fixed_manifest.nonclaims,
    )
    return SSLLSTMZhaoCuiFixedComponents(
        parameters=params,
        protocol=protocol,
        manifest=fixed_manifest,
    )


def tf_ssl_lstm_zhaocui_fixed_score(
    observations: tf.Tensor,
    theta: tf.Tensor,
    config: SSLLSTMStaticConfig,
    *,
    evidence_path: str,
    manifest: SSLLSTMZhaoCuiFixedManifest | None = None,
    std_floor: float = 1.0e-4,
) -> tuple[SSLLSTMZhaoCuiFixedScoreResult, SSLLSTMZhaoCuiFixedComponents]:
    """Evaluate the deterministic fixed replay value and manual score."""

    components = make_ssl_lstm_zhaocui_fixed_components(
        theta,
        config,
        evidence_path=evidence_path,
        manifest=manifest,
        std_floor=std_floor,
    )
    result = _fixed_replay_value_and_score(
        tf.convert_to_tensor(observations, dtype=tf.float64),
        components.parameters,
        components.manifest,
    )
    return result, components


def build_ssl_lstm_zhaocui_fixed_value_score_artifact(
    *,
    protocol: SSLLSTMAdapterProtocol,
    manifest: SSLLSTMZhaoCuiFixedManifest,
    log_likelihood: tf.Tensor,
    score: tf.Tensor,
    finite_difference_max_abs_error: float,
    artifact_role: str = "debug_reference",
    compile_mode: str = "eager",
    jit_compile: bool = False,
    device: str = "CPU-hidden debug",
    tf32_enabled: bool = False,
) -> Mapping[str, object]:
    """Build and validate the Phase-2 debug/reference artifact."""

    score_tensor = tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1])
    value_tensor = tf.convert_to_tensor(log_likelihood, dtype=tf.float64)
    score_finite = bool(
        bool(tf.reduce_all(tf.math.is_finite(score_tensor)))
        and bool(tf.math.is_finite(value_tensor))
    )
    artifact = {
        "schema_version": protocol.artifact_schema_version,
        "artifact_role": artifact_role,
        "target_scope": protocol.contract.value_score.target_scope,
        "filter_name": protocol.filter_name,
        "gradient_path": protocol.gradient_path,
        "value_score_authority": protocol.contract.value_score.value_score_authority,
        "compile_mode": compile_mode,
        "jit_compile": bool(jit_compile),
        "device": device,
        "tf32_enabled": bool(tf32_enabled),
        "seed_policy": protocol.contract.seed_policy,
        "branch_or_randomness_policy": protocol.branch_or_randomness_policy,
        "log_likelihood": float(value_tensor),
        "score": [float(item) for item in tf.unstack(score_tensor)],
        "score_finite": score_finite,
        "finite_difference_check": {
            "max_abs_error": float(finite_difference_max_abs_error),
            "role": "promotion_veto_for_adapter_admission",
        },
        "diagnostic_roles": {
            "score_finite": "promotion_veto",
            "finite_difference_check": "promotion_veto_for_adapter_admission",
            "runtime": "explanatory",
            "score_norm": "explanatory",
            "recenter_frame": "explanatory",
        },
        "nonclaims": manifest.nonclaims,
        "zhaocui_fixed_manifest": manifest.as_dict(),
    }
    return validate_ssl_lstm_value_score_artifact(artifact, protocol=protocol)


def _fixed_replay_value_and_score(
    observations: tf.Tensor,
    params: SSLLSTMConstrainedParameters,
    manifest: SSLLSTMZhaoCuiFixedManifest,
) -> SSLLSTMZhaoCuiFixedScoreResult:
    if observations.shape.rank != 2:
        raise ValueError("observations must be a rank-two tensor")
    if observations.shape[0] is None:
        raise ValueError("observations must have a statically known horizon")
    observed_horizon = int(observations.shape[0])
    if observed_horizon <= 0 or observed_horizon > params.config.horizon:
        raise ValueError("observation horizon must be between one and the SSL-LSTM config horizon")
    if observations.shape[1] is not None and int(observations.shape[1]) != params.config.observation_dim:
        raise ValueError("observation dimension does not match SSL-LSTM config")

    sample_count = int(manifest.reference_sample_count)
    parameter_dim = int(params.config.parameter_dim)
    state_dim = int(params.config.augmented_state_dim)
    latent_dim = int(params.config.latent_dim)
    initial_noise = _fixed_initial_noise(manifest, sample_count, state_dim)
    process_noise = _fixed_process_noise(
        manifest,
        max(observed_horizon - 1, 0),
        sample_count,
        latent_dim,
    )
    state = params.initial_mean[tf.newaxis, :] + initial_noise * params.initial_std[tf.newaxis, :]
    state_score = _initial_state_score(params, initial_noise)
    particle_log_values = tf.zeros([sample_count], dtype=tf.float64)
    particle_scores = tf.zeros([sample_count, parameter_dim], dtype=tf.float64)
    filtered_means: list[tf.Tensor] = []

    for step in range(observed_horizon):
        log_value, log_score = _observation_logpdf_and_score(
            observations[step],
            params,
            state,
            state_score,
        )
        particle_log_values = particle_log_values + log_value
        particle_scores = particle_scores + log_score
        step_weights = tf.nn.softmax(particle_log_values)
        filtered_means.append(tf.reduce_sum(step_weights[:, tf.newaxis] * state, axis=0))
        if step + 1 < observed_horizon:
            state, state_score = _transition_replay_step(
                params,
                state,
                state_score,
                process_noise[step],
            )

    log_likelihood = (
        tf.reduce_logsumexp(particle_log_values)
        - tf.math.log(tf.cast(sample_count, tf.float64))
    )
    normalized_weights = tf.nn.softmax(particle_log_values)
    score = tf.reduce_sum(normalized_weights[:, tf.newaxis] * particle_scores, axis=0)
    frame = _recenter_frame(state, normalized_weights, manifest)
    diagnostics = {
        "derivative_method": "analytic_first_order_fixed_replay",
        "reference_sample_count": sample_count,
        "score_norm": tf.linalg.norm(score),
        "particle_log_likelihood_min": tf.reduce_min(particle_log_values),
        "particle_log_likelihood_max": tf.reduce_max(particle_log_values),
        "recenter_frame": frame,
        "manifest": manifest.as_dict(),
    }
    return SSLLSTMZhaoCuiFixedScoreResult(
        log_likelihood=log_likelihood,
        score=score,
        particle_log_likelihoods=particle_log_values,
        normalized_particle_weights=normalized_weights,
        filtered_means=tf.stack(filtered_means, axis=0),
        final_particles=state,
        diagnostics=diagnostics,
    )


def _fixed_initial_noise(
    manifest: SSLLSTMZhaoCuiFixedManifest,
    sample_count: int,
    state_dim: int,
) -> tf.Tensor:
    return tf.random.stateless_normal(
        [sample_count, state_dim],
        seed=tf.constant(manifest.initial_seed, dtype=tf.int32),
        dtype=tf.float64,
    )


def _fixed_process_noise(
    manifest: SSLLSTMZhaoCuiFixedManifest,
    transition_count: int,
    sample_count: int,
    latent_dim: int,
) -> tf.Tensor:
    return tf.random.stateless_normal(
        [transition_count, sample_count, latent_dim],
        seed=tf.constant(manifest.process_seed, dtype=tf.int32),
        dtype=tf.float64,
    )


def _initial_state_score(
    params: SSLLSTMConstrainedParameters,
    initial_noise: tf.Tensor,
) -> tf.Tensor:
    mean_score = tf.broadcast_to(
        tf.transpose(params.d_initial_mean)[tf.newaxis, :, :],
        [
            tf.shape(initial_noise)[0],
            params.config.augmented_state_dim,
            params.config.parameter_dim,
        ],
    )
    std_score_by_parameter = _std_score_from_covariance_derivative(
        params.d_initial_covariance,
        params.initial_std,
    )
    std_score = initial_noise[:, :, tf.newaxis] * tf.transpose(std_score_by_parameter)[
        tf.newaxis, :, :
    ]
    return mean_score + std_score


def _transition_replay_step(
    params: SSLLSTMConstrainedParameters,
    state: tf.Tensor,
    state_score: tf.Tensor,
    process_noise: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    deterministic_next = ssl_lstm_transition(params, state)
    direct_score = tf.transpose(
        ssl_lstm_transition_parameter_derivative(params, state),
        perm=[1, 2, 0],
    )
    propagated_score = tf.transpose(
        ssl_lstm_transition_state_jvp(
            params,
            state,
            tf.transpose(state_score, perm=[2, 0, 1]),
        ),
        perm=[1, 2, 0],
    )
    process_std_score = _std_score_from_covariance_derivative(
        params.d_ukf_innovation_covariance,
        params.process_std,
    )
    process_score_top = process_noise[:, :, tf.newaxis] * tf.transpose(process_std_score)[
        tf.newaxis, :, :
    ]
    process_score = tf.concat(
        [
            process_score_top,
            tf.zeros(
                [
                    tf.shape(state)[0],
                    2 * params.config.hidden_dim,
                    params.config.parameter_dim,
                ],
                dtype=tf.float64,
            ),
        ],
        axis=1,
    )
    next_state = tf.concat(
        [
            deterministic_next[:, : params.config.latent_dim]
            + process_noise * params.process_std[tf.newaxis, :],
            deterministic_next[:, params.config.latent_dim :],
        ],
        axis=1,
    )
    next_score = propagated_score + direct_score + process_score
    return next_state, next_score


def _observation_logpdf_and_score(
    observation: tf.Tensor,
    params: SSLLSTMConstrainedParameters,
    state: tf.Tensor,
    state_score: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = ssl_lstm_observation(params, state)
    residual = observation[tf.newaxis, :] - mean
    variance = tf.square(params.observation_std)
    log_value = -0.5 * tf.reduce_sum(
        tf.math.log(tf.constant(2.0 * 3.14159265358979323846, dtype=tf.float64))
        + tf.math.log(variance)[tf.newaxis, :]
        + tf.square(residual) / variance[tf.newaxis, :],
        axis=1,
    )

    direct_mean_score = tf.transpose(
        ssl_lstm_observation_parameter_derivative(params, state),
        perm=[1, 2, 0],
    )
    propagated_mean_score = tf.transpose(
        ssl_lstm_observation_state_jvp(
            params,
            state,
            tf.transpose(state_score, perm=[2, 0, 1]),
        ),
        perm=[1, 2, 0],
    )
    mean_score = propagated_mean_score + direct_mean_score
    residual_precision = residual / variance[tf.newaxis, :]
    mean_contribution = tf.einsum("rd,rdp->rp", residual_precision, mean_score)

    observation_std_score = _std_score_from_covariance_derivative(
        params.d_observation_covariance,
        params.observation_std,
    )
    std_factor = -1.0 / params.observation_std[tf.newaxis, :] + tf.square(residual) / tf.pow(
        params.observation_std[tf.newaxis, :],
        3,
    )
    std_contribution = tf.matmul(
        std_factor,
        observation_std_score,
        transpose_b=True,
    )
    return log_value, mean_contribution + std_contribution


def _std_score_from_covariance_derivative(
    covariance_derivative: tf.Tensor,
    std: tf.Tensor,
) -> tf.Tensor:
    variance_derivative = tf.linalg.diag_part(covariance_derivative)
    return variance_derivative / (2.0 * std[tf.newaxis, :])


def _recenter_frame(
    particles: tf.Tensor,
    weights: tf.Tensor,
    manifest: SSLLSTMZhaoCuiFixedManifest,
) -> Mapping[str, tf.Tensor]:
    normalized = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(normalized[:, tf.newaxis] * particles, axis=0)
    centered = particles - mean[tf.newaxis, :]
    covariance = (
        tf.einsum("r,ri,rj->ij", normalized, centered, centered)
        + tf.eye(tf.shape(particles)[1], dtype=tf.float64) * float(manifest.recenter_ridge)
    )
    chol = tf.linalg.cholesky(covariance)
    ess = 1.0 / tf.reduce_sum(tf.square(normalized))
    return {
        "weighted_mean": mean,
        "cholesky_diagonal": tf.linalg.diag_part(chol),
        "effective_sample_size": ess,
        "ridge": tf.constant(float(manifest.recenter_ridge), dtype=tf.float64),
    }


__all__ = [
    "SSLLSTMZhaoCuiFixedComponents",
    "SSLLSTMZhaoCuiFixedManifest",
    "SSLLSTMZhaoCuiFixedScoreResult",
    "build_ssl_lstm_zhaocui_fixed_value_score_artifact",
    "make_ssl_lstm_zhaocui_fixed_components",
    "tf_ssl_lstm_zhaocui_fixed_score",
]
