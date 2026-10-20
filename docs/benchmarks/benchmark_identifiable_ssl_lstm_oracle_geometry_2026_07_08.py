"""Identifiable scalar SSL-LSTM oracle geometry diagnostic.

This benchmark checks local quadratic geometry on a deliberately small,
identified complete-data oracle target.  It is not a filtering-likelihood, HMC,
Zhao-Cui source-faithfulness, convergence, or default-readiness result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference import (  # noqa: E402
    LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS,
    LowRankSPDQuadraticGeometryConfig,
    fit_low_rank_spd_quadratic_geometry,
)
from bayesfilter.nonlinear.ssl_lstm_protocol import SSLLSTMStaticConfig  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (  # noqa: E402
    ssl_lstm_observation,
    ssl_lstm_parameter_slices,
    ssl_lstm_transition,
    unpack_ssl_lstm_parameters,
)
from bayesfilter.nonlinear.ssl_lstm_zhaocui_hmc_minimal import (  # noqa: E402
    minimal_ssl_lstm_theta,
)


SCRIPT_NAME = "benchmark_identifiable_ssl_lstm_oracle_geometry_2026_07_08.py"
SCHEMA_VERSION = "identifiable_ssl_lstm.oracle_geometry.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-identifiable-ssl-lstm-oracle-geometry-test-plan-2026-07-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-identifiable-ssl-lstm-oracle-geometry-test-result-2026-07-08.md"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "identifiable_ssl_lstm_oracle_geometry_cpu_hidden_2026-07-08.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "identifiable_ssl_lstm_oracle_geometry_cpu_hidden_2026-07-08.md"
)
FREE_PARAMETER_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
NONCLAIMS = (
    "complete-data oracle geometry diagnostic only",
    "not a filtering-likelihood validity result",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not sampler superiority evidence",
    "not Zhao-Cui source-faithfulness evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
)


@dataclass(frozen=True)
class IdentifiableOracleGeometrySettings:
    """Configuration for the identifiable oracle geometry diagnostic."""

    horizon: int = 200
    seed: tuple[int, int] = (20260708, 1701)
    simulation_noise_scale: float = 1.0
    free_parameter_names: tuple[str, ...] = FREE_PARAMETER_NAMES
    free_parameter_scale: tuple[float, ...] = (0.35, 0.35, 0.35, 0.35)
    prior_scale: float = 4.0
    low_rank_rank: int = 4
    low_rank_sample_count: int = 260
    low_rank_min_samples_per_parameter: int = 5
    low_rank_trust_radius: float = 0.30
    low_rank_pilot_radius: float = 0.08
    low_rank_pilot_direction_count: int = 512
    low_rank_eigenvalue_floor: float = 1.0e-4
    low_rank_max_condition_number: float = 1.0e5
    low_rank_holdout_rmse_abs_tolerance: float = 5.0e-2
    low_rank_holdout_rmse_rel_tolerance: float = 5.0e-3
    low_rank_fit_max_iterations: int = 300
    map_max_iterations: int = 100
    map_tolerance: float = 1.0e-10
    map_gradient_norm_tolerance: float = 1.0e-6
    precision_relative_frobenius_tolerance: float = 0.45
    precision_max_abs_tolerance: float = 6.0
    precision_max_abs_is_veto: bool = False

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "low_rank_rank",
            "low_rank_sample_count",
            "low_rank_min_samples_per_parameter",
            "low_rank_pilot_direction_count",
            "low_rank_fit_max_iterations",
            "map_max_iterations",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "precision_max_abs_is_veto", bool(self.precision_max_abs_is_veto))
        free_names = tuple(str(item) for item in self.free_parameter_names)
        if len(free_names) != 4:
            raise ValueError("this diagnostic is scoped to exactly four free parameters")
        object.__setattr__(self, "free_parameter_names", free_names)
        scale = tuple(float(item) for item in self.free_parameter_scale)
        if len(scale) != len(free_names):
            raise ValueError("free_parameter_scale must match free_parameter_names")
        if any(not np.isfinite(item) or item <= 0.0 for item in scale):
            raise ValueError("free_parameter_scale entries must be positive finite")
        object.__setattr__(self, "free_parameter_scale", scale)
        for name in (
            "simulation_noise_scale",
            "prior_scale",
            "low_rank_trust_radius",
            "low_rank_pilot_radius",
            "low_rank_eigenvalue_floor",
            "low_rank_max_condition_number",
            "low_rank_holdout_rmse_abs_tolerance",
            "low_rank_holdout_rmse_rel_tolerance",
            "precision_relative_frobenius_tolerance",
            "precision_max_abs_tolerance",
            "map_tolerance",
            "map_gradient_norm_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "simulation_noise_scale":
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def dimension(self) -> int:
        return len(self.free_parameter_names)

    @property
    def effective_low_rank_rank(self) -> int:
        return min(self.low_rank_rank, max(self.dimension - 1, 0))

    @property
    def regression_parameter_count(self) -> int:
        return 1 + self.dimension + 1 + self.effective_low_rank_rank

    @property
    def required_finite_samples(self) -> int:
        return self.low_rank_min_samples_per_parameter * self.regression_parameter_count

    def payload(self) -> Mapping[str, Any]:
        return {
            "horizon": self.horizon,
            "seed": self.seed,
            "simulation_noise_scale": self.simulation_noise_scale,
            "free_parameter_names": self.free_parameter_names,
            "free_parameter_scale": self.free_parameter_scale,
            "prior_scale": self.prior_scale,
            "low_rank_rank": self.low_rank_rank,
            "effective_low_rank_rank": self.effective_low_rank_rank,
            "low_rank_sample_count": self.low_rank_sample_count,
            "low_rank_min_samples_per_parameter": self.low_rank_min_samples_per_parameter,
            "low_rank_trust_radius": self.low_rank_trust_radius,
            "low_rank_pilot_radius": self.low_rank_pilot_radius,
            "low_rank_pilot_direction_count": self.low_rank_pilot_direction_count,
            "low_rank_eigenvalue_floor": self.low_rank_eigenvalue_floor,
            "low_rank_max_condition_number": self.low_rank_max_condition_number,
            "low_rank_holdout_rmse_abs_tolerance": self.low_rank_holdout_rmse_abs_tolerance,
            "low_rank_holdout_rmse_rel_tolerance": self.low_rank_holdout_rmse_rel_tolerance,
            "low_rank_fit_max_iterations": self.low_rank_fit_max_iterations,
            "map_max_iterations": self.map_max_iterations,
            "map_tolerance": self.map_tolerance,
            "map_gradient_norm_tolerance": self.map_gradient_norm_tolerance,
            "precision_relative_frobenius_tolerance": self.precision_relative_frobenius_tolerance,
            "precision_max_abs_tolerance": self.precision_max_abs_tolerance,
            "precision_max_abs_is_veto": self.precision_max_abs_is_veto,
            "regression_parameter_count": self.regression_parameter_count,
            "required_finite_samples": self.required_finite_samples,
        }


@dataclass(frozen=True)
class IdentifiableOracleTarget:
    """Complete-data oracle target bound to a simulated SSL-LSTM path."""

    config: SSLLSTMStaticConfig
    base_theta: tf.Tensor
    free_indices: tuple[int, ...]
    free_parameter_names: tuple[str, ...]
    truth_free: tf.Tensor
    scale: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    settings: IdentifiableOracleGeometrySettings

    def full_theta(self, free_values: tf.Tensor) -> tf.Tensor:
        free = tf.reshape(tf.convert_to_tensor(free_values, dtype=tf.float64), [-1])
        return tf.tensor_scatter_nd_update(
            tf.convert_to_tensor(self.base_theta, dtype=tf.float64),
            tf.constant([[index] for index in self.free_indices], dtype=tf.int32),
            free,
        )

    def log_prob(self, free_values: tf.Tensor) -> tf.Tensor:
        free = tf.reshape(tf.convert_to_tensor(free_values, dtype=tf.float64), [-1])
        full = self.full_theta(free_values)
        params = unpack_ssl_lstm_parameters(full, self.config)
        process_std = tf.reshape(params.process_std, [1])
        observation_std = tf.reshape(params.observation_std, [1])
        process_mean = ssl_lstm_transition(params, self.states[:-1])[:, :1]
        process_residual = self.states[1:, :1] - process_mean
        observation_mean = ssl_lstm_observation(params, self.states)
        observation_residual = self.observations - observation_mean
        process_var = tf.square(process_std)
        observation_var = tf.square(observation_std)
        log_two_pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float64)
        process_value = -0.5 * tf.reduce_sum(
            tf.square(process_residual) / process_var
            + log_two_pi
            + tf.math.log(process_var)
        )
        observation_value = -0.5 * tf.reduce_sum(
            tf.square(observation_residual) / observation_var
            + log_two_pi
            + tf.math.log(observation_var)
        )
        delta = tf.reshape(free, [-1]) - self.truth_free
        prior_variance = tf.constant(
            self.settings.prior_scale * self.settings.prior_scale,
            dtype=tf.float64,
        )
        prior_value = -0.5 * tf.reduce_sum(tf.square(delta) / prior_variance)
        return tf.convert_to_tensor(process_value + observation_value + prior_value, dtype=tf.float64)

    def value_and_score(self, free_values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        free = tf.reshape(tf.convert_to_tensor(free_values, dtype=tf.float64), [-1])
        with tf.GradientTape() as tape:
            tape.watch(free)
            value = self.log_prob(free)
        score = tape.gradient(value, free)
        if score is None:
            score = tf.fill(tf.shape(free), tf.constant(np.nan, dtype=tf.float64))
        return value, tf.convert_to_tensor(score, dtype=tf.float64)


def default_settings() -> IdentifiableOracleGeometrySettings:
    return IdentifiableOracleGeometrySettings()


def scalar_ssl_lstm_config(horizon: int) -> SSLLSTMStaticConfig:
    return SSLLSTMStaticConfig(
        horizon=int(horizon),
        latent_dim=1,
        hidden_dim=1,
        observation_dim=1,
    )


def free_parameter_indices(
    config: SSLLSTMStaticConfig,
    names: Sequence[str] = FREE_PARAMETER_NAMES,
) -> tuple[int, ...]:
    by_name = {name: index for index, name in enumerate(config.parameter_names)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown SSL-LSTM parameter names: {missing}")
    return tuple(int(by_name[name]) for name in names)


def build_identifiable_oracle_target(
    settings: IdentifiableOracleGeometrySettings | None = None,
) -> IdentifiableOracleTarget:
    cfg = default_settings() if settings is None else settings
    config = scalar_ssl_lstm_config(cfg.horizon)
    base_theta = tf.reshape(tf.convert_to_tensor(minimal_ssl_lstm_theta(), dtype=tf.float64), [-1])
    if int(base_theta.shape[0]) != config.parameter_dim:
        raise ValueError("minimal scalar theta does not match scalar SSL-LSTM config")
    free_indices = free_parameter_indices(config, cfg.free_parameter_names)
    truth_free = tf.gather(base_theta, list(free_indices))
    states, observations = simulate_oracle_path(config, base_theta, cfg)
    return IdentifiableOracleTarget(
        config=config,
        base_theta=base_theta,
        free_indices=free_indices,
        free_parameter_names=cfg.free_parameter_names,
        truth_free=tf.convert_to_tensor(truth_free, dtype=tf.float64),
        scale=tf.constant(cfg.free_parameter_scale, dtype=tf.float64),
        states=states,
        observations=observations,
        settings=cfg,
    )


def simulate_oracle_path(
    config: SSLLSTMStaticConfig,
    theta: tf.Tensor,
    settings: IdentifiableOracleGeometrySettings,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Simulate a deterministic complete-data oracle path from the scalar SSL-LSTM."""

    params = unpack_ssl_lstm_parameters(theta, config)
    seed = tf.constant(settings.seed, dtype=tf.int32)
    noise = tf.random.stateless_normal(
        [int(settings.horizon), 2],
        seed=seed,
        dtype=tf.float64,
    )
    state = tf.reshape(params.initial_mean, [1, config.augmented_state_dim])
    states = [tf.reshape(state[0], [config.augmented_state_dim])]
    scale = tf.constant(float(settings.simulation_noise_scale), dtype=tf.float64)
    for time_index in range(1, int(settings.horizon)):
        transition_mean = ssl_lstm_transition(params, state)
        latent = transition_mean[:, :1] + scale * params.process_std[tf.newaxis, :] * noise[time_index : time_index + 1, :1]
        state = tf.concat([latent, transition_mean[:, 1:]], axis=1)
        states.append(tf.reshape(state[0], [config.augmented_state_dim]))
    state_tensor = tf.stack(states, axis=0)
    observation_mean = ssl_lstm_observation(params, state_tensor)
    observations = (
        observation_mean
        + scale * params.observation_std[tf.newaxis, :] * noise[:, 1:2]
    )
    return (
        tf.convert_to_tensor(state_tensor, dtype=tf.float64),
        tf.convert_to_tensor(observations, dtype=tf.float64),
    )


def estimate_map_center(
    target: IdentifiableOracleTarget,
    settings: IdentifiableOracleGeometrySettings | None = None,
) -> tuple[tf.Tensor, Mapping[str, Any]]:
    cfg = target.settings if settings is None else settings
    initial = tf.convert_to_tensor(target.truth_free, dtype=tf.float64)

    def objective_and_grad(position: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(position, dtype=tf.float64), [-1])
        with tf.GradientTape() as tape:
            tape.watch(position)
            objective = -target.log_prob(position)
        gradient = tape.gradient(objective, position)
        if gradient is None:
            gradient = tf.fill(tf.shape(position), tf.constant(np.nan, dtype=tf.float64))
        return tf.convert_to_tensor(objective, dtype=tf.float64), tf.convert_to_tensor(gradient, dtype=tf.float64)

    optimizer = tfp.optimizer.lbfgs_minimize(
        objective_and_grad,
        initial_position=initial,
        max_iterations=int(cfg.map_max_iterations),
        tolerance=tf.constant(float(cfg.map_tolerance), dtype=tf.float64),
        parallel_iterations=1,
    )
    position = tf.reshape(tf.convert_to_tensor(optimizer.position, dtype=tf.float64), [-1])
    value, score = target.value_and_score(position)
    score_scaled = np.asarray(score.numpy(), dtype=float) * np.asarray(target.scale.numpy(), dtype=float)
    diagnostics = {
        "source": "tfp_lbfgs_minimize_negative_log_prob",
        "initial_position_role": "truth_free_parameters",
        "center_position_role": "map_candidate",
        "converged": bool(optimizer.converged.numpy()),
        "failed": bool(optimizer.failed.numpy()),
        "iterations": int(optimizer.num_iterations.numpy()),
        "objective": float(optimizer.objective_value.numpy()),
        "log_prob": float(value.numpy()),
        "score_norm": float(np.linalg.norm(score_scaled)),
        "score_norm_tolerance": float(cfg.map_gradient_norm_tolerance),
        "gradient_norm_passed": bool(
            np.linalg.norm(score_scaled) <= float(cfg.map_gradient_norm_tolerance)
        ),
        "delta_from_truth_norm": float(
            np.linalg.norm(
                (np.asarray(position.numpy(), dtype=float) - np.asarray(target.truth_free.numpy(), dtype=float))
                / np.asarray(target.scale.numpy(), dtype=float)
            )
        ),
    }
    return position, json_ready(diagnostics)


def dense_negative_hessian(
    target: IdentifiableOracleTarget,
    center: tf.Tensor,
) -> np.ndarray:
    center = tf.reshape(tf.convert_to_tensor(center, dtype=tf.float64), [-1])
    with tf.GradientTape(persistent=True) as outer:
        outer.watch(center)
        with tf.GradientTape() as inner:
            inner.watch(center)
            value = target.log_prob(center)
        gradient = inner.gradient(value, center)
    if gradient is None:
        raise RuntimeError("dense Hessian baseline failed: gradient is disconnected")
    hessian = outer.jacobian(gradient, center, experimental_use_pfor=False)
    del outer
    if hessian is None:
        raise RuntimeError("dense Hessian baseline failed: Hessian is disconnected")
    negative = -np.asarray(tf.convert_to_tensor(hessian, dtype=tf.float64).numpy(), dtype=float)
    return 0.5 * (negative + negative.T)


def dense_negative_hessian_at_truth(target: IdentifiableOracleTarget) -> np.ndarray:
    return dense_negative_hessian(target, target.truth_free)


def run_identifiable_geometry_diagnostic(
    settings: IdentifiableOracleGeometrySettings | None = None,
) -> Mapping[str, Any]:
    cfg = default_settings() if settings is None else settings
    start = time.perf_counter()
    target = build_identifiable_oracle_target(cfg)
    center, map_diagnostics = estimate_map_center(target, cfg)
    center_value, center_score = target.value_and_score(center)
    dense_precision = dense_negative_hessian(target, center)
    scale = np.asarray(target.scale.numpy(), dtype=float)
    dense_precision_whitened = (
        scale[:, np.newaxis] * dense_precision * scale[np.newaxis, :]
    )
    dense_summary = eigen_summary(dense_precision)
    dense_whitened_summary = eigen_summary(dense_precision_whitened)

    low_rank_config = LowRankSPDQuadraticGeometryConfig(
        rank=cfg.low_rank_rank,
        sample_count=cfg.low_rank_sample_count,
        min_samples_per_parameter=cfg.low_rank_min_samples_per_parameter,
        trust_radius=cfg.low_rank_trust_radius,
        pilot_radius=cfg.low_rank_pilot_radius,
        pilot_direction_count=cfg.low_rank_pilot_direction_count,
        eigenvalue_floor=cfg.low_rank_eigenvalue_floor,
        max_condition_number=cfg.low_rank_max_condition_number,
        holdout_rmse_abs_tolerance=cfg.low_rank_holdout_rmse_abs_tolerance,
        holdout_rmse_rel_tolerance=cfg.low_rank_holdout_rmse_rel_tolerance,
        fit_max_iterations=cfg.low_rank_fit_max_iterations,
        seed=cfg.seed,
    )
    geometry = fit_low_rank_spd_quadratic_geometry(
        target.value_and_score,
        center,
        scale=target.scale,
        config=low_rank_config,
    )
    if geometry.precision is not None:
        fitted_precision = np.asarray(geometry.precision, dtype=float)
        precision_delta = fitted_precision - dense_precision_whitened
        relative_frobenius = float(
            np.linalg.norm(precision_delta, ord="fro")
            / max(1.0e-12, np.linalg.norm(dense_precision_whitened, ord="fro"))
        )
        max_abs = float(np.max(np.abs(precision_delta)))
    else:
        fitted_precision = None
        relative_frobenius = None
        max_abs = None

    sample_ratio_passed = bool(
        geometry.payload()["diagnostics"].get("finite_sample_count", 0)
        >= cfg.required_finite_samples
    )
    dense_spd_passed = bool(dense_whitened_summary["positive"])
    map_candidate_passed = bool(
        map_diagnostics["gradient_norm_passed"] and not map_diagnostics["failed"]
    )
    relative_agreement_passed = bool(
        geometry.accepted
        and relative_frobenius is not None
        and relative_frobenius <= cfg.precision_relative_frobenius_tolerance
    )
    max_abs_screen_passed = bool(
        geometry.accepted
        and max_abs is not None
        and max_abs <= cfg.precision_max_abs_tolerance
    )
    agreement_passed = bool(
        relative_agreement_passed
        and (max_abs_screen_passed or not cfg.precision_max_abs_is_veto)
    )
    geometry_sanity_passed = bool(
        map_candidate_passed
        and
        dense_spd_passed
        and sample_ratio_passed
        and geometry.accepted
        and agreement_passed
    )
    vetoes = []
    if not map_candidate_passed:
        vetoes.append("map_candidate_not_accepted")
    if not dense_spd_passed:
        vetoes.append("dense_hessian_not_spd")
    if not sample_ratio_passed:
        vetoes.append("finite_sample_ratio_failed")
    if not geometry.accepted:
        vetoes.append(f"low_rank_geometry_{geometry.status}")
    if geometry.accepted and not relative_agreement_passed:
        vetoes.append("low_rank_dense_precision_mismatch")
    if geometry.accepted and cfg.precision_max_abs_is_veto and not max_abs_screen_passed:
        vetoes.append("low_rank_dense_precision_max_abs_mismatch")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_identifiable_oracle_geometry_diagnostic",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": "identifiable_ssl_lstm:complete_data_oracle_geometry:phase1",
        "settings": cfg.payload(),
        "environment": environment_payload(),
        "git": git_payload(),
        "target": target_payload(target),
        "center": {
            "log_prob": float(center_value.numpy()),
            "score_norm": float(np.linalg.norm(center_score.numpy() * scale)),
            "coordinate_system": "free_parameters_center_plus_scale_times_z",
            "position_role": "map_candidate",
            "free_parameter_values": center,
            "map_diagnostics": map_diagnostics,
        },
        "dense_baseline": {
            "source": "tensorflow_autodiff_negative_hessian_at_map_candidate",
            "coordinate_system": "original_free_parameters",
            "precision_eigen_summary": dense_summary,
            "whitened_coordinate_precision_eigen_summary": dense_whitened_summary,
        },
        "low_rank_geometry": geometry.payload(include_arrays=False),
        "comparison": {
            "coordinate_system": "whitened_center_plus_scale_times_z",
            "relative_frobenius_error": relative_frobenius,
            "max_abs_error": max_abs,
            "relative_frobenius_tolerance": cfg.precision_relative_frobenius_tolerance,
            "max_abs_tolerance": cfg.precision_max_abs_tolerance,
            "max_abs_is_veto": cfg.precision_max_abs_is_veto,
            "relative_agreement_passed": relative_agreement_passed,
            "max_abs_screen_passed": max_abs_screen_passed,
            "agreement_passed": agreement_passed,
            "max_abs_role": "explanatory_only",
        },
        "decision": {
            "geometry_sanity_passed": geometry_sanity_passed,
            "map_candidate_passed": map_candidate_passed,
            "dense_spd_passed": dense_spd_passed,
            "sample_ratio_passed": sample_ratio_passed,
            "vetoes": tuple(vetoes),
            "viable_for_next_step": geometry_sanity_passed,
            "next_justified_action": (
                "try the same geometry initializer on a filtering likelihood"
                if geometry_sanity_passed
                else "repair oracle geometry before any filtering-likelihood or HMC run"
            ),
        },
        "inference_status": {
            "hard_veto_screen": "passed" if not vetoes else "failed",
            "statistically_supported_ranking": "none; single diagnostic target",
            "descriptive_only_differences": (
                "runtime, condition numbers, and residual magnitudes are explanatory only"
            ),
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "filtering-likelihood geometry and then bounded HMC diagnostics"
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 python "
                "docs/benchmarks/benchmark_identifiable_ssl_lstm_oracle_geometry_2026_07_08.py"
            ),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "data_version": "stateless_simulated_scalar_ssl_lstm_complete_data_oracle_v1",
            "random_seeds": cfg.seed,
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "nonclaims": NONCLAIMS + tuple(LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS),
    }
    return json_ready(payload)


def target_payload(target: IdentifiableOracleTarget) -> Mapping[str, Any]:
    slices = ssl_lstm_parameter_slices(target.config)
    del slices
    return {
        "model": "scalar_ssl_lstm_complete_data_oracle",
        "horizon": int(target.config.horizon),
        "latent_dim": int(target.config.latent_dim),
        "hidden_dim": int(target.config.hidden_dim),
        "observation_dim": int(target.config.observation_dim),
        "augmented_state_dim": int(target.config.augmented_state_dim),
        "full_parameter_dim": int(target.config.parameter_dim),
        "free_parameter_dim": len(target.free_indices),
        "free_parameter_names": target.free_parameter_names,
        "free_parameter_indices": target.free_indices,
        "fixed_parameter_count": int(target.config.parameter_dim - len(target.free_indices)),
        "truth_free_parameters": target.truth_free,
        "complete_data_path_shape": tuple(int(item) for item in target.states.shape),
        "observation_shape": tuple(int(item) for item in target.observations.shape),
        "likelihood_terms": (
            "latent_process_density_given_complete_augmented_path",
            "observation_density_given_complete_augmented_path",
            "weak_gaussian_prior_on_free_parameters",
        ),
    }


def eigen_summary(matrix: Any) -> Mapping[str, Any]:
    values = np.linalg.eigvalsh(np.asarray(matrix, dtype=float))
    positive = bool(np.all(values > 0.0))
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "condition_number": (
            float(np.max(values) / np.min(values)) if positive else None
        ),
        "positive": positive,
        "nonpositive_count": int(np.sum(values <= 0.0)),
        "values": tuple(float(value) for value in values),
    }


def environment_payload() -> Mapping[str, Any]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "tf_version": tf.__version__,
        "tf_physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "trust_basis": "cpu_hidden_debug_reference_exception",
    }


def git_payload() -> Mapping[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance best effort.
        commit = "unavailable"
    return {"commit": commit}


def write_artifacts(
    payload: Mapping[str, Any],
    *,
    json_path: Path = DEFAULT_JSON_PATH,
    markdown_path: Path = DEFAULT_MARKDOWN_PATH,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    comparison = payload["comparison"]
    dense = payload["dense_baseline"]["whitened_coordinate_precision_eigen_summary"]
    low_rank = payload["low_rank_geometry"]
    lines = [
        "# Identifiable SSL-LSTM Oracle Geometry Diagnostic - 2026-07-08",
        "",
        "## Decision",
        "",
        f"- geometry_sanity_passed: `{decision['geometry_sanity_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Primary Comparison",
        "",
        f"- dense whitened precision min/max: `{dense['min']}` / `{dense['max']}`",
        f"- low-rank accepted/status: `{low_rank['accepted']}` / `{low_rank['status']}`",
        f"- relative Frobenius error: `{comparison['relative_frobenius_error']}`",
        f"- max absolute error: `{comparison['max_abs_error']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    lines.append("")
    return "\n".join(lines)


def json_ready(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return json_ready(value.numpy())
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--horizon", type=int, default=default_settings().horizon)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=default_settings().low_rank_sample_count,
    )
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=default_settings().low_rank_eigenvalue_floor,
    )
    parser.add_argument(
        "--max-condition-number",
        type=float,
        default=default_settings().low_rank_max_condition_number,
    )
    parser.add_argument(
        "--trust-radius",
        type=float,
        default=default_settings().low_rank_trust_radius,
    )
    parser.add_argument(
        "--pilot-direction-count",
        type=int,
        default=default_settings().low_rank_pilot_direction_count,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    settings = replace(
        default_settings(),
        horizon=int(args.horizon),
        low_rank_sample_count=int(args.sample_count),
        low_rank_eigenvalue_floor=float(args.eigenvalue_floor),
        low_rank_max_condition_number=float(args.max_condition_number),
        low_rank_trust_radius=float(args.trust_radius),
        low_rank_pilot_direction_count=int(args.pilot_direction_count),
    )
    payload = run_identifiable_geometry_diagnostic(settings)
    write_artifacts(
        payload,
        json_path=Path(args.json_path),
        markdown_path=Path(args.markdown_path),
    )
    print(json.dumps(payload["decision"], sort_keys=True))
    return 0 if payload["decision"]["geometry_sanity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
