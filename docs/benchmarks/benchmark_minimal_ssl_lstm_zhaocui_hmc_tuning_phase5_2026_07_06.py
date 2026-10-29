"""Phase 5 minimal SSL-LSTM HMC staged tuning diagnostic.

This harness runs the smallest CPU-hidden BayesFilter-owned staged HMC tuning
diagnostic on the minimal ``zhaocui_fixed`` target.  It treats a structured
non-promoting tuning result as useful diagnostic evidence, not as sampler
readiness or posterior correctness.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference import (  # noqa: E402
    HMCKernelTuningConfig,
    LowRankSPDQuadraticGeometryConfig,
    fit_low_rank_spd_quadratic_geometry,
    tune_hmc_kernel,
)
from bayesfilter.nonlinear.ssl_lstm_zhaocui_hmc_minimal import (  # noqa: E402
    MinimalZhaoCuiHMCTargetAdapter,
    initial_minimal_ssl_lstm_hmc_state,
    minimal_ssl_lstm_fixture_payload,
)


SCRIPT_NAME = "benchmark_minimal_ssl_lstm_zhaocui_hmc_tuning_phase5_2026_07_06.py"
PHASE = "PHASE5"
SCHEMA_VERSION = "minimal_ssl_lstm_zhaocui_hmc_validity.phase5_tuning_mass.v1"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-validity-gaps-"
    "phase5-tuning-mass-ladder-subplan-2026-07-06.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-minimal-ssl-lstm-zhaocui-hmc-validity-gaps-"
    "phase5-tuning-mass-ladder-result-2026-07-06.md"
)
PHASE3_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase3_longer_gpu_xla_2026-07-06.json"
)
PHASE4_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase4_divergence_telemetry_cpu_hidden_2026-07-06.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase5_tuning_mass_cpu_hidden_2026-07-06.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase5_tuning_mass_cpu_hidden_2026-07-06.md"
)
DEFAULT_TUNING_DIR = (
    ROOT
    / "docs/benchmarks/"
    "minimal_ssl_lstm_zhaocui_hmc_validity_phase5_tuning_public_artifacts_2026-07-06"
)
NONCLAIMS = (
    "Phase 5 staged tuning diagnostic only",
    "structured tuning handoff is non-promoting unless later validation passes",
    "native divergence telemetry remains unavailable from Phase 4 unless separately repaired",
    "missing native divergence telemetry is not zero divergences",
    "acceptance/runtime/stage status diagnostics are descriptive only",
    "no posterior correctness claim",
    "no broad HMC convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no production-readiness claim",
    "no public API or package readiness claim",
    "no source-faithful Zhao-Cui parity claim",
    "no LEDH evidence",
)
VALID_STRUCTURED_FINAL_STATUSES = {
    "passed",
    "budget_exhausted",
    "repair_or_retry",
    "hard_veto",
}
PROMOTING_FINAL_STATUSES = {"passed"}
NONPROMOTING_FINAL_STATUSES = {"budget_exhausted", "repair_or_retry"}
STRUCTURED_BLOCKER_FINAL_STATUSES = {"hard_veto"}


@dataclass(frozen=True)
class Phase5TuningSettings:
    preset: str = "smoke"
    seed: tuple[int, int] = (20260706, 6501)
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    bootstrap_max_repairs: int = 1
    max_attempts: int = 1
    terminal_phase6_repair_extra_attempts: int = 0
    max_leapfrog_steps: int = 8
    chain_execution_mode: str = "tf_function"
    use_xla: bool = False
    public_timeout_budget_s: float = 90.0
    allow_structured_tuning_hard_veto_artifact: bool = True
    initial_offset_scale: float = 1.0e-3
    initial_covariance_scale: float = 0.01
    initial_geometry_strategy: str = "map_candidate_hessian"
    map_candidate_max_iterations: int = 25
    map_candidate_tolerance: float = 1.0e-7
    map_candidate_gradient_norm_tolerance: float = 1.0e-5
    low_rank_quadratic_rank: int = 4
    low_rank_quadratic_sample_count: int = 180
    low_rank_quadratic_min_samples_per_parameter: int = 5
    low_rank_quadratic_trust_radius: float = 1.0
    low_rank_quadratic_pilot_radius: float = 0.15
    low_rank_quadratic_eigenvalue_floor: float = 1.0
    low_rank_quadratic_max_condition_number: float = 1.0e3
    low_rank_quadratic_holdout_rmse_abs_tolerance: float = 0.25
    low_rank_quadratic_holdout_rmse_rel_tolerance: float = 0.10
    geometry_scaling_c: float = 0.25
    stability_guard: float = 0.5
    covariance_jitter: float = 1.0e-9
    eigenvalue_floor: float = 0.04
    max_condition_number: float = 1.0e6
    allow_geometry_fallback: bool = False

    def __post_init__(self) -> None:
        preset = str(self.preset)
        if preset not in {"smoke", "diagnostic"}:
            raise ValueError("Phase 5 permits only smoke or diagnostic presets")
        object.__setattr__(self, "preset", preset)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        for name in ("bootstrap_max_repairs", "max_attempts", "max_leapfrog_steps"):
            value = int(getattr(self, name))
            if value <= 0 and name != "bootstrap_max_repairs":
                raise ValueError(f"{name} must be positive")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if int(self.max_attempts) > 2:
            raise ValueError("Phase 5 diagnostic max_attempts is capped at 2")
        terminal_extra = int(self.terminal_phase6_repair_extra_attempts)
        if terminal_extra < 0 or terminal_extra > 1:
            raise ValueError("terminal_phase6_repair_extra_attempts must be 0 or 1")
        object.__setattr__(self, "terminal_phase6_repair_extra_attempts", terminal_extra)
        for name in (
            "target_accept_prob",
            "public_timeout_budget_s",
            "initial_offset_scale",
            "initial_covariance_scale",
            "geometry_scaling_c",
            "stability_guard",
            "covariance_jitter",
            "eigenvalue_floor",
            "max_condition_number",
            "low_rank_quadratic_trust_radius",
            "low_rank_quadratic_pilot_radius",
            "low_rank_quadratic_eigenvalue_floor",
            "low_rank_quadratic_max_condition_number",
            "low_rank_quadratic_holdout_rmse_abs_tolerance",
            "low_rank_quadratic_holdout_rmse_rel_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "covariance_jitter":
                if value < 0.0:
                    raise ValueError(f"{name} must be non-negative")
            elif value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "chain_execution_mode", str(self.chain_execution_mode))
        strategy = str(self.initial_geometry_strategy)
        if strategy not in {
            "map_candidate_hessian",
            "initial_covariance",
            "low_rank_spd_quadratic",
        }:
            raise ValueError(
                "initial_geometry_strategy must be map_candidate_hessian, "
                "initial_covariance, or low_rank_spd_quadratic"
            )
        object.__setattr__(self, "initial_geometry_strategy", strategy)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        object.__setattr__(self, "allow_geometry_fallback", bool(self.allow_geometry_fallback))
        for name in (
            "map_candidate_max_iterations",
            "low_rank_quadratic_rank",
            "low_rank_quadratic_sample_count",
            "low_rank_quadratic_min_samples_per_parameter",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "acceptance_band", validate_band(self.acceptance_band))
        object.__setattr__(self, "repair_band", validate_band(self.repair_band))

    def payload(self) -> Mapping[str, Any]:
        return {
            "preset": self.preset,
            "seed": self.seed,
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "bootstrap_max_repairs": self.bootstrap_max_repairs,
            "max_attempts": self.max_attempts,
            "terminal_phase6_repair_extra_attempts": self.terminal_phase6_repair_extra_attempts,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "chain_execution_mode": self.chain_execution_mode,
            "use_xla": self.use_xla,
            "jit_compile": self.use_xla,
            "public_timeout_budget_s": self.public_timeout_budget_s,
            "allow_structured_tuning_hard_veto_artifact": (
                self.allow_structured_tuning_hard_veto_artifact
            ),
            "initial_offset_scale": self.initial_offset_scale,
            "initial_covariance_scale": self.initial_covariance_scale,
            "initial_geometry_strategy": self.initial_geometry_strategy,
            "map_candidate_max_iterations": self.map_candidate_max_iterations,
            "map_candidate_tolerance": self.map_candidate_tolerance,
            "map_candidate_gradient_norm_tolerance": (
                self.map_candidate_gradient_norm_tolerance
            ),
            "low_rank_quadratic_rank": self.low_rank_quadratic_rank,
            "low_rank_quadratic_sample_count": self.low_rank_quadratic_sample_count,
            "low_rank_quadratic_min_samples_per_parameter": (
                self.low_rank_quadratic_min_samples_per_parameter
            ),
            "low_rank_quadratic_trust_radius": self.low_rank_quadratic_trust_radius,
            "low_rank_quadratic_pilot_radius": self.low_rank_quadratic_pilot_radius,
            "low_rank_quadratic_eigenvalue_floor": (
                self.low_rank_quadratic_eigenvalue_floor
            ),
            "low_rank_quadratic_max_condition_number": (
                self.low_rank_quadratic_max_condition_number
            ),
            "low_rank_quadratic_holdout_rmse_abs_tolerance": (
                self.low_rank_quadratic_holdout_rmse_abs_tolerance
            ),
            "low_rank_quadratic_holdout_rmse_rel_tolerance": (
                self.low_rank_quadratic_holdout_rmse_rel_tolerance
            ),
            "geometry_scaling_c": self.geometry_scaling_c,
            "stability_guard": self.stability_guard,
            "covariance_jitter": self.covariance_jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "allow_geometry_fallback": self.allow_geometry_fallback,
            "artifact_mode": "cpu_hidden_staged_tuning_smoke_diagnostic",
        }


def reviewed_phase5_settings() -> Phase5TuningSettings:
    return Phase5TuningSettings()


def validate_band(value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError("band must contain two values")
    lower, upper = (float(value[0]), float(value[1]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("band values must be finite")
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("band values must satisfy 0 <= lower <= upper <= 1")
    return lower, upper


def load_phase3_baseline(path: Path = PHASE3_JSON_PATH) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    continuation_vetoes = tuple(str(item) for item in payload.get("continuation_vetoes", ()))
    promotion_vetoes = tuple(str(item) for item in payload.get("promotion_vetoes", ()))
    return {
        "path": str(path.relative_to(ROOT)),
        "status": payload.get("status"),
        "continuation_vetoes": continuation_vetoes,
        "promotion_screen_status": payload.get("promotion_screen_status"),
        "promotion_vetoes": promotion_vetoes,
        "phase3_preconditions_met": (
            payload.get("status") == "passed"
            and continuation_vetoes == tuple()
            and {"split_rhat_threshold_failed", "ess_threshold_failed", "native_divergence_telemetry_not_exposed"}.issubset(
                set(promotion_vetoes)
            )
        ),
    }


def load_phase4_baseline(path: Path = PHASE4_JSON_PATH) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.relative_to(ROOT)),
        "status": payload.get("status"),
        "hard_vetoes": tuple(str(item) for item in payload.get("hard_vetoes", ())),
        "native_divergence_telemetry_status": payload.get(
            "native_divergence_telemetry_status"
        ),
        "phase4_preconditions_met": (
            payload.get("status") == "passed"
            and tuple(payload.get("hard_vetoes", ())) == tuple()
            and payload.get("native_divergence_telemetry_status")
            == "native_divergence_not_exposed_by_kernel"
        ),
    }


def phase5_tuning_config(
    settings: Phase5TuningSettings,
    *,
    target_scope: str,
    geometry_position_role: str = "initial_position",
    negative_hessian_source: str = "negative_hessian",
) -> HMCKernelTuningConfig:
    factory = (
        HMCKernelTuningConfig.smoke
        if settings.preset == "smoke"
        else HMCKernelTuningConfig.diagnostic
    )
    return factory(
        target_scope=target_scope,
        seed=settings.seed,
        target_accept_prob=settings.target_accept_prob,
        acceptance_band=settings.acceptance_band,
        repair_band=settings.repair_band,
        bootstrap_max_repairs=settings.bootstrap_max_repairs,
        max_attempts=settings.max_attempts,
        terminal_phase6_repair_extra_attempts=settings.terminal_phase6_repair_extra_attempts,
        max_leapfrog_steps=settings.max_leapfrog_steps,
        chain_execution_mode=settings.chain_execution_mode,
        use_xla=settings.use_xla,
        public_timeout_budget_s=settings.public_timeout_budget_s,
        geometry_scaling_c=settings.geometry_scaling_c,
        stability_guard=settings.stability_guard,
        covariance_jitter=settings.covariance_jitter,
        eigenvalue_floor=settings.eigenvalue_floor,
        max_condition_number=settings.max_condition_number,
        allow_geometry_fallback=settings.allow_geometry_fallback,
        geometry_position_role=geometry_position_role,
        negative_hessian_source=negative_hessian_source,
        source="bayesfilter.minimal_ssl_lstm.phase5_tuning_mass",
    )


def initial_covariance(settings: Phase5TuningSettings, *, dimension: int) -> np.ndarray:
    scale = float(settings.initial_covariance_scale)
    return (scale * scale) * np.eye(int(dimension), dtype=float)


def initial_geometry_inputs(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    initial_position: tf.Tensor,
    settings: Phase5TuningSettings,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    fallback_covariance = initial_covariance(
        settings,
        dimension=int(adapter.parameter_dim),
    )
    fallback = {
        "initial_position": np.asarray(initial_position.numpy(), dtype=float),
        "negative_hessian": None,
        "initial_covariance": fallback_covariance,
    }
    if settings.initial_geometry_strategy == "initial_covariance":
        return fallback, {
            "schema": "minimal_ssl_lstm.phase5_initial_geometry.v1",
            "selected_geometry_hint": "initial_covariance",
            "strategy_requested": settings.initial_geometry_strategy,
            "fallback_used": False,
            "position_role": "initial_position",
            "covariance_source": "initial_covariance",
            "nonclaims": (
                "initial covariance diagnostic only",
                "no MAP quality claim",
                "no posterior convergence claim",
            ),
        }
    if settings.initial_geometry_strategy == "low_rank_spd_quadratic":
        low_rank_inputs, low_rank_diagnostics = low_rank_spd_quadratic_geometry_inputs(
            adapter,
            initial_position,
            settings,
        )
        if low_rank_inputs is not None:
            return low_rank_inputs, low_rank_diagnostics
        fallback_inputs, fallback_diagnostics = map_candidate_or_initial_geometry_inputs(
            adapter,
            initial_position,
            settings,
            fallback=fallback,
        )
        return fallback_inputs, {
            **fallback_diagnostics,
            "low_rank_spd_quadratic_attempt": low_rank_diagnostics,
            "fallback_used": True,
            "fallback_reason": (
                "low_rank_spd_quadratic_"
                f"{low_rank_diagnostics.get('status', 'rejected')}"
            ),
        }

    return map_candidate_or_initial_geometry_inputs(
        adapter,
        initial_position,
        settings,
        fallback=fallback,
    )


def map_candidate_or_initial_geometry_inputs(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    initial_position: tf.Tensor,
    settings: Phase5TuningSettings,
    *,
    fallback: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:

    try:
        map_diagnostics = map_candidate_negative_hessian(
            adapter,
            initial_position,
            settings,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic fallback must be explicit.
        return fallback, {
            "schema": "minimal_ssl_lstm.phase5_initial_geometry.v1",
            "selected_geometry_hint": "initial_covariance",
            "strategy_requested": settings.initial_geometry_strategy,
            "fallback_used": True,
            "fallback_reason": "map_candidate_hessian_exception",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "position_role": "initial_position",
            "covariance_source": "initial_covariance",
            "nonclaims": (
                "fallback covariance diagnostic only",
                "no MAP quality claim",
                "no posterior convergence claim",
            ),
        }

    if map_diagnostics["usable"] is not True:
        local_diagnostics = local_initial_negative_hessian(
            adapter,
            initial_position,
            map_status=str(map_diagnostics["status"]),
        )
        if local_diagnostics["usable"] is True:
            return {
                "initial_position": np.asarray(initial_position.numpy(), dtype=float),
                "negative_hessian": np.asarray(
                    local_diagnostics["negative_hessian"],
                    dtype=float,
                ),
                "initial_covariance": None,
            }, {
                **{
                    key: value
                    for key, value in local_diagnostics.items()
                    if key != "negative_hessian"
                },
                "selected_geometry_hint": "negative_hessian",
                "fallback_used": True,
                "fallback_reason": str(map_diagnostics["status"]),
                "position_role": "initial_position",
                "covariance_source": "regularized_negative_hessian_at_initial_position",
                "map_candidate_status": str(map_diagnostics["status"]),
                "map_candidate_score_norm": map_diagnostics.get(
                    "map_candidate_score_norm"
                ),
                "map_candidate_log_prob": map_diagnostics.get("map_candidate_log_prob"),
            }
        return fallback, {
            **{
                key: value
                for key, value in map_diagnostics.items()
                if key not in {"negative_hessian", "position"}
            },
            "selected_geometry_hint": "initial_covariance",
            "fallback_used": True,
            "fallback_reason": map_diagnostics["status"],
            "position_role": "initial_position",
            "covariance_source": "initial_covariance",
        }
    return {
        "initial_position": np.asarray(map_diagnostics["position"], dtype=float),
        "negative_hessian": np.asarray(map_diagnostics["negative_hessian"], dtype=float),
        "initial_covariance": None,
    }, {
        **{
            key: value
            for key, value in map_diagnostics.items()
            if key not in {"negative_hessian", "position"}
        },
        "selected_geometry_hint": "negative_hessian",
        "fallback_used": False,
        "position_role": "map_candidate",
        "covariance_source": "regularized_negative_hessian_at_map_candidate",
    }


def low_rank_spd_quadratic_geometry_inputs(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    initial_position: tf.Tensor,
    settings: Phase5TuningSettings,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any]]:
    center = np.asarray(tf.reshape(initial_position, [-1]).numpy(), dtype=float)
    scale = np.full(int(adapter.parameter_dim), float(adapter.prior_scale), dtype=float)
    config = LowRankSPDQuadraticGeometryConfig(
        rank=int(settings.low_rank_quadratic_rank),
        sample_count=int(settings.low_rank_quadratic_sample_count),
        min_samples_per_parameter=int(
            settings.low_rank_quadratic_min_samples_per_parameter
        ),
        trust_radius=float(settings.low_rank_quadratic_trust_radius),
        pilot_radius=float(settings.low_rank_quadratic_pilot_radius),
        eigenvalue_floor=float(settings.low_rank_quadratic_eigenvalue_floor),
        max_condition_number=float(settings.low_rank_quadratic_max_condition_number),
        holdout_rmse_abs_tolerance=float(
            settings.low_rank_quadratic_holdout_rmse_abs_tolerance
        ),
        holdout_rmse_rel_tolerance=float(
            settings.low_rank_quadratic_holdout_rmse_rel_tolerance
        ),
        seed=settings.seed,
    )

    result = fit_low_rank_spd_quadratic_geometry(
        adapter.log_prob_and_grad,
        center,
        scale=scale,
        config=config,
    )
    payload = result.payload(include_arrays=False)
    diagnostics = {
        "schema": "minimal_ssl_lstm.phase5_initial_geometry.v1",
        "strategy_requested": settings.initial_geometry_strategy,
        "status": result.status,
        "usable": result.accepted,
        "selected_geometry_hint": (
            "negative_hessian" if result.accepted else "low_rank_spd_quadratic_rejected"
        ),
        "fallback_used": not result.accepted,
        "fallback_reason": None if result.accepted else result.status,
        "position_role": (
            "low_rank_spd_quadratic_refined_center"
            if result.center_refinement_accepted
            else "initial_position"
        ),
        "covariance_source": "low_rank_spd_quadratic_precision",
        "low_rank_spd_quadratic_payload": payload,
        "coordinate_transform": "theta = center + prior_scale * z",
        "scale_source": "MinimalZhaoCuiHMCTargetAdapter.prior_scale",
        "classification": "extension_or_invention",
        "reports_map_quality": False,
        "reports_hmc_convergence": False,
        "reports_default_readiness": False,
        "nonclaims": (
            "low-rank SPD quadratic geometry diagnostic only",
            "not a certified MAP covariance",
            "no posterior convergence claim",
            "no sampler readiness claim",
            "no source-faithful Zhao-Cui parity claim",
        ),
    }
    if not result.accepted or result.precision is None:
        return None, diagnostics

    transform = np.diag(1.0 / scale)
    original_precision = transform @ np.asarray(result.precision, dtype=float) @ transform
    original_precision = 0.5 * (original_precision + original_precision.T)
    initial_for_tuning = (
        np.asarray(result.refined_center, dtype=float)
        if result.center_refinement_accepted and result.refined_center is not None
        else center
    )
    eigvals = np.linalg.eigvalsh(original_precision)
    return {
        "initial_position": initial_for_tuning,
        "negative_hessian": original_precision,
        "initial_covariance": None,
    }, {
        **diagnostics,
        "selected_geometry_hint": "negative_hessian",
        "fallback_used": False,
        "fallback_reason": None,
        "negative_hessian_eigen_min": float(np.min(eigvals)),
        "negative_hessian_eigen_max": float(np.max(eigvals)),
        "negative_hessian_nonpositive_eigenvalue_count": int(np.sum(eigvals <= 0.0)),
        "original_coordinate_precision_condition_number": float(
            np.max(eigvals) / np.min(eigvals)
        ),
    }


def local_initial_negative_hessian(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    initial_position: tf.Tensor,
    *,
    map_status: str,
) -> Mapping[str, Any]:
    position = tf.reshape(tf.convert_to_tensor(initial_position, dtype=tf.float64), [-1])
    value, score = adapter.log_prob_and_grad(position)
    score = tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1])
    hessian = score_jacobian(adapter, position)
    hessian_np = np.asarray(hessian.numpy(), dtype=float)
    symmetric = 0.5 * (hessian_np + hessian_np.T)
    eigvals = np.linalg.eigvalsh(symmetric) if np.all(np.isfinite(symmetric)) else np.array([])
    finite_hessian = bool(np.all(np.isfinite(symmetric)) and eigvals.size == int(adapter.parameter_dim))
    finite_value = bool(tf.math.is_finite(value).numpy())
    finite_score = bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    usable = bool(finite_value and finite_score and finite_hessian)
    return {
        "schema": "minimal_ssl_lstm.phase5_initial_geometry.v1",
        "strategy_requested": "local_initial_negative_hessian_after_map_candidate_rejected",
        "status": "usable" if usable else "initial_hessian_nonfinite",
        "usable": usable,
        "map_candidate_status": str(map_status),
        "initial_log_prob": float(value.numpy()),
        "initial_score_norm": float(tf.linalg.norm(score).numpy()),
        "curvature_floor_provenance": (
            "diagnostic prior precision floor from minimal target prior_scale=5.0"
        ),
        "negative_hessian": symmetric,
        "negative_hessian_eigen_min": None
        if not finite_hessian
        else float(np.min(eigvals)),
        "negative_hessian_eigen_max": None
        if not finite_hessian
        else float(np.max(eigvals)),
        "negative_hessian_nonpositive_eigenvalue_count": None
        if not finite_hessian
        else int(np.sum(eigvals <= 0.0)),
        "position_role": "initial_position",
        "covariance_source": "regularized_negative_hessian_at_initial_position",
        "nonclaims": (
            "local initial-position curvature diagnostic only",
            "not a MAP covariance",
            "regularization required before mass use",
            "no posterior convergence claim",
            "no sampler readiness claim",
        ),
    }


def map_candidate_negative_hessian(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    initial_position: tf.Tensor,
    settings: Phase5TuningSettings,
) -> Mapping[str, Any]:
    initial = tf.reshape(tf.convert_to_tensor(initial_position, dtype=tf.float64), [-1])

    def objective_and_grad(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score = adapter.log_prob_and_grad(theta)
        return -tf.convert_to_tensor(value, dtype=tf.float64), -tf.reshape(
            tf.convert_to_tensor(score, dtype=tf.float64),
            [-1],
        )

    optimizer = tfp.optimizer.lbfgs_minimize(
        objective_and_grad,
        initial_position=initial,
        max_iterations=int(settings.map_candidate_max_iterations),
        tolerance=tf.constant(float(settings.map_candidate_tolerance), dtype=tf.float64),
        parallel_iterations=1,
    )
    position = tf.reshape(
        tf.convert_to_tensor(optimizer.position, dtype=tf.float64),
        [-1],
    )
    value, score = adapter.log_prob_and_grad(position)
    score = tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1])
    gradient_norm = float(tf.linalg.norm(score).numpy())
    finite_value = bool(tf.math.is_finite(value).numpy())
    finite_score = bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    hessian = score_jacobian(adapter, position)
    hessian_np = np.asarray(hessian.numpy(), dtype=float)
    symmetric = 0.5 * (hessian_np + hessian_np.T)
    eigvals = np.linalg.eigvalsh(symmetric) if np.all(np.isfinite(symmetric)) else np.array([])
    finite_hessian = bool(np.all(np.isfinite(symmetric)) and eigvals.size == int(adapter.parameter_dim))
    status = "usable"
    if not finite_value or not finite_score:
        status = "map_candidate_nonfinite_value_or_score"
    elif not finite_hessian:
        status = "map_candidate_hessian_nonfinite"
    elif gradient_norm > float(settings.map_candidate_gradient_norm_tolerance):
        status = "map_candidate_gradient_norm_above_diagnostic_tolerance"
    return {
        "schema": "minimal_ssl_lstm.phase5_initial_geometry.v1",
        "strategy_requested": settings.initial_geometry_strategy,
        "status": status,
        "usable": status == "usable",
        "optimizer_converged": bool(optimizer.converged.numpy()),
        "optimizer_failed": bool(optimizer.failed.numpy()),
        "optimizer_iterations": int(optimizer.num_iterations.numpy()),
        "optimizer_objective_value": float(optimizer.objective_value.numpy()),
        "initial_log_prob": float(adapter.log_prob_and_grad(initial)[0].numpy()),
        "map_candidate_log_prob": float(value.numpy()),
        "map_candidate_score_norm": gradient_norm,
        "gradient_norm_tolerance": float(settings.map_candidate_gradient_norm_tolerance),
        "negative_hessian": symmetric,
        "position": position.numpy(),
        "negative_hessian_eigen_min": None
        if not finite_hessian
        else float(np.min(eigvals)),
        "negative_hessian_eigen_max": None
        if not finite_hessian
        else float(np.max(eigvals)),
        "negative_hessian_nonpositive_eigenvalue_count": None
        if not finite_hessian
        else int(np.sum(eigvals <= 0.0)),
        "position_role": "map_candidate",
        "covariance_source": "regularized_negative_hessian_at_map_candidate",
        "nonclaims": (
            "local MAP-candidate geometry diagnostic only",
            "not a certified global MAP",
            "regularization still required before mass use",
            "no posterior convergence claim",
            "no sampler readiness claim",
        ),
    }


def score_jacobian(
    adapter: MinimalZhaoCuiHMCTargetAdapter,
    position: tf.Tensor,
) -> tf.Tensor:
    theta = tf.reshape(tf.convert_to_tensor(position, dtype=tf.float64), [-1])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(theta)
        _value, score = adapter.log_prob_and_grad(theta)
        score = tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1])
    jacobian = tape.jacobian(score, theta, experimental_use_pfor=False)
    del tape
    if jacobian is None:
        raise ValueError("score Jacobian unavailable for MAP-candidate geometry")
    return -tf.convert_to_tensor(jacobian, dtype=tf.float64)


def build_phase5_tuning_artifact(
    *,
    settings: Phase5TuningSettings | None = None,
    output_dir: Path = DEFAULT_TUNING_DIR,
    output_path: Path = DEFAULT_JSON_PATH,
    markdown_output_path: Path = DEFAULT_MARKDOWN_PATH,
    command: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    settings = reviewed_phase5_settings() if settings is None else settings
    command_tuple = tuple(sys.argv if command is None else command)
    started_wall = datetime.now(UTC)
    started_perf = time.perf_counter()
    hard_vetoes: list[str] = []
    errors: list[str] = []

    phase3 = load_phase3_baseline()
    phase4 = load_phase4_baseline()
    if not phase3["phase3_preconditions_met"]:
        hard_vetoes.append("phase3_preconditions_not_met")
    if not phase4["phase4_preconditions_met"]:
        hard_vetoes.append("phase4_preconditions_not_met")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        hard_vetoes.append("cpu_hidden_execution_not_confirmed")

    adapter = MinimalZhaoCuiHMCTargetAdapter(evidence_path=SUBPLAN_PATH)
    initial_position = initial_minimal_ssl_lstm_hmc_state(
        settings.initial_offset_scale
    )
    initial_value, initial_score = adapter.log_prob_and_grad(initial_position)
    if not bool(tf.reduce_all(tf.math.is_finite(initial_value)).numpy()):
        hard_vetoes.append("initial_target_value_nonfinite")
    if not bool(tf.reduce_all(tf.math.is_finite(initial_score)).numpy()):
        hard_vetoes.append("initial_target_score_nonfinite")

    tuning_result_payload: Mapping[str, Any] | None = None
    public_tuning_payload: Mapping[str, Any] | None = None
    progress_payload: Mapping[str, Any] | None = None
    tuning_result_summary: Mapping[str, Any] = {}
    geometry_diagnostics: Mapping[str, Any] = {}
    if not hard_vetoes:
        try:
            geometry_inputs, geometry_diagnostics = initial_geometry_inputs(
                adapter,
                initial_position,
                settings,
            )
            config = phase5_tuning_config(
                settings,
                target_scope=adapter.target_scope,
                geometry_position_role=str(
                    geometry_diagnostics.get("position_role", "initial_position")
                ),
                negative_hessian_source=str(
                    geometry_diagnostics.get("covariance_source", "negative_hessian")
                ),
            )
            result = tune_hmc_kernel(
                adapter=adapter,
                initial_position=geometry_inputs["initial_position"],
                negative_hessian=geometry_inputs["negative_hessian"],
                initial_covariance=geometry_inputs["initial_covariance"],
                config=config,
                output_dir=output_dir,
            )
            tuning_result_payload = result.payload(include_internal_diagnostics=False)
            tuning_result_summary = summarize_tuning_result(result)
            public_path = output_dir / "hmc_kernel_tuning_result.json"
            progress_path = output_dir / "hmc_kernel_tuning_progress.json"
            if public_path.exists():
                public_tuning_payload = json.loads(public_path.read_text(encoding="utf-8"))
            if progress_path.exists():
                progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            hard_vetoes.append("tune_hmc_kernel_runtime_exception")
            errors.append(f"{type(exc).__name__}: {exc}")

    final_status = str(tuning_result_summary.get("final_status", "not_run"))
    diagnostic_role = str(tuning_result_summary.get("diagnostic_role", "not_run"))
    structured_result = bool(
        tuning_result_summary
        and final_status in VALID_STRUCTURED_FINAL_STATUSES
        and tuning_result_payload is not None
    )
    if not hard_vetoes and not structured_result:
        hard_vetoes.append("missing_structured_tuning_result")

    tuning_hard_vetoes = tuple(
        str(item) for item in tuning_result_summary.get("hard_vetoes", ())
    )
    structured_tuning_blocker = bool(
        structured_result
        and final_status in STRUCTURED_BLOCKER_FINAL_STATUSES
        and tuning_hard_vetoes
    )
    if tuning_hard_vetoes and not (
        settings.allow_structured_tuning_hard_veto_artifact
        and structured_tuning_blocker
    ):
        hard_vetoes.append("tuning_result_hard_veto")

    phase_status = "failed" if hard_vetoes else "passed"
    if final_status in PROMOTING_FINAL_STATUSES and phase_status == "passed":
        phase_decision = "structured_tuning_handoff_candidate_nominated"
    elif structured_tuning_blocker and phase_status == "passed":
        phase_decision = "structured_tuning_hard_veto_blocks_phase6"
    elif final_status in NONPROMOTING_FINAL_STATUSES and phase_status == "passed":
        phase_decision = "structured_non_promoting_tuning_result_recorded"
    elif phase_status == "passed":
        phase_decision = "structured_tuning_result_recorded"
    else:
        phase_decision = "invalid_or_hard_vetoed_tuning_artifact"

    runtime_s = time.perf_counter() - started_perf
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": phase_status,
        "phase_decision": phase_decision,
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "errors": tuple(errors),
        "phase3_baseline": phase3,
        "phase4_baseline": phase4,
        "predeclared_settings": settings.payload(),
        "initial_geometry_diagnostics": geometry_diagnostics,
        "tuning_result_summary": tuning_result_summary,
        "tuning_result_payload_public_safe": tuning_result_payload,
        "public_tuning_artifact": public_tuning_payload,
        "public_progress_artifact": progress_payload,
        "native_divergence_status_carried_forward": phase4.get(
            "native_divergence_telemetry_status"
        ),
        "native_divergence_interpretation": (
            "Phase 4 native divergence telemetry remains unavailable; this is "
            "not zero divergences and is not repaired by Phase 5 tuning."
        ),
        "evidence_contract": {
            "question": (
                "Can the repaired BayesFilter staged tuning machinery produce a "
                "finite, internally valid diagnostic artifact and frozen-kernel "
                "handoff candidate for the minimal zhaocui_fixed target?"
            ),
            "baseline_comparator": (
                "Phase 3 fixed-kernel no-adaptation artifact plus Phase 2 oracle "
                "and Phase 4 telemetry-status artifact."
            ),
            "primary_pass_criterion": (
                "Structured tuning result produced with required nonclaims "
                "preserved; public tuner hard vetoes are blocker evidence, not "
                "wrapper invalidity."
            ),
            "promotion_criterion": "none in this phase",
            "veto_diagnostics": (
                "runtime exception",
                "nonfinite target/value/score",
                "invalid staged result",
                "tuning result hard veto",
                "proxy divergence substitution",
                "unsupported zero-divergence claim",
            ),
            "explanatory_only": (
                "acceptance",
                "runtime",
                "stage status",
                "repair triggers",
                "step/leapfrog summaries hidden inside BayesFilter-owned artifacts",
            ),
            "not_concluded": (
                "zero divergences",
                "posterior correctness",
                "broad HMC convergence",
                "tuned-kernel superiority",
                "default readiness",
                "production readiness",
                "source-faithful Zhao-Cui parity",
            ),
        },
        "decision_table": {
            "decision": phase_decision,
            "primary_criterion_status": "passed" if phase_status == "passed" else "failed",
            "veto_diagnostic_status": (
                "no hard vetoes" if not hard_vetoes else ", ".join(dict.fromkeys(hard_vetoes))
            ),
            "main_uncertainty": (
                "This is a bounded CPU-hidden tuning diagnostic; it does not "
                "establish posterior correctness or convergence."
            ),
            "next_justified_action": (
                "If structured and hard-vetoed, repair the stated tuning blocker "
                "before Phase 6; if passed, validate handoff in a separately "
                "reviewed run."
            ),
            "what_is_not_being_concluded": (
                "No zero-divergence, posterior correctness, broad convergence, "
                "ranking, superiority, default-readiness, or production-readiness claim."
            ),
        },
        "inference_status": {
            "hard_veto_screen": "passed" if not hard_vetoes else "failed",
            "statistically_supported_ranking": "not_applicable",
            "descriptive_only_differences": (
                "stage status, acceptance, repair triggers, and runtime are diagnostic only"
            ),
            "default_readiness": "not_checked",
            "next_evidence_needed": (
                "Repair any structured Phase 5 tuning hard veto before Phase 6; "
                "preserve Phase 4 missing-divergence limitation."
            ),
        },
        "run_manifest": {
            "command": command_tuple,
            "cwd": str(ROOT),
            "script": f"docs/benchmarks/{SCRIPT_NAME}",
            "started_at_utc": started_wall.isoformat(),
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "runtime_s": float(runtime_s),
            "environment": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
                "tensorflow_version": tf.__version__,
                "tensorflow_probability_version": tfp.__version__,
            },
            "random_seed": settings.seed,
            "output_artifact": manifest_path(output_path),
            "markdown_artifact": manifest_path(markdown_output_path),
            "public_tuning_output_dir": str(output_dir.relative_to(ROOT))
            if output_dir.is_relative_to(ROOT)
            else str(output_dir),
            "plan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "fixture": minimal_ssl_lstm_fixture_payload(),
        "nonclaims": NONCLAIMS,
    }
    return json_ready(artifact)


def manifest_path(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def summarize_tuning_result(result: Any) -> Mapping[str, Any]:
    loop = result.tune_verify_repair_loop
    attempts = tuple() if loop is None else tuple(loop.attempts)
    stage_statuses = []
    if attempts:
        attempt = attempts[-1]
        for name in (
            "windowed_stage",
            "fixed_mass_step_stage",
            "frozen_step_trajectory_stage",
        ):
            stage = getattr(attempt, name)
            stage_statuses.append(
                {
                    "stage": name,
                    "final_status": None if stage is None else stage.final_status,
                    "passed": None if stage is None else bool(stage.passed),
                    "diagnostic_role": None if stage is None else stage.diagnostic_role,
                    "hard_vetoes": None if stage is None else tuple(stage.hard_vetoes),
                    "repair_triggers": None
                    if stage is None
                    else tuple(getattr(stage, "repair_triggers", ())),
                }
            )
    return {
        "schema": "minimal_ssl_lstm.phase5_tuning_result_summary.v1",
        "final_status": result.final_status,
        "passed": bool(result.passed),
        "diagnostic_role": result.diagnostic_role,
        "hard_vetoes": tuple(result.hard_vetoes),
        "repair_triggers": tuple(result.repair_triggers),
        "final_kernel_hash": result.final_kernel_hash,
        "final_kernel_payload_available": result.final_kernel_payload is not None,
        "target_dimension": result.target_dimension,
        "geometry_available": result.geometry is not None,
        "bootstrap_available": result.bootstrap is not None,
        "loop_available": loop is not None,
        "attempt_count": len(attempts),
        "stage_statuses": tuple(stage_statuses),
        "artifact_path": result.artifact_path,
        "nonclaims": tuple(result.nonclaims),
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_ready(value.numpy())
    return value


def render_markdown(artifact: Mapping[str, Any]) -> str:
    summary = artifact.get("tuning_result_summary", {})
    lines = [
        "# Minimal SSL-LSTM Zhao-Cui HMC Validity Phase 5",
        "",
        "## Summary",
        "",
        f"- Status: `{artifact['status']}`",
        f"- Phase decision: `{artifact['phase_decision']}`",
        f"- Hard vetoes: `{artifact['hard_vetoes']}`",
        f"- Tuning final status: `{summary.get('final_status')}`",
        f"- Tuning diagnostic role: `{summary.get('diagnostic_role')}`",
        f"- Repair triggers: `{summary.get('repair_triggers')}`",
        f"- Final kernel hash: `{summary.get('final_kernel_hash')}`",
        f"- Native divergence carried forward: `{artifact['native_divergence_status_carried_forward']}`",
        "",
        "## Decision Table",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in artifact["decision_table"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Field | Value |",
            "| --- | --- |",
        ]
    )
    for key, value in artifact["inference_status"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(["", "## Stage Statuses", ""])
    for row in summary.get("stage_statuses", ()):
        lines.append(
            f"- `{row['stage']}`: status `{row['final_status']}`, "
            f"passed `{row['passed']}`, role `{row['diagnostic_role']}`"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in artifact["nonclaims"])
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Subplan: `{SUBPLAN_PATH}`",
            f"- Result: `{RESULT_PATH}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_PATH)
    parser.add_argument("--tuning-output-dir", type=Path, default=DEFAULT_TUNING_DIR)
    parser.add_argument("--public-timeout-budget-s", type=float, default=None)
    parser.add_argument("--terminal-phase6-repair-extra-attempts", type=int, default=None)
    parser.add_argument(
        "--initial-geometry-strategy",
        choices=(
            "map_candidate_hessian",
            "initial_covariance",
            "low_rank_spd_quadratic",
        ),
        default=None,
    )
    parser.add_argument("--low-rank-quadratic-sample-count", type=int, default=None)
    args = parser.parse_args(argv)

    settings = reviewed_phase5_settings()
    if args.initial_geometry_strategy is not None:
        settings = replace(
            settings,
            initial_geometry_strategy=str(args.initial_geometry_strategy),
        )
    if args.low_rank_quadratic_sample_count is not None:
        settings = replace(
            settings,
            low_rank_quadratic_sample_count=int(args.low_rank_quadratic_sample_count),
        )
    if args.public_timeout_budget_s is not None:
        settings = replace(
            settings,
            public_timeout_budget_s=float(args.public_timeout_budget_s),
        )
    if args.terminal_phase6_repair_extra_attempts is not None:
        settings = replace(
            settings,
            terminal_phase6_repair_extra_attempts=(
                int(args.terminal_phase6_repair_extra_attempts)
            ),
        )
    artifact = build_phase5_tuning_artifact(
        settings=settings,
        output_dir=args.tuning_output_dir,
        output_path=args.output,
        markdown_output_path=args.markdown_output,
        command=(
            [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
            if argv is not None
            else sys.argv
        ),
    )
    atomic_write_json(args.output, artifact)
    atomic_write_text(args.markdown_output, render_markdown(artifact))
    if artifact["status"] != "passed":
        raise RuntimeError(f"Phase 5 tuning diagnostic failed: {artifact['hard_vetoes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
