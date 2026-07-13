"""Phase 2AC scalar SSL-LSTM sequential-resampling repair.

This diagnostic runs a small CPU-hidden sequential tempering reference pilot in
the Phase 2S/2U MAP-local ``u_new`` coordinate after independent SNIS proposal
attempts failed.  It may nominate a route for independent replication.  It does
not run HMC and does not claim posterior correctness, HMC readiness,
convergence, zero divergences, GPU/XLA readiness, default readiness, or
Zhao-Cui source faithfulness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_"
    "sequential_resampling_repair_2026_07_09.py"
)
SCHEMA_VERSION = (
    "scalar_ssl_lstm.filtering_hmc_validation_phase2ac_sequential_resampling_repair.v1"
)
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md"
)
DEFAULT_PHASE2S_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2T_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2U_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2V_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2W_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2X_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2Y_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2Z_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2AB_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2ab_transport_or_sequential_reference_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md"
)
PHASE2U_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
LAUNCH_COMMIT = "52ee244498988e046a6356f926003b581103083b"
DIMENSION = 4
NONCLAIMS = (
    "Phase 2AC sequential-resampling repair pilot only",
    "not a valid replicated reference by itself",
    "not HMC-vs-reference agreement evidence",
    "not an HMC run",
    "not HMC readiness evidence",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a zero-divergence claim when native divergence is unavailable",
    "not sampler superiority evidence",
    "not statistically supported ranking evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not Zhao-Cui source-faithfulness evidence",
)


@dataclass(frozen=True)
class Phase2ACSequentialSettings:
    """Fixed Phase 2AC sequential-resampling repair settings."""

    particle_count: int = 128
    seed: tuple[int, int] = (20260709, 6801)
    target_ess_ratio: float = 0.70
    minimum_ess_ratio: float = 0.50
    resample_ess_ratio: float = 0.50
    terminal_max_weight_max: float = 0.08
    unique_ancestor_fraction_min: float = 0.25
    rejuvenation_scale: float = 0.45
    rejuvenation_moves_per_stage: int = 1
    rejuvenation_acceptance_min: float = 0.10
    rejuvenation_acceptance_max: float = 0.90
    max_stages: int = 48
    bisection_iterations: int = 32
    beta_tolerance: float = 1.0e-6
    minimum_beta_increment: float = 1.0e-4
    resample_boundary_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        count = int(self.particle_count)
        if count <= 0:
            raise ValueError("particle_count must be positive")
        object.__setattr__(self, "particle_count", count)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        for name in (
            "target_ess_ratio",
            "minimum_ess_ratio",
            "resample_ess_ratio",
            "terminal_max_weight_max",
            "unique_ancestor_fraction_min",
            "rejuvenation_scale",
            "rejuvenation_acceptance_min",
            "rejuvenation_acceptance_max",
            "beta_tolerance",
            "minimum_beta_increment",
            "resample_boundary_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not (0.0 < self.minimum_ess_ratio <= self.target_ess_ratio <= 1.0):
            raise ValueError("ESS thresholds must satisfy 0 < minimum <= target <= 1")
        if not (0.0 < self.resample_ess_ratio <= 1.0):
            raise ValueError("resample_ess_ratio must be in (0, 1]")
        if not (0.0 < self.terminal_max_weight_max <= 1.0):
            raise ValueError("terminal_max_weight_max must be in (0, 1]")
        if not (0.0 < self.unique_ancestor_fraction_min <= 1.0):
            raise ValueError("unique_ancestor_fraction_min must be in (0, 1]")
        if self.rejuvenation_scale <= 0.0:
            raise ValueError("rejuvenation_scale must be positive")
        moves = int(self.rejuvenation_moves_per_stage)
        if moves <= 0:
            raise ValueError("rejuvenation_moves_per_stage must be positive")
        object.__setattr__(self, "rejuvenation_moves_per_stage", moves)
        if not (0.0 <= self.rejuvenation_acceptance_min <= self.rejuvenation_acceptance_max <= 1.0):
            raise ValueError("rejuvenation acceptance interval must be inside [0, 1]")
        for name in ("max_stages", "bisection_iterations"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if (
            self.beta_tolerance <= 0.0
            or self.minimum_beta_increment <= 0.0
            or self.resample_boundary_tolerance <= 0.0
        ):
            raise ValueError("beta, increment, and resample tolerances must be positive")

    def payload(self) -> Mapping[str, Any]:
        return {
            "coordinate": "phase2s_phase2u_map_local_u_new",
            "dimension": DIMENSION,
            "particle_count": self.particle_count,
            "seed": self.seed,
            "base_density": "standard_normal_N_0_I4",
            "target_route": (
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_"
                "retuned_map_local_hmc_screen_2026_07_09.py::build_phase2u_adapter"
            ),
            "launch_commit": LAUNCH_COMMIT,
            "target_ess_ratio": self.target_ess_ratio,
            "minimum_ess_ratio": self.minimum_ess_ratio,
            "resample_ess_ratio": self.resample_ess_ratio,
            "terminal_max_weight_max": self.terminal_max_weight_max,
            "unique_ancestor_fraction_min": self.unique_ancestor_fraction_min,
            "rejuvenation_scale": self.rejuvenation_scale,
            "rejuvenation_moves_per_stage": self.rejuvenation_moves_per_stage,
            "rejuvenation_acceptance_interval": (
                self.rejuvenation_acceptance_min,
                self.rejuvenation_acceptance_max,
            ),
            "max_stages": self.max_stages,
            "bisection_iterations": self.bisection_iterations,
            "beta_tolerance": self.beta_tolerance,
            "minimum_beta_increment": self.minimum_beta_increment,
            "resample_boundary_tolerance": self.resample_boundary_tolerance,
            "repair_policy": (
                "force_nonterminal_resampling_after_minimum_threshold_fallback_or_"
                "ess_within_resample_boundary_tolerance"
            ),
            "terminal_weight_measurement": "beta_1_pre_final_resampling",
            "final_ancestor_measurement": "after_last_completed_resampling_and_rejuvenation",
            "hmc_usage": False,
            "cpu_hidden": True,
            "use_xla": False,
        }


def load_module(module_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase2u = load_module(
    PHASE2U_MODULE_PATH,
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_for_phase2ac",
)


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def standard_normal_log_prob(samples: Any) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError("samples must have shape (n, dim)")
    dim = int(array.shape[1])
    return -0.5 * np.sum(np.square(array), axis=1) - 0.5 * dim * np.log(2.0 * np.pi)


def logsumexp(values: Any) -> float:
    array = np.asarray(values, dtype=float)
    max_value = float(np.max(array))
    return max_value + math.log(float(np.sum(np.exp(array - max_value))))


def normalized_weights_from_log(log_weights: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
    array = np.asarray(log_weights, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        weights = np.full_like(array, np.nan, dtype=float)
        return weights, {
            "finite": False,
            "ess": float("nan"),
            "ess_ratio": float("nan"),
            "max": float("nan"),
            "sum": float("nan"),
            "nonzero_count": 0,
        }
    shift = float(np.max(array))
    shifted = np.exp(array - shift)
    total = float(np.sum(shifted))
    if not np.isfinite(total) or total <= 0.0:
        weights = np.full_like(array, np.nan, dtype=float)
    else:
        weights = shifted / total
    if not np.all(np.isfinite(weights)):
        return weights, {
            "finite": False,
            "ess": float("nan"),
            "ess_ratio": float("nan"),
            "max": float("nan"),
            "sum": float("nan"),
            "nonzero_count": 0,
        }
    ess = float(1.0 / np.sum(np.square(weights)))
    return weights, {
        "finite": True,
        "ess": ess,
        "ess_ratio": float(ess / array.shape[0]),
        "max": float(np.max(weights)),
        "sum": float(np.sum(weights)),
        "nonzero_count": int(np.sum(weights > 0.0)),
    }


def systematic_resample(weights: Any, rng: np.random.Generator) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(array))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must have positive sum")
    normalized = array / total
    count = int(normalized.shape[0])
    start = float(rng.uniform(0.0, 1.0 / count))
    positions = start + np.arange(count, dtype=float) / count
    cumulative = np.cumsum(normalized)
    indices = np.searchsorted(cumulative, positions, side="left")
    return np.minimum(indices, count - 1).astype(int)


def evaluate_target_log_prob(adapter: Any, samples: Any) -> tuple[np.ndarray, Mapping[str, Any]]:
    array = np.asarray(samples, dtype=float)
    vetoes: list[str] = []
    first_error = None
    values = np.full((array.shape[0],), np.nan, dtype=float)
    try:
        tensor = tf.constant(array, dtype=tf.float64)
        value_tensor, _score_tensor = adapter.log_prob_and_grad(tensor)
        values = np.asarray(tf.convert_to_tensor(value_tensor, dtype=tf.float64).numpy(), dtype=float)
    except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic.
        first_error = f"{type(exc).__name__}: {exc}"
        vetoes.append("batched_target_evaluation_exception")
    if values.shape != (array.shape[0],):
        vetoes.append("target_log_prob_shape_mismatch")
    if not np.all(np.isfinite(values)):
        vetoes.append("target_log_prob_nonfinite")
    return values, {
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "first_error": first_error,
        "summary": finite_summary(values),
    }


def select_next_beta(
    current_beta: float,
    log_ratio: Any,
    settings: Phase2ACSequentialSettings,
    current_log_weights: Any | None = None,
) -> Mapping[str, Any]:
    beta = float(current_beta)
    current_weights = (
        np.zeros_like(np.asarray(log_ratio, dtype=float), dtype=float)
        if current_log_weights is None
        else np.asarray(current_log_weights, dtype=float)
    )
    if beta >= 1.0 - settings.beta_tolerance:
        return {
            "next_beta": 1.0,
            "delta_beta": 0.0,
            "ess_ratio": 1.0,
            "target_reached": True,
            "vetoes": (),
            "iterations": 0,
            "rule": "already_at_terminal_beta",
        }
    ratio = np.asarray(log_ratio, dtype=float)
    if ratio.ndim != 1 or not np.all(np.isfinite(ratio)):
        return {
            "next_beta": beta,
            "delta_beta": 0.0,
            "ess_ratio": float("nan"),
            "target_reached": False,
            "vetoes": ("log_ratio_nonfinite",),
            "iterations": 0,
            "rule": "invalid_log_ratio",
        }
    if current_weights.shape != ratio.shape or not np.all(np.isfinite(current_weights)):
        return {
            "next_beta": beta,
            "delta_beta": 0.0,
            "ess_ratio": float("nan"),
            "target_reached": False,
            "vetoes": ("current_log_weights_invalid",),
            "iterations": 0,
            "rule": "invalid_current_log_weights",
        }

    def ess_for(candidate_beta: float) -> float:
        _weights, summary = normalized_weights_from_log(
            current_weights + (candidate_beta - beta) * ratio
        )
        return float(summary["ess_ratio"])

    def largest_beta_for_threshold(threshold: float) -> tuple[float, float, int]:
        candidate_beta = beta
        candidate_ess = ess_for(beta)
        if not np.isfinite(candidate_ess) or candidate_ess < threshold:
            return candidate_beta, candidate_ess, 0
        low = beta
        high = 1.0
        iterations_used = 0
        for iterations_used in range(1, settings.bisection_iterations + 1):
            mid = 0.5 * (low + high)
            mid_ess = ess_for(mid)
            if mid_ess >= threshold:
                candidate_beta = mid
                candidate_ess = mid_ess
                low = mid
            else:
                high = mid
            if high - low <= settings.beta_tolerance:
                break
        return candidate_beta, candidate_ess, iterations_used

    terminal_ess = ess_for(1.0)
    if terminal_ess >= settings.minimum_ess_ratio:
        return {
            "next_beta": 1.0,
            "delta_beta": float(1.0 - beta),
            "ess_ratio": terminal_ess,
            "target_reached": True,
            "vetoes": (),
            "iterations": 0,
            "rule": "terminal_beta_admissible",
        }

    candidate, candidate_ess, iterations = largest_beta_for_threshold(settings.target_ess_ratio)
    rule = "bisection_largest_target_admissible_increment"
    if float(candidate - beta) < settings.minimum_beta_increment:
        candidate, candidate_ess, iterations = largest_beta_for_threshold(settings.minimum_ess_ratio)
        rule = "bisection_largest_minimum_admissible_increment"
    delta = float(candidate - beta)
    vetoes: list[str] = []
    if delta < settings.minimum_beta_increment:
        vetoes.append("temperature_increment_stalled")
    if not np.isfinite(candidate_ess) or candidate_ess < settings.minimum_ess_ratio:
        vetoes.append("temperature_ess_ratio_below_minimum")
    return {
        "next_beta": float(candidate),
        "delta_beta": delta,
        "ess_ratio": float(candidate_ess),
        "target_reached": False,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "iterations": int(iterations),
        "rule": rule,
    }


def phase2ac_resample_reason(
    selection: Mapping[str, Any],
    weight_summary: Mapping[str, Any],
    terminal_before_resampling: bool,
    settings: Phase2ACSequentialSettings,
) -> str | None:
    if terminal_before_resampling:
        return None
    if selection.get("rule") == "bisection_largest_minimum_admissible_increment":
        return "minimum_threshold_fallback"
    ess_ratio = float(weight_summary.get("ess_ratio", float("nan")))
    if (
        np.isfinite(ess_ratio)
        and ess_ratio <= settings.resample_ess_ratio + settings.resample_boundary_tolerance
    ):
        return "ess_within_resample_boundary_tolerance"
    return None


def rejuvenate_particles(
    adapter: Any,
    particles: Any,
    target_log_prob: Any,
    beta: float,
    rng: np.random.Generator,
    settings: Phase2ACSequentialSettings,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    current = np.asarray(particles, dtype=float).copy()
    current_target = np.asarray(target_log_prob, dtype=float).copy()
    current_base = standard_normal_log_prob(current)
    accepted = np.zeros((current.shape[0],), dtype=bool)
    vetoes: list[str] = []
    first_error = None
    proposal = current + rng.normal(
        loc=0.0,
        scale=settings.rejuvenation_scale,
        size=current.shape,
    )
    proposal_target, target_eval = evaluate_target_log_prob(adapter, proposal)
    vetoes.extend(target_eval.get("vetoes", ()))
    if not vetoes:
        proposal_base = standard_normal_log_prob(proposal)
        current_tempered = (1.0 - beta) * current_base + beta * current_target
        proposal_tempered = (1.0 - beta) * proposal_base + beta * proposal_target
        log_alpha = proposal_tempered - current_tempered
        uniforms = np.log(rng.uniform(size=current.shape[0]))
        accepted = uniforms < np.minimum(log_alpha, 0.0)
        current[accepted] = proposal[accepted]
        current_target[accepted] = proposal_target[accepted]
    else:
        first_error = target_eval.get("first_error")
    return current, current_target, {
        "accepted_count": int(np.sum(accepted)),
        "proposal_count": int(current.shape[0]),
        "acceptance_rate": float(np.mean(accepted)) if current.shape[0] else None,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "first_error": first_error,
        "target_eval": target_eval,
    }


def run_phase2ac_sequential_resampling_repair(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    phase2y_payload: Mapping[str, Any],
    phase2z_payload: Mapping[str, Any],
    settings: Phase2ACSequentialSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2ACSequentialSettings() if settings is None else settings
    start = time.perf_counter()
    precondition = validate_phase2ac_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        phase2w_payload,
        phase2x_payload,
        phase2y_payload,
        phase2z_payload,
    )
    vetoes = list(precondition.get("vetoes", ()))
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    sequential: Mapping[str, Any] = {
        "computed": False,
        "vetoes": ("phase2ac_not_run",),
    }
    if not vetoes:
        adapter, adapter_audit = phase2u.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2ac_adapter_not_built")
    if adapter is not None and not adapter_audit.get("vetoes"):
        sequential = run_sequential_tempering(adapter, cfg)
        vetoes.extend(sequential.get("vetoes", ()))

    unique_vetoes = tuple(dict.fromkeys(vetoes))
    gate = evaluate_phase2ac_gate(sequential, cfg, precondition, adapter_audit)
    all_vetoes = tuple(dict.fromkeys((*unique_vetoes, *gate.get("vetoes", ()))))
    passed = bool(not unique_vetoes and gate.get("phase2ac_candidate_nominated") is True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2ac_sequential_resampling_repair",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": None if adapter is None else adapter.target_scope,
        "settings": cfg.payload(),
        "source_artifacts": {
            "phase2s_json": str(DEFAULT_PHASE2S_PATH.relative_to(ROOT)),
            "phase2t_json": str(DEFAULT_PHASE2T_PATH.relative_to(ROOT)),
            "phase2u_json": str(DEFAULT_PHASE2U_PATH.relative_to(ROOT)),
            "phase2v_json": str(DEFAULT_PHASE2V_PATH.relative_to(ROOT)),
            "phase2w_json": str(DEFAULT_PHASE2W_PATH.relative_to(ROOT)),
            "phase2x_json": str(DEFAULT_PHASE2X_PATH.relative_to(ROOT)),
            "phase2y_json": str(DEFAULT_PHASE2Y_PATH.relative_to(ROOT)),
            "phase2z_json": str(DEFAULT_PHASE2Z_PATH.relative_to(ROOT)),
            "phase2ab_baseline_json": str(DEFAULT_PHASE2AB_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "sequential_reference": sequential,
        "phase2ac_gate": gate,
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2ac_sequential_resampling_repair_passed": passed,
            "candidate_nominated_for_phase2ad_replication": gate.get(
                "phase2ac_candidate_nominated",
                False,
            ),
            "vetoes": all_vetoes,
            "terminal_beta": sequential.get("terminal_beta"),
            "stage_count": sequential.get("stage_count"),
            "terminal_pre_final_resampling_ess_ratio": sequential.get(
                "terminal_pre_final_resampling_summary",
                {},
            ).get("ess_ratio"),
            "terminal_pre_final_resampling_max_weight": sequential.get(
                "terminal_pre_final_resampling_summary",
                {},
            ).get("max"),
            "unique_ancestor_fraction": sequential.get("unique_ancestor_fraction"),
            "aggregate_rejuvenation_acceptance": sequential.get(
                "aggregate_rejuvenation_acceptance",
            ),
            "runs_hmc": False,
            "zero_divergence_claim_made": False,
            "viable_for_phase3_gpu_xla_subplan": False,
            "next_justified_action": (
                "write Phase 2AC result and draft/review Phase 2AD independent sequential-reference replication subplan"
                if passed
                else "write Phase 2AC result and draft/reference-method blocker or focused repair subplan"
            ),
        },
        "metric_roles": {
            "phase2ac_sequential_resampling_repair_passed": "primary_phase2ac_nomination_gate",
            "target_log_prob_finiteness": "hard_veto_evidence",
            "temperature_reaches_one": "promotion_veto",
            "terminal_pre_final_resampling_ess_ratio": "promotion_veto",
            "terminal_pre_final_resampling_max_weight": "promotion_veto",
            "unique_ancestor_fraction": "promotion_veto",
            "rejuvenation_acceptance": "promotion_veto",
            "particle_moments": "explanatory_only",
            "phase2v_hmc_moment_comparison": "not_evaluated_in_phase2ac",
            "runtime": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if not unique_vetoes else "failed",
            "reference_validity": (
                "not established; Phase 2AC can only nominate Phase 2AD replication"
            ),
            "hmc_reference_agreement": "not assessed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "temperature schedule, ESS trajectory, ancestor diversity, "
                "rejuvenation acceptance, terminal moments, and runtime"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "zero_divergence_claim": "not made",
            "next_evidence_needed": (
                "Phase 2AD independent sequential-reference replication"
                if passed
                else "reviewed focused repair or reference-method blocker"
            ),
        },
        "decision_table": {
            "decision": "Phase 2AC sequential tempering reference viability pilot",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {all_vetoes}",
            "main_uncertainty": (
                "A one-seed CPU-hidden sequential pilot can nominate a reference route "
                "for replication only; it cannot establish posterior correctness or HMC readiness."
            ),
            "next_justified_action": (
                "draft/review Phase 2AD independent replication subplan"
                if passed
                else "draft focused repair only if the artifact identifies one discriminating repair question"
            ),
            "what_is_not_being_concluded": (
                "No valid replicated reference, HMC-vs-reference agreement, posterior correctness, "
                "HMC readiness, convergence, zero-divergence claim, sampler superiority, "
                "statistical ranking, GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2ac_sequential_resampling_repair_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": (cfg.seed,),
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "subplan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "A passing pilot may reflect a lucky one-seed sequential path or loose diversity gates; "
                "independent replication is required before any agreement interpretation."
            ),
            "what_would_overturn": (
                "Failure to replicate with fresh seeds, nonfinite target evaluations, severe ancestor collapse, "
                "or disagreement under a later valid reference-agreement phase."
            ),
            "weakest_evidence": (
                "One CPU-hidden pilot with 128 particles and no uncertainty analysis."
            ),
        },
        "review_record": {
            "claude_round_1": "VERDICT_REVISE",
            "claude_round_1_process_note": (
                "worker issued read/status commands despite read-only/no-command prompt; no edits were made"
            ),
            "focused_local_review": "VERDICT_AGREE_FOR_IMPLEMENTATION_AND_CPU_HIDDEN_PILOT",
            "review_strength": "Claude compact review plus local focused repair review",
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_phase2ac_handoff(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    phase2y_payload: Mapping[str, Any],
    phase2z_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    phase2u_precondition = phase2u.validate_handoff_artifacts(
        phase2s_payload,
        phase2t_payload,
        phase2u.Phase2UScreenSettings(),
    )
    vetoes.extend(f"phase2u_precondition_{item}" for item in phase2u_precondition.get("vetoes", ()))
    if (
        phase2u_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1"
    ):
        vetoes.append("phase2u_schema_mismatch")
    if phase2u_payload.get("decision", {}).get("phase2u_retuned_map_local_hmc_screen_passed") is not True:
        vetoes.append("phase2u_decision_not_passed")
    if (
        phase2v_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1"
    ):
        vetoes.append("phase2v_schema_mismatch")
    if phase2v_payload.get("decision", {}).get("phase2v_longer_selected_map_local_screen_passed") is not True:
        vetoes.append("phase2v_decision_not_passed")
    if phase2w_payload.get("decision", {}).get("reference_valid") is True:
        vetoes.append("phase2w_reference_unexpectedly_valid")
    if phase2x_payload.get("decision", {}).get("reference_valid") is True:
        vetoes.append("phase2x_reference_unexpectedly_valid")
    if phase2y_payload.get("decision", {}).get("phase2y_target_geometry_localization_passed") is not True:
        vetoes.append("phase2y_decision_not_passed")
    if phase2y_payload.get("decision", {}).get("proposal_family_mismatch_indicated") is not True:
        vetoes.append("phase2y_proposal_family_mismatch_not_indicated")
    if phase2z_payload.get("decision", {}).get("phase2z_proposal_strategy_pilot_passed") is not True:
        vetoes.append("phase2z_decision_not_passed")
    if phase2z_payload.get("decision", {}).get("candidate_nominated") is True:
        vetoes.append("phase2z_candidate_unexpectedly_nominated")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2u_precondition": phase2u_precondition,
        "phase2u_decision": phase2u_payload.get("decision", {}),
        "phase2v_decision": phase2v_payload.get("decision", {}),
        "phase2w_decision": phase2w_payload.get("decision", {}),
        "phase2x_decision": phase2x_payload.get("decision", {}),
        "phase2y_decision": phase2y_payload.get("decision", {}),
        "phase2z_decision": phase2z_payload.get("decision", {}),
    }


def run_sequential_tempering(
    adapter: Any,
    settings: Phase2ACSequentialSettings,
) -> Mapping[str, Any]:
    rng = np.random.default_rng(settings.seed)
    particles = rng.standard_normal(size=(settings.particle_count, DIMENSION))
    ancestor_ids = np.arange(settings.particle_count, dtype=int)
    base_log_prob = standard_normal_log_prob(particles)
    target_log_prob, initial_eval = evaluate_target_log_prob(adapter, particles)
    vetoes = list(initial_eval.get("vetoes", ()))
    if vetoes:
        return {
            "computed": False,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "initial_target_eval": initial_eval,
        }
    log_ratio = target_log_prob - base_log_prob
    beta = 0.0
    stage_rows: list[Mapping[str, Any]] = []
    current_log_weights = np.zeros((settings.particle_count,), dtype=float)
    aggregate_accepts = 0
    aggregate_proposals = 0
    terminal_pre_final_summary: Mapping[str, Any] = {}

    for stage_index in range(settings.max_stages):
        selection = select_next_beta(beta, log_ratio, settings, current_log_weights)
        vetoes.extend(selection.get("vetoes", ()))
        if selection.get("vetoes"):
            break
        next_beta = float(selection["next_beta"])
        incremental = (next_beta - beta) * log_ratio
        current_log_weights = current_log_weights + incremental
        normalized, weight_summary = normalized_weights_from_log(current_log_weights)
        if not weight_summary.get("finite"):
            vetoes.append("normalized_weight_invalid")
            break
        terminal_before_resampling = bool(next_beta >= 1.0 - settings.beta_tolerance)
        if terminal_before_resampling:
            terminal_pre_final_summary = dict(weight_summary)
        resampled = False
        resample_indices = None
        pre_resample_summary = dict(weight_summary)
        resample_reason = phase2ac_resample_reason(
            selection,
            weight_summary,
            terminal_before_resampling,
            settings,
        )
        if resample_reason is not None:
            resample_indices = systematic_resample(normalized, rng)
            particles = particles[resample_indices]
            ancestor_ids = ancestor_ids[resample_indices]
            target_log_prob = target_log_prob[resample_indices]
            base_log_prob = standard_normal_log_prob(particles)
            log_ratio = target_log_prob - base_log_prob
            current_log_weights = np.zeros_like(current_log_weights)
            resampled = True
        rejuvenation_rows = []
        for _move in range(settings.rejuvenation_moves_per_stage):
            particles, target_log_prob, rejuvenation = rejuvenate_particles(
                adapter,
                particles,
                target_log_prob,
                next_beta,
                rng,
                settings,
            )
            vetoes.extend(rejuvenation.get("vetoes", ()))
            rejuvenation_rows.append(rejuvenation)
            aggregate_accepts += int(rejuvenation.get("accepted_count", 0))
            aggregate_proposals += int(rejuvenation.get("proposal_count", 0))
            if rejuvenation.get("vetoes"):
                break
        base_log_prob = standard_normal_log_prob(particles)
        log_ratio = target_log_prob - base_log_prob
        unique_fraction = float(np.unique(ancestor_ids).shape[0] / settings.particle_count)
        stage_rows.append(
            {
                "stage_index": stage_index,
                "previous_beta": beta,
                "beta": next_beta,
                "delta_beta": selection["delta_beta"],
                "selection": selection,
                "pre_resample_weight_summary": pre_resample_summary,
                "terminal_pre_final_resampling_measurement": terminal_before_resampling,
                "resampled": resampled,
                "resample_reason": resample_reason,
                "resample_policy": "phase2ac_force_fallback_or_boundary",
                "resample_index_summary": (
                    None
                    if resample_indices is None
                    else {
                        "unique_parent_count": int(np.unique(resample_indices).shape[0]),
                        "min_parent_index": int(np.min(resample_indices)),
                        "max_parent_index": int(np.max(resample_indices)),
                    }
                ),
                "unique_ancestor_fraction_after_stage": unique_fraction,
                "rejuvenation": rejuvenation_rows,
            }
        )
        beta = next_beta
        if vetoes or beta >= 1.0 - settings.beta_tolerance:
            break

    if beta < 1.0 - settings.beta_tolerance:
        vetoes.append("temperature_schedule_did_not_reach_beta_one")
    if not terminal_pre_final_summary:
        _normalized, terminal_pre_final_summary = normalized_weights_from_log(current_log_weights)
    final_weights, final_weight_summary = normalized_weights_from_log(current_log_weights)
    if not final_weight_summary.get("finite"):
        vetoes.append("final_weight_summary_invalid")
    unique_ancestor_fraction = float(np.unique(ancestor_ids).shape[0] / settings.particle_count)
    aggregate_acceptance = (
        float(aggregate_accepts / aggregate_proposals)
        if aggregate_proposals > 0
        else float("nan")
    )
    terminal_moments = weighted_moments(particles, final_weights)
    return json_ready(
        {
            "computed": True,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "terminal_beta": float(beta),
            "stage_count": int(len(stage_rows)),
            "initial_target_eval": initial_eval,
            "stage_rows": stage_rows,
            "terminal_pre_final_resampling_summary": terminal_pre_final_summary,
            "final_weight_summary": final_weight_summary,
            "unique_ancestor_fraction": unique_ancestor_fraction,
            "aggregate_rejuvenation_acceptance": aggregate_acceptance,
            "aggregate_rejuvenation_accepted_count": int(aggregate_accepts),
            "aggregate_rejuvenation_proposal_count": int(aggregate_proposals),
            "particle_summary": {
                "mean": np.mean(particles, axis=0),
                "std": np.std(particles, axis=0),
                "min": np.min(particles, axis=0),
                "max": np.max(particles, axis=0),
                "max_abs": float(np.max(np.abs(particles))),
            },
            "terminal_weighted_moments": terminal_moments,
            "final_particles": particles,
            "final_log_weights": current_log_weights,
            "final_normalized_weights": final_weights,
        }
    )


def evaluate_phase2ac_gate(
    sequential: Mapping[str, Any],
    settings: Phase2ACSequentialSettings,
    precondition: Mapping[str, Any],
    adapter_audit: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    vetoes.extend(precondition.get("vetoes", ()))
    vetoes.extend(adapter_audit.get("vetoes", ()))
    vetoes.extend(sequential.get("vetoes", ()))
    if sequential.get("computed") is not True:
        vetoes.append("sequential_reference_not_computed")
    if float(sequential.get("terminal_beta", float("nan"))) < 1.0 - settings.beta_tolerance:
        vetoes.append("terminal_beta_not_one")
    if int(sequential.get("stage_count", settings.max_stages + 1)) > settings.max_stages:
        vetoes.append("stage_count_above_max")
    terminal = sequential.get("terminal_pre_final_resampling_summary", {})
    terminal_ess_ratio = float(terminal.get("ess_ratio", float("nan")))
    terminal_max_weight = float(terminal.get("max", float("nan")))
    if not np.isfinite(terminal_ess_ratio) or terminal_ess_ratio < settings.minimum_ess_ratio:
        vetoes.append("terminal_pre_final_resampling_ess_ratio_below_threshold")
    if not np.isfinite(terminal_max_weight) or terminal_max_weight > settings.terminal_max_weight_max:
        vetoes.append("terminal_pre_final_resampling_max_weight_above_threshold")
    min_post_temperature_ess_ratio = min_post_temperature_ess(sequential)
    if (
        not np.isfinite(min_post_temperature_ess_ratio)
        or min_post_temperature_ess_ratio < settings.minimum_ess_ratio
    ):
        vetoes.append("minimum_adaptive_post_temperature_ess_ratio_below_threshold")
    unique_fraction = float(sequential.get("unique_ancestor_fraction", float("nan")))
    if not np.isfinite(unique_fraction) or unique_fraction < settings.unique_ancestor_fraction_min:
        vetoes.append("unique_ancestor_fraction_below_threshold")
    acceptance = float(sequential.get("aggregate_rejuvenation_acceptance", float("nan")))
    if (
        not np.isfinite(acceptance)
        or acceptance < settings.rejuvenation_acceptance_min
        or acceptance > settings.rejuvenation_acceptance_max
    ):
        vetoes.append("aggregate_rejuvenation_acceptance_outside_interval")
    nominated = bool(not vetoes)
    return {
        "phase2ac_candidate_nominated": nominated,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "terminal_pre_final_resampling_ess_ratio": terminal_ess_ratio,
        "terminal_pre_final_resampling_max_weight": terminal_max_weight,
        "minimum_adaptive_post_temperature_ess_ratio": min_post_temperature_ess_ratio,
        "unique_ancestor_fraction": unique_fraction,
        "aggregate_rejuvenation_acceptance": acceptance,
        "interpretation": (
            "candidate nomination for Phase 2AD independent replication only"
            if nominated
            else "no valid patched sequential-reference nomination"
        ),
    }


def min_post_temperature_ess(sequential: Mapping[str, Any]) -> float:
    rows = sequential.get("stage_rows", ())
    values = []
    for row in rows:
        summary = row.get("pre_resample_weight_summary", {})
        value = summary.get("ess_ratio")
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    return float(min(values)) if values else float("nan")


def weighted_moments(samples: Any, weights: Any) -> Mapping[str, Any]:
    sample_array = np.asarray(samples, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if (
        sample_array.ndim != 2
        or weight_array.shape != (sample_array.shape[0],)
        or not np.all(np.isfinite(sample_array))
        or not np.all(np.isfinite(weight_array))
        or not np.isclose(float(np.sum(weight_array)), 1.0, rtol=1.0e-8, atol=1.0e-8)
    ):
        return {
            "computed": False,
            "vetoes": ("weighted_moment_inputs_invalid",),
        }
    mean = np.sum(weight_array[:, np.newaxis] * sample_array, axis=0)
    centered = sample_array - mean
    variance = np.sum(weight_array[:, np.newaxis] * np.square(centered), axis=0)
    return {
        "computed": True,
        "vetoes": (),
        "mean_u_new": mean,
        "std_u_new": np.sqrt(np.maximum(variance, 0.0)),
        "second_moment_variance_u_new": variance,
    }


def finite_summary(values: Any) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return {
        "shape": array.shape,
        "finite_count": int(np.sum(finite)),
        "nonfinite_count": int(np.sum(~finite)),
        "min": float(np.min(array[finite])) if np.any(finite) else None,
        "max": float(np.max(array[finite])) if np.any(finite) else None,
        "mean": float(np.mean(array[finite])) if np.any(finite) else None,
        "max_abs": float(np.max(np.abs(array[finite]))) if np.any(finite) else None,
    }


def environment_payload() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "tf_physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "tf_logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    }


def git_payload() -> Mapping[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "dirty_line_count": len(lines),
        "dirty_preview": lines[:20],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    sequential = payload.get("sequential_reference", {})
    gate = payload.get("phase2ac_gate", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2AC - Sequential Resampling Repair",
        "",
        "## Decision",
        "",
        f"- phase2ac_sequential_resampling_repair_passed: `{decision['phase2ac_sequential_resampling_repair_passed']}`",
        f"- candidate_nominated_for_phase2ad_replication: `{decision['candidate_nominated_for_phase2ad_replication']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- terminal_beta: `{decision['terminal_beta']}`",
        f"- stage_count: `{decision['stage_count']}`",
        f"- terminal_pre_final_resampling_ess_ratio: `{decision['terminal_pre_final_resampling_ess_ratio']}`",
        f"- terminal_pre_final_resampling_max_weight: `{decision['terminal_pre_final_resampling_max_weight']}`",
        f"- unique_ancestor_fraction: `{decision['unique_ancestor_fraction']}`",
        f"- aggregate_rejuvenation_acceptance: `{decision['aggregate_rejuvenation_acceptance']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Gate",
        "",
        f"- gate: `{gate}`",
        "",
        "## Sequential Reference",
        "",
        f"- computed: `{sequential.get('computed')}`",
        f"- vetoes: `{sequential.get('vetoes')}`",
        f"- terminal_pre_final_resampling_summary: `{sequential.get('terminal_pre_final_resampling_summary')}`",
        f"- final_weight_summary: `{sequential.get('final_weight_summary')}`",
        f"- terminal_weighted_moments: `{sequential.get('terminal_weighted_moments')}`",
        "",
        "## Inference Status",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key, value in payload["inference_status"].items():
        lines.append(f"| {key} | {value} |")
    manifest = payload.get("run_manifest", {})
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| command | `{manifest.get('command')}` |",
            f"| git | `{manifest.get('git')}` |",
            f"| environment | `{manifest.get('environment')}` |",
            f"| conda_env | `{manifest.get('conda_env')}` |",
            f"| cpu_gpu_status | {manifest.get('cpu_gpu_status')} |",
            f"| jit_compile | `{manifest.get('jit_compile')}` |",
            f"| tf32_mode | {manifest.get('tf32_mode')} |",
            f"| random_seeds | `{manifest.get('random_seeds')}` |",
            f"| wall_time_seconds | `{manifest.get('wall_time_seconds')}` |",
            f"| output_artifacts | `{manifest.get('output_artifacts')}` |",
            f"| plan_file | `{manifest.get('plan_file')}` |",
            f"| subplan_file | `{manifest.get('subplan_file')}` |",
            f"| result_file | `{manifest.get('result_file')}` |",
        ]
    )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if tf.is_tensor(value):
        return json_ready(value.numpy())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase2s-json", type=Path, default=DEFAULT_PHASE2S_PATH)
    parser.add_argument("--phase2t-json", type=Path, default=DEFAULT_PHASE2T_PATH)
    parser.add_argument("--phase2u-json", type=Path, default=DEFAULT_PHASE2U_PATH)
    parser.add_argument("--phase2v-json", type=Path, default=DEFAULT_PHASE2V_PATH)
    parser.add_argument("--phase2w-json", type=Path, default=DEFAULT_PHASE2W_PATH)
    parser.add_argument("--phase2x-json", type=Path, default=DEFAULT_PHASE2X_PATH)
    parser.add_argument("--phase2y-json", type=Path, default=DEFAULT_PHASE2Y_PATH)
    parser.add_argument("--phase2z-json", type=Path, default=DEFAULT_PHASE2Z_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2ac_sequential_resampling_repair(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
        load_json(args.phase2v_json),
        load_json(args.phase2w_json),
        load_json(args.phase2x_json),
        load_json(args.phase2y_json),
        load_json(args.phase2z_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
        "phase2v_json": str(args.phase2v_json),
        "phase2w_json": str(args.phase2w_json),
        "phase2x_json": str(args.phase2x_json),
        "phase2y_json": str(args.phase2y_json),
        "phase2z_json": str(args.phase2z_json),
        "phase2ab_baseline_json": str(DEFAULT_PHASE2AB_PATH),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
