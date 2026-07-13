"""Phase 2U retuned MAP-local scalar SSL-LSTM HMC screen.

This diagnostic runs a CPU-hidden fixed-grid HMC finite/acceptance screen in
the Phase 2S MAP-local coordinate.  It selects the first predeclared candidate
that passes hard vetoes and the acceptance envelope for a later reviewed longer
screen.  It does not claim posterior correctness, HMC readiness, convergence,
zero divergences, GPU/XLA readiness, default readiness, or Zhao-Cui source
faithfulness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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

from bayesfilter.inference import (  # noqa: E402
    FullChainHMCConfig,
    LatentAffineBatchValueScoreAdapter,
    LatentAffineHMCTransform,
    ValueScoreCapability,
    run_full_chain_tfp_hmc,
    stable_adapter_signature,
)


SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md"
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
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md"
)
GEOMETRY_MODULE_PATH = ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py"
PHASE2U_CANDIDATES = (
    {"num_leapfrog_steps": 2, "step_size": 0.785},
    {"num_leapfrog_steps": 4, "step_size": 0.3925},
    {"num_leapfrog_steps": 8, "step_size": 0.19625},
    {"num_leapfrog_steps": 16, "step_size": 0.098125},
)
NONCLAIMS = (
    "Phase 2U finite/acceptance MAP-local HMC screen only",
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
class Phase2UScreenSettings:
    """Fixed Phase 2U retuned MAP-local screen settings."""

    candidate_grid: tuple[tuple[int, float], ...] = tuple(
        (int(row["num_leapfrog_steps"]), float(row["step_size"]))
        for row in PHASE2U_CANDIDATES
    )
    num_results: int = 64
    num_burnin_steps: int = 4
    seed_base: tuple[int, int] = (20260709, 6301)
    acceptance_lower_exclusive: float = 0.05
    acceptance_upper_exclusive: float = 0.99

    def __post_init__(self) -> None:
        grid = []
        for leapfrogs, step_size in self.candidate_grid:
            leapfrogs = int(leapfrogs)
            step_size = float(step_size)
            if leapfrogs <= 0:
                raise ValueError("candidate leapfrog count must be positive")
            if not np.isfinite(step_size) or step_size <= 0.0:
                raise ValueError("candidate step size must be positive finite")
            grid.append((leapfrogs, step_size))
        object.__setattr__(self, "candidate_grid", tuple(grid))
        for name in ("num_results", "num_burnin_steps"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        seed_base = tuple(int(item) for item in self.seed_base)
        if len(seed_base) != 2:
            raise ValueError("seed_base must contain exactly two integers")
        object.__setattr__(self, "seed_base", seed_base)
        lower = float(self.acceptance_lower_exclusive)
        upper = float(self.acceptance_upper_exclusive)
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("acceptance envelope must satisfy 0 <= lower < upper <= 1")
        object.__setattr__(self, "acceptance_lower_exclusive", lower)
        object.__setattr__(self, "acceptance_upper_exclusive", upper)

    def seed_for_candidate(self, candidate_index: int) -> tuple[int, int]:
        return (self.seed_base[0], self.seed_base[1] + int(candidate_index))

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate_grid": [
                {
                    "num_leapfrog_steps": leapfrogs,
                    "step_size": step_size,
                    "trajectory_length_L_times_epsilon": leapfrogs * step_size,
                }
                for leapfrogs, step_size in self.candidate_grid
            ],
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "seed_base": self.seed_base,
            "seeds": [
                self.seed_for_candidate(index)
                for index, _candidate in enumerate(self.candidate_grid)
            ],
            "chain_execution_mode": "eager",
            "use_xla": False,
            "adaptation_policy": "fixed_kernel_no_adaptation",
            "acceptance_envelope": {
                "lower_exclusive": self.acceptance_lower_exclusive,
                "upper_exclusive": self.acceptance_upper_exclusive,
            },
            "selection_policy": "first_passing_candidate_in_predeclared_order",
        }


class ScalarFilteringFreeParameterAdapter:
    """Adapter for the scalar filtering target in free-parameter coordinates."""

    def __init__(self, target: Any, *, evidence_path: str) -> None:
        self.target = target
        self.parameter_dim = len(target.free_indices)
        self.target_scope = "scalar_ssl_lstm:svd_ukf_filtering_geometry:phase2u_base_free"
        self.evidence_path = str(evidence_path)
        self.free_parameter_names = tuple(str(name) for name in target.free_parameter_names)

    def adapter_signature(self) -> str:
        payload = {
            "target_scope": self.target_scope,
            "parameter_dim": self.parameter_dim,
            "free_parameter_names": self.free_parameter_names,
            "horizon": int(self.target.config.horizon),
            "filter_name": self.target.settings.filter_name,
            "evidence_path": self.evidence_path,
        }
        return stable_json_hash(payload)

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            full_chain_xla_diagnostic_ready=False,
            runtime_backend="docs.benchmarks.scalar_filtering_hmc_validation_phase2u",
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=(
                "Phase 2U base target only",
                "CPU-hidden non-XLA debug/reference execution",
                "no HMC convergence claim",
                "no posterior correctness claim",
            ),
        )

    def log_prob_and_grad(self, free_values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = tf.convert_to_tensor(free_values, dtype=tf.float64)
        if values.shape.rank == 1:
            return self.target.value_and_score(values)
        if values.shape.rank == 2:
            result_values = []
            result_scores = []
            for index in range(int(values.shape[0])):
                value, score = self.target.value_and_score(values[index])
                result_values.append(tf.convert_to_tensor(value, dtype=tf.float64))
                result_scores.append(tf.convert_to_tensor(score, dtype=tf.float64))
            return tf.stack(result_values), tf.stack(result_scores)
        raise ValueError("free_values must have rank 1 or rank 2")


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(json_ready(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_geometry_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_geometry_phase2u_reuse",
        GEOMETRY_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scalar filtering geometry module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_phase2u_retuned_map_local_hmc_screen(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    settings: Phase2UScreenSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2UScreenSettings() if settings is None else settings
    start = time.perf_counter()
    precondition = validate_handoff_artifacts(phase2s_payload, phase2t_payload, cfg)
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    vetoes = list(precondition.get("vetoes", ()))
    if not vetoes:
        adapter, adapter_audit = build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2u_adapter_not_built")

    rows = []
    if adapter is not None and not adapter_audit.get("vetoes"):
        for candidate_index, (leapfrogs, step_size) in enumerate(cfg.candidate_grid):
            rows.append(
                run_candidate(
                    adapter,
                    settings=cfg,
                    candidate_index=candidate_index,
                    num_leapfrog_steps=leapfrogs,
                    step_size=step_size,
                )
            )
    else:
        vetoes.append("candidate_screen_not_run")

    candidate_gate = evaluate_candidate_gate(rows, cfg)
    vetoes.extend(candidate_gate.get("vetoes", ()))
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = bool(not unique_vetoes and candidate_gate.get("selected_candidate") is not None)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2u_retuned_map_local_screen",
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
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "candidate_rows": rows,
        "candidate_gate": candidate_gate,
        "telemetry_policy": telemetry_policy_payload(rows, phase2s_payload, phase2t_payload),
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2u_retuned_map_local_hmc_screen_passed": passed,
            "vetoes": unique_vetoes,
            "selected_candidate": candidate_gate.get("selected_candidate"),
            "passed_candidate_count": candidate_gate.get("passed_candidate_count"),
            "candidate_count": candidate_gate.get("candidate_count"),
            "zero_divergence_claim_made": False,
            "viable_for_longer_map_local_selected_kernel_subplan": passed,
            "next_justified_action": (
                "write Phase 2U result and draft/review longer selected-kernel MAP-local screen subplan"
                if passed
                else "write Phase 2U blocker or narrower tuning/localization repair result"
            ),
        },
        "metric_roles": {
            "phase2u_retuned_map_local_hmc_screen_passed": "primary_phase2u_pass_fail",
            "candidate_hard_vetoes": "hard_veto_evidence",
            "retained_sample_finiteness": "hard_veto_evidence",
            "target_log_prob_finiteness": "hard_veto_evidence",
            "log_accept_ratio_finiteness": "hard_veto_evidence",
            "acceptance_envelope": "phase2u_candidate_selection_gate",
            "native_divergence": "hard_veto_if_available_positive; unavailable is not zero divergences",
            "acceptance_values": "descriptive_after_screen",
            "log_accept_tail_values": "descriptive_only_after_finiteness",
            "target_log_prob_range": "descriptive_only",
            "runtime": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "native_divergence": telemetry_policy_payload(rows, phase2s_payload, phase2t_payload)[
                "native_divergence_interpretation"
            ],
            "zero_divergence_claim": "not made",
            "statistically_supported_ranking": (
                "none; fixed short grid with no uncertainty analysis"
            ),
            "descriptive_only_differences": (
                "per-candidate acceptance, target-log-prob range, log-accept range, "
                "sample range, and runtime"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed; Phase 2U finite/acceptance screen only",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "reviewed longer selected-kernel MAP-local screen"
                if passed
                else "reviewed narrower tuning/localization repair"
            ),
        },
        "decision_table": {
            "decision": "Phase 2U retuned MAP-local fixed-kernel HMC screen",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "A short CPU-hidden grid can nominate one fixed kernel for a later "
                "longer screen, but cannot establish convergence, posterior correctness, "
                "zero divergences when native telemetry is unavailable, or readiness."
            ),
            "next_justified_action": (
                "draft/review longer selected-kernel MAP-local screen"
                if passed
                else "write blocker or narrower repair"
            ),
            "what_is_not_being_concluded": (
                "No posterior correctness, HMC readiness, convergence, zero-divergence "
                "claim when native divergence is unavailable, sampler superiority, "
                "statistical ranking, GPU/XLA readiness, default readiness, or "
                "Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 720 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": cfg.payload()["seeds"],
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
                "A passing candidate may reflect a short-chain local finite/acceptance "
                "screen near the MAP-local center, not posterior validity."
            ),
            "what_would_overturn": (
                "Longer selected-kernel screen failures, reference disagreement, "
                "nonfinite telemetry, positive native divergence when available, or "
                "GPU/XLA mismatch under a reviewed later phase."
            ),
            "weakest_evidence": (
                "One short CPU-hidden grid with no uncertainty analysis and no native "
                "divergence availability guarantee."
            ),
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_handoff_artifacts(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    settings: Phase2UScreenSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2UScreenSettings() if settings is None else settings
    vetoes: list[str] = []
    if (
        phase2s_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1"
    ):
        vetoes.append("phase2s_schema_mismatch")
    if phase2s_payload.get("decision", {}).get("phase2s_geometry_centering_repair_passed") is not True:
        vetoes.append("phase2s_decision_not_passed")
    if phase2s_payload.get("decision", {}).get("vetoes"):
        vetoes.append("phase2s_vetoes_present")
    if (
        phase2t_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2t_map_local_reference_handoff.v1"
    ):
        vetoes.append("phase2t_schema_mismatch")
    if phase2t_payload.get("decision", {}).get("phase2t_map_local_reference_handoff_passed") is not True:
        vetoes.append("phase2t_decision_not_passed")
    if phase2t_payload.get("decision", {}).get("vetoes"):
        vetoes.append("phase2t_vetoes_present")
    next_contract = phase2t_payload.get("phase2u_next_subplan_contract", {})
    expected_grid = [
        {
            "num_leapfrog_steps": int(leapfrogs),
            "step_size": float(step_size),
            "trajectory_length_L_times_epsilon": float(leapfrogs * step_size),
        }
        for leapfrogs, step_size in cfg.candidate_grid
    ]
    actual_grid = next_contract.get("candidate_grid", ())
    if len(actual_grid) != len(expected_grid):
        vetoes.append("phase2t_candidate_grid_count_mismatch")
    else:
        for index, expected in enumerate(expected_grid):
            actual = actual_grid[index]
            if int(actual.get("num_leapfrog_steps", -1)) != expected["num_leapfrog_steps"]:
                vetoes.append(f"phase2t_candidate_{index}_leapfrog_mismatch")
            if abs(float(actual.get("step_size", np.nan)) - expected["step_size"]) > 1.0e-12:
                vetoes.append(f"phase2t_candidate_{index}_step_size_mismatch")
            if (
                abs(
                    float(actual.get("trajectory_length_L_times_epsilon", np.nan))
                    - expected["trajectory_length_L_times_epsilon"]
                )
                > 1.0e-12
            ):
                vetoes.append(f"phase2t_candidate_{index}_trajectory_length_mismatch")
    if next_contract.get("selection_policy_predeclared") is not True:
        vetoes.append("phase2t_selection_policy_not_predeclared")
    if next_contract.get("all_trajectory_lengths_equal_1p57") is not True:
        vetoes.append("phase2t_trajectory_length_contract_failed")
    matrix_checks = validate_map_local_handoff_matrices(phase2s_payload)
    vetoes.extend(matrix_checks.get("vetoes", ()))
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2s_decision": phase2s_payload.get("decision", {}),
        "phase2t_decision": phase2t_payload.get("decision", {}),
        "phase2t_next_contract": next_contract,
        "matrix_checks": matrix_checks,
    }


def validate_map_local_handoff_matrices(
    phase2s_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    handoff = phase2s_payload.get("map_local_handoff", {})
    center = _vector(handoff.get("center_free_parameter_values"), 4, "center", vetoes)
    scale = _vector(handoff.get("scale"), 4, "scale", vetoes)
    precision_z = _matrix(handoff.get("precision_z"), 4, "precision_z", vetoes)
    covariance_z = _matrix(handoff.get("covariance_z"), 4, "covariance_z", vetoes)
    factor_z = _matrix(handoff.get("factor_z"), 4, "factor_z", vetoes, symmetrize=False)
    precision_theta = _matrix(handoff.get("precision_theta"), 4, "precision_theta", vetoes)
    covariance_theta = _matrix(handoff.get("covariance_theta"), 4, "covariance_theta", vetoes)
    diagnostics: dict[str, Any] = {}
    if scale.shape == (4,) and np.any(scale <= 0.0):
        vetoes.append("scale_nonpositive")
    if vetoes:
        return {"passed": False, "vetoes": tuple(dict.fromkeys(vetoes)), "diagnostics": diagnostics}

    identity_error = float(np.max(np.abs(precision_z @ covariance_z - np.eye(4))))
    factor_error = float(np.max(np.abs(factor_z @ factor_z.T - covariance_z)))
    inv_scale = 1.0 / scale
    expected_precision_theta = inv_scale[:, np.newaxis] * precision_z * inv_scale[np.newaxis, :]
    expected_covariance_theta = scale[:, np.newaxis] * covariance_z * scale[np.newaxis, :]
    precision_theta_error = float(np.max(np.abs(expected_precision_theta - precision_theta)))
    covariance_theta_error = float(np.max(np.abs(expected_covariance_theta - covariance_theta)))
    free_factor = np.diag(scale) @ factor_z
    diagnostics.update(
        {
            "center_free_parameter_values": center,
            "scale": scale,
            "precision_z_covariance_z_identity_max_abs_error": identity_error,
            "factor_z_reconstructs_covariance_z_max_abs_error": factor_error,
            "precision_theta_scale_transform_max_abs_error": precision_theta_error,
            "covariance_theta_scale_transform_max_abs_error": covariance_theta_error,
            "precision_z_eigen_summary": eigen_summary(precision_z),
            "covariance_z_eigen_summary": eigen_summary(covariance_z),
            "precision_theta_eigen_summary": eigen_summary(precision_theta),
            "covariance_theta_eigen_summary": eigen_summary(covariance_theta),
            "adapter_factor": free_factor,
            "adapter_factor_orientation": "row_right_transpose",
            "adapter_coordinate_formula": (
                "free = center_free_parameter_values + u_new @ (diag(scale) @ factor_z).T"
            ),
        }
    )
    for name, error in (
        ("precision_z_covariance_z_identity", identity_error),
        ("factor_z_reconstructs_covariance_z", factor_error),
        ("precision_theta_scale_transform", precision_theta_error),
        ("covariance_theta_scale_transform", covariance_theta_error),
    ):
        if not np.isfinite(error) or error > 1.0e-8:
            vetoes.append(f"{name}_failed")
    for name in (
        "precision_z_eigen_summary",
        "covariance_z_eigen_summary",
        "precision_theta_eigen_summary",
        "covariance_theta_eigen_summary",
    ):
        if not _summary_spd_condition(diagnostics[name], cap=1.0e5):
            vetoes.append(f"{name}_not_spd_or_condition_above_cap")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "diagnostics": diagnostics,
    }


def build_phase2u_adapter(
    phase2s_payload: Mapping[str, Any],
) -> tuple[Any | None, Mapping[str, Any]]:
    vetoes: list[str] = []
    matrices = validate_map_local_handoff_matrices(phase2s_payload)
    vetoes.extend(matrices.get("vetoes", ()))
    diagnostics = matrices.get("diagnostics", {})
    if vetoes:
        return None, {"built": False, "vetoes": tuple(dict.fromkeys(vetoes)), "matrix_checks": matrices}

    module = load_geometry_module()
    settings = module.default_settings()
    target = module.build_filtering_geometry_target(settings)
    target_scale = np.asarray(target.scale.numpy(), dtype=float)
    scale = np.asarray(diagnostics["scale"], dtype=float)
    center = np.asarray(diagnostics["center_free_parameter_values"], dtype=float)
    free_names = tuple(str(name) for name in target.free_parameter_names)
    target_value = None
    target_score_norm = None
    if not np.allclose(target_scale, scale, rtol=0.0, atol=1.0e-12):
        vetoes.append("rebuilt_target_scale_mismatch")
    recorded_names = tuple(
        str(name)
        for name in phase2s_payload.get("diagnostic_points", {})
        .get("free_parameter_names", ())
    )
    if recorded_names and recorded_names != free_names:
        vetoes.append("rebuilt_target_parameter_names_mismatch")
    try:
        value, score = target.value_and_score(tf.constant(center, dtype=tf.float64))
        target_value = float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())
        score_np = np.asarray(tf.convert_to_tensor(score, dtype=tf.float64).numpy(), dtype=float)
        target_score_norm = float(np.linalg.norm(score_np))
        if not np.isfinite(target_value):
            vetoes.append("map_local_center_target_value_nonfinite")
        if score_np.shape != (4,) or not np.all(np.isfinite(score_np)):
            vetoes.append("map_local_center_target_score_nonfinite")
    except Exception as exc:  # noqa: BLE001 - fail-closed artifact audit.
        vetoes.append(f"map_local_center_target_replay_exception_{type(exc).__name__}")

    if vetoes:
        return None, {
            "built": False,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "matrix_checks": matrices,
            "target_replay": {
                "value": target_value,
                "score_norm": target_score_norm,
            },
        }

    base_adapter = ScalarFilteringFreeParameterAdapter(target, evidence_path=SUBPLAN_PATH)
    transform = LatentAffineHMCTransform(
        center=center,
        factor=np.asarray(diagnostics["adapter_factor"], dtype=float),
        factor_orientation="row_right_transpose",
        covariance_provenance="phase2s_map_local_covariance_z_cholesky_composed_with_target_scale",
        log_jacobian_convention="constant_omitted",
        nonclaims=(
            "Phase 2U MAP-local affine preconditioner only",
            "no posterior convergence claim",
            "no sampler readiness claim",
        ),
    )
    adapter = LatentAffineBatchValueScoreAdapter(
        base_adapter=base_adapter,
        transform=transform,
        target_scope="scalar_ssl_lstm:svd_ukf_filtering_geometry:phase2u_map_local_u_new",
        evidence_path=SUBPLAN_PATH,
        xla_hmc_ready=False,
        full_chain_xla_diagnostic_ready=False,
        nonclaims=(
            "Phase 2U MAP-local finite/acceptance HMC screen only",
            "TFP HMC coordinate is u_new",
            "dense mass is represented by a fixed affine transform",
            "no HMC convergence claim",
            "no posterior correctness claim",
        ),
    )
    return adapter, json_ready(
        {
            "built": True,
            "vetoes": (),
            "matrix_checks": matrices,
            "target_replay": {
                "map_local_center_value": target_value,
                "map_local_center_score_norm": target_score_norm,
            },
            "coordinate_contract": {
                "hmc_coordinate": "u_new",
                "base_adapter_coordinate": "free parameter values",
                "adapter_formula": (
                    "free = center_free_parameter_values + u_new @ (diag(scale) @ factor_z).T"
                ),
                "factor_orientation": "row_right_transpose",
                "free_parameter_names": free_names,
            },
            "adapter_signature": stable_adapter_signature(adapter),
            "base_adapter_signature": stable_adapter_signature(base_adapter),
        }
    )


def run_candidate(
    adapter: Any,
    *,
    settings: Phase2UScreenSettings,
    candidate_index: int,
    num_leapfrog_steps: int,
    step_size: float,
) -> Mapping[str, Any]:
    start = time.perf_counter()
    hard_vetoes: list[str] = []
    error_message = None
    initial_state = tf.zeros((adapter.parameter_dim,), dtype=tf.float64)
    initial_value = None
    initial_score = None
    result = None
    diagnostics: Mapping[str, Any] = {}
    metadata: Mapping[str, Any] = {}
    samples_summary: Mapping[str, Any] = {}
    trace_summary: Mapping[str, Any] = {}
    try:
        initial_value_tensor, initial_score_tensor = adapter.log_prob_and_grad(initial_state)
        initial_value = float(tf.convert_to_tensor(initial_value_tensor, dtype=tf.float64).numpy())
        initial_score = np.asarray(
            tf.reshape(tf.convert_to_tensor(initial_score_tensor, dtype=tf.float64), [-1]).numpy(),
            dtype=float,
        )
        if not np.isfinite(initial_value):
            hard_vetoes.append("initial_target_value_nonfinite")
        if initial_score.shape != (adapter.parameter_dim,):
            hard_vetoes.append("initial_target_score_shape_mismatch")
        elif not np.all(np.isfinite(initial_score)):
            hard_vetoes.append("initial_target_score_nonfinite")
        config = FullChainHMCConfig(
            num_results=settings.num_results,
            num_burnin_steps=settings.num_burnin_steps,
            step_size=float(step_size),
            num_leapfrog_steps=int(num_leapfrog_steps),
            seed=settings.seed_for_candidate(candidate_index),
            use_xla=False,
            trace_policy="standard",
            adaptation_policy="fixed_kernel_no_adaptation",
            target_scope=adapter.target_scope,
            chain_execution_mode="eager",
        )
        result = run_full_chain_tfp_hmc(adapter, initial_state, config)
    except Exception as exc:  # noqa: BLE001 - fail-closed HMC artifact.
        error_message = f"{type(exc).__name__}: {exc}"
        hard_vetoes.append("hmc_runtime_exception")

    if result is not None:
        diagnostics = dict(result.diagnostics)
        metadata = dict(result.metadata)
        samples_summary = summarize_samples(result.samples)
        trace_summary = summarize_trace(result.trace)
        if int(samples_summary.get("nonfinite_sample_count", 1)) != 0:
            hard_vetoes.append("nonfinite_retained_samples")
        log_accept = trace_summary.get("log_accept_ratio", {})
        if (
            not isinstance(log_accept, Mapping)
            or int(log_accept.get("finite_count", 0)) <= 0
            or int(log_accept.get("nonfinite_count", 1)) != 0
        ):
            hard_vetoes.append("nonfinite_log_accept_ratio")
        target = trace_summary.get("target_log_prob", {})
        if not isinstance(target, Mapping) or target.get("finite") is not True:
            hard_vetoes.append("nonfinite_target_log_prob_trace")
        native = trace_summary.get("native_divergence", {})
        if isinstance(native, Mapping) and native.get("available") and int(native.get("count", 0)) > 0:
            hard_vetoes.append("native_divergence_detected")

    acceptance = None
    if isinstance(trace_summary, Mapping):
        acceptance = trace_summary.get("acceptance_rate")
    status = "passed_hard_vetoes" if not hard_vetoes else "failed_hard_vetoes"
    return json_ready(
        {
            "candidate_index": int(candidate_index),
            "num_leapfrog_steps": int(num_leapfrog_steps),
            "step_size": float(step_size),
            "trajectory_length_L_times_epsilon": float(num_leapfrog_steps * step_size),
            "seed": settings.seed_for_candidate(candidate_index),
            "status": status,
            "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
            "acceptance_rate": acceptance,
            "acceptance_in_envelope": (
                None
                if acceptance is None
                else bool(
                    settings.acceptance_lower_exclusive
                    < float(acceptance)
                    < settings.acceptance_upper_exclusive
                )
            ),
            "runtime_seconds": float(time.perf_counter() - start),
            "initial": {
                "u_new": [0.0] * int(adapter.parameter_dim),
                "value": initial_value,
                "score": None if initial_score is None else initial_score,
                "score_norm": None if initial_score is None else float(np.linalg.norm(initial_score)),
            },
            "hmc_error": error_message,
            "diagnostics": diagnostics,
            "metadata": metadata,
            "samples_summary": samples_summary,
            "trace_summary": trace_summary,
            "metric_roles": {
                "status": "candidate_hard_veto_screen",
                "hard_vetoes": "hard_veto_evidence",
                "acceptance_in_envelope": "phase2u_candidate_selection_gate",
                "retained_sample_finiteness": "hard_veto_evidence",
                "target_log_prob_finiteness": "hard_veto_evidence",
                "log_accept_ratio_finiteness": "hard_veto_evidence",
                "acceptance_rate": "descriptive_after_screen",
                "native_divergence": "hard_veto_if_available_positive; unavailable is not zero divergences",
            },
            "nonclaims": (
                "candidate finite/acceptance screen only",
                "not HMC convergence evidence",
                "not posterior correctness evidence",
                "not a tuned-kernel claim",
            ),
        }
    )


def evaluate_candidate_gate(
    rows: Sequence[Mapping[str, Any]],
    settings: Phase2UScreenSettings,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    selected = None
    passed_rows = []
    if len(rows) != len(settings.candidate_grid):
        vetoes.append("candidate_row_count_mismatch")
    for row in rows:
        index = int(row.get("candidate_index", -1))
        for hard_veto in row.get("hard_vetoes", ()):
            vetoes.append(f"candidate_{index}_{hard_veto}")
        if row.get("status") != "passed_hard_vetoes":
            vetoes.append(f"candidate_{index}_hard_veto_screen_failed")
        acceptance = row.get("acceptance_rate")
        if acceptance is None or not np.isfinite(float(acceptance)):
            vetoes.append(f"candidate_{index}_acceptance_missing_or_nonfinite")
            in_envelope = False
        else:
            in_envelope = bool(
                settings.acceptance_lower_exclusive
                < float(acceptance)
                < settings.acceptance_upper_exclusive
            )
        if row.get("status") == "passed_hard_vetoes" and in_envelope:
            passed_rows.append(row)
            if selected is None:
                selected = {
                    "candidate_index": index,
                    "num_leapfrog_steps": row.get("num_leapfrog_steps"),
                    "step_size": row.get("step_size"),
                    "trajectory_length_L_times_epsilon": row.get(
                        "trajectory_length_L_times_epsilon"
                    ),
                    "acceptance_rate": acceptance,
                    "selection_policy": "first_passing_candidate_in_predeclared_order",
                }
    if selected is None:
        vetoes.append("no_candidate_passed_acceptance_envelope")
    return {
        "passed": selected is not None and not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "selected_candidate": selected,
        "passed_candidate_count": len(passed_rows),
        "candidate_count": len(rows),
        "acceptance_rates": [
            row.get("acceptance_rate")
            for row in rows
        ],
        "acceptance_envelope": {
            "lower_exclusive": settings.acceptance_lower_exclusive,
            "upper_exclusive": settings.acceptance_upper_exclusive,
        },
        "selection_policy": "first_passing_candidate_in_predeclared_order",
    }


def summarize_samples(samples: Any) -> Mapping[str, Any]:
    array = np.asarray(tf.convert_to_tensor(samples, dtype=tf.float64).numpy(), dtype=float)
    finite = np.all(np.isfinite(array), axis=-1)
    return {
        "shape": array.shape,
        "finite_sample_count": int(np.sum(finite)),
        "nonfinite_sample_count": int(np.sum(~finite)),
        "first_u_new": array[0],
        "final_u_new": array[-1],
        "mean_u_new": np.mean(array, axis=0),
        "std_u_new": np.std(array, axis=0),
        "max_abs_u_new": float(np.max(np.abs(array))) if array.size else None,
    }


def summarize_trace(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    summary: dict[str, Any] = {}
    if "is_accepted" in trace:
        accepted = np.asarray(tf.convert_to_tensor(trace["is_accepted"]).numpy(), dtype=bool)
        summary["is_accepted"] = accepted
        summary["acceptance_rate"] = float(np.mean(accepted.astype(float))) if accepted.size else None
        summary["accepted_count"] = int(np.sum(accepted))
        summary["decision_count"] = int(accepted.size)
    if "log_accept_ratio" in trace:
        log_accept = np.asarray(
            tf.convert_to_tensor(trace["log_accept_ratio"], dtype=tf.float64).numpy(),
            dtype=float,
        )
        finite = np.isfinite(log_accept)
        summary["log_accept_ratio"] = {
            "values": log_accept,
            "finite_count": int(np.sum(finite)),
            "nonfinite_count": int(np.sum(~finite)),
            "max_abs_finite": (
                float(np.max(np.abs(log_accept[finite]))) if np.any(finite) else None
            ),
        }
    if "target_log_prob" in trace:
        target_log_prob = np.asarray(
            tf.convert_to_tensor(trace["target_log_prob"], dtype=tf.float64).numpy(),
            dtype=float,
        )
        finite = np.isfinite(target_log_prob)
        summary["target_log_prob"] = {
            "values": target_log_prob,
            "finite": bool(np.all(finite)),
            "min": float(np.min(target_log_prob)) if target_log_prob.size else None,
            "max": float(np.max(target_log_prob)) if target_log_prob.size else None,
        }
    if "divergence" in trace:
        divergence = np.asarray(tf.convert_to_tensor(trace["divergence"]).numpy(), dtype=bool)
        summary["native_divergence"] = {
            "available": True,
            "count": int(np.sum(divergence)),
            "values": divergence,
        }
    else:
        summary["native_divergence"] = {
            "available": False,
            "status": "not_exposed_by_kernel",
            "nonclaim": "unavailable native divergence telemetry is not zero divergences",
        }
    return json_ready(summary)


def telemetry_policy_payload(
    rows: Sequence[Mapping[str, Any]],
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    statuses = []
    positive_count = 0
    available_count = 0
    unavailable_count = 0
    for row in rows:
        native = row.get("trace_summary", {}).get("native_divergence", {})
        if isinstance(native, Mapping) and native.get("available") is True:
            available_count += 1
            count = int(native.get("count", 0))
            positive_count += count
            statuses.append("available")
        else:
            unavailable_count += 1
            if isinstance(native, Mapping):
                statuses.append(str(native.get("status", "unavailable")))
            else:
                statuses.append("unavailable")
    if positive_count > 0:
        interpretation = "positive native divergence detected"
    elif rows and available_count == len(rows):
        interpretation = "native divergence available with zero positive indicators"
    else:
        interpretation = (
            "native divergence unavailable for at least one candidate; unavailable is not zero divergences"
        )
    prior_statuses = (
        phase2t_payload.get("telemetry_policy", {}).get(
            "native_divergence_statuses",
            phase2s_payload.get("telemetry_policy", {}).get("native_divergence_statuses", ()),
        )
    )
    return {
        "native_divergence_statuses": statuses,
        "native_divergence_available_count": available_count,
        "native_divergence_unavailable_count": unavailable_count,
        "native_divergence_positive_count": positive_count,
        "native_divergence_interpretation": interpretation,
        "prior_native_divergence_statuses": prior_statuses,
        "zero_divergence_claim_made": False,
        "unavailable_native_divergence_is_zero_divergence": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def _vector(value: Any, dim: int, name: str, vetoes: list[str]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        vetoes.append(f"{name}_shape_or_finiteness_mismatch")
        return np.full(dim, np.nan)
    return array


def _matrix(
    value: Any,
    dim: int,
    name: str,
    vetoes: list[str],
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim, dim) or not np.all(np.isfinite(array)):
        vetoes.append(f"{name}_shape_or_finiteness_mismatch")
        return np.full((dim, dim), np.nan)
    if symmetrize:
        return 0.5 * (array + array.T)
    return array


def eigen_summary(matrix: Any) -> Mapping[str, Any]:
    values = np.linalg.eigvalsh(0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T))
    finite = bool(np.all(np.isfinite(values)))
    positive = bool(finite and np.min(values) > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": float(np.min(values)) if finite else float("nan"),
        "max": float(np.max(values)) if finite else float("nan"),
        "condition_number": float(np.max(values) / np.min(values)) if positive else float("inf"),
        "eigenvalues": tuple(float(value) for value in values),
    }


def _summary_spd_condition(summary: Mapping[str, Any], *, cap: float) -> bool:
    return bool(
        summary.get("finite") is True
        and summary.get("positive") is True
        and float(summary.get("condition_number", float("inf"))) <= float(cap) * (1.0 + 1.0e-8)
    )


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
    gate = payload.get("candidate_gate", {})
    rows = payload.get("candidate_rows", ())
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2U - Retuned MAP-Local Screen",
        "",
        "## Decision",
        "",
        f"- phase2u_retuned_map_local_hmc_screen_passed: `{decision['phase2u_retuned_map_local_hmc_screen_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- selected_candidate: `{decision['selected_candidate']}`",
        f"- passed_candidate_count: `{decision['passed_candidate_count']}` / `{decision['candidate_count']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Candidate Gate",
        "",
        f"- acceptance rates: `{gate.get('acceptance_rates')}`",
        f"- acceptance envelope: `{gate.get('acceptance_envelope')}`",
        f"- selection policy: {gate.get('selection_policy')}",
        "",
        "## Candidate Rows",
        "",
        "| candidate | L | step | trajectory | seed | status | acceptance | hard vetoes | native divergence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        native = row.get("trace_summary", {}).get("native_divergence")
        lines.append(
            f"| {row.get('candidate_index')} | {row.get('num_leapfrog_steps')} | "
            f"{row.get('step_size')} | {row.get('trajectory_length_L_times_epsilon')} | "
            f"{row.get('seed')} | {row.get('status')} | {row.get('acceptance_rate')} | "
            f"{', '.join(row.get('hard_vetoes', ())) or 'none'} | {native} |"
        )
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| field | value |",
            "| --- | --- |",
        ]
    )
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
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2u_retuned_map_local_hmc_screen(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
