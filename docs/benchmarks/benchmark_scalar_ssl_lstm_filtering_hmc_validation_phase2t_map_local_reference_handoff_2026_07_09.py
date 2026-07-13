"""Phase 2T MAP-local reference handoff diagnostic.

This artifact validates the Phase 2S MAP-local quadratic handoff and drafts
the next retuned fixed-kernel HMC screen boundary.  It does not run HMC and
does not claim posterior correctness, HMC readiness, convergence, zero
divergences, GPU/XLA readiness, default readiness, or Zhao-Cui source
faithfulness.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2t_map_local_reference_handoff.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-result-2026-07-09.md"
)
NEXT_SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md"
)
DEFAULT_PHASE2S_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md"
)
PHASE2U_CANDIDATES = (
    {"num_leapfrog_steps": 2, "step_size": 0.785},
    {"num_leapfrog_steps": 4, "step_size": 0.3925},
    {"num_leapfrog_steps": 8, "step_size": 0.19625},
    {"num_leapfrog_steps": 16, "step_size": 0.098125},
)
NONCLAIMS = (
    "Phase 2T MAP-local reference handoff diagnostic only",
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


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2t_map_local_reference_handoff(
    phase2s_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_phase2s_payload(phase2s_payload)
    matrices = validate_map_local_matrices(phase2s_payload)
    replay = validate_target_replay(phase2s_payload)
    old_projection = old_geometry_projection_diagnostic(phase2s_payload)
    next_plan = phase2u_next_subplan_contract()
    gate = evaluate_phase2t_gate(precondition, matrices, replay, old_projection, next_plan)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2t_map_local_reference_handoff",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": phase2s_payload.get("target_scope"),
        "settings": {
            "matrix_identity_threshold": 1.0e-8,
            "condition_number_cap": 1.0e5,
            "phase2u_candidate_count": len(PHASE2U_CANDIDATES),
            "phase2u_acceptance_envelope": (0.05, 0.99),
            "phase2u_candidate_selection": "first_passing_candidate_in_predeclared_order",
        },
        "source_artifacts": {
            "phase2s_json": str(DEFAULT_PHASE2S_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "map_local_reference": matrices,
        "target_replay": replay,
        "old_geometry_summary_projection": old_projection,
        "phase2u_next_subplan_contract": next_plan,
        "telemetry_policy": telemetry_policy_payload(phase2s_payload),
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": gate["decision"],
        "metric_roles": {
            "phase2t_map_local_reference_handoff_passed": "primary_phase2t_pass_fail",
            "phase2s_artifact_valid": "hard_veto_evidence",
            "matrix_identity_checks": "hard_veto_evidence",
            "theta_z_transform_checks": "hard_veto_evidence",
            "map_candidate_target_replay": "hard_veto_evidence",
            "old_geometry_summary_projection": "explanatory_only_excluded_from_pass_fail",
            "phase2u_next_subplan_contract": "handoff_gate",
            "native_divergence_unavailable": "telemetry_availability_not_zero_divergences",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if gate["decision"]["phase2t_map_local_reference_handoff_passed"] else "failed",
            "statistically_supported_ranking": "none; no sampler run and no method comparison",
            "descriptive_only_differences": "matrix residuals and old-geometry projection diagnostics",
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "zero_divergence_claim": "not made",
            "next_evidence_needed": gate["decision"]["next_justified_action"],
        },
        "decision_table": gate["decision_table"],
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 180 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden artifact analysis",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": phase2s_payload.get("run_manifest", {}).get("random_seeds"),
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "subplan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_phase2s_payload(phase2s_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if (
        phase2s_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1"
    ):
        vetoes.append("phase2s_schema_mismatch")
    decision = phase2s_payload.get("decision", {})
    if decision.get("phase2s_geometry_centering_repair_passed") is not True:
        vetoes.append("phase2s_decision_not_passed")
    if decision.get("vetoes"):
        vetoes.append("phase2s_vetoes_present")
    initializer = phase2s_payload.get("initializer", {})
    if initializer.get("accepted") is not True or initializer.get("status") != "usable":
        vetoes.append("phase2s_initializer_not_usable")
    locator = initializer.get("locator_diagnostics", {})
    if locator.get("accepted_optimizer_position") is not True:
        vetoes.append("phase2s_locator_not_accepted")
    if locator.get("uses_optimizer_inverse_hessian") is not False:
        vetoes.append("phase2s_optimizer_inverse_hessian_boundary_failed")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2s_decision": decision,
        "map_candidate_role": initializer.get("map_candidate_role"),
    }


def validate_map_local_matrices(phase2s_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    vetoes: list[str] = []
    handoff = phase2s_payload.get("map_local_handoff", {})
    center = _vector(handoff.get("center_free_parameter_values"), 4, "center", vetoes)
    scale = _vector(handoff.get("scale"), 4, "scale", vetoes)
    precision_z = _matrix(handoff.get("precision_z"), 4, "precision_z", vetoes)
    covariance_z = _matrix(handoff.get("covariance_z"), 4, "covariance_z", vetoes)
    factor_z = _matrix(
        handoff.get("factor_z"),
        4,
        "factor_z",
        vetoes,
        symmetrize=False,
    )
    precision_theta = _matrix(handoff.get("precision_theta"), 4, "precision_theta", vetoes)
    covariance_theta = _matrix(handoff.get("covariance_theta"), 4, "covariance_theta", vetoes)
    diagnostics: dict[str, Any] = {}
    if vetoes:
        return {"passed": False, "vetoes": tuple(dict.fromkeys(vetoes)), "diagnostics": diagnostics}

    identity_error = float(np.max(np.abs(precision_z @ covariance_z - np.eye(4))))
    factor_error = float(np.max(np.abs(factor_z @ factor_z.T - covariance_z)))
    inv_scale = 1.0 / scale
    expected_precision_theta = inv_scale[:, np.newaxis] * precision_z * inv_scale[np.newaxis, :]
    expected_covariance_theta = scale[:, np.newaxis] * covariance_z * scale[np.newaxis, :]
    precision_theta_error = float(np.max(np.abs(expected_precision_theta - precision_theta)))
    covariance_theta_error = float(np.max(np.abs(expected_covariance_theta - covariance_theta)))
    diagnostics.update(
        {
            "center_free_parameter_values": center,
            "scale": scale,
            "reference_u_new_mean": np.zeros(4),
            "reference_u_new_covariance": np.eye(4),
            "coordinate_formula": handoff.get("coordinate_formula"),
            "precision_z_covariance_z_identity_max_abs_error": identity_error,
            "factor_z_reconstructs_covariance_z_max_abs_error": factor_error,
            "precision_theta_scale_transform_max_abs_error": precision_theta_error,
            "covariance_theta_scale_transform_max_abs_error": covariance_theta_error,
            "precision_z_eigen_summary": eigen_summary(precision_z),
            "covariance_z_eigen_summary": eigen_summary(covariance_z),
            "precision_theta_eigen_summary": eigen_summary(precision_theta),
            "covariance_theta_eigen_summary": eigen_summary(covariance_theta),
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
        summary = diagnostics[name]
        if not _summary_spd_condition(summary, cap=1.0e5):
            vetoes.append(f"{name}_not_spd_or_condition_above_cap")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "diagnostics": diagnostics,
        "nonclaims": (
            "local Gaussian reference in u_new only",
            "not exact posterior covariance",
            "not HMC readiness evidence",
        ),
    }


def validate_target_replay(phase2s_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = phase2s_payload.get("target_replay", {})
    vetoes: list[str] = []
    if replay.get("computed") is not True:
        vetoes.append("phase2s_target_replay_not_computed")
    values = replay.get("values", {})
    map_candidate = values.get("map_candidate", {})
    if map_candidate.get("status") != "finite":
        vetoes.append("map_candidate_target_replay_not_finite")
    if not np.isfinite(float(map_candidate.get("value", np.nan))):
        vetoes.append("map_candidate_target_value_nonfinite")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "map_candidate": map_candidate,
        "locator_position": values.get("locator_position"),
        "truth_free_center": values.get("truth_free_center"),
        "phase2_reference_mean_initial": values.get("phase2_reference_mean_initial"),
    }


def old_geometry_projection_diagnostic(phase2s_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    replay = phase2s_payload.get("target_replay", {}).get("values", {})
    old = replay.get("phase1r_pooled_hmc_mean")
    return {
        "computed": old is not None,
        "role": "old_geometry_summary_projected_for_diagnostics_only",
        "included_in_pass_fail": False,
        "promotion_criterion": False,
        "old_geometry_summary": old,
        "nonclaims": (
            "old Phase 1R summary was sampled under the old truth-free geometry",
            "not evidence that the MAP-local HMC sampler passes",
        ),
    }


def phase2u_next_subplan_contract() -> Mapping[str, Any]:
    candidates = []
    for row in PHASE2U_CANDIDATES:
        candidate = dict(row)
        candidate["trajectory_length_L_times_epsilon"] = (
            int(candidate["num_leapfrog_steps"]) * float(candidate["step_size"])
        )
        candidates.append(candidate)
    return {
        "next_subplan_path": NEXT_SUBPLAN_PATH,
        "candidate_grid": candidates,
        "candidate_grid_predeclared": True,
        "all_trajectory_lengths_equal_1p57": all(
            abs(row["trajectory_length_L_times_epsilon"] - 1.57) <= 1.0e-12
            for row in candidates
        ),
        "selection_policy": "first candidate in listed order that passes hard vetoes and acceptance envelope",
        "selection_policy_predeclared": True,
        "acceptance_envelope": {"lower_exclusive": 0.05, "upper_exclusive": 0.99},
        "hard_vetoes": (
            "runtime_error",
            "nonfinite_samples",
            "nonfinite_target_log_prob",
            "nonfinite_log_accept_ratio",
            "positive_native_divergence_when_available",
            "artifact_invalid",
        ),
        "native_divergence_policy": (
            "positive native divergence is a hard veto when available; unavailable native "
            "divergence is recorded as unavailable and is not zero-divergence evidence"
        ),
        "if_no_candidate_passes": "write blocker/repair result; do not proceed to GPU/XLA",
        "nonclaims": (
            "Phase 2U finite/acceptance retuning screen only",
            "not posterior correctness evidence",
            "not HMC readiness evidence",
        ),
    }


def evaluate_phase2t_gate(
    precondition: Mapping[str, Any],
    matrices: Mapping[str, Any],
    replay: Mapping[str, Any],
    old_projection: Mapping[str, Any],
    next_plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    vetoes.extend(precondition.get("vetoes", ()))
    vetoes.extend(matrices.get("vetoes", ()))
    vetoes.extend(replay.get("vetoes", ()))
    if old_projection.get("included_in_pass_fail") is not False:
        vetoes.append("old_geometry_projection_included_in_pass_fail")
    if old_projection.get("promotion_criterion") is not False:
        vetoes.append("old_geometry_projection_used_as_promotion")
    if next_plan.get("candidate_grid_predeclared") is not True:
        vetoes.append("phase2u_candidate_grid_not_predeclared")
    if next_plan.get("all_trajectory_lengths_equal_1p57") is not True:
        vetoes.append("phase2u_trajectory_length_contract_failed")
    if next_plan.get("selection_policy_predeclared") is not True:
        vetoes.append("phase2u_selection_policy_missing")
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = not unique_vetoes
    decision = {
        "phase2t_map_local_reference_handoff_passed": passed,
        "vetoes": unique_vetoes,
        "viable_for_phase2u_retuned_map_local_hmc_screen_subplan": passed,
        "zero_divergence_claim_made": False,
        "next_justified_action": (
            "draft and review Phase 2U retuned MAP-local fixed-kernel HMC screen subplan"
            if passed
            else "write Phase 2T blocker or narrower handoff repair"
        ),
    }
    return {
        "decision": decision,
        "decision_table": {
            "decision": "Phase 2T MAP-local reference handoff diagnostic",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "Matrix consistency and target replay justify only a next retuned HMC "
                "screen; they do not establish sampler validity or posterior correctness."
            ),
            "next_justified_action": decision["next_justified_action"],
            "what_is_not_being_concluded": (
                "No posterior correctness, HMC readiness, convergence, zero-divergence "
                "claim, sampler superiority, GPU/XLA readiness, default readiness, or "
                "Zhao-Cui source faithfulness."
            ),
        },
    }


def telemetry_policy_payload(phase2s_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    telemetry = phase2s_payload.get("telemetry_policy", {})
    return {
        "native_divergence_statuses": telemetry.get("native_divergence_statuses", ()),
        "native_divergence_interpretation": telemetry.get("native_divergence_interpretation"),
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
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
    diagnostics = payload.get("map_local_reference", {}).get("diagnostics", {})
    next_plan = payload.get("phase2u_next_subplan_contract", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2T - MAP-Local Handoff",
        "",
        "## Decision",
        "",
        f"- phase2t_map_local_reference_handoff_passed: `{decision['phase2t_map_local_reference_handoff_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- viable_for_phase2u_retuned_map_local_hmc_screen_subplan: `{decision['viable_for_phase2u_retuned_map_local_hmc_screen_subplan']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Matrix Checks",
        "",
        f"- precision_z covariance_z identity error: `{diagnostics.get('precision_z_covariance_z_identity_max_abs_error')}`",
        f"- factor_z covariance reconstruction error: `{diagnostics.get('factor_z_reconstructs_covariance_z_max_abs_error')}`",
        f"- precision theta scale-transform error: `{diagnostics.get('precision_theta_scale_transform_max_abs_error')}`",
        f"- covariance theta scale-transform error: `{diagnostics.get('covariance_theta_scale_transform_max_abs_error')}`",
        "",
        "## Phase 2U Handoff",
        "",
        f"- candidate grid: `{next_plan.get('candidate_grid')}`",
        f"- selection policy: {next_plan.get('selection_policy')}",
        f"- native divergence policy: {next_plan.get('native_divergence_policy')}",
        "",
        "## Inference Status",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key, value in payload["inference_status"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
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
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2t_map_local_reference_handoff(load_json(args.phase2s_json))
    payload["source_artifacts"] = {"phase2s_json": str(args.phase2s_json)}
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
