"""Phase 2S MAP-local geometry centering repair diagnostic.

This diagnostic builds a MAP-local SPD quadratic geometry candidate for the
scalar SSL-LSTM filtering target after Phase 2R localized the prior reference
mismatch to an outside-geometry-trust-region outcome.  It does not run HMC and
does not claim posterior correctness, HMC readiness, convergence, zero
divergences, GPU/XLA readiness, default readiness, or Zhao-Cui source
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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference import (  # noqa: E402
    LowRankSPDQuadraticGeometryConfig,
    QuadraticMapCovarianceLocatorConfig,
    QuadraticMapCovarianceMassConfig,
    estimate_quadratic_map_covariance,
)


SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md"
)
DEFAULT_GEOMETRY_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_geometry_cpu_hidden_2026-07-08.json"
)
DEFAULT_MASS_PATH = (
    ROOT / "docs/benchmarks/scalar_ssl_lstm_filtering_mass_handoff_cpu_hidden_2026-07-08.json"
)
DEFAULT_PHASE1R_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2R_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md"
)
GEOMETRY_MODULE_PATH = ROOT / "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_geometry_2026_07_08.py"
NONCLAIMS = (
    "Phase 2S MAP-local geometry centering diagnostic only",
    "optimizer output is a finite-neighborhood locator only",
    "not a certified global MAP",
    "not posterior covariance correctness evidence",
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


def load_geometry_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_geometry_phase2s_reuse",
        GEOMETRY_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load scalar filtering geometry module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_phase2s_geometry_centering_repair(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase1r_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
    phase2r_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_inputs(
        geometry_payload,
        mass_payload,
        phase1r_payload,
        phase2_payload,
        phase2r_payload,
    )
    points = build_diagnostic_points(geometry_payload, mass_payload, phase2_payload)
    module = load_geometry_module()
    settings = module.default_settings()
    target = module.build_filtering_geometry_target(settings)
    target_replay: Mapping[str, Any] = {"computed": False, "vetoes": ()}
    initializer_payload: Mapping[str, Any] | None = None
    handoff: Mapping[str, Any] | None = None

    if not precondition["vetoes"] and not points["vetoes"]:
        initial_position = np.asarray(
            points["points"]["phase2_reference_mean_initial"]["free"],
            dtype=float,
        )
        initializer = estimate_quadratic_map_covariance(
            target.value_and_score,
            initial_position,
            scale=np.asarray(target.scale.numpy(), dtype=float),
            locator_config=QuadraticMapCovarianceLocatorConfig(
                enabled=True,
                max_iterations=50,
                tolerance=1.0e-8,
                log_prob_tolerance=1.0e-8,
                parallel_iterations=1,
            ),
            quadratic_config=LowRankSPDQuadraticGeometryConfig(
                rank=4,
                sample_count=90,
                min_samples_per_parameter=5,
                trust_radius=0.60,
                pilot_radius=0.10,
                pilot_direction_count=96,
                holdout_fraction=0.25,
                eigenvalue_floor=1.0e-4,
                max_condition_number=1.0e5,
                fit_max_iterations=300,
                fit_tolerance=1.0e-8,
                holdout_rmse_abs_tolerance=0.10,
                holdout_rmse_rel_tolerance=0.01,
                center_score_improvement_factor=0.95,
                center_log_prob_tolerance=1.0e-8,
                seed=(20260709, 6201),
            ),
            mass_config=QuadraticMapCovarianceMassConfig(
                jitter=1.0e-9,
                eigenvalue_floor=1.0e-4,
                max_condition_number=1.0e5,
                dense=True,
            ),
        )
        initializer_payload = initializer.payload(include_arrays=True)
        target_replay = replay_target_points(module, target, points, initializer_payload)
        handoff = build_map_local_handoff(initializer_payload, target)
    else:
        target_replay = {
            "computed": False,
            "vetoes": tuple(dict.fromkeys((*precondition["vetoes"], *points["vetoes"]))),
            "reason": "skipped_due_to_failed_precondition_or_points",
        }

    gate = evaluate_phase2s_gate(precondition, points, initializer_payload, target_replay, handoff)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2s_geometry_centering_repair",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": phase2_payload.get("target_scope"),
        "settings": {
            "initial_position_source": "phase2_reference_mean_u_transformed_to_free_parameters",
            "locator": {
                "enabled": True,
                "max_iterations": 50,
                "tolerance": 1.0e-8,
                "log_prob_tolerance": 1.0e-8,
                "fallback_is_phase2s_veto": True,
            },
            "quadratic": {
                "rank": 4,
                "effective_rank_for_dim4": 3,
                "sample_count": 90,
                "min_samples_per_parameter": 5,
                "regression_parameter_count": 9,
                "required_finite_samples": 45,
                "trust_radius": 0.60,
                "pilot_radius": 0.10,
                "pilot_direction_count": 96,
                "holdout_fraction": 0.25,
                "eigenvalue_floor": 1.0e-4,
                "max_condition_number": 1.0e5,
                "holdout_rmse_abs_tolerance": 0.10,
                "holdout_rmse_rel_tolerance": 0.01,
                "seed": (20260709, 6201),
            },
            "mass": {
                "jitter": 1.0e-9,
                "eigenvalue_floor": 1.0e-4,
                "max_condition_number": 1.0e5,
                "dense": True,
            },
        },
        "source_artifacts": {
            "geometry_json": str(DEFAULT_GEOMETRY_PATH.relative_to(ROOT)),
            "mass_json": str(DEFAULT_MASS_PATH.relative_to(ROOT)),
            "phase1r_json": str(DEFAULT_PHASE1R_PATH.relative_to(ROOT)),
            "phase2_json": str(DEFAULT_PHASE2_PATH.relative_to(ROOT)),
            "phase2r_json": str(DEFAULT_PHASE2R_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "diagnostic_points": points,
        "initializer": initializer_payload,
        "map_local_handoff": handoff,
        "target_replay": target_replay,
        "telemetry_policy": telemetry_policy_payload(phase1r_payload, phase2_payload, phase2r_payload),
        "environment": environment_payload(module),
        "git": git_payload(),
        "decision": gate["decision"],
        "metric_roles": {
            "phase2s_geometry_centering_repair_passed": "primary_phase2s_pass_fail",
            "input_artifacts_valid": "hard_veto_evidence",
            "locator_accepted_optimizer_position": "hard_veto_evidence",
            "optimizer_inverse_hessian_unused": "hard_veto_evidence",
            "geometry_finite_sample_count": "hard_veto_evidence",
            "geometry_holdout_rmse": "geometry_fit_gate_only_not_hmc_evidence",
            "geometry_score_rmse": "explanatory_geometry_fit_diagnostic",
            "precision_spd_and_condition": "hard_veto_evidence",
            "target_replay": "diagnostic_only",
            "native_divergence_unavailable": "telemetry_availability_not_zero_divergences",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if gate["decision"]["phase2s_geometry_centering_repair_passed"] else "failed",
            "statistically_supported_ranking": "none; single diagnostic initializer",
            "descriptive_only_differences": (
                "locator movement, target replay values, fit residuals, eigen summaries, "
                "and distances to old reference/HMC summaries"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked until local repair handoff passes",
            "default_readiness": "not assessed",
            "zero_divergence_claim": "not made",
            "next_evidence_needed": gate["decision"]["next_justified_action"],
        },
        "decision_table": gate["decision_table"],
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(module),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": {
                "phase1r": phase1r_payload.get("settings", {}).get("seeds"),
                "quadratic_geometry": (20260709, 6201),
            },
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


def validate_inputs(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase1r_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
    phase2r_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if geometry_payload.get("schema_version") != "scalar_ssl_lstm.filtering_geometry.v1":
        vetoes.append("geometry_schema_mismatch")
    if geometry_payload.get("decision", {}).get("geometry_sanity_passed") is not True:
        vetoes.append("geometry_not_passed")
    if mass_payload.get("schema_version") != "scalar_ssl_lstm.filtering_mass_handoff.v1":
        vetoes.append("mass_schema_mismatch")
    if mass_payload.get("decision", {}).get("mass_handoff_passed") is not True:
        vetoes.append("mass_handoff_not_passed")
    if (
        phase1r_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase1r.v1"
    ):
        vetoes.append("phase1r_schema_mismatch")
    if phase1r_payload.get("decision", {}).get("phase1r_acceptance_repair_screen_passed") is not True:
        vetoes.append("phase1r_screen_not_passed")
    if (
        phase2_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2_local_quadratic_reference.v1"
    ):
        vetoes.append("phase2_schema_mismatch")
    if phase2_payload.get("decision", {}).get("phase2_local_quadratic_reference_agreement_passed") is not False:
        vetoes.append("phase2_expected_failure_missing")
    if (
        phase2r_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2r_localization.v1"
    ):
        vetoes.append("phase2r_schema_mismatch")
    if phase2r_payload.get("decision", {}).get("phase2r_localization_passed") is not True:
        vetoes.append("phase2r_localization_not_passed")
    if phase2r_payload.get("decision", {}).get("selected_outcome") != "outside_geometry_trust_region":
        vetoes.append("phase2r_outcome_not_outside_geometry_trust_region")
    if phase2r_payload.get("transform_checks", {}).get("passed") is not True:
        vetoes.append("phase2r_transform_checks_not_passed")
    if phase2r_payload.get("decision", {}).get("zero_divergence_claim_made") is not False:
        vetoes.append("zero_divergence_claim_present")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase1r_decision": phase1r_payload.get("decision", {}),
        "phase2_decision": phase2_payload.get("decision", {}),
        "phase2r_decision": phase2r_payload.get("decision", {}),
        "coordinate_contract": mass_payload.get("coordinate_contract", {}),
    }


def build_diagnostic_points(
    geometry_payload: Mapping[str, Any],
    mass_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    center = np.asarray(
        geometry_payload.get("center", {}).get("free_parameter_values"),
        dtype=float,
    )
    scale = np.asarray(
        mass_payload.get("coordinate_contract", {}).get("scale"),
        dtype=float,
    )
    factor = np.asarray(
        mass_payload.get("mass_handoff", {}).get("factor"),
        dtype=float,
    )
    ref_u = np.asarray(phase2_payload.get("reference", {}).get("mean_u"), dtype=float)
    pooled_u = np.asarray(phase2_payload.get("hmc_summary", {}).get("pooled_mean_u"), dtype=float)
    if center.shape != (4,) or not np.all(np.isfinite(center)):
        vetoes.append("center_free_shape_or_finiteness_mismatch")
    if scale.shape != (4,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        vetoes.append("scale_shape_or_finiteness_mismatch")
    if factor.shape != (4, 4) or not np.all(np.isfinite(factor)):
        vetoes.append("factor_shape_or_finiteness_mismatch")
    if ref_u.shape != (4,) or not np.all(np.isfinite(ref_u)):
        vetoes.append("reference_mean_u_shape_or_finiteness_mismatch")
    if pooled_u.shape != (4,) or not np.all(np.isfinite(pooled_u)):
        vetoes.append("pooled_hmc_mean_u_shape_or_finiteness_mismatch")
    if vetoes:
        return {"passed": False, "vetoes": tuple(dict.fromkeys(vetoes)), "points": {}}

    reference_free = free_from_u(center, scale, factor, ref_u)
    pooled_free = free_from_u(center, scale, factor, pooled_u)
    return {
        "passed": True,
        "vetoes": (),
        "coordinate_formula": "free = center + scale * (factor @ u)",
        "center_free": center,
        "scale": scale,
        "factor": factor,
        "points": {
            "truth_free_center": {
                "free": center,
                "u": np.zeros(4),
                "role": "old_truth_free_geometry_center",
                "gate_required": False,
            },
            "phase2_reference_mean_initial": {
                "free": reference_free,
                "u": ref_u,
                "role": "phase2_reference_mean_initial_position",
                "gate_required": True,
            },
            "phase1r_pooled_hmc_mean": {
                "free": pooled_free,
                "u": pooled_u,
                "role": "phase1r_pooled_hmc_mean_explanatory",
                "gate_required": False,
            },
        },
        "distances": {
            "reference_initial_minus_truth_free_norm": float(np.linalg.norm(reference_free - center)),
            "pooled_hmc_minus_truth_free_norm": float(np.linalg.norm(pooled_free - center)),
            "pooled_hmc_minus_reference_initial_norm": float(np.linalg.norm(pooled_free - reference_free)),
        },
    }


def free_from_u(
    center: np.ndarray,
    scale: np.ndarray,
    factor: np.ndarray,
    u_value: np.ndarray,
) -> np.ndarray:
    center_np = np.asarray(center, dtype=float)
    scale_np = np.asarray(scale, dtype=float)
    factor_np = np.asarray(factor, dtype=float)
    u_np = np.asarray(u_value, dtype=float)
    return center_np + scale_np * (factor_np @ u_np)


def replay_target_points(
    module: Any,
    target: Any,
    points: Mapping[str, Any],
    initializer_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    replay_points = dict(points.get("points", {}))
    locator = initializer_payload.get("locator_position")
    map_candidate = initializer_payload.get("map_candidate")
    if locator is not None:
        replay_points["locator_position"] = {
            "free": np.asarray(locator, dtype=float),
            "role": "optimizer_locator_position",
            "gate_required": True,
        }
    if map_candidate is not None:
        replay_points["map_candidate"] = {
            "free": np.asarray(map_candidate, dtype=float),
            "role": initializer_payload.get("map_candidate_role"),
            "gate_required": True,
        }
    values: dict[str, Any] = {}
    vetoes: list[str] = []
    for name, row in replay_points.items():
        free = np.asarray(row.get("free"), dtype=float)
        value, score, status = module.safe_value_and_score(
            target,
            module.tf.constant(free, dtype=module.tf.float64),
        )
        score_array = None if score is None else np.asarray(score, dtype=float)
        values[name] = {
            "free": free,
            "u": row.get("u"),
            "role": row.get("role"),
            "gate_required": bool(row.get("gate_required")),
            "value": value,
            "score": score_array,
            "score_norm": None if score_array is None else float(np.linalg.norm(score_array)),
            "status": status,
        }
        if row.get("gate_required") and status != "finite":
            vetoes.append(f"{name}_target_replay_{status}")
    return {
        "computed": True,
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "values": values,
        "role": "target replay for Phase 2S gate-required locator/map points and explanatory comparisons",
    }


def build_map_local_handoff(
    initializer_payload: Mapping[str, Any],
    target: Any,
) -> Mapping[str, Any] | None:
    geometry = initializer_payload.get("geometry") or {}
    if initializer_payload.get("map_candidate") is None:
        return None
    map_candidate = np.asarray(initializer_payload["map_candidate"], dtype=float)
    precision_z = geometry.get("precision")
    covariance_z = geometry.get("covariance")
    factor_z = None
    if precision_z is not None and covariance_z is not None:
        covariance_array = np.asarray(covariance_z, dtype=float)
        try:
            factor_z = np.linalg.cholesky(0.5 * (covariance_array + covariance_array.T))
        except np.linalg.LinAlgError:
            factor_z = None
    return {
        "center_role": initializer_payload.get("map_candidate_role"),
        "center_free_parameter_values": map_candidate,
        "scale": np.asarray(target.scale.numpy(), dtype=float),
        "precision_theta": initializer_payload.get("precision"),
        "covariance_theta": initializer_payload.get("covariance"),
        "precision_z": precision_z,
        "covariance_z": covariance_z,
        "factor_z": factor_z,
        "coordinate_formula": "free = center_free_parameter_values + scale * (factor_z @ u_new)",
        "nonclaims": (
            "MAP-local handoff candidate only",
            "not posterior covariance correctness evidence",
            "not HMC readiness evidence",
        ),
    }


def evaluate_phase2s_gate(
    precondition: Mapping[str, Any],
    points: Mapping[str, Any],
    initializer_payload: Mapping[str, Any] | None,
    target_replay: Mapping[str, Any],
    handoff: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    vetoes.extend(precondition.get("vetoes", ()))
    vetoes.extend(points.get("vetoes", ()))
    if initializer_payload is None:
        vetoes.append("initializer_not_run")
    else:
        if initializer_payload.get("accepted") is not True:
            vetoes.append("initializer_not_accepted")
        if initializer_payload.get("status") != "usable":
            vetoes.append(f"initializer_status_{initializer_payload.get('status')}")
        locator = initializer_payload.get("locator_diagnostics", {})
        if locator.get("accepted_optimizer_position") is not True:
            vetoes.append("locator_fallback_or_not_accepted")
        if locator.get("uses_optimizer_inverse_hessian") is not False:
            vetoes.append("optimizer_inverse_hessian_usage_not_false")
        if initializer_payload.get("map_candidate") is None:
            vetoes.append("map_candidate_missing")
        geometry = initializer_payload.get("geometry") or {}
        if geometry.get("accepted") is not True:
            vetoes.append("geometry_not_accepted")
        if geometry.get("status") != "usable":
            vetoes.append(f"geometry_status_{geometry.get('status')}")
        diagnostics = geometry.get("diagnostics", {})
        regression_parameters = int(diagnostics.get("regression_parameter_count", 0))
        required_finite = int(diagnostics.get("required_finite_samples", 0))
        finite_samples = int(diagnostics.get("finite_sample_count", 0))
        if regression_parameters <= 0:
            vetoes.append("regression_parameter_count_missing")
        elif finite_samples < 5 * regression_parameters:
            vetoes.append("finite_sample_count_below_5x_regression_parameters")
        if finite_samples < required_finite:
            vetoes.append("finite_sample_count_below_required_finite_samples")
        if int(diagnostics.get("holdout_count", 0)) <= 0:
            vetoes.append("holdout_count_zero")
        if diagnostics.get("holdout_passed") is not True:
            vetoes.append("holdout_fit_not_accepted")
        if not _positive_condition_summary(initializer_payload.get("precision_eigen_summary"), 1.0e5):
            vetoes.append("regularized_precision_spd_or_condition_failed")
        if not _positive_condition_summary(initializer_payload.get("covariance_eigen_summary"), 1.0e5):
            vetoes.append("covariance_spd_or_condition_failed")
        mass = initializer_payload.get("mass_matrix") or {}
        if not mass:
            vetoes.append("mass_matrix_missing")
        elif mass.get("regularization_report", {}).get("diagonal_fallback_used") is True:
            vetoes.append("mass_matrix_diagonal_fallback_used")
    vetoes.extend(target_replay.get("vetoes", ()))
    if target_replay.get("computed") is not True:
        vetoes.append("target_replay_not_computed")
    if handoff is None:
        vetoes.append("map_local_handoff_missing")
    elif handoff.get("factor_z") is None:
        vetoes.append("map_local_handoff_factor_missing")
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = not unique_vetoes
    decision = {
        "phase2s_geometry_centering_repair_passed": passed,
        "vetoes": unique_vetoes,
        "viable_for_map_local_reference_subplan": passed,
        "zero_divergence_claim_made": False,
        "next_justified_action": (
            "draft and review MAP-local reference-agreement or retuned fixed-kernel HMC screen subplan"
            if passed
            else "write Phase 2S result and draft narrower initializer repair or blocker"
        ),
    }
    return {
        "decision": decision,
        "decision_table": {
            "decision": "Phase 2S MAP-local geometry centering repair",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "A usable SPD quadratic initializer is still a local diagnostic and does not "
                "establish posterior covariance correctness or HMC readiness."
            ),
            "next_justified_action": decision["next_justified_action"],
            "what_is_not_being_concluded": (
                "No certified global MAP, posterior covariance correctness, HMC readiness, "
                "convergence, zero-divergence claim, sampler superiority, GPU/XLA readiness, "
                "default readiness, or Zhao-Cui source faithfulness."
            ),
        },
    }


def _positive_condition_summary(summary: Any, cap: float) -> bool:
    if not isinstance(summary, Mapping):
        return False
    try:
        return bool(
            summary.get("finite") is True
            and summary.get("positive") is True
            and float(summary.get("condition_number")) <= float(cap) * (1.0 + 1.0e-8)
        )
    except (TypeError, ValueError):
        return False


def telemetry_policy_payload(
    phase1r_payload: Mapping[str, Any],
    phase2_payload: Mapping[str, Any],
    phase2r_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    statuses = phase1r_payload.get("telemetry_policy", {}).get(
        "native_divergence_statuses",
        phase2_payload.get("telemetry_policy", {}).get(
            "native_divergence_statuses",
            phase2r_payload.get("telemetry_policy", {}).get("native_divergence_statuses", ()),
        ),
    )
    return {
        "native_divergence_statuses": statuses,
        "native_divergence_interpretation": (
            "native divergence unavailable for at least one seed; unavailable is not zero divergences"
        ),
        "zero_divergence_claim_made": False,
        "unavailable_native_divergence_is_zero_divergence": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def environment_payload(module: Any | None = None) -> Mapping[str, Any]:
    tf_version = None
    tf_physical_devices: list[Mapping[str, str]] = []
    if module is not None:
        try:
            tf_version = module.tf.__version__
            tf_physical_devices = [
                {"name": device.name, "device_type": device.device_type}
                for device in module.tf.config.list_physical_devices()
            ]
        except Exception:  # noqa: BLE001 - provenance best effort.
            tf_physical_devices = []
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "tf_physical_devices": tf_physical_devices,
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
    initializer = payload.get("initializer") or {}
    geometry = initializer.get("geometry") or {}
    diagnostics = geometry.get("diagnostics", {})
    mass = initializer.get("mass_matrix") or {}
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2S - Geometry Centering Repair",
        "",
        "## Decision",
        "",
        f"- phase2s_geometry_centering_repair_passed: `{decision['phase2s_geometry_centering_repair_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- viable_for_map_local_reference_subplan: `{decision['viable_for_map_local_reference_subplan']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Initializer",
        "",
        f"- accepted/status: `{initializer.get('accepted')}` / `{initializer.get('status')}`",
        f"- map_candidate_role: `{initializer.get('map_candidate_role')}`",
        f"- locator diagnostics: `{initializer.get('locator_diagnostics')}`",
        "",
        "## Geometry",
        "",
        f"- finite sample count: `{diagnostics.get('finite_sample_count')}`",
        f"- required finite samples: `{diagnostics.get('required_finite_samples')}`",
        f"- regression parameter count: `{diagnostics.get('regression_parameter_count')}`",
        f"- holdout count: `{diagnostics.get('holdout_count')}`",
        f"- holdout passed: `{diagnostics.get('holdout_passed')}`",
        f"- holdout rmse: `{diagnostics.get('holdout_rmse')}`",
        f"- score rmse: `{(diagnostics.get('fit') or {}).get('score_rmse')}`",
        f"- precision eigen summary: `{initializer.get('precision_eigen_summary')}`",
        f"- covariance eigen summary: `{initializer.get('covariance_eigen_summary')}`",
        "",
        "## Mass Regularization",
        "",
        f"- regularization report: `{mass.get('regularization_report')}`",
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
    parser.add_argument("--geometry-json", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--mass-json", type=Path, default=DEFAULT_MASS_PATH)
    parser.add_argument("--phase1r-json", type=Path, default=DEFAULT_PHASE1R_PATH)
    parser.add_argument("--phase2-json", type=Path, default=DEFAULT_PHASE2_PATH)
    parser.add_argument("--phase2r-json", type=Path, default=DEFAULT_PHASE2R_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2s_geometry_centering_repair(
        load_json(args.geometry_json),
        load_json(args.mass_json),
        load_json(args.phase1r_json),
        load_json(args.phase2_json),
        load_json(args.phase2r_json),
    )
    payload["source_artifacts"] = {
        "geometry_json": str(args.geometry_json),
        "mass_json": str(args.mass_json),
        "phase1r_json": str(args.phase1r_json),
        "phase2_json": str(args.phase2_json),
        "phase2r_json": str(args.phase2r_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
