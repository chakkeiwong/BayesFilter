"""Build the P6 Algorithm 1 UKF cross-filter calibration artifact.

This runner is intentionally pure Python.  It does not rerun filters and does
not import TensorFlow.  P6 assembles already-written filter-oracle deterministic
calibration rows together with the P5 Algorithm 1 UKF DPF replacement rows.
The output is a claim-class-separated calibration ledger, not a global ranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"

REGISTRY_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-target-route-registry-2026-06-08.json"
OLD_P2_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_2026-06-08.json"
OLD_P3_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p3_conditional_gaussian_mixture_2026-06-08.json"
OLD_P4_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json"
P5_ALG1_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_alg1_ukf_statistical_closeness_2026-06-10.json"
P3_ALG1_VALUES_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_values_2026-06-10.json"
P4_ALG1_GRADIENTS_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_gradients_2026-06-10.json"
P45_CALIBRATION_PATH = REPO_ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p45-cross-model-error-calibration-2026-06-08.json"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p6_alg1_ukf_cross_filter_calibration_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p6-cross-filter-calibration-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p6-cross-filter-calibration-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p6_alg1_ukf_cross_filter_calibration_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p6-alg1-ukf-cross-filter-calibration-2026-06-10.md"

METHOD_ID = "ledh_pfpf_alg1_ukf_no_resampling_tf"
LOCAL_PASS_DECISION = "LOCAL_PASS_P6_ALG1_UKF_CROSS_FILTER_CALIBRATION_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
VETO_DECISION = "P6_ALG1_UKF_CROSS_FILTER_CALIBRATION_VETO_PENDING_REVIEW"
REFERENCE_FLOOR = 1e-12
ALG1_ROUTE_REQUIRED = (
    "method_generation",
    "flow_source_route",
    "covariance_route",
    "prediction_covariance_route",
    "update_covariance_route",
    "flow_anchor_route",
    "core_resampling_route",
    "extension_resampling_route",
    "evidence_route_class",
    "previous_ledh_pfpf_ot_evidence_status",
)
OLD_DPF_METHODS = {"dpf_bootstrap_ot", "dpf_ledh_pfpf_ot"}


class P6Alg1ValidationError(ValueError):
    """Raised when a P6 Algorithm 1 artifact violates the contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P6_ALG1_UKF_CROSS_FILTER_CALIBRATION_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    _write_json(JSON_PATH, payload)
    _write_text(REPORT_PATH, markdown)
    _write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    registry = _load_json(REGISTRY_PATH)
    old_p2 = _load_json(OLD_P2_JSON_PATH)
    old_p3 = _load_json(OLD_P3_JSON_PATH)
    old_p4 = _load_json(OLD_P4_JSON_PATH)
    p5 = _load_json(P5_ALG1_JSON_PATH)
    p3_values = _load_json(P3_ALG1_VALUES_JSON_PATH)
    p4_gradients = _load_json(P4_ALG1_GRADIENTS_JSON_PATH)
    p45 = _load_json(P45_CALIBRATION_PATH)
    _preflight(old_p2, old_p3, old_p4, p5, p3_values, p4_gradients)

    exact_rows = _p2_exact_target_rows(old_p2)
    approximation_rows = _p3_ksc_approximation_rows(old_p3)
    alg1_rows = _alg1_dpf_rows(p5, p3_values, p4_gradients)
    blocked_rows = _blocked_rows(registry, old_p4, p5, p45)
    unstructured_rows = _unstructured_rows(old_p4) + _p3_exact_transformed_unstructured_rows(old_p3)
    explanatory_gaps = _p3_explanatory_approximation_gaps(old_p3)
    summaries = _summaries(exact_rows, approximation_rows, alg1_rows, blocked_rows, unstructured_rows)
    veto = _veto_diagnostics(exact_rows, approximation_rows, alg1_rows, blocked_rows, p5)
    decision = LOCAL_PASS_DECISION if not any(bool(value) for value in veto.values()) else VETO_DECISION

    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": _utc_now(),
        "phase": "P6",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_ARTIFACT_CALIBRATION_ONLY",
        "decision": decision,
        "question": (
            "How do Algorithm 1 UKF DPF value and fixed-branch gradient error "
            "scales compare with valid deterministic filter-oracle calibration "
            "rows, without reviving old LEDH-PFPF-OT evidence or ranking "
            "incompatible model/filter pairs?"
        ),
        "skeptical_plan_audit": {
            "status": "PASS_FOR_ARTIFACT_CALIBRATION_ONLY",
            "wrong_baseline_control": (
                "Deterministic rows retain their own target/reference class; "
                "Algorithm 1 DPF rows use P5 replacement evidence only."
            ),
            "proxy_promotion_control": (
                "Algorithm 1 finite rows remain diagnostic-only because P5 "
                "declared no numeric promotion band."
            ),
            "stop_condition_control": (
                "Every P5 Algorithm 1 eligible target appears as diagnostic or "
                "blocked; rows without structured same-target metrics are "
                "blocked or unstructured."
            ),
            "fairness_control": (
                "Exact-target, approximation-target, and Algorithm 1 DPF rows "
                "are separate ledgers.  No global ranking is emitted."
            ),
            "environment_control": "Pure-Python assembly; TensorFlow is not imported.",
        },
        "evidence_contract": {
            "baseline_comparator": (
                "Historical deterministic filter-oracle rows for valid "
                "same-target calibration, plus P5/P3/P4 Algorithm 1 UKF "
                "replacement artifacts for DPF rows."
            ),
            "primary_criterion": (
                "Produce separated exact-target, approximation-target, "
                "Algorithm 1 DPF diagnostic/blocker, blocked, and unstructured "
                "ledgers with uncertainty or N/A/blocker reasons."
            ),
            "ranking_policy": "no_global_ranking; no cross-target ordering",
            "old_dpf_policy": "old dpf_ledh_pfpf_ot and old dpf_bootstrap_ot rows are historical-only and not consumed as current evidence",
            "data_law_variability_policy": "not_used; no rmse/sd_L promotion or bias excuse is claimed",
        },
        "consumed_artifacts": _consumed_artifacts(),
        "exact_target_calibration_rows": exact_rows,
        "approximation_target_calibration_rows": approximation_rows,
        "algorithm1_dpf_rows": alg1_rows,
        "blocked_rows": blocked_rows,
        "unstructured_metric_rows": unstructured_rows,
        "explanatory_approximation_gaps": explanatory_gaps,
        "historical_old_dpf_quarantine": _historical_old_dpf_quarantine(),
        "route_summaries": summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "reference_floor_for_refinement_ratios": REFERENCE_FLOOR,
            "data_law_variability_used": False,
            "global_ranking_emitted": False,
            "old_ledh_pfpf_ot_used_as_current_evidence": False,
            "algorithm1_uncertainty_reported_for_diagnostic_rows": True,
            "p45_empty_calibration_preserved_for_multistate_blockers": True,
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _p2_exact_target_rows(p2: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target_row in p2["rows"]:
        dim = int(target_row["dim"])
        ref = target_row["dense_reference"]
        refinement = ref["refinement"]
        for route in target_row["deterministic_routes"]:
            rows.append(
                _calibration_row(
                    source_phase="historical_filter_oracle_P2_deterministic",
                    target_id=target_row["target_id"],
                    target_class="exact_target",
                    dim=dim,
                    route=route,
                    reference_route_id="dense_refined_quadrature",
                    reference_claim_class=ref["claim_class"],
                    reference_value=ref["value"],
                    reference_score_norm=refinement["reference_score_norm"],
                    reference_uncertainty={
                        "status": "dense_refinement_recorded",
                        "value_refinement_gap": refinement["value_gap"],
                        "directional_score_refinement_gap": refinement["max_directional_score_gap"],
                        "low_order": refinement["low_order"],
                        "high_order": refinement["high_order"],
                    },
                )
            )
    return rows


def _p3_ksc_approximation_rows(p3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in p3["rows"]["ksc_mixture"]:
        route = source["candidate"]
        reference = source["reference"]
        rows.append(
            _calibration_row(
                source_phase="historical_filter_oracle_P3_deterministic",
                target_id=source["target_id"],
                target_class="approximation_target",
                dim=int(source["dim"]),
                route=route,
                reference_route_id=reference["route_id"],
                reference_claim_class=reference["claim_class"],
                reference_value=reference["value"],
                reference_score_norm=_norm(reference["score"]),
                reference_uncertainty={
                    "status": "finite_component_enumeration_reference",
                    "component_tuple_count": route["diagnostics"]["component_tuple_count"],
                    "certificate_tolerances": route["certificate_tolerances"],
                    "native_sv_exactness": "not_claimed",
                },
            )
        )
    return rows


def _calibration_row(
    *,
    source_phase: str,
    target_id: str,
    target_class: str,
    dim: int,
    route: dict[str, Any],
    reference_route_id: str,
    reference_claim_class: str,
    reference_value: float,
    reference_score_norm: float,
    reference_uncertainty: dict[str, Any],
) -> dict[str, Any]:
    value_error = float(route["value_error"])
    abs_value_error = abs(float(route["abs_value_error"]))
    directional_gap = float(route["directional_score_gap"])
    reference_value_scale = max(1.0, abs(float(reference_value)))
    row = {
        "source_phase": source_phase,
        "target_id": target_id,
        "target_class": target_class,
        "dim": dim,
        "route_id": route["route_id"],
        "method_family": "deterministic_filter_oracle_context",
        "claim_class": route["claim_class"],
        "route_status": route["route_status"],
        "reference_route_id": reference_route_id,
        "reference_claim_class": reference_claim_class,
        "primary_gradient_statistic": route["primary_gradient_statistic"],
        "finite": bool(route["finite"]),
        "value_error": value_error,
        "abs_value_error": abs_value_error,
        "per_observation_value_error": route.get("per_observation_value_error"),
        "value_error_relative_to_reference_value_scale": abs_value_error / reference_value_scale,
        "score_error_norm": route["score_error_norm"],
        "relative_score_error": route["relative_score_error"],
        "directional_score_gap": directional_gap,
        "reference_score_norm": reference_score_norm,
        "reference_uncertainty": reference_uncertainty,
        "valid_for_within_target_calibration": route["claim_class"] == "CERTIFIED_APPROXIMATION",
        "valid_for_global_ranking": False,
        "interpretation": route["interpretation"],
    }
    if reference_uncertainty["status"] == "dense_refinement_recorded":
        row["value_error_to_refinement_floor_ratio"] = abs_value_error / max(
            abs(float(reference_uncertainty["value_refinement_gap"])),
            REFERENCE_FLOOR,
        )
        row["directional_gap_to_refinement_floor_ratio"] = directional_gap / max(
            abs(float(reference_uncertainty["directional_score_refinement_gap"])),
            REFERENCE_FLOOR,
        )
    else:
        row["value_error_to_refinement_floor_ratio"] = None
        row["directional_gap_to_refinement_floor_ratio"] = None
    return row


def _alg1_dpf_rows(
    p5: dict[str, Any],
    p3_values: dict[str, Any],
    p4_gradients: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in p5["rows"]:
        if row["method_id"] != METHOD_ID:
            raise P6Alg1ValidationError(f"unexpected P5 method id {row['method_id']}")
        if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY":
            rows.append(_alg1_diagnostic_row(row, p3_values, p4_gradients))
        elif row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER":
            rows.append(_alg1_blocked_row(row))
        else:
            raise P6Alg1ValidationError(f"unexpected P5 replacement status {row['replacement_status']}")
    return rows


def _alg1_diagnostic_row(
    row: dict[str, Any],
    p3_values: dict[str, Any],
    p4_gradients: dict[str, Any],
) -> dict[str, Any]:
    target_id = row["target_id"]
    value_particle = str(row["value_statistics"]["particle_count"])
    gradient_particle = str(row["gradient_statistics"]["particle_count"])
    value_summary = p3_values["value_summaries"][target_id][METHOD_ID][value_particle]
    gradient_summary = p4_gradients["gradient_summaries"][target_id][gradient_particle]
    gradient_rows = [
        item
        for item in p4_gradients["gradient_rows"]
        if item["model_id"] == target_id
        and item["method_id"] == METHOD_ID
        and int(item["num_particles"]) == int(gradient_particle)
        and item["gradient_reference"] is not None
    ]
    if not gradient_rows:
        raise P6Alg1ValidationError("LGSSM diagnostic row lacks gradient reference rows")
    reference_gradient_norm = _norm(gradient_rows[0]["gradient_reference"])
    reference_value = float(value_summary["mean_value"]) - float(value_summary["mean_value_error"])
    value_scale = max(1.0, abs(reference_value))
    gradient_scale = max(1.0, reference_gradient_norm)
    return {
        "source_phase": "P5/P3/P4_algorithm1_replacement",
        "target_id": target_id,
        "target_class": row["comparison_target_class"],
        "method_id": row["method_id"],
        "method_family": "algorithm1_ukf_dpf",
        "replacement_status": row["replacement_status"],
        "claim_class": "DIAGNOSTIC_ONLY",
        "reference_route": row["reference_route"],
        "reference_value": reference_value,
        "reference_value_scale": value_scale,
        "reference_gradient_norm": reference_gradient_norm,
        "reference_gradient_scale": gradient_scale,
        "value_particle_count": int(value_particle),
        "value_seed_count": int(value_summary["seed_count"]),
        "value_finite_count": int(value_summary["finite_count"]),
        "value_rmse_vs_reference": value_summary["value_rmse_vs_reference"],
        "value_standard_error": value_summary["value_standard_error"],
        "value_ci95": value_summary["value_ci95"],
        "normalized_value_rmse": float(value_summary["value_rmse_vs_reference"]) / value_scale,
        "gradient_particle_count": int(gradient_particle),
        "gradient_seed_count": int(gradient_summary["seed_count"]),
        "gradient_finite_count": int(gradient_summary["finite_count"]),
        "mean_gradient_error_norm": gradient_summary["mean_gradient_error_norm"],
        "gradient_error_norm_standard_error": gradient_summary["gradient_error_norm_standard_error"],
        "gradient_error_norm_ci95": gradient_summary["gradient_error_norm_ci95"],
        "normalized_gradient_error_norm": float(gradient_summary["mean_gradient_error_norm"]) / gradient_scale,
        "route_fields": row["route_fields"],
        "valid_for_within_target_calibration": False,
        "valid_for_global_ranking": False,
        "reason_not_valid_for_calibration": (
            "P5 declared no numeric Algorithm 1 statistical-closeness promotion band; "
            "finite values and gradients remain diagnostic-only."
        ),
        "downgrade_reasons": row["downgrade_reasons"],
        "blockers": [],
        "stochastic_score_claim": row["stochastic_score_claim"],
        "old_coverage_route_id": row["old_coverage_route_id"],
    }


def _alg1_blocked_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_phase": "P5_algorithm1_replacement",
        "target_id": row["target_id"],
        "target_class": row["comparison_target_class"],
        "method_id": row["method_id"],
        "method_family": "algorithm1_ukf_dpf",
        "replacement_status": row["replacement_status"],
        "claim_class": "BLOCKED",
        "reference_route": row["reference_route"],
        "valid_for_within_target_calibration": False,
        "valid_for_global_ranking": False,
        "reason_not_valid_for_calibration": "same-target Algorithm 1 adapter or numeric P5 band missing",
        "blockers": row["blockers"],
        "route_fields": row["route_fields"],
        "value_rmse_vs_reference": None,
        "value_standard_error": None,
        "normalized_value_rmse": None,
        "mean_gradient_error_norm": None,
        "gradient_error_norm_standard_error": None,
        "normalized_gradient_error_norm": None,
        "stochastic_score_claim": row["stochastic_score_claim"],
        "old_coverage_route_id": row["old_coverage_route_id"],
    }


def _blocked_rows(
    registry: dict[str, Any],
    old_p4: dict[str, Any],
    p5: dict[str, Any],
    p45: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in p5["rows"]:
        if row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER":
            rows.append(
                {
                    "source_phase": "P5_algorithm1_replacement",
                    "target_id": row["target_id"],
                    "route_id": row["method_id"],
                    "reason": "; ".join(row["blockers"]),
                    "value_gap": None,
                    "relative_score_error": None,
                    "valid_for_calibration_table": False,
                }
            )
    for row in p45["rows"]:
        rows.append(
            {
                "source_phase": row["phase"],
                "target_id": ",".join(row["target_ids"]),
                "route_id": "cross_model_route_set",
                "reason": row["why_equality_metrics_absent"],
                "value_gap": row["value_gap"],
                "relative_score_error": row["relative_score_error"],
                "valid_for_calibration_table": False,
            }
        )
    p5_targets = set(old_p4["route_summaries"]["p5_dpf_eligible_targets"])
    for target_id, route_map in registry["route_matrix"].items():
        if target_id in p5_targets:
            continue
        for route_id in OLD_DPF_METHODS:
            route = route_map[route_id]
            if route["claim_class"] == "BLOCKED":
                rows.append(
                    {
                        "source_phase": "historical_filter_oracle_registry",
                        "target_id": target_id,
                        "route_id": route_id,
                        "reason": "; ".join(route["blockers"]) or route["route_status"],
                        "value_gap": None,
                        "relative_score_error": None,
                        "valid_for_calibration_table": False,
                    }
                )
    return rows


def _unstructured_rows(old_p4: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in old_p4["rows"]:
        target_id = row["target_id"]
        if target_id not in {
            "p44_m3_quadratic_observation_panel",
            "p44_m4_nonlinear_transition_h2_panel",
        }:
            continue
        rows.append(
            {
                "source_phase": "historical_filter_oracle_P4",
                "target_id": target_id,
                "route_id": "deterministic_routes_from_p44_source_notes",
                "status": "structured_row_metrics_not_available_in_P1_P5_json",
                "source_artifacts": row["classification"]["source_artifacts"],
                "why_not_calibrated": (
                    "P4 records source-artifact summaries, but P6 does not parse "
                    "markdown tables into metrics.  A future phase should add a "
                    "machine-readable P44-M3/M4 metric artifact before calibration."
                ),
                "valid_for_calibration_table": False,
            }
        )
    return rows


def _p3_exact_transformed_unstructured_rows(p3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in p3["rows"]["exact_transformed"]:
        route = source["candidate"]
        reference = source["reference"]
        rows.append(
            {
                "source_phase": "historical_filter_oracle_P3",
                "target_id": source["target_id"],
                "dim": int(source["dim"]),
                "route_id": route["route_id"],
                "reference_route_id": reference["route_id"],
                "status": "reference_uncertainty_not_structured_in_p3_artifact",
                "observed_abs_value_error": route["abs_value_error"],
                "observed_relative_score_error": route["relative_score_error"],
                "observed_directional_score_gap": route["directional_score_gap"],
                "why_not_calibrated": (
                    "P6 requires reference uncertainty in valid exact-target "
                    "calibration rows. P3 exact-transformed rows record "
                    "same-target candidate gaps and certificate tolerances, but "
                    "not a separate machine-readable dense-refinement residual."
                ),
                "certificate_tolerances": route["certificate_tolerances"],
                "valid_for_calibration_table": False,
            }
        )
    return rows


def _p3_explanatory_approximation_gaps(p3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in p3["rows"]["ksc_mixture"]:
        gap = source["approximation_gap_to_exact_transformed"]
        rows.append(
            {
                "source_phase": "historical_filter_oracle_P3",
                "target_id": source["target_id"],
                "dim": int(source["dim"]),
                "gap_type": "ksc_mixture_reference_vs_exact_transformed_reference",
                "policy": gap["policy"],
                "abs_value_gap": gap["abs_value_gap"],
                "directional_score_gap": gap["directional_score_gap"],
                "relative_score_error": gap["relative_score_error"],
                "valid_for_same_target_calibration": False,
            }
        )
    return rows


def _summaries(
    exact_rows: list[dict[str, Any]],
    approximation_rows: list[dict[str, Any]],
    alg1_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    unstructured_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_alg1_rows = [
        row["target_id"]
        for row in alg1_rows
        if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
    ]
    blocked_alg1_rows = [
        row["target_id"]
        for row in alg1_rows
        if row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER"
    ]
    return {
        "exact_target_row_count": len(exact_rows),
        "approximation_target_row_count": len(approximation_rows),
        "algorithm1_dpf_row_count": len(alg1_rows),
        "algorithm1_dpf_diagnostic_rows": diagnostic_alg1_rows,
        "algorithm1_dpf_blocked_rows": blocked_alg1_rows,
        "algorithm1_dpf_promoted_rows": [],
        "blocked_row_count": len(blocked_rows),
        "unstructured_metric_row_count": len(unstructured_rows),
        "valid_within_target_calibration_rows": [
            _route_key(row)
            for row in exact_rows + approximation_rows
            if row["valid_for_within_target_calibration"]
        ],
        "diagnostic_only_rows": [
            _route_key(row)
            for row in exact_rows
            if row["claim_class"] == "DIAGNOSTIC_ONLY"
        ]
        + [f"{row['target_id']}/{row['method_id']}" for row in alg1_rows if row["claim_class"] == "DIAGNOSTIC_ONLY"],
        "global_ranking_policy": "not_emitted",
        "max_abs_value_error_by_evidence_table": {
            "exact_target_deterministic": _max_abs([row["abs_value_error"] for row in exact_rows]),
            "approximation_target_deterministic": _max_abs([row["abs_value_error"] for row in approximation_rows]),
            "algorithm1_dpf_diagnostic_rmse": _max_abs(
                [
                    row["value_rmse_vs_reference"]
                    for row in alg1_rows
                    if isinstance(row.get("value_rmse_vs_reference"), (int, float))
                ]
            ),
        },
        "max_gradient_error_by_evidence_table": {
            "exact_target_deterministic_relative_score": _max_abs([row["relative_score_error"] for row in exact_rows]),
            "approximation_target_deterministic_relative_score": _max_abs(
                [row["relative_score_error"] for row in approximation_rows]
            ),
            "algorithm1_dpf_diagnostic_norm": _max_abs(
                [
                    row["mean_gradient_error_norm"]
                    for row in alg1_rows
                    if isinstance(row.get("mean_gradient_error_norm"), (int, float))
                ]
            ),
        },
    }


def _veto_diagnostics(
    exact_rows: list[dict[str, Any]],
    approximation_rows: list[dict[str, Any]],
    alg1_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    p5: dict[str, Any],
) -> dict[str, bool]:
    return {
        "global_ranking_emitted": False,
        "data_law_variability_used_to_excuse_mismatch": False,
        "old_ledh_pfpf_ot_used_as_current_evidence": any(
            row.get("method_id") == "dpf_ledh_pfpf_ot" for row in alg1_rows
        ),
        "old_dpf_metric_consumed_as_current_algorithm1": False,
        "algorithm1_route_fields_missing": any(
            row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            and not _has_algorithm1_route_fields(row["route_fields"])
            for row in alg1_rows
        ),
        "algorithm1_diagnostic_uncertainty_missing": any(
            row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            and (
                row.get("value_standard_error") is None
                or row.get("gradient_error_norm_standard_error") is None
            )
            for row in alg1_rows
        ),
        "algorithm1_row_promoted": any(
            row["replacement_status"] == "RERUN_ALG1" or row["valid_for_within_target_calibration"]
            for row in alg1_rows
        )
        or bool(p5["route_summaries"]["promoted_rows"]),
        "p44_algorithm1_metric_fabricated": any(
            row["target_id"].startswith("p44_")
            and row["replacement_status"] != "BLOCKED_REQUIRES_ADAPTER"
            for row in alg1_rows
        ),
        "approximation_route_ranked_as_exact": any(
            row["target_class"] != "approximation_target"
            for row in approximation_rows
            if row["reference_route_id"] == "ksc_kalman_mixture_reference"
        ),
        "reference_uncertainty_omitted_from_p2_dense_rows": any(
            row["source_phase"] == "historical_filter_oracle_P2_deterministic"
            and row["reference_uncertainty"]["status"] != "dense_refinement_recorded"
            for row in exact_rows
        ),
        "exact_target_row_lacks_accepted_reference_uncertainty": any(
            row["target_class"] == "exact_target"
            and row["reference_uncertainty"]["status"] not in {"dense_refinement_recorded"}
            for row in exact_rows
        ),
        "blocked_row_has_metric": any(
            row["valid_for_calibration_table"]
            or row["value_gap"] is not None
            or row["relative_score_error"] is not None
            for row in blocked_rows
        ),
        "nonfinite_calibration_row": any(
            not row["finite"] for row in exact_rows + approximation_rows
        ),
        "value_used_to_promote_gradient": False,
        "zhao_cui_used_as_dpf_correctness_oracle": False,
    }


def _preflight(
    old_p2: dict[str, Any],
    old_p3: dict[str, Any],
    old_p4: dict[str, Any],
    p5: dict[str, Any],
    p3_values: dict[str, Any],
    p4_gradients: dict[str, Any],
) -> None:
    if not old_p2.get("decision", "").startswith("PASS_P2_"):
        raise P6Alg1ValidationError("historical P2 deterministic artifact is not ready")
    if not old_p3.get("decision", "").startswith("PASS_P3_"):
        raise P6Alg1ValidationError("historical P3 deterministic artifact is not ready")
    if not old_p4.get("decision", "").startswith("PASS_P4_"):
        raise P6Alg1ValidationError("historical P4 route artifact is not ready")
    if not p5.get("decision", "").startswith("LOCAL_PASS_P5_"):
        raise P6Alg1ValidationError("P5 Algorithm 1 artifact is not ready")
    if not p3_values.get("decision", "").startswith("LOCAL_PASS_P3_"):
        raise P6Alg1ValidationError("P3 Algorithm 1 values artifact is not ready")
    if not p4_gradients.get("decision", "").startswith("LOCAL_PASS_P4_"):
        raise P6Alg1ValidationError("P4 Algorithm 1 gradients artifact is not ready")


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "execution_mode",
        "evidence_contract",
        "exact_target_calibration_rows",
        "approximation_target_calibration_rows",
        "algorithm1_dpf_rows",
        "blocked_rows",
        "unstructured_metric_rows",
        "historical_old_dpf_quarantine",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P6Alg1ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P6Alg1ValidationError(f"invalid P6 decision {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_ARTIFACT_CALIBRATION_ONLY":
        raise P6Alg1ValidationError("P6 should be pure-Python artifact calibration only")
    if payload["run_manifest"].get("tensorflow_imported"):
        raise P6Alg1ValidationError("TensorFlow was imported in P6")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P6Alg1ValidationError(f"P6 veto fired: {payload['veto_diagnostics']}")
    if not payload["exact_target_calibration_rows"]:
        raise P6Alg1ValidationError("P6 should include deterministic exact-target rows")
    if not payload["approximation_target_calibration_rows"]:
        raise P6Alg1ValidationError("P6 should include deterministic approximation-target rows")
    alg1_rows = payload["algorithm1_dpf_rows"]
    if len(alg1_rows) != 4:
        raise P6Alg1ValidationError("P6 should preserve exactly four P5 Algorithm 1 rows")
    if payload["route_summaries"]["algorithm1_dpf_promoted_rows"]:
        raise P6Alg1ValidationError("P6 promoted Algorithm 1 DPF rows")
    if payload["route_summaries"]["algorithm1_dpf_diagnostic_rows"] != ["lgssm_2d_h25_rich"]:
        raise P6Alg1ValidationError("P6 should have exactly one Algorithm 1 diagnostic DPF row")
    for row in alg1_rows:
        if row["method_id"] != METHOD_ID:
            raise P6Alg1ValidationError("old or unexpected DPF method appeared")
        if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY":
            if not _has_algorithm1_route_fields(row["route_fields"]):
                raise P6Alg1ValidationError("Algorithm 1 row missing route fields")
            if row["valid_for_within_target_calibration"]:
                raise P6Alg1ValidationError("Algorithm 1 diagnostic row marked valid for calibration")
            if row["value_standard_error"] is None or row["gradient_error_norm_standard_error"] is None:
                raise P6Alg1ValidationError("Algorithm 1 diagnostic row lacks uncertainty")
        if row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER":
            if not row["blockers"]:
                raise P6Alg1ValidationError("blocked Algorithm 1 row lacks blockers")
            if row["value_rmse_vs_reference"] is not None or row["mean_gradient_error_norm"] is not None:
                raise P6Alg1ValidationError("blocked Algorithm 1 row has fabricated metrics")
    for row in payload["exact_target_calibration_rows"]:
        if row["source_phase"] == "historical_filter_oracle_P2_deterministic":
            if row["reference_uncertainty"]["status"] != "dense_refinement_recorded":
                raise P6Alg1ValidationError("P2 dense row lacks refinement uncertainty")
        if row["route_id"] in OLD_DPF_METHODS:
            raise P6Alg1ValidationError("old DPF appeared in deterministic calibration rows")
    for row in payload["approximation_target_calibration_rows"]:
        if row["target_class"] != "approximation_target":
            raise P6Alg1ValidationError("approximation row target class drifted")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P6 Result: Algorithm 1 UKF Cross-Filter Calibration Replacement",
        "",
        "metadata_date: 2026-06-10",
        "phase: P6",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        f"Status: `{payload['skeptical_plan_audit']['status']}`.",
        "",
        payload["skeptical_plan_audit"]["wrong_baseline_control"],
        "",
        payload["skeptical_plan_audit"]["proxy_promotion_control"],
        "",
        payload["skeptical_plan_audit"]["fairness_control"],
        "",
        "## Decision Table",
        "",
        "| Field | Status |",
        "| --- | --- |",
        "| decision | claim-class separated calibration ledger; no global ranking emitted |",
        f"| exact-target deterministic rows | `{summary['exact_target_row_count']}` |",
        f"| approximation-target deterministic rows | `{summary['approximation_target_row_count']}` |",
        f"| Algorithm 1 DPF rows | `{summary['algorithm1_dpf_row_count']}`; diagnostic `{summary['algorithm1_dpf_diagnostic_rows']}`; blocked `{summary['algorithm1_dpf_blocked_rows']}` |",
        f"| blocked rows | `{summary['blocked_row_count']}` |",
        "| primary uncertainty | Algorithm 1 P5 promotion bands and nonlinear same-target adapters remain missing; P3 exact-transformed and P44-M3/M4 deterministic metrics need structured reference-uncertainty JSON before calibration |",
        "| not concluded | global filter ranking, default-policy change, HMC readiness, production readiness, GPU readiness |",
        "",
        "## Exact-Target Deterministic Calibration Rows",
        "",
        "| target | dim | route | claim | abs value err | rel score err | ref uncertainty |",
        "| --- | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["exact_target_calibration_rows"]:
        lines.append(
            "| `{}` | {} | `{}` | `{}` | {:.6g} | {:.6g} | `{}` |".format(
                row["target_id"],
                row["dim"],
                row["route_id"],
                row["claim_class"],
                row["abs_value_error"],
                row["relative_score_error"],
                row["reference_uncertainty"]["status"],
            )
        )
    lines.extend(
        [
            "",
            "## Approximation-Target Deterministic Calibration Rows",
            "",
            "| target | dim | route | claim | abs value err | rel score err | reference |",
            "| --- | ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["approximation_target_calibration_rows"]:
        lines.append(
            "| `{}` | {} | `{}` | `{}` | {:.6g} | {:.6g} | `{}` |".format(
                row["target_id"],
                row["dim"],
                row["route_id"],
                row["claim_class"],
                row["abs_value_error"],
                row["relative_score_error"],
                row["reference_route_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Algorithm 1 DPF Rows",
            "",
            "| target | method | status | value RMSE/blocked | norm value RMSE | value SE | grad error/blocked | norm grad error | grad error SE |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["algorithm1_dpf_rows"]:
        value = row["value_rmse_vs_reference"] if isinstance(row.get("value_rmse_vs_reference"), (int, float)) else "N/A"
        norm_value = row["normalized_value_rmse"] if isinstance(row.get("normalized_value_rmse"), (int, float)) else "N/A"
        value_se = row["value_standard_error"] if isinstance(row.get("value_standard_error"), (int, float)) else "N/A"
        grad = row["mean_gradient_error_norm"] if isinstance(row.get("mean_gradient_error_norm"), (int, float)) else "N/A"
        norm_grad = (
            row["normalized_gradient_error_norm"]
            if isinstance(row.get("normalized_gradient_error_norm"), (int, float))
            else "N/A"
        )
        grad_se = (
            row["gradient_error_norm_standard_error"]
            if isinstance(row.get("gradient_error_norm_standard_error"), (int, float))
            else "N/A"
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['method_id']}` | `{row['replacement_status']}` | `{value}` | `{norm_value}` | `{value_se}` | `{grad}` | `{norm_grad}` | `{grad_se}` |"
        )
    lines.extend(
        [
            "",
            "## Historical Old DPF Quarantine",
            "",
            *[f"- `{item['path']}`: {item['status']}" for item in payload["historical_old_dpf_quarantine"]],
            "",
            "## Veto Diagnostics",
            "",
            "| Diagnostic | Status |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["veto_diagnostics"].items()],
            "",
            "## Blocked Or Unstructured",
            "",
            f"- Blocked row count: `{summary['blocked_row_count']}`.",
            f"- Unstructured metric row count: `{summary['unstructured_metric_row_count']}`.",
            "- Blocked/unstructured rows are not assigned value or score gaps in P6.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{JSON_PATH.relative_to(REPO_ROOT)}`",
            f"- Report: `{REPORT_PATH.relative_to(REPO_ROOT)}`",
            f"- Result: `{RESULT_PATH}`",
            "",
            "## Run Manifest",
            "",
            "```json",
            json.dumps(payload["run_manifest"], indent=2, sort_keys=True),
            "```",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    dirty = _git(["git", "status", "--short"])
    scoped_paths = [
        str(JSON_PATH.relative_to(REPO_ROOT)),
        str(REPORT_PATH.relative_to(REPO_ROOT)),
        RESULT_PATH,
        str(Path(__file__).relative_to(REPO_ROOT)),
        "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-visible-execution-ledger-2026-06-10.md",
    ]
    scoped_dirty = [
        line for line in dirty.splitlines() if any(path in line for path in scoped_paths)
    ]
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "command": f"{sys.executable} -m {MODULE_PATH}",
        "cpu_gpu_status": "pure_python_artifact_calibration_only; TensorFlow not imported",
        "tensorflow_imported": "tensorflow" in sys.modules,
        "tensorflow_probability_imported": "tensorflow_probability" in sys.modules,
        "timestamp_utc": _utc_now(),
        "dirty_state_line_count": len(dirty.splitlines()) if dirty else 0,
        "dirty_state_digest": _digest_payload({"dirty": dirty}),
        "scoped_dirty_state_summary": "\n".join(scoped_dirty) or "clean_for_p6_paths",
        "seeds": "consumed from P5/P3/P4 Algorithm 1 rows only",
        "particle_counts": "consumed from P5/P3/P4 Algorithm 1 rows only",
        "data_version": "historical deterministic filter-oracle artifacts plus P5/P3/P4 Algorithm 1 replacement artifacts",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _consumed_artifacts() -> list[dict[str, str]]:
    return [
        {"role": "historical_filter_oracle_registry", "path": str(REGISTRY_PATH.relative_to(REPO_ROOT))},
        {"role": "deterministic_p44_m2_context", "path": str(OLD_P2_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "deterministic_sv_context", "path": str(OLD_P3_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "historical_route_classification", "path": str(OLD_P4_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_p5_replacement", "path": str(P5_ALG1_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_p3_values", "path": str(P3_ALG1_VALUES_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_p4_gradients", "path": str(P4_ALG1_GRADIENTS_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "multistate_blocker_calibration", "path": str(P45_CALIBRATION_PATH.relative_to(REPO_ROOT))},
    ]


def _historical_old_dpf_quarantine() -> list[dict[str, str]]:
    return [
        {
            "path": "experiments/dpf_implementation/reports/outputs/dpf_filter_oracle_comparison_p5_dpf_statistical_closeness_2026-06-08.json",
            "status": "HISTORICAL_ONLY_NOT_CURRENT_ALGORITHM1_EVIDENCE",
        },
        {
            "path": "experiments/dpf_implementation/reports/outputs/dpf_filter_oracle_comparison_p6_cross_filter_error_calibration_2026-06-08.json",
            "status": "HISTORICAL_ONLY_NOT_CURRENT_ALGORITHM1_EVIDENCE",
        },
        {
            "path": "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py",
            "status": "QUARANTINED_OLD_IMPLEMENTATION_NOT_IMPORTED",
        },
    ]


def _nonclaims() -> list[str]:
    return [
        "P6 does not certify Algorithm 1 statistical closeness.",
        "P6 does not revive old dpf_ledh_pfpf_ot or dpf_bootstrap_ot rows as current evidence.",
        "P6 does not rank filters globally across incompatible targets.",
        "P6 does not use Zhao-Cui, CUT4, SVD, UKF, or FilterFlow as a DPF correctness oracle.",
        "P6 does not establish nonlinear P44 Algorithm 1 DPF value or gradient closeness.",
        "P6 does not establish stochastic-resampling gradient correctness.",
        "P6 does not establish HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
    ]


def _has_algorithm1_route_fields(route_fields: Any) -> bool:
    return isinstance(route_fields, dict) and all(route_fields.get(key) for key in ALG1_ROUTE_REQUIRED)


def _route_key(row: dict[str, Any]) -> str:
    return f"{row['target_id']}/dim{row['dim']}/{row['route_id']}"


def _max_abs(values: list[Any]) -> float | None:
    numbers = [abs(float(value)) for value in values if isinstance(value, (int, float))]
    return max(numbers) if numbers else None


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
