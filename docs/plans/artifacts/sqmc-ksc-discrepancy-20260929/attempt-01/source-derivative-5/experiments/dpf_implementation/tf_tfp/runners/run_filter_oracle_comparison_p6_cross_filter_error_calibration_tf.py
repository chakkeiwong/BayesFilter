"""Run P6 cross-filter error calibration artifact assembly.

P6 is a pure-Python artifact builder.  It normalizes already-written P1-P5
value and score error rows into claim-class-separated ledgers.  It does not run
new filters and does not create a global ranking across incompatible targets.
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
P1_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p1_lgssm_exact_oracle_2026-06-08.json"
P2_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_2026-06-08.json"
P3_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p3_conditional_gaussian_mixture_2026-06-08.json"
P4_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json"
P5_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_dpf_statistical_closeness_2026-06-08.json"
P45_CALIBRATION_PATH = REPO_ROOT / "docs/plans/bayesfilter-highdim-zhao-cui-p45-cross-model-error-calibration-2026-06-08.json"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p6_cross_filter_error_calibration_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p6-cross-filter-error-calibration-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p6-cross-filter-error-calibration-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p6-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p6_cross_filter_error_calibration_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p6-cross-filter-error-calibration-2026-06-08.md"

REFERENCE_FLOOR = 1e-12
DPF_METHODS = {"dpf_bootstrap_ot", "dpf_ledh_pfpf_ot"}


class P6ValidationError(ValueError):
    """Raised when a P6 artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P6_CROSS_FILTER_ERROR_CALIBRATION_VALIDATED")
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
    p1 = _load_json(P1_JSON_PATH)
    p2 = _load_json(P2_JSON_PATH)
    p3 = _load_json(P3_JSON_PATH)
    p4 = _load_json(P4_JSON_PATH)
    p5 = _load_json(P5_JSON_PATH)
    p45 = _load_json(P45_CALIBRATION_PATH)

    exact_rows = []
    exact_rows.extend(_p2_exact_target_rows(p2))
    p3_exact_unstructured_rows = _p3_exact_transformed_unstructured_rows(p3)

    approximation_rows = _p3_ksc_approximation_rows(p3)
    dpf_rows = _p5_dpf_diagnostic_rows(p5)
    blocked_rows = _blocked_rows(registry, p4, p5, p45)
    unstructured_rows = _unstructured_rows(p4) + p3_exact_unstructured_rows
    explanatory_gaps = _p3_explanatory_approximation_gaps(p3)

    summaries = _summaries(exact_rows, approximation_rows, dpf_rows, blocked_rows, unstructured_rows)
    veto = _veto_diagnostics(exact_rows, approximation_rows, dpf_rows, blocked_rows, unstructured_rows, p5)
    decision = (
        "PASS_P6_CROSS_FILTER_CALIBRATION_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P6_CROSS_FILTER_CALIBRATION_VETO_PENDING_REVIEW"
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": _utc_now(),
        "phase": "P6",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_ARTIFACT_CALIBRATION_ONLY",
        "decision": decision,
        "question": (
            "How large are available DPF, UKF, SVD/sigma-point, CUT4, and "
            "Zhao-Cui value and score errors relative to target-specific "
            "reference scales, without ranking incompatible rows or promoting "
            "diagnostic/blocker evidence?"
        ),
        "skeptical_plan_audit": {
            "status": "PASS_FOR_ARTIFACT_CALIBRATION_ONLY",
            "wrong_baseline_control": "Every row records its target and exact or approximation reference route.",
            "proxy_promotion_control": "DPF P5 rows remain diagnostic/blocked and are excluded from valid calibration tables.",
            "stop_condition_control": "Rows with missing structured metrics are placed in blocked or unstructured ledgers.",
            "fairness_control": "Exact-target and approximation-target tables are separate; no global route ranking is emitted.",
            "environment_control": "Pure-Python assembly consumes existing JSON and does not import TensorFlow.",
        },
        "evidence_contract": {
            "baseline_comparator": (
                "P1-P5 row-level artifacts; dense/Kalman references remain "
                "target-specific and claim-class separated."
            ),
            "primary_criterion": (
                "Produce separated exact-target, approximation-target, DPF "
                "diagnostic, blocked, and unstructured ledgers with reference "
                "uncertainty or evaluator variance where applicable."
            ),
            "ranking_policy": "no_global_ranking; route rows may be compared only within the same target and evidence class",
            "data_law_variability_policy": "not_used; no rmse/sd_L promotion or bias excuse is claimed",
        },
        "consumed_artifacts": _consumed_artifacts(),
        "exact_target_calibration_rows": exact_rows,
        "approximation_target_calibration_rows": approximation_rows,
        "dpf_diagnostic_rows": dpf_rows,
        "blocked_rows": blocked_rows,
        "unstructured_metric_rows": unstructured_rows,
        "explanatory_approximation_gaps": explanatory_gaps,
        "route_summaries": summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "reference_floor_for_refinement_ratios": REFERENCE_FLOOR,
            "data_law_variability_used": False,
            "global_ranking_emitted": False,
            "dpf_evaluator_variance_reported_for_diagnostic_rows": True,
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
                    source_phase="P2",
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


def _p3_exact_transformed_unstructured_rows(p3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in p3["rows"]["exact_transformed"]:
        route = source["candidate"]
        reference = source["reference"]
        rows.append(
            {
                "source_phase": "P3",
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


def _p3_ksc_approximation_rows(p3: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in p3["rows"]["ksc_mixture"]:
        route = source["candidate"]
        reference = source["reference"]
        rows.append(
            _calibration_row(
                source_phase="P3",
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


def _p5_dpf_diagnostic_rows(p5: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in p5["rows"]:
        if row["method_id"] not in DPF_METHODS:
            continue
        stats = row["consumed_statistics"]
        rows.append(
            {
                "source_phase": "P5",
                "target_id": row["target_id"],
                "method_id": row["method_id"],
                "row_decision": row["row_decision"],
                "reference_route": row["reference_route"],
                "comparison_target_class": row["comparison_target_class"],
                "promotion_status": row["promotion_status"],
                "valid_for_calibration_table": False,
                "reason_not_valid_for_calibration": (
                    "DPF row was not promoted in P5; retained as diagnostic or blocked evidence."
                ),
                "seed_count": stats.get("seed_count"),
                "particle_counts": stats.get("particle_counts"),
                "value_error_rmse": stats.get("value_error_rmse"),
                "value_error_standard_error": stats.get("value_error_standard_error"),
                "value_error_ci95": stats.get("value_error_ci95"),
                "score_rmse": stats.get("score_rmse"),
                "gradient_error_norm_standard_error": stats.get(
                    "gradient_error_norm_standard_error"
                ),
                "mean_relative_score_error": stats.get("mean_relative_score_error"),
                "blockers": row.get("blockers", []),
                "downgrade_reasons": row.get("downgrade_reasons", []),
            }
        )
    return rows


def _blocked_rows(
    registry: dict[str, Any],
    p4: dict[str, Any],
    p5: dict[str, Any],
    p45: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for row in p5["rows"]:
        if row["row_decision"] == "BLOCKED":
            rows.append(
                {
                    "source_phase": "P5",
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
    p4_p5_targets = set(p4["route_summaries"]["p5_dpf_eligible_targets"])
    for target_id, route_map in registry["route_matrix"].items():
        if target_id in p4_p5_targets:
            continue
        for route_id in DPF_METHODS:
            route = route_map[route_id]
            if route["claim_class"] == "BLOCKED":
                rows.append(
                    {
                        "source_phase": "P0/P4",
                        "target_id": target_id,
                        "route_id": route_id,
                        "reason": "; ".join(route["blockers"]) or route["route_status"],
                        "value_gap": None,
                        "relative_score_error": None,
                        "valid_for_calibration_table": False,
                    }
                )
    return rows


def _unstructured_rows(p4: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in p4["rows"]:
        target_id = row["target_id"]
        if target_id not in {
            "p44_m3_quadratic_observation_panel",
            "p44_m4_nonlinear_transition_h2_panel",
        }:
            continue
        rows.append(
            {
                "source_phase": "P4/P44",
                "target_id": target_id,
                "route_id": "deterministic_routes_from_p44_source_notes",
                "status": "structured_row_metrics_not_available_in_P1_P5_json",
                "source_artifacts": row["classification"]["source_artifacts"],
                "why_not_calibrated": (
                    "P4 records source-artifact summaries, but P6 does not parse "
                    "markdown tables into metrics. A future phase should add a "
                    "machine-readable P44-M3/M4 metric artifact before calibration."
                ),
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
                "source_phase": "P3",
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
    dpf_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    unstructured_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "exact_target_row_count": len(exact_rows),
        "approximation_target_row_count": len(approximation_rows),
        "dpf_diagnostic_row_count": len(dpf_rows),
        "blocked_row_count": len(blocked_rows),
        "unstructured_metric_row_count": len(unstructured_rows),
        "valid_within_target_calibration_rows": [
            _route_key(row) for row in exact_rows + approximation_rows if row["valid_for_within_target_calibration"]
        ],
        "diagnostic_only_rows": [
            _route_key(row)
            for row in exact_rows
            if row["claim_class"] == "DIAGNOSTIC_ONLY"
        ]
        + [
            f"{row['target_id']}/{row['method_id']}"
            for row in dpf_rows
            if row["row_decision"] == "DOWNGRADED_TO_DIAGNOSTIC_ONLY"
        ],
        "global_ranking_policy": "not_emitted",
        "max_abs_value_error_by_evidence_table": {
            "exact_target": _max_abs([row["abs_value_error"] for row in exact_rows]),
            "approximation_target": _max_abs([row["abs_value_error"] for row in approximation_rows]),
            "dpf_diagnostic_rmse": _max_abs(
                [
                    row["value_error_rmse"]
                    for row in dpf_rows
                    if isinstance(row["value_error_rmse"], (int, float))
                ]
            ),
        },
        "max_relative_score_error_by_evidence_table": {
            "exact_target": _max_abs([row["relative_score_error"] for row in exact_rows]),
            "approximation_target": _max_abs([row["relative_score_error"] for row in approximation_rows]),
            "dpf_diagnostic_mean": _max_abs(
                [
                    row["mean_relative_score_error"]
                    for row in dpf_rows
                    if isinstance(row["mean_relative_score_error"], (int, float))
                ]
            ),
        },
    }


def _veto_diagnostics(
    exact_rows: list[dict[str, Any]],
    approximation_rows: list[dict[str, Any]],
    dpf_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    unstructured_rows: list[dict[str, Any]],
    p5: dict[str, Any],
) -> dict[str, bool]:
    del unstructured_rows
    return {
        "global_ranking_emitted": False,
        "data_law_variability_used_to_excuse_mismatch": False,
        "approximation_route_ranked_as_exact": any(
            row["target_class"] != "approximation_target"
            for row in approximation_rows
            if row["reference_route_id"] == "ksc_kalman_mixture_reference"
        ),
        "reference_uncertainty_omitted_from_p2_dense_rows": any(
            row["source_phase"] == "P2"
            and row["reference_uncertainty"]["status"] != "dense_refinement_recorded"
            for row in exact_rows
        ),
        "exact_target_row_lacks_accepted_reference_uncertainty": any(
            row["target_class"] == "exact_target"
            and row["reference_uncertainty"]["status"]
            not in {"dense_refinement_recorded"}
            for row in exact_rows
        ),
        "dpf_evaluator_variance_omitted": any(
            row["method_id"] in DPF_METHODS
            and row["row_decision"] == "DOWNGRADED_TO_DIAGNOSTIC_ONLY"
            and (
                row["value_error_standard_error"] is None
                or row["gradient_error_norm_standard_error"] is None
            )
            for row in dpf_rows
        ),
        "dpf_promoted_row_in_valid_calibration_table": any(
            row["method_id"] in DPF_METHODS and row["valid_for_calibration_table"]
            for row in dpf_rows
        )
        or bool(p5["route_summaries"]["promoted_rows"]),
        "blocked_row_has_metric": any(
            row["valid_for_calibration_table"]
            or row["value_gap"] is not None
            or row["relative_score_error"] is not None
            for row in blocked_rows
        ),
        "nonfinite_calibration_row": any(
            not row["finite"] for row in exact_rows + approximation_rows
        ),
        "fabricated_metric_for_unstructured_row": False,
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "execution_mode",
        "evidence_contract",
        "exact_target_calibration_rows",
        "approximation_target_calibration_rows",
        "dpf_diagnostic_rows",
        "blocked_rows",
        "unstructured_metric_rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P6ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P6_CROSS_FILTER_CALIBRATION_PENDING_CLAUDE_REVIEW",
        "P6_CROSS_FILTER_CALIBRATION_VETO_PENDING_REVIEW",
    }:
        raise P6ValidationError(f"invalid P6 decision {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_ARTIFACT_CALIBRATION_ONLY":
        raise P6ValidationError("P6 should be pure-Python artifact calibration only")
    if payload["run_manifest"].get("tensorflow_imported"):
        raise P6ValidationError("TensorFlow was imported in P6")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P6ValidationError(f"P6 veto fired: {payload['veto_diagnostics']}")
    if not payload["exact_target_calibration_rows"]:
        raise P6ValidationError("P6 should include available exact-target deterministic rows")
    if not payload["approximation_target_calibration_rows"]:
        raise P6ValidationError("P6 should include available approximation-target rows")
    if not payload["dpf_diagnostic_rows"]:
        raise P6ValidationError("P6 should preserve P5 DPF diagnostic/blocker rows")
    for row in payload["exact_target_calibration_rows"]:
        if row["source_phase"] == "P2" and row["reference_uncertainty"]["status"] != "dense_refinement_recorded":
            raise P6ValidationError("P2 dense row lacks refinement uncertainty")
        if row["reference_uncertainty"]["status"] not in {"dense_refinement_recorded"}:
            raise P6ValidationError("exact-target calibration row lacks accepted reference uncertainty")
        if row["route_id"] in DPF_METHODS:
            raise P6ValidationError("DPF appeared in exact-target calibration rows")
    for row in payload["approximation_target_calibration_rows"]:
        if row["target_class"] != "approximation_target":
            raise P6ValidationError("approximation row target class drifted")
    for row in payload["dpf_diagnostic_rows"]:
        if row["valid_for_calibration_table"]:
            raise P6ValidationError("DPF diagnostic row was marked valid for calibration")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P6 Result: Cross-Filter Error Calibration",
        "",
        "metadata_date: 2026-06-08",
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
        "| decision | calibration tables are claim-class separated; no global ranking emitted |",
        f"| exact-target rows | `{summary['exact_target_row_count']}` |",
        f"| approximation-target rows | `{summary['approximation_target_row_count']}` |",
        f"| DPF diagnostic rows | `{summary['dpf_diagnostic_row_count']}`; not valid calibration rows |",
        f"| blocked rows | `{summary['blocked_row_count']}` |",
        "| primary uncertainty | DPF promotion bands and nonlinear same-target adapters remain missing; P3 exact-transformed and P44-M3/M4 metrics need structured reference-uncertainty JSON before calibration |",
        "| not concluded | global filter ranking, default-policy change, HMC readiness, production readiness, GPU readiness |",
        "",
        "## Exact-Target Calibration Rows",
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
            "## Approximation-Target Calibration Rows",
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
            "## DPF Diagnostic Rows",
            "",
            "| target | method | decision | value RMSE/blocked | value SE | grad-norm SE | mean relative score err |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["dpf_diagnostic_rows"]:
        value = row["value_error_rmse"] if isinstance(row["value_error_rmse"], (int, float)) else "N/A"
        rel = (
            row["mean_relative_score_error"]
            if isinstance(row["mean_relative_score_error"], (int, float))
            else "N/A"
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['method_id']}` | `{row['row_decision']}` | `{value}` | `{row['value_error_standard_error']}` | `{row['gradient_error_norm_standard_error']}` | `{rel}` |"
        )
    lines.extend(
        [
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
        ]
    )
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    lines.append("")
    return "\n".join(lines)


def _manifest() -> dict[str, Any]:
    dirty = _git(["git", "status", "--short"])
    scoped_paths = [
        str(JSON_PATH.relative_to(REPO_ROOT)),
        str(REPORT_PATH.relative_to(REPO_ROOT)),
        RESULT_PATH,
        REVIEW_LEDGER_PATH,
        str(Path(__file__).relative_to(REPO_ROOT)),
        "docs/plans/bayesfilter-dpf-filter-oracle-comparison-visible-execution-ledger-2026-06-08.md",
        "docs/plans/bayesfilter-dpf-filter-oracle-comparison-visible-gated-execution-runbook-2026-06-08.md",
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
        "seeds": "reused only in DPF diagnostic rows from P1/P5",
        "particle_counts": "reused only in DPF diagnostic rows from P1/P5",
        "data_version": "P1-P5 local artifacts; no new observations generated",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _consumed_artifacts() -> list[dict[str, str]]:
    return [
        {"role": "registry", "path": str(REGISTRY_PATH.relative_to(REPO_ROOT))},
        {"role": "p1_lgssm", "path": str(P1_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p2_p44_m2", "path": str(P2_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p3_sv", "path": str(P3_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p4_route_classification", "path": str(P4_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p5_dpf_statistical_closeness", "path": str(P5_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p45_multistate_blocker_calibration", "path": str(P45_CALIBRATION_PATH.relative_to(REPO_ROOT))},
    ]


def _nonclaims() -> list[str]:
    return [
        "no global filter ranking across incompatible targets",
        "no DPF correctness or stochastic-score correctness claim",
        "no approximation route exactness claim",
        "no data-law variability tolerance used to excuse bias",
        "no default-policy change",
        "no HMC readiness, production readiness, GPU readiness, or paper-scale claim",
    ]


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
    raise SystemExit(main())
