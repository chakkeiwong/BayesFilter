"""Run P5 DPF statistical-closeness evidence classification.

P5 consumes reviewed P0-P4 artifacts and classifies every P0/P4 DPF-eligible
row as promoted, diagnostic/downgraded, or blocked.  This runner is deliberately
pure Python: it does not launch a new TensorFlow DPF experiment, because P0 left
the DPF P5 promotion bands as placeholders and the nonlinear P44 DPF adapters
are not yet reviewed same-target executable routes.
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
P4_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p5_dpf_statistical_closeness_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p5-dpf-statistical-closeness-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p5-dpf-statistical-closeness-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p5-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_dpf_statistical_closeness_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p5-dpf-statistical-closeness-2026-06-08.md"

DPF_ROUTE_IDS = ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot")
EXPECTED_P5_TARGETS = {
    "lgssm_2d_h25_rich",
    "p44_m2_cubic_additive_gaussian_panel",
    "p44_m3_quadratic_observation_panel",
    "p44_m4_nonlinear_transition_h2_panel",
}
PLACEHOLDER_TOLERANCE = "dpf_p5_pending_mc_band"
PLACEHOLDER_BAND = "p5_pending_mc_band"
FINAL_PARTICLE_COUNT = "128"


class P5ValidationError(ValueError):
    """Raised when a P5 classification artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P5_DPF_STATISTICAL_CLOSENESS_VALIDATED")
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
    p4 = _load_json(P4_JSON_PATH)

    eligible_targets = _p4_p5_targets(p4)
    rows = []
    for target_id in eligible_targets:
        for route_id in DPF_ROUTE_IDS:
            rows.append(_classify_dpf_row(registry, p1, p2, p4, target_id, route_id))

    summaries = _summaries(rows, eligible_targets)
    veto = _phase_veto_diagnostics(registry, p1, p4, rows)
    decision = (
        "PASS_P5_DPF_STATISTICAL_CLOSENESS_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P5_DPF_STATISTICAL_CLOSENESS_VETO_PENDING_REVIEW"
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": _utc_now(),
        "phase": "P5",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_EVIDENCE_CLASSIFICATION_ONLY",
        "decision": decision,
        "question": (
            "For P0/P4 DPF-eligible rows, can bootstrap-OT and LEDH-PFPF-OT "
            "be promoted as statistically close in value and fixed-branch "
            "gradient to the approved exact/dense references, or must the rows "
            "be downgraded or blocked?"
        ),
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CLASSIFICATION_ONLY_EXECUTION",
            "wrong_baseline_control": (
                "P5 consumes P0/P4 reference-route decisions and P1 Kalman "
                "evidence; it does not compare against BayesFilter-owned "
                "FilterFlow adapters or Zhao-Cui approximations."
            ),
            "proxy_promotion_control": (
                "Placeholder P0 DPF bands, missing directional residuals, and "
                "aggregate-only branch decisions prevent promotion; finite "
                "values or BF/FilterFlow agreement cannot substitute."
            ),
            "stop_condition_control": (
                "Every eligible target-route row is classified as diagnostic/"
                "downgraded or blocked; no row is silently omitted."
            ),
            "fairness_control": (
                "LGSSM exact-target evidence and P44 dense-reference rows are "
                "kept separate; P44 rows are blocked until reviewed same-target "
                "DPF adapters and evaluator variance exist."
            ),
            "environment_control": (
                "Pure-Python classification avoids TensorFlow import and does "
                "not run a hidden numerical experiment."
            ),
        },
        "evidence_contract": {
            "baseline_comparator": (
                "P0 registry and P4 P5-readiness route: exact Kalman for LGSSM; "
                "dense_refined_quadrature for P44-M2/M3/M4."
            ),
            "primary_criterion": (
                "Promote only if row-level numeric P5 tolerance/band, fixed "
                "branch score, paired seeds, evaluator variance, directional "
                "residuals, branch decisions, and value/gradient CI/max-error "
                "criteria are all present and passing."
            ),
            "current_execution_scope": (
                "No new DPF run is performed. Existing P1 LGSSM multi-seed "
                "evidence is consumed; P44 DPF rows are blocked pending reviewed "
                "same-target adapters and variance artifacts."
            ),
            "promotion_policy": (
                "No row is promoted in P5 because DPF P0 promotion bands are "
                "placeholders, not numeric predeclared criteria."
            ),
        },
        "reference_tables": _reference_tables(registry, p4, eligible_targets),
        "consumed_artifacts": _consumed_artifacts(),
        "rows": rows,
        "route_summaries": summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "p1_lgssm_evidence_reused_without_promotion": True,
            "p44_dpf_adapters_not_implemented_in_p5": True,
            "zhao_cui_not_used_as_p5_comparator": True,
            "bf_filterflow_agreement_not_used_as_oracle": True,
            "value_closeness_not_used_to_promote_gradient_closeness": True,
            "stochastic_score_claim": "not_claimed",
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _classify_dpf_row(
    registry: dict[str, Any],
    p1: dict[str, Any],
    p2: dict[str, Any],
    p4: dict[str, Any],
    target_id: str,
    route_id: str,
) -> dict[str, Any]:
    target = next(item for item in registry["targets"] if item["target_id"] == target_id)
    route = registry["route_matrix"][target_id][route_id]
    p4_row = next(row for row in p4["rows"] if row["target_id"] == target_id)
    reference = p4_row["reference_context"]
    base = {
        "target_id": target_id,
        "method_id": route_id,
        "model_family": target["model_family"],
        "reference_route": reference["route_id"],
        "reference_claim_class": reference["claim_class"],
        "comparison_target_class": (
            "exact_target_reference"
            if reference["claim_class"] == "EXACT_ORACLE"
            else "approximation_target_reference"
        ),
        "p0_route": {
            "route_status": route["route_status"],
            "phase_eligibility": route["phase_eligibility"],
            "promotion_tolerance": route["promotion_tolerance"],
            "certification_band": route["certification_band"],
            "primary_gradient_statistic": route["primary_gradient_statistic"],
            "seed_evaluator_variance_policy": route["seed_evaluator_variance_policy"],
        },
        "gradient_object": "fixed_branch_score",
        "stochastic_score_claim": "not_claimed",
        "zhao_cui_is_p5_comparator": p4_row["p5_readiness"]["zhao_cui_is_p5_comparator"],
    }
    placeholder_reasons = _placeholder_band_reasons(route)
    if target_id == "lgssm_2d_h25_rich":
        return {
            **base,
            **_lgssm_diagnostic_decision(p1, route_id, placeholder_reasons),
        }
    return {
        **base,
        **_p44_blocked_decision(target_id, route_id, p2, p4_row, placeholder_reasons),
    }


def _lgssm_diagnostic_decision(
    p1: dict[str, Any],
    route_id: str,
    placeholder_reasons: list[str],
) -> dict[str, Any]:
    summary = p1["method_summaries"][route_id][FINAL_PARTICLE_COUNT]
    final_rows = [
        row
        for row in p1["rows"]
        if row["method_id"] == route_id
        and int(row["num_particles"]) == int(FINAL_PARTICLE_COUNT)
    ]
    value_ci = summary["value_error_ci95"]
    value_ci_includes_zero = float(value_ci[0]) <= 0.0 <= float(value_ci[1])
    missing_promotion_fields = [
        "directional_derivative_residuals_missing_in_consumed_p1_artifact",
        "per_time_branch_decisions_missing_aggregate_counts_only",
    ]
    downgrade_reasons = [
        *placeholder_reasons,
        *missing_promotion_fields,
        "p1_artifact_explicitly_records_no_dpf_correctness_promotion",
    ]
    if not value_ci_includes_zero:
        downgrade_reasons.append("final_particle_count_mean_value_error_ci_excludes_zero")
    return {
        "row_decision": "DOWNGRADED_TO_DIAGNOSTIC_ONLY",
        "promotion_status": "not_promoted",
        "row_evidence_status": "finite_p1_lgssm_multiseed_evidence_available",
        "blockers": [],
        "downgrade_reasons": downgrade_reasons,
        "consumed_statistics": {
            "source_phase": "P1",
            "source_json": str(P1_JSON_PATH.relative_to(REPO_ROOT)),
            "final_particle_count": int(FINAL_PARTICLE_COUNT),
            "seed_count": int(summary["seed_count"]),
            "particle_counts": list(p1["stochastic_contract"]["particle_counts"]),
            "value_error_mean": summary["mean_value_error"],
            "value_error_standard_error": summary["value_error_standard_error"],
            "value_error_ci95": summary["value_error_ci95"],
            "value_error_rmse": summary["value_error_rmse"],
            "max_abs_value_error": summary["max_abs_value_error"],
            "mean_per_observation_value_error": _mean(
                row["per_observation_value_error"] for row in final_rows
            ),
            "max_abs_per_observation_value_error": max(
                abs(float(row["per_observation_value_error"])) for row in final_rows
            ),
            "gradient_error_mean": summary["gradient_error_mean"],
            "gradient_error_standard_error": summary["gradient_error_standard_error"],
            "gradient_error_ci95": summary["gradient_error_ci95"],
            "gradient_error_rmse_by_coordinate": summary["gradient_error_rmse_by_coordinate"],
            "score_rmse": summary["score_rmse"],
            "gradient_error_norm_standard_error": summary["gradient_error_norm_standard_error"],
            "gradient_error_norm_ci95": summary["gradient_error_norm_ci95"],
            "mean_relative_score_error": summary["mean_relative_score_error"],
            "max_relative_score_error": summary["max_relative_score_error"],
            "mean_gradient_cosine_similarity": _mean(
                row["gradient_cosine_similarity"] for row in final_rows
            ),
            "min_gradient_cosine_similarity": min(
                float(row["gradient_cosine_similarity"]) for row in final_rows
            ),
            "directional_derivative_residuals": "missing_in_consumed_p1_artifact",
            "branch_decision_record": "aggregate_only_resampling_count_and_transport_trigger_count",
            "mean_resampling_count": summary["mean_resampling_count"],
            "max_sinkhorn_residual": summary["max_sinkhorn_residual"],
            "min_ess": summary["min_ess"],
            "all_rows_finite": summary["all_rows_finite"],
            "third_particle_count_included": p1["method_summaries"][route_id]["particle_ladder"][
                "third_particle_count_included"
            ],
            "third_particle_count_reasons": p1["method_summaries"][route_id]["particle_ladder"][
                "third_particle_count_reasons"
            ],
        },
    }


def _p44_blocked_decision(
    target_id: str,
    route_id: str,
    p2: dict[str, Any],
    p4_row: dict[str, Any],
    placeholder_reasons: list[str],
) -> dict[str, Any]:
    del route_id
    source_artifacts = list(p4_row["classification"]["source_artifacts"])
    if target_id == p2["target"]["target_id"]:
        source_artifacts.append(str(P2_JSON_PATH.relative_to(REPO_ROOT)))
    blockers = [
        *placeholder_reasons,
        "reviewed_same_target_dpf_adapter_to_dense_reference_missing_for_this_target",
        "paired_seed_evaluator_variance_artifact_missing_for_this_target",
        "fixed_branch_gradient_tied_to_dense_scalar_not_yet_demonstrated_for_this_target",
        "directional_derivative_residuals_and_branch_decisions_missing_for_this_target",
    ]
    return {
        "row_decision": "BLOCKED",
        "promotion_status": "not_promoted",
        "row_evidence_status": "reference_available_but_dpf_execution_artifact_missing",
        "blockers": blockers,
        "downgrade_reasons": [],
        "consumed_statistics": {
            "source_phase": "P2/P4/P44 reference classification only",
            "reference_source_artifacts": source_artifacts,
            "dense_reference_route_status": p4_row["reference_context"]["route_status"],
            "dense_reference_claim_class": p4_row["reference_context"]["claim_class"],
            "dpf_value_gradient_rows": "missing",
            "seed_count": "missing",
            "particle_counts": "missing",
            "evaluator_variance": "missing",
            "directional_derivative_residuals": "missing",
            "branch_decision_record": "missing",
        },
    }


def _placeholder_band_reasons(route: dict[str, Any]) -> list[str]:
    reasons = []
    if route["promotion_tolerance"] == PLACEHOLDER_TOLERANCE:
        reasons.append("p0_dpf_promotion_tolerance_is_placeholder_not_numeric")
    if route["certification_band"] == PLACEHOLDER_BAND:
        reasons.append("p0_dpf_certification_band_is_placeholder_not_numeric")
    return reasons


def _reference_tables(
    registry: dict[str, Any],
    p4: dict[str, Any],
    eligible_targets: list[str],
) -> dict[str, Any]:
    exact_rows = []
    approximation_rows = []
    for target_id in eligible_targets:
        p4_row = next(row for row in p4["rows"] if row["target_id"] == target_id)
        ref = p4_row["reference_context"]
        target = next(item for item in registry["targets"] if item["target_id"] == target_id)
        entry = {
            "target_id": target_id,
            "model_family": target["model_family"],
            "reference_route": ref["route_id"],
            "claim_class": ref["claim_class"],
            "route_status": ref["route_status"],
            "promotion_tolerance": ref["promotion_tolerance"],
            "primary_gradient_statistic": ref["primary_gradient_statistic"],
        }
        if ref["claim_class"] == "EXACT_ORACLE":
            exact_rows.append(entry)
        else:
            approximation_rows.append(entry)
    return {
        "exact_reference_rows": exact_rows,
        "approximation_reference_rows": approximation_rows,
        "p5_comparator_policy": (
            "Only exact Kalman or dense_refined_quadrature references are used. "
            "Zhao-Cui/CUT4/SVD/UKF rows are not P5 DPF comparators here."
        ),
    }


def _summaries(rows: list[dict[str, Any]], eligible_targets: list[str]) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    for row in rows:
        decision_counts[row["row_decision"]] = decision_counts.get(row["row_decision"], 0) + 1
    return {
        "eligible_targets": eligible_targets,
        "eligible_target_count": len(eligible_targets),
        "row_count": len(rows),
        "promoted_rows": [
            _row_key(row) for row in rows if row["row_decision"].startswith("PROMOTED")
        ],
        "downgraded_rows": [
            _row_key(row) for row in rows if row["row_decision"] == "DOWNGRADED_TO_DIAGNOSTIC_ONLY"
        ],
        "blocked_rows": [_row_key(row) for row in rows if row["row_decision"] == "BLOCKED"],
        "decision_counts": decision_counts,
        "promotion_count": sum(1 for row in rows if row["row_decision"].startswith("PROMOTED")),
        "diagnostic_or_downgraded_count": sum(
            1 for row in rows if row["row_decision"] == "DOWNGRADED_TO_DIAGNOSTIC_ONLY"
        ),
        "blocked_count": sum(1 for row in rows if row["row_decision"] == "BLOCKED"),
        "final_status_interpretation": (
            "All P5-eligible DPF rows are classified, but no row is promoted. "
            "LGSSM rows are diagnostic/downgraded; P44 rows are blocked pending "
            "reviewed same-target DPF adapters and numeric P5 bands."
        ),
    }


def _phase_veto_diagnostics(
    registry: dict[str, Any],
    p1: dict[str, Any],
    p4: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, bool]:
    p5_targets = set(_p4_p5_targets(p4))
    row_keys = {_row_key(row) for row in rows}
    expected_row_keys = {
        f"{target_id}/{route_id}"
        for target_id in EXPECTED_P5_TARGETS
        for route_id in DPF_ROUTE_IDS
    }
    promoted = [row for row in rows if row["row_decision"].startswith("PROMOTED")]
    return {
        "eligible_target_set_mismatch": p5_targets != EXPECTED_P5_TARGETS,
        "eligible_row_omitted": row_keys != expected_row_keys,
        "promoted_with_placeholder_band": any(
            row["p0_route"]["promotion_tolerance"] == PLACEHOLDER_TOLERANCE
            or row["p0_route"]["certification_band"] == PLACEHOLDER_BAND
            for row in promoted
        ),
        "p44_row_promoted_without_reviewed_dpf_adapter": any(
            row["target_id"].startswith("p44_") and row["row_decision"].startswith("PROMOTED")
            for row in rows
        ),
        "lgssm_p1_evidence_missing": "method_summaries" not in p1 or "rows" not in p1,
        "missing_seed_variability_in_consumed_lgssm_evidence": any(
            row["target_id"] == "lgssm_2d_h25_rich"
            and int(row["consumed_statistics"]["seed_count"]) < 5
            for row in rows
        ),
        "stochastic_and_fixed_branch_gradients_mixed": any(
            row["gradient_object"] != "fixed_branch_score"
            or row["stochastic_score_claim"] != "not_claimed"
            for row in rows
        ),
        "value_closeness_used_to_promote_gradient": False,
        "same_target_and_approximation_rows_merged": False,
        "zhao_cui_used_as_p5_comparator": any(row["zhao_cui_is_p5_comparator"] for row in rows),
        "tensorflow_imported_in_classification_runner": (
            "tensorflow" in sys.modules or "tensorflow_probability" in sys.modules
        ),
        "registry_missing_expected_target": any(
            target_id not in registry["route_matrix"] for target_id in EXPECTED_P5_TARGETS
        ),
    }


def _p4_p5_targets(p4: dict[str, Any]) -> list[str]:
    return list(p4["route_summaries"]["p5_dpf_eligible_targets"])


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "execution_mode",
        "skeptical_plan_audit",
        "evidence_contract",
        "reference_tables",
        "rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P5ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P5_DPF_STATISTICAL_CLOSENESS_PENDING_CLAUDE_REVIEW",
        "P5_DPF_STATISTICAL_CLOSENESS_VETO_PENDING_REVIEW",
    }:
        raise P5ValidationError(f"invalid P5 decision {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_EVIDENCE_CLASSIFICATION_ONLY":
        raise P5ValidationError("P5 runner should be pure-Python classification only")
    if payload["run_manifest"].get("tensorflow_imported"):
        raise P5ValidationError("TensorFlow was imported in P5 classification runner")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P5ValidationError(f"P5 phase veto fired: {payload['veto_diagnostics']}")
    if set(payload["route_summaries"]["eligible_targets"]) != EXPECTED_P5_TARGETS:
        raise P5ValidationError("P5 eligible target set does not match P4/P0")
    if payload["route_summaries"]["row_count"] != 2 * len(EXPECTED_P5_TARGETS):
        raise P5ValidationError("P5 should classify two DPF routes for every eligible target")
    if payload["route_summaries"]["promotion_count"] != 0:
        raise P5ValidationError("P5 should not promote rows with placeholder DPF bands")
    if payload["route_summaries"]["diagnostic_or_downgraded_count"] != 2:
        raise P5ValidationError("P5 should downgrade the two LGSSM DPF rows to diagnostic only")
    if payload["route_summaries"]["blocked_count"] != 6:
        raise P5ValidationError("P5 should block the six P44 DPF rows")
    for row in payload["rows"]:
        if row["p0_route"]["primary_gradient_statistic"] != "fixed_branch_score":
            raise P5ValidationError(f"{_row_key(row)} gradient statistic drifted")
        if row["p0_route"]["promotion_tolerance"] != PLACEHOLDER_TOLERANCE:
            raise P5ValidationError(f"{_row_key(row)} unexpected DPF tolerance")
        if row["p0_route"]["certification_band"] != PLACEHOLDER_BAND:
            raise P5ValidationError(f"{_row_key(row)} unexpected DPF band")
        if row["row_decision"].startswith("PROMOTED"):
            raise P5ValidationError(f"{_row_key(row)} was incorrectly promoted")
        if row["row_decision"] == "BLOCKED" and not row["blockers"]:
            raise P5ValidationError(f"{_row_key(row)} blocked row lacks blockers")
        if row["row_decision"] == "DOWNGRADED_TO_DIAGNOSTIC_ONLY":
            stats = row["consumed_statistics"]
            if int(stats["seed_count"]) < 5:
                raise P5ValidationError(f"{_row_key(row)} missing seed variability")
            if len(stats["particle_counts"]) < 2:
                raise P5ValidationError(f"{_row_key(row)} missing particle ladder")
            if stats["directional_derivative_residuals"] != "missing_in_consumed_p1_artifact":
                raise P5ValidationError("unexpected directional residual status")
            if not row["downgrade_reasons"]:
                raise P5ValidationError(f"{_row_key(row)} downgraded row lacks reasons")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P5 Result: DPF Statistical Closeness",
        "",
        "metadata_date: 2026-06-08",
        "phase: P5",
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
        "| decision | no DPF row promoted |",
        f"| primary criterion | all `{summary['row_count']}` eligible target-route rows classified |",
        "| veto diagnostics | clear at phase level; row-level blockers/downgrades recorded |",
        "| main uncertainty | numeric P5 DPF bands and reviewed nonlinear same-target adapters are still missing |",
        "| next justified action | P6 may calibrate cross-filter error; a later amendment is needed before DPF nonlinear promotion runs |",
        "| not concluded | DPF correctness, stochastic-score correctness, HMC readiness, production readiness, GPU readiness |",
        "",
        "## Row Summary",
        "",
        f"- Eligible targets: `{summary['eligible_targets']}`.",
        f"- Promoted rows: `{summary['promoted_rows']}`.",
        f"- Downgraded/diagnostic rows: `{summary['downgraded_rows']}`.",
        f"- Blocked rows: `{summary['blocked_rows']}`.",
        "",
        "## Exact Reference Rows",
        "",
        "| target | reference | DPF decision |",
        "| --- | --- | --- |",
    ]
    for ref in payload["reference_tables"]["exact_reference_rows"]:
        target_rows = [row for row in payload["rows"] if row["target_id"] == ref["target_id"]]
        decisions = ", ".join(f"{row['method_id']}={row['row_decision']}" for row in target_rows)
        lines.append(f"| `{ref['target_id']}` | `{ref['reference_route']}` | {decisions} |")
    lines.extend(
        [
            "",
            "## LGSSM DPF Diagnostics",
            "",
            "| method | N | mean value err | value CI95 | score RMSE | mean relative score err | decision |",
            "| --- | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["rows"]:
        if row["target_id"] != "lgssm_2d_h25_rich":
            continue
        stats = row["consumed_statistics"]
        lines.append(
            "| `{}` | {} | {:.6g} | `{}` | {:.6g} | {:.6g} | `{}` |".format(
                row["method_id"],
                stats["final_particle_count"],
                stats["value_error_mean"],
                stats["value_error_ci95"],
                stats["score_rmse"],
                stats["mean_relative_score_error"],
                row["row_decision"],
            )
        )
    lines.extend(
        [
            "",
            "## Blocked Nonlinear DPF Rows",
            "",
            "| target | method | primary blocker |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["rows"]:
        if row["row_decision"] != "BLOCKED":
            continue
        lines.append(
            f"| `{row['target_id']}` | `{row['method_id']}` | {row['blockers'][0]} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Interpretation",
            "",
            "- LGSSM bootstrap-OT and LEDH-PFPF-OT have finite paired multi-seed P1 evidence, but they remain diagnostic in P5.",
            "- P44-M2/M3/M4 references are available, but DPF statistical closeness is blocked until same-target adapters, evaluator variance, branch decisions, and numeric P5 bands are reviewed.",
            "- Zhao-Cui, CUT4, SVD, and UKF rows are not used as P5 DPF comparators.",
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
        line
        for line in dirty.splitlines()
        if any(path in line for path in scoped_paths)
    ]
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "command": f"{sys.executable} -m {MODULE_PATH}",
        "cpu_gpu_status": "pure_python_classification_only; TensorFlow not imported",
        "tensorflow_imported": "tensorflow" in sys.modules,
        "tensorflow_probability_imported": "tensorflow_probability" in sys.modules,
        "timestamp_utc": _utc_now(),
        "dirty_state_line_count": len(dirty.splitlines()) if dirty else 0,
        "dirty_state_digest": _digest_payload({"dirty": dirty}),
        "scoped_dirty_state_summary": "\n".join(scoped_dirty) or "clean_for_p5_paths",
        "seed_list": "reused from P1 for LGSSM only: [101, 202, 303, 404, 505]",
        "particle_counts": "reused from P1 for LGSSM only: [32, 64, 128]",
        "data_version": "P0-P4 local artifacts; no new DPF observations generated",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _consumed_artifacts() -> list[dict[str, str]]:
    return [
        {"role": "registry", "path": str(REGISTRY_PATH.relative_to(REPO_ROOT))},
        {"role": "p1_lgssm_dpf_kalman_evidence", "path": str(P1_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p2_p44_m2_dense_reference", "path": str(P2_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "p4_p5_eligibility", "path": str(P4_JSON_PATH.relative_to(REPO_ROOT))},
    ]


def _nonclaims() -> list[str]:
    return [
        "P5 does not promote DPF bootstrap-OT or LEDH-PFPF-OT correctness.",
        "P5 does not establish stochastic-resampling distribution correctness.",
        "P5 does not use BF/FilterFlow agreement as oracle evidence.",
        "P5 does not use Zhao-Cui, CUT4, SVD, or UKF as DPF comparators.",
        "P5 does not establish nonlinear P44 DPF value or gradient closeness.",
        "P5 does not establish HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
    ]


def _row_key(row: dict[str, Any]) -> str:
    return f"{row['target_id']}/{row['method_id']}"


def _mean(values: Any) -> float:
    numbers = [float(value) for value in values]
    if not numbers:
        return math.nan
    return sum(numbers) / len(numbers)


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
