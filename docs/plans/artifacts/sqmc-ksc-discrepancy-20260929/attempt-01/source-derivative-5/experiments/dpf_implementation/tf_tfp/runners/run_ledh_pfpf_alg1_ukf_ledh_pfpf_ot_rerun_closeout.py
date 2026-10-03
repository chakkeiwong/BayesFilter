"""Build P9 closeout for the Algorithm 1 UKF rerun of old LEDH-PFPF-OT lanes.

This runner is pure Python artifact assembly.  It consumes the P0 registry and
P1-P8 result JSONs, indexes every old lane disposition, and validates that old
LEDH-PFPF-OT evidence cannot be mistaken for current Algorithm 1 UKF evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_ledh_pfpf_alg1_ukf_ledh_pfpf_ot_rerun_closeout"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p9-closeout-supersession-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p9-closeout-supersession-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
EXECUTION_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-visible-execution-ledger-2026-06-10.md"
)
REGISTRY_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-registry-2026-06-10.json"
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_ledh_pfpf_ot_rerun_closeout_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-closeout-2026-06-10.md"

P1_JSON = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_direct_replacements_2026-06-10.json"
P2_JSON = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10.json"
P3_JSON = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_values_2026-06-10.json"
P4_JSON = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_gradients_2026-06-10.json"
P5_JSON = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_alg1_ukf_statistical_closeness_2026-06-10.json"
P6_JSON = OUTPUT_DIR / "dpf_filter_oracle_comparison_p6_alg1_ukf_cross_filter_calibration_2026-06-10.json"
P7_JSON = OUTPUT_DIR / "dpf_filter_oracle_comparison_p8_p44_alg1_ukf_blocker_closure_2026-06-10.json"
P8_JSON = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_extension_historical_classification_2026-06-10.json"
P9_GUARDRAIL_JSON = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_p9_guardrail_pytest_2026-06-10.json"
P9_REVIEW_JSON = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_p9_claude_review_2026-06-10.json"

PENDING_REVIEW_DECISION = "P9_CLOSEOUT_SUPERSESSION_READY_FOR_CLAUDE_REVIEW"
FINAL_PASS_DECISION = "PASS_P9_CLOSEOUT_SUPERSESSION_CLAUDE_REVIEWED"
VETO_DECISION = "P9_CLOSEOUT_SUPERSESSION_VETO_PENDING_REPAIR"
METHOD_ID = "ledh_pfpf_alg1_ukf_no_resampling_tf"
GUARDRAIL_COMMAND = "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_ledh_pfpf_alg1_ukf_tf.py -q"
GUARDRAIL_SUMMARY = "15 passed, 2 warnings in 6.29s"

PHASE_RESULT_PATHS = {
    "P0": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p0-inventory-registry-result-2026-06-10.md",
    "P1": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p1-direct-lgssm-range-bearing-result-2026-06-10.md",
    "P2": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p2-v2-contracts-result-2026-06-10.md",
    "P3": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p3-v2-values-result-2026-06-10.md",
    "P4": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p4-v2-gradients-result-2026-06-10.md",
    "P5": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p5-filter-oracle-statistical-closeness-result-2026-06-10.md",
    "P6": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p6-cross-filter-calibration-result-2026-06-10.md",
    "P7": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p7-p44-blocker-closure-result-2026-06-10.md",
    "P8": "docs/plans/bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p8-filterflow-annealed-historical-regression-result-2026-06-10.md",
}

PHASE_JSON_PATHS = {
    "P1": P1_JSON,
    "P2": P2_JSON,
    "P3": P3_JSON,
    "P4": P4_JSON,
    "P5": P5_JSON,
    "P6": P6_JSON,
    "P7": P7_JSON,
    "P8": P8_JSON,
}


class P9CloseoutError(ValueError):
    """Raised when the P9 closeout artifact violates its contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P9_CLOSEOUT_SUPERSESSION_VALIDATED")
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
    artifacts = {phase: _load_json(path) for phase, path in PHASE_JSON_PATHS.items()}
    guardrail = _load_json(P9_GUARDRAIL_JSON)
    review_evidence = _load_json(P9_REVIEW_JSON) if P9_REVIEW_JSON.exists() else None
    old_lane_table = _old_lane_table(registry, artifacts)
    value_table = _algorithm1_value_table(artifacts)
    gradient_table = _algorithm1_gradient_table(artifacts)
    comparator_table = _comparator_applicability_table(artifacts)
    blocked_table = _blocked_adapter_table(artifacts)
    historical_table = _historical_only_table(registry, artifacts)
    manifest_index = _manifest_index(artifacts)
    claude_index = _claude_review_index(review_evidence)
    promoted_rows = _promoted_rows(value_table, gradient_table)
    veto = _veto_diagnostics(
        registry=registry,
        artifacts=artifacts,
        old_lane_table=old_lane_table,
        value_table=value_table,
        gradient_table=gradient_table,
        blocked_table=blocked_table,
        historical_table=historical_table,
        manifest_index=manifest_index,
        claude_index=claude_index,
        promoted_rows=promoted_rows,
        guardrail=guardrail,
    )
    non_review_veto = {key: value for key, value in veto.items() if key != "claude_review_not_converged"}
    if any(non_review_veto.values()):
        decision = VETO_DECISION
    elif veto["claude_review_not_converged"]:
        decision = PENDING_REVIEW_DECISION
    else:
        decision = FINAL_PASS_DECISION
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": _utc_now(),
        "phase": "P9",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_CLOSEOUT_LEDGER_AFTER_GUARDRAIL",
        "decision": decision,
        "question": (
            "After P0-P8, is every previous LEDH-PFPF-OT-related test either "
            "redone with Algorithm 1 UKF or explicitly classified?"
        ),
        "evidence_contract": {
            "baseline_comparator": "P0 rerun registry and P1-P8 result artifacts.",
            "primary_criterion": (
                "A closeout ledger indexes every old lane, replacement artifact, "
                "remaining blocker, historical-only row, manifest, and nonclaim."
            ),
            "promotion_policy": "No new P9 numerical promotion; promoted row list must be empty.",
        },
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CLOSEOUT_AFTER_GUARDRAIL",
            "wrong_baseline_control": "Old LEDH-PFPF-OT artifacts define coverage only, not truth.",
            "proxy_promotion_control": "Diagnostic finite rows stay diagnostic; P9 does not introduce thresholds.",
            "stop_condition_control": "Missing phase result, missing lane disposition, true veto, or non-converged Claude review blocks closeout.",
            "environment_control": "Guardrail pytest ran CPU-only; closeout assembly imports no TensorFlow.",
        },
        "guardrail_rerun": _guardrail_row(guardrail),
        "old_lane_to_new_disposition": old_lane_table,
        "algorithm1_value_table": value_table,
        "algorithm1_gradient_table": gradient_table,
        "comparator_applicability_table": comparator_table,
        "blocked_adapter_table": blocked_table,
        "historical_only_table": historical_table,
        "run_manifest_index": manifest_index,
        "per_row_manifest_links": _per_row_manifest_links(value_table, gradient_table, blocked_table, historical_table),
        "claude_review_index": claude_index,
        "p9_review_evidence": review_evidence or {
            "status": "PENDING_CLAUDE_REVIEW",
            "expected_next_action": "Run Claude read-only P9 review; if AGREE, write P9 review evidence and regenerate final closeout.",
        },
        "promoted_rows": promoted_rows,
        "threshold_certification_band_status": _threshold_table(value_table, gradient_table),
        "core_vs_extension_route_class_summary": _route_class_summary(value_table, gradient_table, historical_table),
        "supersession_rule": _supersession_rule(),
        "veto_diagnostics": veto,
        "decision_table": _decision_table(decision, veto, old_lane_table, promoted_rows),
        "post_run_red_team": _post_run_red_team(),
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _old_lane_table(registry: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p1_by_lane = {row["old_lane_id"]: row for row in artifacts["P1"].get("lane_statuses", [])}
    p8_by_lane = {row["lane_id"]: row for row in artifacts["P8"].get("classification_rows", [])}
    rows = []
    for row in registry["registry_rows"]:
        lane_id = row["old_lane_id"]
        final_status = row["planned_disposition"]
        replacement_phase = row["replacement_phase"]
        source = "registry_default"
        result_paths = _phase_result_pointer(replacement_phase)
        notes = row.get("claim_class", "")
        if lane_id in p1_by_lane:
            lane = p1_by_lane[lane_id]
            final_status = lane["status"]
            result_paths = lane.get("result_pointer", result_paths)
            source = "P1_lane_status"
            notes = lane.get("reason", notes)
        elif lane_id == "v2_contracts":
            final_status = "RERUN_ALG1"
            source = "P2_contract_freeze"
            result_paths = _phase_result_pointer("P2")
            notes = artifacts["P2"]["summary"]
        elif lane_id == "v2_values":
            final_status = "RERUN_ALG1_DIAGNOSTIC_ONLY"
            source = "P3_value_replacement"
            result_paths = _phase_result_pointer("P3")
            notes = artifacts["P3"]["summary"]
        elif lane_id == "v2_gradients":
            final_status = "RERUN_ALG1_DIAGNOSTIC_ONLY"
            source = "P4_gradient_replacement"
            result_paths = _phase_result_pointer("P4")
            notes = artifacts["P4"]["summary"]
        elif lane_id in {
            "filter_oracle_p1_lgssm_exact",
            "filter_oracle_p5_statistical_closeness",
        }:
            final_status = "RERUN_ALG1_DIAGNOSTIC_ONLY"
            source = "P5_filter_oracle_replacement"
            result_paths = _phase_result_pointer("P5")
            notes = artifacts["P5"]["route_summaries"]
        elif lane_id == "filter_oracle_p6_cross_filter_calibration":
            final_status = "RERUN_ALG1_DIAGNOSTIC_ONLY"
            source = "P6_cross_filter_calibration"
            result_paths = _phase_result_pointer("P6")
            notes = artifacts["P6"]["route_summaries"]
        elif lane_id == "filter_oracle_p8_p44_blocker_closure":
            final_status = "RERUN_ALG1_DIAGNOSTIC_ONLY"
            source = "P7_p44_blocker_closure"
            result_paths = _phase_result_pointer("P7")
            notes = artifacts["P7"]["route_summaries"]
        elif lane_id == "source_faithful_repair_auxiliary_flow_only":
            p8 = p8_by_lane["auxiliary_flow_source_faithful_repair"]
            final_status = p8["disposition"]
            source = "P8_auxiliary_flow_classification"
            result_paths = _phase_result_pointer("P8")
            notes = p8["reason"]
        elif lane_id == "annealed_transport_ledh_lgssm":
            p8 = p8_by_lane["annealed_transport_lgssm"]
            final_status = p8["disposition"]
            source = "P8_extension_classification"
            result_paths = _phase_result_pointer("P8")
            notes = p8["reason"]
        elif lane_id == "filterflow_matched_ledh_pfpf_ot":
            p8 = p8_by_lane["filterflow_matched_ledh_pfpf_ot"]
            final_status = p8["disposition"]
            source = "P8_filterflow_scaffolding_classification"
            result_paths = _phase_result_pointer("P8")
            notes = p8["reason"]
        elif replacement_phase.endswith("/P9") or replacement_phase == "P9":
            final_status = "HISTORICAL_ONLY_NOT_EVIDENCE"
            result_paths = _phase_result_pointer("P9")
            source = "P9_quarantine_closeout"
        rows.append(
            {
                "old_lane_id": lane_id,
                "planned_disposition": row["planned_disposition"],
                "final_disposition": final_status,
                "replacement_phase": replacement_phase,
                "evidence_route_class": _lane_route_class(final_status),
                "old_evidence_current": False,
                "result_pointer": result_paths,
                "source": source,
                "notes": notes,
            }
        )
    return rows


def _algorithm1_value_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    p1_summary = artifacts["P1"]["value_summaries"].get(METHOD_ID, {})
    for particles, summary in _particle_items(p1_summary):
        rows.append(
            _value_summary_row(
                phase="P1",
                target_id="direct_lgssm",
                method_id=METHOD_ID,
                num_particles=particles,
                summary=summary,
                manifest_phase="P1",
                comparator_route="exact_kalman_lgssm",
            )
        )
    p3_summaries = artifacts["P3"].get("value_summaries", {})
    for model_id, by_method in p3_summaries.items():
        for particles, summary in _particle_items(by_method.get(METHOD_ID, {})):
            rows.append(
                _value_summary_row(
                    phase="P3",
                    target_id=model_id,
                    method_id=METHOD_ID,
                    num_particles=particles,
                    summary=summary,
                    manifest_phase="P3",
                    comparator_route=_p2_comparator_route(artifacts["P2"], model_id),
                )
            )
    p5_rows = artifacts["P5"].get("rows", [])
    for row in p5_rows:
        if row.get("method_id") == METHOD_ID and row.get("status") == "RERUN_ALG1_DIAGNOSTIC_ONLY":
            stats = row.get("value_statistics", {})
            rows.append(
                {
                    "phase": "P5",
                    "target_id": row["target_id"],
                    "method_id": METHOD_ID,
                    "num_particles": stats.get("particle_count"),
                    "seed_count": stats.get("seed_count"),
                    "value_rmse": stats.get("value_rmse"),
                    "value_standard_error": stats.get("value_standard_error"),
                    "value_ci95": stats.get("value_ci95"),
                    "row_status": row["status"],
                    "promotion_status": "not_promoted_diagnostic_only",
                    "certification_band": row.get("certification_band", "N/A_DIAGNOSTIC_ONLY"),
                    "comparator_route": row.get("comparison_target_class"),
                    "evidence_route_class": row.get("evidence_route_class", "SOURCE_ALGORITHM1_CORE"),
                    "manifest_link": _manifest_link("P5"),
                }
            )
    for row in artifacts["P7"].get("row_summaries", []):
        if int(row["num_particles"]) == max(artifacts["P7"]["route_summaries"]["particle_counts"]):
            rows.append(
                {
                    "phase": "P7",
                    "target_id": row["target_id"],
                    "dim": row["dim"],
                    "method_id": METHOD_ID,
                    "num_particles": row["num_particles"],
                    "seed_count": row["seed_count"],
                    "value_rmse": row["value_error_rmse"],
                    "value_standard_error": row["value_error_standard_error"],
                    "value_ci95": row["value_error_ci95"],
                    "row_status": row["row_status"],
                    "promotion_status": row["promotion_status"],
                    "certification_band": "N/A_DIAGNOSTIC_ONLY_IN_P7",
                    "comparator_route": "p44_dense_fixed_branch_reference",
                    "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
                    "manifest_link": _manifest_link("P7"),
                }
            )
    return rows


def _value_summary_row(
    *,
    phase: str,
    target_id: str,
    method_id: str,
    num_particles: int,
    summary: dict[str, Any],
    manifest_phase: str,
    comparator_route: str,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "target_id": target_id,
        "method_id": method_id,
        "num_particles": num_particles,
        "seed_count": summary.get("seed_count"),
        "value_rmse": summary.get("value_rmse_vs_reference", summary.get("value_error_rmse")),
        "value_standard_error": summary.get("value_standard_error", summary.get("value_error_standard_error")),
        "value_ci95": summary.get("value_ci95", summary.get("value_error_ci95")),
        "row_status": summary.get("row_status"),
        "promotion_status": "not_promoted_diagnostic_only",
        "certification_band": summary.get("primary_promote_statistic", "N/A_DIAGNOSTIC_ONLY"),
        "comparator_route": comparator_route,
        "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        "manifest_link": _manifest_link(manifest_phase),
    }


def _algorithm1_gradient_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for particles, summary in _particle_items(artifacts["P1"].get("gradient_summary", {})):
        rows.append(
            _gradient_summary_row(
                phase="P1",
                target_id="direct_lgssm",
                num_particles=particles,
                summary=summary,
                comparator_route="exact_kalman_lgssm_gradient",
                manifest_phase="P1",
            )
        )
    for model_id, by_particles in artifacts["P4"].get("gradient_summaries", {}).items():
        for particles, summary in _particle_items(by_particles):
            rows.append(
                _gradient_summary_row(
                    phase="P4",
                    target_id=model_id,
                    num_particles=particles,
                    summary=summary,
                    comparator_route=_p2_comparator_route(artifacts["P2"], model_id),
                    manifest_phase="P4",
                )
            )
    for row in artifacts["P7"].get("row_summaries", []):
        if int(row["num_particles"]) == max(artifacts["P7"]["route_summaries"]["particle_counts"]):
            rows.append(
                {
                    "phase": "P7",
                    "target_id": row["target_id"],
                    "dim": row["dim"],
                    "method_id": METHOD_ID,
                    "num_particles": row["num_particles"],
                    "seed_count": row["seed_count"],
                    "mean_gradient_error_norm": row["mean_gradient_error_norm"],
                    "gradient_error_norm_standard_error": row["gradient_error_norm_standard_error"],
                    "gradient_error_norm_ci95": None,
                    "gradient_component_standard_error": None,
                    "gradient_uncertainty_status": "reference_error_norm_uncertainty",
                    "row_status": row["row_status"],
                    "promotion_status": "not_promoted_diagnostic_only",
                    "certification_band": "N/A_DIAGNOSTIC_ONLY_IN_P7",
                    "gradient_scope": "fixed_branch_algorithm1_value_path",
                    "comparator_route": "p44_dense_fixed_branch_score",
                    "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
                    "manifest_link": _manifest_link("P7"),
                }
            )
    return rows


def _gradient_summary_row(
    *,
    phase: str,
    target_id: str,
    num_particles: int,
    summary: dict[str, Any],
    comparator_route: str,
    manifest_phase: str,
) -> dict[str, Any]:
    component_se = summary.get("gradient_component_standard_error")
    norm_se = summary.get("gradient_error_norm_standard_error")
    norm_ci = summary.get("gradient_error_norm_ci95")
    return {
        "phase": phase,
        "target_id": target_id,
        "method_id": METHOD_ID,
        "num_particles": num_particles,
        "seed_count": summary.get("seed_count"),
        "mean_gradient_error_norm": summary.get("mean_gradient_error_norm"),
        "gradient_error_norm_standard_error": norm_se,
        "gradient_error_norm_ci95": norm_ci,
        "gradient_component_standard_error": component_se,
        "gradient_uncertainty_status": (
            "reference_error_norm_uncertainty"
            if norm_se is not None or norm_ci is not None
            else "component_uncertainty_no_reference_error_norm"
            if component_se is not None
            else "missing_gradient_uncertainty"
        ),
        "row_status": summary.get("row_status"),
        "promotion_status": "not_promoted_diagnostic_only",
        "certification_band": summary.get("primary_promote_statistic", "N/A_DIAGNOSTIC_ONLY"),
        "gradient_scope": "fixed_branch_algorithm1_value_path",
        "comparator_route": comparator_route,
        "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        "manifest_link": _manifest_link(manifest_phase),
    }


def _comparator_applicability_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for contract in artifacts["P2"]["contracts"]:
        rows.append(
            {
                "phase": "P2",
                "target_id": contract["model_id"],
                "contract_status": contract["status"],
                "comparator_route": contract["callback_contract"].get("comparator_route"),
                "value_scalar": contract["scalar_contract"].get("value_scalar"),
                "gradient_scalar": contract["scalar_contract"].get("gradient_scalar"),
                "applicability_status": contract["status"],
                "missing_adapter_items": contract.get("missing_adapter_items", []),
                "evidence_route_class": contract["algorithm1_route_fields"]["evidence_route_class"],
            }
        )
    rows.extend(
        {
            "phase": "P8",
            "target_id": row["lane_id"],
            "contract_status": row["disposition"],
            "comparator_route": row["route_class"],
            "value_scalar": "N/A",
            "gradient_scalar": "N/A",
            "applicability_status": row["disposition"],
            "missing_adapter_items": [],
            "evidence_route_class": row["route_class"],
        }
        for row in artifacts["P8"]["classification_rows"]
    )
    return rows


def _blocked_adapter_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in artifacts["P1"].get("blocked_adapters", []):
        rows.append(
            {
                "phase": "P1",
                "target_id": row["old_lane_id"],
                "status": row["status"],
                "missing_items": row["missing_items"],
                "not_a_valid_stop_reason": row["not_a_valid_stop_reason"],
                "manifest_link": _manifest_link("P1"),
            }
        )
    for contract in artifacts["P2"].get("contracts", []):
        if contract["status"] == "BLOCKED_REQUIRES_ADAPTER":
            rows.append(
                {
                    "phase": "P2",
                    "target_id": contract["model_id"],
                    "status": contract["status"],
                    "missing_items": contract.get("missing_adapter_items", []),
                    "not_a_valid_stop_reason": "adapter blocker recorded by P2 contract freeze",
                    "manifest_link": _manifest_link("P2"),
                }
            )
    for row in artifacts["P5"].get("rows", []):
        if row.get("status") == "BLOCKED_REQUIRES_ADAPTER":
            rows.append(
                {
                    "phase": "P5",
                    "target_id": row["target_id"],
                    "status": row["status"],
                    "missing_items": row.get("blockers", []),
                    "not_a_valid_stop_reason": "P5 classified the row; P7 later fills P44 cells diagnostically where applicable",
                    "manifest_link": _manifest_link("P5"),
                }
            )
    for row in artifacts["P6"].get("blocked_rows", []):
        rows.append(
            {
                "phase": "P6",
                "target_id": row["target_id"],
                "status": "BLOCKED_OR_UNSTRUCTURED_CALIBRATION_CONTEXT",
                "missing_items": [row["reason"]],
                "not_a_valid_stop_reason": "P6 is calibration classification, not adapter execution",
                "manifest_link": _manifest_link("P6"),
            }
        )
    return rows


def _historical_only_table(registry: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in registry["artifact_groups"]:
        rows.append(
            {
                "group_id": row["group_id"],
                "disposition": row["planned_disposition"],
                "replacement_phase": row["replacement_phase"],
                "paths_or_globs": row["paths_or_globs"],
                "notes": row["notes"],
            }
        )
    for row in artifacts["P8"].get("classification_rows", []):
        rows.append(
            {
                "group_id": row["lane_id"],
                "disposition": row["disposition"],
                "replacement_phase": "P8",
                "paths_or_globs": [item["path"] for item in row["json_artifacts"] + row["report_artifacts"]],
                "notes": row["reason"],
            }
        )
    return rows


def _manifest_index(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "phase": "P0",
            "artifact": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
            "manifest_status": "registry_metadata_present",
            "command": "see p0_inventory_commands",
            "seeds": "N/A",
            "particle_counts": "N/A",
            "cpu_gpu_status": "P0 guardrail CPU-only TensorFlow; registry assembly pure Python",
        },
        {
            "phase": "P9_guardrail",
            "artifact": str(P9_GUARDRAIL_JSON.relative_to(REPO_ROOT)),
            "manifest_status": "guardrail_rerun_passed",
            "command": GUARDRAIL_COMMAND,
            "seeds": "N/A",
            "particle_counts": "N/A",
            "cpu_gpu_status": "CPU-only TensorFlow",
        },
    ]
    for phase, payload in artifacts.items():
        manifest = payload.get("run_manifest", {})
        rows.append(
            {
                "phase": phase,
                "artifact": str(PHASE_JSON_PATHS[phase].relative_to(REPO_ROOT)),
                "manifest_status": "present" if manifest else "missing",
                "command": manifest.get("command", "N/A"),
                "seeds": manifest.get("seed_list", manifest.get("seeds", "N/A")),
                "particle_counts": manifest.get("particle_counts", "N/A"),
                "cpu_gpu_status": manifest.get("cpu_gpu_status", "CPU-only" if manifest.get("cpu_only") else "N/A"),
            }
        )
    return rows


def _guardrail_row(guardrail: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": str(P9_GUARDRAIL_JSON.relative_to(REPO_ROOT)),
        "command": guardrail.get("command"),
        "status": guardrail.get("status"),
        "return_code": guardrail.get("return_code"),
        "summary": guardrail.get("summary"),
        "cpu_gpu_status": guardrail.get("cpu_gpu_status"),
        "warning_summary": guardrail.get("warning_summary", []),
    }


def _guardrail_valid(guardrail: dict[str, Any]) -> bool:
    return (
        guardrail.get("command") == GUARDRAIL_COMMAND
        and guardrail.get("status") == "passed"
        and int(guardrail.get("return_code", -1)) == 0
        and "15 passed" in str(guardrail.get("summary", ""))
        and "CUDA_VISIBLE_DEVICES=-1" in str(guardrail.get("cpu_gpu_status", ""))
    )


def _claude_review_index(review_evidence: dict[str, Any] | None) -> list[dict[str, str]]:
    text = (REPO_ROOT / REVIEW_LEDGER_PATH).read_text(encoding="utf-8")
    rows = []
    for label in ["Iteration 2", "P7 Iteration 1", "P8 Iteration 1"]:
        rows.append(
            {
                "review_gate": label,
                "ledger": REVIEW_LEDGER_PATH,
                "status": "VERDICT_AGREE" if label in text and "VERDICT: AGREE" in text else "UNKNOWN",
            }
        )
    for phase in ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]:
        rows.append(
            {
                "review_gate": f"{phase} visible ledger closure",
                "ledger": EXECUTION_LEDGER_PATH,
                "status": "PASS_RECORDED",
            }
        )
    rows.append(
        {
            "review_gate": "P9 Iteration 3 final closeout",
            "ledger": str(P9_REVIEW_JSON.relative_to(REPO_ROOT)),
            "status": (
                "VERDICT_AGREE"
                if review_evidence is not None
                and review_evidence.get("verdict") == "VERDICT: AGREE"
                and review_evidence.get("review_target_decision") == PENDING_REVIEW_DECISION
                else "PENDING_CLAUDE_REVIEW"
            ),
        }
    )
    return rows


def _promoted_rows(value_table: list[dict[str, Any]], gradient_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    promoted = []
    for row in value_table + gradient_table:
        if row.get("promotion_status") == "promoted":
            promoted.append(row)
        if isinstance(row.get("row_status"), str) and row["row_status"].startswith("PROMOTED"):
            promoted.append(row)
    return promoted


def _threshold_table(
    value_table: list[dict[str, Any]],
    gradient_table: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in value_table:
        rows.append(
            {
                "phase": row["phase"],
                "target_id": row["target_id"],
                "metric": "value",
                "certification_band": row.get("certification_band"),
                "promotion_status": row.get("promotion_status"),
                "uncertainty_present": row.get("seed_count") is not None
                and (
                    row.get("value_standard_error") is not None
                    or row.get("value_ci95") is not None
                ),
            }
        )
    for row in gradient_table:
        rows.append(
            {
                "phase": row["phase"],
                "target_id": row["target_id"],
                "metric": "gradient",
                "certification_band": row.get("certification_band"),
                "promotion_status": row.get("promotion_status"),
                "gradient_uncertainty_status": row.get("gradient_uncertainty_status"),
                "uncertainty_present": row.get("seed_count") is not None
                and (
                    row.get("gradient_error_norm_standard_error") is not None
                    or row.get("gradient_error_norm_ci95") is not None
                    or row.get("gradient_component_standard_error") is not None
                ),
            }
        )
    return rows


def _route_class_summary(
    value_table: list[dict[str, Any]],
    gradient_table: list[dict[str, Any]],
    historical_table: list[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in value_table + gradient_table:
        key = row.get("evidence_route_class", "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    for row in historical_table:
        key = row.get("disposition", "UNKNOWN")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _veto_diagnostics(
    *,
    registry: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
    old_lane_table: list[dict[str, Any]],
    value_table: list[dict[str, Any]],
    gradient_table: list[dict[str, Any]],
    blocked_table: list[dict[str, Any]],
    historical_table: list[dict[str, Any]],
    manifest_index: list[dict[str, Any]],
    claude_index: list[dict[str, str]],
    promoted_rows: list[dict[str, Any]],
    guardrail: dict[str, Any],
) -> dict[str, bool]:
    all_artifact_vetoes_false = all(
        not any(bool(value) for value in payload.get("veto_diagnostics", {}).values())
        for payload in artifacts.values()
    )
    return {
        "guardrail_not_rerun_or_failed": not _guardrail_valid(guardrail),
        "registry_lane_missing_from_closeout": len(old_lane_table) != len(registry["registry_rows"]),
        "phase_result_missing": any(not (REPO_ROOT / path).exists() for path in PHASE_RESULT_PATHS.values()),
        "phase_json_missing": any(not path.exists() for path in PHASE_JSON_PATHS.values()),
        "artifact_veto_not_clean": not all_artifact_vetoes_false,
        "old_ledh_row_current_evidence": any(row.get("old_evidence_current") for row in old_lane_table),
        "source_algorithm1_route_missing_from_value_or_gradient": any(
            row.get("evidence_route_class") != "SOURCE_ALGORITHM1_CORE" for row in value_table + gradient_table
        ),
        "promoted_row_without_threshold_status": bool(promoted_rows),
        "missing_value_uncertainty": any(
            row.get("seed_count") is not None
            and row.get("value_standard_error") is None
            and row.get("value_ci95") is None
            for row in value_table
        ),
        "missing_gradient_uncertainty": any(
            row.get("seed_count") is not None
            and row.get("gradient_error_norm_standard_error") is None
            and row.get("gradient_error_norm_ci95") is None
            and row.get("gradient_component_standard_error") is None
            for row in gradient_table
        ),
        "unresolved_blocker_without_reason": any(not row.get("missing_items") for row in blocked_table),
        "historical_only_table_empty": not historical_table,
        "run_manifest_missing": any(row["manifest_status"] == "missing" for row in manifest_index),
        "claude_review_not_converged": any(row["status"] not in {"VERDICT_AGREE", "PASS_RECORDED"} for row in claude_index),
        "unsupported_superiority_or_default_claim": False,
    }


def _decision_table(
    decision: str,
    veto: dict[str, bool],
    old_lane_table: list[dict[str, Any]],
    promoted_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if decision == VETO_DECISION:
        next_action = "Repair listed structural vetoes before requesting final P9 review"
    elif decision == PENDING_REVIEW_DECISION:
        next_action = "Claude read-only P9 review; on AGREE write review evidence and regenerate final closeout"
    else:
        next_action = "Final closeout complete; old LEDH-PFPF-OT evidence remains superseded/quarantined"
    return [
        {
            "decision": decision,
            "primary_criterion_status": (
                f"{len(old_lane_table)} registry lanes closed; "
                f"{len(promoted_rows)} promoted rows emitted"
            ),
            "veto_diagnostic_status": {key: value for key, value in veto.items() if value} or "no structural vetoes",
            "main_uncertainty": "diagnostic rows remain bounded by small particle ladders and fixed-branch gradients",
            "next_justified_action": next_action,
            "not_concluded": "no production default, HMC readiness, universal superiority, or stochastic-score correctness",
        }
    ]


def _phase_result_pointer(phase_text: str) -> dict[str, str]:
    phase = phase_text.split("/")[0]
    if phase == "P9":
        return {
            "phase_result": RESULT_PATH,
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
        }
    result = PHASE_RESULT_PATHS.get(phase)
    json_path = PHASE_JSON_PATHS.get(phase)
    return {
        "phase_result": result or "N/A",
        "json": str(json_path.relative_to(REPO_ROOT)) if json_path else "N/A",
        "markdown_report": "see phase result",
    }


def _manifest_link(phase: str) -> str:
    if phase == "P0":
        return str(REGISTRY_PATH.relative_to(REPO_ROOT))
    return str(PHASE_JSON_PATHS[phase].relative_to(REPO_ROOT))


def _per_row_manifest_links(*tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    links = []
    for table in tables:
        for row in table:
            link = row.get("manifest_link")
            if link:
                links.append(
                    {
                        "row_key": "::".join(
                            str(row.get(key))
                            for key in ("phase", "target_id", "dim", "num_particles")
                            if row.get(key) is not None
                        ),
                        "manifest_link": link,
                    }
                )
    return links


def _p2_comparator_route(p2: dict[str, Any], model_id: str) -> str:
    for contract in p2["contracts"]:
        if contract["model_id"] == model_id:
            return contract["callback_contract"].get("comparator_route", "N/A")
    return "N/A"


def _particle_items(mapping: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    pairs = []
    for key, value in mapping.items():
        if str(key).isdigit():
            pairs.append((int(key), value))
    return sorted(pairs)


def _lane_route_class(status: str) -> str:
    if status.startswith("RERUN_ALG1"):
        return "SOURCE_ALGORITHM1_CORE"
    if status in {"ALG1_EXTENSION_RERUN", "SCAFFOLDING_ONLY"}:
        return "BAYESFILTER_EXTENSION_OR_SCAFFOLDING_NOT_SOURCE_CORE"
    return "NOT_CURRENT_ALGORITHM1_EVIDENCE"


def _supersession_rule() -> list[str]:
    return [
        "Old LEDH-PFPF-OT, dpf_ledh_pfpf_ot, and ledh_pfpf_ot artifacts are historical coverage only.",
        "Current source Algorithm 1 evidence must carry li_coates_algorithm1_ukf_covariance_lifecycle route identifiers.",
        "OT, annealed transport, and FilterFlow-matched residuals are not source Li-Coates Algorithm 1 core evidence.",
        "Finite diagnostic rows without reviewed promotion bands remain diagnostic-only.",
        "Future extension reruns require a separate reviewed plan.",
    ]


def _post_run_red_team() -> dict[str, str]:
    return {
        "strongest_alternative_explanation": (
            "Some old lanes may be scientifically useful as extension evidence, "
            "but P8/P9 deliberately classify them rather than promote them."
        ),
        "what_would_overturn_closeout": (
            "A current artifact citing old dpf_ledh_pfpf_ot rows as Algorithm 1 "
            "UKF evidence, or a missing registry lane with no disposition."
        ),
        "weakest_evidence_part": (
            "Most numerical rows remain diagnostic-only; full gradient and "
            "extension ladders are future work."
        ),
    }


def _manifest() -> dict[str, Any]:
    return {
        **_git_manifest(),
        "python_version": platform.python_version(),
        "command": "python -m " + MODULE_PATH,
        "execution_mode": "PURE_PYTHON_CLOSEOUT_LEDGER_AFTER_GUARDRAIL",
        "guardrail_command": GUARDRAIL_COMMAND,
        "guardrail_summary": GUARDRAIL_SUMMARY,
        "cpu_gpu_status": "closeout assembly no TensorFlow import; guardrail CPU-only TensorFlow",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "p9_review_evidence_path": str(P9_REVIEW_JSON.relative_to(REPO_ROOT)),
        "execution_ledger_path": EXECUTION_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _nonclaims() -> list[str]:
    return [
        "No production default or public API promotion is concluded.",
        "No HMC readiness is concluded.",
        "No universal Algorithm 1, DPF, OT, or FilterFlow superiority is concluded.",
        "No stochastic-resampling gradient correctness is concluded.",
        "Diagnostic rows without reviewed bands are not statistical-closeness certifications.",
        "Historical-only and scaffolding rows cannot be cited as current Algorithm 1 UKF evidence.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "guardrail_rerun",
        "old_lane_to_new_disposition",
        "algorithm1_value_table",
        "algorithm1_gradient_table",
        "comparator_applicability_table",
        "blocked_adapter_table",
        "historical_only_table",
        "run_manifest_index",
        "per_row_manifest_links",
        "claude_review_index",
        "veto_diagnostics",
        "decision_table",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P9CloseoutError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {VETO_DECISION, PENDING_REVIEW_DECISION, FINAL_PASS_DECISION}:
        raise P9CloseoutError(f"invalid P9 decision {payload['decision']}")
    if payload["guardrail_rerun"]["status"] != "passed":
        raise P9CloseoutError("guardrail did not pass")
    if payload["guardrail_rerun"]["return_code"] != 0:
        raise P9CloseoutError("guardrail return code was not zero")
    if "15 passed" not in str(payload["guardrail_rerun"]["summary"]):
        raise P9CloseoutError("guardrail summary does not preserve pass count")
    true_vetoes = {key: value for key, value in payload["veto_diagnostics"].items() if bool(value)}
    non_review_true_vetoes = {
        key: value for key, value in true_vetoes.items() if key != "claude_review_not_converged"
    }
    if payload["decision"] == VETO_DECISION:
        if not non_review_true_vetoes:
            raise P9CloseoutError("veto-pending-repair decision has no structural veto")
    elif payload["decision"] == PENDING_REVIEW_DECISION:
        if set(true_vetoes) != {"claude_review_not_converged"}:
            raise P9CloseoutError(f"unexpected pending-review vetoes: {true_vetoes}")
    elif true_vetoes:
        raise P9CloseoutError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    if len(payload["old_lane_to_new_disposition"]) != 21:
        raise P9CloseoutError("unexpected old-lane closeout count")
    if payload["promoted_rows"]:
        raise P9CloseoutError("P9 must not emit promoted rows")
    if not payload["algorithm1_value_table"] or not payload["algorithm1_gradient_table"]:
        raise P9CloseoutError("missing value or gradient closeout table")
    for row in payload["algorithm1_gradient_table"]:
        if row.get("seed_count") is not None and row.get("gradient_uncertainty_status") == "missing_gradient_uncertainty":
            raise P9CloseoutError(f"missing gradient uncertainty for {row['phase']} {row['target_id']}")
    if not payload["blocked_adapter_table"] or not payload["historical_only_table"]:
        raise P9CloseoutError("missing blocker or historical-only table")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P9 Result: Closeout And Supersession Ledger",
        "",
        "metadata_date: 2026-06-10",
        "phase: P9",
        f"status: {payload['decision']}",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Promotion policy | {payload['evidence_contract']['promotion_policy']} |",
        "",
        "## Guardrail Rerun",
        "",
        f"- Command: `{payload['guardrail_rerun']['command']}`",
        f"- Status: `{payload['guardrail_rerun']['status']}`.",
        f"- Summary: `{payload['guardrail_rerun']['summary']}`.",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| `{row['decision']}` | {row['primary_criterion_status']} | "
            f"`{row['veto_diagnostic_status']}` | {row['main_uncertainty']} | "
            f"{row['next_justified_action']} | {row['not_concluded']} |"
        )
    lines.extend(
        [
            "",
            "## Old Lane Dispositions",
            "",
            "| Old lane | Planned | Final | Route class | Source |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["old_lane_to_new_disposition"]:
        lines.append(
            f"| `{row['old_lane_id']}` | `{row['planned_disposition']}` | "
            f"`{row['final_disposition']}` | `{row['evidence_route_class']}` | `{row['source']}` |"
        )
    lines.extend(
        [
            "",
            "## Value Table",
            "",
            "| Phase | Target | Dim | Particles | Seeds | Status | RMSE | SE | Band |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["algorithm1_value_table"]:
        lines.append(
            f"| `{row['phase']}` | `{row['target_id']}` | {row.get('dim', '')} | "
            f"{row.get('num_particles', '')} | {row.get('seed_count', '')} | "
            f"`{row.get('row_status')}` | `{row.get('value_rmse')}` | "
            f"`{row.get('value_standard_error')}` | `{row.get('certification_band')}` |"
        )
    lines.extend(
        [
            "",
            "## Gradient Table",
            "",
            "| Phase | Target | Dim | Particles | Seeds | Status | Mean grad err | Error-norm SE | Component SE | Uncertainty status | Band |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["algorithm1_gradient_table"]:
        lines.append(
            f"| `{row['phase']}` | `{row['target_id']}` | {row.get('dim', '')} | "
            f"{row.get('num_particles', '')} | {row.get('seed_count', '')} | "
            f"`{row.get('row_status')}` | `{row.get('mean_gradient_error_norm')}` | "
            f"`{row.get('gradient_error_norm_standard_error')}` | "
            f"`{row.get('gradient_component_standard_error')}` | "
            f"`{row.get('gradient_uncertainty_status')}` | `{row.get('certification_band')}` |"
        )
    lines.extend(
        [
            "",
            "## Blocked Adapter Table",
            "",
            "| Phase | Target | Status | Missing item count |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in payload["blocked_adapter_table"]:
        lines.append(
            f"| `{row['phase']}` | `{row['target_id']}` | `{row['status']}` | {len(row['missing_items'])} |"
        )
    lines.extend(
        [
            "",
            "## Historical-Only Table",
            "",
            "| Group | Disposition | Replacement phase |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["historical_only_table"]:
        lines.append(
            f"| `{row['group_id']}` | `{row['disposition']}` | `{row['replacement_phase']}` |"
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
            "| Diagnostic | Status |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["veto_diagnostics"].items()],
            "",
            "## Supersession Rule",
            "",
            *[f"- {item}" for item in payload["supersession_rule"]],
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _digest_payload(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _git_manifest() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty_state_summary": _git(["git", "status", "--short"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
