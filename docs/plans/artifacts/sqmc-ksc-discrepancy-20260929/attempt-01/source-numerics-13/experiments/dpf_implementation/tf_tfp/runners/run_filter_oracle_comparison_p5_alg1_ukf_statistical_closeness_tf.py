"""Classify P5 filter-oracle rows with Algorithm 1 UKF evidence.

This P5 replacement is intentionally pure Python.  The old P5 gate was also a
classification phase, and P0/P2 froze no numeric promotion bands for Algorithm
1 UKF DPF statistical closeness.  Therefore P5 consumes P2-P4 artifacts,
records finite same-target LGSSM diagnostics, and blocks P44 target rows whose
same-target Algorithm 1 adapters are not yet reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
OLD_P4_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json"
P2_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_contracts_2026-06-10.json"
P3_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_values_2026-06-10.json"
P4_JSON_PATH = OUTPUT_DIR / "dpf_v2_ledh_pfpf_alg1_ukf_gradients_2026-06-10.json"

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p5_alg1_ukf_statistical_closeness_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p5-filter-oracle-statistical-closeness-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p5-filter-oracle-statistical-closeness-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_alg1_ukf_statistical_closeness_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p5-alg1-ukf-statistical-closeness-2026-06-10.md"

METHOD_ID = "ledh_pfpf_alg1_ukf_no_resampling_tf"
LOCAL_PASS_DECISION = (
    "LOCAL_PASS_P5_FILTER_ORACLE_ALG1_UKF_STATISTICAL_CLOSENESS_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
)
VETO_DECISION = "P5_FILTER_ORACLE_ALG1_UKF_STATISTICAL_CLOSENESS_VETO_PENDING_REVIEW"
EXPECTED_P5_TARGETS = (
    "lgssm_2d_h25_rich",
    "p44_m2_cubic_additive_gaussian_panel",
    "p44_m3_quadratic_observation_panel",
    "p44_m4_nonlinear_transition_h2_panel",
)
P44_TARGETS = EXPECTED_P5_TARGETS[1:]
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


class P5Alg1ValidationError(ValueError):
    """Raised when the P5 replacement artifact violates the contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        print("P5_FILTER_ORACLE_ALG1_UKF_STATISTICAL_CLOSENESS_VALIDATED")
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
    old_p4 = _load_json(OLD_P4_JSON_PATH)
    p2 = _load_json(P2_JSON_PATH)
    p3 = _load_json(P3_JSON_PATH)
    p4 = _load_json(P4_JSON_PATH)
    _preflight(old_p4, p2, p3, p4)
    rows = [_lgssm_row(p3, p4)]
    rows.extend(_p44_blocked_row(target_id, old_p4) for target_id in P44_TARGETS)
    summaries = _summaries(rows)
    veto = _veto_diagnostics(registry, old_p4, p2, p3, p4, rows)
    decision = LOCAL_PASS_DECISION if not any(bool(value) for value in veto.values()) else VETO_DECISION
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": _utc_now(),
        "phase": "P5",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_CLASSIFICATION_ONLY",
        "decision": decision,
        "question": (
            "Can the filter-oracle P5 rows that previously contained "
            "dpf_ledh_pfpf_ot be replaced by Algorithm 1 UKF evidence or "
            "reviewed target-route blockers?"
        ),
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CLASSIFICATION_ONLY_EXECUTION",
            "wrong_baseline_control": (
                "Old dpf_ledh_pfpf_ot rows define historical coverage only.  "
                "LGSSM uses exact Kalman; P44 rows use the old P4 target registry "
                "only to identify same-target reference requirements."
            ),
            "proxy_promotion_control": (
                "Finite P3/P4 rows, finite differences, ESS, and particle ladders "
                "are diagnostic-only because P0/P2 froze no numeric P5 promotion band."
            ),
            "stop_condition_control": (
                "Every P5-eligible target is classified; P44 rows are blocked "
                "rather than silently omitted or ranked on unsupported adapters."
            ),
        },
        "evidence_contract": {
            "baseline_comparator": (
                "P0/P4 filter-oracle target registry for old eligible target set; "
                "P2-P4 Algorithm 1 UKF artifacts for current evidence."
            ),
            "primary_criterion": (
                "Each old dpf_ledh_pfpf_ot P5-eligible target has a replacement "
                "status.  Algorithm 1 rows include mandatory route fields and "
                "Monte Carlo uncertainty; rows without same-target adapters are "
                "blocked with concrete adapter items."
            ),
            "promotion_policy": (
                "No P5 row is promoted because value and gradient tolerances remain "
                "N/A diagnostic-only or adapter-blocked."
            ),
            "not_concluded": _nonclaims(),
        },
        "gate_definition": {
            "local_decision_semantics": (
                "LOCAL_PASS means the artifact satisfies local classification "
                "completeness and anti-revival checks before Claude review.  It "
                "does not certify statistical closeness."
            ),
            "expected_targets": list(EXPECTED_P5_TARGETS),
            "p44_block_rule": (
                "P44 targets remain BLOCKED_REQUIRES_ADAPTER unless a reviewed "
                "same-target Algorithm 1 adapter and numeric P5 band exist before execution."
            ),
            "promotion_rule": "finite execution is diagnostic-only without numeric predeclared bands",
        },
        "consumed_artifacts": _consumed_artifacts(),
        "rows": rows,
        "route_summaries": summaries,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "old_ledh_pfpf_ot_used_as_current_evidence": False,
            "zhao_cui_used_as_p5_comparator": False,
            "value_closeness_used_to_promote_gradient": False,
            "stochastic_score_claim": "not_claimed",
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _lgssm_row(p3: dict[str, Any], p4: dict[str, Any]) -> dict[str, Any]:
    p3_summary = p3["value_summaries"]["lgssm_2d_h25_rich"][METHOD_ID]["32"]
    p4_summary = p4["gradient_summaries"]["lgssm_2d_h25_rich"]["16"]
    route_fields = next(
        row["route_fields"]
        for row in p3["value_rows"]
        if row["model_id"] == "lgssm_2d_h25_rich" and row["method_id"] == METHOD_ID
    )
    return {
        "old_coverage_route_id": "dpf_ledh_pfpf_ot",
        "target_id": "lgssm_2d_h25_rich",
        "method_id": METHOD_ID,
        "replacement_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
        "promotion_status": "not_promoted",
        "comparison_target_class": "exact_lgssm_oracle",
        "reference_route": "exact_kalman_for_lgssm",
        "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        "route_fields": route_fields,
        "value_scalar": "sum of per-step predictive log normalizers",
        "gradient_scalar": "fixed-branch sum of predictive log normalizers",
        "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P5",
        "gradient_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P5",
        "certification_band": "N/A_DIAGNOSTIC_ONLY_IN_P5",
        "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P5",
        "value_statistics": {
            "source_json": str(P3_JSON_PATH.relative_to(REPO_ROOT)),
            "particle_count": 32,
            "seed_count": int(p3_summary["seed_count"]),
            "finite_count": int(p3_summary["finite_count"]),
            "mean_value": p3_summary["mean_value"],
            "value_standard_error": p3_summary["value_standard_error"],
            "value_ci95": p3_summary["value_ci95"],
            "value_rmse_vs_reference": p3_summary["value_rmse_vs_reference"],
            "min_ess": p3_summary["min_ess"],
        },
        "gradient_statistics": {
            "source_json": str(P4_JSON_PATH.relative_to(REPO_ROOT)),
            "particle_count": 16,
            "seed_count": int(p4_summary["seed_count"]),
            "finite_count": int(p4_summary["finite_count"]),
            "mean_gradient": p4_summary["mean_gradient"],
            "gradient_component_standard_error": p4_summary["gradient_component_standard_error"],
            "mean_gradient_error_norm": p4_summary["mean_gradient_error_norm"],
            "gradient_error_norm_standard_error": p4_summary["gradient_error_norm_standard_error"],
            "gradient_error_norm_ci95": p4_summary["gradient_error_norm_ci95"],
            "gradient_reference_route": "exact_kalman_lgssm_gradient",
        },
        "downgrade_reasons": [
            "P2/P5 numeric promotion bands are N/A diagnostic-only",
            "P4 gradient seed count is diagnostic-only and below the old P5 minimum seed-count expectation",
            "finite Algorithm 1 execution is not a statistical-closeness promotion criterion",
        ],
        "blockers": [],
        "stochastic_score_claim": "not_claimed",
        "nonclaims": [
            "not a stochastic-resampling gradient claim",
            "not a production or HMC-readiness claim",
        ],
    }


def _p44_blocked_row(target_id: str, old_p4: dict[str, Any]) -> dict[str, Any]:
    old_row = next(row for row in old_p4["rows"] if row["target_id"] == target_id)
    ref = old_row["reference_context"]
    return {
        "old_coverage_route_id": "dpf_ledh_pfpf_ot",
        "target_id": target_id,
        "method_id": METHOD_ID,
        "replacement_status": "BLOCKED_REQUIRES_ADAPTER",
        "promotion_status": "not_promoted",
        "comparison_target_class": (
            "exact_or_dense_reference_available_but_no_algorithm1_same_target_adapter"
        ),
        "reference_route": ref["route_id"],
        "evidence_route_class": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "route_fields": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "value_scalar": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "gradient_scalar": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "value_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "gradient_tolerance": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "certification_band": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "value_statistics": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "gradient_statistics": "N/A_BLOCKED_REQUIRES_ADAPTER",
        "downgrade_reasons": [],
        "blockers": [
            "reviewed same-target Algorithm 1 transition_sample/transition_log_density callbacks missing for this P44 target",
            "reviewed same-target Algorithm 1 observation_mean/observation_jacobian/observation_log_density callbacks missing for this P44 target",
            "numeric P5 value and gradient promotion bands not predeclared for this Algorithm 1 target",
            "paired-seed evaluator variance and branch-decision artifacts missing for this target",
        ],
        "source_reference_artifacts": list(old_row["classification"]["source_artifacts"]),
        "stochastic_score_claim": "not_claimed",
        "nonclaims": [
            "adapter blocker is engineering status, not evidence against Algorithm 1",
            "P44 dense or Zhao-Cui evidence is not an Algorithm 1 DPF result",
        ],
    }


def _summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eligible_targets": list(EXPECTED_P5_TARGETS),
        "row_count": len(rows),
        "diagnostic_rows": [
            row["target_id"]
            for row in rows
            if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
        ],
        "blocked_rows": [
            row["target_id"]
            for row in rows
            if row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER"
        ],
        "promoted_rows": [
            row["target_id"]
            for row in rows
            if row["replacement_status"] == "RERUN_ALG1"
        ],
        "status_counts": {
            status: sum(1 for row in rows if row["replacement_status"] == status)
            for status in sorted({row["replacement_status"] for row in rows})
        },
    }


def _veto_diagnostics(
    registry: dict[str, Any],
    old_p4: dict[str, Any],
    p2: dict[str, Any],
    p3: dict[str, Any],
    p4: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, bool]:
    old_targets = tuple(old_p4["route_summaries"]["p5_dpf_eligible_targets"])
    row_targets = tuple(row["target_id"] for row in rows)
    return {
        "eligible_target_set_mismatch": old_targets != EXPECTED_P5_TARGETS,
        "row_count_or_order_mismatch": row_targets != EXPECTED_P5_TARGETS,
        "p2_contract_not_ready": p2.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW",
        "p3_values_not_ready": not str(p3.get("decision", "")).startswith("LOCAL_PASS_P3_"),
        "p4_gradients_not_ready": not str(p4.get("decision", "")).startswith("LOCAL_PASS_P4_"),
        "old_ledh_pfpf_ot_used_as_current_method": any(
            row["method_id"] == "dpf_ledh_pfpf_ot" for row in rows
        ),
        "algorithm1_route_fields_missing": any(
            row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            and not _has_algorithm1_route_fields(row["route_fields"])
            for row in rows
        ),
        "p44_row_promoted_without_same_target_adapter": any(
            row["target_id"].startswith("p44_") and row["replacement_status"] != "BLOCKED_REQUIRES_ADAPTER"
            for row in rows
        ),
        "finite_only_promoted": any(row["replacement_status"] == "RERUN_ALG1" for row in rows),
        "missing_monte_carlo_uncertainty_on_diagnostic_lgssm": any(
            row["target_id"] == "lgssm_2d_h25_rich"
            and (
                not row["value_statistics"].get("value_standard_error")
                or not row["gradient_statistics"].get("gradient_error_norm_standard_error")
            )
            for row in rows
        ),
        "threshold_or_band_missing_without_na_reason": any(
            row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            and (
                not str(row["value_tolerance"]).startswith("N/A_")
                or not str(row["gradient_tolerance"]).startswith("N/A_")
                or not str(row["certification_band"]).startswith("N/A_")
            )
            for row in rows
        ),
        "unsupported_comparator_promoted": False,
        "value_used_to_promote_gradient": False,
        "stochastic_score_claimed": any(row["stochastic_score_claim"] != "not_claimed" for row in rows),
        "zhao_cui_used_as_p5_comparator": False,
        "registry_missing_expected_target": any(
            target_id not in registry["route_matrix"] for target_id in EXPECTED_P5_TARGETS
        ),
    }


def _has_algorithm1_route_fields(route_fields: Any) -> bool:
    return isinstance(route_fields, dict) and all(route_fields.get(key) for key in ALG1_ROUTE_REQUIRED)


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "execution_mode",
        "evidence_contract",
        "gate_definition",
        "rows",
        "route_summaries",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P5Alg1ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P5Alg1ValidationError(f"P5 decision is not local pass: {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_CLASSIFICATION_ONLY":
        raise P5Alg1ValidationError("P5 should be pure Python classification only")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P5Alg1ValidationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    rows = payload["rows"]
    if tuple(row["target_id"] for row in rows) != EXPECTED_P5_TARGETS:
        raise P5Alg1ValidationError("P5 target order drifted")
    if payload["route_summaries"]["promoted_rows"]:
        raise P5Alg1ValidationError("P5 promoted a row")
    if payload["route_summaries"]["diagnostic_rows"] != ["lgssm_2d_h25_rich"]:
        raise P5Alg1ValidationError("P5 should have exactly one diagnostic LGSSM row")
    if set(payload["route_summaries"]["blocked_rows"]) != set(P44_TARGETS):
        raise P5Alg1ValidationError("P5 should block exactly the P44 rows")
    for row in rows:
        if row["method_id"] != METHOD_ID:
            raise P5Alg1ValidationError("old or unexpected method id appeared")
        if row["replacement_status"] == "RERUN_ALG1":
            raise P5Alg1ValidationError("P5 statistical closeness was promoted")
        if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY":
            if not _has_algorithm1_route_fields(row["route_fields"]):
                raise P5Alg1ValidationError("diagnostic row missing Algorithm 1 route fields")
            if int(row["value_statistics"]["seed_count"]) < 5:
                raise P5Alg1ValidationError("LGSSM value seed count below diagnostic expectation")
            if int(row["gradient_statistics"]["seed_count"]) < 3:
                raise P5Alg1ValidationError("LGSSM gradient seed count below P4 diagnostic expectation")
            if not row["downgrade_reasons"]:
                raise P5Alg1ValidationError("diagnostic row lacks downgrade reasons")
        if row["replacement_status"] == "BLOCKED_REQUIRES_ADAPTER" and not row["blockers"]:
            raise P5Alg1ValidationError("blocked row lacks adapter blockers")
    if payload["run_manifest"]["tensorflow_imported"]:
        raise P5Alg1ValidationError("P5 classification imported TensorFlow")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P5 Result: Filter-Oracle Algorithm 1 UKF Statistical Closeness Replacement",
        "",
        "metadata_date: 2026-06-10",
        "phase: P5",
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
        "## Rows",
        "",
        "| Target | Method | Status | Reference | Main reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["rows"]:
        reason = (
            "; ".join(row["downgrade_reasons"])
            if row["replacement_status"] == "RERUN_ALG1_DIAGNOSTIC_ONLY"
            else row["blockers"][0]
        )
        lines.append(
            f"| `{row['target_id']}` | `{row['method_id']}` | "
            f"`{row['replacement_status']}` | `{row['reference_route']}` | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Eligible targets: `{summary['eligible_targets']}`.",
            f"- Diagnostic rows: `{summary['diagnostic_rows']}`.",
            f"- Blocked rows: `{summary['blocked_rows']}`.",
            f"- Promoted rows: `{summary['promoted_rows']}`.",
            "",
            "## LGSSM Diagnostic Statistics",
            "",
            "| Particle | Value seeds | Value SE | Value RMSE | Gradient seeds | Mean grad error norm | Grad error SE |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lgssm = payload["rows"][0]
    lines.append(
        "| {value_particle} | {value_seeds} | {value_se} | {value_rmse} | "
        "{grad_seeds} | {grad_norm} | {grad_se} |".format(
            value_particle=lgssm["value_statistics"]["particle_count"],
            value_seeds=lgssm["value_statistics"]["seed_count"],
            value_se=lgssm["value_statistics"]["value_standard_error"],
            value_rmse=lgssm["value_statistics"]["value_rmse_vs_reference"],
            grad_seeds=lgssm["gradient_statistics"]["seed_count"],
            grad_norm=lgssm["gradient_statistics"]["mean_gradient_error_norm"],
            grad_se=lgssm["gradient_statistics"]["gradient_error_norm_standard_error"],
        )
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
            "## Gate Definition",
            "",
            f"- Local decision semantics: {payload['gate_definition']['local_decision_semantics']}",
            f"- P44 block rule: {payload['gate_definition']['p44_block_rule']}",
            f"- Promotion rule: {payload['gate_definition']['promotion_rule']}",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | every old P5 dpf_ledh_pfpf_ot eligible target is "
                "classified; LGSSM is diagnostic; P44 rows are adapter-blocked | "
                f"`{payload['veto_diagnostics']}` | no numeric P5 promotion bands and no P44 same-target Algorithm 1 adapters | "
                "Claude P5 review, then P6 calibration classification | no statistical-closeness certification, stochastic-score correctness, HMC, production, or GPU claim |"
            ),
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
        "seed_list": "consumed from P3/P4 only",
        "particle_counts": "consumed from P3/P4 only",
        "data_version": "P2-P4 Algorithm 1 artifacts plus historical filter-oracle target registry",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _consumed_artifacts() -> list[dict[str, str]]:
    return [
        {"role": "historical_filter_oracle_registry", "path": str(REGISTRY_PATH.relative_to(REPO_ROOT))},
        {"role": "historical_p5_target_set", "path": str(OLD_P4_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_contracts", "path": str(P2_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_values", "path": str(P3_JSON_PATH.relative_to(REPO_ROOT))},
        {"role": "algorithm1_gradients", "path": str(P4_JSON_PATH.relative_to(REPO_ROOT))},
    ]


def _preflight(old_p4: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], p4: dict[str, Any]) -> None:
    if tuple(old_p4["route_summaries"]["p5_dpf_eligible_targets"]) != EXPECTED_P5_TARGETS:
        raise P5Alg1ValidationError("old P5 eligible target set drifted")
    if p2.get("decision") != "PASS_P2_V2_ALG1_UKF_CONTRACTS_PENDING_CLAUDE_REVIEW":
        raise P5Alg1ValidationError("P2 contract artifact is not ready")
    if not str(p3.get("decision", "")).startswith("LOCAL_PASS_P3_"):
        raise P5Alg1ValidationError("P3 value artifact is not ready")
    if not str(p4.get("decision", "")).startswith("LOCAL_PASS_P4_"):
        raise P5Alg1ValidationError("P4 gradient artifact is not ready")


def _nonclaims() -> list[str]:
    return [
        "P5 does not certify Algorithm 1 statistical closeness.",
        "P5 does not revive old dpf_ledh_pfpf_ot results as evidence.",
        "P5 does not establish nonlinear P44 DPF value or gradient closeness.",
        "P5 does not use Zhao-Cui, CUT4, SVD, UKF, or FilterFlow as a DPF correctness oracle.",
        "P5 does not establish stochastic-resampling gradient correctness.",
        "P5 does not establish HMC readiness, production readiness, GPU readiness, or paper-scale claims.",
    ]


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
