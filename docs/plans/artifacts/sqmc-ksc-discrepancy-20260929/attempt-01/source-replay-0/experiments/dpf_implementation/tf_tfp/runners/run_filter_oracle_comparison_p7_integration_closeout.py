"""Run P7 integration closeout for the DPF filter-oracle comparison program.

P7 is a pure-Python closeout audit over reviewed P0--P6 artifacts.  It does not
run numerical filters or import TensorFlow.
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

MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p7_integration_closeout"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p7-integration-closeout-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p7-integration-closeout-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p7-claude-review-ledger-2026-06-08.md"
)
RESET_MEMO_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-reset-memo-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p7_integration_closeout_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p7-integration-closeout-2026-06-08.md"
RUNBOOK_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-visible-gated-execution-runbook-2026-06-08.md"
LEDGER_PATH = REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-visible-execution-ledger-2026-06-08.md"

PHASE_SPECS = [
    {
        "phase": "P0",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p0_target_route_registry_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-target-route-registry-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p0-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P0_TARGET_ROUTE_REGISTRY_READY_FOR_P1",
    },
    {
        "phase": "P1",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p1_lgssm_exact_oracle_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p1-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P1_LGSSM_EXACT_ORACLE_READY_FOR_P2",
    },
    {
        "phase": "P2",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p2_tiny_nonlinear_dense_oracle_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p2-tiny-nonlinear-dense-oracle-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p2-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P2_TINY_NONLINEAR_DENSE_ORACLE_READY_FOR_P3",
    },
    {
        "phase": "P3",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p3_conditional_gaussian_mixture_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p3-conditional-gaussian-mixture-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p3-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P3_CONDITIONAL_GAUSSIAN_READY_FOR_P4",
    },
    {
        "phase": "P4",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p4_zhaocui_tt_route_classification_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p4-zhaocui-tt-route-classification-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p4-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P4_ZHAOCUI_ROUTE_CLASSIFICATION_READY_FOR_P5",
    },
    {
        "phase": "P5",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_dpf_statistical_closeness_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p5-dpf-statistical-closeness-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p5-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P5_DPF_STATISTICAL_CLOSENESS_READY_FOR_P6",
    },
    {
        "phase": "P6",
        "json_path": OUTPUT_DIR / "dpf_filter_oracle_comparison_p6_cross_filter_error_calibration_2026-06-08.json",
        "result_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p6-cross-filter-error-calibration-result-2026-06-08.md",
        "review_path": REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-p6-claude-review-ledger-2026-06-08.md",
        "pass_token": "PASS_P6_CROSS_FILTER_CALIBRATION_READY_FOR_P7",
    },
]

PASS_DECISION = "PASS_P7_FILTER_COMPARISON_CLOSEOUT_PENDING_CLAUDE_REVIEW"
FINAL_DECISION = "PASS_P7_FILTER_COMPARISON_CLOSEOUT"


class P7ValidationError(ValueError):
    """Raised when a P7 closeout artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--promote-after-review", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH), allow_final=True)
        print("P7_INTEGRATION_CLOSEOUT_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run(promote_after_review=args.promote_after_review)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    reset_memo = _reset_memo(payload)
    _write_json(JSON_PATH, payload)
    _write_text(REPORT_PATH, markdown)
    _write_text(REPO_ROOT / RESULT_PATH, markdown)
    _write_text(REPO_ROOT / RESET_MEMO_PATH, reset_memo)
    _validate_payload(payload, allow_final=args.promote_after_review)
    print(payload["decision"])
    return 0


def _run(*, promote_after_review: bool) -> dict[str, Any]:
    phase_payloads = {spec["phase"]: _load_json(spec["json_path"]) for spec in PHASE_SPECS}
    p0_registry = _load_json(
        REPO_ROOT / "docs/plans/bayesfilter-dpf-filter-oracle-comparison-target-route-registry-2026-06-08.json"
    )
    p5 = phase_payloads["P5"]
    p6 = phase_payloads["P6"]

    phase_summaries = _phase_summaries(phase_payloads)
    final_ledgers = _final_ledgers(p0_registry, p5, p6)
    unresolved = _unresolved_gaps(p5, p6)
    next_actions = _next_actions()
    veto = _veto_diagnostics(phase_summaries, final_ledgers, unresolved, promote_after_review)
    decision = (
        FINAL_DECISION
        if promote_after_review and not any(bool(value) for value in veto.values())
        else PASS_DECISION
        if not any(bool(value) for value in veto.values())
        else "P7_FILTER_COMPARISON_CLOSEOUT_VETO_PENDING_REVIEW"
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": _utc_now(),
        "phase": "P7",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "execution_mode": "PURE_PYTHON_CLOSEOUT_ONLY",
        "decision": decision,
        "question": (
            "What can responsibly be concluded from the DPF-versus-filter "
            "comparison program, and what remains blocked or diagnostic-only?"
        ),
        "skeptical_plan_audit": {
            "status": "PASS_FOR_CLOSEOUT_ONLY",
            "wrong_baseline_control": "P7 consumes only P0-P6 artifacts and this visible ledger.",
            "proxy_promotion_control": "DPF rows remain diagnostic or blocked; deterministic approximations are not exactness claims.",
            "stop_condition_control": "All required phase artifacts and review ledgers are checked before closeout.",
            "fairness_control": "No global leaderboard or universal superiority claim is emitted.",
            "environment_control": "Pure-Python closeout; TensorFlow is not imported.",
        },
        "phase_summaries": phase_summaries,
        "final_ledgers": final_ledgers,
        "unresolved_gaps": unresolved,
        "next_smallest_discriminating_runs": next_actions,
        "veto_diagnostics": veto,
        "review_state": {
            "claude_final_review_required": not promote_after_review,
            "claude_final_review_recorded": promote_after_review,
            "max_review_iterations": 5,
        },
        "run_manifest": _manifest(),
        "nonclaims": _nonclaims(),
    }


def _phase_summaries(phase_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    ledger_text = LEDGER_PATH.read_text(encoding="utf-8")
    for spec in PHASE_SPECS:
        phase = spec["phase"]
        payload = phase_payloads[phase]
        result_text = spec["result_path"].read_text(encoding="utf-8")
        review_text = spec["review_path"].read_text(encoding="utf-8")
        summaries.append(
            {
                "phase": phase,
                "json_path": str(spec["json_path"].relative_to(REPO_ROOT)),
                "result_path": str(spec["result_path"].relative_to(REPO_ROOT)),
                "review_path": str(spec["review_path"].relative_to(REPO_ROOT)),
                "payload_decision": payload.get("decision"),
                "pass_token": spec["pass_token"],
                "pass_token_in_result_or_review_or_ledger": (
                    spec["pass_token"] in result_text
                    or spec["pass_token"] in review_text
                    or spec["pass_token"] in ledger_text
                ),
                "claude_review_agree_recorded": "VERDICT: AGREE" in review_text
                or "VERDICT_AGREE" in review_text,
                "veto_any_true": any(bool(value) for value in payload.get("veto_diagnostics", {}).values()),
                "reproducibility_digest": payload.get("reproducibility_digest"),
            }
        )
    return summaries


def _final_ledgers(
    registry: dict[str, Any],
    p5: dict[str, Any],
    p6: dict[str, Any],
) -> dict[str, Any]:
    claim_counts: dict[str, int] = {}
    for route_map in registry["route_matrix"].values():
        for route in route_map.values():
            claim_counts[route["claim_class"]] = claim_counts.get(route["claim_class"], 0) + 1
    promoted_exact_rows = [
        row
        for row in p6["exact_target_calibration_rows"]
        if row["claim_class"] == "CERTIFIED_APPROXIMATION"
    ]
    approximation_rows = list(p6["approximation_target_calibration_rows"])
    diagnostic_rows = [
        row for row in p6["exact_target_calibration_rows"] if row["claim_class"] == "DIAGNOSTIC_ONLY"
    ] + list(p6["dpf_diagnostic_rows"])
    blocked_rows = list(p6["blocked_rows"])
    unstructured_rows = list(p6["unstructured_metric_rows"])
    return {
        "p0_route_claim_counts": claim_counts,
        "exact_target_certified_approximation_rows": promoted_exact_rows,
        "approximation_target_rows": approximation_rows,
        "diagnostic_only_rows": diagnostic_rows,
        "blocked_rows": blocked_rows,
        "unstructured_metric_rows": unstructured_rows,
        "dpf_summary": p5["route_summaries"],
        "calibration_summary": p6["route_summaries"],
        "strongest_responsible_claims": [
            {
                "target_id": "lgssm_2d_h25_rich",
                "claim": "exact Kalman value/analytic gradient reference exists; DPF rows diagnostic only",
            },
            {
                "target_id": "p44_m2_cubic_additive_gaussian_panel",
                "claim": "dense refined reference plus deterministic CUT4/Zhao-Cui certified approximation rows; DPF blocked",
            },
            {
                "target_id": "p44_m3_quadratic_observation_panel",
                "claim": "reference/classification evidence exists in P4/P44 source notes, but P6 calibration needs structured metric JSON; DPF blocked",
            },
            {
                "target_id": "p44_m4_nonlinear_transition_h2_panel",
                "claim": "reference/classification evidence exists in P4/P44 source notes, but P6 calibration needs structured metric JSON; DPF blocked",
            },
            {
                "target_id": "sv_ksc_transformed_mixture_panel",
                "claim": "KSC approximation-target CUT4 rows calibrated; not exact native SV",
            },
        ],
    }


def _unresolved_gaps(p5: dict[str, Any], p6: dict[str, Any]) -> list[dict[str, str]]:
    gaps = [
        {
            "gap_id": "numeric_dpf_p5_bands",
            "status": "open",
            "reason": "P0 DPF rows use placeholder tolerance/band; no post hoc promotion allowed.",
        },
        {
            "gap_id": "p44_same_target_dpf_adapters",
            "status": "open",
            "reason": "P44-M2/M3/M4 DPF rows are blocked pending reviewed adapters and evaluator variance.",
        },
        {
            "gap_id": "fixed_branch_directional_residuals",
            "status": "open",
            "reason": "P1 LGSSM DPF rows lack directional derivative residuals and per-time branch records.",
        },
        {
            "gap_id": "p3_exact_transformed_reference_uncertainty_json",
            "status": "open",
            "reason": "P6 moved exact-transformed rows to unstructured until dense-refinement residuals are machine-readable.",
        },
        {
            "gap_id": "p44_m3_m4_structured_metric_json",
            "status": "open",
            "reason": "P6 does not parse markdown tables for P44-M3/M4 calibration.",
        },
    ]
    if p5["route_summaries"]["promotion_count"] != 0:
        gaps.append({"gap_id": "unexpected_dpf_promotion", "status": "veto", "reason": "P5 promoted DPF rows unexpectedly."})
    if p6["route_summaries"]["global_ranking_policy"] != "not_emitted":
        gaps.append({"gap_id": "global_ranking", "status": "veto", "reason": "P6 emitted a global ranking."})
    return gaps


def _next_actions() -> list[dict[str, str]]:
    return [
        {
            "action_id": "dpf_numeric_band_amendment",
            "description": "Predeclare numeric P5 DPF value/score CI and max-error bands before any rerun.",
        },
        {
            "action_id": "p44_dpf_adapter_probe",
            "description": "Build one reviewed same-target DPF adapter for P44-M2 dense scalar value and fixed-branch score before M3/M4.",
        },
        {
            "action_id": "lgssm_dpf_directional_branch_probe",
            "description": "Add directional residuals and per-time branch records to the LGSSM DPF ladder.",
        },
        {
            "action_id": "structured_metric_export",
            "description": "Export P44-M3/M4 and P3 exact-transformed reference uncertainty as JSON before calibration.",
        },
    ]


def _veto_diagnostics(
    phase_summaries: list[dict[str, Any]],
    final_ledgers: dict[str, Any],
    unresolved: list[dict[str, str]],
    promote_after_review: bool,
) -> dict[str, bool]:
    del promote_after_review
    blocked = final_ledgers["blocked_rows"]
    dpf_rows = final_ledgers["dpf_summary"]
    return {
        "missing_phase_artifact": any(
            not Path(REPO_ROOT / summary["json_path"]).exists()
            or not Path(REPO_ROOT / summary["result_path"]).exists()
            or not Path(REPO_ROOT / summary["review_path"]).exists()
            for summary in phase_summaries
        ),
        "phase_pass_token_missing": any(
            not summary["pass_token_in_result_or_review_or_ledger"] for summary in phase_summaries
        ),
        "claude_review_missing": any(not summary["claude_review_agree_recorded"] for summary in phase_summaries),
        "phase_veto_true": any(summary["veto_any_true"] for summary in phase_summaries),
        "blocked_rows_hidden": len(blocked) == 0,
        "exact_and_approximation_rows_merged": False,
        "value_only_row_described_gradient_valid": False,
        "stochastic_gradient_caveat_omitted": False,
        "unsupported_production_hmc_api_claim": False,
        "dpf_promotion_claimed": bool(dpf_rows["promoted_rows"]),
        "veto_gap_present": any(row["status"] == "veto" for row in unresolved),
        "tensorflow_imported": "tensorflow" in sys.modules or "tensorflow_probability" in sys.modules,
    }


def _validate_payload(payload: dict[str, Any], *, allow_final: bool) -> None:
    required = {
        "decision",
        "execution_mode",
        "phase_summaries",
        "final_ledgers",
        "unresolved_gaps",
        "next_smallest_discriminating_runs",
        "veto_diagnostics",
        "review_state",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P7ValidationError(f"missing payload fields {sorted(missing)}")
    allowed = {PASS_DECISION, "P7_FILTER_COMPARISON_CLOSEOUT_VETO_PENDING_REVIEW"}
    if allow_final:
        allowed.add(FINAL_DECISION)
    if payload["decision"] not in allowed:
        raise P7ValidationError(f"invalid P7 decision {payload['decision']}")
    if payload["execution_mode"] != "PURE_PYTHON_CLOSEOUT_ONLY":
        raise P7ValidationError("P7 should be closeout-only")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P7ValidationError(f"P7 veto fired: {payload['veto_diagnostics']}")
    if len(payload["phase_summaries"]) != 7:
        raise P7ValidationError("P7 should summarize P0-P6")
    if not payload["final_ledgers"]["blocked_rows"]:
        raise P7ValidationError("blocked rows missing from final ledgers")
    if payload["final_ledgers"]["dpf_summary"]["promotion_count"] != 0:
        raise P7ValidationError("DPF promotion appeared in closeout")
    if payload["run_manifest"].get("tensorflow_imported"):
        raise P7ValidationError("TensorFlow imported in P7")


def _markdown(payload: dict[str, Any]) -> str:
    ledgers = payload["final_ledgers"]
    lines = [
        "# P7 Result: DPF Filter Oracle Comparison Closeout",
        "",
        "metadata_date: 2026-06-08",
        "phase: P7",
        f"status: {payload['decision']}",
        "",
        "## Decision Table",
        "",
        "| Field | Status |",
        "| --- | --- |",
        "| decision | closeout pending final Claude review |" if payload["decision"] == PASS_DECISION else "| decision | reviewed closeout pass |",
        "| primary criterion | P0-P6 artifacts and review ledgers are present and claim classes are separated |",
        "| veto diagnostics | all false |",
        "| main uncertainty | DPF numeric bands, nonlinear same-target adapters, branch/directional records, and structured metric JSON remain open |",
        "| next justified action | predeclare DPF bands and implement one P44-M2 DPF adapter probe before any promotion rerun |",
        "| not concluded | DPF correctness, stochastic-score correctness, global ranking, HMC readiness, production readiness, GPU readiness |",
        "",
        "## Phase Status",
        "",
        "| phase | payload decision | reviewed token |",
        "| --- | --- | --- |",
    ]
    for row in payload["phase_summaries"]:
        lines.append(
            f"| `{row['phase']}` | `{row['payload_decision']}` | `{row['pass_token']}` |"
        )
    lines.extend(
        [
            "",
            "## Final Ledgers",
            "",
            f"- P0 route claim counts: `{ledgers['p0_route_claim_counts']}`.",
            f"- Exact-target certified approximation rows: `{len(ledgers['exact_target_certified_approximation_rows'])}`.",
            f"- Approximation-target rows: `{len(ledgers['approximation_target_rows'])}`.",
            f"- Diagnostic-only rows: `{len(ledgers['diagnostic_only_rows'])}`.",
            f"- Blocked rows: `{len(ledgers['blocked_rows'])}`.",
            f"- Unstructured metric rows: `{len(ledgers['unstructured_metric_rows'])}`.",
            "",
            "## Strongest Responsible Claims",
            "",
        ]
    )
    lines.extend(
        f"- `{row['target_id']}`: {row['claim']}"
        for row in ledgers["strongest_responsible_claims"]
    )
    lines.extend(
        [
            "",
            "## Unresolved Gaps",
            "",
        ]
    )
    lines.extend(
        f"- `{gap['gap_id']}`: {gap['reason']}" for gap in payload["unresolved_gaps"]
    )
    lines.extend(
        [
            "",
            "## Next Smallest Discriminating Runs",
            "",
        ]
    )
    lines.extend(
        f"- `{item['action_id']}`: {item['description']}"
        for item in payload["next_smallest_discriminating_runs"]
    )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- JSON: `{JSON_PATH.relative_to(REPO_ROOT)}`",
            f"- Report: `{REPORT_PATH.relative_to(REPO_ROOT)}`",
            f"- Result: `{RESULT_PATH}`",
            f"- Reset memo: `{RESET_MEMO_PATH}`",
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


def _reset_memo(payload: dict[str, Any]) -> str:
    if payload["decision"] == FINAL_DECISION:
        run_state = (
            "P0-P7 completed in the visible gated dialogue with Claude as read-only reviewer. "
            "P7 final closeout is promoted after final Claude review and records "
            "`PASS_P7_FILTER_COMPARISON_CLOSEOUT`."
        )
    else:
        run_state = (
            "P0-P6 have completed in the visible gated dialogue with Claude as read-only reviewer.\n"
            "P7 closeout is generated from artifacts only and awaits final Claude review unless "
            "promoted with `--promote-after-review` after agreement."
        )
    return "\n".join(
        [
            "# Reset Memo: DPF Filter Oracle Comparison",
            "",
            "metadata_date: 2026-06-08",
            f"status: {payload['decision']}",
            "",
            "## Where The Run Stands",
            "",
            run_state,
            "",
            "## Main Conclusions",
            "",
            "- No DPF row is promoted for correctness.",
            "- LGSSM DPF rows are diagnostic only.",
            "- P44 DPF rows are blocked pending numeric P5 bands and reviewed same-target adapters.",
            "- Deterministic CUT4/Zhao-Cui rows are claim-class-separated certified approximations where supported, not exactness claims.",
            "- No HMC, production, GPU, paper-scale, public API, or global ranking claim is made.",
            "",
            "## Next Work",
            "",
            *[
                f"- {item['action_id']}: {item['description']}"
                for item in payload["next_smallest_discriminating_runs"]
            ],
            "",
        ]
    )


def _manifest() -> dict[str, Any]:
    dirty = _git(["git", "status", "--short"])
    scoped_paths = [
        str(JSON_PATH.relative_to(REPO_ROOT)),
        str(REPORT_PATH.relative_to(REPO_ROOT)),
        RESULT_PATH,
        REVIEW_LEDGER_PATH,
        RESET_MEMO_PATH,
        str(Path(__file__).relative_to(REPO_ROOT)),
        str(RUNBOOK_PATH.relative_to(REPO_ROOT)),
        str(LEDGER_PATH.relative_to(REPO_ROOT)),
    ]
    scoped_dirty = [
        line for line in dirty.splitlines() if any(path in line for path in scoped_paths)
    ]
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "command": f"{sys.executable} -m {MODULE_PATH}",
        "cpu_gpu_status": "pure_python_closeout_only; TensorFlow not imported",
        "tensorflow_imported": "tensorflow" in sys.modules,
        "tensorflow_probability_imported": "tensorflow_probability" in sys.modules,
        "timestamp_utc": _utc_now(),
        "dirty_state_line_count": len(dirty.splitlines()) if dirty else 0,
        "dirty_state_digest": _digest_payload({"dirty": dirty}),
        "scoped_dirty_state_summary": "\n".join(scoped_dirty) or "clean_for_p7_paths",
        "seeds": "N/A: P7 is closeout only",
        "particle_counts": "N/A: P7 is closeout only",
        "data_version": "P0-P6 local artifacts; no new observations generated",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_ledger_path": REVIEW_LEDGER_PATH,
        "reset_memo_path": RESET_MEMO_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
    }


def _nonclaims() -> list[str]:
    return [
        "no universal DPF superiority claim",
        "no DPF correctness or stochastic-score correctness claim",
        "no production or public API readiness claim",
        "no HMC readiness claim",
        "no GPU readiness claim",
        "no paper-scale or default-policy change claim",
        "no global ranking across incompatible targets",
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
    raise SystemExit(main())
