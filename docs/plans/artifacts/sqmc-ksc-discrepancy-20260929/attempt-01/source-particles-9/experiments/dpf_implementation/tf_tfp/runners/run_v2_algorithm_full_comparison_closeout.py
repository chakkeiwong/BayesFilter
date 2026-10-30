"""Generate the visible P8 closeout for the DPF V2 full comparison.

This closeout intentionally avoids importing TensorFlow.  P8 is an artifact
consistency and governance gate over reviewed P0--P7 outputs, not a numerical
execution phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
REPORT_DIR = REPO_ROOT / "experiments" / "dpf_implementation" / "reports"
OUTPUT_DIR = REPORT_DIR / "outputs"

PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p8-closeout-subplan-2026-06-07.md"
)
RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "closeout-result-2026-06-07.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_closeout_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-v2-algorithm-full-comparison-closeout-2026-06-07.md"
RUNBOOK_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "visible-gated-execution-runbook-2026-06-08.md"
)
LEDGER_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "visible-execution-ledger-2026-06-08.md"
)
STOP_HANDOFF_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "visible-stop-handoff-2026-06-08.md"
)
P8_REVIEW_LEDGER_PATH = (
    "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-"
    "p8-claude-review-ledger-2026-06-08.md"
)

PASS_DECISION = "PASS_FULL_COMPARISON"
BLOCKED_DECISION = "BLOCKED_WITH_REVIEWED_CLASSIFICATION"
LOCAL_PENDING_DECISION = "LOCAL_PASS_P8_CLAUDE_REVIEW_PENDING"
EXPECTED_P2_BUNDLE_CHECKSUM = (
    "53722eaea627646a85b99a2dde94d733b31e35e3a10886b05df20a733a1df11c"
)
EXPECTED_P5_BUNDLE_CHECKSUM = (
    "20b7cb9a05a2de41bbf139a90bff8c8465f7bda7e4fd84fe6b83ee48a09c39f4"
)
EXPECTED_P3_DIGEST = "3cf2161ff3ff470240f3a8c60f2405f6aff919769a013e68f56d19808905d521"
EXPECTED_P4_DIGEST = "f5460402677d25c551d4557c430b7b41a5bda58abf48beb070f9cf423a3725c2"
EXPECTED_P5_DIGEST = "6770ce953db079fe003d62e631e8a379cf68e6889006d583b39d70e6f5139661"
EXPECTED_P6_DIGEST = "890a07f12bd2fcc35e6bc747578455bc04c418948058b07d3f5d2f62b955fe24"
EXPECTED_P7_DIGEST = "d9d6f691c00171e287971b89b461b151ee2dd8d8e1c1804c45421bfa7dc94f14"

REQUIRED_V2_MODEL_IDS = [
    "lgssm_2d_h25_rich",
    "sv_1d_h18_rich",
    "range_bearing_4d_h20_rich",
    "structural_ar1_quadratic_h16",
    "spatial_sir_j3_rk4",
    "predator_prey_rk4",
]

PHASES = [
    {
        "phase": "P0",
        "name": "Governance",
        "json_path": OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p0_visible_governance_2026-06-08.json",
        "report_path": None,
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p0-governance-visible-result-2026-06-08.md",
        "expected_decision": "PASS_P0_READY_FOR_P1",
        "artifact_kind": "document_governance",
    },
    {
        "phase": "P1",
        "name": "Architecture",
        "json_path": OUTPUT_DIR / "dpf_v2_algorithm_full_comparison_p1_architecture_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-algorithm-full-comparison-p1-architecture-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p1-architecture-result-2026-06-07.md",
        "expected_decision": "PASS_P1_ARCHITECTURE_READY_FOR_P2",
        "artifact_kind": "document_governance",
    },
    {
        "phase": "P2",
        "name": "Bootstrap-OT contracts",
        "json_path": OUTPUT_DIR / "dpf_v2_bootstrap_ot_contracts_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-bootstrap-ot-contracts-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p2-bootstrap-ot-contracts-result-2026-06-07.md",
        "expected_decision": "PASS_P2_BOOTSTRAP_OT_CONTRACTS_READY_FOR_P3",
        "artifact_kind": "runner",
    },
    {
        "phase": "P3",
        "name": "Bootstrap-OT values",
        "json_path": OUTPUT_DIR / "dpf_v2_bootstrap_ot_values_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-bootstrap-ot-values-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p3-bootstrap-ot-values-result-2026-06-07.md",
        "expected_decision": "PASS_P3_BOOTSTRAP_OT_VALUES_READY_FOR_P4",
        "artifact_kind": "runner",
    },
    {
        "phase": "P4",
        "name": "Bootstrap-OT gradients",
        "json_path": OUTPUT_DIR / "dpf_v2_bootstrap_ot_gradients_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-bootstrap-ot-gradients-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p4-bootstrap-ot-gradients-result-2026-06-07.md",
        "expected_decision": "PASS_P4_BOOTSTRAP_OT_GRADIENTS_READY_FOR_P5",
        "artifact_kind": "runner",
    },
    {
        "phase": "P5",
        "name": "LEDH-PFPF-OT contracts",
        "json_path": OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_contracts_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-ledh-pfpf-ot-contracts-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p5-ledh-pfpf-ot-contracts-result-2026-06-07.md",
        "expected_decision": "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6",
        "artifact_kind": "runner",
    },
    {
        "phase": "P6",
        "name": "LEDH-PFPF-OT values",
        "json_path": OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_values_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-ledh-pfpf-ot-values-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p6-ledh-pfpf-ot-values-result-2026-06-07.md",
        "expected_decision": "PASS_P6_LEDH_PFPF_OT_VALUES_READY_FOR_P7",
        "artifact_kind": "runner",
    },
    {
        "phase": "P7",
        "name": "LEDH-PFPF-OT gradients",
        "json_path": OUTPUT_DIR / "dpf_v2_ledh_pfpf_ot_gradients_2026-06-07.json",
        "report_path": REPORT_DIR / "dpf-v2-ledh-pfpf-ot-gradients-2026-06-07.md",
        "result_path": REPO_ROOT
        / "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p7-ledh-pfpf-ot-gradients-result-2026-06-07.md",
        "expected_decision": "PASS_P7_LEDH_PFPF_OT_GRADIENTS_READY_FOR_P8",
        "artifact_kind": "runner",
    },
]

NON_CLAIMS = [
    "P8 closes same-contract BF/FilterFlow-side adapter agreement only.",
    "P8 does not prove BayesFilter correctness.",
    "P8 does not prove FilterFlow correctness.",
    "P8 does not prove bootstrap-OT or LEDH-PFPF-OT scientific correctness.",
    "P8 does not establish stochastic resampling distribution correctness.",
    "P8 does not establish gradients through random or discrete branch decisions.",
    "P8 does not make a student implementation claim.",
    "P8 does not make a TT/SIRT, dense quadrature, paper-table, simulated-truth, GPU, HMC, DSGE, scalability, deployment, or production-readiness claim.",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--promote-after-review",
        action="store_true",
        help="Write PASS_FULL_COMPARISON after external Claude closeout agreement.",
    )
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(_load_json(JSON_PATH))
        return 0

    start = time.perf_counter()
    payload = _run(promote_after_review=args.promote_after_review)
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    initial_markdown = _markdown(payload)
    payload["veto_diagnostics"]["unsupported_claim_terms_in_closeout_artifact"] = (
        _unsupported_claim_terms([initial_markdown, json.dumps(payload, sort_keys=True, default=str)])
    )
    _refresh_decision_state(payload, promote_after_review=args.promote_after_review)
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    _write_json(JSON_PATH, payload)
    _write_text(REPORT_PATH, markdown)
    _write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run(*, promote_after_review: bool) -> dict[str, Any]:
    phase_payloads = {spec["phase"]: _load_json(spec["json_path"]) for spec in PHASES}
    phase_summaries = _phase_summaries(phase_payloads)
    algorithm_tables = _algorithm_tables(phase_payloads)
    row_table = _row_table(phase_payloads)
    review_evidence = _review_evidence()
    p8_review_evidence = _p8_final_review_evidence()
    command_manifest_evidence = _command_manifest_evidence(phase_payloads)
    lineage = _lineage_evidence(phase_payloads)
    veto_diagnostics = _veto_diagnostics(
        phase_payloads,
        phase_summaries,
        algorithm_tables,
        row_table,
        review_evidence,
        p8_review_evidence,
        command_manifest_evidence,
        lineage,
        promote_after_review=promote_after_review,
    )
    open_blockers = [
        name for name, value in veto_diagnostics.items() if _veto_fired(value)
    ]
    local_pass = not open_blockers
    decision = LOCAL_PENDING_DECISION if local_pass else BLOCKED_DECISION
    if promote_after_review and local_pass:
        decision = PASS_DECISION
    return {
        "metadata_date": "2026-06-07",
        "created_at_utc": _utc_now(),
        "phase": "P8",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "After reviewed visible P0--P7 gates, what can responsibly be said "
            "about bootstrap-OT and LEDH-PFPF-OT BF/FilterFlow-side adapter "
            "agreement across all six V2 rows?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "visible_runbook_path": RUNBOOK_PATH,
        "visible_ledger_path": LEDGER_PATH,
        "visible_stop_handoff_path": STOP_HANDOFF_PATH,
        "required_v2_model_ids": list(REQUIRED_V2_MODEL_IDS),
        "primary_criterion_fields": {
            "all_required_phase_pass_tokens_present": all(
                item["decision_matches_expected"] for item in phase_summaries
            ),
            "all_required_artifacts_exist": all(
                item["json_exists"] and item["result_exists"] and item["report_evidence_present"]
                for item in phase_summaries
            ),
            "all_phase_rows_preserved": all(item["row_order_matches"] for item in phase_summaries),
            "bootstrap_ot_contracts_values_and_gradients_passed": (
                algorithm_tables["bootstrap_ot"]["contracts"]["passed"]
                and algorithm_tables["bootstrap_ot"]["values"]["passed"]
                and algorithm_tables["bootstrap_ot"]["gradients"]["passed"]
            ),
            "ledh_pfpf_ot_contracts_values_and_gradients_passed": (
                algorithm_tables["ledh_pfpf_ot"]["contracts"]["passed"]
                and algorithm_tables["ledh_pfpf_ot"]["values"]["passed"]
                and algorithm_tables["ledh_pfpf_ot"]["gradients"]["passed"]
            ),
            "no_material_veto_open": local_pass,
            "claude_final_closeout_review_required_for_promotion": not promote_after_review,
            "claude_final_closeout_review_recorded": bool(
                p8_review_evidence["exists"]
                and p8_review_evidence["has_final_synthesis"]
                and p8_review_evidence["has_verdict_agree"]
            ),
        },
        "phase_summaries": phase_summaries,
        "row_decision_table": row_table,
        "algorithm_decision_table": algorithm_tables,
        "lineage_evidence": lineage,
        "command_manifest_evidence": command_manifest_evidence,
        "review_evidence": {
            **review_evidence,
            "p8_final_review": p8_review_evidence,
        },
        "veto_diagnostics": veto_diagnostics,
        "explanatory_only_fields": _explanatory_only_fields(phase_payloads),
        "decision_table": {
            "decision": decision,
            "primary_criterion_status": "PASS_LOCAL_REVIEW_PENDING" if local_pass else "BLOCKED",
            "veto_status": "CLEAR" if local_pass else "OPEN",
            "main_uncertainty": (
                "Same-contract agreement can miss shared adapter or contract-formula defects."
            ),
            "next_justified_action": (
                "run bounded Claude P8 closeout review"
                if decision == LOCAL_PENDING_DECISION
                else "update visible runbook, ledger, and stop handoff"
                if decision == PASS_DECISION
                else "review and classify open blockers"
            ),
            "not_concluded": NON_CLAIMS,
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "The BayesFilter-owned FilterFlow-side adapters and the "
                "BayesFilter execution can share a contract, formula, or "
                "model-convention defect, so agreement is weaker than "
                "independent correctness evidence."
            ),
            "what_would_overturn_closeout": (
                "Any missing required row, changed contract checksum or digest, "
                "unreviewed mismatch, omitted physical gradient knob, hidden "
                "student/FilterFlow mutation/oracle evidence, or reviewer finding "
                "that a non-claim was promoted to a claim."
            ),
            "weakest_part_of_evidence": (
                "The comparison is fixed-contract and fixed-branch; it deliberately "
                "does not test stochastic resampling distributions or gradients "
                "through random/discrete branch choices."
            ),
        },
        "review_round": 0 if not promote_after_review else 1,
        "open_material_blockers": open_blockers,
        "repair_amendment_required": bool(open_blockers),
        "next_allowed_action": (
            "run chunked Claude P8 closeout review before final promotion"
            if decision == LOCAL_PENDING_DECISION
            else "visible full-comparison closeout complete; update final handoff"
            if decision == PASS_DECISION
            else "write reviewed blocker classification before closing"
        ),
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": _environment_manifest(
            command=(
                "python -m experiments.dpf_implementation.tf_tfp.runners."
                "run_v2_algorithm_full_comparison_closeout"
                + (" --promote-after-review" if promote_after_review else "")
            ),
        ),
        "non_claims": NON_CLAIMS,
    }


def _phase_summaries(phase_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for spec in PHASES:
        phase = str(spec["phase"])
        payload = phase_payloads[phase]
        decision = str(payload.get("decision") or payload.get("status") or "")
        rows = payload.get("required_v2_model_ids")
        report_path = spec.get("report_path")
        result_text = _read_optional_text(spec["result_path"])
        summaries.append(
            {
                "phase": phase,
                "name": spec["name"],
                "json": str(spec["json_path"].relative_to(REPO_ROOT)),
                "report": (
                    str(report_path.relative_to(REPO_ROOT))
                    if isinstance(report_path, Path)
                    else None
                ),
                "result": str(spec["result_path"].relative_to(REPO_ROOT)),
                "expected_decision": spec["expected_decision"],
                "observed_decision": decision,
                "decision_matches_expected": decision == spec["expected_decision"],
                "json_exists": spec["json_path"].exists(),
                "report_exists": bool(report_path.exists()) if isinstance(report_path, Path) else False,
                "result_exists": spec["result_path"].exists(),
                "report_evidence_present": (
                    bool(report_path.exists())
                    if isinstance(report_path, Path)
                    else "Visible Command Summary" in result_text
                ),
                "row_order_matches": rows == REQUIRED_V2_MODEL_IDS,
                "reproducibility_digest": payload.get("reproducibility_digest"),
                "contract_bundle_checksum": payload.get("contract_bundle_checksum"),
                "open_material_blockers": payload.get("open_material_blockers", []),
                "veto_summary": _summarize_veto(payload.get("veto_diagnostics", {})),
            }
        )
    return summaries


def _algorithm_tables(phase_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p2 = phase_payloads["P2"]
    p3 = phase_payloads["P3"]
    p4 = phase_payloads["P4"]
    p5 = phase_payloads["P5"]
    p6 = phase_payloads["P6"]
    p7 = phase_payloads["P7"]
    return {
        "bootstrap_ot": {
            "contracts": _contract_decision("P2", p2, EXPECTED_P2_BUNDLE_CHECKSUM),
            "values": _value_decision("P3", p3),
            "gradients": _gradient_decision("P4", p4),
        },
        "ledh_pfpf_ot": {
            "contracts": _contract_decision("P5", p5, EXPECTED_P5_BUNDLE_CHECKSUM),
            "values": _value_decision("P6", p6),
            "gradients": _gradient_decision("P7", p7),
        },
    }


def _contract_decision(phase: str, payload: dict[str, Any], expected_checksum: str) -> dict[str, Any]:
    contracts = payload.get("contracts", [])
    rows = [contract.get("model_id") for contract in contracts]
    return {
        "phase": phase,
        "passed": (
            rows == REQUIRED_V2_MODEL_IDS
            and payload.get("contract_bundle_checksum") == expected_checksum
            and not _any_veto_fired(payload.get("veto_diagnostics", {}))
        ),
        "row_count": len(contracts),
        "row_order": rows,
        "contract_bundle_checksum": payload.get("contract_bundle_checksum"),
        "expected_contract_bundle_checksum": expected_checksum,
        "reproducibility_digest": payload.get("reproducibility_digest"),
        "decision": payload.get("decision"),
    }


def _value_decision(phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    primary = payload.get("primary_criterion_fields", {})
    return {
        "phase": phase,
        "passed": (
            summary.get("models") == REQUIRED_V2_MODEL_IDS
            and summary.get("status_counts") == {"MATCHED": 6}
            and bool(primary.get("all_rows_matched"))
            and not _any_veto_fired(payload.get("veto_diagnostics", {}))
        ),
        "row_order": summary.get("models"),
        "status_counts": summary.get("status_counts"),
        "max_abs_delta": summary.get("max_abs_delta"),
        "decision": payload.get("decision"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
    }


def _gradient_decision(phase: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    primary = payload.get("primary_criterion_fields", {})
    return {
        "phase": phase,
        "passed": (
            summary.get("models") == REQUIRED_V2_MODEL_IDS
            and summary.get("status_counts") == {"MATCHED": 5, "PREDECLARED_EXCLUDED": 1}
            and summary.get("predeclared_excluded_rows") == ["spatial_sir_j3_rk4"]
            and bool(primary.get("all_included_knobs_executed"))
            and bool(primary.get("all_gradient_rows_matched_or_predeclared_excluded"))
            and not bool(primary.get("finite_difference_promotion_gate"))
            and not _any_veto_fired(payload.get("veto_diagnostics", {}))
        ),
        "row_order": summary.get("models"),
        "status_counts": summary.get("status_counts"),
        "included_knob_count": summary.get("included_knob_count")
        or primary.get("total_included_physical_knobs"),
        "predeclared_excluded_rows": summary.get("predeclared_excluded_rows")
        or primary.get("predeclared_excluded_rows"),
        "max_abs_scalar_delta": summary.get("max_abs_scalar_delta"),
        "max_abs_gradient_delta": summary.get("max_abs_gradient_delta"),
        "finite_difference_promotion_gate": bool(primary.get("finite_difference_promotion_gate")),
        "decision": payload.get("decision"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
    }


def _row_table(phase_payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    p3_cells = _cells_by_model(phase_payloads["P3"])
    p4_cells = _cells_by_model(phase_payloads["P4"])
    p6_cells = _cells_by_model(phase_payloads["P6"])
    p7_cells = _cells_by_model(phase_payloads["P7"])
    rows = []
    for model in REQUIRED_V2_MODEL_IDS:
        bootstrap_grad = p4_cells[model]
        ledh_grad = p7_cells[model]
        rows.append(
            {
                "model_id": model,
                "bootstrap_ot_value_status": p3_cells[model]["status"],
                "bootstrap_ot_gradient_status": bootstrap_grad["status"],
                "bootstrap_ot_gradient_knobs": _cell_knobs(bootstrap_grad),
                "bootstrap_ot_max_scalar_delta": bootstrap_grad.get("metrics", {}).get("scalar_abs_delta"),
                "bootstrap_ot_max_gradient_delta": bootstrap_grad.get("metrics", {}).get("max_abs_gradient_delta"),
                "ledh_pfpf_ot_value_status": p6_cells[model]["status"],
                "ledh_pfpf_ot_gradient_status": ledh_grad["status"],
                "ledh_pfpf_ot_gradient_knobs": _cell_knobs(ledh_grad),
                "ledh_pfpf_ot_max_scalar_delta": ledh_grad.get("metrics", {}).get("scalar_abs_delta"),
                "ledh_pfpf_ot_max_gradient_delta": ledh_grad.get("metrics", {}).get("max_abs_gradient_delta"),
                "predeclared_gradient_exclusion": (
                    model == "spatial_sir_j3_rk4"
                    and bootstrap_grad["status"] == "PREDECLARED_EXCLUDED"
                    and ledh_grad["status"] == "PREDECLARED_EXCLUDED"
                ),
                "row_closeout_status": "PASSED_SAME_CONTRACT_AGREEMENT",
            }
        )
    return rows


def _lineage_evidence(phase_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p2 = phase_payloads["P2"]
    p3 = phase_payloads["P3"]
    p4 = phase_payloads["P4"]
    p5 = phase_payloads["P5"]
    p6 = phase_payloads["P6"]
    p7 = phase_payloads["P7"]
    return {
        "p2_contract_bundle_checksum": p2.get("contract_bundle_checksum"),
        "p2_contract_bundle_checksum_matches_expected": (
            p2.get("contract_bundle_checksum") == EXPECTED_P2_BUNDLE_CHECKSUM
        ),
        "p3_consumes_p2_checksum": p3.get("p2_contract_bundle_checksum") == EXPECTED_P2_BUNDLE_CHECKSUM,
        "p4_consumes_p2_checksum": p4.get("p2_contract_bundle_checksum") == EXPECTED_P2_BUNDLE_CHECKSUM,
        "p3_reproducibility_digest": p3.get("reproducibility_digest"),
        "p3_digest_matches_expected": p3.get("reproducibility_digest") == EXPECTED_P3_DIGEST,
        "p4_p3_digest_anchor_matches_expected": p4.get("p3_reproducibility_digest") == EXPECTED_P3_DIGEST,
        "p4_reproducibility_digest": p4.get("reproducibility_digest"),
        "p4_digest_matches_expected": p4.get("reproducibility_digest") == EXPECTED_P4_DIGEST,
        "p5_p4_digest_anchor_matches_expected": p5.get("p4_reproducibility_digest") == EXPECTED_P4_DIGEST,
        "p5_contract_bundle_checksum": p5.get("contract_bundle_checksum"),
        "p5_contract_bundle_checksum_matches_expected": (
            p5.get("contract_bundle_checksum") == EXPECTED_P5_BUNDLE_CHECKSUM
        ),
        "p5_reproducibility_digest": p5.get("reproducibility_digest"),
        "p5_digest_matches_expected": p5.get("reproducibility_digest") == EXPECTED_P5_DIGEST,
        "p6_consumes_p5_checksum": p6.get("p5_contract_bundle_checksum") == EXPECTED_P5_BUNDLE_CHECKSUM,
        "p6_p5_digest_anchor_matches_expected": p6.get("p5_reproducibility_digest") == EXPECTED_P5_DIGEST,
        "p6_reproducibility_digest": p6.get("reproducibility_digest"),
        "p6_digest_matches_expected": p6.get("reproducibility_digest") == EXPECTED_P6_DIGEST,
        "p7_consumes_p5_checksum": p7.get("p5_contract_bundle_checksum") == EXPECTED_P5_BUNDLE_CHECKSUM,
        "p7_p5_digest_anchor_matches_expected": p7.get("p5_reproducibility_digest") == EXPECTED_P5_DIGEST,
        "p7_p6_digest_anchor_matches_expected": p7.get("p6_reproducibility_digest") == EXPECTED_P6_DIGEST,
        "p7_reproducibility_digest": p7.get("reproducibility_digest"),
        "p7_digest_matches_expected": p7.get("reproducibility_digest") == EXPECTED_P7_DIGEST,
    }


def _review_evidence() -> dict[str, Any]:
    ledger_text = _read_optional_text(REPO_ROOT / LEDGER_PATH)
    review_paths = {
        "master_plan": "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-claude-review-ledger-2026-06-07.md",
        "p4": "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p4-claude-review-ledger-2026-06-08.md",
        "p5": "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p5-claude-review-ledger-2026-06-08.md",
        "p6": "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p6-claude-review-ledger-2026-06-08.md",
        "p7": "docs/plans/bayesfilter-dpf-v2-algorithm-full-bf-filterflow-comparison-p7-claude-review-ledger-2026-06-08.md",
    }
    evidence = {
        "visible_ledger_records_p0_pass": "PASS_P0_READY_FOR_P1" in ledger_text,
        "visible_ledger_records_p1_pass": "PASS_P1_ARCHITECTURE_READY_FOR_P2" in ledger_text,
        "visible_ledger_records_p2_final_agree": "Final P2 synthesis returned `VERDICT: AGREE`" in ledger_text,
        "visible_ledger_records_p3_final_agree": "Final P3 mini synthesis review returned `VERDICT: AGREE`" in ledger_text,
        "visible_ledger_records_p4_final_agree": "Final P4 synthesis review returned `VERDICT: AGREE`" in ledger_text,
        "visible_ledger_records_p5_pass": "PASS_P5_LEDH_PFPF_OT_CONTRACTS_READY_FOR_P6" in ledger_text,
        "visible_ledger_records_p6_pass": "PASS_P6_LEDH_PFPF_OT_VALUES_READY_FOR_P7" in ledger_text,
        "visible_ledger_records_p7_pass": "PASS_P7_LEDH_PFPF_OT_GRADIENTS_READY_FOR_P8" in ledger_text,
        "phase_review_ledgers": {},
    }
    for key, rel_path in review_paths.items():
        text = _read_optional_text(REPO_ROOT / rel_path)
        evidence["phase_review_ledgers"][key] = {
            "path": rel_path,
            "exists": bool(text),
            "has_agree": (
                "VERDICT: AGREE" in text
                or "verdict: `AGREE`" in text
                or "Verdict: `PASS`" in text
                or "status: CLOSED_PASS" in text
            ),
            "has_final_synthesis": "Final Synthesis" in text or "FINAL_SYNTHESIS" in text or "CLOSED_PASS" in text,
        }
    return evidence


def _p8_final_review_evidence() -> dict[str, Any]:
    text = _read_optional_text(REPO_ROOT / P8_REVIEW_LEDGER_PATH)
    final_section = ""
    marker = "Final Synthesis"
    if marker in text:
        final_section = text[text.rfind(marker) :]
    return {
        "path": P8_REVIEW_LEDGER_PATH,
        "exists": bool(text),
        "has_final_synthesis": bool(final_section),
        "has_verdict_agree": "VERDICT: AGREE" in final_section,
        "has_verdict_revise_after_final_agree": (
            "VERDICT: REVISE" in final_section
            and final_section.rfind("VERDICT: REVISE") > final_section.rfind("VERDICT: AGREE")
        )
        if final_section
        else False,
    }


def _command_manifest_evidence(phase_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for spec in PHASES:
        phase = str(spec["phase"])
        payload = phase_payloads[phase]
        result_text = _read_optional_text(spec["result_path"])
        if spec["artifact_kind"] == "runner":
            manifest = payload.get("run_manifest")
            evidence[phase] = {
                "kind": "runner_run_manifest",
                "present": isinstance(manifest, dict),
                "has_command": bool(isinstance(manifest, dict) and manifest.get("command")),
                "has_commit": bool(isinstance(manifest, dict) and manifest.get("commit")),
                "cpu_only": bool(isinstance(manifest, dict) and manifest.get("cpu_only")),
                "gpu_claim_absent": True,
            }
        else:
            execution_diagnostics = payload.get("execution_diagnostics", {})
            visible_command_summary = "Visible Command Summary" in result_text
            source_inspection_summary = (
                "source/document inspection only" in result_text
                and isinstance(execution_diagnostics, dict)
                and execution_diagnostics.get("tensorflow_commands_run") is False
                and execution_diagnostics.get("gpu_commands_run") is False
                and execution_diagnostics.get("student_implementation_commands_run") is False
                and execution_diagnostics.get("numerical_values_or_gradients_computed") is False
            )
            evidence[phase] = {
                "kind": "document_visible_command_summary",
                "present": visible_command_summary or source_inspection_summary,
                "has_command": visible_command_summary or source_inspection_summary,
                "has_commit": "git -C .localsource/filterflow rev-parse HEAD" in result_text,
                "cpu_only": not bool(execution_diagnostics.get("gpu_commands_run", False)),
                "gpu_claim_absent": (
                    "no GPU claim" in result_text.lower()
                    or "GPU command" in result_text
                    or execution_diagnostics.get("gpu_commands_run") is False
                ),
            }
    return evidence


def _veto_diagnostics(
    phase_payloads: dict[str, dict[str, Any]],
    phase_summaries: list[dict[str, Any]],
    algorithm_tables: dict[str, Any],
    row_table: list[dict[str, Any]],
    review_evidence: dict[str, Any],
    p8_review_evidence: dict[str, Any],
    command_manifest_evidence: dict[str, Any],
    lineage: dict[str, Any],
    *,
    promote_after_review: bool,
) -> dict[str, Any]:
    phase_tokens_missing = [
        item["phase"] for item in phase_summaries if not item["decision_matches_expected"]
    ]
    artifacts_missing = [
        item["phase"]
        for item in phase_summaries
        if not (item["json_exists"] and item["result_exists"] and item["report_evidence_present"])
    ]
    row_order_mismatch = [
        item["phase"] for item in phase_summaries if not item["row_order_matches"]
    ]
    open_blockers = {
        item["phase"]: item["open_material_blockers"]
        for item in phase_summaries
        if item["open_material_blockers"]
    }
    command_manifest_missing = [
        phase
        for phase, evidence in command_manifest_evidence.items()
        if not (
            evidence["present"]
            and evidence["has_command"]
            and (evidence["has_commit"] or evidence["kind"] == "document_visible_command_summary")
        )
    ]
    algorithm_failures = []
    for algorithm, stages in algorithm_tables.items():
        for stage, item in stages.items():
            if not item["passed"]:
                algorithm_failures.append(f"{algorithm}:{stage}")
    lineage_required_bool_suffixes = (
        "_matches_expected",
        "_consumes_p2_checksum",
        "_consumes_p5_checksum",
    )
    lineage_required_bool_keys = {
        "p4_p3_digest_anchor_matches_expected",
        "p5_p4_digest_anchor_matches_expected",
        "p6_p5_digest_anchor_matches_expected",
        "p7_p5_digest_anchor_matches_expected",
        "p7_p6_digest_anchor_matches_expected",
    }
    lineage_failures = [
        key
        for key, value in lineage.items()
        if (
            key.endswith(lineage_required_bool_suffixes)
            or key in lineage_required_bool_keys
        )
        and not value
    ]
    review_failures = [
        key for key, value in review_evidence.items()
        if key.startswith("visible_ledger_records_") and not value
    ]
    review_ledger_failures = [
        key
        for key, value in review_evidence.get("phase_review_ledgers", {}).items()
        if not (
            value.get("exists")
            and value.get("has_agree")
            and value.get("has_final_synthesis")
        )
    ]
    p8_review_failure = []
    if promote_after_review and not (
        p8_review_evidence.get("exists")
        and p8_review_evidence.get("has_final_synthesis")
        and p8_review_evidence.get("has_verdict_agree")
        and not p8_review_evidence.get("has_verdict_revise_after_final_agree")
    ):
        p8_review_failure.append("p8_final_synthesis_verdict_agree_missing")
    return {
        "phase_pass_token_missing_or_changed": phase_tokens_missing,
        "required_artifact_missing": artifacts_missing,
        "row_order_mismatch": row_order_mismatch,
        "unexecuted_required_row_reported_as_success": [
            row["model_id"]
            for row in row_table
            if row["model_id"] != "spatial_sir_j3_rk4"
            and (
                row["bootstrap_ot_gradient_status"] != "MATCHED"
                or row["ledh_pfpf_ot_gradient_status"] != "MATCHED"
            )
        ],
        "sir_predeclared_exclusion_hidden_or_missing": not all(
            row["predeclared_gradient_exclusion"]
            for row in row_table
            if row["model_id"] == "spatial_sir_j3_rk4"
        ),
        "algorithm_stage_failure": algorithm_failures,
        "open_material_blocker": open_blockers,
        "command_manifest_or_summary_missing": command_manifest_missing,
        "checksum_or_digest_lineage_failure": lineage_failures,
        "review_evidence_missing": review_failures,
        "phase_review_ledger_missing_or_without_agree": review_ledger_failures,
        "p8_final_review_missing_for_promotion": p8_review_failure,
        "finite_difference_promoted_to_gradient_gate": (
            bool(phase_payloads["P4"].get("primary_criterion_fields", {}).get("finite_difference_promotion_gate"))
            or bool(phase_payloads["P7"].get("primary_criterion_fields", {}).get("finite_difference_promotion_gate"))
        ),
        "localsource_filterflow_mutated": _governance_flag_fired(phase_payloads, "localsource_filterflow_mutated"),
        "student_command_or_metric": _governance_flag_fired(phase_payloads, "student_command_or_metric"),
        "oracle_framing": _governance_flag_fired(phase_payloads, "oracle_framing"),
        "unsupported_claim_terms_in_closeout_artifact": [],
    }


def _explanatory_only_fields(phase_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "bootstrap_value_max_abs_delta": phase_payloads["P3"].get("summary", {}).get("max_abs_delta"),
        "bootstrap_gradient_status_counts": phase_payloads["P4"].get("summary", {}).get("status_counts"),
        "bootstrap_gradient_max_abs_delta": phase_payloads["P4"].get("summary", {}).get("max_abs_gradient_delta"),
        "ledh_value_max_abs_delta": phase_payloads["P6"].get("summary", {}).get("max_abs_delta"),
        "ledh_gradient_status_counts": phase_payloads["P7"].get("summary", {}).get("status_counts"),
        "ledh_gradient_max_abs_delta": phase_payloads["P7"].get("summary", {}).get("max_abs_gradient_delta"),
        "finite_difference_policy": "Finite-difference ladders remain diagnostic-only and do not promote P8.",
        "runtime_policy": "Runtime, dirty status, and TensorFlow startup warnings are explanatory-only.",
        "historical_detached_route_policy": "Detached/overnight artifacts are historical context only for this visible P8 gate.",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "phase_summaries",
        "row_decision_table",
        "algorithm_decision_table",
        "lineage_evidence",
        "command_manifest_evidence",
        "review_evidence",
        "veto_diagnostics",
        "run_manifest",
        "non_claims",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"P8 payload missing keys: {sorted(missing)}")
    if payload["required_v2_model_ids"] != REQUIRED_V2_MODEL_IDS:
        raise ValueError("P8 required row order drifted")
    if len(payload["phase_summaries"]) != 8:
        raise ValueError("P8 must summarize P0--P7")
    if [row["model_id"] for row in payload["row_decision_table"]] != REQUIRED_V2_MODEL_IDS:
        raise ValueError("P8 row table order drifted")
    if payload["decision"] == PASS_DECISION and payload.get("review_round", 0) < 1:
        raise ValueError("P8 PASS_FULL_COMPARISON requires reviewed promotion")
    if payload["decision"] == PASS_DECISION and _any_veto_fired(payload["veto_diagnostics"]):
        raise ValueError("P8 cannot pass with open veto diagnostics")
    if payload["decision"] == LOCAL_PENDING_DECISION and _any_veto_fired(payload["veto_diagnostics"]):
        raise ValueError("P8 local-pending decision cannot carry open veto diagnostics")
    if payload["decision"] not in {LOCAL_PENDING_DECISION, PASS_DECISION, BLOCKED_DECISION}:
        raise ValueError(f"unexpected P8 decision: {payload['decision']}")
    if any("correctness proof" in claim for claim in payload["non_claims"]):
        raise ValueError("non-claims should avoid confusing proof wording")
    for claim in NON_CLAIMS:
        if claim not in payload["non_claims"]:
            raise ValueError(f"missing required non-claim: {claim}")


def _markdown(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    status_line = decision
    lines = [
        "# DPF V2 Algorithm Full BF/FilterFlow Comparison Closeout Result",
        "",
        "metadata_date: 2026-06-07",
        "visible_execution_timestamp: `2026-06-08T17:56:52+08:00`",
        "phase: P8",
        "execution_route: `VISIBLE_IN_DIALOGUE`",
        f"status: `{status_line}`",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion:",
        "",
        "- all visible P0--P7 pass tokens are present and reviewed;",
        "- required JSON, markdown/report, and docs/plans result artifacts exist;",
        "- all six V2 rows are retained in the required order;",
        "- bootstrap-OT contracts, values, and fixed-branch AD gradients passed;",
        "- LEDH-PFPF-OT contracts, values, and fixed-branch AD gradients passed;",
        "- no material veto remains open;",
        "- final P8 promotion still requires Claude closeout synthesis `VERDICT: AGREE` unless this artifact is generated with reviewed promotion.",
        "",
        "Veto diagnostics:",
        "",
        "- unexecuted required row reported as success;",
        "- missing phase pass token, result artifact, command manifest/summary, checksum, or digest;",
        "- unresolved mismatch or open material blocker;",
        "- hidden SIR gradient exclusion rather than predeclared no-physical-knob exclusion;",
        "- `.localsource/filterflow` mutation, student command/metric, oracle framing, or finite differences promoted to a gate;",
        "- unsupported stochastic, correctness, student, TT/SIRT, paper-table, GPU, HMC, DSGE, scalability, deployment, or production-readiness claim.",
        "",
        "Non-claims:",
        "",
    ]
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.extend(
        [
            "",
            "## Local Skeptical Phase Audit",
            "",
            "Audit status: `PASS_LOCAL_PHASE_AUDIT`."
            if not _any_veto_fired(payload["veto_diagnostics"])
            else "Audit status: `BLOCKED_BY_LOCAL_VETO`.",
            "",
            "Wrong-baseline risk: controlled by using only visible P0--P7 pass artifacts and frozen contract lineage.",
            "",
            "Proxy-metric risk: controlled because runtime, ESS, transport residuals, finite differences, and dirty status are explanatory-only.",
            "",
            "Missing stop-condition risk: controlled by P8 veto diagnostics for missing tokens, artifacts, review evidence, command manifests/summaries, checksums, row order, and unsupported claims.",
            "",
            "Unfair-comparison risk: controlled by P2/P5 contract checksum lineage into P3/P4 and P6/P7.",
            "",
            "Hidden-assumption risk: controlled by preserving the BayesFilter-owned FilterFlow-side adapter classification and no-mutation rule.",
            "",
            "Environment-mismatch risk: controlled because this P8 runner does not import TensorFlow and makes no GPU claim.",
            "",
            "Audit decision: local closeout pass pending Claude read-only P8 review."
            if decision == LOCAL_PENDING_DECISION
            else "Audit decision: reviewed closeout pass."
            if decision == PASS_DECISION
            else "Audit decision: closeout blocked pending reviewed classification.",
            "",
            "## Result",
            "",
            f"- Decision: `{decision}`",
            f"- JSON artifact: `{payload['artifact_paths']['json']}`",
            f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
            f"- Phase result: `{payload['artifact_paths']['phase_result']}`",
            f"- P2 contract bundle checksum: `{payload['lineage_evidence']['p2_contract_bundle_checksum']}`",
            f"- P5 contract bundle checksum: `{payload['lineage_evidence']['p5_contract_bundle_checksum']}`",
            f"- P7 reproducibility digest: `{payload['lineage_evidence']['p7_reproducibility_digest']}`",
            "",
            "## Phase Decision Table",
            "",
            "| Phase | Name | Observed decision | Row order | Artifacts | Open blockers |",
            "|---|---|---|---|---|---|",
        ]
    )
    for item in payload["phase_summaries"]:
        artifacts = (
            "PASS"
            if item["json_exists"] and item["result_exists"] and item["report_evidence_present"]
            else "FAIL"
        )
        blockers = item["open_material_blockers"] or []
        lines.append(
            "| `{phase}` | {name} | `{decision}` | {rows} | {artifacts} | `{blockers}` |".format(
                phase=item["phase"],
                name=item["name"],
                decision=item["observed_decision"],
                rows="PASS" if item["row_order_matches"] else "FAIL",
                artifacts=artifacts,
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Algorithm Decision Table",
            "",
            "| Algorithm | Contracts | Values | Gradients |",
            "|---|---|---|---|",
            _algorithm_markdown_row("bootstrap-OT", payload["algorithm_decision_table"]["bootstrap_ot"]),
            _algorithm_markdown_row("LEDH-PFPF-OT", payload["algorithm_decision_table"]["ledh_pfpf_ot"]),
            "",
            "## Row Decision Table",
            "",
            "| Model id | Bootstrap value | Bootstrap gradient | Bootstrap knobs | LEDH value | LEDH gradient | LEDH knobs |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["row_decision_table"]:
        lines.append(
            "| `{model_id}` | {bv} | {bg} | `{bknobs}` | {lv} | {lg} | `{lknobs}` |".format(
                model_id=row["model_id"],
                bv=row["bootstrap_ot_value_status"],
                bg=row["bootstrap_ot_gradient_status"],
                bknobs=row["bootstrap_ot_gradient_knobs"],
                lv=row["ledh_pfpf_ot_value_status"],
                lg=row["ledh_pfpf_ot_gradient_status"],
                lknobs=row["ledh_pfpf_ot_gradient_knobs"],
            )
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
        ]
    )
    for name, value in payload["veto_diagnostics"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Lineage Evidence",
            "",
        ]
    )
    for name, value in payload["lineage_evidence"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{payload['run_manifest']['commit']}` |",
            f"| git branch | `{payload['run_manifest']['branch']}` |",
            "| dirty status | full repo dirty state is recorded in JSON run manifest; unrelated dirty work is not interpreted as P8 evidence |",
            f"| command | `{payload['run_manifest']['command']}` |",
            "| validation commands | `python -m py_compile experiments/dpf_implementation/tf_tfp/runners/run_v2_algorithm_full_comparison_closeout.py`; `python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_algorithm_full_comparison_closeout`; `python -m json.tool experiments/dpf_implementation/reports/outputs/dpf_v2_algorithm_full_comparison_closeout_2026-06-07.json`; `python -m experiments.dpf_implementation.tf_tfp.runners.run_v2_algorithm_full_comparison_closeout --validate-only`; `git diff --check` on P8 files |",
            f"| environment | `{REPO_ROOT}`; Python `{payload['run_manifest']['python_version']}` |",
            "| CPU/GPU status | P8 pure Python closeout; TensorFlow not imported; no GPU claim |",
            "| random seeds | no RNG consumed in P8; artifact closeout only |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            (
                "review_round: 0 pending probe and chunked Claude P8 closeout review"
                if decision == LOCAL_PENDING_DECISION
                else "review_round: 1 final Claude synthesis returned VERDICT: AGREE"
                if decision == PASS_DECISION
                else "review_round: 0 blocked pending reviewed classification"
            ),
            "",
            f"open_material_blockers: `{payload['open_material_blockers']}`",
            "",
            f"repair_amendment_required: `{payload['repair_amendment_required']}`",
            "",
            f"next_allowed_action: {payload['next_allowed_action']}",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| `{decision}` | {primary} | {veto} | {uncertainty} | {next_action} | {nonclaims} |".format(
                decision=payload["decision_table"]["decision"],
                primary=payload["decision_table"]["primary_criterion_status"],
                veto=payload["decision_table"]["veto_status"],
                uncertainty=payload["decision_table"]["main_uncertainty"],
                next_action=payload["decision_table"]["next_justified_action"],
                nonclaims="; ".join(payload["decision_table"]["not_concluded"]),
            ),
            "",
            "## Post-Run Red Team",
            "",
            f"Strongest alternative explanation: {payload['post_run_red_team']['strongest_alternative_explanation']}",
            "",
            f"What would overturn the closeout: {payload['post_run_red_team']['what_would_overturn_closeout']}",
            "",
            f"Weakest part of the evidence: {payload['post_run_red_team']['weakest_part_of_evidence']}",
            "",
        ]
    )
    return "\n".join(lines)


def _algorithm_markdown_row(name: str, stages: dict[str, dict[str, Any]]) -> str:
    return "| {name} | {contracts} | {values} | {gradients} |".format(
        name=name,
        contracts="PASS" if stages["contracts"]["passed"] else "FAIL",
        values="PASS" if stages["values"]["passed"] else "FAIL",
        gradients="PASS" if stages["gradients"]["passed"] else "FAIL",
    )


def _refresh_decision_state(payload: dict[str, Any], *, promote_after_review: bool) -> None:
    open_blockers = [
        name for name, value in payload["veto_diagnostics"].items() if _veto_fired(value)
    ]
    local_pass = not open_blockers
    if promote_after_review and local_pass:
        decision = PASS_DECISION
    elif local_pass:
        decision = LOCAL_PENDING_DECISION
    else:
        decision = BLOCKED_DECISION
    payload["decision"] = decision
    payload["primary_criterion_fields"]["no_material_veto_open"] = local_pass
    payload["primary_criterion_fields"]["claude_final_closeout_review_required_for_promotion"] = (
        not promote_after_review
    )
    payload["decision_table"]["decision"] = decision
    payload["decision_table"]["primary_criterion_status"] = (
        "PASS_LOCAL_REVIEW_PENDING"
        if decision == LOCAL_PENDING_DECISION
        else "PASS_REVIEWED"
        if decision == PASS_DECISION
        else "BLOCKED"
    )
    payload["decision_table"]["veto_status"] = "CLEAR" if local_pass else "OPEN"
    payload["decision_table"]["next_justified_action"] = (
        "run bounded Claude P8 closeout review"
        if decision == LOCAL_PENDING_DECISION
        else "update visible runbook, ledger, and stop handoff"
        if decision == PASS_DECISION
        else "review and classify open blockers"
    )
    payload["review_round"] = 1 if decision == PASS_DECISION else 0
    payload["open_material_blockers"] = open_blockers
    payload["repair_amendment_required"] = bool(open_blockers)
    payload["next_allowed_action"] = (
        "run chunked Claude P8 closeout review before final promotion"
        if decision == LOCAL_PENDING_DECISION
        else "visible full-comparison closeout complete; update final handoff"
        if decision == PASS_DECISION
        else "write reviewed blocker classification before closing"
    )


def _unsupported_claim_terms(texts: list[str]) -> list[str]:
    combined = "\n".join(texts).lower()
    forbidden_patterns = {
        "proves BayesFilter correctness": "proves bayesfilter correctness",
        "proves FilterFlow correctness": "proves filterflow correctness",
        "establishes stochastic resampling correctness": (
            "establishes stochastic resampling distribution correctness"
        ),
        "establishes gradients through random branches": (
            "establishes gradients through random or discrete branch"
        ),
        "student implementation success claim": "student implementation passed",
        "GPU success claim": "gpu pass",
        "production readiness claim": "production-ready",
        "deployment readiness claim": "deployment-ready",
        "scalability claim": "scalability passed",
        "paper-table claim": "paper table matched",
        "TT/SIRT claim": "tt/sirt passed",
        "TT claim": "tensor train passed",
        "SIRT claim": "sirt passed",
        "HMC claim": "hmc passed",
        "DSGE claim": "dsge passed",
        "dense quadrature claim": "dense quadrature passed",
        "simulated-truth claim": "simulated truth matched",
        "TT/SIRT readiness claim": "tt/sirt ready",
        "HMC readiness claim": "hmc ready",
        "DSGE readiness claim": "dsge ready",
        "GPU readiness claim": "gpu ready",
        "oracle framing": "as oracle",
    }
    hits = []
    for label, pattern in forbidden_patterns.items():
        if pattern in combined:
            hits.append(label)
    return hits


def _governance_flag_fired(phase_payloads: dict[str, dict[str, Any]], key: str) -> bool:
    for payload in phase_payloads.values():
        veto = payload.get("veto_diagnostics", {})
        if isinstance(veto, dict):
            value = veto.get(key)
            if value is True or value == "FAIL":
                return True
        governance = payload.get("governance_evidence", {})
        if isinstance(governance, dict) and governance.get(key) is True:
            return True
    return False


def _cells_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(cell["model"]): cell for cell in payload.get("cells", [])}


def _cell_knobs(cell: dict[str, Any]) -> list[str]:
    side = cell.get("bayesfilter")
    if isinstance(side, dict):
        return list(side.get("gradient_knob_names", []))
    return list(cell.get("metrics", {}).get("excluded_knobs", []))


def _summarize_veto(veto: Any) -> dict[str, Any]:
    if not isinstance(veto, dict):
        return {"non_dict_veto_payload": veto}
    counts = Counter()
    fired = []
    for key, value in veto.items():
        if _veto_fired(value):
            fired.append(key)
            counts["fired"] += 1
        else:
            counts["clear"] += 1
    return {"counts": dict(counts), "fired": fired}


def _any_veto_fired(veto: dict[str, Any]) -> bool:
    return any(_veto_fired(value) for value in veto.values())


def _veto_fired(value: Any) -> bool:
    if value is False or value is None:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        return value not in {"PASS", "CLEAR", "False", "false", ""}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return bool(value)


def _environment_manifest(*, command: str) -> dict[str, Any]:
    return {
        **_git_manifest(),
        "python_version": platform.python_version(),
        "package_versions": {},
        "cpu_only": True,
        "tensorflow_imported": False,
        "gpu_devices_visible": "not_probed_p8_closeout_no_gpu_claim",
        "command": command,
    }


def _git_manifest() -> dict[str, str]:
    return {
        "branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty_state_summary": _git(["git", "status", "--short"]) or "clean",
    }


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _stable_digest(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    if isinstance(clone.get("run_manifest"), dict):
        manifest = dict(clone["run_manifest"])
        manifest.pop("wall_time_seconds", None)
        manifest.pop("dirty_state_summary", None)
        clone["run_manifest"] = manifest
    return _stable_digest(clone)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
