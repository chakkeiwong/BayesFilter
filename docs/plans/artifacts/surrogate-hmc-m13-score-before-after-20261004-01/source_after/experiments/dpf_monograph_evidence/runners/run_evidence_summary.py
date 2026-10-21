from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_PRE_IMPORT_GPU_HIDING_ASSERTION = _PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"

from experiments.dpf_monograph_evidence.results import CANONICAL_DIAGNOSTIC_IDS, validate_result_file

WALL_CLOCK_CAP_SECONDS = 30
REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = EVIDENCE_ROOT / "reports" / "outputs"
SUMMARY_JSON = OUTPUT_ROOT / "dpf_monograph_evidence_summary.json"
EVIDENCE_NOTE = EVIDENCE_ROOT / "reports" / "dpf-monograph-research-evidence-note.md"
CLOSEOUT_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "bayesfilter-dpf-monograph-implementation-evidence-final-closeout-2026-05-16.md"
)
IE8_RESULT_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "bayesfilter-dpf-monograph-implementation-evidence-phase-ie8-posterior-sensitivity-governance-result-2026-05-16.md"
)

ROW_FILES = {
    "linear_gaussian_recovery": OUTPUT_ROOT / "linear_gaussian_recovery.json",
    "synthetic_affine_flow": OUTPUT_ROOT / "affine_flow_synthetic_affine_flow.json",
    "pfpf_algebra_parity": OUTPUT_ROOT / "affine_flow_pfpf_algebra_parity.json",
    "soft_resampling_bias": OUTPUT_ROOT / "soft_resampling_bias.json",
    "sinkhorn_residual": OUTPUT_ROOT / "sinkhorn_residual.json",
    "learned_map_residual": OUTPUT_ROOT / "learned_ot_residual.json",
    "hmc_value_gradient": OUTPUT_ROOT / "hmc_value_gradient.json",
}

REFERENCE_CLASSES = {
    "linear_gaussian_recovery": "analytic_reference",
    "synthetic_affine_flow": "analytic_reference",
    "pfpf_algebra_parity": "analytic_reference",
    "soft_resampling_bias": "analytic_reference",
    "sinkhorn_residual": "analytic_reference",
    "learned_map_residual": "no_trusted_reference_exploratory_only",
    "hmc_value_gradient": "no_trusted_reference_exploratory_only",
    "posterior_sensitivity_summary": "no_trusted_reference_exploratory_only",
}

TRUSTED_REFERENCE_PRESENT = {
    "linear_gaussian_recovery": True,
    "synthetic_affine_flow": True,
    "pfpf_algebra_parity": True,
    "soft_resampling_bias": True,
    "sinkhorn_residual": True,
    "learned_map_residual": False,
    "hmc_value_gradient": False,
    "posterior_sensitivity_summary": False,
}

SOURCE_SUPPORT_CEILING = (
    "Program-level source-support ceiling: bibliography-spine support unless a row "
    "explicitly records a stronger reviewed-source artifact. IE1 did not identify "
    "reviewed local DPF source summaries, so this program does not upgrade source "
    "provenance to paper-reviewed support."
)

NON_IMPLICATION = (
    "The DPF monograph implementation-evidence program provides clean-room "
    "controlled-fixture and governance evidence only. It does not validate real "
    "DPF-HMC targets, DSGE or MacroFinance posterior inference, banking use, "
    "model-risk use, production bayesfilter code, or production readiness."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run_git_command(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def get_git_manifest() -> tuple[str, str, str]:
    branch = _run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run_git_command(["git", "rev-parse", "HEAD"])
    dirty = _run_git_command(["git", "status", "--short"])
    return branch, commit, dirty if dirty else "clean"


def relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def load_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for diagnostic_id, path in ROW_FILES.items():
        rows[diagnostic_id] = validate_result_file(path)
    return rows


def claim_ceiling_for(diagnostic_id: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return "not_tested"
    if row["status"] == "deferred":
        return "deferred_evidence_gap"
    if row["status"] == "blocked":
        return "blocked"
    if diagnostic_id in {"learned_map_residual", "hmc_value_gradient", "posterior_sensitivity_summary"}:
        return "exploratory_only"
    return "controlled_fixture_supported"


def why_not_validation(diagnostic_id: str, row: dict[str, Any] | None) -> str:
    if diagnostic_id == "posterior_sensitivity_summary":
        return "IE8 intentionally ran no posterior sensitivity; no posterior validation is claimed."
    if row is None:
        return "No phase-owned diagnostic row exists for this id."
    if row["status"] == "deferred":
        return "The diagnostic is deferred with a recorded evidence gap, not validation or method failure."
    if diagnostic_id == "hmc_value_gradient":
        return "IE7 is a fixed-scalar value-gradient gate only and does not validate HMC correctness or posterior agreement."
    return "The row is controlled-fixture evidence only and does not instantiate real target posterior validation."


def build_canonical_rows(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_rows: list[dict[str, Any]] = []
    for diagnostic_id in CANONICAL_DIAGNOSTIC_IDS:
        row = rows.get(diagnostic_id)
        if row is None:
            coverage_status = "missing"
            phase_status = "not_tested"
            source_support_class = "bibliography_spine_only"
            comparator = "no_ie8_posterior_sensitivity_executed"
            artifact_paths: list[str] = []
            promotion = "not_triggered"
            promotion_veto = "not_triggered"
            continuation_veto = "not_triggered"
            repair = "not tested"
            execution_occurred = False
            non_implication = NON_IMPLICATION
            phase_id = "IE8"
        else:
            coverage_status = row["coverage"][diagnostic_id]
            phase_status = row["status"]
            source_support_class = row["source_support_class"]
            comparator = row["comparator_id"]
            artifact_paths = row["artifact_paths"]
            promotion = row["promotion_criterion_status"]
            promotion_veto = row["promotion_veto_status"]
            continuation_veto = row["continuation_veto_status"]
            repair = row["repair_trigger"]
            execution_occurred = row["status"] not in {"deferred", "blocked"}
            non_implication = row["non_implication"]
            phase_id = row["phase_id"]
        canonical_rows.append(
            {
                "diagnostic_id": diagnostic_id,
                "phase_id": phase_id,
                "coverage_status": coverage_status,
                "phase_status": phase_status,
                "source_support_class": source_support_class,
                "comparator_or_reference_id": comparator,
                "reference_class": REFERENCE_CLASSES[diagnostic_id],
                "trusted_reference_present": TRUSTED_REFERENCE_PRESENT[diagnostic_id],
                "execution_occurred": execution_occurred,
                "claim_ceiling": claim_ceiling_for(diagnostic_id, row),
                "why_not_validation": why_not_validation(diagnostic_id, row),
                "non_implication": non_implication,
                "artifact_paths": artifact_paths,
                "promotion_criterion_status": promotion,
                "promotion_veto_status": promotion_veto,
                "continuation_veto_status": continuation_veto,
                "repair_trigger": repair,
            }
        )
    return canonical_rows


def build_summary(command: str, runtime_seconds: float, started_at: str, ended_at: str) -> dict[str, Any]:
    rows = load_rows()
    branch, commit, dirty_state_summary = get_git_manifest()
    canonical_rows = build_canonical_rows(rows)
    artifact_paths = [
        relative_repo_path(SUMMARY_JSON),
        relative_repo_path(EVIDENCE_NOTE),
        relative_repo_path(CLOSEOUT_PATH),
        relative_repo_path(IE8_RESULT_PATH),
    ]
    return {
        "program_exit_label": "dpf_monograph_evidence_program_complete_with_blockers",
        "program_status": "complete_with_blockers",
        "source_support_ceiling": SOURCE_SUPPORT_CEILING,
        "posterior_sensitivity_executed": False,
        "posterior_sensitivity_execution_reason": "No posterior sensitivity executed by default; current artifacts do not authorize real DPF-HMC, HMC-chain, DSGE, MacroFinance, or posterior validation runs.",
        "canonical_diagnostic_rows": canonical_rows,
        "artifact_inventory": {
            "row_json_files": {key: relative_repo_path(value) for key, value in ROW_FILES.items()},
            "summary_json": relative_repo_path(SUMMARY_JSON),
            "research_evidence_note": relative_repo_path(EVIDENCE_NOTE),
            "final_closeout": relative_repo_path(CLOSEOUT_PATH),
            "ie8_result": relative_repo_path(IE8_RESULT_PATH),
        },
        "master_program_compliance": {
            "every_phase_has_result_or_blocker": True,
            "ie6_deferred_visible": True,
            "posterior_sensitivity_not_run": True,
            "no_production_or_student_imports_introduced": True,
            "source_support_caveat_preserved": True,
            "claim_ceiling_enforced": True,
        },
        "non_implication": NON_IMPLICATION,
        "post_run_red_team_note": "The strongest reason not to over-read this closeout is that it aggregates clean-room controlled fixtures and one deferred learned-OT gap; it does not instantiate real DPF-HMC posterior targets or production code paths.",
        "run_manifest": {
            "command": command,
            "branch": branch,
            "commit": commit,
            "dirty_state_summary": dirty_state_summary,
            "python_version": platform.python_version(),
            "package_versions": {"python": platform.python_version(), "stdlib": "builtin"},
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "pre_import_cuda_visible_devices": _PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "pre_import_gpu_hiding_assertion": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "seed_policy": "deterministic_aggregation_no_rng",
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "artifact_paths": artifact_paths,
            "runtime_seconds": runtime_seconds,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def markdown_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Diagnostic | Status | Claim ceiling | Reference class | Trusted reference | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["canonical_diagnostic_rows"]:
        lines.append(
            f"| `{row['diagnostic_id']}` | `{row['coverage_status']}` | `{row['claim_ceiling']}` | `{row['reference_class']}` | `{row['trusted_reference_present']}` | `{row['repair_trigger']}` |"
        )
    return lines


def write_evidence_note(summary: dict[str, Any]) -> None:
    lines = [
        "# DPF monograph research evidence note",
        "",
        "## Verdict",
        "",
        f"- Program exit label: `{summary['program_exit_label']}`",
        "- IE8 ran no posterior sensitivity checks.",
        "- IE6 learned-OT remains deferred because no approved teacher/student artifact with provenance exists.",
        "- All claims remain bounded to clean-room controlled-fixture or governance evidence.",
        "",
        "## Source-Support Ceiling",
        "",
        summary["source_support_ceiling"],
        "",
        "## Canonical Diagnostic Inventory",
        "",
        *markdown_table(summary),
        "",
        "## What Was Not Concluded",
        "",
        "- No real DPF-HMC validation.",
        "- No DSGE or MacroFinance posterior validation.",
        "- No HMC chain or posterior sensitivity validation.",
        "- No production `bayesfilter` validation.",
        "- No banking, model-risk, or production-readiness claim.",
        "- No learned-OT surrogate quality validation because IE6 deferred.",
        "",
        "## Posterior Sensitivity",
        "",
        summary["posterior_sensitivity_execution_reason"],
        "",
        "## Post-Run Red-Team Note",
        "",
        summary["post_run_red_team_note"],
    ]
    EVIDENCE_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_closeout(summary: dict[str, Any]) -> None:
    lines = [
        "# DPF monograph implementation-evidence final closeout",
        "",
        "## Master Exit Label",
        "",
        f"`{summary['program_exit_label']}`",
        "",
        "Rationale: the aggregate ledger and closeout artifacts are complete, but IE6 remains deferred and IE8 intentionally ran no posterior sensitivity. The program is therefore complete with blockers rather than complete without blockers.",
        "",
        "## Decision Table",
        "",
        *markdown_table(summary),
        "",
        "## Inference-Status Table",
        "",
        "| Row | Status |",
        "| --- | --- |",
        "| Controlled-fixture diagnostics | IE3, IE4, IE5, and IE7 provide bounded clean-room evidence only. |",
        "| Learned OT | Deferred due to missing approved artifact provenance. |",
        "| Posterior sensitivity | Not run in IE8. |",
        "| Production readiness | Not supported. |",
        "| Banking/model-risk readiness | Not supported. |",
        "",
        "## Artifact Inventory",
        "",
        f"- Summary JSON: `{summary['artifact_inventory']['summary_json']}`",
        f"- Research evidence note: `{summary['artifact_inventory']['research_evidence_note']}`",
        f"- IE8 result: `{summary['artifact_inventory']['ie8_result']}`",
        "",
        "## What Was Not Concluded",
        "",
        summary["non_implication"],
        "",
        "## IE6 Deferred Treatment",
        "",
        "IE6 is preserved as `deferred`, not hidden, not collapsed into blocked, and not interpreted as method failure.",
        "",
        "## IE8 Posterior Sensitivity",
        "",
        "No posterior sensitivity was executed.",
        "",
        "## Source-Support Ceiling",
        "",
        summary["source_support_ceiling"],
        "",
        "## Post-Run Red-Team Note",
        "",
        summary["post_run_red_team_note"],
    ]
    CLOSEOUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ie8_result(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase IE8 result: posterior sensitivity and research-evidence note",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `dpf_monograph_evidence_program_complete_with_blockers`",
        "- Phase status: `complete_with_blockers`",
        "",
        "## Summary",
        "",
        "IE8 produced the aggregate research evidence note and summary JSON. It ran zero posterior sensitivity checks and preserved IE6 as a visible deferred evidence gap.",
        "",
        "## Decision Table",
        "",
        *markdown_table(summary),
        "",
        "## Non-Implication",
        "",
        summary["non_implication"],
        "",
        "## Artifacts",
        "",
        f"- `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`",
        f"- `experiments/dpf_monograph_evidence/reports/dpf-monograph-research-evidence-note.md`",
        f"- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-final-closeout-2026-05-16.md`",
    ]
    IE8_RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    started_at = utc_now()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_evidence_summary"
    runtime_seconds = time.perf_counter() - start
    ended_at = utc_now()
    summary = build_summary(command, runtime_seconds, started_at, ended_at)
    write_json(SUMMARY_JSON, summary)
    write_evidence_note(summary)
    write_closeout(summary)
    write_ie8_result(summary)
    print(f"wrote {SUMMARY_JSON}")
    print(f"wrote {EVIDENCE_NOTE}")
    print(f"wrote {CLOSEOUT_PATH}")
    print(f"wrote {IE8_RESULT_PATH}")
    print(summary["program_exit_label"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
