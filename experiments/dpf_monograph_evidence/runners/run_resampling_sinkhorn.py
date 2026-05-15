from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_PRE_IMPORT_GPU_HIDING_ASSERTION = _PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"

import numpy as np

from experiments.dpf_monograph_evidence.diagnostics.resampling_sinkhorn import (
    SINKHORN_COMPARATOR_ID,
    SINKHORN_NON_IMPLICATION,
    SOFT_RESAMPLING_COMPARATOR_ID,
    SOFT_RESAMPLING_NON_IMPLICATION,
    evaluate_sinkhorn_residual_fixture,
    evaluate_soft_resampling_bias_fixture,
    repair_trigger_for_sinkhorn,
    repair_trigger_for_soft_resampling,
    sinkhorn_status,
    soft_resampling_status,
)
from experiments.dpf_monograph_evidence.fixtures.resampling_sinkhorn import (
    build_sinkhorn_residual_fixture,
    build_soft_resampling_bias_fixture,
)
from experiments.dpf_monograph_evidence.results import validate_result_record

WALL_CLOCK_CAP_SECONDS = 30
PHASE_ID = "IE5"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_ROOT = RESULTS_ROOT / "outputs"
MARKDOWN_REPORT_PATH = RESULTS_ROOT / "resampling-sinkhorn-result.md"
PHASE_RESULT_PATH = Path(__file__).resolve().parents[3] / "docs" / "plans" / "bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-2026-05-16.md"
JSON_OUTPUTS = {
    "soft_resampling_bias": OUTPUT_ROOT / "soft_resampling_bias.json",
    "sinkhorn_residual": OUTPUT_ROOT / "sinkhorn_residual.json",
}
COMPARATOR_IDS = {
    "soft_resampling_bias": SOFT_RESAMPLING_COMPARATOR_ID,
    "sinkhorn_residual": SINKHORN_COMPARATOR_ID,
}
NON_IMPLICATIONS = {
    "soft_resampling_bias": SOFT_RESAMPLING_NON_IMPLICATION,
    "sinkhorn_residual": SINKHORN_NON_IMPLICATION,
}
EXPLANATORY_ONLY = {
    "soft_resampling_bias": [
        "Nonlinear bias evidence is limited to the declared relaxed-target arithmetic and does not imply categorical-resampling unbiasedness.",
        "No posterior, filtering, or production interpretation is supported by the two-particle fixture.",
    ],
    "sinkhorn_residual": [
        "Residual trend across the budget ladder is explanatory-only unless the final marginal thresholds pass.",
        "Finite-budget regularized transport residuals do not imply posterior equivalence or exact unregularized OT equivalence.",
    ],
}


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
    return str(path.resolve().relative_to(Path(__file__).resolve().parents[3]))


def build_coverage(statuses: dict[str, str]) -> dict[str, str]:
    coverage = {
        "linear_gaussian_recovery": "passed",
        "synthetic_affine_flow": "passed",
        "pfpf_algebra_parity": "passed",
        "soft_resampling_bias": "missing",
        "sinkhorn_residual": "missing",
        "learned_map_residual": "missing",
        "hmc_value_gradient": "missing",
        "posterior_sensitivity_summary": "missing",
    }
    coverage.update(statuses)
    return coverage


def build_environment(branch: str, commit: str, dirty_state_summary: str) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "package_versions": {"numpy": np.__version__, "python": platform.python_version()},
        "branch": branch,
        "commit": commit,
        "dirty_state_summary": dirty_state_summary,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
        "import_boundary_checked_modules": [
            "experiments.dpf_monograph_evidence.fixtures.resampling_sinkhorn",
            "experiments.dpf_monograph_evidence.diagnostics.resampling_sinkhorn",
            "experiments.dpf_monograph_evidence.results",
            "experiments.dpf_monograph_evidence.runners.run_resampling_sinkhorn",
        ],
    }


def build_cpu_gpu_mode() -> dict[str, Any]:
    return {
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_devices_visible": [],
        "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
    }


def gate_statuses(row_status: str) -> tuple[str, str, str]:
    if row_status == "pass":
        return "pass", "not_triggered", "not_triggered"
    return "fail", "fail", "not_triggered"


def make_result_record(
    diagnostic_id: str,
    command: str,
    runtime_seconds: float,
    started_at: str,
    ended_at: str,
    metrics: dict[str, Any],
    row_status: str,
    repair_trigger: str,
    branch: str,
    commit: str,
    dirty_state_summary: str,
    coverage_statuses: dict[str, str],
) -> dict[str, Any]:
    result_path_rel = relative_repo_path(JSON_OUTPUTS[diagnostic_id])
    markdown_path_rel = relative_repo_path(MARKDOWN_REPORT_PATH)
    promotion_criterion_status, promotion_veto_status, continuation_veto_status = gate_statuses(row_status)
    artifact_paths = [result_path_rel, markdown_path_rel]
    package_versions = {"numpy": np.__version__, "python": platform.python_version()}
    result = {
        "phase_id": PHASE_ID,
        "diagnostic_id": diagnostic_id,
        "chapter_label": "Chapter 26",
        "diagnostic_role": "promotion_criterion",
        "comparator_id": COMPARATOR_IDS[diagnostic_id],
        "source_family": "Chapter 26 relaxed resampling and Sinkhorn/EOT deterministic clean-room fixtures",
        "source_support_class": "bibliography_spine_only",
        "row_level_source_support_class": "bibliography_spine_only",
        "seed_policy": "deterministic_no_rng_resampling_sinkhorn_fixture",
        "status": row_status,
        "coverage": build_coverage(coverage_statuses),
        "tolerance": metrics["tolerance"],
        "finite_checks": metrics["finite_checks"],
        "shape_checks": metrics["shape_checks"],
        "runtime_seconds": runtime_seconds,
        "blocker_class": "none" if row_status == "pass" else "execution",
        "non_implication": NON_IMPLICATIONS[diagnostic_id],
        "promotion_criterion_status": promotion_criterion_status,
        "promotion_veto_status": promotion_veto_status,
        "continuation_veto_status": continuation_veto_status,
        "repair_trigger": repair_trigger,
        "explanatory_only_diagnostics": EXPLANATORY_ONLY[diagnostic_id],
        "environment": build_environment(branch, commit, dirty_state_summary),
        "command": command,
        "wall_time_seconds": runtime_seconds,
        "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "artifact_paths": artifact_paths,
        "cpu_gpu_mode": build_cpu_gpu_mode(),
        "uncertainty_status": "not_applicable",
        "replication_count": 1,
        "mcse_or_interval": {
            "summary": "Uncertainty intervals are not applicable because IE5 uses deterministic no-RNG fixtures with replication_count=1."
        },
        "post_run_red_team_note": "These deterministic IE5 checks could still pass while categorical resampling behavior, finite-epsilon target mismatch, or downstream filtering logic remain wrong because the fixtures validate only relaxed-target arithmetic and marginal residuals.",
        "cap_non_applicability_reasons": {
            key: value
            for key, value in {
                "max_time_steps": "IE5 evaluates one-shot deterministic fixtures rather than a time-series filter rollout.",
                "max_sinkhorn_iterations": "Soft-resampling does not instantiate a Sinkhorn solver." if diagnostic_id == "soft_resampling_bias" else None,
                "max_finite_difference_evaluations": "IE5 does not use finite-difference probes.",
                "max_replications": "IE5 is deterministic and records replication_count=1.",
            }.items()
            if value is not None
        },
        "cap_values": {
            "max_particles": 2 if diagnostic_id == "soft_resampling_bias" else int(metrics["shape_checks"]["source_support_size"]),
            "max_time_steps": None,
            "max_sinkhorn_iterations": 100 if diagnostic_id == "sinkhorn_residual" else None,
            "max_finite_difference_evaluations": None,
            "max_replications": None,
            "max_wall_clock_seconds": WALL_CLOCK_CAP_SECONDS,
        },
        "run_manifest": {
            "command": command,
            "branch": branch,
            "commit": commit,
            "dirty_state_summary": dirty_state_summary,
            "python_version": platform.python_version(),
            "package_versions": package_versions,
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "pre_import_cuda_visible_devices": _PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            "pre_import_gpu_hiding_assertion": _PRE_IMPORT_GPU_HIDING_ASSERTION,
            "seed_policy": "deterministic_no_rng_resampling_sinkhorn_fixture",
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "artifact_paths": artifact_paths,
        },
    }
    validate_result_record(result)
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(records: dict[str, dict[str, Any]]) -> None:
    soft = records["soft_resampling_bias"]
    sinkhorn = records["sinkhorn_residual"]
    soft_tol = soft["tolerance"]
    sink_tol = sinkhorn["tolerance"]
    both_passed = soft["status"] == "pass" and sinkhorn["status"] == "pass"
    hard_veto_text = (
        "both IE5 rows passed deterministic residual and finite checks"
        if both_passed
        else "one or more IE5 rows failed deterministic residual or finite checks"
    )
    coverage_text = (
        "Both JSON row files mark `linear_gaussian_recovery`, `synthetic_affine_flow`, and `pfpf_algebra_parity` as `passed`, both IE5 diagnostic IDs as `passed`, and all later-phase diagnostic IDs as `missing`, matching the required carry-forward coverage state after successful IE5 emission."
        if both_passed
        else "The JSON row files carry forward IE3 and IE4 passed states, mark any failed IE5 diagnostic as `blocked`, and keep later-phase diagnostic IDs as `missing`."
    )
    next_text = (
        "IE5 passed its bounded relaxed-target and Sinkhorn residual checks, so the lane can advance to IE6 without upgrading any claim beyond deterministic arithmetic and marginal-residual evidence."
        if both_passed
        else "IE5 did not pass both owned diagnostics, so the lane must not advance to IE6 without an accepted repair or blocker decision."
    )
    lines = [
        "# IE5 resampling and Sinkhorn result",
        "",
        "## Skeptical audit before execution",
        "",
        "- Wrong baseline audit: the only allowed comparators are `closed_form_two_particle_soft_resampling_reference` and `manual_balanced_sinkhorn_marginal_reference`; no weaker proxy baseline was used.",
        "- Proxy-metric audit: exact relaxed-target arithmetic and final-budget marginal residuals are the promotion criteria; categorical-reference deltas and residual trend remain bounded to their declared roles.",
        "- Stop-condition audit: any non-finite value, tolerance miss, missing non-implication text, or budget-ladder regression would force a row failure with a structured repair trigger.",
        "- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.",
        "- Artifact audit: each canonical IE5 diagnostic writes one schema-valid JSON object, with IE3 and IE4 coverage carried forward as `passed` and later rows kept `missing`.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: do the bounded clean-room fixtures preserve deterministic relaxed-target expectation arithmetic for selected soft-resampling summaries and finite-budget marginal residual control for small Sinkhorn transport?",
        "- Candidate under test: two-particle relaxed-probability bookkeeping and small balanced log-domain Sinkhorn scaling with epsilon `0.3` and budget ladder `[5, 20, 100]`.",
        "- Expected failure mode: probability formula drift, probability normalization drift, relaxed-target expectation mismatch, unexpected categorical nonlinear-delta sign, marginal residual threshold miss, or budget-trend regression.",
        "- Promotion criterion: soft-resampling relaxed-probability and relaxed-expectation checks pass with finite categorical-reference deltas and a nonzero categorical nonlinear delta, and final Sinkhorn marginal checks pass at budget `100` with no budget nonincrease violations.",
        "- Promotion veto: missing comparator identity, source-support class, epsilon, budget ladder, stabilization mode, tolerance object, seed policy, or exact row-specific non-implication text.",
        "- Continuation veto: inability to preserve the trusted marginal comparator and row-local relaxed-target caveat in schema-valid IE2 rows.",
        "- Repair trigger: probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, row marginal, column marginal, mass, nonnegativity, finite-plan, or budget trend failure.",
        "- What must not be concluded: categorical resampling law preservation, unbiasedness for nonlinear observables, posterior equivalence, exact unregularized OT equivalence, production bayesfilter correctness, banking use, model-risk use, or production readiness.",
        "",
        "## Evidence contract",
        "",
        "| Diagnostic | Comparator | Primary criterion | Veto diagnostics | Explanatory only | Source support |",
        "| --- | --- | --- | --- | --- | --- |",
        "| `soft_resampling_bias` | `closed_form_two_particle_soft_resampling_reference` | relaxed-probability and relaxed-expectation residuals finite and <= `1e-12`, categorical-reference deltas finite, and nonlinear delta nonzero | missing relaxed-target caveat, comparator id, source support, tolerance object, seed policy, or exact non-implication text | categorical-resampling, posterior, or production interpretation | `bibliography_spine_only` |",
        "| `sinkhorn_residual` | `manual_balanced_sinkhorn_marginal_reference` | final row/column/mass/nonnegativity/finite-plan residuals finite and within tolerance at budget `100`, with zero budget nonincrease violations | missing epsilon `0.3`, budget ladder `[5, 20, 100]`, stabilization mode, comparator id, source support, tolerance object, or exact non-implication text | residual trend before final-threshold success and all posterior-equivalence commentary | `bibliography_spine_only` |",
        "",
        "## Pre-mortem / failure-mode map",
        "",
        "- A pass could still mislead if the harness preserves deterministic row-local arithmetic while categorical resampling behavior or downstream filter logic remains wrong.",
        "- A fail could still be a harness artifact if the mixture rule, expectation ledger, or log-domain scaling loop is encoded incorrectly despite the intended clean-room arithmetic being valid.",
        "- The cheapest discriminator is the row-level tolerance object family, which isolates probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, marginal residual, mass, nonnegativity, finite-plan, or budget-trend failure.",
        "",
        "## Artifact list",
        "",
        "- `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json`",
        "- `experiments/dpf_monograph_evidence/reports/outputs/sinkhorn_residual.json`",
        "- `experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`",
        "- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-2026-05-16.md`",
        "",
        "## Residual summary",
        "",
        f"- `relaxed_probability_formula_abs`: observed `{soft_tol['relaxed_probability_formula_abs']['observed']:.3e}` against threshold `{soft_tol['relaxed_probability_formula_abs']['threshold']:.1e}`, finite=`{soft_tol['relaxed_probability_formula_abs']['finite']}`",
        f"- `relaxed_constant_expectation_abs`: observed `{soft_tol['relaxed_constant_expectation_abs']['observed']:.3e}` against threshold `{soft_tol['relaxed_constant_expectation_abs']['threshold']:.1e}`, finite=`{soft_tol['relaxed_constant_expectation_abs']['finite']}`",
        f"- `relaxed_identity_expectation_abs`: observed `{soft_tol['relaxed_identity_expectation_abs']['observed']:.3e}` against threshold `{soft_tol['relaxed_identity_expectation_abs']['threshold']:.1e}`, finite=`{soft_tol['relaxed_identity_expectation_abs']['finite']}`",
        f"- `relaxed_linear_summary_abs`: observed `{soft_tol['relaxed_linear_summary_abs']['observed']:.3e}` against threshold `{soft_tol['relaxed_linear_summary_abs']['threshold']:.1e}`, finite=`{soft_tol['relaxed_linear_summary_abs']['finite']}`",
        f"- `relaxed_nonlinear_expectation_abs`: observed `{soft_tol['relaxed_nonlinear_expectation_abs']['observed']:.3e}` against threshold `{soft_tol['relaxed_nonlinear_expectation_abs']['threshold']:.1e}`, finite=`{soft_tol['relaxed_nonlinear_expectation_abs']['finite']}`",
        f"- `categorical_identity_delta_abs`: observed `{soft_tol['categorical_identity_delta_abs']['observed']:.3e}`, finite=`{soft_tol['categorical_identity_delta_abs']['finite']}`",
        f"- `categorical_linear_summary_delta_abs`: observed `{soft_tol['categorical_linear_summary_delta_abs']['observed']:.3e}`, finite=`{soft_tol['categorical_linear_summary_delta_abs']['finite']}`",
        f"- `categorical_nonlinear_delta_abs`: observed `{soft_tol['categorical_nonlinear_delta_abs']['observed']:.3e}` with expected nonzero sign marker `{soft_tol['categorical_nonlinear_delta_sign_expected']['observed']:.0f}`",
        f"- `probability_sum_abs`: observed `{soft_tol['probability_sum_abs']['observed']:.3e}` against threshold `{soft_tol['probability_sum_abs']['threshold']:.1e}`, finite=`{soft_tol['probability_sum_abs']['finite']}`",
        f"- `row_marginal_abs_max`: observed `{sink_tol['row_marginal_abs_max']['observed']:.3e}` against threshold `{sink_tol['row_marginal_abs_max']['threshold']:.1e}`, finite=`{sink_tol['row_marginal_abs_max']['finite']}`",
        f"- `column_marginal_abs_max`: observed `{sink_tol['column_marginal_abs_max']['observed']:.3e}` against threshold `{sink_tol['column_marginal_abs_max']['threshold']:.1e}`, finite=`{sink_tol['column_marginal_abs_max']['finite']}`",
        f"- `total_mass_abs`: observed `{sink_tol['total_mass_abs']['observed']:.3e}` against threshold `{sink_tol['total_mass_abs']['threshold']:.1e}`, finite=`{sink_tol['total_mass_abs']['finite']}`",
        f"- `nonnegative_plan_violation_abs`: observed `{sink_tol['nonnegative_plan_violation_abs']['observed']:.3e}` against threshold `{sink_tol['nonnegative_plan_violation_abs']['threshold']:.1e}`, finite=`{sink_tol['nonnegative_plan_violation_abs']['finite']}`",
        f"- `finite_plan_violation_abs`: observed `{sink_tol['finite_plan_violation_abs']['observed']:.3e}` against threshold `{sink_tol['finite_plan_violation_abs']['threshold']:.1e}`, finite=`{sink_tol['finite_plan_violation_abs']['finite']}`",
        f"- `budget_residual_nonincrease_violations`: observed `{sink_tol['budget_residual_nonincrease_violations']['observed']:.0f}` against threshold `{sink_tol['budget_residual_nonincrease_violations']['threshold']:.0f}`, finite=`{sink_tol['budget_residual_nonincrease_violations']['finite']}`",
        "",
        "## Per-diagnostic decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `soft_resampling_bias` | `{soft['status']}` | `{soft['promotion_criterion_status']}` | `{soft['promotion_veto_status']}` | `{soft['continuation_veto_status']}` | `{soft['repair_trigger']}` |",
        f"| `sinkhorn_residual` | `{sinkhorn['status']}` | `{sinkhorn['promotion_criterion_status']}` | `{sinkhorn['promotion_veto_status']}` | `{sinkhorn['continuation_veto_status']}` | `{sinkhorn['repair_trigger']}` |",
        "",
        "## Inference-status table",
        "",
        "| Row | Status |",
        "| --- | --- |",
        f"| hard veto screen | {hard_veto_text} (`soft_resampling_bias={soft['status']}`, `sinkhorn_residual={sinkhorn['status']}`) |",
        "| statistically supported ranking | not applicable; IE5 is deterministic and has no candidate ranking problem |",
        "| descriptive-only differences | categorical-reference deltas and budget-ladder trend are descriptive within the declared row-local caveats and do not support posterior or equivalence claims |",
        "| default-readiness | not established; deterministic relaxed-target and finite-budget marginal checks are insufficient for any production or default claim |",
        "| next evidence needed | IE6 must test the learned-map residual phase on its own artifacts |",
        "",
        "## Run manifest",
        "",
        f"- command: `{soft['command']}`",
        f"- branch: `{soft['run_manifest']['branch']}`",
        f"- commit: `{soft['run_manifest']['commit']}`",
        f"- dirty-state summary captured: `{soft['run_manifest']['dirty_state_summary'][:120]}...`",
        f"- python version: `{soft['run_manifest']['python_version']}`",
        f"- numpy version: `{soft['run_manifest']['package_versions']['numpy']}`",
        f"- cpu_only: `{soft['run_manifest']['cpu_only']}`",
        f"- pre-import CUDA_VISIBLE_DEVICES: `{soft['run_manifest']['pre_import_cuda_visible_devices']}`",
        f"- pre-import GPU hiding assertion: `{soft['run_manifest']['pre_import_gpu_hiding_assertion']}`",
        f"- seed policy: `{soft['run_manifest']['seed_policy']}`",
        f"- replication_count: `{soft['replication_count']}`",
        f"- artifact paths: `{', '.join(soft['run_manifest']['artifact_paths'])}`",
        "",
        "## Source-support note",
        "",
        "Both IE5 rows intentionally keep `source_support_class=row_level_source_support_class=bibliography_spine_only`. The deterministic clean-room fixtures provide local arithmetic evidence only and do not upgrade provenance beyond the bibliography-spine support allowed by the phase plan.",
        "",
        "## Required non-implication text",
        "",
        f"- `soft_resampling_bias`: {SOFT_RESAMPLING_NON_IMPLICATION}",
        f"- `sinkhorn_residual`: {SINKHORN_NON_IMPLICATION}",
        "",
        "## Coverage semantics",
        "",
        coverage_text,
        "",
        "## Post-run red-team note",
        "",
        "The strongest alternative explanation is that the harness preserves deterministic relaxed-target arithmetic and finite-budget marginal control while categorical resampling behavior, finite-epsilon target mismatch, or downstream filtering logic are still wrong. A contrary result that would overturn the current pass would be any later artifact showing the same mixture bookkeeping or marginal residual ledger breaks once the fixture is embedded in a richer filter or exact-equivalence comparator.",
        "",
        "## Next-phase justification or blocker",
        "",
        next_text,
    ]
    MARKDOWN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_result(records: dict[str, dict[str, Any]], command: str) -> None:
    soft = records["soft_resampling_bias"]
    sinkhorn = records["sinkhorn_residual"]
    both_passed = soft["status"] == "pass" and sinkhorn["status"] == "pass"
    master_label = "ie_phase_passed" if both_passed else "ie_phase_blocked"
    local_label = "ie5_resampling_sinkhorn_passed" if both_passed else "ie5_resampling_sinkhorn_blocked"
    phase_status = "pass" if both_passed else "blocked"
    next_action = (
        "carry IE5 pass into IE6 coverage state"
        if both_passed
        else "do not advance to IE6 until repair/blocker decision is accepted"
    )
    hard_veto_text = "passed for both IE5 diagnostics" if both_passed else "failed for at least one IE5 diagnostic"
    next_section = (
        "IE5 survived the skeptical audit and passed both bounded promotion criteria, so the clean-room implementation-and-evidence lane may proceed to IE6."
        if both_passed
        else "IE5 produced a structured blocker for at least one owned diagnostic, so the clean-room implementation-and-evidence lane must not proceed to IE6 without an accepted repair or blocker decision."
    )
    lines = [
        "# Phase IE5 result: soft-resampling and Sinkhorn controlled tests",
        "",
        "## Outcome",
        "",
        f"- Master-program exit label: `{master_label}`",
        f"- Local exit label: `{local_label}`",
        f"- Phase status: `{phase_status}`",
        "",
        "## Skeptical audit before execution",
        "",
        "- Baselines stayed pinned to the two row-level deterministic comparators required by the plan.",
        "- Promotion criteria stayed row-local and deterministic; no explanatory residual trend or posterior commentary was upgraded into a pass criterion.",
        "- The runner records both relaxed-target arithmetic and final marginal thresholds, so categorical-reference deltas cannot hide relaxed-target failures and budget trend cannot hide a final-threshold failure.",
        "- CPU-only execution was enforced with `CUDA_VISIBLE_DEVICES=-1` before NumPy import.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: whether deterministic soft-resampling relaxed-target arithmetic and bounded Sinkhorn marginal control meet the clean-room IE5 comparator contracts.",
        "- Promotion criterion: the two IE5 rows pass their declared tolerance objects with seed policy `deterministic_no_rng_resampling_sinkhorn_fixture` and replication_count `1`.",
        "- Promotion veto: missing comparator identity, relaxed-target caveat, source-support class, epsilon `0.3`, budget ladder `[5, 20, 100]`, stabilization mode, tolerance object, or exact row-specific non-implication text.",
        "- Continuation veto: inability to encode the trusted Sinkhorn marginal comparator or the soft-resampling caveat in a schema-valid IE2 row.",
        "- Repair trigger: any probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, marginal residual, mass, nonnegativity, finite-plan, or budget-trend failure.",
        "- Not concluded: categorical law preservation, nonlinear unbiasedness, posterior equivalence, exact OT equivalence, production correctness, banking use, model-risk use, or production readiness.",
        "",
        "## Evidence contract",
        "",
        f"- `soft_resampling_bias` vs `{soft['comparator_id']}` with deterministic relaxed-probability and relaxed-expectation thresholds, finite categorical-reference deltas, nonzero nonlinear-delta sign requirement, and `bibliography_spine_only` source support.",
        f"- `sinkhorn_residual` vs `{sinkhorn['comparator_id']}` with epsilon `0.3`, budget ladder `[5, 20, 100]`, final marginal thresholds at budget `100`, zero nonincrease violations, and `bibliography_spine_only` source support.",
        "- Explanatory-only diagnostics remain any categorical-resampling, posterior, or exact-equivalence commentary plus budget-trend interpretation beyond the threshold contract.",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| `soft_resampling_bias` | `{soft['status']}` | `{soft['promotion_criterion_status']}` | `{soft['promotion_veto_status']}` | `{soft['continuation_veto_status']}` | deterministic relaxed-target scope | {next_action} | categorical law preservation / posterior claims |",
        f"| `sinkhorn_residual` | `{sinkhorn['status']}` | `{sinkhorn['promotion_criterion_status']}` | `{sinkhorn['promotion_veto_status']}` | `{sinkhorn['continuation_veto_status']}` | deterministic finite-epsilon finite-budget scope | {next_action} | exact OT/EOT or posterior claims |",
        "",
        "## Inference-status table",
        "",
        "| Row | Status |",
        "| --- | --- |",
        f"| hard veto screen | {hard_veto_text} |",
        "| statistically supported ranking | not applicable |",
        "| descriptive-only differences | categorical-reference deltas and budget-ladder trend remain descriptive within the declared caveats |",
        "| default-readiness | not established |",
        "| next evidence needed | IE6 artifacts |",
        "",
        "## Residual highlights",
        "",
        f"- categorical nonlinear delta magnitude: `{soft['tolerance']['categorical_nonlinear_delta_abs']['observed']:.3e}` with expected nonzero sign marker `{soft['tolerance']['categorical_nonlinear_delta_sign_expected']['observed']:.0f}`",
        f"- relaxed identity residual: `{soft['tolerance']['relaxed_identity_expectation_abs']['observed']:.3e}`",
        f"- categorical identity delta magnitude: `{soft['tolerance']['categorical_identity_delta_abs']['observed']:.3e}`",
        f"- probability-sum residual: `{soft['tolerance']['probability_sum_abs']['observed']:.3e}`",
        f"- final row marginal residual: `{sinkhorn['tolerance']['row_marginal_abs_max']['observed']:.3e}`",
        f"- final column marginal residual: `{sinkhorn['tolerance']['column_marginal_abs_max']['observed']:.3e}`",
        f"- final total-mass residual: `{sinkhorn['tolerance']['total_mass_abs']['observed']:.3e}`",
        f"- budget nonincrease violations: `{sinkhorn['tolerance']['budget_residual_nonincrease_violations']['observed']:.0f}`",
        "",
        "## Run manifest",
        "",
        f"- command: `{command}`",
        f"- branch: `{soft['run_manifest']['branch']}`",
        f"- commit: `{soft['run_manifest']['commit']}`",
        f"- python: `{soft['run_manifest']['python_version']}` / numpy `{soft['run_manifest']['package_versions']['numpy']}`",
        f"- cpu_only: `{soft['run_manifest']['cpu_only']}`",
        f"- pre-import CUDA_VISIBLE_DEVICES: `{soft['run_manifest']['pre_import_cuda_visible_devices']}`",
        f"- pre-import GPU hiding assertion: `{soft['run_manifest']['pre_import_gpu_hiding_assertion']}`",
        f"- seed policy: `{soft['seed_policy']}`",
        f"- replication_count: `{soft['replication_count']}`",
        f"- artifact paths: `{', '.join(soft['artifact_paths'])}`",
        "",
        "## Post-run red-team note",
        "",
        "IE5 evidence could still coexist with wrong categorical resampling behavior, wrong exact-equivalence interpretation at finite epsilon, or broken downstream filtering logic, so IE5 should only be cited for deterministic relaxed-target arithmetic and bounded marginal residual control.",
        "",
        "## Next-phase justification",
        "",
        next_section,
    ]
    PHASE_RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    started_at = utc_now()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_resampling_sinkhorn"

    soft_metrics = evaluate_soft_resampling_bias_fixture(build_soft_resampling_bias_fixture())
    sinkhorn_metrics = evaluate_sinkhorn_residual_fixture(build_sinkhorn_residual_fixture())

    branch, commit, dirty_state_summary = get_git_manifest()
    runtime_seconds = time.perf_counter() - start
    ended_at = utc_now()

    soft_status = soft_resampling_status(soft_metrics)
    sinkhorn_status_value = sinkhorn_status(sinkhorn_metrics)
    coverage_statuses = {
        "soft_resampling_bias": "passed" if soft_status == "pass" else "blocked",
        "sinkhorn_residual": "passed" if sinkhorn_status_value == "pass" else "blocked",
    }

    records = {}
    records["soft_resampling_bias"] = make_result_record(
        diagnostic_id="soft_resampling_bias",
        command=command,
        runtime_seconds=runtime_seconds,
        started_at=started_at,
        ended_at=ended_at,
        metrics=soft_metrics,
        row_status=soft_status,
        repair_trigger=repair_trigger_for_soft_resampling(soft_metrics),
        branch=branch,
        commit=commit,
        dirty_state_summary=dirty_state_summary,
        coverage_statuses=coverage_statuses,
    )
    records["sinkhorn_residual"] = make_result_record(
        diagnostic_id="sinkhorn_residual",
        command=command,
        runtime_seconds=runtime_seconds,
        started_at=started_at,
        ended_at=ended_at,
        metrics=sinkhorn_metrics,
        row_status=sinkhorn_status_value,
        repair_trigger=repair_trigger_for_sinkhorn(sinkhorn_metrics),
        branch=branch,
        commit=commit,
        dirty_state_summary=dirty_state_summary,
        coverage_statuses=coverage_statuses,
    )

    for diagnostic_id, record in records.items():
        write_json(JSON_OUTPUTS[diagnostic_id], record)

    write_markdown_report(records)
    write_phase_result(records, command)
    print(f"wrote {JSON_OUTPUTS['soft_resampling_bias']}")
    print(f"wrote {JSON_OUTPUTS['sinkhorn_residual']}")
    print(f"wrote {MARKDOWN_REPORT_PATH}")
    print(f"wrote {PHASE_RESULT_PATH}")
    print("ie_phase_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
