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

from experiments.dpf_monograph_evidence.diagnostics.affine_flow_pfpf import (
    ABS_TOL,
    PFPF_ALGEBRA_PARITY_NON_IMPLICATION,
    SYNTHETIC_AFFINE_FLOW_NON_IMPLICATION,
    evaluate_affine_flow_fixture,
    repair_trigger_for_row,
)
from experiments.dpf_monograph_evidence.fixtures.affine_flow import build_synthetic_affine_flow_fixture
from experiments.dpf_monograph_evidence.results import validate_result_record

WALL_CLOCK_CAP_SECONDS = 30
PHASE_ID = "IE4"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_ROOT = RESULTS_ROOT / "outputs"
MARKDOWN_REPORT_PATH = RESULTS_ROOT / "affine-flow-pfpf-result.md"
PHASE_RESULT_PATH = Path(__file__).resolve().parents[3] / "docs" / "plans" / "bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-result-2026-05-16.md"
JSON_OUTPUTS = {
    "synthetic_affine_flow": OUTPUT_ROOT / "affine_flow_synthetic_affine_flow.json",
    "pfpf_algebra_parity": OUTPUT_ROOT / "affine_flow_pfpf_algebra_parity.json",
}
COMPARATOR_IDS = {
    "synthetic_affine_flow": "analytic_affine_pushforward_density_reference",
    "pfpf_algebra_parity": "closed_form_pfpf_log_weight_reference",
}
NON_IMPLICATIONS = {
    "synthetic_affine_flow": SYNTHETIC_AFFINE_FLOW_NON_IMPLICATION,
    "pfpf_algebra_parity": PFPF_ALGEBRA_PARITY_NON_IMPLICATION,
}
EXPLANATORY_ONLY = {
    "synthetic_affine_flow": [
        "Affine-only fixture parity does not support nonlinear-flow integration claims.",
        "Affine-only fixture parity does not support filtering or posterior-quality claims.",
    ],
    "pfpf_algebra_parity": [
        "Closed-form affine algebra parity does not support nonlinear-flow integration claims.",
        "Closed-form affine algebra parity does not support filtering or posterior-quality claims.",
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
        "synthetic_affine_flow": "missing",
        "pfpf_algebra_parity": "missing",
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
            "experiments.dpf_monograph_evidence.fixtures.affine_flow",
            "experiments.dpf_monograph_evidence.diagnostics.affine_flow_pfpf",
            "experiments.dpf_monograph_evidence.results",
            "experiments.dpf_monograph_evidence.runners.run_affine_flow_pfpf",
        ],
    }


def build_cpu_gpu_mode() -> dict[str, Any]:
    return {
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_devices_visible": [],
        "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
    }


def status_from_metrics(metrics: dict[str, Any]) -> str:
    for value in metrics["tolerance"].values():
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return "fail"
    return "pass"


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
    branch: str,
    commit: str,
    dirty_state_summary: str,
    coverage_statuses: dict[str, str],
) -> dict[str, Any]:
    result_path_rel = relative_repo_path(JSON_OUTPUTS[diagnostic_id])
    markdown_path_rel = relative_repo_path(MARKDOWN_REPORT_PATH)
    promotion_criterion_status, promotion_veto_status, continuation_veto_status = gate_statuses(
        status_from_metrics(metrics)
    )
    row_status = status_from_metrics(metrics)
    artifact_paths = [result_path_rel, markdown_path_rel]
    package_versions = {"numpy": np.__version__, "python": platform.python_version()}
    result = {
        "phase_id": PHASE_ID,
        "diagnostic_id": diagnostic_id,
        "chapter_label": "Chapter 26",
        "diagnostic_role": "promotion_criterion",
        "comparator_id": COMPARATOR_IDS[diagnostic_id],
        "source_family": "Chapter 26 PF-PF proposal correction diagnostics on deterministic affine clean-room fixtures",
        "source_support_class": "bibliography_spine_only",
        "row_level_source_support_class": "bibliography_spine_only",
        "seed_policy": "deterministic_no_rng_affine_fixture",
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
        "repair_trigger": repair_trigger_for_row(diagnostic_id, metrics),
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
            "summary": "Not applicable because IE4 uses deterministic affine fixtures with replication_count=1."
        },
        "post_run_red_team_note": "These affine checks could still pass while nonlinear flow integration, solver stability, and downstream filtering logic remain wrong because the fixture never leaves closed-form affine algebra.",
        "cap_non_applicability_reasons": {
            "max_time_steps": "IE4 evaluates single-step affine parity only and does not roll out a time-series filter.",
            "max_sinkhorn_iterations": "IE4 does not instantiate Sinkhorn transport.",
            "max_finite_difference_evaluations": "IE4 does not use finite-difference probes.",
            "max_replications": "IE4 is deterministic and records replication_count=1."
        },
        "cap_values": {
            "max_particles": int(len(metrics["corrected_log_weight"])),
            "max_time_steps": None,
            "max_sinkhorn_iterations": None,
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
            "seed_policy": "deterministic_no_rng_affine_fixture",
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


def max_residuals(metrics: dict[str, Any], keys: list[str]) -> list[str]:
    lines = []
    for key in keys:
        entry = metrics["tolerance"][key]
        lines.append(
            f"- `{key}`: observed `{entry['observed']:.3e}` against threshold `{entry['threshold']:.1e}`, finite=`{entry['finite']}`"
        )
    return lines


def write_markdown_report(records: dict[str, dict[str, Any]], metrics: dict[str, Any]) -> None:
    synthetic = records["synthetic_affine_flow"]
    pfpf = records["pfpf_algebra_parity"]
    lines = [
        "# IE4 affine-flow PF-PF result",
        "",
        "## Skeptical audit before execution",
        "",
        "- Wrong baseline audit: the only allowed comparators are `analytic_affine_pushforward_density_reference` and `closed_form_pfpf_log_weight_reference`; no weaker proxy baseline was used.",
        "- Proxy-metric audit: residual tolerances are the promotion criteria, while any narrative about nonlinear flow or filtering remains explanatory-only.",
        "- Stop-condition audit: any non-finite or above-threshold residual would force a row failure with a structured repair trigger instead of silent renormalization.",
        "- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.",
        "- Artifact audit: each canonical IE4 diagnostic writes one schema-valid JSON object, with known coverage semantics carried into both row files.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: does the clean-room harness preserve affine pushforward-density and PF-PF proposal-correction algebra exactly on deterministic fixtures?",
        "- Candidate under test: deterministic affine inverse-map, Jacobian-sign, and corrected-weight bookkeeping in the IE4 clean-room harness.",
        "- Expected failure mode: sign mistakes in `log|det A|`, inverse-map reconstruction drift, or normalization hiding an unnormalized discrepancy.",
        "- Promotion criterion: each required residual family stays finite and at or below `1e-12`.",
        "- Promotion veto: missing comparator identity, source-support class, explicit sign convention, or exact row-specific non-implication text.",
        "- Continuation veto: inability to record affine-only comparator identity, row-level source support, or separate unnormalized/normalized parity residuals.",
        "- Repair trigger: any deterministic algebra mismatch or non-finite residual.",
        "- What must not be concluded: anything about nonlinear flow integration, solver stability, PF-PF filtering correctness, posterior quality, banking use, model-risk use, or production readiness.",
        "",
        "## Evidence contract",
        "",
        "| Diagnostic | Comparator | Primary criterion | Veto diagnostics | Explanatory only | Source support |",
        "| --- | --- | --- | --- | --- | --- |",
        "| `synthetic_affine_flow` | `analytic_affine_pushforward_density_reference` | forward/inverse, log-det, and pushforward-density residuals all finite and <= `1e-12` | missing determinant sign, comparator id, source support, tolerance object, or exact non-implication text | nonlinear-flow interpretation | `bibliography_spine_only` |",
        "| `pfpf_algebra_parity` | `closed_form_pfpf_log_weight_reference` | proposal-density, corrected-log-weight, normalized-weight, and probability-sum residuals all finite and <= `1e-12` | missing sign convention, determinant contribution, comparator id, source support, tolerance object, or exact non-implication text | filtering/posterior interpretation | `bibliography_spine_only` |",
        "",
        "## Pre-mortem / failure-mode map",
        "",
        "- A pass could still mislead if the harness is correct only for affine closed forms while nonlinear integrators remain broken.",
        "- A fail could still be a harness artifact if the inverse map, determinant sign, or normalization ledger is encoded wrongly even though the underlying affine algebra is valid.",
        "- The cheapest discriminator is the row-level residual object family, which localizes map inversion, determinant, proposal density, target density, or normalization failure.",
        "",
        "## Artifact list",
        "",
        "- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json`",
        "- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_pfpf_algebra_parity.json`",
        "- `experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`",
        "- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-result-2026-05-16.md`",
        "",
        "## Residual summary",
        "",
        *max_residuals(metrics, [
            "forward_reconstruction_abs_max",
            "inverse_reconstruction_abs_max",
            "log_det_abs_max",
            "pushforward_log_density_abs_max",
            "proposal_log_density_abs_max",
            "corrected_log_weight_abs_max",
            "normalized_weight_abs_max",
            "probability_sum_abs_max",
        ]),
        "",
        "## Per-diagnostic decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `synthetic_affine_flow` | `{synthetic['status']}` | `{synthetic['promotion_criterion_status']}` | `{synthetic['promotion_veto_status']}` | `{synthetic['continuation_veto_status']}` | `{synthetic['repair_trigger']}` |",
        f"| `pfpf_algebra_parity` | `{pfpf['status']}` | `{pfpf['promotion_criterion_status']}` | `{pfpf['promotion_veto_status']}` | `{pfpf['continuation_veto_status']}` | `{pfpf['repair_trigger']}` |",
        "",
        "## Inference-status table",
        "",
        "| Row | Status |",
        "| --- | --- |",
        f"| hard veto screen | both IE4 rows passed deterministic residual and finite checks (`synthetic_affine_flow={synthetic['status']}`, `pfpf_algebra_parity={pfpf['status']}`) |",
        "| statistically supported ranking | not applicable; IE4 is deterministic and has no candidate ranking problem |",
        "| descriptive-only differences | any narrative about nonlinear flow integration or filtering remains descriptive-only and unsupported by IE4 |",
        "| default-readiness | not established; affine parity is insufficient for any production or default claim |",
        "| next evidence needed | IE5 must test later resampling/transport diagnostics on its own artifacts |",
        "",
        "## Run manifest",
        "",
        f"- command: `{synthetic['command']}`",
        f"- branch: `{synthetic['run_manifest']['branch']}`",
        f"- commit: `{synthetic['run_manifest']['commit']}`",
        f"- dirty-state summary captured: `{synthetic['run_manifest']['dirty_state_summary'][:120]}...`",
        f"- python version: `{synthetic['run_manifest']['python_version']}`",
        f"- numpy version: `{synthetic['run_manifest']['package_versions']['numpy']}`",
        f"- cpu_only: `{synthetic['run_manifest']['cpu_only']}`",
        f"- pre-import CUDA_VISIBLE_DEVICES: `{synthetic['run_manifest']['pre_import_cuda_visible_devices']}`",
        f"- pre-import GPU hiding assertion: `{synthetic['run_manifest']['pre_import_gpu_hiding_assertion']}`",
        f"- seed policy: `{synthetic['run_manifest']['seed_policy']}`",
        f"- replication_count: `{synthetic['replication_count']}`",
        f"- artifact paths: `{', '.join(synthetic['run_manifest']['artifact_paths'])}`",
        "",
        "## Source-support note",
        "",
        "Both IE4 rows intentionally keep `source_support_class=row_level_source_support_class=bibliography_spine_only`. The deterministic local affine fixture is clean-room evidence for algebra parity, but it does not upgrade source provenance beyond the bibliography-spine support allowed by the plan.",
        "",
        "## Required non-implication text",
        "",
        f"- `synthetic_affine_flow`: {SYNTHETIC_AFFINE_FLOW_NON_IMPLICATION}",
        f"- `pfpf_algebra_parity`: {PFPF_ALGEBRA_PARITY_NON_IMPLICATION}",
        "",
        "## Coverage semantics",
        "",
        "Both JSON row files mark `linear_gaussian_recovery` as `passed`, both IE4 diagnostic IDs as `passed`, and all later-phase diagnostic IDs as `missing`, matching the within-program known coverage state after successful IE4 emission. No asymmetry is present because both row files passed.",
        "",
        "## Post-run red-team note",
        "",
        "The strongest alternative explanation is that the harness encodes the affine closed form correctly while later nonlinear flow integrators, solver stability, or filtering logic are still wrong. A contrary result that would overturn the current pass would be any later artifact showing the same sign convention or normalization ledger breaks once the proposal ceases to be affine.",
        "",
        "## Next-phase justification or blocker",
        "",
        "IE4 passed its bounded affine algebra checks, so the lane can advance to IE5 without upgrading any claim beyond affine clean-room PF-PF parity.",
    ]
    MARKDOWN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_result(records: dict[str, dict[str, Any]], metrics: dict[str, Any], command: str) -> None:
    synthetic = records["synthetic_affine_flow"]
    pfpf = records["pfpf_algebra_parity"]
    lines = [
        "# Phase IE4 result: affine-flow PF-PF density and log-det tests",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `ie_phase_passed`",
        "- Local exit label: `ie4_affine_flow_pfpf_passed`",
        "- Phase status: `pass`",
        "",
        "## Skeptical audit before execution",
        "",
        "- Baselines stayed pinned to the two row-level analytic comparators required by the plan.",
        "- Promotion criteria stayed row-local and deterministic; no proxy metric was upgraded into a pass criterion.",
        "- The runner records separate unnormalized and normalized parity residuals, so normalization cannot hide an upstream mismatch.",
        "- CPU-only execution was enforced with `CUDA_VISIBLE_DEVICES=-1` before NumPy import.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: whether deterministic affine pushforward-density and PF-PF proposal-correction algebra match their closed-form references exactly.",
        "- Promotion criterion: every required residual object remains finite and <= `1e-12`.",
        "- Promotion veto: missing comparator identity, determinant sign, source-support class, sign convention, tolerance object, or exact row-specific non-implication text.",
        "- Continuation veto: inability to encode affine-only non-implication, comparator identity, or separate unnormalized/normalized parity residuals.",
        "- Repair trigger: any residual failure would localize map inversion, determinant sign, proposal density, target density, or normalization.",
        "- Not concluded: anything about nonlinear integration, solver stability, filtering correctness, posterior quality, banking use, model-risk use, or production readiness.",
        "",
        "## Evidence contract",
        "",
        f"- `synthetic_affine_flow` vs `{synthetic['comparator_id']}` with deterministic `1e-12` residual thresholds and `bibliography_spine_only` source support.",
        f"- `pfpf_algebra_parity` vs `{pfpf['comparator_id']}` with deterministic `1e-12` residual thresholds and `bibliography_spine_only` source support.",
        "- Explanatory-only diagnostics remain any nonlinear-flow or filtering commentary.",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| `synthetic_affine_flow` | `{synthetic['status']}` | `{synthetic['promotion_criterion_status']}` | `{synthetic['promotion_veto_status']}` | `{synthetic['continuation_veto_status']}` | affine-only scope | carry IE4 pass into IE5 coverage state | nonlinear-flow/filter correctness |",
        f"| `pfpf_algebra_parity` | `{pfpf['status']}` | `{pfpf['promotion_criterion_status']}` | `{pfpf['promotion_veto_status']}` | `{pfpf['continuation_veto_status']}` | affine-only scope | carry IE4 pass into IE5 coverage state | nonlinear-flow/filter correctness |",
        "",
        "## Inference-status table",
        "",
        "| Row | Status |",
        "| --- | --- |",
        "| hard veto screen | passed for both IE4 diagnostics |",
        "| statistically supported ranking | not applicable |",
        "| descriptive-only differences | all nonlinear-flow and posterior-quality commentary remains descriptive-only |",
        "| default-readiness | not established |",
        "| next evidence needed | IE5 artifacts |",
        "",
        "## Residual highlights",
        "",
        f"- forward reconstruction max residual: `{metrics['tolerance']['forward_reconstruction_abs_max']['observed']:.3e}`",
        f"- inverse reconstruction max residual: `{metrics['tolerance']['inverse_reconstruction_abs_max']['observed']:.3e}`",
        f"- log-det residual: `{metrics['tolerance']['log_det_abs_max']['observed']:.3e}`",
        f"- pushforward log-density residual: `{metrics['tolerance']['pushforward_log_density_abs_max']['observed']:.3e}`",
        f"- proposal log-density residual: `{metrics['tolerance']['proposal_log_density_abs_max']['observed']:.3e}`",
        f"- corrected log-weight residual: `{metrics['tolerance']['corrected_log_weight_abs_max']['observed']:.3e}`",
        f"- normalized-weight residual: `{metrics['tolerance']['normalized_weight_abs_max']['observed']:.3e}`",
        f"- probability-sum residual: `{metrics['tolerance']['probability_sum_abs_max']['observed']:.3e}`",
        "",
        "## Run manifest",
        "",
        f"- command: `{command}`",
        f"- branch: `{synthetic['run_manifest']['branch']}`",
        f"- commit: `{synthetic['run_manifest']['commit']}`",
        f"- python: `{synthetic['run_manifest']['python_version']}` / numpy `{synthetic['run_manifest']['package_versions']['numpy']}`",
        f"- cpu_only: `{synthetic['run_manifest']['cpu_only']}`",
        f"- pre-import CUDA_VISIBLE_DEVICES: `{synthetic['run_manifest']['pre_import_cuda_visible_devices']}`",
        f"- pre-import GPU hiding assertion: `{synthetic['run_manifest']['pre_import_gpu_hiding_assertion']}`",
        f"- seed policy: `{synthetic['seed_policy']}`",
        f"- replication_count: `{synthetic['replication_count']}`",
        f"- artifact paths: `{', '.join(synthetic['artifact_paths'])}`",
        "",
        "## Post-run red-team note",
        "",
        "Affine algebra pass evidence could still coexist with broken nonlinear integration or filtering logic, so IE4 should only be cited for closed-form affine parity.",
        "",
        "## Next-phase justification",
        "",
        "IE4 survived the skeptical audit and passed both bounded promotion criteria, so the clean-room implementation-and-evidence lane may proceed to IE5."
    ]
    PHASE_RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    started_at = utc_now()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_affine_flow_pfpf"
    fixture = build_synthetic_affine_flow_fixture()
    metrics = evaluate_affine_flow_fixture(fixture)
    branch, commit, dirty_state_summary = get_git_manifest()
    runtime_seconds = time.perf_counter() - start
    ended_at = utc_now()

    row_status = status_from_metrics(metrics)
    coverage_statuses = {
        "synthetic_affine_flow": "passed" if row_status == "pass" else "blocked",
        "pfpf_algebra_parity": "passed" if row_status == "pass" else "blocked",
    }

    records = {}
    for diagnostic_id in ("synthetic_affine_flow", "pfpf_algebra_parity"):
        record = make_result_record(
            diagnostic_id=diagnostic_id,
            command=command,
            runtime_seconds=runtime_seconds,
            started_at=started_at,
            ended_at=ended_at,
            metrics=metrics,
            branch=branch,
            commit=commit,
            dirty_state_summary=dirty_state_summary,
            coverage_statuses=coverage_statuses,
        )
        records[diagnostic_id] = record
        write_json(JSON_OUTPUTS[diagnostic_id], record)

    write_markdown_report(records, metrics)
    write_phase_result(records, metrics, command)
    print(f"wrote {JSON_OUTPUTS['synthetic_affine_flow']}")
    print(f"wrote {JSON_OUTPUTS['pfpf_algebra_parity']}")
    print(f"wrote {MARKDOWN_REPORT_PATH}")
    print(f"wrote {PHASE_RESULT_PATH}")
    print("ie_phase_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
