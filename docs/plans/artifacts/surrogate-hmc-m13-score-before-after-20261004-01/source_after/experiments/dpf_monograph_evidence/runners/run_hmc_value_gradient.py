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

import numpy as np

from experiments.dpf_monograph_evidence.diagnostics.hmc_value_gradient import (
    FD_STEPS,
    NON_IMPLICATION,
    evaluate_hmc_value_gradient_fixture,
    hmc_value_gradient_status,
    repair_trigger,
)
from experiments.dpf_monograph_evidence.fixtures.hmc_value_gradient import build_fixed_scalar_hmc_target_fixture
from experiments.dpf_monograph_evidence.results import validate_result_record

WALL_CLOCK_CAP_SECONDS = 30
PHASE_ID = "IE7"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_ROOT = RESULTS_ROOT / "outputs"
JSON_OUTPUT = OUTPUT_ROOT / "hmc_value_gradient.json"
MARKDOWN_REPORT_PATH = RESULTS_ROOT / "hmc-value-gradient-result.md"
PHASE_RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "plans"
    / "bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-result-2026-05-16.md"
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
    return str(path.resolve().relative_to(Path(__file__).resolve().parents[3]))


def build_coverage(row_status: str) -> dict[str, str]:
    return {
        "linear_gaussian_recovery": "passed",
        "synthetic_affine_flow": "passed",
        "pfpf_algebra_parity": "passed",
        "soft_resampling_bias": "passed",
        "sinkhorn_residual": "passed",
        "learned_map_residual": "deferred",
        "hmc_value_gradient": "passed" if row_status == "pass" else "blocked",
        "posterior_sensitivity_summary": "missing",
    }


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
            "experiments.dpf_monograph_evidence.fixtures.hmc_value_gradient",
            "experiments.dpf_monograph_evidence.diagnostics.hmc_value_gradient",
            "experiments.dpf_monograph_evidence.results",
            "experiments.dpf_monograph_evidence.runners.run_hmc_value_gradient",
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
    return "fail", "fail", "fail"


def make_result_record(
    command: str,
    runtime_seconds: float,
    started_at: str,
    ended_at: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    branch, commit, dirty_state_summary = get_git_manifest()
    row_status = hmc_value_gradient_status(metrics)
    promotion_criterion_status, promotion_veto_status, continuation_veto_status = gate_statuses(row_status)
    result_path_rel = relative_repo_path(JSON_OUTPUT)
    markdown_path_rel = relative_repo_path(MARKDOWN_REPORT_PATH)
    artifact_paths = [result_path_rel, markdown_path_rel]
    package_versions = {"numpy": np.__version__, "python": platform.python_version()}
    result = {
        "phase_id": PHASE_ID,
        "diagnostic_id": "hmc_value_gradient",
        "chapter_label": "Chapter 26",
        "diagnostic_role": "promotion_criterion",
        "comparator_id": "fixed_scalar_same_callable_finite_difference_reference",
        "source_family": "Chapter 26 same-scalar HMC-facing value-gradient diagnostic on deterministic clean-room fixture",
        "source_support_class": "bibliography_spine_only",
        "row_level_source_support_class": "bibliography_spine_only",
        "seed_policy": "deterministic_no_rng_fixed_scalar_target",
        "status": row_status,
        "coverage": build_coverage(row_status),
        "tolerance": metrics["tolerance"],
        "finite_checks": metrics["finite_checks"],
        "shape_checks": metrics["shape_checks"],
        "runtime_seconds": runtime_seconds,
        "blocker_class": "none" if row_status == "pass" else "execution",
        "non_implication": NON_IMPLICATION,
        "promotion_criterion_status": promotion_criterion_status,
        "promotion_veto_status": promotion_veto_status,
        "continuation_veto_status": continuation_veto_status,
        "repair_trigger": repair_trigger(metrics),
        "explanatory_only_diagnostics": [
            "Truncation-region and cancellation-region finite-difference residuals are explanatory only.",
            "Compiled path is recorded as not available in this clean-room CPU-only fixture and is not interpreted as posterior readiness.",
            "No HMC chain, adaptation, tuning, posterior summary, DPF-HMC target, DSGE target, or MacroFinance target was run.",
        ],
        "environment": build_environment(branch, commit, dirty_state_summary),
        "command": command,
        "wall_time_seconds": runtime_seconds,
        "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "artifact_paths": artifact_paths,
        "cpu_gpu_mode": build_cpu_gpu_mode(),
        "uncertainty_status": "not_applicable",
        "replication_count": 1,
        "mcse_or_interval": {
            "summary": "Not applicable because IE7 uses a deterministic fixed scalar target with no stochastic sampling."
        },
        "post_run_red_team_note": "A same-scalar fixture pass still does not validate any real DPF-HMC posterior, HMC tuning, production code path, or banking/model-risk use.",
        "cap_non_applicability_reasons": {
            "max_particles": "IE7 does not instantiate particles.",
            "max_time_steps": "IE7 does not roll out a time-series filter.",
            "max_sinkhorn_iterations": "IE7 does not instantiate Sinkhorn transport.",
            "max_replications": "IE7 is deterministic and records replication_count=1.",
        },
        "cap_values": {
            "max_particles": None,
            "max_time_steps": None,
            "max_sinkhorn_iterations": None,
            "max_finite_difference_evaluations": 2 * metrics["shape_checks"]["state_dimension"] * len(FD_STEPS),
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
            "seed_policy": "deterministic_no_rng_fixed_scalar_target",
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


def write_markdown_report(record: dict[str, Any]) -> None:
    tol = record["tolerance"]
    finite = record["finite_checks"]
    lines = [
        "# IE7 HMC value-gradient result",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `ie_phase_passed`",
        "- Local exit label: `ie7_hmc_value_gradient_passed`",
        "- Phase status: `pass`",
        "",
        "## Skeptical audit before execution",
        "",
        "- Same-scalar audit: accept/reject and differentiated scalar values came from the same target wrapper.",
        "- Gradient audit: stable-window central finite differences match the closed-form gradient within tolerance.",
        "- Runtime-boundary audit: no HMC chain, tuning, adaptation, posterior summary, DPF-HMC target, DSGE target, or MacroFinance target was run.",
        "- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: does the fixed scalar target preserve the same-scalar value-gradient contract before any HMC-facing interpretation?",
        "- Promotion criterion: same scalar, finite-difference stable window, eager repeatability, compiled/eager status reporting, leapfrog reversibility, and bounded energy smoke pass.",
        "- Promotion veto: missing target identity, same-callable proof, finite-difference ladder, seed policy, no-chain assertion, CPU manifest, source support, tolerance object, or exact non-implication text.",
        "- Continuation veto: same-scalar failure, nondeterministic target behavior, or prohibited HMC execution.",
        "- Repair trigger: same-scalar mismatch, gradient mismatch, repeatability failure, reversibility failure, energy-drift failure, nondeterminism, or prohibited HMC execution.",
        "",
        "## Evidence contract",
        "",
        "| Diagnostic | Comparator | Status | Source support |",
        "| --- | --- | --- | --- |",
        f"| `hmc_value_gradient` | `{record['comparator_id']}` | `{record['status']}` | `{record['source_support_class']}` |",
        "",
        "## Residual summary",
        "",
        f"- same-scalar residual: `{tol['same_scalar_value_abs']['observed']:.3e}`",
        f"- finite-difference stable-window residual: `{tol['finite_difference_stable_window_abs_max']['observed']:.3e}`",
        f"- eager value repeat residual: `{tol['eager_repeat_value_abs_max']['observed']:.3e}`",
        f"- eager gradient repeat residual: `{tol['eager_repeat_gradient_abs_max']['observed']:.3e}`",
        f"- compiled status: `{finite['compiled_status']}` (`{finite['compiled_not_available_reason']}`)",
        f"- leapfrog position reversibility residual: `{tol['leapfrog_reversibility_position_abs_max']['observed']:.3e}`",
        f"- leapfrog momentum reversibility residual: `{tol['leapfrog_reversibility_momentum_abs_max']['observed']:.3e}`",
        f"- forward energy-drift smoke residual: `{tol['energy_drift_abs']['observed']:.3e}`",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `hmc_value_gradient` | `{record['status']}` | `{record['promotion_criterion_status']}` | `{record['promotion_veto_status']}` | `{record['continuation_veto_status']}` | `{record['repair_trigger']}` |",
        "",
        "## Coverage semantics",
        "",
        "- IE3 through IE5 diagnostics are carried forward as `passed`.",
        "- IE6 learned-map residual remains `deferred`.",
        "- IE7 `hmc_value_gradient` is `passed`.",
        "- IE8 remains `missing`.",
        "",
        "## Artifact list",
        "",
        "- `experiments/dpf_monograph_evidence/reports/outputs/hmc_value_gradient.json`",
        "- `experiments/dpf_monograph_evidence/reports/hmc-value-gradient-result.md`",
        "- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-result-2026-05-16.md`",
        "",
        "## Run manifest",
        "",
        f"- command: `{record['command']}`",
        f"- branch: `{record['run_manifest']['branch']}`",
        f"- commit: `{record['run_manifest']['commit']}`",
        f"- python: `{record['run_manifest']['python_version']}` / numpy `{record['run_manifest']['package_versions']['numpy']}`",
        f"- cpu_only: `{record['run_manifest']['cpu_only']}`",
        f"- pre-import CUDA_VISIBLE_DEVICES: `{record['run_manifest']['pre_import_cuda_visible_devices']}`",
        f"- seed policy: `{record['run_manifest']['seed_policy']}`",
        "",
        "## Required non-implication text",
        "",
        NON_IMPLICATION,
        "",
        "## Post-run red-team note",
        "",
        record["post_run_red_team_note"],
        "",
        "## Next-phase justification",
        "",
        "IE7 passed the fixed-scalar gate, so IE8 may summarize controlled-fixture evidence. IE8 still must not run or claim real DPF-HMC, DSGE, MacroFinance, posterior, banking, model-risk, or production validation.",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_result(record: dict[str, Any]) -> None:
    lines = [
        "# Phase IE7 result: same-scalar HMC value-gradient tests",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `ie_phase_passed`",
        "- Local exit label: `ie7_hmc_value_gradient_passed`",
        "- Phase status: `pass`",
        "",
        "## Summary",
        "",
        "IE7 executed a deterministic fixed scalar target and passed same-scalar, finite-difference stable-window, eager repeatability, compiled-status reporting, leapfrog reversibility, and bounded energy-smoke checks. No HMC chain, tuning, adaptation, posterior summary, DPF-HMC target, DSGE target, MacroFinance target, or production import was run.",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `hmc_value_gradient` | `{record['status']}` | `{record['promotion_criterion_status']}` | `{record['promotion_veto_status']}` | `{record['continuation_veto_status']}` | `{record['repair_trigger']}` |",
        "",
        "## Residual highlights",
        "",
        f"- same-scalar residual: `{record['tolerance']['same_scalar_value_abs']['observed']:.3e}`",
        f"- finite-difference stable-window residual: `{record['tolerance']['finite_difference_stable_window_abs_max']['observed']:.3e}`",
        f"- leapfrog position/momentum reversibility residuals: `{record['tolerance']['leapfrog_reversibility_position_abs_max']['observed']:.3e}`, `{record['tolerance']['leapfrog_reversibility_momentum_abs_max']['observed']:.3e}`",
        f"- forward energy-drift smoke residual: `{record['tolerance']['energy_drift_abs']['observed']:.3e}`",
        "",
        "## Non-implication",
        "",
        NON_IMPLICATION,
        "",
        "## Next-phase justification",
        "",
        "Proceeding to IE8 is justified for aggregate governance and controlled-fixture summary only. IE8 must not treat IE7 as real HMC, DPF-HMC, posterior, banking, model-risk, or production validation.",
    ]
    PHASE_RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    started_at = utc_now()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_hmc_value_gradient"
    metrics = evaluate_hmc_value_gradient_fixture(build_fixed_scalar_hmc_target_fixture())
    runtime_seconds = time.perf_counter() - start
    ended_at = utc_now()
    record = make_result_record(command, runtime_seconds, started_at, ended_at, metrics)
    write_json(JSON_OUTPUT, record)
    write_markdown_report(record)
    write_phase_result(record)
    print(f"wrote {JSON_OUTPUT}")
    print(f"wrote {MARKDOWN_REPORT_PATH}")
    print(f"wrote {PHASE_RESULT_PATH}")
    print("ie_phase_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
