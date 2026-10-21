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

from experiments.dpf_monograph_evidence.results import validate_result_record

WALL_CLOCK_CAP_SECONDS = 30
PHASE_ID = "IE6"
RESULTS_ROOT = Path(__file__).resolve().parent.parent / "reports"
OUTPUT_ROOT = RESULTS_ROOT / "outputs"
JSON_OUTPUT = OUTPUT_ROOT / "learned_ot_residual.json"
MARKDOWN_REPORT_PATH = RESULTS_ROOT / "learned-ot-residual-result.md"
PHASE_RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "plans"
    / "bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-result-2026-05-16.md"
)

NON_IMPLICATION = (
    "IE6 learned-OT diagnostics were deferred because no approved pre-existing "
    "teacher/student artifact with provenance was available. This deferral does not "
    "validate or invalidate learned OT, neural OT, surrogate-map quality, posterior "
    "quality, production bayesfilter code, banking use, model-risk use, or "
    "production readiness."
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


def build_coverage() -> dict[str, str]:
    return {
        "linear_gaussian_recovery": "passed",
        "synthetic_affine_flow": "passed",
        "pfpf_algebra_parity": "passed",
        "soft_resampling_bias": "passed",
        "sinkhorn_residual": "passed",
        "learned_map_residual": "deferred",
        "hmc_value_gradient": "missing",
        "posterior_sensitivity_summary": "missing",
    }


def build_environment(branch: str, commit: str, dirty_state_summary: str) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "package_versions": {"python": platform.python_version(), "stdlib": "builtin"},
        "branch": branch,
        "commit": commit,
        "dirty_state_summary": dirty_state_summary,
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
        "import_boundary_checked_modules": [
            "experiments.dpf_monograph_evidence.results",
            "experiments.dpf_monograph_evidence.runners.run_learned_ot_residual",
        ],
    }


def build_cpu_gpu_mode() -> dict[str, Any]:
    return {
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_devices_visible": [],
        "gpu_hidden_before_import": _PRE_IMPORT_GPU_HIDING_ASSERTION,
    }


def make_result_record(command: str, runtime_seconds: float, started_at: str, ended_at: str) -> dict[str, Any]:
    branch, commit, dirty_state_summary = get_git_manifest()
    result_path_rel = relative_repo_path(JSON_OUTPUT)
    markdown_path_rel = relative_repo_path(MARKDOWN_REPORT_PATH)
    artifact_paths = [result_path_rel, markdown_path_rel]
    package_versions = {"python": platform.python_version(), "stdlib": "builtin"}
    result = {
        "phase_id": PHASE_ID,
        "diagnostic_id": "learned_map_residual",
        "chapter_label": "Chapter 26",
        "diagnostic_role": "continuation_veto",
        "comparator_id": "no_approved_teacher_student_artifact",
        "source_family": "Chapter 26 learned/amortized OT diagnostics; no approved teacher/student artifact available",
        "source_support_class": "bibliography_spine_only",
        "row_level_source_support_class": "bibliography_spine_only",
        "seed_policy": "deterministic_no_rng_deferred_no_artifact",
        "status": "deferred",
        "coverage": build_coverage(),
        "tolerance": {
            "deferred_no_artifact": {
                "threshold": 0.0,
                "observed": 0.0,
                "finite": True,
            }
        },
        "finite_checks": {
            "artifact_gate": {
                "teacher_artifact_path_or_id": None,
                "approval_source": None,
                "approval_date": None,
                "teacher_variant": "unknown",
                "teacher_output_object": "unknown",
                "teacher_origin_class": "unknown_or_missing",
                "student_checkpoint_id": None,
                "training_run_id": None,
                "training_commit": None,
                "training_data_envelope": None,
                "artifact_mode": "missing_artifact_deferred",
                "optimizer_step_count": 0,
                "checkpoint_mutation_flag": 0,
                "artifact_provenance_missing_flag": 1,
            },
            "envelope_tags": [
                {
                    "envelope_tag": "deferred",
                    "shift_axis": "none",
                    "training_envelope_id": "not_applicable_no_artifact",
                    "evaluation_envelope_id": "not_applicable_no_artifact",
                    "reason": "No approved teacher/student artifact with provenance was available.",
                }
            ],
            "residuals_executed": False,
            "finite_summary": True,
        },
        "shape_checks": {
            "map_input_shape": "not_applicable_no_artifact",
            "map_output_shape": "not_applicable_no_artifact",
            "test_point_count": 0,
            "support_size": 0,
            "residual_vector_shape": [0],
            "envelope_slice_count": 0,
        },
        "runtime_seconds": runtime_seconds,
        "blocker_class": "source",
        "non_implication": NON_IMPLICATION,
        "promotion_criterion_status": "not_triggered",
        "promotion_veto_status": "not_triggered",
        "continuation_veto_status": "fail",
        "repair_trigger": "missing artifact provenance",
        "explanatory_only_diagnostics": [
            "Deferred learned-OT residual row records an evidence gap only.",
            "No teacher quality, student quality, posterior quality, or method failure claim is supported.",
        ],
        "environment": build_environment(branch, commit, dirty_state_summary),
        "command": command,
        "wall_time_seconds": runtime_seconds,
        "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "artifact_paths": artifact_paths,
        "cpu_gpu_mode": build_cpu_gpu_mode(),
        "uncertainty_status": "not_applicable",
        "replication_count": 0,
        "mcse_or_interval": {
            "summary": "Not applicable because IE6 deferred before residual execution; no stochastic output was produced."
        },
        "post_run_red_team_note": (
            "The strongest reason not to over-read IE6 is that a clean deferral records only the absence of an approved "
            "teacher/student artifact. It says nothing about whether learned OT would pass or fail under a provenance-bearing artifact."
        ),
        "cap_non_applicability_reasons": {
            "max_particles": "No learned-map residual fixture was instantiated because no approved artifact was available.",
            "max_time_steps": "No filtering rollout was instantiated.",
            "max_sinkhorn_iterations": "No Sinkhorn teacher or residual execution was instantiated.",
            "max_finite_difference_evaluations": "No finite-difference probe was instantiated.",
            "max_replications": "No stochastic replication was run because the phase deferred.",
        },
        "cap_values": {
            "max_particles": None,
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
            "seed_policy": "deterministic_no_rng_deferred_no_artifact",
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
    lines = [
        "# IE6 learned-OT residual result",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `ie_phase_deferred_with_recorded_reason`",
        "- Local exit label: `ie6_learned_ot_residual_deferred_no_artifact`",
        "- Phase status: `deferred`",
        "",
        "## Skeptical audit before execution",
        "",
        "- Artifact gate audit: no approved pre-existing teacher/student artifact with provenance was identified.",
        "- Substitution audit: IE6 did not invent an analytic teacher/student substitute.",
        "- Training boundary audit: no optimizer steps, checkpoint mutation, or network training occurred.",
        "- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before local imports and records a CPU-only manifest.",
        "",
        "## Research-intent ledger",
        "",
        "- Main question: can IE6 evaluate learned/amortized OT residuals against an approved teacher/student artifact?",
        "- Result: no; the required artifact provenance is absent, so residual execution is deferred.",
        "- Promotion criterion: not triggered because no approved artifact exists.",
        "- Continuation veto: triggered for residual execution; future execution requires approved provenance-bearing artifacts.",
        "- Repair trigger: missing artifact provenance.",
        "- Not concluded: no learned-OT, neural-OT, surrogate-quality, posterior, production, banking, model-risk, or readiness claim follows.",
        "",
        "## Evidence contract",
        "",
        "| Diagnostic | Comparator | Status | Source support |",
        "| --- | --- | --- | --- |",
        f"| `learned_map_residual` | `{record['comparator_id']}` | `{record['status']}` | `{record['source_support_class']}` |",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `learned_map_residual` | `{record['status']}` | `{record['promotion_criterion_status']}` | `{record['promotion_veto_status']}` | `{record['continuation_veto_status']}` | `{record['repair_trigger']}` |",
        "",
        "## Coverage semantics",
        "",
        "- IE3 through IE5 diagnostics are carried forward as `passed`.",
        "- `learned_map_residual` is `deferred`.",
        "- IE7 and IE8 remain `missing`.",
        "",
        "## Artifact list",
        "",
        "- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`",
        "- `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`",
        "- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-result-2026-05-16.md`",
        "",
        "## Run manifest",
        "",
        f"- command: `{record['command']}`",
        f"- branch: `{record['run_manifest']['branch']}`",
        f"- commit: `{record['run_manifest']['commit']}`",
        f"- python: `{record['run_manifest']['python_version']}`",
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
        "IE6 deferral does not block IE7 because IE7 uses a fixed scalar HMC-facing target fixture and does not depend on learned-OT residual execution. IE7 must not interpret IE6 as learned-OT validation.",
    ]
    MARKDOWN_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_result(record: dict[str, Any]) -> None:
    lines = [
        "# Phase IE6 result: learned-OT teacher/student/OOD residual tests",
        "",
        "## Outcome",
        "",
        "- Master-program exit label: `ie_phase_deferred_with_recorded_reason`",
        "- Local exit label: `ie6_learned_ot_residual_deferred_no_artifact`",
        "- Phase status: `deferred`",
        "",
        "## Reason",
        "",
        "No approved pre-existing teacher/student artifact with provenance was available. IE6 therefore emitted a schema-valid deferred `learned_map_residual` row and did not execute residual tests.",
        "",
        "## Decision table",
        "",
        "| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| `learned_map_residual` | `{record['status']}` | `{record['promotion_criterion_status']}` | `{record['promotion_veto_status']}` | `{record['continuation_veto_status']}` | `{record['repair_trigger']}` |",
        "",
        "## Artifacts",
        "",
        "- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`",
        "- `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`",
        "",
        "## Non-implication",
        "",
        NON_IMPLICATION,
        "",
        "## Next-phase justification",
        "",
        "Proceeding to IE7 is justified only because IE7 tests an independent fixed scalar value-gradient contract. IE6 remains an explicit learned-OT evidence gap.",
    ]
    PHASE_RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    start = time.perf_counter()
    started_at = utc_now()
    command = "python -m experiments.dpf_monograph_evidence.runners.run_learned_ot_residual"
    runtime_seconds = time.perf_counter() - start
    ended_at = utc_now()
    record = make_result_record(command, runtime_seconds, started_at, ended_at)
    write_json(JSON_OUTPUT, record)
    write_markdown_report(record)
    write_phase_result(record)
    print(f"wrote {JSON_OUTPUT}")
    print(f"wrote {MARKDOWN_REPORT_PATH}")
    print(f"wrote {PHASE_RESULT_PATH}")
    print("ie_phase_deferred_with_recorded_reason")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
