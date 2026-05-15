from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

from experiments.dpf_monograph_evidence import results

IMPORT_SCAN_ROOT = results.EVIDENCE_ROOT
SELF_NAME = "experiments.dpf_monograph_evidence.runners.validate_results"


class ImportBoundaryError(ValueError):
    pass


def iter_python_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if path.name == "__pycache__":
            continue
        yield path


def scan_import_boundaries(root: Path) -> list[str]:
    violations: list[str] = []
    for path in iter_python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if module_name.startswith(results.IMPORT_BOUNDARY_PREFIXES):
                        violations.append(f"{path}: forbidden import {module_name}")
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name.startswith(results.IMPORT_BOUNDARY_PREFIXES):
                    violations.append(f"{path}: forbidden import {module_name}")
    return violations


def make_placeholder_result(result_path: Path) -> dict[str, object]:
    relative_result_path = str(result_path.resolve().relative_to(results.REPO_ROOT))
    return {
        "phase_id": "IE8",
        "diagnostic_id": "posterior_sensitivity_summary",
        "chapter_label": "Chapter 26",
        "diagnostic_role": "explanatory",
        "comparator_id": "ie2_schema_only",
        "source_family": "not_source_dependent",
        "source_support_class": "not_source_dependent",
        "row_level_source_support_class": "not_source_dependent",
        "seed_policy": "fixed_seed_placeholder",
        "status": "deferred",
        "coverage": {
            "linear_gaussian_recovery": "missing",
            "synthetic_affine_flow": "missing",
            "pfpf_algebra_parity": "missing",
            "soft_resampling_bias": "missing",
            "sinkhorn_residual": "missing",
            "learned_map_residual": "missing",
            "hmc_value_gradient": "missing",
            "posterior_sensitivity_summary": "deferred",
        },
        "tolerance": {"summary": "schema-only placeholder; no numerical tolerance evaluated"},
        "finite_checks": {"summary": "no numerical execution in validate-only placeholder"},
        "shape_checks": {"summary": "schema-only placeholder; no tensors instantiated"},
        "runtime_seconds": 0.0,
        "blocker_class": "none",
        "non_implication": "Schema validation readiness does not imply any diagnostic passed, any production filter is correct, or any empirical DPF-HMC claim is supported.",
        "promotion_criterion_status": "not_triggered",
        "promotion_veto_status": "not_triggered",
        "continuation_veto_status": "not_triggered",
        "repair_trigger": "Replace placeholder row with phase-owned diagnostics before any research interpretation.",
        "explanatory_only_diagnostics": ["coverage scaffold only"],
        "environment": {
            "python_version": "placeholder-populate-at-runtime",
            "package_versions": {"stdlib": "builtin"},
            "branch": "placeholder",
            "commit": "placeholder",
            "dirty_state_summary": "placeholder",
            "cpu_only": True,
            "cuda_visible_devices": "-1",
            "gpu_hidden_before_import": True,
            "import_boundary_checked_modules": [SELF_NAME],
        },
        "command": "python -m experiments.dpf_monograph_evidence.runners.validate_results --validate-only",
        "wall_time_seconds": 0.0,
        "wall_clock_cap_seconds": 30,
        "artifact_paths": [relative_result_path],
        "cpu_gpu_mode": {
            "cpu_only": True,
            "cuda_visible_devices": "-1",
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": True,
        },
        "uncertainty_status": "not_applicable",
        "replication_count": 0,
        "mcse_or_interval": {"summary": "not applicable for schema-only placeholder"},
        "post_run_red_team_note": "A passing validator here could still miss future schema drift or overly weak phase-specific diagnostics.",
        "cap_non_applicability_reasons": {
            "max_particles": "No particle system is instantiated in the validate-only placeholder.",
            "max_time_steps": "No time-series rollout is instantiated in the validate-only placeholder.",
            "max_sinkhorn_iterations": "No Sinkhorn solver is instantiated in the validate-only placeholder.",
            "max_finite_difference_evaluations": "No finite-difference probe is instantiated in the validate-only placeholder.",
            "max_replications": "No stochastic replication is instantiated in the validate-only placeholder.",
        },
        "cap_values": {
            "max_particles": None,
            "max_time_steps": None,
            "max_sinkhorn_iterations": None,
            "max_finite_difference_evaluations": None,
            "max_replications": None,
            "max_wall_clock_seconds": 30,
        },
        "run_manifest": {
            "command": "python -m experiments.dpf_monograph_evidence.runners.validate_results --validate-only",
            "branch": "placeholder",
            "commit": "placeholder",
            "dirty_state_summary": "placeholder",
            "python_version": "placeholder-populate-at-runtime",
            "package_versions": {"stdlib": "builtin"},
            "cpu_only": True,
            "cuda_visible_devices": "-1",
            "gpu_devices_visible": [],
            "gpu_hidden_before_import": True,
            "pre_import_cuda_visible_devices": "-1",
            "pre_import_gpu_hiding_assertion": True,
            "seed_policy": "fixed_seed_placeholder",
            "wall_clock_cap_seconds": 30,
            "started_at_utc": "placeholder",
            "ended_at_utc": "placeholder",
            "artifact_paths": [relative_result_path],
        },
    }


def validate_result_paths(paths: list[Path]) -> None:
    for path in paths:
        results.validate_result_file(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate clean-room DPF monograph evidence results.")
    parser.add_argument("result_paths", nargs="*", type=Path, help="JSON result files to validate.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run import-boundary and schema smoke validation without requiring pre-existing results.",
    )
    parser.add_argument(
        "--write-placeholder",
        type=Path,
        help="Optional path for a validate-only placeholder result JSON under experiments/dpf_monograph_evidence/.",
    )
    args = parser.parse_args()

    violations = scan_import_boundaries(IMPORT_SCAN_ROOT)
    if violations:
        raise ImportBoundaryError("\n".join(violations))

    if args.write_placeholder is not None:
        placeholder_path = args.write_placeholder.resolve()
        placeholder_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder = make_placeholder_result(placeholder_path)
        placeholder_path.write_text(json.dumps(placeholder, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.result_paths.append(placeholder_path)

    if args.result_paths:
        validate_result_paths([path.resolve() for path in args.result_paths])
    elif not args.validate_only:
        parser.error("provide result_paths or use --validate-only")
    else:
        results.validate_result_record(make_placeholder_result(IMPORT_SCAN_ROOT / "reports" / "validate_only_placeholder.json"))

    print("validation_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
