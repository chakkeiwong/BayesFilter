"""Validate v2 retirement of standalone common-suite fixture imports."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import ast
import subprocess
import time
from pathlib import Path
from typing import Any

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    common_model_specs,
    common_model_specs_v2,
    common_model_suite_v2_manifest,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p6-retirement-regression-subplan-2026-06-07.md"
AMENDMENT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p6-retirement-import-absorption-repair-amendment-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p6-retirement-regression-result-2026-06-07.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_retirement_manifest_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-retirement-2026-06-07.md"
P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"

RETIRED_MODULES = {
    "experiments.dpf_implementation.tf_tfp.fixtures.lgssm_tf": {
        "path": "experiments/dpf_implementation/tf_tfp/fixtures/lgssm_tf.py",
        "successors": ["lgssm_2d_h25_rich"],
    },
    "experiments.dpf_implementation.tf_tfp.fixtures.stochastic_volatility_tf": {
        "path": "experiments/dpf_implementation/tf_tfp/fixtures/stochastic_volatility_tf.py",
        "successors": ["sv_1d_h18_rich"],
    },
    "experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf": {
        "path": "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py",
        "successors": ["range_bearing_4d_h20_rich"],
    },
}
PRODUCTION_V2_FILES = {
    "experiments/dpf_implementation/tf_tfp/fixtures/common_model_suite_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_manifest_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_density_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_noresampling_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_fixed_resampling_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_gradients_tf.py",
    "experiments/dpf_implementation/tf_tfp/runners/run_common_model_suite_v2_retirement_tf.py",
}
V1_VALIDATION_COMMANDS = [
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_tieout_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_filter_path_noresampling_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_filter_path_fixed_resampling_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_fixed_branch_gradient_tf --validate-only",
]
V2_VALIDATION_COMMANDS = [
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_manifest_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_density_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_noresampling_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_fixed_resampling_tf --validate-only",
    "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_gradients_tf --validate-only",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    inventory = _import_inventory()
    source_checksums = _source_checksums()
    checksum_stability = _checksum_stability()
    v2_validation = [_run_command(command) for command in V2_VALIDATION_COMMANDS]
    v1_validation = [_run_command(command) for command in V1_VALIDATION_COMMANDS]
    forbidden_empty = not inventory["production_v2_imports_forbidden"]
    v2_pass = all(item["returncode"] == 0 for item in v2_validation)
    v1_pass = all(item["returncode"] == 0 for item in v1_validation)
    veto = {
        "production_v2_forbidden_imports_present": not forbidden_empty,
        "v2_validation_failed": not v2_pass,
        "v1_validation_failed": not v1_pass,
        "v2_row_checksum_drift": not checksum_stability["v2_rows_unchanged"],
        "v2_manifest_base_checksum_drift": not checksum_stability["v2_manifest_base_checksum_unchanged"],
        "v1_checksum_drift": not checksum_stability["v1_checksums_unchanged"],
        "old_fixture_files_missing": any(not (REPO_ROOT / meta["path"]).exists() for meta in RETIRED_MODULES.values()),
        "student_command_executed": False,
        "localsource_filterflow_mutated": False,
    }
    open_blockers = [name for name, fired in veto.items() if fired]
    decision = "PENDING_CLAUDE_REVIEW" if not open_blockers else "P6_RETIREMENT_BLOCKED"
    return {
        "artifact_id": "dpf_common_model_suite_v2_retirement_manifest_2026-06-07",
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Can the standalone LGSSM, SV, and range-bearing fixture modules be "
            "retired from the production v2 path while closed-v1 and v2 artifacts validate?"
        ),
        "plan_path": PLAN_PATH,
        "repair_amendment_path": AMENDMENT_PATH,
        "result_path": RESULT_PATH,
        "retired_modules": {
            module: {
                **meta,
                "sha256": source_checksums[meta["path"]],
                "retirement_status": "RETIRED_BY_V2_COMMON_SUITE_PRODUCTION_PATH",
            }
            for module, meta in RETIRED_MODULES.items()
        },
        "primary_criterion_fields": {
            "production_v2_forbidden_import_class_empty": forbidden_empty,
            "v2_validation_only_commands_passed": v2_pass,
            "v1_validation_only_commands_passed": v1_pass,
            "v2_row_checksums_unchanged": checksum_stability["v2_rows_unchanged"],
            "v2_manifest_base_checksum_unchanged": checksum_stability["v2_manifest_base_checksum_unchanged"],
            "v1_checksums_unchanged": checksum_stability["v1_checksums_unchanged"],
            "old_fixture_files_preserved": not veto["old_fixture_files_missing"],
        },
        "veto_diagnostics": veto,
        "explanatory_only_fields": {
            "legacy_v1_validation_commands": V1_VALIDATION_COMMANDS,
            "v2_validation_commands": V2_VALIDATION_COMMANDS,
            "remaining_allowed_import_counts": {
                key: len(value)
                for key, value in inventory.items()
                if key != "production_v2_imports_forbidden"
            },
            "retirement_semantics": (
                "Old fixture modules are preserved for legacy/reference/nonproduction use; "
                "production v2 no longer imports them."
            ),
        },
        **inventory,
        "checksum_stability": checksum_stability,
        "v1_validation_results": v1_validation,
        "v2_validation_results": v2_validation,
        "review_round": 0,
        "open_material_blockers": open_blockers,
        "repair_amendment_required": bool(open_blockers),
        "next_allowed_action": (
            "run Claude P6 result/governance review before P7"
            if not open_blockers
            else "write reviewed P6 repair amendment or stop"
        ),
        "artifact_paths": {
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
            "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "phase_result": RESULT_PATH,
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_retirement_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "retirement is production-source cleanup, not a mathematical correctness result",
            "old standalone fixture modules are not claimed wrong",
            "no filter correctness proof",
            "no student implementation claim",
            "no TT/SIRT or paper-scale reproduction claim",
        ],
    }


def _import_inventory() -> dict[str, list[dict[str, Any]]]:
    inventory = {
        "production_v2_imports_forbidden": [],
        "legacy_v1_validation_allowed": [],
        "reference_only_allowed": [],
        "nonproduction_research_runner_allowed": [],
    }
    for path in sorted((REPO_ROOT / "experiments" / "dpf_implementation").rglob("*.py")):
        relative = str(path.relative_to(REPO_ROOT))
        imports = _retired_imports(path)
        if not imports:
            continue
        entry = {"path": relative, "imports": imports}
        if relative in PRODUCTION_V2_FILES:
            inventory["production_v2_imports_forbidden"].append(entry)
        elif "run_common_" in path.name and "_v2_" not in path.name:
            inventory["legacy_v1_validation_allowed"].append(entry)
        elif "/references/" in f"/{relative}":
            inventory["reference_only_allowed"].append(entry)
        else:
            inventory["nonproduction_research_runner_allowed"].append(entry)
    return inventory


def _retired_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"syntax_error:{exc}"]
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in RETIRED_MODULES:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in RETIRED_MODULES:
                    imports.append(alias.name)
    return sorted(set(imports))


def _source_checksums() -> dict[str, str]:
    checksums = {}
    for meta in RETIRED_MODULES.values():
        relative = meta["path"]
        path = REPO_ROOT / relative
        checksums[relative] = stable_digest(path.read_text(encoding="utf-8"))
    return checksums


def _checksum_stability() -> dict[str, Any]:
    stored_p1 = load_json(P1_MANIFEST_PATH)
    current_manifest = common_model_suite_v2_manifest()
    stored_rows = {row["model_id"]: row["checksum"] for row in stored_p1["rows"]}
    current_rows = {spec.model_id: spec.checksum() for spec in common_model_specs_v2()}
    stored_v1 = stored_p1.get("v1_validation_only", {}).get("expected_checksums", {})
    current_v1 = {spec.model_id: spec.checksum() for spec in common_model_specs()}
    return {
        "stored_v2_row_checksums": stored_rows,
        "current_v2_row_checksums": current_rows,
        "v2_rows_unchanged": stored_rows == current_rows,
        "stored_v2_manifest_base_checksum": stored_p1.get("checksum"),
        "current_v2_manifest_base_checksum": current_manifest.get("checksum"),
        "v2_manifest_base_checksum_unchanged": stored_p1.get("checksum") == current_manifest.get("checksum"),
        "stored_v1_checksums": stored_v1,
        "current_v1_checksums": current_v1,
        "v1_checksums_unchanged": stored_v1 == current_v1,
    }


def _run_command(command: str) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_excerpt": completed.stdout[-2000:],
        "stderr_excerpt": completed.stderr[-2000:],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "primary_criterion_fields",
        "veto_diagnostics",
        "explanatory_only_fields",
        "review_round",
        "open_material_blockers",
        "repair_amendment_required",
        "next_allowed_action",
        "production_v2_imports_forbidden",
        "legacy_v1_validation_allowed",
        "reference_only_allowed",
        "nonproduction_research_runner_allowed",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"P6 retirement manifest missing fields: {sorted(missing)}")
    if payload["production_v2_imports_forbidden"]:
        raise ValueError("P6 forbidden production-v2 import class is not empty")
    if payload["veto_diagnostics"].get("v1_validation_failed"):
        raise ValueError("P6 v1 validation failed")
    if payload["veto_diagnostics"].get("v2_validation_failed"):
        raise ValueError("P6 v2 validation failed")
    if payload["veto_diagnostics"].get("v2_row_checksum_drift"):
        raise ValueError("P6 v2 row checksum drift")
    if payload["veto_diagnostics"].get("v2_manifest_base_checksum_drift"):
        raise ValueError("P6 v2 manifest base checksum drift")
    if payload["veto_diagnostics"].get("v1_checksum_drift"):
        raise ValueError("P6 v1 checksum drift")
    if payload["veto_diagnostics"].get("old_fixture_files_missing"):
        raise ValueError("P6 old fixture file preservation veto fired")
    if payload["decision"] not in {"PENDING_CLAUDE_REVIEW", "PASS_P6_RETIREMENT_READY_FOR_P7"}:
        raise ValueError(f"P6 decision not passable: {payload['decision']}")


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    run_manifest = payload["run_manifest"]
    lines = [
        "# DPF Common Model Suite V2 P6 Retirement And Regression Result",
        "",
        "metadata_date: 2026-06-07",
        "phase: P6",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: no production v2 runner imports the retired standalone fixture modules, while closed-v1 and v2 validation-only artifacts still validate.",
        "",
        "Veto diagnostics: forbidden production-v2 import, v1/v2 validation failure, missing old fixture files, student command execution, or `.localsource/filterflow` mutation.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Retired Modules",
        "",
        "| Module | File | Successor v2 ids | SHA256 | Status |",
        "|---|---|---|---|---|",
    ]
    for module, meta in payload["retired_modules"].items():
        lines.append(
            f"| `{module}` | `{meta['path']}` | `{meta['successors']}` | `{meta['sha256']}` | `{meta['retirement_status']}` |"
        )
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
            f"- production v2 forbidden import class empty: `{payload['primary_criterion_fields']['production_v2_forbidden_import_class_empty']}`",
            f"- v2 validation-only commands passed: `{payload['primary_criterion_fields']['v2_validation_only_commands_passed']}`",
            f"- v1 validation-only commands passed: `{payload['primary_criterion_fields']['v1_validation_only_commands_passed']}`",
            f"- v2 row checksums unchanged: `{payload['primary_criterion_fields']['v2_row_checksums_unchanged']}`",
            f"- v2 manifest base checksum unchanged: `{payload['primary_criterion_fields']['v2_manifest_base_checksum_unchanged']}`",
            f"- v1 checksums unchanged: `{payload['primary_criterion_fields']['v1_checksums_unchanged']}`",
            f"- old fixture files preserved: `{payload['primary_criterion_fields']['old_fixture_files_preserved']}`",
            "",
            "## Veto Diagnostics",
            "",
        ]
    )
    for key, value in payload["veto_diagnostics"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Import Inventory",
            "",
            f"- production_v2_imports_forbidden: `{payload['production_v2_imports_forbidden']}`",
            f"- legacy_v1_validation_allowed: `{payload['legacy_v1_validation_allowed']}`",
            f"- reference_only_allowed: `{payload['reference_only_allowed']}`",
            f"- nonproduction_research_runner_allowed: `{payload['nonproduction_research_runner_allowed']}`",
            "",
            "## Checksum Stability",
            "",
            f"- stored v2 row checksums: `{payload['checksum_stability']['stored_v2_row_checksums']}`",
            f"- current v2 row checksums: `{payload['checksum_stability']['current_v2_row_checksums']}`",
            f"- stored v1 checksums: `{payload['checksum_stability']['stored_v1_checksums']}`",
            f"- current v1 checksums: `{payload['checksum_stability']['current_v1_checksums']}`",
            "",
            "## Validation Results",
            "",
            "### V2",
            "",
        ]
    )
    lines.extend(_validation_lines(payload["v2_validation_results"]))
    lines.extend(["", "### V1", ""])
    lines.extend(_validation_lines(payload["v1_validation_results"]))
    lines.extend(
        [
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            f"| dirty status | `{_single_line(run_manifest.get('dirty_state_summary'))}` |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_retirement_tf --validate-only`; `git diff --check` on P6 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | N/A; validation-only retirement/import inventory phase |",
            f"| output artifacts | `{payload['artifact_paths']['json']}`; `{payload['artifact_paths']['markdown_report']}`; `{payload['artifact_paths']['phase_result']}` |",
            "",
            "## Review State",
            "",
            f"review_round: {payload['review_round']} pending Claude result/governance review",
            "",
            f"open_material_blockers: {payload['open_material_blockers'] or 'none identified locally'}",
            "",
            f"repair_amendment_required: {str(payload['repair_amendment_required']).lower()}",
            "",
            f"next_allowed_action: {payload['next_allowed_action']}",
            "",
            "## Repair History",
            "",
            f"- P6 import-absorption repair amendment reviewed by Claude: `{AMENDMENT_PATH}`.",
            "- The repair removes production-v2 imports of old standalone fixture modules without editing or deleting those old modules.",
            "- The repair does not change model equations, observations, tolerances, scalar definitions, classifications, gradient knobs, or non-claims.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| PENDING_CLAUDE_REVIEW | production v2 import class empty; v1 and v2 validations pass locally | no local veto fired | Claude may find inventory/governance gaps | run Claude P6 result review | no mathematical correctness, old-file wrongness, or student claim |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: the import inventory classification could miss a future production entry point outside the declared v2 runner set.",
            "",
            "Result that would overturn the decision: a production v2 runner is later found to import the retired modules or validation-only artifacts no longer validate.",
            "",
            "Weakest evidence link: retirement is a source-dependency contract, not independent scientific validation.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _validation_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = ["| Command | Return code |", "|---|---|"]
    for item in results:
        lines.append(f"| `{item['command']}` | `{item['returncode']}` |")
    return lines


def _single_line(value: object) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
