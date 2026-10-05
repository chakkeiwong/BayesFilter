"""Write and validate the DPF common model suite v2 manifest."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    common_model_suite_v2_manifest,
    common_model_specs,
    validate_common_model_suite_v2_manifest,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p1-declarative-spec-subplan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p1-declarative-spec-result-2026-06-07.md"
JSON_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-common-model-suite-v2-manifest-2026-06-07.md"
EXPECTED_V1_CHECKSUMS = {
    "lgssm_2d_linear": "86a39792c98796be8fa756369b5fa7f265b8c03af5692ecbfd6a125199cac932",
    "sv_1d_synthetic": "20d4fcd2cb28fd358461d37c37cfa1ef3e7f857c3367e052bf43889c430dfcf7",
    "range_bearing_2d_cv": "a74c4c01f61c1fd637866efc87d739d20a759e09b5e9e6bca12129e9a78b2a88",
}


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
    manifest = common_model_suite_v2_manifest()
    v1_checksums = {spec.model_id: spec.checksum() for spec in common_model_specs()}
    v1_unchanged = v1_checksums == EXPECTED_V1_CHECKSUMS
    manifest["created_at_utc"] = utc_now()
    manifest["question"] = (
        "Can common_model_suite_tf.py expose the six-row v2 production suite "
        "while preserving closed v1 fixture semantics?"
    )
    manifest["plan_path"] = PLAN_PATH
    manifest["result_path"] = RESULT_PATH
    manifest["decision"] = "PENDING_CLAUDE_REVIEW" if v1_unchanged else "P1_V1_REGRESSION_BLOCKED"
    manifest["v1_validation_only"] = {
        "expected_checksums": EXPECTED_V1_CHECKSUMS,
        "actual_checksums": v1_checksums,
        "unchanged": v1_unchanged,
        "note": "common_model_specs() is read only as closed-v1 validation, never as v2 source.",
    }
    manifest["run_manifest"] = environment_manifest(
        command=(
            "CUDA_VISIBLE_DEVICES=-1 python -m "
            "experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_manifest_tf"
        ),
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest["artifact_paths"] = {
        "json": str(JSON_PATH.relative_to(REPO_ROOT)),
        "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "phase_result": RESULT_PATH,
    }
    manifest["primary_criterion_fields"] = {
        **manifest["primary_criterion_fields"],
        "v2_manifest_exact_model_ids": [row["model_id"] for row in manifest["rows"]],
        "v1_validation_only_checksums_unchanged": v1_unchanged,
        "pre_run_classification_rows": len(manifest["pre_run_row_classification_table"]),
    }
    manifest["veto_diagnostics"] = {
        **manifest["veto_diagnostics"],
        "v1_checksum_regression": not v1_unchanged,
        "manifest_validation_error": False,
    }
    return manifest


def _validate_payload(payload: dict[str, Any]) -> None:
    validate_common_model_suite_v2_manifest(payload)
    ids = [row["model_id"] for row in payload["rows"]]
    if tuple(ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 v2 manifest id gate failed: {ids}")
    if payload["decision"] not in {"PENDING_CLAUDE_REVIEW", "PASS_P1_DECLARATIVE_SPEC_READY_FOR_P2"}:
        raise ValueError(f"P1 manifest decision not passable: {payload['decision']}")
    if payload["veto_diagnostics"].get("v1_checksum_regression"):
        raise ValueError("P1 v1 checksum regression veto fired")
    if len(payload["pre_run_row_classification_table"]) != len(EXPECTED_V2_MODEL_IDS):
        raise ValueError("P1 pre-run classification table row count mismatch")


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"] = dict(clone["run_manifest"])
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _markdown(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    classification = payload["pre_run_row_classification_table"]
    run_manifest = payload["run_manifest"]
    lines = [
        "# DPF Common Model Suite V2 P1 Declarative Spec Result",
        "",
        "metadata_date: 2026-06-07",
        "phase: P1",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary criterion: expose exactly the six v2 production rows, with v1 APIs preserved for validation-only reads.",
        "",
        "Veto diagnostics: old v1 API cannot be the v2 source, old 2026-06-06 artifact names cannot be used, v1 checksums cannot drift, and no BF/FF comparison may run in P1.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{payload['artifact_paths']['json']}`",
        f"- Markdown report: `{payload['artifact_paths']['markdown_report']}`",
        f"- Manifest checksum: `{payload['checksum']}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## V2 Rows",
        "",
        "| Model id | Family | P2 | P3 | P4 | P5 |",
        "|---|---|---|---|---|---|",
    ]
    readiness_by_id = {row["model_id"]: row for row in classification}
    for row in rows:
        ready = readiness_by_id[row["model_id"]]
        lines.append(
            f"| `{row['model_id']}` | `{row['family']}` | "
            f"{ready['P2_density']} | {ready['P3_noresampling']} | "
            f"{ready['P4_fixed_ancestor']} | {ready['P5_gradients']} |"
        )
    lines.extend(
        [
            "",
            "## Primary Criterion Fields",
            "",
            f"- exact six-row gate: `{payload['primary_criterion_fields']['v2_manifest_exact_model_ids']}`",
            f"- v1 validation-only checksums unchanged: `{payload['primary_criterion_fields']['v1_validation_only_checksums_unchanged']}`",
            f"- pre-run classification rows: `{payload['primary_criterion_fields']['pre_run_classification_rows']}`",
            "",
            "## Veto Diagnostics",
            "",
            f"- old three-row API used as v2 source: `{payload['veto_diagnostics']['old_three_row_common_model_specs_used_as_v2_source']}`",
            f"- old 2026-06-06 artifact name used: `{payload['veto_diagnostics']['old_2026_06_06_artifact_name_used']}`",
            f"- v1 checksum regression: `{payload['veto_diagnostics']['v1_checksum_regression']}`",
            f"- student command executed: `{payload['veto_diagnostics']['student_command_executed']}`",
            f"- `.localsource/filterflow` mutated: `{payload['veto_diagnostics']['localsource_filterflow_mutated']}`",
            "",
            "## Explanatory Only Fields",
            "",
            f"- v1 validation-only checksums: `{payload['v1_validation_only']['actual_checksums']}`",
            "- FilterFlow-side v2 execution is planned through CPU-only subprocess adapters without mutating `.localsource/filterflow`.",
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{run_manifest.get('commit')}` |",
            f"| git branch | `{run_manifest.get('branch')}` |",
            f"| dirty status | `{_single_line(run_manifest.get('dirty_state_summary'))}` |",
            f"| command | `{run_manifest.get('command')}` |",
            f"| validation commands | `python -m json.tool {payload['artifact_paths']['json']}`; `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-mpl python -m experiments.dpf_implementation.tf_tfp.runners.run_common_model_suite_v2_manifest_tf --validate-only`; `git diff --check` on P1 files |",
            f"| environment | `/home/chakwong/BayesFilter`; Python `{run_manifest.get('python_version')}`; TensorFlow `{run_manifest.get('package_versions', {}).get('tensorflow')}`; TFP `{run_manifest.get('package_versions', {}).get('tensorflow_probability')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow run; pre-import `CUDA_VISIBLE_DEVICES={run_manifest.get('pre_import_cuda_visible_devices')}`; visible GPUs `{run_manifest.get('gpu_devices_visible')}` |",
            "| random seeds | fixture seeds recorded per row in JSON; no stochastic comparison run in P1 |",
            "| dtype | `tf.float64` / JSON dtype `float64` |",
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
            "- Round 1 Claude result review returned `BLOCKED` because this result ledger lacked required `Command Manifest` and `Repair History` sections.",
            "- Repair amendment reviewed by Claude to `PASS`: `docs/plans/bayesfilter-dpf-common-model-suite-v2-production-p1-artifact-adequacy-repair-amendment-2026-06-07.md`.",
            "- Implemented repair is artifact-completeness only; it does not change model ids, fixtures, tolerances, scalar definitions, classifications, adapter semantics, v1 checksums, or any scientific contract.",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
            "|---|---|---|---|---|---|",
            "| PENDING_CLAUDE_REVIEW | six v2 rows exist and v1 validation-only checksums are preserved | no local veto fired | Claude may find artifact/governance gaps | run Claude P1 result review | no BF/FF density/path/gradient agreement, no correctness proof |",
            "",
            "## Post-Run Red Team",
            "",
            "Strongest alternative explanation: P1 only declares the contracts; later runners may still implement an adapter incorrectly.",
            "",
            "Result that would overturn the decision: a later phase or Claude review finds that a row's adapter semantics were underspecified, or that v1/v2 leakage occurred.",
            "",
            "Weakest evidence link: SIR and predator-prey are declared no-lookup adapter rows; P2--P5 must certify execution semantics before comparing values.",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in payload["non_claims"])
    lines.append("")
    return "\n".join(lines)


def _single_line(value: object) -> str:
    return str(value).replace("\n", " | ")


if __name__ == "__main__":
    raise SystemExit(main())
