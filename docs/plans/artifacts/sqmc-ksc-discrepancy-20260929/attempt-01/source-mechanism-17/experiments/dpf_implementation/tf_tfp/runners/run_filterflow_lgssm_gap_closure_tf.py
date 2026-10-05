"""Gap-closure rerun for the filterflow LGSSM matched audit."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import run_filterflow_lgssm_matched_cross_audit_tf as audit
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    load_json,
    stable_digest,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    validate_filterflow_reference_status,
)


JSON_PATH = OUTPUT_DIR / "dpf_filterflow_lgssm_gap_closure_2026-05-30.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-lgssm-gap-closure-2026-05-30.md"
PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-gap-closure-plan-2026-05-30.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = audit._run()
    payload["created_by_runner"] = "run_filterflow_lgssm_gap_closure_tf"
    payload["plan_path"] = PLAN_PATH
    payload["gap_closure_ledger"] = _gap_closure_ledger(payload)
    payload["run_manifest"]["command"] = (
        "CUDA_VISIBLE_DEVICES=-1 python -m "
        "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_lgssm_gap_closure_tf"
    )
    payload["run_manifest"]["pre_import_cuda_visible_devices"] = PRE_IMPORT_CUDA_VISIBLE_DEVICES
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _gap_closure_ledger(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": "filterflow_executable",
            "status": payload["filterflow_status"]["branch"],
            "commit": payload["filterflow_status"]["commit"],
            "diff_summary": payload["filterflow_status"]["diff_summary"],
        },
        {
            "id": "paper_code_setting_ledger",
            "status": "section_5_1_style_not_exact_paper_table_claim",
            "detail": (
                "Matched to executable filterflow simple_linear_comparison settings: transition covariance I_2, "
                "observation covariance 0.1 I_2, T=150, N=25, theta grid 0.25/0.5/0.75, "
                "RegularisedTransform scaling=0.9 and convergence_threshold=1e-3. "
                "Earlier audit recorded a paper/code covariance ambiguity, so exact paper-table reproduction "
                "is not claimed here."
            ),
        },
        {
            "id": "kalman_alignment",
            "status": "pass" if payload["kalman_alignment"]["all_within_tolerance"] else "veto",
            "max_abs_delta": payload["kalman_alignment"]["max_abs_delta"],
        },
        {
            "id": "pf_calibration",
            "status": _comparison_status(payload, "bayesfilter_pf"),
        },
        {
            "id": "corrected_filterflow_style_transport",
            "status": _comparison_status(payload, "bayesfilter_filterflow_style_transport_ess"),
        },
        {
            "id": "fixed_sinkhorn_small_epsilon",
            "status": _comparison_status(payload, "bayesfilter_scaled_fixed_sinkhorn_ess"),
        },
        {
            "id": "gradient_smoothness_replication",
            "status": "not_run_separate_gap",
            "detail": "This gap closure reruns LGSSM likelihood behavior only; gradient/smoothness replication remains separate.",
        },
    ]


def _comparison_status(payload: dict[str, Any], method_id: str) -> str:
    rows = [row for row in payload["comparison"] if row["bayesfilter_method"] == method_id]
    if any(row["bayesfilter_status"] != "executed" for row in rows):
        return "veto_or_missing"
    if all(row["within_one_filterflow_sd"] for row in rows):
        return "within_filterflow_mc_band"
    return "outside_filterflow_mc_band"


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow LGSSM Gap-Closure Rerun

## Decision

`{payload['decision']}`

## Summary

This rerun uses the patched external filterflow branch and the corrected
BayesFilter experimental filterflow-style transport mirror.  It is a
Section-5.1-style executable-code match, not an exact paper-table claim.

## Gap-Closure Ledger

{_ledger_table(payload['gap_closure_ledger'])}

## Filterflow

- Branch: `{payload['filterflow_status']['branch']}`
- Commit: `{payload['filterflow_status']['commit']}`
- Status: `{payload['filterflow_status']['status']}`
- Diff summary: `{payload['filterflow_status']['diff_summary']}`

## Comparison

{audit._comparison_table(payload['comparison'])}

## Non-Implications

{audit._non_implications_markdown()}
- No gradient/smoothness replication is concluded.
- No exact paper-table reproduction is concluded.
"""


def _ledger_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| ID | Status | Detail |", "| --- | --- | --- |"]
    for row in rows:
        detail = row.get("detail", row.get("max_abs_delta", row.get("commit", "")))
        lines.append(f"| `{row['id']}` | `{row['status']}` | {detail} |")
    return "\n".join(lines)


def _validate_payload(payload: dict[str, Any]) -> None:
    audit._validate_payload(payload)
    if payload["plan_path"] != PLAN_PATH:
        raise RuntimeError("wrong gap closure plan path")
    if payload["created_by_runner"] != "run_filterflow_lgssm_gap_closure_tf":
        raise RuntimeError("wrong runner id")
    if "gap_closure_ledger" not in payload:
        raise RuntimeError("missing gap closure ledger")
    validate_filterflow_reference_status(payload["filterflow_status"])


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
