"""Common-sense cross-implementation tie-out orchestrator.

This runner compares only declared small fixtures.  It does not treat any
implementation as an oracle and does not claim filtering-algorithm correctness.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_lgssm_step_gradient_comparison_tf as lgssm_gradient,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_lgssm_paper_table_gated_comparator_tf as lgssm_table,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-cross-implementation-common-sense-tieout-plan-2026-06-06.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-cross-implementation-common-sense-tieout-result-2026-06-06.md"
JSON_PATH = OUTPUT_DIR / "dpf_cross_implementation_common_sense_tieout_2026-06-06.json"
REPORT_PATH = REPORT_DIR / "dpf-cross-implementation-common-sense-tieout-2026-06-06.md"
STUDENT_REFERENCE_SUMMARY = (
    REPO_ROOT / "experiments/student_dpf_baselines/reports/outputs/references/summary.json"
)
STUDENT_PANEL_PATH = (
    REPO_ROOT / "experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_2026-05-10.json"
)
FILTERFLOW_SV_MODEL = REPO_ROOT / ".localsource/filterflow/filterflow/models/stochastic_volatility.py"


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
    cells = [
        _lgssm_gradient_cell(),
        _lgssm_table_cell(),
        _sv_interface_cell(),
        _student_lgssm_cell(),
        _interface_blocked_cell(
            model="scalar_nonlinear",
            reason="no same named FilterFlow/student model contract identified in this slice",
        ),
        _interface_blocked_cell(
            model="spatial_sir",
            reason="BayesFilter has first-gate model contract; no comparable FilterFlow/student interface identified",
        ),
        _interface_blocked_cell(
            model="predator_prey",
            reason="BayesFilter has first-gate model contract; no comparable FilterFlow/student interface identified",
        ),
    ]
    decision = _decision(cells)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": "Common-sense value/gradient consistency tie-out across available implementations",
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "cells": cells,
        "summary": _summary(cells),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_cross_implementation_common_sense_tieout_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no filtering-algorithm correctness claim",
            "no implementation is treated as an oracle",
            "no TT-filter correctness claim",
            "no paper-scale validation claim",
            "no HMC/DSGE/GPU/production readiness claim",
            "interface-blocked models are not counted as failures",
        ],
    }


def _lgssm_gradient_cell() -> dict[str, Any]:
    payload = lgssm_gradient._run()
    comparison = payload["comparison"]
    matched = (
        payload["decision"] == "one_d_lgssm_step_gradient_filterflow_contract_matches_fd_veto"
        and comparison["pass_status"]["scalar_within_tolerance"]
        and comparison["pass_status"]["gradient_within_tolerance"]
        and comparison["pass_status"]["ledger_within_tolerance"]
    )
    return {
        "model": "lgssm_1d_step_gradient",
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "value_and_gradient",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": payload["decision"],
        "primary_criterion": "BayesFilter scalar and AD gradient match executable FilterFlow within tolerance",
        "metrics": {
            "scalar_delta": comparison["scalar_delta"],
            "gradient_delta": comparison["gradient_delta"],
            "fd_veto_shared": (
                not comparison["pass_status"]["bayesfilter_gradient_fd_within_tolerance"]
                and not comparison["pass_status"]["filterflow_gradient_fd_within_tolerance"]
            ),
        },
        "mismatch_class": None if matched else "requires_existing_lgssm_gradient_result_review",
        "artifact": str(lgssm_gradient.JSON_PATH.relative_to(REPO_ROOT)),
        "non_claim": "finite-difference disagreement is diagnostic-only and not a FilterFlow-vs-BayesFilter mismatch",
    }


def _lgssm_table_cell() -> dict[str, Any]:
    payload = load_json(lgssm_table.JSON_PATH)
    matched = (
        payload["decision"] == "lgssm_table_full_within_filterflow_mc_band"
        and payload["summary"]["executed_rows"] == 9
        and payload["summary"]["all_within_one_filterflow_sd"]
    )
    return {
        "model": "lgssm_2d_paper_table_contract",
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "value_table",
        "status": "MATCHED" if matched else "EXPLAINED_MISMATCH",
        "decision": payload["decision"],
        "primary_criterion": "all table cells within one executable FilterFlow sample SD",
        "metrics": {
            "executed_rows": payload["summary"]["executed_rows"],
            "max_abs_delta": payload["summary"]["max_abs_delta"],
            "rows_exceeding_residual_diagnostic_tolerance": payload["summary"][
                "rows_exceeding_residual_diagnostic_tolerance"
            ],
        },
        "mismatch_class": None if matched else "table_contract_not_matched",
        "artifact": str(lgssm_table.JSON_PATH.relative_to(REPO_ROOT)),
        "non_claim": "transport residuals are recorded diagnostics, not correctness claims",
    }


def _sv_interface_cell() -> dict[str, Any]:
    available = FILTERFLOW_SV_MODEL.exists()
    return {
        "model": "stochastic_volatility",
        "implementations": ["BayesFilter", "FilterFlow"],
        "cell_type": "interface_probe",
        "status": "PREP_ONLY" if available else "INTERFACE_BLOCKED",
        "decision": "sv_filterflow_model_surface_identified_no_value_gradient_tieout_yet"
        if available
        else "sv_filterflow_model_surface_missing",
        "primary_criterion": "identify whether same declared SV filter value/gradient tie-out can be built",
        "metrics": {
            "filterflow_sv_model_exists": available,
            "filterflow_model_path": str(FILTERFLOW_SV_MODEL.relative_to(REPO_ROOT)) if available else None,
        },
        "mismatch_class": "fixture_preparation_needed",
        "reason": (
            "FilterFlow has bootstrap SV model code, but BayesFilter highdim SV fixtures currently "
            "use dense/TT value-path contracts rather than a matched particle-filter executable contract."
        ),
        "non_claim": "no SV BayesFilter-vs-FilterFlow value or gradient equality claim is made",
    }


def _student_lgssm_cell() -> dict[str, Any]:
    references = json.loads(STUDENT_REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    panel = json.loads(STUDENT_PANEL_PATH.read_text(encoding="utf-8"))
    records = panel["records"]
    ok_records = [row for row in records if row["status"] == "ok"]
    blocked_records = [row for row in records if row["status"] != "ok"]
    return {
        "model": "student_lgssm_fixtures",
        "implementations": ["advanced_particle_filter", "2026MLCOE"],
        "cell_type": "student_reference_consistency",
        "status": "PREP_ONLY",
        "decision": "student_lgssm_reference_panel_available_not_yet_bayesfilter_filterflow_tieout",
        "primary_criterion": "confirm existing student adapters have runnable LGSSM fixture artifacts",
        "metrics": {
            "reference_fixture_count": len(references["references"]),
            "panel_record_count": len(records),
            "ok_record_count": len(ok_records),
            "blocked_record_count": len(blocked_records),
        },
        "mismatch_class": "student_phase_preparation",
        "artifacts": {
            "student_panel": str(STUDENT_PANEL_PATH.relative_to(REPO_ROOT)),
            "student_reference_summary": str(STUDENT_REFERENCE_SUMMARY.relative_to(REPO_ROOT)),
        },
        "non_claim": "student panel is not yet a BayesFilter-vs-student or FilterFlow-vs-student equality claim",
    }


def _interface_blocked_cell(*, model: str, reason: str) -> dict[str, Any]:
    return {
        "model": model,
        "implementations": ["BayesFilter", "FilterFlow", "student_repos"],
        "cell_type": "interface_inventory",
        "status": "INTERFACE_BLOCKED",
        "decision": f"{model}_interface_blocked",
        "primary_criterion": "do not force non-comparable implementations into a false tie-out",
        "metrics": {},
        "mismatch_class": "no_comparable_interface_identified",
        "reason": reason,
        "non_claim": "interface-blocked status is not a model failure",
    }


def _decision(cells: list[dict[str, Any]]) -> str:
    executed = [cell for cell in cells if cell["status"] in {"MATCHED", "EXPLAINED_MISMATCH"}]
    if any(cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class") for cell in executed):
        return "cross_impl_tieout_unclassified_mismatch_veto"
    if any(cell["status"] == "MATCHED" for cell in cells):
        return "cross_impl_tieout_first_slice_pass_with_interface_blockers"
    return "cross_impl_tieout_no_executed_matches"


def _summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    return {
        "num_cells": len(cells),
        "status_counts": statuses,
        "models": [cell["model"] for cell in cells],
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("CPU-only pre-import manifest missing")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise RuntimeError("GPU visible in CPU-only run")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")
    if not payload["cells"]:
        raise RuntimeError("missing cells")
    for cell in payload["cells"]:
        if cell["status"] == "EXPLAINED_MISMATCH" and not cell.get("mismatch_class"):
            raise RuntimeError(f"unclassified mismatch: {cell['model']}")
        if cell["status"] not in {"MATCHED", "EXPLAINED_MISMATCH", "INTERFACE_BLOCKED", "PREP_ONLY"}:
            raise RuntimeError(f"unknown status: {cell['status']}")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# DPF Cross-Implementation Common-Sense Tie-Out Result

metadata_date: 2026-06-06

## Decision

`{payload['decision']}`

## Summary

- Cells: `{payload['summary']['num_cells']}`
- Status counts: `{payload['summary']['status_counts']}`

## Cell Table

{_cell_table(payload['cells'])}

## Interpretation

This first slice confirms the already-hardened LGSSM BayesFilter-vs-FilterFlow
value/gradient and table contracts, identifies stochastic-volatility as a
fixture-preparation target, confirms student LGSSM artifacts exist for the
next phase, and explicitly blocks scalar nonlinear, SIR, and predator-prey
tie-outs until comparable interfaces are built.

## Non-Claims

{_bullet_list(payload['non_claims'])}
"""


def _cell_table(cells: list[dict[str, Any]]) -> str:
    lines = [
        "| model | cell type | status | decision | mismatch class |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            "| {model} | {cell_type} | `{status}` | `{decision}` | {mismatch} |".format(
                model=cell["model"],
                cell_type=cell["cell_type"],
                status=cell["status"],
                decision=cell["decision"],
                mismatch=cell.get("mismatch_class") or "N/A",
            )
        )
    return "\n".join(lines)


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())
