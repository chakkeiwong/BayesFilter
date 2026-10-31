"""Seeded fixed-ancestor robustness diagnostic for BF/FilterFlow V2 tie-out."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import copy
import time
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
    EXPECTED_V2_MODEL_IDS,
    common_model_specs_v2,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_common_model_suite_v2_fixed_resampling_tf as fixed,
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


PLAN_PATH = "docs/plans/bayesfilter-dpf-bf-filterflow-final-comparison-closeout-and-robustness-plan-2026-06-07.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-bf-filterflow-final-comparison-closeout-and-robustness-result-2026-06-07.md"
JSON_PATH = OUTPUT_DIR / "dpf_bf_filterflow_seeded_ancestor_robustness_2026-06-07.json"
REPORT_PATH = REPORT_DIR / "dpf-bf-filterflow-seeded-ancestor-robustness-2026-06-07.md"
P1_MANIFEST_PATH = OUTPUT_DIR / "dpf_common_model_suite_v2_manifest_2026-06-07.json"
SEEDS = (1101, 2202, 3303)
VALUE_TOLERANCE = fixed.VALUE_TOLERANCE
LEDGER_TOLERANCE = fixed.LEDGER_TOLERANCE


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
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    p1_manifest = load_json(P1_MANIFEST_PATH)
    p1_ids = [row.get("model_id") for row in p1_manifest.get("rows", [])]
    if tuple(p1_ids) != EXPECTED_V2_MODEL_IDS:
        raise ValueError(f"P1 manifest model id gate failed: {p1_ids}")
    ready_ids = fixed._ready_ids_for_phase(p1_manifest, "P4_fixed_ancestor", "READY_FOR_P4")
    specs = common_model_specs_v2()
    filterflow_status = fixed._filterflow_checkout_manifest()
    rows = []
    for seed in SEEDS:
        contracts = [
            _seeded_contract(spec, seed)
            for spec in specs
            if spec.model_id in ready_ids
        ]
        filterflow_payload = fixed._filterflow_path_subprocess(contracts)
        contract_by_id = {contract["model_id"]: contract for contract in contracts}
        for spec in specs:
            if spec.model_id not in ready_ids:
                rows.append(
                    fixed._classified_cell(
                        spec,
                        "CONTRACT_BLOCKED",
                        "row not READY_FOR_P4 in P1 classification",
                    )
                    | {"seed": seed}
                )
                continue
            contract = contract_by_id[spec.model_id]
            bayesfilter = fixed._bayesfilter_path(spec, contract)
            filterflow = fixed._filterflow_cell(filterflow_payload, spec.model_id)
            cell = fixed._cell(spec, contract, bayesfilter, filterflow)
            cell["seed"] = seed
            cell["cell_type"] = "seeded_fixed_ancestor_robustness_diagnostic"
            cell["primary_criterion"] = (
                "diagnostic-only same-schedule branch replay; not a stochastic resampling gate"
            )
            cell["non_claim"] = (
                "seeded ancestor schedule agreement is not stochastic resampling distribution correctness"
            )
            rows.append(cell)
    decision = _decision(rows)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "question": (
            "Do BayesFilter and executable local FilterFlow-side adapters remain "
            "aligned when several seeded pseudo-stochastic ancestor schedules are "
            "frozen and replayed under the V2 fixed-ancestor contract?"
        ),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "p1_manifest_path": str(P1_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "filterflow_reference_policy": fixed.reference_policy(),
        "filterflow_status": filterflow_status,
        "seeds": list(SEEDS),
        "tolerances": {"value_abs": VALUE_TOLERANCE, "ledger_abs": LEDGER_TOLERANCE},
        "primary_criterion_fields": {
            "diagnostic_only": True,
            "ready_model_ids": sorted(ready_ids),
            "executed_rows": sum(1 for row in rows if row["status"] != "CONTRACT_BLOCKED"),
            "matched_executed_rows": sum(1 for row in rows if row["status"] == "MATCHED"),
            "all_executed_rows_matched": all(
                row["status"] == "MATCHED" for row in rows if row["status"] != "CONTRACT_BLOCKED"
            ),
            "schedule_policy": "seeded schedules are generated once, frozen, and replayed on both sides",
        },
        "veto_diagnostics": {
            "student_command_executed": False,
            "localsource_filterflow_mutated": False,
            "stochastic_distribution_claimed": False,
            "rng_equality_claimed": False,
            "finite_difference_used_as_gate": False,
            "missing_filterflow_subprocess_environment": filterflow_status.get("status") == "missing",
            "nonfinite_path_value": any(fixed._cell_nonfinite(row) for row in rows),
            "unclassified_mismatch": any(
                row["status"] not in {"MATCHED", "CONTRACT_BLOCKED", "INTERFACE_BLOCKED", "EXPLAINED_MISMATCH"}
                for row in rows
            ),
        },
        "explanatory_only_fields": {
            "status_counts": fixed._status_counts(rows),
            "max_abs_delta": fixed._max_abs_delta(rows),
            "seed_count": len(SEEDS),
            "model_count_per_seed": len(EXPECTED_V2_MODEL_IDS),
        },
        "cells": rows,
        "summary": {
            "num_cells": len(rows),
            "status_counts": fixed._status_counts(rows),
            "max_abs_delta": fixed._max_abs_delta(rows),
            "diagnostic_scope": "fixed seeded ancestor schedules only",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_bf_filterflow_seeded_ancestor_robustness_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_claims": [
            "no stochastic resampling distribution correctness claim",
            "no random-number-generator equality claim",
            "no differentiable-resampling or gradient-through-ancestor-selection claim",
            "no filtering-algorithm correctness proof",
            "no student-repository tie-out claim",
        ],
    }


def _seeded_contract(spec: Any, seed: int) -> dict[str, Any]:
    contract = fixed._path_contract(spec)
    schedule = _seeded_schedule(spec, contract, seed)
    contract["resampling_flags"] = schedule["resampling_flags"]
    contract["fixed_ancestor_indices"] = schedule["fixed_ancestor_indices"]
    contract["expected_resampling_count"] = int(sum(schedule["resampling_flags"]))
    contract["resampling_policy"] = "seeded_fixed_ancestor_replay_before_proposal"
    contract["diagnostic_seed"] = int(seed)
    contract["schedule_source"] = "tf.random.stateless_uniform converted to frozen ancestor indices"
    contract["contract_checksum"] = stable_digest(contract)
    return contract


def _seeded_schedule(spec: Any, contract: dict[str, Any], seed: int) -> dict[str, Any]:
    horizon = int(contract["horizon"])
    num_particles = int(contract["num_particles"])
    spec_offset = sum(ord(char) for char in spec.model_id) % 997
    raw = tf.random.stateless_uniform(
        [horizon, num_particles],
        seed=tf.constant([seed, spec_offset], dtype=tf.int32),
        minval=0,
        maxval=num_particles,
        dtype=tf.int32,
    ).numpy().tolist()
    step_index = (seed + spec_offset) % horizon
    flags = [False for _ in range(horizon)]
    flags[step_index] = True
    return {
        "resampling_flags": flags,
        "fixed_ancestor_indices": [int(value) for value in raw[step_index]],
    }


def _decision(rows: list[dict[str, Any]]) -> str:
    executed = [row for row in rows if row["status"] != "CONTRACT_BLOCKED"]
    if executed and all(row["status"] == "MATCHED" for row in executed):
        return "PASS_SEEDED_ANCESTOR_DIAGNOSTIC"
    if any(row["status"] == "INTERFACE_BLOCKED" for row in rows):
        return "BLOCKED_SEEDED_ANCESTOR_DIAGNOSTIC_INTERFACE"
    return "CLASSIFIED_SEEDED_ANCESTOR_DIAGNOSTIC_MISMATCH"


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise ValueError("CPU-only pre-import invariant failed")
    if payload["run_manifest"]["gpu_devices_visible"] != []:
        raise ValueError("GPU devices visible in CPU-only diagnostic")
    if payload["decision"] not in {
        "PASS_SEEDED_ANCESTOR_DIAGNOSTIC",
        "BLOCKED_SEEDED_ANCESTOR_DIAGNOSTIC_INTERFACE",
        "CLASSIFIED_SEEDED_ANCESTOR_DIAGNOSTIC_MISMATCH",
    }:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    if payload["veto_diagnostics"]["student_command_executed"]:
        raise ValueError("student command was executed")
    if payload["veto_diagnostics"]["localsource_filterflow_mutated"]:
        raise ValueError(".localsource/filterflow mutation recorded")
    executed = [row for row in payload["cells"] if row["status"] != "CONTRACT_BLOCKED"]
    if payload["decision"] == "PASS_SEEDED_ANCESTOR_DIAGNOSTIC":
        for row in executed:
            if row["status"] != "MATCHED":
                raise ValueError(f"nonmatched executed row in PASS payload: {row['model']} seed {row['seed']}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# BF/FilterFlow Seeded-Ancestor Robustness Diagnostic",
        "",
        f"metadata_date: 2026-06-07",
        f"decision: {payload['decision']}",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Evidence Contract",
        "",
        "Primary status: diagnostic-only branch robustness.  The schedule is seeded, frozen, and replayed on both sides.  This is not RNG equality and not stochastic-resampling distribution correctness.",
        "",
        "## Result",
        "",
        f"- JSON artifact: `{JSON_PATH.relative_to(REPO_ROOT)}`",
        f"- Reproducibility digest: `{payload['reproducibility_digest']}`",
        "",
        "## Summary",
        "",
        f"- status counts: `{payload['summary']['status_counts']}`",
        f"- max abs delta: `{payload['summary']['max_abs_delta']}`",
        f"- seeds: `{payload['seeds']}`",
        "",
        "## Cells",
        "",
        "| Seed | Model | Status | Max abs delta |",
        "|---:|---|---|---:|",
    ]
    for row in payload["cells"]:
        lines.append(
            f"| `{row['seed']}` | `{row['model']}` | `{row['status']}` | "
            f"`{row.get('metrics', {}).get('max_abs_delta', 'N/A')}` |"
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["veto_diagnostics"].items()],
            "",
            "## Command Manifest",
            "",
            "| Field | Value |",
            "|---|---|",
            f"| git commit | `{payload['run_manifest'].get('commit')}` |",
            f"| command | `{payload['run_manifest'].get('command')}` |",
            f"| CPU/GPU status | CPU-only TensorFlow; pre-import `CUDA_VISIBLE_DEVICES={payload['run_manifest'].get('pre_import_cuda_visible_devices')}`; visible GPUs `{payload['run_manifest'].get('gpu_devices_visible')}` |",
            f"| random seeds | `{payload['seeds']}`; used only to generate frozen ancestor schedules |",
            "| dtype | `tf.float64` |",
            "",
            "## Non-Claims",
            "",
            *[f"- {claim}" for claim in payload["non_claims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = copy.deepcopy(payload)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
