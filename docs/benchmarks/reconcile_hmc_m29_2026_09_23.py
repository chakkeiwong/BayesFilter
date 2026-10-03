"""Reconcile the bounded M29 profiling campaign from outer attempt receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


CAP = {"cpu_reference": 1800.0, "gpu": 4800.0}
# File inspection, checkpointing, and source-overlay bookkeeping were not
# launched as metered workers. This is a conservative, explicitly recorded
# allowance rather than scientific compute.
PLANNING_ALLOWANCE = {"cpu_reference": 60.0, "gpu": 0.0}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reconcile(root: Path, *, terminal: bool) -> dict:
    opening_file = root.parent / "m28-r1/reconciliation-terminal.json"
    opening = read(opening_file)["remaining_budget_seconds"]
    rows = []
    for receipt in sorted(root.glob("*/execution.json")):
        value = read(receipt)
        rows.append(
            {
                "attempt": receipt.parent.name,
                "receipt": str(receipt),
                "cpu_reference": float(value.get("cpu_worker_seconds", 0.0)),
                "gpu": float(value.get("gpu_worker_seconds", 0.0)),
                "exit_code": value.get("exit_code"),
                "launch_failure": value.get("launch_failure"),
            }
        )
    pending = [
        str(path.parent)
        for path in sorted(root.glob("*/manifest.json"))
        if not (path.parent / "execution.json").exists()
    ]
    charged = {
        key: sum(row[key] for row in rows) + PLANNING_ALLOWANCE[key]
        for key in CAP
    }
    assert all(charged[key] <= CAP[key] for key in CAP), charged
    if terminal:
        assert not pending, pending
    return {
        "schema": "bayesfilter.hmc_m29_reconciliation.v1",
        "phase": "M29",
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "opening_ledger": "docs/plans/artifacts/hmc-repair-master-2026-09-16/m28-r1/reconciliation-terminal.json",
        "opening_remaining_seconds": opening,
        "phase_budget_seconds": CAP,
        "attempts": rows,
        "attempt_count": len(rows),
        "planning_allowance_seconds": PLANNING_ALLOWANCE,
        "charged_seconds": charged,
        "remaining_phase_seconds": {
            key: CAP[key] - charged[key] for key in CAP
        },
        "remaining_budget_seconds": {
            key: opening[key] - charged[key] for key in CAP
        },
        "pending_attempt_receipts": pending,
        "all_phase_workers_terminal": terminal and not pending,
        "double_count_rule": "Outer execution receipts only; nested children and profiler work are included in their outer receipt.",
        "failures_included": True,
        "terminal_requested": terminal,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terminal", action="store_true")
    args = parser.parse_args()
    # Run after all metered workers end. This short bookkeeping command is
    # included in PLANNING_ALLOWANCE; metering it inside its own receipt set
    # would make a terminal reconciliation see its own pending receipt.
    result = reconcile(args.root.resolve(), terminal=args.terminal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "attempt_count": result["attempt_count"],
                "charged_seconds": result["charged_seconds"],
                "remaining_budget_seconds": result["remaining_budget_seconds"],
                "pending_attempt_receipts": result["pending_attempt_receipts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
