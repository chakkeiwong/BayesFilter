"""Record a completed repair phase and its reviewed next experiment design.

This is offline progress/accounting, not a scientific decision engine or an
approval system. Candidate and posterior failures do not prevent continuation.
The next design must explain their treatment before this command is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import os


def read(path):
    return json.loads(Path(path).read_text())


def refresh(progress_path, phase, *, ledger_path, result_path, next_phase=None,
            next_design_path=None, review_note, output_path):
    progress_path = Path(progress_path)
    progress = read(progress_path)
    ledger_path, result_path, output_path = map(Path, (ledger_path, result_path, output_path))
    ledger = read(ledger_path)
    if progress["active_phase"] != phase or phase not in progress["phases"]:
        raise ValueError("refresh must close the active phase")
    if not review_note.strip() or not result_path.is_file():
        raise ValueError("terminal result and skeptical review note are required")
    if ledger.get("terminal") is not True or ledger.get("invalid_artifacts") or ledger.get("outstanding_workers") or ledger.get("outstanding"):
        raise ValueError("finish the integrity audit and active work before phase refresh")
    budget = ledger["budget_seconds"]
    remaining = budget["remaining"]
    if budget["exceeded"] or any(not math.isfinite(v) or v < 0 for v in remaining.values()):
        raise ValueError("campaign budget exhausted or invalid")
    if next_phase is not None:
        if next_phase not in progress["phases"] or next_phase == phase:
            raise ValueError("unknown or repeated next phase")
        if next_design_path is None or not Path(next_design_path).is_file():
            raise ValueError("write the next experiment design before continuing")
        for key, device in (("cpu_limit", "cpu_reference"), ("gpu_limit", "gpu")):
            if progress["phases"][next_phase][key] > remaining[device]:
                raise ValueError("refresh next phase allocation to fit remaining budget")
    elif next_design_path is not None:
        raise ValueError("terminal program cannot have a next design without a phase")
    files = {"ledger": ledger_path, "result": result_path}
    if next_phase is not None:
        files["next_design"] = Path(next_design_path)
    record = {"schema": "bayesfilter.hmc_phase_refresh.v1", "completed_phase": phase,
        "next_phase": next_phase, "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": {key: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for key, p in files.items()},
        "remaining_worker_seconds": remaining, "skeptical_review": review_note,
        "candidate_failure_is_continuation_veto": False,
        "scientific_gaps_closed": progress["phases"][phase].get("scientific_gaps_closed", False)}
    with output_path.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")
    progress["phases"][phase].update(status="bounded_execution_complete",
        result_file=str(result_path), ledger_file=str(ledger_path), refresh_file=str(output_path))
    progress["opening_ledger"] = str(ledger_path)
    progress["active_phase"] = next_phase
    progress["status"] = "active" if next_phase is not None else "bounded_execution_complete"
    if next_phase is not None:
        progress.pop("completed_utc", None)
        progress.pop("completion_local_date", None)
        progress["phases"][next_phase].update(status="ready_to_execute", design_file=str(next_design_path))
    temporary = progress_path.with_suffix(".tmp")
    with temporary.open("x") as handle:
        json.dump(progress, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, progress_path)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress", type=Path)
    parser.add_argument("phase")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--next-phase")
    parser.add_argument("--next-design", type=Path)
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = refresh(args.progress, args.phase, ledger_path=args.ledger, result_path=args.result,
        next_phase=args.next_phase, next_design_path=args.next_design,
        review_note=args.review_note, output_path=args.output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
