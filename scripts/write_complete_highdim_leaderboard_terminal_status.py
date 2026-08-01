#!/usr/bin/env python3
"""Write the detached complete-leaderboard supervisor terminal status."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--launch-root", required=True)
    parser.add_argument("--started-utc", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--final-message", required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args(argv)

    timed_out = args.exit_code in {124, 137, 143}
    payload = {
        "schema_version": (
            "bayesfilter.complete_highdim_leaderboard.detached_terminal_status.v1"
        ),
        "run_id": args.run_id,
        "supervisor_process_status": (
            "timed_out"
            if timed_out
            else ("codex_exit_zero" if args.exit_code == 0 else "codex_failed")
        ),
        "scientific_program_status": (
            "must_be_read_from_isolated_phase_results_and_final_message"
        ),
        "codex_exit_code": args.exit_code,
        "timed_out": timed_out,
        "timeout_seconds": args.timeout_seconds,
        "started_utc": args.started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root_inside_mount_namespace": args.root,
        "source_root_hidden_by_bind_mount": args.source_root,
        "launch_root": args.launch_root,
        "codex_events": args.events,
        "codex_stderr": args.stderr,
        "codex_final_message": args.final_message,
        "baseline_snapshot": args.baseline,
        "automatic_merge_performed": False,
        "commit_performed_by_wrapper": False,
        "push_performed_by_wrapper": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

