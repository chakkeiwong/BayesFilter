#!/usr/bin/env python3
"""Observe frozen-design pricing; never start, stop, or settle numerical work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def service_state(unit: str) -> dict:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "ActiveState", "-p",
         "SubState", "-p", "ExecMainPID", "-p", "ExecMainStatus"],
        check=True, capture_output=True, text=True, timeout=10)
    return dict(line.split("=", 1) for line in result.stdout.splitlines()
                if "=" in line)


def snapshot(pricing_dir: Path, unit: str) -> dict:
    path = pricing_dir / "result.json"
    result = read_json(path)
    state = service_state(unit)
    rows = [] if result is None else result.get("rows", [])
    methods = []
    for row in rows:
        calls = [call for root in row.get("roots", []) for call in root.get("calls", [])]
        methods.append(dict(method=row.get("method"), status=row.get("status"),
            root_count=len(row.get("roots", [])),
            returned_calls=sum(call.get("status") == "returned" for call in calls),
            running_calls=sum(call.get("status") == "running" for call in calls),
            failed_calls=sum(call.get("status") not in {"returned", "running", "not_started"}
                             for call in calls)))
    active = state.get("ActiveState") in {"active", "activating", "reloading"}
    supervisor_done = result is not None and "supervisor_elapsed_seconds" in result
    terminal = not active and supervisor_done
    return dict(status="terminal_observed" if terminal else "monitoring" if active else
                "terminal_requires_inspection", updated_utc=datetime.now(timezone.utc).isoformat(),
        numerical_unit=unit, service_state=state,
        result_status=None if result is None else result.get("status"),
        result_error=None if result is None else result.get("error"),
        result_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            if path.exists() else None,
        elapsed_seconds=None if result is None else result.get("elapsed_seconds"),
        supervisor_elapsed_seconds=None if result is None else result.get("supervisor_elapsed_seconds"),
        methods=methods, numerical_launch_authority=False, numerical_stop_authority=False,
        settlement_authority=False, gpu_devices_intentionally_hidden=True)


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-dir", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    parser.add_argument("--max-wait-seconds", type=float, required=True)
    args = parser.parse_args()
    if not (0 < args.poll_seconds <= 300 and math.isfinite(args.max_wait_seconds)
            and args.max_wait_seconds > 0):
        parser.error("poll and lifetime must be positive and bounded")
    args.pricing_dir = args.pricing_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    while True:
        try:
            state = snapshot(args.pricing_dir, args.unit)
        except Exception as exc:
            state = dict(status="observer_error", updated_utc=datetime.now(timezone.utc).isoformat(),
                error_type=type(exc).__name__, error=str(exc), numerical_launch_authority=False,
                numerical_stop_authority=False, settlement_authority=False,
                gpu_devices_intentionally_hidden=True)
        state["observer_elapsed_seconds"] = time.monotonic() - started
        write(args.output_dir / "status.json", state)
        if state["status"] in {"terminal_observed", "terminal_requires_inspection", "observer_error"}:
            with (args.output_dir / "terminal-observation.json").open("x") as stream:
                stream.write(json.dumps(state, indent=2, sort_keys=True) + "\n")
            return
        if time.monotonic() - started >= args.max_wait_seconds:
            write(args.output_dir / "observer-limit.json", dict(
                status="observer_limit", updated_utc=datetime.now(timezone.utc).isoformat(),
                elapsed_seconds=time.monotonic() - started, numerical_worker_not_stopped=True))
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
