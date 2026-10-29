#!/usr/bin/env python3
"""Apply an owner-authorized budget addition after the existing master drains.

Standard-library orchestration only; numerical source and saved rows are untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "scripts/run_surrogate_hmc_marginal_expanded.py"
REPORTER = ROOT / "scripts/summarize_surrogate_hmc_marginal_expanded.py"
PLAN = "docs/plans/surrogate-hmc-marginal-budget-extension-2026-10-07.md"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    spec = importlib.util.spec_from_file_location("budget_extension_master", MASTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.identity()


def process_matches(pid, script, output):
    if not pid:
        return False
    try:
        args = (Path("/proc") / str(pid) / "cmdline").read_bytes().decode().split("\0")
        index = args.index("--output")
        actual_output = Path(args[index + 1])
        if not actual_output.is_absolute():
            actual_output = ROOT / actual_output
        return (any(Path(arg).name == script.name for arg in args if arg)
                and actual_output.resolve() == output.resolve())
    except (OSError, UnicodeError, ValueError, IndexError):
        return False


def extend(output, record, original, additional_seconds):
    """Only change budget and status after a verified normal terminal boundary."""
    if not math.isfinite(additional_seconds) or additional_seconds <= 0:
        raise ValueError("additional seconds must be positive and finite")
    campaign = read(output / "campaign.json")
    if campaign["status"] != "INCOMPLETE":
        raise ValueError("only ordinary INCOMPLETE state can be resumed")
    if campaign["configuration"] != original["configuration"]:
        raise ValueError("configuration changed before budget activation")
    if (campaign["source_sha256"] != original["source_sha256"]
            or source_identity() != original["source_sha256"]):
        raise ValueError("source changed before budget activation")
    for attempt in campaign["attempts"]:
        if (attempt["status"] == "RUNNING"
                or process_matches(attempt.get("worker_pid"), MASTER, output)):
            raise ValueError("worker still running; cannot activate budget")
    write(record / "campaign-before-extension.json", campaign)
    for name in ("next-phase.json", "analysis.json"):
        if (output / name).exists():
            write(record / ("before-" + name), read(output / name))
    old_budget = campaign["configuration"]["budget_seconds"]
    new_budget = old_budget + additional_seconds
    campaign.setdefault("budget_extension_history", []).append(dict(
        activated_utc=utc(), previous_budget_seconds=old_budget,
        additional_seconds=additional_seconds, budget_seconds=new_budget,
        authorization="you have 24 hours more", record=str(record), plan=PLAN))
    campaign["configuration"]["budget_seconds"] = new_budget
    campaign["status"] = "RUNNING"
    write(output / "campaign.json", campaign)
    return campaign


def command(output, config):
    args = [sys.executable, str(MASTER), "--output", str(output),
            "--budget-seconds", str(config["budget_seconds"]),
            "--gpu", config["gpu"], "--scopes", ",".join(config["scopes"]),
            "--particle-ladder", ",".join(map(str, config["particles"]))]
    if config.get("prior_campaign"):
        args += ["--prior-campaign", config["prior_campaign"]]
    return args


def stop_reporter(pid, output):
    if not process_matches(pid, REPORTER, output):
        return
    os.kill(pid, signal.SIGTERM)
    end = time.monotonic() + 10
    while process_matches(pid, REPORTER, output) and time.monotonic() < end:
        time.sleep(.1)
    if process_matches(pid, REPORTER, output):
        raise RuntimeError("old reporter did not exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-seconds", type=float, required=True)
    parser.add_argument("--master-pid", type=int, required=True)
    parser.add_argument("--reporter-pid", type=int)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.record = args.record.resolve()
    if not math.isfinite(args.additional_seconds) or args.additional_seconds <= 0:
        parser.error("additional seconds must be positive and finite")
    original = read(args.output / "campaign.json")
    if source_identity() != original["source_sha256"]:
        raise ValueError("current source does not match the running campaign")
    args.record.mkdir(parents=True, exist_ok=False)
    write(args.record / "campaign-at-request.json", original)
    state = dict(
        status="WAITING_FOR_CURRENT_INVOCATION", requested_utc=utc(),
        authorization="you have 24 hours more", additional_seconds=args.additional_seconds,
        old_budget_seconds=original["configuration"]["budget_seconds"],
        new_budget_seconds=original["configuration"]["budget_seconds"] + args.additional_seconds,
        controller_pid=os.getpid(), old_master_pid=args.master_pid, plan=PLAN,
        command=[sys.executable, *sys.argv], environment=sys.executable,
        controller_device="CPU orchestration; no TensorFlow import or numerical kernels",
        worker_gpu=original["configuration"]["gpu"],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    status_path = args.record / "status.json"
    write(status_path, state)
    print(json.dumps(state), flush=True)

    def refresh():
        with (args.record / "reporter.log").open("a") as log:
            completed = subprocess.run(
                [sys.executable, str(REPORTER), "--output", str(args.output)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["reporter_exit_code"] = completed.returncode

    try:
        # The old invocation is already bounded by its original cumulative budget.
        wait_until = time.monotonic() + original["configuration"]["budget_seconds"] + 600
        while process_matches(args.master_pid, MASTER, args.output):
            if time.monotonic() >= wait_until:
                raise TimeoutError("current master failed to drain within its existing bound")
            time.sleep(10)
        current = read(args.output / "campaign.json")
        if current["status"] == "DEFINED_GRID_COMPLETE_DIMENSION_EXTENSION_PENDING":
            state.update(status="DEFINED_GRID_COMPLETE_NO_RESTART_NEEDED", ended_utc=utc())
            write(status_path, state)
            return
        stop_reporter(args.reporter_pid, args.output)
        campaign = extend(args.output, args.record, original, args.additional_seconds)
        child_command = command(args.output, campaign["configuration"])
        state.update(status="STARTING_EXTENDED_MASTER", activated_utc=utc(),
                     master_command=child_command)
        write(status_path, state)
        with (args.record / "master.log").open("w") as log:
            process = subprocess.Popen(child_command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="RUNNING_EXTENDED_MASTER", master_pid=process.pid)
            write(status_path, state)
            print(json.dumps({k: state[k] for k in ("status", "master_pid", "new_budget_seconds")}),
                  flush=True)
            next_report = 0.
            while True:
                if time.monotonic() >= next_report:
                    refresh()
                    write(status_path, state)
                    next_report = time.monotonic() + 60
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    pass
        refresh()
        current = read(args.output / "campaign.json")
        state.update(status="EXTENDED_INVOCATION_FINISHED" if code == 0 else "MASTER_REPAIR_REQUIRED",
                     exit_code=code, campaign_status=current["status"], ended_utc=utc())
        write(status_path, state)
        print(json.dumps({k: state[k] for k in ("status", "exit_code", "campaign_status")}), flush=True)
    except Exception as exc:
        state.update(status="REPAIR_REQUIRED", error=repr(exc), ended_utc=utc())
        write(status_path, state)
        raise


if __name__ == "__main__":
    main()
