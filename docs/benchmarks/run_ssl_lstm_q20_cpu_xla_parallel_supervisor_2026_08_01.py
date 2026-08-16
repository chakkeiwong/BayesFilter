#!/usr/bin/env python3
"""Run seed A and seed B q=20 CPU-XLA diagnostics concurrently."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "docs/benchmarks/run_ssl_lstm_q20_cpu_xla_parallel_training_2026_08_01.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-parallel-training-plan-2026-08-01.md"
SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_parallel_supervisor.v1"
SEED_RANGES = {"seed-a": (0, 24), "seed-b": (25, 49)}


def canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(payload), encoding="ascii")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def run(args: argparse.Namespace) -> int:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise RuntimeError("output root must be inside the repository")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("supervisor output root must be new or empty")
    available = set(os.sched_getaffinity(0))
    required = set(range(50))
    if not required.issubset(available):
        raise RuntimeError("host affinity does not expose CPUs 0..49")
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > 40000.0:
        raise RuntimeError("cap-seconds must be in (0,40000]")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    children: dict[str, subprocess.Popen[str]] = {}
    all_children: dict[str, subprocess.Popen[str]] = {}
    log_handles: dict[str, Any] = {}
    exit_codes: dict[str, int | None] = {}
    commands: dict[str, list[str]] = {}
    for label, (first, last) in SEED_RANGES.items():
        seed_root = output / label
        cpu_range = f"{first}-{last}"
        command = [
            "taskset", "-c", cpu_range, sys.executable, str(CHILD),
            "--stream", label,
            "--cpu-processes", "25",
            "--batch-per-process", "4",
            "--output-root", str(seed_root.relative_to(ROOT)),
            "--cap-seconds", str(args.cap_seconds),
        ]
        if args.debug_stop_after_steps is not None:
            command.extend(["--debug-stop-after-steps", str(args.debug_stop_after_steps)])
        commands[label] = command
        log_path = output / f"{label}-supervisor-child.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        children[label] = process
        all_children[label] = process
        log_handles[label] = log_handle
        exit_codes[label] = None
    write_json(
        output / "supervisor-start.json",
        {
            "schema": SCHEMA,
            "status": "RUNNING",
            "started_unix_seconds": time.time(),
            "cap_seconds": args.cap_seconds,
            "cpu_ranges": {label: list(range(first, last + 1)) for label, (first, last) in SEED_RANGES.items()},
            "commands": commands,
            "plan": str(PLAN.relative_to(ROOT)),
            "child_script": str(CHILD.relative_to(ROOT)),
            "nonclaims": ["supervisor provenance only", "no GPU/default/HMC claim"],
        },
    )
    outputs: dict[str, str] = {label: "" for label in children}
    while children:
        if time.perf_counter() - started >= args.cap_seconds:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
            break
        for label, process in list(children.items()):
            if process.poll() is None:
                continue
            exit_codes[label] = process.returncode
            log_handles[label].close()
            log_path = output / f"{label}-supervisor-child.log"
            outputs[label] = log_path.read_text(encoding="utf-8", errors="replace")
            del children[label]
        if children:
            time.sleep(1.0)
    for label, process in children.items():
        if process.poll() is None:
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        exit_codes[label] = process.returncode
        log_handles[label].close()
        log_path = output / f"{label}-supervisor-child.log"
        outputs[label] = log_path.read_text(encoding="utf-8", errors="replace")
    child_summaries = {
        label: read_json(output / label / "summary.json") for label in SEED_RANGES
    }
    child_results = {
        label: (
            read_json(output / label / label / "debug-smoke-result.json")
            if args.debug_stop_after_steps is not None
            else read_json(output / label / label / "result.json")
        )
        for label in SEED_RANGES
    }
    # Derive terminal status from both process exit codes and child artifacts.
    terminal = {
        label: {
            "pid": all_children[label].pid,
            "exit_code": exit_codes[label],
            "summary_status": None if child_summaries[label] is None else child_summaries[label].get("status"),
            "result_status": None if child_results[label] is None else child_results[label].get("status"),
            "terminal_program_step": None if child_results[label] is None else child_results[label].get("terminal_program_step"),
            "output_tail": outputs[label][-4000:],
        }
        for label in SEED_RANGES
    }
    if args.debug_stop_after_steps is None:
        expected_summary_statuses = {"CPU_DIAGNOSTIC_COMPLETED", "CPU_XLA_DIAGNOSTIC_COMPLETED"}
        expected_result_statuses = {"CPU_XLA_DIAGNOSTIC_SCREEN_PASSED", "CPU_DIAGNOSTIC_SCREEN_PASSED"}
    else:
        expected_summary_statuses = {"CPU_DEBUG_SMOKE_COMPLETED"}
        expected_result_statuses = {"CPU_DEBUG_SMOKE_COMPLETED"}
    success = all(
        row["exit_code"] == 0
        and row["summary_status"] in expected_summary_statuses
        and row["result_status"] in expected_result_statuses
        for row in terminal.values()
    )
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED" if success else "INCOMPLETE_OR_VETOED",
        "elapsed_seconds": time.perf_counter() - started,
        "cap_seconds": args.cap_seconds,
        "debug_stop_after_steps": args.debug_stop_after_steps,
        "cpu_ranges": {label: list(range(first, last + 1)) for label, (first, last) in SEED_RANGES.items()},
        "terminal": terminal,
        "child_summary_paths": {label: str((output / label / "summary.json").relative_to(ROOT)) for label in SEED_RANGES},
        "child_result_paths": {
            label: str((output / label / label / "debug-smoke-result.json" if args.debug_stop_after_steps is not None else output / label / label / "result.json").relative_to(ROOT))
            for label in SEED_RANGES
        },
        "plan": str(PLAN.relative_to(ROOT)),
        "nonclaims": ["parallel CPU-XLA diagnostic only", "no GPU/default/HMC/scientific claim"],
    }
    write_json(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "terminal": terminal}, sort_keys=True))
    return 0 if success else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=40000.0)
    parser.add_argument("--debug-stop-after-steps", type=int)
    args = parser.parse_args()
    if args.debug_stop_after_steps is not None and not 0 < args.debug_stop_after_steps < 250:
        parser.error("--debug-stop-after-steps must be in [1, 249]")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
