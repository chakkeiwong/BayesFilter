#!/usr/bin/env python3
"""Supervise concurrent seed A/B q=20 NeuTra continuation processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_budgeted_continuation_2026_08_06.py"
)
PLAN = ROOT / (
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-budgeted-continuation-plan-2026-08-06.md"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_neutra_budgeted_continuation_supervisor.v1"
MAX_CAP_SECONDS = 43200.0
SEEDS = {
    "seed-a": {
        "cpus": tuple(range(0, 25)),
        "resume": ROOT
        / (
            "docs/plans/artifacts/"
            "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
            "seed-a/seed-a/checkpoint-1500.json"
        ),
    },
    "seed-b": {
        "cpus": tuple(range(25, 50)),
        "resume": ROOT
        / (
            "docs/plans/artifacts/"
            "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
            "seed-b/seed-b/checkpoint-2500.json"
        ),
    },
}


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RuntimeError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def terminate_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return


def terminal_row(
    label: str,
    process: subprocess.Popen[str],
    output: Path,
    log_path: Path,
) -> Mapping[str, Any]:
    result_path = output / label / "result.json"
    manifest_path = output / label / "run-manifest.json"
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    tail = ""
    if log_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "result_status": None if result is None else result.get("status"),
        "terminal_optimizer_step": (
            None if result is None else result.get("terminal_optimizer_step")
        ),
        "selected_continuation_update": (
            None
            if result is None
            else result.get("selection", {}).get("selected_continuation_update")
        ),
        "audit_mean_loss": (
            None if result is None else result.get("audit", {}).get("mean_loss")
        ),
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "result_sha256": sha256(result_path) if result_path.is_file() else None,
        "manifest_status": None if manifest is None else manifest.get("status"),
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "log_path": log_path.relative_to(ROOT).as_posix(),
        "output_tail": tail,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise RuntimeError("output root must be inside the repository")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("supervisor output root must be new or empty")
    available = set(os.sched_getaffinity(0))
    required = {cpu for row in SEEDS.values() for cpu in row["cpus"]}
    if not required.issubset(available):
        raise RuntimeError("supervisor affinity does not expose CPUs 0..49")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    processes: dict[str, subprocess.Popen[str]] = {}
    handles: dict[str, Any] = {}
    logs: dict[str, Path] = {}
    commands: dict[str, list[str]] = {}
    for label, row in SEEDS.items():
        child_output = output / label
        cpu_spec = f"{row['cpus'][0]}-{row['cpus'][-1]}"
        command = [
            "taskset",
            "-c",
            cpu_spec,
            sys.executable,
            str(CHILD),
            "--stream",
            label,
            "--cpu-processes",
            "25",
            "--batch-per-process",
            "4",
            "--resume-checkpoint",
            str(row["resume"]),
            "--output-root",
            child_output.relative_to(ROOT).as_posix(),
            "--cap-seconds",
            str(args.cap_seconds),
        ]
        if args.canary_updates is not None:
            command.extend(("--canary-updates", str(args.canary_updates)))
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "1",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true",
                "TF_CPP_MIN_LOG_LEVEL": "1",
            }
        )
        log_path = output / f"{label}-child.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        commands[label] = command
        logs[label] = log_path
        handles[label] = handle
        processes[label] = process
    start_payload = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_at_utc": started_utc,
        "started_unix_seconds": time.time(),
        "cap_seconds": float(args.cap_seconds),
        "canary_updates": args.canary_updates,
        "physical_gpu": 1,
        "cuda_visible_devices_per_child": "1",
        "tf_force_gpu_allow_growth_per_child": "true",
        "cpu_ranges": {
            label: list(row["cpus"]) for label, row in SEEDS.items()
        },
        "commands": commands,
        "child_pids": {label: process.pid for label, process in processes.items()},
        "child_script": CHILD.relative_to(ROOT).as_posix(),
        "child_script_sha256": sha256(CHILD),
        "plan": PLAN.relative_to(ROOT).as_posix(),
        "plan_sha256": sha256(PLAN),
        "nonclaims": [
            "supervisor and bounded training provenance only",
            "no HMC, convergence, posterior, default, superiority, or scientific claim",
        ],
    }
    write_json(output / "supervisor-start.json", start_payload)
    active = dict(processes)
    cap_fired = False
    try:
        while active:
            if time.perf_counter() - started >= args.cap_seconds:
                cap_fired = True
                for process in active.values():
                    terminate_group(process, signal.SIGTERM)
                break
            for label, process in tuple(active.items()):
                if process.poll() is not None:
                    handles[label].close()
                    del active[label]
            if active:
                time.sleep(1.0)
    except BaseException:
        for process in active.values():
            terminate_group(process, signal.SIGTERM)
        raise
    finally:
        for label, process in active.items():
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                terminate_group(process, signal.SIGKILL)
                process.wait()
            handles[label].close()
        for label, process in processes.items():
            if process.poll() is None:
                process.wait()
            if not handles[label].closed:
                handles[label].close()

    terminal = {
        label: terminal_row(label, process, output, logs[label])
        for label, process in processes.items()
    }
    expected_status = (
        "GPU_CONTINUATION_CANARY_COMPLETED"
        if args.canary_updates is not None
        else "GPU_CONTINUATION_COMPLETED_CANDIDATE_NOMINATED"
    )
    success = not cap_fired and all(
        row["exit_code"] == 0
        and row["result_status"] == expected_status
        and row["manifest_status"] == "FINISHED"
        for row in terminal.values()
    )
    result = {
        **start_payload,
        "status": "COMPLETED" if success else "INCOMPLETE_OR_VETOED",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "cap_fired": cap_fired,
        "expected_child_status": expected_status,
        "terminal": terminal,
        "decision": {
            "primary_criterion_status": "completed" if success else "not_met",
            "veto_diagnostic_status": "clear" if success else "inspect_terminal_rows",
            "main_uncertainty": "downstream HMC geometry is not evaluated",
            "next_justified_action": (
                "fresh per-seed HMC retuning"
                if success and args.canary_updates is None
                else "inspect the smallest failed child diagnostic"
            ),
            "not_concluded": [
                "NeuTra convergence",
                "posterior correctness",
                "HMC readiness",
                "statistical superiority",
            ],
        },
        "inference_status": {
            "hard_veto_screen": "child numerical/device/resource/artifact checks",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": ["loss", "runtime", "selected checkpoint"],
            "default_readiness": "not evaluated",
            "next_evidence_needed": "fresh sequential fixed-HMC validation",
        },
    }
    write_json(output / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=MAX_CAP_SECONDS)
    parser.add_argument("--canary-updates", type=int)
    args = parser.parse_args()
    if not math.isfinite(args.cap_seconds) or not 0.0 < args.cap_seconds <= MAX_CAP_SECONDS:
        parser.error(f"--cap-seconds must be in (0,{MAX_CAP_SECONDS:g}]")
    if args.canary_updates is not None and not 1 <= args.canary_updates < 500:
        parser.error("--canary-updates must be in [1,499]")
    result = run(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "elapsed_seconds": result["elapsed_seconds"],
                "terminal": result["terminal"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
