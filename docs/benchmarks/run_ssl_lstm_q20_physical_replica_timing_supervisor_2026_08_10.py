#!/usr/bin/env python3
"""Supervise bounded sequential physical replica timing children."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_replica_timing_supervisor_2026_08_10.py"
)
CHILD = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_replica_timing_2026_08_10.py"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-"
    "2026-08-10/r1-timing"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "timing.json"
LOG = OUTPUT_ROOT / "supervisor.log"
CHILD_CAP_SECONDS = 2400.0
CAMPAIGN_CAP_SECONDS = 5000.0
TOPOLOGIES = (
    {"label": "threads-04", "threads": 4, "cpu_ids": tuple(range(32, 36))},
    {"label": "threads-32", "threads": 32, "cpu_ids": tuple(range(32, 64))},
)


class ReplicaTimingSupervisorError(RuntimeError):
    """Raised when a timing child or campaign bound fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise ReplicaTimingSupervisorError(f"refusing to overwrite: {path}")
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _child_root(spec: Mapping[str, Any]) -> Path:
    return OUTPUT_ROOT / str(spec["label"])


def _command(spec: Mapping[str, Any]) -> list[str]:
    return [
        "taskset",
        "-c",
        ",".join(str(value) for value in spec["cpu_ids"]),
        "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
        CHILD.as_posix(),
        "--label",
        str(spec["label"]),
        "--threads",
        str(int(spec["threads"])),
        "--output-root",
        _child_root(spec).as_posix(),
    ]


def _run_child(spec: Mapping[str, Any], log: Any) -> float:
    started = time.perf_counter()
    process = subprocess.Popen(
        _command(spec),
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=CHILD_CAP_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise ReplicaTimingSupervisorError(f"child timeout: {spec['label']}")
    if return_code != 0:
        raise ReplicaTimingSupervisorError(
            f"child failed with code {return_code}: {spec['label']}"
        )
    return time.perf_counter() - started


def run_supervisor() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise ReplicaTimingSupervisorError("refusing to overwrite timing campaign")
    _abs(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    _write_json(
        PROGRESS,
        {
            "status": "PHYSICAL_REPLICA_TIMING_RUNNING",
            "completed_topologies": [],
            "total_topologies": len(TOPOLOGIES),
        },
        overwrite=True,
    )
    completed = []
    runtimes = {}
    children = []
    try:
        with _abs(LOG).open("a", encoding="utf-8") as log:
            for spec in TOPOLOGIES:
                if time.perf_counter() - started >= CAMPAIGN_CAP_SECONDS:
                    raise ReplicaTimingSupervisorError("campaign wall cap reached")
                runtimes[str(spec["label"])] = _run_child(spec, log)
                child_path = _child_root(spec) / "timing.json"
                child = json.loads(_abs(child_path).read_text(encoding="utf-8"))
                if child.get("status") != "PHYSICAL_REPLICA_TIMING_CHILD_PASSED":
                    raise ReplicaTimingSupervisorError(f"child did not pass: {child_path}")
                completed.append(str(spec["label"]))
                children.append(child)
                _write_json(
                    PROGRESS,
                    {
                        "status": "PHYSICAL_REPLICA_TIMING_RUNNING",
                        "completed_topologies": completed,
                        "total_topologies": len(TOPOLOGIES),
                        "elapsed_seconds": time.perf_counter() - started,
                        "last_child_sha256": _sha(child_path),
                    },
                    overwrite=True,
                )
        four, thirty_two = children
        four_cached = float(four["call_1_cached"]["seconds_per_transition"])
        thirty_two_cached = float(
            thirty_two["call_1_cached"]["seconds_per_transition"]
        )
        gates = {
            "both_children_passed": len(children) == 2,
            "both_children_one_xla_trace": all(
                int(child["sampler_tracing_count"]) == 1 for child in children
            ),
            "both_children_terminal_target_status_valid": all(
                bool(child["terminal_target_status_all_valid"])
                for child in children
            ),
            "campaign_within_5000_seconds": (
                time.perf_counter() - started <= CAMPAIGN_CAP_SECONDS
            ),
        }
        payload = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_replica_timing.v1",
            "status": (
                "PHYSICAL_REPLICA_TIMING_PASSED"
                if all(gates.values())
                else "PHYSICAL_REPLICA_TIMING_FAILED"
            ),
            "role": "compile_cached_cost_and_cpu_topology_diagnostic_only",
            "gates": gates,
            "children": [
                {
                    "label": child["label"],
                    "path": (_child_root(spec) / "timing.json").as_posix(),
                    "sha256": _sha(_child_root(spec) / "timing.json"),
                    "supervisor_child_seconds": runtimes[str(spec["label"])],
                    "compile_inclusive_seconds": child["call_0_compile_inclusive"]["elapsed_seconds"],
                    "cached_seconds_per_transition": child["call_1_cached"]["seconds_per_transition"],
                }
                for spec, child in zip(TOPOLOGIES, children)
            ],
            "descriptive_comparison": {
                "threads_04_cached_seconds_per_transition": four_cached,
                "threads_32_cached_seconds_per_transition": thirty_two_cached,
                "threads_04_over_threads_32_cached_ratio": (
                    four_cached / thirty_two_cached
                ),
                "faster_observed_topology": (
                    "threads-04" if four_cached < thirty_two_cached else "threads-32"
                ),
            },
            "run_manifest": {
                "git_commit": subprocess.run(
                    ("git", "rev-parse", "HEAD"),
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip(),
                "git_dirty": bool(
                    subprocess.run(
                        ("git", "status", "--porcelain"),
                        cwd=ROOT,
                        check=True,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                ),
                "command": " ".join(sys.argv),
                "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
                "python": platform.python_version(),
                "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
                "wall_time_seconds": time.perf_counter() - started,
                "artifact_root": OUTPUT_ROOT.as_posix(),
                "plan_file": PLAN.as_posix(),
                "result_file": RESULT.as_posix(),
                "source_sha256": {
                    "plan": _sha(PLAN),
                    "supervisor": _sha(RUNNER),
                    "child": _sha(CHILD),
                },
            },
            "nonclaims": (
                "one cached observation per topology is descriptive only",
                "no statistical topology ranking",
                "no travel, convergence, posterior, weight, or predictive claim",
            ),
        }
        if not all(gates.values()):
            raise ReplicaTimingSupervisorError(f"timing gates failed: {gates}")
        _write_json(FINAL, payload)
        _write_json(
            PROGRESS,
            {
                "status": payload["status"],
                "completed_topologies": completed,
                "total_topologies": len(TOPOLOGIES),
                "elapsed_seconds": time.perf_counter() - started,
                "result": FINAL.as_posix(),
            },
            overwrite=True,
        )
        return payload
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_replica_timing.failure.v1",
            "status": "PHYSICAL_REPLICA_TIMING_HARNESS_FAILED",
            "completed_topologies": completed,
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_time_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        raise


if __name__ == "__main__":
    result = run_supervisor()
    print(json.dumps({"status": result["status"]}, sort_keys=True))
