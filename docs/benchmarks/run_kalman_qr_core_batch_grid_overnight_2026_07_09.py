#!/usr/bin/env python
"""Run the Kalman QR core/batch benchmark grid sequentially.

This runner is intentionally conservative: CPU artifacts are run one at a time
so the requested TensorFlow thread settings are not confounded by concurrent
jobs.  GPU runs are launched only if the trusted GPU preflight passes for the
same three-arm harness.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
BENCH = REPO_ROOT / "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
BENCH_DIR = REPO_ROOT / "docs/benchmarks"
LOG_DIR = BENCH_DIR / "logs"
STATUS_PATH = BENCH_DIR / "kalman_qr_core_batch_grid_overnight_status_2026-07-09.json"

DIMENSIONS = ["10", "20", "30"]
PARAMETER_COUNTS = ["50", "150"]
BATCH_SIZES = [1, 4, 16]
CPU_THREADS = [1, 4, 16]
DTYPES = ["float32", "float64"]


def write_status(payload: dict) -> None:
    payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_command(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str],
    status: dict,
    record: dict,
) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record["command"] = " ".join(command)
    record["log_path"] = str(log_path.relative_to(REPO_ROOT))
    record["started_utc"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "running"
    write_status(status)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    record["returncode"] = completed.returncode
    record["elapsed_seconds"] = time.perf_counter() - started
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "passed" if completed.returncode == 0 else "failed"
    write_status(status)
    return completed.returncode


def benchmark_command(
    *,
    dimensions: list[str],
    parameter_counts: list[str],
    batch_size: int,
    device: str,
    dtype: str,
    output_json: Path,
    output_md: Path,
    cpu_threads: int | None = None,
) -> list[str]:
    command = [
        str(PYTHON),
        str(BENCH),
        "--dimensions",
        *dimensions,
        "--parameter-counts",
        *parameter_counts,
        "--timesteps",
        "120",
        "--repeats",
        "1",
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--jit-compile",
        "--dtype",
        dtype,
        "--isolate-each-row",
        "--row-subprocess-timeout-seconds",
        "3600",
        "--output-json",
        str(output_json.relative_to(REPO_ROOT)),
        "--output-md",
        str(output_md.relative_to(REPO_ROOT)),
    ]
    if cpu_threads is not None:
        command.extend(["--cpu-threads", str(cpu_threads)])
    return command


def main() -> int:
    status: dict = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "nonclaims": [
            "descriptive timing only",
            "no statistically supported ranking",
            "no HMC readiness claim",
            "no posterior correctness claim",
            "no production default change",
        ],
        "cpu": [],
        "gpu_preflights": [],
        "gpu": [],
    }
    write_status(status)

    preflight_env = os.environ.copy()
    preflight_env.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
    gpu_preflight_json = BENCH_DIR / "kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.json"
    gpu_preflight_md = BENCH_DIR / "kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.md"
    gpu_preflight_record = {
        "kind": "gpu_preflight",
        "dtype": "float32",
        "batch_size": 1,
        "json_path": str(gpu_preflight_json.relative_to(REPO_ROOT)),
        "markdown_path": str(gpu_preflight_md.relative_to(REPO_ROOT)),
    }
    status["gpu_preflights"].append(gpu_preflight_record)
    gpu_preflight_rc = run_command(
        benchmark_command(
            dimensions=["10"],
            parameter_counts=["50"],
            batch_size=1,
            device="gpu",
            dtype="float32",
            output_json=gpu_preflight_json,
            output_md=gpu_preflight_md,
        ),
        log_path=LOG_DIR / "kalman_qr_core_batch_grid_preflight_gpu_float32_xla_autotune0_overnight_2026-07-09.log",
        env=preflight_env,
        status=status,
        record=gpu_preflight_record,
    )

    status["gpu_status"] = (
        "preflight_passed" if gpu_preflight_rc == 0 else "blocked_gpu_xla_autodiff_preflight"
    )
    write_status(status)

    cpu_failures = 0
    for threads in CPU_THREADS:
        for batch_size in BATCH_SIZES:
            json_path = BENCH_DIR / f"kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch_size}_xla_2026-07-09.json"
            md_path = BENCH_DIR / f"kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch_size}_xla_2026-07-09.md"
            log_path = LOG_DIR / f"kalman_qr_core_batch_grid_cpu_threads{threads}_batch{batch_size}_xla_2026-07-09.log"
            record = {
                "kind": "cpu",
                "threads": threads,
                "batch_size": batch_size,
                "dtype": "float32",
                "json_path": str(json_path.relative_to(REPO_ROOT)),
                "markdown_path": str(md_path.relative_to(REPO_ROOT)),
            }
            status["cpu"].append(record)
            if json_path.exists():
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                    if payload.get("summary", {}).get("run_status") == "complete":
                        record["status"] = "skipped_existing_complete"
                        write_status(status)
                        continue
                except Exception:
                    pass
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "-1"
            rc = run_command(
                benchmark_command(
                    dimensions=DIMENSIONS,
                    parameter_counts=PARAMETER_COUNTS,
                    batch_size=batch_size,
                    device="cpu",
                    dtype="float32",
                    output_json=json_path,
                    output_md=md_path,
                    cpu_threads=threads,
                ),
                log_path=log_path,
                env=env,
                status=status,
                record=record,
            )
            if rc != 0:
                cpu_failures += 1

    if gpu_preflight_rc == 0:
        for dtype in DTYPES:
            for batch_size in BATCH_SIZES:
                json_path = BENCH_DIR / f"kalman_qr_core_batch_grid_gpu_{dtype}_batch{batch_size}_xla_2026-07-09.json"
                md_path = BENCH_DIR / f"kalman_qr_core_batch_grid_gpu_{dtype}_batch{batch_size}_xla_2026-07-09.md"
                log_path = LOG_DIR / f"kalman_qr_core_batch_grid_gpu_{dtype}_batch{batch_size}_xla_2026-07-09.log"
                record = {
                    "kind": "gpu",
                    "dtype": dtype,
                    "batch_size": batch_size,
                    "json_path": str(json_path.relative_to(REPO_ROOT)),
                    "markdown_path": str(md_path.relative_to(REPO_ROOT)),
                }
                status["gpu"].append(record)
                env = os.environ.copy()
                env.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
                run_command(
                    benchmark_command(
                        dimensions=DIMENSIONS,
                        parameter_counts=PARAMETER_COUNTS,
                        batch_size=batch_size,
                        device="gpu",
                        dtype=dtype,
                        output_json=json_path,
                        output_md=md_path,
                    ),
                    log_path=log_path,
                    env=env,
                    status=status,
                    record=record,
                )

    status["status"] = "complete_with_failures" if cpu_failures or gpu_preflight_rc else "complete"
    status["cpu_failure_count"] = cpu_failures
    write_status(status)
    return 1 if cpu_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
