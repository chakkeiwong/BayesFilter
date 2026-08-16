#!/usr/bin/env python3
"""Bounded q=20 CPU profile for a pinned worker/shard topology."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("75-worker CPU profile found a visible GPU")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER  # noqa: E402


SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_batch_grid_profile.v1"
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-cpu-batch-grid-profile-plan-2026-07-31.md")
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
WORKERS = 10
SHARD_SIZE = 10
TOTAL_BATCH = WORKERS * SHARD_SIZE
CPU_IDS = tuple(range(WORKERS))
REPEATS = 4
RSS_CAP_BYTES = 64 * 1024**3


class ProfileError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ProfileError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> Mapping[str, str]:
    paths = {
        "script": SCRIPT,
        "plan": PLAN,
        "pool": Path("bayesfilter/inference/tf_batch_value_score_pool.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"),
    }
    return {name: sha256(ROOT / path) for name, path in paths.items()}


def deterministic_rows() -> tf.Tensor:
    offsets = tf.reshape(
        tf.linspace(tf.constant(-0.08, tf.float64), tf.constant(0.08, tf.float64), TOTAL_BATCH * 4),
        [TOTAL_BATCH, 4],
    )
    return PRIOR_CENTER[tf.newaxis, :] + offsets


def pool_config() -> TFBatchValueScorePoolConfig:
    return TFBatchValueScorePoolConfig(
        factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
            "batch_native_complexity_target_worker_factory"
        ),
        factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh"},
        dimension=4,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(SHARD_SIZE,),
        batch_per_worker=SHARD_SIZE,
        worker_cpu_ids=CPU_IDS,
        timeout_seconds=900.0,
    )


def thread_rows(pids: Sequence[int], cpu_ids: Sequence[int]) -> list[Mapping[str, Any]]:
    rows = []
    for pid, cpu in zip(pids, cpu_ids, strict=True):
        tasks = []
        for name in os.listdir(f"/proc/{pid}/task"):
            if not name.isdigit():
                continue
            tid = int(name)
            try:
                affinity = sorted(os.sched_getaffinity(tid))
            except ProcessLookupError:
                continue
            tasks.append({"tid": tid, "affinity": affinity})
        if not tasks or any(task["affinity"] != [int(cpu)] for task in tasks):
            raise ProfileError(f"worker {pid} thread affinity mismatch for CPU {cpu}")
        rows.append({"pid": int(pid), "assigned_cpu": int(cpu), "native_threads": len(tasks)})
    return rows


def topology_payload() -> Mapping[str, Any]:
    return {
        "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cpu_count": os.cpu_count(),
        "worker_count": WORKERS,
        "batch_per_worker": SHARD_SIZE,
        "total_batch": TOTAL_BATCH,
        "worker_cpu_ids": list(CPU_IDS),
    }


def progress_payload(status: str, startup: Any, calls: Sequence[Mapping[str, Any]], error: str | None) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "startup": startup,
        "calls": list(calls),
        "error": error,
        "topology": topology_payload(),
        "nonclaims": ["profile only", "CPU diagnostic route", "no optimizer or HMC claim"],
    }


def run(args: Any) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise ProfileError("output root must be new, empty, and inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    started = time.perf_counter()
    calls: list[Mapping[str, Any]] = []
    startup = None
    error = None
    rows = deterministic_rows()
    try:
        with TFBatchValueScorePool(pool_config()) as pool:
            startup_started = time.perf_counter()
            pool._ensure_started()
            startup_seconds = time.perf_counter() - startup_started
            if pool._startup is None:
                raise ProfileError("missing pool startup metadata")
            startup = {"seconds": startup_seconds, "metadata": dict(pool._startup)}
            write_json(progress_path, progress_payload("RUNNING", startup, calls, None), replace=True)
            baseline_values = None
            baseline_scores = None
            for repeat_index in range(REPEATS):
                if time.perf_counter() - started + 120.0 >= args.cap_seconds:
                    raise ProfileError("profile wall-time cap exhausted")
                call_started = time.perf_counter()
                values, scores, metadata = pool.evaluate(rows, request_id=f"q20-batch4x75-{repeat_index}")
                total_seconds = time.perf_counter() - call_started
                if metadata["worker_backend"] != "batch_native_value_score":
                    raise ProfileError("worker backend is not batch-native")
                if metadata["worker_shard_sizes"] != [SHARD_SIZE] * WORKERS:
                    raise ProfileError(
                        f"worker shard sizes are not exactly {WORKERS} x {SHARD_SIZE}"
                    )
                assigned = [int(value) for value in metadata["worker_assigned_cpu_ids"]]
                if set(assigned) != set(CPU_IDS):
                    raise ProfileError("worker CPU assignment set mismatch")
                result_pids = [int(value) for value in metadata["worker_result_pids"]]
                if set(result_pids) != set(metadata["startup_worker_pids"]):
                    raise ProfileError("not every persistent worker evaluated the shard")
                bindings = thread_rows(
                    metadata["startup_worker_pids"],
                    [int(row["assigned_cpu"]) for row in metadata["startup_worker_metadata"]],
                )
                parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                worker_bytes = max(
                    int(metadata["active_worker_ru_maxrss_sum_bytes"]),
                    int(metadata["startup_worker_ru_maxrss_sum_bytes"]),
                )
                combined_bytes = parent_bytes + worker_bytes
                if combined_bytes > RSS_CAP_BYTES:
                    raise ProfileError(f"combined RSS exceeds 64 GiB: {combined_bytes}")
                if repeat_index == 0:
                    baseline_values = tf.identity(values)
                    baseline_scores = tf.identity(scores)
                    value_delta = score_delta = 0.0
                else:
                    assert baseline_values is not None and baseline_scores is not None
                    value_delta = float(tf.reduce_max(tf.abs(values - baseline_values)).numpy())
                    score_delta = float(tf.reduce_max(tf.abs(scores - baseline_scores)).numpy())
                call = {
                    "repeat_index": repeat_index,
                    "phase": "first_call" if repeat_index == 0 else "warm_call",
                    "total_seconds": total_seconds,
                    "rows_per_second": TOTAL_BATCH / total_seconds,
                    "worker_runtime_max_seconds": float(metadata["worker_runtime_max_seconds"]),
                    "worker_runtime_seconds": list(metadata["worker_runtime_seconds"]),
                    "worker_runtime_min_seconds": min(float(value) for value in metadata["worker_runtime_seconds"]),
                    "worker_runtime_median_seconds": sorted(float(value) for value in metadata["worker_runtime_seconds"])[WORKERS // 2],
                    "parent_overhead_seconds": max(0.0, total_seconds - float(metadata["worker_runtime_max_seconds"])),
                    "max_value_delta_from_first": value_delta,
                    "max_score_delta_from_first": score_delta,
                    "combined_ru_maxrss_bytes": combined_bytes,
                    "native_threads": sum(int(row["native_threads"]) for row in bindings),
                    "worker_bindings": bindings,
                }
                calls.append(call)
                write_json(progress_path, progress_payload("RUNNING", startup, calls, None), replace=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    warm = [float(call["total_seconds"]) for call in calls if call["repeat_index"] > 0]
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED" if error is None and len(calls) == REPEATS else "PARTIAL_PROFILE_STOP",
        "error": error,
        "topology": topology_payload(),
        "startup": startup,
        "calls": calls,
        "derived": {
            "warm_repeat_count": len(warm),
            "warm_mean_seconds": None if not warm else math.fsum(warm) / len(warm),
            "warm_min_seconds": None if not warm else min(warm),
            "warm_max_seconds": None if not warm else max(warm),
            "warm_mean_rows_per_second": None if not warm else TOTAL_BATCH / (math.fsum(warm) / len(warm)),
            "reference_25_worker_4_row_seconds": 14.364853534396389,
            "reference_ratio": None if not warm else (math.fsum(warm) / len(warm)) / 14.364853534396389,
            "descriptive_only": True,
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "jit_compile": False,
            "dtype": "float64",
            "source_sha256": source_hashes(),
            "plan": PLAN.as_posix(),
            "output_root": args.output_root.as_posix(),
            "wall_seconds": time.perf_counter() - started,
            "rss_cap_bytes": RSS_CAP_BYTES,
        },
        "nonclaims": [
            "profile only",
            "strict CPU tensorflow_eigh route; not current GPU compiled_custom_op parity",
            "no CPU/GPU superiority claim",
            "no NeuTra quality, HMC, posterior, convergence, or default claim",
        ],
    }
    write_json(output / "result.json", result)
    write_json(progress_path, progress_payload(result["status"], startup, calls, error), replace=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--rows-per-worker", type=int, required=True)
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    global WORKERS, SHARD_SIZE, TOTAL_BATCH, CPU_IDS
    WORKERS = int(args.workers)
    SHARD_SIZE = int(args.rows_per_worker)
    TOTAL_BATCH = WORKERS * SHARD_SIZE
    CPU_IDS = tuple(range(WORKERS))
    if WORKERS <= 0 or SHARD_SIZE <= 0 or WORKERS > 128:
        parser.error("--workers must be in [1,128] and --rows-per-worker must be positive")
    if max(CPU_IDS) >= max(os.sched_getaffinity(0)) + 1:
        parser.error("requested worker CPUs are outside the current affinity")
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > 1800.0:
        parser.error("--cap-seconds must be in (0, 1800]")
    result = run(args)
    print(json.dumps({"status": result["status"], "derived": result["derived"], "error": result["error"]}, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
