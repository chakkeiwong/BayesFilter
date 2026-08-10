#!/usr/bin/env python3
"""Profile q=20 target evaluation with 25 CPU workers times batch 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("batch4x25 CPU profile found a visible GPU")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER  # noqa: E402


SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_batch4x25_profile.v1"
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-cpu-batch4x25-profile-plan-2026-07-23.md")
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
WORKERS = 25
SHARD_SIZE = 4
TOTAL_BATCH = WORKERS * SHARD_SIZE
WORKER_CPUS = tuple(range(WORKERS))
REPEATS = 6
HOST_RAM_CAP_BYTES = 64 * 1024**3
BASELINE_SECONDS = 27.255775838400588


class ProfileError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ProfileError(f"artifact exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thread_rows(worker_pids: Sequence[int], cpu_ids: Sequence[int]) -> list[Mapping[str, Any]]:
    rows = []
    for pid, cpu in zip(worker_pids, cpu_ids, strict=True):
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
            raise ProfileError(f"worker {pid} is not fully bound to CPU {cpu}")
        rows.append({"pid": int(pid), "assigned_cpu": int(cpu), "native_threads": len(tasks)})
    return rows


def deterministic_rows() -> tf.Tensor:
    offsets = tf.reshape(
        tf.linspace(tf.constant(-0.08, tf.float64), tf.constant(0.08, tf.float64), TOTAL_BATCH * 4),
        [TOTAL_BATCH, 4],
    )
    return PRIOR_CENTER[tf.newaxis, :] + offsets


def pool_config() -> TFBatchValueScorePoolConfig:
    return TFBatchValueScorePoolConfig(
        factory_path=("bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:batch_native_complexity_target_worker_factory"),
        factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh"},
        dimension=4,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(SHARD_SIZE,),
        batch_per_worker=SHARD_SIZE,
        worker_cpu_ids=WORKER_CPUS,
        timeout_seconds=900.0,
    )


def progress(status: str, calls: Sequence[Mapping[str, Any]], startup: Any, error: Any) -> Mapping[str, Any]:
    return {"schema": SCHEMA, "status": status, "startup": startup, "calls": list(calls), "error": error, "nonclaims": ["profile only", "no optimizer update", "no NeuTra or HMC claim"]}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise ProfileError("output root must be new, empty, and inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    calls: list[Mapping[str, Any]] = []
    startup = None
    error = None
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    baseline_values = None
    baseline_scores = None
    try:
        with TFBatchValueScorePool(pool_config()) as pool:
            startup_started = time.perf_counter()
            pool._ensure_started()
            startup_seconds = time.perf_counter() - startup_started
            assert pool._startup is not None
            startup = {"seconds": startup_seconds, "metadata": dict(pool._startup)}
            write_json(output / "progress.json", progress("RUNNING", calls, startup, None), replace=True)
            rows = deterministic_rows()
            for repeat_index in range(REPEATS):
                if time.perf_counter() - started + 60.0 >= args.cap_seconds:
                    raise ProfileError("profile cap exhausted")
                call_started = time.perf_counter()
                values, scores, metadata = pool.evaluate(rows, request_id=f"batch4x25-repeat-{repeat_index}")
                total_seconds = time.perf_counter() - call_started
                if metadata["worker_backend"] != "batch_native_value_score":
                    raise ProfileError("batch-native worker backend missing")
                assigned = [int(value) for value in metadata["worker_assigned_cpu_ids"]]
                if set(assigned) != set(WORKER_CPUS):
                    raise ProfileError("worker CPU assignment set mismatch")
                result_pids = [int(value) for value in metadata["worker_result_pids"]]
                if set(result_pids) != set(metadata["startup_worker_pids"]):
                    raise ProfileError("one persistent worker did not evaluate the batch shard")
                startup_by_pid = {
                    int(row["pid"]): int(row["assigned_cpu"])
                    for row in metadata["startup_worker_metadata"]
                }
                bindings = thread_rows(
                    metadata["startup_worker_pids"],
                    [startup_by_pid[int(pid)] for pid in metadata["startup_worker_pids"]],
                )
                native_threads = sum(int(row["native_threads"]) for row in bindings)
                parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
                worker_bytes = max(int(metadata["active_worker_ru_maxrss_sum_bytes"]), int(metadata["startup_worker_ru_maxrss_sum_bytes"]))
                combined_bytes = parent_bytes + worker_bytes
                if combined_bytes > HOST_RAM_CAP_BYTES:
                    raise ProfileError("combined RSS exceeds 64 GiB")
                if repeat_index == 0:
                    baseline_values = tf.identity(values)
                    baseline_scores = tf.identity(scores)
                    value_delta = score_delta = 0.0
                else:
                    assert baseline_values is not None and baseline_scores is not None
                    value_delta = float(tf.reduce_max(tf.abs(values - baseline_values)).numpy())
                    score_delta = float(tf.reduce_max(tf.abs(scores - baseline_scores)).numpy())
                row = {
                    "repeat_index": repeat_index,
                    "phase": "first_call" if repeat_index == 0 else "repeat_call",
                    "total_seconds": total_seconds,
                    "worker_runtime_max_seconds": float(metadata["worker_runtime_max_seconds"]),
                    "worker_runtime_seconds": list(metadata["worker_runtime_seconds"]),
                    "parent_overhead_seconds": max(0.0, total_seconds - float(metadata["worker_runtime_max_seconds"])),
                    "max_value_delta_from_first": value_delta,
                    "max_score_delta_from_first": score_delta,
                    "configured_compute_cores": WORKERS,
                    "native_worker_threads": native_threads,
                    "worker_bindings": bindings,
                    "combined_ru_maxrss_bytes": combined_bytes,
                }
                calls.append(row)
                write_json(output / "progress.json", progress("RUNNING", calls, startup, None), replace=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    repeats = [float(row["total_seconds"]) for row in calls if row["repeat_index"] > 0]
    mean = None if not repeats else sum(repeats) / len(repeats)
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED" if error is None and len(calls) == REPEATS else "PARTIAL_PROFILE_STOP",
        "error": error,
        "topology": {"id": "batch4x25", "worker_count": WORKERS, "batch_per_worker": SHARD_SIZE, "total_batch": TOTAL_BATCH, "assigned_worker_cpus": list(WORKER_CPUS), "configured_compute_cores": WORKERS, "native_thread_policy": "recorded_housekeeping_not_compute_core_count"},
        "startup": startup,
        "calls": calls,
        "comparison": {
            "baseline_topology": "4_processes_x_batch25",
            "baseline_repeat_mean_seconds": BASELINE_SECONDS,
            "candidate_repeat_mean_seconds": mean,
            "candidate_over_baseline_ratio": None if mean is None else mean / BASELINE_SECONDS,
            "descriptive_only": True,
        },
        "derived": {
            "estimated_250_updates_seconds": None if mean is None else 250.0 * mean,
            "estimated_two_stream_2000_updates_seconds": None if mean is None else 4000.0 * mean,
            "descriptive_only": True,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "command": " ".join(sys.argv),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": args.cap_seconds,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "jit_compile": False,
            "dtype": "float64",
            "plan": PLAN.as_posix(),
            "source_sha256": {"script": sha256(ROOT / SCRIPT), "plan": sha256(ROOT / PLAN), "pool": sha256(ROOT / "bayesfilter/inference/tf_batch_value_score_pool.py"), "target": sha256(ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py")},
        },
        "nonclaims": ["profile only", "no optimizer update", "no NeuTra quality, HMC, posterior, or ranking claim"],
    }
    write_json(output / "result.json", result)
    write_json(output / "progress.json", progress(result["status"], calls, startup, error), replace=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > 1800.0:
        parser.error("--cap-seconds must be in (0, 1800]")
    result = run(args)
    print(json.dumps({"status": result["status"], "comparison": result["comparison"], "error": result["error"]}, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
