#!/usr/bin/env python3
"""Bounded full-update CPU timing for q=20 with a pinned 25x4 pool."""

from __future__ import annotations

import argparse
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("full-update CPU benchmark found a visible GPU")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    FREE_NAMES,
    PRIOR_CENTER,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_full_update_25x4.v1"
PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-cpu-full-update-25x4-plan-2026-08-01.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
WORKERS = 25
ROWS_PER_WORKER = 4
TOTAL_ROWS = WORKERS * ROWS_PER_WORKER
REPEATS = 4
RSS_CAP_BYTES = 64 * 1024**3
SEED = (20260801, 2504)


class BenchmarkError(RuntimeError):
    pass


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise BenchmarkError(f"refusing to overwrite artifact: {path}")
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
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"),
    }
    return {name: sha256(ROOT / path) for name, path in paths.items()}


def batch() -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(SEED, tf.int32), 0
    )
    return tf.random.stateless_normal([TOTAL_ROWS, 4], seed, dtype=tf.float64)


def pool_config(*, jit_compile: bool) -> TFBatchValueScorePoolConfig:
    cpu_ids = tuple(range(WORKERS))
    return TFBatchValueScorePoolConfig(
        factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
            "batch_native_complexity_target_worker_factory"
        ),
        factory_config={
            "q": 20,
            "principal_sqrt_backend": "tensorflow_eigh",
            "jit_compile": bool(jit_compile),
        },
        dimension=4,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(ROWS_PER_WORKER,),
        batch_per_worker=ROWS_PER_WORKER,
        worker_cpu_ids=cpu_ids,
        timeout_seconds=900.0,
    )


def trainer(target: Any, *, jit_compile: bool) -> NeuTraReverseKLTrainer:
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=4.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260801, 2511),
        jit_compile=bool(jit_compile),
    )
    return NeuTraReverseKLTrainer(target, config)


def thread_rows(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = []
    for item in metadata.get("startup_worker_metadata", ()):
        pid = int(item["pid"])
        cpu = int(item["assigned_cpu"])
        tasks = []
        for name in os.listdir(f"/proc/{pid}/task"):
            if not str(name).isdigit():
                continue
            tid = int(name)
            try:
                affinity = sorted(os.sched_getaffinity(tid))
            except ProcessLookupError:
                continue
            tasks.append({"tid": tid, "affinity": affinity})
        if not tasks or any(task["affinity"] != [cpu] for task in tasks):
            raise BenchmarkError(f"worker {pid} is not pinned to CPU {cpu}")
        rows.append({"pid": pid, "assigned_cpu": cpu, "native_threads": len(tasks)})
    return rows


def evaluate(
    pool: TFBatchValueScorePool, rows: tf.Tensor, request_id: str
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, Any]]:
    values, scores, metadata = pool.evaluate(rows, request_id=request_id)
    if metadata.get("worker_shard_sizes") != [ROWS_PER_WORKER] * WORKERS:
        raise BenchmarkError(
            f"worker shard partition is not exactly {WORKERS}x{ROWS_PER_WORKER}"
        )
    if set(int(value) for value in metadata.get("worker_assigned_cpu_ids", ())) != set(range(WORKERS)):
        raise BenchmarkError("worker CPU assignment set mismatch")
    if set(int(value) for value in metadata.get("worker_result_pids", ())) != set(metadata.get("startup_worker_pids", ())):
        raise BenchmarkError("not every worker evaluated a shard")
    bindings = thread_rows(metadata)
    combined = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024) + max(
        int(metadata.get("active_worker_ru_maxrss_sum_bytes", 0)),
        int(metadata.get("startup_worker_ru_maxrss_sum_bytes", 0)),
    )
    if combined > RSS_CAP_BYTES:
        raise BenchmarkError(f"combined RSS exceeds 64 GiB: {combined}")
    tf.debugging.assert_all_finite(values, "CPU pooled target values")
    tf.debugging.assert_all_finite(scores, "CPU pooled target scores")
    return values, scores, {
        **metadata,
        "combined_ru_maxrss_bytes": combined,
        "native_threads": sum(int(row["native_threads"]) for row in bindings),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT) or (output.exists() and any(output.iterdir())):
        raise BenchmarkError("output root must be new, empty, and inside the repository")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = batch()
    target = batch_native_complexity_posterior_target(
        20,
        jit_compile=bool(args.jit_compile),
        principal_sqrt_backend="tensorflow_eigh",
    )
    train = trainer(target, jit_compile=bool(args.jit_compile))
    calls: list[Mapping[str, Any]] = []
    baseline_loss = None
    error = None
    startup = None
    try:
        with TFBatchValueScorePool(
            pool_config(jit_compile=bool(args.jit_compile))
        ) as pool:
            startup_started = time.perf_counter()
            pool._ensure_started()
            startup = {
                "seconds": time.perf_counter() - startup_started,
                "metadata": dict(pool._startup or {}),
            }
            for repeat_index in range(REPEATS):
                if time.perf_counter() - started + 120.0 >= args.cap_seconds:
                    raise BenchmarkError("benchmark wall-time cap exhausted")
                total_started = time.perf_counter()
                forward_started = time.perf_counter()
                theta, _logdet = train.forward_and_logdet(rows)
                forward_seconds = time.perf_counter() - forward_started
                target_started = time.perf_counter()
                values, scores, metadata = evaluate(
                    pool,
                    theta,
                    request_id=(
                        f"q20-full-update-{WORKERS}x{ROWS_PER_WORKER}-{repeat_index}"
                    ),
                )
                target_seconds = time.perf_counter() - target_started
                update_started = time.perf_counter()
                step = train.train_step_with_external_value_score(rows, values, scores)
                update_seconds = time.perf_counter() - update_started
                total_seconds = time.perf_counter() - total_started
                loss = float(step.loss.numpy())
                gradient_norm = float(step.gradient_norm.numpy())
                if not math.isfinite(loss) or not math.isfinite(gradient_norm):
                    raise BenchmarkError("optimizer update returned nonfinite diagnostics")
                loss_delta = None if baseline_loss is None else loss - baseline_loss
                if baseline_loss is None:
                    baseline_loss = loss
                calls.append(
                    {
                        "phase": "first_call" if repeat_index == 0 else "warm_call",
                        "repeat_index": repeat_index,
                        "step": int(step.step.numpy()),
                        "forward_seconds": forward_seconds,
                        "target_value_score_seconds": target_seconds,
                        "optimizer_update_seconds": update_seconds,
                        "total_seconds": total_seconds,
                        "rows_per_second": TOTAL_ROWS / total_seconds,
                        "loss": loss,
                        "loss_delta_from_first": loss_delta,
                        "gradient_norm": gradient_norm,
                        "clipped_gradient_norm": float(step.clipped_gradient_norm.numpy()),
                        "clipping_applied": bool(step.clipping_applied.numpy()),
                        "worker_runtime_min_seconds": min(float(x) for x in metadata["worker_runtime_seconds"]),
                        "worker_runtime_median_seconds": sorted(float(x) for x in metadata["worker_runtime_seconds"])[WORKERS // 2],
                        "worker_runtime_max_seconds": float(metadata["worker_runtime_max_seconds"]),
                        "parent_overhead_seconds": max(0.0, target_seconds - float(metadata["worker_runtime_max_seconds"])),
                        "combined_ru_maxrss_bytes": int(metadata["combined_ru_maxrss_bytes"]),
                        "native_threads": int(metadata["native_threads"]),
                    }
                )
                write_json(output / "progress.json", {"schema": SCHEMA, "status": "RUNNING", "calls": calls}, replace=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    warm = [float(call["total_seconds"]) for call in calls if call["phase"] == "warm_call"]
    result = {
        "schema": SCHEMA,
        "status": "COMPLETED" if error is None and len(calls) == REPEATS else "PARTIAL_PROFILE_STOP",
        "error": error,
        "topology": {"workers": WORKERS, "rows_per_worker": ROWS_PER_WORKER, "total_rows": TOTAL_ROWS, "cpu_ids": list(range(WORKERS))},
        "target": {"q": 20, "principal_sqrt_backend": "tensorflow_eigh", "jit_compile": bool(args.jit_compile), "dtype": "float64", "evaluation_policy": "batch_native_tensorflow_status_no_row_mapping_v2"},
        "startup": startup,
        "calls": calls,
        "derived": {
            "warm_repeat_count": len(warm),
            "warm_mean_total_seconds": None if not warm else math.fsum(warm) / len(warm),
            "warm_min_total_seconds": None if not warm else min(warm),
            "warm_max_total_seconds": None if not warm else max(warm),
            "warm_mean_rows_per_second": None if not warm else TOTAL_ROWS / (math.fsum(warm) / len(warm)),
            "descriptive_only": True,
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "jit_compile": bool(args.jit_compile),
            "random_seed": list(SEED),
            "source_sha256": source_hashes(),
            "plan": PLAN.as_posix(),
            "output_root": args.output_root.as_posix(),
            "wall_seconds": time.perf_counter() - started,
            "rss_cap_bytes": RSS_CAP_BYTES,
        },
        "nonclaims": [
            "full optimizer timing diagnostic only",
            (
                "CPU-only XLA tensorflow_eigh route"
                if args.jit_compile
                else "CPU-only non-XLA tensorflow_eigh route"
            ),
            "not backend-identical to current hybrid GPU compiled_custom_op route",
            "no CPU/GPU superiority, NeuTra quality, HMC, posterior, convergence, or default claim",
        ],
    }
    write_json(output / "result.json", result)
    write_json(output / "progress.json", {"schema": SCHEMA, "status": result["status"], "calls": calls, "error": error}, replace=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    parser.add_argument("--jit-compile", action="store_true")
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--rows-per-worker", type=int, default=4)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > 1800.0:
        parser.error("--cap-seconds must be in (0, 1800]")
    if args.workers <= 0 or args.workers > 50:
        parser.error("--workers must be in [1,50]")
    if args.rows_per_worker <= 0:
        parser.error("--rows-per-worker must be positive")
    available_cpus = os.sched_getaffinity(0)
    if not set(range(args.workers)).issubset(available_cpus):
        parser.error("requested worker CPUs are outside the current process affinity")
    global WORKERS, ROWS_PER_WORKER, TOTAL_ROWS
    WORKERS = int(args.workers)
    ROWS_PER_WORKER = int(args.rows_per_worker)
    TOTAL_ROWS = WORKERS * ROWS_PER_WORKER
    result = run(args)
    print(json.dumps({"status": result["status"], "derived": result["derived"], "error": result["error"]}, sort_keys=True))
    return 0 if result["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
