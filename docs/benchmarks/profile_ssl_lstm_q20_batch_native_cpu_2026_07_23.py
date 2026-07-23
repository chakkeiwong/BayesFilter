#!/usr/bin/env python3
"""Profile q=20 batch-native CPU value/score startup and steady state."""

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("CPU profile found a visible GPU")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_batch_native_cpu_profile.v1"
PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-batch-native-profile-plan-2026-07-23.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
THREAD_LIMIT = 50
HOST_RAM_CAP_BYTES = 64 * 1024**3
SHARD_SIZES = (2, 3, 16, 25)
REPEATS = {2: 3, 3: 3, 16: 3, 25: 6}


class ProfileError(RuntimeError):
    """Raised when the bounded profile contract fails."""


class Budget:
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 0.0) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ProfileError("profile wall-time cap exhausted")


def _canonical(payload: Any) -> bytes:
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


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ProfileError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(payload))
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _thread_count(pid: int) -> int:
    text = Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1])
    raise ProfileError(f"thread count missing for pid {pid}")


def _resource_snapshot(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    worker_pids = tuple(int(value) for value in metadata["startup_worker_pids"])
    rows = [
        {
            "pid": pid,
            "role": "parent" if pid == os.getpid() else "target_worker",
            "threads": _thread_count(pid),
        }
        for pid in (os.getpid(), *worker_pids)
    ]
    total_threads = sum(int(row["threads"]) for row in rows)
    parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    worker_bytes = max(
        int(metadata.get("active_worker_ru_maxrss_sum_bytes", 0)),
        int(metadata.get("startup_worker_ru_maxrss_sum_bytes", 0)),
    )
    combined_bytes = parent_bytes + worker_bytes
    if total_threads > THREAD_LIMIT:
        raise ProfileError(f"process-tree thread count {total_threads} exceeds 50")
    if combined_bytes > HOST_RAM_CAP_BYTES:
        raise ProfileError("combined parent/worker RSS exceeds 64 GiB")
    return {
        "process_tree_threads": total_threads,
        "thread_rows": rows,
        "parent_ru_maxrss_bytes": parent_bytes,
        "worker_ru_maxrss_sum_bytes": worker_bytes,
        "combined_ru_maxrss_bytes": combined_bytes,
    }


def _pool() -> TFBatchValueScorePool:
    return TFBatchValueScorePool(
        TFBatchValueScorePoolConfig(
            factory_path=(
                "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
                "batch_native_complexity_target_worker_factory"
            ),
            factory_config={
                "q": 20,
                "principal_sqrt_backend": "tensorflow_eigh",
            },
            dimension=4,
            worker_count=4,
            cores_per_worker=1,
            batch_sizes=SHARD_SIZES,
            timeout_seconds=480.0,
        )
    )


def _rows(total_rows: int) -> tf.Tensor:
    offsets = tf.reshape(
        tf.linspace(
            tf.constant(-0.08, tf.float64),
            tf.constant(0.08, tf.float64),
            int(total_rows) * 4,
        ),
        [int(total_rows), 4],
    )
    return PRIOR_CENTER[tf.newaxis, :] + offsets


def _progress_payload(
    *,
    budget: Budget,
    startup: Mapping[str, Any] | None,
    calls: Sequence[Mapping[str, Any]],
    status: str,
    error: str | None = None,
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "elapsed_seconds": budget.elapsed,
        "startup": startup,
        "calls": list(calls),
        "error": error,
        "nonclaims": [
            "profile only",
            "no optimizer update",
            "no NeuTra quality or HMC claim",
        ],
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise ProfileError("output root must remain inside the repository")
    if output.exists() and any(output.iterdir()):
        raise ProfileError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    result_path = output / "result.json"
    budget = Budget(args.cap_seconds)
    started_at = datetime.now(timezone.utc).isoformat()
    calls: list[Mapping[str, Any]] = []
    startup: Mapping[str, Any] | None = None
    error = None

    target_started = time.perf_counter()
    parent_target = batch_native_complexity_posterior_target(
        20,
        jit_compile=False,
        principal_sqrt_backend="tensorflow_eigh",
    )
    parent_target_seconds = time.perf_counter() - target_started
    try:
        with _pool() as pool:
            budget.require(30.0)
            startup_started = time.perf_counter()
            pool._ensure_started()  # Profile the same lazy-start boundary used in production.
            startup_seconds = time.perf_counter() - startup_started
            if pool._startup is None:
                raise ProfileError("pool startup metadata is missing")
            startup = {
                "seconds": startup_seconds,
                "metadata": dict(pool._startup),
                "resources": _resource_snapshot(pool._startup),
            }
            _write_json(
                progress_path,
                _progress_payload(
                    budget=budget,
                    startup=startup,
                    calls=calls,
                    status="RUNNING",
                ),
                replace=True,
            )

            baselines: dict[int, tuple[tf.Tensor, tf.Tensor]] = {}
            for shard_size in SHARD_SIZES:
                total_rows = 4 * int(shard_size)
                rows = _rows(total_rows)
                for repeat_index in range(REPEATS[shard_size]):
                    budget.require(30.0)
                    call_started = time.perf_counter()
                    values, scores, metadata = pool.evaluate(
                        rows,
                        request_id=(
                            f"q20-profile-shard-{shard_size}-repeat-{repeat_index}"
                        ),
                    )
                    total_seconds = time.perf_counter() - call_started
                    resources = _resource_snapshot(metadata)
                    if metadata.get("worker_backend") != "batch_native_value_score":
                        raise ProfileError("worker backend is not batch-native")
                    if repeat_index == 0:
                        baselines[shard_size] = (
                            tf.identity(values),
                            tf.identity(scores),
                        )
                        max_value_delta = 0.0
                        max_score_delta = 0.0
                    else:
                        baseline_values, baseline_scores = baselines[shard_size]
                        max_value_delta = float(
                            tf.reduce_max(tf.abs(values - baseline_values)).numpy()
                        )
                        max_score_delta = float(
                            tf.reduce_max(tf.abs(scores - baseline_scores)).numpy()
                        )
                    row = {
                        "shard_size": shard_size,
                        "total_rows": total_rows,
                        "repeat_index": repeat_index,
                        "phase": "first_call" if repeat_index == 0 else "repeat_call",
                        "total_seconds": total_seconds,
                        "worker_runtime_seconds": list(
                            metadata["worker_runtime_seconds"]
                        ),
                        "worker_runtime_max_seconds": float(
                            metadata["worker_runtime_max_seconds"]
                        ),
                        "parent_overhead_seconds": max(
                            0.0,
                            total_seconds
                            - float(metadata["worker_runtime_max_seconds"]),
                        ),
                        "max_value_delta_from_first": max_value_delta,
                        "max_score_delta_from_first": max_score_delta,
                        "resources": resources,
                    }
                    calls.append(row)
                    _write_json(
                        progress_path,
                        _progress_payload(
                            budget=budget,
                            startup=startup,
                            calls=calls,
                            status="RUNNING",
                        ),
                        replace=True,
                    )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"

    scalar_parity = None
    if error is None:
        scalar = complexity_posterior_target(20, jit_compile=False)
        parity_rows = _rows(2)
        batch_values, batch_scores = parent_target.batch_value_and_score(parity_rows)
        scalar_values = []
        scalar_scores = []
        for index in range(2):
            value, score = scalar.value_and_score(parity_rows[index])
            scalar_values.append(value)
            scalar_scores.append(score)
        scalar_parity = {
            "role": "diagnostic_scalar_authority_only",
            "max_value_abs_error": float(
                tf.reduce_max(tf.abs(batch_values - tf.stack(scalar_values))).numpy()
            ),
            "max_score_abs_error": float(
                tf.reduce_max(tf.abs(batch_scores - tf.stack(scalar_scores))).numpy()
            ),
        }

    size25_repeats = [
        float(row["total_seconds"])
        for row in calls
        if row["shard_size"] == 25 and row["repeat_index"] > 0
    ]
    derived = {
        "size25_repeat_count": len(size25_repeats),
        "size25_repeat_mean_seconds": (
            None
            if not size25_repeats
            else sum(size25_repeats) / len(size25_repeats)
        ),
        "estimated_250_updates_seconds": (
            None
            if not size25_repeats
            else 250.0 * sum(size25_repeats) / len(size25_repeats)
        ),
        "estimated_two_stream_2000_updates_seconds": (
            None
            if not size25_repeats
            else 4000.0 * sum(size25_repeats) / len(size25_repeats)
        ),
        "estimates_are_descriptive_only": True,
    }
    source_paths = {
        "profile": SCRIPT,
        "plan": PLAN,
        "pool": Path("bayesfilter/inference/tf_batch_value_score_pool.py"),
        "target": Path(
            "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
        ),
    }
    payload = {
        "schema": SCHEMA,
        "status": "COMPLETED" if error is None else "PARTIAL_PROFILE_STOP",
        "error": error,
        "parent_target_construction_seconds": parent_target_seconds,
        "startup": startup,
        "calls": calls,
        "derived": derived,
        "scalar_parity": scalar_parity,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ("git", "status", "--porcelain"), cwd=ROOT, text=True
                ).strip()
            ),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": budget.elapsed,
            "cap_seconds": args.cap_seconds,
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "jit_compile": False,
            "dtype": "float64",
            "worker_count": 4,
            "cores_per_worker": 1,
            "source_paths": {
                key: path.as_posix() for key, path in source_paths.items()
            },
            "source_sha256": {
                key: _sha256(ROOT / path) for key, path in source_paths.items()
            },
        },
        "decision_roles": {
            "resource_checks": "hard_veto",
            "finite_and_shape_checks": "hard_veto",
            "timings": "explanatory_diagnostic",
            "scalar_parity": "diagnostic_implementation_check",
        },
        "nonclaims": [
            "profile only",
            "no optimizer update or NeuTra training result",
            "no HMC readiness or posterior correctness claim",
            "no CPU/GPU or architecture ranking",
        ],
    }
    _write_json(result_path, payload)
    _write_json(
        progress_path,
        _progress_payload(
            budget=budget,
            startup=startup,
            calls=calls,
            status=payload["status"],
            error=error,
        ),
        replace=True,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.cap_seconds)
        or args.cap_seconds <= 0.0
        or args.cap_seconds > 1800.0
    ):
        parser.error("--cap-seconds must be in (0, 1800]")
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "error": payload["error"],
                "derived": payload["derived"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
