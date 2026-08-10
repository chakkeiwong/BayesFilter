#!/usr/bin/env python3
"""Diagnostic CPU timing for q=20 batch-100 NeuTra training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROCESS_STARTED = time.perf_counter()
PARENT_INTRA_THREADS = 4
PARENT_INTER_THREADS = 1
WORKER_COUNT = 8
WORKER_INTRA_THREADS = 1
WORKER_INTER_THREADS = 1
CONFIGURED_TF_COMPUTE_THREADS = (
    PARENT_INTRA_THREADS
    + PARENT_INTER_THREADS
    + WORKER_COUNT * (WORKER_INTRA_THREADS + WORKER_INTER_THREADS)
)
THREAD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": str(PARENT_INTRA_THREADS),
    "TF_NUM_INTEROP_THREADS": str(PARENT_INTER_THREADS),
    "TF_CPP_MIN_LOG_LEVEL": "2",
}
IS_POOL_WORKER_IMPORT = os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1"
if not IS_POOL_WORKER_IMPORT:
    os.environ.update(THREAD_ENVIRONMENT)

import tensorflow as tf


tf.config.threading.set_intra_op_parallelism_threads(
    int(os.environ["TF_NUM_INTRAOP_THREADS"])
)
tf.config.threading.set_inter_op_parallelism_threads(
    int(os.environ["TF_NUM_INTEROP_THREADS"])
)
try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError as exc:
    raise RuntimeError("CUDA must be hidden before TensorFlow initialization") from exc
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("CPU timing diagnostic found a visible GPU")


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.cpu_value_score_pool import (  # noqa: E402
    CPUValueScorePool,
    CPUValueScorePoolConfig,
)
from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_neutra_training_timing.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-cpu-training-timing-plan-2026-07-22.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
Q = 20
BATCH_SIZE = 100
HIDDEN_LAYERS = (32, 32)
TIMED_STEPS = 5
CHECKPOINT_EVERY = 250
VALIDATION_BATCH_SIZE = 64
AUDIT_BATCH_SIZE = 256
TRAINING_STEPS = (250, 1250, 2000)
STREAM_COUNTS = (1, 2)
PARAMETERS = {
    "learning_rate": 4.0e-4,
    "initialization_scale": 0.01,
    "gradient_clip_norm": 10.0,
}
SEEDS = {
    "initialization": (20260719, 12101),
    "training": (20260719, 13101),
    "validation": (20260719, 14101),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _thread_count(pid: int) -> int | None:
    status = Path(f"/proc/{int(pid)}/status")
    try:
        lines = status.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError):
        return None
    for line in lines:
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1].strip())
    return None


def _thread_snapshot(worker_pids: Sequence[int]) -> Mapping[str, Any]:
    rows = []
    for pid in (os.getpid(), *tuple(int(item) for item in worker_pids)):
        rows.append(
            {
                "pid": pid,
                "role": "parent" if pid == os.getpid() else "target_worker",
                "native_os_thread_count": _thread_count(pid),
            }
        )
    known = [int(row["native_os_thread_count"]) for row in rows if row["native_os_thread_count"] is not None]
    return {
        "processes": rows,
        "native_os_thread_count_sum": sum(known),
        "native_os_thread_count_is_compute_pool_count": False,
        "note": "TensorFlow creates non-compute housekeeping threads; affinity and configured compute pools are the oversubscription controls.",
    }


def _pool_config() -> CPUValueScorePoolConfig:
    return CPUValueScorePoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_target_tf:"
            "complexity_target_worker_factory"
        ),
        worker_config={"q": Q},
        dimension=4,
        worker_count=WORKER_COUNT,
        cores_per_worker=1,
        timeout_seconds=600.0,
    )


def _trainer(target: Any) -> NeuTraReverseKLTrainer:
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(item) for item in PRIOR_CENTER.numpy()),
        target_parameter_names=target.parameter_names,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=PARAMETERS["learning_rate"],
        initialization_scale=PARAMETERS["initialization_scale"],
        gradient_clip_norm=PARAMETERS["gradient_clip_norm"],
        initialization_seed=SEEDS["initialization"],
        jit_compile=False,
    )
    return NeuTraReverseKLTrainer(target, config)


def _latent_batch(step: int, size: int = BATCH_SIZE) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(SEEDS["training"], tf.int32), int(step)
    )
    return tf.random.stateless_normal((int(size), 4), seed, dtype=tf.float64)


def _full_update(
    trainer: NeuTraReverseKLTrainer,
    pool: CPUValueScorePool,
    *,
    step: int,
) -> Mapping[str, Any]:
    total_started = time.perf_counter()
    z = _latent_batch(step)
    forward_started = time.perf_counter()
    theta, _logdet = trainer.forward_and_logdet(z)
    theta_host = theta.numpy()
    forward_seconds = time.perf_counter() - forward_started
    target_started = time.perf_counter()
    values, scores, metadata = pool.evaluate(
        theta_host,
        request_id=f"q20-cpu-timing-train-{step}",
    )
    target_seconds = time.perf_counter() - target_started
    update_started = time.perf_counter()
    result = trainer.train_step_with_external_value_score(z, values, scores)
    loss = float(result.loss.numpy())
    gradient_norm = float(result.gradient_norm.numpy())
    update_seconds = time.perf_counter() - update_started
    if not math.isfinite(loss) or not math.isfinite(gradient_norm):
        raise FloatingPointError("CPU training update returned nonfinite diagnostics")
    return {
        "step": step,
        "batch_size": BATCH_SIZE,
        "forward_seconds": forward_seconds,
        "target_value_score_seconds": target_seconds,
        "optimizer_update_seconds": update_seconds,
        "total_seconds": time.perf_counter() - total_started,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "worker_runtime_max_seconds": metadata["worker_runtime_max_seconds"],
        "worker_pids": metadata["startup_worker_pids"],
    }


def _shell() -> tf.Tensor:
    rows = [tf.zeros((4,), tf.float64)]
    for coordinate in range(4):
        direction = tf.one_hot(coordinate, 4, dtype=tf.float64) * 4.0
        rows.extend((direction, -direction))
    return tf.stack(rows)


def _checkpoint_cycle(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    pool: CPUValueScorePool,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    validation_z = tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        tf.constant(SEEDS["validation"], tf.int32),
        dtype=tf.float64,
    )
    validation_started = time.perf_counter()
    validation_theta, _ = trainer.forward_and_logdet(validation_z)
    validation_values, _metadata = pool.evaluate_values(
        validation_theta.numpy(), request_id="q20-cpu-timing-validation"
    )
    validation = trainer.validation_batch_with_external_value(
        validation_z, validation_values
    )
    validation_loss = float(tf.reduce_mean(validation.per_sample_loss).numpy())
    validation_seconds = time.perf_counter() - validation_started

    support_started = time.perf_counter()
    payload = trainer.frozen_transport_payload(
        transport_id="q20-cpu-timing-checkpoint",
        target_signature=target.target_signature(),
    )
    frozen = load_frozen_neutra_artifact(
        payload, expected_target_signature=target.target_signature()
    ).transport
    shell = _shell()
    theta = frozen.forward_batch(shell)
    values, scores, _metadata = pool.evaluate(
        theta.numpy(), request_id="q20-cpu-timing-support"
    )
    replay = frozen.inverse_theta_to_z_batch(theta)
    transformed_score = frozen.pullback_score_batch(
        shell, tf.convert_to_tensor(scores, tf.float64)
    ) + frozen.log_abs_det_jacobian_score_batch(shell)
    support_finite = bool(
        tf.reduce_all(tf.math.is_finite(theta)).numpy()
        and tf.reduce_all(tf.math.is_finite(replay)).numpy()
        and tf.reduce_all(tf.math.is_finite(transformed_score)).numpy()
        and all(math.isfinite(float(item)) for item in values)
    )
    support_seconds = time.perf_counter() - support_started
    if not math.isfinite(validation_loss) or not support_finite:
        raise FloatingPointError("CPU checkpoint cycle returned nonfinite diagnostics")
    return {
        "validation_seconds": validation_seconds,
        "support_probe_seconds": support_seconds,
        "total_seconds": time.perf_counter() - started,
        "validation_mean_loss": validation_loss,
        "support_all_finite": support_finite,
    }


def _terminal_audit(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    pool: CPUValueScorePool,
) -> Mapping[str, Any]:
    started = time.perf_counter()
    payload = trainer.frozen_transport_payload(
        transport_id="q20-cpu-timing-terminal-audit",
        target_signature=target.target_signature(),
    )
    frozen = load_frozen_neutra_artifact(
        payload, expected_target_signature=target.target_signature()
    ).transport
    z = tf.random.stateless_normal(
        (AUDIT_BATCH_SIZE, 4),
        tf.random.experimental.stateless_fold_in(
            tf.constant(SEEDS["validation"], tf.int32), 20260721
        ),
        dtype=tf.float64,
    )
    theta = frozen.forward_batch(z)
    logdet = frozen.log_abs_det_jacobian_batch(z)
    values, _metadata = pool.evaluate_values(
        theta.numpy(), request_id="q20-cpu-timing-terminal-audit"
    )
    losses = -tf.convert_to_tensor(values, tf.float64) - logdet
    mean_loss = float(tf.reduce_mean(losses).numpy())
    if not math.isfinite(mean_loss):
        raise FloatingPointError("CPU terminal audit returned nonfinite loss")
    return {
        "batch_size": AUDIT_BATCH_SIZE,
        "seconds": time.perf_counter() - started,
        "mean_loss": mean_loss,
    }


def _extrapolations(
    *,
    process_setup_seconds: float,
    target_construction_seconds: float,
    trainer_construction_seconds: float,
    pool_startup_seconds: float,
    cold_step_seconds: float,
    steady_rows: Sequence[Mapping[str, Any]],
    checkpoint_seconds: float,
    terminal_audit_seconds: float,
) -> Sequence[Mapping[str, Any]]:
    steady = [float(row["total_seconds"]) for row in steady_rows]
    center = statistics.mean(steady)
    low = min(steady)
    high = max(steady)
    rows = []
    fixed_once = process_setup_seconds + target_construction_seconds + pool_startup_seconds
    for streams in STREAM_COUNTS:
        for steps in TRAINING_STEPS:
            checkpoints = steps // CHECKPOINT_EVERY

            def estimate(step_seconds: float) -> float:
                per_stream = (
                    trainer_construction_seconds
                    + checkpoint_seconds
                    + cold_step_seconds
                    + max(0, steps - 1) * step_seconds
                    + checkpoints * checkpoint_seconds
                    + terminal_audit_seconds
                )
                return fixed_once + streams * per_stream

            rows.append(
                {
                    "stream_count": streams,
                    "steps_per_stream": steps,
                    "checkpoint_count_per_stream": checkpoints,
                    "mean_based_seconds": estimate(center),
                    "observed_step_min_sensitivity_seconds": estimate(low),
                    "observed_step_max_sensitivity_seconds": estimate(high),
                    "mean_based_hours": estimate(center) / 3600.0,
                    "sensitivity_role": "descriptive_not_confidence_interval",
                }
            )
    return rows


def run() -> Mapping[str, Any]:
    if CONFIGURED_TF_COMPUTE_THREADS > 50:
        raise RuntimeError("configured TensorFlow compute-pool budget exceeds 50")
    affinity = sorted(int(item) for item in os.sched_getaffinity(0))
    if len(affinity) > 50:
        raise RuntimeError("CPU affinity exceeds the requested 50-CPU limit")
    setup_seconds = time.perf_counter() - PROCESS_STARTED
    target_started = time.perf_counter()
    target = complexity_posterior_target(Q, jit_compile=False)
    target_construction_seconds = time.perf_counter() - target_started
    trainer_started = time.perf_counter()
    trainer = _trainer(target)
    trainer_construction_seconds = time.perf_counter() - trainer_started
    pool_started = time.perf_counter()
    with CPUValueScorePool(_pool_config()) as pool:
        startup_rows = tf.repeat(PRIOR_CENTER[None, :], repeats=WORKER_COUNT, axis=0)
        startup_values, startup_metadata = pool.evaluate_values(
            startup_rows.numpy(), request_id="q20-cpu-timing-pool-startup"
        )
        pool_startup_seconds = time.perf_counter() - pool_started
        if not all(math.isfinite(float(item)) for item in startup_values):
            raise FloatingPointError("CPU pool startup returned nonfinite values")
        cold = _full_update(trainer, pool, step=1)
        steady = [
            _full_update(trainer, pool, step=step)
            for step in range(2, 2 + TIMED_STEPS)
        ]
        checkpoint = _checkpoint_cycle(trainer, target, pool)
        terminal = _terminal_audit(trainer, target, pool)
        threads = _thread_snapshot(startup_metadata["startup_worker_pids"])
        if int(threads["native_os_thread_count_sum"]) > 50:
            raise RuntimeError(
                "realized process-tree native thread count exceeds the strict 50-thread cap"
            )
    extrapolations = _extrapolations(
        process_setup_seconds=setup_seconds,
        target_construction_seconds=target_construction_seconds,
        trainer_construction_seconds=trainer_construction_seconds,
        pool_startup_seconds=pool_startup_seconds,
        cold_step_seconds=float(cold["total_seconds"]),
        steady_rows=steady,
        checkpoint_seconds=float(checkpoint["total_seconds"]),
        terminal_audit_seconds=float(terminal["seconds"]),
    )
    totals = [float(row["total_seconds"]) for row in steady]
    sources = {
        "benchmark": SCRIPT,
        "plan": PLAN,
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "pool": Path("bayesfilter/inference/cpu_value_score_pool.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    }
    return {
        "schema": SCHEMA,
        "status": "DIAGNOSTIC_COMPLETED",
        "question": "CPU wall-time estimate for q=20 batch-100 (32,32) NeuTra training",
        "configuration": {
            "q": Q,
            "batch_size": BATCH_SIZE,
            "hidden_layers": list(HIDDEN_LAYERS),
            "stages": 3,
            "jit_compile_optimizer": False,
            "target_backend": "8 persistent scalar CPU workers",
            "training_parameters": PARAMETERS,
            "timed_steady_steps": TIMED_STEPS,
            "checkpoint_every_steps": CHECKPOINT_EVERY,
        },
        "thread_contract": {
            "configured_tf_compute_pool_threads": CONFIGURED_TF_COMPUTE_THREADS,
            "parent_intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "parent_inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
            "worker_count": WORKER_COUNT,
            "worker_intra_op_threads_each": WORKER_INTRA_THREADS,
            "worker_inter_op_threads_each": WORKER_INTER_THREADS,
            "cpu_affinity": affinity,
            "cpu_affinity_count": len(affinity),
            "environment": THREAD_ENVIRONMENT,
            "realized_native_threads": threads,
        },
        "timings": {
            "process_import_and_setup_seconds": setup_seconds,
            "target_construction_seconds": target_construction_seconds,
            "trainer_construction_seconds": trainer_construction_seconds,
            "pool_startup_and_warm_value_seconds": pool_startup_seconds,
            "cold_first_full_update": cold,
            "steady_full_updates": steady,
            "steady_total_seconds_summary": {
                "count": len(totals),
                "mean": statistics.mean(totals),
                "median": statistics.median(totals),
                "sample_sd": statistics.stdev(totals),
                "min": min(totals),
                "max": max(totals),
                "inference_role": "descriptive_only",
            },
            "checkpoint_validation_and_support": checkpoint,
            "terminal_audit": terminal,
            "measured_diagnostic_wall_seconds": time.perf_counter() - PROCESS_STARTED,
        },
        "extrapolations": extrapolations,
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "random_seeds": SEEDS,
            "host_ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "plan": PLAN.as_posix(),
            "source_paths": {key: path.as_posix() for key, path in sources.items()},
            "source_sha256": {key: _sha256(ROOT / path) for key, path in sources.items()},
        },
        "inference_status": {
            "hard_veto_screen": "passed finite execution and configured thread/affinity bounds",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all timing variation and extrapolation ranges",
            "default_readiness": "not_assessed",
            "next_evidence_needed": "none for the bounded CPU timing question",
        },
        "nonclaims": [
            "diagnostic timing only; trained state is discarded",
            "no transport-quality, HMC-readiness, or posterior-correctness claim",
            "no CPU/GPU ranking claim",
            "five steady steps do not estimate all long-run contention or thermal variation",
            "native OS housekeeping-thread count is not the configured compute-pool budget",
            "CPU timing does not change the repository GPU training default",
        ],
        "repair_history": [
            {
                "attempt": 1,
                "classification": "infrastructure_environment_mismatch_before_timing",
                "failure": (
                    "spawn re-import overwrote the pool-provided one-thread "
                    "TF_NUM_* worker environment with parent settings"
                ),
                "repair": (
                    "preserve the pool worker environment when "
                    "BAYESFILTER_CPU_VALUE_SCORE_WORKER=1"
                ),
                "scientific_interpretation": "none",
            },
            {
                "attempt": 2,
                "classification": "extrapolation_bookkeeping_repair",
                "failure": (
                    "the first completed diagnostic extrapolation omitted one "
                    "initial validation/support cycle per stream and treated "
                    "trainer construction as campaign-fixed"
                ),
                "repair": (
                    "measure target/trainer construction separately and charge "
                    "trainer construction plus the initial checkpoint per stream"
                ),
                "scientific_interpretation": "timing measurements remain valid; corrected extrapolation required",
            },
            {
                "attempt": 3,
                "classification": "strict_thread_cap_repair",
                "failure": (
                    "the relaxed r2 run respected 50-CPU affinity and configured "
                    "compute pools but TensorFlow CPU XLA created 213 native OS threads"
                ),
                "repair": (
                    "use the explicit CPU diagnostic non-XLA exception with 4/1 "
                    "parent TensorFlow pools and 8 one-core workers; fail closed "
                    "when measured process-tree native threads exceed 50"
                ),
                "scientific_interpretation": (
                    "r2 remains a relaxed XLA timing diagnostic; r3 is required "
                    "for the user's literal thread-cap question"
                ),
            },
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must remain inside the repository")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = run()
    output.write_bytes(_canonical(payload))
    print(json.dumps({"status": payload["status"], "output": output.relative_to(ROOT).as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
