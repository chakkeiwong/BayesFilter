#!/usr/bin/env python3
"""Distributed one-chain-per-process q=20 CPU/XLA HMC validation.

This route is deliberately separate from the four-chain GPU-oriented runner.
Each worker owns one chain and preserves its final state between fixed-size
XLA chunks. The parent owns tuning decisions, archive/progress writes, and
cross-chain diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import select
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# The supervisor also imports TensorFlow for diagnostics and tensor archives.
# Hide CUDA before any repository or framework import in this process.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-plan-2026-08-01.md"
)
REPAIR_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-cpu-xla-32x1-hmc-tuning-repair-plan-2026-08-01.md"
)
CHECKPOINTS = {
    "chart-a": Path(
        "docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/"
        "r1/seed-a/seed-a/checkpoint-1500.json"
    ),
    "chart-b": Path(
        "docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/"
        "r1/seed-b/seed-b/checkpoint-2500.json"
    ),
}
CHARTS = ("chart-a", "chart-b")
WORKER_CPUS = tuple(range(32))
SUPERVISOR_CPU = 32
CHAIN_COUNT_PER_CHART = 16
TOTAL_WORKERS = 32
TUNING_L = (2, 4)
TUNING_STEPS = (0.4, 0.5656854249492381, 0.75, 1.0)
REPAIR_TUNING_ARMS = (
    (1, 0.75),
    (1, 0.875),
    (1, 1.0),
    (2, 0.50),
    (2, 0.525),
    (2, 0.55),
)
TUNE_RESULTS = 32
TUNE_BURNIN = 16
TUNE_REPLICATIONS = 2
CONFIRM_RESULTS = 64
CONFIRM_BURNIN = 16
WARM_CHUNK = 500
WARM_MIN = 2000
WARM_WINDOW = 1000
WARM_MAX = 10000
RETAINED_CHUNK = 500
RETAINED_MIN = 1000
RETAINED_MAX = 10000
TARGET_ACCEPT = 0.70
TUNE_POOL_MIN = 0.55
TUNE_POOL_MAX = 0.85
CONFIRM_CHAIN_MIN = 0.35
CONFIRM_CHAIN_MAX = 0.95
RHAT_WARM = 1.05
RHAT_RETAINED = 1.01
ESS_RETAINED = 400.0
STEP_SIZE_CANARY = 0.01
MAX_ABS_LOG_ACCEPT = 1000.0
HOST_RSS_CAP = 64 * 1024**3
CAMPAIGN_CAP = 86400.0
WORKER_START_TIMEOUT = 900.0
COMMAND_TIMEOUT = 1800.0
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)


class ValidationError(RuntimeError):
    pass


def selected_plan(profile: str) -> Path:
    if profile == "original-v1":
        return PLAN
    if profile == "short-trajectory-repair-v1":
        return REPAIR_PLAN
    raise ValidationError(f"unknown tuning profile: {profile}")


def tuning_arms(profile: str) -> tuple[tuple[int, float], ...]:
    if profile == "original-v1":
        arms = tuple(
            (leapfrog, step) for leapfrog in TUNING_L for step in TUNING_STEPS
        )
    elif profile == "short-trajectory-repair-v1":
        arms = REPAIR_TUNING_ARMS
    else:
        raise ValidationError(f"unknown tuning profile: {profile}")
    if any(leapfrog < 2 for leapfrog, _step in arms):
        raise ValidationError("HMC tuning forbids num_leapfrog_steps < 2")
    return arms


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValidationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ValidationError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(json_safe(payload)))
    temporary.replace(path)


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def wait_until(start_ns: int) -> None:
    while True:
        remaining = int(start_ns) - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)


def checkpoint_info(label: str) -> Mapping[str, Any]:
    path = (ROOT / CHECKPOINTS[label]).resolve()
    payload = strict_json(path)
    supplied = payload.get("checkpoint_hash")
    if not isinstance(supplied, str):
        raise ValidationError(f"{label} checkpoint hash missing")
    from bayesfilter.inference.neutra_training_control import validate_joint_training_checkpoint

    validate_joint_training_checkpoint(payload)
    best = payload.get("best_trainer_state")
    if not isinstance(best, Mapping):
        raise ValidationError(f"{label} best trainer state missing")
    controller = payload.get("controller_state")
    if not isinstance(controller, Mapping):
        raise ValidationError(f"{label} controller state missing")
    controller_best_step = int(controller["best_step"])
    trainer_step = int(payload["trainer_state"]["step"])
    if str(controller["best_trainer_state_hash"]) != str(best["state_hash"]):
        raise ValidationError(f"{label} best trainer state hash metadata inconsistent")
    checkpoint_program_step = int(path.stem.split("-")[-1])
    return {
        "label": label,
        "path": CHECKPOINTS[label].as_posix(),
        "sha256": sha256(path),
        "checkpoint_hash": supplied,
        "controller_best_step": controller_best_step,
        "checkpoint_program_step": checkpoint_program_step,
        "trainer_state_step": trainer_step,
        "best_trainer_state_step": int(best["step"]),
        "best_trainer_state_hash": str(best["state_hash"]),
    }


def _worker_main(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValidationError("worker requires CUDA_VISIBLE_DEVICES=-1")
    import tensorflow as tf

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise ValidationError("worker found visible GPU")

    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.hmc import (
        FixedSizeHMCChunkConfig,
        FullChainHMCConfig,
        build_fixed_size_hmc_chunk_runner,
        build_reusable_full_chain_tfp_hmc_runner,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraReverseKLTrainer,
        ssl_lstm_tuned_capacity_neutra_config,
    )
    from bayesfilter.inference.neutra_training_control import validate_joint_training_checkpoint
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES, PRIOR_CENTER

    label = str(args.chart)
    plan_path = selected_plan(str(args.tuning_profile))
    checkpoint_path = (ROOT / CHECKPOINTS[label]).resolve()
    checkpoint = strict_json(checkpoint_path)
    validate_joint_training_checkpoint(checkpoint)
    best_state = checkpoint["best_trainer_state"]
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(v) for v in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=0.0004,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260719, 12101 if label == "chart-a" else 12102),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, config)
    trainer.restore_state(best_state)
    frozen = trainer.frozen_transport_payload(
        transport_id=f"{label}-cpu-xla-best-{best_state['step']}-distributed-hmc",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(frozen, expected_target_signature=target.target_signature())

    class Bridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True

        def __init__(self) -> None:
            self.target_scope = f"{target.target_scope}:distributed:{label}"

        def adapter_signature(self) -> str:
            return target.adapter_signature()

        def target_signature(self) -> str:
            return target.target_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_cpu_xla_distributed_hmc_bridge",
                evidence_path=plan_path.as_posix(),
                target_scope=self.target_scope,
                nonclaims=("distributed CPU-XLA validation only", "no posterior oracle"),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            values = tf.convert_to_tensor(values, tf.float64)
            if values.shape.rank != 2 or values.shape[0] != 1:
                raise ValueError("worker chain target requires shape [1,4]")
            return target.batch_value_and_score(values)

        def log_prob_and_grad_status(self, values: Any) -> tuple[Any, Any, Mapping[str, Any]]:
            return target.neutra_batch_log_prob_and_grad_status(values)

    base = Bridge()
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope=f"{base.target_scope}:best-{best_state['step']}",
        runtime_backend="ssl_lstm_q20_cpu_xla_distributed_fixed_transport_hmc",
        evidence_path=plan_path.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=("CPU-XLA distributed validation only", "no posterior oracle"),
    )
    initial = tf.constant((INITIAL_Z[int(args.chain_index) % 4],), tf.float64)
    runners: dict[tuple[int, int, int], Any] = {}
    full_runners: dict[tuple[int, int, int], Any] = {}

    def fixed_runner(leapfrog: int, max_results: int, burnin: int) -> Any:
        key = (int(leapfrog), int(max_results), int(burnin))
        if key not in runners:
            cfg = FixedSizeHMCChunkConfig(
                max_results=max_results,
                num_burnin_steps=burnin,
                step_size=STEP_SIZE_CANARY,
                num_leapfrog_steps=leapfrog,
                seed=(20260801, 70000 + int(args.chain_index)),
                use_xla=True,
                trace_policy="standard",
                target_scope=adapter.target_scope,
            )
            runners[key] = build_fixed_size_hmc_chunk_runner(adapter, initial, cfg)
        return runners[key]

    def audit_state(audit_state: Any) -> Mapping[str, Any]:
        value, score, status = adapter.log_prob_and_grad_status(audit_state)
        status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
        valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
        floors = tf.convert_to_tensor(status["floor_count_value"], tf.int32)
        min_eigen = tf.convert_to_tensor(status["min_innovation_eigenvalue"], tf.float64)
        all_finite = bool(
            tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value, tf.float64))).numpy()
        ) and bool(
            tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(score, tf.float64))).numpy()
        )
        status_valid = bool(
            tf.reduce_all(
                tf.logical_and(
                    tf.equal(status_code, 0),
                    tf.logical_and(
                        valid,
                        tf.logical_and(
                            tf.equal(floors, 0),
                            tf.logical_and(tf.math.is_finite(min_eigen), min_eigen > 0.0),
                        ),
                    ),
                )
            ).numpy()
        )
        return {
            "target_status_valid": status_valid,
            "target_status_code_max": int(tf.reduce_max(status_code).numpy()),
            "target_status_floor_count_total": int(tf.reduce_sum(floors).numpy()),
            "target_status_min_innovation_eigenvalue": float(tf.reduce_min(min_eigen).numpy()),
            "audited_value_score_all_finite": all_finite,
        }

    def tune_run(step: float, leapfrog: int, results: int, burnin: int, seed: tuple[int, int]) -> Mapping[str, Any]:
        key = (int(leapfrog), int(results), int(burnin))
        runner = full_runners.get(key)
        if runner is None:
            cfg = FullChainHMCConfig(
                num_results=results,
                num_burnin_steps=burnin,
                step_size=STEP_SIZE_CANARY,
                num_leapfrog_steps=leapfrog,
                seed=(20260801, 71000 + int(args.chain_index)),
                use_xla=True,
                trace_policy="standard",
                target_scope=adapter.target_scope,
            )
            runner = build_reusable_full_chain_tfp_hmc_runner(adapter, initial, cfg)
            full_runners[key] = runner
        result = runner.run(seed=seed, step_size=step)
        log_ratio = tf.convert_to_tensor(result.trace["log_accept_ratio"], tf.float64)
        alpha = tf.minimum(tf.ones_like(log_ratio), tf.exp(log_ratio))
        accepted = tf.convert_to_tensor(result.trace["is_accepted"], tf.bool)
        samples = tf.convert_to_tensor(result.samples, tf.float64)
        target_log_prob = tf.convert_to_tensor(result.trace["target_log_prob"], tf.float64)
        proposed_target_log_prob = tf.convert_to_tensor(
            result.trace["proposed_target_log_prob"], tf.float64
        )
        path = tf.concat((initial[tf.newaxis, ...], samples), axis=0)
        moved = bool(tf.reduce_any(tf.not_equal(path[1:], path[:-1])).numpy())
        final_audit = audit_state(samples[-1])
        finite = all(
            bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
            for value in (samples, log_ratio, target_log_prob, proposed_target_log_prob)
        ) and bool(final_audit["audited_value_score_all_finite"])
        native_available = result.diagnostics["native_divergence_status"] == "available"
        divergence_value = result.diagnostics.get("divergence_count")
        divergence_count = (
            0
            if divergence_value is None
            else int(tf.convert_to_tensor(divergence_value, tf.int32).numpy())
        )
        return {
            "mean_acceptance_probability": float(tf.reduce_mean(alpha).numpy()),
            "binary_acceptance_rate": float(tf.reduce_mean(tf.cast(accepted, tf.float64)).numpy()),
            "moved": moved,
            "all_finite": finite,
            "sample_shape": [int(v) for v in samples.shape],
            "wall_seconds": float(result.metadata["sample_chain_call_s"]),
            "log_accept_max_abs": float(tf.reduce_max(tf.abs(log_ratio)).numpy()),
            "log_accept_ratio_nonfinite_count": int(
                tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(log_ratio)), tf.int32)).numpy()
            ),
            "target_log_prob_nonfinite_count": int(
                tf.reduce_sum(
                    tf.cast(
                        tf.logical_not(
                            tf.math.is_finite(
                                tf.concat(
                                    (
                                        tf.reshape(target_log_prob, (-1,)),
                                        tf.reshape(proposed_target_log_prob, (-1,)),
                                    ),
                                    axis=0,
                                )
                            )
                        ),
                        tf.int32,
                    )
                ).numpy()
            ),
            "native_divergence_available": native_available,
            "divergence_count": divergence_count,
            "jit_compile": bool(result.metadata.get("use_xla", False)),
            "sample_chain_invocation_count": int(
                result.metadata["sample_chain_invocation_count"]
            ),
            "affinity": sorted(os.sched_getaffinity(0)),
            "rss_bytes": rss_bytes(),
            **final_audit,
        }

    startup = time.perf_counter()
    ready = {
        "schema": "bayesfilter.ssl_lstm.q20_cpu_xla_distributed_hmc_worker.v1",
        "event": "ready",
        "chart": label,
        "chain_index": int(args.chain_index),
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [v.name for v in tf.config.list_physical_devices("GPU")],
        "tensorflow": tf.__version__,
        "jit_compile": True,
        "dtype": "float64",
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "best_trainer_state_hash": best_state["state_hash"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "startup_seconds": time.perf_counter() - startup,
        "rss_bytes": rss_bytes(),
    }
    print(json.dumps(ready, allow_nan=False), flush=True)
    state = initial
    current_kernel: tuple[float, int] | None = None
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        request = json.loads(line)
        command = request.get("command")
        if command == "stop":
            print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
            return 0
        if command == "tune":
            seed = (20260801, int(request["seed_base"]) + int(args.chain_index))
            result = tune_run(float(request["step_size"]), int(request["num_leapfrog_steps"]), int(request["num_results"]), int(request["num_burnin"]), seed)
            result.update({"event": "tune_done", "chart": label, "chain_index": int(args.chain_index), "arm_id": request["arm_id"]})
            print(json.dumps(json_safe(result), allow_nan=False), flush=True)
            continue
        if command == "confirm":
            seed = (20260801, int(request["seed_base"]) + int(args.chain_index))
            result = tune_run(float(request["step_size"]), int(request["num_leapfrog_steps"]), int(request["num_results"]), int(request["num_burnin"]), seed)
            result.update({"event": "confirm_done", "chart": label, "chain_index": int(args.chain_index), "arm_id": request["arm_id"]})
            print(json.dumps(json_safe(result), allow_nan=False), flush=True)
            continue
        if command == "chunk":
            step = float(request["step_size"])
            leapfrog = int(request["num_leapfrog_steps"])
            max_results = int(request["max_results"])
            if current_kernel != (step, leapfrog):
                current_kernel = (step, leapfrog)
            runner = fixed_runner(leapfrog, max_results, 0)
            wait_until(int(request["start_ns"]))
            tick = time.perf_counter()
            incoming_state = tf.identity(state)
            result = runner.run(active_results=int(request["active_results"]), current_state=incoming_state, seed=(20260801, int(request["seed_base"]) + int(args.chain_index)), step_size=step)
            elapsed = time.perf_counter() - tick
            state = tf.convert_to_tensor(result.final_state, tf.float64)
            samples = tf.boolean_mask(tf.convert_to_tensor(result.samples, tf.float64), tf.convert_to_tensor(result.valid_mask, tf.bool))
            samples = tf.squeeze(samples, axis=1)
            trace = result.trace
            log_ratio = tf.boolean_mask(tf.convert_to_tensor(trace["log_accept_ratio"], tf.float64), tf.convert_to_tensor(result.valid_mask, tf.bool))
            log_ratio = tf.reshape(log_ratio, (-1,))
            accepted = tf.boolean_mask(tf.convert_to_tensor(trace["is_accepted"], tf.bool), tf.convert_to_tensor(result.valid_mask, tf.bool))
            accepted = tf.reshape(accepted, (-1,))
            finite = bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()) and bool(tf.reduce_all(tf.math.is_finite(log_ratio)).numpy())
            path = tf.concat((incoming_state, samples), axis=0)
            moved = bool(tf.reduce_any(tf.not_equal(path[1:], path[:-1])).numpy())
            final_audit = audit_state(state)
            response = {
                "event": "chunk_done",
                "chart": label,
                "chain_index": int(args.chain_index),
                "chunk_index": int(request["chunk_index"]),
                "wall_seconds": elapsed,
                "all_finite": finite and bool(final_audit["audited_value_score_all_finite"]),
                "moved": moved,
                "samples": samples.numpy().tolist(),
                "mapped_samples": loaded.transport.forward_batch(samples).numpy().tolist(),
                "log_accept_ratio": log_ratio.numpy().tolist(),
                "is_accepted": accepted.numpy().tolist(),
                "acceptance_probability_mean": float(tf.reduce_mean(tf.minimum(tf.ones_like(log_ratio), tf.exp(log_ratio))).numpy()),
                "log_accept_max_abs": float(tf.reduce_max(tf.abs(log_ratio)).numpy()) if int(log_ratio.shape[0]) else 0.0,
                "target_log_prob_nonfinite_count": int(trace["target_log_prob_nonfinite_count"].numpy()),
                "log_accept_ratio_nonfinite_count": int(trace["log_accept_ratio_nonfinite_count"].numpy()),
                "native_divergence_available": bool(trace["native_divergence_available"].numpy()),
                "divergence_count": int(trace["divergence_count"].numpy()),
                "jit_compile": bool(result.metadata["jit_compile"]),
                "compile_trace_count": result.metadata["compile_trace_count"],
                "chunk_invocation_count": int(result.metadata["chunk_invocation_count"]),
                "affinity": sorted(os.sched_getaffinity(0)),
                "rss_bytes": rss_bytes(),
                **final_audit,
            }
            print(json.dumps(json_safe(response), allow_nan=False), flush=True)
            continue
        raise ValidationError(f"unknown worker command: {command}")


def worker_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_FORCE_GPU_ALLOW_GROWTH": "false",
        "OMP_NUM_THREADS": "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TF_CPP_MIN_LOG_LEVEL": "1",
    })
    return env


def read_line(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise ValidationError("worker stdout unavailable")
    ready, _, _ = select.select([process.stdout], [], [], float(timeout))
    if not ready:
        raise ValidationError(f"worker {process.pid} timed out")
    line = process.stdout.readline()
    if not line:
        raise ValidationError(f"worker {process.pid} closed stdout")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValidationError("worker response is not an object")
    return value


def send_workers(
    processes: Sequence[subprocess.Popen[str]], payload: Mapping[str, Any]
) -> None:
    line = json.dumps(payload, allow_nan=False) + "\n"
    for process in processes:
        if process.stdin is None:
            raise ValidationError("worker stdin unavailable")
        process.stdin.write(line)
        process.stdin.flush()


def collect_workers(
    processes: Sequence[subprocess.Popen[str]], *, timeout: float
) -> tuple[dict[str, Any], ...]:
    return tuple(read_line(process, timeout) for process in processes)


def command_workers(processes: Sequence[subprocess.Popen[str]], payload: Mapping[str, Any], *, timeout: float) -> tuple[dict[str, Any], ...]:
    send_workers(processes, payload)
    return collect_workers(processes, timeout=timeout)


def start_workers(
    root: Path, *, tuning_profile: str
) -> tuple[list[subprocess.Popen[str]], list[Any], tuple[dict[str, Any], ...]]:
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    for index, cpu in enumerate(WORKER_CPUS):
        chart = "chart-a" if index < CHAIN_COUNT_PER_CHART else "chart-b"
        chain = index if index < CHAIN_COUNT_PER_CHART else index - CHAIN_COUNT_PER_CHART
        log = (root / f"worker-{index:02d}.stderr.log").open("w", encoding="utf-8")
        logs.append(log)
        command = [
            "taskset",
            "-c",
            str(cpu),
            sys.executable,
            str(SCRIPT),
            "--worker",
            "--chart",
            chart,
            "--chain-index",
            str(chain),
            "--tuning-profile",
            tuning_profile,
        ]
        process = subprocess.Popen(command, cwd=ROOT, env=worker_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        processes.append(process)
    ready = tuple(read_line(process, WORKER_START_TIMEOUT) for process in processes)
    if any(row.get("event") != "ready" for row in ready):
        raise ValidationError("worker readiness failed")
    for index, row in enumerate(ready):
        if row.get("affinity") != [WORKER_CPUS[index]]:
            raise ValidationError("worker affinity mismatch")
        if row.get("physical_gpus") != [] or row.get("cuda_visible_devices") != "-1":
            raise ValidationError("worker is not CPU-only")
        if row.get("jit_compile") is not True:
            raise ValidationError("worker is not XLA-enabled")
    return processes, logs, ready


def terminate_workers(processes: Sequence[subprocess.Popen[str]], logs: Sequence[Any]) -> None:
    for process in processes:
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                process.stdin.flush()
            except OSError:
                process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30.0)
    for log in logs:
        log.close()


def acceptance_probability(log_values: Sequence[float]) -> float:
    values = [min(1.0, math.exp(float(v))) for v in log_values]
    return sum(values) / len(values) if values else float("nan")


def chart_processes(
    processes: Sequence[subprocess.Popen[str]], label: str
) -> Sequence[subprocess.Popen[str]]:
    start = 0 if label == "chart-a" else CHAIN_COUNT_PER_CHART
    return processes[start : start + CHAIN_COUNT_PER_CHART]


def worker_rows_pass(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(
        len(rows) == CHAIN_COUNT_PER_CHART
        and all(
            bool(row.get("all_finite"))
            and bool(row.get("moved"))
            and bool(row.get("target_status_valid"))
            and int(row.get("log_accept_ratio_nonfinite_count", 1)) == 0
            and int(row.get("target_log_prob_nonfinite_count", 1)) == 0
            and float(row.get("log_accept_max_abs", math.inf)) <= MAX_ABS_LOG_ACCEPT
            and (
                not bool(row.get("native_divergence_available"))
                or int(row.get("divergence_count", 1)) == 0
            )
            and row.get("jit_compile") is True
            and row.get("affinity") == [
                (0 if str(row.get("chart")) == "chart-a" else CHAIN_COUNT_PER_CHART)
                + int(row.get("chain_index", -1))
            ]
            for row in rows
        )
        and sum(int(row.get("rss_bytes", HOST_RSS_CAP + 1)) for row in rows)
        <= HOST_RSS_CAP
    )


def tuning_candidate(
    *,
    label: str,
    leapfrog: int,
    step: float,
    replications: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    replication_passes = [
        bool(worker_rows_pass(replication["rows"]))
        and bool(replication["aggregate_all_worker_rss_passed"])
        and TUNE_POOL_MIN
        <= float(replication["mean_acceptance_probability"])
        <= TUNE_POOL_MAX
        for replication in replications
    ]
    mean_alpha = sum(
        float(replication["mean_acceptance_probability"])
        for replication in replications
    ) / len(replications)
    return {
        "arm_id": f"{label}-l{leapfrog}-s{step:g}",
        "chart": label,
        "num_leapfrog_steps": leapfrog,
        "step_size": step,
        "replication_count": len(replications),
        "replications": replications,
        "mean_acceptance_probability": mean_alpha,
        "replication_passes": replication_passes,
        "passed": bool(
            len(replications) == TUNE_REPLICATIONS and all(replication_passes)
        ),
    }


def tune_charts(
    processes: Sequence[subprocess.Popen[str]],
    progress: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Mapping[str, Any] | None]:
    candidates: dict[str, list[Mapping[str, Any]]] = {label: [] for label in CHARTS}
    arm_seed_domain = 90000 if profile == "original-v1" else 900000
    for arm_index, (leapfrog, step) in enumerate(tuning_arms(profile)):
        arm_rows: dict[str, list[Mapping[str, Any]]] = {label: [] for label in CHARTS}
        for replication in range(TUNE_REPLICATIONS):
            for label in CHARTS:
                send_workers(
                    chart_processes(processes, label),
                    {
                        "command": "tune",
                        "arm_id": f"{label}-l{leapfrog}-s{step:g}-r{replication + 1}",
                        "num_results": TUNE_RESULTS,
                        "num_burnin": TUNE_BURNIN,
                        "num_leapfrog_steps": leapfrog,
                        "step_size": step,
                        "seed_base": arm_seed_domain
                        + replication * 20000
                        + arm_index * 2000
                        + int(step * 1000),
                    },
                )
            replication_rows: dict[str, tuple[dict[str, Any], ...]] = {}
            for label in CHARTS:
                rows = collect_workers(
                    chart_processes(processes, label), timeout=COMMAND_TIMEOUT
                )
                replication_rows[label] = rows
            global_rss_passed = (
                sum(
                    int(row.get("rss_bytes", HOST_RSS_CAP + 1))
                    for rows in replication_rows.values()
                    for row in rows
                )
                <= HOST_RSS_CAP
            )
            for label in CHARTS:
                rows = replication_rows[label]
                arm_rows[label].append(
                    {
                        "replication": replication + 1,
                        "mean_acceptance_probability": sum(
                            float(row["mean_acceptance_probability"]) for row in rows
                        )
                        / len(rows),
                        "max_abs_log_accept": max(
                            float(row.get("log_accept_max_abs", math.inf)) for row in rows
                        ),
                        "aggregate_rss_bytes": sum(
                            int(row.get("rss_bytes", 0)) for row in rows
                        ),
                        "aggregate_all_worker_rss_passed": global_rss_passed,
                        "rows": rows,
                    }
                )
        for label in CHARTS:
            candidate = tuning_candidate(
                label=label,
                leapfrog=leapfrog,
                step=step,
                replications=arm_rows[label],
            )
            candidates[label].append(candidate)
            progress["tuning"].setdefault(label, []).append(candidate)
        write_json(ROOT / Path(progress["progress_path"]), progress, replace=True)

    selected: dict[str, Mapping[str, Any] | None] = {}
    for label in CHARTS:
        viable = [candidate for candidate in candidates[label] if candidate["passed"]]
        viable.sort(
            key=lambda candidate: (
                abs(float(candidate["mean_acceptance_probability"]) - TARGET_ACCEPT),
                int(candidate["num_leapfrog_steps"]),
                float(candidate["step_size"]),
            )
        )
        selected[label] = None if not viable else dict(viable[0])

    confirmation_labels = [label for label in CHARTS if selected[label] is not None]
    for label in confirmation_labels:
        kernel = selected[label]
        send_workers(
            chart_processes(processes, label),
            {
                "command": "confirm",
                "arm_id": f"{label}-confirmation",
                "num_results": CONFIRM_RESULTS,
                "num_burnin": CONFIRM_BURNIN,
                "num_leapfrog_steps": kernel["num_leapfrog_steps"],
                "step_size": kernel["step_size"],
                "seed_base": (140000 if profile == "original-v1" else 940000)
                + (0 if label == "chart-a" else 10000),
            },
        )
    confirmation_rows = {
        label: collect_workers(
            chart_processes(processes, label), timeout=COMMAND_TIMEOUT
        )
        for label in confirmation_labels
    }
    confirmation_global_rss_passed = (
        sum(
            int(row.get("rss_bytes", HOST_RSS_CAP + 1))
            for rows in confirmation_rows.values()
            for row in rows
        )
        <= HOST_RSS_CAP
    )
    for label in confirmation_labels:
        rows = confirmation_rows[label]
        confirmation = {
            "arm_id": f"{label}-confirmation",
            "per_chain_mean_acceptance_probability": [
                float(row["mean_acceptance_probability"]) for row in rows
            ],
            "aggregate_rss_bytes": sum(int(row.get("rss_bytes", 0)) for row in rows),
            "aggregate_all_worker_rss_passed": confirmation_global_rss_passed,
            "worker_hard_vetoes_passed": worker_rows_pass(rows),
            "passed": bool(
                worker_rows_pass(rows)
                and confirmation_global_rss_passed
                and all(
                    CONFIRM_CHAIN_MIN
                    <= float(row["mean_acceptance_probability"])
                    <= CONFIRM_CHAIN_MAX
                    for row in rows
                )
            ),
            "rows": rows,
        }
        progress["confirmations"][label] = confirmation
        selected[label] = (
            {**dict(selected[label]), "confirmation": confirmation}
            if confirmation["passed"]
            else None
        )
    write_json(ROOT / Path(progress["progress_path"]), progress, replace=True)
    return selected


def diagnostics(latent: Any, mapped: Any, *, rhat: float, warmup: bool) -> Mapping[str, Any]:
    import tensorflow as tf
    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
        rank_normalized_split_rhat_summary,
    )

    latent_tensor = tf.convert_to_tensor(latent, tf.float64)
    mapped_tensor = tf.convert_to_tensor(mapped, tf.float64)
    if warmup:
        latent_report = rank_normalized_split_rhat_summary(latent_tensor, rhat_max=rhat)
        mapped_report = rank_normalized_split_rhat_summary(mapped_tensor, rhat_max=rhat)
        return {
            "diagnostic_role": "warmup_readiness_rhat_only",
            "hmc_coordinates": latent_report,
            "model_parameters": mapped_report,
            "passed": bool(latent_report["passed"] and mapped_report["passed"]),
            "max_rhat": max(
                float(latent_report["max_finite_rhat"] or math.inf),
                float(mapped_report["max_finite_rhat"] or math.inf),
            ),
        }
    thresholds = RankNormalizedHMCThresholds(
        rhat_max=rhat, bulk_ess_min=ESS_RETAINED, tail_ess_min=ESS_RETAINED
    )
    latent_report = rank_normalized_hmc_diagnostics(latent_tensor, thresholds=thresholds)
    mapped_report = rank_normalized_hmc_diagnostics(mapped_tensor, thresholds=thresholds)
    return {
        "diagnostic_role": "retained_rhat_bulk_tail_ess",
        "hmc_coordinates": latent_report,
        "model_parameters": mapped_report,
        "passed": bool(latent_report["passed"] and mapped_report["passed"]),
        "max_rhat": max(float(latent_report["max_rhat"]), float(mapped_report["max_rhat"])),
        "min_bulk_ess": min(
            float(latent_report["min_bulk_ess"]), float(mapped_report["min_bulk_ess"])
        ),
        "min_tail_ess": min(
            float(latent_report["min_tail_ess"]), float(mapped_report["min_tail_ess"])
        ),
    }


def write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValidationError(f"tensor artifact already exists: {path}")
    serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.write_bytes(serialized)
    digest = sha256(path)
    if digest != hashlib.sha256(serialized).hexdigest():
        raise ValidationError(f"tensor artifact hash verification failed: {path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "bytes": len(serialized),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": tensor.dtype.name,
    }


def archive_chunk(
    output: Path,
    *,
    phase: str,
    label: str,
    chunk_index: int,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    import tensorflow as tf

    chunk_root = output / "chunks" / phase / label / f"chunk-{chunk_index:04d}"
    chain_manifests = []
    for chain, row in enumerate(rows):
        chain_root = chunk_root / f"chain-{chain:02d}"
        tensors = {
            "latent_samples": write_tensor(
                chain_root / "latent-samples.tftensor",
                tf.convert_to_tensor(row["samples"], tf.float64),
            ),
            "mapped_samples": write_tensor(
                chain_root / "mapped-samples.tftensor",
                tf.convert_to_tensor(row["mapped_samples"], tf.float64),
            ),
            "log_accept_ratio": write_tensor(
                chain_root / "log-accept-ratio.tftensor",
                tf.convert_to_tensor(row["log_accept_ratio"], tf.float64),
            ),
            "is_accepted": write_tensor(
                chain_root / "is-accepted.tftensor",
                tf.convert_to_tensor(row["is_accepted"], tf.bool),
            ),
        }
        chain_manifests.append(
            {
                "chain_index": chain,
                "worker_chain_index": int(row["chain_index"]),
                "tensors": tensors,
            }
        )
    manifest = {
        "schema": "bayesfilter.ssl_lstm.q20_cpu_xla_distributed_hmc_chunk.v1",
        "phase": phase,
        "chart": label,
        "chunk_index": chunk_index,
        "chain_count": len(rows),
        "draw_count_per_chain": len(rows[0]["samples"]),
        "chains": chain_manifests,
    }
    manifest_path = chunk_root / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "chain_count": len(rows),
        "draw_count_per_chain": len(rows[0]["samples"]),
    }


def chunk_vetoes(
    rows: Sequence[Mapping[str, Any]], *, expected_cpus: Sequence[int]
) -> list[str]:
    vetoes: list[str] = []
    if len(rows) != len(expected_cpus):
        return ["worker_response_count"]
    for local_index, (row, cpu) in enumerate(zip(rows, expected_cpus, strict=True)):
        prefix = f"chain-{local_index:02d}"
        if not bool(row.get("all_finite")):
            vetoes.append(f"{prefix}:nonfinite_state_or_target")
        if not bool(row.get("moved")):
            vetoes.append(f"{prefix}:unmoved")
        if not bool(row.get("target_status_valid")):
            vetoes.append(f"{prefix}:invalid_target_status")
        if int(row.get("target_log_prob_nonfinite_count", 1)) != 0:
            vetoes.append(f"{prefix}:nonfinite_target_log_prob")
        if int(row.get("log_accept_ratio_nonfinite_count", 1)) != 0:
            vetoes.append(f"{prefix}:nonfinite_log_accept_ratio")
        if float(row.get("log_accept_max_abs", math.inf)) > MAX_ABS_LOG_ACCEPT:
            vetoes.append(f"{prefix}:extreme_log_accept_ratio")
        if bool(row.get("native_divergence_available")) and int(
            row.get("divergence_count", 1)
        ) > 0:
            vetoes.append(f"{prefix}:native_divergence")
        if row.get("jit_compile") is not True:
            vetoes.append(f"{prefix}:non_xla")
        if row.get("affinity") != [int(cpu)]:
            vetoes.append(f"{prefix}:affinity_drift")
    if sum(int(row.get("rss_bytes", HOST_RSS_CAP + 1)) for row in rows) > HOST_RSS_CAP:
        vetoes.append("chart_worker_rss_cap")
    return vetoes


def run_preflight(
    processes: Sequence[subprocess.Popen[str]], progress: dict[str, Any]
) -> Mapping[str, Any]:
    payload = {
        "command": "chunk",
        "chunk_index": 0,
        "active_results": 2,
        "max_results": 2,
        "num_leapfrog_steps": 2,
        "step_size": STEP_SIZE_CANARY,
        "seed_base": 700000,
        "start_ns": time.monotonic_ns() + 1_000_000_000,
    }
    for label in CHARTS:
        send_workers(chart_processes(processes, label), payload)
    results = {
        label: collect_workers(
            chart_processes(processes, label), timeout=COMMAND_TIMEOUT
        )
        for label in CHARTS
    }
    vetoes = []
    for label in CHARTS:
        start = 0 if label == "chart-a" else CHAIN_COUNT_PER_CHART
        vetoes.extend(
            f"{label}:{veto}"
            for veto in chunk_vetoes(
                results[label], expected_cpus=WORKER_CPUS[start : start + CHAIN_COUNT_PER_CHART]
            )
        )
    if sum(int(row["rss_bytes"]) for rows in results.values() for row in rows) > HOST_RSS_CAP:
        vetoes.append("aggregate_worker_rss_cap")
    return {
        "schema": "bayesfilter.ssl_lstm.q20_cpu_xla_distributed_hmc_preflight.v1",
        "status": "PREFLIGHT_PASSED" if not vetoes else "PREFLIGHT_FAILED",
        "scientific_role": "mechanics_only_not_tuning_or_validation_evidence",
        "results": results,
        "vetoes": vetoes,
        "progress_path": progress["progress_path"],
    }


def terminal_summary(
    progress: Mapping[str, Any],
    *,
    started: float,
    campaign_cap: float,
    plan_path: Path,
) -> Mapping[str, Any]:
    elapsed = time.perf_counter() - started
    return {
        **progress,
        "elapsed_seconds": elapsed,
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ("git", "status", "--porcelain"), cwd=ROOT, text=True
                ).strip()
            ),
            "command": shlex.join([sys.executable, *sys.argv]),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "wall_seconds": elapsed,
            "cap_seconds": campaign_cap,
            "supervisor_affinity": sorted(os.sched_getaffinity(0)),
            "cpu_gpu_status": "CPU-only; CUDA hidden in supervisor and all workers",
            "jit_compile": True,
            "tuning_replications_per_arm": TUNE_REPLICATIONS,
            "worker_count": TOTAL_WORKERS,
            "chain_count_per_chart": CHAIN_COUNT_PER_CHART,
            "seeds": "domain-separated deterministic seeds recorded in progress",
            "source_sha256": {
                "launcher": sha256(SCRIPT),
                "plan": sha256(ROOT / plan_path),
            },
        },
        "nonclaims": [
            "CPU/XLA distributed validation only",
            "no posterior oracle or scientific-validity claim",
            "timing is descriptive",
        ],
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if sorted(os.sched_getaffinity(0)) != [SUPERVISOR_CPU]:
        raise ValidationError("supervisor must be pinned exclusively to CPU 32")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValidationError("supervisor requires CUDA_VISIBLE_DEVICES=-1")
    if not args.preflight_only:
        raise ValidationError(
            "historical custom tuning is disabled; use "
            "bayesfilter.inference.tune_fixed_transport_hmc_kernel"
        )
    plan_path = selected_plan(str(args.tuning_profile))
    campaign_cap = float(args.campaign_cap_seconds)
    if not math.isfinite(campaign_cap) or campaign_cap <= 0.0:
        raise ValidationError("campaign cap must be positive and finite")
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise ValidationError("output root must be repository-local")
    if output.exists() and any(output.iterdir()):
        raise ValidationError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    info = {label: checkpoint_info(label) for label in CHARTS}
    progress_path = output / "progress.json"
    progress = {
        "schema": "bayesfilter.ssl_lstm.q20_cpu_xla_distributed_hmc_progress.v2",
        "status": "PREFLIGHT_PENDING",
        "plan": plan_path.as_posix(),
        "tuning_profile": str(args.tuning_profile),
        "tuning_arms": [list(arm) for arm in tuning_arms(str(args.tuning_profile))],
        "campaign_cap_seconds": campaign_cap,
        "progress_path": progress_path.relative_to(ROOT).as_posix(),
        "checkpoints": info,
        "tuning": {},
        "confirmations": {},
        "warmup": {},
        "retained": {},
        "vetoes": [],
        "repair_triggers": [],
    }
    write_json(output / "progress.json", progress)
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    started = time.perf_counter()
    try:
        processes, logs, ready = start_workers(
            output, tuning_profile=str(args.tuning_profile)
        )
        progress["ready"] = ready
        progress["aggregate_startup_rss_bytes"] = sum(
            int(row["rss_bytes"]) for row in ready
        )
        if progress["aggregate_startup_rss_bytes"] > HOST_RSS_CAP:
            raise ValidationError("aggregate startup RSS exceeds campaign cap")
        preflight = run_preflight(processes, progress)
        progress["preflight"] = preflight
        progress["status"] = preflight["status"]
        write_json(progress_path, progress, replace=True)
        if preflight["vetoes"]:
            progress["vetoes"].extend(preflight["vetoes"])
            progress["status"] = "PREFLIGHT_FAILED"
            summary = terminal_summary(
                progress,
                started=started,
                campaign_cap=campaign_cap,
                plan_path=plan_path,
            )
            write_json(output / "summary.json", summary)
            return summary
        if args.preflight_only:
            progress["status"] = "PREFLIGHT_ONLY_COMPLETED"
            progress["preflight_nonclaims"] = [
                "mechanics preflight only",
                "not tuning, validation, convergence, or posterior evidence",
            ]
            summary = terminal_summary(
                progress,
                started=started,
                campaign_cap=campaign_cap,
                plan_path=plan_path,
            )
            write_json(output / "summary.json", summary)
            return summary

        progress["status"] = "TUNING"
        write_json(progress_path, progress, replace=True)
        kernels = tune_charts(
            processes, progress, profile=str(args.tuning_profile)
        )
        eligible = [label for label in CHARTS if kernels[label] is not None]
        for label in CHARTS:
            if kernels[label] is None:
                progress["repair_triggers"].append(
                    f"tuning:{label}:no_confirmed_kernel"
                )
        if not eligible:
            progress["status"] = "TUNING_REPAIR_REQUIRED"
            summary = terminal_summary(
                progress,
                started=started,
                campaign_cap=campaign_cap,
                plan_path=plan_path,
            )
            write_json(output / "summary.json", summary)
            return summary
        progress["selected_kernels"] = kernels
        write_json(progress_path, progress, replace=True)
        if str(args.tuning_profile) == "short-trajectory-repair-v1":
            progress["status"] = (
                "TUNING_REPAIR_CONFIRMED"
                if len(eligible) == len(CHARTS)
                else "TUNING_REPAIR_PARTIAL_CONFIRMED"
            )
            summary = terminal_summary(
                progress,
                started=started,
                campaign_cap=campaign_cap,
                plan_path=plan_path,
            )
            write_json(output / "summary.json", summary)
            return summary

        last_rss = {
            label: [
                int(row["rss_bytes"])
                for row in ready[
                    (0 if label == "chart-a" else CHAIN_COUNT_PER_CHART) :
                    (CHAIN_COUNT_PER_CHART if label == "chart-a" else TOTAL_WORKERS)
                ]
            ]
            for label in CHARTS
        }
        warmup_passed: list[str] = []
        for phase, chunk_size, minimum, maximum, rhat_limit in (
            ("warmup", WARM_CHUNK, WARM_MIN, WARM_MAX, RHAT_WARM),
            ("retained", RETAINED_CHUNK, RETAINED_MIN, RETAINED_MAX, RHAT_RETAINED),
        ):
            active = list(eligible if phase == "warmup" else warmup_passed)
            buffers = {
                label: {
                    "latent": [[] for _ in range(CHAIN_COUNT_PER_CHART)],
                    "mapped": [[] for _ in range(CHAIN_COUNT_PER_CHART)],
                }
                for label in active
            }
            for label in active:
                progress[phase][label] = {
                    "count": 0,
                    "chunks": [],
                    "passed": False,
                    "status": "RUNNING",
                }
            progress["status"] = f"{phase.upper()}_RUNNING"
            write_json(progress_path, progress, replace=True)
            chunk_index = 0
            while active:
                if time.perf_counter() - started >= campaign_cap:
                    progress["vetoes"].append("campaign_wall_cap")
                    for label in active:
                        progress[phase][label]["status"] = "CAMPAIGN_CAP"
                    active.clear()
                    break
                start_ns = time.monotonic_ns() + 1_000_000_000
                for label in active:
                    kernel = kernels[label]
                    send_workers(
                        chart_processes(processes, label),
                        {
                            "command": "chunk",
                            "chunk_index": chunk_index,
                            "active_results": chunk_size,
                            "max_results": chunk_size,
                            "step_size": kernel["step_size"],
                            "num_leapfrog_steps": kernel["num_leapfrog_steps"],
                            "seed_base": 200000
                            + (0 if label == "chart-a" else 100000)
                            + (0 if phase == "warmup" else 500000)
                            + chunk_index * 1000,
                            "start_ns": start_ns,
                        },
                    )
                completed: dict[str, tuple[dict[str, Any], ...]] = {}
                for label in active:
                    completed[label] = collect_workers(
                        chart_processes(processes, label), timeout=COMMAND_TIMEOUT
                    )

                next_active = []
                for label in active:
                    rows = completed[label]
                    start_cpu = 0 if label == "chart-a" else CHAIN_COUNT_PER_CHART
                    vetoes = chunk_vetoes(
                        rows,
                        expected_cpus=WORKER_CPUS[
                            start_cpu : start_cpu + CHAIN_COUNT_PER_CHART
                        ],
                    )
                    last_rss[label] = [int(row["rss_bytes"]) for row in rows]
                    if sum(sum(values) for values in last_rss.values()) > HOST_RSS_CAP:
                        vetoes.append("aggregate_worker_rss_cap")
                    if vetoes:
                        progress["vetoes"].extend(
                            f"{phase}:{label}:{veto}" for veto in vetoes
                        )
                        progress[phase][label]["status"] = "HARD_VETO"
                        continue
                    archive = archive_chunk(
                        output,
                        phase=phase,
                        label=label,
                        chunk_index=chunk_index,
                        rows=rows,
                    )
                    for chain, row in enumerate(rows):
                        buffers[label]["latent"][chain].extend(row["samples"])
                        buffers[label]["mapped"][chain].extend(row["mapped_samples"])
                    count = len(buffers[label]["latent"][0])
                    progress[phase][label]["count"] = count
                    progress[phase][label]["chunks"].append(
                        {
                            "chunk_index": chunk_index,
                            "wall_seconds_max": max(float(row["wall_seconds"]) for row in rows),
                            "acceptance_probability_mean": sum(
                                float(row["acceptance_probability_mean"]) for row in rows
                            )
                            / len(rows),
                            "aggregate_worker_rss_bytes": sum(last_rss[label]),
                            "aggregate_all_worker_rss_bytes": sum(
                                sum(values) for values in last_rss.values()
                            ),
                            "archive": archive,
                        }
                    )
                    if count >= minimum:
                        import tensorflow as tf

                        latent = tf.transpose(
                            tf.convert_to_tensor(buffers[label]["latent"], tf.float64),
                            (1, 0, 2),
                        )
                        mapped = tf.transpose(
                            tf.convert_to_tensor(buffers[label]["mapped"], tf.float64),
                            (1, 0, 2),
                        )
                        if phase == "warmup":
                            latent = latent[-WARM_WINDOW:]
                            mapped = mapped[-WARM_WINDOW:]
                        report = diagnostics(
                            latent,
                            mapped,
                            rhat=rhat_limit,
                            warmup=phase == "warmup",
                        )
                        progress[phase][label]["diagnostics"] = report
                        progress[phase][label]["passed"] = bool(report["passed"])
                        if report["passed"]:
                            progress[phase][label]["status"] = "PASSED"
                            if phase == "warmup":
                                warmup_passed.append(label)
                            continue
                    if count >= maximum:
                        progress[phase][label]["status"] = "DIAGNOSTIC_CAP_FAILED"
                        progress["vetoes"].append(
                            f"{phase}:{label}:cap_or_diagnostic_failure"
                        )
                        continue
                    next_active.append(label)
                active = next_active
                chunk_index += 1
                write_json(progress_path, progress, replace=True)

        all_eligible_passed = bool(
            eligible
            and all(
                progress["retained"].get(label, {}).get("passed") is True
                for label in eligible
            )
        )
        if all_eligible_passed and not progress["vetoes"]:
            progress["status"] = (
                "VALIDATION_PASSED" if len(eligible) == len(CHARTS) else "VALIDATION_PARTIAL"
            )
        else:
            progress["status"] = "VALIDATION_VETOED"
        summary = terminal_summary(
            progress,
            started=started,
            campaign_cap=campaign_cap,
            plan_path=plan_path,
        )
        write_json(output / "summary.json", summary)
        return summary
    finally:
        terminate_workers(processes, logs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chart", choices=CHARTS, default="chart-a")
    parser.add_argument("--chain-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/r1"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--tuning-profile",
        choices=("original-v1", "short-trajectory-repair-v1"),
        default="original-v1",
    )
    parser.add_argument(
        "--campaign-cap-seconds", type=float, default=CAMPAIGN_CAP
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker:
        return _worker_main(args)
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
