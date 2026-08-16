#!/usr/bin/env python3
"""Measure 32 one-chain CPU/XLA q=20 NeuTra-HMC processes."""

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


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-plan-2026-08-01.md"
)
CHECKPOINT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
    "seed-a/seed-a/checkpoint-1500.json"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_32x1_chain_hmc_canary.v1"
WORKER_SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_1chain_hmc_worker.v1"
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
WORKER_CPUS = tuple(range(32))
SUPERVISOR_CPU = 32
WARM_ROUNDS = 2
NUM_RESULTS = 2
NUM_BURNIN = 1
NUM_LEAPFROG = 1
STEP_SIZE = 0.01
TRANSITIONS_PER_CALL = NUM_RESULTS + NUM_BURNIN
VALIDATION_MIN_TRANSITIONS = 3000
VALIDATION_LEAPFROG_HYPOTHESIS = 4
WORKER_START_TIMEOUT_SECONDS = 600.0
ROUND_TIMEOUT_SECONDS = 300.0
CANARY_CAP_SECONDS = 1200.0
HOST_WORKER_RSS_CAP_BYTES = 64 * 1024**3
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)


class CanaryError(RuntimeError):
    pass


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanaryError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CanaryError(f"output already exists: {path}")
    path.write_bytes(_canonical(payload))


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _wait_until(start_ns: int) -> None:
    while True:
        remaining = int(start_ns) - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)


def _worker_main(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise CanaryError("CPU worker requires CUDA_VISIBLE_DEVICES=-1")

    import tensorflow as tf

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise CanaryError("CPU worker found a visible TensorFlow GPU")

    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.hmc import (
        FullChainHMCConfig,
        build_reusable_full_chain_tfp_hmc_runner,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.inference.neutra_training import (
        NeuTraReverseKLTrainer,
        ssl_lstm_tuned_capacity_neutra_config,
    )
    from bayesfilter.inference.neutra_training_control import (
        validate_joint_training_checkpoint,
    )
    from bayesfilter.inference.posterior_adapter import ValueScoreCapability
    from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
        batch_native_complexity_posterior_target,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
        FREE_NAMES,
        PRIOR_CENTER,
    )

    class OneChainTargetBridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True

        def __init__(self, target: Any) -> None:
            self.target = target
            self.target_scope = f"{target.target_scope}:cpu_xla_1chain_hmc_canary"

        def adapter_signature(self) -> str:
            return self.target.adapter_signature()

        def target_signature(self) -> str:
            return self.target.target_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_cpu_xla_1chain_hmc_canary_bridge",
                evidence_path=PLAN.as_posix(),
                target_scope=self.target_scope,
                nonclaims=(
                    "CPU-XLA one-chain HMC throughput canary only",
                    "no convergence or posterior claim",
                ),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            tensor = tf.convert_to_tensor(values, tf.float64)
            if tensor.shape.rank != 2 or tensor.shape[0] != 1:
                raise ValueError("one-chain target requires static shape [1,4]")
            return self.target.batch_value_and_score(tensor)

        def log_prob_and_grad_status(
            self, values: Any
        ) -> tuple[Any, Any, Mapping[str, Any]]:
            return self.target.neutra_batch_log_prob_and_grad_status(values)

    started = time.perf_counter()
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    checkpoint = _strict_json(checkpoint_path)
    validate_joint_training_checkpoint(checkpoint)
    best_state = checkpoint.get("best_trainer_state")
    if not isinstance(best_state, Mapping) or int(best_state.get("step", -1)) != 1500:
        raise CanaryError("checkpoint does not bind Seed-A best step 1500")

    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=0.0004,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260719, 12101),
        jit_compile=True,
    )
    trainer = NeuTraReverseKLTrainer(target, trainer_config)
    trainer.restore_state(best_state)
    frozen = trainer.frozen_transport_payload(
        transport_id="seed-a-best-1500-cpu-xla-1chain-hmc-canary",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    base = OneChainTargetBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope=f"{base.target_scope}:seed-a-best-1500",
        runtime_backend="ssl_lstm_q20_cpu_xla_1chain_fixed_transport_hmc_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "short one-chain CPU-XLA HMC mechanics canary only",
            "CPU diagnostic checkpoint remains ineligible for posterior claims",
        ),
    )
    start = INITIAL_Z[int(args.worker_index) % len(INITIAL_Z)]
    initial_state = tf.constant((start,), tf.float64)
    seed = (20260801, 61000 + int(args.worker_index))
    hmc_config = FullChainHMCConfig(
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN,
        step_size=STEP_SIZE,
        num_leapfrog_steps=NUM_LEAPFROG,
        seed=seed,
        use_xla=True,
        trace_policy="standard",
        target_scope=adapter.target_scope,
    )
    runner = build_reusable_full_chain_tfp_hmc_runner(
        adapter, initial_state, hmc_config
    )
    first_tick = time.perf_counter()
    first = runner.run(seed=seed, step_size=STEP_SIZE)
    first_seconds = time.perf_counter() - first_tick
    if not bool(tf.reduce_all(tf.math.is_finite(first.samples)).numpy()):
        raise CanaryError("cold HMC call produced nonfinite samples")
    ready = {
        "schema": WORKER_SCHEMA,
        "event": "ready",
        "pid": os.getpid(),
        "worker_index": int(args.worker_index),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [item.name for item in tf.config.list_physical_devices("GPU")],
        "tensorflow": tf.__version__,
        "jit_compile": True,
        "dtype": "float64",
        "sample_shape": list(first.samples.shape),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "best_trainer_state_hash": best_state["state_hash"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "artifact_signature": loaded.artifact_signature,
        "transport_hash": loaded.manifest.transport_hash,
        "hmc": hmc_config.signature_payload(),
        "first_call_seconds": first_seconds,
        "first_sample_chain_seconds": float(first.metadata["sample_chain_call_s"]),
        "first_all_finite": True,
        "startup_seconds": time.perf_counter() - started,
        "rss_bytes": _rss_bytes(),
    }
    print(json.dumps(ready, allow_nan=False), flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "stop":
            print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
            return 0
        if request.get("command") != "run":
            raise CanaryError("unknown worker command")
        _wait_until(int(request["start_ns"]))
        folded = (
            20260801,
            62000 + int(args.worker_index) * 100 + int(request["round"]),
        )
        tick = time.perf_counter()
        result = runner.run(seed=folded, step_size=STEP_SIZE)
        seconds = time.perf_counter() - tick
        finite = bool(tf.reduce_all(tf.math.is_finite(result.samples)).numpy())
        finite = finite and all(
            bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())
            for value in result.trace.values()
            if getattr(tf.convert_to_tensor(value).dtype, "is_floating", False)
        )
        print(
            json.dumps(
                {
                    "schema": WORKER_SCHEMA,
                    "event": "done",
                    "pid": os.getpid(),
                    "worker_index": int(args.worker_index),
                    "round": int(request["round"]),
                    "seed": list(folded),
                    "wall_seconds": seconds,
                    "sample_chain_seconds": float(result.metadata["sample_chain_call_s"]),
                    "all_finite": finite,
                    "sample_shape": list(result.samples.shape),
                    "rss_bytes": _rss_bytes(),
                },
                allow_nan=False,
            ),
            flush=True,
        )
    return 0


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "false",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TF_CPP_MIN_LOG_LEVEL": "1",
        }
    )
    return environment


def _worker_command(index: int, cpu: int) -> list[str]:
    return [
        "taskset",
        "-c",
        str(cpu),
        str(PYTHON),
        str(SCRIPT),
        "--worker",
        "--worker-index",
        str(index),
        "--checkpoint",
        CHECKPOINT.as_posix(),
    ]


def _read_line(process: subprocess.Popen[str], timeout: float) -> dict[str, Any]:
    if process.stdout is None:
        raise CanaryError("worker stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], float(timeout))
    if not ready:
        raise CanaryError(f"worker {process.pid} output timeout")
    line = process.stdout.readline()
    if not line:
        raise CanaryError(f"worker {process.pid} closed stdout")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise CanaryError("worker response is not a JSON object")
    return value


def _run_arm(output: Path, worker_count: int) -> Mapping[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        arm_started = time.perf_counter()
        for index, cpu in enumerate(WORKER_CPUS[: int(worker_count)]):
            log = (output / f"worker-{index:02d}.stderr.log").open(
                "w", encoding="utf-8"
            )
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    _worker_command(index, cpu),
                    cwd=ROOT,
                    env=_worker_environment(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    bufsize=1,
                )
            )
        ready = tuple(
            _read_line(process, WORKER_START_TIMEOUT_SECONDS) for process in processes
        )
        if any(row.get("event") != "ready" for row in ready):
            raise CanaryError("worker did not emit ready")
        if any(row.get("affinity") != [WORKER_CPUS[index]] for index, row in enumerate(ready)):
            raise CanaryError("worker affinity mismatch")
        if any(row.get("sample_shape") != [2, 1, 4] for row in ready):
            raise CanaryError("worker is not using one-chain state")
        if len({row["target_signature"] for row in ready}) != 1:
            raise CanaryError("target signature mismatch across workers")
        if len({row["transport_hash"] for row in ready}) != 1:
            raise CanaryError("transport hash mismatch across workers")
        cold_rss = sum(int(row["rss_bytes"]) for row in ready)
        if cold_rss > HOST_WORKER_RSS_CAP_BYTES:
            raise CanaryError("aggregate worker RSS exceeded 64 GiB")

        rounds = []
        for round_index in range(WARM_ROUNDS):
            start_ns = time.monotonic_ns() + 1_000_000_000
            request = json.dumps(
                {"command": "run", "round": round_index, "start_ns": start_ns}
            )
            for process in processes:
                if process.stdin is None:
                    raise CanaryError("worker stdin is unavailable")
                process.stdin.write(request + "\n")
                process.stdin.flush()
            parent_tick = time.perf_counter()
            rows = tuple(_read_line(process, ROUND_TIMEOUT_SECONDS) for process in processes)
            parent_wait = time.perf_counter() - parent_tick
            if any(row.get("event") != "done" or not row.get("all_finite") for row in rows):
                raise CanaryError("warm HMC call failed")
            rss_sum = sum(int(row["rss_bytes"]) for row in rows)
            if rss_sum > HOST_WORKER_RSS_CAP_BYTES:
                raise CanaryError("aggregate worker RSS exceeded 64 GiB")
            window = max(float(row["wall_seconds"]) for row in rows)
            rounds.append(
                {
                    "round": round_index,
                    "synchronized_parent_wait_seconds": parent_wait,
                    "concurrent_kernel_window_seconds": window,
                    "aggregate_process_calls_per_second": len(rows) / window,
                    "aggregate_chain_transitions_per_second": (
                        len(rows) * TRANSITIONS_PER_CALL / window
                    ),
                    "aggregate_worker_rss_bytes": rss_sum,
                    "workers": rows,
                }
            )
        for process in processes:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                process.stdin.flush()
        stopped = tuple(_read_line(process, 30.0) for process in processes)
        exit_codes = tuple(process.wait(timeout=30.0) for process in processes)
        if any(code != 0 for code in exit_codes):
            raise CanaryError(f"worker exit failure: {exit_codes}")
        windows = [float(row["concurrent_kernel_window_seconds"]) for row in rounds]
        rates = [float(row["aggregate_process_calls_per_second"]) for row in rounds]
        result = {
            "schema": SCHEMA,
            "status": "PASSED",
            "worker_count": int(worker_count),
            "worker_cpus": list(WORKER_CPUS[: int(worker_count)]),
            "cold_first_call_seconds_max": max(float(row["first_call_seconds"]) for row in ready),
            "cold_startup_seconds_max": max(float(row["startup_seconds"]) for row in ready),
            "warm_window_seconds_mean": sum(windows) / len(windows),
            "warm_window_seconds_max": max(windows),
            "aggregate_process_calls_per_second_mean": sum(rates) / len(rates),
            "aggregate_chain_transitions_per_second_mean": (
                sum(float(row["aggregate_chain_transitions_per_second"]) for row in rounds)
                / len(rounds)
            ),
            "aggregate_worker_rss_bytes_max": max(
                int(row["aggregate_worker_rss_bytes"]) for row in rounds
            ),
            "ready": ready,
            "rounds": rounds,
            "stopped": stopped,
            "exit_codes": list(exit_codes),
            "arm_wall_seconds": time.perf_counter() - arm_started,
        }
        _write_json(output / "result.json", result)
        return result
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        for log in logs:
            log.close()


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if sorted(os.sched_getaffinity(0)) != [SUPERVISOR_CPU]:
        raise CanaryError("supervisor must be pinned exclusively to CPU 32")
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise CanaryError("output root must be repository-local")
    if output.exists() and any(output.iterdir()):
        raise CanaryError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if not set((*WORKER_CPUS, SUPERVISOR_CPU)).issubset(os.cpu_count() and range(os.cpu_count()) or ()):
        raise CanaryError("required CPU IDs are unavailable")
    checkpoint_path = ROOT / CHECKPOINT
    checkpoint = _strict_json(checkpoint_path)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    baseline = _run_arm(output / "p1x1", 1)
    p32 = _run_arm(output / "p32x1", 32)
    if time.perf_counter() - started > CANARY_CAP_SECONDS:
        raise CanaryError("canary cap exceeded")

    base_rate = float(baseline["aggregate_process_calls_per_second_mean"])
    p32_rate = float(p32["aggregate_process_calls_per_second_mean"])
    p32_warm = float(p32["warm_window_seconds_mean"])
    seconds_per_transition_leapfrog = p32_warm / TRANSITIONS_PER_CALL
    minimum_validation_seconds = (
        seconds_per_transition_leapfrog
        * VALIDATION_MIN_TRANSITIONS
        * VALIDATION_LEAPFROG_HYPOTHESIS
    )
    summary = {
        "schema": SCHEMA,
        "status": "PASSED",
        "question_answer": "32 one-chain CPU/XLA HMC workers ran concurrently under a separate supervisor core",
        "plan": PLAN.as_posix(),
        "topology": {
            "supervisor_cpu": SUPERVISOR_CPU,
            "worker_cpus": list(WORKER_CPUS),
            "worker_count": 32,
            "chain_count_per_worker": 1,
        },
        "hmc_work": {
            "num_results": NUM_RESULTS,
            "num_burnin_steps": NUM_BURNIN,
            "num_leapfrog_steps": NUM_LEAPFROG,
            "transitions_per_timed_call": TRANSITIONS_PER_CALL,
            "warm_rounds": WARM_ROUNDS,
        },
        "checkpoint": {
            "path": CHECKPOINT.as_posix(),
            "sha256": _sha256(checkpoint_path),
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "best_step": checkpoint["best_trainer_state"]["step"],
            "best_trainer_state_hash": checkpoint["best_trainer_state"]["state_hash"],
        },
        "baseline_p1x1": baseline,
        "candidate_p32x1": p32,
        "comparison": {
            "p1_aggregate_process_calls_per_second": base_rate,
            "p32_aggregate_process_calls_per_second": p32_rate,
            "p32_speedup_vs_p1": p32_rate / base_rate,
            "p32_parallel_efficiency_vs_p1": p32_rate / (32.0 * base_rate),
            "p32_warm_window_seconds_mean": p32_warm,
            "p32_aggregate_chain_transitions_per_second": p32[
                "aggregate_chain_transitions_per_second_mean"
            ],
        },
        "minimum_validation_time_estimate": {
            "classification": "derived_linear_unproven_estimate",
            "formula": "p32_warm_seconds/3 * 3000_transitions * 4_leapfrog_steps",
            "seconds_per_transition_leapfrog": seconds_per_transition_leapfrog,
            "warmup_transitions_per_chain": 2000,
            "retained_transitions_per_chain": 1000,
            "total_transitions_per_chain": VALIDATION_MIN_TRANSITIONS,
            "leapfrog_steps": VALIDATION_LEAPFROG_HYPOTHESIS,
            "estimated_wall_seconds": minimum_validation_seconds,
            "estimated_wall_hours": minimum_validation_seconds / 3600.0,
            "excludes": [
                "fresh kernel tuning",
                "XLA cold compilation",
                "chunk diagnostics and artifact I/O",
                "additional warmup or retained draws required by R-hat/ESS",
                "nonlinear timing effects from longer sample_chain graphs",
            ],
        },
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
            "started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": CANARY_CAP_SECONDS,
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import in every child",
            "supervisor_affinity": sorted(os.sched_getaffinity(0)),
            "jit_compile": True,
            "sample_wise_loop_or_scalar_fallback": False,
            "source_sha256": {
                "launcher": _sha256(SCRIPT),
                "plan": _sha256(ROOT / PLAN),
                "checkpoint": _sha256(checkpoint_path),
            },
            "output_root": output.relative_to(ROOT).as_posix(),
        },
        "inference_status": {
            "hard_veto_screen": "passed for p1 and p32 canary mechanics",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "cold/warm timing, throughput, scaling, and RSS",
            "default_readiness": "not assessed",
            "next_evidence_needed": "fresh payload-bound HMC tuning and measured longer sequential chunks",
        },
        "nonclaims": [
            "short CPU-XLA one-chain mechanics and throughput canary only",
            "no HMC tuning, convergence, or posterior correctness claim",
            "no CPU default or GPU comparison",
            "minimum validation time is a derived linear estimate, not a measured long-chain runtime",
            "CPU diagnostic checkpoint remains ineligible for posterior promotion",
        ],
    }
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "docs/plans/artifacts/"
            "ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-2026-08-01/r1"
        ),
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
