#!/usr/bin/env python3
"""Measure independent CPU/XLA q=20 NeuTra-HMC process throughput."""

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
    "bayesfilter-ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-plan-2026-08-01.md"
)
CHECKPOINT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
    "seed-a/seed-a/checkpoint-1500.json"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_multiprocess_hmc_canary.v1"
WORKER_SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_hmc_worker.v1"
PYTHON = Path("/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
TOPOLOGIES = (1, 2, 4)
CPU_IDS = (0, 1, 2, 3)
WARM_ROUNDS = 3
NUM_RESULTS = 2
NUM_BURNIN = 1
NUM_LEAPFROG = 1
STEP_SIZE = 0.01
CALL_TRANSITIONS_PER_CHAIN = NUM_RESULTS + NUM_BURNIN
CHAIN_COUNT = 4
WORKER_START_TIMEOUT_SECONDS = 600.0
ROUND_TIMEOUT_SECONDS = 300.0
CANARY_CAP_SECONDS = 1800.0


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CanaryError(f"output already exists: {path}")
    path.write_bytes(_canonical(payload))


def _strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CanaryError(f"expected a JSON object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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

    class HMCBatchTargetBridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True

        def __init__(self, target: Any) -> None:
            self.target = target
            self.target_scope = f"{target.target_scope}:cpu_xla_hmc_canary"

        def adapter_signature(self) -> str:
            return self.target.adapter_signature()

        def target_signature(self) -> str:
            return self.target.target_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_cpu_xla_hmc_canary_bridge",
                evidence_path=PLAN.as_posix(),
                target_scope=self.target_scope,
                nonclaims=(
                    "CPU-XLA HMC mechanics and throughput canary only",
                    "no convergence or posterior claim",
                ),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            tensor = tf.convert_to_tensor(values, tf.float64)
            if tensor.shape.rank != 2:
                raise ValueError("canary HMC target requires rank-two chain state")
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
    if not isinstance(best_state, Mapping):
        raise CanaryError("checkpoint has no best trainer state")
    if int(best_state.get("step", -1)) != 1500:
        raise CanaryError("canary checkpoint best state is not step 1500")

    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    config = ssl_lstm_tuned_capacity_neutra_config(
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
    trainer = NeuTraReverseKLTrainer(target, config)
    trainer.restore_state(best_state)
    frozen = trainer.frozen_transport_payload(
        transport_id="seed-a-best-1500-cpu-xla-hmc-canary",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    base = HMCBatchTargetBridge(target)
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=loaded.transport,
        target_scope=f"{base.target_scope}:seed-a-best-1500",
        runtime_backend="ssl_lstm_q20_cpu_xla_fixed_transport_hmc_canary",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "short CPU-XLA HMC mechanics canary only",
            "CPU diagnostic checkpoint remains ineligible for posterior claims",
        ),
    )
    initial_state = tf.constant(
        (
            (0.0, 0.0, 0.0, 0.0),
            (0.5, -0.5, 0.5, -0.5),
            (-0.5, 0.5, -0.5, 0.5),
            (0.5, 0.5, -0.5, -0.5),
        ),
        tf.float64,
    )
    seed = (20260801, 51000 + int(args.worker_index))
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

    first_started = time.perf_counter()
    first = runner.run(seed=seed, step_size=STEP_SIZE)
    first_seconds = time.perf_counter() - first_started
    if not bool(tf.reduce_all(tf.math.is_finite(first.samples)).numpy()):
        raise CanaryError("first HMC call produced nonfinite samples")
    ready = {
        "schema": WORKER_SCHEMA,
        "event": "ready",
        "pid": os.getpid(),
        "worker_index": int(args.worker_index),
        "affinity": sorted(os.sched_getaffinity(0)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [value.name for value in tf.config.list_physical_devices("GPU")],
        "tensorflow": tf.__version__,
        "jit_compile": True,
        "dtype": "float64",
        "checkpoint_path": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "best_trainer_state_hash": best_state["state_hash"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "frozen_artifact_signature": loaded.artifact_signature,
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
        command = request.get("command")
        if command == "stop":
            print(json.dumps({"event": "stopped", "pid": os.getpid()}), flush=True)
            return 0
        if command != "run":
            raise CanaryError(f"unknown worker command: {command}")
        _wait_until(int(request["start_ns"]))
        folded = (
            20260801,
            52000
            + int(args.worker_index) * 100
            + int(request["round"]),
        )
        tick = time.perf_counter()
        result = runner.run(seed=folded, step_size=STEP_SIZE)
        seconds = time.perf_counter() - tick
        samples_finite = bool(
            tf.reduce_all(tf.math.is_finite(result.samples)).numpy()
        )
        trace_finite = all(
            bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())
            for value in result.trace.values()
            if getattr(tf.convert_to_tensor(value).dtype, "is_floating", False)
        )
        response = {
            "schema": WORKER_SCHEMA,
            "event": "done",
            "pid": os.getpid(),
            "worker_index": int(args.worker_index),
            "round": int(request["round"]),
            "seed": list(folded),
            "wall_seconds": seconds,
            "sample_chain_seconds": float(result.metadata["sample_chain_call_s"]),
            "all_finite": samples_finite and trace_finite,
            "sample_shape": list(result.samples.shape),
            "rss_bytes": _rss_bytes(),
        }
        print(json.dumps(response, allow_nan=False), flush=True)
    return 0


def _read_line(process: subprocess.Popen[str], timeout_seconds: float) -> dict[str, Any]:
    if process.stdout is None:
        raise CanaryError("worker stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], float(timeout_seconds))
    if not ready:
        raise CanaryError(f"worker {process.pid} timed out waiting for output")
    line = process.stdout.readline()
    if not line:
        raise CanaryError(f"worker {process.pid} closed stdout unexpectedly")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise CanaryError("worker response is not a JSON object")
    return value


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


def _run_topology(output: Path, workers: int) -> Mapping[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        topology_started = time.perf_counter()
        for index in range(int(workers)):
            log_path = output / f"worker-{index}.stderr.log"
            log = log_path.open("w", encoding="utf-8")
            logs.append(log)
            process = subprocess.Popen(
                _worker_command(index, CPU_IDS[index]),
                cwd=ROOT,
                env=_worker_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
            processes.append(process)
        ready = tuple(
            _read_line(process, WORKER_START_TIMEOUT_SECONDS) for process in processes
        )
        if any(row.get("event") != "ready" for row in ready):
            raise CanaryError("worker did not emit ready event")
        if any(row.get("affinity") != [CPU_IDS[index]] for index, row in enumerate(ready)):
            raise CanaryError("worker affinity mismatch")
        if len({row["target_signature"] for row in ready}) != 1:
            raise CanaryError("worker target signatures differ")
        if len({row["transport_hash"] for row in ready}) != 1:
            raise CanaryError("worker transport hashes differ")

        rounds: list[Mapping[str, Any]] = []
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
            round_started = time.perf_counter()
            rows = tuple(
                _read_line(process, ROUND_TIMEOUT_SECONDS) for process in processes
            )
            round_wall = time.perf_counter() - round_started
            if any(row.get("event") != "done" or not row.get("all_finite") for row in rows):
                raise CanaryError("worker warm HMC call failed the finite screen")
            kernel_window = max(float(row["wall_seconds"]) for row in rows)
            process_calls = len(rows)
            rounds.append(
                {
                    "round": round_index,
                    "synchronized_parent_wait_seconds": round_wall,
                    "concurrent_kernel_window_seconds": kernel_window,
                    "workers": rows,
                    "aggregate_process_calls_per_second": process_calls / kernel_window,
                    "aggregate_chain_transitions_per_second": (
                        process_calls
                        * CHAIN_COUNT
                        * CALL_TRANSITIONS_PER_CHAIN
                        / kernel_window
                    ),
                }
            )
        for process in processes:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"command": "stop"}) + "\n")
                process.stdin.flush()
        stopped = tuple(_read_line(process, 30.0) for process in processes)
        exit_codes = tuple(process.wait(timeout=30.0) for process in processes)
        if any(code != 0 for code in exit_codes):
            raise CanaryError(f"worker exit codes are not zero: {exit_codes}")
        warm_windows = [float(row["concurrent_kernel_window_seconds"]) for row in rounds]
        aggregate_rates = [float(row["aggregate_process_calls_per_second"]) for row in rounds]
        summary = {
            "schema": SCHEMA,
            "status": "PASSED",
            "workers": int(workers),
            "cpu_ids": list(CPU_IDS[:workers]),
            "topology_wall_seconds": time.perf_counter() - topology_started,
            "ready": ready,
            "rounds": rounds,
            "stopped": stopped,
            "exit_codes": list(exit_codes),
            "cold_first_call_seconds_max": max(float(row["first_call_seconds"]) for row in ready),
            "startup_seconds_max": max(float(row["startup_seconds"]) for row in ready),
            "warm_concurrent_window_seconds_max": max(warm_windows),
            "warm_concurrent_window_seconds_mean": sum(warm_windows) / len(warm_windows),
            "aggregate_process_calls_per_second_min": min(aggregate_rates),
            "aggregate_process_calls_per_second_mean": sum(aggregate_rates) / len(aggregate_rates),
            "aggregate_chain_transitions_per_second_mean": (
                sum(float(row["aggregate_chain_transitions_per_second"]) for row in rounds)
                / len(rounds)
            ),
            "max_worker_rss_bytes": max(
                int(worker["rss_bytes"])
                for round_row in rounds
                for worker in round_row["workers"]
            ),
        }
        _write_json(output / "result.json", summary)
        return summary
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
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise CanaryError("output root must be inside the repository")
    if output.exists() and any(output.iterdir()):
        raise CanaryError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / CHECKPOINT
    if not checkpoint_path.is_file():
        raise CanaryError("required Seed A checkpoint is missing")
    checkpoint = _strict_json(checkpoint_path)
    available = set(os.sched_getaffinity(0))
    if not set(CPU_IDS).issubset(available):
        raise CanaryError("required canary CPU IDs are unavailable")

    started = time.perf_counter()
    started_utc = _utc_now()
    topology_rows = []
    for workers in TOPOLOGIES:
        if time.perf_counter() - started >= CANARY_CAP_SECONDS:
            raise CanaryError("canary wall cap exhausted")
        topology_rows.append(_run_topology(output / f"p{workers}", workers))
    baseline = float(topology_rows[0]["aggregate_process_calls_per_second_mean"])
    comparisons = []
    for row in topology_rows:
        workers = int(row["workers"])
        throughput = float(row["aggregate_process_calls_per_second_mean"])
        comparisons.append(
            {
                "workers": workers,
                "aggregate_process_calls_per_second_mean": throughput,
                "speedup_vs_p1": throughput / baseline,
                "parallel_efficiency_vs_p1": throughput / (baseline * workers),
                "cold_first_call_seconds_max": row["cold_first_call_seconds_max"],
                "warm_concurrent_window_seconds_mean": row["warm_concurrent_window_seconds_mean"],
                "aggregate_chain_transitions_per_second_mean": row["aggregate_chain_transitions_per_second_mean"],
                "max_worker_rss_bytes": row["max_worker_rss_bytes"],
            }
        )
    finished_utc = _utc_now()
    summary = {
        "schema": SCHEMA,
        "status": "PASSED",
        "question_answer": "independent CPU processes executed the same XLA NeuTra-HMC mechanics concurrently",
        "plan": PLAN.as_posix(),
        "checkpoint": {
            "path": CHECKPOINT.as_posix(),
            "sha256": _sha256(checkpoint_path),
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "best_step": checkpoint["best_trainer_state"]["step"],
            "best_trainer_state_hash": checkpoint["best_trainer_state"]["state_hash"],
        },
        "hmc_work": {
            "chain_count": CHAIN_COUNT,
            "num_results": NUM_RESULTS,
            "num_burnin_steps": NUM_BURNIN,
            "num_leapfrog_steps": NUM_LEAPFROG,
            "step_size": STEP_SIZE,
            "transitions_per_chain_per_call": CALL_TRANSITIONS_PER_CHAIN,
            "chain_transitions_per_process_call": CHAIN_COUNT * CALL_TRANSITIONS_PER_CHAIN,
            "warm_rounds": WARM_ROUNDS,
        },
        "comparisons": comparisons,
        "topologies": topology_rows,
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
            "finished_at_utc": finished_utc,
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": CANARY_CAP_SECONDS,
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import in every child",
            "cpu_model": subprocess.check_output(
                ("bash", "-lc", "lscpu | sed -n 's/^Model name:[[:space:]]*//p'"),
                text=True,
            ).strip(),
            "cpu_ids": list(CPU_IDS),
            "jit_compile": True,
            "sample_wise_loop_or_scalar_fallback": False,
            "output_root": output.relative_to(ROOT).as_posix(),
            "plan": PLAN.as_posix(),
            "source_sha256": {
                "launcher": _sha256(SCRIPT),
                "plan": _sha256(ROOT / PLAN),
                "checkpoint": _sha256(checkpoint_path),
            },
        },
        "inference_status": {
            "hard_veto_screen": "passed for canary mechanics",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "cold and warm timing by process count",
            "default_readiness": "not assessed",
            "next_evidence_needed": "fresh payload-bound HMC tuning before retained sampling",
        },
        "nonclaims": [
            "short CPU-XLA mechanics and throughput canary only",
            "no HMC tuning, convergence, or posterior correctness claim",
            "no CPU default or GPU comparison",
            "no statistically supported topology ranking",
            "CPU diagnostic training artifact remains ineligible for posterior promotion",
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
            "ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-2026-08-01/r1"
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
