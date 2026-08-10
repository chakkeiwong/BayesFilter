#!/usr/bin/env python3
"""Run the frozen Chart A L=10 candidate through sequential CPU/XLA HMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import select
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# This is an explicit CPU validation exception. Hide both GPUs before any
# TensorFlow-owning module can be imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-chart-a-l10-sequential-hmc-plan-2026-08-04.md"
)
CHECKPOINT = ROOT / (
    "docs/plans/artifacts/ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/"
    "r1/seed-a/seed-a/checkpoint-1500.json"
)
TUNING_ARTIFACT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/"
    "r1/merged-tuning-result.json"
)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1"
)
DEFAULT_PREFLIGHT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/preflight-r1"
)

SCHEMA = "bayesfilter.ssl_lstm.q20_chart_a_l10_sequential_hmc.v1"
POLICY_ID = "bayesfilter_neutra_sequential_hmc_v1"
EXPECTED_TUNING_SHA256 = (
    "c3018064fcbbe040b3510165138bc7db7de1b378dd0eb4c1a1b8155af796fb19"
)
EXPECTED_KERNEL_HASH = (
    "34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d"
)
EXPECTED_BINDINGS = {
    "checkpoint_sha256": (
        "c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff"
    ),
    "target_signature": (
        "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
    ),
    "base_adapter_signature": (
        "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
    ),
    "transformed_adapter_signature": (
        "9772c5988104a9548e34eb138ffe4e950fb8354580f2395fd96718a35e60103e"
    ),
    "fixed_transport_manifest_hash": (
        "dcb1ec65e7d91a382518a0eef382e3cd8efec78341445f22d4d6ac899ea685eb"
    ),
    "transport_hash": (
        "caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f"
    ),
}
TARGET_SCOPE = (
    "ssl_lstm_neutra_state_complexity_batch_native:q20:"
    "fixed_hmc_api:chart-a:claim_tuning_grid6"
)

CHAIN_COUNT = 4
DIMENSION = 4
CHAIN_CPUS = (
    tuple(range(0, 8)),
    tuple(range(8, 16)),
    tuple(range(16, 24)),
    tuple(range(24, 32)),
)
SUPERVISOR_CPU = 32
THREADS_PER_CHAIN = 8
CHUNK_RESULTS = 500
DEFAULT_CAP_SECONDS = 86_400.0
ARCHIVE_RESERVE_SECONDS = 600.0
FORECAST_MARGIN = 1.25
WORKER_START_TIMEOUT_SECONDS = 900.0
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
ROOT_SEED = (20260804, 41001)


class CampaignError(RuntimeError):
    """Raised when a campaign identity or execution invariant fails."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CampaignError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(payload))
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"expected JSON object: {path}")
    return payload


def _configure_tensorflow(threads: int) -> Any:
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(int(threads))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise CampaignError("CPU/XLA worker found a visible GPU")
    return tf


def _build_chart(*, threads: int) -> tuple[Any, Any, Mapping[str, Any]]:
    """Reconstruct the exact frozen Chart A target and transport."""

    tf = _configure_tensorflow(threads)
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

    checkpoint = _read_json(CHECKPOINT)
    validate_joint_training_checkpoint(checkpoint)
    best_state = checkpoint["best_trainer_state"]
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    trainer_config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(
            float(value) for value in PRIOR_CENTER.numpy().tolist()
        ),
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
        transport_id=(
            f"chart-a-cpu-xla-best-{best_state['step']}-fixed-hmc-api"
        ),
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )

    class Bridge:
        parameter_dim = 4
        parameter_names = tuple(FREE_NAMES)
        supports_retained_draw_batch = False
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True
        target_status_invalid_rows_become_nonfinite = True

        def __init__(self) -> None:
            self.target_scope = f"{target.target_scope}:fixed_hmc_api:chart-a"

        def adapter_signature(self) -> str:
            return target.adapter_signature()

        def value_score_capability(self) -> ValueScoreCapability:
            return ValueScoreCapability(
                value_score_authority="graph_native",
                xla_hmc_ready=True,
                full_chain_xla_diagnostic_ready=True,
                runtime_backend="ssl_lstm_q20_fixed_hmc_api_cpu_xla_bridge",
                evidence_path=PLAN.as_posix(),
                target_scope=self.target_scope,
                nonclaims=(
                    "CPU/XLA exception to repository GPU default",
                    "no posterior oracle",
                ),
            )

        def log_prob_and_grad(self, values: Any) -> tuple[Any, Any]:
            tensor = tf.convert_to_tensor(values, tf.float64)
            if tensor.shape.rank != 2 or tensor.shape[-1] != 4:
                raise ValueError("q=20 HMC target requires static shape [batch,4]")
            return target.batch_value_and_score(tensor)

        def log_prob_and_grad_status(
            self, values: Any
        ) -> tuple[Any, Any, Mapping[str, Any]]:
            return target.neutra_batch_log_prob_and_grad_status(values)

        def target_status_telemetry(self, values: Any) -> Mapping[str, Any]:
            return self.log_prob_and_grad_status(values)[2]

    provenance = {
        "chart": "chart-a",
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(CHECKPOINT),
        "checkpoint_hash": checkpoint["checkpoint_hash"],
        "best_trainer_state_step": int(best_state["step"]),
        "best_trainer_state_hash": best_state["state_hash"],
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "transport_hash": loaded.manifest.transport_hash,
        "transport_artifact_signature": loaded.artifact_signature,
    }
    return Bridge(), loaded.transport, provenance


def load_frozen_kernel(path: Path = TUNING_ARTIFACT) -> Mapping[str, Any]:
    """Load and fail closed on every candidate identity used by this run."""

    if _sha256(path) != EXPECTED_TUNING_SHA256:
        raise CampaignError("merged tuning artifact SHA-256 mismatch")
    payload = _read_json(path)
    if payload.get("passed") is not True or payload.get("final_status") != "passed":
        raise CampaignError("merged tuning artifact did not nominate a candidate")
    kernel = payload.get("final_kernel_payload")
    if not isinstance(kernel, Mapping):
        raise CampaignError("merged tuning artifact lacks a final kernel")
    if _stable_hash(kernel) != EXPECTED_KERNEL_HASH:
        raise CampaignError("selected kernel hash mismatch")
    exact = {
        "num_leapfrog_steps": 10,
        "step_size": 0.4148806556986277,
        "mass_policy": "fixed_identity_z",
        "use_xla": True,
        "shared_scalar_step_across_chain_bank": True,
        "transport_training_or_adaptation_used": False,
        "mass_adaptation_used": False,
    }
    for name, expected in exact.items():
        if kernel.get(name) != expected:
            raise CampaignError(f"selected kernel {name} mismatch")
    for name in (
        "base_adapter_signature",
        "transformed_adapter_signature",
        "fixed_transport_manifest_hash",
    ):
        if str(kernel.get(name)) != EXPECTED_BINDINGS[name]:
            raise CampaignError(f"selected kernel {name} binding mismatch")
    if int(kernel["num_leapfrog_steps"]) < 2:
        raise CampaignError("sequential HMC forbids L=1")
    return dict(kernel)


def _build_adapter(*, threads: int) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    base, transport, provenance = _build_chart(threads=int(threads))
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )

    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=transport,
        target_scope=TARGET_SCOPE,
        runtime_backend="ssl_lstm_q20_chart_a_l10_sequential_cpu_xla",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "CPU/XLA validation exception to repository GPU default",
            "sequential sampler screen is not posterior correctness",
        ),
    )
    kernel = load_frozen_kernel()
    checks = {
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "target_signature": provenance["target_signature"],
        "base_adapter_signature": base.adapter_signature(),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
        "transport_hash": provenance["transport_hash"],
    }
    for name, actual in checks.items():
        if str(actual) != EXPECTED_BINDINGS[name]:
            raise CampaignError(f"runtime {name} mismatch")
    return adapter, provenance, kernel


def _fold_chain_seed(seed: tuple[int, int], chain_index: int) -> tuple[int, int]:
    if len(seed) != 2:
        raise CampaignError("controller seed must have two integers")
    index = int(chain_index)
    if not 0 <= index < CHAIN_COUNT:
        raise CampaignError("chain index is outside the four-chain bank")
    return int(seed[0]) + index + 1, int(seed[1]) + 100_003 * (index + 1)


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": str(THREADS_PER_CHAIN),
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _git_manifest() -> Mapping[str, Any]:
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_dirty": bool(status),
        "status_line_count": len(status),
    }


def _read_worker_response(
    process: subprocess.Popen[str], *, timeout_seconds: float
) -> dict[str, Any]:
    if process.stdout is None:
        raise CampaignError("worker stdout is unavailable")
    ready, _, _ = select.select([process.stdout], [], [], max(0.1, timeout_seconds))
    if not ready:
        raise CampaignError(f"worker {process.pid} response timed out")
    line = process.stdout.readline()
    if not line:
        raise CampaignError(
            f"worker {process.pid} closed stdout with exit {process.poll()}"
        )
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise CampaignError("worker response must be a JSON object")
    return payload


def _send_worker_request(
    process: subprocess.Popen[str], payload: Mapping[str, Any]
) -> None:
    if process.stdin is None:
        raise CampaignError("worker stdin is unavailable")
    process.stdin.write(json.dumps(_json_ready(payload), allow_nan=False) + "\n")
    process.stdin.flush()


def _reassemble_worker_chunk(
    rows: Sequence[Mapping[str, Any]], *, active_results: int
) -> tuple[Any, Mapping[str, Any]]:
    import tensorflow as tf

    ordered = tuple(sorted(rows, key=lambda row: int(row["chain_index"])))
    if tuple(int(row["chain_index"]) for row in ordered) != tuple(range(CHAIN_COUNT)):
        raise CampaignError("worker responses do not cover chains 0..3")
    samples = tf.stack(
        [tf.convert_to_tensor(row["samples"], tf.float64) for row in ordered],
        axis=1,
    )
    expected_samples = (int(active_results), CHAIN_COUNT, DIMENSION)
    if tuple(samples.shape) != expected_samples:
        raise CampaignError(
            f"reassembled sample shape {tuple(samples.shape)} != {expected_samples}"
        )

    def stack(name: str, dtype: Any, *, trailing: bool = False) -> Any:
        result = tf.stack(
            [tf.convert_to_tensor(row["trace"][name], dtype) for row in ordered],
            axis=1,
        )
        expected = (
            (int(active_results), CHAIN_COUNT, DIMENSION)
            if trailing
            else (int(active_results), CHAIN_COUNT)
        )
        if tuple(result.shape) != expected:
            raise CampaignError(
                f"reassembled trace {name} shape {tuple(result.shape)} != {expected}"
            )
        return result

    return samples, {
        "is_accepted": stack("is_accepted", tf.bool),
        "log_accept_ratio": stack("log_accept_ratio", tf.float64),
        "target_log_prob": stack("target_log_prob", tf.float64),
        "proposed_target_log_prob": stack(
            "proposed_target_log_prob", tf.float64
        ),
        "target_score": stack("target_score", tf.float64, trailing=True),
        "delta_h": stack("delta_h", tf.float64),
        # These are actual target telemetry, not a finiteness-derived proxy.
        "target_status_code": stack("target_status_code", tf.int32),
        "target_valid_pre_regularized_score": stack(
            "target_valid_pre_regularized_score", tf.bool
        ),
    }


def _forecast_allows_next_chunk(
    *, elapsed_seconds: float, cap_seconds: float, completed_chunk_seconds: Sequence[float]
) -> bool:
    if not completed_chunk_seconds:
        return elapsed_seconds + ARCHIVE_RESERVE_SECONDS < cap_seconds
    forecast = max(float(value) for value in completed_chunk_seconds)
    required = FORECAST_MARGIN * forecast + ARCHIVE_RESERVE_SECONDS
    return elapsed_seconds + required <= cap_seconds


def _terminate_workers(
    processes: Sequence[subprocess.Popen[str]], logs: Sequence[Any]
) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                _send_worker_request(process, {"command": "stop"})
            except (CampaignError, BrokenPipeError, OSError):
                process.terminate()
    deadline = time.monotonic() + 30.0
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
    for log in logs:
        if not log.closed:
            log.close()


def _start_workers(
    *, output: Path
) -> tuple[list[subprocess.Popen[str]], list[Any], tuple[Mapping[str, Any], ...]]:
    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    try:
        for chain_index, cpus in enumerate(CHAIN_CPUS):
            specification = f"{cpus[0]}-{cpus[-1]}"
            command = (
                "taskset",
                "-c",
                specification,
                sys.executable,
                str(SCRIPT),
                "--mode",
                "worker",
                "--chain-index",
                str(chain_index),
            )
            log = (output / f"chain-{chain_index}.stderr.log").open(
                "w", encoding="utf-8"
            )
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=_worker_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
            processes.append(process)
            logs.append(log)
        ready = tuple(
            _read_worker_response(
                process, timeout_seconds=WORKER_START_TIMEOUT_SECONDS
            )
            for process in processes
        )
        for index, row in enumerate(ready):
            if row.get("event") != "ready" or int(row.get("chain_index", -1)) != index:
                raise CampaignError("worker readiness identity mismatch")
            if tuple(row.get("affinity", ())) != CHAIN_CPUS[index]:
                raise CampaignError("worker affinity differs from reviewed allocation")
            if row.get("physical_gpus") != [] or row.get("cuda_visible_devices") != "-1":
                raise CampaignError("worker did not preserve CPU-only isolation")
            if row.get("kernel_hash") != EXPECTED_KERNEL_HASH:
                raise CampaignError("worker kernel hash mismatch")
            if row.get("policy_id") != POLICY_ID:
                raise CampaignError("worker sequential policy mismatch")
        return processes, logs, ready
    except BaseException:
        _terminate_workers(processes, logs)
        raise


def _run_worker(args: argparse.Namespace) -> int:
    index = int(args.chain_index)
    if not 0 <= index < CHAIN_COUNT:
        raise CampaignError("worker chain index is invalid")
    if tuple(sorted(os.sched_getaffinity(0))) != CHAIN_CPUS[index]:
        raise CampaignError("worker CPU affinity differs from reviewed allocation")

    adapter, provenance, kernel = _build_adapter(threads=THREADS_PER_CHAIN)
    import tensorflow as tf
    from bayesfilter.inference.neutra_hmc import (
        SequentialNeuTraHMCConfig,
        _ChunkRunner,
    )

    config = SequentialNeuTraHMCConfig(
        step_size=float(kernel["step_size"]),
        num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
        seed=ROOT_SEED,
        warmup_chunk_size=CHUNK_RESULTS,
        warmup_min_results=CHUNK_RESULTS,
        warmup_window_results=CHUNK_RESULTS,
        warmup_max_results=CHUNK_RESULTS,
        retained_chunk_size=CHUNK_RESULTS,
        retained_min_results=CHUNK_RESULTS,
        retained_max_results=CHUNK_RESULTS,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
        acceptance_min=0.0,
        acceptance_max=1.0,
        chain_count=CHAIN_COUNT,
        use_xla=True,
        target_status_required=True,
    )
    initial = tf.convert_to_tensor(INITIAL_Z[index], tf.float64)[None, :]
    runner = _ChunkRunner(adapter, initial, config)
    ready = {
        "schema": SCHEMA,
        "event": "ready",
        "policy_id": POLICY_ID,
        "chain_index": index,
        "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "configured_intra_op_threads": THREADS_PER_CHAIN,
        "configured_inter_op_threads": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(item) for item in tf.config.list_physical_devices("GPU")],
        "tensorflow": tf.__version__,
        "jit_compile": True,
        "dtype": "float64",
        "kernel_hash": _stable_hash(kernel),
        "kernel": kernel,
        "provenance": provenance,
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    print(json.dumps(_json_ready(ready), allow_nan=False), flush=True)

    invocation = 0
    while True:
        line = sys.stdin.readline()
        if not line:
            return 0
        request = json.loads(line)
        if request.get("command") == "stop":
            print(json.dumps({"event": "stopped", "chain_index": index}), flush=True)
            return 0
        if request.get("command") != "chunk":
            raise CampaignError("unknown worker command")
        if int(request["active_results"]) != CHUNK_RESULTS:
            raise CampaignError("worker received a non-reviewed chunk size")
        state = tf.convert_to_tensor(request["state"], tf.float64)
        if tuple(state.shape) != (DIMENSION,):
            raise CampaignError("worker state must have shape [4]")
        seed = tuple(int(value) for value in request["seed"])
        started = time.perf_counter()
        samples, trace = runner.run(state[None, :], tf.constant(seed, tf.int32))
        chunk_seconds = time.perf_counter() - started
        samples = tf.squeeze(tf.convert_to_tensor(samples, tf.float64), axis=1)

        def squeeze(name: str, dtype: Any) -> Any:
            return tf.squeeze(tf.convert_to_tensor(trace[name], dtype), axis=1)

        response_trace = {
            "is_accepted": squeeze("is_accepted", tf.bool),
            "log_accept_ratio": squeeze("log_accept_ratio", tf.float64),
            "target_log_prob": squeeze("target_log_prob", tf.float64),
            "proposed_target_log_prob": squeeze(
                "proposed_target_log_prob", tf.float64
            ),
            "target_score": squeeze("target_score", tf.float64),
            "delta_h": squeeze("delta_h", tf.float64),
            "target_status_code": squeeze("target_status_code", tf.int32),
            "target_valid_pre_regularized_score": squeeze(
                "target_valid_pre_regularized_score", tf.bool
            ),
        }
        required_float = (
            samples,
            response_trace["log_accept_ratio"],
            response_trace["target_log_prob"],
            response_trace["proposed_target_log_prob"],
            response_trace["target_score"],
            response_trace["delta_h"],
        )
        all_finite = all(
            bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
            for value in required_float
        )
        invocation += 1
        response = {
            "schema": SCHEMA,
            "event": "chunk_done",
            "request_id": request["request_id"],
            "chain_index": index,
            "seed": seed,
            "active_results": CHUNK_RESULTS,
            "samples": samples,
            "trace": response_trace,
            "chunk_seconds": chunk_seconds,
            "invocation": invocation,
            "all_required_tensors_finite": all_finite,
            "native_divergence_status": "not_exposed_by_kernel",
            "native_divergence_count": None,
            "ru_maxrss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        print(json.dumps(_json_ready(response), allow_nan=False), flush=True)


def _validate_affinity_contract() -> Mapping[str, Any]:
    probes = {}
    assignments = {f"chain-{index}": cpus for index, cpus in enumerate(CHAIN_CPUS)}
    assignments["supervisor"] = (SUPERVISOR_CPU,)
    for label, cpus in assignments.items():
        specification = (
            str(cpus[0]) if len(cpus) == 1 else f"{cpus[0]}-{cpus[-1]}"
        )
        probe = subprocess.run(
            ("taskset", "-c", specification, "true"),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        probes[label] = {"cpus": cpus, "exit_code": probe.returncode}
        if probe.returncode != 0:
            raise CampaignError(f"cannot assign reviewed CPUs for {label}")
    return probes


def _run_preflight(args: argparse.Namespace) -> int:
    if tuple(sorted(os.sched_getaffinity(0))) != (SUPERVISOR_CPU,):
        raise CampaignError("preflight must be pinned to CPU 32")
    output = (ROOT / (args.output_root or DEFAULT_PREFLIGHT_OUTPUT)).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("preflight output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    affinity = _validate_affinity_contract()
    adapter, provenance, kernel = _build_adapter(threads=1)
    import tensorflow as tf

    states = tf.convert_to_tensor(INITIAL_Z, tf.float64)
    values, scores, status = adapter.log_prob_and_grad_status(states)
    status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
    valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
    passed = bool(
        tf.reduce_all(tf.math.is_finite(values)).numpy()
        and tf.reduce_all(tf.math.is_finite(scores)).numpy()
        and tf.reduce_all(tf.equal(status_code, 0)).numpy()
        and tf.reduce_all(valid).numpy()
    )
    if not passed:
        raise CampaignError("initial-state value/score/status preflight failed")
    payload = {
        "schema": SCHEMA,
        "role": "preflight",
        "status": "PREFLIGHT_PASSED",
        "policy_id": POLICY_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "affinity": affinity,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": [str(item) for item in tf.config.list_physical_devices("GPU")],
        "jit_compile": True,
        "dtype": "float64",
        "initial_state_shape": list(states.shape),
        "initial_values": values,
        "initial_scores_all_finite": True,
        "initial_status_code": status_code,
        "initial_valid_pre_regularized_score": valid,
        "kernel": kernel,
        "kernel_hash": _stable_hash(kernel),
        "provenance": provenance,
        "git": _git_manifest(),
        "source_hashes": {
            "launcher": _sha256(SCRIPT),
            "plan": _sha256(ROOT / PLAN),
            "controller": _sha256(ROOT / "bayesfilter/inference/neutra_hmc.py"),
        },
        "route_ledger_test_status": "not_run_missing_referenced_ledger",
        "nonclaims": [
            "preflight only",
            "no HMC transitions, convergence, or posterior claim",
        ],
    }
    _write_json(output / "summary.json", payload)
    print(json.dumps({"status": payload["status"], "wall_seconds": payload["wall_seconds"]}))
    return 0


def _run_campaign(args: argparse.Namespace) -> int:
    import tensorflow as tf
    from bayesfilter.inference.neutra_hmc import (
        NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        SequentialNeuTraHMCConfig,
        run_sequential_neutra_hmc,
    )

    if NEUTRA_SEQUENTIAL_HMC_POLICY_ID != POLICY_ID:
        raise CampaignError("repository sequential HMC policy identity drifted")
    if tuple(sorted(os.sched_getaffinity(0))) != (SUPERVISOR_CPU,):
        raise CampaignError("campaign supervisor must be pinned to CPU 32")
    cap_seconds = float(args.cap_seconds)
    if not 0.0 < cap_seconds <= DEFAULT_CAP_SECONDS:
        raise CampaignError("campaign cap must be positive and no greater than 86400 s")
    output = (ROOT / (args.output_root or DEFAULT_OUTPUT)).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("campaign output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    affinity = _validate_affinity_contract()
    kernel = load_frozen_kernel()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    launch = {
        "schema": SCHEMA,
        "role": "material_launch",
        "status": "RUNNING",
        "policy_id": POLICY_ID,
        "started_utc": started_utc,
        "cap_seconds": cap_seconds,
        "output_root": str(output.relative_to(ROOT)),
        "command": sys.argv,
        "supervisor_pid": os.getpid(),
        "supervisor_affinity": sorted(os.sched_getaffinity(0)),
        "worker_affinity": CHAIN_CPUS,
        "threads_per_chain": THREADS_PER_CHAIN,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kernel": kernel,
        "kernel_hash": _stable_hash(kernel),
        "initial_z": INITIAL_Z,
        "root_seed": ROOT_SEED,
        "affinity_probes": affinity,
        "git": _git_manifest(),
        "plan": PLAN.as_posix(),
        "source_hashes": {
            "launcher": _sha256(SCRIPT),
            "plan": _sha256(ROOT / PLAN),
            "controller": _sha256(ROOT / "bayesfilter/inference/neutra_hmc.py"),
        },
    }
    _write_json(output / "launch.json", launch)

    processes: list[subprocess.Popen[str]] = []
    logs: list[Any] = []
    requests: list[Mapping[str, Any]] = []
    completed_slowest_chunk_seconds: list[float] = []
    try:
        processes, logs, worker_ready = _start_workers(output=output)
        adapter, provenance, runtime_kernel = _build_adapter(threads=1)
        if _stable_hash(runtime_kernel) != EXPECTED_KERNEL_HASH:
            raise CampaignError("supervisor runtime kernel mismatch")
        config = SequentialNeuTraHMCConfig(
            step_size=float(kernel["step_size"]),
            num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
            seed=ROOT_SEED,
            warmup_chunk_size=CHUNK_RESULTS,
            warmup_min_results=2000,
            warmup_window_results=1000,
            warmup_max_results=10000,
            retained_chunk_size=CHUNK_RESULTS,
            retained_min_results=1000,
            retained_max_results=10000,
            warmup_rhat_max=1.05,
            retained_rhat_max=1.01,
            bulk_ess_min=400.0,
            tail_ess_min=400.0,
            acceptance_min=0.35,
            acceptance_max=0.95,
            chain_count=CHAIN_COUNT,
            use_xla=True,
            target_status_required=True,
        )
        request_index = 0

        def run_chunk(state: Any, seed: tuple[int, int], _config: Any):
            nonlocal request_index
            state_tensor = tf.convert_to_tensor(state, tf.float64)
            if tuple(state_tensor.shape) != (CHAIN_COUNT, DIMENSION):
                raise CampaignError("controller supplied an invalid chain state bank")
            request_id = f"chunk-{request_index:04d}"
            folded_seeds = tuple(
                _fold_chain_seed(tuple(seed), index) for index in range(CHAIN_COUNT)
            )
            for index, process in enumerate(processes):
                _send_worker_request(
                    process,
                    {
                        "command": "chunk",
                        "request_id": request_id,
                        "active_results": CHUNK_RESULTS,
                        "state": state_tensor[index],
                        "seed": folded_seeds[index],
                    },
                )
            rows = []
            for process in processes:
                remaining = cap_seconds - (time.perf_counter() - started)
                if remaining <= ARCHIVE_RESERVE_SECONDS:
                    raise CampaignError("campaign cap reached during an active chunk")
                rows.append(
                    _read_worker_response(
                        process,
                        timeout_seconds=remaining - ARCHIVE_RESERVE_SECONDS,
                    )
                )
            for index, row in enumerate(rows):
                if row.get("event") != "chunk_done":
                    raise CampaignError("worker chunk event is invalid")
                if row.get("request_id") != request_id:
                    raise CampaignError("worker request identity mismatch")
                if int(row.get("chain_index", -1)) != index:
                    raise CampaignError("worker chain identity mismatch")
                if tuple(row.get("seed", ())) != folded_seeds[index]:
                    raise CampaignError("worker seed identity mismatch")
                if row.get("all_required_tensors_finite") is not True:
                    raise CampaignError("worker returned nonfinite required tensors")
            worker_seconds = tuple(float(row["chunk_seconds"]) for row in rows)
            slowest = max(worker_seconds)
            completed_slowest_chunk_seconds.append(slowest)
            request = {
                "schema": SCHEMA,
                "request_id": request_id,
                "controller_seed": seed,
                "folded_chain_seeds": folded_seeds,
                "worker_chunk_seconds": worker_seconds,
                "slowest_worker_seconds": slowest,
                "worker_ru_maxrss_bytes": tuple(
                    int(row["ru_maxrss_bytes"]) for row in rows
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(output / "progress" / f"{request_id}.json", request)
            requests.append(request)
            request_index += 1
            return _reassemble_worker_chunk(rows, active_results=CHUNK_RESULTS)

        def budget_check(_transition_leapfrog_count: int) -> bool:
            return _forecast_allows_next_chunk(
                elapsed_seconds=time.perf_counter() - started,
                cap_seconds=cap_seconds,
                completed_chunk_seconds=completed_slowest_chunk_seconds,
            )

        result = run_sequential_neutra_hmc(
            adapter,
            tf.convert_to_tensor(INITIAL_Z, tf.float64),
            config,
            archive_root=output / "archive",
            archive_label="chart-a-l10",
            budget_check=budget_check,
            run_chunk=run_chunk,
        )
        # The shared archived-result payload method in this worktree references
        # a stale schema symbol. Serialize the public dataclass fields directly
        # while preserving the schema emitted by its archive manifest.
        public_result = {
            "schema": "bayesfilter.neutra.sequential_hmc_result.v1",
            **asdict(result),
        }
        _write_json(output / "sequential-result.json", public_result)
        cap_limited = "campaign_resource_cap" in result.diagnostics.get(
            "hard_vetoes", ()
        )
        status = (
            "SEQUENTIAL_SCREEN_PASSED"
            if result.passed
            else "UNDER_BUDGETED_PARTIAL"
            if cap_limited
            else "SEQUENTIAL_SCREEN_NOT_PASSED"
        )
        summary = {
            "schema": SCHEMA,
            "role": "material_result",
            "status": status,
            "policy_id": POLICY_ID,
            "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": cap_seconds,
            "kernel": kernel,
            "kernel_hash": EXPECTED_KERNEL_HASH,
            "provenance": provenance,
            "worker_ready": worker_ready,
            "request_count": len(requests),
            "slowest_chunk_seconds": completed_slowest_chunk_seconds,
            "warmup_results_per_chain": result.warmup_results_per_chain,
            "retained_results_per_chain": result.retained_results_per_chain,
            "stop_reason": result.stop_reason,
            "passed": result.passed,
            "diagnostics": result.diagnostics,
            "archive": result.archive,
            "warmup_excluded_from_posterior": True,
            "native_divergence_status": "not_exposed_by_kernel",
            "native_divergence_count": None,
            "route_ledger_test_status": "not_run_missing_referenced_ledger",
            "git": _git_manifest(),
            "nonclaims": [
                "sequential finite-sample sampler screen only",
                "no posterior correctness or model-validity claim",
                "native divergence unavailability is not zero divergences",
                "no sampler, chart, CPU/GPU, or default-readiness ranking",
            ],
        }
        _write_json(output / "summary.json", summary)
        print(
            json.dumps(
                {
                    "status": status,
                    "stop_reason": result.stop_reason,
                    "warmup_results_per_chain": result.warmup_results_per_chain,
                    "retained_results_per_chain": result.retained_results_per_chain,
                    "wall_seconds": summary["wall_seconds"],
                }
            ),
            flush=True,
        )
        return 0 if result.passed or cap_limited else 2
    finally:
        _terminate_workers(processes, logs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run", "worker"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=float, default=DEFAULT_CAP_SECONDS)
    parser.add_argument("--chain-index", type=int)
    args = parser.parse_args(argv)
    if args.mode == "worker" and args.chain_index is None:
        parser.error("--chain-index is required for worker mode")
    if args.mode != "worker" and args.chain_index is not None:
        parser.error("--chain-index is only valid for worker mode")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "worker":
        return _run_worker(args)
    if args.mode == "preflight":
        return _run_preflight(args)
    return _run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
