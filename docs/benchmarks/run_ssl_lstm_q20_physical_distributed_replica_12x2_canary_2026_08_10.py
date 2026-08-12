#!/usr/bin/env python3
"""Test the frozen 12-worker x 2-row replica topology repair."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_12x2_canary_2026_08_10.py"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
POOL_HELPER = Path("bayesfilter/inference/tf_batch_value_score_pool.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
CHECKPOINT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r4-four-chain-checkpoint/checkpoint.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r5-topology-12x2-canary"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "canary.json"
LOG = OUTPUT_ROOT / "run.log"

WORKERS = 12
ROWS_PER_WORKER = 2
ROWS = 24
WORKER_CPU_IDS = tuple(range(32, 44))
PARENT_CPU_IDS = tuple(range(32, 48))
CAP_SECONDS = 300.0
MATERIAL_TRANSITIONS = 1300
MARGIN = 1.5
MATERIAL_CAP_SECONDS = 20000.0
CHECKPOINT_SHA256 = "8276947db5785786567c5194b469c0938907820faf8d1bafd0265b1d4f87adab"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"


class TopologyCanaryError(RuntimeError):
    """Raised when the 12x2 topology cannot preserve exact evidence."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise TopologyCanaryError(f"refusing to overwrite: {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise TopologyCanaryError(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load_checkpoint_runner() -> Any:
    name = "physical_checkpoint_support_for_12x2"
    spec = importlib.util.spec_from_file_location(name, _abs(CHECKPOINT_RUNNER))
    if spec is None or spec.loader is None:
        raise TopologyCanaryError("cannot load checkpoint support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pool_config() -> Any:
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePoolConfig

    return TFBatchValueScorePoolConfig(
        factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
            "batch_native_complexity_target_worker_factory"
        ),
        factory_config={
            "q": 20,
            "principal_sqrt_backend": "tensorflow_eigh",
            "jit_compile": True,
        },
        dimension=4,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(ROWS_PER_WORKER,),
        batch_per_worker=ROWS_PER_WORKER,
        worker_cpu_ids=WORKER_CPU_IDS,
        timeout_seconds=900.0,
    )


def run_canary() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise TopologyCanaryError("refusing to overwrite 12x2 canary")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise TopologyCanaryError("parent CPU affinity mismatch")
    if _sha(CHECKPOINT) != CHECKPOINT_SHA256 or _sha(GEOMETRY) != GEOMETRY_SHA256:
        raise TopologyCanaryError("bound checkpoint/geometry identity mismatch")
    checkpoint = json.loads(_abs(CHECKPOINT).read_text(encoding="utf-8"))
    if checkpoint.get("status") != "FOUR_CHAIN_CHECKPOINT_PASSED_NOT_NOMINATED":
        raise TopologyCanaryError("r4 checkpoint is not the expected cost rejection")
    if checkpoint["nomination_screen"]["material_projection_within_20000_seconds"]:
        raise TopologyCanaryError("r4 checkpoint did not fail the bound cost screen")
    _write_json(PROGRESS, {"status": "TOPOLOGY_12X2_STARTING"}, overwrite=True)

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise TopologyCanaryError("CPU-only canary found visible GPU")
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
    )

    support = _load_checkpoint_runner()
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = support._chart(tf, geometry)
    chain_centers = tf.gather(chart["latent_centers"], (0, 1, 0, 1))
    initial_state = tf.repeat(
        chain_centers[tf.newaxis, :, :], len(support.BETAS), axis=0
    )
    wave_seconds = []
    with TFBatchValueScorePool(_pool_config()) as pool:
        def evaluator(rows: tf.Tensor, request_id: str):
            wave_started = time.perf_counter()
            latent = tf.ensure_shape(tf.convert_to_tensor(rows, tf.float64), (ROWS, 4))
            theta = chart["center"] + tf.matmul(
                latent, chart["factor"], transpose_b=True
            )
            value, score, status, metadata = pool.evaluate_with_status(
                theta, request_id=request_id
            )
            wave_seconds.append(time.perf_counter() - wave_started)
            return (
                tf.convert_to_tensor(value, tf.float64) + chart["log_abs_determinant"],
                tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
                status,
                metadata,
            )

        initial = initialize_distributed_replica_state(initial_state, evaluator=evaluator)
        identity = support._worker_identity(initial["evaluation_metadata"])
        expected_identity = {
            **identity,
            "worker_count": WORKERS,
            "assigned_cpu_ids": WORKER_CPU_IDS,
        }
        identity_passed = bool(
            len(identity["worker_pids"]) == WORKERS
            and tuple(identity["assigned_cpu_ids"]) == WORKER_CPU_IDS
            and tuple(identity["target_signatures"]) == (support.TARGET_SIGNATURE,)
            and tuple(identity["adapter_signatures"]) == (support.ADAPTER_SIGNATURE,)
            and tuple(identity["status_jit_compile_values"]) == (True,)
        )
        transition_started = time.perf_counter()
        transition = distributed_replica_exchange_transition(
            state=initial["state"],
            base_target_log_prob=initial["base_target_log_prob"],
            base_score=initial["base_score"],
            identities_at_temperature=initial["identities_at_temperature"],
            inverse_temperatures=support.BETAS,
            step_sizes=support.STEPS,
            num_leapfrog_steps=support.LEAPFROG,
            transition_index=0,
            master_seed=(20260810, 7301),
            evaluator=evaluator,
        )
        transition_seconds = time.perf_counter() - transition_started
        cache_started = time.perf_counter()
        cache_value, cache_score, cache_status, _metadata = evaluator(
            tf.reshape(transition["state"], (ROWS, 4)), "terminal-cache-check"
        )
        cache_seconds = time.perf_counter() - cache_started
        cache_value = tf.reshape(cache_value, (len(support.BETAS), support.CHAINS))
        cache_score = tf.reshape(cache_score, (len(support.BETAS), support.CHAINS, 4))
        cache_valid = tf.logical_and(
            tf.convert_to_tensor(cache_status["status_code"], tf.int32) == 0,
            tf.convert_to_tensor(
                cache_status["valid_pre_regularized_score"], tf.bool
            ),
        )
        value_residual = tf.reduce_max(
            tf.abs(cache_value - transition["base_target_log_prob"])
        )
        score_residual = tf.reduce_max(
            tf.abs(cache_score - transition["base_score"])
        )

    invalid = tf.logical_not(transition["hmc_path_valid"])
    invalid_self_rejected = tf.reduce_all(
        tf.logical_not(tf.boolean_mask(transition["hmc_is_accepted"], invalid))
    )
    matrix = tf.cast(transition["swap_is_accepted_matrix"], tf.int32)
    swap_permutation = tf.logical_and(
        tf.reduce_all(
            tf.reduce_sum(matrix, axis=0)
            == tf.ones((len(support.BETAS), support.CHAINS), tf.int32)
        ),
        tf.reduce_all(
            tf.reduce_sum(matrix, axis=1)
            == tf.ones((len(support.BETAS), support.CHAINS), tf.int32)
        ),
    )
    estimated_checkpoint_cost = transition_seconds + cache_seconds / 5.0
    projection = estimated_checkpoint_cost * MATERIAL_TRANSITIONS * MARGIN
    hard_gates = {
        "worker_identity_passed": identity_passed,
        "state_and_log_acceptance_finite": bool(
            tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(transition["state"])),
                        tf.reduce_all(
                            tf.math.is_finite(transition["hmc_log_accept_ratio"])
                        ),
                    )
                )
            ).numpy()
        ),
        "invalid_paths_self_rejected": bool(invalid_self_rejected.numpy()),
        "swap_matrix_is_permutation": bool(swap_permutation.numpy()),
        "terminal_status_all_valid": bool(tf.reduce_all(cache_valid).numpy()),
        "terminal_cache_value_matches": bool((value_residual <= 1.0e-9).numpy()),
        "terminal_cache_score_matches": bool((score_residual <= 1.0e-8).numpy()),
        "wall_time_within_300_seconds": time.perf_counter() - started <= CAP_SECONDS,
    }
    projection_passed = projection <= MATERIAL_CAP_SECONDS
    passed = all(hard_gates.values()) and projection_passed
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.topology_12x2.v1",
        "status": (
            "TOPOLOGY_12X2_CANARY_NOMINATED"
            if passed
            else (
                "TOPOLOGY_12X2_CANARY_VALID_COST_FAILED"
                if all(hard_gates.values())
                else "TOPOLOGY_12X2_CANARY_FAILED"
            )
        ),
        "role": "same_kernel_topology_cost_repair_canary",
        "configuration": {
            "workers": WORKERS,
            "rows_per_worker": ROWS_PER_WORKER,
            "worker_cpu_ids": WORKER_CPU_IDS,
            "inverse_temperatures": support.BETAS,
            "step_sizes": support.STEPS,
            "num_leapfrog_steps": support.LEAPFROG,
            "chains": support.CHAINS,
            "jit_compile": True,
            "training_update": False,
        },
        "hard_gates": hard_gates,
        "material_projection_within_20000_seconds": projection_passed,
        "initial_seconds": wave_seconds[0],
        "transition_seconds": transition_seconds,
        "cache_seconds": cache_seconds,
        "estimated_checkpoint_cost_per_transition_seconds": estimated_checkpoint_cost,
        "projected_minimum_material_seconds_with_50pct_margin": projection,
        "required_maximum_checkpoint_cost_per_transition_seconds": (
            MATERIAL_CAP_SECONDS / (MATERIAL_TRANSITIONS * MARGIN)
        ),
        "r4_checkpoint_cost_per_transition_seconds": checkpoint[
            "checkpoint_cost_per_transition_seconds"
        ],
        "descriptive_r4_over_12x2_cost_ratio": (
            checkpoint["checkpoint_cost_per_transition_seconds"]
            / estimated_checkpoint_cost
        ),
        "worker_identity": expected_identity,
        "wave_seconds": wave_seconds,
        "invalid_hmc_path_count": tf.reduce_sum(tf.cast(invalid, tf.int32)),
        "trace_receipts": {
            name: _write_tensor(
                OUTPUT_ROOT / f"transition-{name}.tftensor", transition[name], tf
            )
            for name in (
                "state",
                "base_target_log_prob",
                "base_score",
                "hmc_is_accepted",
                "hmc_log_accept_ratio",
                "hmc_path_valid",
                "swap_is_accepted_matrix",
            )
        },
        "bindings": {
            "r4_checkpoint_sha256": _sha(CHECKPOINT),
            "geometry_sha256": _sha(GEOMETRY),
        },
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "git_dirty": bool(subprocess.run(
                ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip()),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "wall_time_seconds": time.perf_counter() - started,
            "artifact_root": OUTPUT_ROOT.as_posix(),
            "plan_file": PLAN.as_posix(),
            "result_file": RESULT.as_posix(),
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "checkpoint_runner": _sha(CHECKPOINT_RUNNER),
                "helper": _sha(HELPER),
                "pool_helper": _sha(POOL_HELPER),
            },
        },
        "nonclaims": (
            "one-transition topology cost canary only",
            "no statistical topology ranking",
            "no travel, convergence, posterior, mass, predictive, or default claim",
        ),
    }
    if not all(hard_gates.values()):
        raise TopologyCanaryError(f"12x2 hard gates failed: {hard_gates}")
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": payload["status"],
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run_canary()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.topology_12x2_failure.v1",
            "status": "TOPOLOGY_12X2_CANARY_HARNESS_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_time_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        raise
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
