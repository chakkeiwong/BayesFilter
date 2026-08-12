#!/usr/bin/env python3
"""Run one exact-target distributed physical replica-exchange transition."""

from __future__ import annotations

import hashlib
import json
import math
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
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_canary_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
POOL_HELPER = Path("bayesfilter/inference/tf_batch_value_score_pool.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-"
    "2026-08-10/r1/geometry.json"
)
TIMING = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-"
    "2026-08-10/r1-timing/timing.json"
)
FAILED_R2 = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-"
    "2026-08-10/r2-distributed-canary/canary.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-"
    "2026-08-10/r3-distributed-canary"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "canary.json"
LOG = OUTPUT_ROOT / "run.log"

PARAMETER_DIM = 4
BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
STEPS = tuple(0.05 / math.sqrt(beta) for beta in BETAS)
LEAPFROG = 8
CHAINS = 2
ROWS = len(BETAS) * CHAINS
WORKERS = ROWS
WORKER_CPU_IDS = tuple(range(32, 32 + WORKERS))
PARENT_CPU_IDS = tuple(range(32, 48))
CAP_SECONDS = 3600.0
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
TIMING_SHA256 = "d4a0be4b4ac0a8fe5d4daf1a4a3bfb1425f774e393231a2f987aa5fe248ed4ed"
FAILED_R2_SHA256 = "bfc3b2a4b4afcea87010cbf434d21b911171dfa155e86e8f979f799ac9c6b30f"


class DistributedReplicaCanaryError(RuntimeError):
    """Raised when distributed replica mechanics cannot preserve evidence."""


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
        raise DistributedReplicaCanaryError(f"refusing to overwrite: {path}")
    encoded = json.dumps(
        _safe(payload), sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _append_log(message: str) -> None:
    absolute = _abs(LOG)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    with absolute.open("a", encoding="ascii") as stream:
        stream.write(f"{time.time():.6f} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise DistributedReplicaCanaryError(f"refusing to overwrite tensor: {path}")
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


def _chart(tf: Any, geometry: Mapping[str, Any]) -> Mapping[str, Any]:
    labels = ("plus", "minus")
    centers = tf.constant(
        [geometry["representatives"][label]["position"] for label in labels],
        tf.float64,
    )
    precisions = tf.stack(
        [
            tf.constant(
                geometry["source_curvature"][label]["records"][-1]["precision"],
                tf.float64,
            )
            for label in labels
        ]
    )
    covariances = tf.linalg.inv(precisions)
    center = tf.reduce_mean(centers, axis=0)
    displacement = centers - center
    pooled_covariance = tf.reduce_mean(covariances, axis=0) + tf.einsum(
        "ni,nj->ij", displacement, displacement
    ) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled_covariance)
    factor = tf.matmul(
        eigenvectors * tf.sqrt(eigenvalues)[tf.newaxis, :],
        eigenvectors,
        transpose_b=True,
    )
    latent_centers = tf.transpose(
        tf.linalg.solve(factor, tf.transpose(centers - center))
    )
    return {
        "center": center,
        "factor": factor,
        "latent_centers": latent_centers,
        "source_centers": centers,
        "log_abs_determinant": tf.reduce_sum(tf.math.log(eigenvalues)) / 2.0,
    }


def _pool_config() -> Any:
    from bayesfilter.inference.tf_batch_value_score_pool import (
        TFBatchValueScorePoolConfig,
    )

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
        dimension=PARAMETER_DIM,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(1,),
        batch_per_worker=1,
        worker_cpu_ids=WORKER_CPU_IDS,
        timeout_seconds=900.0,
    )


def _worker_identity_gate(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    workers = tuple(metadata["startup_worker_metadata"])
    signatures = {str(row["target_signature"]) for row in workers}
    adapters = {str(row["adapter_signature"]) for row in workers}
    assigned = {int(row["assigned_cpu"]) for row in workers}
    jit = {bool(row["status_jit_compile"]) for row in workers}
    policies = {str(row["evaluation_policy"]) for row in workers}
    return {
        "worker_count": len(workers),
        "worker_pids": tuple(int(row["pid"]) for row in workers),
        "target_signatures": tuple(sorted(signatures)),
        "adapter_signatures": tuple(sorted(adapters)),
        "assigned_cpu_ids": tuple(sorted(assigned)),
        "status_jit_compile_values": tuple(sorted(jit)),
        "evaluation_policies": tuple(sorted(policies)),
        "passed": (
            len(workers) == WORKERS
            and signatures == {TARGET_SIGNATURE}
            and adapters == {ADAPTER_SIGNATURE}
            and assigned == set(WORKER_CPU_IDS)
            and jit == {True}
            and policies == {"batch_native_tensorflow_status_no_row_mapping_v2"}
        ),
    }


def run_canary() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise DistributedReplicaCanaryError("refusing to overwrite distributed canary")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise DistributedReplicaCanaryError("parent CPU affinity does not match canary")
    if (
        _sha(GEOMETRY) != GEOMETRY_SHA256
        or _sha(TIMING) != TIMING_SHA256
        or _sha(FAILED_R2) != FAILED_R2_SHA256
    ):
        raise DistributedReplicaCanaryError("bound geometry or timing identity mismatch")
    timing = json.loads(_abs(TIMING).read_text(encoding="utf-8"))
    if timing.get("status") != "PHYSICAL_REPLICA_TIMING_PASSED":
        raise DistributedReplicaCanaryError("monolithic timing comparator did not pass")

    _write_json(
        PROGRESS,
        {"status": "DISTRIBUTED_REPLICA_CANARY_STARTING", "completed_waves": 0},
        overwrite=True,
    )
    _append_log("starting distributed exact-target replica canary")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise DistributedReplicaCanaryError("CPU-only canary found visible GPU")
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
    )

    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = _chart(tf, geometry)
    initial_state = tf.repeat(
        chart["latent_centers"][tf.newaxis, :, :], len(BETAS), axis=0
    )
    evaluation_index = 0
    evaluation_rows = []

    with TFBatchValueScorePool(_pool_config()) as pool:
        def evaluator(rows: tf.Tensor, request_id: str):
            nonlocal evaluation_index
            wave_started = time.perf_counter()
            latent = tf.ensure_shape(
                tf.convert_to_tensor(rows, tf.float64), (ROWS, PARAMETER_DIM)
            )
            theta = chart["center"] + tf.matmul(
                latent, chart["factor"], transpose_b=True
            )
            value, score, status, metadata = pool.evaluate_with_status(
                theta, request_id=request_id
            )
            elapsed = time.perf_counter() - wave_started
            evaluation_rows.append(
                {
                    "evaluation_index": evaluation_index,
                    "request_id": request_id,
                    "wall_time_seconds": elapsed,
                    "worker_runtime_max_seconds": metadata[
                        "worker_runtime_max_seconds"
                    ],
                    "worker_pids": metadata["worker_result_pids"],
                    "worker_shard_sizes": metadata["worker_shard_sizes"],
                    "valid_row_count": tf.reduce_sum(
                        tf.cast(
                            tf.logical_and(
                                tf.convert_to_tensor(status["status_code"], tf.int32)
                                == 0,
                                tf.convert_to_tensor(
                                    status["valid_pre_regularized_score"], tf.bool
                                ),
                            ),
                            tf.int32,
                        )
                    ),
                }
            )
            evaluation_index += 1
            _write_json(
                PROGRESS,
                {
                    "status": "DISTRIBUTED_REPLICA_CANARY_RUNNING",
                    "completed_waves": evaluation_index,
                    "last_request_id": request_id,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=True,
            )
            _append_log(
                f"completed evaluation {evaluation_index} {request_id} in {elapsed:.6f}s"
            )
            return (
                tf.convert_to_tensor(value, tf.float64)
                + chart["log_abs_determinant"],
                tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
                status,
                metadata,
            )

        initial_started = time.perf_counter()
        initial = initialize_distributed_replica_state(
            initial_state, evaluator=evaluator
        )
        initial_seconds = time.perf_counter() - initial_started
        identity = _worker_identity_gate(initial["evaluation_metadata"])
        if not bool(identity["passed"]):
            raise DistributedReplicaCanaryError(
                f"persistent worker identity gate failed: {identity}"
            )
        physical_values = (
            initial["base_target_log_prob"] - chart["log_abs_determinant"]
        )
        expected_values = tf.constant(
            [
                geometry["representatives"]["plus"]["log_prob"],
                geometry["representatives"]["minus"]["log_prob"],
            ],
            tf.float64,
        )
        value_residual = tf.reduce_max(
            tf.abs(physical_values - expected_values[tf.newaxis, :])
        )
        score_inf_norm = tf.reduce_max(tf.abs(initial["base_score"]))
        if not bool(
            tf.logical_and(value_residual <= 5.0e-7, score_inf_norm <= 5.0e-7).numpy()
        ):
            raise DistributedReplicaCanaryError("initial representative parity failed")

        transition_started = time.perf_counter()
        transition = distributed_replica_exchange_transition(
            state=initial["state"],
            base_target_log_prob=initial["base_target_log_prob"],
            base_score=initial["base_score"],
            identities_at_temperature=initial["identities_at_temperature"],
            inverse_temperatures=BETAS,
            step_sizes=STEPS,
            num_leapfrog_steps=LEAPFROG,
            transition_index=0,
            master_seed=(20260810, 7101),
            evaluator=evaluator,
        )
        transition_seconds = time.perf_counter() - transition_started

        terminal_value, terminal_score, terminal_status, _metadata = evaluator(
            tf.reshape(transition["state"], (ROWS, PARAMETER_DIM)),
            "terminal-independent-check",
        )
        terminal_value = tf.reshape(terminal_value, (len(BETAS), CHAINS))
        terminal_score = tf.reshape(
            terminal_score, (len(BETAS), CHAINS, PARAMETER_DIM)
        )
        terminal_valid = tf.logical_and(
            tf.convert_to_tensor(terminal_status["status_code"], tf.int32) == 0,
            tf.convert_to_tensor(
                terminal_status["valid_pre_regularized_score"], tf.bool
            ),
        )
        cache_value_residual = tf.reduce_max(
            tf.abs(terminal_value - transition["base_target_log_prob"])
        )
        cache_score_residual = tf.reduce_max(
            tf.abs(terminal_score - transition["base_score"])
        )

    swap_matrix = tf.cast(transition["swap_is_accepted_matrix"], tf.int32)
    swap_permutation = tf.logical_and(
        tf.reduce_all(
            tf.reduce_sum(swap_matrix, axis=0)
            == tf.ones((len(BETAS), CHAINS), tf.int32)
        ),
        tf.reduce_all(
            tf.reduce_sum(swap_matrix, axis=1)
            == tf.ones((len(BETAS), CHAINS), tf.int32)
        ),
    )
    invalid_paths = tf.logical_not(transition["hmc_path_valid"])
    invalid_paths_self_rejected = tf.reduce_all(
        tf.logical_not(
            tf.boolean_mask(transition["hmc_is_accepted"], invalid_paths)
        )
    )
    log_accept_valid = tf.reduce_all(
        tf.logical_or(
            tf.math.is_finite(transition["hmc_log_accept_ratio"]),
            tf.logical_and(
                invalid_paths,
                tf.math.is_inf(transition["hmc_log_accept_ratio"])
                & (transition["hmc_log_accept_ratio"] < 0.0),
            ),
        )
    )
    physical = chart["center"] + tf.matmul(
        tf.reshape(transition["state"], (ROWS, PARAMETER_DIM)),
        chart["factor"],
        transpose_b=True,
    )
    physical = tf.reshape(physical, (len(BETAS), CHAINS, PARAMETER_DIM))
    pre_physical = chart["center"] + tf.matmul(
        tf.reshape(transition["pre_swap_state"], (ROWS, PARAMETER_DIM)),
        chart["factor"],
        transpose_b=True,
    )
    pre_physical = tf.reshape(
        pre_physical, (len(BETAS), CHAINS, PARAMETER_DIM)
    )
    initial_physical = tf.repeat(
        chart["source_centers"][tf.newaxis, :, :], len(BETAS), axis=0
    )
    local_sign_changes = (pre_physical[..., 2] < 0.0) != (
        initial_physical[..., 2] < 0.0
    )
    gates = {
        "persistent_worker_identity_passed": bool(identity["passed"]),
        "initial_representative_parity_passed": bool(
            tf.logical_and(value_residual <= 5.0e-7, score_inf_norm <= 5.0e-7).numpy()
        ),
        "terminal_target_status_all_valid": bool(tf.reduce_all(terminal_valid).numpy()),
        "terminal_cached_value_matches_independent_evaluation": bool(
            (cache_value_residual <= 1.0e-9).numpy()
        ),
        "terminal_cached_score_matches_independent_evaluation": bool(
            (cache_score_residual <= 1.0e-8).numpy()
        ),
        "retained_state_and_momenta_finite": bool(
            tf.reduce_all(
                tf.stack(
                    [
                        tf.reduce_all(tf.math.is_finite(transition[name]))
                        for name in (
                            "state",
                            "pre_swap_state",
                            "initial_momentum",
                            "final_momentum",
                        )
                    ]
                )
            ).numpy()
        ),
        "invalid_proposal_paths_self_rejected": bool(
            invalid_paths_self_rejected.numpy()
        ),
        "log_acceptance_finite_or_forced_negative_infinity": bool(
            log_accept_valid.numpy()
        ),
        "swap_matrix_is_permutation": bool(swap_permutation.numpy()),
        "wall_time_within_3600_seconds": time.perf_counter() - started <= CAP_SECONDS,
    }
    passed = all(gates.values())
    receipt_names = (
        "state",
        "base_target_log_prob",
        "base_score",
        "identities_at_temperature",
        "pre_swap_state",
        "proposed_state",
        "hmc_is_accepted",
        "hmc_log_accept_ratio",
        "hmc_path_valid",
        "initial_momentum",
        "final_momentum",
        "wave_valid_counts",
        "swap_is_proposed_adjacent",
        "swap_is_accepted_adjacent",
        "swap_log_accept_ratio_adjacent",
        "swap_is_accepted_matrix",
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.canary.v1",
        "status": (
            "DISTRIBUTED_REPLICA_CANARY_PASSED"
            if passed
            else "DISTRIBUTED_REPLICA_CANARY_FAILED"
        ),
        "role": "distributed_exact_hmc_replica_mechanics_and_timing_canary",
        "configuration": {
            "inverse_temperatures": BETAS,
            "step_sizes": STEPS,
            "num_leapfrog_steps": LEAPFROG,
            "chains": CHAINS,
            "rows": ROWS,
            "workers": WORKERS,
            "rows_per_worker": 1,
            "worker_cpu_ids": WORKER_CPU_IDS,
            "parent_cpu_ids": PARENT_CPU_IDS,
            "jit_compile": True,
            "target_backend": "persistent_cpu_batch_native_tensorflow_pool",
            "sample_wise_loop_or_scalar_fallback": False,
            "training_update": False,
        },
        "gates": gates,
        "worker_identity": identity,
        "initial_seconds": initial_seconds,
        "transition_seconds": transition_seconds,
        "monolithic_cached_seconds_per_transition": timing[
            "descriptive_comparison"
        ]["threads_04_cached_seconds_per_transition"],
        "descriptive_monolithic_over_distributed_ratio": (
            float(
                timing["descriptive_comparison"][
                    "threads_04_cached_seconds_per_transition"
                ]
            )
            / transition_seconds
        ),
        "initial_value_max_abs_residual": value_residual,
        "initial_latent_score_max_abs": score_inf_norm,
        "terminal_cache_value_max_abs_residual": cache_value_residual,
        "terminal_cache_score_max_abs_residual": cache_score_residual,
        "invalid_hmc_path_count": tf.reduce_sum(tf.cast(invalid_paths, tf.int32)),
        "hmc_acceptance_by_temperature": tf.reduce_mean(
            tf.cast(transition["hmc_is_accepted"], tf.float64), axis=1
        ),
        "adjacent_swap_proposals": tf.reduce_sum(
            tf.cast(transition["swap_is_proposed_adjacent"], tf.int32), axis=1
        ),
        "adjacent_swap_acceptances": tf.reduce_sum(
            tf.cast(transition["swap_is_accepted_adjacent"], tf.int32), axis=1
        ),
        "local_hmc_hot_sign_changes": tf.reduce_sum(
            tf.cast(local_sign_changes[1:], tf.int32)
        ),
        "local_hmc_cold_sign_changes": tf.reduce_sum(
            tf.cast(local_sign_changes[0], tf.int32)
        ),
        "evaluation_waves": evaluation_rows,
        "trace_receipts": {
            name: _write_tensor(
                OUTPUT_ROOT / f"transition-{name}.tftensor",
                transition[name],
                tf,
            )
            for name in receipt_names
        },
        "physical_state_receipt": _write_tensor(
            OUTPUT_ROOT / "transition-physical-state.tftensor", physical, tf
        ),
        "pre_swap_physical_state_receipt": _write_tensor(
            OUTPUT_ROOT / "transition-pre-swap-physical-state.tftensor",
            pre_physical,
            tf,
        ),
        "bindings": {
            "target_signature": TARGET_SIGNATURE,
            "adapter_signature": ADAPTER_SIGNATURE,
            "geometry_sha256": _sha(GEOMETRY),
            "timing_sha256": _sha(TIMING),
            "failed_r2_sha256": _sha(FAILED_R2),
        },
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ("git", "status", "--porcelain"),
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
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
                "helper": _sha(HELPER),
                "pool_helper": _sha(POOL_HELPER),
            },
        },
        "nonclaims": (
            "one transition is mechanics and timing evidence only",
            "no round-trip, convergence, posterior, weight, or predictive claim",
            "one timing observation does not statistically prove a speed ranking",
            "twelve rank-two size-one target shards are sampling-only and perform no training update",
        ),
    }
    if not passed:
        raise DistributedReplicaCanaryError(f"distributed canary gates failed: {gates}")
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": payload["status"],
            "completed_waves": evaluation_index,
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    _append_log(f"completed distributed replica canary: {payload['status']}")
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run_canary()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.failure.v1",
            "status": "DISTRIBUTED_REPLICA_CANARY_HARNESS_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_time_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        _append_log(f"failed distributed replica canary: {type(error).__name__}: {error}")
        raise
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
