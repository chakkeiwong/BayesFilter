#!/usr/bin/env python3
"""Run the bounded four-chain distributed replica timing/travel checkpoint."""

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
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
RESULT = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-result-2026-08-10.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
REPLICA_HELPER = Path("bayesfilter/testing/replica_exchange_tf.py")
POOL_HELPER = Path("bayesfilter/inference/tf_batch_value_score_pool.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json"
)
CANARY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r3-distributed-canary/canary.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r4-four-chain-checkpoint"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "checkpoint.json"
LOG = OUTPUT_ROOT / "run.log"

PARAMETER_DIM = 4
BETAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
STEPS = tuple(0.05 / math.sqrt(beta) for beta in BETAS)
LEAPFROG = 8
CHAINS = 4
ROWS = len(BETAS) * CHAINS
WORKERS = ROWS
TRANSITIONS = 25
CHUNK_SIZE = 5
WORKER_CPU_IDS = tuple(range(32, 32 + WORKERS))
PARENT_CPU_IDS = tuple(range(32, 64))
CAP_SECONDS = 900.0
MATERIAL_MINIMUM_TRANSITIONS = 300 + 1000
MATERIAL_CAP_SECONDS = 20000.0
PROJECTION_MARGIN = 1.5
ACCEPTANCE_LOWER = 0.35
ACCEPTANCE_UPPER = 0.99
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
CANARY_SHA256 = "bfcbb5840622e761e052b5dfe398c6ae194570765294a4f1d159091b1569d471"


class DistributedCheckpointError(RuntimeError):
    """Raised when the four-chain checkpoint cannot preserve valid evidence."""


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
        raise DistributedCheckpointError(f"refusing to overwrite: {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
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
        raise DistributedCheckpointError(f"refusing to overwrite tensor: {path}")
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
        [geometry["representatives"][label]["position"] for label in labels], tf.float64
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
    pooled = tf.reduce_mean(covariances, axis=0) + tf.einsum(
        "ni,nj->ij", displacement, displacement
    ) / 2.0
    eigenvalues, eigenvectors = tf.linalg.eigh(pooled)
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
        dimension=PARAMETER_DIM,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(1,),
        batch_per_worker=1,
        worker_cpu_ids=WORKER_CPU_IDS,
        timeout_seconds=900.0,
    )


def _worker_identity(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    workers = tuple(metadata["startup_worker_metadata"])
    return {
        "worker_count": len(workers),
        "worker_pids": tuple(int(row["pid"]) for row in workers),
        "assigned_cpu_ids": tuple(sorted(int(row["assigned_cpu"]) for row in workers)),
        "target_signatures": tuple(sorted({str(row["target_signature"]) for row in workers})),
        "adapter_signatures": tuple(sorted({str(row["adapter_signature"]) for row in workers})),
        "status_jit_compile_values": tuple(sorted({bool(row["status_jit_compile"]) for row in workers})),
    }


def _identity_passed(identity: Mapping[str, Any]) -> bool:
    return bool(
        int(identity["worker_count"]) == WORKERS
        and tuple(identity["assigned_cpu_ids"]) == WORKER_CPU_IDS
        and tuple(identity["target_signatures"]) == (TARGET_SIGNATURE,)
        and tuple(identity["adapter_signatures"]) == (ADAPTER_SIGNATURE,)
        and tuple(identity["status_jit_compile_values"]) == (True,)
    )


def run_checkpoint() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise DistributedCheckpointError("refusing to overwrite checkpoint")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise DistributedCheckpointError("parent CPU affinity mismatch")
    if _sha(GEOMETRY) != GEOMETRY_SHA256 or _sha(CANARY) != CANARY_SHA256:
        raise DistributedCheckpointError("bound geometry/canary identity mismatch")
    canary = json.loads(_abs(CANARY).read_text(encoding="utf-8"))
    if canary.get("status") != "DISTRIBUTED_REPLICA_CANARY_PASSED":
        raise DistributedCheckpointError("distributed canary did not pass")

    _write_json(
        PROGRESS,
        {
            "status": "FOUR_CHAIN_CHECKPOINT_STARTING",
            "completed_transitions": 0,
            "total_transitions": TRANSITIONS,
        },
        overwrite=True,
    )
    _append_log("starting four-chain distributed checkpoint")
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
    )
    from bayesfilter.testing.replica_exchange_tf import replica_travel_diagnostics

    if tf.config.list_physical_devices("GPU"):
        raise DistributedCheckpointError("CPU-only checkpoint found visible GPU")
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = _chart(tf, geometry)
    chain_centers = tf.gather(chart["latent_centers"], (0, 1, 0, 1))
    initial_state = tf.repeat(chain_centers[tf.newaxis, :, :], len(BETAS), axis=0)
    evaluation_count = 0
    evaluation_wall = []
    cache_check_wall = []
    transition_wall = []
    chunk_receipts = []
    state_rows = []
    pre_rows = []
    identity_rows = []
    hmc_accept_rows = []
    hmc_log_accept_rows = []
    hmc_valid_rows = []
    swap_proposed_rows = []
    swap_accepted_rows = []
    swap_matrix_rows = []
    terminal_cache_value_residuals = []
    terminal_cache_score_residuals = []

    with TFBatchValueScorePool(_pool_config()) as pool:
        def evaluator(rows: tf.Tensor, request_id: str):
            nonlocal evaluation_count
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
            evaluation_count += 1
            evaluation_wall.append(elapsed)
            return (
                tf.convert_to_tensor(value, tf.float64) + chart["log_abs_determinant"],
                tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
                status,
                metadata,
            )

        initial = initialize_distributed_replica_state(initial_state, evaluator=evaluator)
        identity = _worker_identity(initial["evaluation_metadata"])
        if not _identity_passed(identity):
            raise DistributedCheckpointError(f"worker identity failed: {identity}")
        current = {
            name: initial[name]
            for name in (
                "state",
                "base_target_log_prob",
                "base_score",
                "identities_at_temperature",
            )
        }
        for transition_index in range(TRANSITIONS):
            transition_started = time.perf_counter()
            transition = distributed_replica_exchange_transition(
                **current,
                inverse_temperatures=BETAS,
                step_sizes=STEPS,
                num_leapfrog_steps=LEAPFROG,
                transition_index=transition_index,
                master_seed=(20260810, 7201),
                evaluator=evaluator,
            )
            transition_wall.append(time.perf_counter() - transition_started)
            state_rows.append(transition["state"])
            pre_rows.append(transition["pre_swap_state"])
            identity_rows.append(transition["identities_at_temperature"])
            hmc_accept_rows.append(transition["hmc_is_accepted"])
            hmc_log_accept_rows.append(transition["hmc_log_accept_ratio"])
            hmc_valid_rows.append(transition["hmc_path_valid"])
            swap_proposed_rows.append(transition["swap_is_proposed_adjacent"])
            swap_accepted_rows.append(transition["swap_is_accepted_adjacent"])
            swap_matrix_rows.append(transition["swap_is_accepted_matrix"])
            current = {
                name: transition[name]
                for name in (
                    "state",
                    "base_target_log_prob",
                    "base_score",
                    "identities_at_temperature",
                )
            }

            if (transition_index + 1) % CHUNK_SIZE == 0:
                cache_value, cache_score, cache_status, _metadata = evaluator(
                    tf.reshape(current["state"], (ROWS, PARAMETER_DIM)),
                    f"chunk-{transition_index // CHUNK_SIZE:02d}-cache-check",
                )
                cache_check_wall.append(evaluation_wall[-1])
                cache_value = tf.reshape(cache_value, (len(BETAS), CHAINS))
                cache_score = tf.reshape(
                    cache_score, (len(BETAS), CHAINS, PARAMETER_DIM)
                )
                valid = tf.logical_and(
                    tf.convert_to_tensor(cache_status["status_code"], tf.int32) == 0,
                    tf.convert_to_tensor(
                        cache_status["valid_pre_regularized_score"], tf.bool
                    ),
                )
                if not bool(tf.reduce_all(valid).numpy()):
                    raise DistributedCheckpointError("chunk terminal target status invalid")
                value_residual = tf.reduce_max(
                    tf.abs(cache_value - current["base_target_log_prob"])
                )
                score_residual = tf.reduce_max(
                    tf.abs(cache_score - current["base_score"])
                )
                terminal_cache_value_residuals.append(value_residual)
                terminal_cache_score_residuals.append(score_residual)
                if not bool(
                    tf.logical_and(
                        value_residual <= 1.0e-9,
                        score_residual <= 1.0e-8,
                    ).numpy()
                ):
                    raise DistributedCheckpointError("chunk terminal cache parity failed")
                chunk_index = transition_index // CHUNK_SIZE
                start = transition_index + 1 - CHUNK_SIZE
                chunk_tensors = {
                    "state": tf.stack(state_rows[start : transition_index + 1]),
                    "pre_swap_state": tf.stack(pre_rows[start : transition_index + 1]),
                    "identities": tf.stack(identity_rows[start : transition_index + 1]),
                    "hmc_is_accepted": tf.stack(
                        hmc_accept_rows[start : transition_index + 1]
                    ),
                    "hmc_log_accept_ratio": tf.stack(
                        hmc_log_accept_rows[start : transition_index + 1]
                    ),
                    "hmc_path_valid": tf.stack(
                        hmc_valid_rows[start : transition_index + 1]
                    ),
                    "swap_is_proposed_adjacent": tf.stack(
                        swap_proposed_rows[start : transition_index + 1]
                    ),
                    "swap_is_accepted_adjacent": tf.stack(
                        swap_accepted_rows[start : transition_index + 1]
                    ),
                    "swap_is_accepted_matrix": tf.stack(
                        swap_matrix_rows[start : transition_index + 1]
                    ),
                }
                receipts = {
                    name: _write_tensor(
                        OUTPUT_ROOT / f"chunk-{chunk_index:02d}-{name}.tftensor",
                        value,
                        tf,
                    )
                    for name, value in chunk_tensors.items()
                }
                chunk_payload = {
                    "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.checkpoint_chunk.v1",
                    "chunk_index": chunk_index,
                    "transition_start": start,
                    "transition_stop_exclusive": transition_index + 1,
                    "cache_value_max_abs_residual": value_residual,
                    "cache_score_max_abs_residual": score_residual,
                    "receipts": receipts,
                }
                chunk_path = OUTPUT_ROOT / f"chunk-{chunk_index:02d}.json"
                _write_json(chunk_path, chunk_payload)
                chunk_receipts.append(
                    {"path": chunk_path.as_posix(), "sha256": _sha(chunk_path)}
                )
                _write_json(
                    PROGRESS,
                    {
                        "status": "FOUR_CHAIN_CHECKPOINT_RUNNING",
                        "completed_transitions": transition_index + 1,
                        "total_transitions": TRANSITIONS,
                        "completed_chunks": chunk_index + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "last_chunk_sha256": _sha(chunk_path),
                    },
                    overwrite=True,
                )
                _append_log(
                    f"completed chunk {chunk_index} through transition {transition_index}"
                )
            if time.perf_counter() - started > CAP_SECONDS:
                raise DistributedCheckpointError("checkpoint wall cap exceeded")

    states = tf.stack(state_rows)
    pre_states = tf.stack(pre_rows)
    identities = tf.stack(identity_rows)
    hmc_accepted = tf.stack(hmc_accept_rows)
    hmc_log_accept = tf.stack(hmc_log_accept_rows)
    hmc_valid = tf.stack(hmc_valid_rows)
    swap_proposed = tf.stack(swap_proposed_rows)
    swap_accepted = tf.stack(swap_accepted_rows)
    swap_matrices = tf.stack(swap_matrix_rows)
    travel = replica_travel_diagnostics(identities)
    matrix_int = tf.cast(swap_matrices, tf.int32)
    swap_permutations = tf.logical_and(
        tf.reduce_all(
            tf.reduce_sum(matrix_int, axis=2)
            == tf.ones((TRANSITIONS, len(BETAS), CHAINS), tf.int32)
        ),
        tf.reduce_all(
            tf.reduce_sum(matrix_int, axis=1)
            == tf.ones((TRANSITIONS, len(BETAS), CHAINS), tf.int32)
        ),
    )
    invalid = tf.logical_not(hmc_valid)
    invalid_self_rejected = tf.reduce_all(
        tf.logical_not(tf.boolean_mask(hmc_accepted, invalid))
    )
    log_accept_valid = tf.reduce_all(
        tf.logical_or(
            tf.math.is_finite(hmc_log_accept),
            tf.logical_and(
                invalid,
                tf.math.is_inf(hmc_log_accept) & (hmc_log_accept < 0.0),
            ),
        )
    )
    acceptance_probability = tf.exp(
        tf.minimum(hmc_log_accept, tf.constant(0.0, tf.float64))
    )
    mean_acceptance_probability = tf.reduce_mean(acceptance_probability, axis=0)
    acceptance_screen = tf.reduce_all(
        (mean_acceptance_probability >= ACCEPTANCE_LOWER)
        & (mean_acceptance_probability <= ACCEPTANCE_UPPER)
    )
    proposed_counts = tf.reduce_sum(tf.cast(swap_proposed, tf.int32), axis=(0, 2))
    accepted_counts = tf.reduce_sum(tf.cast(swap_accepted, tf.int32), axis=(0, 2))
    every_adjacent_communicates = tf.reduce_all(accepted_counts > 0)

    flat_states = tf.reshape(states, (-1, PARAMETER_DIM))
    physical = chart["center"] + tf.matmul(
        flat_states, chart["factor"], transpose_b=True
    )
    physical = tf.reshape(
        physical, (TRANSITIONS, len(BETAS), CHAINS, PARAMETER_DIM)
    )
    pre_physical = chart["center"] + tf.matmul(
        tf.reshape(pre_states, (-1, PARAMETER_DIM)),
        chart["factor"],
        transpose_b=True,
    )
    pre_physical = tf.reshape(pre_physical, tf.shape(physical))
    initial_physical = tf.repeat(
        tf.gather(chart["source_centers"], (0, 1, 0, 1))[tf.newaxis, :, :],
        len(BETAS),
        axis=0,
    )
    previous = tf.concat((initial_physical[tf.newaxis], physical[:-1]), axis=0)
    local_sign_changes = (pre_physical[..., 2] < 0.0) != (
        previous[..., 2] < 0.0
    )
    cold_signs = physical[:, 0, :, 2] < 0.0
    cold_sign_transitions = tf.reduce_sum(
        tf.cast(cold_signs[1:] != cold_signs[:-1], tf.int32)
    )
    transition_seconds_mean = tf.reduce_mean(tf.constant(transition_wall, tf.float64))
    checkpoint_per_transition = (
        sum(transition_wall) + sum(cache_check_wall)
    ) / TRANSITIONS
    projected_material_seconds = (
        checkpoint_per_transition
        * MATERIAL_MINIMUM_TRANSITIONS
        * PROJECTION_MARGIN
    )
    projection_passed = projected_material_seconds <= MATERIAL_CAP_SECONDS
    hard_gates = {
        "persistent_worker_identity_passed": _identity_passed(identity),
        "all_retained_states_finite": bool(tf.reduce_all(tf.math.is_finite(states)).numpy()),
        "all_invalid_paths_self_rejected": bool(invalid_self_rejected.numpy()),
        "log_acceptance_finite_or_invalid_path_negative_infinity": bool(
            log_accept_valid.numpy()
        ),
        "every_swap_matrix_is_permutation": bool(swap_permutations.numpy()),
        "all_chunk_cache_value_checks_passed": bool(
            tf.reduce_max(tf.stack(terminal_cache_value_residuals)) <= 1.0e-9
        ),
        "all_chunk_cache_score_checks_passed": bool(
            tf.reduce_max(tf.stack(terminal_cache_score_residuals)) <= 1.0e-8
        ),
        "wall_time_within_900_seconds": time.perf_counter() - started <= CAP_SECONDS,
    }
    nomination = {
        "material_projection_within_20000_seconds": projection_passed,
        "every_adjacent_pair_accepted_at_least_one_swap": bool(
            every_adjacent_communicates.numpy()
        ),
        "all_temperature_chain_mean_acceptance_probabilities_in_0.35_0.99": bool(
            acceptance_screen.numpy()
        ),
    }
    passed = all(hard_gates.values())
    nominated = passed and all(nomination.values())
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.checkpoint.v1",
        "status": (
            "FOUR_CHAIN_CHECKPOINT_NOMINATED"
            if nominated
            else (
                "FOUR_CHAIN_CHECKPOINT_PASSED_NOT_NOMINATED"
                if passed
                else "FOUR_CHAIN_CHECKPOINT_FAILED"
            )
        ),
        "role": "four_chain_cost_travel_and_material_nomination_checkpoint",
        "configuration": {
            "inverse_temperatures": BETAS,
            "step_sizes": STEPS,
            "num_leapfrog_steps": LEAPFROG,
            "chains": CHAINS,
            "transitions": TRANSITIONS,
            "chunk_size": CHUNK_SIZE,
            "workers": WORKERS,
            "rows_per_worker": 1,
            "worker_cpu_ids": WORKER_CPU_IDS,
            "master_seed": (20260810, 7201),
            "jit_compile": True,
            "training_update": False,
        },
        "hard_gates": hard_gates,
        "nomination_screen": nomination,
        "worker_identity": identity,
        "mean_transition_seconds": transition_seconds_mean,
        "checkpoint_cost_per_transition_seconds": checkpoint_per_transition,
        "projected_minimum_material_seconds_with_50pct_margin": projected_material_seconds,
        "mean_acceptance_probability_by_temperature_chain": mean_acceptance_probability,
        "realized_acceptance_fraction_by_temperature": tf.reduce_mean(
            tf.cast(hmc_accepted, tf.float64), axis=(0, 2)
        ),
        "adjacent_swap_proposals": proposed_counts,
        "adjacent_swap_acceptances": accepted_counts,
        "round_trip_returns_by_chain_identity": travel["round_trip_returns"],
        "completed_round_trips": tf.reduce_sum(travel["round_trip_returns"]),
        "visited_hot_by_chain_identity": travel["visited_hot"],
        "local_hmc_hot_sign_changes": tf.reduce_sum(
            tf.cast(local_sign_changes[:, 1:], tf.int32)
        ),
        "local_hmc_cold_sign_changes": tf.reduce_sum(
            tf.cast(local_sign_changes[:, 0], tf.int32)
        ),
        "post_swap_cold_sign_transitions": cold_sign_transitions,
        "cold_negative_sign_fraction": tf.reduce_mean(
            tf.cast(cold_signs, tf.float64)
        ),
        "invalid_hmc_path_count": tf.reduce_sum(tf.cast(invalid, tf.int32)),
        "evaluation_wave_seconds": evaluation_wall,
        "cache_check_seconds": cache_check_wall,
        "transition_seconds": transition_wall,
        "chunk_manifests": chunk_receipts,
        "aggregate_receipts": {
            "physical_states": _write_tensor(
                OUTPUT_ROOT / "physical-states.tftensor", physical, tf
            ),
            "pre_swap_physical_states": _write_tensor(
                OUTPUT_ROOT / "pre-swap-physical-states.tftensor", pre_physical, tf
            ),
            "identities": _write_tensor(
                OUTPUT_ROOT / "identities.tftensor", identities, tf
            ),
        },
        "bindings": {
            "target_signature": TARGET_SIGNATURE,
            "adapter_signature": ADAPTER_SIGNATURE,
            "geometry_sha256": _sha(GEOMETRY),
            "distributed_canary_sha256": _sha(CANARY),
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
                "helper": _sha(HELPER),
                "replica_helper": _sha(REPLICA_HELPER),
                "pool_helper": _sha(POOL_HELPER),
            },
        },
        "nonclaims": (
            "25-transition cost/travel checkpoint only",
            "round trips, signs, occupancy, and acceptance are explanatory at this length",
            "no convergence, posterior, mass, predictive, superiority, or default claim",
        ),
    }
    if not passed:
        raise DistributedCheckpointError(f"checkpoint hard gates failed: {hard_gates}")
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": payload["status"],
            "completed_transitions": TRANSITIONS,
            "total_transitions": TRANSITIONS,
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    _append_log(f"completed checkpoint: {payload['status']}")
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run_checkpoint()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_distributed_replica.checkpoint_failure.v1",
            "status": "FOUR_CHAIN_CHECKPOINT_HARNESS_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_time_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        _append_log(f"failed checkpoint: {type(error).__name__}: {error}")
        raise
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
