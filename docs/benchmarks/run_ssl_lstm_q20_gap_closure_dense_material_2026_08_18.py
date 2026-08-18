#!/usr/bin/env python3
"""Run the bounded claim-bearing SSL-LSTM physical replica campaign."""

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
    "docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md"
)
RESULT = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-result-2026-08-18.md"
)
RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_q20_gap_closure_dense_material_2026_08_18.py"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
HELPER = Path("bayesfilter/testing/distributed_replica_exchange_tf.py")
REPLICA_HELPER = Path("bayesfilter/testing/replica_exchange_tf.py")
POOL_HELPER = Path("bayesfilter/inference/tf_batch_value_score_pool.py")
CONVERGENCE_HELPER = Path("bayesfilter/inference/hmc_convergence.py")
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
CANARY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r1-dense-mass-step-0p35/canary.json"
)
MATERIALITY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r7-12x2-numerical-materiality-canary/canary.json"
)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/physical/"
    "r2-dense-mass-material-retry"
)
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "material.json"
LOG = OUTPUT_ROOT / "run.log"

PARAMETER_DIM = 4
WORKERS = 24
ROWS_PER_WORKER = 1
ROWS = 24
WORKER_CPU_IDS = tuple(range(32, 56))
PARENT_CPU_IDS = tuple(range(32, 64))
CHUNK_SIZE = 10
WARMUP_MIN = 300
WARMUP_MAX = 500
WARMUP_INCREMENT = 50
WARMUP_WINDOW = 300
WARMUP_RHAT_MAX = 1.05
RETAINED_MIN = 1000
RETAINED_MAX = 1500
RETAINED_INCREMENT = 250
RETAINED_RHAT_MAX = 1.01
RETAINED_BULK_ESS_MIN = 1000.0
RETAINED_TAIL_ESS_MIN = 400.0
ACCEPTANCE_LOWER = 0.35
ACCEPTANCE_UPPER = 0.99
MIN_ROUND_TRIPS_PER_CHAIN = 1
MASTER_SEED = (20260818, 9101)
HARD_WALL_CAP_SECONDS = 28800.0
FINALIZATION_RESERVE_SECONDS = 300.0
RUNNER_TRANSITION_DEADLINE_SECONDS = (
    HARD_WALL_CAP_SECONDS - FINALIZATION_RESERVE_SECONDS
)
SMC_NEGATIVE_MASS_INTERVAL = (0.405731, 0.536018)
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
GEOMETRY_SHA256 = "dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb"
CANARY_SHA256 = "d7d59ff6ea84a7c31206e16a5e5db7dde8fcaaf2d069a52e91e4f03ec6427e04"
MATERIALITY_SHA256 = "5de1e5d217abd9ae293aff81356955c799ed6328e6a66670b019220f6d27aad2"
WARMUP_MILESTONES = tuple(
    range(WARMUP_MIN, WARMUP_MAX + 1, WARMUP_INCREMENT)
)
RETAINED_MILESTONES = tuple(
    range(RETAINED_MIN, RETAINED_MAX + 1, RETAINED_INCREMENT)
)


class MaterialReplicaError(RuntimeError):
    """Raised when the material campaign harness or a hard invariant fails."""


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
        raise MaterialReplicaError(f"refusing to overwrite {path}")
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
        raise MaterialReplicaError(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)
    digest = hashlib.sha256(encoded).hexdigest()
    if _sha(path) != digest:
        raise MaterialReplicaError(f"tensor hash verification failed: {path}")
    return {
        "path": path.as_posix(),
        "sha256": digest,
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load_checkpoint_runner() -> Any:
    name = "physical_checkpoint_support_for_material_replica"
    spec = importlib.util.spec_from_file_location(name, _abs(CHECKPOINT_RUNNER))
    if spec is None or spec.loader is None:
        raise MaterialReplicaError("cannot load checkpoint support")
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
        dimension=PARAMETER_DIM,
        worker_count=WORKERS,
        cores_per_worker=1,
        batch_sizes=(ROWS_PER_WORKER,),
        batch_per_worker=ROWS_PER_WORKER,
        worker_cpu_ids=WORKER_CPU_IDS,
        timeout_seconds=900.0,
    )


def _next_milestone(completed: int, milestones: tuple[int, ...]) -> int:
    count = int(completed)
    for milestone in milestones:
        if count < int(milestone):
            return int(milestone)
    return int(milestones[-1])


def _window_round_trips(tf: Any, identities: Any) -> Mapping[str, Any]:
    """Count cold-hot-cold returns entirely inside one declared window."""

    values = tf.convert_to_tensor(identities, tf.int32)
    steps, replicas, chains = (int(item) for item in values.shape)
    identity_ids = tf.range(replicas, dtype=tf.int32)
    positions = tf.argmax(
        tf.cast(
            values[:, :, :, tf.newaxis]
            == identity_ids[tf.newaxis, tf.newaxis, tf.newaxis, :],
            tf.int32,
        ),
        axis=1,
        output_type=tf.int32,
    )
    at_cold = positions == 0
    at_hot = positions == replicas - 1

    def update(
        state: tuple[Any, Any], endpoints: tuple[Any, Any]
    ) -> tuple[Any, Any]:
        phase, count = state
        cold, hot = endpoints
        completed = tf.logical_and(phase == 2, cold)
        count = count + tf.cast(completed, tf.int32)
        phase = tf.where(completed, tf.ones_like(phase), phase)
        phase = tf.where(
            tf.logical_and(phase == 0, cold), tf.ones_like(phase), phase
        )
        phase = tf.where(
            tf.logical_and(phase == 1, hot), tf.fill(tf.shape(phase), 2), phase
        )
        return phase, count

    _phase, cumulative = tf.scan(
        update,
        (at_cold, at_hot),
        initializer=(
            tf.zeros((chains, replicas), tf.int32),
            tf.zeros((chains, replicas), tf.int32),
        ),
    )
    returns = cumulative[-1]
    by_chain = tf.reduce_sum(returns, axis=1)
    return {
        "window_steps": steps,
        "round_trip_returns_by_chain_identity": returns,
        "round_trip_returns_by_chain": by_chain,
        "completed_round_trips": tf.reduce_sum(returns),
        "each_chain_has_required_round_trip": tf.reduce_all(
            by_chain >= MIN_ROUND_TRIPS_PER_CHAIN
        ),
    }


def _hot_forgetting(
    tf: Any,
    pre_physical: Any,
    post_physical: Any,
    initial_physical: Any,
) -> Mapping[str, Any]:
    pre = tf.convert_to_tensor(pre_physical, tf.float64)
    post = tf.convert_to_tensor(post_physical, tf.float64)
    initial = tf.convert_to_tensor(initial_physical, tf.float64)
    previous = tf.concat((initial[tf.newaxis, ...], post[:-1]), axis=0)
    hot_pre_sign = pre[:, -1, :, 2] < 0.0
    hot_previous_sign = previous[:, -1, :, 2] < 0.0
    changes = tf.reduce_sum(
        tf.cast(hot_pre_sign != hot_previous_sign, tf.int32), axis=0
    )
    observed_negative = tf.reduce_any(
        tf.concat((hot_previous_sign[:1], hot_pre_sign), axis=0), axis=0
    )
    observed_positive = tf.reduce_any(
        tf.logical_not(
            tf.concat((hot_previous_sign[:1], hot_pre_sign), axis=0)
        ),
        axis=0,
    )
    passed_by_chain = (changes > 0) & observed_negative & observed_positive
    return {
        "local_hmc_hot_sign_changes_by_chain": changes,
        "hot_negative_seen_by_chain": observed_negative,
        "hot_positive_seen_by_chain": observed_positive,
        "passed_by_chain": passed_by_chain,
        "all_chains_passed": tf.reduce_all(passed_by_chain),
    }


def _acceptance_summary(tf: Any, log_acceptance: Any) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(log_acceptance, tf.float64)
    probability = tf.exp(tf.minimum(values, tf.constant(0.0, tf.float64)))
    mean = tf.reduce_mean(probability, axis=0)
    in_band = (mean >= ACCEPTANCE_LOWER) & (mean <= ACCEPTANCE_UPPER)
    return {
        "mean_probability_by_temperature_chain": mean,
        "all_temperature_chain_means_in_band": tf.reduce_all(in_band),
        "lower": ACCEPTANCE_LOWER,
        "upper": ACCEPTANCE_UPPER,
    }


def _diagnose_warmup(tf: Any, physical: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary

    window = tf.convert_to_tensor(physical, tf.float64)[-WARMUP_WINDOW:, 0, :, :]
    report = rank_normalized_split_rhat_summary(
        window, rhat_max=WARMUP_RHAT_MAX
    )
    return {
        "role": "discarded_warmup_recent_window_readiness",
        "window_draws_per_chain": WARMUP_WINDOW,
        "physical": report,
        "passed": bool(report["passed"]),
    }


def _diagnose_retained(tf: Any, physical: Any) -> Mapping[str, Any]:
    from bayesfilter.inference.hmc_convergence import (
        RankNormalizedHMCThresholds,
        rank_normalized_hmc_diagnostics,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES

    cold = tf.convert_to_tensor(physical, tf.float64)[:, 0, :, :]
    report = rank_normalized_hmc_diagnostics(
        cold,
        parameter_names=FREE_NAMES,
        thresholds=RankNormalizedHMCThresholds(
            rhat_max=RETAINED_RHAT_MAX,
            bulk_ess_min=RETAINED_BULK_ESS_MIN,
            tail_ess_min=RETAINED_TAIL_ESS_MIN,
        ),
    )
    return {
        "role": "retained_cold_physical_rank_normalized_admission",
        "physical": report,
        "passed": bool(report["passed"]),
    }


def run_material() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise MaterialReplicaError("refusing to overwrite material result")
    if tuple(sorted(os.sched_getaffinity(0))) != PARENT_CPU_IDS:
        raise MaterialReplicaError("parent CPU affinity mismatch")
    bindings = {
        "geometry_sha256": _sha(GEOMETRY),
        "canary_sha256": _sha(CANARY),
        "materiality_sha256": _sha(MATERIALITY),
    }
    if bindings != {
        "geometry_sha256": GEOMETRY_SHA256,
        "canary_sha256": CANARY_SHA256,
        "materiality_sha256": MATERIALITY_SHA256,
    }:
        raise MaterialReplicaError("bound evidence identity mismatch")
    canary = json.loads(_abs(CANARY).read_text(encoding="utf-8"))
    materiality = json.loads(_abs(MATERIALITY).read_text(encoding="utf-8"))
    if canary.get("status") != "HOT_ENDPOINT_CANARY_PASSED":
        raise MaterialReplicaError("bound dense-mass canary status changed")
    canary_hard_gates = canary["hard_gates"]
    if (
        not canary.get("selection_passed")
        or int(canary_hard_gates.get("invalid_path_count", -1)) != 0
        or not all(
            bool(canary_hard_gates.get(name, False))
            for name in (
                "invalid_paths_self_rejected",
                "log_acceptance_finite_or_invalid_negative_infinity",
                "wall_time_within_cap",
                "worker_identity_passed",
            )
        )
    ):
        raise MaterialReplicaError("bound dense-mass canary gate failed")
    if not materiality["gates"]["numerical_materiality_passed"]:
        raise MaterialReplicaError("numerical materiality canary did not pass")
    launch_git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    launch_git_dirty = bool(
        subprocess.run(
            ("git", "status", "--porcelain"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    launch_source_sha256 = {
        "plan": _sha(PLAN),
        "runner": _sha(RUNNER),
        "checkpoint_runner": _sha(CHECKPOINT_RUNNER),
        "helper": _sha(HELPER),
        "replica_helper": _sha(REPLICA_HELPER),
        "pool_helper": _sha(POOL_HELPER),
        "convergence_helper": _sha(CONVERGENCE_HELPER),
    }

    _write_json(
        PROGRESS,
        {
            "status": "MATERIAL_REPLICA_STARTING",
            "phase": "warmup",
            "completed_transitions": 0,
            "hard_wall_cap_seconds": HARD_WALL_CAP_SECONDS,
        },
        overwrite=True,
    )
    _append_log("starting fresh 24x1 dense-mass material replica campaign")

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise MaterialReplicaError("CPU-only material run found visible GPU")

    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool
    from bayesfilter.testing.distributed_replica_exchange_tf import (
        distributed_replica_exchange_transition,
        initialize_distributed_replica_state,
    )

    support = _load_checkpoint_runner()
    betas = tuple(float(value) for value in canary["configuration"]["inverse_temperatures"])
    step_sizes = tuple(float(value) for value in canary["configuration"]["step_sizes"])
    leapfrog = int(canary["configuration"]["num_leapfrog_steps"])
    mass_matrix = tf.constant(canary["configuration"]["mass_matrix"], tf.float64)
    if len(betas) != 6 or len(step_sizes) != 6 or leapfrog != 8:
        raise MaterialReplicaError("dense-mass canary kernel configuration drift")
    if canary["configuration"].get("mass_policy") != "mean_two_checked_mapped_local_precisions":
        raise MaterialReplicaError("dense-mass canary mass policy drift")
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = support._chart(tf, geometry)
    chain_centers = tf.gather(chart["latent_centers"], (0, 1, 0, 1))
    initial_state = tf.repeat(chain_centers[tf.newaxis, :, :], len(betas), axis=0)
    initial_physical = chart["center"] + tf.matmul(
        tf.reshape(initial_state, (ROWS, PARAMETER_DIM)),
        chart["factor"],
        transpose_b=True,
    )
    initial_physical = tf.reshape(
        initial_physical, (len(betas), 4, PARAMETER_DIM)
    )

    state_rows: list[Any] = []
    pre_rows: list[Any] = []
    identity_rows: list[Any] = []
    hmc_accept_rows: list[Any] = []
    hmc_log_accept_rows: list[Any] = []
    hmc_valid_rows: list[Any] = []
    swap_proposed_rows: list[Any] = []
    swap_accepted_rows: list[Any] = []
    swap_matrix_rows: list[Any] = []
    transition_seconds: list[float] = []
    cache_seconds: list[float] = []
    evaluation_seconds: list[float] = []
    chunk_receipts: list[Mapping[str, Any]] = []
    warmup_checks: list[Mapping[str, Any]] = []
    retained_checks: list[Mapping[str, Any]] = []
    hard_gate_failures: list[str] = []
    warmup_complete = False
    warmup_count = 0
    retained_count = 0
    terminal_status = "MATERIAL_REPLICA_RUNNING"

    with TFBatchValueScorePool(_pool_config()) as pool:
        def evaluator(rows: Any, request_id: str):
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
            evaluation_seconds.append(time.perf_counter() - wave_started)
            return (
                tf.convert_to_tensor(value, tf.float64)
                + chart["log_abs_determinant"],
                tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
                status,
                metadata,
            )

        initialized = initialize_distributed_replica_state(
            initial_state, evaluator=evaluator
        )
        identity = support._worker_identity(initialized["evaluation_metadata"])
        expected_identity = {
            "worker_count": WORKERS,
            "assigned_cpu_ids": WORKER_CPU_IDS,
            "target_signatures": (TARGET_SIGNATURE,),
            "adapter_signatures": (ADAPTER_SIGNATURE,),
            "status_jit_compile_values": (True,),
        }
        for key, expected in expected_identity.items():
            realized = identity[key]
            mismatch = (
                tuple(realized) != tuple(expected)
                if isinstance(expected, tuple)
                else realized != expected
            )
            if mismatch:
                raise MaterialReplicaError(f"worker identity mismatch: {key}")
        current = {
            name: initialized[name]
            for name in (
                "state",
                "base_target_log_prob",
                "base_score",
                "identities_at_temperature",
            )
        }

        while True:
            elapsed = time.perf_counter() - started
            if elapsed >= RUNNER_TRANSITION_DEADLINE_SECONDS:
                terminal_status = "MATERIAL_REPLICA_WALL_CAP_BEFORE_ADMISSION"
                break
            if not warmup_complete:
                target_total = _next_milestone(
                    warmup_count, WARMUP_MILESTONES
                )
                phase = "warmup"
            else:
                target_total = warmup_count + _next_milestone(
                    retained_count, RETAINED_MILESTONES
                )
                phase = "retained"
            chunk_goal = min(CHUNK_SIZE, target_total - len(state_rows))
            if chunk_goal <= 0:
                raise MaterialReplicaError("nonpositive material chunk goal")
            chunk_start = len(state_rows)
            for _local_index in range(chunk_goal):
                transition_index = len(state_rows)
                transition_started = time.perf_counter()
                transition = distributed_replica_exchange_transition(
                    **current,
                    inverse_temperatures=betas,
                    step_sizes=step_sizes,
                    num_leapfrog_steps=leapfrog,
                    transition_index=transition_index,
                    master_seed=MASTER_SEED,
                    evaluator=evaluator,
                    mass_matrix=mass_matrix,
                )
                transition_seconds.append(time.perf_counter() - transition_started)
                path_invalid = tf.logical_not(transition["hmc_path_valid"])
                path_self_rejected = tf.reduce_all(
                    tf.logical_not(
                        tf.boolean_mask(transition["hmc_is_accepted"], path_invalid)
                    )
                )
                finite_log_accept_or_invalid = tf.reduce_all(
                    tf.logical_or(
                        tf.math.is_finite(transition["hmc_log_accept_ratio"]),
                        tf.logical_and(
                            path_invalid,
                            tf.math.is_inf(transition["hmc_log_accept_ratio"])
                            & (transition["hmc_log_accept_ratio"] < 0.0),
                        ),
                    )
                )
                swap_matrix = tf.cast(
                    transition["swap_is_accepted_matrix"], tf.int32
                )
                swap_permutation = tf.logical_and(
                    tf.reduce_all(
                        tf.reduce_sum(swap_matrix, axis=0)
                        == tf.ones(
                            (len(betas), 4), tf.int32
                        )
                    ),
                    tf.reduce_all(
                        tf.reduce_sum(swap_matrix, axis=1)
                        == tf.ones(
                            (len(betas), 4), tf.int32
                        )
                    ),
                )
                retained_finite = tf.reduce_all(
                    tf.stack(
                        (
                            tf.reduce_all(tf.math.is_finite(transition["state"])),
                            tf.reduce_all(
                                tf.math.is_finite(
                                    transition["base_target_log_prob"]
                                )
                            ),
                            tf.reduce_all(
                                tf.math.is_finite(transition["base_score"])
                            ),
                        )
                    )
                )
                transition_hard_gates = {
                    "invalid_paths_self_rejected": bool(path_self_rejected.numpy()),
                    "log_acceptance_finite_or_invalid_negative_infinity": bool(
                        finite_log_accept_or_invalid.numpy()
                    ),
                    "swap_matrix_is_permutation": bool(swap_permutation.numpy()),
                    "retained_state_target_score_finite": bool(
                        retained_finite.numpy()
                    ),
                }
                if not all(transition_hard_gates.values()):
                    hard_gate_failures.extend(
                        f"transition_{transition_index}:{name}"
                        for name, passed_gate in transition_hard_gates.items()
                        if not passed_gate
                    )
                    terminal_status = "MATERIAL_REPLICA_HARD_GATE_FAILED"
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
                if terminal_status == "MATERIAL_REPLICA_HARD_GATE_FAILED":
                    break

            cache_started = time.perf_counter()
            cache_value, cache_score, cache_status, _metadata = evaluator(
                tf.reshape(current["state"], (ROWS, PARAMETER_DIM)),
                f"chunk-{len(chunk_receipts):04d}-terminal-cache",
            )
            cache_seconds.append(time.perf_counter() - cache_started)
            cache_value = tf.reshape(
                cache_value, (len(betas), 4)
            )
            cache_score = tf.reshape(
                cache_score,
                (len(betas), 4, PARAMETER_DIM),
            )
            cache_valid = tf.logical_and(
                tf.convert_to_tensor(cache_status["status_code"], tf.int32) == 0,
                tf.convert_to_tensor(
                    cache_status["valid_pre_regularized_score"], tf.bool
                ),
            )
            if not bool(tf.reduce_all(cache_valid).numpy()):
                hard_gate_failures.append("terminal_cache_target_status_invalid")
                terminal_status = "MATERIAL_REPLICA_HARD_GATE_FAILED"
            # One-row evaluation is deterministic in the accepted baseline; retain
            # residuals as diagnostics without reinstating an uncalibrated absolute veto.
            cache_value_residual = tf.reduce_max(
                tf.abs(cache_value - current["base_target_log_prob"])
            )
            cache_score_residual = tf.reduce_max(
                tf.abs(cache_score - current["base_score"])
            )
            chunk_stop = len(state_rows)
            chunk_index = len(chunk_receipts)
            chunk_tensors = {
                "state": tf.stack(state_rows[chunk_start:chunk_stop]),
                "pre_swap_state": tf.stack(pre_rows[chunk_start:chunk_stop]),
                "identities": tf.stack(identity_rows[chunk_start:chunk_stop]),
                "hmc_is_accepted": tf.stack(hmc_accept_rows[chunk_start:chunk_stop]),
                "hmc_log_accept_ratio": tf.stack(
                    hmc_log_accept_rows[chunk_start:chunk_stop]
                ),
                "hmc_path_valid": tf.stack(hmc_valid_rows[chunk_start:chunk_stop]),
                "swap_is_proposed_adjacent": tf.stack(
                    swap_proposed_rows[chunk_start:chunk_stop]
                ),
                "swap_is_accepted_adjacent": tf.stack(
                    swap_accepted_rows[chunk_start:chunk_stop]
                ),
                "swap_is_accepted_matrix": tf.stack(
                    swap_matrix_rows[chunk_start:chunk_stop]
                ),
            }
            tensor_receipts = {
                name: _write_tensor(
                    OUTPUT_ROOT / f"chunk-{chunk_index:04d}-{name}.tftensor",
                    value,
                    tf,
                )
                for name, value in chunk_tensors.items()
            }
            checkpoint_receipts = {
                name: _write_tensor(
                    OUTPUT_ROOT / f"chunk-{chunk_index:04d}-terminal-{name}.tftensor",
                    current[name],
                    tf,
                )
                for name in (
                    "state",
                    "base_target_log_prob",
                    "base_score",
                    "identities_at_temperature",
                )
            }
            manifest_path = OUTPUT_ROOT / f"chunk-{chunk_index:04d}.json"
            _write_json(
                manifest_path,
                {
                    "schema": "bayesfilter.ssl_lstm.q20_physical_replica_material_chunk.v1",
                    "chunk_index": chunk_index,
                    "phase_at_start": phase,
                    "transition_start_inclusive": chunk_start,
                    "transition_stop_exclusive": chunk_stop,
                    "tensor_receipts": tensor_receipts,
                    "terminal_checkpoint_receipts": checkpoint_receipts,
                    "terminal_cache_value_max_abs_residual": cache_value_residual,
                    "terminal_cache_score_max_abs_residual": cache_score_residual,
                    "terminal_cache_status_all_valid": tf.reduce_all(cache_valid),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            chunk_receipts.append(
                {"path": manifest_path.as_posix(), "sha256": _sha(manifest_path)}
            )
            _append_log(
                f"completed chunk {chunk_index} transitions {chunk_start}:{chunk_stop}"
            )

            if terminal_status == "MATERIAL_REPLICA_HARD_GATE_FAILED":
                break
            if not warmup_complete:
                warmup_count = len(state_rows)
                if warmup_count >= WARMUP_MIN and warmup_count == target_total:
                    warmup_latent = tf.stack(state_rows[:warmup_count])
                    warmup_pre_latent = tf.stack(pre_rows[:warmup_count])
                    warmup_identities = tf.stack(identity_rows[:warmup_count])
                    warmup_physical = chart["center"] + tf.matmul(
                        tf.reshape(warmup_latent, (-1, PARAMETER_DIM)),
                        chart["factor"],
                        transpose_b=True,
                    )
                    warmup_physical = tf.reshape(
                        warmup_physical,
                        (
                            warmup_count,
                            len(betas),
                            4,
                            PARAMETER_DIM,
                        ),
                    )
                    warmup_pre_physical = chart["center"] + tf.matmul(
                        tf.reshape(warmup_pre_latent, (-1, PARAMETER_DIM)),
                        chart["factor"],
                        transpose_b=True,
                    )
                    warmup_pre_physical = tf.reshape(
                        warmup_pre_physical, tf.shape(warmup_physical)
                    )
                    convergence = _diagnose_warmup(tf, warmup_physical)
                    travel = _window_round_trips(tf, warmup_identities)
                    forgetting = _hot_forgetting(
                        tf,
                        warmup_pre_physical,
                        warmup_physical,
                        initial_physical,
                    )
                    check = {
                        "warmup_draws_per_chain": warmup_count,
                        "convergence": convergence,
                        "travel": travel,
                        "hot_forgetting": forgetting,
                        "passed": bool(
                            convergence["passed"]
                            and travel["each_chain_has_required_round_trip"].numpy()
                            and forgetting["all_chains_passed"].numpy()
                        ),
                    }
                    warmup_checks.append(check)
                    if check["passed"]:
                        warmup_complete = True
                        retained_count = 0
                        _append_log(f"warmup ready at {warmup_count} transitions")
                    elif warmup_count >= WARMUP_MAX:
                        terminal_status = "MATERIAL_REPLICA_WARMUP_NOT_READY"
                        break
            else:
                retained_count = len(state_rows) - warmup_count
                if retained_count >= RETAINED_MIN and len(state_rows) == target_total:
                    retained_latent = tf.stack(state_rows[warmup_count:])
                    retained_pre_latent = tf.stack(pre_rows[warmup_count:])
                    retained_identities = tf.stack(identity_rows[warmup_count:])
                    retained_physical = chart["center"] + tf.matmul(
                        tf.reshape(retained_latent, (-1, PARAMETER_DIM)),
                        chart["factor"],
                        transpose_b=True,
                    )
                    retained_physical = tf.reshape(
                        retained_physical,
                        (
                            retained_count,
                            len(betas),
                            4,
                            PARAMETER_DIM,
                        ),
                    )
                    retained_pre_physical = chart["center"] + tf.matmul(
                        tf.reshape(retained_pre_latent, (-1, PARAMETER_DIM)),
                        chart["factor"],
                        transpose_b=True,
                    )
                    retained_pre_physical = tf.reshape(
                        retained_pre_physical, tf.shape(retained_physical)
                    )
                    convergence = _diagnose_retained(tf, retained_physical)
                    travel = _window_round_trips(tf, retained_identities)
                    previous_physical = (
                        chart["center"]
                        + tf.matmul(
                            tf.reshape(state_rows[warmup_count - 1], (-1, PARAMETER_DIM)),
                            chart["factor"],
                            transpose_b=True,
                        )
                    )
                    previous_physical = tf.reshape(
                        previous_physical,
                        (len(betas), 4, PARAMETER_DIM),
                    )
                    forgetting = _hot_forgetting(
                        tf,
                        retained_pre_physical,
                        retained_physical,
                        previous_physical,
                    )
                    acceptance = _acceptance_summary(
                        tf, tf.stack(hmc_log_accept_rows[warmup_count:])
                    )
                    check = {
                        "retained_draws_per_chain": retained_count,
                        "convergence": convergence,
                        "travel": travel,
                        "hot_forgetting": forgetting,
                        "acceptance": acceptance,
                        "passed": bool(
                            convergence["passed"]
                            and travel["each_chain_has_required_round_trip"].numpy()
                            and forgetting["all_chains_passed"].numpy()
                            and acceptance[
                                "all_temperature_chain_means_in_band"
                            ].numpy()
                        ),
                    }
                    retained_checks.append(check)
                    if check["passed"]:
                        terminal_status = "MATERIAL_REPLICA_ADMISSION_PASSED"
                        break
                    if retained_count >= RETAINED_MAX:
                        terminal_status = "MATERIAL_REPLICA_RETAINED_GATES_FAILED"
                        break

            _write_json(
                PROGRESS,
                {
                    "status": "MATERIAL_REPLICA_RUNNING",
                    "phase": "retained" if warmup_complete else "warmup",
                    "completed_transitions": len(state_rows),
                    "warmup_transitions": warmup_count,
                    "retained_transitions": retained_count,
                    "completed_chunks": len(chunk_receipts),
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_chunk": chunk_receipts[-1],
                    "latest_warmup_check": warmup_checks[-1] if warmup_checks else None,
                    "latest_retained_check": retained_checks[-1] if retained_checks else None,
                },
                overwrite=True,
            )

    total_transitions = len(state_rows)
    retained_physical = None
    retained_receipt = None
    retained_negative_fraction = None
    if retained_count > 0:
        retained_latent = tf.stack(state_rows[warmup_count:])
        retained_physical = chart["center"] + tf.matmul(
            tf.reshape(retained_latent, (-1, PARAMETER_DIM)),
            chart["factor"],
            transpose_b=True,
        )
        retained_physical = tf.reshape(
            retained_physical,
            (
                retained_count,
                len(betas),
                4,
                PARAMETER_DIM,
            ),
        )
        retained_receipt = _write_tensor(
            OUTPUT_ROOT / "retained-physical-diagnostic.tftensor",
            retained_physical,
            tf,
        )
        retained_negative_fraction = tf.reduce_mean(
            tf.cast(retained_physical[:, 0, :, 2] < 0.0, tf.float64)
        )
    hmc_valid = tf.stack(hmc_valid_rows) if hmc_valid_rows else tf.zeros((0,), tf.bool)
    hmc_accepted = (
        tf.stack(hmc_accept_rows) if hmc_accept_rows else tf.zeros((0,), tf.bool)
    )
    invalid = tf.logical_not(hmc_valid)
    invalid_self_rejected = bool(
        tf.reduce_all(
            tf.logical_not(tf.boolean_mask(hmc_accepted, invalid))
        ).numpy()
    ) if hmc_valid_rows else True
    if not invalid_self_rejected:
        hard_gate_failures.append("invalid_hmc_path_was_accepted")
        terminal_status = "MATERIAL_REPLICA_HARD_GATE_FAILED"
    wall_seconds = time.perf_counter() - started
    if wall_seconds > HARD_WALL_CAP_SECONDS:
        hard_gate_failures.append("hard_wall_cap_exceeded")
        terminal_status = "MATERIAL_REPLICA_HARD_GATE_FAILED"
    passed = terminal_status == "MATERIAL_REPLICA_ADMISSION_PASSED" and not hard_gate_failures
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_replica_material.v1",
        "status": terminal_status,
        "passed": passed,
        "configuration": {
            "inverse_temperatures": betas,
            "step_sizes": step_sizes,
            "num_leapfrog_steps": leapfrog,
            "mass_matrix": mass_matrix,
            "mass_policy": "mean_two_checked_mapped_local_precisions",
            "chains": 4,
            "workers": WORKERS,
            "rows_per_worker": ROWS_PER_WORKER,
            "worker_cpu_ids": WORKER_CPU_IDS,
            "chunk_size": CHUNK_SIZE,
            "master_seed": MASTER_SEED,
            "jit_compile": True,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        },
        "budget": {
            "user_authorized_seconds": HARD_WALL_CAP_SECONDS,
            "transition_deadline_seconds": RUNNER_TRANSITION_DEADLINE_SECONDS,
            "finalization_reserve_seconds": FINALIZATION_RESERVE_SECONDS,
            "wall_seconds": wall_seconds,
            "total_transitions": total_transitions,
            "warmup_transitions": warmup_count,
            "retained_transitions": retained_count,
        },
        "thresholds": {
            "warmup_min": WARMUP_MIN,
            "warmup_max": WARMUP_MAX,
            "warmup_window": WARMUP_WINDOW,
            "warmup_rhat_max": WARMUP_RHAT_MAX,
            "retained_min": RETAINED_MIN,
            "retained_max": RETAINED_MAX,
            "retained_rhat_max": RETAINED_RHAT_MAX,
            "retained_bulk_ess_min": RETAINED_BULK_ESS_MIN,
            "retained_tail_ess_min": RETAINED_TAIL_ESS_MIN,
            "acceptance_lower": ACCEPTANCE_LOWER,
            "acceptance_upper": ACCEPTANCE_UPPER,
            "minimum_round_trips_per_chain": MIN_ROUND_TRIPS_PER_CHAIN,
        },
        "hard_gate_failures": hard_gate_failures,
        "invalid_hmc_path_count": tf.reduce_sum(tf.cast(invalid, tf.int32)),
        "invalid_paths_self_rejected": invalid_self_rejected,
        "warmup_checks": warmup_checks,
        "retained_checks": retained_checks,
        "cold_negative_sign_fraction": retained_negative_fraction,
        "smc_two_known_region_negative_mass_interval": SMC_NEGATIVE_MASS_INTERVAL,
        "occupancy_role": "explanatory_consistency_only_not_mass_authority",
        "worker_identity": identity,
        "timing": {
            "transition_seconds": transition_seconds,
            "cache_seconds": cache_seconds,
            "evaluation_seconds": evaluation_seconds,
            "mean_transition_seconds": (
                sum(transition_seconds) / len(transition_seconds)
                if transition_seconds
                else None
            ),
        },
        "chunk_manifests": chunk_receipts,
        "retained_physical_diagnostic_receipt": retained_receipt,
        "bindings": bindings,
        "run_manifest": {
            "launch_git_commit": launch_git_commit,
            "launch_git_dirty": launch_git_dirty,
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "wall_time_seconds": wall_seconds,
            "artifact_root": OUTPUT_ROOT.as_posix(),
            "plan_file": PLAN.as_posix(),
            "result_file": RESULT.as_posix(),
            "launch_source_sha256": launch_source_sha256,
        },
        "decision_table": {
            "candidate_admission": terminal_status,
            "primary_criterion_status": (
                "passed" if passed else "not_passed"
            ),
            "veto_status": hard_gate_failures,
            "next_justified_action": (
                "issue_candidate_posterior_archive_and_run_predictive_validation"
                if passed
                else "preserve_failure_and_do_not_issue_posterior_archive"
            ),
            "not_concluded": (
                "exhaustive mode discovery",
                "full posterior authority",
                "predictive equivalence",
                "sampler superiority",
                "default readiness",
            ),
        },
        "inference_status": {
            "hard_veto_screen": "passed" if not hard_gate_failures else "failed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "runtime",
                "acceptance",
                "raw cold occupancy",
                "individual sign paths",
            ),
            "default_readiness": False,
            "next_evidence_needed": (
                "posterior predictive validation" if passed else "candidate repair"
            ),
        },
        "nonclaims": (
            "SMC remains the two-known-region mass authority",
            "raw cold occupancy is not a posterior mass estimate",
            "finite-sample diagnostics are not a convergence proof",
            "two-region travel does not prove exhaustive mode discovery",
            "no predictive or default-readiness claim",
        ),
    }
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": terminal_status,
            "passed": passed,
            "completed_transitions": total_transitions,
            "warmup_transitions": warmup_count,
            "retained_transitions": retained_count,
            "elapsed_seconds": wall_seconds,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    _append_log(f"completed material campaign: {terminal_status}")
    return payload


def main() -> None:
    started = time.perf_counter()
    try:
        payload = run_material()
    except BaseException as error:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20_physical_replica_material_failure.v1",
            "status": "MATERIAL_REPLICA_HARNESS_FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_seconds": time.perf_counter() - started,
        }
        if not _abs(FINAL).exists():
            _write_json(FINAL, failure)
        _write_json(PROGRESS, {**failure, "result": FINAL.as_posix()}, overwrite=True)
        _append_log(f"failed material campaign: {type(error).__name__}: {error}")
        raise
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
