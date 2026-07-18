#!/usr/bin/env python3
"""Run one versioned fused-OT canonical LGSSM GPU/XLA repair node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


CAMPAIGN_ID = "canonical-lgssm-fused-ot-loop-repair-20260718"
SCHEMA_VERSION = "bayesfilter.canonical_lgssm_fused_ot_loop_repair.v1"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-canonical-lgssm-fused-ot-loop-performance-repair-plan-2026-07-18.md"
)
DATASET_SEED = 81100
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
RIDGE = 7.301568984985351e-09
SINKHORN_STEPS = 20
EPSILON = 0.5
SCALING = 0.9
GPU_MEMORY_LIMIT_MIB = 8192
ALLOWED_HORIZONS = (2, 10, 50)
ALLOWED_PARTICLE_COUNTS = (128, 1024)
ALLOWED_ARMS = ("all_active_contract_e", "no_reset_weighted")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _parse_seeds(text: str) -> tuple[int, ...]:
    seeds = tuple(int(token) for token in text.split(",") if token.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty unique comma-separated list")
    return seeds


def _configure_gpu(dtype: tf.dtypes.DType) -> dict[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("trusted GPU execution requires one visible GPU")
    tf.config.set_logical_device_configuration(
        physical[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT_MIB)],
    )
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError(f"expected one logical GPU, got {logical}")
    tf32 = dtype == tf.float32
    tf.config.experimental.enable_tensor_float_32_execution(tf32)
    return {
        "physical_devices": [device.name for device in physical],
        "logical_devices": [device.name for device in logical],
        "memory_limit_mib": GPU_MEMORY_LIMIT_MIB,
        "memory_growth": False,
        "jit_compile": True,
        "dtype": dtype.name,
        "tf32_enabled": tf32,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _kalman(
    observations: tf.Tensor, theta: tf.Tensor, canonical: Any
) -> tuple[float, list[float]]:
    from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood

    observations64 = tf.cast(observations, tf.float64)
    theta64 = tf.cast(theta, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta64)
        phi = theta64[:3]
        q_scale = theta64[3]
        r_scale = theta64[4]
        value = tf_kalman_log_likelihood(
            observations=observations64,
            transition_offset=tf.zeros([3], tf.float64),
            transition_matrix=tf.linalg.diag(phi),
            transition_covariance=tf.square(q_scale) * tf.eye(3, dtype=tf.float64),
            observation_offset=tf.zeros([3], tf.float64),
            observation_matrix=canonical._observation_matrix(tf.float64),
            observation_covariance=tf.square(r_scale) * tf.eye(3, dtype=tf.float64),
            initial_state_mean=tf.zeros([3], tf.float64),
            initial_state_covariance=tf.linalg.diag(
                tf.square(q_scale) / (1.0 - tf.square(phi))
            ),
        )
    score = tape.gradient(value, theta64)
    return float(value), [float(item) for item in score]


def _synchronize(result: dict[str, tf.Tensor]) -> None:
    tf.io.serialize_tensor(result["objective"]).numpy()
    tf.io.serialize_tensor(result["score"]).numpy()


def _serialized_equal(
    first: dict[str, tf.Tensor], second: dict[str, tf.Tensor]
) -> bool:
    names = (
        "objective",
        "per_batch_log_likelihood",
        "score",
        "per_batch_score",
        "valid_chart",
        "reset_valid_history",
        "quotient_marginal_valid_history",
        "tv_column_error_history",
        "maximum_row_error_history",
    )
    return all(
        tf.io.serialize_tensor(first[name]).numpy()
        == tf.io.serialize_tensor(second[name]).numpy()
        for name in names
    )


def _finite(result: dict[str, tf.Tensor]) -> bool:
    return all(
        bool(tf.reduce_all(tf.math.is_finite(result[name])))
        for name in (
            "objective",
            "per_batch_log_likelihood",
            "score",
            "per_batch_score",
            "final_particles",
            "final_log_weights",
        )
    )


def _execute(
    args: argparse.Namespace, *, configured_device: dict[str, Any] | None = None
) -> dict[str, Any]:
    dtype = tf.float64 if args.dtype == "float64" else tf.float32
    device = configured_device or _configure_gpu(dtype)
    from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
    from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    chunks = select_transport_chunks(args.num_particles)
    seeds = _parse_seeds(args.seeds)
    raw_observations = _lgssm_dataset(DATASET_SEED)["observations"][: args.time_steps]
    active = args.arm == "all_active_contract_e"
    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=raw_observations,
        estimator_seeds=seeds,
        num_particles=args.num_particles,
        fixed_reset_mask=[[active] * args.time_steps for _ in seeds],
        prepared_ridge=[[RIDGE] * args.time_steps for _ in seeds],
        epsilon=EPSILON,
        scaling=SCALING,
        sinkhorn_steps=SINKHORN_STEPS,
        balance_steps=args.balance_steps,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=dtype,
    )
    prepared = preparation.prepared_values(prepared_result)
    observations = prepared["observations"]
    callable_ = canonical.make_canonical_value_and_score_tf(
        prepared,
        steps=SINKHORN_STEPS,
        balance_steps=args.balance_steps,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        jit_compile=True,
        dtype=dtype,
    )
    theta = tf.constant(THETA, dtype)
    trace_started = time.perf_counter()
    concrete = callable_.get_concrete_function()
    trace_seconds = time.perf_counter() - trace_started
    first_started = time.perf_counter()
    first = callable_(theta)
    _synchronize(first)
    first_seconds = time.perf_counter() - first_started
    warm_started = time.perf_counter()
    warm = callable_(theta)
    _synchronize(warm)
    warm_seconds = time.perf_counter() - warm_started
    replay_equal = _serialized_equal(first, warm)
    finite = _finite(first)
    active_mask = first["active_reset_history"]
    if active:
        maximum_tv_column = float(
            tf.reduce_max(tf.boolean_mask(first["tv_column_error_history"], active_mask))
        )
        maximum_row_error = float(
            tf.reduce_max(tf.boolean_mask(first["maximum_row_error_history"], active_mask))
        )
        marginal_valid = bool(
            tf.reduce_all(
                tf.boolean_mask(first["quotient_marginal_valid_history"], active_mask)
            )
        )
        reset_valid = bool(
            tf.reduce_all(tf.boolean_mask(first["reset_valid_history"], active_mask))
        )
    else:
        maximum_tv_column = 0.0
        maximum_row_error = 0.0
        marginal_valid = True
        reset_valid = True
    chart_valid = bool(tf.reduce_all(first["valid_chart"]))
    kalman_value, kalman_score = _kalman(observations, theta, canonical)
    aggregate_value = float(first["objective"])
    aggregate_score = [float(item) for item in first["score"]]
    memory = tf.config.experimental.get_memory_info("GPU:0")
    graph_def = concrete.graph.as_graph_def()
    graph_bytes = len(graph_def.SerializeToString())
    operation_types = [operation.type for operation in concrete.graph.get_operations()]
    source_paths = (
        "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    )
    work = {
        name.removeprefix("work_"): int(first[name])
        for name in (
            "work_sinkhorn_state_constructions",
            "work_terminal_balance_state_constructions",
            "work_transport_tile_sweeps",
            "work_marginal_tile_sweeps",
            "work_diagnostic_solver_reconstructions",
            "work_active_reset_rows",
        )
    }
    expected_steps = args.time_steps if active else 0
    work_valid = (
        work["sinkhorn_state_constructions"] == expected_steps
        and work["terminal_balance_state_constructions"] == expected_steps
        and work["transport_tile_sweeps"] == expected_steps
        and work["marginal_tile_sweeps"] == 0
        and work["diagnostic_solver_reconstructions"] == 0
    )
    hard_valid = finite and replay_equal and chart_valid and marginal_valid and reset_valid and work_valid
    return {
        "device": device,
        "git_commit": _git_commit(),
        "source_sha256": {path: _sha256(ROOT / path) for path in source_paths},
        "plan_path": PLAN_PATH,
        "arm": args.arm,
        "time_steps": args.time_steps,
        "num_particles": args.num_particles,
        "estimator_seeds": list(seeds),
        "balance_steps": args.balance_steps,
        "preparation_identity": prepared_result["identity"],
        "theta": list(THETA),
        "per_seed_value": [float(item) for item in first["per_batch_log_likelihood"]],
        "per_seed_physical_score": first["per_batch_score"].numpy().tolist(),
        "aggregate_value": aggregate_value,
        "aggregate_physical_score": aggregate_score,
        "kalman_value": kalman_value,
        "kalman_physical_score": kalman_score,
        "value_difference_to_kalman": aggregate_value - kalman_value,
        "physical_score_difference_to_kalman": [
            candidate - oracle
            for candidate, oracle in zip(aggregate_score, kalman_score, strict=True)
        ],
        "finite": finite,
        "bitwise_replay": replay_equal,
        "chart_valid": chart_valid,
        "marginal_valid": marginal_valid,
        "reset_valid": reset_valid,
        "maximum_tv_column_error": maximum_tv_column,
        "maximum_row_error": maximum_row_error,
        "work": work,
        "work_valid": work_valid,
        "hard_valid": hard_valid,
        "timing_seconds": {
            "trace": trace_seconds,
            "compile_plus_first_execution": first_seconds,
            "warm_execution": warm_seconds,
        },
        "gpu_allocator_bytes": {name: int(value) for name, value in memory.items()},
        "graph": {
            "operation_count": len(operation_types),
            "serialized_bytes": graph_bytes,
            "while_operation_types": sorted(
                operation_type
                for operation_type in operation_types
                if "While" in operation_type
            ),
            "python_horizon_unroll": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--time-steps", type=int, choices=ALLOWED_HORIZONS, required=True)
    parser.add_argument(
        "--num-particles", type=int, choices=ALLOWED_PARTICLE_COUNTS, required=True
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument("--balance-steps", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": args.attempt_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
    }
    try:
        payload["result"] = _execute(args)
        payload["status"] = "node_complete"
    except Exception as error:  # structured resource/numerical failure evidence
        payload["status"] = "node_failed"
        payload["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["wall_time_seconds"] = time.perf_counter() - started
    _write_exclusive(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "attempt_id", "wall_time_seconds")}, indent=2))
    if payload["status"] != "node_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
