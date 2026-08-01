#!/usr/bin/env python3
"""Run one GPU/XLA arm of the canonical LGSSM particle diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


CAMPAIGN_ID = "canonical-lgssm-balancing-kalman-repair-20260717"
DATASET_SEED = 81100
ESTIMATOR_SEEDS = tuple(range(81400, 81416))
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
TIME_STEPS = 2
RIDGE = 7.301568984985351e-09
SINKHORN_STEPS = 20
BALANCE_STEPS = 50
EPSILON = 0.5
SCALING = 0.9
GPU_MEMORY_LIMIT_MIB = 8192
ALLOWED_PARTICLE_COUNTS = (128, 256, 512, 1024)
ALLOWED_ARMS = ("all_active_contract_e", "no_reset_weighted")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _configure_gpu() -> dict[str, Any]:
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
    return {
        "physical_devices": [device.name for device in physical],
        "logical_devices": [device.name for device in logical],
        "memory_limit_mib": GPU_MEMORY_LIMIT_MIB,
        "memory_growth": False,
        "jit_compile": True,
        "dtype": "float64",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _load_modules():
    from bayesfilter.highdim import (
        ledh_contract_e_canonical_lgssm_tf as canonical,
    )
    from bayesfilter.highdim import (
        ledh_contract_e_lgssm_preparation_tf as preparation,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    return canonical, preparation, _lgssm_dataset


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _execute(num_particles: int, arm: str) -> dict[str, Any]:
    device = _configure_gpu()
    canonical, preparation, dataset_factory = _load_modules()
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

    chunks = select_transport_chunks(num_particles)
    observations = tf.convert_to_tensor(
        dataset_factory(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    active = arm == "all_active_contract_e"
    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=ESTIMATOR_SEEDS,
        num_particles=num_particles,
        fixed_reset_mask=[
            [active] * TIME_STEPS for _ in range(len(ESTIMATOR_SEEDS))
        ],
        prepared_ridge=[
            [RIDGE] * TIME_STEPS for _ in range(len(ESTIMATOR_SEEDS))
        ],
        epsilon=EPSILON,
        scaling=SCALING,
        sinkhorn_steps=SINKHORN_STEPS,
        balance_steps=BALANCE_STEPS,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float64,
    )
    callable_ = canonical.make_canonical_value_and_score_tf(
        preparation.prepared_values(prepared_result),
        steps=SINKHORN_STEPS,
        balance_steps=BALANCE_STEPS,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        jit_compile=True,
        dtype=tf.float64,
    )
    theta = tf.constant(THETA, tf.float64)
    chain = tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)
    started = time.perf_counter()
    first = callable_(theta)
    second = callable_(theta)
    wall_time = time.perf_counter() - started
    replay_names = (
        "objective",
        "per_batch_log_likelihood",
        "score",
        "per_batch_score",
        "valid_chart",
        "reset_valid_history",
        "quotient_marginal_valid_history",
    )
    bitwise_replay = all(
        tf.io.serialize_tensor(first[name]).numpy()
        == tf.io.serialize_tensor(second[name]).numpy()
        for name in replay_names
    )
    executed_finite = all(
        bool(tf.reduce_all(tf.math.is_finite(first[name])).numpy())
        for name in (
            "objective",
            "per_batch_log_likelihood",
            "score",
            "per_batch_score",
            "final_particles",
            "final_log_weights",
        )
    )
    active_mask = first["active_reset_history"]
    if active:
        active_marginals = tf.boolean_mask(
            first["quotient_marginal_valid_history"], active_mask
        )
        active_resets = tf.boolean_mask(first["reset_valid_history"], active_mask)
        maximum_row_residual = float(
            tf.reduce_max(
                tf.boolean_mask(
                    first["quotient_row_residual_history"], active_mask
                )
            ).numpy()
        )
        maximum_column_residual = float(
            tf.reduce_max(
                tf.boolean_mask(
                    first["quotient_post_quotient_column_residual_history"],
                    active_mask,
                )
            ).numpy()
        )
        maximum_tolerance = float(
            tf.reduce_max(
                tf.boolean_mask(
                    first["quotient_marginal_roundoff_tolerance_history"],
                    active_mask,
                )
            ).numpy()
        )
        marginal_gate = bool(tf.reduce_all(active_marginals).numpy())
        reset_gate = bool(tf.reduce_all(active_resets).numpy())
    else:
        maximum_row_residual = None
        maximum_column_residual = None
        maximum_tolerance = None
        marginal_gate = True
        reset_gate = True
    graph = callable_.get_concrete_function().graph
    memory = tf.config.experimental.get_memory_info("GPU:0")
    source_paths = (
        "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        "bayesfilter/highdim/ledh_contract_e_reset_tf.py",
        "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
    )
    hard_valid = (
        executed_finite
        and bitwise_replay
        and bool(tf.reduce_all(first["valid_chart"]).numpy())
        and marginal_gate
        and reset_gate
    )
    return {
        "device": device,
        "git_commit": _git_commit(),
        "source_sha256": {path: _sha256(ROOT / path) for path in source_paths},
        "arm": arm,
        "num_particles": num_particles,
        "time_steps": TIME_STEPS,
        "estimator_seeds": list(ESTIMATOR_SEEDS),
        "preparation_identity": prepared_result["identity"],
        "observation_sha256": _tensor_hash(observations),
        "theta": list(THETA),
        "per_seed_value": first["per_batch_log_likelihood"].numpy().tolist(),
        "per_seed_physical_score": first["per_batch_score"].numpy().tolist(),
        "per_seed_hmc_score": (first["per_batch_score"] * chain).numpy().tolist(),
        "aggregate_value": float(first["objective"].numpy()),
        "aggregate_physical_score": first["score"].numpy().tolist(),
        "aggregate_hmc_score": (first["score"] * chain).numpy().tolist(),
        "valid_chart": first["valid_chart"].numpy().tolist(),
        "executed_finite": executed_finite,
        "bitwise_replay": bitwise_replay,
        "marginal_gate": marginal_gate,
        "reset_gate": reset_gate,
        "maximum_active_row_residual": maximum_row_residual,
        "maximum_active_post_quotient_column_residual": maximum_column_residual,
        "maximum_active_marginal_tolerance": maximum_tolerance,
        "hard_valid": hard_valid,
        "wall_time_seconds": wall_time,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "graph_operation_count": len(graph.get_operations()),
        "while_operation_types": sorted(
            operation.type
            for operation in graph.get_operations()
            if "While" in operation.type
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.num_particles not in ALLOWED_PARTICLE_COUNTS:
        raise ValueError(
            f"num_particles must be one of {ALLOWED_PARTICLE_COUNTS}"
        )
    payload: dict[str, Any] = {
        "schema_version": "bayesfilter.canonical_lgssm_particle_arm.v1",
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "arm": args.arm,
        "num_particles": args.num_particles,
    }
    started = time.perf_counter()
    try:
        payload["result"] = _execute(args.num_particles, args.arm)
        payload["status"] = (
            "arm_complete" if payload["result"]["hard_valid"] else "arm_invalid"
        )
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        payload["total_wall_time_seconds"] = time.perf_counter() - started
        _write_exclusive(output, payload)
        raise
    payload["total_wall_time_seconds"] = time.perf_counter() - started
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
