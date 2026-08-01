#!/usr/bin/env python3
"""Run a bounded same-scalar HMC-coordinate FD diagnostic for canonical LGSSM."""

from __future__ import annotations

import hashlib
import json
import math
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
ESTIMATOR_SEEDS = (81400,)
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
TIME_STEPS = 2
NUM_PARTICLES = 128
RIDGE = 7.301568984985351e-09
SINKHORN_STEPS = 20
BALANCE_STEPS = 50
EPSILON = 0.5
SCALING = 0.9
GPU_MEMORY_LIMIT_MIB = 8192
FD_STEP = 2.0**-17
FD_STEP_PROVENANCE = (
    "phase8_nearest_dyadic_cuberoot_binary64_epsilon_repair_20260714"
)


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
    from bayesfilter import ledh_fd_policy
    from bayesfilter.highdim import (
        ledh_contract_e_canonical_lgssm_tf as canonical,
    )
    from bayesfilter.highdim import (
        ledh_contract_e_lgssm_preparation_tf as preparation,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    return ledh_fd_policy, canonical, preparation, _lgssm_dataset


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _physical_from_hmc(hmc: tf.Tensor) -> tf.Tensor:
    return tf.concat([tf.math.tanh(hmc[:3]), tf.math.exp(hmc[3:])], axis=0)


def _branch_hash(result: dict[str, tf.Tensor]) -> str:
    names = (
        "valid_chart",
        "active_reset_history",
        "flow_valid_history",
        "geometry_valid_history",
        "quotient_valid_history",
        "quotient_marginal_valid_history",
        "reset_valid_history",
        "diameter_max_mask",
        "geometry_max_mask",
        "geometry_min_mask",
        "epsilon0_floor_inactive",
        "sinkhorn_running_branch",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("ascii"))
        digest.update(tf.io.serialize_tensor(result[name]).numpy())
    return digest.hexdigest()


def _execute() -> dict[str, Any]:
    device = _configure_gpu()
    fd_policy, canonical, preparation, dataset_factory = _load_modules()
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

    chunks = select_transport_chunks(NUM_PARTICLES)
    observations = tf.convert_to_tensor(
        dataset_factory(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=ESTIMATOR_SEEDS,
        num_particles=NUM_PARTICLES,
        fixed_reset_mask=[[True] * TIME_STEPS],
        prepared_ridge=[[RIDGE] * TIME_STEPS],
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
    physical_center = tf.constant(THETA, tf.float64)
    hmc_center = tf.concat(
        [tf.math.atanh(physical_center[:3]), tf.math.log(physical_center[3:])],
        axis=0,
    )
    started = time.perf_counter()
    center = callable_(physical_center)
    center_branch_hash = _branch_hash(center)
    chain = tf.concat(
        [1.0 - tf.square(physical_center[:3]), physical_center[3:]], axis=0
    )
    hmc_score = center["score"] * chain
    finite_differences = []
    endpoint_records = []
    for index, parameter in enumerate(canonical.PARAMETER_NAMES):
        direction = tf.one_hot(index, canonical.PARAMETER_COUNT, dtype=tf.float64)
        plus_hmc = hmc_center + tf.constant(FD_STEP, tf.float64) * direction
        minus_hmc = hmc_center - tf.constant(FD_STEP, tf.float64) * direction
        plus = callable_(_physical_from_hmc(plus_hmc))
        minus = callable_(_physical_from_hmc(minus_hmc))
        effective_denominator = float((plus_hmc[index] - minus_hmc[index]).numpy())
        finite_difference = float(
            ((plus["objective"] - minus["objective"]) / effective_denominator).numpy()
        )
        finite_differences.append(finite_difference)
        plus_branch_hash = _branch_hash(plus)
        minus_branch_hash = _branch_hash(minus)
        endpoint_records.append(
            {
                "parameter": parameter,
                "parameter_index": index,
                "nominal_hmc_step": FD_STEP,
                "nominal_hmc_step_hex": float(FD_STEP).hex(),
                "effective_denominator": effective_denominator,
                "plus_hmc_coordinate": float(plus_hmc[index].numpy()),
                "minus_hmc_coordinate": float(minus_hmc[index].numpy()),
                "plus_physical_coordinate": float(
                    _physical_from_hmc(plus_hmc)[index].numpy()
                ),
                "minus_physical_coordinate": float(
                    _physical_from_hmc(minus_hmc)[index].numpy()
                ),
                "plus_objective": float(plus["objective"].numpy()),
                "minus_objective": float(minus["objective"].numpy()),
                "finite_difference": finite_difference,
                "plus_branch_hash": plus_branch_hash,
                "minus_branch_hash": minus_branch_hash,
                "branches_match_center": (
                    plus_branch_hash == center_branch_hash
                    and minus_branch_hash == center_branch_hash
                ),
                "charts_valid": bool(
                    tf.reduce_all(plus["valid_chart"])
                    & tf.reduce_all(minus["valid_chart"])
                ),
                "finite": all(
                    math.isfinite(value)
                    for value in (
                        effective_denominator,
                        finite_difference,
                        float(plus["objective"].numpy()),
                        float(minus["objective"].numpy()),
                    )
                ),
            }
        )
    wall_time = time.perf_counter() - started
    policy = fd_policy.evaluate_ledh_fd_policy(
        hmc_score.numpy().tolist(),
        finite_differences,
        canonical.PARAMETER_NAMES,
    )
    endpoint_valid = all(
        record["branches_match_center"]
        and record["charts_valid"]
        and record["finite"]
        and record["effective_denominator"] > 0.0
        for record in endpoint_records
    )
    hard_valid = (
        endpoint_valid
        and policy["status"] == "pass"
        and bool(tf.reduce_all(center["valid_chart"]).numpy())
        and bool(tf.reduce_all(center["reset_valid_history"]).numpy())
        and bool(
            tf.reduce_all(center["quotient_marginal_valid_history"]).numpy()
        )
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    source_paths = (
        "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
        "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        "bayesfilter/ledh_fd_policy.py",
    )
    return {
        "device": device,
        "git_commit": _git_commit(),
        "source_sha256": {path: _sha256(ROOT / path) for path in source_paths},
        "time_steps": TIME_STEPS,
        "num_particles": NUM_PARTICLES,
        "estimator_seeds": list(ESTIMATOR_SEEDS),
        "preparation_identity": prepared_result["identity"],
        "physical_center": list(THETA),
        "hmc_center": hmc_center.numpy().tolist(),
        "hmc_coordinate_definition": "atanh(phi1:3), log(q_scale), log(r_scale)",
        "center_objective": float(center["objective"].numpy()),
        "center_physical_score": center["score"].numpy().tolist(),
        "center_hmc_score": hmc_score.numpy().tolist(),
        "center_branch_hash": center_branch_hash,
        "fd_step": FD_STEP,
        "fd_step_hex": float(FD_STEP).hex(),
        "fd_step_provenance": FD_STEP_PROVENANCE,
        "finite_differences": finite_differences,
        "endpoint_records": endpoint_records,
        "fd_policy": policy,
        "endpoint_valid": endpoint_valid,
        "hard_valid": hard_valid,
        "wall_time_seconds": wall_time,
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "nonclaims": [
            "FD is an implementation diagnostic only",
            "not Kalman agreement",
            "not a confidence interval",
            "not HMC readiness",
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload: dict[str, Any] = {
        "schema_version": "bayesfilter.canonical_lgssm_same_scalar_fd.v1",
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
    }
    started = time.perf_counter()
    try:
        payload["result"] = _execute()
        payload["status"] = (
            "fd_complete" if payload["result"]["hard_valid"] else "fd_invalid"
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
