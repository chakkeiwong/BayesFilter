#!/usr/bin/env python3
"""Select LGSSM terminal balancing from consumed-plan marginals only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

canonical = None
preparation = None
_lgssm_dataset = None
select_transport_chunks = None


CAMPAIGN_ID = "canonical-lgssm-balancing-kalman-repair-20260717"
DATASET_SEED = 81100
DESIGN_SEEDS = tuple(range(81300, 81308))
AUDIT_SEEDS = tuple(range(81320, 81328))
BALANCE_CANDIDATES = (0, 1, 2, 5, 10, 20, 50, 100)
THETA = (0.72, 0.55, 0.35, 0.35, 0.45)
TIME_STEPS = 2
NUM_PARTICLES = 128
RIDGE = 7.301568984985351e-09
SINKHORN_STEPS = 20
EPSILON = 0.5
SCALING = 0.9
GPU_MEMORY_LIMIT_MIB = 8192


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


def _load_project_modules() -> None:
    global canonical, preparation, _lgssm_dataset, select_transport_chunks
    from bayesfilter.highdim import (
        ledh_contract_e_canonical_lgssm_tf as canonical_module,
    )
    from bayesfilter.highdim import (
        ledh_contract_e_lgssm_preparation_tf as preparation_module,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset as dataset_factory,
    )
    from bayesfilter.highdim.transport_chunk_policy import (
        select_transport_chunks as chunk_selector,
    )

    canonical = canonical_module
    preparation = preparation_module
    _lgssm_dataset = dataset_factory
    select_transport_chunks = chunk_selector


def _observations() -> tf.Tensor:
    return tf.convert_to_tensor(
        _lgssm_dataset(DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )


def _prepare(seeds: tuple[int, ...], balance_steps: int) -> dict[str, Any]:
    batch = len(seeds)
    chunks = select_transport_chunks(NUM_PARTICLES)
    return preparation.prepare_contract_e_lgssm_inputs(
        observations=_observations(),
        estimator_seeds=seeds,
        num_particles=NUM_PARTICLES,
        fixed_reset_mask=[[True] * TIME_STEPS for _ in range(batch)],
        prepared_ridge=[[RIDGE] * TIME_STEPS for _ in range(batch)],
        epsilon=EPSILON,
        scaling=SCALING,
        sinkhorn_steps=SINKHORN_STEPS,
        balance_steps=balance_steps,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float64,
    )


def _all_finite(result: dict[str, tf.Tensor]) -> bool:
    names = (
        "objective",
        "per_batch_log_likelihood",
        "minimum_mass",
        "quotient_row_residual_history",
        "quotient_post_quotient_column_residual_history",
        "quotient_marginal_roundoff_tolerance_history",
        "covariance_gap_eigenvalue_history",
    )
    return all(
        bool(tf.reduce_all(tf.math.is_finite(result[name])).numpy())
        for name in names
    )


def _record(result: dict[str, tf.Tensor], identity: dict[str, Any]) -> dict[str, Any]:
    active = result["active_reset_history"]
    row = tf.boolean_mask(result["quotient_row_residual_history"], active)
    column = tf.boolean_mask(
        result["quotient_post_quotient_column_residual_history"], active
    )
    tolerance = tf.boolean_mask(
        result["quotient_marginal_roundoff_tolerance_history"], active
    )
    marginal_valid = tf.boolean_mask(
        result["quotient_marginal_valid_history"], active
    )
    reset_valid = tf.boolean_mask(result["reset_valid_history"], active)
    gap = tf.boolean_mask(result["covariance_gap_eigenvalue_history"], active)
    finite = _all_finite(result)
    pass_all = (
        finite
        and bool(tf.reduce_all(marginal_valid).numpy())
        and bool(tf.reduce_all(reset_valid).numpy())
        and bool(tf.reduce_all(result["flow_valid_history"]).numpy())
        and bool(tf.reduce_all(result["geometry_valid_history"]).numpy())
        and bool(tf.reduce_all(result["quotient_valid_history"]).numpy())
        and bool(tf.reduce_all(gap > 0.0).numpy())
        and bool(tf.reduce_all(result["minimum_mass"] > 0.0).numpy())
    )
    return {
        "balance_steps": identity["balance_steps"],
        "preparation_identity": identity,
        "finite": finite,
        "all_marginals_valid": bool(tf.reduce_all(marginal_valid).numpy()),
        "all_resets_valid": bool(tf.reduce_all(reset_valid).numpy()),
        "maximum_row_residual": float(tf.reduce_max(row).numpy()),
        "maximum_post_quotient_column_residual": float(
            tf.reduce_max(column).numpy()
        ),
        "maximum_roundoff_tolerance": float(tf.reduce_max(tolerance).numpy()),
        "minimum_covariance_gap_eigenvalue": float(tf.reduce_min(gap).numpy()),
        "minimum_mass": float(tf.reduce_min(result["minimum_mass"]).numpy()),
        "pass": pass_all,
    }


def _evaluate(seeds: tuple[int, ...], balance_steps: int) -> dict[str, Any]:
    prepared = _prepare(seeds, balance_steps)
    chunks = select_transport_chunks(NUM_PARTICLES)
    tensors = preparation.prepared_values(prepared)

    @tf.function(
        input_signature=[tf.TensorSpec([canonical.PARAMETER_COUNT], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def execute(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        return canonical._canonical_primal_core(
            theta,
            tensors,
            steps=SINKHORN_STEPS,
            balance_steps=balance_steps,
            row_chunk_size=chunks.row_chunk_size,
            col_chunk_size=chunks.col_chunk_size,
        )

    started = time.perf_counter()
    first = execute(tf.constant(THETA, tf.float64))
    second = execute(tf.constant(THETA, tf.float64))
    record = _record(first, prepared["identity"])
    replay_names = (
        "objective",
        "valid_chart",
        "reset_valid_history",
        "quotient_marginal_valid_history",
        "quotient_row_residual_history",
        "quotient_post_quotient_column_residual_history",
    )
    record["bitwise_replay"] = all(
        tf.io.serialize_tensor(first[name]).numpy()
        == tf.io.serialize_tensor(second[name]).numpy()
        for name in replay_names
    )
    record["pass"] = bool(record["pass"] and record["bitwise_replay"])
    record["wall_time_seconds"] = time.perf_counter() - started
    concrete = execute.get_concrete_function()
    record["graph_operations"] = len(concrete.graph.get_operations())
    record["while_operation_types"] = sorted(
        operation.type
        for operation in concrete.graph.get_operations()
        if "While" in operation.type
    )
    return record


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": "bayesfilter.canonical_lgssm_balance_selection.v1",
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "selection_uses_kalman": False,
        "selection_metrics": [
            "finiteness",
            "chart_and_factor_validity",
            "both_consumed_plan_marginals",
            "positive_covariance_gap",
            "bitwise_replay",
        ],
    }
    try:
        payload["device"] = _configure_gpu()
        _load_project_modules()
        payload["git_commit"] = _git_commit()
        source_paths = (
            "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
            "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
            "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
            "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        )
        payload["source_sha256"] = {
            path: _sha256(ROOT / path) for path in source_paths
        }
        payload["frozen_design"] = {
            "dataset_seed": DATASET_SEED,
            "design_seeds": list(DESIGN_SEEDS),
            "audit_seeds": list(AUDIT_SEEDS),
            "time_steps": TIME_STEPS,
            "num_particles": NUM_PARTICLES,
            "ridge": RIDGE,
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_candidates": list(BALANCE_CANDIDATES),
            "chunks": [
                select_transport_chunks(NUM_PARTICLES).row_chunk_size,
                select_transport_chunks(NUM_PARTICLES).col_chunk_size,
            ],
            "epsilon": EPSILON,
            "scaling": SCALING,
        }
        design_results = []
        selected = None
        for candidate in BALANCE_CANDIDATES:
            result = _evaluate(DESIGN_SEEDS, candidate)
            design_results.append(result)
            if result["pass"]:
                selected = candidate
                break
        payload["design_results"] = design_results
        payload["selected_balance_steps"] = selected
        if selected is None:
            payload["status"] = "no_design_candidate_passed"
            payload["audit_result"] = None
        else:
            audit = _evaluate(AUDIT_SEEDS, selected)
            payload["audit_result"] = audit
            payload["status"] = (
                "selected_schedule_audit_passed"
                if audit["pass"]
                else "selected_schedule_audit_failed_no_retuning"
            )
        payload["wall_time_seconds"] = time.perf_counter() - started
        _write_exclusive(output, payload)
    except BaseException as error:
        payload["status"] = "failed"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        payload["wall_time_seconds"] = time.perf_counter() - started
        _write_exclusive(output, payload)
        raise
    print(json.dumps({"output": str(output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
