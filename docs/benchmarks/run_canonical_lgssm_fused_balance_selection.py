#!/usr/bin/env python3
"""Select terminal balance count from fused-route marginal errors only."""

from __future__ import annotations

import argparse
import json
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

from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner


SCHEMA_VERSION = "bayesfilter.canonical_lgssm_fused_balance_selection.v1"
BALANCE_CANDIDATES = (0, 1, 2, 3, 5, 8)
DESIGN_SEEDS = tuple(range(81300, 81308))
AUDIT_SEEDS = tuple(range(81320, 81328))
TIME_STEPS = 2
NUM_PARTICLES = 128


def _evaluate(seeds: tuple[int, ...], balance_steps: int) -> dict[str, Any]:
    from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
    from bayesfilter.highdim import ledh_contract_e_lgssm_preparation_tf as preparation
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _lgssm_dataset,
    )

    chunks = select_transport_chunks(NUM_PARTICLES)
    observations = tf.convert_to_tensor(
        _lgssm_dataset(runner.DATASET_SEED)["observations"][:TIME_STEPS], tf.float64
    )
    prepared_result = preparation.prepare_contract_e_lgssm_inputs(
        observations=observations,
        estimator_seeds=seeds,
        num_particles=NUM_PARTICLES,
        fixed_reset_mask=[[True] * TIME_STEPS for _ in seeds],
        prepared_ridge=[[runner.RIDGE] * TIME_STEPS for _ in seeds],
        epsilon=runner.EPSILON,
        scaling=runner.SCALING,
        sinkhorn_steps=runner.SINKHORN_STEPS,
        balance_steps=balance_steps,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        dtype=tf.float64,
    )
    prepared = preparation.prepared_values(prepared_result)
    if balance_steps == 0:
        @tf.function(
            input_signature=[tf.TensorSpec([canonical.PARAMETER_COUNT], tf.float64)],
            jit_compile=True,
            reduce_retracing=True,
        )
        def callable_(theta: tf.Tensor) -> dict[str, tf.Tensor]:
            return canonical._canonical_fused_loop_core(
                theta,
                prepared,
                steps=runner.SINKHORN_STEPS,
                balance_steps=0,
                row_chunk_size=chunks.row_chunk_size,
                col_chunk_size=chunks.col_chunk_size,
                execute_contract_e=True,
            )
    else:
        callable_ = canonical.make_canonical_value_and_score_tf(
            prepared,
            steps=runner.SINKHORN_STEPS,
            balance_steps=balance_steps,
            row_chunk_size=chunks.row_chunk_size,
            col_chunk_size=chunks.col_chunk_size,
            jit_compile=True,
            dtype=tf.float64,
        )
    theta = tf.constant(runner.THETA, tf.float64)
    started = time.perf_counter()
    first = callable_(theta)
    runner._synchronize(first)
    first_seconds = time.perf_counter() - started
    warm_started = time.perf_counter()
    warm = callable_(theta)
    runner._synchronize(warm)
    warm_seconds = time.perf_counter() - warm_started
    active = first["active_reset_history"]
    tv_column = tf.boolean_mask(first["tv_column_error_history"], active)
    row_error = tf.boolean_mask(first["maximum_row_error_history"], active)
    marginal = tf.boolean_mask(first["quotient_marginal_valid_history"], active)
    reset = tf.boolean_mask(first["reset_valid_history"], active)
    passed = (
        runner._finite(first)
        and runner._serialized_equal(first, warm)
        and bool(tf.reduce_all(first["valid_chart"]))
        and bool(tf.reduce_all(marginal))
        and bool(tf.reduce_all(reset))
        and int(first["work_sinkhorn_state_constructions"]) == TIME_STEPS
        and int(first["work_diagnostic_solver_reconstructions"]) == 0
        and int(first["work_marginal_tile_sweeps"]) == 0
    )
    return {
        "balance_steps": balance_steps,
        "seeds": list(seeds),
        "pass": passed,
        "maximum_tv_column_error": float(tf.reduce_max(tv_column)),
        "maximum_row_error": float(tf.reduce_max(row_error)),
        "all_marginals_valid": bool(tf.reduce_all(marginal)),
        "all_resets_valid": bool(tf.reduce_all(reset)),
        "bitwise_replay": runner._serialized_equal(first, warm),
        "compile_plus_first_execution_seconds": first_seconds,
        "warm_execution_seconds": warm_seconds,
        "preparation_identity": prepared_result["identity"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": runner.CAMPAIGN_ID,
        "attempt_id": args.attempt_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_uses_only_marginals": True,
        "candidates": list(BALANCE_CANDIDATES),
        "design_seeds": list(DESIGN_SEEDS),
        "audit_seeds": list(AUDIT_SEEDS),
    }
    started = time.perf_counter()
    try:
        payload["device"] = runner._configure_gpu(tf.float64)
        design = []
        selected = None
        for candidate in BALANCE_CANDIDATES:
            record = _evaluate(DESIGN_SEEDS, candidate)
            design.append(record)
            if record["pass"] and candidate > 0:
                selected = candidate
                break
        payload["design"] = design
        payload["selected_balance_steps"] = selected
        if selected is None:
            payload["status"] = "selection_failed_no_candidate"
        else:
            audit = _evaluate(AUDIT_SEEDS, selected)
            payload["audit"] = audit
            payload["status"] = (
                "selection_complete" if audit["pass"] else "audit_failed_no_retuning"
            )
    except Exception as error:
        payload["status"] = "selection_execution_failed"
        payload["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["wall_time_seconds"] = time.perf_counter() - started
    runner._write_exclusive(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_balance_steps": payload.get("selected_balance_steps"),
        "wall_time_seconds": payload["wall_time_seconds"],
    }, indent=2))
    if payload["status"] != "selection_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
