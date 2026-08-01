#!/usr/bin/env python3
"""Offline cheaper-first tuning of one canonical LGSSM execution scope.

The Python loops in this file are supervisor loops only.  The value/score
program invoked for every candidate remains the TensorFlow/XLA graph from
``run_canonical_lgssm_fused_ot_loop_repair``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner
if TYPE_CHECKING:
    from bayesfilter.highdim.ledh_tuning_scope import LEDHTuningScope


CAMPAIGN_ID = "ledh-per-model-scope-tuning-20260719"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ledh-per-model-scope-tuning-master-program-2026-07-19.md"
)
SCHEMA_VERSION = "bayesfilter.ledh_offline_ot_tuning.v1"
TV_COLUMN_GATE = 1.0e-4
ROW_ERROR_GATE = 1.0e-2
NODE_CAP_SECONDS = 7200.0
TOTAL_CAP_SECONDS = 43200.0
MODEL_ID = "canonical_lgssm_m3"
TARGET_ID = "canonical_lgssm_dataset_seed_81100_theta_0p72_0p55_0p35_0p35_0p45"
ROUTE_ID = "contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1"
RESET_CONTRACT_ID = "contract_e_chol_v1"
CHUNK_POLICY_ID = "dpf_transport_exact_divisor_cap3000_v1"
CONTROL_FAMILY_ID = "streaming_ot_sinkhorn_balance_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _parse_positive_candidates(text: str, *, label: str) -> tuple[int, ...]:
    candidates = tuple(int(token) for token in text.split(",") if token.strip())
    if not candidates or any(candidate <= 0 for candidate in candidates):
        raise ValueError(f"{label} candidates must be positive and nonempty")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{label} candidates must be unique")
    return candidates


def _candidate_args(
    *,
    sinkhorn_steps: int,
    balance_steps: int,
    seeds: tuple[int, ...],
    horizon: int,
    num_particles: int,
    attempt_id: str,
    replay_diagnostic: bool,
    campaign_id: str,
    plan_path: str,
) -> Namespace:
    return Namespace(
        time_steps=horizon,
        num_particles=num_particles,
        seeds=",".join(str(seed) for seed in seeds),
        arm="all_active_contract_e",
        balance_steps=balance_steps,
        sinkhorn_steps=sinkhorn_steps,
        dtype="float32",
        kalman_diagnostic=False,
        include_kalman_diagnostic=False,
        include_replay_diagnostic=replay_diagnostic,
        warm_repetitions=1,
        cache_same_cloud_geometry=False,
        attempt_id=attempt_id,
        campaign_id=campaign_id,
        plan_path=plan_path,
    )


def _scope(*, horizon: int, num_particles: int) -> Any:
    from bayesfilter.highdim.ledh_tuning_scope import LEDHTuningScope
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

    chunks = select_transport_chunks(num_particles)

    return LEDHTuningScope(
        model_id=MODEL_ID,
        target_id=TARGET_ID,
        route_id=ROUTE_ID,
        reset_contract_id=RESET_CONTRACT_ID,
        horizon=horizon,
        prepared_data_id=f"lgssm_dataset_seed_81100_observation_prefix_t{horizon}",
        particle_count=num_particles,
        state_dimension=3,
        parameter_count=5,
        dtype="float32",
        tf32_enabled=True,
        jit_compile=True,
        chunk_policy_id=CHUNK_POLICY_ID,
        row_chunk_size=chunks.row_chunk_size,
        col_chunk_size=chunks.col_chunk_size,
        row_blocks=chunks.row_blocks,
        col_blocks=chunks.col_blocks,
        control_family_id=CONTROL_FAMILY_ID,
    )


def _direct_gate(result: dict[str, Any]) -> bool:
    """Apply the declared probability gates, not the internal roundoff gate."""

    work = result.get("work", {})
    expected_steps = int(result["time_steps"])
    identity = result.get("preparation_identity", {})
    device = result.get("device", {})
    graph = result.get("graph", {})
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks

    num_particles = int(result.get("num_particles", -1))
    if num_particles <= 1:
        return False
    chunks = select_transport_chunks(num_particles)
    expected_marginal_sweeps = expected_steps if chunks.row_blocks > 1 else 0
    identity_valid = (
        result.get("sinkhorn_steps") == identity.get("sinkhorn_steps")
        and result.get("balance_steps") == identity.get("balance_steps")
        and identity.get("num_particles") == num_particles
        and identity.get("transport_chunk_policy_id") == chunks.policy_id
        and identity.get("row_chunk_size") == chunks.row_chunk_size
        and identity.get("col_chunk_size") == chunks.col_chunk_size
        and identity.get("transport_block_grid")
        == [chunks.row_blocks, chunks.col_blocks]
        and device.get("dtype") == "float32"
        and device.get("tf32_enabled") is True
        and device.get("jit_compile") is True
    )
    work_valid = (
        work.get("sinkhorn_state_constructions") == expected_steps
        and work.get("terminal_balance_state_constructions") == expected_steps
        and work.get("transport_tile_sweeps") == expected_steps
        and work.get("marginal_tile_sweeps") == expected_marginal_sweeps
        and work.get("diagnostic_solver_reconstructions") == 0
    )
    return bool(
        result.get("finite")
        and (
            result.get("bitwise_replay")
            if result.get("replay_checked")
            else True
        )
        and result.get("chart_valid")
        and result.get("reset_valid")
        and result.get("maximum_tv_column_error", float("inf"))
        <= TV_COLUMN_GATE
        and result.get("maximum_row_error", float("inf")) <= ROW_ERROR_GATE
        and work_valid
        and identity_valid
        and graph.get("python_horizon_unroll") is False
        and "StatelessWhile" in graph.get("while_operation_types", [])
    )


def _node_status(
    *,
    direct_gate_passed: bool,
    elapsed_seconds: float,
    node_cap_seconds: float = NODE_CAP_SECONDS,
) -> str:
    if elapsed_seconds > node_cap_seconds:
        return "FAIL_NODE_CAP"
    return "PASS" if direct_gate_passed else "FAIL_DIRECT_GATE"


def _seed_groups(
    seeds: tuple[int, ...], *, seed_microbatch_size: int
) -> tuple[tuple[int, ...], ...]:
    if seed_microbatch_size <= 0:
        raise ValueError("seed-microbatch-size must be positive")
    return tuple(
        seeds[start : start + seed_microbatch_size]
        for start in range(0, len(seeds), seed_microbatch_size)
    )


def _merge_microbatch_results(
    results: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    seed_groups: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    if not results or len(results) != len(seed_groups):
        raise ValueError("microbatch results must match the declared seed groups")
    for result, group in zip(results, seed_groups, strict=True):
        if result.get("estimator_seeds") != list(group):
            raise ValueError("microbatch result seed order mismatch")

    invariant_names = (
        "git_commit",
        "source_sha256",
        "campaign_id",
        "plan_path",
        "arm",
        "time_steps",
        "num_particles",
        "balance_steps",
        "sinkhorn_steps",
        "cache_same_cloud_geometry",
        "theta",
        "device",
        "graph",
    )
    first = results[0]
    for result in results[1:]:
        for name in invariant_names:
            if result.get(name) != first.get(name):
                raise ValueError(f"microbatch invariant mismatch: {name}")
    per_seed_value = [
        float(value) for result in results for value in result["per_seed_value"]
    ]
    per_seed_score = [
        [float(value) for value in row]
        for result in results
        for row in result["per_seed_physical_score"]
    ]
    if len(per_seed_value) != len(seeds) or len(per_seed_score) != len(seeds):
        raise ValueError("microbatch merge did not preserve every seed")
    aggregate_score = [
        statistics.mean(row[index] for row in per_seed_score)
        for index in range(len(per_seed_score[0]))
    ]
    kalman_values = [result.get("kalman_value") for result in results]
    kalman_scores = [result.get("kalman_physical_score") for result in results]
    include_kalman = kalman_values[0] is not None
    if include_kalman:
        if any(value != kalman_values[0] for value in kalman_values[1:]):
            raise ValueError("microbatch Kalman value mismatch")
        if any(value != kalman_scores[0] for value in kalman_scores[1:]):
            raise ValueError("microbatch Kalman score mismatch")
    elif any(value is not None for value in kalman_values + kalman_scores):
        raise ValueError("microbatch Kalman diagnostics were inconsistently enabled")

    identity = dict(first["preparation_identity"])
    identity["root_seeds_in_order"] = list(seeds)
    identity["tensor_sha256_by_microbatch"] = [
        result["preparation_identity"]["tensor_sha256"] for result in results
    ]
    identity.pop("tensor_sha256", None)
    work_names = tuple(first["work"])
    work = {
        name: sum(int(result["work"][name]) for result in results)
        for name in work_names
    }
    timing_names = ("trace", "compile_plus_first_execution", "warm_execution")
    timing = {}
    for name in timing_names:
        values = [result["timing_seconds"][name] for result in results]
        timing[name] = (
            sum(float(value) for value in values if value is not None)
            if any(value is not None for value in values)
            else None
        )
    timing["by_microbatch"] = [result["timing_seconds"] for result in results]
    allocator_names = tuple(first["gpu_allocator_bytes"])
    allocator = {
        name: max(int(result["gpu_allocator_bytes"][name]) for result in results)
        for name in allocator_names
    }
    maximum_tv = max(float(result["maximum_tv_column_error"]) for result in results)
    maximum_row = max(float(result["maximum_row_error"]) for result in results)
    replay_checked = all(bool(result["replay_checked"]) for result in results)
    bitwise_replay = all(result["bitwise_replay"] is True for result in results)
    merged = {
        **{name: first[name] for name in invariant_names},
        "estimator_seeds": list(seeds),
        "warm_repetitions": first["warm_repetitions"],
        "preparation_identity": identity,
        "per_seed_value": per_seed_value,
        "per_seed_physical_score": per_seed_score,
        "aggregate_value": statistics.mean(per_seed_value),
        "aggregate_physical_score": aggregate_score,
        "kalman_value": kalman_values[0],
        "kalman_physical_score": kalman_scores[0],
        "value_difference_to_kalman": (
            statistics.mean(per_seed_value) - float(kalman_values[0])
            if include_kalman
            else None
        ),
        "physical_score_difference_to_kalman": (
            [
                candidate - oracle
                for candidate, oracle in zip(
                    aggregate_score, kalman_scores[0], strict=True
                )
            ]
            if include_kalman
            else None
        ),
        "finite": all(bool(result["finite"]) for result in results),
        "replay_checked": replay_checked,
        "bitwise_replay": bitwise_replay if replay_checked else None,
        "chart_valid": all(bool(result["chart_valid"]) for result in results),
        "marginal_valid": all(bool(result["marginal_valid"]) for result in results),
        "reset_valid": all(bool(result["reset_valid"]) for result in results),
        "maximum_tv_column_error": maximum_tv,
        "maximum_row_error": maximum_row,
        "tv_column_error_by_seed_time": [
            row for result in results for row in result["tv_column_error_by_seed_time"]
        ],
        "maximum_row_error_by_seed_time": [
            row
            for result in results
            for row in result["maximum_row_error_by_seed_time"]
        ],
        "marginal_valid_by_seed_time": [
            row
            for result in results
            for row in result["marginal_valid_by_seed_time"]
        ],
        "reset_valid_by_seed_time": [
            row for result in results for row in result["reset_valid_by_seed_time"]
        ],
        "chart_valid_by_seed": [
            value for result in results for value in result["chart_valid_by_seed"]
        ],
        "work": work,
        "work_by_microbatch": [result["work"] for result in results],
        "work_valid": all(bool(result["work_valid"]) for result in results),
        "hard_valid": all(bool(result["hard_valid"]) for result in results),
        "timing_seconds": timing,
        "gpu_allocator_bytes": allocator,
        "microbatching": {
            "enabled": len(seed_groups) > 1,
            "compiled_unit": "canonical_prepared_seed_microbatch_value_and_total_score",
            "seed_microbatch_size": max(len(group) for group in seed_groups),
            "seed_groups": [list(group) for group in seed_groups],
            "combination_rule": "ordered per-seed concatenation and exact arithmetic mean",
            "microbatch_count": len(seed_groups),
        },
    }
    return merged


def _record_node(
    *,
    label: str,
    sinkhorn_steps: int,
    balance_steps: int,
    seeds: tuple[int, ...],
    device: dict[str, Any],
    horizon: int,
    scope: Any,
    num_particles: int,
    seed_microbatch_size: int,
    replay_diagnostic: bool = True,
    kalman_diagnostic: bool = False,
    campaign_id: str = CAMPAIGN_ID,
    plan_path: str = PLAN_PATH,
    node_cap_seconds: float = NODE_CAP_SECONDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical

        seed_groups = _seed_groups(
            seeds, seed_microbatch_size=seed_microbatch_size
        )
        compiled_by_batch_size: dict[int, Any] = {}
        results: list[dict[str, Any]] = []
        for microbatch_index, seed_group in enumerate(seed_groups):
            batch_size = len(seed_group)
            compiled = compiled_by_batch_size.get(batch_size)
            if compiled is None:
                compiled = canonical.make_canonical_prepared_value_and_score_tf(
                    batch_size=batch_size,
                    time_steps=horizon,
                    num_particles=num_particles,
                    steps=sinkhorn_steps,
                    balance_steps=balance_steps,
                    row_chunk_size=scope.row_chunk_size,
                    col_chunk_size=scope.col_chunk_size,
                    jit_compile=True,
                    dtype=tf.float32,
                    cache_same_cloud_geometry=False,
                )
                compiled_by_batch_size[batch_size] = compiled
            args = _candidate_args(
                sinkhorn_steps=sinkhorn_steps,
                balance_steps=balance_steps,
                seeds=seed_group,
                horizon=horizon,
                num_particles=num_particles,
                attempt_id=f"{label}-mb{microbatch_index:02d}",
                replay_diagnostic=replay_diagnostic,
                campaign_id=campaign_id,
                plan_path=plan_path,
            )
            args.include_kalman_diagnostic = kalman_diagnostic
            args.kalman_diagnostic = kalman_diagnostic
            results.append(
                runner._execute(
                    args,
                    configured_device=device,
                    compiled_prepared_callable=compiled,
                )
            )
        result = _merge_microbatch_results(
            results,
            seeds=seeds,
            seed_groups=seed_groups,
        )
        elapsed = time.perf_counter() - started
        status = _node_status(
            direct_gate_passed=bool(
                result["hard_valid"]
                and result["maximum_tv_column_error"] <= TV_COLUMN_GATE
                and result["maximum_row_error"] <= ROW_ERROR_GATE
            ),
            elapsed_seconds=elapsed,
            node_cap_seconds=node_cap_seconds,
        )
        node = {
            "label": label,
            "status": status,
            "sinkhorn_steps": sinkhorn_steps,
            "balance_steps": balance_steps,
            "seeds": list(seeds),
            "seed_microbatch_size": seed_microbatch_size,
            "microbatches": [
                {
                    "seeds": list(group),
                    "result": microbatch_result,
                }
                for group, microbatch_result in zip(
                    seed_groups, results, strict=True
                )
            ],
            "result": result,
            "wall_time_seconds": elapsed,
            "within_node_cap": elapsed <= node_cap_seconds,
            "node_cap_seconds": node_cap_seconds,
        }
        node["tuning_scope"] = scope.as_dict()
        node["tuning_scope_sha256"] = scope.scope_sha256
        return node
    except Exception as error:
        message = str(error)
        resource = any(
            token in message.lower()
            for token in ("out of memory", "resource exhausted", "cuda", "xla")
        )
        elapsed = time.perf_counter() - started
        return {
            "label": label,
            "status": "RESOURCE_OR_EXECUTION_FAILURE" if resource else "EXECUTION_FAILURE",
            "sinkhorn_steps": sinkhorn_steps,
            "balance_steps": balance_steps,
            "seeds": list(seeds),
            "seed_microbatch_size": seed_microbatch_size,
            "failure": {
                "type": type(error).__name__,
                "message": message,
                "traceback": traceback.format_exc(),
            },
            "wall_time_seconds": elapsed,
            "within_node_cap": elapsed <= node_cap_seconds,
            "node_cap_seconds": node_cap_seconds,
            "tuning_scope": scope.as_dict(),
            "tuning_scope_sha256": scope.scope_sha256,
        }


def _partition_summary(
    node: dict[str, Any], start: int, stop: int
) -> dict[str, Any]:
    result = node.get("result")
    if result is None:
        return {"status": "NOT_AVAILABLE", "pass": False}
    tv_rows = result["tv_column_error_by_seed_time"][start:stop]
    row_rows = result["maximum_row_error_by_seed_time"][start:stop]
    reset_rows = result["reset_valid_by_seed_time"][start:stop]
    charts = result["chart_valid_by_seed"][start:stop]
    values = result["per_seed_value"][start:stop]
    scores = result["per_seed_physical_score"][start:stop]
    maximum_tv = max(value for row in tv_rows for value in row)
    maximum_row = max(value for row in row_rows for value in row)
    finite = all(math.isfinite(value) for value in values) and all(
        math.isfinite(value) for row in scores for value in row
    )
    passed = bool(
        finite
        and all(charts)
        and all(value for row in reset_rows for value in row)
        and maximum_tv <= TV_COLUMN_GATE
        and maximum_row <= ROW_ERROR_GATE
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "seed_start_index": start,
        "seed_stop_index": stop,
        "maximum_tv_column_error": maximum_tv,
        "maximum_row_error": maximum_row,
        "finite": finite,
        "all_charts_valid": all(charts),
        "all_resets_valid": all(value for row in reset_rows for value in row),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--num-particles", type=int, required=True)
    parser.add_argument("--seed-microbatch-size", type=int, required=True)
    parser.add_argument("--tuning-seed-start", type=int, required=True)
    parser.add_argument("--claim-seed-start", type=int, required=True)
    parser.add_argument("--sinkhorn-candidates", required=True)
    parser.add_argument("--balance-candidates", required=True)
    parser.add_argument("--campaign-id", default=CAMPAIGN_ID)
    parser.add_argument("--plan-path", default=PLAN_PATH)
    parser.add_argument(
        "--node-cap-seconds", type=float, default=NODE_CAP_SECONDS
    )
    parser.add_argument(
        "--total-cap-seconds", type=float, default=TOTAL_CAP_SECONDS
    )
    args = parser.parse_args()
    if min(args.time_steps, args.num_particles, args.seed_microbatch_size) < 1:
        raise ValueError(
            "time-steps, num-particles, and seed-microbatch-size must be positive"
        )
    if args.num_particles <= 1:
        raise ValueError("num-particles must be greater than one")
    if min(args.node_cap_seconds, args.total_cap_seconds) <= 0.0:
        raise ValueError("node and total caps must be positive")
    sinkhorn_candidates = _parse_positive_candidates(
        args.sinkhorn_candidates, label="sinkhorn"
    )
    balance_candidates = _parse_positive_candidates(
        args.balance_candidates, label="balance"
    )
    calibration_seeds = tuple(range(args.tuning_seed_start, args.tuning_seed_start + 8))
    validation_seeds = tuple(range(args.tuning_seed_start + 8, args.tuning_seed_start + 16))
    tuning_seeds = calibration_seeds + validation_seeds
    claim_seeds = tuple(range(args.claim_seed_start, args.claim_seed_start + 16))
    if not set(tuning_seeds).isdisjoint(claim_seeds):
        raise ValueError("tuning and claim seeds must be disjoint")
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": args.campaign_id,
        "attempt_id": args.attempt_id,
        "plan_path": args.plan_path,
        "tuning_scope": None,
        "tuning_scope_sha256": None,
        "status": "RUNNING",
        "selection_metric": {
            "tv_column_tolerance": TV_COLUMN_GATE,
            "maximum_row_error_tolerance": ROW_ERROR_GATE,
            "kalman_used": False,
        },
        "search_order": {
            "sinkhorn_steps": list(sinkhorn_candidates),
            "balance_steps": list(balance_candidates),
            "rule": "exhaust_balance_ladder_before_next_sinkhorn",
        },
        "partitions": {
            "calibration": list(calibration_seeds),
            "validation": list(validation_seeds),
            "claim": list(claim_seeds),
        },
        "num_particles": args.num_particles,
        "seed_microbatch_size": args.seed_microbatch_size,
        "node_cap_seconds": args.node_cap_seconds,
        "total_cap_seconds": args.total_cap_seconds,
        "source_sha256": {
            path: _sha256(ROOT / path)
            for path in (
                "docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py",
                "docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py",
                "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
                "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
                "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
            )
        },
        "git_commit": runner._git_commit(),
        "command": sys.argv,
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpu_required": True,
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
            "gpu_memory_limit_mib": runner.GPU_MEMORY_LIMIT_MIB,
        },
    }
    try:
        device = runner._configure_gpu(tf.float32)
        from bayesfilter.highdim.ledh_tuning_scope import require_scope_match

        scope = _scope(
            horizon=args.time_steps,
            num_particles=args.num_particles,
        )
        payload["device"] = device
        payload["tuning_scope"] = scope.as_dict()
        payload["tuning_scope_sha256"] = scope.scope_sha256
        tuning: list[dict[str, Any]] = []
        selected: tuple[int, int] | None = None
        stop_reason = None
        for sinkhorn_steps in sinkhorn_candidates:
            for balance_steps in balance_candidates:
                if time.perf_counter() - started > args.total_cap_seconds:
                    stop_reason = "total_gpu_budget_exhausted"
                    break
                tuning_node = _record_node(
                    label=f"t{args.time_steps}_s{sinkhorn_steps}_b{balance_steps}_tuning_s16",
                    sinkhorn_steps=sinkhorn_steps,
                    balance_steps=balance_steps,
                    seeds=tuning_seeds,
                    device=device,
                    horizon=args.time_steps,
                    scope=scope,
                    num_particles=args.num_particles,
                    seed_microbatch_size=args.seed_microbatch_size,
                    replay_diagnostic=False,
                    campaign_id=args.campaign_id,
                    plan_path=args.plan_path,
                    node_cap_seconds=args.node_cap_seconds,
                )
                record = {
                    "sinkhorn_steps": sinkhorn_steps,
                    "balance_steps": balance_steps,
                    "tuning_node": tuning_node,
                    "calibration": _partition_summary(
                        tuning_node, 0, len(calibration_seeds)
                    ),
                    "validation": _partition_summary(
                        tuning_node,
                        len(calibration_seeds),
                        len(tuning_seeds),
                    ),
                }
                record["pair_pass"] = bool(
                    tuning_node["status"] == "PASS"
                    and record["calibration"]["pass"]
                    and record["validation"]["pass"]
                )
                tuning.append(record)
                _write_exclusive(output_root / f"candidate_s{sinkhorn_steps}_b{balance_steps}.json", record)
                if record["pair_pass"]:
                    selected = (sinkhorn_steps, balance_steps)
                    break
                if tuning_node["status"] in (
                    "RESOURCE_OR_EXECUTION_FAILURE",
                    "EXECUTION_FAILURE",
                ):
                    stop_reason = "resource_or_execution_veto"
                    break
                gc.collect()
            if selected is not None or stop_reason is not None:
                break
        payload["tuning"] = tuning
        payload["selected_pair"] = (
            {"sinkhorn_steps": selected[0], "balance_steps": selected[1]}
            if selected is not None
            else None
        )
        if selected is None:
            payload["status"] = "NO_PAIR_WITHIN_GRID" if stop_reason is None else stop_reason
        else:
            sinkhorn_steps, balance_steps = selected
            candidate_path = output_root / f"candidate_s{sinkhorn_steps}_b{balance_steps}.json"
            selected_pair_artifact = {
                "schema_version": "bayesfilter.ledh_offline_ot_selection.v1",
                "campaign_id": args.campaign_id,
                "attempt_id": args.attempt_id,
                "tuning_scope": scope.as_dict(),
                "tuning_scope_sha256": scope.scope_sha256,
                "sinkhorn_steps": sinkhorn_steps,
                "balance_steps": balance_steps,
                "selection_rule": "first validation pass after exhausting cheaper balance controls at each sinkhorn rung",
                "selection_metrics": ["TV_col", "E_row"],
                "kalman_used": False,
                "candidate_artifact": str(candidate_path),
                "candidate_artifact_sha256": _sha256(candidate_path),
                "source_sha256": payload["source_sha256"],
                "seed_microbatch_size": args.seed_microbatch_size,
            }
            _write_exclusive(output_root / "selected_pair.json", selected_pair_artifact)
            payload["selected_pair_artifact"] = {
                "path": str(output_root / "selected_pair.json"),
                "sha256": _sha256(output_root / "selected_pair.json"),
            }
            require_scope_match(
                scope, selected_pair_artifact["tuning_scope"], label="selected pair"
            )
            claim = _record_node(
                label=f"t{args.time_steps}_fresh_claim_s16",
                sinkhorn_steps=sinkhorn_steps,
                balance_steps=balance_steps,
                seeds=claim_seeds,
                device=device,
                horizon=args.time_steps,
                scope=scope,
                num_particles=args.num_particles,
                seed_microbatch_size=args.seed_microbatch_size,
                kalman_diagnostic=True,
                campaign_id=args.campaign_id,
                plan_path=args.plan_path,
                node_cap_seconds=args.node_cap_seconds,
            )
            require_scope_match(scope, claim["tuning_scope"], label="claim node")
            _write_exclusive(output_root / f"t{args.time_steps}_fresh_claim_s16.json", claim)
            payload["claim"] = claim
            payload["status"] = "SCOPE_CLAIM_PASS" if claim["status"] == "PASS" else "SCOPE_CLAIM_VETO"
    except Exception as error:
        payload["status"] = "CAMPAIGN_EXECUTION_FAILURE"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["wall_time_seconds"] = time.perf_counter() - started
    _write_exclusive(output_root / "campaign_result.json", payload)
    manifest = {
        "schema_version": "bayesfilter.ledh_offline_ot_tuning_manifest.v1",
        "campaign_id": args.campaign_id,
        "attempt_id": args.attempt_id,
        "git_commit": payload["git_commit"],
        "command": payload["command"],
        "environment": payload["environment"],
        "device": payload.get("device"),
        "seeds": payload["partitions"],
        "wall_time_seconds": payload["wall_time_seconds"],
        "status": payload["status"],
        "plan_path": args.plan_path,
        "result_path": str(output_root / "campaign_result.json"),
        "result_sha256": _sha256(output_root / "campaign_result.json"),
        "artifact_root": str(output_root),
    }
    _write_exclusive(output_root / "run_manifest.json", manifest)
    print(json.dumps({"status": payload["status"], "selected_pair": payload.get("selected_pair"), "wall_time_seconds": payload["wall_time_seconds"]}, indent=2))
    if payload["status"] != "SCOPE_CLAIM_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
