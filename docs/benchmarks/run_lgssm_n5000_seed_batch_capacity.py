#!/usr/bin/env python3
"""Measure eight-seed capacity for the canonical N=5000 LGSSM score route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
from docs.benchmarks import run_ledh_offline_ot_tuning_campaign as campaign


SCHEMA_VERSION = "bayesfilter.lgssm_n5000_seed_batch_capacity.v1"
CAMPAIGN_ID = "lgssm-n5000-seed-batch-capacity-20260720"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-lgssm-n5000-seed-batch-capacity-plan-2026-07-20.md"
)
BASELINE_PATH = (
    "docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/"
    "n5000_repair_scope_attempt01/campaign_result.json"
)
SEEDS = tuple(range(82220, 82228))
SINKHORN_STEPS = 20
BALANCE_STEPS = 5
PARITY_ATOL = 1.0e-4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_baseline(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_pair")
    expected_pair = {
        "sinkhorn_steps": SINKHORN_STEPS,
        "balance_steps": BALANCE_STEPS,
    }
    if selected != expected_pair:
        raise ValueError(f"baseline selected controls mismatch: {selected}")
    claim = payload.get("claim", {})
    microbatches = claim.get("microbatches", [])[: len(SEEDS)]
    observed_seeds = tuple(
        int(seed)
        for microbatch in microbatches
        for seed in microbatch.get("seeds", [])
    )
    if observed_seeds != SEEDS or any(
        len(item.get("seeds", [])) != 1 for item in microbatches
    ):
        raise ValueError(f"baseline singleton seeds mismatch: {observed_seeds}")
    if not all(
        item.get("result", {}).get("hard_valid") is True for item in microbatches
    ):
        raise ValueError("baseline contains a failed singleton hard gate")
    source_hashes = dict(payload.get("source_sha256", {}))
    for item in microbatches:
        for relative, expected in item["result"].get("source_sha256", {}).items():
            prior = source_hashes.setdefault(relative, expected)
            if prior != expected:
                raise ValueError(
                    f"baseline nodes disagree on source hash for {relative}"
                )
    for relative, expected in source_hashes.items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(
                f"baseline source hash mismatch for {relative}: {actual} != {expected}"
            )
    return payload, microbatches


def _baseline_timing(microbatches: list[dict[str, Any]]) -> dict[str, float]:
    trace = sum(
        float(item["result"]["timing_seconds"]["trace"])
        for item in microbatches
    )
    cold = sum(
        float(item["result"]["timing_seconds"]["compile_plus_first_execution"])
        for item in microbatches
    )
    warm = sum(
        float(item["result"]["timing_seconds"]["warm_execution"])
        for item in microbatches
    )
    return {
        "trace_seconds": trace,
        "compile_plus_first_execution_seconds": cold,
        "warm_replay_seconds": warm,
        "trace_cold_and_replay_seconds": trace + cold + warm,
    }


def _parity(
    batched: dict[str, Any], baseline_microbatches: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline_values = [
        float(item["result"]["per_seed_value"][0])
        for item in baseline_microbatches
    ]
    baseline_scores = [
        [float(value) for value in item["result"]["per_seed_physical_score"][0]]
        for item in baseline_microbatches
    ]
    batch_values = [float(value) for value in batched["per_seed_value"]]
    batch_scores = [
        [float(value) for value in row]
        for row in batched["per_seed_physical_score"]
    ]
    if len(batch_values) != len(baseline_values) or len(batch_scores) != len(
        baseline_scores
    ):
        raise ValueError("batched result did not preserve all baseline seeds")
    value_differences = [
        abs(candidate - reference)
        for candidate, reference in zip(batch_values, baseline_values, strict=True)
    ]
    score_differences = [
        [
            abs(candidate - reference)
            for candidate, reference in zip(
                candidate_row, reference_row, strict=True
            )
        ]
        for candidate_row, reference_row in zip(
            batch_scores, baseline_scores, strict=True
        )
    ]
    maximum_value = max(value_differences)
    maximum_score = max(value for row in score_differences for value in row)
    return {
        "absolute_tolerance": PARITY_ATOL,
        "maximum_absolute_value_difference": maximum_value,
        "maximum_absolute_score_difference": maximum_score,
        "value_difference_by_seed": value_differences,
        "score_difference_by_seed_parameter": score_differences,
        "pass": bool(
            maximum_value <= PARITY_ATOL and maximum_score <= PARITY_ATOL
        ),
    }


def _speed(
    batched: dict[str, Any], baseline_timing: dict[str, float]
) -> dict[str, Any]:
    timing = batched["timing_seconds"]
    trace = float(timing["trace"])
    cold = float(timing["compile_plus_first_execution"])
    warm = float(timing["warm_execution"])
    total = trace + cold + warm
    return {
        "equal_seed_count": len(SEEDS),
        "singleton_baseline": baseline_timing,
        "batch8": {
            "trace_seconds": trace,
            "compile_plus_first_execution_seconds": cold,
            "warm_replay_seconds": warm,
            "trace_cold_and_replay_seconds": total,
        },
        "cold_execution_speedup": baseline_timing[
            "compile_plus_first_execution_seconds"
        ]
        / cold,
        "warm_execution_speedup": baseline_timing["warm_replay_seconds"] / warm,
        "trace_cold_and_replay_speedup": baseline_timing[
            "trace_cold_and_replay_seconds"
        ]
        / total,
        "batch8_warm_seeds_per_second": len(SEEDS) / warm,
        "singleton_warm_seeds_per_second": len(SEEDS)
        / baseline_timing["warm_replay_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--timing-context",
        choices=("uncontended", "externally_contended"),
        required=True,
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")

    baseline_path = ROOT / BASELINE_PATH
    baseline, baseline_microbatches = _load_baseline(baseline_path)
    baseline_timing = _baseline_timing(baseline_microbatches)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": args.attempt_id,
        "plan_path": PLAN_PATH,
        "status": "RUNNING",
        "question": (
            "N=5000 canonical total-score capacity and speed at seed "
            "microbatch size eight"
        ),
        "timing_context": {
            "classification": args.timing_context,
            "clean_speed_comparison": args.timing_context == "uncontended",
            "interpretation": (
                "clean equal-work timing comparison"
                if args.timing_context == "uncontended"
                else "observed throughput under external GPU contention; not a clean speed ratio"
            ),
        },
        "baseline": {
            "path": BASELINE_PATH,
            "sha256": _sha256(baseline_path),
            "attempt_id": baseline["attempt_id"],
            "source_sha256": baseline["source_sha256"],
            "seed_microbatch_size": 1,
            "seeds": list(SEEDS),
            "timing": baseline_timing,
        },
        "candidate": {
            "time_steps": 50,
            "num_particles": 5000,
            "row_chunk_size": 2500,
            "col_chunk_size": 2500,
            "transport_block_grid": [2, 2],
            "sinkhorn_steps": SINKHORN_STEPS,
            "balance_steps": BALANCE_STEPS,
            "seed_microbatch_size": len(SEEDS),
            "seeds": list(SEEDS),
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
            "score": "canonical_total_score",
        },
        "command": sys.argv,
        "git_commit": runner._git_commit(),
        "environment": {
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpu_memory_limit_mib": runner.GPU_MEMORY_LIMIT_MIB,
        },
    }
    try:
        device = runner._configure_gpu(tf.float32)
        tf.config.experimental.reset_memory_stats("GPU:0")
        scope = campaign._scope(horizon=50, num_particles=5000)
        node = campaign._record_node(
            label="t50_n5000_s20_b5_seed_batch8_capacity",
            sinkhorn_steps=SINKHORN_STEPS,
            balance_steps=BALANCE_STEPS,
            seeds=SEEDS,
            device=device,
            horizon=50,
            scope=scope,
            num_particles=5000,
            seed_microbatch_size=len(SEEDS),
            replay_diagnostic=True,
            kalman_diagnostic=False,
            campaign_id=CAMPAIGN_ID,
            plan_path=PLAN_PATH,
        )
        payload["device"] = device
        payload["node"] = node
        if node.get("status") != "PASS":
            raise RuntimeError(
                f"size-eight canonical node failed: {node.get('status')}"
            )
        result = node["result"]
        parity = _parity(result, baseline_microbatches)
        speed = _speed(result, baseline_timing)
        peak_bytes = int(result["gpu_allocator_bytes"]["peak"])
        memory = {
            "peak_allocator_bytes": peak_bytes,
            "peak_allocator_mib": peak_bytes / (1024.0 * 1024.0),
            "logical_device_limit_mib": runner.GPU_MEMORY_LIMIT_MIB,
            "fraction_of_limit": peak_bytes
            / (runner.GPU_MEMORY_LIMIT_MIB * 1024.0 * 1024.0),
            "below_limit": peak_bytes
            < runner.GPU_MEMORY_LIMIT_MIB * 1024 * 1024,
        }
        payload["parity"] = parity
        payload["speed"] = speed
        payload["speed"]["clean_comparison"] = (
            args.timing_context == "uncontended"
        )
        payload["memory"] = memory
        payload["status"] = (
            "PASS_BATCH8_NOMINATED"
            if parity["pass"] and memory["below_limit"] and result["hard_valid"]
            else "FAIL_BATCH8_NOT_NOMINATED"
        )
    except Exception as error:
        payload["status"] = "EXECUTION_FAILURE"
        payload["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["wall_time_seconds"] = time.perf_counter() - started
    result_path = output_root / "result.json"
    _write_exclusive(result_path, payload)
    manifest = {
        "schema_version": (
            "bayesfilter.lgssm_n5000_seed_batch_capacity_manifest.v1"
        ),
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": args.attempt_id,
        "git_commit": payload["git_commit"],
        "command": payload["command"],
        "environment": payload["environment"],
        "device": payload.get("device"),
        "seeds": list(SEEDS),
        "wall_time_seconds": payload["wall_time_seconds"],
        "status": payload["status"],
        "plan_path": PLAN_PATH,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
    }
    _write_exclusive(output_root / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "memory": payload.get("memory"),
                "speed": payload.get("speed"),
                "parity": payload.get("parity"),
                "wall_time_seconds": payload["wall_time_seconds"],
            },
            indent=2,
        )
    )
    if payload["status"] != "PASS_BATCH8_NOMINATED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
