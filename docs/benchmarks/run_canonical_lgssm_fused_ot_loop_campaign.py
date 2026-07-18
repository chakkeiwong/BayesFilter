#!/usr/bin/env python3
"""Execute the final conditional TF32 T=2/10/50 fused LGSSM ladder."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner


SCHEMA_VERSION = "bayesfilter.canonical_lgssm_fused_ot_loop_campaign.v1"
SEEDS_BY_HORIZON = {
    2: tuple(range(81500, 81516)),
    10: tuple(range(81520, 81536)),
    50: tuple(range(81540, 81556)),
}
BALANCE_STEPS = 2
NUM_PARTICLES = 1024
NODE_CAP_SECONDS = {2: 1200.0, 10: 1200.0, 50: 2700.0}


def _node_args(
    *, horizon: int, seeds: tuple[int, ...], arm: str, attempt_id: str
) -> Namespace:
    return Namespace(
        time_steps=horizon,
        num_particles=NUM_PARTICLES,
        seeds=",".join(str(seed) for seed in seeds),
        arm=arm,
        balance_steps=BALANCE_STEPS,
        dtype="float32",
        attempt_id=attempt_id,
    )


def _run_node(
    output_root: Path,
    device: dict[str, Any],
    *,
    horizon: int,
    seeds: tuple[int, ...],
    arm: str,
    label: str,
) -> dict[str, Any]:
    tf.config.experimental.reset_memory_stats("GPU:0")
    started = time.perf_counter()
    node: dict[str, Any] = {
        "schema_version": runner.SCHEMA_VERSION,
        "campaign_id": runner.CAMPAIGN_ID,
        "attempt_id": f"attempt05-{label}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        node["result"] = runner._execute(
            _node_args(
                horizon=horizon,
                seeds=seeds,
                arm=arm,
                attempt_id=node["attempt_id"],
            ),
            configured_device=device,
        )
        node["status"] = "node_complete"
    except Exception as error:
        node["status"] = "node_failed"
        node["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    node["wall_time_seconds"] = time.perf_counter() - started
    node["within_node_cap"] = node["wall_time_seconds"] <= NODE_CAP_SECONDS[horizon]
    runner._write_exclusive(output_root / f"{label}.json", node)
    gc.collect()
    return node


def _precision_gate(
    float64_path: Path, float32_node: dict[str, Any]
) -> dict[str, Any]:
    reference = json.loads(float64_path.read_text(encoding="utf-8"))["result"]
    candidate = float32_node["result"]
    reference_value = float(reference["aggregate_value"])
    candidate_value = float(candidate["aggregate_value"])
    reference_score = [float(item) for item in reference["aggregate_physical_score"]]
    candidate_score = [float(item) for item in candidate["aggregate_physical_score"]]
    value_drift = candidate_value - reference_value
    score_drift = [
        candidate_item - reference_item
        for candidate_item, reference_item in zip(
            candidate_score, reference_score, strict=True
        )
    ]
    sign_consistent = all(
        reference_item == 0.0
        or candidate_item == 0.0
        or math.copysign(1.0, reference_item) == math.copysign(1.0, candidate_item)
        for reference_item, candidate_item in zip(
            reference_score, candidate_score, strict=True
        )
    )
    no_order_one_coordinate_drift = all(
        abs(drift) < abs(reference_item)
        for drift, reference_item in zip(score_drift, reference_score, strict=True)
        if reference_item != 0.0
    )
    value_not_order_one = abs(value_drift) < abs(reference_value)
    passed = (
        float32_node["status"] == "node_complete"
        and bool(candidate["hard_valid"])
        and sign_consistent
        and no_order_one_coordinate_drift
        and value_not_order_one
    )
    return {
        "pass": passed,
        "float64_source": str(float64_path),
        "value_drift": value_drift,
        "score_drift": score_drift,
        "sign_consistent": sign_consistent,
        "no_order_one_coordinate_drift": no_order_one_coordinate_drift,
        "value_not_order_one": value_not_order_one,
        "interpretation": "hard-veto precision screen; continuous drift remains descriptive",
    }


def _node_promotes(node: dict[str, Any]) -> bool:
    return (
        node["status"] == "node_complete"
        and node["within_node_cap"]
        and bool(node["result"]["hard_valid"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--float64-t2-reference", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = args.summary_output.resolve()
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": runner.CAMPAIGN_ID,
        "attempt_id": "attempt05-combined-tf32-t2-t10-t50",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "balance_steps": BALANCE_STEPS,
        "num_particles": NUM_PARTICLES,
        "nodes": {},
        "stop_reason": None,
    }
    started = time.perf_counter()
    try:
        device = runner._configure_gpu(tf.float32)
        summary["device"] = device

        t2 = _run_node(
            output_root,
            device,
            horizon=2,
            seeds=SEEDS_BY_HORIZON[2],
            arm="all_active_contract_e",
            label="t2_claim_s16_tf32",
        )
        summary["nodes"]["t2_claim_s16_tf32"] = t2
        if not _node_promotes(t2):
            summary["stop_reason"] = "t2_claim_hard_or_resource_veto"
        else:
            precision = _precision_gate(args.float64_t2_reference.resolve(), t2)
            summary["t2_precision_gate"] = precision
            if not precision["pass"]:
                summary["stop_reason"] = "t2_tf32_precision_veto"

        if summary["stop_reason"] is None:
            inactive = _run_node(
                output_root,
                device,
                horizon=2,
                seeds=(SEEDS_BY_HORIZON[2][0],),
                arm="no_reset_weighted",
                label="t2_inactive_resource_s1_tf32",
            )
            summary["nodes"]["t2_inactive_resource_s1_tf32"] = inactive
            if not _node_promotes(inactive) or any(
                inactive["result"]["work"][name] != 0
                for name in (
                    "sinkhorn_state_constructions",
                    "terminal_balance_state_constructions",
                    "transport_tile_sweeps",
                    "marginal_tile_sweeps",
                    "diagnostic_solver_reconstructions",
                )
            ):
                summary["stop_reason"] = "t2_inactive_zero_ot_veto"

        for horizon in (10, 50):
            if summary["stop_reason"] is not None:
                break
            witness = _run_node(
                output_root,
                device,
                horizon=horizon,
                seeds=(SEEDS_BY_HORIZON[horizon][0],),
                arm="all_active_contract_e",
                label=f"t{horizon}_resource_s1_tf32",
            )
            summary["nodes"][f"t{horizon}_resource_s1_tf32"] = witness
            if not _node_promotes(witness):
                summary["stop_reason"] = f"t{horizon}_resource_veto"
                break
            claim = _run_node(
                output_root,
                device,
                horizon=horizon,
                seeds=SEEDS_BY_HORIZON[horizon],
                arm="all_active_contract_e",
                label=f"t{horizon}_claim_s16_tf32",
            )
            summary["nodes"][f"t{horizon}_claim_s16_tf32"] = claim
            if not _node_promotes(claim):
                summary["stop_reason"] = f"t{horizon}_claim_veto"
                break

        summary["status"] = (
            "campaign_complete"
            if summary["stop_reason"] is None
            else "campaign_stopped_on_declared_veto"
        )
    except Exception as error:
        summary["status"] = "campaign_execution_failed"
        summary["stop_reason"] = "supervisor_exception"
        summary["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    summary["wall_time_seconds"] = time.perf_counter() - started
    runner._write_exclusive(summary_path, summary)
    print(json.dumps({
        "status": summary["status"],
        "stop_reason": summary["stop_reason"],
        "nodes": list(summary["nodes"]),
        "wall_time_seconds": summary["wall_time_seconds"],
    }, indent=2))
    if summary["status"] != "campaign_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
