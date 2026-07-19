#!/usr/bin/env python3
"""Execute the final conditional TF32 T=2/10/50 fused LGSSM ladder."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
CAMPAIGN_ID = "canonical-lgssm-tf32-balance-horizon-continuation-20260718"
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-canonical-lgssm-tf32-balance-and-horizon-continuation-plan-2026-07-18.md"
)
SEEDS_BY_HORIZON = {
    2: tuple(range(81500, 81516)),
    10: tuple(range(81520, 81536)),
    50: tuple(range(81540, 81556)),
}
NUM_PARTICLES = 1024
NODE_CAP_SECONDS = {2: 1200.0, 10: 1200.0, 50: 2700.0}


def _node_args(
    *,
    horizon: int,
    seeds: tuple[int, ...],
    arm: str,
    attempt_id: str,
    balance_steps: int,
) -> Namespace:
    return Namespace(
        time_steps=horizon,
        num_particles=NUM_PARTICLES,
        seeds=",".join(str(seed) for seed in seeds),
        arm=arm,
        balance_steps=balance_steps,
        dtype="float32",
        attempt_id=attempt_id,
        campaign_id=CAMPAIGN_ID,
        plan_path=PLAN_PATH,
    )


def _run_node(
    output_root: Path,
    device: dict[str, Any],
    *,
    horizon: int,
    seeds: tuple[int, ...],
    arm: str,
    label: str,
    balance_steps: int,
    attempt_prefix: str,
) -> dict[str, Any]:
    tf.config.experimental.reset_memory_stats("GPU:0")
    started = time.perf_counter()
    node: dict[str, Any] = {
        "schema_version": runner.SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": f"{attempt_prefix}-{label}",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        node["result"] = runner._execute(
            _node_args(
                horizon=horizon,
                seeds=seeds,
                arm=arm,
                attempt_id=node["attempt_id"],
                balance_steps=balance_steps,
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
    float64_path: Path,
    float32_node: dict[str, Any],
    *,
    balance_steps: int,
) -> dict[str, Any]:
    reference_node = json.loads(float64_path.read_text(encoding="utf-8"))
    reference = reference_node["result"]
    candidate = float32_node["result"]
    expected_seeds = list(SEEDS_BY_HORIZON[2])
    reference_identity_valid = (
        reference_node.get("status") == "node_complete"
        and bool(reference.get("hard_valid"))
        and reference.get("campaign_id") == CAMPAIGN_ID
        and reference.get("plan_path") == PLAN_PATH
        and reference.get("time_steps") == 2
        and reference.get("num_particles") == NUM_PARTICLES
        and reference.get("balance_steps") == balance_steps
        and reference.get("estimator_seeds") == expected_seeds
        and reference.get("device", {}).get("dtype") == "float64"
        and reference.get("device", {}).get("tf32_enabled") is False
    )
    candidate_identity_valid = (
        candidate.get("campaign_id") == CAMPAIGN_ID
        and candidate.get("plan_path") == PLAN_PATH
        and candidate.get("time_steps") == 2
        and candidate.get("num_particles") == NUM_PARTICLES
        and candidate.get("balance_steps") == balance_steps
        and candidate.get("estimator_seeds") == expected_seeds
        and candidate.get("device", {}).get("dtype") == "float32"
        and candidate.get("device", {}).get("tf32_enabled") is True
    )
    same_source_sha256 = reference.get("source_sha256") == candidate.get(
        "source_sha256"
    )
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
        and reference_identity_valid
        and candidate_identity_valid
        and same_source_sha256
        and bool(candidate["hard_valid"])
        and sign_consistent
        and no_order_one_coordinate_drift
        and value_not_order_one
    )
    return {
        "pass": passed,
        "float64_source": str(float64_path),
        "same_program_identity": (
            reference_identity_valid and candidate_identity_valid
        ),
        "reference_identity_valid": reference_identity_valid,
        "candidate_identity_valid": candidate_identity_valid,
        "same_source_sha256": same_source_sha256,
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


def _selection_gate(path: Path, balance_steps: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        payload.get("status") == "selection_complete"
        and payload.get("campaign_id") == CAMPAIGN_ID
        and payload.get("selected_balance_steps") == balance_steps
        and payload.get("dtype") == "float32"
        and payload.get("tf32_enabled") is True
        and payload.get("time_steps") == 2
        and payload.get("num_particles") == NUM_PARTICLES
        and payload.get("selection_uses_only_marginals") is True
        and bool(payload.get("design"))
        and bool(payload["design"][-1].get("pass"))
        and bool(payload.get("audit", {}).get("pass"))
    )
    return {
        "pass": valid,
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "attempt_id": payload.get("attempt_id"),
        "selected_balance_steps": payload.get("selected_balance_steps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--float64-t2-reference", type=Path, required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    parser.add_argument("--balance-steps", type=int, required=True)
    parser.add_argument("--attempt-prefix", required=True)
    args = parser.parse_args()
    if args.balance_steps <= 0:
        raise ValueError("balance-steps must be positive")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    summary_path = args.summary_output.resolve()
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite {summary_path}")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "attempt_id": f"{args.attempt_prefix}-combined-tf32-t2-t10-t50",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "balance_steps": args.balance_steps,
        "num_particles": NUM_PARTICLES,
        "nodes": {},
        "stop_reason": None,
        "git_commit": runner._git_commit(),
        "plan_path": PLAN_PATH,
        "command": sys.argv,
    }
    started = time.perf_counter()
    try:
        selection_gate = _selection_gate(
            args.selection_artifact.resolve(), args.balance_steps
        )
        summary["selection_gate"] = selection_gate
        if not selection_gate["pass"]:
            summary["stop_reason"] = "selection_artifact_veto"

        device = runner._configure_gpu(tf.float32)
        summary["device"] = device

        if summary["stop_reason"] is None:
            t2 = _run_node(
                output_root,
                device,
                horizon=2,
                seeds=SEEDS_BY_HORIZON[2],
                arm="all_active_contract_e",
                label="t2_claim_s16_tf32",
                balance_steps=args.balance_steps,
                attempt_prefix=args.attempt_prefix,
            )
            summary["nodes"]["t2_claim_s16_tf32"] = t2
            if not _node_promotes(t2):
                summary["stop_reason"] = "t2_claim_hard_or_resource_veto"
            else:
                precision = _precision_gate(
                    args.float64_t2_reference.resolve(),
                    t2,
                    balance_steps=args.balance_steps,
                )
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
                balance_steps=args.balance_steps,
                attempt_prefix=args.attempt_prefix,
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
                balance_steps=args.balance_steps,
                attempt_prefix=args.attempt_prefix,
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
                balance_steps=args.balance_steps,
                attempt_prefix=args.attempt_prefix,
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
