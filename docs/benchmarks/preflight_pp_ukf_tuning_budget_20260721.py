#!/usr/bin/env python3
"""Prospective PP-UKF tuning resource preflight.

This is a planning diagnostic. It reads measured target cost and an explicitly
recorded prior transition observation, then emits a bounded decision. It does
not import TensorFlow, launch tuning, or authorize posterior sampling.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observed-transitions", type=int, default=100)
    parser.add_argument("--observed-wall-seconds", type=float, default=47.0 * 60.0)
    parser.add_argument("--planned-transitions", type=int, default=1000)
    parser.add_argument("--campaign-cap-seconds", type=float, default=4.0 * 3600.0)
    parser.add_argument("--canary-cap-seconds", type=float, default=30.0 * 60.0)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    if args.observed_transitions <= 0 or args.planned_transitions <= 0:
        raise ValueError("transition counts must be positive")
    if args.observed_wall_seconds <= 0.0 or args.campaign_cap_seconds <= 0.0:
        raise ValueError("wall-time values must be positive")
    transition_rate = float(args.observed_transitions) / float(args.observed_wall_seconds)
    transition_seconds = float(args.planned_transitions) / transition_rate
    retained_batch_mean = float(
        benchmark["timings"]["retained_flat_value_score_status"]["mean_seconds"]
    )
    retained_batches = (args.planned_transitions + 63) // 64
    retained_health_seconds = retained_batches * retained_batch_mean
    bootstrap_seconds = float(args.observed_wall_seconds) * 0.15
    phase_overhead_seconds = max(300.0, 0.10 * transition_seconds)
    projected = transition_seconds + retained_health_seconds + bootstrap_seconds + phase_overhead_seconds
    margin_target = 0.75 * float(args.campaign_cap_seconds)
    row = {
        "schema": "bayesfilter.pp_ukf_tuning_budget_preflight.v1",
        "role": "prospective_resource_gate_only",
        "timestamp_unix": time.time(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "benchmark_path": str(args.benchmark),
        "benchmark_schema": benchmark.get("schema"),
        "target_signature": benchmark.get("target_signature"),
        "transport_sha256": benchmark.get("transport_sha256"),
        "inputs": {
            "observed_transitions": args.observed_transitions,
            "observed_wall_seconds": args.observed_wall_seconds,
            "planned_transitions": args.planned_transitions,
            "campaign_cap_seconds": args.campaign_cap_seconds,
            "canary_cap_seconds": args.canary_cap_seconds,
        },
        "measured": {
            "retained_flat_batch_mean_seconds": retained_batch_mean,
            "retained_flat_batch_draws": 64,
            "observed_transition_rate_per_second": transition_rate,
        },
        "projection": {
            "transition_seconds": transition_seconds,
            "retained_health_seconds": retained_health_seconds,
            "bootstrap_seconds_allowance": bootstrap_seconds,
            "phase_overhead_seconds": phase_overhead_seconds,
            "full_tuning_projected_wall_seconds": projected,
            "full_tuning_projected_wall_hours": projected / 3600.0,
            "promotion_margin_seconds": margin_target,
        },
        "decision": {
            "phase4_target_cost_gate": "PASS" if benchmark.get("parity", {}).get("flat_value_near") else "VETO",
            "phase7_canary_authorized": bool(args.canary_cap_seconds <= args.campaign_cap_seconds),
            "phase8_full_tuning_authorized": bool(projected <= margin_target),
            "phase8_resource_veto": bool(projected > margin_target),
            "stop_reason": (
                "projected_full_tuning_exceeds_75_percent_campaign_cap"
                if projected > margin_target
                else None
            ),
        },
        "nonclaims": [
            "no tuning result",
            "no posterior or convergence claim",
            "projection is not a performance ranking",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"preflight output must be fresh: {args.output}")
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(row["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
