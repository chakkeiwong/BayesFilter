#!/usr/bin/env python3
"""Aggregate the paired N=5000/N=10000 single-seed Kalman diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SEED = 82220
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-lgssm-n10000-single-seed-kalman-diagnostic-plan-2026-07-20.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _relative_error(candidate: float, oracle: float) -> float | None:
    if oracle == 0.0:
        return None
    return (candidate - oracle) / oracle


def _comparison(candidate: float, baseline: float, oracle: float) -> dict[str, Any]:
    candidate_error = candidate - oracle
    baseline_error = baseline - oracle
    candidate_absolute = abs(candidate_error)
    baseline_absolute = abs(baseline_error)
    return {
        "kalman": oracle,
        "n5000": baseline,
        "n10000": candidate,
        "n5000_error": baseline_error,
        "n10000_error": candidate_error,
        "n5000_relative_error": _relative_error(baseline, oracle),
        "n10000_relative_error": _relative_error(candidate, oracle),
        "absolute_error_change_n10000_minus_n5000": (
            candidate_absolute - baseline_absolute
        ),
        "n10000_descriptively_closer": candidate_absolute < baseline_absolute,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    node_payload = json.loads(args.node.read_text(encoding="utf-8"))
    if node_payload.get("status") != "node_complete":
        raise ValueError(f"N=10000 node is not complete: {node_payload.get('status')}")
    node = node_payload["result"]
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    claim = baseline_payload["claim"]
    baseline_nodes = [
        item for item in claim["microbatches"] if item.get("seeds") == [SEED]
    ]
    if len(baseline_nodes) != 1:
        raise ValueError("expected exactly one N=5000 singleton baseline node")
    baseline = baseline_nodes[0]["result"]

    required_n10000 = {
        "time_steps": 50,
        "num_particles": 10000,
        "estimator_seeds": [SEED],
        "sinkhorn_steps": 20,
        "balance_steps": 5,
        "hard_valid": True,
        "bitwise_replay": True,
    }
    mismatches = {
        name: {"expected": expected, "actual": node.get(name)}
        for name, expected in required_n10000.items()
        if node.get(name) != expected
    }
    identity = node["preparation_identity"]
    for name, expected in {
        "row_chunk_size": 2500,
        "col_chunk_size": 2500,
        "transport_block_grid": [4, 4],
    }.items():
        if identity.get(name) != expected:
            mismatches[f"identity.{name}"] = {
                "expected": expected,
                "actual": identity.get(name),
            }
    if mismatches:
        raise ValueError(f"N=10000 node contract mismatch: {mismatches}")
    if baseline.get("estimator_seeds") != [SEED] or not baseline.get("hard_valid"):
        raise ValueError("N=5000 baseline is not the valid paired singleton")
    kalman_value = float(node["kalman_value"])
    kalman_score = [float(value) for value in node["kalman_physical_score"]]
    if baseline.get("kalman_value") != node.get("kalman_value"):
        raise ValueError("paired nodes disagree on Kalman value")
    if baseline.get("kalman_physical_score") != node.get("kalman_physical_score"):
        raise ValueError("paired nodes disagree on Kalman score")
    n10000_value = float(node["per_seed_value"][0])
    n5000_value = float(baseline["per_seed_value"][0])
    n10000_score = [float(value) for value in node["per_seed_physical_score"][0]]
    n5000_score = [float(value) for value in baseline["per_seed_physical_score"][0]]
    comparisons = {
        "value": _comparison(n10000_value, n5000_value, kalman_value),
        "score": {
            name: _comparison(candidate, prior, oracle)
            for name, candidate, prior, oracle in zip(
                PARAMETER_NAMES,
                n10000_score,
                n5000_score,
                kalman_score,
                strict=True,
            )
        },
    }
    all_numbers = [n10000_value, n5000_value, kalman_value]
    all_numbers.extend(n10000_score + n5000_score + kalman_score)
    if not all(math.isfinite(value) for value in all_numbers):
        raise ValueError("comparison contains a non-finite value")
    closer_outputs = [
        name
        for name, result in {
            "value": comparisons["value"],
            **comparisons["score"],
        }.items()
        if result["n10000_descriptively_closer"]
    ]
    payload = {
        "schema_version": "bayesfilter.lgssm_n10000_single_seed_kalman.v1",
        "status": "DIAGNOSTIC_COMPLETE",
        "plan_path": PLAN_PATH,
        "seed": SEED,
        "nonclaims": [
            "one seed does not estimate bias",
            "controls (20,5) are not tuned for N=10000",
            "no statistical ranking or monotonic convergence claim",
            "no HMC or default-readiness claim",
        ],
        "n10000_node": {
            "path": str(args.node),
            "sha256": _sha256(args.node),
            "hard_valid": node["hard_valid"],
            "gpu_allocator_bytes": node["gpu_allocator_bytes"],
            "timing_seconds": node["timing_seconds"],
            "maximum_tv_column_error": node["maximum_tv_column_error"],
            "maximum_row_error": node["maximum_row_error"],
        },
        "n5000_baseline": {
            "path": str(args.baseline),
            "sha256": _sha256(args.baseline),
        },
        "comparisons": comparisons,
        "descriptively_closer_output_count": len(closer_outputs),
        "descriptively_closer_outputs": closer_outputs,
    }
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "descriptively_closer_outputs": closer_outputs,
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

