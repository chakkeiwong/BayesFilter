#!/usr/bin/env python3
"""Build a read-only T=50 comparison table for the shared LGSSM metric.

This consumes preserved aggregate artifacts. It does not rerun TensorFlow or
modify historical results. The comparison is descriptive because the arms use
different particle counts, controls, execution backends, and random-number
construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
CRITICAL_VALUE = 3.036283222821165
VALUE_MARGIN = 0.001
SCORE_MARGIN = 0.05


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _screen(intervals: dict[str, dict[str, float]], hard_valid: bool) -> str:
    margins = (VALUE_MARGIN,) + (SCORE_MARGIN,) * 5
    ordered = [intervals[label] for label in LABELS]
    if not hard_valid:
        return "screen_fail"
    if all(
        item["lower"] >= -margin and item["upper"] <= margin
        for item, margin in zip(ordered, margins, strict=True)
    ):
        return "screen_pass"
    if any(
        item["lower"] > margin or item["upper"] < -margin
        for item, margin in zip(ordered, margins, strict=True)
    ):
        return "screen_fail"
    return "inconclusive"


def _scope_row(
    *,
    name: str,
    artifact: Path,
    aggregate: dict[str, Any],
    scope: dict[str, Any],
    intervals: dict[str, dict[str, float]],
    hard_valid: bool,
    controls: dict[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    return {
        "name": name,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": _sha256(artifact),
        "num_particles": scope["particle_count"],
        "controls": controls,
        "seeds": seeds,
        "hard_valid": hard_valid,
        "screen": _screen(intervals, hard_valid),
        "relative_error_intervals": intervals,
        "scope": scope,
        "aggregate_source": aggregate,
    }


def build(*, n5000_path: Path, n10000_path: Path, new_path: Path) -> dict[str, Any]:
    n5000 = _load(n5000_path)
    n5000_scope = n5000["scopes"]["5000"]
    n10000 = _load(n10000_path)
    n10000_scope = n10000["n10000"]
    new = _load(new_path)
    new_summary = next(
        item
        for item in new["summaries"]
        if item["horizon"] == 50 and item["method"] == "cubature"
    )

    new_controls = new["configuration"].get("controls", new["configuration"])
    rows = [
        _scope_row(
            name="Contract E N=5000",
            artifact=n5000_path,
            aggregate=n5000,
            scope=n5000_scope["tuning_scope"],
            intervals=n5000_scope["relative_error_intervals"],
            hard_valid=n5000_scope["hard_valid"],
            controls=n5000_scope["controls"],
            seeds=n5000_scope["partitions"]["claim"],
        ),
        _scope_row(
            name="Contract E N=10000",
            artifact=n10000_path,
            aggregate=n10000,
            scope=n10000_scope["tuning_scope"],
            intervals=n10000_scope["relative_error_intervals"],
            hard_valid=n10000_scope["hard_valid"],
            controls=n10000_scope["controls"],
            seeds=n10000_scope["partitions"]["claim"],
        ),
        _scope_row(
            name="Cubature/GenUT N=1008",
            artifact=new_path,
            aggregate=new,
            scope={
                "particle_count": new["configuration"]["num_particles"],
                "horizon": 50,
                "dataset_seed": new["configuration"]["dataset_seed"],
                "reset_design": "cubature_or_gaussian_genut",
                "dtype": new["configuration"]["dtype"],
                "tf32_enabled": new["configuration"]["tf32_mode"] == "enabled",
                "jit_compile": new["configuration"]["jit_compile"],
            },
            intervals=new_summary["relative_error_intervals"],
            hard_valid=new["hard_valid"],
            controls={
                "epsilon": new_controls["epsilon"],
                "sinkhorn_steps": new_controls["sinkhorn_steps"],
                "ridge": new_controls["ridge"],
            },
            seeds=new["configuration"].get(
                "claim_seeds", new["configuration"].get("particle_seeds", [])
            ),
        ),
    ]
    target_match = all(
        row["scope"].get("dataset_seed", 81100) == 81100 for row in rows
    )
    return {
        "schema_version": "bayesfilter.lgssm_shared_metric_comparison.v1",
        "comparison_target": {
            "horizon": 50,
            "labels": list(LABELS),
            "critical_value": CRITICAL_VALUE,
            "value_margin": VALUE_MARGIN,
            "hmc_score_margin": SCORE_MARGIN,
            "dataset_seed": 81100,
            "target_match": target_match,
            "metric": "HMC-transformed six-coordinate relative error with simultaneous CI",
        },
        "rows": rows,
        "comparability": {
            "metric_identical": True,
            "target_identical": target_match,
            "paired_common_random_numbers": False,
            "reason_not_paired": (
                "The prior Contract E preparations use Philox keys [seed, domain_tag] "
                "and float64 raw draws cast to float32; Cubature/GenUT uses stateless "
                "keys [seed, horizon] and [seed, horizon+100]. Different N also changes "
                "the draw shape. Reusing seed labels is not a paired CRN design."
            ),
            "algorithm_controls_match": False,
            "control_difference": (
                "Prior tuned arms use epsilon=0.5, sinkhorn_steps=20, balance_steps "
                "5 or 8, XLA, and Contract E-Chol. The new diagnostic uses epsilon=2, "
                "sinkhorn_steps=8, no JIT, and Cubature/GenUT residual injection."
            ),
            "interpretation": (
                "The table supports descriptive placement on the same error scale. "
                "It does not support a causal or statistically paired improvement claim."
            ),
        },
        "nonclaims": [
            "no method superiority claim",
            "no paired cross-arm hypothesis test",
            "no 1/N convergence claim",
            "no exact filtering or nonlinear-model claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n5000", type=Path, required=True)
    parser.add_argument("--n10000", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build(
        n5000_path=args.n5000.resolve(),
        n10000_path=args.n10000.resolve(),
        new_path=args.new.resolve(),
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": [row["name"] for row in payload["rows"]]}, indent=2))


if __name__ == "__main__":
    main()
