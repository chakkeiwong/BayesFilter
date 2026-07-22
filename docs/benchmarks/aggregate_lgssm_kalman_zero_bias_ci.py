#!/usr/bin/env python3
"""Reclassify preserved LGSSM claims using a zero-bias CI criterion.

The historical +/-5% score band remains in the original artifacts. This
diagnostic asks the separate question whether the confidence interval for the
mean relative bias contains zero. The simultaneous interval is primary because
six outputs are inspected together; the ordinary coordinate-wise interval is
reported for context.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
DF = 15
ORDINARY_T_CRITICAL = 2.131449545559776
SIMULTANEOUS_T_CRITICAL = 3.036283222821165


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _interval(mean: float, standard_error: float, critical: float) -> dict[str, float]:
    radius = critical * standard_error
    return {"lower": mean - radius, "upper": mean + radius}


def _scope_zero_bias(scope: dict[str, Any]) -> dict[str, Any]:
    intervals = scope["relative_error_intervals"]
    outputs = {}
    for label in LABELS:
        source = intervals[label]
        ordinary = _interval(
            float(source["mean"]),
            float(source["standard_error"]),
            ORDINARY_T_CRITICAL,
        )
        simultaneous = _interval(
            float(source["mean"]),
            float(source["standard_error"]),
            SIMULTANEOUS_T_CRITICAL,
        )
        outputs[label] = {
            "mean_relative_bias": float(source["mean"]),
            "standard_error": float(source["standard_error"]),
            "ordinary_95_ci": ordinary,
            "simultaneous_95_ci": simultaneous,
            "ordinary_zero_bias_not_rejected": ordinary["lower"] <= 0.0 <= ordinary["upper"],
            "simultaneous_zero_bias_not_rejected": simultaneous["lower"]
            <= 0.0
            <= simultaneous["upper"],
        }
    screen = all(
        item["simultaneous_zero_bias_not_rejected"] for item in outputs.values()
    )
    return {
        "num_particles": scope["num_particles"],
        "engineering_hard_valid": scope["hard_valid"],
        "historical_screen": scope["screen"],
        "outputs": outputs,
        "simultaneous_zero_bias_screen": "not_rejected" if screen else "rejected",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n5000-aggregate", type=Path, required=True)
    parser.add_argument("--n10000-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n5000_path = args.n5000_aggregate.resolve()
    n10000_path = args.n10000_aggregate.resolve()
    n5000_payload = _load(n5000_path)
    n10000_payload = _load(n10000_path)
    n5000 = n5000_payload["scopes"]["5000"]
    n10000 = n10000_payload["n10000"]
    for scope in (n5000, n10000):
        if scope["binding"]["all_valid"] is not True:
            raise ValueError("zero-bias reclassification requires bound scopes")
        if scope["hard_valid"] is not True:
            raise ValueError("zero-bias reclassification requires engineering-valid scopes")

    scopes = {"5000": _scope_zero_bias(n5000), "10000": _scope_zero_bias(n10000)}
    payload = {
        "schema_version": "bayesfilter.lgssm_kalman_zero_bias_ci.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_reclassification_complete",
        "criterion": {
            "primary": "simultaneous_95_percent_ci_for_each_mean_relative_bias_contains_zero",
            "family_size": 6,
            "degrees_of_freedom": DF,
            "simultaneous_t_critical": SIMULTANEOUS_T_CRITICAL,
            "ordinary_coordinatewise_t_critical": ORDINARY_T_CRITICAL,
            "interpretation": "failure to reject zero bias is not proof of equivalence",
            "historical_plus_minus_5_percent_band_preserved": True,
        },
        "scopes": scopes,
        "q_scale_diagnosis": {
            "N5000_simultaneous_zero_bias": scopes["5000"]["outputs"]["q_scale"][
                "simultaneous_zero_bias_not_rejected"
            ],
            "N10000_simultaneous_zero_bias": scopes["10000"]["outputs"]["q_scale"][
                "simultaneous_zero_bias_not_rejected"
            ],
            "interpretation": (
                "q_scale remains a detected mean-bias coordinate under the revised "
                "zero-bias test; its long-horizon Kalman increments are cancellation "
                "sensitive and q enters both initial and transition covariance scales."
            ),
        },
        "nonclaims": [
            "not an equivalence proof",
            "not a scientifically justified practical margin",
            "not a paired cross-particle ranking",
            "not HMC or posterior readiness",
        ],
        "source_artifacts": {
            "n5000_aggregate": str(n5000_path),
            "n5000_aggregate_sha256": _sha256(n5000_path),
            "n10000_aggregate": str(n10000_path),
            "n10000_aggregate_sha256": _sha256(n10000_path),
        },
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "scopes": scopes}, indent=2))


if __name__ == "__main__":
    main()
