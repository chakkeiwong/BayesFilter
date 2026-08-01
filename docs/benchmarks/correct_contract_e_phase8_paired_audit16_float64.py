#!/usr/bin/env python3
"""Recompute the preserved paired-audit intervals with a float64 Student law."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _interval(values: list[float], critical: float) -> dict[str, float]:
    n = len(values)
    mean = math.fsum(values) / n
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(n)
    half_width = critical * standard_error
    return {
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_value": critical,
        "half_width": half_width,
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def corrected_payload(source_path: Path) -> dict[str, Any]:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    model = source["interval_model"]
    member_alpha = float(model["member_two_sided_alpha"])
    degrees = int(model["degrees_of_freedom"])
    critical = float(
        tfp.distributions.StudentT(
            df=tf.constant(float(degrees), tf.float64),
            loc=tf.constant(0.0, tf.float64),
            scale=tf.constant(1.0, tf.float64),
        ).quantile(tf.constant(1.0 - member_alpha / 2.0, tf.float64)).numpy()
    )
    names = source["quantity_names"]
    paired = source["paired_absolute_loss_difference"]
    contract_rows = source["per_arm_signed_normalized_error"]["all_active_contract_e"]
    loss_intervals = []
    directions = []
    for index, name in enumerate(names):
        interval = _interval([row[index] for row in paired], critical)
        if interval["upper"] < 0.0:
            direction = "contract_e_lower_mean_absolute_error"
        elif interval["lower"] > 0.0:
            direction = "contract_e_higher_mean_absolute_error"
        else:
            direction = "inconclusive"
        directions.append(direction)
        loss_intervals.append({"quantity": name, "direction": direction, **interval})
    overall = (
        directions[0]
        if len(set(directions)) == 1 and directions[0] != "inconclusive"
        else "mixed_or_inconclusive"
    )
    contract_intervals = []
    for index, name in enumerate(names):
        interval = _interval([row[index] for row in contract_rows], critical)
        boundary = 0.001 if index == 0 else 0.05
        equivalent = interval["lower"] > -boundary and interval["upper"] < boundary
        contract_intervals.append(
            {"quantity": name, "boundary": boundary, "equivalent": equivalent, **interval}
        )
    original_directions = [item["direction"] for item in source["paired_loss_intervals"]]
    original_equivalence = [item["equivalent"] for item in source["contract_e_mean_error_intervals"]]
    classifications_unchanged = (
        original_directions == directions
        and original_equivalence == [item["equivalent"] for item in contract_intervals]
        and source["overall_paired_loss_classification"] == overall
    )
    return {
        "schema_version": "bayesfilter.contract_e_phase8.paired_audit16_float64_correction.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FLOAT64_REPORTING_CORRECTION_COMPLETE",
        "source_artifact": str(source_path),
        "source_sha256": _sha256(source_path),
        "source_critical_value": model["critical_value"],
        "corrected_critical_value": critical,
        "critical_value_difference": critical - float(model["critical_value"]),
        "paired_loss_intervals": loss_intervals,
        "contract_e_mean_error_intervals": contract_intervals,
        "overall_paired_loss_classification": overall,
        "classifications_unchanged": classifications_unchanged,
        "interval_model": {
            **model,
            "critical_value": critical,
            "parameter_dtype": "float64",
        },
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "expensive_arms_rerun": False,
        },
        "nonclaims": [
            "reporting-only correction from preserved arrays",
            "no power, distribution-free, primary-shape, admission, HMC, or leaderboard claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = corrected_payload(args.source.resolve())
    if not payload["classifications_unchanged"]:
        raise RuntimeError("float64 correction changed a classification")
    _write_exclusive(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "critical_value": payload["corrected_critical_value"],
                "classifications_unchanged": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
