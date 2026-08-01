#!/usr/bin/env python3
"""Reproduce and localize STR-UKF GenUT non-finite increments seed by seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the serious runner first: it configures and verifies memory growth
# immediately after TensorFlow import, before target constants initialize a GPU.
from docs.benchmarks import run_genut_str_ukf_leaderboard as campaign

tf = campaign.tf

PLAN = Path(
    "docs/plans/bayesfilter-genut-str-ukf-nonfinite-root-cause-plan-2026-07-22.md"
)
SCHEMA = "bayesfilter.genut_str_ukf_nonfinite_increment_trace.v1"
SEEDS = tuple(range(2026072291, 2026072299))
CONTROLS = {
    "epsilon": 4.0,
    "sinkhorn_steps": 4,
    "balance_steps": 8,
    "ridge": 1.0e-6,
}


def _json_number(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return value


def _first_false(mask: tf.Tensor) -> int | None:
    values = [bool(item) for item in mask.numpy()]
    return next((index for index, finite in enumerate(values) if not finite), None)


def run(output_root: Path) -> dict[str, object]:
    campaign._require_serious_gpu_policy()  # noqa: SLF001
    output_root.mkdir(parents=True, exist_ok=False)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("non-finite trace requires a logical GPU")
    _states, observations64 = campaign.generate_frozen_structural_dataset_tf()
    observations = tf.cast(observations64, tf.float32)
    theta = tf.cast(campaign.structural_truth_source(), tf.float32)
    design = campaign._genut_design()  # noqa: SLF001
    evaluate = campaign._make_evaluator(CONTROLS, campaign.HORIZON)  # noqa: SLF001
    rows = []
    for seed in SEEDS:
        initial, process = campaign._particle_noise(seed)  # noqa: SLF001
        started = time.perf_counter()
        value, score, diagnostics = evaluate(
            theta, observations, initial, process, design
        )
        value_increments = diagnostics["value_increments"]
        score_increments = diagnostics["score_increments"]
        value_increment_finite = tf.math.is_finite(value_increments)
        score_increment_finite = tf.reduce_all(
            tf.math.is_finite(score_increments), axis=1
        )
        score_coordinate_finite = tf.math.is_finite(score_increments)
        first_score_time = _first_false(score_increment_finite)
        first_score_coordinates = (
            []
            if first_score_time is None
            else [
                index
                for index, finite in enumerate(
                    score_coordinate_finite[first_score_time].numpy()
                )
                if not bool(finite)
            ]
        )
        row = {
            "seed": seed,
            "wall_time_seconds_including_first_compile": time.perf_counter() - started,
            "device": str(value.device),
            "value": _json_number(float(value.numpy())),
            "score": [_json_number(float(item)) for item in score.numpy()],
            "value_finite": bool(tf.math.is_finite(value).numpy()),
            "score_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
            "first_nonfinite_value_increment_time": _first_false(
                value_increment_finite
            ),
            "first_nonfinite_score_increment_time": first_score_time,
            "first_nonfinite_score_coordinates": first_score_coordinates,
            "value_increments": [
                _json_number(float(item)) for item in value_increments.numpy()
            ],
            "score_increments": [
                [_json_number(float(item)) for item in values]
                for values in score_increments.numpy()
            ],
            "max_transition_residual": _json_number(
                float(diagnostics["max_transition_residual"].numpy())
            ),
            "max_mean_residual": _json_number(
                float(diagnostics["max_mean_residual"].numpy())
            ),
            "max_row_residual": _json_number(
                float(diagnostics["max_row_residual"].numpy())
            ),
            "max_col_residual": _json_number(
                float(diagnostics["max_col_residual"].numpy())
            ),
        }
        rows.append(row)
        (output_root / f"seed_{seed}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    failing = [
        row["seed"]
        for row in rows
        if not row["value_finite"] or not row["score_finite"]
    ]
    payload = {
        "schema_version": SCHEMA,
        "plan": PLAN.as_posix(),
        "target_scope": campaign.STRUCTURAL_UKF_SCOPE,
        "particle_count": campaign.N,
        "horizon": campaign.HORIZON,
        "controls": CONTROLS,
        "seeds": SEEDS,
        "failing_seeds": failing,
        "all_finite": not failing,
        "rows": rows,
        "device": {
            "logical_devices": [item.name for item in logical],
            "dtype": "float32",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": True,
        },
        "memory_policy": campaign.MEMORY_POLICY,
        "allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "classification": "diagnostic_reproduction_only_consumed_claim_seeds",
    }
    (output_root / "increment_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_root)
    print(
        json.dumps(
            {
                "failing_seeds": result["failing_seeds"],
                "output": str(args.output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
