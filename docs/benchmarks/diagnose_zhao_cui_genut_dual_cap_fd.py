#!/usr/bin/env python3
"""Attribute cross-model dual-cap FD failures to baseline or selected arm."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
from docs.benchmarks import run_zhao_cui_genut_dual_cap_cross_model as campaign


STEPS = (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
SEED = campaign.CLAIM_SEEDS[0]


def _fd(target, evaluator, step: float):
    initial, process = base._noise(
        SEED, int(target["observations"].shape[0]), target["state_dim"]
    )
    theta = target["theta"]
    origin = evaluator(
        theta, target["observations"], initial, process, target["design"]
    )
    rows = []
    for index in range(target["parameter_dim"]):
        direction = tf.one_hot(index, target["parameter_dim"], dtype=tf.float32)
        plus = evaluator(
            theta + step * direction,
            target["observations"],
            initial,
            process,
            target["design"],
        )[0]
        minus = evaluator(
            theta - step * direction,
            target["observations"],
            initial,
            process,
            target["design"],
        )[0]
        finite_difference = (plus - minus) / (2.0 * step)
        manual = origin[1][index]
        absolute = tf.abs(finite_difference - manual)
        normalized = absolute / tf.maximum(tf.abs(manual), 1.0)
        rows.append(
            {
                "parameter": index,
                "manual_score": float(manual.numpy()),
                "finite_difference": float(finite_difference.numpy()),
                "absolute_residual": float(absolute.numpy()),
                "normalized_residual": float(normalized.numpy()),
            }
        )
    return rows


def run(source: Path, output: Path):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    prior = json.loads(source.read_text(encoding="utf-8"))
    targets = base._build_targets()
    models = []
    for prior_model in prior["models"]:
        if prior_model["row_id"] not in {"lgssm_T50", "predator_prey_T20"}:
            continue
        target = targets[prior_model["row_id"]]
        arms = []
        for arm_id, controls in (
            ("baseline", prior_model["baseline"]["controls"]),
            ("selected", prior_model["selected"]["controls"]),
        ):
            evaluator = campaign._make_evaluator(target, controls)
            step_rows = []
            for step in STEPS:
                rows = _fd(target, evaluator, step)
                step_rows.append(
                    {
                        "step": step,
                        "rows": rows,
                        "maximum_absolute_residual": max(
                            row["absolute_residual"] for row in rows
                        ),
                        "maximum_normalized_residual": max(
                            row["normalized_residual"] for row in rows
                        ),
                    }
                )
            arms.append({"arm_id": arm_id, "controls": controls, "steps": step_rows})
        models.append({"row_id": prior_model["row_id"], "arms": arms})
    payload = {
        "schema": "bayesfilter.zhao_cui_genut_dual_cap_fd_attribution.v1",
        "role": "explanatory_diagnostic_only_not_promotion_retest",
        "source_result": str(source.relative_to(ROOT)),
        "seed": SEED,
        "steps": STEPS,
        "models": models,
        "memory_policy": dict(memory_policy),
        "device": [device.name for device in tf.config.list_logical_devices("GPU")],
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": [
            "step ladder does not replace the predeclared h=1e-3 gate",
            "no score accuracy, posterior, HMC, default, or superiority claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.source.resolve(), args.output.resolve())
    print(json.dumps({"status": "complete", "output": str(args.output.resolve()), "wall_time_seconds": payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
