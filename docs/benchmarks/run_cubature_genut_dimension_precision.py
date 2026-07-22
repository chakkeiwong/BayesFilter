#!/usr/bin/env python3
"""Bounded d=2/d=4 candidate dimension and precision diagnostic.

The compiled candidate is TensorFlow-only. Host conversion occurs only after
the compiled call to serialize a diagnostic artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


VALUE_TOLERANCE = 5.0e-3
SCORE_TOLERANCE = 2.0e-2

def _adapter(dimension: int):
    from bayesfilter.highdim.cubature_genut_filter import CandidateModelAdapter
    def initial(theta, noise):
        return noise + theta[0]

    def initial_tangent(theta, noise):
        return tf.concat(
            [tf.ones_like(noise)[..., None], tf.zeros_like(noise)[..., None]], axis=-1
        )[..., :2]

    def transition(theta, particles, noise, _time):
        return tf.tanh(particles + 0.1 * noise + theta[1])

    def transition_tangent(theta, particles, noise, tangent, _time):
        output = tf.tanh(particles + 0.1 * noise + theta[1])
        explicit = tf.concat(
            [tf.zeros_like(particles)[..., None], tf.ones_like(particles)[..., None]],
            axis=-1,
        )
        return (1.0 - tf.square(output))[..., None] * (tangent + explicit)

    def observation(theta, particles, observation_value, _time):
        residual = observation_value - particles
        return -0.5 * tf.reduce_sum(tf.square(residual), axis=1)

    def observation_tangent(theta, particles, tangent, observation_value, _time):
        return tf.reduce_sum((observation_value - particles)[..., None] * tangent, axis=1)

    return CandidateModelAdapter(
        state_dimension=dimension,
        parameter_count=2,
        initial_value=initial,
        initial_tangent=initial_tangent,
        transition_value=transition,
        transition_tangent=transition_tangent,
        observation_value=observation,
        observation_tangent=observation_tangent,
    )


def _base_inputs(dimension: int, particle_count: int, horizon: int, seed: int):
    with tf.device("/CPU:0"):
        return (
            tf.constant([0.2, -0.1], tf.float64),
            tf.random.stateless_normal(
                [horizon, dimension], [seed, seed + 1], dtype=tf.float64
            ),
            tf.random.stateless_normal(
                [particle_count, dimension], [seed + 2, seed + 3], dtype=tf.float64
            ),
            tf.random.stateless_normal(
                [horizon, particle_count, dimension],
                [seed + 4, seed + 5],
                dtype=tf.float64,
            ),
        )


def _run(
    dimension: int,
    particle_count: int,
    horizon: int,
    seed: int,
    *,
    gpu: bool,
    dtype,
    finite_value_score,
    cubature_design,
    base_inputs,
):
    adapter = _adapter(dimension)
    del seed
    with tf.device("/CPU:0"):
        theta, observations, initial, process = (
            tf.cast(item, dtype) for item in base_inputs
        )
        design = cubature_design(
            dim=dimension, num_particles=particle_count, dtype=dtype
        )

    @tf.function(jit_compile=gpu, reduce_retracing=True)
    def evaluate(theta_, observations_, initial_, process_, design_):
        if gpu:
            with tf.device("/GPU:0"):
                return finite_value_score(adapter, theta_, observations_, initial_, process_, design_)
        with tf.device("/CPU:0"):
            return finite_value_score(adapter, theta_, observations_, initial_, process_, design_)

    start = time.perf_counter()
    device = "/GPU:0" if gpu else "/CPU:0"
    with tf.device(device):
        value, score, diagnostics = evaluate(
            theta, observations, initial, process, design
        )
    elapsed = time.perf_counter() - start
    return {
        "dimension": dimension,
        "N": particle_count,
        "T": horizon,
        "dtype": dtype.name,
        "gpu": gpu,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy().tolist()],
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "device": str(value.device),
        "finite": bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "placement_valid": ("GPU" in str(value.device).upper()) if gpu else ("CPU" in str(value.device).upper()),
        "elapsed_seconds": elapsed,
    }


def run(output_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    physical = tf.config.list_physical_devices("GPU")
    tf.config.set_soft_device_placement(False)
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.highdim.cubature_genut_candidate import cubature_design
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    rows = []
    for dimension, particle_count in ((2, 12), (4, 8)):
        base_inputs = _base_inputs(dimension, particle_count, 2, 1200 + dimension)
        rows.append(
            _run(
                dimension,
                particle_count,
                2,
                1200 + dimension,
                gpu=True,
                dtype=tf.float32,
                finite_value_score=finite_value_score,
                cubature_design=cubature_design,
                base_inputs=base_inputs,
            )
        )
        rows.append(
            _run(
                dimension,
                particle_count,
                2,
                1200 + dimension,
                gpu=False,
                dtype=tf.float64,
                finite_value_score=finite_value_score,
                cubature_design=cubature_design,
                base_inputs=base_inputs,
            )
        )
    comparisons = []
    for index in range(0, len(rows), 2):
        gpu_row = rows[index]
        cpu_row = rows[index + 1]
        comparisons.append(
            {
                "dimension": gpu_row["dimension"],
                "N": gpu_row["N"],
                "T": gpu_row["T"],
                "value_absolute_difference": abs(gpu_row["value"] - cpu_row["value"]),
                "score_max_absolute_difference": max(
                    abs(left - right)
                    for left, right in zip(gpu_row["score"], cpu_row["score"])
                ),
            }
        )
    payload = {
        "schema_version": "bayesfilter.cubature_genut_dimension_precision.v1",
        "campaign_id": "cubature-genut-dimension-precision-20260721",
        "host": platform.node(),
        "physical_devices": [item.name for item in physical],
        "memory_policy": dict(memory_policy),
        "tf32_enabled": True,
        "rows": rows,
        "comparisons": comparisons,
        "hard_valid": all(
            row["finite"] and row["placement_valid"] for row in rows
        )
        and all(
            row["value_absolute_difference"] <= VALUE_TOLERANCE
            and row["score_max_absolute_difference"] <= SCORE_TOLERANCE
            for row in comparisons
        ),
        "precision_budget": {
            "value_absolute_difference": VALUE_TOLERANCE,
            "score_max_absolute_difference": SCORE_TOLERANCE,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": list(sys.argv),
            "python": sys.version,
            "tensorflow": tf.__version__,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
        },
        "nonclaims": ["diagnostic only", "no default or leaderboard admission"],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"]}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    run(parser.parse_args().output_root.resolve())
