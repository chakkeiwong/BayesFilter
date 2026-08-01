#!/usr/bin/env python3
"""Trusted GPU/XLA smoke for the candidate nonlinear core."""

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


def _inputs(cubature_design):
    # Build the shared inputs on CPU.  The reference graph must not inherit a
    # default GPU placement merely because a GPU is visible to this process.
    with tf.device("/CPU:0"):
        theta = tf.constant([0.25, -0.15], tf.float32)
        observations = tf.constant([[0.2], [-0.1]], tf.float32)
        initial = tf.random.stateless_normal([12, 1], seed=[71, 72], dtype=tf.float32)
        process = tf.random.stateless_normal([2, 12, 1], seed=[73, 74], dtype=tf.float32)
        design = cubature_design(dim=1, num_particles=12)
    return theta, observations, initial, process, design


def run(output_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("GPU/XLA smoke requires a visible GPU")
    # A parity reference must fail on an unsupported CPU op rather than being
    # silently relocated to the GPU by TensorFlow's default soft placement.
    tf.config.set_soft_device_placement(False)
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("no logical GPU after memory-growth configuration")
    from bayesfilter.highdim.cubature_genut_adapters import (
        exact_transformed_sv_candidate_adapter,
    )
    from bayesfilter.highdim.cubature_genut_candidate import cubature_design
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = exact_transformed_sv_candidate_adapter()
    theta, observations, initial, process, design = _inputs(cubature_design)

    @tf.function(jit_compile=False)
    def cpu_reference(theta_, observations_, initial_, process_, design_):
        with tf.device("/CPU:0"):
            return finite_value_score(
                adapter, theta_, observations_, initial_, process_, design_
            )

    @tf.function(jit_compile=True)
    def gpu_xla(theta_, observations_, initial_, process_, design_):
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter, theta_, observations_, initial_, process_, design_
            )

    with tf.device("/CPU:0"):
        reference = cpu_reference(theta, observations, initial, process, design)
    actual = gpu_xla(theta, observations, initial, process, design)
    value_diff = float(tf.abs(actual[0] - reference[0]).numpy())
    score_diff = float(tf.reduce_max(tf.abs(actual[1] - reference[1])).numpy())
    reference_device = str(reference[0].device)
    actual_device = str(actual[0].device)
    memory = tf.config.experimental.get_memory_info("GPU:0")
    payload = {
        "schema_version": "bayesfilter.cubature_genut_gpu_xla_smoke.v1",
        "campaign_id": "cubature-genut-gpu-xla-smoke-sv-t2-20260721",
        "host": platform.node(),
        "physical_devices": [item.name for item in physical],
        "logical_devices": [item.name for item in logical],
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
        "memory_growth": True,
        "memory_policy": dict(memory_policy),
        "value_absolute_difference": value_diff,
        "score_max_absolute_difference": score_diff,
        "reference_device": reference_device,
        "actual_device": actual_device,
        "reference_is_cpu": "CPU" in reference_device.upper(),
        "actual_is_gpu": "GPU" in actual_device.upper(),
        "gpu_allocator": {key: int(value) for key, value in memory.items()},
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": list(sys.argv),
            "python": sys.version,
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "started_utc": started_utc,
            "wall_seconds": time.perf_counter() - started,
        },
        "finite": bool(tf.reduce_all(tf.math.is_finite(actual[0])).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(actual[1])).numpy()),
        "hard_valid": (
            value_diff < 2.0e-3
            and score_diff < 2.0e-2
            and "CPU" in reference_device.upper()
            and "GPU" in actual_device.upper()
        ),
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "nonclaims": ["smoke only", "no high-dimensional or full-horizon readiness", "no default or leaderboard admission"],
    }
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_root.resolve())


if __name__ == "__main__":
    main()
