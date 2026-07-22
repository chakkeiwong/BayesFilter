#!/usr/bin/env python3
"""Bounded GPU/XLA scaling ladder for the candidate Cubature/GenUT core.

This is a feasibility diagnostic for the exact transformed-SV scalar adapter.
It does not provide a high-dimensional or accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf


N_VALUES = (12, 24, 48, 96)
T_VALUES = (2, 10, 50)


def _cell_inputs(*, n: int, horizon: int, seed: int, cubature_design):
    with tf.device("/CPU:0"):
        theta = tf.constant([0.25, -0.15], tf.float32)
        observations = tf.random.stateless_normal(
            [horizon, 1], seed=[seed, seed + 1], dtype=tf.float32
        )
        initial = tf.random.stateless_normal(
            [n, 1], seed=[seed + 2, seed + 3], dtype=tf.float32
        )
        process = tf.random.stateless_normal(
            [horizon, n, 1], seed=[seed + 4, seed + 5], dtype=tf.float32
        )
        design = cubature_design(dim=1, num_particles=n)
    return theta, observations, initial, process, design


def _run_cell(*, n: int, horizon: int, seed: int, adapter, finite_value_score, cubature_design) -> dict[str, object]:
    theta, observations, initial, process, design = _cell_inputs(
        n=n, horizon=horizon, seed=seed, cubature_design=cubature_design
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def gpu_xla(theta_, observations_, initial_, process_, design_):
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter, theta_, observations_, initial_, process_, design_
            )

    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
    except (AttributeError, RuntimeError, tf.errors.InvalidArgumentError):
        pass
    start = time.perf_counter()
    first = gpu_xla(theta, observations, initial, process, design)
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    second = gpu_xla(theta, observations, initial, process, design)
    warm_seconds = time.perf_counter() - start
    value, score, diagnostics = second
    memory = tf.config.experimental.get_memory_info("GPU:0")
    value_finite = bool(tf.math.is_finite(value).numpy())
    score_finite = bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    diagnostics_finite = bool(
        tf.reduce_all(
            tf.stack(
                [
                    tf.math.is_finite(diagnostics["max_mean_residual"]),
                    tf.math.is_finite(diagnostics["max_row_residual"]),
                    tf.math.is_finite(diagnostics["max_col_residual"]),
                ]
            )
        ).numpy()
    )
    output_device = str(value.device)
    # Force the first result to remain observable so compilation cannot be
    # mistaken for successful numerical execution.
    first_finite = bool(tf.math.is_finite(first[0]).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(first[1])).numpy()
    )
    return {
        "N": n,
        "T": horizon,
        "seed": seed,
        "value": float(value.numpy()),
        "score": [float(item) for item in score.numpy().tolist()],
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "value_finite": value_finite,
        "score_finite": score_finite,
        "diagnostics_finite": diagnostics_finite,
        "first_result_finite": first_finite,
        "output_device": output_device,
        "output_is_gpu": "GPU" in output_device.upper(),
        "compile_seconds": compile_seconds,
        "warmed_seconds": warm_seconds,
        "gpu_allocator": {key: int(item) for key, item in memory.items()},
        "hard_valid": (
            value_finite
            and score_finite
            and diagnostics_finite
            and first_finite
            and "GPU" in output_device.upper()
        ),
    }


def run(
    output_root: Path,
    *,
    n_values: tuple[int, ...] = N_VALUES,
    t_values: tuple[int, ...] = T_VALUES,
) -> dict[str, object]:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("GPU/XLA scaling requires a visible GPU")
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
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    partial_path = output_root / "partial.json"
    partial = {
        "schema_version": "bayesfilter.cubature_genut_gpu_xla_scaling_partial.v1",
        "campaign_id": "cubature-genut-gpu-xla-scaling-sv-20260721",
        "n_values": list(n_values),
        "t_values": list(t_values),
        "rows": rows,
        "status": "RUNNING",
    }
    partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    for index, (horizon, n) in enumerate(
        (item for item in ((t, n) for t in t_values for n in n_values))
    ):
        try:
            row = _run_cell(
                n=n,
                horizon=horizon,
                seed=9100 + index * 17,
                adapter=adapter,
                finite_value_score=finite_value_score,
                cubature_design=cubature_design,
            )
        except Exception as error:  # preserve a bounded cell failure
            row = {
                "N": n,
                "T": horizon,
                "seed": 9100 + index * 17,
                "hard_valid": False,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "traceback": traceback.format_exc(),
            }
        rows.append(row)
        partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    payload = {
        "schema_version": "bayesfilter.cubature_genut_gpu_xla_scaling.v1",
        "campaign_id": "cubature-genut-gpu-xla-scaling-sv-20260721",
        "host": platform.node(),
        "physical_devices": [item.name for item in physical],
        "logical_devices": [item.name for item in logical],
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
        "memory_policy": dict(memory_policy),
        "adapter": "exact_transformed_sv_candidate_v1",
        "n_values": list(n_values),
        "t_values": list(t_values),
        "rows": rows,
        "hard_valid": all(bool(row["hard_valid"]) for row in rows),
        "nonclaims": [
            "scalar adapter feasibility diagnostic only",
            "no high-dimensional accuracy or full-horizon filtering claim",
            "no default or leaderboard admission",
        ],
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
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
    }
    partial["status"] = "COMPLETE"
    partial_path.write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n")
    (output_root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(output_root), "hard_valid": payload["hard_valid"]}, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--n-values", nargs="+", type=int, default=list(N_VALUES))
    parser.add_argument("--t-values", nargs="+", type=int, default=list(T_VALUES))
    args = parser.parse_args()
    run(
        args.output_root.resolve(),
        n_values=tuple(args.n_values),
        t_values=tuple(args.t_values),
    )


if __name__ == "__main__":
    main()
