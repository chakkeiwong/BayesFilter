#!/usr/bin/env python3
"""Validate the full source-row generalized-SV SGQF value/score route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    return value


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import tensorflow as tf
    import tensorflow_probability as tfp

    memory_policy = None
    physical_gpus = tf.config.list_physical_devices("GPU")
    if args.mode == "gpu":
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = dict(
            configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        )
        tf.config.set_soft_device_placement(False)
        logical_gpus = tf.config.list_logical_devices("GPU")
        if not logical_gpus:
            raise RuntimeError("generalized-SV GPU validation requires a logical GPU")
    else:
        logical_gpus = tf.config.list_logical_devices("GPU")
        if physical_gpus or logical_gpus:
            raise RuntimeError("CPU validation must hide all GPUs before TensorFlow import")

    from bayesfilter.highdim.generalized_sv_sgqf_tf import (
        generalized_sv_dense_value_reference_status,
        generalized_sv_sgqf_value_only_status,
        generalized_sv_sgqf_value_score_status,
        make_generalized_sv_sgqf_route,
    )

    route = make_generalized_sv_sgqf_route()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    @tf.function(
        input_signature=[
            tf.TensorSpec([3], tf.float64),
            tf.TensorSpec([1008, 1], tf.float64),
        ],
        jit_compile=True,
    )
    def compiled(theta, observations):
        device = "/GPU:0" if args.mode == "gpu" else "/CPU:0"
        with tf.device(device):
            value, score, status = generalized_sv_sgqf_value_score_status(
                theta, observations, sparse_level=3
            )
            value_only, value_status = generalized_sv_sgqf_value_only_status(
                theta, observations, sparse_level=3
            )
            return value, score, status, value_only, value_status

    value, score, status, value_only, value_status = compiled(
        route.theta, route.observations
    )
    value_float = float(value.numpy())
    score_values = [float(item) for item in score.numpy()]
    value_only_float = float(value_only.numpy())
    value_device = str(value.device)
    score_device = str(score.device)
    expected_device = "GPU" if args.mode == "gpu" else "CPU"
    hard_vetoes: list[str] = []
    if expected_device not in value_device.upper() or expected_device not in score_device.upper():
        hard_vetoes.append("result_device_placement_failed")
    if int(status["status_code"].numpy()) != 0:
        hard_vetoes.append("score_status_failed")
    if int(value_status["status_code"].numpy()) != 0:
        hard_vetoes.append("value_status_failed")
    if int(status["transition_count"].numpy()) != 1008:
        hard_vetoes.append("transition_count_failed")
    same_scalar_gap = abs(value_float - value_only_float)
    if same_scalar_gap > 1.0e-10:
        hard_vetoes.append("value_score_same_scalar_failed")

    cpu_diagnostics = None
    parity = None
    if args.mode == "cpu":
        dense, dense_status = generalized_sv_dense_value_reference_status(
            route.theta, route.observations, order=41
        )
        level5, level5_score, level5_status = generalized_sv_sgqf_value_score_status(
            route.theta, route.observations, sparse_level=5
        )
        fd = []
        step = tf.constant(1.0e-5, tf.float64)
        for index in range(3):
            direction = tf.one_hot(index, 3, dtype=tf.float64)
            plus, _ = generalized_sv_sgqf_value_only_status(
                route.theta + step * direction, route.observations, sparse_level=3
            )
            minus, _ = generalized_sv_sgqf_value_only_status(
                route.theta - step * direction, route.observations, sparse_level=3
            )
            fd.append(float(((plus - minus) / (2.0 * step)).numpy()))
        fd_max_gap = max(abs(actual - expected) for actual, expected in zip(score_values, fd))
        dense_gap = abs(value_float - float(dense.numpy()))
        level5_gap = abs(value_float - float(level5.numpy()))
        level5_score_gap = max(
            abs(actual - expected)
            for actual, expected in zip(score_values, level5_score.numpy())
        )
        if int(dense_status["status_code"].numpy()) != 0:
            hard_vetoes.append("dense_reference_status_failed")
        if int(level5_status["status_code"].numpy()) != 0:
            hard_vetoes.append("level5_status_failed")
        if dense_gap > 1.0e-4:
            hard_vetoes.append("level3_dense_reference_gap_failed")
        if level5_gap > 1.0e-4:
            hard_vetoes.append("level3_level5_value_gap_failed")
        if fd_max_gap > 5.0e-6:
            hard_vetoes.append("manual_score_full_horizon_fd_failed")
        cpu_diagnostics = {
            "dense_order": 41,
            "dense_value": float(dense.numpy()),
            "level3_dense_absolute_gap": dense_gap,
            "level5_value": float(level5.numpy()),
            "level5_score": _safe(level5_score),
            "level3_level5_value_absolute_gap": level5_gap,
            "level3_level5_score_max_absolute_gap": level5_score_gap,
            "central_fd_step": 1.0e-5,
            "central_fd_score": fd,
            "manual_fd_max_absolute_gap": fd_max_gap,
        }
    else:
        if args.cpu_reference is None:
            raise ValueError("GPU mode requires --cpu-reference")
        reference = json.loads(args.cpu_reference.read_text(encoding="utf-8"))
        if reference.get("route_identity") != route.route_identity:
            raise ValueError("CPU reference route identity mismatch")
        cpu_result = reference["result"]
        if "CPU" not in str(cpu_result["value_device"]).upper():
            raise ValueError("CPU reference is not CPU-placed")
        value_gap = abs(value_float - float(cpu_result["value"]))
        score_gap = max(
            abs(actual - expected)
            for actual, expected in zip(score_values, cpu_result["score"])
        )
        value_tolerance = 1.0e-9 * max(1.0, abs(float(cpu_result["value"])))
        score_tolerance = 1.0e-8 * max(
            1.0, max(abs(float(item)) for item in cpu_result["score"])
        )
        if value_gap > value_tolerance:
            hard_vetoes.append("cpu_gpu_value_parity_failed")
        if score_gap > score_tolerance:
            hard_vetoes.append("cpu_gpu_score_parity_failed")
        parity = {
            "cpu_reference": _display(args.cpu_reference),
            "value_absolute_gap": value_gap,
            "value_tolerance": value_tolerance,
            "score_max_absolute_gap": score_gap,
            "score_tolerance": score_tolerance,
        }

    allocator = (
        {
            key: int(item)
            for key, item in tf.config.experimental.get_memory_info("GPU:0").items()
        }
        if args.mode == "gpu"
        else None
    )
    result_path = args.output_root / "result.json"
    payload: dict[str, object] = {
        "schema_version": "bayesfilter.generalized_sv_sgqf_validation.v1",
        "status": "PASS" if not hard_vetoes else "BLOCKED",
        "mode": args.mode,
        "row_id": route.manifest["row_id"],
        "route_identity": route.route_identity,
        "route_manifest": dict(route.manifest),
        "result": {
            "value": value_float,
            "score": score_values,
            "status": _safe(status),
            "value_only": value_only_float,
            "value_status": _safe(value_status),
            "same_scalar_absolute_gap": same_scalar_gap,
            "value_device": value_device,
            "score_device": score_device,
            "jit_compile": True,
        },
        "cpu_diagnostics": cpu_diagnostics,
        "parity": parity,
        "hard_vetoes": hard_vetoes,
        "memory_policy": memory_policy,
        "gpu_allocator_bytes": allocator,
        "physical_gpus": [item.name for item in physical_gpus],
        "logical_gpus": [item.name for item in logical_gpus],
        "engineering_decision": (
            "full_T1008_level3_value_score_XLA_pass"
            if not hard_vetoes
            else "engineering_gate_blocked"
        ),
        "numerical_decision": (
            "manual_score_and_level3_dense_refinement_gates_pass"
            if not hard_vetoes
            else "numerical_candidate_blocked"
        ),
        "scientific_decision": "raw_y_gaussian_projection_candidate_viable_not_exact_or_superior",
        "nonclaims": [
            "no exact nonlinear likelihood or posterior claim",
            "no NativeGeneralizedSVSSM or KSC target substitution",
            "no statistically supported ranking or SGQF superiority claim",
            "GPU execution is engineering evidence only",
        ],
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
            ).hexdigest(),
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "platform": platform.platform(),
            "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "unset"),
            "started_utc": started_utc,
            "wall_seconds": time.perf_counter() - started,
            "plan": "docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-active-campaign-note-2026-07-22.md",
            "result": _display(result_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if hard_vetoes:
        raise RuntimeError(f"generalized-SV validation vetoes: {hard_vetoes}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path, default=None)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.mode == "cpu" and args.cpu_reference is not None:
        parser.error("--cpu-reference applies only to GPU mode")
    _run(args)


if __name__ == "__main__":
    main()
