#!/usr/bin/env python3
"""Validate the source-order predator-prey SGQF physical value/score route."""

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
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory_policy = dict(configure_tensorflow_gpu_memory_growth(tf, require_gpu=True))
        tf.config.set_soft_device_placement(False)
        logical_gpus = tf.config.list_logical_devices("GPU")
        if not logical_gpus:
            raise RuntimeError("predator-prey GPU validation requires a logical GPU")
    else:
        logical_gpus = tf.config.list_logical_devices("GPU")
        if physical_gpus or logical_gpus:
            raise RuntimeError("CPU validation must hide all GPUs before TensorFlow import")

    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        make_predator_prey_source_sgqf_route,
    )
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import PP_TRUTH_PHYSICAL

    route = make_predator_prey_source_sgqf_route()
    point = PP_TRUTH_PHYSICAL[None, :]
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()

    @tf.function(
        input_signature=[tf.TensorSpec([1, 6], tf.float64)], jit_compile=True
    )
    def compiled(parameters):
        if args.mode == "gpu":
            with tf.device("/GPU:0"):
                return route.physical_value_score_status(parameters)
        with tf.device("/CPU:0"):
            return route.physical_value_score_status(parameters)

    value, score, status = compiled(point)
    value_device = str(value.device)
    score_device = str(score.device)
    hard_vetoes = []
    expected_device = "GPU" if args.mode == "gpu" else "CPU"
    if expected_device not in value_device.upper() or expected_device not in score_device.upper():
        hard_vetoes.append("result_device_placement_failed")
    if int(status["status_code"][0].numpy()) != 0:
        hard_vetoes.append("covariance_or_finite_status_failed")
    if int(status["transition_count"][0].numpy()) != 20:
        hard_vetoes.append("transition_count_failed")

    parity = None
    if args.mode == "gpu":
        if args.cpu_reference is None:
            raise ValueError("GPU mode requires --cpu-reference")
        reference = json.loads(args.cpu_reference.read_text(encoding="utf-8"))
        if reference.get("route_identity") != route.route_identity:
            raise ValueError("CPU reference route identity mismatch")
        cpu_result = reference["result"]
        if "CPU" not in str(cpu_result["value_device"]).upper():
            raise ValueError("CPU reference is not CPU-placed")
        value_difference = abs(float(value[0].numpy()) - float(cpu_result["value"][0]))
        score_difference = max(
            abs(float(actual) - float(expected))
            for actual, expected in zip(score[0].numpy(), cpu_result["score"][0])
        )
        value_tolerance = 1e-9 * max(1.0, abs(float(cpu_result["value"][0])))
        score_tolerance = 1e-8 * max(
            1.0, max(abs(float(item)) for item in cpu_result["score"][0])
        )
        if value_difference > value_tolerance:
            hard_vetoes.append("cpu_gpu_value_parity_failed")
        if score_difference > score_tolerance:
            hard_vetoes.append("cpu_gpu_score_parity_failed")
        parity = {
            "cpu_reference": _display(args.cpu_reference),
            "value_absolute_difference": value_difference,
            "value_tolerance": value_tolerance,
            "score_max_absolute_difference": score_difference,
            "score_tolerance": score_tolerance,
        }

    allocator = (
        {key: int(item) for key, item in tf.config.experimental.get_memory_info("GPU:0").items()}
        if args.mode == "gpu"
        else None
    )
    result_path = args.output_root / "result.json"
    payload: dict[str, object] = {
        "schema_version": "bayesfilter.predator_prey_source_sgqf_validation.v1",
        "status": "PASS" if not hard_vetoes else "BLOCKED",
        "mode": args.mode,
        "route_identity": route.route_identity,
        "route_manifest": dict(route.manifest),
        "result": {
            "value": _safe(value),
            "score": _safe(score),
            "status": _safe(status),
            "value_device": value_device,
            "score_device": score_device,
            "jit_compile": True,
        },
        "parity": parity,
        "hard_vetoes": hard_vetoes,
        "memory_policy": memory_policy,
        "gpu_allocator_bytes": allocator,
        "physical_gpus": [item.name for item in physical_gpus],
        "logical_gpus": [item.name for item in logical_gpus],
        "engineering_decision": (
            "full_T20_physical_value_score_XLA_pass"
            if not hard_vetoes
            else "engineering_gate_blocked"
        ),
        "numerical_decision": "finite_level2_candidate_manual_score_fd_qualified_on_cpu",
        "scientific_decision": "deterministic_approximation_viable_not_exact_or_superior",
        "nonclaims": [
            "no exact nonlinear likelihood claim",
            "no posterior or HMC readiness claim",
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
            "plan": "docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-master-program-2026-07-22.md",
            "result": _display(result_path),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if hard_vetoes:
        raise RuntimeError(f"predator-prey validation vetoes: {hard_vetoes}")
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
