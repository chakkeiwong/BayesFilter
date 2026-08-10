#!/usr/bin/env python3
"""Trusted GPU/XLA readiness cell for one frozen GenUT NeuTra target."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN = "docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md"
CENTERS = {
    "lgssm": (1.17, 0.805, 0.48, -1.02, -0.824),
    "ksc_sv": (0.31863936396437514, -0.31863936396437514),
    "austria_sir": (0.0, 0.0, 0.0),
    "predator_prey": (0.0, -0.8416212335729142, 0.0, -0.8416212335729142, 0.0, 0.0),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CENTERS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--particle-count", type=int, default=1008)
    parser.add_argument("--fd-step", type=float, default=0.0)
    parser.add_argument("--tuning-artifact", type=Path)
    parser.add_argument("--deterministic-ops", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        materialized = value.numpy()
        return materialized.tolist() if materialized.shape else materialized.item()
    return value


def _write(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _args()
    output = args.output_root
    if output.exists():
        raise RuntimeError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(not args.disable_tf32)
    if args.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    from bayesfilter.highdim.cubature_genut_neutra_targets import (
        make_genut_neutra_target,
    )
    from bayesfilter.inference.neutra_batching import bind_batch_native_neutra_target

    controls = None
    if args.tuning_artifact is not None:
        from bayesfilter.highdim.cubature_genut_neutra_targets import GenUTControls

        tuning_payload = json.loads(args.tuning_artifact.read_text(encoding="utf-8"))
        selected = tuning_payload["selected_controls"]
        controls = GenUTControls(
            epsilon=float(selected["epsilon"]),
            sinkhorn_steps=int(selected["sinkhorn_steps"]),
            balance_steps=int(selected["balance_steps"]),
            ridge=float(selected["ridge"]),
            higher_moment_correction_steps=int(
                selected["higher_moment_correction_steps"]
            ),
            higher_moment_strength=float(selected["higher_moment_strength"]),
            higher_moment_floor=float(selected["higher_moment_floor"]),
            tuning_scope=str(tuning_payload["tuning_scope_id"]),
            tuning_artifact=str(args.tuning_artifact),
        )
    target = make_genut_neutra_target(
        args.model, particle_count=args.particle_count, controls=controls
    )
    binding = bind_batch_native_neutra_target(
        target, target_signature=target.target_signature
    )
    center = tf.constant(CENTERS[args.model], tf.float64)
    offsets = tf.linspace(
        tf.constant(-0.002, tf.float64),
        tf.constant(0.002, tf.float64),
        args.batch_size,
    )
    theta = center[None, :] + offsets[:, None]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_score(values):
        values = tf.ensure_shape(
            values, [args.batch_size, target.parameter_dim]
        )
        with tf.device("/GPU:0"):
            return target.neutra_batch_log_prob_and_grad_status(values)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def value_only(values):
        values = tf.ensure_shape(
            values, [args.batch_size, target.parameter_dim]
        )
        with tf.device("/GPU:0"):
            return target.batch_value_status(values)

    compile_started = time.monotonic()
    value, score, status = value_score(theta)
    compile_run_seconds = time.monotonic() - compile_started
    endpoint_started = time.monotonic()
    endpoint, endpoint_status = value_only(theta)
    endpoint_compile_run_seconds = time.monotonic() - endpoint_started
    replay_started = time.monotonic()
    replay_value, replay_score, replay_status = value_score(theta)
    replay_endpoint, _ = value_only(theta)
    replay_seconds = time.monotonic() - replay_started
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    all_valid = bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
    endpoint_valid = bool(
        tf.reduce_all(endpoint_status["valid_pre_regularized_score"]).numpy()
    )
    value_endpoint_error = float(tf.reduce_max(tf.abs(value - endpoint)).numpy())
    value_endpoint_relative_error = float(
        tf.reduce_max(
            tf.abs(value - endpoint)
            / tf.maximum(tf.maximum(tf.abs(value), tf.abs(endpoint)), 1.0)
        ).numpy()
    )
    value_replay_error = float(tf.reduce_max(tf.abs(value - replay_value)).numpy())
    score_replay_error = float(tf.reduce_max(tf.abs(score - replay_score)).numpy())
    endpoint_replay_error = float(
        tf.reduce_max(tf.abs(endpoint - replay_endpoint)).numpy()
    )
    fd = None
    if args.fd_step > 0.0:
        parameter_count = target.parameter_dim
        step = tf.constant(args.fd_step, tf.float64)
        eye = tf.eye(parameter_count, dtype=tf.float64)
        if args.batch_size != 2:
            raise ValueError("finite-difference diagnostic requires batch size 2")
        fd_values = []
        fd_valid = []
        # Diagnostic-only host loop over coordinates. Each target call remains
        # a genuine B=2 batch; this path never updates NeuTra parameters.
        for index in range(parameter_count):
            endpoints = tf.stack(
                [center + step * eye[index], center - step * eye[index]]
            )
            endpoint_values, endpoint_status = value_only(endpoints)
            fd_values.append((endpoint_values[0] - endpoint_values[1]) / (2.0 * step))
            fd_valid.append(
                tf.reduce_all(endpoint_status["valid_pre_regularized_score"])
            )
        finite_difference = tf.stack(fd_values)
        center_value, center_score, center_status = value_score(tf.stack([center, center]))
        denominator = tf.maximum(
            tf.maximum(tf.abs(finite_difference), tf.abs(center_score[0])),
            tf.ones([parameter_count], tf.float64),
        )
        relative_error = tf.abs(finite_difference - center_score[0]) / denominator
        fd = {
            "step": args.fd_step,
            "stencil_valid": bool(tf.reduce_all(tf.stack(fd_valid)).numpy()),
            "center_valid": bool(
                tf.reduce_all(center_status["valid_pre_regularized_score"]).numpy()
            ),
            "center_value": center_value[0],
            "score": center_score[0],
            "finite_difference": finite_difference,
            "relative_error": relative_error,
            "maximum_relative_error": tf.reduce_max(relative_error),
            "diagnostic_coordinate_python_loop_used": True,
            "training_sample_loop_used": False,
        }
    fd_passed = bool(
        fd is None
        or (
            fd["stencil_valid"]
            and fd["center_valid"]
            and float(fd["maximum_relative_error"].numpy()) <= 0.05
        )
    )
    passed = bool(
        all_valid
        and endpoint_valid
        and value_endpoint_relative_error <= 2.0e-4
        and value_replay_error == 0.0
        and score_replay_error == 0.0
        and endpoint_replay_error == 0.0
        and fd_passed
        and "GPU" in str(value.device).upper()
        and "GPU" in str(endpoint.device).upper()
    )
    result = {
        "schema": "bayesfilter.genut_four_model_neutra_readiness_cell.v1",
        "model": args.model,
        "passed_capacity_replay_endpoint_gate": passed,
        "target_signature": target.target_signature,
        "adapter_signature": target.adapter_signature(),
        "target_scope": target.target_scope,
        "control_status": target.control_status,
        "controls": target.controls.payload(),
        "particle_count": args.particle_count,
        "batch_size": args.batch_size,
        "value": value,
        "score": score,
        "status": status,
        "endpoint": endpoint,
        "endpoint_status": endpoint_status,
        "value_endpoint_max_absolute_error": value_endpoint_error,
        "value_endpoint_max_relative_error": value_endpoint_relative_error,
        "same_process_value_replay_max_absolute_error": value_replay_error,
        "same_process_score_replay_max_absolute_error": score_replay_error,
        "same_process_endpoint_replay_max_absolute_error": endpoint_replay_error,
        "value_device": str(value.device),
        "score_device": str(score.device),
        "endpoint_device": str(endpoint.device),
        "compile_and_first_run_seconds": compile_run_seconds,
        "endpoint_compile_and_first_run_seconds": endpoint_compile_run_seconds,
        "warm_replay_seconds": replay_seconds,
        "gpu_allocator": {
            "current_bytes": int(allocator["current"]),
            "peak_bytes": int(allocator["peak"]),
        },
        "finite_difference": fd,
        "finite_difference_gate_passed": fd_passed,
        "batch_binding": binding.payload(),
        "memory_policy": memory_policy,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "deterministic_ops_enabled": args.deterministic_ops,
        "jit_compile": True,
        "dtype": "float32_filter_float64_posterior",
        "wall_time_seconds": time.monotonic() - started,
        "plan": PLAN,
        "nonclaims": (
            "capacity, replay, endpoint, and optional FD diagnostic only",
            "no NeuTra training, HMC, convergence, or posterior claim",
        ),
    }
    _write(output / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write(
        output / "run_manifest.json",
        {
            "schema": "bayesfilter.genut_four_model_neutra_readiness_manifest.v1",
            "git_commit": commit,
            "command": tuple(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu": tuple(str(item) for item in tf.config.list_physical_devices("GPU")),
            "memory_policy": memory_policy,
            "device": "/GPU:0",
            "jit_compile": True,
            "tf32_enabled": not args.disable_tf32,
            "deterministic_ops_enabled": args.deterministic_ops,
            "particle_count": args.particle_count,
            "batch_size": args.batch_size,
            "target_signature": target.target_signature,
            "data_id": target.data_id,
            "controls": target.controls.payload(),
            "noise_seed": 140000,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output),
            "plan": PLAN,
            "result": str(output / "result.json"),
        },
    )
    print(json.dumps({"model": args.model, "passed": passed, "result": str(output / "result.json")}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        destination = None
        try:
            destination = _args().output_root
            destination.mkdir(parents=True, exist_ok=True)
            _write(
                destination / "failure.json",
                {
                    "schema": "bayesfilter.genut_four_model_neutra_readiness_failure.v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "plan": PLAN,
                },
            )
        finally:
            raise
