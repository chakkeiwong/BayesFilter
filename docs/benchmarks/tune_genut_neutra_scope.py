#!/usr/bin/env python3
"""Bounded exact-scope control tuning for one campaign GenUT target."""

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
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PLAN = "docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md"
CENTERS = {
    "lgssm": (1.17, 0.805, 0.48, -1.02, -0.824),
    "ksc_sv": (0.31863936396437514, -0.31863936396437514),
    "predator_prey": (0.0, -0.8416212335729142, 0.0, -0.8416212335729142, 0.0, 0.0),
}
GRID = (
    (2.0, 4, 0.2),
    (4.0, 4, 0.2),
    (8.0, 4, 0.2),
    (2.0, 0, 0.0),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CENTERS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=1008)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def _ready(value):
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        materialized = value.numpy()
        return materialized.tolist() if materialized.shape else materialized.item()
    return value


def _write(path: Path, payload) -> None:
    path.write_text(
        json.dumps(_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _evaluate(model, controls, *, particle_count: int, noise_seed: int, offset: float):
    from bayesfilter.highdim.cubature_genut_neutra_targets import make_genut_neutra_target

    target = make_genut_neutra_target(
        model,
        particle_count=particle_count,
        noise_seed=noise_seed,
        controls=controls,
    )
    center = tf.constant(CENTERS[model], tf.float64)
    direction = tf.where(
        tf.equal(tf.math.floormod(tf.range(target.parameter_dim), 2), 0),
        tf.ones([target.parameter_dim], tf.float64),
        -tf.ones([target.parameter_dim], tf.float64),
    )
    theta = tf.stack([center + offset * direction, center - offset * direction])

    @tf.function(jit_compile=True)
    def value_score(values):
        values = tf.ensure_shape(values, [2, target.parameter_dim])
        with tf.device("/GPU:0"):
            return target.neutra_batch_log_prob_and_grad_status(values)

    @tf.function(jit_compile=True)
    def value_only(values):
        values = tf.ensure_shape(values, [2, target.parameter_dim])
        with tf.device("/GPU:0"):
            return target.batch_value_status(values)

    started = time.monotonic()
    value, score, status = value_score(theta)
    endpoint, endpoint_status = value_only(theta)
    elapsed = time.monotonic() - started
    endpoint_relative_error = tf.reduce_max(
        tf.abs(value - endpoint)
        / tf.maximum(tf.maximum(tf.abs(value), tf.abs(endpoint)), 1.0)
    )
    valid = tf.reduce_all(status["valid_pre_regularized_score"]) & tf.reduce_all(
        endpoint_status["valid_pre_regularized_score"]
    )
    residual_objective = tf.reduce_max(
        status["maximum_skew_residual"] + status["maximum_kurtosis_residual"]
    )
    return {
        "noise_seed": noise_seed,
        "offset": offset,
        "valid": valid,
        "value": value,
        "score": score,
        "endpoint": endpoint,
        "endpoint_max_relative_error": endpoint_relative_error,
        "maximum_shape_residual_sum": residual_objective,
        "status": status,
        "elapsed_seconds": elapsed,
        "target_signature": target.target_signature,
    }


def main() -> int:
    args = _args()
    if args.particle_count != 1008:
        raise ValueError("this tuner is bound to the campaign N=1008 scopes")
    if args.output_root.exists():
        raise RuntimeError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(not args.disable_tf32)
    tf.config.experimental.enable_op_determinism()
    from bayesfilter.highdim.cubature_genut_neutra_targets import GenUTControls

    arms = []
    for arm_index, (epsilon, shape_steps, shape_strength) in enumerate(GRID):
        controls = GenUTControls(
            epsilon=epsilon,
            sinkhorn_steps=8,
            balance_steps=8,
            ridge=1.0e-5,
            higher_moment_correction_steps=shape_steps,
            higher_moment_strength=shape_strength,
            higher_moment_floor=1.0e-5,
            tuning_scope=f"{args.model}_scope_tuning_candidate_{arm_index}",
            tuning_artifact="in_progress",
        )
        calibration = _evaluate(
            args.model,
            controls,
            particle_count=args.particle_count,
            noise_seed=141001,
            offset=0.06,
        )
        validation = _evaluate(
            args.model,
            controls,
            particle_count=args.particle_count,
            noise_seed=141002,
            offset=0.1,
        )
        eligible = bool(
            calibration["valid"].numpy()
            and validation["valid"].numpy()
            and float(calibration["endpoint_max_relative_error"].numpy()) <= 2.0e-4
            and float(validation["endpoint_max_relative_error"].numpy()) <= 2.0e-4
        )
        arms.append(
            {
                "arm_index": arm_index,
                "controls": controls.payload(),
                "calibration": calibration,
                "validation": validation,
                "eligible": eligible,
                "selection_objective_calibration_shape_residual": calibration[
                    "maximum_shape_residual_sum"
                ],
            }
        )
    eligible_arms = [arm for arm in arms if arm["eligible"]]
    if not eligible_arms:
        selected = None
        passed = False
    else:
        selected = min(
            eligible_arms,
            key=lambda arm: float(
                arm["selection_objective_calibration_shape_residual"].numpy()
            ),
        )
        passed = True
    selected_controls = None
    if selected is not None:
        selected_controls = {
            key: value
            for key, value in selected["controls"].items()
            if key
            in {
                "epsilon",
                "sinkhorn_steps",
                "balance_steps",
                "ridge",
                "higher_moment_correction_steps",
                "higher_moment_strength",
                "higher_moment_floor",
            }
        }
    scope_ids = {
        "lgssm": "lgssm_T50_N1008_initial_observation_first_fp32_tf32_deterministic_v2",
        "ksc_sv": "ksc_sv_T1000_N1008_initial_observation_first_fp32_tf32_deterministic_v2",
        "predator_prey": "predator_prey_T20_N1008_initial_observation_first_fp32_tf32_deterministic_v2",
    }
    scope_id = scope_ids[args.model]
    if args.disable_tf32:
        scope_id = scope_id.replace("fp32_tf32", "fp32_no_tf32")
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema": "bayesfilter.genut_neutra_scope_tuning.v1",
        "passed": passed,
        "model": args.model,
        "tuning_scope_id": scope_id,
        "particle_count": args.particle_count,
        "calibration_noise_seed": 141001,
        "validation_noise_seed": 141002,
        "claim_noise_seed_reserved_untouched": 140000,
        "grid": GRID,
        "arms": arms,
        "selected_arm_index": None if selected is None else selected["arm_index"],
        "selected_controls": selected_controls,
        "selection_rule": (
            "valid calibration+validation and endpoint relative error <=2e-4; "
            "then minimum calibration maximum raw skew+kurtosis residual sum"
        ),
        "validation_role": "veto_only_not_ranking",
        "memory_policy": memory_policy,
        "gpu_allocator": {
            "current_bytes": int(allocator["current"]),
            "peak_bytes": int(allocator["peak"]),
        },
        "jit_compile": True,
        "tf32_enabled": not args.disable_tf32,
        "deterministic_ops_enabled": True,
        "wall_time_seconds": time.monotonic() - started,
        "plan": PLAN,
        "nonclaims": (
            "control tuning only",
            "raw shape residual is a selection diagnostic, not posterior evidence",
            "no NeuTra or HMC claim",
        ),
    }
    _write(args.output_root / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.genut_neutra_scope_tuning_manifest.v1",
            "git_commit": commit,
            "command": tuple(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "device": "/GPU:0",
            "memory_policy": memory_policy,
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "particle_count": args.particle_count,
            "jit_compile": True,
            "tf32_enabled": not args.disable_tf32,
            "deterministic_ops_enabled": True,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(args.output_root),
            "plan": PLAN,
            "result": str(args.output_root / "result.json"),
        },
    )
    print(json.dumps({"model": args.model, "passed": passed, "selected": None if selected is None else selected["arm_index"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            destination = _args().output_root
            destination.mkdir(parents=True, exist_ok=True)
            _write(
                destination / "failure.json",
                {
                    "schema": "bayesfilter.genut_neutra_scope_tuning_failure.v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "plan": PLAN,
                },
            )
        finally:
            raise
