#!/usr/bin/env python3
"""Target-aligned Kalman gross-error diagnostic for the GenUT LGSSM cell."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
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
CENTER = (1.17, 0.805, 0.48, -1.02, -0.824)
LOWER = (-0.95, -0.95, -0.95, 0.05, 0.05)
UPPER = (0.95, 0.95, 0.95, 2.0, 2.0)
MATRIX = (
    (1.0, 0.25, -0.15),
    (0.2, 1.1, 0.3),
    (-0.1, 0.35, 0.9),
)
VALUE_ABS_TOLERANCE = 10.0
SCORE_MAX_ABS_TOLERANCE = 10.0
SOURCE_ORACLE_PHYSICAL = (0.72, 0.55, 0.35, 0.35, 0.45)
SOURCE_ORACLE_LIKELIHOOD = -136.07597463460453
SOURCE_ORACLE_ABS_TOLERANCE = 1.0e-6


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _physical_and_log_chart(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    probability = 0.5 * (
        1.0 + tf.math.erf(theta / tf.sqrt(tf.constant(2.0, tf.float64)))
    )
    lower = tf.constant(LOWER, tf.float64)
    width = tf.constant(
        tuple(upper - lower for lower, upper in zip(LOWER, UPPER)), tf.float64
    )
    physical = lower + width * probability
    # Uniform physical prior and chart width cancel exactly.
    log_prior_and_jacobian = tf.reduce_sum(
        -0.5 * tf.square(theta) - 0.5 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
    )
    return physical, log_prior_and_jacobian


def _initial_observation_kalman_log_likelihood(
    physical: tf.Tensor, observations: tf.Tensor
) -> tf.Tensor:
    phi = physical[:3]
    q_scale = physical[3]
    r_scale = physical[4]
    transition = tf.linalg.diag(phi)
    transition_covariance = tf.square(q_scale) * tf.eye(3, dtype=tf.float64)
    observation_matrix = tf.constant(MATRIX, tf.float64)
    observation_covariance = tf.square(r_scale) * tf.eye(3, dtype=tf.float64)
    mean = tf.zeros([3], tf.float64)
    covariance = tf.linalg.diag(tf.square(q_scale) / (1.0 - tf.square(phi)))
    total = tf.constant(0.0, tf.float64)
    for index in tf.range(tf.shape(observations)[0]):
        if index > 0:
            mean = tf.linalg.matvec(transition, mean)
            covariance = (
                transition @ covariance @ tf.transpose(transition)
                + transition_covariance
            )
        innovation = observations[index] - tf.linalg.matvec(observation_matrix, mean)
        innovation_covariance = (
            observation_matrix @ covariance @ tf.transpose(observation_matrix)
            + observation_covariance
        )
        factor = tf.linalg.cholesky(innovation_covariance)
        solve = tf.linalg.cholesky_solve(factor, innovation[:, None])[:, 0]
        total -= 0.5 * (
            3.0 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
            + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
            + tf.tensordot(innovation, solve, axes=1)
        )
        gain = tf.linalg.cholesky_solve(
            factor, observation_matrix @ covariance
        )
        gain = tf.transpose(gain)
        mean += tf.linalg.matvec(gain, innovation)
        left = tf.eye(3, dtype=tf.float64) - gain @ observation_matrix
        covariance = (
            left @ covariance @ tf.transpose(left)
            + gain @ observation_covariance @ tf.transpose(gain)
        )
        covariance = 0.5 * (covariance + tf.transpose(covariance))
    return total


def _posterior_value_score(
    theta: tf.Tensor, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        physical, log_chart = _physical_and_log_chart(theta)
        likelihood = _initial_observation_kalman_log_likelihood(
            physical, observations
        )
        posterior = likelihood + log_chart
    score = tape.gradient(posterior, theta)
    if score is None:
        raise RuntimeError("Kalman posterior gradient is unavailable")
    return posterior, score, likelihood


def main() -> int:
    args = _args()
    if args.output_root.exists():
        raise RuntimeError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(not args.disable_tf32)
    tf.config.experimental.enable_op_determinism()
    claim = json.loads(args.claim_artifact.read_text(encoding="utf-8"))
    if claim.get("model") != "lgssm" or not claim.get(
        "passed_capacity_replay_endpoint_gate", False
    ):
        raise ValueError("claim artifact is not a passing LGSSM readiness cell")
    finite_difference = claim.get("finite_difference")
    if not isinstance(finite_difference, dict) or not finite_difference.get(
        "center_valid", False
    ):
        raise ValueError("claim artifact lacks a valid center value/score")

    from bayesfilter.highdim.cubature_genut_neutra_targets import (
        GenUTControls,
        make_genut_neutra_target,
    )

    controls = claim["controls"]
    target = make_genut_neutra_target(
        "lgssm",
        particle_count=1008,
        controls=GenUTControls(
            epsilon=float(controls["epsilon"]),
            sinkhorn_steps=int(controls["sinkhorn_steps"]),
            balance_steps=int(controls["balance_steps"]),
            ridge=float(controls["ridge"]),
            higher_moment_correction_steps=int(
                controls["higher_moment_correction_steps"]
            ),
            higher_moment_strength=float(controls["higher_moment_strength"]),
            higher_moment_floor=float(controls["higher_moment_floor"]),
            tuning_scope=str(controls["tuning_scope"]),
            tuning_artifact=str(controls["tuning_artifact"]),
        ),
    )
    if target.target_signature != claim["target_signature"]:
        raise ValueError("claim target signature does not match current LGSSM target")
    # Target inputs are reconstructed under their bound GPU deterministic
    # policy; the independent Kalman arithmetic itself is forced to CPU.
    with tf.device("/CPU:0"):
        theta = tf.constant(CENTER, tf.float64)
        observations = tf.cast(tf.identity(target.observations), tf.float64)
        exact_value, exact_score, exact_likelihood = _posterior_value_score(
            theta, observations
        )
        source_oracle_replay = _initial_observation_kalman_log_likelihood(
            tf.constant(SOURCE_ORACLE_PHYSICAL, tf.float64), observations
        )
    source_oracle_abs_error = tf.abs(
        source_oracle_replay - tf.constant(SOURCE_ORACLE_LIKELIHOOD, tf.float64)
    )
    genut_value = tf.constant(finite_difference["center_value"], tf.float64)
    genut_score = tf.constant(finite_difference["score"], tf.float64)
    value_error = genut_value - exact_value
    score_error = genut_score - exact_score
    score_max_abs_error = tf.reduce_max(tf.abs(score_error))
    direction_cosine = tf.tensordot(genut_score, exact_score, axes=1) / (
        tf.linalg.norm(genut_score) * tf.linalg.norm(exact_score)
    )
    passed = bool(
        tf.math.is_finite(exact_value).numpy()
        and tf.reduce_all(tf.math.is_finite(exact_score)).numpy()
        and float(source_oracle_abs_error.numpy()) <= SOURCE_ORACLE_ABS_TOLERANCE
        and abs(float(value_error.numpy())) <= VALUE_ABS_TOLERANCE
        and float(score_max_abs_error.numpy()) <= SCORE_MAX_ABS_TOLERANCE
        and float(direction_cosine.numpy()) > 0.0
    )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema": "bayesfilter.genut_lgssm_kalman_gross_error.v1",
        "passed": passed,
        "target_signature": target.target_signature,
        "event_order": "observe_stationary_initial_state_then_transition_before_t_greater_than_zero",
        "coordinate": "five_probit_box_posterior",
        "center": list(CENTER),
        "physical_center": _physical_and_log_chart(theta)[0].numpy().tolist(),
        "source_oracle_physical": list(SOURCE_ORACLE_PHYSICAL),
        "source_oracle_expected_likelihood": SOURCE_ORACLE_LIKELIHOOD,
        "source_oracle_replayed_likelihood": float(source_oracle_replay.numpy()),
        "source_oracle_absolute_error": float(source_oracle_abs_error.numpy()),
        "exact_kalman_likelihood": float(exact_likelihood.numpy()),
        "exact_kalman_posterior": float(exact_value.numpy()),
        "exact_kalman_posterior_score": exact_score.numpy().tolist(),
        "genut_posterior": float(genut_value.numpy()),
        "genut_posterior_score": genut_score.numpy().tolist(),
        "posterior_value_error": float(value_error.numpy()),
        "posterior_score_error": score_error.numpy().tolist(),
        "posterior_score_max_absolute_error": float(score_max_abs_error.numpy()),
        "posterior_score_direction_cosine": float(direction_cosine.numpy()),
        "thresholds": {
            "posterior_value_absolute_error_max": VALUE_ABS_TOLERANCE,
            "posterior_score_max_absolute_error_max": SCORE_MAX_ABS_TOLERANCE,
            "posterior_score_direction_cosine_min_exclusive": 0.0,
            "source_oracle_absolute_error_max": SOURCE_ORACLE_ABS_TOLERANCE,
        },
        "role": "gross_target_error_veto_not_accuracy_or_ranking",
        "kalman_arithmetic_device": str(exact_value.device),
        "target_input_reconstruction_device_policy": "trusted_gpu_deterministic",
        "tf32_enabled": not args.disable_tf32,
        "memory_policy": memory_policy,
        "gpu_allocator": {
            "current_bytes": int(allocator["current"]),
            "peak_bytes": int(allocator["peak"]),
        },
        "claim_artifact": str(args.claim_artifact),
        "wall_time_seconds": time.monotonic() - started,
        "plan": PLAN,
        "nonclaims": [
            "not Kalman equivalence",
            "not a GenUT accuracy or superiority claim",
            "not NeuTra training or HMC evidence",
        ],
    }
    _write(args.output_root / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    _write(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.genut_lgssm_kalman_gross_error_manifest.v1",
            "git_commit": commit,
            "command": list(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "target_input_device": "/GPU:0",
            "kalman_arithmetic_device": str(exact_value.device),
            "memory_policy": memory_policy,
            "gpu_allocator": {
                "current_bytes": int(allocator["current"]),
                "peak_bytes": int(allocator["peak"]),
            },
            "tf32_enabled": not args.disable_tf32,
            "deterministic_ops_enabled": True,
            "target_signature": target.target_signature,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(args.output_root),
            "plan": PLAN,
            "result": str(args.output_root / "result.json"),
        },
    )
    print(json.dumps({"passed": passed, "result": str(args.output_root / "result.json")}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
