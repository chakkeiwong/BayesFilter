#!/usr/bin/env python3
"""Real-scope scalar parity diagnostic for one final GenUT NeuTra target."""

from __future__ import annotations

import argparse
import json
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
CENTERS = {
    "lgssm": (1.17, 0.805, 0.48, -1.02, -0.824),
    "ksc_sv": (0.31863936396437514, -0.31863936396437514),
    "predator_prey": (0.0, -0.8416212335729142, 0.0, -0.8416212335729142, 0.0, 0.0),
}
VALUE_RELATIVE_TOLERANCE = 2.0e-4
SCORE_RELATIVE_TOLERANCE = 2.0e-3


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(CENTERS), required=True)
    parser.add_argument("--claim-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--diagnostic-disable-higher-moments", action="store_true")
    parser.add_argument("--diagnostic-disable-tf32", action="store_true")
    return parser.parse_args()


def _write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _controls(payload):
    from bayesfilter.highdim.cubature_genut_neutra_targets import GenUTControls

    return GenUTControls(
        epsilon=float(payload["epsilon"]),
        sinkhorn_steps=int(payload["sinkhorn_steps"]),
        balance_steps=int(payload["balance_steps"]),
        ridge=float(payload["ridge"]),
        higher_moment_correction_steps=int(payload["higher_moment_correction_steps"]),
        higher_moment_strength=float(payload["higher_moment_strength"]),
        higher_moment_floor=float(payload["higher_moment_floor"]),
        tuning_scope=str(payload["tuning_scope"]),
        tuning_artifact=str(payload["tuning_artifact"]),
    )


def _scalar_adapter(model: str):
    from bayesfilter.highdim.cubature_genut_adapters import (
        diagonal_lgssm_candidate_adapter,
        ksc_mixture_sv_candidate_adapter,
        predator_prey_candidate_adapter,
    )

    if model == "lgssm":
        return diagonal_lgssm_candidate_adapter(
            observation_matrix=tf.constant(
                ((1.0, 0.25, -0.15), (0.2, 1.1, 0.3), (-0.1, 0.35, 0.9)),
                tf.float32,
            )
        )
    if model == "ksc_sv":
        return ksc_mixture_sv_candidate_adapter()
    return predator_prey_candidate_adapter()


def main() -> int:
    args = _args()
    if args.output_root.exists():
        raise RuntimeError(f"output root must be fresh: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(
        not args.diagnostic_disable_tf32
    )
    tf.config.experimental.enable_op_determinism()
    claim = json.loads(args.claim_artifact.read_text(encoding="utf-8"))
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if claim.get("model") != args.model or not claim.get(
        "passed_capacity_replay_endpoint_gate", False
    ):
        raise ValueError("claim artifact is not a passing cell for the requested model")
    if claim.get("control_status") != "repository_tuning_artifact_bound":
        raise ValueError("scalar parity requires a repository-bound tuning artifact")

    from bayesfilter.highdim.cubature_genut_filter import finite_value_score
    from bayesfilter.highdim.cubature_genut_neutra_targets import (
        _core_kwargs,
        _filter_theta_and_jacobian,
        make_genut_neutra_target,
    )

    controls = _controls(claim["controls"])
    target = make_genut_neutra_target(
        args.model,
        particle_count=1008,
        controls=(
            controls
            if not args.diagnostic_disable_higher_moments
            else type(controls)(
                epsilon=controls.epsilon,
                sinkhorn_steps=controls.sinkhorn_steps,
                balance_steps=controls.balance_steps,
                ridge=controls.ridge,
                higher_moment_correction_steps=0,
                higher_moment_strength=0.0,
                higher_moment_floor=controls.higher_moment_floor,
                tuning_scope="diagnostic_higher_moments_disabled",
                tuning_artifact="diagnostic_only_not_claim_bound",
            )
        ),
    )
    if (
        not args.diagnostic_disable_higher_moments
        and target.target_signature != claim["target_signature"]
    ):
        raise ValueError("claim target signature does not match reconstructed target")
    center = tf.constant(CENTERS[args.model], tf.float64)
    filter_theta, chain, prior_jacobian, prior_score = _filter_theta_and_jacobian(
        target, center[None, :]
    )
    scalar_adapter = _scalar_adapter(args.model)

    @tf.function(jit_compile=True)
    def scalar_program(theta):
        with tf.device("/GPU:0"):
            return finite_value_score(
                scalar_adapter,
                theta,
                target.observations,
                target.initial_noise,
                target.process_noise,
                target.design,
                **_core_kwargs(target),
            )

    @tf.function(jit_compile=True)
    def batch_program(theta):
        with tf.device("/GPU:0"):
            return target.neutra_batch_log_prob_and_grad_status(theta)

    scalar_likelihood, scalar_filter_score, scalar_status = scalar_program(
        tf.cast(filter_theta[0], tf.float32)
    )
    batch_value, batch_score, batch_status = batch_program(
        tf.repeat(center[None, :], repeats=args.batch_size, axis=0)
    )
    scalar_value = tf.cast(scalar_likelihood, tf.float64) + prior_jacobian[0]
    scalar_score = (
        tf.cast(scalar_filter_score, tf.float64) * chain[0] + prior_score[0]
    )
    value_relative_error = tf.abs(batch_value[0] - scalar_value) / tf.maximum(
        tf.maximum(tf.abs(batch_value[0]), tf.abs(scalar_value)), 1.0
    )
    score_relative_error = tf.abs(batch_score[0] - scalar_score) / tf.maximum(
        tf.maximum(tf.abs(batch_score[0]), tf.abs(scalar_score)),
        tf.ones_like(scalar_score),
    )
    scalar_valid = bool(scalar_status["program_valid"].numpy())
    batch_valid = bool(batch_status["valid_pre_regularized_score"][0].numpy())
    passed = bool(
        scalar_valid
        and batch_valid
        and float(value_relative_error.numpy()) <= VALUE_RELATIVE_TOLERANCE
        and float(tf.reduce_max(score_relative_error).numpy())
        <= SCORE_RELATIVE_TOLERANCE
    )
    allocator = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema": "bayesfilter.genut_neutra_real_scope_scalar_parity.v1",
        "model": args.model,
        "passed": passed,
        "target_signature": target.target_signature,
        "claim_target_signature": claim["target_signature"],
        "batch_size": args.batch_size,
        "diagnostic_higher_moments_disabled": (
            args.diagnostic_disable_higher_moments
        ),
        "diagnostic_tf32_disabled": args.diagnostic_disable_tf32,
        "control_status": target.control_status,
        "scalar_program_valid": scalar_valid,
        "batch_program_valid": batch_valid,
        "scalar_likelihood": float(scalar_likelihood.numpy()),
        "scalar_filter_score": scalar_filter_score.numpy().tolist(),
        "prior_and_chart_value": float(prior_jacobian[0].numpy()),
        "prior_and_chart_score": prior_score[0].numpy().tolist(),
        "scalar_posterior_value": float(scalar_value.numpy()),
        "scalar_posterior_score": scalar_score.numpy().tolist(),
        "batch_posterior_value": float(batch_value[0].numpy()),
        "batch_posterior_score": batch_score[0].numpy().tolist(),
        "posterior_value_relative_error": float(value_relative_error.numpy()),
        "posterior_score_relative_error": score_relative_error.numpy().tolist(),
        "posterior_score_max_relative_error": float(
            tf.reduce_max(score_relative_error).numpy()
        ),
        "thresholds": {
            "posterior_value_relative_error_max": VALUE_RELATIVE_TOLERANCE,
            "posterior_score_max_relative_error_max": SCORE_RELATIVE_TOLERANCE,
        },
        "scalar_route_role": "independent_diagnostic_only_never_training_fallback",
        "batch_training_fallback_used": False,
        "memory_policy": memory_policy,
        "gpu_allocator": {
            "current_bytes": int(allocator["current"]),
            "peak_bytes": int(allocator["peak"]),
        },
        "jit_compile": True,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "deterministic_ops_enabled": True,
        "claim_artifact": str(args.claim_artifact),
        "wall_time_seconds": time.monotonic() - started,
        "plan": PLAN,
        "nonclaims": [
            "no filter-accuracy, NeuTra-quality, HMC, or posterior claim",
            "scalar route is not eligible for NeuTra optimizer updates",
        ],
    }
    _write(args.output_root / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    _write(
        args.output_root / "run_manifest.json",
        {
            "schema": "bayesfilter.genut_neutra_real_scope_scalar_parity_manifest.v1",
            "git_commit": commit,
            "command": list(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "device": "/GPU:0",
            "memory_policy": memory_policy,
            "gpu_allocator": result["gpu_allocator"],
            "jit_compile": True,
            "tf32_enabled": result["tf32_enabled"],
            "deterministic_ops_enabled": True,
            "target_signature": target.target_signature,
            "particle_count": 1008,
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(args.output_root),
            "plan": PLAN,
            "result": str(args.output_root / "result.json"),
        },
    )
    print(json.dumps({"model": args.model, "passed": passed}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
