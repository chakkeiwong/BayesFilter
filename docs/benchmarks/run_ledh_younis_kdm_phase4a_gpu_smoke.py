"""Bounded GPU/XLA mechanics smoke for the complete Phase 4A KDM route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-ledh-younis-kdm-phase4-integrated-plan-2026-09-07.md"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _model(dtype: tf.dtypes.DType):
    from bayesfilter.highdim.ledh_canonical_score_tf import NonlinearScoreModel

    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return tf.sin(points) + d_points + theta[0] * tf.cos(points) * d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=dtype), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=0.4 * tf.eye(2, dtype=dtype),
        observation_covariance=0.6 * tf.eye(2, dtype=dtype),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.zeros(
            [tf.shape(points)[0], 2, 2], dtype=dtype
        ),
    )


def _float(value: tf.Tensor) -> float:
    return float(value.numpy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--tf32-mode", choices=("enabled", "disabled"), default="enabled"
    )
    arguments = parser.parse_args()
    if arguments.particle_count < 4 or arguments.particle_count % 4 != 0:
        raise ValueError("particle-count must be a positive multiple of four")
    if arguments.horizon < 2:
        raise ValueError("horizon must be at least two to exercise feedback")
    if arguments.repeats < 2:
        raise ValueError("repeats must be at least two")

    started = time.time()
    wall_start = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(
        arguments.tf32_mode == "enabled"
    )
    logical_devices = tf.config.list_logical_devices("GPU")
    if len(logical_devices) != 1:
        raise RuntimeError("smoke requires exactly one visible logical GPU")

    from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
        INTEGRATED_OBSERVATION_CLASSIFICATION,
        INTEGRATED_OBSERVATION_ROUTE_ID,
        make_integrated_linear_gaussian_kdm_kernel,
    )

    dtype = tf.float32 if arguments.dtype == "float32" else tf.float64
    particle_count = arguments.particle_count
    horizon = arguments.horizon
    with tf.device("/GPU:0"):
        initial_states = tf.random.stateless_normal(
            [particle_count, 2], seed=[2609, 7001], dtype=dtype
        )
        initial_covariances = tf.broadcast_to(
            tf.eye(2, dtype=dtype), [particle_count, 2, 2]
        )
        noises = tf.random.stateless_normal(
            [horizon, particle_count, 2], seed=[2609, 7002], dtype=dtype
        )
        observations = tf.random.stateless_normal(
            [horizon, 2], seed=[2609, 7003], dtype=dtype
        )
        observation_matrix = tf.eye(2, dtype=dtype)
        d_observation_matrix = tf.zeros([2, 2], dtype)
        bandwidths = tf.broadcast_to(
            0.08 * tf.eye(2, dtype=dtype),
            [horizon, particle_count, 2, 2],
        )
        d_bandwidths = tf.zeros_like(bandwidths)
        theta = tf.constant([0.6], dtype)
        base = tf.concat([tf.eye(2, dtype=dtype), -tf.eye(2, dtype=dtype)], axis=0)
        reset_design = tf.tile(base, [particle_count // 4, 1])

    options = {
        "flow_substeps": 6,
        "reset_policy": "contract_e",
        "reset_design": reset_design,
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }
    common = {
        "model": _model(dtype),
        "theta_dimension": 1,
        "particle_count": particle_count,
        "state_dimension": 2,
        "observation_dimension": 2,
        "horizon": horizon,
        "canonical_options": options,
        "bandwidth_is_zero": False,
        "dtype": dtype,
    }
    graph_kernel = make_integrated_linear_gaussian_kdm_kernel(
        **common, jit_compile=False
    )
    xla_kernel = make_integrated_linear_gaussian_kdm_kernel(
        **common, jit_compile=True
    )
    inputs = (
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        observation_matrix,
        d_observation_matrix,
        bandwidths,
        d_bandwidths,
    )

    graph_start = time.perf_counter()
    graph_result = graph_kernel(*inputs)
    graph_result["value"].numpy()
    graph_first_seconds = time.perf_counter() - graph_start

    xla_start = time.perf_counter()
    xla_result = xla_kernel(*inputs)
    xla_result["value"].numpy()
    xla_first_seconds = time.perf_counter() - xla_start

    repeat_seconds = []
    repeated_values = []
    for _ in range(arguments.repeats):
        repeat_start = time.perf_counter()
        repeated = xla_kernel(*inputs)
        repeated_values.append(_float(repeated["value"]))
        repeat_seconds.append(time.perf_counter() - repeat_start)

    value_error = tf.abs(xla_result["value"] - graph_result["value"])
    score_error = tf.reduce_max(
        tf.abs(xla_result["score"] - graph_result["score"])
    )
    state_error = tf.reduce_max(
        tf.abs(
            xla_result["states_after_reset"]
            - graph_result["states_after_reset"]
        )
    )
    weight_error = tf.reduce_max(
        tf.abs(
            xla_result["posterior_weights"]
            - graph_result["posterior_weights"]
        )
    )
    concrete = xla_kernel.get_concrete_function()
    must_compile = bool(concrete.function_def.attr["_XlaMustCompile"].b)
    memory_info = tf.config.experimental.get_memory_info("GPU:0")
    output_device = str(xla_result["value"].device)
    finite = bool(
        (
            tf.math.is_finite(xla_result["value"])
            & tf.reduce_all(tf.math.is_finite(xla_result["score"]))
            & xla_result["valid"]
        ).numpy()
    )
    thresholds = {
        "value_abs": 2.0e-4,
        "score_abs": 5.0e-4,
        "state_abs": 5.0e-4,
        "weight_abs": 2.0e-4,
    }
    errors = {
        "value_abs": _float(value_error),
        "score_abs": _float(score_error),
        "state_abs": _float(state_error),
        "weight_abs": _float(weight_error),
    }
    passed = (
        finite
        and must_compile
        and "GPU:0" in output_device
        and all(errors[name] <= limit for name, limit in thresholds.items())
        and max(repeated_values) == min(repeated_values)
    )

    root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path("bayesfilter/highdim/ledh_canonical_score_tf.py"),
        Path("bayesfilter/highdim/ledh_canonical_score_stages_tf.py"),
        Path("bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py"),
        Path(PLAN_PATH),
    )
    result = {
        "schema": "bayesfilter.ledh_younis_kdm_phase4a_gpu_smoke.v1",
        "status": "PASS" if passed else "FAIL",
        "role": "bounded_engineering_smoke_not_research_evidence",
        "question": "Does the complete Phase 4A shared-executor endpoint run reproducibly under GPU/XLA and agree with its non-XLA graph?",
        "promotion_criterion": "finite valid output, XLA must-compile flag, GPU output placement, deterministic replay, and graph/XLA errors within declared thresholds",
        "promotion_vetoes": "nonfinite, invalid factor/reset, CPU placement, no XLA compilation, nondeterminism, or parity threshold failure",
        "explanatory_only": ["compile time", "steady-state time", "allocator bytes"],
        "nonclaims": [
            "does not show lower model-score error",
            "does not tune bandwidth or LEDH controls",
            "does not establish DSGE, HMC, production, or default readiness",
        ],
        "route_id": INTEGRATED_OBSERVATION_ROUTE_ID,
        "route_classification": INTEGRATED_OBSERVATION_CLASSIFICATION,
        "target_label": xla_kernel.target_label,
        "complete_mixture_posterior": xla_kernel.complete_mixture_posterior,
        "model": "two_dimensional_nonlinear_mechanics_fixture",
        "particle_count": particle_count,
        "horizon": horizon,
        "bandwidth_diagonal": 0.08,
        "dtype": dtype.name,
        "tf32_mode": arguments.tf32_mode,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "jit_compile": True,
        "xla_must_compile_attribute": must_compile,
        "output_device": output_device,
        "graph_xla_errors": errors,
        "graph_xla_thresholds": thresholds,
        "valid": finite,
        "atom_identity_all_valid": bool(
            tf.reduce_all(xla_result["atom_identity_valid"]).numpy()
        ),
        "observation_map_all_valid": bool(
            tf.reduce_all(xla_result["observation_map_valid"]).numpy()
        ),
        "effective_model_tolerance": _float(
            xla_result["effective_model_tolerance"]
        ),
        "effective_covariance_tolerance": _float(
            xla_result["effective_covariance_tolerance"]
        ),
        "deterministic_repeated_values": repeated_values,
        "timing_seconds": {
            "graph_first": graph_first_seconds,
            "xla_compile_and_first": xla_first_seconds,
            "xla_repeats": repeat_seconds,
        },
        "gpu_allocator_bytes": {
            "current": int(memory_info["current"]),
            "peak": int(memory_info["peak"]),
        },
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "environment_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "gpu_memory_policy": memory_policy,
        "logical_devices": [device.name for device in logical_devices],
        "trust_basis": TRUST_BASIS,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
        "command": shlex.join((sys.executable, *sys.argv)),
        "started_unix": started,
        "wall_time_seconds": time.perf_counter() - wall_start,
        "plan_file": PLAN_PATH,
        "result_file": str(arguments.output),
        "source_sha256": {
            str(path): _sha256(root / path) for path in source_paths
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=False)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(arguments.output), "errors": errors, "timing_seconds": result["timing_seconds"]}, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
