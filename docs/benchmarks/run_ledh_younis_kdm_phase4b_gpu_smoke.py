#!/usr/bin/env python3
"""Run the bounded Phase 4B anchor/replay GPU/XLA implementation smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

# Import this dependency before TensorFlow-dependent route modules.  The
# numerical imports below are intentionally delayed until after device policy
# configuration in ``_run``.
from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "docs/plans/bayesfilter-ledh-younis-kdm-phase4b-resampling-reference-plan-2026-09-08.md"
)
SOURCES = (
    ROOT / "docs/benchmarks/run_ledh_younis_kdm_phase4b_gpu_smoke.py",
    ROOT / "bayesfilter/highdim/ledh_canonical_score_tf.py",
    ROOT / "bayesfilter/highdim/ledh_canonical_reset_score_tf.py",
    ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
    ROOT / "bayesfilter/highdim/ledh_younis_kdm_tf.py",
    ROOT / "bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py",
    PLAN,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _model(dtype: tf.dtypes.DType) -> Any:
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
            tf.eye(2, dtype=points.dtype), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=tf.constant([[0.35, 0.02], [0.02, 0.30]], dtype),
        observation_covariance=tf.constant([[0.55, 0.01], [0.01, 0.50]], dtype),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.zeros(
            [tf.shape(points)[0], 2, 2], points.dtype
        ),
    )


def _fixture(dtype: tf.dtypes.DType) -> dict[str, tf.Tensor]:
    particle_count, dimension, horizon = 8, 2, 2
    initial_states = tf.random.stateless_normal(
        [particle_count, dimension], seed=[20260908, 11], dtype=dtype
    ) * tf.cast(0.7, dtype)
    index = tf.cast(tf.range(particle_count), dtype)
    diagonal = tf.stack(
        [
            tf.cast(0.45, dtype) + tf.cast(0.02, dtype) * index,
            tf.cast(0.60, dtype) + tf.cast(0.015, dtype) * index,
        ],
        axis=1,
    )
    initial_covariances = (
        tf.linalg.diag(diagonal)
        + tf.constant([[0.0, 0.01], [0.01, 0.0]], dtype)[tf.newaxis, :, :]
    )
    noises = tf.random.stateless_normal(
        [horizon, particle_count, dimension],
        seed=[20260908, 13],
        dtype=dtype,
    )
    observations = tf.random.stateless_normal(
        [horizon, dimension], seed=[20260908, 17], dtype=dtype
    ) * tf.cast(0.8, dtype)
    common_bandwidth = tf.constant([[0.09, 0.012], [0.012, 0.07]], dtype)
    bandwidths = tf.broadcast_to(common_bandwidth, [horizon, dimension, dimension])
    stratified_uniforms = tf.random.stateless_uniform(
        [horizon, particle_count],
        seed=[20260908, 19],
        minval=tf.cast(0.05, dtype),
        maxval=tf.cast(0.95, dtype),
        dtype=dtype,
    )
    kdm_noises = tf.random.stateless_normal(
        [horizon, particle_count, dimension],
        seed=[20260908, 23],
        dtype=dtype,
    ) * tf.cast(0.6, dtype)
    return {
        "theta": tf.constant([0.42], dtype),
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": observations,
        "bandwidths": bandwidths,
        "d_bandwidths": tf.zeros_like(bandwidths),
        "stratified_uniforms": stratified_uniforms,
        "kdm_noises": kdm_noises,
    }


def _options(dtype: tf.dtypes.DType) -> dict[str, Any]:
    base = tf.concat([tf.eye(2, dtype=dtype), -tf.eye(2, dtype=dtype)], axis=0)
    return {
        "flow_substeps": 4,
        "reset_policy": "contract_e",
        "reset_design": tf.tile(base, [2, 1]),
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }


def _float(value: tf.Tensor) -> float:
    return float(tf.convert_to_tensor(value).numpy())


def _maximum_absolute(value: tf.Tensor) -> float:
    return _float(tf.reduce_max(tf.abs(value)))


def _run(args: argparse.Namespace) -> dict[str, Any]:
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    # Several route modules create TensorFlow tensors at import time.  Keep
    # those imports after physical-device memory growth has been configured.
    from bayesfilter.highdim.ledh_younis_kdm_resampling_tf import (
        BANDWIDTH_POLICY,
        DERIVATIVE_SEMANTICS,
        INCOMING_WEIGHT_POLICY,
        NORMALIZATION_TOLERANCE_POLICY,
        RESAMPLING_OPERATION_SEMANTICS,
        RESAMPLING_ROUTE_CLASSIFICATION,
        RESAMPLING_ROUTE_ID,
        SCORE_OUTPUT_SEMANTICS,
        SOURCE_SCOPE,
        make_resampling_anchor_kernel,
        make_resampling_replay_kernel,
    )
    from bayesfilter.highdim.ledh_younis_kdm_tf import RESKDM_IWSG_FINITE_TARGET

    tf.config.experimental.enable_tensor_float_32_execution(args.tf32)
    dtype = tf.float64 if args.dtype == "float64" else tf.float32
    fixture = _fixture(dtype)
    options = _options(dtype)
    model = _model(dtype)
    common = {
        "theta_dimension": 1,
        "particle_count": 8,
        "state_dimension": 2,
        "observation_dimension": 2,
        "horizon": 2,
        "canonical_options": options,
        "dtype": dtype,
        "covariance_mark_policy": args.covariance_mark_policy,
        "jit_compile": args.jit_compile,
    }
    anchor_kernel = make_resampling_anchor_kernel(model, **common)
    replay_kernel = make_resampling_replay_kernel(model, **common)
    shared = (
        fixture["theta"],
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        fixture["bandwidths"],
        fixture["d_bandwidths"],
    )

    start = time.perf_counter()
    anchor = anchor_kernel(
        *shared, fixture["stratified_uniforms"], fixture["kdm_noises"]
    )
    _ = anchor["value"].numpy()
    anchor_first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    replay = replay_kernel(
        *shared,
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    _ = replay["value"].numpy()
    replay_first_seconds = time.perf_counter() - start

    start = time.perf_counter()
    replay_warm = replay_kernel(
        *shared,
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    _ = replay_warm["value"].numpy()
    replay_warm_seconds = time.perf_counter() - start

    epsilon = 1.0e-5 if dtype == tf.float64 else 2.0e-3
    plus_shared = (fixture["theta"] + tf.cast([epsilon], dtype), *shared[1:])
    minus_shared = (fixture["theta"] - tf.cast([epsilon], dtype), *shared[1:])
    plus = replay_kernel(
        *plus_shared,
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    minus = replay_kernel(
        *minus_shared,
        anchor["fixed_samples"],
        anchor["fixed_proposal_log_densities"],
        anchor["component_indices"],
    )
    finite_difference = (plus["value"] - minus["value"]) / tf.cast(2.0 * epsilon, dtype)
    score_error = tf.abs(replay["score"][0] - finite_difference)
    score_scale = tf.maximum(tf.abs(finite_difference), tf.ones([], dtype))
    score_relative_error = score_error / score_scale
    tolerance = 2.0e-5 if dtype == tf.float64 else 5.0e-3
    value_replay_error = tf.abs(anchor["value"] - replay["value"])
    score_replay_error = tf.reduce_max(tf.abs(anchor["score"] - replay["score"]))
    machine_epsilon = (
        2.220446049250313e-16 if dtype == tf.float64 else 1.1920928955078125e-7
    )
    value_replay_scale = tf.maximum(
        tf.reduce_max(tf.abs(tf.stack([anchor["value"], replay["value"]]))),
        tf.ones([], dtype),
    )
    score_replay_scale = tf.maximum(
        tf.reduce_max(tf.abs(tf.concat([anchor["score"], replay["score"]], axis=0))),
        tf.ones([], dtype),
    )
    value_replay_tolerance = (
        tf.cast(128.0 * machine_epsilon, dtype) * value_replay_scale
    )
    score_replay_tolerance = (
        tf.cast(128.0 * machine_epsilon, dtype) * score_replay_scale
    )
    sample_replay_error = tf.reduce_max(
        tf.abs(anchor["fixed_samples"] - replay["fixed_samples"])
    )
    incoming_weight_error = tf.reduce_max(
        tf.abs(replay["incoming_log_weights"][1] - replay["outgoing_log_weights"][0])
    )
    incoming_weight_tangent_error = tf.reduce_max(
        tf.abs(
            replay["d_incoming_log_weights"][1] - replay["d_outgoing_log_weights"][0]
        )
    )
    pairwise_mask_count = tf.reduce_sum(
        tf.cast(anchor["higher_moment_pairwise_target_mask"] > 0.0, tf.int32)
    )
    pairwise_pre_cap_rms = tf.reduce_max(
        anchor["higher_moment_maximum_pairwise_pre_cap_particle_rms"]
    )
    cap_diagnostics_finite = tf.reduce_all(
        tf.math.is_finite(
            tf.stack(
                [
                    tf.reduce_min(
                        anchor["higher_moment_minimum_pairwise_particle_cap_scale"]
                    ),
                    tf.reduce_max(
                        anchor["higher_moment_maximum_coordinatewise_pre_cap_absolute"]
                    ),
                    tf.reduce_max(
                        anchor["higher_moment_maximum_coordinatewise_post_cap_absolute"]
                    ),
                    tf.reduce_mean(
                        anchor["higher_moment_fraction_coordinatewise_cap_active"]
                    ),
                    tf.reduce_min(
                        anchor["higher_moment_minimum_coordinatewise_cap_derivative"]
                    ),
                ]
            )
        )
    )
    pass_status = bool(
        (
            anchor["valid"]
            & replay["valid"]
            & tf.reduce_all(anchor["higher_moment_valid"])
            & tf.reduce_all(anchor["higher_moment_pairwise_configured"])
            & (pairwise_mask_count == tf.constant(4, tf.int32))
            & (pairwise_pre_cap_rms > tf.zeros([], dtype))
            & cap_diagnostics_finite
            & (score_relative_error <= tf.cast(tolerance, dtype))
            & (value_replay_error <= value_replay_tolerance)
            & (score_replay_error <= score_replay_tolerance)
            & (sample_replay_error == tf.zeros([], dtype))
            & (incoming_weight_error == tf.zeros([], dtype))
            & (incoming_weight_tangent_error == tf.zeros([], dtype))
            & (replay["pair_count"] == tf.constant(128, tf.int32))
        ).numpy()
    )
    memory = tf.config.experimental.get_memory_info("GPU:0")
    logical_gpus = [device.name for device in tf.config.list_logical_devices("GPU")]
    return {
        "schema": "bayesfilter.ledh_younis_kdm_phase4b_gpu_smoke.v4",
        "status": "PASS" if pass_status else "FAIL",
        "scientific_status": "IMPLEMENTATION_SMOKE_ONLY_NO_SCORE_QUALITY_CLAIM",
        "target_label": RESKDM_IWSG_FINITE_TARGET,
        "route_id": RESAMPLING_ROUTE_ID,
        "route_classification": RESAMPLING_ROUTE_CLASSIFICATION,
        "operation_semantics": RESAMPLING_OPERATION_SEMANTICS,
        "covariance_mark_policy": args.covariance_mark_policy,
        "bandwidth_policy": BANDWIDTH_POLICY,
        "incoming_weight_policy": INCOMING_WEIGHT_POLICY,
        "normalization_tolerance_policy": NORMALIZATION_TOLERANCE_POLICY,
        "derivative_semantics": DERIVATIVE_SEMANTICS,
        "score_output_semantics": SCORE_OUTPUT_SEMANTICS,
        "source_scope": SOURCE_SCOPE,
        "scope": {"dimension": 2, "particle_count": 8, "horizon": 2},
        "settings": {
            "dtype": dtype.name,
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "jit_compile": bool(args.jit_compile),
            "flow_substeps": 4,
            "reset_sinkhorn_steps": 4,
            "reset_balance_steps": 2,
            "correction_steps": 1,
            "pairwise_steps": 1,
            "coordinate_cap": 0.95,
            "finite_difference_step": epsilon,
            "relative_tolerance": tolerance,
            "replay_roundoff_multiplier": 128.0,
            "machine_epsilon": machine_epsilon,
            "effective_normalization_tolerance": (
                anchor_kernel.effective_normalization_tolerance
            ),
        },
        "diagnostics": {
            "anchor_valid": bool(anchor["valid"].numpy()),
            "replay_valid": bool(replay["valid"].numpy()),
            "anchor_value": _float(anchor["value"]),
            "replay_value": _float(replay["value"]),
            "replay_score": _float(replay["score"][0]),
            "finite_difference_score": _float(finite_difference),
            "score_absolute_error": _float(score_error),
            "score_relative_error": _float(score_relative_error),
            "anchor_replay_value_error": _float(value_replay_error),
            "anchor_replay_score_error": _float(score_replay_error),
            "anchor_replay_value_tolerance": _float(value_replay_tolerance),
            "anchor_replay_score_tolerance": _float(score_replay_tolerance),
            "sample_replay_error": _float(sample_replay_error),
            "incoming_log_weight_recurrence_error": _float(incoming_weight_error),
            "incoming_log_weight_tangent_recurrence_error": _float(
                incoming_weight_tangent_error
            ),
            "maximum_nonzero_incoming_weight_tangent": _maximum_absolute(
                replay["d_incoming_log_weights"][1]
            ),
            "maximum_anchor_log_ratio_error": _float(
                anchor["maximum_anchor_log_ratio_error"]
            ),
            "maximum_responsibility_row_sum_error": _float(
                replay["maximum_responsibility_row_sum_error"]
            ),
            "maximum_weight_sum_error": _float(replay["maximum_weight_sum_error"]),
            "maximum_weight_tangent_sum_error": _float(
                replay["maximum_weight_tangent_sum_error"]
            ),
            "maximum_anchor_importance_weight_sum_error": _float(
                anchor["maximum_anchor_importance_weight_sum_error"]
            ),
            "maximum_nonzero_importance_weight_sum_tangent": _maximum_absolute(
                replay["d_importance_weight_sums"]
            ),
            "minimum_bandwidth_eigenvalue": _float(
                replay["minimum_bandwidth_eigenvalue"]
            ),
            "pair_count": int(replay["pair_count"].numpy()),
            "higher_moment_valid": bool(
                tf.reduce_all(anchor["higher_moment_valid"]).numpy()
            ),
            "pairwise_configured_every_step": bool(
                tf.reduce_all(anchor["higher_moment_pairwise_configured"]).numpy()
            ),
            "off_diagonal_pairwise_target_mask_count": int(pairwise_mask_count.numpy()),
            "maximum_pairwise_pre_cap_particle_rms": _float(pairwise_pre_cap_rms),
            "maximum_pairwise_post_cap_particle_rms": _float(
                tf.reduce_max(
                    anchor["higher_moment_maximum_pairwise_post_cap_particle_rms"]
                )
            ),
            "minimum_pairwise_particle_cap_scale": _float(
                tf.reduce_min(
                    anchor["higher_moment_minimum_pairwise_particle_cap_scale"]
                )
            ),
            "maximum_coordinatewise_pre_cap_absolute": _float(
                tf.reduce_max(
                    anchor["higher_moment_maximum_coordinatewise_pre_cap_absolute"]
                )
            ),
            "maximum_coordinatewise_post_cap_absolute": _float(
                tf.reduce_max(
                    anchor["higher_moment_maximum_coordinatewise_post_cap_absolute"]
                )
            ),
            "mean_coordinatewise_cap_active_fraction": _float(
                tf.reduce_mean(
                    anchor["higher_moment_fraction_coordinatewise_cap_active"]
                )
            ),
            "minimum_coordinatewise_cap_derivative": _float(
                tf.reduce_min(
                    anchor["higher_moment_minimum_coordinatewise_cap_derivative"]
                )
            ),
        },
        "timings_seconds": {
            "anchor_compile_plus_first": anchor_first_seconds,
            "replay_compile_plus_first": replay_first_seconds,
            "replay_warm": replay_warm_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
            ),
            "logical_gpus": logical_gpus,
            "gpu_memory_policy": memory_policy,
            "gpu_allocator_bytes": {
                "current": int(memory["current"]),
                "peak": int(memory["peak"]),
            },
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "manifest": {
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "plan": str(PLAN.relative_to(ROOT)),
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in SOURCES
            },
            "random_seeds": [
                [20260908, 11],
                [20260908, 13],
                [20260908, 17],
                [20260908, 19],
                [20260908, 23],
            ],
        },
        "nonclaims": [
            "This is not the derivative of the unchanged ATOM-FINITE scalar.",
            "This smoke does not measure model-score bias or variance.",
            "This smoke does not establish DSGE, HMC, canonical, or production readiness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), required=True)
    parser.add_argument("--tf32", type=_parse_bool, default=False)
    parser.add_argument("--jit-compile", type=_parse_bool, default=True)
    parser.add_argument(
        "--covariance-mark-policy",
        choices=(
            "responsibility_conditional_mean_no_scatter_v1",
            "fixed_selected_component_mark_v1",
        ),
        default="responsibility_conditional_mean_no_scatter_v1",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    result = _run(args)
    result["manifest"]["wall_time_seconds"] = time.perf_counter() - started
    result["manifest"]["output"] = str((output_dir / "result.json").relative_to(ROOT))
    output_path = output_dir / "result.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "output": str(output_path)}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
