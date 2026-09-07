"""Localize the Phase 4A graph/XLA discrepancy under TF32."""

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
SMOKE_PATH = "docs/benchmarks/run_ledh_younis_kdm_phase4a_gpu_smoke.py"
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


def _fixture(dtype: tf.dtypes.DType, particle_count: int, horizon: int):
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
    return (
        _model(dtype),
        options,
        (
            theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            observation_matrix,
            d_observation_matrix,
            bandwidths,
            d_bandwidths,
        ),
    )


def _kernel(dtype, model, options, particle_count, horizon, *, jit_compile):
    from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
        make_integrated_linear_gaussian_kdm_kernel,
    )

    return make_integrated_linear_gaussian_kdm_kernel(
        model=model,
        theta_dimension=1,
        particle_count=particle_count,
        state_dimension=2,
        observation_dimension=2,
        horizon=horizon,
        canonical_options=options,
        bandwidth_is_zero=False,
        dtype=dtype,
        jit_compile=jit_compile,
    )


def _run(kernel, inputs):
    result = kernel(*inputs)
    # Materialize all outputs before returning so compilation and execution
    # errors cannot be confused with a later comparison failure.
    for value in result.values():
        if isinstance(value, tf.Tensor):
            value.numpy()
    return result


def _finite_valid(result) -> bool:
    finite = tf.constant(True)
    for name in (
        "value",
        "score",
        "posterior_weights",
        "states_after_reset",
        "pre_flow",
        "children",
        "predicted_covariances",
        "post_covariances",
    ):
        finite &= tf.reduce_all(tf.math.is_finite(result[name]))
    return bool((finite & result["valid"]).numpy())


def _summary(result, *, must_compile: bool | None):
    return {
        "value": float(result["value"].numpy()),
        "score": [float(x) for x in result["score"].numpy().reshape(-1)],
        "valid": bool(result["valid"].numpy()),
        "finite_valid": _finite_valid(result),
        "factor_all_valid": bool(tf.reduce_all(result["factor_valid"]).numpy()),
        "atom_identity_all_valid": bool(
            tf.reduce_all(result["atom_identity_valid"]).numpy()
        ),
        "observation_map_all_valid": bool(
            tf.reduce_all(result["observation_map_valid"]).numpy()
        ),
        "atom_factor_value_error_max": float(
            result["atom_factor_value_error_max"].numpy()
        ),
        "atom_factor_tangent_error_max": float(
            result["atom_factor_tangent_error_max"].numpy()
        ),
        "observation_map_value_error_max": float(
            result["observation_map_value_error_max"].numpy()
        ),
        "observation_map_tangent_error_max": float(
            result["observation_map_tangent_error_max"].numpy()
        ),
        "must_compile": must_compile,
    }


COMPARE_KEYS = (
    "pre_flow",
    "d_pre_flow",
    "children",
    "d_children",
    "prior_observation_logits",
    "d_prior_observation_logits",
    "observation_log_density",
    "d_observation_log_density",
    "posterior_logits",
    "d_posterior_logits",
    "posterior_weights",
    "d_posterior_weights",
    "predicted_covariances",
    "post_covariances",
    "states_after_reset",
    "d_states_after_reset",
    "value",
    "score",
)

STAGE_KEYS = (
    "pre_flow",
    "children",
    "prior_observation_logits",
    "observation_log_density",
    "posterior_logits",
    "posterior_weights",
    "predicted_covariances",
    "post_covariances",
    "states_after_reset",
)


def _errors(left, right):
    return {
        key: float(
            tf.reduce_max(
                tf.abs(left[key] - tf.cast(right[key], left[key].dtype))
            ).numpy()
        )
        for key in COMPARE_KEYS
    }


def _must_compile(kernel) -> bool:
    concrete = kernel.get_concrete_function()
    return bool(concrete.function_def.attr["_XlaMustCompile"].b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=2)
    arguments = parser.parse_args()
    if arguments.particle_count < 4 or arguments.particle_count % 4 != 0:
        raise ValueError("particle-count must be a positive multiple of four")
    if arguments.horizon < 2:
        raise ValueError("horizon must be at least two")

    wall_start = time.perf_counter()
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    logical_devices = tf.config.list_logical_devices("GPU")
    if len(logical_devices) != 1:
        raise RuntimeError("calibration requires exactly one visible logical GPU")

    from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
        INTEGRATED_OBSERVATION_CLASSIFICATION,
        INTEGRATED_OBSERVATION_ROUTE_ID,
    )

    particle_count = arguments.particle_count
    horizon = arguments.horizon
    dtype32 = tf.float32
    model32, options32, inputs32 = _fixture(dtype32, particle_count, horizon)

    # Compile and execute the no-TF32 baseline before switching the global
    # setting. This makes its graph/XLA pair the local reference arm.
    tf.config.experimental.enable_tensor_float_32_execution(False)
    off_graph_kernel = _kernel(
        dtype32,
        model32,
        options32,
        particle_count,
        horizon,
        jit_compile=False,
    )
    off_xla_kernel = _kernel(
        dtype32,
        model32,
        options32,
        particle_count,
        horizon,
        jit_compile=True,
    )
    off_graph = _run(off_graph_kernel, inputs32)
    off_xla = _run(off_xla_kernel, inputs32)
    off_xla_must_compile = _must_compile(off_xla_kernel)

    # Build the TF32 pair after the setting change; they use exactly the same
    # fixed tensors and captured numerical controls as the baseline pair.
    tf.config.experimental.enable_tensor_float_32_execution(True)
    on_graph_kernel = _kernel(
        dtype32,
        model32,
        options32,
        particle_count,
        horizon,
        jit_compile=False,
    )
    on_xla_kernel = _kernel(
        dtype32,
        model32,
        options32,
        particle_count,
        horizon,
        jit_compile=True,
    )
    on_graph = _run(on_graph_kernel, inputs32)
    on_xla = _run(on_xla_kernel, inputs32)
    on_xla_must_compile = _must_compile(on_xla_kernel)

    # A float64 XLA arm calibrates ordinary float32 roundoff without changing
    # the fixed random stream: it consumes the float32 inputs cast to float64.
    model64, options64, _ = _fixture(tf.float64, particle_count, horizon)
    inputs64 = tuple(tf.cast(value, tf.float64) for value in inputs32)
    f64_xla_kernel = _kernel(
        tf.float64,
        model64,
        options64,
        particle_count,
        horizon,
        jit_compile=True,
    )
    f64_xla = _run(f64_xla_kernel, inputs64)
    f64_xla_must_compile = _must_compile(f64_xla_kernel)

    thresholds = {
        "value": 2.0e-4,
        "score": 5.0e-4,
        "states_after_reset": 5.0e-4,
        "posterior_weights": 2.0e-4,
    }
    off_graph_xla_errors = _errors(off_graph, off_xla)
    on_graph_xla_errors = _errors(on_graph, on_xla)
    on_xla_off_xla_errors = _errors(on_xla, off_xla)
    off_xla_f64_xla_errors = _errors(off_xla, f64_xla)
    first_stage_over_threshold = next(
        (
            key
            for key in STAGE_KEYS
            if on_graph_xla_errors[key] > 1.0e-4
        ),
        None,
    )
    baseline_pass = (
        _finite_valid(off_graph)
        and _finite_valid(off_xla)
        and off_xla_must_compile
        and all(
            off_graph_xla_errors[key] <= thresholds[key]
            for key in thresholds
        )
    )
    tf32_parity_veto = any(
        on_graph_xla_errors[key] > thresholds[key] for key in thresholds
    )

    root = Path(__file__).resolve().parents[2]
    source_paths = (
        Path(SMOKE_PATH),
        Path("bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py"),
        Path("bayesfilter/highdim/ledh_canonical_score_tf.py"),
        Path(PLAN_PATH),
    )
    result = {
        "schema": "bayesfilter.ledh_younis_kdm_phase4a_tf32_calibration.v1",
        "status": "TF32_NUMERICAL_PARITY_VETO" if tf32_parity_veto else "NO_TF32_PARITY_VETO",
        "role": "bounded_engineering_calibration_not_research_evidence",
        "question": "Does the TF32 graph/XLA mismatch arise before or after a declared route stage, and how large is TF32 drift relative to no-TF32 and float64 arms?",
        "route_id": INTEGRATED_OBSERVATION_ROUTE_ID,
        "route_classification": INTEGRATED_OBSERVATION_CLASSIFICATION,
        "particle_count": particle_count,
        "horizon": horizon,
        "dtype": "float32_primary_float64_reference",
        "tf32_setting_at_end": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "baseline_no_tf32_pass": baseline_pass,
        "tf32_graph_xla_parity_veto": tf32_parity_veto,
        "first_tf32_graph_xla_stage_over_1e-4": first_stage_over_threshold,
        "thresholds": thresholds,
        "arms": {
            "float32_no_tf32_graph": _summary(off_graph, must_compile=None),
            "float32_no_tf32_xla": _summary(
                off_xla, must_compile=off_xla_must_compile
            ),
            "float32_tf32_graph": _summary(on_graph, must_compile=None),
            "float32_tf32_xla": _summary(
                on_xla, must_compile=on_xla_must_compile
            ),
            "float64_tf32_setting_xla": _summary(
                f64_xla, must_compile=f64_xla_must_compile
            ),
        },
        "errors": {
            "float32_no_tf32_graph_vs_xla": off_graph_xla_errors,
            "float32_tf32_graph_vs_xla": on_graph_xla_errors,
            "float32_tf32_xla_vs_no_tf32_xla": on_xla_off_xla_errors,
            "float32_no_tf32_xla_vs_float64_xla": off_xla_f64_xla_errors,
        },
        "interpretation": {
            "stage_errors_are_explanatory_only": True,
            "no_tf32_graph_xla_pair_is_local_baseline": True,
            "float64_arm_is_roundoff_reference_not_oracle": True,
            "does_not_establish_score_accuracy_or_default_readiness": True,
        },
        "gpu_memory_policy": memory_policy,
        "logical_devices": [device.name for device in logical_devices],
        "output_device": str(on_xla["value"].device),
        "trust_basis": TRUST_BASIS,
        "tensorflow_version": tf.__version__,
        "python": sys.executable,
        "environment_prefix": sys.prefix,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
        "command": shlex.join((sys.executable, *sys.argv)),
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "baseline_no_tf32_pass": baseline_pass,
                "first_stage_over_1e-4": first_stage_over_threshold,
                "errors": {
                    "tf32_graph_vs_xla": on_graph_xla_errors,
                    "tf32_xla_vs_no_tf32_xla": on_xla_off_xla_errors,
                },
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    if not baseline_pass:
        raise SystemExit("no-TF32 baseline failed; calibration is invalid")


if __name__ == "__main__":
    main()
