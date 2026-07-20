"""Compare LEDH-PFPF-OT gradients across transport data structures.

This is a focused HMC-facing diagnostic.  It compares value and score for:

* dense transport with a fixed pre-flow tensor;
* streaming transport with the same fixed pre-flow tensor;
* streaming transport with an equivalent callback returning the same pre-flow;
* optionally, streaming transport with a dynamic callback using current
  particles.

The first three arms are intended to isolate data-structure differences.  The
dynamic callback is a separate route diagnostic because it changes the value
recursion.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="cpu")
_PRE_PARSER.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _UNKNOWN = _PRE_PARSER.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf


DEFAULT_DTYPE_NAME = "float32"
DEFAULT_TF32_MODE = "enabled"
DTYPE = tf.float32

NONCLAIMS = (
    "focused deterministic gradient-structure diagnostic only",
    "dense arm is a small-reference oracle, not a scalable implementation",
    "no HMC convergence or energy-conservation claim",
    "no posterior validity claim",
    "single fixture and single seed only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--obs-dim", type=int, default=4)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=4)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=32)
    parser.add_argument("--col-chunk-size", type=int, default=32)
    parser.add_argument("--particle-chunk-size", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--no-jit-compile", action="store_true")
    parser.add_argument("--dtype", choices=("float64", "float32"), default=DEFAULT_DTYPE_NAME)
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default=DEFAULT_TF32_MODE,
    )
    parser.add_argument("--include-dynamic-callback", action="store_true")
    parser.add_argument("--structure-value-atol", type=float, default=1.0e-6)
    parser.add_argument("--structure-value-rtol", type=float, default=1.0e-6)
    parser.add_argument("--structure-score-atol", type=float, default=1.0e-5)
    parser.add_argument("--structure-score-rtol", type=float, default=1.0e-5)
    parser.add_argument("--device", default="/CPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.state_dim <= 0 or args.obs_dim <= 0:
        raise ValueError("state_dim and obs_dim must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update({
        "dtype": args.dtype,
        "tf_dtype": DTYPE.name,
        "tf32_mode": args.tf32_mode,
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
    })
    return metadata


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _fixture(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    batch_size = args.batch_size
    state_dim = args.state_dim
    obs_dim = args.obs_dim
    num_particles = args.num_particles
    time_steps = args.time_steps

    batch = np.arange(batch_size, dtype=np.float64)
    state_grid = np.linspace(-1.0, 1.0, state_dim, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, num_particles, dtype=np.float64)
    time_grid = np.arange(time_steps, dtype=np.float64)

    theta = np.stack(
        [
            np.ones(batch_size, dtype=np.float64),
            np.zeros(batch_size, dtype=np.float64),
            np.zeros(batch_size, dtype=np.float64),
        ],
        axis=1,
    )
    initial_particles = (
        0.08 * rng.standard_normal((batch_size, num_particles, state_dim))
        + 0.03 * state_grid[None, None, :]
        + 0.01 * particle_grid[None, :, None]
        + 0.0001 * batch[:, None, None]
    )
    diagonal = 0.86 + 0.08 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    base_transition = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        base_transition[row] = np.diag(diagonal + 0.00001 * row)
        base_transition[row] += 0.006 * np.eye(state_dim, k=1, dtype=np.float64)
        base_transition[row] += -0.004 * np.eye(state_dim, k=-1, dtype=np.float64)

    q_diag = 0.04 + 0.004 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    r_diag = 0.06 + 0.006 * np.linspace(0.0, 1.0, obs_dim, dtype=np.float64)
    base_transition_covariance = np.tile(np.diag(q_diag)[None, :, :], (batch_size, 1, 1))
    base_observation_covariance = np.tile(np.diag(r_diag)[None, :, :], (batch_size, 1, 1))

    observation_matrix = np.zeros((batch_size, obs_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        for obs_index in range(obs_dim):
            state_index = obs_index % state_dim
            observation_matrix[row, obs_index, state_index] = 1.0
            if state_dim > 1:
                observation_matrix[row, obs_index, (state_index + 1) % state_dim] = 0.025

    observations = 0.05 * np.sin(
        0.023 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0)
    )
    observations += 0.02 * np.cos(
        0.011 * time_grid[:, None] * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0)
    )
    time_wave = 0.012 * np.sin(0.017 * time_grid[:, None] * (np.arange(state_dim) + 1))
    particle_wave = 0.006 * np.cos(0.11 * particle_grid[:, None] * (np.arange(state_dim) + 1))

    if args.transport_policy == "active-all":
        fixed_resampling_mask = np.ones((batch_size, time_steps), dtype=bool)
    elif args.transport_policy == "active-odd":
        mask = (np.arange(time_steps)[None, :] % 2) == 1
        fixed_resampling_mask = np.broadcast_to(mask, (batch_size, time_steps)).copy()
    else:
        fixed_resampling_mask = np.zeros((batch_size, time_steps), dtype=bool)

    return {
        "theta": theta,
        "observations": observations,
        "initial_particles": initial_particles,
        "base_transition": base_transition,
        "base_transition_covariance": base_transition_covariance,
        "base_observation_covariance": base_observation_covariance,
        "observation_matrix": observation_matrix,
        "time_wave": time_wave,
        "particle_wave": particle_wave,
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _to_tensors(fixture: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        dtype = tf.bool if value.dtype == np.bool_ else DTYPE
        tensors[name] = tf.constant(value, dtype=dtype)
    return tensors


def _model(theta: tf.Tensor, tensors: dict[str, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    transition_matrix = theta[:, 0, None, None] * tensors["base_transition"]
    transition_covariance = tf.exp(theta[:, 1, None, None]) * tensors[
        "base_transition_covariance"
    ]
    observation_covariance = tf.exp(theta[:, 2, None, None]) * tensors[
        "base_observation_covariance"
    ]
    return transition_matrix, transition_covariance, observation_covariance


def _pre_flow_tensor(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    transition_matrix: tf.Tensor,
) -> tf.Tensor:
    del theta
    transitioned = tf.einsum("bnj,bdj->bnd", tensors["initial_particles"], transition_matrix)
    return (
        transitioned[:, None, :, :]
        + tensors["time_wave"][None, :, None, :]
        + tensors["particle_wave"][None, None, :, :]
    )


def _make_pre_flow_step_fn(
    tensors: dict[str, tf.Tensor],
    transition_matrix: tf.Tensor,
) -> Any:
    def _pre_flow_step(particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        transitioned = tf.einsum("bnj,bdj->bnd", particles, transition_matrix)
        return (
            transitioned
            + tensors["time_wave"][time_index][None, None, :]
            + tensors["particle_wave"][None, :, :]
        )

    return _pre_flow_step


def _make_equivalent_pre_flow_step_fn(pre_flow: tf.Tensor) -> Any:
    def _pre_flow_step(_particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        return pre_flow[:, time_index, :, :]

    return _pre_flow_step


def _make_observation_fn(observation_matrix: tf.Tensor) -> Any:
    def _observation(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    return _observation


def _make_observation_jacobian_fn(observation_matrix: tf.Tensor) -> Any:
    def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
        batch_size = points.shape[0]
        num_particles = points.shape[1]
        if batch_size is None or num_particles is None:
            raise ValueError("gradient fixture requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], DTYPE)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * np.pi, dtype=DTYPE))
        + logdet[:, None]
        + quad
    )


def _make_transition_log_density(
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
) -> Any:
    def _transition_log_density(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance)

    return _transition_log_density


def _make_observation_log_density(
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
) -> Any:
    def _observation_log_density(
        x: tf.Tensor,
        observation: tf.Tensor,
        _time_index: tf.Tensor,
    ) -> tf.Tensor:
        del _time_index
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(
            predicted - observation[None, None, :],
            observation_covariance,
        )

    return _observation_log_density


def _value_for_arm(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    arm_id: str,
) -> tf.Tensor:
    transition_matrix, transition_covariance, observation_covariance = _model(theta, tensors)
    pre_flow = _pre_flow_tensor(theta, tensors, transition_matrix)
    common = dict(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_covariance=observation_covariance,
        observation_fn=_make_observation_fn(tensors["observation_matrix"]),
        observation_jacobian_fn=_make_observation_jacobian_fn(tensors["observation_matrix"]),
        observation_residual_fn=_observation_residual,
        transition_log_density_fn=_make_transition_log_density(
            transition_matrix,
            transition_covariance,
        ),
        observation_log_density_fn=_make_observation_log_density(
            tensors["observation_matrix"],
            observation_covariance,
        ),
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )
    if arm_id == "original_dense_tensor":
        result = core_tf.batched_ledh_pfpf_ot_value_core_tf(
            pre_flow_particles=pre_flow,
            transport_gradient_mode="raw",
            transport_plan_mode="dense",
            **common,
        )
    elif arm_id == "streaming_dense_tensor":
        result = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            pre_flow_particles=pre_flow,
            transport_plan_mode="dense",
            particle_chunk_size=args.particle_chunk_size,
            return_history=False,
            **common,
        )
    elif arm_id == "streaming_streaming_tensor":
        result = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            pre_flow_particles=pre_flow,
            transport_plan_mode="streaming",
            particle_chunk_size=args.particle_chunk_size,
            return_history=False,
            **common,
        )
    elif arm_id == "streaming_streaming_equivalent_callback":
        result = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            pre_flow_step_fn=_make_equivalent_pre_flow_step_fn(pre_flow),
            transport_plan_mode="streaming",
            particle_chunk_size=args.particle_chunk_size,
            return_history=False,
            **common,
        )
    elif arm_id == "streaming_streaming_dynamic_callback":
        result = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
            pre_flow_step_fn=_make_pre_flow_step_fn(tensors, transition_matrix),
            transport_plan_mode="streaming",
            particle_chunk_size=args.particle_chunk_size,
            return_history=False,
            **common,
        )
    else:
        raise ValueError(f"unknown arm_id {arm_id!r}")
    return result.log_likelihood


def _score_for_arm(
    theta: tf.Tensor,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    arm_id: str,
) -> tuple[tf.Tensor, tf.Tensor]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = _value_for_arm(theta, tensors, args, arm_id)
        objective = tf.reduce_sum(value)
    score = tape.gradient(
        objective,
        theta,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    if score is None:
        score = tf.zeros_like(theta)
    return value, score


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _device_check(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU outputs, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _arm_ids(include_dynamic: bool) -> list[str]:
    arms = [
        "original_dense_tensor",
        "streaming_dense_tensor",
        "streaming_streaming_tensor",
        "streaming_streaming_equivalent_callback",
    ]
    if include_dynamic:
        arms.append("streaming_streaming_dynamic_callback")
    return arms


def _run_arm(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    arm_id: str,
) -> dict[str, Any]:
    theta = tensors["theta"]
    jit_compile = not args.no_jit_compile

    @tf.function(jit_compile=jit_compile, reduce_retracing=True)
    def compiled() -> tuple[tf.Tensor, tf.Tensor]:
        return _score_for_arm(theta, tensors, args, arm_id)

    start = time.perf_counter()
    value, score = compiled()
    _materialize(value, score)
    compile_and_first = time.perf_counter() - start
    for _ in range(args.warmups):
        _materialize(*compiled())
    timings: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        value, score = compiled()
        _materialize(value, score)
        timings.append(time.perf_counter() - start)
    value_np = value.numpy()
    score_np = score.numpy()
    devices = _device_check((value, score), args.expect_device_kind)
    finite = bool(np.isfinite(value_np).all() and np.isfinite(score_np).all())
    return {
        "arm_id": arm_id,
        "finite": finite,
        "value": value_np.tolist(),
        "score": score_np.tolist(),
        "value_preview": [float(v) for v in value_np.reshape(-1)[:8]],
        "score_preview": [float(v) for v in score_np.reshape(-1)[:8]],
        "compile_and_first_call_seconds": compile_and_first,
        "jit_compile": jit_compile,
        "warm_call_timings_seconds": timings,
        "warm_call_median_seconds": float(np.median(timings)) if timings else None,
        "output_devices": devices,
    }


def _drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float | list[int]]:
    delta = candidate - reference
    abs_delta = np.abs(delta)
    if reference.size == 0:
        return {
            "shape": list(reference.shape),
            "max_abs": 0.0,
            "rms_abs": 0.0,
            "max_relative_to_max1_abs_reference": 0.0,
        }
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(abs_delta)),
        "rms_abs": float(np.sqrt(np.mean(delta * delta))),
        "max_relative_to_max1_abs_reference": float(
            np.max(abs_delta / np.maximum(1.0, np.abs(reference)))
        ),
    }


def _score_geometry(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 or cand_norm == 0.0:
        cosine = float("nan")
    else:
        cosine = float(np.dot(ref, cand) / (ref_norm * cand_norm))
    return {
        "reference_l2_norm": ref_norm,
        "candidate_l2_norm": cand_norm,
        "norm_ratio_candidate_to_reference": cand_norm / ref_norm if ref_norm else float("nan"),
        "cosine_similarity": cosine,
    }


def _comparisons(arms: list[dict[str, Any]], reference_arm_id: str) -> list[dict[str, Any]]:
    by_arm = {arm["arm_id"]: arm for arm in arms}
    reference = by_arm[reference_arm_id]
    ref_value = np.asarray(reference["value"], dtype=np.float64)
    ref_score = np.asarray(reference["score"], dtype=np.float64)
    comparisons: list[dict[str, Any]] = []
    for arm in arms:
        if arm["arm_id"] == reference_arm_id:
            continue
        value = np.asarray(arm["value"], dtype=np.float64)
        score = np.asarray(arm["score"], dtype=np.float64)
        comparisons.append(
            {
                "arm_id": arm["arm_id"],
                "value_drift_vs_reference": _drift(ref_value, value),
                "score_drift_vs_reference": _drift(ref_score, score),
                "score_geometry_vs_reference": _score_geometry(ref_score, score),
            }
        )
    return comparisons


def _structure_passed(args: argparse.Namespace, comparisons: list[dict[str, Any]]) -> bool:
    primary_ids = {
        "streaming_dense_tensor",
        "streaming_streaming_tensor",
        "streaming_streaming_equivalent_callback",
    }
    for comparison in comparisons:
        if comparison["arm_id"] not in primary_ids:
            continue
        value = comparison["value_drift_vs_reference"]
        score = comparison["score_drift_vs_reference"]
        value_ok = (
            value["max_abs"] <= args.structure_value_atol
            or value["max_relative_to_max1_abs_reference"] <= args.structure_value_rtol
        )
        score_ok = (
            score["max_abs"] <= args.structure_score_atol
            or score["max_relative_to_max1_abs_reference"] <= args.structure_score_rtol
        )
        if not (value_ok and score_ok):
            return False
    return True


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# LEDH-PFPF-OT Gradient Structure Comparison",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Overall passed: `{result['overall_passed']}`",
        f"- Shape: `{result['shape']}`",
        f"- Precision: `{result['precision']}`",
        "",
        "## Arms",
        "",
        "| arm | finite | compile+first s | warm median s | score preview |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for arm in result["arms"]:
        lines.append(
            f"| {arm['arm_id']} | {arm['finite']} | "
            f"{arm['compile_and_first_call_seconds']:.6g} | "
            f"{arm['warm_call_median_seconds']} | `{arm['score_preview']}` |"
        )
    lines.extend(
        [
            "",
            "## Drift Vs Original Dense Tensor",
            "",
            "| arm | value max abs | score max abs | score rel | score cosine | norm ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for comparison in result["comparisons"]:
        value = comparison["value_drift_vs_reference"]
        score = comparison["score_drift_vs_reference"]
        geom = comparison["score_geometry_vs_reference"]
        lines.append(
            f"| {comparison['arm_id']} | {value['max_abs']:.6g} | "
            f"{score['max_abs']:.6g} | "
            f"{score['max_relative_to_max1_abs_reference']:.6g} | "
            f"{geom['cosine_similarity']:.9g} | "
            f"{geom['norm_ratio_candidate_to_reference']:.9g} |"
        )
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    tensors = _to_tensors(_fixture(args))
    with tf.device(args.device):
        arms = [
            _run_arm(tensors, args, arm_id)
            for arm_id in _arm_ids(args.include_dynamic_callback)
        ]
    comparisons = _comparisons(arms, "original_dense_tensor")
    finite_all = all(arm["finite"] for arm in arms)
    structure_passed = _structure_passed(args, comparisons)
    result: dict[str, Any] = {
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "precision": precision,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
            "parameter_dim": 3,
        },
        "transport": {
            "policy": args.transport_policy,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
        },
        "tolerances": {
            "structure_value_atol": args.structure_value_atol,
            "structure_value_rtol": args.structure_value_rtol,
            "structure_score_atol": args.structure_score_atol,
            "structure_score_rtol": args.structure_score_rtol,
        },
        "jit_compile": not args.no_jit_compile,
        "reference_arm": "original_dense_tensor",
        "arms": arms,
        "comparisons": comparisons,
        "finite_all": finite_all,
        "structure_passed": structure_passed,
        "overall_passed": bool(finite_all and structure_passed),
        "nonclaims": list(NONCLAIMS),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = Path(args.markdown_output)
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_output, result, output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
