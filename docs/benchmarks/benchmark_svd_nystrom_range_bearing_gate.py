"""Range-bearing nonlinear Gaussian gate for SVD-Nystrom no-HMC promotion.

This benchmark runs the same DPF LEDH-PFPF-OT streaming comparator and locked
SVD-Nystrom route on the fixed range-bearing fixture.  It is a P04 evidence
artifact only; it does not certify posterior correctness, default readiness, or
scientific superiority.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, NamedTuple


_PRE = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
_PRE.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from docs.benchmarks import benchmark_actual_sir_nystrom_compiled_redo as redo_base  # noqa: E402
from experiments.dpf_implementation.fixtures import range_bearing  # noqa: E402
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import nystrom_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling.nystrom_transport_tf import (  # noqa: E402
    nystrom_transport_resample_tensors_tf,
)


NYSTROM_RESIDUAL_THRESHOLD = 5.0e-2
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-5
ESS_FRACTION_MIN_THRESHOLD = 0.005
LOG_LIKELIHOOD_NORMALIZED_DELTA_THRESHOLD = 0.05
NONCLAIMS = (
    "P04 range-bearing nonlinear Gaussian DPF gate only",
    "no default promotion claim",
    "no posterior correctness claim",
    "no statistical superiority claim",
    "no HMC readiness claim",
    "no broad nonlinear validity claim",
)


class NystromValueTensors(NamedTuple):
    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    final_log_weights: tf.Tensor
    max_row_residual: tf.Tensor
    max_column_residual: tf.Tensor
    finite_factors: tf.Tensor
    finite_particles: tf.Tensor
    iterations_used_max: tf.Tensor
    route_invocations: tf.Tensor
    min_kernel_denominator: tf.Tensor
    denominator_floor_hits: tf.Tensor
    max_abs_log_scaling_gauge_shift: tf.Tensor
    scaling_normalization_applications: tf.Tensor
    max_factor_diag_error: tf.Tensor
    min_factor_diagonal: tf.Tensor
    max_factor_diagonal: tf.Tensor
    landmark_core_min_eigenvalue: tf.Tensor
    landmark_core_max_eigenvalue: tf.Tensor
    landmark_core_condition_proxy: tf.Tensor
    landmark_core_effective_rank_min: tf.Tensor
    left_factor_min: tf.Tensor
    left_factor_max: tf.Tensor
    core_matrix_min: tf.Tensor
    core_matrix_max: tf.Tensor
    raw_kernel_min: tf.Tensor
    projected_kernel_min: tf.Tensor
    projection_floor_hits: tf.Tensor
    scaling_u_min: tf.Tensor
    scaling_u_max: tf.Tensor
    scaling_v_min: tf.Tensor
    scaling_v_max: tf.Tensor


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--route", choices=("streaming", "nystrom", "both"), default="both")
    parser.add_argument("--fixture-id", default="range_bearing_gaussian_moderate")
    parser.add_argument("--seed", type=int, default=84000)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=4096)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=512)
    parser.add_argument("--col-chunk-size", type=int, default=512)
    parser.add_argument("--particle-chunk-size", type=int, default=512)
    parser.add_argument("--nystrom-rank", type=int, default=32)
    parser.add_argument("--nystrom-epsilon", type=float, default=0.5)
    parser.add_argument("--nystrom-max-iterations", type=int, default=160)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--nystrom-cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--nystrom-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--nystrom-diagnostics", action="store_true")
    parser.add_argument(
        "--nystrom-core-solver",
        choices=("cholesky", "eigh_truncated", "svd_truncated"),
        default="svd_truncated",
    )
    parser.add_argument("--nystrom-core-rcond", type=float, default=1.0e-6)
    parser.add_argument(
        "--nystrom-kernel-mode",
        choices=("raw", "positive_projected"),
        default="raw",
    )
    parser.add_argument(
        "--nystrom-scaling-normalization",
        choices=("none", "balanced"),
        default="none",
    )
    parser.add_argument("--history-mode", choices=("full", "value-only"), default="full")
    parser.add_argument(
        "--paired-threshold-mode",
        choices=("gate", "record-only"),
        default="gate",
    )
    parser.add_argument(
        "--paired-delta-threshold",
        type=float,
        default=LOG_LIKELIHOOD_NORMALIZED_DELTA_THRESHOLD,
    )
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--selected-physical-gpu", default=None)
    parser.add_argument("--gpu-selection-note", default=None)
    parser.add_argument("--phase-id", default="SVD-NYSTROM-NOHMC-PROMOTION-P04-RANGE-BEARING")
    parser.add_argument("--capture-route-exceptions", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args(argv)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.fixture_id != "range_bearing_gaussian_moderate":
        raise ValueError("only range_bearing_gaussian_moderate is authorized")
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.time_steps > range_bearing.make_fixture(args.fixture_id).horizon:
        raise ValueError("time_steps exceeds range-bearing fixture horizon")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.nystrom_rank <= 0 or args.nystrom_rank > args.num_particles:
        raise ValueError("nystrom_rank must be positive and <= num_particles")
    if args.sinkhorn_iterations <= 0 or args.nystrom_max_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    for name in (
        "sinkhorn_epsilon",
        "annealed_scaling",
        "annealed_convergence_threshold",
        "nystrom_epsilon",
        "nystrom_convergence_threshold",
        "nystrom_cholesky_jitter",
        "nystrom_denominator_floor",
        "nystrom_core_rcond",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.paired_delta_threshold) or args.paired_delta_threshold <= 0.0:
        raise ValueError("paired_delta_threshold must be finite and positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")


def _tf_dtype(dtype_name: str) -> tf.DType:
    return tf.float64 if dtype_name == "float64" else tf.float32


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _tf_dtype(args.dtype)
    core_tf.DTYPE = dtype
    streaming_tf.DTYPE = dtype
    nystrom_transport_tf.DTYPE = dtype
    nystrom_transport_tf.DEFAULT_DTYPE = dtype
    if args.tf32_mode == "enabled":
        tf.config.experimental.enable_tensor_float_32_execution(True)
    elif args.tf32_mode == "disabled":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        tf32_enabled: bool | str = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    except Exception as exc:  # noqa: BLE001
        tf32_enabled = f"unavailable:{type(exc).__name__}"
    return {
        "dtype": args.dtype,
        "tf_dtype": dtype.name,
        "tf32_mode": args.tf32_mode,
        "tf32_execution_enabled": tf32_enabled,
        "default_execution_target": "gpu",
        "default_algorithm_target": "ledh_pfpf_ot_tf32",
    }


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _range_bearing_observation_tf(points: tf.Tensor) -> tf.Tensor:
    px = points[..., 0]
    py = points[..., 1]
    obs = tf.stack(
        [
            tf.sqrt(tf.square(px) + tf.square(py) + tf.cast(1.0e-12, points.dtype)),
            tf.atan2(py, px),
        ],
        axis=-1,
    )
    return obs


def _wrap_angle_tf(value: tf.Tensor) -> tf.Tensor:
    pi = tf.cast(math.pi, value.dtype)
    two_pi = tf.cast(2.0 * math.pi, value.dtype)
    return tf.math.floormod(value + pi, two_pi) - pi


def _observation_residual_tf(predicted: tf.Tensor, observed: tf.Tensor) -> tf.Tensor:
    residual = observed[None, None, :] - predicted
    range_resid = residual[..., 0]
    bearing_resid = _wrap_angle_tf(residual[..., 1])
    return tf.stack([range_resid, bearing_resid], axis=-1)


def _range_bearing_jacobian_tf(points: tf.Tensor) -> tf.Tensor:
    dtype = points.dtype
    px = points[..., 0]
    py = points[..., 1]
    radius_sq = tf.square(px) + tf.square(py) + tf.cast(1.0e-12, dtype)
    radius = tf.sqrt(radius_sq)
    zeros = tf.zeros_like(px)
    row_range = tf.stack([px / radius, py / radius, zeros, zeros], axis=-1)
    row_bearing = tf.stack([-py / radius_sq, px / radius_sq, zeros, zeros], axis=-1)
    return tf.stack([row_range, row_bearing], axis=-2)


def _fixture_tensors(args: argparse.Namespace, dtype: tf.DType) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = range_bearing.make_fixture(args.fixture_id)
    rng = np.random.default_rng(int(args.seed))
    initial = range_bearing.initial_sample(rng, args.num_particles, fixture)
    observations = fixture.observations[: args.time_steps]
    fixed_mask = tf.ones([1, args.time_steps], dtype=tf.bool)
    tensors = {
        "observations": tf.constant(observations, dtype=dtype),
        "initial_particles": tf.constant(initial[None, :, :], dtype=dtype),
        "transition_matrix": tf.constant(fixture.A[None, :, :], dtype=dtype),
        "transition_covariance": tf.constant(fixture.Q[None, :, :], dtype=dtype),
        "observation_covariance": tf.constant(fixture.R[None, :, :], dtype=dtype),
        "fixed_resampling_mask": fixed_mask,
    }
    semantics = {
        "fixture_id": fixture.name,
        "fixture_generation_seed": fixture.fixture_generation_seed,
        "model_checksum": fixture.model_checksum,
        "observation_checksum": fixture.observation_checksum,
        "model_definition": fixture.model_definition(),
        "state_dim": fixture.state_dim,
        "obs_dim": fixture.obs_dim,
        "fixture_horizon": fixture.horizon,
        "time_steps_used": args.time_steps,
        "particle_seed": int(args.seed),
    }
    return tensors, semantics


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    dtype = residuals.dtype
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(int(residuals.shape[-1]), dtype)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * math.pi, dtype))
        + logdet[:, None]
        + quad
    )


def _callbacks(tensors: dict[str, tf.Tensor]) -> dict[str, Any]:
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]

    def pre_flow_step_fn(particles: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bnj,bdj->bnd", particles, transition_matrix)

    def prior_mean_fn(points: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bnj,bdj->bnd", points, transition_matrix)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return _range_bearing_observation_tf(points)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        return _range_bearing_jacobian_tf(points)

    def observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return _observation_residual_tf(h_ref, observation)

    def transition_log_density_fn(x_next: tf.Tensor, x_prev: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance)

    def observation_log_density_fn(x: tf.Tensor, observation: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        predicted = _range_bearing_observation_tf(x)
        residual = _observation_residual_tf(predicted, observation)
        return _batched_gaussian_logpdf(residual, observation_covariance)

    return {
        "pre_flow_step_fn": pre_flow_step_fn,
        "prior_mean_fn": prior_mean_fn,
        "observation_fn": observation_fn,
        "observation_jacobian_fn": observation_jacobian_fn,
        "observation_residual_fn": observation_residual_fn,
        "transition_log_density_fn": transition_log_density_fn,
        "observation_log_density_fn": observation_log_density_fn,
    }


def _nystrom_value_core(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> NystromValueTensors:
    observations = tensors["observations"]
    particles = tensors["initial_particles"]
    fixed_mask = tensors["fixed_resampling_mask"]
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    time_steps = int(observations.shape[0])
    dtype = particles.dtype
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    max_row_residual = tf.constant(0.0, dtype=dtype)
    max_column_residual = tf.constant(0.0, dtype=dtype)
    finite_factors = tf.constant(True)
    finite_particles = tf.constant(True)
    iterations_used_max = tf.constant(0, dtype=tf.int32)
    route_invocations = tf.constant(0, dtype=tf.int32)
    min_kernel_denominator = tf.constant(float("inf"), dtype=dtype)
    denominator_floor_hits = tf.constant(0.0, dtype=dtype)
    max_abs_log_scaling_gauge_shift = tf.constant(0.0, dtype=dtype)
    scaling_normalization_applications = tf.constant(0.0, dtype=dtype)
    max_factor_diag_error = tf.constant(0.0, dtype=dtype)
    min_factor_diagonal = tf.constant(float("inf"), dtype=dtype)
    max_factor_diagonal = tf.constant(float("-inf"), dtype=dtype)
    landmark_core_min_eigenvalue = tf.constant(float("inf"), dtype=dtype)
    landmark_core_max_eigenvalue = tf.constant(float("-inf"), dtype=dtype)
    landmark_core_condition_proxy = tf.constant(0.0, dtype=dtype)
    landmark_core_effective_rank_min = tf.constant(float("inf"), dtype=dtype)
    left_factor_min = tf.constant(float("inf"), dtype=dtype)
    left_factor_max = tf.constant(float("-inf"), dtype=dtype)
    core_matrix_min = tf.constant(float("inf"), dtype=dtype)
    core_matrix_max = tf.constant(float("-inf"), dtype=dtype)
    raw_kernel_min = tf.constant(float("inf"), dtype=dtype)
    projected_kernel_min = tf.constant(float("inf"), dtype=dtype)
    projection_floor_hits = tf.constant(0.0, dtype=dtype)
    scaling_u_min = tf.constant(float("inf"), dtype=dtype)
    scaling_u_max = tf.constant(float("-inf"), dtype=dtype)
    scaling_v_min = tf.constant(float("inf"), dtype=dtype)
    scaling_v_max = tf.constant(float("-inf"), dtype=dtype)

    if args.history_mode == "full":
        means_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size, state_dim]))
        vars_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size, state_dim]))
        ess_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size]))
    else:
        means_ta = tf.TensorArray(dtype=dtype, size=0)
        vars_ta = tf.TensorArray(dtype=dtype, size=0)
        ess_ta = tf.TensorArray(dtype=dtype, size=0)

    for time_index in range(time_steps):
        time_tensor = tf.constant(time_index, dtype=tf.int32)
        observation = observations[time_index]
        ancestors = particles
        pre_flow = callbacks["pre_flow_step_fn"](particles, time_tensor)
        step_prior_mean_fn = lambda points, t=time_tensor: callbacks["prior_mean_fn"](points, t)
        flow = streaming_tf.batched_ledh_flow_streaming_particles_tf(
            pre_flow_particles=pre_flow,
            ancestors=ancestors,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=callbacks["observation_fn"],
            observation_jacobian_fn=callbacks["observation_jacobian_fn"],
            observation_residual_fn=callbacks["observation_residual_fn"],
            prior_mean_fn=step_prior_mean_fn,
            particle_chunk_size=args.particle_chunk_size,
        )
        post_flow = flow.post_flow_particles
        corrected_log_weights = (
            log_weights
            + callbacks["transition_log_density_fn"](post_flow, ancestors, time_tensor)
            + callbacks["observation_log_density_fn"](post_flow, observation, time_tensor)
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)
        log_likelihood += incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        if args.history_mode == "full":
            mean, variance = core_tf._weighted_mean_and_variance(post_flow, weights)
            means_ta = means_ta.write(time_index, mean)
            vars_ta = vars_ta.write(time_index, variance)
            ess_ta = ess_ta.write(time_index, ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, core_tf._log_weight_floor()))
        mask = fixed_mask[:, time_index]

        def do_transport():
            resampled = nystrom_transport_resample_tensors_tf(
                post_flow,
                normalized_log_weights,
                rank=args.nystrom_rank,
                epsilon=args.nystrom_epsilon,
                max_iterations=args.nystrom_max_iterations,
                convergence_threshold=args.nystrom_convergence_threshold,
                cholesky_jitter=args.nystrom_cholesky_jitter,
                denominator_floor=args.nystrom_denominator_floor,
                core_solver=args.nystrom_core_solver,
                core_rcond=args.nystrom_core_rcond,
                kernel_mode=args.nystrom_kernel_mode,
                scaling_normalization=args.nystrom_scaling_normalization,
                diagnostics_enabled=args.nystrom_diagnostics,
            )
            return (
                tf.cast(resampled.particles, dtype),
                tf.cast(resampled.log_weights, dtype),
                tf.cast(resampled.max_row_residual, dtype),
                tf.cast(resampled.max_column_residual, dtype),
                resampled.finite_factors,
                resampled.finite_particles,
                resampled.iterations_used,
                tf.constant(1, dtype=tf.int32),
                tf.cast(resampled.min_kernel_denominator, dtype),
                tf.cast(resampled.denominator_floor_hits, dtype),
                tf.cast(resampled.max_abs_log_scaling_gauge_shift, dtype),
                tf.cast(resampled.scaling_normalization_applications, dtype),
                tf.cast(resampled.max_factor_diag_error, dtype),
                tf.cast(resampled.min_factor_diagonal, dtype),
                tf.cast(resampled.max_factor_diagonal, dtype),
                tf.cast(resampled.landmark_core_min_eigenvalue, dtype),
                tf.cast(resampled.landmark_core_max_eigenvalue, dtype),
                tf.cast(resampled.landmark_core_condition_proxy, dtype),
                tf.cast(resampled.landmark_core_effective_rank, dtype),
                tf.cast(resampled.left_factor_min, dtype),
                tf.cast(resampled.left_factor_max, dtype),
                tf.cast(resampled.core_matrix_min, dtype),
                tf.cast(resampled.core_matrix_max, dtype),
                tf.cast(resampled.raw_kernel_min, dtype),
                tf.cast(resampled.projected_kernel_min, dtype),
                tf.cast(resampled.projection_floor_hits, dtype),
                tf.cast(resampled.scaling_u_min, dtype),
                tf.cast(resampled.scaling_u_max, dtype),
                tf.cast(resampled.scaling_v_min, dtype),
                tf.cast(resampled.scaling_v_max, dtype),
            )

        def skip_transport():
            nan = tf.constant(float("nan"), dtype=dtype)
            return (
                post_flow,
                normalized_log_weights,
                tf.constant(0.0, dtype=dtype),
                tf.constant(0.0, dtype=dtype),
                tf.constant(True),
                tf.constant(True),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                nan,
                tf.constant(0.0, dtype=dtype),
                tf.constant(0.0, dtype=dtype),
                tf.constant(0.0, dtype=dtype),
                nan,
                nan,
                nan,
                nan,
                nan,
                tf.constant(0.0, dtype=dtype),
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
            )

        if args.transport_policy == "active-all":
            step = do_transport()
        elif args.transport_policy == "no-resampling":
            step = skip_transport()
        else:
            step = tf.cond(tf.reduce_any(mask), do_transport, skip_transport)
        (
            particles,
            log_weights,
            row_residual,
            column_residual,
            step_finite_factors,
            step_finite_particles,
            step_iterations,
            step_invocations,
            step_min_kernel_denominator,
            step_denominator_floor_hits,
            step_max_abs_log_scaling_gauge_shift,
            step_scaling_normalization_applications,
            step_max_factor_diag_error,
            step_min_factor_diagonal,
            step_max_factor_diagonal,
            step_landmark_core_min_eigenvalue,
            step_landmark_core_max_eigenvalue,
            step_landmark_core_condition_proxy,
            step_landmark_core_effective_rank,
            step_left_factor_min,
            step_left_factor_max,
            step_core_matrix_min,
            step_core_matrix_max,
            step_raw_kernel_min,
            step_projected_kernel_min,
            step_projection_floor_hits,
            step_scaling_u_min,
            step_scaling_u_max,
            step_scaling_v_min,
            step_scaling_v_max,
        ) = step
        has_invocation = step_invocations > tf.constant(0, dtype=tf.int32)
        max_row_residual = tf.maximum(max_row_residual, row_residual)
        max_column_residual = tf.maximum(max_column_residual, column_residual)
        finite_factors = tf.logical_and(finite_factors, step_finite_factors)
        finite_particles = tf.logical_and(finite_particles, step_finite_particles)
        iterations_used_max = tf.maximum(iterations_used_max, step_iterations)
        route_invocations += step_invocations
        min_kernel_denominator = tf.where(
            has_invocation,
            tf.minimum(min_kernel_denominator, step_min_kernel_denominator),
            min_kernel_denominator,
        )
        denominator_floor_hits += tf.where(
            has_invocation,
            step_denominator_floor_hits,
            tf.constant(0.0, dtype=dtype),
        )
        max_abs_log_scaling_gauge_shift = tf.where(
            has_invocation,
            tf.maximum(max_abs_log_scaling_gauge_shift, step_max_abs_log_scaling_gauge_shift),
            max_abs_log_scaling_gauge_shift,
        )
        scaling_normalization_applications += tf.where(
            has_invocation,
            step_scaling_normalization_applications,
            tf.constant(0.0, dtype=dtype),
        )
        max_factor_diag_error = tf.where(
            has_invocation,
            tf.maximum(max_factor_diag_error, step_max_factor_diag_error),
            max_factor_diag_error,
        )
        min_factor_diagonal = tf.where(
            has_invocation,
            tf.minimum(min_factor_diagonal, step_min_factor_diagonal),
            min_factor_diagonal,
        )
        max_factor_diagonal = tf.where(
            has_invocation,
            tf.maximum(max_factor_diagonal, step_max_factor_diagonal),
            max_factor_diagonal,
        )
        landmark_core_min_eigenvalue = tf.where(
            has_invocation,
            tf.minimum(landmark_core_min_eigenvalue, step_landmark_core_min_eigenvalue),
            landmark_core_min_eigenvalue,
        )
        landmark_core_max_eigenvalue = tf.where(
            has_invocation,
            tf.maximum(landmark_core_max_eigenvalue, step_landmark_core_max_eigenvalue),
            landmark_core_max_eigenvalue,
        )
        landmark_core_condition_proxy = tf.where(
            has_invocation,
            tf.maximum(landmark_core_condition_proxy, step_landmark_core_condition_proxy),
            landmark_core_condition_proxy,
        )
        landmark_core_effective_rank_min = tf.where(
            has_invocation,
            tf.minimum(landmark_core_effective_rank_min, step_landmark_core_effective_rank),
            landmark_core_effective_rank_min,
        )
        left_factor_min = tf.where(has_invocation, tf.minimum(left_factor_min, step_left_factor_min), left_factor_min)
        left_factor_max = tf.where(has_invocation, tf.maximum(left_factor_max, step_left_factor_max), left_factor_max)
        core_matrix_min = tf.where(has_invocation, tf.minimum(core_matrix_min, step_core_matrix_min), core_matrix_min)
        core_matrix_max = tf.where(has_invocation, tf.maximum(core_matrix_max, step_core_matrix_max), core_matrix_max)
        raw_kernel_min = tf.where(has_invocation, tf.minimum(raw_kernel_min, step_raw_kernel_min), raw_kernel_min)
        projected_kernel_min = tf.where(
            has_invocation,
            tf.minimum(projected_kernel_min, step_projected_kernel_min),
            projected_kernel_min,
        )
        projection_floor_hits += tf.where(
            has_invocation,
            step_projection_floor_hits,
            tf.constant(0.0, dtype=dtype),
        )
        scaling_u_min = tf.where(has_invocation, tf.minimum(scaling_u_min, step_scaling_u_min), scaling_u_min)
        scaling_u_max = tf.where(has_invocation, tf.maximum(scaling_u_max, step_scaling_u_max), scaling_u_max)
        scaling_v_min = tf.where(has_invocation, tf.minimum(scaling_v_min, step_scaling_v_min), scaling_v_min)
        scaling_v_max = tf.where(has_invocation, tf.maximum(scaling_v_max, step_scaling_v_max), scaling_v_max)

    if args.history_mode == "full":
        filtered_means = means_ta.stack()
        filtered_variances = vars_ta.stack()
        ess_by_time = ess_ta.stack()
        filtered_means.set_shape([time_steps, batch_size, state_dim])
        filtered_variances.set_shape([time_steps, batch_size, state_dim])
        ess_by_time.set_shape([time_steps, batch_size])
    else:
        filtered_means = tf.zeros([0, batch_size, state_dim], dtype=dtype)
        filtered_variances = tf.zeros([0, batch_size, state_dim], dtype=dtype)
        ess_by_time = tf.zeros([0, batch_size], dtype=dtype)

    return NystromValueTensors(
        log_likelihood=log_likelihood,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        ess_by_time=ess_by_time,
        final_log_weights=log_weights,
        max_row_residual=max_row_residual,
        max_column_residual=max_column_residual,
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        iterations_used_max=iterations_used_max,
        route_invocations=route_invocations,
        min_kernel_denominator=min_kernel_denominator,
        denominator_floor_hits=denominator_floor_hits,
        max_abs_log_scaling_gauge_shift=max_abs_log_scaling_gauge_shift,
        scaling_normalization_applications=scaling_normalization_applications,
        max_factor_diag_error=max_factor_diag_error,
        min_factor_diagonal=min_factor_diagonal,
        max_factor_diagonal=max_factor_diagonal,
        landmark_core_min_eigenvalue=landmark_core_min_eigenvalue,
        landmark_core_max_eigenvalue=landmark_core_max_eigenvalue,
        landmark_core_condition_proxy=landmark_core_condition_proxy,
        landmark_core_effective_rank_min=landmark_core_effective_rank_min,
        left_factor_min=left_factor_min,
        left_factor_max=left_factor_max,
        core_matrix_min=core_matrix_min,
        core_matrix_max=core_matrix_max,
        raw_kernel_min=raw_kernel_min,
        projected_kernel_min=projected_kernel_min,
        projection_floor_hits=projection_floor_hits,
        scaling_u_min=scaling_u_min,
        scaling_u_max=scaling_u_max,
        scaling_v_min=scaling_v_min,
        scaling_v_max=scaling_v_max,
    )


def _streaming_outputs(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    value = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=tensors["transition_matrix"],
        transition_covariance=tensors["transition_covariance"],
        observation_covariance=tensors["observation_covariance"],
        observation_fn=callbacks["observation_fn"],
        observation_jacobian_fn=callbacks["observation_jacobian_fn"],
        observation_residual_fn=callbacks["observation_residual_fn"],
        transition_log_density_fn=callbacks["transition_log_density_fn"],
        observation_log_density_fn=callbacks["observation_log_density_fn"],
        prior_mean_fn=callbacks["prior_mean_fn"],
        pre_flow_step_fn=callbacks["pre_flow_step_fn"],
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=args.history_mode == "full",
    )
    return value.log_likelihood, value.filtered_means, value.filtered_variances, value.ess_by_time


def _route_names(route: str) -> list[str]:
    return ["streaming", "nystrom"] if route == "both" else [route]


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "max": max(timings),
    }


def _materialize(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _float(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _as_float_list(value: tf.Tensor) -> Any:
    return tf.cast(value, tf.float64).numpy().tolist()


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _validate_device(outputs: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect_device_kind == "gpu" and not all("GPU" in device.upper() for device in devices):
        raise RuntimeError(f"expected GPU outputs, got {devices}")
    if expect_device_kind == "cpu" and not all("CPU" in device.upper() for device in devices):
        raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _route_row(
    route: str,
    outputs: tuple[tf.Tensor, ...],
    *,
    compile_and_first: float,
    timings: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_devices = _validate_device((outputs[0],), args.expect_device_kind)
    hard_vetoes: list[str] = []
    for tensor, name in zip(outputs[:4], ("log_likelihood", "filtered_means", "filtered_variances", "ess_by_time")):
        if not _finite(tensor):
            hard_vetoes.append(f"nonfinite_{name}")
    row: dict[str, Any] = {
        "route": route,
        "status": "PASS",
        "hard_vetoes": hard_vetoes,
        "output_devices": output_devices,
        "compile_and_first_call_seconds": compile_and_first,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "log_likelihood": _as_float_list(outputs[0]),
        "history_returned": args.history_mode == "full",
    }
    if args.history_mode == "full":
        ess_fraction_min = _float(tf.reduce_min(outputs[3])) / float(args.num_particles)
        row.update(
            {
                "filtered_means": _as_float_list(outputs[1]),
                "filtered_variances": _as_float_list(outputs[2]),
                "ess_by_time": _as_float_list(outputs[3]),
                "ess_min": _float(tf.reduce_min(outputs[3])),
                "ess_fraction_min": ess_fraction_min,
            }
        )
        if ess_fraction_min < ESS_FRACTION_MIN_THRESHOLD:
            hard_vetoes.append("ess_fraction_min_threshold")
    if route == "nystrom":
        final_residual = _float(tf.reduce_max(tf.abs(tf.reduce_logsumexp(outputs[4], axis=1))))
        row.update(
            {
                "final_logsumexp_residual": final_residual,
                "max_row_residual": _float(outputs[5]),
                "max_column_residual": _float(outputs[6]),
                "finite_factors": bool(outputs[7].numpy()),
                "finite_particles": bool(outputs[8].numpy()),
                "iterations_used_max": int(outputs[9].numpy()),
                "route_invocations": int(outputs[10].numpy()),
                "nystrom_rank": args.nystrom_rank,
                "nystrom_epsilon": args.nystrom_epsilon,
                "nystrom_max_iterations": args.nystrom_max_iterations,
                "nystrom_diagnostics_enabled": args.nystrom_diagnostics,
                "nystrom_core_solver": args.nystrom_core_solver,
                "nystrom_core_rcond": args.nystrom_core_rcond,
                "nystrom_kernel_mode": args.nystrom_kernel_mode,
                "nystrom_kernel_mode_scope": (
                    "diagnostic_dense_positive_projection"
                    if args.nystrom_kernel_mode == "positive_projected"
                    else "raw_factor_application"
                ),
                "nystrom_scaling_normalization": args.nystrom_scaling_normalization,
                "min_kernel_denominator": _float(outputs[11]),
                "denominator_floor_hits": _float(outputs[12]),
                "max_abs_log_scaling_gauge_shift": _float(outputs[13]),
                "scaling_normalization_applications": _float(outputs[14]),
                "max_factor_diag_error": _float(outputs[15]),
                "min_factor_diagonal": _float(outputs[16]),
                "max_factor_diagonal": _float(outputs[17]),
                "landmark_core_min_eigenvalue": _float(outputs[18]),
                "landmark_core_max_eigenvalue": _float(outputs[19]),
                "landmark_core_condition_proxy": _float(outputs[20]),
                "landmark_core_effective_rank_min": _float(outputs[21]),
                "left_factor_min": _float(outputs[22]),
                "left_factor_max": _float(outputs[23]),
                "core_matrix_min": _float(outputs[24]),
                "core_matrix_max": _float(outputs[25]),
                "raw_kernel_min": _float(outputs[26]),
                "projected_kernel_min": _float(outputs[27]),
                "projection_floor_hits": _float(outputs[28]),
                "scaling_u_min": _float(outputs[29]),
                "scaling_u_max": _float(outputs[30]),
                "scaling_v_min": _float(outputs[31]),
                "scaling_v_max": _float(outputs[32]),
                "transport_object_kind": "nystrom_kernel_factors",
                "transport_matrix_materialized": False,
            }
        )
        if row["final_logsumexp_residual"] > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
            hard_vetoes.append("final_logsumexp_residual_threshold")
        if row["max_row_residual"] > NYSTROM_RESIDUAL_THRESHOLD:
            hard_vetoes.append("nystrom_row_residual_threshold")
        if row["max_column_residual"] > NYSTROM_RESIDUAL_THRESHOLD:
            hard_vetoes.append("nystrom_column_residual_threshold")
        if not row["finite_factors"]:
            hard_vetoes.append("nonfinite_nystrom_factors")
        if not row["finite_particles"]:
            hard_vetoes.append("nonfinite_nystrom_particles")
    row["status"] = "PASS" if not hard_vetoes else "FAIL"
    row["hard_vetoes"] = hard_vetoes
    return row


def _exception_traceback_tail(exc: BaseException, *, max_lines: int = 16) -> list[str]:
    return traceback.format_exception(type(exc), exc, exc.__traceback__)[-max_lines:]


def _route_exception_row(
    route: str,
    exc: BaseException,
    *,
    stage: str,
    compile_and_first: float | None,
    timings: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "route": route,
        "status": "FAIL",
        "hard_vetoes": ["route_exception"],
        "output_devices": [],
        "compile_and_first_call_seconds": compile_and_first,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "history_returned": args.history_mode == "full",
        "exception": {
            "stage": stage,
            "type": type(exc).__name__,
            "module": type(exc).__module__,
            "message": str(exc),
            "traceback_tail": _exception_traceback_tail(exc),
        },
    }


def _paired(rows: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    if set(rows) != {"streaming", "nystrom"}:
        return None
    if any("exception" in row or "log_likelihood" not in row for row in rows.values()):
        return None
    streaming_ll = tf.constant(rows["streaming"]["log_likelihood"], dtype=tf.float64)
    nystrom_ll = tf.constant(rows["nystrom"]["log_likelihood"], dtype=tf.float64)
    ll_delta = nystrom_ll - streaming_ll
    normalized = tf.abs(ll_delta) / tf.cast(args.time_steps * 2, tf.float64)
    paired = {
        "log_likelihood_delta_by_seed": ll_delta.numpy().tolist(),
        "normalized_abs_delta_by_seed": normalized.numpy().tolist(),
        "log_likelihood_max_abs_delta": _float(tf.reduce_max(tf.abs(ll_delta))),
        "log_likelihood_mean_abs_delta": _float(tf.reduce_mean(tf.abs(ll_delta))),
        "normalized_max_abs_delta": _float(tf.reduce_max(normalized)),
        "normalized_mean_abs_delta": _float(tf.reduce_mean(normalized)),
        "paired_threshold_mode": args.paired_threshold_mode,
        "paired_delta_threshold_role": _paired_delta_threshold_role(args),
        "thresholds": {
            "normalized_abs_delta": args.paired_delta_threshold,
            "normalized_abs_delta_role": _paired_delta_threshold_role(args),
        },
    }
    stream_median = rows["streaming"]["warm_call_timing_summary_seconds"].get("median")
    nystrom_median = rows["nystrom"]["warm_call_timing_summary_seconds"].get("median")
    paired["warm_median_streaming_over_nystrom_descriptive"] = (
        float(stream_median) / float(nystrom_median)
        if stream_median is not None and nystrom_median not in (None, 0.0)
        else None
    )
    return paired


def _paired_delta_threshold_role(args: argparse.Namespace) -> str:
    if args.paired_threshold_mode == "gate":
        return "hard_veto"
    return "record_only_descriptive_not_calibrated"


def _paired_vetoes(paired: dict[str, Any] | None, args: argparse.Namespace) -> list[str]:
    if paired is None:
        return []
    if args.paired_threshold_mode == "record-only":
        return []
    if paired["normalized_max_abs_delta"] > args.paired_delta_threshold:
        return ["paired_normalized_log_likelihood_delta"]
    return []


def _evidence_primary_pass(args: argparse.Namespace) -> str:
    if args.paired_threshold_mode == "record-only":
        return (
            "finite outputs, route metadata, no residual/ESS/log-weight veto; "
            "paired normalized deltas are recorded as descriptive scale evidence only"
        )
    return "finite outputs, route metadata, no residual/ESS/log-weight veto, and no normalized paired-delta exceedance"


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    smi_rows = redo_base._nvidia_smi_rows()
    selected_physical_gpu = redo_base._selected_physical_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"), smi_rows)
    dtype = _tf_dtype(args.dtype)
    tensors, fixture_semantics = _fixture_tensors(args, dtype)
    callbacks = _callbacks(tensors)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_streaming() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return _streaming_outputs(tensors, callbacks, args)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_nystrom() -> tuple[tf.Tensor, ...]:
        value = _nystrom_value_core(tensors, callbacks, args)
        return (
            value.log_likelihood,
            value.filtered_means,
            value.filtered_variances,
            value.ess_by_time,
            value.final_log_weights,
            value.max_row_residual,
            value.max_column_residual,
            value.finite_factors,
            value.finite_particles,
            value.iterations_used_max,
            value.route_invocations,
            value.min_kernel_denominator,
            value.denominator_floor_hits,
            value.max_abs_log_scaling_gauge_shift,
            value.scaling_normalization_applications,
            value.max_factor_diag_error,
            value.min_factor_diagonal,
            value.max_factor_diagonal,
            value.landmark_core_min_eigenvalue,
            value.landmark_core_max_eigenvalue,
            value.landmark_core_condition_proxy,
            value.landmark_core_effective_rank_min,
            value.left_factor_min,
            value.left_factor_max,
            value.core_matrix_min,
            value.core_matrix_max,
            value.raw_kernel_min,
            value.projected_kernel_min,
            value.projection_floor_hits,
            value.scaling_u_min,
            value.scaling_u_max,
            value.scaling_v_min,
            value.scaling_v_max,
        )

    route_fns = {"streaming": compiled_streaming, "nystrom": compiled_nystrom}
    rows_by_route: dict[str, dict[str, Any]] = {}
    started = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    start_all = time.perf_counter()
    memory_before = redo_base._gpu_memory_info()
    with tf.device(args.device):
        for route in _route_names(args.route):
            fn = route_fns[route]
            start = time.perf_counter()
            timings: list[float] = []
            compile_and_first: float | None = None
            try:
                outputs = fn()
                _materialize(*outputs)
                compile_and_first = time.perf_counter() - start
                for _ in range(args.warmups):
                    try:
                        _materialize(*fn())
                    except Exception as exc:  # noqa: BLE001
                        if not args.capture_route_exceptions:
                            raise
                        rows_by_route[route] = _route_exception_row(
                            route,
                            exc,
                            stage="warmup",
                            compile_and_first=compile_and_first,
                            timings=timings,
                            args=args,
                        )
                        break
                else:
                    for _ in range(args.repeats):
                        repeat_start = time.perf_counter()
                        try:
                            outputs = fn()
                            _materialize(*outputs)
                        except Exception as exc:  # noqa: BLE001
                            if not args.capture_route_exceptions:
                                raise
                            rows_by_route[route] = _route_exception_row(
                                route,
                                exc,
                                stage="repeat",
                                compile_and_first=compile_and_first,
                                timings=timings,
                                args=args,
                            )
                            break
                        timings.append(time.perf_counter() - repeat_start)
                    else:
                        rows_by_route[route] = _route_row(
                            route,
                            outputs,
                            compile_and_first=compile_and_first,
                            timings=timings,
                            args=args,
                        )
            except Exception as exc:  # noqa: BLE001
                if not args.capture_route_exceptions:
                    raise
                rows_by_route[route] = _route_exception_row(
                    route,
                    exc,
                    stage="compile_and_first",
                    compile_and_first=time.perf_counter() - start,
                    timings=timings,
                    args=args,
                )
    memory_after = redo_base._gpu_memory_info()
    wall = time.perf_counter() - start_all
    ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    paired = _paired(rows_by_route, args)
    aggregate_hard_vetoes = [
        f"{route}:{veto}"
        for route, row in rows_by_route.items()
        for veto in row["hard_vetoes"]
    ]
    aggregate_hard_vetoes.extend(f"paired:{veto}" for veto in _paired_vetoes(paired, args))
    if args.expect_device_kind == "gpu":
        if not logical_gpus:
            aggregate_hard_vetoes.append("gpu_device_evidence_missing")
        if (
            precision.get("tf32_execution_enabled") is not True
            and args.dtype == "float32"
            and args.tf32_mode == "enabled"
        ):
            aggregate_hard_vetoes.append("tf32_not_recorded_enabled_for_float32")
    return {
        "schema_version": "svd_nystrom_range_bearing_gate.v1",
        "status": "PASS" if not aggregate_hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "hard_vetoes": aggregate_hard_vetoes,
        "evidence_contract": {
            "question": "P04 range-bearing nonlinear Gaussian DPF SVD-Nystrom viability gate.",
            "baseline": "same-artifact compiled streaming TF32 DPF route",
            "candidate": "locked SVD-Nystrom DPF route",
            "primary_pass": _evidence_primary_pass(args),
            "promotion_role": "P04 nonlinear Gaussian gate only; no default readiness claim",
        },
        "shape": {
            "batch_size": 1,
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 4,
            "obs_dim": 2,
        },
        "seed": int(args.seed),
        "fixture": fixture_semantics,
        "history_mode": args.history_mode,
        "jit_compile": args.jit_compile,
        "route_request": args.route,
        "routes_executed": _route_names(args.route),
        "rows": [rows_by_route[route] for route in _route_names(args.route)],
        "paired_comparability": paired,
        "precision": precision,
        "transport": {
            "transport_policy": args.transport_policy,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "nystrom_rank": args.nystrom_rank,
            "nystrom_epsilon": args.nystrom_epsilon,
            "nystrom_max_iterations": args.nystrom_max_iterations,
            "nystrom_convergence_threshold": args.nystrom_convergence_threshold,
            "nystrom_diagnostics_enabled": args.nystrom_diagnostics,
            "nystrom_core_solver": args.nystrom_core_solver,
            "nystrom_core_rcond": args.nystrom_core_rcond,
            "nystrom_kernel_mode": args.nystrom_kernel_mode,
            "nystrom_scaling_normalization": args.nystrom_scaling_normalization,
        },
        "thresholds": {
            "nystrom_residual": NYSTROM_RESIDUAL_THRESHOLD,
            "final_logsumexp_residual": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction_min": ESS_FRACTION_MIN_THRESHOLD,
            "normalized_abs_log_likelihood_delta": args.paired_delta_threshold,
            "normalized_abs_log_likelihood_delta_role": _paired_delta_threshold_role(args),
            "paired_threshold_mode": args.paired_threshold_mode,
        },
        "run_manifest": {
            "git_commit": redo_base._run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": redo_base._run_text(["git", "status", "--short"]),
            "command": " ".join(sys.argv),
            "working_directory": str(ROOT),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow_version": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_cuda_visible_devices": args.cuda_visible_devices,
            "device": args.device,
            "device_scope": args.device_scope,
            "expect_device_kind": args.expect_device_kind,
            "selected_physical_gpu_argument": args.selected_physical_gpu,
            "gpu_selection_note": args.gpu_selection_note,
            "physical_gpus": physical_gpus,
            "logical_gpus": logical_gpus,
            "nvidia_smi_rows": smi_rows,
            "selected_physical_gpu": selected_physical_gpu,
            "started_at": started,
            "ended_at": ended,
            "wall_time_seconds": wall,
            "capture_route_exceptions": args.capture_route_exceptions,
            "gpu_memory_info_before": memory_before,
            "gpu_memory_info_after": memory_after,
            "output": args.output,
            "markdown_output": args.markdown_output,
        },
        "inference_status": {
            "hard_veto_screen": "PASS" if not aggregate_hard_vetoes else "FAIL",
            "statistically_supported_ranking": "NO",
            "descriptive_only_differences": (
                "timing, residual magnitudes, and per-row deltas are descriptive; "
                "paired deltas are record-only in record-only mode"
                if args.paired_threshold_mode == "record-only"
                else "timing, residual magnitudes, and per-row deltas are descriptive"
            ),
            "default_readiness": "NO",
            "next_evidence_needed": (
                "P04C aggregate scale extraction and reviewed P04D threshold freeze; do not launch P05"
                if args.paired_threshold_mode == "record-only"
                else "P05-P08 remaining gated phases if P04 passes"
            ),
        },
        "nonclaims": list(NONCLAIMS),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy().tolist())
    return value


def write_markdown(result: dict[str, Any], path: Path, json_path: Path) -> None:
    lines = [
        "# SVD-Nystrom Range-Bearing Gate",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Shape: `{result['shape']}`",
        f"- Fixture: `{result['fixture']['fixture_id']}`",
        f"- Route request: `{result['route_request']}`",
        f"- JIT compile: `{result['jit_compile']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        "",
        "## Routes",
        "",
        "| Route | Status | Compile+first seconds | Warm median seconds | ESS fraction min | Hard vetoes |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {route} | `{status}` | `{first}` | `{median}` | `{ess}` | `{vetoes}` |".format(
                route=row["route"],
                status=row["status"],
                first=row["compile_and_first_call_seconds"],
                median=(row.get("warm_call_timing_summary_seconds") or {}).get("median"),
                ess=row.get("ess_fraction_min"),
                vetoes=row["hard_vetoes"],
            )
        )
    exception_rows = [row for row in result["rows"] if "exception" in row]
    if exception_rows:
        lines.extend(["", "## Route Exceptions", ""])
        for row in exception_rows:
            exc = row["exception"]
            lines.append(
                "- {route}: `{module}.{type}` at `{stage}`: `{message}`".format(
                    route=row["route"],
                    module=exc["module"],
                    type=exc["type"],
                    stage=exc["stage"],
                    message=exc["message"],
                )
            )
    lines.extend(["", "## Paired Comparability", ""])
    paired = result.get("paired_comparability")
    if paired is None:
        lines.append("- Not applicable.")
    else:
        for key in (
            "log_likelihood_max_abs_delta",
            "normalized_max_abs_delta",
            "warm_median_streaming_over_nystrom_descriptive",
        ):
            lines.append(f"- {key}: `{paired.get(key)}`")
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = build_result(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        write_markdown(_json_ready(result), Path(args.markdown_output), output_path)
    if not args.quiet:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
