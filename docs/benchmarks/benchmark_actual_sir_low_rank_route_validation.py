"""Actual-SIR d18 route validation for streaming versus low-rank resampling.

This owned harness reuses the existing P8j actual-SIR tensor/callback setup and
adds route-level diagnostics for the low-rank coupling solver route.  It is a
validation artifact generator only; it does not change package-level
BayesFilter defaults or claim posterior correctness, HMC readiness, public API
readiness, dense Sinkhorn equivalence, or broad production readiness.
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

import tensorflow as tf  # noqa: E402

from docs.benchmarks import benchmark_p8j_tf32_batched_actual_sir as actual_sir  # noqa: E402
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import (  # noqa: E402
    low_rank_coupling_solver_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.low_rank_coupling_solver_tf import (  # noqa: E402
    low_rank_coupling_solver_resample_tensors_tf,
)


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-low-rank-default-certification-"
    "master-program-2026-06-24.md"
)
FACTOR_RESIDUAL_THRESHOLD = 5.0e-3
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-5
ESS_FRACTION_MIN_THRESHOLD = 0.01
NONCLAIMS = (
    "actual-SIR d18 route-validation harness only",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no package-level default readiness claim",
    "no broad production readiness claim",
    "no dense Sinkhorn equivalence claim",
    "no broad scalable-OT selection claim",
    "no statistical ranking claim",
)


class LowRankValueTensors(NamedTuple):
    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    final_particles: tf.Tensor
    final_log_weights: tf.Tensor
    final_particle_mean: tf.Tensor
    final_logsumexp_residual: tf.Tensor
    route_invocations: tf.Tensor
    active_resampling_mask_count: tf.Tensor
    active_resampling_batch_entries: tf.Tensor
    max_factor_marginal_residual: tf.Tensor
    max_induced_row_residual: tf.Tensor
    max_induced_column_residual: tf.Tensor
    projection_iterations_used_max: tf.Tensor
    finite_factors: tf.Tensor
    finite_particles: tf.Tensor
    nonnegative_factors: tf.Tensor
    positive_g: tf.Tensor


class StreamingValueTensors(NamedTuple):
    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    final_particles: tf.Tensor
    final_log_weights: tf.Tensor
    final_particle_mean: tf.Tensor
    final_logsumexp_residual: tf.Tensor
    route_invocations: tf.Tensor
    active_resampling_mask_count: tf.Tensor
    active_resampling_batch_entries: tf.Tensor
    max_row_residual: tf.Tensor
    max_column_residual: tf.Tensor


def _parse_int_csv(value: str) -> list[int]:
    entries = [item.strip() for item in str(value).split(",") if item.strip()]
    if not entries:
        raise ValueError("expected at least one integer")
    return [int(item) for item in entries]


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--route", choices=("streaming", "low_rank", "both"), default="low_rank")
    parser.add_argument("--batch-seeds", default="81120,81121,81122,81123,81124")
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--low-rank-rank", type=int, default=16)
    parser.add_argument("--low-rank-assignment-epsilon", type=float, default=0.25)
    parser.add_argument("--low-rank-alpha", type=float, default=1.0e-8)
    parser.add_argument("--low-rank-max-projection-iterations", type=int, default=120)
    parser.add_argument("--low-rank-convergence-threshold", type=float, default=1.0e-6)
    parser.add_argument("--low-rank-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument(
        "--streaming-timing-source",
        choices=("compiled_core",),
        default="compiled_core",
        help="Use the compiled streaming core for route timing.",
    )
    parser.add_argument(
        "--low-rank-timing-source",
        choices=("compiled_core",),
        default="compiled_core",
        help="Use the compiled tensor-only low-rank core for route timing.",
    )
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument(
        "--tf32-mode",
        choices=("default", "enabled", "disabled"),
        default="enabled",
    )
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--phase-id", default="ACTUAL-SIR-LR-ROUTE-VALIDATION")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args(argv)
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.low_rank_rank <= 0:
        raise ValueError("low_rank_rank must be positive")
    if args.low_rank_rank > args.num_particles:
        raise ValueError("low_rank_rank must be <= num_particles")
    if args.low_rank_assignment_epsilon <= 0.0:
        raise ValueError("low_rank_assignment_epsilon must be positive")
    if args.low_rank_alpha <= 0.0:
        raise ValueError("low_rank_alpha must be positive")
    if args.low_rank_alpha * args.low_rank_rank >= 1.0:
        raise ValueError("low_rank_alpha must be smaller than 1/low_rank_rank")
    if args.low_rank_max_projection_iterations <= 0:
        raise ValueError("low_rank_max_projection_iterations must be positive")
    if args.low_rank_convergence_threshold <= 0.0:
        raise ValueError("low_rank_convergence_threshold must be positive")
    if args.low_rank_denominator_floor <= 0.0:
        raise ValueError("low_rank_denominator_floor must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")
    if args.streaming_timing_source != "compiled_core":
        raise ValueError("streaming_timing_source must be compiled_core")
    if args.low_rank_timing_source != "compiled_core":
        raise ValueError("low_rank_timing_source must be compiled_core")
    if args.jit_compile is not True:
        raise ValueError("jit_compile must be true for the actual-SIR low-rank route")


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    precision = actual_sir._configure_precision(args)
    dtype = tf.float64 if args.dtype == "float64" else tf.float32
    low_rank_coupling_solver_tf.DTYPE = dtype
    low_rank_coupling_solver_tf.DEFAULT_DTYPE = dtype
    return precision


def _configure_gpus() -> tuple[list[str], list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return ([str(device) for device in physical_gpus], [str(device) for device in logical_gpus])


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


def _run_text(args: list[str], *, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unavailable"


def _nvidia_smi_rows() -> list[dict[str, str]]:
    text = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=5.0,
    )
    if text == "unavailable" or not text:
        return []
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "uuid": parts[2],
                    "memory_used_mib": parts[3],
                    "utilization_gpu_percent": parts[4],
                }
            )
    return rows


def _selected_physical_gpu(cuda_visible_devices: str | None, smi_rows: list[dict[str, str]]) -> dict[str, Any]:
    if cuda_visible_devices == "-1":
        return {"status": "cpu_hidden", "index": None, "name": None, "uuid": None}
    if not smi_rows:
        return {"status": "unavailable", "index": None, "name": None, "uuid": None}
    if cuda_visible_devices:
        first = cuda_visible_devices.split(",")[0].strip()
        for row in smi_rows:
            if row["index"] == first or row["uuid"] == first:
                return {"status": "selected", **row}
    row = smi_rows[0]
    return {"status": "selected_default_visible_zero", **row}


def _summary(timings: list[float]) -> dict[str, float]:
    if not timings:
        return {}
    return {
        "min": min(timings),
        "median": statistics.median(timings),
        "mean": statistics.fmean(timings),
        "max": max(timings),
    }


def _float(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _bool(value: Any) -> bool:
    return bool(tf.reduce_all(tf.cast(value, tf.bool)).numpy())


def _as_float_array(value: tf.Tensor) -> Any:
    return tf.cast(value, tf.float64).numpy().tolist()


def _finite_tensor(value: tf.Tensor) -> bool:
    return _bool(tf.math.is_finite(value))


def _normalize_log_weights(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return core_tf._normalize_log_weights(log_weights)


def _log_weight_floor() -> tf.Tensor:
    return core_tf._log_weight_floor()


def _weighted_mean_and_variance(particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return core_tf._weighted_mean_and_variance(particles, weights)


def _route_names(route: str) -> list[str]:
    if route == "both":
        return ["streaming", "low_rank"]
    return [route]


def _materialize_tensors(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _active_step_counts_tensor(fixed_mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    active_by_step = tf.reduce_any(fixed_mask, axis=0)
    return (
        tf.reduce_sum(tf.cast(active_by_step, tf.int32)),
        tf.reduce_sum(tf.cast(fixed_mask, tf.int32)),
    )


def _streaming_value_core(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> StreamingValueTensors:
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
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=actual_sir.DTYPE)
    max_row_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_column_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    route_invocations = tf.constant(0, dtype=tf.int32)
    active_steps, active_batch_entries = _active_step_counts_tensor(fixed_mask)
    means_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size, state_dim]),
    )
    variances_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size, state_dim]),
    )
    ess_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size]),
    )

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
        target_transition = callbacks["transition_log_density_fn"](post_flow, ancestors, time_tensor)
        target_observation = callbacks["observation_log_density_fn"](post_flow, observation, time_tensor)
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means_ta = means_ta.write(time_index, mean)
        variances_ta = variances_ta.write(time_index, variance)
        ess_ta = ess_ta.write(time_index, ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))
        mask = fixed_mask[:, time_index]

        def do_transport():
            transported = core_tf.batched_annealed_transport_core_tf(
                post_flow,
                normalized_log_weights,
                mask,
                epsilon=args.sinkhorn_epsilon,
                scaling=args.annealed_scaling,
                convergence_threshold=args.annealed_convergence_threshold,
                max_iterations=args.sinkhorn_iterations,
                transport_gradient_mode="raw",
                transport_plan_mode="streaming",
                transport_ad_mode="stabilized",
                row_chunk_size=args.row_chunk_size,
                col_chunk_size=args.col_chunk_size,
            )
            return (
                transported.particles,
                transported.log_weights,
                tf.cast(transported.max_row_residual, actual_sir.DTYPE),
                tf.cast(transported.max_column_residual, actual_sir.DTYPE),
                tf.constant(1, dtype=tf.int32),
            )

        def skip_transport():
            return (
                post_flow,
                normalized_log_weights,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0, dtype=tf.int32),
            )

        particles, log_weights, row_residual, column_residual, step_invocations = tf.cond(
            tf.reduce_any(mask),
            do_transport,
            skip_transport,
        )
        max_row_residual = tf.maximum(max_row_residual, row_residual)
        max_column_residual = tf.maximum(max_column_residual, column_residual)
        route_invocations += step_invocations

    filtered_means = means_ta.stack()
    filtered_variances = variances_ta.stack()
    ess_by_time = ess_ta.stack()
    filtered_means.set_shape([time_steps, batch_size, state_dim])
    filtered_variances.set_shape([time_steps, batch_size, state_dim])
    ess_by_time.set_shape([time_steps, batch_size])
    final_weights = tf.exp(log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True))
    final_particle_mean, _ = _weighted_mean_and_variance(particles, final_weights)
    final_logsumexp_residual = tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))
    return StreamingValueTensors(
        log_likelihood=log_likelihood,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        ess_by_time=ess_by_time,
        final_particles=particles,
        final_log_weights=log_weights,
        final_particle_mean=final_particle_mean,
        final_logsumexp_residual=final_logsumexp_residual,
        route_invocations=route_invocations,
        active_resampling_mask_count=active_steps,
        active_resampling_batch_entries=active_batch_entries,
        max_row_residual=max_row_residual,
        max_column_residual=max_column_residual,
    )


def _low_rank_value_core(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> LowRankValueTensors:
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
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=actual_sir.DTYPE)
    max_factor_marginal_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_induced_row_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_induced_column_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    projection_iterations_used_max = tf.constant(0, dtype=tf.int32)
    finite_factors = tf.constant(True)
    finite_particles = tf.constant(True)
    nonnegative_factors = tf.constant(True)
    positive_g = tf.constant(True)
    route_invocations = tf.constant(0, dtype=tf.int32)
    active_steps, active_batch_entries = _active_step_counts_tensor(fixed_mask)
    means_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size, state_dim]),
    )
    variances_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size, state_dim]),
    )
    ess_ta = tf.TensorArray(
        dtype=actual_sir.DTYPE,
        size=time_steps,
        element_shape=tf.TensorShape([batch_size]),
    )

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
        target_transition = callbacks["transition_log_density_fn"](post_flow, ancestors, time_tensor)
        target_observation = callbacks["observation_log_density_fn"](post_flow, observation, time_tensor)
        corrected_log_weights = (
            log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means_ta = means_ta.write(time_index, mean)
        variances_ta = variances_ta.write(time_index, variance)
        ess_ta = ess_ta.write(time_index, ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))
        mask = fixed_mask[:, time_index]

        def do_transport():
            resampled = low_rank_coupling_solver_resample_tensors_tf(
                post_flow,
                normalized_log_weights,
                rank=args.low_rank_rank,
                assignment_epsilon=args.low_rank_assignment_epsilon,
                alpha=args.low_rank_alpha,
                max_projection_iterations=args.low_rank_max_projection_iterations,
                convergence_threshold=args.low_rank_convergence_threshold,
                denominator_floor=args.low_rank_denominator_floor,
            )
            return (
                tf.cast(resampled.particles, actual_sir.DTYPE),
                tf.cast(resampled.log_weights, actual_sir.DTYPE),
                tf.cast(resampled.max_factor_marginal_residual, actual_sir.DTYPE),
                tf.cast(resampled.max_induced_row_residual, actual_sir.DTYPE),
                tf.cast(resampled.max_induced_column_residual, actual_sir.DTYPE),
                resampled.projection_iterations_used,
                resampled.finite_factors,
                resampled.finite_particles,
                resampled.nonnegative_factors,
                resampled.positive_g,
                tf.constant(1, dtype=tf.int32),
            )

        def skip_transport():
            return (
                post_flow,
                normalized_log_weights,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0, dtype=tf.int32),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(0, dtype=tf.int32),
            )

        (
            particles,
            log_weights,
            factor_residual,
            induced_row_residual,
            induced_column_residual,
            projection_iterations_used,
            step_finite_factors,
            step_finite_particles,
            step_nonnegative_factors,
            step_positive_g,
            step_route_invocation,
        ) = tf.cond(tf.reduce_any(mask), do_transport, skip_transport)
        max_factor_marginal_residual = tf.maximum(max_factor_marginal_residual, factor_residual)
        max_induced_row_residual = tf.maximum(max_induced_row_residual, induced_row_residual)
        max_induced_column_residual = tf.maximum(max_induced_column_residual, induced_column_residual)
        projection_iterations_used_max = tf.maximum(projection_iterations_used_max, projection_iterations_used)
        finite_factors = tf.logical_and(finite_factors, step_finite_factors)
        finite_particles = tf.logical_and(finite_particles, step_finite_particles)
        nonnegative_factors = tf.logical_and(nonnegative_factors, step_nonnegative_factors)
        positive_g = tf.logical_and(positive_g, step_positive_g)
        route_invocations += step_route_invocation

    filtered_means = means_ta.stack()
    filtered_variances = variances_ta.stack()
    ess_by_time = ess_ta.stack()
    filtered_means.set_shape([time_steps, batch_size, state_dim])
    filtered_variances.set_shape([time_steps, batch_size, state_dim])
    ess_by_time.set_shape([time_steps, batch_size])
    final_weights = tf.exp(log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True))
    final_particle_mean, _ = _weighted_mean_and_variance(particles, final_weights)
    final_logsumexp_residual = tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))
    return LowRankValueTensors(
        log_likelihood=log_likelihood,
        filtered_means=filtered_means,
        filtered_variances=filtered_variances,
        ess_by_time=ess_by_time,
        final_particles=particles,
        final_log_weights=log_weights,
        final_particle_mean=final_particle_mean,
        final_logsumexp_residual=final_logsumexp_residual,
        route_invocations=route_invocations,
        active_resampling_mask_count=active_steps,
        active_resampling_batch_entries=active_batch_entries,
        max_factor_marginal_residual=max_factor_marginal_residual,
        max_induced_row_residual=max_induced_row_residual,
        max_induced_column_residual=max_induced_column_residual,
        projection_iterations_used_max=projection_iterations_used_max,
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        nonnegative_factors=nonnegative_factors,
        positive_g=positive_g,
    )


def _run_streaming_compiled_core_timed(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, ...]:
        value = _streaming_value_core(tensors, callbacks, args)
        return (
            value.log_likelihood,
            value.filtered_means,
            value.filtered_variances,
            value.ess_by_time,
            value.final_particles,
            value.final_log_weights,
            value.final_particle_mean,
            value.final_logsumexp_residual,
            value.route_invocations,
            value.active_resampling_mask_count,
            value.active_resampling_batch_entries,
            value.max_row_residual,
            value.max_column_residual,
        )

    memory_before = _gpu_memory_info()
    start = time.perf_counter()
    outputs = compiled_outputs()
    _materialize_tensors(*outputs)
    first_call_seconds = time.perf_counter() - start

    for _ in range(args.warmups):
        _materialize_tensors(*compiled_outputs())

    timings: list[float] = []
    last_outputs = outputs
    for _ in range(args.repeats):
        start = time.perf_counter()
        last_outputs = compiled_outputs()
        _materialize_tensors(*last_outputs)
        timings.append(time.perf_counter() - start)
    memory_after = _gpu_memory_info()
    return {
        "first_call_seconds": first_call_seconds,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "compiled_core_outputs": {
            "log_likelihood": last_outputs[0],
            "filtered_means": last_outputs[1],
            "filtered_variances": last_outputs[2],
            "ess_by_time": last_outputs[3],
        },
        "loop": _streaming_loop_from_outputs(last_outputs, args),
        "streaming_timing_source": "compiled_core",
        "diagnostic_loop_seconds_explanatory": None,
        "memory_maxrss_kb_explanatory": None,
        "jit_compile": args.jit_compile,
    }


def _streaming_loop_from_outputs(outputs: tuple[tf.Tensor, ...], args: argparse.Namespace) -> dict[str, Any]:
    batch_size = len(args.batch_seeds)
    return {
        "route": "streaming",
        "log_likelihood": outputs[0],
        "filtered_means": outputs[1],
        "filtered_variances": outputs[2],
        "ess_by_time": outputs[3],
        "final_particles": outputs[4],
        "final_log_weights": outputs[5],
        "final_particle_mean": outputs[6],
        "final_logsumexp_residual": outputs[7],
        "route_invocations": int(outputs[8].numpy()),
        "active_resampling_mask_count": int(outputs[9].numpy()),
        "active_resampling_batch_entries": int(outputs[10].numpy()),
        "transport_matrix_shapes": [[batch_size, 0, 0]],
        "step_diagnostics": [],
        "streaming_row_residuals": [_float(outputs[11])],
        "streaming_column_residuals": [_float(outputs[12])],
        "batch_size": batch_size,
        "num_particles": args.num_particles,
        "state_dim": 18,
        "time_steps": args.time_steps,
    }


def _low_rank_loop_from_outputs(outputs: tuple[tf.Tensor, ...], args: argparse.Namespace) -> dict[str, Any]:
    batch_size = len(args.batch_seeds)
    return {
        "route": "low_rank",
        "log_likelihood": outputs[0],
        "filtered_means": outputs[1],
        "filtered_variances": outputs[2],
        "ess_by_time": outputs[3],
        "final_particles": outputs[4],
        "final_log_weights": outputs[5],
        "final_particle_mean": outputs[6],
        "final_logsumexp_residual": outputs[7],
        "route_invocations": int(outputs[8].numpy()),
        "active_resampling_mask_count": int(outputs[9].numpy()),
        "active_resampling_batch_entries": int(outputs[10].numpy()),
        "transport_matrix_shapes": [[batch_size, 0, 0]],
        "step_diagnostics": [
            {
                "component_id": "low_rank_coupling_solver_tf",
                "mathematical_object": "low_rank_coupling_solver_route",
                "source_status": "source_locked",
                "semantic_class": "semantic_replacement",
                "implementation_scope": "solver_route_dykstra_projection_compiled_tensor_core",
                "source_route": "extension_or_invention",
                "source_route_components": {
                    "factored_coupling_parameterization": "source_faithful",
                    "low_rank_lazy_apply": "source_faithful",
                    "factor_marginal_diagnostics": "source_faithful",
                    "dykstra_style_projection": "source_faithful",
                    "deterministic_initialization": "fixed_hmc_adaptation",
                    "fixed_iteration_schedule": "fixed_hmc_adaptation",
                    "phase1_scaled_transport_adapter": "fixed_hmc_adaptation",
                    "cost_nudged_assignment_kernel": "extension_or_invention",
                },
                "transport_object_kind": "low_rank_coupling_factors",
                "transport_matrix_materialized": False,
                "projection_iterations_used": int(outputs[14].numpy()),
                "max_factor_marginal_residual": _float(outputs[11]),
                "max_induced_row_residual": _float(outputs[12]),
                "max_induced_column_residual": _float(outputs[13]),
                "finite_factors": bool(outputs[15].numpy()),
                "finite_particles": bool(outputs[16].numpy()),
                "nonnegative_factors": bool(outputs[17].numpy()),
                "positive_g": bool(outputs[18].numpy()),
            }
        ],
        "streaming_row_residuals": [],
        "streaming_column_residuals": [],
        "batch_size": batch_size,
        "num_particles": args.num_particles,
        "state_dim": 18,
        "time_steps": args.time_steps,
    }


def _run_low_rank_compiled_core_timed(
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, ...]:
        value = _low_rank_value_core(tensors, callbacks, args)
        return (
            value.log_likelihood,
            value.filtered_means,
            value.filtered_variances,
            value.ess_by_time,
            value.final_particles,
            value.final_log_weights,
            value.final_particle_mean,
            value.final_logsumexp_residual,
            value.route_invocations,
            value.active_resampling_mask_count,
            value.active_resampling_batch_entries,
            value.max_factor_marginal_residual,
            value.max_induced_row_residual,
            value.max_induced_column_residual,
            value.projection_iterations_used_max,
            value.finite_factors,
            value.finite_particles,
            value.nonnegative_factors,
            value.positive_g,
        )

    memory_before = _gpu_memory_info()
    start = time.perf_counter()
    outputs = compiled_outputs()
    _materialize_tensors(*outputs)
    first_call_seconds = time.perf_counter() - start

    for _ in range(args.warmups):
        _materialize_tensors(*compiled_outputs())

    timings: list[float] = []
    last_outputs = outputs
    for _ in range(args.repeats):
        start = time.perf_counter()
        last_outputs = compiled_outputs()
        _materialize_tensors(*last_outputs)
        timings.append(time.perf_counter() - start)
    memory_after = _gpu_memory_info()
    return {
        "first_call_seconds": first_call_seconds,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "loop": _low_rank_loop_from_outputs(last_outputs, args),
        "low_rank_timing_source": "compiled_core",
        "jit_compile": args.jit_compile,
        "memory_maxrss_kb_explanatory": None,
    }


def _max_diag(step_diagnostics: list[dict[str, Any]], key: str) -> float | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return max(float(value) for value in values) if values else None


def _all_diag(step_diagnostics: list[dict[str, Any]], key: str) -> bool | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return all(bool(value) for value in values) if values else None


def _route_hard_vetoes(loop: dict[str, Any], args: argparse.Namespace, output_devices: list[str]) -> list[str]:
    hard_vetoes: list[str] = []
    route = loop["route"]
    if not _finite_tensor(loop["log_likelihood"]):
        hard_vetoes.append("nonfinite_log_likelihood")
    if not _finite_tensor(loop["filtered_means"]):
        hard_vetoes.append("nonfinite_filtered_means")
    if not _finite_tensor(loop["filtered_variances"]):
        hard_vetoes.append("nonfinite_filtered_variances")
    if not _finite_tensor(loop["ess_by_time"]):
        hard_vetoes.append("nonfinite_ess_by_time")
    if not _finite_tensor(loop["final_particles"]):
        hard_vetoes.append("nonfinite_final_particles")
    if not _finite_tensor(loop["final_log_weights"]):
        hard_vetoes.append("nonfinite_final_log_weights")
    if loop["route_invocations"] != loop["active_resampling_mask_count"]:
        hard_vetoes.append("route_invocation_count_mismatch")
    if loop["active_resampling_mask_count"] > 0 and loop["route_invocations"] <= 0:
        hard_vetoes.append("route_invocations_zero")
    if any(shape[-2:] != [0, 0] for shape in loop["transport_matrix_shapes"]):
        hard_vetoes.append("transport_matrix_materialized")
    if _float(loop["final_logsumexp_residual"]) > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("final_logsumexp_residual_threshold")
    ess_fraction_min = _float(tf.reduce_min(loop["ess_by_time"])) / float(args.num_particles)
    if ess_fraction_min < ESS_FRACTION_MIN_THRESHOLD:
        hard_vetoes.append("ess_fraction_min_threshold")
    if args.expect_device_kind == "gpu" and not all("GPU" in device.upper() for device in output_devices):
        hard_vetoes.append("expected_gpu_outputs_missing")
    if args.expect_device_kind == "cpu" and not all("CPU" in device.upper() for device in output_devices):
        hard_vetoes.append("expected_cpu_outputs_missing")

    if route == "low_rank":
        step_diagnostics = loop["step_diagnostics"]
        max_factor = _max_diag(step_diagnostics, "max_factor_marginal_residual")
        if _all_diag(step_diagnostics, "finite_factors") is False:
            hard_vetoes.append("nonfinite_factors")
        if _all_diag(step_diagnostics, "finite_particles") is False:
            hard_vetoes.append("low_rank_nonfinite_particles")
        if _all_diag(step_diagnostics, "nonnegative_factors") is False:
            hard_vetoes.append("negative_factor")
        if _all_diag(step_diagnostics, "positive_g") is False:
            hard_vetoes.append("nonpositive_g")
        if loop["active_resampling_mask_count"] > 0 and max_factor is None:
            hard_vetoes.append("missing_factor_diagnostics")
        if max_factor is not None and max_factor > FACTOR_RESIDUAL_THRESHOLD:
            hard_vetoes.append("factor_marginal_residual_threshold")
    return hard_vetoes


def _run_timed_route(
    route: str,
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if route == "streaming" and args.streaming_timing_source == "compiled_core":
        return _run_streaming_compiled_core_timed(tensors, callbacks, args)
    if route == "low_rank" and args.low_rank_timing_source == "compiled_core":
        return _run_low_rank_compiled_core_timed(tensors, callbacks, args)
    raise ValueError(f"route {route!r} must use compiled_core timing")


def _route_row(route: str, timed: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    loop = timed["loop"]
    output_devices = [
        loop["log_likelihood"].device,
        loop["filtered_means"].device,
        loop["filtered_variances"].device,
        loop["ess_by_time"].device,
        loop["final_particle_mean"].device,
    ]
    hard_vetoes = _route_hard_vetoes(loop, args, output_devices)
    step_diagnostics = loop["step_diagnostics"]
    row = {
        "route": route,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "batch_size": loop["batch_size"],
        "time_steps": loop["time_steps"],
        "num_particles": loop["num_particles"],
        "state_dim": loop["state_dim"],
        "obs_dim": 9,
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "route_invocations": loop["route_invocations"],
        "active_resampling_mask_count": loop["active_resampling_mask_count"],
        "active_resampling_batch_entries": loop["active_resampling_batch_entries"],
        "transport_matrix_shapes": loop["transport_matrix_shapes"],
        "transport_matrix_materialized": any(shape[-2:] != [0, 0] for shape in loop["transport_matrix_shapes"]),
        "finite_log_likelihood": _finite_tensor(loop["log_likelihood"]),
        "finite_filtered_means": _finite_tensor(loop["filtered_means"]),
        "finite_filtered_variances": _finite_tensor(loop["filtered_variances"]),
        "finite_ess_by_time": _finite_tensor(loop["ess_by_time"]),
        "finite_final_particles": _finite_tensor(loop["final_particles"]),
        "finite_final_log_weights": _finite_tensor(loop["final_log_weights"]),
        "final_logsumexp_residual": _float(loop["final_logsumexp_residual"]),
        "ess_fraction_min": _float(tf.reduce_min(loop["ess_by_time"])) / float(args.num_particles),
        "ess_min": _float(tf.reduce_min(loop["ess_by_time"])),
        "ess_max": _float(tf.reduce_max(loop["ess_by_time"])),
        "output_devices": output_devices,
        "first_call_seconds": timed["first_call_seconds"],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "warm_call_timings_seconds": timed["warm_call_timings_seconds"],
        "warm_call_timing_summary_seconds": timed["warm_call_timing_summary_seconds"],
        "gpu_memory_info_before": timed["gpu_memory_info_before"],
        "gpu_memory_info_after": timed["gpu_memory_info_after"],
        "memory_maxrss_kb_explanatory": timed["memory_maxrss_kb_explanatory"],
        "jit_compile": timed.get("jit_compile", False),
        "log_likelihood": _as_float_array(loop["log_likelihood"]),
        "filtered_means": _as_float_array(loop["filtered_means"]),
        "filtered_variances": _as_float_array(loop["filtered_variances"]),
        "ess_by_time": _as_float_array(loop["ess_by_time"]),
        "final_particle_mean": _as_float_array(loop["final_particle_mean"]),
    }
    if route == "streaming":
        compiled_outputs = timed.get("compiled_core_outputs")
        compiled_core_delta = None
        if isinstance(compiled_outputs, dict) and "diagnostic_loop_seconds_explanatory" in timed and timed.get("diagnostic_loop_seconds_explanatory") is not None:
            compiled_core_delta = {
                "log_likelihood_max_abs_delta_vs_diagnostic_loop": _float(
                    tf.reduce_max(tf.abs(compiled_outputs["log_likelihood"] - loop["log_likelihood"]))
                ),
                "filtered_mean_max_abs_delta_vs_diagnostic_loop": _float(
                    tf.reduce_max(tf.abs(compiled_outputs["filtered_means"] - loop["filtered_means"]))
                ),
                "filtered_variance_max_abs_delta_vs_diagnostic_loop": _float(
                    tf.reduce_max(tf.abs(compiled_outputs["filtered_variances"] - loop["filtered_variances"]))
                ),
                "ess_max_abs_delta_vs_diagnostic_loop": _float(
                    tf.reduce_max(tf.abs(compiled_outputs["ess_by_time"] - loop["ess_by_time"]))
                ),
            }
        row.update(
            {
                "transport_object_kind": "streaming_transport",
                "streaming_timing_source": timed.get("streaming_timing_source", "diagnostic_loop"),
                "diagnostic_loop_seconds_explanatory": timed.get("diagnostic_loop_seconds_explanatory"),
                "compiled_core_delta_vs_diagnostic_loop": compiled_core_delta,
                "streaming_max_row_residual": max(loop["streaming_row_residuals"]) if loop["streaming_row_residuals"] else None,
                "streaming_max_column_residual": max(loop["streaming_column_residuals"]) if loop["streaming_column_residuals"] else None,
                "sinkhorn_iterations": args.sinkhorn_iterations,
                "sinkhorn_epsilon": args.sinkhorn_epsilon,
                "annealed_scaling": args.annealed_scaling,
                "annealed_convergence_threshold": args.annealed_convergence_threshold,
            }
        )
    else:
        row.update(
            {
                "transport_object_kind": "low_rank_coupling_factors",
                "low_rank_timing_source": timed.get("low_rank_timing_source", "diagnostic_loop"),
                "low_rank_rank": args.low_rank_rank,
                "low_rank_assignment_epsilon": args.low_rank_assignment_epsilon,
                "low_rank_alpha": args.low_rank_alpha,
                "low_rank_max_projection_iterations": args.low_rank_max_projection_iterations,
                "low_rank_convergence_threshold": args.low_rank_convergence_threshold,
                "low_rank_denominator_floor": args.low_rank_denominator_floor,
                "all_finite_factors": _all_diag(step_diagnostics, "finite_factors"),
                "all_nonnegative_factors": _all_diag(step_diagnostics, "nonnegative_factors"),
                "all_positive_g": _all_diag(step_diagnostics, "positive_g"),
                "max_factor_marginal_residual": _max_diag(step_diagnostics, "max_factor_marginal_residual"),
                "max_induced_row_residual": _max_diag(step_diagnostics, "max_induced_row_residual"),
                "max_induced_column_residual": _max_diag(step_diagnostics, "max_induced_column_residual"),
                "projection_iterations_used_max_explanatory": _max_diag(step_diagnostics, "projection_iterations_used"),
                "transport_object_kind_diagnostics": [
                    diag.get("transport_object_kind") for diag in step_diagnostics
                ],
                "source_route_components": step_diagnostics[-1].get("source_route_components", {}) if step_diagnostics else {},
            }
        )
    return row


def _relative_l2(delta: tf.Tensor, reference: tf.Tensor) -> float:
    numerator = _float(tf.linalg.norm(tf.reshape(tf.cast(delta, tf.float64), [-1])))
    denominator = _float(tf.linalg.norm(tf.reshape(tf.cast(reference, tf.float64), [-1])))
    if denominator <= 0.0:
        return math.inf if numerator > 0.0 else 0.0
    return numerator / denominator


def _rms(delta: tf.Tensor) -> float:
    value = tf.reshape(tf.cast(delta, tf.float64), [-1])
    return _float(tf.sqrt(tf.reduce_mean(value * value)))


def _paired_comparability(rows: list[dict[str, Any]], timed: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if set(timed) != {"streaming", "low_rank"}:
        return None
    streaming = timed["streaming"]["loop"]
    low_rank = timed["low_rank"]["loop"]
    ll_delta = tf.cast(low_rank["log_likelihood"] - streaming["log_likelihood"], tf.float64)
    mean_delta = tf.cast(low_rank["filtered_means"] - streaming["filtered_means"], tf.float64)
    var_delta = tf.cast(low_rank["filtered_variances"] - streaming["filtered_variances"], tf.float64)
    final_mean_delta = tf.cast(low_rank["final_particle_mean"] - streaming["final_particle_mean"], tf.float64)
    streaming_warm = next((row for row in rows if row["route"] == "streaming"), {})
    low_rank_warm = next((row for row in rows if row["route"] == "low_rank"), {})
    streaming_median = (streaming_warm.get("warm_call_timing_summary_seconds") or {}).get("median")
    low_rank_median = (low_rank_warm.get("warm_call_timing_summary_seconds") or {}).get("median")
    warm_ratio = (
        float(streaming_median) / float(low_rank_median)
        if streaming_median is not None and low_rank_median not in (None, 0.0)
        else None
    )
    return {
        "role": "diagnostic_until_p03_aggregate_enforces_support_rows",
        "log_likelihood_delta_by_seed": _as_float_array(ll_delta),
        "log_likelihood_max_abs_delta": _float(tf.reduce_max(tf.abs(ll_delta))),
        "log_likelihood_mean_abs_delta": _float(tf.reduce_mean(tf.abs(ll_delta))),
        "filtered_mean_relative_l2": _relative_l2(mean_delta, streaming["filtered_means"]),
        "filtered_mean_rms": _rms(mean_delta),
        "filtered_variance_relative_l2": _relative_l2(var_delta, streaming["filtered_variances"]),
        "filtered_variance_rms": _rms(var_delta),
        "final_particle_mean_relative_l2": _relative_l2(final_mean_delta, streaming["final_particle_mean"]),
        "final_particle_mean_abs_l2": _float(tf.linalg.norm(tf.reshape(final_mean_delta, [-1]))),
        "warm_median_streaming_over_low_rank": warm_ratio,
        "thresholds": {
            "log_likelihood_max_abs_delta": 10.0,
            "log_likelihood_mean_abs_delta": 5.0,
            "filtered_mean_relative_l2": 0.20,
            "filtered_mean_rms": 2.5,
            "filtered_variance_relative_l2": 0.75,
            "filtered_variance_rms": 25.0,
            "final_particle_mean_relative_l2": 0.20,
            "final_particle_mean_abs_l2": 25.0,
            "warm_median_streaming_over_low_rank": 1.25,
        },
    }


def _run_manifest(
    args: argparse.Namespace,
    *,
    precision: dict[str, Any],
    physical_gpus: list[str],
    logical_gpus: list[str],
    selected_physical_gpu: dict[str, Any],
    smi_rows: list[dict[str, str]],
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
) -> dict[str, Any]:
    return {
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": _run_text(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requested_cuda_visible_devices": args.cuda_visible_devices,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "nvidia_smi_rows": smi_rows,
        "selected_physical_gpu": selected_physical_gpu,
        "gpu_fallback_status": "none_recorded_by_harness",
        "precision": precision,
        "phase_id": args.phase_id,
        "plan_path": PLAN_PATH,
        "jit_compile": args.jit_compile,
        "streaming_timing_source": args.streaming_timing_source,
        "low_rank_timing_source": args.low_rank_timing_source,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
        "output": args.output,
        "markdown_output": args.markdown_output,
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    smi_rows = _nvidia_smi_rows()
    selected_physical_gpu = _selected_physical_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"), smi_rows)
    callbacks = actual_sir._dpf_sir_callbacks()
    tensors, sir_semantics = actual_sir._build_actual_sir_tensors(args)
    adapter_callbacks = actual_sir._make_actual_sir_callbacks(callbacks, tensors, args.batch_seeds, args)

    started_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    start = time.perf_counter()
    timed: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    with tf.device(args.device):
        for route in _route_names(args.route):
            timed[route] = _run_timed_route(route, tensors, adapter_callbacks, args)
            rows.append(_route_row(route, timed[route], args))
    ended_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    wall_time = time.perf_counter() - start

    aggregate_hard_vetoes = [f"{row['route']}:{veto}" for row in rows for veto in row["hard_vetoes"]]
    actual_sir_semantics_pass = bool(
        sir_semantics.get("row_id") == "zhao_cui_spatial_sir_austria_j9_T20"
        and sir_semantics.get("state_dimension") == 18
        and sir_semantics.get("observation_dimension") == 9
        and sir_semantics.get("actual_sir_callbacks_used") is True
    )
    if not actual_sir_semantics_pass:
        aggregate_hard_vetoes.append("actual_sir_semantics_missing")
    manifest = _run_manifest(
        args,
        precision=precision,
        physical_gpus=physical_gpus,
        logical_gpus=logical_gpus,
        selected_physical_gpu=selected_physical_gpu,
        smi_rows=smi_rows,
        started_at=started_at,
        ended_at=ended_at,
        wall_time_seconds=wall_time,
    )
    paired = _paired_comparability(rows, timed)
    return {
        "schema_version": "actual_sir_low_rank_route_validation.v1",
        "status": "PASS" if not aggregate_hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "algorithm_under_test": "actual-SIR d18 LEDH/PFPF-OT streaming route versus low-rank coupling solver-route resampling",
        "route_request": args.route,
        "routes_executed": _route_names(args.route),
        "hard_vetoes": aggregate_hard_vetoes,
        "thresholds": {
            "factor_marginal_residual": FACTOR_RESIDUAL_THRESHOLD,
            "final_logsumexp_residual": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction_min": ESS_FRACTION_MIN_THRESHOLD,
            "low_rank_transport_matrix_shape_suffix": [0, 0],
            "runtime_memory_role": "explanatory_except_p03_promotion_screen",
        },
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "transport_policy": args.transport_policy,
        "sir_semantics": sir_semantics,
        "actual_sir_semantics_pass": actual_sir_semantics_pass,
        "paired_comparability": paired,
        "rows": rows,
        "run_manifest": manifest,
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
        "# Actual-SIR Low-Rank Route Validation",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Route request: `{result['route_request']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Shape: `{result['shape']}`",
        f"- Actual-SIR semantics pass: `{result['actual_sir_semantics_pass']}`",
        "",
        "## Routes",
        "",
        "| Route | Status | Invocations | Active steps | Warm median seconds | ESS fraction min | Final logsumexp residual | Hard vetoes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {route} | `{status}` | `{inv}` | `{active}` | `{median}` | `{ess}` | `{logsum}` | `{vetoes}` |".format(
                route=row["route"],
                status=row["status"],
                inv=row["route_invocations"],
                active=row["active_resampling_mask_count"],
                median=(row.get("warm_call_timing_summary_seconds") or {}).get("median"),
                ess=row["ess_fraction_min"],
                logsum=row["final_logsumexp_residual"],
                vetoes=row["hard_vetoes"],
            )
        )
    lines.extend(["", "## Paired Comparability", ""])
    paired = result.get("paired_comparability")
    if paired is None:
        lines.append("- Not applicable.")
    else:
        for key in (
            "log_likelihood_max_abs_delta",
            "log_likelihood_mean_abs_delta",
            "filtered_mean_relative_l2",
            "filtered_mean_rms",
            "filtered_variance_relative_l2",
            "filtered_variance_rms",
            "final_particle_mean_relative_l2",
            "final_particle_mean_abs_l2",
            "warm_median_streaming_over_low_rank",
        ):
            lines.append(f"- {key}: `{paired.get(key)}`")
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            f"- Git commit: `{result['run_manifest']['git_commit']}`",
            f"- Command: `{result['run_manifest']['command']}`",
            f"- Device scope: `{result['run_manifest']['device_scope']}`",
            f"- CUDA_VISIBLE_DEVICES: `{result['run_manifest']['cuda_visible_devices']}`",
            f"- Selected physical GPU: `{result['run_manifest']['selected_physical_gpu']}`",
            "",
            "## Nonclaims",
            "",
        ]
    )
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
        write_markdown(result, Path(args.markdown_output), output_path)
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
