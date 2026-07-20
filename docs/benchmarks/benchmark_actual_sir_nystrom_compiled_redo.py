"""Compiled actual-SIR d18 streaming-vs-Nystrom redo benchmark.

This harness replaces the contaminated Python-loop timing protocol from the
first actual-SIR Nystrom default-promotion lane.  It runs the production-style
compiled streaming value path and a compiled tensor-only Nystrom route on the
same actual-SIR tensors, device, dtype, TF32 mode, and timing protocol.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling import nystrom_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling.nystrom_transport_tf import (  # noqa: E402
    nystrom_transport_resample_tensors_tf,
)


NONCLAIMS = (
    "redo benchmark after prior runtime protocol quarantine",
    "no default readiness claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no statistical ranking claim without replicated uncertainty analysis",
)
NYSTROM_RESIDUAL_THRESHOLD = 5.0e-2
LOG_LIKELIHOOD_MAX_ABS_DELTA_THRESHOLD = 10.0
LOG_LIKELIHOOD_MEAN_ABS_DELTA_THRESHOLD = 5.0


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
    parser.add_argument("--route", choices=("streaming", "nystrom", "both"), default="both")
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=2048)
    parser.add_argument("--col-chunk-size", type=int, default=2048)
    parser.add_argument("--particle-chunk-size", type=int, default=1024)
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
        default="cholesky",
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
    parser.add_argument("--history-mode", choices=("full", "value-only"), default="value-only")
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
    parser.add_argument("--phase-id", default="ACTUAL-SIR-NYSTROM-COMPILED-REDO")
    parser.add_argument("--quiet", action="store_true")
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
    if args.nystrom_rank <= 0 or args.nystrom_rank > args.num_particles:
        raise ValueError("nystrom_rank must be positive and <= num_particles")
    if args.sinkhorn_iterations <= 0 or args.nystrom_max_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if args.nystrom_core_rcond <= 0.0:
        raise ValueError("nystrom_core_rcond must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    precision = actual_sir._configure_precision(args)
    dtype = tf.float64 if args.dtype == "float64" else tf.float32
    nystrom_transport_tf.DTYPE = dtype
    nystrom_transport_tf.DEFAULT_DTYPE = dtype
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


def _run_text(command: list[str], *, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()
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
    return {"status": "selected_default_visible_zero", **smi_rows[0]}


def _route_names(route: str) -> list[str]:
    if route == "both":
        return ["streaming", "nystrom"]
    return [route]


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
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=actual_sir.DTYPE)
    max_row_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_column_residual = tf.constant(0.0, dtype=actual_sir.DTYPE)
    finite_factors = tf.constant(True)
    finite_particles = tf.constant(True)
    iterations_used_max = tf.constant(0, dtype=tf.int32)
    route_invocations = tf.constant(0, dtype=tf.int32)
    min_kernel_denominator = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    denominator_floor_hits = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_abs_log_scaling_gauge_shift = tf.constant(0.0, dtype=actual_sir.DTYPE)
    scaling_normalization_applications = tf.constant(0.0, dtype=actual_sir.DTYPE)
    max_factor_diag_error = tf.constant(0.0, dtype=actual_sir.DTYPE)
    min_factor_diagonal = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    max_factor_diagonal = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)
    landmark_core_min_eigenvalue = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    landmark_core_max_eigenvalue = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)
    landmark_core_condition_proxy = tf.constant(0.0, dtype=actual_sir.DTYPE)
    landmark_core_effective_rank_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    left_factor_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    left_factor_max = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)
    core_matrix_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    core_matrix_max = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)
    raw_kernel_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    projected_kernel_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    projection_floor_hits = tf.constant(0.0, dtype=actual_sir.DTYPE)
    scaling_u_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    scaling_u_max = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)
    scaling_v_min = tf.constant(float("inf"), dtype=actual_sir.DTYPE)
    scaling_v_max = tf.constant(float("-inf"), dtype=actual_sir.DTYPE)

    if args.history_mode == "full":
        means_ta = tf.TensorArray(
            dtype=actual_sir.DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        vars_ta = tf.TensorArray(
            dtype=actual_sir.DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size, state_dim]),
        )
        ess_ta = tf.TensorArray(
            dtype=actual_sir.DTYPE,
            size=time_steps,
            element_shape=tf.TensorShape([batch_size]),
        )
    else:
        means_ta = tf.TensorArray(dtype=actual_sir.DTYPE, size=0)
        vars_ta = tf.TensorArray(dtype=actual_sir.DTYPE, size=0)
        ess_ta = tf.TensorArray(dtype=actual_sir.DTYPE, size=0)

    def step_body(
        time_tensor: tf.Tensor,
        current_particles: tf.Tensor,
        current_log_weights: tf.Tensor,
        current_log_likelihood: tf.Tensor,
        current_max_row_residual: tf.Tensor,
        current_max_column_residual: tf.Tensor,
        current_finite_factors: tf.Tensor,
        current_finite_particles: tf.Tensor,
        current_iterations_used_max: tf.Tensor,
        current_route_invocations: tf.Tensor,
        current_min_kernel_denominator: tf.Tensor,
        current_denominator_floor_hits: tf.Tensor,
        current_max_abs_log_scaling_gauge_shift: tf.Tensor,
        current_scaling_normalization_applications: tf.Tensor,
        current_max_factor_diag_error: tf.Tensor,
        current_min_factor_diagonal: tf.Tensor,
        current_max_factor_diagonal: tf.Tensor,
        current_landmark_core_min_eigenvalue: tf.Tensor,
        current_landmark_core_max_eigenvalue: tf.Tensor,
        current_landmark_core_condition_proxy: tf.Tensor,
        current_landmark_core_effective_rank_min: tf.Tensor,
        current_left_factor_min: tf.Tensor,
        current_left_factor_max: tf.Tensor,
        current_core_matrix_min: tf.Tensor,
        current_core_matrix_max: tf.Tensor,
        current_raw_kernel_min: tf.Tensor,
        current_projected_kernel_min: tf.Tensor,
        current_projection_floor_hits: tf.Tensor,
        current_scaling_u_min: tf.Tensor,
        current_scaling_u_max: tf.Tensor,
        current_scaling_v_min: tf.Tensor,
        current_scaling_v_max: tf.Tensor,
        current_means_ta: tf.TensorArray,
        current_vars_ta: tf.TensorArray,
        current_ess_ta: tf.TensorArray,
    ):
        observation = observations[time_tensor]
        ancestors = current_particles
        pre_flow = callbacks["pre_flow_step_fn"](current_particles, time_tensor)
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
            current_log_weights
            + target_transition
            + target_observation
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)
        next_log_likelihood = current_log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        if args.history_mode == "full":
            mean, variance = core_tf._weighted_mean_and_variance(post_flow, weights)
            current_means_ta = current_means_ta.write(time_tensor, mean)
            current_vars_ta = current_vars_ta.write(time_tensor, variance)
            current_ess_ta = current_ess_ta.write(time_tensor, ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, core_tf._log_weight_floor()))
        mask = fixed_mask[:, time_tensor]

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
                tf.cast(resampled.particles, actual_sir.DTYPE),
                tf.cast(resampled.log_weights, actual_sir.DTYPE),
                tf.cast(resampled.max_row_residual, actual_sir.DTYPE),
                tf.cast(resampled.max_column_residual, actual_sir.DTYPE),
                resampled.finite_factors,
                resampled.finite_particles,
                resampled.iterations_used,
                tf.constant(1, dtype=tf.int32),
                tf.cast(resampled.min_kernel_denominator, actual_sir.DTYPE),
                tf.cast(resampled.denominator_floor_hits, actual_sir.DTYPE),
                tf.cast(resampled.max_abs_log_scaling_gauge_shift, actual_sir.DTYPE),
                tf.cast(resampled.scaling_normalization_applications, actual_sir.DTYPE),
                tf.cast(resampled.max_factor_diag_error, actual_sir.DTYPE),
                tf.cast(resampled.min_factor_diagonal, actual_sir.DTYPE),
                tf.cast(resampled.max_factor_diagonal, actual_sir.DTYPE),
                tf.cast(resampled.landmark_core_min_eigenvalue, actual_sir.DTYPE),
                tf.cast(resampled.landmark_core_max_eigenvalue, actual_sir.DTYPE),
                tf.cast(resampled.landmark_core_condition_proxy, actual_sir.DTYPE),
                tf.cast(resampled.landmark_core_effective_rank, actual_sir.DTYPE),
                tf.cast(resampled.left_factor_min, actual_sir.DTYPE),
                tf.cast(resampled.left_factor_max, actual_sir.DTYPE),
                tf.cast(resampled.core_matrix_min, actual_sir.DTYPE),
                tf.cast(resampled.core_matrix_max, actual_sir.DTYPE),
                tf.cast(resampled.raw_kernel_min, actual_sir.DTYPE),
                tf.cast(resampled.projected_kernel_min, actual_sir.DTYPE),
                tf.cast(resampled.projection_floor_hits, actual_sir.DTYPE),
                tf.cast(resampled.scaling_u_min, actual_sir.DTYPE),
                tf.cast(resampled.scaling_u_max, actual_sir.DTYPE),
                tf.cast(resampled.scaling_v_min, actual_sir.DTYPE),
                tf.cast(resampled.scaling_v_max, actual_sir.DTYPE),
            )

        def skip_transport():
            nan = tf.constant(float("nan"), dtype=actual_sir.DTYPE)
            return (
                post_flow,
                normalized_log_weights,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(True),
                tf.constant(True),
                tf.constant(0, dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                nan,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                tf.constant(0.0, dtype=actual_sir.DTYPE),
                nan,
                nan,
                nan,
                nan,
                nan,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
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

        (
            next_particles,
            next_log_weights,
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
        ) = tf.cond(tf.reduce_any(mask), do_transport, skip_transport)
        has_invocation = step_invocations > tf.constant(0, dtype=tf.int32)
        return (
            time_tensor + tf.constant(1, dtype=tf.int32),
            next_particles,
            next_log_weights,
            next_log_likelihood,
            tf.maximum(current_max_row_residual, row_residual),
            tf.maximum(current_max_column_residual, column_residual),
            tf.logical_and(current_finite_factors, step_finite_factors),
            tf.logical_and(current_finite_particles, step_finite_particles),
            tf.maximum(current_iterations_used_max, step_iterations),
            current_route_invocations + step_invocations,
            tf.where(
                has_invocation,
                tf.minimum(current_min_kernel_denominator, step_min_kernel_denominator),
                current_min_kernel_denominator,
            ),
            current_denominator_floor_hits + tf.where(
                has_invocation,
                step_denominator_floor_hits,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
            ),
            tf.where(
                has_invocation,
                tf.maximum(current_max_abs_log_scaling_gauge_shift, step_max_abs_log_scaling_gauge_shift),
                current_max_abs_log_scaling_gauge_shift,
            ),
            current_scaling_normalization_applications + tf.where(
                has_invocation,
                step_scaling_normalization_applications,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
            ),
            tf.where(
                has_invocation,
                tf.maximum(current_max_factor_diag_error, step_max_factor_diag_error),
                current_max_factor_diag_error,
            ),
            tf.where(
                has_invocation,
                tf.minimum(current_min_factor_diagonal, step_min_factor_diagonal),
                current_min_factor_diagonal,
            ),
            tf.where(
                has_invocation,
                tf.maximum(current_max_factor_diagonal, step_max_factor_diagonal),
                current_max_factor_diagonal,
            ),
            tf.where(
                has_invocation,
                tf.minimum(current_landmark_core_min_eigenvalue, step_landmark_core_min_eigenvalue),
                current_landmark_core_min_eigenvalue,
            ),
            tf.where(
                has_invocation,
                tf.maximum(current_landmark_core_max_eigenvalue, step_landmark_core_max_eigenvalue),
                current_landmark_core_max_eigenvalue,
            ),
            tf.where(
                has_invocation,
                tf.maximum(current_landmark_core_condition_proxy, step_landmark_core_condition_proxy),
                current_landmark_core_condition_proxy,
            ),
            tf.where(
                has_invocation,
                tf.minimum(current_landmark_core_effective_rank_min, step_landmark_core_effective_rank),
                current_landmark_core_effective_rank_min,
            ),
            tf.where(has_invocation, tf.minimum(current_left_factor_min, step_left_factor_min), current_left_factor_min),
            tf.where(has_invocation, tf.maximum(current_left_factor_max, step_left_factor_max), current_left_factor_max),
            tf.where(has_invocation, tf.minimum(current_core_matrix_min, step_core_matrix_min), current_core_matrix_min),
            tf.where(has_invocation, tf.maximum(current_core_matrix_max, step_core_matrix_max), current_core_matrix_max),
            tf.where(has_invocation, tf.minimum(current_raw_kernel_min, step_raw_kernel_min), current_raw_kernel_min),
            tf.where(
                has_invocation,
                tf.minimum(current_projected_kernel_min, step_projected_kernel_min),
                current_projected_kernel_min,
            ),
            current_projection_floor_hits + tf.where(
                has_invocation,
                step_projection_floor_hits,
                tf.constant(0.0, dtype=actual_sir.DTYPE),
            ),
            tf.where(has_invocation, tf.minimum(current_scaling_u_min, step_scaling_u_min), current_scaling_u_min),
            tf.where(has_invocation, tf.maximum(current_scaling_u_max, step_scaling_u_max), current_scaling_u_max),
            tf.where(has_invocation, tf.minimum(current_scaling_v_min, step_scaling_v_min), current_scaling_v_min),
            tf.where(has_invocation, tf.maximum(current_scaling_v_max, step_scaling_v_max), current_scaling_v_max),
            current_means_ta,
            current_vars_ta,
            current_ess_ta,
        )

    def cond(
        time_tensor: tf.Tensor,
        _particles: tf.Tensor,
        _log_weights: tf.Tensor,
        _log_likelihood: tf.Tensor,
        _max_row_residual: tf.Tensor,
        _max_column_residual: tf.Tensor,
        _finite_factors: tf.Tensor,
        _finite_particles: tf.Tensor,
        _iterations_used_max: tf.Tensor,
        _route_invocations: tf.Tensor,
        _min_kernel_denominator: tf.Tensor,
        _denominator_floor_hits: tf.Tensor,
        _max_abs_log_scaling_gauge_shift: tf.Tensor,
        _scaling_normalization_applications: tf.Tensor,
        _max_factor_diag_error: tf.Tensor,
        _min_factor_diagonal: tf.Tensor,
        _max_factor_diagonal: tf.Tensor,
        _landmark_core_min_eigenvalue: tf.Tensor,
        _landmark_core_max_eigenvalue: tf.Tensor,
        _landmark_core_condition_proxy: tf.Tensor,
        _landmark_core_effective_rank_min: tf.Tensor,
        _left_factor_min: tf.Tensor,
        _left_factor_max: tf.Tensor,
        _core_matrix_min: tf.Tensor,
        _core_matrix_max: tf.Tensor,
        _raw_kernel_min: tf.Tensor,
        _projected_kernel_min: tf.Tensor,
        _projection_floor_hits: tf.Tensor,
        _scaling_u_min: tf.Tensor,
        _scaling_u_max: tf.Tensor,
        _scaling_v_min: tf.Tensor,
        _scaling_v_max: tf.Tensor,
        _means_ta: tf.TensorArray,
        _vars_ta: tf.TensorArray,
        _ess_ta: tf.TensorArray,
    ) -> tf.Tensor:
        return time_tensor < tf.constant(time_steps, dtype=tf.int32)

    (
        _,
        particles,
        log_weights,
        log_likelihood,
        max_row_residual,
        max_column_residual,
        finite_factors,
        finite_particles,
        iterations_used_max,
        route_invocations,
        min_kernel_denominator,
        denominator_floor_hits,
        max_abs_log_scaling_gauge_shift,
        scaling_normalization_applications,
        max_factor_diag_error,
        min_factor_diagonal,
        max_factor_diagonal,
        landmark_core_min_eigenvalue,
        landmark_core_max_eigenvalue,
        landmark_core_condition_proxy,
        landmark_core_effective_rank_min,
        left_factor_min,
        left_factor_max,
        core_matrix_min,
        core_matrix_max,
        raw_kernel_min,
        projected_kernel_min,
        projection_floor_hits,
        scaling_u_min,
        scaling_u_max,
        scaling_v_min,
        scaling_v_max,
        means_ta,
        vars_ta,
        ess_ta,
    ) = tf.while_loop(
        cond,
        step_body,
        loop_vars=(
            tf.constant(0, dtype=tf.int32),
            particles,
            log_weights,
            log_likelihood,
            max_row_residual,
            max_column_residual,
            finite_factors,
            finite_particles,
            iterations_used_max,
            route_invocations,
            min_kernel_denominator,
            denominator_floor_hits,
            max_abs_log_scaling_gauge_shift,
            scaling_normalization_applications,
            max_factor_diag_error,
            min_factor_diagonal,
            max_factor_diagonal,
            landmark_core_min_eigenvalue,
            landmark_core_max_eigenvalue,
            landmark_core_condition_proxy,
            landmark_core_effective_rank_min,
            left_factor_min,
            left_factor_max,
            core_matrix_min,
            core_matrix_max,
            raw_kernel_min,
            projected_kernel_min,
            projection_floor_hits,
            scaling_u_min,
            scaling_u_max,
            scaling_v_min,
            scaling_v_max,
            means_ta,
            vars_ta,
            ess_ta,
        ),
        parallel_iterations=1,
        maximum_iterations=time_steps,
    )

    if args.history_mode == "full":
        filtered_means = means_ta.stack()
        filtered_variances = vars_ta.stack()
        ess_by_time = ess_ta.stack()
        filtered_means.set_shape([time_steps, batch_size, state_dim])
        filtered_variances.set_shape([time_steps, batch_size, state_dim])
        ess_by_time.set_shape([time_steps, batch_size])
    else:
        filtered_means = tf.zeros([0, batch_size, state_dim], dtype=actual_sir.DTYPE)
        filtered_variances = tf.zeros([0, batch_size, state_dim], dtype=actual_sir.DTYPE)
        ess_by_time = tf.zeros([0, batch_size], dtype=actual_sir.DTYPE)

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


def _timed_call(fn) -> tuple[float, list[float], tuple[tf.Tensor, ...]]:
    start = time.perf_counter()
    outputs = fn()
    _materialize(*outputs)
    compile_and_first = time.perf_counter() - start
    return compile_and_first, [], outputs


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
        row.update(
            {
                "filtered_means": _as_float_list(outputs[1]),
                "filtered_variances": _as_float_list(outputs[2]),
                "ess_by_time": _as_float_list(outputs[3]),
                "ess_min": _float(tf.reduce_min(outputs[3])),
                "ess_fraction_min": _float(tf.reduce_min(outputs[3])) / float(args.num_particles),
            }
        )
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
                "min_kernel_denominator": _float(outputs[11]),
                "denominator_floor_hits": _float(outputs[12]),
                "nystrom_scaling_normalization": args.nystrom_scaling_normalization,
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
            }
        )
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


def _paired(rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if set(rows) != {"streaming", "nystrom"}:
        return None
    streaming_ll = tf.constant(rows["streaming"]["log_likelihood"], dtype=tf.float64)
    nystrom_ll = tf.constant(rows["nystrom"]["log_likelihood"], dtype=tf.float64)
    ll_delta = nystrom_ll - streaming_ll
    paired = {
        "log_likelihood_delta_by_seed": ll_delta.numpy().tolist(),
        "log_likelihood_max_abs_delta": _float(tf.reduce_max(tf.abs(ll_delta))),
        "log_likelihood_mean_abs_delta": _float(tf.reduce_mean(tf.abs(ll_delta))),
        "thresholds": {
            "log_likelihood_max_abs_delta": LOG_LIKELIHOOD_MAX_ABS_DELTA_THRESHOLD,
            "log_likelihood_mean_abs_delta": LOG_LIKELIHOOD_MEAN_ABS_DELTA_THRESHOLD,
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


def _paired_vetoes(paired: dict[str, Any] | None) -> list[str]:
    if paired is None:
        return []
    vetoes = []
    if paired["log_likelihood_max_abs_delta"] > LOG_LIKELIHOOD_MAX_ABS_DELTA_THRESHOLD:
        vetoes.append("paired_log_likelihood_max_abs_delta")
    if paired["log_likelihood_mean_abs_delta"] > LOG_LIKELIHOOD_MEAN_ABS_DELTA_THRESHOLD:
        vetoes.append("paired_log_likelihood_mean_abs_delta")
    return vetoes


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    smi_rows = _nvidia_smi_rows()
    selected_physical_gpu = _selected_physical_gpu(os.environ.get("CUDA_VISIBLE_DEVICES"), smi_rows)
    callbacks = actual_sir._dpf_sir_callbacks()
    tensors, sir_semantics = actual_sir._build_actual_sir_tensors(args)
    adapter_callbacks = actual_sir._make_actual_sir_callbacks(callbacks, tensors, args.batch_seeds, args)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_streaming() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return _streaming_outputs(tensors, adapter_callbacks, args)

    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_nystrom() -> tuple[tf.Tensor, ...]:
        value = _nystrom_value_core(tensors, adapter_callbacks, args)
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
    memory_before = _gpu_memory_info()
    with tf.device(args.device):
        for route in _route_names(args.route):
            fn = route_fns[route]
            start = time.perf_counter()
            outputs = fn()
            _materialize(*outputs)
            compile_and_first = time.perf_counter() - start
            for _ in range(args.warmups):
                _materialize(*fn())
            timings: list[float] = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                outputs = fn()
                _materialize(*outputs)
                timings.append(time.perf_counter() - start)
            rows_by_route[route] = _route_row(
                route,
                outputs,
                compile_and_first=compile_and_first,
                timings=timings,
                args=args,
            )
    memory_after = _gpu_memory_info()
    wall = time.perf_counter() - start_all
    ended = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    paired = _paired(rows_by_route)
    aggregate_hard_vetoes = [
        f"{route}:{veto}"
        for route, row in rows_by_route.items()
        for veto in row["hard_vetoes"]
    ]
    aggregate_hard_vetoes.extend(f"paired:{veto}" for veto in _paired_vetoes(paired))
    if args.expect_device_kind == "gpu":
        if not logical_gpus:
            aggregate_hard_vetoes.append("gpu_device_evidence_missing")
        if precision.get("tf32_execution_enabled") is not True and args.dtype == "float32" and args.tf32_mode == "enabled":
            aggregate_hard_vetoes.append("tf32_not_recorded_enabled_for_float32")
    return {
        "schema_version": "actual_sir_nystrom_compiled_redo.v1",
        "status": "PASS" if not aggregate_hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "hard_vetoes": aggregate_hard_vetoes,
        "evidence_contract": {
            "question": "Redo actual-SIR Nystrom benchmark under compiled comparable route conditions.",
            "baseline": "compiled production-style streaming TF32 actual-SIR value path",
            "candidate": "compiled tensor-only fixed-rank Nystrom actual-SIR route",
            "primary_pass": "finite GPU outputs, compiled route evidence, no Nystrom residual veto, paired log-likelihood thresholds for paired rows",
            "promotion_role": "redo evidence only; no default readiness without replicated uncertainty, stress, and HMC gates",
            "quarantined_prior_artifacts": "prior Python-loop actual-SIR Nystrom timing artifacts are non-authoritative for speed/ranking",
        },
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "history_mode": args.history_mode,
        "jit_compile": args.jit_compile,
        "route_request": args.route,
        "routes_executed": _route_names(args.route),
        "rows": [rows_by_route[route] for route in _route_names(args.route)],
        "paired_comparability": paired,
        "precision": precision,
        "sir_semantics": sir_semantics,
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
            "nystrom_kernel_mode_scope": (
                "diagnostic_dense_positive_projection"
                if args.nystrom_kernel_mode == "positive_projected"
                else "raw_factor_application"
            ),
            "nystrom_scaling_normalization": args.nystrom_scaling_normalization,
            "nystrom_scaling_normalization_scope": (
                "none"
                if args.nystrom_scaling_normalization == "none"
                else "opt_in_batchwise_sinkhorn_factor_gauge_balancing"
            ),
        },
        "run_manifest": {
            "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": _run_text(["git", "status", "--short"]),
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
            "gpu_memory_info_before": memory_before,
            "gpu_memory_info_after": memory_after,
            "output": args.output,
            "markdown_output": args.markdown_output,
        },
        "inference_status": {
            "hard_veto_screen": "PASS" if not aggregate_hard_vetoes else "FAIL",
            "statistically_supported_ranking": "NO",
            "descriptive_only_differences": "warm timing ratios are descriptive until replicated uncertainty analysis",
            "default_readiness": "NO",
            "next_evidence_needed": "small compiled redo, then replicated serious rows if the small redo passes",
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
        "# Actual-SIR Nystrom Compiled Redo Benchmark",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Shape: `{result['shape']}`",
        f"- Route request: `{result['route_request']}`",
        f"- JIT compile: `{result['jit_compile']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        "",
        "## Routes",
        "",
        "| Route | Status | Compile+first seconds | Warm median seconds | Hard vetoes |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {route} | `{status}` | `{first}` | `{median}` | `{vetoes}` |".format(
                route=row["route"],
                status=row["status"],
                first=row["compile_and_first_call_seconds"],
                median=(row.get("warm_call_timing_summary_seconds") or {}).get("median"),
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
        write_markdown(result, Path(args.markdown_output), output_path)
    if not args.quiet:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
