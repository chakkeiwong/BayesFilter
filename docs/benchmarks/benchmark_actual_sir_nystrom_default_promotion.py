"""Actual-SIR d18 Nystrom route pilot for default-promotion testing.

This harness reuses the existing P8j actual-SIR tensor/callback setup and
compares the current streaming TF32 route with the experimental fixed-rank
Nystrom route.  It is a serious-model viability screen only and does not change
BayesFilter defaults.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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
    nystrom_transport_resample_tf,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-default-promotion-plan-2026-06-22.md"
RESULT_PATH = "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-actual-sir-nystrom-default-promotion-result-2026-06-22.md"
NYSTROM_RESIDUAL_THRESHOLD = 5.0e-2
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-5
ESS_FRACTION_MIN_THRESHOLD = 0.01
NONCLAIMS = (
    "actual-SIR d18 Nystrom viability screen only",
    "no default readiness claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no dense Sinkhorn equivalence claim",
    "no broad scalable-OT selection claim",
    "no statistical ranking claim",
)


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
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--nystrom-rank", type=int, default=32)
    parser.add_argument("--nystrom-epsilon", type=float, default=0.5)
    parser.add_argument("--nystrom-max-iterations", type=int, default=160)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--nystrom-cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--nystrom-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float32")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--selected-physical-gpu", default=None)
    parser.add_argument("--gpu-selection-note", default=None)
    parser.add_argument("--phase-id", default="ACTUAL-SIR-NYSTROM-PILOT")
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
    if args.nystrom_rank <= 0:
        raise ValueError("nystrom_rank must be positive")
    if args.nystrom_rank > args.num_particles:
        raise ValueError("nystrom_rank must be <= num_particles")
    if args.nystrom_epsilon <= 0.0:
        raise ValueError("nystrom_epsilon must be positive")
    if args.nystrom_max_iterations <= 0:
        raise ValueError("nystrom_max_iterations must be positive")
    if args.nystrom_convergence_threshold <= 0.0:
        raise ValueError("nystrom_convergence_threshold must be positive")
    if args.nystrom_cholesky_jitter < 0.0:
        raise ValueError("nystrom_cholesky_jitter must be non-negative")
    if args.nystrom_denominator_floor <= 0.0:
        raise ValueError("nystrom_denominator_floor must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn_iterations must be positive")
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


def _route_names(route: str) -> list[str]:
    if route == "both":
        return ["streaming", "nystrom"]
    return [route]


def _active_step_count(mask: tf.Tensor) -> tuple[int, int]:
    mask_np = mask.numpy()
    active_steps = 0
    for time_index in range(mask_np.shape[1]):
        column = mask_np[:, time_index]
        if bool(column.any()) and not bool(column.all()):
            raise ValueError("actual-SIR Nystrom harness supports all-batch or no-batch resampling per step")
        if bool(column.all()):
            active_steps += 1
    return active_steps, int(mask_np.sum())


def _materialize_loop_outputs(loop: dict[str, Any]) -> None:
    for key in (
        "log_likelihood",
        "filtered_means",
        "filtered_variances",
        "ess_by_time",
        "final_particle_mean",
        "final_log_weights",
    ):
        value = loop.get(key)
        if isinstance(value, tf.Tensor):
            value.numpy()


def _run_route_loop(
    route: str,
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
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
    means: list[tf.Tensor] = []
    variances: list[tf.Tensor] = []
    esses: list[tf.Tensor] = []
    route_invocations = 0
    transport_matrix_shapes: list[list[int]] = []
    step_diagnostics: list[dict[str, Any]] = []
    streaming_row_residuals: list[float] = []
    streaming_column_residuals: list[float] = []
    active_steps, active_batch_entries = _active_step_count(fixed_mask)

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
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = core_tf._weighted_mean_and_variance(post_flow, weights)
        means.append(mean)
        variances.append(variance)
        esses.append(ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, core_tf._log_weight_floor()))
        mask = fixed_mask[:, time_index]
        active = bool(tf.reduce_any(mask).numpy())
        if not active:
            particles = post_flow
            log_weights = normalized_log_weights
            continue
        route_invocations += 1
        if route == "streaming":
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
            transport_matrix_shapes.append(transported.transport_matrix.shape.as_list())
            streaming_row_residuals.append(_float(transported.max_row_residual))
            streaming_column_residuals.append(_float(transported.max_column_residual))
            particles = transported.particles
            log_weights = transported.log_weights
        elif route == "nystrom":
            resampled = nystrom_transport_resample_tf(
                post_flow,
                normalized_log_weights,
                rank=args.nystrom_rank,
                epsilon=args.nystrom_epsilon,
                max_iterations=args.nystrom_max_iterations,
                convergence_threshold=args.nystrom_convergence_threshold,
                cholesky_jitter=args.nystrom_cholesky_jitter,
                denominator_floor=args.nystrom_denominator_floor,
            )
            transport_matrix_shapes.append(resampled.transport_matrix.shape.as_list())
            step_diagnostics.append(dict(resampled.diagnostics))
            particles = tf.convert_to_tensor(resampled.particles, dtype=actual_sir.DTYPE)
            log_weights = tf.convert_to_tensor(resampled.log_weights, dtype=actual_sir.DTYPE)
        else:
            raise ValueError(f"unknown route: {route}")

    filtered_means = tf.stack(means, axis=0)
    filtered_variances = tf.stack(variances, axis=0)
    ess_by_time = tf.stack(esses, axis=0)
    final_weights = tf.exp(log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True))
    final_particle_mean, _ = core_tf._weighted_mean_and_variance(particles, final_weights)
    final_logsumexp_residual = tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))
    return {
        "route": route,
        "log_likelihood": log_likelihood,
        "filtered_means": filtered_means,
        "filtered_variances": filtered_variances,
        "ess_by_time": ess_by_time,
        "final_particles": particles,
        "final_log_weights": log_weights,
        "final_particle_mean": final_particle_mean,
        "final_logsumexp_residual": final_logsumexp_residual,
        "route_invocations": route_invocations,
        "active_resampling_mask_count": active_steps,
        "active_resampling_batch_entries": active_batch_entries,
        "transport_matrix_shapes": transport_matrix_shapes,
        "step_diagnostics": step_diagnostics,
        "streaming_row_residuals": streaming_row_residuals,
        "streaming_column_residuals": streaming_column_residuals,
        "batch_size": batch_size,
        "num_particles": num_particles,
        "state_dim": state_dim,
        "time_steps": time_steps,
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
    if route == "nystrom":
        step_diagnostics = loop["step_diagnostics"]
        max_row = _max_diag(step_diagnostics, "max_row_residual")
        max_col = _max_diag(step_diagnostics, "max_column_residual")
        if _all_diag(step_diagnostics, "finite_factors") is False:
            hard_vetoes.append("nonfinite_nystrom_factors")
        if _all_diag(step_diagnostics, "finite_particles") is False:
            hard_vetoes.append("nystrom_nonfinite_particles")
        if _all_diag(step_diagnostics, "transport_matrix_materialized") is not False:
            hard_vetoes.append("nystrom_transport_matrix_materialized")
        if loop["active_resampling_mask_count"] > 0 and max_row is None:
            hard_vetoes.append("missing_nystrom_residual_diagnostics")
        if max_row is not None and max_row > NYSTROM_RESIDUAL_THRESHOLD:
            hard_vetoes.append("nystrom_row_residual_threshold")
        if max_col is not None and max_col > NYSTROM_RESIDUAL_THRESHOLD:
            hard_vetoes.append("nystrom_column_residual_threshold")
    return hard_vetoes


def _run_timed_route(route: str, tensors: dict[str, tf.Tensor], callbacks: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    memory_before = _gpu_memory_info()
    start = time.perf_counter()
    loop = _run_route_loop(route, tensors, callbacks, args)
    _materialize_loop_outputs(loop)
    first_call_seconds = time.perf_counter() - start
    for _ in range(args.warmups):
        warm_loop = _run_route_loop(route, tensors, callbacks, args)
        _materialize_loop_outputs(warm_loop)
    timings: list[float] = []
    last_loop = loop
    for _ in range(args.repeats):
        start = time.perf_counter()
        last_loop = _run_route_loop(route, tensors, callbacks, args)
        _materialize_loop_outputs(last_loop)
        timings.append(time.perf_counter() - start)
    memory_after = _gpu_memory_info()
    return {
        "first_call_seconds": first_call_seconds,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "loop": last_loop,
        "memory_maxrss_kb_explanatory": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


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
        "log_likelihood": _as_float_array(loop["log_likelihood"]),
        "filtered_means": _as_float_array(loop["filtered_means"]),
        "filtered_variances": _as_float_array(loop["filtered_variances"]),
        "ess_by_time": _as_float_array(loop["ess_by_time"]),
        "final_particle_mean": _as_float_array(loop["final_particle_mean"]),
    }
    if route == "streaming":
        row.update(
            {
                "transport_object_kind": "streaming_transport",
                "streaming_max_row_residual": max(loop["streaming_row_residuals"]) if loop["streaming_row_residuals"] else None,
                "streaming_max_column_residual": max(loop["streaming_column_residuals"]) if loop["streaming_column_residuals"] else None,
            }
        )
    else:
        step_diagnostics = loop["step_diagnostics"]
        row.update(
            {
                "transport_object_kind": "nystrom_kernel_factors",
                "nystrom_rank": args.nystrom_rank,
                "nystrom_epsilon": args.nystrom_epsilon,
                "nystrom_max_iterations": args.nystrom_max_iterations,
                "nystrom_convergence_threshold": args.nystrom_convergence_threshold,
                "nystrom_cholesky_jitter": args.nystrom_cholesky_jitter,
                "nystrom_denominator_floor": args.nystrom_denominator_floor,
                "all_finite_factors": _all_diag(step_diagnostics, "finite_factors"),
                "all_finite_resampled_particles": _all_diag(step_diagnostics, "finite_particles"),
                "max_row_residual": _max_diag(step_diagnostics, "max_row_residual"),
                "max_column_residual": _max_diag(step_diagnostics, "max_column_residual"),
                "iterations_used_max_explanatory": _max_diag(step_diagnostics, "iterations_used"),
                "landmark_indices_by_step": [diag.get("landmark_indices") for diag in step_diagnostics],
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
    if set(timed) != {"streaming", "nystrom"}:
        return None
    streaming = timed["streaming"]["loop"]
    nystrom = timed["nystrom"]["loop"]
    ll_delta = tf.cast(nystrom["log_likelihood"] - streaming["log_likelihood"], tf.float64)
    mean_delta = tf.cast(nystrom["filtered_means"] - streaming["filtered_means"], tf.float64)
    var_delta = tf.cast(nystrom["filtered_variances"] - streaming["filtered_variances"], tf.float64)
    final_mean_delta = tf.cast(nystrom["final_particle_mean"] - streaming["final_particle_mean"], tf.float64)
    streaming_warm = next((row for row in rows if row["route"] == "streaming"), {})
    nystrom_warm = next((row for row in rows if row["route"] == "nystrom"), {})
    streaming_median = (streaming_warm.get("warm_call_timing_summary_seconds") or {}).get("median")
    nystrom_median = (nystrom_warm.get("warm_call_timing_summary_seconds") or {}).get("median")
    warm_ratio = (
        float(streaming_median) / float(nystrom_median)
        if streaming_median is not None and nystrom_median not in (None, 0.0)
        else None
    )
    return {
        "role": "actual_sir_nystrom_viability_screen",
        "log_likelihood_delta_by_seed": _as_float_array(ll_delta),
        "log_likelihood_max_abs_delta": _float(tf.reduce_max(tf.abs(ll_delta))),
        "log_likelihood_mean_abs_delta": _float(tf.reduce_mean(tf.abs(ll_delta))),
        "filtered_mean_relative_l2": _relative_l2(mean_delta, streaming["filtered_means"]),
        "filtered_mean_rms": _rms(mean_delta),
        "filtered_variance_relative_l2": _relative_l2(var_delta, streaming["filtered_variances"]),
        "filtered_variance_rms": _rms(var_delta),
        "final_particle_mean_relative_l2": _relative_l2(final_mean_delta, streaming["final_particle_mean"]),
        "final_particle_mean_abs_l2": _float(tf.linalg.norm(tf.reshape(final_mean_delta, [-1]))),
        "warm_median_streaming_over_nystrom": warm_ratio,
        "thresholds": {
            "log_likelihood_max_abs_delta": 10.0,
            "log_likelihood_mean_abs_delta": 5.0,
            "filtered_mean_relative_l2": 0.20,
            "filtered_mean_rms": 2.5,
            "filtered_variance_relative_l2": 0.75,
            "filtered_variance_rms": 25.0,
            "final_particle_mean_relative_l2": 0.20,
            "final_particle_mean_abs_l2": 25.0,
        },
    }


def _paired_vetoes(paired: dict[str, Any] | None) -> list[str]:
    if paired is None:
        return []
    vetoes: list[str] = []
    if paired["log_likelihood_max_abs_delta"] > 10.0:
        vetoes.append("paired_log_likelihood_max_abs_delta")
    if paired["log_likelihood_mean_abs_delta"] > 5.0:
        vetoes.append("paired_log_likelihood_mean_abs_delta")
    if paired["filtered_mean_relative_l2"] > 0.20 and paired["filtered_mean_rms"] > 2.5:
        vetoes.append("paired_filtered_mean_delta")
    if paired["filtered_variance_relative_l2"] > 0.75 and paired["filtered_variance_rms"] > 25.0:
        vetoes.append("paired_filtered_variance_delta")
    if paired["final_particle_mean_relative_l2"] > 0.20 and paired["final_particle_mean_abs_l2"] > 25.0:
        vetoes.append("paired_final_particle_mean_delta")
    return vetoes


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
        "selected_physical_gpu_argument": args.selected_physical_gpu,
        "gpu_selection_note": args.gpu_selection_note,
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "nvidia_smi_rows": smi_rows,
        "selected_physical_gpu": selected_physical_gpu,
        "selected_physical_gpu_policy": "physical GPU1 preferred if usable, otherwise physical GPU0",
        "precision": precision,
        "phase_id": args.phase_id,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
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
    paired = _paired_comparability(rows, timed)
    aggregate_hard_vetoes.extend(f"paired:{veto}" for veto in _paired_vetoes(paired))
    if args.expect_device_kind == "gpu":
        if not logical_gpus:
            aggregate_hard_vetoes.append("gpu_device_evidence_missing")
        if precision.get("tf32_execution_enabled") is not True and args.dtype == "float32" and args.tf32_mode == "enabled":
            aggregate_hard_vetoes.append("tf32_not_recorded_enabled_for_float32")
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
    return {
        "schema_version": "actual_sir_nystrom_default_promotion.v1",
        "status": "PASS" if not aggregate_hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "algorithm_under_test": "actual-SIR d18 LEDH/PFPF-OT streaming route versus fixed-rank Nystrom resampling",
        "route_request": args.route,
        "routes_executed": _route_names(args.route),
        "hard_vetoes": aggregate_hard_vetoes,
        "thresholds": {
            "nystrom_residual": NYSTROM_RESIDUAL_THRESHOLD,
            "final_logsumexp_residual": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction_min": ESS_FRACTION_MIN_THRESHOLD,
            "nystrom_transport_matrix_shape_suffix": [0, 0],
            "runtime_memory_role": "explanatory_until_replicated_default_promotion_lane",
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
        "inference_status": {
            "hard_veto_screen": "PASS" if not aggregate_hard_vetoes else "FAIL",
            "statistically_supported_ranking": "NO",
            "descriptive_only_differences": "runtime and memory are descriptive in this pilot",
            "default_readiness": "NO",
            "next_evidence_needed": "P02 GPU pilot pass, then B=5,T=20,N=1024 actual-SIR row and replicated ladder",
        },
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
        "# Actual-SIR Nystrom Default-Promotion Pilot",
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
            "warm_median_streaming_over_nystrom",
        ):
            lines.append(f"- {key}: `{paired.get(key)}`")
    lines.extend(
        [
            "",
            "## Inference Status",
            "",
            "| Ledger | Status |",
            "| --- | --- |",
        ]
    )
    for key, value in result["inference_status"].items():
        lines.append(f"| `{key}` | `{value}` |")
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
    if not args.quiet:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
