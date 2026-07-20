"""Paired Nystrom versus streaming LEDH/PFPF-OT effectiveness pilot.

This lane-owned harness compares the experimental fixed-rank Nystrom transport
candidate with the current streaming TF32 LEDH/PFPF-OT route on a common
deterministic downstream fixture.  It is a pilot usefulness screen only: it does
not establish posterior correctness, statistical superiority, default
readiness, HMC readiness, or public API readiness.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
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
_PRE.add_argument("--mode", choices=("tiny-cpu", "paired-gpu"), default="tiny-cpu")
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default=None)
_PRE.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu" or (_PRE_ARGS.device_scope is None and _PRE_ARGS.mode == "tiny-cpu"):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

try:  # noqa: E402
    import tensorflow_probability as tfp  # noqa: E402
except Exception:  # noqa: BLE001
    tfp = None

from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (  # noqa: E402
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.experimental_batched_ledh_pfpf_ot_tf import (  # noqa: E402
    _log_weight_floor,
    _normalize_log_weights,
    _weighted_mean_and_variance,
    batched_annealed_transport_core_tf,
    uniform_log_weights,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling import nystrom_transport_tf  # noqa: E402
from experiments.dpf_implementation.tf_tfp.resampling.nystrom_transport_tf import (  # noqa: E402
    nystrom_transport_resample_tf,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-effectiveness-leaderboard-plan-2026-06-22.md"
PILOT_RESULT_PATH = "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-nystrom-effectiveness-leaderboard-result-2026-06-22.md"
FIXTURE_ID = "nystrom_effectiveness_ledh_lgssm_common_v1"
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-6
ESS_FRACTION_THRESHOLD = 1.0e-2
NYSTROM_RESIDUAL_THRESHOLD = 5.0e-2
STATE_MEAN_RELATIVE_L2_THRESHOLD = 0.5
STATE_MEAN_ABSOLUTE_L2_THRESHOLD = 1.0
LOG_LIKELIHOOD_ABSOLUTE_L2_THRESHOLD = 1.0
P02_ROW_TIMEOUT_SECONDS = 900
DEFAULT_PAIRED_GPU_COUNTS = [1024, 4096]
NONCLAIMS = (
    "paired usefulness pilot only",
    "no speedup claim",
    "no superiority claim",
    "no statistical ranking claim",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no production/default route change",
    "no dense Sinkhorn equivalence claim",
)


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("tiny-cpu", "paired-gpu"), default=_PRE_ARGS.mode)
    parser.add_argument("--particle-counts", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--nystrom-max-iterations", type=int, default=160)
    parser.add_argument("--nystrom-convergence-threshold", type=float, default=1.0e-4)
    parser.add_argument("--cholesky-jitter", type=float, default=1.0e-8)
    parser.add_argument("--denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-scaling", type=float, default=0.9)
    parser.add_argument("--sinkhorn-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--particle-chunk-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--row-timeout-seconds", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("cpu", "gpu", "any"), default=None)
    parser.add_argument("--tf32-mode", choices=("enabled", "disabled", "default"), default=None)
    parser.add_argument("--trust-context", default=None)
    parser.add_argument("--selected-physical-gpu", default=None)
    parser.add_argument("--gpu-selection-note", default=None)
    parser.add_argument("--phase-id", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args(argv)
    args = _apply_mode_defaults(args)
    _validate_args(args)
    return args


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "tiny-cpu":
        args.particle_counts = args.particle_counts or [16]
        args.time_steps = args.time_steps or 1
        args.state_dim = args.state_dim or 3
        args.obs_dim = args.obs_dim or 2
        args.rank = args.rank or 4
        args.dtype = args.dtype or "float64"
        args.device_scope = args.device_scope or "cpu"
        args.device = args.device or "/CPU:0"
        args.expect_device_kind = args.expect_device_kind or "cpu"
        args.tf32_mode = args.tf32_mode or "disabled"
        args.trust_context = args.trust_context or "cpu_hidden_local"
        args.row_timeout_seconds = args.row_timeout_seconds or 60
    else:
        args.particle_counts = args.particle_counts or list(DEFAULT_PAIRED_GPU_COUNTS)
        args.time_steps = args.time_steps or 2
        args.state_dim = args.state_dim or 8
        args.obs_dim = args.obs_dim or 6
        args.rank = args.rank or 32
        args.dtype = args.dtype or "float32"
        args.device_scope = args.device_scope or "visible"
        args.device = args.device or "/GPU:0"
        args.expect_device_kind = args.expect_device_kind or "gpu"
        args.tf32_mode = args.tf32_mode or "enabled"
        args.trust_context = args.trust_context or "trusted_gpu_required"
        args.row_timeout_seconds = args.row_timeout_seconds or P02_ROW_TIMEOUT_SECONDS
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if any(count <= 1 for count in args.particle_counts):
        raise ValueError("particle-counts must be greater than 1")
    for name in ("batch_size", "time_steps", "state_dim", "obs_dim", "rank"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.rank > min(args.particle_counts):
        raise ValueError("rank must be <= every particle count")
    for name in ("epsilon", "nystrom_convergence_threshold", "denominator_floor", "sinkhorn_convergence_threshold"):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.cholesky_jitter < 0.0:
        raise ValueError("cholesky-jitter must be non-negative")
    if not 0.0 < args.sinkhorn_scaling <= 1.0:
        raise ValueError("sinkhorn-scaling must be in (0, 1]")
    if args.nystrom_max_iterations <= 0 or args.sinkhorn_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_timeout_seconds <= 0:
        raise ValueError("row-timeout-seconds must be positive")
    if args.mode == "tiny-cpu" and args.device_scope != "cpu":
        raise ValueError("tiny-cpu mode requires --device-scope cpu")
    if args.mode == "paired-gpu" and args.device_scope != "visible":
        raise ValueError("paired-gpu mode requires --device-scope visible")


def _tf_dtype(dtype_name: str) -> tf.DType:
    if dtype_name == "float64":
        return tf.float64
    if dtype_name == "float32":
        return tf.float32
    raise ValueError(f"unsupported dtype: {dtype_name}")


def _configure_tf(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _tf_dtype(args.dtype)
    core_tf.DTYPE = dtype
    streaming_tf.DTYPE = dtype
    annealed_transport_tf.DTYPE = dtype
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
        "tf32_requested": args.tf32_mode,
        "tf32_execution_recorded": tf32_enabled,
    }


def _configure_gpus() -> dict[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    for gpu in physical:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical = tf.config.list_logical_devices("GPU")
    return {
        "physical_gpus": [str(device) for device in physical],
        "logical_gpus": [str(device) for device in logical],
    }


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


def _reset_gpu_memory_stats() -> dict[str, Any]:
    try:
        tf.config.experimental.reset_memory_stats("GPU:0")
        return {"status": "reset"}
    except (AttributeError, ValueError, RuntimeError) as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def _stable_lgssm_fixture(args: argparse.Namespace, particle_count: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed + particle_count + 17 * args.state_dim + 29 * args.obs_dim)
    batch_size = args.batch_size
    state_dim = args.state_dim
    obs_dim = args.obs_dim
    time_steps = args.time_steps
    batch = np.arange(batch_size, dtype=np.float64)
    particle_grid = np.linspace(-1.0, 1.0, particle_count, dtype=np.float64)
    state_grid = np.linspace(-0.5, 0.5, state_dim, dtype=np.float64)
    initial_particles = (
        0.045 * rng.standard_normal((batch_size, particle_count, state_dim))
        + 0.035 * np.sin(particle_grid[None, :, None] * (np.arange(state_dim)[None, None, :] + 1.0))
        + 0.02 * state_grid[None, None, :]
        + 0.0001 * batch[:, None, None]
    )
    diagonal = 0.72 + 0.04 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    transition_matrix = np.zeros((batch_size, state_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        transition_matrix[row] = np.diag(diagonal + 0.00001 * row)
        transition_matrix[row] += 0.003 * np.eye(state_dim, k=1, dtype=np.float64)
        transition_matrix[row] -= 0.002 * np.eye(state_dim, k=-1, dtype=np.float64)
    q_diag = 0.30 + 0.02 * np.linspace(0.0, 1.0, state_dim, dtype=np.float64)
    r_diag = 0.45 + 0.02 * np.linspace(0.0, 1.0, obs_dim, dtype=np.float64)
    transition_covariance = np.tile(np.diag(q_diag)[None, :, :], (batch_size, 1, 1))
    observation_covariance = np.tile(np.diag(r_diag)[None, :, :], (batch_size, 1, 1))
    observation_matrix = np.zeros((batch_size, obs_dim, state_dim), dtype=np.float64)
    for row in range(batch_size):
        for obs_index in range(obs_dim):
            state_index = obs_index % state_dim
            observation_matrix[row, obs_index, state_index] = 0.55
            if state_dim > 1:
                observation_matrix[row, obs_index, (state_index + 1) % state_dim] = 0.05
    observations = 0.03 * np.sin(
        0.23 * (np.arange(time_steps, dtype=np.float64)[:, None] + 1.0)
        * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0)
    )
    observations += 0.01 * np.cos(
        0.17 * (np.arange(time_steps, dtype=np.float64)[:, None] + 1.0)
        * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0)
    )
    time_wave = 0.01 * np.sin(
        0.31 * (np.arange(time_steps, dtype=np.float64)[:, None] + 1.0)
        * (np.arange(state_dim, dtype=np.float64)[None, :] + 1.0)
    )
    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "observation_matrix": observation_matrix,
        "time_wave": time_wave,
        "particle_grid": particle_grid,
        "fixed_resampling_mask": np.ones((batch_size, time_steps), dtype=bool),
    }


def _to_tensors(fixture: dict[str, np.ndarray], dtype: tf.DType) -> dict[str, tf.Tensor]:
    return {name: tf.constant(value, dtype=tf.bool if value.dtype == np.bool_ else dtype) for name, value in fixture.items()}


def _make_pre_flow_step_fn(tensors: dict[str, tf.Tensor]):
    def _pre_flow_step(particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        transitioned = tf.einsum("bnj,bdj->bnd", particles, tensors["transition_matrix"])
        state_dim = particles.shape[-1]
        if state_dim is None:
            raise ValueError("static state dimension required")
        particle_wave = 0.004 * tf.cos(
            0.7
            * tensors["particle_grid"][:, None]
            * (tf.cast(tf.range(state_dim), particles.dtype)[None, :] + 1.0)
        )
        return transitioned + tensors["time_wave"][time_index][None, None, :] + particle_wave[None, :, :]

    return _pre_flow_step


def _make_observation_fn(observation_matrix: tf.Tensor):
    def _observation(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    return _observation


def _make_observation_jacobian_fn(observation_matrix: tf.Tensor):
    def _observation_jacobian(points: tf.Tensor) -> tf.Tensor:
        batch_size = points.shape[0]
        num_particles = points.shape[1]
        if batch_size is None or num_particles is None:
            raise ValueError("effectiveness fixture requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals)))
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], dtype)
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * np.pi, dtype)) + logdet[:, None] + quad)


def _make_transition_log_density(transition_matrix: tf.Tensor, transition_covariance: tf.Tensor, dtype: tf.DType):
    def _transition_log_density(x_next: tf.Tensor, x_prev: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        del _time_index
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance, dtype)

    return _transition_log_density


def _make_observation_log_density(observation_matrix: tf.Tensor, observation_covariance: tf.Tensor, dtype: tf.DType):
    def _observation_log_density(x: tf.Tensor, observation: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        del _time_index
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(predicted - observation[None, None, :], observation_covariance, dtype)

    return _observation_log_density


def _float(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _bool(value: Any) -> bool:
    return bool(tf.reduce_all(tf.cast(value, tf.bool)).numpy())


def _max_diag(step_diagnostics: list[dict[str, Any]], key: str) -> float | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return max(float(value) for value in values) if values else None


def _all_diag(step_diagnostics: list[dict[str, Any]], key: str) -> bool | None:
    values = [diag.get(key) for diag in step_diagnostics if diag.get(key) is not None]
    return all(bool(value) for value in values) if values else None


def _run_filter_loop(tensors: dict[str, tf.Tensor], args: argparse.Namespace, route: str, dtype: tf.DType) -> dict[str, Any]:
    observations = tensors["observations"]
    particles = tensors["initial_particles"]
    fixed_mask = tensors["fixed_resampling_mask"]
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]
    observation_matrix = tensors["observation_matrix"]
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    time_steps = int(observations.shape[0])
    log_weights = uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    means = []
    variances = []
    esses = []
    transport_shapes: list[list[int]] = []
    step_diagnostics: list[dict[str, Any]] = []
    transition_log_density = _make_transition_log_density(transition_matrix, transition_covariance, dtype)
    observation_log_density = _make_observation_log_density(observation_matrix, observation_covariance, dtype)
    pre_flow_step_fn = _make_pre_flow_step_fn(tensors)
    mask_np = fixed_mask.numpy()
    if any(bool(np.any(mask_np[:, t])) and not bool(np.all(mask_np[:, t])) for t in range(time_steps)):
        raise ValueError("effectiveness harness supports all-batch or no-batch transport per step")
    active_count = sum(1 for t in range(time_steps) if bool(np.all(mask_np[:, t])))
    for t in range(time_steps):
        observation = observations[t]
        pre_flow = pre_flow_step_fn(particles, tf.constant(t, dtype=tf.int32))
        flow = streaming_tf.batched_ledh_flow_streaming_particles_tf(
            pre_flow_particles=pre_flow,
            ancestors=particles,
            observation=observation,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
            observation_fn=_make_observation_fn(observation_matrix),
            observation_jacobian_fn=_make_observation_jacobian_fn(observation_matrix),
            observation_residual_fn=_observation_residual,
            particle_chunk_size=args.particle_chunk_size,
        )
        post_flow = flow.post_flow_particles
        corrected_log_weights = (
            log_weights
            + transition_log_density(post_flow, particles, tf.constant(t, dtype=tf.int32))
            + observation_log_density(post_flow, observation, tf.constant(t, dtype=tf.int32))
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood = log_likelihood + incremental
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means.append(mean)
        variances.append(variance)
        esses.append(1.0 / tf.reduce_sum(weights * weights, axis=1))
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))
        if bool(np.all(mask_np[:, t])):
            if route == "streaming":
                transported = batched_annealed_transport_core_tf(
                    post_flow,
                    normalized_log_weights,
                    fixed_mask[:, t],
                    epsilon=args.epsilon,
                    scaling=args.sinkhorn_scaling,
                    convergence_threshold=args.sinkhorn_convergence_threshold,
                    max_iterations=args.sinkhorn_iterations,
                    transport_gradient_mode="raw",
                    transport_plan_mode="streaming",
                    transport_ad_mode="stabilized",
                    row_chunk_size=args.row_chunk_size,
                    col_chunk_size=args.col_chunk_size,
                )
                transport_shapes.append(transported.transport_matrix.shape.as_list())
                step_diagnostics.append(
                    {
                        "max_row_residual": _float(transported.max_row_residual),
                        "max_column_residual": _float(transported.max_column_residual),
                    }
                )
                particles = tf.convert_to_tensor(transported.particles, dtype=dtype)
                log_weights = tf.convert_to_tensor(transported.log_weights, dtype=dtype)
            elif route == "nystrom":
                resample = nystrom_transport_resample_tf(
                    post_flow,
                    normalized_log_weights,
                    rank=args.rank,
                    epsilon=args.epsilon,
                    max_iterations=args.nystrom_max_iterations,
                    convergence_threshold=args.nystrom_convergence_threshold,
                    cholesky_jitter=args.cholesky_jitter,
                    denominator_floor=args.denominator_floor,
                )
                transport_shapes.append(resample.transport_matrix.shape.as_list())
                step_diagnostics.append(dict(resample.diagnostics))
                particles = tf.convert_to_tensor(resample.particles, dtype=dtype)
                log_weights = tf.convert_to_tensor(resample.log_weights, dtype=dtype)
            else:
                raise ValueError(f"unknown route: {route}")
        else:
            particles = post_flow
            log_weights = normalized_log_weights
    filtered_means = tf.stack(means, axis=0)
    filtered_variances = tf.stack(variances, axis=0)
    ess_by_time = tf.stack(esses, axis=0)
    return {
        "log_likelihood": log_likelihood,
        "filtered_means": filtered_means,
        "filtered_variances": filtered_variances,
        "ess_by_time": ess_by_time,
        "final_particles": particles,
        "final_log_weights": log_weights,
        "transport_invocations": active_count,
        "active_resampling_mask_count": active_count,
        "active_resampling_batch_entries": int(np.sum(mask_np)),
        "transport_matrix_shapes": transport_shapes,
        "step_diagnostics": step_diagnostics,
        "output_log_weight_normalization_residual": _float(tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))),
    }


def _route_callable(tensors: dict[str, tf.Tensor], args: argparse.Namespace, route: str, dtype: tf.DType):
    def _call() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        loop = _run_filter_loop(tensors, args, route, dtype)
        return (
            loop["log_likelihood"],
            loop["filtered_means"],
            loop["filtered_variances"],
            loop["ess_by_time"],
            loop["final_particles"],
            loop["final_log_weights"],
        )

    return _call


def _materialize(outputs: tuple[tf.Tensor, ...]) -> list[np.ndarray]:
    return [tensor.numpy() for tensor in outputs]


def _validate_device(outputs: tuple[tf.Tensor, ...], expect: str) -> list[str]:
    devices = [tensor.device for tensor in outputs]
    if expect == "gpu" and not all("GPU" in device.upper() for device in devices):
        raise RuntimeError(f"expected GPU outputs, got {devices}")
    if expect == "cpu" and not all("CPU" in device.upper() for device in devices):
        raise RuntimeError(f"expected CPU outputs, got {devices}")
    return devices


def _summarize_timings(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _preview(array: np.ndarray, limit: int = 8) -> list[float]:
    return [float(v) for v in array.reshape(-1)[: min(limit, array.size)]]


def _memory_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int | None:
    if isinstance(before.get(key), int) and isinstance(after.get(key), int):
        return int(after[key]) - int(before[key])
    return None


def _route_hard_vetoes(route: str, particle_count: int, loop: dict[str, Any], arrays: list[np.ndarray]) -> list[str]:
    vetoes: list[str] = []
    log_likelihood, means, variances, ess, particles, log_weights = arrays
    if not all(np.isfinite(array).all() for array in arrays):
        vetoes.append("nonfinite_output")
    if loop["transport_invocations"] <= 0:
        vetoes.append(f"{route}_transport_invocations_zero")
    if loop["transport_invocations"] != loop["active_resampling_mask_count"]:
        vetoes.append(f"{route}_transport_invocation_count_mismatch")
    if loop["output_log_weight_normalization_residual"] > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        vetoes.append("output_log_weight_normalization_threshold")
    if float(np.min(ess) / particle_count) < ESS_FRACTION_THRESHOLD:
        vetoes.append("ess_fraction_threshold")
    for shape in loop["transport_matrix_shapes"]:
        if shape[-2:] != [0, 0]:
            vetoes.append(f"{route}_transport_matrix_materialized")
            break
    if route == "nystrom":
        diagnostics = loop["step_diagnostics"]
        max_row = _max_diag(diagnostics, "max_row_residual")
        max_col = _max_diag(diagnostics, "max_column_residual")
        if max_row is None or max_row > NYSTROM_RESIDUAL_THRESHOLD:
            vetoes.append("nystrom_row_residual_threshold")
        if max_col is None or max_col > NYSTROM_RESIDUAL_THRESHOLD:
            vetoes.append("nystrom_column_residual_threshold")
        if _all_diag(diagnostics, "finite_factors") is False:
            vetoes.append("nonfinite_nystrom_factors")
        if _all_diag(diagnostics, "finite_particles") is False:
            vetoes.append("nonfinite_nystrom_particles")
        if _all_diag(diagnostics, "transport_matrix_materialized") is not False:
            vetoes.append("nystrom_transport_matrix_materialized")
    del log_likelihood, means, variances, particles, log_weights
    return vetoes


def _run_route_row(args: argparse.Namespace, particle_count: int, route: str) -> dict[str, Any]:
    dtype = _tf_dtype(args.dtype)
    fixture = _stable_lgssm_fixture(args, particle_count)
    tensors = _to_tensors(fixture, dtype)
    callable_route = _route_callable(tensors, args, route, dtype)
    memory_reset = _reset_gpu_memory_stats()
    memory_before = _gpu_memory_info()
    start = time.perf_counter()
    status = "PASS"
    hard_vetoes: list[str] = []
    error: str | None = None
    compile_and_first = None
    warm_timings: list[float] = []
    output_devices: list[str] = []
    loop: dict[str, Any] | None = None
    arrays: list[np.ndarray] | None = None
    try:
        with tf.device(args.device):
            first_outputs = callable_route()
            arrays = _materialize(first_outputs)
            compile_and_first = time.perf_counter() - start
            output_devices = _validate_device(first_outputs, args.expect_device_kind)
            for _ in range(args.warmups):
                _materialize(callable_route())
            for _ in range(args.repeats):
                row_start = time.perf_counter()
                _materialize(callable_route())
                warm_timings.append(time.perf_counter() - row_start)
            loop = _run_filter_loop(tensors, args, route, dtype)
        if loop is not None and arrays is not None:
            hard_vetoes.extend(_route_hard_vetoes(route, particle_count, loop, arrays))
    except Exception as exc:  # noqa: BLE001
        status = "ERROR"
        error = f"{type(exc).__name__}: {exc}"
        hard_vetoes.append("route_execution_error")
    wall_time = time.perf_counter() - start
    memory_after = _gpu_memory_info()
    if wall_time > args.row_timeout_seconds:
        hard_vetoes.append("row_timeout_exceeded_after_completion")
    if hard_vetoes and status == "PASS":
        status = "FAIL"
    result: dict[str, Any] = {
        "route": route,
        "particle_count": particle_count,
        "status": status,
        "error": error,
        "hard_vetoes": hard_vetoes,
        "timeout_policy_seconds": args.row_timeout_seconds,
        "timeout_status": "completed_over_timeout" if wall_time > args.row_timeout_seconds else "completed_within_timeout",
        "wall_time_seconds": wall_time,
        "compile_and_first_call_seconds": compile_and_first,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "warm_call_timings_seconds": warm_timings,
        "warm_call_timing_summary_seconds": _summarize_timings(warm_timings),
        "memory_info_before": memory_before,
        "memory_info_after": memory_after,
        "memory_stats_reset": memory_reset,
        "memory_peak_delta_bytes": _memory_delta(memory_before, memory_after, "peak"),
        "memory_current_delta_bytes": _memory_delta(memory_before, memory_after, "current"),
        "memory_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "output_devices": output_devices,
        "fixture_id": FIXTURE_ID,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": particle_count,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
        },
    }
    if loop is not None and arrays is not None:
        log_likelihood, means, variances, ess, particles, log_weights = arrays
        result.update(
            {
                "transport_invocations": loop["transport_invocations"],
                "active_resampling_mask_count": loop["active_resampling_mask_count"],
                "active_resampling_batch_entries": loop["active_resampling_batch_entries"],
                "transport_matrix_shapes": loop["transport_matrix_shapes"],
                "transport_matrix_materialized": any(shape[-2:] != [0, 0] for shape in loop["transport_matrix_shapes"]),
                "output_log_weight_normalization_residual": loop["output_log_weight_normalization_residual"],
                "finite_output": bool(all(np.isfinite(array).all() for array in arrays)),
                "ess_min": float(np.min(ess)),
                "ess_fraction_min": float(np.min(ess) / particle_count),
                "log_likelihood": log_likelihood.tolist(),
                "state_mean": np.mean(particles, axis=1).tolist(),
                "state_mean_norm": float(np.linalg.norm(np.mean(particles, axis=1))),
                "log_likelihood_preview": _preview(log_likelihood),
                "filtered_means_preview": _preview(means),
                "filtered_variances_preview": _preview(variances),
                "ess_by_time_preview": _preview(ess),
                "final_particles_preview": _preview(particles),
                "final_log_weights_preview": _preview(log_weights),
            }
        )
        if route == "nystrom":
            diagnostics = loop["step_diagnostics"]
            result.update(
                {
                    "max_row_residual": _max_diag(diagnostics, "max_row_residual"),
                    "max_column_residual": _max_diag(diagnostics, "max_column_residual"),
                    "all_finite_factors": _all_diag(diagnostics, "finite_factors"),
                    "all_finite_resampled_particles": _all_diag(diagnostics, "finite_particles"),
                    "iterations_used_max": _max_diag(diagnostics, "iterations_used"),
                    "landmark_indices_by_step": [diag.get("landmark_indices") for diag in diagnostics],
                }
            )
        else:
            result.update(
                {
                    "max_row_residual": _max_diag(loop["step_diagnostics"], "max_row_residual"),
                    "max_column_residual": _max_diag(loop["step_diagnostics"], "max_column_residual"),
                }
            )
    return result


def _paired_comparability(streaming: dict[str, Any], nystrom: dict[str, Any]) -> dict[str, Any]:
    if streaming.get("status") != "PASS" or nystrom.get("status") != "PASS":
        return {"status": "NOT_EVALUATED", "hard_vetoes": ["route_not_passed"]}
    hard_vetoes: list[str] = []
    if streaming.get("shape") != nystrom.get("shape"):
        hard_vetoes.append("shape_mismatch")
    if not streaming.get("finite_output") or not nystrom.get("finite_output"):
        hard_vetoes.append("nonfinite_output")
    if streaming.get("output_log_weight_normalization_residual", float("inf")) > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("streaming_log_weight_normalization")
    if nystrom.get("output_log_weight_normalization_residual", float("inf")) > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("nystrom_log_weight_normalization")
    if streaming.get("ess_fraction_min", 0.0) < ESS_FRACTION_THRESHOLD:
        hard_vetoes.append("streaming_ess_fraction")
    if nystrom.get("ess_fraction_min", 0.0) < ESS_FRACTION_THRESHOLD:
        hard_vetoes.append("nystrom_ess_fraction")
    stream_mean = np.asarray(streaming.get("state_mean", []), dtype=np.float64)
    nystrom_mean = np.asarray(nystrom.get("state_mean", []), dtype=np.float64)
    if stream_mean.shape != nystrom_mean.shape:
        hard_vetoes.append("state_mean_shape_mismatch")
        state_abs = None
        state_rel = None
    else:
        state_abs = float(np.linalg.norm(nystrom_mean - stream_mean))
        state_rel = state_abs / max(float(np.linalg.norm(stream_mean)), 1.0e-12)
        if state_abs > STATE_MEAN_ABSOLUTE_L2_THRESHOLD and state_rel > STATE_MEAN_RELATIVE_L2_THRESHOLD:
            hard_vetoes.append("state_mean_proxy_l2_threshold")
    stream_ll = np.asarray(streaming.get("log_likelihood", []), dtype=np.float64)
    nystrom_ll = np.asarray(nystrom.get("log_likelihood", []), dtype=np.float64)
    if stream_ll.shape != nystrom_ll.shape:
        hard_vetoes.append("log_likelihood_shape_mismatch")
        ll_abs = None
    else:
        ll_abs = float(np.linalg.norm(nystrom_ll - stream_ll))
        if ll_abs > LOG_LIKELIHOOD_ABSOLUTE_L2_THRESHOLD:
            hard_vetoes.append("log_likelihood_proxy_l2_threshold")
    return {
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "state_mean_absolute_l2": state_abs,
        "state_mean_relative_l2": state_rel,
        "log_likelihood_absolute_l2": ll_abs,
        "thresholds": {
            "state_mean_relative_l2": STATE_MEAN_RELATIVE_L2_THRESHOLD,
            "state_mean_absolute_l2": STATE_MEAN_ABSOLUTE_L2_THRESHOLD,
            "log_likelihood_absolute_l2": LOG_LIKELIHOOD_ABSOLUTE_L2_THRESHOLD,
            "log_weight_normalization": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction": ESS_FRACTION_THRESHOLD,
        },
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _paired_effectiveness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_n: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_n.setdefault(int(row["particle_count"]), {})[str(row["route"])] = row
    paired_rows = []
    for particle_count in sorted(by_n):
        routes = by_n[particle_count]
        if "streaming" not in routes or "nystrom" not in routes:
            continue
        comparability = _paired_comparability(routes["streaming"], routes["nystrom"])
        memory_ratio = _ratio(
            routes["streaming"].get("memory_peak_delta_bytes"),
            routes["nystrom"].get("memory_peak_delta_bytes"),
        )
        speed_ratio = _ratio(
            routes["streaming"].get("warm_call_timing_summary_seconds", {}).get("median"),
            routes["nystrom"].get("warm_call_timing_summary_seconds", {}).get("median"),
        )
        paired_rows.append(
            {
                "particle_count": particle_count,
                "comparability": comparability,
                "streaming_to_nystrom_peak_memory_ratio_descriptive": memory_ratio,
                "streaming_to_nystrom_warm_median_time_ratio_descriptive": speed_ratio,
                "descriptive_resource_signal": memory_ratio is not None or speed_ratio is not None,
            }
        )
    return {
        "paired_rows": paired_rows,
        "required_rows_passed": bool(paired_rows) and all(row["comparability"]["status"] == "PASS" for row in paired_rows),
        "claim_status": "VIABLE_PAIRED_PILOT" if paired_rows and all(row["comparability"]["status"] == "PASS" for row in paired_rows) else "NOT_SUPPORTED_CURRENT_EVIDENCE",
        "ranking_status": "NOT_STATISTICALLY_SUPPORTED_SINGLE_PILOT",
    }


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _run_manifest(
    args: argparse.Namespace,
    *,
    started_at: str,
    ended_at: str,
    wall_time_seconds: float,
    tf_metadata: dict[str, Any],
    device_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "git_commit": _git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": _git_output(["git", "status", "--short"]),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        "tensorflow_probability_version": getattr(tfp, "__version__", "unavailable"),
        "device_scope": args.device_scope,
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "trust_context": args.trust_context,
        "selected_physical_gpu": args.selected_physical_gpu,
        "gpu_selection_note": args.gpu_selection_note,
        "selected_physical_gpu_policy": "physical GPU1 preferred if usable, otherwise physical GPU0",
        "same_physical_gpu_required_for_paired_claim": True,
        **tf_metadata,
        **device_metadata,
        "fixture_id": FIXTURE_ID,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "state_dim": args.state_dim,
        "obs_dim": args.obs_dim,
        "particle_counts": list(args.particle_counts),
        "rank": args.rank,
        "epsilon": args.epsilon,
        "row_timeout_seconds": args.row_timeout_seconds,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
        "plan_path": PLAN_PATH,
        "phase_result_path": PILOT_RESULT_PATH,
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    tf.random.set_seed(args.seed)
    tf_metadata = _configure_tf(args)
    device_metadata = _configure_gpus()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for particle_count in args.particle_counts:
        rows.append(_run_route_row(args, particle_count, "streaming"))
        rows.append(_run_route_row(args, particle_count, "nystrom"))
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    wall_time = time.perf_counter() - start
    paired = _paired_effectiveness(rows)
    hard_vetoes = [
        f"N={row.get('particle_count')}:{row.get('route')}:{veto}"
        for row in rows
        for veto in row.get("hard_vetoes", [])
    ]
    for paired_row in paired["paired_rows"]:
        if paired_row["comparability"]["status"] != "PASS":
            for veto in paired_row["comparability"]["hard_vetoes"]:
                hard_vetoes.append(f"N={paired_row['particle_count']}:paired:{veto}")
    if args.mode == "paired-gpu":
        if args.dtype == "float32" and tf_metadata["tf32_execution_recorded"] is not True:
            hard_vetoes.append("tf32_not_recorded_enabled_for_float32")
        if not device_metadata["logical_gpus"]:
            hard_vetoes.append("gpu_device_evidence_missing")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
            hard_vetoes.append("selected_gpu_not_recorded_as_gpu0_or_gpu1")
    manifest = _run_manifest(
        args,
        started_at=started_at,
        ended_at=ended_at,
        wall_time_seconds=wall_time,
        tf_metadata=tf_metadata,
        device_metadata=device_metadata,
    )
    return {
        "status": "PASS" if not hard_vetoes else "FAIL",
        "phase": args.phase_id or ("P01_TINY_CPU" if args.mode == "tiny-cpu" else "P02_PAIRED_GPU_PILOT"),
        "mode": args.mode,
        "algorithm_family": "fixed_rank_nystrom_kernel_sinkhorn_ledh_pfpf_ot_effectiveness_pilot",
        "candidate": "nystrom",
        "comparator": "streaming_tf32",
        "fixture_id": FIXTURE_ID,
        "hard_vetoes": hard_vetoes,
        "paired_effectiveness": paired,
        "thresholds": {
            "log_weight_normalization": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction": ESS_FRACTION_THRESHOLD,
            "nystrom_residual": NYSTROM_RESIDUAL_THRESHOLD,
            "state_mean_relative_l2": STATE_MEAN_RELATIVE_L2_THRESHOLD,
            "state_mean_absolute_l2": STATE_MEAN_ABSOLUTE_L2_THRESHOLD,
            "log_likelihood_absolute_l2": LOG_LIKELIHOOD_ABSOLUTE_L2_THRESHOLD,
            "p02_row_timeout_seconds": P02_ROW_TIMEOUT_SECONDS,
        },
        "rows": rows,
        "run_manifest": manifest,
        "nonclaims": list(NONCLAIMS),
        "inference_status": {
            "hard_veto_screen": "PASS" if not hard_vetoes else "FAIL",
            "statistically_supported_ranking": "NO",
            "descriptive_only_differences": "runtime and memory ratios are descriptive in this single pilot",
            "default_readiness": "NO",
            "next_evidence_needed": "replicated paired ladder with uncertainty analysis before ranking or default claims",
        },
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy())
    return value


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# Nystrom LEDH/PFPF-OT Effectiveness Leaderboard Pilot",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Mode: `{result['mode']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Paired claim status: `{result['paired_effectiveness'].get('claim_status')}`",
        f"- Ranking status: `{result['paired_effectiveness'].get('ranking_status')}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| N | Route | Status | Vetoes | Warm median | Peak memory delta | ESS fraction min | Row residual | Column residual |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["rows"]:
        median = row.get("warm_call_timing_summary_seconds", {}).get("median")
        lines.append(
            "| {n} | `{route}` | `{status}` | `{vetoes}` | `{median}` | `{mem}` | `{ess}` | `{row_res}` | `{col_res}` |".format(
                n=row.get("particle_count"),
                route=row.get("route"),
                status=row.get("status"),
                vetoes=row.get("hard_vetoes"),
                median=median,
                mem=row.get("memory_peak_delta_bytes"),
                ess=row.get("ess_fraction_min"),
                row_res=row.get("max_row_residual"),
                col_res=row.get("max_column_residual"),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Rows",
            "",
            "| N | Comparability | State L2 | Log-likelihood L2 | Memory ratio | Time ratio |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["paired_effectiveness"].get("paired_rows", []):
        comp = row["comparability"]
        lines.append(
            "| {n} | `{status}` | `{state}` | `{ll}` | `{mem}` | `{time}` |".format(
                n=row["particle_count"],
                status=comp["status"],
                state=comp.get("state_mean_absolute_l2"),
                ll=comp.get("log_likelihood_absolute_l2"),
                mem=row["streaming_to_nystrom_peak_memory_ratio_descriptive"],
                time=row["streaming_to_nystrom_warm_median_time_ratio_descriptive"],
            )
        )
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
            f"- CUDA_VISIBLE_DEVICES: `{result['run_manifest']['cuda_visible_devices']}`",
            f"- Selected physical GPU: `{result['run_manifest']['selected_physical_gpu']}`",
            f"- GPU selection note: `{result['run_manifest']['gpu_selection_note']}`",
            f"- TF32 recorded: `{result['run_manifest']['tf32_execution_recorded']}`",
            f"- Device scope: `{result['run_manifest']['device_scope']}`",
            "",
            "## Non-Claims",
            "",
        ]
    )
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    result = build_result(args)
    output = Path(args.output)
    markdown = Path(args.markdown_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(result, markdown, output)
    if not args.quiet:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
