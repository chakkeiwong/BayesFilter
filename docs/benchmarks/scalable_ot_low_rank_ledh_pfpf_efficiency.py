"""Low-rank versus streaming LEDH/PFPF-OT efficiency harness.

This lane-owned harness tests whether the low-rank solver route can reduce
resource proxies for LEDH/PFPF-OT TF32 at large particle counts.  It uses a
common deterministic LGSSM-shaped fixture and existing TensorFlow route
components.  Runtime and memory are evidence only under the governed paired
screen in the matching plan; this file does not establish posterior
correctness, default readiness, dense Sinkhorn equivalence, or broad scalable
OT selection.
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
_PRE.add_argument("--device-scope", choices=("cpu", "visible"), default=None)
_PRE.add_argument("--cuda-visible-devices", default=None)
_PRE_ARGS, _ = _PRE.parse_known_args()
if _PRE_ARGS.cuda_visible_devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _PRE_ARGS.cuda_visible_devices
elif _PRE_ARGS.device_scope == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
from experiments.dpf_implementation.tf_tfp.resampling import (  # noqa: E402
    annealed_transport_tf,
    low_rank_coupling_solver_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling.low_rank_coupling_solver_tf import (  # noqa: E402
    low_rank_coupling_solver_resample_tf,
)


PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-efficiency-validation-"
    "master-program-2026-06-21.md"
)
P01_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-efficiency-validation-"
    "p01-harness-result-2026-06-21.md"
)
P02_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-efficiency-validation-"
    "p02-paired-gpu-result-2026-06-21.md"
)
P03_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-efficiency-validation-"
    "p03-large-n-result-2026-06-21.md"
)

FIXTURE_ID = "ledh_lgssm_efficiency_common_v1"
FACTOR_RESIDUAL_THRESHOLD = 5.0e-3
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-6
ESS_FRACTION_THRESHOLD = 0.01
STATE_MEAN_RELATIVE_L2_THRESHOLD = 0.5
STATE_MEAN_ABSOLUTE_L2_THRESHOLD = 1.0
MEMORY_IMPROVEMENT_FACTOR = 2.0
SPEED_IMPROVEMENT_FACTOR = 1.25
P02_ROW_TIMEOUT_SECONDS = 900
P03_ROW_TIMEOUT_SECONDS = 1200
DEFAULT_PAIRED_LADDER = [1024, 2048, 4096, 8192, 16384, 32768, 50000, 100000]
NONCLAIMS = (
    "bounded resource-proxy efficiency harness only",
    "no posterior correctness claim",
    "no HMC readiness claim",
    "no public API readiness claim",
    "no production/default readiness claim",
    "no dense Sinkhorn equivalence claim",
    "no broad scalable-OT selection claim",
    "no statistical ranking claim",
    "no streaming superiority claim at unpaired large-N rows",
)


def _parse_args() -> argparse.Namespace:
    return _parse_args_impl(None)


def _parse_args_from_list_for_test(argv: list[str]) -> argparse.Namespace:
    return _parse_args_impl(argv)


def _parse_args_impl(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--mode",
        choices=("small", "paired-gpu", "large-n"),
        default="small",
    )
    parser.add_argument("--routes", choices=("both", "streaming", "low-rank"), default="both")
    parser.add_argument("--particle-counts", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--assignment-epsilon", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=1.0e-8)
    parser.add_argument("--max-projection-iterations", type=int, default=240)
    parser.add_argument("--convergence-threshold", type=float, default=1.0e-6)
    parser.add_argument("--denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.5)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=1024)
    parser.add_argument("--col-chunk-size", type=int, default=1024)
    parser.add_argument("--particle-chunk-size", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--row-timeout-seconds", type=int, default=None)
    parser.add_argument("--stop-streaming-after-failure", action="store_true")
    parser.add_argument("--row-subprocess-timeouts", dest="row_subprocess_timeouts", action="store_true", default=None)
    parser.add_argument("--no-row-subprocess-timeouts", dest="row_subprocess_timeouts", action="store_false")
    parser.add_argument("--reuse-existing-row-artifacts", action="store_true")
    parser.add_argument("--compile-streaming", action="store_true")
    parser.add_argument("--compile-low-rank", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", default=None)
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default=None)
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default=None)
    parser.add_argument("--trust-context", default=None)
    parser.add_argument("--phase-id", default=None)
    parser.add_argument("--phase-result-path", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--internal-single-row", action="store_true")
    parser.add_argument("--single-row-route", choices=("streaming", "low_rank"), default=None)
    parser.add_argument("--single-row-particle-count", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    args = parser.parse_args(argv)
    args = _apply_mode_defaults(args)
    _validate_args(args)
    return args


def _apply_mode_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.mode == "small":
        args.particle_counts = args.particle_counts or [32]
        args.batch_size = args.batch_size or 1
        args.time_steps = args.time_steps or 2
        args.state_dim = args.state_dim or 3
        args.obs_dim = args.obs_dim or 2
        args.rank = args.rank or 4
        args.assignment_epsilon = args.assignment_epsilon or 0.0625
        args.dtype = args.dtype or "float64"
        args.device_scope = args.device_scope or "cpu"
        args.device = args.device or "/CPU:0"
        args.expect_device_kind = args.expect_device_kind or "cpu"
        args.tf32_mode = args.tf32_mode or "disabled"
        args.trust_context = args.trust_context or "cpu_hidden_local"
        args.row_timeout_seconds = args.row_timeout_seconds or 60
        if args.row_subprocess_timeouts is None:
            args.row_subprocess_timeouts = False
    elif args.mode == "paired-gpu":
        args.particle_counts = args.particle_counts or list(DEFAULT_PAIRED_LADDER)
        args.batch_size = args.batch_size or 1
        args.time_steps = args.time_steps or 2
        args.state_dim = args.state_dim or 8
        args.obs_dim = args.obs_dim or 6
        args.rank = args.rank or 16
        args.assignment_epsilon = args.assignment_epsilon or 0.015625
        args.dtype = args.dtype or "float32"
        args.device_scope = args.device_scope or "visible"
        args.device = args.device or "/GPU:0"
        args.expect_device_kind = args.expect_device_kind or "gpu"
        args.tf32_mode = args.tf32_mode or "enabled"
        args.trust_context = args.trust_context or "trusted_gpu_required"
        args.row_timeout_seconds = args.row_timeout_seconds or P02_ROW_TIMEOUT_SECONDS
        args.stop_streaming_after_failure = True
        if args.row_subprocess_timeouts is None:
            args.row_subprocess_timeouts = True
    else:
        args.routes = "low-rank" if args.routes == "both" else args.routes
        args.particle_counts = args.particle_counts or [50000, 100000]
        args.batch_size = args.batch_size or 1
        args.time_steps = args.time_steps or 1
        args.state_dim = args.state_dim or 8
        args.obs_dim = args.obs_dim or 6
        args.rank = args.rank or 16
        args.assignment_epsilon = args.assignment_epsilon or 0.015625
        args.dtype = args.dtype or "float32"
        args.device_scope = args.device_scope or "visible"
        args.device = args.device or "/GPU:0"
        args.expect_device_kind = args.expect_device_kind or "gpu"
        args.tf32_mode = args.tf32_mode or "enabled"
        args.trust_context = args.trust_context or "trusted_gpu_required"
        args.row_timeout_seconds = args.row_timeout_seconds or P03_ROW_TIMEOUT_SECONDS
        if args.row_subprocess_timeouts is None:
            args.row_subprocess_timeouts = True
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if any(count <= 1 for count in args.particle_counts):
        raise ValueError("particle-counts must be greater than 1")
    for name in ("batch_size", "time_steps", "state_dim", "obs_dim", "rank"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.assignment_epsilon <= 0.0:
        raise ValueError("assignment-epsilon must be positive")
    if args.alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if args.max_projection_iterations <= 0:
        raise ValueError("max-projection-iterations must be positive")
    if args.convergence_threshold <= 0.0 or args.denominator_floor <= 0.0:
        raise ValueError("solver thresholds/floors must be positive")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.row_timeout_seconds <= 0:
        raise ValueError("row-timeout-seconds must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect GPU outputs")
    if args.mode == "large-n" and args.routes == "streaming":
        raise ValueError("large-n mode is low-rank only unless explicitly redesigned")
    if args.internal_single_row:
        if args.single_row_route is None or args.single_row_particle_count is None:
            raise ValueError("--internal-single-row requires --single-row-route and --single-row-particle-count")
        if args.single_row_particle_count <= 1:
            raise ValueError("--single-row-particle-count must be greater than 1")


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
    low_rank_coupling_solver_tf.DTYPE = dtype
    low_rank_coupling_solver_tf.DEFAULT_DTYPE = dtype
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
        "tf32_hard_gate": args.mode != "small",
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
    rng = np.random.default_rng(args.seed + particle_count)
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

    observations = np.zeros((time_steps, obs_dim), dtype=np.float64)
    time_grid = np.arange(time_steps, dtype=np.float64)
    observations += 0.03 * np.sin(
        0.23 * (time_grid[:, None] + 1.0) * (np.arange(obs_dim, dtype=np.float64)[None, :] + 1.0)
    )
    observations += 0.01 * np.cos(
        0.17 * (time_grid[:, None] + 1.0) * (np.arange(obs_dim, dtype=np.float64)[None, :] + 2.0)
    )

    time_wave = 0.01 * np.sin(0.31 * (time_grid[:, None] + 1.0) * (np.arange(state_dim) + 1.0))
    fixed_resampling_mask = np.ones((batch_size, time_steps), dtype=bool)
    return {
        "observations": observations,
        "initial_particles": initial_particles,
        "transition_matrix": transition_matrix,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "observation_matrix": observation_matrix,
        "time_wave": time_wave,
        "particle_grid": particle_grid,
        "fixed_resampling_mask": fixed_resampling_mask,
    }


def _to_tensors(fixture: dict[str, np.ndarray], dtype: tf.DType) -> dict[str, tf.Tensor]:
    tensors: dict[str, tf.Tensor] = {}
    for name, value in fixture.items():
        tensors[name] = tf.constant(value, dtype=tf.bool if value.dtype == np.bool_ else dtype)
    return tensors


def _make_pre_flow_step_fn(tensors: dict[str, tf.Tensor]):
    def _pre_flow_step(particles: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        transitioned = tf.einsum("bnj,bdj->bnd", particles, tensors["transition_matrix"])
        state_dim = particles.shape[-1]
        if state_dim is None:
            raise ValueError("static state dimension required")
        particle_wave = 0.004 * tf.cos(
            0.7 * tensors["particle_grid"][:, None] * (tf.cast(tf.range(state_dim), particles.dtype)[None, :] + 1.0)
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
            raise ValueError("efficiency fixture requires static batch and particle dimensions")
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    return _observation_jacobian


def _observation_residual(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
    return observation[None, None, :] - h_ref


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals))
    )
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(residuals.shape[-1], dtype)
    return -0.5 * (
        dim * tf.math.log(tf.constant(2.0 * np.pi, dtype))
        + logdet[:, None]
        + quad
    )


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
    transport_matrix_shapes: list[list[int]] = []
    step_diagnostics: list[dict[str, Any]] = []
    transport_invocations = 0
    active_count = 0
    mask_np = fixed_mask.numpy()
    if any(bool(np.any(mask_np[:, t])) and not bool(np.all(mask_np[:, t])) for t in range(time_steps)):
        raise ValueError("efficiency harness supports all-batch or no-batch transport per step")
    active_count = sum(1 for t in range(time_steps) if bool(np.all(mask_np[:, t])))

    transition_log_density = _make_transition_log_density(transition_matrix, transition_covariance, dtype)
    observation_log_density = _make_observation_log_density(observation_matrix, observation_covariance, dtype)
    pre_flow_step_fn = _make_pre_flow_step_fn(tensors)

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
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means.append(mean)
        variances.append(variance)
        esses.append(ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))

        if bool(np.all(mask_np[:, t])):
            transport_invocations += 1
            if route == "streaming":
                transported = batched_annealed_transport_core_tf(
                    post_flow,
                    normalized_log_weights,
                    fixed_mask[:, t],
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
                step_diagnostics.append(
                    {
                        "max_row_residual": _float(transported.max_row_residual),
                        "max_column_residual": _float(transported.max_column_residual),
                    }
                )
                particles = tf.convert_to_tensor(transported.particles, dtype=dtype)
                log_weights = tf.convert_to_tensor(transported.log_weights, dtype=dtype)
            elif route == "low_rank":
                resample = low_rank_coupling_solver_resample_tf(
                    post_flow,
                    normalized_log_weights,
                    rank=args.rank,
                    assignment_epsilon=args.assignment_epsilon,
                    alpha=args.alpha,
                    max_projection_iterations=args.max_projection_iterations,
                    convergence_threshold=args.convergence_threshold,
                    denominator_floor=args.denominator_floor,
                )
                transport_matrix_shapes.append(resample.transport_matrix.shape.as_list())
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
    final_log_weight_normalization = _float(tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1))))
    return {
        "log_likelihood": log_likelihood,
        "filtered_means": filtered_means,
        "filtered_variances": filtered_variances,
        "ess_by_time": ess_by_time,
        "final_particles": particles,
        "final_log_weights": log_weights,
        "transport_invocations": transport_invocations,
        "active_resampling_mask_count": active_count,
        "active_resampling_batch_entries": int(np.sum(mask_np)),
        "transport_matrix_shapes": transport_matrix_shapes,
        "step_diagnostics": step_diagnostics,
        "output_log_weight_normalization_residual": final_log_weight_normalization,
    }


def _make_route_callable(tensors: dict[str, tf.Tensor], args: argparse.Namespace, route: str, dtype: tf.DType):
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

    should_compile = (route == "streaming" and args.compile_streaming) or (route == "low_rank" and args.compile_low_rank)
    if should_compile:
        return tf.function(_call, jit_compile=True, reduce_retracing=True)
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


def _route_hard_vetoes(
    route: str,
    particle_count: int,
    loop: dict[str, Any],
    arrays: list[np.ndarray],
) -> list[str]:
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
    if route == "low_rank":
        diagnostics = loop["step_diagnostics"]
        max_factor = _max_diag(diagnostics, "max_factor_marginal_residual")
        if max_factor is None:
            vetoes.append("missing_factor_diagnostics")
        elif max_factor > FACTOR_RESIDUAL_THRESHOLD:
            vetoes.append("factor_marginal_residual_threshold")
        if _all_diag(diagnostics, "finite_factors") is False:
            vetoes.append("nonfinite_factors")
        if _all_diag(diagnostics, "finite_particles") is False:
            vetoes.append("nonfinite_low_rank_particles")
        if _all_diag(diagnostics, "nonnegative_factors") is False:
            vetoes.append("negative_low_rank_factor")
        if _all_diag(diagnostics, "positive_g") is False:
            vetoes.append("nonpositive_low_rank_g")
    del log_likelihood, means, variances, particles, log_weights
    return vetoes


def _run_route_row(
    args: argparse.Namespace,
    particle_count: int,
    route: str,
    *,
    force_skipped: str | None = None,
) -> dict[str, Any]:
    if force_skipped is not None:
        return {
            "route": route,
            "particle_count": particle_count,
            "status": "SKIPPED",
            "skip_reason": force_skipped,
            "hard_vetoes": [],
        }

    dtype = _tf_dtype(args.dtype)
    fixture = _stable_lgssm_fixture(args, particle_count)
    tensors = _to_tensors(fixture, dtype)
    callable_route = _make_route_callable(tensors, args, route, dtype)
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
        "memory_peak_delta_bytes": _memory_peak_delta(memory_before, memory_after),
        "memory_current_delta_bytes": _memory_current_delta(memory_before, memory_after),
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
                "transport_matrix_materialized": any(
                    shape[-2:] != [0, 0] for shape in loop["transport_matrix_shapes"]
                ),
                "output_log_weight_normalization_residual": loop["output_log_weight_normalization_residual"],
                "finite_output": bool(all(np.isfinite(array).all() for array in arrays)),
                "ess_min": float(np.min(ess)),
                "ess_fraction_min": float(np.min(ess) / particle_count),
                "log_likelihood_preview": _preview(log_likelihood),
                "filtered_means_preview": _preview(means),
                "filtered_variances_preview": _preview(variances),
                "ess_by_time_preview": _preview(ess),
                "final_particles_preview": _preview(particles),
                "final_log_weights_preview": _preview(log_weights),
                "state_mean": np.mean(particles, axis=1).tolist(),
                "state_mean_norm": float(np.linalg.norm(np.mean(particles, axis=1))),
            }
        )
        if route == "low_rank":
            diagnostics = loop["step_diagnostics"]
            result.update(
                {
                    "low_rank_resampling_invocations": loop["transport_invocations"],
                    "max_factor_marginal_residual": _max_diag(diagnostics, "max_factor_marginal_residual"),
                    "max_induced_row_residual": _max_diag(diagnostics, "max_induced_row_residual"),
                    "max_induced_column_residual": _max_diag(diagnostics, "max_induced_column_residual"),
                    "all_finite_factors": _all_diag(diagnostics, "finite_factors"),
                    "all_finite_particles": _all_diag(diagnostics, "finite_particles"),
                    "all_nonnegative_factors": _all_diag(diagnostics, "nonnegative_factors"),
                    "all_positive_g": _all_diag(diagnostics, "positive_g"),
                    "projection_iterations_used_max": _max_diag(diagnostics, "projection_iterations_used"),
                }
            )
    return result


def _run_route_row_subprocess(
    args: argparse.Namespace,
    particle_count: int,
    route: str,
    *,
    force_skipped: str | None = None,
) -> dict[str, Any]:
    if force_skipped is not None:
        return _run_route_row(args, particle_count, route, force_skipped=force_skipped)

    row_json = Path(args.output).with_name(
        f"{Path(args.output).stem}-row-{route}-n{particle_count}.json"
    )
    row_md = Path(args.output).with_name(
        f"{Path(args.output).stem}-row-{route}-n{particle_count}.md"
    )
    if args.reuse_existing_row_artifacts:
        if row_json.exists():
            try:
                payload = json.loads(row_json.read_text(encoding="utf-8"))
                row = payload["rows"][0]
                row["row_subprocess"] = {
                    "status": "reused_existing",
                    "json": str(row_json),
                    "markdown": str(row_md),
                }
                return row
            except (OSError, json.JSONDecodeError, KeyError, IndexError) as exc:
                return {
                    "route": route,
                    "particle_count": particle_count,
                    "status": "ERROR",
                    "error": f"reuse_existing_row_artifact_error:{type(exc).__name__}: {exc}",
                    "hard_vetoes": ["row_artifact_reuse_error"],
                    "timeout_policy_seconds": args.row_timeout_seconds,
                    "timeout_status": "reuse_error",
                    "fixture_id": FIXTURE_ID,
                }
        return {
            "route": route,
            "particle_count": particle_count,
            "status": "TIMEOUT" if args.mode == "paired-gpu" and route == "streaming" else "MISSING",
            "error": "reconstructed_missing_streaming_timeout"
            if args.mode == "paired-gpu" and route == "streaming"
            else "reuse_existing_row_artifact_missing",
            "hard_vetoes": ["row_timeout"]
            if args.mode == "paired-gpu" and route == "streaming"
            else ["row_artifact_missing"],
            "timeout_policy_seconds": args.row_timeout_seconds,
            "timeout_status": "timeout_enforced_reconstructed"
            if args.mode == "paired-gpu" and route == "streaming"
            else "reuse_missing",
            "fixture_id": FIXTURE_ID,
            "shape": {
                "batch_size": args.batch_size,
                "time_steps": args.time_steps,
                "num_particles": particle_count,
                "state_dim": args.state_dim,
                "obs_dim": args.obs_dim,
            },
        }
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        args.mode,
        "--internal-single-row",
        "--single-row-route",
        route,
        "--single-row-particle-count",
        str(particle_count),
        "--routes",
        route.replace("_", "-"),
        "--particle-counts",
        str(particle_count),
        "--batch-size",
        str(args.batch_size),
        "--time-steps",
        str(args.time_steps),
        "--state-dim",
        str(args.state_dim),
        "--obs-dim",
        str(args.obs_dim),
        "--rank",
        str(args.rank),
        "--assignment-epsilon",
        str(args.assignment_epsilon),
        "--alpha",
        str(args.alpha),
        "--max-projection-iterations",
        str(args.max_projection_iterations),
        "--convergence-threshold",
        str(args.convergence_threshold),
        "--denominator-floor",
        str(args.denominator_floor),
        "--sinkhorn-iterations",
        str(args.sinkhorn_iterations),
        "--sinkhorn-epsilon",
        str(args.sinkhorn_epsilon),
        "--annealed-scaling",
        str(args.annealed_scaling),
        "--annealed-convergence-threshold",
        str(args.annealed_convergence_threshold),
        "--row-chunk-size",
        str(args.row_chunk_size),
        "--col-chunk-size",
        str(args.col_chunk_size),
        "--particle-chunk-size",
        str(args.particle_chunk_size),
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--row-timeout-seconds",
        str(args.row_timeout_seconds),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--device-scope",
        args.device_scope,
        "--expect-device-kind",
        args.expect_device_kind,
        "--tf32-mode",
        args.tf32_mode,
        "--trust-context",
        args.trust_context,
        "--phase-id",
        f"{args.phase_id or args.mode}:row:{route}:N={particle_count}",
        "--phase-result-path",
        _phase_result_path(args),
        "--quiet",
        "--output",
        str(row_json),
        "--markdown-output",
        str(row_md),
    ]
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.compile_streaming:
        cmd.append("--compile-streaming")
    if args.compile_low_rank:
        cmd.append("--compile-low-rank")

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    elif args.device_scope == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.row_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return _timeout_row(args, particle_count, route, row_json, started, exc)

    if row_json.exists():
        try:
            payload = json.loads(row_json.read_text(encoding="utf-8"))
            row = payload["rows"][0]
            row["row_subprocess"] = {
                "status": "completed",
                "returncode": completed.returncode,
                "json": str(row_json),
                "markdown": str(row_md),
                "stderr_tail": completed.stderr[-2000:],
            }
            if completed.returncode != 0 and row.get("status") == "PASS":
                row["status"] = "ERROR"
                row.setdefault("hard_vetoes", []).append("row_subprocess_nonzero_exit")
            return row
        except (OSError, json.JSONDecodeError, KeyError, IndexError) as exc:
            return _subprocess_error_row(args, particle_count, route, row_json, started, completed, exc)

    return _subprocess_error_row(args, particle_count, route, row_json, started, completed, None)


def _timeout_row(
    args: argparse.Namespace,
    particle_count: int,
    route: str,
    row_json: Path,
    started: float,
    exc: subprocess.TimeoutExpired,
) -> dict[str, Any]:
    row = {
        "route": route,
        "particle_count": particle_count,
        "status": "TIMEOUT",
        "error": f"TimeoutExpired: exceeded {args.row_timeout_seconds}s",
        "hard_vetoes": ["row_timeout"],
        "timeout_policy_seconds": args.row_timeout_seconds,
        "timeout_status": "timeout_enforced",
        "wall_time_seconds": time.perf_counter() - started,
        "row_subprocess": {
            "status": "timeout",
            "json": str(row_json),
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        },
        "fixture_id": FIXTURE_ID,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": particle_count,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
        },
    }
    _write_timeout_row_artifact(args, row, row_json)
    return row


def _write_timeout_row_artifact(args: argparse.Namespace, row: dict[str, Any], row_json: Path) -> None:
    row_md = row_json.with_suffix(".md")
    payload = {
        "status": "FAIL",
        "phase": f"{args.phase_id or args.mode}:row:{row['route']}:N={row['particle_count']}",
        "mode": args.mode,
        "algorithm_family": "low_rank_solver_route_ledh_pfpf_ot_efficiency",
        "artifact_role": "parent_enforced_row_timeout_sidecar",
        "hard_vetoes": list(row.get("hard_vetoes", [])),
        "rows": [row],
        "run_manifest": {
            "command": " ".join(sys.argv),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_scope": args.device_scope,
            "device": args.device,
            "expect_device_kind": args.expect_device_kind,
            "trust_context": args.trust_context,
            "row_timeout_seconds": args.row_timeout_seconds,
            "row_subprocess_timeouts": bool(args.row_subprocess_timeouts),
            "plan_path": PLAN_PATH,
            "phase_result_path": _phase_result_path(args),
            "fixture_id": FIXTURE_ID,
        },
        "nonclaims": list(NONCLAIMS),
    }
    row_json.parent.mkdir(parents=True, exist_ok=True)
    row_json.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    row_md.write_text(
        "\n".join(
            [
                "# Low-Rank LEDH/PFPF-OT Row Timeout Sidecar",
                "",
                f"- Status: `{row['status']}`",
                f"- Route: `{row['route']}`",
                f"- N: `{row['particle_count']}`",
                f"- Timeout seconds: `{row['timeout_policy_seconds']}`",
                f"- Timeout status: `{row['timeout_status']}`",
                f"- Hard vetoes: `{row['hard_vetoes']}`",
                f"- JSON artifact: `{row_json}`",
                "",
                "This artifact was written by the parent row-timeout handler after",
                "the launched route subprocess exceeded the fixed timeout.",
                "",
                "## Non-Claims",
                "",
                *[f"- {claim}" for claim in NONCLAIMS],
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _subprocess_error_row(
    args: argparse.Namespace,
    particle_count: int,
    route: str,
    row_json: Path,
    started: float,
    completed: subprocess.CompletedProcess[str],
    exc: Exception | None,
) -> dict[str, Any]:
    reason = f"{type(exc).__name__}: {exc}" if exc is not None else "missing_row_json"
    return {
        "route": route,
        "particle_count": particle_count,
        "status": "ERROR",
        "error": reason,
        "hard_vetoes": ["row_subprocess_error"],
        "timeout_policy_seconds": args.row_timeout_seconds,
        "timeout_status": "completed_with_error",
        "wall_time_seconds": time.perf_counter() - started,
        "row_subprocess": {
            "status": "error",
            "returncode": completed.returncode,
            "json": str(row_json),
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
        },
        "fixture_id": FIXTURE_ID,
        "shape": {
            "batch_size": args.batch_size,
            "time_steps": args.time_steps,
            "num_particles": particle_count,
            "state_dim": args.state_dim,
            "obs_dim": args.obs_dim,
        },
    }


def _memory_peak_delta(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    if isinstance(before.get("peak"), int) and isinstance(after.get("peak"), int):
        return int(after["peak"]) - int(before["peak"])
    return None


def _memory_current_delta(before: dict[str, Any], after: dict[str, Any]) -> int | None:
    if isinstance(before.get("current"), int) and isinstance(after.get("current"), int):
        return int(after["current"]) - int(before["current"])
    return None


def _paired_comparability(streaming: dict[str, Any], low_rank: dict[str, Any]) -> dict[str, Any]:
    if streaming.get("status") != "PASS" or low_rank.get("status") != "PASS":
        return {
            "status": "NOT_EVALUATED",
            "hard_vetoes": ["route_not_passed"],
        }
    hard_vetoes: list[str] = []
    stream_shape = streaming.get("shape")
    low_rank_shape = low_rank.get("shape")
    if stream_shape != low_rank_shape:
        hard_vetoes.append("shape_mismatch")
    if not streaming.get("finite_output") or not low_rank.get("finite_output"):
        hard_vetoes.append("nonfinite_output")
    if streaming.get("output_log_weight_normalization_residual", float("inf")) > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("streaming_log_weight_normalization")
    if low_rank.get("output_log_weight_normalization_residual", float("inf")) > LOG_WEIGHT_NORMALIZATION_THRESHOLD:
        hard_vetoes.append("low_rank_log_weight_normalization")
    if streaming.get("ess_fraction_min", 0.0) < ESS_FRACTION_THRESHOLD:
        hard_vetoes.append("streaming_ess_fraction")
    if low_rank.get("ess_fraction_min", 0.0) < ESS_FRACTION_THRESHOLD:
        hard_vetoes.append("low_rank_ess_fraction")
    stream_mean = np.asarray(streaming.get("state_mean", []), dtype=np.float64)
    low_rank_mean = np.asarray(low_rank.get("state_mean", []), dtype=np.float64)
    if stream_mean.shape != low_rank_mean.shape:
        hard_vetoes.append("state_mean_shape_mismatch")
        absolute_l2 = None
        relative_l2 = None
    else:
        diff = low_rank_mean - stream_mean
        absolute_l2 = float(np.linalg.norm(diff))
        denom = max(float(np.linalg.norm(stream_mean)), 1.0e-12)
        relative_l2 = absolute_l2 / denom
        if absolute_l2 > STATE_MEAN_ABSOLUTE_L2_THRESHOLD and relative_l2 > STATE_MEAN_RELATIVE_L2_THRESHOLD:
            hard_vetoes.append("state_mean_proxy_l2_threshold")
    return {
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "state_mean_absolute_l2": absolute_l2,
        "state_mean_relative_l2": relative_l2,
        "thresholds": {
            "log_weight_normalization": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction_min": ESS_FRACTION_THRESHOLD,
            "state_mean_relative_l2": STATE_MEAN_RELATIVE_L2_THRESHOLD,
            "state_mean_absolute_l2": STATE_MEAN_ABSOLUTE_L2_THRESHOLD,
        },
    }


def _paired_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_n: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "SKIPPED":
            continue
        by_n.setdefault(int(row["particle_count"]), {})[str(row["route"])] = row

    paired_rows = []
    envelope_rows = []
    for particle_count in sorted(by_n):
        routes = by_n[particle_count]
        if "streaming" not in routes or "low_rank" not in routes:
            continue
        comparability = _paired_comparability(routes["streaming"], routes["low_rank"])
        memory_ratio = _ratio(
            routes["streaming"].get("memory_peak_delta_bytes"),
            routes["low_rank"].get("memory_peak_delta_bytes"),
        )
        speed_ratio = _ratio(
            routes["streaming"].get("warm_call_timing_summary_seconds", {}).get("median"),
            routes["low_rank"].get("warm_call_timing_summary_seconds", {}).get("median"),
        )
        paired_rows.append(
            {
                "particle_count": particle_count,
                "comparability": comparability,
                "memory_improvement_ratio": memory_ratio,
                "speed_improvement_ratio": speed_ratio,
                "memory_screen_pass": memory_ratio is not None and memory_ratio >= MEMORY_IMPROVEMENT_FACTOR,
                "speed_screen_pass": speed_ratio is not None and speed_ratio >= SPEED_IMPROVEMENT_FACTOR,
                "resource_proxy_screen_pass": comparability["status"] == "PASS"
                and (
                    (memory_ratio is not None and memory_ratio >= MEMORY_IMPROVEMENT_FACTOR)
                    or (speed_ratio is not None and speed_ratio >= SPEED_IMPROVEMENT_FACTOR)
                ),
            }
        )
        streaming = routes["streaming"]
        low_rank = routes["low_rank"]
        if (
            streaming.get("status") in {"TIMEOUT", "ERROR"}
            and low_rank.get("status") == "PASS"
            and not low_rank.get("hard_vetoes")
        ):
            envelope_rows.append(
                {
                    "particle_count": particle_count,
                    "streaming_status": streaming.get("status"),
                    "streaming_hard_vetoes": streaming.get("hard_vetoes", []),
                    "low_rank_status": low_rank.get("status"),
                    "low_rank_hard_vetoes": low_rank.get("hard_vetoes", []),
                    "support": "executable_envelope_for_that_row_only",
                }
            )
    adjacent_support = _adjacent_support(paired_rows)
    return {
        "paired_rows": paired_rows,
        "envelope_rows": envelope_rows,
        "adjacent_resource_proxy_support": adjacent_support,
        "executable_envelope_support": bool(envelope_rows),
        "claim_status": "SUPPORTED_BOUNDED" if adjacent_support else "NOT_SUPPORTED_CURRENT_EVIDENCE",
        "thresholds": {
            "memory_improvement_factor": MEMORY_IMPROVEMENT_FACTOR,
            "speed_improvement_factor": SPEED_IMPROVEMENT_FACTOR,
        },
    }


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _adjacent_support(paired_rows: list[dict[str, Any]]) -> bool:
    passing = {row["particle_count"] for row in paired_rows if row.get("resource_proxy_screen_pass")}
    ladder = DEFAULT_PAIRED_LADDER
    return any(left in passing and right in passing for left, right in zip(ladder[:-1], ladder[1:]))


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _phase_result_path(args: argparse.Namespace) -> str:
    if args.phase_result_path:
        return args.phase_result_path
    return {
        "small": P01_RESULT_PATH,
        "paired-gpu": P02_RESULT_PATH,
        "large-n": P03_RESULT_PATH,
    }[args.mode]


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
        "selected_physical_gpu_policy": "GPU1 preferred; GPU0 only if GPU1 busy/unavailable/unsuitable",
        "same_physical_gpu_required_for_paired_claim": True,
        **tf_metadata,
        **device_metadata,
        "fixture_id": FIXTURE_ID,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "state_dim": args.state_dim,
        "obs_dim": args.obs_dim,
        "particle_counts": list(args.particle_counts),
        "routes": args.routes,
        "rank": args.rank,
        "assignment_epsilon": args.assignment_epsilon,
        "row_timeout_seconds": args.row_timeout_seconds,
        "row_subprocess_timeouts": bool(args.row_subprocess_timeouts),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
        "plan_path": PLAN_PATH,
        "phase_result_path": _phase_result_path(args),
    }


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    tf.random.set_seed(args.seed)
    tf_metadata = _configure_tf(args)
    device_metadata = _configure_gpus()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    if args.internal_single_row:
        rows.append(
            _run_route_row(
                args,
                int(args.single_row_particle_count),
                str(args.single_row_route),
            )
        )
    else:
        streaming_stopped = False
        for particle_count in args.particle_counts:
            route_names = ["streaming", "low_rank"] if args.routes == "both" else [args.routes.replace("-", "_")]
            for route in route_names:
                skip_reason = None
                if route == "streaming" and streaming_stopped:
                    skip_reason = "streaming_stopped_after_first_failure"
                if args.row_subprocess_timeouts:
                    row = _run_route_row_subprocess(args, particle_count, route, force_skipped=skip_reason)
                else:
                    row = _run_route_row(args, particle_count, route, force_skipped=skip_reason)
                rows.append(row)
                if (
                    args.stop_streaming_after_failure
                    and route == "streaming"
                    and row.get("status") not in {"PASS", "SKIPPED"}
                ):
                    streaming_stopped = True
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    wall_time = time.perf_counter() - start
    route_vetoes = []
    for row in rows:
        for veto in row.get("hard_vetoes", []):
            if (
                args.mode == "paired-gpu"
                and row.get("route") == "streaming"
                and veto == "row_timeout"
            ):
                continue
            route_vetoes.append(f"N={row.get('particle_count')}:{row.get('route')}:{veto}")
    paired = _paired_efficiency(rows) if args.routes == "both" else {
        "paired_rows": [],
        "claim_status": "NOT_EVALUATED_LOW_RANK_ONLY",
    }
    hard_vetoes = list(route_vetoes)
    if args.mode == "small" and paired.get("paired_rows"):
        if any(row["comparability"]["status"] != "PASS" for row in paired["paired_rows"]):
            hard_vetoes.append("small_output_comparability_failed")
    if args.mode != "small" and args.tf32_mode == "enabled" and tf_metadata["tf32_execution_recorded"] is not True:
        hard_vetoes.append("tf32_not_enabled_for_tf32_claim")
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
        "phase": args.phase_id
        or {
            "small": "LOW_RANK_LEDH_EFFICIENCY_P01",
            "paired-gpu": "LOW_RANK_LEDH_EFFICIENCY_P02",
            "large-n": "LOW_RANK_LEDH_EFFICIENCY_P03",
        }[args.mode],
        "mode": args.mode,
        "algorithm_family": "low_rank_solver_route_ledh_pfpf_ot_efficiency",
        "algorithm_under_test": "LEDH/PFPF-OT with streaming transport versus P = Q diag(1/g) R^T lazy low-rank resampling",
        "candidate_classification": {
            "directly_anchored_components": [
                "Q diag(1/g) R^T factor form",
                "lazy low-rank apply",
                "factor marginal diagnostics",
            ],
            "fixed_hmc_adaptation": [
                "deterministic rank",
                "deterministic assignment epsilon",
                "fixed projection schedule",
            ],
            "extension_or_invention": [
                "cost-nudged assignment kernel",
                "diagnostic simplified solver objective/stabilization",
            ],
        },
        "thresholds": {
            "factor_residual": FACTOR_RESIDUAL_THRESHOLD,
            "log_weight_normalization": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
            "ess_fraction": ESS_FRACTION_THRESHOLD,
            "state_mean_relative_l2": STATE_MEAN_RELATIVE_L2_THRESHOLD,
            "state_mean_absolute_l2": STATE_MEAN_ABSOLUTE_L2_THRESHOLD,
            "memory_improvement_factor": MEMORY_IMPROVEMENT_FACTOR,
            "speed_improvement_factor": SPEED_IMPROVEMENT_FACTOR,
            "p02_row_timeout_seconds": P02_ROW_TIMEOUT_SECONDS,
            "p03_row_timeout_seconds": P03_ROW_TIMEOUT_SECONDS,
        },
        "hard_vetoes": hard_vetoes,
        "paired_efficiency": paired,
        "rows": rows,
        "run_manifest": manifest,
        "nonclaims": list(NONCLAIMS),
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
        "# Low-Rank LEDH/PFPF-OT Efficiency Harness",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Mode: `{result['mode']}`",
        f"- Algorithm: `{result['algorithm_under_test']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
        f"- Paired claim status: `{result['paired_efficiency'].get('claim_status')}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| N | Route | Status | Vetoes | Warm median | Peak memory delta | ESS fraction min | Invocations |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["rows"]:
        median = row.get("warm_call_timing_summary_seconds", {}).get("median")
        lines.append(
            "| {n} | `{route}` | `{status}` | `{vetoes}` | `{median}` | `{mem}` | `{ess}` | `{inv}` |".format(
                n=row.get("particle_count"),
                route=row.get("route"),
                status=row.get("status"),
                vetoes=row.get("hard_vetoes"),
                median=median,
                mem=row.get("memory_peak_delta_bytes"),
                ess=row.get("ess_fraction_min"),
                inv=row.get("transport_invocations"),
            )
        )
    lines.extend(
        [
            "",
            "## Paired Rows",
            "",
            "| N | Comparability | Memory ratio | Speed ratio | Resource screen |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in result["paired_efficiency"].get("paired_rows", []):
        lines.append(
            "| {n} | `{comp}` | `{mem}` | `{speed}` | `{screen}` |".format(
                n=row["particle_count"],
                comp=row["comparability"]["status"],
                mem=row["memory_improvement_ratio"],
                speed=row["speed_improvement_ratio"],
                screen=row["resource_proxy_screen_pass"],
            )
        )
    lines.extend(
        [
            "",
            "## Executable Envelope Rows",
            "",
            "| N | Streaming status | Low-rank status | Support |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for row in result["paired_efficiency"].get("envelope_rows", []):
        lines.append(
            "| {n} | `{streaming}` | `{low_rank}` | `{support}` |".format(
                n=row["particle_count"],
                streaming=row["streaming_status"],
                low_rank=row["low_rank_status"],
                support=row["support"],
            )
        )
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            f"- Git commit: `{result['run_manifest']['git_commit']}`",
            f"- Device scope: `{result['run_manifest']['device_scope']}`",
            f"- CUDA_VISIBLE_DEVICES: `{result['run_manifest']['cuda_visible_devices']}`",
            f"- TF32 recorded: `{result['run_manifest']['tf32_execution_recorded']}`",
            f"- Fixture: `{result['run_manifest']['fixture_id']}`",
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


if __name__ == "__main__":
    main()
