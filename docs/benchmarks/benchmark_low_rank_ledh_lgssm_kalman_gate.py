"""LGSSM exact-Kalman gate for low-rank LEDH-PFPF-OT.

This P01A/P01B harness creates deterministic TensorFlow LGSSM fixtures, runs
the existing streaming and low-rank LEDH-PFPF-OT routes, and compares filtering
summaries with an exact Kalman reference.  Local CPU-hidden runs are command
shape and harness checks only.  GPU/TF32/XLA claims require the separately
approved trusted-GPU P01B runtime.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple


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
import tensorflow_probability as tfp  # noqa: E402

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


tfd = tfp.distributions

PLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-model-suite-promotion-"
    "master-program-2026-06-24.md"
)
P01_SUBPLAN_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-model-suite-promotion-"
    "p01-lgssm-kalman-subplan-2026-06-24.md"
)
P01A_RESULT_PATH = (
    "docs/plans/bayesfilter-dpf-ledh-pfpf-ot-low-rank-model-suite-promotion-"
    "p01-lgssm-kalman-implementation-result-2026-06-24.md"
)

FACTOR_RESIDUAL_THRESHOLD = 5.0e-3
LOG_WEIGHT_NORMALIZATION_THRESHOLD = 1.0e-5
ESS_FRACTION_MIN_THRESHOLD = 0.005
PINNED_CASES: dict[str, dict[str, Any]] = {
    "lgssm_small_exact_ref": {
        "role": "smoke/exact-reference sanity",
        "state_dim": 4,
        "obs_dim": 3,
        "time_steps": 12,
        "num_particles": 1024,
        "seeds": (91001, 91002, 91003),
        "mean_rmse_max": 0.25,
        "variance_rmse_max": 0.35,
        "loglik_abs_delta_max": 12.0,
    },
    "lgssm_medium_exact_ref": {
        "role": "promotion quality screen",
        "state_dim": 16,
        "obs_dim": 8,
        "time_steps": 20,
        "num_particles": 2048,
        "seeds": (91011, 91012, 91013),
        "mean_rmse_max": 0.35,
        "variance_rmse_max": 0.50,
        "loglik_abs_delta_max": 25.0,
    },
    "lgssm_informative_obs_stress": {
        "role": "high-information stress",
        "state_dim": 16,
        "obs_dim": 12,
        "time_steps": 20,
        "num_particles": 2048,
        "seeds": (91021, 91022, 91023),
        "mean_rmse_max": 0.45,
        "variance_rmse_max": 0.65,
        "loglik_abs_delta_max": 35.0,
    },
}
NONCLAIMS = (
    "LGSSM exact-Kalman gate artifact only",
    "no model-suite promotion claim",
    "no statistical superiority claim",
    "no nonlinear posterior correctness claim",
    "no dense Sinkhorn equivalence claim",
    "no HMC readiness claim",
    "no package/public default readiness claim",
)


@dataclass(frozen=True)
class LGSSMGateFixture:
    case_id: str
    role: str
    A: tf.Tensor
    C: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    seed: int

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self.C.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])


class RouteValueTensors(NamedTuple):
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
    parser.add_argument("--case-ids", nargs="+", default=list(PINNED_CASES))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--route", choices=("low_rank", "streaming", "both"), default="both")
    parser.add_argument("--num-particles", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--low-rank-rank", type=int, default=16)
    parser.add_argument("--low-rank-assignment-epsilon", type=float, default=0.25)
    parser.add_argument("--low-rank-alpha", type=float, default=1.0e-8)
    parser.add_argument("--low-rank-max-projection-iterations", type=int, default=120)
    parser.add_argument("--low-rank-convergence-threshold", type=float, default=1.0e-6)
    parser.add_argument("--low-rank-denominator-floor", type=float, default=1.0e-30)
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument("--row-chunk-size", type=int, default=128)
    parser.add_argument("--col-chunk-size", type=int, default=128)
    parser.add_argument("--particle-chunk-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--jit-compile", dest="jit_compile", action="store_true", default=True)
    parser.add_argument("--no-jit-compile", dest="jit_compile", action="store_false")
    parser.add_argument("--tf32-mode", choices=("default", "enabled", "disabled"), default="enabled")
    parser.add_argument("--device", default="/GPU:0")
    parser.add_argument("--device-scope", choices=("cpu", "visible"), default=_PRE_ARGS.device_scope)
    parser.add_argument("--cuda-visible-devices", default=_PRE_ARGS.cuda_visible_devices)
    parser.add_argument("--expect-device-kind", choices=("any", "cpu", "gpu"), default="gpu")
    parser.add_argument("--phase-id", default="LOW_RANK_LEDH_MODEL_SUITE_P01_LGSSM_KALMAN")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds is not None:
        args.seeds = _parse_int_csv(args.seeds)
    _validate_args(args)
    return args


def _validate_args(args: argparse.Namespace) -> None:
    unknown = [case_id for case_id in args.case_ids if case_id not in PINNED_CASES]
    if unknown:
        raise ValueError(f"unknown P01 case ids: {unknown}")
    if args.num_particles is not None and args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    for name in ("time_steps", "state_dim", "obs_dim"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.low_rank_rank <= 0:
        raise ValueError("low_rank_rank must be positive")
    particle_floor = min(_case_shape(case_id, args)["num_particles"] for case_id in args.case_ids)
    if args.low_rank_rank > particle_floor:
        raise ValueError("low_rank_rank must be <= num_particles for every selected case")
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
    for name in ("row_chunk_size", "col_chunk_size", "particle_chunk_size"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be nonnegative and repeats must be positive")
    if args.device_scope == "cpu" and args.expect_device_kind == "gpu":
        raise ValueError("cpu device scope cannot expect gpu outputs")


def _case_shape(case_id: str, args: argparse.Namespace) -> dict[str, int]:
    spec = PINNED_CASES[case_id]
    return {
        "state_dim": int(args.state_dim or spec["state_dim"]),
        "obs_dim": int(args.obs_dim or spec["obs_dim"]),
        "time_steps": int(args.time_steps or spec["time_steps"]),
        "num_particles": int(args.num_particles or spec["num_particles"]),
    }


def _route_names(route: str) -> list[str]:
    return ["streaming", "low_rank"] if route == "both" else [route]


def _dtype(args: argparse.Namespace) -> tf.DType:
    return tf.float64 if args.dtype == "float64" else tf.float32


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype(args)
    core_tf.DTYPE = dtype
    streaming_tf.DTYPE = dtype
    low_rank_coupling_solver_tf.DTYPE = dtype
    low_rank_coupling_solver_tf.DEFAULT_DTYPE = dtype
    if args.tf32_mode == "enabled":
        tf.config.experimental.enable_tensor_float_32_execution(True)
    elif args.tf32_mode == "disabled":
        tf.config.experimental.enable_tensor_float_32_execution(False)
    try:
        tf32_recorded: bool | str = bool(tf.config.experimental.tensor_float_32_execution_enabled())
    except Exception as exc:  # noqa: BLE001
        tf32_recorded = f"unavailable:{type(exc).__name__}"
    return {
        "dtype": args.dtype,
        "tf_dtype": dtype.name,
        "tf32_requested": args.tf32_mode,
        "tf32_execution_recorded": tf32_recorded,
    }


def _configure_gpus() -> dict[str, list[str]]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    for gpu in physical_gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    return {
        "physical_gpus": [str(device) for device in physical_gpus],
        "logical_gpus": [str(device) for device in logical_gpus],
    }


def build_lgssm_gate_fixture(case_id: str, seed: int, args: argparse.Namespace) -> LGSSMGateFixture:
    shape = _case_shape(case_id, args)
    dtype = _dtype(args)
    state_dim = shape["state_dim"]
    obs_dim = shape["obs_dim"]
    horizon = shape["time_steps"]
    spec = PINNED_CASES[case_id]

    state_index = tf.cast(tf.range(state_dim), dtype)
    obs_index = tf.cast(tf.range(obs_dim), dtype)
    diagonal = 0.72 + 0.12 * (state_index + 1.0) / tf.cast(state_dim, dtype)
    A = tf.linalg.diag(diagonal)
    if state_dim > 1:
        A = A + 0.015 * tf.linalg.diag(tf.ones([state_dim - 1], dtype=dtype), k=1)
        A = A - 0.010 * tf.linalg.diag(tf.ones([state_dim - 1], dtype=dtype), k=-1)
    C = _observation_matrix(obs_dim, state_dim, dtype)
    q_diag = 0.10 + 0.03 * (state_index + 1.0) / tf.cast(state_dim, dtype)
    r_base = 0.22 if case_id != "lgssm_informative_obs_stress" else 0.12
    r_diag = r_base + 0.02 * (obs_index + 1.0) / tf.cast(obs_dim, dtype)
    Q = tf.linalg.diag(q_diag)
    R = tf.linalg.diag(r_diag)
    m0 = 0.05 * tf.sin((state_index + 1.0) * tf.constant(0.7, dtype=dtype))
    P0 = tf.linalg.diag(0.35 + 0.04 * (state_index + 1.0) / tf.cast(state_dim, dtype))
    states, observations = _simulate_lgssm(A, C, Q, R, m0, P0, horizon, seed, dtype)
    return LGSSMGateFixture(
        case_id=case_id,
        role=str(spec["role"]),
        A=A,
        C=C,
        Q=Q,
        R=R,
        m0=m0,
        P0=P0,
        states=states,
        observations=observations,
        seed=int(seed),
    )


def _observation_matrix(obs_dim: int, state_dim: int, dtype: tf.DType) -> tf.Tensor:
    obs = tf.range(obs_dim, dtype=tf.int32)
    primary = tf.math.floormod(obs, state_dim)
    secondary = tf.math.floormod(obs + 1, state_dim)
    tertiary = tf.math.floormod(obs + 3, state_dim)
    indices = tf.concat(
        [
            tf.stack([obs, primary], axis=1),
            tf.stack([obs, secondary], axis=1),
            tf.stack([obs, tertiary], axis=1),
        ],
        axis=0,
    )
    values = tf.concat(
        [
            tf.fill([obs_dim], tf.cast(0.62, dtype)),
            tf.fill([obs_dim], tf.cast(0.12, dtype)),
            tf.fill([obs_dim], tf.cast(0.04, dtype)),
        ],
        axis=0,
    )
    return tf.scatter_nd(indices, values, [obs_dim, state_dim])


def _simulate_lgssm(
    A: tf.Tensor,
    C: tf.Tensor,
    Q: tf.Tensor,
    R: tf.Tensor,
    m0: tf.Tensor,
    P0: tf.Tensor,
    horizon: int,
    seed: int,
    dtype: tf.DType,
) -> tuple[tf.Tensor, tf.Tensor]:
    x0 = _mvnormal_sample(m0, P0, seed, 1, dtype)
    q_zero = tf.zeros([int(A.shape[0])], dtype=dtype)
    r_zero = tf.zeros([int(C.shape[0])], dtype=dtype)
    states = [x0]
    observations = []
    for time_index in range(horizon):
        transition_noise = _mvnormal_sample(q_zero, Q, seed, 100 + time_index, dtype)
        state = tf.linalg.matvec(A, states[-1]) + transition_noise
        obs_noise = _mvnormal_sample(r_zero, R, seed, 1000 + time_index, dtype)
        observation = tf.linalg.matvec(C, state) + obs_noise
        states.append(state)
        observations.append(observation)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _mvnormal_sample(loc: tf.Tensor, covariance: tf.Tensor, seed: int, salt: int, dtype: tf.DType) -> tf.Tensor:
    dist = tfd.MultivariateNormalTriL(
        loc=tf.cast(loc, dtype),
        scale_tril=tf.linalg.cholesky(tf.cast(covariance, dtype)),
    )
    return dist.sample(seed=tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32))


def sample_initial_particles(fixture: LGSSMGateFixture, num_particles: int, seed: int, dtype: tf.DType) -> tf.Tensor:
    dist = tfd.MultivariateNormalTriL(
        loc=tf.cast(fixture.m0, dtype),
        scale_tril=tf.linalg.cholesky(tf.cast(fixture.P0, dtype)),
    )
    particles = dist.sample(
        [1, num_particles],
        seed=tf.constant([int(seed) % 2147483647, 17], dtype=tf.int32),
    )
    return tf.cast(particles, dtype)


def run_kalman_reference(fixture: LGSSMGateFixture, dtype: tf.DType) -> dict[str, tf.Tensor]:
    m_prev = tf.cast(fixture.m0, dtype)
    p_prev = tf.cast(fixture.P0, dtype)
    A = tf.cast(fixture.A, dtype)
    C = tf.cast(fixture.C, dtype)
    Q = tf.cast(fixture.Q, dtype)
    R = tf.cast(fixture.R, dtype)
    eye = tf.eye(fixture.state_dim, dtype=dtype)
    means = []
    covariances = []
    log_likelihood = tf.constant(0.0, dtype=dtype)
    for observation in tf.unstack(tf.cast(fixture.observations, dtype), axis=0):
        m_pred = tf.linalg.matvec(A, m_prev)
        p_pred = A @ p_prev @ tf.transpose(A) + Q
        y_pred = tf.linalg.matvec(C, m_pred)
        s_mat = C @ p_pred @ tf.transpose(C) + R
        residual = tf.reshape(observation, [-1]) - y_pred
        log_likelihood += _gaussian_logpdf_zero_mean(tf.reshape(residual, [1, -1]), s_mat, dtype)[0]
        k_gain = tf.transpose(tf.linalg.solve(s_mat, C @ p_pred))
        m_filt = m_pred + tf.linalg.matvec(k_gain, residual)
        p_filt = (eye - k_gain @ C) @ p_pred
        p_filt = 0.5 * (p_filt + tf.transpose(p_filt))
        means.append(m_filt)
        covariances.append(p_filt)
        m_prev = m_filt
        p_prev = p_filt
    filtered_means = tf.stack(means, axis=0)
    filtered_covariances = tf.stack(covariances, axis=0)
    return {
        "filtered_means": filtered_means,
        "filtered_variances": tf.linalg.diag_part(filtered_covariances),
        "log_likelihood": log_likelihood,
        "finite": (
            tf.reduce_all(tf.math.is_finite(filtered_means))
            & tf.reduce_all(tf.math.is_finite(filtered_covariances))
            & tf.math.is_finite(log_likelihood)
        ),
    }


def _fixture_tensors(fixture: LGSSMGateFixture, num_particles: int, seed: int, dtype: tf.DType) -> dict[str, Any]:
    fixed_mask = tf.ones([1, fixture.horizon], dtype=tf.bool)
    return {
        "observations": tf.cast(fixture.observations, dtype),
        "initial_particles": sample_initial_particles(fixture, num_particles, seed, dtype),
        "transition_matrix": tf.cast(fixture.A[None, :, :], dtype),
        "transition_covariance": tf.cast(fixture.Q[None, :, :], dtype),
        "observation_matrix": tf.cast(fixture.C[None, :, :], dtype),
        "observation_covariance": tf.cast(fixture.R[None, :, :], dtype),
        "fixed_resampling_mask": fixed_mask,
        "fixed_resampling_policy": "all_active",
    }


def _callbacks(tensors: dict[str, tf.Tensor], dtype: tf.DType) -> dict[str, Any]:
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_matrix = tensors["observation_matrix"]
    observation_covariance = tensors["observation_covariance"]

    def pre_flow_step_fn(particles: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bnj,bdj->bnd", particles, transition_matrix)

    def prior_mean_fn(points: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bnj,bdj->bnd", points, transition_matrix)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bmd,bnd->bnm", observation_matrix, points)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        batch_size = int(points.shape[0])
        num_particles = int(points.shape[1])
        return tf.tile(observation_matrix[:, None, :, :], [1, num_particles, 1, 1])

    def observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return observation[None, None, :] - h_ref

    def transition_log_density_fn(x_next: tf.Tensor, x_prev: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        mean = tf.einsum("bnj,bdj->bnd", x_prev, transition_matrix)
        return _batched_gaussian_logpdf(x_next - mean, transition_covariance, dtype)

    def observation_log_density_fn(x: tf.Tensor, observation: tf.Tensor, _time_index: tf.Tensor) -> tf.Tensor:
        predicted = tf.einsum("bmd,bnd->bnm", observation_matrix, x)
        return _batched_gaussian_logpdf(predicted - observation[None, None, :], observation_covariance, dtype)

    return {
        "pre_flow_step_fn": pre_flow_step_fn,
        "prior_mean_fn": prior_mean_fn,
        "observation_fn": observation_fn,
        "observation_jacobian_fn": observation_jacobian_fn,
        "observation_residual_fn": observation_residual_fn,
        "transition_log_density_fn": transition_log_density_fn,
        "observation_log_density_fn": observation_log_density_fn,
    }


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(residuals)))
    quad = tf.reduce_sum(solved * residuals, axis=-1)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
    dim = tf.cast(int(residuals.shape[-1]), dtype)
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype)) + logdet[:, None] + quad)


def _gaussian_logpdf_zero_mean(residuals: tf.Tensor, covariance: tf.Tensor, dtype: tf.DType) -> tf.Tensor:
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.cholesky_solve(chol, tf.transpose(residuals))
    quad = tf.reduce_sum(tf.transpose(solved) * residuals, axis=1)
    dim = tf.cast(int(covariance.shape[0]), dtype)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype)) + logdet + quad)


def _normalize_log_weights(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return core_tf._normalize_log_weights(log_weights)


def _weighted_mean_and_variance(particles: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    return core_tf._weighted_mean_and_variance(particles, weights)


def _log_weight_floor() -> tf.Tensor:
    return core_tf._log_weight_floor()


def _active_step_counts_tensor(fixed_mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    active_by_step = tf.reduce_any(fixed_mask, axis=0)
    return tf.reduce_sum(tf.cast(active_by_step, tf.int32)), tf.reduce_sum(tf.cast(fixed_mask, tf.int32))


def _static_bool_tensor_value(value: tf.Tensor) -> bool | None:
    static_value = tf.get_static_value(value)
    if static_value is None:
        return None
    return bool(static_value)


def _select_transport_step(
    mask: tf.Tensor,
    transport_fn: Callable[[], tuple[tf.Tensor, ...]],
    skip_transport_fn: Callable[[], tuple[tf.Tensor, ...]],
) -> tuple[tf.Tensor, ...]:
    mask_active = tf.reduce_any(mask)
    static_mask_active = _static_bool_tensor_value(mask_active)
    if static_mask_active is True:
        return transport_fn()
    if static_mask_active is False:
        return skip_transport_fn()
    return tf.cond(mask_active, transport_fn, skip_transport_fn)


def _route_value_core(
    route: str,
    tensors: dict[str, Any],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
) -> RouteValueTensors:
    observations = tensors["observations"]
    particles = tensors["initial_particles"]
    fixed_mask = tensors["fixed_resampling_mask"]
    fixed_resampling_policy = tensors.get("fixed_resampling_policy", "dynamic")
    transition_matrix = tensors["transition_matrix"]
    transition_covariance = tensors["transition_covariance"]
    observation_covariance = tensors["observation_covariance"]
    batch_size = int(particles.shape[0])
    num_particles = int(particles.shape[1])
    state_dim = int(particles.shape[2])
    time_steps = int(observations.shape[0])
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=dtype)
    means_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size, state_dim]))
    vars_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size, state_dim]))
    ess_ta = tf.TensorArray(dtype=dtype, size=time_steps, element_shape=tf.TensorShape([batch_size]))
    active_steps, active_entries = _active_step_counts_tensor(fixed_mask)
    route_invocations = tf.constant(0, dtype=tf.int32)
    max_factor = tf.constant(0.0, dtype=dtype)
    max_row = tf.constant(0.0, dtype=dtype)
    max_col = tf.constant(0.0, dtype=dtype)
    max_iter = tf.constant(0, dtype=tf.int32)
    finite_factors = tf.constant(True)
    finite_particles = tf.constant(True)
    nonnegative_factors = tf.constant(True)
    positive_g = tf.constant(True)

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
        weights, incremental = _normalize_log_weights(corrected_log_weights)
        log_likelihood += incremental
        ess = 1.0 / tf.reduce_sum(weights * weights, axis=1)
        mean, variance = _weighted_mean_and_variance(post_flow, weights)
        means_ta = means_ta.write(time_index, mean)
        vars_ta = vars_ta.write(time_index, variance)
        ess_ta = ess_ta.write(time_index, ess)
        normalized_log_weights = tf.math.log(tf.maximum(weights, _log_weight_floor()))
        mask = fixed_mask[:, time_index]

        def streaming_transport():
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
                tf.constant(0.0, dtype=dtype),
                tf.cast(transported.max_row_residual, dtype),
                tf.cast(transported.max_column_residual, dtype),
                tf.constant(0, dtype=tf.int32),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(1, dtype=tf.int32),
            )

        def low_rank_transport():
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
                tf.cast(resampled.particles, dtype),
                tf.cast(resampled.log_weights, dtype),
                tf.cast(resampled.max_factor_marginal_residual, dtype),
                tf.cast(resampled.max_induced_row_residual, dtype),
                tf.cast(resampled.max_induced_column_residual, dtype),
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
                tf.constant(0.0, dtype=dtype),
                tf.constant(0.0, dtype=dtype),
                tf.constant(0.0, dtype=dtype),
                tf.constant(0, dtype=tf.int32),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(True),
                tf.constant(0, dtype=tf.int32),
            )

        if route == "streaming":
            transport_fn = streaming_transport
        elif route == "low_rank":
            transport_fn = low_rank_transport
        else:
            raise ValueError(f"unknown route: {route}")
        if fixed_resampling_policy == "all_active":
            (
                particles,
                log_weights,
                factor_residual,
                row_residual,
                col_residual,
                iterations_used,
                step_finite_factors,
                step_finite_particles,
                step_nonnegative_factors,
                step_positive_g,
                step_invocation,
            ) = transport_fn()
        elif fixed_resampling_policy == "all_inactive":
            (
                particles,
                log_weights,
                factor_residual,
                row_residual,
                col_residual,
                iterations_used,
                step_finite_factors,
                step_finite_particles,
                step_nonnegative_factors,
                step_positive_g,
                step_invocation,
            ) = skip_transport()
        else:
            (
                particles,
                log_weights,
                factor_residual,
                row_residual,
                col_residual,
                iterations_used,
                step_finite_factors,
                step_finite_particles,
                step_nonnegative_factors,
                step_positive_g,
                step_invocation,
            ) = _select_transport_step(mask, transport_fn, skip_transport)
        max_factor = tf.maximum(max_factor, factor_residual)
        max_row = tf.maximum(max_row, row_residual)
        max_col = tf.maximum(max_col, col_residual)
        max_iter = tf.maximum(max_iter, iterations_used)
        finite_factors = tf.logical_and(finite_factors, step_finite_factors)
        finite_particles = tf.logical_and(finite_particles, step_finite_particles)
        nonnegative_factors = tf.logical_and(nonnegative_factors, step_nonnegative_factors)
        positive_g = tf.logical_and(positive_g, step_positive_g)
        route_invocations += step_invocation

    filtered_means = means_ta.stack()
    filtered_variances = vars_ta.stack()
    ess_by_time = ess_ta.stack()
    final_weights = tf.exp(log_weights - tf.reduce_logsumexp(log_weights, axis=1, keepdims=True))
    final_particle_mean, _ = _weighted_mean_and_variance(particles, final_weights)
    final_logsumexp_residual = tf.reduce_max(tf.abs(tf.reduce_logsumexp(log_weights, axis=1)))
    return RouteValueTensors(
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
        active_resampling_batch_entries=active_entries,
        max_factor_marginal_residual=max_factor,
        max_induced_row_residual=max_row,
        max_induced_column_residual=max_col,
        projection_iterations_used_max=max_iter,
        finite_factors=finite_factors,
        finite_particles=finite_particles,
        nonnegative_factors=nonnegative_factors,
        positive_g=positive_g,
    )


def _compiled_route_outputs(
    route: str,
    tensors: dict[str, tf.Tensor],
    callbacks: dict[str, Any],
    args: argparse.Namespace,
    dtype: tf.DType,
):
    @tf.function(jit_compile=args.jit_compile, reduce_retracing=True)
    def compiled_outputs() -> tuple[tf.Tensor, ...]:
        value = _route_value_core(route, tensors, callbacks, args, dtype)
        return tuple(value)

    return compiled_outputs


def _materialize_tensors(*tensors: tf.Tensor) -> None:
    for tensor in tensors:
        tensor.numpy()


def _run_route_case(route: str, fixture: LGSSMGateFixture, args: argparse.Namespace) -> dict[str, Any]:
    dtype = _dtype(args)
    shape = _case_shape(fixture.case_id, args)
    tensors = _fixture_tensors(fixture, shape["num_particles"], fixture.seed, dtype)
    callbacks = _callbacks(tensors, dtype)
    compiled_outputs = _compiled_route_outputs(route, tensors, callbacks, args, dtype)
    start = time.perf_counter()
    with tf.device(args.device):
        outputs = compiled_outputs()
        _materialize_tensors(*outputs)
        first_call_seconds = time.perf_counter() - start
        for _ in range(args.warmups):
            _materialize_tensors(*compiled_outputs())
        timings = []
        last_outputs = outputs
        for _ in range(args.repeats):
            row_start = time.perf_counter()
            last_outputs = compiled_outputs()
            _materialize_tensors(*last_outputs)
            timings.append(time.perf_counter() - row_start)
    kalman = run_kalman_reference(fixture, dtype)
    loop = _loop_from_outputs(route, last_outputs, fixture, shape)
    errors = _kalman_errors(loop, kalman)
    output_devices = [
        last_outputs[0].device,
        last_outputs[1].device,
        last_outputs[2].device,
        last_outputs[3].device,
    ]
    hard_vetoes = _route_hard_vetoes(route, loop, errors, kalman, output_devices, args)
    return {
        "case_id": fixture.case_id,
        "case_role": fixture.role,
        "seed": fixture.seed,
        "route": route,
        "status": "PASS" if not hard_vetoes else "FAIL",
        "hard_vetoes": hard_vetoes,
        "shape": shape,
        "thresholds": _case_thresholds(fixture.case_id),
        "first_call_seconds": first_call_seconds,
        "warm_call_timings_seconds": timings,
        "warm_call_timing_summary_seconds": _summary(timings),
        "output_devices": output_devices,
        **loop,
        **errors,
    }


def _loop_from_outputs(
    route: str,
    outputs: tuple[tf.Tensor, ...],
    fixture: LGSSMGateFixture,
    shape: dict[str, int],
) -> dict[str, Any]:
    diagnostics = {
        "transport_object_kind": "streaming_transport"
        if route == "streaming"
        else "low_rank_coupling_factors",
        "transport_matrix_materialized": False,
        "transport_matrix_shapes": [[1, 0, 0]],
        "route_invocations": int(outputs[8].numpy()),
        "active_resampling_mask_count": int(outputs[9].numpy()),
        "active_resampling_batch_entries": int(outputs[10].numpy()),
        "max_factor_marginal_residual": _float(outputs[11]),
        "max_induced_row_residual": _float(outputs[12]),
        "max_induced_column_residual": _float(outputs[13]),
        "projection_iterations_used_max": int(outputs[14].numpy()),
        "finite_factors": bool(outputs[15].numpy()),
        "finite_particles": bool(outputs[16].numpy()),
        "nonnegative_factors": bool(outputs[17].numpy()),
        "positive_g": bool(outputs[18].numpy()),
    }
    return {
        "fixture": {
            "case_id": fixture.case_id,
            "role": fixture.role,
            "seed": fixture.seed,
            "state_dim": fixture.state_dim,
            "obs_dim": fixture.obs_dim,
            "time_steps": fixture.horizon,
        },
        "log_likelihood": outputs[0],
        "filtered_means": outputs[1],
        "filtered_variances": outputs[2],
        "ess_by_time": outputs[3],
        "final_particle_mean": outputs[6],
        "final_logsumexp_residual": _float(outputs[7]),
        "finite_output": _finite(outputs[0]) and _finite(outputs[1]) and _finite(outputs[2]) and _finite(outputs[3]),
        "ess_min": _float(tf.reduce_min(outputs[3])),
        "ess_fraction_min": _float(tf.reduce_min(outputs[3])) / float(shape["num_particles"]),
        "filtered_means_preview": _preview(outputs[1]),
        "filtered_variances_preview": _preview(outputs[2]),
        "log_likelihood_value": _float(outputs[0]),
        "final_particle_mean_preview": _preview(outputs[6]),
        **diagnostics,
    }


def _kalman_errors(loop: dict[str, Any], kalman: dict[str, tf.Tensor]) -> dict[str, Any]:
    means = tf.squeeze(tf.cast(loop["filtered_means"], tf.float64), axis=1)
    variances = tf.squeeze(tf.cast(loop["filtered_variances"], tf.float64), axis=1)
    ref_means = tf.cast(kalman["filtered_means"], tf.float64)
    ref_variances = tf.cast(kalman["filtered_variances"], tf.float64)
    loglik = tf.reshape(tf.cast(loop["log_likelihood"], tf.float64), [-1])[0]
    ref_loglik = tf.cast(kalman["log_likelihood"], tf.float64)
    return {
        "kalman_reference_finite": bool(kalman["finite"].numpy()),
        "kalman_log_likelihood": _float(ref_loglik),
        "mean_rmse": _float(tf.sqrt(tf.reduce_mean(tf.square(means - ref_means)))),
        "variance_rmse": _float(tf.sqrt(tf.reduce_mean(tf.square(variances - ref_variances)))),
        "loglik_abs_delta": _float(tf.abs(loglik - ref_loglik)),
    }


def _case_thresholds(case_id: str) -> dict[str, float]:
    spec = PINNED_CASES[case_id]
    return {
        "mean_rmse_max": float(spec["mean_rmse_max"]),
        "variance_rmse_max": float(spec["variance_rmse_max"]),
        "loglik_abs_delta_max": float(spec["loglik_abs_delta_max"]),
        "factor_residual_max": FACTOR_RESIDUAL_THRESHOLD,
        "log_weight_normalization_max": LOG_WEIGHT_NORMALIZATION_THRESHOLD,
        "ess_fraction_min": ESS_FRACTION_MIN_THRESHOLD,
    }


def _route_hard_vetoes(
    route: str,
    loop: dict[str, Any],
    errors: dict[str, Any],
    kalman: dict[str, tf.Tensor],
    output_devices: list[str],
    args: argparse.Namespace,
) -> list[str]:
    vetoes: list[str] = []
    thresholds = _case_thresholds(loop["fixture"]["case_id"])
    if not errors["kalman_reference_finite"]:
        vetoes.append("nonfinite_kalman_reference")
    if not loop["finite_output"]:
        vetoes.append("nonfinite_route_output")
    if loop["route_invocations"] != loop["active_resampling_mask_count"]:
        vetoes.append("route_invocation_count_mismatch")
    if loop["active_resampling_mask_count"] > 0 and loop["route_invocations"] <= 0:
        vetoes.append("route_invocations_zero")
    if loop["transport_matrix_materialized"]:
        vetoes.append("transport_matrix_materialized")
    if loop["final_logsumexp_residual"] > thresholds["log_weight_normalization_max"]:
        vetoes.append("final_logsumexp_residual_threshold")
    if loop["ess_fraction_min"] < thresholds["ess_fraction_min"]:
        vetoes.append("ess_fraction_min_threshold")
    if route == "low_rank":
        if loop["max_factor_marginal_residual"] > thresholds["factor_residual_max"]:
            vetoes.append("factor_marginal_residual_threshold")
        if not loop["finite_factors"]:
            vetoes.append("nonfinite_factors")
        if not loop["finite_particles"]:
            vetoes.append("low_rank_nonfinite_particles")
        if not loop["nonnegative_factors"]:
            vetoes.append("negative_factors")
        if not loop["positive_g"]:
            vetoes.append("nonpositive_g")
    if route == "low_rank":
        if errors["mean_rmse"] > thresholds["mean_rmse_max"]:
            vetoes.append("mean_rmse_threshold")
        if errors["variance_rmse"] > thresholds["variance_rmse_max"]:
            vetoes.append("variance_rmse_threshold")
        if errors["loglik_abs_delta"] > thresholds["loglik_abs_delta_max"]:
            vetoes.append("loglik_abs_delta_threshold")
    if args.expect_device_kind == "gpu" and not all("GPU" in device.upper() for device in output_devices):
        vetoes.append("expected_gpu_outputs_missing")
    if args.expect_device_kind == "cpu" and not all("CPU" in device.upper() for device in output_devices):
        vetoes.append("expected_cpu_outputs_missing")
    del kalman
    return vetoes


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    tf_metadata = _configure_precision(args)
    device_metadata = _configure_gpus()
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    start = time.perf_counter()
    rows = []
    for case_id in args.case_ids:
        spec = PINNED_CASES[case_id]
        seeds = list(args.seeds if args.seeds is not None else spec["seeds"])
        for seed in seeds:
            fixture = build_lgssm_gate_fixture(case_id, int(seed), args)
            for route in _route_names(args.route):
                rows.append(_run_route_case(route, fixture, args))
    ended_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    hard_vetoes = [
        f"{row['case_id']}:{row['seed']}:{row['route']}:{veto}"
        for row in rows
        for veto in row.get("hard_vetoes", [])
    ]
    if args.device_scope == "cpu":
        evidence_class = "cpu_hidden_command_shape_debug_only"
    elif args.expect_device_kind == "gpu":
        evidence_class = "trusted_gpu_candidate_only_if_launcher_records_trusted_context"
    else:
        evidence_class = "local_debug_only"
    return {
        "schema_version": "low_rank_ledh_lgssm_kalman_gate.v1",
        "status": "PASS" if not hard_vetoes else "FAIL",
        "phase": args.phase_id,
        "evidence_class": evidence_class,
        "algorithm_family": "low_rank_ledh_pfpf_ot_lgssm_exact_kalman_gate",
        "candidate": {
            "route": "low_rank_ledh_pfpf_ot",
            "candidate_id": "r16_eps0p25_alpha1em08_it120",
            "rank": args.low_rank_rank,
            "assignment_epsilon": args.low_rank_assignment_epsilon,
            "alpha": args.low_rank_alpha,
            "max_projection_iterations": args.low_rank_max_projection_iterations,
            "convergence_threshold": args.low_rank_convergence_threshold,
            "denominator_floor": args.low_rank_denominator_floor,
        },
        "hard_vetoes": hard_vetoes,
        "rows": rows,
        "pinned_cases": {case_id: _json_ready(PINNED_CASES[case_id]) for case_id in args.case_ids},
        "run_manifest": _run_manifest(
            args,
            started_at=started_at,
            ended_at=ended_at,
            wall_time_seconds=time.perf_counter() - start,
            tf_metadata=tf_metadata,
            device_metadata=device_metadata,
        ),
        "nonclaims": list(NONCLAIMS),
    }


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
        "tensorflow_probability_version": tfp.__version__,
        "device_scope": args.device_scope,
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": args.jit_compile,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "case_ids": list(args.case_ids),
        "seeds_override": list(args.seeds) if args.seeds is not None else None,
        "plan_path": PLAN_PATH,
        "subplan_path": P01_SUBPLAN_PATH,
        "p01a_result_path": P01A_RESULT_PATH,
        **tf_metadata,
        **device_metadata,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_time_seconds": wall_time_seconds,
    }


def _git_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _float(value: Any) -> float:
    return float(tf.reshape(tf.cast(value, tf.float64), [-1])[0].numpy())


def _preview(value: tf.Tensor, limit: int = 8) -> list[float]:
    flat = tf.reshape(tf.cast(value, tf.float64), [-1])
    return [float(item) for item in flat[: min(limit, int(flat.shape[0]))].numpy().tolist()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        try:
            return _json_ready(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def write_markdown(result: dict[str, Any], path: Path, json_path: Path | None = None) -> None:
    lines = [
        "# Low-Rank LEDH LGSSM Exact-Kalman Gate",
        "",
        f"- Status: `{result['status']}`",
        f"- Phase: `{result['phase']}`",
        f"- Evidence class: `{result['evidence_class']}`",
        f"- Hard vetoes: `{result['hard_vetoes']}`",
    ]
    if json_path is not None:
        lines.append(f"- JSON artifact: `{json_path}`")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| Case | Seed | Route | Status | Mean RMSE | Var RMSE | Loglik delta | Invocations | Vetoes |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in result["rows"]:
        lines.append(
            "| {case} | {seed} | `{route}` | `{status}` | {mean} | {var} | {ll} | {inv} | `{vetoes}` |".format(
                case=row["case_id"],
                seed=row["seed"],
                route=row["route"],
                status=row["status"],
                mean=row["mean_rmse"],
                var=row["variance_rmse"],
                ll=row["loglik_abs_delta"],
                inv=row["route_invocations"],
                vetoes=row["hard_vetoes"],
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
            f"- JIT compile: `{result['run_manifest']['jit_compile']}`",
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
    markdown = Path(args.markdown_output) if args.markdown_output else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(_json_ready(result), markdown, output)
    if not args.quiet:
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
