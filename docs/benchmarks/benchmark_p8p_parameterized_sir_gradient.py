"""P8p parameterized SIR d18 gradient diagnostic for LEDH-PFPF-OT.

This is a gated diagnostic harness, not a leaderboard runner.  It threads a
small three-parameter theta through the current actual-SIR streaming
LEDH-PFPF-OT route under fixed observations, fixed initial particles, fixed
stateless process-noise streams, and relaxed Sinkhorn OT.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PRE_PARSER = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_PRE_PARSER.add_argument("--device-scope", choices=("cpu", "visible"), default="visible")
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

import tensorflow as tf

from bayesfilter import highdim
from bayesfilter.highdim.ledh_historical_raw_policy import (
    require_historical_raw_diagnostic_opt_in,
)
from bayesfilter.highdim.ledh_forward_contract import (
    make_parameterized_sir_diagnostic_forward_contract,
)
from docs.benchmarks import benchmark_p8j_tf32_batched_actual_sir as p8j_sir
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_streaming_tf as streaming_tf,
)
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)
from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from scripts.filtering_value_gradient_benchmark_run_p8d_numeric import (
    _dpf_sir_callbacks,
    _sir_observations,
)


DTYPE = tf.float32
PARAMETER_NAMES = (
    "log_kappa_scale",
    "log_nu_scale",
    "log_obs_noise_scale",
)
MANUAL_SCORE_COMPONENT_NAMES = (
    "observation_density_covariance",
    "ledh_flow_observation_covariance",
    "transition_mean_from_transition_density",
    "transition_mean_from_ledh_flow_prior",
    "transition_mean_from_pre_flow_clamp",
)

_SIR_MODEL = highdim.zhao_cui_sir_austria_model()
_SIR_BASE_KAPPA = _SIR_MODEL.kappa
_SIR_BASE_NU = _SIR_MODEL.nu
_SIR_BASE_OBSERVATION_COVARIANCE = _SIR_MODEL.observation_covariance
_SIR_ADJACENCY_MATRIX = _SIR_MODEL._adjacency_matrix  # noqa: SLF001
_SIR_NEIGHBOR_DEGREE = _SIR_MODEL._neighbor_degree  # noqa: SLF001
_SIR_RK4_SUBSTEPS = int(_SIR_MODEL._rk4_substeps)  # noqa: SLF001
_SIR_DELTA = _SIR_MODEL.delta
_SIR_INFECTIOUS_INDICES = tuple(range(1, 18, 2))
_SIR_INFECTIOUS_SELECTOR = tf.one_hot(
    tf.constant(_SIR_INFECTIOUS_INDICES, dtype=tf.int32),
    depth=18,
    dtype=tf.float64,
)
NONCLAIMS = (
    "P8p diagnostic parameterized SIR d18 target only",
    "not the fixed-parameter P8o leaderboard cell",
    "not stochastic PF marginal-gradient correctness",
    "not exact nonlinear likelihood correctness",
    "not HMC/NUTS readiness",
    "not posterior convergence",
    "not production/default readiness",
    "not Zhao-Cui TT/SIRT or MATLAB parity",
)


def _parse_int_csv(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one integer")
    return parsed


def _parse_float_csv(value: str, *, expected: int) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if len(parsed) != expected:
        raise ValueError(f"expected {expected} comma-separated floats")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError("theta entries must be finite")
    return parsed


def _parse_positive_float_csv(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("expected at least one positive float")
    if not all(math.isfinite(item) and item > 0.0 for item in parsed):
        raise ValueError("all finite-difference ladder steps must be positive and finite")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--batch-seeds", default="81120")
    parser.add_argument("--time-steps", type=int, default=3)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--theta", default="0,0,0")
    parser.add_argument("--phase-label", default="P8p diagnostic")
    parser.add_argument("--fd-step", type=float, default=1.0e-3)
    parser.add_argument("--fd-step-ladder", default="")
    parser.add_argument(
        "--diagnostic-components",
        choices=("all", "obs-noise"),
        default="all",
    )
    parser.add_argument(
        "--check-isolated-observation-noise",
        action="store_true",
    )
    parser.add_argument("--repeat-evaluations", type=int, default=2)
    parser.add_argument("--repeat-atol", type=float, default=1.0e-5)
    parser.add_argument("--theta-zero-parity-atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--check-theta-zero-p8j-parity",
        action="store_true",
    )
    parser.add_argument(
        "--transport-policy",
        choices=("active-all", "active-odd", "no-resampling"),
        default="active-all",
    )
    parser.add_argument("--sinkhorn-iterations", type=int, default=10)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=1.0)
    parser.add_argument("--annealed-scaling", type=float, default=0.9)
    parser.add_argument("--annealed-convergence-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--transport-plan-mode",
        choices=("streaming", "dense"),
        default="streaming",
    )
    parser.add_argument(
        "--transport-gradient-mode",
        choices=(
            "raw",
            core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        ),
        default="raw",
    )
    parser.add_argument(
        "--transport-ad-mode",
        choices=(
            "stabilized",
            "diff-scale",
            "diff-keys",
            "diff-potentials",
            "full",
        ),
        default="stabilized",
    )
    parser.add_argument("--row-chunk-size", type=int, default=8)
    parser.add_argument("--col-chunk-size", type=int, default=8)
    parser.add_argument("--particle-chunk-size", type=int, default=8)
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-raw-diagnostic", action="store_true")
    parser.add_argument("--no-fail-on-veto", action="store_true")
    args = parser.parse_args()
    args.batch_seeds = _parse_int_csv(args.batch_seeds)
    args.theta_values = _parse_float_csv(args.theta, expected=3)
    args.fd_step_ladder_values = (
        _parse_positive_float_csv(args.fd_step_ladder)
        if args.fd_step_ladder.strip()
        else [float(args.fd_step)]
    )
    if args.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if args.time_steps > int(_sir_observations().shape[0]):
        raise ValueError("time_steps exceeds available SIR observations")
    if args.num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if args.fd_step <= 0.0:
        raise ValueError("fd-step must be positive")
    if args.repeat_evaluations <= 0:
        raise ValueError("repeat-evaluations must be positive")
    if args.repeat_atol < 0.0 or args.theta_zero_parity_atol < 0.0:
        raise ValueError("tolerances must be nonnegative")
    if args.sinkhorn_iterations <= 0:
        raise ValueError("sinkhorn-iterations must be positive")
    if args.row_chunk_size <= 0 or args.col_chunk_size <= 0 or args.particle_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")
    return args


def _configure_precision(args: argparse.Namespace) -> dict[str, Any]:
    global DTYPE
    DTYPE = tf.float64 if args.dtype == "float64" else tf.float32
    core_tf.DTYPE = DTYPE
    streaming_tf.DTYPE = DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    p8j_sir.DTYPE = DTYPE
    if args.tf32_mode != "default":
        tf.config.experimental.enable_tensor_float_32_execution(args.tf32_mode == "enabled")
    metadata = core_tf.precision_policy_metadata()
    metadata.update(
        {
            "dtype": args.dtype,
            "tf_dtype": DTYPE.name,
            "tf32_mode": args.tf32_mode,
            "tf32_execution_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
        }
    )
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


def _gpu_memory_info() -> dict[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        return {"status": "unavailable"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _make_fixed_resampling_mask(batch_size: int, time_steps: int, policy: str) -> tf.Tensor:
    return p8j_sir._make_fixed_resampling_mask(batch_size, time_steps, policy)


def _make_transition_noise_tensor(
    *,
    batch_seeds: list[int],
    time_steps: int,
    num_particles: int,
    state_dim: int,
) -> tf.Tensor:
    """Build the fixed SIR process-noise tensor used by value and score paths."""

    time_rows = []
    for time_index in range(int(time_steps)):
        batch_rows = []
        for seed in batch_seeds:
            seed_tensor = tf.stack(
                [
                    tf.constant(int(seed) % 2147483647, dtype=tf.int32),
                    tf.math.floormod(
                        tf.constant(1140 + int(time_index), dtype=tf.int32),
                        tf.constant(2147483647, dtype=tf.int32),
                    ),
                ]
            )
            batch_rows.append(
                tf.random.stateless_normal(
                    [num_particles, state_dim],
                    seed=seed_tensor,
                    dtype=DTYPE,
                )
            )
        time_rows.append(tf.stack(batch_rows, axis=0))
    return tf.stack(time_rows, axis=1)


def _batched_gaussian_logpdf(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    return p8j_sir._batched_gaussian_logpdf(residuals, covariance)


def _build_base_tensors(args: argparse.Namespace) -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    p8j_args = argparse.Namespace(
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        batch_seeds=args.batch_seeds,
        transport_policy=args.transport_policy,
    )
    tensors, semantics = p8j_sir._build_actual_sir_tensors(p8j_args)
    tensors = dict(tensors)
    tensors["transition_noise"] = _make_transition_noise_tensor(
        batch_seeds=args.batch_seeds,
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        state_dim=18,
    )
    return tensors, semantics


def _theta_components(theta_values: list[float]) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    return tuple(tf.constant(value, dtype=DTYPE) for value in theta_values)  # type: ignore[return-value]


def _scaled_parameters(
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> dict[str, tf.Tensor]:
    log_kappa_scale, log_nu_scale, log_obs_noise_scale = theta_components
    base_kappa = tf.cast(_SIR_BASE_KAPPA, DTYPE)
    base_nu = tf.cast(_SIR_BASE_NU, DTYPE)
    base_observation_covariance = tf.cast(_SIR_BASE_OBSERVATION_COVARIANCE, DTYPE)
    return {
        "kappa": _scale_vector_parameter(base_kappa, log_kappa_scale),
        "nu": _scale_vector_parameter(base_nu, log_nu_scale),
        "observation_covariance": _scale_matrix_parameter(
            base_observation_covariance,
            log_obs_noise_scale,
        ),
    }


def _scale_vector_parameter(base: tf.Tensor, log_scale: tf.Tensor) -> tf.Tensor:
    scale = tf.exp(tf.convert_to_tensor(log_scale, dtype=DTYPE))
    base = tf.convert_to_tensor(base, dtype=DTYPE)
    if scale.shape.rank == 0:
        return base * scale
    if scale.shape.rank == 1:
        return base[tf.newaxis, :] * scale[:, tf.newaxis]
    raise ValueError("theta vector parameter scale must be scalar or rank 1")


def _scale_matrix_parameter(base: tf.Tensor, log_scale: tf.Tensor) -> tf.Tensor:
    scale = tf.exp(
        tf.constant(2.0, dtype=DTYPE) * tf.convert_to_tensor(log_scale, dtype=DTYPE)
    )
    base = tf.convert_to_tensor(base, dtype=DTYPE)
    if scale.shape.rank == 0:
        return base * scale
    if scale.shape.rank == 1:
        return base[tf.newaxis, :, :] * scale[:, tf.newaxis, tf.newaxis]
    raise ValueError("theta matrix parameter scale must be scalar or rank 1")


def _batch_vector_parameter(value: tf.Tensor) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=DTYPE)
    if tensor.shape.rank == 1:
        return tensor[tf.newaxis, tf.newaxis, :]
    if tensor.shape.rank == 2:
        return tensor[:, tf.newaxis, :]
    raise ValueError("batch vector parameter must have rank 1 or 2")


def _batch_matrix_parameter(value: tf.Tensor, batch_size: int) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=DTYPE)
    if tensor.shape.rank == 2:
        return tf.tile(tensor[tf.newaxis, :, :], [batch_size, 1, 1])
    if tensor.shape.rank == 3:
        if tensor.shape[0] is not None and int(tensor.shape[0]) != batch_size:
            raise ValueError("rank-3 matrix parameter batch dimension mismatch")
        return tensor
    raise ValueError("batch matrix parameter must have rank 2 or 3")


def _sir_flatten_components(susceptible: tf.Tensor, infectious: tf.Tensor) -> tf.Tensor:
    output_shape = tf.concat(
        [tf.shape(susceptible)[:-1], [tf.shape(susceptible)[-1] * 2]],
        axis=0,
    )
    return tf.reshape(tf.stack([susceptible, infectious], axis=3), output_shape)


def _sir_infectious_vjp(cotangent: tf.Tensor) -> tf.Tensor:
    bar_infectious = tf.convert_to_tensor(cotangent, dtype=DTYPE)
    zeros = tf.zeros_like(bar_infectious)
    return _sir_flatten_components(zeros, bar_infectious)


def _sir_rhs_tf(
    state: tf.Tensor,
    *,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
) -> tf.Tensor:
    susceptible = state[:, :, 0::2]
    infectious = state[:, :, 1::2]
    susceptible_neighbor = (
        tf.linalg.matmul(susceptible, adjacency, transpose_b=True)
        - susceptible * neighbor_degree[tf.newaxis, tf.newaxis, :]
    )
    infectious_neighbor = (
        tf.linalg.matmul(infectious, adjacency, transpose_b=True)
        - infectious * neighbor_degree[tf.newaxis, tf.newaxis, :]
    )
    infection = kappa[tf.newaxis, tf.newaxis, :] * susceptible * infectious
    d_susceptible = -infection + tf.constant(0.5, dtype=DTYPE) * susceptible_neighbor
    d_infectious = (
        infection
        - nu[tf.newaxis, tf.newaxis, :] * infectious
        + tf.constant(0.5, dtype=DTYPE) * infectious_neighbor
    )
    return _sir_flatten_components(d_susceptible, d_infectious)


def _sir_rhs_vjp_tf(
    state: tf.Tensor,
    upstream: tf.Tensor,
    *,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    state = tf.convert_to_tensor(state, dtype=DTYPE)
    upstream = tf.convert_to_tensor(upstream, dtype=DTYPE)
    susceptible = state[:, :, 0::2]
    infectious = state[:, :, 1::2]
    bar_susceptible_rhs = upstream[:, :, 0::2]
    bar_infectious_rhs = upstream[:, :, 1::2]

    bar_state_susceptible = tf.zeros_like(susceptible)
    bar_state_infectious = tf.zeros_like(infectious)

    bar_infection = -bar_susceptible_rhs + bar_infectious_rhs
    bar_kappa = tf.reduce_sum(bar_infection * susceptible * infectious, axis=1)
    bar_state_susceptible += bar_infection * kappa[tf.newaxis, tf.newaxis, :] * infectious
    bar_state_infectious += bar_infection * kappa[tf.newaxis, tf.newaxis, :] * susceptible

    bar_nu = tf.reduce_sum(-bar_infectious_rhs * infectious, axis=1)
    bar_state_infectious += -nu[tf.newaxis, tf.newaxis, :] * bar_infectious_rhs

    half = tf.constant(0.5, dtype=DTYPE)
    bar_susceptible_neighbor = half * bar_susceptible_rhs
    bar_infectious_neighbor = half * bar_infectious_rhs
    bar_state_susceptible += (
        tf.linalg.matmul(bar_susceptible_neighbor, adjacency)
        - bar_susceptible_neighbor * neighbor_degree[tf.newaxis, tf.newaxis, :]
    )
    bar_state_infectious += (
        tf.linalg.matmul(bar_infectious_neighbor, adjacency)
        - bar_infectious_neighbor * neighbor_degree[tf.newaxis, tf.newaxis, :]
    )

    return (
        _sir_flatten_components(bar_state_susceptible, bar_state_infectious),
        bar_kappa,
        bar_nu,
    )


def _sir_transition_mean_with_aux_tf(
    points: tf.Tensor,
    *,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
    substeps: int,
    step_size: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    state = tf.cast(points, DTYPE)
    half = tf.constant(0.5, dtype=DTYPE)
    substeps_tensor = tf.constant(int(substeps), dtype=tf.int32)
    state_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k1_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k2_input_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k2_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k3_input_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k3_ta = tf.TensorArray(DTYPE, size=substeps_tensor)
    k4_input_ta = tf.TensorArray(DTYPE, size=substeps_tensor)

    def body(
        index: tf.Tensor,
        running_state: tf.Tensor,
        state_acc: tf.TensorArray,
        k1_acc: tf.TensorArray,
        k2_input_acc: tf.TensorArray,
        k2_acc: tf.TensorArray,
        k3_input_acc: tf.TensorArray,
        k3_acc: tf.TensorArray,
        k4_input_acc: tf.TensorArray,
    ):
        start = running_state
        k1 = _sir_rhs_tf(
            start,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k2_input = start + half * step_size * k1
        k2 = _sir_rhs_tf(
            k2_input,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k3_input = start + half * step_size * k2
        k3 = _sir_rhs_tf(
            k3_input,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        k4_input = start + half * step_size * k3
        k4 = _sir_rhs_tf(
            k4_input,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        state = start + (step_size / tf.constant(6.0, dtype=DTYPE)) * (
            k1
            + tf.constant(2.0, dtype=DTYPE) * k2
            + tf.constant(2.0, dtype=DTYPE) * k3
            + k4
        )
        return (
            index + 1,
            state,
            state_acc.write(index, start),
            k1_acc.write(index, k1),
            k2_input_acc.write(index, k2_input),
            k2_acc.write(index, k2),
            k3_input_acc.write(index, k3_input),
            k3_acc.write(index, k3),
            k4_input_acc.write(index, k4_input),
        )

    (
        _,
        state,
        state_ta,
        k1_ta,
        k2_input_ta,
        k2_ta,
        k3_input_ta,
        k3_ta,
        k4_input_ta,
    ) = tf.while_loop(
        lambda index, *_: index < substeps_tensor,
        body,
        (
            tf.constant(0, dtype=tf.int32),
            state,
            state_ta,
            k1_ta,
            k2_input_ta,
            k2_ta,
            k3_input_ta,
            k3_ta,
            k4_input_ta,
        ),
    )
    return state, {
        "state": state_ta.stack(),
        "k1": k1_ta.stack(),
        "k2_input": k2_input_ta.stack(),
        "k2": k2_ta.stack(),
        "k3_input": k3_input_ta.stack(),
        "k3": k3_ta.stack(),
        "k4_input": k4_input_ta.stack(),
    }


def _sir_transition_mean_vjp_tf(
    aux: dict[str, tf.Tensor],
    upstream: tf.Tensor,
    *,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    adjacency: tf.Tensor,
    neighbor_degree: tf.Tensor,
    step_size: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    bar_state = tf.convert_to_tensor(upstream, dtype=DTYPE)
    batch_size = tf.shape(bar_state)[0]
    compartments = tf.shape(kappa)[0]
    bar_kappa = tf.zeros([batch_size, compartments], dtype=DTYPE)
    bar_nu = tf.zeros([batch_size, compartments], dtype=DTYPE)
    half = tf.constant(0.5, dtype=DTYPE)
    sixth = step_size / tf.constant(6.0, dtype=DTYPE)
    third = step_size / tf.constant(3.0, dtype=DTYPE)

    def body(
        index: tf.Tensor,
        running_bar_state: tf.Tensor,
        running_bar_kappa: tf.Tensor,
        running_bar_nu: tf.Tensor,
    ):
        bar_start = running_bar_state
        bar_k1 = sixth * running_bar_state
        bar_k2 = third * running_bar_state
        bar_k3 = third * running_bar_state
        bar_k4 = sixth * running_bar_state

        bar_k4_input, d_kappa, d_nu = _sir_rhs_vjp_tf(
            aux["k4_input"][index],
            bar_k4,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        bar_kappa_step = running_bar_kappa + d_kappa
        bar_nu_step = running_bar_nu + d_nu
        bar_start += bar_k4_input
        bar_k3 += half * step_size * bar_k4_input

        bar_k3_input, d_kappa, d_nu = _sir_rhs_vjp_tf(
            aux["k3_input"][index],
            bar_k3,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        bar_kappa_step += d_kappa
        bar_nu_step += d_nu
        bar_start += bar_k3_input
        bar_k2 += half * step_size * bar_k3_input

        bar_k2_input, d_kappa, d_nu = _sir_rhs_vjp_tf(
            aux["k2_input"][index],
            bar_k2,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        bar_kappa_step += d_kappa
        bar_nu_step += d_nu
        bar_start += bar_k2_input
        bar_k1 += half * step_size * bar_k2_input

        bar_k1_input, d_kappa, d_nu = _sir_rhs_vjp_tf(
            aux["state"][index],
            bar_k1,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
        )
        bar_kappa_step += d_kappa
        bar_nu_step += d_nu
        return index - 1, bar_start + bar_k1_input, bar_kappa_step, bar_nu_step

    substeps_tensor = tf.shape(aux["state"])[0]
    _, bar_state, bar_kappa, bar_nu = tf.while_loop(
        lambda index, *_: index >= 0,
        body,
        (substeps_tensor - 1, bar_state, bar_kappa, bar_nu),
    )
    return bar_state, bar_kappa, bar_nu


def _manual_transport_vjp_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
    upstream_particles: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    if args.transport_plan_mode != "streaming":
        raise ValueError("manual reverse route supports streaming transport only")
    if args.transport_ad_mode not in {"stabilized", "full"}:
        raise ValueError("manual reverse route requires transport_ad_mode='stabilized' or 'full'")
    allowed = {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }
    if args.transport_gradient_mode not in allowed:
        raise ValueError("manual reverse route requires a manual streaming transport gradient mode")
    if args.transport_policy == "no-resampling":
        return tf.zeros_like(post_flow), tf.zeros_like(normalized_log_weights)

    active_upstream = tf.where(mask[:, None, None], upstream_particles, tf.zeros_like(upstream_particles))
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=DTYPE)
    scaling = tf.convert_to_tensor(args.annealed_scaling, dtype=DTYPE)
    steps = core_tf._manual_dense_finite_steps(args.sinkhorn_iterations)  # noqa: SLF001
    if args.transport_ad_mode == "full":
        center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
        scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
        scaled_x = (post_flow - center) / scale[:, None, None]
        epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
        (
            d_scaled_x,
            d_particles,
            d_logw,
            d_epsilon0,
        ) = annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_pullback(  # noqa: SLF001
            scaled_x,
            post_flow,
            normalized_log_weights,
            epsilon,
            epsilon0,
            scaling,
            active_upstream,
            steps=steps,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
        d_scaled_x += _filterflow_epsilon_start_vjp(scaled_x, d_epsilon0)
        d_post_flow = d_particles + _scaled_centered_particles_vjp(
            post_flow,
            center,
            scale,
            d_scaled_x,
        )
        return (
            tf.convert_to_tensor(d_post_flow, dtype=DTYPE),
            tf.convert_to_tensor(d_logw, dtype=DTYPE),
        )

    center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
    scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
    scaled_x = (post_flow - center) / scale[:, None, None]
    epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
    float_n = tf.cast(tf.shape(post_flow)[1], DTYPE)
    uniform_log_weight = -tf.math.log(float_n) * tf.ones_like(normalized_log_weights)
    alpha, beta = (
        annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_stopped_scale_keys(  # noqa: SLF001
            normalized_log_weights,
            uniform_log_weight,
            scaled_x,
            epsilon,
            epsilon0,
            scaling,
            steps=steps,
            row_chunk_size=args.row_chunk_size,
            col_chunk_size=args.col_chunk_size,
        )
    )
    (
        d_scaled_x_transport,
        d_particles,
        d_alpha,
        d_beta,
        d_logw_transport,
    ) = annealed_transport_tf._filterflow_streaming_transport_from_potentials_vjp(  # noqa: SLF001
        scaled_x,
        post_flow,
        alpha,
        beta,
        epsilon,
        normalized_log_weights,
        float_n,
        active_upstream,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )
    (
        d_log_alpha,
        _d_log_beta,
        d_scaled_x_sinkhorn,
    ) = annealed_transport_tf._filterflow_streaming_finite_sinkhorn_potentials_vjp_stopped_scale_keys(  # noqa: SLF001
        normalized_log_weights,
        uniform_log_weight,
        scaled_x,
        d_alpha,
        d_beta,
        epsilon,
        epsilon0,
        scaling,
        steps=steps,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
    )
    d_scaled_x = d_scaled_x_transport + d_scaled_x_sinkhorn
    d_post_flow = d_particles + d_scaled_x / scale[:, None, None]
    d_logw = d_logw_transport + d_log_alpha
    return d_post_flow, d_logw


def _filterflow_epsilon_start_vjp(
    scaled_x: tf.Tensor,
    d_epsilon0: tf.Tensor,
) -> tf.Tensor:
    scaled_x = tf.convert_to_tensor(scaled_x, dtype=DTYPE)
    d_epsilon0 = tf.reshape(tf.convert_to_tensor(d_epsilon0, dtype=DTYPE), [-1])
    max_value = tf.reduce_max(scaled_x, axis=[1, 2])
    min_value = tf.reduce_min(scaled_x, axis=[1, 2])
    coordinate_range = max_value - min_value
    active = tf.cast(
        coordinate_range * coordinate_range >= tf.constant(1.0e-6, DTYPE),
        DTYPE,
    )
    max_mask = tf.cast(scaled_x == max_value[:, None, None], DTYPE)
    min_mask = tf.cast(scaled_x == min_value[:, None, None], DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=[1, 2], keepdims=True)
    min_count = tf.reduce_sum(min_mask, axis=[1, 2], keepdims=True)
    common = 2.0 * coordinate_range * d_epsilon0 * active
    return (
        common[:, None, None] * max_mask / max_count
        - common[:, None, None] * min_mask / min_count
    )


def _scaled_centered_particles_vjp(
    particles: tf.Tensor,
    center: tf.Tensor,
    scale: tf.Tensor,
    d_scaled_x: tf.Tensor,
) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, dtype=DTYPE)
    scale = tf.reshape(tf.convert_to_tensor(scale, dtype=DTYPE), [-1])
    d_scaled_x = tf.convert_to_tensor(d_scaled_x, dtype=DTYPE)
    centered = particles - center
    d_centered = d_scaled_x / scale[:, None, None]
    d_scale = -tf.reduce_sum(
        d_scaled_x * centered / (scale[:, None, None] * scale[:, None, None]),
        axis=[1, 2],
    )
    return (
        d_centered
        - tf.reduce_mean(d_centered, axis=1, keepdims=True)
        + _filterflow_scale_vjp(particles, d_scale)
    )


def _filterflow_scale_vjp(
    particles: tf.Tensor,
    d_scale: tf.Tensor,
) -> tf.Tensor:
    particles = tf.convert_to_tensor(particles, dtype=DTYPE)
    d_scale = tf.reshape(tf.convert_to_tensor(d_scale, dtype=DTYPE), [-1])
    num_particles = tf.cast(tf.shape(particles)[1], DTYPE)
    state_dim = tf.shape(particles)[2]
    dimension = tf.cast(state_dim, DTYPE)
    mean = tf.reduce_mean(particles, axis=1, keepdims=True)
    centered = particles - mean
    variance = tf.reduce_mean(centered * centered, axis=1)
    std = tf.sqrt(variance)
    diameter = tf.reduce_max(std, axis=1)
    active_diameter = tf.cast(diameter != 0.0, DTYPE)
    max_mask = tf.cast(std == diameter[:, None], DTYPE)
    max_count = tf.reduce_sum(max_mask, axis=1, keepdims=True)
    d_std = (
        d_scale[:, None]
        * tf.sqrt(dimension)
        * active_diameter[:, None]
        * max_mask
        / max_count
    )
    safe_std = tf.where(std > 0.0, std, tf.ones_like(std))
    d_variance = tf.where(std > 0.0, d_std / (2.0 * safe_std), tf.zeros_like(d_std))
    return (2.0 / num_particles) * centered * d_variance[:, None, :]


def _manual_forward_transport_tf(
    *,
    post_flow: tf.Tensor,
    normalized_log_weights: tf.Tensor,
    mask: tf.Tensor,
    args: argparse.Namespace,
) -> tuple[tf.Tensor, tf.Tensor]:
    if args.transport_plan_mode != "streaming":
        raise ValueError("manual reverse route supports streaming transport only")
    if args.transport_ad_mode not in {"stabilized", "full"}:
        raise ValueError("manual reverse route requires transport_ad_mode='stabilized' or 'full'")
    if args.transport_gradient_mode not in {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("manual reverse route requires a manual streaming transport gradient mode")
    if args.transport_policy == "no-resampling":
        return post_flow, normalized_log_weights

    batch_size, num_particles, _state_dim = core_tf._static_shape(  # noqa: SLF001
        post_flow,
        "post_flow",
    )
    center = tf.reduce_mean(post_flow, axis=1, keepdims=True)
    scale = annealed_transport_tf._filterflow_scale(post_flow)  # noqa: SLF001
    if args.transport_ad_mode == "stabilized":
        center = tf.stop_gradient(center)
        scale = tf.stop_gradient(scale)
    scaled_x = (post_flow - center) / scale[:, None, None]
    epsilon = tf.convert_to_tensor(args.sinkhorn_epsilon, dtype=DTYPE)
    epsilon0 = annealed_transport_tf._filterflow_epsilon_start(scaled_x)  # noqa: SLF001
    if args.transport_ad_mode == "stabilized":
        epsilon0 = tf.stop_gradient(epsilon0)
    scaling = tf.convert_to_tensor(args.annealed_scaling, dtype=DTYPE)
    steps = core_tf._manual_dense_finite_steps(args.sinkhorn_iterations)  # noqa: SLF001
    if args.transport_ad_mode == "full":
        transported, _row_residual = (
            annealed_transport_tf._filterflow_manual_streaming_finite_transport_total_vjp(  # noqa: SLF001
                scaled_x,
                post_flow,
                normalized_log_weights,
                epsilon,
                epsilon0,
                scaling,
                steps=steps,
                row_chunk_size=args.row_chunk_size,
                col_chunk_size=args.col_chunk_size,
            )
        )
    elif args.transport_gradient_mode == core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE:
        transported, _row_residual = (
            annealed_transport_tf._filterflow_manual_streaming_blockwise_vjp_finite_transport_stopped_scale_keys(  # noqa: SLF001
                scaled_x,
                post_flow,
                normalized_log_weights,
                epsilon,
                epsilon0,
                scaling,
                steps=steps,
                row_chunk_size=args.row_chunk_size,
                col_chunk_size=args.col_chunk_size,
            )
        )
    else:
        transported, _row_residual = (
            annealed_transport_tf._filterflow_manual_streaming_finite_transport_stopped_scale_keys(  # noqa: SLF001
                scaled_x,
                post_flow,
                normalized_log_weights,
                epsilon,
                epsilon0,
                scaling,
                steps=steps,
                row_chunk_size=args.row_chunk_size,
                col_chunk_size=args.col_chunk_size,
            )
        )
    uniform_log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    next_particles = tf.where(mask[:, None, None], transported, post_flow)
    next_log_weights = tf.where(mask[:, None], uniform_log_weights, normalized_log_weights)
    return next_particles, next_log_weights


def _theta_score_from_parameter_cotangents(
    *,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    observation_covariance: tf.Tensor,
    bar_kappa: tf.Tensor | None = None,
    bar_nu: tf.Tensor | None = None,
    bar_observation_covariance: tf.Tensor | None = None,
) -> tf.Tensor:
    batch_size = (
        tf.shape(bar_kappa)[0]
        if bar_kappa is not None
        else tf.shape(bar_nu)[0]
        if bar_nu is not None
        else tf.shape(bar_observation_covariance)[0]
    )
    score = tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE)
    if bar_kappa is not None:
        score += tf.stack(
            [
                tf.reduce_sum(bar_kappa * kappa[tf.newaxis, :], axis=1),
                tf.zeros([batch_size], dtype=DTYPE),
                tf.zeros([batch_size], dtype=DTYPE),
            ],
            axis=1,
        )
    if bar_nu is not None:
        score += tf.stack(
            [
                tf.zeros([batch_size], dtype=DTYPE),
                tf.reduce_sum(bar_nu * nu[tf.newaxis, :], axis=1),
                tf.zeros([batch_size], dtype=DTYPE),
            ],
            axis=1,
        )
    if bar_observation_covariance is not None:
        score += tf.stack(
            [
                tf.zeros([batch_size], dtype=DTYPE),
                tf.zeros([batch_size], dtype=DTYPE),
                tf.reduce_sum(
                    bar_observation_covariance
                    * tf.constant(2.0, dtype=DTYPE)
                    * observation_covariance,
                    axis=[1, 2],
                ),
            ],
            axis=1,
        )
    return score


def _regional_kappa_score_from_cotangent(
    *,
    kappa: tf.Tensor,
    bar_kappa: tf.Tensor,
) -> tf.Tensor:
    """Per-region chain-rule contribution to scalar log-kappa score."""

    return tf.convert_to_tensor(bar_kappa, dtype=DTYPE) * tf.convert_to_tensor(
        kappa,
        dtype=DTYPE,
    )[tf.newaxis, :]


def _regional_nu_score_from_cotangent(
    *,
    nu: tf.Tensor,
    bar_nu: tf.Tensor,
) -> tf.Tensor:
    """Per-region chain-rule contribution to scalar log-nu score."""

    return tf.convert_to_tensor(bar_nu, dtype=DTYPE) * tf.convert_to_tensor(
        nu,
        dtype=DTYPE,
    )[tf.newaxis, :]


def _add_manual_score_component(
    component_scores: tf.Tensor,
    component_name: str,
    score: tf.Tensor,
) -> tf.Tensor:
    component_index = MANUAL_SCORE_COMPONENT_NAMES.index(component_name)
    selector = tf.one_hot(
        component_index,
        len(MANUAL_SCORE_COMPONENT_NAMES),
        dtype=DTYPE,
    )
    return component_scores + selector[:, tf.newaxis, tf.newaxis] * score[tf.newaxis, :, :]


def _add_regional_kappa_score_component(
    component_scores: tf.Tensor,
    component_name: str,
    regional_score: tf.Tensor,
) -> tf.Tensor:
    component_index = MANUAL_SCORE_COMPONENT_NAMES.index(component_name)
    selector = tf.one_hot(
        component_index,
        len(MANUAL_SCORE_COMPONENT_NAMES),
        dtype=DTYPE,
    )
    return (
        component_scores
        + selector[:, tf.newaxis, tf.newaxis]
        * tf.convert_to_tensor(regional_score, dtype=DTYPE)[tf.newaxis, :, :]
    )


def _add_regional_nu_score_component(
    component_scores: tf.Tensor,
    component_name: str,
    regional_score: tf.Tensor,
) -> tf.Tensor:
    component_index = MANUAL_SCORE_COMPONENT_NAMES.index(component_name)
    selector = tf.one_hot(
        component_index,
        len(MANUAL_SCORE_COMPONENT_NAMES),
        dtype=DTYPE,
    )
    return (
        component_scores
        + selector[:, tf.newaxis, tf.newaxis]
        * tf.convert_to_tensor(regional_score, dtype=DTYPE)[tf.newaxis, :, :]
    )


def _make_parameterized_callbacks(
    *,
    tensors: dict[str, tf.Tensor],
    seeds: list[int],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
):
    scaled = _scaled_parameters(theta_components)
    return _make_sir_callbacks_from_scaled_parameters(
        tensors=tensors,
        seeds=seeds,
        args=args,
        kappa=scaled["kappa"],
        nu=scaled["nu"],
        observation_covariance=scaled["observation_covariance"],
    )


def _make_sir_callbacks_from_scaled_parameters(
    *,
    tensors: dict[str, tf.Tensor],
    seeds: list[int],
    args: argparse.Namespace,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    observation_covariance: tf.Tensor,
):
    kappa = tf.convert_to_tensor(kappa, dtype=DTYPE)
    nu = tf.convert_to_tensor(nu, dtype=DTYPE)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=DTYPE)
    adjacency = tf.cast(_SIR_ADJACENCY_MATRIX, DTYPE)
    neighbor_degree = tf.cast(_SIR_NEIGHBOR_DEGREE, DTYPE)
    substeps = int(_SIR_RK4_SUBSTEPS)
    step_size = tf.cast(_SIR_DELTA, DTYPE) / tf.cast(substeps, DTYPE)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    process_chol = tf.linalg.cholesky(transition_covariance[0])
    selector = tf.cast(_SIR_INFECTIOUS_SELECTOR, DTYPE)
    batch_size = len(seeds)
    num_particles = args.num_particles
    transition_noise = tf.cast(tensors["transition_noise"], DTYPE)
    if int(transition_noise.shape[0]) != batch_size:
        raise ValueError("transition_noise batch dimension must match batch_seeds")
    if int(transition_noise.shape[2]) != num_particles:
        raise ValueError("transition_noise particle dimension must match num_particles")
    kappa_batch = _batch_vector_parameter(kappa)
    nu_batch = _batch_vector_parameter(nu)
    observation_covariance_batch = _batch_matrix_parameter(
        observation_covariance,
        batch_size,
    )

    def apply_sir_process_noise_policy(points: tf.Tensor) -> tf.Tensor:
        susceptible = tf.maximum(points[:, :, 0::2], tf.constant(0.0, dtype=DTYPE))
        infectious = points[:, :, 1::2]
        reshaped = tf.reshape(
            tf.stack([susceptible, infectious], axis=3),
            tf.shape(points),
        )
        reshaped.set_shape(points.shape)
        return reshaped

    def sir_rhs(points: tf.Tensor) -> tf.Tensor:
        susceptible = points[:, :, 0::2]
        infectious = points[:, :, 1::2]
        susceptible_neighbor = (
            tf.linalg.matmul(susceptible, adjacency, transpose_b=True)
            - susceptible * neighbor_degree[tf.newaxis, tf.newaxis, :]
        )
        infectious_neighbor = (
            tf.linalg.matmul(infectious, adjacency, transpose_b=True)
            - infectious * neighbor_degree[tf.newaxis, tf.newaxis, :]
        )
        infection = kappa_batch * susceptible * infectious
        d_susceptible = -infection + tf.constant(0.5, dtype=DTYPE) * susceptible_neighbor
        d_infectious = (
            infection
            - nu_batch * infectious
            + tf.constant(0.5, dtype=DTYPE) * infectious_neighbor
        )
        reshaped = tf.reshape(
            tf.stack([d_susceptible, d_infectious], axis=3),
            tf.shape(points),
        )
        reshaped.set_shape(points.shape)
        return reshaped

    def transition_mean(points: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        del time_index
        state = tf.cast(points, DTYPE)
        for _ in range(substeps):
            k1 = sir_rhs(state)
            k2 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k1)
            k3 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k2)
            k4 = sir_rhs(state + tf.constant(0.5, dtype=DTYPE) * step_size * k3)
            state = state + (step_size / tf.constant(6.0, dtype=DTYPE)) * (
                k1
                + tf.constant(2.0, dtype=DTYPE) * k2
                + tf.constant(2.0, dtype=DTYPE) * k3
                + k4
            )
        return state

    def pre_flow_step(points: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
        mean = transition_mean(points, time_index)
        noise_tensor = tf.gather(transition_noise, time_index, axis=1)
        pushed = mean + tf.einsum("bnd,ed->bne", noise_tensor, process_chol)
        return apply_sir_process_noise_policy(pushed)

    def observation_fn(points: tf.Tensor) -> tf.Tensor:
        return tf.gather(points, list(range(1, 18, 2)), axis=2)

    def observation_jacobian_fn(points: tf.Tensor) -> tf.Tensor:
        chunk_particles = points.shape[1]
        if chunk_particles is None:
            raise ValueError("P8p SIR gradient diagnostic requires static chunk dimension")
        return tf.tile(
            selector[tf.newaxis, tf.newaxis, :, :],
            [batch_size, int(chunk_particles), 1, 1],
        )

    def observation_residual_fn(h_ref: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        return observation[tf.newaxis, tf.newaxis, :] - h_ref

    def transition_log_density_fn(
        x_next: tf.Tensor,
        x_prev: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        residual = x_next - transition_mean(x_prev, time_index)
        return _batched_gaussian_logpdf(
            residual,
            tf.cast(tensors["transition_covariance"], DTYPE),
        )

    def observation_log_density_fn(
        x: tf.Tensor,
        observation: tf.Tensor,
        time_index: tf.Tensor,
    ) -> tf.Tensor:
        del time_index
        residual = observation_fn(x) - observation[tf.newaxis, tf.newaxis, :]
        return _batched_gaussian_logpdf(residual, observation_covariance_batch)

    return {
        "prior_mean_fn": transition_mean,
        "pre_flow_step_fn": pre_flow_step,
        "observation_fn": observation_fn,
        "observation_jacobian_fn": observation_jacobian_fn,
        "observation_residual_fn": observation_residual_fn,
        "transition_log_density_fn": transition_log_density_fn,
        "observation_log_density_fn": observation_log_density_fn,
        "observation_covariance": observation_covariance_batch,
    }


def _value_core(
    *,
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> streaming_tf.StreamingLEDHPFPFOTValueTensors:
    callbacks = _make_parameterized_callbacks(
        tensors=tensors,
        seeds=args.batch_seeds,
        args=args,
        theta_components=theta_components,
    )
    return streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=tensors["transition_matrix"],
        transition_covariance=tensors["transition_covariance"],
        observation_covariance=callbacks["observation_covariance"],
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
        transport_gradient_mode=args.transport_gradient_mode,
        transport_plan_mode=args.transport_plan_mode,
        transport_ad_mode=args.transport_ad_mode,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=False,
    )


def _objective_from_components(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor]:
    value = _value_core(
        tensors=tensors,
        args=args,
        theta_components=theta_components,
    ).log_likelihood
    objective = tf.reduce_mean(value)
    return objective, value


def _manual_value_and_score_from_components(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    *,
    return_score_decomposition: bool = False,
    return_regional_kappa_decomposition: bool = False,
    return_regional_nu_decomposition: bool = False,
) -> dict[str, tf.Tensor]:
    """Manual fixed-branch value and score using TensorFlow time-scan state."""

    if (
        return_regional_kappa_decomposition or return_regional_nu_decomposition
    ) and not return_score_decomposition:
        raise ValueError(
            "regional decomposition requires return_score_decomposition=True "
            "so component labels are unambiguous"
        )
    if args.transport_plan_mode != "streaming":
        raise ValueError("manual reverse route supports streaming transport only")
    if args.transport_ad_mode not in {"stabilized", "full"}:
        raise ValueError("manual reverse route requires transport_ad_mode='stabilized' or 'full'")
    if args.transport_gradient_mode not in {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("manual reverse route requires a manual streaming transport gradient mode")

    scaled = _scaled_parameters(theta_components)
    kappa = tf.convert_to_tensor(scaled["kappa"], dtype=DTYPE)
    nu = tf.convert_to_tensor(scaled["nu"], dtype=DTYPE)
    observation_covariance = _batch_matrix_parameter(
        scaled["observation_covariance"],
        len(args.batch_seeds),
    )
    adjacency = tf.cast(_SIR_ADJACENCY_MATRIX, DTYPE)
    neighbor_degree = tf.cast(_SIR_NEIGHBOR_DEGREE, DTYPE)
    substeps = int(_SIR_RK4_SUBSTEPS)
    step_size = tf.cast(_SIR_DELTA, DTYPE) / tf.cast(substeps, DTYPE)
    process_chol = tf.linalg.cholesky(
        tf.cast(tensors["transition_covariance"], DTYPE)
    )
    selector = tf.cast(_SIR_INFECTIOUS_SELECTOR, DTYPE)
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    transition_noise = tf.cast(tensors["transition_noise"], DTYPE)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    batch_size, num_particles, state_dim = core_tf._static_shape(  # noqa: SLF001
        particles,
        "initial_particles",
    )
    time_steps_tensor = tf.shape(observations)[0]
    if state_dim != 18:
        raise ValueError("manual P8p SIR route expects state_dim=18")
    if int(transition_noise.shape[0]) != batch_size:
        raise ValueError("transition_noise batch dimension must match initial_particles")
    if int(transition_noise.shape[2]) != num_particles:
        raise ValueError("transition_noise particle dimension must match initial_particles")
    if int(transition_noise.shape[3]) != state_dim:
        raise ValueError("transition_noise state dimension must match initial_particles")

    h_jac_full = tf.tile(
        selector[tf.newaxis, tf.newaxis, :, :],
        [batch_size, num_particles, 1, 1],
    )

    scalar_keys = (
        "prior_means",
        "post_flow",
        "predicted_observation",
        "observation",
        "corrected_log_weights",
        "normalized_log_weights",
    )
    bool_keys = ("pushed_susceptible_positive", "mask")
    transition_keys = (
        "state",
        "k1",
        "k2_input",
        "k2",
        "k3_input",
        "k3",
        "k4_input",
    )
    flow_keys = (
        "x0",
        "prior_means",
        "observation_jacobian",
        "observation_residual",
        "transition_covariance",
        "observation_covariance",
        "transition_covariance_stable",
        "observation_covariance_stable",
        "prior_chol",
        "prior_precision",
        "obs_precision",
        "pseudo_observation",
        "post_precision",
        "post_precision_stable",
        "post_covariance_unstabilized",
        "post_covariance",
        "post_chol",
        "prior_inv",
        "affine_transform",
        "delta",
        "info",
    )
    scalar_tas = {
        key: tf.TensorArray(DTYPE, size=time_steps_tensor, infer_shape=False)
        for key in scalar_keys
    }
    bool_tas = {
        key: tf.TensorArray(tf.bool, size=time_steps_tensor, infer_shape=False)
        for key in bool_keys
    }
    transition_tas = {
        key: tf.TensorArray(DTYPE, size=time_steps_tensor, infer_shape=False)
        for key in transition_keys
    }
    flow_tas = {
        key: tf.TensorArray(DTYPE, size=time_steps_tensor, infer_shape=False)
        for key in flow_keys
    }

    def write_tensor_dict(tas, index, values):
        return {key: tas[key].write(index, values[key]) for key in tas}

    def forward_body(
        time_index: tf.Tensor,
        running_particles: tf.Tensor,
        running_log_weights: tf.Tensor,
        running_log_likelihood: tf.Tensor,
        running_scalar_tas,
        running_bool_tas,
        running_transition_tas,
        running_flow_tas,
    ):
        observation = observations[time_index]
        prior_means, transition_aux = _sir_transition_mean_with_aux_tf(
            running_particles,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
            substeps=substeps,
            step_size=step_size,
        )
        noise_tensor = transition_noise[:, time_index, :, :]
        pushed = prior_means + tf.einsum("bnd,bed->bne", noise_tensor, process_chol)
        pushed_susceptible = pushed[:, :, 0::2]
        pre_flow = _sir_flatten_components(
            tf.maximum(pushed_susceptible, tf.constant(0.0, dtype=DTYPE)),
            pushed[:, :, 1::2],
        )
        h_ref = tf.gather(pre_flow, _SIR_INFECTIOUS_INDICES, axis=2)
        residual = observation[tf.newaxis, tf.newaxis, :] - h_ref
        flow, flow_aux = core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_flow,
            prior_means=prior_means,
            observation_jacobian=h_jac_full,
            observation_residual=residual,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - prior_means,
            transition_covariance,
        )
        predicted_observation = tf.gather(post_flow, _SIR_INFECTIOUS_INDICES, axis=2)
        observation_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            predicted_observation - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
        )
        corrected_log_weights = (
            running_log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
        )
        mask = fixed_resampling_mask[:, time_index]
        next_particles, next_log_weights = _manual_forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
        )
        scalar_values = {
            "prior_means": prior_means,
            "post_flow": post_flow,
            "predicted_observation": predicted_observation,
            "observation": observation,
            "corrected_log_weights": corrected_log_weights,
            "normalized_log_weights": normalized_log_weights,
        }
        bool_values = {
            "pushed_susceptible_positive": pushed_susceptible > 0.0,
            "mask": mask,
        }
        flow_values = {key: getattr(flow_aux, key) for key in flow_keys}
        return (
            time_index + 1,
            next_particles,
            next_log_weights,
            running_log_likelihood + incremental,
            write_tensor_dict(running_scalar_tas, time_index, scalar_values),
            write_tensor_dict(running_bool_tas, time_index, bool_values),
            write_tensor_dict(running_transition_tas, time_index, transition_aux),
            write_tensor_dict(running_flow_tas, time_index, flow_values),
        )

    (
        _,
        particles,
        log_weights,
        log_likelihood,
        scalar_tas,
        bool_tas,
        transition_tas,
        flow_tas,
    ) = tf.while_loop(
        lambda time_index, *_: time_index < time_steps_tensor,
        forward_body,
        (
            tf.constant(0, dtype=tf.int32),
            particles,
            core_tf.uniform_log_weights(batch_size, num_particles),
            tf.zeros([batch_size], dtype=DTYPE),
            scalar_tas,
            bool_tas,
            transition_tas,
            flow_tas,
        ),
    )

    component_scores = tf.zeros(
        [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, len(PARAMETER_NAMES)],
        dtype=DTYPE,
    )
    regional_kappa_components = tf.zeros(
        [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, tf.shape(kappa)[0]],
        dtype=DTYPE,
    )
    regional_nu_components = tf.zeros(
        [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, tf.shape(nu)[0]],
        dtype=DTYPE,
    )

    def reverse_body(
        time_index: tf.Tensor,
        running_bar_particles: tf.Tensor,
        running_bar_log_weights: tf.Tensor,
        running_score: tf.Tensor,
        running_component_scores: tf.Tensor,
        running_regional_kappa: tf.Tensor,
        running_regional_nu: tf.Tensor,
    ):
        mask = bool_tas["mask"].read(time_index)
        mask.set_shape([batch_size])
        post_flow = scalar_tas["post_flow"].read(time_index)
        post_flow.set_shape([batch_size, num_particles, state_dim])
        normalized_log_weights = scalar_tas["normalized_log_weights"].read(time_index)
        normalized_log_weights.set_shape([batch_size, num_particles])
        bar_post_transport, bar_normalized_from_transport = _manual_transport_vjp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
            upstream_particles=running_bar_particles,
        )
        inactive = tf.logical_not(mask)
        bar_post = bar_post_transport + tf.where(
            inactive[:, None, None],
            running_bar_particles,
            tf.zeros_like(running_bar_particles),
        )
        bar_normalized_log_weights = bar_normalized_from_transport + tf.where(
            inactive[:, None],
            running_bar_log_weights,
            tf.zeros_like(running_bar_log_weights),
        )
        bar_corrected, _weights, _incremental, _floor_active = (
            core_tf._normalize_log_weights_with_floor_vjp(  # noqa: SLF001
                tf.ensure_shape(
                    scalar_tas["corrected_log_weights"].read(time_index),
                    [batch_size, num_particles],
                ),
                bar_normalized_log_weights,
                tf.ones([batch_size], dtype=DTYPE),
            )
        )
        correction_bars = core_tf._log_weight_correction_vjp(bar_corrected)  # noqa: SLF001
        next_bar_log_weights = correction_bars["current_log_weights"]

        prior_means = scalar_tas["prior_means"].read(time_index)
        prior_means.set_shape([batch_size, num_particles, state_dim])
        transition_vjp = core_tf._transition_gaussian_log_density_vjp(  # noqa: SLF001
            post_flow,
            prior_means,
            transition_covariance,
            correction_bars["transition_log_density"],
        )
        predicted_observation = scalar_tas["predicted_observation"].read(time_index)
        predicted_observation.set_shape([batch_size, num_particles, len(_SIR_INFECTIOUS_INDICES)])
        observation = scalar_tas["observation"].read(time_index)
        observation.set_shape([len(_SIR_INFECTIOUS_INDICES)])
        observation_vjp = core_tf._observation_gaussian_log_density_vjp(  # noqa: SLF001
            predicted_observation,
            observation,
            observation_covariance,
            correction_bars["observation_log_density"],
            residual_convention="model_minus_observation",
        )
        bar_post += transition_vjp["x_next"]
        bar_post += _sir_infectious_vjp(observation_vjp["predicted_observation"])
        observation_density_score = _theta_score_from_parameter_cotangents(
            kappa=kappa,
            nu=nu,
            observation_covariance=observation_covariance,
            bar_observation_covariance=observation_vjp["observation_covariance"],
        )
        next_score = running_score + observation_density_score
        next_component_scores = running_component_scores
        next_regional_kappa = running_regional_kappa
        next_regional_nu = running_regional_nu
        if return_score_decomposition:
            next_component_scores = _add_manual_score_component(
                next_component_scores,
                "observation_density_covariance",
                observation_density_score,
            )

        flow_values = {key: flow_tas[key].read(time_index) for key in flow_keys}
        for key in (
            "x0",
            "prior_means",
            "delta",
            "info",
        ):
            flow_values[key].set_shape([batch_size, num_particles, state_dim])
        for key in ("observation_residual", "pseudo_observation"):
            flow_values[key].set_shape([batch_size, num_particles, len(_SIR_INFECTIOUS_INDICES)])
        flow_values["observation_jacobian"].set_shape(
            [batch_size, num_particles, len(_SIR_INFECTIOUS_INDICES), state_dim]
        )
        for key in (
            "transition_covariance",
            "transition_covariance_stable",
            "prior_chol",
            "prior_precision",
            "prior_inv",
        ):
            flow_values[key].set_shape([batch_size, state_dim, state_dim])
        for key in ("observation_covariance", "observation_covariance_stable", "obs_precision"):
            flow_values[key].set_shape(
                [batch_size, len(_SIR_INFECTIOUS_INDICES), len(_SIR_INFECTIOUS_INDICES)]
            )
        for key in (
            "post_precision",
            "post_precision_stable",
            "post_covariance_unstabilized",
            "post_covariance",
            "post_chol",
            "affine_transform",
        ):
            flow_values[key].set_shape([batch_size, num_particles, state_dim, state_dim])
        flow_aux = core_tf._BatchedLEDHLinearizedFlowAux(  # noqa: SLF001
            **flow_values
        )
        flow_vjp = core_tf._batched_ledh_linearized_flow_vjp(  # noqa: SLF001
            flow_aux,
            bar_post,
            correction_bars["pre_flow_log_density"],
            correction_bars["forward_log_det"],
        )
        bar_pre_flow = flow_vjp.pre_flow_particles - _sir_infectious_vjp(
            flow_vjp.observation_residual
        )
        ledh_flow_observation_score = _theta_score_from_parameter_cotangents(
            kappa=kappa,
            nu=nu,
            observation_covariance=observation_covariance,
            bar_observation_covariance=flow_vjp.observation_covariance,
        )
        next_score += ledh_flow_observation_score
        if return_score_decomposition:
            next_component_scores = _add_manual_score_component(
                next_component_scores,
                "ledh_flow_observation_covariance",
                ledh_flow_observation_score,
            )

        pushed_mask = bool_tas["pushed_susceptible_positive"].read(time_index)
        pushed_mask.set_shape([batch_size, num_particles, len(_SIR_INFECTIOUS_INDICES)])
        bar_pushed = _sir_flatten_components(
            tf.where(pushed_mask, bar_pre_flow[:, :, 0::2], tf.zeros_like(bar_pre_flow[:, :, 0::2])),
            bar_pre_flow[:, :, 1::2],
        )
        transition_aux = {key: transition_tas[key].read(time_index) for key in transition_keys}
        for value in transition_aux.values():
            value.set_shape([int(_SIR_RK4_SUBSTEPS), batch_size, num_particles, state_dim])
        if return_score_decomposition:
            (
                bar_particles_transition_density,
                bar_kappa_transition_density,
                bar_nu_transition_density,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,
                transition_vjp["transition_mean"],
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            transition_density_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_transition_density,
                bar_nu=bar_nu_transition_density,
            )
            next_score += transition_density_score
            if return_regional_kappa_decomposition:
                next_regional_kappa = _add_regional_kappa_score_component(
                    next_regional_kappa,
                    "transition_mean_from_transition_density",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_transition_density,
                    ),
                )
            if return_regional_nu_decomposition:
                next_regional_nu = _add_regional_nu_score_component(
                    next_regional_nu,
                    "transition_mean_from_transition_density",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_transition_density,
                    ),
                )
            next_component_scores = _add_manual_score_component(
                next_component_scores,
                "transition_mean_from_transition_density",
                transition_density_score,
            )

            (
                bar_particles_ledh_flow_prior,
                bar_kappa_ledh_flow_prior,
                bar_nu_ledh_flow_prior,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,
                flow_vjp.prior_means,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            ledh_flow_prior_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_ledh_flow_prior,
                bar_nu=bar_nu_ledh_flow_prior,
            )
            next_score += ledh_flow_prior_score
            if return_regional_kappa_decomposition:
                next_regional_kappa = _add_regional_kappa_score_component(
                    next_regional_kappa,
                    "transition_mean_from_ledh_flow_prior",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_ledh_flow_prior,
                    ),
                )
            if return_regional_nu_decomposition:
                next_regional_nu = _add_regional_nu_score_component(
                    next_regional_nu,
                    "transition_mean_from_ledh_flow_prior",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_ledh_flow_prior,
                    ),
                )
            next_component_scores = _add_manual_score_component(
                next_component_scores,
                "transition_mean_from_ledh_flow_prior",
                ledh_flow_prior_score,
            )

            (
                bar_particles_pre_flow_clamp,
                bar_kappa_pre_flow_clamp,
                bar_nu_pre_flow_clamp,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,
                bar_pushed,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            pre_flow_clamp_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_pre_flow_clamp,
                bar_nu=bar_nu_pre_flow_clamp,
            )
            next_score += pre_flow_clamp_score
            if return_regional_kappa_decomposition:
                next_regional_kappa = _add_regional_kappa_score_component(
                    next_regional_kappa,
                    "transition_mean_from_pre_flow_clamp",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_pre_flow_clamp,
                    ),
                )
            if return_regional_nu_decomposition:
                next_regional_nu = _add_regional_nu_score_component(
                    next_regional_nu,
                    "transition_mean_from_pre_flow_clamp",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_pre_flow_clamp,
                    ),
                )
            next_component_scores = _add_manual_score_component(
                next_component_scores,
                "transition_mean_from_pre_flow_clamp",
                pre_flow_clamp_score,
            )
            next_bar_particles = (
                bar_particles_transition_density
                + bar_particles_ledh_flow_prior
                + bar_particles_pre_flow_clamp
            )
        else:
            bar_prior_means = (
                transition_vjp["transition_mean"]
                + flow_vjp.prior_means
                + bar_pushed
            )
            (
                next_bar_particles,
                bar_kappa,
                bar_nu,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,
                bar_prior_means,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            next_score += _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa,
                bar_nu=bar_nu,
            )
            if return_regional_kappa_decomposition:
                next_regional_kappa = _add_regional_kappa_score_component(
                    next_regional_kappa,
                    "transition_mean_from_transition_density",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa,
                    ),
                )
            if return_regional_nu_decomposition:
                next_regional_nu = _add_regional_nu_score_component(
                    next_regional_nu,
                    "transition_mean_from_transition_density",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu,
                    ),
                )

        return (
            time_index - 1,
            next_bar_particles,
            next_bar_log_weights,
            next_score,
            next_component_scores,
            next_regional_kappa,
            next_regional_nu,
        )

    (
        _,
        _bar_particles,
        _bar_log_weights,
        per_seed_score,
        component_scores,
        regional_kappa_components,
        regional_nu_components,
    ) = tf.while_loop(
        lambda time_index, *_: time_index >= 0,
        reverse_body,
        (
            time_steps_tensor - 1,
            tf.zeros_like(particles),
            tf.zeros_like(log_weights),
            tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE),
            component_scores,
            regional_kappa_components,
            regional_nu_components,
        ),
    )

    result = {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_score, axis=0),
        "per_seed_gradient": per_seed_score,
    }
    if return_score_decomposition:
        result["manual_score_components"] = component_scores
    if return_regional_kappa_decomposition:
        result["regional_kappa_score_components"] = regional_kappa_components
        result["regional_kappa_score_per_seed"] = tf.reduce_sum(
            regional_kappa_components,
            axis=0,
        )
    if return_regional_nu_decomposition:
        result["regional_nu_score_components"] = regional_nu_components
        result["regional_nu_score_per_seed"] = tf.reduce_sum(
            regional_nu_components,
            axis=0,
        )
    return result


def _manual_value_and_score_from_components_python_record_reference(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    *,
    return_score_decomposition: bool = False,
    return_regional_kappa_decomposition: bool = False,
    return_regional_nu_decomposition: bool = False,
) -> dict[str, tf.Tensor]:
    """Manual fixed-branch value and score for the P8p SIR LEDH route."""

    if (
        return_regional_kappa_decomposition or return_regional_nu_decomposition
    ) and not return_score_decomposition:
        raise ValueError(
            "regional decomposition requires return_score_decomposition=True "
            "so component labels are unambiguous"
        )
    if args.transport_plan_mode != "streaming":
        raise ValueError("manual reverse route supports streaming transport only")
    if args.transport_ad_mode not in {"stabilized", "full"}:
        raise ValueError("manual reverse route requires transport_ad_mode='stabilized' or 'full'")
    if args.transport_gradient_mode not in {
        core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        core_tf.MANUAL_STREAMING_BLOCKWISE_VJP_FINITE_TRANSPORT_GRADIENT_MODE,
    }:
        raise ValueError("manual reverse route requires a manual streaming transport gradient mode")

    scaled = _scaled_parameters(theta_components)
    kappa = tf.convert_to_tensor(scaled["kappa"], dtype=DTYPE)
    nu = tf.convert_to_tensor(scaled["nu"], dtype=DTYPE)
    observation_covariance = _batch_matrix_parameter(
        scaled["observation_covariance"],
        len(args.batch_seeds),
    )
    adjacency = tf.cast(_SIR_ADJACENCY_MATRIX, DTYPE)
    neighbor_degree = tf.cast(_SIR_NEIGHBOR_DEGREE, DTYPE)
    substeps = int(_SIR_RK4_SUBSTEPS)
    step_size = tf.cast(_SIR_DELTA, DTYPE) / tf.cast(substeps, DTYPE)
    process_chol = tf.linalg.cholesky(
        tf.cast(tensors["transition_covariance"], DTYPE)
    )
    selector = tf.cast(_SIR_INFECTIOUS_SELECTOR, DTYPE)
    observations = tf.cast(tensors["observations"], DTYPE)
    fixed_resampling_mask = tf.convert_to_tensor(tensors["fixed_resampling_mask"], dtype=tf.bool)
    transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
    transition_matrix = tf.cast(tensors["transition_matrix"], DTYPE)
    transition_noise = tf.cast(tensors["transition_noise"], DTYPE)
    particles = tf.cast(tensors["initial_particles"], DTYPE)
    batch_size, num_particles, state_dim = core_tf._static_shape(  # noqa: SLF001
        particles,
        "initial_particles",
    )
    time_steps = int(observations.shape[0])
    if state_dim != 18:
        raise ValueError("manual P8p SIR route expects state_dim=18")
    if int(transition_noise.shape[0]) != batch_size:
        raise ValueError("transition_noise batch dimension must match initial_particles")
    if int(transition_noise.shape[2]) != num_particles:
        raise ValueError("transition_noise particle dimension must match initial_particles")
    if int(transition_noise.shape[3]) != state_dim:
        raise ValueError("transition_noise state dimension must match initial_particles")
    log_weights = core_tf.uniform_log_weights(batch_size, num_particles)
    log_likelihood = tf.zeros([batch_size], dtype=DTYPE)
    records: list[dict[str, tf.Tensor | list[dict[str, tf.Tensor]]]] = []

    h_jac_full = tf.tile(
        selector[tf.newaxis, tf.newaxis, :, :],
        [batch_size, num_particles, 1, 1],
    )

    for time_index in range(time_steps):
        observation = observations[time_index]
        ancestors = particles
        current_log_weights = log_weights
        prior_means, transition_aux = _sir_transition_mean_with_aux_tf(
            ancestors,
            kappa=kappa,
            nu=nu,
            adjacency=adjacency,
            neighbor_degree=neighbor_degree,
            substeps=substeps,
            step_size=step_size,
        )
        noise_tensor = transition_noise[:, time_index, :, :]
        pushed = prior_means + tf.einsum("bnd,bed->bne", noise_tensor, process_chol)
        pushed_susceptible = pushed[:, :, 0::2]
        pre_flow = _sir_flatten_components(
            tf.maximum(pushed_susceptible, tf.constant(0.0, dtype=DTYPE)),
            pushed[:, :, 1::2],
        )
        h_ref = tf.gather(pre_flow, _SIR_INFECTIOUS_INDICES, axis=2)
        residual = observation[tf.newaxis, tf.newaxis, :] - h_ref
        flow, flow_aux = core_tf._batched_ledh_linearized_flow_with_aux_tf(  # noqa: SLF001
            pre_flow_particles=pre_flow,
            prior_means=prior_means,
            observation_jacobian=h_jac_full,
            observation_residual=residual,
            transition_covariance=transition_covariance,
            observation_covariance=observation_covariance,
        )
        post_flow = flow.post_flow_particles
        transition_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            post_flow - prior_means,
            transition_covariance,
        )
        predicted_observation = tf.gather(post_flow, _SIR_INFECTIOUS_INDICES, axis=2)
        observation_log_density = core_tf._batched_gaussian_logpdf(  # noqa: SLF001
            predicted_observation - observation[tf.newaxis, tf.newaxis, :],
            observation_covariance,
        )
        corrected_log_weights = (
            current_log_weights
            + transition_log_density
            + observation_log_density
            - flow.pre_flow_log_density
            + flow.forward_log_det
        )
        weights, incremental = core_tf._normalize_log_weights(corrected_log_weights)  # noqa: SLF001
        log_likelihood = log_likelihood + incremental
        normalized_log_weights = tf.math.log(
            tf.maximum(weights, core_tf._log_weight_floor())  # noqa: SLF001
        )
        mask = fixed_resampling_mask[:, time_index]
        particles, log_weights = _manual_forward_transport_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
        )
        records.append(
            {
                "ancestors": ancestors,
                "current_log_weights": current_log_weights,
                "prior_means": prior_means,
                "transition_aux": transition_aux,
                "pushed_susceptible_positive": pushed_susceptible > 0.0,
                "pre_flow": pre_flow,
                "flow_aux": flow_aux,
                "post_flow": post_flow,
                "predicted_observation": predicted_observation,
                "observation": observation,
                "corrected_log_weights": corrected_log_weights,
                "weights": weights,
                "normalized_log_weights": normalized_log_weights,
                "mask": mask,
            }
        )

    bar_particles = tf.zeros_like(particles)
    bar_log_weights = tf.zeros_like(log_weights)
    per_seed_score = tf.zeros([batch_size, len(PARAMETER_NAMES)], dtype=DTYPE)
    if return_regional_kappa_decomposition:
        regional_kappa_components = tf.zeros(
            [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, tf.shape(kappa)[0]],
            dtype=DTYPE,
        )
    if return_regional_nu_decomposition:
        regional_nu_components = tf.zeros(
            [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, tf.shape(nu)[0]],
            dtype=DTYPE,
        )
    if return_score_decomposition:
        component_scores = tf.zeros(
            [len(MANUAL_SCORE_COMPONENT_NAMES), batch_size, len(PARAMETER_NAMES)],
            dtype=DTYPE,
        )
    for record in reversed(records):
        mask = tf.convert_to_tensor(record["mask"], dtype=tf.bool)
        post_flow = tf.convert_to_tensor(record["post_flow"], dtype=DTYPE)
        normalized_log_weights = tf.convert_to_tensor(
            record["normalized_log_weights"],
            dtype=DTYPE,
        )
        bar_post_transport, bar_normalized_from_transport = _manual_transport_vjp_tf(
            post_flow=post_flow,
            normalized_log_weights=normalized_log_weights,
            mask=mask,
            args=args,
            upstream_particles=bar_particles,
        )
        inactive = tf.logical_not(mask)
        bar_post = bar_post_transport + tf.where(
            inactive[:, None, None],
            bar_particles,
            tf.zeros_like(bar_particles),
        )
        bar_normalized_log_weights = bar_normalized_from_transport + tf.where(
            inactive[:, None],
            bar_log_weights,
            tf.zeros_like(bar_log_weights),
        )
        bar_corrected, _weights, _incremental, _floor_active = (
            core_tf._normalize_log_weights_with_floor_vjp(  # noqa: SLF001
                tf.convert_to_tensor(record["corrected_log_weights"], dtype=DTYPE),
                bar_normalized_log_weights,
                tf.ones([batch_size], dtype=DTYPE),
            )
        )
        correction_bars = core_tf._log_weight_correction_vjp(bar_corrected)  # noqa: SLF001
        bar_log_weights = correction_bars["current_log_weights"]

        transition_vjp = core_tf._transition_gaussian_log_density_vjp(  # noqa: SLF001
            post_flow,
            tf.convert_to_tensor(record["prior_means"], dtype=DTYPE),
            transition_covariance,
            correction_bars["transition_log_density"],
        )
        observation_vjp = core_tf._observation_gaussian_log_density_vjp(  # noqa: SLF001
            tf.convert_to_tensor(record["predicted_observation"], dtype=DTYPE),
            tf.convert_to_tensor(record["observation"], dtype=DTYPE),
            observation_covariance,
            correction_bars["observation_log_density"],
            residual_convention="model_minus_observation",
        )
        bar_post += transition_vjp["x_next"]
        bar_post += _sir_infectious_vjp(observation_vjp["predicted_observation"])
        observation_density_score = _theta_score_from_parameter_cotangents(
            kappa=kappa,
            nu=nu,
            observation_covariance=observation_covariance,
            bar_observation_covariance=observation_vjp["observation_covariance"],
        )
        per_seed_score += observation_density_score
        if return_score_decomposition:
            component_scores = _add_manual_score_component(
                component_scores,
                "observation_density_covariance",
                observation_density_score,
            )

        flow_vjp = core_tf._batched_ledh_linearized_flow_vjp(  # noqa: SLF001
            record["flow_aux"],  # type: ignore[arg-type]
            bar_post,
            correction_bars["pre_flow_log_density"],
            correction_bars["forward_log_det"],
        )
        bar_pre_flow = flow_vjp.pre_flow_particles - _sir_infectious_vjp(
            flow_vjp.observation_residual
        )
        ledh_flow_observation_score = _theta_score_from_parameter_cotangents(
            kappa=kappa,
            nu=nu,
            observation_covariance=observation_covariance,
            bar_observation_covariance=flow_vjp.observation_covariance,
        )
        per_seed_score += ledh_flow_observation_score
        if return_score_decomposition:
            component_scores = _add_manual_score_component(
                component_scores,
                "ledh_flow_observation_covariance",
                ledh_flow_observation_score,
            )

        pushed_mask = tf.convert_to_tensor(
            record["pushed_susceptible_positive"],
            dtype=tf.bool,
        )
        bar_pushed = _sir_flatten_components(
            tf.where(pushed_mask, bar_pre_flow[:, :, 0::2], tf.zeros_like(bar_pre_flow[:, :, 0::2])),
            bar_pre_flow[:, :, 1::2],
        )
        transition_aux = record["transition_aux"]  # type: ignore[assignment]
        if return_score_decomposition:
            (
                bar_particles_transition_density,
                bar_kappa_transition_density,
                bar_nu_transition_density,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,  # type: ignore[arg-type]
                transition_vjp["transition_mean"],
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            transition_density_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_transition_density,
                bar_nu=bar_nu_transition_density,
            )
            per_seed_score += transition_density_score
            if return_regional_kappa_decomposition:
                regional_kappa_components = _add_regional_kappa_score_component(
                    regional_kappa_components,
                    "transition_mean_from_transition_density",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_transition_density,
                    ),
                )
            if return_regional_nu_decomposition:
                regional_nu_components = _add_regional_nu_score_component(
                    regional_nu_components,
                    "transition_mean_from_transition_density",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_transition_density,
                    ),
                )
            component_scores = _add_manual_score_component(
                component_scores,
                "transition_mean_from_transition_density",
                transition_density_score,
            )

            (
                bar_particles_ledh_flow_prior,
                bar_kappa_ledh_flow_prior,
                bar_nu_ledh_flow_prior,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,  # type: ignore[arg-type]
                flow_vjp.prior_means,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            ledh_flow_prior_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_ledh_flow_prior,
                bar_nu=bar_nu_ledh_flow_prior,
            )
            per_seed_score += ledh_flow_prior_score
            if return_regional_kappa_decomposition:
                regional_kappa_components = _add_regional_kappa_score_component(
                    regional_kappa_components,
                    "transition_mean_from_ledh_flow_prior",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_ledh_flow_prior,
                    ),
                )
            if return_regional_nu_decomposition:
                regional_nu_components = _add_regional_nu_score_component(
                    regional_nu_components,
                    "transition_mean_from_ledh_flow_prior",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_ledh_flow_prior,
                    ),
                )
            component_scores = _add_manual_score_component(
                component_scores,
                "transition_mean_from_ledh_flow_prior",
                ledh_flow_prior_score,
            )

            (
                bar_particles_pre_flow_clamp,
                bar_kappa_pre_flow_clamp,
                bar_nu_pre_flow_clamp,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,  # type: ignore[arg-type]
                bar_pushed,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            pre_flow_clamp_score = _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa_pre_flow_clamp,
                bar_nu=bar_nu_pre_flow_clamp,
            )
            per_seed_score += pre_flow_clamp_score
            if return_regional_kappa_decomposition:
                regional_kappa_components = _add_regional_kappa_score_component(
                    regional_kappa_components,
                    "transition_mean_from_pre_flow_clamp",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa_pre_flow_clamp,
                    ),
                )
            if return_regional_nu_decomposition:
                regional_nu_components = _add_regional_nu_score_component(
                    regional_nu_components,
                    "transition_mean_from_pre_flow_clamp",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu_pre_flow_clamp,
                    ),
                )
            component_scores = _add_manual_score_component(
                component_scores,
                "transition_mean_from_pre_flow_clamp",
                pre_flow_clamp_score,
            )
            bar_particles = (
                bar_particles_transition_density
                + bar_particles_ledh_flow_prior
                + bar_particles_pre_flow_clamp
            )
        else:
            bar_prior_means = (
                transition_vjp["transition_mean"]
                + flow_vjp.prior_means
                + bar_pushed
            )
            (
                bar_particles,
                bar_kappa,
                bar_nu,
            ) = _sir_transition_mean_vjp_tf(
                transition_aux,  # type: ignore[arg-type]
                bar_prior_means,
                kappa=kappa,
                nu=nu,
                adjacency=adjacency,
                neighbor_degree=neighbor_degree,
                step_size=step_size,
            )
            per_seed_score += _theta_score_from_parameter_cotangents(
                kappa=kappa,
                nu=nu,
                observation_covariance=observation_covariance,
                bar_kappa=bar_kappa,
                bar_nu=bar_nu,
            )
            if return_regional_kappa_decomposition:
                regional_kappa_components = _add_regional_kappa_score_component(
                    regional_kappa_components,
                    "transition_mean_from_transition_density",
                    _regional_kappa_score_from_cotangent(
                        kappa=kappa,
                        bar_kappa=bar_kappa,
                    ),
                )
            if return_regional_nu_decomposition:
                regional_nu_components = _add_regional_nu_score_component(
                    regional_nu_components,
                    "transition_mean_from_transition_density",
                    _regional_nu_score_from_cotangent(
                        nu=nu,
                        bar_nu=bar_nu,
                    ),
                )

    result = {
        "objective": tf.reduce_mean(log_likelihood),
        "log_likelihood": log_likelihood,
        "gradient_tensor": tf.reduce_mean(per_seed_score, axis=0),
        "per_seed_gradient": per_seed_score,
    }
    if return_score_decomposition:
        result["manual_score_components"] = component_scores
    if return_regional_kappa_decomposition:
        result["regional_kappa_score_components"] = regional_kappa_components
        result["regional_kappa_score_per_seed"] = tf.reduce_sum(
            regional_kappa_components,
            axis=0,
        )
    if return_regional_nu_decomposition:
        result["regional_nu_score_components"] = regional_nu_components
        result["regional_nu_score_per_seed"] = tf.reduce_sum(
            regional_nu_components,
            axis=0,
        )
    return result


def _value_for_theta(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> tuple[tf.Tensor, tf.Tensor]:
    return _objective_from_components(tensors, args, _theta_components(theta_values))


def _gradient_diagnostic(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> dict[str, Any]:
    theta_components = _theta_components(theta_values)
    with tf.GradientTape(persistent=True) as tape:
        for component in theta_components:
            tape.watch(component)
        objective, value = _objective_from_components(tensors, args, theta_components)
    gradients = tape.gradient(objective, theta_components)
    jacobian_columns = [tape.jacobian(value, component) for component in theta_components]
    del tape
    connected = [gradient is not None for gradient in gradients]
    gradient_values = [
        tf.constant(float("nan"), dtype=DTYPE) if gradient is None else gradient
        for gradient in gradients
    ]
    jacobian_values = [
        tf.fill(tf.shape(value), tf.constant(float("nan"), dtype=DTYPE))
        if jacobian is None
        else jacobian
        for jacobian in jacobian_columns
    ]
    gradient_tensor = tf.stack(gradient_values)
    per_seed_gradient = tf.stack(jacobian_values, axis=1)
    return {
        "objective": objective,
        "log_likelihood": value,
        "gradient_tensor": gradient_tensor,
        "per_seed_gradient": per_seed_gradient,
        "connectivity_by_component": dict(zip(PARAMETER_NAMES, connected, strict=True)),
        "gradients_connected": bool(all(connected)),
    }


def _manual_gradient_diagnostic(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> dict[str, Any]:
    result = _manual_value_and_score_from_components(
        tensors,
        args,
        _theta_components(theta_values),
    )
    return {
        **result,
        "connectivity_by_component": dict.fromkeys(PARAMETER_NAMES, True),
        "gradients_connected": True,
        "score_route": "manual_reverse_scan_no_autodiff",
    }


def _finite_difference_diagnostic(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    step = float(args.fd_step)
    component_indices = (
        [PARAMETER_NAMES.index("log_obs_noise_scale")]
        if args.diagnostic_components == "obs-noise"
        else list(range(len(PARAMETER_NAMES)))
    )
    for index in component_indices:
        name = PARAMETER_NAMES[index]
        plus = list(theta_values)
        minus = list(theta_values)
        plus[index] += step
        minus[index] -= step
        plus_objective, _ = _value_for_theta(tensors, args, plus)
        minus_objective, _ = _value_for_theta(tensors, args, minus)
        central = (plus_objective - minus_objective) / tf.constant(2.0 * step, dtype=DTYPE)
        diagnostics.append(
            {
                "parameter": name,
                "fd_step": step,
                "plus_objective": float(plus_objective.numpy()),
                "minus_objective": float(minus_objective.numpy()),
                "central_difference": float(central.numpy()),
                "finite": bool(tf.reduce_all(tf.math.is_finite(central)).numpy()),
            }
        )
    return diagnostics


def _finite_difference_ladder_diagnostic(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    component_indices = (
        [PARAMETER_NAMES.index("log_obs_noise_scale")]
        if args.diagnostic_components == "obs-noise"
        else list(range(len(PARAMETER_NAMES)))
    )
    for step in args.fd_step_ladder_values:
        for index in component_indices:
            name = PARAMETER_NAMES[index]
            plus = list(theta_values)
            minus = list(theta_values)
            plus[index] += step
            minus[index] -= step
            plus_objective, _ = _value_for_theta(tensors, args, plus)
            minus_objective, _ = _value_for_theta(tensors, args, minus)
            central = (plus_objective - minus_objective) / tf.constant(
                2.0 * step,
                dtype=DTYPE,
            )
            diagnostics.append(
                {
                    "parameter": name,
                    "fd_step": float(step),
                    "plus_objective": float(plus_objective.numpy()),
                    "minus_objective": float(minus_objective.numpy()),
                    "central_difference": float(central.numpy()),
                    "finite": bool(tf.reduce_all(tf.math.is_finite(central)).numpy()),
                }
            )
    return diagnostics


def _isolated_observation_noise_objective(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_components: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
) -> tf.Tensor:
    callbacks = _make_parameterized_callbacks(
        tensors=tensors,
        seeds=args.batch_seeds,
        args=args,
        theta_components=theta_components,
    )
    initial_particles = tf.cast(tensors["initial_particles"], DTYPE)
    observations = tf.cast(tensors["observations"], DTYPE)
    log_terms = []
    for time_index in range(int(observations.shape[0])):
        observation = observations[time_index]
        log_terms.append(
            callbacks["observation_log_density_fn"](
                initial_particles,
                observation,
                tf.constant(time_index, dtype=tf.int32),
            )
        )
    # Average over batch and particles.  This isolates the Gaussian observation
    # covariance path from LEDH flow, target correction, and transport.
    return tf.reduce_mean(tf.stack(log_terms))


def _isolated_observation_noise_diagnostic(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
    theta_values: list[float],
) -> dict[str, Any]:
    obs_index = PARAMETER_NAMES.index("log_obs_noise_scale")
    theta_components = _theta_components(theta_values)
    with tf.GradientTape() as tape:
        tape.watch(theta_components[obs_index])
        objective = _isolated_observation_noise_objective(
            tensors,
            args,
            theta_components,
        )
    gradient = tape.gradient(objective, theta_components[obs_index])
    gradient_value = (
        tf.constant(float("nan"), dtype=DTYPE)
        if gradient is None
        else tf.convert_to_tensor(gradient, dtype=DTYPE)
    )
    ladder = []
    for step in args.fd_step_ladder_values:
        plus = list(theta_values)
        minus = list(theta_values)
        plus[obs_index] += step
        minus[obs_index] -= step
        plus_objective = _isolated_observation_noise_objective(
            tensors,
            args,
            _theta_components(plus),
        )
        minus_objective = _isolated_observation_noise_objective(
            tensors,
            args,
            _theta_components(minus),
        )
        central = (plus_objective - minus_objective) / tf.constant(
            2.0 * step,
            dtype=DTYPE,
        )
        ladder.append(
            {
                "fd_step": float(step),
                "plus_objective": float(plus_objective.numpy()),
                "minus_objective": float(minus_objective.numpy()),
                "central_difference": float(central.numpy()),
                "ad_minus_fd": float((gradient_value - central).numpy()),
                "finite": bool(tf.reduce_all(tf.math.is_finite(central)).numpy()),
            }
        )
    return {
        "checked": True,
        "objective": float(objective.numpy()),
        "ad_gradient_log_obs_noise_scale": float(gradient_value.numpy()),
        "gradient_connected": gradient is not None,
        "gradient_finite": bool(tf.reduce_all(tf.math.is_finite(gradient_value)).numpy()),
        "finite_difference_ladder": ladder,
        "scope": "initial_particles_observation_log_density_only_no_flow_no_transport",
    }


def _p8j_fixed_value(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> tf.Tensor:
    callbacks = _dpf_sir_callbacks()
    adapter_callbacks = p8j_sir._make_actual_sir_callbacks(
        callbacks,
        tensors,
        args.batch_seeds,
        args,
    )
    value = streaming_tf.streaming_batched_ledh_pfpf_ot_value_core_tf(
        observations=tensors["observations"],
        initial_particles=tensors["initial_particles"],
        fixed_resampling_mask=tensors["fixed_resampling_mask"],
        transition_matrix=tensors["transition_matrix"],
        transition_covariance=tensors["transition_covariance"],
        observation_covariance=tensors["observation_covariance"],
        observation_fn=adapter_callbacks["observation_fn"],
        observation_jacobian_fn=adapter_callbacks["observation_jacobian_fn"],
        observation_residual_fn=adapter_callbacks["observation_residual_fn"],
        transition_log_density_fn=adapter_callbacks["transition_log_density_fn"],
        observation_log_density_fn=adapter_callbacks["observation_log_density_fn"],
        prior_mean_fn=adapter_callbacks["prior_mean_fn"],
        pre_flow_step_fn=adapter_callbacks["pre_flow_step_fn"],
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        annealed_scaling=args.annealed_scaling,
        annealed_convergence_threshold=args.annealed_convergence_threshold,
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_gradient_mode=args.transport_gradient_mode,
        transport_plan_mode=args.transport_plan_mode,
        transport_ad_mode=args.transport_ad_mode,
        row_chunk_size=args.row_chunk_size,
        col_chunk_size=args.col_chunk_size,
        particle_chunk_size=args.particle_chunk_size,
        return_history=False,
    )
    return value.log_likelihood


def _theta_zero_p8j_parity(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    p8p_objective, p8p_value = _value_for_theta(tensors, args, [0.0, 0.0, 0.0])
    p8j_value = _p8j_fixed_value(tensors, args)
    delta = p8p_value - p8j_value
    max_abs = tf.reduce_max(tf.abs(delta))
    return {
        "checked": True,
        "p8p_theta_zero_log_likelihood": _to_float_list(p8p_value),
        "p8j_fixed_log_likelihood": _to_float_list(p8j_value),
        "p8p_theta_zero_objective": float(p8p_objective.numpy()),
        "value_delta": _to_float_list(delta),
        "max_abs_delta": float(max_abs.numpy()),
        "atol": float(args.theta_zero_parity_atol),
        "pass": bool(max_abs.numpy() <= args.theta_zero_parity_atol),
    }


def _repeat_diagnostics(
    tensors: dict[str, tf.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gradients: list[tf.Tensor] = []
    per_seed_gradients: list[tf.Tensor] = []
    values: list[tf.Tensor] = []
    objectives: list[tf.Tensor] = []
    connections: list[dict[str, bool]] = []
    for _ in range(args.repeat_evaluations):
        diag = _gradient_diagnostic(tensors, args, args.theta_values)
        gradients.append(diag["gradient_tensor"])
        per_seed_gradients.append(diag["per_seed_gradient"])
        values.append(diag["log_likelihood"])
        objectives.append(diag["objective"])
        connections.append(diag["connectivity_by_component"])
    value_stack = tf.stack(values)
    gradient_stack = tf.stack(gradients)
    per_seed_gradient_stack = tf.stack(per_seed_gradients)
    objective_stack = tf.stack(objectives)
    repeat_value_delta = (
        tf.reduce_max(tf.abs(value_stack[1:] - value_stack[:1]))
        if args.repeat_evaluations > 1
        else tf.constant(0.0, dtype=DTYPE)
    )
    repeat_gradient_delta = (
        tf.reduce_max(tf.abs(gradient_stack[1:] - gradient_stack[:1]))
        if args.repeat_evaluations > 1
        else tf.constant(0.0, dtype=DTYPE)
    )
    repeat_per_seed_gradient_delta = (
        tf.reduce_max(tf.abs(per_seed_gradient_stack[1:] - per_seed_gradient_stack[:1]))
        if args.repeat_evaluations > 1
        else tf.constant(0.0, dtype=DTYPE)
    )
    return {
        "objective": objectives[0],
        "log_likelihood": values[0],
        "gradient_tensor": gradients[0],
        "per_seed_gradient": per_seed_gradients[0],
        "connectivity_by_component": connections[0],
        "all_connectivity_repeats_match": all(item == connections[0] for item in connections),
        "repeat_objectives": _to_float_list(objective_stack),
        "repeat_evaluations": int(args.repeat_evaluations),
        "repeat_value_max_abs_delta": float(repeat_value_delta.numpy()),
        "repeat_gradient_max_abs_delta": float(repeat_gradient_delta.numpy()),
        "repeat_per_seed_gradient_max_abs_delta": float(
            repeat_per_seed_gradient_delta.numpy()
        ),
        "repeat_atol": float(args.repeat_atol),
        "repeat_pass": bool(
            repeat_value_delta.numpy() <= args.repeat_atol
            and repeat_gradient_delta.numpy() <= args.repeat_atol
            and repeat_per_seed_gradient_delta.numpy() <= args.repeat_atol
        ),
    }


def _to_float_list(value: tf.Tensor) -> list[float]:
    tensor = tf.reshape(tf.convert_to_tensor(value), [-1])
    return [float(item) for item in tensor.numpy().tolist()]


def _to_float_matrix(value: tf.Tensor) -> list[list[float]]:
    tensor = tf.convert_to_tensor(value)
    if tensor.shape.rank != 2:
        raise ValueError("matrix_output must have rank 2")
    return [[float(item) for item in row] for row in tensor.numpy().tolist()]


def _mc_noise_summary(per_seed_gradient: tf.Tensor) -> dict[str, dict[str, Any]]:
    gradient = tf.convert_to_tensor(per_seed_gradient, dtype=DTYPE)
    if gradient.shape.rank != 2:
        raise ValueError("per_seed_gradient must have rank 2")
    batch_size = int(gradient.shape[0])
    if batch_size <= 0:
        raise ValueError("per_seed_gradient must have at least one row")
    mean = tf.reduce_mean(gradient, axis=0)
    centered = gradient - mean[tf.newaxis, :]
    if batch_size > 1:
        sample_variance = tf.reduce_sum(tf.square(centered), axis=0) / tf.cast(
            batch_size - 1,
            DTYPE,
        )
    else:
        sample_variance = tf.zeros_like(mean)
    sample_sd = tf.sqrt(sample_variance)
    standard_error = sample_sd / tf.sqrt(tf.cast(batch_size, DTYPE))
    min_value = tf.reduce_min(gradient, axis=0)
    max_value = tf.reduce_max(gradient, axis=0)
    return {
        name: {
            "mean": float(mean[index].numpy()),
            "sample_sd_across_seed_contributions": float(sample_sd[index].numpy()),
            "standard_error_of_batch_mean": float(standard_error[index].numpy()),
            "min": float(min_value[index].numpy()),
            "max": float(max_value[index].numpy()),
            "batch_size": batch_size,
        }
        for index, name in enumerate(PARAMETER_NAMES)
    }


def _gradient_geometry_summary(per_seed_gradient: tf.Tensor) -> dict[str, Any]:
    gradient = tf.convert_to_tensor(per_seed_gradient, dtype=DTYPE)
    if gradient.shape.rank != 2:
        raise ValueError("per_seed_gradient must have rank 2")
    batch_size = int(gradient.shape[0])
    centered = gradient - tf.reduce_mean(gradient, axis=0, keepdims=True)
    if batch_size > 1:
        covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(
            batch_size - 1,
            DTYPE,
        )
    else:
        covariance = tf.zeros([len(PARAMETER_NAMES), len(PARAMETER_NAMES)], dtype=DTYPE)
    sd = tf.sqrt(tf.maximum(tf.linalg.diag_part(covariance), tf.constant(0.0, DTYPE)))
    denom = sd[:, tf.newaxis] * sd[tf.newaxis, :]
    correlation = tf.where(
        denom > 0.0,
        covariance / denom,
        tf.zeros_like(covariance),
    )
    return {
        "parameter_order": list(PARAMETER_NAMES),
        "seed_gradient_covariance": _to_float_matrix(covariance),
        "seed_gradient_correlation": _to_float_matrix(correlation),
    }


def _finite_tensor(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _validate_device(tensors: tuple[tf.Tensor, ...], expect_device_kind: str) -> list[str]:
    devices = [tensor.device for tensor in tensors]
    if expect_device_kind == "gpu":
        if not all("GPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected GPU tensors, got {devices}")
    elif expect_device_kind == "cpu":
        if not all("CPU" in device.upper() for device in devices):
            raise RuntimeError(f"expected CPU tensors, got {devices}")
    return devices


def main() -> None:
    raise RuntimeError("ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY: this route is preserved only as provenance and cannot emit new evidence")
    args = _parse_args()
    require_historical_raw_diagnostic_opt_in(
        args, route_name="parameterized-SIR raw-barycentric diagnostic"
    )
    precision = _configure_precision(args)
    physical_gpus, logical_gpus = _configure_gpus()
    tensors, sir_semantics = _build_base_tensors(args)
    memory_before = _gpu_memory_info()
    start = time.perf_counter()
    with tf.device(args.device):
        repeat = _repeat_diagnostics(tensors, args)
        fd = _finite_difference_diagnostic(tensors, args, args.theta_values)
        fd_ladder = _finite_difference_ladder_diagnostic(tensors, args, args.theta_values)
        if args.check_isolated_observation_noise:
            isolated_observation_noise = _isolated_observation_noise_diagnostic(
                tensors,
                args,
                args.theta_values,
            )
        else:
            isolated_observation_noise = {"checked": False}
        if args.check_theta_zero_p8j_parity:
            parity = _theta_zero_p8j_parity(tensors, args)
        else:
            parity = {
                "checked": False,
                "pass": False,
                "max_abs_delta": None,
                "atol": float(args.theta_zero_parity_atol),
            }
    elapsed = time.perf_counter() - start
    memory_after = _gpu_memory_info()

    objective = repeat["objective"]
    log_likelihood = repeat["log_likelihood"]
    gradient = repeat["gradient_tensor"]
    per_seed_gradient = repeat["per_seed_gradient"]
    output_devices = _validate_device(
        (objective, log_likelihood, gradient),
        args.expect_device_kind,
    )
    gradient_values = _to_float_list(gradient)
    gradient_finite = _finite_tensor(gradient)
    value_finite = _finite_tensor(log_likelihood) and _finite_tensor(objective)
    connectivity = repeat["connectivity_by_component"]
    fd_finite = all(bool(item["finite"]) for item in fd)
    zero_gradient_components = [
        name
        for name, value in zip(PARAMETER_NAMES, gradient_values, strict=True)
        if value == 0.0
    ]
    categorical_resampling_used = False
    relaxed_sinkhorn_ot_used = args.transport_policy != "no-resampling"
    resampling_mask_fixed = True
    random_streams_fixed = True
    forward_contract = make_parameterized_sir_diagnostic_forward_contract(
        time_steps=args.time_steps,
        num_particles=args.num_particles,
        batch_seeds=args.batch_seeds,
    ).to_manifest()
    primary_pass = bool(
        parity["checked"]
        and parity["pass"]
        and value_finite
        and gradient_finite
        and all(bool(item) for item in connectivity.values())
        and len(zero_gradient_components) == 0
        and fd_finite
        and repeat["repeat_pass"]
        and repeat["all_connectivity_repeats_match"]
        and random_streams_fixed
        and resampling_mask_fixed
        and relaxed_sinkhorn_ot_used
        and not categorical_resampling_used
    )
    result: dict[str, Any] = {
        "schema_version": "filter_bench.p8p_parameterized_sir_gradient.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "git_commit": _git_commit(),
        "phase": args.phase_label,
        "status": "pass" if primary_pass else "blocked_or_failed",
        "elapsed_seconds": elapsed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "device": args.device,
        "device_scope": args.device_scope,
        "expect_device_kind": args.expect_device_kind,
        "output_devices": output_devices,
        "precision": precision,
        "shape": {
            "batch_size": len(args.batch_seeds),
            "time_steps": args.time_steps,
            "num_particles": args.num_particles,
            "state_dim": 18,
            "obs_dim": 9,
            "parameter_dim": 3,
        },
        "forward_contract": forward_contract,
        "target_scope": forward_contract["row_scope"],
        "target_scalar": forward_contract["target_scalar"],
        "target_output_tensor_field": forward_contract["output_tensor_field"],
        "target_density_fields": forward_contract["target_density_fields"],
        "proposal_flow_fields": forward_contract["proposal_flow_fields"],
        "correction_formula": forward_contract["correction_formula"],
        "sir_semantics": sir_semantics,
        "theta": dict(zip(PARAMETER_NAMES, [float(x) for x in args.theta_values], strict=True)),
        "theta_transform": {
            "kappa": "base_kappa * exp(log_kappa_scale)",
            "nu": "base_nu * exp(log_nu_scale)",
            "observation_covariance": (
                "base_observation_covariance * exp(2 * log_obs_noise_scale)"
            ),
        },
        "batch_seeds": [int(seed) for seed in args.batch_seeds],
        "initial_particle_seed_policy": "P8j initial_sample with fixed batch seeds",
        "process_noise_seed_policy": (
            "tf.random.stateless_normal seed=[batch_seed, 1140 + time_index], "
            "independent of theta"
        ),
        "random_streams_fixed_across_theta": random_streams_fixed,
        "repeat_evaluations": repeat["repeat_evaluations"],
        "repeat_objectives": repeat["repeat_objectives"],
        "repeat_value_max_abs_delta": repeat["repeat_value_max_abs_delta"],
        "repeat_gradient_max_abs_delta": repeat["repeat_gradient_max_abs_delta"],
        "repeat_per_seed_gradient_max_abs_delta": repeat[
            "repeat_per_seed_gradient_max_abs_delta"
        ],
        "repeat_atol": repeat["repeat_atol"],
        "repeat_pass": repeat["repeat_pass"],
        "transport_policy": args.transport_policy,
        "transport": {
            "value_core_mode": "streaming",
            "transport_plan_mode": args.transport_plan_mode,
            "transport_ad_mode": args.transport_ad_mode,
            "gradient_mode": args.transport_gradient_mode,
            "row_chunk_size": args.row_chunk_size,
            "col_chunk_size": args.col_chunk_size,
            "particle_chunk_size": args.particle_chunk_size,
            "sinkhorn_iterations": args.sinkhorn_iterations,
            "sinkhorn_epsilon": args.sinkhorn_epsilon,
            "annealed_scaling": args.annealed_scaling,
            "annealed_convergence_threshold": args.annealed_convergence_threshold,
            "dense_transport_matrix_materialized": False,
        },
        "resampling_mask_fixed": resampling_mask_fixed,
        "relaxed_sinkhorn_ot_used": relaxed_sinkhorn_ot_used,
        "categorical_resampling_used": categorical_resampling_used,
        "theta_zero_p8j_parity_checked": bool(parity["checked"]),
        "theta_zero_p8j_parity": parity,
        "theta_zero_p8j_value_delta_max_abs": parity["max_abs_delta"],
        "objective": float(objective.numpy()),
        "log_likelihood": _to_float_list(log_likelihood),
        "value_finite": value_finite,
        "connectivity_by_component": connectivity,
        "gradient": dict(zip(PARAMETER_NAMES, gradient_values, strict=True)),
        "gradient_values": gradient_values,
        "per_seed_gradient_contributions": _to_float_matrix(per_seed_gradient),
        "monte_carlo_gradient_noise": _mc_noise_summary(per_seed_gradient),
        "seed_gradient_geometry": _gradient_geometry_summary(per_seed_gradient),
        "gradient_finite": gradient_finite,
        "gradient_norm": float(tf.linalg.norm(gradient).numpy()),
        "zero_gradient_components": zero_gradient_components,
        "finite_difference_by_component": fd,
        "finite_difference_ladder": fd_ladder,
        "finite_difference_all_finite": fd_finite,
        "isolated_observation_noise_diagnostic": isolated_observation_noise,
        "gpu_memory_info_before": memory_before,
        "gpu_memory_info_after": memory_after,
        "primary_pass": primary_pass,
        "nonclaims": list(NONCLAIMS),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not primary_pass and not args.no_fail_on_veto:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
