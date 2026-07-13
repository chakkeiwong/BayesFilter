"""Independent scalar-LGSSM forecast oracle for A3 validation fixtures.

This module intentionally does not import the BayesFilter filtering or forecast
implementations.  Its analytic route uses finite-horizon sums, while its
simulation route applies the scalar model equations directly to materialized
stateless innovations.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions

FORECAST_HORIZON = 10
FLOAT64_EPSILON = 2.0**-52
PSD_TOLERANCE_MULTIPLIER = 256
DEFAULT_QUANTILE_PROBABILITIES = (0.05, 0.25, 0.50, 0.75, 0.95)
ANALYTIC_STATUS_VALID = "VALID"
ANALYTIC_STATUS_DEGENERATE_LOG_VARIANCE = "DEGENERATE_LOG_VARIANCE_INVALID"

_TERMINAL_FAMILY_CODE = 3101
_PROCESS_FAMILY_CODE = 3102
_OBSERVATION_FAMILY_CODE = 3103


def _require_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved <= 0:
        raise ValueError(f"{name} must be positive")
    return int(resolved)


def _require_int32(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        resolved = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if resolved < -(2**31) or resolved > 2**31 - 1:
        raise ValueError(f"{name} must fit int32")
    return int(resolved)


def _require_float64_tensor(value: Any, *, shape: tuple[int, ...], name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype != tf.float64:
        raise TypeError(f"{name} must have dtype float64")
    if tensor.shape.rank != len(shape) or tuple(tensor.shape.as_list()) != shape:
        raise ValueError(f"{name} must have static shape {shape}")
    if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _require_seed(seed: Any) -> tf.Tensor:
    tensor = tf.convert_to_tensor(seed)
    if tensor.dtype != tf.int32:
        raise TypeError("seed must have dtype int32")
    if tensor.shape != (2,):
        raise ValueError("seed must have static shape (2,)")
    return tensor


@dataclass(frozen=True)
class ScalarLGSSMParameters:
    """Parameters for the scalar terminal-law forecast problem."""

    transition_coefficient: tf.Tensor
    transition_offset: tf.Tensor
    observation_coefficient: tf.Tensor
    observation_offset: tf.Tensor
    terminal_mean: tf.Tensor
    terminal_variance: tf.Tensor
    process_variance: tf.Tensor
    observation_variance: tf.Tensor

    def __post_init__(self) -> None:
        names = (
            "transition_coefficient",
            "transition_offset",
            "observation_coefficient",
            "observation_offset",
            "terminal_mean",
            "terminal_variance",
            "process_variance",
            "observation_variance",
        )
        for name in names:
            tensor = _require_float64_tensor(getattr(self, name), shape=(), name=name)
            object.__setattr__(self, name, tensor)
        for name in ("terminal_variance", "process_variance", "observation_variance"):
            if bool(getattr(self, name) < tf.constant(0.0, tf.float64)):
                raise ValueError(f"{name} must be nonnegative")

    def as_tensor(self) -> tf.Tensor:
        """Return the fixed-order float64 parameter vector used by XLA kernels."""

        return tf.stack(
            (
                self.transition_coefficient,
                self.transition_offset,
                self.observation_coefficient,
                self.observation_offset,
                self.terminal_mean,
                self.terminal_variance,
                self.process_variance,
                self.observation_variance,
            )
        )


@dataclass(frozen=True)
class ScalarLGSSMAnalyticForecast:
    """Exact finite-horizon Gaussian observation law and state intermediates."""

    state_mean: tf.Tensor
    state_covariance: tf.Tensor
    observation_mean: tf.Tensor
    observation_covariance: tf.Tensor
    observation_variance: tf.Tensor
    observation_log_variance: tf.Tensor
    observation_third_central_moment: tf.Tensor
    observation_fourth_central_moment: tf.Tensor
    quantile_probabilities: tf.Tensor
    observation_quantiles: tf.Tensor
    state_symmetry_residual: tf.Tensor
    observation_symmetry_residual: tf.Tensor
    minimum_state_covariance_eigenvalue: tf.Tensor
    minimum_observation_covariance_eigenvalue: tf.Tensor
    state_psd_tolerance: tf.Tensor
    observation_psd_tolerance: tf.Tensor
    degenerate_variance_mask: tf.Tensor
    log_variance_valid: tf.Tensor
    status: tf.Tensor

    @property
    def minimum_covariance_eigenvalue(self) -> tf.Tensor:
        """Backward-compatible alias for the observation covariance diagnostic."""

        return self.minimum_observation_covariance_eigenvalue

    @property
    def psd_tolerance(self) -> tf.Tensor:
        """Backward-compatible alias for the observation PSD tolerance."""

        return self.observation_psd_tolerance


@dataclass(frozen=True)
class ScalarLGSSMInnovationBank:
    """Materialized independent standard-normal families for direct simulation."""

    terminal_standard_normal: tf.Tensor
    process_standard_normal: tf.Tensor
    observation_standard_normal: tf.Tensor
    root_seed: tf.Tensor
    arm_id: int

    @property
    def chain_count(self) -> int:
        return int(self.terminal_standard_normal.shape[0])

    @property
    def draw_count(self) -> int:
        return int(self.terminal_standard_normal.shape[1])

    @property
    def forecast_replication_count(self) -> int:
        return int(self.terminal_standard_normal.shape[2])


@dataclass(frozen=True)
class ScalarLGSSMSimulatedPaths:
    """Direct equation-level simulation with the full forecast hierarchy."""

    terminal_states: tf.Tensor
    states: tf.Tensor
    observations: tf.Tensor
    process_innovations: tf.Tensor
    observation_innovations: tf.Tensor


def _validate_horizon(horizon: Any) -> int:
    resolved = _require_positive_int(horizon, name="horizon")
    if resolved != FORECAST_HORIZON:
        raise ValueError("the A3 scalar LGSSM horizon is frozen at 10")
    return resolved


def _quantile_tensor(probabilities: Any) -> tf.Tensor:
    # Python sequences are numeric configuration and are canonicalized to the
    # oracle dtype. Explicit tensors remain strict so float32 cannot enter
    # unnoticed through an already-materialized fixture.
    values = (
        tf.convert_to_tensor(probabilities)
        if tf.is_tensor(probabilities)
        else tf.convert_to_tensor(probabilities, dtype=tf.float64)
    )
    if values.dtype != tf.float64:
        raise TypeError("quantile_probabilities must have dtype float64")
    if values.shape.rank != 1 or values.shape[0] is None or values.shape[0] <= 0:
        raise ValueError("quantile_probabilities must have a nonempty static rank-one shape")
    if not bool(tf.reduce_all(tf.math.is_finite(values))):
        raise ValueError("quantile_probabilities must contain only finite values")
    zero = tf.constant(0.0, tf.float64)
    one = tf.constant(1.0, tf.float64)
    if not bool(tf.reduce_all(tf.logical_and(values > zero, values < one))):
        raise ValueError("quantile_probabilities must lie strictly between zero and one")
    if int(values.shape[0]) > 1 and not bool(tf.reduce_all(values[1:] > values[:-1])):
        raise ValueError("quantile_probabilities must be strictly increasing")
    return values


def _analytic_kernel(parameters: tf.Tensor, probabilities: tf.Tensor) -> tuple[tf.Tensor, ...]:
    a, b, c, d, terminal_mean, terminal_variance, process_variance, observation_variance = tf.unstack(
        parameters, num=8
    )
    state_means = []
    for horizon in range(1, FORECAST_HORIZON + 1):
        geometric_sum = tf.add_n(
            [tf.pow(a, tf.constant(power, tf.float64)) for power in range(horizon)]
        )
        state_means.append(
            tf.pow(a, tf.constant(horizon, tf.float64)) * terminal_mean
            + b * geometric_sum
        )
    state_mean = tf.stack(state_means)

    covariance_rows = []
    for left_horizon in range(1, FORECAST_HORIZON + 1):
        row = []
        for right_horizon in range(1, FORECAST_HORIZON + 1):
            innovation_sum = tf.add_n(
                [
                    tf.pow(a, tf.constant(left_horizon - index, tf.float64))
                    * tf.pow(a, tf.constant(right_horizon - index, tf.float64))
                    for index in range(1, min(left_horizon, right_horizon) + 1)
                ]
            )
            row.append(
                tf.pow(a, tf.constant(left_horizon + right_horizon, tf.float64))
                * terminal_variance
                + process_variance * innovation_sum
            )
        covariance_rows.append(tf.stack(row))
    state_covariance = tf.stack(covariance_rows)
    observation_mean = c * state_mean + d
    observation_covariance = (
        tf.square(c) * state_covariance
        + observation_variance * tf.eye(FORECAST_HORIZON, dtype=tf.float64)
    )
    marginal_variance = tf.linalg.diag_part(observation_covariance)
    log_variance = tf.where(
        marginal_variance > tf.constant(0.0, tf.float64),
        tf.math.log(marginal_variance),
        tf.fill([FORECAST_HORIZON], tf.constant(float("-inf"), tf.float64)),
    )
    third_moment = tf.zeros([FORECAST_HORIZON], tf.float64)
    fourth_moment = tf.constant(3.0, tf.float64) * tf.square(marginal_variance)
    # TFP supplies the reviewed Gaussian quantile implementation. Degenerate
    # marginals are represented by their point mass after the TFP call.
    safe_scale = tf.sqrt(
        tf.where(
            marginal_variance > tf.constant(0.0, tf.float64),
            marginal_variance,
            tf.ones_like(marginal_variance),
        )
    )
    tfp_quantiles = tfd.Normal(
        loc=observation_mean[:, tf.newaxis],
        scale=safe_scale[:, tf.newaxis],
    ).quantile(probabilities[tf.newaxis, :])
    quantiles = tf.where(
        marginal_variance[:, tf.newaxis] > tf.constant(0.0, tf.float64),
        tfp_quantiles,
        observation_mean[:, tf.newaxis],
    )
    state_symmetry_residual = tf.reduce_max(
        tf.abs(state_covariance - tf.transpose(state_covariance))
    )
    observation_symmetry_residual = tf.reduce_max(
        tf.abs(observation_covariance - tf.transpose(observation_covariance))
    )
    state_eigenvalues = tf.linalg.eigvalsh(state_covariance)
    observation_eigenvalues = tf.linalg.eigvalsh(observation_covariance)
    state_norm = tf.sqrt(tf.reduce_sum(tf.square(state_covariance)))
    observation_norm = tf.sqrt(tf.reduce_sum(tf.square(observation_covariance)))
    tolerance_multiplier = tf.constant(
        PSD_TOLERANCE_MULTIPLIER * FLOAT64_EPSILON, tf.float64
    )
    state_psd_tolerance = tolerance_multiplier * tf.maximum(
        tf.constant(1.0, tf.float64), state_norm
    )
    observation_psd_tolerance = tolerance_multiplier * tf.maximum(
        tf.constant(1.0, tf.float64), observation_norm
    )
    degenerate_variance_mask = marginal_variance == tf.constant(0.0, tf.float64)
    log_variance_valid = tf.logical_not(tf.reduce_any(degenerate_variance_mask))
    return (
        state_mean,
        state_covariance,
        observation_mean,
        observation_covariance,
        marginal_variance,
        log_variance,
        third_moment,
        fourth_moment,
        quantiles,
        state_symmetry_residual,
        observation_symmetry_residual,
        tf.reduce_min(state_eigenvalues),
        tf.reduce_min(observation_eigenvalues),
        state_psd_tolerance,
        observation_psd_tolerance,
        degenerate_variance_mask,
        log_variance_valid,
    )


_ANALYTIC_PROGRAMS: dict[int, Callable[..., tuple[tf.Tensor, ...]]] = {}


def scalar_lgssm_analytic_compiled_program(
    quantile_count: int = len(DEFAULT_QUANTILE_PROBABILITIES),
) -> Callable[..., tuple[tf.Tensor, ...]]:
    """Return a reusable static-shape XLA analytic-oracle program."""

    count = _require_positive_int(quantile_count, name="quantile_count")
    compiled = _ANALYTIC_PROGRAMS.get(count)
    if compiled is None:
        compiled = tf.function(
            _analytic_kernel,
            input_signature=(
                tf.TensorSpec([8], tf.float64),
                tf.TensorSpec([count], tf.float64),
            ),
            jit_compile=True,
            reduce_retracing=True,
        )
        _ANALYTIC_PROGRAMS[count] = compiled
    return compiled


def analytic_scalar_lgssm_forecast(
    parameters: ScalarLGSSMParameters,
    *,
    horizon: int = FORECAST_HORIZON,
    quantile_probabilities: Any = DEFAULT_QUANTILE_PROBABILITIES,
    jit_compile: bool = True,
) -> ScalarLGSSMAnalyticForecast:
    """Derive the exact 1-to-10-step observation law using direct finite sums."""

    if not isinstance(parameters, ScalarLGSSMParameters):
        raise TypeError("parameters must be ScalarLGSSMParameters")
    _validate_horizon(horizon)
    if not isinstance(jit_compile, bool):
        raise TypeError("jit_compile must be bool")
    probabilities = _quantile_tensor(quantile_probabilities)
    if jit_compile:
        program = scalar_lgssm_analytic_compiled_program(int(probabilities.shape[0]))
        tensors = program(parameters.as_tensor(), probabilities)
    else:
        tensors = _analytic_kernel(parameters.as_tensor(), probabilities)
    tensors = tuple(tensors)
    finite_indices = tuple(index for index in range(len(tensors)) if index != 5)
    if not all(
        bool(tf.reduce_all(tf.math.is_finite(tensors[index])))
        for index in finite_indices
        if tensors[index].dtype != tf.bool
    ):
        raise ValueError("analytic scalar LGSSM forecast produced a nonfinite value")
    marginal_variance = tensors[4]
    if bool(tf.reduce_any(marginal_variance < tf.constant(0.0, tf.float64))):
        raise ValueError("analytic scalar LGSSM produced a negative marginal variance")
    state_symmetry_residual = tensors[9]
    observation_symmetry_residual = tensors[10]
    minimum_state_eigenvalue = tensors[11]
    minimum_observation_eigenvalue = tensors[12]
    state_psd_tolerance = tensors[13]
    observation_psd_tolerance = tensors[14]
    if bool(state_symmetry_residual > state_psd_tolerance):
        raise ValueError("analytic scalar LGSSM state covariance is materially asymmetric")
    if bool(observation_symmetry_residual > observation_psd_tolerance):
        raise ValueError(
            "analytic scalar LGSSM observation covariance is materially asymmetric"
        )
    if bool(minimum_state_eigenvalue < -state_psd_tolerance):
        raise ValueError("analytic scalar LGSSM state covariance is materially indefinite")
    if bool(minimum_observation_eigenvalue < -observation_psd_tolerance):
        raise ValueError(
            "analytic scalar LGSSM observation covariance is materially indefinite"
        )
    positive_variance = marginal_variance > tf.constant(0.0, tf.float64)
    if bool(tf.reduce_any(tf.logical_and(positive_variance, ~tf.math.is_finite(tensors[5])))):
        raise ValueError("positive analytic variance produced a nonfinite log variance")
    log_variance_valid = tensors[16]
    status = tf.constant(
        ANALYTIC_STATUS_VALID
        if bool(log_variance_valid)
        else ANALYTIC_STATUS_DEGENERATE_LOG_VARIANCE
    )
    return ScalarLGSSMAnalyticForecast(
        state_mean=tensors[0],
        state_covariance=tensors[1],
        observation_mean=tensors[2],
        observation_covariance=tensors[3],
        observation_variance=tensors[4],
        observation_log_variance=tensors[5],
        observation_third_central_moment=tensors[6],
        observation_fourth_central_moment=tensors[7],
        quantile_probabilities=probabilities,
        observation_quantiles=tensors[8],
        state_symmetry_residual=state_symmetry_residual,
        observation_symmetry_residual=observation_symmetry_residual,
        minimum_state_covariance_eigenvalue=minimum_state_eigenvalue,
        minimum_observation_covariance_eigenvalue=minimum_observation_eigenvalue,
        state_psd_tolerance=state_psd_tolerance,
        observation_psd_tolerance=observation_psd_tolerance,
        degenerate_variance_mask=tensors[15],
        log_variance_valid=log_variance_valid,
        status=status,
    )


def make_scalar_lgssm_innovation_bank(
    *,
    chain_count: int,
    draw_count: int,
    forecast_replication_count: int,
    seed: Any,
    arm_id: int,
    horizon: int = FORECAST_HORIZON,
) -> ScalarLGSSMInnovationBank:
    """Materialize disjoint terminal/process/observation Philox families."""

    chains = _require_positive_int(chain_count, name="chain_count")
    draws = _require_positive_int(draw_count, name="draw_count")
    replications = _require_positive_int(
        forecast_replication_count, name="forecast_replication_count"
    )
    _validate_horizon(horizon)
    arm = _require_int32(arm_id, name="arm_id")
    root_seed = _require_seed(seed)
    arm_seed = tf.random.experimental.stateless_fold_in(
        root_seed, tf.constant(arm, tf.int32), alg="philox"
    )

    def family_seed(code: int) -> tf.Tensor:
        return tf.random.experimental.stateless_fold_in(
            arm_seed, tf.constant(code, tf.int32), alg="philox"
        )

    leading_shape = [chains, draws, replications]
    terminal = tf.random.stateless_normal(
        leading_shape,
        family_seed(_TERMINAL_FAMILY_CODE),
        dtype=tf.float64,
        alg="philox",
    )
    process = tf.random.stateless_normal(
        [*leading_shape, FORECAST_HORIZON],
        family_seed(_PROCESS_FAMILY_CODE),
        dtype=tf.float64,
        alg="philox",
    )
    observation = tf.random.stateless_normal(
        [*leading_shape, FORECAST_HORIZON],
        family_seed(_OBSERVATION_FAMILY_CODE),
        dtype=tf.float64,
        alg="philox",
    )
    return ScalarLGSSMInnovationBank(
        terminal_standard_normal=terminal,
        process_standard_normal=process,
        observation_standard_normal=observation,
        root_seed=root_seed,
        arm_id=arm,
    )


def _validate_innovation_bank(bank: ScalarLGSSMInnovationBank) -> tuple[int, int, int]:
    if not isinstance(bank, ScalarLGSSMInnovationBank):
        raise TypeError("innovation_bank must be ScalarLGSSMInnovationBank")
    terminal = tf.convert_to_tensor(bank.terminal_standard_normal)
    if terminal.dtype != tf.float64:
        raise TypeError("terminal_standard_normal must have dtype float64")
    if terminal.shape.rank != 3 or any(axis is None for axis in terminal.shape):
        raise ValueError("terminal_standard_normal must have fully static shape [chain, draw, replication]")
    shape = tuple(int(axis) for axis in terminal.shape)
    if any(axis <= 0 for axis in shape):
        raise ValueError("innovation hierarchy dimensions must be positive")
    _require_float64_tensor(terminal, shape=shape, name="terminal_standard_normal")
    extended = (*shape, FORECAST_HORIZON)
    _require_float64_tensor(
        bank.process_standard_normal, shape=extended, name="process_standard_normal"
    )
    _require_float64_tensor(
        bank.observation_standard_normal,
        shape=extended,
        name="observation_standard_normal",
    )
    _require_seed(bank.root_seed)
    _require_int32(bank.arm_id, name="arm_id")
    return shape


def _simulation_kernel(
    parameters: tf.Tensor,
    terminal_standard_normal: tf.Tensor,
    process_standard_normal: tf.Tensor,
    observation_standard_normal: tf.Tensor,
) -> tuple[tf.Tensor, ...]:
    a, b, c, d, terminal_mean, terminal_variance, process_variance, observation_variance = tf.unstack(
        parameters, num=8
    )
    terminal_states = terminal_mean + tf.sqrt(terminal_variance) * terminal_standard_normal
    previous = terminal_states
    states = []
    observations = []
    process_innovations = []
    observation_innovations = []
    for horizon_index in range(FORECAST_HORIZON):
        process_noise = tf.sqrt(process_variance) * process_standard_normal[..., horizon_index]
        next_state = a * previous + b + process_noise
        observation_noise = (
            tf.sqrt(observation_variance)
            * observation_standard_normal[..., horizon_index]
        )
        observation = c * next_state + d + observation_noise
        states.append(next_state)
        observations.append(observation)
        process_innovations.append(process_noise)
        observation_innovations.append(observation_noise)
        previous = next_state
    return (
        terminal_states,
        tf.stack(states, axis=-1),
        tf.stack(observations, axis=-1),
        tf.stack(process_innovations, axis=-1),
        tf.stack(observation_innovations, axis=-1),
    )


_SIMULATION_PROGRAMS: dict[
    tuple[int, int, int], Callable[..., tuple[tf.Tensor, ...]]
] = {}


def scalar_lgssm_simulation_compiled_program(
    chain_count: int,
    draw_count: int,
    forecast_replication_count: int,
) -> Callable[..., tuple[tf.Tensor, ...]]:
    """Return a reusable static-shape XLA direct-simulation program."""

    shape = (
        _require_positive_int(chain_count, name="chain_count"),
        _require_positive_int(draw_count, name="draw_count"),
        _require_positive_int(
            forecast_replication_count, name="forecast_replication_count"
        ),
    )
    compiled = _SIMULATION_PROGRAMS.get(shape)
    if compiled is None:
        extended = (*shape, FORECAST_HORIZON)
        compiled = tf.function(
            _simulation_kernel,
            input_signature=(
                tf.TensorSpec([8], tf.float64),
                tf.TensorSpec(shape, tf.float64),
                tf.TensorSpec(extended, tf.float64),
                tf.TensorSpec(extended, tf.float64),
            ),
            jit_compile=True,
            reduce_retracing=True,
        )
        _SIMULATION_PROGRAMS[shape] = compiled
    return compiled


def simulate_scalar_lgssm_forecast(
    parameters: ScalarLGSSMParameters,
    innovation_bank: ScalarLGSSMInnovationBank,
    *,
    horizon: int = FORECAST_HORIZON,
    jit_compile: bool = True,
) -> ScalarLGSSMSimulatedPaths:
    """Apply the scalar state and observation equations to a materialized bank."""

    if not isinstance(parameters, ScalarLGSSMParameters):
        raise TypeError("parameters must be ScalarLGSSMParameters")
    _validate_horizon(horizon)
    if not isinstance(jit_compile, bool):
        raise TypeError("jit_compile must be bool")
    shape = _validate_innovation_bank(innovation_bank)
    inputs = (
        parameters.as_tensor(),
        innovation_bank.terminal_standard_normal,
        innovation_bank.process_standard_normal,
        innovation_bank.observation_standard_normal,
    )
    if jit_compile:
        program = scalar_lgssm_simulation_compiled_program(*shape)
        tensors = program(*inputs)
    else:
        tensors = _simulation_kernel(*inputs)
    tensors = tuple(tensors)
    if len(tensors) != 5 or not all(
        bool(tf.reduce_all(tf.math.is_finite(tensor))) for tensor in tensors
    ):
        raise ValueError("direct scalar LGSSM simulation produced invalid outputs")
    return ScalarLGSSMSimulatedPaths(
        terminal_states=tensors[0],
        states=tensors[1],
        observations=tensors[2],
        process_innovations=tensors[3],
        observation_innovations=tensors[4],
    )
