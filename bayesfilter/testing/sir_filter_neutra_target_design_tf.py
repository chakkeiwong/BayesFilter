"""Graph-native parameterized Austria SIR filter posteriors for P6."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.highdim.models import zhao_cui_sir_austria_model
from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.batched_svd_sigma_point_tf import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
)
from bayesfilter.nonlinear.fixed_sgqf_tf import (
    tf_fixed_sgqf_level2_axis_cloud,
)
from bayesfilter.ssm import (
    BayesianSSMProblem,
    FilterProgram,
    ParameterChart,
    ParameterPrior,
    SSMDataSignature,
    SSMStaticShape,
    SSMTargetContract,
    stable_ssm_target_signature,
)


SIR_DATASET_ID = "zhao_cui_austria_parameter_extension_y1_y20"
SIR_DATASET_SEED = 81120
SIR_HORIZON = 20
SIR_STATE_SHA256 = "8cd5a079f5799f0e0b769e5ac21a4bdf460475a72319f07dc27fb037eb5774e0"
SIR_OBSERVATION_SHA256 = (
    "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
)
SIR_WRONG_TIME_ORDER_SHA256 = (
    "c4df0ac33a28bde16cad169892f49705f1dfa4f3541eeac9ae4afa4aa33cf041"
)
SIR_PARAMETER_NAMES = (
    "log_kappa_scale",
    "log_nu_scale",
    "log_observation_noise_scale",
)
SIR_STATE_DIM = 18
SIR_OBSERVATION_DIM = 9
SIR_PARAMETER_DIM = 3
SIR_PRIOR_SCALE = tf.constant(0.5, tf.float64)
SIR_UKF_SCOPE = "SIR-UKF-three-log-scale-y1-y20-v1"
SIR_SGQF_SCOPE = "SIR-SGQF-level2-axis-three-log-scale-y1-y20-v1"
SIR_UKF_NONCLAIMS = (
    "BayesFilter three-parameter extension of the fixed-parameter Zhao-Cui SIR example",
    "principal-square-root UKF approximate observed-data filter posterior",
    "no HMC convergence, NeuTra training, filter exactness, calibration, forecasting, or readiness claim",
)
SIR_SGQF_NONCLAIMS = (
    "BayesFilter three-parameter extension of the fixed-parameter Zhao-Cui SIR example",
    "fixed level-2 37-point SGQF approximate observed-data filter posterior",
    "negative center quadrature weight is retained",
    "no HMC convergence, NeuTra training, filter exactness, calibration, forecasting, or readiness claim",
)

_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float64)
_INITIAL_MEAN = tf.reshape(
    tf.stack(
        (
            486.0 + tf.cast(tf.range(1, 10), tf.float64),
            14.0 - tf.cast(tf.range(1, 10), tf.float64),
        ),
        axis=1,
    ),
    [SIR_STATE_DIM],
)
_INITIAL_COVARIANCE = tf.eye(SIR_STATE_DIM, dtype=tf.float64)
_PROCESS_COVARIANCE = tf.eye(SIR_STATE_DIM, dtype=tf.float64)
_BASE_OBSERVATION_COVARIANCE = 100.0 * tf.eye(
    SIR_OBSERVATION_DIM, dtype=tf.float64
)
_ADJACENCY = tf.constant(
    [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    ],
    tf.float64,
)
_DEGREE = tf.reduce_sum(_ADJACENCY, axis=1)
_OBSERVATION_MATRIX = tf.one_hot(
    tf.constant(tuple(range(1, SIR_STATE_DIM, 2)), tf.int32),
    depth=SIR_STATE_DIM,
    dtype=tf.float64,
)
_RK4_STEP = tf.constant(0.005, tf.float64)
_RK4_SUBSTEPS = 4
_MIN_VARIANCE = tf.constant(1.0e-12, tf.float64)


def generate_frozen_sir_dataset_tf() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Replay x0:x20 and freeze the paper-consistent y1:y20 observations."""

    with tf.device("/CPU:0"):
        states, all_observations = zhao_cui_sir_austria_model().simulate(
            final_time=SIR_HORIZON,
            seed=SIR_DATASET_SEED,
        )
        states = tf.convert_to_tensor(states, tf.float64)
        all_observations = tf.convert_to_tensor(all_observations, tf.float64)
        observations = all_observations[1 : SIR_HORIZON + 1]
        if _tensor_hash(states) != SIR_STATE_SHA256:
            raise ValueError("frozen SIR state hash mismatch")
        if _tensor_hash(observations) != SIR_OBSERVATION_SHA256:
            raise ValueError("frozen SIR y1:y20 hash mismatch")
        if _tensor_hash(all_observations[:SIR_HORIZON]) != SIR_WRONG_TIME_ORDER_SHA256:
            raise ValueError("frozen SIR wrong-time-order sentinel hash mismatch")
        if bool(tf.reduce_any(states[:, 0::2] < 0.0).numpy()):
            raise ValueError("frozen SIR fixture activates susceptible clipping")
        return states, observations, all_observations


def sir_prior_value_score(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the independent Normal(0, 0.5^2) log-scale prior."""

    values = _rank2_theta(theta)
    standardized = values / SIR_PRIOR_SCALE
    value = tf.reduce_sum(
        -0.5 * tf.square(standardized)
        - tf.math.log(SIR_PRIOR_SCALE)
        - 0.5 * _LOG_TWO_PI,
        axis=1,
    )
    return value, -values / tf.square(SIR_PRIOR_SCALE)


def sir_identity_chart_jacobian_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the zero log-Jacobian term for the identity log-scale chart."""

    values = _rank2_theta(theta)
    return tf.zeros(tf.shape(values)[:1], tf.float64), tf.zeros_like(values)


def sir_scaled_parameters(theta: Any) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    values = _rank2_theta(theta)
    kappa = 0.1 * tf.exp(values[:, 0:1]) * tf.ones([1, 9], tf.float64)
    nu = 18.0 * tf.exp(values[:, 1:2]) * tf.ones([1, 9], tf.float64)
    observation_covariance = (
        tf.exp(2.0 * values[:, 2])[:, None, None]
        * _BASE_OBSERVATION_COVARIANCE[None, :, :]
    )
    return kappa, nu, observation_covariance


def sir_prior_predictive_tf(
    *, batch_size: int, horizon: int, seed: tf.Tensor
) -> Mapping[str, tf.Tensor]:
    """Simulate an unprojected batched prior-predictive SIR ensemble."""

    if batch_size < 1 or horizon < 1:
        raise ValueError("batch_size and horizon must be positive")
    roots = tf.random.experimental.stateless_split(
        tf.convert_to_tensor(seed, tf.int32), 4
    )
    theta = SIR_PRIOR_SCALE * tf.random.stateless_normal(
        [batch_size, SIR_PARAMETER_DIM], roots[0], dtype=tf.float64
    )
    initial_noise = tf.random.stateless_normal(
        [batch_size, SIR_STATE_DIM], roots[1], dtype=tf.float64
    )
    process_noise = tf.random.stateless_normal(
        [horizon, batch_size, SIR_STATE_DIM], roots[2], dtype=tf.float64
    )
    observation_noise = tf.random.stateless_normal(
        [horizon, batch_size, SIR_OBSERVATION_DIM], roots[3], dtype=tf.float64
    )
    _kappa, _nu, observation_covariance = sir_scaled_parameters(theta)
    observation_factor = tf.linalg.cholesky(observation_covariance)
    initial_state = _INITIAL_MEAN[None, :] + initial_noise
    states = tf.TensorArray(tf.float64, size=horizon, clear_after_read=False)
    observations = tf.TensorArray(tf.float64, size=horizon, clear_after_read=False)

    def body(
        index: tf.Tensor,
        previous: tf.Tensor,
        state_array: tf.TensorArray,
        observation_array: tf.TensorArray,
    ):
        transitioned = sir_rk4_transition_value(theta, previous[:, None, :])[:, 0, :]
        current = transitioned + process_noise[index]
        observation_mean = tf.einsum("oi,bi->bo", _OBSERVATION_MATRIX, current)
        observation = observation_mean + tf.einsum(
            "bij,bj->bi", observation_factor, observation_noise[index]
        )
        return (
            index + 1,
            current,
            state_array.write(index, current),
            observation_array.write(index, observation),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.constant(horizon, tf.int32),
        body,
        (
            tf.constant(0, tf.int32),
            initial_state,
            states,
            observations,
        ),
        parallel_iterations=1,
    )
    return {
        "theta": theta,
        "initial_state": initial_state,
        "states": tf.transpose(result[2].stack(), [1, 0, 2]),
        "observations": tf.transpose(result[3].stack(), [1, 0, 2]),
    }


def sir_rk4_transition_value(theta: Any, previous_points: tf.Tensor) -> tf.Tensor:
    """Propagate batched SIR points without constructing unused Jacobians."""

    values = _rank2_theta(theta)
    state = tf.convert_to_tensor(previous_points)
    if state.dtype != tf.float64:
        raise ValueError("SIR transition requires float64 state")
    if (
        state.shape.rank != 3
        or state.shape[0] != values.shape[0]
        or state.shape[2] != SIR_STATE_DIM
    ):
        raise ValueError("SIR points require shape [batch, point, 18]")
    kappa, nu, _observation_covariance = sir_scaled_parameters(values)

    def rhs(current: tf.Tensor) -> tf.Tensor:
        susceptible = current[..., 0::2]
        infectious = current[..., 1::2]
        susceptible_neighbor = (
            tf.einsum("brj,kj->brk", susceptible, _ADJACENCY)
            - susceptible * _DEGREE[None, None, :]
        )
        infectious_neighbor = (
            tf.einsum("brj,kj->brk", infectious, _ADJACENCY)
            - infectious * _DEGREE[None, None, :]
        )
        infection = kappa[:, None, :] * susceptible * infectious
        return tf.reshape(
            tf.stack(
                (
                    -infection + 0.5 * susceptible_neighbor,
                    infection
                    - nu[:, None, :] * infectious
                    + 0.5 * infectious_neighbor,
                ),
                axis=-1,
            ),
            tf.shape(current),
        )

    def body(index: tf.Tensor, current: tf.Tensor):
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * _RK4_STEP * k1)
        k3 = rhs(current + 0.5 * _RK4_STEP * k2)
        # Match the author source's half-step fourth stage.
        k4 = rhs(current + 0.5 * _RK4_STEP * k3)
        return (
            index + 1,
            current + (_RK4_STEP / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
        )

    return tf.while_loop(
        lambda index, *_unused: index < tf.constant(_RK4_SUBSTEPS, tf.int32),
        body,
        (tf.constant(0, tf.int32), state),
        parallel_iterations=1,
    )[1]


def sir_bootstrap_pf_log_likelihood_tf(
    theta: tf.Tensor,
    *,
    observations: tf.Tensor,
    particle_count: int,
    replicate_count: int,
    seed: tf.Tensor,
) -> tf.Tensor:
    """Return independent bootstrap-PF likelihood estimates for one theta."""

    parameters = tf.convert_to_tensor(theta)
    if parameters.dtype != tf.float64 or parameters.shape != (SIR_PARAMETER_DIM,):
        raise ValueError("bootstrap PF theta must have float64 shape [3]")
    y = tf.convert_to_tensor(observations)
    if y.dtype != tf.float64 or y.shape.rank != 2 or y.shape[1] != SIR_OBSERVATION_DIM:
        raise ValueError("bootstrap PF observations require float64 shape [time, 9]")
    if particle_count < 2 or replicate_count < 1:
        raise ValueError("bootstrap PF requires particles >=2 and replicates >=1")
    root_count = 1 + 2 * int(y.shape[0])
    roots = tf.random.experimental.stateless_split(
        tf.convert_to_tensor(seed, tf.int32), root_count
    )
    particles = _INITIAL_MEAN[None, None, :] + tf.random.stateless_normal(
        [replicate_count, particle_count, SIR_STATE_DIM],
        roots[0],
        dtype=tf.float64,
    )
    log_likelihood = tf.zeros([replicate_count], tf.float64)
    repeated_theta = tf.broadcast_to(
        parameters[None, :], [replicate_count, SIR_PARAMETER_DIM]
    )
    _kappa, _nu, observation_covariance = sir_scaled_parameters(repeated_theta)
    observation_variance = tf.linalg.diag_part(observation_covariance)[:, None, :]
    log_normalizer = tf.cast(particle_count, tf.float64)

    def body(index: tf.Tensor, current: tf.Tensor, total: tf.Tensor):
        transition_mean = sir_rk4_transition_value(repeated_theta, current)
        process_noise = tf.random.stateless_normal(
            [replicate_count, particle_count, SIR_STATE_DIM],
            roots[1 + 2 * index],
            dtype=tf.float64,
        )
        proposed = transition_mean + process_noise
        infectious = proposed[..., 1::2]
        residual = y[index][None, None, :] - infectious
        log_weights = tf.reduce_sum(
            -0.5
            * (
                _LOG_TWO_PI
                + tf.math.log(observation_variance)
                + tf.square(residual) / observation_variance
            ),
            axis=2,
        )
        increment = tf.reduce_logsumexp(log_weights, axis=1) - tf.math.log(
            log_normalizer
        )
        ancestor = tf.random.stateless_categorical(
            log_weights,
            num_samples=particle_count,
            seed=roots[2 + 2 * index],
            dtype=tf.int32,
        )
        resampled = tf.gather(proposed, ancestor, axis=1, batch_dims=1)
        return index + 1, resampled, total + increment

    return tf.while_loop(
        lambda index, *_unused: index < tf.shape(y)[0],
        body,
        (tf.constant(0, tf.int32), particles, log_likelihood),
        parallel_iterations=1,
    )[2]


def sir_rk4_transition_value_state_source_jacobians(
    theta: Any, previous_points: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Propagate batched SIR points and exact source/state sensitivities."""

    values = _rank2_theta(theta)
    state = tf.convert_to_tensor(previous_points, tf.float64)
    if (
        state.shape.rank != 3
        or state.shape[0] != values.shape[0]
        or state.shape[2] != SIR_STATE_DIM
    ):
        raise ValueError("SIR points require shape [batch, point, 18]")
    batch_size = int(values.shape[0])
    point_count = tf.shape(state)[1]
    state_jacobian = tf.broadcast_to(
        tf.eye(SIR_STATE_DIM, dtype=tf.float64)[None, None, :, :],
        [batch_size, point_count, SIR_STATE_DIM, SIR_STATE_DIM],
    )
    source_jacobian = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, point_count, SIR_STATE_DIM], tf.float64
    )
    kappa, nu, _observation_covariance = sir_scaled_parameters(values)

    def stage(
        stage_state: tf.Tensor,
        stage_state_jacobian: tf.Tensor,
        stage_source_jacobian: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        rhs, rhs_state, rhs_source = _sir_rhs_jacobians(
            stage_state, kappa=kappa, nu=nu
        )
        return (
            rhs,
            tf.einsum("brij,brjk->brik", rhs_state, stage_state_jacobian),
            tf.einsum("brij,bprj->bpri", rhs_state, stage_source_jacobian)
            + rhs_source,
        )

    def body(
        _index: tf.Tensor,
        current_state: tf.Tensor,
        current_state_jacobian: tf.Tensor,
        current_source_jacobian: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        k1, a1, b1 = stage(
            current_state, current_state_jacobian, current_source_jacobian
        )
        k2, a2, b2 = stage(
            current_state + 0.5 * _RK4_STEP * k1,
            current_state_jacobian + 0.5 * _RK4_STEP * a1,
            current_source_jacobian + 0.5 * _RK4_STEP * b1,
        )
        k3, a3, b3 = stage(
            current_state + 0.5 * _RK4_STEP * k2,
            current_state_jacobian + 0.5 * _RK4_STEP * a2,
            current_source_jacobian + 0.5 * _RK4_STEP * b2,
        )
        # The author source's sir_step uses a half-step fourth stage.
        k4, a4, b4 = stage(
            current_state + 0.5 * _RK4_STEP * k3,
            current_state_jacobian + 0.5 * _RK4_STEP * a3,
            current_source_jacobian + 0.5 * _RK4_STEP * b3,
        )
        scale = _RK4_STEP / 6.0
        return (
            tf.constant(0, tf.int32),
            current_state + scale * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            current_state_jacobian + scale * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
            current_source_jacobian + scale * (b1 + 2.0 * b2 + 2.0 * b3 + b4),
        )

    def loop_body(index, current_state, current_state_jacobian, current_source_jacobian):
        _unused, next_state, next_state_jacobian, next_source_jacobian = body(
            index, current_state, current_state_jacobian, current_source_jacobian
        )
        return index + 1, next_state, next_state_jacobian, next_source_jacobian

    result = tf.while_loop(
        lambda index, *_unused: index < tf.constant(_RK4_SUBSTEPS, tf.int32),
        loop_body,
        (tf.constant(0, tf.int32), state, state_jacobian, source_jacobian),
        parallel_iterations=1,
    )
    return result[1], result[2], result[3]


def _sir_rhs_jacobians(
    state: tf.Tensor, *, kappa: tf.Tensor, nu: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    susceptible = state[..., 0::2]
    infectious = state[..., 1::2]
    susceptible_neighbor = (
        tf.einsum("brj,kj->brk", susceptible, _ADJACENCY)
        - susceptible * _DEGREE[None, None, :]
    )
    infectious_neighbor = (
        tf.einsum("brj,kj->brk", infectious, _ADJACENCY)
        - infectious * _DEGREE[None, None, :]
    )
    infection = kappa[:, None, :] * susceptible * infectious
    rhs_s = -infection + 0.5 * susceptible_neighbor
    rhs_i = infection - nu[:, None, :] * infectious + 0.5 * infectious_neighbor
    rhs = tf.reshape(
        tf.stack((rhs_s, rhs_i), axis=-1),
        [tf.shape(state)[0], tf.shape(state)[1], SIR_STATE_DIM],
    )

    neighbor_jacobian = 0.5 * (
        _ADJACENCY - tf.linalg.diag(_DEGREE)
    )[None, None, :, :]
    s_s = neighbor_jacobian - tf.linalg.diag(
        kappa[:, None, :] * infectious
    )
    s_i = -tf.linalg.diag(kappa[:, None, :] * susceptible)
    i_s = tf.linalg.diag(kappa[:, None, :] * infectious)
    i_i = neighbor_jacobian + tf.linalg.diag(
        kappa[:, None, :] * susceptible - nu[:, None, :]
    )
    block_rows = tf.stack(
        (tf.stack((s_s, s_i), axis=-3), tf.stack((i_s, i_i), axis=-3)),
        axis=-4,
    )
    # Convert [out-kind, in-kind, out-compartment, in-compartment] to the
    # interleaved state order (S1,I1,...,S9,I9) on both matrix axes.
    block_rows = tf.transpose(block_rows, [0, 1, 4, 2, 5, 3])
    state_jacobian = tf.reshape(
        block_rows,
        [tf.shape(state)[0], tf.shape(state)[1], SIR_STATE_DIM, SIR_STATE_DIM],
    )

    zeros = tf.zeros_like(infection)
    source_s = tf.stack((-infection, zeros, zeros), axis=1)
    source_i = tf.stack((infection, -nu[:, None, :] * infectious, zeros), axis=1)
    source_jacobian = tf.reshape(
        tf.stack((source_s, source_i), axis=-1),
        [tf.shape(state)[0], SIR_PARAMETER_DIM, tf.shape(state)[1], SIR_STATE_DIM],
    )
    return rhs, state_jacobian, source_jacobian


def _build_ukf_model_and_derivatives(
    theta: tf.Tensor,
) -> tuple[TFBatchedStructuralStateSpace, TFBatchedStructuralFirstDerivatives]:
    values = _rank2_theta(theta)
    batch_size = int(values.shape[0])
    _kappa, _nu, observation_covariance = sir_scaled_parameters(values)
    zeros_state = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, SIR_STATE_DIM], tf.float64
    )
    zeros_state_covariance = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, SIR_STATE_DIM, SIR_STATE_DIM], tf.float64
    )
    d_observation_covariance = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, SIR_OBSERVATION_DIM, SIR_OBSERVATION_DIM],
        tf.float64,
    )
    d_observation_covariance = tf.tensor_scatter_nd_update(
        d_observation_covariance,
        indices=tf.stack(
            (
                tf.range(batch_size, dtype=tf.int32),
                tf.fill([batch_size], tf.constant(2, tf.int32)),
            ),
            axis=1,
        ),
        updates=2.0 * observation_covariance,
    )

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        next_state, _state_jacobian, _source_jacobian = (
            sir_rk4_transition_value_state_source_jacobians(values, previous)
        )
        return next_state + innovation

    def observe(states: tf.Tensor) -> tf.Tensor:
        return tf.einsum("oi,bri->bro", _OBSERVATION_MATRIX, states)

    def transition_state_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return sir_rk4_transition_value_state_source_jacobians(values, previous)[1]

    def transition_innovation_jacobian(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(
            tf.eye(SIR_STATE_DIM, dtype=tf.float64)[None, None, :, :],
            [batch_size, tf.shape(innovation)[1], SIR_STATE_DIM, SIR_STATE_DIM],
        )

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        del innovation
        return sir_rk4_transition_value_state_source_jacobians(values, previous)[2]

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(
            _OBSERVATION_MATRIX[None, None, :, :],
            [batch_size, tf.shape(states)[1], SIR_OBSERVATION_DIM, SIR_STATE_DIM],
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros(
            [batch_size, SIR_PARAMETER_DIM, tf.shape(states)[1], SIR_OBSERVATION_DIM],
            tf.float64,
        )

    model = TFBatchedStructuralStateSpace(
        initial_mean=tf.broadcast_to(_INITIAL_MEAN[None, :], [batch_size, SIR_STATE_DIM]),
        initial_covariance=tf.broadcast_to(
            _INITIAL_COVARIANCE[None, :, :],
            [batch_size, SIR_STATE_DIM, SIR_STATE_DIM],
        ),
        innovation_covariance=tf.broadcast_to(
            _PROCESS_COVARIANCE[None, :, :],
            [batch_size, SIR_STATE_DIM, SIR_STATE_DIM],
        ),
        observation_covariance=observation_covariance,
        transition_fn=transition,
        observation_fn=observe,
        name="parameterized_austria_sir_principal_sqrt_ukf_y1_y20",
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=zeros_state,
        d_initial_covariance=zeros_state_covariance,
        d_innovation_covariance=zeros_state_covariance,
        d_observation_covariance=d_observation_covariance,
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="parameterized_austria_sir_manual_rk4_source_derivatives",
    )
    return model, derivatives


def sir_ukf_likelihood_value_score_status(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    values = _rank2_theta(theta)
    model, derivatives = _build_ukf_model_and_derivatives(values)
    value, score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            values,
            tf.convert_to_tensor(observations, tf.float64),
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, tf.float64),
            innovation_floor=_MIN_VARIANCE,
            spectral_gap_tolerance=tf.constant(1.0e-8, tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, tf.float64),
            principal_sqrt_backend="tensorflow_eigh",
            jitter=tf.constant(0.0, tf.float64),
        )
    )
    valid = tf.logical_and(
        tf.equal(diagnostics["principal_sqrt_target_valid_count"], 1),
        tf.logical_and(
            tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1)
        ),
    )
    return value, score, {
        "status_code": tf.where(valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)),
        "valid_pre_regularized_score": valid,
        "floor_count_value": diagnostics["placement_floor_count"] + diagnostics["innovation_floor_count"],
        "min_innovation_eigenvalue": diagnostics["min_innovation_eigenvalue"],
        "principal_sqrt_target_row_class_code": diagnostics["principal_sqrt_target_row_class_code"],
    }


def sir_ukf_likelihood_value_score(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, _status = sir_ukf_likelihood_value_score_status(
        theta, observations=observations
    )
    return value, score


def _cholesky_derivative(factor: tf.Tensor, d_covariance: tf.Tensor) -> tf.Tensor:
    dimension = int(factor.shape[-1])
    factor_inverse = tf.linalg.triangular_solve(
        factor,
        tf.eye(dimension, batch_shape=[tf.shape(factor)[0]], dtype=tf.float64),
    )
    inner = tf.einsum(
        "bij,bpjk,bkl->bpil",
        factor_inverse,
        d_covariance,
        tf.linalg.matrix_transpose(factor_inverse),
    )
    lower = tf.linalg.band_part(inner, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    return tf.einsum("bij,bpjk->bpik", factor, phi)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _weighted_covariance(centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return _symmetrize(tf.einsum("r,bri,brj->bij", weights, centered, centered))


def _weighted_covariance_derivative(
    centered: tf.Tensor, d_centered: tf.Tensor, weights: tf.Tensor
) -> tf.Tensor:
    return _symmetrize(
        tf.einsum("r,bpri,brj->bpij", weights, d_centered, centered)
        + tf.einsum("r,bri,bprj->bpij", weights, centered, d_centered)
    )


def sir_sgqf_likelihood_value_score_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the fixed 37-point level-2 SGQF likelihood and manual score."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    cloud_points = tf.convert_to_tensor(nodes, tf.float64)
    cloud_weights = tf.convert_to_tensor(weights, tf.float64)
    batch_size = int(values.shape[0])
    _kappa, _nu, observation_covariance = sir_scaled_parameters(values)
    d_observation_covariance = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, SIR_OBSERVATION_DIM, SIR_OBSERVATION_DIM],
        tf.float64,
    )
    d_observation_covariance = tf.tensor_scatter_nd_update(
        d_observation_covariance,
        indices=tf.stack(
            (tf.range(batch_size, dtype=tf.int32), tf.fill([batch_size], 2)), axis=1
        ),
        updates=2.0 * observation_covariance,
    )
    mean = tf.broadcast_to(_INITIAL_MEAN[None, :], [batch_size, SIR_STATE_DIM])
    covariance = tf.broadcast_to(
        _INITIAL_COVARIANCE[None, :, :],
        [batch_size, SIR_STATE_DIM, SIR_STATE_DIM],
    )
    d_mean = tf.zeros([batch_size, SIR_PARAMETER_DIM, SIR_STATE_DIM], tf.float64)
    d_covariance = tf.zeros(
        [batch_size, SIR_PARAMETER_DIM, SIR_STATE_DIM, SIR_STATE_DIM], tf.float64
    )
    total_value = tf.zeros([batch_size], tf.float64)
    total_score = tf.zeros([batch_size, SIR_PARAMETER_DIM], tf.float64)
    valid = tf.ones([batch_size], tf.bool)
    min_predictive = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_innovation = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_filtered = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))

    def condition(index, *_loop_values):
        return index < tf.shape(y)[0]

    def body(
        index,
        current_mean,
        current_covariance,
        current_d_mean,
        current_d_covariance,
        value_total,
        score_total,
        current_valid,
        current_min_predictive,
        current_min_innovation,
        current_min_filtered,
    ):
        previous_factor = tf.linalg.cholesky(current_covariance)
        d_previous_factor = _cholesky_derivative(previous_factor, current_d_covariance)
        previous_points = current_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, previous_factor
        )
        d_previous_points = current_d_mean[:, :, None, :] + tf.einsum(
            "rd,bpnd->bprn", cloud_points, d_previous_factor
        )
        transition_values, transition_state_jacobian, d_transition_direct = (
            sir_rk4_transition_value_state_source_jacobians(values, previous_points)
        )
        d_transition_values = (
            tf.einsum("brij,bprj->bpri", transition_state_jacobian, d_previous_points)
            + d_transition_direct
        )
        predicted_mean = tf.einsum("r,bri->bi", cloud_weights, transition_values)
        d_predicted_mean = tf.einsum("r,bpri->bpi", cloud_weights, d_transition_values)
        centered_predicted = transition_values - predicted_mean[:, None, :]
        d_centered_predicted = d_transition_values - d_predicted_mean[:, :, None, :]
        predicted_covariance = _symmetrize(
            _PROCESS_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_predicted, cloud_weights)
        )
        d_predicted_covariance = _weighted_covariance_derivative(
            centered_predicted, d_centered_predicted, cloud_weights
        )
        predicted_factor = tf.linalg.cholesky(predicted_covariance)
        d_predicted_factor = _cholesky_derivative(
            predicted_factor, d_predicted_covariance
        )
        predictive_points = predicted_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, predicted_factor
        )
        d_predictive_points = d_predicted_mean[:, :, None, :] + tf.einsum(
            "rd,bpnd->bprn", cloud_points, d_predicted_factor
        )
        observation_points = tf.einsum(
            "oi,bri->bro", _OBSERVATION_MATRIX, predictive_points
        )
        d_observation_points = tf.einsum(
            "oi,bpri->bpro", _OBSERVATION_MATRIX, d_predictive_points
        )
        observation_mean = tf.einsum("r,bro->bo", cloud_weights, observation_points)
        d_observation_mean = tf.einsum(
            "r,bpro->bpo", cloud_weights, d_observation_points
        )
        centered_observation = observation_points - observation_mean[:, None, :]
        d_centered_observation = (
            d_observation_points - d_observation_mean[:, :, None, :]
        )
        innovation_covariance = _symmetrize(
            observation_covariance
            + _weighted_covariance(centered_observation, cloud_weights)
        )
        d_innovation_covariance = (
            d_observation_covariance
            + _weighted_covariance_derivative(
                centered_observation, d_centered_observation, cloud_weights
            )
        )
        centered_state = predictive_points - predicted_mean[:, None, :]
        d_centered_state = d_predictive_points - d_predicted_mean[:, :, None, :]
        cross_covariance = tf.einsum(
            "r,bri,brj->bij", cloud_weights, centered_state, centered_observation
        )
        d_cross_covariance = (
            tf.einsum(
                "r,bpri,brj->bpij",
                cloud_weights,
                d_centered_state,
                centered_observation,
            )
            + tf.einsum(
                "r,bri,bprj->bpij",
                cloud_weights,
                centered_state,
                d_centered_observation,
            )
        )
        innovation = y[index][None, :] - observation_mean
        d_innovation = -d_observation_mean
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.eye(
                SIR_OBSERVATION_DIM,
                batch_shape=[batch_size],
                dtype=tf.float64,
            ),
        )
        innovation_solve = tf.linalg.matvec(innovation_precision, innovation)
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=1
        )
        increment = -0.5 * (
            tf.cast(SIR_OBSERVATION_DIM, tf.float64) * _LOG_TWO_PI
            + log_det
            + tf.reduce_sum(innovation * innovation_solve, axis=1)
        )
        trace_term = tf.linalg.trace(
            tf.einsum("bij,bpjk->bpik", innovation_precision, d_innovation_covariance)
        )
        innovation_term = 2.0 * tf.einsum(
            "bpi,bi->bp", d_innovation, innovation_solve
        )
        quadratic_term = tf.einsum(
            "bi,bpij,bj->bp",
            innovation_solve,
            d_innovation_covariance,
            innovation_solve,
        )
        score_increment = -0.5 * (trace_term + innovation_term - quadratic_term)
        gain = tf.matmul(cross_covariance, innovation_precision)
        d_gain = (
            tf.einsum("bpij,bjk->bpik", d_cross_covariance, innovation_precision)
            - tf.einsum(
                "bij,bpjk,bkl->bpil",
                gain,
                d_innovation_covariance,
                innovation_precision,
            )
        )
        filtered_mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
        d_filtered_mean = (
            d_predicted_mean
            + tf.einsum("bpij,bj->bpi", d_gain, innovation)
            + tf.einsum("bij,bpj->bpi", gain, d_innovation)
        )
        filtered_covariance = _symmetrize(
            predicted_covariance
            - gain @ innovation_covariance @ tf.linalg.matrix_transpose(gain)
        )
        d_filtered_covariance = _symmetrize(
            d_predicted_covariance
            - tf.einsum(
                "bpij,bjk,bkl->bpil",
                d_gain,
                innovation_covariance,
                tf.linalg.matrix_transpose(gain),
            )
            - tf.einsum(
                "bij,bpjk,bkl->bpil",
                gain,
                d_innovation_covariance,
                tf.linalg.matrix_transpose(gain),
            )
            - tf.einsum(
                "bij,bjk,bpkl->bpil",
                gain,
                innovation_covariance,
                tf.linalg.matrix_transpose(d_gain),
            )
        )
        predictive_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(predicted_covariance), axis=1
        )
        innovation_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(innovation_covariance), axis=1
        )
        filtered_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(filtered_covariance), axis=1
        )
        step_valid = tf.logical_and(
            predictive_eigenvalue > _MIN_VARIANCE,
            tf.logical_and(
                innovation_eigenvalue > _MIN_VARIANCE,
                filtered_eigenvalue > _MIN_VARIANCE,
            ),
        )
        step_valid = tf.logical_and(
            step_valid,
            tf.logical_and(
                tf.math.is_finite(increment),
                tf.reduce_all(tf.math.is_finite(score_increment), axis=1),
            ),
        )
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            d_filtered_mean,
            d_filtered_covariance,
            value_total + increment,
            score_total + score_increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predictive_eigenvalue),
            tf.minimum(current_min_innovation, innovation_eigenvalue),
            tf.minimum(current_min_filtered, filtered_eigenvalue),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            mean,
            covariance,
            d_mean,
            d_covariance,
            total_value,
            total_score,
            valid,
            min_predictive,
            min_innovation,
            min_filtered,
        ),
        parallel_iterations=1,
    )
    return result[5], result[6], {
        "status_code": tf.where(result[7], tf.zeros_like(result[5], tf.int32), tf.ones_like(result[5], tf.int32)),
        "valid_pre_regularized_score": result[7],
        "floor_count_value": tf.zeros_like(result[5], tf.int32),
        "min_predictive_eigenvalue": result[8],
        "min_innovation_eigenvalue": result[9],
        "min_filtered_eigenvalue": result[10],
    }


def sir_sgqf_likelihood_value_only_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the fixed 37-point SGQF scalar without score propagation."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    cloud_points = tf.convert_to_tensor(nodes, tf.float64)
    cloud_weights = tf.convert_to_tensor(weights, tf.float64)
    batch_size = int(values.shape[0])
    _kappa, _nu, observation_covariance = sir_scaled_parameters(values)
    mean = tf.broadcast_to(_INITIAL_MEAN[None, :], [batch_size, SIR_STATE_DIM])
    covariance = tf.broadcast_to(
        _INITIAL_COVARIANCE[None, :, :],
        [batch_size, SIR_STATE_DIM, SIR_STATE_DIM],
    )
    total_value = tf.zeros([batch_size], tf.float64)
    valid = tf.ones([batch_size], tf.bool)
    min_predictive = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_innovation = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    min_filtered = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))
    observation_identity = tf.eye(
        SIR_OBSERVATION_DIM, batch_shape=[batch_size], dtype=tf.float64
    )

    def body(index, current_mean, current_covariance, value_total,
             current_valid, current_min_predictive, current_min_innovation,
             current_min_filtered):
        previous_factor = tf.linalg.cholesky(current_covariance)
        previous_points = current_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, previous_factor
        )
        transition_values = sir_rk4_transition_value(values, previous_points)
        predicted_mean = tf.einsum(
            "r,bri->bi", cloud_weights, transition_values
        )
        centered_predicted = transition_values - predicted_mean[:, None, :]
        predicted_covariance = _symmetrize(
            _PROCESS_COVARIANCE[None, :, :]
            + _weighted_covariance(centered_predicted, cloud_weights)
        )
        predicted_factor = tf.linalg.cholesky(predicted_covariance)
        predictive_points = predicted_mean[:, None, :] + tf.einsum(
            "rd,bnd->brn", cloud_points, predicted_factor
        )
        observation_points = tf.einsum(
            "oi,bri->bro", _OBSERVATION_MATRIX, predictive_points
        )
        observation_mean = tf.einsum(
            "r,bro->bo", cloud_weights, observation_points
        )
        centered_observation = observation_points - observation_mean[:, None, :]
        innovation_covariance = _symmetrize(
            observation_covariance
            + _weighted_covariance(centered_observation, cloud_weights)
        )
        centered_state = predictive_points - predicted_mean[:, None, :]
        cross_covariance = tf.einsum(
            "r,bri,brj->bij",
            cloud_weights,
            centered_state,
            centered_observation,
        )
        innovation = y[index][None, :] - observation_mean
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        precision = tf.linalg.cholesky_solve(
            innovation_factor, observation_identity
        )
        solve = tf.linalg.matvec(precision, innovation)
        increment = -0.5 * (
            tf.cast(SIR_OBSERVATION_DIM, tf.float64) * _LOG_TWO_PI
            + 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=1
            )
            + tf.reduce_sum(innovation * solve, axis=1)
        )
        gain = cross_covariance @ precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = _symmetrize(
            predicted_covariance
            - gain @ innovation_covariance @ tf.linalg.matrix_transpose(gain)
        )
        predictive_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(predicted_covariance), axis=1
        )
        innovation_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(innovation_covariance), axis=1
        )
        filtered_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(filtered_covariance), axis=1
        )
        step_valid = tf.logical_and(
            predictive_eigenvalue > _MIN_VARIANCE,
            tf.logical_and(
                innovation_eigenvalue > _MIN_VARIANCE,
                filtered_eigenvalue > _MIN_VARIANCE,
            ),
        )
        step_valid = tf.logical_and(step_valid, tf.math.is_finite(increment))
        return (
            index + 1,
            filtered_mean,
            filtered_covariance,
            value_total + increment,
            tf.logical_and(current_valid, step_valid),
            tf.minimum(current_min_predictive, predictive_eigenvalue),
            tf.minimum(current_min_innovation, innovation_eigenvalue),
            tf.minimum(current_min_filtered, filtered_eigenvalue),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.shape(y)[0],
        body,
        (
            tf.constant(0, tf.int32), mean, covariance, total_value, valid,
            min_predictive, min_innovation, min_filtered,
        ),
        parallel_iterations=1,
    )
    return result[3], {
        "status_code": tf.where(
            result[4], tf.zeros([batch_size], tf.int32), tf.ones([batch_size], tf.int32)
        ),
        "valid_value": result[4],
        "min_predictive_eigenvalue": result[5],
        "min_innovation_eigenvalue": result[6],
        "min_filtered_eigenvalue": result[7],
    }


def sir_sgqf_posterior_value_only(
    theta: Any,
    *,
    observations: tf.Tensor,
    nodes: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    likelihood, _status = sir_sgqf_likelihood_value_only_status(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    prior, _ = sir_prior_value_score(theta)
    jacobian, _ = sir_identity_chart_jacobian_value_score(theta)
    return likelihood + prior + jacobian


def sir_sgqf_likelihood_value_score(
    theta: Any, *, observations: tf.Tensor, nodes: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, _status = sir_sgqf_likelihood_value_score_status(
        theta, observations=observations, nodes=nodes, weights=weights
    )
    return value, score


def _posterior_value_score(
    likelihood_fn: Any, theta: Any
) -> tuple[tf.Tensor, tf.Tensor]:
    likelihood_value, likelihood_score = likelihood_fn(theta)
    prior_value, prior_score = sir_prior_value_score(theta)
    jacobian_value, jacobian_score = sir_identity_chart_jacobian_value_score(theta)
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
    )


class SIRUKFNeuTraAdapter:
    dtype = tf.float64
    parameter_dim = SIR_PARAMETER_DIM
    parameter_names = SIR_PARAMETER_NAMES

    def __init__(self, *, observations: tf.Tensor, contract: SSMTargetContract) -> None:
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.contract = contract
        self.target_scope = SIR_UKF_SCOPE
        self._adapter_signature = _semantic_hash(
            {
                "schema": "bayesfilter.testing.sir_ukf_neutra_adapter.v1",
                "target_signature": stable_ssm_target_signature(contract),
                "dtype": self.dtype.name,
                "parameter_names": self.parameter_names,
            }
        )

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_batched_principal_sqrt_ukf_parameterized_sir",
            evidence_path="bayesfilter/testing/sir_filter_neutra_target_design_tf.py",
            target_scope=self.target_scope,
            nonclaims=SIR_UKF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        return self.log_prob_and_grad(theta)[0]

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return _posterior_value_score(
            lambda values: sir_ukf_likelihood_value_score(
                values, observations=self.observations
            ),
            theta,
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score, status = sir_ukf_likelihood_value_score_status(
            theta, observations=self.observations
        )
        prior_value, prior_score = sir_prior_value_score(theta)
        return likelihood_value + prior_value, likelihood_score + prior_score, status

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = sir_ukf_likelihood_value_score_status(
            theta, observations=self.observations
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
        }


class SIRSGQFNeuTraAdapter:
    dtype = tf.float64
    parameter_dim = SIR_PARAMETER_DIM
    parameter_names = SIR_PARAMETER_NAMES

    def __init__(self, *, observations: tf.Tensor, contract: SSMTargetContract) -> None:
        cloud = tf_fixed_sgqf_level2_axis_cloud(SIR_STATE_DIM)
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.nodes = tf.convert_to_tensor(cloud.points, tf.float64)
        self.weights = tf.convert_to_tensor(cloud.weights, tf.float64)
        self.contract = contract
        self.target_scope = SIR_SGQF_SCOPE
        self._adapter_signature = _semantic_hash(
            {
                "schema": "bayesfilter.testing.sir_sgqf_neutra_adapter.v1",
                "target_signature": stable_ssm_target_signature(contract),
                "dtype": self.dtype.name,
                "parameter_names": self.parameter_names,
                "point_count": cloud.point_count,
            }
        )

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_fixed_level2_sgqf_parameterized_sir",
            evidence_path="bayesfilter/testing/sir_filter_neutra_target_design_tf.py",
            target_scope=self.target_scope,
            nonclaims=SIR_SGQF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        return self.log_prob_and_grad(theta)[0]

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return _posterior_value_score(
            lambda values: sir_sgqf_likelihood_value_score(
                values,
                observations=self.observations,
                nodes=self.nodes,
                weights=self.weights,
            ),
            theta,
        )

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        likelihood_value, likelihood_score, status = sir_sgqf_likelihood_value_score_status(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        prior_value, prior_score = sir_prior_value_score(theta)
        return likelihood_value + prior_value, likelihood_score + prior_score, status

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = sir_sgqf_likelihood_value_score_status(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
        }


class SIRUKFLikelihoodRecomposer:
    def __init__(self, adapter: SIRUKFNeuTraAdapter) -> None:
        self.observations = adapter.observations

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return sir_ukf_likelihood_value_score(theta, observations=self.observations)


class SIRSGQFLikelihoodRecomposer:
    def __init__(self, adapter: SIRSGQFNeuTraAdapter) -> None:
        self.observations = adapter.observations
        self.nodes = adapter.nodes
        self.weights = adapter.weights

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return sir_sgqf_likelihood_value_score(
            theta,
            observations=self.observations,
            nodes=self.nodes,
            weights=self.weights,
        )


def make_sir_ukf_neutra_adapter(
    *, observations: tf.Tensor | None = None
) -> SIRUKFNeuTraAdapter:
    y = _validated_observations(observations)
    contract = make_sir_target_contract(
        cell_id="SIR-UKF",
        data_hash=_tensor_hash(y),
        filter_kind="principal_sqrt_ukf",
        filter_payload={
            "backend": "tf_principal_sqrt_ukf",
            "principal_sqrt_backend": "tensorflow_eigh_xla_portable",
            "score": "manual_sir_rk4_state_parameter_jacobians",
        },
    )
    return SIRUKFNeuTraAdapter(observations=y, contract=contract)


def make_sir_sgqf_neutra_adapter(
    *, observations: tf.Tensor | None = None
) -> SIRSGQFNeuTraAdapter:
    y = _validated_observations(observations)
    cloud = tf_fixed_sgqf_level2_axis_cloud(SIR_STATE_DIM)
    cloud_hash = hashlib.sha256(
        bytes(tf.io.serialize_tensor(cloud.points).numpy())
        + bytes(tf.io.serialize_tensor(cloud.weights).numpy())
    ).hexdigest()
    contract = make_sir_target_contract(
        cell_id="SIR-SGQF",
        data_hash=_tensor_hash(y),
        filter_kind="fixed_level2_sgqf",
        filter_payload={
            "point_count": cloud.point_count,
            "negative_weight_count": cloud.negative_weight_count,
            "cloud_hash": cloud_hash,
            "score": "manual_cholesky_and_sir_rk4_derivatives",
        },
    )
    return SIRSGQFNeuTraAdapter(observations=y, contract=contract)


def make_sir_target_contract(
    *,
    cell_id: str,
    data_hash: str,
    filter_kind: str,
    filter_payload: Mapping[str, Any],
) -> SSMTargetContract:
    shape = SSMStaticShape(
        horizon=SIR_HORIZON,
        state_dim=SIR_STATE_DIM,
        observation_dim=SIR_OBSERVATION_DIM,
        innovation_dim=SIR_STATE_DIM,
        parameter_dim=SIR_PARAMETER_DIM,
    )
    model_semantics = {
        "model_id": "zhao-cui-austria-sir-three-log-scale-extension-v1",
        "source_boundary": "paper_SIR_fixed_parameters_BayesFilter_parameter_extension",
        "theta_truth": (0.0, 0.0, 0.0),
        "truth_role": "explanatory_only",
        "state_order": tuple(
            item for index in range(1, 10) for item in (f"S{index}", f"I{index}")
        ),
        "observation_order": tuple(f"I{index}" for index in range(1, 10)),
        "initial_mean": tuple(float(item) for item in _INITIAL_MEAN.numpy()),
        "initial_covariance": "I18",
        "process_covariance": "I18",
        "base_observation_covariance": "100*I9",
        "theta_scaling": (
            "kappa=0.1*exp(theta0)",
            "nu=18*exp(theta1)",
            "R=100*exp(2*theta2)*I9",
        ),
        "rk4_delta": 0.02,
        "rk4_internal_step": 0.005,
        "rk4_fourth_stage": "state_plus_half_step_k3_author_source_variant",
        "transition_density": "unprojected_additive_gaussian",
        "fixture_clipping_inactive": True,
        "time_order": "x0_prior_then_transition_and_y1_through_y20",
    }
    problem = BayesianSSMProblem(
        problem_id=f"{cell_id}-three-log-scale-parameterized-austria-SIR",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id=SIR_DATASET_ID,
            observation_shape=(SIR_HORIZON, SIR_OBSERVATION_DIM),
            data_hash=f"sha256:{data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": f"sha256:{_semantic_hash(model_semantics)}",
        },
    )
    chart_semantics = {
        "transform_id": "identity-three-log-scale-chart",
        "parameter_order": SIR_PARAMETER_NAMES,
        "log_abs_det_jacobian": 0.0,
    }
    chart = ParameterChart(
        parameter_names=SIR_PARAMETER_NAMES,
        unconstrained_dim=SIR_PARAMETER_DIM,
        constrained_shape=(SIR_PARAMETER_DIM,),
        transform_manifest={
            **chart_semantics,
            "transform_hash": f"sha256:{_semantic_hash(chart_semantics)}",
        },
        log_jacobian_convention="included_in_chart",
    )
    prior_semantics = {
        "prior_id": "independent-normal-log-scale-prior-v1",
        "mean": (0.0, 0.0, 0.0),
        "scale": (0.5, 0.5, 0.5),
        "parameter_order": SIR_PARAMETER_NAMES,
    }
    prior = ParameterPrior(
        prior_manifest={
            **prior_semantics,
            "prior_hash": f"sha256:{_semantic_hash(prior_semantics)}",
        },
        support_policy="unbounded",
        log_density_authority="graph_native",
    )
    filter_semantics = {
        "filter_id": f"parameterized-austria-SIR-{filter_kind}-y1-y20-v1",
        "filter_kind": filter_kind,
        "observed_data_likelihood": True,
        "time_order": "transition_then_observe_y1_to_y20",
        "innovation_floor": 1.0e-12,
        **dict(filter_payload),
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "parameterized_spatial_sir_rk4",
            "additive_gaussian_process_observation",
            filter_kind,
        ),
        deterministic_target_policy="deterministic",
        approximation_semantics="deterministic_approximation",
        filter_manifest={
            **filter_semantics,
            "filter_hash": f"sha256:{_semantic_hash(filter_semantics)}",
        },
    )
    return SSMTargetContract(
        problem=problem,
        chart=chart,
        prior=prior,
        filter_program=filter_program,
        frozen_transport=None,
    )


def _validated_observations(observations: tf.Tensor | None) -> tf.Tensor:
    if observations is None:
        _states, observations, _all = generate_frozen_sir_dataset_tf()
    y = tf.convert_to_tensor(observations, tf.float64)
    if y.shape != (SIR_HORIZON, SIR_OBSERVATION_DIM):
        raise ValueError("SIR observations require frozen shape [20, 9]")
    if _tensor_hash(y) != SIR_OBSERVATION_SHA256:
        raise ValueError("SIR observations do not match frozen y1:y20 hash")
    return y


def _rank2_theta(theta: Any) -> tf.Tensor:
    values = tf.convert_to_tensor(theta)
    if values.dtype != tf.float64:
        raise ValueError("SIR target requires float64 theta")
    if values.shape.rank != 2 or values.shape[-1] != SIR_PARAMETER_DIM:
        raise ValueError("SIR target requires theta shape [batch, 3]")
    if values.shape[0] is None:
        raise ValueError("SIR target requires a static batch dimension")
    return values


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
