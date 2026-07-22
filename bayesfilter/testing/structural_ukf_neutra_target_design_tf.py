"""Graph-native structural UKF target-design kernels for NeuTra P5."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability
from bayesfilter.nonlinear.batched_svd_sigma_point_tf import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralFirstDerivatives,
    TFBatchedStructuralStateSpace,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_unit_sigma_point_rule
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


STRUCTURAL_PARAMETER_NAMES = (
    "rho_source_probit",
    "sigma_source_probit",
    "phi_source_probit",
    "gamma_source_probit",
    "R_source_probit",
)
STRUCTURAL_PARAMETER_LOWER = tf.constant([0.05, 0.05, 0.05, 0.02, 0.02], tf.float64)
STRUCTURAL_PARAMETER_UPPER = tf.constant([0.98, 1.25, 0.98, 1.00, 1.00], tf.float64)
STRUCTURAL_PARAMETER_WIDTH = STRUCTURAL_PARAMETER_UPPER - STRUCTURAL_PARAMETER_LOWER
STRUCTURAL_TRUTH_PHYSICAL = tf.constant([0.8, 0.5, 0.7, 0.4, 0.25], tf.float64)
STRUCTURAL_INITIAL_MEAN = tf.zeros([2], tf.float64)
STRUCTURAL_INITIAL_COVARIANCE = tf.linalg.diag(tf.constant([0.04, 0.09], tf.float64))
STRUCTURAL_NEGATIVE_CONTROL_K_VARIANCE = tf.constant(0.04, tf.float64)
STRUCTURAL_FINAL_HORIZON = 100
STRUCTURAL_FINAL_SEED = (20260716, 15001)
STRUCTURAL_FINAL_STATE_SHA256 = (
    "fe77f0e0000db93281116e7e81ddd303e9706b9e402bfaf7141a1aa1005c0ca9"
)
STRUCTURAL_FINAL_OBSERVATION_SHA256 = (
    "ab7885b135d8098c6e516e06733ef99399ea07f4a39292670b578da4a0efbae3"
)
STRUCTURAL_UKF_SCOPE = "STR-UKF-five-probit-T100-structural-innovation-v1"
STRUCTURAL_UKF_NONCLAIMS = (
    "principal-square-root UKF approximate structural-model posterior",
    "local target-design information is not global identifiability",
    "no filter exactness, HMC convergence, NeuTra, calibration, or readiness claim",
)

_LOG_TWO_PI = tf.constant(math.log(2.0 * math.pi), tf.float64)
_SQRT_TWO = tf.constant(math.sqrt(2.0), tf.float64)
_OBSERVATION_ROW = tf.constant([1.0, 1.0], tf.float64)


def _rank2_theta(theta: Any) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank != 2 or values.shape[-1] != 5:
        raise ValueError("structural target requires theta shape [batch, 5]")
    if values.shape[0] is None:
        raise ValueError("structural target requires a static batch dimension")
    return values


def structural_truth_source() -> tf.Tensor:
    probabilities = (STRUCTURAL_TRUTH_PHYSICAL - STRUCTURAL_PARAMETER_LOWER) / (
        STRUCTURAL_PARAMETER_WIDTH
    )
    return _SQRT_TWO * tf.math.erfinv(2.0 * probabilities - 1.0)


def structural_source_chart(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    values = _rank2_theta(theta)
    return structural_source_chart_dtype(values)


def structural_source_chart_dtype(theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the established source chart without changing the input dtype."""

    values = tf.convert_to_tensor(theta)
    if values.shape.rank != 2 or values.shape[-1] != 5:
        raise ValueError("structural target requires theta shape [batch, 5]")
    if not values.dtype.is_floating:
        raise ValueError("structural target requires floating-point theta")
    lower = tf.cast(STRUCTURAL_PARAMETER_LOWER, values.dtype)
    width = tf.cast(STRUCTURAL_PARAMETER_WIDTH, values.dtype)
    sqrt_two = tf.sqrt(tf.constant(2.0, values.dtype))
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, values.dtype))
    probabilities = 0.5 * (1.0 + tf.math.erf(values / sqrt_two))
    density = tf.exp(-0.5 * tf.square(values) - 0.5 * log_two_pi)
    return lower[None, :] + width[None, :] * probabilities, width[None, :] * density


def structural_source_uniform_prior_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = _rank2_theta(theta)
    log_volume = tf.reduce_sum(tf.math.log(STRUCTURAL_PARAMETER_WIDTH))
    return tf.fill(tf.shape(values)[:1], -log_volume), tf.zeros_like(values)


def structural_source_probit_jacobian_value_score(
    theta: Any,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = _rank2_theta(theta)
    log_density = -0.5 * tf.square(values) - 0.5 * _LOG_TWO_PI
    value = tf.reduce_sum(
        tf.math.log(STRUCTURAL_PARAMETER_WIDTH)[None, :] + log_density,
        axis=1,
    )
    return value, -values


def simulate_structural_trajectories_tf(
    physical_parameters: tf.Tensor,
    *,
    horizon: int,
    seed: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Simulate an independent batch with one compiled time loop."""

    physical = tf.convert_to_tensor(physical_parameters, tf.float64)
    if physical.shape.rank != 2 or physical.shape[-1] != 5:
        raise ValueError("physical_parameters must have shape [batch, 5]")
    if physical.shape[0] is None:
        raise ValueError("simulator requires a static batch dimension")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    batch_size = int(physical.shape[0])
    roots = tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 3)
    initial_noise = tf.random.stateless_normal([batch_size, 2], roots[0], dtype=tf.float64)
    innovations = tf.random.stateless_normal(
        [horizon, batch_size], roots[1], dtype=tf.float64
    )
    observation_noise = tf.random.stateless_normal(
        [horizon, batch_size], roots[2], dtype=tf.float64
    )
    initial_scale = tf.constant([0.2, 0.3], tf.float64)
    initial_state = STRUCTURAL_INITIAL_MEAN[None, :] + initial_noise * initial_scale[None, :]
    states = tf.TensorArray(tf.float64, size=horizon, clear_after_read=False)
    observations = tf.TensorArray(tf.float64, size=horizon, clear_after_read=False)
    residuals = tf.TensorArray(tf.float64, size=horizon, clear_after_read=False)
    rho, sigma, phi, gamma, observation_variance = tf.unstack(physical, axis=1)

    def body(
        index: tf.Tensor,
        previous: tf.Tensor,
        state_array: tf.TensorArray,
        observation_array: tf.TensorArray,
        residual_array: tf.TensorArray,
    ):
        def initial_value() -> tf.Tensor:
            return previous

        def transition_value() -> tf.Tensor:
            m_value = rho * previous[:, 0] + sigma * innovations[index]
            k_value = phi * previous[:, 1] + gamma * tf.square(m_value)
            return tf.stack([m_value, k_value], axis=1)

        current = tf.cond(tf.equal(index, 0), initial_value, transition_value)
        residual = current[:, 1] - tf.where(
            tf.equal(index, 0),
            current[:, 1],
            phi * previous[:, 1] + gamma * tf.square(current[:, 0]),
        )
        y_value = (
            current[:, 0]
            + current[:, 1]
            + tf.sqrt(observation_variance) * observation_noise[index]
        )
        return (
            index + 1,
            current,
            state_array.write(index, current),
            observation_array.write(index, y_value[:, None]),
            residual_array.write(index, residual),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.constant(horizon, tf.int32),
        body,
        (
            tf.constant(0, tf.int32),
            initial_state,
            states,
            observations,
            residuals,
        ),
        parallel_iterations=1,
    )
    return (
        tf.transpose(result[2].stack(), [1, 0, 2]),
        tf.transpose(result[3].stack(), [1, 0, 2]),
        tf.transpose(result[4].stack(), [1, 0]),
    )


def structural_prior_predictive_tf(
    *, batch_size: int, horizon: int, seed: tf.Tensor
) -> Mapping[str, tf.Tensor]:
    roots = tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 2)
    source = tf.random.stateless_normal([batch_size, 5], roots[0], dtype=tf.float64)
    physical, _derivative = structural_source_chart(source)
    states, observations, residuals = simulate_structural_trajectories_tf(
        physical, horizon=horizon, seed=roots[1]
    )
    magnitude = tf.reduce_max(
        tf.concat([tf.abs(states), tf.abs(observations)], axis=2), axis=[1, 2]
    )
    finite = tf.reduce_all(
        tf.math.is_finite(tf.concat([states, observations], axis=2)), axis=[1, 2]
    )
    return {
        "source": source,
        "physical": physical,
        "states": states,
        "observations": observations,
        "deterministic_residuals": residuals,
        "trajectory_max_magnitude": magnitude,
        "trajectory_all_finite": finite,
    }


def generate_frozen_structural_dataset_tf() -> tuple[tf.Tensor, tf.Tensor]:
    with tf.device("/CPU:0"):
        states, observations, residuals = simulate_structural_trajectories_tf(
            STRUCTURAL_TRUTH_PHYSICAL[None, :],
            horizon=STRUCTURAL_FINAL_HORIZON,
            seed=tf.constant(STRUCTURAL_FINAL_SEED, tf.int32),
        )
    tf.debugging.assert_near(
        residuals, tf.zeros_like(residuals), atol=tf.constant(2.0e-14, tf.float64)
    )
    return states[0], observations[0]


def structural_transition_value(
    theta: Any, previous_points: tf.Tensor, innovation_points: tf.Tensor
) -> tf.Tensor:
    values = _rank2_theta(theta)
    previous = tf.convert_to_tensor(previous_points, tf.float64)
    innovation = tf.convert_to_tensor(innovation_points, tf.float64)
    return structural_transition_value_dtype(values, previous, innovation)


def structural_transition_value_dtype(
    theta: Any, previous_points: tf.Tensor, innovation_points: tf.Tensor
) -> tf.Tensor:
    """Evaluate the Chapter 18b transition in the caller's floating dtype."""

    values = tf.convert_to_tensor(theta)
    previous = tf.convert_to_tensor(previous_points, dtype=values.dtype)
    innovation = tf.convert_to_tensor(innovation_points, dtype=values.dtype)
    if values.shape.rank != 2 or values.shape[-1] != 5:
        raise ValueError("structural target requires theta shape [batch, 5]")
    if previous.shape.rank != 3 or previous.shape[-1] != 2:
        raise ValueError("structural previous points require shape [batch, points, 2]")
    if innovation.shape.rank != 3 or innovation.shape[-1] != 1:
        raise ValueError("structural innovations require shape [batch, points, 1]")
    physical, _derivative = structural_source_chart_dtype(values)
    rho, sigma, phi, gamma, _observation_variance = tf.unstack(physical, axis=1)
    m_value = rho[:, None] * previous[:, :, 0] + sigma[:, None] * innovation[:, :, 0]
    k_value = phi[:, None] * previous[:, :, 1] + gamma[:, None] * tf.square(m_value)
    return tf.stack([m_value, k_value], axis=2)


def structural_transition_tangent_dtype(
    theta: Any,
    previous_points: tf.Tensor,
    innovation_points: tf.Tensor,
    previous_tangent: tf.Tensor,
) -> tf.Tensor:
    """Propagate the total source-coordinate tangent of the structural law."""

    values = tf.convert_to_tensor(theta)
    previous = tf.convert_to_tensor(previous_points, dtype=values.dtype)
    innovation = tf.convert_to_tensor(innovation_points, dtype=values.dtype)
    tangent = tf.convert_to_tensor(previous_tangent, dtype=values.dtype)
    if tangent.shape.rank != 4 or tangent.shape[-2:] != (2, 5):
        raise ValueError(
            "structural previous tangent requires shape [batch, points, 2, 5]"
        )
    physical, dphysical = structural_source_chart_dtype(values)
    rho, sigma, phi, gamma, _observation_variance = tf.unstack(physical, axis=1)
    next_points = structural_transition_value_dtype(values, previous, innovation)
    basis = tf.eye(5, dtype=values.dtype)
    direct_m = (
        previous[:, :, 0, None]
        * dphysical[:, 0, None, None]
        * basis[0][None, None, :]
        + innovation[:, :, 0, None]
        * dphysical[:, 1, None, None]
        * basis[1][None, None, :]
    )
    m_tangent = rho[:, None, None] * tangent[:, :, 0, :] + direct_m
    direct_k = (
        previous[:, :, 1, None]
        * dphysical[:, 2, None, None]
        * basis[2][None, None, :]
        + tf.square(next_points[:, :, 0, None])
        * dphysical[:, 3, None, None]
        * basis[3][None, None, :]
    )
    k_tangent = (
        phi[:, None, None] * tangent[:, :, 1, :]
        + 2.0
        * gamma[:, None, None]
        * next_points[:, :, 0, None]
        * m_tangent
        + direct_k
    )
    return tf.stack([m_tangent, k_tangent], axis=2)


def structural_transition_residual(
    theta: Any,
    previous_points: tf.Tensor,
    next_points: tf.Tensor,
) -> tf.Tensor:
    return structural_transition_residual_dtype(theta, previous_points, next_points)


def structural_transition_residual_dtype(
    theta: Any,
    previous_points: tf.Tensor,
    next_points: tf.Tensor,
) -> tf.Tensor:
    """Return the pointwise Chapter 18b support residual in the input dtype."""

    values = tf.convert_to_tensor(theta)
    previous = tf.convert_to_tensor(previous_points, dtype=values.dtype)
    next_values = tf.convert_to_tensor(next_points, dtype=values.dtype)
    physical, _derivative = structural_source_chart_dtype(values)
    _rho, _sigma, phi, gamma, _observation_variance = tf.unstack(physical, axis=1)
    return (
        next_values[:, :, 1]
        - phi[:, None] * previous[:, :, 1]
        - gamma[:, None] * tf.square(next_values[:, :, 0])
    )[:, :, None]


def structural_observation_log_density_dtype(
    theta: Any, state_points: tf.Tensor, observation: tf.Tensor
) -> tf.Tensor:
    """Evaluate the established scalar observation density pointwise."""

    values = tf.convert_to_tensor(theta)
    states = tf.convert_to_tensor(state_points, dtype=values.dtype)
    observed = tf.convert_to_tensor(observation, dtype=values.dtype)
    physical, _derivative = structural_source_chart_dtype(values)
    observation_variance = physical[:, 4, None]
    residual = observed[:, None, 0] - tf.reduce_sum(states, axis=2)
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, values.dtype))
    return -0.5 * (
        tf.square(residual) / observation_variance
        + tf.math.log(observation_variance)
        + log_two_pi
    )


def structural_observation_log_density_tangent_dtype(
    theta: Any,
    state_points: tf.Tensor,
    state_tangent: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    """Return the total source-coordinate tangent of the observation density."""

    values = tf.convert_to_tensor(theta)
    states = tf.convert_to_tensor(state_points, dtype=values.dtype)
    tangent = tf.convert_to_tensor(state_tangent, dtype=values.dtype)
    observed = tf.convert_to_tensor(observation, dtype=values.dtype)
    physical, dphysical = structural_source_chart_dtype(values)
    observation_variance = physical[:, 4, None]
    residual = observed[:, None, 0] - tf.reduce_sum(states, axis=2)
    state_term = (
        residual[:, :, None]
        / observation_variance[:, :, None]
        * tf.reduce_sum(tangent, axis=2)
    )
    variance_score = 0.5 * (
        tf.square(residual) / tf.square(observation_variance)
        - 1.0 / observation_variance
    )
    variance_term = (
        variance_score[:, :, None]
        * dphysical[:, 4, None, None]
        * tf.eye(5, dtype=values.dtype)[4][None, None, :]
    )
    return state_term + variance_term


def _initial_observation_update(
    theta: tf.Tensor, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    values = _rank2_theta(theta)
    batch_size = int(values.shape[0])
    physical, dphysical = structural_source_chart(values)
    observation_variance = physical[:, 4]
    d_observation_variance = dphysical[:, 4]
    projected_covariance = tf.constant(0.13, tf.float64)
    innovation_variance = projected_covariance + observation_variance
    y0 = tf.reshape(tf.convert_to_tensor(observations, tf.float64)[0], [])
    covariance_observation = tf.constant([0.04, 0.09], tf.float64)
    gain = covariance_observation[None, :] / innovation_variance[:, None]
    mean = gain * y0
    covariance = STRUCTURAL_INITIAL_COVARIANCE[None, :, :] - (
        covariance_observation[None, :, None]
        * covariance_observation[None, None, :]
        / innovation_variance[:, None, None]
    )
    d_mean = tf.zeros([batch_size, 5, 2], tf.float64)
    d_mean_R = -gain * (d_observation_variance / innovation_variance)[:, None] * y0
    d_mean = tf.tensor_scatter_nd_update(
        d_mean,
        tf.stack([tf.range(batch_size), tf.fill([batch_size], 4)], axis=1),
        d_mean_R,
    )
    d_covariance = tf.zeros([batch_size, 5, 2, 2], tf.float64)
    outer = covariance_observation[:, None] * covariance_observation[None, :]
    d_covariance_R = (
        outer[None, :, :]
        * (d_observation_variance / tf.square(innovation_variance))[:, None, None]
    )
    d_covariance = tf.tensor_scatter_nd_update(
        d_covariance,
        tf.stack([tf.range(batch_size), tf.fill([batch_size], 4)], axis=1),
        d_covariance_R,
    )
    initial_value = -0.5 * (
        _LOG_TWO_PI
        + tf.math.log(innovation_variance)
        + tf.square(y0) / innovation_variance
    )
    initial_score = tf.zeros([batch_size, 5], tf.float64)
    score_R = -0.5 * d_observation_variance * (
        1.0 / innovation_variance - tf.square(y0) / tf.square(innovation_variance)
    )
    initial_score = tf.tensor_scatter_nd_update(
        initial_score,
        tf.stack([tf.range(batch_size), tf.fill([batch_size], 4)], axis=1),
        score_R,
    )
    return mean, covariance, d_mean, d_covariance, initial_value, initial_score


def build_structural_ukf_model_and_derivatives(
    theta: Any, observations: tf.Tensor
) -> tuple[
    TFBatchedStructuralStateSpace,
    TFBatchedStructuralFirstDerivatives,
    tf.Tensor,
    tf.Tensor,
]:
    values = _rank2_theta(theta)
    batch_size = int(values.shape[0])
    physical, dphysical = structural_source_chart(values)
    mean, covariance, d_mean, d_covariance, initial_value, initial_score = (
        _initial_observation_update(values, observations)
    )

    def transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        return structural_transition_value(values, previous, innovation)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(states, axis=2, keepdims=True)

    def residual(
        previous: tf.Tensor, innovation: tf.Tensor, next_state: tf.Tensor
    ) -> tf.Tensor:
        del innovation
        return structural_transition_residual(values, previous, next_state)

    def transition_state_jacobian(
        previous: tf.Tensor, innovation: tf.Tensor
    ) -> tf.Tensor:
        next_state = structural_transition_value(values, previous, innovation)
        rho, _sigma, phi, gamma, _R = tf.unstack(physical, axis=1)
        zeros = tf.zeros_like(next_state[:, :, 0])
        return tf.stack(
            [
                tf.stack([tf.broadcast_to(rho[:, None], tf.shape(zeros)), zeros], axis=2),
                tf.stack(
                    [
                        2.0 * gamma[:, None] * next_state[:, :, 0] * rho[:, None],
                        tf.broadcast_to(phi[:, None], tf.shape(zeros)),
                    ],
                    axis=2,
                ),
            ],
            axis=2,
        )

    def transition_innovation_jacobian(
        previous: tf.Tensor, innovation: tf.Tensor
    ) -> tf.Tensor:
        next_state = structural_transition_value(values, previous, innovation)
        _rho, sigma, _phi, gamma, _R = tf.unstack(physical, axis=1)
        dm = tf.broadcast_to(sigma[:, None], tf.shape(next_state[:, :, 0]))
        dk = 2.0 * gamma[:, None] * next_state[:, :, 0] * sigma[:, None]
        return tf.stack([dm, dk], axis=2)[:, :, :, None]

    def d_transition(previous: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        next_state = structural_transition_value(values, previous, innovation)
        _rho, _sigma, _phi, gamma, _R = tf.unstack(physical, axis=1)
        point_count = tf.shape(previous)[1]
        direct_m = tf.stack(
            [
                previous[:, :, 0] * dphysical[:, 0, None],
                innovation[:, :, 0] * dphysical[:, 1, None],
                tf.zeros([batch_size, point_count], tf.float64),
                tf.zeros([batch_size, point_count], tf.float64),
                tf.zeros([batch_size, point_count], tf.float64),
            ],
            axis=1,
        )
        direct_k = tf.stack(
            [
                2.0 * gamma[:, None] * next_state[:, :, 0] * direct_m[:, 0, :],
                2.0 * gamma[:, None] * next_state[:, :, 0] * direct_m[:, 1, :],
                previous[:, :, 1] * dphysical[:, 2, None],
                tf.square(next_state[:, :, 0]) * dphysical[:, 3, None],
                tf.zeros([batch_size, point_count], tf.float64),
            ],
            axis=1,
        )
        return tf.stack([direct_m, direct_k], axis=3)

    def observation_state_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(
            _OBSERVATION_ROW[None, None, None, :],
            [batch_size, tf.shape(states)[1], 1, 2],
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([batch_size, 5, tf.shape(states)[1], 1], tf.float64)

    innovation_covariance = tf.ones([batch_size, 1, 1], tf.float64)
    observation_covariance = physical[:, 4, None, None]
    d_innovation_covariance = tf.zeros([batch_size, 5, 1, 1], tf.float64)
    d_observation_covariance = tf.zeros([batch_size, 5, 1, 1], tf.float64)
    d_observation_covariance = tf.tensor_scatter_nd_update(
        d_observation_covariance,
        tf.stack([tf.range(batch_size), tf.fill([batch_size], 4)], axis=1),
        dphysical[:, 4, None, None],
    )
    model = TFBatchedStructuralStateSpace(
        initial_mean=mean,
        initial_covariance=covariance,
        innovation_covariance=innovation_covariance,
        observation_covariance=observation_covariance,
        transition_fn=transition,
        observation_fn=observe,
        deterministic_residual_fn=residual,
        name="chapter18b_quadratic_structural_principal_sqrt_ukf",
    )
    derivatives = TFBatchedStructuralFirstDerivatives(
        d_initial_mean=d_mean,
        d_initial_covariance=d_covariance,
        d_innovation_covariance=d_innovation_covariance,
        d_observation_covariance=d_observation_covariance,
        transition_state_jacobian_fn=transition_state_jacobian,
        transition_innovation_jacobian_fn=transition_innovation_jacobian,
        d_transition_fn=d_transition,
        observation_state_jacobian_fn=observation_state_jacobian,
        d_observation_fn=d_observation,
        name="chapter18b_manual_structural_source_derivatives",
    )
    return model, derivatives, initial_value, initial_score


def structural_ukf_likelihood_value_score_status(
    theta: Any,
    *,
    observations: tf.Tensor,
    principal_sqrt_backend: str = "tensorflow_eigh",
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    model, derivatives, initial_value, initial_score = (
        build_structural_ukf_model_and_derivatives(values, y)
    )
    remaining_value, remaining_score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            values,
            y[1:],
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            placement_floor=tf.constant(0.0, tf.float64),
            innovation_floor=tf.constant(1.0e-12, tf.float64),
            spectral_gap_tolerance=tf.constant(1.0e-8, tf.float64),
            fixed_null_tolerance=tf.constant(1.0e-10, tf.float64),
            principal_sqrt_backend=principal_sqrt_backend,
            jitter=tf.constant(0.0, tf.float64),
        )
    )
    value = initial_value + remaining_value
    score = initial_score + remaining_score
    valid = tf.logical_and(
        tf.equal(diagnostics["principal_sqrt_target_valid_count"], 1),
        tf.logical_and(
            tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score), axis=1)
        ),
    )
    return value, score, {
        "status_code": tf.where(valid, tf.zeros_like(value, tf.int32), tf.ones_like(value, tf.int32)),
        "valid_pre_regularized_score": valid,
        "deterministic_residual": diagnostics["deterministic_residual"],
        "min_placement_eigenvalue": diagnostics["min_placement_eigenvalue"],
        "min_innovation_eigenvalue": diagnostics["min_innovation_eigenvalue"],
        "min_placement_eigen_gap": diagnostics["min_placement_eigen_gap"],
        "min_innovation_eigen_gap": diagnostics["min_innovation_eigen_gap"],
        "floor_count_value": (
            diagnostics["placement_floor_count"]
            + diagnostics["innovation_floor_count"]
        ),
        "innovation_condition_estimate": (
            diagnostics["max_innovation_covariance_abs_entry"]
            / tf.maximum(
                diagnostics["min_innovation_eigenvalue"],
                tf.constant(1.0e-300, tf.float64),
            )
        ),
        "principal_sqrt_target_row_class_code": diagnostics[
            "principal_sqrt_target_row_class_code"
        ],
    }


def structural_ukf_likelihood_value_only_status(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, Mapping[str, tf.Tensor]]:
    """Evaluate the Chapter 18b UKF scalar without derivative allocation."""

    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        _principal_sqrt_value_factor,
    )

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    batch_size = int(values.shape[0])
    physical, _ = structural_source_chart(values)
    observation_variance = physical[:, 4]
    y0 = tf.reshape(y[0], [])
    innovation_variance = tf.constant(0.13, tf.float64) + observation_variance
    covariance_observation = tf.constant([0.04, 0.09], tf.float64)
    gain0 = covariance_observation[None, :] / innovation_variance[:, None]
    mean = gain0 * y0
    covariance = STRUCTURAL_INITIAL_COVARIANCE[None, :, :] - (
        covariance_observation[None, :, None]
        * covariance_observation[None, None, :]
        / innovation_variance[:, None, None]
    )
    total_value = -0.5 * (
        _LOG_TWO_PI + tf.math.log(innovation_variance)
        + tf.square(y0) / innovation_variance
    )
    rule = tf_unit_sigma_point_rule(3, rule="unscented")
    invalid_count = tf.zeros([batch_size], tf.int32)
    roundoff_count = tf.zeros([batch_size], tf.int32)
    maximum_residual = tf.zeros([batch_size], tf.float64)
    min_innovation = tf.fill([batch_size], tf.constant(float("inf"), tf.float64))

    def body(index, current_mean, current_covariance, value_total,
             current_invalid, current_roundoff, current_residual,
             current_min_innovation):
        upper = tf.concat(
            (current_covariance, tf.zeros([batch_size, 2, 1], tf.float64)), axis=2
        )
        lower = tf.concat(
            (tf.zeros([batch_size, 1, 2], tf.float64),
             tf.ones([batch_size, 1, 1], tf.float64)), axis=2
        )
        augmented_covariance = tf.concat((upper, lower), axis=1)
        placement, _implemented, placement_invalid, placement_repair, _ = (
            _principal_sqrt_value_factor(
                augmented_covariance, singular_floor=tf.constant(0.0, tf.float64)
            )
        )
        augmented_mean = tf.concat(
            (current_mean, tf.zeros([batch_size, 1], tf.float64)), axis=1
        )
        points = augmented_mean[:, None, :] + tf.einsum(
            "ra,bda->brd", rule.offsets, placement
        )
        previous_points = points[:, :, :2]
        predicted_points = structural_transition_value(
            values, previous_points, points[:, :, 2:]
        )
        residual = tf.reduce_max(
            tf.abs(structural_transition_residual(
                values, previous_points, predicted_points
            )), axis=(1, 2)
        )
        predicted_mean = tf.einsum(
            "r,bri->bi", rule.mean_weights, predicted_points
        )
        centered_state = predicted_points - predicted_mean[:, None, :]
        predicted_covariance = tf.einsum(
            "r,bri,brj->bij", rule.covariance_weights,
            centered_state, centered_state
        )
        predicted_covariance = 0.5 * (
            predicted_covariance + tf.linalg.matrix_transpose(predicted_covariance)
        )
        observation_points = tf.reduce_sum(predicted_points, axis=2, keepdims=True)
        observation_mean = tf.einsum(
            "r,bri->bi", rule.mean_weights, observation_points
        )
        centered_observation = observation_points - observation_mean[:, None, :]
        raw_innovation_covariance = (
            tf.einsum(
                "r,bri,brj->bij", rule.covariance_weights,
                centered_observation, centered_observation
            ) + observation_variance[:, None, None]
        )
        (_sqrt, implemented_innovation, innovation_invalid,
         innovation_repair, innovation_minimum) = _principal_sqrt_value_factor(
            raw_innovation_covariance,
            singular_floor=tf.constant(1.0e-12, tf.float64),
        )
        cross_covariance = tf.einsum(
            "r,bri,brj->bij", rule.covariance_weights,
            centered_state, centered_observation
        )
        innovation_factor = tf.linalg.cholesky(implemented_innovation)
        innovation = y[index][None, :] - observation_mean
        solve = tf.linalg.cholesky_solve(
            innovation_factor, innovation[:, :, None]
        )[:, :, 0]
        precision = tf.linalg.cholesky_solve(
            innovation_factor, tf.ones([batch_size, 1, 1], tf.float64)
        )
        increment = -0.5 * (
            _LOG_TWO_PI
            + 2.0 * tf.math.log(innovation_factor[:, 0, 0])
            + tf.reduce_sum(innovation * solve, axis=1)
        )
        gain = cross_covariance @ precision
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        filtered_covariance = (
            predicted_covariance
            - gain @ implemented_innovation @ tf.linalg.matrix_transpose(gain)
        )
        filtered_covariance = 0.5 * (
            filtered_covariance + tf.linalg.matrix_transpose(filtered_covariance)
        )
        step_invalid = tf.cast(
            tf.logical_or(placement_invalid, innovation_invalid), tf.int32
        )
        step_repair = tf.cast(
            tf.logical_or(placement_repair, innovation_repair), tf.int32
        )
        return (
            index + 1, filtered_mean, filtered_covariance,
            value_total + increment,
            tf.maximum(current_invalid, step_invalid),
            tf.maximum(current_roundoff, step_repair),
            tf.maximum(current_residual, residual),
            tf.minimum(current_min_innovation, innovation_minimum),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.shape(y)[0],
        body,
        (
            tf.constant(1, tf.int32), mean, covariance, total_value,
            invalid_count, roundoff_count, maximum_residual, min_innovation,
        ),
        parallel_iterations=1,
    )
    valid = tf.logical_and(
        result[4] == 0,
        tf.logical_and(
            tf.math.is_finite(result[3]),
            result[6] <= tf.constant(2.0e-14, tf.float64),
        ),
    )
    checked = tf.where(
        valid, result[3],
        tf.fill(tf.shape(result[3]), tf.constant(-1.0e100, tf.float64))
    )
    return checked, {
        "status_code": tf.where(
            valid, tf.zeros_like(result[4]), tf.ones_like(result[4])
        ),
        "valid_value": valid,
        "deterministic_residual": result[6],
        "artificial_k_noise_allowed": tf.zeros([batch_size], tf.bool),
        "roundoff_repair_count": result[5],
        "min_innovation_eigenvalue": result[7],
    }


def structural_ukf_posterior_value_only(
    theta: Any, *, observations: tf.Tensor
) -> tf.Tensor:
    likelihood, _status = structural_ukf_likelihood_value_only_status(
        theta, observations=observations
    )
    prior, _ = structural_source_uniform_prior_value_score(theta)
    jacobian, _ = structural_source_probit_jacobian_value_score(theta)
    return likelihood + prior + jacobian


def structural_ukf_likelihood_value_score(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score, _status = structural_ukf_likelihood_value_score_status(
        theta, observations=observations
    )
    return value, score


def _structural_posterior_value_score_status(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
    likelihood_value, likelihood_score, status = (
        structural_ukf_likelihood_value_score_status(
            theta, observations=observations
        )
    )
    prior_value, prior_score = structural_source_uniform_prior_value_score(theta)
    jacobian_value, jacobian_score = (
        structural_source_probit_jacobian_value_score(theta)
    )
    return (
        likelihood_value + prior_value + jacobian_value,
        likelihood_score + prior_score + jacobian_score,
        status,
    )


class StructuralUKFNeuTraAdapter:
    """Batch-native adapter for the frozen Chapter 18b structural posterior."""

    dtype = tf.float64
    parameter_dim = 5
    parameter_names = STRUCTURAL_PARAMETER_NAMES

    def __init__(self, *, observations: tf.Tensor, contract: SSMTargetContract) -> None:
        self.observations = tf.convert_to_tensor(observations, tf.float64)
        self.contract = contract
        self.target_scope = STRUCTURAL_UKF_SCOPE
        payload = {
            "schema": "bayesfilter.testing.structural_ukf_neutra_adapter.v1",
            "target_signature": stable_ssm_target_signature(contract),
            "dtype": self.dtype.name,
            "parameter_names": self.parameter_names,
            "time_order": "x0_then_y0_then_structural_transitions_y1_to_y99",
        }
        self._adapter_signature = _semantic_hash(payload)

    def adapter_signature(self) -> str:
        return self._adapter_signature

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="tensorflow_batched_principal_sqrt_ukf_structural",
            evidence_path=(
                "bayesfilter/testing/structural_ukf_neutra_target_design_tf.py"
            ),
            target_scope=self.target_scope,
            nonclaims=STRUCTURAL_UKF_NONCLAIMS,
        )

    def log_prob(self, theta: Any) -> tf.Tensor:
        value, _score, _status = _structural_posterior_value_score_status(
            theta, observations=self.observations
        )
        return value

    def log_prob_and_grad(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _status = _structural_posterior_value_score_status(
            theta, observations=self.observations
        )
        return value, score

    def neutra_batch_log_prob_and_grad_status(
        self, theta: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return _structural_posterior_value_score_status(
            theta, observations=self.observations
        )

    def target_status_telemetry(self, theta: Any) -> Mapping[str, tf.Tensor]:
        _value, _score, status = _structural_posterior_value_score_status(
            theta, observations=self.observations
        )
        return {
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
        }


class StructuralUKFLikelihoodRecomposer:
    """Independent structural UKF likelihood component for recomposition."""

    def __init__(self, adapter: StructuralUKFNeuTraAdapter) -> None:
        self.observations = adapter.observations

    def __call__(self, theta: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return structural_ukf_likelihood_value_score(
            theta, observations=self.observations
        )


def make_structural_ukf_neutra_adapter(
    *, observations: tf.Tensor | None = None
) -> StructuralUKFNeuTraAdapter:
    if observations is None:
        states, observations = generate_frozen_structural_dataset_tf()
        if _tensor_hash(states) != STRUCTURAL_FINAL_STATE_SHA256:
            raise ValueError("frozen structural state hash mismatch")
    y = tf.convert_to_tensor(observations, tf.float64)
    if y.shape != (STRUCTURAL_FINAL_HORIZON, 1):
        raise ValueError("structural observations require frozen shape [100, 1]")
    observation_hash = _tensor_hash(y)
    if observation_hash != STRUCTURAL_FINAL_OBSERVATION_SHA256:
        raise ValueError("frozen structural observation hash mismatch")
    contract = make_structural_ukf_target_contract(
        horizon=STRUCTURAL_FINAL_HORIZON,
        data_hash=observation_hash,
    )
    return StructuralUKFNeuTraAdapter(observations=y, contract=contract)


def make_structural_ukf_target_contract(
    *, horizon: int, data_hash: str
) -> SSMTargetContract:
    shape = SSMStaticShape(
        horizon=int(horizon),
        state_dim=2,
        observation_dim=1,
        innovation_dim=1,
        parameter_dim=5,
    )
    model_semantics = {
        "model_id": "chapter18b-quadratic-structural-T100-v1",
        "equations": (
            "m_t=rho*m_(t-1)+sigma*epsilon_t",
            "k_t=phi*k_(t-1)+gamma*m_t^2",
            "y_t=m_t+k_t+e_t",
        ),
        "initial_mean": (0.0, 0.0),
        "initial_covariance": ((0.04, 0.0), (0.0, 0.09)),
        "innovation_law": "epsilon_t~N(0,1); no independent k innovation",
        "observation_variance_parameter": "R",
        "time_order": "x0_from_initial_then_y0; structural transitions y1_to_y99",
        "physical_truth": tuple(float(item) for item in STRUCTURAL_TRUTH_PHYSICAL),
        "truth_role": "synthetic_data_generation_and_design_center_only",
    }
    problem = BayesianSSMProblem(
        problem_id="chapter18b-structural-principal-sqrt-ukf-five-probit",
        static_shape=shape,
        data_signature=SSMDataSignature(
            dataset_id="chapter18b_structural_T100_seed_20260716_15001",
            observation_shape=(int(horizon), 1),
            data_hash=f"sha256:{data_hash}",
        ),
        target_coordinate_convention="unconstrained",
        model_manifest={
            **model_semantics,
            "model_hash": f"sha256:{_semantic_hash(model_semantics)}",
        },
    )
    chart_semantics = {
        "transform_id": "structural-five-probit-uniform-box-chart-v1",
        "parameter_order": STRUCTURAL_PARAMETER_NAMES,
        "lower": tuple(float(item) for item in STRUCTURAL_PARAMETER_LOWER),
        "upper": tuple(float(item) for item in STRUCTURAL_PARAMETER_UPPER),
    }
    chart = ParameterChart(
        parameter_names=STRUCTURAL_PARAMETER_NAMES,
        unconstrained_dim=5,
        constrained_shape=(5,),
        transform_manifest={
            **chart_semantics,
            "transform_hash": f"sha256:{_semantic_hash(chart_semantics)}",
        },
        log_jacobian_convention="included_in_chart",
    )
    prior_semantics = {
        "prior_id": "structural-independent-uniform-parameter-box-v1",
        "physical_support": tuple(
            (float(lower), float(upper))
            for lower, upper in zip(
                STRUCTURAL_PARAMETER_LOWER, STRUCTURAL_PARAMETER_UPPER, strict=True
            )
        ),
        "parameter_order": ("rho", "sigma", "phi", "gamma", "R"),
    }
    prior = ParameterPrior(
        prior_manifest={
            **prior_semantics,
            "prior_hash": f"sha256:{_semantic_hash(prior_semantics)}",
        },
        support_policy="enforced_by_transform",
        log_density_authority="graph_native",
    )
    filter_semantics = {
        "filter_id": "chapter18b-structural-principal-sqrt-ukf-v1",
        "engine": "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
        "backend": "tf_principal_sqrt_ukf",
        "principal_sqrt_backend": "tensorflow_eigh_xla_portable",
        "score": "manual_forward_structural_source_derivatives",
        "integration_space": "lagged_state_plus_scalar_innovation",
        "deterministic_completion": "k_t=phi*k_(t-1)+gamma*m_t^2",
        "artificial_k_noise_allowed": False,
        "jitter": 0.0,
        "innovation_floor": 1.0e-12,
        "floor_role": "classified_invalid_guard_not_model_noise",
    }
    filter_program = FilterProgram(
        filter_id=str(filter_semantics["filter_id"]),
        required_model_capabilities=(
            "quadratic_structural_transition",
            "scalar_innovation_integration",
            "deterministic_completion_residual",
            "principal_square_root_ukf",
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


def _tensor_hash(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _semantic_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def structural_ukf_innovation_history_tf(
    theta: Any, *, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return value-only predictive observation mean and variance histories."""

    values = _rank2_theta(theta)
    y = tf.convert_to_tensor(observations, tf.float64)
    horizon = int(y.shape[0])
    batch_size = int(values.shape[0])
    physical, _derivative = structural_source_chart(values)
    _rho, _sigma, _phi, _gamma, observation_variance = tf.unstack(physical, axis=1)
    mean, covariance, _d_mean, _d_covariance, _value, _score = (
        _initial_observation_update(values, y)
    )
    rule = tf_unit_sigma_point_rule(3, rule="unscented")
    mu_history = tf.zeros([horizon, batch_size], tf.float64)
    variance_history = tf.tensor_scatter_nd_update(
        tf.zeros([horizon, batch_size], tf.float64),
        [[0]],
        [(tf.constant(0.13, tf.float64) + observation_variance)],
    )

    def body(
        index: tf.Tensor,
        current_mean: tf.Tensor,
        current_covariance: tf.Tensor,
        means: tf.Tensor,
        variances: tf.Tensor,
    ):
        upper = tf.concat(
            [current_covariance, tf.zeros([batch_size, 2, 1], tf.float64)], axis=2
        )
        lower = tf.concat(
            [tf.zeros([batch_size, 1, 2], tf.float64), tf.ones([batch_size, 1, 1], tf.float64)],
            axis=2,
        )
        augmented_covariance = tf.concat([upper, lower], axis=1)
        eigenvalues, eigenvectors = tf.linalg.eigh(
            0.5 * (augmented_covariance + tf.linalg.matrix_transpose(augmented_covariance))
        )
        factor = tf.einsum(
            "bij,bj,bkj->bik", eigenvectors, tf.sqrt(eigenvalues), eigenvectors
        )
        augmented_mean = tf.concat([current_mean, tf.zeros([batch_size, 1], tf.float64)], axis=1)
        points = augmented_mean[:, None, :] + tf.einsum("ra,bda->brd", rule.offsets, factor)
        predicted = structural_transition_value(values, points[:, :, :2], points[:, :, 2:])
        predicted_mean = tf.einsum("r,bri->bi", rule.mean_weights, predicted)
        centered_state = predicted - predicted_mean[:, None, :]
        predicted_covariance = tf.einsum(
            "r,bri,brj->bij", rule.covariance_weights, centered_state, centered_state
        )
        observation_points = tf.reduce_sum(predicted, axis=2)
        observation_mean = tf.einsum("r,br->b", rule.mean_weights, observation_points)
        centered_observation = observation_points - observation_mean[:, None]
        innovation_variance = (
            tf.einsum("r,br,br->b", rule.covariance_weights, centered_observation, centered_observation)
            + observation_variance
        )
        cross_covariance = tf.einsum(
            "r,bri,br->bi", rule.covariance_weights, centered_state, centered_observation
        )
        gain = cross_covariance / innovation_variance[:, None]
        next_mean = predicted_mean + gain * (y[index, 0] - observation_mean)[:, None]
        next_covariance = predicted_covariance - (
            cross_covariance[:, :, None] * cross_covariance[:, None, :]
            / innovation_variance[:, None, None]
        )
        next_covariance = 0.5 * (
            next_covariance + tf.linalg.matrix_transpose(next_covariance)
        )
        return (
            index + 1,
            next_mean,
            next_covariance,
            tf.tensor_scatter_nd_update(means, index[None, None], observation_mean[None, :]),
            tf.tensor_scatter_nd_update(
                variances, index[None, None], innovation_variance[None, :]
            ),
        )

    result = tf.while_loop(
        lambda index, *_unused: index < tf.constant(horizon, tf.int32),
        body,
        (tf.constant(1, tf.int32), mean, covariance, mu_history, variance_history),
        parallel_iterations=1,
    )
    return tf.transpose(result[3], [1, 0]), tf.transpose(result[4], [1, 0])


def structural_likelihood_information_tf(
    theta: Any,
    *,
    observations: tf.Tensor,
    finite_difference_step: tf.Tensor | float = 5.0e-5,
) -> Mapping[str, tf.Tensor]:
    """Return a likelihood-only information surrogate from batched central FD."""

    values = _rank2_theta(theta)
    batch_size = int(values.shape[0])
    step = tf.reshape(tf.convert_to_tensor(finite_difference_step, tf.float64), [])
    directions = step * tf.eye(5, dtype=tf.float64)
    perturbed = tf.concat(
        [
            values[:, None, :] + directions[None, :, :],
            values[:, None, :] - directions[None, :, :],
        ],
        axis=1,
    )
    flat = tf.reshape(perturbed, [batch_size * 10, 5])
    flat_means, flat_variances = structural_ukf_innovation_history_tf(
        flat, observations=observations
    )
    means_by_direction = tf.reshape(
        flat_means, [batch_size, 10, tf.shape(flat_means)[1]]
    )
    log_variances_by_direction = tf.reshape(
        tf.math.log(flat_variances),
        [batch_size, 10, tf.shape(flat_variances)[1]],
    )
    d_means = tf.transpose(
        (means_by_direction[:, :5] - means_by_direction[:, 5:]) / (2.0 * step),
        [0, 2, 1],
    )
    d_log_variances = tf.transpose(
        (
            log_variances_by_direction[:, :5]
            - log_variances_by_direction[:, 5:]
        )
        / (2.0 * step),
        [0, 2, 1],
    )
    means, variances = structural_ukf_innovation_history_tf(
        values, observations=observations
    )
    contributions = (
        tf.einsum("btp,btq,bt->btpq", d_means, d_means, 1.0 / variances)
        + 0.5 * tf.einsum(
            "btp,btq->btpq", d_log_variances, d_log_variances
        )
    )
    cumulative = tf.cumsum(contributions, axis=1)
    return {
        "predictive_mean": means,
        "innovation_variance": variances,
        "d_predictive_mean": d_means,
        "d_log_innovation_variance": d_log_variances,
        "information_contributions": contributions,
        "cumulative_information": cumulative,
        "finite_difference_step": step,
    }


def structural_negative_control_one_step_tf() -> Mapping[str, tf.Tensor]:
    """Return the chapter one-step structural and explicit-noise comparison."""

    theta = structural_truth_source()[None, :]
    rule = tf_unit_sigma_point_rule(3, rule="unscented")
    augmented_mean = tf.zeros([1, 3], tf.float64)
    augmented_covariance = tf.linalg.diag(tf.constant([[0.04, 0.09, 1.0]], tf.float64))
    factor = tf.linalg.diag(tf.sqrt(tf.linalg.diag_part(augmented_covariance)))
    points = augmented_mean[:, None, :] + tf.einsum("ra,bda->brd", rule.offsets, factor)
    predicted = structural_transition_value(theta, points[:, :, :2], points[:, :, 2:])
    observation_points = tf.reduce_sum(predicted, axis=2)
    observation_mean = tf.einsum("r,br->b", rule.mean_weights, observation_points)
    centered = observation_points - observation_mean[:, None]
    structural_variance = (
        tf.einsum("r,br,br->b", rule.covariance_weights, centered, centered)
        + STRUCTURAL_TRUTH_PHYSICAL[4]
    )
    negative_variance = structural_variance + STRUCTURAL_NEGATIVE_CONTROL_K_VARIANCE
    innovation = tf.constant(0.3, tf.float64) - observation_mean
    structural_log_likelihood = -0.5 * (
        _LOG_TWO_PI + tf.math.log(structural_variance) + tf.square(innovation) / structural_variance
    )
    negative_log_likelihood = -0.5 * (
        _LOG_TWO_PI + tf.math.log(negative_variance) + tf.square(innovation) / negative_variance
    )
    negative_innovation_offsets = tf.constant(
        [-math.sqrt(0.08), 0.0, math.sqrt(0.08)], tf.float64
    )
    return {
        "structural_observation_mean": observation_mean,
        "structural_innovation_variance": structural_variance,
        "negative_control_innovation_variance": negative_variance,
        "structural_log_likelihood": structural_log_likelihood,
        "negative_control_log_likelihood": negative_log_likelihood,
        "negative_control_k_variance_increment": STRUCTURAL_NEGATIVE_CONTROL_K_VARIANCE,
        "negative_control_pointwise_residuals": negative_innovation_offsets,
        "intended_pointwise_residuals": structural_transition_residual(
            theta, points[:, :, :2], predicted
        ),
    }
