"""TensorFlow QR/square-root linear Gaussian Kalman backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor,
    factor_solve,
    lower_factor_from_horizontal_stack,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural import FilterRunMetadata


TFQRLinearValueBackend = Literal["tf_qr", "tf_masked_qr"]


def _to_tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _matvec(matrix: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matvec(matrix, vector)


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _matrix_at_time(matrix: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    if matrix.shape.rank == 3:
        return matrix[time_index]
    return matrix


def _vector_at_time(vector: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    if vector.shape.rank == 2:
        return vector[time_index]
    return vector


def _validate_mask_shape(observations: tf.Tensor, observation_mask: tf.Tensor) -> None:
    tf.debugging.assert_equal(
        tf.shape(observation_mask),
        tf.shape(observations),
        message="Observation mask shape must match observations shape.",
    )


def _static_num_timesteps(observations: tf.Tensor) -> int:
    n_timesteps = observations.shape[0]
    if n_timesteps is None:
        raise ValueError("QR square-root filters require a static observation length")
    return int(n_timesteps)


@tf.function
def tf_qr_sqrt_kalman_log_likelihood(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tf.Tensor:
    """QR/square-root prediction-error log likelihood."""

    value, _, _ = tf_qr_sqrt_kalman_filter(
        observations=observations,
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        jitter=jitter,
    )
    return value


@tf.function
def tf_qr_sqrt_masked_kalman_log_likelihood(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tf.Tensor:
    """QR/square-root prediction-error log likelihood with static masks."""

    value, _, _ = tf_qr_sqrt_masked_kalman_filter(
        observations=observations,
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        observation_mask=observation_mask,
        jitter=jitter,
    )
    return value


@tf.function
def tf_qr_sqrt_kalman_filter(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Dense direct-QR square-root Kalman recursion."""

    y = _as_observation_matrix(observations)
    n_timesteps = _static_num_timesteps(y)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _to_tensor(transition_covariance)
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _to_tensor(observation_covariance)
    mean = _to_tensor(initial_state_mean)
    initial_state_covariance = _to_tensor(initial_state_covariance)
    jitter_tensor = tf.cast(jitter, tf.float64)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    means = []
    covariances = []

    for t in range(n_timesteps):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        transition_covariance_factor = cholesky_factor(Q, 0.0)
        observation_covariance_factor = cholesky_factor(H + jitter_tensor * obs_identity, 0.0)

        predicted_mean = c + _matvec(T, mean)
        prediction_stack = tf.concat(
            (T @ covariance_factor, transition_covariance_factor),
            axis=1,
        )
        predicted_factor = lower_factor_from_horizontal_stack(prediction_stack)
        predicted_covariance = predicted_factor @ tf.transpose(predicted_factor)

        innovation = y[t] - (d + _matvec(Z, predicted_mean))
        innovation_stack = tf.concat(
            (Z @ predicted_factor, observation_covariance_factor),
            axis=1,
        )
        innovation_factor = lower_factor_from_horizontal_stack(innovation_stack)
        innovation_precision = factor_solve(innovation_factor, obs_identity)
        kalman_gain = predicted_covariance @ tf.transpose(Z) @ innovation_precision

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ Z
        update_stack = tf.concat(
            (
                joseph_left @ predicted_factor,
                kalman_gain @ observation_covariance_factor,
            ),
            axis=1,
        )
        filtered_factor = lower_factor_from_horizontal_stack(update_stack)
        filtered_covariance = filtered_factor @ tf.transpose(filtered_factor)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        contribution = -0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi) + log_det + mahalanobis
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution
        means.append(filtered_mean)
        covariances.append(filtered_covariance)

    return log_likelihood, tf.stack(means, axis=0), tf.stack(covariances, axis=0)



@tf.function
def tf_qr_sqrt_masked_kalman_filter(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Direct-QR square-root Kalman recursion with static observation masks."""

    y = _as_observation_matrix(observations)
    n_timesteps = _static_num_timesteps(y)
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    _validate_mask_shape(y, observation_mask)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _to_tensor(transition_covariance)
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _to_tensor(observation_covariance)
    mean = _to_tensor(initial_state_mean)
    initial_state_covariance = _to_tensor(initial_state_covariance)
    jitter_tensor = tf.cast(jitter, tf.float64)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=tf.float64)
    means = []
    covariances = []

    for t in range(n_timesteps):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        transition_covariance_factor = cholesky_factor(Q, 0.0)
        base_observation_covariance = H + jitter_tensor * obs_identity

        predicted_mean = c + _matvec(T, mean)
        prediction_stack = tf.concat(
            (T @ covariance_factor, transition_covariance_factor),
            axis=1,
        )
        predicted_factor = lower_factor_from_horizontal_stack(prediction_stack)
        predicted_covariance = predicted_factor @ tf.transpose(predicted_factor)

        row_weight = tf.cast(observation_mask[t], tf.float64)
        missing_weight = 1.0 - row_weight
        row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
        masked_observation_matrix = Z * row_weight[:, tf.newaxis]
        masked_observation_covariance = (
            base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)
        )
        observation_covariance_factor = cholesky_factor(masked_observation_covariance, 0.0)
        innovation = (y[t] - (d + _matvec(Z, predicted_mean))) * row_weight
        innovation_stack = tf.concat(
            (masked_observation_matrix @ predicted_factor, observation_covariance_factor),
            axis=1,
        )
        innovation_factor = lower_factor_from_horizontal_stack(innovation_stack)
        innovation_precision = factor_solve(innovation_factor, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ tf.transpose(masked_observation_matrix)
            @ innovation_precision
        )

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ masked_observation_matrix
        update_stack = tf.concat(
            (
                joseph_left @ predicted_factor,
                kalman_gain @ observation_covariance_factor,
            ),
            axis=1,
        )
        filtered_factor = lower_factor_from_horizontal_stack(update_stack)
        filtered_covariance = filtered_factor @ tf.transpose(filtered_factor)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        missing_count = tf.reduce_sum(missing_weight)
        contribution = -0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
            - missing_count * dummy_log_norm
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution
        means.append(filtered_mean)
        covariances.append(filtered_covariance)

    return log_likelihood, tf.stack(means, axis=0), tf.stack(covariances, axis=0)



def _metadata(
    *,
    filter_name: str,
    model: TFLinearGaussianStateSpace,
) -> FilterRunMetadata:
    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="value_only",
        compiled_status="tf_function",
    )


def _diagnostics(
    *,
    backend: str,
    mask_convention: str,
    jitter: tf.Tensor | float,
) -> TFFilterDiagnostics:
    return TFFilterDiagnostics(
        backend=backend,
        mask_convention=mask_convention,
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=tf.float64),
            singular_floor=tf.constant(0.0, dtype=tf.float64),
            floor_count=tf.constant(0, dtype=tf.int32),
            psd_projection_residual=tf.constant(0.0, dtype=tf.float64),
            implemented_covariance=None,
            branch_label="qr_square_root",
            derivative_target="implemented_regularized_law",
        ),
    )


def tf_qr_linear_gaussian_log_likelihood(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    *,
    backend: TFQRLinearValueBackend = "tf_qr",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> TFFilterValueResult:
    """Dispatch to a QR/square-root TensorFlow linear Gaussian value backend."""

    y = _as_observation_matrix(observations)
    mask = observation_mask if observation_mask is not None else model.observation_mask
    if backend == "tf_qr":
        if mask is None:
            value, filtered_means, filtered_covariances = tf_qr_sqrt_kalman_filter(
                observations=y,
                transition_offset=model.transition_offset,
                transition_matrix=model.transition_matrix,
                transition_covariance=model.transition_covariance,
                observation_offset=model.observation_offset,
                observation_matrix=model.observation_matrix,
                observation_covariance=model.observation_covariance,
                initial_state_mean=model.initial_mean,
                initial_state_covariance=model.initial_covariance,
                jitter=jitter,
            )
            filter_name = "tf_qr_sqrt_kalman"
            mask_convention = "none"
        else:
            value, filtered_means, filtered_covariances = tf_qr_sqrt_masked_kalman_filter(
                observations=y,
                transition_offset=model.transition_offset,
                transition_matrix=model.transition_matrix,
                transition_covariance=model.transition_covariance,
                observation_offset=model.observation_offset,
                observation_matrix=model.observation_matrix,
                observation_covariance=model.observation_covariance,
                initial_state_mean=model.initial_mean,
                initial_state_covariance=model.initial_covariance,
                observation_mask=mask,
                jitter=jitter,
            )
            filter_name = "tf_qr_sqrt_masked_kalman"
            mask_convention = "static_dummy_row"
    elif backend == "tf_masked_qr":
        if mask is None:
            raise ValueError("tf_masked_qr requires an observation mask")
        value, filtered_means, filtered_covariances = tf_qr_sqrt_masked_kalman_filter(
            observations=y,
            transition_offset=model.transition_offset,
            transition_matrix=model.transition_matrix,
            transition_covariance=model.transition_covariance,
            observation_offset=model.observation_offset,
            observation_matrix=model.observation_matrix,
            observation_covariance=model.observation_covariance,
            initial_state_mean=model.initial_mean,
            initial_state_covariance=model.initial_covariance,
            observation_mask=mask,
            jitter=jitter,
        )
        filter_name = "tf_qr_sqrt_masked_kalman"
        mask_convention = "static_dummy_row"
    else:
        raise ValueError(f"unknown TensorFlow QR linear Gaussian backend: {backend}")

    if not return_filtered:
        filtered_means = None
        filtered_covariances = None

    return TFFilterValueResult(
        log_likelihood=value,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        metadata=_metadata(filter_name=filter_name, model=model),
        diagnostics=_diagnostics(
            backend=backend,
            mask_convention=mask_convention,
            jitter=jitter,
        ),
    )
