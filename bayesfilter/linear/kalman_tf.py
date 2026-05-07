"""TensorFlow linear Gaussian Kalman value backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural import FilterRunMetadata


TFLinearValueBackend = Literal["tf_cholesky", "tf_masked_cholesky"]


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _scalar_float64(value: tf.Tensor | float) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _matrix_at_time(matrix: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    if matrix.shape.rank == 3:
        return matrix[time_index]
    return matrix


def _vector_at_time(vector: tf.Tensor, time_index: tf.Tensor) -> tf.Tensor:
    if vector.shape.rank == 2:
        return vector[time_index]
    return vector


def _innovation_log_prob(
    innovation: tf.Tensor,
    innovation_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    innovation_covariance = _symmetrize(innovation_covariance)
    chol = tf.linalg.cholesky(innovation_covariance)
    distribution = tfp.distributions.MultivariateNormalTriL(
        loc=tf.zeros(tf.shape(innovation)[0], dtype=tf.float64),
        scale_tril=chol,
    )
    log_prob = distribution.log_prob(innovation)
    solve_innovation = tf.linalg.cholesky_solve(
        chol,
        innovation[:, tf.newaxis],
    )[:, 0]
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return log_prob, chol, solve_innovation, log_det


def _dense_step(
    *,
    time_index: tf.Tensor,
    row: tf.Tensor,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    jitter: tf.Tensor,
    state_identity: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    c = _vector_at_time(transition_offset, time_index)
    T = _matrix_at_time(transition_matrix, time_index)
    Q = _matrix_at_time(transition_covariance, time_index)
    d = _vector_at_time(observation_offset, time_index)
    Z = _matrix_at_time(observation_matrix, time_index)
    H = _matrix_at_time(observation_covariance, time_index)
    observation_dim = tf.shape(Z)[0]
    observation_noise = H + jitter * tf.eye(observation_dim, dtype=tf.float64)

    predicted_mean = c + tf.linalg.matvec(T, mean)
    predicted_covariance = _symmetrize(T @ covariance @ tf.transpose(T) + Q)
    innovation = row - (d + tf.linalg.matvec(Z, predicted_mean))
    innovation_covariance = Z @ predicted_covariance @ tf.transpose(Z) + observation_noise
    log_prob, chol, _, _ = _innovation_log_prob(innovation, innovation_covariance)

    gain_rhs = predicted_covariance @ tf.transpose(Z)
    kalman_gain = tf.transpose(tf.linalg.cholesky_solve(chol, tf.transpose(gain_rhs)))
    filtered_mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
    left = state_identity - kalman_gain @ Z
    filtered_covariance = _symmetrize(
        left @ predicted_covariance @ tf.transpose(left)
        + kalman_gain @ observation_noise @ tf.transpose(kalman_gain)
    )
    return filtered_mean, filtered_covariance, log_prob


def _masked_step(
    *,
    time_index: tf.Tensor,
    row: tf.Tensor,
    row_mask: tf.Tensor,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    jitter: tf.Tensor,
    state_identity: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    c = _vector_at_time(transition_offset, time_index)
    T = _matrix_at_time(transition_matrix, time_index)
    Q = _matrix_at_time(transition_covariance, time_index)
    d = _vector_at_time(observation_offset, time_index)
    Z = _matrix_at_time(observation_matrix, time_index)
    H = _matrix_at_time(observation_covariance, time_index)
    observation_dim = tf.shape(Z)[0]
    obs_identity = tf.eye(observation_dim, dtype=tf.float64)
    base_observation_noise = H + jitter * obs_identity

    predicted_mean = c + tf.linalg.matvec(T, mean)
    predicted_covariance = _symmetrize(T @ covariance @ tf.transpose(T) + Q)

    row_weight = tf.cast(row_mask, tf.float64)
    missing_weight = 1.0 - row_weight
    row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
    expected_observation = d + tf.linalg.matvec(Z, predicted_mean)
    innovation = (row - expected_observation) * row_weight
    masked_observation_matrix = Z * row_weight[:, tf.newaxis]
    masked_observation_noise = (
        base_observation_noise * row_outer + tf.linalg.diag(missing_weight)
    )
    innovation_covariance = (
        masked_observation_matrix
        @ predicted_covariance
        @ tf.transpose(masked_observation_matrix)
        + masked_observation_noise
    )
    log_prob, chol, _, _ = _innovation_log_prob(innovation, innovation_covariance)

    gain_rhs = predicted_covariance @ tf.transpose(masked_observation_matrix)
    kalman_gain = tf.transpose(tf.linalg.cholesky_solve(chol, tf.transpose(gain_rhs)))
    filtered_mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
    left = state_identity - kalman_gain @ masked_observation_matrix
    filtered_covariance = _symmetrize(
        left @ predicted_covariance @ tf.transpose(left)
        + kalman_gain @ masked_observation_noise @ tf.transpose(kalman_gain)
    )
    missing_count = tf.reduce_sum(missing_weight)
    dummy_log_norm = tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float64))
    adjusted_log_prob = log_prob + 0.5 * missing_count * dummy_log_norm
    return filtered_mean, filtered_covariance, adjusted_log_prob


def _validate_mask_shape(observations: tf.Tensor, observation_mask: tf.Tensor) -> None:
    tf.debugging.assert_equal(
        tf.shape(observation_mask),
        tf.shape(observations),
        message="Observation mask shape must match observations shape.",
    )


@tf.function(reduce_retracing=True)
def tf_kalman_log_likelihood(
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
    """Prediction-error log likelihood for a dense TF linear Gaussian model."""

    value, _, _ = tf_kalman_filter(
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
        return_filtered=False,
    )
    return value


@tf.function(reduce_retracing=True)
def tf_masked_kalman_log_likelihood(
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
    """Prediction-error log likelihood with static-shape observation masking."""

    value, _, _ = tf_masked_kalman_filter(
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
        return_filtered=False,
    )
    return value


@tf.function(reduce_retracing=True)
def tf_kalman_filter(
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
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Dense TF Kalman recursion with optional filtered-state tensors."""

    y = _as_observation_matrix(observations)
    transition_offset = tf.convert_to_tensor(transition_offset, dtype=tf.float64)
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    transition_covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    observation_offset = tf.convert_to_tensor(observation_offset, dtype=tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    mean = tf.convert_to_tensor(initial_state_mean, dtype=tf.float64)
    covariance = _symmetrize(
        tf.convert_to_tensor(initial_state_covariance, dtype=tf.float64)
    )
    jitter = _scalar_float64(jitter)

    state_identity = tf.eye(tf.shape(mean)[0], dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    means = tf.TensorArray(tf.float64, size=tf.shape(y)[0])
    covariances = tf.TensorArray(tf.float64, size=tf.shape(y)[0])

    for t in tf.range(tf.shape(y)[0]):
        mean, covariance, contribution = _dense_step(
            time_index=t,
            row=y[t],
            mean=mean,
            covariance=covariance,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            jitter=jitter,
            state_identity=state_identity,
        )
        log_likelihood = log_likelihood + contribution
        if return_filtered:
            means = means.write(t, mean)
            covariances = covariances.write(t, covariance)

    filtered_means = means.stack() if return_filtered else None
    filtered_covariances = covariances.stack() if return_filtered else None
    return log_likelihood, filtered_means, filtered_covariances


@tf.function(reduce_retracing=True)
def tf_masked_kalman_filter(
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
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Static-shape masked TF Kalman recursion.

    Missing observation rows are replaced by zero residuals, zero loadings, and
    independent unit-variance dummy observations.  The dummy standard-normal
    log normalizers are removed, so an all-missing period contributes exactly
    zero measurement log likelihood while the state prediction still advances.
    """

    y = _as_observation_matrix(observations)
    transition_offset = tf.convert_to_tensor(transition_offset, dtype=tf.float64)
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    transition_covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    observation_offset = tf.convert_to_tensor(observation_offset, dtype=tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    mean = tf.convert_to_tensor(initial_state_mean, dtype=tf.float64)
    covariance = _symmetrize(
        tf.convert_to_tensor(initial_state_covariance, dtype=tf.float64)
    )
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    jitter = _scalar_float64(jitter)
    _validate_mask_shape(y, observation_mask)

    state_identity = tf.eye(tf.shape(mean)[0], dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    means = tf.TensorArray(tf.float64, size=tf.shape(y)[0])
    covariances = tf.TensorArray(tf.float64, size=tf.shape(y)[0])

    for t in tf.range(tf.shape(y)[0]):
        mean, covariance, contribution = _masked_step(
            time_index=t,
            row=y[t],
            row_mask=observation_mask[t],
            mean=mean,
            covariance=covariance,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            jitter=jitter,
            state_identity=state_identity,
        )
        log_likelihood = log_likelihood + contribution
        if return_filtered:
            means = means.write(t, mean)
            covariances = covariances.write(t, covariance)

    filtered_means = means.stack() if return_filtered else None
    filtered_covariances = covariances.stack() if return_filtered else None
    return log_likelihood, filtered_means, filtered_covariances


def _metadata(
    *,
    filter_name: str,
    model: TFLinearGaussianStateSpace,
    compiled_status: str,
) -> FilterRunMetadata:
    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="value_only",
        compiled_status=compiled_status,
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
            branch_label="cholesky",
            derivative_target="implemented_regularized_law",
        ),
    )


def tf_linear_gaussian_log_likelihood(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    *,
    backend: TFLinearValueBackend = "tf_cholesky",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> TFFilterValueResult:
    """Dispatch to a TensorFlow linear Gaussian value backend."""

    y = _as_observation_matrix(observations)
    mask = observation_mask if observation_mask is not None else model.observation_mask
    if backend == "tf_cholesky":
        if mask is None:
            value, filtered_means, filtered_covariances = tf_kalman_filter(
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
                return_filtered=return_filtered,
            )
            mask_convention = "none"
            filter_name = "tf_cholesky_kalman"
        else:
            value, filtered_means, filtered_covariances = tf_masked_kalman_filter(
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
                return_filtered=return_filtered,
            )
            mask_convention = "static_dummy_row"
            filter_name = "tf_masked_cholesky_kalman"
    elif backend == "tf_masked_cholesky":
        if mask is None:
            raise ValueError("tf_masked_cholesky requires an observation mask")
        value, filtered_means, filtered_covariances = tf_masked_kalman_filter(
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
            return_filtered=return_filtered,
        )
        mask_convention = "static_dummy_row"
        filter_name = "tf_masked_cholesky_kalman"
    else:
        raise ValueError(f"unknown TensorFlow linear Gaussian backend: {backend}")

    diagnostics = _diagnostics(
        backend=backend,
        mask_convention=mask_convention,
        jitter=jitter,
    )
    return TFFilterValueResult(
        log_likelihood=value,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        metadata=_metadata(
            filter_name=filter_name,
            model=model,
            compiled_status="tf_function",
        ),
        diagnostics=diagnostics,
    )
