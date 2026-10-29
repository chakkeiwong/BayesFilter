"""TensorFlow eigen/SVD linear Gaussian Kalman value backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.svd_factor_tf import (
    eigh_logdet,
    eigh_solve,
    floor_count,
    psd_eigh,
    symmetrize,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural import FilterRunMetadata


TFSVDLinearValueBackend = Literal["tf_svd", "tf_masked_svd"]


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
        raise ValueError("SVD/eigen filters require a static observation length")
    return int(n_timesteps)


@tf.function
def tf_svd_kalman_log_likelihood(
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
    singular_floor: tf.Tensor | float = 1e-12,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Dense prediction-error likelihood using eigensolves for innovations."""

    value, floor_count_value, residual, implemented_covariance = tf_svd_kalman_filter(
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
        singular_floor=singular_floor,
        return_filtered=False,
    )
    return value, floor_count_value, residual, implemented_covariance


@tf.function
def tf_svd_masked_kalman_log_likelihood(
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
    singular_floor: tf.Tensor | float = 1e-12,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Masked prediction-error likelihood using eigensolves for innovations."""

    value, floor_count_value, residual, implemented_covariance = (
        tf_svd_masked_kalman_filter(
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
            singular_floor=singular_floor,
            return_filtered=False,
        )
    )
    return value, floor_count_value, residual, implemented_covariance


@tf.function
def tf_svd_kalman_filter(
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
    singular_floor: tf.Tensor | float = 1e-12,
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Dense SVD/eigen Kalman recursion.

    The likelihood and update use the implemented innovation covariance whose
    eigenvalues are floored by ``singular_floor``.  The returned covariance is
    the last implemented innovation covariance, for diagnostics only.
    """

    y = _as_observation_matrix(observations)
    n_timesteps = _static_num_timesteps(y)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _to_tensor(transition_covariance)
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _to_tensor(observation_covariance)
    mean = _to_tensor(initial_state_mean)
    covariance = symmetrize(_to_tensor(initial_state_covariance))
    jitter_tensor = tf.cast(jitter, tf.float64)
    singular_floor_tensor = tf.cast(singular_floor, tf.float64)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    max_floor_count = tf.constant(0, dtype=tf.int32)
    max_projection_residual = tf.constant(0.0, dtype=tf.float64)
    last_implemented_covariance = tf.zeros((obs_dim, obs_dim), dtype=tf.float64)
    means = []
    covariances = []

    for t in range(n_timesteps):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        observation_noise = H + jitter_tensor * obs_identity

        predicted_mean = c + _matvec(T, mean)
        predicted_covariance = symmetrize(T @ covariance @ tf.transpose(T) + Q)
        innovation = y[t] - (d + _matvec(Z, predicted_mean))
        raw_innovation_covariance = symmetrize(
            Z @ predicted_covariance @ tf.transpose(Z) + observation_noise
        )
        eigenvalues, floored, eigenvectors, implemented_covariance, residual = psd_eigh(
            raw_innovation_covariance,
            singular_floor_tensor,
        )
        solve_innovation = eigh_solve(eigenvectors, floored, innovation)
        innovation_precision = eigh_solve(eigenvectors, floored, obs_identity)
        mahalanobis = tf.reduce_sum(innovation * solve_innovation)
        log_det = eigh_logdet(floored)
        contribution = -0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )

        kalman_gain = predicted_covariance @ tf.transpose(Z) @ innovation_precision
        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ Z
        filtered_covariance = symmetrize(
            joseph_left @ predicted_covariance @ tf.transpose(joseph_left)
            + kalman_gain @ observation_noise @ tf.transpose(kalman_gain)
        )

        mean = filtered_mean
        covariance = filtered_covariance
        log_likelihood = log_likelihood + contribution
        max_floor_count = tf.maximum(
            max_floor_count,
            floor_count(eigenvalues, singular_floor_tensor),
        )
        max_projection_residual = tf.maximum(max_projection_residual, residual)
        last_implemented_covariance = implemented_covariance
        if return_filtered:
            means.append(filtered_mean)
            covariances.append(filtered_covariance)

    return (
        log_likelihood,
        max_floor_count,
        max_projection_residual,
        last_implemented_covariance,
    )


@tf.function
def tf_svd_masked_kalman_filter(
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
    singular_floor: tf.Tensor | float = 1e-12,
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Static-shape masked SVD/eigen Kalman recursion."""

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
    covariance = symmetrize(_to_tensor(initial_state_covariance))
    jitter_tensor = tf.cast(jitter, tf.float64)
    singular_floor_tensor = tf.cast(singular_floor, tf.float64)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    max_floor_count = tf.constant(0, dtype=tf.int32)
    max_projection_residual = tf.constant(0.0, dtype=tf.float64)
    last_implemented_covariance = tf.zeros((obs_dim, obs_dim), dtype=tf.float64)
    means = []
    covariances = []

    for t in range(n_timesteps):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        base_observation_noise = H + jitter_tensor * obs_identity

        predicted_mean = c + _matvec(T, mean)
        predicted_covariance = symmetrize(T @ covariance @ tf.transpose(T) + Q)
        row_weight = tf.cast(observation_mask[t], tf.float64)
        missing_weight = 1.0 - row_weight
        row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
        masked_observation_matrix = Z * row_weight[:, tf.newaxis]
        masked_observation_noise = (
            base_observation_noise * row_outer + tf.linalg.diag(missing_weight)
        )
        innovation = (y[t] - (d + _matvec(Z, predicted_mean))) * row_weight
        raw_innovation_covariance = symmetrize(
            masked_observation_matrix
            @ predicted_covariance
            @ tf.transpose(masked_observation_matrix)
            + masked_observation_noise
        )
        eigenvalues, floored, eigenvectors, implemented_covariance, residual = psd_eigh(
            raw_innovation_covariance,
            singular_floor_tensor,
        )
        solve_innovation = eigh_solve(eigenvectors, floored, innovation)
        innovation_precision = eigh_solve(eigenvectors, floored, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ tf.transpose(masked_observation_matrix)
            @ innovation_precision
        )
        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ masked_observation_matrix
        filtered_covariance = symmetrize(
            joseph_left @ predicted_covariance @ tf.transpose(joseph_left)
            + kalman_gain @ masked_observation_noise @ tf.transpose(kalman_gain)
        )
        mahalanobis = tf.reduce_sum(innovation * solve_innovation)
        log_det = eigh_logdet(floored)
        missing_count = tf.reduce_sum(missing_weight)
        contribution = -0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
            - missing_count * dummy_log_norm
        )

        mean = filtered_mean
        covariance = filtered_covariance
        log_likelihood = log_likelihood + contribution
        max_floor_count = tf.maximum(
            max_floor_count,
            floor_count(eigenvalues, singular_floor_tensor),
        )
        max_projection_residual = tf.maximum(max_projection_residual, residual)
        last_implemented_covariance = implemented_covariance
        if return_filtered:
            means.append(filtered_mean)
            covariances.append(filtered_covariance)

    return (
        log_likelihood,
        max_floor_count,
        max_projection_residual,
        last_implemented_covariance,
    )


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
    singular_floor: tf.Tensor | float,
    floor_count_value: tf.Tensor,
    psd_projection_residual: tf.Tensor,
    implemented_covariance: tf.Tensor,
) -> TFFilterDiagnostics:
    return TFFilterDiagnostics(
        backend=backend,
        mask_convention=mask_convention,
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=tf.float64),
            singular_floor=tf.convert_to_tensor(singular_floor, dtype=tf.float64),
            floor_count=tf.convert_to_tensor(floor_count_value, dtype=tf.int32),
            psd_projection_residual=tf.convert_to_tensor(
                psd_projection_residual,
                dtype=tf.float64,
            ),
            implemented_covariance=tf.convert_to_tensor(
                implemented_covariance,
                dtype=tf.float64,
            ),
            branch_label="eigh_value_with_floor_metadata",
            derivative_target="blocked",
        ),
        extra={
            "factorization": "tf.linalg.eigh",
            "derivative_status_reason": "SVD/eigen derivatives are not certified in value Phase 2.",
        },
    )


def tf_svd_linear_gaussian_log_likelihood(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    *,
    backend: TFSVDLinearValueBackend = "tf_svd",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
    singular_floor: tf.Tensor | float = 1e-12,
) -> TFFilterValueResult:
    """Dispatch to a TensorFlow SVD/eigen linear Gaussian value backend."""

    y = _as_observation_matrix(observations)
    mask = observation_mask if observation_mask is not None else model.observation_mask
    if backend == "tf_svd":
        if mask is None:
            value, floor_count_value, residual, implemented_covariance = (
                tf_svd_kalman_log_likelihood(
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
                    singular_floor=singular_floor,
                )
            )
            filter_name = "tf_svd_kalman"
            mask_convention = "none"
        else:
            value, floor_count_value, residual, implemented_covariance = (
                tf_svd_masked_kalman_log_likelihood(
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
                    singular_floor=singular_floor,
                )
            )
            filter_name = "tf_svd_masked_kalman"
            mask_convention = "static_dummy_row"
    elif backend == "tf_masked_svd":
        if mask is None:
            raise ValueError("tf_masked_svd requires an observation mask")
        value, floor_count_value, residual, implemented_covariance = (
            tf_svd_masked_kalman_log_likelihood(
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
                singular_floor=singular_floor,
            )
        )
        filter_name = "tf_svd_masked_kalman"
        mask_convention = "static_dummy_row"
    else:
        raise ValueError(f"unknown TensorFlow SVD linear Gaussian backend: {backend}")

    return TFFilterValueResult(
        log_likelihood=value,
        filtered_means=None,
        filtered_covariances=None,
        metadata=_metadata(filter_name=filter_name, model=model),
        diagnostics=_diagnostics(
            backend=backend,
            mask_convention=mask_convention,
            jitter=jitter,
            singular_floor=singular_floor,
            floor_count_value=floor_count_value,
            psd_projection_residual=residual,
            implemented_covariance=implemented_covariance,
        ),
    )
