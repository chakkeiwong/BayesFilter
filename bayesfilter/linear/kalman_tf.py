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


def _cholesky_validity(
    matrix: tf.Tensor,
    *,
    psd_tolerance: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return a backend-portable SPD/PSD validity mask and safe factor.

    ``tf.linalg.eigvalsh`` is useful telemetry, but CUDA/XLA can return an
    incorrect zero or signed value for the tiny eigenvalues that arise in
    batched state-space covariances.  Cholesky is already the factorization
    used by the likelihood and is stable on the same matrices.  A positive
    ``psd_tolerance`` is used only for structural PSD prechecks; it does not
    alter a covariance passed to the probability law.  Invalid rows receive an
    identity factor so NaNs cannot escape the checked branch.
    """
    symmetric = _symmetrize(tf.convert_to_tensor(matrix, tf.float64))
    finite = tf.reduce_all(
        tf.math.is_finite(symmetric), axis=(-2, -1)
    )
    dimension = tf.shape(symmetric)[-1]
    identity = tf.eye(
        dimension,
        batch_shape=tf.shape(symmetric)[:-2],
        dtype=tf.float64,
    )
    candidate = symmetric + tf.cast(psd_tolerance, tf.float64) * identity
    safe_candidate = tf.where(
        finite[..., tf.newaxis, tf.newaxis], candidate, identity
    )
    factor = tf.linalg.cholesky(safe_candidate)
    diagonal = tf.linalg.diag_part(factor)
    valid = (
        finite
        & tf.reduce_all(tf.math.is_finite(factor), axis=(-2, -1))
        & tf.reduce_all(diagonal > tf.constant(0.0, tf.float64), axis=-1)
    )
    safe_factor = tf.where(
        valid[..., tf.newaxis, tf.newaxis], factor, identity
    )
    return valid, safe_factor


def _spectral_telemetry(
    matrix: tf.Tensor,
    valid: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return signed minimum-spectrum and condition telemetry portably.

    For a valid symmetric positive-definite matrix, singular values equal its
    eigenvalues.  CUDA/XLA has a known failure mode in ``eigvalsh`` for the
    tiny positive modes used by this model, whereas the SVD path is stable.
    Invalid matrices are already rejected by ``valid``; their minimum value is
    reported as a signed singular-value proxy so callers retain a negative
    diagnostic without allowing it to enter the probability law.
    """
    symmetric = _symmetrize(tf.convert_to_tensor(matrix, tf.float64))
    finite = tf.reduce_all(
        tf.math.is_finite(symmetric), axis=tuple(range(-2, 0))
    )
    dimension = tf.shape(symmetric)[-1]
    identity = tf.eye(
        dimension,
        batch_shape=tf.shape(symmetric)[:-2],
        dtype=tf.float64,
    )
    safe_matrix = tf.where(
        finite[..., tf.newaxis, tf.newaxis], symmetric, identity
    )
    singular_values = tf.stop_gradient(
        tf.linalg.svd(safe_matrix, compute_uv=False)
    )
    minimum = tf.reduce_min(singular_values, axis=-1)
    maximum = tf.reduce_max(singular_values, axis=-1)
    signed_minimum = tf.where(valid, minimum, -minimum)
    condition = tf.where(
        valid,
        maximum / tf.maximum(minimum, tf.constant(1.0e-300, tf.float64)),
        tf.zeros_like(maximum),
    )
    return signed_minimum, condition


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
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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
    eigenvalues = tf.linalg.eigvalsh(_symmetrize(innovation_covariance))
    min_eigenvalue = tf.reduce_min(eigenvalues)
    condition_estimate = tf.reduce_max(eigenvalues) / tf.maximum(min_eigenvalue, tf.constant(1.0e-300, tf.float64))

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
    return filtered_mean, filtered_covariance, adjusted_log_prob, min_eigenvalue, condition_estimate


def _checked_masked_step(
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
    active: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Advance only when the innovation covariance is positive definite.

    Cholesky validity precedes the probability-law factorization, preventing an
    indefinite innovation covariance from injecting NaNs into an HMC target.
    Eigenvalues remain telemetry only because CUDA/XLA eigensolvers can lose
    tiny positive eigenvalues. No ridge, floor, or modified probability law is
    introduced. The caller must reject the whole likelihood when any returned
    validity flag is false.
    """
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
    innovation_covariance = _symmetrize(
        masked_observation_matrix
        @ predicted_covariance
        @ tf.transpose(masked_observation_matrix)
        + masked_observation_noise
    )
    locally_valid, _diagnostic_factor = _cholesky_validity(
        innovation_covariance
    )
    min_eigenvalue, condition_estimate = _spectral_telemetry(
        innovation_covariance, locally_valid
    )
    step_valid = tf.logical_and(active, locally_valid)

    # XLA differentiates every statically unrolled step. Selecting an identity
    # covariance on an already-invalid branch keeps the probability-law
    # Cholesky total without altering a valid likelihood or admitting an
    # invalid step.
    safe_innovation_covariance = tf.where(
        locally_valid, innovation_covariance, obs_identity
    )
    log_prob, chol, _, _ = _innovation_log_prob(
        innovation, safe_innovation_covariance
    )
    gain_rhs = predicted_covariance @ tf.transpose(masked_observation_matrix)
    kalman_gain = tf.transpose(
        tf.linalg.cholesky_solve(chol, tf.transpose(gain_rhs))
    )
    filtered_mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
    left = state_identity - kalman_gain @ masked_observation_matrix
    filtered_covariance = _symmetrize(
        left @ predicted_covariance @ tf.transpose(left)
        + kalman_gain @ masked_observation_noise @ tf.transpose(kalman_gain)
    )
    missing_count = tf.reduce_sum(missing_weight)
    dummy_log_norm = tf.math.log(
        tf.constant(2.0 * math.pi, dtype=tf.float64)
    )
    adjusted_log_prob = log_prob + 0.5 * missing_count * dummy_log_norm
    next_mean = tf.where(step_valid, filtered_mean, mean)
    next_covariance = tf.where(step_valid, filtered_covariance, covariance)
    contribution = tf.where(
        step_valid, adjusted_log_prob, tf.constant(0.0, tf.float64)
    )
    return (
        next_mean,
        next_covariance,
        contribution,
        min_eigenvalue,
        condition_estimate,
        step_valid,
    )


def _validate_mask_shape(observations: tf.Tensor, observation_mask: tf.Tensor) -> None:
    tf.debugging.assert_equal(
        tf.shape(observation_mask),
        tf.shape(observations),
        message="Observation mask shape must match observations shape.",
    )


def _validate_checked_batched_static_shapes(
    *,
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the fixed `[B, ...]` contract used by NeuTra target batches."""

    expected_ranks = {
        "transition_offset": (transition_offset, 2),
        "transition_matrix": (transition_matrix, 3),
        "transition_covariance": (transition_covariance, 3),
        "observation_offset": (observation_offset, 2),
        "observation_matrix": (observation_matrix, 3),
        "observation_covariance": (observation_covariance, 3),
        "initial_state_mean": (initial_state_mean, 2),
        "initial_state_covariance": (initial_state_covariance, 3),
    }
    if observations.shape.rank != 2:
        raise ValueError("observations must have rank 2 [time, observation]")
    for name, (tensor, rank) in expected_ranks.items():
        if tensor.shape.rank != rank:
            raise ValueError(f"{name} must have rank {rank}")
    static = (
        observations.shape[0],
        initial_state_mean.shape[0],
        initial_state_mean.shape[1],
        observation_offset.shape[1],
    )
    if any(value is None for value in static):
        raise ValueError("checked batched-static filter requires fixed dimensions")
    time_steps, batch_size, state_dim, observation_dim = (
        int(value) for value in static
    )
    expected_shapes = {
        "transition_offset": (batch_size, state_dim),
        "transition_matrix": (batch_size, state_dim, state_dim),
        "transition_covariance": (batch_size, state_dim, state_dim),
        "observation_offset": (batch_size, observation_dim),
        "observation_matrix": (batch_size, observation_dim, state_dim),
        "observation_covariance": (
            batch_size,
            observation_dim,
            observation_dim,
        ),
        "initial_state_mean": (batch_size, state_dim),
        "initial_state_covariance": (batch_size, state_dim, state_dim),
    }
    for name, expected in expected_shapes.items():
        if tuple(expected_ranks[name][0].shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if observations.shape[1] != observation_dim:
        raise ValueError("observations and state-space tensors disagree")
    return time_steps, batch_size, state_dim, observation_dim


@tf.function
def tf_masked_kalman_filter_checked_batched_static_with_diagnostics(
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
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Checked covariance Kalman value for a true proposal batch.

    The leading axis indexes independent parameter proposals. TensorFlow
    linear algebra processes that axis natively; only the observation-time
    recursion is sequential. Invalid innovation covariances are replaced by an
    identity solely to keep Cholesky total, while `valid` remains false and the
    caller rejects the complete target row. No ridge, floor, or altered
    likelihood is introduced.

    Returns `(value, minimum_innovation_eigenvalue,
    maximum_innovation_condition, valid)` with one entry per batch row.
    """

    y = _as_observation_matrix(observations)
    transition_offset = tf.convert_to_tensor(transition_offset, tf.float64)
    transition_matrix = tf.convert_to_tensor(transition_matrix, tf.float64)
    transition_covariance = tf.convert_to_tensor(
        transition_covariance, tf.float64
    )
    observation_offset = tf.convert_to_tensor(observation_offset, tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, tf.float64)
    observation_covariance = tf.convert_to_tensor(
        observation_covariance, tf.float64
    )
    mean = tf.convert_to_tensor(initial_state_mean, tf.float64)
    covariance = _symmetrize(
        tf.convert_to_tensor(initial_state_covariance, tf.float64)
    )
    observation_mask = tf.convert_to_tensor(observation_mask, tf.bool)
    _validate_mask_shape(y, observation_mask)
    time_steps, batch_size, state_dim, observation_dim = (
        _validate_checked_batched_static_shapes(
            observations=y,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            initial_state_mean=mean,
            initial_state_covariance=covariance,
        )
    )
    jitter = _scalar_float64(jitter)
    state_identity = tf.eye(
        state_dim, batch_shape=[batch_size], dtype=tf.float64
    )
    observation_identity = tf.eye(
        observation_dim, batch_shape=[batch_size], dtype=tf.float64
    )
    log_likelihood = tf.zeros([batch_size], tf.float64)
    active = tf.ones([batch_size], tf.bool)
    minimum_eigenvalue = tf.fill(
        [batch_size], tf.constant(float("inf"), tf.float64)
    )
    maximum_condition = tf.zeros([batch_size], tf.float64)
    log_two_pi = tf.math.log(tf.constant(2.0 * math.pi, tf.float64))

    for time_index in range(time_steps):
        predicted_mean = transition_offset + tf.linalg.matvec(
            transition_matrix, mean
        )
        predicted_covariance = _symmetrize(
            transition_matrix
            @ covariance
            @ tf.linalg.matrix_transpose(transition_matrix)
            + transition_covariance
        )
        row_weight = tf.cast(observation_mask[time_index], tf.float64)
        missing_weight = 1.0 - row_weight
        row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
        masked_observation_matrix = (
            observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]
        )
        masked_observation_noise = (
            (observation_covariance + jitter * observation_identity)
            * row_outer[tf.newaxis, :, :]
            + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]
        )
        expected_observation = observation_offset + tf.linalg.matvec(
            observation_matrix, predicted_mean
        )
        innovation = (
            y[time_index][tf.newaxis, :] - expected_observation
        ) * row_weight[tf.newaxis, :]
        innovation_covariance = _symmetrize(
            masked_observation_matrix
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(masked_observation_matrix)
            + masked_observation_noise
        )
        locally_valid, _diagnostic_factor = _cholesky_validity(
            innovation_covariance
        )
        local_minimum, condition = _spectral_telemetry(
            innovation_covariance, locally_valid
        )
        step_valid = active & locally_valid
        safe_innovation_covariance = tf.where(
            locally_valid[:, tf.newaxis, tf.newaxis],
            innovation_covariance,
            observation_identity,
        )
        innovation_factor = tf.linalg.cholesky(safe_innovation_covariance)
        solved_innovation = tf.linalg.cholesky_solve(
            innovation_factor, innovation[..., tf.newaxis]
        )[..., 0]
        log_determinant = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)), axis=-1
        )
        mahalanobis = tf.reduce_sum(innovation * solved_innovation, axis=-1)
        contribution = -0.5 * (
            tf.cast(observation_dim, tf.float64) * log_two_pi
            + log_determinant
            + mahalanobis
        ) + 0.5 * tf.reduce_sum(missing_weight) * log_two_pi

        gain_rhs = (
            predicted_covariance
            @ tf.linalg.matrix_transpose(masked_observation_matrix)
        )
        kalman_gain = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_factor, tf.linalg.matrix_transpose(gain_rhs)
            )
        )
        filtered_mean = predicted_mean + tf.linalg.matvec(
            kalman_gain, innovation
        )
        left = state_identity - kalman_gain @ masked_observation_matrix
        filtered_covariance = _symmetrize(
            left
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(left)
            + kalman_gain
            @ masked_observation_noise
            @ tf.linalg.matrix_transpose(kalman_gain)
        )
        mean = tf.where(step_valid[:, tf.newaxis], filtered_mean, mean)
        covariance = tf.where(
            step_valid[:, tf.newaxis, tf.newaxis],
            filtered_covariance,
            covariance,
        )
        log_likelihood = log_likelihood + tf.where(
            step_valid, contribution, tf.zeros_like(contribution)
        )
        minimum_eigenvalue = tf.minimum(
            minimum_eigenvalue, local_minimum
        )
        maximum_condition = tf.maximum(maximum_condition, condition)
        active = step_valid

    return (
        log_likelihood,
        minimum_eigenvalue,
        maximum_condition,
        active,
    )


@tf.function
def tf_masked_kalman_filter_checked_batched_static_value(
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
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return checked batched-static likelihood values and row validity."""

    value, _minimum, _condition, valid = (
        tf_masked_kalman_filter_checked_batched_static_with_diagnostics(
            observations,
            transition_offset,
            transition_matrix,
            transition_covariance,
            observation_offset,
            observation_matrix,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
            observation_mask,
            jitter,
        )
    )
    return value, valid


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
        mean, covariance, contribution, _, _ = _masked_step(
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


@tf.function(reduce_retracing=True)
def tf_masked_kalman_filter_with_diagnostics(
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
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Masked covariance-form likelihood plus innovation conditioning traces.

    This is the same Cholesky recursion as ``tf_masked_kalman_filter``. The
    additional tensors are the per-period minimum innovation eigenvalue and
    spectral condition estimate; no covariance floor or ridge is introduced.
    """
    y = _as_observation_matrix(observations)
    transition_offset = tf.convert_to_tensor(transition_offset, dtype=tf.float64)
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    transition_covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    observation_offset = tf.convert_to_tensor(observation_offset, dtype=tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    mean = tf.convert_to_tensor(initial_state_mean, dtype=tf.float64)
    covariance = _symmetrize(tf.convert_to_tensor(initial_state_covariance, dtype=tf.float64))
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    jitter = _scalar_float64(jitter)
    _validate_mask_shape(y, observation_mask)
    state_identity = tf.eye(tf.shape(mean)[0], dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    min_eigenvalues = tf.TensorArray(tf.float64, size=tf.shape(y)[0])
    condition_estimates = tf.TensorArray(tf.float64, size=tf.shape(y)[0])
    for t in tf.range(tf.shape(y)[0]):
        mean, covariance, contribution, min_eigenvalue, condition_estimate = _masked_step(
            time_index=t, row=y[t], row_mask=observation_mask[t], mean=mean,
            covariance=covariance, transition_offset=transition_offset,
            transition_matrix=transition_matrix, transition_covariance=transition_covariance,
            observation_offset=observation_offset, observation_matrix=observation_matrix,
            observation_covariance=observation_covariance, jitter=jitter,
            state_identity=state_identity,
        )
        log_likelihood += contribution
        min_eigenvalues = min_eigenvalues.write(t, min_eigenvalue)
        condition_estimates = condition_estimates.write(t, condition_estimate)
    return (
        log_likelihood,
        min_eigenvalues.stack(),
        condition_estimates.stack(),
        mean,
        covariance,
    )


def _tf_masked_kalman_filter_checked_static(
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
    *,
    collect_diagnostics: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tuple[list[tf.Tensor], ...] | None]:
    """Run the statically unrolled checked recursion with optional telemetry."""
    y = _as_observation_matrix(observations)
    transition_offset = tf.convert_to_tensor(transition_offset, dtype=tf.float64)
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float64)
    transition_covariance = tf.convert_to_tensor(transition_covariance, dtype=tf.float64)
    observation_offset = tf.convert_to_tensor(observation_offset, dtype=tf.float64)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype=tf.float64)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype=tf.float64)
    mean = tf.convert_to_tensor(initial_state_mean, dtype=tf.float64)
    covariance = _symmetrize(tf.convert_to_tensor(initial_state_covariance, dtype=tf.float64))
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    jitter = _scalar_float64(jitter)
    _validate_mask_shape(y, observation_mask)
    state_identity = tf.eye(tf.shape(mean)[0], dtype=tf.float64)
    time_steps = y.shape[0]
    if time_steps is None:
        raise ValueError(
            "checked masked HMC recursion requires a static observation horizon"
        )
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    active = tf.constant(True)
    min_eigenvalues: list[tf.Tensor] = []
    condition_estimates: list[tf.Tensor] = []
    validity: list[tf.Tensor] = []
    # The horizon is part of the frozen target signature. Static unrolling
    # avoids TensorFlow's while_grad TemporaryVariable collision, which makes
    # a symbolically looped score fail from the third period onward.
    for t in range(int(time_steps)):
        (
            mean,
            covariance,
            contribution,
            min_eigenvalue,
            condition_estimate,
            step_valid,
        ) = _checked_masked_step(
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
            active=active,
        )
        log_likelihood += contribution
        active = step_valid
        if collect_diagnostics:
            min_eigenvalues.append(min_eigenvalue)
            condition_estimates.append(condition_estimate)
            validity.append(step_valid)
    diagnostics = None
    if collect_diagnostics:
        diagnostics = (min_eigenvalues, condition_estimates, validity)
    return log_likelihood, active, mean, covariance, diagnostics


def tf_masked_kalman_filter_checked_with_diagnostics(
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
) -> tuple[
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
    tf.Tensor,
]:
    """Masked likelihood with a pre-Cholesky innovation validity trace.

    Valid inputs use the same covariance-form recursion as
    :func:`tf_masked_kalman_filter_with_diagnostics`. Invalid inputs return a
    finite partial value plus false validity; callers must finite-reject the
    complete target. No regularization is added.
    """
    log_likelihood, _active, mean, covariance, diagnostics = (
        _tf_masked_kalman_filter_checked_static(
            observations,
            transition_offset,
            transition_matrix,
            transition_covariance,
            observation_offset,
            observation_matrix,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
            observation_mask,
            jitter,
            collect_diagnostics=True,
        )
    )
    if diagnostics is None:
        raise RuntimeError("checked Kalman diagnostics were not collected")
    min_eigenvalues, condition_estimates, validity = diagnostics
    return (
        log_likelihood,
        tf.stack(min_eigenvalues),
        tf.stack(condition_estimates),
        tf.stack(validity),
        mean,
        covariance,
    )


def tf_masked_kalman_filter_checked_value(
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
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return checked likelihood and final validity without telemetry arrays.

    This path retains the same positive-definiteness check, total Cholesky
    branch, and static horizon unroll as the diagnostic route. It only omits
    per-period eigenvalue, condition, and validity materialization, allowing an
    HMC caller to keep those observability tensors out of its compiled graph.
    """
    log_likelihood, valid, _mean, _covariance, _diagnostics = (
        _tf_masked_kalman_filter_checked_static(
            observations,
            transition_offset,
            transition_matrix,
            transition_covariance,
            observation_offset,
            observation_matrix,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
            observation_mask,
            jitter,
            collect_diagnostics=False,
        )
    )
    return log_likelihood, valid


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
