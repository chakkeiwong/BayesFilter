"""TensorFlow QR/square-root linear Gaussian Kalman backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.dtypes_tf import as_float_tensor, common_floating_dtype
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor,
    factor_solve,
    lower_factor_from_horizontal_stack,
)
from bayesfilter.linear.types_tf import TFLinearGaussianStateSpace
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural import FilterRunMetadata


TFQRLinearValueBackend = Literal["tf_qr", "tf_masked_qr"]


def _to_tensor(value: object, dtype: tf.DType) -> tf.Tensor:
    return as_float_tensor(value, dtype)


def _value_dtype(
    observations: object,
    transition_offset: object,
    transition_matrix: object,
    transition_covariance: object,
    observation_offset: object,
    observation_matrix: object,
    observation_covariance: object,
    initial_state_mean: object,
    initial_state_covariance: object,
    jitter: object,
    *,
    context: str,
) -> tf.DType:
    return common_floating_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context=context,
    )


def _matvec(matrix: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matvec(matrix, vector)


def _matrix_transpose(matrix: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matrix_transpose(matrix)


def _as_observation_matrix(
    observations: tf.Tensor,
    dtype: tf.DType | None = None,
) -> tf.Tensor:
    if dtype is None:
        dtype = common_floating_dtype(observations, context="observations")
    y = as_float_tensor(observations, dtype, name="observations")
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


def _as_batched_static_observation_matrix(
    observations: tf.Tensor,
    dtype: tf.DType | None = None,
) -> tf.Tensor:
    y = _as_observation_matrix(observations, dtype)
    if y.shape.rank != 2:
        raise ValueError("batched-static observations must have shape [time, observation]")
    return y


def _validate_batched_static_shapes(
    *,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> None:
    """Fail closed for the Phase 6 batched-static QR shape contract."""

    expected_ranks = {
        "initial_state_mean": (initial_state_mean, 2),
        "initial_state_covariance": (initial_state_covariance, 3),
        "transition_offset": (transition_offset, 2),
        "transition_matrix": (transition_matrix, 3),
        "transition_covariance": (transition_covariance, 3),
        "observation_offset": (observation_offset, 2),
        "observation_matrix": (observation_matrix, 3),
        "observation_covariance": (observation_covariance, 3),
    }
    for name, (tensor, rank) in expected_ranks.items():
        if tensor.shape.rank != rank:
            raise ValueError(f"{name} must have rank {rank} for batched-static QR")


def _batched_qr_positive(matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Batched thin QR with a positive diagonal in the triangular factor."""

    dtype = common_floating_dtype(matrix, context="batched QR matrix")
    matrix = as_float_tensor(matrix, dtype, name="matrix")
    q, r = tf.linalg.qr(matrix, full_matrices=False)
    signs = tf.sign(tf.linalg.diag_part(r))
    signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
    return q * signs[..., tf.newaxis, :], signs[..., :, tf.newaxis] * r


def _batched_lower_factor_from_horizontal_stack(stack: tf.Tensor) -> tf.Tensor:
    """Return batched lower factors ``L`` with ``L L.T = stack stack.T``."""

    _, r = _batched_qr_positive(_matrix_transpose(stack))
    return _matrix_transpose(r)


def _batched_factor_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve batched ``(factor @ factor.T) x = rhs`` for lower factors."""

    dtype = common_floating_dtype(factor, rhs, context="batched factor solve inputs")
    factor = as_float_tensor(factor, dtype, name="factor")
    rhs = as_float_tensor(rhs, dtype, name="rhs")
    first = tf.linalg.triangular_solve(factor, rhs, lower=True)
    return tf.linalg.triangular_solve(_matrix_transpose(factor), first, lower=False)


def _batched_cholesky_factor(covariance: tf.Tensor, jitter: tf.Tensor | float = 0.0) -> tf.Tensor:
    """Return batched lower Cholesky factors of symmetrized covariances."""

    dtype = common_floating_dtype(covariance, jitter, context="batched Cholesky inputs")
    covariance = _to_tensor(covariance, dtype)
    if covariance.shape.rank != 3:
        raise ValueError("batched Cholesky requires covariance shape [batch, dim, dim]")
    symmetric = 0.5 * (covariance + _matrix_transpose(covariance))
    jitter_tensor = _to_tensor(jitter, dtype)
    dim = tf.shape(symmetric)[-1]
    identity = tf.eye(dim, batch_shape=[tf.shape(symmetric)[0]], dtype=dtype)
    return tf.linalg.cholesky(symmetric + jitter_tensor * identity)


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
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """QR/square-root prediction-error log likelihood."""

    return tf_qr_sqrt_kalman_log_likelihood_compact(
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
        jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance),
    )


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

    return tf_qr_sqrt_masked_kalman_log_likelihood_compact(
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


@tf.function
def tf_qr_sqrt_kalman_log_likelihood_compact(
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
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """QR/square-root prediction-error log likelihood without filtered stacks."""

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="QR compact value inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    for t in range(n_timesteps):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        transition_covariance_factor = cholesky_factor(Q, 0.0)
        observation_covariance_factor = cholesky_factor(H + jitter_tensor * obs_identity, 0.0)
        observation_update_covariance_factor = (
            observation_covariance_factor
            if jitter_updates_filtered_covariance
            else cholesky_factor(H, 0.0)
        )

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
                kalman_gain @ observation_update_covariance_factor,
            ),
            axis=1,
        )
        filtered_factor = lower_factor_from_horizontal_stack(update_stack)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution

    return log_likelihood


@tf.function(reduce_retracing=True)
def tf_qr_sqrt_kalman_log_likelihood_while_loop(
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
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """QR/square-root log likelihood with a dynamic TensorFlow time loop.

    This value-only kernel is the graph-compressed counterpart to
    ``tf_qr_sqrt_kalman_log_likelihood_compact``.  It carries only the current
    filtered mean, current covariance square root, and accumulated log
    likelihood through a ``tf.while_loop`` body, so tracing and XLA graph size
    do not grow by duplicating one Python loop body per observation time.
    """

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="QR while-loop value inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = tf.shape(y)[0]
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean0 = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    state_dim = tf.shape(mean0)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor0 = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood0 = tf.constant(0.0, dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    t0 = tf.constant(0, dtype=tf.int32)

    def cond(t, *_state):
        return t < n_timesteps

    def body(t, mean, covariance_factor, log_likelihood):
        c = _vector_at_time(transition_offset, t)
        T = _matrix_at_time(transition_matrix, t)
        Q = _matrix_at_time(transition_covariance, t)
        d = _vector_at_time(observation_offset, t)
        Z = _matrix_at_time(observation_matrix, t)
        H = _matrix_at_time(observation_covariance, t)
        transition_covariance_factor = cholesky_factor(Q, 0.0)
        observation_covariance_factor = cholesky_factor(H + jitter_tensor * obs_identity, 0.0)
        observation_update_covariance_factor = (
            observation_covariance_factor
            if jitter_updates_filtered_covariance
            else cholesky_factor(H, 0.0)
        )

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
                kalman_gain @ observation_update_covariance_factor,
            ),
            axis=1,
        )
        filtered_factor = lower_factor_from_horizontal_stack(update_stack)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis
        )
        return t + 1, filtered_mean, filtered_factor, log_likelihood + contribution

    _, _, _, log_likelihood = tf.while_loop(
        cond,
        body,
        (t0, mean0, covariance_factor0, log_likelihood0),
        maximum_iterations=n_timesteps,
        parallel_iterations=1,
    )
    return log_likelihood


def _tf_qr_sqrt_factorized_kalman_log_likelihood_impl(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise_factor: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_factor: tf.Tensor,
    jitter: tf.Tensor | float,
    jitter_updates_filtered_covariance: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Evaluate a QR likelihood from an exact rectangular process factor."""

    dtype = common_floating_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_noise_factor,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_factor,
        jitter,
        context="factorized QR likelihood inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_noise_factor = _to_tensor(transition_noise_factor, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean0 = _to_tensor(initial_state_mean, dtype)
    initial_state_factor = _to_tensor(initial_state_factor, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    if transition_noise_factor.shape.rank != 2:
        raise ValueError("transition_noise_factor must have rank 2")
    if initial_state_factor.shape.rank != 2:
        raise ValueError("initial_state_factor must have rank 2")
    state_dim_static = mean0.shape[0]
    factor_state_dim = transition_noise_factor.shape[0]
    factor_innovation_dim = transition_noise_factor.shape[1]
    initial_factor_state_dim = initial_state_factor.shape[0]
    initial_factor_width = initial_state_factor.shape[1]
    if (
        state_dim_static is None
        or factor_state_dim is None
        or factor_innovation_dim is None
        or initial_factor_state_dim is None
        or initial_factor_width is None
    ):
        raise ValueError(
            "initial state and covariance factors need static dimensions"
        )
    if factor_state_dim != state_dim_static:
        raise ValueError(
            "transition_noise_factor first dimension must equal state dimension"
        )
    if initial_factor_state_dim != state_dim_static:
        raise ValueError(
            "initial_state_factor first dimension must equal state dimension"
        )

    n_timesteps = tf.shape(y)[0]
    state_dim = tf.shape(mean0)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor0 = lower_factor_from_horizontal_stack(initial_state_factor)
    observation_covariance_factor = cholesky_factor(
        observation_covariance + jitter_tensor * obs_identity,
        0.0,
    )
    observation_update_covariance_factor = (
        observation_covariance_factor
        if jitter_updates_filtered_covariance
        else cholesky_factor(observation_covariance, 0.0)
    )
    increments0 = tf.zeros((tf.shape(y)[0],), dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    t0 = tf.constant(0, dtype=tf.int32)
    log_likelihood0 = tf.constant(0.0, dtype=dtype)

    def cond(t, *_state):
        return t < n_timesteps

    def body(t, mean, covariance_factor, log_likelihood, increments):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        predicted_factor = lower_factor_from_horizontal_stack(
            tf.concat(
                (
                    transition_matrix @ covariance_factor,
                    transition_noise_factor,
                ),
                axis=1,
            )
        )
        predicted_covariance = predicted_factor @ tf.transpose(predicted_factor)

        innovation = y[t] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        innovation_factor = lower_factor_from_horizontal_stack(
            tf.concat(
                (
                    observation_matrix @ predicted_factor,
                    observation_covariance_factor,
                ),
                axis=1,
            )
        )
        innovation_precision = factor_solve(innovation_factor, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ tf.transpose(observation_matrix)
            @ innovation_precision
        )

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ observation_matrix
        filtered_factor = lower_factor_from_horizontal_stack(
            tf.concat(
                (
                    joseph_left @ predicted_factor,
                    kalman_gain @ observation_update_covariance_factor,
                ),
                axis=1,
            )
        )

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor))
        )
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        return (
            t + 1,
            filtered_mean,
            filtered_factor,
            log_likelihood + contribution,
            tf.tensor_scatter_nd_update(
                increments,
                tf.reshape(t, (1, 1)),
                tf.reshape(contribution, (1,)),
            ),
        )

    _, _, _, log_likelihood, increments = tf.while_loop(
        cond,
        body,
        (t0, mean0, covariance_factor0, log_likelihood0, increments0),
        maximum_iterations=n_timesteps,
        parallel_iterations=1,
    )
    return log_likelihood, increments


@tf.function(jit_compile=True)
def tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise_factor: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_factor: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return total and per-period QR log likelihood from a process factor.

    Both covariance factors may be rectangular. The implemented covariance
    laws are exactly ``factor @ factor.T``; no process/initial jitter or
    spectral repair is introduced.
    """

    return _tf_qr_sqrt_factorized_kalman_log_likelihood_impl(
        observations,
        transition_offset,
        transition_matrix,
        transition_noise_factor,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_factor,
        jitter,
        bool(jitter_updates_filtered_covariance),
    )


@tf.function
def tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments_graph(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise_factor: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_factor: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Non-XLA diagnostic counterpart to the factorized QR value kernel."""

    return _tf_qr_sqrt_factorized_kalman_log_likelihood_impl(
        observations,
        transition_offset,
        transition_matrix,
        transition_noise_factor,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_factor,
        jitter,
        bool(jitter_updates_filtered_covariance),
    )


@tf.function(jit_compile=True)
def tf_qr_sqrt_factorized_kalman_log_likelihood(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_noise_factor: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_factor: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """Return a QR log likelihood for a possibly rank-deficient process law."""

    value, _increments = (
        tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments(
            observations,
            transition_offset,
            transition_matrix,
            transition_noise_factor,
            observation_offset,
            observation_matrix,
            observation_covariance,
            initial_state_mean,
            initial_state_factor,
            jitter,
            jitter_updates_filtered_covariance,
        )
    )
    return value


@tf.function
def tf_qr_sqrt_kalman_log_likelihood_batched_static(
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
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """Batched-static QR/square-root prediction-error log likelihood.

    This helper is intentionally separate from the scalar compact QR API.  Its
    leading dimension is a batch/chain dimension, not time.  It supports shared
    observations ``[T, M]`` and time-invariant state-space tensors with shapes
    ``[B, N]``, ``[B, N, N]``, ``[B, M]``, ``[B, M, N]``, and ``[B, M, M]``.
    It does not implement batched time-varying tensors.
    """

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="QR batched-static value inputs",
    )
    y = _as_batched_static_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    _validate_batched_static_shapes(
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=mean,
        initial_state_covariance=initial_state_covariance,
    )
    jitter_tensor = _to_tensor(jitter, dtype)

    batch_size = tf.shape(mean)[0]
    state_dim = tf.shape(mean)[1]
    obs_dim = tf.shape(observation_offset)[1]
    state_identity = tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)
    obs_identity = tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)
    covariance_factor = _batched_cholesky_factor(initial_state_covariance, 0.0)
    transition_covariance_factor = _batched_cholesky_factor(transition_covariance, 0.0)
    observation_covariance_factor = _batched_cholesky_factor(
        observation_covariance + jitter_tensor * obs_identity,
        0.0,
    )
    observation_update_covariance_factor = (
        observation_covariance_factor
        if jitter_updates_filtered_covariance
        else _batched_cholesky_factor(observation_covariance, 0.0)
    )
    log_likelihood = tf.zeros((batch_size,), dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        prediction_stack = tf.concat(
            (
                transition_matrix @ covariance_factor,
                transition_covariance_factor,
            ),
            axis=2,
        )
        predicted_factor = _batched_lower_factor_from_horizontal_stack(prediction_stack)
        predicted_covariance = predicted_factor @ _matrix_transpose(predicted_factor)

        innovation = y[t][tf.newaxis, :] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        innovation_stack = tf.concat(
            (
                observation_matrix @ predicted_factor,
                observation_covariance_factor,
            ),
            axis=2,
        )
        innovation_factor = _batched_lower_factor_from_horizontal_stack(innovation_stack)
        innovation_precision = _batched_factor_solve(innovation_factor, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ _matrix_transpose(observation_matrix)
            @ innovation_precision
        )

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ observation_matrix
        update_stack = tf.concat(
            (
                joseph_left @ predicted_factor,
                kalman_gain @ observation_update_covariance_factor,
            ),
            axis=2,
        )
        filtered_factor = _batched_lower_factor_from_horizontal_stack(update_stack)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[..., tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)),
            axis=-1,
        )
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution

    return log_likelihood


@tf.function(reduce_retracing=True)
def tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop(
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
    jitter_updates_filtered_covariance: bool = True,
) -> tf.Tensor:
    """Batched-static QR log likelihood with a dynamic TensorFlow time loop.

    The leading dimension is a batch/chain dimension.  The time recursion is a
    single ``tf.while_loop`` body carrying only current batch means, current
    batch covariance factors, and accumulated batch log likelihoods.
    """

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="QR batched-static while-loop value inputs",
    )
    y = _as_batched_static_observation_matrix(observations, dtype)
    n_timesteps = tf.shape(y)[0]
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean0 = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    _validate_batched_static_shapes(
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=mean0,
        initial_state_covariance=initial_state_covariance,
    )
    jitter_tensor = _to_tensor(jitter, dtype)

    batch_size = tf.shape(mean0)[0]
    state_dim = tf.shape(mean0)[1]
    obs_dim = tf.shape(observation_offset)[1]
    state_identity = tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)
    obs_identity = tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)
    covariance_factor0 = _batched_cholesky_factor(initial_state_covariance, 0.0)
    transition_covariance_factor = _batched_cholesky_factor(transition_covariance, 0.0)
    observation_covariance_factor = _batched_cholesky_factor(
        observation_covariance + jitter_tensor * obs_identity,
        0.0,
    )
    observation_update_covariance_factor = (
        observation_covariance_factor
        if jitter_updates_filtered_covariance
        else _batched_cholesky_factor(observation_covariance, 0.0)
    )
    log_likelihood0 = tf.zeros((batch_size,), dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    t0 = tf.constant(0, dtype=tf.int32)

    def cond(t, *_state):
        return t < n_timesteps

    def body(t, mean, covariance_factor, log_likelihood):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        prediction_stack = tf.concat(
            (
                transition_matrix @ covariance_factor,
                transition_covariance_factor,
            ),
            axis=2,
        )
        predicted_factor = _batched_lower_factor_from_horizontal_stack(prediction_stack)
        predicted_covariance = predicted_factor @ _matrix_transpose(predicted_factor)

        innovation = y[t][tf.newaxis, :] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        innovation_stack = tf.concat(
            (
                observation_matrix @ predicted_factor,
                observation_covariance_factor,
            ),
            axis=2,
        )
        innovation_factor = _batched_lower_factor_from_horizontal_stack(innovation_stack)
        innovation_precision = _batched_factor_solve(innovation_factor, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ _matrix_transpose(observation_matrix)
            @ innovation_precision
        )

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ observation_matrix
        update_stack = tf.concat(
            (
                joseph_left @ predicted_factor,
                kalman_gain @ observation_update_covariance_factor,
            ),
            axis=2,
        )
        filtered_factor = _batched_lower_factor_from_horizontal_stack(update_stack)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[..., tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)),
            axis=-1,
        )
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis
        )
        return t + 1, filtered_mean, filtered_factor, log_likelihood + contribution

    _, _, _, log_likelihood = tf.while_loop(
        cond,
        body,
        (t0, mean0, covariance_factor0, log_likelihood0),
        maximum_iterations=n_timesteps,
        parallel_iterations=1,
    )
    return log_likelihood


@tf.function
def tf_qr_sqrt_masked_kalman_log_likelihood_batched_static(
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
    """Batched-static masked QR/square-root prediction-error log likelihood.

    The leading dimension of every state-space tensor is a batch/chain
    dimension.  Observations and the boolean mask are shared across the batch
    with shape ``[T, M]``.  The masking convention matches the scalar compact
    QR helper: missing rows are dummy finite observations with unit innovation
    variance and their dummy normalizing constants are subtracted.
    """

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="masked QR batched-static value inputs",
    )
    y = _as_batched_static_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    _validate_mask_shape(y, observation_mask)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    _validate_batched_static_shapes(
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=mean,
        initial_state_covariance=initial_state_covariance,
    )
    jitter_tensor = _to_tensor(jitter, dtype)

    batch_size = tf.shape(mean)[0]
    state_dim = tf.shape(mean)[1]
    obs_dim = tf.shape(observation_offset)[1]
    state_identity = tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)
    obs_identity = tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)
    covariance_factor = _batched_cholesky_factor(initial_state_covariance, 0.0)
    transition_covariance_factor = _batched_cholesky_factor(transition_covariance, 0.0)
    log_likelihood = tf.zeros((batch_size,), dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=dtype)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        prediction_stack = tf.concat(
            (
                transition_matrix @ covariance_factor,
                transition_covariance_factor,
            ),
            axis=2,
        )
        predicted_factor = _batched_lower_factor_from_horizontal_stack(prediction_stack)
        predicted_covariance = predicted_factor @ _matrix_transpose(predicted_factor)

        base_observation_covariance = observation_covariance + jitter_tensor * obs_identity
        row_weight = tf.cast(observation_mask[t], dtype)
        missing_weight = 1.0 - row_weight
        row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
        masked_observation_matrix = observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]
        masked_observation_covariance = (
            base_observation_covariance * row_outer[tf.newaxis, :, :]
            + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]
        )
        observation_covariance_factor = _batched_cholesky_factor(
            masked_observation_covariance,
            0.0,
        )

        innovation = (
            y[t][tf.newaxis, :]
            - (observation_offset + _matvec(observation_matrix, predicted_mean))
        ) * row_weight[tf.newaxis, :]
        innovation_stack = tf.concat(
            (
                masked_observation_matrix @ predicted_factor,
                observation_covariance_factor,
            ),
            axis=2,
        )
        innovation_factor = _batched_lower_factor_from_horizontal_stack(innovation_stack)
        innovation_precision = _batched_factor_solve(innovation_factor, obs_identity)
        kalman_gain = (
            predicted_covariance
            @ _matrix_transpose(masked_observation_matrix)
            @ innovation_precision
        )

        filtered_mean = predicted_mean + _matvec(kalman_gain, innovation)
        joseph_left = state_identity - kalman_gain @ masked_observation_matrix
        update_stack = tf.concat(
            (
                joseph_left @ predicted_factor,
                kalman_gain @ observation_covariance_factor,
            ),
            axis=2,
        )
        filtered_factor = _batched_lower_factor_from_horizontal_stack(update_stack)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[..., tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation), axis=[-2, -1])
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)),
            axis=-1,
        )
        missing_count = tf.reduce_sum(missing_weight)
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
            - missing_count * dummy_log_norm
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution

    return log_likelihood


@tf.function
def tf_qr_sqrt_masked_kalman_log_likelihood_compact(
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
    """Masked QR/square-root log likelihood without filtered stacks."""

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="masked QR compact value inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    _validate_mask_shape(y, observation_mask)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=dtype)

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

        row_weight = tf.cast(observation_mask[t], dtype)
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

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        missing_count = tf.reduce_sum(missing_weight)
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
            - missing_count * dummy_log_norm
        )
        mean = filtered_mean
        covariance_factor = filtered_factor
        log_likelihood = log_likelihood + contribution

    return log_likelihood


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
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Dense direct-QR square-root Kalman recursion."""

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="QR filtered value inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
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
        observation_update_covariance_factor = (
            observation_covariance_factor
            if jitter_updates_filtered_covariance
            else cholesky_factor(H, 0.0)
        )

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
                kalman_gain @ observation_update_covariance_factor,
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
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi) + log_det + mahalanobis
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

    dtype = _value_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        jitter,
        context="masked QR filtered value inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_num_timesteps(y)
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    _validate_mask_shape(y, observation_mask)
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)
    jitter_tensor = _to_tensor(jitter, dtype)

    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(_matrix_at_time(observation_matrix, tf.constant(0, dtype=tf.int32)))[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    covariance_factor = cholesky_factor(initial_state_covariance, 0.0)
    log_likelihood = tf.constant(0.0, dtype=dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=dtype)
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

        row_weight = tf.cast(observation_mask[t], dtype)
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
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
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
    dtype: tf.DType = tf.float64,
) -> TFFilterDiagnostics:
    dtype = tf.as_dtype(dtype)
    return TFFilterDiagnostics(
        backend=backend,
        mask_convention=mask_convention,
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=dtype),
            singular_floor=tf.constant(0.0, dtype=dtype),
            floor_count=tf.constant(0, dtype=tf.int32),
            psd_projection_residual=tf.constant(0.0, dtype=dtype),
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
    jitter_updates_filtered_covariance: bool = True,
) -> TFFilterValueResult:
    """Dispatch to a QR/square-root TensorFlow linear Gaussian value backend."""

    mask = observation_mask if observation_mask is not None else model.observation_mask
    dtype = _value_dtype(
        observations,
        model.transition_offset,
        model.transition_matrix,
        model.transition_covariance,
        model.observation_offset,
        model.observation_matrix,
        model.observation_covariance,
        model.initial_mean,
        model.initial_covariance,
        jitter,
        context="TensorFlow QR dispatcher inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    if backend == "tf_qr":
        if mask is None:
            if return_filtered:
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
                    jitter_updates_filtered_covariance=bool(
                        jitter_updates_filtered_covariance),
                )
            else:
                value = tf_qr_sqrt_kalman_log_likelihood_compact(
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
                    jitter_updates_filtered_covariance=bool(
                        jitter_updates_filtered_covariance),
                )
                filtered_means = None
                filtered_covariances = None
            filter_name = "tf_qr_sqrt_kalman"
            mask_convention = "none"
        else:
            if return_filtered:
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
            else:
                value = tf_qr_sqrt_masked_kalman_log_likelihood_compact(
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
                filtered_means = None
                filtered_covariances = None
            filter_name = "tf_qr_sqrt_masked_kalman"
            mask_convention = "static_dummy_row"
    elif backend == "tf_masked_qr":
        if mask is None:
            raise ValueError("tf_masked_qr requires an observation mask")
        if return_filtered:
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
        else:
            value = tf_qr_sqrt_masked_kalman_log_likelihood_compact(
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
            filtered_means = None
            filtered_covariances = None
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
            dtype=dtype,
        ),
    )
