"""Independent LGSSM value/score references for Younis-KDM diagnostics.

The references are deliberately separate from the LEDH executor.  They include
an exact analytical Kalman recursion and a fixed-stream bootstrap particle
filter comparator.  Neither route uses autodiff.
"""

from __future__ import annotations

import math

import tensorflow as tf


Tensor = tf.Tensor

EXACT_LGSSM_SCORE_TARGET = "EXACT-LGSSM-SCORE"
BOOTSTRAP_FIXED_STREAM_TARGET = "BOOTSTRAP-FIXED-STREAM-FINITE"


def _static_matrix_dimensions(
    observations: Tensor,
    transition_matrix: Tensor,
    observation_matrix: Tensor,
) -> tuple[int, int, int]:
    if transition_matrix.shape.rank != 2:
        raise ValueError("transition_matrix must have shape [D,D]")
    state_dimension = transition_matrix.shape[0]
    if state_dimension is None or transition_matrix.shape[1] != state_dimension:
        raise ValueError("transition_matrix must be a statically known square matrix")
    if observation_matrix.shape.rank != 2:
        raise ValueError("observation_matrix must have shape [O,D]")
    observation_dimension = observation_matrix.shape[0]
    if observation_dimension is None or observation_matrix.shape[1] != state_dimension:
        raise ValueError("observation_matrix must have a statically known shape [O,D]")
    if observations.shape.rank != 2:
        raise ValueError("observations must have shape [T,O]")
    horizon = observations.shape[0]
    if horizon is None or observations.shape[1] != observation_dimension:
        raise ValueError("observations must have a statically known shape [T,O]")
    return int(state_dimension), int(observation_dimension), int(horizon)


def matrix_lgssm_value_and_directional_score(
    observations: Tensor,
    transition_matrix: Tensor,
    d_transition_matrix: Tensor,
    observation_matrix: Tensor,
    initial_mean: Tensor,
    initial_covariance: Tensor,
    process_covariance: Tensor,
    observation_covariance: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return exact LGSSM log likelihood and a transition-matrix derivative.

    The transition is ``x_t = A x_{t-1} + epsilon_t`` and the observation is
    ``y_t = H x_t + nu_t``.  Only ``A`` depends on the scalar differentiation
    coordinate, with supplied total tangent ``dA``.  ``Q``, ``R``, and the
    initial Gaussian law are fixed.  The score is propagated analytically
    through every prediction, innovation, gain, mean, and covariance update.
    """

    transition_matrix = tf.convert_to_tensor(transition_matrix)
    dtype = transition_matrix.dtype
    if not dtype.is_floating:
        raise ValueError("transition_matrix must use a floating TensorFlow dtype")
    observations = tf.convert_to_tensor(observations, dtype)
    d_transition_matrix = tf.convert_to_tensor(d_transition_matrix, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    initial_mean = tf.convert_to_tensor(initial_mean, dtype)
    initial_covariance = tf.convert_to_tensor(initial_covariance, dtype)
    process_covariance = tf.convert_to_tensor(process_covariance, dtype)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype)
    state_dimension, observation_dimension, _ = _static_matrix_dimensions(
        observations, transition_matrix, observation_matrix
    )
    tf.ensure_shape(d_transition_matrix, [state_dimension, state_dimension])
    tf.ensure_shape(initial_mean, [state_dimension])
    tf.ensure_shape(initial_covariance, [state_dimension, state_dimension])
    tf.ensure_shape(process_covariance, [state_dimension, state_dimension])
    tf.ensure_shape(
        observation_covariance,
        [observation_dimension, observation_dimension],
    )

    mean = initial_mean
    covariance = initial_covariance
    d_mean = tf.zeros_like(mean)
    d_covariance = tf.zeros_like(covariance)
    value = tf.zeros([], dtype)
    score = tf.zeros([], dtype)
    identity = tf.eye(state_dimension, dtype=dtype)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)

    for observation in tf.unstack(observations, axis=0):
        predicted_mean = tf.linalg.matvec(transition_matrix, mean)
        d_predicted_mean = tf.linalg.matvec(
            d_transition_matrix, mean
        ) + tf.linalg.matvec(transition_matrix, d_mean)
        predicted_covariance = (
            transition_matrix
            @ covariance
            @ tf.linalg.matrix_transpose(transition_matrix)
            + process_covariance
        )
        d_predicted_covariance = (
            d_transition_matrix
            @ covariance
            @ tf.linalg.matrix_transpose(transition_matrix)
            + transition_matrix
            @ d_covariance
            @ tf.linalg.matrix_transpose(transition_matrix)
            + transition_matrix
            @ covariance
            @ tf.linalg.matrix_transpose(d_transition_matrix)
        )

        innovation = observation - tf.linalg.matvec(observation_matrix, predicted_mean)
        d_innovation = -tf.linalg.matvec(observation_matrix, d_predicted_mean)
        innovation_covariance = (
            observation_matrix
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(observation_matrix)
            + observation_covariance
        )
        d_innovation_covariance = (
            observation_matrix
            @ d_predicted_covariance
            @ tf.linalg.matrix_transpose(observation_matrix)
        )
        innovation_chol = tf.linalg.cholesky(innovation_covariance)
        solved_innovation = tf.squeeze(
            tf.linalg.cholesky_solve(innovation_chol, innovation[:, None]),
            axis=1,
        )
        inverse_innovation_covariance = tf.linalg.cholesky_solve(
            innovation_chol,
            tf.eye(observation_dimension, dtype=dtype),
        )
        log_determinant = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_chol))
        )
        value = value - 0.5 * (
            tf.cast(observation_dimension, dtype) * log_two_pi
            + log_determinant
            + tf.reduce_sum(innovation * solved_innovation)
        )
        d_log_determinant = tf.linalg.trace(
            inverse_innovation_covariance @ d_innovation_covariance
        )
        d_quadratic = 2.0 * tf.reduce_sum(
            d_innovation * solved_innovation
        ) - tf.reduce_sum(
            solved_innovation
            * tf.linalg.matvec(d_innovation_covariance, solved_innovation)
        )
        score = score - 0.5 * (d_log_determinant + d_quadratic)

        covariance_observation_transpose = (
            predicted_covariance @ tf.linalg.matrix_transpose(observation_matrix)
        )
        gain = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol,
                tf.linalg.matrix_transpose(covariance_observation_transpose),
            )
        )
        d_gain_rhs = (
            d_predicted_covariance @ tf.linalg.matrix_transpose(observation_matrix)
            - gain @ d_innovation_covariance
        )
        d_gain = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                innovation_chol,
                tf.linalg.matrix_transpose(d_gain_rhs),
            )
        )
        mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        d_mean = (
            d_predicted_mean
            + tf.linalg.matvec(d_gain, innovation)
            + tf.linalg.matvec(gain, d_innovation)
        )
        left = identity - gain @ observation_matrix
        d_left = -d_gain @ observation_matrix
        covariance = left @ predicted_covariance @ tf.linalg.matrix_transpose(
            left
        ) + gain @ observation_covariance @ tf.linalg.matrix_transpose(gain)
        d_covariance = (
            d_left @ predicted_covariance @ tf.linalg.matrix_transpose(left)
            + left @ d_predicted_covariance @ tf.linalg.matrix_transpose(left)
            + left @ predicted_covariance @ tf.linalg.matrix_transpose(d_left)
            + d_gain @ observation_covariance @ tf.linalg.matrix_transpose(gain)
            + gain @ observation_covariance @ tf.linalg.matrix_transpose(d_gain)
        )
        covariance = 0.5 * (covariance + tf.linalg.matrix_transpose(covariance))
        d_covariance = 0.5 * (d_covariance + tf.linalg.matrix_transpose(d_covariance))

    return value, score


def bootstrap_lgssm_fixed_stream_value_and_directional_score(
    initial_states: Tensor,
    transition_noises: Tensor,
    observations: Tensor,
    resampling_offsets: Tensor,
    transition_matrix: Tensor,
    d_transition_matrix: Tensor,
    observation_matrix: Tensor,
    process_covariance: Tensor,
    observation_covariance: Tensor,
) -> dict[str, Tensor]:
    """Run a bootstrap PF and its fixed-index analytical path derivative.

    Systematic-resampling offsets are fixed.  The realized integer ancestors
    are piecewise constant and are therefore gathered with their state
    tangents.  This is the total derivative of that finite fixed-stream
    bootstrap program, not an unbiased score identity for discrete resampling.
    """

    initial_states = tf.convert_to_tensor(initial_states)
    dtype = initial_states.dtype
    if not dtype.is_floating:
        raise ValueError("initial_states must use a floating TensorFlow dtype")
    transition_noises = tf.convert_to_tensor(transition_noises, dtype)
    observations = tf.convert_to_tensor(observations, dtype)
    resampling_offsets = tf.convert_to_tensor(resampling_offsets, dtype)
    transition_matrix = tf.convert_to_tensor(transition_matrix, dtype)
    d_transition_matrix = tf.convert_to_tensor(d_transition_matrix, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    process_covariance = tf.convert_to_tensor(process_covariance, dtype)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype)
    state_dimension, observation_dimension, horizon = _static_matrix_dimensions(
        observations, transition_matrix, observation_matrix
    )
    if initial_states.shape.rank != 2 or initial_states.shape[1] != state_dimension:
        raise ValueError("initial_states must have a statically known shape [N,D]")
    particle_count = initial_states.shape[0]
    if particle_count is None or particle_count < 2:
        raise ValueError("initial_states must contain a statically known N >= 2")
    tf.ensure_shape(
        transition_noises,
        [horizon, particle_count, state_dimension],
    )
    tf.ensure_shape(resampling_offsets, [horizon])
    tf.ensure_shape(d_transition_matrix, [state_dimension, state_dimension])
    tf.ensure_shape(process_covariance, [state_dimension, state_dimension])
    tf.ensure_shape(
        observation_covariance,
        [observation_dimension, observation_dimension],
    )
    tf.debugging.assert_all_finite(
        resampling_offsets, "resampling offsets must be finite"
    )
    tf.debugging.assert_greater_equal(
        resampling_offsets, tf.zeros_like(resampling_offsets)
    )
    tf.debugging.assert_less(resampling_offsets, tf.ones_like(resampling_offsets))

    process_chol = tf.linalg.cholesky(process_covariance)
    observation_chol = tf.linalg.cholesky(observation_covariance)
    observation_normalizer = tf.cast(observation_dimension, dtype) * tf.constant(
        math.log(2.0 * math.pi), dtype
    ) + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(observation_chol)))
    count = tf.cast(particle_count, dtype)
    states = initial_states
    d_states = tf.zeros_like(states)
    total = tf.zeros([], dtype)
    score = tf.zeros([], dtype)
    minimum_ess = count
    ancestor_history = []

    for time_index in range(horizon):
        children = tf.linalg.matmul(
            states, transition_matrix, transpose_b=True
        ) + tf.linalg.matmul(
            transition_noises[time_index], process_chol, transpose_b=True
        )
        d_children = tf.linalg.matmul(
            states, d_transition_matrix, transpose_b=True
        ) + tf.linalg.matmul(d_states, transition_matrix, transpose_b=True)
        observed = tf.linalg.matmul(children, observation_matrix, transpose_b=True)
        d_observed = tf.linalg.matmul(d_children, observation_matrix, transpose_b=True)
        residual = observations[time_index][None, :] - observed
        d_residual = -d_observed
        solved_residual = tf.linalg.matrix_transpose(
            tf.linalg.cholesky_solve(
                observation_chol,
                tf.linalg.matrix_transpose(residual),
            )
        )
        log_weights = -0.5 * (
            observation_normalizer + tf.reduce_sum(residual * solved_residual, axis=1)
        )
        d_log_weights = -tf.reduce_sum(d_residual * solved_residual, axis=1)
        log_normalizer = tf.reduce_logsumexp(log_weights)
        normalized_weights = tf.exp(log_weights - log_normalizer)
        increment_tangent = tf.reduce_sum(normalized_weights * d_log_weights)
        total = total + log_normalizer - tf.math.log(count)
        score = score + increment_tangent
        minimum_ess = tf.minimum(
            minimum_ess,
            tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights))),
        )

        cumulative = tf.math.cumsum(normalized_weights)
        cumulative = tf.concat([cumulative[:-1], tf.ones([1], dtype)], axis=0)
        positions = (
            resampling_offsets[time_index] + tf.cast(tf.range(particle_count), dtype)
        ) / count
        ancestors = tf.searchsorted(
            cumulative, positions, side="right", out_type=tf.int32
        )
        ancestor_history.append(ancestors)
        states = tf.gather(children, ancestors)
        d_states = tf.gather(d_children, ancestors)

    finite = (
        tf.math.is_finite(total)
        & tf.math.is_finite(score)
        & tf.reduce_all(tf.math.is_finite(states))
        & tf.reduce_all(tf.math.is_finite(d_states))
    )
    return {
        "value": total,
        "score": score,
        "minimum_ess": minimum_ess,
        "ancestor_indices": tf.stack(ancestor_history),
        "final_states": states,
        "d_final_states": d_states,
        "valid": finite,
    }


def make_bootstrap_lgssm_fixed_stream_kernel(
    *,
    transition_matrix_base: Tensor,
    transition_matrix_direction: Tensor,
    observation_matrix: Tensor,
    process_covariance: Tensor,
    observation_covariance: Tensor,
    particle_count: int,
    horizon: int,
    dtype: tf.dtypes.DType | str = tf.float64,
    jit_compile: bool = True,
):
    """Build a stable-shape XLA-capable bootstrap comparator kernel."""

    dtype = tf.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("dtype must be floating")
    particle_count = int(particle_count)
    horizon = int(horizon)
    if particle_count < 2 or horizon < 1:
        raise ValueError("particle_count must be >= 2 and horizon must be positive")
    base = tf.convert_to_tensor(transition_matrix_base, dtype)
    direction = tf.convert_to_tensor(transition_matrix_direction, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    process_covariance = tf.convert_to_tensor(process_covariance, dtype)
    observation_covariance = tf.convert_to_tensor(observation_covariance, dtype)
    state_dimension = base.shape[0]
    observation_dimension = observation_matrix.shape[0]
    if state_dimension is None or observation_dimension is None:
        raise ValueError("captured matrices must have statically known dimensions")
    tf.ensure_shape(base, [state_dimension, state_dimension])
    tf.ensure_shape(direction, [state_dimension, state_dimension])
    tf.ensure_shape(observation_matrix, [observation_dimension, state_dimension])

    @tf.function(
        input_signature=[
            tf.TensorSpec([1], dtype),
            tf.TensorSpec([particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
            tf.TensorSpec([horizon, observation_dimension], dtype),
            tf.TensorSpec([horizon], dtype),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        theta: Tensor,
        initial_states: Tensor,
        transition_noises: Tensor,
        observations: Tensor,
        resampling_offsets: Tensor,
    ) -> dict[str, Tensor]:
        transition = base + theta[0] * direction
        return bootstrap_lgssm_fixed_stream_value_and_directional_score(
            initial_states,
            transition_noises,
            observations,
            resampling_offsets,
            transition,
            direction,
            observation_matrix,
            process_covariance,
            observation_covariance,
        )

    kernel.target_label = BOOTSTRAP_FIXED_STREAM_TARGET
    kernel.derivative_semantics = (
        "analytical_total_derivative_of_fixed_stream_finite_bootstrap_program_"
        "with_realized_resampling_indices_piecewise_constant"
    )
    kernel.jit_compile = bool(jit_compile)
    return kernel


def scalar_lgssm_value_and_score(
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    initial_variance: float = 1.0,
    process_variance: float = 0.1,
    observation_variance: float = 0.2,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return exact ``log p_theta(y)`` and its score for scalar AR(1).

    The model is

    ``x_t = theta[0] x_{t-1} + Normal(0, process_variance)`` and
    ``y_t = x_t + Normal(0, observation_variance)``, with
    ``x_0 ~ Normal(0, initial_variance)``.

    The derivative is propagated through prediction, innovation, and Kalman
    covariance update.  No autodiff or LEDH implementation is used here.
    """

    theta = tf.convert_to_tensor(theta)
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    if theta.shape.rank != 1 or theta.shape[0] != 1:
        raise ValueError("theta must have shape [1]")
    if observations.shape.rank != 1 or observations.shape[0] is None:
        raise ValueError("observations must have a statically known shape [T]")
    dtype = theta.dtype
    if not dtype.is_floating:
        raise ValueError("theta must use a floating TensorFlow dtype")
    if initial_variance <= 0.0:
        raise ValueError("initial_variance must be positive")
    if process_variance <= 0.0 or observation_variance <= 0.0:
        raise ValueError("noise variances must be positive")

    phi = theta[0]
    process = tf.constant(process_variance, dtype)
    observation_noise = tf.constant(observation_variance, dtype)
    mean = tf.zeros([], dtype)
    variance = tf.constant(initial_variance, dtype)
    d_mean = tf.zeros([], dtype)
    d_variance = tf.zeros([], dtype)
    value = tf.zeros([], dtype)
    score = tf.zeros([], dtype)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)

    for observation in tf.unstack(observations, axis=0):
        predicted_mean = phi * mean
        predicted_variance = phi * phi * variance + process
        d_predicted_mean = mean + phi * d_mean
        d_predicted_variance = 2.0 * phi * variance + phi * phi * d_variance

        innovation = observation - predicted_mean
        d_innovation = -d_predicted_mean
        innovation_variance = predicted_variance + observation_noise
        d_innovation_variance = d_predicted_variance

        value = value - 0.5 * (
            log_two_pi
            + tf.math.log(innovation_variance)
            + innovation * innovation / innovation_variance
        )
        score = score - 0.5 * d_innovation_variance / innovation_variance
        score = score - innovation * d_innovation / innovation_variance
        score = score + 0.5 * (
            innovation
            * innovation
            * d_innovation_variance
            / (innovation_variance * innovation_variance)
        )

        gain = predicted_variance / innovation_variance
        d_gain = (
            d_predicted_variance * innovation_variance
            - predicted_variance * d_innovation_variance
        ) / (innovation_variance * innovation_variance)
        mean = predicted_mean + gain * innovation
        d_mean = d_predicted_mean + d_gain * innovation + gain * d_innovation
        variance = predicted_variance - gain * predicted_variance
        d_variance = (
            d_predicted_variance
            - d_gain * predicted_variance
            - gain * d_predicted_variance
        )

    return value, score


__all__ = [
    "BOOTSTRAP_FIXED_STREAM_TARGET",
    "EXACT_LGSSM_SCORE_TARGET",
    "bootstrap_lgssm_fixed_stream_value_and_directional_score",
    "make_bootstrap_lgssm_fixed_stream_kernel",
    "matrix_lgssm_value_and_directional_score",
    "scalar_lgssm_value_and_score",
]
