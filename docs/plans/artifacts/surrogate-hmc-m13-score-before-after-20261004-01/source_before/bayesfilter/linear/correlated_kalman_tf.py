"""Covariance-form TensorFlow Kalman filtering with correlated innovations.

The model convention is

``x_t = c_t + T_t x_(t-1) + u_t`` and
``y_t = d_t + Z_t x_t + eta_t``,

where ``S_t = Cov(u_t, eta_t')`` has shape ``[state, observation]``.  The
same-date innovation covariance and state/innovation covariance are therefore

``F_t = Z_t P_t Z_t' + Z_t S_t + S_t' Z_t' + R_t`` and
``C_t = P_t Z_t' + S_t``.

This dedicated API deliberately does not use ``TFLinearGaussianStateSpace``.
That shared container belongs to independent-noise dispatchers, which must not
silently ignore a nonzero ``S_t``.  Process covariance is consumed directly,
so exact positive-semidefinite/singular ``Q_t`` is supported without a ridge.
The caller remains responsible for constructing jointly valid ``Q_t/S_t/R_t``
primitive-noise blocks; the innovation covariance must be positive definite on
observed rows.
"""

from __future__ import annotations

import math

import tensorflow as tf


_LOG_2PI = tf.constant(math.log(2.0 * math.pi), dtype=tf.float64)


def _as_float64(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _require_rank(tensor: tf.Tensor, rank: int, name: str) -> None:
    if tensor.shape.rank is not None and tensor.shape.rank != rank:
        raise ValueError(f"{name} must have rank {rank}")
    tf.debugging.assert_rank(tensor, rank, message=f"{name} must have rank {rank}")


def _require_shape(tensor: tf.Tensor, expected: tf.Tensor, name: str) -> None:
    tf.debugging.assert_equal(
        tf.shape(tensor),
        expected,
        message=f"{name} has an incompatible shape",
    )


def _require_finite(tensor: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(tensor, f"{name} must be finite")


def _require_symmetric(matrix: tf.Tensor, name: str) -> tf.Tensor:
    tf.debugging.assert_near(
        matrix,
        tf.linalg.matrix_transpose(matrix),
        atol=1.0e-10,
        rtol=1.0e-10,
        message=f"{name} must be symmetric",
    )
    return _symmetrize(matrix)


def _canonicalize_observations(
    observations: tf.Tensor,
    observation_mask: tf.Tensor | None,
) -> tf.Tensor:
    if observation_mask is None:
        _require_finite(observations, "observations")
        return observations

    observed_entries_are_finite = tf.reduce_all(
        tf.logical_or(
            tf.logical_not(observation_mask),
            tf.math.is_finite(observations),
        )
    )
    tf.debugging.assert_equal(
        observed_entries_are_finite,
        True,
        message="Observed entries must be finite",
    )
    # A multiplication mask is unsafe because NaN * 0 remains NaN.
    return tf.where(observation_mask, observations, tf.zeros_like(observations))


def _prepare_scalar_inputs(
    *,
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor | None,
) -> tuple[tf.Tensor, ...]:
    y = _as_float64(observations)
    c = _as_float64(transition_offset)
    transition = _as_float64(transition_matrix)
    process_covariance = _as_float64(transition_covariance)
    d = _as_float64(observation_offset)
    observation = _as_float64(observation_matrix)
    measurement_covariance = _as_float64(observation_covariance)
    cross_covariance = _as_float64(state_measurement_cross_covariance)
    mean = _as_float64(initial_state_mean)
    covariance = _as_float64(initial_state_covariance)

    for name, tensor, rank in (
        ("observations", y, 2),
        ("transition_offset", c, 2),
        ("transition_matrix", transition, 3),
        ("transition_covariance", process_covariance, 3),
        ("observation_offset", d, 2),
        ("observation_matrix", observation, 3),
        ("observation_covariance", measurement_covariance, 3),
        ("state_measurement_cross_covariance", cross_covariance, 3),
        ("initial_state_mean", mean, 1),
        ("initial_state_covariance", covariance, 2),
    ):
        _require_rank(tensor, rank, name)

    time_dim = tf.shape(y)[0]
    observation_dim = tf.shape(y)[1]
    state_dim = tf.shape(mean)[0]
    tf.debugging.assert_positive(time_dim, message="observations must contain time rows")
    tf.debugging.assert_positive(observation_dim, message="observation dimension must be positive")
    tf.debugging.assert_positive(state_dim, message="state dimension must be positive")

    _require_shape(c, tf.stack([time_dim, state_dim]), "transition_offset")
    _require_shape(
        transition,
        tf.stack([time_dim, state_dim, state_dim]),
        "transition_matrix",
    )
    _require_shape(
        process_covariance,
        tf.stack([time_dim, state_dim, state_dim]),
        "transition_covariance",
    )
    _require_shape(d, tf.stack([time_dim, observation_dim]), "observation_offset")
    _require_shape(
        observation,
        tf.stack([time_dim, observation_dim, state_dim]),
        "observation_matrix",
    )
    _require_shape(
        measurement_covariance,
        tf.stack([time_dim, observation_dim, observation_dim]),
        "observation_covariance",
    )
    _require_shape(
        cross_covariance,
        tf.stack([time_dim, state_dim, observation_dim]),
        "state_measurement_cross_covariance",
    )
    _require_shape(covariance, tf.stack([state_dim, state_dim]), "initial_state_covariance")

    mask = None
    if observation_mask is not None:
        mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
        _require_rank(mask, 2, "observation_mask")
        _require_shape(mask, tf.shape(y), "observation_mask")
    y = _canonicalize_observations(y, mask)

    for name, tensor in (
        ("transition_offset", c),
        ("transition_matrix", transition),
        ("transition_covariance", process_covariance),
        ("observation_offset", d),
        ("observation_matrix", observation),
        ("observation_covariance", measurement_covariance),
        ("state_measurement_cross_covariance", cross_covariance),
        ("initial_state_mean", mean),
        ("initial_state_covariance", covariance),
    ):
        _require_finite(tensor, name)

    process_covariance = _require_symmetric(
        process_covariance,
        "transition_covariance",
    )
    measurement_covariance = _require_symmetric(
        measurement_covariance,
        "observation_covariance",
    )
    covariance = _require_symmetric(covariance, "initial_state_covariance")
    return (
        y,
        c,
        transition,
        process_covariance,
        d,
        observation,
        measurement_covariance,
        cross_covariance,
        mean,
        covariance,
        mask,
    )


def _prepare_batched_inputs(
    *,
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor | None,
) -> tuple[tf.Tensor, ...]:
    y = _as_float64(observations)
    c = _as_float64(transition_offset)
    transition = _as_float64(transition_matrix)
    process_covariance = _as_float64(transition_covariance)
    d = _as_float64(observation_offset)
    observation = _as_float64(observation_matrix)
    measurement_covariance = _as_float64(observation_covariance)
    cross_covariance = _as_float64(state_measurement_cross_covariance)
    mean = _as_float64(initial_state_mean)
    covariance = _as_float64(initial_state_covariance)

    for name, tensor, rank in (
        ("observations", y, 2),
        ("transition_offset", c, 3),
        ("transition_matrix", transition, 4),
        ("transition_covariance", process_covariance, 4),
        ("observation_offset", d, 3),
        ("observation_matrix", observation, 4),
        ("observation_covariance", measurement_covariance, 4),
        ("state_measurement_cross_covariance", cross_covariance, 4),
        ("initial_state_mean", mean, 2),
        ("initial_state_covariance", covariance, 3),
    ):
        _require_rank(tensor, rank, name)

    batch_dim = tf.shape(mean)[0]
    state_dim = tf.shape(mean)[1]
    time_dim = tf.shape(y)[0]
    observation_dim = tf.shape(y)[1]
    for value, message in (
        (batch_dim, "batch dimension must be positive"),
        (state_dim, "state dimension must be positive"),
        (time_dim, "observations must contain time rows"),
        (observation_dim, "observation dimension must be positive"),
    ):
        tf.debugging.assert_positive(value, message=message)

    _require_shape(c, tf.stack([batch_dim, time_dim, state_dim]), "transition_offset")
    _require_shape(
        transition,
        tf.stack([batch_dim, time_dim, state_dim, state_dim]),
        "transition_matrix",
    )
    _require_shape(
        process_covariance,
        tf.stack([batch_dim, time_dim, state_dim, state_dim]),
        "transition_covariance",
    )
    _require_shape(d, tf.stack([batch_dim, time_dim, observation_dim]), "observation_offset")
    _require_shape(
        observation,
        tf.stack([batch_dim, time_dim, observation_dim, state_dim]),
        "observation_matrix",
    )
    _require_shape(
        measurement_covariance,
        tf.stack([batch_dim, time_dim, observation_dim, observation_dim]),
        "observation_covariance",
    )
    _require_shape(
        cross_covariance,
        tf.stack([batch_dim, time_dim, state_dim, observation_dim]),
        "state_measurement_cross_covariance",
    )
    _require_shape(
        covariance,
        tf.stack([batch_dim, state_dim, state_dim]),
        "initial_state_covariance",
    )

    mask = None
    if observation_mask is not None:
        mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
        _require_rank(mask, 2, "observation_mask")
        _require_shape(mask, tf.shape(y), "observation_mask")
    y = _canonicalize_observations(y, mask)

    for name, tensor in (
        ("transition_offset", c),
        ("transition_matrix", transition),
        ("transition_covariance", process_covariance),
        ("observation_offset", d),
        ("observation_matrix", observation),
        ("observation_covariance", measurement_covariance),
        ("state_measurement_cross_covariance", cross_covariance),
        ("initial_state_mean", mean),
        ("initial_state_covariance", covariance),
    ):
        _require_finite(tensor, name)

    process_covariance = _require_symmetric(
        process_covariance,
        "transition_covariance",
    )
    measurement_covariance = _require_symmetric(
        measurement_covariance,
        "observation_covariance",
    )
    covariance = _require_symmetric(covariance, "initial_state_covariance")
    return (
        y,
        c,
        transition,
        process_covariance,
        d,
        observation,
        measurement_covariance,
        cross_covariance,
        mean,
        covariance,
        mask,
    )


def _scalar_step(
    *,
    row: tf.Tensor,
    row_mask: tf.Tensor | None,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    cross_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    predicted_mean = transition_offset + tf.linalg.matvec(transition_matrix, mean)
    predicted_covariance = _symmetrize(
        transition_matrix @ covariance @ tf.transpose(transition_matrix)
        + transition_covariance
    )
    expected_observation = observation_offset + tf.linalg.matvec(
        observation_matrix,
        predicted_mean,
    )

    missing_count = tf.constant(0.0, dtype=tf.float64)
    if row_mask is not None:
        row_weight = tf.cast(row_mask, tf.float64)
        missing_weight = 1.0 - row_weight
        observation_matrix = observation_matrix * row_weight[:, tf.newaxis]
        cross_covariance = cross_covariance * row_weight[tf.newaxis, :]
        observation_covariance = (
            observation_covariance
            * row_weight[:, tf.newaxis]
            * row_weight[tf.newaxis, :]
            + tf.linalg.diag(missing_weight)
        )
        missing_count = tf.reduce_sum(missing_weight)
    else:
        row_weight = tf.ones_like(row)
    innovation = (row - expected_observation) * row_weight

    observation_cross = observation_matrix @ cross_covariance
    innovation_covariance = _symmetrize(
        observation_matrix
        @ predicted_covariance
        @ tf.transpose(observation_matrix)
        + observation_cross
        + tf.transpose(observation_cross)
        + observation_covariance
    )
    state_innovation_covariance = (
        predicted_covariance @ tf.transpose(observation_matrix) + cross_covariance
    )
    innovation_factor = tf.linalg.cholesky(innovation_covariance)
    _require_finite(innovation_factor, "innovation Cholesky factor")
    innovation_solve = tf.linalg.cholesky_solve(
        innovation_factor,
        innovation[:, tf.newaxis],
    )[:, 0]
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
    log_probability = -0.5 * (
        tf.cast(tf.shape(row)[0], tf.float64) * _LOG_2PI
        + log_det
        + tf.reduce_sum(innovation * innovation_solve)
    ) + 0.5 * missing_count * _LOG_2PI

    gain = tf.transpose(
        tf.linalg.cholesky_solve(
            innovation_factor,
            tf.transpose(state_innovation_covariance),
        )
    )
    filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
    filtered_covariance = _symmetrize(
        predicted_covariance - gain @ tf.transpose(state_innovation_covariance)
    )
    _require_finite(filtered_mean, "filtered mean")
    _require_finite(filtered_covariance, "filtered covariance")
    return filtered_mean, filtered_covariance, log_probability


def _batched_step(
    *,
    row: tf.Tensor,
    row_mask: tf.Tensor | None,
    mean: tf.Tensor,
    covariance: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    cross_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    predicted_mean = transition_offset + tf.einsum("bij,bj->bi", transition_matrix, mean)
    predicted_covariance = _symmetrize(
        tf.matmul(
            tf.matmul(transition_matrix, covariance),
            transition_matrix,
            transpose_b=True,
        )
        + transition_covariance
    )

    expected_observation = observation_offset + tf.einsum(
        "bmn,bn->bm",
        observation_matrix,
        predicted_mean,
    )
    missing_count = tf.constant(0.0, dtype=tf.float64)
    if row_mask is not None:
        row_weight = tf.cast(row_mask, tf.float64)
        missing_weight = 1.0 - row_weight
        observation_matrix = observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]
        cross_covariance = cross_covariance * row_weight[tf.newaxis, tf.newaxis, :]
        observation_covariance = (
            observation_covariance
            * row_weight[tf.newaxis, :, tf.newaxis]
            * row_weight[tf.newaxis, tf.newaxis, :]
            + tf.linalg.diag(missing_weight)[tf.newaxis, :, :]
        )
        missing_count = tf.reduce_sum(missing_weight)
    else:
        row_weight = tf.ones_like(row)
    innovation = (
        row[tf.newaxis, :] - expected_observation
    ) * row_weight[tf.newaxis, :]

    observation_cross = tf.matmul(observation_matrix, cross_covariance)
    innovation_covariance = _symmetrize(
        tf.matmul(
            tf.matmul(observation_matrix, predicted_covariance),
            observation_matrix,
            transpose_b=True,
        )
        + observation_cross
        + tf.linalg.matrix_transpose(observation_cross)
        + observation_covariance
    )
    state_innovation_covariance = (
        tf.matmul(predicted_covariance, observation_matrix, transpose_b=True)
        + cross_covariance
    )
    innovation_factor = tf.linalg.cholesky(innovation_covariance)
    _require_finite(innovation_factor, "innovation Cholesky factor")
    innovation_solve = tf.linalg.cholesky_solve(
        innovation_factor,
        innovation[:, :, tf.newaxis],
    )[:, :, 0]
    log_det = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(innovation_factor)),
        axis=-1,
    )
    log_probability = -0.5 * (
        tf.cast(tf.shape(row)[0], tf.float64) * _LOG_2PI
        + log_det
        + tf.einsum("bi,bi->b", innovation, innovation_solve)
    ) + 0.5 * missing_count * _LOG_2PI

    gain = tf.linalg.matrix_transpose(
        tf.linalg.cholesky_solve(
            innovation_factor,
            tf.linalg.matrix_transpose(state_innovation_covariance),
        )
    )
    filtered_mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
    filtered_covariance = _symmetrize(
        predicted_covariance
        - tf.matmul(gain, state_innovation_covariance, transpose_b=True)
    )
    _require_finite(filtered_mean, "filtered mean")
    _require_finite(filtered_covariance, "filtered covariance")
    return filtered_mean, filtered_covariance, log_probability


def _run_scalar(
    inputs: tuple[tf.Tensor, ...],
    *,
    collect_filtered: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    (
        y,
        c,
        transition,
        process_covariance,
        d,
        observation,
        measurement_covariance,
        cross_covariance,
        mean,
        covariance,
        mask,
    ) = inputs
    time_dim = tf.shape(y)[0]
    state_dim = tf.shape(mean)[0]
    if collect_filtered:
        means = tf.zeros(tf.stack([time_dim, state_dim]), dtype=tf.float64)
        covariances = tf.zeros(
            tf.stack([time_dim, state_dim, state_dim]),
            dtype=tf.float64,
        )
    else:
        means = tf.zeros(tf.stack([0, state_dim]), dtype=tf.float64)
        covariances = tf.zeros(tf.stack([0, state_dim, state_dim]), dtype=tf.float64)

    def condition(
        time_index: tf.Tensor,
        _mean: tf.Tensor,
        _covariance: tf.Tensor,
        _value: tf.Tensor,
        _means: tf.Tensor,
        _covariances: tf.Tensor,
    ) -> tf.Tensor:
        return time_index < time_dim

    def body(
        time_index: tf.Tensor,
        current_mean: tf.Tensor,
        current_covariance: tf.Tensor,
        value: tf.Tensor,
        stored_means: tf.Tensor,
        stored_covariances: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        next_mean, next_covariance, contribution = _scalar_step(
            row=y[time_index],
            row_mask=None if mask is None else mask[time_index],
            mean=current_mean,
            covariance=current_covariance,
            transition_offset=c[time_index],
            transition_matrix=transition[time_index],
            transition_covariance=process_covariance[time_index],
            observation_offset=d[time_index],
            observation_matrix=observation[time_index],
            observation_covariance=measurement_covariance[time_index],
            cross_covariance=cross_covariance[time_index],
        )
        if collect_filtered:
            index = tf.reshape(time_index, [1, 1])
            stored_means = tf.tensor_scatter_nd_update(
                stored_means,
                index,
                next_mean[tf.newaxis, :],
            )
            stored_covariances = tf.tensor_scatter_nd_update(
                stored_covariances,
                index,
                next_covariance[tf.newaxis, :, :],
            )
        return (
            time_index + 1,
            next_mean,
            next_covariance,
            value + contribution,
            stored_means,
            stored_covariances,
        )

    _, _, _, value, means, covariances = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, dtype=tf.int32),
            mean,
            covariance,
            tf.constant(0.0, dtype=tf.float64),
            means,
            covariances,
        ),
        parallel_iterations=1,
        maximum_iterations=time_dim,
    )
    return value, means, covariances


def _run_batched(
    inputs: tuple[tf.Tensor, ...],
    *,
    collect_filtered: bool,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    (
        y,
        c,
        transition,
        process_covariance,
        d,
        observation,
        measurement_covariance,
        cross_covariance,
        mean,
        covariance,
        mask,
    ) = inputs
    batch_dim = tf.shape(mean)[0]
    time_dim = tf.shape(y)[0]
    state_dim = tf.shape(mean)[1]
    if collect_filtered:
        means = tf.zeros(tf.stack([time_dim, batch_dim, state_dim]), dtype=tf.float64)
        covariances = tf.zeros(
            tf.stack([time_dim, batch_dim, state_dim, state_dim]),
            dtype=tf.float64,
        )
    else:
        means = tf.zeros(tf.stack([0, batch_dim, state_dim]), dtype=tf.float64)
        covariances = tf.zeros(
            tf.stack([0, batch_dim, state_dim, state_dim]),
            dtype=tf.float64,
        )

    def condition(
        time_index: tf.Tensor,
        _mean: tf.Tensor,
        _covariance: tf.Tensor,
        _value: tf.Tensor,
        _means: tf.Tensor,
        _covariances: tf.Tensor,
    ) -> tf.Tensor:
        return time_index < time_dim

    def body(
        time_index: tf.Tensor,
        current_mean: tf.Tensor,
        current_covariance: tf.Tensor,
        value: tf.Tensor,
        stored_means: tf.Tensor,
        stored_covariances: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        next_mean, next_covariance, contribution = _batched_step(
            row=y[time_index],
            row_mask=None if mask is None else mask[time_index],
            mean=current_mean,
            covariance=current_covariance,
            transition_offset=c[:, time_index],
            transition_matrix=transition[:, time_index],
            transition_covariance=process_covariance[:, time_index],
            observation_offset=d[:, time_index],
            observation_matrix=observation[:, time_index],
            observation_covariance=measurement_covariance[:, time_index],
            cross_covariance=cross_covariance[:, time_index],
        )
        if collect_filtered:
            index = tf.reshape(time_index, [1, 1])
            stored_means = tf.tensor_scatter_nd_update(
                stored_means,
                index,
                next_mean[tf.newaxis, :, :],
            )
            stored_covariances = tf.tensor_scatter_nd_update(
                stored_covariances,
                index,
                next_covariance[tf.newaxis, :, :, :],
            )
        return (
            time_index + 1,
            next_mean,
            next_covariance,
            value + contribution,
            stored_means,
            stored_covariances,
        )

    _, _, _, value, means, covariances = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, dtype=tf.int32),
            mean,
            covariance,
            tf.zeros([batch_dim], dtype=tf.float64),
            means,
            covariances,
        ),
        parallel_iterations=1,
        maximum_iterations=time_dim,
    )
    return (
        value,
        tf.transpose(means, perm=[1, 0, 2]),
        tf.transpose(covariances, perm=[1, 0, 2, 3]),
    )


@tf.function(reduce_retracing=True)
def tf_correlated_kalman_filter(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return scalar likelihood and filtered moments with shapes ``[]/[T,n]/[T,n,n]``."""

    return _run_scalar(
        _prepare_scalar_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=None,
        ),
        collect_filtered=True,
    )


@tf.function(reduce_retracing=True)
def tf_masked_correlated_kalman_filter(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return masked scalar likelihood and filtered moments using dummy rows."""

    return _run_scalar(
        _prepare_scalar_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=observation_mask,
        ),
        collect_filtered=True,
    )


@tf.function(reduce_retracing=True)
def tf_correlated_kalman_log_likelihood(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> tf.Tensor:
    """Return the scalar correlated-innovation prediction-error likelihood."""

    value, _, _ = _run_scalar(
        _prepare_scalar_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=None,
        ),
        collect_filtered=False,
    )
    return value


@tf.function(reduce_retracing=True)
def tf_masked_correlated_kalman_log_likelihood(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
) -> tf.Tensor:
    """Return the static-shape masked correlated likelihood."""

    value, _, _ = _run_scalar(
        _prepare_scalar_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=observation_mask,
        ),
        collect_filtered=False,
    )
    return value


@tf.function(reduce_retracing=True)
def tf_correlated_kalman_filter_batched_time_varying(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return batched likelihood and moments with shapes ``[B]/[B,T,n]/[B,T,n,n]``."""

    return _run_batched(
        _prepare_batched_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=None,
        ),
        collect_filtered=True,
    )


@tf.function(reduce_retracing=True)
def tf_masked_correlated_kalman_filter_batched_time_varying(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return batched masked likelihood and filtered moments."""

    return _run_batched(
        _prepare_batched_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=observation_mask,
        ),
        collect_filtered=True,
    )


@tf.function(reduce_retracing=True)
def tf_correlated_kalman_log_likelihood_batched_time_varying(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
) -> tf.Tensor:
    """Return the batch-native time-varying correlated likelihood ``[B]``."""

    value, _, _ = _run_batched(
        _prepare_batched_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=None,
        ),
        collect_filtered=False,
    )
    return value


@tf.function(reduce_retracing=True)
def tf_masked_correlated_kalman_log_likelihood_batched_time_varying(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    state_measurement_cross_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
) -> tf.Tensor:
    """Return the batch-native static-shape masked correlated likelihood ``[B]``."""

    value, _, _ = _run_batched(
        _prepare_batched_inputs(
            observations=observations,
            transition_offset=transition_offset,
            transition_matrix=transition_matrix,
            transition_covariance=transition_covariance,
            observation_offset=observation_offset,
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
            state_measurement_cross_covariance=state_measurement_cross_covariance,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            observation_mask=observation_mask,
        ),
        collect_filtered=False,
    )
    return value
