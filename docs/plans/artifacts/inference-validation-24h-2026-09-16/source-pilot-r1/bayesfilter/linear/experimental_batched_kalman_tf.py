"""Experimental batch-native TensorFlow Kalman value and score kernels.

This module is intentionally not exported from ``bayesfilter.linear`` while the
batch-over-parameters contract is being tested.  It assumes dense observations,
time-invariant model tensors, and a shared observation series for all batch
rows.
"""

from __future__ import annotations

import math

import tensorflow as tf


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _as_observation_matrix(observations: tf.Tensor) -> tf.Tensor:
    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must be one- or two-dimensional")
    return y


def _to_tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _check_rank(tensor: tf.Tensor, rank: int, name: str) -> None:
    if tensor.shape.rank is not None and tensor.shape.rank != rank:
        raise ValueError(f"{name} must have rank {rank}")
    tf.debugging.assert_rank(tensor, rank, message=f"{name} must have rank {rank}")


def _check_last_dim(tensor: tf.Tensor, expected: tf.Tensor, name: str) -> None:
    tf.debugging.assert_equal(
        tf.shape(tensor)[-1],
        expected,
        message=f"{name} has incompatible trailing dimension",
    )


def _check_square_batch_matrix(tensor: tf.Tensor, dim: tf.Tensor, name: str) -> None:
    _check_rank(tensor, 3, name)
    tf.debugging.assert_equal(
        tf.shape(tensor)[-2:],
        tf.stack([dim, dim]),
        message=f"{name} must have shape [B, dim, dim]",
    )


def _check_batched_value_shapes(
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
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    _check_rank(observations, 2, "observations")
    _check_rank(initial_state_mean, 2, "initial_state_mean")
    batch_dim = tf.shape(initial_state_mean)[0]
    state_dim = tf.shape(initial_state_mean)[1]
    obs_dim = tf.shape(observations)[1]

    for name, tensor in (
        ("transition_offset", transition_offset),
        ("observation_offset", observation_offset),
    ):
        _check_rank(tensor, 2, name)
        tf.debugging.assert_equal(
            tf.shape(tensor)[0],
            batch_dim,
            message=f"{name} batch dimension must match initial_state_mean",
        )
    _check_last_dim(transition_offset, state_dim, "transition_offset")
    _check_last_dim(observation_offset, obs_dim, "observation_offset")

    for name, tensor in (
        ("transition_matrix", transition_matrix),
        ("transition_covariance", transition_covariance),
        ("initial_state_covariance", initial_state_covariance),
    ):
        _check_square_batch_matrix(tensor, state_dim, name)
        tf.debugging.assert_equal(
            tf.shape(tensor)[0],
            batch_dim,
            message=f"{name} batch dimension must match initial_state_mean",
        )

    _check_rank(observation_matrix, 3, "observation_matrix")
    tf.debugging.assert_equal(
        tf.shape(observation_matrix),
        tf.stack([batch_dim, obs_dim, state_dim]),
        message="observation_matrix must have shape [B, observation_dim, state_dim]",
    )
    _check_square_batch_matrix(observation_covariance, obs_dim, "observation_covariance")
    tf.debugging.assert_equal(
        tf.shape(observation_covariance)[0],
        batch_dim,
        message="observation_covariance batch dimension must match initial_state_mean",
    )
    return batch_dim, state_dim, obs_dim


def _check_batched_derivative_shapes(
    *,
    batch_dim: tf.Tensor,
    state_dim: tf.Tensor,
    obs_dim: tf.Tensor,
    d_initial_state_mean: tf.Tensor,
    d_initial_state_covariance: tf.Tensor,
    d_transition_offset: tf.Tensor,
    d_transition_matrix: tf.Tensor,
    d_transition_covariance: tf.Tensor,
    d_observation_offset: tf.Tensor,
    d_observation_matrix: tf.Tensor,
    d_observation_covariance: tf.Tensor,
) -> tf.Tensor:
    _check_rank(d_initial_state_mean, 3, "d_initial_state_mean")
    parameter_dim = tf.shape(d_initial_state_mean)[1]
    expected_vector_state = tf.stack([batch_dim, parameter_dim, state_dim])
    expected_matrix_state = tf.stack([batch_dim, parameter_dim, state_dim, state_dim])
    expected_vector_obs = tf.stack([batch_dim, parameter_dim, obs_dim])
    expected_observation_matrix = tf.stack(
        [batch_dim, parameter_dim, obs_dim, state_dim]
    )
    expected_matrix_obs = tf.stack([batch_dim, parameter_dim, obs_dim, obs_dim])
    expected_shapes = (
        ("d_initial_state_mean", d_initial_state_mean, expected_vector_state),
        ("d_initial_state_covariance", d_initial_state_covariance, expected_matrix_state),
        ("d_transition_offset", d_transition_offset, expected_vector_state),
        ("d_transition_matrix", d_transition_matrix, expected_matrix_state),
        ("d_transition_covariance", d_transition_covariance, expected_matrix_state),
        ("d_observation_offset", d_observation_offset, expected_vector_obs),
        ("d_observation_matrix", d_observation_matrix, expected_observation_matrix),
        ("d_observation_covariance", d_observation_covariance, expected_matrix_obs),
    )
    for name, tensor, expected in expected_shapes:
        tf.debugging.assert_equal(
            tf.shape(tensor),
            expected,
            message=f"{name} has incompatible batched derivative shape",
        )
    return parameter_dim


def _batched_cholesky_solve(chol: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky_solve(chol, rhs)


@tf.function(reduce_retracing=True)
def tf_batched_kalman_filter(
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
    """Dense batch-native Kalman filter over independent parameter rows.

    The leading batch axis indexes independent model parameter proposals.  The
    observation series is shared across all batch rows and time remains
    sequential.
    """

    y = _as_observation_matrix(observations)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _symmetrize(_to_tensor(transition_covariance))
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _symmetrize(_to_tensor(observation_covariance))
    mean = _to_tensor(initial_state_mean)
    covariance = _symmetrize(_to_tensor(initial_state_covariance))
    jitter_tensor = tf.convert_to_tensor(jitter, dtype=tf.float64)

    batch_dim, state_dim, obs_dim = _check_batched_value_shapes(
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
    state_identity = tf.eye(state_dim, dtype=tf.float64)[tf.newaxis, :, :]
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)[tf.newaxis, :, :]
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.zeros([batch_dim], dtype=tf.float64)
    means = tf.TensorArray(tf.float64, size=tf.shape(y)[0])
    covariances = tf.TensorArray(tf.float64, size=tf.shape(y)[0])

    for t in tf.range(tf.shape(y)[0]):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[
                (mean, tf.TensorShape([None, None])),
                (covariance, tf.TensorShape([None, None, None])),
                (log_likelihood, tf.TensorShape([None])),
            ]
        )
        predicted_mean = transition_offset + tf.einsum(
            "bij,bj->bi",
            transition_matrix,
            mean,
        )
        predicted_covariance = _symmetrize(
            tf.matmul(
                tf.matmul(transition_matrix, covariance),
                transition_matrix,
                transpose_b=True,
            )
            + transition_covariance
        )
        expected_observation = observation_offset + tf.einsum(
            "bij,bj->bi",
            observation_matrix,
            predicted_mean,
        )
        innovation = y[t][tf.newaxis, :] - expected_observation
        innovation_covariance = _symmetrize(
            tf.matmul(
                tf.matmul(observation_matrix, predicted_covariance),
                observation_matrix,
                transpose_b=True,
            )
            + observation_covariance
            + jitter_tensor * obs_identity
        )
        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        innovation_solve = _batched_cholesky_solve(
            innovation_factor,
            innovation[:, :, tf.newaxis],
        )[:, :, 0]
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)),
            axis=-1,
        )
        mahalanobis = tf.einsum("bi,bi->b", innovation, innovation_solve)
        log_likelihood = log_likelihood - 0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )

        innovation_precision = _batched_cholesky_solve(
            innovation_factor,
            tf.tile(obs_identity, [batch_dim, 1, 1]),
        )
        gain = tf.matmul(
            tf.matmul(predicted_covariance, observation_matrix, transpose_b=True),
            innovation_precision,
        )
        left = state_identity - tf.matmul(gain, observation_matrix)
        observation_update_covariance = observation_covariance + jitter_tensor * obs_identity
        mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
        covariance = _symmetrize(
            tf.matmul(
                tf.matmul(left, predicted_covariance),
                left,
                transpose_b=True,
            )
            + tf.matmul(
                tf.matmul(gain, observation_update_covariance),
                gain,
                transpose_b=True,
            )
        )
        if return_filtered:
            means = means.write(t, mean)
            covariances = covariances.write(t, covariance)

    filtered_means = means.stack() if return_filtered else None
    filtered_covariances = covariances.stack() if return_filtered else None
    return log_likelihood, filtered_means, filtered_covariances


@tf.function(reduce_retracing=True)
def tf_batched_kalman_value_and_score(
    observations: tf.Tensor,
    transition_offset: tf.Tensor,
    transition_matrix: tf.Tensor,
    transition_covariance: tf.Tensor,
    observation_offset: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_state_mean: tf.Tensor,
    initial_state_covariance: tf.Tensor,
    d_initial_state_mean: tf.Tensor,
    d_initial_state_covariance: tf.Tensor,
    d_transition_offset: tf.Tensor,
    d_transition_matrix: tf.Tensor,
    d_transition_covariance: tf.Tensor,
    d_observation_offset: tf.Tensor,
    d_observation_matrix: tf.Tensor,
    d_observation_covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
    jitter_updates_filtered_covariance: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Dense batch-native Kalman likelihood and analytic first-order score."""

    y = _as_observation_matrix(observations)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _symmetrize(_to_tensor(transition_covariance))
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _symmetrize(_to_tensor(observation_covariance))
    mean = _to_tensor(initial_state_mean)
    covariance = _symmetrize(_to_tensor(initial_state_covariance))
    d_mean = _to_tensor(d_initial_state_mean)
    d_covariance = _symmetrize(_to_tensor(d_initial_state_covariance))
    d_transition_offset = _to_tensor(d_transition_offset)
    d_transition_matrix = _to_tensor(d_transition_matrix)
    d_transition_covariance = _symmetrize(_to_tensor(d_transition_covariance))
    d_observation_offset = _to_tensor(d_observation_offset)
    d_observation_matrix = _to_tensor(d_observation_matrix)
    d_observation_covariance = _symmetrize(_to_tensor(d_observation_covariance))
    jitter_tensor = tf.convert_to_tensor(jitter, dtype=tf.float64)

    batch_dim, state_dim, obs_dim = _check_batched_value_shapes(
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
    parameter_dim = _check_batched_derivative_shapes(
        batch_dim=batch_dim,
        state_dim=state_dim,
        obs_dim=obs_dim,
        d_initial_state_mean=d_mean,
        d_initial_state_covariance=d_covariance,
        d_transition_offset=d_transition_offset,
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=d_observation_offset,
        d_observation_matrix=d_observation_matrix,
        d_observation_covariance=d_observation_covariance,
    )

    state_identity = tf.eye(state_dim, dtype=tf.float64)[tf.newaxis, :, :]
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)[tf.newaxis, :, :]
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    log_likelihood = tf.zeros([batch_dim], dtype=tf.float64)
    score = tf.zeros([batch_dim, parameter_dim], dtype=tf.float64)

    for t in tf.range(tf.shape(y)[0]):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[
                (mean, tf.TensorShape([None, None])),
                (d_mean, tf.TensorShape([None, None, None])),
                (covariance, tf.TensorShape([None, None, None])),
                (d_covariance, tf.TensorShape([None, None, None, None])),
                (log_likelihood, tf.TensorShape([None])),
                (score, tf.TensorShape([None, None])),
            ]
        )
        predicted_mean = transition_offset + tf.einsum(
            "bij,bj->bi",
            transition_matrix,
            mean,
        )
        d_predicted_mean = (
            d_transition_offset
            + tf.einsum("bpij,bj->bpi", d_transition_matrix, mean)
            + tf.einsum("bij,bpj->bpi", transition_matrix, d_mean)
        )
        predicted_covariance = _symmetrize(
            tf.matmul(
                tf.matmul(transition_matrix, covariance),
                transition_matrix,
                transpose_b=True,
            )
            + transition_covariance
        )
        d_predicted_covariance = _symmetrize(
            tf.einsum(
                "bpij,bjk,blk->bpil",
                d_transition_matrix,
                covariance,
                transition_matrix,
            )
            + tf.einsum(
                "bij,bpjk,blk->bpil",
                transition_matrix,
                d_covariance,
                transition_matrix,
            )
            + tf.einsum(
                "bij,bjk,bplk->bpil",
                transition_matrix,
                covariance,
                d_transition_matrix,
            )
            + d_transition_covariance
        )

        expected_observation = observation_offset + tf.einsum(
            "bij,bj->bi",
            observation_matrix,
            predicted_mean,
        )
        innovation = y[t][tf.newaxis, :] - expected_observation
        d_innovation = (
            -d_observation_offset
            - tf.einsum("bpij,bj->bpi", d_observation_matrix, predicted_mean)
            - tf.einsum("bij,bpj->bpi", observation_matrix, d_predicted_mean)
        )
        innovation_covariance = _symmetrize(
            tf.matmul(
                tf.matmul(observation_matrix, predicted_covariance),
                observation_matrix,
                transpose_b=True,
            )
            + observation_covariance
            + jitter_tensor * obs_identity
        )
        d_innovation_covariance = _symmetrize(
            tf.einsum(
                "bpmi,bij,blj->bpml",
                d_observation_matrix,
                predicted_covariance,
                observation_matrix,
            )
            + tf.einsum(
                "bmi,bpij,blj->bpml",
                observation_matrix,
                d_predicted_covariance,
                observation_matrix,
            )
            + tf.einsum(
                "bmi,bij,bplj->bpml",
                observation_matrix,
                predicted_covariance,
                d_observation_matrix,
            )
            + d_observation_covariance
        )

        innovation_factor = tf.linalg.cholesky(innovation_covariance)
        innovation_solve = _batched_cholesky_solve(
            innovation_factor,
            innovation[:, :, tf.newaxis],
        )[:, :, 0]
        innovation_precision = _batched_cholesky_solve(
            innovation_factor,
            tf.tile(obs_identity, [batch_dim, 1, 1]),
        )
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor)),
            axis=-1,
        )
        mahalanobis = tf.einsum("bi,bi->b", innovation, innovation_solve)
        log_likelihood = log_likelihood - 0.5 * (
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )

        trace_terms = tf.einsum(
            "bij,bpji->bp",
            innovation_precision,
            d_innovation_covariance,
        )
        innovation_derivative_terms = tf.einsum(
            "bpi,bi->bp",
            d_innovation,
            innovation_solve,
        )
        quadratic_terms = tf.einsum(
            "bi,bpij,bj->bp",
            innovation_solve,
            d_innovation_covariance,
            innovation_solve,
        )
        score = score - 0.5 * (
            trace_terms + 2.0 * innovation_derivative_terms - quadratic_terms
        )

        d_innovation_precision = -tf.einsum(
            "bij,bpjk,bkl->bpil",
            innovation_precision,
            d_innovation_covariance,
            innovation_precision,
        )
        gain_rhs = tf.matmul(
            predicted_covariance,
            observation_matrix,
            transpose_b=True,
        )
        gain = tf.matmul(gain_rhs, innovation_precision)
        d_gain = (
            tf.einsum(
                "bpij,bmj,bmk->bpik",
                d_predicted_covariance,
                observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "bij,bpmj,bmk->bpik",
                predicted_covariance,
                d_observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "bij,bmj,bpmk->bpik",
                predicted_covariance,
                observation_matrix,
                d_innovation_precision,
            )
        )

        joseph_left = state_identity - tf.matmul(gain, observation_matrix)
        d_joseph_left = -(
            tf.einsum("bpik,bkj->bpij", d_gain, observation_matrix)
            + tf.einsum("bik,bpkj->bpij", gain, d_observation_matrix)
        )
        observation_update_covariance = (
            observation_covariance + jitter_tensor * obs_identity
            if jitter_updates_filtered_covariance
            else observation_covariance
        )
        d_observation_update_covariance = d_observation_covariance
        mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
        d_mean = (
            d_predicted_mean
            + tf.einsum("bpij,bj->bpi", d_gain, innovation)
            + tf.einsum("bij,bpj->bpi", gain, d_innovation)
        )
        covariance = _symmetrize(
            tf.matmul(
                tf.matmul(joseph_left, predicted_covariance),
                joseph_left,
                transpose_b=True,
            )
            + tf.matmul(
                tf.matmul(gain, observation_update_covariance),
                gain,
                transpose_b=True,
            )
        )
        d_covariance = _symmetrize(
            tf.einsum(
                "bpia,bac,bjc->bpij",
                d_joseph_left,
                predicted_covariance,
                joseph_left,
            )
            + tf.einsum(
                "bia,bpac,bjc->bpij",
                joseph_left,
                d_predicted_covariance,
                joseph_left,
            )
            + tf.einsum(
                "bia,bac,bpjc->bpij",
                joseph_left,
                predicted_covariance,
                d_joseph_left,
            )
            + tf.einsum(
                "bpia,bac,bjc->bpij",
                d_gain,
                observation_update_covariance,
                gain,
            )
            + tf.einsum(
                "bia,bpac,bjc->bpij",
                gain,
                d_observation_update_covariance,
                gain,
            )
            + tf.einsum(
                "bia,bac,bpjc->bpij",
                gain,
                observation_update_covariance,
                d_gain,
            )
        )

    return log_likelihood, score
