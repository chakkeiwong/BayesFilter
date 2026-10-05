"""Reviewed covariance-form analytical batch Kalman score kernel.

This production-facing module exposes only the fixed-shape value/score
recursion. It assumes dense observations, time-invariant model tensors, and a
shared observation series for all batch rows. The covariance and score
recursions are written explicitly and never invoke TensorFlow autodiff. Only
innovation covariance matrices are Cholesky-factorized, so positive-semidefinite
initial and process covariance blocks are supported when the resulting
innovations are positive definite. Callers must validate proposals before
entering this kernel and retain the Metropolis correction at the sampler
boundary.
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


def _check_last_dim(tensor: tf.Tensor, expected: int, name: str) -> None:
    if tensor.shape[-1] is not None and tensor.shape[-1] != expected:
        raise ValueError(f"{name} has incompatible trailing shape")
    tf.debugging.assert_equal(
        tf.shape(tensor)[-1],
        expected,
        message=f"{name} has incompatible trailing dimension",
    )


def _check_square_batch_matrix(tensor: tf.Tensor, dim: tf.Tensor, name: str) -> None:
    _check_rank(tensor, 3, name)
    if not tensor.shape[-2:].is_compatible_with((dim, dim)):
        raise ValueError(f"{name} has incompatible matrix shape")
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
    batch_dim = initial_state_mean.shape[0]
    state_dim = initial_state_mean.shape[1]
    obs_dim = observations.shape[1]
    if batch_dim is None or state_dim is None or obs_dim is None:
        raise ValueError("batch, state and observation shape must be fixed before XLA tracing")

    for name, tensor in (
        ("transition_offset", transition_offset),
        ("observation_offset", observation_offset),
    ):
        _check_rank(tensor, 2, name)
        if tensor.shape[0] is not None and tensor.shape[0] != batch_dim:
            raise ValueError(f"{name} has incompatible batch shape")
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
        if tensor.shape[0] is not None and tensor.shape[0] != batch_dim:
            raise ValueError(f"{name} has incompatible batch shape")
        tf.debugging.assert_equal(
            tf.shape(tensor)[0],
            batch_dim,
            message=f"{name} batch dimension must match initial_state_mean",
        )

    _check_rank(observation_matrix, 3, "observation_matrix")
    if not observation_matrix.shape.is_compatible_with((batch_dim, obs_dim, state_dim)):
        raise ValueError("observation_matrix has incompatible shape")
    if observation_covariance.shape[0] is not None and observation_covariance.shape[0] != batch_dim:
        raise ValueError("observation_covariance has incompatible batch shape")
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
    parameter_dim = d_initial_state_mean.shape[1]
    if parameter_dim is None:
        raise ValueError("parameter shape must be fixed before XLA tracing")
    expected_vector_state = (batch_dim, parameter_dim, state_dim)
    expected_matrix_state = (batch_dim, parameter_dim, state_dim, state_dim)
    expected_vector_obs = (batch_dim, parameter_dim, obs_dim)
    expected_observation_matrix = (batch_dim, parameter_dim, obs_dim, state_dim)
    expected_matrix_obs = (batch_dim, parameter_dim, obs_dim, obs_dim)
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
        if not tensor.shape.is_compatible_with(expected):
            raise ValueError(f"{name} has incompatible batched derivative shape")
        tf.debugging.assert_equal(
            tf.shape(tensor),
            expected,
            message=f"{name} has incompatible batched derivative shape",
        )
    return parameter_dim


def _batched_cholesky_solve(chol: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky_solve(chol, rhs)


@tf.function(jit_compile=True, reduce_retracing=True)
def tf_batched_covariance_kalman_value_and_score(
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
    """Return dense batch-native likelihood and explicit first-order score.

    The leading batch axis indexes independent proposals and the parameter
    axis on each derivative tensor carries the manually supplied derivatives.
    No row is dispatched through a Python callback; time is the only
    sequential axis.
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

    def time_step(t, covariance, d_covariance, d_mean, log_likelihood, mean, score):
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
        return t + 1, covariance, d_covariance, d_mean, log_likelihood, mean, score

    _, covariance, d_covariance, d_mean, log_likelihood, mean, score = tf.while_loop(
        lambda t, covariance, d_covariance, d_mean, log_likelihood, mean, score: t < tf.shape(y)[0],
        time_step, (tf.constant(0, tf.int32), covariance, d_covariance, d_mean, log_likelihood, mean, score), parallel_iterations=1,
        maximum_iterations=tf.shape(y)[0],
    )

    return log_likelihood, score


__all__ = ["tf_batched_covariance_kalman_value_and_score"]
