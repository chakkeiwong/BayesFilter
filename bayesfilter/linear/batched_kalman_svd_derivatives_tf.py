"""Batch-native SVD/eigh graph-status Kalman value and score kernel."""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import tensorflow as tf

from bayesfilter.linear.kalman_svd_derivatives_tf import (
    SVD_LINEAR_SCORE_STATUS_BLOCKED_ACTIVE_FLOOR,
    SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT,
    SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED,
)


class BatchedSVDLinearGaussianScoreResult(NamedTuple):
    """Per-row likelihood, score, and graph-status telemetry."""

    log_likelihood: tf.Tensor
    score: tf.Tensor
    status_code: tf.Tensor
    valid_pre_regularized_score: tf.Tensor
    floor_count_value: tf.Tensor
    min_innovation_eigenvalue: tf.Tensor
    innovation_condition_estimate: tf.Tensor
    psd_projection_residual: tf.Tensor
    invalid_eigensolver_input: tf.Tensor


@tf.function(reduce_retracing=True, jit_compile=True)
def tf_batched_svd_linear_gaussian_score_first_order_graph_status(
    observations: Any,
    *,
    transition_offset: Any,
    transition_matrix: Any,
    transition_covariance: Any,
    observation_offset: Any,
    observation_matrix: Any,
    observation_covariance: Any,
    initial_state_mean: Any,
    initial_state_covariance: Any,
    d_initial_state_mean: Any,
    d_initial_state_covariance: Any,
    d_transition_offset: Any,
    d_transition_matrix: Any,
    d_transition_covariance: Any,
    d_observation_offset: Any,
    d_observation_matrix: Any,
    d_observation_covariance: Any,
    jitter: Any = 0.0,
    singular_floor: Any = 1.0e-12,
    eigendecomposition: Any = None,
) -> BatchedSVDLinearGaussianScoreResult:
    """Evaluate independent LGSSM rows with one time-axis TensorFlow loop."""

    y = tf.convert_to_tensor(observations, tf.float64)
    if y.shape.rank == 1:
        y = y[:, tf.newaxis]
    if y.shape.rank != 2:
        raise ValueError("observations must have rank 1 or 2")
    transition_offset = _tensor(transition_offset)
    transition_matrix = _tensor(transition_matrix)
    transition_covariance = _symmetrize(_tensor(transition_covariance))
    observation_offset = _tensor(observation_offset)
    observation_matrix = _tensor(observation_matrix)
    observation_covariance = _symmetrize(_tensor(observation_covariance))
    mean0 = _tensor(initial_state_mean)
    covariance0 = _symmetrize(_tensor(initial_state_covariance))
    dmean0 = _tensor(d_initial_state_mean)
    dcovariance0 = _symmetrize(_tensor(d_initial_state_covariance))
    d_transition_offset = _tensor(d_transition_offset)
    d_transition_matrix = _tensor(d_transition_matrix)
    d_transition_covariance = _symmetrize(_tensor(d_transition_covariance))
    d_observation_offset = _tensor(d_observation_offset)
    d_observation_matrix = _tensor(d_observation_matrix)
    d_observation_covariance = _symmetrize(_tensor(d_observation_covariance))
    jitter = tf.convert_to_tensor(jitter, tf.float64)
    singular_floor = tf.convert_to_tensor(singular_floor, tf.float64)

    batch_size, parameter_dim, state_dim, observation_dim = _validate_shapes(
        y,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        mean0,
        covariance0,
        dmean0,
        dcovariance0,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
    )
    identity_state = tf.eye(
        state_dim, batch_shape=(batch_size,), dtype=tf.float64
    )
    identity_observation = tf.eye(
        observation_dim, batch_shape=(batch_size,), dtype=tf.float64
    )
    two_pi = tf.constant(2.0 * math.pi, tf.float64)

    def cond(t, *_state):
        return t < tf.shape(y)[0]

    def body(
        t,
        mean,
        covariance,
        dmean,
        dcovariance,
        log_likelihood,
        score,
        max_floor_count,
        max_projection_residual,
        min_innovation_eigenvalue,
        max_innovation_condition,
        invalid_eigensolver_input,
    ):
        predicted_mean = transition_offset + tf.einsum(
            "bij,bj->bi", transition_matrix, mean
        )
        d_predicted_mean = (
            d_transition_offset
            + tf.einsum("bpij,bj->bpi", d_transition_matrix, mean)
            + tf.einsum("bij,bpj->bpi", transition_matrix, dmean)
        )
        predicted_covariance = _symmetrize(
            transition_matrix
            @ covariance
            @ tf.linalg.matrix_transpose(transition_matrix)
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
                dcovariance,
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
            "bij,bj->bi", observation_matrix, predicted_mean
        )
        innovation = y[t][tf.newaxis, :] - expected_observation
        d_innovation = (
            -d_observation_offset
            - tf.einsum(
                "bpij,bj->bpi", d_observation_matrix, predicted_mean
            )
            - tf.einsum(
                "bij,bpj->bpi", observation_matrix, d_predicted_mean
            )
        )
        innovation_covariance = _symmetrize(
            observation_matrix
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(observation_matrix)
            + observation_covariance
            + jitter * identity_observation
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

        row_input_valid = tf.logical_and(
            tf.reduce_all(tf.math.is_finite(innovation_covariance), axis=(1, 2)),
            tf.math.is_finite(singular_floor),
        )
        benign_scale = tf.maximum(
            tf.abs(tf.where(tf.math.is_finite(singular_floor), singular_floor, 1.0)),
            tf.constant(1.0, tf.float64),
        )
        safe_covariance = tf.where(
            row_input_valid[:, tf.newaxis, tf.newaxis],
            innovation_covariance,
            benign_scale * identity_observation,
        )
        eigh = tf.linalg.eigh if eigendecomposition is None else eigendecomposition
        eigenvalues, eigenvectors = eigh(safe_covariance)
        safe_floor = tf.where(
            tf.math.is_finite(singular_floor), singular_floor, tf.constant(1.0, tf.float64)
        )
        floored = tf.maximum(eigenvalues, safe_floor)
        implemented_covariance = _symmetrize(
            eigenvectors
            @ tf.linalg.diag(floored)
            @ tf.linalg.matrix_transpose(eigenvectors)
        )
        step_projection_residual = tf.linalg.norm(
            implemented_covariance - innovation_covariance, axis=(1, 2)
        )
        step_projection_residual = tf.where(
            row_input_valid,
            step_projection_residual,
            tf.fill(tf.shape(row_input_valid), tf.constant(float("nan"), tf.float64)),
        )
        innovation_solve = _eigh_solve(eigenvectors, floored, innovation)
        innovation_precision = _eigh_solve(
            eigenvectors, floored, identity_observation
        )
        log_det = tf.reduce_sum(tf.math.log(floored), axis=1)
        mahalanobis = tf.einsum("bi,bi->b", innovation, innovation_solve)
        log_likelihood = log_likelihood - 0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )

        trace_terms = tf.einsum(
            "bij,bpji->bp", innovation_precision, d_innovation_covariance
        )
        innovation_derivative_terms = tf.einsum(
            "bpi,bi->bp", d_innovation, innovation_solve
        )
        quadratic_terms = tf.einsum(
            "bi,bpij,bj->bp",
            innovation_solve,
            d_innovation_covariance,
            innovation_solve,
        )
        score = score - 0.5 * (
            trace_terms
            + 2.0 * innovation_derivative_terms
            - quadratic_terms
        )

        d_innovation_precision = -tf.einsum(
            "bij,bpjk,bkl->bpil",
            innovation_precision,
            d_innovation_covariance,
            innovation_precision,
        )
        gain = (
            predicted_covariance
            @ tf.linalg.matrix_transpose(observation_matrix)
            @ innovation_precision
        )
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
        joseph_left = identity_state - gain @ observation_matrix
        d_joseph_left = -(
            tf.einsum("bpik,bkj->bpij", d_gain, observation_matrix)
            + tf.einsum("bik,bpkj->bpij", gain, d_observation_matrix)
        )
        observation_noise = observation_covariance + jitter * identity_observation
        mean = predicted_mean + tf.einsum("bij,bj->bi", gain, innovation)
        dmean = (
            d_predicted_mean
            + tf.einsum("bpij,bj->bpi", d_gain, innovation)
            + tf.einsum("bij,bpj->bpi", gain, d_innovation)
        )
        covariance = _symmetrize(
            joseph_left
            @ predicted_covariance
            @ tf.linalg.matrix_transpose(joseph_left)
            + gain @ observation_noise @ tf.linalg.matrix_transpose(gain)
        )
        dcovariance = _symmetrize(
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
                "bpia,bac,bjc->bpij", d_gain, observation_noise, gain
            )
            + tf.einsum(
                "bia,bpac,bjc->bpij", gain, d_observation_covariance, gain
            )
            + tf.einsum(
                "bia,bac,bpjc->bpij", gain, observation_noise, d_gain
            )
        )

        step_floor_count = tf.reduce_sum(
            tf.cast(eigenvalues <= safe_floor, tf.int32), axis=1
        )
        max_floor_count = tf.maximum(max_floor_count, step_floor_count)
        max_projection_residual = tf.maximum(
            max_projection_residual, step_projection_residual
        )
        step_minimum = tf.reduce_min(eigenvalues, axis=1)
        min_innovation_eigenvalue = tf.minimum(
            min_innovation_eigenvalue, step_minimum
        )
        step_condition = tf.reduce_max(eigenvalues, axis=1) / tf.maximum(
            step_minimum, tf.constant(1.0e-300, tf.float64)
        )
        max_innovation_condition = tf.maximum(
            max_innovation_condition, step_condition
        )
        invalid_eigensolver_input = tf.logical_or(
            invalid_eigensolver_input, tf.logical_not(row_input_valid)
        )
        return (
            t + 1,
            mean,
            covariance,
            dmean,
            dcovariance,
            log_likelihood,
            score,
            max_floor_count,
            max_projection_residual,
            min_innovation_eigenvalue,
            max_innovation_condition,
            invalid_eigensolver_input,
        )

    initial_state = (
        tf.constant(0, tf.int32),
        mean0,
        covariance0,
        dmean0,
        dcovariance0,
        tf.zeros((batch_size,), tf.float64),
        tf.zeros((batch_size, parameter_dim), tf.float64),
        tf.zeros((batch_size,), tf.int32),
        tf.zeros((batch_size,), tf.float64),
        tf.fill((batch_size,), tf.constant(float("inf"), tf.float64)),
        tf.zeros((batch_size,), tf.float64),
        tf.zeros((batch_size,), tf.bool),
    )
    final = tf.while_loop(cond, body, initial_state, parallel_iterations=1)
    log_likelihood = final[5]
    score = final[6]
    floor_count_value = final[7]
    projection_residual = final[8]
    min_innovation_eigenvalue = final[9]
    condition_estimate = final[10]
    invalid_eigensolver_input = final[11]
    active_floor = floor_count_value > 0
    status_code = tf.where(
        invalid_eigensolver_input,
        tf.constant(SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT, tf.int32),
        tf.where(
            active_floor,
            tf.constant(SVD_LINEAR_SCORE_STATUS_BLOCKED_ACTIVE_FLOOR, tf.int32),
            tf.constant(SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED, tf.int32),
        ),
    )
    valid = tf.logical_and(
        tf.logical_not(active_floor), tf.logical_not(invalid_eigensolver_input)
    )
    return BatchedSVDLinearGaussianScoreResult(
        log_likelihood=log_likelihood,
        score=score,
        status_code=status_code,
        valid_pre_regularized_score=valid,
        floor_count_value=floor_count_value,
        min_innovation_eigenvalue=min_innovation_eigenvalue,
        innovation_condition_estimate=condition_estimate,
        psd_projection_residual=projection_residual,
        invalid_eigensolver_input=invalid_eigensolver_input,
    )


def _eigh_solve(
    eigenvectors: tf.Tensor,
    eigenvalues: tf.Tensor,
    rhs: tf.Tensor,
) -> tf.Tensor:
    if rhs.shape.rank == 2:
        projected = tf.linalg.matvec(
            tf.linalg.matrix_transpose(eigenvectors), rhs
        )
        return tf.linalg.matvec(eigenvectors, projected / eigenvalues)
    projected = tf.linalg.matrix_transpose(eigenvectors) @ rhs
    return eigenvectors @ (projected / eigenvalues[:, :, tf.newaxis])


def _tensor(value: Any) -> tf.Tensor:
    return tf.convert_to_tensor(value, tf.float64)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _validate_shapes(y: tf.Tensor, *tensors: tf.Tensor) -> tuple[tf.Tensor, int, int, int]:
    (
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        mean,
        covariance,
        dmean,
        dcovariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
    ) = tensors
    parameter_dim = dmean.shape[1]
    state_dim = mean.shape[1]
    observation_dim = observation_offset.shape[1]
    if parameter_dim is None or state_dim is None or observation_dim is None:
        raise ValueError("parameter, state, and observation dimensions must be static")
    batch_size = tf.shape(mean)[0]
    _assert_batch_shape(transition_offset, (None, state_dim), batch_size, "transition_offset")
    _assert_batch_shape(transition_matrix, (None, state_dim, state_dim), batch_size, "transition_matrix")
    _assert_batch_shape(transition_covariance, (None, state_dim, state_dim), batch_size, "transition_covariance")
    _assert_batch_shape(observation_offset, (None, observation_dim), batch_size, "observation_offset")
    _assert_batch_shape(observation_matrix, (None, observation_dim, state_dim), batch_size, "observation_matrix")
    _assert_batch_shape(observation_covariance, (None, observation_dim, observation_dim), batch_size, "observation_covariance")
    _assert_batch_shape(mean, (None, state_dim), batch_size, "mean")
    _assert_batch_shape(covariance, (None, state_dim, state_dim), batch_size, "covariance")
    _assert_batch_shape(dmean, (None, parameter_dim, state_dim), batch_size, "dmean")
    _assert_batch_shape(dcovariance, (None, parameter_dim, state_dim, state_dim), batch_size, "dcovariance")
    _assert_batch_shape(d_transition_offset, (None, parameter_dim, state_dim), batch_size, "d_transition_offset")
    _assert_batch_shape(d_transition_matrix, (None, parameter_dim, state_dim, state_dim), batch_size, "d_transition_matrix")
    _assert_batch_shape(d_transition_covariance, (None, parameter_dim, state_dim, state_dim), batch_size, "d_transition_covariance")
    _assert_batch_shape(d_observation_offset, (None, parameter_dim, observation_dim), batch_size, "d_observation_offset")
    _assert_batch_shape(d_observation_matrix, (None, parameter_dim, observation_dim, state_dim), batch_size, "d_observation_matrix")
    _assert_batch_shape(d_observation_covariance, (None, parameter_dim, observation_dim, observation_dim), batch_size, "d_observation_covariance")
    if y.shape[1] is not None and int(y.shape[1]) != int(observation_dim):
        raise ValueError("observation width mismatch")
    return batch_size, int(parameter_dim), int(state_dim), int(observation_dim)


def _assert_batch_shape(
    tensor: tf.Tensor,
    expected: tuple[int | None, ...],
    batch_size: tf.Tensor,
    name: str,
) -> None:
    if not tensor.shape.is_compatible_with(tf.TensorShape(expected)):
        raise ValueError(f"{name} has shape {tensor.shape}, expected {expected}")
    tf.debugging.assert_equal(
        tf.shape(tensor)[0], batch_size, message=f"{name} batch mismatch"
    )
