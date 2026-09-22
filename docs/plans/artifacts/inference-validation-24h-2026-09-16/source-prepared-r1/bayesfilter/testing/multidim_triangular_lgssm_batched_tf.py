"""Batch-native TensorFlow materialization for the exact 18D LGSSM target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.testing.multidim_triangular_lgssm_tf import (
    load_lower_triangular_lgssm_contract,
    raw_truth_from_contract,
)


PARAMETER_DIM = 18
STATE_DIM = 4
OBSERVATION_DIM = 4

_TRANSITION_BASIS = tf.constant(
    [
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]],
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
    ],
    dtype=tf.float64,
)


@dataclass(frozen=True)
class BatchedLowerTriangularLGSSMMaterialization:
    """Batch-leading model tensors and first derivatives."""

    raw_parameters: tf.Tensor
    initial_mean: tf.Tensor
    initial_covariance: tf.Tensor
    transition_offset: tf.Tensor
    transition_matrix: tf.Tensor
    transition_covariance: tf.Tensor
    observation_offset: tf.Tensor
    observation_matrix: tf.Tensor
    observation_covariance: tf.Tensor
    process_std: tf.Tensor
    observation_std: tf.Tensor
    d_initial_mean: tf.Tensor
    d_initial_covariance: tf.Tensor
    d_transition_offset: tf.Tensor
    d_transition_matrix: tf.Tensor
    d_transition_covariance: tf.Tensor
    d_observation_offset: tf.Tensor
    d_observation_matrix: tf.Tensor
    d_observation_covariance: tf.Tensor


def materialize_lower_triangular_lgssm_batch(
    raw_parameters: Any,
    contract: Mapping[str, Any] | None = None,
) -> BatchedLowerTriangularLGSSMMaterialization:
    """Materialize exact model tensors and first derivatives for `[B,18]`."""

    manifest = load_lower_triangular_lgssm_contract() if contract is None else contract
    _validate_contract(manifest)
    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    if raw.shape.rank != 2:
        raise ValueError("raw_parameters must have rank 2")
    if raw.shape[-1] is not None and int(raw.shape[-1]) != PARAMETER_DIM:
        raise ValueError("raw_parameters trailing dimension must equal 18")
    tf.debugging.assert_equal(
        tf.shape(raw)[1],
        tf.constant(PARAMETER_DIM, tf.int32),
        message="raw_parameters trailing dimension must equal 18",
    )

    batch_size = tf.shape(raw)[0]
    rho_max = tf.constant(float(manifest["transform"]["rho_max"]), tf.float64)
    lower_scale = tf.constant(
        float(manifest["transform"]["lower_scale"]), tf.float64
    )
    transition_values = tf.concat(
        (rho_max * tf.tanh(raw[:, 0:4]), lower_scale * tf.tanh(raw[:, 4:10])),
        axis=1,
    )
    transition = tf.einsum("bk,kij->bij", transition_values, _TRANSITION_BASIS)
    transition_derivative_values = tf.concat(
        (
            rho_max * (1.0 - tf.square(tf.tanh(raw[:, 0:4]))),
            lower_scale * (1.0 - tf.square(tf.tanh(raw[:, 4:10]))),
            tf.zeros((batch_size, 8), tf.float64),
        ),
        axis=1,
    )
    d_transition = tf.einsum(
        "bp,pij->bpij",
        transition_derivative_values,
        _parameter_transition_basis(),
    )

    process_std = tf.exp(raw[:, 10:14])
    observation_std = tf.exp(raw[:, 14:18])
    process_covariance = tf.linalg.diag(tf.square(process_std))
    observation_covariance = tf.linalg.diag(tf.square(observation_std))
    d_process_covariance = _diagonal_covariance_derivatives(
        raw,
        parameter_start=10,
    )
    d_observation_covariance = _diagonal_covariance_derivatives(
        raw,
        parameter_start=14,
    )
    stationary, d_stationary = _stationary_covariance_and_derivatives(
        transition,
        process_covariance,
        d_transition,
        d_process_covariance,
    )

    model_zeros = tf.zeros((batch_size, STATE_DIM), tf.float64)
    derivative_zeros = tf.zeros(
        (batch_size, PARAMETER_DIM, STATE_DIM), tf.float64
    )
    matrix_derivative_zeros = tf.zeros(
        (batch_size, PARAMETER_DIM, STATE_DIM, STATE_DIM), tf.float64
    )
    observation_matrix = tf.broadcast_to(
        tf.eye(OBSERVATION_DIM, STATE_DIM, dtype=tf.float64)[tf.newaxis, :, :],
        (batch_size, OBSERVATION_DIM, STATE_DIM),
    )
    return BatchedLowerTriangularLGSSMMaterialization(
        raw_parameters=raw,
        initial_mean=model_zeros,
        initial_covariance=stationary,
        transition_offset=model_zeros,
        transition_matrix=transition,
        transition_covariance=process_covariance,
        observation_offset=model_zeros,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        process_std=process_std,
        observation_std=observation_std,
        d_initial_mean=derivative_zeros,
        d_initial_covariance=d_stationary,
        d_transition_offset=derivative_zeros,
        d_transition_matrix=d_transition,
        d_transition_covariance=d_process_covariance,
        d_observation_offset=derivative_zeros,
        d_observation_matrix=matrix_derivative_zeros,
        d_observation_covariance=d_observation_covariance,
    )


def gaussian_raw_prior_log_prob_and_score_batch(
    raw_parameters: Any,
    contract: Mapping[str, Any] | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return the scalar authority's Gaussian prior over a leading batch."""

    manifest = load_lower_triangular_lgssm_contract() if contract is None else contract
    _validate_contract(manifest)
    raw = tf.convert_to_tensor(raw_parameters, dtype=tf.float64)
    if raw.shape.rank != 2:
        raise ValueError("raw_parameters must have rank 2")
    center = raw_truth_from_contract(dict(manifest))[tf.newaxis, :]
    scales = tf.constant(
        [
            0.50,
            0.50,
            0.50,
            0.50,
            0.60,
            0.60,
            0.60,
            0.60,
            0.60,
            0.60,
            0.35,
            0.35,
            0.35,
            0.35,
            0.35,
            0.35,
            0.35,
            0.35,
        ],
        tf.float64,
    )[tf.newaxis, :]
    standardized = (raw - center) / scales
    return (
        -0.5 * tf.reduce_sum(tf.square(standardized), axis=1),
        -(raw - center) / tf.square(scales),
    )


def _stationary_covariance_and_derivatives(
    transition: tf.Tensor,
    covariance: tf.Tensor,
    d_transition: tf.Tensor,
    d_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    batch_size = tf.shape(transition)[0]
    system = (
        tf.eye(STATE_DIM * STATE_DIM, batch_shape=(batch_size,), dtype=tf.float64)
        - _batched_kron(transition, transition)
    )
    stationary_flat = tf.linalg.solve(
        system,
        tf.reshape(covariance, (batch_size, STATE_DIM * STATE_DIM, 1)),
    )
    stationary = _symmetrize(
        tf.reshape(stationary_flat, (batch_size, STATE_DIM, STATE_DIM))
    )
    transition_p = transition[:, tf.newaxis, :, :]
    stationary_p = stationary[:, tf.newaxis, :, :]
    derivative_rhs = _symmetrize(
        d_transition @ stationary_p @ tf.linalg.matrix_transpose(transition_p)
        + transition_p @ stationary_p @ tf.linalg.matrix_transpose(d_transition)
        + d_covariance
    )
    rhs_columns = tf.transpose(
        tf.reshape(
            derivative_rhs,
            (batch_size, PARAMETER_DIM, STATE_DIM * STATE_DIM),
        ),
        (0, 2, 1),
    )
    derivative_solution = tf.linalg.solve(system, rhs_columns)
    d_stationary = _symmetrize(
        tf.reshape(
            tf.transpose(derivative_solution, (0, 2, 1)),
            (batch_size, PARAMETER_DIM, STATE_DIM, STATE_DIM),
        )
    )
    return stationary, d_stationary


def _diagonal_covariance_derivatives(
    raw: tf.Tensor,
    *,
    parameter_start: int,
) -> tf.Tensor:
    updates = 2.0 * tf.exp(2.0 * raw[:, parameter_start : parameter_start + 4])
    parameter_basis = tf.one_hot(
        tf.range(parameter_start, parameter_start + 4),
        depth=PARAMETER_DIM,
        dtype=tf.float64,
    )
    diagonal_basis = tf.linalg.diag(tf.eye(STATE_DIM, dtype=tf.float64))
    return tf.einsum(
        "bk,kp,kij->bpij",
        updates,
        parameter_basis,
        diagonal_basis,
    )


def _parameter_transition_basis() -> tf.Tensor:
    return tf.concat(
        (_TRANSITION_BASIS, tf.zeros((8, STATE_DIM, STATE_DIM), tf.float64)),
        axis=0,
    )


def _batched_kron(left: tf.Tensor, right: tf.Tensor) -> tf.Tensor:
    product = tf.einsum("bij,bkl->bikjl", left, right)
    return tf.reshape(
        product,
        (tf.shape(left)[0], STATE_DIM * STATE_DIM, STATE_DIM * STATE_DIM),
    )


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _validate_contract(contract: Mapping[str, Any]) -> None:
    static = contract["static_shape"]
    if (
        int(static["parameter_dim"]) != PARAMETER_DIM
        or int(static["state_dim"]) != STATE_DIM
        or int(static["observation_dim"]) != OBSERVATION_DIM
    ):
        raise ValueError("batch materializer requires the exact 18D/4D/4D contract")
