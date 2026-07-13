"""TensorFlow QR/square-root analytic Kalman score and Hessian backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.dtypes_tf import as_float_tensor, common_floating_dtype
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor_derivatives,
    cholesky_factor_first_derivatives,
    factor_covariance_derivatives,
    factor_covariance_first_derivatives,
    factor_solve,
    stack_qr_lower_factor_first_derivatives,
    stack_qr_lower_factor_derivatives,
    trace_factor_solve,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from bayesfilter.results_tf import TFFilterDerivativeResult
from bayesfilter.structural import FilterRunMetadata


TFQRLinearDerivativeBackend = Literal["tf_qr_sqrt", "tf_masked_qr_sqrt"]


def _to_tensor(value: object, dtype: tf.DType) -> tf.Tensor:
    return as_float_tensor(value, dtype)


def _score_dtype(
    *values: object,
    context: str,
) -> tf.DType:
    return common_floating_dtype(*values, context=context)


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


def _static_dim(tensor: tf.Tensor, axis: int, name: str) -> int:
    size = tensor.shape[axis]
    if size is None:
        raise ValueError(f"{name} must be statically known for QR derivative scans")
    return int(size)


def _validate_mask_shape(observations: tf.Tensor, observation_mask: tf.Tensor) -> None:
    tf.debugging.assert_equal(
        tf.shape(observation_mask),
        tf.shape(observations),
        message="Observation mask shape must match observations shape.",
    )


def _validate_batched_static_score_shapes(
    *,
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
) -> None:
    """Fail closed for the batched-static QR score shape contract."""

    expected_ranks = {
        "initial_state_mean": (initial_state_mean, 2),
        "initial_state_covariance": (initial_state_covariance, 3),
        "transition_offset": (transition_offset, 2),
        "transition_matrix": (transition_matrix, 3),
        "transition_covariance": (transition_covariance, 3),
        "observation_offset": (observation_offset, 2),
        "observation_matrix": (observation_matrix, 3),
        "observation_covariance": (observation_covariance, 3),
        "d_initial_state_mean": (d_initial_state_mean, 3),
        "d_initial_state_covariance": (d_initial_state_covariance, 4),
        "d_transition_offset": (d_transition_offset, 3),
        "d_transition_matrix": (d_transition_matrix, 4),
        "d_transition_covariance": (d_transition_covariance, 4),
        "d_observation_offset": (d_observation_offset, 3),
        "d_observation_matrix": (d_observation_matrix, 4),
        "d_observation_covariance": (d_observation_covariance, 4),
    }
    for name, (tensor, rank) in expected_ranks.items():
        if tensor.shape.rank != rank:
            raise ValueError(f"{name} must have rank {rank} for batched-static QR score")

    batch_size = tf.shape(initial_state_mean)[0]
    parameter_dim = tf.shape(d_initial_state_mean)[1]
    for name, tensor in (
        ("initial_state_covariance", initial_state_covariance),
        ("transition_offset", transition_offset),
        ("transition_matrix", transition_matrix),
        ("transition_covariance", transition_covariance),
        ("observation_offset", observation_offset),
        ("observation_matrix", observation_matrix),
        ("observation_covariance", observation_covariance),
        ("d_initial_state_mean", d_initial_state_mean),
        ("d_initial_state_covariance", d_initial_state_covariance),
        ("d_transition_offset", d_transition_offset),
        ("d_transition_matrix", d_transition_matrix),
        ("d_transition_covariance", d_transition_covariance),
        ("d_observation_offset", d_observation_offset),
        ("d_observation_matrix", d_observation_matrix),
        ("d_observation_covariance", d_observation_covariance),
    ):
        tf.debugging.assert_equal(
            tf.shape(tensor)[0],
            batch_size,
            message=f"{name} batch dimension must match initial_state_mean",
        )
    for name, tensor in (
        ("d_initial_state_covariance", d_initial_state_covariance),
        ("d_transition_offset", d_transition_offset),
        ("d_transition_matrix", d_transition_matrix),
        ("d_transition_covariance", d_transition_covariance),
        ("d_observation_offset", d_observation_offset),
        ("d_observation_matrix", d_observation_matrix),
        ("d_observation_covariance", d_observation_covariance),
    ):
        tf.debugging.assert_equal(
            tf.shape(tensor)[1],
            parameter_dim,
            message=f"{name} parameter dimension must match d_initial_state_mean",
        )


def _batched_qr_positive(matrix: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Batched thin QR with a positive diagonal in the triangular factor."""

    dtype = common_floating_dtype(matrix, context="batched QR derivative matrix")
    matrix = as_float_tensor(matrix, dtype, name="matrix")
    q, r = tf.linalg.qr(matrix, full_matrices=False)
    signs = tf.sign(tf.linalg.diag_part(r))
    signs = tf.where(tf.equal(signs, 0.0), tf.ones_like(signs), signs)
    return q * signs[..., tf.newaxis, :], signs[..., :, tf.newaxis] * r


def _batched_right_solve_upper(matrix: tf.Tensor, upper: tf.Tensor) -> tf.Tensor:
    """Return ``matrix @ inv(upper)`` for batched upper-triangular factors."""

    solved_t = tf.linalg.triangular_solve(
        _matrix_transpose(upper),
        _matrix_transpose(matrix),
        lower=True,
    )
    return _matrix_transpose(solved_t)


def _batched_omega_from_a(a: tf.Tensor) -> tf.Tensor:
    lower = tf.linalg.band_part(a, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(a))
    return lower - _matrix_transpose(lower)


def _batched_qr_factor_derivative(
    matrix: tf.Tensor,
    dmatrix: tf.Tensor,
    q: tf.Tensor,
    r: tf.Tensor,
) -> tf.Tensor:
    """Return the batched derivative of the positive-diagonal QR ``R`` factor."""

    dmatrix_r_inv = _batched_right_solve_upper(dmatrix, r)
    a = _matrix_transpose(q) @ dmatrix_r_inv
    omega = _batched_omega_from_a(a)
    return (a - omega) @ r


def _batched_stack_qr_lower_factor_first_derivatives(
    stack: tf.Tensor,
    dstack: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Factor batched ``stack @ stack.T`` and return first derivatives."""

    rows = stack.shape[-2]
    columns = stack.shape[-1]
    if rows is not None and columns is not None and columns < rows:
        raise ValueError("batched QR derivative stack requires K>=N")
    tf.debugging.assert_greater_equal(
        tf.shape(stack)[-1],
        tf.shape(stack)[-2],
        message="batched QR derivative stack requires K>=N",
    )
    matrix = _matrix_transpose(stack)
    dmatrix = _matrix_transpose(dstack)
    q, r = _batched_qr_positive(matrix)
    dr = _batched_qr_factor_derivative(
        matrix[:, tf.newaxis, :, :],
        dmatrix,
        q[:, tf.newaxis, :, :],
        r[:, tf.newaxis, :, :],
    )
    factor = _matrix_transpose(r)
    dfactor = _matrix_transpose(dr)
    return factor, dfactor, tf.reduce_min(tf.linalg.diag_part(factor), axis=-1)


def _batched_factor_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve batched ``(factor @ factor.T) x = rhs`` for lower factors."""

    dtype = common_floating_dtype(factor, rhs, context="batched factor solve inputs")
    factor = as_float_tensor(factor, dtype, name="factor")
    rhs = as_float_tensor(rhs, dtype, name="rhs")
    squeeze = rhs.shape.rank == 2
    rhs_matrix = rhs[..., tf.newaxis] if squeeze else rhs
    first = tf.linalg.triangular_solve(factor, rhs_matrix, lower=True)
    second = tf.linalg.triangular_solve(_matrix_transpose(factor), first, lower=False)
    return second[..., 0] if squeeze else second


def _batched_symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + _matrix_transpose(matrix))


def _batched_cholesky_factor(
    covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tf.Tensor:
    dtype = common_floating_dtype(covariance, jitter, context="batched Cholesky inputs")
    covariance = _batched_symmetrize(as_float_tensor(covariance, dtype, name="covariance"))
    jitter_tensor = as_float_tensor(jitter, dtype, name="jitter")
    batch_size = tf.shape(covariance)[0]
    dim = tf.shape(covariance)[-1]
    identity = tf.eye(dim, batch_shape=[batch_size], dtype=dtype)
    return tf.linalg.cholesky(covariance + jitter_tensor * identity)


def _batched_cholesky_factor_first_derivatives(
    covariance: tf.Tensor,
    dcovariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Differentiate batched ``covariance + jitter I = L L.T`` to first order."""

    dtype = common_floating_dtype(
        covariance,
        dcovariance,
        jitter,
        context="batched Cholesky first-derivative inputs",
    )
    covariance = _batched_symmetrize(as_float_tensor(covariance, dtype, name="covariance"))
    dcovariance = as_float_tensor(dcovariance, dtype, name="dcovariance")
    factor = _batched_cholesky_factor(covariance, jitter=jitter)
    factor_with_parameter_axis = factor[:, tf.newaxis, :, :]
    left = tf.linalg.triangular_solve(
        factor_with_parameter_axis,
        _batched_symmetrize(dcovariance),
        lower=True,
    )
    b = _batched_right_solve_upper(
        left,
        _matrix_transpose(factor_with_parameter_axis),
    )
    g = tf.linalg.band_part(b, -1, 0) - 0.5 * tf.linalg.diag(tf.linalg.diag_part(b))
    return factor, factor_with_parameter_axis @ g


def _batched_factor_covariance_first_derivatives(
    factor: tf.Tensor,
    dfactor: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert batched first factor derivatives into covariance derivatives."""

    dtype = common_floating_dtype(
        factor,
        dfactor,
        context="batched factor covariance first-derivative inputs",
    )
    factor = as_float_tensor(factor, dtype, name="factor")
    dfactor = as_float_tensor(dfactor, dtype, name="dfactor")
    covariance = factor @ _matrix_transpose(factor)
    factor_with_parameter_axis = factor[:, tf.newaxis, :, :]
    factor_t = _matrix_transpose(factor)
    dcovariance = _batched_symmetrize(
        dfactor @ factor_t[:, tf.newaxis, :, :]
        + factor_with_parameter_axis @ _matrix_transpose(dfactor)
    )
    return covariance, dcovariance


@tf.function
def _tf_qr_sqrt_kalman_score(
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
) -> tuple[tf.Tensor, tf.Tensor]:
    """Dense direct-QR square-root log likelihood and analytic score.

    Private diagnostic helper used to isolate first-order QR derivative costs
    before deciding whether a public score-only API is warranted.
    """

    dtype = _score_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        d_initial_state_mean,
        d_initial_state_covariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
        jitter,
        context="QR first-order score inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_dim(y, 0, "observation length")
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)

    dmean = _to_tensor(d_initial_state_mean, dtype)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance, dtype)
    d_transition_offset = _to_tensor(d_transition_offset, dtype)
    d_transition_matrix = _to_tensor(d_transition_matrix, dtype)
    d_transition_covariance = _to_tensor(d_transition_covariance, dtype)
    d_observation_offset = _to_tensor(d_observation_offset, dtype)
    d_observation_matrix = _to_tensor(d_observation_matrix, dtype)
    d_observation_covariance = _to_tensor(d_observation_covariance, dtype)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    jitter_tensor = _to_tensor(jitter, dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    covariance_factor, dcovariance_factor = cholesky_factor_first_derivatives(
        initial_state_covariance,
        d_initial_state_covariance,
        jitter=0.0,
    )
    transition_covariance_factor, dtransition_covariance_factor = (
        cholesky_factor_first_derivatives(
            transition_covariance,
            d_transition_covariance,
            jitter=0.0,
        )
    )
    observation_covariance_factor, dobservation_covariance_factor = (
        cholesky_factor_first_derivatives(
            observation_covariance + jitter_tensor * obs_identity,
            d_observation_covariance,
            jitter=0.0,
        )
    )

    score = tf.zeros((parameter_dim,), dtype=dtype)
    log_likelihood = tf.constant(0.0, dtype=dtype)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        dpredicted_mean_values = []
        for i in range(parameter_dim):
            dpredicted_mean_values.append(
                d_transition_offset[i]
                + _matvec(d_transition_matrix[i], mean)
                + _matvec(transition_matrix, dmean[i])
            )
        dpredicted_mean = tf.stack(dpredicted_mean_values, axis=0)

        prediction_left = transition_matrix @ covariance_factor
        prediction_stack = tf.concat(
            (prediction_left, transition_covariance_factor),
            axis=1,
        )
        dprediction_stack_values = []
        for i in range(parameter_dim):
            dprediction_stack_values.append(
                tf.concat(
                    (
                        d_transition_matrix[i] @ covariance_factor
                        + transition_matrix @ dcovariance_factor[i],
                        dtransition_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
        dprediction_stack = tf.stack(dprediction_stack_values, axis=0)
        predicted_factor, dpredicted_factor, _ = stack_qr_lower_factor_first_derivatives(
            prediction_stack,
            dprediction_stack,
        )
        predicted_covariance, dpredicted_covariance = factor_covariance_first_derivatives(
            predicted_factor,
            dpredicted_factor,
        )

        innovation = y[t] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        dinnovation_values = []
        for i in range(parameter_dim):
            dinnovation_values.append(
                -d_observation_offset[i]
                - _matvec(d_observation_matrix[i], predicted_mean)
                - _matvec(observation_matrix, dpredicted_mean[i])
            )
        dinnovation = tf.stack(dinnovation_values, axis=0)

        innovation_left = observation_matrix @ predicted_factor
        innovation_stack = tf.concat(
            (innovation_left, observation_covariance_factor),
            axis=1,
        )
        dinnovation_stack_values = []
        for i in range(parameter_dim):
            dinnovation_stack_values.append(
                tf.concat(
                    (
                        d_observation_matrix[i] @ predicted_factor
                        + observation_matrix @ dpredicted_factor[i],
                        dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
        dinnovation_stack = tf.stack(dinnovation_stack_values, axis=0)
        innovation_factor, dinnovation_factor, _ = (
            stack_qr_lower_factor_first_derivatives(
                innovation_stack,
                dinnovation_stack,
            )
        )
        _, dS = factor_covariance_first_derivatives(
            innovation_factor,
            dinnovation_factor,
        )
        innovation_solve = factor_solve(innovation_factor, innovation)
        innovation_precision = factor_solve(innovation_factor, obs_identity)

        score_contrib_values = []
        dSinv_values = []
        for i in range(parameter_dim):
            dSinv_i = -factor_solve(innovation_factor, dS[i] @ innovation_precision)
            score_contrib_values.append(
                -0.5
                * (
                    trace_factor_solve(innovation_factor, dS[i])
                    + 2.0 * tf.tensordot(dinnovation[i], innovation_solve, axes=1)
                    - tf.tensordot(
                        innovation_solve,
                        _matvec(dS[i], innovation_solve),
                        axes=1,
                    )
                )
            )
            dSinv_values.append(dSinv_i)
        score = score + tf.stack(score_contrib_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)

        kalman_gain = (
            predicted_covariance
            @ tf.transpose(observation_matrix)
            @ innovation_precision
        )
        dK_values = []
        for i in range(parameter_dim):
            dK_values.append(
                dpredicted_covariance[i]
                @ tf.transpose(observation_matrix)
                @ innovation_precision
                + predicted_covariance
                @ tf.transpose(d_observation_matrix[i])
                @ innovation_precision
                + predicted_covariance @ tf.transpose(observation_matrix) @ dSinv[i]
            )
        dK = tf.stack(dK_values, axis=0)

        joseph_left = state_identity - kalman_gain @ observation_matrix
        d_joseph_left_values = []
        for i in range(parameter_dim):
            d_joseph_left_values.append(
                -dK[i] @ observation_matrix - kalman_gain @ d_observation_matrix[i]
            )
        d_joseph_left = tf.stack(d_joseph_left_values, axis=0)

        update_left = joseph_left @ predicted_factor
        update_right = kalman_gain @ observation_covariance_factor
        update_stack = tf.concat((update_left, update_right), axis=1)
        dupdate_stack_values = []
        for i in range(parameter_dim):
            dupdate_stack_values.append(
                tf.concat(
                    (
                        d_joseph_left[i] @ predicted_factor
                        + joseph_left @ dpredicted_factor[i],
                        dK[i] @ observation_covariance_factor
                        + kalman_gain @ dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
        dupdate_stack = tf.stack(dupdate_stack_values, axis=0)
        covariance_factor, dcovariance_factor, _ = (
            stack_qr_lower_factor_first_derivatives(
                update_stack,
                dupdate_stack,
            )
        )

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        dmean_values = []
        for i in range(parameter_dim):
            dmean_values.append(
                dpredicted_mean[i]
                + _matvec(dK[i], innovation)
                + _matvec(kalman_gain, dinnovation[i])
            )
        dmean = tf.stack(dmean_values, axis=0)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        log_likelihood = log_likelihood + contribution

    return log_likelihood, score


@tf.function
def tf_qr_sqrt_kalman_score_hessian(
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
    d2_initial_state_mean: tf.Tensor,
    d2_initial_state_covariance: tf.Tensor,
    d2_transition_offset: tf.Tensor,
    d2_transition_matrix: tf.Tensor,
    d2_transition_covariance: tf.Tensor,
    d2_observation_offset: tf.Tensor,
    d2_observation_matrix: tf.Tensor,
    d2_observation_covariance: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Dense direct-QR square-root log likelihood, score, and Hessian."""

    dtype = _score_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        d_initial_state_mean,
        d_initial_state_covariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
        d2_initial_state_mean,
        d2_initial_state_covariance,
        d2_transition_offset,
        d2_transition_matrix,
        d2_transition_covariance,
        d2_observation_offset,
        d2_observation_matrix,
        d2_observation_covariance,
        jitter,
        context="QR score Hessian inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_dim(y, 0, "observation length")
    transition_offset = _to_tensor(transition_offset, dtype)
    transition_matrix = _to_tensor(transition_matrix, dtype)
    transition_covariance = _to_tensor(transition_covariance, dtype)
    observation_offset = _to_tensor(observation_offset, dtype)
    observation_matrix = _to_tensor(observation_matrix, dtype)
    observation_covariance = _to_tensor(observation_covariance, dtype)
    mean = _to_tensor(initial_state_mean, dtype)
    initial_state_covariance = _to_tensor(initial_state_covariance, dtype)

    dmean = _to_tensor(d_initial_state_mean, dtype)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance, dtype)
    d_transition_offset = _to_tensor(d_transition_offset, dtype)
    d_transition_matrix = _to_tensor(d_transition_matrix, dtype)
    d_transition_covariance = _to_tensor(d_transition_covariance, dtype)
    d_observation_offset = _to_tensor(d_observation_offset, dtype)
    d_observation_matrix = _to_tensor(d_observation_matrix, dtype)
    d_observation_covariance = _to_tensor(d_observation_covariance, dtype)
    ddmean = _to_tensor(d2_initial_state_mean, dtype)
    d2_initial_state_covariance = _to_tensor(d2_initial_state_covariance, dtype)
    d2_transition_offset = _to_tensor(d2_transition_offset, dtype)
    d2_transition_matrix = _to_tensor(d2_transition_matrix, dtype)
    d2_transition_covariance = _to_tensor(d2_transition_covariance, dtype)
    d2_observation_offset = _to_tensor(d2_observation_offset, dtype)
    d2_observation_matrix = _to_tensor(d2_observation_matrix, dtype)
    d2_observation_covariance = _to_tensor(d2_observation_covariance, dtype)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    jitter_tensor = _to_tensor(jitter, dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    covariance_factor, dcovariance_factor, d2covariance_factor = (
        cholesky_factor_derivatives(
            initial_state_covariance,
            d_initial_state_covariance,
            d2_initial_state_covariance,
            jitter=0.0,
        )
    )
    transition_covariance_factor, dtransition_covariance_factor, d2transition_covariance_factor = (
        cholesky_factor_derivatives(
            transition_covariance,
            d_transition_covariance,
            d2_transition_covariance,
            jitter=0.0,
        )
    )
    observation_covariance_factor, dobservation_covariance_factor, d2observation_covariance_factor = (
        cholesky_factor_derivatives(
            observation_covariance + jitter_tensor * obs_identity,
            d_observation_covariance,
            d2_observation_covariance,
            jitter=0.0,
        )
    )

    score = tf.zeros((parameter_dim,), dtype=dtype)
    hessian = tf.zeros((parameter_dim, parameter_dim), dtype=dtype)
    log_likelihood = tf.constant(0.0, dtype=dtype)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        dpredicted_mean_values = []
        d2predicted_mean_rows = []
        for i in range(parameter_dim):
            dpredicted_mean_i = (
                d_transition_offset[i]
                + _matvec(d_transition_matrix[i], mean)
                + _matvec(transition_matrix, dmean[i])
            )
            dpredicted_mean_values.append(dpredicted_mean_i)
            d2predicted_mean_values = []
            for j in range(parameter_dim):
                d2predicted_mean_values.append(
                    d2_transition_offset[i, j]
                    + _matvec(d2_transition_matrix[i, j], mean)
                    + _matvec(d_transition_matrix[i], dmean[j])
                    + _matvec(d_transition_matrix[j], dmean[i])
                    + _matvec(transition_matrix, ddmean[i, j])
                )
            d2predicted_mean_rows.append(tf.stack(d2predicted_mean_values, axis=0))
        dpredicted_mean = tf.stack(dpredicted_mean_values, axis=0)
        d2predicted_mean = tf.stack(d2predicted_mean_rows, axis=0)

        prediction_left = transition_matrix @ covariance_factor
        prediction_stack = tf.concat(
            (prediction_left, transition_covariance_factor),
            axis=1,
        )
        dprediction_stack_values = []
        d2prediction_stack_rows = []
        for i in range(parameter_dim):
            dprediction_stack_values.append(
                tf.concat(
                    (
                        d_transition_matrix[i] @ covariance_factor
                        + transition_matrix @ dcovariance_factor[i],
                        dtransition_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2prediction_stack_values = []
            for j in range(parameter_dim):
                d2prediction_stack_values.append(
                    tf.concat(
                        (
                            d2_transition_matrix[i, j] @ covariance_factor
                            + d_transition_matrix[i] @ dcovariance_factor[j]
                            + d_transition_matrix[j] @ dcovariance_factor[i]
                            + transition_matrix @ d2covariance_factor[i, j],
                            d2transition_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2prediction_stack_rows.append(tf.stack(d2prediction_stack_values, axis=0))
        dprediction_stack = tf.stack(dprediction_stack_values, axis=0)
        d2prediction_stack = tf.stack(d2prediction_stack_rows, axis=0)
        predicted_factor, dpredicted_factor, d2predicted_factor, _ = (
            stack_qr_lower_factor_derivatives(
                prediction_stack,
                dprediction_stack,
                d2prediction_stack,
            )
        )
        predicted_covariance, dpredicted_covariance, d2predicted_covariance = (
            factor_covariance_derivatives(
                predicted_factor,
                dpredicted_factor,
                d2predicted_factor,
            )
        )

        innovation = y[t] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        dinnovation_values = []
        d2innovation_rows = []
        for i in range(parameter_dim):
            dinnovation_values.append(
                -d_observation_offset[i]
                - _matvec(d_observation_matrix[i], predicted_mean)
                - _matvec(observation_matrix, dpredicted_mean[i])
            )
            d2innovation_values = []
            for j in range(parameter_dim):
                d2innovation_values.append(
                    -d2_observation_offset[i, j]
                    - _matvec(d2_observation_matrix[i, j], predicted_mean)
                    - _matvec(d_observation_matrix[i], dpredicted_mean[j])
                    - _matvec(d_observation_matrix[j], dpredicted_mean[i])
                    - _matvec(observation_matrix, d2predicted_mean[i, j])
                )
            d2innovation_rows.append(tf.stack(d2innovation_values, axis=0))
        dinnovation = tf.stack(dinnovation_values, axis=0)
        d2innovation = tf.stack(d2innovation_rows, axis=0)

        innovation_left = observation_matrix @ predicted_factor
        innovation_stack = tf.concat(
            (innovation_left, observation_covariance_factor),
            axis=1,
        )
        dinnovation_stack_values = []
        d2innovation_stack_rows = []
        for i in range(parameter_dim):
            dinnovation_stack_values.append(
                tf.concat(
                    (
                        d_observation_matrix[i] @ predicted_factor
                        + observation_matrix @ dpredicted_factor[i],
                        dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2innovation_stack_values = []
            for j in range(parameter_dim):
                d2innovation_stack_values.append(
                    tf.concat(
                        (
                            d2_observation_matrix[i, j] @ predicted_factor
                            + d_observation_matrix[i] @ dpredicted_factor[j]
                            + d_observation_matrix[j] @ dpredicted_factor[i]
                            + observation_matrix @ d2predicted_factor[i, j],
                            d2observation_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2innovation_stack_rows.append(tf.stack(d2innovation_stack_values, axis=0))
        dinnovation_stack = tf.stack(dinnovation_stack_values, axis=0)
        d2innovation_stack = tf.stack(d2innovation_stack_rows, axis=0)
        innovation_factor, dinnovation_factor, d2innovation_factor, _ = (
            stack_qr_lower_factor_derivatives(
                innovation_stack,
                dinnovation_stack,
                d2innovation_stack,
            )
        )
        _, dS, d2S = factor_covariance_derivatives(
            innovation_factor,
            dinnovation_factor,
            d2innovation_factor,
        )
        innovation_solve = factor_solve(innovation_factor, innovation)
        innovation_precision = factor_solve(innovation_factor, obs_identity)

        dsolve_innovation_values = []
        dSinv_values = []
        score_contrib_values = []
        for i in range(parameter_dim):
            dsolve_i = factor_solve(
                innovation_factor,
                dinnovation[i] - _matvec(dS[i], innovation_solve),
            )
            dSinv_i = -factor_solve(innovation_factor, dS[i] @ innovation_precision)
            score_i = -0.5 * (
                trace_factor_solve(innovation_factor, dS[i])
                + 2.0 * tf.tensordot(dinnovation[i], innovation_solve, axes=1)
                - tf.tensordot(
                    innovation_solve,
                    _matvec(dS[i], innovation_solve),
                    axes=1,
                )
            )
            dsolve_innovation_values.append(dsolve_i)
            dSinv_values.append(dSinv_i)
            score_contrib_values.append(score_i)
        dsolve_innovation = tf.stack(dsolve_innovation_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        score_contrib = tf.stack(score_contrib_values, axis=0)
        score = score + score_contrib

        d2Sinv_rows = []
        hessian_contrib_rows = []
        for i in range(parameter_dim):
            d2Sinv_values = []
            hessian_contrib_values = []
            for j in range(parameter_dim):
                d2Sinv_ij = factor_solve(
                    innovation_factor,
                    dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - d2S[i, j] @ innovation_precision,
                )
                trace_term = tf.linalg.trace(
                    dSinv[j] @ dS[i] + innovation_precision @ d2S[i, j]
                )
                quad_term = (
                    2.0 * tf.tensordot(d2innovation[i, j], innovation_solve, axes=1)
                    + 2.0
                    * tf.tensordot(dinnovation[i], dsolve_innovation[j], axes=1)
                )
                curvature_term = (
                    tf.tensordot(
                        dinnovation[j],
                        _matvec(innovation_precision @ dS[i], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(dSinv[j] @ dS[i], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(innovation_precision @ d2S[i, j], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(innovation_precision @ dS[i], dsolve_innovation[j]),
                        axes=1,
                    )
                )
                hessian_contrib_values.append(
                    -0.5 * (trace_term + quad_term - curvature_term)
                )
                d2Sinv_values.append(d2Sinv_ij)
            d2Sinv_rows.append(tf.stack(d2Sinv_values, axis=0))
            hessian_contrib_rows.append(tf.stack(hessian_contrib_values, axis=0))
        d2Sinv = tf.stack(d2Sinv_rows, axis=0)
        hessian_contrib = tf.stack(hessian_contrib_rows, axis=0)
        hessian = hessian + hessian_contrib

        kalman_gain = (
            predicted_covariance
            @ tf.transpose(observation_matrix)
            @ innovation_precision
        )
        dK_values = []
        d2K_rows = []
        for i in range(parameter_dim):
            dK_i = (
                dpredicted_covariance[i]
                @ tf.transpose(observation_matrix)
                @ innovation_precision
                + predicted_covariance
                @ tf.transpose(d_observation_matrix[i])
                @ innovation_precision
                + predicted_covariance @ tf.transpose(observation_matrix) @ dSinv[i]
            )
            dK_values.append(dK_i)
            d2K_values = []
            for j in range(parameter_dim):
                d2K_values.append(
                    d2predicted_covariance[i, j]
                    @ tf.transpose(observation_matrix)
                    @ innovation_precision
                    + dpredicted_covariance[i]
                    @ tf.transpose(d_observation_matrix[j])
                    @ innovation_precision
                    + dpredicted_covariance[i]
                    @ tf.transpose(observation_matrix)
                    @ dSinv[j]
                    + dpredicted_covariance[j]
                    @ tf.transpose(d_observation_matrix[i])
                    @ innovation_precision
                    + predicted_covariance
                    @ tf.transpose(d2_observation_matrix[i, j])
                    @ innovation_precision
                    + predicted_covariance
                    @ tf.transpose(d_observation_matrix[i])
                    @ dSinv[j]
                    + dpredicted_covariance[j]
                    @ tf.transpose(observation_matrix)
                    @ dSinv[i]
                    + predicted_covariance
                    @ tf.transpose(d_observation_matrix[j])
                    @ dSinv[i]
                    + predicted_covariance
                    @ tf.transpose(observation_matrix)
                    @ d2Sinv[i, j]
                )
            d2K_rows.append(tf.stack(d2K_values, axis=0))
        dK = tf.stack(dK_values, axis=0)
        d2K = tf.stack(d2K_rows, axis=0)

        joseph_left = state_identity - kalman_gain @ observation_matrix
        d_joseph_left_values = []
        d2_joseph_left_rows = []
        for i in range(parameter_dim):
            d_joseph_left_values.append(
                -dK[i] @ observation_matrix - kalman_gain @ d_observation_matrix[i]
            )
            d2_joseph_left_values = []
            for j in range(parameter_dim):
                d2_joseph_left_values.append(
                    -d2K[i, j] @ observation_matrix
                    - dK[i] @ d_observation_matrix[j]
                    - dK[j] @ d_observation_matrix[i]
                    - kalman_gain @ d2_observation_matrix[i, j]
                )
            d2_joseph_left_rows.append(tf.stack(d2_joseph_left_values, axis=0))
        d_joseph_left = tf.stack(d_joseph_left_values, axis=0)
        d2_joseph_left = tf.stack(d2_joseph_left_rows, axis=0)

        update_left = joseph_left @ predicted_factor
        update_right = kalman_gain @ observation_covariance_factor
        update_stack = tf.concat((update_left, update_right), axis=1)
        dupdate_stack_values = []
        d2update_stack_rows = []
        for i in range(parameter_dim):
            dupdate_stack_values.append(
                tf.concat(
                    (
                        d_joseph_left[i] @ predicted_factor
                        + joseph_left @ dpredicted_factor[i],
                        dK[i] @ observation_covariance_factor
                        + kalman_gain @ dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2update_stack_values = []
            for j in range(parameter_dim):
                d2update_stack_values.append(
                    tf.concat(
                        (
                            d2_joseph_left[i, j] @ predicted_factor
                            + d_joseph_left[i] @ dpredicted_factor[j]
                            + d_joseph_left[j] @ dpredicted_factor[i]
                            + joseph_left @ d2predicted_factor[i, j],
                            d2K[i, j] @ observation_covariance_factor
                            + dK[i] @ dobservation_covariance_factor[j]
                            + dK[j] @ dobservation_covariance_factor[i]
                            + kalman_gain @ d2observation_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2update_stack_rows.append(tf.stack(d2update_stack_values, axis=0))
        dupdate_stack = tf.stack(dupdate_stack_values, axis=0)
        d2update_stack = tf.stack(d2update_stack_rows, axis=0)
        covariance_factor, dcovariance_factor, d2covariance_factor, _ = (
            stack_qr_lower_factor_derivatives(
                update_stack,
                dupdate_stack,
                d2update_stack,
            )
        )

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        dmean_values = []
        ddmean_rows = []
        for i in range(parameter_dim):
            dmean_values.append(
                dpredicted_mean[i]
                + _matvec(dK[i], innovation)
                + _matvec(kalman_gain, dinnovation[i])
            )
            ddmean_values = []
            for j in range(parameter_dim):
                ddmean_values.append(
                    d2predicted_mean[i, j]
                    + _matvec(d2K[i, j], innovation)
                    + _matvec(dK[i], dinnovation[j])
                    + _matvec(dK[j], dinnovation[i])
                    + _matvec(kalman_gain, d2innovation[i, j])
                )
            ddmean_rows.append(tf.stack(ddmean_values, axis=0))
        dmean = tf.stack(dmean_values, axis=0)
        ddmean = tf.stack(ddmean_rows, axis=0)

        solve_innovation = tf.linalg.triangular_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
            lower=True,
        )
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        contribution = -0.5 * (
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        log_likelihood = log_likelihood + contribution

    return log_likelihood, score, 0.5 * (hessian + tf.transpose(hessian))


@tf.function
def tf_qr_sqrt_masked_kalman_score_hessian(
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
    d2_initial_state_mean: tf.Tensor,
    d2_initial_state_covariance: tf.Tensor,
    d2_transition_offset: tf.Tensor,
    d2_transition_matrix: tf.Tensor,
    d2_transition_covariance: tf.Tensor,
    d2_observation_offset: tf.Tensor,
    d2_observation_matrix: tf.Tensor,
    d2_observation_covariance: tf.Tensor,
    observation_mask: tf.Tensor,
    jitter: tf.Tensor | float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """QR square-root log likelihood, score, and Hessian with static masks.

    Missing rows follow the value backend's dummy-row convention: the masked
    innovation is zero, the masked observation covariance is the identity in
    missing coordinates, and the independent dummy normalizer is subtracted
    from the reported log likelihood.
    """

    dtype = _score_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        d_initial_state_mean,
        d_initial_state_covariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
        d2_initial_state_mean,
        d2_initial_state_covariance,
        d2_transition_offset,
        d2_transition_matrix,
        d2_transition_covariance,
        d2_observation_offset,
        d2_observation_matrix,
        d2_observation_covariance,
        jitter,
        context="masked QR score Hessian inputs",
    )
    y = _as_observation_matrix(observations, dtype)
    n_timesteps = _static_dim(y, 0, "observation length")
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

    dmean = _to_tensor(d_initial_state_mean, dtype)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance, dtype)
    d_transition_offset = _to_tensor(d_transition_offset, dtype)
    d_transition_matrix = _to_tensor(d_transition_matrix, dtype)
    d_transition_covariance = _to_tensor(d_transition_covariance, dtype)
    d_observation_offset = _to_tensor(d_observation_offset, dtype)
    d_observation_matrix = _to_tensor(d_observation_matrix, dtype)
    d_observation_covariance = _to_tensor(d_observation_covariance, dtype)
    ddmean = _to_tensor(d2_initial_state_mean, dtype)
    d2_initial_state_covariance = _to_tensor(d2_initial_state_covariance, dtype)
    d2_transition_offset = _to_tensor(d2_transition_offset, dtype)
    d2_transition_matrix = _to_tensor(d2_transition_matrix, dtype)
    d2_transition_covariance = _to_tensor(d2_transition_covariance, dtype)
    d2_observation_offset = _to_tensor(d2_observation_offset, dtype)
    d2_observation_matrix = _to_tensor(d2_observation_matrix, dtype)
    d2_observation_covariance = _to_tensor(d2_observation_covariance, dtype)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    jitter_tensor = _to_tensor(jitter, dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=dtype)

    covariance_factor, dcovariance_factor, d2covariance_factor = (
        cholesky_factor_derivatives(
            initial_state_covariance,
            d_initial_state_covariance,
            d2_initial_state_covariance,
            jitter=0.0,
        )
    )
    transition_covariance_factor, dtransition_covariance_factor, d2transition_covariance_factor = (
        cholesky_factor_derivatives(
            transition_covariance,
            d_transition_covariance,
            d2_transition_covariance,
            jitter=0.0,
        )
    )

    score = tf.zeros((parameter_dim,), dtype=dtype)
    hessian = tf.zeros((parameter_dim, parameter_dim), dtype=dtype)
    log_likelihood = tf.constant(0.0, dtype=dtype)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        dpredicted_mean_values = []
        d2predicted_mean_rows = []
        for i in range(parameter_dim):
            dpredicted_mean_i = (
                d_transition_offset[i]
                + _matvec(d_transition_matrix[i], mean)
                + _matvec(transition_matrix, dmean[i])
            )
            dpredicted_mean_values.append(dpredicted_mean_i)
            d2predicted_mean_values = []
            for j in range(parameter_dim):
                d2predicted_mean_values.append(
                    d2_transition_offset[i, j]
                    + _matvec(d2_transition_matrix[i, j], mean)
                    + _matvec(d_transition_matrix[i], dmean[j])
                    + _matvec(d_transition_matrix[j], dmean[i])
                    + _matvec(transition_matrix, ddmean[i, j])
                )
            d2predicted_mean_rows.append(tf.stack(d2predicted_mean_values, axis=0))
        dpredicted_mean = tf.stack(dpredicted_mean_values, axis=0)
        d2predicted_mean = tf.stack(d2predicted_mean_rows, axis=0)

        prediction_left = transition_matrix @ covariance_factor
        prediction_stack = tf.concat(
            (prediction_left, transition_covariance_factor),
            axis=1,
        )
        dprediction_stack_values = []
        d2prediction_stack_rows = []
        for i in range(parameter_dim):
            dprediction_stack_values.append(
                tf.concat(
                    (
                        d_transition_matrix[i] @ covariance_factor
                        + transition_matrix @ dcovariance_factor[i],
                        dtransition_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2prediction_stack_values = []
            for j in range(parameter_dim):
                d2prediction_stack_values.append(
                    tf.concat(
                        (
                            d2_transition_matrix[i, j] @ covariance_factor
                            + d_transition_matrix[i] @ dcovariance_factor[j]
                            + d_transition_matrix[j] @ dcovariance_factor[i]
                            + transition_matrix @ d2covariance_factor[i, j],
                            d2transition_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2prediction_stack_rows.append(tf.stack(d2prediction_stack_values, axis=0))
        dprediction_stack = tf.stack(dprediction_stack_values, axis=0)
        d2prediction_stack = tf.stack(d2prediction_stack_rows, axis=0)
        predicted_factor, dpredicted_factor, d2predicted_factor, _ = (
            stack_qr_lower_factor_derivatives(
                prediction_stack,
                dprediction_stack,
                d2prediction_stack,
            )
        )
        predicted_covariance, dpredicted_covariance, d2predicted_covariance = (
            factor_covariance_derivatives(
                predicted_factor,
                dpredicted_factor,
                d2predicted_factor,
            )
        )

        row_weight = tf.cast(observation_mask[t], dtype)
        missing_weight = 1.0 - row_weight
        row_outer = row_weight[:, tf.newaxis] * row_weight[tf.newaxis, :]
        masked_observation_matrix = observation_matrix * row_weight[:, tf.newaxis]
        d_masked_observation_matrix = (
            d_observation_matrix * row_weight[tf.newaxis, :, tf.newaxis]
        )
        d2_masked_observation_matrix = (
            d2_observation_matrix * row_weight[tf.newaxis, tf.newaxis, :, tf.newaxis]
        )
        base_observation_covariance = observation_covariance + jitter_tensor * obs_identity
        masked_observation_covariance = (
            base_observation_covariance * row_outer + tf.linalg.diag(missing_weight)
        )
        d_masked_observation_covariance = (
            d_observation_covariance * row_outer[tf.newaxis, :, :]
        )
        d2_masked_observation_covariance = (
            d2_observation_covariance * row_outer[tf.newaxis, tf.newaxis, :, :]
        )
        observation_covariance_factor, dobservation_covariance_factor, d2observation_covariance_factor = (
            cholesky_factor_derivatives(
                masked_observation_covariance,
                d_masked_observation_covariance,
                d2_masked_observation_covariance,
                jitter=0.0,
            )
        )

        innovation = (
            y[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))
        ) * row_weight
        dinnovation_values = []
        d2innovation_rows = []
        for i in range(parameter_dim):
            dinnovation_values.append(
                (
                    -d_observation_offset[i]
                    - _matvec(d_observation_matrix[i], predicted_mean)
                    - _matvec(observation_matrix, dpredicted_mean[i])
                )
                * row_weight
            )
            d2innovation_values = []
            for j in range(parameter_dim):
                d2innovation_values.append(
                    (
                        -d2_observation_offset[i, j]
                        - _matvec(d2_observation_matrix[i, j], predicted_mean)
                        - _matvec(d_observation_matrix[i], dpredicted_mean[j])
                        - _matvec(d_observation_matrix[j], dpredicted_mean[i])
                        - _matvec(observation_matrix, d2predicted_mean[i, j])
                    )
                    * row_weight
                )
            d2innovation_rows.append(tf.stack(d2innovation_values, axis=0))
        dinnovation = tf.stack(dinnovation_values, axis=0)
        d2innovation = tf.stack(d2innovation_rows, axis=0)

        innovation_left = masked_observation_matrix @ predicted_factor
        innovation_stack = tf.concat(
            (innovation_left, observation_covariance_factor),
            axis=1,
        )
        dinnovation_stack_values = []
        d2innovation_stack_rows = []
        for i in range(parameter_dim):
            dinnovation_stack_values.append(
                tf.concat(
                    (
                        d_masked_observation_matrix[i] @ predicted_factor
                        + masked_observation_matrix @ dpredicted_factor[i],
                        dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2innovation_stack_values = []
            for j in range(parameter_dim):
                d2innovation_stack_values.append(
                    tf.concat(
                        (
                            d2_masked_observation_matrix[i, j] @ predicted_factor
                            + d_masked_observation_matrix[i] @ dpredicted_factor[j]
                            + d_masked_observation_matrix[j] @ dpredicted_factor[i]
                            + masked_observation_matrix @ d2predicted_factor[i, j],
                            d2observation_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2innovation_stack_rows.append(tf.stack(d2innovation_stack_values, axis=0))
        dinnovation_stack = tf.stack(dinnovation_stack_values, axis=0)
        d2innovation_stack = tf.stack(d2innovation_stack_rows, axis=0)
        innovation_factor, dinnovation_factor, d2innovation_factor, _ = (
            stack_qr_lower_factor_derivatives(
                innovation_stack,
                dinnovation_stack,
                d2innovation_stack,
            )
        )
        _, dS, d2S = factor_covariance_derivatives(
            innovation_factor,
            dinnovation_factor,
            d2innovation_factor,
        )
        innovation_solve = factor_solve(innovation_factor, innovation)
        innovation_precision = factor_solve(innovation_factor, obs_identity)

        dsolve_innovation_values = []
        dSinv_values = []
        score_contrib_values = []
        for i in range(parameter_dim):
            dsolve_i = factor_solve(
                innovation_factor,
                dinnovation[i] - _matvec(dS[i], innovation_solve),
            )
            dSinv_i = -factor_solve(innovation_factor, dS[i] @ innovation_precision)
            score_i = -0.5 * (
                trace_factor_solve(innovation_factor, dS[i])
                + 2.0 * tf.tensordot(dinnovation[i], innovation_solve, axes=1)
                - tf.tensordot(
                    innovation_solve,
                    _matvec(dS[i], innovation_solve),
                    axes=1,
                )
            )
            dsolve_innovation_values.append(dsolve_i)
            dSinv_values.append(dSinv_i)
            score_contrib_values.append(score_i)
        dsolve_innovation = tf.stack(dsolve_innovation_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        score_contrib = tf.stack(score_contrib_values, axis=0)
        score = score + score_contrib

        d2Sinv_rows = []
        hessian_contrib_rows = []
        for i in range(parameter_dim):
            d2Sinv_values = []
            hessian_contrib_values = []
            for j in range(parameter_dim):
                d2Sinv_ij = factor_solve(
                    innovation_factor,
                    dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - d2S[i, j] @ innovation_precision,
                )
                trace_term = tf.linalg.trace(
                    dSinv[j] @ dS[i] + innovation_precision @ d2S[i, j]
                )
                quad_term = (
                    2.0 * tf.tensordot(d2innovation[i, j], innovation_solve, axes=1)
                    + 2.0
                    * tf.tensordot(dinnovation[i], dsolve_innovation[j], axes=1)
                )
                curvature_term = (
                    tf.tensordot(
                        dinnovation[j],
                        _matvec(innovation_precision @ dS[i], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(dSinv[j] @ dS[i], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(innovation_precision @ d2S[i, j], innovation_solve),
                        axes=1,
                    )
                    + tf.tensordot(
                        innovation,
                        _matvec(innovation_precision @ dS[i], dsolve_innovation[j]),
                        axes=1,
                    )
                )
                hessian_contrib_values.append(
                    -0.5 * (trace_term + quad_term - curvature_term)
                )
                d2Sinv_values.append(d2Sinv_ij)
            d2Sinv_rows.append(tf.stack(d2Sinv_values, axis=0))
            hessian_contrib_rows.append(tf.stack(hessian_contrib_values, axis=0))
        d2Sinv = tf.stack(d2Sinv_rows, axis=0)
        hessian_contrib = tf.stack(hessian_contrib_rows, axis=0)
        hessian = hessian + hessian_contrib

        kalman_gain = (
            predicted_covariance
            @ tf.transpose(masked_observation_matrix)
            @ innovation_precision
        )
        dK_values = []
        d2K_rows = []
        for i in range(parameter_dim):
            dK_i = (
                dpredicted_covariance[i]
                @ tf.transpose(masked_observation_matrix)
                @ innovation_precision
                + predicted_covariance
                @ tf.transpose(d_masked_observation_matrix[i])
                @ innovation_precision
                + predicted_covariance @ tf.transpose(masked_observation_matrix) @ dSinv[i]
            )
            dK_values.append(dK_i)
            d2K_values = []
            for j in range(parameter_dim):
                d2K_values.append(
                    d2predicted_covariance[i, j]
                    @ tf.transpose(masked_observation_matrix)
                    @ innovation_precision
                    + dpredicted_covariance[i]
                    @ tf.transpose(d_masked_observation_matrix[j])
                    @ innovation_precision
                    + dpredicted_covariance[i]
                    @ tf.transpose(masked_observation_matrix)
                    @ dSinv[j]
                    + dpredicted_covariance[j]
                    @ tf.transpose(d_masked_observation_matrix[i])
                    @ innovation_precision
                    + predicted_covariance
                    @ tf.transpose(d2_masked_observation_matrix[i, j])
                    @ innovation_precision
                    + predicted_covariance
                    @ tf.transpose(d_masked_observation_matrix[i])
                    @ dSinv[j]
                    + dpredicted_covariance[j]
                    @ tf.transpose(masked_observation_matrix)
                    @ dSinv[i]
                    + predicted_covariance
                    @ tf.transpose(d_masked_observation_matrix[j])
                    @ dSinv[i]
                    + predicted_covariance
                    @ tf.transpose(masked_observation_matrix)
                    @ d2Sinv[i, j]
                )
            d2K_rows.append(tf.stack(d2K_values, axis=0))
        dK = tf.stack(dK_values, axis=0)
        d2K = tf.stack(d2K_rows, axis=0)

        joseph_left = state_identity - kalman_gain @ masked_observation_matrix
        d_joseph_left_values = []
        d2_joseph_left_rows = []
        for i in range(parameter_dim):
            d_joseph_left_values.append(
                -dK[i] @ masked_observation_matrix
                - kalman_gain @ d_masked_observation_matrix[i]
            )
            d2_joseph_left_values = []
            for j in range(parameter_dim):
                d2_joseph_left_values.append(
                    -d2K[i, j] @ masked_observation_matrix
                    - dK[i] @ d_masked_observation_matrix[j]
                    - dK[j] @ d_masked_observation_matrix[i]
                    - kalman_gain @ d2_masked_observation_matrix[i, j]
                )
            d2_joseph_left_rows.append(tf.stack(d2_joseph_left_values, axis=0))
        d_joseph_left = tf.stack(d_joseph_left_values, axis=0)
        d2_joseph_left = tf.stack(d2_joseph_left_rows, axis=0)

        update_left = joseph_left @ predicted_factor
        update_right = kalman_gain @ observation_covariance_factor
        update_stack = tf.concat((update_left, update_right), axis=1)
        dupdate_stack_values = []
        d2update_stack_rows = []
        for i in range(parameter_dim):
            dupdate_stack_values.append(
                tf.concat(
                    (
                        d_joseph_left[i] @ predicted_factor
                        + joseph_left @ dpredicted_factor[i],
                        dK[i] @ observation_covariance_factor
                        + kalman_gain @ dobservation_covariance_factor[i],
                    ),
                    axis=1,
                )
            )
            d2update_stack_values = []
            for j in range(parameter_dim):
                d2update_stack_values.append(
                    tf.concat(
                        (
                            d2_joseph_left[i, j] @ predicted_factor
                            + d_joseph_left[i] @ dpredicted_factor[j]
                            + d_joseph_left[j] @ dpredicted_factor[i]
                            + joseph_left @ d2predicted_factor[i, j],
                            d2K[i, j] @ observation_covariance_factor
                            + dK[i] @ dobservation_covariance_factor[j]
                            + dK[j] @ dobservation_covariance_factor[i]
                            + kalman_gain @ d2observation_covariance_factor[i, j],
                        ),
                        axis=1,
                    )
                )
            d2update_stack_rows.append(tf.stack(d2update_stack_values, axis=0))
        dupdate_stack = tf.stack(dupdate_stack_values, axis=0)
        d2update_stack = tf.stack(d2update_stack_rows, axis=0)
        covariance_factor, dcovariance_factor, d2covariance_factor, _ = (
            stack_qr_lower_factor_derivatives(
                update_stack,
                dupdate_stack,
                d2update_stack,
            )
        )

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        dmean_values = []
        ddmean_rows = []
        for i in range(parameter_dim):
            dmean_values.append(
                dpredicted_mean[i]
                + _matvec(dK[i], innovation)
                + _matvec(kalman_gain, dinnovation[i])
            )
            ddmean_values = []
            for j in range(parameter_dim):
                ddmean_values.append(
                    d2predicted_mean[i, j]
                    + _matvec(d2K[i, j], innovation)
                    + _matvec(dK[i], dinnovation[j])
                    + _matvec(dK[j], dinnovation[i])
                    + _matvec(kalman_gain, d2innovation[i, j])
                )
            ddmean_rows.append(tf.stack(ddmean_values, axis=0))
        dmean = tf.stack(dmean_values, axis=0)
        ddmean = tf.stack(ddmean_rows, axis=0)

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
        log_likelihood = log_likelihood + contribution

    return log_likelihood, score, 0.5 * (hessian + tf.transpose(hessian))


def _metadata(
    *,
    filter_name: str,
    model: TFLinearGaussianStateSpace,
    differentiability_status: str = "analytic_score_hessian",
) -> FilterRunMetadata:
    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status=differentiability_status,
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


def _require_time_invariant_score_inputs(model: TFLinearGaussianStateSpace) -> None:
    """Reject per-time state-space tensors in the dynamic public score backend."""

    for name in (
        "transition_offset",
        "transition_matrix",
        "transition_covariance",
        "observation_offset",
        "observation_matrix",
        "observation_covariance",
    ):
        tensor = getattr(model, name)
        if tensor.shape.rank not in (1, 2):
            raise ValueError(
                "tf_qr_linear_gaussian_score currently supports time-invariant "
                f"state-space tensors; {name} has rank {tensor.shape.rank}"
            )


@tf.function(reduce_retracing=True)
def tf_qr_sqrt_kalman_score(
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
    """Dynamic-time QR square-root Kalman log likelihood and analytic score.

    This public score-only kernel avoids Python unrolling over the observation
    horizon. It intentionally does not return a Hessian; public Hessian
    authority remains a separate contract.
    """

    dtype = _score_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        d_initial_state_mean,
        d_initial_state_covariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
        jitter,
        context="dynamic QR score inputs",
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

    dmean0 = _to_tensor(d_initial_state_mean, dtype)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance, dtype)
    d_transition_offset = _to_tensor(d_transition_offset, dtype)
    d_transition_matrix = _to_tensor(d_transition_matrix, dtype)
    d_transition_covariance = _to_tensor(d_transition_covariance, dtype)
    d_observation_offset = _to_tensor(d_observation_offset, dtype)
    d_observation_matrix = _to_tensor(d_observation_matrix, dtype)
    d_observation_covariance = _to_tensor(d_observation_covariance, dtype)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean0)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=dtype)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    jitter_tensor = _to_tensor(jitter, dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    covariance_factor0, dcovariance_factor0 = cholesky_factor_first_derivatives(
        initial_state_covariance,
        d_initial_state_covariance,
        jitter=0.0,
    )
    transition_covariance_factor, dtransition_covariance_factor = (
        cholesky_factor_first_derivatives(
            transition_covariance,
            d_transition_covariance,
            jitter=0.0,
        )
    )
    observation_covariance_factor, dobservation_covariance_factor = (
        cholesky_factor_first_derivatives(
            observation_covariance + jitter_tensor * obs_identity,
            d_observation_covariance,
            jitter=0.0,
        )
    )
    if jitter_updates_filtered_covariance:
        observation_update_covariance_factor = observation_covariance_factor
        dobservation_update_covariance_factor = dobservation_covariance_factor
    else:
        observation_update_covariance_factor, dobservation_update_covariance_factor = (
            cholesky_factor_first_derivatives(
                observation_covariance,
                d_observation_covariance,
                jitter=0.0,
            )
        )

    score0 = tf.zeros((parameter_dim,), dtype=dtype)
    log_likelihood0 = tf.constant(0.0, dtype=dtype)
    t0 = tf.constant(0, dtype=tf.int32)

    def cond(t, *_state):
        return t < n_timesteps

    def body(
        t,
        mean,
        dmean,
        covariance_factor,
        dcovariance_factor,
        log_likelihood,
        score,
    ):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        dpredicted_mean = (
            d_transition_offset
            + tf.einsum("pij,j->pi", d_transition_matrix, mean)
            + tf.einsum("ij,pj->pi", transition_matrix, dmean)
        )

        prediction_left = transition_matrix @ covariance_factor
        prediction_stack = tf.concat(
            (prediction_left, transition_covariance_factor),
            axis=1,
        )
        dprediction_left = (
            tf.einsum("pij,jk->pik", d_transition_matrix, covariance_factor)
            + tf.einsum("ij,pjk->pik", transition_matrix, dcovariance_factor)
        )
        dprediction_stack = tf.concat(
            (dprediction_left, dtransition_covariance_factor),
            axis=2,
        )
        predicted_factor, dpredicted_factor, _ = (
            stack_qr_lower_factor_first_derivatives(
                prediction_stack,
                dprediction_stack,
            )
        )
        predicted_covariance, dpredicted_covariance = (
            factor_covariance_first_derivatives(
                predicted_factor,
                dpredicted_factor,
            )
        )

        innovation = y[t] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        dinnovation = (
            -d_observation_offset
            - tf.einsum("pij,j->pi", d_observation_matrix, predicted_mean)
            - tf.einsum("ij,pj->pi", observation_matrix, dpredicted_mean)
        )

        innovation_left = observation_matrix @ predicted_factor
        innovation_stack = tf.concat(
            (innovation_left, observation_covariance_factor),
            axis=1,
        )
        dinnovation_left = (
            tf.einsum("pij,jk->pik", d_observation_matrix, predicted_factor)
            + tf.einsum("ij,pjk->pik", observation_matrix, dpredicted_factor)
        )
        dinnovation_stack = tf.concat(
            (dinnovation_left, dobservation_covariance_factor),
            axis=2,
        )
        innovation_factor, dinnovation_factor, _ = (
            stack_qr_lower_factor_first_derivatives(
                innovation_stack,
                dinnovation_stack,
            )
        )
        _, dS = factor_covariance_first_derivatives(
            innovation_factor,
            dinnovation_factor,
        )
        innovation_solve = factor_solve(innovation_factor, innovation)
        innovation_precision = factor_solve(innovation_factor, obs_identity)

        trace_terms = tf.einsum("ab,pba->p", innovation_precision, dS)
        innovation_derivative_terms = tf.einsum(
            "pi,i->p",
            dinnovation,
            innovation_solve,
        )
        quadratic_terms = tf.einsum(
            "i,pij,j->p",
            innovation_solve,
            dS,
            innovation_solve,
        )
        score = score - 0.5 * (
            trace_terms + 2.0 * innovation_derivative_terms - quadratic_terms
        )
        dSinv = -tf.einsum(
            "ab,pbc,cd->pad",
            innovation_precision,
            dS,
            innovation_precision,
        )

        kalman_gain = (
            predicted_covariance
            @ tf.transpose(observation_matrix)
            @ innovation_precision
        )
        dK = (
            tf.einsum(
                "pij,kj,kl->pil",
                dpredicted_covariance,
                observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "ij,pkj,kl->pil",
                predicted_covariance,
                d_observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "ij,kj,pkl->pil",
                predicted_covariance,
                observation_matrix,
                dSinv,
            )
        )

        joseph_left = state_identity - kalman_gain @ observation_matrix
        d_joseph_left = (
            -tf.einsum("pij,jk->pik", dK, observation_matrix)
            - tf.einsum("ij,pjk->pik", kalman_gain, d_observation_matrix)
        )
        update_left = joseph_left @ predicted_factor
        update_right = kalman_gain @ observation_update_covariance_factor
        update_stack = tf.concat((update_left, update_right), axis=1)
        dupdate_left = (
            tf.einsum("pij,jk->pik", d_joseph_left, predicted_factor)
            + tf.einsum("ij,pjk->pik", joseph_left, dpredicted_factor)
        )
        dupdate_right = (
            tf.einsum("pij,jk->pik", dK, observation_update_covariance_factor)
            + tf.einsum(
                "ij,pjk->pik",
                kalman_gain,
                dobservation_update_covariance_factor,
            )
        )
        dupdate_stack = tf.concat((dupdate_left, dupdate_right), axis=2)
        covariance_factor, dcovariance_factor, _ = (
            stack_qr_lower_factor_first_derivatives(
                update_stack,
                dupdate_stack,
            )
        )

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        dmean = (
            dpredicted_mean
            + tf.einsum("pij,j->pi", dK, innovation)
            + tf.einsum("ij,pj->pi", kalman_gain, dinnovation)
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
        log_likelihood = log_likelihood + contribution
        return (
            t + 1,
            mean,
            dmean,
            covariance_factor,
            dcovariance_factor,
            log_likelihood,
            score,
        )

    (
        _,
        _mean,
        _dmean,
        _covariance_factor,
        _dcovariance_factor,
        log_likelihood,
        score,
    ) = tf.while_loop(
        cond,
        body,
        (
            t0,
            mean0,
            dmean0,
            covariance_factor0,
            dcovariance_factor0,
            log_likelihood0,
            score0,
        ),
        maximum_iterations=n_timesteps,
        parallel_iterations=1,
    )
    return log_likelihood, score


@tf.function(jit_compile=True, reduce_retracing=True)
def tf_qr_sqrt_kalman_score_batched_static(
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
    """Batched-static QR square-root Kalman log likelihood and score.

    The leading dimension is an independent batch/proposal axis ``B``.  The
    derivative tensors carry a separate parameter axis ``P``.  This kernel is
    batch-native over ``B`` and does not call the scalar score kernel per row.
    """

    dtype = _score_dtype(
        observations,
        transition_offset,
        transition_matrix,
        transition_covariance,
        observation_offset,
        observation_matrix,
        observation_covariance,
        initial_state_mean,
        initial_state_covariance,
        d_initial_state_mean,
        d_initial_state_covariance,
        d_transition_offset,
        d_transition_matrix,
        d_transition_covariance,
        d_observation_offset,
        d_observation_matrix,
        d_observation_covariance,
        jitter,
        context="batched-static QR score inputs",
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

    dmean0 = _to_tensor(d_initial_state_mean, dtype)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance, dtype)
    d_transition_offset = _to_tensor(d_transition_offset, dtype)
    d_transition_matrix = _to_tensor(d_transition_matrix, dtype)
    d_transition_covariance = _to_tensor(d_transition_covariance, dtype)
    d_observation_offset = _to_tensor(d_observation_offset, dtype)
    d_observation_matrix = _to_tensor(d_observation_matrix, dtype)
    d_observation_covariance = _to_tensor(d_observation_covariance, dtype)

    _validate_batched_static_score_shapes(
        transition_offset=transition_offset,
        transition_matrix=transition_matrix,
        transition_covariance=transition_covariance,
        observation_offset=observation_offset,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_state_mean=mean0,
        initial_state_covariance=initial_state_covariance,
        d_initial_state_mean=dmean0,
        d_initial_state_covariance=d_initial_state_covariance,
        d_transition_offset=d_transition_offset,
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=d_observation_offset,
        d_observation_matrix=d_observation_matrix,
        d_observation_covariance=d_observation_covariance,
    )

    batch_size = tf.shape(mean0)[0]
    parameter_dim = tf.shape(dmean0)[1]
    state_dim = tf.shape(mean0)[1]
    obs_dim = tf.shape(observation_offset)[1]
    state_identity = tf.eye(state_dim, batch_shape=[batch_size], dtype=dtype)
    obs_identity = tf.eye(obs_dim, batch_shape=[batch_size], dtype=dtype)
    jitter_tensor = _to_tensor(jitter, dtype)
    two_pi = tf.constant(2.0 * math.pi, dtype=dtype)

    covariance_factor0, dcovariance_factor0 = _batched_cholesky_factor_first_derivatives(
        initial_state_covariance,
        d_initial_state_covariance,
        jitter=0.0,
    )
    transition_covariance_factor, dtransition_covariance_factor = (
        _batched_cholesky_factor_first_derivatives(
            transition_covariance,
            d_transition_covariance,
            jitter=0.0,
        )
    )
    observation_covariance_factor, dobservation_covariance_factor = (
        _batched_cholesky_factor_first_derivatives(
            observation_covariance + jitter_tensor * obs_identity,
            d_observation_covariance,
            jitter=0.0,
        )
    )
    if jitter_updates_filtered_covariance:
        observation_update_covariance_factor = observation_covariance_factor
        dobservation_update_covariance_factor = dobservation_covariance_factor
    else:
        observation_update_covariance_factor, dobservation_update_covariance_factor = (
            _batched_cholesky_factor_first_derivatives(
                observation_covariance,
                d_observation_covariance,
                jitter=0.0,
            )
        )

    score0 = tf.zeros((batch_size, parameter_dim), dtype=dtype)
    log_likelihood0 = tf.zeros((batch_size,), dtype=dtype)
    t0 = tf.constant(0, dtype=tf.int32)

    def cond(t, *_state):
        return t < n_timesteps

    def body(
        t,
        mean,
        dmean,
        covariance_factor,
        dcovariance_factor,
        log_likelihood,
        score,
    ):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        dpredicted_mean = (
            d_transition_offset
            + tf.einsum("bpij,bj->bpi", d_transition_matrix, mean)
            + tf.einsum("bij,bpj->bpi", transition_matrix, dmean)
        )

        prediction_left = transition_matrix @ covariance_factor
        prediction_stack = tf.concat(
            (prediction_left, transition_covariance_factor),
            axis=2,
        )
        dprediction_left = (
            tf.einsum("bpij,bjk->bpik", d_transition_matrix, covariance_factor)
            + tf.einsum("bij,bpjk->bpik", transition_matrix, dcovariance_factor)
        )
        dprediction_stack = tf.concat(
            (dprediction_left, dtransition_covariance_factor),
            axis=3,
        )
        predicted_factor, dpredicted_factor, _ = (
            _batched_stack_qr_lower_factor_first_derivatives(
                prediction_stack,
                dprediction_stack,
            )
        )
        predicted_covariance, dpredicted_covariance = (
            _batched_factor_covariance_first_derivatives(
                predicted_factor,
                dpredicted_factor,
            )
        )

        innovation = y[t][tf.newaxis, :] - (
            observation_offset + _matvec(observation_matrix, predicted_mean)
        )
        dinnovation = (
            -d_observation_offset
            - tf.einsum("bpij,bj->bpi", d_observation_matrix, predicted_mean)
            - tf.einsum("bij,bpj->bpi", observation_matrix, dpredicted_mean)
        )

        innovation_left = observation_matrix @ predicted_factor
        innovation_stack = tf.concat(
            (innovation_left, observation_covariance_factor),
            axis=2,
        )
        dinnovation_left = (
            tf.einsum("bpij,bjk->bpik", d_observation_matrix, predicted_factor)
            + tf.einsum("bij,bpjk->bpik", observation_matrix, dpredicted_factor)
        )
        dinnovation_stack = tf.concat(
            (dinnovation_left, dobservation_covariance_factor),
            axis=3,
        )
        innovation_factor, dinnovation_factor, _ = (
            _batched_stack_qr_lower_factor_first_derivatives(
                innovation_stack,
                dinnovation_stack,
            )
        )
        _, dS = _batched_factor_covariance_first_derivatives(
            innovation_factor,
            dinnovation_factor,
        )
        innovation_solve = _batched_factor_solve(innovation_factor, innovation)
        innovation_precision = _batched_factor_solve(innovation_factor, obs_identity)

        trace_terms = tf.einsum("bij,bpji->bp", innovation_precision, dS)
        innovation_derivative_terms = tf.einsum(
            "bpi,bi->bp",
            dinnovation,
            innovation_solve,
        )
        quadratic_terms = tf.einsum(
            "bi,bpij,bj->bp",
            innovation_solve,
            dS,
            innovation_solve,
        )
        score = score - 0.5 * (
            trace_terms + 2.0 * innovation_derivative_terms - quadratic_terms
        )
        dSinv = -tf.einsum(
            "bij,bpjk,bkl->bpil",
            innovation_precision,
            dS,
            innovation_precision,
        )

        kalman_gain = (
            predicted_covariance
            @ _matrix_transpose(observation_matrix)
            @ innovation_precision
        )
        dK = (
            tf.einsum(
                "bpij,bkj,bkl->bpil",
                dpredicted_covariance,
                observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "bij,bpkj,bkl->bpil",
                predicted_covariance,
                d_observation_matrix,
                innovation_precision,
            )
            + tf.einsum(
                "bij,bkj,bpkl->bpil",
                predicted_covariance,
                observation_matrix,
                dSinv,
            )
        )

        joseph_left = state_identity - kalman_gain @ observation_matrix
        d_joseph_left = (
            -tf.einsum("bpij,bjk->bpik", dK, observation_matrix)
            - tf.einsum("bij,bpjk->bpik", kalman_gain, d_observation_matrix)
        )
        update_left = joseph_left @ predicted_factor
        update_right = kalman_gain @ observation_update_covariance_factor
        update_stack = tf.concat((update_left, update_right), axis=2)
        dupdate_left = (
            tf.einsum("bpij,bjk->bpik", d_joseph_left, predicted_factor)
            + tf.einsum("bij,bpjk->bpik", joseph_left, dpredicted_factor)
        )
        dupdate_right = (
            tf.einsum("bpij,bjk->bpik", dK, observation_update_covariance_factor)
            + tf.einsum(
                "bij,bpjk->bpik",
                kalman_gain,
                dobservation_update_covariance_factor,
            )
        )
        dupdate_stack = tf.concat((dupdate_left, dupdate_right), axis=3)
        covariance_factor, dcovariance_factor, _ = (
            _batched_stack_qr_lower_factor_first_derivatives(
                update_stack,
                dupdate_stack,
            )
        )

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        dmean = (
            dpredicted_mean
            + tf.einsum("bpij,bj->bpi", dK, innovation)
            + tf.einsum("bij,bpj->bpi", kalman_gain, dinnovation)
        )

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
            tf.cast(obs_dim, dtype) * tf.math.log(two_pi)
            + log_det
            + mahalanobis
        )
        log_likelihood = log_likelihood + contribution
        return (
            t + 1,
            mean,
            dmean,
            covariance_factor,
            dcovariance_factor,
            log_likelihood,
            score,
        )

    (
        _,
        _mean,
        _dmean,
        _covariance_factor,
        _dcovariance_factor,
        log_likelihood,
        score,
    ) = tf.while_loop(
        cond,
        body,
        (
            t0,
            mean0,
            dmean0,
            covariance_factor0,
            dcovariance_factor0,
            log_likelihood0,
            score0,
        ),
        maximum_iterations=n_timesteps,
        parallel_iterations=1,
    )
    return log_likelihood, score


def tf_qr_linear_gaussian_score(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    derivatives: TFLinearGaussianStateSpaceDerivatives,
    *,
    backend: Literal["tf_qr_sqrt_score"] = "tf_qr_sqrt_score",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
    jitter_updates_filtered_covariance: bool = True,
) -> TFFilterDerivativeResult:
    """Return public QR/square-root linear Gaussian likelihood and score.

    This score-only API is the public authority surface for graph-native HMC
    value/score use. It uses a dynamic TensorFlow loop over time and does not
    certify Hessian authority.
    """

    if backend != "tf_qr_sqrt_score":
        raise ValueError(f"unknown TensorFlow QR score backend: {backend}")
    mask = observation_mask if observation_mask is not None else model.observation_mask
    if mask is not None:
        raise ValueError("tf_qr_linear_gaussian_score supports dense observations only")
    _require_time_invariant_score_inputs(model)
    dtype = _score_dtype(
        observations,
        model.transition_offset,
        model.transition_matrix,
        model.transition_covariance,
        model.observation_offset,
        model.observation_matrix,
        model.observation_covariance,
        model.initial_mean,
        model.initial_covariance,
        derivatives.d_initial_mean,
        derivatives.d_initial_covariance,
        derivatives.d_transition_offset,
        derivatives.d_transition_matrix,
        derivatives.d_transition_covariance,
        derivatives.d_observation_offset,
        derivatives.d_observation_matrix,
        derivatives.d_observation_covariance,
        jitter,
        context="TensorFlow QR score dispatcher inputs",
    )
    log_likelihood, score = tf_qr_sqrt_kalman_score(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        d_initial_state_mean=derivatives.d_initial_mean,
        d_initial_state_covariance=derivatives.d_initial_covariance,
        d_transition_offset=derivatives.d_transition_offset,
        d_transition_matrix=derivatives.d_transition_matrix,
        d_transition_covariance=derivatives.d_transition_covariance,
        d_observation_offset=derivatives.d_observation_offset,
        d_observation_matrix=derivatives.d_observation_matrix,
        d_observation_covariance=derivatives.d_observation_covariance,
        jitter=jitter,
        jitter_updates_filtered_covariance=bool(jitter_updates_filtered_covariance),
    )
    return TFFilterDerivativeResult(
        log_likelihood=log_likelihood,
        score=score,
        hessian=None,
        metadata=_metadata(
            filter_name="tf_qr_sqrt_score_dynamic_kalman",
            model=model,
            differentiability_status="analytic_score_no_hessian",
        ),
        diagnostics=_diagnostics(
            backend=backend,
            mask_convention="none",
            jitter=jitter,
            dtype=dtype,
        ),
    )


def tf_qr_linear_gaussian_score_hessian(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    derivatives: TFLinearGaussianStateSpaceDerivatives,
    *,
    backend: TFQRLinearDerivativeBackend = "tf_qr_sqrt",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Dispatch to the QR/square-root analytic linear Gaussian derivative backend."""

    mask = observation_mask if observation_mask is not None else model.observation_mask
    dtype = _score_dtype(
        observations,
        model.transition_offset,
        model.transition_matrix,
        model.transition_covariance,
        model.observation_offset,
        model.observation_matrix,
        model.observation_covariance,
        model.initial_mean,
        model.initial_covariance,
        derivatives.d_initial_mean,
        derivatives.d_initial_covariance,
        derivatives.d_transition_offset,
        derivatives.d_transition_matrix,
        derivatives.d_transition_covariance,
        derivatives.d_observation_offset,
        derivatives.d_observation_matrix,
        derivatives.d_observation_covariance,
        derivatives.d2_initial_mean,
        derivatives.d2_initial_covariance,
        derivatives.d2_transition_offset,
        derivatives.d2_transition_matrix,
        derivatives.d2_transition_covariance,
        derivatives.d2_observation_offset,
        derivatives.d2_observation_matrix,
        derivatives.d2_observation_covariance,
        jitter,
        context="TensorFlow QR score Hessian dispatcher inputs",
    )
    if backend == "tf_qr_sqrt":
        if mask is None:
            log_likelihood, score, hessian = tf_qr_sqrt_kalman_score_hessian(
                observations=observations,
                transition_offset=model.transition_offset,
                transition_matrix=model.transition_matrix,
                transition_covariance=model.transition_covariance,
                observation_offset=model.observation_offset,
                observation_matrix=model.observation_matrix,
                observation_covariance=model.observation_covariance,
                initial_state_mean=model.initial_mean,
                initial_state_covariance=model.initial_covariance,
                d_initial_state_mean=derivatives.d_initial_mean,
                d_initial_state_covariance=derivatives.d_initial_covariance,
                d_transition_offset=derivatives.d_transition_offset,
                d_transition_matrix=derivatives.d_transition_matrix,
                d_transition_covariance=derivatives.d_transition_covariance,
                d_observation_offset=derivatives.d_observation_offset,
                d_observation_matrix=derivatives.d_observation_matrix,
                d_observation_covariance=derivatives.d_observation_covariance,
                d2_initial_state_mean=derivatives.d2_initial_mean,
                d2_initial_state_covariance=derivatives.d2_initial_covariance,
                d2_transition_offset=derivatives.d2_transition_offset,
                d2_transition_matrix=derivatives.d2_transition_matrix,
                d2_transition_covariance=derivatives.d2_transition_covariance,
                d2_observation_offset=derivatives.d2_observation_offset,
                d2_observation_matrix=derivatives.d2_observation_matrix,
                d2_observation_covariance=derivatives.d2_observation_covariance,
                jitter=jitter,
            )
            filter_name = "tf_qr_sqrt_differentiated_kalman"
            mask_convention = "none"
        else:
            log_likelihood, score, hessian = tf_qr_sqrt_masked_kalman_score_hessian(
                observations=observations,
                transition_offset=model.transition_offset,
                transition_matrix=model.transition_matrix,
                transition_covariance=model.transition_covariance,
                observation_offset=model.observation_offset,
                observation_matrix=model.observation_matrix,
                observation_covariance=model.observation_covariance,
                initial_state_mean=model.initial_mean,
                initial_state_covariance=model.initial_covariance,
                d_initial_state_mean=derivatives.d_initial_mean,
                d_initial_state_covariance=derivatives.d_initial_covariance,
                d_transition_offset=derivatives.d_transition_offset,
                d_transition_matrix=derivatives.d_transition_matrix,
                d_transition_covariance=derivatives.d_transition_covariance,
                d_observation_offset=derivatives.d_observation_offset,
                d_observation_matrix=derivatives.d_observation_matrix,
                d_observation_covariance=derivatives.d_observation_covariance,
                d2_initial_state_mean=derivatives.d2_initial_mean,
                d2_initial_state_covariance=derivatives.d2_initial_covariance,
                d2_transition_offset=derivatives.d2_transition_offset,
                d2_transition_matrix=derivatives.d2_transition_matrix,
                d2_transition_covariance=derivatives.d2_transition_covariance,
                d2_observation_offset=derivatives.d2_observation_offset,
                d2_observation_matrix=derivatives.d2_observation_matrix,
                d2_observation_covariance=derivatives.d2_observation_covariance,
                observation_mask=mask,
                jitter=jitter,
            )
            filter_name = "tf_qr_sqrt_masked_differentiated_kalman"
            mask_convention = "static_dummy_row"
    elif backend == "tf_masked_qr_sqrt":
        if mask is None:
            raise ValueError("tf_masked_qr_sqrt requires an observation mask")
        log_likelihood, score, hessian = tf_qr_sqrt_masked_kalman_score_hessian(
            observations=observations,
            transition_offset=model.transition_offset,
            transition_matrix=model.transition_matrix,
            transition_covariance=model.transition_covariance,
            observation_offset=model.observation_offset,
            observation_matrix=model.observation_matrix,
            observation_covariance=model.observation_covariance,
            initial_state_mean=model.initial_mean,
            initial_state_covariance=model.initial_covariance,
            d_initial_state_mean=derivatives.d_initial_mean,
            d_initial_state_covariance=derivatives.d_initial_covariance,
            d_transition_offset=derivatives.d_transition_offset,
            d_transition_matrix=derivatives.d_transition_matrix,
            d_transition_covariance=derivatives.d_transition_covariance,
            d_observation_offset=derivatives.d_observation_offset,
            d_observation_matrix=derivatives.d_observation_matrix,
            d_observation_covariance=derivatives.d_observation_covariance,
            d2_initial_state_mean=derivatives.d2_initial_mean,
            d2_initial_state_covariance=derivatives.d2_initial_covariance,
            d2_transition_offset=derivatives.d2_transition_offset,
            d2_transition_matrix=derivatives.d2_transition_matrix,
            d2_transition_covariance=derivatives.d2_transition_covariance,
            d2_observation_offset=derivatives.d2_observation_offset,
            d2_observation_matrix=derivatives.d2_observation_matrix,
            d2_observation_covariance=derivatives.d2_observation_covariance,
            observation_mask=mask,
            jitter=jitter,
        )
        filter_name = "tf_qr_sqrt_masked_differentiated_kalman"
        mask_convention = "static_dummy_row"
    else:
        raise ValueError(f"unknown TensorFlow QR derivative backend: {backend}")
    return TFFilterDerivativeResult(
        log_likelihood=log_likelihood,
        score=score,
        hessian=hessian,
        metadata=_metadata(filter_name=filter_name, model=model),
        diagnostics=_diagnostics(
            backend=backend,
            mask_convention=mask_convention,
            jitter=jitter,
            dtype=dtype,
        ),
    )


def _tf_qr_linear_gaussian_score(
    observations: tf.Tensor,
    model: TFLinearGaussianStateSpace,
    derivatives: TFLinearGaussianStateSpaceDerivatives,
    *,
    backend: Literal["tf_qr_sqrt"] = "tf_qr_sqrt",
    observation_mask: tf.Tensor | None = None,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return the dense QR/square-root linear Gaussian likelihood and score.

    Private diagnostic helper.  The v1 public derivative surface remains the
    existing score/Hessian wrapper until a separate API-freeze decision promotes
    a score-only contract.
    """

    mask = observation_mask if observation_mask is not None else model.observation_mask
    if backend != "tf_qr_sqrt":
        raise ValueError(f"unknown TensorFlow QR score backend: {backend}")
    if mask is not None:
        raise ValueError(
            "_tf_qr_linear_gaussian_score currently supports dense observations only"
        )
    dtype = _score_dtype(
        observations,
        model.transition_offset,
        model.transition_matrix,
        model.transition_covariance,
        model.observation_offset,
        model.observation_matrix,
        model.observation_covariance,
        model.initial_mean,
        model.initial_covariance,
        derivatives.d_initial_mean,
        derivatives.d_initial_covariance,
        derivatives.d_transition_offset,
        derivatives.d_transition_matrix,
        derivatives.d_transition_covariance,
        derivatives.d_observation_offset,
        derivatives.d_observation_matrix,
        derivatives.d_observation_covariance,
        jitter,
        context="private TensorFlow QR score dispatcher inputs",
    )
    log_likelihood, score = _tf_qr_sqrt_kalman_score(
        observations=observations,
        transition_offset=model.transition_offset,
        transition_matrix=model.transition_matrix,
        transition_covariance=model.transition_covariance,
        observation_offset=model.observation_offset,
        observation_matrix=model.observation_matrix,
        observation_covariance=model.observation_covariance,
        initial_state_mean=model.initial_mean,
        initial_state_covariance=model.initial_covariance,
        d_initial_state_mean=derivatives.d_initial_mean,
        d_initial_state_covariance=derivatives.d_initial_covariance,
        d_transition_offset=derivatives.d_transition_offset,
        d_transition_matrix=derivatives.d_transition_matrix,
        d_transition_covariance=derivatives.d_transition_covariance,
        d_observation_offset=derivatives.d_observation_offset,
        d_observation_matrix=derivatives.d_observation_matrix,
        d_observation_covariance=derivatives.d_observation_covariance,
        jitter=jitter,
    )
    return TFFilterDerivativeResult(
        log_likelihood=log_likelihood,
        score=score,
        hessian=None,
        metadata=_metadata(
            filter_name="tf_qr_sqrt_score_kalman",
            model=model,
            differentiability_status="analytic_score",
        ),
        diagnostics=_diagnostics(
            backend=backend,
            mask_convention="none",
            jitter=jitter,
            dtype=dtype,
        ),
    )
