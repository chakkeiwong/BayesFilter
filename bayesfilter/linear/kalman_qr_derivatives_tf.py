"""TensorFlow QR/square-root analytic Kalman score and Hessian backends."""

from __future__ import annotations

import math
from typing import Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor_derivatives,
    factor_covariance_derivatives,
    factor_solve,
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

    y = _as_observation_matrix(observations)
    n_timesteps = _static_dim(y, 0, "observation length")
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _to_tensor(transition_covariance)
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _to_tensor(observation_covariance)
    mean = _to_tensor(initial_state_mean)
    initial_state_covariance = _to_tensor(initial_state_covariance)

    dmean = _to_tensor(d_initial_state_mean)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance)
    d_transition_offset = _to_tensor(d_transition_offset)
    d_transition_matrix = _to_tensor(d_transition_matrix)
    d_transition_covariance = _to_tensor(d_transition_covariance)
    d_observation_offset = _to_tensor(d_observation_offset)
    d_observation_matrix = _to_tensor(d_observation_matrix)
    d_observation_covariance = _to_tensor(d_observation_covariance)
    ddmean = _to_tensor(d2_initial_state_mean)
    d2_initial_state_covariance = _to_tensor(d2_initial_state_covariance)
    d2_transition_offset = _to_tensor(d2_transition_offset)
    d2_transition_matrix = _to_tensor(d2_transition_matrix)
    d2_transition_covariance = _to_tensor(d2_transition_covariance)
    d2_observation_offset = _to_tensor(d2_observation_offset)
    d2_observation_matrix = _to_tensor(d2_observation_matrix)
    d2_observation_covariance = _to_tensor(d2_observation_covariance)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    jitter_tensor = tf.cast(jitter, tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)

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

    score = tf.zeros((parameter_dim,), dtype=tf.float64)
    hessian = tf.zeros((parameter_dim, parameter_dim), dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)

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
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
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

    y = _as_observation_matrix(observations)
    n_timesteps = _static_dim(y, 0, "observation length")
    observation_mask = tf.convert_to_tensor(observation_mask, dtype=tf.bool)
    _validate_mask_shape(y, observation_mask)
    transition_offset = _to_tensor(transition_offset)
    transition_matrix = _to_tensor(transition_matrix)
    transition_covariance = _to_tensor(transition_covariance)
    observation_offset = _to_tensor(observation_offset)
    observation_matrix = _to_tensor(observation_matrix)
    observation_covariance = _to_tensor(observation_covariance)
    mean = _to_tensor(initial_state_mean)
    initial_state_covariance = _to_tensor(initial_state_covariance)

    dmean = _to_tensor(d_initial_state_mean)
    d_initial_state_covariance = _to_tensor(d_initial_state_covariance)
    d_transition_offset = _to_tensor(d_transition_offset)
    d_transition_matrix = _to_tensor(d_transition_matrix)
    d_transition_covariance = _to_tensor(d_transition_covariance)
    d_observation_offset = _to_tensor(d_observation_offset)
    d_observation_matrix = _to_tensor(d_observation_matrix)
    d_observation_covariance = _to_tensor(d_observation_covariance)
    ddmean = _to_tensor(d2_initial_state_mean)
    d2_initial_state_covariance = _to_tensor(d2_initial_state_covariance)
    d2_transition_offset = _to_tensor(d2_transition_offset)
    d2_transition_matrix = _to_tensor(d2_transition_matrix)
    d2_transition_covariance = _to_tensor(d2_transition_covariance)
    d2_observation_offset = _to_tensor(d2_observation_offset)
    d2_observation_matrix = _to_tensor(d2_observation_matrix)
    d2_observation_covariance = _to_tensor(d2_observation_covariance)

    parameter_dim = _static_dim(d_transition_offset, 0, "parameter dimension")
    state_dim = tf.shape(mean)[0]
    obs_dim = tf.shape(observation_matrix)[0]
    state_identity = tf.eye(state_dim, dtype=tf.float64)
    obs_identity = tf.eye(obs_dim, dtype=tf.float64)
    jitter_tensor = tf.cast(jitter, tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    dummy_log_norm = tf.constant(math.log(2.0 * math.pi), dtype=tf.float64)

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

    score = tf.zeros((parameter_dim,), dtype=tf.float64)
    hessian = tf.zeros((parameter_dim, parameter_dim), dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)

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

        row_weight = tf.cast(observation_mask[t], tf.float64)
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
            tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi)
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
) -> FilterRunMetadata:
    return FilterRunMetadata(
        filter_name=filter_name,
        partition=model.partition,
        integration_space="full_state",
        deterministic_completion="none",
        approximation_label=None,
        differentiability_status="analytic_score_hessian",
        compiled_status="tf_function",
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
            branch_label="qr_square_root",
            derivative_target="implemented_regularized_law",
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
        ),
    )
