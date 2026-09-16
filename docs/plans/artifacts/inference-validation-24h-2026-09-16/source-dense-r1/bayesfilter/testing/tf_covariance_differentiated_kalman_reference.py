"""Covariance-form TF differentiated Kalman reference backend for tests.

This module is intentionally under ``bayesfilter.testing``.  It is copied from
MacroFinance as a debugging oracle for BayesFilter derivative work, including
eager trace rows that convert tensors to NumPy arrays.  Do not re-export it as a
production filtering backend.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


def _matvec(matrix: tf.Tensor, vector: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matvec(matrix, vector)


def _to_tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def tf_differentiated_kalman_loglik_grad(
    observations,
    model,
    derivatives,
    jitter: float = 1e-9,
    return_trace: bool = False,
) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, list[dict[str, object]]]:
    """Return the Kalman log likelihood and analytical score only.

    This is the production value+score workload used by TFP HMC leapfrog and MAP
    gradient steps. It intentionally does not allocate second-order state
    derivatives or Hessian accumulators. Trace mode delegates to the full
    reference implementation and discards Hessian output so debugging keeps the
    established row format.
    """
    if return_trace:
        loglik, grad, _hess, trace_rows = tf_differentiated_kalman_loglik(
            observations,
            model,
            derivatives,
            jitter=jitter,
            return_trace=True,
        )
        return loglik, grad, trace_rows
    return tf_differentiated_kalman_loglik_grad_graph(observations, model, derivatives, jitter=jitter)


def tf_differentiated_kalman_loglik_grad_graph(
    observations,
    model,
    derivatives,
    jitter: float = 1e-9,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Graph-native first-order differentiated Kalman recursion.

    The loop over time is a `tf.while_loop`, so tracing cost does not scale by
    unrolling the observation panel. The small one-country parameter loop remains
    static inside the loop body; it is bounded by the parameter dimension and is
    much cheaper than unrolling over `T`.
    """
    observations = _to_tensor(observations)
    initial_mean = _to_tensor(model.initial_mean)
    initial_covariance = _to_tensor(model.initial_covariance)
    transition_offset = _to_tensor(model.transition_offset)
    transition_matrix = _to_tensor(model.transition_matrix)
    transition_covariance = _to_tensor(model.transition_covariance)
    observation_offset = _to_tensor(model.observation_offset)
    observation_matrix = _to_tensor(model.observation_matrix)
    observation_covariance = _to_tensor(model.observation_covariance)

    d_initial_mean = _to_tensor(derivatives.d_initial_mean)
    d_initial_covariance = _to_tensor(derivatives.d_initial_covariance)
    d_transition_offset = _to_tensor(derivatives.d_transition_offset)
    d_transition_matrix = _to_tensor(derivatives.d_transition_matrix)
    d_transition_covariance = _to_tensor(derivatives.d_transition_covariance)
    d_observation_offset = _to_tensor(derivatives.d_observation_offset)
    d_observation_matrix = _to_tensor(derivatives.d_observation_matrix)
    d_observation_covariance = _to_tensor(derivatives.d_observation_covariance)

    parameter_dim = int(d_transition_offset.shape[0])
    state_dim = int(transition_matrix.shape[0])
    obs_dim = int(observation_matrix.shape[0])
    identity_state = tf.eye(state_dim, dtype=tf.float64)
    identity_obs = tf.eye(obs_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    jitter_tf = tf.cast(jitter, tf.float64)
    n_timesteps = tf.shape(observations)[0]

    def cond(t, mean, covariance, dmean, dcov, loglik, grad):
        del mean, covariance, dmean, dcov, loglik, grad
        return t < n_timesteps

    def body(t, mean, covariance, dmean, dcov, loglik, grad):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        predicted_covariance = transition_matrix @ covariance @ tf.transpose(transition_matrix) + transition_covariance
        predicted_covariance = _symmetrize(predicted_covariance)

        dpred_mean_values = []
        dpred_cov_values = []
        for i in range(parameter_dim):
            dpred_mean_i = d_transition_offset[i] + _matvec(d_transition_matrix[i], mean) + _matvec(transition_matrix, dmean[i])
            dpred_cov_i = (
                d_transition_matrix[i] @ covariance @ tf.transpose(transition_matrix)
                + transition_matrix @ dcov[i] @ tf.transpose(transition_matrix)
                + transition_matrix @ covariance @ tf.transpose(d_transition_matrix[i])
                + d_transition_covariance[i]
            )
            dpred_mean_values.append(dpred_mean_i)
            dpred_cov_values.append(_symmetrize(dpred_cov_i))
        dpred_mean = tf.stack(dpred_mean_values, axis=0)
        dpred_cov = tf.stack(dpred_cov_values, axis=0)

        innovation = observations[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))
        innovation_covariance = observation_matrix @ predicted_covariance @ tf.transpose(observation_matrix) + observation_covariance + jitter_tf * identity_obs
        innovation_covariance = _symmetrize(innovation_covariance)
        chol = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(chol, identity_obs)
        precision_innovation = _matvec(innovation_precision, innovation)

        dinnovation_values = []
        dS_values = []
        dSinv_values = []
        grad_contrib_values = []
        for i in range(parameter_dim):
            dinnovation_i = -d_observation_offset[i] - _matvec(d_observation_matrix[i], predicted_mean) - _matvec(observation_matrix, dpred_mean[i])
            dS_i = (
                d_observation_matrix[i] @ predicted_covariance @ tf.transpose(observation_matrix)
                + observation_matrix @ dpred_cov[i] @ tf.transpose(observation_matrix)
                + observation_matrix @ predicted_covariance @ tf.transpose(d_observation_matrix[i])
                + d_observation_covariance[i]
            )
            dS_i = _symmetrize(dS_i)
            dSinv_i = -innovation_precision @ dS_i @ innovation_precision
            grad_i = -0.5 * (
                tf.linalg.trace(innovation_precision @ dS_i)
                + 2.0 * tf.tensordot(dinnovation_i, precision_innovation, axes=1)
                - tf.tensordot(innovation, _matvec(innovation_precision @ dS_i, precision_innovation), axes=1)
            )
            dinnovation_values.append(dinnovation_i)
            dS_values.append(dS_i)
            dSinv_values.append(dSinv_i)
            grad_contrib_values.append(grad_i)
        dinnovation = tf.stack(dinnovation_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        grad_contrib = tf.stack(grad_contrib_values, axis=0)

        kalman_gain = predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision
        dK_values = []
        for i in range(parameter_dim):
            dK_values.append(
                dpred_cov[i] @ tf.transpose(observation_matrix) @ innovation_precision
                + predicted_covariance @ tf.transpose(d_observation_matrix[i]) @ innovation_precision
                + predicted_covariance @ tf.transpose(observation_matrix) @ dSinv[i]
            )
        dK = tf.stack(dK_values, axis=0)

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        covariance = (identity_state - kalman_gain @ observation_matrix) @ predicted_covariance
        covariance = _symmetrize(covariance)

        dmean_values = []
        dcov_values = []
        for i in range(parameter_dim):
            dmean_i = dpred_mean[i] + _matvec(dK[i], innovation) + _matvec(kalman_gain, dinnovation[i])
            dcov_i = (
                -dK[i] @ observation_matrix @ predicted_covariance
                - kalman_gain @ d_observation_matrix[i] @ predicted_covariance
                + (identity_state - kalman_gain @ observation_matrix) @ dpred_cov[i]
            )
            dmean_values.append(dmean_i)
            dcov_values.append(_symmetrize(dcov_i))
        dmean = tf.stack(dmean_values, axis=0)
        dcov = tf.stack(dcov_values, axis=0)

        solve_innovation = tf.linalg.triangular_solve(chol, tf.expand_dims(innovation, axis=-1), lower=True)
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        contrib = -0.5 * (tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi) + log_det + mahalanobis)
        return t + 1, mean, covariance, dmean, dcov, loglik + contrib, grad + grad_contrib

    _t, _mean, _covariance, _dmean, _dcov, loglik, grad = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, dtype=tf.int32),
            tf.identity(initial_mean),
            tf.identity(initial_covariance),
            tf.identity(d_initial_mean),
            tf.identity(d_initial_covariance),
            tf.constant(0.0, dtype=tf.float64),
            tf.zeros((parameter_dim,), dtype=tf.float64),
        ),
        parallel_iterations=1,
    )
    return loglik, grad


def tf_differentiated_kalman_loglik_grad_hessian_graph(
    observations,
    model,
    derivatives,
    jitter: float = 1e-9,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Graph-native second-order differentiated Kalman recursion.

    This preserves the covariance-form equations of the NumPy oracle but moves
    the time recursion into `tf.while_loop`. Static parameter loops are retained
    inside the loop body to avoid an algebra rewrite while still removing the
    panel-length graph explosion that made naive `tf.function` unusable.
    """
    observations = _to_tensor(observations)
    initial_mean = _to_tensor(model.initial_mean)
    initial_covariance = _to_tensor(model.initial_covariance)
    transition_offset = _to_tensor(model.transition_offset)
    transition_matrix = _to_tensor(model.transition_matrix)
    transition_covariance = _to_tensor(model.transition_covariance)
    observation_offset = _to_tensor(model.observation_offset)
    observation_matrix = _to_tensor(model.observation_matrix)
    observation_covariance = _to_tensor(model.observation_covariance)

    d_initial_mean = _to_tensor(derivatives.d_initial_mean)
    d_initial_covariance = _to_tensor(derivatives.d_initial_covariance)
    d_transition_offset = _to_tensor(derivatives.d_transition_offset)
    d_transition_matrix = _to_tensor(derivatives.d_transition_matrix)
    d_transition_covariance = _to_tensor(derivatives.d_transition_covariance)
    d_observation_offset = _to_tensor(derivatives.d_observation_offset)
    d_observation_matrix = _to_tensor(derivatives.d_observation_matrix)
    d_observation_covariance = _to_tensor(derivatives.d_observation_covariance)
    d2_initial_mean = _to_tensor(derivatives.d2_initial_mean)
    d2_initial_covariance = _to_tensor(derivatives.d2_initial_covariance)
    d2_transition_offset = _to_tensor(derivatives.d2_transition_offset)
    d2_transition_matrix = _to_tensor(derivatives.d2_transition_matrix)
    d2_transition_covariance = _to_tensor(derivatives.d2_transition_covariance)
    d2_observation_offset = _to_tensor(derivatives.d2_observation_offset)
    d2_observation_matrix = _to_tensor(derivatives.d2_observation_matrix)
    d2_observation_covariance = _to_tensor(derivatives.d2_observation_covariance)

    parameter_dim = int(d_transition_offset.shape[0])
    state_dim = int(transition_matrix.shape[0])
    obs_dim = int(observation_matrix.shape[0])
    identity_state = tf.eye(state_dim, dtype=tf.float64)
    identity_obs = tf.eye(obs_dim, dtype=tf.float64)
    two_pi = tf.constant(2.0 * math.pi, dtype=tf.float64)
    jitter_tf = tf.cast(jitter, tf.float64)
    n_timesteps = tf.shape(observations)[0]

    def cond(t, mean, covariance, dmean, dcov, ddmean, ddcov, loglik, grad, hess):
        del mean, covariance, dmean, dcov, ddmean, ddcov, loglik, grad, hess
        return t < n_timesteps

    def body(t, mean, covariance, dmean, dcov, ddmean, ddcov, loglik, grad, hess):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        predicted_covariance = transition_matrix @ covariance @ tf.transpose(transition_matrix) + transition_covariance
        predicted_covariance = _symmetrize(predicted_covariance)

        dpred_mean_values = []
        dpred_cov_values = []
        d2pred_mean_rows = []
        d2pred_cov_rows = []
        for i in range(parameter_dim):
            dpred_mean_i = d_transition_offset[i] + _matvec(d_transition_matrix[i], mean) + _matvec(transition_matrix, dmean[i])
            dpred_cov_i = (
                d_transition_matrix[i] @ covariance @ tf.transpose(transition_matrix)
                + transition_matrix @ dcov[i] @ tf.transpose(transition_matrix)
                + transition_matrix @ covariance @ tf.transpose(d_transition_matrix[i])
                + d_transition_covariance[i]
            )
            dpred_mean_values.append(dpred_mean_i)
            dpred_cov_values.append(_symmetrize(dpred_cov_i))
            d2pred_mean_values = []
            d2pred_cov_values = []
            for j in range(parameter_dim):
                d2pred_mean_ij = (
                    d2_transition_offset[i, j]
                    + _matvec(d2_transition_matrix[i, j], mean)
                    + _matvec(d_transition_matrix[i], dmean[j])
                    + _matvec(d_transition_matrix[j], dmean[i])
                    + _matvec(transition_matrix, ddmean[i, j])
                )
                d2pred_cov_ij = (
                    d2_transition_matrix[i, j] @ covariance @ tf.transpose(transition_matrix)
                    + d_transition_matrix[i] @ dcov[j] @ tf.transpose(transition_matrix)
                    + d_transition_matrix[i] @ covariance @ tf.transpose(d_transition_matrix[j])
                    + d_transition_matrix[j] @ dcov[i] @ tf.transpose(transition_matrix)
                    + transition_matrix @ ddcov[i, j] @ tf.transpose(transition_matrix)
                    + transition_matrix @ dcov[i] @ tf.transpose(d_transition_matrix[j])
                    + d_transition_matrix[j] @ covariance @ tf.transpose(d_transition_matrix[i])
                    + transition_matrix @ dcov[j] @ tf.transpose(d_transition_matrix[i])
                    + transition_matrix @ covariance @ tf.transpose(d2_transition_matrix[i, j])
                    + d2_transition_covariance[i, j]
                )
                d2pred_mean_values.append(d2pred_mean_ij)
                d2pred_cov_values.append(_symmetrize(d2pred_cov_ij))
            d2pred_mean_rows.append(tf.stack(d2pred_mean_values, axis=0))
            d2pred_cov_rows.append(tf.stack(d2pred_cov_values, axis=0))
        dpred_mean = tf.stack(dpred_mean_values, axis=0)
        dpred_cov = tf.stack(dpred_cov_values, axis=0)
        d2pred_mean = tf.stack(d2pred_mean_rows, axis=0)
        d2pred_cov = tf.stack(d2pred_cov_rows, axis=0)

        innovation = observations[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))
        innovation_covariance = observation_matrix @ predicted_covariance @ tf.transpose(observation_matrix) + observation_covariance + jitter_tf * identity_obs
        innovation_covariance = _symmetrize(innovation_covariance)
        chol = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(chol, identity_obs)
        precision_innovation = _matvec(innovation_precision, innovation)

        dinnovation_values = []
        dS_values = []
        dSinv_values = []
        grad_contrib_values = []
        for i in range(parameter_dim):
            dinnovation_i = -d_observation_offset[i] - _matvec(d_observation_matrix[i], predicted_mean) - _matvec(observation_matrix, dpred_mean[i])
            dS_i = (
                d_observation_matrix[i] @ predicted_covariance @ tf.transpose(observation_matrix)
                + observation_matrix @ dpred_cov[i] @ tf.transpose(observation_matrix)
                + observation_matrix @ predicted_covariance @ tf.transpose(d_observation_matrix[i])
                + d_observation_covariance[i]
            )
            dS_i = _symmetrize(dS_i)
            dSinv_i = -innovation_precision @ dS_i @ innovation_precision
            grad_i = -0.5 * (
                tf.linalg.trace(innovation_precision @ dS_i)
                + 2.0 * tf.tensordot(dinnovation_i, precision_innovation, axes=1)
                - tf.tensordot(innovation, _matvec(innovation_precision @ dS_i, precision_innovation), axes=1)
            )
            dinnovation_values.append(dinnovation_i)
            dS_values.append(dS_i)
            dSinv_values.append(dSinv_i)
            grad_contrib_values.append(grad_i)
        dinnovation = tf.stack(dinnovation_values, axis=0)
        dS = tf.stack(dS_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        grad_contrib = tf.stack(grad_contrib_values, axis=0)

        d2innovation_rows = []
        d2S_rows = []
        d2Sinv_rows = []
        hess_contrib_rows = []
        for i in range(parameter_dim):
            d2innovation_values = []
            d2S_values = []
            d2Sinv_values = []
            hess_contrib_values = []
            for j in range(parameter_dim):
                d2innovation_ij = (
                    -d2_observation_offset[i, j]
                    - _matvec(d2_observation_matrix[i, j], predicted_mean)
                    - _matvec(d_observation_matrix[i], dpred_mean[j])
                    - _matvec(d_observation_matrix[j], dpred_mean[i])
                    - _matvec(observation_matrix, d2pred_mean[i, j])
                )
                d2S_ij = (
                    d2_observation_matrix[i, j] @ predicted_covariance @ tf.transpose(observation_matrix)
                    + d_observation_matrix[i] @ dpred_cov[j] @ tf.transpose(observation_matrix)
                    + d_observation_matrix[i] @ predicted_covariance @ tf.transpose(d_observation_matrix[j])
                    + d_observation_matrix[j] @ dpred_cov[i] @ tf.transpose(observation_matrix)
                    + observation_matrix @ d2pred_cov[i, j] @ tf.transpose(observation_matrix)
                    + observation_matrix @ dpred_cov[i] @ tf.transpose(d_observation_matrix[j])
                    + d_observation_matrix[j] @ predicted_covariance @ tf.transpose(d_observation_matrix[i])
                    + observation_matrix @ dpred_cov[j] @ tf.transpose(d_observation_matrix[i])
                    + observation_matrix @ predicted_covariance @ tf.transpose(d2_observation_matrix[i, j])
                    + d2_observation_covariance[i, j]
                )
                d2S_ij = _symmetrize(d2S_ij)
                d2Sinv_ij = (
                    innovation_precision @ dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + innovation_precision @ dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - innovation_precision @ d2S_ij @ innovation_precision
                )
                trace_term = tf.linalg.trace(dSinv[j] @ dS[i] + innovation_precision @ d2S_ij)
                quad_term = (
                    2.0 * tf.tensordot(d2innovation_ij, precision_innovation, axes=1)
                    + 2.0 * tf.tensordot(dinnovation[i], _matvec(dSinv[j], innovation), axes=1)
                    + 2.0 * tf.tensordot(dinnovation[i], _matvec(innovation_precision, dinnovation[j]), axes=1)
                )
                curvature_term = (
                    tf.tensordot(dinnovation[j], _matvec(innovation_precision @ dS[i], precision_innovation), axes=1)
                    + tf.tensordot(innovation, _matvec(dSinv[j] @ dS[i], precision_innovation), axes=1)
                    + tf.tensordot(innovation, _matvec(innovation_precision @ d2S_ij, precision_innovation), axes=1)
                    + tf.tensordot(innovation, _matvec(innovation_precision @ dS[i], _matvec(dSinv[j], innovation)), axes=1)
                    + tf.tensordot(innovation, _matvec(innovation_precision @ dS[i], _matvec(innovation_precision, dinnovation[j])), axes=1)
                )
                hess_contrib_values.append(-0.5 * (trace_term + quad_term - curvature_term))
                d2innovation_values.append(d2innovation_ij)
                d2S_values.append(d2S_ij)
                d2Sinv_values.append(d2Sinv_ij)
            d2innovation_rows.append(tf.stack(d2innovation_values, axis=0))
            d2S_rows.append(tf.stack(d2S_values, axis=0))
            d2Sinv_rows.append(tf.stack(d2Sinv_values, axis=0))
            hess_contrib_rows.append(tf.stack(hess_contrib_values, axis=0))
        d2innovation = tf.stack(d2innovation_rows, axis=0)
        d2S = tf.stack(d2S_rows, axis=0)
        d2Sinv = tf.stack(d2Sinv_rows, axis=0)
        hess_contrib = tf.stack(hess_contrib_rows, axis=0)

        kalman_gain = predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision
        dK_values = []
        d2K_rows = []
        for i in range(parameter_dim):
            dK_i = (
                dpred_cov[i] @ tf.transpose(observation_matrix) @ innovation_precision
                + predicted_covariance @ tf.transpose(d_observation_matrix[i]) @ innovation_precision
                + predicted_covariance @ tf.transpose(observation_matrix) @ dSinv[i]
            )
            dK_values.append(dK_i)
            d2K_values = []
            for j in range(parameter_dim):
                d2K_values.append(
                    d2pred_cov[i, j] @ tf.transpose(observation_matrix) @ innovation_precision
                    + dpred_cov[i] @ tf.transpose(d_observation_matrix[j]) @ innovation_precision
                    + dpred_cov[i] @ tf.transpose(observation_matrix) @ dSinv[j]
                    + dpred_cov[j] @ tf.transpose(d_observation_matrix[i]) @ innovation_precision
                    + predicted_covariance @ tf.transpose(d2_observation_matrix[i, j]) @ innovation_precision
                    + predicted_covariance @ tf.transpose(d_observation_matrix[i]) @ dSinv[j]
                    + dpred_cov[j] @ tf.transpose(observation_matrix) @ dSinv[i]
                    + predicted_covariance @ tf.transpose(d_observation_matrix[j]) @ dSinv[i]
                    + predicted_covariance @ tf.transpose(observation_matrix) @ d2Sinv[i, j]
                )
            d2K_rows.append(tf.stack(d2K_values, axis=0))
        dK = tf.stack(dK_values, axis=0)
        d2K = tf.stack(d2K_rows, axis=0)

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        covariance = (identity_state - kalman_gain @ observation_matrix) @ predicted_covariance
        covariance = _symmetrize(covariance)

        dmean_values = []
        dcov_values = []
        ddmean_rows = []
        ddcov_rows = []
        for i in range(parameter_dim):
            dmean_i = dpred_mean[i] + _matvec(dK[i], innovation) + _matvec(kalman_gain, dinnovation[i])
            dcov_i = (
                -dK[i] @ observation_matrix @ predicted_covariance
                - kalman_gain @ d_observation_matrix[i] @ predicted_covariance
                + (identity_state - kalman_gain @ observation_matrix) @ dpred_cov[i]
            )
            dmean_values.append(dmean_i)
            dcov_values.append(_symmetrize(dcov_i))
            ddmean_values = []
            ddcov_values = []
            for j in range(parameter_dim):
                ddmean_ij = d2pred_mean[i, j] + _matvec(d2K[i, j], innovation) + _matvec(dK[i], dinnovation[j]) + _matvec(dK[j], dinnovation[i]) + _matvec(kalman_gain, d2innovation[i, j])
                ddcov_ij = (
                    -d2K[i, j] @ observation_matrix @ predicted_covariance
                    - dK[i] @ d_observation_matrix[j] @ predicted_covariance
                    - dK[i] @ observation_matrix @ dpred_cov[j]
                    - dK[j] @ d_observation_matrix[i] @ predicted_covariance
                    - kalman_gain @ d2_observation_matrix[i, j] @ predicted_covariance
                    - kalman_gain @ d_observation_matrix[i] @ dpred_cov[j]
                    - dK[j] @ observation_matrix @ dpred_cov[i]
                    - kalman_gain @ d_observation_matrix[j] @ dpred_cov[i]
                    + (identity_state - kalman_gain @ observation_matrix) @ d2pred_cov[i, j]
                )
                ddmean_values.append(ddmean_ij)
                ddcov_values.append(_symmetrize(ddcov_ij))
            ddmean_rows.append(tf.stack(ddmean_values, axis=0))
            ddcov_rows.append(tf.stack(ddcov_values, axis=0))
        dmean = tf.stack(dmean_values, axis=0)
        dcov = tf.stack(dcov_values, axis=0)
        ddmean = tf.stack(ddmean_rows, axis=0)
        ddcov = tf.stack(ddcov_rows, axis=0)

        solve_innovation = tf.linalg.triangular_solve(chol, tf.expand_dims(innovation, axis=-1), lower=True)
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        contrib = -0.5 * (tf.cast(obs_dim, tf.float64) * tf.math.log(two_pi) + log_det + mahalanobis)
        return (
            t + 1,
            mean,
            covariance,
            dmean,
            dcov,
            ddmean,
            ddcov,
            loglik + contrib,
            grad + grad_contrib,
            hess + hess_contrib,
        )

    _t, _mean, _covariance, _dmean, _dcov, _ddmean, _ddcov, loglik, grad, hess = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, dtype=tf.int32),
            tf.identity(initial_mean),
            tf.identity(initial_covariance),
            tf.identity(d_initial_mean),
            tf.identity(d_initial_covariance),
            tf.identity(d2_initial_mean),
            tf.identity(d2_initial_covariance),
            tf.constant(0.0, dtype=tf.float64),
            tf.zeros((parameter_dim,), dtype=tf.float64),
            tf.zeros((parameter_dim, parameter_dim), dtype=tf.float64),
        ),
        parallel_iterations=1,
    )
    return loglik, grad, _symmetrize(hess)


def tf_differentiated_kalman_loglik(
    observations,
    model,
    derivatives,
    jitter: float = 1e-9,
    return_trace: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor, list[dict[str, object]]]:
    """TensorFlow-native analytical differentiated Kalman recursion.

    Non-trace calls use a graph-native `tf.while_loop` over time so compiled
    execution does not build a graph with one copy of the filter body per
    observation. Trace calls keep the original eager/reference implementation and
    its `.numpy()` debug rows for detailed audits.
    """
    if not return_trace:
        return tf_differentiated_kalman_loglik_grad_hessian_graph(
            observations,
            model,
            derivatives,
            jitter=jitter,
        )
    observations = _to_tensor(observations)
    trace_rows = [] if return_trace else None
    initial_mean = _to_tensor(model.initial_mean)
    initial_covariance = _to_tensor(model.initial_covariance)
    transition_offset = _to_tensor(model.transition_offset)
    transition_matrix = _to_tensor(model.transition_matrix)
    transition_covariance = _to_tensor(model.transition_covariance)
    observation_offset = _to_tensor(model.observation_offset)
    observation_matrix = _to_tensor(model.observation_matrix)
    observation_covariance = _to_tensor(model.observation_covariance)

    d_initial_mean = _to_tensor(derivatives.d_initial_mean)
    d_initial_covariance = _to_tensor(derivatives.d_initial_covariance)
    d_transition_offset = _to_tensor(derivatives.d_transition_offset)
    d_transition_matrix = _to_tensor(derivatives.d_transition_matrix)
    d_transition_covariance = _to_tensor(derivatives.d_transition_covariance)
    d_observation_offset = _to_tensor(derivatives.d_observation_offset)
    d_observation_matrix = _to_tensor(derivatives.d_observation_matrix)
    d_observation_covariance = _to_tensor(derivatives.d_observation_covariance)
    d2_initial_mean = _to_tensor(derivatives.d2_initial_mean)
    d2_initial_covariance = _to_tensor(derivatives.d2_initial_covariance)
    d2_transition_offset = _to_tensor(derivatives.d2_transition_offset)
    d2_transition_matrix = _to_tensor(derivatives.d2_transition_matrix)
    d2_transition_covariance = _to_tensor(derivatives.d2_transition_covariance)
    d2_observation_offset = _to_tensor(derivatives.d2_observation_offset)
    d2_observation_matrix = _to_tensor(derivatives.d2_observation_matrix)
    d2_observation_covariance = _to_tensor(derivatives.d2_observation_covariance)

    parameter_dim = int(d_transition_offset.shape[0])
    n_timesteps = int(observations.shape[0])
    state_dim = int(transition_matrix.shape[0])
    obs_dim = int(observation_matrix.shape[0])

    mean = tf.identity(initial_mean)
    covariance = tf.identity(initial_covariance)
    grad = tf.zeros((parameter_dim,), dtype=tf.float64)
    hess = tf.zeros((parameter_dim, parameter_dim), dtype=tf.float64)
    dmean = tf.identity(d_initial_mean)
    dcov = tf.identity(d_initial_covariance)
    ddmean = tf.identity(d2_initial_mean)
    ddcov = tf.identity(d2_initial_covariance)
    identity_state = tf.eye(state_dim, dtype=tf.float64)
    identity_obs = tf.eye(obs_dim, dtype=tf.float64)
    loglik = tf.constant(0.0, dtype=tf.float64)

    for t in range(n_timesteps):
        predicted_mean = transition_offset + _matvec(transition_matrix, mean)
        predicted_covariance = transition_matrix @ covariance @ tf.transpose(transition_matrix) + transition_covariance
        predicted_covariance = 0.5 * (predicted_covariance + tf.transpose(predicted_covariance))

        dpred_mean_values = []
        dpred_cov_values = []
        d2pred_mean_rows = []
        d2pred_cov_rows = []
        for i in range(parameter_dim):
            dpred_mean_i = d_transition_offset[i] + _matvec(d_transition_matrix[i], mean) + _matvec(transition_matrix, dmean[i])
            dpred_cov_i = (
                d_transition_matrix[i] @ covariance @ tf.transpose(transition_matrix)
                + transition_matrix @ dcov[i] @ tf.transpose(transition_matrix)
                + transition_matrix @ covariance @ tf.transpose(d_transition_matrix[i])
                + d_transition_covariance[i]
            )
            dpred_cov_i = 0.5 * (dpred_cov_i + tf.transpose(dpred_cov_i))
            dpred_mean_values.append(dpred_mean_i)
            dpred_cov_values.append(dpred_cov_i)

            d2pred_mean_values = []
            d2pred_cov_values = []
            for j in range(parameter_dim):
                d2pred_mean_ij = (
                    d2_transition_offset[i, j]
                    + _matvec(d2_transition_matrix[i, j], mean)
                    + _matvec(d_transition_matrix[i], dmean[j])
                    + _matvec(d_transition_matrix[j], dmean[i])
                    + _matvec(transition_matrix, ddmean[i, j])
                )
                d2pred_cov_ij = (
                    d2_transition_matrix[i, j] @ covariance @ tf.transpose(transition_matrix)
                    + d_transition_matrix[i] @ dcov[j] @ tf.transpose(transition_matrix)
                    + d_transition_matrix[i] @ covariance @ tf.transpose(d_transition_matrix[j])
                    + d_transition_matrix[j] @ dcov[i] @ tf.transpose(transition_matrix)
                    + transition_matrix @ ddcov[i, j] @ tf.transpose(transition_matrix)
                    + transition_matrix @ dcov[i] @ tf.transpose(d_transition_matrix[j])
                    + d_transition_matrix[j] @ covariance @ tf.transpose(d_transition_matrix[i])
                    + transition_matrix @ dcov[j] @ tf.transpose(d_transition_matrix[i])
                    + transition_matrix @ covariance @ tf.transpose(d2_transition_matrix[i, j])
                    + d2_transition_covariance[i, j]
                )
                d2pred_cov_ij = 0.5 * (d2pred_cov_ij + tf.transpose(d2pred_cov_ij))
                d2pred_mean_values.append(d2pred_mean_ij)
                d2pred_cov_values.append(d2pred_cov_ij)
            d2pred_mean_rows.append(tf.stack(d2pred_mean_values, axis=0))
            d2pred_cov_rows.append(tf.stack(d2pred_cov_values, axis=0))
        dpred_mean = tf.stack(dpred_mean_values, axis=0)
        dpred_cov = tf.stack(dpred_cov_values, axis=0)
        d2pred_mean = tf.stack(d2pred_mean_rows, axis=0)
        d2pred_cov = tf.stack(d2pred_cov_rows, axis=0)

        innovation = observations[t] - (observation_offset + _matvec(observation_matrix, predicted_mean))
        innovation_covariance = observation_matrix @ predicted_covariance @ tf.transpose(observation_matrix) + observation_covariance + tf.cast(jitter, tf.float64) * identity_obs
        innovation_covariance = 0.5 * (innovation_covariance + tf.transpose(innovation_covariance))
        chol = tf.linalg.cholesky(innovation_covariance)
        innovation_precision = tf.linalg.cholesky_solve(chol, identity_obs)

        dinnovation_values = []
        dS_values = []
        dSinv_values = []
        grad_contrib_values = []
        for i in range(parameter_dim):
            dinnovation_i = -d_observation_offset[i] - _matvec(d_observation_matrix[i], predicted_mean) - _matvec(observation_matrix, dpred_mean[i])
            dS_i = (
                d_observation_matrix[i] @ predicted_covariance @ tf.transpose(observation_matrix)
                + observation_matrix @ dpred_cov[i] @ tf.transpose(observation_matrix)
                + observation_matrix @ predicted_covariance @ tf.transpose(d_observation_matrix[i])
                + d_observation_covariance[i]
            )
            dS_i = 0.5 * (dS_i + tf.transpose(dS_i))
            dSinv_i = -innovation_precision @ dS_i @ innovation_precision
            dinnovation_values.append(dinnovation_i)
            dS_values.append(dS_i)
            dSinv_values.append(dSinv_i)
            grad_i = -0.5 * (
                tf.linalg.trace(innovation_precision @ dS_i)
                + 2.0 * tf.tensordot(dinnovation_i, _matvec(innovation_precision, innovation), axes=1)
                - tf.tensordot(innovation, tf.linalg.matvec(innovation_precision @ dS_i, tf.linalg.matvec(innovation_precision, innovation)), axes=1)
            )
            grad_contrib_values.append(grad_i)
            grad = tf.tensor_scatter_nd_update(grad, [[i]], [grad[i] + grad_i])
        dinnovation = tf.stack(dinnovation_values, axis=0)
        dS = tf.stack(dS_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        grad_contrib = tf.stack(grad_contrib_values, axis=0)

        d2innovation_rows = []
        d2S_rows = []
        d2Sinv_rows = []
        hess_contrib_rows = []
        for i in range(parameter_dim):
            d2innovation_values = []
            d2S_values = []
            d2Sinv_values = []
            hess_contrib_values = []
            for j in range(parameter_dim):
                d2innovation_ij = (
                    -d2_observation_offset[i, j]
                    - _matvec(d2_observation_matrix[i, j], predicted_mean)
                    - _matvec(d_observation_matrix[i], dpred_mean[j])
                    - _matvec(d_observation_matrix[j], dpred_mean[i])
                    - _matvec(observation_matrix, d2pred_mean[i, j])
                )
                d2S_ij = (
                    d2_observation_matrix[i, j] @ predicted_covariance @ tf.transpose(observation_matrix)
                    + d_observation_matrix[i] @ dpred_cov[j] @ tf.transpose(observation_matrix)
                    + d_observation_matrix[i] @ predicted_covariance @ tf.transpose(d_observation_matrix[j])
                    + d_observation_matrix[j] @ dpred_cov[i] @ tf.transpose(observation_matrix)
                    + observation_matrix @ d2pred_cov[i, j] @ tf.transpose(observation_matrix)
                    + observation_matrix @ dpred_cov[i] @ tf.transpose(d_observation_matrix[j])
                    + d_observation_matrix[j] @ predicted_covariance @ tf.transpose(d_observation_matrix[i])
                    + observation_matrix @ dpred_cov[j] @ tf.transpose(d_observation_matrix[i])
                    + observation_matrix @ predicted_covariance @ tf.transpose(d2_observation_matrix[i, j])
                    + d2_observation_covariance[i, j]
                )
                d2S_ij = 0.5 * (d2S_ij + tf.transpose(d2S_ij))
                d2Sinv_ij = (
                    innovation_precision @ dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + innovation_precision @ dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - innovation_precision @ d2S_ij @ innovation_precision
                )
                trace_term = tf.linalg.trace(dSinv[j] @ dS[i] + innovation_precision @ d2S_ij)
                quad_term = (
                    2.0 * tf.tensordot(d2innovation_ij, _matvec(innovation_precision, innovation), axes=1)
                    + 2.0 * tf.tensordot(dinnovation[i], _matvec(dSinv[j], innovation), axes=1)
                    + 2.0 * tf.tensordot(dinnovation[i], _matvec(innovation_precision, dinnovation[j]), axes=1)
                )
                curvature_term = (
                    tf.tensordot(dinnovation[j], tf.linalg.matvec(innovation_precision @ dS[i], tf.linalg.matvec(innovation_precision, innovation)), axes=1)
                    + tf.tensordot(innovation, tf.linalg.matvec(dSinv[j] @ dS[i], tf.linalg.matvec(innovation_precision, innovation)), axes=1)
                    + tf.tensordot(innovation, tf.linalg.matvec(innovation_precision @ d2S_ij, tf.linalg.matvec(innovation_precision, innovation)), axes=1)
                    + tf.tensordot(innovation, tf.linalg.matvec(innovation_precision @ dS[i], tf.linalg.matvec(dSinv[j], innovation)), axes=1)
                    + tf.tensordot(innovation, tf.linalg.matvec(innovation_precision @ dS[i], tf.linalg.matvec(innovation_precision, dinnovation[j])), axes=1)
                )
                hess_ij = -0.5 * (trace_term + quad_term - curvature_term)
                hess_contrib_values.append(hess_ij)
                hess = tf.tensor_scatter_nd_update(hess, [[i, j]], [hess[i, j] + hess_ij])
                d2innovation_values.append(d2innovation_ij)
                d2S_values.append(d2S_ij)
                d2Sinv_values.append(d2Sinv_ij)
            d2innovation_rows.append(tf.stack(d2innovation_values, axis=0))
            d2S_rows.append(tf.stack(d2S_values, axis=0))
            d2Sinv_rows.append(tf.stack(d2Sinv_values, axis=0))
            hess_contrib_rows.append(tf.stack(hess_contrib_values, axis=0))
        d2innovation = tf.stack(d2innovation_rows, axis=0)
        d2S = tf.stack(d2S_rows, axis=0)
        d2Sinv = tf.stack(d2Sinv_rows, axis=0)
        hess_contrib = tf.stack(hess_contrib_rows, axis=0)

        kalman_gain = predicted_covariance @ tf.transpose(observation_matrix) @ innovation_precision
        dK_values = []
        d2K_rows = []
        for i in range(parameter_dim):
            dK_i = (
                dpred_cov[i] @ tf.transpose(observation_matrix) @ innovation_precision
                + predicted_covariance @ tf.transpose(d_observation_matrix[i]) @ innovation_precision
                + predicted_covariance @ tf.transpose(observation_matrix) @ dSinv[i]
            )
            dK_values.append(dK_i)
            d2K_values = []
            for j in range(parameter_dim):
                d2K_ij = (
                    d2pred_cov[i, j] @ tf.transpose(observation_matrix) @ innovation_precision
                    + dpred_cov[i] @ tf.transpose(d_observation_matrix[j]) @ innovation_precision
                    + dpred_cov[i] @ tf.transpose(observation_matrix) @ dSinv[j]
                    + dpred_cov[j] @ tf.transpose(d_observation_matrix[i]) @ innovation_precision
                    + predicted_covariance @ tf.transpose(d2_observation_matrix[i, j]) @ innovation_precision
                    + predicted_covariance @ tf.transpose(d_observation_matrix[i]) @ dSinv[j]
                    + dpred_cov[j] @ tf.transpose(observation_matrix) @ dSinv[i]
                    + predicted_covariance @ tf.transpose(d_observation_matrix[j]) @ dSinv[i]
                    + predicted_covariance @ tf.transpose(observation_matrix) @ d2Sinv[i, j]
                )
                d2K_values.append(d2K_ij)
            d2K_rows.append(tf.stack(d2K_values, axis=0))
        dK = tf.stack(dK_values, axis=0)
        d2K = tf.stack(d2K_rows, axis=0)

        mean = predicted_mean + _matvec(kalman_gain, innovation)
        covariance = (identity_state - kalman_gain @ observation_matrix) @ predicted_covariance
        covariance = 0.5 * (covariance + tf.transpose(covariance))

        dmean_values = []
        dcov_values = []
        ddmean_rows = []
        ddcov_rows = []
        for i in range(parameter_dim):
            dmean_i = dpred_mean[i] + tf.linalg.matvec(dK[i], innovation) + tf.linalg.matvec(kalman_gain, dinnovation[i])
            dcov_i = (
                -dK[i] @ observation_matrix @ predicted_covariance
                - kalman_gain @ d_observation_matrix[i] @ predicted_covariance
                + (identity_state - kalman_gain @ observation_matrix) @ dpred_cov[i]
            )
            dcov_i = 0.5 * (dcov_i + tf.transpose(dcov_i))
            dmean_values.append(dmean_i)
            dcov_values.append(dcov_i)
            ddmean_values = []
            ddcov_values = []
            for j in range(parameter_dim):
                ddmean_ij = d2pred_mean[i, j] + tf.linalg.matvec(d2K[i, j], innovation) + tf.linalg.matvec(dK[i], dinnovation[j]) + tf.linalg.matvec(dK[j], dinnovation[i]) + tf.linalg.matvec(kalman_gain, d2innovation[i, j])
                ddcov_ij = (
                    -d2K[i, j] @ observation_matrix @ predicted_covariance
                    - dK[i] @ d_observation_matrix[j] @ predicted_covariance
                    - dK[i] @ observation_matrix @ dpred_cov[j]
                    - dK[j] @ d_observation_matrix[i] @ predicted_covariance
                    - kalman_gain @ d2_observation_matrix[i, j] @ predicted_covariance
                    - kalman_gain @ d_observation_matrix[i] @ dpred_cov[j]
                    - dK[j] @ observation_matrix @ dpred_cov[i]
                    - kalman_gain @ d_observation_matrix[j] @ dpred_cov[i]
                    + (identity_state - kalman_gain @ observation_matrix) @ d2pred_cov[i, j]
                )
                ddcov_ij = 0.5 * (ddcov_ij + tf.transpose(ddcov_ij))
                ddmean_values.append(ddmean_ij)
                ddcov_values.append(ddcov_ij)
            ddmean_rows.append(tf.stack(ddmean_values, axis=0))
            ddcov_rows.append(tf.stack(ddcov_values, axis=0))
        dmean = tf.stack(dmean_values, axis=0)
        dcov = tf.stack(dcov_values, axis=0)
        ddmean = tf.stack(ddmean_rows, axis=0)
        ddcov = tf.stack(ddcov_rows, axis=0)

        chol = tf.linalg.cholesky(innovation_covariance)
        solve_innovation = tf.linalg.triangular_solve(chol, tf.expand_dims(innovation, axis=-1), lower=True)
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        contrib = -0.5 * (tf.cast(obs_dim, tf.float64) * tf.math.log(tf.constant(2.0 * 3.141592653589793, dtype=tf.float64)) + log_det + mahalanobis)
        loglik += contrib

        if trace_rows is not None:
            trace_rows.append(
                {
                    "t": t,
                    "innovation": np.asarray(innovation.numpy(), dtype=np.float64),
                    "innovation_covariance": np.asarray(innovation_covariance.numpy(), dtype=np.float64),
                    "innovation_precision": np.asarray(innovation_precision.numpy(), dtype=np.float64),
                    "grad_contrib": np.asarray(grad_contrib.numpy(), dtype=np.float64),
                    "hess_contrib": np.asarray(hess_contrib.numpy(), dtype=np.float64),
                    "dinnovation": np.asarray(dinnovation.numpy(), dtype=np.float64),
                    "dS": np.asarray(dS.numpy(), dtype=np.float64),
                    "d2innovation": np.asarray(d2innovation.numpy(), dtype=np.float64),
                    "d2S": np.asarray(d2S.numpy(), dtype=np.float64),
                    "dmean": np.asarray(dmean.numpy(), dtype=np.float64),
                    "dcov": np.asarray(dcov.numpy(), dtype=np.float64),
                    "ddmean": np.asarray(ddmean.numpy(), dtype=np.float64),
                    "ddcov": np.asarray(ddcov.numpy(), dtype=np.float64),
                    "dK": np.asarray(dK.numpy(), dtype=np.float64),
                    "d2K": np.asarray(d2K.numpy(), dtype=np.float64),
                    "dSinv": np.asarray(dSinv.numpy(), dtype=np.float64),
                    "d2Sinv": np.asarray(d2Sinv.numpy(), dtype=np.float64),
                    "loglik": float(loglik.numpy()),
                    "contrib": float(contrib.numpy()),
                }
            )

    hess = 0.5 * (hess + tf.transpose(hess))
    if trace_rows is not None:
        return loglik, grad, hess, trace_rows
    return loglik, grad, hess
