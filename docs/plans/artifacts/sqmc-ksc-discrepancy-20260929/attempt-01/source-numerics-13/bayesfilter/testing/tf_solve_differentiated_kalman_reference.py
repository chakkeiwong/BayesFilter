"""Solve-form TF differentiated Kalman reference backend for tests.

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


def _cholesky_solve(chol: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.linalg.cholesky_solve(chol, rhs)


def _cholesky_solve_vector(chol: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    return tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(rhs, axis=-1)), axis=-1)


def _trace_solve(chol: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    return tf.linalg.trace(_cholesky_solve(chol, matrix))


def tf_solve_differentiated_kalman_loglik(
    observations,
    model,
    derivatives,
    jitter: float = 1e-9,
    return_trace: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor, list[dict[str, object]]]:
    """TensorFlow analytical Kalman derivatives using Cholesky solves as the main primitive."""
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
        observation_covariance_jittered = observation_covariance + tf.cast(jitter, tf.float64) * identity_obs
        innovation_covariance = observation_matrix @ predicted_covariance @ tf.transpose(observation_matrix) + observation_covariance_jittered
        innovation_covariance = 0.5 * (innovation_covariance + tf.transpose(innovation_covariance))
        chol = tf.linalg.cholesky(innovation_covariance)
        innovation_solve = _cholesky_solve_vector(chol, innovation)
        innovation_precision = _cholesky_solve(chol, identity_obs)

        dinnovation_values = []
        dS_values = []
        dsolve_innovation_values = []
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
            dsolve_innovation_i = _cholesky_solve_vector(chol, dinnovation_i - _matvec(dS_i, innovation_solve))
            dSinv_i = -_cholesky_solve(chol, dS_i @ innovation_precision)
            grad_i = -0.5 * (
                _trace_solve(chol, dS_i)
                + 2.0 * tf.tensordot(dinnovation_i, innovation_solve, axes=1)
                - tf.tensordot(innovation_solve, _matvec(dS_i, innovation_solve), axes=1)
            )
            dinnovation_values.append(dinnovation_i)
            dS_values.append(dS_i)
            dsolve_innovation_values.append(dsolve_innovation_i)
            dSinv_values.append(dSinv_i)
            grad_contrib_values.append(grad_i)
            grad = tf.tensor_scatter_nd_update(grad, [[i]], [grad[i] + grad_i])
        dinnovation = tf.stack(dinnovation_values, axis=0)
        dS = tf.stack(dS_values, axis=0)
        dsolve_innovation = tf.stack(dsolve_innovation_values, axis=0)
        dSinv = tf.stack(dSinv_values, axis=0)
        grad_contrib = tf.stack(grad_contrib_values, axis=0)

        d2innovation_rows = []
        d2S_rows = []
        d2solve_innovation_rows = []
        d2Sinv_rows = []
        hess_contrib_rows = []
        for i in range(parameter_dim):
            d2innovation_values = []
            d2S_values = []
            d2solve_innovation_values = []
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
                d2solve_innovation_ij = _cholesky_solve_vector(
                    chol,
                    d2innovation_ij
                    - _matvec(d2S_ij, innovation_solve)
                    - _matvec(dS[i], dsolve_innovation[j])
                    - _matvec(dS[j], dsolve_innovation[i]),
                )
                d2Sinv_ij = _cholesky_solve(
                    chol,
                    dS[j] @ innovation_precision @ dS[i] @ innovation_precision
                    + dS[i] @ innovation_precision @ dS[j] @ innovation_precision
                    - d2S_ij @ innovation_precision,
                )
                trace_term = tf.linalg.trace(dSinv[j] @ dS[i] + innovation_precision @ d2S_ij)
                quad_term = (
                    2.0 * tf.tensordot(d2innovation_ij, innovation_solve, axes=1)
                    + 2.0 * tf.tensordot(dinnovation[i], dsolve_innovation[j], axes=1)
                )
                curvature_term = (
                    tf.tensordot(dinnovation[j], _matvec(innovation_precision @ dS[i], innovation_solve), axes=1)
                    + tf.tensordot(innovation, _matvec(dSinv[j] @ dS[i], innovation_solve), axes=1)
                    + tf.tensordot(innovation, _matvec(innovation_precision @ d2S_ij, innovation_solve), axes=1)
                    + tf.tensordot(innovation, _matvec(innovation_precision @ dS[i], dsolve_innovation[j]), axes=1)
                )
                hess_ij = -0.5 * (trace_term + quad_term - curvature_term)
                hess = tf.tensor_scatter_nd_update(hess, [[i, j]], [hess[i, j] + hess_ij])
                d2innovation_values.append(d2innovation_ij)
                d2S_values.append(d2S_ij)
                d2solve_innovation_values.append(d2solve_innovation_ij)
                d2Sinv_values.append(d2Sinv_ij)
                hess_contrib_values.append(hess_ij)
            d2innovation_rows.append(tf.stack(d2innovation_values, axis=0))
            d2S_rows.append(tf.stack(d2S_values, axis=0))
            d2solve_innovation_rows.append(tf.stack(d2solve_innovation_values, axis=0))
            d2Sinv_rows.append(tf.stack(d2Sinv_values, axis=0))
            hess_contrib_rows.append(tf.stack(hess_contrib_values, axis=0))
        d2innovation = tf.stack(d2innovation_rows, axis=0)
        d2S = tf.stack(d2S_rows, axis=0)
        d2solve_innovation = tf.stack(d2solve_innovation_rows, axis=0)
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
        joseph_left = identity_state - kalman_gain @ observation_matrix
        covariance = joseph_left @ predicted_covariance @ tf.transpose(joseph_left) + kalman_gain @ observation_covariance_jittered @ tf.transpose(kalman_gain)
        covariance = 0.5 * (covariance + tf.transpose(covariance))

        d_joseph_left_values = []
        d2_joseph_left_rows = []
        for i in range(parameter_dim):
            d_joseph_left_i = -dK[i] @ observation_matrix - kalman_gain @ d_observation_matrix[i]
            d_joseph_left_values.append(d_joseph_left_i)
            d2_joseph_left_values = []
            for j in range(parameter_dim):
                d2_joseph_left_ij = (
                    -d2K[i, j] @ observation_matrix
                    - dK[i] @ d_observation_matrix[j]
                    - dK[j] @ d_observation_matrix[i]
                    - kalman_gain @ d2_observation_matrix[i, j]
                )
                d2_joseph_left_values.append(d2_joseph_left_ij)
            d2_joseph_left_rows.append(tf.stack(d2_joseph_left_values, axis=0))
        d_joseph_left = tf.stack(d_joseph_left_values, axis=0)
        d2_joseph_left = tf.stack(d2_joseph_left_rows, axis=0)

        dmean_values = []
        dcov_values = []
        ddmean_rows = []
        ddcov_rows = []
        for i in range(parameter_dim):
            dmean_i = dpred_mean[i] + _matvec(dK[i], innovation) + _matvec(kalman_gain, dinnovation[i])
            dcov_i = (
                d_joseph_left[i] @ predicted_covariance @ tf.transpose(joseph_left)
                + joseph_left @ dpred_cov[i] @ tf.transpose(joseph_left)
                + joseph_left @ predicted_covariance @ tf.transpose(d_joseph_left[i])
                + dK[i] @ observation_covariance_jittered @ tf.transpose(kalman_gain)
                + kalman_gain @ d_observation_covariance[i] @ tf.transpose(kalman_gain)
                + kalman_gain @ observation_covariance_jittered @ tf.transpose(dK[i])
            )
            dcov_i = 0.5 * (dcov_i + tf.transpose(dcov_i))
            dmean_values.append(dmean_i)
            dcov_values.append(dcov_i)
            ddmean_values = []
            ddcov_values = []
            for j in range(parameter_dim):
                ddmean_ij = d2pred_mean[i, j] + _matvec(d2K[i, j], innovation) + _matvec(dK[i], dinnovation[j]) + _matvec(dK[j], dinnovation[i]) + _matvec(kalman_gain, d2innovation[i, j])
                ddcov_ij = (
                    d2_joseph_left[i, j] @ predicted_covariance @ tf.transpose(joseph_left)
                    + d_joseph_left[i] @ dpred_cov[j] @ tf.transpose(joseph_left)
                    + d_joseph_left[i] @ predicted_covariance @ tf.transpose(d_joseph_left[j])
                    + d_joseph_left[j] @ dpred_cov[i] @ tf.transpose(joseph_left)
                    + joseph_left @ d2pred_cov[i, j] @ tf.transpose(joseph_left)
                    + joseph_left @ dpred_cov[i] @ tf.transpose(d_joseph_left[j])
                    + d_joseph_left[j] @ predicted_covariance @ tf.transpose(d_joseph_left[i])
                    + joseph_left @ dpred_cov[j] @ tf.transpose(d_joseph_left[i])
                    + joseph_left @ predicted_covariance @ tf.transpose(d2_joseph_left[i, j])
                    + d2K[i, j] @ observation_covariance_jittered @ tf.transpose(kalman_gain)
                    + dK[i] @ d_observation_covariance[j] @ tf.transpose(kalman_gain)
                    + dK[i] @ observation_covariance_jittered @ tf.transpose(dK[j])
                    + dK[j] @ d_observation_covariance[i] @ tf.transpose(kalman_gain)
                    + kalman_gain @ d2_observation_covariance[i, j] @ tf.transpose(kalman_gain)
                    + kalman_gain @ d_observation_covariance[i] @ tf.transpose(dK[j])
                    + dK[j] @ observation_covariance_jittered @ tf.transpose(dK[i])
                    + kalman_gain @ d_observation_covariance[j] @ tf.transpose(dK[i])
                    + kalman_gain @ observation_covariance_jittered @ tf.transpose(d2K[i, j])
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

        solve_innovation = tf.linalg.triangular_solve(chol, tf.expand_dims(innovation, axis=-1), lower=True)
        mahalanobis = tf.reduce_sum(tf.square(solve_innovation))
        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
        contrib = -0.5 * (tf.cast(obs_dim, tf.float64) * tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float64)) + log_det + mahalanobis)
        loglik += contrib

        if trace_rows is not None:
            solve_residual = _matvec(innovation_covariance, innovation_solve) - innovation
            trace_rows.append(
                {
                    "t": t,
                    "innovation": np.asarray(innovation.numpy(), dtype=np.float64),
                    "innovation_covariance": np.asarray(innovation_covariance.numpy(), dtype=np.float64),
                    "innovation_precision": np.asarray(innovation_precision.numpy(), dtype=np.float64),
                    "innovation_solve": np.asarray(innovation_solve.numpy(), dtype=np.float64),
                    "dsolve_innovation": np.asarray(dsolve_innovation.numpy(), dtype=np.float64),
                    "d2solve_innovation": np.asarray(d2solve_innovation.numpy(), dtype=np.float64),
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
                    "cholesky_min_pivot": float(tf.reduce_min(tf.linalg.diag_part(chol)).numpy()),
                    "solve_residual_norm": float(tf.linalg.norm(solve_residual).numpy()),
                }
            )

    hess = 0.5 * (hess + tf.transpose(hess))
    if trace_rows is not None:
        return loglik, grad, hess, trace_rows
    return loglik, grad, hess
