"""Kalman filter/smoother oracle for LGSSM analytical log-likelihood and score.

This module provides exact analytical log-likelihood and its parameter gradient
for linear-Gaussian state-space models via the Kalman filter and RTS smoother.
It serves as the primary oracle for Phase 4A integrated KDM score validation.

The oracle uses TensorFlow's GradientTape to compute the exact score, which is
the gold standard for Phase 4A paired score MSE comparisons.
"""

from __future__ import annotations

import tensorflow as tf


def kalman_filter_marginal_loglik(
    observations: tf.Tensor,
    transition_matrix: tf.Tensor,
    process_covariance: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    initial_mean: tf.Tensor,
    initial_covariance: tf.Tensor,
) -> tf.Tensor:
    """
    Compute exact marginal log-likelihood for LGSSM via Kalman filter.

    Standard Kalman filter recursion with innovation form. Returns the sum
    of innovation log-likelihoods across all time steps.

    Args:
        observations: [T, obs_dim] observed sequence
        transition_matrix: [state_dim, state_dim] state transition F
        process_covariance: [state_dim, state_dim] process noise Q
        observation_matrix: [obs_dim, state_dim] observation map C
        observation_covariance: [obs_dim, obs_dim] observation noise R
        initial_mean: [state_dim] initial state mean
        initial_covariance: [state_dim, state_dim] initial state covariance

    Returns:
        scalar log p(y_{1:T} | theta)
    """
    dtype = observations.dtype
    T = tf.shape(observations)[0]
    obs_dim = tf.shape(observations)[1]

    # Initialize
    mean = initial_mean  # [state_dim]
    cov = initial_covariance  # [state_dim, state_dim]

    loglik = tf.constant(0.0, dtype=dtype)
    log_2pi = tf.constant(1.8378770664093453, dtype=dtype)  # log(2π)

    for t in range(T):
        y_t = observations[t]  # [obs_dim]

        # Predict
        mean_pred = tf.linalg.matvec(transition_matrix, mean)
        cov_pred = (
            transition_matrix @ cov @ tf.transpose(transition_matrix)
            + process_covariance
        )

        # Innovation
        y_pred = tf.linalg.matvec(observation_matrix, mean_pred)
        innovation = y_t - y_pred  # [obs_dim]

        S = (
            observation_matrix @ cov_pred @ tf.transpose(observation_matrix)
            + observation_covariance
        )  # [obs_dim, obs_dim]

        # Log-likelihood contribution
        S_chol = tf.linalg.cholesky(S)
        log_det_S = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(S_chol)))

        innovation_norm_sq = tf.reduce_sum(
            tf.square(tf.linalg.triangular_solve(S_chol, innovation[:, None], lower=True))
        )

        loglik_t = -0.5 * (
            tf.cast(obs_dim, dtype) * log_2pi
            + log_det_S
            + innovation_norm_sq
        )
        loglik = loglik + loglik_t

        # Update
        K = cov_pred @ tf.transpose(observation_matrix) @ tf.linalg.inv(S)
        mean = mean_pred + tf.linalg.matvec(K, innovation)
        cov = cov_pred - K @ S @ tf.transpose(K)

    return loglik


def kalman_oracle_value_and_score(
    observations: tf.Tensor,
    theta: tf.Tensor,
    theta_to_lgssm_params,
    dtype: tf.dtypes.DType = tf.float32,
) -> dict[str, tf.Tensor]:
    """
    Compute exact LGSSM log-likelihood and its gradient via Kalman filter.

    This is the Phase 4A oracle: exact analytical log p(y | theta) and
    its gradient with respect to theta.

    Args:
        observations: [T, obs_dim] observed sequence
        theta: [theta_dim] parameter vector
        theta_to_lgssm_params: Callable that maps theta to LGSSM parameters:
            {
                'transition_matrix': [state_dim, state_dim],
                'process_covariance': [state_dim, state_dim],
                'observation_matrix': [obs_dim, state_dim],
                'observation_covariance': [obs_dim, obs_dim],
                'initial_mean': [state_dim],
                'initial_covariance': [state_dim, state_dim],
            }
            All returned tensors must be TensorFlow ops that depend on theta
            for gradient tracking to work.
        dtype: TensorFlow dtype

    Returns:
        {
            'value': scalar log-likelihood,
            'score': [theta_dim] gradient of log-likelihood,
        }
    """
    observations = tf.convert_to_tensor(observations, dtype=dtype)

    @tf.function
    def _compute_loglik(theta_inner):
        params = theta_to_lgssm_params(theta_inner)

        return kalman_filter_marginal_loglik(
            observations=observations,
            transition_matrix=params['transition_matrix'],
            process_covariance=params['process_covariance'],
            observation_matrix=params['observation_matrix'],
            observation_covariance=params['observation_covariance'],
            initial_mean=params['initial_mean'],
            initial_covariance=params['initial_covariance'],
        )

    theta = tf.convert_to_tensor(theta, dtype=dtype)

    with tf.GradientTape() as tape:
        tape.watch(theta)
        loglik = _compute_loglik(theta)

    score = tape.gradient(loglik, theta)

    return {
        'value': loglik,
        'score': score,
    }


__all__ = [
    'kalman_filter_marginal_loglik',
    'kalman_oracle_value_and_score',
]
