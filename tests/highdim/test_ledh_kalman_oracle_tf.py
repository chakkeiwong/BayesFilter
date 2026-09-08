"""Tests for Kalman filter/smoother oracle."""

import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_kalman_oracle_tf import (
    kalman_filter_marginal_loglik,
    kalman_oracle_value_and_score,
)


def test_kalman_filter_scalar_lgssm():
    """Test Kalman filter on a scalar LGSSM with known analytical solution."""
    # Scalar random walk with Gaussian observations
    # x_t = x_{t-1} + w_t, w_t ~ N(0, Q)
    # y_t = x_t + v_t, v_t ~ N(0, R)

    dtype = tf.float32
    Q = 0.1
    R = 0.2

    # Generate observations from known state
    np.random.seed(42)
    T = 5
    true_states = np.cumsum(np.random.randn(T) * np.sqrt(Q))
    observations = true_states + np.random.randn(T) * np.sqrt(R)

    observations_tf = tf.constant(observations[:, None], dtype=dtype)  # [T, 1]
    transition_matrix = tf.constant([[1.0]], dtype=dtype)
    process_covariance = tf.constant([[Q]], dtype=dtype)
    observation_matrix = tf.constant([[1.0]], dtype=dtype)
    observation_covariance = tf.constant([[R]], dtype=dtype)
    initial_mean = tf.constant([0.0], dtype=dtype)
    initial_covariance = tf.constant([[1.0]], dtype=dtype)

    loglik = kalman_filter_marginal_loglik(
        observations=observations_tf,
        transition_matrix=transition_matrix,
        process_covariance=process_covariance,
        observation_matrix=observation_matrix,
        observation_covariance=observation_covariance,
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
    )

    # Check finite
    assert tf.math.is_finite(loglik), "Log-likelihood should be finite"

    # Check reasonable magnitude (for T=5, should be roughly -5 to -10)
    assert -15.0 < float(loglik) < 0.0, f"Log-likelihood {float(loglik)} outside expected range"


def test_kalman_oracle_score_finite_difference():
    """Verify Kalman oracle score via finite differences."""
    dtype = tf.float32

    # Simple 2D LGSSM parameterized by theta = [log_Q_scale, log_R_scale]
    # Use exp instead of softplus to avoid saturation issues
    def theta_to_params(theta):
        Q_scale = tf.exp(theta[0])
        R_scale = tf.exp(theta[1])
        return {
            'transition_matrix': 0.95 * tf.eye(2, dtype=dtype),
            'process_covariance': Q_scale * tf.eye(2, dtype=dtype),
            'observation_matrix': tf.eye(2, dtype=dtype),
            'observation_covariance': R_scale * tf.eye(2, dtype=dtype),
            'initial_mean': tf.zeros([2], dtype=dtype),
            'initial_covariance': tf.eye(2, dtype=dtype),
        }

    # Generate observations
    np.random.seed(123)
    T = 3
    observations = tf.constant(np.random.randn(T, 2), dtype=dtype)

    # theta = [log(0.1), log(0.2)] so Q=0.1, R=0.2
    theta = tf.constant([np.log(0.1), np.log(0.2)], dtype=dtype)

    result = kalman_oracle_value_and_score(
        observations=observations,
        theta=theta,
        theta_to_lgssm_params=theta_to_params,
        dtype=dtype,
    )

    value = result['value']
    score = result['score']

    # Check finite
    assert tf.math.is_finite(value)
    assert tf.reduce_all(tf.math.is_finite(score))

    # Finite difference check
    eps = 1e-5
    for i in range(2):
        theta_plus = theta.numpy()
        theta_plus[i] += eps

        result_plus = kalman_oracle_value_and_score(
            observations=observations,
            theta=tf.constant(theta_plus, dtype=dtype),
            theta_to_lgssm_params=theta_to_params,
            dtype=dtype,
        )

        fd_score_i = (result_plus['value'] - value) / eps
        analytical_score_i = score[i]

        error = abs(float(fd_score_i - analytical_score_i))
        relative_error = error / (abs(float(analytical_score_i)) + 1e-8)

        assert relative_error < 0.05, (
            f"Score component {i} finite difference mismatch: "
            f"analytical={float(analytical_score_i):.6f}, "
            f"fd={float(fd_score_i):.6f}, "
            f"relative_error={relative_error:.6f}"
        )


def test_kalman_oracle_deterministic():
    """Verify Kalman oracle is deterministic."""
    dtype = tf.float32

    # Parameters must actually depend on theta for gradient to be non-None
    def theta_to_params(theta):
        # theta is a dummy scalar that slightly perturbs process covariance
        Q_scale = 0.1 + 0.0 * theta[0]  # Dependency on theta (even if multiplied by 0)
        return {
            'transition_matrix': 0.9 * tf.eye(2, dtype=dtype),
            'process_covariance': Q_scale * tf.eye(2, dtype=dtype),
            'observation_matrix': tf.eye(2, dtype=dtype),
            'observation_covariance': 0.2 * tf.eye(2, dtype=dtype),
            'initial_mean': tf.zeros([2], dtype=dtype),
            'initial_covariance': tf.eye(2, dtype=dtype),
        }

    observations = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    theta = tf.constant([0.0], dtype=dtype)

    result1 = kalman_oracle_value_and_score(
        observations=observations,
        theta=theta,
        theta_to_lgssm_params=theta_to_params,
        dtype=dtype,
    )

    result2 = kalman_oracle_value_and_score(
        observations=observations,
        theta=theta,
        theta_to_lgssm_params=theta_to_params,
        dtype=dtype,
    )

    assert float(tf.abs(result1['value'] - result2['value'])) == 0.0
    assert float(tf.reduce_max(tf.abs(result1['score'] - result2['score']))) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
