"""
TensorFlow Linear Gaussian State Space Model.

Mirrors: models/linear_gaussian.py (NumPy version)

x_t = A @ x_{t-1} + v_t,  v_t ~ N(0, Q)
y_t = C @ x_t + w_t,      w_t ~ N(0, R)
"""

import tensorflow as tf
from .base import TFStateSpaceModel


def make_lgssm(
    A: tf.Tensor,
    C: tf.Tensor,
    Q: tf.Tensor,
    R: tf.Tensor,
    m0: tf.Tensor,
    P0: tf.Tensor,
    dtype: tf.DType = tf.float64,
) -> TFStateSpaceModel:
    """
    Create a TF Linear Gaussian State Space Model.

    Args:
        A: [nx, nx] State transition matrix
        C: [ny, nx] Observation matrix
        Q: [nx, nx] Process noise covariance
        R: [ny, ny] Observation noise covariance
        m0: [nx] Initial state mean
        P0: [nx, nx] Initial state covariance

    Returns:
        TFStateSpaceModel instance
    """
    A = tf.constant(A, dtype=dtype)
    C = tf.constant(C, dtype=dtype)
    Q = tf.constant(Q, dtype=dtype)
    R = tf.constant(R, dtype=dtype)
    m0 = tf.constant(m0, dtype=dtype)
    P0 = tf.constant(P0, dtype=dtype)

    nx = A.shape[0]
    ny = C.shape[0]

    # Ensure symmetry
    Q = 0.5 * (Q + tf.transpose(Q))
    R = 0.5 * (R + tf.transpose(R))
    P0 = 0.5 * (P0 + tf.transpose(P0))

    # Capture A, C as constants in closure — no retracing
    A_t = tf.transpose(A)
    C_t = tf.transpose(C)

    def dynamics_mean(x: tf.Tensor) -> tf.Tensor:
        """x: [N, nx] -> [N, nx]"""
        return x @ A_t

    def dynamics_jacobian(x: tf.Tensor) -> tf.Tensor:
        """x: [nx] -> [nx, nx]"""
        return A

    def obs_mean(x: tf.Tensor) -> tf.Tensor:
        """x: [N, nx] -> [N, ny]"""
        return x @ C_t

    def obs_jacobian(x: tf.Tensor) -> tf.Tensor:
        """x: [nx] -> [ny, nx]"""
        return C

    return TFStateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0,
        initial_cov=P0,
        dynamics_mean=dynamics_mean,
        dynamics_cov=Q,
        dynamics_jacobian=dynamics_jacobian,
        obs_mean=obs_mean,
        obs_cov=R,
        obs_jacobian=obs_jacobian,
        dtype=dtype,
    )
