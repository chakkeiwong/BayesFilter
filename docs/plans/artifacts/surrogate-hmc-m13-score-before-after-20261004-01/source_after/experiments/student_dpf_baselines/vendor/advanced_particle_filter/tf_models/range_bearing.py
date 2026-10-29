"""
TensorFlow Range-Bearing State Space Model with Student-t measurement noise.

Mirrors: models/range_bearing.py (NumPy version)

State: [px, py, vx, vy] - position and velocity
Observation: [range, bearing] with Student-t noise
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import Tuple, Optional

from .base import TFStateSpaceModel


def _wrap_angle(a: tf.Tensor) -> tf.Tensor:
    """Wrap angle to [-pi, pi]."""
    pi = tf.constant(3.141592653589793, dtype=a.dtype)
    return (a + pi) % (2.0 * pi) - pi


def _h_range_bearing(x: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """
    Range-bearing observation function.

    Args:
        x: [N, 4] states with [px, py, vx, vy]  (always batched)
        eps: Small constant for numerical stability

    Returns:
        y: [N, 2] observations [range, bearing]
    """
    px = x[:, 0]
    py = x[:, 1]
    r = tf.sqrt(px ** 2 + py ** 2 + eps)
    th = tf.atan2(py, px)
    return tf.stack([r, th], axis=-1)


def _H_jac_range_bearing(x: tf.Tensor, eps: tf.Tensor) -> tf.Tensor:
    """
    Jacobian of range-bearing observation.

    Args:
        x: [4] state vector
        eps: Small constant

    Returns:
        H: [2, 4] Jacobian matrix
    """
    dtype = x.dtype
    px = x[0]
    py = x[1]
    r2 = px ** 2 + py ** 2 + eps
    r = tf.sqrt(r2)

    dr_dpx = px / r
    dr_dpy = py / r
    dth_dpx = -py / r2
    dth_dpy = px / r2

    zero = tf.constant(0.0, dtype=dtype)
    H = tf.stack([
        tf.stack([dr_dpx, dr_dpy, zero, zero]),
        tf.stack([dth_dpx, dth_dpy, zero, zero]),
    ])
    return H


def make_range_bearing_ssm(
    dt: float = 0.01,
    q_diag: float = 0.01,
    nu: float = 2.0,
    s_r: float = 0.01,
    s_th: float = 0.01,
    m0: Tuple[float, ...] = (1.0, 0.5, 0.01, 0.01),
    P0_diag: Tuple[float, ...] = (0.1, 0.1, 0.1, 0.1),
    eps_val: float = 1e-6,
    dtype: tf.DType = tf.float64,
) -> TFStateSpaceModel:
    """
    Range-Bearing SSM with Student-t measurement noise (TF version).

    Args:
        dt: Time step
        q_diag: Process noise diagonal value
        nu: Degrees of freedom for Student-t
        s_r: Scale for range noise
        s_th: Scale for bearing noise
        m0: Initial state mean
        P0_diag: Initial state variance diagonal
        eps_val: Numerical stability constant
        dtype: TF dtype

    Returns:
        TFStateSpaceModel instance
    """
    nx = 4
    ny = 2

    eps = tf.constant(eps_val, dtype=dtype)
    nu_tf = tf.constant(nu, dtype=dtype)
    s_r_tf = tf.constant(s_r, dtype=dtype)
    s_th_tf = tf.constant(s_th, dtype=dtype)

    # Dynamics: constant velocity
    A = tf.constant([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=dtype)
    A_t = tf.transpose(A)

    Q = tf.linalg.diag(tf.fill([nx], tf.constant(q_diag, dtype=dtype)))
    Q = 0.5 * (Q + tf.transpose(Q)) + eps * tf.eye(nx, dtype=dtype)

    m0_vec = tf.constant(m0, dtype=dtype)
    P0_mat = tf.linalg.diag(tf.constant(P0_diag, dtype=dtype))
    P0_mat = 0.5 * (P0_mat + tf.transpose(P0_mat)) + eps * tf.eye(nx, dtype=dtype)

    # R: Gaussian approximation for flow algorithms
    R = tf.linalg.diag(tf.constant([s_r ** 2, s_th ** 2], dtype=dtype))

    # --- Callables ---

    def dynamics_mean(x: tf.Tensor) -> tf.Tensor:
        return x @ A_t

    def dynamics_jacobian(x: tf.Tensor) -> tf.Tensor:
        return A

    def obs_mean(x: tf.Tensor) -> tf.Tensor:
        return _h_range_bearing(x, eps)

    def obs_jacobian(x: tf.Tensor) -> tf.Tensor:
        return _H_jac_range_bearing(x, eps)

    def observation_log_prob(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        """
        Log probability using Student-t distribution.

        Args:
            x: [N, 4] particles
            y: [2] observation [range, bearing]

        Returns:
            log_prob: [N]
        """
        y_pred = _h_range_bearing(x, eps)

        # Range residual
        res_r = y[0] - y_pred[:, 0]

        # Bearing residual (wrapped)
        res_th = _wrap_angle(y[1] - y_pred[:, 1])

        # Student-t log-likelihoods via tfp
        dist_r = tfp.distributions.StudentT(df=nu_tf, loc=0.0, scale=s_r_tf)
        dist_th = tfp.distributions.StudentT(df=nu_tf, loc=0.0, scale=s_th_tf)

        return dist_r.log_prob(res_r) + dist_th.log_prob(res_th)

    return TFStateSpaceModel(
        state_dim=nx,
        obs_dim=ny,
        initial_mean=m0_vec,
        initial_cov=P0_mat,
        dynamics_mean=dynamics_mean,
        dynamics_cov=Q,
        dynamics_jacobian=dynamics_jacobian,
        obs_mean=obs_mean,
        obs_cov=R,
        obs_jacobian=obs_jacobian,
        obs_log_prob=observation_log_prob,
        dtype=dtype,
    )
