"""
Linear-Gaussian observation variant of the SVSSM for pipeline validation.

Model:
    h_{t+1} = mu + Phi (h_t - mu) + eta_t,    eta_t ~ N(0, Sigma_eta)
    y_t     = h_t + nu_t,                      nu_t  ~ N(0, Sigma_obs)

Same dynamics as SVSSM (same mu, Phi, Sigma_eta to estimate), but the
observation is linear-Gaussian instead of the exp-transformed SVSSM
observation. This means:
  - Kalman filter gives exact log p(y | theta)
  - Any DPF-vs-Kalman gap is purely pipeline bias, not observation nonlinearity
  - Gradient of exact log p(y | theta) is available via autodiff through KF

Provides two interfaces:
  1. make_lg_obs_model(mu, Phi, Sigma_eta_chol, Sigma_obs)
       -> TFStateSpaceModel for use with existing KF / BPF
  2. lg_obs_sample_initial, lg_obs_sample_dynamics, lg_obs_observation_log_prob
       -> batched [B, N, D] functions for use with DPF

Both share the same dynamics; only the observation differs from svssm.py.
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import NamedTuple

tfd = tfp.distributions

# Reuse SVSSMParams — dynamics are identical
from .svssm import SVSSMParams, svssm_sample_initial, svssm_sample_dynamics


# ============================================================================
# Interface 1: TFStateSpaceModel for KF / BPF  (unbatched, [N, nx] convention)
# ============================================================================

def make_lg_obs_model(
    mu,
    Phi,
    Sigma_eta_chol,
    Sigma_obs,
    dtype=tf.float64,
):
    """
    Construct a TFStateSpaceModel with linear-Gaussian observation.

    Because TFStateSpaceModel expects dynamics of the form x_t = f(x_{t-1}) + noise,
    we express the mean-reverting dynamics h_{t+1} = mu + Phi(h_t - mu) + eta
    as h_{t+1} = (I - Phi) mu + Phi h_t + eta, i.e.:
        A = Phi
        bias = (I - Phi) mu   (folded into dynamics_mean)
        Q = Sigma_eta_chol @ Sigma_eta_chol^T
        C = I   (observation matrix)
        R = Sigma_obs

    Args:
        mu:             [d] or array-like
        Phi:            [d, d] or array-like
        Sigma_eta_chol: [d, d] lower-triangular
        Sigma_obs:      [d, d] observation noise covariance
        dtype:          tf.DType

    Returns:
        TFStateSpaceModel instance
    """
    # Lazy import to avoid circular dependency at module level
    from advanced_particle_filter.tf_models.base import TFStateSpaceModel

    mu = tf.constant(mu, dtype=dtype)
    Phi = tf.constant(Phi, dtype=dtype)
    Sigma_eta_chol = tf.constant(Sigma_eta_chol, dtype=dtype)
    Sigma_obs = tf.constant(Sigma_obs, dtype=dtype)

    d = mu.shape[0]

    Q = Sigma_eta_chol @ tf.transpose(Sigma_eta_chol)
    Q = 0.5 * (Q + tf.transpose(Q))

    R = 0.5 * (Sigma_obs + tf.transpose(Sigma_obs))

    # Dynamics: h_{t+1} = Phi h_t + (I - Phi) mu + eta
    bias = tf.linalg.matvec(tf.eye(d, dtype=dtype) - Phi, mu)
    Phi_t = tf.transpose(Phi)

    def dynamics_mean(x):
        """x: [N, d] -> [N, d]"""
        return x @ Phi_t + bias

    def dynamics_jacobian(x):
        """x: [d] -> [d, d]"""
        return Phi

    # Observation: y_t = h_t + noise  =>  C = I
    C = tf.eye(d, dtype=dtype)

    def obs_mean(x):
        """x: [N, d] -> [N, d]"""
        return x

    def obs_jacobian(x):
        """x: [d] -> [d, d]"""
        return C

    # Initial distribution: N(mu, Q) as approximation to stationary
    m0 = mu
    P0 = Q + 1e-6 * tf.eye(d, dtype=dtype)

    return TFStateSpaceModel(
        state_dim=d,
        obs_dim=d,
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


# ============================================================================
# Interface 2: Batched [B, N, D] functions for DPF  (mirrors svssm.py)
# ============================================================================

# sample_initial and sample_dynamics are identical to SVSSM — reuse directly.
# Only observation_log_prob changes.

lg_obs_sample_initial = svssm_sample_initial
lg_obs_sample_dynamics = svssm_sample_dynamics


@tf.function(reduce_retracing=True)
def lg_obs_observation_log_prob(
    y_t: tf.Tensor,
    particles: tf.Tensor,
    Sigma_obs_chol: tf.Tensor,
) -> tf.Tensor:
    """
    Log p(y_t | h_t) for the linear-Gaussian observation y_t = h_t + nu_t.

    Args:
        y_t:            [d]  observation at time t
        particles:      [B, N, d]  state particles
        Sigma_obs_chol: [d, d]  Cholesky factor of observation noise covariance
                        (shared across batch, not parameter-dependent)

    Returns:
        log_prob: [B, N]
    """
    # Residual: y_t - h_t, broadcast to [B, N, d]
    residual = y_t[tf.newaxis, tf.newaxis, :] - particles  # [B, N, d]

    # Mahalanobis distance via Cholesky solve:
    # solved = L^{-1} residual^T  for each (b, n)
    # Reshape to [d, B*N], solve, reshape back
    B = tf.shape(particles)[0]
    N = tf.shape(particles)[1]
    d = tf.shape(particles)[2]
    dtype = particles.dtype

    flat_res = tf.reshape(residual, [B * N, d])       # [B*N, d]
    # L^{-1} @ flat_res^T : [d, B*N]
    solved = tf.linalg.triangular_solve(
        Sigma_obs_chol,
        tf.transpose(flat_res),
        lower=True,
    )
    mahal_sq = tf.reduce_sum(solved ** 2, axis=0)     # [B*N]

    # log det of Sigma_obs = 2 * sum log diag(L)
    log_det = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(Sigma_obs_chol))
    )

    d_f = tf.cast(d, dtype)
    pi = tf.constant(3.141592653589793, dtype=dtype)
    log_prob = -0.5 * (d_f * tf.math.log(2.0 * pi) + log_det + mahal_sq)

    return tf.reshape(log_prob, [B, N])               # [B, N]


# ============================================================================
# Data simulation (unbatched, for generating synthetic data)
# ============================================================================

def simulate_lg_obs(
    mu: tf.Tensor,
    Phi: tf.Tensor,
    Sigma_eta_chol: tf.Tensor,
    Sigma_obs_chol: tf.Tensor,
    T: int,
    rng: tf.random.Generator,
):
    """
    Simulate one trajectory from the linear-Gaussian-observation model.

    Args:
        mu:             [d]
        Phi:            [d, d]
        Sigma_eta_chol: [d, d]
        Sigma_obs_chol: [d, d]
        T:              int
        rng:            tf.random.Generator

    Returns:
        h: [T, d]  latent trajectory
        y: [T, d]  observations
    """
    dtype = mu.dtype
    d = tf.shape(mu)[0]

    h_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)
    y_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)

    h_curr = mu

    for t in tf.range(T):
        def propagate():
            centered = h_curr - mu
            mean = mu + tf.linalg.matvec(Phi, centered)
            eta = tf.linalg.matvec(
                Sigma_eta_chol, rng.normal(shape=[d], dtype=dtype)
            )
            return mean + eta

        h_new = tf.cond(t > 0, propagate, lambda: h_curr)
        h_ta = h_ta.write(t, h_new)

        # y_t = h_t + nu_t,  nu_t ~ N(0, Sigma_obs)
        nu = tf.linalg.matvec(
            Sigma_obs_chol, rng.normal(shape=[d], dtype=dtype)
        )
        y_ta = y_ta.write(t, h_new + nu)

        h_curr = h_new

    return h_ta.stack(), y_ta.stack()
