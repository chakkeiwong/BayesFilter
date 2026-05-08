"""
Corenflos's linear-Gaussian state-space model (Corenflos et al. 2021).

State and observation:
    X_{t+1} | X_t ~ N(diag(theta_1, theta_2) X_t, 0.5 I_2)
    Y_t     | X_t ~ N(X_t, 0.1 I_2)

Parameters: theta = (theta_1, theta_2) in R^2.
True for our experiments: theta = (0.5, 0.5).

Implementation note
-------------------
The DPF in tf_filters/differentiable_particle.py is hardcoded to use
SVSSMParams as its parameter container and svssm_sample_initial /
svssm_sample_dynamics for state propagation. To avoid forking the DPF,
we reuse those exact dynamics by setting:
    mu               = 0       (so the SVSSM dynamics  h = mu + Phi(h-mu) + n
                                degenerates to         h = Phi h + n)
    Phi              = diag(theta_1, theta_2)
    Sigma_eta_chol   = sqrt(0.5) * I_2  (chol of the dynamics noise cov)

Then we provide a Corenflos-specific obs_log_prob_fn implementing the
N(X, 0.1 I_2) observation likelihood instead of SVSSM's N(0, exp(h)).
"""

import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .svssm import SVSSMParams


# Constants pinned to Corenflos's spec.
SIGMA_X2 = 0.5      # dynamics noise variance per coordinate
SIGMA_Y2 = 0.1      # observation noise variance per coordinate
STATE_DIM = 2

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Construct SVSSMParams from theta
# ---------------------------------------------------------------------------
def make_corenflos_params(theta: tf.Tensor) -> SVSSMParams:
    """Convert theta = (theta_1, theta_2) to a (degenerate) SVSSMParams.

    Args:
        theta: [B, 2] AR coefficients per chain.

    Returns:
        SVSSMParams with mu=0, Phi=diag(theta), Sigma_eta_chol=sqrt(0.5) I.
    """
    dtype = theta.dtype
    B = tf.shape(theta)[0]

    mu = tf.zeros([B, STATE_DIM], dtype=dtype)

    # Phi = diag(theta) per chain.
    Phi = tf.linalg.diag(theta)                              # [B, 2, 2]

    # Sigma_eta = SIGMA_X2 * I_2  ->  chol = sqrt(SIGMA_X2) * I_2.
    chol_scalar = tf.cast(math.sqrt(SIGMA_X2), dtype)
    eye = tf.eye(STATE_DIM, dtype=dtype, batch_shape=[B])
    Sigma_eta_chol = chol_scalar * eye                       # [B, 2, 2]

    return SVSSMParams(mu=mu, Phi=Phi, Sigma_eta_chol=Sigma_eta_chol)


# ---------------------------------------------------------------------------
# Observation log-prob: N(particles, 0.1 I_2)
# ---------------------------------------------------------------------------
@tf.function(reduce_retracing=True)
def corenflos_lg_observation_log_prob(
    y_t: tf.Tensor,
    particles: tf.Tensor,
) -> tf.Tensor:
    """Log p(y_t | X_t) for y_t | X_t ~ N(X_t, SIGMA_Y2 * I_2).

    Diagonal Gaussian; decomposes per-coordinate.

    Args:
        y_t:       [d]  single observation at time t
        particles: [B, N, d]  state particles
    Returns:
        log_prob: [B, N]
    """
    dtype = particles.dtype
    sigma2 = tf.cast(SIGMA_Y2, dtype)

    log_2pi = tf.constant(math.log(2.0 * math.pi), dtype=dtype)
    # Constant per coord: -0.5 log(2 pi sigma^2).
    log_norm = -0.5 * (log_2pi + tf.math.log(sigma2))
    # Squared residual per coord: (y - x)^2 / (2 sigma^2).
    y_b = y_t[tf.newaxis, tf.newaxis, :]                      # [1, 1, d]
    diff = particles - y_b                                    # [B, N, d]
    sq_res = tf.reduce_sum(tf.square(diff), axis=-1)          # [B, N]

    d_f = tf.cast(tf.shape(particles)[-1], dtype)
    return d_f * log_norm - sq_res / (2.0 * sigma2)


# ---------------------------------------------------------------------------
# Simulate one trajectory
# ---------------------------------------------------------------------------
def simulate_corenflos_lg(theta_true, T: int, seed: int = 0):
    """Simulate one (X_{1:T}, Y_{1:T}) trajectory.

    Initial state X_1 ~ N(0, I_2) (a reasonable diffuse choice; the model
    spec doesn't pin it).

    Args:
        theta_true: [2] true AR coefficients (will be cast to fp64).
        T:          number of timesteps.
        seed:       RNG seed.

    Returns:
        x_true:  [T, 2]  latent states
        y_obs:   [T, 2]  observations
        truth:   dict with theta_true (fp64 tensor)
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta_true, dtype=np.float64)

    x = np.zeros((T, 2))
    y = np.zeros((T, 2))

    # X_1 ~ N(0, I_2)
    x[0] = rng.normal(size=2)
    y[0] = x[0] + math.sqrt(SIGMA_Y2) * rng.normal(size=2)

    for t in range(1, T):
        x[t] = theta * x[t-1] + math.sqrt(SIGMA_X2) * rng.normal(size=2)
        y[t] = x[t] + math.sqrt(SIGMA_Y2) * rng.normal(size=2)

    return (
        tf.constant(x, dtype=tf.float64),
        tf.constant(y, dtype=tf.float64),
        {'theta': tf.constant(theta, dtype=tf.float64)},
    )
