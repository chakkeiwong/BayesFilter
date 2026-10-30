"""
HMC parameterization for Corenflos's bivariate linear-Gaussian SSM.

Mapping: identity. z[:, 0] = theta_1, z[:, 1] = theta_2.
Stability is enforced by a soft barrier in the prior, not by a transform.

The unpack returns SVSSMParams (the existing DPF parameter container)
populated as: mu=0, Phi=diag(theta), Sigma_eta_chol=sqrt(0.5) I.
This lets the unmodified DPF run on this model (see corenflos_lg.py for
the rationale).
"""

import numpy as np
import tensorflow as tf

from ..tf_models.corenflos_lg import make_corenflos_params


STATE_DIM = 2
TOTAL_DIM = 2          # |theta| = 2
THETA_START, THETA_END = 0, 2


@tf.function(reduce_retracing=True)
def unpack_batched(z: tf.Tensor):
    """Unpack [B, 2] z into a (degenerate) SVSSMParams container.

    Identity map: theta = z. Stability is enforced via the prior barrier.
    """
    return make_corenflos_params(z)


@tf.function(reduce_retracing=True)
def log_prior_batched(
    z: tf.Tensor,
    theta_scale: float = 1.0,
    barrier_weight: float = 100.0,
    barrier_threshold: float = 0.98,
) -> tf.Tensor:
    """Soft-barrier prior on theta.

      - Weakly informative N(0, theta_scale^2) on each theta_i.
      - Soft barrier penalising |theta_i| > barrier_threshold (default 0.98).
        At theta_i = 1, penalty is barrier_weight * (1 - 0.98)^2 = 0.04 *
        barrier_weight, mild but enough to deter the chain from drifting
        into the non-stationary regime.

    No Jacobian correction needed (identity transformation).

    Args:
        z: [B, 2]
    Returns:
        log_prior: [B]
    """
    dtype = z.dtype
    s = tf.cast(theta_scale, dtype)
    bw = tf.cast(barrier_weight, dtype)
    bt = tf.cast(barrier_threshold, dtype)

    lp_gauss = -0.5 * tf.reduce_sum(tf.square(z / s), axis=-1)
    excess = tf.maximum(tf.abs(z) - bt, tf.cast(0.0, dtype))
    lp_barrier = -bw * tf.reduce_sum(tf.square(excess), axis=-1)
    return lp_gauss + lp_barrier


# ---------------------------------------------------------------------------
# Yule-Walker data-driven warm start
# ---------------------------------------------------------------------------
def warm_start_corenflos_lg(
    y_obs: tf.Tensor,
    B_chain: int = 4,
    jitter: float = 0.05,
    seed: int = 0,
) -> tf.Tensor:
    """Yule-Walker warm start.

    Given observations Y_{1:T} ~= X_{1:T} + small noise (since obs noise
    is 0.1 vs dynamics noise 0.5), we estimate theta_i by lag-1 sample
    autocorrelation of Y[:, i].

    Replicate to B_chain chains and add small Gaussian jitter.

    Args:
        y_obs:   [T, 2]
        B_chain: number of chains.
        jitter:  std of Gaussian noise added per chain.
        seed:    RNG seed.

    Returns:
        z0: [B_chain, 2]
    """
    Y = y_obs.numpy() if hasattr(y_obs, 'numpy') else np.asarray(y_obs)
    Y_centered = Y - Y.mean(axis=0)
    var_Y = (Y_centered ** 2).mean(axis=0)
    cov_lag1 = (Y_centered[:-1] * Y_centered[1:]).mean(axis=0)
    theta_init = cov_lag1 / np.maximum(var_Y, 1e-8)
    theta_init = np.clip(theta_init, -0.95, 0.95)

    rng = np.random.default_rng(seed)
    z0 = np.tile(theta_init[None, :], (B_chain, 1))
    z0 = z0 + jitter * rng.normal(size=z0.shape)
    return tf.constant(z0, dtype=tf.float64)
