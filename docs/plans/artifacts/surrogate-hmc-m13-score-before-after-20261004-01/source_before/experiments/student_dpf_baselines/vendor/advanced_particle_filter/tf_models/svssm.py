"""
TensorFlow Stochastic Volatility State Space Model (SVSSM) for
differentiable calibration via HMC.

Model:
    h_{t+1} = mu + Phi (h_t - mu) + eta_t,   eta_t ~ N(0, Sigma_eta)
    y_t | h_t ~ N(0, Omega_t)
    Omega_t = diag(exp(h_t/2)) P diag(exp(h_t/2))

For this PoC we fix P = I, so Omega_t = diag(exp(h_t)).

WHY THIS IS NOT A `TFStateSpaceModel` SUBCLASS
----------------------------------------------
The base class `TFStateSpaceModel` in tf_models/base.py wraps all
parameters as `tf.constant(...)` in __init__ and precomputes Cholesky
factors at construction. That is correct for fixed-parameter filtering
but breaks HMC, which needs parameters to flow as traced tensors so that
the log-likelihood is differentiable w.r.t. them.

Rather than modify the base class (risking regression in existing
filters), we define `SVSSMParams` — a thin container holding the
parameter tensors as traced attributes — plus `make_svssm_fns(params)`
that returns the per-call functions needed by the differentiable PF.

BATCH CONVENTION
----------------
All per-batch-element parameters have a leading batch dimension B which
combines HMC chains × MC replicas (flattened):
    mu_batched:       [B, d]
    Phi_batched:      [B, d, d]
    Sigma_eta_chol:   [B, d, d]  lower triangular

Particle tensors have shape [B, N, d].
"""

import tensorflow as tf
import tensorflow_probability as tfp
from typing import NamedTuple

tfd = tfp.distributions


class SVSSMParams(NamedTuple):
    """
    Batched SVSSM parameters. All fields are tf.Tensor with leading batch
    dim B = B_chain * B_mc.

    Attributes:
        mu:             [B, d]
        Phi:            [B, d, d]
        Sigma_eta_chol: [B, d, d]  lower triangular, positive diagonal
    """
    mu: tf.Tensor
    Phi: tf.Tensor
    Sigma_eta_chol: tf.Tensor


# ============================================================================
# Dynamics and observation log-probs — all batched, all JIT-safe
# ============================================================================

@tf.function(reduce_retracing=True)
def svssm_sample_initial(
    params: SVSSMParams,
    n_particles: int,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Sample initial particles approximately from the stationary distribution.

    Approximation: use N(mu, Sigma_eta) rather than the true stationary
    covariance (which solves discrete Lyapunov Sigma = Phi Sigma Phi^T +
    Sigma_eta). For a weakly informative start, Sigma_eta is fine.

    Args:
        params: SVSSMParams with batch dim B
        n_particles: number of particles per batch element
        rng: tf.random.Generator

    Returns:
        particles: [B, N, d]
    """
    B = tf.shape(params.mu)[0]
    d = tf.shape(params.mu)[1]
    dtype = params.mu.dtype

    # noise ~ N(0, I) of shape [B, N, d]; scale by Cholesky
    noise_std = rng.normal(shape=[B, n_particles, d], dtype=dtype)
    # For each b:  noise_b @ L_b^T   where L_b = Sigma_eta_chol[b]
    # This is einsum('bij,bnj->bni'): L_b^T applied to each noise vector
    noise = tf.einsum('bij,bnj->bni', params.Sigma_eta_chol, noise_std)

    # Broadcast mu: [B, 1, d]
    mean = params.mu[:, tf.newaxis, :]
    return mean + noise


@tf.function(reduce_retracing=True)
def svssm_sample_dynamics(
    particles: tf.Tensor,
    params: SVSSMParams,
    rng: tf.random.Generator,
) -> tf.Tensor:
    """
    Propagate particles one step:  h_{t+1} = mu + Phi (h_t - mu) + eta.

    Args:
        particles: [B, N, d] current h_t values
        params:    SVSSMParams (batch dim B matches)
        rng:       tf.random.Generator

    Returns:
        next_particles: [B, N, d]
    """
    B = tf.shape(particles)[0]
    N = tf.shape(particles)[1]
    d = tf.shape(particles)[2]
    dtype = particles.dtype

    # centered = h_t - mu,  shape [B, N, d]
    centered = particles - params.mu[:, tf.newaxis, :]

    # Phi @ centered, batched:  einsum('bij,bnj->bni')
    propagated = tf.einsum('bij,bnj->bni', params.Phi, centered)

    mean = params.mu[:, tf.newaxis, :] + propagated  # [B, N, d]

    # Innovation noise: eta ~ N(0, Sigma_eta)
    noise_std = rng.normal(shape=[B, N, d], dtype=dtype)
    noise = tf.einsum('bij,bnj->bni', params.Sigma_eta_chol, noise_std)

    return mean + noise


@tf.function(reduce_retracing=True)
def svssm_observation_log_prob(
    y_t: tf.Tensor,
    particles: tf.Tensor,
) -> tf.Tensor:
    """
    Log p(y_t | h_t) for y_t | h_t ~ N(0, diag(exp(h_t))) (P = I).

    Decomposes per-dimension (diagonal covariance):
        log p(y|h) = sum_i [-0.5 log(2 pi) - h_i/2 - 0.5 y_i^2 exp(-h_i)]

    More numerically stable than building the full covariance and calling
    MultivariateNormalDiag.

    Args:
        y_t:       [d]  single observation at time t
        particles: [B, N, d]  log-variance particles

    Returns:
        log_prob: [B, N]
    """
    dtype = particles.dtype
    # Broadcast y_t over [B, N, d]: y_t has shape [d] -> [1, 1, d]
    y_b = y_t[tf.newaxis, tf.newaxis, :]

    log_2pi = tf.constant(1.8378770664093453, dtype=dtype)  # log(2 pi)
    per_dim = (
        -0.5 * log_2pi
        - particles / 2.0
        - 0.5 * tf.square(y_b) * tf.exp(-particles)
    )  # [B, N, d]

    return tf.reduce_sum(per_dim, axis=-1)  # [B, N]


# ============================================================================
# Simulation (data generation) — NumPy-free, runs eagerly
# ============================================================================

def simulate_svssm(
    mu: tf.Tensor,
    Phi: tf.Tensor,
    Sigma_eta_chol: tf.Tensor,
    T: int,
    rng: tf.random.Generator,
):
    """
    Simulate one trajectory from the SVSSM. Non-batched (single chain).

    Args:
        mu:             [d]
        Phi:            [d, d]
        Sigma_eta_chol: [d, d] lower triangular
        T:              int
        rng:            tf.random.Generator

    Returns:
        h: [T, d]  latent log-variance trajectory
        y: [T, d]  observations
    """
    dtype = mu.dtype
    d = tf.shape(mu)[0]

    h_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)
    y_ta = tf.TensorArray(dtype=dtype, size=T, dynamic_size=False)

    h_curr = mu  # start at stationary mean

    for t in tf.range(T):
        # h_{t+1} = mu + Phi(h_t - mu) + eta    (skip for t=0: h[0] = mu)
        def propagate():
            centered = h_curr - mu
            mean = mu + tf.linalg.matvec(Phi, centered)
            eta_std = rng.normal(shape=[d], dtype=dtype)
            eta = tf.linalg.matvec(Sigma_eta_chol, eta_std)
            return mean + eta

        h_new = tf.cond(t > 0, propagate, lambda: h_curr)
        h_ta = h_ta.write(t, h_new)

        # y_t ~ N(0, diag(exp(h_new)))
        std = tf.exp(h_new / 2.0)
        w = rng.normal(shape=[d], dtype=dtype)
        y_ta = y_ta.write(t, std * w)

        h_curr = h_new

    return h_ta.stack(), y_ta.stack()
