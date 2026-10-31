"""
Landscape diagnostics for validating the DPF-HMC pipeline.

Compares log p(y | theta) and its gradient across methods:
  - Kalman filter (exact, ground truth for linear-Gaussian models)
  - Bootstrap particle filter (consistent estimator, no differentiable resampling)
  - DPF with soft resampling (differentiable, ST-biased)
  - DPF with OT/Sinkhorn resampling (differentiable, less biased)

The core idea: if the DPF log-likelihood landscape agrees with Kalman/BPF
across a region of theta-space, HMC will sample the correct posterior.
If it diverges, the posterior is biased — and the shape of the divergence
tells us *how* it's biased.

All functions follow the TF migration guideline:
  - tf.function on compute-heavy paths
  - tf.random.Generator for reproducibility
  - tf.float64 throughout
  - No Python control flow on tensor values
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Callable


DTYPE = tf.float64


# ============================================================================
# Method wrappers: each returns log p(y | theta) for a single theta
# ============================================================================

def kalman_log_lik(
    theta_np: np.ndarray,
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    make_model_fn: Callable,
) -> float:
    """
    Exact log p(y | theta) via standalone Kalman filter.

    Uses kalman_log_likelihood from kalman_ll.py which correctly handles
    affine dynamics h_{t+1} = Phi h_t + (I - Phi) mu + noise.
    (The TFKalmanFilter class in tf_filters/kalman.py only handles
    zero-mean linear dynamics F @ m, missing the bias term.)

    Args:
        theta_np:     [9] numpy array, unconstrained parameters
        observations: [T, d]
        Sigma_obs:    [d, d] fixed observation noise
        make_model_fn: unused (kept for API compatibility)

    Returns:
        scalar log-likelihood
    """
    from ..hmc.parameterization import unpack_batched
    from .kalman_ll import kalman_log_likelihood

    z = tf.constant(theta_np[np.newaxis, :], dtype=DTYPE)  # [1, 9]
    params = unpack_batched(z)
    mu = params.mu[0]
    Phi = params.Phi[0]
    Sigma_eta_chol = params.Sigma_eta_chol[0]

    ll = kalman_log_likelihood(mu, Phi, Sigma_eta_chol, Sigma_obs, observations)
    return float(ll.numpy())


def kalman_log_lik_and_grad(
    theta_np: np.ndarray,
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    make_model_fn: Callable,
) -> tuple:
    """
    Exact log p(y | theta) and its gradient via standalone differentiable
    Kalman filter. Gradients flow through all parameters because the
    Kalman recursion operates on raw tensors, not tf.constant-wrapped ones.

    Returns:
        (log_lik: float, grad: np.ndarray of shape [9])
    """
    from ..hmc.parameterization import unpack_batched
    from .kalman_ll import kalman_log_likelihood

    z = tf.Variable(theta_np[np.newaxis, :].astype(np.float64))  # [1, 9]

    with tf.GradientTape() as tape:
        params = unpack_batched(z)
        mu = params.mu[0]
        Phi = params.Phi[0]
        Sigma_eta_chol = params.Sigma_eta_chol[0]

        ll = kalman_log_likelihood(
            mu, Phi, Sigma_eta_chol, Sigma_obs, observations
        )

    grad = tape.gradient(ll, z)
    return float(ll.numpy()), grad.numpy()[0]  # [9]


def bpf_log_lik(
    theta_np: np.ndarray,
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    make_model_fn: Callable,
    n_particles: int = 500,
    seed: int = 0,
) -> float:
    """
    Log p_hat(y | theta) via bootstrap particle filter.

    Args:
        theta_np:     [9] unconstrained parameters
        observations: [T, d]
        Sigma_obs:    [d, d]
        make_model_fn: function(mu, Phi, L_chol, Sigma_obs) -> TFStateSpaceModel
        n_particles:  int
        seed:         int

    Returns:
        scalar log-likelihood estimate
    """
    from advanced_particle_filter.tf_filters.particle import TFBootstrapParticleFilter
    from ..hmc.parameterization import unpack_batched

    z = tf.constant(theta_np[np.newaxis, :], dtype=DTYPE)
    params = unpack_batched(z)
    mu = params.mu[0]
    Phi = params.Phi[0]
    Sigma_eta_chol = params.Sigma_eta_chol[0]

    model = make_model_fn(mu.numpy(), Phi.numpy(), Sigma_eta_chol.numpy(),
                          Sigma_obs.numpy())
    bpf = TFBootstrapParticleFilter(n_particles=n_particles, seed=seed)
    rng = tf.random.Generator.from_seed(seed)
    result = bpf.filter(model, observations, rng=rng)
    return float(result.log_likelihood.numpy())


def dpf_log_lik_and_grad(
    theta_np: np.ndarray,
    observations: tf.Tensor,
    dpf,
    rng: tf.random.Generator,
) -> tuple:
    """
    Log p_hat(y | theta) and gradient via DPF (soft or OT).

    Uses the batched DPF pipeline. B=1 (single chain, single MC replica).
    The DPF instance already has obs_log_prob_fn configured at construction.

    Args:
        theta_np:       [9] unconstrained parameters
        observations:   [T, d]
        dpf:            TFDifferentiableParticleFilter instance
        rng:            tf.random.Generator

    Returns:
        (log_lik: float, grad: np.ndarray of shape [9])
    """
    from ..hmc.parameterization import unpack_batched

    z = tf.Variable(theta_np[np.newaxis, :].astype(np.float64))  # [1, 9]

    with tf.GradientTape() as tape:
        params = unpack_batched(z)
        result = dpf.filter(params, observations, rng)
        ll = result.log_evidence[0]  # scalar

    grad = tape.gradient(ll, z)
    return float(ll.numpy()), grad.numpy()[0]  # [9]


# ============================================================================
# Landscape sweep functions
# ============================================================================

def landscape_1d(
    theta_center: np.ndarray,
    param_idx: int,
    param_range: float,
    n_grid: int,
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    make_model_fn: Callable,
    dpf_soft,
    dpf_ot,
    rng_soft: tf.random.Generator,
    rng_ot: tf.random.Generator,
    n_seeds_bpf: int = 10,
    n_seeds_dpf: int = 5,
    n_particles_bpf: int = 500,
) -> dict:
    """
    Sweep one parameter dimension and compute log-likelihood + gradient
    for all methods at each grid point.

    Args:
        theta_center:   [9] center point (e.g., theta_true in unconstrained space)
        param_idx:      which parameter to sweep (0-8)
        param_range:    sweep from center - range to center + range
        n_grid:         number of grid points
        observations:   [T, d]
        Sigma_obs:      [d, d]
        make_model_fn:  factory for TFStateSpaceModel
        dpf_soft:       TFDifferentiableParticleFilter with resampler='soft'
        dpf_ot:         TFDifferentiableParticleFilter with resampler='sinkhorn'
        rng_soft, rng_ot: tf.random.Generator
        n_seeds_bpf:    BPF replicates per grid point
        n_seeds_dpf:    DPF replicates per grid point
        n_particles_bpf: particles for BPF

    Returns:
        dict with keys:
            'grid':         [n_grid] parameter values
            'kalman_ll':    [n_grid] exact log-likelihood
            'kalman_grad':  [n_grid] exact gradient for this param
            'bpf_ll_mean':  [n_grid] BPF mean log-lik
            'bpf_ll_std':   [n_grid] BPF std
            'soft_ll_mean': [n_grid]
            'soft_ll_std':  [n_grid]
            'soft_grad_mean': [n_grid] autodiff gradient mean
            'soft_grad_std':  [n_grid]
            'ot_ll_mean':   [n_grid]
            'ot_ll_std':    [n_grid]
            'ot_grad_mean': [n_grid]
            'ot_grad_std':  [n_grid]
    """
    grid = np.linspace(
        theta_center[param_idx] - param_range,
        theta_center[param_idx] + param_range,
        n_grid,
    )

    results = {
        'grid': grid,
        'param_idx': param_idx,
        'kalman_ll': np.zeros(n_grid),
        'kalman_grad': np.zeros(n_grid),
        'bpf_ll_mean': np.zeros(n_grid),
        'bpf_ll_std': np.zeros(n_grid),
        'soft_ll_mean': np.zeros(n_grid),
        'soft_ll_std': np.zeros(n_grid),
        'soft_grad_mean': np.zeros(n_grid),
        'soft_grad_std': np.zeros(n_grid),
        'ot_ll_mean': np.zeros(n_grid),
        'ot_ll_std': np.zeros(n_grid),
        'ot_grad_mean': np.zeros(n_grid),
        'ot_grad_std': np.zeros(n_grid),
    }

    for i, val in enumerate(grid):
        theta = theta_center.copy()
        theta[param_idx] = val

        # --- Kalman (exact) ---
        kl, kg = kalman_log_lik_and_grad(
            theta, observations, Sigma_obs, make_model_fn
        )
        results['kalman_ll'][i] = kl
        results['kalman_grad'][i] = kg[param_idx]

        # --- BPF (multiple seeds) ---
        bpf_lls = []
        for s in range(n_seeds_bpf):
            ll = bpf_log_lik(
                theta, observations, Sigma_obs, make_model_fn,
                n_particles=n_particles_bpf, seed=s * 1000 + i,
            )
            bpf_lls.append(ll)
        results['bpf_ll_mean'][i] = np.mean(bpf_lls)
        results['bpf_ll_std'][i] = np.std(bpf_lls)

        # --- DPF soft (multiple seeds) ---
        soft_lls, soft_grads = [], []
        for s in range(n_seeds_dpf):
            ll, g = dpf_log_lik_and_grad(
                theta, observations, dpf_soft, rng_soft,
            )
            soft_lls.append(ll)
            soft_grads.append(g[param_idx])
        results['soft_ll_mean'][i] = np.mean(soft_lls)
        results['soft_ll_std'][i] = np.std(soft_lls)
        results['soft_grad_mean'][i] = np.mean(soft_grads)
        results['soft_grad_std'][i] = np.std(soft_grads)

        # --- DPF OT (multiple seeds) ---
        ot_lls, ot_grads = [], []
        for s in range(n_seeds_dpf):
            ll, g = dpf_log_lik_and_grad(
                theta, observations, dpf_ot, rng_ot,
            )
            ot_lls.append(ll)
            ot_grads.append(g[param_idx])
        results['ot_ll_mean'][i] = np.mean(ot_lls)
        results['ot_ll_std'][i] = np.std(ot_lls)
        results['ot_grad_mean'][i] = np.mean(ot_grads)
        results['ot_grad_std'][i] = np.std(ot_grads)

        print(f"  grid[{i:2d}/{n_grid}] param[{param_idx}]={val:+.3f}  "
              f"KF={kl:.1f}  BPF={results['bpf_ll_mean'][i]:.1f}  "
              f"soft={results['soft_ll_mean'][i]:.1f}  "
              f"OT={results['ot_ll_mean'][i]:.1f}")

    return results


def landscape_2d(
    theta_center: np.ndarray,
    param_idx_a: int,
    param_idx_b: int,
    range_a: float,
    range_b: float,
    n_grid: int,
    observations: tf.Tensor,
    Sigma_obs: tf.Tensor,
    make_model_fn: Callable,
    dpf_soft,
    dpf_ot,
    rng_soft: tf.random.Generator,
    rng_ot: tf.random.Generator,
    n_seeds_bpf: int = 5,
    n_seeds_dpf: int = 3,
    n_particles_bpf: int = 500,
) -> dict:
    """
    2D sweep over two parameter dimensions.

    Returns:
        dict with keys:
            'grid_a', 'grid_b':  [n_grid] arrays
            'kalman_ll':         [n_grid, n_grid] exact log-lik
            'bpf_ll':            [n_grid, n_grid] BPF mean
            'soft_ll':           [n_grid, n_grid] DPF-soft mean
            'ot_ll':             [n_grid, n_grid] DPF-OT mean
    """
    grid_a = np.linspace(
        theta_center[param_idx_a] - range_a,
        theta_center[param_idx_a] + range_a,
        n_grid,
    )
    grid_b = np.linspace(
        theta_center[param_idx_b] - range_b,
        theta_center[param_idx_b] + range_b,
        n_grid,
    )

    shape = (n_grid, n_grid)
    out = {
        'grid_a': grid_a, 'grid_b': grid_b,
        'param_idx_a': param_idx_a, 'param_idx_b': param_idx_b,
        'kalman_ll': np.zeros(shape),
        'bpf_ll': np.zeros(shape),
        'soft_ll': np.zeros(shape),
        'ot_ll': np.zeros(shape),
    }

    total = n_grid * n_grid
    count = 0
    for i, va in enumerate(grid_a):
        for j, vb in enumerate(grid_b):
            theta = theta_center.copy()
            theta[param_idx_a] = va
            theta[param_idx_b] = vb

            # Kalman
            out['kalman_ll'][i, j] = kalman_log_lik(
                theta, observations, Sigma_obs, make_model_fn
            )

            # BPF
            bpf_vals = [
                bpf_log_lik(theta, observations, Sigma_obs, make_model_fn,
                            n_particles=n_particles_bpf, seed=s * 1000 + count)
                for s in range(n_seeds_bpf)
            ]
            out['bpf_ll'][i, j] = np.mean(bpf_vals)

            # DPF soft
            soft_vals = [
                dpf_log_lik_and_grad(theta, observations, dpf_soft, rng_soft)[0]
                for _ in range(n_seeds_dpf)
            ]
            out['soft_ll'][i, j] = np.mean(soft_vals)

            # DPF OT
            ot_vals = [
                dpf_log_lik_and_grad(theta, observations, dpf_ot, rng_ot)[0]
                for _ in range(n_seeds_dpf)
            ]
            out['ot_ll'][i, j] = np.mean(ot_vals)

            count += 1
            if count % 10 == 0:
                print(f"  2D sweep: {count}/{total}")

    return out
