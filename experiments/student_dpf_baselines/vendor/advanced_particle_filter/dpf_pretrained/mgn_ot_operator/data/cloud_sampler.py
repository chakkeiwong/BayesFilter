"""
Source B training cloud generator.

Samples weighted empirical measures (particle clouds) from a broad distribution
covering the regimes that a particle filter encounters during operation:

  - Mixture of Gaussians in R^d  (1, 2, or 3 components).
  - Random covariance matrices via Cholesky factors.
  - Dirichlet-distributed weights with concentration parameter varied over
    ~2 orders of magnitude -- covers uniform weights (high ESS) through
    highly peaked weights (low ESS / near-degeneracy).

Everything here is numpy-based: cloud generation runs on CPU, the clouds
are then stacked into TF tensors for downstream precompute / training.
Rationale: numpy is faster for small per-cloud sampling, and we precompute
the dataset offline anyway.
"""

import numpy as np


def sample_mixture_params(d, n_components, rng, mean_range=3.0,
                          cov_scale=0.5, cov_floor=0.1):
    """Sample a random Gaussian mixture in R^d."""
    means = rng.uniform(-mean_range, mean_range, size=(n_components, d))
    covs = np.empty((n_components, d, d))
    for c in range(n_components):
        L = rng.normal(0, cov_scale, size=(d, d))
        covs[c] = L @ L.T + cov_floor * np.eye(d)
    mix_w = rng.dirichlet(np.ones(n_components))
    return means, covs, mix_w


def sample_positions_from_mixture(means, covs, mix_w, N, rng):
    """Draw N iid samples from a Gaussian mixture."""
    comp_choices = rng.choice(len(mix_w), size=N, p=mix_w)
    d = means.shape[1]
    x = np.empty((N, d))
    for c in range(len(mix_w)):
        mask = comp_choices == c
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        x[mask] = rng.multivariate_normal(means[c], covs[c], size=n_c)
    return x


def sample_cloud(N, d, rng,
                 unimodal_prob=0.5,
                 dirichlet_alpha_log10_range=(-1.0, 1.0)):
    """Sample one weighted particle cloud.

    Args:
        N: number of particles.
        d: state dimension.
        rng: numpy Generator.
        unimodal_prob: probability of sampling a unimodal (1-component) cloud.
        dirichlet_alpha_log10_range: range of log10(alpha) for the Dirichlet
            concentration parameter that generates the weights.

    Returns:
        x: (N, d) float32 particle positions.
        w: (N,)  float32 particle weights, sum to 1.
    """
    # Decide mixture structure.
    if rng.random() < unimodal_prob:
        n_comp = 1
    else:
        n_comp = int(rng.choice([2, 3], p=[0.6, 0.4]))

    means, covs, mix_w = sample_mixture_params(d, n_comp, rng)
    x = sample_positions_from_mixture(means, covs, mix_w, N, rng)

    # Weights: Dirichlet with varying concentration.
    log10_alpha = rng.uniform(*dirichlet_alpha_log10_range)
    alpha = 10.0 ** log10_alpha
    w = rng.dirichlet(alpha * np.ones(N))

    return x.astype(np.float32), w.astype(np.float32)


def sample_clouds_batch(n_clouds, N, d, rng, **kwargs):
    """Sample n_clouds clouds, all with the same N and d. Returns stacked arrays."""
    xs = np.empty((n_clouds, N, d), dtype=np.float32)
    ws = np.empty((n_clouds, N), dtype=np.float32)
    for i in range(n_clouds):
        xs[i], ws[i] = sample_cloud(N, d, rng, **kwargs)
    return xs, ws


# ---------------------------------------------------------------------------
# Diagnostics: effective sample size for sanity-checking weight distribution
# ---------------------------------------------------------------------------
def effective_sample_size(w):
    """ESS = 1 / sum w_i^2. Returns scalar or per-row."""
    w = np.asarray(w)
    if w.ndim == 1:
        return 1.0 / np.sum(w ** 2)
    return 1.0 / np.sum(w ** 2, axis=-1)
