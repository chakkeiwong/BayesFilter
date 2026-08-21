"""Exact grid posterior for the K=1 gate model (diagnostic NumPy module).

Computes the marginal parameter posterior p(mu, log_sd | y) on a grid by
15-node/dim Gauss-Hermite integration over the 5 standard-normal latents,
and the Kalman likelihood of the bound-removed (affine) variant for gate
G2.0. Diagnostic reference only; never a runtime path.
"""

from __future__ import annotations

import itertools

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))


def _gh_grid(dim: int, nodes: int):
    x, w = np.polynomial.hermite_e.hermegauss(nodes)  # weight e^{-x^2/2}
    logw = np.log(w) - 0.5 * LOG2PI + 0.5 * x * x  # so sum w_i f(x_i) ~ E f
    # note: hermegauss weights integrate against e^{-x^2/2}; dividing by
    # sqrt(2 pi) makes them a discrete standard normal: sum w_norm = 1.
    w_norm = w / np.sqrt(2.0 * np.pi)
    grids = np.array(list(itertools.product(x, repeat=dim)))
    logws = np.log(np.array(list(itertools.product(w_norm, repeat=dim)))).sum(1)
    return grids, logws


def _gate_states(mu, raws, gm):
    x = mu + gm.p0_sd * raws[..., 0]
    out = []
    for t in range(gm.horizon):
        x = mu + gm.phi * (x - mu) + gm.q_sd * raws[..., 1 + t]
        out.append(x)
    return np.stack(out, axis=-1)  # [..., T]


def gate_loglik(y, mu, log_sd, gm, hard_bound=True, gh_nodes=15):
    """log p(y | mu, log_sd) by GH integration over the 5 latents."""
    raws, logws = _gh_grid(1 + gm.horizon, gh_nodes)
    states = _gate_states(mu, raws, gm)  # [G, T]
    mean = np.maximum(gm.lower_bound, states) if hard_bound else states
    sd = np.exp(log_sd)
    z = (np.asarray(y)[None, :] - mean) / sd
    obs_ll = np.sum(-0.5 * z * z - log_sd - 0.5 * LOG2PI, axis=1)  # [G]
    a = logws + obs_ll
    amax = a.max()
    return amax + np.log(np.exp(a - amax).sum())


def gate_grid_posterior(y, gm, mu_grid, logsd_grid, hard_bound=True,
                        n_grid=2000):
    """Normalized posterior on the parameter grid via the dense grid filter."""
    logpost = np.empty((len(mu_grid), len(logsd_grid)))
    for i, mu in enumerate(mu_grid):
        for j, ls in enumerate(logsd_grid):
            lp = gate_gridfilter_loglik(y, mu, ls, gm, hard_bound, n_grid)
            lp += (-0.5 * ((mu - gm.prior_mu_mean) / gm.prior_mu_sd) ** 2
                   - np.log(gm.prior_mu_sd) - 0.5 * LOG2PI)
            lp += (-0.5 * ((ls - gm.prior_lognoise_mean)
                           / gm.prior_lognoise_sd) ** 2
                   - np.log(gm.prior_lognoise_sd) - 0.5 * LOG2PI)
            logpost[i, j] = lp
    logpost -= logpost.max()
    post = np.exp(logpost)
    post /= post.sum()
    return post


def gate_kalman_loglik(y, mu, log_sd, gm):
    """Exact Kalman likelihood of the bound-removed (affine) gate model."""
    sd2 = np.exp(2.0 * log_sd)
    m, p = mu, gm.p0_sd**2
    ll = 0.0
    for t in range(gm.horizon):
        m = mu + gm.phi * (m - mu)
        p = gm.phi**2 * p + gm.q_sd**2
        s = p + sd2
        v = y[t] - m
        ll += -0.5 * (LOG2PI + np.log(s) + v * v / s)
        k = p / s
        m = m + k * v
        p = p - k * p
    return ll


def gate_gridfilter_loglik(y, mu, log_sd, gm, hard_bound=True, n_grid=2000,
                           span_sd=8.0):
    """log p(y | mu, log_sd) by a dense scalar grid filter (exact reference).

    Classic deterministic grid filter for the 1-D gate model: discretize
    the state, propagate through the AR(1) Gaussian kernel, weight by the
    (censored) observation density. This is the survey ladder's dense-grid
    reference, robust to the narrow observation-likelihood ridge that
    defeats latent-space Gauss--Hermite quadrature.
    """
    stat_sd = gm.q_sd / np.sqrt(1.0 - gm.phi**2)
    span = span_sd * max(gm.p0_sd, stat_sd)
    x = np.linspace(mu - span, mu + span, n_grid)
    dx = x[1] - x[0]
    sd = np.exp(log_sd)

    def norm_pdf(v, s):
        return np.exp(-0.5 * (v / s) ** 2) / (s * np.sqrt(2 * np.pi))

    # transition kernel matrix K[i, j] = p(x_i | x_j) * dx
    pred_mean = mu + gm.phi * (x - mu)
    K = norm_pdf(x[:, None] - pred_mean[None, :], gm.q_sd) * dx
    dens = norm_pdf(x - mu, gm.p0_sd)  # p(x0)
    ll = 0.0
    for t in range(gm.horizon):
        dens = K @ dens  # predictive density on grid
        mean_obs = np.maximum(gm.lower_bound, x) if hard_bound else x
        lik = norm_pdf(y[t] - mean_obs, sd)
        joint = dens * lik
        step = joint.sum() * dx
        ll += np.log(step)
        dens = joint / (joint.sum() * dx)
    return ll
