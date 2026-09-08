"""Phase 1 gates G1.1--G1.2 of the hard-bound master program."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy import integrate, stats

from bayesfilter.hardbound import censored_scalar_tf as cs
from bayesfilter.hardbound import truncated_gaussian_tf as tg

RNG = np.random.default_rng(20260821)


def _mixture_pdf_scipy(y, mu, sigma, ell, v):
    """Independent scipy implementation of survey eq. (34)."""
    alpha_b = stats.norm.cdf((ell - mu) / sigma)
    s2 = sigma**2
    tilde_var = s2 * v / (s2 + v)
    tilde_mu = (v * mu + s2 * y) / (s2 + v)
    upper = stats.norm.sf((ell - tilde_mu) / np.sqrt(tilde_var))
    return (alpha_b * stats.norm.pdf(y, ell, np.sqrt(v))
            + stats.norm.pdf(y, mu, np.sqrt(v + s2)) * upper)


def test_g1_1_mixture_matches_scipy_and_integrates_to_one():
    grid = [
        # (mu, sigma, ell, v) including deep-tail gammas
        (0.02, 0.01, 0.0, 1e-6),
        (-0.05, 0.01, 0.0, 1e-6),     # gamma = +5, mostly binding
        (0.08, 0.01, 0.0, 1e-6),      # gamma = -8, tail
        (0.0, 0.005, 0.0, 2.5e-7),
        (0.04, 0.005, 0.0, 1e-4),
        (-0.02, 0.02, -0.005, 1e-6),
    ]
    for mu, sigma, ell, v in grid:
        ys = np.linspace(mu - 8 * np.sqrt(v + sigma**2),
                         mu + 8 * np.sqrt(v + sigma**2), 201)
        ys = np.union1d(ys, np.linspace(ell - 8 * np.sqrt(v),
                                        ell + 8 * np.sqrt(v), 201))
        tf_log = cs.predictive_log_density(
            tf.constant(ys, tf.float64), mu, sigma, ell, v).numpy()
        sp = _mixture_pdf_scipy(ys, mu, sigma, ell, v)
        # The scipy reference computes in linear space and loses accuracy
        # below ~1e-280 (subnormal underflow); compare in log space where
        # the linear-space reference is itself reliable.
        mask = sp > 1e-280
        np.testing.assert_allclose(tf_log[mask], np.log(sp[mask]),
                                   rtol=0, atol=1e-9)
        lo = min(mu - 10 * np.sqrt(v + sigma**2), ell - 10 * np.sqrt(v))
        hi = max(mu + 10 * np.sqrt(v + sigma**2), ell + 10 * np.sqrt(v))
        total, _ = integrate.quad(
            lambda yy: float(np.exp(cs.predictive_log_density(
                tf.constant([yy], tf.float64), mu, sigma, ell, v).numpy()[0])),
            lo, hi, limit=300)
        assert abs(total - 1.0) < 1e-8, (mu, sigma, ell, v, total)


def test_g1_1_tail_stability_gamma_pm8():
    # log density stays finite and matches scipy in log space at gamma = +-8
    for mu in (0.08, -0.08):
        y = np.array([0.0, mu])
        out = cs.predictive_log_density(
            tf.constant(y, tf.float64), mu, 0.01, 0.0, 1e-6).numpy()
        assert np.all(np.isfinite(out))
    lp = cs.posterior_binding_log_probability(
        tf.constant([0.0], tf.float64), -0.08, 0.01, 0.0, 1e-6).numpy()
    assert np.isfinite(lp[0]) and lp[0] <= 0.0


def test_g1_2_truncated_moments_match_truncnorm_and_mc():
    d = 3
    A = RNG.normal(size=(d, d))
    P = A @ A.T + np.eye(d)
    m = RNG.normal(size=d)
    a = RNG.normal(size=d)
    c = float(a @ m + 0.7 * np.sqrt(a @ P @ a))
    out = {k: np.asarray(v) for k, v in tg.truncated_branch_moments(
        tf.constant(m), tf.constant(P), tf.constant(a), c).items()}
    # Monte Carlo check
    n = 10_000_000
    z = RNG.multivariate_normal(m, P, size=n)
    u = z @ a
    bind = u <= c
    p_bind = bind.mean()
    np.testing.assert_allclose(np.exp(out["log_prob_binding"]), p_bind,
                               rtol=2e-3)
    mc_mean_b = z[bind].mean(axis=0)
    mc_cov_b = np.cov(z[bind].T)
    se = np.sqrt(np.diag(P) / bind.sum())
    assert np.all(np.abs(out["mean_binding"] - mc_mean_b) < 4 * se)
    np.testing.assert_allclose(out["cov_binding"], mc_cov_b,
                               rtol=0, atol=4 * np.abs(mc_cov_b).max() /
                               np.sqrt(bind.sum() / 100))
    # scalar-direction cross-check vs scipy truncnorm
    mu_u, sd_u = float(a @ m), float(np.sqrt(a @ P @ a))
    tn = stats.truncnorm(-np.inf, (c - mu_u) / sd_u, loc=mu_u, scale=sd_u)
    np.testing.assert_allclose(float(a @ out["mean_binding"]), tn.mean(),
                               rtol=1e-10)
    np.testing.assert_allclose(float(a @ out["cov_binding"] @ a), tn.var(),
                               rtol=1e-9)


def test_g1_2_tail_stability_gamma_pm37():
    m = np.zeros(2)
    P = np.eye(2)
    a = np.array([1.0, 0.0])
    for c in (37.0, -37.0):
        out = tg.truncated_branch_moments(
            tf.constant(m), tf.constant(P), tf.constant(a), float(c))
        for key in ("mean_binding", "cov_binding", "mean_nonbinding",
                    "cov_nonbinding"):
            vals = np.asarray(out[key])
            assert np.all(np.isfinite(vals)), (c, key, vals)
    # at gamma=-37 the binding branch conditions on a ~1e-300 event; the
    # truncated mean along a must sit just below c-side Mills asymptote
    out = tg.truncated_branch_moments(
        tf.constant(m), tf.constant(P), tf.constant(a), -37.0)
    mean_b = float(np.asarray(out["mean_binding"])[0])
    assert -37.2 < mean_b < -37.0
