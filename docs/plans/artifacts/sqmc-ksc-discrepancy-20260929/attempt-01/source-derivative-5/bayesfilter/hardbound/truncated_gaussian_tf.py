"""Affine-boundary truncated-Gaussian branch quantities (survey eqs. 21--28).

For z ~ N(m, P) and boundary a^T z = c: branch log probabilities and the
exact truncated means/covariances of both branches, evaluated with
log-cdf/log-survival Mills ratios so the tails (|gamma| up to ~37 in
float64) remain finite and accurate.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float64
_STD = tfp.distributions.Normal(tf.constant(0.0, DTYPE),
                                tf.constant(1.0, DTYPE))


def _log_mills_lower(gamma):
    """log lambda_b = log phi(gamma) - log Phi(gamma)."""
    return _STD.log_prob(gamma) - _STD.log_cdf(gamma)


def _log_mills_upper(gamma):
    """log lambda_n = log phi(gamma) - log (1 - Phi(gamma))."""
    return _STD.log_prob(gamma) - _STD.log_survival_function(gamma)


def truncated_branch_moments(m, P, a, c):
    """Branch probabilities and truncated moments, eqs. (22), (27), (28).

    m: [d], P: [d, d], a: [d], c: scalar. Returns a dict with
    log_prob_binding, log_prob_nonbinding, mean/cov for each branch.
    """
    m = tf.convert_to_tensor(m, DTYPE)
    P = tf.convert_to_tensor(P, DTYPE)
    a = tf.convert_to_tensor(a, DTYPE)
    c = tf.convert_to_tensor(c, DTYPE)
    Pa = tf.linalg.matvec(P, a)
    var_b = tf.tensordot(a, Pa, 1)
    sd_b = tf.sqrt(var_b)
    gamma = (c - tf.tensordot(a, m, 1)) / sd_b
    lam_b = tf.exp(_log_mills_lower(gamma))
    lam_n = tf.exp(_log_mills_upper(gamma))
    outer = Pa[:, None] * Pa[None, :] / var_b
    mean_b = m - Pa / sd_b * lam_b
    cov_b = P - outer * lam_b * (lam_b + gamma)
    mean_n = m + Pa / sd_b * lam_n
    cov_n = P - outer * lam_n * (lam_n - gamma)
    return {
        "gamma": gamma,
        "log_prob_binding": _STD.log_cdf(gamma),
        "log_prob_nonbinding": _STD.log_survival_function(gamma),
        "mean_binding": mean_b,
        "cov_binding": cov_b,
        "mean_nonbinding": mean_n,
        "cov_nonbinding": cov_n,
    }
