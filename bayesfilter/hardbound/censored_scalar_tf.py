"""Censored-scalar observation model closed forms (survey eqs. 29--35).

Shadow rate i* ~ N(mu, sigma^2), i = max(ell, i*), y = i + u with
u ~ N(0, V). Predictive density, binding probability, and posterior
binding probability, in log-space-stable float64 TF.
"""

from __future__ import annotations

import tensorflow as tf

DTYPE = tf.float64
_LOG2PI = tf.constant(1.8378770664093453, DTYPE)


def _norm_logpdf(x, mean, var):
    z = (x - mean) / tf.sqrt(var)
    return -0.5 * z * z - 0.5 * tf.math.log(var) - 0.5 * _LOG2PI


def _std_logcdf(z):
    return tfd_normal_logcdf(z)


def tfd_normal_logcdf(z):
    import tensorflow_probability as tfp

    return tfp.distributions.Normal(
        tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE)
    ).log_cdf(tf.convert_to_tensor(z, DTYPE))


def binding_log_probability(mu, sigma, ell):
    """log alpha_b = log Phi((ell - mu)/sigma), survey eq. (30)."""
    mu = tf.convert_to_tensor(mu, DTYPE)
    return _std_logcdf((tf.convert_to_tensor(ell, DTYPE) - mu)
                       / tf.convert_to_tensor(sigma, DTYPE))


def predictive_log_density(y, mu, sigma, ell, obs_var):
    """log p(y) of the two-term mixture, survey eq. (34), stable form.

    p(y) = alpha_b N(y; ell, V)
         + N(y; mu, V + sigma^2) * (1 - Phi((ell - mu_t~)/sigma~)).
    """
    y = tf.convert_to_tensor(y, DTYPE)
    mu = tf.convert_to_tensor(mu, DTYPE)
    sigma = tf.convert_to_tensor(sigma, DTYPE)
    ell = tf.convert_to_tensor(ell, DTYPE)
    v = tf.convert_to_tensor(obs_var, DTYPE)
    s2 = sigma * sigma
    log_a = binding_log_probability(mu, sigma, ell) + _norm_logpdf(y, ell, v)
    # complete-the-square quantities, eq. (33)
    tilde_var = s2 * v / (s2 + v)
    tilde_mu = (v * mu + s2 * y) / (s2 + v)
    # survival Phi(-z) in log space for the upper tail mass
    upper_log = _std_logcdf(-(ell - tilde_mu) / tf.sqrt(tilde_var))
    log_b = _norm_logpdf(y, mu, v + s2) + upper_log
    return tf.reduce_logsumexp(tf.stack([log_a, log_b], axis=0), axis=0)


def posterior_binding_log_probability(y, mu, sigma, ell, obs_var):
    """log Pr(binding | y), survey eq. (35)."""
    y = tf.convert_to_tensor(y, DTYPE)
    v = tf.convert_to_tensor(obs_var, DTYPE)
    log_a = binding_log_probability(mu, sigma, ell) + _norm_logpdf(
        y, tf.convert_to_tensor(ell, DTYPE), v)
    return log_a - predictive_log_density(y, mu, sigma, ell, obs_var)
