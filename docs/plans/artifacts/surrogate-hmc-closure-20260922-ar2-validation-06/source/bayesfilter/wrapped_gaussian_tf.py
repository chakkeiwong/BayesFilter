"""Wrapped Gaussian observation law with controlled alias truncation.

The radial coordinate is real and the angular coordinate is modulo 2*pi.
Correlated Gaussian noise is handled by conditioning the angle on the radius.
Only the conditional angular Gaussian is wrapped. The returned directional
derivative differentiates both residuals and the complete covariance.
The finite sum omits at most the probability mass declared by alias_policy;
it is not an exactly normalized finite replacement for the infinite sum.
"""
from functools import lru_cache
import math

import tensorflow as tf


def alias_policy(maximum_sigma=math.pi, tail_mass_tolerance=1e-14):
    """Bound omitted angular probability, uniformly over sigma <= maximum_sigma."""
    if not (math.isfinite(maximum_sigma) and maximum_sigma > 0
            and math.isfinite(tail_mass_tolerance) and 0 < tail_mass_tolerance < 1):
        raise ValueError("finite positive scale and probability tolerance required")
    aliases = 0
    while math.erfc((2*aliases+1)*math.pi/(math.sqrt(2)*maximum_sigma)) > tail_mass_tolerance:
        aliases += 1
        if aliases > 4096:
            raise ValueError("declared alias budget exceeded")
    return dict(maximum_conditional_sigma=maximum_sigma, aliases_each_side=aliases,
                omitted_mass_upper_bound=math.erfc((2*aliases+1)*math.pi/(math.sqrt(2)*maximum_sigma)),
                tail_mass_tolerance=tail_mass_tolerance)


@lru_cache(maxsize=8)
def wrapped_gaussian_kernel(maximum_sigma=math.pi, tail_mass_tolerance=1e-14):
    policy = alias_policy(maximum_sigma, tail_mass_tolerance)
    aliases = policy["aliases_each_side"]

    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float64),
                                 tf.TensorSpec([2, 2], tf.float64),
                                 tf.TensorSpec([None, 2], tf.float64),
                                 tf.TensorSpec([2, 2], tf.float64)],
                 jit_compile=True, autograph=False)
    def evaluate(residual, covariance, d_residual, d_covariance):
        radial_variance, cross = covariance[0, 0], covariance[1, 0]
        # Invalid inputs are flagged, never ridged into a different accepted law.
        base_valid = (tf.reduce_all(tf.math.is_finite(covariance))
                      & tf.reduce_all(covariance == tf.transpose(covariance))
                      & (radial_variance > 0.))
        v = tf.where(base_valid, radial_variance, tf.constant(1., tf.float64))
        beta = cross/v
        conditional_variance = covariance[1, 1] - cross*beta
        valid = (base_valid & tf.math.is_finite(conditional_variance)
                 & (conditional_variance > 0.) & (conditional_variance <= maximum_sigma**2))
        s2 = tf.where(valid, conditional_variance, tf.constant(1., tf.float64))
        r = residual[:, 0]
        angle = residual[:, 1] - beta*r
        pi = tf.constant(math.pi, tf.float64)
        angle = tf.math.floormod(angle+pi, 2*pi)-pi
        shifts = 2*pi*tf.cast(tf.range(-aliases, aliases+1), tf.float64)
        u = angle[:, None]+shifts[None, :]
        log_terms = -.5*(u*u/s2+tf.math.log(s2)+tf.math.log(2*pi))
        value = -.5*(r*r/v+tf.math.log(v)+tf.math.log(2*pi))+tf.reduce_logsumexp(log_terms, axis=1)
        dv, dc = d_covariance[0, 0], d_covariance[1, 0]
        dbeta = (dc-beta*dv)/v
        ds2 = d_covariance[1, 1]-2*beta*dc+beta*beta*dv
        dr = d_residual[:, 0]
        da = d_residual[:, 1]-dbeta*r-beta*dr
        d_terms = -u*da[:, None]/s2+.5*(u*u/s2-1)*ds2/s2
        tangent = -r*dr/v+.5*(r*r/v-1)*dv/v+tf.reduce_sum(tf.nn.softmax(log_terms, axis=1)*d_terms, axis=1)
        value_valid = valid & tf.reduce_all(tf.math.is_finite(residual), axis=1) & tf.math.is_finite(value)
        score_valid = (value_valid & tf.reduce_all(tf.math.is_finite(d_residual), axis=1)
                       & tf.reduce_all(tf.math.is_finite(d_covariance))
                       & tf.reduce_all(d_covariance == tf.transpose(d_covariance)) & tf.math.is_finite(tangent))
        return dict(value=tf.where(value_valid, value, tf.constant(-math.inf, tf.float64)),
                    tangent=tf.where(score_valid, tangent, tf.constant(math.nan, tf.float64)),
                    value_valid=value_valid, score_valid=score_valid)
    return evaluate


def wrapped_gaussian_log_density(residual, covariance, *, maximum_sigma=math.pi):
    residual = tf.convert_to_tensor(residual, tf.float64)
    covariance = tf.convert_to_tensor(covariance, tf.float64)
    return wrapped_gaussian_kernel(maximum_sigma)(residual, covariance,
        tf.zeros_like(residual), tf.zeros_like(covariance))["value"]
