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


def _wrapped_gaussian_evaluate(residual, covariance, d_residual, d_covariance,
                               *, maximum_sigma, aliases):
    """One numerical implementation for shared and row-specific covariances."""
    dtype = residual.dtype
    radial_variance, cross = covariance[..., 0, 0], covariance[..., 1, 0]
    base_valid = (tf.reduce_all(tf.math.is_finite(covariance), axis=[-2, -1])
                  & tf.reduce_all(covariance == tf.linalg.matrix_transpose(covariance), axis=[-2, -1])
                  & (radial_variance > 0.))
    v = tf.where(base_valid, radial_variance, tf.ones_like(radial_variance))
    beta = cross/v
    conditional_variance = covariance[..., 1, 1]-cross*beta
    valid = (base_valid & tf.math.is_finite(conditional_variance)
             & (conditional_variance > 0.)
             & (conditional_variance <= tf.constant(maximum_sigma**2, dtype)))
    s2 = tf.where(valid, conditional_variance, tf.ones_like(conditional_variance))
    r = residual[..., 0]
    angle = residual[..., 1]-beta*r
    pi = tf.constant(math.pi, dtype)
    angle = tf.math.floormod(angle+pi, 2*pi)-pi
    shifts = 2*pi*tf.cast(tf.range(-aliases, aliases+1), dtype)
    u = angle[..., None]+shifts
    log_terms = -.5*(u*u/s2[..., None]+tf.math.log(s2[..., None])+tf.math.log(2*pi))
    value = -.5*(r*r/v+tf.math.log(v)+tf.math.log(2*pi))+tf.reduce_logsumexp(log_terms, axis=-1)
    dv, dc = d_covariance[..., 0, 0], d_covariance[..., 1, 0]
    dbeta = (dc-beta*dv)/v
    ds2 = d_covariance[..., 1, 1]-2*beta*dc+beta*beta*dv
    dr = d_residual[..., 0]
    da = d_residual[..., 1]-dbeta*r-beta*dr
    d_terms = (-u*da[..., None]/s2[..., None]
               +.5*(u*u/s2[..., None]-1)*ds2[..., None]/s2[..., None])
    tangent = (-r*dr/v+.5*(r*r/v-1)*dv/v
               +tf.reduce_sum(tf.nn.softmax(log_terms, axis=-1)*d_terms, axis=-1))
    value_valid = valid & tf.reduce_all(tf.math.is_finite(residual), axis=-1) & tf.math.is_finite(value)
    score_valid = (value_valid & tf.reduce_all(tf.math.is_finite(d_residual), axis=-1)
                   & tf.reduce_all(tf.math.is_finite(d_covariance), axis=[-2, -1])
                   & tf.reduce_all(d_covariance == tf.linalg.matrix_transpose(d_covariance), axis=[-2, -1])
                   & tf.math.is_finite(tangent))
    return dict(value=tf.where(value_valid, value, tf.constant(-math.inf, dtype)),
                tangent=tf.where(score_valid, tangent, tf.constant(math.nan, dtype)),
                value_valid=value_valid, score_valid=score_valid)


def _make_wrapped_gaussian_kernel(maximum_sigma, tail_mass_tolerance, dtype, *, batched_covariance):
    policy = alias_policy(maximum_sigma, tail_mass_tolerance)
    dtype = tf.as_dtype(dtype)
    if dtype not in (tf.float32, tf.float64):
        raise ValueError("wrapped Gaussian kernels require float32 or float64")
    covariance_shape = [None, 2, 2] if batched_covariance else [2, 2]

    @tf.function(input_signature=[tf.TensorSpec([None, 2], dtype),
                                 tf.TensorSpec(covariance_shape, dtype),
                                 tf.TensorSpec([None, 2], dtype),
                                 tf.TensorSpec(covariance_shape, dtype)],
                 jit_compile=True, autograph=False)
    def evaluate(residual, covariance, d_residual, d_covariance):
        return _wrapped_gaussian_evaluate(residual, covariance, d_residual, d_covariance,
            maximum_sigma=maximum_sigma, aliases=policy["aliases_each_side"])
    return evaluate


@lru_cache(maxsize=8)
def wrapped_gaussian_kernel(maximum_sigma=math.pi, tail_mass_tolerance=1e-14, *, dtype=tf.float64):
    """Rows sharing a covariance; preserves the original FP64 default."""
    return _make_wrapped_gaussian_kernel(maximum_sigma, tail_mass_tolerance, dtype, batched_covariance=False)


@lru_cache(maxsize=8)
def wrapped_gaussian_batch_kernel(maximum_sigma=math.pi, tail_mass_tolerance=1e-14, *, dtype=tf.float64):
    """Rows with their own covariances, using the identical numerical core."""
    return _make_wrapped_gaussian_kernel(maximum_sigma, tail_mass_tolerance, dtype, batched_covariance=True)


def wrapped_gaussian_log_density(residual, covariance, *, maximum_sigma=math.pi):
    residual = tf.convert_to_tensor(residual, tf.float64)
    covariance = tf.convert_to_tensor(covariance, tf.float64)
    return wrapped_gaussian_kernel(maximum_sigma)(residual, covariance,
        tf.zeros_like(residual), tf.zeros_like(covariance))["value"]
