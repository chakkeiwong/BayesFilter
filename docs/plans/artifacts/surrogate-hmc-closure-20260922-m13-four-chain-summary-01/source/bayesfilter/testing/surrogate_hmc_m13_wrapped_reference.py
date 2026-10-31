"""Independent FP64 wrapped-normal reference over all positive noise scales.

Diagnostic/reference only. The canonical M13 force is not imported or changed.
"""
import math
import tensorflow as tf

F64 = tf.float64


def angular_terms(delta, sigma):
    """Return angular log density, d(log density)/d(delta), and validity.

    For sigma <= 1 use nine spatial aliases. For sigma > 1 use 32 Fourier
    terms. The local plan derives uniform truncation bounds for both branches.
    """
    delta, sigma = tf.convert_to_tensor(delta, F64), tf.convert_to_tensor(sigma, F64)
    pi = tf.constant(math.pi, F64)
    delta = tf.math.floormod(delta+pi, 2*pi)-pi
    small_sigma = tf.minimum(sigma, tf.constant(1., F64))
    aliases = delta[..., None]+tf.cast(tf.range(-4, 5), F64)*2*pi
    z = aliases/small_sigma[..., None]
    powers = -.5*z*z
    normal_log = tf.reduce_logsumexp(powers, -1)-tf.math.log(small_sigma)-.5*tf.math.log(2*pi)
    normal_score = -tf.reduce_sum(tf.nn.softmax(powers, -1)*z, -1)/small_sigma

    # At sigma >= 40 even the first Fourier coefficient is below exp(-800),
    # below the least positive FP64 subnormal. Its FP64 value and score are zero.
    wide_sigma = tf.minimum(tf.maximum(sigma, tf.constant(1., F64)), tf.constant(40., F64))
    n = tf.cast(tf.range(1, 33), F64)
    coefficients = tf.exp(-.5*tf.square(wide_sigma[..., None]*n))
    argument = delta[..., None]*n
    denominator = 1+2*tf.reduce_sum(coefficients*tf.cos(argument), -1)
    fourier_log = tf.math.log(denominator)-tf.math.log(2*pi)
    fourier_score = -2*tf.reduce_sum(coefficients*n*tf.sin(argument), -1)/denominator
    use_spatial = sigma <= 1.
    value = tf.where(use_spatial, normal_log, fourier_log)
    score = tf.where(use_spatial, normal_score, fourier_score)
    valid = (tf.math.is_finite(delta) & tf.math.is_finite(sigma) & (sigma > 0)
             & tf.math.is_finite(value) & tf.math.is_finite(score))
    return value, score, valid


def observation_log_density(position, observed, sigma):
    """Same diagonal M13 observation law, without the old sigma <= .08 limit."""
    radius = tf.sqrt(tf.reduce_sum(position*position, -1)+tf.constant(1e-12, F64))
    delta = observed[..., 1]-tf.atan2(position[..., 1], position[..., 0])
    angular, _, valid = angular_terms(delta, sigma[1])
    standardized = (observed[..., 0]-radius)/sigma[0]
    radial = -.5*tf.square(standardized)-tf.math.log(sigma[0])-.5*tf.math.log(tf.constant(2*math.pi, F64))
    value = radial+angular
    return tf.where(valid & (sigma[0] > 0) & tf.math.is_finite(sigma[0]),
                    value, tf.constant(float("nan"), F64))
