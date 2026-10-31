"""Independent M10 reference in log arithmetic; requires consumer qualification.

Uses the same finite observation-root quadrature as nonlinear_quadrature.py.
This diagnostic/reference implementation preserves its positive-sum integral
while avoiding density underflow. Its autodiff score is not a DCTR score.
"""
from functools import lru_cache
import math

import tensorflow as tf

from bayesfilter.score_study.gaussian_tf import parameterized_model
from bayesfilter.testing.surrogate_hmc_nonlinear_quadrature import quadrature_rule
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import (
    source_theta, TRANSITION_CURVE, OBSERVATION_CURVE)

DTYPE = tf.float64


def log_weight(weight):
    # Zero-width quadrature intervals contribute zero mass and zero derivative.
    return tf.where(weight > 0, tf.math.log(tf.where(weight > 0, weight, tf.ones_like(weight))),
                    tf.constant(-math.inf, DTYPE))


@lru_cache(maxsize=8)
def make_log_reference(legendre_order=32, normal_order=128, *, with_score=True):
    if legendre_order not in (12, 20, 32) or normal_order not in (64, 128):
        raise ValueError("unplanned M10 log quadrature configuration")
    nodes, weights = quadrature_rule(legendre_order, normal=False)
    gh, gw = quadrature_rule(normal_order, normal=True)
    normal_offsets = tf.constant([-10., -6., -3., -1., 0., 1., 3., 6., 10.], DTYPE)
    root_offsets = tf.constant([-16., -8., -4., -2., -1., -.5, 0., .5, 1., 2., 4., 8., 16.], DTYPE)
    c, b = tf.cast(TRANSITION_CURVE, DTYPE), tf.cast(OBSERVATION_CURVE, DTYPE)
    normalizer = tf.constant(.5*math.log(2*math.pi), DTYPE)

    def log_density(x, mean, sd):
        return -.5*((x-mean)/sd)**2-normalizer-tf.math.log(sd)

    def compute(scales, observations):
        a, _, h, _, m, _, p, _, q, _, r, _ = parameterized_model(source_theta(scales), 1, 1)
        a, h, m, sp, sq, sr = a[0, 0], h[0, 0], m[0], tf.sqrt(p[0, 0]), tf.sqrt(q[0, 0]), tf.sqrt(r[0, 0])

        def f(x):
            return a*x+c*tf.sin(x)

        def partition(lower, upper, mean, sd, y):
            discriminant = tf.maximum(h*h+4*b*y, tf.constant(0., DTYPE))
            spread = tf.sqrt(discriminant)
            roots = tf.stack([(-h-spread)/(2*b), (-h+spread)/(2*b)])
            width = 2*sr/(tf.sqrt(discriminant+4*b*sr)+spread)
            obs_breaks = tf.reshape(roots[:, None]+width*root_offsets[None, :], [-1])
            breaks = tf.concat([lower[..., None], upper[..., None],
                mean[..., None]+sd*normal_offsets,
                tf.broadcast_to(obs_breaks, tf.concat([tf.shape(mean), [26]], 0)),
                tf.broadcast_to((-h/(2*b))[None], tf.concat([tf.shape(mean), [1]], 0))], -1)
            breaks = tf.sort(tf.minimum(tf.maximum(breaks, lower[..., None]), upper[..., None]), axis=-1)
            half = (breaks[..., 1:]-breaks[..., :-1])/2
            middle = (breaks[..., 1:]+breaks[..., :-1])/2
            shape = tf.concat([tf.shape(mean), [-1]], 0)
            return (tf.reshape(middle[..., None]+half[..., None]*nodes, shape),
                    tf.reshape(half[..., None]*weights, shape))

        outer_sd = tf.sqrt((a+tf.abs(c))**2*sp**2+sq**2)
        x1, w1 = partition(f(m-10*sp)-10*sq, f(m+10*sp)+10*sq, f(m), outer_sd, observations[0, 0])

        def initial_convolution():
            initial = m+sp*gh
            p1 = tf.reduce_logsumexp(log_weight(gw)[None, :]+log_density(x1[:, None], f(initial)[None, :], sq), -1)
            return p1, tf.constant(0., DTYPE)

        def inverse_convolution():
            u = x1[:, None]-sq*gh[None, :]
            inverse = u/a
            for _ in range(16):
                inverse -= (f(inverse)-u)/(a+c*tf.cos(inverse))
            error = tf.reduce_max(tf.abs(f(inverse)-u)/(1+tf.abs(u)))
            p1 = tf.reduce_logsumexp(log_weight(gw)[None, :]+log_density(inverse, m, sp)
                                    -tf.math.log(a+c*tf.cos(inverse)), -1)
            return p1, error

        initial = sq > (a+tf.abs(c))*sp
        log_p1, error = tf.cond(initial, initial_convolution, inverse_convolution)
        mean2 = f(x1)
        x2, w2 = partition(mean2-10*sq, mean2+10*sq, mean2, sq, observations[1, 0])
        log_inner = log_weight(w2)+log_density(x2, mean2[:, None], sq)+log_density(observations[1, 0], h*x2+b*x2*x2, sr)
        log_outer = log_weight(w1)+log_p1+log_density(observations[0, 0], h*x1+b*x1*x1, sr)
        log_joint = log_outer[:, None]+log_inner
        value = tf.reduce_logsumexp(log_joint)
        thresholds = tf.constant([.5, 2.], DTYPE)
        cdf = .5*tf.math.erfc(((h*x2+b*x2*x2)[None, :, :]-thresholds[:, None, None])/
                              (tf.sqrt(tf.constant(2., DTYPE))*sr))
        predictive = tf.reduce_sum(cdf*tf.exp(log_joint-value)[None, :, :], axis=[1, 2])
        return value, predictive, error, initial

    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE), tf.TensorSpec([2, 1], DTYPE)],
                 jit_compile=True, autograph=False)
    def evaluate(scales, observations):
        if with_score:
            with tf.GradientTape() as tape:
                tape.watch(scales)
                value, predictive, residual, initial = compute(scales, observations)
            score = tape.gradient(value, scales)
        else:
            value, predictive, residual, initial = compute(scales, observations)
            score = tf.zeros_like(scales)
        valid = (tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))
                 & tf.reduce_all(tf.math.is_finite(predictive)) & tf.reduce_all(predictive >= -1e-12)
                 & tf.reduce_all(predictive <= 1+1e-12) & (residual <= 1e-11))
        return dict(log_likelihood=value, score=score, predictive=predictive,
                    inverse_relative_residual=residual, initial_convolution=initial, valid=valid)
    return evaluate
