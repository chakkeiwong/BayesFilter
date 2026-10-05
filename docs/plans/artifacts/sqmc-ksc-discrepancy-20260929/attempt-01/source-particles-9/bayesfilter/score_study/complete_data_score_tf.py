"""Analytical physical-model log-density partials with latent states fixed.

These are the additive terms in Fisher's identity, not derivatives of a
particle simulator. The initial law is part of the six-parameter model.
"""
import tensorflow as tf

from .conditional_means_tf import conditional_mean_and_tangent
from .gaussian_tf import gaussian_log_density_and_tangent


def initial_score(x, model):
    _, _, _, _, mean, dmean, covariance, dcovariance, _, _, _, _ = model
    return gaussian_log_density_and_tangent(
        x - mean, -tf.broadcast_to(dmean[:, None, :], [6, *x.shape]),
        covariance, dcovariance[:, None, :, :])[1]


def increment_score(previous, x, observation, model, transition_curve=0., observation_curve=0.):
    A, dA, H, dH, _, _, _, _, Q, dQ, R, dR = model
    zero = tf.zeros([6, *x.shape], x.dtype)
    mean, dmean = conditional_mean_and_tangent(previous, zero, A, dA, transition_curve)
    observed, dobserved = conditional_mean_and_tangent(
        x, zero, H, dH, observation_curve, quadratic=True)
    transition = gaussian_log_density_and_tangent(x - mean, -dmean, Q, dQ[:, None, :, :])[1]
    likelihood = gaussian_log_density_and_tangent(
        observation - observed, -dobserved, R, dR[:, None, :, :])[1]
    return transition + likelihood
