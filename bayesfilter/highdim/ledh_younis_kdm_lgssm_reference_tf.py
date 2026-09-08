"""Independent scalar LGSSM value/score reference for Phase 4A diagnostics.

The reference is deliberately separate from the LEDH executor.  It computes
the exact marginal likelihood of a scalar stationary AR(1) state-space model
with fixed process and observation variances and differentiates the Kalman
recursion analytically with respect to the transition coefficient.
"""

from __future__ import annotations

import math

import tensorflow as tf


def scalar_lgssm_value_and_score(
    theta: tf.Tensor,
    observations: tf.Tensor,
    *,
    initial_variance: float = 1.0,
    process_variance: float = 0.1,
    observation_variance: float = 0.2,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return exact ``log p_theta(y)`` and its score for scalar AR(1).

    The model is

    ``x_t = theta[0] x_{t-1} + Normal(0, process_variance)`` and
    ``y_t = x_t + Normal(0, observation_variance)``, with
    ``x_0 ~ Normal(0, initial_variance)``.

    The derivative is propagated through prediction, innovation, and Kalman
    covariance update.  No autodiff or LEDH implementation is used here.
    """

    theta = tf.convert_to_tensor(theta)
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    if theta.shape.rank != 1 or theta.shape[0] != 1:
        raise ValueError("theta must have shape [1]")
    if observations.shape.rank != 1 or observations.shape[0] is None:
        raise ValueError("observations must have a statically known shape [T]")
    dtype = theta.dtype
    if not dtype.is_floating:
        raise ValueError("theta must use a floating TensorFlow dtype")
    if initial_variance <= 0.0:
        raise ValueError("initial_variance must be positive")
    if process_variance <= 0.0 or observation_variance <= 0.0:
        raise ValueError("noise variances must be positive")

    phi = theta[0]
    process = tf.constant(process_variance, dtype)
    observation_noise = tf.constant(observation_variance, dtype)
    mean = tf.zeros([], dtype)
    variance = tf.constant(initial_variance, dtype)
    d_mean = tf.zeros([], dtype)
    d_variance = tf.zeros([], dtype)
    value = tf.zeros([], dtype)
    score = tf.zeros([], dtype)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)

    for observation in tf.unstack(observations, axis=0):
        predicted_mean = phi * mean
        predicted_variance = phi * phi * variance + process
        d_predicted_mean = mean + phi * d_mean
        d_predicted_variance = (
            2.0 * phi * variance + phi * phi * d_variance
        )

        innovation = observation - predicted_mean
        d_innovation = -d_predicted_mean
        innovation_variance = predicted_variance + observation_noise
        d_innovation_variance = d_predicted_variance

        value = value - 0.5 * (
            log_two_pi
            + tf.math.log(innovation_variance)
            + innovation * innovation / innovation_variance
        )
        score = score - 0.5 * d_innovation_variance / innovation_variance
        score = score - innovation * d_innovation / innovation_variance
        score = score + 0.5 * (
            innovation
            * innovation
            * d_innovation_variance
            / (innovation_variance * innovation_variance)
        )

        gain = predicted_variance / innovation_variance
        d_gain = (
            d_predicted_variance * innovation_variance
            - predicted_variance * d_innovation_variance
        ) / (innovation_variance * innovation_variance)
        mean = predicted_mean + gain * innovation
        d_mean = (
            d_predicted_mean + d_gain * innovation + gain * d_innovation
        )
        variance = predicted_variance - gain * predicted_variance
        d_variance = (
            d_predicted_variance
            - d_gain * predicted_variance
            - gain * d_predicted_variance
        )

    return value, score


__all__ = ["scalar_lgssm_value_and_score"]
