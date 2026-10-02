"""Tensor-batched analytical second-order direct-QR Kalman recurrence.

A jet is (value, parameter tangents, parameter-pair tangents). These are
explicit product/solve rules, not autodiff of the finite filter program.
Factor rules: ch12_factor_derivatives, QR first/second and Cholesky equations.
"""

from __future__ import annotations

import math

import tensorflow as tf

from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor_derivatives,
    factor_covariance_derivatives,
    factor_solve,
    stack_qr_lower_factor_derivatives,
)


def _add(a, b):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _negative(a):
    return -a[0], -a[1], -a[2]


def _transpose(a):
    return (
        tf.linalg.matrix_transpose(a[0]),
        tf.linalg.matrix_transpose(a[1]),
        tf.linalg.matrix_transpose(a[2]),
    )


def _column(a):
    return a[0][..., None], a[1][..., None], a[2][..., None]


def _matmul(a, b):
    av, ad, ah = a
    bv, bd, bh = b
    return (
        av @ bv,
        ad @ bv + av @ bd,
        ah @ bv + ad[:, None] @ bd[None, :] + ad[None, :] @ bd[:, None] + av @ bh,
    )


def _concat(a, b):
    return (
        tf.concat((a[0], b[0]), -1),
        tf.concat((a[1], b[1]), -1),
        tf.concat((a[2], b[2]), -1),
    )


def _scale(a, scale):
    return a[0] * scale, a[1] * scale, a[2] * scale


def _constant(value, parameters):
    return (
        value,
        tf.zeros(tf.concat(([parameters], tf.shape(value)), 0), value.dtype),
        tf.zeros(
            tf.concat(([parameters, parameters], tf.shape(value)), 0), value.dtype
        ),
    )


def _qr(stack):
    value, first, second, _ = stack_qr_lower_factor_derivatives(*stack)
    return value, first, second


def qr_kalman_second_order(
    observations,
    mask,
    initial_mean,
    initial_covariance,
    transition_offset,
    transition_matrix,
    process_covariance,
    observation_offset,
    observation_matrix,
    observation_covariance,
    jitter,
):
    """Run sequential dates; batch all first and second parameter directions.

    Inputs other than observations/mask/jitter are jets with static parameter
    and spatial dimensions. Missing observations use the same dummy-identity
    likelihood and Joseph update as the public masked direct-QR API.
    """
    dtype = observations.dtype
    parameters = initial_mean[1].shape[0]
    state_dim = initial_mean[0].shape[0]
    obs_dim = observation_matrix[0].shape[0]
    identity = _constant(tf.eye(state_dim, dtype=dtype), parameters)
    obs_identity = tf.eye(obs_dim, dtype=dtype)
    mean = _column(initial_mean)
    factor = cholesky_factor_derivatives(*initial_covariance)
    process_factor = cholesky_factor_derivatives(*process_covariance)
    observation_noise = _add(
        observation_covariance, _constant(jitter * obs_identity, parameters)
    )
    c = _column(transition_offset)
    d = _column(observation_offset)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)
    value = tf.zeros([], dtype)
    score = tf.zeros([parameters], dtype)
    hessian = tf.zeros([parameters, parameters], dtype)

    def step(t, mean, factor, value, score, hessian):
        predicted_mean = _add(c, _matmul(transition_matrix, mean))
        predicted_factor = _qr(
            _concat(_matmul(transition_matrix, factor), process_factor)
        )
        predicted_covariance = factor_covariance_derivatives(*predicted_factor)
        weight = tf.cast(mask[t], dtype)
        outer = weight[:, None] * weight[None, :]
        z = _scale(observation_matrix, weight[:, None])
        noise = _add(
            _scale(observation_noise, outer),
            _constant(tf.linalg.diag(1.0 - weight), parameters),
        )
        noise_factor = cholesky_factor_derivatives(*noise)
        innovation = _scale(
            _add(
                _constant(observations[t, :, None], parameters),
                _negative(_add(d, _matmul(observation_matrix, predicted_mean))),
            ),
            weight[:, None],
        )
        innovation_factor = _qr(_concat(_matmul(z, predicted_factor), noise_factor))
        _, ds, hs = factor_covariance_derivatives(*innovation_factor)
        precision = factor_solve(innovation_factor[0], obs_identity)
        dp = -precision @ ds @ precision
        hp = (
            precision
            @ (
                ds[:, None] @ precision @ ds[None, :]
                + ds[None, :] @ precision @ ds[:, None]
                - hs
            )
            @ precision
        )
        precision_jet = precision, dp, hp
        quadratic = _matmul(_transpose(innovation), _matmul(precision_jet, innovation))
        logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor[0]))
        )
        dlogdet = tf.linalg.trace(precision @ ds)
        hlogdet = tf.linalg.trace(precision @ hs + dp[None, :] @ ds[:, None])
        value -= 0.5 * (
            tf.reduce_sum(weight) * log_two_pi + logdet + quadratic[0][0, 0]
        )
        score -= 0.5 * (dlogdet + quadratic[1][..., 0, 0])
        hessian -= 0.5 * (hlogdet + quadratic[2][..., 0, 0])
        gain = _matmul(_matmul(predicted_covariance, _transpose(z)), precision_jet)
        mean = _add(predicted_mean, _matmul(gain, innovation))
        joseph = _add(identity, _negative(_matmul(gain, z)))
        factor = _qr(
            _concat(_matmul(joseph, predicted_factor), _matmul(gain, noise_factor))
        )
        return t + 1, mean, factor, value, score, hessian

    _, _, _, value, score, hessian = tf.while_loop(
        lambda t, *_: t < tf.shape(observations)[0],
        step,
        (tf.constant(0), mean, factor, value, score, hessian),
        parallel_iterations=1,
    )
    return value, score, 0.5 * (hessian + tf.transpose(hessian))
