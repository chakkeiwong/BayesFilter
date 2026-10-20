"""TensorFlow UKF reference for the experimental range-bearing fixture."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.range_bearing_tf import (
    DTYPE,
    RangeBearingTFFixture,
    observation_residual_tf,
    range_bearing_observation_tf,
)


@dataclass(frozen=True)
class UKFTFResult:
    filtered_means: tf.Tensor
    filtered_covariances: tf.Tensor
    predicted_observations: tf.Tensor
    finite: bool
    approximate_reference: bool = True
    reference_id: str = "ukf_range_bearing_tf_approximate_reference"


def run_range_bearing_ukf_tf(
    fixture: RangeBearingTFFixture,
    *,
    alpha: float = 0.5,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> UKFTFResult:
    n = fixture.state_dim
    lam = alpha * alpha * (n + kappa) - n
    wm, wc = _unscented_weights(n, lam, alpha, beta)
    mean = tf.identity(fixture.m0)
    covariance = tf.identity(fixture.P0)
    filtered_means = []
    filtered_covariances = []
    predicted_observations = []

    for observation in tf.unstack(fixture.observations, axis=0):
        mean_pred = tf.linalg.matvec(fixture.A, mean)
        covariance_pred = fixture.A @ covariance @ tf.transpose(fixture.A) + fixture.Q
        covariance_pred = _stabilize_covariance(covariance_pred)
        sigma = _sigma_points(mean_pred, covariance_pred, lam)
        obs_sigma = range_bearing_observation_tf(sigma)
        obs_mean = _weighted_angle_observation_mean(obs_sigma, wm)
        obs_residuals = observation_residual_tf(obs_mean, obs_sigma)
        state_residuals = sigma - mean_pred
        s_mat = tf.identity(fixture.R)
        cross = tf.zeros([n, fixture.obs_dim], dtype=DTYPE)
        for i in range(int(sigma.shape[0])):
            obs_i = obs_residuals[i]
            state_i = state_residuals[i]
            s_mat = s_mat + wc[i] * tf.tensordot(obs_i, obs_i, axes=0)
            cross = cross + wc[i] * tf.tensordot(state_i, obs_i, axes=0)
        s_mat = _stabilize_covariance(s_mat)
        innovation = observation_residual_tf(obs_mean, observation)
        k_gain = tf.transpose(tf.linalg.solve(s_mat, tf.transpose(cross)))
        mean = mean_pred + tf.linalg.matvec(k_gain, innovation)
        covariance = covariance_pred - k_gain @ s_mat @ tf.transpose(k_gain)
        covariance = _stabilize_covariance(covariance)
        filtered_means.append(mean)
        filtered_covariances.append(covariance)
        predicted_observations.append(obs_mean)

    means_tensor = tf.stack(filtered_means, axis=0)
    covariances_tensor = tf.stack(filtered_covariances, axis=0)
    observations_tensor = tf.stack(predicted_observations, axis=0)
    finite = bool(
        tf.reduce_all(tf.math.is_finite(means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(covariances_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(observations_tensor)).numpy()
    )
    return UKFTFResult(
        filtered_means=means_tensor,
        filtered_covariances=covariances_tensor,
        predicted_observations=observations_tensor,
        finite=finite,
    )


def _unscented_weights(
    n: int,
    lam: float,
    alpha: float,
    beta: float,
) -> tuple[tf.Tensor, tf.Tensor]:
    denom = tf.constant(n + lam, dtype=DTYPE)
    count = 2 * n + 1
    wm = tf.ones([count], dtype=DTYPE) / (2.0 * denom)
    wc = tf.identity(wm)
    first_wm = tf.constant(lam, dtype=DTYPE) / denom
    first_wc = first_wm + tf.constant(1.0 - alpha * alpha + beta, dtype=DTYPE)
    wm = tf.concat([tf.reshape(first_wm, [1]), wm[1:]], axis=0)
    wc = tf.concat([tf.reshape(first_wc, [1]), wc[1:]], axis=0)
    return wm, wc


def _sigma_points(mean: tf.Tensor, covariance: tf.Tensor, lam: float) -> tf.Tensor:
    n = int(mean.shape[0])
    scale = tf.constant(n + lam, dtype=DTYPE)
    root = tf.linalg.cholesky(_stabilize_covariance(scale * covariance))
    points = [mean]
    for i in range(n):
        column = root[:, i]
        points.append(mean + column)
        points.append(mean - column)
    return tf.stack(points, axis=0)


def _weighted_angle_observation_mean(values: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    range_mean = tf.reduce_sum(weights * values[:, 0])
    sin_mean = tf.reduce_sum(weights * tf.sin(values[:, 1]))
    cos_mean = tf.reduce_sum(weights * tf.cos(values[:, 1]))
    return tf.stack([range_mean, tf.atan2(sin_mean, cos_mean)], axis=0)


def _stabilize_covariance(covariance: tf.Tensor) -> tf.Tensor:
    sym = 0.5 * (covariance + tf.transpose(covariance))
    eigvals = tf.linalg.eigvalsh(sym)
    min_eig = tf.reduce_min(eigvals)
    jitter = tf.maximum(tf.constant(1e-10, dtype=DTYPE) - min_eig, 0.0)
    return sym + tf.eye(int(sym.shape[0]), dtype=DTYPE) * jitter
