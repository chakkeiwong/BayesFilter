"""Exact Kalman reference for the linear structural AR(1) completion fixture."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.fixtures.structural_ar1_quadratic_tf import (
    DTYPE,
    StructuralAR1QuadraticTFFixture,
)


@dataclass(frozen=True)
class KalmanStructuralAR1Result:
    scalar: tf.Tensor
    filtered_means: tf.Tensor
    filtered_covariances: tf.Tensor
    log_likelihood: tf.Tensor
    finite: bool
    reference_id: str = "exact_kalman_structural_ar1_linear_tf"


def run_kalman_structural_ar1_linear_tf(
    fixture: StructuralAR1QuadraticTFFixture,
    *,
    b: tf.Tensor,
) -> KalmanStructuralAR1Result:
    b = tf.cast(b, DTYPE)
    transition_matrix = tf.stack(
        [
            tf.stack([fixture.rho, tf.constant(0.0, DTYPE)]),
            tf.stack([b * fixture.rho, fixture.a]),
        ],
        axis=0,
    )
    shock_loading = tf.reshape(tf.stack([fixture.sigma, b * fixture.sigma]), [2, 1])
    transition_covariance = tf.linalg.matmul(shock_loading, shock_loading, transpose_b=True)
    observation_matrix = tf.reshape(tf.stack([fixture.lam, tf.constant(1.0, DTYPE)]), [1, 2])
    observation_covariance = tf.reshape(fixture.observation_scale * fixture.observation_scale, [1, 1])
    mean = tf.stack([fixture.m0_mean, fixture.k0], axis=0)
    covariance = tf.linalg.diag(tf.stack([fixture.m0_variance, tf.constant(1e-10, DTYPE)]))
    log_likelihood = tf.constant(0.0, DTYPE)
    filtered_means = []
    filtered_covariances = []

    for observation in tf.unstack(tf.cast(fixture.observations, DTYPE), axis=0):
        predicted_mean = tf.linalg.matvec(transition_matrix, mean)
        predicted_covariance = (
            tf.linalg.matmul(
                transition_matrix,
                tf.linalg.matmul(covariance, transition_matrix, transpose_b=True),
            )
            + transition_covariance
        )
        innovation = tf.reshape(observation, [1]) - tf.linalg.matvec(observation_matrix, predicted_mean)
        innovation_covariance = (
            tf.linalg.matmul(
                observation_matrix,
                tf.linalg.matmul(predicted_covariance, observation_matrix, transpose_b=True),
            )
            + observation_covariance
        )
        chol = tf.linalg.cholesky(innovation_covariance)
        solved_innovation = tf.linalg.cholesky_solve(chol, tf.reshape(innovation, [1, 1]))
        increment = -0.5 * (
            tf.constant(1.0, DTYPE) * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
            + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
            + tf.squeeze(tf.linalg.matmul(tf.reshape(innovation, [1, 1]), solved_innovation, transpose_a=True))
        )
        cross_covariance = tf.linalg.matmul(predicted_covariance, observation_matrix, transpose_b=True)
        kalman_gain = tf.linalg.cholesky_solve(chol, tf.transpose(cross_covariance))
        kalman_gain = tf.transpose(kalman_gain)
        mean = predicted_mean + tf.linalg.matvec(kalman_gain, innovation)
        covariance = predicted_covariance - tf.linalg.matmul(
            kalman_gain,
            tf.linalg.matmul(innovation_covariance, kalman_gain, transpose_b=True),
        )
        covariance = 0.5 * (covariance + tf.transpose(covariance))
        filtered_means.append(mean)
        filtered_covariances.append(covariance)
        log_likelihood = log_likelihood + increment

    means = tf.stack(filtered_means, axis=0)
    covariances = tf.stack(filtered_covariances, axis=0)
    scalar = -log_likelihood
    finite = bool(
        tf.math.is_finite(scalar).numpy()
        and tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(covariances)).numpy()
    )
    return KalmanStructuralAR1Result(
        scalar=scalar,
        filtered_means=means,
        filtered_covariances=covariances,
        log_likelihood=log_likelihood,
        finite=finite,
    )
