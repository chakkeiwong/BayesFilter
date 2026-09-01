"""TensorFlow linear-Gaussian LEDH-PFPF fixture map and density identity."""

from __future__ import annotations

from typing import Any, Mapping

import tensorflow as tf


def _flow_coefficients(
    prior_mean: tf.Tensor,
    prior_covariance: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation: tf.Tensor,
    lam: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    dimension = prior_mean.shape[0]
    identity = tf.eye(dimension, dtype=tf.float64)
    h = tf.cast(observation_matrix, tf.float64)
    p = tf.cast(prior_covariance, tf.float64)
    r = tf.cast(observation_covariance, tf.float64)
    z = tf.cast(observation, tf.float64)
    innovation_covariance = lam * tf.matmul(h, tf.matmul(p, h, transpose_b=True)) + r
    inverse_innovation = tf.linalg.inv(innovation_covariance)
    # Equation (7): -1/2 P H^T (lambda H P H^T + R)^-1 H.
    p_h_t = tf.matmul(p, h, transpose_b=True)
    a = -0.5 * tf.matmul(
        tf.matmul(p_h_t, inverse_innovation), h
    )
    r_inverse = tf.linalg.inv(r)
    b_inner = tf.matmul(
        identity + lam * a,
        tf.matmul(p_h_t, tf.matmul(r_inverse, z[..., None])),
    )[:, 0] + tf.matmul(a, prior_mean[..., None])[:, 0]
    b = tf.matmul(identity + 2.0 * lam * a, b_inner[..., None])[:, 0]
    return a, b


def ledh_flow(
    initial_points: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_covariance: tf.Tensor,
    observation_matrix: tf.Tensor,
    observation_covariance: tf.Tensor,
    observation: tf.Tensor,
    step_sizes: tf.Tensor,
) -> tuple[tf.Tensor, dict[str, Any]]:
    """Apply the repeated affine LEDH map and accumulate log determinants."""

    points = tf.convert_to_tensor(initial_points, tf.float64)
    prior_mean = tf.convert_to_tensor(prior_mean, tf.float64)
    prior_covariance = tf.convert_to_tensor(prior_covariance, tf.float64)
    step_sizes = tf.convert_to_tensor(step_sizes, tf.float64)
    if points.shape.rank != 2 or prior_mean.shape.rank != 1:
        raise ValueError("invalid LEDH fixture shapes")
    if step_sizes.shape.rank != 1 or step_sizes.shape[0] is None:
        raise ValueError("step schedule must be static rank one")
    if float(tf.reduce_min(step_sizes).numpy()) <= 0.0:
        raise ValueError("step sizes must be positive")
    if abs(float(tf.reduce_sum(step_sizes).numpy()) - 1.0) > 1.0e-12:
        raise ValueError("step sizes must sum to one")
    covariance_chol = tf.linalg.cholesky(prior_covariance)
    if not bool(tf.reduce_all(tf.math.is_finite(covariance_chol)).numpy()):
        raise ValueError("prior covariance is not positive definite")
    current = points
    lam = tf.constant(0.0, tf.float64)
    logdet = tf.constant(0.0, tf.float64)
    matrices: list[tf.Tensor] = []
    offsets: list[tf.Tensor] = []
    determinants: list[tf.Tensor] = []
    for step in tf.unstack(step_sizes):
        lam = lam + step
        a, b = _flow_coefficients(
            prior_mean,
            prior_covariance,
            observation_matrix,
            observation_covariance,
            observation,
            lam,
        )
        matrix = tf.eye(points.shape[1], dtype=tf.float64) + step * a
        determinant = tf.linalg.det(matrix)
        if not bool(tf.math.is_finite(determinant).numpy()) or float(tf.abs(determinant).numpy()) <= 0.0:
            raise ValueError("LEDH step is noninvertible")
        current = tf.matmul(current, matrix, transpose_b=True) + step * b[None, :]
        logdet = logdet + tf.math.log(tf.abs(determinant))
        matrices.append(matrix)
        offsets.append(step * b)
        determinants.append(determinant)
    return current, {
        "matrices": matrices,
        "offsets": offsets,
        "determinants": tf.stack(determinants),
        "logdet": logdet,
        "final_lambda": lam,
        "prior_covariance_cholesky": covariance_chol,
    }


def ledh_inverse(final_points: tf.Tensor, flow: Mapping[str, Any]) -> tf.Tensor:
    """Reverse the stored affine steps exactly."""

    current = tf.convert_to_tensor(final_points, tf.float64)
    for matrix, offset in reversed(tuple(zip(flow["matrices"], flow["offsets"]))):
        current = tf.linalg.matrix_transpose(
            tf.linalg.solve(matrix, tf.transpose(current - offset[None, :]))
        )
    return current


def gaussian_log_density(points: tf.Tensor, mean: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    points = tf.convert_to_tensor(points, tf.float64)
    mean = tf.convert_to_tensor(mean, tf.float64)
    covariance = tf.convert_to_tensor(covariance, tf.float64)
    chol = tf.linalg.cholesky(covariance)
    centered = points - mean[None, :]
    solved = tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True)
    quadratic = tf.reduce_sum(tf.square(solved), axis=0)
    logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    dimension = tf.cast(tf.shape(points)[1], tf.float64)
    return -0.5 * (dimension * tf.math.log(2.0 * tf.constant(3.141592653589793, tf.float64)) + logdet + quadratic)


__all__ = ["gaussian_log_density", "ledh_flow", "ledh_inverse"]
