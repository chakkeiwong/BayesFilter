"""TensorFlow GenUT sigma-point fixture implementation.

The construction follows the bounded 2d+1 asymmetric moment equations in
Ebeigbe et al. It is a quadrature/representation diagnostic, not a density or
IID sample generator.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


def _sym(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.transpose(value))


def _moments(points: tf.Tensor, weights: tf.Tensor) -> dict[str, tf.Tensor]:
    points = tf.cast(points, tf.float64)
    weights = tf.cast(weights, tf.float64)
    weights = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return {
        "mean": mean,
        "covariance": _sym(covariance),
        "marginal_third_central": tf.reduce_sum(
            weights[:, None] * tf.pow(centered, 3), axis=0
        ),
        "marginal_fourth_central": tf.reduce_sum(
            weights[:, None] * tf.pow(centered, 4), axis=0
        ),
    }


def generalized_unscented_transform(
    points: tf.Tensor,
    weights: tf.Tensor,
    *,
    ridge: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    """Construct 2d+1 GenUT points and weights from a finite weighted cloud."""

    points = tf.convert_to_tensor(points, tf.float64)
    weights = tf.convert_to_tensor(weights, tf.float64)
    if points.shape.rank != 2 or weights.shape.rank != 1:
        raise ValueError("points must be [N,d] and weights must be [N]")
    particle_count, dimension = points.shape
    if particle_count is None or dimension is None or particle_count < 2:
        raise ValueError("static nontrivial fixture shape is required")
    if weights.shape[0] != particle_count:
        raise ValueError("point/weight count mismatch")
    source = _moments(points, weights)
    chol = tf.linalg.cholesky(
        source["covariance"] + tf.cast(ridge, tf.float64) * tf.eye(dimension, dtype=tf.float64)
    )
    centered = points - source["mean"][None, :]
    z = tf.transpose(
        tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True)
    )
    standardized = _moments(z, weights)
    skew = standardized["marginal_third_central"]
    kurtosis = standardized["marginal_fourth_central"]
    discriminant = 4.0 * kurtosis - 3.0 * tf.square(skew)
    feasible_discriminant = tf.reduce_all(discriminant > 0.0)
    u = 0.5 * (-skew + tf.sqrt(tf.maximum(discriminant, 0.0)))
    v = u + skew
    feasible_offsets = tf.reduce_all(tf.logical_and(u > 0.0, v > 0.0))
    positive_denominator = v * (u + v)
    w_plus = 1.0 / positive_denominator
    w_minus = (v / u) * w_plus
    w_zero = 1.0 - tf.reduce_sum(w_plus + w_minus)
    feasible_weights = w_zero >= 0.0
    feasible = tf.logical_and(
        tf.logical_and(feasible_discriminant, feasible_offsets), feasible_weights
    )
    sigma_z = tf.concat(
        (
            tf.zeros([1, dimension], tf.float64),
            -tf.linalg.diag(u),
            tf.linalg.diag(v),
        ),
        axis=0,
    )
    sigma_points = source["mean"][None, :] + tf.matmul(sigma_z, chol, transpose_b=True)
    sigma_weights = tf.concat((w_zero[None], w_minus, w_plus), axis=0)
    sigma = _moments(sigma_points, sigma_weights)
    diagnostics: dict[str, Any] = {
        **source,
        "standardized_skewness": skew,
        "standardized_kurtosis": kurtosis,
        "discriminant": discriminant,
        "u": u,
        "v": v,
        "w_zero": w_zero,
        "w_minus": w_minus,
        "w_plus": w_plus,
        "feasible_discriminant": feasible_discriminant,
        "feasible_offsets": feasible_offsets,
        "feasible_weights": feasible_weights,
        "feasible": feasible,
        "sigma_mean": sigma["mean"],
        "sigma_covariance": sigma["covariance"],
        "sigma_marginal_third_central": sigma["marginal_third_central"],
        "sigma_marginal_fourth_central": sigma["marginal_fourth_central"],
        "mean_residual": tf.reduce_max(tf.abs(sigma["mean"] - source["mean"])),
        "covariance_residual": tf.reduce_max(
            tf.abs(sigma["covariance"] - source["covariance"])
        ),
        "third_moment_residual": tf.reduce_max(
            tf.abs(
                sigma["marginal_third_central"]
                - source["marginal_third_central"]
            )
        ),
        "fourth_moment_residual": tf.reduce_max(
            tf.abs(
                sigma["marginal_fourth_central"]
                - source["marginal_fourth_central"]
            )
        ),
        "ridge": tf.constant(ridge, tf.float64),
    }
    return sigma_points, sigma_weights, diagnostics


__all__ = ["generalized_unscented_transform"]
