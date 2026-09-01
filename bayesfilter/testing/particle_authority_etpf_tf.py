"""TensorFlow second-order LETF/ETPF fixture implementation.

This module is a bounded diagnostic implementation of the Acevedo et al.
Sinkhorn-plus-Riccati construction. It is not the q=20 production route and
must not be used to claim IID posterior samples or an exact density.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


Tensor = tf.Tensor


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.transpose(value))


def sinkhorn_first_order_transform(
    points: Tensor,
    weights: Tensor,
    *,
    regularization: float = 10.0,
    sinkhorn_steps: int = 200,
    tolerance: float = 1.0e-8,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Construct a first-order LETF matrix from a regularized OT kernel.

    Rows index forecast particles and columns index equal-weight analysis
    members. The returned matrix has column sums one and row sums N*w after
    the finite-iteration marginal correction described by the source route.
    """

    points = tf.convert_to_tensor(points, tf.float64)
    weights = tf.convert_to_tensor(weights, tf.float64)
    if points.shape.rank != 2 or weights.shape.rank != 1:
        raise ValueError("points must be [N,d] and weights must be [N]")
    particle_count = points.shape[0]
    dimension = points.shape[1]
    if particle_count is None or dimension is None or particle_count < 2:
        raise ValueError("static nontrivial fixture shape is required")
    if weights.shape[0] != particle_count:
        raise ValueError("point/weight count mismatch")
    if regularization <= 0.0 or sinkhorn_steps <= 0 or tolerance <= 0.0:
        raise ValueError("invalid Sinkhorn controls")
    weights = weights / tf.reduce_sum(weights)
    deltas = points[:, None, :] - points[None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=-1)
    scale = tf.maximum(tf.reduce_max(cost), tf.constant(1.0e-12, tf.float64))
    kernel = tf.exp(-tf.cast(regularization, tf.float64) * cost / scale)
    ones = tf.ones([particle_count], tf.float64)
    left = ones
    right = ones
    particle_mass = tf.cast(particle_count, tf.float64) * weights
    for _ in range(sinkhorn_steps):
        left = particle_mass / (tf.linalg.matvec(kernel, right) + 1.0e-30)
        right = 1.0 / (tf.linalg.matvec(tf.transpose(kernel), left) + 1.0e-30)
    transport = left[:, None] * kernel * right[None, :]
    row_mass = tf.reduce_sum(transport, axis=1) / tf.cast(particle_count, tf.float64)
    # Equation (54) of the checked source corrects finite Sinkhorn row mass
    # without changing the column sums.
    corrected = transport - (row_mass - weights)[:, None]
    row_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(corrected, axis=1) - particle_mass)
    )
    column_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(corrected, axis=0) - ones)
    )
    return corrected, {
        "sinkhorn_transport": transport,
        "sinkhorn_row_mass": row_mass,
        "row_residual": row_residual,
        "column_residual": column_residual,
        "base_nonnegative_fraction": tf.reduce_mean(
            tf.cast(transport >= 0.0, tf.float64)
        ),
        "regularization": tf.constant(regularization, tf.float64),
        "sinkhorn_steps": tf.constant(sinkhorn_steps, tf.int32),
        "tolerance": tf.constant(tolerance, tf.float64),
    }


def second_order_etpf_transform(
    points: Tensor,
    weights: Tensor,
    *,
    regularization: float = 10.0,
    sinkhorn_steps: int = 200,
    riccati_step: float = 0.1,
    riccati_max_steps: int = 1000,
    riccati_tolerance: float = 1.0e-3,
) -> tuple[Tensor, dict[str, Any]]:
    """Apply the source's Sinkhorn first-order map and Riccati correction."""

    if riccati_step <= 0.0 or riccati_max_steps <= 0 or riccati_tolerance <= 0.0:
        raise ValueError("invalid Riccati controls")
    points = tf.convert_to_tensor(points, tf.float64)
    weights = tf.convert_to_tensor(weights, tf.float64)
    particle_count = points.shape[0]
    if particle_count is None:
        raise ValueError("static particle count is required")
    transport, sinkhorn = sinkhorn_first_order_transform(
        points,
        weights,
        regularization=regularization,
        sinkhorn_steps=sinkhorn_steps,
    )
    weights = weights / tf.reduce_sum(weights)
    identity = tf.eye(particle_count, dtype=tf.float64)
    weight_matrix = tf.linalg.diag(weights)
    weight_outer = weights[:, None] * weights[None, :]
    centered_transform = transport - weights[:, None]
    covariance_target = tf.cast(particle_count, tf.float64) * (
        weight_matrix - weight_outer
    )
    riccati_a = covariance_target - tf.matmul(
        centered_transform, centered_transform, transpose_b=True
    )
    delta = tf.zeros_like(transport)
    converged = False
    delta_increment = tf.constant(float("inf"), tf.float64)
    iterations = 0
    for index in range(riccati_max_steps):
        derivative = (
            -tf.matmul(centered_transform, delta)
            - tf.matmul(delta, centered_transform, transpose_b=True)
            + riccati_a
            - tf.matmul(delta, delta)
        )
        candidate = _sym(delta + tf.cast(riccati_step, tf.float64) * derivative)
        delta_increment = tf.reduce_max(tf.abs(candidate - delta))
        delta = candidate
        iterations = index + 1
        if float(delta_increment.numpy()) <= riccati_tolerance:
            converged = True
            break
    corrected_transform = transport + delta
    corrected_centered = corrected_transform - weights[:, None]
    analysis = tf.matmul(tf.transpose(corrected_transform), points)
    weighted_mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    analysis_mean = tf.reduce_mean(analysis, axis=0)
    analysis_centered = analysis - analysis_mean[None, :]
    analysis_covariance = tf.matmul(
        analysis_centered, analysis_centered, transpose_a=True
    ) / tf.cast(particle_count, tf.float64)
    weighted_centered = points - weighted_mean[None, :]
    weighted_covariance = tf.einsum(
        "n,ni,nj->ij", weights, weighted_centered, weighted_centered
    )
    diagnostics: dict[str, Any] = {
        **sinkhorn,
        "corrected_transform": corrected_transform,
        "correction": delta,
        "riccati_a": riccati_a,
        "riccati_iterations": tf.constant(iterations, tf.int32),
        "riccati_converged": tf.constant(converged),
        "riccati_last_increment": delta_increment,
        "corrected_column_residual": tf.reduce_max(
            tf.abs(
                tf.reduce_sum(corrected_transform, axis=0)
                - tf.ones([particle_count], tf.float64)
            )
        ),
        "corrected_row_residual": tf.reduce_max(
            tf.abs(tf.reduce_sum(corrected_transform, axis=1) - tf.cast(particle_count, tf.float64) * weights)
        ),
        "analysis_mean": analysis_mean,
        "weighted_mean": weighted_mean,
        "analysis_covariance": _sym(analysis_covariance),
        "weighted_covariance": _sym(weighted_covariance),
        "mean_residual": tf.reduce_max(tf.abs(analysis_mean - weighted_mean)),
        "covariance_residual": tf.reduce_max(
            tf.abs(_sym(analysis_covariance) - _sym(weighted_covariance))
        ),
        "corrected_negative_fraction": tf.reduce_mean(
            tf.cast(corrected_transform < 0.0, tf.float64)
        ),
        "corrected_centered_row_sum_residual": tf.reduce_max(
            tf.abs(tf.reduce_sum(corrected_centered, axis=1))
        ),
    }
    return analysis, diagnostics


__all__ = ["sinkhorn_first_order_transform", "second_order_etpf_transform"]
