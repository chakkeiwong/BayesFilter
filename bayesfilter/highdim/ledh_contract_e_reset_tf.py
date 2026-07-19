"""TensorFlow Contract E-Chol reset on an already transported particle cloud."""

from __future__ import annotations

from typing import Any

import tensorflow as tf


def _sym(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _apply_rows(points: tf.Tensor, operator: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matmul(points, operator, transpose_b=True)


def _weighted_moments(
    points: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, :, None] * points, axis=1)
    centered = points - mean[:, None, :]
    covariance = tf.einsum("bn,bni,bnj->bij", weights, centered, centered)
    return mean, _sym(covariance)


def _uniform_moments(points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(points, axis=1)
    centered = points - mean[:, None, :]
    count = tf.cast(tf.shape(points)[1], points.dtype)
    covariance = tf.einsum("bni,bnj->bij", centered, centered) / count
    return mean, _sym(covariance)


def _factor_condition_proxy(chol: tf.Tensor) -> tf.Tensor:
    dimension = tf.shape(chol)[-1]
    identity = tf.eye(dimension, batch_shape=[tf.shape(chol)[0]], dtype=chol.dtype)
    inverse_action = tf.linalg.triangular_solve(chol, identity)
    return tf.linalg.norm(chol, ord="fro", axis=[-2, -1]) * tf.linalg.norm(
        inverse_action, ord="fro", axis=[-2, -1]
    )


def _contract_e_chol_cloud_forward_core(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
) -> dict[str, tf.Tensor]:
    dtype = source_particles.dtype
    batch_size = tf.shape(source_particles)[0]
    state_dimension = tf.shape(source_particles)[2]
    ridge = tf.cast(ridge, dtype) + tf.zeros([batch_size], dtype=dtype)
    identity = tf.eye(
        state_dimension, batch_shape=[batch_size], dtype=dtype
    )
    ridge_identity = ridge[:, None, None] * identity

    target_mean, target_cov = _weighted_moments(
        source_particles, normalized_weights
    )
    plus_mean, plus_cov = _uniform_moments(transported_particles)
    gap = _sym(target_cov - plus_cov)
    gap_eigenvalues = tf.linalg.eigvalsh(gap)
    gap_chol = tf.linalg.cholesky(gap + ridge_identity)

    injected_particles = transported_particles + _apply_rows(
        residual_design, gap_chol
    )
    injected_mean, injected_cov = _uniform_moments(injected_particles)
    centered_injected = injected_particles - injected_mean[:, None, :]

    target_chol = tf.linalg.cholesky(target_cov + ridge_identity)
    injected_chol = tf.linalg.cholesky(injected_cov + ridge_identity)
    solved = tf.linalg.triangular_solve(
        injected_chol,
        tf.linalg.matrix_transpose(target_chol),
        adjoint=True,
    )
    affine = tf.linalg.matrix_transpose(solved)
    particles = target_mean[:, None, :] + _apply_rows(centered_injected, affine)
    output_mean, output_cov = _uniform_moments(particles)

    ridged_left = tf.linalg.matmul(
        tf.linalg.matmul(affine, injected_cov + ridge_identity),
        affine,
        transpose_b=True,
    )
    ridged_right = target_cov + ridge_identity
    ridged_identity_residual = ridged_left - ridged_right
    ridged_identity_absolute_scale = tf.linalg.matmul(
        tf.linalg.matmul(
            tf.abs(affine), tf.abs(injected_cov + ridge_identity)
        ),
        tf.abs(affine),
        transpose_b=True,
    ) + tf.abs(ridged_right)
    raw_covariance_residual = output_cov - target_cov
    predicted_raw_covariance_residual = ridge[:, None, None] * (
        identity - tf.linalg.matmul(affine, affine, transpose_b=True)
    )
    residual_design_sum = tf.reduce_sum(residual_design, axis=1)
    residual_design_absolute_scale = tf.reduce_sum(
        tf.abs(residual_design), axis=1
    )
    mean_residual = output_mean - target_mean

    gap_diagonal = tf.linalg.diag_part(gap_chol)
    target_diagonal = tf.linalg.diag_part(target_chol)
    injected_diagonal = tf.linalg.diag_part(injected_chol)
    finite = tf.reduce_all(
        tf.stack(
            [
                tf.reduce_all(tf.math.is_finite(particles), axis=[1, 2]),
                tf.reduce_all(tf.math.is_finite(gap_chol), axis=[1, 2]),
                tf.reduce_all(tf.math.is_finite(target_chol), axis=[1, 2]),
                tf.reduce_all(tf.math.is_finite(injected_chol), axis=[1, 2]),
            ],
            axis=1,
        ),
        axis=1,
    )
    factor_diagonal_positive = (
        tf.reduce_all(gap_diagonal > 0, axis=1)
        & tf.reduce_all(target_diagonal > 0, axis=1)
        & tf.reduce_all(injected_diagonal > 0, axis=1)
    )

    return {
        "particles": particles,
        "target_mean": target_mean,
        "target_cov": target_cov,
        "plus_mean": plus_mean,
        "plus_cov": plus_cov,
        "gap": gap,
        "gap_eigenvalues": gap_eigenvalues,
        "gap_chol": gap_chol,
        "injected_particles": injected_particles,
        "injected_mean": injected_mean,
        "injected_cov": injected_cov,
        "centered_injected": centered_injected,
        "target_chol": target_chol,
        "injected_chol": injected_chol,
        "affine": affine,
        "output_mean": output_mean,
        "output_cov": output_cov,
        "residual_design_sum": residual_design_sum,
        "residual_design_absolute_scale": residual_design_absolute_scale,
        "ridged_identity_residual": ridged_identity_residual,
        "ridged_identity_absolute_scale": ridged_identity_absolute_scale,
        "ridged_identity_left": ridged_left,
        "ridged_identity_right": ridged_right,
        "raw_covariance_residual": raw_covariance_residual,
        "predicted_raw_covariance_residual": predicted_raw_covariance_residual,
        "mean_residual": mean_residual,
        "gap_chol_diagonal": gap_diagonal,
        "target_chol_diagonal": target_diagonal,
        "injected_chol_diagonal": injected_diagonal,
        "gap_condition_proxy": _factor_condition_proxy(gap_chol),
        "target_condition_proxy": _factor_condition_proxy(target_chol),
        "injected_condition_proxy": _factor_condition_proxy(injected_chol),
        "finite": finite,
        "factor_diagonal_positive": factor_diagonal_positive,
        "ridge": ridge,
    }


def _weighted_moments_jvp(
    points: tf.Tensor,
    weights: tf.Tensor,
    points_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    mean, _ = _weighted_moments(points, weights)
    mean_tangent = tf.reduce_sum(
        weights_tangent[:, :, None] * points
        + weights[:, :, None] * points_tangent,
        axis=1,
    )
    centered = points - mean[:, None, :]
    centered_tangent = points_tangent - mean_tangent[:, None, :]
    covariance_tangent = tf.einsum(
        "bn,bni,bnj->bij", weights_tangent, centered, centered
    )
    covariance_tangent += tf.einsum(
        "bn,bni,bnj->bij", weights, centered_tangent, centered
    )
    covariance_tangent += tf.einsum(
        "bn,bni,bnj->bij", weights, centered, centered_tangent
    )
    return mean_tangent, _sym(covariance_tangent)


def _uniform_moments_jvp(
    points: tf.Tensor, points_tangent: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(points, axis=1)
    mean_tangent = tf.reduce_mean(points_tangent, axis=1)
    centered = points - mean[:, None, :]
    centered_tangent = points_tangent - mean_tangent[:, None, :]
    count = tf.cast(tf.shape(points)[1], points.dtype)
    covariance_tangent = (
        tf.einsum("bni,bnj->bij", centered_tangent, centered)
        + tf.einsum("bni,bnj->bij", centered, centered_tangent)
    ) / count
    return mean_tangent, _sym(covariance_tangent)


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    left = tf.linalg.triangular_solve(chol, matrix_tangent)
    inner = tf.linalg.triangular_solve(
        chol, tf.linalg.matrix_transpose(left)
    )
    lower = tf.linalg.band_part(inner, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    return tf.linalg.matmul(chol, phi)


def _right_triangular_solve(
    right_hand_side: tf.Tensor, lower: tf.Tensor
) -> tf.Tensor:
    solved_transpose = tf.linalg.triangular_solve(
        lower, tf.linalg.matrix_transpose(right_hand_side), adjoint=True
    )
    return tf.linalg.matrix_transpose(solved_transpose)


def _contract_e_chol_cloud_jvp_core(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    transported_particles_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _contract_e_chol_cloud_forward_core(
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
    )
    return _contract_e_chol_cloud_jvp_from_forward_core(
        forward,
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
        source_particles_tangent,
        normalized_weights_tangent,
        transported_particles_tangent,
        residual_design_tangent,
        ridge_tangent,
    )


def _contract_e_chol_cloud_jvp_from_forward_core(
    forward: dict[str, tf.Tensor],
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    transported_particles_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Differentiate Contract E while reusing its exact forward factorization."""

    batch_size = tf.shape(source_particles)[0]
    state_dimension = tf.shape(source_particles)[2]
    identity = tf.eye(
        state_dimension, batch_shape=[batch_size], dtype=source_particles.dtype
    )
    ridge_tangent = tf.cast(ridge_tangent, source_particles.dtype) + tf.zeros(
        [batch_size], dtype=source_particles.dtype
    )
    ridge_identity_tangent = ridge_tangent[:, None, None] * identity

    target_mean_tangent, target_cov_tangent = _weighted_moments_jvp(
        source_particles,
        normalized_weights,
        source_particles_tangent,
        normalized_weights_tangent,
    )
    plus_mean_tangent, plus_cov_tangent = _uniform_moments_jvp(
        transported_particles, transported_particles_tangent
    )
    gap_tangent = _sym(target_cov_tangent - plus_cov_tangent)
    gap_chol_tangent = _cholesky_jvp(
        forward["gap_chol"], gap_tangent + ridge_identity_tangent
    )
    injected_particles_tangent = (
        transported_particles_tangent
        + _apply_rows(residual_design_tangent, forward["gap_chol"])
        + _apply_rows(residual_design, gap_chol_tangent)
    )
    injected_mean_tangent, injected_cov_tangent = _uniform_moments_jvp(
        forward["injected_particles"], injected_particles_tangent
    )
    centered_injected_tangent = (
        injected_particles_tangent - injected_mean_tangent[:, None, :]
    )
    target_chol_tangent = _cholesky_jvp(
        forward["target_chol"], target_cov_tangent + ridge_identity_tangent
    )
    injected_chol_tangent = _cholesky_jvp(
        forward["injected_chol"],
        injected_cov_tangent + ridge_identity_tangent,
    )
    affine_tangent = _right_triangular_solve(
        target_chol_tangent
        - tf.linalg.matmul(forward["affine"], injected_chol_tangent),
        forward["injected_chol"],
    )
    particles_tangent = (
        target_mean_tangent[:, None, :]
        + _apply_rows(centered_injected_tangent, forward["affine"])
        + _apply_rows(forward["centered_injected"], affine_tangent)
    )
    return {
        "particles": particles_tangent,
        "target_mean": target_mean_tangent,
        "target_cov": target_cov_tangent,
        "plus_mean": plus_mean_tangent,
        "plus_cov": plus_cov_tangent,
        "gap": gap_tangent,
        "gap_chol": gap_chol_tangent,
        "injected_particles": injected_particles_tangent,
        "injected_mean": injected_mean_tangent,
        "injected_cov": injected_cov_tangent,
        "centered_injected": centered_injected_tangent,
        "target_chol": target_chol_tangent,
        "injected_chol": injected_chol_tangent,
        "affine": affine_tangent,
    }


def _cholesky_vjp(chol: tf.Tensor, chol_bar: tf.Tensor) -> tf.Tensor:
    product = tf.linalg.matmul(chol, chol_bar, transpose_a=True)
    lower = tf.linalg.band_part(product, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(lower))
    left = tf.linalg.triangular_solve(chol, phi, adjoint=True)
    solved_transpose = tf.linalg.triangular_solve(
        chol, tf.linalg.matrix_transpose(left), adjoint=True
    )
    return _sym(tf.linalg.matrix_transpose(solved_transpose))


def _uniform_covariance_vjp(
    points: tf.Tensor, covariance_bar: tf.Tensor
) -> tf.Tensor:
    centered = points - tf.reduce_mean(points, axis=1, keepdims=True)
    count = tf.cast(tf.shape(points)[1], points.dtype)
    operator = _sym(covariance_bar)
    row_action = tf.reduce_sum(
        centered[:, :, None, :] * operator[:, None, :, :], axis=-1
    )
    return (2.0 / count) * row_action


def _weighted_moments_vjp(
    points: tf.Tensor,
    weights: tf.Tensor,
    mean: tf.Tensor,
    mean_bar: tf.Tensor,
    covariance_bar: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    centered = points - mean[:, None, :]
    covariance_bar = _sym(covariance_bar)
    points_bar = weights[:, :, None] * (
        mean_bar[:, None, :]
        + 2.0 * _apply_rows(centered, covariance_bar)
    )
    weights_bar = tf.reduce_sum(points * mean_bar[:, None, :], axis=2)
    weights_bar += tf.einsum(
        "bni,bij,bnj->bn", centered, covariance_bar, centered
    )
    return points_bar, weights_bar


def _contract_e_chol_cloud_vjp_core(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    upstream_particles: tf.Tensor,
) -> dict[str, Any]:
    forward = _contract_e_chol_cloud_forward_core(
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
    )
    target_mean_bar = tf.reduce_sum(upstream_particles, axis=1)
    centered_injected_bar = _apply_rows(
        upstream_particles, tf.linalg.matrix_transpose(forward["affine"])
    )
    affine_bar = tf.einsum(
        "bni,bnj->bij", upstream_particles, forward["centered_injected"]
    )
    injected_particles_bar = centered_injected_bar - tf.reduce_mean(
        centered_injected_bar, axis=1, keepdims=True
    )

    target_chol_bar = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(
            forward["injected_chol"],
            tf.linalg.matrix_transpose(affine_bar),
        )
    )
    injected_chol_bar = -tf.linalg.matmul(
        forward["affine"], target_chol_bar, transpose_a=True
    )
    target_cov_from_affine_bar = _cholesky_vjp(
        forward["target_chol"], target_chol_bar
    )
    injected_cov_bar = _cholesky_vjp(
        forward["injected_chol"], injected_chol_bar
    )
    injected_particles_bar += _uniform_covariance_vjp(
        forward["injected_particles"], injected_cov_bar
    )

    transported_particles_bar = injected_particles_bar
    residual_design_bar = _apply_rows(
        injected_particles_bar, tf.linalg.matrix_transpose(forward["gap_chol"])
    )
    gap_chol_bar = tf.einsum(
        "bni,bnj->bij", injected_particles_bar, residual_design
    )
    gap_bar = _cholesky_vjp(forward["gap_chol"], gap_chol_bar)
    target_cov_bar = target_cov_from_affine_bar + _sym(gap_bar)
    plus_cov_bar = -_sym(gap_bar)
    transported_particles_bar += _uniform_covariance_vjp(
        transported_particles, plus_cov_bar
    )

    source_particles_bar, normalized_weights_bar = _weighted_moments_vjp(
        source_particles,
        normalized_weights,
        forward["target_mean"],
        target_mean_bar,
        target_cov_bar,
    )
    ridge_bar = (
        tf.linalg.trace(gap_bar)
        + tf.linalg.trace(target_cov_from_affine_bar)
        + tf.linalg.trace(injected_cov_bar)
    )
    return {
        "source_particles": source_particles_bar,
        "normalized_weights": normalized_weights_bar,
        "transported_particles": transported_particles_bar,
        "residual_design": residual_design_bar,
        "ridge": ridge_bar,
        "intermediates": {
            "target_mean_bar": target_mean_bar,
            "affine_bar": affine_bar,
            "target_chol_bar": target_chol_bar,
            "injected_chol_bar": injected_chol_bar,
            "target_cov_from_affine_bar": target_cov_from_affine_bar,
            "injected_cov_bar": injected_cov_bar,
            "injected_particles_bar": injected_particles_bar,
            "gap_chol_bar": gap_chol_bar,
            "gap_bar": gap_bar,
            "target_cov_total_bar": target_cov_bar,
            "plus_cov_bar": plus_cov_bar,
        },
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_chol_cloud_forward_tf(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Apply the fixed-ridge Contract E-Chol cloud reset."""

    return _contract_e_chol_cloud_forward_core(
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_chol_cloud_jvp_tf(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    source_particles_tangent: tf.Tensor,
    normalized_weights_tangent: tf.Tensor,
    transported_particles_tangent: tf.Tensor,
    residual_design_tangent: tf.Tensor,
    ridge_tangent: tf.Tensor,
) -> tf.Tensor:
    """Return only the output-particle tangent for the five cloud inputs."""

    return _contract_e_chol_cloud_jvp_core(
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
        source_particles_tangent,
        normalized_weights_tangent,
        transported_particles_tangent,
        residual_design_tangent,
        ridge_tangent,
    )["particles"]


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_chol_cloud_vjp_tf(
    source_particles: tf.Tensor,
    normalized_weights: tf.Tensor,
    transported_particles: tf.Tensor,
    residual_design: tf.Tensor,
    ridge: tf.Tensor,
    upstream_particles: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Return separate input adjoints for a particle-output cotangent."""

    result = _contract_e_chol_cloud_vjp_core(
        source_particles,
        normalized_weights,
        transported_particles,
        residual_design,
        ridge,
        upstream_particles,
    )
    return {
        "source_particles": result["source_particles"],
        "normalized_weights": result["normalized_weights"],
        "transported_particles": result["transported_particles"],
        "residual_design": result["residual_design"],
        "ridge": result["ridge"],
    }
