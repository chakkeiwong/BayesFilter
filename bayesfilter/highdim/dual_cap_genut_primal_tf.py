"""Primal-only dual-cap GenUT shape correction for LEDH reset clouds."""

from __future__ import annotations

import tensorflow as tf


Tensor = tf.Tensor
DUAL_CAP_PRIMAL_ID = "dual_cap_genut_primal_b098_p8_radial2_v1"


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _right_solve(lower: Tensor, rows: Tensor) -> Tensor:
    return tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(lower, tf.linalg.matrix_transpose(rows))
    )


def _weighted_moments(points: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, _sym(covariance)


def _uniform_moments(points: Tensor) -> tuple[Tensor, Tensor]:
    mean = tf.reduce_mean(points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("ni,nj->ij", centered, centered) / tf.cast(
        tf.shape(points)[0], points.dtype
    )
    return mean, _sym(covariance)


def _standardize_uniform(points: Tensor) -> Tensor:
    mean, covariance = _uniform_moments(points)
    return _right_solve(tf.linalg.cholesky(covariance), points - mean[None, :])


def _pair_moments(standardized: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    squared = tf.square(standardized)
    co_skew = tf.einsum("n,ni,nj->ij", weights, squared, standardized)
    co_kurtosis = tf.einsum("n,ni,nj->ij", weights, squared, squared)
    off_diagonal = 1.0 - tf.eye(
        tf.shape(standardized)[1], dtype=standardized.dtype
    )
    return co_skew * off_diagonal, co_kurtosis * off_diagonal


def _diagonal_iteration(
    standardized: Tensor,
    target_skew: Tensor,
    target_kurtosis: Tensor,
    *,
    strength: float,
    floor: float,
) -> Tensor:
    u = _standardize_uniform(standardized)
    m3 = tf.reduce_mean(tf.pow(u, 3.0), axis=0)
    m4 = tf.reduce_mean(tf.pow(u, 4.0), axis=0)
    residual = tf.stack([target_skew - m3, target_kurtosis - m4], axis=-1)

    direction3 = tf.square(u) - 1.0 - m3[None, :] * u
    direction4 = tf.pow(u, 3.0) - m3[None, :] - m4[None, :] * u
    j33 = tf.reduce_mean(3.0 * tf.square(u) * direction3, axis=0)
    j34 = tf.reduce_mean(3.0 * tf.square(u) * direction4, axis=0)
    j43 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction3, axis=0)
    j44 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction4, axis=0)
    jacobian = tf.stack(
        [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
        axis=-2,
    )
    normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
    normal += tf.cast(floor, u.dtype) * tf.eye(
        2, batch_shape=[tf.shape(u)[1]], dtype=u.dtype
    )
    rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
    coefficient = tf.cast(strength, u.dtype) * tf.linalg.solve(
        normal, rhs[:, :, None]
    )[:, :, 0]
    corrected = (
        u
        + direction3 * coefficient[None, :, 0]
        + direction4 * coefficient[None, :, 1]
    )
    return _standardize_uniform(corrected)


def _pairwise_iteration(
    standardized: Tensor,
    target_co_skew: Tensor,
    target_co_kurtosis: Tensor,
    *,
    strength: float,
    floor: float,
    particle_rms_cap: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    count = tf.shape(standardized)[0]
    uniform = tf.fill(
        [count], tf.cast(1.0, standardized.dtype) / tf.cast(count, standardized.dtype)
    )
    co_skew, co_kurtosis = _pair_moments(standardized, uniform)
    off_diagonal = 1.0 - tf.eye(
        tf.shape(standardized)[1], dtype=standardized.dtype
    )
    residual3 = off_diagonal * (target_co_skew - co_skew)
    residual4 = off_diagonal * (target_co_kurtosis - co_kurtosis)

    squared = tf.square(standardized)
    row3 = tf.linalg.matmul(standardized, residual3, transpose_b=True)
    column3 = tf.linalg.matmul(squared, residual3)
    row4 = tf.linalg.matmul(squared, residual4, transpose_b=True)
    dimension_scale = tf.cast(
        tf.maximum(tf.shape(standardized)[1] - 1, 1), standardized.dtype
    )
    direction = (
        2.0 * standardized * row3
        + column3
        + 2.0 * standardized * row4
    ) / dimension_scale
    direction -= tf.reduce_mean(direction, axis=0, keepdims=True)
    cross = tf.reduce_mean(
        standardized[:, :, None] * direction[:, None, :], axis=0
    )
    projected = direction - tf.linalg.matmul(standardized, _sym(cross))
    rms = tf.sqrt(
        tf.reduce_mean(tf.square(projected)) + tf.cast(floor, standardized.dtype)
    )
    normalized = projected / rms
    pre_cap_rms = tf.sqrt(tf.reduce_mean(tf.square(normalized), axis=1))
    if particle_rms_cap > 0.0:
        cap = tf.cast(particle_rms_cap, standardized.dtype)
        row_scale = tf.math.rsqrt(
            1.0 + tf.reduce_mean(tf.square(normalized), axis=1) / tf.square(cap)
        )
        normalized *= row_scale[:, None]
    else:
        row_scale = tf.ones_like(pre_cap_rms)
    post_cap_rms = tf.sqrt(tf.reduce_mean(tf.square(normalized), axis=1))
    corrected = standardized + tf.cast(strength, standardized.dtype) * normalized
    return (
        _standardize_uniform(corrected),
        tf.reduce_max(pre_cap_rms),
        tf.reduce_max(post_cap_rms),
        tf.reduce_min(row_scale),
    )


def dual_cap_genut_primal(
    source: Tensor,
    weights: Tensor,
    reset_points: Tensor,
    *,
    diagonal_steps: int = 4,
    diagonal_strength: float = 0.2,
    diagonal_floor: float = 1.0e-5,
    pairwise_steps: int = 4,
    pairwise_strength: float = 0.02,
    pairwise_floor: float = 1.0e-5,
    pairwise_particle_rms_cap: float = 2.0,
    coordinate_cap: float = 0.98,
    coordinate_cap_power: int = 8,
) -> dict[str, Tensor]:
    """Apply the owner-selected dual-cap family without derivative work."""

    source = tf.convert_to_tensor(source)
    weights = tf.convert_to_tensor(weights, source.dtype)
    reset_points = tf.convert_to_tensor(reset_points, source.dtype)
    if source.shape != reset_points.shape:
        raise ValueError("source and reset cloud shapes must match")
    if source.shape.rank != 2 or source.shape[1] is None:
        raise ValueError("dual-cap primal requires a static rank-two state cloud")
    if weights.shape != (source.shape[0],):
        raise ValueError("dual-cap weights have the wrong shape")
    if (
        diagonal_steps < 0
        or diagonal_strength < 0.0
        or diagonal_floor <= 0.0
        or pairwise_steps < 0
        or pairwise_strength < 0.0
        or pairwise_floor <= 0.0
        or pairwise_particle_rms_cap < 0.0
        or coordinate_cap <= 0.0
        or coordinate_cap >= 1.0
        or coordinate_cap_power < 2
        or coordinate_cap_power % 2 != 0
    ):
        raise ValueError("invalid dual-cap controls")

    target_mean, target_covariance = _weighted_moments(source, weights)
    target_cholesky = tf.linalg.cholesky(target_covariance)
    source_standardized = _right_solve(
        target_cholesky, source - target_mean[None, :]
    )
    target_skew = tf.reduce_sum(
        weights[:, None] * tf.pow(source_standardized, 3.0), axis=0
    )
    target_kurtosis = tf.reduce_sum(
        weights[:, None] * tf.pow(source_standardized, 4.0), axis=0
    )
    target_co_skew, target_co_kurtosis = _pair_moments(
        source_standardized, weights
    )

    standardized = _standardize_uniform(reset_points)
    for _ in range(diagonal_steps):
        standardized = _diagonal_iteration(
            standardized,
            target_skew,
            target_kurtosis,
            strength=diagonal_strength,
            floor=diagonal_floor,
        )

    maximum_pre_cap_rms = tf.zeros([], source.dtype)
    maximum_post_cap_rms = tf.zeros([], source.dtype)
    minimum_radial_scale = tf.ones([], source.dtype)
    if source.shape[1] > 1:
        for _ in range(pairwise_steps):
            standardized, pre_rms, post_rms, minimum_scale = _pairwise_iteration(
                standardized,
                target_co_skew,
                target_co_kurtosis,
                strength=pairwise_strength,
                floor=pairwise_floor,
                particle_rms_cap=pairwise_particle_rms_cap,
            )
            maximum_pre_cap_rms = tf.maximum(maximum_pre_cap_rms, pre_rms)
            maximum_post_cap_rms = tf.maximum(maximum_post_cap_rms, post_rms)
            minimum_radial_scale = tf.minimum(minimum_radial_scale, minimum_scale)

    pre_coordinate_cap = standardized
    cap = tf.cast(coordinate_cap, source.dtype)
    power = tf.cast(coordinate_cap_power, source.dtype)
    scaled_power = tf.pow(pre_coordinate_cap / cap, power)
    denominator = tf.pow(1.0 + scaled_power, 1.0 / power)
    capped = pre_coordinate_cap / denominator
    cap_derivative = tf.pow(1.0 + scaled_power, -1.0 / power - 1.0)
    standardized = _standardize_uniform(capped)
    particles = target_mean[None, :] + tf.linalg.matmul(
        standardized, target_cholesky, transpose_b=True
    )

    output_mean, output_covariance = _uniform_moments(particles)
    mean_residual = tf.reduce_max(tf.abs(output_mean - target_mean))
    covariance_residual = tf.reduce_max(
        tf.abs(output_covariance - target_covariance)
    )
    valid = (
        tf.reduce_all(tf.math.is_finite(particles))
        & tf.math.is_finite(mean_residual)
        & tf.math.is_finite(covariance_residual)
        & (mean_residual <= tf.cast(5.0e-4, source.dtype))
        & (covariance_residual <= tf.cast(5.0e-3, source.dtype))
    )
    return {
        "particles": particles,
        "valid": valid,
        "mean_residual": mean_residual,
        "covariance_residual": covariance_residual,
        "maximum_pairwise_pre_cap_particle_rms": maximum_pre_cap_rms,
        "maximum_pairwise_post_cap_particle_rms": maximum_post_cap_rms,
        "minimum_pairwise_particle_cap_scale": minimum_radial_scale,
        "maximum_coordinatewise_pre_cap_absolute": tf.reduce_max(
            tf.abs(pre_coordinate_cap)
        ),
        "maximum_coordinatewise_post_cap_absolute": tf.reduce_max(tf.abs(capped)),
        "mean_coordinatewise_cap_displacement": tf.reduce_mean(
            tf.abs(capped - pre_coordinate_cap)
        ),
        "fraction_coordinatewise_cap_active": tf.reduce_mean(
            tf.cast(tf.abs(capped - pre_coordinate_cap) > 1.0e-7, source.dtype)
        ),
        "minimum_coordinatewise_cap_derivative": tf.reduce_min(cap_derivative),
    }


__all__ = ["DUAL_CAP_PRIMAL_ID", "dual_cap_genut_primal"]
