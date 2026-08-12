"""Batch-native finite GenUT value and bounded-memory forward score.

This module is deliberately separate from the scalar reference implementation.
It carries a genuine posterior batch axis through every TensorFlow operation;
there is no sample-row map or scalar fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


Tensor = tf.Tensor
InitialValue = Callable[[Tensor, Tensor], Tensor]
InitialTangent = Callable[[Tensor, Tensor], Tensor]
TransitionValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
TransitionTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationValue = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]
ObservationTangent = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class BatchCandidateModelAdapter:
    """Leading-batch GenUT model equations and their total forward JVP."""

    state_dimension: int
    parameter_count: int
    initial_value: InitialValue
    initial_tangent: InitialTangent
    transition_value: TransitionValue
    transition_tangent: TransitionTangent
    observation_value: ObservationValue
    observation_tangent: ObservationTangent

    def __post_init__(self) -> None:
        if self.state_dimension < 1 or self.parameter_count < 1:
            raise ValueError("adapter dimensions must be positive")


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _sym_tangent(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.transpose(value, [0, 2, 1, 3]))


def _right_solve(lower: Tensor, rows: Tensor) -> Tensor:
    return tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(lower, tf.linalg.matrix_transpose(rows))
    )


def _cholesky_jvp(chol: Tensor, matrix_tangent: Tensor) -> Tensor:
    """Batched Cholesky JVP for tangents `[B,d,d,p]`."""

    batch_size = tf.shape(chol)[0]
    dimension = tf.shape(chol)[1]
    parameter_count = tf.shape(matrix_tangent)[-1]
    flat_chol = tf.reshape(
        tf.broadcast_to(chol[:, None, :, :], [batch_size, parameter_count, dimension, dimension]),
        [-1, dimension, dimension],
    )
    flat_tangent = tf.reshape(
        tf.transpose(matrix_tangent, [0, 3, 1, 2]),
        [-1, dimension, dimension],
    )
    flat_result = reset._cholesky_jvp(flat_chol, flat_tangent)  # noqa: SLF001
    return tf.transpose(
        tf.reshape(flat_result, [batch_size, parameter_count, dimension, dimension]),
        [0, 2, 3, 1],
    )


def _right_solve_jvp(
    lower: Tensor,
    lower_tangent: Tensor,
    rows: Tensor,
    rows_tangent: Tensor,
) -> Tensor:
    solved = _right_solve(lower, rows)
    rhs_tangent = rows_tangent - tf.einsum(
        "bni,bjip->bnjp", solved, lower_tangent
    )
    batch_size = tf.shape(lower)[0]
    parameter_count = tf.shape(rows_tangent)[-1]
    particle_count = tf.shape(rows)[1]
    dimension = tf.shape(rows)[2]
    flat_lower = tf.reshape(
        tf.broadcast_to(
            lower[:, None, :, :],
            [batch_size, parameter_count, dimension, dimension],
        ),
        [-1, dimension, dimension],
    )
    flat_rhs = tf.reshape(
        tf.transpose(rhs_tangent, [0, 3, 1, 2]),
        [-1, particle_count, dimension],
    )
    flat_result = _right_solve(flat_lower, flat_rhs)
    return tf.transpose(
        tf.reshape(
            flat_result,
            [batch_size, parameter_count, particle_count, dimension],
        ),
        [0, 2, 3, 1],
    )


def _weighted_moments_jvp(
    points: Tensor,
    weights: Tensor,
    points_tangent: Tensor,
    weights_tangent: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = tf.reduce_sum(weights[:, :, None] * points, axis=1)
    mean_tangent = tf.reduce_sum(
        weights_tangent[:, :, None, :] * points[:, :, :, None]
        + weights[:, :, None, None] * points_tangent,
        axis=1,
    )
    centered = points - mean[:, None, :]
    centered_tangent = points_tangent - mean_tangent[:, None, :, :]
    covariance = tf.einsum("bn,bni,bnj->bij", weights, centered, centered)
    covariance_tangent = tf.einsum(
        "bnp,bni,bnj->bijp", weights_tangent, centered, centered
    )
    covariance_tangent += tf.einsum(
        "bn,bnip,bnj->bijp", weights, centered_tangent, centered
    )
    covariance_tangent += tf.einsum(
        "bn,bni,bnjp->bijp", weights, centered, centered_tangent
    )
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _uniform_moments_jvp(
    points: Tensor, points_tangent: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    mean = tf.reduce_mean(points, axis=1)
    mean_tangent = tf.reduce_mean(points_tangent, axis=1)
    centered = points - mean[:, None, :]
    centered_tangent = points_tangent - mean_tangent[:, None, :, :]
    count = tf.cast(tf.shape(points)[1], points.dtype)
    covariance = tf.einsum("bni,bnj->bij", centered, centered) / count
    covariance_tangent = (
        tf.einsum("bnip,bnj->bijp", centered_tangent, centered)
        + tf.einsum("bni,bnjp->bijp", centered, centered_tangent)
    ) / count
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _sinkhorn_barycentric_batch_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
) -> dict[str, Tensor]:
    if epsilon <= 0.0 or sinkhorn_steps <= 0 or balance_steps < 0:
        raise ValueError("epsilon and Sinkhorn counts must be valid")
    deltas = particles[:, :, None, :] - particles[:, None, :, :]
    delta_tangent = (
        particle_tangent[:, :, None, :, :] - particle_tangent[:, None, :, :, :]
    )
    cost = tf.reduce_sum(tf.square(deltas), axis=3)
    cost_tangent = 2.0 * tf.reduce_sum(
        deltas[:, :, :, :, None] * delta_tangent, axis=3
    )
    mean_cost = tf.reduce_mean(cost, axis=[1, 2])
    floor = tf.cast(1.0e-3, particles.dtype)
    cost_scale = tf.maximum(mean_cost, floor)
    cost_scale_tangent = tf.where(
        mean_cost[:, None] > floor,
        tf.reduce_mean(cost_tangent, axis=[1, 2]),
        tf.zeros_like(tf.reduce_mean(cost_tangent, axis=[1, 2])),
    )
    epsilon_tensor = tf.cast(epsilon, particles.dtype)
    exponent = -cost / (cost_scale[:, None, None] * epsilon_tensor)
    exponent_tangent = -(
        cost_tangent / cost_scale[:, None, None, None]
        - cost[:, :, :, None]
        * cost_scale_tangent[:, None, None, :]
        / tf.square(cost_scale)[:, None, None, None]
    ) / epsilon_tensor
    kernel = tf.exp(exponent)
    kernel_tangent = kernel[:, :, :, None] * exponent_tangent
    batch_size = tf.shape(particles)[0]
    particle_count = tf.shape(particles)[1]
    parameter_count = tf.shape(particle_tangent)[-1]
    uniform = tf.fill(
        [batch_size, particle_count],
        tf.cast(1.0, particles.dtype) / tf.cast(particle_count, particles.dtype),
    )
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    left_tangent = tf.zeros([batch_size, particle_count, parameter_count], particles.dtype)
    right_tangent = tf.zeros_like(left_tangent)
    tiny = tf.cast(1.0e-7, particles.dtype)

    def body(index, left_value, right_value, left_jvp, right_jvp):
        left_denominator = tf.einsum("bij,bj->bi", kernel, right_value) + tiny
        left_denominator_tangent = (
            tf.einsum("bijp,bj->bip", kernel_tangent, right_value)
            + tf.einsum("bij,bjp->bip", kernel, right_jvp)
        )
        left_new = uniform / left_denominator
        left_tangent_new = -uniform[:, :, None] * left_denominator_tangent / tf.square(
            left_denominator
        )[:, :, None]
        right_denominator = tf.einsum("bij,bi->bj", kernel, left_new) + tiny
        right_denominator_tangent = (
            tf.einsum("bijp,bi->bjp", kernel_tangent, left_new)
            + tf.einsum("bij,bip->bjp", kernel, left_tangent_new)
        )
        right_new = weights / right_denominator
        right_tangent_new = (
            weight_tangent / right_denominator[:, :, None]
            - weights[:, :, None]
            * right_denominator_tangent
            / tf.square(right_denominator)[:, :, None]
        )
        return index + 1, left_new, right_new, left_tangent_new, right_tangent_new

    _, left, right, left_tangent, right_tangent = tf.while_loop(
        lambda index, *_: index < tf.cast(sinkhorn_steps + balance_steps, tf.int32),
        body,
        (tf.zeros([], tf.int32), left, right, left_tangent, right_tangent),
        parallel_iterations=1,
    )
    coupling = left[:, :, None] * kernel * right[:, None, :]
    coupling_tangent = (
        left_tangent[:, :, None, :] * kernel[:, :, :, None] * right[:, None, :, None]
        + left[:, :, None, None] * kernel_tangent * right[:, None, :, None]
        + left[:, :, None, None] * kernel[:, :, :, None] * right_tangent[:, None, :, :]
    )
    row_mass = tf.reduce_sum(coupling, axis=2)
    row_mass_tangent = tf.reduce_sum(coupling_tangent, axis=2)
    numerator = tf.einsum("bij,bjd->bid", coupling, particles)
    numerator_tangent = (
        tf.einsum("bijp,bjd->bidp", coupling_tangent, particles)
        + tf.einsum("bij,bjdp->bidp", coupling, particle_tangent)
    )
    barycentric = numerator / row_mass[:, :, None]
    barycentric_tangent = (
        numerator_tangent
        - barycentric[:, :, :, None] * row_mass_tangent[:, :, None, :]
    ) / row_mass[:, :, None, None]
    quotient = uniform[:, :, None] * coupling / row_mass[:, :, None]
    column_mass = tf.reduce_sum(quotient, axis=1)
    column_residual = column_mass - weights
    column_tv = 0.5 * tf.reduce_sum(tf.abs(column_residual), axis=1)
    row_valid = (
        tf.reduce_all(tf.math.is_finite(row_mass), axis=1)
        & tf.reduce_all(row_mass > tiny, axis=1)
        & tf.reduce_all(tf.math.is_finite(barycentric), axis=[1, 2])
        & tf.reduce_all(tf.math.is_finite(barycentric_tangent), axis=[1, 2, 3])
        & (column_tv <= tf.cast(1.0e-4, particles.dtype))
    )
    return {
        "particles": barycentric,
        "particles_tangent": barycentric_tangent,
        "minimum_row_mass": tf.reduce_min(row_mass, axis=1),
        "maximum_raw_row_residual": tf.reduce_max(
            tf.abs(row_mass - uniform), axis=1
        ),
        "maximum_post_quotient_column_residual": tf.reduce_max(
            tf.abs(column_residual), axis=1
        ),
        "post_quotient_column_tv_error": column_tv,
        "marginal_valid": row_valid,
    }


def _sinkhorn_barycentric_batch_value(
    particles: Tensor,
    weights: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
) -> dict[str, Tensor]:
    """Value-only Sinkhorn row quotient without any tangent allocation."""

    if epsilon <= 0.0 or sinkhorn_steps <= 0 or balance_steps < 0:
        raise ValueError("epsilon and Sinkhorn counts must be valid")
    deltas = particles[:, :, None, :] - particles[:, None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=3)
    cost_scale = tf.maximum(
        tf.reduce_mean(cost, axis=[1, 2]), tf.cast(1.0e-3, particles.dtype)
    )
    kernel = tf.exp(
        -cost / (cost_scale[:, None, None] * tf.cast(epsilon, particles.dtype))
    )
    batch_size = tf.shape(particles)[0]
    particle_count = tf.shape(particles)[1]
    uniform = tf.fill(
        [batch_size, particle_count],
        tf.cast(1.0, particles.dtype) / tf.cast(particle_count, particles.dtype),
    )
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    tiny = tf.cast(1.0e-7, particles.dtype)

    def body(index, left_value, right_value):
        left_new = uniform / (
            tf.einsum("bij,bj->bi", kernel, right_value) + tiny
        )
        right_new = weights / (
            tf.einsum("bij,bi->bj", kernel, left_new) + tiny
        )
        return index + 1, left_new, right_new

    _, left, right = tf.while_loop(
        lambda index, *_: index < tf.cast(sinkhorn_steps + balance_steps, tf.int32),
        body,
        (tf.zeros([], tf.int32), left, right),
        parallel_iterations=1,
    )
    coupling = left[:, :, None] * kernel * right[:, None, :]
    row_mass = tf.reduce_sum(coupling, axis=2)
    barycentric = tf.einsum("bij,bjd->bid", coupling, particles) / row_mass[:, :, None]
    quotient = uniform[:, :, None] * coupling / row_mass[:, :, None]
    column_residual = tf.reduce_sum(quotient, axis=1) - weights
    column_tv = 0.5 * tf.reduce_sum(tf.abs(column_residual), axis=1)
    valid = (
        tf.reduce_all(tf.math.is_finite(row_mass), axis=1)
        & tf.reduce_all(row_mass > tiny, axis=1)
        & tf.reduce_all(tf.math.is_finite(barycentric), axis=[1, 2])
        & (column_tv <= tf.cast(1.0e-4, particles.dtype))
    )
    return {
        "particles": barycentric,
        "minimum_row_mass": tf.reduce_min(row_mass, axis=1),
        "maximum_raw_row_residual": tf.reduce_max(
            tf.abs(row_mass - uniform), axis=1
        ),
        "maximum_post_quotient_column_residual": tf.reduce_max(
            tf.abs(column_residual), axis=1
        ),
        "post_quotient_column_tv_error": column_tv,
        "marginal_valid": valid,
    }


def _restore_cloud_batch_value(
    particles: Tensor,
    weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> dict[str, Tensor]:
    transport = _sinkhorn_barycentric_batch_value(
        particles,
        weights,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
    )
    barycentric = transport["particles"]
    batch_size = tf.shape(particles)[0]
    particle_count = tf.shape(particles)[1]
    dimension = tf.shape(particles)[2]
    target_mean = tf.reduce_sum(weights[:, :, None] * particles, axis=1)
    centered_source = particles - target_mean[:, None, :]
    target_covariance = tf.einsum(
        "bn,bni,bnj->bij", weights, centered_source, centered_source
    )
    transported_mean = tf.reduce_mean(barycentric, axis=1)
    centered_transported = barycentric - transported_mean[:, None, :]
    transported_covariance = tf.einsum(
        "bni,bnj->bij", centered_transported, centered_transported
    ) / tf.cast(particle_count, particles.dtype)
    covariance_gap = _sym(target_covariance - transported_covariance)
    minimum_gap_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(covariance_gap), axis=1
    )
    gap_valid = tf.math.is_finite(minimum_gap_eigenvalue) & (
        minimum_gap_eigenvalue + tf.cast(ridge, particles.dtype) > 0.0
    )
    pre_valid = transport["marginal_valid"] & gap_valid
    safe_barycentric = tf.where(
        pre_valid[:, None, None],
        barycentric,
        tf.broadcast_to(target_mean[:, None, :], tf.shape(barycentric)),
    )
    design_batch = tf.broadcast_to(
        design[None, :, :] if design.shape.rank == 2 else design,
        [batch_size, particle_count, dimension],
    )
    forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        particles,
        weights,
        safe_barycentric,
        design_batch,
        tf.fill([batch_size], tf.cast(ridge, particles.dtype)),
    )
    valid = pre_valid & forward["finite"] & forward["factor_diagonal_positive"]
    return {
        "particles": forward["particles"],
        "valid": valid,
        "mean_residual": tf.reduce_max(tf.abs(forward["mean_residual"]), axis=1),
        "minimum_gap_eigenvalue": minimum_gap_eigenvalue,
        "minimum_row_mass": transport["minimum_row_mass"],
        "maximum_raw_row_residual": transport["maximum_raw_row_residual"],
        "maximum_post_quotient_column_residual": transport[
            "maximum_post_quotient_column_residual"
        ],
    }


def _restore_cloud_batch_jvp(
    particles: Tensor,
    weights: Tensor,
    particle_tangent: Tensor,
    weight_tangent: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> dict[str, Tensor]:
    transport = _sinkhorn_barycentric_batch_jvp(
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        epsilon=epsilon,
        sinkhorn_steps=sinkhorn_steps,
        balance_steps=balance_steps,
    )
    barycentric = transport["particles"]
    barycentric_tangent = transport["particles_tangent"]
    batch_size = tf.shape(particles)[0]
    particle_count = tf.shape(particles)[1]
    dimension = tf.shape(particles)[2]
    parameter_count = tf.shape(particle_tangent)[-1]
    target_mean, target_cov, target_mean_tangent, _ = _weighted_moments_jvp(
        particles, weights, particle_tangent, weight_tangent
    )
    current_mean = tf.reduce_mean(barycentric, axis=1)
    centered = barycentric - current_mean[:, None, :]
    current_cov = tf.einsum("bni,bnj->bij", centered, centered) / tf.cast(
        particle_count, particles.dtype
    )
    covariance_gap = _sym(target_cov - current_cov)
    minimum_gap_eigenvalue = tf.reduce_min(
        tf.linalg.eigvalsh(covariance_gap), axis=1
    )
    gap_valid = tf.math.is_finite(minimum_gap_eigenvalue) & (
        minimum_gap_eigenvalue + tf.cast(ridge, particles.dtype) > 0.0
    )
    pre_valid = transport["marginal_valid"] & gap_valid
    safe_barycentric = tf.where(
        pre_valid[:, None, None],
        barycentric,
        tf.broadcast_to(target_mean[:, None, :], tf.shape(barycentric)),
    )
    safe_barycentric_tangent = tf.where(
        pre_valid[:, None, None, None],
        barycentric_tangent,
        tf.broadcast_to(
            target_mean_tangent[:, None, :, :], tf.shape(barycentric_tangent)
        ),
    )
    design_batch = tf.broadcast_to(
        design[None, :, :] if design.shape.rank == 2 else design,
        [batch_size, particle_count, dimension],
    )
    ridge_batch = tf.fill([batch_size], tf.cast(ridge, particles.dtype))
    forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        particles, weights, safe_barycentric, design_batch, ridge_batch
    )

    # The reset JVP accepts one direction per batch member.  Flattening B*p is
    # tensor batching over directions, not iteration over posterior samples.
    source_flat = tf.reshape(
        tf.broadcast_to(
            particles[:, None, :, :],
            [batch_size, parameter_count, particle_count, dimension],
        ),
        [-1, particle_count, dimension],
    )
    weights_flat = tf.reshape(
        tf.broadcast_to(
            weights[:, None, :], [batch_size, parameter_count, particle_count]
        ),
        [-1, particle_count],
    )
    transported_flat = tf.reshape(
        tf.broadcast_to(
            safe_barycentric[:, None, :, :],
            [batch_size, parameter_count, particle_count, dimension],
        ),
        [-1, particle_count, dimension],
    )
    design_flat = tf.reshape(
        tf.broadcast_to(
            design_batch[:, None, :, :],
            [batch_size, parameter_count, particle_count, dimension],
        ),
        [-1, particle_count, dimension],
    )
    ridge_flat = tf.reshape(
        tf.broadcast_to(ridge_batch[:, None], [batch_size, parameter_count]), [-1]
    )
    flat_forward = reset._contract_e_chol_cloud_forward_core(  # noqa: SLF001
        source_flat, weights_flat, transported_flat, design_flat, ridge_flat
    )
    flat_jvp = reset._contract_e_chol_cloud_jvp_from_forward_core(  # noqa: SLF001
        flat_forward,
        source_flat,
        weights_flat,
        transported_flat,
        design_flat,
        ridge_flat,
        tf.reshape(
            tf.transpose(particle_tangent, [0, 3, 1, 2]),
            [-1, particle_count, dimension],
        ),
        tf.reshape(
            tf.transpose(weight_tangent, [0, 2, 1]), [-1, particle_count]
        ),
        tf.reshape(
            tf.transpose(safe_barycentric_tangent, [0, 3, 1, 2]),
            [-1, particle_count, dimension],
        ),
        tf.zeros_like(design_flat),
        tf.zeros_like(ridge_flat),
    )["particles"]
    restored_tangent = tf.transpose(
        tf.reshape(
            flat_jvp,
            [batch_size, parameter_count, particle_count, dimension],
        ),
        [0, 2, 3, 1],
    )
    reset_valid = (
        pre_valid
        & forward["finite"]
        & forward["factor_diagonal_positive"]
        & tf.reduce_all(
            tf.reshape(
                flat_forward["finite"], [batch_size, parameter_count]
            ),
            axis=1,
        )
        & tf.reduce_all(
            tf.reshape(
                flat_forward["factor_diagonal_positive"],
                [batch_size, parameter_count],
            ),
            axis=1,
        )
        & tf.reduce_all(tf.math.is_finite(restored_tangent), axis=[1, 2, 3])
    )
    return {
        "particles": forward["particles"],
        "particles_tangent": restored_tangent,
        "valid": reset_valid,
        "mean_residual": tf.reduce_max(tf.abs(forward["mean_residual"]), axis=1),
        "minimum_gap_eigenvalue": minimum_gap_eigenvalue,
        "minimum_row_mass": transport["minimum_row_mass"],
        "maximum_raw_row_residual": transport["maximum_raw_row_residual"],
        "maximum_post_quotient_column_residual": transport[
            "maximum_post_quotient_column_residual"
        ],
        "post_quotient_column_tv_error": transport[
            "post_quotient_column_tv_error"
        ],
        "marginal_valid": transport["marginal_valid"],
    }


def _shape_iteration_batch_jvp(
    points: Tensor,
    points_tangent: Tensor,
    target_skew: Tensor,
    target_kurtosis: Tensor,
    target_skew_tangent: Tensor,
    target_kurtosis_tangent: Tensor,
    *,
    strength: float,
    floor: float,
) -> tuple[Tensor, Tensor]:
    mean, covariance, mean_tangent, covariance_tangent = _uniform_moments_jvp(
        points, points_tangent
    )
    chol = tf.linalg.cholesky(covariance)
    chol_tangent = _cholesky_jvp(chol, covariance_tangent)
    centered = points - mean[:, None, :]
    centered_tangent = points_tangent - mean_tangent[:, None, :, :]
    standardized = _right_solve(chol, centered)
    standardized_tangent = _right_solve_jvp(
        chol, chol_tangent, centered, centered_tangent
    )
    m3 = tf.reduce_mean(tf.pow(standardized, 3.0), axis=1)
    m4 = tf.reduce_mean(tf.pow(standardized, 4.0), axis=1)
    m3_tangent = tf.reduce_mean(
        3.0 * tf.pow(standardized[:, :, :, None], 2.0) * standardized_tangent,
        axis=1,
    )
    m4_tangent = tf.reduce_mean(
        4.0 * tf.pow(standardized[:, :, :, None], 3.0) * standardized_tangent,
        axis=1,
    )
    residual3 = target_skew - m3
    residual4 = target_kurtosis - m4
    residual3_tangent = target_skew_tangent - m3_tangent
    residual4_tangent = target_kurtosis_tangent - m4_tangent
    direction3 = tf.square(standardized) - 1.0 - m3[:, None, :] * standardized
    direction4 = (
        tf.pow(standardized, 3.0)
        - m3[:, None, :]
        - m4[:, None, :] * standardized
    )
    direction3_tangent = (
        2.0 * standardized[:, :, :, None] * standardized_tangent
        - m3_tangent[:, None, :, :] * standardized[:, :, :, None]
        - m3[:, None, :, None] * standardized_tangent
    )
    direction4_tangent = (
        3.0 * tf.square(standardized)[:, :, :, None] * standardized_tangent
        - m3_tangent[:, None, :, :]
        - m4_tangent[:, None, :, :] * standardized[:, :, :, None]
        - m4[:, None, :, None] * standardized_tangent
    )
    j33 = tf.reduce_mean(3.0 * tf.square(standardized) * direction3, axis=1)
    j34 = tf.reduce_mean(3.0 * tf.square(standardized) * direction4, axis=1)
    j43 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction3, axis=1)
    j44 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction4, axis=1)
    j33_tangent = tf.reduce_mean(
        6.0 * standardized[:, :, :, None] * standardized_tangent * direction3[:, :, :, None]
        + 3.0 * tf.square(standardized)[:, :, :, None] * direction3_tangent,
        axis=1,
    )
    j34_tangent = tf.reduce_mean(
        6.0 * standardized[:, :, :, None] * standardized_tangent * direction4[:, :, :, None]
        + 3.0 * tf.square(standardized)[:, :, :, None] * direction4_tangent,
        axis=1,
    )
    j43_tangent = tf.reduce_mean(
        12.0 * tf.square(standardized)[:, :, :, None] * standardized_tangent * direction3[:, :, :, None]
        + 4.0 * tf.pow(standardized, 3.0)[:, :, :, None] * direction3_tangent,
        axis=1,
    )
    j44_tangent = tf.reduce_mean(
        12.0 * tf.square(standardized)[:, :, :, None] * standardized_tangent * direction4[:, :, :, None]
        + 4.0 * tf.pow(standardized, 3.0)[:, :, :, None] * direction4_tangent,
        axis=1,
    )
    jacobian = tf.stack(
        [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
        axis=-2,
    )
    jacobian_tangent = tf.stack(
        [
            tf.stack([j33_tangent, j34_tangent], axis=-2),
            tf.stack([j43_tangent, j44_tangent], axis=-2),
        ],
        axis=-3,
    )
    residual = tf.stack([residual3, residual4], axis=-1)
    residual_tangent = tf.stack(
        [residual3_tangent, residual4_tangent], axis=-2
    )
    normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
    normal += tf.cast(floor, points.dtype) * tf.eye(
        2, batch_shape=[tf.shape(points)[0], tf.shape(points)[2]], dtype=points.dtype
    )
    rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
    coefficient = tf.cast(strength, points.dtype) * tf.linalg.solve(
        normal, rhs[:, :, :, None]
    )[:, :, :, 0]
    normal_tangent = (
        tf.einsum("bdaip,bdaj->bdijp", jacobian_tangent, jacobian)
        + tf.einsum("bdai,bdajp->bdijp", jacobian, jacobian_tangent)
    )
    rhs_tangent = (
        tf.einsum("bdaip,bda->bdip", jacobian_tangent, residual)
        + tf.einsum("bdai,bdap->bdip", jacobian, residual_tangent)
    )
    if strength > 0.0:
        solve_rhs = rhs_tangent - tf.einsum(
            "bdijp,bdj->bdip",
            normal_tangent,
            coefficient / tf.cast(strength, points.dtype),
        )
    else:
        solve_rhs = tf.zeros_like(rhs_tangent)
    batch_size = tf.shape(points)[0]
    dimension = tf.shape(points)[2]
    parameter_count = tf.shape(points_tangent)[-1]
    normal_flat = tf.reshape(
        tf.broadcast_to(
            normal[:, :, None, :, :],
            [batch_size, dimension, parameter_count, 2, 2],
        ),
        [-1, 2, 2],
    )
    rhs_flat = tf.reshape(
        tf.transpose(solve_rhs, [0, 1, 3, 2]), [-1, 2, 1]
    )
    coefficient_tangent = tf.cast(strength, points.dtype) * tf.transpose(
        tf.reshape(
            tf.linalg.solve(normal_flat, rhs_flat)[:, :, 0],
            [batch_size, dimension, parameter_count, 2],
        ),
        [0, 1, 3, 2],
    )
    corrected = (
        standardized
        + direction3 * coefficient[:, None, :, 0]
        + direction4 * coefficient[:, None, :, 1]
    )
    corrected_tangent = (
        standardized_tangent
        + direction3_tangent * coefficient[:, None, :, 0, None]
        + direction3[:, :, :, None] * coefficient_tangent[:, None, :, 0, :]
        + direction4_tangent * coefficient[:, None, :, 1, None]
        + direction4[:, :, :, None] * coefficient_tangent[:, None, :, 1, :]
    )
    new_mean, new_covariance, new_mean_tangent, new_covariance_tangent = (
        _uniform_moments_jvp(corrected, corrected_tangent)
    )
    new_chol = tf.linalg.cholesky(new_covariance)
    new_chol_tangent = _cholesky_jvp(new_chol, new_covariance_tangent)
    return (
        _right_solve(new_chol, corrected - new_mean[:, None, :]),
        _right_solve_jvp(
            new_chol,
            new_chol_tangent,
            corrected - new_mean[:, None, :],
            corrected_tangent - new_mean_tangent[:, None, :, :],
        ),
    )


def _higher_moment_batch_jvp(
    source: Tensor,
    weights: Tensor,
    source_tangent: Tensor,
    weights_tangent: Tensor,
    points: Tensor,
    points_tangent: Tensor,
    *,
    correction_steps: int,
    strength: float,
    floor: float,
) -> dict[str, Tensor]:
    if correction_steps < 0 or strength < 0.0 or floor <= 0.0:
        raise ValueError("invalid higher-moment correction controls")
    mean, covariance, mean_tangent, covariance_tangent = _weighted_moments_jvp(
        source, weights, source_tangent, weights_tangent
    )
    target_chol = tf.linalg.cholesky(covariance)
    target_chol_tangent = _cholesky_jvp(target_chol, covariance_tangent)
    centered = source - mean[:, None, :]
    centered_tangent = source_tangent - mean_tangent[:, None, :, :]
    standardized_source = _right_solve(target_chol, centered)
    standardized_source_tangent = _right_solve_jvp(
        target_chol, target_chol_tangent, centered, centered_tangent
    )
    target_skew = tf.reduce_sum(
        weights[:, :, None] * tf.pow(standardized_source, 3.0), axis=1
    )
    target_kurtosis = tf.reduce_sum(
        weights[:, :, None] * tf.pow(standardized_source, 4.0), axis=1
    )
    target_skew_tangent = tf.reduce_sum(
        weights_tangent[:, :, None, :]
        * tf.pow(standardized_source[:, :, :, None], 3.0)
        + weights[:, :, None, None]
        * 3.0
        * tf.pow(standardized_source[:, :, :, None], 2.0)
        * standardized_source_tangent,
        axis=1,
    )
    target_kurtosis_tangent = tf.reduce_sum(
        weights_tangent[:, :, None, :]
        * tf.pow(standardized_source[:, :, :, None], 4.0)
        + weights[:, :, None, None]
        * 4.0
        * tf.pow(standardized_source[:, :, :, None], 3.0)
        * standardized_source_tangent,
        axis=1,
    )
    current_mean, current_covariance, current_mean_tangent, current_covariance_tangent = (
        _uniform_moments_jvp(points, points_tangent)
    )
    current_chol = tf.linalg.cholesky(current_covariance)
    current_chol_tangent = _cholesky_jvp(
        current_chol, current_covariance_tangent
    )
    standardized = _right_solve(
        current_chol, points - current_mean[:, None, :]
    )
    standardized_tangent = _right_solve_jvp(
        current_chol,
        current_chol_tangent,
        points - current_mean[:, None, :],
        points_tangent - current_mean_tangent[:, None, :, :],
    )
    for _ in range(correction_steps):
        standardized, standardized_tangent = _shape_iteration_batch_jvp(
            standardized,
            standardized_tangent,
            target_skew,
            target_kurtosis,
            target_skew_tangent,
            target_kurtosis_tangent,
            strength=strength,
            floor=floor,
        )
    output = mean[:, None, :] + tf.linalg.matmul(
        standardized, target_chol, transpose_b=True
    )
    output_tangent = (
        mean_tangent[:, None, :, :]
        + tf.einsum("bnip,bji->bnjp", standardized_tangent, target_chol)
        + tf.einsum("bni,bjip->bnjp", standardized, target_chol_tangent)
    )
    skew_residual = target_skew - tf.reduce_mean(
        tf.pow(standardized, 3.0), axis=1
    )
    kurtosis_residual = target_kurtosis - tf.reduce_mean(
        tf.pow(standardized, 4.0), axis=1
    )
    valid = tf.reduce_all(tf.math.is_finite(output), axis=[1, 2]) & tf.reduce_all(
        tf.math.is_finite(output_tangent), axis=[1, 2, 3]
    )
    return {
        "particles": output,
        "particles_tangent": output_tangent,
        "skew_residual": skew_residual,
        "kurtosis_residual": kurtosis_residual,
        "valid": valid,
    }


def _higher_moment_batch_value(
    source: Tensor,
    weights: Tensor,
    points: Tensor,
    *,
    correction_steps: int,
    strength: float,
    floor: float,
) -> dict[str, Tensor]:
    """Selected diagonal shape correction without derivative tensors."""

    if correction_steps < 0 or strength < 0.0 or floor <= 0.0:
        raise ValueError("invalid higher-moment correction controls")

    def moments(values, values_weights=None):
        if values_weights is None:
            mean_value = tf.reduce_mean(values, axis=1)
            centered_value = values - mean_value[:, None, :]
            covariance_value = tf.einsum(
                "bni,bnj->bij", centered_value, centered_value
            ) / tf.cast(tf.shape(values)[1], values.dtype)
        else:
            mean_value = tf.reduce_sum(
                values_weights[:, :, None] * values, axis=1
            )
            centered_value = values - mean_value[:, None, :]
            covariance_value = tf.einsum(
                "bn,bni,bnj->bij",
                values_weights,
                centered_value,
                centered_value,
            )
        return mean_value, _sym(covariance_value)

    mean, covariance = moments(source, weights)
    target_chol = tf.linalg.cholesky(covariance)
    source_standardized = _right_solve(
        target_chol, source - mean[:, None, :]
    )
    target_skew = tf.reduce_sum(
        weights[:, :, None] * tf.pow(source_standardized, 3.0), axis=1
    )
    target_kurtosis = tf.reduce_sum(
        weights[:, :, None] * tf.pow(source_standardized, 4.0), axis=1
    )
    point_mean, point_covariance = moments(points)
    standardized = _right_solve(
        tf.linalg.cholesky(point_covariance), points - point_mean[:, None, :]
    )
    for _ in range(correction_steps):
        m3 = tf.reduce_mean(tf.pow(standardized, 3.0), axis=1)
        m4 = tf.reduce_mean(tf.pow(standardized, 4.0), axis=1)
        residual3 = target_skew - m3
        residual4 = target_kurtosis - m4
        direction3 = tf.square(standardized) - 1.0 - m3[:, None, :] * standardized
        direction4 = (
            tf.pow(standardized, 3.0)
            - m3[:, None, :]
            - m4[:, None, :] * standardized
        )
        j33 = tf.reduce_mean(3.0 * tf.square(standardized) * direction3, axis=1)
        j34 = tf.reduce_mean(3.0 * tf.square(standardized) * direction4, axis=1)
        j43 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction3, axis=1)
        j44 = tf.reduce_mean(4.0 * tf.pow(standardized, 3.0) * direction4, axis=1)
        jacobian = tf.stack(
            [tf.stack([j33, j34], axis=-1), tf.stack([j43, j44], axis=-1)],
            axis=-2,
        )
        residual = tf.stack([residual3, residual4], axis=-1)
        normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
        normal += tf.cast(floor, points.dtype) * tf.eye(
            2,
            batch_shape=[tf.shape(points)[0], tf.shape(points)[2]],
            dtype=points.dtype,
        )
        rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
        coefficient = tf.cast(strength, points.dtype) * tf.linalg.solve(
            normal, rhs[:, :, :, None]
        )[:, :, :, 0]
        corrected = (
            standardized
            + direction3 * coefficient[:, None, :, 0]
            + direction4 * coefficient[:, None, :, 1]
        )
        corrected_mean, corrected_covariance = moments(corrected)
        standardized = _right_solve(
            tf.linalg.cholesky(corrected_covariance),
            corrected - corrected_mean[:, None, :],
        )
    output = mean[:, None, :] + tf.linalg.matmul(
        standardized, target_chol, transpose_b=True
    )
    skew_residual = target_skew - tf.reduce_mean(
        tf.pow(standardized, 3.0), axis=1
    )
    kurtosis_residual = target_kurtosis - tf.reduce_mean(
        tf.pow(standardized, 4.0), axis=1
    )
    return {
        "particles": output,
        "skew_residual": skew_residual,
        "kurtosis_residual": kurtosis_residual,
        "valid": tf.reduce_all(tf.math.is_finite(output), axis=[1, 2]),
    }


def batch_finite_value(
    adapter: BatchCandidateModelAdapter,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    design: Tensor,
    *,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    transition_before_first_observation: bool = True,
    higher_moment_correction_steps: int = 0,
    higher_moment_strength: float = 0.0,
    higher_moment_floor: float = 1.0e-6,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Evaluate only the finite GenUT scalar for a leading posterior batch."""

    theta = tf.convert_to_tensor(theta, dtype=initial_noise.dtype)
    if theta.shape.rank != 2:
        raise ValueError("batch GenUT theta must have shape [batch, parameter]")
    batch_static = theta.shape[0]
    particle_static = initial_noise.shape[0]
    if batch_static is None or particle_static is None:
        raise ValueError("batch GenUT XLA core requires static batch and particle counts")
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    particles = adapter.initial_value(theta, initial_noise)
    particles = tf.ensure_shape(
        particles,
        [batch_static, particle_static, adapter.state_dimension],
    )
    process_noise = tf.convert_to_tensor(process_noise, dtype=theta.dtype)
    design = tf.convert_to_tensor(design, dtype=theta.dtype)
    batch_size = tf.shape(theta)[0]
    particle_count = tf.shape(particles)[1]
    weights = tf.fill(
        [batch_static, particle_static],
        tf.cast(1.0, theta.dtype) / tf.cast(particle_count, theta.dtype),
    )
    total = tf.zeros([batch_static], theta.dtype)
    valid = tf.ones([batch_static], tf.bool)
    max_mean = tf.zeros([batch_static], theta.dtype)
    max_row = tf.zeros([batch_static], theta.dtype)
    max_column = tf.zeros([batch_static], theta.dtype)
    min_gap = tf.fill([batch_static], tf.constant(float("inf"), theta.dtype))
    max_skew = tf.zeros([batch_static], theta.dtype)
    max_kurtosis = tf.zeros([batch_static], theta.dtype)
    horizon = tf.shape(observations)[0]

    def body(
        time_index,
        particles_value,
        weights_value,
        total_value,
        valid_value,
        max_mean_value,
        max_row_value,
        max_column_value,
        min_gap_value,
        max_skew_value,
        max_kurtosis_value,
    ):
        transition = tf.logical_or(
            tf.constant(transition_before_first_observation),
            tf.not_equal(time_index, tf.constant(0, tf.int32)),
        )
        next_particles = tf.cond(
            transition,
            lambda: adapter.transition_value(
                theta, particles_value, process_noise[time_index], time_index
            ),
            lambda: particles_value,
        )
        log_likelihood = adapter.observation_value(
            theta, next_particles, observations[time_index], time_index
        )
        log_weights = tf.math.log(weights_value) + log_likelihood
        increment = tf.reduce_logsumexp(log_weights, axis=1)
        normalized_weights = tf.exp(log_weights - increment[:, None])
        stage_valid = (
            tf.reduce_all(tf.math.is_finite(next_particles), axis=[1, 2])
            & tf.reduce_all(tf.math.is_finite(log_likelihood), axis=1)
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(normalized_weights), axis=1)
            & (tf.abs(tf.reduce_sum(normalized_weights, axis=1) - 1.0) <= tf.cast(1.0e-4, theta.dtype))
        )
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored = _restore_cloud_batch_value(
            next_particles,
            normalized_weights,
            current_design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
        )
        higher = _higher_moment_batch_value(
            next_particles,
            normalized_weights,
            restored["particles"],
            correction_steps=higher_moment_correction_steps,
            strength=higher_moment_strength,
            floor=higher_moment_floor,
        )
        higher_particles = tf.ensure_shape(
            higher["particles"],
            [batch_static, particle_static, adapter.state_dimension],
        )
        step_valid = stage_valid & restored["valid"] & higher["valid"]
        uniform_weights = tf.fill(
            [batch_static, particle_static],
            tf.cast(1.0, theta.dtype) / tf.cast(particle_count, theta.dtype),
        )
        return (
            time_index + 1,
            tf.where(step_valid[:, None, None], higher_particles, particles_value),
            uniform_weights,
            total_value + tf.where(stage_valid, increment, tf.zeros_like(increment)),
            valid_value & step_valid,
            tf.maximum(max_mean_value, restored["mean_residual"]),
            tf.maximum(max_row_value, restored["maximum_raw_row_residual"]),
            tf.maximum(max_column_value, restored["maximum_post_quotient_column_residual"]),
            tf.minimum(min_gap_value, restored["minimum_gap_eigenvalue"]),
            tf.maximum(max_skew_value, tf.reduce_max(tf.abs(higher["skew_residual"]), axis=1)),
            tf.maximum(max_kurtosis_value, tf.reduce_max(tf.abs(higher["kurtosis_residual"]), axis=1)),
        )

    (
        _,
        _particles,
        _weights,
        total,
        valid,
        max_mean,
        max_row,
        max_column,
        min_gap,
        max_skew,
        max_kurtosis,
    ) = tf.while_loop(
        lambda time_index, *_: time_index < horizon,
        body,
        (
            tf.zeros([], tf.int32),
            particles,
            weights,
            total,
            valid,
            max_mean,
            max_row,
            max_column,
            min_gap,
            max_skew,
            max_kurtosis,
        ),
        parallel_iterations=1,
    )
    diagnostics = {
        "program_valid": valid,
        "max_mean_residual": max_mean,
        "max_row_residual": max_row,
        "max_col_residual": max_column,
        "minimum_covariance_gap_eigenvalue": min_gap,
        "maximum_skew_residual": max_skew,
        "maximum_kurtosis_residual": max_kurtosis,
    }
    return tf.where(valid, total, tf.constant(float("nan"), theta.dtype)), diagnostics


def batch_finite_value_score(
    adapter: BatchCandidateModelAdapter,
    theta: Tensor,
    observations: Tensor,
    initial_noise: Tensor,
    process_noise: Tensor,
    design: Tensor,
    *,
    epsilon: float = 2.0,
    sinkhorn_steps: int = 8,
    balance_steps: int = 8,
    ridge: float = 1.0e-5,
    transition_before_first_observation: bool = True,
    higher_moment_correction_steps: int = 0,
    higher_moment_strength: float = 0.0,
    higher_moment_floor: float = 1.0e-6,
) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
    """Evaluate a genuine leading-batch finite GenUT value and total score."""

    theta = tf.convert_to_tensor(theta, dtype=initial_noise.dtype)
    if theta.shape.rank != 2:
        raise ValueError("batch GenUT theta must have shape [batch, parameter]")
    batch_static = theta.shape[0]
    particle_static = initial_noise.shape[0]
    if batch_static is None or particle_static is None:
        raise ValueError("batch GenUT XLA core requires static batch and particle counts")
    observations = tf.convert_to_tensor(observations, dtype=theta.dtype)
    initial_noise = tf.convert_to_tensor(initial_noise, dtype=theta.dtype)
    process_noise = tf.convert_to_tensor(process_noise, dtype=theta.dtype)
    design = tf.convert_to_tensor(design, dtype=theta.dtype)
    particles = adapter.initial_value(theta, initial_noise)
    particles = tf.ensure_shape(
        particles,
        [batch_static, particle_static, adapter.state_dimension],
    )
    particle_tangent = adapter.initial_tangent(theta, initial_noise)
    particle_tangent = tf.ensure_shape(
        particle_tangent,
        [
            batch_static,
            particle_static,
            adapter.state_dimension,
            adapter.parameter_count,
        ],
    )
    batch_size = tf.shape(theta)[0]
    particle_count = tf.shape(particles)[1]
    parameter_count = adapter.parameter_count
    weights = tf.fill(
        [batch_static, particle_static],
        tf.cast(1.0, theta.dtype) / tf.cast(particle_count, theta.dtype),
    )
    weight_tangent = tf.zeros(
        [batch_static, particle_static, parameter_count], theta.dtype
    )
    total = tf.zeros([batch_static], theta.dtype)
    score = tf.zeros([batch_static, parameter_count], theta.dtype)
    valid = tf.ones([batch_static], tf.bool)
    maximum_mean_residual = tf.zeros([batch_static], theta.dtype)
    maximum_row_residual = tf.zeros([batch_static], theta.dtype)
    maximum_column_residual = tf.zeros([batch_static], theta.dtype)
    minimum_gap_eigenvalue = tf.fill(
        [batch_static], tf.constant(float("inf"), theta.dtype)
    )
    maximum_skew_residual = tf.zeros([batch_static], theta.dtype)
    maximum_kurtosis_residual = tf.zeros([batch_static], theta.dtype)
    horizon = tf.shape(observations)[0]

    def body(
        time_index,
        particles_value,
        particles_jvp,
        weights_value,
        weights_jvp,
        total_value,
        score_value,
        valid_value,
        max_mean,
        max_row,
        max_column,
        min_gap,
        max_skew,
        max_kurtosis,
    ):
        transition = tf.logical_or(
            tf.constant(transition_before_first_observation),
            tf.not_equal(time_index, tf.constant(0, tf.int32)),
        )
        noise = process_noise[time_index]
        next_particles = tf.cond(
            transition,
            lambda: adapter.transition_value(
                theta, particles_value, noise, time_index
            ),
            lambda: particles_value,
        )
        next_tangent = tf.cond(
            transition,
            lambda: adapter.transition_tangent(
                theta, particles_value, noise, particles_jvp, time_index
            ),
            lambda: particles_jvp,
        )
        observation = observations[time_index]
        log_likelihood = adapter.observation_value(
            theta, next_particles, observation, time_index
        )
        log_likelihood_tangent = adapter.observation_tangent(
            theta, next_particles, next_tangent, observation, time_index
        )
        log_weights = tf.math.log(weights_value) + log_likelihood
        log_weight_tangent = (
            weights_jvp / weights_value[:, :, None] + log_likelihood_tangent
        )
        increment = tf.reduce_logsumexp(log_weights, axis=1)
        normalized_weights = tf.exp(log_weights - increment[:, None])
        increment_tangent = tf.reduce_sum(
            normalized_weights[:, :, None] * log_weight_tangent, axis=1
        )
        normalized_weight_tangent = normalized_weights[:, :, None] * (
            log_weight_tangent - increment_tangent[:, None, :]
        )
        stage_valid = (
            tf.reduce_all(tf.math.is_finite(next_particles), axis=[1, 2])
            & tf.reduce_all(tf.math.is_finite(next_tangent), axis=[1, 2, 3])
            & tf.reduce_all(tf.math.is_finite(log_likelihood), axis=1)
            & tf.reduce_all(tf.math.is_finite(log_likelihood_tangent), axis=[1, 2])
            & tf.math.is_finite(increment)
            & tf.reduce_all(tf.math.is_finite(normalized_weights), axis=1)
            & tf.reduce_all(
                tf.math.is_finite(normalized_weight_tangent), axis=[1, 2]
            )
            & (tf.abs(tf.reduce_sum(normalized_weights, axis=1) - 1.0) <= tf.cast(1.0e-4, theta.dtype))
        )
        current_design = design if design.shape.rank == 2 else design[time_index]
        restored = _restore_cloud_batch_jvp(
            next_particles,
            normalized_weights,
            next_tangent,
            normalized_weight_tangent,
            current_design,
            epsilon=epsilon,
            sinkhorn_steps=sinkhorn_steps,
            balance_steps=balance_steps,
            ridge=ridge,
        )
        higher = _higher_moment_batch_jvp(
            next_particles,
            normalized_weights,
            next_tangent,
            normalized_weight_tangent,
            restored["particles"],
            restored["particles_tangent"],
            correction_steps=higher_moment_correction_steps,
            strength=higher_moment_strength,
            floor=higher_moment_floor,
        )
        higher_particles = tf.ensure_shape(
            higher["particles"],
            [batch_static, particle_static, adapter.state_dimension],
        )
        higher_tangent = tf.ensure_shape(
            higher["particles_tangent"],
            [
                batch_static,
                particle_static,
                adapter.state_dimension,
                parameter_count,
            ],
        )
        step_valid = stage_valid & restored["valid"] & higher["valid"]
        uniform_weights = tf.fill(
            [batch_static, particle_static],
            tf.cast(1.0, theta.dtype) / tf.cast(particle_count, theta.dtype),
        )
        return (
            time_index + 1,
            tf.where(step_valid[:, None, None], higher_particles, particles_value),
            tf.where(
                step_valid[:, None, None, None],
                higher_tangent,
                particles_jvp,
            ),
            uniform_weights,
            tf.zeros_like(weights_jvp),
            total_value + tf.where(stage_valid, increment, tf.zeros_like(increment)),
            score_value + tf.where(
                stage_valid[:, None], increment_tangent, tf.zeros_like(increment_tangent)
            ),
            valid_value & step_valid,
            tf.maximum(max_mean, restored["mean_residual"]),
            tf.maximum(max_row, restored["maximum_raw_row_residual"]),
            tf.maximum(max_column, restored["maximum_post_quotient_column_residual"]),
            tf.minimum(min_gap, restored["minimum_gap_eigenvalue"]),
            tf.maximum(max_skew, tf.reduce_max(tf.abs(higher["skew_residual"]), axis=1)),
            tf.maximum(
                max_kurtosis,
                tf.reduce_max(tf.abs(higher["kurtosis_residual"]), axis=1),
            ),
        )

    (
        _,
        _particles,
        _particle_tangent,
        _weights,
        _weight_tangent,
        total,
        score,
        valid,
        maximum_mean_residual,
        maximum_row_residual,
        maximum_column_residual,
        minimum_gap_eigenvalue,
        maximum_skew_residual,
        maximum_kurtosis_residual,
    ) = tf.while_loop(
        lambda time_index, *_: time_index < horizon,
        body,
        (
            tf.zeros([], tf.int32),
            particles,
            particle_tangent,
            weights,
            weight_tangent,
            total,
            score,
            valid,
            maximum_mean_residual,
            maximum_row_residual,
            maximum_column_residual,
            minimum_gap_eigenvalue,
            maximum_skew_residual,
            maximum_kurtosis_residual,
        ),
        parallel_iterations=1,
    )
    nan = tf.constant(float("nan"), theta.dtype)
    diagnostics = {
        "program_valid": valid,
        "max_mean_residual": maximum_mean_residual,
        "max_row_residual": maximum_row_residual,
        "max_col_residual": maximum_column_residual,
        "minimum_covariance_gap_eigenvalue": minimum_gap_eigenvalue,
        "maximum_skew_residual": maximum_skew_residual,
        "maximum_kurtosis_residual": maximum_kurtosis_residual,
    }
    return (
        tf.where(valid, total, nan),
        tf.where(valid[:, None], score, tf.fill(tf.shape(score), nan)),
        diagnostics,
    )


__all__ = [
    "BatchCandidateModelAdapter",
    "batch_finite_value",
    "batch_finite_value_score",
]
