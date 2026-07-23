"""Finite higher-moment correction for the opt-in Contract E candidate.

The map in this module is deliberately bounded and finite.  It is not a claim
that a nonconvex exact moment projection exists for every cloud.  The returned
JVP differentiates the complete executed map without TensorFlow autodiff.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_reset_tf as reset


def _sym(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _sym_tangent(value: tf.Tensor) -> tf.Tensor:
    return 0.5 * (value + tf.transpose(value, [1, 0, 2]))


def _right_solve(lower: tf.Tensor, rows: tf.Tensor) -> tf.Tensor:
    return tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(lower, tf.linalg.matrix_transpose(rows))
    )


def _right_solve_jvp(
    lower: tf.Tensor,
    lower_tangent: tf.Tensor,
    rows: tf.Tensor,
    rows_tangent: tf.Tensor,
) -> tf.Tensor:
    solved = _right_solve(lower, rows)
    rhs_tangent = rows_tangent - tf.einsum(
        "ni,jip->njp", solved, lower_tangent
    )
    batched_lower = tf.broadcast_to(
        lower[None, :, :],
        [tf.shape(rows_tangent)[-1], tf.shape(lower)[0], tf.shape(lower)[1]],
    )
    batched_rhs = tf.transpose(rhs_tangent, [2, 0, 1])
    solved_tangent = _right_solve(batched_lower, batched_rhs)
    return tf.transpose(solved_tangent, [1, 2, 0])


def _cholesky_jvp(chol: tf.Tensor, matrix_tangent: tf.Tensor) -> tf.Tensor:
    parameter_count = tf.shape(matrix_tangent)[-1]
    batched_chol = tf.broadcast_to(
        chol[None, :, :],
        [parameter_count, tf.shape(chol)[0], tf.shape(chol)[1]],
    )
    batched_tangent = tf.transpose(matrix_tangent, [2, 0, 1])
    return tf.transpose(
        reset._cholesky_jvp(batched_chol, batched_tangent), [1, 2, 0]
    )


def _weighted_moments_jvp(
    points: tf.Tensor,
    weights: tf.Tensor,
    points_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    mean_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * points[:, :, None]
        + weights[:, None, None] * points_tangent,
        axis=0,
    )
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance_tangent = tf.einsum(
        "np,ni,nj->ijp", weights_tangent, centered, centered
    )
    covariance_tangent += tf.einsum(
        "n,nip,nj->ijp", weights, centered_tangent, centered
    )
    covariance_tangent += tf.einsum(
        "n,ni,njp->ijp", weights, centered, centered_tangent
    )
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _uniform_moments_jvp(
    points: tf.Tensor, points_tangent: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    count = tf.cast(tf.shape(points)[0], points.dtype)
    mean = tf.reduce_mean(points, axis=0)
    mean_tangent = tf.reduce_mean(points_tangent, axis=0)
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    uniform = tf.ones_like(points[:, 0]) / count
    covariance = tf.einsum("n,ni,nj->ij", uniform, centered, centered)
    covariance_tangent = tf.einsum(
        "n,nip,nj->ijp", uniform, centered_tangent, centered
    ) + tf.einsum(
        "n,ni,njp->ijp", uniform, centered, centered_tangent
    )
    return mean, _sym(covariance), mean_tangent, _sym_tangent(covariance_tangent)


def _weighted_diag_moments_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    mean: tf.Tensor,
    mean_tangent: tf.Tensor,
    chol: tf.Tensor,
    chol_tangent: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    centered = source - mean[None, :]
    centered_tangent = source_tangent - mean_tangent[None, :, :]
    standardized = _right_solve(chol, centered)
    standardized_tangent = _right_solve_jvp(
        chol, chol_tangent, centered, centered_tangent
    )
    skew = tf.reduce_sum(weights[:, None] * tf.pow(standardized, 3.0), axis=0)
    kurt = tf.reduce_sum(weights[:, None] * tf.pow(standardized, 4.0), axis=0)
    skew_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * tf.pow(standardized[:, :, None], 3.0)
        + weights[:, None, None]
        * 3.0
        * tf.pow(standardized[:, :, None], 2.0)
        * standardized_tangent,
        axis=0,
    )
    kurt_tangent = tf.reduce_sum(
        weights_tangent[:, None, :] * tf.pow(standardized[:, :, None], 4.0)
        + weights[:, None, None]
        * 4.0
        * tf.pow(standardized[:, :, None], 3.0)
        * standardized_tangent,
        axis=0,
    )
    return skew, kurt, skew_tangent, kurt_tangent


def _shape_iteration_jvp(
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    target_skew: tf.Tensor,
    target_kurt: tf.Tensor,
    target_skew_tangent: tf.Tensor,
    target_kurt_tangent: tf.Tensor,
    *,
    strength: float,
    floor: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    mean, covariance, mean_tangent, covariance_tangent = _uniform_moments_jvp(
        points, points_tangent
    )
    chol = tf.linalg.cholesky(covariance)
    chol_tangent = _cholesky_jvp(chol, covariance_tangent)
    centered = points - mean[None, :]
    centered_tangent = points_tangent - mean_tangent[None, :, :]
    u = _right_solve(chol, centered)
    u_tangent = _right_solve_jvp(chol, chol_tangent, centered, centered_tangent)
    m3 = tf.reduce_mean(tf.pow(u, 3.0), axis=0)
    m4 = tf.reduce_mean(tf.pow(u, 4.0), axis=0)
    m3_tangent = tf.reduce_mean(3.0 * tf.pow(u[:, :, None], 2.0) * u_tangent, axis=0)
    m4_tangent = tf.reduce_mean(4.0 * tf.pow(u[:, :, None], 3.0) * u_tangent, axis=0)
    d3 = target_skew - m3
    d4 = target_kurt - m4
    d3_tangent = target_skew_tangent - m3_tangent
    d4_tangent = target_kurt_tangent - m4_tangent

    # Project the moment gradients away from the affine mean/scale tangent
    # space.  Unlike odd/even Hermite updates, these directions can change
    # skewness and kurtosis from a symmetric standardized cloud.
    direction3 = tf.square(u) - 1.0 - m3[None, :] * u
    direction4 = tf.pow(u, 3.0) - m3[None, :] - m4[None, :] * u
    direction3_tangent = (
        2.0 * u[:, :, None] * u_tangent
        - m3_tangent[None, :, :] * u[:, :, None]
        - m3[None, :, None] * u_tangent
    )
    direction4_tangent = (
        3.0 * tf.square(u)[:, :, None] * u_tangent
        - m3_tangent[None, :, :]
        - m4_tangent[None, :, :] * u[:, :, None]
        - m4[None, :, None] * u_tangent
    )

    j33 = tf.reduce_mean(3.0 * tf.square(u) * direction3, axis=0)
    j34 = tf.reduce_mean(3.0 * tf.square(u) * direction4, axis=0)
    j43 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction3, axis=0)
    j44 = tf.reduce_mean(4.0 * tf.pow(u, 3.0) * direction4, axis=0)
    j33_tangent = tf.reduce_mean(
        6.0 * u[:, :, None] * u_tangent * direction3[:, :, None]
        + 3.0 * tf.square(u)[:, :, None] * direction3_tangent,
        axis=0,
    )
    j34_tangent = tf.reduce_mean(
        6.0 * u[:, :, None] * u_tangent * direction4[:, :, None]
        + 3.0 * tf.square(u)[:, :, None] * direction4_tangent,
        axis=0,
    )
    j43_tangent = tf.reduce_mean(
        12.0 * tf.square(u)[:, :, None] * u_tangent * direction3[:, :, None]
        + 4.0 * tf.pow(u, 3.0)[:, :, None] * direction3_tangent,
        axis=0,
    )
    j44_tangent = tf.reduce_mean(
        12.0 * tf.square(u)[:, :, None] * u_tangent * direction4[:, :, None]
        + 4.0 * tf.pow(u, 3.0)[:, :, None] * direction4_tangent,
        axis=0,
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
    residual = tf.stack([d3, d4], axis=-1)
    residual_tangent = tf.stack([d3_tangent, d4_tangent], axis=-2)
    normal = tf.linalg.matmul(jacobian, jacobian, transpose_a=True)
    normal += tf.cast(floor, points.dtype) * tf.eye(
        2, batch_shape=[tf.shape(points)[1]], dtype=points.dtype
    )
    rhs = tf.linalg.matvec(jacobian, residual, transpose_a=True)
    coefficient = tf.cast(strength, points.dtype) * tf.linalg.solve(
        normal, rhs[:, :, None]
    )[:, :, 0]

    normal_tangent = (
        tf.einsum("daip,daj->dijp", jacobian_tangent, jacobian)
        + tf.einsum("dai,dajp->dijp", jacobian, jacobian_tangent)
    )
    rhs_tangent = (
        tf.einsum("daip,da->dip", jacobian_tangent, residual)
        + tf.einsum("dai,dap->dip", jacobian, residual_tangent)
    )
    coefficient_rhs_tangent = rhs_tangent - tf.einsum(
        "dijp,dj->dip", normal_tangent, coefficient / tf.cast(strength, points.dtype)
    ) if strength > 0.0 else tf.zeros_like(rhs_tangent)
    parameter_count = tf.shape(points_tangent)[-1]
    normal_batch = tf.broadcast_to(
        normal[None, :, :, :],
        [parameter_count, tf.shape(normal)[0], 2, 2],
    )
    coefficient_tangent = tf.cast(strength, points.dtype) * tf.transpose(
        tf.linalg.solve(
            normal_batch,
            tf.transpose(coefficient_rhs_tangent, [2, 0, 1])[:, :, :, None],
        )[:, :, :, 0],
        [1, 2, 0],
    )
    coefficient3 = coefficient[:, 0]
    coefficient4 = coefficient[:, 1]
    coefficient3_tangent = coefficient_tangent[:, 0, :]
    coefficient4_tangent = coefficient_tangent[:, 1, :]
    corrected = (
        u
        + direction3 * coefficient3[None, :]
        + direction4 * coefficient4[None, :]
    )
    corrected_tangent = (
        u_tangent
        + direction3_tangent * coefficient3[None, :, None]
        + direction3[:, :, None] * coefficient3_tangent[None, :, :]
        + direction4_tangent * coefficient4[None, :, None]
        + direction4[:, :, None] * coefficient4_tangent[None, :, :]
    )
    corrected_mean, corrected_cov, corrected_mean_tangent, corrected_cov_tangent = _uniform_moments_jvp(
        corrected, corrected_tangent
    )
    corrected_chol = tf.linalg.cholesky(corrected_cov)
    corrected_chol_tangent = _cholesky_jvp(corrected_chol, corrected_cov_tangent)
    output = _right_solve(corrected_chol, corrected - corrected_mean[None, :])
    output_tangent = _right_solve_jvp(
        corrected_chol,
        corrected_chol_tangent,
        corrected - corrected_mean[None, :],
        corrected_tangent - corrected_mean_tangent[None, :, :],
    )
    return output, output_tangent, target_skew - tf.reduce_mean(tf.pow(output, 3.0), axis=0), target_kurt - tf.reduce_mean(tf.pow(output, 4.0), axis=0)


def higher_moment_shape_jvp(
    source: tf.Tensor,
    weights: tf.Tensor,
    source_tangent: tf.Tensor,
    weights_tangent: tf.Tensor,
    points: tf.Tensor,
    points_tangent: tf.Tensor,
    *,
    correction_steps: int,
    strength: float,
    floor: float,
) -> dict[str, tf.Tensor]:
    """Apply bounded diagonal third/fourth-moment correction and its JVP."""
    if correction_steps < 0 or strength < 0.0 or floor <= 0.0:
        raise ValueError("invalid higher-moment correction controls")
    if correction_steps == 0:
        state_dim = tf.shape(source)[1]
        return {
            "particles": points,
            "particles_tangent": points_tangent,
            "target_skew": tf.zeros([state_dim], source.dtype),
            "target_kurtosis": tf.zeros([state_dim], source.dtype),
            "skew_residual": tf.zeros([state_dim], source.dtype),
            "kurtosis_residual": tf.zeros([state_dim], source.dtype),
            "valid": tf.reduce_all(tf.math.is_finite(points))
            & tf.reduce_all(tf.math.is_finite(points_tangent)),
        }
    state_dim = tf.shape(source)[1]
    mean, covariance, mean_tangent, covariance_tangent = _weighted_moments_jvp(
        source, weights, source_tangent, weights_tangent
    )
    target_chol = tf.linalg.cholesky(covariance)
    target_chol_tangent = _cholesky_jvp(target_chol, covariance_tangent)
    target_skew, target_kurt, target_skew_tangent, target_kurt_tangent = _weighted_diag_moments_jvp(
        source, weights, source_tangent, weights_tangent, mean, mean_tangent, target_chol, target_chol_tangent
    )

    steps = tf.constant(correction_steps, tf.int32)
    residual3 = tf.zeros([state_dim], source.dtype)
    residual4 = tf.zeros([state_dim], source.dtype)

    def body(index, current, current_tangent, _r3, _r4):
        next_points, next_tangent, next_r3, next_r4 = _shape_iteration_jvp(
            current, current_tangent, target_skew, target_kurt,
            target_skew_tangent, target_kurt_tangent,
            strength=strength, floor=floor
        )
        return index + 1, next_points, next_tangent, next_r3, next_r4

    # Initialize the loop state with a standardization of the input cloud.
    initial_mean, initial_cov, initial_mean_tangent, initial_cov_tangent = _uniform_moments_jvp(points, points_tangent)
    initial_chol = tf.linalg.cholesky(initial_cov)
    initial_chol_tangent = _cholesky_jvp(initial_chol, initial_cov_tangent)
    initial_standardized = _right_solve(initial_chol, points - initial_mean[None, :])
    initial_standardized_tangent = _right_solve_jvp(
        initial_chol, initial_chol_tangent, points - initial_mean[None, :], points_tangent - initial_mean_tangent[None, :, :]
    )

    _, standardized, standardized_tangent, residual3, residual4 = tf.while_loop(
        lambda index, *_: index < steps,
        body,
        (tf.zeros([], tf.int32), initial_standardized, initial_standardized_tangent, residual3, residual4),
        parallel_iterations=1,
    )
    output = mean[None, :] + tf.linalg.matmul(standardized, target_chol, transpose_b=True)
    output_tangent = (
        mean_tangent[None, :, :]
        + tf.einsum("nip,ji->njp", standardized_tangent, target_chol)
        + tf.einsum("ni,jip->njp", standardized, target_chol_tangent)
    )
    valid = tf.reduce_all(tf.math.is_finite(output)) & tf.reduce_all(tf.math.is_finite(output_tangent))
    return {
        "particles": output,
        "particles_tangent": output_tangent,
        "target_skew": target_skew,
        "target_kurtosis": target_kurt,
        "skew_residual": residual3,
        "kurtosis_residual": residual4,
        "valid": valid,
    }


__all__ = ["higher_moment_shape_jvp"]
