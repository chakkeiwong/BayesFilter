"""Scale-aware finite GenUT shape steps in TensorFlow.

These helpers do not choose moment targets.  They solve one declared local
least-squares problem and optionally cap the resulting particle displacement.
"""

from __future__ import annotations

import tensorflow as tf


GENUT_SHAPE_SOLVER_ID = "genut_column_scaled_lm_smooth_rms_trust_v1"


def scaled_lm_coefficients_value(
    jacobian: tf.Tensor,
    residual: tf.Tensor,
    *,
    strength: float,
    damping: float,
    scale_floor: float,
) -> dict[str, tf.Tensor]:
    """Solve a column-scaled Levenberg--Marquardt subproblem."""

    if strength < 0.0 or damping <= 0.0 or scale_floor <= 0.0:
        raise ValueError("scaled LM controls must be positive")
    dtype = jacobian.dtype
    floor = tf.cast(scale_floor, dtype)
    column_scale = tf.sqrt(
        tf.reduce_sum(tf.square(jacobian), axis=-2) + tf.square(floor)
    )
    scaled_jacobian = jacobian / column_scale[..., None, :]
    system = tf.linalg.matmul(
        scaled_jacobian, scaled_jacobian, transpose_a=True
    )
    system += tf.cast(damping, dtype) * tf.eye(2, dtype=dtype)
    rhs = tf.linalg.matvec(scaled_jacobian, residual, transpose_a=True)
    scaled_coefficient = tf.linalg.solve(system, rhs[..., None])[..., 0]
    coefficient = (
        tf.cast(strength, dtype) * scaled_coefficient / column_scale
    )
    eigenvalues = tf.linalg.eigvalsh(system)
    return {
        "coefficient": coefficient,
        "column_scale": column_scale,
        "scaled_system": system,
        "scaled_system_condition": eigenvalues[..., -1]
        / eigenvalues[..., 0],
    }


def scaled_lm_coefficients_jvp(
    jacobian: tf.Tensor,
    residual: tf.Tensor,
    jacobian_tangent: tf.Tensor,
    residual_tangent: tf.Tensor,
    *,
    strength: float,
    damping: float,
    scale_floor: float,
) -> dict[str, tf.Tensor]:
    """Solve the scaled LM system and differentiate the complete solve."""

    result = scaled_lm_coefficients_value(
        jacobian,
        residual,
        strength=strength,
        damping=damping,
        scale_floor=scale_floor,
    )
    column_scale = result["column_scale"]
    scaled_jacobian = jacobian / column_scale[..., None, :]
    column_scale_tangent = tf.reduce_sum(
        jacobian[..., :, :, None] * jacobian_tangent, axis=-3
    ) / column_scale[..., :, None]
    scaled_jacobian_tangent = (
        jacobian_tangent / column_scale[..., None, :, None]
        - jacobian[..., :, :, None]
        * column_scale_tangent[..., None, :, :]
        / tf.square(column_scale)[..., None, :, None]
    )
    system_tangent = tf.einsum(
        "...kip,...kj->...ijp", scaled_jacobian_tangent, scaled_jacobian
    ) + tf.einsum(
        "...ki,...kjp->...ijp", scaled_jacobian, scaled_jacobian_tangent
    )
    rhs_tangent = tf.einsum(
        "...kip,...k->...ip", scaled_jacobian_tangent, residual
    ) + tf.einsum(
        "...ki,...kp->...ip", scaled_jacobian, residual_tangent
    )
    if strength > 0.0:
        scaled_coefficient = (
            result["coefficient"]
            * column_scale
            / tf.cast(strength, jacobian.dtype)
        )
    else:
        scaled_coefficient = tf.zeros_like(result["coefficient"])
    solve_rhs = rhs_tangent - tf.einsum(
        "...ijp,...j->...ip", system_tangent, scaled_coefficient
    )
    scaled_coefficient_tangent = tf.linalg.solve(
        result["scaled_system"], solve_rhs
    )
    coefficient_tangent = tf.cast(strength, jacobian.dtype) * (
        scaled_coefficient_tangent / column_scale[..., :, None]
        - scaled_coefficient[..., :, None]
        * column_scale_tangent
        / tf.square(column_scale)[..., :, None]
    )
    result.update(
        {
            "coefficient_tangent": coefficient_tangent,
            "column_scale_tangent": column_scale_tangent,
        }
    )
    return result


def smooth_rms_cap_value(
    displacement: tf.Tensor, *, radius: float
) -> dict[str, tf.Tensor]:
    """Smoothly cap each row's coordinate RMS below the given radius."""

    if radius <= 0.0:
        raise ValueError("smooth RMS cap radius must be positive")
    mean_square = tf.reduce_mean(tf.square(displacement), axis=-1)
    radius_tensor = tf.cast(radius, displacement.dtype)
    scale = tf.math.rsqrt(1.0 + mean_square / tf.square(radius_tensor))
    capped = displacement * scale[..., None]
    return {
        "displacement": capped,
        "scale": scale,
        "pre_rms": tf.sqrt(mean_square),
        "post_rms": tf.sqrt(tf.reduce_mean(tf.square(capped), axis=-1)),
    }


def smooth_rms_cap_jvp(
    displacement: tf.Tensor,
    displacement_tangent: tf.Tensor,
    *,
    radius: float,
) -> dict[str, tf.Tensor]:
    """Apply the smooth row-RMS cap and its total JVP."""

    result = smooth_rms_cap_value(displacement, radius=radius)
    mean_square = tf.reduce_mean(tf.square(displacement), axis=-1)
    mean_square_tangent = 2.0 * tf.reduce_mean(
        displacement[..., :, None] * displacement_tangent, axis=-2
    )
    radius_tensor = tf.cast(radius, displacement.dtype)
    base = 1.0 + mean_square / tf.square(radius_tensor)
    scale_tangent = (
        -0.5
        * tf.pow(base[..., None], -1.5)
        * mean_square_tangent
        / tf.square(radius_tensor)
    )
    capped_tangent = (
        displacement_tangent * result["scale"][..., None, None]
        + displacement[..., :, None] * scale_tangent[..., None, :]
    )
    result["displacement_tangent"] = capped_tangent
    result["scale_tangent"] = scale_tangent
    return result


def necessary_marginal_feasibility(
    skewness: tf.Tensor, kurtosis: tf.Tensor, particle_count: tf.Tensor
) -> dict[str, tf.Tensor]:
    """Return necessary, not sufficient, equal-weight moment diagnostics."""

    upper = tf.cast(particle_count - 1, kurtosis.dtype)
    pearson_margin = kurtosis - tf.square(skewness) - 1.0
    upper_margin = upper - kurtosis
    return {
        "pearson_margin": pearson_margin,
        "finite_particle_upper_margin": upper_margin,
        "valid": (pearson_margin >= 0.0) & (upper_margin >= 0.0),
    }


__all__ = [
    "GENUT_SHAPE_SOLVER_ID",
    "necessary_marginal_feasibility",
    "scaled_lm_coefficients_jvp",
    "scaled_lm_coefficients_value",
    "smooth_rms_cap_jvp",
    "smooth_rms_cap_value",
]
