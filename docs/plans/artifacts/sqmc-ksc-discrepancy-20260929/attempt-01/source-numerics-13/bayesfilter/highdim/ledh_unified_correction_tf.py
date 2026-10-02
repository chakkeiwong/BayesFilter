"""Batched higher-moment correction boundary for the unified LEDH engine.

This branch-by-abstraction boundary preserves the existing, oracle-gated
single-cloud finite program while exposing explicit B and K axes. ``tf.map_fn``
executes as TensorFlow control flow (never a Python row loop, pfor, or autodiff).
The public shape contract remains stable when the internals are fully fused.
"""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim.higher_moment_contract_e import higher_moment_shape_jvp

Tensor = tf.Tensor

_VECTOR_KEYS = (
    "target_skew",
    "target_kurtosis",
    "skew_residual",
    "kurtosis_residual",
)
_MATRIX_KEYS = (
    "target_pairwise_co_skew",
    "target_pairwise_co_kurtosis",
    "pairwise_co_skew_residual",
    "pairwise_co_kurtosis_residual",
    "pairwise_target_mask",
    "pairwise_co_skew_target_mask",
    "pairwise_co_kurtosis_target_mask",
)
_SCALAR_FLOAT_KEYS = (
    "maximum_pairwise_pre_cap_particle_rms",
    "maximum_pairwise_post_cap_particle_rms",
    "minimum_pairwise_particle_cap_scale",
    "maximum_coordinatewise_pre_cap_absolute",
    "maximum_coordinatewise_post_cap_absolute",
    "mean_coordinatewise_cap_displacement",
    "fraction_coordinatewise_cap_active",
    "minimum_coordinatewise_cap_derivative",
    "projected_cumulant_residual_norm",
    "projected_cumulant_third_residual_norm",
    "projected_cumulant_fourth_residual_norm",
    "minimum_pearson_feasibility_margin",
    "minimum_finite_particle_upper_margin",
    "maximum_diagonal_scaled_system_condition",
    "maximum_diagonal_pre_cap_particle_rms",
    "maximum_diagonal_post_cap_particle_rms",
)


def _output_signature(
    particle_count: int, state_dim: int, direction_count: int, dtype: tf.DType
) -> dict[str, tf.TensorSpec]:
    signature: dict[str, tf.TensorSpec] = {
        "particles": tf.TensorSpec([particle_count, state_dim], dtype),
        "particles_tangent": tf.TensorSpec(
            [particle_count, state_dim, direction_count], dtype
        ),
        "target_source_id": tf.TensorSpec([], tf.int32),
        "valid": tf.TensorSpec([], tf.bool),
    }
    signature.update(
        {key: tf.TensorSpec([state_dim], dtype) for key in _VECTOR_KEYS}
    )
    signature.update(
        {key: tf.TensorSpec([state_dim, state_dim], dtype) for key in _MATRIX_KEYS}
    )
    signature.update(
        {key: tf.TensorSpec([], dtype) for key in _SCALAR_FLOAT_KEYS}
    )
    return signature


def batched_higher_moment_shape_jvp(
    source: Tensor,
    weights: Tensor,
    source_tangent: Tensor,
    weights_tangent: Tensor,
    points: Tensor,
    points_tangent: Tensor,
    *,
    correction_steps: int,
    strength: float,
    floor: float = 1.0e-5,
    diagonal_lm_damping: float = 0.0,
    diagonal_lm_scale_floor: float = 1.0e-6,
    diagonal_trust_radius: float = 0.0,
    pairwise_correction_steps: int = 0,
    pairwise_strength: float = 0.0,
    pairwise_floor: float = 1.0e-5,
    pairwise_particle_rms_cap: float = 0.0,
    coordinatewise_standardized_cap: float = 0.0,
    coordinatewise_standardized_cap_power: int = 8,
) -> dict[str, Tensor]:
    """Correct B clouds with K analytical tangent directions.

    Inputs use source/points ``[B,N,D]``, weights ``[B,N]``, particle tangents
    ``[K,B,N,D]``, and weight tangents ``[K,B,N]``. Outputs preserve B first,
    except particle tangents which use ``[K,B,N,D]`` like the unified engine.
    """
    source = tf.convert_to_tensor(source)
    dtype = source.dtype
    weights = tf.cast(weights, dtype)
    source_tangent = tf.cast(source_tangent, dtype)
    weights_tangent = tf.cast(weights_tangent, dtype)
    points = tf.cast(points, dtype)
    points_tangent = tf.cast(points_tangent, dtype)
    if source.shape.rank != 3 or points.shape.rank != 3:
        raise ValueError("source and points must have shape [B,N,D]")
    if source_tangent.shape.rank != 4 or points_tangent.shape.rank != 4:
        raise ValueError("particle tangents must have shape [K,B,N,D]")
    if weights.shape.rank != 2 or weights_tangent.shape.rank != 3:
        raise ValueError("weights must be [B,N] and tangents [K,B,N]")
    particle_count = int(source.shape[1])
    state_dim = int(source.shape[2])
    direction_count = int(source_tangent.shape[0])

    source_tangent_by_row = tf.transpose(source_tangent, [1, 2, 3, 0])
    weights_tangent_by_row = tf.transpose(weights_tangent, [1, 2, 0])
    points_tangent_by_row = tf.transpose(points_tangent, [1, 2, 3, 0])

    def correct_row(row):
        row_source, row_weights, row_source_tangent, row_weights_tangent, row_points, row_points_tangent = row
        return higher_moment_shape_jvp(
            row_source,
            row_weights,
            row_source_tangent,
            row_weights_tangent,
            row_points,
            row_points_tangent,
            correction_steps=correction_steps,
            strength=strength,
            floor=floor,
            diagonal_lm_damping=diagonal_lm_damping,
            diagonal_lm_scale_floor=diagonal_lm_scale_floor,
            diagonal_trust_radius=diagonal_trust_radius,
            pairwise_correction_steps=pairwise_correction_steps,
            pairwise_strength=pairwise_strength,
            pairwise_floor=pairwise_floor,
            pairwise_particle_rms_cap=pairwise_particle_rms_cap,
            coordinatewise_standardized_cap=coordinatewise_standardized_cap,
            coordinatewise_standardized_cap_power=coordinatewise_standardized_cap_power,
        )

    result = tf.map_fn(
        correct_row,
        (
            source,
            weights,
            source_tangent_by_row,
            weights_tangent_by_row,
            points,
            points_tangent_by_row,
        ),
        fn_output_signature=_output_signature(
            particle_count, state_dim, direction_count, dtype
        ),
        parallel_iterations=1,
    )
    result["particles_tangent"] = tf.transpose(
        result["particles_tangent"], [3, 0, 1, 2]
    )
    return result


__all__ = ["batched_higher_moment_shape_jvp"]
