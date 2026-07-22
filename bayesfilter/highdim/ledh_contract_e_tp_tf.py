"""Dense TensorFlow primitives for experimental Contract E--TP.

The model owns teacher construction and feature evaluation.  This module only
normalizes the finite teacher and applies a prepared, fixed projection chart.
It is deliberately separate from the canonical Contract E--Chol route.
"""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf


ALGORITHM_ID = "contract_e_tp_experimental_v1"
StreamingBlockProgram = Callable[
    [tuple[tf.Tensor, ...], tf.Tensor],
    tuple[tf.Tensor, tf.Tensor, tf.Tensor],
]


def _dtype_epsilon(dtype: tf.dtypes.DType) -> tf.Tensor:
    return tf.cast(tf.experimental.numpy.finfo(dtype.as_numpy_dtype).eps, dtype)


def _as_indices(active_indices: tf.Tensor | tuple[int, ...]) -> tf.Tensor:
    return tf.convert_to_tensor(active_indices, dtype=tf.int32)


def _validate_teacher_inputs(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
) -> None:
    tf.debugging.assert_rank(teacher_points, 2)
    tf.debugging.assert_rank(log_unnormalized_weights, 1)
    tf.debugging.assert_rank(teacher_features, 2)
    tf.debugging.assert_equal(
        tf.shape(teacher_points)[0], tf.shape(log_unnormalized_weights)[0]
    )
    tf.debugging.assert_equal(
        tf.shape(teacher_features)[1], tf.shape(log_unnormalized_weights)[0]
    )
    tf.debugging.assert_all_finite(teacher_points, "teacher points must be finite")
    tf.debugging.assert_all_finite(
        log_unnormalized_weights, "teacher log weights must be finite"
    )
    tf.debugging.assert_all_finite(teacher_features, "teacher features must be finite")
    tf.debugging.assert_equal(
        teacher_features[0],
        tf.ones_like(teacher_features[0]),
        message="feature row zero must be the exact mass feature",
    )


def _teacher_input_validity(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
) -> tf.Tensor:
    return (
        tf.reduce_all(tf.math.is_finite(teacher_points))
        & tf.reduce_all(tf.math.is_finite(log_unnormalized_weights))
        & tf.reduce_all(tf.math.is_finite(teacher_features))
        & tf.reduce_all(tf.equal(teacher_features[0], 1.0))
    )


def _validate_fixed_indices(
    active_indices: tf.Tensor,
    teacher_count: tf.Tensor,
    required_count: tf.Tensor | None = None,
) -> None:
    tf.debugging.assert_rank(active_indices, 1)
    tf.debugging.assert_non_negative(active_indices)
    tf.debugging.assert_less(active_indices, teacher_count)
    unique_indices, _ = tf.unique(active_indices)
    tf.debugging.assert_equal(
        tf.size(unique_indices),
        tf.size(active_indices),
        message="the prepared active indices must be unique",
    )
    if required_count is not None:
        tf.debugging.assert_equal(
            tf.size(active_indices),
            required_count,
            message="a square chart requires one fixed anchor per feature",
        )


def _fixed_index_validity(
    active_indices: tf.Tensor,
    teacher_count: tf.Tensor,
    required_count: tf.Tensor | None = None,
) -> tf.Tensor:
    unique_indices, _ = tf.unique(active_indices)
    valid = (
        tf.reduce_all(active_indices >= 0)
        & tf.reduce_all(active_indices < teacher_count)
        & tf.equal(tf.size(unique_indices), tf.size(active_indices))
    )
    if required_count is not None:
        valid &= tf.equal(tf.size(active_indices), required_count)
    return valid


def _validate_row_scale(row_scale: tf.Tensor, feature_count: tf.Tensor) -> None:
    tf.debugging.assert_rank(row_scale, 1)
    tf.debugging.assert_equal(tf.size(row_scale), feature_count)
    tf.debugging.assert_all_finite(row_scale, "row scales must be finite")
    tf.debugging.assert_positive(row_scale, "row scales must be positive")


def _row_scale_validity(row_scale: tf.Tensor, feature_count: tf.Tensor) -> tf.Tensor:
    return (
        tf.equal(tf.size(row_scale), feature_count)
        & tf.reduce_all(tf.math.is_finite(row_scale))
        & tf.reduce_all(row_scale > 0)
    )


def _linear_chart_diagnostics(
    scaled_matrix: tf.Tensor,
    scaled_target: tf.Tensor,
    weights: tf.Tensor,
) -> dict[str, tf.Tensor]:
    singular_values = tf.linalg.svd(scaled_matrix, compute_uv=False)
    largest_singular_value = tf.reduce_max(singular_values)
    smallest_singular_value = tf.reduce_min(singular_values)
    dimension = tf.cast(tf.shape(scaled_matrix)[0], scaled_matrix.dtype)
    epsilon = _dtype_epsilon(scaled_matrix.dtype)
    rank_tolerance = epsilon * dimension * largest_singular_value
    condition_number = largest_singular_value / smallest_singular_value
    gamma = dimension * epsilon / (1.0 - dimension * epsilon)
    condition_roundoff = condition_number * gamma
    forward_error_bound = condition_roundoff / (1.0 - condition_roundoff)
    residual = tf.linalg.matvec(scaled_matrix, weights) - scaled_target
    relative_residual = tf.linalg.norm(residual) / tf.maximum(
        tf.linalg.norm(scaled_target), tf.constant(1.0, scaled_target.dtype)
    )
    finite = (
        tf.reduce_all(tf.math.is_finite(scaled_matrix))
        & tf.reduce_all(tf.math.is_finite(scaled_target))
        & tf.reduce_all(tf.math.is_finite(weights))
    )
    full_rank = smallest_singular_value > rank_tolerance
    roundoff_valid = condition_roundoff < 1.0
    residual_valid = relative_residual <= forward_error_bound
    positive = tf.reduce_all(weights > 0)
    return {
        "singular_values": singular_values,
        "smallest_singular_value": smallest_singular_value,
        "largest_singular_value": largest_singular_value,
        "rank_tolerance": rank_tolerance,
        "condition_number": condition_number,
        "condition_roundoff": condition_roundoff,
        "forward_error_bound": forward_error_bound,
        "scaled_relative_residual": relative_residual,
        "minimum_weight": tf.reduce_min(weights),
        "finite": finite,
        "full_rank": full_rank,
        "roundoff_valid": roundoff_valid,
        "residual_valid": residual_valid,
        "positive": positive,
        "valid_chart": finite & full_rank & roundoff_valid & residual_valid & positive,
    }


def _assert_valid_chart(
    diagnostics: dict[str, tf.Tensor],
    weights: tf.Tensor,
) -> None:
    tf.debugging.assert_all_finite(weights, "projected weights must be finite")
    tf.debugging.assert_greater(
        diagnostics["smallest_singular_value"],
        diagnostics["rank_tolerance"],
        message="prepared projection chart is rank deficient",
    )
    tf.debugging.assert_less(
        diagnostics["condition_roundoff"],
        tf.constant(1.0, weights.dtype),
        message="projection chart is too ill-conditioned for its dtype",
    )
    tf.debugging.assert_less_equal(
        diagnostics["scaled_relative_residual"],
        diagnostics["forward_error_bound"],
        message="projection residual exceeds its condition-number roundoff bound",
    )
    tf.debugging.assert_positive(
        weights,
        message="prepared projection chart produced a nonpositive weight",
    )


def _poison_invalid(value: tf.Tensor, valid: tf.Tensor) -> tf.Tensor:
    return tf.where(
        valid,
        value,
        tf.fill(tf.shape(value), tf.constant(float("nan"), value.dtype)),
    )


def _dense_teacher_reduce_core(
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Normalize a dense finite teacher and reduce its feature expectation."""

    log_normalizer = tf.reduce_logsumexp(log_unnormalized_weights)
    normalized_weights = tf.exp(log_unnormalized_weights - log_normalizer)
    target = tf.linalg.matvec(teacher_features, normalized_weights)
    return {
        "log_normalizer": log_normalizer,
        "normalized_weights": normalized_weights,
        "target": target,
    }


def _dense_teacher_reduce_jvp_core(
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    forward = _dense_teacher_reduce_core(log_unnormalized_weights, teacher_features)
    log_normalizer_tangent = tf.tensordot(
        forward["normalized_weights"], log_unnormalized_weights_tangent, axes=1
    )
    normalized_weights_tangent = forward["normalized_weights"] * (
        log_unnormalized_weights_tangent - log_normalizer_tangent
    )
    target_tangent = (
        tf.linalg.matvec(teacher_features_tangent, forward["normalized_weights"])
        + tf.linalg.matvec(teacher_features, normalized_weights_tangent)
    )
    return {
        **forward,
        "log_normalizer_tangent": log_normalizer_tangent,
        "normalized_weights_tangent": normalized_weights_tangent,
        "target_tangent": target_tangent,
    }


def _contract_e_tp_dense_square_forward_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Apply a prepared square Contract E--TP chart and fail closed."""

    _validate_teacher_inputs(teacher_points, log_unnormalized_weights, teacher_features)
    active_indices = _as_indices(active_indices)
    feature_count = tf.shape(teacher_features)[0]
    _validate_fixed_indices(active_indices, tf.shape(teacher_points)[0], feature_count)
    _validate_row_scale(row_scale, feature_count)
    input_valid = (
        _teacher_input_validity(
            teacher_points, log_unnormalized_weights, teacher_features
        )
        & _fixed_index_validity(
            active_indices, tf.shape(teacher_points)[0], feature_count
        )
        & _row_scale_validity(row_scale, feature_count)
    )
    teacher = _dense_teacher_reduce_core(log_unnormalized_weights, teacher_features)
    active_features = tf.gather(teacher_features, active_indices, axis=1)
    scaled_matrix = active_features / row_scale[:, None]
    scaled_target = teacher["target"] / row_scale
    student_weights = tf.linalg.solve(scaled_matrix, scaled_target[:, None])[:, 0]
    student_points = tf.gather(teacher_points, active_indices, axis=0)
    matched_target = tf.linalg.matvec(active_features, student_weights)
    diagnostics = _linear_chart_diagnostics(
        scaled_matrix, scaled_target, student_weights
    )
    _assert_valid_chart(diagnostics, student_weights)
    valid_chart = input_valid & diagnostics["valid_chart"]
    return {
        **teacher,
        **diagnostics,
        "input_valid": input_valid,
        "valid_chart": valid_chart,
        "active_indices": active_indices,
        "active_features": active_features,
        "scaled_matrix": scaled_matrix,
        "scaled_target": scaled_target,
        "student_points": _poison_invalid(student_points, valid_chart),
        "student_weights": _poison_invalid(student_weights, valid_chart),
        "matched_target": _poison_invalid(matched_target, valid_chart),
        "feature_residual": matched_target - teacher["target"],
        "row_scale": row_scale,
    }


def _contract_e_tp_dense_square_jvp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Explicit tangent of the dense fixed square chart."""

    forward = _contract_e_tp_dense_square_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
    )
    teacher_jvp = _dense_teacher_reduce_jvp_core(
        log_unnormalized_weights,
        teacher_features,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
    )
    active_features_tangent = tf.gather(
        teacher_features_tangent, forward["active_indices"], axis=1
    )
    scaled_matrix_tangent = active_features_tangent / row_scale[:, None]
    scaled_target_tangent = teacher_jvp["target_tangent"] / row_scale
    student_weights_tangent = tf.linalg.solve(
        forward["scaled_matrix"],
        (
            scaled_target_tangent
            - tf.linalg.matvec(scaled_matrix_tangent, forward["student_weights"])
        )[:, None],
    )[:, 0]
    matched_target_tangent = (
        tf.linalg.matvec(active_features_tangent, forward["student_weights"])
        + tf.linalg.matvec(forward["active_features"], student_weights_tangent)
    )
    return {
        **forward,
        "log_normalizer_tangent": teacher_jvp["log_normalizer_tangent"],
        "normalized_weights_tangent": teacher_jvp["normalized_weights_tangent"],
        "target_tangent": teacher_jvp["target_tangent"],
        "active_features_tangent": active_features_tangent,
        "student_points_tangent": tf.gather(
            teacher_points_tangent, forward["active_indices"], axis=0
        ),
        "student_weights_tangent": student_weights_tangent,
        "matched_target_tangent": matched_target_tangent,
    }


def _scatter_active_columns(
    active_values: tf.Tensor,
    active_indices: tf.Tensor,
    column_count: tf.Tensor,
) -> tf.Tensor:
    transposed = tf.scatter_nd(
        active_indices[:, None],
        tf.transpose(active_values),
        [column_count, tf.shape(active_values)[0]],
    )
    return tf.transpose(transposed)


def _contract_e_tp_dense_square_vjp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Manual transpose derivative with separate teacher-input adjoints."""

    forward = _contract_e_tp_dense_square_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
    )
    point_count = tf.shape(teacher_points)[0]
    point_bar = tf.scatter_nd(
        forward["active_indices"][:, None],
        upstream_student_points,
        tf.shape(teacher_points),
    )
    active_feature_bar = tf.einsum(
        "i,j->ij", upstream_matched_target, forward["student_weights"]
    )
    student_weight_bar = upstream_student_weights + tf.linalg.matvec(
        forward["active_features"], upstream_matched_target, transpose_a=True
    )
    scaled_target_bar = tf.linalg.solve(
        tf.transpose(forward["scaled_matrix"]), student_weight_bar[:, None]
    )[:, 0]
    scaled_matrix_bar = -tf.einsum(
        "i,j->ij", scaled_target_bar, forward["student_weights"]
    )
    target_bar = scaled_target_bar / row_scale
    active_feature_bar += scaled_matrix_bar / row_scale[:, None]
    feature_bar = tf.einsum(
        "i,j->ij", target_bar, forward["normalized_weights"]
    )
    feature_bar += _scatter_active_columns(
        active_feature_bar, forward["active_indices"], point_count
    )
    normalized_weight_bar = tf.linalg.matvec(
        teacher_features, target_bar, transpose_a=True
    )
    centered_weight_bar = normalized_weight_bar - tf.tensordot(
        forward["normalized_weights"], normalized_weight_bar, axes=1
    )
    log_weight_bar = forward["normalized_weights"] * centered_weight_bar
    log_weight_bar += upstream_log_normalizer * forward["normalized_weights"]
    return {
        **forward,
        "teacher_points_bar": point_bar,
        "log_unnormalized_weights_bar": log_weight_bar,
        "teacher_features_bar": feature_bar,
    }


def _matrix_condition_diagnostics(matrix: tf.Tensor) -> dict[str, tf.Tensor]:
    """Return finite-program rank and roundoff diagnostics without assertions."""

    singular_values = tf.linalg.svd(matrix, compute_uv=False)
    largest = tf.reduce_max(singular_values)
    smallest = tf.reduce_min(singular_values)
    dimension = tf.cast(tf.shape(matrix)[0], matrix.dtype)
    epsilon = _dtype_epsilon(matrix.dtype)
    tolerance = epsilon * dimension * largest
    condition = largest / smallest
    gamma = dimension * epsilon / (1.0 - dimension * epsilon)
    condition_roundoff = condition * gamma
    forward_error_bound = condition_roundoff / (1.0 - condition_roundoff)
    finite = tf.reduce_all(tf.math.is_finite(matrix)) & tf.reduce_all(
        tf.math.is_finite(singular_values)
    )
    full_rank = finite & (smallest > tolerance)
    roundoff_valid = tf.math.is_finite(condition_roundoff) & (
        condition_roundoff < 1.0
    )
    return {
        "singular_values": singular_values,
        "smallest_singular_value": smallest,
        "largest_singular_value": largest,
        "rank_tolerance": tolerance,
        "condition_number": condition,
        "condition_roundoff": condition_roundoff,
        "forward_error_bound": forward_error_bound,
        "finite": finite,
        "full_rank": full_rank,
        "roundoff_valid": roundoff_valid,
    }


def _contract_e_tp_diagonal_kkt_forward_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Apply a fixed overcomplete Pearson chart using only a q-by-q solve.

    This candidate core deliberately does not assert on numerical chart
    failures.  Invalid prepared inputs are routed through a harmless identity
    Cholesky, reported by ``valid_chart=False``, and claim-bearing outputs are
    poisoned with NaNs.
    """

    teacher_points = tf.convert_to_tensor(teacher_points)
    log_unnormalized_weights = tf.convert_to_tensor(log_unnormalized_weights)
    teacher_features = tf.convert_to_tensor(teacher_features)
    active_indices = _as_indices(active_indices)
    row_scale = tf.convert_to_tensor(row_scale, teacher_features.dtype)
    reference_weights = tf.convert_to_tensor(
        reference_weights, teacher_features.dtype
    )
    teacher_count = tf.shape(teacher_points)[0]
    feature_count = tf.shape(teacher_features)[0]
    anchor_count = tf.size(active_indices)

    indices_in_range = tf.reduce_all(active_indices >= 0) & tf.reduce_all(
        active_indices < teacher_count
    )
    safe_indices = tf.where(
        (active_indices >= 0) & (active_indices < teacher_count),
        active_indices,
        tf.zeros_like(active_indices),
    )
    unique_indices, _ = tf.unique(safe_indices)
    indices_valid = (
        indices_in_range
        & tf.equal(tf.size(unique_indices), anchor_count)
        & tf.greater_equal(anchor_count, feature_count)
    )
    shapes_valid = (
        tf.equal(tf.shape(teacher_points)[0], tf.size(log_unnormalized_weights))
        & tf.equal(tf.shape(teacher_features)[1], teacher_count)
        & tf.equal(tf.size(row_scale), feature_count)
        & tf.equal(tf.size(reference_weights), anchor_count)
    )
    input_valid = (
        shapes_valid
        & indices_valid
        & _teacher_input_validity(
            teacher_points, log_unnormalized_weights, teacher_features
        )
        & _row_scale_validity(row_scale, feature_count)
        & tf.reduce_all(tf.math.is_finite(reference_weights))
        & tf.reduce_all(reference_weights > 0.0)
    )

    teacher = _dense_teacher_reduce_core(log_unnormalized_weights, teacher_features)
    active_features = tf.gather(teacher_features, safe_indices, axis=1)
    scaled_matrix = active_features / row_scale[:, None]
    scaled_target = teacher["target"] / row_scale
    inverse_precision_features = reference_weights[:, None] * tf.transpose(
        scaled_matrix
    )
    constraint_gram = tf.linalg.matmul(
        scaled_matrix, inverse_precision_features
    )
    matrix_diagnostics = _matrix_condition_diagnostics(scaled_matrix)
    gram_diagnostics = _matrix_condition_diagnostics(constraint_gram)
    solve_valid = (
        input_valid
        & matrix_diagnostics["finite"]
        & matrix_diagnostics["full_rank"]
        & matrix_diagnostics["roundoff_valid"]
        & gram_diagnostics["finite"]
        & gram_diagnostics["full_rank"]
        & gram_diagnostics["roundoff_valid"]
        & tf.reduce_all(tf.math.is_finite(scaled_target))
    )
    safe_gram = tf.where(
        solve_valid,
        constraint_gram,
        tf.eye(feature_count, dtype=constraint_gram.dtype),
    )
    constraint_residual = scaled_target - tf.linalg.matvec(
        scaled_matrix, reference_weights
    )
    safe_residual = tf.where(
        solve_valid, constraint_residual, tf.zeros_like(constraint_residual)
    )
    gram_cholesky = tf.linalg.cholesky(safe_gram)
    multiplier = tf.linalg.cholesky_solve(
        gram_cholesky, safe_residual[:, None]
    )[:, 0]
    student_weights_raw = reference_weights + tf.linalg.matvec(
        inverse_precision_features, multiplier
    )
    scaled_feature_residual = (
        tf.linalg.matvec(scaled_matrix, student_weights_raw) - scaled_target
    )
    scaled_relative_residual = tf.linalg.norm(scaled_feature_residual) / tf.maximum(
        tf.linalg.norm(scaled_target), tf.constant(1.0, scaled_target.dtype)
    )
    output_finite = (
        tf.reduce_all(tf.math.is_finite(student_weights_raw))
        & tf.reduce_all(tf.math.is_finite(multiplier))
        & tf.math.is_finite(scaled_relative_residual)
    )
    positive = tf.reduce_all(student_weights_raw > 0.0)
    residual_valid = scaled_relative_residual <= gram_diagnostics[
        "forward_error_bound"
    ]
    valid_chart = solve_valid & output_finite & positive & residual_valid
    student_points_raw = tf.gather(teacher_points, safe_indices, axis=0)
    matched_target_raw = tf.linalg.matvec(active_features, student_weights_raw)
    feature_residual = matched_target_raw - teacher["target"]
    return {
        **teacher,
        "input_valid": input_valid,
        "solve_valid": solve_valid,
        "output_finite": output_finite,
        "positive": positive,
        "residual_valid": residual_valid,
        "valid_chart": valid_chart,
        "active_indices": safe_indices,
        "active_features": active_features,
        "scaled_matrix": scaled_matrix,
        "scaled_target": scaled_target,
        "reference_weights": reference_weights,
        "inverse_precision_features": inverse_precision_features,
        "constraint_gram": constraint_gram,
        "constraint_residual": constraint_residual,
        "gram_cholesky": gram_cholesky,
        "multiplier": multiplier,
        "matrix_singular_values": matrix_diagnostics["singular_values"],
        "matrix_rank_tolerance": matrix_diagnostics["rank_tolerance"],
        "matrix_condition_number": matrix_diagnostics["condition_number"],
        "matrix_condition_roundoff": matrix_diagnostics["condition_roundoff"],
        "matrix_full_rank": matrix_diagnostics["full_rank"],
        "gram_singular_values": gram_diagnostics["singular_values"],
        "gram_rank_tolerance": gram_diagnostics["rank_tolerance"],
        "gram_condition_number": gram_diagnostics["condition_number"],
        "gram_condition_roundoff": gram_diagnostics["condition_roundoff"],
        "gram_forward_error_bound": gram_diagnostics["forward_error_bound"],
        "gram_full_rank": gram_diagnostics["full_rank"],
        "scaled_relative_residual": scaled_relative_residual,
        "minimum_weight": tf.reduce_min(student_weights_raw),
        "student_points": _poison_invalid(student_points_raw, valid_chart),
        "student_weights": _poison_invalid(student_weights_raw, valid_chart),
        "matched_target": _poison_invalid(matched_target_raw, valid_chart),
        "feature_residual": feature_residual,
        "row_scale": row_scale,
    }


def _contract_e_tp_diagonal_kkt_jvp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Explicit total JVP with frozen indices, scales, and Pearson reference."""

    forward = _contract_e_tp_diagonal_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
    )
    teacher_jvp = _dense_teacher_reduce_jvp_core(
        log_unnormalized_weights,
        teacher_features,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
    )
    active_features_tangent = tf.gather(
        teacher_features_tangent, forward["active_indices"], axis=1
    )
    matrix_tangent = active_features_tangent / row_scale[:, None]
    target_tangent = teacher_jvp["target_tangent"] / row_scale
    inverse_features_tangent = reference_weights[:, None] * tf.transpose(
        matrix_tangent
    )
    gram_tangent = (
        tf.linalg.matmul(matrix_tangent, forward["inverse_precision_features"])
        + tf.linalg.matmul(forward["scaled_matrix"], inverse_features_tangent)
    )
    residual_tangent = target_tangent - tf.linalg.matvec(
        matrix_tangent, reference_weights
    )
    multiplier_tangent = tf.linalg.cholesky_solve(
        forward["gram_cholesky"],
        (
            residual_tangent
            - tf.linalg.matvec(gram_tangent, forward["multiplier"])
        )[:, None],
    )[:, 0]
    student_weights_tangent_raw = (
        tf.linalg.matvec(inverse_features_tangent, forward["multiplier"])
        + tf.linalg.matvec(
            forward["inverse_precision_features"], multiplier_tangent
        )
    )
    matched_target_tangent_raw = (
        tf.linalg.matvec(active_features_tangent, forward["student_weights"])
        + tf.linalg.matvec(
            forward["active_features"], student_weights_tangent_raw
        )
    )
    valid = forward["valid_chart"]
    return {
        **forward,
        "log_normalizer_tangent": _poison_invalid(
            teacher_jvp["log_normalizer_tangent"], valid
        ),
        "normalized_weights_tangent": _poison_invalid(
            teacher_jvp["normalized_weights_tangent"], valid
        ),
        "target_tangent": _poison_invalid(teacher_jvp["target_tangent"], valid),
        "active_features_tangent": active_features_tangent,
        "student_points_tangent": _poison_invalid(
            tf.gather(
                teacher_points_tangent, forward["active_indices"], axis=0
            ),
            valid,
        ),
        "student_weights_tangent": _poison_invalid(
            student_weights_tangent_raw, valid
        ),
        "matched_target_tangent": _poison_invalid(
            matched_target_tangent_raw, valid
        ),
    }


def _contract_e_tp_diagonal_kkt_multi_jvp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Vectorized explicit JVP with directions on the last tensor axis."""

    forward = _contract_e_tp_diagonal_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
    )
    log_normalizer_tangent = tf.einsum(
        "m,mp->p",
        forward["normalized_weights"],
        log_unnormalized_weights_tangent,
    )
    normalized_weights_tangent = forward["normalized_weights"][:, None] * (
        log_unnormalized_weights_tangent - log_normalizer_tangent[None, :]
    )
    target_tangent = (
        tf.einsum(
            "qmp,m->qp",
            teacher_features_tangent,
            forward["normalized_weights"],
        )
        + tf.linalg.matmul(teacher_features, normalized_weights_tangent)
    )
    active_features_tangent = tf.gather(
        teacher_features_tangent, forward["active_indices"], axis=1
    )
    matrix_tangent = active_features_tangent / row_scale[:, None, None]
    scaled_target_tangent = target_tangent / row_scale[:, None]
    inverse_features_tangent = reference_weights[:, None, None] * tf.transpose(
        matrix_tangent, [1, 0, 2]
    )
    gram_tangent = (
        tf.einsum(
            "qkp,kr->qrp",
            matrix_tangent,
            forward["inverse_precision_features"],
        )
        + tf.einsum(
            "qk,krp->qrp",
            forward["scaled_matrix"],
            inverse_features_tangent,
        )
    )
    residual_tangent = scaled_target_tangent - tf.einsum(
        "qkp,k->qp", matrix_tangent, reference_weights
    )
    multiplier_tangent = tf.linalg.cholesky_solve(
        forward["gram_cholesky"],
        residual_tangent
        - tf.einsum("qrp,r->qp", gram_tangent, forward["multiplier"]),
    )
    student_weights_tangent_raw = (
        tf.einsum(
            "kqp,q->kp", inverse_features_tangent, forward["multiplier"]
        )
        + tf.linalg.matmul(
            forward["inverse_precision_features"], multiplier_tangent
        )
    )
    matched_target_tangent_raw = (
        tf.einsum(
            "qkp,k->qp", active_features_tangent, forward["student_weights"]
        )
        + tf.linalg.matmul(
            forward["active_features"], student_weights_tangent_raw
        )
    )
    valid = forward["valid_chart"]
    return {
        **forward,
        "log_normalizer_tangent": _poison_invalid(
            log_normalizer_tangent, valid
        ),
        "normalized_weights_tangent": _poison_invalid(
            normalized_weights_tangent, valid
        ),
        "target_tangent": _poison_invalid(target_tangent, valid),
        "active_features_tangent": active_features_tangent,
        "student_points_tangent": _poison_invalid(
            tf.gather(
                teacher_points_tangent, forward["active_indices"], axis=0
            ),
            valid,
        ),
        "student_weights_tangent": _poison_invalid(
            student_weights_tangent_raw, valid
        ),
        "matched_target_tangent": _poison_invalid(
            matched_target_tangent_raw, valid
        ),
    }


def _contract_e_tp_diagonal_kkt_vjp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Explicit transpose derivative for the frozen-reference Pearson chart."""

    forward = _contract_e_tp_diagonal_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
    )
    point_count = tf.shape(teacher_points)[0]
    point_bar = tf.scatter_nd(
        forward["active_indices"][:, None],
        upstream_student_points,
        tf.shape(teacher_points),
    )
    active_feature_bar = tf.einsum(
        "i,j->ij", upstream_matched_target, forward["student_weights"]
    )
    student_weight_bar = upstream_student_weights + tf.linalg.matvec(
        forward["active_features"], upstream_matched_target, transpose_a=True
    )
    matrix_bar = tf.einsum(
        "i,j->ij",
        forward["multiplier"],
        reference_weights * student_weight_bar,
    )
    multiplier_bar = tf.linalg.matvec(
        forward["scaled_matrix"],
        reference_weights * student_weight_bar,
    )
    residual_bar = tf.linalg.cholesky_solve(
        forward["gram_cholesky"], multiplier_bar[:, None]
    )[:, 0]
    gram_bar = -tf.einsum("i,j->ij", residual_bar, forward["multiplier"])
    scaled_target_bar = residual_bar
    matrix_bar -= tf.einsum("i,j->ij", residual_bar, reference_weights)
    matrix_bar += tf.linalg.matmul(
        gram_bar + tf.transpose(gram_bar),
        forward["scaled_matrix"] * reference_weights[None, :],
    )
    target_bar = scaled_target_bar / row_scale
    active_feature_bar += matrix_bar / row_scale[:, None]
    feature_bar = tf.einsum(
        "i,j->ij", target_bar, forward["normalized_weights"]
    )
    feature_bar += _scatter_active_columns(
        active_feature_bar, forward["active_indices"], point_count
    )
    normalized_weight_bar = tf.linalg.matvec(
        teacher_features, target_bar, transpose_a=True
    )
    normalized_weight_bar -= tf.tensordot(
        forward["normalized_weights"], normalized_weight_bar, axes=1
    )
    log_weight_bar = forward["normalized_weights"] * normalized_weight_bar
    log_weight_bar += upstream_log_normalizer * forward["normalized_weights"]
    valid = forward["valid_chart"]
    return {
        **forward,
        "teacher_points_bar": _poison_invalid(point_bar, valid),
        "log_unnormalized_weights_bar": _poison_invalid(log_weight_bar, valid),
        "teacher_features_bar": _poison_invalid(feature_bar, valid),
    }


def _contract_e_tp_dense_kkt_forward_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Apply the explicit overcomplete equality-constrained KKT chart."""

    _validate_teacher_inputs(teacher_points, log_unnormalized_weights, teacher_features)
    active_indices = _as_indices(active_indices)
    feature_count = tf.shape(teacher_features)[0]
    anchor_count = tf.size(active_indices)
    _validate_fixed_indices(active_indices, tf.shape(teacher_points)[0])
    tf.debugging.assert_greater_equal(anchor_count, feature_count)
    _validate_row_scale(row_scale, feature_count)
    tf.debugging.assert_shapes(
        [(reference_weights, ("m",)), (precision, ("m", "m"))]
    )
    tf.debugging.assert_equal(tf.size(reference_weights), anchor_count)
    tf.debugging.assert_all_finite(
        reference_weights, "reference weights must be finite"
    )
    tf.debugging.assert_all_finite(precision, "KKT precision must be finite")
    precision_scale = tf.maximum(
        tf.reduce_max(tf.abs(precision)), tf.constant(1.0, precision.dtype)
    )
    symmetry_roundoff_bound = (
        _dtype_epsilon(precision.dtype)
        * tf.cast(tf.shape(precision)[0], precision.dtype)
        * precision_scale
    )
    tf.debugging.assert_less_equal(
        tf.reduce_max(tf.abs(precision - tf.transpose(precision))),
        symmetry_roundoff_bound,
        message="KKT precision must be symmetric to dtype roundoff",
    )
    tf.debugging.assert_positive(
        tf.linalg.eigvalsh(precision), message="KKT precision must be positive definite"
    )
    precision_eigenvalues = tf.linalg.eigvalsh(precision)
    precision_scale = tf.maximum(
        tf.reduce_max(tf.abs(precision)), tf.constant(1.0, precision.dtype)
    )
    symmetry_roundoff_bound = (
        _dtype_epsilon(precision.dtype)
        * tf.cast(tf.shape(precision)[0], precision.dtype)
        * precision_scale
    )
    input_valid = (
        _teacher_input_validity(
            teacher_points, log_unnormalized_weights, teacher_features
        )
        & _fixed_index_validity(active_indices, tf.shape(teacher_points)[0])
        & tf.greater_equal(anchor_count, feature_count)
        & _row_scale_validity(row_scale, feature_count)
        & tf.reduce_all(tf.math.is_finite(reference_weights))
        & tf.reduce_all(tf.math.is_finite(precision))
        & tf.less_equal(
            tf.reduce_max(tf.abs(precision - tf.transpose(precision))),
            symmetry_roundoff_bound,
        )
        & tf.reduce_all(precision_eigenvalues > 0)
    )
    teacher = _dense_teacher_reduce_core(log_unnormalized_weights, teacher_features)
    active_features = tf.gather(teacher_features, active_indices, axis=1)
    scaled_matrix = active_features / row_scale[:, None]
    scaled_target = teacher["target"] / row_scale
    precision_inverse_features = tf.linalg.solve(
        precision, tf.transpose(scaled_matrix)
    )
    constraint_gram = tf.linalg.matmul(scaled_matrix, precision_inverse_features)
    constraint_residual = scaled_target - tf.linalg.matvec(
        scaled_matrix, reference_weights
    )
    multiplier = tf.linalg.solve(constraint_gram, constraint_residual[:, None])[:, 0]
    student_weights = reference_weights + tf.linalg.matvec(
        precision_inverse_features, multiplier
    )
    student_points = tf.gather(teacher_points, active_indices, axis=0)
    matched_target = tf.linalg.matvec(active_features, student_weights)
    diagnostics = _linear_chart_diagnostics(
        scaled_matrix, scaled_target, student_weights
    )
    _assert_valid_chart(diagnostics, student_weights)
    valid_chart = input_valid & diagnostics["valid_chart"]
    return {
        **teacher,
        **diagnostics,
        "input_valid": input_valid,
        "valid_chart": valid_chart,
        "active_indices": active_indices,
        "active_features": active_features,
        "scaled_matrix": scaled_matrix,
        "scaled_target": scaled_target,
        "reference_weights": reference_weights,
        "precision": precision,
        "precision_inverse_features": precision_inverse_features,
        "constraint_gram": constraint_gram,
        "constraint_residual": constraint_residual,
        "multiplier": multiplier,
        "student_points": _poison_invalid(student_points, valid_chart),
        "student_weights": _poison_invalid(student_weights, valid_chart),
        "matched_target": _poison_invalid(matched_target, valid_chart),
        "feature_residual": matched_target - teacher["target"],
        "row_scale": row_scale,
    }


def _contract_e_tp_dense_kkt_jvp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
    reference_weights_tangent: tf.Tensor,
    precision_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Explicit tangent of the dense fixed KKT chart."""

    forward = _contract_e_tp_dense_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        precision,
    )
    teacher_jvp = _dense_teacher_reduce_jvp_core(
        log_unnormalized_weights,
        teacher_features,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
    )
    active_features_tangent = tf.gather(
        teacher_features_tangent, forward["active_indices"], axis=1
    )
    matrix_tangent = active_features_tangent / row_scale[:, None]
    target_tangent = teacher_jvp["target_tangent"] / row_scale
    inverse_features_tangent = tf.linalg.solve(
        precision,
        tf.transpose(matrix_tangent)
        - tf.linalg.matmul(precision_tangent, forward["precision_inverse_features"]),
    )
    gram_tangent = (
        tf.linalg.matmul(matrix_tangent, forward["precision_inverse_features"])
        + tf.linalg.matmul(forward["scaled_matrix"], inverse_features_tangent)
    )
    residual_tangent = (
        target_tangent
        - tf.linalg.matvec(matrix_tangent, reference_weights)
        - tf.linalg.matvec(forward["scaled_matrix"], reference_weights_tangent)
    )
    multiplier_tangent = tf.linalg.solve(
        forward["constraint_gram"],
        (
            residual_tangent
            - tf.linalg.matvec(gram_tangent, forward["multiplier"])
        )[:, None],
    )[:, 0]
    student_weights_tangent = (
        reference_weights_tangent
        + tf.linalg.matvec(inverse_features_tangent, forward["multiplier"])
        + tf.linalg.matvec(forward["precision_inverse_features"], multiplier_tangent)
    )
    return {
        **forward,
        "log_normalizer_tangent": teacher_jvp["log_normalizer_tangent"],
        "normalized_weights_tangent": teacher_jvp["normalized_weights_tangent"],
        "target_tangent": teacher_jvp["target_tangent"],
        "active_features_tangent": active_features_tangent,
        "student_points_tangent": tf.gather(
            teacher_points_tangent, forward["active_indices"], axis=0
        ),
        "student_weights_tangent": student_weights_tangent,
        "matched_target_tangent": (
            tf.linalg.matvec(active_features_tangent, forward["student_weights"])
            + tf.linalg.matvec(forward["active_features"], student_weights_tangent)
        ),
    }


def _contract_e_tp_dense_kkt_vjp_core(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Manual transpose derivative of the overcomplete KKT chart."""

    forward = _contract_e_tp_dense_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        precision,
    )
    point_count = tf.shape(teacher_points)[0]
    point_bar = tf.scatter_nd(
        forward["active_indices"][:, None],
        upstream_student_points,
        tf.shape(teacher_points),
    )
    active_feature_bar = tf.einsum(
        "i,j->ij", upstream_matched_target, forward["student_weights"]
    )
    q_bar = upstream_student_weights + tf.linalg.matvec(
        forward["active_features"], upstream_matched_target, transpose_a=True
    )

    reference_bar = tf.identity(q_bar)
    inverse_features_bar = tf.einsum("i,j->ij", q_bar, forward["multiplier"])
    multiplier_bar = tf.linalg.matvec(
        forward["precision_inverse_features"], q_bar, transpose_a=True
    )
    residual_bar = tf.linalg.solve(
        tf.transpose(forward["constraint_gram"]), multiplier_bar[:, None]
    )[:, 0]
    gram_bar = -tf.einsum("i,j->ij", residual_bar, forward["multiplier"])
    scaled_target_bar = tf.identity(residual_bar)
    scaled_matrix_bar = -tf.einsum("i,j->ij", residual_bar, reference_weights)
    reference_bar -= tf.linalg.matvec(
        forward["scaled_matrix"], residual_bar, transpose_a=True
    )

    scaled_matrix_bar += tf.linalg.matmul(
        gram_bar, forward["precision_inverse_features"], transpose_b=True
    )
    inverse_features_bar += tf.linalg.matmul(
        forward["scaled_matrix"], gram_bar, transpose_a=True
    )
    solved_bar = tf.linalg.solve(
        tf.transpose(precision), inverse_features_bar
    )
    scaled_matrix_bar += tf.transpose(solved_bar)
    precision_bar = -tf.linalg.matmul(
        solved_bar, forward["precision_inverse_features"], transpose_b=True
    )

    target_bar = scaled_target_bar / row_scale
    active_feature_bar += scaled_matrix_bar / row_scale[:, None]
    feature_bar = tf.einsum(
        "i,j->ij", target_bar, forward["normalized_weights"]
    )
    feature_bar += _scatter_active_columns(
        active_feature_bar, forward["active_indices"], point_count
    )
    normalized_weight_bar = tf.linalg.matvec(
        teacher_features, target_bar, transpose_a=True
    )
    normalized_weight_bar -= tf.tensordot(
        forward["normalized_weights"], normalized_weight_bar, axes=1
    )
    log_weight_bar = forward["normalized_weights"] * normalized_weight_bar
    log_weight_bar += upstream_log_normalizer * forward["normalized_weights"]
    return {
        **forward,
        "teacher_points_bar": point_bar,
        "log_unnormalized_weights_bar": log_weight_bar,
        "teacher_features_bar": feature_bar,
        "reference_weights_bar": reference_bar,
        "precision_bar": precision_bar,
    }


def _streaming_block_mask(
    start: tf.Tensor, teacher_count: tf.Tensor, block_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    indices = start + tf.range(block_size, dtype=tf.int32)
    return indices, indices < teacher_count


def _streaming_first_pass(
    sources: tuple[tf.Tensor, ...],
    teacher_count: tf.Tensor,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> tf.Tensor:
    negative_infinity = tf.constant(float("-inf"), sources[0].dtype)

    def condition(start: tf.Tensor, maximum: tf.Tensor) -> tf.Tensor:
        del maximum
        return start < teacher_count

    def body(start: tf.Tensor, maximum: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        points, log_weights, features = block_program(sources, start)
        del points, features
        _, valid = _streaming_block_mask(start, teacher_count, block_size)
        block_maximum = tf.reduce_max(
            tf.where(valid, log_weights, negative_infinity)
        )
        return start + block_size, tf.maximum(maximum, block_maximum)

    _, maximum = tf.while_loop(
        condition,
        body,
        (tf.constant(0, tf.int32), negative_infinity),
        maximum_iterations=(teacher_count + block_size - 1) // block_size,
    )
    return tf.stop_gradient(maximum)


def _validate_streaming_block(
    points: tf.Tensor,
    log_weights: tf.Tensor,
    features: tf.Tensor,
    block_size: int,
    feature_count: tf.Tensor,
) -> None:
    tf.debugging.assert_rank(points, 2)
    tf.debugging.assert_rank(log_weights, 1)
    tf.debugging.assert_rank(features, 2)
    tf.debugging.assert_equal(tf.shape(points)[0], block_size)
    tf.debugging.assert_equal(tf.size(log_weights), block_size)
    tf.debugging.assert_equal(tf.shape(features)[0], feature_count)
    tf.debugging.assert_equal(tf.shape(features)[1], block_size)
    tf.debugging.assert_all_finite(points, "streamed teacher points must be finite")
    tf.debugging.assert_all_finite(
        log_weights, "streamed teacher log weights must be finite"
    )
    tf.debugging.assert_all_finite(features, "streamed teacher features must be finite")
    tf.debugging.assert_equal(
        features[0],
        tf.ones_like(features[0]),
        message="streamed feature row zero must be the exact mass feature",
    )


def _streaming_block_validity(
    points: tf.Tensor,
    log_weights: tf.Tensor,
    features: tf.Tensor,
) -> tf.Tensor:
    return (
        tf.reduce_all(tf.math.is_finite(points))
        & tf.reduce_all(tf.math.is_finite(log_weights))
        & tf.reduce_all(tf.math.is_finite(features))
        & tf.reduce_all(tf.equal(features[0], 1.0))
    )


def _contract_e_tp_streaming_square_forward_core(
    sources: tuple[tf.Tensor, ...],
    teacher_count: tf.Tensor | int,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> dict[str, tf.Tensor]:
    """Two-pass fixed-chart projection without retaining the dense teacher."""

    if not sources:
        raise ValueError("streaming projection requires at least one source tensor")
    teacher_count = tf.convert_to_tensor(teacher_count, tf.int32)
    tf.debugging.assert_positive(teacher_count)
    active_indices = _as_indices(active_indices)
    feature_count = tf.size(row_scale)
    _validate_fixed_indices(active_indices, teacher_count, feature_count)
    _validate_row_scale(row_scale, feature_count)
    fixed_input_valid = (
        _fixed_index_validity(active_indices, teacher_count, feature_count)
        & _row_scale_validity(row_scale, feature_count)
    )
    maximum = _streaming_first_pass(
        sources, teacher_count, block_size, block_program
    )
    anchor_count = tf.size(active_indices)
    first_points, _, _ = block_program(sources, tf.constant(0, tf.int32))
    state_dimension = tf.shape(first_points)[1]
    dtype = first_points.dtype
    initial_feature_sum = tf.zeros([feature_count], dtype)
    initial_active_points = tf.zeros([anchor_count, state_dimension], dtype)
    initial_active_features = tf.zeros([feature_count, anchor_count], dtype)
    initial_hits = tf.zeros([anchor_count], tf.int32)

    def condition(
        start: tf.Tensor,
        weight_sum: tf.Tensor,
        feature_sum: tf.Tensor,
        active_points: tf.Tensor,
        active_features: tf.Tensor,
        active_hits: tf.Tensor,
        blocks_valid: tf.Tensor,
    ) -> tf.Tensor:
        del weight_sum, feature_sum, active_points, active_features, active_hits
        del blocks_valid
        return start < teacher_count

    def body(
        start: tf.Tensor,
        weight_sum: tf.Tensor,
        feature_sum: tf.Tensor,
        active_points: tf.Tensor,
        active_features: tf.Tensor,
        active_hits: tf.Tensor,
        blocks_valid: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        points, log_weights, features = block_program(sources, start)
        _validate_streaming_block(
            points, log_weights, features, block_size, feature_count
        )
        indices, valid = _streaming_block_mask(start, teacher_count, block_size)
        scaled_weights = tf.where(
            valid, tf.exp(log_weights - maximum), tf.zeros_like(log_weights)
        )
        hits = tf.cast(
            tf.equal(indices[:, None], active_indices[None, :]) & valid[:, None],
            dtype,
        )
        return (
            start + block_size,
            weight_sum + tf.reduce_sum(scaled_weights),
            feature_sum + tf.linalg.matvec(features, scaled_weights),
            active_points + tf.linalg.matmul(hits, points, transpose_a=True),
            active_features + tf.linalg.matmul(features, hits),
            active_hits + tf.cast(tf.reduce_sum(hits, axis=0), tf.int32),
            blocks_valid & _streaming_block_validity(points, log_weights, features),
        )

    (
        _,
        weight_sum,
        feature_sum,
        active_points,
        active_features,
        active_hits,
        blocks_valid,
    ) = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            tf.constant(0.0, dtype),
            initial_feature_sum,
            initial_active_points,
            initial_active_features,
            initial_hits,
            tf.constant(True),
        ),
        maximum_iterations=(teacher_count + block_size - 1) // block_size,
    )
    tf.debugging.assert_equal(
        active_hits,
        tf.ones_like(active_hits),
        message="each frozen anchor must be encountered exactly once",
    )
    target = feature_sum / weight_sum
    log_normalizer = maximum + tf.math.log(weight_sum)
    scaled_matrix = active_features / row_scale[:, None]
    scaled_target = target / row_scale
    student_weights = tf.linalg.solve(scaled_matrix, scaled_target[:, None])[:, 0]
    matched_target = tf.linalg.matvec(active_features, student_weights)
    diagnostics = _linear_chart_diagnostics(
        scaled_matrix, scaled_target, student_weights
    )
    _assert_valid_chart(diagnostics, student_weights)
    anchors_valid = tf.reduce_all(tf.equal(active_hits, 1))
    input_valid = fixed_input_valid & blocks_valid & anchors_valid
    valid_chart = input_valid & diagnostics["valid_chart"]
    return {
        **diagnostics,
        "input_valid": input_valid,
        "blocks_valid": blocks_valid,
        "anchors_valid": anchors_valid,
        "valid_chart": valid_chart,
        "log_normalizer": log_normalizer,
        "normalization_maximum": maximum,
        "scaled_weight_sum": weight_sum,
        "target": target,
        "active_indices": active_indices,
        "active_features": active_features,
        "scaled_matrix": scaled_matrix,
        "scaled_target": scaled_target,
        "student_points": _poison_invalid(active_points, valid_chart),
        "student_weights": _poison_invalid(student_weights, valid_chart),
        "matched_target": _poison_invalid(matched_target, valid_chart),
        "feature_residual": matched_target - target,
        "row_scale": row_scale,
        "teacher_count": teacher_count,
        "block_size": tf.constant(block_size, tf.int32),
    }


def _contract_e_tp_streaming_square_jvp_core(
    sources: tuple[tf.Tensor, ...],
    source_tangents: tuple[tf.Tensor, ...],
    teacher_count: tf.Tensor | int,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> dict[str, tf.Tensor]:
    """Forward recomputation JVP for the two-pass streaming square chart."""

    if len(sources) != len(source_tangents):
        raise ValueError("every streaming source requires one tangent")
    forward = _contract_e_tp_streaming_square_forward_core(
        sources,
        teacher_count,
        active_indices,
        row_scale,
        block_size=block_size,
        block_program=block_program,
    )
    teacher_count = forward["teacher_count"]
    active_indices = forward["active_indices"]
    feature_count = tf.size(row_scale)
    anchor_count = tf.size(active_indices)
    state_dimension = tf.shape(forward["student_points"])[1]
    dtype = forward["student_points"].dtype

    def condition(
        start: tf.Tensor,
        weight_sum_tangent: tf.Tensor,
        feature_sum_tangent: tf.Tensor,
        active_points_tangent: tf.Tensor,
        active_features_tangent: tf.Tensor,
    ) -> tf.Tensor:
        del weight_sum_tangent, feature_sum_tangent
        del active_points_tangent, active_features_tangent
        return start < teacher_count

    def body(
        start: tf.Tensor,
        weight_sum_tangent: tf.Tensor,
        feature_sum_tangent: tf.Tensor,
        active_points_tangent: tf.Tensor,
        active_features_tangent: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        with tf.autodiff.ForwardAccumulator(sources, source_tangents) as accumulator:
            points, log_weights, features = block_program(sources, start)
        points_tangent = accumulator.jvp(points)
        log_weights_tangent = accumulator.jvp(log_weights)
        features_tangent = accumulator.jvp(features)
        indices, valid = _streaming_block_mask(start, teacher_count, block_size)
        scaled_weights = tf.where(
            valid,
            tf.exp(log_weights - forward["normalization_maximum"]),
            tf.zeros_like(log_weights),
        )
        scaled_weights_tangent = scaled_weights * log_weights_tangent
        hits = tf.cast(
            tf.equal(indices[:, None], active_indices[None, :]) & valid[:, None],
            dtype,
        )
        return (
            start + block_size,
            weight_sum_tangent + tf.reduce_sum(scaled_weights_tangent),
            feature_sum_tangent
            + tf.linalg.matvec(features_tangent, scaled_weights)
            + tf.linalg.matvec(features, scaled_weights_tangent),
            active_points_tangent
            + tf.linalg.matmul(hits, points_tangent, transpose_a=True),
            active_features_tangent + tf.linalg.matmul(features_tangent, hits),
        )

    (
        _,
        weight_sum_tangent,
        feature_sum_tangent,
        active_points_tangent,
        active_features_tangent,
    ) = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            tf.constant(0.0, dtype),
            tf.zeros([feature_count], dtype),
            tf.zeros([anchor_count, state_dimension], dtype),
            tf.zeros([feature_count, anchor_count], dtype),
        ),
        maximum_iterations=(teacher_count + block_size - 1) // block_size,
    )
    log_normalizer_tangent = weight_sum_tangent / forward["scaled_weight_sum"]
    target_tangent = (
        feature_sum_tangent
        - forward["target"] * weight_sum_tangent
    ) / forward["scaled_weight_sum"]
    matrix_tangent = active_features_tangent / row_scale[:, None]
    scaled_target_tangent = target_tangent / row_scale
    student_weights_tangent = tf.linalg.solve(
        forward["scaled_matrix"],
        (
            scaled_target_tangent
            - tf.linalg.matvec(matrix_tangent, forward["student_weights"])
        )[:, None],
    )[:, 0]
    matched_target_tangent = (
        tf.linalg.matvec(active_features_tangent, forward["student_weights"])
        + tf.linalg.matvec(forward["active_features"], student_weights_tangent)
    )
    return {
        **forward,
        "log_normalizer_tangent": log_normalizer_tangent,
        "target_tangent": target_tangent,
        "student_points_tangent": active_points_tangent,
        "active_features_tangent": active_features_tangent,
        "student_weights_tangent": student_weights_tangent,
        "matched_target_tangent": matched_target_tangent,
    }


def _contract_e_tp_streaming_square_vjp_core(
    sources: tuple[tf.Tensor, ...],
    teacher_count: tf.Tensor | int,
    active_indices: tf.Tensor | tuple[int, ...],
    row_scale: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> dict[str, tf.Tensor]:
    """Transpose recomputation with one accumulated adjoint per source."""

    forward = _contract_e_tp_streaming_square_forward_core(
        sources,
        teacher_count,
        active_indices,
        row_scale,
        block_size=block_size,
        block_program=block_program,
    )
    q_bar = upstream_student_weights + tf.linalg.matvec(
        forward["active_features"], upstream_matched_target, transpose_a=True
    )
    scaled_target_bar = tf.linalg.solve(
        tf.transpose(forward["scaled_matrix"]), q_bar[:, None]
    )[:, 0]
    scaled_matrix_bar = -tf.einsum(
        "i,j->ij", scaled_target_bar, forward["student_weights"]
    )
    target_bar = scaled_target_bar / row_scale
    active_feature_bar = (
        tf.einsum(
            "i,j->ij", upstream_matched_target, forward["student_weights"]
        )
        + scaled_matrix_bar / row_scale[:, None]
    )
    teacher_count = forward["teacher_count"]
    active_indices = forward["active_indices"]

    def condition(start: tf.Tensor, *source_bars: tf.Tensor) -> tf.Tensor:
        del source_bars
        return start < teacher_count

    def body(start: tf.Tensor, *source_bars: tf.Tensor) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape() as tape:
            tape.watch(sources)
            points, log_weights, features = block_program(sources, start)
            indices, valid = _streaming_block_mask(start, teacher_count, block_size)
            scaled_weights = tf.where(
                valid,
                tf.exp(log_weights - forward["normalization_maximum"]),
                tf.zeros_like(log_weights),
            )
            normalized_weights = scaled_weights / forward["scaled_weight_sum"]
            feature_value_bar = tf.linalg.matvec(
                features, target_bar, transpose_a=True
            )
            log_weight_bar = normalized_weights * (
                feature_value_bar
                - tf.tensordot(forward["target"], target_bar, axes=1)
                + upstream_log_normalizer
            )
            feature_bar = tf.einsum("i,j->ij", target_bar, normalized_weights)
            hits = tf.cast(
                tf.equal(indices[:, None], active_indices[None, :])
                & valid[:, None],
                points.dtype,
            )
            point_bar = tf.linalg.matmul(hits, upstream_student_points)
            feature_bar += tf.linalg.matmul(
                active_feature_bar, hits, transpose_b=True
            )
            objective = (
                tf.reduce_sum(points * tf.stop_gradient(point_bar))
                + tf.reduce_sum(log_weights * tf.stop_gradient(log_weight_bar))
                + tf.reduce_sum(features * tf.stop_gradient(feature_bar))
            )
        block_source_bars = tape.gradient(
            objective, sources, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        return (start + block_size,) + tuple(
            accumulated + block
            for accumulated, block in zip(source_bars, block_source_bars, strict=True)
        )

    loop_result = tf.while_loop(
        condition,
        body,
        (tf.constant(0, tf.int32),) + tuple(tf.zeros_like(value) for value in sources),
        maximum_iterations=(teacher_count + block_size - 1) // block_size,
    )
    return {
        **forward,
        "source_bars": loop_result[1:],
        "target_bar": target_bar,
        "active_features_bar": active_feature_bar,
    }


def make_contract_e_tp_streaming_square_forward_tf(
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> Callable[
    [tuple[tf.Tensor, ...], tf.Tensor, tf.Tensor, tf.Tensor],
    dict[str, tf.Tensor],
]:
    """Bind a fixed block program into an XLA-default streaming evaluator."""

    if block_size <= 0:
        raise ValueError("streaming block size must be positive")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(
        sources: tuple[tf.Tensor, ...],
        teacher_count: tf.Tensor,
        active_indices: tf.Tensor,
        row_scale: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return _contract_e_tp_streaming_square_forward_core(
            sources,
            teacher_count,
            active_indices,
            row_scale,
            block_size=block_size,
            block_program=block_program,
        )

    return evaluate


def make_contract_e_tp_streaming_square_jvp_tf(
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> Callable[..., dict[str, tf.Tensor]]:
    """Bind a fixed block program into an XLA-default streaming JVP."""

    if block_size <= 0:
        raise ValueError("streaming block size must be positive")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(
        sources: tuple[tf.Tensor, ...],
        source_tangents: tuple[tf.Tensor, ...],
        teacher_count: tf.Tensor,
        active_indices: tf.Tensor,
        row_scale: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return _contract_e_tp_streaming_square_jvp_core(
            sources,
            source_tangents,
            teacher_count,
            active_indices,
            row_scale,
            block_size=block_size,
            block_program=block_program,
        )

    return evaluate


def make_contract_e_tp_streaming_square_vjp_tf(
    *,
    block_size: int,
    block_program: StreamingBlockProgram,
) -> Callable[..., dict[str, tf.Tensor]]:
    """Bind a fixed block program into an XLA-default streaming VJP."""

    if block_size <= 0:
        raise ValueError("streaming block size must be positive")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(
        sources: tuple[tf.Tensor, ...],
        teacher_count: tf.Tensor,
        active_indices: tf.Tensor,
        row_scale: tf.Tensor,
        upstream_student_points: tf.Tensor,
        upstream_student_weights: tf.Tensor,
        upstream_matched_target: tf.Tensor,
        upstream_log_normalizer: tf.Tensor,
    ) -> dict[str, tf.Tensor]:
        return _contract_e_tp_streaming_square_vjp_core(
            sources,
            teacher_count,
            active_indices,
            row_scale,
            upstream_student_points,
            upstream_student_weights,
            upstream_matched_target,
            upstream_log_normalizer,
            block_size=block_size,
            block_program=block_program,
        )

    return evaluate


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_square_forward_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default public wrapper for the experimental dense square chart."""

    return _contract_e_tp_dense_square_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_square_jvp_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default explicit JVP for the dense square chart."""

    return _contract_e_tp_dense_square_jvp_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        teacher_points_tangent,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_square_vjp_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default explicit VJP for the dense square chart."""

    result = _contract_e_tp_dense_square_vjp_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        upstream_student_points,
        upstream_student_weights,
        upstream_matched_target,
        upstream_log_normalizer,
    )
    return {
        "teacher_points": result["teacher_points_bar"],
        "log_unnormalized_weights": result["log_unnormalized_weights_bar"],
        "teacher_features": result["teacher_features_bar"],
    }


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_kkt_forward_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default public wrapper for the experimental dense KKT chart."""

    return _contract_e_tp_dense_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        precision,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_diagonal_kkt_forward_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default nonthrowing Pearson chart with a diagonal precision."""

    return _contract_e_tp_diagonal_kkt_forward_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_diagonal_kkt_jvp_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default explicit JVP of the frozen-reference Pearson chart."""

    return _contract_e_tp_diagonal_kkt_jvp_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        teacher_points_tangent,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_kkt_jvp_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
    teacher_points_tangent: tf.Tensor,
    log_unnormalized_weights_tangent: tf.Tensor,
    teacher_features_tangent: tf.Tensor,
    reference_weights_tangent: tf.Tensor,
    precision_tangent: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default explicit JVP for the dense KKT chart."""

    return _contract_e_tp_dense_kkt_jvp_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        precision,
        teacher_points_tangent,
        log_unnormalized_weights_tangent,
        teacher_features_tangent,
        reference_weights_tangent,
        precision_tangent,
    )


@tf.function(jit_compile=True, reduce_retracing=True)
def contract_e_tp_dense_kkt_vjp_tf(
    teacher_points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    teacher_features: tf.Tensor,
    active_indices: tf.Tensor,
    row_scale: tf.Tensor,
    reference_weights: tf.Tensor,
    precision: tf.Tensor,
    upstream_student_points: tf.Tensor,
    upstream_student_weights: tf.Tensor,
    upstream_matched_target: tf.Tensor,
    upstream_log_normalizer: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """XLA-default explicit VJP for the dense KKT chart."""

    result = _contract_e_tp_dense_kkt_vjp_core(
        teacher_points,
        log_unnormalized_weights,
        teacher_features,
        active_indices,
        row_scale,
        reference_weights,
        precision,
        upstream_student_points,
        upstream_student_weights,
        upstream_matched_target,
        upstream_log_normalizer,
    )
    return {
        "teacher_points": result["teacher_points_bar"],
        "log_unnormalized_weights": result["log_unnormalized_weights_bar"],
        "teacher_features": result["teacher_features_bar"],
        "reference_weights": result["reference_weights_bar"],
        "precision": result["precision_bar"],
    }
