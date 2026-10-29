"""Model-agnostic TensorFlow diagnostics for finite-filter score errors."""

from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf


class TFScoreComparisonDiagnostics(NamedTuple):
    """Score-error diagnostics in one declared parameter coordinate system."""

    reference_score: tf.Tensor
    reference_increment_sum: tf.Tensor
    score_error: tf.Tensor
    absolute_error_norm: tf.Tensor
    reference_score_norm: tf.Tensor
    increment_energy: tf.Tensor
    relative_total_score_norm_error: tf.Tensor
    relative_increment_energy_error: tf.Tensor
    average_opg: tf.Tensor
    average_opg_eigenvalues: tf.Tensor
    shrunk_average_opg: tf.Tensor
    realized_ridge: tf.Tensor
    ridge_floor_active: tf.Tensor
    ridge_scale_diagonal: tf.Tensor
    average_metric: tf.Tensor
    total_metric: tf.Tensor
    total_metric_eigenvalues: tf.Tensor
    total_metric_condition_proxy: tf.Tensor
    rms_total_metric_error: tf.Tensor
    maximum_diagonal_standardized_error: tf.Tensor


def _require_float_tensor(value: tf.Tensor, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value)
    if not tensor.dtype.is_floating:
        raise TypeError(f"{name} must have floating dtype, got {tensor.dtype.name}")
    return tensor


def _undefined_safe_ratio(numerator: tf.Tensor, denominator: tf.Tensor) -> tf.Tensor:
    nan = tf.constant(float("nan"), numerator.dtype)
    return tf.where(denominator > 0.0, numerator / denominator, nan)


def tf_score_comparison_diagnostics(
    *,
    candidate_score: tf.Tensor,
    reference_score_increments: tf.Tensor,
    diagonal_shrinkage: tf.Tensor | float,
    base_ridge: tf.Tensor | float,
    ridge_floor: tf.Tensor | float,
    ridge_scale_diagonal: tf.Tensor,
) -> TFScoreComparisonDiagnostics:
    """Compute score norm and regularized average-OPG diagnostics.

    `candidate_score` has shape `[..., p]`; every leading candidate is compared
    with the same reference. `reference_score_increments` has shape `[T, p]`.
    All inputs must already use the same declared parameter coordinates.
    """

    increments = _require_float_tensor(
        reference_score_increments, "reference_score_increments"
    )
    if increments.shape.rank != 2:
        raise ValueError("reference_score_increments must have shape [T, p]")
    if increments.shape[-1] is None:
        raise ValueError("reference score parameter count must be statically known")
    candidate = tf.convert_to_tensor(candidate_score, dtype=increments.dtype)
    if candidate.shape.rank is None or candidate.shape.rank < 1:
        raise ValueError("candidate_score must have shape [..., p]")
    if candidate.shape[-1] not in (None, increments.shape[-1]):
        raise ValueError("candidate and reference parameter counts differ")
    ridge_scale = tf.convert_to_tensor(
        ridge_scale_diagonal, dtype=increments.dtype
    )
    if ridge_scale.shape.rank != 1:
        raise ValueError("ridge_scale_diagonal must have shape [p]")
    if ridge_scale.shape[-1] not in (None, increments.shape[-1]):
        raise ValueError("ridge scale and reference parameter counts differ")

    shrinkage = tf.convert_to_tensor(diagonal_shrinkage, dtype=increments.dtype)
    epsilon0 = tf.convert_to_tensor(base_ridge, dtype=increments.dtype)
    epsilon_min = tf.convert_to_tensor(ridge_floor, dtype=increments.dtype)
    for value, name in (
        (shrinkage, "diagonal_shrinkage"),
        (epsilon0, "base_ridge"),
        (epsilon_min, "ridge_floor"),
    ):
        if value.shape.rank != 0:
            raise ValueError(f"{name} must be scalar")

    checks = (
        tf.debugging.assert_positive(
            tf.shape(increments)[0], message="at least one score increment is required"
        ),
        tf.debugging.assert_all_finite(increments, "score increments must be finite"),
        tf.debugging.assert_all_finite(candidate, "candidate score must be finite"),
        tf.debugging.assert_equal(
            tf.shape(candidate)[-1],
            tf.shape(increments)[-1],
            message="candidate and reference parameter counts differ",
        ),
        tf.debugging.assert_equal(
            tf.shape(ridge_scale)[0],
            tf.shape(increments)[-1],
            message="ridge scale and reference parameter counts differ",
        ),
        tf.debugging.assert_greater_equal(
            shrinkage,
            tf.constant(0.0, increments.dtype),
            message="diagonal_shrinkage must be in [0, 1]",
        ),
        tf.debugging.assert_less_equal(
            shrinkage,
            tf.constant(1.0, increments.dtype),
            message="diagonal_shrinkage must be in [0, 1]",
        ),
        tf.debugging.assert_positive(
            epsilon0, message="base_ridge must be positive"
        ),
        tf.debugging.assert_non_negative(
            epsilon_min, message="ridge_floor must be nonnegative"
        ),
        tf.debugging.assert_positive(
            ridge_scale, message="ridge_scale_diagonal must be positive"
        ),
        tf.debugging.assert_all_finite(
            ridge_scale, "ridge_scale_diagonal must be finite"
        ),
    )
    with tf.control_dependencies(checks):
        increments = tf.identity(increments)
        candidate = tf.identity(candidate)
        ridge_scale = tf.identity(ridge_scale)

    horizon = tf.cast(tf.shape(increments)[0], increments.dtype)
    parameter_count = tf.cast(tf.shape(increments)[1], increments.dtype)
    reference = tf.reduce_sum(increments, axis=0)
    error = candidate - reference
    absolute_error_norm = tf.linalg.norm(error, axis=-1)
    reference_norm = tf.linalg.norm(reference)
    increment_energy = tf.linalg.norm(increments)
    average_opg = tf.einsum("tp,tq->pq", increments, increments) / horizon
    diagonal_opg = tf.linalg.diag(tf.linalg.diag_part(average_opg))
    shrunk_average_opg = (
        (1.0 - shrinkage) * average_opg + shrinkage * diagonal_opg
    )
    vanishing_ridge = epsilon0 / horizon
    realized_ridge = tf.maximum(vanishing_ridge, epsilon_min)
    average_metric = shrunk_average_opg + realized_ridge * tf.linalg.diag(
        ridge_scale
    )
    total_metric = horizon * average_metric
    solved_error = tf.linalg.cholesky_solve(
        tf.linalg.cholesky(total_metric), tf.linalg.matrix_transpose(error[..., None, :])
    )
    quadratic = tf.reduce_sum(
        tf.linalg.matrix_transpose(error[..., None, :]) * solved_error,
        axis=[-2, -1],
    )
    rms_metric_error = tf.sqrt(tf.maximum(quadratic, 0.0) / parameter_count)
    diagonal_standardized = tf.abs(error) / tf.sqrt(tf.linalg.diag_part(total_metric))

    total_metric_eigenvalues = tf.linalg.eigvalsh(total_metric)
    return TFScoreComparisonDiagnostics(
        reference_score=reference,
        reference_increment_sum=reference,
        score_error=error,
        absolute_error_norm=absolute_error_norm,
        reference_score_norm=reference_norm,
        increment_energy=increment_energy,
        relative_total_score_norm_error=_undefined_safe_ratio(
            absolute_error_norm, reference_norm
        ),
        relative_increment_energy_error=_undefined_safe_ratio(
            absolute_error_norm, increment_energy
        ),
        average_opg=average_opg,
        average_opg_eigenvalues=tf.linalg.eigvalsh(average_opg),
        shrunk_average_opg=shrunk_average_opg,
        realized_ridge=realized_ridge,
        ridge_floor_active=epsilon_min > vanishing_ridge,
        ridge_scale_diagonal=ridge_scale,
        average_metric=average_metric,
        total_metric=total_metric,
        total_metric_eigenvalues=total_metric_eigenvalues,
        total_metric_condition_proxy=(
            total_metric_eigenvalues[-1] / total_metric_eigenvalues[0]
        ),
        rms_total_metric_error=rms_metric_error,
        maximum_diagonal_standardized_error=tf.reduce_max(
            diagonal_standardized, axis=-1
        ),
    )


__all__ = ["TFScoreComparisonDiagnostics", "tf_score_comparison_diagnostics"]
