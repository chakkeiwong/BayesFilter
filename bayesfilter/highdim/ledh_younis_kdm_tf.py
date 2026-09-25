"""Diagnostic Younis-style Gaussian KDM algebra for the LEDH score study.

This module is deliberately not a canonical LEDH route.  It implements the
continuous mixture and fixed-proposal IWSG algebra needed to test whether a
KDM can help the score calculation.  The conditional proposal correction is a
BayesFilter extension: it retains the model transition/observation factors
and the LEDH forward Jacobian in a PF-PF importance ratio.

All numerical kernels are TensorFlow functions with static signatures.  The
all-pairs mixture work is explicit and reported as diagnostic-only; callers
must not use this module as an admitted production lane.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from inspect import signature as callable_signature
from typing import Any

import tensorflow as tf

Tensor = tf.Tensor

ROUTE_ID = "ledh_younis_kdm_algebra_diagnostic_v1"
ROUTE_CLASSIFICATION = "source_aligned_extension_diagnostic_only"
ANCHORED_MODEL_IS_ROUTE_ID = "ledh_younis_kdm_anchored_model_is_diagnostic_v1"
FULL_MIXTURE_RESAMPLING_ROUTE_ID = (
    "ledh_younis_kdm_full_mixture_resampling_reference_v1"
)
ATOM_FINITE_TARGET = "ATOM-FINITE"
KDM_FINITE_TARGET = "KDM-FINITE"
RESKDM_IWSG_FINITE_TARGET = "RESKDM-IWSG-FINITE"
RESKDM_SELF_NORMALIZED_FINITE_TARGET = "RESKDM-SN-FINITE"
MODEL_IS_TARGET = "MODEL-IS"
AUXILIARY_ROUTE_ID = "ledh_younis_kdm_auxiliary_no_feedback_v1"
AUXILIARY_ROLE = "diagnostic_auxiliary_frozen_canonical_trajectory"


def _dtype_from_value(dtype: tf.dtypes.DType | str) -> tf.dtypes.DType:
    result = tf.as_dtype(dtype)
    if not result.is_floating:
        raise ValueError("KDM kernels require a floating TensorFlow dtype")
    return result


def _static_positive(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_rows(value: Tensor) -> Tensor:
    rank = value.shape.rank
    if rank is None or rank < 1:
        raise ValueError("KDM tensors require a statically known rank")
    if rank == 1:
        return tf.math.is_finite(value)
    return tf.reduce_all(tf.math.is_finite(value), axis=list(range(1, rank)))


def _finite_tangents_per_row(value: Tensor) -> Tensor:
    rank = value.shape.rank
    if rank is None or rank < 2:
        raise ValueError("KDM tangents require direction and row axes")
    axes = [0, *range(2, rank)]
    return tf.reduce_all(tf.math.is_finite(value), axis=axes)


def _batched_gaussian_mixture_core(
    values: Tensor,
    weights: Tensor,
    means: Tensor,
    covariances: Tensor,
    d_values: Tensor,
    d_weights: Tensor,
    d_means: Tensor,
    d_covariances: Tensor,
    *,
    rank_tolerance: Tensor,
    normalization_tolerance: Tensor,
) -> Mapping[str, Tensor]:
    """Evaluate one mixture row per value and its complete fixed-value tangent.

    Shapes are ``values [M,D]``, ``weights [M,N]``, ``means [M,N,D]``, and
    ``covariances [M,N,D,D]``.  Tangents have a leading direction axis.  The
    evaluation locations are fixed unless ``d_values`` is nonzero.
    """

    values = tf.convert_to_tensor(values)
    weights = tf.convert_to_tensor(weights)
    means = tf.convert_to_tensor(means)
    covariances = tf.convert_to_tensor(covariances)
    d_values = tf.convert_to_tensor(d_values)
    d_weights = tf.convert_to_tensor(d_weights)
    d_means = tf.convert_to_tensor(d_means)
    d_covariances = tf.convert_to_tensor(d_covariances)

    row_count = tf.shape(values)[0]
    component_count = tf.shape(means)[1]
    dimension = tf.shape(values)[1]

    symmetric_covariances = 0.5 * (
        covariances + tf.linalg.matrix_transpose(covariances)
    )
    scale = tf.maximum(
        tf.reduce_max(tf.abs(symmetric_covariances), axis=[-2, -1]),
        tf.ones_like(tf.reduce_max(tf.abs(symmetric_covariances), axis=[-2, -1])),
    )
    eigenvalues = tf.linalg.eigvalsh(symmetric_covariances)
    minimum_eigenvalue = tf.reduce_min(eigenvalues, axis=-1)
    covariance_valid = minimum_eigenvalue > rank_tolerance * scale
    finite_inputs = (
        _finite_rows(values)
        & _finite_rows(weights)
        & _finite_rows(means)
        & _finite_rows(covariances)
        & _finite_tangents_per_row(d_values)
        & _finite_tangents_per_row(d_weights)
        & _finite_tangents_per_row(d_means)
        & _finite_tangents_per_row(d_covariances)
    )
    positive_weights = tf.reduce_all(weights > tf.zeros_like(weights), axis=1)
    weight_sums = tf.reduce_sum(weights, axis=1)
    normalized_weights = tf.abs(weight_sums - tf.ones_like(weight_sums)) <= (
        normalization_tolerance
    )
    normalized_weight_tangents = tf.reduce_all(
        tf.abs(tf.reduce_sum(d_weights, axis=2)) <= normalization_tolerance,
        axis=0,
    )
    row_valid = (
        finite_inputs
        & positive_weights
        & normalized_weights
        & normalized_weight_tangents
        & tf.reduce_all(covariance_valid, axis=1)
    )

    # Invalid rows use an identity covariance only to keep compiled arithmetic
    # finite; the returned validity flag remains false and callers fail closed.
    dimension_value = tf.shape(covariances)[-1]
    identity = tf.eye(dimension_value, dtype=values.dtype)
    safe_covariances = tf.where(
        covariance_valid[..., tf.newaxis, tf.newaxis],
        symmetric_covariances,
        identity[tf.newaxis, tf.newaxis, :, :],
    )
    factors = tf.linalg.cholesky(safe_covariances)
    residual = values[:, tf.newaxis, :] - means
    solved = tf.linalg.triangular_solve(factors, residual[..., tf.newaxis], lower=True)[
        ..., 0
    ]
    precision_residual = tf.linalg.cholesky_solve(factors, residual[..., tf.newaxis])[
        ..., 0
    ]
    quadratic = tf.reduce_sum(tf.square(solved), axis=-1)
    log_determinant = 2.0 * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(factors)), axis=-1
    )
    log_two_pi = tf.constant(math.log(2.0 * math.pi), values.dtype)
    log_normalizer = tf.cast(dimension, values.dtype) * log_two_pi + log_determinant
    component_log_density = -0.5 * (quadratic + log_normalizer)
    joint_log_density = tf.math.log(weights) + component_log_density
    log_density = tf.reduce_logsumexp(joint_log_density, axis=1)
    responsibilities = tf.exp(joint_log_density - log_density[:, tf.newaxis])

    # d log k(z; x, B) = r^T (dx - dz) +
    # 1/2 [r^T dB r - tr(B^{-1} dB)], r=B^{-1}(z-x).
    symmetric_d_covariances = 0.5 * (
        d_covariances + tf.linalg.matrix_transpose(d_covariances)
    )
    mean_displacement = d_means - d_values[:, :, tf.newaxis, :]
    mean_term = tf.einsum("mnd,kmnd->kmn", precision_residual, mean_displacement)
    covariance_quadratic = tf.einsum(
        "mnd,kmnde,mne->kmn",
        precision_residual,
        symmetric_d_covariances,
        precision_residual,
    )
    d_covariances_mnkd = tf.transpose(symmetric_d_covariances, [1, 2, 0, 3, 4])
    solved_covariances = tf.linalg.cholesky_solve(
        factors[:, :, tf.newaxis, :, :], d_covariances_mnkd
    )
    covariance_trace = tf.linalg.trace(solved_covariances)
    covariance_trace = tf.transpose(covariance_trace, [2, 0, 1])
    d_log_kernel = mean_term + 0.5 * (covariance_quadratic - covariance_trace)
    d_log_weights = d_weights / weights[tf.newaxis, :, :]
    d_log_component = d_log_weights + d_log_kernel
    d_log_density = tf.reduce_sum(
        responsibilities[tf.newaxis, :, :] * d_log_component, axis=2
    )

    nan_density = tf.fill(tf.shape(log_density), tf.cast(float("nan"), values.dtype))
    nan_tangent = tf.fill(tf.shape(d_log_density), tf.cast(float("nan"), values.dtype))
    finite_log_density = tf.where(row_valid, log_density, nan_density)
    finite_responsibilities = tf.where(
        row_valid[:, tf.newaxis],
        responsibilities,
        tf.fill(tf.shape(responsibilities), tf.cast(float("nan"), values.dtype)),
    )
    finite_tangent = tf.where(row_valid[tf.newaxis, :], d_log_density, nan_tangent)
    return {
        "log_density": finite_log_density,
        "responsibilities": finite_responsibilities,
        "d_log_density": finite_tangent,
        "d_log_component": tf.where(
            row_valid[tf.newaxis, :, tf.newaxis],
            d_log_component,
            tf.fill(
                tf.shape(d_log_component),
                tf.cast(float("nan"), values.dtype),
            ),
        ),
        "component_log_density": tf.where(
            row_valid[:, tf.newaxis],
            component_log_density,
            tf.fill(
                tf.shape(component_log_density), tf.cast(float("nan"), values.dtype)
            ),
        ),
        "minimum_eigenvalue": tf.reduce_min(minimum_eigenvalue, axis=1),
        "valid": row_valid,
        "complexity_pair_count": tf.cast(row_count * component_count, tf.int32),
    }


def make_gaussian_kdm_kernel(
    *,
    evaluation_count: int,
    component_count: int,
    dimension: int,
    direction_count: int = 1,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create a fixed-shape global KDM evaluator and analytical tangent."""

    evaluation_count = _static_positive("evaluation_count", evaluation_count)
    component_count = _static_positive("component_count", component_count)
    dimension = _static_positive("dimension", dimension)
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if float(rank_tolerance) <= 0.0 or float(normalization_tolerance) <= 0.0:
        raise ValueError("tolerances must be positive")
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)

    @tf.function(
        input_signature=[
            tf.TensorSpec([evaluation_count, dimension], dtype),
            tf.TensorSpec([component_count], dtype),
            tf.TensorSpec([component_count, dimension], dtype),
            tf.TensorSpec([component_count, dimension, dimension], dtype),
            tf.TensorSpec([direction_count, evaluation_count, dimension], dtype),
            tf.TensorSpec([direction_count, component_count], dtype),
            tf.TensorSpec([direction_count, component_count, dimension], dtype),
            tf.TensorSpec(
                [direction_count, component_count, dimension, dimension], dtype
            ),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        values: Tensor,
        weights: Tensor,
        means: Tensor,
        covariances: Tensor,
        d_values: Tensor,
        d_weights: Tensor,
        d_means: Tensor,
        d_covariances: Tensor,
    ) -> Mapping[str, Tensor]:
        weights_b = tf.broadcast_to(
            weights[tf.newaxis, :], [evaluation_count, component_count]
        )
        means_b = tf.broadcast_to(
            means[tf.newaxis, :, :], [evaluation_count, component_count, dimension]
        )
        covariances_b = tf.broadcast_to(
            covariances[tf.newaxis, :, :, :],
            [evaluation_count, component_count, dimension, dimension],
        )
        d_weights_b = tf.broadcast_to(
            d_weights[:, tf.newaxis, :],
            [direction_count, evaluation_count, component_count],
        )
        d_means_b = tf.broadcast_to(
            d_means[:, tf.newaxis, :, :],
            [direction_count, evaluation_count, component_count, dimension],
        )
        d_covariances_b = tf.broadcast_to(
            d_covariances[:, tf.newaxis, :, :, :],
            [direction_count, evaluation_count, component_count, dimension, dimension],
        )
        return _batched_gaussian_mixture_core(
            values,
            weights_b,
            means_b,
            covariances_b,
            d_values,
            d_weights_b,
            d_means_b,
            d_covariances_b,
            rank_tolerance=rank_tolerance_tensor,
            normalization_tolerance=normalization_tolerance_tensor,
        )

    kernel.route_id = ROUTE_ID
    kernel.target_label = KDM_FINITE_TARGET
    return kernel


def make_full_mixture_iwsg_resampling_kernel(
    *,
    particle_count: int,
    dimension: int,
    direction_count: int = 1,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    symmetry_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create the exact full-mixture fixed-anchor IWSG resampling primitive.

    The fixed samples have no tangent input. Each sample is evaluated against
    every component; the supplied proposal density is an anchor constant. UKF
    covariance marks are transported by the complete mixture responsibilities,
    including both responsibility and source-mark tangents.
    """

    particle_count = _static_positive("particle_count", particle_count)
    dimension = _static_positive("dimension", dimension)
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if (
        float(rank_tolerance) <= 0.0
        or float(normalization_tolerance) <= 0.0
        or float(symmetry_tolerance) <= 0.0
    ):
        raise ValueError("resampling tolerances must be positive")
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)
    symmetry_tolerance_tensor = tf.constant(float(symmetry_tolerance), dtype)

    @tf.function(
        input_signature=[
            tf.TensorSpec([particle_count, dimension], dtype),
            tf.TensorSpec([particle_count], dtype),
            tf.TensorSpec([particle_count, dimension], dtype),
            tf.TensorSpec([particle_count, dimension, dimension], dtype),
            tf.TensorSpec([particle_count, dimension, dimension], dtype),
            tf.TensorSpec([particle_count], dtype),
            tf.TensorSpec([direction_count, particle_count], dtype),
            tf.TensorSpec([direction_count, particle_count, dimension], dtype),
            tf.TensorSpec(
                [direction_count, particle_count, dimension, dimension], dtype
            ),
            tf.TensorSpec(
                [direction_count, particle_count, dimension, dimension], dtype
            ),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        samples: Tensor,
        component_weights: Tensor,
        component_means: Tensor,
        bandwidth_covariances: Tensor,
        covariance_marks: Tensor,
        proposal_log_density: Tensor,
        d_component_weights: Tensor,
        d_component_means: Tensor,
        d_bandwidth_covariances: Tensor,
        d_covariance_marks: Tensor,
    ) -> Mapping[str, Tensor]:
        weights_b = tf.broadcast_to(
            component_weights[tf.newaxis, :],
            [particle_count, particle_count],
        )
        means_b = tf.broadcast_to(
            component_means[tf.newaxis, :, :],
            [particle_count, particle_count, dimension],
        )
        bandwidths_b = tf.broadcast_to(
            bandwidth_covariances[tf.newaxis, :, :, :],
            [particle_count, particle_count, dimension, dimension],
        )
        d_weights_b = tf.broadcast_to(
            d_component_weights[:, tf.newaxis, :],
            [direction_count, particle_count, particle_count],
        )
        d_means_b = tf.broadcast_to(
            d_component_means[:, tf.newaxis, :, :],
            [direction_count, particle_count, particle_count, dimension],
        )
        d_bandwidths_b = tf.broadcast_to(
            d_bandwidth_covariances[:, tf.newaxis, :, :, :],
            [
                direction_count,
                particle_count,
                particle_count,
                dimension,
                dimension,
            ],
        )
        density = _batched_gaussian_mixture_core(
            samples,
            weights_b,
            means_b,
            bandwidths_b,
            tf.zeros([direction_count, particle_count, dimension], dtype),
            d_weights_b,
            d_means_b,
            d_bandwidths_b,
            rank_tolerance=rank_tolerance_tensor,
            normalization_tolerance=normalization_tolerance_tensor,
        )

        log_ratio = density["log_density"] - proposal_log_density
        log_particle_count = tf.math.log(tf.cast(particle_count, dtype))
        importance_log_weights = log_ratio - log_particle_count
        importance_weights = tf.exp(importance_log_weights)
        d_importance_log_weights = density["d_log_density"]
        d_importance_weights = (
            importance_weights[tf.newaxis, :] * d_importance_log_weights
        )
        log_weight_normalizer = tf.reduce_logsumexp(log_ratio)
        normalized_log_weights = log_ratio - log_weight_normalizer
        normalized_weights = tf.exp(normalized_log_weights)
        d_log_weight_normalizer = tf.reduce_sum(
            normalized_weights[tf.newaxis, :] * density["d_log_density"],
            axis=1,
        )
        d_normalized_log_weights = (
            density["d_log_density"] - d_log_weight_normalizer[:, tf.newaxis]
        )
        d_normalized_weights = (
            normalized_weights[tf.newaxis, :] * d_normalized_log_weights
        )

        responsibilities = density["responsibilities"]
        d_responsibilities = responsibilities[tf.newaxis, :, :] * (
            density["d_log_component"] - density["d_log_density"][:, :, tf.newaxis]
        )
        transported_covariance_marks = tf.einsum(
            "ji,iab->jab", responsibilities, covariance_marks
        )
        d_transported_covariance_marks = tf.einsum(
            "kji,iab->kjab", d_responsibilities, covariance_marks
        ) + tf.einsum("ji,kiab->kjab", responsibilities, d_covariance_marks)

        transposed_bandwidths = tf.linalg.matrix_transpose(bandwidth_covariances)
        symmetric_bandwidths = 0.5 * (bandwidth_covariances + transposed_bandwidths)
        bandwidth_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_bandwidths), axis=[-2, -1]),
            tf.ones([particle_count], dtype),
        )
        bandwidth_symmetry_error = tf.reduce_max(
            tf.abs(bandwidth_covariances - transposed_bandwidths),
            axis=[-2, -1],
        )
        transposed_d_bandwidths = tf.linalg.matrix_transpose(d_bandwidth_covariances)
        symmetric_d_bandwidths = 0.5 * (
            d_bandwidth_covariances + transposed_d_bandwidths
        )
        d_bandwidth_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_d_bandwidths), axis=[-2, -1]),
            tf.ones([direction_count, particle_count], dtype),
        )
        d_bandwidth_symmetry_error = tf.reduce_max(
            tf.abs(d_bandwidth_covariances - transposed_d_bandwidths),
            axis=[-2, -1],
        )
        bandwidth_valid = (
            _finite_rows(bandwidth_covariances)
            & _finite_tangents_per_row(d_bandwidth_covariances)
            & (bandwidth_symmetry_error <= symmetry_tolerance_tensor * bandwidth_scale)
            & tf.reduce_all(
                d_bandwidth_symmetry_error
                <= symmetry_tolerance_tensor * d_bandwidth_scale,
                axis=0,
            )
        )

        transposed_marks = tf.linalg.matrix_transpose(covariance_marks)
        symmetric_marks = 0.5 * (covariance_marks + transposed_marks)
        mark_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_marks), axis=[-2, -1]),
            tf.ones([particle_count], dtype),
        )
        mark_symmetry_error = tf.reduce_max(
            tf.abs(covariance_marks - transposed_marks), axis=[-2, -1]
        )
        mark_minimum_eigenvalue = tf.reduce_min(
            tf.linalg.eigvalsh(symmetric_marks), axis=-1
        )
        transposed_d_marks = tf.linalg.matrix_transpose(d_covariance_marks)
        symmetric_d_marks = 0.5 * (d_covariance_marks + transposed_d_marks)
        d_mark_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_d_marks), axis=[-2, -1]),
            tf.ones([direction_count, particle_count], dtype),
        )
        d_mark_symmetry_error = tf.reduce_max(
            tf.abs(d_covariance_marks - transposed_d_marks),
            axis=[-2, -1],
        )
        mark_valid = (
            _finite_rows(covariance_marks)
            & _finite_tangents_per_row(d_covariance_marks)
            & (mark_symmetry_error <= symmetry_tolerance_tensor * mark_scale)
            & tf.reduce_all(
                d_mark_symmetry_error <= symmetry_tolerance_tensor * d_mark_scale,
                axis=0,
            )
            & (mark_minimum_eigenvalue > rank_tolerance_tensor * mark_scale)
        )
        proposal_valid = tf.math.is_finite(proposal_log_density)
        all_valid = (
            tf.reduce_all(density["valid"])
            & tf.reduce_all(proposal_valid)
            & tf.reduce_all(bandwidth_valid)
            & tf.reduce_all(mark_valid)
            & tf.reduce_all(tf.math.is_finite(log_ratio))
            & tf.reduce_all(tf.math.is_finite(importance_weights))
            & tf.reduce_all(tf.math.is_finite(d_importance_weights))
            & tf.reduce_all(tf.math.is_finite(d_responsibilities))
            & tf.reduce_all(tf.math.is_finite(transported_covariance_marks))
            & tf.reduce_all(tf.math.is_finite(d_transported_covariance_marks))
        )
        nan = tf.cast(float("nan"), dtype)

        def checked(value: Tensor) -> Tensor:
            return tf.where(all_valid, value, tf.fill(tf.shape(value), nan))

        return {
            "log_density": density["log_density"],
            "d_log_density": density["d_log_density"],
            "component_log_density": density["component_log_density"],
            "d_log_component": density["d_log_component"],
            "responsibilities": responsibilities,
            "d_responsibilities": checked(d_responsibilities),
            "log_ratio": checked(log_ratio),
            "importance_log_weights": checked(importance_log_weights),
            "importance_weights": checked(importance_weights),
            "d_importance_log_weights": checked(d_importance_log_weights),
            "d_importance_weights": checked(d_importance_weights),
            "normalized_log_weights": checked(normalized_log_weights),
            "normalized_weights": checked(normalized_weights),
            "d_normalized_log_weights": checked(d_normalized_log_weights),
            "d_normalized_weights": checked(d_normalized_weights),
            "transported_covariance_marks": checked(transported_covariance_marks),
            "d_transported_covariance_marks": checked(d_transported_covariance_marks),
            "anchor_log_ratio_error": tf.reduce_max(tf.abs(log_ratio)),
            "responsibility_row_sum_error": tf.reduce_max(
                tf.abs(
                    tf.reduce_sum(responsibilities, axis=1)
                    - tf.ones([particle_count], dtype)
                )
            ),
            "weight_sum_error": tf.abs(
                tf.reduce_sum(normalized_weights) - tf.ones([], dtype)
            ),
            "weight_tangent_sum_error": tf.reduce_max(
                tf.abs(tf.reduce_sum(d_normalized_weights, axis=1))
            ),
            "importance_weight_sum": tf.reduce_sum(importance_weights),
            "importance_weight_tangent_sum": tf.reduce_sum(
                d_importance_weights, axis=1
            ),
            "minimum_bandwidth_eigenvalue": tf.reduce_min(
                density["minimum_eigenvalue"]
            ),
            "minimum_source_mark_eigenvalue": tf.reduce_min(mark_minimum_eigenvalue),
            "bandwidth_symmetry_error": tf.reduce_max(bandwidth_symmetry_error),
            "bandwidth_tangent_symmetry_error": tf.reduce_max(
                d_bandwidth_symmetry_error
            ),
            "source_mark_symmetry_error": tf.reduce_max(mark_symmetry_error),
            "source_mark_tangent_symmetry_error": tf.reduce_max(d_mark_symmetry_error),
            "valid_rows": density["valid"] & proposal_valid,
            "bandwidths_valid": bandwidth_valid,
            "source_marks_valid": mark_valid,
            "valid": all_valid,
            "complexity_pair_count": density["complexity_pair_count"],
        }

    kernel.route_id = FULL_MIXTURE_RESAMPLING_ROUTE_ID
    kernel.target_label = RESKDM_IWSG_FINITE_TARGET
    kernel.route_classification = (
        "younis_raw_iwsg_all_components_with_bayesfilter_mark_extension_reference_only"
    )
    kernel.sample_tangent_policy = "fixed_anchor_samples_no_location_tangent_v1"
    kernel.proposal_policy = "fixed_anchor_marginal_mixture_density_v1"
    kernel.covariance_mark_policy = "responsibility_conditional_mean_no_scatter_v1"
    return kernel


def make_conditional_gaussian_kdm_kernel(
    *,
    evaluation_count: int,
    ancestor_count: int,
    component_count: int,
    dimension: int,
    direction_count: int = 1,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create a fixed-shape conditional KDM density evaluator.

    Each evaluation row selects one ancestor.  The returned density is the
    complete component mixture for that ancestor, which is the denominator
    required by a mixture proposal even when one component label was sampled.
    """

    evaluation_count = _static_positive("evaluation_count", evaluation_count)
    ancestor_count = _static_positive("ancestor_count", ancestor_count)
    component_count = _static_positive("component_count", component_count)
    dimension = _static_positive("dimension", dimension)
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if float(rank_tolerance) <= 0.0 or float(normalization_tolerance) <= 0.0:
        raise ValueError("tolerances must be positive")
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(normalization_tolerance, dtype)

    @tf.function(
        input_signature=[
            tf.TensorSpec([evaluation_count, dimension], dtype),
            tf.TensorSpec([evaluation_count], tf.int32),
            tf.TensorSpec([ancestor_count, component_count], dtype),
            tf.TensorSpec([ancestor_count, component_count, dimension], dtype),
            tf.TensorSpec(
                [ancestor_count, component_count, dimension, dimension], dtype
            ),
            tf.TensorSpec([direction_count, evaluation_count, dimension], dtype),
            tf.TensorSpec([direction_count, ancestor_count, component_count], dtype),
            tf.TensorSpec(
                [direction_count, ancestor_count, component_count, dimension], dtype
            ),
            tf.TensorSpec(
                [
                    direction_count,
                    ancestor_count,
                    component_count,
                    dimension,
                    dimension,
                ],
                dtype,
            ),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        values: Tensor,
        ancestor_indices: Tensor,
        weights: Tensor,
        means: Tensor,
        covariances: Tensor,
        d_values: Tensor,
        d_weights: Tensor,
        d_means: Tensor,
        d_covariances: Tensor,
    ) -> Mapping[str, Tensor]:
        index_valid = tf.logical_and(
            ancestor_indices >= 0,
            ancestor_indices < tf.constant(ancestor_count, tf.int32),
        )
        safe_indices = tf.clip_by_value(ancestor_indices, 0, ancestor_count - 1)
        selected_weights = tf.gather(weights, safe_indices)
        selected_means = tf.gather(means, safe_indices)
        selected_covariances = tf.gather(covariances, safe_indices)
        selected_d_weights = tf.gather(d_weights, safe_indices, axis=1)
        selected_d_means = tf.gather(d_means, safe_indices, axis=1)
        selected_d_covariances = tf.gather(d_covariances, safe_indices, axis=1)
        result = _batched_gaussian_mixture_core(
            values,
            selected_weights,
            selected_means,
            selected_covariances,
            d_values,
            selected_d_weights,
            selected_d_means,
            selected_d_covariances,
            rank_tolerance=rank_tolerance_tensor,
            normalization_tolerance=normalization_tolerance_tensor,
        )
        valid = result["valid"] & index_valid
        nan = tf.cast(float("nan"), dtype)
        result = dict(result)
        result["valid"] = valid
        result["log_density"] = tf.where(valid, result["log_density"], nan)
        result["d_log_density"] = tf.where(
            valid[tf.newaxis, :], result["d_log_density"], nan
        )
        return result

    kernel.route_id = ROUTE_ID
    kernel.target_label = KDM_FINITE_TARGET
    kernel.route_role = "conditional_kdm_proposal_density"
    return kernel


def make_iwsg_kernel(
    *,
    evaluation_count: int,
    component_count: int,
    dimension: int,
    direction_count: int = 1,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create Younis IWSG over fixed samples and a fixed proposal density.

    Sample locations are deliberately absent from the tangent interface.  In
    Younis IWSG they are held fixed; allowing a ``d_samples`` input would mix
    the fixed-proposal estimator with a different pathwise derivative.
    """

    evaluation_count = _static_positive("evaluation_count", evaluation_count)
    component_count = _static_positive("component_count", component_count)
    dimension = _static_positive("dimension", dimension)
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if float(rank_tolerance) <= 0.0 or float(normalization_tolerance) <= 0.0:
        raise ValueError("tolerances must be positive")
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)

    @tf.function(
        input_signature=[
            tf.TensorSpec([evaluation_count, dimension], dtype),
            tf.TensorSpec([component_count], dtype),
            tf.TensorSpec([component_count, dimension], dtype),
            tf.TensorSpec([component_count, dimension, dimension], dtype),
            tf.TensorSpec([direction_count, component_count], dtype),
            tf.TensorSpec([direction_count, component_count, dimension], dtype),
            tf.TensorSpec(
                [direction_count, component_count, dimension, dimension], dtype
            ),
            tf.TensorSpec([evaluation_count], dtype),
            tf.TensorSpec([evaluation_count], dtype),
            tf.TensorSpec([direction_count, evaluation_count], dtype),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        samples: Tensor,
        weights: Tensor,
        means: Tensor,
        covariances: Tensor,
        d_weights: Tensor,
        d_means: Tensor,
        d_covariances: Tensor,
        proposal_log_density: Tensor,
        integrand: Tensor,
        d_integrand: Tensor,
    ) -> Mapping[str, Tensor]:
        global_weights = tf.broadcast_to(
            weights[tf.newaxis, :], [evaluation_count, component_count]
        )
        global_means = tf.broadcast_to(
            means[tf.newaxis, :, :], [evaluation_count, component_count, dimension]
        )
        global_covariances = tf.broadcast_to(
            covariances[tf.newaxis, :, :, :],
            [evaluation_count, component_count, dimension, dimension],
        )
        global_d_weights = tf.broadcast_to(
            d_weights[:, tf.newaxis, :],
            [direction_count, evaluation_count, component_count],
        )
        global_d_means = tf.broadcast_to(
            d_means[:, tf.newaxis, :, :],
            [direction_count, evaluation_count, component_count, dimension],
        )
        global_d_covariances = tf.broadcast_to(
            d_covariances[:, tf.newaxis, :, :, :],
            [direction_count, evaluation_count, component_count, dimension, dimension],
        )
        density = _batched_gaussian_mixture_core(
            samples,
            global_weights,
            global_means,
            global_covariances,
            tf.zeros([direction_count, evaluation_count, dimension], dtype),
            global_d_weights,
            global_d_means,
            global_d_covariances,
            rank_tolerance=rank_tolerance_tensor,
            normalization_tolerance=normalization_tolerance_tensor,
        )
        finite = (
            density["valid"]
            & tf.math.is_finite(proposal_log_density)
            & tf.math.is_finite(integrand)
            & tf.reduce_all(tf.math.is_finite(d_integrand), axis=0)
        )
        log_ratio = density["log_density"] - proposal_log_density
        ratio = tf.exp(log_ratio)
        estimate = tf.reduce_mean(ratio * integrand)
        tangent_estimate = tf.reduce_mean(
            ratio[tf.newaxis, :]
            * (density["d_log_density"] * integrand[tf.newaxis, :] + d_integrand),
            axis=1,
        )
        anchor_weight_error = tf.reduce_max(tf.abs(ratio - tf.ones_like(ratio)))
        return {
            "estimate": tf.where(
                tf.reduce_all(finite), estimate, tf.cast(float("nan"), dtype)
            ),
            "d_estimate": tf.where(
                tf.reduce_all(finite),
                tangent_estimate,
                tf.fill([direction_count], tf.cast(float("nan"), dtype)),
            ),
            "log_ratio": tf.where(
                finite,
                log_ratio,
                tf.fill([evaluation_count], tf.cast(float("nan"), dtype)),
            ),
            "importance_ratio": tf.where(
                finite, ratio, tf.fill([evaluation_count], tf.cast(float("nan"), dtype))
            ),
            "anchor_weight_error": anchor_weight_error,
            "valid": finite,
            "density_log_density": density["log_density"],
            "density_d_log_density": density["d_log_density"],
        }

    kernel.route_id = ROUTE_ID
    kernel.target_label = KDM_FINITE_TARGET
    return kernel


def make_anchored_pfpf_kdm_weight_kernel(
    *,
    particle_count: int,
    ancestor_count: int,
    direction_count: int = 1,
    dtype: tf.dtypes.DType | str = tf.float64,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create an anchored conditional-KDM PF-PF importance kernel.

    The ancestor-selection law and complete KDM proposal density are values
    frozen at the proposal anchor.  Their tangents are absent by construction.
    This is the fixed-denominator principle used by IWSG, applied here to a
    separately derived model-target importance ratio.  Subtracting arbitrary
    proposal tangents would not estimate the derivative of the model
    normalizer under the frozen sampling law.

    ``forward_map_valid`` is a required per-particle certificate from a
    checked Jacobian calculation.  A finite log determinant alone does not
    establish that the declared forward map is invertible on the sampled
    branch, so the correction fails closed when this certificate is false.
    """

    particle_count = _static_positive("particle_count", particle_count)
    ancestor_count = _static_positive("ancestor_count", ancestor_count)
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if float(normalization_tolerance) <= 0.0:
        raise ValueError("normalization_tolerance must be positive")
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)

    signature = [
        tf.TensorSpec([ancestor_count], dtype),
        tf.TensorSpec([ancestor_count], dtype),
        tf.TensorSpec([particle_count], tf.int32),
        tf.TensorSpec([particle_count], dtype),
        tf.TensorSpec([particle_count], dtype),
        tf.TensorSpec([particle_count], dtype),
        tf.TensorSpec([particle_count], tf.bool),
        tf.TensorSpec([particle_count], dtype),
        tf.TensorSpec([direction_count, ancestor_count], dtype),
        tf.TensorSpec([direction_count, particle_count], dtype),
        tf.TensorSpec([direction_count, particle_count], dtype),
        tf.TensorSpec([direction_count, particle_count], dtype),
    ]

    @tf.function(
        input_signature=signature,
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        ancestor_log_weights: Tensor,
        ancestor_log_selection_probs: Tensor,
        ancestor_indices: Tensor,
        transition_log_density: Tensor,
        observation_log_density: Tensor,
        forward_log_det: Tensor,
        forward_map_valid: Tensor,
        proposal_log_density: Tensor,
        d_ancestor_log_weights: Tensor,
        d_transition_log_density: Tensor,
        d_observation_log_density: Tensor,
        d_forward_log_det: Tensor,
    ) -> Mapping[str, Tensor]:
        index_valid = tf.logical_and(
            ancestor_indices >= 0,
            ancestor_indices < tf.constant(ancestor_count, tf.int32),
        )
        safe_indices = tf.clip_by_value(ancestor_indices, 0, ancestor_count - 1)
        log_weights = (
            tf.gather(ancestor_log_weights, safe_indices)
            + transition_log_density
            + observation_log_density
            + forward_log_det
            - tf.gather(ancestor_log_selection_probs, safe_indices)
            - proposal_log_density
        )
        d_log_weights = (
            tf.gather(d_ancestor_log_weights, safe_indices, axis=1)
            + d_transition_log_density
            + d_observation_log_density
            + d_forward_log_det
        )
        normalized_ancestor_weights = (
            tf.abs(tf.reduce_logsumexp(ancestor_log_weights))
            <= normalization_tolerance_tensor
        )
        normalized_selection_probs = (
            tf.abs(tf.reduce_logsumexp(ancestor_log_selection_probs))
            <= normalization_tolerance_tensor
        )
        finite = (
            index_valid
            & forward_map_valid
            & normalized_ancestor_weights
            & normalized_selection_probs
        )
        finite = (finite & tf.math.is_finite(log_weights)
                  & tf.math.is_finite(transition_log_density)
                  & tf.math.is_finite(observation_log_density)
                  & tf.math.is_finite(forward_log_det)
                  & tf.math.is_finite(proposal_log_density))
        finite = finite & tf.reduce_all(tf.math.is_finite(d_log_weights), axis=0)
        finite = finite & tf.reduce_all(tf.math.is_finite(ancestor_log_weights))
        finite = finite & tf.reduce_all(tf.math.is_finite(ancestor_log_selection_probs))
        finite = finite & tf.reduce_all(tf.math.is_finite(d_ancestor_log_weights))
        normalizer = tf.reduce_logsumexp(log_weights) - tf.math.log(
            tf.cast(particle_count, dtype)
        )
        normalized = tf.exp(log_weights - tf.reduce_logsumexp(log_weights))
        d_normalizer = tf.reduce_sum(normalized[tf.newaxis, :] * d_log_weights, axis=1)
        nan = tf.cast(float("nan"), dtype)
        return {
            "log_weights": tf.where(
                finite, log_weights, tf.fill([particle_count], nan)
            ),
            "normalized_weights": tf.where(
                finite, normalized, tf.fill([particle_count], nan)
            ),
            "normalizer": tf.where(tf.reduce_all(finite), normalizer, nan),
            "d_log_weights": tf.where(
                finite[tf.newaxis, :],
                d_log_weights,
                tf.fill([direction_count, particle_count], nan),
            ),
            "d_normalizer": tf.where(
                tf.reduce_all(finite),
                d_normalizer,
                tf.fill([direction_count], nan),
            ),
            "valid": finite,
        }

    kernel.route_id = ANCHORED_MODEL_IS_ROUTE_ID
    kernel.target_label = MODEL_IS_TARGET
    kernel.proposal_derivative_policy = "frozen_anchor_no_denominator_tangent_v1"
    return kernel


def make_linear_gaussian_kdm_normalizer_kernel(
    *,
    particle_count: int,
    state_dimension: int,
    observation_dimension: int,
    direction_count: int = 1,
    bandwidth_is_zero: bool = False,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create the controlled linear-Gaussian KDM observation normalizer.

    With ``bandwidth_is_zero=False`` this evaluates the explicitly changed
    ``KDM-FINITE`` scalar obtained by convolving each Gaussian observation
    factor with a positive component covariance.  With ``bandwidth_is_zero``
    it evaluates the separate ``ATOM-FINITE`` branch and requires zero
    bandwidth and bandwidth tangent.  The latter branch never factors or
    inverts a zero covariance.
    """

    particle_count = _static_positive("particle_count", particle_count)
    state_dimension = _static_positive("state_dimension", state_dimension)
    observation_dimension = _static_positive(
        "observation_dimension", observation_dimension
    )
    direction_count = _static_positive("direction_count", direction_count)
    dtype = _dtype_from_value(dtype)
    if float(rank_tolerance) <= 0.0 or float(normalization_tolerance) <= 0.0:
        raise ValueError("tolerances must be positive")
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)

    signature = [
        tf.TensorSpec([particle_count, state_dimension], dtype),
        tf.TensorSpec([particle_count], dtype),
        tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
        tf.TensorSpec([observation_dimension, state_dimension], dtype),
        tf.TensorSpec([observation_dimension, observation_dimension], dtype),
        tf.TensorSpec([observation_dimension], dtype),
        tf.TensorSpec([direction_count, particle_count, state_dimension], dtype),
        tf.TensorSpec([direction_count, particle_count], dtype),
        tf.TensorSpec(
            [direction_count, particle_count, state_dimension, state_dimension],
            dtype,
        ),
        tf.TensorSpec([direction_count, observation_dimension, state_dimension], dtype),
        tf.TensorSpec(
            [direction_count, observation_dimension, observation_dimension], dtype
        ),
        tf.TensorSpec([direction_count, observation_dimension], dtype),
    ]

    @tf.function(
        input_signature=signature,
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        states: Tensor,
        weights: Tensor,
        bandwidths: Tensor,
        observation_matrix: Tensor,
        observation_covariance: Tensor,
        observation: Tensor,
        d_states: Tensor,
        d_weights: Tensor,
        d_bandwidths: Tensor,
        d_observation_matrix: Tensor,
        d_observation_covariance: Tensor,
        d_observation: Tensor,
    ) -> Mapping[str, Tensor]:
        symmetric_bandwidths = 0.5 * (
            bandwidths + tf.linalg.matrix_transpose(bandwidths)
        )
        symmetric_d_bandwidths = 0.5 * (
            d_bandwidths + tf.linalg.matrix_transpose(d_bandwidths)
        )
        bandwidth_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_bandwidths), axis=[-2, -1]),
            tf.ones([particle_count], dtype),
        )
        bandwidth_eigenvalues = tf.linalg.eigvalsh(symmetric_bandwidths)
        minimum_bandwidth_eigenvalue = tf.reduce_min(bandwidth_eigenvalues, axis=-1)
        if bandwidth_is_zero:
            bandwidth_valid = tf.reduce_all(
                tf.abs(symmetric_bandwidths) <= normalization_tolerance_tensor,
                axis=[-2, -1],
            )
            tangent_bandwidth_valid = tf.reduce_all(
                tf.abs(symmetric_d_bandwidths) <= normalization_tolerance_tensor,
                axis=[0, -3, -2, -1],
            )
            effective_bandwidths = tf.zeros_like(symmetric_bandwidths)
            effective_d_bandwidths = tf.zeros_like(symmetric_d_bandwidths)
        else:
            bandwidth_valid = minimum_bandwidth_eigenvalue > (
                rank_tolerance_tensor * bandwidth_scale
            )
            tangent_bandwidth_valid = tf.ones([particle_count], tf.bool)
            effective_bandwidths = symmetric_bandwidths
            effective_d_bandwidths = symmetric_d_bandwidths

        symmetric_observation_covariance = 0.5 * (
            observation_covariance + tf.linalg.matrix_transpose(observation_covariance)
        )
        symmetric_d_observation_covariance = 0.5 * (
            d_observation_covariance
            + tf.linalg.matrix_transpose(d_observation_covariance)
        )
        observation_scale = tf.maximum(
            tf.reduce_max(tf.abs(symmetric_observation_covariance)),
            tf.constant(1.0, dtype),
        )
        observation_eigenvalues = tf.linalg.eigvalsh(symmetric_observation_covariance)
        observation_valid = tf.reduce_min(observation_eigenvalues) > (
            rank_tolerance_tensor * observation_scale
        )
        effective_covariances = symmetric_observation_covariance[
            tf.newaxis, :, :
        ] + tf.einsum(
            "od,ndf,pf->nop",
            observation_matrix,
            effective_bandwidths,
            observation_matrix,
        )
        effective_covariances = 0.5 * (
            effective_covariances + tf.linalg.matrix_transpose(effective_covariances)
        )
        effective_d_covariances = (
            symmetric_d_observation_covariance[:, tf.newaxis, :, :]
            + tf.einsum(
                "kod,ndf,pf->knop",
                d_observation_matrix,
                effective_bandwidths,
                observation_matrix,
            )
            + tf.einsum(
                "od,kndf,pf->knop",
                observation_matrix,
                effective_d_bandwidths,
                observation_matrix,
            )
            + tf.einsum(
                "od,ndf,kpf->knop",
                observation_matrix,
                effective_bandwidths,
                d_observation_matrix,
            )
        )
        effective_d_covariances = 0.5 * (
            effective_d_covariances
            + tf.linalg.matrix_transpose(effective_d_covariances)
        )
        covariance_eigenvalues = tf.linalg.eigvalsh(effective_covariances)
        covariance_valid = tf.reduce_min(covariance_eigenvalues, axis=-1) > (
            rank_tolerance_tensor
            * tf.maximum(
                tf.reduce_max(tf.abs(effective_covariances), axis=[-2, -1]),
                tf.ones([particle_count], dtype),
            )
        )
        finite_inputs = (
            tf.reduce_all(tf.math.is_finite(states), axis=1)
            & tf.math.is_finite(weights)
            & tf.reduce_all(tf.math.is_finite(symmetric_bandwidths), axis=[-2, -1])
            & tf.reduce_all(tf.math.is_finite(observation_matrix))
            & tf.reduce_all(tf.math.is_finite(observation_covariance))
            & tf.reduce_all(tf.math.is_finite(observation))
            & tf.reduce_all(tf.math.is_finite(d_states), axis=[0, 2])
            & tf.reduce_all(tf.math.is_finite(d_weights), axis=0)
            & tf.reduce_all(tf.math.is_finite(d_bandwidths), axis=[0, 2, 3])
            & tf.reduce_all(tf.math.is_finite(d_observation_matrix))
            & tf.reduce_all(tf.math.is_finite(d_observation_covariance))
            & tf.reduce_all(tf.math.is_finite(d_observation))
        )
        positive_weights = weights > tf.zeros_like(weights)
        normalized_weights = (
            tf.abs(tf.reduce_sum(weights) - tf.constant(1.0, dtype))
            <= normalization_tolerance_tensor
        )
        normalized_weight_tangents = tf.reduce_all(
            tf.abs(tf.reduce_sum(d_weights, axis=1)) <= normalization_tolerance_tensor
        )
        row_valid = (
            finite_inputs
            & positive_weights
            & normalized_weights
            & normalized_weight_tangents
            & tangent_bandwidth_valid
            & bandwidth_valid
            & covariance_valid
            & observation_valid
        )

        identity = tf.eye(observation_dimension, dtype=dtype)
        safe_covariances = tf.where(
            covariance_valid[:, tf.newaxis, tf.newaxis],
            effective_covariances,
            identity[tf.newaxis, :, :],
        )
        factors = tf.linalg.cholesky(safe_covariances)
        means = tf.einsum("od,nd->no", observation_matrix, states)
        residual = observation[tf.newaxis, :] - means
        solved = tf.linalg.triangular_solve(
            factors, residual[..., tf.newaxis], lower=True
        )[..., 0]
        precision_residual = tf.linalg.cholesky_solve(
            factors, residual[..., tf.newaxis]
        )[..., 0]
        quadratic = tf.reduce_sum(tf.square(solved), axis=-1)
        log_determinant = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(factors)), axis=-1
        )
        log_two_pi = tf.constant(math.log(2.0 * math.pi), dtype)
        component_log_density = -0.5 * (
            tf.cast(observation_dimension, dtype) * log_two_pi
            + log_determinant
            + quadratic
        )
        joint_log_density = tf.math.log(weights) + component_log_density
        value = tf.reduce_logsumexp(joint_log_density)
        responsibilities = tf.exp(joint_log_density - value)

        d_means = tf.einsum("kod,nd->kno", d_observation_matrix, states) + tf.einsum(
            "od,knd->kno", observation_matrix, d_states
        )
        mean_displacement = d_means - d_observation[:, tf.newaxis, :]
        mean_term = tf.einsum("no,kno->kn", precision_residual, mean_displacement)
        covariance_quadratic = tf.einsum(
            "no,knoe,ne->kn",
            precision_residual,
            effective_d_covariances,
            precision_residual,
        )
        d_covariances_nk = tf.transpose(effective_d_covariances, [1, 0, 2, 3])
        solved_covariances = tf.linalg.cholesky_solve(
            factors[:, tf.newaxis, :, :], d_covariances_nk
        )
        covariance_trace = tf.linalg.trace(solved_covariances)
        covariance_trace = tf.transpose(covariance_trace, [1, 0])
        d_log_component = (
            d_weights / weights[tf.newaxis, :]
            + mean_term
            + 0.5 * (covariance_quadratic - covariance_trace)
        )
        score = tf.reduce_sum(responsibilities[tf.newaxis, :] * d_log_component, axis=1)
        nan = tf.cast(float("nan"), dtype)
        return {
            "value": tf.where(tf.reduce_all(row_valid), value, nan),
            "score": tf.where(
                tf.reduce_all(row_valid), score, tf.fill([direction_count], nan)
            ),
            "component_log_density": tf.where(
                row_valid,
                component_log_density,
                tf.fill([particle_count], nan),
            ),
            "responsibilities": tf.where(
                row_valid, responsibilities, tf.fill([particle_count], nan)
            ),
            "minimum_bandwidth_eigenvalue": tf.reduce_min(minimum_bandwidth_eigenvalue),
            "component_valid": row_valid,
            "valid": tf.reduce_all(row_valid),
        }

    kernel.route_id = ROUTE_ID
    kernel.target_label = ATOM_FINITE_TARGET if bandwidth_is_zero else KDM_FINITE_TARGET
    return kernel


def make_subspace_gaussian_kdm_kernel(
    *,
    evaluation_count: int,
    component_count: int,
    ambient_dimension: int,
    support_dimension: int,
    direction_count: int = 1,
    support_tolerance: float = 1.0e-8,
    dtype: tf.dtypes.DType | str = tf.float64,
    rank_tolerance: float = 1.0e-12,
    normalization_tolerance: float = 1.0e-8,
    jit_compile: bool = True,
) -> Callable[..., Mapping[str, Tensor]]:
    """Create a fixed-chart, rank-aware Gaussian mixture evaluator.

    The shared ``support_basis`` has orthonormal columns and defines the
    innovation/chart coordinates.  The mixture density is with respect to
    that support measure, not ambient Lebesgue measure.  The basis is frozen
    for the supplied call; a parameter-dependent chart must provide its own
    Jacobian/tangent in a later route.
    """

    evaluation_count = _static_positive("evaluation_count", evaluation_count)
    component_count = _static_positive("component_count", component_count)
    ambient_dimension = _static_positive("ambient_dimension", ambient_dimension)
    support_dimension = _static_positive("support_dimension", support_dimension)
    direction_count = _static_positive("direction_count", direction_count)
    if support_dimension > ambient_dimension:
        raise ValueError("support_dimension cannot exceed ambient_dimension")
    dtype = _dtype_from_value(dtype)
    if float(support_tolerance) <= 0.0:
        raise ValueError("support_tolerance must be positive")
    if float(rank_tolerance) <= 0.0 or float(normalization_tolerance) <= 0.0:
        raise ValueError("tolerances must be positive")
    support_tolerance_tensor = tf.constant(float(support_tolerance), dtype)
    rank_tolerance_tensor = tf.constant(float(rank_tolerance), dtype)
    normalization_tolerance_tensor = tf.constant(float(normalization_tolerance), dtype)

    signature = [
        tf.TensorSpec([evaluation_count, ambient_dimension], dtype),
        tf.TensorSpec([component_count], dtype),
        tf.TensorSpec([component_count, ambient_dimension], dtype),
        tf.TensorSpec([component_count, support_dimension, support_dimension], dtype),
        tf.TensorSpec([ambient_dimension, support_dimension], dtype),
        tf.TensorSpec([direction_count, evaluation_count, ambient_dimension], dtype),
        tf.TensorSpec([direction_count, component_count], dtype),
        tf.TensorSpec([direction_count, component_count, ambient_dimension], dtype),
        tf.TensorSpec(
            [direction_count, component_count, support_dimension, support_dimension],
            dtype,
        ),
    ]

    @tf.function(
        input_signature=signature,
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        values: Tensor,
        weights: Tensor,
        means: Tensor,
        coordinate_covariances: Tensor,
        support_basis: Tensor,
        d_values: Tensor,
        d_weights: Tensor,
        d_means: Tensor,
        d_coordinate_covariances: Tensor,
    ) -> Mapping[str, Tensor]:
        gram = tf.linalg.matmul(support_basis, support_basis, transpose_a=True)
        gram_residual = gram - tf.eye(support_dimension, dtype=dtype)
        basis_finite = tf.reduce_all(tf.math.is_finite(support_basis))
        basis_valid = basis_finite & (
            tf.reduce_max(tf.abs(gram_residual)) <= support_tolerance_tensor
        )

        means_b = tf.broadcast_to(
            means[tf.newaxis, :, :],
            [evaluation_count, component_count, ambient_dimension],
        )
        d_means_b = tf.broadcast_to(
            d_means[:, tf.newaxis, :, :],
            [direction_count, evaluation_count, component_count, ambient_dimension],
        )
        coordinate_values = tf.einsum("dr,md->mr", support_basis, values)
        coordinate_means = tf.einsum("dr,mnd->mnr", support_basis, means_b)
        ambient_delta = values[:, tf.newaxis, :] - means_b
        coordinate_delta = coordinate_values[:, tf.newaxis, :] - coordinate_means
        reconstructed_delta = tf.einsum("dr,mnr->mnd", support_basis, coordinate_delta)
        support_residual = ambient_delta - reconstructed_delta
        support_scale = tf.maximum(
            tf.linalg.norm(ambient_delta, axis=-1),
            tf.ones([evaluation_count, component_count], dtype),
        )
        support_residual_norm = tf.linalg.norm(support_residual, axis=-1)
        support_valid = tf.reduce_all(
            support_residual_norm <= support_tolerance_tensor * support_scale,
            axis=1,
        )

        d_coordinate_values = tf.einsum("dr,kmd->kmr", support_basis, d_values)
        d_coordinate_means = tf.einsum("dr,kmnd->kmnr", support_basis, d_means_b)
        d_ambient_delta = d_values[:, :, tf.newaxis, :] - d_means_b
        d_coordinate_delta = (
            d_coordinate_values[:, :, tf.newaxis, :] - d_coordinate_means
        )
        d_reconstructed_delta = tf.einsum(
            "dr,kmnr->kmnd", support_basis, d_coordinate_delta
        )
        d_support_residual = d_ambient_delta - d_reconstructed_delta
        d_support_residual_norm = tf.linalg.norm(d_support_residual, axis=-1)
        d_support_valid = tf.reduce_all(
            d_support_residual_norm
            <= support_tolerance_tensor * support_scale[tf.newaxis, :, :],
            axis=[0, 2],
        )

        result = _batched_gaussian_mixture_core(
            coordinate_values,
            tf.broadcast_to(
                weights[tf.newaxis, :], [evaluation_count, component_count]
            ),
            coordinate_means,
            tf.broadcast_to(
                coordinate_covariances[tf.newaxis, :, :, :],
                [
                    evaluation_count,
                    component_count,
                    support_dimension,
                    support_dimension,
                ],
            ),
            d_coordinate_values,
            tf.broadcast_to(
                d_weights[:, tf.newaxis, :],
                [direction_count, evaluation_count, component_count],
            ),
            d_coordinate_means,
            tf.broadcast_to(
                d_coordinate_covariances[:, tf.newaxis, :, :, :],
                [
                    direction_count,
                    evaluation_count,
                    component_count,
                    support_dimension,
                    support_dimension,
                ],
            ),
            rank_tolerance=rank_tolerance_tensor,
            normalization_tolerance=normalization_tolerance_tensor,
        )
        valid = result["valid"] & support_valid & d_support_valid & basis_valid
        nan = tf.cast(float("nan"), dtype)
        result = dict(result)
        result["valid"] = valid
        result["log_density"] = tf.where(valid, result["log_density"], nan)
        result["d_log_density"] = tf.where(
            valid[tf.newaxis, :], result["d_log_density"], nan
        )
        result["support_residual_max"] = tf.reduce_max(support_residual_norm)
        result["tangent_support_residual_max"] = tf.reduce_max(d_support_residual_norm)
        result["basis_valid"] = basis_valid
        result["support_dimension"] = tf.constant(support_dimension, tf.int32)
        return result

    kernel.route_id = ROUTE_ID
    kernel.target_label = KDM_FINITE_TARGET
    kernel.chart_policy = "fixed_orthonormal_support_chart_v1"
    return kernel


def make_canonical_linear_gaussian_kdm_auxiliary_program(
    model: Any,
    *,
    theta_shape: tuple[int, ...],
    particle_count: int,
    state_dimension: int,
    observation_dimension: int,
    horizon: int,
    canonical_options: Mapping[str, Any],
    dtype: tf.dtypes.DType | str = tf.float64,
    theta_dtype: tf.dtypes.DType | str | None = None,
    jit_compile: bool = True,
    model_tolerance: float = 1.0e-8,
) -> Callable[..., Mapping[str, Tensor]]:
    """Own one native KDM auxiliary with fixed callbacks and configuration.

    The returned tensor-only function accepts theta, initial states/covariances,
    noises, observations, observation matrix/tangent and bandwidths/tangents.
    Callbacks and Python closure cells must remain fixed throughout this owner's
    lifetime. Rebuild the owner when they change. No global cache is used.
    Retain this function for repeated evaluations. XLA executable memory can
    survive owner collection, so workloads changing callback/configuration
    repeatedly should bound their worker-process lifetime. Tensor operands may
    change without rebuilding the owner when their signature stays fixed.
    Steps retain their complete public fields as time-stacked tensors; the public
    compatibility wrapper formats them only after the numerical recurrence.
    KDM is diagnostic only and never feeds back into the analytical trajectory.
    """
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        _value_and_analytical_score_impl,
        canonical_value_and_analytical_score,
    )

    options = dict(canonical_options)
    for forbidden in ("with_score", "return_trace"):
        if forbidden in options:
            raise ValueError(f"canonical_options must not contain {forbidden}")
    if options.get("reset_policy") != "contract_e":
        raise ValueError("KDM auxiliary requires reset_policy='contract_e'")
    if options.get("reset_design") is None:
        raise ValueError("KDM auxiliary requires an explicit Contract-E reset design")
    if int(options.get("annealed_stages", 1)) != 1:
        raise ValueError(
            "KDM auxiliary prior-observation trace requires annealed_stages=1"
        )
    if int(options.get("correction_steps", 0)) < 1:
        raise ValueError("KDM auxiliary requires the diagonal trust-region mechanism")
    if int(options.get("pairwise_steps", 0)) < 1:
        raise ValueError("KDM auxiliary requires the pairwise dual-cap mechanism")
    if float(options.get("coordinate_cap", 0.0)) <= 0.0:
        raise ValueError("KDM auxiliary requires the coordinate cap mechanism")
    if model.observation_log_density_fn is not None:
        raise ValueError(
            "linear-Gaussian KDM auxiliary does not support a custom observation density"
        )
    if float(model_tolerance) <= 0.0:
        raise ValueError("model_tolerance must be positive")

    callable_signature(canonical_value_and_analytical_score).bind_partial(**options)
    particle_count = _static_positive("particle_count", particle_count)
    state_dimension = _static_positive("state_dimension", state_dimension)
    observation_dimension = _static_positive("observation_dimension", observation_dimension)
    if isinstance(horizon, bool) or int(horizon) < 0:
        raise ValueError("horizon must be a nonnegative integer")
    horizon = int(horizon)
    dtype = _dtype_from_value(dtype)
    theta_dtype = dtype if theta_dtype is None else _dtype_from_value(theta_dtype)
    common = {
        "particle_count": particle_count,
        "state_dimension": state_dimension,
        "observation_dimension": observation_dimension,
        "direction_count": 1,
        "dtype": dtype,
        "jit_compile": jit_compile,
    }
    atom_kernel = make_linear_gaussian_kdm_normalizer_kernel(
        bandwidth_is_zero=True, **common
    )
    kdm_kernel = make_linear_gaussian_kdm_normalizer_kernel(
        bandwidth_is_zero=False, **common
    )
    zero_bandwidth = tf.zeros(
        [particle_count, state_dimension, state_dimension], dtype
    )
    zero_d_bandwidth = tf.zeros(
        [1, particle_count, state_dimension, state_dimension], dtype
    )

    signature = [
        tf.TensorSpec(theta_shape, theta_dtype),
        tf.TensorSpec([particle_count, state_dimension], dtype),
        tf.TensorSpec([particle_count, state_dimension, state_dimension], dtype),
        tf.TensorSpec([horizon, particle_count, state_dimension], dtype),
        tf.TensorSpec([horizon, observation_dimension], dtype),
        tf.TensorSpec([observation_dimension, state_dimension], dtype),
        tf.TensorSpec([observation_dimension, state_dimension], dtype),
        tf.TensorSpec([horizon, particle_count, state_dimension, state_dimension], dtype),
        tf.TensorSpec([horizon, particle_count, state_dimension, state_dimension], dtype),
    ]

    @tf.function(input_signature=signature, jit_compile=jit_compile, autograph=False)
    def program(theta, initial_states, initial_covariances, noises, observations,
                observation_matrix, d_observation_matrix, bandwidths, d_bandwidths):
        if horizon == 0:
            # The empty finite sum is independent of every operand. Avoid tracing
            # out-of-bounds observation slices in an unreachable XLA loop body.
            zero, zero_score = tf.zeros([], dtype), tf.zeros([1], dtype)
            return {
                "canonical_value": zero, "canonical_score": zero_score,
                "atom_reconstruction_value": zero, "atom_reconstruction_score": zero_score,
                "atom_value_error": zero, "atom_score_error": zero_score,
                "kdm_auxiliary_value": zero, "kdm_auxiliary_score": zero_score,
                "value_shift": zero, "score_shift": zero_score,
                "canonical_value_unchanged": tf.constant(True),
                "kdm_feedback_into_canonical": tf.constant(False),
                "valid": tf.constant(True), "model_valid": tf.constant(True),
                "steps": {
                    "base_log_normalizer": tf.zeros([0], dtype),
                    "atom_observation_value": tf.zeros([0], dtype),
                    "kdm_observation_value": tf.zeros([0], dtype),
                    "value_shift": tf.zeros([0], dtype),
                    "valid": tf.zeros([0], tf.bool),
                },
            }
        canonical_value, canonical_score, trace = _value_and_analytical_score_impl(
            model,
            theta,
            initial_states,
            initial_covariances,
            noises,
            observations,
            with_score=True,
            return_trace=True,
            _stacked_trace=True,
            **options,
        )
        d_observation_covariance = (
            model.observation_covariance_tangent_fn(theta)
            if model.observation_covariance_tangent_fn is not None
            else tf.zeros_like(model.observation_covariance)
        )
        zero_observation_tangent = tf.zeros([1, observation_dimension], dtype)
        atom_total = tf.zeros([], dtype)
        atom_score = tf.zeros([1], dtype)
        kdm_total = tf.zeros([], dtype)
        kdm_score = tf.zeros([1], dtype)
        step_records = {
            "base_log_normalizer": tf.TensorArray(dtype, size=horizon, element_shape=[]),
            "atom_observation_value": tf.TensorArray(dtype, size=horizon, element_shape=[]),
            "kdm_observation_value": tf.TensorArray(dtype, size=horizon, element_shape=[]),
            "value_shift": tf.TensorArray(dtype, size=horizon, element_shape=[]),
            "valid": tf.TensorArray(tf.bool, size=horizon, element_shape=[]),
        }
        all_models_valid = tf.constant(True)
        all_valid = tf.constant(True)

        def time_step(time_index, atom_total, atom_score, kdm_total, kdm_score,
                      all_valid, all_models_valid, step_records):
            record = tf.nest.map_structure(lambda value: value[time_index], trace)
            children = record["children"]
            d_children = record["d_children"]
            expected_observation = tf.einsum("od,nd->no", observation_matrix, children)
            actual_observation = model.observation_fn(children)
            expected_observation_tangent = tf.einsum(
                "od,nd->no", d_observation_matrix, children
            ) + tf.einsum("od,nd->no", observation_matrix, d_children)
            actual_observation_tangent = model.observation_tangent_fn(children, d_children)
            scale = tf.maximum(
                tf.reduce_max(tf.abs(expected_observation)), tf.constant(1.0, dtype)
            )
            model_valid = (
                tf.reduce_max(tf.abs(actual_observation - expected_observation))
                <= tf.cast(model_tolerance, dtype) * scale
            ) & (
                tf.reduce_max(
                    tf.abs(actual_observation_tangent - expected_observation_tangent)
                )
                <= tf.cast(model_tolerance, dtype) * scale
            )
            arguments = (
                children,
                record["prior_observation_weights"],
                observation_matrix,
                model.observation_covariance,
                record["observation"],
                d_children[tf.newaxis, :, :],
                record["d_prior_observation_weights"][tf.newaxis, :],
                d_observation_matrix[tf.newaxis, :, :],
                d_observation_covariance[tf.newaxis, :, :],
                zero_observation_tangent,
            )
            atom = atom_kernel(
                arguments[0],
                arguments[1],
                zero_bandwidth,
                *arguments[2:7],
                zero_d_bandwidth,
                *arguments[7:],
            )
            kdm = kdm_kernel(
                arguments[0],
                arguments[1],
                bandwidths[time_index],
                *arguments[2:7],
                d_bandwidths[time_index][tf.newaxis, :, :, :],
                *arguments[7:],
            )
            base_value = record["prior_observation_log_normalizer"]
            base_score = record["d_prior_observation_log_normalizer"][tf.newaxis]
            atom_total += base_value + atom["value"]
            atom_score += base_score + atom["score"]
            kdm_total += base_value + kdm["value"]
            kdm_score += base_score + kdm["score"]
            step_valid = model_valid & atom["valid"] & kdm["valid"]
            all_valid &= step_valid
            step_record = {
                "base_log_normalizer": base_value,
                "atom_observation_value": atom["value"],
                "kdm_observation_value": kdm["value"],
                "value_shift": kdm["value"] - atom["value"],
                "valid": step_valid,
            }
            step_records = tf.nest.map_structure(
                lambda buffer, value: buffer.write(time_index, value), step_records, step_record)
            return (time_index + 1, atom_total, atom_score, kdm_total, kdm_score,
                    all_valid, all_models_valid & model_valid, step_records)

        (_, atom_total, atom_score, kdm_total, kdm_score, all_valid,
         all_models_valid, step_records) = tf.while_loop(
            lambda time_index, *_: time_index < horizon, time_step,
            (tf.constant(0), atom_total, atom_score, kdm_total, kdm_score,
             all_valid, all_models_valid, step_records),
            maximum_iterations=horizon, parallel_iterations=1)
        step_records = tf.nest.map_structure(lambda buffer: buffer.stack(), step_records)
        nan = tf.constant(float("nan"), dtype)
        atom_total = tf.where(all_models_valid, atom_total, nan)
        atom_score = tf.where(all_models_valid, atom_score, nan)
        kdm_total = tf.where(all_models_valid, kdm_total, nan)
        kdm_score = tf.where(all_models_valid, kdm_score, nan)
        atom_value_error = atom_total - canonical_value
        atom_score_error = atom_score - canonical_score
        all_valid &= (tf.abs(atom_value_error) <= tf.cast(model_tolerance, dtype)) & (
            tf.reduce_max(tf.abs(atom_score_error)) <= tf.cast(model_tolerance, dtype)
        )
        return {
            "canonical_value": canonical_value,
            "canonical_score": canonical_score,
            "atom_reconstruction_value": atom_total,
            "atom_reconstruction_score": atom_score,
            "atom_value_error": atom_value_error,
            "atom_score_error": atom_score_error,
            "kdm_auxiliary_value": kdm_total,
            "kdm_auxiliary_score": kdm_score,
            "value_shift": kdm_total - canonical_value,
            "score_shift": kdm_score - canonical_score,
            "canonical_value_unchanged": tf.constant(True),
            "kdm_feedback_into_canonical": tf.constant(False),
            "valid": all_valid,
            "steps": step_records,
            "model_valid": all_models_valid,
        }

    return program


def canonical_linear_gaussian_kdm_auxiliary(
    model: Any,
    theta: Tensor,
    initial_states: Tensor,
    initial_covariances: Tensor,
    noises: Tensor,
    observations: Tensor,
    observation_matrix: Tensor,
    d_observation_matrix: Tensor,
    bandwidths: Tensor,
    d_bandwidths: Tensor,
    *,
    canonical_options: Mapping[str, Any],
    jit_compile: bool = True,
    model_tolerance: float = 1.0e-8,
) -> Mapping[str, Any]:
    """Evaluate the native auxiliary and present complete public step records.

    Retain ``make_canonical_linear_gaussian_kdm_auxiliary_program`` for repeated
    calls with fixed callbacks/configuration. This compatibility API deliberately
    rebuilds its owner, so mutable Python callback closures retain call semantics.
    Model mismatch raises ValueError at this completed eager assertion boundary;
    an enclosing graph receives false validity and NaN auxiliary results.
    """
    theta = tf.convert_to_tensor(theta)
    initial_states = tf.convert_to_tensor(initial_states)
    dtype = initial_states.dtype
    initial_covariances = tf.convert_to_tensor(initial_covariances, dtype)
    noises = tf.convert_to_tensor(noises, dtype)
    observations = tf.convert_to_tensor(observations, dtype)
    observation_matrix = tf.convert_to_tensor(observation_matrix, dtype)
    d_observation_matrix = tf.convert_to_tensor(d_observation_matrix, dtype)
    bandwidths = tf.convert_to_tensor(bandwidths, dtype)
    d_bandwidths = tf.convert_to_tensor(d_bandwidths, dtype)

    horizon = int(observations.shape[0])
    particle_count = int(initial_states.shape[0])
    state_dimension = int(initial_states.shape[1])
    observation_dimension = int(observations.shape[1])
    tf.ensure_shape(observation_matrix, [observation_dimension, state_dimension])
    tf.ensure_shape(d_observation_matrix, [observation_dimension, state_dimension])
    tf.ensure_shape(
        bandwidths,
        [horizon, particle_count, state_dimension, state_dimension],
    )
    tf.ensure_shape(
        d_bandwidths,
        [horizon, particle_count, state_dimension, state_dimension],
    )

    program = make_canonical_linear_gaussian_kdm_auxiliary_program(
        model, theta_shape=tuple(theta.shape), particle_count=particle_count,
        state_dimension=state_dimension, observation_dimension=observation_dimension,
        horizon=horizon, canonical_options=canonical_options, dtype=dtype,
        theta_dtype=theta.dtype, jit_compile=jit_compile, model_tolerance=model_tolerance)
    numerical = dict(program(theta, initial_states, initial_covariances, noises,
                             observations, observation_matrix, d_observation_matrix,
                             bandwidths, d_bandwidths))
    model_valid = numerical.pop("model_valid")
    if tf.executing_eagerly() and not bool(model_valid.numpy()):
        raise ValueError("observation callbacks do not match the supplied linear map/tangent")
    stacked_steps = numerical.pop("steps")
    # Fixed output presentation only; all numerical accumulation is already done.
    steps = tuple({"time_index": index, **tf.nest.map_structure(
        lambda value, index=index: value[index], stacked_steps)} for index in range(horizon))
    return {
        "route_id": AUXILIARY_ROUTE_ID,
        "route_role": AUXILIARY_ROLE,
        "canonical_target_label": ATOM_FINITE_TARGET,
        "auxiliary_target_label": KDM_FINITE_TARGET,
        **numerical, "steps": steps,
    }


def atom_expectation(
    values: Tensor,
    weights: Tensor,
    *,
    d_values: Tensor | None = None,
    d_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the explicit zero-bandwidth atom branch.

    This is intentionally separate from the Gaussian KDM evaluator: a zero
    covariance has no ordinary ambient Gaussian density.
    """

    values = tf.convert_to_tensor(values)
    weights = tf.convert_to_tensor(weights, values.dtype)
    if values.shape.rank != 1 or weights.shape != values.shape:
        raise ValueError("atom values and weights must be equal-length vectors")
    if d_values is None:
        d_values = tf.zeros([1, tf.shape(values)[0]], values.dtype)
    if d_weights is None:
        d_weights = tf.zeros([1, tf.shape(values)[0]], values.dtype)
    d_values = tf.convert_to_tensor(d_values, values.dtype)
    d_weights = tf.convert_to_tensor(d_weights, values.dtype)
    estimate = tf.reduce_sum(weights * values)
    d_estimate = tf.reduce_sum(
        d_weights * values[tf.newaxis, :] + weights[tf.newaxis, :] * d_values,
        axis=1,
    )
    return estimate, d_estimate


__all__ = [
    "ANCHORED_MODEL_IS_ROUTE_ID",
    "ATOM_FINITE_TARGET",
    "AUXILIARY_ROLE",
    "AUXILIARY_ROUTE_ID",
    "FULL_MIXTURE_RESAMPLING_ROUTE_ID",
    "KDM_FINITE_TARGET",
    "MODEL_IS_TARGET",
    "RESKDM_IWSG_FINITE_TARGET",
    "RESKDM_SELF_NORMALIZED_FINITE_TARGET",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "atom_expectation",
    "canonical_linear_gaussian_kdm_auxiliary",
    "make_anchored_pfpf_kdm_weight_kernel",
    "make_canonical_linear_gaussian_kdm_auxiliary_program",
    "make_conditional_gaussian_kdm_kernel",
    "make_full_mixture_iwsg_resampling_kernel",
    "make_gaussian_kdm_kernel",
    "make_iwsg_kernel",
    "make_linear_gaussian_kdm_normalizer_kernel",
    "make_subspace_gaussian_kdm_kernel",
]
