"""Smooth-branch TensorFlow score/Hessian wrapper for SVD-CUT."""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood
from bayesfilter.results_tf import TFFilterDerivativeResult
from bayesfilter.structural_tf import (
    TFStructuralStateSpace,
    structural_block_metadata,
    structural_filter_metadata,
)


TFStructuralModelBuilder = Callable[[tf.Tensor], TFStructuralStateSpace]


def _checked_smooth_value(
    value: tf.Tensor,
    diagnostics: dict[str, tf.Tensor],
    spectral_gap_tolerance: tf.Tensor,
) -> tf.Tensor:
    assertions = [
        tf.debugging.assert_equal(
            diagnostics["placement_floor_count"],
            tf.constant(0, dtype=tf.int32),
            message="blocked_active_floor: SVD-CUT placement floor is active",
        ),
        tf.debugging.assert_equal(
            diagnostics["innovation_floor_count"],
            tf.constant(0, dtype=tf.int32),
            message="blocked_active_floor: SVD-CUT innovation floor is active",
        ),
        tf.debugging.assert_greater(
            diagnostics["min_placement_eigen_gap"],
            spectral_gap_tolerance,
            message="blocked_weak_spectral_gap: SVD-CUT placement spectrum is not separated",
        ),
        tf.debugging.assert_greater(
            diagnostics["min_innovation_eigen_gap"],
            spectral_gap_tolerance,
            message="blocked_weak_spectral_gap: SVD-CUT innovation spectrum is not separated",
        ),
        tf.debugging.assert_all_finite(value, "blocked_nonfinite_value: SVD-CUT value is nonfinite"),
    ]
    with tf.control_dependencies(assertions):
        return tf.identity(value)


def tf_svd_cut4_score_hessian(
    observations: tf.Tensor,
    parameters: tf.Tensor,
    model_builder: TFStructuralModelBuilder,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> TFFilterDerivativeResult:
    """Return score and Hessian of the implemented smooth-branch SVD-CUT law."""

    params = tf.convert_to_tensor(parameters, dtype=tf.float64)
    spectral_gap_tolerance = tf.convert_to_tensor(spectral_gap_tolerance, dtype=tf.float64)

    with tf.GradientTape() as outer:
        outer.watch(params)
        with tf.GradientTape() as inner:
            inner.watch(params)
            model = model_builder(params)
            value, _means, _covariances, raw_diagnostics = tf_svd_cut4_log_likelihood(
                observations,
                model,
                placement_floor=placement_floor,
                innovation_floor=innovation_floor,
                rank_tolerance=rank_tolerance,
                jitter=jitter,
                return_filtered=False,
            )
            checked_value = _checked_smooth_value(
                value,
                raw_diagnostics,
                spectral_gap_tolerance,
            )
        score = inner.gradient(checked_value, params)
    hessian = outer.jacobian(score, params)

    model = model_builder(params)
    block_metadata = dict(structural_block_metadata(model))
    extra = {
        **block_metadata,
        "rule": "CUT4-G",
        "augmented_dim": raw_diagnostics["augmented_dim"],
        "point_count": raw_diagnostics["point_count"],
        "polynomial_degree": raw_diagnostics["polynomial_degree"],
        "max_integration_rank": raw_diagnostics["max_integration_rank"],
        "support_residual": raw_diagnostics["support_residual"],
        "deterministic_residual": raw_diagnostics["deterministic_residual"],
        "min_placement_eigen_gap": raw_diagnostics["min_placement_eigen_gap"],
        "min_innovation_eigen_gap": raw_diagnostics["min_innovation_eigen_gap"],
        "derivative_branch": "smooth_separated_spectrum",
        "derivative_method": "tensorflow_autodiff_of_implemented_svd_cut_value",
    }
    diagnostics = TFFilterDiagnostics(
        backend="tf_svd_cut4_score_hessian",
        mask_convention="none",
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=tf.float64),
            singular_floor=tf.convert_to_tensor(innovation_floor, dtype=tf.float64),
            floor_count=raw_diagnostics["innovation_floor_count"],
            psd_projection_residual=raw_diagnostics["innovation_psd_projection_residual"],
            implemented_covariance=raw_diagnostics["implemented_innovation_covariance"],
            branch_label="svd_cut4_smooth_derivative",
            derivative_target="implemented_regularized_law",
        ),
        extra=extra,
    )
    return TFFilterDerivativeResult(
        log_likelihood=checked_value,
        score=score,
        hessian=hessian,
        metadata=structural_filter_metadata(
            model,
            filter_name="tf_svd_cut4_score_hessian",
            differentiability_status="smooth_branch_score_hessian",
            compiled_status="eager_tf",
        ),
        diagnostics=diagnostics,
    )
