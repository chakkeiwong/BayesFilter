"""TensorFlow SVD-CUT value filter wrappers."""

from __future__ import annotations

from typing import Mapping

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.nonlinear.cut_tf import tf_cut4g_sigma_point_rule
from bayesfilter.nonlinear.sigma_points_tf import (
    tf_svd_sigma_point_log_likelihood_with_rule,
)
from bayesfilter.results_tf import TFFilterValueResult
from bayesfilter.structural_tf import (
    TFStructuralStateSpace,
    structural_block_metadata,
    structural_filter_metadata,
)


def tf_svd_cut4_log_likelihood(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor | None, Mapping[str, tf.Tensor]]:
    """Evaluate the structural SVD-CUT4-G Gaussian value likelihood."""

    rule = tf_cut4g_sigma_point_rule(
        model.partition.state_dim + model.partition.innovation_dim,
    )
    return tf_svd_sigma_point_log_likelihood_with_rule(
        observations,
        model,
        sigma_rule=rule,
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        jitter=jitter,
        return_filtered=return_filtered,
    )


def tf_svd_cut4_filter(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> TFFilterValueResult:
    """Return a value result for the structural SVD-CUT4-G filter."""

    value, filtered_means, filtered_covariances, raw_diagnostics = tf_svd_cut4_log_likelihood(
        observations,
        model,
        placement_floor=placement_floor,
        innovation_floor=innovation_floor,
        rank_tolerance=rank_tolerance,
        jitter=jitter,
        return_filtered=return_filtered,
    )
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
        "factorization": "tf.linalg.eigh",
        "derivative_status_reason": "SVD-CUT derivatives are not certified in value Phase 5.",
    }
    diagnostics = TFFilterDiagnostics(
        backend="tf_svd_cut4",
        mask_convention="none",
        regularization=TFRegularizationDiagnostics(
            jitter=tf.convert_to_tensor(jitter, dtype=tf.float64),
            singular_floor=tf.convert_to_tensor(innovation_floor, dtype=tf.float64),
            floor_count=raw_diagnostics["innovation_floor_count"],
            psd_projection_residual=raw_diagnostics["innovation_psd_projection_residual"],
            implemented_covariance=raw_diagnostics["implemented_innovation_covariance"],
            branch_label="svd_cut4_value",
            derivative_target="blocked",
        ),
        extra=extra,
    )
    return TFFilterValueResult(
        log_likelihood=value,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        metadata=structural_filter_metadata(
            model,
            filter_name="tf_svd_cut4_filter",
            differentiability_status="value_only",
            compiled_status="eager_tf",
        ),
        diagnostics=diagnostics,
    )
