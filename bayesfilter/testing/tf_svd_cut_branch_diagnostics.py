"""SVD-CUT branch-frequency diagnostics for v1 readiness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import tensorflow as tf

from bayesfilter.nonlinear.svd_cut_derivatives_tf import tf_svd_cut4_score_hessian
from bayesfilter.structural_tf import TFStructuralStateSpace


TFStructuralModelBuilder = Callable[[tf.Tensor], TFStructuralStateSpace]


@dataclass(frozen=True)
class SVDCUTBranchSummary:
    """Small eager-mode summary of SVD-CUT derivative branch outcomes."""

    total_count: int
    smooth_count: int
    active_floor_count: int
    weak_spectral_gap_count: int
    nonfinite_count: int
    other_blocked_count: int
    min_placement_eigen_gap: float
    min_innovation_eigen_gap: float
    max_support_residual: float
    max_deterministic_residual: float
    max_integration_rank: int
    max_point_count: int

    @property
    def smooth_fraction(self) -> float:
        return 0.0 if self.total_count == 0 else self.smooth_count / self.total_count


def svd_cut_branch_frequency_summary(
    observations: tf.Tensor,
    parameter_grid: tf.Tensor,
    model_builder: TFStructuralModelBuilder,
    *,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    jitter: tf.Tensor | float = 0.0,
) -> SVDCUTBranchSummary:
    """Aggregate SVD-CUT derivative branch labels over a small parameter grid."""

    grid = tf.convert_to_tensor(parameter_grid, dtype=tf.float64)
    smooth_count = 0
    active_floor_count = 0
    weak_spectral_gap_count = 0
    nonfinite_count = 0
    other_blocked_count = 0
    min_placement_gap = float("inf")
    min_innovation_gap = float("inf")
    max_support_residual = 0.0
    max_deterministic_residual = 0.0
    max_integration_rank = 0
    max_point_count = 0

    for row in tf.unstack(grid, axis=0):
        try:
            result = tf_svd_cut4_score_hessian(
                observations,
                row,
                model_builder,
                placement_floor=placement_floor,
                innovation_floor=innovation_floor,
                rank_tolerance=rank_tolerance,
                spectral_gap_tolerance=spectral_gap_tolerance,
                jitter=jitter,
            )
            smooth_count += 1
            extra = result.diagnostics.extra
            min_placement_gap = min(
                min_placement_gap,
                float(tf.convert_to_tensor(extra["min_placement_eigen_gap"]).numpy()),
            )
            min_innovation_gap = min(
                min_innovation_gap,
                float(tf.convert_to_tensor(extra["min_innovation_eigen_gap"]).numpy()),
            )
            max_support_residual = max(
                max_support_residual,
                float(tf.convert_to_tensor(extra["support_residual"]).numpy()),
            )
            max_deterministic_residual = max(
                max_deterministic_residual,
                float(tf.convert_to_tensor(extra["deterministic_residual"]).numpy()),
            )
            max_integration_rank = max(
                max_integration_rank,
                int(tf.convert_to_tensor(extra["max_integration_rank"]).numpy()),
            )
            max_point_count = max(
                max_point_count,
                int(tf.convert_to_tensor(extra["point_count"]).numpy()),
            )
        except tf.errors.InvalidArgumentError as exc:
            message = str(exc)
            if "blocked_active_floor" in message:
                active_floor_count += 1
            elif "blocked_weak_spectral_gap" in message:
                weak_spectral_gap_count += 1
            elif "blocked_nonfinite_value" in message:
                nonfinite_count += 1
            else:
                other_blocked_count += 1

    total_count = int(grid.shape[0])
    if smooth_count == 0:
        min_placement_gap = float("nan")
        min_innovation_gap = float("nan")

    return SVDCUTBranchSummary(
        total_count=total_count,
        smooth_count=smooth_count,
        active_floor_count=active_floor_count,
        weak_spectral_gap_count=weak_spectral_gap_count,
        nonfinite_count=nonfinite_count,
        other_blocked_count=other_blocked_count,
        min_placement_eigen_gap=min_placement_gap,
        min_innovation_eigen_gap=min_innovation_gap,
        max_support_residual=max_support_residual,
        max_deterministic_residual=max_deterministic_residual,
        max_integration_rank=max_integration_rank,
        max_point_count=max_point_count,
    )
