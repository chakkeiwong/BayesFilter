"""Fixed-center regional score curvature for an opt-in position initializer.

The numerical controller executes as one TensorFlow float64 XLA program.
Configuration validation and completed-result reporting remain on the host.
It neither optimizes the center nor builds an HMC mass artifact. A successful
regional quadratic fit is not evidence of posterior whitening or convergence.
"""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf

from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from bayesfilter.inference.posterior_curvature_tf import posterior_curvature_controller

POSTERIOR_CURVATURE_REFINEMENT_NONCLAIMS = (
    "fixed-center regional score-curvature initializer only",
    "the supplied center is not optimized or certified as a MAP",
    "not a posterior covariance correctness claim",
    "not an HMC mass matrix or HMC readiness result",
    "not NeuTra training or whitening evidence",
    "not convergence or default-readiness evidence",
)


@dataclass(frozen=True)
class PosteriorCurvatureRefinementConfig:
    """Prospective engineering gates, not universal scientific thresholds.

    ``coordinate_half_width`` is the box half-width for ``uniform_box`` and the
    Euclidean radius for ``uniform_ball``. A ball radius of one means
    ``||z||_2 <= 1`` in pilot coordinates; it is a geometric local envelope,
    not a 68% posterior region. The box fit is the default; the independent
    standard-normal proposal audit is mandatory for either fit design.
    Counts include padded and unsuccessful
    attempted rows. Lineage must contain JSON-compatible data and is copied on
    construction.
    """

    coordinate_half_width: float = 1.0
    replicate_count: int = 2
    rows_per_partition: int | None = None
    batch_size: int = 64
    seed: int = 20260908
    max_physical_rows: int = 10000
    max_design_condition_number: float = 1.0e6
    max_precision_condition_number: float = 1.0e10
    selection_relative_rmse_cap: float = 0.20
    audit_relative_rmse_cap: float = 0.20
    proposal_relative_rmse_cap: float = 0.35
    replicate_generalized_eigenvalue_spread_cap: float = 1.5
    factor_relative_tolerance: float = 1.0e-10
    factor_absolute_tolerance: float = 1.0e-12
    lineage: Mapping[str, Any] = field(default_factory=dict)
    fit_design: str = field(default="uniform_box", kw_only=True)

    def __post_init__(self) -> None:
        for name in (
            "coordinate_half_width", "max_design_condition_number",
            "max_precision_condition_number", "selection_relative_rmse_cap",
            "audit_relative_rmse_cap", "proposal_relative_rmse_cap",
            "replicate_generalized_eigenvalue_spread_cap",
            "factor_relative_tolerance", "factor_absolute_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if self.fit_design not in {"uniform_box", "uniform_ball"}:
            raise ValueError("fit_design must be uniform_box or uniform_ball")
        if min(self.max_design_condition_number, self.max_precision_condition_number) <= 1.0:
            raise ValueError("condition number limits must exceed one")
        if self.replicate_generalized_eigenvalue_spread_cap < 1.0:
            raise ValueError("replicate spread cap must be at least one")
        for name in ("replicate_count", "batch_size", "seed", "max_physical_rows"):
            object.__setattr__(self, name, operator.index(getattr(self, name)))
        if self.replicate_count < 2 or self.batch_size < 2:
            raise ValueError("replicate_count and batch_size must be at least two")
        if not -(2**31) <= self.seed < 2**31:
            raise ValueError("seed must fit int32")
        if self.max_physical_rows <= 0:
            raise ValueError("max_physical_rows must be positive")
        if self.rows_per_partition is not None:
            rows = operator.index(self.rows_per_partition)
            if rows <= 0:
                raise ValueError("rows_per_partition must be positive")
            object.__setattr__(self, "rows_per_partition", rows)
        object.__setattr__(self, "lineage", json.loads(json.dumps(dict(self.lineage), allow_nan=False)))


@dataclass(frozen=True)
class PosteriorCurvatureRefinementResult:
    """Tensor snapshots and diagnostics; rejected results expose no geometry.

    ``precision_z`` is K in pilot coordinates, not raw position precision.
    ``refined_covariance`` is F K^-1 F.T. ``refined_factor`` is its lower
    Cholesky, ready for an explicit caller-side guide update after acceptance.
    """

    accepted: bool
    status: str
    center: tf.Tensor
    pilot_factor: tf.Tensor
    refined_factor: tf.Tensor | None
    refined_covariance: tf.Tensor | None
    precision_z: tf.Tensor | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = POSTERIOR_CURVATURE_REFINEMENT_NONCLAIMS

    def payload(self) -> dict[str, Any]:
        """Return strict-JSON data; nonfinite diagnostic numbers become null."""
        return _json_ready({
            "schema": "bayesfilter.posterior_curvature_refinement.v1",
            "accepted": self.accepted, "status": self.status,
            "center": self.center, "pilot_factor": self.pilot_factor,
            "refined_factor": self.refined_factor,
            "refined_covariance": self.refined_covariance,
            "precision_z": self.precision_z, "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        })


def refine_posterior_local_curvature(
    batched_value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: Any,
    pilot_factor: Any,
    *,
    batched_eligibility_fn: Callable[[tf.Tensor], tf.Tensor],
    config: PosteriorCurvatureRefinementConfig | None = None,
) -> PosteriorCurvatureRefinementResult:
    """Fit fixed-intercept score curvature without recentering or clipping.

    For row positions theta = c + z @ F.T, transform raw analytical scores by
    score_z = score_theta @ F. Fit g_z(0) - g_z(z) ~= z @ K.T using independent
    training/selection partitions. Freeze the mean accepted precision before
    the untouched audit; then validate a separate Gaussian cloud using the
    refined factor. Neither holdout may refit, shrink, or select a fallback.

    Callbacks receive a fixed [batch_size, dimension] float64 tensor. Return
    matching float64 log densities [batch_size] and analytical scores of the
    same target [batch_size, dimension]. Eligibility must be bool [batch_size]
    and identify finite rejection sentinels. Programming errors propagate;
    numerical/support failures return a rejection with no usable geometry.

    Target and eligibility callbacks must support TensorFlow tracing. All
    numerical control runs inside one XLA invocation, without host callbacks or
    an eager fallback. Call this reporting API outside tf.function; tensor-only
    consumers can compose the native posterior_curvature_controller directly.
    """
    if not tf.executing_eagerly():
        raise RuntimeError("completed curvature reports require eager execution")
    cfg = PosteriorCurvatureRefinementConfig() if config is None else config
    fixed_center = _vector(center, "center")
    dimension = int(fixed_center.shape[0])
    factor = _factor(pilot_factor, dimension)
    rows = max(32, 4 * dimension) if cfg.rows_per_partition is None else cfg.rows_per_partition
    if rows < dimension:
        raise ValueError("rows_per_partition must be at least the dimension")
    padded = ((rows + cfg.batch_size - 1) // cfg.batch_size) * cfg.batch_size
    if cfg.batch_size + (2 * cfg.replicate_count + 2) * padded > cfg.max_physical_rows:
        raise ValueError("configured curvature refinement exceeds max_physical_rows before target calls")
    program = posterior_curvature_controller(
        batched_value_and_score_fn, batched_eligibility_fn, dimension, cfg,
    )
    record = program(fixed_center, factor, tf.constant(cfg.seed, tf.int32))
    return posterior_curvature_result(record, fixed_center, factor, cfg)


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _json_ready(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _vector(value: Any, name: str) -> tf.Tensor:
    tensor = tf.identity(tf.convert_to_tensor(value))
    if tensor.dtype != tf.float64:
        raise TypeError(f"{name} must have dtype float64")
    if tensor.shape.rank != 1 or tensor.shape[0] == 0:
        raise ValueError(f"{name} must have shape [dimension]")
    if not _finite(tensor):
        raise ValueError(f"{name} must be finite")
    return tensor


def _factor(value: Any, dimension: int) -> tf.Tensor:
    tensor = tf.identity(tf.convert_to_tensor(value))
    if tensor.dtype != tf.float64:
        raise TypeError("pilot_factor must have dtype float64")
    if tensor.shape != (dimension, dimension):
        raise ValueError("pilot_factor must have shape [dimension, dimension]")
    if not _finite(tensor):
        raise ValueError("pilot_factor must be finite")
    if not bool(tf.reduce_all(tensor == tf.linalg.band_part(tensor, -1, 0))):
        raise ValueError("pilot_factor must be lower triangular")
    if not bool(tf.reduce_all(tf.linalg.diag_part(tensor) > 0.0)):
        raise ValueError("pilot_factor must have a positive diagonal")
    return tensor


__all__ = [
    "POSTERIOR_CURVATURE_REFINEMENT_NONCLAIMS", "PosteriorCurvatureRefinementConfig",
    "PosteriorCurvatureRefinementResult", "refine_posterior_local_curvature",
]
