"""Low-rank SPD quadratic geometry fitting utilities.

The fitted geometry is a diagnostic initializer for HMC mass construction.  It
does not certify a MAP, posterior correctness, sampler convergence, or default
readiness.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig, xla_svd

from bayesfilter.inference._exact_incumbent import (
    ExactCandidate,
    candidates_from_rows,
)
from bayesfilter.ops.geometry_random_tf import STREAM_ID, GeometryTensorStream
from bayesfilter.ops.host_tensor_io import numeric_tensor

LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS = (
    "low-rank quadratic geometry diagnostic only",
    "not a certified MAP covariance",
    "not posterior correctness evidence",
    "not HMC convergence evidence",
    "not sampler superiority evidence",
    "not default-readiness evidence",
    "not source-faithful Zhao-Cui evidence",
)


@dataclass(frozen=True)
class LowRankSPDQuadraticGeometryConfig:
    """Configuration for low-rank SPD quadratic geometry fitting."""

    rank: int = 4
    sample_count: int | None = None
    min_samples_per_parameter: int = 5
    trust_radius: float = 1.0
    pilot_radius: float = 0.15
    pilot_direction_count: int | None = None
    holdout_fraction: float = 0.25
    eigenvalue_floor: float = 1.0
    max_condition_number: float = 1.0e3
    fit_max_iterations: int = 300
    fit_tolerance: float = 1.0e-8
    holdout_rmse_abs_tolerance: float = 5.0e-2
    holdout_rmse_rel_tolerance: float = 1.0e-1
    center_score_improvement_factor: float = 0.95
    center_log_prob_tolerance: float = 1.0e-8
    constrain_center_refinement_to_trust_region: bool = False
    seed: int | Sequence[int] = 20260708

    def __post_init__(self) -> None:
        for name in ("rank", "min_samples_per_parameter", "fit_max_iterations"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.sample_count is not None:
            sample_count = int(self.sample_count)
            if sample_count <= 0:
                raise ValueError("sample_count must be positive when supplied")
            object.__setattr__(self, "sample_count", sample_count)
        if self.pilot_direction_count is not None:
            count = int(self.pilot_direction_count)
            if count <= 0:
                raise ValueError("pilot_direction_count must be positive when supplied")
            object.__setattr__(self, "pilot_direction_count", count)
        for name in (
            "trust_radius",
            "pilot_radius",
            "eigenvalue_floor",
            "max_condition_number",
            "fit_tolerance",
            "holdout_rmse_abs_tolerance",
            "holdout_rmse_rel_tolerance",
            "center_score_improvement_factor",
            "center_log_prob_tolerance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        holdout = float(self.holdout_fraction)
        if not math.isfinite(holdout) or not 0.0 <= holdout < 0.8:
            raise ValueError(
                "holdout_fraction must be finite and satisfy 0 <= value < 0.8"
            )
        object.__setattr__(self, "holdout_fraction", holdout)
        if float(self.max_condition_number) <= 1.0:
            raise ValueError("max_condition_number must be greater than 1")
        if float(self.center_score_improvement_factor) >= 1.0:
            raise ValueError("center_score_improvement_factor must be less than 1")
        object.__setattr__(
            self,
            "constrain_center_refinement_to_trust_region",
            bool(self.constrain_center_refinement_to_trust_region),
        )
        object.__setattr__(self, "seed", _normalize_seed(self.seed))

    def payload(self) -> Mapping[str, Any]:
        return {
            "rank": self.rank,
            "sample_count": self.sample_count,
            "min_samples_per_parameter": self.min_samples_per_parameter,
            "trust_radius": self.trust_radius,
            "pilot_radius": self.pilot_radius,
            "pilot_direction_count": self.pilot_direction_count,
            "holdout_fraction": self.holdout_fraction,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "fit_max_iterations": self.fit_max_iterations,
            "fit_tolerance": self.fit_tolerance,
            "holdout_rmse_abs_tolerance": self.holdout_rmse_abs_tolerance,
            "holdout_rmse_rel_tolerance": self.holdout_rmse_rel_tolerance,
            "center_score_improvement_factor": self.center_score_improvement_factor,
            "center_log_prob_tolerance": self.center_log_prob_tolerance,
            "constrain_center_refinement_to_trust_region": (
                self.constrain_center_refinement_to_trust_region
            ),
            "seed": self.seed,
            "random_stream": STREAM_ID,
        }


@dataclass(frozen=True)
class LowRankSPDQuadraticGeometryResult:
    """Structured result for a low-rank SPD quadratic geometry attempt."""

    accepted: bool
    status: str
    dimension: int
    rank: int
    center: tf.Tensor
    scale: tf.Tensor
    precision: tf.Tensor | None
    covariance: tf.Tensor | None
    q_basis: tf.Tensor | None
    linear_term: tf.Tensor | None
    intercept: float | None
    lambda0: float | None
    mu: tf.Tensor | None
    refined_center: tf.Tensor | None
    center_refinement_accepted: bool
    diagnostics: Mapping[str, Any]
    best_evaluated_position: tf.Tensor | None = None
    best_evaluated_value: float | None = None
    best_evaluated_score: tf.Tensor | None = None
    best_evaluated_source: str | None = None
    best_evaluated_index: int | None = None
    exact_evaluation_count: int = 0
    nonclaims: tuple[str, ...] = LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "rank", int(self.rank))
        for name in ("center", "scale"):
            array = numeric_tensor(getattr(self, name), tf.float64)
            object.__setattr__(self, name, array)
        for name in ("precision", "covariance", "q_basis", "linear_term", "mu"):
            value = getattr(self, name)
            if value is not None:
                array = numeric_tensor(value, tf.float64)
                object.__setattr__(self, name, array)
        if self.refined_center is not None:
            refined = numeric_tensor(self.refined_center, tf.float64)
            object.__setattr__(self, "refined_center", refined)
        for name in ("best_evaluated_position", "best_evaluated_score"):
            value = getattr(self, name)
            if value is not None:
                array = tf.reshape(numeric_tensor(value, tf.float64), [-1])
                object.__setattr__(self, name, array)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(
            self, "center_refinement_accepted", bool(self.center_refinement_accepted)
        )
        object.__setattr__(
            self, "exact_evaluation_count", int(self.exact_evaluation_count)
        )
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(
            self, "nonclaims", tuple(str(item) for item in self.nonclaims)
        )

    def payload(self, *, include_arrays: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "bayesfilter.low_rank_spd_quadratic_geometry.v2",
            "accepted": self.accepted,
            "status": self.status,
            "dimension": self.dimension,
            "rank": self.rank,
            "center_refinement_accepted": self.center_refinement_accepted,
            "intercept": self.intercept,
            "lambda0": self.lambda0,
            "best_evaluated_value": self.best_evaluated_value,
            "best_evaluated_source": self.best_evaluated_source,
            "best_evaluated_index": self.best_evaluated_index,
            "exact_evaluation_count": self.exact_evaluation_count,
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if self.mu is not None:
            payload["mu"] = tuple(self.mu.numpy().tolist())
        if self.precision is not None:
            payload["precision_eigen_summary"] = self.diagnostics.get("precision_eigen_summary")
            if payload["precision_eigen_summary"] is None:
                payload["precision_eigen_summary"] = _eigen_summary(self.precision)
        if self.covariance is not None:
            payload["covariance_eigen_summary"] = self.diagnostics.get("covariance_eigen_summary")
            if payload["covariance_eigen_summary"] is None:
                payload["covariance_eigen_summary"] = _eigen_summary(self.covariance)
        if include_arrays:
            payload.update(
                {
                    "center": self.center,
                    "scale": self.scale,
                    "precision": self.precision,
                    "covariance": self.covariance,
                    "q_basis": self.q_basis,
                    "linear_term": self.linear_term,
                    "refined_center": self.refined_center,
                    "best_evaluated_position": self.best_evaluated_position,
                    "best_evaluated_score": self.best_evaluated_score,
                }
            )
        return _json_ready(payload)


def fit_low_rank_spd_quadratic_geometry(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: Any,
    *,
    batched_value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]
    | None = None,
    scale: Any | None = None,
    config: LowRankSPDQuadraticGeometryConfig | None = None,
) -> LowRankSPDQuadraticGeometryResult:
    """Fit a constrained low-rank SPD quadratic geometry around ``center``.

    ``value_and_score_fn`` must accept a one-dimensional TensorFlow tensor and
    return scalar log probability and gradient in the original coordinates.
    When supplied, ``batched_value_and_score_fn`` accepts ``[batch, dimension]``
    and returns ``[batch]`` values plus ``[batch, dimension]`` scores. It is
    used only for the pilot and design clouds; center and refinement checks
    remain on the scalar callback.
    The fitted quadratic is in whitened coordinates ``theta = center + scale*z``.
    """

    from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
    from bayesfilter.inference.quadratic_geometry_full_tf import (
        geometry_program,
        prepare_geometry_inputs,
    )

    cfg = LowRankSPDQuadraticGeometryConfig() if config is None else config
    fixed_center = _vector(center, "center")
    dimension = int(fixed_center.shape[0])
    fixed_scale = _scale_vector(scale, dimension)
    inputs = prepare_geometry_inputs(dimension, cfg)
    program = geometry_program(value_and_score_fn, dimension, cfg,
        batched_callback=batched_value_and_score_fn)
    raw = program(fixed_center, fixed_scale, *inputs)
    return geometry_result(raw, fixed_center, fixed_scale, cfg,
        batched=batched_value_and_score_fn is not None)


def _fit_constrained_quadratic(
    z_train: tf.Tensor,
    y_train: tf.Tensor,
    score_train: tf.Tensor,
    *,
    q_basis: tf.Tensor,
    cfg: LowRankSPDQuadraticGeometryConfig,
    dim: int,
    rank: int,
    center_score_z: tf.Tensor,
) -> Mapping[str, Any]:
    inputs = tuple(
        numeric_tensor(value, tf.float64)
        for value in (z_train, y_train, score_train, q_basis, center_score_z)
    )
    z, _, score, _, center_score = inputs
    if (
        z.shape.rank != 2
        or z.shape[1] != dim
        or score.shape != z.shape
        or center_score.shape != (dim,)
    ):
        return {"status": "fit_shape_mismatch"}
    try:
        result = _run_numerical(
            _quadratic_fit_kernel,
            *inputs,
            tf.constant(cfg.eigenvalue_floor, tf.float64),
            tf.constant(cfg.max_condition_number, tf.float64),
        )
    except tf.errors.OpError:
        return {"status": "score_curvature_lstsq_failed"}
    if not bool(result.pop("finite")):
        return {"status": "fit_nonfinite"}
    resolved = bool(result.pop("design_resolved"))
    retained_condition = result.pop("retained_design_condition")
    roundoff_indicator = result.pop("design_roundoff_indicator")
    roundoff_limit = result.pop("design_roundoff_limit")
    if not resolved:
        return {"status": "fit_design_ill_conditioned",
            "score_design_rank": int(result["score_design_rank"]),
            "score_design_retained_condition_number": float(retained_condition),
            "score_design_roundoff_indicator": float(roundoff_indicator),
            "score_design_roundoff_limit": float(roundoff_limit)}
    result["score_design_condition_number"] = _design_condition_number(
        result.pop("singular_values")
    )
    residual_sum_squares = result.pop("residual_sum_squares")
    result["score_lstsq_residual_sum_squares"] = (
        float(residual_sum_squares)
        if int(result["score_design_rank"]) == rank + 1 and z.shape[0] * dim > rank + 1
        else None
    )
    return {
        "status": "usable",
        "fit_method": "score_difference_linear_least_squares_with_value_intercept",
        "optimizer_converged": True,
        "optimizer_failed": False,
        "optimizer_iterations": 0,
        **result,
    }


def _score_curvature_design(
    z_train, score_train_z, center_score_z, *, q_basis, dim, rank
):
    z = numeric_tensor(z_train, tf.float64)
    score = numeric_tensor(score_train_z, tf.float64)
    center = numeric_tensor(center_score_z, tf.float64)
    q = numeric_tensor(q_basis, tf.float64)
    projected = tf.matmul(z, q)
    design = tf.concat((z[..., None], projected[:, None, :] * q[None, :, :]), axis=2)
    return tf.reshape(design, [-1, rank + 1]), tf.reshape(center[None, :] - score, [-1])


def _quadratic_fit_kernel(z, y, score, q, center_score, floor, condition_cap, *, use_xla_svd=True, active_rows=None,
                          minimum_active_rows=1):
    dim, rank = q.shape
    if active_rows is not None:
        from bayesfilter.inference.quadratic_geometry_rows_tf import (
            compact_fit_statistics,
            compact_qr,
            compact_solution,
        )

        active = tf.range(z.shape[0]) < active_rows
        z = tf.where(active[:, None], z, tf.zeros_like(z))
        y = tf.where(active, y, tf.zeros_like(y))
        score = tf.where(active[:, None], score, tf.zeros_like(score))
    design, response = _score_curvature_design(
        z, score, center_score, q_basis=q, dim=dim, rank=rank
    )
    reduced_q, reduced_r = (tf.linalg.qr(design, full_matrices=False) if active_rows is None
                           else compact_qr(design, active_rows, dim, minimum_active_rows))
    if use_xla_svd:
        decomposition = xla_svd(
            reduced_r, max_iter=100, epsilon=math.ulp(1.0), precision_config=""
        )
        # Raw XlaSvd has no TensorFlow shape inference. Bind its full SVD
        # schema so enclosing conditional/loop state remains statically known.
        rows, columns = reduced_r.shape
        singular = tf.ensure_shape(decomposition.s, [min(rows, columns)])
        left_r = tf.ensure_shape(decomposition.u, [rows, rows])
        right_r = tf.ensure_shape(decomposition.v, [columns, columns])
    else:
        # Explicit graph-reference diagnostic; the default XLA tolerance remains fixed.
        singular, left_r, right_r = tf.linalg.svd(reduced_r, full_matrices=False)
    extent = singular.shape[0]
    right = right_r[:, :extent]
    # NumPy lstsq(rcond=None) uses eps*max(rows,columns), unlike Eigen COD.
    relative_cutoff = (tf.constant(math.ulp(1.0) * max(design.shape), tf.float64) if active_rows is None
        else tf.constant(math.ulp(1.0), tf.float64) * tf.cast(tf.maximum(active_rows * dim, rank + 1), tf.float64))
    threshold = tf.reduce_max(singular) * relative_cutoff
    keep = singular > threshold
    retained_condition = tf.where(tf.reduce_any(keep),
        tf.reduce_max(singular) / tf.reduce_min(tf.where(keep, singular, tf.constant(float('inf'), tf.float64))),
        tf.constant(1., tf.float64))
    roundoff_indicator = relative_cutoff * retained_condition
    roundoff_limit = tf.constant(math.sqrt(math.ulp(1.0)), tf.float64)
    design_resolved = tf.math.is_finite(roundoff_indicator) & (roundoff_indicator <= roundoff_limit)
    inverse = tf.where(keep, tf.math.reciprocal(singular), tf.zeros_like(singular))
    if active_rows is None:
        left = tf.matmul(reduced_q, left_r[:, :extent])
        raw = tf.linalg.matvec(right, inverse * tf.linalg.matvec(left, response, transpose_a=True))
    else:
        raw = compact_solution(reduced_q, left_r[:, :extent], right, inverse, response, active_rows, dim, minimum_active_rows)
    lambda0 = tf.maximum(floor, raw[0])
    raw_mu = raw[1:]
    upper = (condition_cap - 1.0) * lambda0
    mu = tf.clip_by_value(raw_mu, 0.0, upper)
    precision = lambda0 * tf.eye(dim, dtype=tf.float64) + tf.matmul(
        q * mu[None, :], q, transpose_b=True
    )
    precision = 0.5 * (precision + tf.transpose(precision))
    without_intercept = _predict_quadratic(
        z,
        intercept=tf.constant(0.0, tf.float64),
        linear=center_score,
        lambda0=lambda0,
        mu=mu,
        q_basis=q,
    )
    if active_rows is None:
        intercept = tf.reduce_mean(y - without_intercept)
        residual = intercept + without_intercept - y
        score_residual = (
            tf.linalg.matvec(design, tf.concat((lambda0[None], mu), axis=0)) - response
        )
        loss = tf.reduce_mean(residual**2)
        score_rmse = tf.sqrt(tf.reduce_mean(score_residual**2))
        residual_sum_squares = tf.reduce_sum((tf.linalg.matvec(design, raw) - response) ** 2)
    else:
        intercept, loss, score_rmse, residual_sum_squares = compact_fit_statistics(
            y, without_intercept, design, response, raw, tf.concat((lambda0[None], mu), axis=0), active_rows, dim, minimum_active_rows)
    finite = (
        tf.reduce_all(tf.math.is_finite(raw))
        & tf.math.is_finite(loss)
        & tf.math.is_finite(lambda0)
        & (lambda0 > 0.0)
        & tf.reduce_all(tf.math.is_finite(mu))
        & tf.reduce_all(tf.math.is_finite(precision))
        & tf.reduce_all(tf.math.is_finite(center_score))
        & tf.math.is_finite(score_rmse)
    )
    if active_rows is not None:
        finite &= (active_rows >= minimum_active_rows) & (active_rows <= z.shape[0])
    return {
        "loss": loss,
        "score_rmse": score_rmse,
        "finite": finite,
        "design_resolved": design_resolved,
        "retained_design_condition": retained_condition,
        "design_roundoff_indicator": roundoff_indicator,
        "design_roundoff_limit": roundoff_limit,
        "score_design_rank": tf.math.count_nonzero(keep),
        "singular_values": singular,
        "residual_sum_squares": residual_sum_squares,
        "raw_lambda0": raw[0],
        "raw_mu": raw_mu,
        "mu_clipped_count": tf.math.count_nonzero((raw_mu < 0.0) | (raw_mu > upper)),
        "intercept": intercept,
        "linear_term": center_score,
        "lambda0": lambda0,
        "mu": mu,
        "precision": precision,
        "condition_bound": (
            lambda0 + tf.reduce_max(tf.concat((mu, tf.zeros([1], tf.float64)), axis=0))
        )
        / lambda0,
    }


@lru_cache(maxsize=64)
def _compiled_numerical(kernel, signature):
    return tf.function(
        kernel, input_signature=signature, jit_compile=True, autograph=False
    )


def _run_numerical(kernel, *inputs):
    return _compiled_numerical(
        kernel, tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)
    )(*inputs)


def _design_condition_number(singular_values: tf.Tensor) -> float | None:
    values = numeric_tensor(singular_values, tf.float64)
    valid = tf.math.is_finite(values) & (values > 0.0)
    if not bool(tf.reduce_any(valid)):
        return None
    return float(
        tf.reduce_max(tf.where(valid, values, tf.zeros_like(values)))
        / tf.reduce_min(
            tf.where(
                valid, values, tf.fill(values.shape, tf.constant(math.inf, tf.float64))
            )
        )
    )


def _pilot_q_basis(
    value_and_score_fn,
    *,
    batched_value_and_score_fn,
    center,
    scale,
    rank,
    cfg,
    rng,
    center_value,
    center_score_z,
    start_index,
):
    dim = int(center.shape[0])
    if rank == 0:
        return tf.zeros([dim, 0], tf.float64), {"positive_curvature_count": 0}, ()
    count = (
        int(cfg.pilot_direction_count)
        if cfg.pilot_direction_count is not None
        else max(4 * dim, 2 * rank + 8)
    )
    directions = numeric_tensor(rng.normal(size=(count, dim)), tf.float64)
    norms = tf.linalg.norm(directions, axis=1)
    directions = (
        tf.boolean_mask(directions, norms > 0.0)
        / tf.boolean_mask(norms, norms > 0.0)[:, None]
    )
    h = float(cfg.pilot_radius)
    plus = center[None, :] + h * directions * scale[None, :]
    minus = center[None, :] - h * directions * scale[None, :]
    split = int(directions.shape[0])
    if batched_value_and_score_fn is None:
        points = tf.reshape(tf.stack((plus, minus), axis=1), [-1, dim])
        route, batch_size = "tensorflow_scalar_row_loop", 1
    else:
        points = tf.concat((plus, minus), axis=0)
        route, batch_size = "batched_value_and_score", 2 * split
    values, scores = _evaluate_values_scores(
        value_and_score_fn,
        points,
        batched_value_and_score_fn=batched_value_and_score_fn,
    )
    candidates = candidates_from_rows(
        points, values, scores, start_index=start_index, source_role="pilot"
    )
    if batched_value_and_score_fn is None:
        score_plus, score_minus = scores[::2], scores[1::2]
    else:
        score_plus, score_minus = scores[:split], scores[split:]
    q_full, eigenvalues, count_positive, minimum, maximum = _run_numerical(
        _pilot_sketch_kernel,
        directions,
        score_plus,
        score_minus,
        scale,
        tf.constant(h, tf.float64),
    )
    q = _run_numerical(_thin_qr, q_full[:, :rank])
    positive = int(count_positive)
    return (
        q,
        {
            "pilot_direction_count": count,
            "finite_positive_curvature_count": positive,
            "curvature_min": float(minimum) if positive else None,
            "curvature_max": float(maximum) if positive else None,
            "curvature_source": "central_score_difference_directional_curvature",
            "center_score_norm": float(tf.linalg.norm(center_score_z)),
            "sketch_eigenvalues": tuple(float(value) for value in eigenvalues[::-1]),
            "basis_source": "directional_curvature_sketch"
            if positive
            else "identity_fallback",
            "evaluation_route": route,
            "evaluation_batch_size": batch_size,
        },
        tuple(candidates),
    )


def _thin_qr(matrix):
    return tf.linalg.qr(matrix, full_matrices=False)[0]


def _pilot_sketch_kernel(directions, plus, minus, scale, h, *, eigenpairs=None):
    curvature = tf.reduce_sum((minus * scale - plus * scale) * directions, axis=1) / (
        2.0 * h
    )
    valid = (
        tf.reduce_all(tf.math.is_finite(plus), axis=1)
        & tf.reduce_all(tf.math.is_finite(minus), axis=1)
        & tf.math.is_finite(curvature)
        & (curvature > 0.0)
    )
    count, dimension = directions.shape

    def add(i, sketch):
        contribution = curvature[i] * (directions[i, :, None] * directions[i, None, :])
        return i + 1, tf.where(valid[i], sketch + contribution, sketch)

    _, sketch = tf.while_loop(
        lambda i, _: i < count,
        add,
        (tf.constant(0), tf.zeros([dimension, dimension], tf.float64)),
        maximum_iterations=count,
    )
    positive = tf.math.count_nonzero(valid)

    def fitted():
        symmetric = .5 * (sketch + tf.transpose(sketch))
        if eigenpairs is None:
            values, vectors = xla_self_adjoint_eig(
                symmetric, lower=True, max_iter=100, epsilon=math.ulp(1.0))
        else:
            values, vectors = eigenpairs(symmetric)
        return vectors[:, ::-1], values

    vectors, values = tf.cond(
        positive > 0,
        fitted,
        lambda: (
            tf.eye(dimension, dtype=tf.float64),
            tf.zeros([dimension], tf.float64),
        ),
    )
    return (
        vectors,
        values,
        positive,
        tf.reduce_min(tf.where(valid, curvature, tf.constant(math.inf, tf.float64))),
        tf.reduce_max(tf.where(valid, curvature, tf.zeros_like(curvature))),
    )


def _evaluate_center_refinement(
    *,
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: tf.Tensor,
    scale: tf.Tensor,
    precision: tf.Tensor,
    linear: tf.Tensor,
    cfg: LowRankSPDQuadraticGeometryConfig,
    center_value: float,
    center_score_norm: float,
) -> Mapping[str, Any]:
    constrained = bool(cfg.constrain_center_refinement_to_trust_region)
    if constrained:
        step = _solve_spd_quadratic_trust_region(
            precision,
            linear,
            radius=float(cfg.trust_radius),
        )
        if step["status"] != "usable":
            return {
                "accepted": False,
                "reason": str(step["status"]),
                "step_method": "exact_spd_quadratic_trust_region",
                "trust_region_constrained": True,
            }
        z_star = numeric_tensor(step["step"], tf.float64)
    else:
        try:
            z_star = tf.linalg.solve(precision, linear[:, None])[:, 0]
        except tf.errors.OpError:
            return {"accepted": False, "reason": "precision_solve_failed"}
        step = {
            "status": "usable",
            "step": z_star,
            "step_method": "unconstrained_spd_quadratic_solve",
            "boundary_active": False,
            "lagrange_multiplier": 0.0,
            "predicted_improvement": float(
                tf.reduce_sum(linear * z_star)
                - 0.5 * tf.reduce_sum(z_star * tf.linalg.matvec(precision, z_star))
            ),
        }
    z_norm = float(tf.linalg.norm(z_star))
    refined = center + z_star * scale
    value, score, status = _evaluate_value_score(value_and_score_fn, refined)
    if status != "finite":
        return {
            "accepted": False,
            "reason": "refined_value_or_score_nonfinite",
            "z_norm": z_norm,
            "refined_center": refined,
            "step_method": step["step_method"],
            "trust_region_constrained": constrained,
            "boundary_active": bool(step["boundary_active"]),
            "lagrange_multiplier": float(step["lagrange_multiplier"]),
            "predicted_improvement": float(step["predicted_improvement"]),
            "exact_evaluation_count": 1,
            "refined_target_finite": False,
        }
    score_norm = float(tf.linalg.norm(score * scale))
    actual_improvement = float(value - center_value)
    predicted_improvement = float(step["predicted_improvement"])
    improvement_ratio = (
        actual_improvement / predicted_improvement
        if predicted_improvement > 0.0
        else None
    )
    accepted = bool(
        z_norm <= float(cfg.trust_radius)
        and value >= center_value - float(cfg.center_log_prob_tolerance)
        and score_norm <= float(cfg.center_score_improvement_factor) * center_score_norm
        and (not constrained or actual_improvement > 0.0)
        and (not constrained or predicted_improvement > 0.0)
        and (
            not constrained
            or (improvement_ratio is not None and math.isfinite(improvement_ratio))
        )
    )
    reasons = []
    if z_norm > float(cfg.trust_radius):
        reasons.append("outside_trust_radius")
    if value < center_value - float(cfg.center_log_prob_tolerance):
        reasons.append("log_prob_decreased")
    if score_norm > float(cfg.center_score_improvement_factor) * center_score_norm:
        reasons.append("score_norm_not_improved_enough")
    if constrained and actual_improvement <= 0.0:
        reasons.append("actual_improvement_not_positive")
    if constrained and predicted_improvement <= 0.0:
        reasons.append("predicted_improvement_not_positive")
    if constrained and (
        improvement_ratio is None or not math.isfinite(improvement_ratio)
    ):
        reasons.append("improvement_ratio_nonfinite")
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else ";".join(reasons),
        "z_norm": z_norm,
        "step_method": step["step_method"],
        "trust_region_constrained": constrained,
        "boundary_active": bool(step["boundary_active"]),
        "lagrange_multiplier": float(step["lagrange_multiplier"]),
        "predicted_improvement": predicted_improvement,
        "actual_improvement": actual_improvement,
        "actual_to_predicted_improvement_ratio": improvement_ratio,
        "center_log_prob": float(center_value),
        "refined_log_prob": float(value),
        "center_score_norm": float(center_score_norm),
        "refined_score_norm": score_norm,
        "refined_score": score,
        "refined_target_finite": True,
        "exact_evaluation_count": 1,
        "refined_center": refined,
    }


def _canonical_replay(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    incumbent: ExactCandidate | None,
    *,
    evaluation_index: int,
) -> Mapping[str, Any]:
    """Replay the selected exact row without changing its earlier provenance."""

    if incumbent is None:
        return {"attempted": False, "valid": False, "matches": False}
    value, score, status = _evaluate_value_score(
        value_and_score_fn, numeric_tensor(incumbent.position, tf.float64)
    )
    valid = status == "finite"
    matches = bool(
        valid
        and abs(value - incumbent.value) <= 1.0e-12 + 1.0e-10 * abs(incumbent.value)
        and bool(
            tf.reduce_all(
                tf.abs(score - incumbent.score)
                <= 1.0e-11 + 1.0e-9 * tf.abs(incumbent.score)
            )
        )
    )
    return {
        "attempted": True,
        "evaluation_index": int(evaluation_index),
        "valid": valid,
        "matches": matches,
        "value": value,
        "source": incumbent.source_role,
    }


def _solve_spd_quadratic_trust_region(precision, linear, *, radius):
    """Solve the same SPD quadratic trust ball with native bracketing/bisection."""
    matrix = numeric_tensor(precision, tf.float64)
    vector = tf.reshape(numeric_tensor(linear, tf.float64), [-1])
    radius_value = float(radius)
    if (
        matrix.shape.rank != 2
        or matrix.shape != (vector.shape[0], vector.shape[0])
        or not bool(tf.reduce_all(tf.math.is_finite(matrix)))
        or not bool(tf.reduce_all(tf.math.is_finite(vector)))
        or not math.isfinite(radius_value)
        or radius_value <= 0.0
    ):
        return {"status": "trust_region_invalid_input"}
    try:
        result = _run_numerical(
            _trust_region_kernel, matrix, vector, tf.constant(radius_value, tf.float64)
        )
    except tf.errors.OpError:
        return {"status": "trust_region_eigendecomposition_failed"}
    status = int(result.pop("status_code"))
    if status:
        return {
            "status": (
                "trust_region_precision_not_spd",
                "trust_region_bracket_failed",
                "trust_region_solution_invalid",
            )[status - 1]
        }
    result = {
        key: value if key == "step" else value.numpy().item()
        for key, value in result.items()
    }
    return {
        "status": "usable",
        "step_method": "exact_spd_quadratic_trust_region",
        **result,
    }


def _trust_region_kernel(matrix, vector, radius, *, eigenpairs=None):
    symmetric = 0.5 * (matrix + tf.transpose(matrix))
    if eigenpairs is None:
        eigenvalues, eigenvectors = xla_self_adjoint_eig(
            symmetric, lower=True, max_iter=100, epsilon=math.ulp(1.0)
        )
    else:
        eigenvalues, eigenvectors = eigenpairs(symmetric)
    eigenvalues = tf.ensure_shape(eigenvalues, vector.shape)
    eigenvectors = tf.ensure_shape(eigenvectors, matrix.shape)
    spd = tf.reduce_all(tf.math.is_finite(eigenvalues)) & (
        tf.reduce_min(eigenvalues) > 0.0
    )
    projected = tf.linalg.matvec(eigenvectors, vector, transpose_a=True)

    def solution(multiplier):
        return tf.linalg.matvec(eigenvectors, projected / (eigenvalues + multiplier))

    unconstrained = solution(tf.constant(0.0, tf.float64))
    norm = tf.linalg.norm(unconstrained)
    boundary = norm > radius * (1.0 + 1e-12)

    def constrained():
        _, upper = tf.while_loop(
            lambda i, u: tf.math.is_finite(u) & (tf.linalg.norm(solution(u)) > radius),
            lambda i, u: (i + 1, 2.0 * u),
            (tf.constant(0), tf.constant(1.0, tf.float64)),
            maximum_iterations=1024,
        )

        def bisect(i, lower, upper):
            middle = 0.5 * (lower + upper)
            outside = tf.linalg.norm(solution(middle)) > radius
            return (
                i + 1,
                tf.where(outside, middle, lower),
                tf.where(outside, upper, middle),
            )

        iterations, _, multiplier = tf.while_loop(
            lambda i, l, u: (
                (i < 100)
                & (u - l > 1e-12 * tf.maximum(tf.constant(1.0, tf.float64), u))
            ),
            bisect,
            (tf.constant(0), tf.constant(0.0, tf.float64), upper),
            maximum_iterations=100,
        )
        return solution(multiplier), multiplier, iterations, tf.math.is_finite(upper)

    step, multiplier, iterations, bracket = tf.cond(
        boundary,
        constrained,
        lambda: (
            unconstrained,
            tf.constant(0.0, tf.float64),
            tf.constant(0),
            tf.constant(True),
        ),
    )
    step_norm = tf.linalg.norm(step)
    predicted = tf.reduce_sum(vector * step) - 0.5 * tf.reduce_sum(
        step * tf.linalg.matvec(symmetric, step)
    )
    valid = (
        tf.reduce_all(tf.math.is_finite(step))
        & tf.math.is_finite(predicted)
        & (step_norm <= radius * (1.0 + 1e-10))
    )
    return {
        "status_code": tf.where(~spd, 1, tf.where(~bracket, 2, tf.where(~valid, 3, 0))),
        "step": step,
        "boundary_active": boundary,
        "lagrange_multiplier": multiplier,
        "bisection_iterations": iterations,
        "unconstrained_step_norm": norm,
        "step_norm": step_norm,
        "predicted_improvement": predicted,
    }


def _evaluate_values_scores(
    value_and_score_fn, theta_samples, *, batched_value_and_score_fn=None
):
    samples = numeric_tensor(theta_samples, tf.float64)
    if samples.shape.rank != 2:
        raise ValueError("theta_samples must have shape [batch, dimension]")
    count, dimension = samples.shape
    nan = tf.constant(math.nan, tf.float64)
    try:
        call = _compiled_cloud(
            value_and_score_fn, batched_value_and_score_fn, count, dimension
        )
        values, scores = call(samples)
    except Exception:  # noqa: BLE001 - preserve fail-closed callback boundary.
        return tf.fill([count], nan), tf.fill(samples.shape, nan)
    finite = tf.math.is_finite(values) & tf.reduce_all(
        tf.math.is_finite(scores), axis=1
    )
    return tf.where(finite, values, nan), tf.where(finite[:, None], scores, nan)


@lru_cache(maxsize=64)
def _compiled_cloud(scalar, batched, count, dimension):
    @tf.function(
        input_signature=[tf.TensorSpec([count, dimension], tf.float64)],
        jit_compile=True,
        autograph=False,
    )
    def evaluate(points):
        if batched is not None:
            values, scores = batched(points)
            return (
                tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [count]),
                tf.ensure_shape(
                    tf.convert_to_tensor(scores, tf.float64), [count, dimension]
                ),
            )

        def row(i, values, scores):
            value, score = scalar(points[i])
            value = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
            score = tf.ensure_shape(
                tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1]), [dimension]
            )
            return (
                i + 1,
                tf.tensor_scatter_nd_update(values, [[i]], [value]),
                tf.tensor_scatter_nd_update(scores, [[i]], score[None, :]),
            )

        _, values, scores = tf.while_loop(
            lambda i, *_: i < count,
            row,
            (
                tf.constant(0),
                tf.zeros([count], tf.float64),
                tf.zeros([count, dimension], tf.float64),
            ),
            maximum_iterations=count,
        )
        return values, scores

    return evaluate


def _evaluate_value_score(value_and_score_fn, theta):
    point = numeric_tensor(theta, tf.float64)
    try:
        value, score = _run_numerical(value_and_score_fn, point)
        value = float(tf.convert_to_tensor(value, tf.float64))
        score = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
    except Exception:  # noqa: BLE001 - preserve fail-closed callback boundary.
        return (
            math.nan,
            tf.fill(point.shape, tf.constant(math.nan, tf.float64)),
            "exception",
        )
    if not math.isfinite(value) or not bool(tf.reduce_all(tf.math.is_finite(score))):
        return value, score, "nonfinite"
    return value, score, "finite"


def _predict_quadratic(
    z: tf.Tensor,
    *,
    intercept: float,
    linear: tf.Tensor,
    lambda0: float,
    mu: tf.Tensor,
    q_basis: tf.Tensor,
) -> tf.Tensor:
    zq = tf.matmul(z, q_basis)
    return (
        tf.convert_to_tensor(intercept, tf.float64)
        + tf.linalg.matvec(z, linear)
        - 0.5
        * tf.convert_to_tensor(lambda0, tf.float64)
        * tf.reduce_sum(tf.square(z), axis=1)
        - 0.5 * tf.reduce_sum(tf.square(zq) * mu[None, :], axis=1)
    )


def _sample_trust_ball(
    sample_count: int,
    dim: int,
    *,
    radius: float,
    rng: GeometryTensorStream,
) -> tf.Tensor:
    return numeric_tensor(
        rng.ball(int(sample_count), int(dim), radius=float(radius)), tf.float64
    )


def _rejected_result(
    *,
    status: str,
    center: tf.Tensor,
    scale: tf.Tensor,
    dim: int,
    rank: int,
    diagnostics: Mapping[str, Any],
    incumbent: ExactCandidate | None = None,
    exact_evaluation_count: int = 0,
) -> LowRankSPDQuadraticGeometryResult:
    return LowRankSPDQuadraticGeometryResult(
        accepted=False,
        status=status,
        dimension=dim,
        rank=rank,
        center=center,
        scale=scale,
        precision=None,
        covariance=None,
        q_basis=None,
        linear_term=None,
        intercept=None,
        lambda0=None,
        mu=None,
        refined_center=None,
        center_refinement_accepted=False,
        diagnostics={**dict(diagnostics), "rejection_status": status},
        best_evaluated_position=None if incumbent is None else incumbent.position,
        best_evaluated_value=None if incumbent is None else incumbent.value,
        best_evaluated_score=None if incumbent is None else incumbent.score,
        best_evaluated_source=None if incumbent is None else incumbent.source_role,
        best_evaluated_index=None if incumbent is None else incumbent.evaluation_index,
        exact_evaluation_count=exact_evaluation_count,
    )


def _base_diagnostics(
    *,
    cfg: LowRankSPDQuadraticGeometryConfig,
    dim: int,
    rank: int,
    regression_parameter_count: int,
    required_finite_samples: int,
    sample_count: int,
    center_value: float,
    center_score_norm: float | None,
) -> dict[str, Any]:
    return {
        "config": cfg.payload(),
        "random_stream": STREAM_ID,
        "dimension": int(dim),
        "rank": int(rank),
        "regression_parameter_count": int(regression_parameter_count),
        "required_finite_samples": int(required_finite_samples),
        "sample_count": int(sample_count),
        "center_log_prob": float(center_value),
        "center_score_norm": center_score_norm,
        "coordinate_system": "whitened_center_plus_scale_times_z",
        "precision_form": "lambda0_identity_plus_q_diag_mu_q_transpose",
        "sample_ratio_rule": "finite_sample_count >= min_samples_per_parameter * regression_parameter_count",
        "classification": "extension_or_invention",
        "reports_map_quality": False,
        "reports_hmc_convergence": False,
        "reports_default_readiness": False,
    }


def _vector(value: Any, name: str) -> tf.Tensor:
    vector = tf.reshape(numeric_tensor(value, tf.float64), [-1])
    if vector.shape.rank != 1 or vector.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(tf.reduce_all(tf.math.is_finite(vector))):
        raise ValueError(f"{name} must be finite")
    return vector


def _scale_vector(value: Any | None, dim: int) -> tf.Tensor:
    if value is None:
        scale = tf.ones([int(dim)], tf.float64)
    else:
        scale = tf.reshape(numeric_tensor(value, tf.float64), [-1])
        if scale.shape[0] == 1:
            scale = tf.fill([int(dim)], scale[0])
    if scale.shape != (int(dim),):
        raise ValueError("scale must be scalar or match center dimension")
    if not bool(tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.0))):
        raise ValueError("scale must be positive finite")
    return scale


def _normalize_seed(seed: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        values = tuple(int(item) for item in seed)
        if len(values) == 0:
            raise ValueError("seed sequence must be non-empty")
        if len(values) == 1:
            return values[0], values[0] ^ 0x9E3779B9
        return values[0], values[1]
    value = int(seed)
    return value, value ^ 0x9E3779B9


def _eigen_summary(matrix: Any) -> Mapping[str, Any]:
    square = numeric_tensor(matrix, tf.float64)
    symmetric = 0.5 * (square + tf.transpose(square))
    eigvals = tf.linalg.eigvalsh(symmetric)
    finite = bool(tf.reduce_all(tf.math.is_finite(eigvals)))
    positive = bool(finite and float(tf.reduce_min(eigvals)) > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": float(tf.reduce_min(eigvals)) if finite else float("nan"),
        "max": float(tf.reduce_max(eigvals)) if finite else float("nan"),
        "condition_number": (
            float(tf.reduce_max(eigvals) / tf.reduce_min(eigvals))
            if positive
            else float("inf")
        ),
        "eigenvalues": tuple(float(value) for value in eigvals),
    }


def _rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    return float(
        tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    numeric_tensor(y_true, tf.float64)
                    - numeric_tensor(y_pred, tf.float64)
                )
            )
        )
    )


def _artifact_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if tf.is_tensor(value) or hasattr(value, "__array_interface__"):
        return numeric_tensor(value).numpy().tolist()
    return value
