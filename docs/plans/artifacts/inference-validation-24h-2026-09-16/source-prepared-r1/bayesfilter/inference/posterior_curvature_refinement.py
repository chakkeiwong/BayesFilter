"""Fixed-center regional score curvature for an opt-in position initializer.

The orchestration is bounded and eager; numerical work uses TensorFlow float64.
It neither optimizes the center nor builds an HMC mass artifact. A successful
regional quadratic fit is not evidence of posterior whitening or convergence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
import json
import math
import operator
from typing import Any

import tensorflow as tf

from bayesfilter.inference.score_curvature_tf import (
    _relative_response_rmse,
    fit_dense_score_precision_tf,
)


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


@dataclass
class _EvaluationLedger:
    physical_rows: int = 0
    logical_rows: int = 0
    padded_rows: int = 0
    eligibility_batches: int = 0
    callback_batches: int = 0
    target_rows: int = 0
    target_logical_rows: int = 0
    completed_logical_rows: int = 0
    role: str = "center"
    partitions: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **counts: int) -> None:
        for name, count in counts.items():
            setattr(self, name, getattr(self, name) + count)
            self.partitions[-1][name] = self.partitions[-1].get(name, 0) + count


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

    Call this eager orchestrator outside tf.function. A stable compiled target
    is supported; compiling this orchestration or an HMC chain is not claimed.
    """
    if not tf.executing_eagerly():
        raise RuntimeError("curvature refinement orchestration requires eager execution")
    cfg = PosteriorCurvatureRefinementConfig() if config is None else config
    fixed_center = _vector(center, "center")
    dimension = int(fixed_center.shape[0])
    factor = _factor(pilot_factor, dimension)
    rows = max(32, 4 * dimension) if cfg.rows_per_partition is None else cfg.rows_per_partition
    if rows < dimension:
        raise ValueError("rows_per_partition must be at least the dimension")
    fit_partitions = 2 * cfg.replicate_count
    planned_rows = cfg.batch_size + (fit_partitions + 2) * _padded_rows(rows, cfg.batch_size)
    if planned_rows > cfg.max_physical_rows:
        raise ValueError("configured curvature refinement exceeds max_physical_rows before target calls")
    ledger = _EvaluationLedger()
    details: dict[str, Any] = {}

    def finish(status: str, covariance=None, refined_factor=None, precision=None):
        accepted = status == "eligible_for_local_position_factor"
        diagnostics = {
            name: getattr(ledger, name) for name in (
                "physical_rows", "logical_rows", "padded_rows", "eligibility_batches",
                "callback_batches", "target_rows", "target_logical_rows", "completed_logical_rows",
            )
        }
        diagnostics.update({
            "logical_rows_per_partition": rows,
            "physical_rows_per_partition": _padded_rows(rows, cfg.batch_size),
            "planned_physical_rows": planned_rows, "batch_size": cfg.batch_size,
            "max_physical_rows": cfg.max_physical_rows, "seed": cfg.seed,
            "partitions": ledger.partitions,
            "failure_partition": None if accepted else ledger.role,
            "center_held_fixed": True,
            "fit_design": cfg.fit_design,
            "pilot_coordinate_half_width": cfg.coordinate_half_width,
            "proposal_distribution": "standard_normal_in_refined_factor_coordinates",
            "lineage": json.loads(json.dumps(cfg.lineage, allow_nan=False)),
            **details,
        })
        return PosteriorCurvatureRefinementResult(
            accepted, status, fixed_center, factor,
            refined_factor if accepted else None,
            covariance if accepted else None,
            precision if accepted else None, diagnostics,
        )

    def evaluate(theta, role, seed=None):
        return _evaluate(
            batched_value_and_score_fn, batched_eligibility_fn, theta,
            batch_size=cfg.batch_size, ledger=ledger, role=role, seed=seed,
        )

    status, center_scores = evaluate(fixed_center[tf.newaxis, :], "center")
    if status != "ok":
        return finish(status)
    center_score_z = tf.linalg.matvec(factor, center_scores[0], transpose_a=True)
    if not _finite(center_score_z):
        return finish("nonfinite_transformed_score")
    training = []
    selection = []
    for partition in range(fit_partitions):
        role = "training" if partition < cfg.replicate_count else "selection"
        role = f"{role}_{partition % cfg.replicate_count}"
        seed = _seed(cfg.seed, partition)
        offsets = _draw_offsets(
            rows, dimension, seed, cfg.coordinate_half_width, cfg.fit_design,
        )
        theta = fixed_center[tf.newaxis, :] + tf.matmul(offsets, factor, transpose_b=True)
        status, scores = evaluate(theta, role, seed)
        if status != "ok":
            return finish(status)
        scores_z = tf.matmul(scores, factor)
        if not _finite(scores_z):
            return finish("nonfinite_transformed_score")
        destination = training if partition < cfg.replicate_count else selection
        destination.append((offsets, scores_z))

    ledger.role = "fit"
    fits = []
    details["replicates"] = fits
    for replicate in range(cfg.replicate_count):
        try:
            fit = fit_dense_score_precision_tf(
                center_score_z, *training[replicate],
                selection_offsets=selection[replicate][0],
                selection_scores=selection[replicate][1],
            )
        except tf.errors.InvalidArgumentError:
            return finish("curvature_fit_numerical_failure")
        metrics = {name: float(fit[name].numpy()) for name in (
            "design_condition", "precision_condition", "selection_relative_rmse",
        )}
        valid = (
            _finite(fit["raw_precision"]) and _finite(fit["raw_eigenvalues"])
            and bool(fit["raw_spd"].numpy())
            and int(fit["design_rank"].numpy()) == dimension
            and _within(metrics["design_condition"], cfg.max_design_condition_number)
            and _within(metrics["precision_condition"], cfg.max_precision_condition_number)
            and _within(metrics["selection_relative_rmse"], cfg.selection_relative_rmse_cap)
        )
        fits.append({"precision_z": fit["raw_precision"], **metrics,
                     "raw_spd": bool(fit["raw_spd"].numpy()),
                     "design_rank": int(fit["design_rank"].numpy()), "accepted": valid})
        if not valid:
            return finish("curvature_fit_rejected")

    ledger.role = "replicate_consensus"
    try:
        spread = _precision_spread(tuple(item["precision_z"] for item in fits))
    except tf.errors.InvalidArgumentError:
        return finish("replicate_numerical_failure")
    details["replicate_generalized_eigenvalue_spread"] = spread
    if not _within(spread, cfg.replicate_generalized_eigenvalue_spread_cap):
        return finish("replicate_instability")
    precision = _symmetric(tf.add_n([item["precision_z"] / cfg.replicate_count for item in fits]))
    if not _finite(precision):
        return finish("nonfinite_consensus_precision")

    audit_seed = _seed(cfg.seed, fit_partitions)
    offsets = _draw_offsets(
        rows, dimension, audit_seed, cfg.coordinate_half_width, cfg.fit_design,
    )
    theta = fixed_center[tf.newaxis, :] + tf.matmul(offsets, factor, transpose_b=True)
    status, scores = evaluate(theta, "audit", audit_seed)
    if status != "ok":
        return finish(status)
    audit_rmse = float(_relative_response_rmse(precision, center_score_z, offsets, scores @ factor).numpy())
    details["audit_relative_rmse"] = audit_rmse
    if not _within(audit_rmse, cfg.audit_relative_rmse_cap):
        return finish("audit_rejected")

    ledger.role = "factorization"
    try:
        precision_cholesky = tf.linalg.cholesky(precision)
        if not _finite(precision_cholesky):
            return finish("factorization_failed")
        inverse_action = tf.linalg.triangular_solve(precision_cholesky, tf.transpose(factor))
        covariance = _symmetric(tf.matmul(inverse_action, inverse_action, transpose_a=True))
        if not _finite(covariance):
            return finish("factorization_failed")
        refined_factor = tf.linalg.cholesky(covariance)
        if not _finite(refined_factor) or not bool(tf.reduce_all(tf.linalg.diag_part(refined_factor) > 0.0)):
            return finish("factorization_failed")
    except tf.errors.InvalidArgumentError:
        return finish("factorization_failed")
    reconstruction = refined_factor @ tf.transpose(refined_factor) - covariance
    absolute_error = float(tf.reduce_max(tf.abs(reconstruction)).numpy())
    covariance_scale = tf.reduce_max(tf.abs(covariance))
    relative_error = float((tf.linalg.norm(reconstruction / covariance_scale) /
                            tf.linalg.norm(covariance / covariance_scale)).numpy())
    details.update(factor_reconstruction_abs=absolute_error, factor_reconstruction_rel=relative_error)
    if not math.isfinite(absolute_error) or not math.isfinite(relative_error) or not (
        absolute_error <= cfg.factor_absolute_tolerance or relative_error <= cfg.factor_relative_tolerance
    ):
        return finish("factor_reconstruction_failed")
    normalized_center_score = tf.linalg.matvec(refined_factor, center_scores[0], transpose_a=True)
    center_norm = _stable_norm(normalized_center_score)
    details["center_score_refined_l2"] = center_norm
    if not math.isfinite(center_norm):
        return finish("nonfinite_refined_center_score")

    proposal_seed = _seed(cfg.seed, fit_partitions + 1)
    latent = tf.random.stateless_normal([rows, dimension], proposal_seed, dtype=tf.float64)
    delta = tf.matmul(latent, refined_factor, transpose_b=True)
    theta = fixed_center[tf.newaxis, :] + delta
    status, scores = evaluate(theta, "proposal", proposal_seed)
    if status != "ok":
        return finish(status)
    offsets = tf.transpose(tf.linalg.triangular_solve(factor, tf.transpose(delta)))
    proposal_rmse = float(_relative_response_rmse(precision, center_score_z, offsets, scores @ factor).numpy())
    details["proposal_relative_rmse"] = proposal_rmse
    if not _within(proposal_rmse, cfg.proposal_relative_rmse_cap):
        return finish("refined_proposal_rejected")
    return finish("eligible_for_local_position_factor", covariance, refined_factor, precision)


def _evaluate(value_score_fn, eligibility_fn, rows, *, batch_size, ledger, role, seed):
    """Charge attempts before support/target calls, including failed batches."""
    ledger.role = role
    ledger.partitions.append({"role": role, "seed": None if seed is None else seed.numpy().tolist()})
    collected = []
    for start in range(0, int(rows.shape[0]), batch_size):
        chunk = rows[start:start + batch_size]
        count = int(chunk.shape[0])
        padding = batch_size - count
        chunk = tf.concat((chunk, tf.repeat(chunk[-1:], padding, axis=0)), axis=0)
        ledger.record(physical_rows=batch_size, logical_rows=count, padded_rows=padding)
        if not _finite(chunk):
            return "nonfinite_position", None
        ledger.record(eligibility_batches=1)
        eligible = tf.convert_to_tensor(eligibility_fn(chunk))
        if eligible.dtype != tf.bool:
            raise TypeError(f"{role} eligibility callback must return bool")
        if eligible.shape != (batch_size,):
            raise ValueError(f"{role} eligibility callback must return [batch_size]")
        if not bool(tf.reduce_all(eligible).numpy()):
            return "ineligible_target_row", None
        ledger.record(callback_batches=1, target_rows=batch_size, target_logical_rows=count)
        values, scores = (tf.convert_to_tensor(item) for item in value_score_fn(chunk))
        if values.dtype != tf.float64 or scores.dtype != tf.float64:
            raise TypeError(f"{role} target callback must return float64 values and scores")
        if values.shape != (batch_size,) or scores.shape != chunk.shape:
            raise ValueError(f"{role} target callback returned an invalid batch shape")
        if not _finite(values):
            return "nonfinite_target_value", None
        if not _finite(scores):
            return "nonfinite_target_score", None
        ledger.record(completed_logical_rows=count)
        collected.append(scores[:count])
    return "ok", tf.concat(collected, axis=0)


def _precision_spread(precisions: tuple[tf.Tensor, ...]) -> float:
    """Bound all pairwise generalized eigenvalues and their reciprocals."""
    spread = 1.0
    for first, second in combinations(precisions, 2):
        cholesky = tf.linalg.cholesky(first)
        solved = tf.linalg.triangular_solve(cholesky, second)
        transformed = tf.transpose(tf.linalg.triangular_solve(cholesky, tf.transpose(solved)))
        eigenvalues = tf.linalg.eigvalsh(_symmetric(transformed))
        if not _finite(eigenvalues) or not bool(tf.reduce_all(eigenvalues > 0.0)):
            return float("inf")
        spread = max(spread, float(tf.reduce_max(tf.maximum(eigenvalues, 1.0 / eigenvalues)).numpy()))
    return spread


def _draw_offsets(rows, dimension, seed, width, design):
    if design == "uniform_box":
        return _uniform_offsets(rows, dimension, seed, width)
    return _uniform_ball_offsets(rows, dimension, seed, width)


def _uniform_offsets(rows, dimension, seed, width):
    return tf.random.stateless_uniform([rows, dimension], seed, minval=-width, maxval=width, dtype=tf.float64)


def _uniform_ball_offsets(rows, dimension, seed, radius):
    """Draw z = radius U^(1/D) v, with v uniform on the unit sphere.

    Normalizing a Gaussian gives the isotropic direction; an independently
    seeded U^(1/D) radius gives uniform volume, with E[zz'] = radius^2 I/(D+2).
    Normalizing a uniform box would bias directions. The radius controls probe
    locations only and never multiplies the returned position covariance.
    """
    direction_seed = tf.random.experimental.stateless_fold_in(seed, 0)
    radius_seed = tf.random.experimental.stateless_fold_in(seed, 1)
    direction = tf.random.stateless_normal([rows, dimension], direction_seed, dtype=tf.float64)
    norms = tf.linalg.norm(direction, axis=1, keepdims=True)
    tf.debugging.assert_greater(norms, tf.constant(0.0, tf.float64))
    unit = direction / norms
    radial = tf.random.stateless_uniform(
        [rows, 1], radius_seed, minval=0.0, maxval=1.0, dtype=tf.float64,
    )
    exponent = tf.constant(1.0 / dimension, tf.float64)
    return unit * (radius * tf.pow(radial, exponent))


def _within(value: float, cap: float) -> bool:
    return math.isfinite(value) and value <= cap


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _stable_norm(value: tf.Tensor) -> float:
    scale = tf.reduce_max(tf.abs(value))
    return float((scale * tf.linalg.norm(tf.math.divide_no_nan(value, scale))).numpy())


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


def _symmetric(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * matrix + 0.5 * tf.transpose(matrix)


def _padded_rows(rows: int, batch_size: int) -> int:
    return ((rows + batch_size - 1) // batch_size) * batch_size


def _seed(seed: int, role_index: int) -> tf.Tensor:
    return tf.random.experimental.stateless_fold_in(tf.constant([seed, 0], tf.int32), role_index)


__all__ = [
    "POSTERIOR_CURVATURE_REFINEMENT_NONCLAIMS", "PosteriorCurvatureRefinementConfig",
    "PosteriorCurvatureRefinementResult", "refine_posterior_local_curvature",
]
