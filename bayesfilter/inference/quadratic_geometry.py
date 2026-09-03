"""Low-rank SPD quadratic geometry fitting utilities.

The fitted geometry is a diagnostic initializer for HMC mass construction.  It
does not certify a MAP, posterior correctness, sampler convergence, or default
readiness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf

from bayesfilter.inference._exact_incumbent import (
    ExactCandidate,
    candidates_from_rows,
    select_exact_incumbent,
)


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
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        holdout = float(self.holdout_fraction)
        if not np.isfinite(holdout) or not 0.0 <= holdout < 0.8:
            raise ValueError("holdout_fraction must be finite and satisfy 0 <= value < 0.8")
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
        }


@dataclass(frozen=True)
class LowRankSPDQuadraticGeometryResult:
    """Structured result for a low-rank SPD quadratic geometry attempt."""

    accepted: bool
    status: str
    dimension: int
    rank: int
    center: np.ndarray
    scale: np.ndarray
    precision: np.ndarray | None
    covariance: np.ndarray | None
    q_basis: np.ndarray | None
    linear_term: np.ndarray | None
    intercept: float | None
    lambda0: float | None
    mu: np.ndarray | None
    refined_center: np.ndarray | None
    center_refinement_accepted: bool
    diagnostics: Mapping[str, Any]
    best_evaluated_position: np.ndarray | None = None
    best_evaluated_value: float | None = None
    best_evaluated_score: np.ndarray | None = None
    best_evaluated_source: str | None = None
    best_evaluated_index: int | None = None
    exact_evaluation_count: int = 0
    nonclaims: tuple[str, ...] = LOW_RANK_SPD_QUADRATIC_GEOMETRY_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(self, "rank", int(self.rank))
        for name in ("center", "scale"):
            array = np.asarray(getattr(self, name), dtype=float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("precision", "covariance", "q_basis", "linear_term", "mu"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        if self.refined_center is not None:
            refined = np.asarray(self.refined_center, dtype=float).copy()
            refined.setflags(write=False)
            object.__setattr__(self, "refined_center", refined)
        for name in ("best_evaluated_position", "best_evaluated_score"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).reshape([-1]).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "center_refinement_accepted", bool(self.center_refinement_accepted))
        object.__setattr__(self, "exact_evaluation_count", int(self.exact_evaluation_count))
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

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
            payload["mu"] = tuple(float(value) for value in self.mu)
        if self.precision is not None:
            payload["precision_eigen_summary"] = _eigen_summary(self.precision)
        if self.covariance is not None:
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
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
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

    cfg = LowRankSPDQuadraticGeometryConfig() if config is None else config
    center_np = _vector(center, "center")
    dim = int(center_np.size)
    scale_np = _scale_vector(scale, dim)
    requested_rank = min(int(cfg.rank), dim)
    rank = min(requested_rank, max(dim - 1, 0))
    regression_parameter_count = 1 + dim + 1 + rank
    required_finite_samples = int(cfg.min_samples_per_parameter) * regression_parameter_count
    sample_count = (
        int(cfg.sample_count)
        if cfg.sample_count is not None
        else max(required_finite_samples + 20, 8 * regression_parameter_count)
    )

    base_seed = int(cfg.seed[0]) ^ (int(cfg.seed[1]) << 16)
    rng = np.random.default_rng(base_seed)

    center_value, center_score, center_status = _evaluate_value_score(
        value_and_score_fn,
        center_np,
    )
    if center_status != "finite":
        return _rejected_result(
            status="center_value_or_score_nonfinite",
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=_base_diagnostics(
                cfg=cfg,
                dim=dim,
                rank=rank,
                regression_parameter_count=regression_parameter_count,
                required_finite_samples=required_finite_samples,
                sample_count=sample_count,
                center_value=center_value,
                center_score_norm=None,
            ),
        )
    center_score_z = center_score * scale_np
    center_score_norm = float(np.linalg.norm(center_score_z))

    q_basis, pilot_diagnostics, pilot_candidates = _pilot_q_basis(
        value_and_score_fn,
        batched_value_and_score_fn=batched_value_and_score_fn,
        center=center_np,
        scale=scale_np,
        rank=rank,
        cfg=cfg,
        rng=rng,
        center_value=center_value,
        center_score_z=center_score_z,
        start_index=1,
    )
    z_samples = _sample_trust_ball(
        sample_count,
        dim,
        radius=float(cfg.trust_radius),
        rng=rng,
    )
    theta_samples = center_np[np.newaxis, :] + z_samples * scale_np[np.newaxis, :]
    values, scores = _evaluate_values_scores(
        value_and_score_fn,
        theta_samples,
        batched_value_and_score_fn=batched_value_and_score_fn,
    )
    exact_candidates: list[ExactCandidate] = [
        ExactCandidate(
            position=center_np,
            value=center_value,
            score=center_score,
            evaluation_index=0,
            source_role="center",
        ),
        *pilot_candidates,
        *candidates_from_rows(
            theta_samples,
            values,
            scores,
            start_index=1 + len(pilot_candidates),
            source_role="design",
        ),
    ]
    exact_evaluation_count = len(exact_candidates)
    incumbent = select_exact_incumbent(exact_candidates)
    finite_mask = np.isfinite(values) & np.all(np.isfinite(scores), axis=1)
    finite_sample_count = int(np.sum(finite_mask))
    diagnostics = _base_diagnostics(
        cfg=cfg,
        dim=dim,
        rank=rank,
        regression_parameter_count=regression_parameter_count,
        required_finite_samples=required_finite_samples,
        sample_count=sample_count,
        center_value=center_value,
        center_score_norm=center_score_norm,
    )
    diagnostics.update(
        {
            "finite_sample_count": finite_sample_count,
            "nonfinite_sample_count": int(sample_count - finite_sample_count),
            "pilot": pilot_diagnostics,
            "design_evaluation_route": (
                "batched_value_and_score"
                if batched_value_and_score_fn is not None
                else "scalar_value_and_score_loop"
            ),
        }
    )
    if finite_sample_count < required_finite_samples:
        return _rejected_result(
            status="insufficient_finite_samples",
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=diagnostics,
            incumbent=incumbent,
            exact_evaluation_count=exact_evaluation_count,
        )

    z_finite = z_samples[finite_mask]
    y_finite = values[finite_mask]
    score_finite = scores[finite_mask] * scale_np[np.newaxis, :]
    order = rng.permutation(finite_sample_count)
    holdout_count = int(np.floor(float(cfg.holdout_fraction) * finite_sample_count))
    max_holdout = max(0, finite_sample_count - required_finite_samples)
    holdout_count = min(holdout_count, max_holdout)
    holdout_index = order[:holdout_count]
    train_index = order[holdout_count:]
    z_train = z_finite[train_index]
    y_train = y_finite[train_index]
    score_train = score_finite[train_index]
    z_holdout = z_finite[holdout_index]
    y_holdout = y_finite[holdout_index]

    fit = _fit_constrained_quadratic(
        z_train,
        y_train,
        score_train,
        q_basis=q_basis,
        cfg=cfg,
        dim=dim,
        rank=rank,
        center_score_z=center_score_z,
    )
    if fit["status"] != "usable":
        diagnostics.update({"fit": fit})
        return _rejected_result(
            status=str(fit["status"]),
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=diagnostics,
            incumbent=incumbent,
            exact_evaluation_count=exact_evaluation_count,
        )

    precision = np.asarray(fit["precision"], dtype=float)
    covariance = np.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    linear = np.asarray(fit["linear_term"], dtype=float)
    train_pred = _predict_quadratic(
        z_train,
        intercept=float(fit["intercept"]),
        linear=linear,
        lambda0=float(fit["lambda0"]),
        mu=np.asarray(fit["mu"], dtype=float),
        q_basis=q_basis,
    )
    train_rmse = _rmse(y_train, train_pred)
    if holdout_count > 0:
        holdout_pred = _predict_quadratic(
            z_holdout,
            intercept=float(fit["intercept"]),
            linear=linear,
            lambda0=float(fit["lambda0"]),
            mu=np.asarray(fit["mu"], dtype=float),
            q_basis=q_basis,
        )
        holdout_rmse = _rmse(y_holdout, holdout_pred)
        # A log density is defined only up to an additive constant. Scale the
        # relative gate by local variation, never by the arbitrary level at
        # the fit center, so shifting every target value cannot relax the gate.
        holdout_scale = max(1.0, float(np.std(y_train - center_value)))
        holdout_threshold = max(
            float(cfg.holdout_rmse_abs_tolerance),
            float(cfg.holdout_rmse_rel_tolerance) * holdout_scale,
        )
        holdout_passed = bool(holdout_rmse <= holdout_threshold)
    else:
        holdout_rmse = None
        holdout_threshold = None
        holdout_passed = True

    precision_summary = _eigen_summary(precision)
    covariance_summary = _eigen_summary(covariance)
    center_refinement = _evaluate_center_refinement(
        value_and_score_fn=value_and_score_fn,
        center=center_np,
        scale=scale_np,
        precision=precision,
        linear=linear,
        cfg=cfg,
        center_value=float(center_value),
        center_score_norm=center_score_norm,
    )
    exact_evaluation_count += int(center_refinement.get("exact_evaluation_count", 0))
    refined_score = center_refinement.get("refined_score")
    if (
        center_refinement.get("refined_center") is not None
        and center_refinement.get("refined_log_prob") is not None
        and refined_score is not None
    ):
        exact_candidates.append(
            ExactCandidate(
                position=np.asarray(center_refinement["refined_center"], dtype=float),
                value=float(center_refinement["refined_log_prob"]),
                score=np.asarray(refined_score, dtype=float),
                evaluation_index=exact_evaluation_count - 1,
                source_role="surrogate_replay",
                eligible=bool(center_refinement.get("refined_target_finite", True)),
            )
        )
        incumbent = select_exact_incumbent(exact_candidates)
    replay = _canonical_replay(
        value_and_score_fn,
        incumbent,
        evaluation_index=exact_evaluation_count,
    )
    exact_evaluation_count += int(replay["attempted"])
    diagnostics.update(
        {
            "fit": {
                **fit,
                "precision": None,
            },
            "train_rmse": train_rmse,
            "holdout_count": holdout_count,
            "holdout_rmse": holdout_rmse,
            "holdout_threshold": holdout_threshold,
            "holdout_passed": holdout_passed,
            "precision_eigen_summary": precision_summary,
            "covariance_eigen_summary": covariance_summary,
            "center_refinement": center_refinement,
            "best_evaluated_replay": replay,
            "artifact_hash": _artifact_hash(
                {
                    "config": cfg.payload(),
                    "center": center_np,
                    "scale": scale_np,
                    "precision": precision,
                    "linear": linear,
                    "q_basis": q_basis,
                    "diagnostics": {
                        "finite_sample_count": finite_sample_count,
                        "train_rmse": train_rmse,
                        "holdout_rmse": holdout_rmse,
                    },
                }
            ),
        }
    )
    if not holdout_passed:
        return _rejected_result(
            status="holdout_fit_rejected",
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=diagnostics,
            incumbent=incumbent,
            exact_evaluation_count=exact_evaluation_count,
        )
    if not precision_summary["positive"]:
        return _rejected_result(
            status="precision_not_spd",
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=diagnostics,
            incumbent=incumbent,
            exact_evaluation_count=exact_evaluation_count,
        )
    if precision_summary["condition_number"] > float(cfg.max_condition_number) * (
        1.0 + 1.0e-8
    ):
        return _rejected_result(
            status="precision_condition_above_cap",
            center=center_np,
            scale=scale_np,
            dim=dim,
            rank=rank,
            diagnostics=diagnostics,
            incumbent=incumbent,
            exact_evaluation_count=exact_evaluation_count,
        )

    refined_center = (
        np.asarray(center_refinement["refined_center"], dtype=float)
        if center_refinement["accepted"]
        else None
    )
    return LowRankSPDQuadraticGeometryResult(
        accepted=True,
        status="usable",
        dimension=dim,
        rank=rank,
        center=center_np,
        scale=scale_np,
        precision=precision,
        covariance=covariance,
        q_basis=q_basis,
        linear_term=linear,
        intercept=float(fit["intercept"]),
        lambda0=float(fit["lambda0"]),
        mu=np.asarray(fit["mu"], dtype=float),
        refined_center=refined_center,
        center_refinement_accepted=bool(center_refinement["accepted"]),
        diagnostics=diagnostics,
        best_evaluated_position=None if incumbent is None else incumbent.position,
        best_evaluated_value=None if incumbent is None else incumbent.value,
        best_evaluated_score=None if incumbent is None else incumbent.score,
        best_evaluated_source=None if incumbent is None else incumbent.source_role,
        best_evaluated_index=None if incumbent is None else incumbent.evaluation_index,
        exact_evaluation_count=exact_evaluation_count,
    )


def _fit_constrained_quadratic(
    z_train: np.ndarray,
    y_train: np.ndarray,
    score_train: np.ndarray,
    *,
    q_basis: np.ndarray,
    cfg: LowRankSPDQuadraticGeometryConfig,
    dim: int,
    rank: int,
    center_score_z: np.ndarray,
) -> Mapping[str, Any]:
    z_np = np.asarray(z_train, dtype=float)
    y_np = np.asarray(y_train, dtype=float)
    score_np = np.asarray(score_train, dtype=float)
    q_np = np.asarray(q_basis, dtype=float)
    center_score_np = np.asarray(center_score_z, dtype=float)
    if (
        z_np.ndim != 2
        or z_np.shape[1] != dim
        or score_np.shape != z_np.shape
        or center_score_np.shape != (dim,)
    ):
        return {"status": "fit_shape_mismatch"}

    design, response = _score_curvature_design(
        z_np,
        score_np,
        center_score_np,
        q_basis=q_np,
        dim=dim,
        rank=rank,
    )
    try:
        raw_solution, residuals, design_rank, singular_values = np.linalg.lstsq(
            design,
            response,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return {"status": "score_curvature_lstsq_failed"}
    if not np.all(np.isfinite(raw_solution)):
        return {"status": "fit_nonfinite"}

    raw_lambda = float(raw_solution[0])
    raw_mu = np.asarray(raw_solution[1 : 1 + rank], dtype=float)
    lambda0 = max(float(cfg.eigenvalue_floor), raw_lambda)
    if not np.isfinite(lambda0) or lambda0 <= 0.0:
        return {"status": "fit_nonfinite"}
    mu_upper = (float(cfg.max_condition_number) - 1.0) * lambda0
    mu = np.clip(raw_mu, 0.0, mu_upper)

    precision = lambda0 * np.eye(dim, dtype=float)
    if rank:
        precision = precision + (q_np * mu[np.newaxis, :]) @ q_np.T
    precision = 0.5 * (precision + precision.T)
    linear = center_score_np
    prediction_without_intercept = _predict_quadratic(
        z_np,
        intercept=0.0,
        linear=linear,
        lambda0=lambda0,
        mu=mu,
        q_basis=q_np,
    )
    intercept = float(np.mean(y_np - prediction_without_intercept))
    residual = (
        _predict_quadratic(
            z_np,
            intercept=intercept,
            linear=linear,
            lambda0=lambda0,
            mu=mu,
            q_basis=q_np,
        )
        - y_np
    )
    predicted_score_delta = design @ np.concatenate([[lambda0], mu])
    score_residual = predicted_score_delta - response
    loss_value = float(np.mean(np.square(residual)))
    score_rmse = float(np.sqrt(np.mean(np.square(score_residual))))
    finite = bool(
        np.isfinite(loss_value)
        and np.isfinite(lambda0)
        and np.all(np.isfinite(mu))
        and np.all(np.isfinite(precision))
        and np.all(np.isfinite(linear))
        and np.isfinite(score_rmse)
    )
    if not finite:
        return {"status": "fit_nonfinite"}
    return {
        "status": "usable",
        "fit_method": "score_difference_linear_least_squares_with_value_intercept",
        "optimizer_converged": True,
        "optimizer_failed": False,
        "optimizer_iterations": 0,
        "loss": loss_value,
        "score_rmse": score_rmse,
        "score_design_rank": int(design_rank),
        "score_design_condition_number": _design_condition_number(singular_values),
        "score_lstsq_residual_sum_squares": (
            None if np.size(residuals) == 0 else float(np.sum(residuals))
        ),
        "raw_lambda0": raw_lambda,
        "raw_mu": raw_mu,
        "mu_clipped_count": int(np.sum((raw_mu < 0.0) | (raw_mu > mu_upper))),
        "intercept": intercept,
        "linear_term": linear,
        "lambda0": lambda0,
        "mu": mu,
        "precision": precision,
        "condition_bound": float((lambda0 + float(np.max(mu, initial=0.0))) / lambda0),
    }


def _score_curvature_design(
    z_train: np.ndarray,
    score_train_z: np.ndarray,
    center_score_z: np.ndarray,
    *,
    q_basis: np.ndarray,
    dim: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    response: list[float] = []
    for z_row, score_row in zip(z_train, score_train_z, strict=True):
        score_delta = center_score_z - score_row
        zq = z_row @ q_basis if rank else np.zeros([0], dtype=float)
        for coord in range(dim):
            row = np.empty(1 + rank, dtype=float)
            row[0] = z_row[coord]
            if rank:
                row[1:] = zq * q_basis[coord, :]
            rows.append(row)
            response.append(float(score_delta[coord]))
    return np.vstack(rows), np.asarray(response, dtype=float)


def _design_condition_number(singular_values: np.ndarray) -> float | None:
    values = np.asarray(singular_values, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    return float(np.max(values) / np.min(values))


def _pilot_q_basis(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None,
    center: np.ndarray,
    scale: np.ndarray,
    rank: int,
    cfg: LowRankSPDQuadraticGeometryConfig,
    rng: np.random.Generator,
    center_value: float,
    center_score_z: np.ndarray,
    start_index: int,
) -> tuple[np.ndarray, Mapping[str, Any], tuple[ExactCandidate, ...]]:
    dim = int(center.size)
    if rank == 0:
        return (
            np.zeros([dim, 0], dtype=float),
            {"positive_curvature_count": 0},
            (),
        )
    count = (
        int(cfg.pilot_direction_count)
        if cfg.pilot_direction_count is not None
        else max(4 * dim, 2 * rank + 8)
    )
    directions = rng.normal(size=(count, dim))
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 0.0] / norms[norms > 0.0, np.newaxis]
    sketch = np.zeros([dim, dim], dtype=float)
    curvatures: list[float] = []
    h = float(cfg.pilot_radius)
    if batched_value_and_score_fn is None:
        pilot_rows = []
        pilot_candidates: list[ExactCandidate] = []
        for direction in directions:
            plus = center + (h * direction) * scale
            minus = center - (h * direction) * scale
            value_plus, score_plus, status_plus = _evaluate_value_score(
                value_and_score_fn, plus
            )
            value_minus, score_minus, status_minus = _evaluate_value_score(
                value_and_score_fn, minus
            )
            pilot_rows.append(
                (value_plus, score_plus, status_plus, value_minus, score_minus, status_minus)
            )
            pilot_candidates.extend(
                (
                    ExactCandidate(
                        position=plus,
                        value=value_plus,
                        score=score_plus,
                        evaluation_index=start_index + len(pilot_candidates),
                        source_role="pilot",
                        eligible=status_plus == "finite",
                    ),
                    ExactCandidate(
                        position=minus,
                        value=value_minus,
                        score=score_minus,
                        evaluation_index=start_index + len(pilot_candidates) + 1,
                        source_role="pilot",
                        eligible=status_minus == "finite",
                    ),
                )
            )
        evaluation_route = "scalar_value_and_score_loop"
        evaluation_batch_size = 1
    else:
        plus_points = center[np.newaxis, :] + h * directions * scale[np.newaxis, :]
        minus_points = center[np.newaxis, :] - h * directions * scale[np.newaxis, :]
        pilot_points = np.concatenate((plus_points, minus_points), axis=0)
        pilot_values, pilot_scores = _evaluate_values_scores(
            value_and_score_fn,
            pilot_points,
            batched_value_and_score_fn=batched_value_and_score_fn,
        )
        pilot_candidates = list(
            candidates_from_rows(
                pilot_points,
                pilot_values,
                pilot_scores,
                start_index=start_index,
                source_role="pilot",
            )
        )
        split = int(directions.shape[0])
        pilot_rows = []
        for index in range(split):
            value_plus = float(pilot_values[index])
            score_plus = pilot_scores[index]
            value_minus = float(pilot_values[split + index])
            score_minus = pilot_scores[split + index]
            status_plus = (
                "finite"
                if np.isfinite(value_plus) and np.all(np.isfinite(score_plus))
                else "nonfinite"
            )
            status_minus = (
                "finite"
                if np.isfinite(value_minus) and np.all(np.isfinite(score_minus))
                else "nonfinite"
            )
            pilot_rows.append(
                (value_plus, score_plus, status_plus, value_minus, score_minus, status_minus)
            )
        evaluation_route = "batched_value_and_score"
        evaluation_batch_size = int(pilot_points.shape[0])

    for direction, pilot_row in zip(directions, pilot_rows, strict=True):
        (
            _value_plus,
            score_plus,
            status_plus,
            _value_minus,
            score_minus,
            status_minus,
        ) = pilot_row
        if status_plus != "finite" or status_minus != "finite":
            continue
        score_plus_z = score_plus * scale
        score_minus_z = score_minus * scale
        curvature = float(
            np.dot(score_minus_z - score_plus_z, direction) / (2.0 * h)
        )
        if np.isfinite(curvature) and curvature > 0.0:
            curvatures.append(curvature)
            sketch += curvature * np.outer(direction, direction)
    if curvatures:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (sketch + sketch.T))
        order = np.argsort(eigvals)[::-1]
        q = eigvecs[:, order[:rank]]
    else:
        q = np.eye(dim, rank, dtype=float)
        eigvals = np.zeros(dim, dtype=float)
    q, _ = np.linalg.qr(q)
    q = q[:, :rank]
    return q, {
        "pilot_direction_count": int(count),
        "finite_positive_curvature_count": int(len(curvatures)),
        "curvature_min": None if not curvatures else float(np.min(curvatures)),
        "curvature_max": None if not curvatures else float(np.max(curvatures)),
        "curvature_source": "central_score_difference_directional_curvature",
        "center_score_norm": float(np.linalg.norm(center_score_z)),
        "sketch_eigenvalues": tuple(float(value) for value in np.sort(eigvals)[::-1]),
        "basis_source": (
            "directional_curvature_sketch" if curvatures else "identity_fallback"
        ),
        "evaluation_route": evaluation_route,
        "evaluation_batch_size": evaluation_batch_size,
    }, tuple(pilot_candidates)


def _evaluate_center_refinement(
    *,
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: np.ndarray,
    scale: np.ndarray,
    precision: np.ndarray,
    linear: np.ndarray,
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
        z_star = np.asarray(step["step"], dtype=float)
    else:
        try:
            z_star = np.linalg.solve(precision, linear)
        except np.linalg.LinAlgError:
            return {"accepted": False, "reason": "precision_solve_failed"}
        step = {
            "status": "usable",
            "step": z_star,
            "step_method": "unconstrained_spd_quadratic_solve",
            "boundary_active": False,
            "lagrange_multiplier": 0.0,
            "predicted_improvement": float(
                np.dot(linear, z_star)
                - 0.5 * np.dot(z_star, precision @ z_star)
            ),
        }
    z_norm = float(np.linalg.norm(z_star))
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
    score_norm = float(np.linalg.norm(score * scale))
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
            or (improvement_ratio is not None and np.isfinite(improvement_ratio))
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
        improvement_ratio is None or not np.isfinite(improvement_ratio)
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
        value_and_score_fn, np.asarray(incumbent.position, dtype=float)
    )
    valid = status == "finite"
    matches = bool(
        valid
        and np.isclose(value, incumbent.value, rtol=1.0e-10, atol=1.0e-12)
        and np.allclose(score, incumbent.score, rtol=1.0e-9, atol=1.0e-11)
    )
    return {
        "attempted": True,
        "evaluation_index": int(evaluation_index),
        "valid": valid,
        "matches": matches,
        "value": value,
        "source": incumbent.source_role,
    }


def _solve_spd_quadratic_trust_region(
    precision: np.ndarray,
    linear: np.ndarray,
    *,
    radius: float,
) -> Mapping[str, Any]:
    """Solve ``max b'z - 0.5 z'Kz`` over an Euclidean trust ball.

    For positive-definite ``K``, the constrained solution satisfies
    ``(K + lambda I) z = b``. The unconstrained solution is used when it lies
    inside the ball; otherwise a deterministic bisection solves
    ``||z(lambda)||_2 = radius``.
    """

    matrix = np.asarray(precision, dtype=float)
    vector = np.asarray(linear, dtype=float).reshape([-1])
    radius_value = float(radius)
    if (
        matrix.ndim != 2
        or matrix.shape != (vector.size, vector.size)
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(vector))
        or not np.isfinite(radius_value)
        or radius_value <= 0.0
    ):
        return {"status": "trust_region_invalid_input"}
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError:
        return {"status": "trust_region_eigendecomposition_failed"}
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 0.0:
        return {"status": "trust_region_precision_not_spd"}

    projected = eigenvectors.T @ vector

    def solution(lagrange_multiplier: float) -> np.ndarray:
        return eigenvectors @ (projected / (eigenvalues + lagrange_multiplier))

    unconstrained = solution(0.0)
    unconstrained_norm = float(np.linalg.norm(unconstrained))
    if unconstrained_norm <= radius_value * (1.0 + 1.0e-12):
        step = unconstrained
        lagrange_multiplier = 0.0
        boundary_active = False
        iterations = 0
    else:
        lower = 0.0
        upper = 1.0
        while float(np.linalg.norm(solution(upper))) > radius_value:
            upper *= 2.0
            if not np.isfinite(upper):
                return {"status": "trust_region_bracket_failed"}
        iterations = 0
        for iterations in range(1, 101):
            midpoint = 0.5 * (lower + upper)
            midpoint_norm = float(np.linalg.norm(solution(midpoint)))
            if midpoint_norm > radius_value:
                lower = midpoint
            else:
                upper = midpoint
            if upper - lower <= 1.0e-12 * max(1.0, upper):
                break
        lagrange_multiplier = upper
        step = solution(lagrange_multiplier)
        boundary_active = True

    step_norm = float(np.linalg.norm(step))
    predicted_improvement = float(
        np.dot(vector, step) - 0.5 * np.dot(step, symmetric @ step)
    )
    if (
        not np.all(np.isfinite(step))
        or not np.isfinite(predicted_improvement)
        or step_norm > radius_value * (1.0 + 1.0e-10)
    ):
        return {"status": "trust_region_solution_invalid"}
    return {
        "status": "usable",
        "step": step,
        "step_method": "exact_spd_quadratic_trust_region",
        "boundary_active": boundary_active,
        "lagrange_multiplier": float(lagrange_multiplier),
        "bisection_iterations": int(iterations),
        "unconstrained_step_norm": unconstrained_norm,
        "step_norm": step_norm,
        "predicted_improvement": predicted_improvement,
    }


def _evaluate_values_scores(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    theta_samples: np.ndarray,
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(theta_samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("theta_samples must have shape [batch, dimension]")
    if batched_value_and_score_fn is not None:
        try:
            values, scores = batched_value_and_score_fn(
                tf.constant(samples, dtype=tf.float64)
            )
            values_np = np.asarray(
                tf.convert_to_tensor(values, dtype=tf.float64).numpy(), dtype=float
            )
            scores_np = np.asarray(
                tf.convert_to_tensor(scores, dtype=tf.float64).numpy(), dtype=float
            )
        except Exception:  # noqa: BLE001 - fail-closed diagnostic path.
            return (
                np.full(samples.shape[0], np.nan, dtype=float),
                np.full(samples.shape, np.nan, dtype=float),
            )
        if values_np.shape != (samples.shape[0],) or scores_np.shape != samples.shape:
            return (
                np.full(samples.shape[0], np.nan, dtype=float),
                np.full(samples.shape, np.nan, dtype=float),
            )
        finite = np.isfinite(values_np) & np.all(np.isfinite(scores_np), axis=1)
        return (
            np.where(finite, values_np, np.nan),
            np.where(finite[:, np.newaxis], scores_np, np.nan),
        )

    values = []
    scores = []
    for theta in samples:
        value, score, status = _evaluate_value_score(value_and_score_fn, theta)
        values.append(value if status == "finite" else np.nan)
        scores.append(score if status == "finite" else np.full_like(theta, np.nan, dtype=float))
    return np.asarray(values, dtype=float), np.asarray(scores, dtype=float)


def _evaluate_value_score(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    theta: np.ndarray,
) -> tuple[float, np.ndarray, str]:
    try:
        value, score = value_and_score_fn(tf.constant(theta, dtype=tf.float64))
        value_np = float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())
        score_np = np.asarray(tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1]).numpy(), dtype=float)
    except Exception:  # noqa: BLE001 - diagnostic rejection path.
        return float("nan"), np.full_like(theta, np.nan, dtype=float), "exception"
    if not np.isfinite(value_np) or not np.all(np.isfinite(score_np)):
        return value_np, score_np, "nonfinite"
    return value_np, score_np, "finite"


def _predict_quadratic(
    z: np.ndarray,
    *,
    intercept: float,
    linear: np.ndarray,
    lambda0: float,
    mu: np.ndarray,
    q_basis: np.ndarray,
) -> np.ndarray:
    zq = z @ q_basis
    return (
        float(intercept)
        + z @ linear
        - 0.5 * float(lambda0) * np.sum(np.square(z), axis=1)
        - 0.5 * np.sum(np.square(zq) * mu[np.newaxis, :], axis=1)
    )


def _sample_trust_ball(
    sample_count: int,
    dim: int,
    *,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    directions = rng.normal(size=(int(sample_count), int(dim)))
    norms = np.linalg.norm(directions, axis=1)
    bad = norms <= 0.0
    while np.any(bad):
        directions[bad] = rng.normal(size=(int(np.sum(bad)), int(dim)))
        norms = np.linalg.norm(directions, axis=1)
        bad = norms <= 0.0
    directions = directions / norms[:, np.newaxis]
    radii = float(radius) * rng.random(int(sample_count)) ** (1.0 / float(dim))
    return directions * radii[:, np.newaxis]


def _rejected_result(
    *,
    status: str,
    center: np.ndarray,
    scale: np.ndarray,
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


def _vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape([-1])
    if vector.ndim != 1 or vector.size <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def _scale_vector(value: Any | None, dim: int) -> np.ndarray:
    if value is None:
        scale = np.ones(int(dim), dtype=float)
    else:
        scale = np.asarray(value, dtype=float).reshape([-1])
        if scale.size == 1:
            scale = np.full(int(dim), float(scale[0]), dtype=float)
    if scale.shape != (int(dim),):
        raise ValueError("scale must be scalar or match center dimension")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
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
    square = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (square + square.T)
    eigvals = np.linalg.eigvalsh(symmetric)
    finite = bool(np.all(np.isfinite(eigvals)))
    positive = bool(finite and float(np.min(eigvals)) > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": float(np.min(eigvals)) if finite else float("nan"),
        "max": float(np.max(eigvals)) if finite else float("nan"),
        "condition_number": (
            float(np.max(eigvals) / np.min(eigvals)) if positive else float("inf")
        ),
        "eigenvalues": tuple(float(value) for value in eigvals),
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred)))))


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
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tf.Tensor):
        return _json_ready(value.numpy())
    return value
