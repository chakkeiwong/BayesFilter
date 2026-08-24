"""Truth-blind posterior-local location and scale initialization.

This module composes three existing BayesFilter numerical tools without
promoting any of them into a MAP certificate:

* a smooth bounded L-BFGS chart locates a finite posterior neighborhood;
* fresh constrained SPD quadratic fits nominate exact uphill movements; and
* fixed-center replicate/holdout/audit score fits select terminal curvature.

For numerical coordinate units ``D = diag(scale)`` and standardized offsets
``z``, the terminal score model is ``g_z(c + D z) ~= g_z(c) - K z``.  Hence
``Cov(theta) = D K^{-1} D``.  The returned marginal standard deviations may
initialize trainable transport output scales.  The full covariance remains
diagnostic geometry; this API does not install a fixed affine transport.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import tensorflow as tf

from bayesfilter.inference._exact_incumbent import (
    ExactCandidate,
    select_exact_incumbent,
)
from bayesfilter.inference.fixed_center_curvature import (
    FixedCenterCurvatureResult,
    FixedCenterCurvatureThresholds,
    fit_fixed_center_curvature,
)
from bayesfilter.inference.joint_center import (
    JointCenterLocatorConfig,
    locate_joint_center,
)
from bayesfilter.inference.quadratic_geometry import (
    LowRankSPDQuadraticGeometryConfig,
    fit_low_rank_spd_quadratic_geometry,
)

POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS = (
    "posterior-local location and scale initializer only",
    "truth-blind bounded localization from one supplied start",
    "not a certified local or global MAP",
    "not a posterior covariance correctness claim",
    "not posterior mode-coverage evidence",
    "not a fixed affine transport",
    "not NeuTra training or whitening evidence",
    "not HMC readiness, convergence, or default-readiness evidence",
)


@dataclass(frozen=True)
class PosteriorLocalInitializerConfig:
    """Prospectively fixed budgets and centeredness rules.

    The locator searches ``z = r*tanh(u/r)`` around the supplied initial point.
    At least two independently seeded movement fits are required.  A material
    move on the final allowed fit or curvature cloud returns location-only
    evidence and never attaches stale-centered covariance.
    """

    locator_box_radius: float = 4.0
    locator_config: JointCenterLocatorConfig = field(
        default_factory=lambda: JointCenterLocatorConfig(jit_compile=False)
    )
    min_movement_fits: int = 2
    max_movement_attempts: int = 3
    max_curvature_attempts: int = 2
    objective_improvement_tolerance: float = 1.0e-8
    scaled_center_tolerance: float = 1.0e-8
    curvature_radius: float = 0.05
    training_rows_per_replicate: int | None = None
    selection_rows_per_replicate: int | None = None
    audit_rows: int | None = None
    replicate_count: int = 2
    factor_max: int = 2
    dense_eigenvalue_floor: float = 1.0e-8
    max_condition_number: float = 1.0e8
    shrinkage_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    structured_target_family: str | None = None
    max_exact_evaluations: int = 5000
    seed: int | Sequence[int] = 20260825

    def __post_init__(self) -> None:
        for name in (
            "locator_box_radius",
            "curvature_radius",
            "dense_eigenvalue_floor",
            "max_condition_number",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("objective_improvement_tolerance", "scaled_center_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "min_movement_fits",
            "max_movement_attempts",
            "max_curvature_attempts",
            "replicate_count",
            "max_exact_evaluations",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.min_movement_fits < 2:
            raise ValueError("min_movement_fits must be at least two")
        if self.max_movement_attempts < self.min_movement_fits:
            raise ValueError("max_movement_attempts must cover min_movement_fits")
        if self.replicate_count < 2:
            raise ValueError("replicate_count must be at least two")
        for name in (
            "training_rows_per_replicate",
            "selection_rows_per_replicate",
            "audit_rows",
        ):
            value = getattr(self, name)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive when supplied")
                object.__setattr__(self, name, value)
        factor_max = int(self.factor_max)
        if factor_max not in (1, 2):
            raise ValueError("factor_max must be one or two")
        object.__setattr__(self, "factor_max", factor_max)
        if self.structured_target_family not in (None, "factor_1", "factor_2"):
            raise ValueError("structured_target_family must be factor_1 or factor_2")
        if self.structured_target_family == "factor_2" and factor_max != 2:
            raise ValueError("factor_2 structured target requires factor_max=2")
        weights = tuple(float(value) for value in self.shrinkage_weights)
        if not weights or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in weights
        ):
            raise ValueError("shrinkage_weights must be finite values in [0, 1]")
        object.__setattr__(self, "shrinkage_weights", weights)
        object.__setattr__(self, "seed", _normalize_seed(self.seed))

    def payload(self) -> Mapping[str, Any]:
        return {
            "locator_box_radius": self.locator_box_radius,
            "locator_config": self.locator_config.payload(),
            "min_movement_fits": self.min_movement_fits,
            "max_movement_attempts": self.max_movement_attempts,
            "max_curvature_attempts": self.max_curvature_attempts,
            "objective_improvement_tolerance": self.objective_improvement_tolerance,
            "scaled_center_tolerance": self.scaled_center_tolerance,
            "curvature_radius": self.curvature_radius,
            "training_rows_per_replicate": self.training_rows_per_replicate,
            "selection_rows_per_replicate": self.selection_rows_per_replicate,
            "audit_rows": self.audit_rows,
            "replicate_count": self.replicate_count,
            "factor_max": self.factor_max,
            "dense_eigenvalue_floor": self.dense_eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "shrinkage_weights": self.shrinkage_weights,
            "structured_target_family": self.structured_target_family,
            "max_exact_evaluations": self.max_exact_evaluations,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PosteriorLocalInitializerResult:
    """A centered local initializer or a typed location-only rejection."""

    accepted: bool
    status: str
    dimension: int
    center: np.ndarray
    center_value: float
    center_score: np.ndarray
    scale: np.ndarray
    precision_z: np.ndarray | None
    covariance_z: np.ndarray | None
    precision_theta: np.ndarray | None
    covariance_theta: np.ndarray | None
    marginal_standard_deviations: np.ndarray | None
    initial_output_shift: np.ndarray | None
    initial_output_scale_log: np.ndarray | None
    locator: Mapping[str, Any]
    movement_fits: tuple[Mapping[str, Any], ...]
    curvature: FixedCenterCurvatureResult | None
    exact_evaluation_ledger: tuple[Mapping[str, Any], ...]
    exact_evaluation_count: int
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "dimension", int(self.dimension))
        for name in ("center", "center_score", "scale"):
            array = np.asarray(getattr(self, name), dtype=float).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in (
            "precision_z",
            "covariance_z",
            "precision_theta",
            "covariance_theta",
            "marginal_standard_deviations",
            "initial_output_shift",
            "initial_output_scale_log",
        ):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "locator", _json_ready(dict(self.locator)))
        object.__setattr__(
            self,
            "movement_fits",
            tuple(_json_ready(dict(row)) for row in self.movement_fits),
        )
        object.__setattr__(
            self,
            "exact_evaluation_ledger",
            tuple(_json_ready(dict(row)) for row in self.exact_evaluation_ledger),
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
            "schema": "bayesfilter.posterior_local_initializer.v1",
            "accepted": self.accepted,
            "status": self.status,
            "dimension": self.dimension,
            "center_value": self.center_value,
            "locator": self.locator,
            "movement_fits": self.movement_fits,
            "curvature": None if self.curvature is None else self.curvature.payload(),
            "exact_evaluation_ledger": self.exact_evaluation_ledger,
            "exact_evaluation_count": self.exact_evaluation_count,
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if self.precision_z is not None:
            payload["precision_z_eigen_summary"] = _eigen_summary(self.precision_z)
        if self.covariance_theta is not None:
            payload["covariance_theta_eigen_summary"] = _eigen_summary(
                self.covariance_theta
            )
        if include_arrays:
            payload.update(
                {
                    "center": self.center,
                    "center_score": self.center_score,
                    "scale": self.scale,
                    "precision_z": self.precision_z,
                    "covariance_z": self.covariance_z,
                    "precision_theta": self.precision_theta,
                    "covariance_theta": self.covariance_theta,
                    "marginal_standard_deviations": self.marginal_standard_deviations,
                    "initial_output_shift": self.initial_output_shift,
                    "initial_output_scale_log": self.initial_output_scale_log,
                }
            )
        return _json_ready(payload)


class _EligibilityTrackingEvaluator:
    """Apply one finite/status contract inside scalar and batch TF callbacks."""

    def __init__(
        self,
        scalar_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
        *,
        dimension: int,
        max_rows: int,
        batched_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]] | None,
        eligibility_fn: Callable[[tf.Tensor], tf.Tensor] | None,
        batched_eligibility_fn: Callable[[tf.Tensor], tf.Tensor] | None,
    ) -> None:
        self.scalar_fn = scalar_fn
        self.batched_fn = batched_fn
        self.eligibility_fn = eligibility_fn
        self.batched_eligibility_fn = batched_eligibility_fn
        self.dimension = int(dimension)
        self.max_rows = tf.constant(int(max_rows), tf.int64)
        self.evaluated_rows = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.invalid_rows = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.mismatch_rows = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.budget_exhausted = tf.Variable(False, trainable=False, dtype=tf.bool)

    def scalar(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [self.dimension])

        def denied() -> tuple[tf.Tensor, tf.Tensor]:
            update = self.budget_exhausted.assign(True)
            with tf.control_dependencies([update]):
                return tf.constant(np.nan, tf.float64), tf.fill(
                    [self.dimension], tf.constant(np.nan, tf.float64)
                )

        def evaluate() -> tuple[tf.Tensor, tf.Tensor]:
            update = self.evaluated_rows.assign_add(1)
            with tf.control_dependencies([update]):
                value, score = self.scalar_fn(tf.identity(position))
            value = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
            score = tf.reshape(
                tf.convert_to_tensor(score, tf.float64), [self.dimension]
            )
            finite = tf.logical_and(
                tf.math.is_finite(value), tf.reduce_all(tf.math.is_finite(score))
            )
            eligible = (
                finite
                if self.eligibility_fn is None
                else tf.reshape(
                    tf.convert_to_tensor(self.eligibility_fn(position), tf.bool), []
                )
            )
            mismatch = (
                tf.constant(False)
                if self.eligibility_fn is None
                else tf.not_equal(finite, eligible)
            )
            valid = tf.logical_and(
                finite, tf.logical_and(eligible, tf.logical_not(mismatch))
            )
            updates = (
                self.invalid_rows.assign_add(tf.cast(tf.logical_not(valid), tf.int64)),
                self.mismatch_rows.assign_add(tf.cast(mismatch, tf.int64)),
            )
            with tf.control_dependencies(updates):
                return (
                    tf.where(valid, value, tf.constant(np.nan, tf.float64)),
                    tf.where(
                        valid,
                        score,
                        tf.fill([self.dimension], tf.constant(np.nan, tf.float64)),
                    ),
                )

        return tf.cond(
            self.evaluated_rows.read_value() < self.max_rows,
            evaluate,
            denied,
        )

    def batched(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        if self.batched_fn is None:
            raise ValueError("no batched target callback was supplied")
        positions = tf.ensure_shape(
            tf.convert_to_tensor(theta, tf.float64), [None, self.dimension]
        )
        rows = tf.cast(tf.shape(positions)[0], tf.int64)

        def denied() -> tuple[tf.Tensor, tf.Tensor]:
            update = self.budget_exhausted.assign(True)
            with tf.control_dependencies([update]):
                return (
                    tf.fill([tf.shape(positions)[0]], tf.constant(np.nan, tf.float64)),
                    tf.fill(tf.shape(positions), tf.constant(np.nan, tf.float64)),
                )

        def evaluate() -> tuple[tf.Tensor, tf.Tensor]:
            update = self.evaluated_rows.assign_add(rows)
            with tf.control_dependencies([update]):
                values, scores = self.batched_fn(tf.identity(positions))
            values = tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [None])
            scores = tf.ensure_shape(
                tf.convert_to_tensor(scores, tf.float64), [None, self.dimension]
            )
            finite = tf.logical_and(
                tf.math.is_finite(values),
                tf.reduce_all(tf.math.is_finite(scores), axis=1),
            )
            eligible = (
                finite
                if self.batched_eligibility_fn is None
                else tf.reshape(
                    tf.convert_to_tensor(
                        self.batched_eligibility_fn(positions), tf.bool
                    ),
                    [-1],
                )
            )
            mismatch = (
                tf.zeros_like(finite)
                if self.batched_eligibility_fn is None
                else tf.not_equal(finite, eligible)
            )
            valid = tf.logical_and(
                finite, tf.logical_and(eligible, tf.logical_not(mismatch))
            )
            updates = (
                self.invalid_rows.assign_add(
                    tf.reduce_sum(tf.cast(tf.logical_not(valid), tf.int64))
                ),
                self.mismatch_rows.assign_add(
                    tf.reduce_sum(tf.cast(mismatch, tf.int64))
                ),
            )
            with tf.control_dependencies(updates):
                return (
                    tf.where(
                        valid,
                        values,
                        tf.fill(tf.shape(values), tf.constant(np.nan, tf.float64)),
                    ),
                    tf.where(
                        valid[:, None],
                        scores,
                        tf.fill(tf.shape(scores), tf.constant(np.nan, tf.float64)),
                    ),
                )

        return tf.cond(
            self.evaluated_rows.read_value() + rows <= self.max_rows,
            evaluate,
            denied,
        )

    def diagnostics(self) -> Mapping[str, Any]:
        return {
            "status_callback_supplied": self.eligibility_fn is not None,
            "batched_status_callback_supplied": self.batched_eligibility_fn is not None,
            "evaluated_rows": int(self.evaluated_rows.numpy()),
            "invalid_rows": int(self.invalid_rows.numpy()),
            "mismatch_rows": int(self.mismatch_rows.numpy()),
            "budget_exhausted": bool(self.budget_exhausted.numpy()),
            "finite_status_agreement_required": True,
        }


def initialize_posterior_local_location_scale(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: Any,
    *,
    scale: Any | None = None,
    batched_value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]]
    | None = None,
    eligibility_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    batched_eligibility_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
    config: PosteriorLocalInitializerConfig | None = None,
    movement_config: LowRankSPDQuadraticGeometryConfig | None = None,
    curvature_thresholds: FixedCenterCurvatureThresholds,
) -> PosteriorLocalInitializerResult:
    """Build a centered posterior-local initializer without claiming a MAP.

    Eligibility callbacks return boolean strict-valid status for the same rows
    evaluated by the value/score callbacks.  When present, finite/status
    disagreement is a contract failure.  This deliberately rejects finite
    HMC-rejection sentinels instead of treating their zero score as stationarity.
    """

    cfg = PosteriorLocalInitializerConfig() if config is None else config
    move_cfg = (
        LowRankSPDQuadraticGeometryConfig(
            constrain_center_refinement_to_trust_region=True
        )
        if movement_config is None
        else movement_config
    )
    if not move_cfg.constrain_center_refinement_to_trust_region:
        raise ValueError(
            "movement_config must constrain refinement to the trust region"
        )
    initial = _vector(initial_position, "initial_position")
    dimension = int(initial.size)
    units = (
        np.ones(dimension, dtype=float)
        if scale is None
        else _positive_vector(scale, dimension, "scale")
    )
    if (
        batched_value_and_score_fn is not None
        and eligibility_fn is not None
        and batched_eligibility_fn is None
    ):
        raise ValueError(
            "batched_eligibility_fn is required when batched values and scalar eligibility are supplied"
        )
    if batched_eligibility_fn is not None and batched_value_and_score_fn is None:
        raise ValueError("batched_eligibility_fn requires batched_value_and_score_fn")

    evaluator = _EligibilityTrackingEvaluator(
        value_and_score_fn,
        dimension=dimension,
        max_rows=cfg.max_exact_evaluations,
        batched_fn=batched_value_and_score_fn,
        eligibility_fn=eligibility_fn,
        batched_eligibility_fn=batched_eligibility_fn,
    )
    movement_rows: list[Mapping[str, Any]] = []
    ledger: list[Mapping[str, Any]] = []
    candidates: list[ExactCandidate] = []
    locator_payload: Mapping[str, Any] = {"status": "not_run"}
    curvature: FixedCenterCurvatureResult | None = None

    def record_candidate(
        position: np.ndarray,
        value: float,
        score: np.ndarray,
        source: str,
    ) -> tuple[ExactCandidate, bool]:
        previous = select_exact_incumbent(candidates)
        candidate = ExactCandidate(
            position=position,
            value=value,
            score=score,
            evaluation_index=len(ledger),
            source_role=source,
        )
        candidates.append(candidate)
        selected = select_exact_incumbent(candidates)
        promoted = bool(selected is candidate and selected is not previous)
        ledger.append(
            {
                "ledger_index": len(ledger),
                "source": source,
                "position": position,
                "value": value,
                "score_l2": float(np.linalg.norm(score * units)),
                "promoted": promoted,
            }
        )
        return candidate, promoted

    initial_value, initial_score, initial_ok = _evaluate_numpy(
        evaluator.scalar, initial
    )
    if not initial_ok:
        status = (
            "eligibility_contract_mismatch"
            if evaluator.diagnostics()["mismatch_rows"] > 0
            else "initial_target_invalid"
        )
        return _build_result(
            accepted=False,
            status=status,
            center=initial,
            center_value=initial_value,
            center_score=initial_score,
            scale=units,
            locator=locator_payload,
            movement_fits=movement_rows,
            curvature=None,
            ledger=ledger,
            evaluator=evaluator,
            config=cfg,
            movement_config=move_cfg,
        )
    record_candidate(initial, initial_value, initial_score, "initial_replay")

    radius = float(cfg.locator_box_radius)
    initial_tf = tf.constant(initial, tf.float64)
    units_tf = tf.constant(units, tf.float64)

    def bounded_value_and_score(u: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        u = tf.reshape(tf.convert_to_tensor(u, tf.float64), [dimension])
        tanh = tf.math.tanh(u / radius)
        z = radius * tanh
        theta = initial_tf + units_tf * z
        value, score_theta = evaluator.scalar(theta)
        # This is an optimizer chart, not a change of posterior variables, so
        # only the exact chain rule appears and there is no Jacobian density term.
        score_u = score_theta * units_tf * (1.0 - tf.square(tanh))
        return value, score_u

    locator_result = locate_joint_center(
        bounded_value_and_score,
        np.zeros(dimension),
        scale=np.ones(dimension),
        config=cfg.locator_config,
    )
    locator_payload = {
        **locator_result.payload(),
        "chart": "z = radius * tanh(u / radius)",
        "chart_radius": radius,
        "chart_center_role": "truth_blind_initial_position",
    }
    if locator_result.best_evaluated_position is not None:
        u_best = np.asarray(locator_result.best_evaluated_position, dtype=float)
        z_best = radius * np.tanh(u_best / radius)
        theta_best = initial + units * z_best
        value_best, score_best, valid_best = _evaluate_numpy(
            evaluator.scalar, theta_best
        )
        if valid_best:
            record_candidate(
                theta_best, value_best, score_best, "bounded_locator_best_replay"
            )

    mismatch_status = _tracker_failure_status(evaluator)
    if mismatch_status is not None:
        incumbent = select_exact_incumbent(candidates)
        assert incumbent is not None
        return _build_result(
            accepted=False,
            status=mismatch_status,
            center=incumbent.position,
            center_value=incumbent.value,
            center_score=incumbent.score,
            scale=units,
            locator=locator_payload,
            movement_fits=movement_rows,
            curvature=None,
            ledger=ledger,
            evaluator=evaluator,
            config=cfg,
            movement_config=move_cfg,
        )

    successful_fits = 0
    final_material_move = False
    base_seed = tuple(int(value) for value in move_cfg.seed)
    for attempt in range(cfg.max_movement_attempts):
        incumbent = select_exact_incumbent(candidates)
        assert incumbent is not None
        fit_center = np.asarray(incumbent.position, dtype=float).copy()
        fit_seed = (base_seed[0], base_seed[1] + attempt)
        attempt_cfg = replace(move_cfg, seed=fit_seed)
        geometry = fit_low_rank_spd_quadratic_geometry(
            evaluator.scalar,
            fit_center,
            batched_value_and_score_fn=(
                evaluator.batched if batched_value_and_score_fn is not None else None
            ),
            scale=units,
            config=attempt_cfg,
        )
        previous_value = float(incumbent.value)
        previous_position = fit_center.copy()
        promoted = False
        if geometry.best_evaluated_position is not None:
            nominated = np.asarray(geometry.best_evaluated_position, dtype=float)
            nominated_value, nominated_score, nominated_ok = _evaluate_numpy(
                evaluator.scalar, nominated
            )
            if nominated_ok:
                _candidate, promoted = record_candidate(
                    nominated,
                    nominated_value,
                    nominated_score,
                    f"movement_fit[{attempt}]_{geometry.best_evaluated_source}_replay",
                )
        incumbent = select_exact_incumbent(candidates)
        assert incumbent is not None
        center_moved = bool(
            incumbent.value > previous_value
            and not np.array_equal(incumbent.position, previous_position)
        )
        scaled_move = float(
            np.linalg.norm((incumbent.position - previous_position) / units)
        )
        improvement = float(incumbent.value - previous_value)
        material_move = bool(
            center_moved
            and improvement > cfg.objective_improvement_tolerance
            and scaled_move > cfg.scaled_center_tolerance
        )
        final_material_move = material_move
        if geometry.accepted:
            successful_fits += 1
        movement_rows.append(
            {
                "attempt": attempt,
                "seed": f"{fit_seed[0]}:{fit_seed[1]}",
                "fit_center": fit_center,
                "geometry_status": geometry.status,
                "geometry_accepted": geometry.accepted,
                "geometry_exact_evaluation_count": geometry.exact_evaluation_count,
                "geometry_best_source": geometry.best_evaluated_source,
                "candidate_promoted": promoted,
                "center_moved": center_moved,
                "material_center_move": material_move,
                "scaled_center_move": scaled_move,
                "objective_improvement": improvement,
                "successful_fit_count": successful_fits,
                "covariance_handoff_eligible": False,
            }
        )
        mismatch_status = _tracker_failure_status(evaluator)
        if mismatch_status is not None:
            return _build_result(
                accepted=False,
                status=mismatch_status,
                center=incumbent.position,
                center_value=incumbent.value,
                center_score=incumbent.score,
                scale=units,
                locator=locator_payload,
                movement_fits=movement_rows,
                curvature=None,
                ledger=ledger,
                evaluator=evaluator,
                config=cfg,
                movement_config=move_cfg,
            )
        if successful_fits >= cfg.min_movement_fits and not material_move:
            break

    incumbent = select_exact_incumbent(candidates)
    assert incumbent is not None
    if successful_fits < cfg.min_movement_fits:
        return _build_result(
            accepted=False,
            status="insufficient_successful_movement_fits",
            center=incumbent.position,
            center_value=incumbent.value,
            center_score=incumbent.score,
            scale=units,
            locator=locator_payload,
            movement_fits=movement_rows,
            curvature=None,
            ledger=ledger,
            evaluator=evaluator,
            config=cfg,
            movement_config=move_cfg,
        )
    if final_material_move:
        return _build_result(
            accepted=False,
            status="movement_not_centered_within_attempt_budget",
            center=incumbent.position,
            center_value=incumbent.value,
            center_score=incumbent.score,
            scale=units,
            locator=locator_payload,
            movement_fits=movement_rows,
            curvature=None,
            ledger=ledger,
            evaluator=evaluator,
            config=cfg,
            movement_config=move_cfg,
        )

    train_rows, selection_rows, audit_rows = _cloud_row_counts(cfg, dimension)
    curvature_attempt_records: list[Mapping[str, Any]] = []
    seed = tuple(int(value) for value in cfg.seed)
    for curvature_attempt in range(cfg.max_curvature_attempts):
        incumbent = select_exact_incumbent(candidates)
        assert incumbent is not None
        center = np.asarray(incumbent.position, dtype=float).copy()
        center_value, center_score, center_ok = _evaluate_numpy(
            evaluator.scalar, center
        )
        if not center_ok:
            status = _tracker_failure_status(evaluator) or "curvature_center_invalid"
            return _build_result(
                accepted=False,
                status=status,
                center=center,
                center_value=center_value,
                center_score=center_score,
                scale=units,
                locator=locator_payload,
                movement_fits=movement_rows,
                curvature=None,
                ledger=ledger,
                evaluator=evaluator,
                config=cfg,
                movement_config=move_cfg,
                extra={"curvature_attempts": curvature_attempt_records},
            )
        record_candidate(
            center,
            center_value,
            center_score,
            f"curvature[{curvature_attempt}]_center_replay",
        )
        partition_offsets: list[np.ndarray] = []
        partition_values: list[np.ndarray] = []
        partition_scores: list[np.ndarray] = []
        partition_names: list[str] = []
        partition_seeds: list[tuple[int, int]] = []
        specs = [
            *(
                (f"training[{index}]", train_rows)
                for index in range(cfg.replicate_count)
            ),
            *(
                (f"selection[{index}]", selection_rows)
                for index in range(cfg.replicate_count)
            ),
            ("audit", audit_rows),
        ]
        for partition_index, (name, rows) in enumerate(specs):
            partition_seed = (
                seed[0] + curvature_attempt,
                seed[1] + 1000 * curvature_attempt + partition_index,
            )
            offsets = _sample_ball(
                rows,
                dimension,
                radius=cfg.curvature_radius,
                seed=partition_seed,
            )
            theta = center[None, :] + offsets * units[None, :]
            values, scores, valid = _evaluate_rows(evaluator, theta)
            if not valid:
                status = _tracker_failure_status(evaluator) or "curvature_cloud_invalid"
                return _build_result(
                    accepted=False,
                    status=status,
                    center=center,
                    center_value=center_value,
                    center_score=center_score,
                    scale=units,
                    locator=locator_payload,
                    movement_fits=movement_rows,
                    curvature=None,
                    ledger=ledger,
                    evaluator=evaluator,
                    config=cfg,
                    movement_config=move_cfg,
                    extra={"curvature_attempts": curvature_attempt_records},
                )
            partition_names.append(name)
            partition_seeds.append(partition_seed)
            partition_offsets.append(offsets)
            partition_values.append(values)
            partition_scores.append(scores * units[None, :])
            for row_index in range(rows):
                record_candidate(
                    theta[row_index],
                    float(values[row_index]),
                    scores[row_index],
                    f"curvature[{curvature_attempt}]_{name}_row",
                )

        next_incumbent = select_exact_incumbent(candidates)
        assert next_incumbent is not None
        moved = bool(
            next_incumbent.value > center_value
            and not np.array_equal(next_incumbent.position, center)
        )
        scaled_move = float(np.linalg.norm((next_incumbent.position - center) / units))
        improvement = float(next_incumbent.value - center_value)
        curvature_attempt_records.append(
            {
                "attempt": curvature_attempt,
                "center": center,
                "partition_names": partition_names,
                "partition_seeds": partition_seeds,
                "partition_rows": [int(array.shape[0]) for array in partition_offsets],
                "center_moved": moved,
                "scaled_center_move": scaled_move,
                "objective_improvement": improvement,
                "fit_attempted": not moved,
            }
        )
        if moved:
            if curvature_attempt + 1 == cfg.max_curvature_attempts:
                return _build_result(
                    accepted=False,
                    status="curvature_not_centered_within_attempt_budget",
                    center=next_incumbent.position,
                    center_value=next_incumbent.value,
                    center_score=next_incumbent.score,
                    scale=units,
                    locator=locator_payload,
                    movement_fits=movement_rows,
                    curvature=None,
                    ledger=ledger,
                    evaluator=evaluator,
                    config=cfg,
                    movement_config=move_cfg,
                    extra={"curvature_attempts": curvature_attempt_records},
                )
            continue

        train_offsets = np.stack(partition_offsets[: cfg.replicate_count])
        train_scores_z = np.stack(partition_scores[: cfg.replicate_count])
        selection_start = cfg.replicate_count
        selection_stop = 2 * cfg.replicate_count
        selection_offsets = np.stack(partition_offsets[selection_start:selection_stop])
        selection_scores_z = np.stack(partition_scores[selection_start:selection_stop])
        audit_offsets_z = partition_offsets[-1]
        audit_scores_z = partition_scores[-1]
        curvature = fit_fixed_center_curvature(
            center,
            center_score * units,
            train_offsets,
            train_scores_z,
            selection_offsets,
            selection_scores_z,
            audit_offsets_z,
            audit_scores_z,
            thresholds=curvature_thresholds,
            factor_max=cfg.factor_max,
            dense_eigenvalue_floor=cfg.dense_eigenvalue_floor,
            max_condition_number=cfg.max_condition_number,
            shrinkage_weights=cfg.shrinkage_weights,
            structured_target_family=cfg.structured_target_family,
            lineage={
                "role": "posterior_local_terminal_curvature",
                "curvature_attempt": curvature_attempt,
                "partition_seeds": partition_seeds,
                "fit_center_equals_exact_incumbent": True,
            },
        )
        if (
            not curvature.accepted
            or curvature.selected_precision_z is None
            or curvature.selected_covariance_z is None
        ):
            return _build_result(
                accepted=False,
                status=f"curvature_{curvature.status}",
                center=center,
                center_value=center_value,
                center_score=center_score,
                scale=units,
                locator=locator_payload,
                movement_fits=movement_rows,
                curvature=curvature,
                ledger=ledger,
                evaluator=evaluator,
                config=cfg,
                movement_config=move_cfg,
                extra={"curvature_attempts": curvature_attempt_records},
            )

        precision_z = np.asarray(curvature.selected_precision_z, dtype=float)
        covariance_z = np.asarray(curvature.selected_covariance_z, dtype=float)
        precision_theta = precision_z / (units[:, None] * units[None, :])
        covariance_theta = covariance_z * units[:, None] * units[None, :]
        marginal = np.sqrt(np.diag(covariance_theta))
        if not np.all(np.isfinite(marginal) & (marginal > 0.0)):
            return _build_result(
                accepted=False,
                status="physical_marginal_scale_invalid",
                center=center,
                center_value=center_value,
                center_score=center_score,
                scale=units,
                locator=locator_payload,
                movement_fits=movement_rows,
                curvature=curvature,
                ledger=ledger,
                evaluator=evaluator,
                config=cfg,
                movement_config=move_cfg,
                extra={"curvature_attempts": curvature_attempt_records},
            )
        movement_rows[-1] = {
            **movement_rows[-1],
            "covariance_handoff_eligible": True,
        }
        return _build_result(
            accepted=True,
            status="usable_posterior_local_initializer",
            center=center,
            center_value=center_value,
            center_score=center_score,
            scale=units,
            locator=locator_payload,
            movement_fits=movement_rows,
            curvature=curvature,
            ledger=ledger,
            evaluator=evaluator,
            config=cfg,
            movement_config=move_cfg,
            precision_z=precision_z,
            covariance_z=covariance_z,
            precision_theta=precision_theta,
            covariance_theta=covariance_theta,
            marginal=marginal,
            extra={"curvature_attempts": curvature_attempt_records},
        )

    raise RuntimeError("unreachable posterior-local curvature loop")


def _build_result(
    *,
    accepted: bool,
    status: str,
    center: Any,
    center_value: float,
    center_score: Any,
    scale: np.ndarray,
    locator: Mapping[str, Any],
    movement_fits: Sequence[Mapping[str, Any]],
    curvature: FixedCenterCurvatureResult | None,
    ledger: Sequence[Mapping[str, Any]],
    evaluator: _EligibilityTrackingEvaluator,
    config: PosteriorLocalInitializerConfig,
    movement_config: LowRankSPDQuadraticGeometryConfig,
    precision_z: np.ndarray | None = None,
    covariance_z: np.ndarray | None = None,
    precision_theta: np.ndarray | None = None,
    covariance_theta: np.ndarray | None = None,
    marginal: np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> PosteriorLocalInitializerResult:
    center_array = np.asarray(center, dtype=float).reshape([-1])
    score_array = np.asarray(center_score, dtype=float).reshape([-1])
    evaluator_diagnostics = evaluator.diagnostics()
    return PosteriorLocalInitializerResult(
        accepted=accepted,
        status=status,
        dimension=int(center_array.size),
        center=center_array,
        center_value=float(center_value),
        center_score=score_array,
        scale=scale,
        precision_z=precision_z,
        covariance_z=covariance_z,
        precision_theta=precision_theta,
        covariance_theta=covariance_theta,
        marginal_standard_deviations=marginal,
        initial_output_shift=center_array if accepted else None,
        initial_output_scale_log=(None if marginal is None else np.log(marginal)),
        locator=locator,
        movement_fits=tuple(movement_fits),
        curvature=curvature,
        exact_evaluation_ledger=tuple(ledger),
        exact_evaluation_count=evaluator_diagnostics["evaluated_rows"],
        diagnostics={
            "classification": (
                "posterior_local_initializer_accepted"
                if accepted
                else "posterior_local_initializer_rejected"
            ),
            "eligibility_contract": evaluator_diagnostics,
            "hmc_rejection_policy_permitted": False,
            "base_distribution_changed": False,
            "base_distribution_contract": "IID standard normal remains external to this initializer",
            "full_covariance_installed_as_fixed_transport": False,
            "terminal_stationarity_required": False,
            "global_map_claim": False,
            "config": config.payload(),
            "movement_config": movement_config.payload(),
            **({} if extra is None else dict(extra)),
        },
    )


def _tracker_failure_status(evaluator: _EligibilityTrackingEvaluator) -> str | None:
    diagnostics = evaluator.diagnostics()
    if diagnostics["mismatch_rows"] > 0:
        return "eligibility_contract_mismatch"
    if diagnostics["budget_exhausted"]:
        return "exact_evaluation_budget_exhausted"
    return None


def _evaluate_numpy(
    callback: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    position: np.ndarray,
) -> tuple[float, np.ndarray, bool]:
    try:
        value, score = callback(tf.constant(position, tf.float64))
        value_np = float(tf.convert_to_tensor(value, tf.float64).numpy())
        score_np = np.asarray(
            tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1]).numpy(),
            dtype=float,
        )
    except Exception:  # noqa: BLE001 - typed diagnostic failure is returned.
        return float("nan"), np.full_like(position, np.nan, dtype=float), False
    valid = bool(np.isfinite(value_np) and np.all(np.isfinite(score_np)))
    return value_np, score_np, valid


def _evaluate_rows(
    evaluator: _EligibilityTrackingEvaluator,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    try:
        if evaluator.batched_fn is not None:
            values, scores = evaluator.batched(tf.constant(positions, tf.float64))
            values_np = np.asarray(values.numpy(), dtype=float)
            scores_np = np.asarray(scores.numpy(), dtype=float)
        else:
            rows = [_evaluate_numpy(evaluator.scalar, row) for row in positions]
            values_np = np.asarray([row[0] for row in rows], dtype=float)
            scores_np = np.asarray([row[1] for row in rows], dtype=float)
    except Exception:  # noqa: BLE001 - typed diagnostic failure is returned.
        return (
            np.full(positions.shape[0], np.nan),
            np.full_like(positions, np.nan),
            False,
        )
    valid = bool(
        values_np.shape == (positions.shape[0],)
        and scores_np.shape == positions.shape
        and np.all(np.isfinite(values_np))
        and np.all(np.isfinite(scores_np))
    )
    return values_np, scores_np, valid


def _cloud_row_counts(
    config: PosteriorLocalInitializerConfig, dimension: int
) -> tuple[int, int, int]:
    default_fit_rows = max(2 * dimension, 3 * dimension - 1)
    training = (
        default_fit_rows
        if config.training_rows_per_replicate is None
        else config.training_rows_per_replicate
    )
    selection = (
        default_fit_rows
        if config.selection_rows_per_replicate is None
        else config.selection_rows_per_replicate
    )
    audit = 2 * dimension if config.audit_rows is None else config.audit_rows
    if training + selection < 4 * dimension:
        raise ValueError("training plus selection rows must total at least 4N")
    if audit < 2 * dimension:
        raise ValueError("audit_rows must total at least 2N")
    return int(training), int(selection), int(audit)


def _sample_ball(
    rows: int,
    dimension: int,
    *,
    radius: float,
    seed: tuple[int, int],
) -> np.ndarray:
    combined_seed = int(seed[0]) ^ (int(seed[1]) << 16)
    rng = np.random.default_rng(combined_seed)
    directions = rng.normal(size=(rows, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    while np.any(norms == 0.0):
        zero = np.flatnonzero(norms[:, 0] == 0.0)
        directions[zero] = rng.normal(size=(zero.size, dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= norms
    radial = radius * rng.uniform(0.25, 1.0, size=(rows, 1)) ** (1.0 / dimension)
    return directions * radial


def _normalize_seed(seed: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(seed, (int, np.integer)):
        value = int(seed)
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value, value ^ 0x5A17
    values = tuple(int(value) for value in seed)
    if len(values) != 2 or any(value < 0 for value in values):
        raise ValueError("seed must be a non-negative integer or length-two sequence")
    return values


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array.copy()


def _positive_vector(value: Any, dimension: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,) or not np.all(np.isfinite(array) & (array > 0.0)):
        raise ValueError(f"{name} must be positive finite with shape [{dimension}]")
    return array.copy()


def _eigen_summary(matrix: np.ndarray) -> Mapping[str, Any]:
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    return {
        "minimum": float(np.min(eigenvalues)),
        "maximum": float(np.max(eigenvalues)),
        "condition_number": float(np.max(eigenvalues) / np.min(eigenvalues)),
        "positive": bool(np.all(eigenvalues > 0.0)),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
