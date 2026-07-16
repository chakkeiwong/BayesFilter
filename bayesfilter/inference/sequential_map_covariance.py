"""Sequential exact-target MAP localization and fresh local covariance fitting.

The fitted standardized score model is ``g(z) = g(0) - K z`` for
``theta = center + diag(scale) z``. A covariance is produced only after the
exact scaled score passes the configured stationarity gate. This initializer
does not certify a global MAP, posterior correctness, or HMC readiness.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.mass_matrix import covariance_from_precision
from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
    fit_factor_correlation_score_geometry,
)


SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS = (
    "local exact-stationary MAP candidate only",
    "fresh local quadratic covariance diagnostic only",
    "not a certified global MAP",
    "not posterior correctness evidence",
    "not HMC readiness evidence",
    "not convergence evidence",
    "not default-readiness evidence",
)


@dataclass(frozen=True)
class SequentialMapCovarianceConfig:
    terminal_score_max_abs: float = 0.1
    initial_radius: float = 0.25
    search_sample_count: int = 32
    regression_sample_count: int = 198
    terminal_sample_count: int = 198
    max_attempts: int = 8
    max_exact_evaluations: int = 2048
    locator_max_iterations: int = 50
    locator_max_line_search_iterations: int = 20
    locator_standardized_box_radius: float = 4.0
    locator_gradient_tolerance: float = 1.0e-8
    locator_stopping_condition: str = "converged_all"
    locator_policy: str = "multistart"
    ridge: float = 1.0e-10
    eigenvalue_floor: float = 1.0e-8
    max_condition_number: float = 1.0e8
    minimum_radius: float = 1.0e-4
    maximum_radius: float = 1.0
    acceptance_ratio: float = 0.10
    shrink_threshold: float = 0.25
    expansion_threshold: float = 0.75
    shrink_factor: float = 0.5
    expansion_factor: float = 2.0
    score_reduction_factor: float = 0.95
    holdout_fraction: float = 0.25
    score_holdout_relative_rmse: float = 0.35
    terminal_projection_relative_frobenius_cap: float = 0.25
    max_stalled_attempts: int = 3
    refinement_geometry_policy: str = "full_symmetric"
    dimension_scaled_search: bool = False
    orthogonal_antithetic_search: bool = False
    reuse_search_scores: bool = False
    structured_fresh_sample_multiplier: int = 4
    structured_max_factors: int = 2
    structured_holdout_score_relative_rmse: float = 0.35
    max_terminal_fit_attempts: int | None = None
    require_proposal_score_reduction: bool = True
    stop_on_stalled_attempts: bool = True
    seed: tuple[int, int] = (2026, 715)

    def __post_init__(self) -> None:
        positive_floats = (
            "terminal_score_max_abs",
            "initial_radius",
            "ridge",
            "eigenvalue_floor",
            "locator_standardized_box_radius",
            "locator_gradient_tolerance",
            "minimum_radius",
            "maximum_radius",
            "acceptance_ratio",
            "shrink_threshold",
            "expansion_threshold",
            "shrink_factor",
            "expansion_factor",
            "score_reduction_factor",
            "holdout_fraction",
            "score_holdout_relative_rmse",
            "terminal_projection_relative_frobenius_cap",
            "structured_holdout_score_relative_rmse",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")
            object.__setattr__(self, name, value)
        condition = float(self.max_condition_number)
        if not np.isfinite(condition) or condition <= 1.0:
            raise ValueError("max_condition_number must be finite and greater than 1")
        object.__setattr__(self, "max_condition_number", condition)
        for name in (
            "search_sample_count",
            "regression_sample_count",
            "terminal_sample_count",
            "max_attempts",
            "max_exact_evaluations",
            "locator_max_iterations",
            "locator_max_line_search_iterations",
            "max_stalled_attempts",
            "structured_fresh_sample_multiplier",
            "structured_max_factors",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_terminal_fit_attempts is not None:
            terminal_attempts = int(self.max_terminal_fit_attempts)
            if terminal_attempts <= 0:
                raise ValueError("max_terminal_fit_attempts must be positive")
            object.__setattr__(
                self, "max_terminal_fit_attempts", terminal_attempts
            )
        seed = tuple(int(value) for value in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        object.__setattr__(self, "seed", seed)
        stopping_condition = str(self.locator_stopping_condition)
        if stopping_condition not in {"converged_all", "converged_any"}:
            raise ValueError(
                "locator_stopping_condition must be 'converged_all' or "
                "'converged_any'"
            )
        object.__setattr__(
            self, "locator_stopping_condition", stopping_condition
        )
        locator_policy = str(self.locator_policy)
        if locator_policy not in {"multistart", "center_first"}:
            raise ValueError(
                "locator_policy must be 'multistart' or 'center_first'"
            )
        object.__setattr__(self, "locator_policy", locator_policy)
        geometry_policy = str(self.refinement_geometry_policy)
        if geometry_policy not in {"full_symmetric", "factor_correlation"}:
            raise ValueError(
                "refinement_geometry_policy must be 'full_symmetric' or "
                "'factor_correlation'"
            )
        object.__setattr__(self, "refinement_geometry_policy", geometry_policy)
        if self.structured_max_factors not in (1, 2):
            raise ValueError("structured_max_factors must be one or two")
        for name in (
            "dimension_scaled_search",
            "orthogonal_antithetic_search",
            "reuse_search_scores",
            "require_proposal_score_reduction",
            "stop_on_stalled_attempts",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.minimum_radius > self.initial_radius or self.initial_radius > self.maximum_radius:
            raise ValueError("radii must satisfy minimum <= initial <= maximum")
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must lie strictly between zero and one")
        if not 0.0 < self.shrink_factor < 1.0:
            raise ValueError("shrink_factor must lie strictly between zero and one")
        if self.expansion_factor <= 1.0:
            raise ValueError("expansion_factor must be greater than one")
        if not (
            self.acceptance_ratio <= self.shrink_threshold <= self.expansion_threshold
        ):
            raise ValueError("trust-region ratio thresholds must be ordered")


@dataclass(frozen=True)
class SequentialMapCovarianceResult:
    accepted: bool
    status: str
    map_candidate: np.ndarray | None
    precision: np.ndarray | None
    covariance: np.ndarray | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(self, "nonclaims", tuple(self.nonclaims))

    def payload(self) -> Mapping[str, Any]:
        return _json_ready(
            {
                "schema": "bayesfilter.sequential_map_covariance.v1",
                "accepted": self.accepted,
                "status": self.status,
                "map_candidate": self.map_candidate,
                "precision": self.precision,
                "covariance": self.covariance,
                "diagnostics": self.diagnostics,
                "nonclaims": self.nonclaims,
            }
        )


def estimate_sequential_map_covariance(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_positions: Sequence[Any],
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
    batched_locator_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
    scale: Any | None = None,
    config: SequentialMapCovarianceConfig | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> SequentialMapCovarianceResult:
    """Locate an exact stationary point and fit independent terminal geometry."""

    cfg = SequentialMapCovarianceConfig() if config is None else config
    starts = tf.convert_to_tensor(initial_positions, dtype=tf.float64)
    if starts.shape.rank != 2 or starts.shape[0] is None or starts.shape[1] is None:
        raise ValueError("initial_positions must have static shape [starts, dimension]")
    dimension = int(starts.shape[1])
    scale_tf = (
        tf.ones([dimension], dtype=tf.float64)
        if scale is None
        else tf.reshape(tf.convert_to_tensor(scale, dtype=tf.float64), [-1])
    )
    if scale_tf.shape != (dimension,) or not bool(
        tf.reduce_all(tf.math.is_finite(scale_tf) & (scale_tf > 0.0)).numpy()
    ):
        raise ValueError("scale must be positive finite with one entry per dimension")

    evaluations = 0
    locator_objective_evaluations = 0

    start_count = int(starts.shape[0])
    if cfg.locator_policy == "center_first" and start_count != 1:
        raise ValueError("center_first locator_policy requires exactly one center")
    _emit_progress(
        progress_callback,
        "initializer_started",
        start_count=start_count,
        dimension=dimension,
        locator_policy=cfg.locator_policy,
        locator_stopping_condition=cfg.locator_stopping_condition,
    )
    candidates: list[tf.Tensor] = [starts[index] for index in range(start_count)]
    locator_rows: list[Mapping[str, Any]] = []
    if cfg.locator_policy == "center_first":
        locator_rows.append(
            {
                "finite": True,
                "coordinate_system": "reviewed_exact_center",
                "locator_policy": "center_first",
                "locator_skipped": True,
                "skip_reason": "exact_center_admission",
            }
        )
        _emit_progress(
            progress_callback,
            "locator_skipped_center_first",
            locator_policy=cfg.locator_policy,
            start_count=start_count,
        )
    elif batched_locator_value_and_score_fn is not None and start_count > 1:
        locator_calls = 0

        def batched_standardized_objective(u: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            nonlocal locator_calls
            unconstrained = tf.convert_to_tensor(u, tf.float64)
            box_radius = tf.constant(cfg.locator_standardized_box_radius, tf.float64)
            z = box_radius * tf.math.tanh(unconstrained / box_radius)
            values, scores = batched_locator_value_and_score_fn(starts + scale_tf[None, :] * z)
            values = tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [start_count])
            scores = tf.ensure_shape(tf.convert_to_tensor(scores, tf.float64), [start_count, dimension])
            derivative = 1.0 - tf.square(z / box_radius)
            locator_calls += 1
            _emit_progress(
                progress_callback,
                "locator_objective_completed",
                locator_objective_calls=locator_calls,
                log_posterior=values.numpy(),
                max_abs_scaled_score=tf.reduce_max(
                    tf.abs(scores * scale_tf[None, :]), axis=1
                ).numpy(),
                max_abs_transformed_gradient=tf.reduce_max(
                    tf.abs(scores * scale_tf[None, :] * derivative), axis=1
                ).numpy(),
                standardized_position=z.numpy(),
            )
            return -values, -(scores * scale_tf[None, :] * derivative)

        stopping_condition = (
            tfp.optimizer.converged_all
            if cfg.locator_stopping_condition == "converged_all"
            else tfp.optimizer.converged_any
        )
        optimizer = tfp.optimizer.lbfgs_minimize(
            batched_standardized_objective,
            initial_position=tf.zeros([start_count, dimension], tf.float64),
            tolerance=tf.constant(cfg.locator_gradient_tolerance, tf.float64),
            max_iterations=cfg.locator_max_iterations,
            max_line_search_iterations=cfg.locator_max_line_search_iterations,
            parallel_iterations=1,
            stopping_condition=stopping_condition,
        )
        objective_calls = int(optimizer.num_objective_evaluations.numpy())
        locator_objective_evaluations += objective_calls * start_count
        box_radius = tf.constant(cfg.locator_standardized_box_radius, tf.float64)
        endpoints_z = box_radius * tf.math.tanh(tf.convert_to_tensor(optimizer.position) / box_radius)
        endpoints = starts + scale_tf[None, :] * endpoints_z
        values, scores = batched_locator_value_and_score_fn(endpoints)
        locator_objective_evaluations += start_count
        for index in range(start_count):
            finite = bool(
                (tf.math.is_finite(values[index]) & tf.reduce_all(tf.math.is_finite(scores[index]))).numpy()
            )
            if finite:
                candidates.append(endpoints[index])
            locator_rows.append({
                "finite": finite,
                "converged": bool(optimizer.converged.numpy()[index]),
                "failed": bool(optimizer.failed.numpy()[index]),
                "iterations": int(optimizer.num_iterations.numpy()),
                "objective_calls": objective_calls,
                "conservative_row_evaluations": objective_calls * start_count,
                "coordinate_system": "start_centered_prior_standardized_smooth_box",
                "standardized_box_radius": cfg.locator_standardized_box_radius,
                "gradient_tolerance": cfg.locator_gradient_tolerance,
                "stopping_condition": cfg.locator_stopping_condition,
                "endpoint_standardized_norm": float(tf.linalg.norm(endpoints_z[index]).numpy()),
                "native_batched_locator": True,
            })
        _emit_progress(
            progress_callback,
            "locator_completed",
            locator_objective_calls=objective_calls,
            iterations=int(optimizer.num_iterations.numpy()),
            converged=optimizer.converged.numpy(),
            failed=optimizer.failed.numpy(),
            stopping_condition=cfg.locator_stopping_condition,
        )
    else:
        for start in tuple(candidates):
            def standardized_objective(z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                unconstrained = tf.reshape(tf.convert_to_tensor(z, tf.float64), [-1])
                box_radius = tf.constant(
                    cfg.locator_standardized_box_radius, tf.float64
                )
                z = box_radius * tf.math.tanh(unconstrained / box_radius)
                value, score = _scalar_value_score(
                    value_and_score_fn, start + scale_tf * z, dimension
                )
                transform_derivative = 1.0 - tf.square(z / box_radius)
                return -value, -(scale_tf * score * transform_derivative)

            try:
                optimizer = tfp.optimizer.lbfgs_minimize(
                    standardized_objective,
                    initial_position=tf.zeros([dimension], tf.float64),
                    tolerance=tf.constant(
                        cfg.locator_gradient_tolerance, tf.float64
                    ),
                    max_iterations=cfg.locator_max_iterations,
                    max_line_search_iterations=cfg.locator_max_line_search_iterations,
                    parallel_iterations=1,
                    stopping_condition=(
                        tfp.optimizer.converged_all
                        if cfg.locator_stopping_condition == "converged_all"
                        else tfp.optimizer.converged_any
                    ),
                )
                endpoint_unconstrained = tf.reshape(
                    tf.convert_to_tensor(optimizer.position), [-1]
                )
                box_radius = tf.constant(
                    cfg.locator_standardized_box_radius, tf.float64
                )
                endpoint_z = box_radius * tf.math.tanh(
                    endpoint_unconstrained / box_radius
                )
                endpoint = start + scale_tf * endpoint_z
                objective_evaluations = int(
                    optimizer.num_objective_evaluations.numpy()
                )
                locator_objective_evaluations += objective_evaluations
                value, score = _scalar_value_score(
                    value_and_score_fn, endpoint, dimension
                )
                evaluations += 1
                finite = bool(
                    (
                        tf.math.is_finite(value)
                        & tf.reduce_all(tf.math.is_finite(score))
                    ).numpy()
                )
                if finite:
                    candidates.append(endpoint)
                locator_rows.append(
                    {
                        "finite": finite,
                        "converged": bool(optimizer.converged.numpy()),
                        "failed": bool(optimizer.failed.numpy()),
                        "iterations": int(optimizer.num_iterations.numpy()),
                        "objective_evaluations": objective_evaluations,
                        "coordinate_system": (
                            "start_centered_prior_standardized_smooth_box"
                        ),
                        "standardized_box_radius": (
                            cfg.locator_standardized_box_radius
                        ),
                        "gradient_tolerance": cfg.locator_gradient_tolerance,
                        "stopping_condition": cfg.locator_stopping_condition,
                        "endpoint_standardized_norm": float(
                            tf.linalg.norm(endpoint_z).numpy()
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - starts remain candidates.
                locator_rows.append(
                    {"finite": False, "exception_type": type(exc).__name__}
                )

    finite_candidates: list[tuple[float, tf.Tensor, tf.Tensor]] = []
    for candidate in candidates:
        value, score = _scalar_value_score(value_and_score_fn, candidate, dimension)
        evaluations += 1
        if bool(
            (tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))).numpy()
        ):
            finite_candidates.append((float(value.numpy()), candidate, score))
    if not finite_candidates:
        return _rejected(
            "no_finite_locator_candidate",
            evaluations + locator_objective_evaluations,
            locator_rows,
            cfg,
        )
    evaluations += locator_objective_evaluations
    if evaluations > cfg.max_exact_evaluations:
        return _rejected(
            "maximum_exact_evaluations_after_bounded_locator",
            evaluations,
            locator_rows,
            cfg,
            map_candidate=finite_candidates[0][1].numpy(),
        )
    finite_candidates.sort(key=lambda row: row[0], reverse=True)
    center_value, center, center_score = finite_candidates[0]
    _emit_progress(
        progress_callback,
        "candidate_selected",
        finite_candidate_count=len(finite_candidates),
        selected_log_posterior=center_value,
        selected_max_abs_scaled_score=float(
            tf.reduce_max(tf.abs(scale_tf * center_score)).numpy()
        ),
    )
    radius = cfg.initial_radius
    history: list[Mapping[str, Any]] = []
    stalled = 0
    terminal_fit: Mapping[str, Any] | None = None
    terminal_fit_attempts = 0

    for attempt in range(cfg.max_attempts):
        scaled_score = scale_tf * center_score
        max_score = float(tf.reduce_max(tf.abs(scaled_score)).numpy())
        if max_score <= cfg.terminal_score_max_abs:
            if (
                cfg.max_terminal_fit_attempts is not None
                and terminal_fit_attempts >= cfg.max_terminal_fit_attempts
            ):
                break
            if (
                cfg.refinement_geometry_policy == "factor_correlation"
                and evaluations + cfg.terminal_sample_count
                > cfg.max_exact_evaluations
            ):
                return _rejected(
                    "maximum_exact_evaluations_before_terminal_fit",
                    evaluations,
                    locator_rows,
                    cfg,
                    map_candidate=center.numpy(),
                    extra={"history": history},
                )
            _emit_progress(
                progress_callback,
                "terminal_fit_started",
                attempt=attempt,
                exact_evaluations=evaluations,
                max_abs_scaled_score=max_score,
                radius=radius,
            )
            terminal_fit, evaluations = _fit_score_curvature(
                value_and_score_fn,
                center,
                center_score,
                scale_tf,
                dimension=dimension,
                radius=radius,
                sample_count=cfg.terminal_sample_count,
                seed=(cfg.seed[0], cfg.seed[1] + 100003 + attempt),
                config=cfg,
                evaluations=evaluations,
                batched_value_and_score_fn=batched_value_and_score_fn,
            )
            terminal_fit_attempts += 1
            _emit_progress(
                progress_callback,
                "terminal_fit_completed",
                attempt=attempt,
                exact_evaluations=evaluations,
                fit_status=terminal_fit.get("status"),
                projection_relative_frobenius=terminal_fit.get(
                    "projection_relative_frobenius"
                ),
            )
            if terminal_fit["status"] != "usable":
                radius *= cfg.shrink_factor
                history.append({"attempt": attempt, "action": "terminal_fit_rejected", **terminal_fit})
                if radius < cfg.minimum_radius:
                    break
                continue
            if terminal_fit["projection_relative_frobenius"] > cfg.terminal_projection_relative_frobenius_cap:
                return _rejected(
                    "terminal_projection_exceeds_cap", evaluations, locator_rows, cfg,
                    map_candidate=center.numpy(), extra={"history": history, "terminal_fit": terminal_fit},
                )
            break

        search_sample_count = (
            dimension_scaled_search_count(dimension)
            if cfg.dimension_scaled_search
            else cfg.search_sample_count
        )
        structured_fresh_count = cfg.structured_fresh_sample_multiplier * dimension
        proposal_replay_reserve = (
            cfg.structured_max_factors
            if cfg.refinement_geometry_policy == "factor_correlation"
            else 1
        )
        fit_evaluation_count = (
            structured_fresh_count
            if cfg.refinement_geometry_policy == "factor_correlation"
            else cfg.regression_sample_count
        )
        required = search_sample_count + fit_evaluation_count + proposal_replay_reserve
        if evaluations + required > cfg.max_exact_evaluations:
            return _rejected(
                "maximum_exact_evaluations", evaluations, locator_rows, cfg,
                map_candidate=center.numpy(), extra={"history": history},
            )
        cloud_builder = (
            _orthogonal_antithetic_cloud
            if cfg.orthogonal_antithetic_search
            else _antithetic_cloud
        )
        search_z = cloud_builder(
            search_sample_count, dimension, radius,
            (cfg.seed[0], cfg.seed[1] + 1000 + attempt),
        )
        search_theta = center[None, :] + search_z * scale_tf[None, :]
        search_rows = [(center_value, center, center_score)]
        search_values, search_scores = _evaluate_cloud(
            value_and_score_fn,
            search_theta,
            dimension,
            batched_value_and_score_fn=batched_value_and_score_fn,
        )
        evaluations += search_sample_count
        for row, value, score in zip(
            tf.unstack(search_theta),
            tf.unstack(search_values),
            tf.unstack(search_scores),
            strict=True,
        ):
            if bool((tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(score))).numpy()):
                search_rows.append((float(value.numpy()), row, score))
        search_rows.sort(key=lambda item: item[0], reverse=True)
        selected_value, selected_center, selected_score = search_rows[0]
        recentered = selected_value > center_value
        center_value, center, center_score = selected_value, selected_center, selected_score

        structured_data: Mapping[str, Any] | None = None
        if cfg.refinement_geometry_policy == "factor_correlation":
            structured_data, evaluations = _structured_factor_fit_data(
                value_and_score_fn,
                center,
                center_score,
                scale_tf,
                search_theta=search_theta,
                search_scores=search_scores,
                dimension=dimension,
                radius=radius,
                fresh_sample_count=structured_fresh_count,
                seed=(cfg.seed[0], cfg.seed[1] + 10000 + attempt),
                evaluations=evaluations,
                batched_value_and_score_fn=batched_value_and_score_fn,
                reuse_search_scores=cfg.reuse_search_scores,
            )
            fit = _fit_factor_from_data(structured_data, factor_count=1, config=cfg)
        else:
            fit, evaluations = _fit_score_curvature(
                value_and_score_fn, center, center_score, scale_tf,
                dimension=dimension, radius=radius,
                sample_count=cfg.regression_sample_count,
                seed=(cfg.seed[0], cfg.seed[1] + 10000 + attempt),
                config=cfg, evaluations=evaluations,
                batched_value_and_score_fn=batched_value_and_score_fn,
            )
        row_diag: dict[str, Any] = {
            "attempt": attempt, "radius_before": radius, "recentered": recentered,
            "center_value": center_value, "fit": fit,
        }
        if fit["status"] != "usable" and not (
            cfg.refinement_geometry_policy == "factor_correlation"
            and cfg.structured_max_factors == 2
        ):
            radius *= cfg.shrink_factor
            stalled += 1
            history.append({**row_diag, "action": "fit_rejected_contract"})
            if radius < cfg.minimum_radius or (
                cfg.stop_on_stalled_attempts and stalled >= cfg.max_stalled_attempts
            ):
                break
            continue

        proposal_rows: list[Mapping[str, Any]] = []
        accepted = False
        selected_fit = fit
        actual = float("-inf")
        predicted = float("-inf")
        rho = float("-inf")
        old_norm = float(tf.linalg.norm(scale_tf * center_score).numpy())
        new_norm = old_norm
        step_info: Mapping[str, Any] = {"boundary_active": False}
        factor_candidates = [1]
        if (
            cfg.refinement_geometry_policy == "factor_correlation"
            and cfg.structured_max_factors == 2
        ):
            factor_candidates.append(2)
        for factor_count in factor_candidates:
            if factor_count == 2:
                if accepted:
                    break
                selected_fit = _fit_factor_from_data(
                    structured_data, factor_count=2, config=cfg
                )
                if selected_fit["status"] != "usable":
                    proposal_rows.append(
                        {
                            "factor_count": 2,
                            "fit_status": selected_fit["status"],
                            "proposal_evaluated": False,
                        }
                    )
                    continue
            if selected_fit["status"] != "usable":
                proposal_rows.append(
                    {
                        "factor_count": factor_count,
                        "fit_status": selected_fit["status"],
                        "proposal_evaluated": False,
                    }
                )
                continue
            step_info = _solve_trust_region_tf(
                tf.convert_to_tensor(selected_fit["projected_precision_z"], tf.float64),
                scale_tf * center_score,
                radius,
            )
            step = tf.convert_to_tensor(step_info["step"], tf.float64)
            proposal = center + scale_tf * step
            proposal_value_tf, proposal_score = _scalar_value_score(
                value_and_score_fn, proposal, dimension
            )
            evaluations += 1
            proposal_value = float(proposal_value_tf.numpy())
            actual = proposal_value - center_value
            predicted = float(step_info["predicted_improvement"])
            rho = actual / predicted if predicted > 0.0 else float("-inf")
            old_norm = float(tf.linalg.norm(scale_tf * center_score).numpy())
            new_norm = float(tf.linalg.norm(scale_tf * proposal_score).numpy())
            finite = bool(
                (tf.math.is_finite(proposal_value_tf) & tf.reduce_all(tf.math.is_finite(proposal_score))).numpy()
            )
            score_reduction_passed = bool(
                new_norm <= cfg.score_reduction_factor * old_norm
            )
            accepted = bool(
                finite
                and actual > 0.0
                and predicted > 0.0
                and rho >= cfg.acceptance_ratio
                and (
                    score_reduction_passed
                    or not cfg.require_proposal_score_reduction
                )
            )
            proposal_rows.append(
                {
                    "factor_count": (
                        factor_count
                        if cfg.refinement_geometry_policy == "factor_correlation"
                        else None
                    ),
                    "fit_status": selected_fit["status"],
                    "proposal_evaluated": True,
                    "actual_improvement": actual,
                    "predicted_improvement": predicted,
                    "rho": rho,
                    "score_norm_before": old_norm,
                    "score_norm_after": new_norm,
                    "score_reduction_passed": score_reduction_passed,
                    "accepted": accepted,
                }
            )
            if accepted:
                break
        if accepted:
            center_value, center, center_score = proposal_value, proposal, proposal_score
            stalled = 0
        else:
            stalled += 1
        if rho < cfg.shrink_threshold or not accepted:
            radius *= cfg.shrink_factor
            radius_action = "contract"
        elif rho >= cfg.expansion_threshold and bool(step_info["boundary_active"]):
            radius = min(cfg.maximum_radius, radius * cfg.expansion_factor)
            radius_action = "expand"
        else:
            radius_action = "retain"
        history.append({
            **row_diag,
            "fit": selected_fit,
            "action": "proposal_accepted" if accepted else "proposal_rejected",
            "actual_improvement": actual, "predicted_improvement": predicted,
            "rho": rho, "score_norm_before": old_norm, "score_norm_after": new_norm,
            "boundary_active": bool(step_info["boundary_active"]),
            "radius_action": radius_action, "radius_after": radius,
            "proposal_attempts": proposal_rows,
        })
        _emit_progress(
            progress_callback,
            "refinement_attempt_completed",
            attempt=attempt,
            exact_evaluations=evaluations,
            action="proposal_accepted" if accepted else "proposal_rejected",
            max_abs_scaled_score=float(
                tf.reduce_max(tf.abs(scale_tf * center_score)).numpy()
            ),
            radius=radius,
        )
        if radius < cfg.minimum_radius or (
            cfg.stop_on_stalled_attempts and stalled >= cfg.max_stalled_attempts
        ):
            break

    scaled_score = scale_tf * center_score
    max_score = float(tf.reduce_max(tf.abs(scaled_score)).numpy())
    if (
        max_score <= cfg.terminal_score_max_abs
        and (terminal_fit is None or terminal_fit["status"] != "usable")
        and evaluations + cfg.terminal_sample_count <= cfg.max_exact_evaluations
        and (
            cfg.max_terminal_fit_attempts is None
            or terminal_fit_attempts < cfg.max_terminal_fit_attempts
        )
    ):
        terminal_fit, evaluations = _fit_score_curvature(
            value_and_score_fn,
            center,
            center_score,
            scale_tf,
            dimension=dimension,
            radius=radius,
            sample_count=cfg.terminal_sample_count,
            seed=(cfg.seed[0], cfg.seed[1] + 200003),
            config=cfg,
            evaluations=evaluations,
            batched_value_and_score_fn=batched_value_and_score_fn,
        )
        terminal_fit_attempts += 1
    if max_score > cfg.terminal_score_max_abs or terminal_fit is None or terminal_fit["status"] != "usable":
        return _rejected(
            "sequential_refinement_without_terminal_geometry", evaluations,
            locator_rows, cfg, map_candidate=center.numpy(),
            extra={"terminal_max_abs_scaled_score": max_score, "history": history},
        )
    if terminal_fit["projection_relative_frobenius"] > cfg.terminal_projection_relative_frobenius_cap:
        return _rejected(
            "terminal_projection_exceeds_cap",
            evaluations,
            locator_rows,
            cfg,
            map_candidate=center.numpy(),
            extra={"history": history, "terminal_fit": terminal_fit},
        )

    precision_z = tf.convert_to_tensor(terminal_fit["projected_precision_z"], tf.float64)
    inv_scale = tf.math.reciprocal(scale_tf)
    precision_theta = precision_z * inv_scale[:, None] * inv_scale[None, :]
    mass = covariance_from_precision(
        precision_theta.numpy(),
        source="sequential_fresh_terminal_score_fit",
        jitter=0.0,
        eigenvalue_floor=cfg.eigenvalue_floor,
        max_condition_number=cfg.max_condition_number,
        dense=True,
    )
    _emit_progress(
        progress_callback,
        "initializer_completed",
        accepted=True,
        status="usable",
        exact_evaluations=evaluations,
        terminal_max_abs_scaled_score=max_score,
    )
    return SequentialMapCovarianceResult(
        accepted=True,
        status="usable",
        map_candidate=center.numpy(),
        precision=mass.regularized_precision,
        covariance=mass.covariance,
        diagnostics={
            "exact_evaluations": evaluations,
            "terminal_max_abs_scaled_score": max_score,
            "terminal_fit_fresh": True,
            "terminal_fit_attempts": terminal_fit_attempts,
            "search_seed": list(cfg.seed),
            "terminal_seed": terminal_fit["seed"],
            "precision_coordinate_system": "theta",
            "regression_coordinate_system": "z",
            "locator": locator_rows,
            "history": history,
            "terminal_fit": terminal_fit,
        },
    )


def _antithetic_cloud(
    sample_count: int, dimension: int, radius: float, seed: tuple[int, int]
) -> tf.Tensor:
    half = (sample_count + 1) // 2
    directions = tf.random.stateless_normal(
        [half, dimension], seed=tf.constant(seed, tf.int32), dtype=tf.float64
    )
    directions /= tf.maximum(tf.linalg.norm(directions, axis=1, keepdims=True), 1.0e-15)
    radii = tf.linspace(
        tf.constant(radius / float(half), tf.float64),
        tf.constant(radius, tf.float64),
        half,
    )[:, None]
    cloud = tf.concat([directions * radii, -directions * radii], axis=0)
    return cloud[:sample_count]


def dimension_scaled_search_count(dimension: int) -> int:
    """Return the reviewed even antithetic search count for one dimension."""

    size = int(dimension)
    if size <= 0:
        raise ValueError("dimension must be positive")
    raw = (
        float(size * size)
        if size <= 10
        else 100.0 + float(size - 10) * np.log(float(size - 10))
    )
    return 2 * int(np.ceil(raw / 2.0))


def _orthogonal_antithetic_cloud(
    sample_count: int,
    dimension: int,
    radius: float,
    seed: tuple[int, int],
) -> tf.Tensor:
    """Generate stateless randomized orthogonal frames and antithetic pairs."""

    count = int(sample_count)
    size = int(dimension)
    if count <= 0 or size <= 0:
        raise ValueError("sample_count and dimension must be positive")
    pair_count = (count + 1) // 2
    frame_count = (pair_count + size - 1) // size
    directions = []
    for frame in range(frame_count):
        normal = tf.random.stateless_normal(
            [size, size],
            seed=tf.constant((seed[0], seed[1] + 7919 * frame), tf.int32),
            dtype=tf.float64,
        )
        orthogonal, diagonal = tf.linalg.qr(normal)
        signs = tf.where(
            tf.linalg.diag_part(diagonal) >= 0.0,
            tf.ones([size], tf.float64),
            -tf.ones([size], tf.float64),
        )
        directions.append(tf.transpose(orthogonal * signs[None, :]))
    positive = tf.concat(directions, axis=0)[:pair_count]
    radii = tf.linspace(
        tf.constant(radius / float(pair_count), tf.float64),
        tf.constant(radius, tf.float64),
        pair_count,
    )[:, None]
    cloud = tf.reshape(
        tf.stack((positive * radii, -positive * radii), axis=1),
        [-1, size],
    )
    return cloud[:count]


def _structured_factor_fit_data(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: tf.Tensor,
    center_score: tf.Tensor,
    scale: tf.Tensor,
    *,
    search_theta: tf.Tensor,
    search_scores: tf.Tensor,
    dimension: int,
    radius: float,
    fresh_sample_count: int,
    seed: tuple[int, int],
    evaluations: int,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None,
    reuse_search_scores: bool,
) -> tuple[Mapping[str, Any], int]:
    """Build independent fresh train/holdout frames plus eligible reused rows."""

    if fresh_sample_count < 4 * dimension or fresh_sample_count % 2:
        raise ValueError("structured fresh sample count must be even and at least 4N")
    fresh_train_count = fresh_sample_count // 2
    fresh_holdout_count = fresh_sample_count - fresh_train_count
    train_z = _orthogonal_antithetic_cloud(
        fresh_train_count, dimension, radius, seed
    )
    holdout_z = _orthogonal_antithetic_cloud(
        fresh_holdout_count,
        dimension,
        radius,
        (seed[0], seed[1] + 104729),
    )
    fresh_z = tf.concat((train_z, holdout_z), axis=0)
    _fresh_values, fresh_scores = _evaluate_cloud(
        function,
        center[None, :] + fresh_z * scale[None, :],
        dimension,
        batched_value_and_score_fn=batched_value_and_score_fn,
    )
    evaluations += fresh_sample_count
    fresh_train_scores = fresh_scores[:fresh_train_count]
    holdout_scores = fresh_scores[fresh_train_count:]

    reused_z = tf.zeros([0, dimension], tf.float64)
    reused_scores = tf.zeros([0, dimension], tf.float64)
    if reuse_search_scores:
        translated = (search_theta - center[None, :]) / scale[None, :]
        finite = tf.reduce_all(
            tf.math.is_finite(translated) & tf.math.is_finite(search_scores), axis=1
        )
        nearby = tf.linalg.norm(translated, axis=1) <= radius * (1.0 + 1.0e-12)
        nonzero = tf.linalg.norm(translated, axis=1) > 1.0e-12
        eligible = finite & nearby & nonzero
        reused_z = tf.boolean_mask(translated, eligible)
        reused_scores = tf.boolean_mask(search_scores, eligible)

    reused_count = int(tf.shape(reused_z)[0].numpy())
    training_z = tf.concat((train_z, reused_z), axis=0)
    training_scores = tf.concat((fresh_train_scores, reused_scores), axis=0)
    if reused_count:
        weights = tf.concat(
            (
                tf.fill(
                    [fresh_train_count],
                    tf.constant(0.5 / fresh_train_count, tf.float64),
                ),
                tf.fill(
                    [reused_count],
                    tf.constant(0.5 / reused_count, tf.float64),
                ),
            ),
            axis=0,
        )
    else:
        weights = tf.fill(
            [fresh_train_count],
            tf.constant(1.0 / fresh_train_count, tf.float64),
        )
    return {
        "center_score_z": scale * center_score,
        "training_offsets_z": training_z,
        "training_scores_z": training_scores * scale[None, :],
        "holdout_offsets_z": holdout_z,
        "holdout_scores_z": holdout_scores * scale[None, :],
        "training_weights": weights,
        "fresh_training_count": fresh_train_count,
        "fresh_holdout_count": fresh_holdout_count,
        "reused_training_count": reused_count,
        "unique_fresh_evaluations": fresh_sample_count,
    }, evaluations


def _fit_factor_from_data(
    data: Mapping[str, Any] | None,
    *,
    factor_count: int,
    config: SequentialMapCovarianceConfig,
) -> Mapping[str, Any]:
    if data is None:
        return {"status": "missing_structured_fit_data"}
    result = fit_factor_correlation_score_geometry(
        data["center_score_z"],
        data["training_offsets_z"],
        data["training_scores_z"],
        data["holdout_offsets_z"],
        data["holdout_scores_z"],
        training_weights=data["training_weights"],
        config=FactorCorrelationGeometryConfig(
            factor_count=factor_count,
            max_condition_number=config.max_condition_number,
            holdout_score_relative_rmse=(
                config.structured_holdout_score_relative_rmse
            ),
        ),
    )
    payload = dict(result.payload())
    payload.update(
        {
            "status": result.status,
            "rank": result.parameter_count,
            "projected_precision_z": (
                None if result.precision_z is None else result.precision_z
            ),
            "projection_relative_frobenius": 0.0,
            "fresh_training_count": int(data["fresh_training_count"]),
            "fresh_holdout_count": int(data["fresh_holdout_count"]),
            "reused_training_count": int(data["reused_training_count"]),
        }
    )
    return _json_ready(payload)


def _fit_score_curvature(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    center: tf.Tensor,
    center_score: tf.Tensor,
    scale: tf.Tensor,
    *,
    dimension: int,
    radius: float,
    sample_count: int,
    seed: tuple[int, int],
    config: SequentialMapCovarianceConfig,
    evaluations: int,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None,
) -> tuple[Mapping[str, Any], int]:
    coefficient_count = dimension * (dimension + 1) // 2
    if sample_count * dimension < coefficient_count + dimension:
        return {"status": "insufficient_symmetric_support", "seed": list(seed)}, evaluations
    z = _antithetic_cloud(sample_count, dimension, radius, seed)
    _, scores = _evaluate_cloud(
        function,
        center[None, :] + z * scale[None, :],
        dimension,
        batched_value_and_score_fn=batched_value_and_score_fn,
    )
    evaluations += sample_count
    response = scale[None, :] * center_score[None, :] - scale[None, :] * scores
    design = _symmetric_score_design(z, dimension)
    holdout_count = max(1, int(round(sample_count * config.holdout_fraction)))
    train_count = sample_count - holdout_count
    train_design = tf.reshape(design[:train_count], [-1, coefficient_count])
    train_response = tf.reshape(response[:train_count], [-1, 1])
    singular_values = tf.linalg.svd(train_design, compute_uv=False)
    tolerance = tf.reduce_max(singular_values) * tf.cast(tf.shape(train_design)[0], tf.float64) * tf.experimental.numpy.finfo(tf.float64.as_numpy_dtype).eps
    rank = int(tf.reduce_sum(tf.cast(singular_values > tolerance, tf.int32)).numpy())
    if rank < coefficient_count:
        return {"status": "rank_deficient_symmetric_fit", "rank": rank, "seed": list(seed)}, evaluations
    ridge = tf.sqrt(tf.constant(config.ridge, tf.float64)) * tf.eye(coefficient_count, dtype=tf.float64)
    beta = tf.linalg.lstsq(
        tf.concat([train_design, ridge], axis=0),
        tf.concat([train_response, tf.zeros([coefficient_count, 1], tf.float64)], axis=0),
        fast=False,
    )[:, 0]
    precision = _unpack_symmetric(beta, dimension)
    train_prediction = tf.einsum("nrc,c->nr", design[:train_count], beta)
    holdout_prediction = tf.einsum("nrc,c->nr", design[train_count:], beta)
    train_rmse = float(tf.sqrt(tf.reduce_mean((train_prediction - response[:train_count]) ** 2)).numpy())
    holdout_error = tf.sqrt(tf.reduce_mean((holdout_prediction - response[train_count:]) ** 2))
    holdout_scale = tf.maximum(tf.sqrt(tf.reduce_mean(response[train_count:] ** 2)), 1.0e-15)
    holdout_relative = float((holdout_error / holdout_scale).numpy())
    raw_eigenvalues, eigenvectors = tf.linalg.eigh(precision)
    floor = tf.maximum(
        tf.constant(config.eigenvalue_floor, tf.float64),
        tf.reduce_max(raw_eigenvalues) / config.max_condition_number,
    )
    projected_eigenvalues = tf.maximum(raw_eigenvalues, floor)
    projected = tf.matmul(eigenvectors * projected_eigenvalues[None, :], eigenvectors, transpose_b=True)
    projection_relative = float(
        (tf.linalg.norm(projected - precision) / tf.maximum(tf.linalg.norm(precision), 1.0e-15)).numpy()
    )
    status = "usable" if holdout_relative <= config.score_holdout_relative_rmse else "score_holdout_failed"
    return {
        "status": status, "seed": list(seed), "rank": rank,
        "train_score_rmse": train_rmse,
        "holdout_score_relative_rmse": holdout_relative,
        "raw_eigenvalues": raw_eigenvalues.numpy(),
        "projected_eigenvalues": projected_eigenvalues.numpy(),
        "projection_relative_frobenius": projection_relative,
        "projected_precision_z": projected.numpy(),
    }, evaluations


def _evaluate_cloud(
    scalar_function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    points: tf.Tensor,
    dimension: int,
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None,
) -> tuple[tf.Tensor, tf.Tensor]:
    rows = tf.convert_to_tensor(points, tf.float64)
    row_count = int(rows.shape[0])
    if batched_value_and_score_fn is not None:
        values, scores = batched_value_and_score_fn(rows)
        return (
            tf.ensure_shape(tf.convert_to_tensor(values, tf.float64), [row_count]),
            tf.ensure_shape(
                tf.convert_to_tensor(scores, tf.float64), [row_count, dimension]
            ),
        )
    values = []
    scores = []
    for point in tf.unstack(rows):
        value, score = _scalar_value_score(scalar_function, point, dimension)
        values.append(value)
        scores.append(score)
    return tf.stack(values), tf.stack(scores)


def _symmetric_score_design(z: tf.Tensor, dimension: int) -> tf.Tensor:
    columns = []
    for row in range(dimension):
        for column in range(row, dimension):
            contribution = z[:, column, None] * tf.one_hot(
                row, dimension, dtype=tf.float64
            )[None, :]
            if row != column:
                contribution += z[:, row, None] * tf.one_hot(
                    column, dimension, dtype=tf.float64
                )[None, :]
            columns.append(contribution)
    return tf.stack(columns, axis=2)


def _unpack_symmetric(coefficients: tf.Tensor, dimension: int) -> tf.Tensor:
    matrix = tf.zeros([dimension, dimension], tf.float64)
    index = 0
    for row in range(dimension):
        for column in range(row, dimension):
            value = coefficients[index]
            matrix += tf.scatter_nd([[row, column]], [value], [dimension, dimension])
            if row != column:
                matrix += tf.scatter_nd([[column, row]], [value], [dimension, dimension])
            index += 1
    return matrix


def _solve_trust_region_tf(precision: tf.Tensor, linear: tf.Tensor, radius: float) -> Mapping[str, Any]:
    eigenvalues, eigenvectors = tf.linalg.eigh(precision)
    projected = tf.linalg.matvec(eigenvectors, linear, transpose_a=True)
    unconstrained = tf.linalg.matvec(eigenvectors, projected / eigenvalues)
    unconstrained_norm = float(tf.linalg.norm(unconstrained).numpy())
    boundary = unconstrained_norm > radius
    if boundary:
        lower = tf.constant(0.0, tf.float64)
        upper = tf.constant(1.0, tf.float64)
        for _ in range(80):
            norm = tf.linalg.norm(tf.linalg.matvec(eigenvectors, projected / (eigenvalues + upper)))
            if float(norm.numpy()) <= radius:
                break
            upper *= 2.0
        for _ in range(80):
            middle = 0.5 * (lower + upper)
            norm = tf.linalg.norm(tf.linalg.matvec(eigenvectors, projected / (eigenvalues + middle)))
            if float(norm.numpy()) > radius:
                lower = middle
            else:
                upper = middle
        step = tf.linalg.matvec(eigenvectors, projected / (eigenvalues + upper))
    else:
        step = unconstrained
    predicted = tf.tensordot(linear, step, 1) - 0.5 * tf.tensordot(step, tf.linalg.matvec(precision, step), 1)
    return {"step": step.numpy(), "boundary_active": boundary, "predicted_improvement": float(predicted.numpy())}


def _scalar_value_score(
    function: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    theta: tf.Tensor,
    dimension: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    value, score = function(tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1]))
    value = tf.reshape(tf.convert_to_tensor(value, tf.float64), [])
    score = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
    if score.shape != (dimension,):
        raise ValueError("target score must have one entry per parameter")
    return value, score


def _rejected(
    status: str,
    evaluations: int,
    locator_rows: Sequence[Mapping[str, Any]],
    config: SequentialMapCovarianceConfig,
    *,
    map_candidate: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> SequentialMapCovarianceResult:
    return SequentialMapCovarianceResult(
        accepted=False,
        status=status,
        map_candidate=map_candidate,
        precision=None,
        covariance=None,
        diagnostics={
            "exact_evaluations": evaluations,
            "max_exact_evaluations": config.max_exact_evaluations,
            "locator": locator_rows,
            **({} if extra is None else dict(extra)),
        },
    )


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    stage: str,
    **payload: Any,
) -> None:
    """Publish one host-side semantic event without changing target math."""

    if callback is not None:
        callback(_json_ready({"stage": str(stage), **payload}))


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
