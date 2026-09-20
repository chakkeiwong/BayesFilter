"""Sequential exact-target MAP localization and fresh local covariance fitting.

The fitted standardized score model is ``g(z) = g(0) - K z`` for
``theta = center + diag(scale) z``. A covariance is produced only after the
exact scaled score passes the configured stationarity gate. This initializer
does not certify a global MAP, posterior correctness, or HMC readiness.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
    _factor_result_from_computed,
    fit_factor_correlation_score_geometry,
)
from bayesfilter.inference.factor_correlation_geometry import (
    _rejected as _rejected_factor_fit,
)
from bayesfilter.inference.mass_matrix import covariance_from_precision
from bayesfilter.inference.sequential_attempts_tf import attempts_program
from bayesfilter.inference.sequential_batched_locator_tf import (
    batched_locator_program,
    buffered_batched_locator_program,
)
from bayesfilter.inference.sequential_factor_attempt_tf import (
    empty_second_program,
    second_factor_program,
)
from bayesfilter.inference.sequential_locator_tf import scalar_locator_program
from bayesfilter.inference.sequential_preparation_tf import (
    cloud_program,
    evaluation_program,
    trust_region_program,
)
from bayesfilter.inference.sequential_score_fit_tf import (
    partition_schema,
    score_fit_program,
)
from bayesfilter.inference.sequential_selection_tf import replay_program, search_program
from bayesfilter.inference.sequential_structured_fit_tf import (
    structured_fit_data_program,
)
from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from bayesfilter.ops.host_tensor_io import numeric_tensor

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
    proposal_score_acceptance_policy: str = "fractional"
    record_refinement_movement_diagnostics: bool = False
    pair_disjoint_score_holdout: bool = False

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
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive finite")
            object.__setattr__(self, name, value)
        condition = float(self.max_condition_number)
        if not math.isfinite(condition) or condition <= 1.0:
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
        score_policy = str(self.proposal_score_acceptance_policy)
        if score_policy not in {"fractional", "resolvable_decrease"}:
            raise ValueError(
                "proposal_score_acceptance_policy must be 'fractional' or "
                "'resolvable_decrease'"
            )
        object.__setattr__(
            self, "proposal_score_acceptance_policy", score_policy
        )
        if self.structured_max_factors not in (1, 2):
            raise ValueError("structured_max_factors must be one or two")
        for name in (
            "dimension_scaled_search",
            "orthogonal_antithetic_search",
            "reuse_search_scores",
            "require_proposal_score_reduction",
            "stop_on_stalled_attempts",
            "record_refinement_movement_diagnostics",
            "pair_disjoint_score_holdout",
        ):
            object.__setattr__(self, name, bool(getattr(self, name)))
        if self.pair_disjoint_score_holdout and (
            self.regression_sample_count % 2
            or self.terminal_sample_count % 2
        ):
            raise ValueError(
                "pair-disjoint score holdout requires even regression and "
                "terminal sample counts"
            )
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
    map_candidate: tf.Tensor | None
    precision: tf.Tensor | None
    covariance: tf.Tensor | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = SEQUENTIAL_MAP_COVARIANCE_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = numeric_tensor(value, tf.float64)
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
    """Locate an exact stationary point and fit independent terminal geometry.

    Batched locator objective progress is buffered during XLA execution and
    delivered in order afterwards; callbacks cannot interrupt individual calls.
    Incomplete buffered telemetry fails explicitly before event delivery.
    """

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
    candidates: list[tf.Tensor] = []
    locator_rows: list[Mapping[str, Any]] = []
    selected_candidate = None
    if cfg.locator_policy == "center_first":
        candidates.append(starts[0])
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
        arguments = (value_and_score_fn, batched_locator_value_and_score_fn,
            start_count, dimension, cfg.locator_standardized_box_radius,
            cfg.locator_gradient_tolerance, cfg.locator_max_iterations,
            cfg.locator_max_line_search_iterations, cfg.locator_stopping_condition)
        program = (batched_locator_program(*arguments) if progress_callback is None else
            buffered_batched_locator_program(*arguments, device=starts.device))
        located = program(starts, scale_tf)
        selected_candidate = located["selected"]
        evaluations += int(located["exact_evaluations"])
        locator_objective_evaluations += int(located["objective_evaluations"])
        objective_calls = int(located["objective_calls"])
        iterations = int(located["iterations"])
        if progress_callback is not None:
            # All decisions and observations were computed in the enclosing
            # program. These loops serialize completed reports only.
            trace = located["trace"][:int(located["trace_count"])].numpy().tolist()
            for call, rows in enumerate(trace, start=1):
                values, scaled, transformed, positions = zip(
                    *((row[0], row[1], row[2], row[3:]) for row in rows), strict=True)
                _emit_progress(progress_callback, "locator_objective_completed",
                    locator_objective_calls=call, log_posterior=values,
                    max_abs_scaled_score=scaled, max_abs_transformed_gradient=transformed,
                    standardized_position=positions,
                    delivery_mode="buffered_after_compiled_locator")
        for finite, converged, failed, norm in zip(located["endpoint_finite"].numpy().tolist(),
                located["converged"].numpy().tolist(), located["failed"].numpy().tolist(),
                located["endpoint_standardized_norm"].numpy().tolist(), strict=True):
            locator_rows.append({
                "finite": finite, "converged": converged, "failed": failed,
                "iterations": iterations, "objective_calls": objective_calls,
                "conservative_row_evaluations": objective_calls * start_count,
                "coordinate_system": "start_centered_prior_standardized_smooth_box",
                "standardized_box_radius": cfg.locator_standardized_box_radius,
                "gradient_tolerance": cfg.locator_gradient_tolerance,
                "stopping_condition": cfg.locator_stopping_condition,
                "endpoint_standardized_norm": norm, "native_batched_locator": True,
            })
        _emit_progress(progress_callback, "locator_completed",
            locator_objective_calls=objective_calls, iterations=iterations,
            converged=located["converged"].numpy(), failed=located["failed"].numpy(),
            stopping_condition=cfg.locator_stopping_condition)
    else:
        located = scalar_locator_program(value_and_score_fn, start_count, dimension,
            cfg.locator_standardized_box_radius, cfg.locator_gradient_tolerance,
            cfg.locator_max_iterations, cfg.locator_max_line_search_iterations,
            cfg.locator_stopping_condition)(starts, scale_tf)
        selected_candidate = located["selected"]
        evaluations += int(located["exact_evaluations"])
        locator_objective_evaluations += int(located["objective_evaluations"])
        # Native arrays already contain every numerical decision. This loop
        # reconstructs the existing per-start reporting schema only.
        for finite, diagnostic, norm in zip(located["endpoint_finite"].numpy().tolist(),
                located["optimizer_diagnostics"].numpy().tolist(),
                located["endpoint_standardized_norm"].numpy().tolist(), strict=True):
            converged, failed, iterations, objective_evaluations = diagnostic
            locator_rows.append({
                "finite": finite, "converged": bool(converged), "failed": bool(failed),
                "iterations": iterations, "objective_evaluations": objective_evaluations,
                "coordinate_system": "start_centered_prior_standardized_smooth_box",
                "standardized_box_radius": cfg.locator_standardized_box_radius,
                "gradient_tolerance": cfg.locator_gradient_tolerance,
                "stopping_condition": cfg.locator_stopping_condition,
                "endpoint_standardized_norm": norm,
            })

    if selected_candidate is None:
        selected_candidate = _replay_locator_candidates(value_and_score_fn,
            tf.stack(candidates) if candidates else tf.zeros([0, dimension], tf.float64))
        evaluations += len(candidates)
    finite_candidate_count = int(selected_candidate["finite_count"])
    if finite_candidate_count == 0:
        return _rejected(
            "no_finite_locator_candidate",
            evaluations + locator_objective_evaluations,
            locator_rows,
            cfg,
        )
    # Rank every exact replay before any rejection branch.  A budget failure
    # still exposes the best finite candidate as a diagnostic; returning the
    # first row here would make the reported incumbent depend on start order.
    evaluations += locator_objective_evaluations
    if evaluations > cfg.max_exact_evaluations:
        return _rejected(
            "maximum_exact_evaluations_after_bounded_locator",
            evaluations,
            locator_rows,
            cfg,
            map_candidate=selected_candidate["position"].numpy(),
        )
    center_value = float(selected_candidate["value"])
    center, center_score = selected_candidate["position"], selected_candidate["score"]
    refinement_origin = center if cfg.record_refinement_movement_diagnostics else None
    refinement_movement_initial = (
        {
            "position_z": tf.zeros([dimension], tf.float64),
            "value": center_value,
            "score_norm": float(tf.linalg.norm(scale_tf * center_score).numpy()),
        }
        if cfg.record_refinement_movement_diagnostics
        else None
    )
    _emit_progress(
        progress_callback,
        "candidate_selected",
        finite_candidate_count=finite_candidate_count,
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
        pre_value = center_value
        pre_center = center
        pre_score = center_score
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
                    extra={
                        "history": history,
                        **(
                            {"refinement_movement_initial": refinement_movement_initial}
                            if refinement_movement_initial is not None
                            else {}
                        ),
                    },
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
            terminal_winner = _fit_best_exact_candidate(terminal_fit)
            if terminal_winner is not None and terminal_winner[0] > center_value:
                center_value, center, center_score = terminal_winner
                history.append(
                    {
                        "attempt": attempt,
                        "action": "terminal_fit_cloud_recentered",
                        "center_value": center_value,
                        "fit": terminal_fit,
                    }
                )
                terminal_fit = None
                stalled = 0
                continue
            if terminal_fit["status"] != "usable":
                radius *= cfg.shrink_factor
                history.append({"attempt": attempt, "action": "terminal_fit_rejected", **terminal_fit})
                if radius < cfg.minimum_radius:
                    break
                continue
            if terminal_fit["projection_relative_frobenius"] > cfg.terminal_projection_relative_frobenius_cap:
                return _rejected(
                    "terminal_projection_exceeds_cap", evaluations, locator_rows, cfg,
                    map_candidate=center.numpy(),
                    extra={
                        "history": history,
                        "terminal_fit": terminal_fit,
                        **(
                            {"refinement_movement_initial": refinement_movement_initial}
                            if refinement_movement_initial is not None
                            else {}
                        ),
                    },
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
                map_candidate=center.numpy(),
                extra={
                    "history": history,
                    **(
                        {"refinement_movement_initial": refinement_movement_initial}
                        if refinement_movement_initial is not None
                        else {}
                    ),
                },
            )
        search = _search_exact_candidates(value_and_score_fn, batched_value_and_score_fn,
            center, tf.constant(center_value, tf.float64), center_score, scale_tf,
            tf.constant(radius, tf.float64), tf.constant((cfg.seed[0], cfg.seed[1] + 1000 + attempt), tf.int32),
            sample_count=search_sample_count, orthogonal=cfg.orthogonal_antithetic_search)
        evaluations += search_sample_count
        search_theta, search_scores = search["search_positions"], search["search_scores"]
        recentered = bool(search["recentered"])
        selected_value = float(search["value"])
        selected_center, selected_score = search["position"], search["score"]
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
        fit_winner = _fit_best_exact_candidate(fit)
        if fit_winner is not None and fit_winner[0] > center_value:
            center_value, center, center_score = fit_winner
            history.append(
                {
                    "attempt": attempt,
                    "action": "fit_cloud_recentered",
                    "center_value": center_value,
                    "fit": fit,
                    "radius_action": "retain",
                    "radius_after": radius,
                    "proposal_score_gate": {
                        "policy": cfg.proposal_score_acceptance_policy,
                        "active": cfg.require_proposal_score_reduction,
                        "passed": not cfg.require_proposal_score_reduction,
                        "proposal_evaluated": False,
                    },
                }
            )
            stalled = 0
            continue
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
            movement = None
            if cfg.record_refinement_movement_diagnostics:
                movement = _refinement_movement_payload(
                    refinement_origin=refinement_origin,
                    scale=scale_tf,
                    pre_center=pre_center,
                    pre_value=pre_value,
                    pre_score=pre_score,
                    search_center=center,
                    search_value=center_value,
                    search_score=center_score,
                    evaluated_proposal=None,
                    evaluated_proposal_value=None,
                    evaluated_proposal_score=None,
                    terminal_center=center,
                    terminal_value=center_value,
                    terminal_score=center_score,
                    radius_before=float(row_diag["radius_before"]),
                    radius_after=radius,
                    search_recentered=recentered,
                    proposal_accepted=False,
                )
            history.append(
                {
                    **row_diag,
                    "action": "fit_rejected_contract",
                    **(
                        {"refinement_movement": movement}
                        if movement is not None
                        else {}
                    ),
                }
            )
            if radius < cfg.minimum_radius or (
                cfg.stop_on_stalled_attempts and stalled >= cfg.max_stalled_attempts
            ):
                break
            continue

        first_usable = fit["status"] == "usable"
        first_precision = (tf.convert_to_tensor(fit["projected_precision_z"], tf.float64)
            if first_usable else tf.zeros([dimension, dimension], tf.float64))
        if cfg.refinement_geometry_policy == "factor_correlation" and cfg.structured_max_factors == 2:
            numerical = structured_data["_native_factor_data"]
            second = second_factor_program(dimension, int(numerical["training_offsets_z"].shape[0]),
                int(numerical["holdout_offsets_z"].shape[0]), cfg)
            second_inputs = (numerical["center_score_z"], numerical["training_offsets_z"], numerical["training_scores_z"],
                numerical["holdout_offsets_z"], numerical["holdout_scores_z"], numerical["training_weights"],
                numerical["active_training_rows"])
        else:
            second, second_inputs = empty_second_program(dimension), ()
        proposals = attempts_program(value_and_score_fn, second, dimension, cfg)(
            center, tf.constant(center_value, tf.float64), center_score, scale_tf,
            tf.constant(radius, tf.float64), tf.constant(stalled), tf.constant(first_usable),
            first_precision, second_inputs)
        attempted = int(proposals["attempted"])
        evaluations += int(proposals["evaluations"])
        selected_fit = fit
        if attempted == 2:
            factor_config = FactorCorrelationGeometryConfig(factor_count=2,
                max_condition_number=cfg.max_condition_number,
                holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
            second_result = _factor_result_from_native(proposals["second_fit"]["computed"], factor_config,
                dimension, int(numerical["active_training_rows"]), int(numerical["holdout_offsets_z"].shape[0]))
            selected_fit = _factor_fit_payload(second_result, structured_data)
        last = proposals["last"]
        accepted = bool(proposals["accepted"])
        evaluated = bool(proposals["last_evaluated"])
        evaluated_proposal = last["position"] if evaluated else None
        evaluated_proposal_value = float(last["value"]) if evaluated else None
        evaluated_proposal_score = last["score"] if evaluated else None
        actual, predicted, rho = float(last["actual"]), float(last["predicted"]), float(last["rho"])
        old_norm, new_norm = float(last["old_norm"]), float(last["new_norm"])
        step_info = {"boundary_active": bool(last["boundary"])}
        proposal_score_gate = (_proposal_gate_payload(last, cfg) if evaluated else {
            "policy": cfg.proposal_score_acceptance_policy, "active": cfg.require_proposal_score_reduction,
            "passed": not cfg.require_proposal_score_reduction})
        # Restore reporting rows only after all numerical attempts, decisions,
        # incumbent selection and radius/stall updates have completed in XLA.
        # Transfer each completed column once; indexing these host records must
        # not dispatch one eager TensorFlow slicing kernel for every field.
        history_values = {key: values.numpy().tolist() for key, values in proposals["histories"].items()}
        evaluated_rows = proposals["evaluated"].numpy().tolist()
        proposal_rows = []
        for index in range(attempted):
            row_fit = fit if index == 0 else selected_fit
            if not evaluated_rows[index]:
                proposal_rows.append({"factor_count": index + 1,
                    "fit_status": row_fit["status"], "proposal_evaluated": False})
                continue
            row = {key: values[index] for key, values in history_values.items()}
            proposal_rows.append({"factor_count": index + 1 if cfg.refinement_geometry_policy == "factor_correlation" else None,
                "fit_status": row_fit["status"], "proposal_evaluated": True,
                "actual_improvement": float(row["actual"]), "predicted_improvement": float(row["predicted"]),
                "rho": float(row["rho"]), "score_norm_before": float(row["old_norm"]),
                "score_norm_after": float(row["new_norm"]), "score_reduction_passed": bool(row["score_passed"]),
                "proposal_score_gate": _proposal_gate_payload(row, cfg), "accepted": bool(row["accepted"])})
        incumbent_promoted_without_model_acceptance = bool(proposals["promoted_without_acceptance"])
        center_value, center, center_score = float(proposals["center_value"]), proposals["center"], proposals["center_score"]
        stalled = int(proposals["stalled"])
        radius = float(proposals["radius_after"])
        radius_action = ("contract", "expand", "retain")[int(proposals["radius_action"])]
        history.append({
            **row_diag,
            "fit": selected_fit,
            "action": "proposal_accepted" if accepted else "proposal_rejected",
            "actual_improvement": actual, "predicted_improvement": predicted,
            "rho": rho, "score_norm_before": old_norm, "score_norm_after": new_norm,
            "proposal_score_gate": proposal_score_gate,
            "boundary_active": bool(step_info["boundary_active"]),
            "radius_action": radius_action, "radius_after": radius,
            "proposal_attempts": proposal_rows,
            "exact_incumbent_promoted_without_model_acceptance": (
                incumbent_promoted_without_model_acceptance
            ),
            **(
                {
                    "refinement_movement": _refinement_movement_payload(
                        refinement_origin=refinement_origin,
                        scale=scale_tf,
                        pre_center=pre_center,
                        pre_value=pre_value,
                        pre_score=pre_score,
                        search_center=selected_center,
                        search_value=selected_value,
                        search_score=selected_score,
                        evaluated_proposal=evaluated_proposal,
                        evaluated_proposal_value=evaluated_proposal_value,
                        evaluated_proposal_score=evaluated_proposal_score,
                        terminal_center=center,
                        terminal_value=center_value,
                        terminal_score=center_score,
                        radius_before=float(row_diag["radius_before"]),
                        radius_after=radius,
                        search_recentered=recentered,
                        proposal_accepted=accepted,
                    )
                }
                if cfg.record_refinement_movement_diagnostics
                else {}
            ),
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
            extra={
                "terminal_max_abs_scaled_score": max_score,
                "history": history,
                **(
                    {"refinement_movement_initial": refinement_movement_initial}
                    if refinement_movement_initial is not None
                    else {}
                ),
            },
        )
    if terminal_fit["projection_relative_frobenius"] > cfg.terminal_projection_relative_frobenius_cap:
        return _rejected(
            "terminal_projection_exceeds_cap",
            evaluations,
            locator_rows,
            cfg,
            map_candidate=center.numpy(),
            extra={
                "history": history,
                "terminal_fit": terminal_fit,
                **(
                    {"refinement_movement_initial": refinement_movement_initial}
                    if refinement_movement_initial is not None
                    else {}
                ),
            },
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
            "proposal_score_acceptance_policy": cfg.proposal_score_acceptance_policy,
            "proposal_score_gate_active": cfg.require_proposal_score_reduction,
            **(
                {"refinement_movement_initial": refinement_movement_initial}
                if refinement_movement_initial is not None
                else {}
            ),
        },
    )


def _refinement_movement_payload(
    *,
    refinement_origin: tf.Tensor,
    scale: tf.Tensor,
    pre_center: tf.Tensor,
    pre_value: float,
    pre_score: tf.Tensor,
    search_center: tf.Tensor,
    search_value: float,
    search_score: tf.Tensor,
    evaluated_proposal: tf.Tensor | None,
    evaluated_proposal_value: float | None,
    evaluated_proposal_score: tf.Tensor | None,
    terminal_center: tf.Tensor,
    terminal_value: float,
    terminal_score: tf.Tensor,
    radius_before: float,
    radius_after: float,
    search_recentered: bool,
    proposal_accepted: bool,
) -> Mapping[str, Any]:
    """Record exact refinement states without changing the transition logic."""

    def position_z(value: tf.Tensor) -> tf.Tensor:
        return (value - refinement_origin) / scale

    def score_norm(value: tf.Tensor) -> float:
        return float(tf.linalg.norm(scale * value).numpy())

    return {
        "pre_position_z": position_z(pre_center),
        "search_position_z": position_z(search_center),
        "evaluated_proposal_position_z": (
            None if evaluated_proposal is None else position_z(evaluated_proposal)
        ),
        "terminal_position_z": position_z(terminal_center),
        "pre_value": float(pre_value),
        "search_value": float(search_value),
        "evaluated_proposal_value": (
            None
            if evaluated_proposal_value is None
            else float(evaluated_proposal_value)
        ),
        "terminal_value": float(terminal_value),
        "pre_score_norm": score_norm(pre_score),
        "search_score_norm": score_norm(search_score),
        "evaluated_proposal_score_norm": (
            None
            if evaluated_proposal_score is None
            else score_norm(evaluated_proposal_score)
        ),
        "terminal_score_norm": score_norm(terminal_score),
        "radius_before": float(radius_before),
        "radius_after": float(radius_after),
        "search_recentered": bool(search_recentered),
        "proposal_accepted": bool(proposal_accepted),
    }


def _antithetic_cloud(
    sample_count: int, dimension: int, radius: float, seed: tuple[int, int]
) -> tf.Tensor:
    return cloud_program(sample_count, dimension, False)(
        tf.convert_to_tensor(radius, tf.float64), tf.convert_to_tensor(seed, tf.int32))


def dimension_scaled_search_count(dimension: int) -> int:
    """Return the reviewed even antithetic search count for one dimension."""

    size = int(dimension)
    if size <= 0:
        raise ValueError("dimension must be positive")
    raw = (
        float(size * size)
        if size <= 10
        else 100.0 + float(size - 10) * math.log(float(size - 10))
    )
    return 2 * math.ceil(raw / 2.0)


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
    return cloud_program(count, size, True)(
        tf.convert_to_tensor(radius, tf.float64), tf.convert_to_tensor(seed, tf.int32))


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

    result = structured_data_program(function, batched_value_and_score_fn, dimension,
        fresh_sample_count, int(search_theta.shape[0]), reuse_search_scores)(
        center, center_score, scale, tf.convert_to_tensor(radius, tf.float64),
        tf.convert_to_tensor(seed, tf.int32), search_theta, search_scores)
    has_winner = int(result["best_index"]) >= 0
    # Preserve the historical compact record at this host boundary. The
    # optimizer consumes the fixed-capacity tensors directly, never these views.
    active_rows = int(result["active_training_rows"])
    return {
        "center_score_z": result["center_score_z"],
        "training_offsets_z": result["training_offsets_z"][:active_rows],
        "training_scores_z": result["training_scores_z"][:active_rows],
        "holdout_offsets_z": result["holdout_offsets_z"],
        "holdout_scores_z": result["holdout_scores_z"],
        "training_weights": result["training_weights"][:active_rows],
        "_native_factor_data": result,
        "fresh_training_count": int(result["fresh_training_count"]),
        "fresh_holdout_count": int(result["fresh_holdout_count"]),
        "reused_training_count": int(result["reused_training_count"]),
        "unique_fresh_evaluations": fresh_sample_count,
        "best_exact_value": float(result["best_value"]) if has_winner else None,
        "best_exact_position": result["best_position"].numpy().tolist() if has_winner else None,
        "best_exact_score": result["best_score"].numpy().tolist() if has_winner else None,
        "best_exact_source": "structured_fit_cloud" if has_winner else None,
    }, evaluations + fresh_sample_count


def _fit_factor_from_data(
    data: Mapping[str, Any] | None,
    *,
    factor_count: int,
    config: SequentialMapCovarianceConfig,
) -> Mapping[str, Any]:
    if data is None:
        return {"status": "missing_structured_fit_data"}
    factor_config = FactorCorrelationGeometryConfig(factor_count=factor_count,
        max_condition_number=config.max_condition_number,
        holdout_score_relative_rmse=config.structured_holdout_score_relative_rmse)
    if "_native_factor_data" not in data:
        result = fit_factor_correlation_score_geometry(data["center_score_z"],
            data["training_offsets_z"], data["training_scores_z"], data["holdout_offsets_z"],
            data["holdout_scores_z"], training_weights=data["training_weights"], config=factor_config)
    else:
        numerical = data["_native_factor_data"]
        dimension = int(numerical["center_score_z"].shape[0])
        rows = int(numerical["training_offsets_z"].shape[0])
        holdout_rows = int(numerical["holdout_offsets_z"].shape[0])
        program = structured_fit_data_program(dimension, rows, holdout_rows, factor_config)
        computed = program(numerical["center_score_z"], numerical["training_offsets_z"], numerical["training_scores_z"],
            numerical["holdout_offsets_z"], numerical["holdout_scores_z"], numerical["training_weights"],
            numerical["active_training_rows"])
        result = _factor_result_from_native(computed, factor_config, dimension,
            int(numerical["active_training_rows"]), holdout_rows)
    return _factor_fit_payload(result, data)


def _factor_result_from_native(computed, factor_config, dimension, active_rows, holdout_rows):
    """Materialize the native input gate and complete fit without refitting."""
    input_status = int(computed["input_status"])
    if input_status == 1:
        count = 2 * dimension if factor_config.factor_count == 1 else 3 * dimension - 1
        return _rejected_factor_fit(factor_config, dimension,
            "factor_parameterization_dimensionally_unidentified", parameter_count=count,
            diagnostics={"parameter_count": count,
                "symmetric_covariance_entry_count": dimension * (dimension + 1) // 2})
    if input_status == 2:
        return _rejected_factor_fit(factor_config, dimension, "nonfinite_fit_inputs")
    if input_status in (3, 4):
        raise ValueError("prepared training rows and active weights must be valid")
    return _factor_result_from_computed(computed["fit"], factor_config, dimension,
        active_rows, holdout_rows, True)


def _factor_fit_payload(result, data):
    """Restore existing reporting metadata after numerical completion."""
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
            "best_exact_value": data.get("best_exact_value"),
            "best_exact_position": data.get("best_exact_position"),
            "best_exact_score": data.get("best_exact_score"),
            "best_exact_source": data.get("best_exact_source"),
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
    if config.pair_disjoint_score_holdout:
        train_indices, holdout_indices = _score_fit_partition_indices(
            sample_count,
            config.holdout_fraction,
            pair_disjoint=True,
        )
        support_count = len(train_indices) // 2
    else:
        support_count = sample_count
    if support_count * dimension < coefficient_count + dimension:
        return {"status": "insufficient_symmetric_support", "seed": list(seed)}, evaluations
    training_indices, holdout_indices = partition_schema(sample_count,
        config.holdout_fraction, pair_disjoint=config.pair_disjoint_score_holdout)
    result = score_fit_program(function, batched_value_and_score_fn, sample_count,
        dimension, training_indices, holdout_indices)(center, center_score, scale,
        tf.convert_to_tensor(radius, tf.float64), tf.convert_to_tensor(seed, tf.int32),
        tf.constant(config.ridge, tf.float64), tf.constant(config.eigenvalue_floor, tf.float64),
        tf.constant(config.max_condition_number, tf.float64),
        tf.constant(config.score_holdout_relative_rmse, tf.float64))
    evaluations += sample_count
    has_winner = int(result["best_index"]) >= 0
    best_exact = {
        "best_exact_value": float(result["best_value"]) if has_winner else None,
        "best_exact_position": result["best_position"] if has_winner else None,
        "best_exact_score": result["best_score"] if has_winner else None,
        "best_exact_source": "score_fit_cloud" if has_winner else None,
    }
    status = ("rank_deficient_symmetric_fit", "usable", "score_holdout_failed")[int(result["status"])]
    payload = {"status": status, "rank": int(result["rank"]), "seed": list(seed), **best_exact}
    if status == "rank_deficient_symmetric_fit":
        return payload, evaluations
    # Restore the original host result schema after the complete numerical call.
    # Fit arrays remain frozen to preserve the old materialization boundary.
    return {
        **payload,
        "train_score_rmse": float(result["train_score_rmse"]),
        "holdout_score_relative_rmse": float(result["holdout_score_relative_rmse"]),
        "raw_eigenvalues": tf.stop_gradient(result["raw_eigenvalues"]),
        "projected_eigenvalues": tf.stop_gradient(result["projected_eigenvalues"]),
        "projection_relative_frobenius": float(result["projection_relative_frobenius"]),
        "projected_precision_z": tf.stop_gradient(result["projected_precision_z"]),
        **({"pair_disjoint_score_holdout": True,
            "training_sample_count": len(training_indices),
            "holdout_sample_count": len(holdout_indices)} if config.pair_disjoint_score_holdout else {}),
    }, evaluations


def _replay_locator_candidates(function, positions):
    """Replay every locator candidate with the scalar authority and select."""
    return replay_program(function, int(positions.shape[0]), int(positions.shape[1]))(positions)


def _search_exact_candidates(scalar, batched, center, value, score, scale, radius, seed,
                             *, sample_count, orthogonal):
    """Enclose seeded search, exact values/scores and stable incumbent choice."""
    return search_program(scalar, batched, sample_count, int(center.shape[0]), orthogonal)(
        center, value, score, scale, radius, seed)


def _fit_best_exact_candidate(
    fit: Mapping[str, Any],
) -> tuple[float, tf.Tensor, tf.Tensor] | None:
    """Extract a finite exact cloud winner from one fit result."""

    value = fit.get("best_exact_value")
    position = fit.get("best_exact_position")
    score = fit.get("best_exact_score")
    if value is None or position is None or score is None:
        return None
    value_float = float(value)
    position_tensor = tf.reshape(tf.convert_to_tensor(position, tf.float64), [-1])
    score_tensor = tf.reshape(tf.convert_to_tensor(score, tf.float64), [-1])
    finite = bool(
        (
            tf.math.is_finite(value_float)
            & tf.reduce_all(
                tf.math.is_finite(position_tensor) & tf.math.is_finite(score_tensor)
            )
        ).numpy()
    )
    if not finite:
        return None
    return value_float, position_tensor, score_tensor


def _score_fit_partition_indices(
    sample_count: int,
    holdout_fraction: float,
    *,
    pair_disjoint: bool,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Partition score rows while optionally keeping antithetic pairs together."""

    training, holdout = partition_schema(sample_count, holdout_fraction, pair_disjoint=pair_disjoint)
    return tf.constant(training, tf.int64), tf.constant(holdout, tf.int64)


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
    if row_count == 0 and batched_value_and_score_fn is None:
        raise ValueError("scalar cloud must contain at least one row")
    return evaluation_program(scalar_function, batched_value_and_score_fn, row_count, dimension)(rows)


def _solve_trust_region_tf(precision: tf.Tensor, linear: tf.Tensor, radius: float) -> Mapping[str, Any]:
    step, boundary, predicted = trust_region_program(int(linear.shape[0]))(
        precision, linear, tf.convert_to_tensor(radius, tf.float64))
    return {"step": step, "boundary_active": bool(boundary), "predicted_improvement": float(predicted)}


def _proposal_gate_payload(row, config):
    """Restore optional reporting fields from a completed native decision."""
    return {"policy": config.proposal_score_acceptance_policy,
        "active": config.require_proposal_score_reduction,
        "passed": bool(row["score_passed"]), "legacy_fractional_passed": bool(row["legacy_passed"]),
        "required_score_norm_max": float(row["required_norm_max"]) if config.require_proposal_score_reduction else None,
        "numerical_resolution_floor": (float(row["resolution_floor"])
            if config.require_proposal_score_reduction and config.proposal_score_acceptance_policy == "resolvable_decrease"
            else None)}


def _proposal_score_gate(
    old_norm: float,
    new_norm: float,
    *,
    policy: str,
    fractional_factor: float,
    active: bool,
) -> Mapping[str, Any]:
    """Legacy diagnostic authority; runtime uses the native proposal program."""

    old = float(old_norm)
    new = float(new_norm)
    factor = float(fractional_factor)
    finite = bool(math.isfinite(old) and math.isfinite(new))
    legacy_threshold = factor * old if finite else float("nan")
    legacy_passed = bool(finite and new <= legacy_threshold)
    if not active:
        return {
            "policy": str(policy),
            "active": False,
            "passed": True,
            "legacy_fractional_passed": legacy_passed,
            "numerical_resolution_floor": None,
            "required_score_norm_max": None,
        }
    if policy == "fractional":
        return {
            "policy": policy,
            "active": True,
            "passed": legacy_passed,
            "legacy_fractional_passed": legacy_passed,
            "numerical_resolution_floor": None,
            "required_score_norm_max": legacy_threshold,
        }
    if policy != "resolvable_decrease":
        raise ValueError(f"unknown proposal score acceptance policy {policy!r}")
    resolution_floor = float(
        math.sqrt(sys.float_info.epsilon) * max(1.0, abs(old))
    )
    required_max = old - resolution_floor
    return {
        "policy": policy,
        "active": True,
        "passed": bool(finite and old - new > resolution_floor),
        "legacy_fractional_passed": legacy_passed,
        "numerical_resolution_floor": resolution_floor,
        "required_score_norm_max": required_max,
    }


def _proposal_is_accepted(
    finite: bool,
    *,
    actual: float,
    predicted: float,
    rho: float,
    acceptance_ratio: float,
    score_gate_passed: bool,
) -> bool:
    """Legacy diagnostic authority for the native acceptance conjunction."""

    return bool(
        finite
        and actual > 0.0
        and predicted > 0.0
        and rho >= acceptance_ratio
        and score_gate_passed
    )


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
            "proposal_score_acceptance_policy": (
                config.proposal_score_acceptance_policy
            ),
            "proposal_score_gate_active": (
                config.require_proposal_score_reduction
            ),
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
    if tf.is_tensor(value) or hasattr(value, "__array_interface__"):
        return numeric_tensor(value).numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value
