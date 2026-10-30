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
from bayesfilter.inference.sequential_controller_report import sequential_result
from bayesfilter.inference.sequential_controller_tf import sequential_controller
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

    Numerical localization, refinement and mass preparation execute in one XLA
    call. Progress after initializer_started is buffered and delivered in order
    after completion; callbacks cannot interrupt that call. Buffered locator
    overflow stops further target calls and raises before event delivery. Use an
    independent process deadline when an interruptible wall-time limit is needed.
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

    start_count = int(starts.shape[0])
    if cfg.locator_policy == "center_first" and start_count != 1:
        raise ValueError("center_first locator_policy requires exactly one center")
    _emit_progress(progress_callback, "initializer_started", start_count=start_count,
        dimension=dimension, locator_policy=cfg.locator_policy,
        locator_stopping_condition=cfg.locator_stopping_condition)
    search_count = dimension_scaled_search_count(dimension) if cfg.dimension_scaled_search else cfg.search_sample_count
    owner = sequential_controller(value_and_score_fn, batched_value_and_score_fn,
        batched_locator_value_and_score_fn, start_count, dimension, cfg, search_count,
        progress=progress_callback is not None, device=starts.device)
    computed = owner(starts, scale_tf)
    return sequential_result(computed, cfg, start_count, dimension, progress_callback,
        emit_started=False)


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
    """Independent legacy diagnostic; runtime uses tensor movement records."""

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


def _factor_result_from_native(computed, factor_config, dimension, active_rows, holdout_rows, *, decision=None):
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
        active_rows, holdout_rows, True, decision=decision)


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
