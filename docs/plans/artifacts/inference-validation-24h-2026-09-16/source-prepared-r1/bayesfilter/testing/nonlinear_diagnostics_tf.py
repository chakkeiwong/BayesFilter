"""Testing diagnostics for nonlinear TensorFlow sigma-point filters.

These helpers are BayesFilter-local testing tools.  They normalize value and
analytic-score diagnostic envelopes so tests and benchmark scripts can compare
SVD cubature, SVD-UKF, and SVD-CUT4 without depending on backend-specific
container details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_filter
from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
    TFStructuralFirstDerivatives,
    tf_svd_cubature_score,
    tf_svd_cut4_score,
    tf_svd_ukf_score,
)
from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_filter
from bayesfilter.results_tf import TFFilterDerivativeResult, TFFilterValueResult
from bayesfilter.structural_tf import TFStructuralStateSpace

TFNonlinearSigmaPointBackend = Literal["tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4"]
TFNonlinearBranchMode = Literal["value", "score"]
TFStructuralModelBuilder = Callable[[tf.Tensor], TFStructuralStateSpace]
TFStructuralDerivativeBuilder = Callable[[tf.Tensor], TFStructuralFirstDerivatives]


@dataclass(frozen=True)
class NonlinearSigmaPointDiagnosticSnapshot:
    """Scalar diagnostic vocabulary shared by nonlinear sigma-point backends."""

    backend: str
    mode: str
    rule: str
    point_count: int
    polynomial_degree: int
    max_integration_rank: int
    placement_floor_count: int
    innovation_floor_count: int
    min_placement_eigen_gap: float
    min_innovation_eigen_gap: float
    support_residual: float
    deterministic_residual: float
    placement_psd_projection_residual: float
    innovation_psd_projection_residual: float
    implemented_covariance_trace: float
    regularization_branch_label: str
    regularization_derivative_target: str
    derivative_branch: str
    derivative_method: str
    structural_null_count: int
    structural_null_covariance_residual: float
    fixed_null_derivative_residual: float
    sigma_point_variable: str


@dataclass(frozen=True)
class NonlinearSigmaPointBranchSummary:
    """Small eager-mode summary of value or score branch outcomes."""

    backend: str
    mode: str
    total_count: int
    ok_count: int
    active_floor_count: int
    weak_spectral_gap_count: int
    nonfinite_count: int
    other_blocked_count: int
    min_placement_eigen_gap: float
    min_innovation_eigen_gap: float
    max_support_residual: float
    max_deterministic_residual: float
    max_structural_null_covariance_residual: float
    max_fixed_null_derivative_residual: float
    max_structural_null_count: int
    max_integration_rank: int
    max_point_count: int
    failure_labels: tuple[str, ...]

    @property
    def ok_fraction(self) -> float:
        return 0.0 if self.total_count == 0 else self.ok_count / self.total_count


def tf_nonlinear_sigma_point_value_filter(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    *,
    backend: TFNonlinearSigmaPointBackend,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
    return_filtered: bool = False,
) -> TFFilterValueResult:
    """Dispatch to one of the three nonlinear sigma-point value filters."""

    if backend == "tf_svd_cut4":
        return tf_svd_cut4_filter(
            observations,
            model,
            placement_floor=placement_floor,
            innovation_floor=innovation_floor,
            rank_tolerance=rank_tolerance,
            jitter=jitter,
            return_filtered=return_filtered,
        )
    if backend in {"tf_svd_cubature", "tf_svd_ukf"}:
        return tf_svd_sigma_point_filter(
            observations,
            model,
            backend=backend,
            placement_floor=placement_floor,
            innovation_floor=innovation_floor,
            rank_tolerance=rank_tolerance,
            jitter=jitter,
            return_filtered=return_filtered,
        )
    raise ValueError(f"unknown nonlinear sigma-point backend: {backend}")


def tf_nonlinear_sigma_point_score(
    observations: tf.Tensor,
    model: TFStructuralStateSpace,
    derivatives: TFStructuralFirstDerivatives,
    *,
    backend: TFNonlinearSigmaPointBackend,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    fixed_null_tolerance: tf.Tensor | float = 1e-10,
    jitter: tf.Tensor | float = 0.0,
    allow_fixed_null_support: bool = False,
) -> TFFilterDerivativeResult:
    """Dispatch to one of the analytic first-order sigma-point scores."""

    kwargs = {
        "placement_floor": placement_floor,
        "innovation_floor": innovation_floor,
        "rank_tolerance": rank_tolerance,
        "spectral_gap_tolerance": spectral_gap_tolerance,
        "fixed_null_tolerance": fixed_null_tolerance,
        "jitter": jitter,
        "allow_fixed_null_support": allow_fixed_null_support,
    }
    if backend == "tf_svd_cubature":
        return tf_svd_cubature_score(observations, model, derivatives, **kwargs)
    if backend == "tf_svd_ukf":
        return tf_svd_ukf_score(observations, model, derivatives, **kwargs)
    if backend == "tf_svd_cut4":
        return tf_svd_cut4_score(observations, model, derivatives, **kwargs)
    raise ValueError(f"unknown nonlinear sigma-point backend: {backend}")


def nonlinear_sigma_point_diagnostic_snapshot(
    result: TFFilterValueResult | TFFilterDerivativeResult,
    *,
    mode: TFNonlinearBranchMode,
) -> NonlinearSigmaPointDiagnosticSnapshot:
    """Extract comparable scalar diagnostics from a value or score result."""

    diagnostics = result.diagnostics
    if not isinstance(diagnostics, TFFilterDiagnostics):
        raise TypeError("result must carry TFFilterDiagnostics")
    extra = diagnostics.extra
    regularization = diagnostics.regularization
    implemented_covariance = regularization.implemented_covariance
    if implemented_covariance is None:
        implemented_trace = float("nan")
    else:
        implemented_trace = _to_float(tf.linalg.trace(_as_tensor(implemented_covariance)))
    return NonlinearSigmaPointDiagnosticSnapshot(
        backend=diagnostics.backend,
        mode=mode,
        rule=_as_str(extra["rule"]),
        point_count=_to_int(extra["point_count"]),
        polynomial_degree=_to_int(extra["polynomial_degree"]),
        max_integration_rank=_to_int(extra["max_integration_rank"]),
        placement_floor_count=_to_int(extra.get("placement_floor_count", 0)),
        innovation_floor_count=_to_int(extra.get("innovation_floor_count", regularization.floor_count)),
        min_placement_eigen_gap=_to_float(extra["min_placement_eigen_gap"]),
        min_innovation_eigen_gap=_to_float(extra["min_innovation_eigen_gap"]),
        support_residual=_to_float(extra["support_residual"]),
        deterministic_residual=_to_float(extra["deterministic_residual"]),
        placement_psd_projection_residual=_to_float(
            extra.get("placement_psd_projection_residual", 0.0)
        ),
        innovation_psd_projection_residual=_to_float(
            extra.get(
                "innovation_psd_projection_residual",
                regularization.psd_projection_residual,
            )
        ),
        implemented_covariance_trace=implemented_trace,
        regularization_branch_label=str(regularization.branch_label),
        regularization_derivative_target=str(regularization.derivative_target),
        derivative_branch=str(extra.get("derivative_branch", "value_only")),
        derivative_method=str(extra.get("derivative_method", "value_only")),
        structural_null_count=_to_int(extra.get("structural_null_count", 0)),
        structural_null_covariance_residual=_to_float(
            extra.get("structural_null_covariance_residual", 0.0)
        ),
        fixed_null_derivative_residual=_to_float(
            extra.get("fixed_null_derivative_residual", 0.0)
        ),
        sigma_point_variable=str(extra.get("sigma_point_variable", "not_declared")),
    )


def nonlinear_sigma_point_value_branch_summary(
    observations: tf.Tensor,
    parameter_grid: tf.Tensor,
    model_builder: TFStructuralModelBuilder,
    *,
    backend: TFNonlinearSigmaPointBackend,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    jitter: tf.Tensor | float = 0.0,
) -> NonlinearSigmaPointBranchSummary:
    """Aggregate finite value diagnostics over a small parameter grid."""

    def evaluate(row: tf.Tensor) -> TFFilterValueResult:
        return tf_nonlinear_sigma_point_value_filter(
            observations,
            model_builder(row),
            backend=backend,
            placement_floor=placement_floor,
            innovation_floor=innovation_floor,
            rank_tolerance=rank_tolerance,
            jitter=jitter,
        )

    return _branch_summary_from_grid(
        parameter_grid,
        backend=backend,
        mode="value",
        evaluate=evaluate,
        snapshot_mode="value",
    )


def nonlinear_sigma_point_score_branch_summary(
    observations: tf.Tensor,
    parameter_grid: tf.Tensor,
    model_builder: TFStructuralModelBuilder,
    derivative_builder: TFStructuralDerivativeBuilder,
    *,
    backend: TFNonlinearSigmaPointBackend,
    placement_floor: tf.Tensor | float = 0.0,
    innovation_floor: tf.Tensor | float = 1e-12,
    rank_tolerance: tf.Tensor | float = 1e-12,
    spectral_gap_tolerance: tf.Tensor | float = 1e-8,
    fixed_null_tolerance: tf.Tensor | float = 1e-10,
    jitter: tf.Tensor | float = 0.0,
    allow_fixed_null_support: bool = False,
) -> NonlinearSigmaPointBranchSummary:
    """Aggregate analytic score branch diagnostics over a small parameter grid."""

    def evaluate(row: tf.Tensor) -> TFFilterDerivativeResult:
        return tf_nonlinear_sigma_point_score(
            observations,
            model_builder(row),
            derivative_builder(row),
            backend=backend,
            placement_floor=placement_floor,
            innovation_floor=innovation_floor,
            rank_tolerance=rank_tolerance,
            spectral_gap_tolerance=spectral_gap_tolerance,
            fixed_null_tolerance=fixed_null_tolerance,
            jitter=jitter,
            allow_fixed_null_support=allow_fixed_null_support,
        )

    return _branch_summary_from_grid(
        parameter_grid,
        backend=backend,
        mode="score",
        evaluate=evaluate,
        snapshot_mode="score",
    )


def _branch_summary_from_grid(
    parameter_grid: tf.Tensor,
    *,
    backend: str,
    mode: str,
    evaluate: Callable[[tf.Tensor], TFFilterValueResult | TFFilterDerivativeResult],
    snapshot_mode: TFNonlinearBranchMode,
) -> NonlinearSigmaPointBranchSummary:
    grid = tf.convert_to_tensor(parameter_grid, dtype=tf.float64)
    ok_count = 0
    active_floor_count = 0
    weak_spectral_gap_count = 0
    nonfinite_count = 0
    other_blocked_count = 0
    min_placement_gap = float("inf")
    min_innovation_gap = float("inf")
    max_support_residual = 0.0
    max_deterministic_residual = 0.0
    max_structural_null_covariance_residual = 0.0
    max_fixed_null_derivative_residual = 0.0
    max_structural_null_count = 0
    max_integration_rank = 0
    max_point_count = 0
    failure_labels: list[str] = []

    for row in tf.unstack(grid, axis=0):
        try:
            result = evaluate(row)
            value = tf.convert_to_tensor(result.log_likelihood, dtype=tf.float64)
            tf.debugging.assert_all_finite(
                value,
                "blocked_nonfinite_value: nonlinear sigma-point value is nonfinite",
            )
            if isinstance(result, TFFilterDerivativeResult):
                tf.debugging.assert_all_finite(
                    result.score,
                    "blocked_nonfinite_score: nonlinear sigma-point score is nonfinite",
                )
            snapshot = nonlinear_sigma_point_diagnostic_snapshot(
                result,
                mode=snapshot_mode,
            )
            ok_count += 1
            min_placement_gap = min(min_placement_gap, snapshot.min_placement_eigen_gap)
            min_innovation_gap = min(min_innovation_gap, snapshot.min_innovation_eigen_gap)
            max_support_residual = max(max_support_residual, snapshot.support_residual)
            max_deterministic_residual = max(
                max_deterministic_residual,
                snapshot.deterministic_residual,
            )
            max_structural_null_covariance_residual = max(
                max_structural_null_covariance_residual,
                snapshot.structural_null_covariance_residual,
            )
            max_fixed_null_derivative_residual = max(
                max_fixed_null_derivative_residual,
                snapshot.fixed_null_derivative_residual,
            )
            max_structural_null_count = max(
                max_structural_null_count,
                snapshot.structural_null_count,
            )
            max_integration_rank = max(max_integration_rank, snapshot.max_integration_rank)
            max_point_count = max(max_point_count, snapshot.point_count)
        except tf.errors.InvalidArgumentError as exc:
            message = str(exc)
            if "blocked_active_floor" in message:
                active_floor_count += 1
                _append_failure_label(failure_labels, "blocked_active_floor")
            elif "blocked_weak_spectral_gap" in message:
                weak_spectral_gap_count += 1
                _append_failure_label(failure_labels, "blocked_weak_spectral_gap")
            elif "blocked_nonfinite" in message:
                nonfinite_count += 1
                _append_failure_label(failure_labels, "blocked_nonfinite")
            else:
                other_blocked_count += 1
                _append_failure_label(failure_labels, _failure_label(message))

    total_count = int(grid.shape[0])
    if ok_count == 0:
        min_placement_gap = float("nan")
        min_innovation_gap = float("nan")
    return NonlinearSigmaPointBranchSummary(
        backend=backend,
        mode=mode,
        total_count=total_count,
        ok_count=ok_count,
        active_floor_count=active_floor_count,
        weak_spectral_gap_count=weak_spectral_gap_count,
        nonfinite_count=nonfinite_count,
        other_blocked_count=other_blocked_count,
        min_placement_eigen_gap=min_placement_gap,
        min_innovation_eigen_gap=min_innovation_gap,
        max_support_residual=max_support_residual,
        max_deterministic_residual=max_deterministic_residual,
        max_structural_null_covariance_residual=max_structural_null_covariance_residual,
        max_fixed_null_derivative_residual=max_fixed_null_derivative_residual,
        max_structural_null_count=max_structural_null_count,
        max_integration_rank=max_integration_rank,
        max_point_count=max_point_count,
        failure_labels=tuple(failure_labels),
    )


def _append_failure_label(labels: list[str], label: str, *, limit: int = 5) -> None:
    if label not in labels and len(labels) < limit:
        labels.append(label)


def _failure_label(message: str) -> str:
    for marker in (
        "blocked_structural_null_covariance",
        "blocked_fixed_null_derivative",
        "blocked_active_floor",
        "blocked_weak_spectral_gap",
        "blocked_nonfinite_value",
        "blocked_nonfinite_score",
    ):
        if marker in message:
            return marker
    first_line = message.splitlines()[0] if message else "blocked_unknown"
    return first_line[:80]


def _as_tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, dtype=tf.float64)


def _to_float(value: object) -> float:
    return float(_as_tensor(value).numpy())


def _to_int(value: object) -> int:
    return int(tf.convert_to_tensor(value).numpy())


def _as_str(value: object) -> str:
    tensor = tf.convert_to_tensor(value)
    if tensor.dtype == tf.string:
        return tensor.numpy().decode("utf-8")
    return str(value)
