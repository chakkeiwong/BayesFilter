"""Quadratic MAP-candidate covariance initializer.

This module provides a diagnostic initializer for later HMC tuning.  It uses a
local optimizer only to locate a finite neighborhood, then treats the
constrained SPD quadratic geometry as the covariance/precision authority.

It does not certify a global MAP, posterior covariance correctness, HMC
readiness, sampler convergence, or source-faithful Zhao-Cui behavior.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.joint_center import (
    JointCenterLocatorConfig,
    locate_joint_center,
)
from bayesfilter.inference.mass_matrix import (
    MassMatrixResult,
    covariance_from_precision,
)
from bayesfilter.inference.quadratic_geometry import (
    LowRankSPDQuadraticGeometryConfig,
    LowRankSPDQuadraticGeometryResult,
    fit_low_rank_spd_quadratic_geometry,
)


QUADRATIC_MAP_COVARIANCE_NONCLAIMS = (
    "quadratic MAP-candidate covariance diagnostic only",
    "optimizer output is a locator only",
    "not a certified global MAP",
    "not posterior covariance correctness evidence",
    "not HMC convergence evidence",
    "not HMC readiness evidence",
    "not sampler superiority evidence",
    "not default-readiness evidence",
    "not source-faithful Zhao-Cui evidence",
)

ITERATIVE_QUADRATIC_MAP_COVARIANCE_NONCLAIMS = (
    "iterative quadratic trust-region diagnostic only",
    "each accepted move requires exact target improvement checks",
    "optimizer output is a locator only",
    "not a certified global MAP",
    "not posterior covariance correctness evidence",
    "not HMC convergence evidence",
    "not HMC readiness evidence",
    "not sampler superiority evidence",
    "not default-readiness evidence",
    "not source-faithful Zhao-Cui evidence",
)


@dataclass(frozen=True)
class QuadraticMapCovarianceLocatorConfig:
    """Configuration for the finite-neighborhood locator."""

    enabled: bool = True
    max_iterations: int = 50
    tolerance: float = 1.0e-8
    log_prob_tolerance: float = 1.0e-8
    parallel_iterations: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", bool(self.enabled))
        for name in ("max_iterations", "parallel_iterations"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("tolerance", "log_prob_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def payload(self) -> Mapping[str, Any]:
        return {
            "enabled": self.enabled,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "log_prob_tolerance": self.log_prob_tolerance,
            "parallel_iterations": self.parallel_iterations,
        }


@dataclass(frozen=True)
class QuadraticMapCovarianceMassConfig:
    """Configuration for converting fitted precision to covariance."""

    jitter: float = 1.0e-9
    eigenvalue_floor: float | None = None
    max_condition_number: float | None = None
    dense: bool = True

    def __post_init__(self) -> None:
        jitter = float(self.jitter)
        if not np.isfinite(jitter) or jitter < 0.0:
            raise ValueError("jitter must be finite and non-negative")
        object.__setattr__(self, "jitter", jitter)
        if self.eigenvalue_floor is not None:
            floor = float(self.eigenvalue_floor)
            if not np.isfinite(floor) or floor < 0.0:
                raise ValueError("eigenvalue_floor must be finite and non-negative")
            object.__setattr__(self, "eigenvalue_floor", floor)
        if self.max_condition_number is not None:
            condition = float(self.max_condition_number)
            if not np.isfinite(condition) or condition <= 1.0:
                raise ValueError("max_condition_number must be finite and greater than 1")
            object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "dense", bool(self.dense))

    def payload(self) -> Mapping[str, Any]:
        return {
            "jitter": self.jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "dense": self.dense,
        }


@dataclass(frozen=True)
class IterativeQuadraticMapCovarianceConfig:
    """Configuration for bounded local quadratic recentering."""

    max_refinement_steps: int = 8
    terminal_score_max_abs: float = 1.0e-1

    def __post_init__(self) -> None:
        steps = int(self.max_refinement_steps)
        if steps <= 0:
            raise ValueError("max_refinement_steps must be positive")
        terminal = float(self.terminal_score_max_abs)
        if not np.isfinite(terminal) or terminal <= 0.0:
            raise ValueError("terminal_score_max_abs must be positive finite")
        object.__setattr__(self, "max_refinement_steps", steps)
        object.__setattr__(self, "terminal_score_max_abs", terminal)

    def payload(self) -> Mapping[str, Any]:
        return {
            "max_refinement_steps": self.max_refinement_steps,
            "terminal_score_max_abs": self.terminal_score_max_abs,
        }


@dataclass(frozen=True)
class QuadraticMapCovarianceResult:
    """Structured result for a quadratic covariance initializer attempt."""

    accepted: bool
    status: str
    dimension: int
    initial_position: np.ndarray
    locator_position: np.ndarray
    map_candidate: np.ndarray | None
    map_candidate_role: str
    precision: np.ndarray | None
    covariance: np.ndarray | None
    covariance_source: str | None
    locator_diagnostics: Mapping[str, Any]
    geometry: LowRankSPDQuadraticGeometryResult | None
    mass_matrix: MassMatrixResult | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = QUADRATIC_MAP_COVARIANCE_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "dimension", int(self.dimension))
        for name in ("initial_position", "locator_position"):
            array = np.asarray(getattr(self, name), dtype=float).reshape([-1]).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "map_candidate_role", str(self.map_candidate_role))
        if self.covariance_source is not None:
            object.__setattr__(self, "covariance_source", str(self.covariance_source))
        object.__setattr__(self, "locator_diagnostics", _json_ready(dict(self.locator_diagnostics)))
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def payload(self, *, include_arrays: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "bayesfilter.quadratic_map_covariance.v1",
            "accepted": self.accepted,
            "status": self.status,
            "dimension": self.dimension,
            "map_candidate_role": self.map_candidate_role,
            "covariance_source": self.covariance_source,
            "locator_diagnostics": self.locator_diagnostics,
            "geometry": None
            if self.geometry is None
            else self.geometry.payload(include_arrays=include_arrays),
            "mass_matrix": None
            if self.mass_matrix is None
            else _mass_matrix_payload(self.mass_matrix, include_arrays=include_arrays),
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if self.precision is not None:
            payload["precision_eigen_summary"] = _eigen_summary(self.precision)
        if self.covariance is not None:
            payload["covariance_eigen_summary"] = _eigen_summary(self.covariance)
        if include_arrays:
            payload.update(
                {
                    "initial_position": self.initial_position,
                    "locator_position": self.locator_position,
                    "map_candidate": self.map_candidate,
                    "precision": self.precision,
                    "covariance": self.covariance,
                }
            )
        return _json_ready(payload)


@dataclass(frozen=True)
class IterativeQuadraticMapCovarianceResult:
    """Structured result for bounded iterative quadratic recentering."""

    accepted: bool
    status: str
    dimension: int
    initial_position: np.ndarray
    locator_position: np.ndarray
    map_candidate: np.ndarray | None
    map_candidate_role: str
    precision: np.ndarray | None
    covariance: np.ndarray | None
    covariance_source: str | None
    locator_diagnostics: Mapping[str, Any]
    iterations: tuple[Mapping[str, Any], ...]
    terminal_geometry: LowRankSPDQuadraticGeometryResult | None
    mass_matrix: MassMatrixResult | None
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...] = ITERATIVE_QUADRATIC_MAP_COVARIANCE_NONCLAIMS

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "dimension", int(self.dimension))
        for name in ("initial_position", "locator_position"):
            array = np.asarray(getattr(self, name), dtype=float).reshape([-1]).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "map_candidate_role", str(self.map_candidate_role))
        if self.covariance_source is not None:
            object.__setattr__(self, "covariance_source", str(self.covariance_source))
        object.__setattr__(
            self,
            "locator_diagnostics",
            _json_ready(dict(self.locator_diagnostics)),
        )
        object.__setattr__(
            self,
            "iterations",
            tuple(_json_ready(dict(item)) for item in self.iterations),
        )
        object.__setattr__(self, "diagnostics", _json_ready(dict(self.diagnostics)))
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def payload(self, *, include_arrays: bool = False) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "bayesfilter.iterative_quadratic_map_covariance.v1",
            "accepted": self.accepted,
            "status": self.status,
            "dimension": self.dimension,
            "map_candidate_role": self.map_candidate_role,
            "covariance_source": self.covariance_source,
            "locator_diagnostics": self.locator_diagnostics,
            "iterations": self.iterations,
            "terminal_geometry": (
                None
                if self.terminal_geometry is None
                else self.terminal_geometry.payload(include_arrays=include_arrays)
            ),
            "mass_matrix": (
                None
                if self.mass_matrix is None
                else _mass_matrix_payload(self.mass_matrix, include_arrays=include_arrays)
            ),
            "diagnostics": self.diagnostics,
            "nonclaims": self.nonclaims,
        }
        if self.precision is not None:
            payload["precision_eigen_summary"] = _eigen_summary(self.precision)
        if self.covariance is not None:
            payload["covariance_eigen_summary"] = _eigen_summary(self.covariance)
        if include_arrays:
            payload.update(
                {
                    "initial_position": self.initial_position,
                    "locator_position": self.locator_position,
                    "map_candidate": self.map_candidate,
                    "precision": self.precision,
                    "covariance": self.covariance,
                }
            )
        return _json_ready(payload)


def estimate_quadratic_map_covariance(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: Any,
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
    scale: Any | None = None,
    locator_config: QuadraticMapCovarianceLocatorConfig | None = None,
    quadratic_config: LowRankSPDQuadraticGeometryConfig | None = None,
    mass_config: QuadraticMapCovarianceMassConfig | None = None,
) -> QuadraticMapCovarianceResult:
    """Estimate a diagnostic MAP candidate and covariance initializer.

    ``value_and_score_fn`` must accept a one-dimensional TensorFlow tensor and
    return scalar log probability and gradient in the same coordinates as
    ``initial_position``. When supplied, ``batched_value_and_score_fn`` accepts
    ``[batch, dimension]`` positions and returns batched values and scores for
    the quadratic pilot/design clouds. L-BFGS is used only to choose the geometry center.
    Accepted covariance is rebuilt from the fitted SPD precision via
    :func:`covariance_from_precision`.  When ``scale`` is supplied, the
    quadratic fit is performed in whitened ``z`` coordinates, but the returned
    ``precision`` and ``covariance`` are transformed back to the original
    ``initial_position`` coordinates.
    """

    locator_cfg = (
        QuadraticMapCovarianceLocatorConfig()
        if locator_config is None
        else locator_config
    )
    geometry_cfg = (
        LowRankSPDQuadraticGeometryConfig()
        if quadratic_config is None
        else quadratic_config
    )
    mass_cfg = QuadraticMapCovarianceMassConfig() if mass_config is None else mass_config

    initial_np = _vector(initial_position, "initial_position")
    dim = int(initial_np.size)
    initial_value, initial_score, initial_status = _evaluate_value_score(
        value_and_score_fn,
        initial_np,
        dim,
    )
    if initial_status != "finite":
        return _rejected_result(
            status="initial_value_or_score_nonfinite",
            initial_position=initial_np,
            locator_position=initial_np,
            locator_diagnostics={
                "status": "not_run_initial_nonfinite",
                "initial_log_prob": initial_value,
                "initial_evaluation_status": initial_status,
            },
            geometry=None,
            mass_matrix=None,
            diagnostics={
                "classification": "diagnostic_initializer_rejected",
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    locator_position, locator_diagnostics = _run_locator(
        value_and_score_fn=value_and_score_fn,
        initial_position=initial_np,
        initial_value=initial_value,
        initial_score=initial_score,
        config=locator_cfg,
    )

    geometry = fit_low_rank_spd_quadratic_geometry(
        value_and_score_fn,
        locator_position,
        batched_value_and_score_fn=batched_value_and_score_fn,
        scale=scale,
        config=geometry_cfg,
    )
    if not geometry.accepted or geometry.precision is None:
        return _rejected_result(
            status=f"geometry_{geometry.status}",
            initial_position=initial_np,
            locator_position=locator_position,
            locator_diagnostics=locator_diagnostics,
            geometry=geometry,
            mass_matrix=None,
            diagnostics={
                "classification": "diagnostic_initializer_rejected",
                "geometry_status": geometry.status,
                "geometry_accepted": geometry.accepted,
                "mass_matrix_attempted": False,
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    geometry_incumbent = (
        None
        if geometry.best_evaluated_position is None
        else np.asarray(geometry.best_evaluated_position, dtype=float).reshape([-1])
    )
    if geometry_incumbent is not None and _geometry_incumbent_move_is_material(
        geometry
    ):
        return _rejected_result(
            status="covariance_center_mismatch_requires_refit",
            initial_position=initial_np,
            locator_position=locator_position,
            locator_diagnostics=locator_diagnostics,
            geometry=geometry,
            mass_matrix=None,
            map_candidate=geometry_incumbent,
            map_candidate_role=f"best_exact_{geometry.best_evaluated_source}",
            diagnostics={
                "classification": "diagnostic_initializer_rejected",
                "geometry_status": geometry.status,
                "geometry_accepted": geometry.accepted,
                "stale_centered_covariance_prevented": True,
                "covariance_fit_center": geometry.center,
                "exact_incumbent_source": geometry.best_evaluated_source,
                "exact_incumbent_value": geometry.best_evaluated_value,
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    theta_precision = _precision_from_geometry_to_theta(geometry)
    transform_diagnostics = _coordinate_transform_diagnostics(geometry)
    try:
        mass = covariance_from_precision(
            theta_precision,
            source="low_rank_spd_quadratic_geometry_precision_theta_coordinates",
            jitter=mass_cfg.jitter,
            eigenvalue_floor=mass_cfg.eigenvalue_floor,
            max_condition_number=mass_cfg.max_condition_number,
            dense=mass_cfg.dense,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic path.
        return _rejected_result(
            status="mass_matrix_regularization_failed",
            initial_position=initial_np,
            locator_position=locator_position,
            locator_diagnostics=locator_diagnostics,
            geometry=geometry,
            mass_matrix=None,
            diagnostics={
                "classification": "diagnostic_initializer_rejected",
                "geometry_status": geometry.status,
                "mass_matrix_attempted": True,
                **transform_diagnostics,
                "mass_matrix_exception_type": type(exc).__name__,
                "mass_matrix_exception": str(exc),
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    map_candidate = np.asarray(geometry.center, dtype=float).copy()
    map_candidate_role = "exact_incumbent_geometry_center"
    precision = mass.regularized_precision
    if precision is None:
        return _rejected_result(
            status="mass_matrix_precision_missing",
            initial_position=initial_np,
            locator_position=locator_position,
            locator_diagnostics=locator_diagnostics,
            geometry=geometry,
            mass_matrix=mass,
            diagnostics={
                "classification": "diagnostic_initializer_rejected",
                "geometry_status": geometry.status,
                "mass_matrix_attempted": True,
                **transform_diagnostics,
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    return QuadraticMapCovarianceResult(
        accepted=True,
        status="usable",
        dimension=dim,
        initial_position=initial_np,
        locator_position=locator_position,
        map_candidate=map_candidate,
        map_candidate_role=map_candidate_role,
        precision=precision,
        covariance=mass.covariance,
        covariance_source=mass.source,
        locator_diagnostics=locator_diagnostics,
        geometry=geometry,
        mass_matrix=mass,
        diagnostics={
            "classification": "diagnostic_initializer_accepted",
            "precision_authority": (
                "low_rank_spd_quadratic_geometry_precision_transformed_to_theta"
            ),
            "covariance_authority": "covariance_from_precision",
            **transform_diagnostics,
            "optimizer_authority": "locator_only",
            "geometry_center_refinement_accepted": geometry.center_refinement_accepted,
            "map_candidate_role": map_candidate_role,
            "locator_config": locator_cfg.payload(),
            "quadratic_config": geometry_cfg.payload(),
            "mass_config": mass_cfg.payload(),
            "reports_map_quality": False,
            "reports_hmc_convergence": False,
            "reports_default_readiness": False,
        },
    )


def estimate_iterative_quadratic_map_covariance(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: Any,
    *,
    batched_value_and_score_fn: Callable[
        [tf.Tensor], tuple[tf.Tensor, tf.Tensor]
    ]
    | None = None,
    scale: Any | None = None,
    locator_config: QuadraticMapCovarianceLocatorConfig | None = None,
    quadratic_config: LowRankSPDQuadraticGeometryConfig | None = None,
    mass_config: QuadraticMapCovarianceMassConfig | None = None,
    iterative_config: IterativeQuadraticMapCovarianceConfig | None = None,
    fit_start_callback: Callable[[int, np.ndarray], None] | None = None,
    iteration_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> IterativeQuadraticMapCovarianceResult:
    """Iterate accepted local quadratic trust-region steps, then build mass.

    The exact target accepts every recentering move. A terminal covariance is
    built only from a fresh geometry fit centered at a point whose maximum
    absolute score in scaled coordinates passes ``terminal_score_max_abs``.
    Existing one-shot behavior remains in :func:`estimate_quadratic_map_covariance`.
    """

    locator_cfg = (
        QuadraticMapCovarianceLocatorConfig()
        if locator_config is None
        else locator_config
    )
    geometry_cfg = (
        LowRankSPDQuadraticGeometryConfig()
        if quadratic_config is None
        else quadratic_config
    )
    mass_cfg = QuadraticMapCovarianceMassConfig() if mass_config is None else mass_config
    iterative_cfg = (
        IterativeQuadraticMapCovarianceConfig()
        if iterative_config is None
        else iterative_config
    )
    if not geometry_cfg.constrain_center_refinement_to_trust_region:
        raise ValueError(
            "iterative quadratic recentering requires constrained center refinement"
        )

    initial_np = _vector(initial_position, "initial_position")
    dim = int(initial_np.size)
    scale_np = _scale_vector(scale, dim)
    initial_value, initial_score, initial_status = _evaluate_value_score(
        value_and_score_fn,
        initial_np,
        dim,
    )
    if initial_status != "finite":
        return _iterative_rejected_result(
            status="initial_value_or_score_nonfinite",
            initial_position=initial_np,
            locator_position=initial_np,
            locator_diagnostics={
                "status": "not_run_initial_nonfinite",
                "initial_log_prob": initial_value,
                "initial_evaluation_status": initial_status,
            },
            iterations=(),
            terminal_geometry=None,
            diagnostics={
                "classification": "iterative_diagnostic_initializer_rejected",
                "reports_map_quality": False,
                "reports_hmc_convergence": False,
                "reports_default_readiness": False,
            },
        )

    locator_position, locator_diagnostics = _run_locator(
        value_and_score_fn=value_and_score_fn,
        initial_position=initial_np,
        initial_value=initial_value,
        initial_score=initial_score,
        config=locator_cfg,
    )
    current = locator_position.copy()
    iteration_records: list[Mapping[str, Any]] = []
    terminal_geometry: LowRankSPDQuadraticGeometryResult | None = None

    for fit_index in range(iterative_cfg.max_refinement_steps + 1):
        center_value, center_score, center_status = _evaluate_value_score(
            value_and_score_fn,
            current,
            dim,
        )
        if center_status != "finite":
            return _iterative_rejected_result(
                status=f"iteration_{fit_index}_center_value_or_score_nonfinite",
                initial_position=initial_np,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=terminal_geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_np,
                ),
            )
        score_z = center_score * scale_np
        center_score_norm = float(np.linalg.norm(score_z))
        center_score_max_abs = float(np.max(np.abs(score_z)))
        terminal_before_fit = bool(
            center_score_max_abs <= iterative_cfg.terminal_score_max_abs
        )

        if fit_start_callback is not None:
            fit_start_callback(fit_index, current.copy())
        geometry = fit_low_rank_spd_quadratic_geometry(
            value_and_score_fn,
            current,
            batched_value_and_score_fn=batched_value_and_score_fn,
            scale=scale_np,
            config=geometry_cfg,
        )
        terminal_geometry = geometry
        refinement = dict(geometry.diagnostics.get("center_refinement", {}))
        record = {
            "fit_index": fit_index,
            "center": current,
            "center_log_prob": center_value,
            "center_score_norm": center_score_norm,
            "center_score_max_abs": center_score_max_abs,
            "terminal_score_gate": iterative_cfg.terminal_score_max_abs,
            "terminal_before_fit": terminal_before_fit,
            "geometry_accepted": geometry.accepted,
            "geometry_status": geometry.status,
            "center_refinement": refinement,
            "geometry_diagnostics": geometry.diagnostics,
        }
        iteration_records.append(record)
        if iteration_callback is not None:
            iteration_callback(_json_ready(record))

        if not geometry.accepted or geometry.precision is None:
            return _iterative_rejected_result(
                status=f"iteration_{fit_index}_geometry_{geometry.status}",
                initial_position=initial_np,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_np,
                ),
            )

        geometry_incumbent = (
            None
            if geometry.best_evaluated_position is None
            else np.asarray(geometry.best_evaluated_position, dtype=float).reshape([-1])
        )
        if geometry_incumbent is not None and _geometry_incumbent_move_is_material(
            geometry
        ):
            record["exact_incumbent_promoted"] = True
            record["exact_incumbent_source"] = geometry.best_evaluated_source
            record["exact_incumbent_value"] = geometry.best_evaluated_value
            if fit_index >= iterative_cfg.max_refinement_steps:
                return _iterative_rejected_result(
                    status="maximum_refinement_steps_after_exact_incumbent_move",
                    initial_position=initial_np,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics=_iterative_diagnostics(
                        iterative_cfg,
                        geometry_cfg,
                        mass_cfg,
                        scale_np,
                    ),
                )
            current = geometry_incumbent.copy()
            continue

        if terminal_before_fit:
            theta_precision = _precision_from_geometry_to_theta(geometry)
            try:
                mass = covariance_from_precision(
                    theta_precision,
                    source=(
                        "iterative_low_rank_spd_quadratic_geometry_"
                        "precision_theta_coordinates"
                    ),
                    jitter=mass_cfg.jitter,
                    eigenvalue_floor=mass_cfg.eigenvalue_floor,
                    max_condition_number=mass_cfg.max_condition_number,
                    dense=mass_cfg.dense,
                )
            except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic path.
                return _iterative_rejected_result(
                    status="mass_matrix_regularization_failed",
                    initial_position=initial_np,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics={
                        **_iterative_diagnostics(
                            iterative_cfg,
                            geometry_cfg,
                            mass_cfg,
                            scale_np,
                        ),
                        "mass_matrix_exception_type": type(exc).__name__,
                        "mass_matrix_exception": str(exc),
                    },
                )
            precision = mass.regularized_precision
            if precision is None:
                return _iterative_rejected_result(
                    status="mass_matrix_precision_missing",
                    initial_position=initial_np,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics=_iterative_diagnostics(
                        iterative_cfg,
                        geometry_cfg,
                        mass_cfg,
                        scale_np,
                    ),
                    mass_matrix=mass,
                )
            return IterativeQuadraticMapCovarianceResult(
                accepted=True,
                status="usable",
                dimension=dim,
                initial_position=initial_np,
                locator_position=locator_position,
                map_candidate=current,
                map_candidate_role="iterative_terminal_exact_score_center",
                precision=precision,
                covariance=mass.covariance,
                covariance_source=mass.source,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                mass_matrix=mass,
                diagnostics={
                    **_iterative_diagnostics(
                        iterative_cfg,
                        geometry_cfg,
                        mass_cfg,
                        scale_np,
                    ),
                    **_coordinate_transform_diagnostics(geometry),
                    "classification": "iterative_diagnostic_initializer_accepted",
                    "terminal_fit_index": fit_index,
                    "accepted_refinement_step_count": fit_index,
                    "terminal_score_max_abs": center_score_max_abs,
                    "precision_authority": (
                        "terminal_low_rank_spd_quadratic_geometry_"
                        "precision_transformed_to_theta"
                    ),
                    "covariance_authority": "covariance_from_precision",
                },
            )

        if fit_index >= iterative_cfg.max_refinement_steps:
            return _iterative_rejected_result(
                status="maximum_refinement_steps_without_terminal_score",
                initial_position=initial_np,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_np,
                ),
            )
        if not geometry.center_refinement_accepted or geometry.refined_center is None:
            return _iterative_rejected_result(
                status=f"iteration_{fit_index}_center_refinement_rejected",
                initial_position=initial_np,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_np,
                ),
            )
        current = np.asarray(geometry.refined_center, dtype=float).copy()

    raise RuntimeError("unreachable iterative quadratic initializer state")


def _run_locator(
    *,
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: np.ndarray,
    initial_value: float,
    initial_score: np.ndarray,
    config: QuadraticMapCovarianceLocatorConfig,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    dim = int(initial_position.size)
    initial_score_norm = float(np.linalg.norm(initial_score))
    base = {
        "schema": "bayesfilter.quadratic_map_covariance.locator.v1",
        "method": "tfp_lbfgs_minimize_negative_log_prob",
        "optimizer_role": "finite_neighborhood_locator_only",
        "uses_optimizer_inverse_hessian": False,
        "initial_log_prob": float(initial_value),
        "initial_score_norm": initial_score_norm,
        "config": config.payload(),
    }
    if not config.enabled:
        return initial_position.copy(), {
            **base,
            "status": "disabled_initial_position",
            "accepted_optimizer_position": False,
            "locator_log_prob": float(initial_value),
            "locator_score_norm": initial_score_norm,
        }

    try:
        locator = locate_joint_center(
            value_and_score_fn,
            initial_position,
            config=JointCenterLocatorConfig(
                max_iterations=int(config.max_iterations),
                gradient_tolerance=float(config.tolerance),
                parallel_iterations=int(config.parallel_iterations),
                jit_compile=False,
                max_objective_evaluations=max(
                    601, int(config.max_iterations) * 25 + 1
                ),
            ),
        )
        candidate = (
            initial_position.copy()
            if locator.best_evaluated_position is None
            else np.asarray(locator.best_evaluated_position, dtype=float).copy()
        )
        candidate_value = (
            float(initial_value)
            if locator.best_evaluated_objective is None
            else float(locator.best_evaluated_objective)
        )
        candidate_score = (
            np.asarray(initial_score, dtype=float)
            if locator.best_evaluated_score is None
            else np.asarray(locator.best_evaluated_score, dtype=float)
        )
        candidate_status = (
            "finite"
            if np.isfinite(candidate_value) and np.all(np.isfinite(candidate_score))
            else "nonfinite"
        )
        accepted = bool(
            candidate_status == "finite"
            and candidate_value >= initial_value - float(config.log_prob_tolerance)
            and locator.best_evaluated_source != "initial"
        )
        diagnostics = {
            **base,
            "status": (
                "tfp_lbfgs_locator_accepted"
                if accepted
                else "tfp_lbfgs_locator_rejected_initial_fallback"
            ),
            "accepted_optimizer_position": accepted,
            "optimizer_converged": locator.optimizer_converged,
            "optimizer_failed": locator.optimizer_failed,
            "optimizer_iterations": locator.optimizer_iterations,
            "optimizer_objective_value": -float(locator.endpoint_objective),
            "optimizer_endpoint_log_prob": float(locator.endpoint_objective),
            "best_evaluated_source": locator.best_evaluated_source,
            "best_evaluated_callback_index": locator.best_evaluated_callback_index,
            "joint_center_status": locator.status,
            "candidate_log_prob": candidate_value,
            "candidate_score_norm": (
                None
                if candidate_status != "finite"
                else float(np.linalg.norm(candidate_score))
            ),
            "candidate_evaluation_status": candidate_status,
            "locator_log_prob": (
                float(candidate_value) if accepted else float(initial_value)
            ),
            "locator_score_norm": (
                float(np.linalg.norm(candidate_score)) if accepted else initial_score_norm
            ),
            "fallback_reason": None if accepted else candidate_status,
        }
        return (candidate.copy() if accepted else initial_position.copy()), diagnostics
    except Exception as exc:  # noqa: BLE001 - fail-soft locator path.
        return initial_position.copy(), {
            **base,
            "status": "tfp_lbfgs_locator_exception_initial_fallback",
            "accepted_optimizer_position": False,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "locator_log_prob": float(initial_value),
            "locator_score_norm": initial_score_norm,
        }


def _rejected_result(
    *,
    status: str,
    initial_position: np.ndarray,
    locator_position: np.ndarray,
    locator_diagnostics: Mapping[str, Any],
    geometry: LowRankSPDQuadraticGeometryResult | None,
    mass_matrix: MassMatrixResult | None,
    diagnostics: Mapping[str, Any],
    map_candidate: np.ndarray | None = None,
    map_candidate_role: str = "none_rejected",
) -> QuadraticMapCovarianceResult:
    initial_np = np.asarray(initial_position, dtype=float).reshape([-1])
    locator_np = np.asarray(locator_position, dtype=float).reshape([-1])
    return QuadraticMapCovarianceResult(
        accepted=False,
        status=status,
        dimension=int(initial_np.size),
        initial_position=initial_np,
        locator_position=locator_np,
        map_candidate=map_candidate,
        map_candidate_role=map_candidate_role,
        precision=None,
        covariance=None,
        covariance_source=None,
        locator_diagnostics=locator_diagnostics,
        geometry=geometry,
        mass_matrix=mass_matrix,
        diagnostics={**dict(diagnostics), "rejection_status": status},
    )


def _iterative_rejected_result(
    *,
    status: str,
    initial_position: np.ndarray,
    locator_position: np.ndarray,
    locator_diagnostics: Mapping[str, Any],
    iterations: tuple[Mapping[str, Any], ...],
    terminal_geometry: LowRankSPDQuadraticGeometryResult | None,
    diagnostics: Mapping[str, Any],
    mass_matrix: MassMatrixResult | None = None,
) -> IterativeQuadraticMapCovarianceResult:
    return IterativeQuadraticMapCovarianceResult(
        accepted=False,
        status=status,
        dimension=int(np.asarray(initial_position).size),
        initial_position=initial_position,
        locator_position=locator_position,
        map_candidate=None,
        map_candidate_role="none_rejected",
        precision=None,
        covariance=None,
        covariance_source=None,
        locator_diagnostics=locator_diagnostics,
        iterations=iterations,
        terminal_geometry=terminal_geometry,
        mass_matrix=mass_matrix,
        diagnostics={
            **dict(diagnostics),
            "classification": "iterative_diagnostic_initializer_rejected",
            "rejection_status": status,
            "reports_map_quality": False,
            "reports_hmc_convergence": False,
            "reports_default_readiness": False,
        },
    )


def _iterative_diagnostics(
    iterative_config: IterativeQuadraticMapCovarianceConfig,
    quadratic_config: LowRankSPDQuadraticGeometryConfig,
    mass_config: QuadraticMapCovarianceMassConfig,
    scale: np.ndarray,
) -> Mapping[str, Any]:
    return {
        "iterative_config": iterative_config.payload(),
        "quadratic_config": quadratic_config.payload(),
        "mass_config": mass_config.payload(),
        "score_coordinate_system": "scale_times_theta_score",
        "scale": scale,
        "optimizer_authority": "locator_only",
        "reports_map_quality": False,
        "reports_hmc_convergence": False,
        "reports_default_readiness": False,
    }


def _evaluate_value_score(
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    theta: np.ndarray,
    dim: int,
) -> tuple[float, np.ndarray, str]:
    theta_np = np.asarray(theta, dtype=float).reshape([-1])
    if theta_np.shape != (int(dim),) or not np.all(np.isfinite(theta_np)):
        return float("nan"), np.full(int(dim), np.nan, dtype=float), "position_nonfinite"
    try:
        value, score = value_and_score_fn(tf.constant(theta_np, dtype=tf.float64))
        value_np = float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())
        score_np = np.asarray(
            tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1]).numpy(),
            dtype=float,
        )
    except Exception:  # noqa: BLE001 - diagnostic rejection path.
        return float("nan"), np.full(int(dim), np.nan, dtype=float), "exception"
    if score_np.shape != (int(dim),):
        return value_np, np.full(int(dim), np.nan, dtype=float), "score_shape_mismatch"
    if not np.isfinite(value_np) or not np.all(np.isfinite(score_np)):
        return value_np, score_np, "nonfinite"
    return value_np, score_np, "finite"


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
        raise ValueError("scale must be scalar or match initial position dimension")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scale must be positive finite")
    return scale


def _precision_from_geometry_to_theta(
    geometry: LowRankSPDQuadraticGeometryResult,
) -> np.ndarray:
    if geometry.precision is None:
        raise ValueError("geometry precision is required")
    precision_z = np.asarray(geometry.precision, dtype=float)
    scale = np.asarray(geometry.scale, dtype=float).reshape([-1])
    if precision_z.shape != (scale.size, scale.size):
        raise ValueError("geometry precision shape must match geometry scale")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("geometry scale must be positive finite")
    inverse_scale = 1.0 / scale
    precision_theta = (
        inverse_scale[:, np.newaxis]
        * precision_z
        * inverse_scale[np.newaxis, :]
    )
    return 0.5 * (precision_theta + precision_theta.T)


def _geometry_incumbent_move_is_material(
    geometry: LowRankSPDQuadraticGeometryResult,
) -> bool:
    """Apply the geometry's frozen objective tolerance to centeredness.

    Exact incumbent provenance is retained even for roundoff-sized
    improvements. Curvature is invalidated only when the selected point moves
    by more than numerical noise and improves log probability beyond the
    pre-existing center tolerance.
    """

    if geometry.best_evaluated_position is None or geometry.best_evaluated_value is None:
        return False
    center = np.asarray(geometry.center, dtype=float)
    best = np.asarray(geometry.best_evaluated_position, dtype=float)
    scale = np.asarray(geometry.scale, dtype=float)
    z_norm = float(np.linalg.norm((best - center) / scale))
    center_value = float(geometry.diagnostics.get("center_log_prob", np.nan))
    tolerance = float(
        geometry.diagnostics.get("config", {}).get(
            "center_log_prob_tolerance", 1.0e-8
        )
    )
    improvement = float(geometry.best_evaluated_value) - center_value
    return bool(
        np.isfinite(improvement)
        and improvement > tolerance
        and z_norm > 1.0e-10
    )


def _coordinate_transform_diagnostics(
    geometry: LowRankSPDQuadraticGeometryResult,
) -> Mapping[str, Any]:
    scale = np.asarray(geometry.scale, dtype=float).reshape([-1])
    return {
        "geometry_fit_coordinate_system": "whitened_z",
        "geometry_coordinate_transform": "theta = center + scale * z",
        "geometry_precision_coordinate_system": "z",
        "mass_precision_coordinate_system": "theta",
        "mass_covariance_coordinate_system": "theta",
        "precision_transform": "P_theta = diag(1 / scale) @ P_z @ diag(1 / scale)",
        "covariance_transform": "C_theta = diag(scale) @ C_z @ diag(scale)",
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
        "scale_all_ones": bool(np.allclose(scale, np.ones_like(scale))),
    }


def _mass_matrix_payload(
    mass: MassMatrixResult,
    *,
    include_arrays: bool,
) -> Mapping[str, Any]:
    payload: dict[str, Any] = {
        "source": mass.source,
        "matrix_kind": mass.matrix_kind,
        "jitter": mass.jitter,
        "eigenvalue_floor": mass.eigenvalue_floor,
        "precision_eigen_summary": mass.precision_eigen_summary,
        "covariance_eigen_summary": mass.covariance_eigen_summary,
        "regularization_report": mass.regularization_report,
    }
    if include_arrays:
        payload.update(
            {
                "covariance": mass.covariance,
                "regularized_precision": mass.regularized_precision,
            }
        )
    return _json_ready(payload)


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


def _tensor_bool(value: Any) -> bool:
    return bool(tf.convert_to_tensor(value).numpy())


def _tensor_int(value: Any) -> int:
    return int(tf.convert_to_tensor(value).numpy())


def _tensor_float(value: Any) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return tuple(_json_ready(item) for item in value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)
