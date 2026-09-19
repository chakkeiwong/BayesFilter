"""Quadratic MAP-candidate covariance initializer.

This module provides a diagnostic initializer for later HMC tuning.  It uses a
local optimizer only to locate a finite neighborhood, then treats the
constrained SPD quadratic geometry as the covariance/precision authority.

It does not certify a global MAP, posterior covariance correctness, HMC
readiness, sampler convergence, or source-faithful Zhao-Cui behavior.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp  # noqa: F401 - retained diagnostic optimizer patch point
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_self_adjoint_eig

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
from bayesfilter.ops.host_tensor_io import numeric_tensor

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
            if not math.isfinite(value) or value < 0.0:
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
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("jitter must be finite and non-negative")
        object.__setattr__(self, "jitter", jitter)
        if self.eigenvalue_floor is not None:
            floor = float(self.eigenvalue_floor)
            if not math.isfinite(floor) or floor < 0.0:
                raise ValueError("eigenvalue_floor must be finite and non-negative")
            object.__setattr__(self, "eigenvalue_floor", floor)
        if self.max_condition_number is not None:
            condition = float(self.max_condition_number)
            if not math.isfinite(condition) or condition <= 1.0:
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
        if not math.isfinite(terminal) or terminal <= 0.0:
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
    initial_position: tf.Tensor
    locator_position: tf.Tensor
    map_candidate: tf.Tensor | None
    map_candidate_role: str
    precision: tf.Tensor | None
    covariance: tf.Tensor | None
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
            array = tf.identity(tf.reshape(numeric_tensor(getattr(self, name), tf.float64), [-1]))
            object.__setattr__(self, name, array)
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = tf.identity(numeric_tensor(value, tf.float64))
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
    initial_position: tf.Tensor
    locator_position: tf.Tensor
    map_candidate: tf.Tensor | None
    map_candidate_role: str
    precision: tf.Tensor | None
    covariance: tf.Tensor | None
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
            array = tf.identity(tf.reshape(numeric_tensor(getattr(self, name), tf.float64), [-1]))
            object.__setattr__(self, name, array)
        for name in ("map_candidate", "precision", "covariance"):
            value = getattr(self, name)
            if value is not None:
                array = tf.identity(numeric_tensor(value, tf.float64))
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

    initial_tensor = _vector(initial_position, "initial_position")
    dim = int(initial_tensor.shape[0])
    initial_value, initial_score, initial_status = _evaluate_value_score(
        value_and_score_fn,
        initial_tensor,
        dim,
    )
    if initial_status != "finite":
        return _rejected_result(
            status="initial_value_or_score_nonfinite",
            initial_position=initial_tensor,
            locator_position=initial_tensor,
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
        initial_position=initial_tensor,
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
            initial_position=initial_tensor,
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
        else tf.reshape(numeric_tensor(geometry.best_evaluated_position, tf.float64), [-1])
    )
    if geometry_incumbent is not None and _geometry_incumbent_move_is_material(
        geometry
    ):
        return _rejected_result(
            status="covariance_center_mismatch_requires_refit",
            initial_position=initial_tensor,
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
            initial_position=initial_tensor,
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

    map_candidate = tf.identity(numeric_tensor(geometry.center, tf.float64))
    map_candidate_role = "exact_incumbent_geometry_center"
    precision = mass.regularized_precision
    if precision is None:
        return _rejected_result(
            status="mass_matrix_precision_missing",
            initial_position=initial_tensor,
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
        initial_position=initial_tensor,
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
    fit_start_callback: Callable[[int, tf.Tensor], None] | None = None,
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

    initial_tensor = _vector(initial_position, "initial_position")
    dim = int(initial_tensor.shape[0])
    scale_tensor = _scale_vector(scale, dim)
    initial_value, initial_score, initial_status = _evaluate_value_score(
        value_and_score_fn,
        initial_tensor,
        dim,
    )
    if initial_status != "finite":
        return _iterative_rejected_result(
            status="initial_value_or_score_nonfinite",
            initial_position=initial_tensor,
            locator_position=initial_tensor,
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
        initial_position=initial_tensor,
        initial_value=initial_value,
        initial_score=initial_score,
        config=locator_cfg,
    )
    current = tf.identity(locator_position)
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
                initial_position=initial_tensor,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=terminal_geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_tensor,
                ),
            )
        score_norm, score_max = _run_numerical(_scaled_score_summary, center_score, scale_tensor)
        center_score_norm = float(score_norm)
        center_score_max_abs = float(score_max)
        terminal_before_fit = bool(
            center_score_max_abs <= iterative_cfg.terminal_score_max_abs
        )

        if fit_start_callback is not None:
            fit_start_callback(fit_index, tf.identity(current))
        geometry = fit_low_rank_spd_quadratic_geometry(
            value_and_score_fn,
            current,
            batched_value_and_score_fn=batched_value_and_score_fn,
            scale=scale_tensor,
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
                initial_position=initial_tensor,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_tensor,
                ),
            )

        geometry_incumbent = (
            None
            if geometry.best_evaluated_position is None
            else tf.reshape(numeric_tensor(geometry.best_evaluated_position, tf.float64), [-1])
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
                    initial_position=initial_tensor,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics=_iterative_diagnostics(
                        iterative_cfg,
                        geometry_cfg,
                        mass_cfg,
                        scale_tensor,
                    ),
                )
            current = tf.identity(geometry_incumbent)
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
                    initial_position=initial_tensor,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics={
                        **_iterative_diagnostics(
                            iterative_cfg,
                            geometry_cfg,
                            mass_cfg,
                            scale_tensor,
                        ),
                        "mass_matrix_exception_type": type(exc).__name__,
                        "mass_matrix_exception": str(exc),
                    },
                )
            precision = mass.regularized_precision
            if precision is None:
                return _iterative_rejected_result(
                    status="mass_matrix_precision_missing",
                    initial_position=initial_tensor,
                    locator_position=locator_position,
                    locator_diagnostics=locator_diagnostics,
                    iterations=tuple(iteration_records),
                    terminal_geometry=geometry,
                    diagnostics=_iterative_diagnostics(
                        iterative_cfg,
                        geometry_cfg,
                        mass_cfg,
                        scale_tensor,
                    ),
                    mass_matrix=mass,
                )
            return IterativeQuadraticMapCovarianceResult(
                accepted=True,
                status="usable",
                dimension=dim,
                initial_position=initial_tensor,
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
                        scale_tensor,
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
                initial_position=initial_tensor,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_tensor,
                ),
            )
        if not geometry.center_refinement_accepted or geometry.refined_center is None:
            return _iterative_rejected_result(
                status=f"iteration_{fit_index}_center_refinement_rejected",
                initial_position=initial_tensor,
                locator_position=locator_position,
                locator_diagnostics=locator_diagnostics,
                iterations=tuple(iteration_records),
                terminal_geometry=geometry,
                diagnostics=_iterative_diagnostics(
                    iterative_cfg,
                    geometry_cfg,
                    mass_cfg,
                    scale_tensor,
                ),
            )
        current = tf.identity(numeric_tensor(geometry.refined_center, tf.float64))

    raise RuntimeError("unreachable iterative quadratic initializer state")


def _run_locator(
    *,
    value_and_score_fn: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_position: tf.Tensor,
    initial_value: float,
    initial_score: tf.Tensor,
    config: QuadraticMapCovarianceLocatorConfig,
) -> tuple[tf.Tensor, Mapping[str, Any]]:
    initial_score_norm = float(_run_numerical(_norm, initial_score))
    base = {
        "schema": "bayesfilter.quadratic_map_covariance.locator.v1",
        "method": "tfp_lbfgs_minimize_negative_log_prob",
        "optimizer_role": "finite_neighborhood_locator_only",
        "uses_optimizer_inverse_hessian": False,
        "jit_compile": True,
        "initial_log_prob": float(initial_value),
        "initial_score_norm": initial_score_norm,
        "config": config.payload(),
    }
    if not config.enabled:
        return tf.identity(initial_position), {
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
                jit_compile=True,
                max_objective_evaluations=max(
                    601, int(config.max_iterations) * 25 + 1
                ),
            ),
        )
        candidate = (
            tf.identity(initial_position)
            if locator.best_evaluated_position is None
            else tf.identity(numeric_tensor(locator.best_evaluated_position, tf.float64))
        )
        candidate_value = (
            float(initial_value)
            if locator.best_evaluated_objective is None
            else float(locator.best_evaluated_objective)
        )
        candidate_score = (
            numeric_tensor(initial_score, tf.float64)
            if locator.best_evaluated_score is None
            else numeric_tensor(locator.best_evaluated_score, tf.float64)
        )
        candidate_status = (
            "finite"
            if math.isfinite(candidate_value) and bool(tf.reduce_all(tf.math.is_finite(candidate_score)))
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
                else float(_run_numerical(_norm, candidate_score))
            ),
            "candidate_evaluation_status": candidate_status,
            "locator_log_prob": (
                float(candidate_value) if accepted else float(initial_value)
            ),
            "locator_score_norm": (
                float(_run_numerical(_norm, candidate_score)) if accepted else initial_score_norm
            ),
            "fallback_reason": None if accepted else candidate_status,
        }
        return (tf.identity(candidate) if accepted else tf.identity(initial_position)), diagnostics
    except Exception as exc:  # noqa: BLE001 - fail-soft locator path.
        return tf.identity(initial_position), {
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
    initial_position: tf.Tensor,
    locator_position: tf.Tensor,
    locator_diagnostics: Mapping[str, Any],
    geometry: LowRankSPDQuadraticGeometryResult | None,
    mass_matrix: MassMatrixResult | None,
    diagnostics: Mapping[str, Any],
    map_candidate: tf.Tensor | None = None,
    map_candidate_role: str = "none_rejected",
) -> QuadraticMapCovarianceResult:
    initial_tensor = tf.reshape(numeric_tensor(initial_position, tf.float64), [-1])
    locator_tensor = tf.reshape(numeric_tensor(locator_position, tf.float64), [-1])
    return QuadraticMapCovarianceResult(
        accepted=False,
        status=status,
        dimension=int(initial_tensor.shape[0]),
        initial_position=initial_tensor,
        locator_position=locator_tensor,
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
    initial_position: tf.Tensor,
    locator_position: tf.Tensor,
    locator_diagnostics: Mapping[str, Any],
    iterations: tuple[Mapping[str, Any], ...],
    terminal_geometry: LowRankSPDQuadraticGeometryResult | None,
    diagnostics: Mapping[str, Any],
    mass_matrix: MassMatrixResult | None = None,
) -> IterativeQuadraticMapCovarianceResult:
    return IterativeQuadraticMapCovarianceResult(
        accepted=False,
        status=status,
        dimension=int(tf.size(initial_position)),
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
    scale: tf.Tensor,
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
    theta: tf.Tensor,
    dim: int,
) -> tuple[float, tf.Tensor, str]:
    point = tf.reshape(numeric_tensor(theta, tf.float64), [-1])
    missing = tf.fill([dim], tf.constant(math.nan, tf.float64))
    if point.shape != (int(dim),) or not bool(tf.reduce_all(tf.math.is_finite(point))):
        return math.nan, missing, "position_nonfinite"
    try:
        value, score = _run_numerical(value_and_score_fn, point)
        value = float(numeric_tensor(value, tf.float64))
        score = tf.reshape(numeric_tensor(score, tf.float64), [-1])
    except Exception:  # noqa: BLE001 - diagnostic rejection path.
        return math.nan, missing, "exception"
    if score.shape != (int(dim),):
        return value, missing, "score_shape_mismatch"
    if not math.isfinite(value) or not bool(tf.reduce_all(tf.math.is_finite(score))):
        return value, score, "nonfinite"
    return value, score, "finite"


def _vector(value: Any, name: str) -> tf.Tensor:
    vector = tf.reshape(numeric_tensor(value, tf.float64), [-1])
    if vector.shape[0] is None or vector.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not bool(tf.reduce_all(tf.math.is_finite(vector))):
        raise ValueError(f"{name} must be finite")
    return vector


def _scale_vector(value: Any | None, dim: int) -> tf.Tensor:
    if value is None:
        scale = tf.ones([dim], tf.float64)
    else:
        scale = tf.reshape(numeric_tensor(value, tf.float64), [-1])
        if scale.shape[0] == 1:
            scale = tf.broadcast_to(scale, [dim])
    if scale.shape != (int(dim),):
        raise ValueError("scale must be scalar or match initial position dimension")
    if not bool(tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.0))):
        raise ValueError("scale must be positive finite")
    return scale


def _precision_from_geometry_to_theta(
    geometry: LowRankSPDQuadraticGeometryResult,
) -> tf.Tensor:
    if geometry.precision is None:
        raise ValueError("geometry precision is required")
    precision_z = numeric_tensor(geometry.precision, tf.float64)
    scale = tf.reshape(numeric_tensor(geometry.scale, tf.float64), [-1])
    if precision_z.shape != (scale.shape[0], scale.shape[0]):
        raise ValueError("geometry precision shape must match geometry scale")
    if not bool(tf.reduce_all(tf.math.is_finite(scale) & (scale > 0.0))):
        raise ValueError("geometry scale must be positive finite")
    return _run_numerical(_precision_transform, precision_z, scale)


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
    center = numeric_tensor(geometry.center, tf.float64)
    best = numeric_tensor(geometry.best_evaluated_position, tf.float64)
    scale = numeric_tensor(geometry.scale, tf.float64)
    center_value = float(geometry.diagnostics.get("center_log_prob", math.nan))
    tolerance = float(
        geometry.diagnostics.get("config", {}).get(
            "center_log_prob_tolerance", 1.0e-8
        )
    )
    return bool(_run_numerical(_material_move, best, center, scale,
        tf.constant(geometry.best_evaluated_value, tf.float64),
        tf.constant(center_value, tf.float64), tf.constant(tolerance, tf.float64)))


def _coordinate_transform_diagnostics(
    geometry: LowRankSPDQuadraticGeometryResult,
) -> Mapping[str, Any]:
    scale = tf.reshape(numeric_tensor(geometry.scale, tf.float64), [-1])
    minimum, maximum, all_ones = _run_numerical(_scale_summary, scale)
    return {
        "geometry_fit_coordinate_system": "whitened_z",
        "geometry_coordinate_transform": "theta = center + scale * z",
        "geometry_precision_coordinate_system": "z",
        "mass_precision_coordinate_system": "theta",
        "mass_covariance_coordinate_system": "theta",
        "precision_transform": "P_theta = diag(1 / scale) @ P_z @ diag(1 / scale)",
        "covariance_transform": "C_theta = diag(scale) @ C_z @ diag(scale)",
        "scale_min": float(minimum),
        "scale_max": float(maximum),
        "scale_all_ones": bool(all_ones),
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
    eigvals, valid, minimum, maximum, condition = _run_numerical(
        _eigen_statistics, numeric_tensor(matrix, tf.float64))
    finite = bool(valid)
    positive = bool(finite and float(minimum) > 0.0)
    return {
        "finite": finite,
        "positive": positive,
        "min": float(minimum) if finite else float("nan"),
        "max": float(maximum) if finite else float("nan"),
        "condition_number": (
            float(condition) if positive else float("inf")
        ),
        "eigenvalues": tuple(eigvals.numpy().tolist()),
    }


def _tensor_bool(value: Any) -> bool:
    return bool(tf.convert_to_tensor(value).numpy())


def _tensor_int(value: Any) -> int:
    return int(tf.convert_to_tensor(value).numpy())


def _tensor_float(value: Any) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


@lru_cache(maxsize=64)
def _numerical_program(kernel, signature):
    """Bound callback/shape variants; numerical calls never fall back to Python."""
    return tf.function(kernel, input_signature=signature, jit_compile=True, autograph=False)


def _run_numerical(kernel, *inputs):
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in inputs)
    return _numerical_program(kernel, signature)(*inputs)


def _norm(vector):
    return tf.linalg.norm(vector)


def _scaled_score_summary(score, scale):
    scaled = score * scale
    return tf.linalg.norm(scaled), tf.reduce_max(tf.abs(scaled))


def _precision_transform(precision_z, scale):
    inverse_scale = 1.0 / scale
    precision_theta = inverse_scale[:, None] * precision_z * inverse_scale[None, :]
    return 0.5 * (precision_theta + tf.transpose(precision_theta))


def _material_move(best, center, scale, best_value, center_value, tolerance):
    z_norm = tf.linalg.norm((best - center) / scale)
    improvement = best_value - center_value
    return tf.math.is_finite(improvement) & (improvement > tolerance) & (z_norm > 1.0e-10)


def _scale_summary(scale):
    # Preserve NumPy allclose's existing defaults against an all-ones reference.
    all_ones = tf.reduce_all(tf.abs(scale - 1.0) <= tf.constant(1e-8 + 1e-5, tf.float64))
    return tf.reduce_min(scale), tf.reduce_max(scale), all_ones


def _eigen_statistics(square):
    # The default XLA eigensolver epsilon discards small off-diagonal entries
    # in a badly scaled binary64 matrix. Match the existing geometry kernels.
    eigvals, _ = xla_self_adjoint_eig(0.5 * (square + tf.transpose(square)),
        lower=True, max_iter=100, epsilon=math.ulp(1.0))
    minimum, maximum = tf.reduce_min(eigvals), tf.reduce_max(eigvals)
    return eigvals, tf.reduce_all(tf.math.is_finite(eigvals)), minimum, maximum, maximum / minimum


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if tf.is_tensor(value) or hasattr(value, "__array_interface__"):
        materialized = numeric_tensor(value).numpy().tolist()
        if isinstance(materialized, list):
            return tuple(_json_ready(item) for item in materialized)
        return materialized
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)
