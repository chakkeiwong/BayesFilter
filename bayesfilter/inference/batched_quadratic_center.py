"""Bounded TF quadratic localization and a fresh position-pilot factor.

The target supplies raw analytical scores. In coordinates theta=c+diag(s)z,
fit b-g_z(z) ~= Kz, then maximize b'p-p'Kp/2 in a Euclidean trust ball.
The final factor satisfies LL'=diag(s) K^-1 diag(s); it is not momentum mass.
An exact incumbent change invalidates the fitted anchor. No MAP, posterior
covariance, whitening, HMC, GPU or whole-initializer XLA claim follows.

Uniform-cloud and paired-local refinement execute as one stable TF/XLA program
with a native round loop. The host validates inputs and formats completed
histories. Composition with the multistart locator is a separate execution
boundary; whole-initializer XLA is unproved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_local_center import (
    BatchedLocalCenterConfig,
    locate_batched_local_center,
)
from bayesfilter.inference.mass_matrix_tf import eigenpair_program
from bayesfilter.inference.paired_score_pilot_tf import (
    validate_paired_steps,
)
from bayesfilter.inference.quadratic_round_report import paired_quadratic_result
from bayesfilter.inference.quadratic_rounds_tf import quadratic_controller
from bayesfilter.ops.compiled_tensor_program_tf import in_xla_context


def _norm(vector):
    scale = tf.reduce_max(tf.abs(vector))
    return scale * tf.linalg.norm(tf.math.divide_no_nan(vector, scale))


def solve_spd_quadratic_trust_region_tf(precision, linear, radius):
    """Solve the SPD trust subproblem by its monotone KKT secular equation.

    Inputs are float64 K[D,D], b[D], Delta[]. Return tensor-valued validity,
    p, multiplier, boundary flag and predicted log-density improvement. Raw
    indefinite/nonfinite inputs are invalid; safe internal operands only avoid
    evaluating undefined linear algebra and never turn rejection into evidence.
    The finite bisection bound replaces unbounded multiplier-bracketing loops.
    """
    matrix = tf.convert_to_tensor(precision, tf.float64)
    vector = tf.convert_to_tensor(linear, tf.float64)
    radius = tf.convert_to_tensor(radius, tf.float64)
    dimension = vector.shape[0]
    if matrix.shape != (dimension, dimension) or radius.shape.rank != 0:
        raise ValueError("expected precision[D,D], linear[D], radius[]")
    finite = tf.reduce_all(tf.math.is_finite(matrix)) & tf.reduce_all(tf.math.is_finite(vector))
    finite &= tf.math.is_finite(radius) & (radius > 0)
    symmetric = matrix * 0.5 + tf.transpose(matrix) * 0.5
    scale = tf.reduce_max(tf.abs(symmetric))
    finite &= tf.math.is_finite(scale) & (scale > 0)
    safe_scale = tf.where(finite, scale, 1.0)
    safe_matrix = tf.where(finite, symmetric / safe_scale, tf.eye(dimension, dtype=tf.float64))
    eigenvalues, eigenvectors = (eigenpair_program(dimension)(safe_matrix)
                                if in_xla_context() else tf.linalg.eigh(safe_matrix))
    valid = finite & tf.reduce_all(eigenvalues > 0) & tf.reduce_all(tf.math.is_finite(eigenvalues))
    safe_eigenvalues = tf.where(valid, eigenvalues, tf.ones_like(eigenvalues))
    projected = tf.linalg.matvec(eigenvectors, tf.where(valid, vector / safe_scale, 0.0), transpose_a=True)
    bound = tf.where(valid, radius, 1.0)

    def solution(multiplier):
        return tf.linalg.matvec(eigenvectors, projected / (safe_eigenvalues + multiplier))

    newton = solution(tf.constant(0.0, tf.float64))
    boundary = _norm(newton) > bound
    upper = _norm(projected) / bound

    def bisect(count, lower, upper):
        midpoint = lower * 0.5 + upper * 0.5
        outside = _norm(solution(midpoint)) > bound
        return count + 1, tf.where(outside, midpoint, lower), tf.where(outside, upper, midpoint)

    _, _, upper = tf.while_loop(
        lambda count, lower, upper: (count < 100) & boundary,
        bisect, (tf.constant(0), tf.constant(0.0, tf.float64), upper),
        parallel_iterations=1,
    )
    multiplier = tf.where(boundary, upper, 0.0)
    step = solution(multiplier)
    predicted = tf.tensordot(vector, step, 1) - 0.5 * tf.tensordot(step, tf.linalg.matvec(matrix, step), 1)
    valid &= tf.reduce_all(tf.math.is_finite(step)) & tf.math.is_finite(predicted)
    valid &= tf.math.is_finite(multiplier * safe_scale) & (_norm(step) <= bound * (1 + 1e-10))
    return {"valid": valid, "step": step, "multiplier": multiplier * safe_scale,
            "boundary_active": boundary, "predicted_improvement": predicted}


@dataclass(frozen=True)
class BatchedQuadraticCenterConfig:
    """Localization probes and trust radius do not set the output factor width."""

    batch_size: int = 4
    rows_per_cloud: int | None = None
    max_fit_rounds: int = 8
    max_physical_rows: int = 100000
    seed: int = 20260910
    fit_half_width: float = 0.1
    initial_trust_radius: float = 1.0
    maximum_trust_radius: float = 4.0
    centeredness_cap: float = 0.5
    model_relative_rmse_cap: float = 0.20
    precision_condition_cap: float = 1e10
    replay_atol: float = 1e-10
    replay_rtol: float = 1e-10
    jit_compile_trust: bool = True
    pilot_method: str = "uniform_cloud"
    paired_steps: tuple[float, float] = (0.001, 0.0001)

    def __post_init__(self):
        if self.pilot_method not in {"uniform_cloud", "paired_local"}:
            raise ValueError("unknown pilot_method")
        validate_paired_steps(self.paired_steps)
        if self.pilot_method == "paired_local" and self.rows_per_cloud is not None:
            raise ValueError("paired_local fixes each design to 2D rows")
        if self.batch_size < 2 or self.max_fit_rounds < 1 or self.max_physical_rows < 1:
            raise ValueError("positive limits and batch_size>=2 required")
        if self.rows_per_cloud is not None and self.rows_per_cloud < 1:
            raise ValueError("rows_per_cloud must be positive")
        for name in ("fit_half_width", "initial_trust_radius", "maximum_trust_radius",
                     "centeredness_cap", "model_relative_rmse_cap", "precision_condition_cap",
                     "replay_atol", "replay_rtol"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.initial_trust_radius > self.maximum_trust_radius or self.precision_condition_cap <= 1:
            raise ValueError("inconsistent trust radius or precision condition bound")

    def planned_rows(self, dimension):
        if self.pilot_method == "paired_local":
            padded = self.batch_size * ((2 * dimension + self.batch_size - 1) // self.batch_size)
            return 2 * self.batch_size + self.max_fit_rounds * (3 * padded + 2 * self.batch_size)
        rows = max(32, 4 * dimension) if self.rows_per_cloud is None else self.rows_per_cloud
        if rows < dimension:
            raise ValueError("cloud must have at least dimension rows")
        padded = self.batch_size * ((rows + self.batch_size - 1) // self.batch_size)
        return 2 * self.batch_size + self.max_fit_rounds * (2 * padded + 2 * self.batch_size)


@dataclass(frozen=True)
class BatchedQuadraticCenterResult:
    """Rejected results expose the exact incumbent, but no usable geometry."""

    accepted: bool
    status: str
    center: tf.Tensor
    center_value: tf.Tensor
    center_score: tf.Tensor
    pilot_factor: tf.Tensor | None
    precision_z: tf.Tensor | None
    diagnostics: dict[str, Any]

    def payload(self):
        def ready(value):
            if tf.is_tensor(value):
                return ready(value.numpy().tolist())
            if isinstance(value, dict):
                return {key: ready(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [ready(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return {"schema": "bayesfilter.batched_quadratic_center.v1",
                **ready({field.name: getattr(self, field.name) for field in fields(self)}),
                "nonclaims": ["local center and position pilot only", "not a MAP or posterior covariance",
                              "not whitening, HMC, GPU or full-initializer XLA readiness"]}


def refine_batched_quadratic_center(callback, center, scale, *, config=None, _initial_evidence=None):
    """Refine a fresh exact center using fixed [B,D] analytical target batches.

    Every cloud/proposal enters exact selection, including a better point from
    a poorly predicted move. A fresh model at the selected point is mandatory
    before factor handoff. Model-check clouds belong to localization; the later
    fixed-center regional curvature/audit must use independent streams.
    """
    if not tf.executing_eagerly():
        raise RuntimeError("initializer validation and reporting require eager execution")
    cfg = BatchedQuadraticCenterConfig() if config is None else config
    center = tf.convert_to_tensor(center, tf.float64)
    scale = tf.convert_to_tensor(scale, tf.float64)
    if center.shape.rank != 1 or not center.shape.is_fully_defined() or scale.shape != center.shape:
        raise ValueError("center and scale must have static shape [D]")
    dimension = int(center.shape[0])
    if dimension < 1:
        raise ValueError("dimension must be positive")
    tf.debugging.assert_all_finite(center, "center must be finite")
    tf.debugging.assert_all_finite(scale, "scale must be finite")
    tf.debugging.assert_positive(scale)
    planned = cfg.planned_rows(dimension)
    if planned > cfg.max_physical_rows:
        raise ValueError("whole refinement exceeds max_physical_rows before target calls")
    # The explicit non-JIT option selects the complete graph reference. Each
    # family shares the same ordered TensorFlow recurrence and result schema.
    program = quadratic_controller(callback, dimension, cfg, jit_compile=cfg.jit_compile_trust)
    value, score = ((tf.constant(0., tf.float64), tf.zeros_like(center))
                    if _initial_evidence is None else _initial_evidence)
    computed = program(center, scale, tf.convert_to_tensor(cfg.seed, tf.int32),
        tf.constant(_initial_evidence is not None), tf.convert_to_tensor(value, tf.float64),
        tf.convert_to_tensor(score, tf.float64))
    return paired_quadratic_result(computed, cfg, dimension, jit_compile=cfg.jit_compile_trust,
        trace_count=program.experimental_get_tracing_count())


def initialize_batched_posterior_local_location_scale(callback, initial_positions, scale, *, config=None, locator_config=None):
    """Compose independent bounded TFP starts with actual quadratic refinement.

    Reserve the entire procedure before its first target request. A rejected
    localizer, including its explicit cap veto, cannot enter refinement. Local
    geometry is never handed off with the pre-refinement anchor's identity.
    """
    cfg = BatchedQuadraticCenterConfig() if config is None else config
    locator_cfg = BatchedLocalCenterConfig(trust_refinement_rounds=1) if locator_config is None else locator_config
    positions = tf.convert_to_tensor(initial_positions, tf.float64)
    if positions.shape.rank != 2 or positions.shape[0] != cfg.batch_size:
        raise ValueError("initial_positions must match the configured independent batch")
    planned = cfg.batch_size * locator_cfg.maximum_physical_rows_multiplier + cfg.planned_rows(int(positions.shape[1]))
    if planned > cfg.max_physical_rows:
        raise ValueError("whole initializer exceeds max_physical_rows before target calls")
    located = locate_batched_local_center(callback, positions, scale, config=locator_cfg)
    used = int(located.physical_target_rows)
    if not bool(located.accepted):
        return BatchedQuadraticCenterResult(False, "localizer_" + located.status, located.center,
                                           located.center_value, located.center_score, None, None,
                                           {"localizer": located.payload(), "physical_rows": used,
                                            "planned_physical_rows": planned})
    refined = refine_batched_quadratic_center(callback, located.center, scale,
                                              config=replace(cfg, max_physical_rows=cfg.max_physical_rows-used),
                                              _initial_evidence=(located.center_value, located.center_score))
    details = dict(refined.diagnostics)
    details["refinement_physical_rows"] = details["physical_rows"]
    details["physical_rows"] += used
    details["refinement_callback_batches"] = details["callback_batches"]
    details["callback_batches"] += int(located.target_callback_batches)
    details["planned_physical_rows"] = planned
    selected_index = details["selected_evaluation_index"]
    details["selected_evaluation_index"] = int(located.selected_evaluation_index) if selected_index == 0 else used + selected_index
    details["localizer"] = located.payload()
    return replace(refined, diagnostics=details)
