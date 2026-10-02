"""Bounded TF quadratic localization and a fresh position-pilot factor.

The target supplies raw analytical scores. In coordinates theta=c+diag(s)z,
fit b-g_z(z) ~= Kz, then maximize b'p-p'Kp/2 in a Euclidean trust ball.
The final factor satisfies LL'=diag(s) K^-1 diag(s); it is not momentum mass.
An exact incumbent change invalidates the fitted anchor. No MAP, posterior
covariance, whitening, HMC, GPU or whole-initializer XLA claim follows.

Orchestration is bounded and eager; target calls remain fixed-batch and the
trust kernel is reusable TF/XLA. Robust score fitting is compiled without XLA
because MatrixSolveLs(fast=False) is not an admitted whole-XLA path.
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
from bayesfilter.inference.paired_score_pilot_tf import (
    fit_paired_score_precision_tf,
    paired_score_probe_designs_tf,
    validate_paired_steps,
)
from bayesfilter.inference.score_curvature_tf import fit_dense_score_precision_tf


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
    eigenvalues, eigenvectors = tf.linalg.eigh(safe_matrix)
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
        raise RuntimeError("bounded initializer orchestration requires eager execution")
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
    rows = max(32, 4 * dimension) if cfg.rows_per_cloud is None else cfg.rows_per_cloud
    if cfg.pilot_method == "paired_local":
        rows = 2 * dimension
    details = {"physical_rows": 0, "callback_batches": 0, "padded_rows": 0,
               "invalid_rows": 0, "planned_physical_rows": planned, "selected_evaluation_index": -1,
               "rounds": [], "candidate_batches": [], "full_initializer_xla": False,
               "jit_compile_trust": cfg.jit_compile_trust, "seed": cfg.seed,
               "pilot_method": cfg.pilot_method, "paired_steps": cfg.paired_steps}
    incumbent = [center, tf.constant(-math.inf, tf.float64), tf.zeros_like(center)]

    def finish(status, factor=None, precision=None):
        accepted = status == "local_center_candidate"
        return BatchedQuadraticCenterResult(accepted, status, *incumbent,
                                           factor if accepted else None,
                                           precision if accepted else None, details)

    def evaluate(points, role, record=True):
        collected = []
        for start in range(0, int(points.shape[0]), cfg.batch_size):
            chunk = points[start:start + cfg.batch_size]
            padding = cfg.batch_size - int(chunk.shape[0])
            chunk = tf.concat((chunk, tf.repeat(chunk[-1:], padding, axis=0)), axis=0)
            first_index = details["physical_rows"]
            details["physical_rows"] += cfg.batch_size
            details["padded_rows"] += padding
            if not bool(tf.reduce_all(tf.math.is_finite(chunk))):
                return None
            details["callback_batches"] += 1
            values, scores, eligible = (tf.convert_to_tensor(item) for item in callback(chunk))
            if values.dtype != tf.float64 or scores.dtype != tf.float64 or eligible.dtype != tf.bool:
                raise TypeError("callback must return float64 values/scores and bool eligibility")
            if values.shape != (cfg.batch_size,) or scores.shape != chunk.shape or eligible.shape != values.shape:
                raise ValueError("callback returned an invalid fixed-batch shape")
            valid = eligible & tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=1)
            details["invalid_rows"] += int(tf.reduce_sum(tf.cast(~valid, tf.int32)))
            details["candidate_batches"].append({"role": role, "first_index": first_index,
                                                "positions": chunk, "values": values,
                                                "scores": scores, "valid": valid})
            if record:
                masked = tf.where(valid, values, tf.constant(-math.inf, tf.float64))
                selected = tf.argmax(masked, output_type=tf.int32)
                if bool(masked[selected] > incumbent[1]):
                    incumbent[:] = [chunk[selected], values[selected], scores[selected]]
                    details["selected_evaluation_index"] = first_index + int(selected)
            if not bool(tf.reduce_all(valid)):
                return None
            count = cfg.batch_size - padding if record else cfg.batch_size
            collected.append((values[:count], scores[:count]))
        return tf.concat([item[0] for item in collected], 0), tf.concat([item[1] for item in collected], 0)

    def replay():
        result = evaluate(incumbent[0][None, :], "replay", record=False)
        if result is None:
            return False
        return all(bool(tf.reduce_all(tf.abs(actual - expected) <= cfg.replay_atol + cfg.replay_rtol * tf.abs(expected)))
                   for actual, expected in zip(result, incumbent[1:]))

    if evaluate(center[None, :], "initial") is None:
        return finish("initial_target_invalid")
    if _initial_evidence is not None and not all(
        bool(tf.reduce_all(tf.abs(actual - expected) <= cfg.replay_atol + cfg.replay_rtol * tf.abs(expected)))
        for actual, expected in zip(incumbent[1:], _initial_evidence)
    ):
        return finish("localizer_refinement_replay_mismatch")
    if _initial_evidence is not None:
        incumbent[1:] = _initial_evidence
    signature = [tf.TensorSpec([dimension, dimension], tf.float64),
                 tf.TensorSpec([dimension], tf.float64), tf.TensorSpec([], tf.float64)]
    trust = tf.function(solve_spd_quadratic_trust_region_tf, input_signature=signature,
                        autograph=False, jit_compile=cfg.jit_compile_trust)
    fit_signature = [tf.TensorSpec([dimension], tf.float64)] + [tf.TensorSpec([rows, dimension], tf.float64)] * 4

    def fit(center_score, offsets, scores, check_offsets, check_scores):
        if cfg.pilot_method == "paired_local":
            return fit_paired_score_precision_tf(
                center_score, offsets, scores, check_offsets, check_scores, steps=cfg.paired_steps,
                precision_condition_cap=cfg.precision_condition_cap,
                model_relative_rmse_cap=cfg.model_relative_rmse_cap)
        return fit_dense_score_precision_tf(center_score, offsets, scores,
                                            selection_offsets=check_offsets, selection_scores=check_scores)

    fitted = tf.function(fit, input_signature=fit_signature, autograph=False, jit_compile=False)
    radius = tf.constant(cfg.initial_trust_radius, tf.float64)
    for round_index in range(cfg.max_fit_rounds):
        if not replay():
            return finish("replay_mismatch")
        anchor, anchor_value, anchor_score = incumbent
        partitions = []
        if cfg.pilot_method == "paired_local":
            designs = paired_score_probe_designs_tf(dimension, seed=cfg.seed, round_index=round_index,
                                                   steps=cfg.paired_steps)
            for offsets, role in zip(designs, ("paired_fit_large", "paired_fit_small", "model_check")):
                evaluated = evaluate(anchor[None, :] + offsets * scale, role)
                if evaluated is None:
                    return finish("curvature_target_invalid")
                if role == "model_check":
                    partitions.append(offsets)
                partitions.append(evaluated[1] * scale)
        else:
            for partition in range(2):
                offsets = tf.random.stateless_uniform([rows, dimension], [cfg.seed, 2 * round_index + partition],
                                                      minval=-cfg.fit_half_width, maxval=cfg.fit_half_width,
                                                      dtype=tf.float64)
                evaluated = evaluate(anchor[None, :] + offsets * scale, "fit" if partition == 0 else "model_check")
                if evaluated is None:
                    return finish("curvature_target_invalid")
                partitions.extend((offsets, evaluated[1] * scale))
        center_score = anchor_score * scale
        if not bool(tf.reduce_all(tf.math.is_finite(center_score))) or not all(
            bool(tf.reduce_all(tf.math.is_finite(value))) for value in partitions
        ):
            return finish("nonfinite_scaled_score")
        try:
            model = fitted(center_score, *partitions)
        except tf.errors.InvalidArgumentError:
            return finish("quadratic_fit_numerical_failure")
        precision = model["raw_precision"]
        good_model = model["raw_spd"] & (model["design_rank"] == dimension)
        good_model &= tf.reduce_all(tf.math.is_finite(precision))
        good_model &= model["precision_condition"] <= cfg.precision_condition_cap
        good_model &= model["selection_relative_rmse"] <= cfg.model_relative_rmse_cap
        if cfg.pilot_method == "paired_local":
            good_model &= model["accepted"]
        report = {"anchor": anchor, "anchor_value": anchor_value, "radius": radius,
                  "model": model, "cloud_changed_incumbent": tf.reduce_any(incumbent[0] != anchor)}
        details["rounds"].append(report)
        if not bool(good_model):
            return finish("quadratic_model_rejected")
        cholesky = tf.linalg.cholesky(precision)
        solved = tf.linalg.triangular_solve(cholesky, tf.linalg.diag(scale))
        covariance = tf.matmul(solved, solved, transpose_a=True)
        factor = tf.linalg.cholesky(covariance)
        reconstructed = factor @ tf.transpose(factor)
        covariance_scale = tf.reduce_max(tf.abs(covariance))
        reconstruction = tf.linalg.norm((reconstructed - covariance) / covariance_scale)
        reconstruction /= tf.linalg.norm(covariance / covariance_scale)
        factor_valid = tf.reduce_all(tf.math.is_finite(factor)) & tf.reduce_all(tf.linalg.diag_part(factor) > 0)
        factor_valid &= tf.math.is_finite(reconstruction) & (reconstruction <= 1e-10)
        if not bool(factor_valid):
            return finish("pilot_factorization_failed")
        centeredness = _norm(tf.linalg.matvec(factor, anchor_score, transpose_a=True))
        report["centeredness"] = centeredness
        if not bool(tf.math.is_finite(centeredness)):
            return finish("nonfinite_centeredness")
        if bool(centeredness <= cfg.centeredness_cap) and not bool(report["cloud_changed_incumbent"]):
            if not replay():
                return finish("replay_mismatch")
            details["fit_trace_count"] = fitted.experimental_get_tracing_count()
            details["trust_trace_count"] = trust.experimental_get_tracing_count()
            return finish("local_center_candidate", factor, precision)
        step = trust(precision, center_score, radius)
        if not bool(step["valid"]):
            return finish("trust_solve_failed")
        proposed = evaluate((anchor + scale * step["step"])[None, :], "proposal")
        predicted = step["predicted_improvement"]
        actual = tf.constant(-math.inf, tf.float64) if proposed is None else proposed[0][0] - anchor_value
        ratio = tf.where(predicted > 0, actual / predicted, tf.constant(-math.inf, tf.float64))
        report.update(step=step, actual_improvement=actual, ratio=ratio,
                      trust_step_accepted=tf.math.is_finite(ratio) & (actual > 0) & (ratio > 0.1))
        radius = tf.where(~tf.math.is_finite(ratio) | (ratio < 0.25), radius * 0.25,
                          tf.where((ratio > 0.75) & step["boundary_active"],
                                   tf.minimum(2 * radius, cfg.maximum_trust_radius), radius))
    return finish("refinement_round_limit")


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
