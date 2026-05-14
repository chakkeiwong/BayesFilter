"""Model B/C thorough-testing harness for BayesFilter V1.

This script is benchmark/test infrastructure, not production code.  It keeps
the BC1-BC3 Model B/C grids, finite-difference score checks, and horizon/noise
diagnostics in one place so phase artifacts have row-level provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402

from bayesfilter.testing import (  # noqa: E402
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    nonlinear_sigma_point_diagnostic_snapshot,
    tf_nonlinear_sigma_point_score,
    tf_nonlinear_sigma_point_value_filter,
)


BACKENDS = ("tf_svd_cubature", "tf_svd_ukf", "tf_svd_cut4")
MODELS = ("model_b_nonlinear_accumulation", "model_c_autonomous_nonlinear_growth")
SEED = 20260515


@dataclass(frozen=True)
class EvaluationSnapshot:
    status: str
    failure_label: str | None
    log_likelihood: float | None
    score: list[float] | None
    point_count: int | None
    max_integration_rank: int | None
    placement_floor_count: int | None
    innovation_floor_count: int | None
    min_placement_eigen_gap: float | None
    min_innovation_eigen_gap: float | None
    support_residual: float | None
    deterministic_residual: float | None
    structural_null_count: int | None
    structural_null_covariance_residual: float | None
    fixed_null_derivative_residual: float | None
    derivative_branch: str | None
    derivative_method: str | None
    sigma_point_variable: str | None
    error: str | None


@dataclass(frozen=True)
class BranchBoxRow:
    phase: str
    model: str
    backend: str
    row_family: str
    row_index: int
    seed: int | None
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    parameter_box: tuple[tuple[float, float], ...]
    allow_fixed_null_support: bool
    value: EvaluationSnapshot
    score: EvaluationSnapshot
    claim_status: str


@dataclass(frozen=True)
class ScoreStressRow:
    phase: str
    model: str
    backend: str
    row_family: str
    row_index: int
    seed: int | None
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    allow_fixed_null_support: bool
    primary_step: float
    step_ladder: tuple[float, ...]
    absolute_tolerance: float
    relative_tolerance: float
    analytic_score: tuple[float, ...] | None
    finite_difference_score: tuple[float, ...] | None
    max_abs_residual: float | None
    max_relative_residual: float | None
    ladder_max_abs_residuals: tuple[float | None, ...]
    ladder_max_relative_residuals: tuple[float | None, ...]
    compiled_value_residual: float | None
    compiled_score_residual: float | None
    branch_label: str | None
    status: str
    failure_label: str | None
    error: str | None


@dataclass(frozen=True)
class HorizonNoiseRow:
    phase: str
    model: str
    backend: str
    panel_family: str
    seed: int | None
    horizon: int
    observation_noise_scale: float
    parameter_names: tuple[str, ...]
    parameters: tuple[float, ...]
    allow_fixed_null_support: bool
    value: EvaluationSnapshot
    score: EvaluationSnapshot
    runtime_seconds: float
    claim_status: str


def _model_b_box() -> tuple[tuple[float, float], ...]:
    return ((0.55, 0.85), (0.15, 0.40), (0.45, 1.10))


def _model_c_box() -> tuple[tuple[float, float], ...]:
    return ((0.60, 1.40), (0.60, 1.40), (0.10, 0.50))


def _parameter_names(model: str) -> tuple[str, ...]:
    if model == "model_b_nonlinear_accumulation":
        return ("rho", "sigma", "beta")
    if model == "model_c_autonomous_nonlinear_growth":
        return ("sigma_u", "sigma_y", "P0x")
    raise ValueError(f"unknown model: {model}")


def _parameter_box(model: str) -> tuple[tuple[float, float], ...]:
    if model == "model_b_nonlinear_accumulation":
        return _model_b_box()
    if model == "model_c_autonomous_nonlinear_growth":
        return _model_c_box()
    raise ValueError(f"unknown model: {model}")


def _deterministic_rows(model: str) -> tuple[tuple[float, ...], ...]:
    if model == "model_b_nonlinear_accumulation":
        return (
            (0.55, 0.15, 0.45),
            (0.70, 0.25, 0.80),
            (0.85, 0.40, 1.10),
            (0.55, 0.40, 1.10),
            (0.85, 0.15, 0.45),
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return (
            (0.60, 0.60, 0.10),
            (1.00, 1.00, 0.20),
            (1.40, 1.40, 0.50),
            (0.60, 1.40, 0.50),
            (1.40, 0.60, 0.10),
        )
    raise ValueError(f"unknown model: {model}")


def _seeded_rows(model: str, *, seed: int = SEED, count: int = 5) -> tuple[tuple[float, ...], ...]:
    rng = random.Random(seed + (101 if model.endswith("growth") else 0))
    rows = []
    for _ in range(count):
        rows.append(tuple(lo + (hi - lo) * rng.random() for lo, hi in _parameter_box(model)))
    return tuple(rows)


def _all_branch_rows(model: str) -> tuple[tuple[str, int, int | None, tuple[float, ...]], ...]:
    rows = [
        ("deterministic_grid", index, None, values)
        for index, values in enumerate(_deterministic_rows(model))
    ]
    rows.extend(
        ("seeded_random_box", index, SEED, values)
        for index, values in enumerate(_seeded_rows(model))
    )
    return tuple(rows)


def _default_parameters(model: str, *, observation_noise_scale: float = 1.0) -> tf.Tensor:
    if model == "model_b_nonlinear_accumulation":
        return tf.constant([0.70, 0.25, 0.80], dtype=tf.float64)
    if model == "model_c_autonomous_nonlinear_growth":
        return tf.constant([1.00, 1.00 * observation_noise_scale, 0.20], dtype=tf.float64)
    raise ValueError(f"unknown model: {model}")


def _allow_fixed_null_support(model: str) -> bool:
    return model == "model_c_autonomous_nonlinear_growth"


def _model_and_derivatives(
    model: str,
    params: tf.Tensor,
    *,
    observation_noise_scale: float = 1.0,
):
    params = tf.convert_to_tensor(params, dtype=tf.float64)
    if model == "model_b_nonlinear_accumulation":
        return (
            make_nonlinear_accumulation_model_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
                observation_sigma=tf.constant(0.30 * observation_noise_scale, dtype=tf.float64),
            ),
            make_nonlinear_accumulation_first_derivatives_tf(
                rho=params[0],
                sigma=params[1],
                beta=params[2],
            ),
        )
    if model == "model_c_autonomous_nonlinear_growth":
        return (
            make_univariate_nonlinear_growth_model_tf(
                process_sigma=params[0],
                observation_sigma=params[1],
                initial_variance=params[2],
            ),
            make_univariate_nonlinear_growth_first_derivatives_tf(
                process_sigma=params[0],
                observation_sigma=params[1],
            ),
        )
    raise ValueError(f"unknown model: {model}")


def _observations(
    model: str,
    *,
    horizon: int | None = None,
    panel_family: str = "default",
    seed: int | None = None,
) -> tf.Tensor:
    base = (
        model_b_observations_tf()
        if model == "model_b_nonlinear_accumulation"
        else model_c_observations_tf()
    )
    if horizon is None:
        return base
    repeats = math.ceil(horizon / int(base.shape[0]))
    panel = tf.tile(base, [repeats, 1])[:horizon]
    if panel_family == "seeded_stochastic":
        if seed is None:
            raise ValueError("seeded_stochastic panels require a seed")
        noise = tf.random.stateless_normal(
            tf.shape(panel),
            seed=tf.constant([seed, horizon], dtype=tf.int32),
            dtype=tf.float64,
        )
        panel = panel + tf.constant(0.03, dtype=tf.float64) * noise
    return panel


def _evaluate_value(
    observations: tf.Tensor,
    model_name: str,
    backend: str,
    params: tf.Tensor,
    *,
    observation_noise_scale: float = 1.0,
) -> EvaluationSnapshot:
    try:
        model, _ = _model_and_derivatives(
            model_name,
            params,
            observation_noise_scale=observation_noise_scale,
        )
        result = tf_nonlinear_sigma_point_value_filter(
            observations,
            model,
            backend=backend,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
        )
        tf.debugging.assert_all_finite(result.log_likelihood, "blocked_nonfinite_value")
        snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="value")
        return _snapshot_from_result(
            status="ok",
            failure_label=None,
            result=result,
            snapshot=snapshot,
            score=None,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return _blocked_snapshot(exc)


def _evaluate_score(
    observations: tf.Tensor,
    model_name: str,
    backend: str,
    params: tf.Tensor,
    *,
    observation_noise_scale: float = 1.0,
) -> EvaluationSnapshot:
    try:
        model, derivatives = _model_and_derivatives(
            model_name,
            params,
            observation_noise_scale=observation_noise_scale,
        )
        result = tf_nonlinear_sigma_point_score(
            observations,
            model,
            derivatives,
            backend=backend,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
            allow_fixed_null_support=_allow_fixed_null_support(model_name),
        )
        tf.debugging.assert_all_finite(result.log_likelihood, "blocked_nonfinite_value")
        tf.debugging.assert_all_finite(result.score, "blocked_nonfinite_score")
        snapshot = nonlinear_sigma_point_diagnostic_snapshot(result, mode="score")
        return _snapshot_from_result(
            status="ok",
            failure_label=None,
            result=result,
            snapshot=snapshot,
            score=result.score,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        return _blocked_snapshot(exc)


def _snapshot_from_result(
    *,
    status: str,
    failure_label: str | None,
    result: Any,
    snapshot: Any,
    score: tf.Tensor | None,
    error: str | None,
) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        status=status,
        failure_label=failure_label,
        log_likelihood=_to_float(result.log_likelihood),
        score=None if score is None else _to_float_list(score),
        point_count=snapshot.point_count,
        max_integration_rank=snapshot.max_integration_rank,
        placement_floor_count=snapshot.placement_floor_count,
        innovation_floor_count=snapshot.innovation_floor_count,
        min_placement_eigen_gap=snapshot.min_placement_eigen_gap,
        min_innovation_eigen_gap=snapshot.min_innovation_eigen_gap,
        support_residual=snapshot.support_residual,
        deterministic_residual=snapshot.deterministic_residual,
        structural_null_count=snapshot.structural_null_count,
        structural_null_covariance_residual=snapshot.structural_null_covariance_residual,
        fixed_null_derivative_residual=snapshot.fixed_null_derivative_residual,
        derivative_branch=snapshot.derivative_branch,
        derivative_method=snapshot.derivative_method,
        sigma_point_variable=snapshot.sigma_point_variable,
        error=error,
    )


def _blocked_snapshot(exc: Exception) -> EvaluationSnapshot:
    label = _failure_label(str(exc))
    return EvaluationSnapshot(
        status=label,
        failure_label=label,
        log_likelihood=None,
        score=None,
        point_count=None,
        max_integration_rank=None,
        placement_floor_count=None,
        innovation_floor_count=None,
        min_placement_eigen_gap=None,
        min_innovation_eigen_gap=None,
        support_residual=None,
        deterministic_residual=None,
        structural_null_count=None,
        structural_null_covariance_residual=None,
        fixed_null_derivative_residual=None,
        derivative_branch=None,
        derivative_method=None,
        sigma_point_variable=None,
        error=str(exc).splitlines()[0],
    )


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
    return "blocked_other"


def run_branch_boxes() -> dict[str, Any]:
    rows = []
    for model in MODELS:
        observations = _observations(model)
        for backend in BACKENDS:
            for row_family, row_index, seed, values in _all_branch_rows(model):
                params = tf.constant(values, dtype=tf.float64)
                value = _evaluate_value(observations, model, backend, params)
                score = _evaluate_score(observations, model, backend, params)
                claim_status = "stable" if value.status == "ok" and score.status == "ok" else "blocked"
                rows.append(
                    BranchBoxRow(
                        phase="BC1",
                        model=model,
                        backend=backend,
                        row_family=row_family,
                        row_index=row_index,
                        seed=seed,
                        parameter_names=_parameter_names(model),
                        parameters=tuple(float(x) for x in values),
                        parameter_box=_parameter_box(model),
                        allow_fixed_null_support=_allow_fixed_null_support(model),
                        value=value,
                        score=score,
                        claim_status=claim_status,
                    )
                )
    return _payload(
        phase="BC1",
        claim_scope="row_level_branch_box_diagnostics",
        rows=[asdict(row) for row in rows],
        config={
            "seed": SEED,
            "deterministic_rows_per_model": 5,
            "seeded_rows_per_model": 5,
            "models": MODELS,
            "backends": BACKENDS,
            "model_b_box": _model_b_box(),
            "model_c_box": _model_c_box(),
            "model_c_score_allow_fixed_null_support": True,
        },
    )


def run_score_stress() -> dict[str, Any]:
    rows = []
    for model in MODELS:
        observations = _observations(model)
        tolerance = _tolerance(model)
        for backend in BACKENDS:
            for row_family, row_index, seed, values in _all_branch_rows(model):
                rows.append(
                    _score_stress_row(
                        model=model,
                        backend=backend,
                        row_family=row_family,
                        row_index=row_index,
                        seed=seed,
                        values=values,
                        observations=observations,
                        tolerance=tolerance,
                    )
                )
    return _payload(
        phase="BC2",
        claim_scope="analytic_score_vs_centered_finite_difference_on_bc1_boxes",
        rows=[asdict(row) for row in rows],
        config={
            "seed": SEED,
            "models": MODELS,
            "backends": BACKENDS,
            "tolerances": {
                model: _tolerance(model)
                for model in MODELS
            },
            "model_c_score_allow_fixed_null_support": True,
            "pass_fail_metric": "primary_step_max_abs_and_relative_residual",
        },
    )


def _score_stress_row(
    *,
    model: str,
    backend: str,
    row_family: str,
    row_index: int,
    seed: int | None,
    values: tuple[float, ...],
    observations: tf.Tensor,
    tolerance: dict[str, Any],
) -> ScoreStressRow:
    params = tf.constant(values, dtype=tf.float64)
    try:
        score_snapshot = _evaluate_score(observations, model, backend, params)
        if score_snapshot.status != "ok" or score_snapshot.score is None:
            raise ValueError(score_snapshot.failure_label or "blocked_score_branch")
        analytic_score = tf.constant(score_snapshot.score, dtype=tf.float64)

        ladder_abs: list[float | None] = []
        ladder_rel: list[float | None] = []
        primary_fd = None
        primary_abs = None
        primary_rel = None
        for step in tolerance["step_ladder"]:
            if not _finite_difference_branch_is_stable(
                observations,
                model,
                backend,
                params,
                step=step,
            ):
                ladder_abs.append(None)
                ladder_rel.append(None)
                continue
            fd_score = _finite_difference_score(observations, model, backend, params, step=step)
            abs_residual = tf.abs(analytic_score - fd_score)
            relative_residual = abs_residual / tf.maximum(
                tf.ones_like(abs_residual),
                tf.maximum(tf.abs(analytic_score), tf.abs(fd_score)),
            )
            max_abs = _to_float(tf.reduce_max(abs_residual))
            max_rel = _to_float(tf.reduce_max(relative_residual))
            ladder_abs.append(max_abs)
            ladder_rel.append(max_rel)
            if math.isclose(step, tolerance["primary_step"]):
                primary_fd = fd_score
                primary_abs = max_abs
                primary_rel = max_rel

        if primary_fd is None or primary_abs is None or primary_rel is None:
            raise ValueError("blocked_fd_branch_change")
        compiled_value_residual, compiled_score_residual = _compiled_score_parity(
            observations,
            model,
            backend,
            params,
            analytic_score,
            float(score_snapshot.log_likelihood),
        )
        passes = (
            primary_abs <= tolerance["absolute_tolerance"]
            and primary_rel <= tolerance["relative_tolerance"]
        )
        status = "ok" if passes else "blocked_tolerance_residual"
        failure_label = None if passes else "blocked_tolerance_residual"
        return ScoreStressRow(
            phase="BC2",
            model=model,
            backend=backend,
            row_family=row_family,
            row_index=row_index,
            seed=seed,
            parameter_names=_parameter_names(model),
            parameters=tuple(float(x) for x in values),
            allow_fixed_null_support=_allow_fixed_null_support(model),
            primary_step=tolerance["primary_step"],
            step_ladder=tuple(tolerance["step_ladder"]),
            absolute_tolerance=tolerance["absolute_tolerance"],
            relative_tolerance=tolerance["relative_tolerance"],
            analytic_score=tuple(_to_float_list(analytic_score)),
            finite_difference_score=tuple(_to_float_list(primary_fd)),
            max_abs_residual=primary_abs,
            max_relative_residual=primary_rel,
            ladder_max_abs_residuals=tuple(ladder_abs),
            ladder_max_relative_residuals=tuple(ladder_rel),
            compiled_value_residual=compiled_value_residual,
            compiled_score_residual=compiled_score_residual,
            branch_label=score_snapshot.derivative_branch,
            status=status,
            failure_label=failure_label,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - artifact path.
        label = _failure_label(str(exc))
        if label == "blocked_other" and "blocked_tolerance_residual" in str(exc):
            label = "blocked_tolerance_residual"
        if label == "blocked_other" and "blocked_fd_branch_change" in str(exc):
            label = "blocked_fd_branch_change"
        return ScoreStressRow(
            phase="BC2",
            model=model,
            backend=backend,
            row_family=row_family,
            row_index=row_index,
            seed=seed,
            parameter_names=_parameter_names(model),
            parameters=tuple(float(x) for x in values),
            allow_fixed_null_support=_allow_fixed_null_support(model),
            primary_step=tolerance["primary_step"],
            step_ladder=tuple(tolerance["step_ladder"]),
            absolute_tolerance=tolerance["absolute_tolerance"],
            relative_tolerance=tolerance["relative_tolerance"],
            analytic_score=None,
            finite_difference_score=None,
            max_abs_residual=None,
            max_relative_residual=None,
            ladder_max_abs_residuals=tuple(None for _ in tolerance["step_ladder"]),
            ladder_max_relative_residuals=tuple(None for _ in tolerance["step_ladder"]),
            compiled_value_residual=None,
            compiled_score_residual=None,
            branch_label=None,
            status=label,
            failure_label=label,
            error=str(exc).splitlines()[0],
        )


def _tolerance(model: str) -> dict[str, Any]:
    if model == "model_b_nonlinear_accumulation":
        return {
            "absolute_tolerance": 2.0e-3,
            "relative_tolerance": 2.0e-3,
            "primary_step": 2.0e-5,
            "step_ladder": (5.0e-5, 2.0e-5, 1.0e-5),
            "source": "P1 baseline used 5e-4 at the center; widened box gets a 4x finite-difference margin.",
        }
    if model == "model_c_autonomous_nonlinear_growth":
        return {
            "absolute_tolerance": 1.0e-2,
            "relative_tolerance": 1.0e-2,
            "primary_step": 1.0e-5,
            "step_ladder": (3.0e-5, 1.0e-5, 3.0e-6),
            "source": "P1 baseline used 1e-3 at the center; structural fixed-support box gets a conservative 10x finite-difference margin.",
        }
    raise ValueError(f"unknown model: {model}")


def _finite_difference_branch_is_stable(
    observations: tf.Tensor,
    model: str,
    backend: str,
    params: tf.Tensor,
    *,
    step: float,
) -> bool:
    for index in range(int(params.shape[0])):
        direction = tf.one_hot(index, int(params.shape[0]), dtype=tf.float64) * step
        for shifted in (params + direction, params - direction):
            result = _evaluate_score(observations, model, backend, shifted)
            if result.status != "ok":
                return False
    return True


def _finite_difference_score(
    observations: tf.Tensor,
    model: str,
    backend: str,
    params: tf.Tensor,
    *,
    step: float,
) -> tf.Tensor:
    pieces = []
    for index in range(int(params.shape[0])):
        direction = tf.one_hot(index, int(params.shape[0]), dtype=tf.float64) * step
        plus = _value_scalar(observations, model, backend, params + direction)
        minus = _value_scalar(observations, model, backend, params - direction)
        pieces.append((plus - minus) / (2.0 * step))
    return tf.stack(pieces)


def _value_scalar(
    observations: tf.Tensor,
    model: str,
    backend: str,
    params: tf.Tensor,
    *,
    observation_noise_scale: float = 1.0,
) -> tf.Tensor:
    structural_model, _ = _model_and_derivatives(
        model,
        params,
        observation_noise_scale=observation_noise_scale,
    )
    result = tf_nonlinear_sigma_point_value_filter(
        observations,
        structural_model,
        backend=backend,
        innovation_floor=tf.constant(1e-12, dtype=tf.float64),
    )
    return result.log_likelihood


def _compiled_score_parity(
    observations: tf.Tensor,
    model: str,
    backend: str,
    params: tf.Tensor,
    eager_score: tf.Tensor,
    eager_value: float,
) -> tuple[float, float]:
    @tf.function(reduce_retracing=True)
    def compiled(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        structural_model, derivatives = _model_and_derivatives(model, theta)
        result = tf_nonlinear_sigma_point_score(
            observations,
            structural_model,
            derivatives,
            backend=backend,
            innovation_floor=tf.constant(1e-12, dtype=tf.float64),
            spectral_gap_tolerance=tf.constant(1e-8, dtype=tf.float64),
            allow_fixed_null_support=_allow_fixed_null_support(model),
        )
        return result.log_likelihood, result.score

    graph_value, graph_score = compiled(params)
    return (
        abs(_to_float(graph_value) - eager_value),
        _to_float(tf.reduce_max(tf.abs(graph_score - eager_score))),
    )


def run_horizon_noise() -> dict[str, Any]:
    horizons = (3, 8, 16, 32)
    noise_scales = (1.0, 0.5, 0.25)
    panel_families = (("deterministic", None), ("seeded_stochastic", SEED))
    rows = []
    for model in MODELS:
        for backend in BACKENDS:
            for panel_family, seed in panel_families:
                for horizon in horizons:
                    observations = _observations(
                        model,
                        horizon=horizon,
                        panel_family=panel_family,
                        seed=seed,
                    )
                    for noise_scale in noise_scales:
                        params = _default_parameters(
                            model,
                            observation_noise_scale=noise_scale,
                        )
                        start = time.perf_counter()
                        value = _evaluate_value(
                            observations,
                            model,
                            backend,
                            params,
                            observation_noise_scale=noise_scale,
                        )
                        score = _evaluate_score(
                            observations,
                            model,
                            backend,
                            params,
                            observation_noise_scale=noise_scale,
                        )
                        runtime = time.perf_counter() - start
                        rows.append(
                            HorizonNoiseRow(
                                phase="BC3",
                                model=model,
                                backend=backend,
                                panel_family=panel_family,
                                seed=seed,
                                horizon=horizon,
                                observation_noise_scale=noise_scale,
                                parameter_names=_parameter_names(model),
                                parameters=tuple(_to_float_list(params)),
                                allow_fixed_null_support=_allow_fixed_null_support(model),
                                value=value,
                                score=score,
                                runtime_seconds=runtime,
                                claim_status=(
                                    "stable_envelope_row"
                                    if value.status == "ok" and score.status == "ok"
                                    else "blocked"
                                ),
                            )
                        )
    return _payload(
        phase="BC3",
        claim_scope="center_parameter_horizon_noise_envelope",
        rows=[asdict(row) for row in rows],
        config={
            "seed": SEED,
            "horizons": horizons,
            "observation_noise_scales": noise_scales,
            "panel_families": panel_families,
            "models": MODELS,
            "backends": BACKENDS,
            "model_c_score_allow_fixed_null_support": True,
        },
    )


def _payload(
    *,
    phase: str,
    claim_scope: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    return _json_safe(
        {
            "benchmark": "bayesfilter_v1_model_bc_testing",
            "phase": phase,
            "claim_scope": claim_scope,
            "config": config,
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "tensorflow": tf.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_devices": [
                    {"name": device.name, "device_type": device.device_type}
                    for device in tf.config.list_logical_devices()
                ],
            },
            "rows": rows,
        }
    )


def _markdown(payload: dict[str, Any], json_path: Path) -> str:
    rows = payload["rows"]
    status_counts: dict[str, int] = {}
    for row in rows:
        key = row.get("claim_status") or row.get("status") or "unknown"
        status_counts[key] = status_counts.get(key, 0) + 1
    lines = [
        f"# BayesFilter V1 Model B/C {payload['phase']} Artifact",
        "",
        f"The JSON file is authoritative: `{json_path}`.",
        "",
        "## Claim Scope",
        "",
        str(payload["claim_scope"]),
        "",
        "## Status Counts",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for key in sorted(status_counts):
        lines.append(f"| `{key}` | {status_counts[key]} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This artifact is BayesFilter-local testing evidence.",
            "- Model C score rows require `allow_fixed_null_support=True`.",
            "- Dense one-step projection and finite differences are diagnostics, not exact full nonlinear likelihood evidence.",
        ]
    )
    return "\n".join(lines)


def _to_float(value: Any) -> float:
    return float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())


def _to_float_list(value: Any) -> list[float]:
    return [float(x) for x in tf.reshape(tf.convert_to_tensor(value, dtype=tf.float64), [-1]).numpy()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("branch-boxes", "score-stress", "horizon-noise"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()

    if args.phase == "branch-boxes":
        payload = run_branch_boxes()
    elif args.phase == "score-stress":
        payload = run_score_stress()
    elif args.phase == "horizon-noise":
        payload = run_horizon_noise()
    else:  # pragma: no cover - argparse validates this.
        raise ValueError(args.phase)

    args.output.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(json.dumps(payload, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
