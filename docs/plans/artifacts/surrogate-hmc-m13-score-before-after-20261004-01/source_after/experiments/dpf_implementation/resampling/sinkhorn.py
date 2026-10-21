"""Finite Sinkhorn relaxed resampling for experimental OT-DPF diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SinkhornResampleResult:
    """Finite Sinkhorn relaxed-resampling output."""

    particles: np.ndarray
    coupling: np.ndarray
    source_weights: np.ndarray
    target_weights: np.ndarray
    diagnostics: dict[str, Any]


def pairwise_squared_euclidean(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    """Return a squared Euclidean cost matrix."""

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = x_arr if y is None else np.asarray(y, dtype=np.float64)
    diff = x_arr[:, None, :] - y_arr[None, :, :]
    return np.sum(diff * diff, axis=2)


def sinkhorn_resample(
    particles: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float = 0.5,
    max_iterations: int = 80,
    tolerance: float = 1e-7,
    cost: np.ndarray | None = None,
    stabilization: str = "log_domain",
) -> SinkhornResampleResult:
    """Compute a finite-budget EOT coupling and barycentric relaxed cloud.

    The source marginal is the normalized particle weight vector.  The target
    marginal is uniform.  The returned cloud is the barycentric projection of
    the finite coupling onto the original particle locations, scaled by the
    target marginal.  This is a relaxed numerical object, not categorical
    resampling.
    """

    x = np.asarray(particles, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    source = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x.shape[0] != source.shape[0]:
        raise ValueError("particles and weights must agree on particle count")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(source)):
        raise FloatingPointError("particles and weights must be finite")
    source_total = float(np.sum(source))
    if not np.isfinite(source_total) or source_total <= 0.0:
        raise FloatingPointError("source weights must have positive finite mass")
    source = source / source_total
    n = x.shape[0]
    target = np.full(n, 1.0 / n, dtype=np.float64)
    cost_matrix = pairwise_squared_euclidean(x) if cost is None else np.asarray(cost, dtype=np.float64)
    if cost_matrix.shape != (n, n):
        raise ValueError("cost matrix must have shape (N, N)")
    if not np.all(np.isfinite(cost_matrix)):
        raise FloatingPointError("cost matrix must be finite")

    if stabilization != "log_domain":
        raise ValueError("only log_domain stabilization is implemented")
    log_source = np.log(np.maximum(source, np.finfo(np.float64).tiny))
    log_target = np.log(target)
    kernel_log = -cost_matrix / epsilon
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(n, dtype=np.float64)
    last_row_residual = np.inf
    last_column_residual = np.inf

    for iteration in range(1, max_iterations + 1):
        log_u = log_source - _logsumexp_matrix(kernel_log + log_v[None, :], axis=1)
        log_v = log_target - _logsumexp_matrix(kernel_log + log_u[:, None], axis=0)
        if iteration == max_iterations or iteration % 10 == 0:
            coupling = np.exp(log_u[:, None] + kernel_log + log_v[None, :])
            row_residual = np.max(np.abs(np.sum(coupling, axis=1) - source))
            column_residual = np.max(np.abs(np.sum(coupling, axis=0) - target))
            last_row_residual = float(row_residual)
            last_column_residual = float(column_residual)
            if max(last_row_residual, last_column_residual) <= tolerance:
                break
    else:
        iteration = max_iterations

    coupling = np.exp(log_u[:, None] + kernel_log + log_v[None, :])
    column_mass = np.sum(coupling, axis=0)
    safe_column_mass = np.maximum(column_mass, np.finfo(np.float64).tiny)
    relaxed_particles = (coupling.T @ x) / safe_column_mass[:, None]
    row_residuals = np.sum(coupling, axis=1) - source
    column_residuals = column_mass - target
    diagnostics = {
        "component_id": "finite_sinkhorn_relaxed_resampler",
        "mathematical_object": "finite_budget_entropic_ot_coupling",
        "epsilon": float(epsilon),
        "max_iterations": int(max_iterations),
        "iterations_used": int(iteration),
        "tolerance": float(tolerance),
        "stabilization": stabilization,
        "cost_function": "pairwise_squared_euclidean",
        "target_marginal": "uniform",
        "max_row_residual": float(np.max(np.abs(row_residuals))),
        "max_column_residual": float(np.max(np.abs(column_residuals))),
        "total_mass_residual": float(abs(np.sum(coupling) - 1.0)),
        "min_coupling": float(np.min(coupling)),
        "finite_coupling": bool(np.all(np.isfinite(coupling))),
        "finite_particles": bool(np.all(np.isfinite(relaxed_particles))),
        "last_checked_row_residual": last_row_residual,
        "last_checked_column_residual": last_column_residual,
        "resampling_status": "relaxed_finite_sinkhorn_not_categorical",
    }
    if diagnostics["max_row_residual"] > tolerance * 10.0:
        raise FloatingPointError("Sinkhorn row residual exceeded tolerance envelope")
    if diagnostics["max_column_residual"] > tolerance * 10.0:
        raise FloatingPointError("Sinkhorn column residual exceeded tolerance envelope")
    if diagnostics["min_coupling"] < -1e-12:
        raise FloatingPointError("Sinkhorn coupling has negative entries")
    if not diagnostics["finite_coupling"] or not diagnostics["finite_particles"]:
        raise FloatingPointError("Sinkhorn emitted non-finite values")
    return SinkhornResampleResult(
        particles=relaxed_particles,
        coupling=coupling,
        source_weights=source,
        target_weights=target,
        diagnostics=diagnostics,
    )


def _logsumexp_matrix(values: np.ndarray, *, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    if not np.all(np.isfinite(max_values)):
        raise FloatingPointError("logsumexp received non-finite maximum")
    summed = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    result = max_values + np.log(summed)
    return np.squeeze(result, axis=axis)
