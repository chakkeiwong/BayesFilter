"""Metric functions for clean-room controlled DPF baseline records."""

from __future__ import annotations

from typing import Any

import numpy as np

from experiments.controlled_dpf_baseline.fixtures.range_bearing import (
    RangeBearingFixture,
    observation_residual,
    range_bearing_observation,
)


REQUIRED_METRICS = (
    "state_rmse",
    "position_rmse",
    "final_position_error",
    "observation_proxy_rmse",
    "runtime_seconds",
)


def compute_metrics(
    fixture: RangeBearingFixture,
    filtered_means: np.ndarray,
    *,
    runtime_seconds: float,
    particles: np.ndarray | None = None,
    covariances: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute required proxy metrics for aligned filtered means."""

    means = np.asarray(filtered_means, dtype=float)
    aligned_means, aligned_states, post_means = _align_means(fixture, means)
    state_error = aligned_means - aligned_states
    position_error = state_error[:, :2]
    predicted_obs = range_bearing_observation(post_means)
    obs_residual = observation_residual(predicted_obs, fixture.observations)
    metrics = {
        "state_rmse": _rmse(state_error),
        "position_rmse": _rmse(position_error),
        "final_position_error": float(
            np.linalg.norm(aligned_means[-1, :2] - aligned_states[-1, :2])
        ),
        "observation_proxy_rmse": _rmse(obs_residual),
        "runtime_seconds": float(runtime_seconds),
    }
    metrics["finite_outputs"] = finite_output_diagnostics(
        means=means,
        particles=particles,
        covariances=covariances,
        metrics=metrics,
    )
    return metrics


def finite_output_diagnostics(
    *,
    means: np.ndarray,
    particles: np.ndarray | None,
    covariances: np.ndarray | None,
    metrics: dict[str, Any],
) -> dict[str, bool]:
    """Return finite-output flags for required result arrays and scalars."""

    scalar_values = [
        value
        for key, value in metrics.items()
        if key != "finite_outputs" and isinstance(value, int | float)
    ]
    return {
        "means_finite": bool(np.all(np.isfinite(means))),
        "particles_finite": (
            None if particles is None else bool(np.all(np.isfinite(particles)))
        ),
        "covariances_finite": (
            None if covariances is None else bool(np.all(np.isfinite(covariances)))
        ),
        "scalar_metrics_finite": bool(np.all(np.isfinite(scalar_values))),
    }


def required_metrics_are_finite(metrics: dict[str, Any]) -> bool:
    """Return whether all required scalar metrics and finite flags pass."""

    for key in REQUIRED_METRICS:
        value = metrics.get(key)
        if not isinstance(value, int | float) or not np.isfinite(value):
            return False
    finite_outputs = metrics.get("finite_outputs", {})
    for key in ("means_finite", "scalar_metrics_finite"):
        if finite_outputs.get(key) is not True:
            return False
    for key in ("particles_finite", "covariances_finite"):
        value = finite_outputs.get(key)
        if value is False:
            return False
    return True


def _align_means(
    fixture: RangeBearingFixture,
    means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if means.ndim != 2 or means.shape[1] != fixture.state_dim:
        raise ValueError(
            f"filtered means must have shape (T, {fixture.state_dim}); got {means.shape}"
        )
    if means.shape[0] == fixture.horizon + 1:
        return means, fixture.states, means[1:]
    if means.shape[0] == fixture.horizon:
        return means, fixture.states[1:], means
    raise ValueError(
        "filtered means must contain either horizon post-observation means "
        "or horizon+1 means including the initial state"
    )


def _rmse(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr**2)))
