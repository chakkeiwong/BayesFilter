"""Minimal clean-room flow-assisted particle-filter baseline."""

from __future__ import annotations

import time
import traceback
from typing import Any

import numpy as np

from experiments.controlled_dpf_baseline.fixtures.range_bearing import (
    RangeBearingFixture,
    observation_residual,
    range_bearing_jacobian,
    range_bearing_observation,
)
from experiments.controlled_dpf_baseline.metrics import compute_metrics
from experiments.controlled_dpf_baseline.results import (
    make_failure_record,
    make_ok_record,
    make_target,
)


def run_clean_room_particle_flow(
    fixture: RangeBearingFixture,
    *,
    seed: int,
    num_particles: int,
    flow_steps: int,
    grid: str,
) -> dict[str, Any]:
    """Run the clean-room regularized particle-flow/bootstrap-PF scaffold."""

    target = make_target(
        grid=grid,
        fixture_name=fixture.name,
        num_particles=num_particles,
        flow_steps=flow_steps,
    )
    start = time.perf_counter()
    try:
        rng = np.random.default_rng(seed)
        result = _run_filter(
            fixture=fixture,
            rng=rng,
            num_particles=num_particles,
            flow_steps=flow_steps,
        )
        runtime_seconds = time.perf_counter() - start
        metrics = compute_metrics(
            fixture,
            result["filtered_means"],
            runtime_seconds=runtime_seconds,
            particles=result["particles"],
            covariances=result["covariances"],
        )
        diagnostics = {
            "average_ess": float(np.mean(result["ess_by_time"])),
            "min_ess": float(np.min(result["ess_by_time"])),
            "resampling_count": int(result["resampling_count"]),
            "ess_by_time": result["ess_by_time"],
            "resampled_by_time": result["resampled_by_time"],
            "flow_steps": int(flow_steps),
            "resampling_threshold": 0.5,
            "log_likelihood": None,
            "algorithm_note": (
                "clean-room EKF-style innovation flow followed by Gaussian "
                "likelihood weighting and systematic resampling"
            ),
        }
        return make_ok_record(
            fixture_name=fixture.name,
            seed=seed,
            num_particles=num_particles,
            flow_steps=flow_steps,
            horizon=fixture.horizon,
            target=target,
            runtime_seconds=runtime_seconds,
            metrics=metrics,
            diagnostics=diagnostics,
            provenance=_provenance(fixture),
        )
    except Exception as exc:  # pragma: no cover - exercised through failure records
        runtime_seconds = time.perf_counter() - start
        return make_failure_record(
            status="failed",
            failure_reason="failed_unexpected_exception",
            fixture_name=fixture.name,
            seed=seed,
            num_particles=num_particles,
            flow_steps=flow_steps,
            horizon=fixture.horizon,
            target=target,
            runtime_seconds=runtime_seconds,
            diagnostics={
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
            },
            provenance=_provenance(fixture),
        )


def _run_filter(
    *,
    fixture: RangeBearingFixture,
    rng: np.random.Generator,
    num_particles: int,
    flow_steps: int,
) -> dict[str, Any]:
    particles = rng.multivariate_normal(fixture.m0, fixture.P0, size=num_particles)
    weights = np.full(num_particles, 1.0 / num_particles, dtype=float)
    filtered_means = np.zeros((fixture.horizon + 1, fixture.state_dim), dtype=float)
    covariances = np.zeros(
        (fixture.horizon + 1, fixture.state_dim, fixture.state_dim),
        dtype=float,
    )
    filtered_means[0] = np.average(particles, axis=0, weights=weights)
    covariances[0] = _weighted_covariance(particles, weights)
    ess_by_time: list[float] = []
    resampled_by_time: list[bool] = []
    resampling_count = 0

    for t, observed in enumerate(fixture.observations):
        process_noise = rng.multivariate_normal(
            np.zeros(fixture.state_dim),
            fixture.Q,
            size=num_particles,
        )
        particles = particles @ fixture.A.T + process_noise
        if flow_steps > 0:
            particles = _apply_innovation_flow(
                particles=particles,
                observed=observed,
                R=fixture.R,
                flow_steps=flow_steps,
            )
        log_weights = _gaussian_log_likelihoods(particles, observed, fixture.R)
        weights = _normalize_log_weights(log_weights)
        ess = float(1.0 / np.sum(weights**2))
        filtered_means[t + 1] = np.average(particles, axis=0, weights=weights)
        covariances[t + 1] = _weighted_covariance(particles, weights)
        should_resample = ess < 0.5 * num_particles
        if should_resample:
            indices = _systematic_resample(weights, rng)
            particles = particles[indices]
            particles = _regularize_particles(particles, rng)
            weights = np.full(num_particles, 1.0 / num_particles, dtype=float)
            resampling_count += 1
        ess_by_time.append(ess)
        resampled_by_time.append(bool(should_resample))

    return {
        "filtered_means": filtered_means,
        "covariances": covariances,
        "particles": particles,
        "ess_by_time": np.asarray(ess_by_time, dtype=float),
        "resampled_by_time": resampled_by_time,
        "resampling_count": resampling_count,
    }


def _apply_innovation_flow(
    *,
    particles: np.ndarray,
    observed: np.ndarray,
    R: np.ndarray,
    flow_steps: int,
) -> np.ndarray:
    flowed = np.asarray(particles, dtype=float).copy()
    for _ in range(flow_steps):
        covariance = np.cov(flowed, rowvar=False)
        covariance = np.atleast_2d(covariance) + 1e-6 * np.eye(flowed.shape[1])
        for i, particle in enumerate(flowed):
            predicted = range_bearing_observation(particle)
            residual = observation_residual(predicted, observed)
            H = range_bearing_jacobian(particle)
            innovation_cov = H @ covariance @ H.T + R
            gain = covariance @ H.T @ np.linalg.pinv(innovation_cov)
            flowed[i] = particle + (gain @ residual) / float(flow_steps)
    return flowed


def _gaussian_log_likelihoods(
    particles: np.ndarray,
    observed: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    predicted = range_bearing_observation(particles)
    residual = observation_residual(predicted, observed)
    inv_R = np.linalg.inv(R)
    log_det = float(np.linalg.slogdet(R)[1])
    obs_dim = int(R.shape[0])
    quadratic = np.einsum("ni,ij,nj->n", residual, inv_R, residual)
    return -0.5 * (quadratic + log_det + obs_dim * np.log(2.0 * np.pi))


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.full(log_weights.shape[0], 1.0 / log_weights.shape[0])
    return weights / total


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_particles = int(weights.shape[0])
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _regularize_particles(
    particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if particles.shape[0] < 2:
        return particles
    covariance = np.cov(particles, rowvar=False)
    covariance = np.atleast_2d(covariance)
    jitter_cov = 1e-4 * covariance + 1e-9 * np.eye(particles.shape[1])
    jitter = rng.multivariate_normal(
        np.zeros(particles.shape[1]),
        jitter_cov,
        size=particles.shape[0],
    )
    return particles + jitter


def _weighted_covariance(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = np.average(particles, axis=0, weights=weights)
    centered = particles - mean
    return (centered * weights[:, None]).T @ centered


def _provenance(fixture: RangeBearingFixture) -> dict[str, Any]:
    return {
        "fixture_generation_seed": fixture.fixture_generation_seed,
        "fixture_target": fixture.target,
        "algorithm_source": "clean_room_specification",
        "student_code_imported": False,
    }
