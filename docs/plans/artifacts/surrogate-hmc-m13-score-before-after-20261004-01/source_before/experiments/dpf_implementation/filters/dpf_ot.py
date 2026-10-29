"""Finite-Sinkhorn relaxed OT-DPF core for experimental diagnostics."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from experiments.dpf_implementation.filters.bootstrap_pf import (
    ParticleFilterResult,
    normalize_log_weights,
    weighted_mean_and_variance,
)
from experiments.dpf_implementation.resampling.sinkhorn import sinkhorn_resample


Array = np.ndarray


def run_ot_dpf(
    *,
    observations: Array,
    initial_sample: Callable[[np.random.Generator, int], Array],
    transition_sample: Callable[[np.random.Generator, Array, int], Array],
    observation_log_density: Callable[[Array, Array, int], Array],
    seed: int,
    num_particles: int,
    ess_threshold_ratio: float = 0.5,
    sinkhorn_epsilon: float = 0.5,
    sinkhorn_iterations: int = 80,
    sinkhorn_tolerance: float = 1e-7,
) -> ParticleFilterResult:
    """Run a bootstrap proposal with finite Sinkhorn relaxed resampling."""

    rng = np.random.default_rng(seed)
    particles = np.asarray(initial_sample(rng, num_particles), dtype=np.float64)
    log_weights = np.full(num_particles, -math.log(num_particles), dtype=np.float64)
    filtered_means: list[Array] = []
    filtered_variances: list[Array] = []
    ess_by_time: list[float] = []
    resampling_count = 0
    log_likelihood = 0.0
    diagnostics: list[dict[str, float | int | str | bool]] = []

    for t, observation in enumerate(observations):
        if t > 0:
            particles = np.asarray(transition_sample(rng, particles, t), dtype=np.float64)
        obs_log_weights = np.asarray(
            observation_log_density(particles, observation, t), dtype=np.float64
        )
        weights, incremental = normalize_log_weights(log_weights + obs_log_weights)
        log_likelihood += incremental
        ess = float(1.0 / np.sum(weights * weights))
        mean, variance = weighted_mean_and_variance(particles, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        ess_by_time.append(ess)
        if ess < ess_threshold_ratio * num_particles:
            resampled = sinkhorn_resample(
                particles,
                weights,
                epsilon=sinkhorn_epsilon,
                max_iterations=sinkhorn_iterations,
                tolerance=sinkhorn_tolerance,
            )
            particles = resampled.particles
            log_weights = np.full(num_particles, -math.log(num_particles), dtype=np.float64)
            resampling_count += 1
            diag = {
                "time_index": int(t),
                "ess": ess,
                "ess_ratio": float(ess / num_particles),
                "resampled": True,
                "resampling_method": "finite_sinkhorn_relaxed",
                **resampled.diagnostics,
            }
        else:
            log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
            diag = {
                "time_index": int(t),
                "ess": ess,
                "ess_ratio": float(ess / num_particles),
                "resampled": False,
                "resampling_method": "none",
            }
        diagnostics.append(diag)

    filtered_means_array = np.asarray(filtered_means, dtype=np.float64)
    filtered_variances_array = np.asarray(filtered_variances, dtype=np.float64)
    ess_array = np.asarray(ess_by_time, dtype=np.float64)
    finite = bool(
        np.isfinite(log_likelihood)
        and np.all(np.isfinite(filtered_means_array))
        and np.all(np.isfinite(filtered_variances_array))
        and np.all(np.isfinite(ess_array))
    )
    return ParticleFilterResult(
        method_id="ot_dpf_finite_sinkhorn_relaxed",
        seed=int(seed),
        num_particles=int(num_particles),
        log_likelihood_estimate=float(log_likelihood),
        filtered_means=filtered_means_array,
        filtered_variances=filtered_variances_array,
        ess_by_time=ess_array,
        resampling_count=int(resampling_count),
        resampling_diagnostics=diagnostics,
        finite=finite,
    )
