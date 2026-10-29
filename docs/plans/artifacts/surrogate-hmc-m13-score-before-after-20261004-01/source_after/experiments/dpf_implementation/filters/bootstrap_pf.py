"""Classical bootstrap PF core for experimental DPF diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ParticleFilterResult:
    method_id: str
    seed: int
    num_particles: int
    log_likelihood_estimate: float
    filtered_means: Array
    filtered_variances: Array
    ess_by_time: Array
    resampling_count: int
    resampling_diagnostics: list[dict[str, float | int | str | bool]]
    finite: bool


def run_bootstrap_particle_filter(
    *,
    observations: Array,
    initial_sample: Callable[[np.random.Generator, int], Array],
    transition_sample: Callable[[np.random.Generator, Array, int], Array],
    observation_log_density: Callable[[Array, Array, int], Array],
    seed: int,
    num_particles: int,
    ess_threshold_ratio: float = 0.5,
    method_id: str = "bootstrap_sir_pf",
) -> ParticleFilterResult:
    """Run a bootstrap/SIR PF with systematic resampling."""

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
            indices = systematic_resample(weights, rng)
            particles = particles[indices]
            log_weights = np.full(num_particles, -math.log(num_particles), dtype=np.float64)
            resampling_count += 1
            did_resample = True
        else:
            log_weights = np.log(np.maximum(weights, np.finfo(np.float64).tiny))
            did_resample = False
        diagnostics.append(
            {
                "time_index": int(t),
                "ess": ess,
                "ess_ratio": float(ess / num_particles),
                "resampled": did_resample,
                "resampling_method": "systematic" if did_resample else "none",
            }
        )

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
        method_id=method_id,
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


def normalize_log_weights(log_weights: Array) -> tuple[Array, float]:
    normalizer = logsumexp(log_weights)
    weights = np.exp(log_weights - normalizer)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("normalized weights have nonpositive total")
    return weights / total, float(normalizer)


def logsumexp(values: Array) -> float:
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        raise FloatingPointError("logsumexp received no finite maximum")
    return float(max_value + math.log(float(np.sum(np.exp(values - max_value)))))


def systematic_resample(weights: Array, rng: np.random.Generator) -> Array:
    n_particles = int(weights.shape[0])
    positions = (rng.random() + np.arange(n_particles)) / n_particles
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def weighted_mean_and_variance(particles: Array, weights: Array) -> tuple[Array, Array]:
    mean = np.sum(weights[:, None] * particles, axis=0)
    centered = particles - mean
    variance = np.sum(weights[:, None] * centered * centered, axis=0)
    return mean, variance
