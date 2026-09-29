"""TensorFlow bootstrap particle filter for experimental OT-DPF diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import tensorflow as tf


DTYPE = tf.float64


@dataclass(frozen=True)
class ParticleFilterTFResult:
    method_id: str
    seed: int
    num_particles: int
    log_likelihood_estimate: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess_by_time: tf.Tensor
    resampling_count: int
    resampling_diagnostics: list[dict[str, Any]]
    finite: bool


def run_bootstrap_particle_filter_tf(
    *,
    observations: tf.Tensor,
    initial_sample: Callable[[int, int], tf.Tensor],
    transition_sample: Callable[[tf.Tensor, int, int], tf.Tensor],
    observation_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    seed: int,
    num_particles: int,
    ess_threshold_ratio: float = 0.5,
    method_id: str = "bootstrap_multinomial_pf_tf",
) -> ParticleFilterTFResult:
    particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    filtered_means = []
    filtered_variances = []
    ess_by_time = []
    diagnostics: list[dict[str, Any]] = []
    resampling_count = 0
    log_likelihood = tf.constant(0.0, dtype=DTYPE)

    for t, observation in enumerate(tf.unstack(tf.cast(observations, DTYPE), axis=0)):
        particles = tf.cast(transition_sample(particles, seed, t), DTYPE)
        obs_log_weights = tf.cast(observation_log_density(particles, observation, t), DTYPE)
        weights, incremental = normalize_log_weights_tf(log_weights + obs_log_weights)
        log_likelihood = log_likelihood + incremental
        ess = 1.0 / tf.reduce_sum(weights * weights)
        mean, variance = weighted_mean_and_variance_tf(particles, weights)
        filtered_means.append(mean)
        filtered_variances.append(variance)
        ess_by_time.append(ess)
        do_resample = bool((ess < ess_threshold_ratio * num_particles).numpy())
        if do_resample:
            indices = tf.random.stateless_categorical(
                tf.reshape(tf.math.log(tf.maximum(weights, 1e-300)), [1, -1]),
                num_particles,
                seed=_seed_pair(seed, 7000 + t),
                dtype=tf.int32,
            )[0]
            particles = tf.gather(particles, indices)
            log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
            resampling_count += 1
        else:
            log_weights = tf.math.log(tf.maximum(weights, tf.constant(1e-300, dtype=DTYPE)))
        diagnostics.append(
            {
                "time_index": int(t),
                "ess": float(ess.numpy()),
                "ess_ratio": float((ess / tf.cast(num_particles, DTYPE)).numpy()),
                "resampled": bool(do_resample),
                "resampling_method": "stateless_multinomial" if do_resample else "none",
                "backend": "tensorflow",
            }
        )

    filtered_means_tensor = tf.stack(filtered_means, axis=0)
    filtered_variances_tensor = tf.stack(filtered_variances, axis=0)
    ess_tensor = tf.stack(ess_by_time, axis=0)
    finite = bool(
        tf.math.is_finite(log_likelihood).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_means_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(filtered_variances_tensor)).numpy()
        and tf.reduce_all(tf.math.is_finite(ess_tensor)).numpy()
    )
    return ParticleFilterTFResult(
        method_id=method_id,
        seed=int(seed),
        num_particles=int(num_particles),
        log_likelihood_estimate=log_likelihood,
        filtered_means=filtered_means_tensor,
        filtered_variances=filtered_variances_tensor,
        ess_by_time=ess_tensor,
        resampling_count=int(resampling_count),
        resampling_diagnostics=diagnostics,
        finite=finite,
    )


def normalize_log_weights_tf(log_weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    normalizer = tf.reduce_logsumexp(tf.cast(log_weights, DTYPE))
    weights = tf.exp(tf.cast(log_weights, DTYPE) - normalizer)
    total = tf.reduce_sum(weights)
    return weights / total, normalizer


def weighted_mean_and_variance_tf(
    particles: tf.Tensor,
    weights: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    particles = tf.cast(particles, DTYPE)
    weights = tf.cast(weights, DTYPE)
    mean = tf.reduce_sum(weights[:, None] * particles, axis=0)
    centered = particles - mean
    variance = tf.reduce_sum(weights[:, None] * centered * centered, axis=0)
    return mean, variance


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)
