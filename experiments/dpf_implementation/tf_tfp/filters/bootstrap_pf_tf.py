"""TensorFlow bootstrap particle filter for experimental OT-DPF diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple
from functools import lru_cache

import tensorflow as tf

from bayesfilter.ops.stateless_random_tf import stateless_categorical_cpu_stream


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


class BootstrapFilterTensors(NamedTuple):
    log_likelihood: tf.Tensor
    filtered_means: tf.Tensor
    filtered_variances: tf.Tensor
    ess: tf.Tensor
    resampled: tf.Tensor
    finite: tf.Tensor


@lru_cache(maxsize=8)
def make_bootstrap_particle_filter_tf(
    observation_spec: tf.TensorSpec, *, initial_sample, transition_sample,
    observation_log_density, seed: int, num_particles: int,
    ess_threshold_ratio: float = 0.5, jit_compile: bool = True,
):
    """Bind one complete native filter with a stable observation signature.

    Callbacks must accept a tensor date and consist of TensorFlow operations.
    Non-JIT execution is an explicit diagnostic option. The bounded cache keeps
    repeated calls on the same configured route from rebuilding its graph.
    """
    horizon = observation_spec.shape[0]
    if horizon is None or horizon < 1:
        raise ValueError("bootstrap filter requires a static positive horizon")

    def evaluate(observations):
        observations = tf.cast(observations, DTYPE)
        particles = tf.cast(initial_sample(num_particles, seed), DTYPE)
        log_uniform = -tf.math.log(tf.cast(num_particles, DTYPE))
        log_weights = tf.fill([num_particles], log_uniform)
        means = tf.TensorArray(DTYPE, horizon, element_shape=particles.shape[1:])
        variances = tf.TensorArray(DTYPE, horizon, element_shape=particles.shape[1:])
        esses = tf.TensorArray(DTYPE, horizon, element_shape=[])
        resampled = tf.TensorArray(tf.bool, horizon, element_shape=[])

        def body(t, particles, log_weights, total, means, variances, esses, resampled):
            particles = tf.cast(transition_sample(particles, seed, t), DTYPE)
            observation_weights = tf.cast(observation_log_density(particles, observations[t], t), DTYPE)
            weights, increment = normalize_log_weights_tf(log_weights + observation_weights)
            ess = 1. / tf.reduce_sum(weights * weights)
            mean, variance = weighted_mean_and_variance_tf(particles, weights)
            trigger = ess < ess_threshold_ratio * num_particles
            def resample():
                indices = stateless_categorical_cpu_stream(
                    tf.math.log(tf.maximum(weights, 1e-300)),
                    num_particles, _seed_pair(seed, 7000 + t))
                return tf.gather(particles, indices), tf.fill([num_particles], log_uniform)
            particles, log_weights = tf.cond(trigger, resample,
                lambda: (particles, tf.math.log(tf.maximum(weights, tf.constant(1e-300, DTYPE)))))
            return (t + 1, particles, log_weights, total + increment, means.write(t, mean),
                    variances.write(t, variance), esses.write(t, ess), resampled.write(t, trigger))

        _, _, _, total, means, variances, esses, resampled = tf.while_loop(
            lambda t, *_: t < horizon, body,
            (tf.constant(0), particles, log_weights, tf.constant(0., DTYPE), means, variances, esses, resampled),
            maximum_iterations=horizon, parallel_iterations=1)
        means, variances, esses = means.stack(), variances.stack(), esses.stack()
        finite = (tf.math.is_finite(total) & tf.reduce_all(tf.math.is_finite(means))
                  & tf.reduce_all(tf.math.is_finite(variances)) & tf.reduce_all(tf.math.is_finite(esses)))
        return BootstrapFilterTensors(total, means, variances, esses, resampled.stack(), finite)
    return tf.function(evaluate, input_signature=[observation_spec], jit_compile=jit_compile, autograph=False)


def run_bootstrap_particle_filter_tf(
    *, observations: tf.Tensor, initial_sample: Callable[[int, int], tf.Tensor],
    transition_sample: Callable[[tf.Tensor, int, int], tf.Tensor],
    observation_log_density: Callable[[tf.Tensor, tf.Tensor, int], tf.Tensor],
    seed: int, num_particles: int, ess_threshold_ratio: float = 0.5,
    method_id: str = "bootstrap_multinomial_pf_tf", jit_compile: bool = True,
) -> ParticleFilterTFResult:
    """Run the complete XLA filter, then materialize diagnostic records."""
    observations = tf.convert_to_tensor(observations, DTYPE)
    call = make_bootstrap_particle_filter_tf(tf.TensorSpec(observations.shape, observations.dtype),
        initial_sample=initial_sample, transition_sample=transition_sample,
        observation_log_density=observation_log_density, seed=seed, num_particles=num_particles,
        ess_threshold_ratio=ess_threshold_ratio, jit_compile=jit_compile)
    result = call(observations)
    esses, triggers = result.ess.numpy().tolist(), result.resampled.numpy().tolist()
    # Reporting only; numerical ESS selection happens inside the native loop.
    diagnostics = [dict(time_index=t, ess=ess, ess_ratio=ess / num_particles,
                        resampled=trigger, resampling_method="stateless_multinomial" if trigger else "none",
                        backend="tensorflow", jit_compile=bool(jit_compile))
                   for t, (ess, trigger) in enumerate(zip(esses, triggers))]
    return ParticleFilterTFResult(method_id, int(seed), int(num_particles), result.log_likelihood,
        result.filtered_means, result.filtered_variances, result.ess,
        int(tf.reduce_sum(tf.cast(result.resampled, tf.int32)).numpy()), diagnostics,
        bool(result.finite.numpy()))


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
    return tf.stack((tf.cast(seed % 2147483647, tf.int32), tf.cast(salt % 2147483647, tf.int32)))
