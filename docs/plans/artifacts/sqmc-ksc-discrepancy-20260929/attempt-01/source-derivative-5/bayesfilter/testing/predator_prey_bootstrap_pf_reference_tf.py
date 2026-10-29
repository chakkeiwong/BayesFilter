"""Vectorized stateless bootstrap-PF reference for predator-prey P4."""

from __future__ import annotations

import math
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    _INITIAL_MEAN,
    _LOG_TWO_PI,
    _PROCESS_COVARIANCE,
    _RK4_INTERNAL_STEP,
    _RK4_SUBSTEPS,
    _rank2_theta,
    rk4_transition_value,
)


def predator_prey_bootstrap_pf_reference(
    theta: Any,
    *,
    observations: tf.Tensor,
    seeds: tf.Tensor,
    num_particles: int,
) -> Mapping[str, tf.Tensor]:
    """Run independent stateless bootstrap filters for one source point."""

    source = _rank2_theta(theta)
    if source.shape[0] != 1:
        raise ValueError("PF reference accepts one parameter point per call")
    y = tf.convert_to_tensor(observations, tf.float64)
    seed_values = tf.reshape(tf.convert_to_tensor(seeds, tf.int32), (-1,))
    seed_count = seed_values.shape[0]
    if seed_count is None:
        raise ValueError("PF reference requires a static seed count")
    seed_count = int(seed_count)
    particle_count = int(num_particles)
    if particle_count <= 0:
        raise ValueError("num_particles must be positive")

    repeated_theta = tf.broadcast_to(source, [seed_count, 6])
    initial_noise = tf.map_fn(
        lambda seed: tf.random.stateless_normal(
            [particle_count, 2],
            seed=tf.stack((seed, tf.constant(101, tf.int32))),
            dtype=tf.float64,
        ),
        seed_values,
        fn_output_signature=tf.TensorSpec([particle_count, 2], tf.float64),
    )
    particles = _INITIAL_MEAN[None, None, :] + initial_noise
    log_likelihood = tf.zeros([seed_count], tf.float64)
    minimum_ess = tf.fill([seed_count], tf.cast(particle_count, tf.float64))
    minimum_state = tf.reduce_min(particles, axis=[1, 2])
    finite = tf.ones([seed_count], tf.bool)

    def condition(index, *_loop_values):
        return index < tf.shape(y)[0]

    def body(index, current_particles, current_log_likelihood, current_minimum_ess, current_minimum_state, current_finite):
        def propagate() -> tf.Tensor:
            means = rk4_transition_value(repeated_theta, current_particles)
            process_noise = tf.map_fn(
                lambda seed: tf.random.stateless_normal(
                    [particle_count, 2],
                    seed=tf.stack((seed, tf.constant(1000, tf.int32) + index)),
                    dtype=tf.float64,
                ),
                seed_values,
                fn_output_signature=tf.TensorSpec([particle_count, 2], tf.float64),
            )
            return means + 2.0 * process_noise

        proposed_particles = tf.cond(
            index > 0, propagate, lambda: current_particles
        )
        residual = y[index][None, None, :] - proposed_particles
        log_weights = -0.5 * (
            2.0 * _LOG_TWO_PI
            + tf.constant(math.log(16.0), tf.float64)
            + tf.reduce_sum(tf.square(residual) / 4.0, axis=2)
        )
        log_normalizer = tf.reduce_logsumexp(log_weights, axis=1)
        increment = log_normalizer - tf.math.log(
            tf.cast(particle_count, tf.float64)
        )
        normalized_weights = tf.exp(log_weights - log_normalizer[:, None])
        ess = tf.math.reciprocal(
            tf.reduce_sum(tf.square(normalized_weights), axis=1)
        )
        resampling_seeds = tf.stack(
            (seed_values, tf.fill([seed_count], tf.constant(3000, tf.int32) + index)),
            axis=1,
        )
        uniforms = tf.map_fn(
            lambda seed: tf.random.stateless_uniform(
                [particle_count], seed=seed, dtype=tf.float64
            ),
            resampling_seeds,
            fn_output_signature=tf.TensorSpec([particle_count], tf.float64),
        )
        cumulative_weights = tf.math.cumsum(normalized_weights, axis=1)
        cumulative_weights = tf.concat(
            (cumulative_weights[:, :-1], tf.ones([seed_count, 1], tf.float64)),
            axis=1,
        )
        indices = tf.searchsorted(
            cumulative_weights, uniforms, side="right", out_type=tf.int32
        )
        resampled = tf.gather(proposed_particles, indices, batch_dims=1)
        step_finite = tf.logical_and(
            tf.math.is_finite(increment),
            tf.reduce_all(tf.math.is_finite(resampled), axis=[1, 2]),
        )
        return (
            index + 1,
            resampled,
            current_log_likelihood + increment,
            tf.minimum(current_minimum_ess, ess),
            tf.minimum(current_minimum_state, tf.reduce_min(proposed_particles, axis=[1, 2])),
            tf.logical_and(current_finite, step_finite),
        )

    result = tf.while_loop(
        condition,
        body,
        (
            tf.constant(0, tf.int32),
            particles,
            log_likelihood,
            minimum_ess,
            minimum_state,
            finite,
        ),
        parallel_iterations=1,
    )
    return {
        "log_likelihood": result[2],
        "minimum_ess": result[3],
        "minimum_state": result[4],
        "finite": result[5],
        "resampling_count": tf.fill([seed_count], tf.shape(y)[0]),
        "num_particles": tf.fill([seed_count], particle_count),
        "seed": seed_values,
    }
