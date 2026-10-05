"""Frozen source-order exact transformed-SV target for moment-teacher tests."""

from __future__ import annotations

import tensorflow as tf


ACTUAL_SV_DATASET_ID = "source_order_exact_transformed_sv_seed_83120_y1_y20_v1"
ACTUAL_SV_HORIZON = 20
ACTUAL_SV_PHYSICAL_PARAMETERS = (0.6, 0.4, 1.0)


def actual_sv_unconstrained_theta_tf() -> tf.Tensor:
    """Return `(Phi^-1(gamma), log(beta))` for `(0.6, 0.4)`."""

    gamma = tf.constant(ACTUAL_SV_PHYSICAL_PARAMETERS[0], tf.float64)
    z_gamma = tf.sqrt(tf.constant(2.0, tf.float64)) * tf.math.erfinv(
        2.0 * gamma - 1.0
    )
    return tf.stack(
        [z_gamma, tf.math.log(tf.constant(ACTUAL_SV_PHYSICAL_PARAMETERS[1], tf.float64))]
    )


def generate_source_order_actual_sv_dataset_tf(
    *, horizon: int = ACTUAL_SV_HORIZON
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Generate stationary `x0 -> x1 -> y1` exact transformed-SV data."""

    if horizon < 1:
        raise ValueError("actual-SV horizon must be positive")
    gamma, beta, sigma = (
        tf.constant(value, tf.float64) for value in ACTUAL_SV_PHYSICAL_PARAMETERS
    )
    initial_noise = tf.random.stateless_normal(
        [], [83120, 101], dtype=tf.float64
    )
    process_noise = tf.random.stateless_normal(
        [horizon], [83120, 102], dtype=tf.float64
    )
    observation_noise = tf.random.stateless_normal(
        [horizon], [83120, 103], dtype=tf.float64
    )
    x0 = sigma * initial_noise / tf.sqrt(1.0 - tf.square(gamma))
    states = tf.TensorArray(tf.float64, size=horizon, element_shape=())

    def body(index, previous, output):
        current = gamma * previous + sigma * process_noise[index]
        return index + 1, current, output.write(index, current)

    _, _, states = tf.while_loop(
        lambda index, *_: index < horizon,
        body,
        (tf.zeros([], tf.int32), x0, states),
        maximum_iterations=horizon,
        parallel_iterations=1,
    )
    state_path = states.stack()
    raw_observations = beta * tf.exp(0.5 * state_path) * observation_noise
    transformed_observations = tf.math.log(tf.square(raw_observations))
    return (
        state_path[:, None],
        raw_observations[:, None],
        transformed_observations[:, None],
    )


__all__ = [
    "ACTUAL_SV_DATASET_ID",
    "ACTUAL_SV_HORIZON",
    "ACTUAL_SV_PHYSICAL_PARAMETERS",
    "actual_sv_unconstrained_theta_tf",
    "generate_source_order_actual_sv_dataset_tf",
]
