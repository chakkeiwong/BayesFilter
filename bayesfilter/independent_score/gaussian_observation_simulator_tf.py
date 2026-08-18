"""Exact full-path Gaussian observation simulator for ratio-score calibration."""

from __future__ import annotations

import tensorflow as tf


DTYPE = tf.float64
OBSERVATION_DIMENSION = 9
PARAMETER_DIMENSION = 3
ORACLE_SEED = 91823


def _directions(horizon: int) -> tuple[tf.Tensor, tf.Tensor]:
    dimension = int(horizon) * OBSERVATION_DIMENSION
    first = tf.ones([dimension], DTYPE)
    parity = tf.cast(tf.range(dimension) % 2, DTYPE)
    second = 1.0 - 2.0 * parity
    return tf.reshape(first, [int(horizon), OBSERVATION_DIMENSION]), tf.reshape(
        second, [int(horizon), OBSERVATION_DIMENSION]
    )


def simulate_observation_paths_from_noise(
    theta: tf.Tensor, standard_normal_noise: tf.Tensor
) -> tf.Tensor:
    """Generate independent Gaussian paths for the three-parameter exact family."""

    parameters = tf.reshape(tf.cast(tf.convert_to_tensor(theta), DTYPE), [3])
    noise = tf.cast(tf.convert_to_tensor(standard_normal_noise), DTYPE)
    if noise.shape.rank != 3 or noise.shape[1] is None or noise.shape[2] != 9:
        raise ValueError("standard_normal_noise must have shape [batch,T,9]")
    first, second = _directions(int(noise.shape[1]))
    mean = parameters[0] * first + parameters[1] * second
    scale = tf.exp(parameters[2])
    return mean[None, :, :] + scale * noise


def make_compiled_observation_simulator(horizon: int):
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")

    @tf.function(jit_compile=True)
    def run(theta: tf.Tensor, standard_normal_noise: tf.Tensor) -> tf.Tensor:
        return simulate_observation_paths_from_noise(theta, standard_normal_noise)

    return run


def fixed_observed_path(horizon: int) -> tf.Tensor:
    """Return paired prefixes of one fixed standard-normal observation path."""

    if int(horizon) <= 0 or int(horizon) > 50:
        raise ValueError("oracle horizon must lie in [1,50]")
    full = tf.random.stateless_normal(
        [50, OBSERVATION_DIMENSION], [ORACLE_SEED, 73], dtype=DTYPE
    )
    return full[: int(horizon)]


def exact_score(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    """Return the exact marginal observation-density score of this family."""

    parameters = tf.reshape(tf.cast(tf.convert_to_tensor(theta), DTYPE), [3])
    path = tf.cast(tf.convert_to_tensor(observations), DTYPE)
    if path.shape.rank != 2 or path.shape[0] is None or path.shape[1] != 9:
        raise ValueError("observations must have shape [T,9]")
    first, second = _directions(int(path.shape[0]))
    mean = parameters[0] * first + parameters[1] * second
    residual = path - mean
    inverse_variance = tf.exp(-2.0 * parameters[2])
    dimension = tf.cast(tf.size(path), DTYPE)
    return tf.stack(
        [
            inverse_variance * tf.reduce_sum(first * residual),
            inverse_variance * tf.reduce_sum(second * residual),
            -dimension + inverse_variance * tf.reduce_sum(tf.square(residual)),
        ]
    )


__all__ = [
    "OBSERVATION_DIMENSION",
    "PARAMETER_DIMENSION",
    "exact_score",
    "fixed_observed_path",
    "make_compiled_observation_simulator",
    "simulate_observation_paths_from_noise",
]
