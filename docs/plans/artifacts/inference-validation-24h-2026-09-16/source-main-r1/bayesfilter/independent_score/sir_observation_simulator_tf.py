"""Standalone batched Austria-SIR observation-path simulator.

The module exposes the generative law only. It contains no likelihood,
derivative, latent posterior, or state-estimation implementation.
"""

from __future__ import annotations

import tensorflow as tf


DTYPE = tf.float64
STATE_DIMENSION = 18
OBSERVATION_DIMENSION = 9
PARAMETER_DIMENSION = 3
SUBSTEPS = 4
STEP = tf.constant(0.005, DTYPE)
INITIAL_MEAN = tf.constant(
    [
        487.0, 13.0, 488.0, 12.0, 489.0, 11.0, 490.0, 10.0, 491.0,
        9.0, 492.0, 8.0, 493.0, 7.0, 494.0, 6.0, 495.0, 5.0,
    ],
    DTYPE,
)
BASE_KAPPA = tf.fill([OBSERVATION_DIMENSION], tf.constant(0.1, DTYPE))
BASE_NU = tf.fill([OBSERVATION_DIMENSION], tf.constant(18.0, DTYPE))
BASE_OBSERVATION_SCALE = tf.constant(10.0, DTYPE)
ADJACENCY = tf.constant(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
    ],
    DTYPE,
)
NEIGHBOR_DEGREE = tf.reduce_sum(ADJACENCY, axis=1)


def _rhs(state: tf.Tensor, kappa: tf.Tensor, nu: tf.Tensor) -> tf.Tensor:
    susceptible = state[:, 0::2]
    infectious = state[:, 1::2]
    susceptible_neighbor = (
        tf.linalg.matmul(susceptible, ADJACENCY, transpose_b=True)
        - susceptible * NEIGHBOR_DEGREE[None, :]
    )
    infectious_neighbor = (
        tf.linalg.matmul(infectious, ADJACENCY, transpose_b=True)
        - infectious * NEIGHBOR_DEGREE[None, :]
    )
    infection = kappa[None, :] * susceptible * infectious
    derivative_s = -infection + 0.5 * susceptible_neighbor
    derivative_i = infection - nu[None, :] * infectious + 0.5 * infectious_neighbor
    return tf.reshape(tf.stack([derivative_s, derivative_i], axis=2), tf.shape(state))


def _transition_mean(state: tf.Tensor, kappa: tf.Tensor, nu: tf.Tensor) -> tf.Tensor:
    current = state
    for _ in range(SUBSTEPS):
        k1 = _rhs(current, kappa, nu)
        k2 = _rhs(current + 0.5 * STEP * k1, kappa, nu)
        k3 = _rhs(current + 0.5 * STEP * k2, kappa, nu)
        k4 = _rhs(current + 0.5 * STEP * k3, kappa, nu)
        current = current + (STEP / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def simulate_observation_paths_from_noise(
    theta: tf.Tensor,
    initial_noise: tf.Tensor,
    transition_noise: tf.Tensor,
    observation_noise: tf.Tensor,
) -> tf.Tensor:
    """Generate `y_1:y_T` for batched standard-normal noise banks."""

    parameters = tf.reshape(
        tf.cast(tf.convert_to_tensor(theta), DTYPE), [PARAMETER_DIMENSION]
    )
    initial_noise = tf.cast(tf.convert_to_tensor(initial_noise), DTYPE)
    transition_noise = tf.cast(tf.convert_to_tensor(transition_noise), DTYPE)
    observation_noise = tf.cast(tf.convert_to_tensor(observation_noise), DTYPE)
    if initial_noise.shape.rank != 2 or initial_noise.shape[1] != STATE_DIMENSION:
        raise ValueError("initial_noise must have shape [batch,18]")
    if transition_noise.shape.rank != 3 or transition_noise.shape[2] != STATE_DIMENSION:
        raise ValueError("transition_noise must have shape [batch,T,18]")
    if observation_noise.shape.rank != 3 or observation_noise.shape[2] != OBSERVATION_DIMENSION:
        raise ValueError("observation_noise must have shape [batch,T,9]")
    if transition_noise.shape[1] != observation_noise.shape[1]:
        raise ValueError("transition and observation horizons differ")
    horizon = transition_noise.shape[1]
    if horizon is None:
        raise ValueError("simulation horizon must be static")
    kappa = BASE_KAPPA * tf.exp(parameters[0])
    nu = BASE_NU * tf.exp(parameters[1])
    observation_scale = BASE_OBSERVATION_SCALE * tf.exp(parameters[2])
    state = INITIAL_MEAN[None, :] + initial_noise
    outputs = []
    for time_index in range(int(horizon)):
        latent = _transition_mean(state, kappa, nu) + transition_noise[:, time_index, :]
        susceptible = tf.maximum(latent[:, 0::2], 0.0)
        infectious = latent[:, 1::2]
        state = tf.reshape(tf.stack([susceptible, infectious], axis=2), tf.shape(latent))
        outputs.append(
            infectious + observation_scale * observation_noise[:, time_index, :]
        )
    return tf.stack(outputs, axis=1)


def make_compiled_observation_simulator(horizon: int):
    """Return the batch-native XLA generative program for one static horizon."""

    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")

    @tf.function(jit_compile=True)
    def run(
        theta: tf.Tensor,
        initial_noise: tf.Tensor,
        transition_noise: tf.Tensor,
        observation_noise: tf.Tensor,
    ) -> tf.Tensor:
        return simulate_observation_paths_from_noise(
            theta, initial_noise, transition_noise, observation_noise
        )

    return run


def fixed_observed_path(seed: int = 81120, horizon: int = 50) -> tf.Tensor:
    """Reproduce the source simulator's sequential RNG order and return `y_1:y_T`."""

    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    generator = tf.random.Generator.from_seed(int(seed))
    state = INITIAL_MEAN + generator.normal([STATE_DIMENSION], dtype=DTYPE)
    generator.normal([OBSERVATION_DIMENSION], dtype=DTYPE)  # source y_0 draw
    observations = []
    for _ in range(int(horizon)):
        latent = _transition_mean(state[None, :], BASE_KAPPA, BASE_NU)[0]
        latent += generator.normal([STATE_DIMENSION], dtype=DTYPE)
        state = tf.reshape(
            tf.stack([tf.maximum(latent[0::2], 0.0), latent[1::2]], axis=1),
            [STATE_DIMENSION],
        )
        observations.append(
            state[1::2]
            + BASE_OBSERVATION_SCALE
            * generator.normal([OBSERVATION_DIMENSION], dtype=DTYPE)
        )
    return tf.stack(observations)


__all__ = [
    "OBSERVATION_DIMENSION",
    "PARAMETER_DIMENSION",
    "STATE_DIMENSION",
    "fixed_observed_path",
    "make_compiled_observation_simulator",
    "simulate_observation_paths_from_noise",
]
