"""Simulation-supervised guide selection for the continuous SIR consumer law.

Complete-data scores are fixed-path training labels, not filter scores.
This research component issues no canonical admission or tuning identity.
"""

import math

import tensorflow as tf

from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec


def moment_features(states):
    """First and raw upper-triangular second moments, with a leading batch."""
    dimension = states.shape[-1]
    indices = [row * dimension + column for row in range(dimension)
               for column in range(row, dimension)]
    products = states[..., :, None] * states[..., None, :]
    flat = tf.reshape(products, tf.concat([tf.shape(states)[:-1], [dimension * dimension]], 0))
    return tf.concat([states, tf.gather(flat, indices, axis=-1)], axis=-1)


def supervised_squared_loss(predictions, labels, scales):
    """Return component losses and validity; scales must be frozen and positive."""
    valid = tf.reduce_all(tf.math.is_finite(predictions))
    valid &= tf.reduce_all(tf.math.is_finite(labels))
    valid &= tf.reduce_all(tf.math.is_finite(scales) & (scales > 0))
    losses = tf.square((predictions - labels) / scales)
    valid &= tf.reduce_all(tf.math.is_finite(losses))
    return losses, valid


def make_sir_supervision_programs(horizon, batch_size, *, dtype=tf.float64,
                                  jit_compile=True):
    """Return batched simulation and fixed-path joint-density/score kernels."""
    if horizon < 1 or batch_size < 1:
        raise ValueError("Positive horizon and batch size are required")
    spec = NonlinearSQMCSpec("sir_d18")
    initial_mean = spec.initial_mean(dtype)
    parameter_signature = tf.TensorSpec([3], dtype)
    path_signature = tf.TensorSpec([batch_size, horizon + 1, 18], dtype)
    observation_signature = tf.TensorSpec([batch_size, horizon, 9], dtype)

    @tf.function(input_signature=[parameter_signature, tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def simulate(theta, seed):
        model, _ = spec.model(theta, tf.zeros([3], dtype))
        states = initial_mean + tf.random.stateless_normal(
            [batch_size, 18], tf.stack([seed, 0]), dtype=dtype)
        process_factor = tf.linalg.cholesky(model.process_covariance)
        observation_factor = tf.linalg.cholesky(model.observation_covariance)
        paths = tf.TensorArray(dtype, horizon + 1).write(0, states)
        observations = tf.TensorArray(dtype, horizon)

        def step(time, current, paths, observations):
            noise = tf.random.stateless_normal(
                [batch_size, 18], tf.stack([seed, 2 * time + 1]), dtype=dtype)
            current = model.transition_mean_fn(theta, current) + tf.linalg.matmul(
                noise, process_factor, transpose_b=True)
            observed = model.observation_fn(current) + tf.linalg.matmul(
                tf.random.stateless_normal([batch_size, 9],
                    tf.stack([seed, 2 * time + 2]), dtype=dtype),
                observation_factor, transpose_b=True)
            return time + 1, current, paths.write(time + 1, current), observations.write(time, observed)

        _, _, paths, observations = tf.while_loop(
            lambda time, *_: time < horizon, step,
            (tf.constant(0), states, paths, observations), parallel_iterations=1)
        return tf.transpose(paths.stack(), [1, 0, 2]), tf.transpose(observations.stack(), [1, 0, 2])

    @tf.function(input_signature=[parameter_signature, path_signature, observation_signature],
                 jit_compile=jit_compile, autograph=False)
    def label(theta, paths, observations):
        previous = tf.reshape(paths[:, :-1], [-1, 18])
        following = tf.reshape(paths[:, 1:], [-1, 18])
        observed = tf.reshape(observations, [-1, 9])
        model, _ = spec.model(theta, tf.zeros([3], dtype))
        residual = following - model.transition_mean_fn(theta, previous)
        log_joint = -0.5 * tf.reduce_sum(tf.square(paths[:, 0] - initial_mean), axis=-1)
        log_joint -= tf.constant(9 * math.log(2 * math.pi), dtype)
        increments = -0.5 * tf.reduce_sum(residual * residual, axis=-1)
        increments -= tf.constant(9 * math.log(2 * math.pi), dtype)
        observation_residual = observed - model.observation_fn(following)
        observation_variance = model.observation_covariance[0, 0]
        observation_square = tf.reduce_sum(tf.square(observation_residual), axis=-1)
        increments += -0.5 * (observation_square / observation_variance
                              + 9 * tf.math.log(2 * math.pi * observation_variance))
        log_joint += tf.reduce_sum(tf.reshape(increments, [batch_size, horizon]), axis=-1)
        components = []
        for coordinate in range(3):
            directional, _ = spec.model(theta, tf.one_hot(coordinate, 3, dtype=dtype))
            tangent = directional.transition_mean_tangent_fn(theta, previous, tf.zeros_like(previous))
            contribution = tf.reduce_sum(residual * tangent, axis=-1)
            if coordinate == 2:
                contribution += observation_square / observation_variance - 9
            components.append(tf.reduce_sum(tf.reshape(contribution, [batch_size, horizon]), axis=-1))
        score = tf.stack(components, axis=-1)
        valid = tf.reduce_all(tf.math.is_finite(score)) & tf.reduce_all(tf.math.is_finite(log_joint))
        return dict(log_joint=log_joint, score=score, valid=valid)

    return simulate, label


def validate_seed_partitions(partitions):
    """Reject reused observation seeds across or within data partitions."""
    seen = set()
    for name, seeds in partitions.items():
        if not seeds or any(not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"Invalid seed partition: {name}")
        if len(set(seeds)) != len(seeds) or seen.intersection(seeds):
            raise ValueError("Observation seed partitions must be disjoint")
        seen.update(seeds)
