"""Independent recurrence and seeded-resampling checks for native PF execution."""

import numpy as np
import pytest
import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    make_bootstrap_particle_filter_tf,
)


def _initial(count, seed):
    del seed
    return tf.reshape(tf.linspace(tf.constant(-1.0, tf.float64), 1.0, count), [-1, 1])


def _transition(particles, seed, date):
    del seed
    return particles * 0.9 + tf.cast(date, tf.float64) * 0.05


def _observation(particles, observation, date):
    del date
    return -tf.square(particles[:, 0] - observation[0]) / 0.06


def _reference(observations, threshold, seed=71, count=16):
    # Independent NumPy diagnostic, with the original TensorFlow categorical
    # draw as the authority for the pre-repair stateless random stream.
    particles = np.linspace(-1.0, 1.0, count)
    weights = np.full(count, 1.0 / count)
    total, means, variances, esses, triggers = 0.0, [], [], [], []
    for date, observation in enumerate(observations[:, 0]):
        particles = particles * 0.9 + date * 0.05
        log_weights = np.log(weights) - np.square(particles - observation) / 0.06
        shift = log_weights.max()
        likelihood = np.exp(log_weights - shift).sum()
        weights = np.exp(log_weights - shift) / likelihood
        total += shift + np.log(likelihood)
        mean = weights @ particles
        means.append([mean])
        variances.append([weights @ np.square(particles - mean)])
        ess = 1.0 / (weights @ weights)
        esses.append(ess)
        trigger = ess < threshold * count
        triggers.append(trigger)
        if trigger:
            indices = tf.random.stateless_categorical(
                tf.constant(np.log(np.maximum(weights, 1e-300))[None, :]), count,
                [seed, 7000 + date], dtype=tf.int32)[0].numpy()
            particles = particles[indices]
            weights = np.full(count, 1.0 / count)
    return total, means, variances, esses, triggers, True


@pytest.mark.parametrize("threshold", [0.0, 0.7, 1.1])
@pytest.mark.parametrize("jit", [False, True])
def test_bootstrap_preserves_fixed_input_and_original_resampling_stream(threshold, jit):
    observations = tf.constant([[0.0], [0.7], [-0.4], [0.1]], tf.float64)
    call = make_bootstrap_particle_filter_tf(tf.TensorSpec(observations.shape, tf.float64),
        initial_sample=_initial, transition_sample=_transition,
        observation_log_density=_observation, seed=71, num_particles=16,
        ess_threshold_ratio=threshold, jit_compile=jit)
    actual = call(observations)
    reference = _reference(observations.numpy(), threshold)
    for value, expected in zip(actual[:4], reference[:4]):
        np.testing.assert_allclose(value.numpy(), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(actual.resampled.numpy(), reference[4])
    assert bool(actual.finite)
    assert call.experimental_get_tracing_count() == 1


def test_bootstrap_native_dates_do_not_unroll_with_horizon():
    def graph_nodes(horizon):
        call = make_bootstrap_particle_filter_tf(tf.TensorSpec([horizon, 1], tf.float64),
            initial_sample=_initial, transition_sample=_transition,
            observation_log_density=_observation, seed=71, num_particles=16)
        graph = call.get_concrete_function().graph.as_graph_def()
        nodes = [*graph.node, *(node for fn in graph.library.function for node in fn.node_def)]
        assert not ({node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"})
        return len(nodes)
    assert graph_nodes(2) == graph_nodes(8)
