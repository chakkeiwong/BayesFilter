from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    exact_transformed_sv_candidate_adapter,
    parameterized_austria_sir_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.cubature_genut_filter import (
    CandidateModelAdapter,
    finite_value_score,
)


def _two_dimensional_adapter():
    def initial(theta, noise):
        return noise + theta[0]

    def initial_tangent(theta, noise):
        return tf.stack([tf.ones_like(noise), tf.zeros_like(noise)], axis=-1)

    def transition(theta, particles, noise, _time):
        return tf.tanh(particles + 0.1 * noise + theta[1])

    def transition_tangent(theta, particles, noise, tangent, _time):
        output = tf.tanh(particles + 0.1 * noise + theta[1])
        return (1.0 - tf.square(output))[:, :, None] * (
            tangent + tf.concat(
                [tf.zeros_like(particles[:, :, None]), tf.ones_like(particles[:, :, None])],
                axis=2,
            )
        )

    def observation(theta, particles, observation_value, _time):
        residual = observation_value - particles
        return -0.5 * tf.reduce_sum(tf.square(residual), axis=1)

    def observation_tangent(theta, particles, tangent, observation_value, _time):
        residual = observation_value - particles
        return tf.reduce_sum(residual[:, :, None] * tangent, axis=1)

    return CandidateModelAdapter(
        state_dimension=2,
        parameter_count=2,
        initial_value=initial,
        initial_tangent=initial_tangent,
        transition_value=transition,
        transition_tangent=transition_tangent,
        observation_value=observation,
        observation_tangent=observation_tangent,
    )


def test_two_dimensional_candidate_replays_and_scores():
    adapter = _two_dimensional_adapter()
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1, -0.2], [0.3, 0.4]], tf.float32)
    initial_noise = tf.random.stateless_normal([12, 2], seed=[61, 62])
    process_noise = tf.random.stateless_normal([2, 12, 2], seed=[63, 64])
    design = cubature_design(dim=2, num_particles=12)
    value, score, diagnostics = finite_value_score(
        adapter, theta, observations, initial_noise, process_noise, design
    )
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    assert float(diagnostics["max_mean_residual"].numpy()) < 1e-3


def test_exact_transformed_sv_pilot_has_finite_same_scalar_score():
    adapter = exact_transformed_sv_candidate_adapter()
    theta = tf.constant([0.25, -0.15], tf.float32)
    observations = tf.constant([[0.2], [-0.1], [0.3], [-0.4]], tf.float32)
    initial_noise = tf.random.stateless_normal([12, 1], seed=[51, 52])
    process_noise = tf.random.stateless_normal([4, 12, 1], seed=[53, 54])
    design = cubature_design(dim=1, num_particles=12)
    value, score, diagnostics = finite_value_score(
        adapter, theta, observations, initial_noise, process_noise, design
    )
    steps = tf.constant([2.0e-3, 2.0e-3], tf.float32)
    finite_difference = []
    for index in range(2):
        plus = tf.tensor_scatter_nd_add(theta, [[index]], [steps[index]])
        minus = tf.tensor_scatter_nd_sub(theta, [[index]], [steps[index]])
        finite_difference.append(
            (
                finite_value_score(adapter, plus, observations, initial_noise, process_noise, design)[0]
                - finite_value_score(adapter, minus, observations, initial_noise, process_noise, design)[0]
            ) / (2.0 * steps[index])
        )
    tf.debugging.assert_near(score, tf.stack(finite_difference), atol=6e-3, rtol=6e-3)
    assert np.isfinite(float(value.numpy()))
    assert float(diagnostics["max_mean_residual"].numpy()) < 1e-4


def test_exact_transformed_sv_adapter_has_no_runtime_autodiff_or_fd():
    source = inspect.getsource(exact_transformed_sv_candidate_adapter)
    assert "GradientTape" not in source
    assert "finite_difference" not in source


def test_parameterized_austria_sir_manual_transition_and_observation_tangents():
    adapter = parameterized_austria_sir_candidate_adapter()
    theta = tf.constant([0.03, -0.02, 0.04], tf.float32)
    noise = tf.zeros([2, 18], tf.float32)
    particles = adapter.initial_value(theta, noise)
    tangent = adapter.initial_tangent(theta, noise)
    process = tf.zeros_like(particles)
    transitioned = adapter.transition_value(
        theta, particles, process, tf.constant(0, tf.int32)
    )
    manual_transition = adapter.transition_tangent(
        theta, particles, process, tangent, tf.constant(0, tf.int32)
    )
    observation = transitioned[0, 1::2] + tf.linspace(-0.2, 0.2, 9)
    manual_observation = adapter.observation_tangent(
        theta,
        transitioned,
        manual_transition,
        observation,
        tf.constant(0, tf.int32),
    )

    transition_fd = []
    observation_fd = []
    step = tf.constant(2.0e-3, tf.float32)
    for index in range(3):
        direction = tf.one_hot(index, 3, dtype=tf.float32)
        plus_state = adapter.transition_value(
            theta + step * direction, particles, process, tf.constant(0, tf.int32)
        )
        minus_state = adapter.transition_value(
            theta - step * direction, particles, process, tf.constant(0, tf.int32)
        )
        transition_fd.append((plus_state - minus_state) / (2.0 * step))
        plus_value = adapter.observation_value(
            theta + step * direction,
            plus_state,
            observation,
            tf.constant(0, tf.int32),
        )
        minus_value = adapter.observation_value(
            theta - step * direction,
            minus_state,
            observation,
            tf.constant(0, tf.int32),
        )
        observation_fd.append((plus_value - minus_value) / (2.0 * step))
    tf.debugging.assert_near(
        manual_transition,
        tf.stack(transition_fd, axis=-1),
        atol=2.0e-2,
        rtol=2.0e-3,
    )
    tf.debugging.assert_near(
        manual_observation,
        tf.stack(observation_fd, axis=-1),
        atol=2.0e-2,
        rtol=2.0e-3,
    )


def test_parameterized_austria_sir_adapter_has_no_runtime_autodiff_or_fd():
    source = inspect.getsource(parameterized_austria_sir_candidate_adapter)
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "finite_difference" not in source
