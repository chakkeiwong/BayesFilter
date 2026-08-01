from __future__ import annotations

import inspect

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.cubature_genut_filter import (
    CandidateModelAdapter,
    _restore_cloud_jvp,
    _sinkhorn_barycentric_jvp,
    _sinkhorn_barycentric_jvp_core,
    finite_value_score,
)


def _adapter() -> CandidateModelAdapter:
    def initial(theta, noise):
        return tf.exp(theta[0]) * noise

    def initial_tangent(theta, noise):
        tangent = tf.zeros([tf.shape(noise)[0], 1, 2], noise.dtype)
        return tf.concat([noise[:, :, None] * tf.exp(theta[0]), tangent[:, :, 1:]], axis=2)

    def transition(theta, particles, noise, _time):
        return tf.math.tanh(theta[0] * particles + noise)

    def transition_tangent(theta, particles, noise, particle_tangent, _time):
        output = tf.math.tanh(theta[0] * particles + noise)
        inside_tangent = theta[0] * particle_tangent
        inside_tangent += tf.concat(
            [particles[:, :, None], tf.zeros_like(particles[:, :, None])], axis=2
        )
        return (1.0 - tf.square(output))[:, :, None] * inside_tangent

    def observation(theta, particles, observation_value, _time):
        sigma = tf.exp(theta[1])
        residual = observation_value[0] - tf.square(particles[:, 0])
        return -0.5 * (tf.square(residual) / tf.square(sigma) + 2.0 * tf.math.log(sigma))

    def observation_tangent(theta, particles, particle_tangent, observation_value, _time):
        sigma = tf.exp(theta[1])
        residual = observation_value[0] - tf.square(particles[:, 0])
        residual_tangent = -2.0 * particles[:, 0, None] * particle_tangent[:, 0, :]
        tangent = -residual[:, None] * residual_tangent / tf.square(sigma)
        explicit_scale = tf.stack(
            [tf.zeros_like(residual), tf.square(residual) / tf.square(sigma) - 1.0],
            axis=-1,
        )
        return tangent + explicit_scale

    return CandidateModelAdapter(
        state_dimension=1,
        parameter_count=2,
        initial_value=initial,
        initial_tangent=initial_tangent,
        transition_value=transition,
        transition_tangent=transition_tangent,
        observation_value=observation,
        observation_tangent=observation_tangent,
    )


def _value(theta, observations, initial_noise, process_noise, design):
    return finite_value_score(
        _adapter(), theta, observations, initial_noise, process_noise, design
    )[0]


def test_generic_nonlinear_value_and_score_match_same_scalar_fd():
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1], [-0.2], [0.3]], tf.float32)
    initial_noise = tf.random.stateless_normal([12, 1], seed=[41, 42])
    process_noise = tf.random.stateless_normal([3, 12, 1], seed=[43, 44])
    design = cubature_design(dim=1, num_particles=12)
    value, score, diagnostics = finite_value_score(
        _adapter(), theta, observations, initial_noise, process_noise, design
    )
    steps = tf.constant([2.0e-3, 2.0e-3], tf.float32)
    finite_difference = []
    for index in range(2):
        plus = tf.tensor_scatter_nd_add(theta, [[index]], [steps[index]])
        minus = tf.tensor_scatter_nd_sub(theta, [[index]], [steps[index]])
        finite_difference.append(
            (_value(plus, observations, initial_noise, process_noise, design)
             - _value(minus, observations, initial_noise, process_noise, design))
            / (2.0 * steps[index])
        )
    tf.debugging.assert_near(score, tf.stack(finite_difference), atol=4e-3, rtol=4e-3)
    assert np.isfinite(float(value.numpy()))
    assert float(diagnostics["max_mean_residual"].numpy()) < 1e-4
    assert float(diagnostics["max_row_residual"].numpy()) < 1e-2
    assert float(diagnostics["max_col_residual"].numpy()) < 1e-2


def test_sinkhorn_barycentric_jvp_matches_forward_accumulator():
    particles = tf.constant([[-1.2], [-0.3], [0.4], [1.1]], tf.float64)
    weights = tf.constant([0.15, 0.20, 0.30, 0.35], tf.float64)
    particle_direction = tf.constant([[0.2], [-0.1], [0.05], [-0.15]], tf.float64)
    weight_direction = tf.constant([0.01, -0.02, 0.03, -0.02], tf.float64)
    with tf.autodiff.ForwardAccumulator(
        (particles, weights), (particle_direction, weight_direction)
    ) as accumulator:
        barycentric = _sinkhorn_barycentric_jvp(
            particles,
            weights,
            tf.zeros([4, 1, 1], tf.float64),
            tf.zeros([4, 1], tf.float64),
            epsilon=2.0,
            sinkhorn_steps=8,
        )[0]
    automatic = accumulator.jvp(barycentric)
    manual = _sinkhorn_barycentric_jvp(
        particles,
        weights,
        particle_direction[:, :, None],
        weight_direction[:, None],
        epsilon=2.0,
        sinkhorn_steps=8,
    )[1][:, :, 0]
    tf.debugging.assert_near(manual, automatic, atol=1e-10, rtol=1e-10)


def test_realized_row_quotient_stays_inside_source_convex_hull():
    particles = tf.constant([[-4.0], [-0.5], [1.0], [3.0]], tf.float64)
    weights = tf.constant([0.001, 0.002, 0.007, 0.99], tf.float64)
    result = _sinkhorn_barycentric_jvp_core(
        particles,
        weights,
        tf.zeros([4, 1, 1], tf.float64),
        tf.zeros([4, 1], tf.float64),
        epsilon=0.1,
        sinkhorn_steps=1,
        balance_steps=0,
    )
    quotient = result["particles"][:, 0]
    old_substitution = 4.0 * tf.linalg.matvec(result["coupling"], particles[:, 0])
    tf.debugging.assert_greater_equal(quotient, tf.reduce_min(particles))
    tf.debugging.assert_less_equal(quotient, tf.reduce_max(particles))
    assert bool(
        tf.reduce_any(
            (old_substitution < tf.reduce_min(particles))
            | (old_substitution > tf.reduce_max(particles))
        ).numpy()
    )


def test_terminal_balance_reduces_post_quotient_column_error():
    particles = tf.constant([[-4.0], [-0.5], [1.0], [3.0]], tf.float64)
    weights = tf.constant([0.001, 0.002, 0.007, 0.99], tf.float64)
    zeros = tf.zeros([4, 1, 1], tf.float64)
    zero_weights = tf.zeros([4, 1], tf.float64)
    unbalanced = _sinkhorn_barycentric_jvp_core(
        particles,
        weights,
        zeros,
        zero_weights,
        epsilon=0.1,
        sinkhorn_steps=1,
        balance_steps=0,
    )
    balanced = _sinkhorn_barycentric_jvp_core(
        particles,
        weights,
        zeros,
        zero_weights,
        epsilon=0.1,
        sinkhorn_steps=1,
        balance_steps=64,
    )
    tf.debugging.assert_less(
        balanced["post_quotient_column_tv_error"],
        unbalanced["post_quotient_column_tv_error"],
    )
    tf.debugging.assert_positive(balanced["minimum_row_mass"])


def test_composed_restore_jvp_matches_forward_accumulator():
    particles = tf.constant([[-1.2], [-0.3], [0.4], [1.1]], tf.float64)
    weights = tf.constant([0.15, 0.20, 0.30, 0.35], tf.float64)
    design = tf.constant([[-1.0], [1.0], [-1.0], [1.0]], tf.float64)
    particle_direction = tf.constant([[0.1], [-0.08], [0.03], [-0.05]], tf.float64)
    weight_direction = tf.constant([0.01, -0.02, 0.03, -0.02], tf.float64)
    with tf.autodiff.ForwardAccumulator(
        (particles, weights), (particle_direction, weight_direction)
    ) as accumulator:
        restored = _restore_cloud_jvp(
            particles,
            weights,
            tf.zeros([4, 1, 1], tf.float64),
            tf.zeros([4, 1], tf.float64),
            design,
            epsilon=2.0,
            sinkhorn_steps=8,
            ridge=1.0e-4,
            parameter_count=1,
        )[0]
    automatic = accumulator.jvp(restored)
    manual = _restore_cloud_jvp(
        particles,
        weights,
        particle_direction[:, :, None],
        weight_direction[:, None],
        design,
        epsilon=2.0,
        sinkhorn_steps=8,
        ridge=1.0e-4,
        parameter_count=1,
    )[1][:, :, 0]
    tf.debugging.assert_near(manual, automatic, atol=1e-9, rtol=1e-9)


def test_generic_core_replays_and_has_no_autodiff_or_runtime_fd():
    source = inspect.getsource(finite_value_score)
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source


def test_invalid_finite_program_is_returned_as_nonfinite_not_consumed():
    theta = tf.constant([0.2, -0.1], tf.float32)
    observations = tf.constant([[0.1]], tf.float32)
    initial_noise = tf.random.stateless_normal([12, 1], seed=[51, 52])
    process_noise = tf.random.stateless_normal([1, 12, 1], seed=[53, 54])
    design = cubature_design(dim=1, num_particles=12)
    value, score, diagnostics = finite_value_score(
        _adapter(),
        theta,
        observations,
        initial_noise,
        process_noise,
        design,
        epsilon=0.01,
        sinkhorn_steps=1,
        balance_steps=0,
    )
    if not bool(diagnostics["program_valid"].numpy()):
        assert not bool(tf.math.is_finite(value).numpy())
        assert not bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_candidate_xla_core_has_no_python_loop_or_host_numeric_path():
    source = inspect.getsource(finite_value_score)
    assert "for " not in source
    assert "while " not in source
    assert ".numpy(" not in source
    assert "numpy" not in source
    assert "np." not in source
    assert "finite_difference" not in source
