from __future__ import annotations

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import cubature_genut_batch_tf as batch
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.cubature_genut_batch_adapters import (
    parameterized_austria_sir_batch_adapter,
)


def _small_case():
    adapter = parameterized_austria_sir_batch_adapter()
    count = 36
    theta = tf.zeros([1, 3], tf.float32)
    initial_noise = tf.random.stateless_normal([count, 18], [170817, 1])
    process_noise = tf.random.stateless_normal([1, count, 18], [170817, 2])
    observations = tf.zeros([1, 9], tf.float32)
    design = cubature_design(dim=18, num_particles=count)
    kwargs = {
        "epsilon": 8.0,
        "sinkhorn_steps": 2,
        "balance_steps": 2,
        "ridge": 1.0e-5,
        "transition_before_first_observation": True,
        "higher_moment_correction_steps": 0,
        "higher_moment_strength": 0.2,
        "higher_moment_floor": 1.0e-5,
    }
    return adapter, theta, observations, initial_noise, process_noise, design, kwargs


def test_correction_zero_value_endpoints_are_exactly_equal() -> None:
    adapter, theta, observations, initial_noise, process_noise, design, kwargs = _small_case()
    value, value_status = batch.batch_finite_value(
        adapter, theta, observations, initial_noise, process_noise, design, **kwargs
    )
    score_value, _score, score_status = batch.batch_finite_value_score(
        adapter, theta, observations, initial_noise, process_noise, design, **kwargs
    )
    tf.debugging.assert_equal(value_status["program_valid"], score_status["program_valid"])
    tf.debugging.assert_equal(value, score_value)


def test_active_correction_value_endpoints_are_exactly_equal() -> None:
    adapter, theta, observations, initial_noise, process_noise, design, kwargs = _small_case()
    kwargs = kwargs | {
        "higher_moment_correction_steps": 2,
        "higher_moment_strength": 0.1,
    }
    value, value_status = batch.batch_finite_value(
        adapter, theta, observations, initial_noise, process_noise, design, **kwargs
    )
    score_value, _score, score_status = batch.batch_finite_value_score(
        adapter, theta, observations, initial_noise, process_noise, design, **kwargs
    )
    tf.debugging.assert_equal(value_status["program_valid"], score_status["program_valid"])
    tf.debugging.assert_equal(value, score_value)


def test_public_score_traces_in_graph_and_xla() -> None:
    adapter, theta, observations, initial_noise, process_noise, design, kwargs = _small_case()

    def call(values):
        return batch.batch_finite_value_score(
            adapter,
            values,
            observations,
            initial_noise,
            process_noise,
            design,
            **kwargs,
        )

    eager_value, eager_score, eager_status = call(theta)
    for jit_compile in (False, True):
        compiled = tf.function(call, autograph=False, jit_compile=jit_compile)
        value, score, status = compiled(theta)
        tf.debugging.assert_near(value, eager_value, atol=2.0e-5, rtol=2.0e-5)
        tf.debugging.assert_near(score, eager_score, atol=2.0e-5, rtol=2.0e-5)
        tf.debugging.assert_equal(status["program_valid"], eager_status["program_valid"])


def test_zero_tangent_restore_primal_is_exactly_shared() -> None:
    adapter, theta, _observations, initial_noise, process_noise, design, kwargs = _small_case()
    particles = adapter.initial_value(theta, initial_noise)
    particles = adapter.transition_value(theta, particles, process_noise[0], tf.constant(0))
    count = tf.shape(particles)[1]
    weights = tf.fill([1, count], 1.0 / tf.cast(count, tf.float32))
    particle_tangent = tf.zeros([1, count, 18, 3], tf.float32)
    weight_tangent = tf.zeros([1, count, 3], tf.float32)
    value = batch._restore_cloud_batch_value(  # noqa: SLF001
        particles,
        weights,
        design,
        epsilon=kwargs["epsilon"],
        sinkhorn_steps=kwargs["sinkhorn_steps"],
        balance_steps=kwargs["balance_steps"],
        ridge=kwargs["ridge"],
    )
    jvp = batch._restore_cloud_batch_jvp(  # noqa: SLF001
        particles,
        weights,
        particle_tangent,
        weight_tangent,
        design,
        epsilon=kwargs["epsilon"],
        sinkhorn_steps=kwargs["sinkhorn_steps"],
        balance_steps=kwargs["balance_steps"],
        ridge=kwargs["ridge"],
    )
    tf.debugging.assert_equal(value["particles"], jvp["particles"])
    tf.debugging.assert_equal(value["valid"], jvp["valid"])


def test_tangent_only_invalidity_fails_closed() -> None:
    adapter, theta, observations, initial_noise, process_noise, design, kwargs = _small_case()

    def invalid_initial_tangent(values, noise):
        return tf.fill(
            [tf.shape(values)[0], tf.shape(noise)[0], 18, 3],
            tf.constant(float("nan"), tf.float32),
        )

    invalid_adapter = batch.BatchCandidateModelAdapter(
        adapter.state_dimension,
        adapter.parameter_count,
        adapter.initial_value,
        invalid_initial_tangent,
        adapter.transition_value,
        adapter.transition_tangent,
        adapter.observation_value,
        adapter.observation_tangent,
    )
    value, score, status = batch.batch_finite_value_score(
        invalid_adapter,
        theta,
        observations,
        initial_noise,
        process_noise,
        design,
        **kwargs,
    )
    assert not bool(status["program_valid"][0].numpy())
    assert not bool(tf.math.is_finite(value[0]).numpy())
    assert not bool(tf.reduce_any(tf.math.is_finite(score[0])).numpy())
