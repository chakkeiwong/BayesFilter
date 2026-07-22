from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    generalized_sv_prior_mean_candidate_adapter,
    ksc_mixture_sv_candidate_adapter,
    reduced_sir_candidate_adapter,
)
from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
    reduced_latent_preclip_sir_model,
)


def _central_difference(function, theta: tf.Tensor, step: tf.Tensor) -> tf.Tensor:
    columns = []
    width = int(theta.shape[0])
    for index in range(width):
        direction = tf.one_hot(index, width, dtype=theta.dtype)
        columns.append(
            (function(theta + step[index] * direction) - function(theta - step[index] * direction))
            / (2.0 * step[index])
        )
    return tf.stack(columns, axis=-1)


def _check_composed_adapter(adapter, theta, initial_noise, process_noise, observation, step):
    particles = adapter.initial_value(theta, initial_noise)
    tangent = adapter.initial_tangent(theta, initial_noise)
    initial_fd = _central_difference(
        lambda value: adapter.initial_value(value, initial_noise), theta, step
    )
    tf.debugging.assert_near(tangent, initial_fd, rtol=4e-3, atol=4e-4)

    transitioned = adapter.transition_value(
        theta, particles, process_noise, tf.constant(1, tf.int32)
    )
    transitioned_tangent = adapter.transition_tangent(
        theta, particles, process_noise, tangent, tf.constant(1, tf.int32)
    )
    transition_fd = _central_difference(
        lambda value: adapter.transition_value(
            value,
            adapter.initial_value(value, initial_noise),
            process_noise,
            tf.constant(1, tf.int32),
        ),
        theta,
        step,
    )
    tf.debugging.assert_near(
        transitioned_tangent, transition_fd, rtol=6e-3, atol=6e-4
    )

    log_likelihood = adapter.observation_value(
        theta, transitioned, observation, tf.constant(1, tf.int32)
    )
    score = adapter.observation_tangent(
        theta,
        transitioned,
        transitioned_tangent,
        observation,
        tf.constant(1, tf.int32),
    )
    observation_fd = _central_difference(
        lambda value: adapter.observation_value(
            value,
            adapter.transition_value(
                value,
                adapter.initial_value(value, initial_noise),
                process_noise,
                tf.constant(1, tf.int32),
            ),
            observation,
            tf.constant(1, tf.int32),
        ),
        theta,
        step,
    )
    tf.debugging.assert_near(score, observation_fd, rtol=8e-3, atol=8e-4)
    assert bool(tf.reduce_all(tf.math.is_finite(log_likelihood)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_ksc_mixture_adapter_manual_tangents_match_local_fd() -> None:
    _check_composed_adapter(
        ksc_mixture_sv_candidate_adapter(),
        theta=tf.constant([0.2533471, -0.9162907], tf.float32),
        initial_noise=tf.constant([[-0.7], [0.2], [1.1], [-1.4]], tf.float32),
        process_noise=tf.constant([[0.1], [-0.2], [0.3], [0.05]], tf.float32),
        observation=tf.constant([-1.3], tf.float32),
        step=tf.constant([2e-3, 2e-3], tf.float32),
    )


def test_generalized_sv_adapter_manual_tangents_match_local_fd() -> None:
    _check_composed_adapter(
        generalized_sv_prior_mean_candidate_adapter(),
        theta=tf.constant([1.5, -2.0, 0.2], tf.float32),
        initial_noise=tf.constant([[-0.7], [0.2], [1.1], [-1.4]], tf.float32),
        process_noise=tf.constant([[0.1], [-0.2], [0.3], [0.05]], tf.float32),
        observation=tf.constant([0.4], tf.float32),
        step=tf.constant([2e-3, 2e-3, 2e-3], tf.float32),
    )


def test_reduced_sir_adapter_matches_formal_preclip_transition_timing() -> None:
    adapter = reduced_sir_candidate_adapter(
        transition_before_first_observation=False,
        mechanics_fixture_only=True,
    )
    model = reduced_latent_preclip_sir_model()
    theta32 = tf.zeros([3], tf.float32)
    theta64 = tf.zeros([3], tf.float64)
    points32 = tf.constant([[-0.3, 0.2], [0.4, -0.1]], tf.float32)
    zero_noise = tf.zeros_like(points32)

    for time_index in (1, 2):
        actual = adapter.transition_value(
            theta32,
            points32,
            zero_noise,
            tf.constant(time_index, tf.int32),
        )
        physical = model.physical_state(
            tf.cast(points32, tf.float64), time_index=time_index - 1
        )
        expected = model.physical_model.transition_mean(theta64, physical)
        tf.debugging.assert_near(
            actual, tf.cast(expected, tf.float32), rtol=2e-5, atol=2e-6
        )


def test_reduced_sir_requires_explicit_mechanics_fixture_opt_in() -> None:
    try:
        reduced_sir_candidate_adapter()
    except ValueError as exc:
        assert "artificial mechanics fixture" in str(exc)
    else:
        raise AssertionError("reduced SIR must fail closed outside mechanics tests")
