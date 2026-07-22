from __future__ import annotations

import ast
import inspect
import textwrap

import tensorflow as tf

from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
    reduced_latent_preclip_sir_model,
)
from bayesfilter.highdim.sir_online_score_teacher_tf import (
    initial_log_density_and_score,
    make_online_sir_teacher,
    observation_log_density_and_score,
    online_sir_value_and_score_teacher,
    static_spec_from_model,
    transition_log_density_and_score,
)


DTYPE = tf.float64


def _fixture():
    return (
        reduced_latent_preclip_sir_model(),
        tf.constant([0.03, -0.02, 0.04], DTYPE),
        tf.constant([[0.15], [0.1]], DTYPE),
        tf.constant([86100, 86101], tf.int32),
    )


def test_local_initial_transition_and_observation_scores_match_autodiff() -> None:
    model, theta, observations, _ = _fixture()
    spec = static_spec_from_model(model)
    previous = tf.constant([[0.3, 0.2], [-0.1, 0.4]], DTYPE)
    current = tf.constant([[0.25, 0.15], [0.05, 0.35]], DTYPE)

    initial_value, initial_score = initial_log_density_and_score(
        theta, previous, spec=spec
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        automatic_initial = model.initial_log_density(theta, previous)
    initial_jacobian = tape.jacobian(
        automatic_initial,
        theta,
        unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )
    tf.debugging.assert_near(initial_value, automatic_initial, atol=2e-13)
    tf.debugging.assert_equal(initial_score, initial_jacobian)

    transition_value, transition_score = transition_log_density_and_score(
        theta, previous, current, time_index=1, spec=spec
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        automatic_transition = model.transition_log_density(
            theta, previous, current, t=1
        )
    transition_jacobian = tape.jacobian(automatic_transition, theta)
    tf.debugging.assert_near(transition_value, automatic_transition, atol=2e-13)
    tf.debugging.assert_near(transition_score, transition_jacobian, atol=2e-11)

    observation_value, observation_score = observation_log_density_and_score(
        theta, current, observations[1], time_index=1, spec=spec
    )
    with tf.GradientTape() as tape:
        tape.watch(theta)
        automatic_observation = model.observation_log_density(
            theta, current, observations[1], t=1
        )
    observation_jacobian = tape.jacobian(automatic_observation, theta)
    tf.debugging.assert_near(observation_value, automatic_observation, atol=2e-13)
    tf.debugging.assert_near(observation_score, observation_jacobian, atol=2e-11)


def test_teacher_is_deterministic_finite_and_backward_rows_normalize() -> None:
    model, theta, observations, seeds = _fixture()
    first = online_sir_value_and_score_teacher(
        model, theta, observations, seeds, num_particles=32
    )
    second = online_sir_value_and_score_teacher(
        model, theta, observations, seeds, num_particles=32
    )
    tf.debugging.assert_equal(first["log_likelihood"], second["log_likelihood"])
    tf.debugging.assert_equal(first["score"], second["score"])
    assert bool(tf.reduce_all(first["finite"]).numpy())
    assert float(tf.reduce_max(first["maximum_backward_row_sum_error"]).numpy()) < 1e-12
    assert bool(tf.reduce_all(first["minimum_ess"] > 0.0).numpy())


def test_t1_score_reduces_to_normalized_initial_local_score() -> None:
    model, theta, observations, seeds = _fixture()
    result = online_sir_value_and_score_teacher(
        model, theta, observations[:1], seeds, num_particles=32
    )
    expected = tf.reduce_sum(
        result["initial_normalized_weights"][:, :, None] * result["initial_marks"],
        axis=1,
    )
    tf.debugging.assert_equal(result["score"], expected)
    tf.debugging.assert_equal(result["score_history"][:, 0, :], expected)


def test_previous_marks_and_transition_score_are_both_required_at_t2() -> None:
    model, theta, observations, seeds = _fixture()
    total = online_sir_value_and_score_teacher(
        model, theta, observations, seeds, num_particles=64
    )
    stopped_marks = online_sir_value_and_score_teacher(
        model,
        theta,
        observations,
        seeds,
        num_particles=64,
        stop_previous_marks=True,
    )
    stopped_transition = online_sir_value_and_score_teacher(
        model,
        theta,
        observations,
        seeds,
        num_particles=64,
        stop_transition_score=True,
    )
    assert float(tf.reduce_max(tf.abs(total["score"] - stopped_marks["score"])).numpy()) > 1e-5
    assert float(
        tf.reduce_max(tf.abs(total["score"] - stopped_transition["score"])).numpy()
    ) > 1e-5


def test_teacher_defaults_to_xla_and_has_no_python_time_or_particle_loop() -> None:
    model, _, observations, seeds = _fixture()
    factory_parameter = inspect.signature(make_online_sir_teacher).parameters[
        "jit_compile"
    ]
    assert factory_parameter.default is True
    source = textwrap.dedent(inspect.getsource(online_sir_value_and_score_teacher))
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    assert "tf.while_loop" in source
    assert "ledh" not in source.lower()
    assert "contract_e" not in source.lower()
    teacher = make_online_sir_teacher(
        model, observations, seeds, num_particles=16, jit_compile=False
    )
    result = teacher(tf.constant([0.03, -0.02, 0.04], DTYPE))
    assert result["score"].shape == (2, 3)
