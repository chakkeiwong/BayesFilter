from __future__ import annotations

import tensorflow as tf
import pytest

from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    PP_SOURCE_OBSERVATION_SHA256,
    PP_SOURCE_SGQF_ROUTE_ID,
    PP_SOURCE_STATE_SHA256,
    generate_source_order_predator_prey_dataset_tf,
    make_predator_prey_source_sgqf_route,
    pp_source_sgqf_physical_value_only_status,
    pp_source_sgqf_physical_value_score_status,
)
from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
    PP_OBSERVATION_SHA256,
    PP_TRUTH_PHYSICAL,
    _tensor_hash,
    generate_frozen_predator_prey_dataset_tf,
)


def test_source_dataset_is_distinct_twenty_transition_target() -> None:
    states, observations = generate_source_order_predator_prey_dataset_tf()
    _amended_states, amended_observations = generate_frozen_predator_prey_dataset_tf()

    assert states.shape == (21, 2)
    assert observations.shape == (20, 2)
    assert _tensor_hash(states) == PP_SOURCE_STATE_SHA256
    assert _tensor_hash(observations) == PP_SOURCE_OBSERVATION_SHA256
    assert _tensor_hash(amended_observations) == PP_OBSERVATION_SHA256
    assert PP_SOURCE_OBSERVATION_SHA256 != PP_OBSERVATION_SHA256
    assert float(tf.reduce_min(states).numpy()) < 0.0


def test_source_t1_and_t2_execute_one_and_two_transitions() -> None:
    route = make_predator_prey_source_sgqf_route()
    point = PP_TRUTH_PHYSICAL[None, :]

    value_t1, status_t1 = pp_source_sgqf_physical_value_only_status(
        point,
        observations=route.observations[:1],
        nodes=route.nodes,
        weights=route.weights,
    )
    value_t2, status_t2 = pp_source_sgqf_physical_value_only_status(
        point,
        observations=route.observations[:2],
        nodes=route.nodes,
        weights=route.weights,
    )

    assert bool(tf.reduce_all(tf.math.is_finite(value_t1)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(value_t2)).numpy())
    tf.debugging.assert_equal(status_t1["transition_count"], [1])
    tf.debugging.assert_equal(status_t2["transition_count"], [2])


def test_source_physical_manual_score_matches_same_scalar_finite_difference() -> None:
    route = make_predator_prey_source_sgqf_route()
    point = PP_TRUTH_PHYSICAL[None, :]
    value, score, status = route.physical_value_score_status(point)
    value_only, value_status = route.physical_value_only_status(point)
    steps = tf.constant([1e-5, 1e-3, 1e-4, 1e-5, 1e-5, 1e-5], tf.float64)
    columns = []
    for coordinate in range(6):
        shift = steps[coordinate] * tf.one_hot(coordinate, 6, dtype=tf.float64)
        plus, _ = route.physical_value_only_status(point + shift[None, :])
        minus, _ = route.physical_value_only_status(point - shift[None, :])
        columns.append((plus - minus) / (2.0 * steps[coordinate]))
    finite_difference = tf.stack(columns, axis=1)

    tf.debugging.assert_near(value, value_only, atol=1e-10, rtol=1e-12)
    tf.debugging.assert_near(score, finite_difference, atol=5e-5, rtol=5e-5)
    tf.debugging.assert_equal(status["status_code"], [0])
    tf.debugging.assert_equal(value_status["status_code"], [0])
    tf.debugging.assert_equal(status["transition_count"], [20])


def test_source_route_identity_is_sealed_and_amended_data_is_rejected() -> None:
    first = make_predator_prey_source_sgqf_route()
    second = make_predator_prey_source_sgqf_route()
    assert first.route_identity == second.route_identity
    assert len(first.route_identity) == 64
    assert first.manifest["route_id"] == PP_SOURCE_SGQF_ROUTE_ID
    assert first.manifest["parameter_coordinate"] == "physical=(r,K,a,s,u,v)"
    assert first.manifest["time_order"] == (
        "x0_then_20_transition_then_observe_steps_y1_y20"
    )

    _states, amended = generate_frozen_predator_prey_dataset_tf()
    try:
        type(first)(
            states=first.states,
            observations=amended,
            nodes=first.nodes,
            weights=first.weights,
            route_identity=first.route_identity,
            manifest=first.manifest,
        )
    except ValueError as exc:
        assert "observation identity rejected" in str(exc)
    else:
        raise AssertionError("amended observations must not issue source-route identity")


def test_source_public_physical_endpoint_rejects_parameter_box_boundary() -> None:
    route = make_predator_prey_source_sgqf_route()
    boundary = tf.constant([[0.1, 114.0, 25.0, 0.3, 0.5, 0.5]], tf.float64)
    try:
        route.physical_value_score_status(boundary)
    except ValueError as exc:
        assert "strictly interior" in str(exc)
    else:
        raise AssertionError("physical boundary point must be rejected")


def test_source_physical_score_is_batch_permutation_equivariant() -> None:
    route = make_predator_prey_source_sgqf_route()
    points = tf.stack(
        [
            PP_TRUTH_PHYSICAL,
            tf.constant([0.7, 118.0, 24.0, 0.4, 0.6, 0.45], tf.float64),
            tf.constant([0.45, 122.0, 27.0, 0.25, 0.35, 0.7], tf.float64),
        ]
    )
    value, score, status = route.physical_value_score_status(points)
    permutation = tf.constant([2, 0, 1], tf.int32)
    permuted_value, permuted_score, permuted_status = route.physical_value_score_status(
        tf.gather(points, permutation)
    )

    tf.debugging.assert_near(permuted_value, tf.gather(value, permutation))
    tf.debugging.assert_near(permuted_score, tf.gather(score, permutation))
    tf.debugging.assert_equal(permuted_status["status_code"], tf.gather(status["status_code"], permutation))


def test_source_public_score_call_graph_does_not_use_runtime_autodiff(monkeypatch: pytest.MonkeyPatch) -> None:
    route = make_predator_prey_source_sgqf_route()

    def forbidden_gradient_tape(*_args, **_kwargs):
        raise AssertionError("runtime autodiff is forbidden for source SGQF score")

    monkeypatch.setattr(tf, "GradientTape", forbidden_gradient_tape)
    value, score, status = route.physical_value_score_status(PP_TRUTH_PHYSICAL[None, :])
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    tf.debugging.assert_equal(status["status_code"], [0])
