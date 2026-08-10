from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    diagonal_lgssm_candidate_adapter,
    ksc_mixture_sv_candidate_adapter,
    parameterized_austria_sir_candidate_adapter,
    predator_prey_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_batch_adapters import (
    diagonal_lgssm_batch_adapter,
    ksc_mixture_sv_batch_adapter,
    parameterized_austria_sir_batch_adapter,
    predator_prey_batch_adapter,
)
from bayesfilter.highdim.cubature_genut_batch_tf import (
    batch_finite_value,
    batch_finite_value_score,
)
from bayesfilter.highdim.cubature_genut_candidate import cubature_design
from bayesfilter.highdim.cubature_genut_filter import finite_value_score


def _case(name: str):
    if name == "lgssm":
        matrix = tf.eye(3, dtype=tf.float32)
        return (
            diagonal_lgssm_batch_adapter(observation_matrix=matrix),
            diagonal_lgssm_candidate_adapter(observation_matrix=matrix),
            tf.constant([0.7, 0.5, 0.3, 0.4, 0.5], tf.float32),
            3,
            3,
            12,
            True,
        )
    if name == "ksc":
        return (
            ksc_mixture_sv_batch_adapter(),
            ksc_mixture_sv_candidate_adapter(),
            tf.constant([0.1, -0.4], tf.float32),
            1,
            1,
            12,
            False,
        )
    if name == "austria":
        return (
            parameterized_austria_sir_batch_adapter(),
            parameterized_austria_sir_candidate_adapter(),
            tf.zeros([3], tf.float32),
            18,
            9,
            36,
            True,
        )
    if name == "predator_prey":
        return (
            predator_prey_batch_adapter(),
            predator_prey_candidate_adapter(),
            tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float32),
            2,
            2,
            12,
            False,
        )
    raise AssertionError(name)


def _inputs(dimension: int, observation_dimension: int, count: int):
    return (
        tf.zeros([2, observation_dimension], tf.float32),
        tf.random.stateless_normal([count, dimension], [101, 1]),
        tf.random.stateless_normal([2, count, dimension], [101, 2]),
        cubature_design(dim=dimension, num_particles=count),
    )


def _controls(transition_first: bool):
    return {
        "epsilon": 2.0,
        "sinkhorn_steps": 2,
        "balance_steps": 2,
        "ridge": 1.0e-5,
        "transition_before_first_observation": transition_first,
        "higher_moment_correction_steps": 0,
        "higher_moment_strength": 0.0,
        "higher_moment_floor": 1.0e-5,
    }


def test_all_model_callbacks_preserve_leading_batch_shapes() -> None:
    for name in ("lgssm", "ksc", "austria", "predator_prey"):
        adapter, _scalar, theta, dimension, observation_dimension, count, _ = _case(name)
        observations, initial, process, _design = _inputs(
            dimension, observation_dimension, count
        )
        values = tf.stack([theta, theta])
        particles = adapter.initial_value(values, initial)
        tangent = adapter.initial_tangent(values, initial)
        transitioned = adapter.transition_value(values, particles, process[0], tf.constant(0))
        transitioned_tangent = adapter.transition_tangent(
            values, particles, process[0], tangent, tf.constant(0)
        )
        likelihood = adapter.observation_value(
            values, transitioned, observations[0], tf.constant(0)
        )
        likelihood_tangent = adapter.observation_tangent(
            values,
            transitioned,
            transitioned_tangent,
            observations[0],
            tf.constant(0),
        )
        assert particles.shape == (2, count, dimension)
        assert tangent.shape == (2, count, dimension, adapter.parameter_count)
        assert likelihood.shape == (2, count)
        assert likelihood_tangent.shape == (2, count, adapter.parameter_count)
        assert bool(tf.reduce_all(tf.math.is_finite(likelihood_tangent)).numpy())


def test_batch_rows_match_scalar_reference_without_shape_correction() -> None:
    tolerances = {
        "lgssm": (2.0e-6, 2.0e-5),
        "ksc": (2.0e-6, 2.0e-5),
        "austria": (2.0e-4, 2.0e-3),
        "predator_prey": (2.0e-5, 2.0e-4),
    }
    for name in ("lgssm", "ksc", "austria", "predator_prey"):
        adapter, scalar, theta, dimension, observation_dimension, count, transition_first = _case(name)
        observations, initial, process, design = _inputs(
            dimension, observation_dimension, count
        )
        kwargs = _controls(transition_first)
        batch_value, batch_score, batch_status = batch_finite_value_score(
            adapter,
            tf.stack([theta, theta]),
            observations,
            initial,
            process,
            design,
            **kwargs,
        )
        scalar_value, scalar_score, scalar_status = finite_value_score(
            scalar,
            theta,
            observations,
            initial,
            process,
            design,
            **kwargs,
        )
        assert bool(tf.reduce_all(batch_status["program_valid"]).numpy())
        assert bool(scalar_status["program_valid"].numpy())
        value_scale = tf.maximum(tf.abs(scalar_value), 1.0)
        score_scale = tf.maximum(tf.abs(scalar_score), 1.0)
        value_error = tf.reduce_max(tf.abs(batch_value - scalar_value) / value_scale)
        score_error = tf.reduce_max(tf.abs(batch_score - scalar_score) / score_scale)
        value_tolerance, score_tolerance = tolerances[name]
        assert float(value_error.numpy()) <= value_tolerance
        assert float(score_error.numpy()) <= score_tolerance


def test_value_only_endpoint_matches_value_score_route_with_shape_correction() -> None:
    adapter, _scalar, theta, dimension, observation_dimension, count, transition_first = _case(
        "lgssm"
    )
    observations, initial, process, design = _inputs(
        dimension, observation_dimension, count
    )
    values = tf.stack([theta, theta + 0.01])
    kwargs = _controls(transition_first) | {
        "higher_moment_correction_steps": 2,
        "higher_moment_strength": 0.1,
    }
    value_only, value_status = batch_finite_value(
        adapter, values, observations, initial, process, design, **kwargs
    )
    value_score, _score, score_status = batch_finite_value_score(
        adapter, values, observations, initial, process, design, **kwargs
    )
    tf.debugging.assert_equal(
        value_status["program_valid"], score_status["program_valid"]
    )
    tf.debugging.assert_equal(value_only, value_score)
