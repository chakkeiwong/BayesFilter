from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_younis_kdm_lgssm_reference_tf import (
    BOOTSTRAP_FIXED_STREAM_TARGET,
    bootstrap_lgssm_fixed_stream_value_and_directional_score,
    make_bootstrap_lgssm_fixed_stream_kernel,
    matrix_lgssm_value_and_directional_score,
    scalar_lgssm_value_and_score,
)


def test_scalar_lgssm_reference_score_matches_value_finite_difference():
    dtype = tf.float64
    observations = tf.constant([0.3, -0.8, 0.4, 1.1], dtype)
    theta = tf.constant([0.72], dtype)
    epsilon = 1.0e-6
    value, score = scalar_lgssm_value_and_score(theta, observations)
    plus, _ = scalar_lgssm_value_and_score(
        theta + tf.constant([epsilon], dtype), observations
    )
    minus, _ = scalar_lgssm_value_and_score(
        theta - tf.constant([epsilon], dtype), observations
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isfinite(float(value.numpy()))
    assert np.isfinite(float(score.numpy()))
    np.testing.assert_allclose(
        score.numpy(), finite_difference.numpy(), rtol=2.0e-8, atol=2.0e-9
    )


def test_scalar_lgssm_reference_rejects_bad_shapes():
    with np.testing.assert_raises(ValueError):
        scalar_lgssm_value_and_score(
            tf.constant([0.7, 0.1], tf.float64),
            tf.constant([0.2], tf.float64),
        )


def _matrix_fixture():
    dtype = tf.float64
    return {
        "observations": tf.constant(
            [[0.25, -0.4], [0.7, 0.1], [-0.2, 0.55], [0.1, -0.3]], dtype
        ),
        "transition_base": tf.constant([[0.0, 0.12], [-0.08, 0.10]], dtype),
        "transition_direction": tf.constant([[1.0, 0.0], [0.0, 0.7]], dtype),
        "observation_matrix": tf.constant([[1.0, 0.25], [-0.15, 0.9]], dtype),
        "initial_mean": tf.constant([0.1, -0.2], dtype),
        "initial_covariance": tf.constant([[0.8, 0.12], [0.12, 0.6]], dtype),
        "process_covariance": tf.constant([[0.12, 0.025], [0.025, 0.09]], dtype),
        "observation_covariance": tf.constant([[0.22, 0.035], [0.035, 0.18]], dtype),
    }


def _matrix_oracle(fixture, theta):
    transition = fixture["transition_base"] + theta * fixture["transition_direction"]
    return matrix_lgssm_value_and_directional_score(
        fixture["observations"],
        transition,
        fixture["transition_direction"],
        fixture["observation_matrix"],
        fixture["initial_mean"],
        fixture["initial_covariance"],
        fixture["process_covariance"],
        fixture["observation_covariance"],
    )


def test_matrix_lgssm_analytical_score_matches_value_finite_difference():
    fixture = _matrix_fixture()
    theta = tf.constant(0.72, tf.float64)
    epsilon = tf.constant(1.0e-6, tf.float64)
    value, score = _matrix_oracle(fixture, theta)
    plus, _ = _matrix_oracle(fixture, theta + epsilon)
    minus, _ = _matrix_oracle(fixture, theta - epsilon)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isfinite(float(value.numpy()))
    np.testing.assert_allclose(
        score.numpy(), finite_difference.numpy(), rtol=3.0e-8, atol=3.0e-9
    )


def test_matrix_lgssm_reference_reduces_to_scalar_reference():
    observations = tf.constant([[0.3], [-0.8], [0.4], [1.1]], tf.float64)
    theta = tf.constant([0.72], tf.float64)
    matrix_value, matrix_score = matrix_lgssm_value_and_directional_score(
        observations,
        tf.reshape(theta, [1, 1]),
        tf.ones([1, 1], tf.float64),
        tf.ones([1, 1], tf.float64),
        tf.zeros([1], tf.float64),
        tf.ones([1, 1], tf.float64),
        tf.constant([[0.1]], tf.float64),
        tf.constant([[0.2]], tf.float64),
    )
    scalar_value, scalar_score = scalar_lgssm_value_and_score(
        theta, tf.reshape(observations, [-1])
    )
    np.testing.assert_allclose(matrix_value.numpy(), scalar_value.numpy(), atol=2e-14)
    np.testing.assert_allclose(matrix_score.numpy(), scalar_score.numpy(), atol=2e-14)


def _bootstrap_call(fixture, theta):
    transition = fixture["transition_base"] + theta * fixture["transition_direction"]
    initial_states = tf.constant(
        [[-0.7, 0.2], [-0.2, -0.4], [0.1, 0.5], [0.6, -0.1], [0.8, 0.3]],
        tf.float64,
    )
    noises = tf.constant(
        [
            [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.3], [0.5, -0.2], [-0.4, 0.2]],
            [[-0.1, 0.3], [0.2, -0.5], [0.4, 0.1], [-0.2, 0.2], [0.3, -0.4]],
            [[0.3, 0.2], [-0.5, -0.1], [0.2, 0.4], [0.1, -0.3], [-0.2, 0.5]],
            [[-0.4, 0.1], [0.3, 0.2], [-0.1, -0.2], [0.2, 0.3], [0.5, -0.1]],
        ],
        tf.float64,
    )
    offsets = tf.constant([0.17, 0.61, 0.34, 0.79], tf.float64)
    result = bootstrap_lgssm_fixed_stream_value_and_directional_score(
        initial_states,
        noises,
        fixture["observations"],
        offsets,
        transition,
        fixture["transition_direction"],
        fixture["observation_matrix"],
        fixture["process_covariance"],
        fixture["observation_covariance"],
    )
    return result, initial_states, noises, offsets


def test_bootstrap_fixed_stream_score_matches_finite_difference_and_factory():
    fixture = _matrix_fixture()
    theta = tf.constant(0.72, tf.float64)
    epsilon = tf.constant(1.0e-6, tf.float64)
    result, initial_states, noises, offsets = _bootstrap_call(fixture, theta)
    plus, _, _, _ = _bootstrap_call(fixture, theta + epsilon)
    minus, _, _, _ = _bootstrap_call(fixture, theta - epsilon)
    np.testing.assert_array_equal(
        result["ancestor_indices"].numpy(), plus["ancestor_indices"].numpy()
    )
    np.testing.assert_array_equal(
        result["ancestor_indices"].numpy(), minus["ancestor_indices"].numpy()
    )
    finite_difference = (plus["value"] - minus["value"]) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["score"].numpy(), finite_difference.numpy(), rtol=2.0e-8, atol=2.0e-9
    )
    assert bool(result["valid"].numpy())
    assert float(result["minimum_ess"].numpy()) < 5.0

    kernel = make_bootstrap_lgssm_fixed_stream_kernel(
        transition_matrix_base=fixture["transition_base"],
        transition_matrix_direction=fixture["transition_direction"],
        observation_matrix=fixture["observation_matrix"],
        process_covariance=fixture["process_covariance"],
        observation_covariance=fixture["observation_covariance"],
        particle_count=5,
        horizon=4,
        dtype=tf.float64,
        jit_compile=False,
    )
    compiled = kernel(
        tf.reshape(theta, [1]),
        initial_states,
        noises,
        fixture["observations"],
        offsets,
    )
    assert kernel.target_label == BOOTSTRAP_FIXED_STREAM_TARGET
    np.testing.assert_array_equal(
        compiled["ancestor_indices"].numpy(), result["ancestor_indices"].numpy()
    )
    np.testing.assert_allclose(compiled["value"].numpy(), result["value"].numpy())
    np.testing.assert_allclose(compiled["score"].numpy(), result["score"].numpy())


def test_bootstrap_fixed_stream_rejects_invalid_offset():
    fixture = _matrix_fixture()
    _, initial_states, noises, offsets = _bootstrap_call(
        fixture, tf.constant(0.72, tf.float64)
    )
    invalid_offsets = tf.tensor_scatter_nd_update(offsets, [[1]], [1.0])
    with pytest.raises(tf.errors.InvalidArgumentError):
        bootstrap_lgssm_fixed_stream_value_and_directional_score(
            initial_states,
            noises,
            fixture["observations"],
            invalid_offsets,
            fixture["transition_base"] + 0.72 * fixture["transition_direction"],
            fixture["transition_direction"],
            fixture["observation_matrix"],
            fixture["process_covariance"],
            fixture["observation_covariance"],
        )
