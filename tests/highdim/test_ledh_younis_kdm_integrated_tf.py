"""Phase 4A gates for full-feedback kernelized observation weighting."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)
from bayesfilter.highdim.ledh_younis_kdm_integrated_tf import (
    INTEGRATED_OBSERVATION_CLASSIFICATION,
    INTEGRATED_OBSERVATION_ROUTE_ID,
    integrated_linear_gaussian_kdm_value_and_analytical_score,
    linear_gaussian_kdm_observation_factors,
    make_integrated_linear_gaussian_kdm_kernel,
)
from bayesfilter.highdim.ledh_younis_kdm_tf import (
    ATOM_FINITE_TARGET,
    KDM_FINITE_TARGET,
)


DTYPE = tf.float64


def _model(*, covariance_tangent_without_density_callback: bool = False):
    def transition_mean_fn(theta, points):
        return points + theta[0] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return tf.sin(points) + d_points + theta[0] * tf.cos(points) * d_points

    covariance_tangent_fn = None
    if covariance_tangent_without_density_callback:
        covariance_tangent_fn = lambda theta: 0.2 * tf.eye(2, dtype=DTYPE)

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=0.4 * tf.eye(2, dtype=DTYPE),
        observation_covariance=0.6 * tf.eye(2, dtype=DTYPE),
        observation_covariance_tangent_fn=covariance_tangent_fn,
    )


def _full_fixture():
    rng = np.random.default_rng(260907)
    particle_count, dimension, horizon = 8, 2, 2
    initial_states = tf.constant(
        rng.standard_normal((particle_count, dimension)), DTYPE
    )
    initial_covariances = tf.broadcast_to(
        tf.eye(dimension, dtype=DTYPE),
        [particle_count, dimension, dimension],
    )
    noises = tf.constant(
        rng.standard_normal((horizon, particle_count, dimension)), DTYPE
    )
    observations = tf.constant(rng.standard_normal((horizon, dimension)), DTYPE)
    base = np.concatenate([np.eye(dimension), -np.eye(dimension)], axis=0)
    design = tf.constant(
        np.tile(base, (particle_count // (2 * dimension), 1)), DTYPE
    )
    options = {
        "flow_substeps": 6,
        "reset_policy": "contract_e",
        "reset_design": design,
        "reset_sinkhorn_steps": 4,
        "reset_balance_steps": 2,
        "correction_steps": 1,
        "pairwise_steps": 1,
        "coordinate_cap": 0.95,
    }
    return {
        "theta": tf.constant([0.6], DTYPE),
        "initial_states": initial_states,
        "initial_covariances": initial_covariances,
        "noises": noises,
        "observations": observations,
        "observation_matrix": tf.eye(dimension, dtype=DTYPE),
        "d_observation_matrix": tf.zeros([dimension, dimension], DTYPE),
        "zero_bandwidths": tf.zeros(
            [horizon, particle_count, dimension, dimension], DTYPE
        ),
        "positive_bandwidths": tf.broadcast_to(
            0.08 * tf.eye(dimension, dtype=DTYPE),
            [horizon, particle_count, dimension, dimension],
        ),
        "options": options,
    }


def _run_integrated(fixture, theta, bandwidths, *, bandwidth_is_zero):
    return integrated_linear_gaussian_kdm_value_and_analytical_score(
        _model(),
        theta,
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        fixture["observation_matrix"],
        fixture["d_observation_matrix"],
        bandwidths,
        tf.zeros_like(bandwidths),
        canonical_options=fixture["options"],
        bandwidth_is_zero=bandwidth_is_zero,
    )


def test_gaussian_factor_total_tangent_matches_combined_finite_difference():
    states = np.array([[-0.4, 0.2], [0.7, -0.5], [1.1, 0.3]])
    d_states = np.array([[0.1, -0.2], [-0.3, 0.05], [0.2, 0.4]])
    bandwidths = np.array(
        [
            [[0.30, 0.04], [0.04, 0.20]],
            [[0.18, -0.02], [-0.02, 0.24]],
            [[0.25, 0.03], [0.03, 0.16]],
        ]
    )
    d_bandwidths = np.array(
        [
            [[0.03, -0.01], [-0.01, -0.02]],
            [[-0.02, 0.015], [0.015, 0.01]],
            [[0.01, 0.02], [0.02, 0.025]],
        ]
    )
    matrix = np.array([[1.2, -0.35]])
    d_matrix = np.array([[0.08, 0.12]])
    covariance = np.array([[0.7]])
    d_covariance = np.array([[0.09]])
    observation = np.array([0.45])

    result = linear_gaussian_kdm_observation_factors(
        tf.constant(states, DTYPE),
        tf.constant(d_states, DTYPE),
        tf.constant(bandwidths, DTYPE),
        tf.constant(d_bandwidths, DTYPE),
        tf.constant(matrix, DTYPE),
        tf.constant(d_matrix, DTYPE),
        tf.constant(covariance, DTYPE),
        tf.constant(d_covariance, DTYPE),
        tf.constant(observation, DTYPE),
        bandwidth_is_zero=False,
    )
    assert bool(result["valid"].numpy())

    epsilon = 2.0e-6

    def value_at(sign):
        zeros_states = tf.zeros_like(tf.constant(d_states, DTYPE))
        zeros_bandwidths = tf.zeros_like(tf.constant(d_bandwidths, DTYPE))
        zeros_matrix = tf.zeros_like(tf.constant(d_matrix, DTYPE))
        zeros_covariance = tf.zeros_like(tf.constant(d_covariance, DTYPE))
        out = linear_gaussian_kdm_observation_factors(
            tf.constant(states + sign * epsilon * d_states, DTYPE),
            zeros_states,
            tf.constant(bandwidths + sign * epsilon * d_bandwidths, DTYPE),
            zeros_bandwidths,
            tf.constant(matrix + sign * epsilon * d_matrix, DTYPE),
            zeros_matrix,
            tf.constant(covariance + sign * epsilon * d_covariance, DTYPE),
            zeros_covariance,
            tf.constant(observation, DTYPE),
            bandwidth_is_zero=False,
        )
        return out["log_factor"].numpy()

    finite_difference = (value_at(1.0) - value_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["d_log_factor"].numpy(),
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_zero_bandwidth_is_exact_full_trajectory_call_chain_identity():
    fixture = _full_fixture()
    canonical_value, canonical_score, canonical_trace = (
        canonical_value_and_analytical_score(
            _model(),
            fixture["theta"],
            fixture["initial_states"],
            fixture["initial_covariances"],
            fixture["noises"],
            fixture["observations"],
            with_score=True,
            return_trace=True,
            **fixture["options"],
        )
    )
    integrated = _run_integrated(
        fixture,
        fixture["theta"],
        fixture["zero_bandwidths"],
        bandwidth_is_zero=True,
    )

    assert integrated["route_id"] == INTEGRATED_OBSERVATION_ROUTE_ID
    assert integrated["route_classification"] == INTEGRATED_OBSERVATION_CLASSIFICATION
    assert integrated["target_label"] == ATOM_FINITE_TARGET
    assert bool(integrated["valid"].numpy())
    np.testing.assert_allclose(integrated["value"].numpy(), canonical_value.numpy())
    np.testing.assert_allclose(integrated["score"].numpy(), canonical_score.numpy())
    for expected_step, actual_step in zip(canonical_trace, integrated["trace"]):
        for field in (
            "posterior_weights",
            "d_posterior_weights",
            "states_after_reset",
            "d_states_after_reset",
            "post_covariances",
        ):
            np.testing.assert_allclose(
                actual_step[field].numpy(), expected_step[field].numpy(),
                rtol=2.0e-13, atol=2.0e-13,
            )


def test_positive_bandwidth_changes_feedback_and_total_score_matches_difference():
    fixture = _full_fixture()
    atom = _run_integrated(
        fixture,
        fixture["theta"],
        fixture["zero_bandwidths"],
        bandwidth_is_zero=True,
    )
    candidate = _run_integrated(
        fixture,
        fixture["theta"],
        fixture["positive_bandwidths"],
        bandwidth_is_zero=False,
    )
    assert candidate["target_label"] == KDM_FINITE_TARGET
    assert not bool(candidate["complete_mixture_posterior"].numpy())
    assert bool(candidate["kernel_feedback_into_reset"].numpy())
    assert bool(candidate["valid"].numpy())
    assert abs(float(candidate["value"].numpy() - atom["value"].numpy())) > 1.0e-6
    state_shift = tf.reduce_max(
        tf.abs(
            candidate["trace"][-1]["states_after_reset"]
            - atom["trace"][-1]["states_after_reset"]
        )
    )
    assert float(state_shift.numpy()) > 1.0e-7

    epsilon = 2.0e-6
    plus = _run_integrated(
        fixture,
        fixture["theta"] + tf.constant([epsilon], DTYPE),
        fixture["positive_bandwidths"],
        bandwidth_is_zero=False,
    )
    minus = _run_integrated(
        fixture,
        fixture["theta"] - tf.constant([epsilon], DTYPE),
        fixture["positive_bandwidths"],
        bandwidth_is_zero=False,
    )
    finite_difference = (plus["value"] - minus["value"]) / (2.0 * epsilon)
    error = tf.abs(candidate["score"][0] - finite_difference)
    scale = tf.maximum(tf.abs(finite_difference), tf.constant(1.0, DTYPE))
    assert float((error / scale).numpy()) < 2.0e-4


def test_rank_deficient_and_mixed_zero_psd_bandwidths_are_supported():
    states = tf.constant([[0.0, 0.2], [0.5, -0.1]], DTYPE)
    mixed_bandwidths = tf.constant(
        [[[0.3, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]], DTYPE
    )
    result = linear_gaussian_kdm_observation_factors(
        states,
        tf.zeros_like(states),
        mixed_bandwidths,
        tf.zeros_like(mixed_bandwidths),
        tf.eye(2, dtype=DTYPE),
        tf.zeros([2, 2], DTYPE),
        0.5 * tf.eye(2, dtype=DTYPE),
        tf.zeros([2, 2], DTYPE),
        tf.constant([0.1, -0.2], DTYPE),
        bandwidth_is_zero=False,
    )
    assert bool(result["valid"].numpy())

    indefinite = tf.constant(
        [[[0.2, 0.0], [0.0, -0.05]], [[0.1, 0.0], [0.0, 0.1]]], DTYPE
    )
    invalid = linear_gaussian_kdm_observation_factors(
        states,
        tf.zeros_like(states),
        indefinite,
        tf.zeros_like(indefinite),
        tf.eye(2, dtype=DTYPE),
        tf.zeros([2, 2], DTYPE),
        0.5 * tf.eye(2, dtype=DTYPE),
        tf.zeros([2, 2], DTYPE),
        tf.constant([0.1, -0.2], DTYPE),
        bandwidth_is_zero=False,
    )
    assert not bool(invalid["valid"].numpy())
    np.testing.assert_array_equal(
        invalid["component_valid"].numpy(), [False, True]
    )
    assert np.isnan(invalid["log_factor"].numpy()[0])
    assert np.isfinite(invalid["log_factor"].numpy()[1])


def test_integrated_route_supports_covariance_tangent_in_gaussian_fallback():
    fixture = _full_fixture()
    result = integrated_linear_gaussian_kdm_value_and_analytical_score(
        _model(covariance_tangent_without_density_callback=True),
        fixture["theta"],
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        fixture["observation_matrix"],
        fixture["d_observation_matrix"],
        fixture["zero_bandwidths"],
        tf.zeros_like(fixture["zero_bandwidths"]),
        canonical_options=fixture["options"],
        bandwidth_is_zero=True,
    )
    assert bool(result["valid"].numpy())
    assert float(result["atom_factor_tangent_error_max"].numpy()) < 1.0e-12


def _fully_parameterized_model(theta_fixed, direction):
    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    direction = tf.convert_to_tensor(direction, DTYPE)
    observation_matrix = theta_fixed[1] * tf.eye(2, dtype=DTYPE)
    d_observation_matrix = direction[1] * tf.eye(2, dtype=DTYPE)

    def transition_mean_fn(theta, points):
        return theta[0] * points + 0.05 * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            direction[0] * points
            + theta[0] * d_points
            + 0.05 * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return tf.einsum("od,nd->no", observation_matrix, points)

    def observation_tangent_fn(points, d_points):
        return tf.einsum(
            "od,nd->no", d_observation_matrix, points
        ) + tf.einsum("od,nd->no", observation_matrix, d_points)

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            observation_matrix, [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.exp(2.0 * theta_fixed[2]) * tf.eye(2, dtype=DTYPE),
        observation_covariance=tf.exp(2.0 * theta_fixed[3])
        * tf.eye(2, dtype=DTYPE),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.broadcast_to(
            d_observation_matrix, [tf.shape(points)[0], 2, 2]
        ),
        process_covariance_tangent_fn=lambda theta: (
            2.0
            * tf.exp(2.0 * theta[2])
            * direction[2]
            * tf.eye(2, dtype=DTYPE)
        ),
        observation_covariance_tangent_fn=lambda theta: (
            2.0
            * tf.exp(2.0 * theta[3])
            * direction[3]
            * tf.eye(2, dtype=DTYPE)
        ),
    )


def test_full_feedback_total_tangent_includes_model_covar_map_and_bandwidth():
    fixture = _full_fixture()
    theta = tf.constant([0.82, 1.08, -0.32, -0.18, -0.55], DTYPE)
    direction = tf.constant([0.4, -0.3, 0.2, 0.5, -0.25], DTYPE)
    bandwidth_scale = tf.exp(2.0 * theta[4]) * tf.constant(0.12, DTYPE)
    bandwidths = tf.broadcast_to(
        bandwidth_scale * tf.eye(2, dtype=DTYPE),
        tf.shape(fixture["positive_bandwidths"]),
    )
    d_bandwidths = 2.0 * direction[4] * bandwidths
    observation_matrix = theta[1] * tf.eye(2, dtype=DTYPE)
    d_observation_matrix = direction[1] * tf.eye(2, dtype=DTYPE)

    result = integrated_linear_gaussian_kdm_value_and_analytical_score(
        _fully_parameterized_model(theta, direction),
        theta,
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        observation_matrix,
        d_observation_matrix,
        bandwidths,
        d_bandwidths,
        canonical_options=fixture["options"],
        bandwidth_is_zero=False,
    )
    assert bool(result["valid"].numpy())

    def value_at(theta_value):
        zero_direction = tf.zeros([5], DTYPE)
        matrix = theta_value[1] * tf.eye(2, dtype=DTYPE)
        bandwidth_scale_value = (
            tf.exp(2.0 * theta_value[4]) * tf.constant(0.12, DTYPE)
        )
        bandwidth_value = tf.broadcast_to(
            bandwidth_scale_value * tf.eye(2, dtype=DTYPE),
            tf.shape(fixture["positive_bandwidths"]),
        )
        return integrated_linear_gaussian_kdm_value_and_analytical_score(
            _fully_parameterized_model(theta_value, zero_direction),
            theta_value,
            fixture["initial_states"],
            fixture["initial_covariances"],
            fixture["noises"],
            fixture["observations"],
            matrix,
            tf.zeros_like(matrix),
            bandwidth_value,
            tf.zeros_like(bandwidth_value),
            canonical_options=fixture["options"],
            bandwidth_is_zero=False,
        )["value"]

    epsilon = tf.constant(2.0e-6, DTYPE)
    finite_difference = (
        value_at(theta + epsilon * direction)
        - value_at(theta - epsilon * direction)
    ) / (2.0 * epsilon)
    error = tf.abs(result["score"][0] - finite_difference)
    scale = tf.maximum(tf.abs(finite_difference), tf.constant(1.0, DTYPE))
    assert float((error / scale).numpy()) < 3.0e-4, (
        result["score"].numpy(), finite_difference.numpy()
    )


def test_integrated_route_requires_full_reset_and_both_corrections():
    fixture = _full_fixture()
    required = fixture["options"]
    bad_options = [
        {**required, "reset_policy": "none"},
        {**required, "reset_design": None},
        {**required, "correction_steps": 0},
        {**required, "pairwise_steps": 0},
        {**required, "coordinate_cap": 0.0},
        {**required, "annealed_stages": 2},
    ]
    for options in bad_options:
        with pytest.raises(ValueError):
            integrated_linear_gaussian_kdm_value_and_analytical_score(
                _model(),
                fixture["theta"],
                fixture["initial_states"],
                fixture["initial_covariances"],
                fixture["noises"],
                fixture["observations"],
                fixture["observation_matrix"],
                fixture["d_observation_matrix"],
                fixture["zero_bandwidths"],
                tf.zeros_like(fixture["zero_bandwidths"]),
                canonical_options=options,
                bandwidth_is_zero=True,
            )


def test_fixed_shape_complete_endpoint_matches_eager_and_cpu_xla():
    fixture = _full_fixture()
    eager = _run_integrated(
        fixture,
        fixture["theta"],
        fixture["positive_bandwidths"],
        bandwidth_is_zero=False,
    )
    for jit_compile in (False, True):
        kernel = make_integrated_linear_gaussian_kdm_kernel(
            _model(),
            theta_dimension=1,
            particle_count=8,
            state_dimension=2,
            observation_dimension=2,
            horizon=2,
            canonical_options=fixture["options"],
            bandwidth_is_zero=False,
            dtype=DTYPE,
            jit_compile=jit_compile,
        )
        result = kernel(
            fixture["theta"],
            fixture["initial_states"],
            fixture["initial_covariances"],
            fixture["noises"],
            fixture["observations"],
            fixture["observation_matrix"],
            fixture["d_observation_matrix"],
            fixture["positive_bandwidths"],
            tf.zeros_like(fixture["positive_bandwidths"]),
        )
        assert kernel.route_id == INTEGRATED_OBSERVATION_ROUTE_ID
        assert kernel.target_label == KDM_FINITE_TARGET
        assert kernel.jit_compile is jit_compile
        assert bool(result["valid"].numpy())
        assert np.all(result["atom_identity_valid"].numpy())
        assert np.all(result["observation_map_valid"].numpy())
        np.testing.assert_allclose(
            result["value"].numpy(), eager["value"].numpy(),
            rtol=2.0e-11, atol=2.0e-11,
        )
        np.testing.assert_allclose(
            result["score"].numpy(), eager["score"].numpy(),
            rtol=2.0e-10, atol=2.0e-10,
        )
        np.testing.assert_allclose(
            result["states_after_reset"].numpy(),
            np.stack(
                [step["states_after_reset"].numpy() for step in eager["trace"]]
            ),
            rtol=2.0e-10,
            atol=2.0e-10,
        )


def test_float32_guard_tolerance_is_dtype_aware_without_changing_values():
    states = tf.constant([[0.2, -0.1], [0.3, 0.4]], tf.float32)
    bandwidths = tf.broadcast_to(
        0.1 * tf.eye(2, dtype=tf.float32), [2, 2, 2]
    )
    result = linear_gaussian_kdm_observation_factors(
        states,
        tf.zeros_like(states),
        bandwidths,
        tf.zeros_like(bandwidths),
        tf.eye(2, dtype=tf.float32),
        tf.zeros([2, 2], tf.float32),
        0.6 * tf.eye(2, dtype=tf.float32),
        tf.zeros([2, 2], tf.float32),
        tf.constant([0.1, -0.2], tf.float32),
        bandwidth_is_zero=False,
        covariance_tolerance=1.0e-10,
    )
    expected_floor = 64.0 * 2.0**-23
    np.testing.assert_allclose(
        result["effective_covariance_tolerance"].numpy(),
        expected_floor,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert bool(result["valid"].numpy())
