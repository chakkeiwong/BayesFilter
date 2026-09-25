"""Bounded algebra and tangent gates for the diagnostic Younis KDM kernels."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_younis_kdm_tf import (
    ANCHORED_MODEL_IS_ROUTE_ID,
    ATOM_FINITE_TARGET,
    AUXILIARY_ROLE,
    AUXILIARY_ROUTE_ID,
    KDM_FINITE_TARGET,
    MODEL_IS_TARGET,
    ROUTE_ID,
    atom_expectation,
    canonical_linear_gaussian_kdm_auxiliary,
    make_conditional_gaussian_kdm_kernel,
    make_gaussian_kdm_kernel,
    make_iwsg_kernel,
    make_linear_gaussian_kdm_normalizer_kernel,
    make_anchored_pfpf_kdm_weight_kernel,
    make_subspace_gaussian_kdm_kernel,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)


DTYPE = tf.float64


def _trace_model():
    def transition_mean_fn(theta, points):
        return points + theta[..., 0:1] * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            tf.sin(points)
            + d_points
            + theta[..., 0:1] * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return points

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            tf.eye(2, dtype=points.dtype), [tf.shape(points)[0], 2, 2]
        )

    def observation_tangent_fn(points, d_points):
        return d_points

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=tf.constant(0.4 * np.eye(2), DTYPE),
        observation_covariance=tf.constant(0.6 * np.eye(2), DTYPE),
    )


def _mixture_fixture():
    values = np.array([[-0.3], [0.4], [1.1]], dtype=np.float64)
    weights = np.array([0.35, 0.65], dtype=np.float64)
    means = np.array([[0.1], [1.0]], dtype=np.float64)
    covariances = np.array([[[0.7]], [[1.3]]], dtype=np.float64)
    d_values = np.array([[[0.2], [-0.1], [0.3]]], dtype=np.float64)
    d_weights = np.array([[0.15, -0.15]], dtype=np.float64)
    d_means = np.array([[[0.25], [-0.4]]], dtype=np.float64)
    d_covariances = np.array([[[[0.12]], [[-0.08]]]], dtype=np.float64)
    return (
        values,
        weights,
        means,
        covariances,
        d_values,
        d_weights,
        d_means,
        d_covariances,
    )


def _global_kernel(evaluation_count=3, jit_compile=False):
    return make_gaussian_kdm_kernel(
        evaluation_count=evaluation_count,
        component_count=2,
        dimension=1,
        direction_count=1,
        dtype=DTYPE,
        jit_compile=jit_compile,
    )


def _global_log_density(values, weights, means, covariances):
    evaluation_count = int(np.asarray(values).shape[0])
    kernel = _global_kernel(evaluation_count=evaluation_count, jit_compile=False)
    zeros = (
        np.zeros((1, evaluation_count, 1), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1, 2, 1), dtype=np.float64),
        np.zeros((1, 2, 1, 1), dtype=np.float64),
    )
    result = kernel(
        tf.constant(values, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(covariances, DTYPE),
        *(tf.constant(item, DTYPE) for item in zeros),
    )
    return np.asarray(result["log_density"].numpy())


def test_kdm_complete_tangent_matches_finite_difference():
    fixture = _mixture_fixture()
    values, weights, means, covariances, *directions = fixture
    kernel = _global_kernel(jit_compile=False)
    assert kernel.route_id == ROUTE_ID
    assert kernel.target_label == KDM_FINITE_TARGET
    result = kernel(
        *(tf.constant(item, DTYPE) for item in fixture)
    )

    assert np.all(result["valid"].numpy())
    responsibilities = result["responsibilities"].numpy()
    np.testing.assert_allclose(
        responsibilities.sum(axis=1), np.ones(3), rtol=0.0, atol=2e-14
    )

    epsilon = 2.0e-6
    plus = [
        values + epsilon * directions[0][0],
        weights + epsilon * directions[1][0],
        means + epsilon * directions[2][0],
        covariances + epsilon * directions[3][0],
    ]
    minus = [
        values - epsilon * directions[0][0],
        weights - epsilon * directions[1][0],
        means - epsilon * directions[2][0],
        covariances - epsilon * directions[3][0],
    ]
    finite_difference = (
        _global_log_density(*plus) - _global_log_density(*minus)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["d_log_density"].numpy()[0],
        finite_difference,
        rtol=3e-7,
        atol=3e-8,
    )


def test_kdm_component_permutation_and_xla_smoke():
    fixture = _mixture_fixture()
    values, weights, means, covariances, *_ = fixture
    zeros = (
        np.zeros((1, 3, 1), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1, 2, 1), dtype=np.float64),
        np.zeros((1, 2, 1, 1), dtype=np.float64),
    )
    kernel = _global_kernel(jit_compile=False)
    original = kernel(
        tf.constant(values, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(covariances, DTYPE),
        *(tf.constant(item, DTYPE) for item in zeros),
    )
    order = [1, 0]
    permuted = kernel(
        tf.constant(values, DTYPE),
        tf.constant(weights[order], DTYPE),
        tf.constant(means[order], DTYPE),
        tf.constant(covariances[order], DTYPE),
        *(tf.constant(item, DTYPE) for item in zeros),
    )
    np.testing.assert_allclose(
        original["log_density"].numpy(), permuted["log_density"].numpy(),
        rtol=0.0, atol=2e-14,
    )
    np.testing.assert_allclose(
        original["responsibilities"].numpy(),
        permuted["responsibilities"].numpy()[:, order],
        rtol=0.0, atol=2e-14,
    )

    xla_kernel = make_gaussian_kdm_kernel(
        evaluation_count=2,
        component_count=2,
        dimension=1,
        dtype=tf.float32,
        jit_compile=True,
    )
    xla_result = xla_kernel(
        tf.constant(values[:2], tf.float32),
        tf.constant(weights, tf.float32),
        tf.constant(means, tf.float32),
        tf.constant(covariances, tf.float32),
        tf.zeros([1, 2, 1], tf.float32),
        tf.zeros([1, 2], tf.float32),
        tf.zeros([1, 2, 1], tf.float32),
        tf.zeros([1, 2, 1, 1], tf.float32),
    )
    assert np.all(xla_result["valid"].numpy())


def test_kdm_rejects_zero_bandwidth_and_keeps_atom_branch_separate():
    kernel = _global_kernel(jit_compile=False)
    fixture = _mixture_fixture()
    values, weights, means, covariances, *_ = fixture
    zero_covariances = np.zeros_like(covariances)
    zeros = (
        np.zeros((1, 3, 1), dtype=np.float64),
        np.zeros((1, 2), dtype=np.float64),
        np.zeros((1, 2, 1), dtype=np.float64),
        np.zeros((1, 2, 1, 1), dtype=np.float64),
    )
    result = kernel(
        tf.constant(values, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(zero_covariances, DTYPE),
        *(tf.constant(item, DTYPE) for item in zeros),
    )
    assert not np.any(result["valid"].numpy())
    assert np.all(np.isnan(result["log_density"].numpy()))

    atom_values = tf.constant([1.0, 2.0, -0.5], DTYPE)
    atom_weights = tf.constant([0.2, 0.3, 0.5], DTYPE)
    estimate, tangent = atom_expectation(
        atom_values,
        atom_weights,
        d_values=tf.constant([[0.4, -0.2, 0.1]], DTYPE),
        d_weights=tf.constant([[0.1, -0.1, 0.0]], DTYPE),
    )
    np.testing.assert_allclose(estimate.numpy(), 0.55, atol=1e-14)
    np.testing.assert_allclose(tangent.numpy(), [-0.03], atol=1e-14)


def test_conditional_kdm_uses_complete_ancestor_mixture():
    kernel = make_conditional_gaussian_kdm_kernel(
        evaluation_count=3,
        ancestor_count=2,
        component_count=2,
        dimension=1,
        dtype=DTYPE,
        jit_compile=False,
    )
    values = tf.constant([[-0.2], [0.5], [1.4]], DTYPE)
    indices = tf.constant([0, 1, 0], tf.int32)
    weights = tf.constant([[0.25, 0.75], [0.6, 0.4]], DTYPE)
    means = tf.constant([[[0.0], [1.0]], [[-1.0], [1.5]]], DTYPE)
    covariances = tf.constant(
        [[[[0.6]], [[0.9]]], [[[1.1]], [[0.8]]]], DTYPE
    )
    zeros = (
        tf.zeros([1, 3, 1], DTYPE),
        tf.zeros([1, 2, 2], DTYPE),
        tf.zeros([1, 2, 2, 1], DTYPE),
        tf.zeros([1, 2, 2, 1, 1], DTYPE),
    )
    result = kernel(values, indices, weights, means, covariances, *zeros)
    assert kernel.target_label == KDM_FINITE_TARGET
    assert kernel.route_role == "conditional_kdm_proposal_density"
    assert np.all(result["valid"].numpy())
    expected = []
    for row, ancestor in zip(values.numpy(), indices.numpy()):
        expected.append(
            _global_log_density(
                row.reshape(1, 1),
                weights.numpy()[ancestor],
                means.numpy()[ancestor],
                covariances.numpy()[ancestor],
            )[0]
        )
    np.testing.assert_allclose(
        result["log_density"].numpy(), np.asarray(expected),
        rtol=0.0, atol=2e-14,
    )


def test_iwsg_anchor_and_tangent_match_fixed_proposal_difference():
    fixture = _mixture_fixture()
    values, weights, means, covariances, *directions = fixture
    base_density = _global_log_density(values, weights, means, covariances)
    integrand = np.array([0.7, 1.2, -0.4], dtype=np.float64)
    d_integrand = np.array([[0.1, -0.2, 0.3]], dtype=np.float64)
    kernel = make_iwsg_kernel(
        evaluation_count=3,
        component_count=2,
        dimension=1,
        dtype=DTYPE,
        jit_compile=False,
    )
    result = kernel(
        tf.constant(values, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(means, DTYPE),
        tf.constant(covariances, DTYPE),
        *(tf.constant(item, DTYPE) for item in directions[1:]),
        tf.constant(base_density, DTYPE),
        tf.constant(integrand, DTYPE),
        tf.constant(d_integrand, DTYPE),
    )
    assert np.all(result["valid"].numpy())
    np.testing.assert_allclose(result["anchor_weight_error"].numpy(), 0.0, atol=2e-14)
    np.testing.assert_allclose(
        result["estimate"].numpy(), integrand.mean(), rtol=0.0, atol=2e-14
    )

    epsilon = 2.0e-6
    direction = directions

    def estimate_at(sign):
        params = [
            values,
            weights + sign * epsilon * direction[1][0],
            means + sign * epsilon * direction[2][0],
            covariances + sign * epsilon * direction[3][0],
        ]
        zeros = [
            np.zeros((1, 2), dtype=np.float64),
            np.zeros((1, 2, 1), dtype=np.float64),
            np.zeros((1, 2, 1, 1), dtype=np.float64),
        ]
        out = kernel(
            tf.constant(params[0], DTYPE),
            tf.constant(params[1], DTYPE),
            tf.constant(params[2], DTYPE),
            tf.constant(params[3], DTYPE),
            *(tf.constant(item, DTYPE) for item in zeros),
            tf.constant(base_density, DTYPE),
            tf.constant(integrand, DTYPE),
            tf.constant(d_integrand * 0.0, DTYPE),
        )
        return float(out["estimate"].numpy())

    finite_difference = (estimate_at(1.0) - estimate_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["d_estimate"].numpy()[0], finite_difference + d_integrand.mean(),
        rtol=4e-7, atol=4e-8,
    )


def test_anchored_pfpf_kdm_correction_freezes_proposal_denominator():
    kernel = make_anchored_pfpf_kdm_weight_kernel(
        particle_count=3,
        ancestor_count=2,
        direction_count=1,
        dtype=DTYPE,
        jit_compile=False,
    )
    assert kernel.target_label == MODEL_IS_TARGET
    assert kernel.route_id == ANCHORED_MODEL_IS_ROUTE_ID
    assert (
        kernel.proposal_derivative_policy
        == "frozen_anchor_no_denominator_tangent_v1"
    )
    ancestor_log_weights = tf.math.log(tf.constant([0.6, 0.4], DTYPE))
    ancestor_log_selection_probs = tf.math.log(tf.constant([0.55, 0.45], DTYPE))
    ancestor_indices = tf.constant([0, 1, 0], tf.int32)
    transition = tf.constant([-0.1, -0.3, -0.2], DTYPE)
    observation = tf.constant([-0.4, -0.8, -0.5], DTYPE)
    logdet = tf.constant([0.05, 0.1, 0.0], DTYPE)
    proposal = tf.constant([-0.6, -0.9, -0.7], DTYPE)
    d_ancestor = tf.constant([[0.11, -0.13]], DTYPE)
    d_transition = tf.constant([[0.02, -0.01, 0.03]], DTYPE)
    d_observation = tf.constant([[-0.04, 0.06, -0.02]], DTYPE)
    d_logdet = tf.constant([[0.01, 0.02, 0.0]], DTYPE)
    result = kernel(
        ancestor_log_weights,
        ancestor_log_selection_probs,
        ancestor_indices,
        transition,
        observation,
        logdet,
        tf.ones([3], tf.bool),
        proposal,
        d_ancestor,
        d_transition,
        d_observation,
        d_logdet,
    )
    target_logs = np.log([0.6, 0.4])
    selection_logs = np.log([0.55, 0.45])
    expected = np.array([
        target_logs[0] - 0.1 - 0.4 + 0.05 - selection_logs[0] + 0.6,
        target_logs[1] - 0.3 - 0.8 + 0.1 - selection_logs[1] + 0.9,
        target_logs[0] - 0.2 - 0.5 + 0.0 - selection_logs[0] + 0.7,
    ])
    np.testing.assert_allclose(result["log_weights"].numpy(), expected, atol=1e-14)
    expected_d = np.array([
        0.11 + 0.02 - 0.04 + 0.01,
        -0.13 - 0.01 + 0.06 + 0.02,
        0.11 + 0.03 - 0.02 + 0.0,
    ])
    np.testing.assert_allclose(
        result["d_log_weights"].numpy()[0], expected_d, atol=1e-14
    )
    assert bool(result["valid"].numpy().all())

    invalid = kernel(
        ancestor_log_weights,
        ancestor_log_selection_probs,
        ancestor_indices,
        transition,
        observation,
        logdet,
        tf.constant([True, False, True]),
        proposal,
        d_ancestor,
        d_transition,
        d_observation,
        d_logdet,
    )
    assert not bool(invalid["valid"].numpy()[1])

    unnormalized = kernel(
        ancestor_log_weights + tf.constant(0.1, DTYPE),
        ancestor_log_selection_probs,
        ancestor_indices,
        transition,
        observation,
        logdet,
        tf.ones([3], tf.bool),
        proposal,
        d_ancestor,
        d_transition,
        d_observation,
        d_logdet,
    )
    assert not bool(unnormalized["valid"].numpy().any())
    assert np.isnan(invalid["log_weights"].numpy()[1])


def _normalizer_fixture():
    return (
        np.array([[-0.4], [0.2], [1.1]], dtype=np.float64),
        np.array([0.2, 0.5, 0.3], dtype=np.float64),
        np.array([[[0.25]], [[0.4]], [[0.7]]], dtype=np.float64),
        np.array([[1.3]], dtype=np.float64),
        np.array([[0.6]], dtype=np.float64),
        np.array([0.25], dtype=np.float64),
        np.array([[[0.15], [-0.2], [0.3]]], dtype=np.float64),
        np.array([[0.08, -0.03, -0.05]], dtype=np.float64),
        np.array([[[[0.06]], [[-0.04]], [[0.1]]]], dtype=np.float64),
        np.array([[[0.12]]], dtype=np.float64),
        np.array([[[0.05]]], dtype=np.float64),
        np.array([[0.07]], dtype=np.float64),
    )


def _normalizer_kernel(*, bandwidth_is_zero):
    return make_linear_gaussian_kdm_normalizer_kernel(
        particle_count=3,
        state_dimension=1,
        observation_dimension=1,
        dtype=DTYPE,
        bandwidth_is_zero=bandwidth_is_zero,
        jit_compile=False,
    )


def test_linear_gaussian_kdm_normalizer_complete_tangent_and_target_shift():
    fixture = _normalizer_fixture()
    kernel = _normalizer_kernel(bandwidth_is_zero=False)
    result = kernel(*(tf.constant(item, DTYPE) for item in fixture))
    assert bool(result["valid"].numpy())
    assert kernel.target_label == KDM_FINITE_TARGET

    epsilon = 2.0e-6
    base = list(fixture)
    directions = fixture[6:]

    def value_at(sign):
        params = list(base)
        for index, direction in zip(range(6), directions[:6]):
            params[index] = base[index] + sign * epsilon * direction[0]
        zeros = [
            np.zeros_like(fixture[6]),
            np.zeros_like(fixture[7]),
            np.zeros_like(fixture[8]),
            np.zeros_like(fixture[9]),
            np.zeros_like(fixture[10]),
            np.zeros_like(fixture[11]),
        ]
        params[6:] = zeros
        out = kernel(*(tf.constant(item, DTYPE) for item in params))
        return float(out["value"].numpy())

    finite_difference = (value_at(1.0) - value_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["score"].numpy()[0], finite_difference,
        rtol=4e-7, atol=4e-8,
    )

    zero_tangents = [np.zeros_like(item) for item in fixture[6:]]
    for active_index, active_direction in enumerate(fixture[6:]):
        tangents = list(zero_tangents)
        tangents[active_index] = active_direction

        def active_value_at(sign):
            params = list(fixture[:6])
            params[active_index] = (
                fixture[active_index]
                + sign * epsilon * active_direction[0]
            )
            out = kernel(
                *(tf.constant(item, DTYPE) for item in [*params, *tangents])
            )
            return float(out["value"].numpy())

        active_fd = (
            active_value_at(1.0) - active_value_at(-1.0)
        ) / (2.0 * epsilon)
        active_result = kernel(
            *(tf.constant(item, DTYPE) for item in [*fixture[:6], *tangents])
        )
        np.testing.assert_allclose(
            active_result["score"].numpy()[0], active_fd,
            rtol=5e-7, atol=5e-8,
        )

    atom_fixture = list(fixture)
    atom_fixture[2] = np.zeros_like(atom_fixture[2])
    atom_fixture[8] = np.zeros_like(atom_fixture[8])
    atom_kernel = _normalizer_kernel(bandwidth_is_zero=True)
    atom_result = atom_kernel(
        *(tf.constant(item, DTYPE) for item in atom_fixture)
    )
    assert bool(atom_result["valid"].numpy())
    assert atom_kernel.target_label == ATOM_FINITE_TARGET
    assert abs(float(result["value"].numpy()) - float(atom_result["value"].numpy())) > 1e-5

    def atom_value_at(sign):
        params = list(atom_fixture)
        for index, direction in (
            (0, directions[0]),
            (1, directions[1]),
            (3, directions[3]),
            (4, directions[4]),
            (5, directions[5]),
        ):
            params[index] = atom_fixture[index] + sign * epsilon * direction[0]
        params[6:] = [np.zeros_like(item) for item in atom_fixture[6:]]
        out = atom_kernel(*(tf.constant(item, DTYPE) for item in params))
        return float(out["value"].numpy())

    atom_fd = (atom_value_at(1.0) - atom_value_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        atom_result["score"].numpy()[0], atom_fd,
        rtol=4e-7, atol=4e-8,
    )


def test_linear_gaussian_kdm_normalizer_rejects_wrong_bandwidth_branch():
    fixture = _normalizer_fixture()
    positive_kernel = _normalizer_kernel(bandwidth_is_zero=False)
    zero_fixture = list(fixture)
    zero_fixture[2] = np.zeros_like(zero_fixture[2])
    zero_fixture[8] = np.zeros_like(zero_fixture[8])
    invalid_positive = positive_kernel(
        *(tf.constant(item, DTYPE) for item in zero_fixture)
    )
    assert not bool(invalid_positive["valid"].numpy())

    atom_kernel = _normalizer_kernel(bandwidth_is_zero=True)
    invalid_atom = atom_kernel(*(tf.constant(item, DTYPE) for item in fixture))
    assert not bool(invalid_atom["valid"].numpy())


def test_linear_gaussian_kdm_normalizer_multivariate_shape_smoke():
    rng = np.random.default_rng(119)
    particle_count, state_dimension, observation_dimension = 4, 2, 2
    direction_count = 2
    states = rng.normal(size=(particle_count, state_dimension))
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    raw = rng.normal(size=(particle_count, state_dimension, state_dimension))
    bandwidths = np.einsum("nij,nkj->nik", raw, raw) + 0.4 * np.eye(2)[None]
    observation_matrix = rng.normal(size=(observation_dimension, state_dimension))
    observation_covariance = np.array([[0.8, 0.1], [0.1, 0.7]])
    observation = rng.normal(size=(observation_dimension,))
    d_states = rng.normal(size=(direction_count, particle_count, state_dimension))
    d_weights = np.array(
        [[0.02, -0.01, -0.005, -0.005], [-0.01, 0.01, 0.0, 0.0]]
    )
    d_bandwidths = rng.normal(
        size=(direction_count, particle_count, state_dimension, state_dimension)
    )
    d_bandwidths = 0.5 * (d_bandwidths + d_bandwidths.transpose(0, 1, 3, 2))
    d_observation_matrix = rng.normal(
        size=(direction_count, observation_dimension, state_dimension)
    )
    d_observation_covariance = rng.normal(
        size=(direction_count, observation_dimension, observation_dimension)
    )
    d_observation_covariance = 0.5 * (
        d_observation_covariance + d_observation_covariance.transpose(0, 2, 1)
    )
    d_observation = rng.normal(size=(direction_count, observation_dimension))
    kernel = make_linear_gaussian_kdm_normalizer_kernel(
        particle_count=particle_count,
        state_dimension=state_dimension,
        observation_dimension=observation_dimension,
        direction_count=direction_count,
        dtype=DTYPE,
        jit_compile=False,
    )
    result = kernel(
        *(tf.constant(item, DTYPE) for item in [
            states,
            weights,
            bandwidths,
            observation_matrix,
            observation_covariance,
            observation,
            d_states,
            d_weights,
            d_bandwidths,
            d_observation_matrix,
            d_observation_covariance,
            d_observation,
        ])
    )
    assert bool(result["valid"].numpy())
    assert result["score"].shape == (direction_count,)
    assert np.all(np.isfinite(result["score"].numpy()))


def _subspace_fixture():
    return (
        np.array([[-0.4, 0.0], [0.2, 0.0], [1.1, 0.0]], dtype=np.float64),
        np.array([0.2, 0.5, 0.3], dtype=np.float64),
        np.array([[0.1, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float64),
        np.array([[[0.25]], [[0.4]], [[0.7]]], dtype=np.float64),
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([[[0.15, 0.0], [-0.2, 0.0], [0.3, 0.0]]], dtype=np.float64),
        np.array([[0.08, -0.03, -0.05]], dtype=np.float64),
        np.array([[[0.05, 0.0], [-0.1, 0.0], [0.15, 0.0]]], dtype=np.float64),
        np.array([[[[0.06]], [[-0.04]], [[0.1]]]], dtype=np.float64),
    )


def _subspace_kernel():
    return make_subspace_gaussian_kdm_kernel(
        evaluation_count=3,
        component_count=3,
        ambient_dimension=2,
        support_dimension=1,
        dtype=DTYPE,
        jit_compile=False,
    )


def test_subspace_kdm_handles_rank_deficient_support_and_complete_tangent():
    fixture = _subspace_fixture()
    kernel = _subspace_kernel()
    result = kernel(*(tf.constant(item, DTYPE) for item in fixture))
    assert bool(result["valid"].numpy().all())
    assert kernel.chart_policy == "fixed_orthonormal_support_chart_v1"
    np.testing.assert_allclose(
        result["support_residual_max"].numpy(), 0.0, atol=2e-14
    )

    epsilon = 2.0e-6
    base = list(fixture)
    directions = fixture[5:]

    def value_at(sign):
        params = list(base[:5])
        for index, direction in zip((0, 1, 2, 3), directions):
            params[index] = base[index] + sign * epsilon * direction[0]
        zeros = [np.zeros_like(item) for item in directions]
        out = kernel(*(tf.constant(item, DTYPE) for item in [*params, *zeros]))
        return np.asarray(out["log_density"].numpy())

    finite_difference = (value_at(1.0) - value_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["d_log_density"].numpy()[0], finite_difference,
        rtol=5e-7, atol=5e-8,
    )


def test_subspace_kdm_rejects_off_support_points_and_bad_chart():
    fixture = list(_subspace_fixture())
    kernel = _subspace_kernel()
    fixture[0][0, 1] = 1.0e-3
    off_support = kernel(*(tf.constant(item, DTYPE) for item in fixture))
    assert not bool(off_support["valid"].numpy()[0])
    assert bool(off_support["valid"].numpy()[1])

    fixture = list(_subspace_fixture())
    fixture[4] = np.array([[1.0], [0.2]], dtype=np.float64)
    bad_chart = kernel(*(tf.constant(item, DTYPE) for item in fixture))
    assert not bool(bad_chart["basis_valid"].numpy())
    assert not bool(bad_chart["valid"].numpy().any())


def test_subspace_kdm_accepts_rank_deficient_covariance_that_ambient_kdm_rejects():
    fixture = _subspace_fixture()
    subspace = _subspace_kernel()(*(tf.constant(item, DTYPE) for item in fixture))
    assert bool(subspace["valid"].numpy().all())

    ambient_kernel = make_gaussian_kdm_kernel(
        evaluation_count=3,
        component_count=3,
        dimension=2,
        dtype=DTYPE,
        jit_compile=False,
    )
    rank_deficient_covariances = np.zeros((3, 2, 2), dtype=np.float64)
    rank_deficient_covariances[:, 0, 0] = fixture[3][:, 0, 0]
    ambient = ambient_kernel(
        tf.constant(fixture[0], DTYPE),
        tf.constant(fixture[1], DTYPE),
        tf.constant(fixture[2], DTYPE),
        tf.constant(rank_deficient_covariances, DTYPE),
        tf.zeros([1, 3, 2], DTYPE),
        tf.zeros([1, 3], DTYPE),
        tf.zeros([1, 3, 2], DTYPE),
        tf.zeros([1, 3, 2, 2], DTYPE),
    )
    assert not bool(ambient["valid"].numpy().any())


def test_kdm_atom_sidecar_reads_the_actual_canonical_endpoint_trace():
    model = _trace_model()
    rng = np.random.default_rng(127)
    initial_states = tf.constant(rng.normal(size=(4, 2)), DTYPE)
    initial_covariances = tf.constant(np.stack([np.eye(2)] * 4), DTYPE)
    noises = tf.constant(rng.normal(size=(1, 4, 2)), DTYPE)
    observations = tf.constant(rng.normal(size=(1, 2)), DTYPE)
    design = tf.constant(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], DTYPE
    )
    theta = tf.constant([0.6], DTYPE)
    canonical_kwargs = dict(
        flow_substeps=2,
        with_score=True,
        reset_policy="contract_e",
        reset_design=design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=2,
        reset_balance_steps=2,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_damping=1.0e-2,
        correction_lm_scale_floor=1.0e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.98,
        coordinate_cap_power=8,
    )

    def run(theta_value, *, with_tangent):
        kwargs = dict(canonical_kwargs)
        kwargs["with_score"] = with_tangent
        return canonical_value_and_analytical_score(
            model,
            tf.constant([theta_value], DTYPE),
            initial_states,
            initial_covariances,
            noises,
            observations,
            return_trace=True,
            **kwargs,
        )

    value, score, trace = run(0.6, with_tangent=True)
    assert len(trace) == 1
    record = trace[0]
    required = {
        "pre_flow",
        "d_pre_flow",
        "children",
        "d_children",
        "posterior_weights",
        "d_posterior_weights",
        "prior_observation_weights",
        "d_prior_observation_weights",
        "prior_observation_log_normalizer",
        "d_prior_observation_log_normalizer",
        "observation",
        "states_after_reset",
        "d_states_after_reset",
    }
    assert required.issubset(record)
    assert bool(tf.reduce_all(tf.math.is_finite(record["children"])).numpy())
    np.testing.assert_allclose(
        tf.reduce_sum(record["prior_observation_weights"]).numpy(),
        1.0,
        atol=2e-12,
    )

    kdm_atom = make_linear_gaussian_kdm_normalizer_kernel(
        particle_count=4,
        state_dimension=2,
        observation_dimension=2,
        dtype=DTYPE,
        bandwidth_is_zero=True,
        jit_compile=False,
    )
    zero_bandwidth = tf.zeros([4, 2, 2], DTYPE)
    zero_d_bandwidth = tf.zeros([1, 4, 2, 2], DTYPE)
    identity = tf.eye(2, dtype=DTYPE)
    zero_model_tangents = (
        tf.zeros([1, 2, 2], DTYPE),
        tf.zeros([1, 2, 2], DTYPE),
        tf.zeros([1, 2], DTYPE),
    )

    def sidecar_value(theta_value, *, tangent):
        _, _, current_trace = run(theta_value, with_tangent=tangent)
        current = current_trace[0]
        d_states = current["d_children"][None, :, :] if tangent else tf.zeros(
            [1, 4, 2], DTYPE
        )
        d_weights = (
            current["d_prior_observation_weights"][None, :]
            if tangent
            else tf.zeros([1, 4], DTYPE)
        )
        result = kdm_atom(
            current["children"],
            current["prior_observation_weights"],
            zero_bandwidth,
            identity,
            tf.constant(0.6 * np.eye(2), DTYPE),
            current["observation"],
            d_states,
            d_weights,
            zero_d_bandwidth,
            *zero_model_tangents,
        )
        assert bool(result["valid"].numpy())
        return result, current

    sidecar, current = sidecar_value(0.6, tangent=True)
    _, score_without_trace = canonical_value_and_analytical_score(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        **canonical_kwargs,
    )
    np.testing.assert_allclose(score_without_trace.numpy(), score.numpy(), atol=2e-12)
    np.testing.assert_allclose(
        (
            current["prior_observation_log_normalizer"]
            + sidecar["value"]
        ).numpy(),
        value.numpy(),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        (
            current["d_prior_observation_log_normalizer"]
            + sidecar["score"][0]
        ).numpy(),
        score.numpy()[0],
        rtol=2e-10,
        atol=2e-10,
    )
    epsilon = 2.0e-5
    plus = sidecar_value(0.6 + epsilon, tangent=False)[0]["value"].numpy()
    minus = sidecar_value(0.6 - epsilon, tangent=False)[0]["value"].numpy()
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        sidecar["score"].numpy()[0], finite_difference,
        rtol=2e-4, atol=2e-5,
    )
    assert np.isfinite(value.numpy())


@pytest.mark.parametrize("reset_steps", [2, 8])
def test_auxiliary_api_uses_canonical_trace_and_has_no_feedback(reset_steps):
    model = _trace_model()
    rng = np.random.default_rng(131)
    initial_states = tf.constant(rng.normal(size=(4, 2)), DTYPE)
    initial_covariances = tf.constant(np.stack([np.eye(2)] * 4), DTYPE)
    noises = tf.constant(rng.normal(size=(2, 4, 2)), DTYPE)
    observations = tf.constant(rng.normal(size=(2, 2)), DTYPE)
    design = tf.constant(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], DTYPE
    )
    theta = tf.constant([0.6], DTYPE)
    options = {
        "flow_substeps": 2,
        "reset_policy": "contract_e",
        "reset_design": design,
        "reset_epsilon": 2.0,
        "reset_sinkhorn_steps": reset_steps,
        "reset_balance_steps": reset_steps,
        "correction_steps": 4,
        "correction_strength": 0.2,
        "correction_lm_damping": 1.0e-2,
        "correction_lm_scale_floor": 1.0e-4,
        "correction_trust_radius": 0.5,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "pairwise_rms_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    }
    bandwidths = tf.broadcast_to(
        tf.constant([[[0.25, 0.0], [0.0, 0.25]]], DTYPE),
        [2, 4, 2, 2],
    )

    def auxiliary_at(theta_value):
        return canonical_linear_gaussian_kdm_auxiliary(
            model,
            tf.constant([theta_value], DTYPE),
            initial_states,
            initial_covariances,
            noises,
            observations,
            tf.eye(2, dtype=DTYPE),
            tf.zeros([2, 2], DTYPE),
            bandwidths,
            tf.zeros([2, 4, 2, 2], DTYPE),
            canonical_options=options,
            jit_compile=False,
        )

    result = auxiliary_at(0.6)
    assert result["route_id"] == AUXILIARY_ROUTE_ID
    assert result["route_role"] == AUXILIARY_ROLE
    assert bool(result["valid"].numpy()) == (reset_steps == 8)
    assert bool(result["canonical_value_unchanged"].numpy())
    assert not bool(result["kdm_feedback_into_canonical"].numpy())
    assert result["canonical_target_label"] == ATOM_FINITE_TARGET
    assert result["auxiliary_target_label"] == KDM_FINITE_TARGET
    assert abs(float(result["value_shift"].numpy())) > 1e-10
    direct_value, direct_score = canonical_value_and_analytical_score(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        with_score=True,
        **options,
    )
    np.testing.assert_allclose(result["canonical_value"].numpy(), direct_value.numpy())
    np.testing.assert_allclose(result["canonical_score"].numpy(), direct_score.numpy())
    if reset_steps == 2:
        # The first reset's column-mass residual exceeds the existing 1e-4 gate.
        # Preserve the original fixture as rejection evidence; do not weaken it.
        assert np.isneginf(result["canonical_value"].numpy())
        np.testing.assert_array_equal(result["canonical_score"].numpy(), [0.])
        return
    epsilon = 2.0e-5
    finite_difference = (
        auxiliary_at(0.6 + epsilon)["kdm_auxiliary_value"].numpy()
        - auxiliary_at(0.6 - epsilon)["kdm_auxiliary_value"].numpy()
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result["kdm_auxiliary_score"].numpy()[0],
        finite_difference,
        rtol=4e-4,
        atol=4e-5,
    )
