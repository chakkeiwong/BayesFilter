from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
from numpy.polynomial.hermite import hermgauss

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as model
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)


def _rule(order: int) -> tuple[tf.Tensor, tf.Tensor]:
    nodes, weights = hermgauss(order)
    return (
        tf.constant(np.sqrt(2.0) * nodes, DTYPE),
        tf.constant(weights / np.sqrt(np.pi), DTYPE),
    )


def test_target_model_local_scores_match_fixed_state_autodiff() -> None:
    parents = tf.constant([[0.2, -0.4, 0.7], [-0.5, 0.1, 0.3]], DTYPE)
    children = tf.constant([[0.1, -0.2, 0.4], [0.6, -0.1, -0.3]], DTYPE)
    observation = tf.constant([0.25, -0.35, 0.15], DTYPE)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        transition_log_density, analytic_transition_score = (
            model._target_transition_log_density_and_score(
                THETA, children, parents
            )
        )
        components = model.lgssm._lgssm_components(THETA, 1)
        predicted = tf.linalg.matmul(
            children,
            components["observation_matrix"],
            transpose_b=True,
        )
        observation_log_density = model.lgssm._gaussian_log_density_forward_core(
            (predicted - observation[None, :])[None, :, :],
            components["observation_covariance"],
        )["value"][0]
    autodiff_transition = tape.jacobian(transition_log_density, THETA)
    autodiff_observation = tape.jacobian(observation_log_density, THETA)
    np.testing.assert_allclose(
        analytic_transition_score,
        autodiff_transition,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        model._target_observation_score(THETA, children, observation),
        autodiff_observation,
        rtol=2e-14,
        atol=2e-14,
    )


def test_conditional_future_likelihood_matches_independent_kalman_loop() -> None:
    points = tf.constant(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.4, 0.1, -0.6]], DTYPE
    )
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][1:5], DTYPE
    )
    actual = model._conditional_future_log_likelihood(THETA, points, observations)
    expected = []
    components = model.lgssm._lgssm_components(THETA, 1)
    for point in tf.unstack(points):
        mean = point
        covariance = tf.zeros([3, 3], DTYPE)
        value = tf.constant(0.0, DTYPE)
        for observation in tf.unstack(observations):
            mean = tf.linalg.matvec(components["transition_matrix"][0], mean)
            covariance = (
                components["transition_matrix"][0]
                @ covariance
                @ tf.transpose(components["transition_matrix"][0])
                + components["transition_covariance"][0]
            )
            predicted = tf.linalg.matvec(components["observation_matrix"], mean)
            innovation_covariance = (
                components["observation_matrix"]
                @ covariance
                @ tf.transpose(components["observation_matrix"])
                + components["observation_covariance"][0]
            )
            value += model.lgssm._gaussian_log_density_forward_core(
                (predicted - observation)[None, None, :],
                innovation_covariance[None, :, :],
            )["value"][0, 0]
            gain = tf.transpose(
                tf.linalg.cholesky_solve(
                    tf.linalg.cholesky(innovation_covariance),
                    components["observation_matrix"] @ covariance,
                )
            )
            mean = mean + tf.linalg.matvec(gain, observation - predicted)
            covariance = covariance - gain @ components["observation_matrix"] @ covariance
            covariance = 0.5 * (covariance + tf.transpose(covariance))
        expected.append(value)
    np.testing.assert_allclose(actual, tf.stack(expected), rtol=2e-14, atol=2e-14)


def test_backward_information_features_match_conditional_kalman_values() -> None:
    points = tf.constant(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.4, 0.1, -0.6]], DTYPE
    )
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:5], DTYPE
    )
    matrices, vectors = model._backward_information_parameters(
        THETA, observations
    )
    for time_index in range(4):
        direct = model._conditional_future_log_likelihood(
            THETA, points, observations[time_index + 1 :]
        )
        direct -= direct[0]
        information_features = model._continuation_features_from_information(
            points, matrices[time_index], vectors[time_index]
        )[-1]
        np.testing.assert_allclose(
            tf.math.log(information_features),
            direct,
            rtol=2e-13,
            atol=2e-13,
        )


def test_backward_information_feature_total_derivative_matches_direct_kalman() -> None:
    points = tf.constant(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.4, 0.1, -0.6]], DTYPE
    )
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:5], DTYPE
    )
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        matrices, vectors = model._backward_information_parameters(
            THETA, observations
        )
        information_log_feature = tf.math.log(
            model._continuation_features_from_information(
                points, matrices[1], vectors[1]
            )[-1]
        )
        direct = model._conditional_future_log_likelihood(
            THETA, points, observations[2:]
        )
        direct -= direct[0]
    information_jacobian = tape.jacobian(information_log_feature, THETA)
    direct_jacobian = tape.jacobian(direct, THETA)
    np.testing.assert_allclose(
        information_jacobian,
        direct_jacobian,
        rtol=3e-12,
        atol=3e-12,
    )


def test_finite_lookahead_information_matches_local_future_windows() -> None:
    points = tf.constant(
        [[0.0, 0.0, 0.0], [0.3, -0.2, 0.5], [-0.4, 0.1, -0.6]], DTYPE
    )
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:6], DTYPE
    )
    matrices, vectors = model._finite_lookahead_information_parameters(
        THETA, observations, 2
    )
    for time_index in range(5):
        stop = min(6, time_index + 3)
        direct = model._conditional_future_log_likelihood(
            THETA, points, observations[time_index + 1 : stop]
        )
        direct -= direct[0]
        feature = model._continuation_features_from_information(
            points, matrices[time_index], vectors[time_index]
        )[-1]
        np.testing.assert_allclose(
            tf.math.log(feature), direct, rtol=3e-13, atol=3e-13
        )


def test_loop_finite_lookahead_matches_unrolled_values_and_total_derivative() -> None:
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:10], DTYPE
    )
    for lookahead_steps in (1, 2, 8):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(THETA)
            old_matrices, old_vectors = model._finite_lookahead_information_parameters(
                THETA, observations, lookahead_steps
            )
            new_matrices, new_vectors = (
                model._finite_lookahead_information_parameters_loop(
                    THETA, observations, lookahead_steps
                )
            )
            old_scalar = tf.reduce_sum(old_matrices) + tf.reduce_sum(old_vectors)
            new_scalar = tf.reduce_sum(new_matrices) + tf.reduce_sum(new_vectors)
        np.testing.assert_allclose(new_matrices, old_matrices, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(new_vectors, old_vectors, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(
            tape.gradient(new_scalar, THETA),
            tape.gradient(old_scalar, THETA),
            rtol=2e-13,
            atol=1e-13,
        )


def test_progressive_feature_marks_are_centered_under_teacher() -> None:
    nodes, weights = _rule(3)
    parents, parent_log_weights, innovations, innovation_log_weights = (
        model.initial_parents(THETA, nodes, weights)
    )
    parent_marks = model._initial_target_model_score_marks(THETA, parents)
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:2], DTYPE
    )
    flow = model._flow_correction(THETA, parents, innovations, observations[0])
    log_weights = (
        tf.repeat(parent_log_weights, tf.shape(innovations)[0])
        + tf.tile(innovation_log_weights, [tf.shape(parents)[0]])
        + flow["log_correction"]
    )
    marks = model._target_model_progressive_score_marks(
        THETA,
        parents,
        parent_log_weights,
        parent_marks,
        flow["particles"],
        observations[0],
    )
    features, mean, centered = model._progressive_features(
        THETA,
        flow["particles"],
        log_weights,
        marks,
        innovations,
        innovation_log_weights,
        observations[1],
    )
    normalized = tf.nn.softmax(log_weights)
    np.testing.assert_allclose(
        tf.einsum("n,np->p", normalized, centered),
        tf.zeros([5], DTYPE),
        rtol=0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        mean,
        tf.einsum("n,np->p", normalized, marks),
        rtol=0.0,
        atol=2e-14,
    )
    assert features.shape[0] == model.PROGRESSIVE_FEATURE_COUNT
