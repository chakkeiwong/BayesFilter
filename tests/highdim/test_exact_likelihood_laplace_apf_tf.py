"""Mechanics tests for fixed exact-likelihood Laplace proposal banks."""

from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    complete_gaussian_mixture_log_density,
)
from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import (
    DTYPE,
    FixedLaplaceConfig,
    make_fixed_laplace_bank_kernel,
    regular_simplex_vertices,
    require_valid_laplace_result,
    single_start_offsets,
)


def _one_step_config(*, start_scale: float = 0.0) -> FixedLaplaceConfig:
    return FixedLaplaceConfig(
        tempering_schedule=(1.0,),
        step_fractions=(1.0,),
        start_scale=start_scale,
        stationarity_relative_tolerance=1.0e-11,
    )


def _linear_gaussian_callbacks(
    observation_matrix: tf.Tensor,
    observation_variance: float,
):
    matrix = tf.convert_to_tensor(observation_matrix, DTYPE)
    variance = tf.constant(observation_variance, DTYPE)

    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        prediction = tf.linalg.matmul(states, matrix, transpose_b=True)
        residual = observation[tf.newaxis, :] - prediction
        dimension = tf.cast(tf.shape(observation)[0], DTYPE)
        return -0.5 * (
            dimension * tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * variance)
            + tf.reduce_sum(tf.square(residual), axis=1) / variance
        )

    def score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        prediction = tf.linalg.matmul(states, matrix, transpose_b=True)
        residual = observation[tf.newaxis, :] - prediction
        return tf.linalg.matmul(residual, matrix) / variance

    def negative_hessian(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        information = tf.linalg.matmul(matrix, matrix, transpose_a=True) / variance
        return tf.broadcast_to(
            information[tf.newaxis, :, :],
            [tf.shape(states)[0], int(matrix.shape[1]), int(matrix.shape[1])],
        )

    return log_likelihood, score, negative_hessian


def test_config_rejects_unexamined_or_incomplete_schedules() -> None:
    with pytest.raises(ValueError, match="equal nonzero length"):
        FixedLaplaceConfig((1.0,), (), 0.0, 1e-8)
    with pytest.raises(ValueError, match="final tempering"):
        FixedLaplaceConfig((0.5,), (1.0,), 0.0, 1e-8)
    with pytest.raises(ValueError, match="nondecreasing"):
        FixedLaplaceConfig((1.0, 0.5, 1.0), (1.0, 1.0, 1.0), 0.0, 1e-8)
    with pytest.raises(ValueError, match="step fractions"):
        FixedLaplaceConfig((1.0,), (0.0,), 0.0, 1e-8)


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_regular_simplex_has_declared_moments(dimension: int) -> None:
    vertices = regular_simplex_vertices(dimension)
    assert vertices.shape == (dimension + 1, dimension)
    tf.debugging.assert_near(
        tf.reduce_mean(vertices, axis=0),
        tf.zeros([dimension], DTYPE),
        atol=2e-14,
    )
    tf.debugging.assert_near(
        tf.einsum("ki,kj->ij", vertices, vertices)
        / tf.cast(dimension + 1, DTYPE),
        tf.eye(dimension, dtype=DTYPE),
        atol=2e-14,
    )


def test_linear_gaussian_one_step_is_exact_including_evidence() -> None:
    batch = 3
    prior_means = tf.constant([[0.3, -0.2], [-0.1, 0.5], [0.7, 0.4]], DTYPE)
    prior_covariance = tf.constant([[1.2, 0.2], [0.2, 0.8]], DTYPE)
    prior_covariances = tf.broadcast_to(prior_covariance[None, :, :], [batch, 2, 2])
    observation_matrix = tf.constant([[1.0, -0.4]], DTYPE)
    observation = tf.constant([0.25], DTYPE)
    variance = 0.6
    callbacks = _linear_gaussian_callbacks(observation_matrix, variance)
    kernel = make_fixed_laplace_bank_kernel(
        batch_size=batch,
        state_dim=2,
        observation_dim=1,
        component_offsets=single_start_offsets(2),
        log_likelihood_fn=callbacks[0],
        state_score_fn=callbacks[1],
        state_negative_hessian_fn=callbacks[2],
        config=_one_step_config(),
        jit_compile=False,
    )
    result = kernel(prior_means, prior_covariances, observation)
    require_valid_laplace_result(result)

    precision = tf.linalg.inv(prior_covariance)
    information = tf.linalg.matmul(
        observation_matrix, observation_matrix, transpose_a=True
    ) / variance
    expected_covariance = tf.linalg.inv(precision + information)
    expected_mean = tf.einsum(
        "ij,bj->bi",
        expected_covariance,
        tf.einsum("ij,bj->bi", precision, prior_means)
        + tf.linalg.matmul(
            tf.broadcast_to(observation[None, :], [batch, 1]),
            observation_matrix,
        )
        / variance,
    )
    tf.debugging.assert_near(
        result["component_means"][:, 0, :], expected_mean, atol=2e-12
    )
    tf.debugging.assert_near(
        result["component_covariances"][:, 0, :, :],
        tf.broadcast_to(expected_covariance[None, :, :], [batch, 2, 2]),
        atol=2e-12,
    )

    innovation = observation[0] - tf.einsum(
        "d,bd->b", observation_matrix[0], prior_means
    )
    innovation_variance = (
        tf.einsum(
            "i,ij,j->", observation_matrix[0], prior_covariance, observation_matrix[0]
        )
        + variance
    )
    expected_evidence = -0.5 * (
        tf.math.log(tf.constant(2.0 * math.pi, DTYPE) * innovation_variance)
        + tf.square(innovation) / innovation_variance
    )
    tf.debugging.assert_near(
        result["component_log_evidence"][:, 0], expected_evidence, atol=3e-12
    )
    objective_before = result["iteration_objective_before_step"][0, :, 0]
    objective_after = result["iteration_objective_after_step"][0, :, 0]
    tf.debugging.assert_greater_equal(
        objective_after,
        objective_before - 2e-14 * (1.0 + tf.abs(objective_before)),
    )
    assert kernel.experimental_get_tracing_count() == 1


def test_indefinite_posterior_precision_fails_closed_without_repair() -> None:
    def log_likelihood(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        return tf.reduce_sum(tf.square(states), axis=1)

    def score(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        return 2.0 * states

    def negative_hessian(states: tf.Tensor, observation: tf.Tensor) -> tf.Tensor:
        del observation
        return tf.broadcast_to(
            -2.0 * tf.eye(1, dtype=DTYPE)[None, :, :],
            [tf.shape(states)[0], 1, 1],
        )

    kernel = make_fixed_laplace_bank_kernel(
        batch_size=2,
        state_dim=1,
        observation_dim=1,
        component_offsets=single_start_offsets(1),
        log_likelihood_fn=log_likelihood,
        state_score_fn=score,
        state_negative_hessian_fn=negative_hessian,
        config=_one_step_config(),
        jit_compile=False,
    )
    result = kernel(
        tf.zeros([2, 1], DTYPE),
        tf.ones([2, 1, 1], DTYPE),
        tf.zeros([1], DTYPE),
    )
    assert not bool(result["valid"].numpy())
    assert bool(result["finite"].numpy())
    with pytest.raises(ValueError, match="invalid row"):
        require_valid_laplace_result(result)


def test_complete_simplex_bank_density_is_permutation_invariant() -> None:
    callbacks = _linear_gaussian_callbacks(tf.constant([[1.0, 0.3]], DTYPE), 0.8)
    kernel = make_fixed_laplace_bank_kernel(
        batch_size=4,
        state_dim=2,
        observation_dim=1,
        component_offsets=regular_simplex_vertices(2),
        log_likelihood_fn=callbacks[0],
        state_score_fn=callbacks[1],
        state_negative_hessian_fn=callbacks[2],
        config=_one_step_config(start_scale=1.0),
        jit_compile=False,
    )
    result = kernel(
        tf.zeros([4, 2], DTYPE),
        tf.broadcast_to(tf.eye(2, dtype=DTYPE)[None, :, :], [4, 2, 2]),
        tf.constant([0.4], DTYPE),
    )
    require_valid_laplace_result(result)
    points = tf.constant(
        [[-0.5, 0.2], [0.1, -0.7], [0.6, 0.9], [-0.2, -0.1]], DTYPE
    )
    weights = result["evidence_component_weights"]
    tf.debugging.assert_near(
        tf.reduce_sum(weights, axis=1), tf.ones([4], DTYPE), atol=2e-14
    )
    direct = complete_gaussian_mixture_log_density(
        points,
        result["component_means"],
        result["component_cholesky"],
        weights,
    )
    permutation = tf.constant([2, 0, 1], tf.int32)
    permuted = complete_gaussian_mixture_log_density(
        points,
        tf.gather(result["component_means"], permutation, axis=1),
        tf.gather(result["component_cholesky"], permutation, axis=1),
        tf.gather(weights, permutation, axis=1),
    )
    tf.debugging.assert_near(direct, permuted, atol=2e-13)
