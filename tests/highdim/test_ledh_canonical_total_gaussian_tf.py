"""Total-derivative gates for the shared Gaussian LEDH score executor."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)


DTYPE = tf.float64


def _parameterized_model(theta_fixed, direction):
    theta_fixed = tf.convert_to_tensor(theta_fixed, DTYPE)
    direction = tf.convert_to_tensor(direction, DTYPE)
    dimension = 2
    observation_matrix = theta_fixed[1] * tf.eye(dimension, dtype=DTYPE)
    d_observation_matrix = direction[1] * tf.eye(dimension, dtype=DTYPE)
    process_covariance = tf.exp(2.0 * theta_fixed[2]) * tf.eye(
        dimension, dtype=DTYPE
    )
    observation_covariance = tf.exp(2.0 * theta_fixed[3]) * tf.eye(
        dimension, dtype=DTYPE
    )

    def transition_mean_fn(theta, points):
        return theta[0] * points + 0.08 * tf.sin(points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            direction[0] * points
            + theta[0] * d_points
            + 0.08 * tf.cos(points) * d_points
        )

    def observation_fn(points):
        return tf.einsum("od,nd->no", observation_matrix, points)

    def observation_tangent_fn(points, d_points):
        return tf.einsum(
            "od,nd->no", d_observation_matrix, points
        ) + tf.einsum("od,nd->no", observation_matrix, d_points)

    def observation_jacobian_fn(points):
        return tf.broadcast_to(
            observation_matrix, [tf.shape(points)[0], dimension, dimension]
        )

    def observation_jacobian_tangent_fn(points, d_points):
        del d_points
        return tf.broadcast_to(
            d_observation_matrix,
            [tf.shape(points)[0], dimension, dimension],
        )

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=observation_fn,
        observation_jacobian_fn=observation_jacobian_fn,
        observation_tangent_fn=observation_tangent_fn,
        process_covariance=process_covariance,
        observation_covariance=observation_covariance,
        observation_jacobian_tangent_fn=observation_jacobian_tangent_fn,
        process_covariance_tangent_fn=lambda theta: (
            2.0
            * tf.exp(2.0 * theta[2])
            * direction[2]
            * tf.eye(dimension, dtype=DTYPE)
        ),
        observation_covariance_tangent_fn=lambda theta: (
            2.0
            * tf.exp(2.0 * theta[3])
            * direction[3]
            * tf.eye(dimension, dtype=DTYPE)
        ),
    )


def _fixture():
    rng = np.random.default_rng(11260907)
    particle_count, dimension, horizon = 8, 2, 2
    base = np.concatenate([np.eye(dimension), -np.eye(dimension)], axis=0)
    return {
        "initial_states": tf.constant(
            rng.standard_normal((particle_count, dimension)), DTYPE
        ),
        "initial_covariances": tf.broadcast_to(
            tf.eye(dimension, dtype=DTYPE),
            [particle_count, dimension, dimension],
        ),
        "noises": tf.constant(
            rng.standard_normal((horizon, particle_count, dimension)), DTYPE
        ),
        "observations": tf.constant(
            rng.standard_normal((horizon, dimension)), DTYPE
        ),
        "options": {
            "flow_substeps": 7,
            "reset_policy": "contract_e",
            "reset_design": tf.constant(
                np.tile(base, (particle_count // (2 * dimension), 1)), DTYPE
            ),
            "reset_sinkhorn_steps": 4,
            "reset_balance_steps": 2,
            "correction_steps": 1,
            "pairwise_steps": 1,
            "coordinate_cap": 0.94,
        },
    }


def _value(theta, fixture):
    model = _parameterized_model(theta, tf.zeros([4], DTYPE))
    value, _ = canonical_value_and_analytical_score(
        model,
        theta,
        fixture["initial_states"],
        fixture["initial_covariances"],
        fixture["noises"],
        fixture["observations"],
        with_score=False,
        **fixture["options"],
    )
    return value


def test_gaussian_fallback_is_total_for_h_q_and_r_directions():
    fixture = _fixture()
    theta = tf.constant([0.82, 1.10, -0.35, -0.20], DTYPE)
    directions = (
        tf.constant([0.0, 1.0, 0.0, 0.0], DTYPE),
        tf.constant([0.0, 0.0, 1.0, 0.0], DTYPE),
        tf.constant([0.0, 0.0, 0.0, 1.0], DTYPE),
        tf.constant([0.4, -0.3, 0.2, 0.5], DTYPE),
    )
    epsilon = tf.constant(2.0e-6, DTYPE)

    for direction in directions:
        model = _parameterized_model(theta, direction)
        _, score = canonical_value_and_analytical_score(
            model,
            theta,
            fixture["initial_states"],
            fixture["initial_covariances"],
            fixture["noises"],
            fixture["observations"],
            with_score=True,
            **fixture["options"],
        )
        finite_difference = (
            _value(theta + epsilon * direction, fixture)
            - _value(theta - epsilon * direction, fixture)
        ) / (2.0 * epsilon)
        error = tf.abs(score[0] - finite_difference)
        scale = tf.maximum(tf.abs(finite_difference), tf.constant(1.0, DTYPE))
        assert float((error / scale).numpy()) < 3.0e-4, (
            direction.numpy(), score.numpy(), finite_difference.numpy()
        )
