"""Contract-E covariance-carry and total-tangent call-chain gates."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_reset_score_tf import (
    sinkhorn_contract_e_reset_triple_with_tangent,
)
from bayesfilter.highdim.ledh_canonical_score_stages_tf import (
    ukf_predict_with_parameter_tangent,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)


DTYPE = tf.float64


def _reset_inputs():
    rng = np.random.default_rng(26090731)
    particle_count, dimension = 8, 2
    children = rng.normal(size=(particle_count, dimension))
    d_children = 0.15 * rng.normal(size=(particle_count, dimension))
    weights = np.array([0.06, 0.09, 0.12, 0.18, 0.11, 0.14, 0.13, 0.17])
    d_weights = np.array([0.01, -0.02, 0.015, -0.005, 0.012, -0.007, 0.003, -0.008])
    assert abs(float(np.sum(d_weights))) < 1.0e-15
    covariances = np.stack(
        [
            np.array(
                [
                    [0.45 + 0.08 * index, 0.015 * (index - 2)],
                    [0.015 * (index - 2), 0.60 + 0.05 * index],
                ]
            )
            for index in range(particle_count)
        ]
    )
    raw_d_covariances = 0.025 * rng.normal(
        size=(particle_count, dimension, dimension)
    )
    d_covariances = 0.5 * (
        raw_d_covariances + raw_d_covariances.transpose(0, 2, 1)
    )
    base = np.concatenate([np.eye(dimension), -np.eye(dimension)], axis=0)
    design = np.tile(base, (particle_count // (2 * dimension), 1))
    return (
        children,
        d_children,
        covariances,
        d_covariances,
        weights,
        d_weights,
        design,
    )


def _reset_call(values, *, zero_tangents: bool = False):
    (
        children,
        d_children,
        covariances,
        d_covariances,
        weights,
        d_weights,
        design,
    ) = values
    if zero_tangents:
        d_children = np.zeros_like(d_children)
        d_covariances = np.zeros_like(d_covariances)
        d_weights = np.zeros_like(d_weights)
    return sinkhorn_contract_e_reset_triple_with_tangent(
        tf.constant(children, DTYPE),
        tf.constant(d_children, DTYPE),
        tf.constant(covariances, DTYPE),
        tf.constant(d_covariances, DTYPE),
        tf.constant(weights, DTYPE),
        tf.constant(d_weights, DTYPE),
        tf.constant(design, DTYPE),
        epsilon=0.55,
        sinkhorn_steps=4,
        balance_steps=2,
        ridge=1.0e-6,
    )


def test_contract_e_covariance_uses_same_target_by_source_transport():
    values = _reset_inputs()
    (
        _,
        _,
        carried,
        d_carried,
        transport,
        d_transport,
    ) = _reset_call(values)
    covariances = tf.constant(values[2], DTYPE)
    d_covariances = tf.constant(values[3], DTYPE)
    expected = tf.einsum("ri,iab->rab", transport, covariances)
    expected_tangent = (
        tf.einsum("ri,iab->rab", d_transport, covariances)
        + tf.einsum("ri,iab->rab", transport, d_covariances)
    )

    np.testing.assert_allclose(
        tf.reduce_sum(transport, axis=1).numpy(),
        np.ones(8),
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        tf.reduce_sum(d_transport, axis=1).numpy(),
        np.zeros(8),
        rtol=0.0,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(carried.numpy(), expected.numpy(), rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        d_carried.numpy(), expected_tangent.numpy(), rtol=2e-13, atol=2e-13
    )
    assert float(tf.reduce_max(tf.abs(carried - covariances)).numpy()) > 1.0e-3


def test_contract_e_covariance_carry_total_tangent_matches_finite_difference():
    values = list(_reset_inputs())
    result = _reset_call(values)
    epsilon = 2.0e-6

    def value_at(sign: float):
        perturbed = [np.array(value, copy=True) for value in values]
        perturbed[0] = values[0] + sign * epsilon * values[1]
        perturbed[2] = values[2] + sign * epsilon * values[3]
        perturbed[4] = values[4] + sign * epsilon * values[5]
        return _reset_call(perturbed, zero_tangents=True)[2]

    finite_difference = (value_at(1.0) - value_at(-1.0)) / (2.0 * epsilon)
    np.testing.assert_allclose(
        result[3].numpy(),
        finite_difference.numpy(),
        rtol=3.0e-6,
        atol=3.0e-8,
    )


def _linear_model(matrix_base: tf.Tensor):
    def transition_mean_fn(theta, points):
        return tf.einsum("ij,nj->ni", theta[0] * matrix_base, points)

    def transition_mean_tangent_fn(theta, points, d_points):
        return (
            tf.einsum("ij,nj->ni", matrix_base, points)
            + tf.einsum("ij,nj->ni", theta[0] * matrix_base, d_points)
        )

    return NonlinearScoreModel(
        transition_mean_fn=transition_mean_fn,
        transition_mean_tangent_fn=transition_mean_tangent_fn,
        observation_fn=lambda points: points,
        observation_jacobian_fn=lambda points: tf.broadcast_to(
            tf.eye(2, dtype=DTYPE), [tf.shape(points)[0], 2, 2]
        ),
        observation_tangent_fn=lambda points, d_points: d_points,
        process_covariance=0.3 * tf.eye(2, dtype=DTYPE),
        observation_covariance=0.5 * tf.eye(2, dtype=DTYPE),
        observation_jacobian_tangent_fn=lambda points, d_points: tf.zeros(
            [tf.shape(points)[0], 2, 2], DTYPE
        ),
    )


def test_two_step_canonical_endpoint_consumes_carried_covariance_and_tangent():
    rng = np.random.default_rng(26090732)
    particle_count, dimension, horizon = 8, 2, 2
    matrix_base = tf.constant([[0.82, 0.10], [-0.06, 0.74]], DTYPE)
    model = _linear_model(matrix_base)
    theta = tf.constant([0.78], DTYPE)
    initial_states = tf.constant(
        rng.normal(size=(particle_count, dimension)), DTYPE
    )
    initial_covariances = tf.stack(
        [
            tf.constant(
                [[0.4 + 0.08 * i, 0.01 * i], [0.01 * i, 0.7 + 0.04 * i]],
                DTYPE,
            )
            for i in range(particle_count)
        ]
    )
    noises = tf.constant(
        rng.normal(size=(horizon, particle_count, dimension)), DTYPE
    )
    observations = tf.constant(rng.normal(size=(horizon, dimension)), DTYPE)
    base = np.concatenate([np.eye(dimension), -np.eye(dimension)], axis=0)
    design = tf.constant(
        np.tile(base, (particle_count // (2 * dimension), 1)), DTYPE
    )
    value, score, trace = canonical_value_and_analytical_score(
        model,
        theta,
        initial_states,
        initial_covariances,
        noises,
        observations,
        flow_substeps=6,
        reset_policy="contract_e",
        reset_design=design,
        reset_sinkhorn_steps=4,
        reset_balance_steps=2,
        correction_steps=1,
        pairwise_steps=1,
        coordinate_cap=0.95,
        with_score=True,
        return_trace=True,
    )
    assert bool(tf.math.is_finite(value).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())

    first = trace[0]
    second = trace[1]
    expected_carried = tf.einsum(
        "ri,iab->rab", first["reset_transport"], first["post_covariances"]
    )
    # The value identity is the orientation/call-chain check. The tangent is
    # already checked independently above and is consumed below by UKF.
    np.testing.assert_allclose(
        first["covariances_after_reset"].numpy(),
        expected_carried.numpy(),
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    transition = theta[0] * matrix_base

    def mean_fn(points):
        return tf.einsum("ij,nj->ni", transition, points)

    def mean_tangent_fn(points, d_points):
        return (
            tf.einsum("ij,nj->ni", matrix_base, points)
            + tf.einsum("ij,nj->ni", transition, d_points)
        )

    expected_next = ukf_predict_with_parameter_tangent(
        first["states_after_reset"],
        first["covariances_after_reset"],
        first["d_states_after_reset"],
        first["d_covariances_after_reset"],
        mean_fn,
        mean_tangent_fn,
        model.process_covariance,
    )
    wrong_source_order = ukf_predict_with_parameter_tangent(
        first["states_after_reset"],
        first["post_covariances"],
        first["d_states_after_reset"],
        tf.zeros_like(first["post_covariances"]),
        mean_fn,
        mean_tangent_fn,
        model.process_covariance,
    )
    np.testing.assert_allclose(
        second["predicted_covariances"].numpy(),
        expected_next[1].numpy(),
        rtol=3.0e-12,
        atol=3.0e-12,
    )
    assert (
        float(
            tf.reduce_max(
                tf.abs(second["predicted_covariances"] - wrong_source_order[1])
            ).numpy()
        )
        > 1.0e-4
    )
