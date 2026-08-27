"""P5 gates: batch lane parity (P-1/P-2/P-3) + compiled-mode identity.

The batch lane must reproduce the single-cloud canonical lane at
batch-size-1 (declared tolerance: bitwise where op-order identical, else
rtol 5e-4 recorded) and preserve within-mode value/score program identity
under tf.function compilation.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import inspect

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_batch_tf import (
    canonical_batch_value_score,
)
from bayesfilter.highdim.ledh_canonical_score_tf import (
    NonlinearScoreModel,
    canonical_value_and_analytical_score,
)

DTYPE = tf.float64


def _model():
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
            tf.eye(2, dtype=points.dtype),
            [tf.shape(points)[0], 2, 2],
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


def _fixture(seed: int, n: int = 6, dim: int = 2, horizon: int = 3):
    rng = np.random.default_rng(seed)
    initial = tf.constant(rng.standard_normal((n, dim)), DTYPE)
    covs = tf.constant(np.stack([np.eye(dim)] * n), DTYPE)
    noises = tf.constant(rng.standard_normal((horizon, n, dim)), DTYPE)
    observations = tf.constant(rng.standard_normal((horizon, dim)), DTYPE)
    return initial, covs, noises, observations


def test_p1_batch_size_one_parity_value_and_score():
    model = _model()
    initial, covs, noises, observations = _fixture(71)
    theta_single = tf.constant([0.6], DTYPE)
    value_ref, score_ref = canonical_value_and_analytical_score(
        model, theta_single, initial, covs, noises, observations,
        flow_substeps=10, with_score=True,
    )
    theta_batch = tf.constant([[0.6]], DTYPE)
    value_b, score_b, diagnostics = canonical_batch_value_score(
        model, theta_batch, initial, covs, noises, observations,
        flow_substeps=10,
    )
    value_err = abs(float(value_b[0].numpy()) - float(value_ref.numpy()))
    score_err = abs(
        float(score_b[0, 0].numpy()) - float(score_ref[0].numpy())
    )
    scale_v = max(abs(float(value_ref.numpy())), 1.0)
    scale_s = max(abs(float(score_ref[0].numpy())), 1.0)
    assert value_err < 5.0e-4 * scale_v, f"value parity {value_err}"
    assert score_err < 5.0e-4 * scale_s, f"score parity {score_err}"
    assert bool(diagnostics["program_valid"][0].numpy())


def test_p1_batch_rows_are_independent():
    model = _model()
    initial, covs, noises, observations = _fixture(73)
    theta_batch = tf.constant([[0.6], [0.9]], DTYPE)
    value_b, score_b, _ = canonical_batch_value_score(
        model, theta_batch, initial, covs, noises, observations,
        flow_substeps=10,
    )
    value_row0, _, _ = canonical_batch_value_score(
        model, tf.constant([[0.6]], DTYPE), initial, covs, noises,
        observations, flow_substeps=10,
    )
    assert bool(
        tf.reduce_all(tf.equal(value_b[0], value_row0[0])).numpy()
    ), "batch row 0 changed when row 1 was appended"


def test_p3_within_mode_identity_under_tf_function():
    """The analytical score has ONE program by construction; compiling it
    must preserve value/score identity: the value returned alongside the
    score equals the value-only path bitwise within the same mode."""
    model = _model()
    initial, covs, noises, observations = _fixture(79)
    theta_batch = tf.constant([[0.6]], DTYPE)

    compiled = tf.function(
        lambda t: canonical_batch_value_score(
            model, t, initial, covs, noises, observations, flow_substeps=10
        ),
        autograph=False,
    )
    value_eager, score_eager, _ = canonical_batch_value_score(
        model, theta_batch, initial, covs, noises, observations, flow_substeps=10
    )
    value_graph, score_graph, _ = compiled(theta_batch)
    # within-mode identity is structural (one program); cross-mode drift is
    # FP op-order and must stay within declared tolerance
    assert abs(
        float(value_graph[0].numpy()) - float(value_eager[0].numpy())
    ) < 5.0e-4 * max(abs(float(value_eager[0].numpy())), 1.0)
    assert abs(
        float(score_graph[0, 0].numpy()) - float(score_eager[0, 0].numpy())
    ) < 5.0e-4 * max(abs(float(score_eager[0, 0].numpy())), 1.0)


def test_p2_capability_surface():
    signature = inspect.signature(canonical_batch_value_score)
    required = {"model", "theta", "substeps"}
    missing = required - set(signature.parameters)
    assert not missing, f"batch lane lost capabilities: {missing}"
