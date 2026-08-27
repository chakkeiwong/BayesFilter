"""Focused tests for the source-faithful LEDH-PFPF fixture."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.testing.particle_authority_ledh_tf import ledh_flow, ledh_inverse


def test_ledh_flow_inverse_and_determinant_product() -> None:
    points = tf.constant([[-1.0, -0.5], [0.0, 0.2], [1.0, 0.8]], tf.float64)
    prior_mean = tf.constant([0.2, -0.1], tf.float64)
    prior_covariance = tf.constant([[1.0, 0.2], [0.2, 1.5]], tf.float64)
    observation_matrix = tf.constant([[1.0, 0.3], [-0.2, 1.0]], tf.float64)
    observation_covariance = tf.constant([[0.4, 0.05], [0.05, 0.6]], tf.float64)
    observation = tf.constant([0.7, -0.4], tf.float64)
    steps = tf.fill([10], tf.constant(0.1, tf.float64))
    final, flow = ledh_flow(
        points, prior_mean, prior_covariance, observation_matrix,
        observation_covariance, observation, steps
    )
    recovered = ledh_inverse(final, flow)
    assert float(tf.reduce_max(tf.abs(recovered - points)).numpy()) <= 1.0e-10
    composed = tf.eye(2, dtype=tf.float64)
    for matrix in flow["matrices"]:
        composed = tf.matmul(matrix, composed)
    assert float(tf.abs(tf.math.log(tf.abs(tf.linalg.det(composed))) - flow["logdet"]).numpy()) <= 1.0e-10


def test_ledh_flow_records_nonzero_step_determinants() -> None:
    points = tf.constant([[0.0, 0.0], [1.0, 1.0]], tf.float64)
    args = (
        tf.constant([0.0, 0.0], tf.float64),
        tf.constant([[1.0, 0.1], [0.1, 1.0]], tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.eye(2, dtype=tf.float64),
        tf.constant([0.2, -0.1], tf.float64),
        tf.fill([4], tf.constant(0.25, tf.float64)),
    )
    _final, flow = ledh_flow(points, *args)
    assert bool(tf.reduce_all(tf.abs(flow["determinants"]) > 0.0).numpy())
