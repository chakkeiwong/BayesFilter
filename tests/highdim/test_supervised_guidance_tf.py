"""CPU-only independent reference checks for simulation-supervised guidance."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.score_study.supervised_guidance_tf import (
    make_sir_supervision_programs, moment_features, supervised_squared_loss,
    validate_seed_partitions,
)


def test_moment_features_and_independent_label_risk_identity():
    states = tf.constant([[2., 3.]], tf.float64)
    np.testing.assert_array_equal(moment_features(states), [[2., 3., 4., 6., 9.]])
    conditional_mean = tf.constant([[2., -1.]], tf.float64)
    predictions = conditional_mean + tf.constant([[.3, -.2], [-.3, .2]], tf.float64)
    labels = conditional_mean + tf.constant([[1., 2.], [-1., -2.]], tf.float64)
    scales = tf.constant([2., 3.], tf.float64)
    noisy, valid = supervised_squared_loss(predictions[:, None], labels[None], scales)
    exact, _ = supervised_squared_loss(predictions, conditional_mean, scales)
    irreducible, _ = supervised_squared_loss(labels, conditional_mean, scales)
    np.testing.assert_allclose(tf.reduce_mean(noisy, axis=[0, 1]),
        tf.reduce_mean(exact + irreducible, axis=0), atol=1e-15)
    assert bool(valid)
    assert not bool(supervised_squared_loss(predictions, labels, tf.zeros([2], tf.float64))[1])


def test_sir_complete_data_label_is_fixed_path_density_derivative():
    simulate, label = make_sir_supervision_programs(2, 2, jit_compile=False)
    theta = tf.constant([.02, -.03, .01], tf.float64)
    paths, observations = simulate(theta, tf.constant(261019501))
    result = label(theta, paths, observations)
    assert bool(result["valid"])
    for coordinate in range(3):
        increment = tf.one_hot(coordinate, 3, dtype=tf.float64) * 1e-5
        numerical = (label(theta + increment, paths, observations)["log_joint"]
                     - label(theta - increment, paths, observations)["log_joint"]) / 2e-5
        np.testing.assert_allclose(result["score"][:, coordinate], numerical, rtol=2e-6, atol=2e-6)
    residual = observations - paths[:, 1:, 1::2]
    observation_score = tf.reduce_sum(tf.square(residual), axis=[1, 2]) / (100 * tf.exp(2 * theta[2])) - 18
    np.testing.assert_allclose(result["score"][:, 2], observation_score, atol=1e-12)


def test_simulation_partition_leakage_rejected():
    validate_seed_partitions({"calibration": [1, 2], "validation": [3], "test": [4]})
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_partitions({"calibration": [1], "test": [1]})
    with pytest.raises(ValueError, match="disjoint"):
        validate_seed_partitions({"test": [1, 1]})


def test_gaussian_first_and_second_moment_risk_identity():
    nodes, weights = np.polynomial.hermite.hermgauss(5)
    states = 2.0 + np.sqrt(6.0) * nodes
    labels = tf.constant(np.column_stack([states, states ** 2]), tf.float64)
    probabilities = tf.constant(weights / np.sqrt(np.pi), tf.float64)
    posterior_moments = tf.constant([2., 7.], tf.float64)
    prediction = posterior_moments + tf.constant([.3, -.4], tf.float64)
    observed, valid = supervised_squared_loss(prediction, labels, tf.ones([2], tf.float64))
    expected = tf.square(prediction - posterior_moments) + tf.constant([3., 66.], tf.float64)
    np.testing.assert_allclose(tf.reduce_sum(probabilities[:, None] * observed, axis=0), expected, atol=1e-12)
    assert bool(valid)
