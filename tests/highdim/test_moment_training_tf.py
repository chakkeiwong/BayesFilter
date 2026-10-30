"""CPU diagnostic checks for the simulated moment-training objective."""

import math
import json

import numpy as np
import tensorflow as tf

from bayesfilter.score_study.moment_training_tf import (
    calibrate_moments, make_guide_adapters, make_moment_endpoint, make_moment_objective,
    random_direction_descent,
)


def test_adaptation_identity_and_translated_density():
    quadratic, mixture = make_guide_adapters(3, 2, 2, True, jit_compile=False)
    precision = tf.eye(2, batch_shape=[3], dtype=tf.float64)
    linear = tf.ones([3, 2], tf.float64)
    eta = tf.zeros([2, 3], tf.float64)
    actual = quadratic(eta, precision, linear)
    np.testing.assert_array_equal(actual[0], precision)
    np.testing.assert_array_equal(actual[1], linear)
    eta = tf.constant([[math.log(2), .4, -.2], [0., 0., 0.]], tf.float64)
    actual = quadratic(eta, precision, linear)
    np.testing.assert_allclose(actual[0][0], 2 * np.eye(2))
    np.testing.assert_allclose(actual[1][0], [2.8, 1.6])
    np.testing.assert_array_equal(actual[0][-1], precision[-1])
    logits = tf.zeros([3, 2], tf.float64)
    centers = tf.ones([3, 2, 2], tf.float64)
    covariances = tf.eye(2, batch_shape=[3, 2], dtype=tf.float64)
    result = mixture(eta, logits, centers, covariances, tf.constant([True, True, False]))
    np.testing.assert_allclose(result[1][0], [[1.4, .8], [1.4, .8]])
    np.testing.assert_allclose(result[2][0], np.tile(.5 * np.eye(2), [2, 1, 1]))


def test_calibrated_moment_risk_decomposition():
    labels = tf.constant([[-2., 1.], [-1., 0.], [1., 2.], [2., 1.]], tf.float64)
    calibration = calibrate_moments(labels)
    objective = make_moment_objective(4, 2, *calibration, jit_compile=False)
    alternative = labels + tf.constant([.4, -.3], tf.float64)
    reference_losses = [objective(labels, label)["loss"] for label in labels]
    alternative_losses = [objective(alternative, label)["loss"] for label in labels]
    feature_difference = (objective(alternative, labels[0])["estimate"]
                          - objective(labels, labels[0])["estimate"]) / calibration[2]
    exact = .5 * (tf.reduce_mean(feature_difference[:2] ** 2)
                  + tf.reduce_mean(feature_difference[2:] ** 2))
    np.testing.assert_allclose(tf.reduce_mean(alternative_losses) - tf.reduce_mean(reference_losses), exact,
                               rtol=1e-12, atol=1e-12)


def test_real_optimizer_updates_coefficients_and_rejects_invalid_trials():
    target = tf.constant([[.2, -.4, .3]], tf.float64)
    initial = tf.zeros_like(target)
    history = []

    def objective(eta, update):
        if bool(tf.reduce_any(tf.abs(eta) > .7)):
            return math.inf
        return tf.reduce_sum(tf.square(eta - target))

    eta, accepted = random_direction_descent(objective, initial, seed=123,
        updates=64, perturbation=.02, step_size=.2, record=history.append, jit_compile=False)
    assert accepted > 2
    assert float(objective(eta, 0)) < .01 * float(objective(initial, 0))
    assert all(row["loss"] <= previous["loss"] for previous, row in zip(history, history[1:]))


def test_label_is_only_in_loss_and_cannot_change_adapter():
    quadratic, _ = make_guide_adapters(2, 2, 1, False, jit_compile=False)
    eta = tf.constant([[.1, .2, -.1]], tf.float64)
    precision = tf.eye(2, batch_shape=[2], dtype=tf.float64)
    linear = tf.zeros([2, 2], tf.float64)
    first = quadratic(eta, precision, linear)
    labels = tf.constant([[-1., -1.], [1., 1.]], tf.float64)
    objective = make_moment_objective(2, 2, *calibrate_moments(labels), jit_compile=False)
    assert float(objective(labels, tf.constant([2., 2.], tf.float64))["loss"]) != float(
        objective(labels, tf.constant([0., 0.], tf.float64))["loss"])
    second = quadratic(eta, precision, linear)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_endpoint_accepts_shared_scalar_value_and_preserves_health_flags():
    states = tf.constant([[-1., 0.], [1., 2.]], tf.float64)
    objective = make_moment_objective(2, 2, *calibrate_moments(states), jit_compile=False)

    @tf.function(input_signature=[tf.TensorSpec([2, 2], tf.float64), tf.TensorSpec([], tf.bool)])
    def kernel(cloud, valid):
        trace = [{"states_after_reset": cloud, "reset_valid": valid,
                  "optional_feature_enabled": tf.constant(False),
                  "margin": tf.constant(.1, tf.float64)}]
        return tf.constant(2., tf.float64), tf.constant([3.], tf.float64), trace

    endpoint = make_moment_endpoint(kernel, objective, 2, jit_compile=False)
    result = endpoint(states, tf.constant(True), states[0])
    assert bool(result["valid"])
    assert float(result["value"]) == 2.
    assert float(result["score"]) == 3.
    assert not bool(result["checks"]["optional_feature_enabled"])
    assert not bool(endpoint(states, tf.constant(False), states[0])["valid"])


def test_invalid_evaluation_retains_diagnostics_as_strict_json():
    from bayesfilter.score_study.moment_training_campaign import (
        evaluation_diagnostics, serializable,
    )

    output = dict(valid=tf.constant(False), value=tf.constant(-math.inf),
                  score=tf.constant(math.nan),
                  checks={"program_valid": tf.constant(False)},
                  diagnostics={"margin": tf.constant([-math.inf, 1.])})
    restored = json.loads(json.dumps(serializable(evaluation_diagnostics([output])),
                                    allow_nan=False))
    assert restored == [{"valid": False, "value": "-inf", "score": "nan",
                         "checks": {"program_valid": False},
                         "diagnostics": {"margin": ["-inf", 1.]}}]


def test_unavailable_intermediate_shape_rejects_finite_terminal_loss():
    states = tf.constant([[1., 2.], [2., 3.]], tf.float64)
    objective = make_moment_objective(2, 2, *calibrate_moments(states), jit_compile=False)

    @tf.function(input_signature=[tf.TensorSpec([], tf.bool)])
    def kernel(available):
        return tf.constant(2., tf.float64), tf.constant([3.], tf.float64), [
            {"states_after_reset": states, "program_valid": tf.constant(True),
             "higher_moment_materialized_shape_available": available},
            {"states_after_reset": states, "program_valid": tf.constant(True),
             "higher_moment_materialized_shape_available": tf.constant(True)}]

    endpoint = make_moment_endpoint(kernel, objective, 2, jit_compile=False)
    healthy = endpoint(tf.constant(True), states[0])
    collapsed = endpoint(tf.constant(False), states[0])
    assert bool(healthy["valid"])
    assert not bool(collapsed["valid"])
    tf.debugging.assert_equal(healthy["loss"], collapsed["loss"])
