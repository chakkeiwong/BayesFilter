"""Independent CPU reference checks for the SIR diagnostic message adapter."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import (
    SIRGuideFitter,
    next_policy_log_integral,
    positive_information_projection,
)


def test_positive_projection_is_exact_on_healthy_input():
    precision = tf.constant([[[2., .2], [.2, 1.]]], tf.float64)
    linear = tf.constant([[.3, -.4]], tf.float64)
    center = tf.constant([[3., -2.]], tf.float64)
    projected, vector, changed, _ = positive_information_projection(precision, linear, center)
    np.testing.assert_array_equal(projected, precision)
    np.testing.assert_array_equal(vector, linear)
    assert not bool(changed[0])


def test_positive_projection_preserves_center_score():
    precision = tf.constant([[[-2., .4], [.4, 1.]]], tf.float64)
    linear = tf.constant([[.3, -.4]], tf.float64)
    center = tf.constant([[3., -2.]], tf.float64)
    projected, vector, changed, _ = positive_information_projection(precision, linear, center)
    assert bool(changed[0])
    assert float(tf.reduce_min(tf.linalg.eigvalsh(projected))) > 0
    np.testing.assert_allclose(vector - tf.linalg.matvec(projected, center),
                               linear - tf.linalg.matvec(precision, center), atol=2e-14)


def test_integral_uses_next_policy_and_no_current_observation():
    policy = {
        "A": tf.constant([[[[99., 0.], [0., 99.]]], [[[88., 0.], [0., 88.]]],
                          [[[.2, 0.], [0., .4]]], [[[.3, 0.], [0., .1]]]], tf.float64),
        "b": tf.constant([[[99., 99.]], [[88., 88.]], [[.5, -.2]], [[-.1, .4]]], tf.float64),
        "c": tf.constant([[99.], [88.], [1.2], [-.3]], tf.float64),
    }
    points = tf.constant([[[.4, -.8], [1., 2.]], [[-.4, .2], [.1, -.3]]], tf.float64)
    result = next_policy_log_integral(lambda states: .7 * states, points, policy,
                                      tf.eye(2, dtype=tf.float64), tf.constant(0., tf.float64))
    means = .7 * points.numpy()
    diagonal = np.diagonal(policy["A"].numpy()[2:, 0], axis1=-2, axis2=-1)
    posterior_variance = 1 / (1 + 2 * diagonal)
    shift = means - policy["b"].numpy()[2:, 0, None]
    expected = .5 * np.sum(np.log(posterior_variance), axis=-1)[:, None]
    expected = expected - policy["c"].numpy()[2:] + .5 * np.sum(
        shift ** 2 * posterior_variance[:, None] - means ** 2, axis=-1)
    np.testing.assert_allclose(result[2], expected, atol=2e-15)
    assert bool(tf.reduce_all(result[4]))


def test_sir_initial_state_has_no_observation_potential():
    fitter = SIRGuideFitter(2, particles=256, components=8, jit_compile=False)
    states = tf.zeros([3, 1, 4, 18], tf.float64)
    observations = tf.concat([tf.zeros([1, 1, 1, 10], tf.float64),
                              tf.ones([2, 1, 1, 10], tf.float64)], axis=0)
    density = fitter.observation(observations, states)
    np.testing.assert_array_equal(density[0], np.zeros([1, 4]))
    np.testing.assert_allclose(density[1:], -4.5 * np.log(200 * np.pi) - .045)
    assert fitter.engine.h == 3
    assert fitter.engine.solver == "centered_qr"
