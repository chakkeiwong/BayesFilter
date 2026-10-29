"""M10 reference law/wiring and independent identities; no HMC evidence."""
import math

import tensorflow as tf

from bayesfilter.score_study.gaussian_tf import parameterized_model
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import (
    BASE_THETA, PRIOR_SD, source_theta, log_prior, grid_reference,
    linear_kalman_log_likelihood, direct_innovation_reference, generate_dataset)


def test_noise_coordinates_preserve_source_initialization_and_proper_prior():
    theta = source_theta(tf.constant([.2, -.3], tf.float64))
    expected = list(BASE_THETA); expected[1] += .2; expected[2] -= .3
    tf.debugging.assert_near(theta, tf.constant(expected, tf.float64), atol=1e-15)
    a, _, h, _, m, _, p, _, q, _, r, _ = parameterized_model(theta, 1, 1)
    # These factors would be lost by a generic scalar-model approximation.
    assert abs(float(h[0, 0])-1.15*.9) < 1e-8
    assert abs(float(m[0])-.7*.25) < 1e-8
    assert abs(float(p[0, 0])-1.15*math.exp(-.6)) < 1e-8
    value, score = log_prior(tf.constant([.2, -.3], tf.float64))
    assert abs(float(value)-(-(.2**2+.3**2)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2))) < 1e-14
    tf.debugging.assert_near(score, -tf.constant([.2, -.3], tf.float64)/PRIOR_SD**2, atol=1e-14)


def test_grid_wrapper_matches_independent_linear_recursion_and_total_noise_derivatives():
    observations = tf.constant([[.2], [-.4]], tf.float64)
    scales = tf.constant([.15, -.2], tf.float64)
    linear = grid_reference(257, 9., 0., 0.)
    tf.debugging.assert_near(linear(scales, observations)['log_likelihood'],
        linear_kalman_log_likelihood(scales, observations), atol=1e-7, rtol=0.)
    nonlinear = grid_reference(257, 9.)
    result = nonlinear(scales, observations)
    for coordinate in (0, 1):
        delta = tf.one_hot(coordinate, 2, dtype=tf.float64)*1e-5
        derivative = (nonlinear(scales+delta, observations)['log_likelihood']
                      -nonlinear(scales-delta, observations)['log_likelihood'])/(2e-5)
        tf.debugging.assert_near(derivative, result['score'][coordinate], atol=1e-7, rtol=0.)


def test_direct_innovation_integration_and_frozen_data_replay():
    first = generate_dataset([20260920, 98765])
    tf.debugging.assert_equal(first, generate_dataset([20260920, 98765]))
    assert first.dtype == tf.float32 and first.shape == (2, 1)
    observations = tf.constant([[.2], [-.4]], tf.float64)
    scales = tf.constant([0., 0.], tf.float64)
    independent = direct_innovation_reference(64)(scales, observations)
    expected = grid_reference(257, 9.)(scales, observations)['log_likelihood']
    tf.debugging.assert_near(independent, expected, atol=1e-6, rtol=0.)
