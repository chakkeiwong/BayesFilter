"""Independent normalization, sum and derivative checks for M13 wrapped law."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_m13_wrapped_reference import (
    angular_terms, observation_log_density,
)

F64 = tf.float64


@pytest.mark.parametrize("sigma", [.002, .08, .5, 1., 1.00001, 2., 8., 40.])
def test_wrapped_density_and_score_match_long_spatial_reference(sigma):
    delta = tf.constant([-math.pi, -2.8, -.1, 0., .4, 2.9, math.pi], F64)
    value, score, valid = angular_terms(delta, tf.constant(sigma, F64))
    aliases = delta[:, None]+2*tf.constant(math.pi, F64)*tf.cast(tf.range(-256, 257), F64)
    powers = -.5*tf.square(aliases/sigma)
    reference = tf.reduce_logsumexp(powers, -1)-math.log(sigma)-.5*math.log(2*math.pi)
    reference_score = -tf.reduce_sum(tf.nn.softmax(powers)*aliases, -1)/sigma**2
    tf.debugging.assert_equal(valid, True)
    tf.debugging.assert_near(value, reference, atol=2e-12, rtol=2e-14)
    tf.debugging.assert_near(score, reference_score, atol=2e-10, rtol=2e-12)


@pytest.mark.parametrize("sigma", [.02, .08, .5, 1., 1.01, 2., 8., 100.])
def test_wrapped_density_integrates_to_one(sigma):
    count = 4096
    delta = (tf.cast(tf.range(count), F64)+.5)*(2*math.pi/count)-math.pi
    value, _, valid = angular_terms(delta, tf.constant(sigma, F64))
    tf.debugging.assert_equal(valid, True)
    integral = tf.reduce_mean(tf.exp(value))*(2*math.pi)
    tf.debugging.assert_near(integral, tf.constant(1., F64), atol=2e-13, rtol=0.)


@pytest.mark.parametrize("sigma", [.08, .7, 1., 1.01, 2., 40.])
def test_angular_score_matches_total_derivative(sigma):
    delta = tf.constant([-3., -.5, .1, 2.7], F64)
    with tf.GradientTape() as tape:
        tape.watch(delta)
        value, score, _ = angular_terms(delta, tf.constant(sigma, F64))
        total = tf.reduce_sum(value)
    tf.debugging.assert_near(score, tape.gradient(total, delta), atol=2e-11, rtol=2e-12)


def test_old_observation_law_is_unchanged_on_its_domain():
    from bayesfilter.testing.surrogate_hmc_m13_reference import observation_log_density as old
    position = tf.constant([[1., 2.], [-3., .1], [2., -1.]], F64)
    observed = tf.constant([[2.1, .9], [3.2, -3.1], [2.4, 3.]], F64)
    for angular in (.02, .04, .08):
        sigma = tf.constant([.5, angular], F64)
        tf.debugging.assert_near(observation_log_density(position, observed, sigma),
                                old(position, observed, sigma), atol=1e-10, rtol=2e-14)


@pytest.mark.parametrize("sigma", [0., -1., float("nan"), float("inf")])
def test_invalid_scales_remain_invalid(sigma):
    _, _, valid = angular_terms(tf.constant([0., .3], F64), tf.constant(sigma, F64))
    tf.debugging.assert_equal(valid, False)
