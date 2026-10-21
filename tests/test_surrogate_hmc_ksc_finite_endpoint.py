"""Independent coordinate/status checks; real canonical wiring is a GPU probe."""
from types import SimpleNamespace
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ksc_finite_endpoint import make_finite_endpoint
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status


class AnalyticFixture:
    lower = tf.constant([.1, .1], tf.float32)
    upper = tf.constant([.9, .9], tf.float32)
    identity = "independent-test-density"

    def physical(self, theta):
        value = -tf.reduce_sum(theta**2, axis=-1)
        status = endpoint_status(value)
        # A value remains usable when a separately labelled force is invalid.
        status["force_valid"] = tf.zeros_like(status["value_valid"])
        return value, tf.zeros_like(theta), status


def consumer():
    return SimpleNamespace(fixture=AnalyticFixture(), manifest={"batch_size": 4})


def test_physical_chart_and_sign_use_named_density():
    c = consumer()
    endpoint, physical, chart = make_finite_endpoint(c)
    q = tf.constant([[0., 0.], [-1., .5], [.4, -.8], [1.2, 1.4]], tf.float64)
    theta = .1 + .8 * .5 * tf.math.erfc(-q / math.sqrt(2.))
    value, status = physical(theta)
    expected, _, _ = c.fixture.physical(tf.cast(theta, tf.float32))
    tf.debugging.assert_equal(value, tf.cast(expected, tf.float64))
    density, chart_status = chart(q)
    jacobian = tf.reduce_sum(math.log(.8) - q**2 / 2 - math.log(2 * math.pi) / 2, axis=-1)
    tf.debugging.assert_near(density, value + jacobian, atol=1e-14, rtol=1e-14)
    tf.debugging.assert_equal(endpoint.function(q).potential, -density)
    assert bool(tf.reduce_all(status["value_valid"] & chart_status["value_valid"]))
    assert not bool(tf.reduce_any(status["force_valid"]))


def test_extreme_finite_probit_is_rounding_not_support_exclusion():
    _, _, chart = make_finite_endpoint(consumer())
    q = tf.constant([[-12., -12.], [12., 12.], [-9., 9.], [0., 0.]], tf.float64)
    value, status = chart(q)
    assert bool(tf.reduce_all(tf.math.is_finite(value) & status["value_valid"]))
    assert status["coordinate_boundary_rounding"].numpy().tolist() == [True, True, True, False]
    assert not bool(tf.reduce_any(status["outside_declared_support"]))


def test_physical_support_and_nonfinite_chart_fail_closed():
    _, physical, chart = make_finite_endpoint(consumer())
    _, status = physical(tf.constant([[.1, .5], [.5, .9], [.09, .5], [.5, .5]], tf.float64))
    assert status["value_valid"].numpy().tolist() == [False, False, False, True]
    assert status["outside_declared_support"].numpy().tolist() == [True, True, True, False]
    _, status = chart(tf.constant([[float("inf"), 0.], [float("nan"), 0.], [-float("inf"), 0.], [0., 0.]], tf.float64))
    assert status["value_valid"].numpy().tolist() == [False, False, False, True]
    assert not bool(tf.reduce_any(status["outside_declared_support"]))


def test_wrong_prior_is_rejected():
    c = consumer()
    c.fixture.lower = tf.constant([.2, .1], tf.float32)
    with pytest.raises(tf.errors.InvalidArgumentError):
        make_finite_endpoint(c)
