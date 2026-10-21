"""Independent coordinate-law and full-physical-call wiring checks."""
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_lg1_finite_endpoint import make_finite_endpoint
from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status


def test_uniform_physical_prior_gets_one_jacobian_and_boundary_flags():
    calls = []
    def full_physical(phi):
        calls.append(phi)
        value = tf.fill([4], -tf.math.log(tf.constant(1.9, tf.float32)))
        status = endpoint_status(value)
        # An imperfect/invalid force does not invalidate a computable endpoint.
        status["force_valid"] = tf.zeros([4], tf.bool)
        return value, tf.zeros_like(phi), status
    fixture = SimpleNamespace(lower=tf.constant([-.95]), upper=tf.constant([.95]),
                              physical=full_physical, identity="instrumented_full_physical")
    consumer = SimpleNamespace(fixture=fixture, manifest={"batch_size":4})
    endpoint, physical, chart = make_finite_endpoint(consumer)
    z = tf.constant([[-100.],[-2.],[2.],[100.]], tf.float64)
    value, status = chart(z)
    expected = -tf.nn.softplus(-z[:,0])-tf.nn.softplus(z[:,0])
    constant_roundoff = -tf.cast(tf.math.log(tf.constant(1.9,tf.float32)),tf.float64)+tf.math.log(tf.constant(1.9,tf.float64))
    tf.debugging.assert_near(value, expected+constant_roundoff, atol=1e-12, rtol=1e-12)
    tf.debugging.assert_equal(status["coordinate_boundary_rounding"], [True,False,False,True])
    tf.debugging.assert_equal(status["value_valid"], [True]*4)
    tf.debugging.assert_equal(status["force_valid"], [False]*4)
    tf.debugging.assert_equal(status["outside_declared_support"], [False]*4)
    tf.debugging.assert_equal(endpoint.function(z).potential, -value)
    _, support = physical(tf.constant([[-.95],[-.4],[.4],[.95]], tf.float64))
    tf.debugging.assert_equal(support["outside_declared_support"], [True,False,False,True])
    tf.debugging.assert_equal(support["value_valid"], [False,True,True,False])
    assert len(calls) == 2  # One trace for each fixed-signature graph.


def test_underlying_value_failure_cannot_be_hidden_by_coordinate_protection():
    def full_physical(phi):
        value = tf.zeros([4],tf.float32)
        status = endpoint_status(value)
        status["value_valid"] = tf.constant([False,True,True,False])
        return value, tf.zeros_like(phi), status
    consumer = SimpleNamespace(manifest={"batch_size":4}, fixture=SimpleNamespace(
        lower=tf.constant([-.95]), upper=tf.constant([.95]), physical=full_physical, identity="failing"))
    _, _, chart = make_finite_endpoint(consumer)
    _, status = chart(tf.constant([[-100.],[-2.],[2.],[100.]],tf.float64))
    tf.debugging.assert_equal(status["value_valid"], [False,True,True,False])
    _, invalid = chart(tf.constant([[float("nan")],[float("inf")],[-float("inf")],[0.]],tf.float64))
    tf.debugging.assert_equal(invalid["value_valid"][:3], [False]*3)
    tf.debugging.assert_positive(invalid["value_failure_code"][:3])
    tf.debugging.assert_equal(invalid["force_valid"][:3], [False]*3)
