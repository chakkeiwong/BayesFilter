"""Exact-function FD order, stochastic counterexamples and direction geometry."""
import pytest
import tensorflow as tf

from bayesfilter.score_study.finite_difference_tf import (
    DESIGNS, checked_nodes, make_direction_solver, make_stencil_kernel,
    replicated_stencil_diagnostics, stencil,
)


@pytest.mark.parametrize("kind,order", [("central3", 2), ("central5", 4), ("cubic11", 4)])
def test_deterministic_order_on_exponential(kind, order):
    nodes, _ = stencil(kind, tf.float64)
    kernel = make_stencil_kernel(kind, jit_compile=True)
    errors = []
    for step in (0.1, 0.05, 0.025):
        h = tf.constant(step, tf.float64)
        estimate, valid = kernel(tf.exp(nodes * h)[None, :], h)
        assert bool(valid) and estimate.shape == (1,)
        errors.append(tf.abs(estimate[0] - 1))
    slopes = tf.math.log(tf.stack(errors[:-1])/tf.stack(errors[1:])) / tf.math.log(tf.constant(2., tf.float64))
    tf.debugging.assert_near(slopes, tf.fill([2], tf.cast(order, tf.float64)), atol=0.08, rtol=0.)


def test_c4_does_not_imply_fourth_order():
    nodes, coefficients = stencil("central5", tf.float64)
    errors = []
    for h in (.1, .05):
        x = nodes * h
        errors.append(tf.abs(tf.reduce_sum(coefficients * tf.sign(x) * tf.abs(x)**4.5)/h))
    slope = tf.math.log(errors[0]/errors[1]) / tf.math.log(tf.constant(2., tf.float64))
    tf.debugging.assert_near(slope, tf.constant(3.5, tf.float64), atol=1e-10)


def test_covariance_cross_term_and_shared_noise_cancellation():
    nodes, c = stencil("central5", tf.float64)
    h = tf.constant(.1, tf.float64)
    exact = tf.exp(nodes * h)
    noise = tf.random.stateless_normal([32, 1], [7, 8], dtype=tf.float64)
    result = replicated_stencil_diagnostics(exact[None, :] + noise, c, h, exact, tf.constant(1., tf.float64))
    assert abs(float(result["sample_variance"])) < 1e-12
    tf.debugging.assert_near(result["mse"], result["empirical_mse_identity"], atol=1e-12)
    # Parameter-dependent coupled noise has O(h^2) node-increment variance,
    # hence bounded derivative variance rather than an obligatory h^-2 slope.
    variances = []
    for h in (tf.constant(.1, tf.float64), tf.constant(.01, tf.float64)):
        values = tf.exp(nodes * h)[None, :] + noise * nodes[None, :] * h
        result = replicated_stencil_diagnostics(values, c, h, tf.exp(nodes*h), tf.constant(1., tf.float64))
        variances.append(result["sample_variance"])
        tf.debugging.assert_near(result["mse"], result["empirical_mse_identity"], atol=1e-11)
    tf.debugging.assert_near(*variances, atol=1e-11)


def test_direction_reconstruction_and_rank_rejection():
    V = tf.constant([[1., 0., 1.], [0., 1., 1.]], tf.float64)
    truth = tf.constant([2., -3.], tf.float64)
    solver = make_direction_solver(2, 3, jit_compile=True)
    result = solver(V, tf.linalg.matvec(tf.transpose(V), truth))
    assert bool(result[-1]) and int(result[1]) == 2
    tf.debugging.assert_near(result[0], truth, atol=1e-12)
    result = solver(tf.constant([[1., 2., 3.], [2., 4., 6.]], tf.float64), tf.ones([3], tf.float64))
    assert not bool(result[-1]) and int(result[1]) == 1
    assert bool(tf.reduce_all(tf.math.is_nan(result[0])))


def test_positive_domain_and_dtype_guards():
    theta, direction = tf.constant([1.], tf.float32), tf.constant([1.], tf.float32)
    for h, match in ((0., "nonpositive"), (1e-20, "coincide"), (1., "domain")):
        with pytest.raises(ValueError, match=match):
            checked_nodes(theta, direction, tf.constant(h, tf.float32), "central5", tf.constant([0.], tf.float32), tf.constant([2.], tf.float32))
