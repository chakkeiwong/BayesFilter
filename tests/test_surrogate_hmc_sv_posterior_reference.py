"""Independent checks of actual-SV prediction and posterior quadrature."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_sv_reference import normal_quadrature, raw_observation_log_density
from bayesfilter.testing.surrogate_hmc_sv_posterior_reference import (
    make_statistics, normalize_likelihood, assess_refinement, PARAMETER_INTERVALS)


def test_predictive_absolute_returns_against_independent_two_step_integral():
    theta = tf.constant([[.2, .8], [.6, .4]], tf.float64)
    observations = tf.constant([.3], tf.float64)
    kernel = make_statistics(intervals=512, batch_size=2, horizon=1)
    likelihood, actual, loss = kernel(theta, observations)
    nodes, weights = normal_quadrature(128)
    x1 = nodes[None, :]/tf.sqrt(1-theta[:, 0, None]**2)
    mass = tf.exp(raw_observation_log_density(x1, theta[:, 1, None], observations[0]))*weights[None, :]
    normalization = tf.reduce_sum(mass, axis=1)
    x2 = theta[:, 0, None, None]*x1[:, :, None]+nodes[None, None, :]
    denominator = tf.sqrt(tf.constant(2., tf.float64))*theta[:, 1, None, None, None]*tf.exp(.5*x2[..., None])
    probability = tf.math.erf(tf.constant([.25, 1.], tf.float64)/denominator)
    expected = tf.reduce_sum(mass[:, :, None, None]*weights[None, None, :, None]*probability, axis=[1, 2])/normalization[:, None]
    tf.debugging.assert_near(actual, expected, atol=1e-8, rtol=1e-8)
    tf.debugging.assert_near(likelihood, tf.math.log(normalization), atol=1e-8, rtol=1e-8)
    tf.debugging.assert_less(loss, tf.constant(1e-10, tf.float64))


def test_uniform_prior_integration_and_fractional_cell_cdfs():
    grids = []
    for intervals in (*PARAMETER_INTERVALS, 128, 128):
        knots = tf.linspace(tf.constant(.1, tf.float64), tf.constant(.9, tf.float64), intervals+1)
        density, log_evidence = normalize_likelihood(knots, knots, tf.zeros([intervals+1, intervals+1], tf.float64))
        tf.debugging.assert_near(log_evidence, tf.constant(0., tf.float64), atol=1e-12)
        predictions = tf.broadcast_to(tf.constant([.2, .8], tf.float64), [intervals+1, intervals+1, 2])
        grids.append((knots, knots, density, predictions))
    result = assess_refinement(grids)
    tf.debugging.assert_near(result['means'], tf.constant([.5, .5], tf.float64), atol=1e-12)
    tf.debugging.assert_near(result['cuts'], tf.constant([[.18, .5, .82]]*2, tf.float64), atol=1e-12)
    tf.debugging.assert_near(result['expectations'][-1, 5:11], tf.constant([.1, .5, .9]*2, tf.float64), atol=1e-12)
    tf.debugging.assert_near(result['expectations'][-1, -2:], tf.constant([.2, .8], tf.float64), atol=1e-12)
    assert bool(result['qualified'])
    damaged = list(grids)
    a, b, density, predictions = damaged[-1]
    damaged[-1] = (a, b, density, predictions+.05)
    assert not bool(assess_refinement(damaged)['qualified'])
