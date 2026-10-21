"""Independent analytic laws test quadrature normalization and CDF cell cuts."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import (
    expectation_graph, marginal_cdf, quantile_graph, summarize_grid, trapezoid_weights,
)


def test_fractional_cells_integrate_a_linear_density_exactly():
    axis = tf.constant([0., .25, .5, .75, 1.], tf.float64)
    density = 2. * axis
    cuts = tf.constant([-.1, 0., .13, .5, .89, 1., 1.1], tf.float64)
    expected = tf.clip_by_value(cuts, 0., 1.)**2
    tf.debugging.assert_near(marginal_cdf(axis,density,cuts),expected,atol=1e-14,rtol=1e-14)
    quantiles = quantile_graph(axis,density)
    tf.debugging.assert_near(quantiles,tf.sqrt(tf.constant([.1,.5,.9],tf.float64)),atol=1e-13,rtol=1e-13)


def test_rectangle_integration_preserves_marginals_moments_and_prediction():
    a = tf.linspace(tf.constant(.5,tf.float64),tf.constant(1.2,tf.float64),257)
    b = tf.linspace(tf.constant(-1.,tf.float64),tf.constant(1.,tf.float64),257)
    density = tf.ones([257,257],tf.float64) / 1.4
    tf.debugging.assert_near(tf.reduce_sum(density*trapezoid_weights(a)[:,None]*trapezoid_weights(b)[None,:]),
                             tf.constant(1.,tf.float64),atol=1e-14,rtol=1e-14)
    means,scales = summarize_grid(a,b,density)
    tf.debugging.assert_near(means,tf.constant([.85,0.],tf.float64),atol=1e-13,rtol=1e-13)
    tf.debugging.assert_near(scales,tf.constant([.7,2.],tf.float64)/tf.sqrt(tf.constant(12.,tf.float64)),
                             atol=1e-5,rtol=1e-5)
    cuts = tf.stack([.5+.7*tf.constant([.1,.5,.9],tf.float64),-1.+2.*tf.constant([.1,.5,.9],tf.float64)])
    prediction = tf.broadcast_to(tf.constant([.2,.8],tf.float64),[257,257,2])
    actual = expectation_graph(a,b,density,prediction,means,scales,cuts)
    expected = tf.constant([0.,0.,1.,1.,0.,.1,.5,.9,.1,.5,.9,.2,.8],tf.float64)
    tf.debugging.assert_near(actual,expected,atol=1e-11,rtol=1e-11)
