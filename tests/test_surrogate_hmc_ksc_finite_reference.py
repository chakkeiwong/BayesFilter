"""Independent integrable densities test the D/R quadrature decision, not HMC."""
import math

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ksc_finite_reference import summarize_finite_grids


def reference():
    return dict(means=[.5, .5], scales=[.8/math.sqrt(12)]*2,
        cuts=[[.18, .5, .82]]*2, margins=[.1]*5+[.02]*6+[.03]*2,
        expectations=[[0., 0., 1., 1., 0., .1, .5, .9, .1, .5, .9, .2, .7]],
        allowance=[1e-12]*13, function_names=[str(i) for i in range(13)])


def test_uniform_density_reference_qualifies_against_analytic_integrals():
    knots = tf.linspace(tf.constant(.1, tf.float64), tf.constant(.9, tf.float64), 129)
    prediction = tf.broadcast_to(tf.constant([.2, .7], tf.float64), [129, 129, 2])
    report, grids = summarize_finite_grids(knots, tf.zeros([129, 129], tf.float64), prediction, reference())
    assert report["qualified"]
    assert report["D_vs_R"]["status"] == "EQUIVALENT"
    tf.debugging.assert_near(report["expectations"][-1], tf.constant(reference()["expectations"][-1], tf.float64), atol=2e-4)
    assert [int(grid[0].shape[0]) for grid in grids] == [17, 33, 65, 129]


def test_resolved_density_discrepancy_is_not_a_sampler_failure():
    knots = tf.linspace(tf.constant(.1, tf.float64), tf.constant(.9, tf.float64), 129)
    log_density = tf.broadcast_to(3*(knots[:, None]-.5), [129, 129])
    prediction = tf.broadcast_to(tf.constant([.2, .7], tf.float64), [129, 129, 2])
    report, _ = summarize_finite_grids(knots, log_density, prediction, reference())
    assert report["qualified"]
    assert report["D_vs_R"]["status"] == "DISCREPANCY"
    assert report["posterior_comparisons"]["C/D"] == "NOT_RUN"
    assert not report["surrogate_hmc_tested"]
