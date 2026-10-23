"""CPU-only independent checks of the optional convex calibration solver."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_beta_tf import make_pairwise_beta_solver, pilot_log_objective, pilot_log_terms, pilot_objective, pilot_ratios


def test_pairwise_known_boundary_optimum():
    logs = tf.math.log(tf.constant([[[1., 2., .5]]] * 4, tf.float64))
    target = tf.zeros([4, 1], tf.float64)
    fitted = make_pairwise_beta_solver(4, 1, jit_compile=False)(logs, target, tf.constant(.1, tf.float64), tf.constant(1e-10, tf.float64))
    assert bool(fitted['converged'])
    np.testing.assert_allclose(fitted['beta'], [.1, .9, 0.], atol=1e-10)


def test_log_objective_agrees_with_direct_formula():
    logs = tf.random.stateless_normal([3, 11, 3], [17, 22], dtype=tf.float64)
    target = tf.random.stateless_normal([3, 11], [17, 23], dtype=tf.float64)
    beta = tf.constant([.4, 0., .6], tf.float64)
    expected = pilot_objective(beta, *pilot_ratios(logs, target))
    actual = pilot_log_objective(beta, *pilot_log_terms(logs, target))
    for calculated, reference in zip(actual, expected[:2]):
        np.testing.assert_allclose(calculated, reference, rtol=1e-12, atol=1e-12)


def test_log_objective_retains_small_density_terms():
    logs = tf.constant([[[-500., 0., -500.], [0., 0., 0.]]], tf.float64)
    target = tf.constant([[-500., 0.]], tf.float64)
    value, gradient = pilot_log_objective(tf.constant([.4, 0., .6], tf.float64), *pilot_log_terms(logs, target))
    np.testing.assert_allclose(value, 2., rtol=1e-12)
    np.testing.assert_allclose(gradient, [-2., -8., -2.], rtol=1e-12)


def test_pairwise_interior_symmetry():
    logs = tf.math.log(tf.constant([[[3., 1., 1.], [1., 3., 1.], [1., 1., 3.]]], tf.float64))
    fitted = make_pairwise_beta_solver(1, 3, jit_compile=False)(logs, tf.zeros([1, 3], tf.float64), tf.constant(.1, tf.float64), tf.constant(1e-9, tf.float64))
    assert bool(fitted['converged'])
    np.testing.assert_allclose(fitted['beta'], [1 / 3] * 3, atol=1e-8)
