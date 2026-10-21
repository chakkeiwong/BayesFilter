"""RWM reference mechanics against an analytically known Gaussian endpoint."""
import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import TargetPotentialEvaluation
from bayesfilter.testing.surrogate_hmc_m13_rwm_baseline import random_walk_graph, health


def test_frozen_rwm_samples_gaussian_and_does_not_adapt():
    dtype = tf.float64
    def endpoint(x):
        valid = tf.reduce_all(tf.math.is_finite(x), -1)
        return TargetPotentialEvaluation(.5*tf.reduce_sum(x*x, -1), dict(value_valid=valid))
    graph = random_walk_graph(endpoint, 2500, adapt=False)
    starts = tf.constant([[-1., -1.], [-.5, .5], [.5, -.5], [1., 1.]], dtype)
    chunk = graph(starts, tf.constant(0., dtype), tf.eye(2, dtype=dtype), tf.constant([391, 221]))
    assert health(chunk)
    assert float(chunk["log_scale"]) == 0.
    cloud = chunk["states"][500:]
    assert bool(tf.reduce_all(tf.abs(tf.reduce_mean(cloud, [0, 1])) < .15))
    assert bool(tf.reduce_all(tf.abs(tf.math.reduce_std(cloud, [0, 1])-1.) < .15))
    assert bool(tf.reduce_all(chunk["log_accept"] <= 0.))
    replay = graph(starts, tf.constant(0., dtype), tf.eye(2, dtype=dtype), tf.constant([391, 221]))
    tf.debugging.assert_equal(replay["states"], chunk["states"])


def test_invalid_endpoint_is_flagged_and_rejected():
    dtype = tf.float64
    def endpoint(x):
        valid = tf.reduce_all(tf.abs(x) <= .01, -1)
        return TargetPotentialEvaluation(tf.zeros([4], dtype), dict(value_valid=valid))
    chunk = random_walk_graph(endpoint, 10, adapt=False)(
        tf.zeros([4, 2], dtype), tf.constant(0., dtype),
        tf.eye(2, dtype=dtype), tf.constant([139, 221]))
    assert not health(chunk)
    tf.debugging.assert_equal(chunk["states"], tf.zeros([10, 4, 2], dtype))
