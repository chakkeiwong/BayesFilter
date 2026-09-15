"""Independent identities for the bounded SGQF initialization diagnostic."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from bayesfilter.highdim import observation_guided_tt_tf as obs
from bayesfilter.highdim import pair_block_tt_tf as pair
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_standard_normal_ghq_level_rule
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import _normalized_hermite_values
from docs.benchmarks import observation_tt_sgqf_projection_diagnostic as projection

D = tf.float64


def test_coefficients_against_independent_gauss_hermite_integral():
    mean = tf.constant([.2, -.15], D)
    covariance = tf.constant([[.8, .3], [.3, 1.4]], D)
    actual, _ = projection.coefficient_kernel(2, 3, False)(mean, covariance)
    rule = tf_standard_normal_ghq_level_rule(16)
    x, y = tf.meshgrid(rule.nodes, rule.nodes, indexing='ij')
    wx, wy = tf.meshgrid(rule.weights, rule.weights, indexing='ij')
    rows = tf.stack([tf.reshape(x, [-1]), tf.reshape(y, [-1])], axis=1)
    weights = tf.reshape(wx*wy, [-1])
    chart = obs.Chart.from_moments(mean, covariance)
    standard = obs.Chart(tf.zeros([2], D), tf.eye(2, dtype=D))
    amplitude = tf.exp(.5*(chart.log_prob(rows)-standard.log_prob(rows)))
    expected = tf.einsum('n,ni,nj->ij', weights*amplitude,
        _normalized_hermite_values(rows[:, 0], 3), _normalized_hermite_values(rows[:, 1], 3))
    tf.debugging.assert_near(actual, expected, atol=2e-11, rtol=2e-11)


def test_identity_gaussian_limit_and_graph_parity():
    mean, covariance = tf.zeros([4], D), tf.eye(4, dtype=D)
    coefficients, logM = projection.coefficient_kernel(4, 3, False)(mean, covariance)
    tf.debugging.assert_equal(tf.reshape(coefficients, [-1]), tf.one_hot(0, 256, dtype=D))
    tf.debugging.assert_equal(logM, tf.constant(0., D))
    mean = tf.constant([.1, -.2], D)
    covariance = tf.constant([[1.1, .2], [.2, .9]], D)
    graph = projection.coefficient_kernel(2, 3, False)(mean, covariance)
    compiled = projection.coefficient_kernel(2, 3, True)(mean, covariance)
    tf.debugging.assert_near(graph[0], compiled[0], rtol=1e-12, atol=1e-12)


def test_full_rank_pair_compression_reconstructs_coefficients():
    coefficients = tf.random.stateless_normal([2]*4, [18, 9], dtype=D)
    cores, residual, _ = projection.pair_svd(coefficients, 4)
    tf.debugging.assert_near(projection.reconstruct(cores), coefficients, atol=1e-12)
    tf.debugging.assert_less(residual, tf.constant(1e-23, D))
    tf.debugging.assert_near(pair.pair_total_mass(cores), tf.reduce_sum(coefficients**2), atol=1e-12)


def test_coordinate_joint_equals_current_times_backward_conditional():
    model = obs.SVModel(tf.constant([[.6]], D), tf.constant([[1.5625]], D))
    current = obs.Chart(tf.constant([-.4], D), tf.constant([[.8]], D))
    condition = obs.Chart(tf.constant([.2], D), tf.constant([[1.1]], D))
    mean, covariance = projection.paired_gaussian(model, current, condition)
    rows = tf.constant([[.1, -.6], [1.2, .9], [-1.3, .4]], D)
    qcoord = obs.Chart.from_moments(mean, covariance)
    x, z = current.forward(rows[:, :1]), condition.forward(rows[:, 1:])
    P, S = 1.21, .36*1.21+1.
    K, B = P*.6/S, P-(P*.6/S)**2*S
    backmean = condition.mean + K*(x-.6*condition.mean)
    back = obs.Chart(tf.zeros([1], D), tf.constant([[B**.5]], D))
    expected = current.log_prob(x)+back.log_prob(z-backmean)+current.logdet+condition.logdet
    tf.debugging.assert_near(qcoord.log_prob(rows), expected, atol=2e-12)


def test_explicit_initialization_reaches_compiled_fitter_and_default_is_unchanged():
    rows = tf.random.stateless_normal([24, 1, 2], [22, 1], dtype=D)
    features = pair._pair_features(rows, 1)
    target, weights = tf.ones([24], D), tf.ones([24], D)
    initial = pair.initial_pair_cores(1, 1, 1)
    kwargs = dict(degree=1, rank=1, sweeps=1, proximal_steps=1, jit_compile=False)
    default, info = pair.fit_pair_features(features, target, weights, **kwargs)
    supplied, supplied_info = pair.fit_pair_features(features, target, weights, initial=initial, **kwargs)
    tf.debugging.assert_equal(default[0], supplied[0])
    assert info['initialization'].startswith('stateless_7919')
    assert supplied_info['initialization'] == 'explicit_initial_cores'
    changed = (initial[0]*.7,)
    expected = pair.compiled_pair_fitter(1, 1, 1, 1, 1, False)(features, target, weights,
        tf.constant(0., D), changed)[0]
    actual, _ = pair.fit_pair_features(features, target, weights, initial=changed, **kwargs)
    tf.debugging.assert_equal(actual[0], expected[0])
    assert float(tf.reduce_max(tf.abs(actual[0]-default[0]))) > 1e-5
