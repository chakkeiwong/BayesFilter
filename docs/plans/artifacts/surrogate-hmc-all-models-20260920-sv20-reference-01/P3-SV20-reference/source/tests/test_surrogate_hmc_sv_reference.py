"""Independent actual-SV law and quadrature checks; intentional CPU references."""
import math

import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_sv_reference import (
    chart_and_log_prior, direct_short_likelihood, generate_dataset,
    make_sv_reference, raw_observation_log_density, transformed_observation_log_density)


def test_raw_transformed_density_and_existing_model_agree():
    from bayesfilter.highdim.models import StochasticVolatilitySSM
    x = tf.constant([[-3.], [.2], [2.]], tf.float64)
    beta = tf.constant(.4, tf.float64)
    y = tf.constant(-.7, tf.float64)
    raw = raw_observation_log_density(x[:, 0], beta, y)
    transformed = transformed_observation_log_density(x[:, 0], beta, tf.math.log(y*y))
    tf.debugging.assert_near(raw, transformed-tf.math.log(tf.abs(y)), atol=1e-13)
    model = StochasticVolatilitySSM()
    theta = model.unconstrained_from_physical(tf.constant(.6, tf.float64), beta)
    actual = model.observation_log_density(theta, x, tf.reshape(y, [1]), 1)
    tf.debugging.assert_near(raw, actual, atol=1e-13)


def test_preserved_chart_prior_and_chain_rule():
    theta = tf.constant([[-1.2, .3], [.7, -1.]], tf.float64)
    physical, derivative, prior = chart_and_log_prior(theta)
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        source_chart_physical_parameters, source_uniform_prior_value_score,
        source_two_probit_jacobian_value_score)
    tf.debugging.assert_near(physical, tf.stack(source_chart_physical_parameters(theta), -1), atol=1e-14)
    pv, ps = source_uniform_prior_value_score(theta)
    jv, js = source_two_probit_jacobian_value_score(theta)
    tf.debugging.assert_near(prior, pv+jv, atol=1e-14)
    tf.debugging.assert_near(ps+js, -theta, atol=1e-14)
    reference = make_sv_reference(intervals=256, half_width=12., horizon=2, batch_size=2)
    obs = tf.constant([.3, -.7], tf.float64)
    result = reference.physical_value_score(physical, obs)
    value, score = reference.unconstrained_value_score(theta, obs)
    tf.debugging.assert_near(value, result['log_likelihood']+prior, atol=1e-12)
    tf.debugging.assert_near(score, result['score']*derivative-theta, atol=1e-12)


def test_short_reference_against_independent_gaussian_integration():
    physical = tf.constant([[.2, .8], [.6, .4]], tf.float64)
    for horizon in (1, 2):
        obs = tf.constant([.3, -.7][:horizon], tf.float64)
        reference = make_sv_reference(intervals=512, half_width=16., horizon=horizon, batch_size=2)
        result = reference.physical_value_score(physical, obs)
        independent = direct_short_likelihood(physical, obs, order=256)
        tf.debugging.assert_near(result['log_likelihood'], independent, atol=1e-7, rtol=0.)
        tf.debugging.assert_near(tf.reduce_sum(result['final_mass'], 1), tf.ones([2], tf.float64), atol=1e-13)
        tf.debugging.assert_near(tf.reduce_sum(result['final_mass_derivative'], 1), tf.zeros([2, 2], tf.float64), atol=1e-12)


def test_stationary_and_recursive_scores_are_total_finite_grid_derivatives():
    reference = make_sv_reference(intervals=256, half_width=12., horizon=3, batch_size=2)
    physical = tf.constant([[.25, .7], [.85, .2]], tf.float64)
    observations = tf.constant([.2, -.8, .4], tf.float64)
    result = reference.physical_value_score(physical, observations)
    for coordinate in (0, 1):
        delta = tf.one_hot(coordinate, 2, dtype=tf.float64)*1e-6
        plus = reference.physical_value_score(physical+delta, observations)['log_likelihood']
        minus = reference.physical_value_score(physical-delta, observations)['log_likelihood']
        tf.debugging.assert_near((plus-minus)/2e-6, result['score'][:, coordinate], atol=1e-6, rtol=1e-6)
    # A permutation must permute every row; changing another row cannot alter row zero.
    reversed_result = reference.physical_value_score(tf.reverse(physical, [0]), observations)
    changed = reference.physical_value_score(tf.concat([physical[:1], [[.4, .5]]], 0), observations)
    for name in result:
        tf.debugging.assert_equal(reversed_result[name], tf.reverse(result[name], [0]))
        tf.debugging.assert_equal(changed[name][0], result[name][0])


def test_truncated_probability_is_reported_not_hidden_by_renormalization():
    reference = make_sv_reference(intervals=64, half_width=.5, horizon=1, batch_size=1)
    result = reference.physical_value_score(tf.constant([[.9, .4]], tf.float64), tf.constant([.3], tf.float64))
    assert float(result['maximum_predicted_domain_loss'][0]) > .8
    full = make_sv_reference(intervals=256, half_width=16., horizon=1, batch_size=1)
    baseline = full.physical_value_score(tf.constant([[.9, .4]], tf.float64), tf.constant([.3], tf.float64))
    assert float(result['log_likelihood'][0]) < float(baseline['log_likelihood'][0])


def test_datasets_are_replayable_and_partitions_independent():
    first = generate_dataset(tf.constant([20260920, 11100]))
    replay = generate_dataset(tf.constant([20260920, 11100]))
    second = generate_dataset(tf.constant([20260920, 11200]))
    for left, right in zip(first, replay):
        tf.debugging.assert_equal(left, right)
    assert bool(tf.reduce_any(first[2] != second[2]))
    assert first[2].dtype == tf.float32 and first[2].shape == (20,)
    assert bool(tf.reduce_all(tf.math.is_finite(first[2])))
