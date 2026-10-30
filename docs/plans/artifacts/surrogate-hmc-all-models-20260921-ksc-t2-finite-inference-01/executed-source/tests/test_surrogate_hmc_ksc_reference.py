"""Independent scientific checks for the exact T2 KSC model reference."""
import copy
import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.sv_mixture_cut4 import KSCMixtureTransformedSVSSM, ksc_1998_log_chi_square_mixture
from bayesfilter.testing.surrogate_hmc_ksc_reference import (
    check_dataset, component_statistics, make_ksc_dataset, make_ksc_reference, physical_parameters,
)


def sequential_mixture_loglik(theta, observations):
    """Independent exact Gaussian update recursion, without moment collapse."""
    mixture = ksc_1998_log_chi_square_mixture()
    gamma, beta = theta
    stationary = 1 / (1 - gamma**2)
    s1 = stationary + mixture.variances
    r1 = observations[0] - 2 * tf.math.log(beta) - mixture.means
    logw1 = tf.math.log(mixture.weights) - .5 * (tf.math.log(2 * math.pi * s1) + r1**2 / s1)
    m1 = stationary / s1 * r1
    v1 = stationary * mixture.variances / s1
    pred_m = gamma * m1
    pred_v = gamma**2 * v1 + 1
    r2 = observations[1] - pred_m[:, None] - 2 * tf.math.log(beta) - mixture.means[None]
    s2 = pred_v[:, None] + mixture.variances[None]
    logw2 = tf.math.log(mixture.weights)[None] - .5 * (tf.math.log(2 * math.pi * s2) + r2**2 / s2)
    return tf.reduce_logsumexp(logw1[:, None] + logw2)


@pytest.fixture(scope="module")
def dataset():
    return make_ksc_dataset(partition="diagnostic")


def test_exact_mixture_equals_sequential_enumeration_and_stationary_score(dataset):
    reference = make_ksc_reference(dataset, batch_size=5)
    theta = tf.constant([[.1, .1], [.9, .9], [.89, .11], [.11, .89], [.5, .5]], tf.float64)
    values, scores = reference.physical_value_score(theta)
    observations = check_dataset(dataset)
    expected = tf.stack([sequential_mixture_loglik(point, observations) for point in tf.unstack(theta)])
    tf.debugging.assert_near(values, expected, atol=2e-12, rtol=2e-12)
    for dimension in range(2):
        delta = tf.one_hot(dimension, 2, dtype=tf.float64)[None] * 1e-5
        differences = (reference.physical_value_score(theta + delta)[0]
                       - reference.physical_value_score(theta - delta)[0]) / 2e-5
        tf.debugging.assert_near(scores[:, dimension], differences, atol=2e-6, rtol=2e-6)
    assert reference.physical_value_score.experimental_get_tracing_count() == 1


def test_prior_chart_once_and_extreme_finite_positions(dataset):
    reference = make_ksc_reference(dataset, batch_size=4)
    q = tf.constant([[-2., 1.], [0., 0.], [3., -3.], [7., -7.]], tf.float64)
    values, scores = reference.unconstrained_value_score(q)
    likelihood, physical_scores = reference.physical_value_score(physical_parameters(q))
    expected = likelihood - .5 * tf.reduce_sum(q**2, axis=1) - math.log(2 * math.pi)
    jacobian = .8 * tf.exp(-.5 * q**2) / math.sqrt(2 * math.pi)
    tf.debugging.assert_near(values, expected, atol=1e-12, rtol=1e-12)
    tf.debugging.assert_near(scores, physical_scores * jacobian - q, atol=1e-11, rtol=1e-11)


def test_reference_uses_repository_observation_law(dataset):
    model = KSCMixtureTransformedSVSSM()
    theta = model.unconstrained_from_physical(.5, .5)
    states = tf.constant([[-2.], [0.], [2.]], tf.float64)
    observed = check_dataset(dataset)[0:1]
    density = model.observation_log_density(theta, states, observed, t=0)
    mixture = ksc_1998_log_chi_square_mixture()
    residual = observed[0] - states - 2 * math.log(.5) - mixture.means[None]
    direct = tf.reduce_logsumexp(tf.math.log(mixture.weights)[None] - .5 * (
        tf.math.log(2 * math.pi * mixture.variances)[None] + residual**2 / mixture.variances[None]), axis=1)
    tf.debugging.assert_near(density, direct, atol=2e-12, rtol=2e-12)


def test_predictive_probabilities_and_conditional_variance(dataset):
    reference = make_ksc_reference(dataset, batch_size=3)
    theta = tf.constant([[.11, .89], [.5, .5], [.89, .11]], tf.float64)
    likelihood, probabilities = reference.statistics(theta)
    tf.debugging.assert_near(likelihood, reference.physical_value_score(theta)[0], atol=1e-12, rtol=1e-12)
    tf.debugging.assert_positive(component_statistics(theta, check_dataset(dataset))[3])
    assert bool(tf.reduce_all((probabilities >= 0) & (probabilities <= 1)))
    assert bool(tf.reduce_all(probabilities[:, 0] <= probabilities[:, 1]))


def test_data_roles_are_disjoint_and_wrong_target_is_rejected(dataset):
    calibration = make_ksc_dataset(partition="calibration")
    validation = make_ksc_dataset(partition="validation")
    assert calibration["seeds"] != validation["seeds"]
    assert calibration["observations"] != validation["observations"]
    for field, value in (("law", "actual SV"), ("horizon", 20), ("raw_transform_applied", True)):
        bad = copy.deepcopy(dataset)
        bad[field] = value
        with pytest.raises(ValueError):
            check_dataset(bad)
