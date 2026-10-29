"""Independent callback mathematics and endpoint-to-canonical wiring checks."""
import math

import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_program_fixtures_tf as fixtures
from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
from bayesfilter.testing.surrogate_hmc_sv_consumer import actual_sv_model, make_sv_consumer, make_sv_fixture_target


def dataset():
    # Synthetic mechanics fixture only: same supported law, no campaign data.
    return dict(model_id='SV20', horizon=20, sigma=1., attempt=1, replicate=0,
        partition='diagnostic', observation_dtype='float32',
        raw_observations=[.25, -.5] * 10, prior='independent Uniform(.1,.9)',
        chart='gamma,beta=.1+.8*Phi(z)',
        law='stationary x0; transition then Normal(0,beta^2*exp(x)) observation')


def test_actual_raw_density_and_total_callback_derivatives():
    model = actual_sv_model(tf.float64)
    theta = tf.constant([[.2, .8], [.6, .4], [.8, .2]], tf.float64)
    direction = tf.constant([[.4, -.3], [-.2, .5], [.1, -.2]], tf.float64)
    x = tf.constant([[-2.], [.5], [2.]], tf.float64)
    dx = tf.constant([[.2], [-.1], [.3]], tf.float64)
    obs = tf.constant([2 * math.log(.35)], tf.float64)
    raw_expected = -.5 * (math.log(2 * math.pi) + 2 * tf.math.log(theta[:, 1])
        + x[:, 0] + (.35 / theta[:, 1])**2 * tf.exp(-x[:, 0]))
    tf.debugging.assert_near(model.observation_log_density_fn(theta, x, obs), raw_expected, atol=1e-12)
    eps = 1e-6
    pairs = [
        (lambda t, a: model.transition_mean_fn(t, a), model.transition_mean_tangent_fn(theta, x, dx, direction)),
        (lambda t, a: model.parameter_observation_fn(t, a), model.parameter_observation_tangent_fn(theta, x, dx, direction)),
        (lambda t, a: model.observation_log_density_fn(t, a, obs),
         model.observation_log_density_tangent_fn(theta, x, obs, dx, direction)),
    ]
    for value, tangent in pairs:
        fd = (value(theta + eps * direction, x + eps * dx) - value(theta - eps * direction, x - eps * dx)) / (2 * eps)
        tf.debugging.assert_near(tangent, fd, atol=1e-7, rtol=1e-7)
    mean, covariance = model.initial_distribution_fn(theta[1])
    dm, dcov = model.initial_distribution_tangent_fn(theta[1], direction[1])
    left = model.initial_distribution_fn(theta[1] - eps * direction[1])[1]
    right = model.initial_distribution_fn(theta[1] + eps * direction[1])[1]
    tf.debugging.assert_near(covariance, tf.constant([[1 / (1 - .6**2)]], tf.float64), atol=1e-12)
    tf.debugging.assert_near(dcov, (right-left)/(2*eps), atol=1e-9)
    tf.debugging.assert_equal(mean, dm)
    tf.debugging.assert_near(model.observation_covariance, tf.constant([[math.pi**2 / 2]], tf.float64), atol=1e-12)


def test_consumer_probit_prior_status_and_actual_canonical_call(monkeypatch):
    calls = []

    def canonical(model, theta, directions, base, covariance, noises, observations, **kwargs):
        calls.append((model, theta.shape, directions.shape, kwargs))
        b = tf.shape(theta)[0]
        valid = theta[:, 0] < .7
        zeros = tf.zeros([b], tf.int32)
        status = dict(value_valid=tf.ones([b], tf.bool), force_valid=valid,
            value_failure_code=zeros, force_failure_code=tf.where(valid, zeros, 1),
            first_failure_time=-tf.ones([b], tf.int32), first_failure_stage=zeros,
            first_failure_particle=-tf.ones([b], tf.int32),
            direction_value_valid=tf.ones([b, 2], tf.bool),
            direction_force_valid=tf.broadcast_to(valid[:, None], [b, 2]))
        return -tf.reduce_sum(theta**2, axis=-1), -2*theta, status

    monkeypatch.setattr(fixtures, 'canonical_batch_fused_value_score', canonical)
    consumer = make_sv_consumer(dataset(), controls=protection_controls(1.), batch_size=2, particle_count=4)
    z = tf.constant([[0., -.5], [1., .2]], tf.float64)
    value, score, status = consumer.evaluate_force(z)
    physical = .1 + .4 * tf.math.erfc(-z / math.sqrt(2.))
    expected = -tf.reduce_sum(physical**2 + .5*z**2 + .5*math.log(2*math.pi), axis=-1)
    tf.debugging.assert_near(tf.cast(value, tf.float64), expected, atol=1e-6)
    expected_score = -2*physical * .8 * tf.exp(-.5*z**2)/math.sqrt(2*math.pi) - z
    tf.debugging.assert_near(tf.cast(score[:1], tf.float64), expected_score[:1], atol=1e-6)
    tf.debugging.assert_equal(score[1], [0., 0.])
    tf.debugging.assert_equal(status['force_valid'], [True, False])
    tf.debugging.assert_equal(consumer.force.function(z).force, -tf.cast(score, tf.float64))
    assert calls and calls[0][1] == (2, 2)
    assert calls[0][2] == (2, 2, 2)
    assert calls[0][3]['reset_policy'] == 'contract_e'
    assert calls[0][0].initial_distribution_fn is not None
    assert calls[0][3]['pairwise_steps'] > 0 and calls[0][3]['correction_steps'] > 0
    extreme = consumer.evaluate_force(tf.constant([[12., 0.], [float('nan'), 0.]], tf.float64))[2]
    tf.debugging.assert_equal(extreme['force_valid'], [False, False])
    tf.debugging.assert_equal(extreme['outside_declared_support'], [False, False])
    assert consumer.force.identity != consumer.endpoint.identity


@pytest.mark.parametrize('change', [{'raw_observations': [0.] * 20}, {'chart': 'logistic'},
                                   {'law': 'KSC mixture'}, {'sigma': .2}])
def test_rejects_wrong_target_or_unsupported_flow_transform(change):
    with pytest.raises(ValueError):
        make_sv_fixture_target(dict(dataset(), **change), controls=protection_controls(1.), particle_count=4)
