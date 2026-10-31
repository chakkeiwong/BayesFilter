"""Independent callback derivatives and M10-to-canonical consumer wiring."""
import math

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_failure_status_tf import FailureCode, FailureStage
from bayesfilter.inference.surrogate_hmc_dctr_selection import protection_controls
from bayesfilter.testing import surrogate_hmc_nonlinear_consumer as module
from bayesfilter.testing.surrogate_hmc_nonlinear_reference import BASE_THETA, PRIOR_SD


def dataset():
    return dict(horizon=2, dtype='float32', role='calibration', regime='nominal',
                raw_observations=[[.25], [-.5]])


def test_model_law_and_all_analytical_callback_derivatives():
    model = module.nonlinear_noise_model(tf.float64)
    theta = tf.constant([-.3, .2], tf.float64)
    direction = tf.constant([.4, -.6], tf.float64)
    x = tf.constant([[-1.2], [.3], [2.]], tf.float64)
    dx = tf.constant([[.1], [-.2], [.5]], tf.float64)
    eps = 1e-6
    tf.debugging.assert_near(model.transition_mean_fn(theta, x),
        BASE_THETA[0]*x+tf.cast(.2, tf.float64)*tf.sin(x), atol=1e-12)
    tf.debugging.assert_near(model.observation_fn(x),
        (1+tf.cast(.15, tf.float64))*BASE_THETA[3]*x+tf.cast(.12, tf.float64)*x*x, atol=1e-12)
    pairs = [
        (lambda t, a: model.transition_mean_fn(t, a),
         model.transition_mean_tangent_fn(theta, x, dx, direction)),
        (lambda t, a: model.observation_fn(a), model.observation_tangent_fn(x, dx)),
        (lambda t, a: model.observation_jacobian_fn(a), model.observation_jacobian_tangent_fn(x, dx)),
        (lambda t, a: model.process_covariance_fn(t), model.process_covariance_tangent_fn(theta, direction)),
        (lambda t, a: model.observation_covariance_fn(t), model.observation_covariance_tangent_fn(theta, direction)),
    ]
    for value, tangent in pairs:
        fd = (value(theta+eps*direction, x+eps*dx)-value(theta-eps*direction, x-eps*dx))/(2*eps)
        tf.debugging.assert_near(tangent, fd, atol=1e-9, rtol=1e-8)
    tf.debugging.assert_near(model.process_covariance_fn(theta),
        tf.constant([[math.exp(2*(BASE_THETA[1]-.3))]], tf.float64), atol=1e-12)
    tf.debugging.assert_near(model.observation_covariance_fn(theta),
        tf.constant([[math.exp(2*(BASE_THETA[2]+.2))]], tf.float64), atol=1e-12)
    for value, tangent in zip(model.initial_distribution_fn(theta),
                              model.initial_distribution_tangent_fn(theta, direction)):
        tf.debugging.assert_equal(tangent, tf.zeros_like(value))


def test_prior_failure_codes_and_typed_shared_engine_wiring(monkeypatch):
    calls = []

    def canonical(model, theta, directions, base, covariance, noises, observations, **settings):
        calls.append((model, directions.shape, settings))
        b = tf.shape(theta)[0]
        valid = theta[:, 0] < .5
        zeros = tf.zeros([b], tf.int32)
        return (tf.zeros([b], tf.float32), tf.zeros([b, 2], tf.float32),
            dict(value_valid=tf.ones([b], tf.bool), force_valid=valid,
                 value_failure_code=zeros, force_failure_code=tf.where(valid, zeros, int(FailureCode.INVALID_TANGENT)),
                 first_failure_stage=tf.where(valid, -tf.ones_like(zeros), int(FailureStage.CALLBACK)),
                 first_failure_time=-tf.ones_like(zeros), first_failure_particle=-tf.ones_like(zeros),
                 direction_value_valid=tf.ones([b, 2], tf.bool),
                 direction_force_valid=tf.broadcast_to(valid[:, None], [b, 2])))

    monkeypatch.setattr(module, 'canonical_batch_fused_value_score', canonical)
    consumer = module.make_nonlinear_consumer(dataset(), controls=protection_controls(1.),
                                             batch_size=2, particle_count=4)
    z = tf.constant([[.1, -.2], [.6, .2]], tf.float64)
    value, score, status = consumer.evaluate_force(z)
    expected = -tf.reduce_sum(z*z, -1)/(2*PRIOR_SD**2)-math.log(2*math.pi*PRIOR_SD**2)
    tf.debugging.assert_near(tf.cast(value, tf.float64), expected, atol=2e-6)
    tf.debugging.assert_near(tf.cast(score[:1], tf.float64), -z[:1]/PRIOR_SD**2, atol=1e-6)
    tf.debugging.assert_equal(score[1], [0., 0.])
    field = consumer.force.function(z)
    tf.debugging.assert_equal(field.force, -tf.cast(score, tf.float64))
    tf.debugging.assert_equal(field.force_valid, [True, False])
    assert calls and calls[0][1] == (2, 2, 2)
    assert calls[0][0].process_covariance_tangent_fn is not None
    assert calls[0][2]['reset_policy'] == 'contract_e'
    assert calls[0][2]['pairwise_steps'] > 0 and calls[0][2]['correction_steps'] > 0
    assert consumer.force.identity != consumer.endpoint.identity
    # One nonfinite input and one finite coordinate whose proper prior overflows.
    invalid = consumer.evaluate_force(tf.constant([[float('nan'), 0.], [-1e30, 0.]], tf.float64))[2]
    tf.debugging.assert_equal(invalid['force_valid'], [False, False])
    tf.debugging.assert_equal(invalid['outside_declared_support'], [False, False])
    tf.debugging.assert_equal(invalid['value_failure_code'], [1, 1])
    tf.debugging.assert_equal(invalid['force_failure_code'], [1, 1])
    tf.debugging.assert_equal(invalid['first_failure_stage'], [int(FailureStage.PRIOR)]*2)


def test_local_endpoint_never_turns_prior_tails_into_support_rejection():
    consumer = module.make_nonlinear_consumer(dataset(), controls=protection_controls(1.),
                                             batch_size=2, particle_count=4)
    z = tf.constant([[0., 0.], [1., 0.]], tf.float64)
    value, score = consumer.evaluate_reference(z)
    endpoint = consumer.endpoint.function(z)
    tf.debugging.assert_equal(endpoint.potential[0], -value[0])
    tf.debugging.assert_equal(consumer.reference_force.function(z).force[0], -score[0])
    tf.debugging.assert_equal(endpoint.status['numerical_failure'], [False, True])
    tf.debugging.assert_equal(endpoint.status['outside_declared_support'], [False, False])
    tf.debugging.assert_equal(consumer.reference_force.function(z).force_valid, [True, False])
    assert consumer.manifest['retained_inference'] is False


def test_incomplete_canonical_controls_rejected():
    with pytest.raises(ValueError, match='complete canonical'):
        module.make_nonlinear_fixture(dataset(), controls=dict(protection_controls(1.), pairwise_steps=0),
                                      particle_count=4)


def test_partition_roles_are_preserved_and_claim_data_cannot_select_controls():
    for role in ('calibration', 'validation'):
        for regime in ('nominal', 'stress'):
            data = dict(dataset(), role=role, regime=regime)
            fixture = module.make_nonlinear_fixture(data, controls=protection_controls(1.), particle_count=4)
            assert fixture.manifest['data_role'] == role
            assert fixture.manifest['data_regime'] == regime
            if (role, regime) != ('calibration', 'nominal'):
                with pytest.raises(ValueError, match='local mechanics endpoint'):
                    module.make_nonlinear_consumer(data, controls=protection_controls(1.), batch_size=2)
    with pytest.raises(ValueError, match='calibration or validation'):
        module.make_nonlinear_fixture(dict(dataset(), role='claim'), controls=protection_controls(1.))
