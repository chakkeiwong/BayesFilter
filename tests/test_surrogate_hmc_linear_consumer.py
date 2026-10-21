"""Cheap wiring tests; the real full-DCTR GPU parity check is separate."""
import tensorflow as tf

from bayesfilter.highdim import ledh_program_fixtures_tf as fixtures
from bayesfilter.testing.surrogate_hmc_linear_consumer import make_linear_consumer
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls


def test_consumer_reaches_canonical_batch_and_preserves_sign_chart_and_status(monkeypatch):
    calls = []

    def canonical(model, theta, directions, base, covariance, noises, observations, **kwargs):
        calls.append((theta.shape, directions.shape, kwargs))
        value = -tf.reduce_sum(theta * theta, axis=-1)
        valid = theta[:, 0] < 1.05
        zeros = tf.zeros([2], tf.int32)
        status = dict(value_valid=tf.ones([2], tf.bool), force_valid=valid,
            value_failure_code=zeros, force_failure_code=tf.where(valid, zeros, 1),
            first_failure_time=-tf.ones([2], tf.int32), first_failure_stage=zeros,
            first_failure_particle=-tf.ones([2], tf.int32),
            direction_value_valid=tf.ones([2, 2], tf.bool),
            direction_force_valid=tf.broadcast_to(valid[:, None], [2, 2]))
        return value, -2 * theta, status

    monkeypatch.setattr(fixtures, "canonical_batch_fused_value_score", canonical)
    data = dict(model_id="LG2", horizon=20, partition="calibration", attempt=1, replicate=0,
                transition_before_observation=True, observations=[[.25, -.5]] * 20)
    controls = warm_start_controls()
    consumer = make_linear_consumer(data, controls=controls, batch_size=2, particle_count=4)
    z = tf.constant([[-.5, .25], [2., -.4]], tf.float64)
    value, score, status = consumer.evaluate_force(z)
    field = consumer.force.function(z)
    assert len(calls) == 1
    assert calls[0][0] == (2, 2) and calls[0][1] == (2, 2, 2)
    assert calls[0][2]["reset_policy"] == "contract_e"
    assert calls[0][2]["pairwise_steps"] == controls["pairwise_steps"]
    assert calls[0][2]["coordinate_cap"] == controls["coordinate_cap"]
    sigmoid = tf.sigmoid(tf.cast(z, tf.float32))
    width = tf.constant([.7, 2.], tf.float32)
    theta = tf.constant([.5, -1.], tf.float32) + width * sigmoid
    expected = -2 * theta * width * sigmoid * (1 - sigmoid) + 1 - 2 * sigmoid
    tf.debugging.assert_near(score[:1], expected[:1])
    tf.debugging.assert_equal(field.force, -tf.cast(score, tf.float64))
    tf.debugging.assert_equal(field.force_valid, status["force_valid"])
    tf.debugging.assert_equal(field.force_valid, [True, False])
    before = consumer.endpoint.function(z)
    consumer.evaluate_force(z)
    after = consumer.endpoint.function(z)
    tf.debugging.assert_equal(before.potential, after.potential)
    logp, exact_score = consumer.evaluate_reference(z)
    tf.debugging.assert_near(before.potential, -logp)
    tf.debugging.assert_near(consumer.exact_force.function(z).force, -exact_score)
    assert consumer.force.identity != consumer.endpoint.identity
    assert value.shape == (2,) and exact_score.shape == (2, 2)
