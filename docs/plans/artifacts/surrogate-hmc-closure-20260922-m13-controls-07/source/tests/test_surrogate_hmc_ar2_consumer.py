"""Finite-program derivatives and endpoint binding for structural AR2."""
import tensorflow as tf

from bayesfilter.highdim.ledh_structural_ar2_target_tf import make_ar2_target
from bayesfilter.testing.surrogate_hmc_ar2_consumer import make_ar2_consumer
from bayesfilter.testing.surrogate_hmc_ar2_reference import datasets, make_reference
from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls


def test_both_noise_scale_scores_include_full_canonical_and_prior_dependence():
    data = datasets(horizon=2)["calibration"]
    target = make_ar2_target(data, particle_count=8, controls=warm_start_controls(), dtype=tf.float64)
    evaluate = target.compile(2)
    points = tf.constant([[.2, -.15], [-.3, .1]], tf.float64)
    values, scores, status = evaluate(points)
    assert bool(tf.reduce_all(status["program_valid"]))
    eps = tf.constant(1e-5, tf.float64)
    fd = tf.stack([(evaluate(points+eps*d)[0]-evaluate(points-eps*d)[0])/(2*eps)
                    for d in tf.unstack(tf.eye(2, dtype=tf.float64))], -1)
    tf.debugging.assert_near(scores, fd, atol=2e-6, rtol=2e-6)
    tf.debugging.assert_all_finite(values, "finite structural target")


def test_consumer_calls_canonical_force_and_independent_model_endpoint():
    data = datasets(horizon=2)["validation"]
    consumer = make_ar2_consumer(data, controls=warm_start_controls(), particle_count=8, batch_size=2)
    points = tf.constant([[.1, -.1], [-.2, .2]], tf.float64)
    _, score, status = consumer.evaluate_force(points)
    field = consumer.force.function(points)
    tf.debugging.assert_equal(field.force, -tf.cast(score, tf.float64))
    tf.debugging.assert_equal(field.force_valid, status["force_valid"])
    _, reference, _ = make_reference(2)
    logp, _ = reference(points, tf.constant(data["observations"], tf.float64))
    tf.debugging.assert_near(consumer.endpoint.function(points).potential, -logp)
    assert bool(tf.reduce_all(field.force_valid))
    assert consumer.fixture.manifest["innovation_dimension"] == 1
    assert consumer.fixture.manifest["initial_law"] == "N(0,I2)"
