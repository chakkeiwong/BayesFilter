"""Independent reference checks, not posterior or DCTR promotion."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ar2_reference import datasets, make_reference


def test_ar2_kalman_matches_joint_observation_gaussian_and_derivative():
    observations = tf.constant(datasets()["calibration"]["observations"], tf.float64)
    stats, value_score, joint = make_reference()
    theta = tf.constant([[0., 0.], [-1., 1.], [1., -1.]], tf.float64)
    ll, probability = stats(theta, observations)
    tf.debugging.assert_near(ll, joint(theta, observations), atol=2e-10, rtol=2e-10)
    assert bool(tf.reduce_all((probability >= 0) & (probability <= 1)))
    assert bool(tf.reduce_all(probability[:, 0] >= probability[:, 1]))
    _, gradient = value_score(theta, observations)
    eps = tf.constant(1e-5, tf.float64)
    fd = tf.stack([(value_score(theta+eps*d, observations)[0]-value_score(theta-eps*d, observations)[0])/(2*eps)
                  for d in tf.unstack(tf.eye(2, dtype=tf.float64))], -1)
    tf.debugging.assert_near(gradient, fd, atol=2e-6, rtol=2e-6)


def test_ar2_data_are_fixed_and_partitioned():
    first, second = datasets(), datasets()
    assert first == second
    assert len({tuple(record["seed"]) for record in first.values()}) == 3
    assert all(record["initial_law"] == "N(0,I2)" and record["transition_before_observation"] for record in first.values())
    assert first["calibration"]["observations"] != first["claim"]["observations"]
