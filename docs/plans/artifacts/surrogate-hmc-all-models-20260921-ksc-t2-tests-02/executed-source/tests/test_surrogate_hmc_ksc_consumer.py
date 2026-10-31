"""Independent reference checks of the KSC analytical model callbacks."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_ksc_consumer import ksc_model
from bayesfilter.testing.surrogate_hmc_ksc_reference import make_ksc_dataset, check_dataset, component_statistics
from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture


def test_density_total_tangent_includes_state_and_beta_dependence():
    model = ksc_model(tf.float64)
    theta = tf.constant([[.2, .3], [.5, .5], [.8, .7]], tf.float64)
    state = tf.constant([[-2.], [.5], [3.]], tf.float64)
    direction = tf.constant([[.1, -.3], [-.2, .2], [.3, .1]], tf.float64)
    dstate = tf.constant([[.7], [-.4], [.2]], tf.float64)
    observation = tf.constant([-1.3], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch((theta, state))
        value = model.observation_log_density_fn(theta, state, observation)
    gradient_theta, gradient_state = tape.gradient(value, (theta, state))
    expected = tf.reduce_sum(gradient_theta * direction, axis=1) + tf.reduce_sum(gradient_state * dstate, axis=1)
    actual = model.observation_log_density_tangent_fn(theta, state, observation, dstate, direction)
    tf.debugging.assert_near(actual, expected, atol=1e-12, rtol=1e-12)


def test_conditional_x2_moments_equal_independent_sequential_updates():
    data = make_ksc_dataset(partition="diagnostic")
    observation = check_dataset(data)
    mixture = ksc_1998_log_chi_square_mixture()
    theta = tf.constant([[.15, .85], [.5, .5], [.85, .15]], tf.float64)
    _, _, means, variances = component_statistics(theta, observation)
    for index, point in enumerate(tf.unstack(theta)):
        gamma, beta = point
        stationary = 1 / (1 - gamma**2)
        innovation_variance = stationary + mixture.variances
        residual = observation[0] - 2 * tf.math.log(beta) - mixture.means
        updated_mean = stationary / innovation_variance * residual
        updated_variance = stationary * mixture.variances / innovation_variance
        predicted_mean = gamma * updated_mean
        predicted_variance = gamma**2 * updated_variance + 1
        second_variance = predicted_variance[:, None] + mixture.variances[None]
        gain = predicted_variance[:, None] / second_variance
        second_residual = observation[1] - predicted_mean[:, None] - 2 * tf.math.log(beta) - mixture.means[None]
        expected_mean = predicted_mean[:, None] + gain * second_residual
        expected_variance = predicted_variance[:, None] * mixture.variances[None] / second_variance
        tf.debugging.assert_near(means[index], tf.reshape(expected_mean, [-1]), atol=1e-12, rtol=1e-12)
        tf.debugging.assert_near(variances[index], tf.reshape(expected_variance, [-1]), atol=1e-12, rtol=1e-12)
