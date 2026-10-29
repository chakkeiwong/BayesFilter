"""Tiny CPU reference checks: independent joint Gaussian and stationary series."""
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_linear_reference import make_linear_reference


def dense_stationary_log_likelihood(theta, observations, spec, with_predictive=False):
    transition = theta[0] * spec["base_f"]
    process = tf.exp(2. * theta[1]) * spec["base_q"]
    term = process
    covariance = term
    # Independent convergent stationary series; no shared Lyapunov solver.
    for _ in range(1024):
        term = transition @ term @ tf.transpose(transition)
        covariance = covariance + term
    powers = [tf.eye(spec["d"], dtype=tf.float64)]
    for _ in range(1, spec["t"]):
        powers.append(transition @ powers[-1])
    rows = []
    for i in range(spec["t"]):
        cells = []
        for j in range(spec["t"]):
            lag = powers[abs(i-j)] @ covariance
            lag = lag if i >= j else tf.transpose(lag)
            value = spec["h"] @ lag @ tf.transpose(spec["h"])
            cells.append(value + spec["r"] if i == j else value)
        rows.append(tf.concat(cells, axis=1))
    chol = tf.linalg.cholesky(tf.concat(rows, axis=0))
    whitened = tf.linalg.triangular_solve(chol, tf.reshape(observations, [-1, 1]))
    size = int(observations.shape.num_elements())
    log_likelihood = -.5 * (size * tf.math.log(tf.constant(2*3.141592653589793, tf.float64))
                  + 2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
                  + tf.reduce_sum(whitened**2))
    if not with_predictive:
        return log_likelihood
    powers.append(transition @ powers[-1])
    cross = tf.concat([spec["h"] @ powers[spec["t"]-i] @ covariance @ tf.transpose(spec["h"])
                       for i in range(spec["t"])], axis=1)
    conditional_mean = cross @ tf.linalg.cholesky_solve(chol, tf.reshape(observations, [-1, 1]))
    conditional_covariance = spec["h"] @ covariance @ tf.transpose(spec["h"]) + spec["r"]
    conditional_covariance -= cross @ tf.linalg.cholesky_solve(chol, tf.transpose(cross))
    probability = .5 * tf.math.erfc(-conditional_mean[:, 0] / tf.sqrt(
        2. * tf.linalg.diag_part(conditional_covariance)))
    return log_likelihood, probability


def test_LG2_stationary_scale_and_chart_match_independent_joint_gaussian():
    reference = make_linear_reference("LG2")
    observations = tf.reshape(tf.linspace(tf.constant(-.8, tf.float64), tf.constant(.9, tf.float64), 40), [20, 2])
    theta = tf.constant([[1., 0.], [.51, -.95], [1.19, .95]], tf.float64)
    spec = reference.specification
    lower, upper = tf.constant(spec["lower"],tf.float64), tf.constant(spec["upper"],tf.float64)
    fraction = (theta-lower)/(upper-lower)
    z = tf.math.log(fraction)-tf.math.log1p(-fraction)
    actual_value, actual_score = reference.unconstrained_value_score(z, observations)
    expected_values, expected_scores, expected_predictive = [], [], []
    for index in range(3):
        position = z[index]
        with tf.GradientTape() as tape:
            tape.watch(position)
            physical = lower + (upper-lower)*tf.sigmoid(position)
            value, predictive = dense_stationary_log_likelihood(physical, observations, spec, with_predictive=True)
            value -= tf.reduce_sum(tf.nn.softplus(position)+tf.nn.softplus(-position))
        expected_values.append(value)
        expected_scores.append(tape.gradient(value, position))
        expected_predictive.append(predictive)
    tf.debugging.assert_near(actual_value,tf.stack(expected_values),atol=1e-9,rtol=0.)
    tf.debugging.assert_near(actual_score,tf.stack(expected_scores),atol=1e-8,rtol=1e-8)
    actual_physical, actual_predictive = reference.physical_statistics(theta, observations)
    tf.debugging.assert_near(actual_predictive,tf.stack(expected_predictive),atol=1e-10,rtol=1e-10)
    tf.debugging.assert_near(reference.physical_log_likelihood(theta,observations),actual_physical,atol=1e-12,rtol=1e-12)
    reordered = reference.unconstrained_value_score(tf.gather(z,[2,0,1]),observations)
    tf.debugging.assert_equal(reordered[0],tf.gather(actual_value,[2,0,1]))
    single = reference.unconstrained_value_score(z[1:2],observations)
    tf.debugging.assert_near(single[0],actual_value[1:2],atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(single[1],actual_score[1:2],atol=1e-11,rtol=1e-11)
