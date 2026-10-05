"""Independent-reference tests; CPU checks, not nonlinear filter admission."""
import json
import math
import statistics
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.testing.nonlinear_bootstrap_fisher_reference_tf import (
    complete_increment, gaussian_log_density, make_bootstrap_fisher_kernel,
)


@pytest.mark.parametrize('name', ['predator_prey', 'sir_d18'])
def test_complete_data_score_matches_fixed_state_finite_difference(name):
    spec = NonlinearSQMCSpec(name)
    theta = spec.default_theta(tf.float64)
    theta += tf.cast(tf.range(spec.parameter_count), tf.float64) * .001
    parents = spec.initial_mean(tf.float64)[None, :] + tf.reshape(
        tf.cast(tf.range(3*spec.dimension), tf.float64)*.01, [3, spec.dimension])
    model, _ = spec.model(theta, tf.zeros_like(theta))
    states = model.transition_mean_fn(theta, parents) + .2 * tf.sin(parents)
    observation = model.observation_fn(states)[0] + .7
    score = complete_increment(spec.model, theta, parents, states, observation)

    def joint(at):
        current, _ = spec.model(at, tf.zeros_like(at))
        transition = gaussian_log_density(
            states-current.transition_mean_fn(at, parents), current.process_covariance)
        observation_log = (current.observation_log_density_fn(at, states, observation)
                           if current.observation_log_density_fn is not None else
                           gaussian_log_density(observation[None, :]-current.observation_fn(states),
                                                current.observation_covariance))
        return transition + observation_log

    columns = []
    for coordinate in range(spec.parameter_count):
        step = 1.e-5 * max(1., abs(float(theta[coordinate])))
        delta = tf.one_hot(coordinate, spec.parameter_count, dtype=tf.float64)*step
        columns.append((joint(theta+delta)-joint(theta-delta))/(2*step))
    finite_difference = tf.stack(columns, axis=1)
    tf.debugging.assert_near(score, finite_difference, atol=2.e-6, rtol=3.e-5)
    print(json.dumps({'fixture': name, 'fixed_state_score_max_abs_error':
                      float(tf.reduce_max(tf.abs(score-finite_difference)))}))


def test_bootstrap_fisher_agrees_with_exact_transition_first_kalman_reference():
    dtype = tf.float64

    def factory(theta, direction):
        variance = tf.exp(2*theta[1])*tf.eye(1, dtype=dtype)
        return SimpleNamespace(
            transition_mean_fn=lambda theta, x: .7*x+theta[0],
            transition_mean_tangent_fn=lambda theta, x, dx: .7*dx+direction[0],
            observation_fn=lambda x: x,
            process_covariance=tf.constant([[.4]], dtype),
            observation_covariance=variance,
            process_covariance_tangent_fn=None,
            observation_covariance_tangent_fn=lambda theta: 2*variance*direction[1],
            transition_log_density_tangent_fn=None,
            observation_log_density_fn=None,
            observation_log_density_tangent_fn=None,
        ), None

    observations = [.4, -.2, 1.1]
    theta_values = [.3, -.2]

    def exact(theta):
        mean, variance, total = 0., 1., 0.
        noise = math.exp(2*theta[1])
        for observation in observations:
            mean, variance = .7*mean+theta[0], .49*variance+.4
            residual, innovation = observation-mean, variance+noise
            total += -.5*(math.log(2*math.pi*innovation)+residual**2/innovation)
            gain = variance/innovation
            mean, variance = mean+gain*residual, (1-gain)*variance
        return total

    exact_score = []
    for coordinate in range(2):
        plus, minus = theta_values.copy(), theta_values.copy()
        plus[coordinate] += 1.e-6
        minus[coordinate] -= 1.e-6
        exact_score.append((exact(plus)-exact(minus))/2.e-6)
    kernel = make_bootstrap_fisher_kernel(factory, tf.zeros([1], dtype), 2, 1,
                                          3, 4096, jit_compile=False)
    columns = [[], [], []]
    for seed in range(170, 178):
        ell, score, trace = kernel(tf.constant(theta_values, dtype),
                                  tf.constant(observations, dtype)[:, None], tf.constant(seed))
        tf.debugging.assert_all_finite(trace, 'reference trace must be finite')
        for column, value in zip(columns, [float(ell), *score.numpy().tolist()]):
            column.append(value)
    expected = [exact(theta_values), *exact_score]
    observed = [statistics.mean(column) for column in columns]
    mcse = [statistics.stdev(column)/math.sqrt(len(column)) for column in columns]
    for actual, target, se in zip(observed, expected, mcse):
        assert abs(actual-target) < 6*se+.01, (actual, target, se)
    print(json.dumps({'fixture': 'linear_gaussian_transition_first', 'exact': expected,
                      'reference_mean': observed, 'mcse': mcse, 'replications': 8,
                      'particles': 4096, 'CPU_only': True, 'jit_compile': False}))
