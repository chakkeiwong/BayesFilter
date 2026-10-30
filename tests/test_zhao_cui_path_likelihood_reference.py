"""CPU independent-reference checks; NumPy/SciPy supply Gaussian density authority."""
import numpy as np
import pytest
from scipy.stats import multivariate_normal
import tensorflow as tf
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.testing.zhao_cui_path_likelihood_reference import make_path_log_joint


@pytest.mark.parametrize('model_name', ['predator_prey', 'sir_d18'])
def test_xla_path_values_match_timewise_gaussians_at_perturbed_parameters(model_name):
    spec = NonlinearSQMCSpec(model_name)
    rng = np.random.default_rng(40621)
    horizon = 3
    mean = spec.initial_mean(tf.float64).numpy()
    paths = mean + rng.normal(size=(3, horizon + 1, spec.dimension))
    theta0 = spec.default_theta(tf.float64).numpy()
    scale = theta0 if model_name == 'predator_prey' else np.ones_like(theta0)
    observations = paths[0, 1:, :] if model_name == 'predator_prey' else paths[0, 1:, 1::2]
    observations = observations + rng.normal(size=observations.shape)
    evaluate = make_path_log_joint(model_name, horizon)
    values = []
    for step in (0., .013, -.009):
        theta = tf.constant(theta0 + step * scale, tf.float64)
        model, _ = spec.model(theta, tf.zeros_like(theta))
        expected = []
        for path in paths:
            value = multivariate_normal.logpdf(path[0], mean, np.eye(spec.dimension))
            for t in range(horizon):
                prediction = model.transition_mean_fn(theta, tf.constant(path[t:t+1])).numpy()[0]
                obs_prediction = model.observation_fn(tf.constant(path[t+1:t+2])).numpy()[0]
                value += multivariate_normal.logpdf(path[t+1], prediction, model.process_covariance.numpy())
                value += multivariate_normal.logpdf(observations[t], obs_prediction, model.observation_covariance.numpy())
            expected.append(value)
        actual = evaluate(theta, tf.constant(paths), tf.constant(observations)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        values.append(actual)
    assert not np.array_equal(values[0], values[1])
    assert evaluate.experimental_get_tracing_count() == 1


def test_sir_observation_scale_keeps_log_determinant():
    spec = NonlinearSQMCSpec('sir_d18')
    mean = spec.initial_mean(tf.float64).numpy()
    paths = np.broadcast_to(mean, (2, 2, 18)).copy()
    observations = mean[None, 1::2].copy()
    evaluate = make_path_log_joint('sir_d18', 1)
    base = evaluate(tf.constant([0., 0., 0.], tf.float64), tf.constant(paths), tf.constant(observations)).numpy()
    wider = evaluate(tf.constant([0., 0., .1], tf.float64), tf.constant(paths), tf.constant(observations)).numpy()
    np.testing.assert_allclose(wider - base, -.9, rtol=1e-12, atol=1e-10)
