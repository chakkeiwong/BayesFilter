"""Complete SRUKF likelihood/mean/covariance checks under observation rescaling."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.nonlinear.rectangular_srukf_tf import (
    TFRectangularSRUKFModel,
    tf_rectangular_srukf_value,
)
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def independent_kalman(observations, transition, observation, prior_factor, process_factor, observation_factor):
    mean = np.zeros(3)
    covariance = prior_factor @ prior_factor.T
    likelihood = 0.
    for point in observations:
        mean = transition @ mean
        covariance = transition @ covariance @ transition.T + process_factor @ process_factor.T
        residual = point - observation @ mean
        variance = observation @ covariance @ observation.T + observation_factor @ observation_factor.T
        gain = np.linalg.solve(variance, observation @ covariance).T
        likelihood -= .5 * (3. * np.log(2. * np.pi) + np.linalg.slogdet(variance)[1] +
            residual @ np.linalg.solve(variance, residual))
        mean += gain @ residual
        covariance -= gain @ variance @ gain.T
    return likelihood, mean, .5 * (covariance + covariance.T)


@pytest.mark.parametrize('horizon', [1, 3])
def test_complete_value_route_preserves_scale_equivariance(horizon, request):
    transition = np.array([[.8, .1, -.05], [.04, .7, .15], [-.02, .06, .9]])
    observation = np.array([[2., 1., -.5], [1., 3., .25], [-.5, .25, 1.]])
    prior = np.diag([.5, .3, .4])
    process = np.diag([.2, .1, .15])
    noise = np.diag([.15, .1, .2])
    points = np.array([[.1, -.2, .3], [.2, .1, -.1], [-.1, .15, .2]])[:horizon]
    reference = independent_kalman(points, transition, observation, prior, process, noise)
    transition_tf, observation_tf = tf.constant(transition, D), tf.constant(observation, D)
    records = []
    for scale in (1e-12, 1e-6, 1.):
        scale_tf = tf.constant(scale, D)
        model = TFRectangularSRUKFModel(tf.zeros([1, 3], D), tf.constant(prior[None], D),
            tf.constant(process[None], D), tf.constant(noise[None] * scale, D),
            lambda state, process: tf.linalg.matmul(state, transition_tf, transpose_b=True) + process,
            lambda state, scale_tf=scale_tf: scale_tf * tf.linalg.matmul(state, observation_tf, transpose_b=True))
        operands = tf.constant(points[None] * scale, D)
        before = tf_rectangular_srukf_value(operands, model, jit_compile=False)
        after = tf_rectangular_srukf_value(operands, model)
        expected_value = reference[0] - 3. * horizon * np.log(scale)
        covariance = after.filtered_factor.numpy()[0] @ after.filtered_factor.numpy()[0].T
        original_covariance = before.filtered_factor.numpy()[0] @ before.filtered_factor.numpy()[0].T
        np.testing.assert_allclose(after.log_likelihood, [expected_value], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(after.filtered_mean, reference[1][None], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(covariance, reference[2], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(after.log_likelihood, before.log_likelihood, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(after.filtered_mean, before.filtered_mean, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(covariance, original_covariance, rtol=1e-10, atol=1e-10)
        assert bool(after.diagnostics['on_support'][0])
        assert int(after.diagnostics['minimum_observation_rank'][0]) == 3
        records.append({'scale': scale, 'likelihood': clean(after.log_likelihood),
            'reference_likelihood': expected_value, 'graph_likelihood': clean(before.log_likelihood),
            'mean': clean(after.filtered_mean), 'reference_mean': clean(reference[1]),
            'covariance': clean(covariance), 'reference_covariance': clean(reference[2])})
    save(request, f'srukf-scale-horizon-{horizon}.json', {'horizon': horizon, 'records': records,
        'nonclaims': ['Linear model equivalence; no nonlinear accuracy or posterior claim.']})
