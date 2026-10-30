"""Independent CPU diagnostic checks of the SIR KDM proposal mechanism."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
from bayesfilter.highdim.ledh_neural_kdm_tf import (
    FrozenConditionalKDMProposal, ObservationKDMNetwork,
)
from bayesfilter.highdim.ledh_physical_proposal_tf import conditional_moments, physical_mixture_draw
from bayesfilter.score_study.moment_training_tf import make_moment_endpoint, make_moment_objective
from tests.highdim.test_ledh_backward_guide_tf import _arguments, _curved_observation_model
from tests.highdim.test_ledh_canonical_total_gaussian_tf import DTYPE, _parameterized_model
from tests.highdim.test_ledh_physical_proposal_tf import _case, _kernel


def _coefficients():
    return (tf.constant([[[.2, -.1], [-.3, .4]]] * 2, DTYPE),
            tf.constant([[[.1, -.2], [.05, .15]]] * 2, DTYPE))


def _neural_kernel(fixture, beta=.5, model_factory=_parameterized_model):
    return build_backward_guided_ledh_kernel(model_factory, None,
        horizon=2, particle_count=8, state_dimension=2, observation_dimension=2,
        parameter_dimension=4, dtype=DTYPE, jit_compile=False,
        physical_mixture_weight=beta, neural_mixture_components=2, return_trace=True,
        **fixture["options"])


def test_neural_physical_density_is_full_independent_tfp_mixture():
    fixture, theta, uniforms = _case()
    model = _parameterized_model(theta, tf.zeros_like(theta))
    anchors, noise = fixture["initial_states"], fixture["noises"][0]
    zero = tf.zeros_like(anchors)
    matrix = tf.eye(2, batch_shape=[8], dtype=DTYPE) * .8
    flow_mean = anchors + .2
    flow_factor = matrix @ tf.linalg.cholesky(model.process_covariance)
    baseline = flow_mean + tf.linalg.matvec(flow_factor, noise)
    determinant = tf.linalg.logdet(matrix)
    flow = (baseline, zero, determinant, tf.zeros_like(determinant),
            flow_mean, zero, matrix, tf.zeros_like(matrix))
    guide = FrozenConditionalKDMProposal(*_coefficients())
    result = physical_mixture_draw(model, guide, 0, anchors, zero,
        fixture["observations"][0], tf.zeros_like(model.process_covariance),
        tf.zeros_like(model.observation_covariance), noise, uniforms[0], .5, flow)
    points, _, actual, _, valid, count = result
    means, covariance, *_ = conditional_moments(model, anchors, zero,
        fixture["observations"][0], tf.zeros_like(model.process_covariance),
        tf.zeros_like(model.observation_covariance))
    factor = tf.linalg.cholesky(covariance)
    offsets, log_scales = _coefficients()
    components = tfp.distributions.MultivariateNormalTriL(
        means[:, None] + tf.linalg.matvec(factor[:, None], offsets[0]),
        factor[:, None] * tf.exp(log_scales[0])[None, :, None, :])
    component_log = components.log_prob(points[:, None]) - tf.math.log(tf.constant(4., DTYPE))
    flow_law = tfp.distributions.MultivariateNormalTriL(flow_mean, flow_factor)
    expected = tf.reduce_logsumexp(tf.concat([
        (flow_law.log_prob(points) - tf.math.log(tf.constant(2., DTYPE)))[:, None],
        component_log], axis=1), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    assert bool(valid) and int(count) == 4


def test_zero_neural_parameters_recover_conditional_consumer():
    fixture, theta, uniforms = _case()
    arguments = _arguments(theta, tf.ones_like(theta) * .1, fixture)
    actual = _neural_kernel(fixture)(*arguments, uniforms,
        tf.zeros([2, 2, 2], DTYPE), tf.zeros([2, 2, 2], DTYPE))
    expected = _kernel(fixture, None, .5, return_trace=True)(*arguments, uniforms)
    for observed, baseline in zip(actual[:2], expected[:2]):
        np.testing.assert_allclose(observed, baseline, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(actual[2][-1]["states_after_reset"],
                               expected[2][-1]["states_after_reset"], atol=3e-11)


def test_neural_consumer_analytical_theta_tangent():
    fixture, theta, uniforms = _case()
    kernel = _neural_kernel(fixture, model_factory=_curved_observation_model)
    direction = tf.constant([.4, -.3, .2, .5], DTYPE)
    actual = kernel(*_arguments(theta, direction, fixture), uniforms, *_coefficients())
    delta = 1e-5
    values = [kernel(*_arguments(theta + sign * delta * direction, tf.zeros_like(theta), fixture),
                     uniforms, *_coefficients())[0] for sign in (-1., 1.)]
    assert all(bool(step["program_valid"]) for step in actual[2])
    np.testing.assert_allclose(actual[1][0], (values[1] - values[0]) / (2 * delta),
                               rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1000.])
def test_invalid_neural_scale_rejects(bad):
    fixture, theta, uniforms = _case()
    offsets, scales = _coefficients()
    scales = tf.tensor_scatter_nd_update(scales, [[0, 0, 0]], [tf.cast(bad, DTYPE)])
    value, _, trace = _neural_kernel(fixture)(
        *_arguments(theta, tf.zeros_like(theta), fixture), uniforms, offsets, scales)
    assert not bool(trace[0]["program_valid"])
    assert not bool(tf.math.is_finite(value))


def test_actual_terminal_loss_reaches_every_neural_tensor_and_matches_difference():
    fixture, theta, uniforms = _case()
    network = ObservationKDMNetwork(2, 2, 2, 4, tf.zeros([2, 2], DTYPE),
        tf.ones([2, 2], DTYPE), seed=263044001, jit_compile=False)
    objective = make_moment_objective(8, 2, tf.zeros([2], DTYPE),
        tf.ones([2], DTYPE), tf.ones([5], DTYPE), jit_compile=False)
    endpoint = make_moment_endpoint(_neural_kernel(fixture), objective, 2, jit_compile=False)
    arguments = _arguments(theta, tf.zeros_like(theta), fixture)

    def loss():
        coefficients = network.predict(fixture["observations"])
        result = endpoint(*arguments, uniforms, *coefficients, tf.constant([.7, -.8], DTYPE))
        assert bool(result["valid"])
        return result["loss"]

    with tf.GradientTape() as tape:
        value = loss()
    gradients = tape.gradient(value, network.parameters)
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients)
    assert all(float(tf.linalg.norm(gradient)) > 0. for gradient in gradients)
    directions = [tf.random.stateless_normal(parameter.shape, [901, index], dtype=DTYPE)
                  for index, parameter in enumerate(network.parameters)]
    expected = sum(tf.reduce_sum(gradient * direction)
                   for gradient, direction in zip(gradients, directions))
    initial = [tf.identity(parameter) for parameter in network.parameters]
    delta = 1e-5
    values = []
    for sign in (-1., 1.):
        for parameter, start, direction in zip(network.parameters, initial, directions):
            parameter.assign(start + sign * delta * direction)
        values.append(loss())
    np.testing.assert_allclose(expected, (values[1] - values[0]) / (2 * delta),
                               rtol=2e-3, atol=2e-4)
