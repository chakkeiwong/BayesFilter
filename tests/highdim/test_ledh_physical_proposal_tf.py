"""CPU diagnostic tests, never claim-bearing LEDH score or tuning evidence."""

import os
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim import ledh_canonical_score_tf as shared
from bayesfilter.highdim import ledh_physical_proposal_tf as physical
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenQuadraticBackwardGuide,
    build_backward_guided_ledh_kernel,
)
from tests.highdim.test_ledh_backward_guide_tf import (
    _arguments, _curved_observation_model, _mixture,
)
from tests.highdim.test_ledh_canonical_total_gaussian_tf import (
    DTYPE, _fixture, _parameterized_model,
)


def _case():
    fixture = _fixture()
    fixture["options"].update(reset_epsilon=1., reset_sinkhorn_steps=5, reset_balance_steps=32)
    theta = tf.constant([.82, 1.1, -.35, -.2], DTYPE)
    uniforms = tf.constant([[.1, .8, .2, .7, .3, .6, .4, .9]] * 2, DTYPE)
    return fixture, theta, uniforms


def _kernel(fixture, guide, beta, model_factory=_parameterized_model, **extra):
    return build_backward_guided_ledh_kernel(
        model_factory, guide, horizon=2, particle_count=8, state_dimension=2,
        observation_dimension=2, parameter_dimension=4, dtype=DTYPE,
        jit_compile=False, physical_mixture_weight=beta,
        **fixture["options"], **extra)


@pytest.mark.parametrize("family", ["none", "quadratic", "mixture"])
def test_physical_shared_program_total_derivative(family):
    fixture, theta, uniforms = _case()
    guide = None
    if family == "quadratic":
        guide = FrozenQuadraticBackwardGuide(
            tf.eye(2, batch_shape=[2], dtype=DTYPE) * .2, tf.ones([2, 2], DTYPE) * .1)
    if family == "mixture":
        guide = replace(_mixture(), conditioning="mixture_product_moments")
    kernel = _kernel(fixture, guide, .5, _curved_observation_model)
    direction = tf.constant([.4, -.3, .2, .5], DTYPE)
    d_states = .03 * tf.ones_like(fixture["initial_states"])
    d_covariances = .02 * fixture["initial_covariances"]
    value, derivative = kernel(*_arguments(theta, direction, fixture, d_states, d_covariances), uniforms)
    delta = tf.constant(2e-6, DTYPE)
    values = []
    for sign in (-1, 1):
        moved = dict(fixture)
        moved["initial_states"] = fixture["initial_states"] + sign * delta * d_states
        moved["initial_covariances"] = fixture["initial_covariances"] + sign * delta * d_covariances
        values.append(kernel(*_arguments(theta + sign * delta * direction,
                                        tf.zeros_like(theta), moved), uniforms)[0])
    assert bool(tf.math.is_finite(value))
    np.testing.assert_allclose(derivative[0], (values[1] - values[0]) / (2 * delta),
                               rtol=3e-4, atol=3e-5)
    assert kernel.experimental_get_tracing_count() == 1


def test_beta_zero_is_bitwise_unchanged_shared_program():
    fixture, theta, uniforms = _case()
    arguments = _arguments(theta, tf.ones_like(theta) * .1, fixture)
    actual = shared._value_and_analytical_score_impl(
        _parameterized_model(theta, arguments[1]), theta, *arguments[2:6],
        with_score=True, initial_state_tangent=arguments[6],
        initial_covariance_tangent=arguments[7], physical_mixture_weight=0.,
        physical_component_uniforms=uniforms, **fixture["options"])
    expected = shared._value_and_analytical_score_impl(
        _parameterized_model(theta, arguments[1]), theta, *arguments[2:6],
        with_score=True, initial_state_tangent=arguments[6],
        initial_covariance_tangent=arguments[7], **fixture["options"])
    for observed, baseline in zip(actual, expected):
        np.testing.assert_array_equal(observed, baseline)


def _primitive(beta, scale=1.):
    fixture, theta, uniforms = _case()
    model = _parameterized_model(theta, tf.zeros_like(theta))
    anchors = fixture["initial_states"]
    noise = fixture["noises"][0]
    matrix = tf.eye(2, batch_shape=[8], dtype=DTYPE) * scale
    flow_mean = anchors + .2
    baseline = flow_mean + tf.linalg.matvec(matrix @ tf.linalg.cholesky(model.process_covariance), noise)
    zero = tf.zeros_like(anchors)
    determinant = tf.linalg.logdet(matrix)
    flow = (baseline, zero, determinant, tf.zeros_like(determinant),
            flow_mean, zero, matrix, tf.zeros_like(matrix))
    result = physical.physical_mixture_draw(
        model, None, 0, anchors, zero, fixture["observations"][0],
        tf.zeros_like(model.process_covariance), tf.zeros_like(model.observation_covariance),
        noise, uniforms[0], beta, flow)
    return result, model, anchors, fixture, matrix, flow_mean


def test_full_mixture_denominator_at_both_selected_components():
    result, model, anchors, fixture, matrix, flow_mean = _primitive(.5, .7)
    points, _, log_density, _, valid, count = result
    means, covariance, *_ = physical.conditional_moments(
        model, anchors, tf.zeros_like(anchors), fixture["observations"][0],
        tf.zeros_like(model.process_covariance), tf.zeros_like(model.observation_covariance))
    baseline_law = tfp.distributions.MultivariateNormalTriL(
        flow_mean, matrix @ tf.linalg.cholesky(model.process_covariance))
    output_law = tfp.distributions.MultivariateNormalTriL(means, tf.linalg.cholesky(covariance))
    expected = tf.reduce_logsumexp(tf.stack([
        baseline_law.log_prob(points), output_law.log_prob(points)], -1), -1) - tf.math.log(tf.constant(2., DTYPE))
    np.testing.assert_allclose(log_density, expected, rtol=2e-13, atol=2e-13)
    assert bool(valid)
    assert int(count) == 4


def test_exact_conditional_cancels_flow_and_parent_weight_is_predictive():
    result, model, anchors, fixture, _, _ = _primitive(1., .7)
    altered = _primitive(1., 3.)[0]
    for actual, expected in zip(result, altered):
        np.testing.assert_array_equal(actual, expected)
    points, _, log_density, _, valid, _ = result
    transition = tfp.distributions.MultivariateNormalTriL(anchors, tf.linalg.cholesky(model.process_covariance))
    observation = tfp.distributions.MultivariateNormalTriL(
        model.observation_fn(points), tf.linalg.cholesky(model.observation_covariance))
    jacobian = model.observation_jacobian_fn(anchors)
    predictive = tfp.distributions.MultivariateNormalTriL(
        model.observation_fn(anchors), tf.linalg.cholesky(
            jacobian @ model.process_covariance @ tf.linalg.matrix_transpose(jacobian)
            + model.observation_covariance))
    weight = transition.log_prob(points) + observation.log_prob(fixture["observations"][0]) - log_density
    np.testing.assert_allclose(weight, predictive.log_prob(fixture["observations"][0]), atol=2e-12)
    assert bool(valid)
    assert float(tf.math.reduce_std(weight)) > 0.


@pytest.mark.parametrize("precision,expected", [(.2, True), (-.01, False)])
def test_physical_guide_validity_matches_consumed_guide(precision, expected):
    fixture, theta, uniforms = _case()
    guide = FrozenQuadraticBackwardGuide(
        tf.eye(2, batch_shape=[2], dtype=DTYPE) * precision, tf.zeros([2, 2], DTYPE))
    value, _, trace = _kernel(fixture, guide, .5, return_trace=True)(
        *_arguments(theta, tf.zeros_like(theta), fixture), uniforms)
    for step in trace:
        assert bool(step["backward_guide_valid"]) is expected
        assert bool(step["guided_proposal_valid"]) is expected
        assert bool(step["physical_proposal_valid"]) is expected
        assert bool(step["program_valid"]) is expected
    assert bool(tf.math.is_finite(value)) is expected


@pytest.mark.parametrize("physical_beta", [None, .5])
def test_conditioning_margin_reaches_actual_moment_endpoint(physical_beta):
    from bayesfilter.score_study.moment_training_tf import make_moment_endpoint, make_moment_objective

    fixture, theta, uniforms = _case()
    guide = replace(_mixture(), conditioning="mixture_product_moments",
                    active=tf.constant([True, False]))
    kernel = _kernel(fixture, guide, physical_beta, return_trace=True)
    arguments = _arguments(theta, tf.zeros_like(theta), fixture)
    if physical_beta is not None:
        arguments = (*arguments, uniforms)
    _, _, trace = kernel(*arguments)
    assert bool(trace[0]["guide_conditioning_margin_measured"])
    assert float(trace[0]["guide_conditioning_min_relative_spd_margin"]) > 0.
    assert not bool(trace[1]["guide_conditioning_margin_measured"])
    assert float(trace[1]["guide_conditioning_min_relative_spd_margin"]) == 0.
    assert bool(trace[1]["backward_guide_valid"])
    objective = make_moment_objective(8, 2, tf.zeros([2], DTYPE),
        tf.ones([2], DTYPE), tf.ones([5], DTYPE), jit_compile=False)
    endpoint = make_moment_endpoint(kernel, objective, 2, jit_compile=False)
    output = endpoint(*arguments, tf.zeros([2], DTYPE))
    np.testing.assert_array_equal(output["diagnostics"]["guide_conditioning_margin_measured"], [True, False])
    np.testing.assert_array_equal(output["diagnostics"]["guide_conditioning_min_relative_spd_margin"],
        [step["guide_conditioning_min_relative_spd_margin"] for step in trace])
    assert bool(output["valid"])


def test_factory_calls_shared_flow_and_physical_consumer(monkeypatch):
    fixture, theta, uniforms = _case()
    calls = []
    original_flow = shared._flow_substeps_with_tangent
    original_proposal = physical.physical_mixture_draw

    def record_flow(*arguments, **options):
        calls.append("flow")
        return original_flow(*arguments, **options)

    def record_proposal(*arguments, **options):
        calls.append("physical")
        return original_proposal(*arguments, **options)

    monkeypatch.setattr(shared, "_flow_substeps_with_tangent", record_flow)
    monkeypatch.setattr(physical, "physical_mixture_draw", record_proposal)
    kernel = _kernel(fixture, None, .5, return_trace=True)
    _, _, trace = kernel(*_arguments(theta, tf.zeros_like(theta), fixture), uniforms)
    assert calls == ["flow", "physical"] * 2
    assert all(bool(step["physical_proposal_valid"]) for step in trace)
    assert all(int(step["physical_output_component_count"]) == 4 for step in trace)


def test_dynamic_quadratic_signature_preserves_uniform_position():
    fixture, theta, uniforms = _case()
    kernel = _kernel(fixture, None, .5, dynamic_quadratic=True)
    actual = kernel(*_arguments(theta, tf.zeros_like(theta), fixture), uniforms,
                    tf.zeros([2, 2, 2], DTYPE), tf.zeros([2, 2], DTYPE))
    expected = _kernel(fixture, None, .5)(*_arguments(theta, tf.zeros_like(theta), fixture), uniforms)
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-11, atol=1e-11)


def test_dynamic_product_mixture_reaches_same_physical_consumer():
    fixture, theta, uniforms = _case()
    guide = replace(_mixture(), conditioning="mixture_product_moments")
    kernel = _kernel(fixture, None, .5, mixture_components=int(guide.logits.shape[1]),
                     mixture_conditioning="mixture_product_moments")
    arguments = _arguments(theta, tf.ones_like(theta) * .1, fixture)
    actual = kernel(*arguments, uniforms, guide.logits, guide.centers,
                    guide.covariances, guide.active)
    expected = _kernel(fixture, guide, .5)(*arguments, uniforms)
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("invalid_uniform", [-.1, 1., float("nan")])
def test_invalid_component_uniform_cannot_issue_valid_score(invalid_uniform):
    fixture, theta, uniforms = _case()
    uniforms = tf.tensor_scatter_nd_update(uniforms, [[0, 0]],
                                          tf.constant([invalid_uniform], DTYPE))
    _, _, trace = _kernel(fixture, None, .5, return_trace=True)(
        *_arguments(theta, tf.zeros_like(theta), fixture), uniforms)
    assert not bool(trace[0]["physical_proposal_valid"])
