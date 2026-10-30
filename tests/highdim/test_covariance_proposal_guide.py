"""CPU/FP64 independent diagnostics for the optional guided mixture candidate."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide,
    FrozenQuadraticBackwardGuide,
)
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec


DTYPE = tf.float64


def fixed_design(count=24, dimension=2, horizon=3):
    return (
        tf.random.stateless_normal([count, dimension], [861, 1], dtype=DTYPE),
        tf.random.stateless_normal([horizon, count, dimension], [861, 2], dtype=DTYPE),
        tf.random.stateless_uniform([horizon, count, 2], [861, 3], dtype=DTYPE),
    )


def future_guide(kind):
    precision = tf.constant([[[.6, .1], [.1, .4]], [[.3, -.08], [-.08, .5]], [[0., 0.], [0., 0.]]], DTYPE)
    linear = tf.constant([[.4, -.2], [-.1, .3], [0., 0.]], DTYPE)
    if kind == "zero":
        return FrozenQuadraticBackwardGuide(tf.zeros_like(precision), tf.zeros_like(linear))
    if kind == "quadratic":
        return FrozenQuadraticBackwardGuide(precision, linear)
    return FrozenMixtureBackwardGuide(
        tf.constant([[.2, -.3], [-.4, .1], [0., 0.]], DTYPE),
        tf.constant([[[-.8, .4], [.7, -.6]], [[-.4, -.7], [.9, .5]], [[0., 0.], [0., 0.]]], DTYPE),
        tf.constant([[[[.8, .15], [.15, 1.1]], [[1.2, -.2], [-.2, .7]]]] * 3, DTYPE),
        tf.constant([True, True, False]),
    )


def test_guided_maps_keep_prior_and_match_independent_density():
    spec = LGSSMSpec("diagonal_ar", 1)
    theta = tf.constant([.9, .6, .8], DTYPE)
    model, _ = spec.model(theta, tf.zeros_like(theta))
    ancestors, noise, uniforms = fixed_design(dimension=1, horizon=1)
    observation = tf.constant([.5], DTYPE)
    controls = core.ProposalControls(proposal_policy="original_three_branch", flow_steps=64, reset_steps=5)
    guide = FrozenQuadraticBackwardGuide(tf.constant([[[.7]]], DTYPE), tf.constant([[.4]], DTYPE))
    plain = core.build_maps(model, theta, ancestors, tf.zeros_like(ancestors), observation, controls)
    guided = core.build_maps(model, theta, ancestors, tf.zeros_like(ancestors), observation, controls,
                             backward_guide=guide, time=0)
    assert bool(guided[12])
    for index in (0, 2, 4, 6):
        np.testing.assert_array_equal(guided[index][0], plain[index][0])
    predictive_mean = .9 * np.mean(ancestors.numpy())
    predictive_variance = .9 ** 2 * np.var(ancestors.numpy()) + .6 ** 2
    expected_variance = 1 / (1 / predictive_variance + 1 / .8 ** 2 + .7)
    expected_mean = expected_variance * (predictive_mean / predictive_variance + .5 / .8 ** 2 + .4)
    global_means = guided[0][1, :, 0].numpy()
    np.testing.assert_allclose(np.mean(global_means), expected_mean, atol=1e-12)
    np.testing.assert_allclose(np.var(global_means) + guided[2][1, 0, 0, 0], expected_variance, atol=1e-12)
    conditional_variance = 1 / (1 / .6 ** 2 + 1 / .8 ** 2 + .7)
    np.testing.assert_allclose(guided[2][2, :, 0, 0], conditional_variance, rtol=.02)
    beta = tf.constant([.2, .3, .5], DTYPE)
    children, d_children = core.draw_from_maps(guided, beta, uniforms[0], noise[0])
    density, _, valid = core.component_densities(children, d_children, *guided[:4])
    assert bool(valid)
    independent = []
    for branch in range(3):
        means = guided[0][branch, :, 0].numpy()
        variances = guided[2][branch, :, 0, 0].numpy()
        residual = children.numpy()[:, 0, None] - means[None]
        gaussian = np.exp(-.5 * residual ** 2 / variances[None]) / np.sqrt(2 * np.pi * variances[None])
        independent.append(np.log(np.mean(gaussian, axis=1)))
    np.testing.assert_allclose(density, np.stack(independent, axis=-1), atol=2e-12)
    log_mixture = tf.reduce_logsumexp(density + tf.math.log(beta), axis=-1)
    assert float(tf.reduce_max(tf.exp(density[:, 0] - log_mixture))) <= 1 / .2 + 1e-12


def test_zero_future_guide_preserves_shared_consumer():
    spec = LGSSMSpec("diagonal_ar", 2)
    theta = tf.constant([.8, -.3, .6, .8], DTYPE)
    arguments = (theta, tf.constant([.2, -.1, .3, .4], DTYPE),
                 tf.constant([[.5, -.2], [.2, .8], [-.4, .3]], DTYPE),
                 *fixed_design(), tf.constant([.2, .3, .5], DTYPE))
    controls = core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5, correction_steps=2)
    plain = core.make_filter(spec, 24, 3, controls, jit_compile=False)(*arguments)
    guided = core.make_filter(spec, 24, 3, controls, jit_compile=False, backward_guide=future_guide("zero"))(*arguments)
    assert bool(plain["valid"]) and bool(guided["valid"])
    for key in ("value", "score", "particles", "particle_tangent"):
        np.testing.assert_allclose(guided[key], plain[key], rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("kind,correction", [("quadratic", 0), ("mixture", 0), ("mixture", 2)])
def test_complete_guided_mixture_total_derivatives(kind, correction):
    spec = LGSSMSpec("diagonal_ar", 2)
    theta = tf.constant([.8, -.3, .6, .8], DTYPE)
    controls = core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5, correction_steps=correction)
    program = core.make_filter(spec, 24, 3, controls, jit_compile=False, backward_guide=future_guide(kind))
    arguments = (tf.constant([[.5, -.2], [.2, .8], [-.4, .3]], DTYPE),
                 *fixed_design(), tf.constant([.2, .3, .5], DTYPE))
    delta = tf.constant(2e-5, DTYPE)
    for parameter in range(spec.parameter_count):
        direction = tf.one_hot(parameter, spec.parameter_count, dtype=DTYPE)
        result = program(theta, direction, *arguments)
        plus = program(theta + delta * direction, tf.zeros_like(theta), *arguments)
        minus = program(theta - delta * direction, tf.zeros_like(theta), *arguments)
        assert all(bool(item["valid"]) and int(item["steps_completed"]) == 3 for item in (result, plus, minus))
        np.testing.assert_allclose(result["score"], (plus["value"] - minus["value"]) / (2 * delta), rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(result["particle_tangent"], (plus["particles"] - minus["particles"]) / (2 * delta), rtol=1e-5, atol=1e-6)


def test_one_component_guide_agrees_with_quadratic_maps():
    spec = LGSSMSpec("diagonal_ar", 2)
    theta = tf.constant([.8, -.3, .6, .8], DTYPE)
    model, _ = spec.model(theta, tf.constant([.1, .2, .3, .4], DTYPE))
    ancestors, _, _ = fixed_design()
    guide = future_guide("quadratic")
    covariance = tf.linalg.inv(guide.precision[:1])
    center = tf.linalg.matvec(covariance, guide.linear[:1])
    mixture = FrozenMixtureBackwardGuide(tf.zeros([1, 1], DTYPE), center[:, None], covariance[:, None], tf.constant([True]))
    arguments = (model, theta, ancestors, .2 * ancestors, tf.constant([.5, -.2], DTYPE), core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5))
    quadratic_maps = core.build_maps(*arguments, backward_guide=guide, time=0)
    mixture_maps = core.build_maps(*arguments, backward_guide=mixture, time=0)
    for actual, expected in zip(mixture_maps[:12], quadratic_maps[:12]):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_guide_call_chain_and_invalid_precision(monkeypatch):
    calls = []
    original = core.build_maps

    def recorded(*arguments, **keywords):
        calls.append(keywords.get("backward_guide"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(core, "build_maps", recorded)
    spec = LGSSMSpec("diagonal_ar", 2)
    guide = FrozenQuadraticBackwardGuide(-tf.eye(2, batch_shape=[3], dtype=DTYPE), tf.zeros([3, 2], DTYPE))
    program = core.make_filter(spec, 24, 3, core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5), jit_compile=False, backward_guide=guide)
    theta = tf.constant([.8, -.3, .6, .8], DTYPE)
    result = program(theta, tf.ones_like(theta), tf.zeros([3, 2], DTYPE), *fixed_design(), tf.constant([.2, .3, .5], DTYPE))
    assert calls and all(item is guide for item in calls)
    assert not bool(result["valid"])
    assert int(result["steps_completed"]) == 1
    with pytest.raises(ValueError, match="shape"):
        core.make_filter(spec, 24, 2, core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5), backward_guide=guide)
