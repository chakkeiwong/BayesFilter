"""CPU-only independent references for the noncanonical KSC guide diagnostic."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide, FrozenQuadraticBackwardGuide)
from bayesfilter.highdim.sqmc_ksc_tf import (
    KSCSpec, MEANS, VARIANCES, WEIGHTS, enumeration_reference,
    grid_reference, observation_terms, persistence)
from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
from bayesfilter.score_study.ksc_guidance_diagnostic_tf import (
    CONDITIONS, future_guides, make_clouds)


@pytest.mark.parametrize("condition", tuple(CONDITIONS))
def test_future_mixture_gradient_and_moving_anchor_tangent(condition):
    theta = tf.constant([1.5, 0.0], tf.float64)
    observations = tf.constant(CONDITIONS[condition], tf.float64)
    guide = FrozenMixtureBackwardGuide(*future_guides(theta, observations)["mixture"])
    anchors = tf.constant([[-3.0], [0.0], [3.0]], tf.float64)
    tangent = tf.constant([[0.2], [-0.3], [0.4]], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(anchors)
        residual = (observations[1, 0] - persistence(theta) * anchors
                    - 2.0 * theta[1] - tf.constant(MEANS, tf.float64))
        variance = 1.0 + tf.constant(VARIANCES, tf.float64)
        log_terms = (tf.math.log(tf.constant(WEIGHTS, tf.float64))
                     - 0.5 * (tf.square(residual) / variance + tf.math.log(variance)))
        log_future = tf.reduce_sum(tf.reduce_logsumexp(log_terms, axis=1))
    spatial_gradient = tape.gradient(log_future, anchors)
    precision, linear, precision_tangent, linear_tangent, valid = guide.evaluate(
        tf.constant(0), anchors, tangent)
    assert bool(valid)
    np.testing.assert_allclose(linear - tf.linalg.matvec(precision, anchors),
                               spatial_gradient, atol=2e-13, rtol=2e-13)
    step = tf.constant(1e-5, tf.float64)
    plus = guide.evaluate(tf.constant(0), anchors + step * tangent, tangent)
    minus = guide.evaluate(tf.constant(0), anchors - step * tangent, tangent)
    np.testing.assert_allclose(precision_tangent, (plus[0] - minus[0]) / (2.0 * step),
                               atol=1e-10, rtol=1e-8)
    np.testing.assert_allclose(linear_tangent, (plus[1] - minus[1]) / (2.0 * step),
                               atol=1e-10, rtol=1e-8)
    terminal = guide.evaluate(tf.constant(1), anchors, tangent)
    assert bool(terminal[-1])
    for component in terminal[:-1]:
        np.testing.assert_array_equal(component, tf.zeros_like(component))


def test_shared_observation_callback_is_actual_mixture_and_total_tangent():
    spec = KSCSpec()
    theta = tf.constant([1.5, 0.0], tf.float64)
    direction = tf.constant([0.2, -0.3], tf.float64)
    particles = tf.constant([[-8.0], [0.0], [3.0]], tf.float64)
    tangent = tf.constant([[0.3], [-0.2], [0.1]], tf.float64)
    observation = tf.constant([0.2], tf.float64)
    model, _ = spec.model(theta, direction)
    actual, actual_tangent = core.observation_density(
        model, theta, particles, tangent, observation)
    reference, _ = observation_terms(theta, particles[:, 0], observation[0])
    np.testing.assert_allclose(actual, reference, atol=2e-13)
    step = tf.constant(1e-5, tf.float64)
    plus = observation_terms(theta + step * direction,
                             (particles + step * tangent)[:, 0], observation[0])[0]
    minus = observation_terms(theta - step * direction,
                              (particles - step * tangent)[:, 0], observation[0])[0]
    np.testing.assert_allclose(actual_tangent, (plus - minus) / (2.0 * step),
                               atol=2e-10, rtol=2e-8)


@pytest.mark.parametrize("family", ["gaussian", "mixture"])
def test_ksc_wrappers_use_shared_consumer_and_actual_pilot_density(monkeypatch, family):
    spec = KSCSpec()
    theta = spec.default_theta(tf.float64)
    observations = tf.constant(CONDITIONS["ordinary"], tf.float64)
    clouds = make_clouds(12, 617000)
    parameters = future_guides(theta, observations)[family]
    beta = tf.constant([0.1, 0.9, 0.0], tf.float64)
    controls = core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5, correction_steps=2)
    original = core.make_filter
    calls = []

    def observed(*arguments, **keywords):
        calls.append(keywords.get("backward_guide"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(core, "make_filter", observed)
    evaluate, pilot = make_diagnostic_programs(
        12, 2, controls, spec=spec,
        guide_family="mixture" if family == "mixture" else "quadratic",
        mixture_components=7 if family == "mixture" else None, jit_compile=False)
    result = evaluate(theta, observations, *clouds, beta, *parameters)
    guide_type = FrozenMixtureBackwardGuide if family == "mixture" else FrozenQuadraticBackwardGuide
    guide = guide_type(*parameters)
    direct = original(spec, 12, 2, controls, tf.float64, jit_compile=False,
                      backward_guide=guide)
    assert bool(result["valid"])
    for coordinate in range(2):
        reference = direct(theta, tf.one_hot(coordinate, 2, dtype=tf.float64),
                           observations, *clouds, beta)
        np.testing.assert_allclose(result["value"], reference["value"], atol=1e-12)
        np.testing.assert_allclose(result["score"][coordinate], reference["score"], atol=1e-12)
        np.testing.assert_allclose(result["direction_trace"][coordinate], reference["trace"], atol=1e-12)
    pilot_clouds = make_clouds(12, 617010)
    pilot_result = pilot(theta, observations, *clouds, beta, *parameters, *pilot_clouds[1:])
    assert bool(pilot_result["valid"])
    assert calls and all(isinstance(item, guide_type) for item in calls)
    model, _ = spec.model(theta, tf.zeros_like(theta))
    maps = core.build_maps(model, theta, result["ancestor_history"][0],
                           tf.zeros_like(clouds[0]), observations[0], controls,
                           backward_guide=guide, time=tf.constant(0))
    children, child_tangent = core.draw_from_maps(
        maps, tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)),
        pilot_clouds[2][0], pilot_clouds[1][0])
    density, _, valid = core.component_densities(children, child_tangent, *maps[:4])
    likelihood, _ = observation_terms(theta, children[:, 0], observations[0, 0])
    assert bool(valid)
    np.testing.assert_allclose(pilot_result["log_target"][0], likelihood + density[:, 0], atol=1e-12)
    if family == "mixture":
        step = tf.constant(5e-5, tf.float64)
        for coordinate in range(2):
            offset = step * tf.one_hot(coordinate, 2, dtype=tf.float64)
            plus = evaluate(theta + offset, observations, *clouds, beta, *parameters)
            minus = evaluate(theta - offset, observations, *clouds, beta, *parameters)
            difference = (plus["value"] - minus["value"]) / (2.0 * step)
            np.testing.assert_allclose(result["score"][coordinate], difference, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize("condition,horizon", [("ordinary", 1),
    ("ordinary", 2), ("negative_future_tail", 2), ("positive_future_tail", 2)])
def test_exact_branch_reference_matches_independent_grid(condition, horizon):
    theta = tf.constant([1.5, 0.0], tf.float64)
    observations = tf.constant(CONDITIONS[condition][:horizon], tf.float64)
    exact_value, exact_score = enumeration_reference(horizon)(theta, observations)
    grid_value, grid_score = grid_reference(nodes=1201, bound=48.0, jit_compile=False)(theta, observations)
    np.testing.assert_allclose(exact_value, grid_value, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(exact_score, grid_score, atol=1e-5, rtol=0.0)
