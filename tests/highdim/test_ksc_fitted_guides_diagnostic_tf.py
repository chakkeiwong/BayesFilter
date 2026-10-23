"""CPU FP64/no-XLA references for frozen KSC fits, with GPUs intentionally hidden."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import (
    FrozenMixtureBackwardGuide, FrozenQuadraticBackwardGuide)
from bayesfilter.highdim.sqmc_ksc_tf import (
    KSCSpec, MEANS, VARIANCES, WEIGHTS, observation_terms, persistence)
from bayesfilter.replication.controlled_smc_tf import ScalarGaussianControlledSMC
from bayesfilter.score_study import ksc_fitted_guides_diagnostic_tf as fitting
from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs
from bayesfilter.score_study.ksc_guidance_diagnostic_tf import CONDITIONS, make_clouds


def fit_inputs(passes=1, particles=64):
    normal = tf.random.stateless_normal([passes, 2, particles], [641701, 1], dtype=tf.float64)
    uniform = tf.random.stateless_uniform([passes, 2], [641701, 2], dtype=tf.float64)
    return normal, uniform


def kernel_inputs(centers=64, replicates=16):
    uniform = tf.random.stateless_uniform([centers], [641702, 1], dtype=tf.float64)
    design = tf.random.stateless_normal([centers], [641702, 2], dtype=tf.float64)
    transition = tf.random.stateless_normal([centers, replicates], [641702, 3], dtype=tf.float64)
    return uniform, design, transition


def test_fitting_observation_density_supports_matrix_of_states():
    theta = tf.constant([1.5, 0.2], tf.float64)
    states = tf.reshape(tf.linspace(tf.constant(-9.0, tf.float64), 4.0, 12), [4, 3])
    actual = fitting.ksc_observation_log_density(tf.constant(-0.4, tf.float64), states, theta[1])
    expected, _ = observation_terms(theta, tf.reshape(states, [-1]), tf.constant(-0.4, tf.float64))
    np.testing.assert_allclose(actual, tf.reshape(expected, [4, 3]), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("passes", [1, 2])
@pytest.mark.parametrize("condition", tuple(CONDITIONS))
def test_quadratic_fit_matches_replicated_engine_and_once_integrated_future(condition, passes):
    theta = tf.constant([1.5, 0.0], tf.float64)
    gamma = float(persistence(theta))
    observations = tf.reshape(tf.constant(CONDITIONS[condition], tf.float64), [2])
    normal, uniform = fit_inputs(passes)
    actual = fitting.make_ksc_quadratic_fit(theta, 64, passes, jit_compile=False)(
        observations, normal, uniform)
    engine = ScalarGaussianControlledSMC(
        observations=2, particles=64, alpha=gamma, process_variance=1.0,
        initial_mean=0.0, initial_variance=gamma ** 2 + 1.0,
        observation_log_density=lambda observation, states: fitting.ksc_observation_log_density(
            observation, states, theta[1]),
        precision_policy="source_scalar_projection", jit_compile=False)
    policy = tf.zeros([2, 3], tf.float64)
    diagnostics = []
    forwards = []
    for iteration in range(passes):
        forward = engine.forward(policy, observations, normal[iteration], uniform[iteration])
        fitted = engine.fit(policy, forward["x"], forward["log_incremental_weight"])
        policy = fitted["policy"]
        forwards.append(forward)
        diagnostics.append(fitted)
    np.testing.assert_allclose(actual["policy"], policy, atol=2e-13, rtol=2e-13)
    for field in ("gram_eigenvalue_ratio", "max_residual", "projected"):
        np.testing.assert_allclose(actual[field], tf.stack([item[field] for item in diagnostics]),
                                   atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(actual["pilot_ess"], tf.stack([item["ess"] for item in forwards]),
                               atol=2e-13, rtol=2e-13)
    np.testing.assert_allclose(actual["pilot_logZ"], tf.stack([item["logZ"] for item in forwards]),
                               atol=2e-13, rtol=2e-13)
    denominator = 1.0 + 2.0 * policy[1, 0]
    expected_precision = 2.0 * gamma ** 2 * policy[1, 0] / denominator
    expected_linear = -gamma * policy[1, 1] / denominator
    np.testing.assert_allclose(actual["precision"][:, 0, 0], [expected_precision, 0.0], atol=2e-13)
    np.testing.assert_allclose(actual["linear"][:, 0], [expected_linear, 0.0], atol=2e-13)
    assert bool(actual["fit_valid"]) == all(
        bool(item["valid"]) for item in forwards + diagnostics)
    assert bool(actual["guide_eligible"]) == (bool(actual["fit_valid"]) and float(expected_precision) >= 0.0)
    assert bool(actual["density_eligible"]) == (bool(actual["guide_eligible"]) and float(expected_precision) > 0.0)
    np.testing.assert_allclose(actual["source_projection_offset"], 0.001, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("condition", tuple(CONDITIONS))
def test_positive_kernel_fit_matches_independent_inverse_design_reference(condition):
    theta = tf.constant([1.5, 0.2], tf.float64)
    observations = tf.reshape(tf.constant(CONDITIONS[condition], tf.float64), [2])
    inputs = kernel_inputs()
    actual = fitting.make_ksc_kernel_fit(theta, 64, 16, 1.0, jit_compile=False)(observations, *inputs)
    gamma = float(persistence(theta))
    component_uniform, design_normal, transition_normal = [item.numpy() for item in inputs]
    weights, means, variances = np.array(WEIGHTS), np.array(MEANS), np.array(VARIANCES)
    broad_mean = (float(observations[1]) - 2.0 * float(theta[1]) - np.sum(weights * means)) / gamma
    broad_variance = 4.0 * np.max(1.0 + variances) / gamma ** 2
    predictive_variance = gamma ** 2 + 1.0
    nodes = np.where(component_uniform < 0.5, np.sqrt(predictive_variance) * design_normal,
                     broad_mean + np.sqrt(broad_variance) * design_normal)
    log_design = np.logaddexp(
        -0.5 * (nodes ** 2 / predictive_variance + np.log(2.0 * np.pi * predictive_variance)),
        -0.5 * ((nodes - broad_mean) ** 2 / broad_variance + np.log(2.0 * np.pi * broad_variance))) - np.log(2.0)
    future_states = gamma * nodes[:, None] + transition_normal
    residual = float(observations[1]) - future_states[..., None] - 2.0 * float(theta[1]) - means
    log_likelihood = np.logaddexp.reduce(
        np.log(weights) - 0.5 * (residual ** 2 / variances + np.log(2.0 * np.pi * variances)), axis=-1)
    log_future = np.logaddexp.reduce(log_likelihood, axis=1) - np.log(16.0)
    log_unnormalized = log_future - log_design
    log_normalizer = np.logaddexp.reduce(log_unnormalized)
    log_weights = log_unnormalized - log_normalizer
    normalized = np.exp(log_weights)
    mean = np.sum(normalized * nodes)
    variance = np.sum(normalized * (nodes - mean) ** 2)
    effective_size = 1.0 / np.sum(normalized ** 2)
    bandwidth = np.sqrt(variance) * effective_size ** -0.2
    for field, expected in (
        ("nodes", nodes), ("log_design_density", log_design),
        ("log_future_estimate", log_future), ("log_weights", log_weights),
        ("log_normalization_estimate", log_normalizer - np.log(64.0)),
        ("raw_variance", variance), ("ess", effective_size),
        ("max_weight", np.max(normalized)), ("bandwidth", bandwidth)):
        np.testing.assert_allclose(actual[field], expected, atol=5e-13, rtol=2e-13)
    np.testing.assert_allclose(actual["centers"][0, :, 0], nodes, atol=2e-13)
    np.testing.assert_allclose(actual["covariances"][0, :, 0, 0], bandwidth ** 2, atol=2e-13)
    np.testing.assert_allclose(actual["logits"][0], log_weights, atol=5e-13)
    np.testing.assert_array_equal(actual["active"], [True, False])
    incorrect = np.exp(log_future - np.logaddexp.reduce(log_future))
    assert np.max(np.abs(incorrect - normalized)) > 0.001
    assert bool(actual["fit_valid"]) and bool(actual["guide_eligible"]) and bool(actual["density_eligible"])


@pytest.mark.parametrize("coefficient", [-0.1, 0.0, 0.2])
def test_quadratic_eligibility_rejects_negative_precision_without_clipping(monkeypatch, coefficient):
    class EngineFixture:
        def __init__(self, **settings):
            self.particles = settings["particles"]

        def forward(self, policy, observations, normal, uniform):
            return {"x": normal, "log_incremental_weight": tf.zeros_like(normal),
                    "valid": tf.constant(True), "ess": tf.fill([2], tf.cast(self.particles, tf.float64)),
                    "logZ": tf.zeros([2], tf.float64)}

        def fit(self, policy, states, log_weights):
            return {"policy": tf.constant([[0.0, 0.0, 0.0], [coefficient, 0.0, 0.0]], tf.float64),
                    "valid": tf.constant(True), "gram_eigenvalue_ratio": tf.ones([2], tf.float64),
                    "max_residual": tf.zeros([2], tf.float64), "projected": tf.zeros([2], tf.bool)}

    monkeypatch.setattr(fitting, "ScalarGaussianControlledSMC", EngineFixture)
    theta = tf.constant([1.5, 0.0], tf.float64)
    fitted = fitting.make_ksc_quadratic_fit(theta, 64, 1, jit_compile=False)(
        tf.constant([0.2, -0.4], tf.float64), *fit_inputs())
    assert bool(fitted["fit_valid"])
    assert bool(fitted["guide_eligible"]) == (coefficient >= 0.0)
    assert bool(fitted["density_eligible"]) == (coefficient > 0.0)
    expected = 2.0 * float(persistence(theta)) ** 2 * coefficient / (1.0 + 2.0 * coefficient)
    np.testing.assert_allclose(fitted["precision"][0, 0, 0], expected, atol=2e-14)


@pytest.mark.parametrize("factor", [0.0, -1.0, float("inf"), float("nan")])
def test_kernel_fit_rejects_invalid_bandwidth_factor(factor):
    with pytest.raises(ValueError):
        fitting.make_ksc_kernel_fit(tf.constant([1.5, 0.0], tf.float64), 64, 16, factor, jit_compile=False)


def test_kernel_fit_rejects_degenerate_design_without_hidden_ridge():
    fit = fitting.make_ksc_kernel_fit(tf.constant([1.5, 0.0], tf.float64), 64, 16, 1.0, jit_compile=False)
    fitted = fit(tf.constant([0.2, -0.4], tf.float64), tf.zeros([64], tf.float64),
                 tf.zeros([64], tf.float64), tf.zeros([64, 16], tf.float64))
    assert float(fitted["raw_variance"]) == 0.0
    assert float(fitted["bandwidth"]) == 0.0
    assert not bool(fitted["fit_valid"])
    assert not bool(fitted["guide_eligible"])


@pytest.mark.parametrize("bandwidth_factor", [1e-200, 1e200])
def test_kernel_fit_rejects_unrepresentable_kernel_variance(bandwidth_factor):
    fit = fitting.make_ksc_kernel_fit(
        tf.constant([1.5, 0.0], tf.float64), 64, 16, bandwidth_factor, jit_compile=False)
    fitted = fit(tf.constant([0.2, -0.4], tf.float64), *kernel_inputs())
    assert not bool(fitted["fit_valid"])
    assert not bool(fitted["guide_eligible"])


@pytest.mark.parametrize("bad_uniform", [-0.1, 1.0, float("nan")])
def test_kernel_fit_rejects_invalid_component_uniform(bad_uniform):
    inputs = kernel_inputs()
    fit = fitting.make_ksc_kernel_fit(tf.constant([1.5, 0.0], tf.float64), 64, 16, 1.0, jit_compile=False)
    fitted = fit(tf.constant([0.2, -0.4], tf.float64), tf.fill([64], tf.constant(bad_uniform, tf.float64)),
                 *inputs[1:])
    assert not bool(fitted["fit_valid"])
    assert not bool(fitted["guide_eligible"])


@pytest.mark.parametrize("family", ["quadratic", "mixture"])
def test_frozen_fits_use_shared_consumer_with_total_derivatives_and_physical_target(monkeypatch, family):
    spec = KSCSpec()
    theta = tf.constant([1.5, 0.0], tf.float64)
    observations = tf.constant(CONDITIONS["ordinary"], tf.float64)
    if family == "quadratic":
        fitted = fitting.make_ksc_quadratic_fit(theta, 64, 1, jit_compile=False)(
            tf.reshape(observations, [2]), *fit_inputs())
        parameters = fitted["precision"], fitted["linear"]
        guide_type = FrozenQuadraticBackwardGuide
    else:
        fitted = fitting.make_ksc_kernel_fit(theta, 64, 16, 1.0, jit_compile=False)(
            tf.reshape(observations, [2]), *kernel_inputs())
        parameters = tuple(fitted[field] for field in ("logits", "centers", "covariances", "active"))
        guide_type = FrozenMixtureBackwardGuide
    assert bool(fitted["guide_eligible"])
    clouds = make_clouds(12, 641703)
    beta = tf.constant([0.1, 0.9, 0.0], tf.float64)
    controls = core.ProposalControls(reset_steps=5, correction_steps=2)
    original = core.make_filter
    calls = []

    def observed(*arguments, **keywords):
        calls.append(keywords.get("backward_guide"))
        return original(*arguments, **keywords)

    monkeypatch.setattr(core, "make_filter", observed)
    evaluate, pilot = make_diagnostic_programs(
        12, 2, controls, spec=spec, guide_family=family,
        mixture_components=64 if family == "mixture" else None, jit_compile=False)
    result = evaluate(theta, observations, *clouds, beta, *parameters)
    guide = guide_type(*parameters)
    direct = original(spec, 12, 2, controls, tf.float64, jit_compile=False, backward_guide=guide)
    assert bool(result["valid"])
    for coordinate in range(2):
        reference = direct(theta, tf.one_hot(coordinate, 2, dtype=tf.float64), observations, *clouds, beta)
        np.testing.assert_allclose(result["value"], reference["value"], atol=1e-12)
        np.testing.assert_allclose(result["score"][coordinate], reference["score"], atol=1e-12)
        np.testing.assert_allclose(result["direction_trace"][coordinate], reference["trace"], atol=1e-12)
        offset = tf.constant(5e-5, tf.float64) * tf.one_hot(coordinate, 2, dtype=tf.float64)
        plus = evaluate(theta + offset, observations, *clouds, beta, *parameters)
        minus = evaluate(theta - offset, observations, *clouds, beta, *parameters)
        assert bool(plus["valid"]) and bool(minus["valid"])
        difference = (plus["value"] - minus["value"]) / tf.constant(1e-4, tf.float64)
        np.testing.assert_allclose(result["score"][coordinate], difference, atol=2e-7, rtol=2e-5)
    pilot_clouds = make_clouds(12, 641704)
    pilot_result = pilot(theta, observations, *clouds, beta, *parameters, *pilot_clouds[1:])
    assert bool(pilot_result["valid"])
    assert calls and all(isinstance(item, guide_type) for item in calls)
    model, _ = spec.model(theta, tf.zeros_like(theta))
    maps = core.build_maps(model, theta, result["ancestor_history"][0], tf.zeros_like(clouds[0]),
                           observations[0], controls, backward_guide=guide, time=tf.constant(0))
    children, child_tangent = core.draw_from_maps(
        maps, tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)), pilot_clouds[2][0], pilot_clouds[1][0])
    density, _, valid = core.component_densities(children, child_tangent, *maps[:4])
    likelihood, _ = observation_terms(theta, children[:, 0], observations[0, 0])
    assert bool(valid)
    np.testing.assert_allclose(pilot_result["log_target"][0], likelihood + density[:, 0], atol=1e-12)
    assert evaluate.experimental_get_tracing_count() == 1
