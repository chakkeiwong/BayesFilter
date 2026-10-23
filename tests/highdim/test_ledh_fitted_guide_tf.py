"""CPU-only independent-reference tests for the fitted scalar guide bridge."""

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_canonical_score_tf as shared_score
from bayesfilter.highdim.ledh_backward_guide_tf import FrozenQuadraticBackwardGuide
from bayesfilter.replication.kdm_iwsg_tf import GaussianMixtureIWSG
from bayesfilter.score_study.backward_guide_pilot_tf import (
    REFERENCE_CONTROLS,
    make_exact_future_information,
    make_guided_reference_kernel,
    make_scalar_fit,
    reference_theta,
    single_component_representation,
)
from bayesfilter.score_study.canonical_adapter_tf import make_canonical_kernel


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.9])
def test_actual_replicated_fit_matches_future_information(alpha):
    observations = tf.constant([0.2, -0.3, 0.7, 0.5], tf.float64)
    fit = make_scalar_fit(4, 64, alpha, jit_compile=False)
    result = fit(observations, tf.random.stateless_normal([4, 64], [16, 1016], dtype=tf.float64),
                 tf.random.stateless_uniform([4], [16, 1017], dtype=tf.float64))
    exact = make_exact_future_information(4, alpha, jit_compile=False)(observations)
    assert bool(result["valid"])
    assert not bool(tf.reduce_any(result["projected"]))
    np.testing.assert_allclose(result["precision"], exact[0], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result["linear"], exact[1], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(2.0 * result["policy"][:, 0], 4.0 + exact[0][:, 0, 0], atol=2e-10)
    np.testing.assert_allclose(-result["policy"][:, 1], 4.0 * observations + exact[1][:, 0], atol=2e-10)
    assert float(result["precision"][-1, 0, 0]) == 0.0
    assert float(result["linear"][-1, 0]) == 0.0
    if alpha == 0.0:
        np.testing.assert_array_equal(result["precision"], np.zeros([4, 1, 1]))
        np.testing.assert_array_equal(result["linear"], np.zeros([4, 1]))


@pytest.mark.parametrize("variance,quadratic", [(0.0, 1.0), (0.09, -8.0)])
def test_invalid_transition_integral_fails_closed(variance, quadratic):
    policy = tf.constant([[1.0, 0.0, 0.0], [quadratic, 0.0, 0.0]], tf.float64)
    guide = FrozenQuadraticBackwardGuide.from_scalar_controlled_smc_future(policy, 0.9, variance)
    assert not bool(guide.evaluate(0, tf.zeros([3, 1], tf.float64),
                                   tf.zeros([3, 1], tf.float64))[-1])


def test_single_component_matches_quadratic_and_replicated_kdm_density():
    guide = FrozenQuadraticBackwardGuide(
        tf.constant([[[2.5]], [[0.0]]], tf.float64),
        tf.constant([[0.8], [0.0]], tf.float64))
    mixture = single_component_representation(guide)
    anchors = tf.constant([[-0.4], [0.2], [0.8]], tf.float64)
    tangents = tf.ones_like(anchors) * 0.3
    for time in range(2):
        for actual, expected in zip(mixture.evaluate(time, anchors, tangents),
                                    guide.evaluate(time, anchors, tangents)):
            np.testing.assert_allclose(actual, expected, atol=2e-14)
    density = GaussianMixtureIWSG(
        batch_size=1, particles=1, samples=3, dimensions=1,
        density_weight_epsilon=0.0, injection_epsilon=0.0, jit_compile=False)
    output = density.evaluate(mixture.centers[:1], tf.ones([1, 1], tf.float64),
                              tf.sqrt(mixture.covariances[:1, 0, :, 0]), anchors[None])
    expected = -0.5 * ((anchors[:, 0] - 0.32)**2 * 2.5
                       + math.log(2.0 * math.pi / 2.5))
    np.testing.assert_allclose(output["log_density"][0], expected, atol=2e-14)
    assert bool(output["valid"])


def test_dynamic_consumer_shared_wiring_zero_limit_and_frozen_total_derivative(monkeypatch):
    calls = []
    original = shared_score._value_and_analytical_score_impl

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_score, "_value_and_analytical_score_impl", record)
    controls = {**REFERENCE_CONTROLS, "flow_substeps": 4}
    kernel = make_guided_reference_kernel(8, 2, controls, jit_compile=False,
                                         all_directions=False)
    theta = reference_theta(0.8)
    direction = tf.constant([0.2, -0.4, 0.1, 0.3, 0.0, 0.1], tf.float64)
    observations = tf.constant([[0.2], [-0.3]], tf.float64)
    initial_noise = tf.random.stateless_normal([8, 1], [16, 1018], dtype=tf.float64)
    noises = tf.random.stateless_normal([2, 8, 1], [16, 1019], dtype=tf.float64)
    precision = tf.constant([[[0.7]], [[0.0]]], tf.float64)
    linear = tf.constant([[-0.2], [0.0]], tf.float64)
    arguments = (direction, observations, initial_noise, noises, precision, linear)
    value, score, _diagnostics = kernel(theta, *arguments)
    step = 2e-5
    difference = (kernel(theta + step * direction, *arguments)[0]
                  - kernel(theta - step * direction, *arguments)[0]) / (2.0 * step)
    np.testing.assert_allclose(score, difference, rtol=2e-5, atol=2e-7)
    assert bool(tf.math.is_finite(value))
    assert calls and all(call["backward_guide"] is not None for call in calls)
    assert all(call["reset_policy"] == "contract_e" for call in calls)
    zero_result = kernel(theta, direction, observations, initial_noise, noises,
                         tf.zeros_like(precision), tf.zeros_like(linear))
    plain = make_canonical_kernel(1, 1, 8, 2, tuple(sorted(controls.items())),
                                  jit_compile=False)
    reset_design = tf.tile(tf.constant([[-1.0], [1.0]], tf.float64), [4, 1])
    baseline = plain(theta, direction, observations, initial_noise, noises, reset_design)
    np.testing.assert_allclose(zero_result[:2], baseline, rtol=1e-12, atol=1e-12)
    assert kernel.experimental_get_tracing_count() == 1
