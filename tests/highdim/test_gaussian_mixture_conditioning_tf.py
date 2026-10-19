"""Independent CPU/FP64 references for optional product-moment conditioning."""

import os
from dataclasses import replace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim import ledh_backward_guide_tf as guides
from bayesfilter.highdim.gaussian_mixture_conditioning_tf import gaussian_mixture_product_moments
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec


def case():
    return tuple(tf.constant(value, tf.float64) for value in (
        [[.4, -.7], [-.2, .8]],
        [[[1.3, .25], [.25, .8]], [[.6, -.1], [-.1, 1.7]]],
        [[.2, -.3], [.1, .4]],
        [[[.1, -.03], [-.03, -.2]], [[-.1, .05], [.05, .3]]],
        [-.3, .4, -.1], [[-1., .3], [.7, -.8], [1.8, 1.2]],
        [[[.6, .1], [.1, .9]], [[1.4, -.2], [-.2, .7]], [[.9, .3], [.3, 1.2]]]))


def independent_product(mean, covariance, logits, centers, component_covariances):
    means, covariances, log_weights = [], [], []
    for center, component_covariance, logit in zip(centers, component_covariances, logits):
        combined = covariance + component_covariance
        gain = np.linalg.solve(combined, covariance).T
        means.append(mean + gain @ (center - mean))
        covariances.append(covariance - gain @ covariance)
        residual = mean - center
        log_weights.append(logit - .5 * (np.linalg.slogdet(combined)[1]
                                         + residual @ np.linalg.solve(combined, residual)))
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= weights.sum()
    output_mean = np.einsum("k,ki->i", weights, means)
    centered = np.asarray(means) - output_mean
    output_covariance = np.einsum("k,kij->ij", weights,
                                  covariances + centered[:, :, None] * centered[:, None, :])
    return output_mean, output_covariance


def test_batched_heteroscedastic_product_and_total_tangents():
    arguments = case()
    program = tf.function(gaussian_mixture_product_moments,
                          input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in arguments],
                          jit_compile=False)
    result = program(*arguments)
    assert bool(result[4]) and float(result[5]) > 0
    mean, covariance, d_mean, d_covariance, logits, centers, component_covariances = (
        value.numpy() for value in arguments)
    for batch in range(2):
        expected = independent_product(mean[batch], covariance[batch], logits, centers, component_covariances)
        for index in range(2):
            np.testing.assert_allclose(result[index][batch], expected[index], atol=2e-14)
        for step in (2e-5, 1e-5):
            plus = independent_product(mean[batch] + step * d_mean[batch],
                                       covariance[batch] + step * d_covariance[batch],
                                       logits, centers, component_covariances)
            minus = independent_product(mean[batch] - step * d_mean[batch],
                                        covariance[batch] - step * d_covariance[batch],
                                        logits, centers, component_covariances)
            for index in range(2):
                np.testing.assert_allclose(result[index + 2][batch],
                                           (plus[index] - minus[index]) / (2 * step), atol=2e-10)


@pytest.mark.parametrize("replacement", ([[1., 2.], [2., 1.]], [[1., .2], [.1, 1.]],
                                          [[float("nan"), 0.], [0., 1.]]))
def test_invalid_covariance_is_rejected(replacement):
    arguments = list(case())
    arguments[1] = tf.constant([replacement, [[1., 0.], [0., 1.]]], tf.float64)
    assert not bool(gaussian_mixture_product_moments(*arguments)[4])


def guide_fixture(active=True, components=3):
    arguments = case()
    return guides.FrozenMixtureBackwardGuide(
        tf.stack([arguments[4][:components]] * 2),
        tf.stack([arguments[5][:components]] * 2),
        tf.stack([arguments[6][:components]] * 2), tf.constant([active, False]))


def test_inactive_guide_returns_exact_inputs():
    mean, covariance, d_mean, d_covariance, *_ = case()
    result = guide_fixture(False).condition_gaussian(0, mean, covariance, d_mean, d_covariance)
    for actual, expected in zip(result[:4], (mean, covariance, d_mean, d_covariance)):
        np.testing.assert_array_equal(actual, expected)


def filter_case():
    theta = tf.constant([.8, -.3, .6, .8], tf.float64)
    clouds = (tf.random.stateless_normal([12, 2], [8218, 1], dtype=tf.float64),
              tf.random.stateless_normal([2, 12, 2], [8218, 2], dtype=tf.float64),
              tf.random.stateless_uniform([2, 12, 2], [8218, 3], dtype=tf.float64))
    observations = tf.constant([[.5, -.2], [-.4, .3]], tf.float64)
    return theta, clouds, observations


@pytest.mark.parametrize("components", (1, 3))
def test_shared_maps_preserve_identity_local_and_match_product(components):
    theta, clouds, observations = filter_case()
    model, _ = LGSSMSpec("diagonal_ar", 2).model(theta, tf.ones_like(theta))
    baseline = core.ProposalControls(reset_steps=5, correction_steps=2)
    repaired = replace(baseline, global_guide_conditioning="mixture_product_moments")
    arguments = (model, theta, clouds[0], tf.zeros_like(clouds[0]), observations[0])
    guide = guide_fixture(components=components)
    old = core.build_maps(*arguments, baseline, backward_guide=guide, time=0)
    new = core.build_maps(*arguments, repaired, backward_guide=guide, time=0)
    assert bool(old[12]) and bool(new[12])
    for index in range(8):
        for branch in (0, 2):
            np.testing.assert_array_equal(old[index][branch], new[index][branch])
    if components == 1:
        for index in range(8):
            np.testing.assert_allclose(old[index][1], new[index][1], atol=3e-14)
    else:
        assert np.max(np.abs(old[2][1].numpy() - new[2][1].numpy())) > .01
        plain = core.build_maps(*arguments, baseline)
        prior_means = plain[0][1].numpy()
        current_mean = prior_means.mean(axis=0)
        centered = prior_means - current_mean
        current_covariance = (plain[2][1].numpy().mean(axis=0)
                              + centered.T @ centered / len(centered))
        expected_mean, expected_covariance = independent_product(
            current_mean, current_covariance, guide.logits[0].numpy(),
            guide.centers[0].numpy(), guide.covariances[0].numpy())
        new_means = new[0][1].numpy()
        new_centered = new_means - new_means.mean(axis=0)
        actual_covariance = new[2][1].numpy().mean(axis=0) + new_centered.T @ new_centered / len(new_centered)
        np.testing.assert_allclose(new_means.mean(axis=0), expected_mean, atol=3e-14)
        np.testing.assert_allclose(actual_covariance, expected_covariance, atol=3e-14)


def test_actual_filter_wires_shared_helper_and_has_total_derivative(monkeypatch):
    calls = []
    original = guides.gaussian_mixture_product_moments

    def observed(*arguments):
        calls.append(tuple(arguments[0].shape))
        return original(*arguments)

    monkeypatch.setattr(guides, "gaussian_mixture_product_moments", observed)
    theta, clouds, observations = filter_case()
    controls = core.ProposalControls(reset_steps=5, correction_steps=2,
                                     global_guide_conditioning="mixture_product_moments")
    program = core.make_filter(LGSSMSpec("diagonal_ar", 2), 12, 2, controls,
                               backward_guide=guide_fixture(), jit_compile=False)
    direction = tf.constant([.2, -.1, .3, .4], tf.float64)
    arguments = (observations, *clouds, tf.constant([.1, .45, .45], tf.float64))
    result = program(theta, direction, *arguments)
    assert calls and set(calls) == {(1, 2)}
    assert bool(result["valid"])
    for step in (2e-5, 1e-5):
        plus = program(theta + step * direction, direction, *arguments)
        minus = program(theta - step * direction, direction, *arguments)
        assert bool(plus["valid"]) and bool(minus["valid"])
        np.testing.assert_allclose(result["score"], (plus["value"] - minus["value"]) / (2 * step),
                                   rtol=2e-6, atol=2e-7)


def test_product_mode_rejects_unsupported_guide_at_configuration():
    controls = core.ProposalControls(global_guide_conditioning="mixture_product_moments")
    with pytest.raises(ValueError, match="requires a mixture guide"):
        core.make_filter(LGSSMSpec("diagonal_ar", 2), 12, 2, controls)
