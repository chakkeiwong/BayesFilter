"""CPU-only independent identities and actual global/local consumer wiring."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.score_study.ksc_information_diagnostic_tf import (
    gaussian_product, make_information_probe, mixture_information)


def tensor(value):
    return tf.constant(value, tf.float64)


@pytest.mark.parametrize("centers", [[0.3], [-2.0, 2.0]])
def test_gaussian_product_includes_between_component_variance(centers):
    count = len(centers)
    result = gaussian_product(tensor(0.2), tensor(1.3), tensor([0.0] * count),
                              tensor(centers), tensor([0.4] * count))
    consumer = 1.0 / (1.0 / 1.3 + 1.0 / 0.4)
    assert float(result["within"]) == pytest.approx(consumer, abs=1e-14)
    assert float(result["variance"] - consumer) == pytest.approx(float(result["between"]), abs=1e-14)
    assert (float(result["between"]) > 0.0) == (count > 1)


def test_mixture_hessian_matches_independent_score_difference():
    anchors = tensor([-1.0, 0.0, 1.0])
    arguments = (tensor([0.2, -0.3]), tensor([-2.0, 2.0]), tensor([0.4, 0.7]))
    result = mixture_information(anchors, *arguments)
    step = tensor(1e-5)
    derivative = (mixture_information(anchors + step, *arguments)["score"]
                  - mixture_information(anchors - step, *arguments)["score"]) / (2.0 * step)
    assert float(tf.reduce_max(tf.abs(derivative + result["negative_hessian"]))) < 1e-7
    assert bool(tf.reduce_any(result["negative_hessian"] < 0.0))


def test_shared_consumer_calls_guide_at_both_anchor_sets(monkeypatch):
    calls = []
    original = core._backward_information

    def record(guide, time, anchors, tangents):
        calls.append(anchors.shape.as_list())
        return original(guide, time, anchors, tangents)

    monkeypatch.setattr(core, "_backward_information", record)
    probe = make_information_probe(2, core.ProposalControls(reset_steps=5))
    result = probe(tensor([1.5, 0.0]), tensor([[0.2], [-0.4]]),
                   tf.reshape(tf.linspace(tensor(-2.0), tensor(2.0), 32), [32, 1]),
                   tensor([[0.0, 0.0], [0.0, 0.0]]),
                   tensor([[[-2.0], [2.0]], [[0.0], [0.0]]]),
                   tensor([[[[0.4]], [[0.4]]], [[[1.0]], [[1.0]]]]),
                   tf.constant([True, False]))
    assert calls == [[1, 1], [32, 1]]
    assert bool(result["valid"])
    assert float(result["identity_relative_error"]) < 1e-12
    assert float(result["variance_ratio"]) > 1.0
    assert probe.experimental_get_tracing_count() == 1
