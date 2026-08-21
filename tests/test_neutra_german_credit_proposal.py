"""Focused tests for German defensive pushed-proposal mechanics."""

from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_german_credit_proposal import (
    defensive_base_mixture_log_prob,
    reference_diagonal_mixture_log_prob,
    reference_marginal_unconstrained_parameters,
    sample_defensive_pushed_proposal,
    validate_defensive_base_mixture,
)
from bayesfilter.inference.neutra_german_credit_target import GermanCreditTargetSpec
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)


def _identity_initialized_transport() -> WeightedDenseIAFTransport:
    return WeightedDenseIAFTransport(
        WeightedNeuTraConfig(
            dimension=3,
            hidden_layers=(4,),
            stages=1,
            initialization_scale=0.0,
            initialization_seed=(1, 2),
        )
    )


def test_base_mixture_log_prob_matches_standard_normal_component() -> None:
    rows = tf.constant(((0.0, 0.0), (1.0, -2.0)), tf.float64)
    actual = defensive_base_mixture_log_prob(rows, (1.0, 2.0), (0.75, 0.25))
    components = []
    for scale, probability in ((1.0, 0.75), (2.0, 0.25)):
        components.append(
            math.log(probability)
            - math.log(2.0 * math.pi)
            - 2.0 * math.log(scale)
            - tf.reduce_sum(tf.square(rows), axis=1) / (2.0 * scale * scale)
        )
    expected = tf.reduce_logsumexp(tf.stack(components, axis=1), axis=1)
    tf.debugging.assert_near(actual, expected, atol=1.0e-12)


def test_identity_initialized_push_has_matching_latent_and_physical() -> None:
    transport = _identity_initialized_transport()
    physical, latent, log_proposal, labels = sample_defensive_pushed_proposal(
        transport, 64, (1.0, 1.5), (0.8, 0.2), seed=(3, 4)
    )
    tf.debugging.assert_near(physical, latent, atol=1.0e-12)
    tf.debugging.assert_near(
        log_proposal,
        defensive_base_mixture_log_prob(latent, (1.0, 1.5), (0.8, 0.2)),
        atol=1.0e-12,
    )
    assert labels.shape == (64,)


def test_proposal_validation_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        validate_defensive_base_mixture((1.0, 2.0), (0.8, 0.3))


def test_reference_marginal_transform_recovers_constrained_moments() -> None:
    spec = GermanCreditTargetSpec(
        name="german_gamma_scales2",
        observation_count=1,
        feature_count=1,
        dimension=3,
        design=((1.0,),),
        response=(1.0,),
        reference_mean=(0.5, 2.0, 3.0),
        reference_square=(1.25, 5.0, 12.0),
        data_path="data",
        data_sha256="0" * 64,
        reference_path="reference",
        reference_sha256="1" * 64,
    )
    location, standard_deviation = reference_marginal_unconstrained_parameters(spec)
    assert float(location[0].numpy()) == pytest.approx(0.5)
    assert float(tf.square(standard_deviation[0]).numpy()) == pytest.approx(1.0)
    for index, (mean, square) in enumerate(((2.0, 5.0), (3.0, 12.0)), start=1):
        mu = float(location[index].numpy())
        variance = float(tf.square(standard_deviation[index]).numpy())
        assert math.exp(mu + 0.5 * variance) == pytest.approx(mean)
        assert math.exp(2.0 * mu + 2.0 * variance) == pytest.approx(square)
    value = reference_diagonal_mixture_log_prob(
        tf.zeros((2, 3), tf.float64), spec
    )
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
