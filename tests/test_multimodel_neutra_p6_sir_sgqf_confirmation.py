from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from docs.benchmarks import (
    run_multimodel_neutra_p6_sir_sgqf_neutra_confirmation as campaign,
)


def test_physical_transform_matches_declared_sir_scales() -> None:
    source = tf.constant([[[0.0, 0.0, 0.0]]], tf.float64)

    physical = campaign._physical_parameters(tf, source)

    tf.debugging.assert_near(physical, [[[0.1, 18.0, 10.0]]], atol=0.0)


def test_identical_draws_pass_simultaneous_physical_mean_agreement() -> None:
    draws = tf.random.stateless_normal(
        (500, 4, 3), seed=(20260716, 32401), dtype=tf.float64
    ) * 0.02

    result = campaign._physical_mean_agreement(
        tf=tf,
        tfp=tfp,
        candidate_source_samples=draws,
        comparator_source_samples=draws,
    )

    assert result["passed"] is True
    assert result["supported_disagreement"] is False
    assert result["unresolved_precision"] is False
    assert len(result["parameter_rows"]) == 3


def test_terminal_classification_separates_disagreement_from_precision() -> None:
    sequential = {"hard_vetoes": (), "warmup_passed": True, "passed": False}
    disagreement = {
        "convergence": {"passed": True},
        "physical_mean_agreement": {
            "passed": False,
            "supported_disagreement": True,
        },
    }
    unresolved = {
        "convergence": {"passed": True},
        "physical_mean_agreement": {
            "passed": False,
            "supported_disagreement": False,
        },
    }

    assert campaign._classify_terminal(sequential, disagreement) == (
        "SAMPLER_BLOCKED_SAME_TARGET_MEAN_DISAGREEMENT"
    )
    assert campaign._classify_terminal(sequential, unresolved) == (
        "EVIDENCE_BLOCKED_AGREEMENT_PRECISION"
    )
