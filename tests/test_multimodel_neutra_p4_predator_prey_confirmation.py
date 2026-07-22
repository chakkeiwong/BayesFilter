from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp
import pytest

from docs.benchmarks import (
    run_multimodel_neutra_p4_predator_prey_neutra_confirmation as campaign,
)


def test_identical_physical_draws_pass_simultaneous_mean_equivalence() -> None:
    draws = tf.random.stateless_normal(
        (500, 4, 6), seed=(20260716, 14001), dtype=tf.float64
    )

    result = campaign._physical_mean_agreement(
        tf=tf,
        tfp=tfp,
        candidate_source_samples=draws,
        comparator_source_samples=draws,
    )

    assert result["passed"] is True
    assert result["supported_disagreement"] is False
    assert result["unresolved_precision"] is False
    assert result["mean_mcse_definition"].endswith(
        "sqrt_split_chain_cross_chain_ess"
    )
    assert all(row["simultaneous_upper_bound"] <= row["practical_margin"] for row in result["parameter_rows"])


def test_terminal_classification_separates_disagreement_from_precision() -> None:
    sequential = {
        "hard_vetoes": (),
        "warmup_passed": True,
        "passed": False,
    }
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


def test_confirmation_claim_is_mean_level_and_training_hashes_are_frozen() -> None:
    assert campaign.NONCLAIMS[0] == (
        "same-target physical-posterior-mean agreement only"
    )
    assert set(campaign.EXPECTED_TRAINING_RESULT_SHA256) == set(campaign.CELLS)
    assert all(
        len(value) == 64
        for value in campaign.EXPECTED_TRAINING_RESULT_SHA256.values()
    )


def test_training_hash_normalization_accepts_only_equivalent_sha256_forms() -> None:
    digest = "a" * 64

    assert campaign._bare_sha256(digest, "bare") == digest
    assert campaign._bare_sha256(f"sha256:{digest}", "prefixed") == digest
    with pytest.raises(campaign.P4NeuTraConfirmationError, match="SHA-256"):
        campaign._bare_sha256("sha256:not-the-same-digest", "invalid")


def test_probe_sources_are_hash_bound_and_only_order_candidates() -> None:
    for cell, expected_target in (
        ("PP-UKF", "036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30"),
        ("PP-SGQF", "8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad"),
    ):
        rows, reference = campaign._load_probe_source(cell, expected_target)
        ordered = campaign.tuning_hmc._ordered_probe_candidates(rows)

        assert len(rows) == len(campaign.STEP_SIZES)
        assert reference["scientific_role"] == "hash_verified_candidate_ordering_only"
        assert reference["old_warmup_and_retained_samples_reused"] is False
        assert ordered[0]["step_size"] == 0.2


def test_repair_seeds_are_fresh_and_stage_disjoint() -> None:
    all_roots = []
    for cell in campaign.CELLS:
        roots = (
            campaign.TUNING_VERIFICATION_SEEDS[cell],
            campaign.WARMUP_SEEDS[cell],
            campaign.RETAINED_SEEDS[cell],
        )
        assert len(set(roots)) == 3
        assert all(seed[0] == 20260716 for seed in roots)
        all_roots.extend(roots)

    assert len(set(all_roots)) == len(all_roots)
