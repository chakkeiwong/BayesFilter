from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2w_payload():
    return {
        "importance_reference": {
            "mean_u_new": [0.1, -0.2, 0.3, -0.4],
        }
    }


def _phase2x_payload():
    return {
        "importance_reference": {
            "std_u_new": [2.0, 3.0, 4.5, 0.5],
        }
    }


def _phase2y_payload():
    return {
        "anchor_evaluation": {
            "rows": [
                {
                    "relation": "top_weight",
                    "u_new": [3.0, 0.0, 0.0, 0.0],
                },
                {
                    "relation": "top_weight",
                    "u_new": [0.0, -4.0, 0.0, 0.0],
                },
                {
                    "relation": "antithetic_partner",
                    "u_new": [-3.0, 0.0, 0.0, 0.0],
                },
            ]
        }
    }


def test_deterministic_counts_uses_stable_largest_fractional_remainder() -> None:
    harness = _load_harness()

    counts = harness.deterministic_counts([0.2, 0.2, 0.6], 11)

    np.testing.assert_array_equal(counts, [2, 2, 7])
    assert int(np.sum(counts)) == 11


def test_student_t_log_prob_matches_manual_one_dim_product() -> None:
    harness = _load_harness()
    sample = np.array([[1.0, 0.0, -1.0, 2.0]])
    center = np.zeros(4)
    scale = np.ones(4)
    df = 4.0

    actual = harness.independent_student_t_log_prob(sample, center, scale, df)[0]

    normalizer = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    expected = sum(
        normalizer - 0.5 * (df + 1.0) * math.log1p(float(value * value) / df)
        for value in sample[0]
    )
    np.testing.assert_allclose(actual, expected)


def test_student_t_mixture_log_prob_matches_manual_logsumexp() -> None:
    harness = _load_harness()
    proposal = harness.StudentTMixtureProposal(
        name="test",
        centers=np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        scales=np.ones((2, 4)),
        weights=np.array([0.25, 0.75]),
        df=4.0,
        seed=(1, 2),
        sample_count=8,
    )
    sample = np.array([[0.5, 0.0, 0.0, 0.0]])

    actual = harness.student_t_mixture_log_prob(sample, proposal)[0]

    terms = []
    for weight, center, scale in zip(
        proposal.weights,
        proposal.centers,
        proposal.scales,
        strict=True,
    ):
        terms.append(
            math.log(float(weight))
            + harness.independent_student_t_log_prob(sample, center, scale, proposal.df)[0]
        )
    expected = max(terms) + math.log(sum(math.exp(term - max(terms)) for term in terms))
    np.testing.assert_allclose(actual, expected)


def test_build_phase2z_proposals_locks_names_weights_scales_and_seeds() -> None:
    harness = _load_harness()

    proposals = harness.build_phase2z_proposals(
        _phase2w_payload(),
        _phase2x_payload(),
        _phase2y_payload(),
    )

    assert [proposal.name for proposal in proposals] == list(harness.PHASE2Z_CANDIDATES)
    assert [proposal.seed for proposal in proposals] == [
        (20260709, 6701),
        (20260709, 6702),
        (20260709, 6703),
        (20260709, 6704),
    ]
    np.testing.assert_allclose(proposals[0].scales[0], [2.0, 3.0, 4.0, 1.5])
    np.testing.assert_allclose(proposals[2].weights[:2], [0.25, 0.25])
    assert np.isclose(np.sum(proposals[2].weights), 1.0)
    assert proposals[2].centers.shape[0] == 4
    assert proposals[3].centers.shape[0] == 6


def test_sample_student_t_mixture_uses_intended_log_density_not_sample_fraction() -> None:
    harness = _load_harness()
    proposal = harness.StudentTMixtureProposal(
        name="test",
        centers=np.array([[0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]),
        scales=np.ones((2, 4)),
        weights=np.array([0.2, 0.8]),
        df=4.0,
        seed=(11, 12),
        sample_count=11,
    )

    generated = harness.sample_student_t_mixture(proposal)

    assert generated["generated"] is True
    np.testing.assert_array_equal(generated["component_counts"], [2, 9])
    samples = np.asarray(generated["samples"], dtype=float)
    log_prob = np.asarray(generated["proposal_log_prob"], dtype=float)
    replay = harness.student_t_mixture_log_prob(samples, proposal)
    np.testing.assert_allclose(log_prob, replay)


def test_candidate_gate_nominates_only_rows_passing_all_pilot_screens() -> None:
    harness = _load_harness()
    rows = [
        {
            "candidate_name": "good",
            "nominated_for_independent_replication": True,
            "importance_summary": {
                "ess": 300.0,
                "ess_ratio": 0.07,
                "weight_summary": {"max": 0.04},
            },
        },
        {
            "candidate_name": "bad",
            "nominated_for_independent_replication": False,
            "importance_summary": {
                "ess": 100.0,
                "ess_ratio": 0.02,
                "weight_summary": {"max": 0.2},
            },
        },
    ]

    gate = harness.evaluate_candidate_gate(rows)

    assert gate["nominated_candidate_count"] == 1
    assert gate["nominated_candidates"][0]["candidate_name"] == "good"
    assert gate["interpretation"] == "pilot nomination only; independent replication required"
