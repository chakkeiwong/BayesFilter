from __future__ import annotations

import math

import pytest

from bayesfilter.inference import (
    FixedTransportCandidateSelectionConfig,
    select_fixed_transport_candidate_set,
)


def _observation(**updates):
    return {
        "all_finite": True,
        "target_status_valid": True,
        "all_chain_movement": True,
        "native_divergence_count": 0,
        "max_rhat": 1.005,
        "min_bulk_ess": 500.0,
        "min_tail_ess": 250.0,
        "max_mcse_sd_ratio": 0.05,
        **updates,
    }


def test_retains_all_viable_candidates_and_nominates_descriptively():
    calls = []

    def validate(candidate, rung):
        calls.append((candidate["candidate_id"], rung))
        return _observation()

    result = select_fixed_transport_candidate_set(
        [{"candidate_id": 1, "score": 2.0}, {"candidate_id": 2, "score": 1.0}],
        validate_rung=validate,
        score_candidate=lambda candidate, _: (candidate["score"],),
        score_metadata={"provenance": "test_fixture", "uncertainty_status": "not_estimated"},
    )
    assert calls == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert [row["candidate_id"] for row in result["viable_candidates"]] == [1, 2]
    assert result["selected_candidate"]["candidate_id"] == 2
    assert result["selected_candidate_index"] == 1
    assert result["statistical_ranking_supported"] is False
    assert result["score_metadata"]["provenance"] == "test_fixture"
    assert "all_candidate_validation_draws_discarded" not in result
    assert result["validation_draws_policy"].startswith("caller_managed")


@pytest.mark.parametrize("failed_rung", [0, 1])
def test_candidate_local_rejection_does_not_stop_other_candidates(failed_rung):
    calls = []

    def validate(candidate, rung):
        calls.append((candidate["candidate_id"], rung))
        return _observation(max_rhat=1.2 if candidate["candidate_id"] == 1 and rung == failed_rung else 1.005)

    result = select_fixed_transport_candidate_set(
        [{"candidate_id": 1}, {"candidate_id": 2}], validate_rung=validate,
    )
    assert calls == [(1, rung) for rung in range(failed_rung + 1)] + [(2, 0), (2, 1)]
    assert result["selected_candidate"] is None
    assert result["nomination_status"] == "nomination_not_requested"
    assert "rhat_screen_failed" in result["candidate_rows"][0]["hard_vetoes"]
    assert len(result["continuation_candidates"]) == 1


@pytest.mark.parametrize("updates,reason", [
    ({"target_status_valid": None}, "target_status_invalid"),
    ({"all_finite": False}, "nonfinite_candidate_observation"),
    ({"all_chain_movement": False}, "chain_movement_failed"),
    ({"native_divergence_count": 1}, "native_divergence_detected"),
    ({"native_divergence_count": -1}, "native_divergence_count_invalid"),
    ({"native_divergence_count": 0.5}, "native_divergence_count_invalid"),
    ({"native_divergence_count": True}, "native_divergence_count_invalid"),
    ({"native_divergence_status": "available", "native_divergence_count": None}, "native_divergence_count_missing"),
    ({"min_bulk_ess": 399}, "bulk_ess_screen_failed"),
    ({"min_tail_ess": 199}, "tail_ess_screen_failed"),
    ({"max_mcse_sd_ratio": 0.11}, "mcse_sd_screen_failed"),
    ({"max_mcse_sd_ratio": -1}, "max_mcse_sd_ratio_negative"),
    ({"max_rhat": math.nan}, "max_rhat_nonfinite_or_missing"),
])
def test_hard_screens_fail_closed(updates, reason):
    result = select_fixed_transport_candidate_set([{}], validate_rung=lambda *_: _observation(**updates))
    assert reason in result["candidate_rows"][0]["hard_vetoes"]
    assert result["nomination_status"] == "no_viable_candidate"


@pytest.mark.parametrize("rungs", [0, -1, 1.5, True, math.nan, math.inf])
def test_invalid_rung_count(rungs):
    with pytest.raises(ValueError, match="positive integer"):
        FixedTransportCandidateSelectionConfig(rungs=rungs)


def test_duplicate_ids_rejected_before_any_callback():
    calls = []
    with pytest.raises(ValueError, match="unique"):
        select_fixed_transport_candidate_set(
            [{"candidate_id": 1}, {"candidate_id": 1}],
            validate_rung=lambda *args: calls.append(args),
        )
    assert not calls


def test_callback_failure_is_not_candidate_rejection():
    def validate(*_):
        raise RuntimeError("runner_failed")

    with pytest.raises(RuntimeError, match="runner_failed"):
        select_fixed_transport_candidate_set([{}], validate_rung=validate)


@pytest.mark.parametrize("score", [(), (math.nan,), (math.inf,)])
def test_invalid_nomination_score(score):
    with pytest.raises(ValueError, match="finite nonempty"):
        select_fixed_transport_candidate_set(
            [{}], validate_rung=lambda *_: _observation(), score_candidate=lambda *_: score,
        )


def test_tensor_diagnostics_use_unranked_mean_ess():
    import tensorflow as tf
    from bayesfilter.inference import fixed_transport_candidate_diagnostics
    from bayesfilter.inference.hmc_convergence import _real_fft_cross_chain_ess, _split_chains

    samples = tf.exp(2.0 * tf.random.stateless_normal([500, 4, 2], [123, 456], dtype=tf.float64))
    observation = fixed_transport_candidate_diagnostics(
        samples, initial_state=tf.zeros([4, 2], tf.float64),
        mechanics={"log_accept_ratio_finite": True, "target_log_prob_finite": True,
                   "target_status_telemetry": {"all_status_valid": True}},
        config=FixedTransportCandidateSelectionConfig(), parameter_names=("z[0]", "z[1]"),
    )
    expected = tf.math.rsqrt(_real_fft_cross_chain_ess(_split_chains(samples)))
    tf.debugging.assert_near(tf.constant(observation["mean_mcse_sd_ratio"], tf.float64), expected)
    assert observation["max_mcse_sd_ratio"] != pytest.approx(
        1.0 / math.sqrt(observation["min_bulk_ess"]), rel=1e-4,
    )
    assert observation["target_status_valid"] is True
    assert observation["mcse_method"] == "unranked_split_chain_ess"


def test_missing_target_status_and_degenerate_draws_are_not_valid():
    import tensorflow as tf
    from bayesfilter.inference import fixed_transport_candidate_diagnostics

    observation = fixed_transport_candidate_diagnostics(
        tf.ones([32, 4, 1], tf.float64), initial_state=tf.zeros([4, 1], tf.float64),
        mechanics={"log_accept_ratio_finite": True, "target_log_prob_finite": True},
        config=FixedTransportCandidateSelectionConfig(), parameter_names=("z[0]",),
    )
    assert observation["target_status_valid"] is False
    assert not math.isfinite(observation["max_mcse_sd_ratio"])
