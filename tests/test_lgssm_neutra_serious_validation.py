from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.testing import lgssm_neutra_serious_validation_tf as campaign


def test_static_inputs_and_seed_ledger_are_pinned() -> None:
    inputs = campaign.validate_static_campaign_inputs()
    seed_check = campaign.validate_seed_ledger()

    assert inputs.center.shape == (18,)
    assert inputs.factor.shape == (18, 18)
    assert np.array_equal(inputs.center, inputs.truth)
    assert seed_check["passed"] is True
    assert seed_check["root_seed_count"] > 20


def test_campaign_contract_fixes_runtime_and_evidence_gates() -> None:
    payload = campaign.campaign_contract_payload()

    assert payload["target_signature"] == campaign.EXPECTED_TARGET_SIGNATURE
    assert payload["training"]["phase4_steps"] == 1000
    assert payload["tuning"]["verification_results_per_chain"] == 1000
    assert payload["serious"]["results_per_chain"] == 4000
    assert payload["serious"]["rhat_max"] == 1.01
    assert payload["runtime"]["worker_count"] == 2
    assert payload["runtime"]["chains_per_worker"] == 2


def test_common_probe_is_stable_and_nontrivial() -> None:
    first = campaign.common_probe_points()
    second = campaign.common_probe_points()

    assert first.shape == (4, 18)
    assert np.array_equal(first, second)
    assert np.any(first < 0.0) and np.any(first > 0.0)
    assert len(campaign.common_probe_hash()) == 64


def test_worker_trace_reduction_preserves_chain_axis() -> None:
    first = {
        "is_accepted": np.ones((3, 2), dtype=bool),
        "log_accept_ratio": np.zeros((3, 2)),
        "target_status_telemetry": {
            "status_code": np.zeros((3, 2), dtype=np.int32),
            "valid_pre_regularized_score": np.ones((3, 2), dtype=bool),
        },
    }
    second = {
        "is_accepted": np.zeros((3, 2), dtype=bool),
        "log_accept_ratio": -np.ones((3, 2)),
        "target_status_telemetry": {
            "status_code": np.zeros((3, 2), dtype=np.int32),
            "valid_pre_regularized_score": np.ones((3, 2), dtype=bool),
        },
    }

    combined = campaign._combine_worker_trace((first, second))

    assert combined["is_accepted"].shape == (3, 4)
    assert combined["log_accept_ratio"].shape == (3, 4)
    assert combined["target_status_telemetry"]["status_code"].shape == (3, 4)
    assert np.all(combined["is_accepted"][:, :2])
    assert not np.any(combined["is_accepted"][:, 2:])


def test_cross_device_probe_parity_passes_and_fails_closed() -> None:
    values = {
        "probe_hash": campaign.common_probe_hash(),
        "theta": np.zeros((4, 18)),
        "logdet": np.zeros(4),
        "value": np.zeros(4),
        "score": np.zeros((4, 18)),
    }
    passed = campaign._cross_device_probe_parity(values, values)
    assert passed["passed"] is True

    changed = dict(values)
    changed["score"] = np.full((4, 18), 1.0e-4)
    failed = campaign._cross_device_probe_parity(values, changed)
    assert failed["passed"] is False
    with pytest.raises(campaign.LGSSMNeuTraCampaignError, match="parity"):
        campaign._assert_cross_device_probe_parity(failed)


def test_posterior_summary_agreement_and_recovery_fixture() -> None:
    rng = np.random.default_rng(20260713)
    base = rng.normal(size=(4000, 4, 18))
    truth = np.zeros(18)
    names = tuple(f"p{index}" for index in range(18))

    result = campaign._serious_posterior_summaries(
        candidate_samples=base,
        comparator_samples=base.copy(),
        truth=truth,
        parameter_names=names,
    )

    assert result["posterior_agreement_passed"] is True
    assert result["max_posterior_agreement_combined_mcse"] == 0.0
    assert result["recovery_passed"] is True
    assert len(result["parameter_rows"]) == 18


def test_serious_health_does_not_call_missing_divergence_zero() -> None:
    shape = (5, 4, 18)
    trace = {
        "log_accept_ratio": np.zeros((5, 4)),
        "target_log_prob": np.zeros((5, 4)),
    }
    diagnostics = {
        "acceptance_rate": 0.7,
        "divergence_status": "not_exposed_by_kernel",
        "divergence_count": None,
        "target_status_telemetry": {
            "telemetry_failure_veto": False,
            "all_status_valid": True,
        },
    }

    result = campaign._serious_health_screen(
        samples=np.zeros(shape),
        raw_samples=np.zeros(shape),
        diagnostics=diagnostics,
        trace=trace,
    )

    assert result["passed"] is True
    assert result["divergence_count"] is None
    assert "not exposed" in result["native_divergence_interpretation"]


def test_phase5_repair_is_limited_to_finite_tuning_failures() -> None:
    class Result:
        hard_vetoes = ("verification_modern_rank_folded_rhat_failed",)
        fixed_grid_scale_selection_payload = {
            "attempts": (
                {
                    "probe_diagnostics": {
                        "samples_all_finite": True,
                        "log_accept_ratio_finite": True,
                        "target_log_prob_finite": True,
                        "target_status_telemetry": {
                            "telemetry_failure_veto": False,
                        },
                        "divergence_count": None,
                    }
                },
            )
        }

    assert campaign._phase5_repair_allowed(Result()) is True
    Result.hard_vetoes = ("verification_target_status_telemetry_failure",)
    assert campaign._phase5_repair_allowed(Result()) is False


def test_phase5_tuning_config_builds_fresh_modern_verifier() -> None:
    config = campaign._phase5_tuning_config(
        candidate_id="affine_control",
        screen_seed=(20260713, 2101),
        verification_seed=(20260713, 2201),
        scales=campaign.TUNING_PRIMARY_SCALES,
        output_filename="tuning_result.json",
        source_suffix="test",
    )

    assert config.chain_count == 4
    assert config.use_xla is True
    assert config.target_status_trace_policy == "per_chain_step"
    assert config.verification_num_results == 1000
    assert config.verification_seed_base != config.screen_seed_base
    assert config.require_modern_rank_normalized_verification is True


def test_score_parity_summary_checks_score_not_only_forward() -> None:
    reference = {
        "theta": np.zeros((2, 18)),
        "logdet": np.zeros(2),
        "value": np.zeros(2),
        "score": np.zeros((2, 18)),
    }
    exact = campaign._score_parity_summary(reference=reference, explicit=reference)
    assert exact["passed"] is True

    changed = dict(reference)
    changed["score"] = np.full((2, 18), 1.0e-4)
    mismatch = campaign._score_parity_summary(reference=reference, explicit=changed)
    assert mismatch["theta_max_abs"] == 0.0
    assert mismatch["score_max_abs"] == pytest.approx(1.0e-4)
    assert mismatch["passed"] is False
