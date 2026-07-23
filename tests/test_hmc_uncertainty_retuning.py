from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from bayesfilter.inference import (
    HMCUncertaintyConfirmationAdmission,
    HMCUncertaintyRetuningPolicy,
    admit_hmc_uncertainty_nomination_for_confirmation,
    fixed_metric_candidate_record_from_payload,
    hmc_uncertainty_retuning_summary_from_payload,
    summarize_hmc_uncertainty_for_retuning,
)
from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    FixedMetricCandidateRecord,
    FixedMetricScreenRecord,
    FixedMetricScreenRequest,
    FixedMetricSearchLineage,
    fixed_metric_search_seed,
)
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    evaluate_hmc_acceptance_evidence,
)


L3_MEANS = (
    0.7575028289916661,
    0.7888629407557741,
    0.6756485243233381,
    0.789112437216353,
    0.8165639953296601,
    0.7151151499276303,
    0.7546791423797898,
    0.7889505996095132,
    0.7689819293951692,
    0.710027670353917,
    0.754775670321763,
    0.793948573806288,
)

L8_MEANS = (
    0.83511767313915,
    0.7720874844001084,
    0.7671110659737481,
    0.8020331470624646,
    0.8532858951732523,
    0.8216183681059347,
    0.7971514386657754,
    0.8154395642187293,
    0.8971957118120253,
    0.7483292373570458,
    0.8544225918519539,
    0.8545452695273488,
)

LINEAGE = FixedMetricSearchLineage(
    coordinate_signature="coordinate",
    metric_signature="metric",
    private_start_bank_content_signature="start-bank",
    common_state_signature="common-state",
)


def _candidate_payload(
    *,
    chain_means: tuple[float, ...] = L3_MEANS,
    promotion_veto: bool = False,
) -> dict:
    policy = HMCAcceptancePolicy()
    screens = []
    for replication_index in range(3):
        means = np.asarray(
            chain_means[4 * replication_index : 4 * (replication_index + 1)],
            dtype=float,
        )
        draws = np.arange(64, dtype=float)[:, None, None]
        chains = np.arange(4, dtype=float)[None, :, None]
        samples = (
            np.zeros((64, 4, 2), dtype=float)
            if promotion_veto and replication_index == 0
            else np.concatenate((draws + chains, 2.0 * draws + chains), axis=2)
        )
        evidence = evaluate_hmc_acceptance_evidence(
            samples=samples,
            log_accept_ratio=np.log(np.tile(means, (64, 1))),
            is_accepted=np.ones((64, 4), dtype=bool),
            policy=policy,
        )
        request = FixedMetricScreenRequest(
            round_index=0,
            stage="screen",
            num_leapfrog_steps=3,
            replication_index=replication_index,
            seed=fixed_metric_search_seed(
                (20260722, 8700),
                domain="round_0_screen_64",
                num_leapfrog_steps=3,
                replication_index=replication_index,
            ),
            tuned_step_size=0.28824213093792567,
            num_results=64,
            lineage=LINEAGE,
        )
        screens.append(
            FixedMetricScreenRecord(
                request=request,
                evidence_payload=evidence.payload(),
            )
        )
    candidate = FixedMetricCandidateRecord(
        round_index=0,
        num_leapfrog_steps=3,
        tune_seed=fixed_metric_search_seed(
            (20260722, 8700), domain="round_0_tune", num_leapfrog_steps=3
        ),
        tuned_step_size=0.28824213093792567,
        screens=tuple(screens),
    )
    return json.loads(json.dumps(candidate.payload()))


def test_one_sd_nomination_is_explicit_and_non_promoting() -> None:
    summary = summarize_hmc_uncertainty_for_retuning(L3_MEANS)
    assert summary.disposition == "provisional_nomination"
    assert summary.nominated is True
    assert summary.grand_mean == pytest.approx(np.mean(L3_MEANS))
    assert summary.sample_standard_deviation == pytest.approx(np.std(L3_MEANS, ddof=1))
    assert summary.standard_error == pytest.approx(
        summary.sample_standard_deviation / math.sqrt(12)
    )
    assert summary.interval == pytest.approx(
        (
            summary.grand_mean - summary.sample_standard_deviation,
            summary.grand_mean + summary.sample_standard_deviation,
        )
    )
    assert summary.payload()["nonclaims"][-1] == "no candidate ranking"
    assert summary.payload()["policy"]["promotion_effect"] == "none"


def test_high_mean_candidate_is_outside_practical_region() -> None:
    summary = summarize_hmc_uncertainty_for_retuning(L8_MEANS)
    # L=8 is rejected earlier because one chain-run mean exceeds the declared
    # repair region; this prevents uncertainty from masking a chain-local issue.
    assert summary.disposition == "repair_region_violation"
    assert summary.nominated is False


def test_hard_veto_wins_over_nomination() -> None:
    summary = summarize_hmc_uncertainty_for_retuning(L3_MEANS, hard_vetoes=("movement_gate_failed",))
    assert summary.disposition == "hard_veto"
    assert summary.nominated is False
    assert summary.hard_vetoes == ("movement_gate_failed",)


@pytest.mark.parametrize("values", [(), (0.7,) * 11, (0.7,) * 12 + (float("nan"),), (1.1,) * 12])
def test_invalid_summary_cannot_be_nominated(values) -> None:
    summary = summarize_hmc_uncertainty_for_retuning(values)
    assert summary.disposition == "invalid_summary"
    assert summary.nominated is False


def test_policy_payload_is_explicit_and_validates() -> None:
    policy = HMCUncertaintyRetuningPolicy(interval_multiplier=1.5)
    payload = policy.payload()
    assert payload["interval_multiplier"] == 1.5
    assert payload["expected_chain_run_count"] == 12
    with pytest.raises(ValueError):
        HMCUncertaintyRetuningPolicy(interval_multiplier=0.0)


def test_summary_payload_round_trip_and_schema_guard() -> None:
    summary = summarize_hmc_uncertainty_for_retuning(L3_MEANS)
    restored = hmc_uncertainty_retuning_summary_from_payload(summary.payload())
    assert restored == summary
    restored_json = hmc_uncertainty_retuning_summary_from_payload(
        json.loads(json.dumps(summary.payload()))
    )
    assert restored_json == summary
    bad = dict(summary.payload())
    bad["schema"] = "wrong"
    with pytest.raises(ValueError, match="schema"):
        hmc_uncertainty_retuning_summary_from_payload(bad)


def test_uncertainty_nominee_admission_is_lineage_bound_and_non_promoting() -> None:
    candidate_payload = _candidate_payload()
    stored_means = tuple(
        value
        for screen in candidate_payload["screens"]
        for value in screen["acceptance_evidence"]["chain_means"]
    )
    nomination = summarize_hmc_uncertainty_for_retuning(stored_means)

    admission = admit_hmc_uncertainty_nomination_for_confirmation(
        candidate_payload,
        nomination.payload(),
        source_artifact_sha256="a" * 64,
    )

    assert isinstance(admission, HMCUncertaintyConfirmationAdmission)
    assert admission.candidate.survivor is False
    assert admission.payload()["fixed_step_size"] == 0.28824213093792567
    assert admission.payload()["original_candidate_promotion_unchanged"] is True
    assert admission.payload()["retained_sampling_authorized"] is False
    assert len(admission.signature) == 64


def test_candidate_payload_parser_is_strict() -> None:
    payload = _candidate_payload()
    restored = fixed_metric_candidate_record_from_payload(payload)
    assert json.loads(json.dumps(restored.payload())) == payload

    tampered = json.loads(json.dumps(payload))
    tampered["survivor"] = True
    with pytest.raises(ValueError, match="inconsistent"):
        fixed_metric_candidate_record_from_payload(tampered)


def test_confirmation_admission_rejects_tampered_nomination_means() -> None:
    nomination = summarize_hmc_uncertainty_for_retuning(L3_MEANS).payload()
    tampered = json.loads(json.dumps(nomination))
    tampered["chain_run_means"][0] += 0.001
    with pytest.raises(ValueError, match="inconsistent"):
        admit_hmc_uncertainty_nomination_for_confirmation(
            _candidate_payload(), tampered, source_artifact_sha256="b" * 64
        )


def test_confirmation_admission_rejects_lineage_tampering() -> None:
    payload = _candidate_payload()
    payload["screens"][1]["lineage"]["metric_signature"] = "changed"
    nomination = summarize_hmc_uncertainty_for_retuning(L3_MEANS)
    with pytest.raises(ValueError, match="lineage mismatch"):
        admit_hmc_uncertainty_nomination_for_confirmation(
            payload, nomination.payload(), source_artifact_sha256="c" * 64
        )


def test_confirmation_admission_preserves_real_promotion_vetoes() -> None:
    candidate = _candidate_payload(promotion_veto=True)
    means = tuple(
        value
        for screen in candidate["screens"]
        for value in screen["acceptance_evidence"]["chain_means"]
    )
    nomination = summarize_hmc_uncertainty_for_retuning(means)
    with pytest.raises(ValueError, match="confirmation veto"):
        admit_hmc_uncertainty_nomination_for_confirmation(
            candidate, nomination.payload(), source_artifact_sha256="d" * 64
        )


def test_public_admission_import_does_not_load_tensorflow() -> None:
    output = subprocess.check_output(
        (
            sys.executable,
            "-c",
            "import sys; "
            "from bayesfilter.inference import "
            "admit_hmc_uncertainty_nomination_for_confirmation; "
            "print(any(n == 'tensorflow' or n.startswith('tensorflow.') "
            "for n in sys.modules))",
        ),
        text=True,
    ).strip()
    assert output == "False"
