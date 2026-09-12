from __future__ import annotations

from pathlib import Path

import pytest

from bayesfilter.runtime.campaign_budget_ledger import (
    CampaignBudgetLedger,
    CampaignBudgetLedgerError,
)


def _create(tmp_path: Path) -> CampaignBudgetLedger:
    return CampaignBudgetLedger.create(
        tmp_path / "campaign_budget_ledger.json",
        campaign_id="test-campaign",
        total_budget_seconds=100.0,
        source_hash="source-v1",
        plan_hash="plan-v1",
        claim_boundary="readiness-only",
        initial_consumed_seconds=10.0,
    )


def test_ledger_reserves_and_settles_budget_durably(tmp_path: Path) -> None:
    ledger = _create(tmp_path)
    ledger.start_attempt(
        attempt_id="attempt-1",
        output_root=tmp_path / "attempt-1",
        seed_namespace={"arm": "factor"},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    ledger.reserve_chunk(
        attempt_id="attempt-1",
        arm="factor",
        chunk_index=0,
        reserve_seconds=20.0,
        output_root=tmp_path / "attempt-1",
        seed_namespace={"chunk": 0},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    ledger.reserve_chunk(
        attempt_id="attempt-1",
        arm="factor",
        chunk_index=1,
        reserve_seconds=30.0,
        output_root=tmp_path / "attempt-1",
        seed_namespace={"chunk": 1},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )

    assert ledger.remaining_seconds() == pytest.approx(40.0)
    settled = ledger.settle_arm(
        attempt_id="attempt-1",
        arm="factor",
        measured_seconds=35.0,
        status="completed",
    )
    assert settled["released_seconds"] == pytest.approx(15.0)
    assert ledger.remaining_seconds() == pytest.approx(55.0)
    ledger.finish_attempt(attempt_id="attempt-1", status="completed")

    payload = ledger.read()
    assert payload["consumed_seconds"] == pytest.approx(45.0)
    assert payload["released_seconds"] == pytest.approx(15.0)
    assert payload["attempts"][0]["status"] == "completed"
    assert ledger.checksum()


def test_ledger_allows_released_budget_to_be_reused(tmp_path: Path) -> None:
    ledger = CampaignBudgetLedger.create(
        tmp_path / "campaign_budget_ledger.json",
        campaign_id="test-campaign",
        total_budget_seconds=100.0,
        source_hash="source-v1",
        plan_hash="plan-v1",
        claim_boundary="readiness-only",
    )

    for attempt_index in range(2):
        attempt_id = f"attempt-{attempt_index}"
        output_root = tmp_path / attempt_id
        ledger.start_attempt(
            attempt_id=attempt_id,
            output_root=output_root,
            seed_namespace={"attempt": attempt_index},
            source_hash="source-v1",
            plan_hash="plan-v1",
        )
        ledger.reserve_chunk(
            attempt_id=attempt_id,
            arm="factor",
            chunk_index=0,
            reserve_seconds=80.0,
            output_root=output_root,
            seed_namespace={"attempt": attempt_index},
            source_hash="source-v1",
            plan_hash="plan-v1",
        )
        ledger.settle_arm(
            attempt_id=attempt_id,
            arm="factor",
            measured_seconds=10.0,
            status="completed",
        )
        ledger.finish_attempt(attempt_id=attempt_id, status="completed")

    payload = ledger.read()
    assert payload["consumed_seconds"] == pytest.approx(20.0)
    assert payload["released_seconds"] == pytest.approx(140.0)
    assert ledger.remaining_seconds() == pytest.approx(80.0)


def test_ledger_corrects_a_conservative_settlement_from_a_durable_receipt(tmp_path: Path) -> None:
    ledger = _create(tmp_path)
    output_root = tmp_path / "attempt-1"
    ledger.start_attempt(
        attempt_id="attempt-1",
        output_root=output_root,
        seed_namespace={},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    ledger.reserve_chunk(
        attempt_id="attempt-1",
        arm="strict",
        chunk_index=0,
        reserve_seconds=80.0,
        output_root=output_root,
        seed_namespace={},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    ledger.settle_arm(
        attempt_id="attempt-1",
        arm="strict",
        measured_seconds=80.0,
        status="conservative_unobserved_worker_bound",
    )
    corrected = ledger.correct_settled_arm(
        attempt_id="attempt-1",
        arm="strict",
        measured_seconds=12.0,
        status="recovered_observed_worker_receipt",
        previous_status="conservative_unobserved_worker_bound",
        repair="durable worker receipt verified",
    )
    ledger.finish_attempt(attempt_id="attempt-1", status="completed_reconciled")

    assert corrected["measured_seconds"] == pytest.approx(12.0)
    assert ledger.remaining_seconds() == pytest.approx(78.0)
    payload = ledger.read()
    assert payload["consumed_seconds"] == pytest.approx(22.0)
    assert payload["attempts"][0]["arms"][0]["status"] == "recovered_observed_worker_receipt"


def test_ledger_fails_closed_on_budget_or_binding_drift(tmp_path: Path) -> None:
    ledger = _create(tmp_path)
    ledger.start_attempt(
        attempt_id="attempt-1",
        output_root=tmp_path / "attempt-1",
        seed_namespace={},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    with pytest.raises(CampaignBudgetLedgerError, match="plan hash changed"):
        ledger.reserve_chunk(
            attempt_id="attempt-1",
            arm="factor",
            chunk_index=0,
            reserve_seconds=1.0,
            output_root=tmp_path / "attempt-1",
            seed_namespace={},
            source_hash="source-v1",
            plan_hash="plan-v2",
        )
    with pytest.raises(CampaignBudgetLedgerError, match="exceeds remaining"):
        ledger.reserve_chunk(
            attempt_id="attempt-1",
            arm="factor",
            chunk_index=0,
            reserve_seconds=91.0,
            output_root=tmp_path / "attempt-1",
            seed_namespace={},
            source_hash="source-v1",
            plan_hash="plan-v1",
        )


def test_ledger_rejects_output_root_reuse(tmp_path: Path) -> None:
    ledger = _create(tmp_path)
    ledger.start_attempt(
        attempt_id="attempt-1",
        output_root=tmp_path / "attempt-1",
        seed_namespace={},
        source_hash="source-v1",
        plan_hash="plan-v1",
    )
    with pytest.raises(CampaignBudgetLedgerError, match="attempt already exists"):
        ledger.start_attempt(
            attempt_id="attempt-1",
            output_root=tmp_path / "attempt-2",
            seed_namespace={},
            source_hash="source-v1",
            plan_hash="plan-v1",
        )
