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
