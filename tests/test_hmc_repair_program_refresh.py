"""Phase continuation preserves failed candidates and respects real stop conditions."""
import json

import pytest

from scripts.refresh_hmc_repair_program import refresh


def fixture(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"active_phase": "M14", "phases": {
        "M14": {"scientific_gaps_closed": False}, "M15": {"cpu_limit": 40, "gpu_limit": 20}}}))
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"terminal": True, "invalid_artifacts": [], "outstanding_workers": [],
        "candidate_failures": ["warmup_cap"], "budget_seconds": {
            "remaining": {"cpu_reference": 100, "gpu": 100}, "exceeded": False}}))
    result = tmp_path / "result.md"
    result.write_text("A candidate reached its cap; next design investigates it.")
    design = tmp_path / "next.md"
    design.write_text("Refreshed design, evidence contract, finite budget and skeptical audit.")
    kwargs = dict(ledger_path=ledger, result_path=result, next_phase="M15",
        next_design_path=design, review_note="Failed candidate triggers the next planned repair.",
        output_path=tmp_path / "refresh.json")
    return progress, kwargs


def test_phase_refresh_continues_after_candidate_failure(tmp_path):
    progress, kwargs = fixture(tmp_path)
    record = refresh(progress, "M14", **kwargs)
    updated = json.loads(progress.read_text())
    assert updated["active_phase"] == "M15"
    assert updated["phases"]["M15"]["status"] == "ready_to_execute"
    assert record["remaining_worker_seconds"]["gpu"] == 100
    assert not record["scientific_gaps_closed"]
    assert not record["candidate_failure_is_continuation_veto"]


@pytest.mark.parametrize("failure", ["integrity", "running", "budget", "missing_design"])
def test_phase_refresh_stops_on_actual_continuation_blocker(tmp_path, failure):
    progress, kwargs = fixture(tmp_path)
    previous = progress.read_bytes()
    ledger = json.loads(kwargs["ledger_path"].read_text())
    if failure == "integrity":
        ledger["invalid_artifacts"] = ["corrupted reference"]
    elif failure == "running":
        ledger["outstanding_workers"] = ["unfinished fit"]
    elif failure == "budget":
        ledger["budget_seconds"]["remaining"]["gpu"] = 5
    else:
        kwargs["next_design_path"].unlink()
    kwargs["ledger_path"].write_text(json.dumps(ledger))
    with pytest.raises(ValueError):
        refresh(progress, "M14", **kwargs)
    assert progress.read_bytes() == previous
    assert not kwargs["output_path"].exists()
