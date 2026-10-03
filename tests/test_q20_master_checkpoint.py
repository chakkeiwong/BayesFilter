"""Master scheduling/accounting fixtures; synthetic times are not cost evidence."""
import json
from pathlib import Path

import pytest

from bayesfilter.inference import q20_master_checkpoint as master
from bayesfilter.inference.q20_campaign_runtime import atomic_json
from bayesfilter.inference.q20_training_resume import checksum
from tests.test_q20_master_bootstrap_repair import fixture


def setup(tmp_path, monkeypatch):
    campaign, previous, repo = fixture(tmp_path, monkeypatch)
    path = previous / "campaign.json"
    state = json.loads(path.read_text())
    attempt = previous / "old-preparation"
    progress = attempt / "worker/data/preparation_progress.json"
    progress.parent.mkdir(parents=True)
    atomic_json(progress, {"events": [
        {"phase": "bootstrap_initialization.completed", "elapsed_seconds": 280., "details": {}},
        {"phase": "bootstrap.bootstrap_round_hmc_call_complete", "details": {
            "bootstrap_diagnostic_screen_num_results": 32,
            "bootstrap_diagnostic_screen_num_burnin_steps": 8, "elapsed_s": 1500.}},
        {"phase": "bootstrap.bootstrap_round_hmc_call_start", "details": {}}]})
    state["attempts"] = [
        {"stage": "repair-qualify-beta1", "status": "completed", "elapsed_seconds": 200., "directory": str(previous / "qualify")},
        {"stage": "repair-preparation-beta1", "status": "timed_out", "elapsed_seconds": 4800., "directory": str(attempt)}]
    atomic_json(path, state)
    allowance = json.loads((previous / "settled-allowance.json").read_text())
    allowance["source_sha256"] = checksum(path)
    atomic_json(previous / "settled-allowance.json", allowance)
    campaign.state["allowance"]["predecessor_sha256"] = checksum(path)
    return campaign, previous, repo


def test_measured_budget_and_completed_stage_reuse(tmp_path, monkeypatch):
    campaign, previous, repo = setup(tmp_path, monkeypatch)
    original = campaign.numerical_stage
    def deferred(name, request, **kw):
        result = original(name, request, **kw)
        if request["stage"] == "price-preparation":
            result["result"].update(status="preparation_deferred_by_cost", bootstrap_passed=True)
        return result
    monkeypatch.setattr(campaign, "numerical_stage", deferred)
    result = master.checkpoint_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert result["status"] == "MASTER_STARTUP_PASSED_MASS_DEFERRED"
    assert result["details"]["mass_adaptation_completed"] is False
    costs = result["predecessor"]["costs"]
    factor = campaign.config["budget"]["forecast_safety_factor"]
    assert costs["seconds_per_transition"] == 37.5
    assert costs["qualification_cap_seconds"] == factor*200+100
    assert costs["preparation_cap_seconds"] == factor*1780+100
    assert len(campaign.calls) == 4
    repeated = master.checkpoint_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert repeated == result and len(campaign.calls) == 4


def test_stale_allowance_cannot_renew_budget(tmp_path, monkeypatch):
    campaign, previous, repo = setup(tmp_path, monkeypatch)
    campaign.state["allowance"]["diagnostic_remaining_seconds"] += 1.
    with pytest.raises(ValueError, match="debit"):
        master.checkpoint_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert not campaign.calls


def test_unfunded_stage_is_deferred_without_launch(tmp_path, monkeypatch):
    campaign, previous, repo = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "remaining", lambda diagnostic=False: 1.)
    result = master.checkpoint_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert result["status"] == "MASTER_CHECKPOINT_REPAIR_PAUSED"
    assert not campaign.calls
