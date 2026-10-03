"""Accounting and source-isolation tests; fake timings are not cost evidence."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from bayesfilter.inference import q20_master_repair as repair
from bayesfilter.inference.q20_campaign_runtime import atomic_json, source_snapshot
from bayesfilter.inference.q20_production_config import digest
from bayesfilter.inference.q20_training_resume import checksum
from tests.test_q20_master_refresh import small_repo
from tests.test_q20_production_repair import tiny_protocol


class FakeCampaign:
    def __init__(self, root, repo, config, allowance):
        self.root, self.repo, self.config = root, repo, config
        root.mkdir()
        self.path = root / "campaign.json"
        self.state = {"sources": source_snapshot(repo), "allowance": allowance,
            "stages": {}, "attempts": [], "spent_seconds": 0.}
        self.calls = []

    def save(self):
        atomic_json(self.path, self.state)

    def remaining(self, diagnostic=False):
        key = "diagnostic_remaining_seconds" if diagnostic else "campaign_remaining_seconds"
        return self.state["allowance"][key] - self.state["spent_seconds"]

    def numerical_stage(self, name, request, *, cap_seconds, **kwargs):
        if name in self.state["stages"]:
            return self.state["stages"][name]
        self.calls.append((name, copy.deepcopy(request), cap_seconds))
        self.state["attempts"].append({"stage": name, "elapsed_seconds": 10., "status": "completed"})
        self.state["spent_seconds"] += 10.
        if request["stage"] == "qualify":
            receipt = {"sources": self.state["sources"], "betas": {str(request["betas"][0]): "passed"}}
            result = {**receipt, "checksum": digest(receipt)}
        else:
            result = {"beta": request["beta"], "kind": "classical_preparation", "wall_seconds": 10.}
        path = self.root / (name + ".json")
        atomic_json(path, result)
        final = {"completed": True, "result": result, "result_path": str(path)}
        self.state["stages"][name] = final
        return final


def fixture(tmp_path, monkeypatch):
    config = tiny_protocol()
    config["jit_compile"] = True  # Fake qualification dispatch, no GPU or HMC.
    repo = small_repo(tmp_path / "repo")
    previous = tmp_path / "previous"
    previous.mkdir()
    checkpoint = previous / "checkpoint.json"
    atomic_json(checkpoint, {"fixture": True})
    state = {"config_hash": digest(config), "campaign_limit": 110000., "diagnostic_limit": 35000.,
        "spent_seconds": 5000., "diagnostic_spent_seconds": 5000.,
        "attempts": [{"stage": "refresh-old", "status": "completed", "elapsed_seconds": 5000.}],
        "sources": source_snapshot(repo), "stages": {},
        "training_resume": {"checkpoint": str(checkpoint), "sha256": checksum(checkpoint)}}
    for stage in ("refresh-price-training", "refresh-price-downstream"):
        path = previous / (stage + ".json")
        payload = {"completed": True, "result_path": str(path), "result": {
            "hmc": [], "sources": state["sources"], "missing_cost_categories": ["classical_preparation", "multi_chart_mixture_dispatch", "classical_metric_specific_transition"]}}
        atomic_json(path, payload)
        state["stages"][stage] = {"result_path": str(path), "result_sha256": checksum(path),
                                 "artifact_hashes": {str(path): checksum(path)}}
    atomic_json(previous / "campaign.json", state)
    settled = {"source_ledger": str(previous / "campaign.json"),
        "source_sha256": checksum(previous / "campaign.json"), "all_attempts_settled": True,
        "campaign_remaining_seconds": 105000., "diagnostic_remaining_seconds": 30000.}
    atomic_json(previous / "settled-allowance.json", settled)
    verification = tmp_path / "verification.json"
    atomic_json(verification, {"wall_seconds": 12., "status": "passed", "sources": source_snapshot(repo)})
    allowance = {"predecessor_sha256": settled["source_sha256"],
        "verification": {"path": str(verification), "sha256": checksum(verification)},
        "campaign_remaining_seconds": 105000.-192., "diagnostic_remaining_seconds": 30000.-192.}
    monkeypatch.setattr(repair, "read_training_checkpoint", lambda *a, **k: {})
    campaign = FakeCampaign(tmp_path / "campaign", repo, config, allowance)
    return campaign, previous, repo


def test_master_requalifies_only_changed_work_and_reuses_completed_stages(tmp_path, monkeypatch):
    campaign, previous, repo = fixture(tmp_path, monkeypatch)
    result = repair.repair_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert result["status"] == "MASTER_BOOTSTRAP_REPAIR_PREPARATION_PASSED"
    assert [call[1]["stage"] for call in campaign.calls] == ["qualify", "qualify", "price-preparation", "price-preparation"]
    assert result["predecessor"]["phase_cap_seconds"] == 14400.-5000.-192.
    assert result["remaining_phase_seconds"] == 14400.-5000.-192.-40.
    for name, request, cap in campaign.calls[2:]:
        assert cap == 4000. and request["max_seconds"] == 3900.
    assert result["details"]["missing_cost_categories"] == ["multi_chart_mixture_dispatch", "classical_metric_specific_transition"]
    repeated = repair.repair_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert repeated == result
    assert len(campaign.calls) == 4 and campaign.state["spent_seconds"] == 40.


@pytest.mark.parametrize("defect", ["allowance", "verification", "old_artifact", "ledger"])
def test_stale_or_renewed_evidence_blocks_before_launch(tmp_path, monkeypatch, defect):
    campaign, previous, repo = fixture(tmp_path, monkeypatch)
    if defect == "allowance":
        campaign.state["allowance"]["campaign_remaining_seconds"] += 180.
    elif defect == "verification":
        Path(campaign.state["allowance"]["verification"]["path"]).write_text("{}")
    elif defect == "old_artifact":
        (previous / "refresh-price-downstream.json").write_text("{}")
    else:
        (previous / "campaign.json").write_text("{}")
    with pytest.raises(ValueError):
        repair.repair_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert not campaign.calls


def test_stage_timeout_cannot_renew_its_budget_on_resume(tmp_path, monkeypatch):
    campaign, previous, repo = fixture(tmp_path, monkeypatch)
    campaign.state["attempts"] = [{"stage": "repair-preparation-beta0.5", "elapsed_seconds": 4000., "status": "timed_out"}]
    campaign.state["spent_seconds"] = 4000.
    result = repair.repair_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert result["status"] == "MASTER_BOOTSTRAP_REPAIR_INCOMPLETE"
    assert "classical_preparation" in result["details"]["missing_cost_categories"]
    assert not any(call[0] == "repair-preparation-beta0.5" for call in campaign.calls)
    assert all(cap <= 4000 for _, _, cap in campaign.calls)


def test_source_import_rejects_unrelated_numerical_change(tmp_path):
    before = small_repo(tmp_path / "before")
    after = tmp_path / "after"
    shutil.copytree(before, after)
    previous = source_snapshot(before)
    path = after / "bayesfilter/inference/q20_master_program.py"
    path.write_text("VERSION = 1\ndef execute_master():\n    return 'repair'\n")
    assert repair.check_repair_sources(previous, before, after) == ["bayesfilter/inference/q20_master_program.py"]
    (after / "bayesfilter/inference/target.py").write_text("TARGET = 8\n")
    with pytest.raises(ValueError, match="unreviewed repair source change"):
        repair.check_repair_sources(previous, before, after)


@pytest.mark.parametrize("error_type, expected", [(None, "candidate_preparation_failure"), ("ValueError", "execution_failure")])
def test_captured_runtime_exception_is_a_continuation_veto(tmp_path, error_type, expected):
    path = tmp_path / "worker/data"
    path.mkdir(parents=True)
    atomic_json(path / "preparation_progress.json", {"failure": {"details": {
        "rounds": [{"diagnostics": {"error_type": error_type}}]}}})
    result = {"type": "HMCPreparationFailure", "message": "bootstrap failed",
              "attempt": {"directory": str(tmp_path)}}
    assert repair.preparation_failure_classification(result) == expected


def test_fresh_source_retry_debits_all_prior_worker_costs(tmp_path, monkeypatch):
    campaign, previous, repo = fixture(tmp_path, monkeypatch)
    prior = tmp_path / "repair-prior"
    prior.mkdir()
    state = {"config_hash": digest(campaign.config), "repair_predecessor": {
        "sha256": campaign.state["allowance"]["predecessor_sha256"]}, "allowance": {},
        "campaign_limit": 105000.-192., "diagnostic_limit": 30000.-192.,
        "spent_seconds": 397., "diagnostic_spent_seconds": 397.,
        "attempts": [{"stage": "repair-preparation-beta0.5", "status": "failed", "diagnostic": True, "elapsed_seconds": 397.}]}
    ledger = prior / "campaign.json"
    atomic_json(ledger, state)
    atomic_json(prior / "settled-allowance.json", {"source_ledger": str(ledger),
        "source_sha256": checksum(ledger), "all_attempts_settled": True,
        "campaign_remaining_seconds": state["campaign_limit"]-397.,
        "diagnostic_remaining_seconds": state["diagnostic_limit"]-397.})
    history = [{"campaign": str(prior), "sha256": checksum(ledger)}]
    campaign.state["allowance"].update(repair_history=history)
    campaign.state["allowance"]["campaign_remaining_seconds"] -= 397.
    campaign.state["allowance"]["diagnostic_remaining_seconds"] -= 397.
    result = repair.repair_master(campaign, previous_campaign=previous, previous_source_root=repo)
    assert result["predecessor"]["prior_repair_worker_seconds"] == 397.
    assert result["remaining_phase_seconds"] == 14400.-5000.-192.-397.-40.
    preparation_call = next(row for row in campaign.calls if row[0] == "repair-preparation-beta0.5")
    assert preparation_call[2] == 4000.-397.
    assert preparation_call[1]["max_seconds"] == 4000.-397.-100.
    with pytest.raises(ValueError, match="duplicate"):
        repair.prior_repair_costs(history*2,
            predecessor_sha256=campaign.state["allowance"]["predecessor_sha256"], config=campaign.config)
