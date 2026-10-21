"""CPU-only supervisor checks with stub workers; no sampler evidence."""
import json
from types import SimpleNamespace

from scripts import run_surrogate_hmc_range_flow16 as queue


def test_queue_preserves_budget_and_uses_fresh_flow16_selection(tmp_path, monkeypatch):
    campaign, prior, output = tmp_path/"campaign", tmp_path/"prior", tmp_path/"output"
    campaign.mkdir()
    prior.mkdir()
    monkeypatch.setattr(queue, "CAMPAIGN", campaign)
    monkeypatch.setattr(queue, "PRIOR", prior)
    monkeypatch.setattr(queue, "process_identity", lambda pid: None)
    ledger = campaign/"budget.json"
    queue.preparation.save(ledger, dict(authorized_gpu_wall_seconds=8000.,
        max_attempts=1, attempts=[]))
    queue.preparation.save(prior/"all-models-scope-execution.json",
        dict(scopes=dict(range_bearing={})))
    commands = []
    def fake_run(command, **kwargs):
        commands.append(command)
        if command[2] == "advance":
            (output/"preparation/range_bearing/filter-tuning/range_bearing").mkdir(parents=True)
            queue.preparation.save(output/"preparation/range_bearing/filter-tuning/range_bearing/filter-selection.json",
                dict(selected=dict(controls=dict(flow_steps=16))))
            record = json.loads(ledger.read_text())
            record["attempts"] = [dict(wall_seconds=900.), dict(wall_seconds=600.)]
            queue.preparation.save(ledger, record)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess, "run", fake_run)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "fixture_commit")
    assert queue.main(["--wait-pid", "123", "--output", str(output)]) == 0
    assert len(commands) == 2
    assert all(c[c.index("--model")+1] == "range_bearing" for c in commands)
    assert commands[0][commands[0].index("--flow-steps")+1] == "16"
    assert "--flow-steps" not in commands[1]
    assert commands[1][commands[1].index("--advance-root")+1] == str(output/"preparation")
    assert float(commands[1][commands[1].index("--budget-seconds")+1]) == 6500.
    assert json.loads(ledger.read_text())["authorized_gpu_wall_seconds"] == 8000.
    scope = json.loads((output/"execution.json").read_text())["scopes"]["range_bearing"]
    assert scope["tuning_seed_offset"] == 400
    assert scope["runtime_precision"] == "uniform"
    assert scope["control_family"] == "fixed_epsilon1_steps5"


def test_queue_does_not_spend_a_running_reservation(tmp_path, monkeypatch):
    campaign, output = tmp_path/"campaign", tmp_path/"output"
    campaign.mkdir()
    monkeypatch.setattr(queue, "CAMPAIGN", campaign)
    monkeypatch.setattr(queue, "process_identity", lambda pid: None)
    record = dict(authorized_gpu_wall_seconds=8000., max_attempts=4,
        attempts=[dict(status="RUNNING", reserved_seconds=7000.)])
    queue.preparation.save(campaign/"budget.json", record)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "fixture_commit")
    monkeypatch.setattr(queue.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not launch a second ledger writer")))
    assert queue.main(["--wait-pid", "123", "--output", str(output)]) == 2
    assert json.loads((campaign/"budget.json").read_text()) == record
    assert json.loads((output/"state.json").read_text())["status"] == "RUNNING_ATTEMPT_REQUIRES_RECONCILIATION"
