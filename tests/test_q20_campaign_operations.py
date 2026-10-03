"""Host supervisor recovery tests; no posterior claims or GPU work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def module(name):
    path = Path(__file__).resolve().parents[1] / f"scripts/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


driver = module("q20_campaign_supervisor")
control = module("q20_campaign_control")


class FakeCampaign:
    def __init__(self, root, outcomes):
        from bayesfilter.inference.q20_production_config import protocol_template
        self.root, self.outcomes, self.calls = root, list(outcomes), 0
        self.config = protocol_template()
        self.state = {"attempts": [], "stages": {}, "stage_limits": {}, "spent_seconds": 0.,
                      "deadline_extension": {"initial_tuning_attempts": 1}}

    def remaining(self, diagnostic=False):
        return 1000. - self.state["spent_seconds"]

    def stage_remaining(self, name):
        return self.state["stage_limits"].get(name, self.config["budget"]["arm_cap_seconds"]) - sum(a["elapsed_seconds"] for a in self.state["attempts"] if a["stage"] == name)

    def save(self):
        pass

    def numerical_stage(self, name, request, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        folder = self.root/f"attempt-{self.calls}"
        driver.write_json(folder/"worker/data/chunks/identity.json", {"role": "mechanics fixture"})
        attempt = {"stage": name, "directory": str(folder), "status": "completed", "diagnostic": False,
                   "elapsed_seconds": 2., "cap_seconds": kwargs["cap_seconds"]}
        self.state["attempts"].append(attempt)
        self.state["spent_seconds"] += 2.
        if outcome == "resource":
            attempt["failure_classification"] = "resource_unavailable"
            return {"completed": False, "status": "waiting_for_gpu"}
        if outcome == "transient":
            return {"completed": False, "status": "worker_failure", "type": "UnavailableError"}
        if outcome == "unknown":
            attempt["status"] = "failed"
            return {"completed": False, "status": "worker_failure", "type": "ValueError"}
        if outcome == "budget":
            return {"completed": False, "budget_paused": True, "status": "budget_paused"}
        return {"completed": True, "status": "computed", "result": {"passed": outcome == "pass"}}


@pytest.fixture
def job(monkeypatch):
    monkeypatch.setattr(driver.time, "time", lambda: 100.)
    monkeypatch.setattr(driver.time, "sleep", lambda _: None)
    return {"deadline_epoch": 10000., "deadline_local": "test calendar bound", "poll_seconds": 30.,
            "max_recovery_attempts_per_phase": 1}


def test_resource_wait_and_checkpoint_recovery_finish_without_reapproval(tmp_path, job):
    c = FakeCampaign(tmp_path, ["transient", "resource", "pass"])
    result = driver.execute_phase(c, job, "sample-neutra", {"stage": "sample"}, reserve=500.)
    assert result["completed"] and c.calls == 3
    assert c.state["spent_seconds"] == 6.
    assert c.state["stage_limits"]["sample-neutra"] == 500.
    assert len(c.state["recovery_attempts"]["sample-neutra"]) == 1
    events = [json.loads(x) for x in (tmp_path/"phase-events.jsonl").read_text().splitlines()]
    assert {"infrastructure_repair", "resource_wait", "phase_complete"} <= {e["event"] for e in events}
    assert driver.read_json(tmp_path/"next-phase.json")["event"] == "phase_complete"
    assert driver.remaining_repair(c) == 7198.


def test_repeated_failure_stops_after_one_recovery(tmp_path, job):
    from bayesfilter.inference.q20_master_program import StageIncomplete
    c = FakeCampaign(tmp_path, ["transient", "transient", "pass"])
    with pytest.raises(StageIncomplete, match="attempt_limit"):
        driver.execute_phase(c, job, "sample-neutra", {"stage": "sample"}, reserve=500.)
    assert c.calls == 2
    assert driver.read_json(tmp_path/"next-phase.json")["decision"] == "agent_repair_required"


@pytest.mark.parametrize("outcome", ["unknown", "budget"])
def test_unknown_and_budget_stops_cannot_retry_or_become_scientific_rejection(tmp_path, job, outcome):
    from bayesfilter.inference.q20_master_program import StageIncomplete
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    c = FakeCampaign(tmp_path, [outcome, "pass"])
    with pytest.raises((StageIncomplete, StageBudgetPause)):
        driver.execute_phase(c, job, "sample-neutra", {"stage": "sample"}, reserve=500.)
    assert c.calls == 1
    assert not c.state.get("recovery_attempts")


def test_numerical_candidate_failure_returns_to_declared_master_branch(tmp_path, job):
    c = FakeCampaign(tmp_path, ["candidate_failed", "pass"])
    result = driver.execute_phase(c, job, "sample-neutra", {"stage": "sample"}, reserve=500.)
    assert result["completed"] and not result["result"]["passed"]
    assert c.calls == 1


def test_exhausted_stage_cannot_be_renewed_by_retry(tmp_path, job):
    from bayesfilter.inference.q20_stage_budget import StageBudgetPause
    c = FakeCampaign(tmp_path, ["pass"])
    c.state["stage_limits"]["sample-neutra"] = 1.
    with pytest.raises(StageBudgetPause):
        driver.execute_phase(c, job, "sample-neutra", {"stage": "sample"}, reserve=500.)
    assert c.calls == 0


def test_shared_invalidity_cannot_be_a_transient_repair(tmp_path):
    attempt = {"directory": str(tmp_path), "status": "interrupted"}
    driver.write_json(tmp_path/"worker/data/tuning/tuning_checkpoint.json",
                      {"result": {"completion_status": "shared_invalidity"}})
    assert not driver.recoverable_interruption({"type": "UnavailableError"}, attempt, "tune")


def pricing_repair_campaign(tmp_path):
    c = FakeCampaign(tmp_path, ["pass"])
    c.state["attempts"] = [{"stage": "price-ensemble", "status": "timed_out",
        "directory": str(tmp_path/"prior"), "elapsed_seconds": 20., "diagnostic": True}]
    c.state["spent_seconds"] = 20.
    c.state["stage_limits"]["price-ensemble"] = 220.
    c.state["operational_repairs"] = {"price-ensemble": {
        "kind": "diagnostic_timeout_reallocation", "attempt_seconds": 200.,
        "cumulative_limit_seconds": 220., "attempt_limit": 2}}
    c.config["execution"]["diagnostic_attempt_seconds"] = 20.
    return c


def test_pricing_reallocation_retries_once_and_preserves_spend(tmp_path, job):
    c = pricing_repair_campaign(tmp_path)
    result = driver.execute_phase(c, job, "price-ensemble",
        {"stage": "price", "method": "ensemble"}, diagnostic=True)
    assert result["completed"]
    assert c.calls == 1 and c.state["attempts"][-1]["cap_seconds"] == 200.
    assert c.state["spent_seconds"] == 22.
    assert c.state["stage_limits"]["price-ensemble"] == 220.
    from bayesfilter.inference.q20_master_program import StageIncomplete
    with pytest.raises(StageIncomplete, match="attempt_limit"):
        driver.execute_phase(c, job, "price-ensemble",
            {"stage": "price", "method": "ensemble"}, diagnostic=True)
    assert c.calls == 1


@pytest.mark.parametrize("bound", ["diagnostic", "calendar", "cumulative"])
def test_pricing_reallocation_obeys_every_remaining_bound(tmp_path, job, bound):
    c = pricing_repair_campaign(tmp_path)
    if bound == "diagnostic":
        c.remaining = lambda diagnostic=False: 40. if diagnostic else 980.
    elif bound == "calendar":
        job["deadline_epoch"] = 145.
    else:
        c.state["attempts"][0]["elapsed_seconds"] = 180.
    result = driver.execute_phase(c, job, "price-ensemble",
        {"stage": "price", "method": "ensemble"}, diagnostic=True)
    assert result["completed"]
    assert c.state["attempts"][-1]["cap_seconds"] == 40.


def test_pricing_reallocation_cannot_change_another_stage_or_renew_budget(tmp_path):
    c = pricing_repair_campaign(tmp_path)
    with pytest.raises(ValueError, match="restricted"):
        driver.diagnostic_reallocation(c, "price-ensemble", {"stage": "tune"}, True)
    c.state["stage_limits"]["price-ensemble"] += 1.
    with pytest.raises(ValueError, match="cumulative"):
        driver.diagnostic_reallocation(c, "price-ensemble",
            {"stage": "price", "method": "ensemble"}, True)


def test_status_reads_the_explicit_successor(tmp_path, monkeypatch):
    control.write(tmp_path/"job.json", {"campaign_name": "campaign-02",
        "parent_campaign": str(tmp_path/"nonexistent"), "parent_unit": "parent",
        "deadline_local": "recorded", "plan_file": "plan"})
    (tmp_path/"campaign-02").mkdir()
    control.write(tmp_path/"campaign-02/campaign.json", {"status": "CONTINUATION_PREPARED",
        "campaign_limit": 1000., "spent_seconds": 20., "attempts": []})
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "service", lambda _: {"ActiveState": "inactive"})
    assert control.status()["campaign_status"] == "CONTINUATION_PREPARED"
    assert control.status()["remaining_campaign_hours_observed"] == 980./3600.


@pytest.mark.parametrize("failure", [None, "sample"])
def test_repaired_phase_runner_preserves_full_master_progression(tmp_path, job, failure):
    from tests.test_q20_estimation_objective import harness
    from bayesfilter.inference.q20_master_program import run_estimation_attempts
    original, stage, finish, calls = harness(tmp_path, plain_failure=failure)
    original.config["execution"]["termination_grace_seconds"] = .000001
    c = FakeCampaign(tmp_path, [])
    c.config = original.config
    c.state["campaign_limit"] = 10000.
    c.remaining = lambda diagnostic=False: 10000. - c.state["spent_seconds"]
    def numerical(name, request, **kwargs):
        folder = tmp_path/(name+"-receipt")
        folder.mkdir(exist_ok=True)
        attempt = {"stage": name, "directory": str(folder), "status": "completed",
                   "diagnostic": False, "elapsed_seconds": .00001}
        c.state["attempts"].append(attempt)
        c.state["spent_seconds"] += .00001
        return {**stage(name, request), "completed": True, "status": "computed"}
    c.numerical_stage = numerical
    result = run_estimation_attempts(c, lambda name, request, **kw:
        driver.execute_phase(c, job, name, request, **kw), finish)
    assert result["status"] == "SMOKE_ESTIMATION_CHECKS_PASSED"
    assert result["details"]["method"] == ("ensemble" if failure else "neutra")
    stages = [request["stage"] for _, request in calls]
    assert stages[-2:] == ["reference", "assess"]
    assert stages.count("ensemble") == bool(failure)
    phases = [json.loads(x) for x in (tmp_path/"phase-events.jsonl").read_text().splitlines()]
    assert phases[-1]["event"] == "phase_complete"


def test_ensure_is_idempotent_and_keeps_active_worker(tmp_path, monkeypatch):
    script = tmp_path/"driver.py"
    script.write_text("# fixed fixture\n")
    control.write(tmp_path/"job.json", {"driver_path": str(script), "driver_sha256": driver.checksum(script)})
    control.write(tmp_path/"launch.json", {"unit": "existing.service"})
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "service", lambda _: {"ActiveState": "active", "MainPID": "123"})
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **k: pytest.fail("active supervisor must not restart"))
    assert control.ensure()["action"] == "already_running"


def test_ensure_refuses_to_replace_observer_that_started_numerical_work(tmp_path, monkeypatch):
    script = tmp_path/"driver.py"
    script.write_text("# fixture\n")
    control.write(tmp_path/"job.json", {"driver_path": str(script), "driver_sha256": driver.checksum(script),
                                       "deadline_epoch": 9999999999.})
    old = tmp_path/"old"
    old.mkdir()
    control.write(old/"status.json", {"status": "CONTINUING_MASTER", "numerical_work_launched": True})
    monkeypatch.setattr(control, "ROOT", tmp_path)
    monkeypatch.setattr(control, "OLD", old)
    monkeypatch.setattr(control, "service", lambda _: {"ActiveState": "active"})
    monkeypatch.setattr(control.subprocess, "run", lambda *a, **k: pytest.fail("must not stop a numerical service"))
    with pytest.raises(RuntimeError, match="taken over"):
        control.ensure()
