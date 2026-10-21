"""CPU-only queue/budget diagnostics; subprocess stubs provide no HMC evidence."""
import json
from pathlib import Path

import pytest

from scripts import run_surrogate_hmc_60h_continuation as queue


def initialize(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "preflight", lambda: {})
    queue.master.write(tmp_path / "manifest.json", {"supervisor_sources": {}})
    queue.master.write(tmp_path / "budget.json", dict(
        authorized_gpu_wall_seconds=queue.TOTAL_SECONDS, max_attempts=6, attempts=[]))


def fake_worker(monkeypatch, outcome, seen, *, malformed=False):
    class Worker:
        def __init__(self, command, **kwargs):
            model = command[command.index("--model") + 1]
            seconds = float(command[command.index("--budget-seconds") + 1])
            timeout = float(command[command.index("--per-model-timeout") + 1])
            assert seconds == timeout
            assert command[command.index("--gpu") + 1] == queue.GPU
            assert command[command.index("--priors") + 1] == queue.master.LITERATURE_PRIOR
            out = Path(command[command.index("--output") + 1])
            ledger_path = Path(command[command.index("--campaign-ledger") + 1])
            ledger = queue.master.read(ledger_path)
            available = ledger["authorized_gpu_wall_seconds"] - sum(
                row["wall_seconds"] for row in ledger["attempts"])
            assert seconds <= available
            seen.append((model, seconds))
            status = outcome.get(model, "VERIFIED")
            row = dict(status=status, wall_seconds=seconds, result=str(out / model / "inference/hmc.json"))
            ledger["attempts"].append(row)
            queue.master.write(ledger_path, ledger)
            queue.master.write(out / "master.json", dict(
                status="INCOMPLETE_WORKER_ERROR" if malformed else (
                    "SELECTED_MODELS_HMC_SAMPLING_COMPLETE" if status == "VERIFIED"
                    else "INCOMPLETE_SEE_MODEL_STATUS"), models={model: {"inference": row}}))
            self.returncode = 0 if status == "VERIFIED" else 2
            self.pid = 12345

        def poll(self):
            return self.returncode

    monkeypatch.setattr(queue.subprocess, "Popen", Worker)


def test_queue_preserves_order_and_exact_60_hour_cap(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {}, seen)
    assert queue.run(tmp_path) == 0
    assert seen == [("range_bearing", 28800), ("predator_prey", 57600), ("sir_d18", 129600)]
    state = queue.master.read(tmp_path / "state.json")
    assert state["status"] == "COMPLETE_RECORDED_SAMPLING_RULE"
    assert state["remaining_worker_seconds_estimate"] == 0
    next_plan = queue.master.read(tmp_path / "next-phase-plan.json")
    assert next_plan["part_of_master"] and not next_plan["automatic_continuation"]
    assert next_plan["pending_models"] == []
    assert not next_plan["redundant_validation_phase"]


@pytest.mark.parametrize("failure", ["HEALTH_VETO", "STAGE_TIME_BUDGET_STOP", "HMC_TUNING_REPAIR_REQUIRED"])
def test_candidate_failure_continues_other_models(tmp_path, monkeypatch, failure):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {"range_bearing": failure}, seen)
    assert queue.run(tmp_path) == 2
    assert len(seen) == 3
    state = queue.master.read(tmp_path / "state.json")
    assert state["models"]["range_bearing"]["status"] == failure
    assert state["models"]["sir_d18"]["status"] == "VERIFIED"
    assert state["status"] == "INCOMPLETE_SEE_MODEL_RESULTS"


def test_corrupt_controller_result_stops_queue(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {}, seen, malformed=True)
    with pytest.raises(RuntimeError, match="infrastructure failed"):
        queue.run(tmp_path)
    assert len(seen) == 1
    assert queue.master.read(tmp_path / "state.json")["status"] == "SUPERVISOR_ERROR"


def test_remaining_budget_limits_next_worker(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    ledger = queue.master.read(tmp_path / "budget.json")
    ledger["attempts"].append(dict(status="REPAIR_ATTEMPT", wall_seconds=queue.TOTAL_SECONDS - 100))
    queue.master.write(tmp_path / "budget.json", ledger)
    seen = []
    fake_worker(monkeypatch, {}, seen)
    assert queue.run(tmp_path) == 2
    assert seen == [("range_bearing", 100)]
    state = queue.master.read(tmp_path / "state.json")
    assert state["models"]["sir_d18"]["status"] == "CAMPAIGN_BUDGET_STOP"


def test_changed_source_fails_before_worker_launch(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    path = Path(queue.__file__)
    queue.master.write(tmp_path / "manifest.json", {"supervisor_sources": {
        str(path.relative_to(queue.ROOT)): "wrong"}})
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("worker launched"))
    with pytest.raises(ValueError, match="supervisor changed"):
        queue.run(tmp_path)


def initialize_retry(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    ledger = queue.master.read(tmp_path / "budget.json")
    ledger["attempts"] = [dict(model=m, status=s, wall_seconds=t) for m, s, t in (
        ("range_bearing", "HEALTH_VETO", 6087.644374644005),
        ("predator_prey", "VERIFIED", 3600.), ("sir_d18", "VERIFIED", 3600.))]
    queue.master.write(tmp_path / "budget.json", ledger)
    queue.master.write(tmp_path / "state.json", dict(
        status="INCOMPLETE_SEE_MODEL_RESULTS", pid=123, models={}))
    monkeypatch.setattr(queue, "_pid_running", lambda pid: False)
    root = tmp_path / "range-bearing-retry-01"
    root.mkdir()
    queue.master.write(root / "manifest.json", dict(source_sha256={}, maximum_retry_seconds=22712.355625355995))
    queue.master.write(tmp_path / "queued-range-bearing.json", dict(status="QUEUED", output=str(root)))


def test_retry_waits_for_original_queue_and_reuses_only_unspent_allocation(tmp_path, monkeypatch):
    initialize_retry(tmp_path, monkeypatch)
    parent = queue.master.read(tmp_path / "state.json")
    parent["status"] = "RUNNING"
    queue.master.write(tmp_path / "state.json", parent)
    monkeypatch.setattr(queue, "_pid_running", lambda pid: queue.master.read(tmp_path / "state.json")["status"] == "RUNNING")
    before = (tmp_path / "budget.json").read_bytes()
    waits, seen = [], []
    def wait(seconds):
        waits.append(seconds)
        assert (tmp_path / "budget.json").read_bytes() == before
        assert not seen
        parent["status"] = "INCOMPLETE_SEE_MODEL_RESULTS"
        queue.master.write(tmp_path / "state.json", parent)
    monkeypatch.setattr(queue.time, "sleep", wait)
    fake_worker(monkeypatch, {}, seen)
    assert queue.run_range_bearing_retry(tmp_path) == 0
    assert waits == [30]
    assert seen == [("range_bearing", pytest.approx(22712.355625355995))]
    assert queue.master.read(tmp_path / "queued-range-bearing.json")["status"] == "VERIFIED"
    assert not queue.master.read(tmp_path / "next-phase-plan.json")["automatic_continuation"]


@pytest.mark.parametrize("cause", ["dead_parent", "parent_error", "active_ledger", "source_drift"])
def test_retry_cannot_launch_with_unresolved_parent_or_sources(tmp_path, monkeypatch, cause):
    initialize_retry(tmp_path, monkeypatch)
    parent = queue.master.read(tmp_path / "state.json")
    if cause == "dead_parent":
        parent["status"] = "RUNNING"
    elif cause == "parent_error":
        parent["status"] = "SUPERVISOR_ERROR"
    elif cause == "active_ledger":
        ledger = queue.master.read(tmp_path / "budget.json")
        ledger["attempts"][-1]["status"] = "RUNNING"
        queue.master.write(tmp_path / "budget.json", ledger)
    else:
        path = tmp_path / "range-bearing-retry-01/manifest.json"
        manifest = queue.master.read(path)
        manifest["source_sha256"] = {str(Path(queue.__file__).relative_to(queue.ROOT)): "wrong"}
        queue.master.write(path, manifest)
    queue.master.write(tmp_path / "state.json", parent)
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("worker launched"))
    with pytest.raises((RuntimeError, ValueError)):
        queue.run_range_bearing_retry(tmp_path)
    assert queue.master.read(tmp_path / "queued-range-bearing.json")["status"] == "RETRY_SUPERVISOR_ERROR"


@pytest.mark.parametrize("limit", ["time", "attempts"])
def test_retry_respects_remaining_campaign_limits(tmp_path, monkeypatch, limit):
    initialize_retry(tmp_path, monkeypatch)
    ledger = queue.master.read(tmp_path / "budget.json")
    if limit == "time":
        ledger["attempts"][-1]["wall_seconds"] = queue.TOTAL_SECONDS - 100. - sum(
            row["wall_seconds"] for row in ledger["attempts"][:-1])
    else:
        ledger["max_attempts"] = 3
    queue.master.write(tmp_path / "budget.json", ledger)
    seen = []
    fake_worker(monkeypatch, {}, seen)
    code = queue.run_range_bearing_retry(tmp_path)
    assert (code, seen) == ((0, [("range_bearing", 100.)]) if limit == "time" else (2, []))


def test_enqueue_is_durable_unique_and_does_not_write_active_budget(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    monkeypatch.setattr(queue, "preflight", lambda: {"range_bearing": {"selection": "frozen"}})
    monkeypatch.setattr(queue, "_pid_running", lambda pid: True)
    queue.master.write(tmp_path / "state.json", dict(status="RUNNING", pid=123, models={}))
    before = (tmp_path / "budget.json").read_bytes()
    class Detached:
        def __init__(self, command, **kwargs):
            assert command[2] == "run-range-bearing-retry"
            assert kwargs["start_new_session"]
            self.pid = 456
    monkeypatch.setattr(queue.subprocess, "Popen", Detached)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **kw: "test-commit")
    result = queue.enqueue_range_bearing(tmp_path)
    assert result["status"] == "QUEUED" and result["pid"] == 456
    assert (tmp_path / "budget.json").read_bytes() == before
    with pytest.raises(FileExistsError, match="already queued"):
        queue.enqueue_range_bearing(tmp_path)
