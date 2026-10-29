"""CPU-only regression checks for preserving a running worker during handoff."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from scripts import run_surrogate_hmc_48h_queue_repair as repair


@pytest.fixture
def processes():
    children = []

    def spawn():
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        children.append(child)
        return child, repair.process(child.pid)

    yield spawn
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        child.wait(timeout=5)


def write_json(path, value):
    path.write_text(json.dumps(value))


def test_parent_handoff_preserves_separate_worker(processes):
    parent, parent_identity = processes()
    worker, worker_identity = processes()
    os.kill(parent.pid, signal.SIGSTOP)
    repair.wait_stopped(parent_identity)

    repair.terminate_parent(parent_identity, worker_identity)

    assert not repair.alive(parent_identity)
    assert repair.alive(worker_identity)
    assert worker.poll() is None


def test_parent_handoff_rejects_shared_process_group(processes):
    _, parent_identity = processes()
    _, worker_identity = processes()
    worker_identity = {**worker_identity, "pgid": parent_identity["pgid"]}

    with pytest.raises(RuntimeError):
        repair.terminate_parent(parent_identity, worker_identity)

    assert repair.alive(parent_identity)
    assert repair.alive(worker_identity)


def test_cancellation_before_takeover_resumes_original_parent(tmp_path, processes):
    parent, parent_identity = processes()
    successor, _ = processes()
    receipt = tmp_path / "handoff.json"
    write_json(receipt, {"status": "PREPARED"})
    os.kill(parent.pid, signal.SIGSTOP)
    repair.wait_stopped(parent_identity)

    assert repair.cancel_handoff(receipt, parent_identity, successor)

    assert json.loads(receipt.read_text())["status"] == (
        "CANCELLED_OLD_SUPERVISOR_RESUMED"
    )
    assert repair.alive(parent_identity)
    assert repair.process(parent.pid)["state"] not in {"T", "t"}
    assert successor.poll() is not None


def test_cancellation_after_takeover_preserves_successor(tmp_path, processes):
    parent, parent_identity = processes()
    successor, successor_identity = processes()
    receipt = tmp_path / "handoff.json"
    write_json(receipt, {"status": "TAKEN_OVER"})
    parent.terminate()
    parent.wait(timeout=5)

    assert not repair.cancel_handoff(receipt, parent_identity, successor)

    assert json.loads(receipt.read_text())["status"] == "TAKEN_OVER"
    assert repair.alive(successor_identity)


def test_monitor_enforces_deadline_for_adopted_worker(processes):
    _, worker_identity = processes()

    assert not repair.monitor_adopted(worker_identity, time.time() - 1)

    assert not repair.alive(worker_identity)


def test_monitor_reports_already_finished_worker(processes):
    worker, worker_identity = processes()
    worker.terminate()
    worker.wait(timeout=5)

    assert repair.monitor_adopted(worker_identity, time.time() + 30)


def test_process_identity_does_not_accept_reused_pid(processes):
    _, identity = processes()

    assert not repair.alive({**identity, "start_ticks": identity["start_ticks"] + 1})
    assert repair.alive(identity)


def test_energy_resume_keeps_saved_kernel_and_shared_budget(tmp_path):
    command = repair.energy_resume_command(1234, tmp_path)
    assert "--hmc-target-accept" not in command
    assert "--hmc-initial-step-size" not in command
    assert command[command.index("--resume-manifest") + 1] == str(
        tmp_path / "range_bearing/range_bearing/inference/public-tuning/manifest.json")
    assert command[command.index("--output") + 1] == str(tmp_path / repair.ENERGY_RESUME_NAME)
    assert command[command.index("--campaign-ledger") + 1] == str(tmp_path / "budget.json")
    assert command[command.index("--advance-root") + 1] == str(repair.queue.RANGE_PREPARATION)
    assert command[command.index("--priors") + 1] == repair.master.LITERATURE_PRIOR
    assert command[command.index("--budget-seconds") + 1] == "1234"


def test_energy_resume_runs_after_live_predator_and_before_sir():
    assert [row[0] for row in repair.phase_order(True)] == [
        "predator_prey", "range_bearing", "sir_d18"]
    assert [row[0] for row in repair.phase_order(False)] == [
        "range_bearing", "predator_prey", "sir_d18"]


def test_stopped_restart_requires_terminal_idle_queue_and_budget(tmp_path, monkeypatch):
    import time
    deadline = time.time() + 10000
    manifest = dict(deadline_epoch=deadline, supervisor_sources={})
    state = dict(pid=123, status="INCOMPLETE_SEE_MODEL_RESULTS", deadline_epoch=deadline,
        models={"predator_prey": {"status": "VERIFIED"},
                "range_bearing": {"status": "HEALTH_VETO"},
                "sir_d18": {"status": "STAGE_ERROR"}})
    ledger = dict(authorized_gpu_wall_seconds=repair.queue.TOTAL_SECONDS,
        max_attempts=6, deadline_epoch=deadline, attempts=[{"status": "STAGE_ERROR"}]*4)
    write_json(tmp_path/"manifest.json", manifest)
    write_json(tmp_path/"state.json", state)
    monkeypatch.setattr(repair, "process", lambda pid: None)
    monkeypatch.setattr(repair.queue, "campaign_remaining", lambda path: (ledger, 10000))
    assert repair.validate_stopped(tmp_path)[1]["models"] == state["models"]
    ledger["attempts"].append({"status": "STAGE_ERROR"})
    with pytest.raises(RuntimeError, match="budget"):
        repair.validate_stopped(tmp_path)
    ledger["attempts"].pop()
    ledger["attempts"][0] = {"status": "RUNNING"}
    with pytest.raises(RuntimeError, match="idle"):
        repair.validate_stopped(tmp_path)


def test_retry_changes_only_output_directory(tmp_path):
    seconds = 1234
    expected = repair.queue.command_for(
        "predator_prey", seconds, repair.queue.PREPARATION, tmp_path
    )
    expected[expected.index("--output") + 1] = str(tmp_path / repair.RETRY_NAME)

    actual = repair.retry_command(seconds, tmp_path)

    assert actual == expected
    assert actual[actual.index("--campaign-ledger") + 1] == str(
        tmp_path / "budget.json"
    )
    assert actual[actual.index("--resume-manifest") + 1] == str(
        repair.queue.resume_manifest("predator_prey")
    )


def phase_fixture(tmp_path, status="INCOMPLETE_SEE_MODEL_STATUS"):
    phase_output = tmp_path / "phase"
    phase_output.mkdir()
    write_json(
        phase_output / "master.json",
        {
            "status": status,
            "models": {
                "range_bearing": {
                    "inference": {
                        "status": "BUDGET_STOP",
                        "wall_seconds": 12.5,
                        "result": "preserved-result.json",
                    }
                }
            },
        },
    )
    state = {
        "current_model": "range_bearing",
        "active_master_pid": 12345,
        "models": {},
    }
    return phase_output, state


def test_adopted_phase_records_unobserved_exit_honestly(tmp_path, monkeypatch):
    phase_output, state = phase_fixture(tmp_path)
    monkeypatch.setattr(
        repair.queue, "campaign_remaining", lambda _: ({"attempts": []}, 100)
    )

    repair.finish_phase(tmp_path, state, phase_output, None)

    row = state["models"]["range_bearing"]
    assert row["status"] == "BUDGET_STOP"
    assert row["returncode"] is None
    assert row["exit_code_observed"] is False
    assert row["wall_seconds"] == 12.5
    assert row["result"] == "preserved-result.json"
    assert "active_master_pid" not in state


def test_phase_cannot_finish_while_ledger_is_running(tmp_path, monkeypatch):
    phase_output, state = phase_fixture(tmp_path)
    monkeypatch.setattr(
        repair.queue,
        "campaign_remaining",
        lambda _: ({"attempts": [{"status": "RUNNING"}]}, 100),
    )

    with pytest.raises(RuntimeError):
        repair.finish_phase(tmp_path, state, phase_output, None)

    assert state["models"] == {}
    assert state["active_master_pid"] == 12345


@pytest.mark.parametrize("returncode", [1, -signal.SIGKILL])
def test_unexpected_worker_exit_is_not_success(tmp_path, monkeypatch, returncode):
    phase_output, state = phase_fixture(tmp_path)
    monkeypatch.setattr(
        repair.queue, "campaign_remaining", lambda _: ({"attempts": []}, 100)
    )

    with pytest.raises(RuntimeError):
        repair.finish_phase(tmp_path, state, phase_output, returncode)

    assert state["models"] == {}


def test_nonterminal_phase_is_not_accepted(tmp_path, monkeypatch):
    phase_output, state = phase_fixture(tmp_path, status="RUNNING")
    monkeypatch.setattr(
        repair.queue, "campaign_remaining", lambda _: ({"attempts": []}, 100)
    )

    with pytest.raises(RuntimeError):
        repair.finish_phase(tmp_path, state, phase_output, 0)

    assert state["models"] == {}
