"""CPU-only supervisor diagnostics; stub workers provide no HMC evidence."""

from pathlib import Path

import pytest

from scripts import run_surrogate_hmc_48h_extension as queue


def initialize(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "preflight", lambda: {})
    monkeypatch.setattr(queue.time, "time", lambda: 1000.0)
    queue.master.write(
        tmp_path / "manifest.json",
        {
            "supervisor_sources": {},
            "deadline_epoch": 1000.0 + queue.TOTAL_SECONDS,
            "deadline_utc": "2099-01-01T00:00:00+00:00",
        },
    )
    queue.master.write(
        tmp_path / "budget.json",
        {
            "authorized_gpu_wall_seconds": queue.TOTAL_SECONDS,
            "max_attempts": 6,
            "attempts": [],
        },
    )


def fake_worker(monkeypatch, outcomes, seen, *, malformed=False):
    class Worker:
        def __init__(self, command, **kwargs):
            model = command[command.index("--model") + 1]
            seconds = float(command[command.index("--budget-seconds") + 1])
            assert seconds == float(
                command[command.index("--per-model-timeout") + 1]
            )
            assert command[command.index("--gpu") + 1] == queue.GPU
            output = Path(command[command.index("--output") + 1])
            ledger_path = Path(command[command.index("--campaign-ledger") + 1])
            ledger = queue.master.read(ledger_path)
            available = ledger["authorized_gpu_wall_seconds"] - sum(
                row["wall_seconds"] for row in ledger["attempts"]
            )
            assert seconds <= available
            seen.append((model, seconds))
            status = outcomes.get(model, "VERIFIED")
            row = {
                "status": status,
                "wall_seconds": seconds,
                "result": str(output / model / "inference" / "hmc.json"),
            }
            ledger["attempts"].append(row)
            queue.master.write(ledger_path, ledger)
            phase_status = (
                "SELECTED_MODELS_HMC_SAMPLING_COMPLETE"
                if status == "VERIFIED"
                else "INCOMPLETE_SEE_MODEL_STATUS"
            )
            queue.master.write(
                output / "master.json",
                {
                    "status": "INCOMPLETE_WORKER_ERROR" if malformed else phase_status,
                    "models": {model: {"inference": row}},
                },
            )
            self.returncode = 0 if status == "VERIFIED" else 2
            self.pid = 12345

        def poll(self):
            return self.returncode

    monkeypatch.setattr(queue.subprocess, "Popen", Worker)


@pytest.mark.parametrize("model", ["predator_prey", "sir_d18"])
def test_healthy_models_resume_saved_chain(model, tmp_path):
    command = queue.command_for(model, 1234.0, queue.PREPARATION, tmp_path)
    assert command[command.index("--resume-manifest") + 1] == str(
        queue.resume_manifest(model)
    )
    assert "--hmc-target-accept" not in command
    assert "--hmc-initial-step-size" not in command


def test_range_bearing_retunes_without_resuming_vetoed_chain(tmp_path):
    command = queue.command_for(
        "range_bearing", 1234.0, queue.RANGE_PREPARATION, tmp_path
    )
    assert "--resume-manifest" not in command
    assert command[command.index("--hmc-target-accept") + 1] == "0.95"
    assert command[command.index("--hmc-initial-step-size") + 1] == "0.005"


def test_allocation_obeys_remaining_budget_and_deadline():
    assert queue.allocation("predator_prey", 100, 1000, 2000, now=1000) == 100
    assert queue.allocation("range_bearing", 100, 50, 2000, now=1000) == 50
    assert queue.allocation("sir_d18", 100, 500, 2000, now=1000) == 500
    assert queue.allocation("sir_d18", 100, 500, 1100, now=1000) == 40


def test_queue_finishes_all_models_within_total_budget(tmp_path, monkeypatch):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {}, seen)
    assert queue.run(tmp_path) == 0
    assert seen == [
        ("predator_prey", 8 * 3600),
        ("range_bearing", 8 * 3600),
        ("sir_d18", 32 * 3600),
    ]
    state = queue.master.read(tmp_path / "state.json")
    assert state["status"] == "COMPLETE_RECORDED_SAMPLING_RULE"
    assert set(state["models"]) == {row[0] for row in seen}


def test_candidate_health_failure_does_not_abandon_later_model(
    tmp_path, monkeypatch
):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {"range_bearing": "HEALTH_VETO"}, seen)
    assert queue.run(tmp_path) == 2
    assert [row[0] for row in seen] == [
        "predator_prey", "range_bearing", "sir_d18"
    ]
    state = queue.master.read(tmp_path / "state.json")
    assert state["status"] == "INCOMPLETE_SEE_MODEL_RESULTS"
    assert state["models"]["sir_d18"]["status"] == "VERIFIED"


def test_invalid_master_artifact_stops_instead_of_silently_continuing(
    tmp_path, monkeypatch
):
    initialize(tmp_path, monkeypatch)
    seen = []
    fake_worker(monkeypatch, {}, seen, malformed=True)
    assert queue.run(tmp_path) == 2
    assert [row[0] for row in seen] == ["predator_prey"]
    state = queue.master.read(tmp_path / "state.json")
    assert state["status"] == "SUPERVISOR_ERROR"

