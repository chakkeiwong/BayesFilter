#!/usr/bin/env python3
"""Repair an authorized continuation queue while preserving its live worker.

Orchestration only: no TensorFlow import, numerical changes, or new budget.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_48h_extension as queue

master = queue.master
PLAN = ROOT / "docs/plans/surrogate-hmc-48h-queue-repair-2026-10-18.md"
RETRY_NAME = "predator_prey-retry-01"


def process(pid):
    """Linux process identity; zombies count as finished, never as a live job."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return None
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]),
                pgid=int(fields[2]), start_ticks=int(fields[19]),
                command=[part.decode() for part in command if part])


def alive(identity):
    actual = process(identity["pid"])
    return bool(actual and actual["start_ticks"] == identity["start_ticks"]
                and actual["state"] not in ("Z", "X"))


def signal_live(identity, sig):
    if alive(identity):
        try:
            os.kill(identity["pid"], sig)
        except ProcessLookupError:
            pass


@contextmanager
def handoff_lock(receipt_path):
    """Serialize takeover against launcher cancellation and rollback."""
    with receipt_path.with_suffix(".lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def wait_stopped(identity):
    for _ in range(100):
        current = process(identity["pid"])
        if not current or current["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("Supervisor exited before handoff snapshot")
        if current["state"] in ("T", "t"):
            return
        time.sleep(0.05)
    raise RuntimeError("Supervisor did not stop for handoff snapshot")


def terminate_parent(parent, worker):
    """Signal exactly one supervisor, preserving the separately sessioned job."""
    if parent["pid"] == worker["pid"] or parent["pgid"] == worker["pgid"]:
        raise RuntimeError("Supervisor and worker must have separate process groups")
    if not alive(parent) or not alive(worker):
        raise RuntimeError("Handoff requires both saved process identities to be live")
    signal_live(parent, signal.SIGTERM)
    signal_live(parent, signal.SIGCONT)
    for _ in range(100):
        if not alive(parent):
            return
        time.sleep(0.05)
    raise RuntimeError("Previous supervisor did not terminate")


def stop_adopted(worker):
    """Stop only the saved worker's separate process group, tolerating exit races."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not alive(worker):
            return
        try:
            os.killpg(worker["pgid"], sig)
        except ProcessLookupError:
            return
        for _ in range(100):
            if not alive(worker):
                return
            time.sleep(0.1)
    if alive(worker):
        raise RuntimeError("Adopted worker did not stop")


def monitor_adopted(worker, deadline, publish=None):
    """Maintain the existing deadline even if publishing progress has failed."""
    while alive(worker):
        if time.time() >= deadline - 10:
            stop_adopted(worker)
            return False
        if publish is not None:
            publish()
        time.sleep(min(15, max(0, deadline - 10 - time.time())))
    return True


def cancel_handoff(receipt_path, parent, child):
    """Cancel only before takeover; never kill the sole remaining supervisor."""
    with handoff_lock(receipt_path):
        if not alive(parent):
            return False
        if receipt_path.exists():
            receipt = master.read(receipt_path)
            receipt.update(status="CANCELLED_OLD_SUPERVISOR_RESUMED")
            master.write(receipt_path, receipt)
        if child is not None and child.poll() is None:
            queue.stop_group(child)
        signal_live(parent, signal.SIGCONT)
        return True


def validate(output):
    manifest = master.read(output / "manifest.json")
    state = master.read(output / "state.json")
    ledger, _ = queue.campaign_remaining(output / "budget.json")
    if state["status"] != "RUNNING" or state["current_model"] != "range_bearing":
        raise RuntimeError("Expected the original active range-bearing phase")
    if ledger["authorized_gpu_wall_seconds"] != queue.TOTAL_SECONDS or ledger["max_attempts"] != 6:
        raise RuntimeError("Original campaign budget changed")
    if manifest["deadline_epoch"] != ledger["deadline_epoch"] or state["deadline_epoch"] != manifest["deadline_epoch"]:
        raise RuntimeError("Campaign deadlines disagree")
    running = [row for row in ledger["attempts"] if row["status"] == "RUNNING"]
    if len(running) != 1 or running[0]["model"] != "range_bearing":
        raise RuntimeError("Expected exactly one active range-bearing reservation")
    parent, worker = process(state["pid"]), process(state["active_master_pid"])
    if not parent or not worker or not alive(parent) or not alive(worker):
        raise RuntimeError("Expected live supervisor and worker")
    if not any(part.endswith("run_surrogate_hmc_48h_extension.py") for part in parent["command"]):
        raise RuntimeError("Unexpected original supervisor command")
    if worker["command"] != state["active_command"] or worker["ppid"] != parent["pid"]:
        raise RuntimeError("Worker identity or parent changed")
    if parent["pgid"] != parent["pid"] or worker["pgid"] != worker["pid"]:
        raise RuntimeError("Expected separately sessioned supervisor and worker")
    if state["models"].get("predator_prey", {}).get("status") != "STAGE_ERROR":
        raise RuntimeError("Predator–prey is not the recorded failed attempt")
    for path, expected in manifest["supervisor_sources"].items():
        if master.digest(ROOT / path) != expected:
            raise RuntimeError(f"Pinned source changed: {path}")
    return manifest, state, ledger, parent, worker


def retry_command(seconds, output):
    command = queue.command_for("predator_prey", seconds, queue.PREPARATION, output)
    command[command.index("--output") + 1] = str(output / RETRY_NAME)
    return command


def finish_phase(output, state, phase_output, returncode):
    phase = master.read(phase_output / "master.json")
    model = state["current_model"]
    ledger, _ = queue.campaign_remaining(output / "budget.json")
    if any(row["status"] == "RUNNING" for row in ledger["attempts"]):
        raise RuntimeError("Finished phase left a running budget reservation")
    if returncode not in (None, 0, 2) or phase["status"] not in (
            "SELECTED_MODELS_HMC_SAMPLING_COMPLETE", "INCOMPLETE_SEE_MODEL_STATUS"):
        raise RuntimeError(f"Invalid terminal phase for {model}: {returncode}")
    row = phase["models"][model]["inference"]
    state["models"][model] = dict(status=row["status"], returncode=returncode,
                                   exit_code_observed=returncode is not None,
                                   wall_seconds=row.get("wall_seconds"), result=row.get("result"))
    state.pop("active_master_pid", None)


def run(output, receipt_path):
    receipt = master.read(receipt_path)
    manifest = master.read(output / "manifest.json")
    state = receipt["saved_state"]
    child = None
    adopted = receipt["worker"]
    took_over = False
    try:
        if manifest["deadline_epoch"] != receipt["deadline_epoch"]:
            raise RuntimeError("Deadline changed before takeover")
        for source, expected in receipt["sources"].items():
            if master.digest(ROOT / source) != expected:
                raise RuntimeError(f"Handoff source changed: {source}")
        with handoff_lock(receipt_path):
            receipt = master.read(receipt_path)
            if receipt["status"] != "PREPARED":
                raise RuntimeError("Handoff was cancelled or already consumed")
            terminate_parent(receipt["parent"], adopted)
            took_over = True
            receipt.update(status="TAKEN_OVER", supervisor_pid=os.getpid(),
                           takeover_utc=master.stamp())
            master.write(receipt_path, receipt)
        queue.RUNS = (("range_bearing", 8 * 3600, queue.RANGE_PREPARATION),
                      ("predator_prey", 8 * 3600, queue.PREPARATION),
                      ("sir_d18", 32 * 3600, queue.PREPARATION))
        state.setdefault("prior_attempts", []).append(dict(
            model="predator_prey", **state["models"].pop("predator_prey")))
        state.update(pid=os.getpid(), queue_repair_plan=str(PLAN),
                     previous_supervisor_pid=receipt["parent"]["pid"])
        completed = monitor_adopted(adopted, manifest["deadline_epoch"],
                                    lambda: queue.publish(output, state, manifest))
        if not completed:
            queue.close_interrupted_attempt(
                output, time.monotonic() - state["phase_started_monotonic"], "DEADLINE_STOP")
            state["models"]["range_bearing"] = dict(status="DEADLINE_STOP")
        if "range_bearing" not in state["models"]:
            finish_phase(output, state, output / "range_bearing", None)
        state.pop("active_master_pid", None)
        adopted = None
        queue.publish(output, state, manifest)
        for model, cap, root in queue.RUNS[1:]:
            ledger, remaining = queue.campaign_remaining(output / "budget.json")
            seconds = queue.allocation(model, cap, remaining, manifest["deadline_epoch"])
            if seconds <= 60 or queue.campaign_attempts_remaining(ledger) <= 0:
                state["models"][model] = dict(status="CAMPAIGN_BUDGET_STOP")
                queue.publish(output, state, manifest)
                continue
            command = (retry_command(seconds, output) if model == "predator_prey"
                       else queue.command_for(model, seconds, root, output))
            phase_output = Path(command[command.index("--output") + 1])
            if phase_output.exists():
                raise FileExistsError(phase_output)
            state.update(current_model=model, active_command=command,
                         phase_started_monotonic=time.monotonic(), phase_budget_seconds=seconds)
            queue.publish(output, state, manifest)
            with (output / f"{phase_output.name}-master.log").open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["active_master_pid"] = child.pid
                while child.poll() is None:
                    queue.publish(output, state, manifest)
                    if time.time() >= manifest["deadline_epoch"] - 10:
                        queue.stop_group(child)
                        queue.close_interrupted_attempt(
                            output, time.monotonic() - state["phase_started_monotonic"], "DEADLINE_STOP")
                        state["models"][model] = dict(status="DEADLINE_STOP", returncode=child.returncode)
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if model not in state["models"]:
                finish_phase(output, state, phase_output, child.returncode)
            child = None
            state.pop("active_master_pid", None)
            queue.publish(output, state, manifest)
        state.update(current_model=None, status="COMPLETE_RECORDED_SAMPLING_RULE" if all(
            state["models"].get(model, {}).get("status") == "VERIFIED"
            for model, _, _ in queue.RUNS) else "INCOMPLETE_SEE_MODEL_RESULTS")
    except BaseException:
        error = traceback.format_exc()
        if not took_over and alive(receipt["parent"]):
            signal_live(receipt["parent"], signal.SIGCONT)
            receipt.update(status="HANDOFF_FAILED_OLD_SUPERVISOR_RESUMED", error=traceback.format_exc())
            master.write(receipt_path, receipt)
            return 2
        queue.stop_group(child)
        # Once the old supervisor has exited, retain deadline supervision even
        # if progress publication fails. A live worker must not be orphaned.
        if adopted is not None and alive(adopted):
            monitor_adopted(adopted, manifest["deadline_epoch"])
        queue.close_interrupted_attempt(
            output, time.monotonic() - state["phase_started_monotonic"], "SUPERVISOR_ERROR")
        state.update(status="SUPERVISOR_ERROR", error=error)
    state["finished_utc"] = master.stamp()
    queue.publish(output, state, manifest)
    queue.terminal_note(output, state)
    return 0 if state["status"] == "COMPLETE_RECORDED_SAMPLING_RULE" else 2


def launch(output):
    _, _, _, parent, _ = validate(output)
    repair = output / "queue-repair-01"
    repair.mkdir(exist_ok=False)
    frozen = False
    child = None
    receipt = repair / "handoff.json"
    try:
        os.kill(parent["pid"], signal.SIGSTOP)
        frozen = True
        wait_stopped(parent)
        manifest, state, ledger, checked_parent, worker = validate(output)
        if checked_parent["start_ticks"] != parent["start_ticks"]:
            raise RuntimeError("Supervisor changed during handoff")
        master.write(receipt, dict(status="PREPARED", created_utc=master.stamp(),
            parent=parent, worker=worker, saved_state=state, saved_budget=ledger,
            deadline_epoch=manifest["deadline_epoch"], plan=str(PLAN),
            sources={str(path.relative_to(ROOT)): master.digest(path)
                     for path in (Path(__file__).resolve(), PLAN)}))
        with (repair / "supervisor.log").open("x") as log:
            child = subprocess.Popen([str(queue.PYTHON), str(Path(__file__).resolve()),
                "run", "--output", str(output), "--receipt", str(receipt)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "TF_FORCE_GPU_ALLOW_GROWTH": "true"})
        for _ in range(100):
            recorded = master.read(receipt)
            if recorded["status"] == "TAKEN_OVER":
                print(json.dumps(dict(supervisor_pid=child.pid, preserved_worker_pid=worker["pid"],
                                      deadline_utc=manifest["deadline_utc"], receipt=str(receipt))))
                return 0
            if child.poll() is not None:
                raise RuntimeError(f"Successor startup failed: {child.returncode}")
            time.sleep(0.1)
        raise RuntimeError("Successor did not acknowledge takeover")
    except BaseException:
        if frozen:
            cancel_handoff(receipt, parent, child)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "run"))
    parser.add_argument("--output", type=Path, default=queue.OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action == "run" and args.receipt is None:
        parser.error("run requires --receipt")
    raise SystemExit(launch(args.output) if args.action == "launch" else run(args.output, args.receipt))
