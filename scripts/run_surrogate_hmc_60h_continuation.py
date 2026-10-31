#!/usr/bin/env python3
"""Bounded continuation of the existing surrogate-HMC master; no numerical code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_five_model_master as master
from scripts.run_surrogate_hmc_performance_repair import campaign_remaining, campaign_attempts_remaining

PYTHON = master.PYTHON
PLAN = ROOT / "docs/plans/surrogate-hmc-60h-continuation-2026-10-16.md"
ARTIFACTS = ROOT / "docs/plans/artifacts"
PREVIOUS = ARTIFACTS / "surrogate-hmc-cost-repair-24h-20261013-01"
OUTPUT = ARTIFACTS / "surrogate-hmc-60h-continuation-20261016-01"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
RUNS = (
    ("range_bearing", 8 * 3600, ARTIFACTS / "surrogate-hmc-range-bearing-flow16-20261016-01/preparation"),
    ("predator_prey", 16 * 3600, PREVIOUS / "literature-prior-other-four-advance-01"),
    ("sir_d18", 36 * 3600, PREVIOUS / "literature-prior-other-four-advance-01"),
)
TOTAL_SECONDS = 60 * 3600


def preflight():
    prior, _ = campaign_remaining(PREVIOUS / "budget.json")
    if any(row.get("status") == "RUNNING" for row in prior["attempts"]):
        raise ValueError("previous campaign still has an active worker")
    prepared = {}
    for model, _, root in RUNS:
        prepared.update(master.advance_prepared_selections([root], (model,), master.LITERATURE_PRIOR))
    selected = master.read(prepared["range_bearing"]["selection"])
    if selected["selected"]["controls"]["flow_steps"] != 16:
        raise ValueError("range-bearing continuation requires the owner's 16 flow steps")
    return prepared


def command_for(model, seconds, prepared_root, output, *, phase_output=None):
    return [str(PYTHON), str(ROOT / "scripts/run_surrogate_hmc_five_model_master.py"),
        "continue-advance", "--model", model, "--advance-root", str(prepared_root),
        "--campaign-ledger", str(output / "budget.json"), "--output", str(phase_output or output / model),
        "--budget-seconds", str(seconds), "--per-model-timeout", str(seconds),
        "--gpu", GPU, "--priors", master.LITERATURE_PRIOR]


def publish(output, state):
    ledger, unreserved = campaign_remaining(output / "budget.json")
    now = time.monotonic()
    charged = sum(float(row.get("wall_seconds", max(0., now - state["phase_started_monotonic"])
        if row.get("status") == "RUNNING" else row.get("reserved_seconds", 0.)))
        for row in ledger["attempts"])
    state.update(updated_utc=master.stamp(), consumed_worker_seconds_estimate=charged,
        remaining_worker_seconds_estimate=max(0., TOTAL_SECONDS - charged),
        unreserved_worker_seconds=unreserved)
    master.write(output / "state.json", state)
    pending = [model for model, _, _ in RUNS if model not in state["models"]]
    master.write(output / "next-phase-plan.json", dict(status=state["status"],
        updated_utc=state["updated_utc"], part_of_master=True,
        automatic_continuation=state["status"] == "RUNNING",
        current_model=state.get("current_model"), pending_models=pending,
        action="finish current adaptive phase, then run the next listed model" if pending
            else "review saved model diagnostics and remaining gaps",
        remaining_worker_seconds_estimate=state["remaining_worker_seconds_estimate"],
        warmup="2000 minimum; latest1000 modern R-hat <=1.05; maximum10000 per chain",
        retained="cumulative modern R-hat <=1.01 or maximum20000 per chain",
        redundant_validation_phase=False, models=state["models"]))


def terminal_note(output, state):
    lines = ["# Surrogate-HMC 60-hour continuation result", "",
        f"Updated {master.stamp()}. Status: {state['status']}.", "",
        "| Model | Recorded outcome | Evidence |", "|---|---|---|"]
    for model, _, _ in RUNS:
        row = state["models"].get(model, {})
        lines.append(f"| {model} | {row.get('status', 'NOT_RUN')} | {model}/master.json |")
    lines += ["", "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
        "|---|---|---|---|---|---|",
        "| Preserve all attempts | Only VERIFIED records complete the declared sampling rule | Inspect each hmc.json; budget stops are incomplete evidence | Posterior coverage and endpoint approximation | Review failed candidates and unused budget; continue authorized repairs where justified | General posterior correctness or superiority |",
        "", "| Inference status | Assessment |", "|---|---|",
        "| Hard veto screen | Per-model worker results and logs control; no silent reinterpretation |",
        "| Statistically supported ranking | None; no method comparison was run |",
        "| Descriptive-only differences | Acceptance, ESS, R-hat and runtime require their model-specific context |",
        "| Default-readiness | Not established by this continuation |",
        "| Next evidence needed | Review cumulative diagnostics and any incomplete model phases |",
        "", "Post-run limitation: numerical health and cumulative R-hat do not establish",
        "complete posterior exploration or exactness of an approximate endpoint.",
        "LGSSM/KSC results remain in the preceding campaign. No extra validation chain was added.", ""]
    (output / "result.md").write_text("\n".join(lines))


def run(output):
    manifest = master.read(output / "manifest.json")
    for path, expected in manifest["supervisor_sources"].items():
        if master.digest(ROOT / path) != expected:
            raise ValueError("continuation supervisor changed after launch: " + path)
    state = dict(status="RUNNING", started_utc=master.stamp(), pid=os.getpid(),
        plan=str(PLAN), current_model=None, models={}, phase_started_monotonic=time.monotonic())
    active = None
    try:
        preflight()
        publish(output, state)
        for model, cap, prepared_root in RUNS:
            _, remaining = campaign_remaining(output / "budget.json")
            seconds = min(cap, remaining)
            if seconds <= 60:
                state["models"][model] = dict(status="CAMPAIGN_BUDGET_STOP")
                continue
            command = command_for(model, seconds, prepared_root, output)
            state.update(current_model=model, active_command=command,
                phase_started_monotonic=time.monotonic(), active_model_budget_seconds=seconds)
            publish(output, state)
            with (output / (model + "-master.log")).open("w") as log:
                active = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["active_master_pid"] = active.pid
                while active.poll() is None:
                    publish(output, state)
                    try:
                        active.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        continue
            code = active.returncode
            active = None
            phase = master.read(output / model / "master.json")
            row = phase["models"][model]["inference"]
            ledger, _ = campaign_remaining(output / "budget.json")
            if any(item.get("status") == "RUNNING" for item in ledger["attempts"]):
                raise RuntimeError("finished master left an unresolved worker charge")
            if code not in (0, 2) or phase["status"] not in (
                    "SELECTED_MODELS_HMC_SAMPLING_COMPLETE", "INCOMPLETE_SEE_MODEL_STATUS"):
                raise RuntimeError("inference infrastructure failed; inspect " + str(output / model))
            state["models"][model] = dict(status=row["status"], returncode=code,
                wall_seconds=row.get("wall_seconds"), result=row.get("result"))
            publish(output, state)
        state.update(status="COMPLETE_RECORDED_SAMPLING_RULE" if all(
            row["status"] == "VERIFIED" for row in state["models"].values())
            else "INCOMPLETE_SEE_MODEL_RESULTS", current_model=None, finished_utc=master.stamp())
    except BaseException:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=10)
        state.update(status="SUPERVISOR_ERROR", traceback=traceback.format_exc(), finished_utc=master.stamp())
        raise
    finally:
        state.pop("active_master_pid", None)
        publish(output, state)
        if state["status"] != "RUNNING":
            terminal_note(output, state)
    return 0 if state["status"] == "COMPLETE_RECORDED_SAMPLING_RULE" else 2


def launch(output):
    prepared = preflight()
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total",
        "--format=csv,noheader"], text=True)
    if GPU not in devices:
        raise RuntimeError("prepared GPU is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    previous, _ = campaign_remaining(PREVIOUS / "budget.json")
    master.write(output / "budget.json", dict(authorized_gpu_wall_seconds=TOTAL_SECONDS,
        max_attempts=6, attempts=[], used_seconds=0., remaining_seconds=TOTAL_SECONDS,
        plan=str(PLAN), authorization="owner agreed to the proposed additional 60 GPU-hours",
        previous_ledger=str(PREVIOUS / "budget.json"),
        previous_charged_seconds=previous["used_seconds"], allocations=dict((m, cap) for m, cap, _ in RUNS)))
    paths = (Path(__file__), ROOT / "scripts/run_surrogate_hmc_five_model_master.py",
        ROOT / "scripts/run_surrogate_hmc_performance_repair.py", PLAN)
    master.write(output / "manifest.json", dict(started_utc=master.stamp(), plan=str(PLAN),
        command=[sys.executable, *sys.argv], environment=str(PYTHON), gpu_uuid=GPU,
        device_inventory=devices, cpu_only=False, gpu_trust_basis="escalated_local_GPU_execution",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        prepared=prepared, supervisor_sources={str(p.relative_to(ROOT)): master.digest(p) for p in paths},
        total_gpu_worker_seconds=TOTAL_SECONDS, previous_results_preserved=True,
        worker_manifests="<model>/<model>/inference/manifest.json",
        memory_policy="workers require and verify memory growth before device initialization"))
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
    command = [str(PYTHON), str(Path(__file__).resolve()), "run", "--output", str(output)]
    with (output / "supervisor.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    master.write(output / "launcher.json", dict(pid=child.pid, command=command, started_utc=master.stamp()))
    return dict(status="LAUNCHED", pid=child.pid, output=str(output))


RETRY_PLAN = ROOT / "docs/plans/surrogate-hmc-force-fallback-repair-2026-10-16.md"


def _pid_running(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        # A detached child's zombie can outlive its completed work.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def retry_allowance(output):
    ledger, remaining = campaign_remaining(output / "budget.json")
    if ledger["authorized_gpu_wall_seconds"] != TOTAL_SECONDS:
        raise ValueError("retry must use the approved 60-hour ledger")
    used = sum(float(row.get("wall_seconds", row.get("reserved_seconds", 0.)))
               for row in ledger["attempts"] if row.get("model") == "range_bearing")
    cap = next(cap for model, cap, _ in RUNS if model == "range_bearing")
    return ledger, min(remaining, max(0., cap - used))


def retry_ready(output):
    parent = master.read(output / "state.json")
    ledger, _ = retry_allowance(output)
    alive = _pid_running(parent.get("pid"))
    if parent["status"] == "RUNNING":
        if not alive:
            raise RuntimeError("parent supervisor died without a terminal receipt")
        return False
    if parent["status"] not in ("COMPLETE_RECORDED_SAMPLING_RULE", "INCOMPLETE_SEE_MODEL_RESULTS"):
        raise RuntimeError("parent supervisor ended with an infrastructure error")
    if alive:
        return False
    if any(row.get("status") == "RUNNING" for row in ledger["attempts"]):
        raise RuntimeError("unresolved active budget entry; retry cannot start")
    return True


def enqueue_range_bearing(output):
    prepared = preflight()
    ledger, seconds = retry_allowance(output)
    if seconds <= 60 or campaign_attempts_remaining(ledger) <= 0:
        raise ValueError("no authorized range-bearing retry budget or attempt remains")
    parent = master.read(output / "state.json")
    retry_ready(output)  # Validate parent health; it may legitimately still be running.
    queue_path = output / "queued-range-bearing.json"
    if queue_path.exists():
        raise FileExistsError("range-bearing retry is already queued")
    retry_root = output / "range-bearing-retry-01"
    retry_root.mkdir(parents=True, exist_ok=False)
    paths = list(master.sources("range_bearing")) + [Path(__file__), RETRY_PLAN,
        ROOT / "scripts/run_surrogate_hmc_performance_repair.py",
        ROOT / "bayesfilter/inference/hmc_tensorflow_resume.py"]
    master.write(retry_root / "manifest.json", dict(
        plan=str(RETRY_PLAN), parent_output=str(output), prepared=prepared["range_bearing"],
        command=[sys.executable, *sys.argv], python=str(PYTHON), gpu_uuid=GPU,
        cpu_only=False, gpu_trust_basis="escalated_local_GPU_execution",
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): master.digest(p) for p in paths},
        maximum_retry_seconds=seconds, campaign_ledger=str(output / "budget.json"),
        original_range_result=parent.get("models", {}).get("range_bearing"),
        parent_active_master_pid=parent.get("active_master_pid"),
        active_worker_policy="already loaded code is unchanged; new workers load repaired controller"))
    state = dict(status="QUEUED", model="range_bearing", part_of_master=True,
        queued_utc=master.stamp(), output=str(retry_root), plan=str(RETRY_PLAN),
        after="current predator-prey and SIR queue", maximum_worker_seconds=seconds,
        force_fallback_policy="warn_deterministic_position_force_v1")
    master.write(queue_path, state)
    command = [str(PYTHON), str(Path(__file__).resolve()), "run-range-bearing-retry", "--output", str(output)]
    try:
        with (retry_root / "supervisor.log").open("w") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    except BaseException:
        state.update(status="RETRY_LAUNCH_ERROR", traceback=traceback.format_exc())
        master.write(queue_path, state)
        raise
    master.write(retry_root / "launcher.json", dict(pid=child.pid, command=command, started_utc=master.stamp()))
    return dict(status="QUEUED", pid=child.pid, queue=str(queue_path), maximum_worker_seconds=seconds)


def run_range_bearing_retry(output):
    queue_path = output / "queued-range-bearing.json"
    state = master.read(queue_path)
    if state["status"] != "QUEUED":
        raise ValueError("retry is already running or terminal")
    root = Path(state["output"])
    manifest = master.read(root / "manifest.json")
    active = None

    def save():
        state.update(updated_utc=master.stamp(), pid=os.getpid())
        master.write(queue_path, state)

    try:
        while not retry_ready(output):
            state.update(status="WAITING_FOR_CURRENT_QUEUE")
            save()
            time.sleep(30)
        for path, expected in manifest["source_sha256"].items():
            if master.digest(ROOT / path) != expected:
                raise ValueError("queued retry source changed: " + path)
        preflight()
        ledger, seconds = retry_allowance(output)
        seconds = min(seconds, manifest["maximum_retry_seconds"])
        if seconds <= 60 or campaign_attempts_remaining(ledger) <= 0:
            state.update(status="CAMPAIGN_BUDGET_OR_ATTEMPT_STOP")
            return 2
        prepared_root = next(p for m, _, p in RUNS if m == "range_bearing")
        command = command_for("range_bearing", seconds, prepared_root, output, phase_output=root / "run")
        state.update(status="RUNNING", started_utc=master.stamp(), allowed_worker_seconds=seconds,
                     command=command)
        save()
        with (root / "master.log").open("w") as log:
            active = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state["active_master_pid"] = active.pid
            save()
            while active.poll() is None:
                master.write(output / "next-phase-plan.json", dict(state,
                    automatic_continuation=True, current_model="range_bearing", pending_models=[],
                    action="continue repaired range-bearing adaptive sampling", redundant_validation_phase=False))
                time.sleep(30)
        result = master.read(root / "run/master.json")
        ledger, _ = retry_allowance(output)
        if (active.returncode not in (0, 2) or result["status"] not in (
                "SELECTED_MODELS_HMC_SAMPLING_COMPLETE", "INCOMPLETE_SEE_MODEL_STATUS")
                or any(row.get("status") == "RUNNING" for row in ledger["attempts"])):
            raise RuntimeError("retry master or budget finalization failed")
        row = result["models"]["range_bearing"]["inference"]
        state.update(status=row["status"], result=row, returncode=active.returncode)
        return 0 if row["status"] == "VERIFIED" else 2
    except BaseException:
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=10)
        state.update(status="RETRY_SUPERVISOR_ERROR", traceback=traceback.format_exc())
        raise
    finally:
        state.update(finished_utc=master.stamp())
        state.pop("active_master_pid", None)
        save()
        master.write(root / "result.json", state)
        # The queue record remains authoritative even when parent recovery is needed.
        if not _pid_running(master.read(output / "state.json").get("pid")):
            master.write(output / "next-phase-plan.json", dict(state,
                automatic_continuation=False, action="review saved model diagnostics and remaining gaps",
                redundant_validation_phase=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "launch", "run", "status",
                                            "enqueue-range-bearing", "run-range-bearing-retry"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.command == "check":
        result = {name: dict(status="PASS", selection=row["selection"],
            selection_hash=row["selection_hash"]) for name, row in preflight().items()}
    elif args.command == "launch":
        result = launch(output)
    elif args.command == "run":
        return run(output)
    elif args.command == "enqueue-range-bearing":
        result = enqueue_range_bearing(output)
    elif args.command == "run-range-bearing-retry":
        return run_range_bearing_retry(output)
    else:
        result = master.read(output / "state.json")
        result = {k: result.get(k) for k in ("status", "pid", "current_model", "active_master_pid",
            "updated_utc", "remaining_worker_seconds_estimate", "models")}
        if (output / "queued-range-bearing.json").exists():
            result["range_bearing_retry"] = master.read(output / "queued-range-bearing.json")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
