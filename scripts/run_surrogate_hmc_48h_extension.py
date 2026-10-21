#!/usr/bin/env python3
"""Bounded continuation of the existing five-model surrogate-HMC campaign.

This orchestration layer changes no numerical kernel. See the linked plan for
the sampling-rule interpretation, preserved target and additional time budget.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from scripts import run_surrogate_hmc_five_model_master as master
from scripts.run_surrogate_hmc_performance_repair import (
    campaign_attempts_remaining,
    campaign_remaining,
)

PYTHON = master.PYTHON
PLAN = ROOT / "docs/plans/surrogate-hmc-48h-extension-2026-10-18.md"
ARTIFACTS = ROOT / "docs/plans/artifacts"
PREVIOUS = ARTIFACTS / "surrogate-hmc-60h-continuation-20261016-01"
PREPARATION = (ARTIFACTS / "surrogate-hmc-cost-repair-24h-20261013-01"
               / "literature-prior-other-four-advance-01")
RANGE_PREPARATION = (ARTIFACTS / "surrogate-hmc-range-bearing-flow16-20261016-01"
                     / "preparation")
OUTPUT = ARTIFACTS / "surrogate-hmc-48h-extension-20261018-01/run"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
TOTAL_SECONDS = 48 * 3600
RUNS = (
    ("predator_prey", 8 * 3600, PREPARATION),
    ("range_bearing", 8 * 3600, RANGE_PREPARATION),
    ("sir_d18", 32 * 3600, PREPARATION),
)


def resume_manifest(model):
    return PREVIOUS / model / model / "inference/public-tuning/manifest.json"


def preflight():
    prior, _ = campaign_remaining(PREVIOUS / "budget.json")
    if any(row.get("status") == "RUNNING" for row in prior["attempts"]):
        raise RuntimeError("Previous campaign still records a running attempt")
    prepared = {}
    for model, _, root in RUNS:
        prepared.update(master.advance_prepared_selections(
            [root], (model,), master.LITERATURE_PRIOR))
        if model != "range_bearing" and not resume_manifest(model).is_file():
            raise FileNotFoundError(resume_manifest(model))
    selection = master.read(Path(prepared["range_bearing"]["selection"]))
    if selection["selected"]["controls"]["flow_steps"] != 16:
        raise RuntimeError("Range bearing must retain its prepared 16-flow-step scope")
    return prepared


def command_for(model, seconds, prepared_root, output):
    command = [
        str(PYTHON), str(ROOT / "scripts/run_surrogate_hmc_five_model_master.py"),
        "continue-advance", "--model", model, "--advance-root", str(prepared_root),
        "--campaign-ledger", str(output / "budget.json"),
        "--output", str(output / model), "--budget-seconds", str(seconds),
        "--per-model-timeout", str(seconds), "--gpu", GPU,
        "--priors", master.LITERATURE_PRIOR,
    ]
    if model == "range_bearing":
        command += ["--hmc-target-accept", "0.70", "--hmc-initial-step-size", "0.005"]
    else:
        command += ["--resume-manifest", str(resume_manifest(model))]
    return command


def allocation(model, cap, remaining, deadline, now=None):
    """Reserve a bounded phase; unused earlier allocations belong to SIR."""
    seconds_left = deadline - (time.time() if now is None else now) - 60.0
    return min(remaining if model == "sir_d18" else cap, remaining, seconds_left)


def publish(output, state, manifest):
    ledger, unreserved = campaign_remaining(output / "budget.json")
    elapsed = max(0.0, time.monotonic() - state["phase_started_monotonic"])
    charged = sum(float(row.get("wall_seconds", elapsed if row.get("status") == "RUNNING"
                                else row.get("reserved_seconds", 0.0)))
                  for row in ledger["attempts"])
    state.update(updated_utc=master.stamp(), consumed_worker_seconds_estimate=charged,
                 remaining_worker_seconds_estimate=max(0.0, TOTAL_SECONDS - charged),
                 unreserved_worker_seconds=unreserved)
    master.write(output / "state.json", state)
    pending = [model for model, _, _ in RUNS if model not in state["models"]]
    master.write(output / "next-phase-plan.json", dict(
        status=state["status"], updated_utc=state["updated_utc"], part_of_master=True,
        automatic_continuation=state["status"] == "RUNNING",
        current_model=state.get("current_model"), pending_models=pending,
        action="Finish current adaptive phase, then advance through the saved queue",
        remaining_worker_seconds_estimate=state["remaining_worker_seconds_estimate"],
        deadline_utc=manifest["deadline_utc"], deadline_epoch=manifest["deadline_epoch"],
        warmup="minimum 2000; latest 1000 modern R-hat <=1.05; maximum 10000 per chain",
        retained="cumulative modern R-hat <=1.01 or maximum 20000 per chain",
        redundant_validation_phase=False, models=state["models"],
    ))


def stop_group(child):
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)


def close_interrupted_attempt(output, elapsed, status):
    ledger = master.read(output / "budget.json")
    for row in ledger["attempts"]:
        if row.get("status") == "RUNNING":
            row.update(status=status, finished_utc=master.stamp(),
                       wall_seconds=min(float(row["reserved_seconds"]), max(0.0, elapsed)))
    used = sum(float(row.get("wall_seconds", row.get("reserved_seconds", 0.0)))
               for row in ledger["attempts"])
    ledger.update(used_gpu_wall_seconds=used,
                  remaining_gpu_wall_seconds=max(0.0, TOTAL_SECONDS - used))
    master.write(output / "budget.json", ledger)


def terminal_note(output, state):
    rows = ["| Model | Outcome | Evidence |", "|---|---|---|",
            "| LGSSM | Saved sampling rule passed; preserved | Earlier literature-prior run |",
            "| KSC SV | Saved sampling rule passed; preserved | Earlier literature-prior run |"]
    for model, _, _ in RUNS:
        row = state["models"].get(model, {})
        evidence = row.get("result") or str(output / model / model / "inference/hmc.json")
        rows.append(f"| {model} | {row.get('status', 'NOT_RUN')} | {evidence} |")
    text = (f"# Surrogate-HMC 48-hour extension result\n\nStatus: {state['status']}. "
            f"Updated {master.stamp()}.\n\n" + "\n".join(rows) + "\n\n"
            "| Decision field | Finding |\n|---|---|\n"
            f"| Decision | {state['status']} |\n"
            "| Primary criterion | Recorded adaptive sampling rule; per-model artifacts carry the decision |\n"
            "| Veto diagnostics | Existing health rules retained; a health-vetoed candidate is rejected |\n"
            "| Main uncertainty | Mixing within time allocation; posterior/reference agreement not checked |\n"
            "| Next justified action | Use individual terminal status: complete, time-limited, or candidate repair |\n"
            "| Not concluded | Exact-posterior accuracy, heuristic dominance or method superiority |\n\n"
            "| Inference status | Finding |\n|---|---|\n"
            "| Hard veto screen | Read each hmc.json; no health threshold was relaxed |\n"
            "| Statistically supported ranking | None; this is a continuation, not a comparison |\n"
            "| Descriptive-only differences | Acceptance, ESS, intermediate R-hat and runtime |\n"
            "| Default-readiness | Not established by this extension |\n"
            "| Next evidence needed | Unfinished sampling and independent posterior agreement, as applicable |\n\n"
            "No extra validation chain was run. VERIFIED means the recorded sampling rule passed. "
            "A rejected candidate does not reject the research direction. All prior attempts are preserved. "
            "The strongest alternative explanation for an incomplete result is insufficient time or "
            "unfavorable HMC geometry; exact-model posterior agreement remains unevaluated.\n")
    (output / "result.md").write_text(text)


def run(output):
    manifest = master.read(output / "manifest.json")
    for path, expected in manifest["supervisor_sources"].items():
        if master.digest(ROOT / path) != expected:
            raise RuntimeError(f"Supervisor source changed before execution: {path}")
    state = dict(status="RUNNING", started_utc=master.stamp(), pid=os.getpid(),
                 plan=str(PLAN), models={}, current_model=None,
                 phase_started_monotonic=time.monotonic(),
                 deadline_utc=manifest["deadline_utc"], deadline_epoch=manifest["deadline_epoch"])
    child = None
    try:
        preflight()
        publish(output, state, manifest)
        for model, cap, root in RUNS:
            ledger, remaining = campaign_remaining(output / "budget.json")
            seconds = allocation(model, cap, remaining, manifest["deadline_epoch"])
            if seconds <= 60 or campaign_attempts_remaining(ledger) <= 0:
                state["models"][model] = dict(status="CAMPAIGN_BUDGET_STOP")
                publish(output, state, manifest)
                continue
            command = command_for(model, seconds, root, output)
            state.update(current_model=model, active_command=command,
                         phase_started_monotonic=time.monotonic(), phase_budget_seconds=seconds)
            publish(output, state, manifest)
            with (output / f"{model}-master.log").open("w") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["active_master_pid"] = child.pid
                while child.poll() is None:
                    publish(output, state, manifest)
                    if time.time() >= manifest["deadline_epoch"] - 10:
                        stop_group(child)
                        close_interrupted_attempt(output, time.monotonic() - state["phase_started_monotonic"],
                                                  "DEADLINE_STOP")
                        state["models"][model] = dict(status="DEADLINE_STOP", returncode=child.returncode)
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
            if model not in state["models"]:
                phase = master.read(output / model / "master.json")
                row = phase["models"][model]["inference"]
                ledger, _ = campaign_remaining(output / "budget.json")
                if any(item.get("status") == "RUNNING" for item in ledger["attempts"]):
                    raise RuntimeError("Finished master left a running ledger entry")
                if child.returncode not in (0, 2) or phase["status"] not in (
                        "SELECTED_MODELS_HMC_SAMPLING_COMPLETE", "INCOMPLETE_SEE_MODEL_STATUS"):
                    raise RuntimeError(f"Unexpected master failure for {model}: {child.returncode}")
                state["models"][model] = dict(status=row["status"], returncode=child.returncode,
                                              wall_seconds=row.get("wall_seconds"), result=row.get("result"))
            state.pop("active_master_pid", None)
            child = None
            publish(output, state, manifest)
        state.update(current_model=None,
                     status="COMPLETE_RECORDED_SAMPLING_RULE" if all(
                         state["models"].get(model, {}).get("status") == "VERIFIED"
                         for model, _, _ in RUNS) else "INCOMPLETE_SEE_MODEL_RESULTS")
    except BaseException:
        stop_group(child)
        close_interrupted_attempt(output, time.monotonic() - state["phase_started_monotonic"],
                                  "SUPERVISOR_ERROR")
        state.update(status="SUPERVISOR_ERROR", error=traceback.format_exc())
    finally:
        state.pop("active_master_pid", None)
        state["finished_utc"] = master.stamp()
        publish(output, state, manifest)
        terminal_note(output, state)
    return 0 if state["status"] == "COMPLETE_RECORDED_SAMPLING_RULE" else 2


def launch(output):
    prepared = preflight()
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total", "--format=csv,noheader"], text=True)
    if GPU not in inventory:
        raise RuntimeError("Requested GPU is absent from trusted device inventory")
    output.mkdir(parents=True, exist_ok=False)
    prior, _ = campaign_remaining(PREVIOUS / "budget.json")
    deadline = time.time() + TOTAL_SECONDS
    deadline_utc = dt.datetime.fromtimestamp(deadline, dt.timezone.utc).isoformat()
    master.write(output / "budget.json", dict(
        authorized_gpu_wall_seconds=TOTAL_SECONDS, max_attempts=6, attempts=[],
        used_gpu_wall_seconds=0.0, remaining_gpu_wall_seconds=TOTAL_SECONDS,
        plan=str(PLAN), authorization="Owner requested another 48 hours and continuation on 2026-10-18",
        previous_ledger=str(PREVIOUS / "budget.json"),
        previous_charged_seconds=sum(float(row.get("wall_seconds", row.get("reserved_seconds", 0.0)))
                                     for row in prior["attempts"]),
        previous_budget_carried_forward=False, deadline_epoch=deadline, deadline_utc=deadline_utc,
        allocations="predator_prey <=8h; fresh range_bearing <=8h; sir_d18 all remaining"))
    sources = [Path(__file__).resolve(), PLAN,
               ROOT / "bayesfilter/testing/surrogate_hmc_scope_source_compatibility.py",
               ROOT / "scripts/run_surrogate_hmc_five_model_master.py",
               ROOT / "scripts/run_surrogate_hmc_performance_repair.py"]
    master.write(output / "manifest.json", dict(
        started_utc=master.stamp(), plan=str(PLAN), command=[sys.executable, *sys.argv],
        environment=str(PYTHON), git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        gpu_uuid=GPU, device_inventory=inventory.strip(), cpu_only=False,
        gpu_trust="escalated_gpu_launch", prepared=prepared,
        supervisor_sources={str(path.relative_to(ROOT)): master.digest(path) for path in sources},
        total_gpu_worker_seconds=TOTAL_SECONDS, previous_results_preserved=True,
        worker_manifests="<model>/<model>/inference/manifest.json",
        memory_policy="Workers verify and record memory growth before GPU initialization",
        deadline_epoch=deadline, deadline_utc=deadline_utc))
    command = [str(PYTHON), str(Path(__file__).resolve()), "run", "--output", str(output)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", TF_FORCE_GPU_ALLOW_GROWTH="true")
    with (output / "supervisor.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result = dict(status="LAUNCHED", pid=child.pid, command=command, output=str(output),
                  started_utc=master.stamp(), deadline_utc=deadline_utc)
    master.write(output / "launcher.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "launch", "run", "status"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.action == "run":
        return run(args.output)
    if args.action == "check":
        result = dict(status="PREFLIGHT_PASSED", models=list(preflight()),
                      total_worker_seconds=TOTAL_SECONDS, output=str(args.output), plan=str(PLAN))
    elif args.action == "launch":
        result = launch(args.output)
    else:
        result = master.read(args.output / "state.json")
        result = {key: result.get(key) for key in (
            "status", "pid", "active_master_pid", "current_model", "updated_utc", "deadline_utc",
            "remaining_worker_seconds_estimate", "models", "error")}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
