#!/usr/bin/env python3
"""Bounded range-only continuation of the authorized five-model campaign.

This host-only supervisor waits for the existing ledger writer, then uses the
preparation and HMC masters. It contains no numerical implementation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_performance_repair as preparation

PLAN = ROOT/"docs/plans/surrogate-hmc-range-bearing-flow16-2026-10-16.md"
CAMPAIGN = ROOT/"docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01"
PRIOR = ROOT/"docs/plans/artifacts/surrogate-hmc-literature-priors-20261015-01"
DEFAULT_OUTPUT = ROOT/"docs/plans/artifacts/surrogate-hmc-range-bearing-flow16-20261016-01"
GPU = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"


def process_identity(pid):
    """Start ticks distinguish the intended process from later PID reuse."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if tail[0] == "Z" else tail[19]
    except FileNotFoundError:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger_path = CAMPAIGN/"budget.json"
    identity = process_identity(args.wait_pid)
    state = dict(status="WAITING_FOR_EXISTING_MASTER", plan=str(PLAN),
        command=[sys.executable, *sys.argv], wait_pid=args.wait_pid,
        wait_process_start_ticks=identity, flow_steps=16, particles=4032,
        dtype="float32", tf32=False, prior=preparation.LITERATURE_PRIOR,
        campaign_ledger=str(ledger_path), part_of_master=True,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started_utc=datetime.now(timezone.utc).isoformat(), phases=[])

    def publish(status, **details):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
        preparation.save(output/"state.json", state)

    def execute(command, phase):
        publish("RUNNING", phase=phase, active_command=command)
        begin = time.monotonic()
        with (output/(phase+".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state["phases"].append(dict(phase=phase, command=command,
            returncode=result.returncode, supervisor_wall_seconds=time.monotonic()-begin))
        return result.returncode

    publish("WAITING_FOR_EXISTING_MASTER")
    begin = time.monotonic()
    try:
        while identity is not None and process_identity(args.wait_pid) == identity:
            if time.monotonic()-begin > 43200:
                publish("WAIT_LIMIT_REACHED", next_action="inspect the existing master before resuming")
                return 2
            time.sleep(30)
            publish("WAITING_FOR_EXISTING_MASTER", waiting_seconds=time.monotonic()-begin)
        ledger, available = preparation.campaign_remaining(ledger_path)
        if any(row.get("status") == "RUNNING" for row in ledger["attempts"]):
            publish("RUNNING_ATTEMPT_REQUIRES_RECONCILIATION")
            return 2
        if available <= 60:
            publish("BUDGET_EXHAUSTED", remaining_campaign_seconds=available)
            return 2
        if preparation.campaign_attempts_remaining(ledger) < 3:
            previous = ledger["max_attempts"]
            ledger["max_attempts"] = previous+3
            ledger.setdefault("attempt_limit_revisions", []).append(dict(
                previous=previous, revised=previous+3, plan=str(PLAN),
                reason="owner-selected range-only 16-step repair: tuning, mechanics, HMC; compute cap unchanged",
                utc=datetime.now(timezone.utc).isoformat()))
            preparation.save(ledger_path, ledger)
        scope_plan = json.loads((PRIOR/"all-models-scope-execution.json").read_text())
        scope_plan["scopes"]["range_bearing"].update(tuning_seed_offset=400,
            tuning_timeout_seconds=1800, runtime_precision="uniform",
            control_family="fixed_epsilon1_steps5")
        preparation.save(output/"execution.json", scope_plan)
        command = [sys.executable, str(ROOT/"scripts/run_surrogate_hmc_performance_repair.py"),
            "advance", "--model", "range_bearing", "--output", str(output/"preparation"),
            "--campaign-ledger", str(ledger_path), "--execution-plan", str(output/"execution.json"),
            "--precision-results", str(CAMPAIGN/"fixed-design-full-precision-01/precision.json"),
            "--controls-warm-start-root", str(CAMPAIGN/"literature-prior-other-four-advance-01"),
            "--flow-steps", "16", "--priors", preparation.LITERATURE_PRIOR,
            "--gpu", GPU, "--budget-seconds", str(min(3600., available)),
            "--per-model-timeout", "1800", "--mechanics-timeout", "1800"]
        if execute(command, "preparation"):
            publish("PREPARATION_REPAIR_REQUIRED")
            return 2
        selection = output/"preparation/range_bearing/filter-tuning/range_bearing/filter-selection.json"
        selected = json.loads(selection.read_text())
        if selected["selected"]["controls"]["flow_steps"] != 16:
            raise ValueError("fresh selection did not preserve the owner's 16 flow steps")
        _, available = preparation.campaign_remaining(ledger_path)
        if available <= 60:
            publish("BUDGET_EXHAUSTED", remaining_campaign_seconds=available)
            return 2
        command = [sys.executable, str(ROOT/"scripts/run_surrogate_hmc_five_model_master.py"),
            "continue-advance", "--model", "range_bearing", "--advance-root", str(output/"preparation"),
            "--campaign-ledger", str(ledger_path), "--output", str(output/"hmc"),
            "--budget-seconds", str(min(32400., available)), "--per-model-timeout", "32400",
            "--gpu", GPU, "--priors", preparation.LITERATURE_PRIOR]
        code = execute(command, "inference")
        publish("FINISHED" if code == 0 else "INFERENCE_REPAIR_OR_BUDGET_REQUIRED",
            result=str(output/"hmc/range_bearing/inference/hmc.json"))
        return code
    except Exception as error:
        publish("SUPERVISOR_ERROR", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
