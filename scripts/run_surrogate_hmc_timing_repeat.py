#!/usr/bin/env python3
"""Bounded diagnostic timing repeats with GPU telemetry; no inference claims."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_consumer_followups as followups
from scripts import run_surrogate_hmc_performance_repair as master


def observe():
    row = {"utc": datetime.now(timezone.utc).isoformat()}
    queries = {
        "gpu": ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,clocks.sm,power.draw", "--format=csv,noheader"],
        "compute_apps": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    }
    for key, command in queries.items():
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=10)
            row[key] = {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            row[key] = {"error": str(exc)}
    return row


def execute(campaign, output):
    campaign, output = campaign.resolve(), output.resolve()
    if not followups.ready(campaign):
        raise RuntimeError("Prior phase must be terminal with no active ledger worker")
    scope = json.loads((campaign / "final-scope-execution.json").read_text())["scopes"]["ksc"]
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "command": sys.argv, "python": sys.executable,
             "plan": str(campaign / "ksc-timing-repeat-contract.md"), "attempts": [],
             "contention_status": "observations_only_not_proof_of_exclusivity"}
    for index in (1, 2):
        if not followups.ready(campaign):
            state["status"] = "LEDGER_BUSY_OR_PREVIOUS_PHASE_NOT_READY"
            break
        parent = output / ("repeat-%02d" % index)
        parent.mkdir()
        command = followups.comparison_command(campaign, parent, "ksc", scope, 2400)
        if command is None:
            raise RuntimeError("Missing frozen KSC selection")
        record = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
        state["attempts"].append(record)
        master.save(output / "next-phase-plan.json", state)
        start = time.monotonic()
        with (parent / "master.log").open("w") as log, (parent / "telemetry.jsonl").open("w") as telemetry:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while True:
                telemetry.write(json.dumps(observe()) + "\n")
                telemetry.flush()
                try:
                    code = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    continue
        record.update(exit_code=code, wall_seconds=time.monotonic()-start)
        result_path = parent / "ksc/ksc/consumer-paired.json"
        record["result"] = str(result_path)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            record["status"] = result["status"]
        if code != 0 or record.get("status") != followups.PASS:
            state["status"] = "REPAIR_REQUIRED"
            break
    else:
        state["status"] = "TIMING_REPEATS_COMPLETE"
    state["next"] = "Assess KSC timings with telemetry; continue the existing master cost review."
    master.save(output / "next-phase-plan.json", state)
    return 0 if state["status"] == "TIMING_REPEATS_COMPLETE" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(execute(args.campaign, args.output))
