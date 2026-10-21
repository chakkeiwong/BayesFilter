#!/usr/bin/env python3
"""Serial continuation of the authorized computation master; no inference claims."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_performance_repair as master

TERMINAL = {"COMPUTATION_CHECKS_COMPLETE_POSTERIOR_NOT_RUN", "REPAIR_OR_BUDGET_REQUIRED"}
PASS = "CONSUMER_PAIRED_PARITY_PASS_COST_MEASURED"


def read(path):
    return json.loads(Path(path).read_text())


def ready(campaign, advance_dir=None):
    advance_dir = Path(advance_dir) if advance_dir is not None else campaign / "final-scope-advance-01"
    advance = advance_dir / "advance.json"
    if not advance.is_file() or read(advance).get("status") not in TERMINAL:
        return False
    return not any(row.get("status") == "RUNNING"
                   for row in read(campaign / "budget.json")["attempts"])


def comparison_command(campaign, output, name, scope, timeout, advance_dir=None):
    advance_dir = Path(advance_dir) if advance_dir is not None else campaign / "final-scope-advance-01"
    selection = advance_dir / name / "filter-tuning" / name / "filter-selection.json"
    if not selection.is_file() or read(selection).get("selected") is None:
        return None
    return [sys.executable, str(ROOT / "scripts/run_surrogate_hmc_performance_repair.py"),
        "run", "--model", name, "--output", str(output / name),
        "--consumer-cost", "--consumer-paired-only", "--consumer-baseline",
        str(campaign / "baseline-source/bayesfilter/highdim/covariance_proposal_score_batch_tf.py"),
        "--selection", str(selection), "--shared-gaussian-contraction", scope["contraction"],
        "--runtime-precision", scope["runtime_precision"],
        "--tuning-seed-offset", str(scope["tuning_seed_offset"]),
        "--campaign-ledger", str(campaign / "budget.json"),
        "--budget-seconds", str(timeout), "--per-model-timeout", str(timeout)]


def snapshot(path):
    """Observability only. Empty WSL process lists do not prove exclusivity."""
    records = {"utc": datetime.now(timezone.utc).isoformat(),
               "contention_status": "unverified; inspect GPU/process observations"}
    queries = {
        "gpu": ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used",
                "--format=csv,noheader"],
        "processes": ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                      "--format=csv,noheader"]}
    for name, command in queries.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            records[name] = dict(returncode=result.returncode, stdout=result.stdout,
                                 stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            records[name] = {"observation_error": str(exc)}
    master.save(path, records)


def execute(campaign, output, wait_seconds=36000, per_model_seconds=2400,
            *, advance_dir=None, execution_plan=None):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    advance_dir = Path(advance_dir).resolve() if advance_dir is not None else campaign / "final-scope-advance-01"
    execution_plan = Path(execution_plan).resolve() if execution_plan is not None else campaign / "final-scope-execution.json"
    if any(not math.isfinite(v) or v <= 0 for v in (wait_seconds, per_model_seconds)):
        raise ValueError("finite positive wait and per-model limits required")
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="WAITING_FOR_CURRENT_MASTER", campaign=str(campaign),
        source_advance_dir=str(advance_dir), execution_plan=str(execution_plan),
        command=sys.argv, python=sys.executable, created_utc=datetime.now(timezone.utc).isoformat(),
        evidence_contract=str(campaign / "consumer-paired-contract.md"),
        per_model_seconds=per_model_seconds, models=[], posterior_sampling_complete=False)
    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        master.save(output / "next-phase-plan.json", state)
    save()
    deadline = time.monotonic() + wait_seconds
    while not ready(campaign, advance_dir):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            state.update(status="WAIT_LIMIT_REACHED",
                         next="inspect prior controller and remaining campaign budget")
            save()
            return 1
        time.sleep(min(30, remaining))
    scopes = master.execution_scopes(execution_plan)
    for name in master.MODELS:
        ledger, remaining = master.campaign_remaining(campaign / "budget.json")
        if remaining <= 0 or master.campaign_attempts_remaining(ledger) <= 0:
            state.update(status="BUDGET_OR_ATTEMPT_LIMIT_REACHED",
                         next="report completed comparisons and deferred scopes")
            save()
            return 1
        if any(row.get("status") == "RUNNING" for row in ledger["attempts"]):
            raise RuntimeError("another campaign worker started; preserve single-worker execution")
        timeout = min(per_model_seconds, remaining)
        command = comparison_command(campaign, output, name, scopes[name], timeout, advance_dir)
        if command is None:
            state["models"].append(dict(model=name, status="FRESH_SELECTION_UNAVAILABLE",
                next="repair the prior tuning attempt within the unchanged campaign"))
            save()
            continue
        state.update(status="RUNNING", active_model=name,
                     next="finish paired actual-consumer comparison; continue independent models")
        save()
        snapshot(output / (name + "-before.json"))
        started = time.monotonic()
        with (output / (name + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        snapshot(output / (name + "-after.json"))
        artifact = output / name / name / "consumer-paired.json"
        paired = read(artifact) if artifact.is_file() else {}
        state["models"].append(dict(model=name, returncode=result.returncode,
            status=paired.get("status", "COMPARISON_FAILED_OR_TIMED_OUT"),
            maximum_scaled_force_gap=paired.get("maximum_scaled_force_gap"),
            candidate_baseline_warm_ratio=paired.get("candidate_baseline_warm_ratio"),
            wall_seconds=time.monotonic()-started, command=command, artifact=str(artifact)))
        save()
    all_pass = len(state["models"]) == len(master.MODELS) and all(
        row["status"] == PASS and row.get("returncode") == 0 for row in state["models"])
    state.update(status="PAIRED_CONSUMER_CHECKS_COMPLETE" if all_pass else "LOCAL_REPAIR_REQUIRED",
        active_model=None, next="assess numerical parity and descriptive cost; repair failed scopes within budget",
        remaining_campaign_seconds=master.campaign_remaining(campaign / "budget.json")[1])
    save()
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=36000)
    parser.add_argument("--per-model-seconds", type=float, default=2400)
    parser.add_argument("--advance-dir", type=Path)
    parser.add_argument("--execution-plan", type=Path)
    args = parser.parse_args()
    sys.exit(execute(args.campaign, args.output, args.wait_seconds, args.per_model_seconds,
                     advance_dir=args.advance_dir, execution_plan=args.execution_plan))
