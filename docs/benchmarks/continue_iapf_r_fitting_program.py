"""Budgeted guide diagnosis and separate controller experiment; CPU R reference."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker
from run_iapf_r_fitting_comparison import ROOT, OUT, WORKER, inspect, read_csv, write_json

CONTROLLER_PLAN = "docs/plans/iapf-r-controller-window-comparison-2026-09-21.md"
GUIDE_WORKER = "docs/benchmarks/diagnose_iapf_r_fitted_guides.R"


def main():
    ledger = json.loads((OUT/"manifest.json").read_text())
    if ledger["status"] != "planned workers complete; terminal analysis and review pending" or ledger.get("inflight"):
        raise RuntimeError("fitting workers must finish before the next stage")
    for name, expected in ledger["source_hashes"].items():
        if hashlib.sha256((OUT/"source-snapshot"/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("original fitting source snapshot changed")
    sources = [ledger["plan"], CONTROLLER_PLAN, GUIDE_WORKER, WORKER,
               "docs/benchmarks/reference_iapf_paper.R", "docs/benchmarks/reference_iapf_plausible_choices.R",
               "docs/benchmarks/diagnose_iapf_r_validation_tails.R",
               "docs/benchmarks/run_iapf_r_replication.py", "docs/benchmarks/run_iapf_r_fitting_comparison.py",
               "docs/benchmarks/continue_iapf_r_fitting_program.py",
               "docs/benchmarks/summarize_iapf_r_fitting_comparison.py", "tests/reference_iapf_controller_window.R"]
    snapshots = OUT/"followthrough-source-snapshot"
    if snapshots.exists():
        raise RuntimeError("follow-through already launched")
    hashes = snapshot_sources(ROOT, sources, snapshots)
    ledger["followthrough"] = dict(plan=CONTROLLER_PLAN, source_hashes=hashes,
                                  transferred_remaining_seconds=ledger["remaining_seconds"],
                                  original_allocation_unchanged=True,
                                  numerical_default_unchanged=True,
                                  implementation_change="optional stopping window, default k+1, shared controller")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def checkpoint(stage):
        ledger["status"] = stage
        ledger["worker_seconds"] = sum(a["worker_seconds"] for a in ledger["attempts"])
        ledger["remaining_seconds"] = ledger["budget_seconds"]-ledger["worker_seconds"]
        write_json(OUT/"manifest.json", ledger)
        (OUT/"checkpoint.md").write_text(
            "# Active independent R continuation\n\n"
            "Question: constrained-fit outcomes and a separate controller-only mechanism check.\n"
            f"Stage: {stage}.\nUsed {ledger['worker_seconds']:.6f}/2400 worker seconds; "
            f"{len(ledger['attempts'])}/16 launches. No new allocation.\n"
            "Evidence: manifest.json, calibration-decision.json, attempt directories and immutable source snapshots.\n"
            "Next: complete the recorded guide/controller stage or its scoped repair, then terminal analysis "
            "and master update. Candidate failure is not a continuation veto.\n")

    def execute(spec, worker, arguments):
        if len(ledger["attempts"]) >= 16 or spec["timeout_seconds"]+1 > ledger["remaining_seconds"]:
            ledger.setdefault("unlaunched", []).append(spec)
            checkpoint("full reservation unavailable; terminal review required")
            return False
        path = OUT/spec["name"]
        path.mkdir()
        spec["command"] = r_worker_command(snapshots, worker, path/"results", *arguments)
        spec["started_utc"] = datetime.now(timezone.utc).isoformat()
        ledger["inflight"] = [spec]
        checkpoint(spec["stage"] + " running")
        print("BEGIN", spec["name"], flush=True)
        started = time.monotonic()
        with (path/"worker.log").open("w") as log:
            try:
                spec["exit_code"] = run_bounded_worker(spec["command"], cwd=ROOT, env=env, log=log,
                                                       timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                spec["exit_code"] = 124
                spec["failure_class"] = "bounded_worker_timeout"
        spec["worker_seconds"] = time.monotonic()-started
        spec["completed_utc"] = datetime.now(timezone.utc).isoformat()
        if spec["stage"] == "controller":
            for name in ("observations.csv", "kalman.csv"):
                file = path/"results"/name
                if file.exists():
                    spec[name+"_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            spec["inspection"] = inspect(spec)
        else:
            spec["coverage_rows"] = len(read_csv(path/"results"/"coverage.csv"))
            spec["shape_rows"] = len(read_csv(path/"results"/"guide-shapes.csv"))
        ledger["attempts"].append(spec)
        ledger.pop("inflight")
        checkpoint(spec["stage"] + " completed")
        print("END",spec["name"],spec["exit_code"],round(spec["worker_seconds"],3),flush=True)
        return True

    for source,name in [("/tmp/iapf-controller-window-tests.log","controller-window-tests.log"),
                        ("/tmp/iapf-controller-regressions.log","controller-regressions.log"),
                        ("/tmp/iapf-controller-identities.log","controller-identities.log")]:
        shutil.copyfile(source,OUT/name)
    checkpoint("fitting complete; follow-through prepared")
    number = len(ledger["attempts"])+1
    spec = dict(name=f"attempt{number:02d}-saved-guide-shapes",stage="guide_diagnostic",timeout_seconds=120)
    if execute(spec,GUIDE_WORKER,[OUT]) and spec["exit_code"] != 0:
        checkpoint("guide diagnosis failed; local repair needed")
        return
    for data_seed, first in ((89400080,2501),(89500080,2601)):
        number = len(ledger["attempts"])+1
        cap = 400
        previous = [a for a in ledger["attempts"] if a["stage"] == "controller"]
        if previous and ledger["remaining_seconds"] < 401:
            available_cap = int(ledger["remaining_seconds"])-1
            if available_cap > 1.25*previous[0]["worker_seconds"]:
                cap = available_cap
        spec = dict(name=f"attempt{number:02d}-controller-{data_seed}",stage="controller",arm="controller",
                    dimension=80,repeats=8,first=first,data_seed=data_seed,fit_maxit=200,timeout_seconds=cap)
        if not execute(spec,WORKER,["controller",80,8,first,data_seed,200]):
            return
    checkpoint("fitting and controller workers complete; terminal analysis pending")


if __name__ == "__main__":
    main()
