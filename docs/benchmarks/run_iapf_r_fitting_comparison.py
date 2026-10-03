"""Bounded independent CPU R fitting comparison; not a runtime implementation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/iapf-r-full-filter-fitting-comparison-2026-09-21.md"
OUT = ROOT / "docs/plans/artifacts/iapf-r-full-filter-fitting-20260921-01"
WORKER = "docs/benchmarks/compare_iapf_r_fitting.R"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def inspect(attempt):
    path = OUT / attempt["name"] / "results"
    rows = read_csv(path / "replicates.csv")
    methods = (["qr", "short_qr", "bpf", "fully_adapted", "sis"] if attempt["arm"] == "controller" else
               ["qr", "bpf", "fully_adapted", "sis"] if attempt["arm"] == "baselines" else [attempt["arm"]])
    expected = {(str(i), m) for i in range(attempt["first"], attempt["first"] + attempt["repeats"])
                for m in methods}
    expected -= {tuple(key) for key in attempt.get("completed_before", [])}
    observed = [(r["replication"], r["method"]) for r in rows]
    if len(set(observed)) != len(observed) or not set(observed).issubset(expected):
        raise RuntimeError("duplicate or unexpected replicate identity")
    for row in rows:
        index = {"bpf": 2, "fully_adapted": 3, "sis": 4}.get(row["method"], 1)
        expected_seed = 53000000 + 100000*attempt["dimension"] + 10*int(row["replication"]) + index
        if int(row["seed"]) != expected_seed:
            raise RuntimeError("wrong filter seed")
    good = [r for r in rows if r["status"] == "complete"]
    fitted = [r for r in good if r["method"] in ("qr", "short_qr") or r["method"].startswith("box")]
    fits = read_csv(path / "fits.csv")
    tails = read_csv(path / "tails.csv")
    prefixes = read_csv(path / "prefixes.csv")
    # Timeout can leave partial per-replica diagnostics. Exclude those from
    # acceptance and require exact records for every fully recorded replica.
    fitted_keys = {(r["replication"], r["method"]) for r in fitted}
    legacy_method = "qr" if attempt["arm"] == "baselines" else attempt["arm"]
    fit_rows = [r for r in fits if (r["replication"], r.get("method", legacy_method)) in fitted_keys]
    tail_rows = [r for r in tails if (r["replication"], r.get("method", legacy_method)) in fitted_keys]
    completed_keys = {(r["replication"], r["method"]) for r in good}
    prefix_rows = [r for r in prefixes if (r["replication"], r["method"]) in completed_keys]
    records_ok = (len(fit_rows) == sum(100*(int(r["iterations"])-1) for r in fitted)
                  and len(tail_rows) == 100*len(fitted) and len(prefix_rows) == 100*len(good))
    fit_ok = all(int(r["convergence"]) == 0 for r in fit_rows)
    tail_ok = all(r["passed"] == "TRUE" for r in tail_rows)
    return dict(expected_rows=len(expected), observed_rows=len(rows), complete_rows=len(good),
                failed_rows=len(rows)-len(good), records_ok=records_ok,
                fit_rows=len(fit_rows), converged_fits=fit_ok, tail_rows=len(tail_rows),
                tail_pass=tail_ok,
                active_bound_fits=sum(r.get("boundary") == "TRUE" for r in fit_rows),
                eligible=(attempt["exit_code"] == 0 and set(observed) == expected
                          and len(good) == len(expected) and records_ok and fit_ok and tail_ok))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [PLAN, WORKER, "docs/benchmarks/reference_iapf_paper.R",
               "docs/benchmarks/reference_iapf_plausible_choices.R",
               "docs/benchmarks/diagnose_iapf_r_validation_tails.R",
               "docs/benchmarks/run_iapf_r_replication.py",
               "docs/benchmarks/run_iapf_r_fitting_comparison.py",
               "tests/reference_iapf_replication_gaps.R"]
    snapshots = OUT / "source-snapshot"
    hashes = snapshot_sources(ROOT, sources, snapshots)
    ledger = dict(status="running", plan=PLAN, result=str(OUT / "result.md"),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  source_hashes=hashes, budget_seconds=2400, mechanics_budget_seconds=300,
                  launch_limit=16, max_workers=2, attempts=[], worker_seconds=0,
                  environment=dict(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                                   device="independent CPU R; GPU intentionally hidden",
                                   python=os.sys.executable,
                                   r_version=subprocess.check_output(["Rscript", "--version"], stderr=subprocess.STDOUT, text=True).strip()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def checkpoint(stage):
        ledger["status"] = stage
        ledger["worker_seconds"] = sum(a["worker_seconds"] for a in ledger["attempts"])
        ledger["remaining_seconds"] = ledger["budget_seconds"]-ledger["worker_seconds"]
        write_json(OUT / "manifest.json", ledger)
        (OUT / "checkpoint.md").write_text(
            "# Active fitting comparison\n\nQuestion: constrained Equation15 versus QR full filters.\n"
            f"Stage: {stage}.\nUsed: {ledger['worker_seconds']:.6f}/2400 summed worker seconds; "
            f"launches {len(ledger['attempts'])}/16.\n"
            "Evidence: manifest.json, source-snapshot/, attempt*/results/.\n"
            "Next: finish the recorded stage; nominate only by complete calibration evidence; "
            "preserve failures and diagnose before any repair. Controller is fixed.\n")

    def execute(spec):
        path = OUT / spec["name"]
        path.mkdir()
        command = r_worker_command(snapshots, WORKER, path / "results", spec["arm"],
                                   spec["dimension"], spec["repeats"], spec["first"],
                                   spec["data_seed"], spec["fit_maxit"])
        spec["command"] = command
        spec["started_utc"] = datetime.now(timezone.utc).isoformat()
        print("BEGIN", spec["name"], flush=True)
        started = time.monotonic()
        with (path / "worker.log").open("w") as log:
            try:
                spec["exit_code"] = run_bounded_worker(command, cwd=ROOT, env=env, log=log,
                                                       timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                spec["exit_code"] = 124
                spec["failure_class"] = "bounded_worker_timeout"
        spec["worker_seconds"] = time.monotonic()-started
        spec["completed_utc"] = datetime.now(timezone.utc).isoformat()
        for name in ("observations.csv", "kalman.csv"):
            file = path / "results" / name
            if file.exists():
                spec[name + "_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
        spec["inspection"] = inspect(spec)
        print("END", spec["name"], spec["exit_code"], round(spec["worker_seconds"], 3),
              spec["inspection"], flush=True)
        return spec

    def run_pair(jobs, stage):
        reserve = sum(s["timeout_seconds"] + 1 for s in jobs)
        if reserve > ledger["remaining_seconds"] or len(ledger["attempts"]) + len(jobs) > 16:
            ledger.setdefault("unlaunched", []).extend(jobs)
            checkpoint("allocation prevents full reservation")
            return False
        ledger["inflight"] = jobs
        checkpoint(stage)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(execute, jobs))
        ledger["attempts"].extend(results)
        ledger.pop("inflight")
        checkpoint(stage + " pair completed")
        return True

    checkpoint("calibration prepared")
    number = 0
    for arm in ("qr", "box0.5", "box1", "box2"):
        jobs = []
        for d, first in ((5, 2101), (20, 2201)):
            number += 1
            jobs.append(dict(name=f"attempt{number:02d}-calibration-d{d}-{arm}", stage="calibration",
                             arm=arm, dimension=d, repeats=4, first=first, data_seed=89100000+d,
                             fit_maxit=200, timeout_seconds=180 if d == 5 else 360))
        if not run_pair(jobs, "calibration"):
            return
    eligible = []
    for arm in ("box1", "box0.5", "box2"):
        cells = [a for a in ledger["attempts"] if a["arm"] == arm]
        if len(cells) == 2 and all(a["inspection"]["eligible"] for a in cells):
            eligible.append(arm)
    decision = dict(eligible_arms=eligible, nominated=eligible[0] if eligible else None,
                    fit_maxit=200, rule="nominal width 1, then .5, then 2; no performance ranking",
                    frozen_utc=datetime.now(timezone.utc).isoformat(), validation_opened=False)
    write_json(OUT / "calibration-decision.json", decision)
    if not eligible:
        checkpoint("no admissible constrained arm; diagnose saved failures before validation")
        return
    for data_seed, first in ((89200020, 2301), (89300020, 2401)):
        jobs = []
        for arm in (decision["nominated"], "baselines"):
            number += 1
            jobs.append(dict(name=f"attempt{number:02d}-validation-{data_seed}-{arm}", stage="validation",
                             arm=arm, dimension=20, repeats=16, first=first, data_seed=data_seed,
                             fit_maxit=200, timeout_seconds=200 if arm == "baselines" else 600))
        if not run_pair(jobs, "validation"):
            return
    checkpoint("planned workers complete; terminal analysis and review pending")


if __name__ == "__main__":
    main()
