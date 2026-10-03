"""Bounded CPU independent-reference investigation, with preserved failures."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker
import run_iapf_r_plausible_choices as reporting

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/iapf-r-replication-gap-audit-2026-09-21.md"
OUT = ROOT / "docs/plans/artifacts/iapf-r-replication-gap-audit-20260921-01"
WORKER = "docs/benchmarks/compare_iapf_r_plausible_choices.R"
DIAGNOSTIC = "docs/benchmarks/diagnose_iapf_r_replication_gaps.R"
SAVED = "docs/plans/artifacts/iapf-r-plausible-reconstruction-20260921-01/attempt08-validation-delayed-d80/results"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [PLAN, WORKER, DIAGNOSTIC,
        "docs/benchmarks/reference_iapf_paper.R",
        "docs/benchmarks/reference_iapf_plausible_choices.R",
        "docs/benchmarks/diagnose_iapf_r_validation_tails.R",
        "docs/benchmarks/run_iapf_r_replication.py",
        "docs/benchmarks/run_iapf_r_plausible_choices.py",
        "docs/benchmarks/run_iapf_r_replication_gap_audit.py",
        "tests/reference_iapf_replication_gaps.R",
        "tests/reference_iapf_plausible_choices.R",
        SAVED + "/observations.csv", SAVED + "/settings.R",
        *(SAVED + f"/iapf-{i}.rds" for i in range(1401,1405))]
    hashes = snapshot_sources(ROOT, sources, OUT / "source-snapshot")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    ledger = {"schema": "iapf_replication_gap_audit_v1", "plan": PLAN,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip(),
        "source_hashes": hashes, "budget_seconds": 2400, "mechanics_budget_seconds": 300,
        "cpu_only": True, "gpu_intentionally_hidden": True, "maximum_workers": 2,
        "environment": subprocess.check_output(["Rscript", "--version"],text=True,stderr=subprocess.STDOUT).strip(),
        "thread_environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")},
        "driver_command": ["/home/chakwong/anaconda3/envs/tftwogpu/bin/python",str(Path(__file__).resolve())],
        "started_utc": datetime.now(timezone.utc).isoformat(), "attempts": [],
        "result_file": str((OUT / "result.md").relative_to(ROOT)),
        "original_author_settings_or_data": False,
        "paper_sha256": hashlib.sha256((ROOT / ".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf").read_bytes()).hexdigest()}
    for path in ("/tmp/iapf-gap-identities.log", "/tmp/iapf-gap-regression.log"):
        if Path(path).exists():
            shutil.copy2(path,OUT / Path(path).name)
    jobs = [dict(name="attempt01-fit-sensitivity",mode="fit",timeout_seconds=120),
        dict(name="attempt02-validation-d20-a",dimension=20,data_seed=87000020,first=1601,repeats=32,timeout_seconds=400),
        dict(name="attempt03-validation-d20-b",dimension=20,data_seed=88000020,first=1701,repeats=32,timeout_seconds=400),
        dict(name="attempt04-validation-d40",dimension=40,data_seed=87000040,first=1801,repeats=32,timeout_seconds=500),
        dict(name="attempt05-fixed-guides",mode="fixed",timeout_seconds=500)]
    pilot = dict(name="attempt06-constrained-pilot",mode="pilot",timeout_seconds=180)
    assert sum(j["timeout_seconds"] for j in jobs)+pilot["timeout_seconds"] <= ledger["budget_seconds"]
    reporting.OUT = OUT
    summaries = []

    def checkpoint(status):
        ledger["status"] = status
        ledger["worker_seconds"] = sum(a.get("worker_seconds",0) for a in ledger["attempts"])
        ledger["remaining_worker_seconds"] = ledger["budget_seconds"]-ledger["worker_seconds"]
        reporting.write_json(OUT / "manifest.json",ledger)
        reporting.write_json(OUT / "summary.json",summaries)
        (OUT / "checkpoint.md").write_text(
            f"# iAPF replication-gap audit\n\nPlan: {PLAN}\n\n"
            f"Status: {status}. Finished workers: {len(ledger['attempts'])}. "
            f"Spent {ledger['worker_seconds']:.3f}/2400 seconds; outstanding "
            "jobs retain their reserved timeouts. See manifest and per-attempt logs.\n")

    def execute(spec):
        spec = dict(spec)
        directory = OUT / spec["name"]
        directory.mkdir()
        if "mode" in spec:
            runner, args = DIAGNOSTIC, [spec["mode"]]
        else:
            spec.update(stage="validation",arm="delayed",baselines=True)
            runner, args = WORKER, ["delayed", spec["dimension"],spec["repeats"],
                spec["first"],spec["data_seed"],"yes"]
        command = r_worker_command(OUT / "source-snapshot",runner,directory / "results",*args)
        spec.update(command=command,started_utc=datetime.now(timezone.utc).isoformat())
        reporting.write_json(directory / "launch.json",spec)
        print("BEGIN",spec["name"],flush=True)
        start = time.monotonic()
        with (directory / "run.log").open("w") as log:
            try:
                spec["exit_code"] = run_bounded_worker(command,cwd=ROOT,env=env,log=log,timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                spec["exit_code"] = 124
                spec["failure_class"] = "bounded_worker_timeout"
        spec["worker_seconds"] = time.monotonic()-start
        spec["completed_utc"] = datetime.now(timezone.utc).isoformat()
        print("END",spec["name"],spec["exit_code"],round(spec["worker_seconds"],3),flush=True)
        return spec

    checkpoint("launching reviewed experiments")
    report_seconds = 0.0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(execute,j):j for j in jobs}
        # Consume the original five. The pilot's scientific prerequisite is the
        # fit sensitivity result, not any validation performance statistic.
        for future in as_completed(futures):
            attempt = future.result()
            ledger["attempts"].append(attempt)
            if "mode" not in attempt:
                start = time.monotonic()
                s = reporting.inspect_attempt(attempt)
                reporting.validation_diagnostics(attempt,s)
                summaries.append(s)
                report_seconds += time.monotonic()-start
            checkpoint("experiments running; candidate failures do not stop independent cells")
        fit_rows = reporting.read_csv(OUT / "attempt01-fit-sensitivity/results/fit-status.csv")
        fit_attempt = next(a for a in ledger["attempts"] if a.get("mode")=="fit")
        if fit_attempt["exit_code"]==0 and any(r["arm"]!="qr" and r["status"]=="converged" for r in fit_rows):
            attempt = pool.submit(execute,pilot).result()
            ledger["attempts"].append(attempt)
        else:
            ledger["pilot_skipped"] = "No converged constrained fit; preserve fit failures before a full trial."
    ledger["report_seconds"] = report_seconds
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint("workers finished; source audit and terminal review pending")
    print("COMPLETE",ledger["worker_seconds"],flush=True)


if __name__ == "__main__":
    main()
