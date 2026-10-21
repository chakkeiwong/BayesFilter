#!/usr/bin/env python3
"""Bounded continuation of the interrupted M13 RWM reference comparator."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/surrogate-hmc-m13-rwm-resume-2026-09-26.md"
REPAIR_CAP = 10000.0
CPU_ALLOWANCE = 6000.0
SOURCES = (
    "scripts/run_surrogate_hmc_m13_rwm_resume.py",
    "bayesfilter/testing/surrogate_hmc_m13_rwm_resume.py",
    "bayesfilter/testing/surrogate_hmc_m13_rwm_baseline.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_coverage.py",
    "bayesfilter/testing/surrogate_hmc_m13_endpoint_mechanics.py",
    "bayesfilter/testing/surrogate_hmc_m13_four_chain_cost.py",
    PLAN,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    from bayesfilter.inference.surrogate_hmc_program import file_hash, run_job, write_json
    if args.worker:
        from bayesfilter.testing.surrogate_hmc_m13_rwm_resume import run_resume
        report = run_resume(args.output, args.previous, args.reference, args.coverage)
        print(json.dumps(report, sort_keys=True), flush=True)
        return 0

    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    receipts = [ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01/accounting.json"]
    previous_repair_cpu = sum(float(json.loads(p.read_text())["charged_this_run"]["cpu_worker_seconds"])
                              for p in receipts)
    if previous_repair_cpu + CPU_ALLOWANCE > REPAIR_CAP:
        raise ValueError("resume allowance exceeds the approved repair allocation")
    out.mkdir(parents=True, exist_ok=False)
    hashes = {name: file_hash(ROOT / name) for name in SOURCES}
    ledger = dict(limits=dict(cpu_worker_seconds=CPU_ALLOWANCE, gpu_seconds=0.),
                  charged=dict(cpu_worker_seconds=0., gpu_seconds=0.), jobs=[])
    command = [sys.executable, str(Path(__file__).resolve()), "--previous", str(args.previous.resolve()),
               "--reference", str(args.reference.resolve()), "--coverage", str(args.coverage.resolve()),
               "--output", str(out / "worker"), "--worker"]
    manifest = dict(plan=PLAN, command=sys.argv, worker_command=command,
        previous_repair_cpu_seconds=previous_repair_cpu,
        repair_cap_cpu_seconds=REPAIR_CAP, cpu_allowance_seconds=CPU_ALLOWANCE,
        source_hashes=hashes, commit=subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True).strip(), device="cpu", gpu_intentionally_hidden=True,
        reference=str(args.reference.resolve()), coverage=str(args.coverage.resolve()),
        previous=str(args.previous.resolve()), result_file=str(out / "report.json"))
    write_json(out / "run-manifest.json", manifest)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    report = {"status": "RUNNING", "stage": "rwm-resume"}
    try:
        job = run_job(command, out / "worker", ledger, device="cpu", timeout=CPU_ALLOWANCE,
                      cpu_allowance=CPU_ALLOWANCE - 5.)
        changed = [name for name, digest in hashes.items() if file_hash(ROOT / name) != digest]
        report.update(status=job["status"], source_unchanged=not changed, changed_sources=changed)
        if changed:
            report["status"] = "SOURCE_CHANGED"
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        charged = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        ledger["charged"]["cpu_worker_seconds"] += charged
        write_json(out / "accounting.json", dict(charged_this_run=ledger["charged"], jobs=ledger["jobs"]))
        manifest.update(wall_seconds=time.monotonic() - started,
                        cpu_worker_seconds=ledger["charged"]["cpu_worker_seconds"],
                        gpu_seconds=0., peak_child_host_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
        write_json(out / "run-manifest.json", manifest)
        write_json(out / "report.json", report)
    print(json.dumps(dict(**report, output=str(out), charged=ledger["charged"])), flush=True)
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
