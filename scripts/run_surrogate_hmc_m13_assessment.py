#!/usr/bin/env python3
"""Bounded CPU diagnostic stages of the approved M13 surrogate-HMC master."""
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
PLAN = "docs/plans/surrogate-hmc-m13-retained-assessment-2026-09-26.md"
SOURCES = (
    "scripts/run_surrogate_hmc_m13_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_reference.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_coverage.py",
    "tests/test_surrogate_hmc_m13_posterior_assessment.py", PLAN,
    "bayesfilter/testing/surrogate_hmc_m13_rwm_baseline.py",
    "tests/test_surrogate_hmc_m13_rwm_baseline.py",
    "docs/plans/surrogate-hmc-m13-rwm-baseline-2026-09-26.md",
    "bayesfilter/testing/surrogate_hmc_m13_retained.py",
    "tests/test_surrogate_hmc_m13_retained.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("review", "reference-coverage", "rwm"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--coverage", type=Path)
    args = parser.parse_args()
    if args.stage == "rwm" and (args.reference is None or args.coverage is None):
        parser.error("RWM requires --reference and --coverage")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "2"
    from bayesfilter.inference.surrogate_hmc_program import file_hash, write_json, run_job
    if args.worker:
        if args.stage == "review":
            import pytest
            return pytest.main(["-q", str(ROOT/SOURCES[4]),
                                str(ROOT/"tests/test_surrogate_hmc_m13_rwm_baseline.py"),
                                str(ROOT/"tests/test_surrogate_hmc_m13_retained.py")])
        if args.stage == "rwm":
            from bayesfilter.testing.surrogate_hmc_m13_rwm_baseline import run_baseline
            run_baseline(args.output/"rwm", args.reference, args.coverage)
            return 0
        from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import prepare_reference
        from bayesfilter.testing.surrogate_hmc_m13_posterior_coverage import run_coverage
        prepare_reference(args.output/"reference")
        result = run_coverage(args.output/"reference/reference.json", args.output/"coverage.json")
        write_json(args.output/"report.json", dict(status="M13_REFERENCE_COVERAGE_COMPLETE",
            qualified=result["qualified"], posterior_qualified=False, ranking_supported=False))
        return 0

    out = args.output.resolve()
    artifacts = ROOT/"docs/plans/artifacts"
    receipts = sorted(list(artifacts.glob("surrogate-hmc-m13-assessment-*/accounting.json"))+
        list(artifacts.glob("surrogate-hmc-closure-20260922-m13-materiality-controls-*/accounting.json")))
    previous_cpu = sum(json.loads(p.read_text())["charged_this_run"]["cpu_worker_seconds"]
                       for p in receipts)
    remaining = 6000.-previous_cpu
    if remaining <= 10.:
        raise ValueError("approved baseline CPU allocation exhausted")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = resource.getrusage(resource.RUSAGE_SELF)
    hashes = {p: file_hash(ROOT/p) for p in SOURCES}
    allowance = min({"review": 180., "reference-coverage": 1800., "rwm": 3500.}[args.stage],
                    remaining-5.)
    ledger = dict(limits=dict(cpu_worker_seconds=allowance, gpu_seconds=0.),
                  charged=dict(cpu_worker_seconds=0., gpu_seconds=0.), jobs=[])
    command = [sys.executable, str(Path(__file__).resolve()), "--stage", args.stage,
               "--output", str(out/"worker"), "--worker"]
    if args.stage == "rwm":
        command += ["--reference", str(args.reference.resolve()),
                    "--coverage", str(args.coverage.resolve())]
    manifest = dict(plan=PLAN, command=sys.argv, worker_command=command,
        previous_cpu_seconds=previous_cpu, remaining_baseline_cpu_seconds=remaining,
        prior_receipts={str(p): file_hash(p) for p in receipts},
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes=hashes, python=sys.executable, device="cpu", gpu_intentionally_hidden=True,
        cpu_exception="independent reference and interval diagnostic, not the HMC default",
        cpu_allowance_seconds=allowance, parent_plan="docs/plans/surrogate-hmc-m13-sampling-continuation-2026-09-25.md",
        data_seed=1704, result_file=str(out/"report.json"),
        allocation="approved shared controls/reference/baselines 6000 CPU seconds; controls already deducted")
    write_json(out/"run-manifest.json", manifest)
    report = dict(status="RUNNING", stage=args.stage)
    try:
        job = run_job(command, out/"worker", ledger, device="cpu",
                      timeout=allowance, cpu_allowance=allowance-5.)
        changed = [p for p, digest in hashes.items() if file_hash(ROOT/p) != digest]
        report.update(status=job["status"], source_unchanged=not changed, changed_sources=changed)
        if changed:
            report["status"] = "SOURCE_CHANGED"
    except Exception as exc:
        report.update(status="FAILED", error=type(exc).__name__+": "+str(exc))
    finally:
        after = resource.getrusage(resource.RUSAGE_SELF)
        ledger["charged"]["cpu_worker_seconds"] += (
            after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime)
        write_json(out/"accounting.json", dict(charged_this_run=ledger["charged"], jobs=ledger["jobs"]))
        manifest["wall_seconds"] = time.monotonic()-started
        manifest.update(ledger["charged"])
        manifest["peak_child_host_bytes"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024
        write_json(out/"run-manifest.json", manifest)
        write_json(out/"report.json", report)
    print(json.dumps(dict(**report, output=str(out), charged=ledger["charged"])), flush=True)
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
