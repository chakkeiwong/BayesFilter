#!/usr/bin/env python3
"""Bounded continuation worker for the M13 retained surrogate-HMC pilot.

The parent waits for public tuning and the same-endpoint RWM baseline, then
launches the retained consumer through the repository run_job watchdog.  It
does not alter source code, tuning, or posterior criteria while waiting.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TUNING_STAGE = "m13-public-tuning"
PLAN = "docs/plans/surrogate-hmc-cumulative-master-repair-2026-10-02.md"
CPU_CAP = 56000.0
GPU_CAP = 48000.0


def read_json(path: Path):
    return json.loads(path.read_text())


def wait_for_prerequisites(tuning_root: Path, baseline_report: Path, *,
                           poll_seconds: float, timeout_seconds: float):
    started = time.monotonic()
    while True:
        tuning_report = tuning_root / "report.json"
        if tuning_report.exists():
            payload = read_json(tuning_report)
            stage = payload.get("stages", {}).get(TUNING_STAGE, {})
            execution = stage.get("execution")
            result = stage.get("result", {})
            if execution == "COMPLETED":
                if payload.get("source_unchanged") is not True:
                    raise RuntimeError("public tuning source snapshot changed")
                if result.get("qualified") is not True:
                    raise RuntimeError(
                        "public tuning completed without a qualified verified artifact"
                    )
                tuning_ready = True
            elif execution in {"FAILED", "SOURCE_CHANGED", "MISSING_REQUIRED_REPORT",
                                "EXECUTION_FAILED", "JOB_CPU_BUDGET_LIMIT",
                                "WALL_TIME_LIMIT"}:
                raise RuntimeError(f"public tuning terminal status: {execution}")
            else:
                tuning_ready = False
        else:
            tuning_ready = False

        baseline_ready = False
        if baseline_report.exists():
            baseline = read_json(baseline_report)
            if baseline.get("status") == "M13_RWM_BASELINE_COMPLETE":
                if baseline.get("health_passed") is not True:
                    raise RuntimeError("RWM baseline completed without health")
                baseline_ready = True
            elif baseline.get("status") in {
                "FAILED", "M13_RWM_HEALTH_FAILURE", "JOB_CPU_BUDGET_LIMIT",
                "CPU_BUDGET_LIMIT", "WALL_TIME_LIMIT", "SOURCE_CHANGED",
            }:
                raise RuntimeError("RWM baseline terminal status: " + str(baseline.get("status")))
        if tuning_ready and baseline_ready:
            return
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("prerequisites did not complete within bounded wait")
        print(
            f"waiting for prerequisites: public={tuning_ready} baseline={baseline_ready}",
            flush=True,
        )
        time.sleep(poll_seconds)


def worker(args):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    import tensorflow as tf
    memory = configure_tensorflow_gpu_memory_growth(tf)
    from bayesfilter.testing.surrogate_hmc_m13_retained import retained_stage
    result = retained_stage(
        args.tuning_root, args.output, args.reference, args.coverage, args.baseline_report
    )
    result["memory_policy"] = memory
    result["device"] = "gpu"
    result["gpu_trust_basis"] = "owner_designated_managed_session_visible_gpu_trusted"
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0  # Successful execution; the cumulative sampling verdict is in report.json.


def parent(args):
    tuning_root = args.tuning_root.resolve()
    baseline_report = args.baseline_report.resolve()
    reference = args.reference.resolve()
    coverage = args.coverage.resolve()
    wait_for_prerequisites(
        tuning_root, baseline_report, poll_seconds=args.poll_seconds,
        timeout_seconds=args.wait_timeout_seconds,
    )
    from bayesfilter.inference.surrogate_hmc_program import (
        file_hash, run_job, validated_budget, write_json,
    )
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"fresh retained output required: {out}")
    sources = [
        Path(__file__),
        ROOT / "bayesfilter/testing/surrogate_hmc_m13_retained.py",
        ROOT / "bayesfilter/testing/surrogate_hmc_m13_cumulative.py",
        ROOT / "bayesfilter/testing/surrogate_hmc_m13_public_tuning.py",
        ROOT / "bayesfilter/inference/hmc_tensorflow_verified.py",
        ROOT / "bayesfilter/inference/hmc_tensorflow_tuning.py",
        ROOT / PLAN,
    ]
    source_hashes = {str(path.relative_to(ROOT)): file_hash(path) for path in sources}
    ledger = validated_budget(GPU_CAP, CPU_CAP)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--tuning-root", str(tuning_root), "--baseline-report", str(baseline_report),
        "--reference", str(reference), "--coverage", str(coverage),
        "--output", str(out / "result"),
    ]
    manifest = {
        "schema": "bayesfilter.surrogate_hmc_m13_retained_continuation.v1",
        "plan": PLAN, "command": sys.argv, "worker_command": command,
        "source_hashes": source_hashes, "device": "gpu",
        "cpu_allowance_seconds": CPU_CAP, "gpu_allowance_seconds": GPU_CAP,
        "timeout_seconds": GPU_CAP, "max_timeout_seconds": GPU_CAP,
        "tuning_root": str(tuning_root), "baseline_report": str(baseline_report),
        "reference": str(reference), "coverage": str(coverage),
        "budget_provenance": "approved M13 continuation retained cap",
        "warmup_excluded": True, "confirmation_data_seed_3701_used": False,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run-manifest.json", manifest)
    job = run_job(
        command, out / "worker", ledger, device="gpu", timeout=GPU_CAP,
        cpu_allowance=CPU_CAP - 5.0, max_timeout_seconds=GPU_CAP,
    )
    result_path = out / "result" / "report.json"
    result = read_json(result_path) if result_path.exists() else {}
    changed = [path for path, digest in source_hashes.items()
               if file_hash(ROOT / path) != digest]
    report = {
        "schema": manifest["schema"], "status": job["status"],
        "source_unchanged": not changed, "changed_sources": changed,
        "job": job, "result": result,
        "sampling_qualified": not changed and job["status"] == "COMPLETED" and result.get("sampling_qualified") is True,
        "posterior_reference_qualified": False, "independent_validation_required": False,
        "tuning_report_sha256": file_hash(tuning_root / "report.json"),
        "baseline_report_sha256": file_hash(baseline_report),
        "reference_sha256": file_hash(reference),
        "coverage_sha256": file_hash(coverage),
        "scope_limit": "exploratory M13 pilot; no all-model or efficiency claim",
    }
    write_json(out / "report.json", report)
    manifest.update(charged=ledger["charged"], wall_seconds=job["wall_seconds"])
    write_json(out / "run-manifest.json", manifest)
    write_json(out / "accounting.json", {
        "charged_this_run": ledger["charged"], "jobs": ledger["jobs"],
    })
    print(json.dumps({"status": report["status"], "result": str(result_path),
                      "charged": ledger["charged"]}), flush=True)
    return 0 if report["status"] == "COMPLETED" and not changed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-root", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.0)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
