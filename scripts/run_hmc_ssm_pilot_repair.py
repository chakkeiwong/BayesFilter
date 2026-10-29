"""Diagnose startup and resume frozen SSM pilots within the existing campaign."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
from datetime import datetime, timezone
import math
import os
import signal
import subprocess
import time

from bayesfilter.testing.inference_validation.campaign_recovery import (
    checked_frozen_source, continue_frozen_fit, RESOURCE_STOPS,
)
from bayesfilter.testing.inference_validation.fit_supervision import supervise_fit
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation.timeout_policy import TimeoutPolicy
from bayesfilter.testing.inference_validation import ssm_campaign
from scripts.run_hmc_funded_ssm_queue import charged_ssm_seconds, check_wall_time
from scripts.run_hmc_c1_recovery_then_ssm import settle

PLAN = "docs/plans/bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md"
POOL_SECONDS = 6 * 3600.
QUANTUM_SECONDS = 1775.
DIAGNOSTIC_SECONDS = 350.
MAX_ADDITIONAL_ATTEMPTS = 2
CASE_ORDER = (0, 7, 1, 2, 3, 4)
# Reuse the enclosing shutdown allowance as an inside-cap reporting reserve.
# Numerical children must stop before systemd can signal the coordinator.
CLOSEOUT_RESERVE_SECONDS = 30.


def recovery_schedule(mode):
    """One explicit final nonlinear allocation; no new round for other pilots."""
    if mode == "recover-k7-final":
        return ((3, 7),)
    return tuple((round_index + 1, case)
                 for round_index in range(MAX_ADDITIONAL_ATTEMPTS) for case in CASE_ORDER)


def recovery_limits(prior, *, final_k7, remaining_seconds):
    original = sum(r["elapsed_seconds"] for r in prior if not r.get("campaign_continuation"))
    if final_k7:
        if sum(bool(r.get("campaign_continuation")) for r in prior) != MAX_ADDITIONAL_ATTEMPTS:
            raise ValueError("final K7 allocation requires exactly two prior continuations")
        quantum = max(0., remaining_seconds)
        return sum(r["elapsed_seconds"] for r in prior) + quantum, quantum, 3
    return original + MAX_ADDITIONAL_ATTEMPTS * QUANTUM_SECONDS, QUANTUM_SECONDS, MAX_ADDITIONAL_ATTEMPTS


def allocation(config, pool_root, mode):
    ledger = read_json(config["ledger"])
    if ledger.get("active_reservation"):
        raise ValueError("another campaign reservation is active")
    check_wall_time(config)
    config = {**config, "ssm_accounting_roots": [*config["ssm_accounting_roots"], str(pool_root)]}
    spent = sum(row["elapsed_seconds"] for row in ledger["records"]
                if Path(row["receipt"]).is_relative_to(pool_root))
    cap = min(POOL_SECONDS - spent, ssm_campaign.free_gpu_seconds(ledger, c1_active=False),
        config["ssm_gpu_cap_seconds"] - charged_ssm_seconds(config, ledger),
        config["campaign_started_epoch"] + 46 * 3600 - time.time()) - 30.
    if mode == "diagnose":
        cap = min(cap, 2 * DIAGNOSTIC_SECONDS)
    elif mode == "diagnose-followup":
        cap = min(cap, 2 * QUANTUM_SECONDS)
    elif mode == "diagnose-k6-recovery":
        cap = min(cap, QUANTUM_SECONDS)
    if cap <= CLOSEOUT_RESERVE_SECONDS:
        raise ValueError("repair pool, SSM allowance or original wall time exhausted")
    return cap


def complete_price(cell, original_job):
    """A full resumed workload costs every attempt, never just its final tail."""
    path = cell / "replication-0000"
    exits = sorted(path.glob("process-attempt-*-exit.json"))
    receipts = [read_json(p) for p in exits]
    result = {"case": cell.name, "status": "incomplete", "price_seconds": None,
              "receipts": [str(p) for p in exits]}
    if not receipts or receipts[-1]["status"] != "complete":
        return result
    assessment = path / "independent_assessment.json"
    if file_hash(assessment) != receipts[-1].get("assessment_sha256"):
        raise ValueError("continued pilot assessment changed")
    design = read_json(cell / "isolated_design.json")
    pipeline = read_json(path / "pipeline.json")
    members = [m for m in pipeline["members"] if m.get("status") == "assessed"]
    requested = design["options"]["posterior_member_count"]
    minimum = design["options"]["posterior_settings"]["retained_min_results"]
    if (pipeline["completion"] != "complete" or len(members) != requested
            or any(m.get("recorded_retained_count", 0) < minimum for m in members)):
        return {**result, "status": "complete_process_incomplete_declared_workload"}
    original_outer = sum(a["elapsed_seconds"] for a in original_job["attempts"])
    original_inner = sum(r["elapsed_seconds"] for r in receipts if not r.get("campaign_continuation"))
    continued = sum(max(r.get("continuation_invocation_seconds", 0.), r["elapsed_seconds"])
                    for r in receipts if r.get("campaign_continuation"))
    cost = max(original_outer, original_inner) + continued
    return {**result, "status": "complete_declared_workload", "price_seconds": cost,
            "posterior_outcome_used_for_pricing": False, "requested_members": requested,
            "original_elapsed_seconds": max(original_outer, original_inner),
            "continuation_elapsed_seconds": continued, "candidate_count": pipeline["candidate_count"],
            "retained_counts": [m["recorded_retained_count"] for m in members]}


def worker(args):
    config = read_json(args.config)
    prepared = Path(config["ssm_root"])
    source = prepared / "source"
    frozen = checked_frozen_source(source)
    if frozen["identity"] != config["source_identity"]:
        raise ValueError("configured numerical source mismatch")
    ssm_campaign.check_preflight(prepared)
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise ValueError("memory growth required before framework import")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-"):
        raise ValueError("a recorded GPU UUID is required")
    seconds = min(args.seconds, allocation(config, args.output.parent, args.mode))
    if (args.work_deadline_monotonic is None
            or not math.isfinite(args.work_deadline_monotonic)):
        raise ValueError("worker requires the launcher's finite absolute work deadline")
    started = time.monotonic()
    deadline = min(args.work_deadline_monotonic, started + seconds - CLOSEOUT_RESERVE_SECONDS)
    if deadline <= started:
        raise ValueError("no numerical time remains after startup and closeout reserve")
    args.output.mkdir(parents=True, exist_ok=False)
    ledger_path = Path(config["ledger"])
    ledger = read_json(ledger_path)
    receipt_path = args.output / "execution.json"
    ledger["active_reservation"] = {"service": args.service, "receipt": str(receipt_path),
        "maximum_gpu_seconds_including_shutdown": seconds + 30.}
    write_json(ledger_path, ledger)
    outcomes = []
    code = 1
    manifest = {"plan_file": PLAN, "command": [sys.executable, *sys.argv], "source": frozen,
        "coordinator_file": str(Path(__file__).resolve()), "coordinator_sha256": file_hash(__file__),
        "environment": sys.executable, "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_growth_required": True, "mode": args.mode, "pool_cap_seconds": POOL_SECONDS,
        "maximum_additional_gpu_seconds": seconds + 30., "original_campaign_started_epoch": config["campaign_started_epoch"],
        "work_deadline_monotonic": deadline, "closeout_reserve_seconds": CLOSEOUT_RESERVE_SECONDS,
        "initial_numerical_allowance_seconds": deadline - started,
        "data_and_seed_authority": str(prepared / "pricing.json"),
        "result_file": str(args.output / "result.json"), "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "status.json", {"status": "running", "mode": args.mode})
    try:
        if args.mode.startswith("diagnose"):
            for case in ((6,) if args.mode == "diagnose-k6-recovery" else (5, 6)):
                restarts = 0
                if args.mode != "diagnose":
                    prior_root = args.output.parent / ("diagnosis-r2" if args.mode == "diagnose-k6-recovery" else "diagnosis-r1")
                    prior_file = prior_root / f"K{case}/result.json"
                    if prior_file.exists():
                        previous = read_json(prior_file)
                        if previous["status"] == "prepared":
                            outcomes.append({"case": f"K{case}", "status": "reused_preparation_diagnostic",
                                             "diagnostic_result": str(prior_file)})
                            continue
                        if previous.get("details", {}).get("stage") != "windowed_mass":
                            raise ValueError("follow-up requires a diagnosed warmup failure")
                        restarts = 3
                    elif read_json(prior_root / f"K{case}-supervision.json")["status"] not in RESOURCE_STOPS:
                        raise ValueError("follow-up requires a resource-limited preparation diagnostic")
                check_wall_time(config)
                available = min(DIAGNOSTIC_SECONDS if args.mode == "diagnose" else QUANTUM_SECONDS,
                                deadline - time.monotonic())
                if available <= 0:
                    break
                directory = args.output / f"K{case}"
                command = [sys.executable, str(REPO / "scripts/diagnose_hmc_ssm_bootstrap.py"),
                    str(source), str(prepared / f"pricing/price-K{case}/isolated_design.json"),
                    str(directory), "--rounds", "20", "--preparation-restarts", str(restarts)]
                row = supervise_fit(command, args.output / f"K{case}.log", available, "gpu",
                    cell_deadline=deadline, timeout_policy=TimeoutPolicy(gpu_admission_mode="shared"), cwd=source)
                outcomes.append({"case": f"K{case}", **row, "diagnostic_result": str(directory / "result.json")})
                write_json(args.output / "status.json", {"status": "running", "outcomes": outcomes})
        else:
            original_index = read_json(prepared / "pricing/run_index.json")
            # Two rounds share the common pool; one difficult fit cannot consume
            # both quanta before independent model mechanisms get an opportunity.
            for round_number, case in recovery_schedule(args.mode):
                check_wall_time(config)
                if time.monotonic() >= deadline:
                    break
                cell = prepared / f"pricing/price-K{case}"
                prior = [read_json(p) for p in sorted((cell / "replication-0000").glob("process-attempt-*-exit.json"))]
                cap, quantum, attempts = recovery_limits(prior,
                    final_k7=args.mode == "recover-k7-final", remaining_seconds=deadline-time.monotonic())
                output = args.output / f"round-{round_number}-K{case}"
                row = continue_frozen_fit(cell=cell, source=source, replication=0, output=output,
                    cumulative_cap_seconds=cap, quantum_seconds=quantum, deadline=deadline,
                    max_additional_attempts=attempts)
                outcomes.append({"case": f"K{case}", "round": round_number, **row})
                write_json(args.output / "status.json", {"status": "running", "outcomes": outcomes,
                    "remaining_seconds": max(0., deadline - time.monotonic())})
                print({"case": f"K{case}", "round": round_number, "status": row["status"]}, flush=True)
                if row["status"] not in RESOURCE_STOPS | {"complete", "reused_final_assessment",
                        "ineligible_or_allocation_exhausted", "no_resumable_numerical_checkpoint"}:
                    raise RuntimeError("continued fit failed; diagnose before another launch")
            prices = [complete_price(prepared / f"pricing/price-K{k}", original_index["jobs"][f"price-K{k}"])
                      for k in range(8)]
            write_json(args.output / "continued-prices.json", {"original_index": str(prepared / "pricing/run_index.json"),
                "original_index_sha256": file_hash(prepared / "pricing/run_index.json"), "prices": prices,
                "original_main_denominator": 32, "ranking_established": False,
                "main_launch_requires_recomputed_affordability": True})
        code = 0
    except BaseException as exc:
        write_json(args.output / "failure.json", {"exception": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        elapsed = time.monotonic() - started
        result = {"exit_code": code, "mode": args.mode, "outcomes": outcomes,
                  "elapsed_seconds": elapsed, "posterior_claim": False}
        write_json(args.output / "result.json", result)
        write_json(args.output / "status.json", {"status": "complete" if code == 0 else "requires_diagnosis", **result})
        receipt = {"receipt": str(receipt_path), "resource": "gpu", "elapsed_seconds": elapsed,
            "exit_code": code, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "accounting": "enclosing repair charged once; original and nested fit receipts not charged again"}
        write_json(receipt_path, receipt)
        settle(ledger_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("diagnose", "diagnose-followup", "diagnose-k6-recovery", "recover", "recover-k7-final"), required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--work-deadline-monotonic", type=float)
    args = parser.parse_args()
    args.config, args.output = args.config.resolve(), args.output.resolve()
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if args.worker:
        worker(args)
        return
    config = read_json(args.config)
    seconds = allocation(config, args.output.parent, args.mode)
    if args.output.exists() or (args.output.parent / (args.output.name + "-launch.json")).exists():
        raise ValueError("launch output already exists; use a fresh versioned directory")
    work_deadline = time.monotonic() + seconds - CLOSEOUT_RESERVE_SECONDS
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--service-type=exec",
        "--unit", args.service, "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={math.floor(seconds)}", "--property=TimeoutStopSec=30",
        f"--working-directory={REPO}", f"--setenv=CUDA_VISIBLE_DEVICES={args.gpu}",
        "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true", "--setenv=TF_NUM_INTRAOP_THREADS=2",
        "--setenv=TF_NUM_INTEROP_THREADS=2", "--setenv=OMP_NUM_THREADS=2",
        "--setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0", sys.executable, str(Path(__file__).resolve()),
        str(args.config), str(args.output), "--mode", args.mode, "--service", args.service,
        "--gpu", args.gpu, "--worker", "--seconds", str(seconds),
        "--work-deadline-monotonic", str(work_deadline)]
    write_json(args.output.parent / (args.output.name + "-launch.json"), {"command": command, "plan_file": PLAN})
    confirmed = False
    try:
        result = subprocess.run(command, check=False, timeout=seconds + 30.)
        confirmed = True
    finally:
        if not confirmed:
            stop = subprocess.run(["systemctl", "--user", "stop", args.service], check=False, timeout=30)
            confirmed = stop.returncode == 0
        active = read_json(config["ledger"]).get("active_reservation", {})
        if confirmed and active.get("service") == args.service:
            receipt = {"receipt": active["receipt"], "resource": "gpu", "elapsed_seconds": seconds + 30.,
                "exit_code": -1, "accounting": "conservative cap after confirmed repair group shutdown"}
            write_json(active["receipt"], receipt)
            settle(Path(config["ledger"]), receipt)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
