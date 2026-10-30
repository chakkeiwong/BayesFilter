"""Allocate original SSM main lanes from complete cumulative pilot costs.

This framework-free coordinator executes the unchanged frozen numerical
package. Resource allocations change explicitly; numerical designs do not.
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import math
import os
import signal
import subprocess
import time

from bayesfilter.testing.inference_validation.campaign_recovery import (
    checked_frozen_source, _runtime_checked, RESOURCE_STOPS,
)
from bayesfilter.testing.inference_validation.designs import ValidationDesign, digest
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation import ssm_campaign
from scripts.run_hmc_funded_ssm_queue import charged_ssm_seconds, check_wall_time
from scripts.run_hmc_c1_recovery_then_ssm import settle
from scripts.run_hmc_ssm_pilot_repair import complete_price

PLAN = "docs/plans/bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md"
REPAIR_ROOT = REPO / "docs/plans/artifacts/hmc-ssm-pilot-repair-2026-09-29"
MAIN_CAP_SECONDS = 22 * 3600.
MARGIN = 1.5  # Inherited, uncalibrated planning margin; not a runtime quantile.
CLOSEOUT_SECONDS = 30.  # Existing engineering shutdown/reporting allowance.
RESOURCE_OPTIONS = {"fit_process_timeout_seconds", "timeout_policy", "plan_file"}
DATA_OPTIONS = {"data", "dataset_id", "data_seed", "data_version"}


def numerical_workload(payload):
    """Compare declared work across the originally frozen data/seed slots."""
    value = dict(payload)
    for name in ("design_id", "seed", "phase", "budget_seconds", "purpose"):
        value.pop(name, None)
    value["options"] = {k: v for k, v in value["options"].items()
                        if k not in RESOURCE_OPTIONS | DATA_OPTIONS}
    return value


def checked_inputs(config):
    prepared = Path(config["ssm_root"])
    source = checked_frozen_source(prepared / "source")
    if source["identity"] != config["source_identity"]:
        raise ValueError("configured frozen source mismatch")
    for path, sha in read_json(prepared / "prepared-inputs.json").items():
        if file_hash(prepared / path) != sha:
            raise ValueError("frozen prepared inputs changed")
    ssm_campaign.check_preflight(prepared)
    template = read_json(prepared / "main-unpriced.json")
    pilots = read_json(prepared / "pricing.json")
    index = read_json(prepared / "pricing/run_index.json")
    if (index["source"]["identity"] != source["identity"]
            or index["suite_identity"] != digest(pilots)
            or index["execution_options"] != {
                "reuse_leapfrog_graphs": True, "share_unused_budget": False}):
        raise ValueError("original pilot index changed source or execution policy")
    prices = {}
    for original in pilots["designs"]:
        case = original["options"]["campaign_case"]
        cell = prepared / "pricing" / original["design_id"]
        if read_json(cell / "isolated_design.json") != original:
            raise ValueError("pilot numerical design changed")
        identity = read_json(cell / "isolation_identity.json")
        if identity != {"design_identity": ValidationDesign.from_payload(original).identity,
                        "source_identity": source["identity"], "reuse_leapfrog_graphs": True}:
            raise ValueError("pilot identity mismatch")
        price = complete_price(cell, index["jobs"][original["design_id"]])
        if price["status"] == "complete_declared_workload":
            prefix = str(price["receipts"][-1]).removesuffix("-exit.json")
            _runtime_checked(cell / "replication-0000", prefix,
                             ValidationDesign.from_payload(original), identity)
            for receipt_path in price["receipts"]:
                earlier = read_json(receipt_path)
                if earlier.get("campaign_continuation"):
                    evidence = Path(earlier["allocation_file"]).parent / "preserved-evidence.json"
                    if (not earlier.get("preserved_evidence_verified")
                            or any(file_hash(p) != sha for p, sha in read_json(evidence).items())):
                        raise ValueError("continued pilot evidence changed")
            if not math.isfinite(price["price_seconds"]) or price["price_seconds"] <= 0:
                raise ValueError("invalid complete cumulative price")
        price["main_workload_matches"] = all(
            numerical_workload(row) == numerical_workload(original)
            for row in template["designs"] if row["options"]["campaign_case"] == case)
        if not price["main_workload_matches"] and price["status"] == "complete_declared_workload":
            price.update(status="unpriced_main_workload_mismatch", price_seconds=None)
        prices[case] = price
    return template, read_json(prepared / "datasets.json"), prices, source


def allocate_lanes(template, datasets, prices, available_seconds):
    """Fund whole original lanes in case order; keep every unavailable slot."""
    if not math.isfinite(available_seconds) or available_seconds < 0:
        raise ValueError("invalid main allocation")
    funded, dispositions, lanes = {}, [], {}
    remaining = available_seconds
    for case in sorted({d["options"]["campaign_case"] for d in template["designs"]}):
        rows = [d for d in template["designs"] if d["options"]["campaign_case"] == case]
        if len(rows) != 4:
            raise ValueError("original main lane must retain four slots")
        price = prices.get(case, {})
        reason = "no complete cumulative pilot price"
        if price.get("status") == "complete_declared_workload":
            seconds = price["price_seconds"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid complete cumulative price")
            cost = math.ceil(MARGIN * seconds)
            slot = cost + CLOSEOUT_SECONDS
            needed = slot * len(rows)
            reference_ok = case == "K6" or all(
                d["reference_checked"] for d in datasets[rows[0]["scenario"]["target"]])
            if not reference_ok:
                reason = "independent reference unavailable"
            elif needed > remaining:
                reason = "complete lane exceeds remaining global allocation"
            else:
                funded[case] = []
                for row in rows:
                    design = ValidationDesign.from_payload(row)
                    nominal = max(1., cost / MARGIN * 1.25 - 10.)
                    options = {**design.options, "fit_process_timeout_seconds": nominal,
                        "timeout_policy": {**design.options["timeout_policy"],
                            "max_extension_seconds": max(0., cost - nominal - 10.)}}
                    priced = replace(design, budget_seconds=float(cost), options=options)
                    funded[case].append({"design": priced.payload(), "slot_seconds": slot})
                remaining -= needed
                reason = "funded from complete cumulative price"
            lanes[case] = {"pilot_seconds": seconds, "fit_allowance_seconds": cost,
                           "slot_seconds_including_closeout": slot, "lane_seconds": needed}
        dispositions.extend({"design_id": d["design_id"], "case": case, "disposition": reason}
                            for d in rows)
    # First original data/seed slot for each case, then the second, etc.
    jobs = [funded[case][i] for i in range(4) for case in sorted(funded)]
    return {"jobs": jobs, "dispositions": dispositions, "lanes": lanes,
            "original_main_denominator": len(template["designs"]),
            "allocated_seconds": available_seconds - remaining,
            "remaining_unallocated_seconds": remaining, "planning_margin": MARGIN,
            "runtime_tail_guarantee": False, "posterior_outcomes_used_for_allocation": False}


def available_main_seconds(config, output, *, now=None):
    ledger = read_json(config["ledger"])
    if ledger.get("active_reservation"):
        raise ValueError("another campaign reservation is active")
    check_wall_time(config)
    accounting = {**config, "ssm_accounting_roots": [
        *config["ssm_accounting_roots"], str(REPAIR_ROOT), str(output.parent)]}
    prior_main = sum(r["elapsed_seconds"] for r in ledger["records"]
                     if Path(r["receipt"]).is_relative_to(output.parent))
    now = time.time() if now is None else now
    # The stage has its own closeout and systemd shutdown reserve. Every cell
    # additionally budgets its small launcher/receipt overhead explicitly.
    return max(0., min(MAIN_CAP_SECONDS - prior_main,
        config["ssm_gpu_cap_seconds"] - charged_ssm_seconds(accounting, ledger),
        ssm_campaign.free_gpu_seconds(ledger, c1_active=False),
        config["campaign_started_epoch"] + 46 * 3600 - now) - 2 * CLOSEOUT_SECONDS)


def may_start(config, deadline, slot_seconds, *, now_epoch=None, now_monotonic=None):
    now_epoch = time.time() if now_epoch is None else now_epoch
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    return (now_epoch < config["campaign_started_epoch"] + 42 * 3600
            and slot_seconds <= deadline - now_monotonic)


def inspect_cell(root, design, source):
    """Require native fit provenance and terminal evidence; never infer success."""
    path = root / design.design_id / "replication-0000"
    receipts = sorted(path.glob("process-attempt-*-exit.json"))
    if not receipts:
        raise ValueError("main cell lacks a terminal numerical receipt")
    receipt = read_json(receipts[-1])
    prefix = str(receipts[-1]).removesuffix("-exit.json")
    _runtime_checked(path, prefix, design, {"source_identity": source["identity"]})
    if receipt["status"] not in RESOURCE_STOPS | {"complete"}:
        raise ValueError("main numerical failure requires diagnosis")
    row = {"status": receipt["status"], "receipt": str(receipts[-1]),
           "receipt_sha256": file_hash(receipts[-1])}
    if receipt["status"] == "complete":
        assessment = path / "independent_assessment.json"
        if file_hash(assessment) != receipt.get("assessment_sha256"):
            raise ValueError("main completed assessment changed")
        row["assessment"] = str(assessment)
        row["assessment_sha256"] = file_hash(assessment)
    return row


def worker(args):
    config, allocation = read_json(args.config), read_json(args.output / "allocation.json")
    template, datasets, prices, source = checked_inputs(config)
    if source != allocation["source"] or prices != allocation["prices"]:
        raise ValueError("pricing evidence changed between allocation and launch")
    rebuilt = allocate_lanes(template, datasets, prices, allocation["available_seconds"])
    if digest(rebuilt) != digest(allocation["allocation"]):
        raise ValueError("main allocation changed")
    if (os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu):
        raise ValueError("GPU identity and memory growth must be set before workers")
    if (args.seconds is None or not math.isfinite(args.seconds) or args.seconds <= CLOSEOUT_SECONDS
            or args.work_deadline is None or not math.isfinite(args.work_deadline)):
        raise ValueError("main worker requires finite launcher time limits")
    if allocation["allocation"]["allocated_seconds"] > available_main_seconds(config, args.output):
        raise ValueError("remaining campaign time no longer funds this allocation")
    started = time.monotonic()
    deadline = min(args.work_deadline, started + args.seconds - CLOSEOUT_SECONDS)
    receipt_path = args.output / "execution.json"
    ledger_path = Path(config["ledger"])
    ledger = read_json(ledger_path)
    ledger["active_reservation"] = {"service": args.service, "receipt": str(receipt_path),
        "maximum_gpu_seconds_including_shutdown": args.seconds + CLOSEOUT_SECONDS}
    write_json(ledger_path, ledger)
    rows, code = [], 1
    write_json(args.output / "manifest.json", {"command": [sys.executable, *sys.argv],
        "source": source, "coordinator_sha256": file_hash(__file__),
        "environment": sys.executable, "gpu_uuid": args.gpu,
        "memory_growth_required": True, "work_deadline_monotonic": deadline,
        "plan_file": PLAN, "result_file": str(args.output / "result.json"),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        for entry in allocation["allocation"]["jobs"]:
            design = ValidationDesign.from_payload(entry["design"])
            if not may_start(config, deadline, entry["slot_seconds"]):
                rows.append({"design_id": design.design_id, "status": "unstarted_original_deadline"})
                continue
            cell = args.output / "cells" / design.design_id
            suite_path = args.output / "suites" / (design.design_id + ".json")
            write_json(suite_path, ssm_campaign.suite("original-main-" + design.design_id, [design],
                original_main_denominator=32, allocation_plan=PLAN))
            command = [sys.executable, "-m", "bayesfilter.testing.inference_validation", "run",
                str(suite_path), "--output", str(cell), "--max-workers", "1", "--reuse-leapfrog-graphs"]
            write_json(args.output / "status.json", {"status": "running", "current": design.design_id,
                "outcomes": rows, "original_main_denominator": 32})
            cell_started = time.monotonic()
            with (suite_path.with_suffix(".log")).open("w") as log:
                result = subprocess.run(command, cwd=Path(config["ssm_root"]) / "source",
                    stdout=log, stderr=subprocess.STDOUT, check=False, timeout=entry["slot_seconds"])
            row = inspect_cell(cell, design, source)
            if result.returncode != 0 and row["status"] == "complete":
                raise ValueError("main cell coordinator failed after numerical completion")
            rows.append({"design_id": design.design_id, "case": design.options["campaign_case"],
                         "invocation_seconds": time.monotonic() - cell_started,
                         "coordinator_exit_code": result.returncode, **row})
            print(design.design_id, row["status"], flush=True)
        code = 0
    except BaseException as exc:
        write_json(args.output / "failure.json", {"exception": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        elapsed = time.monotonic() - started
        result = {"exit_code": code, "outcomes": rows, "elapsed_seconds": elapsed,
                  "original_main_denominator": 32, "posterior_calibration_claim": False}
        write_json(args.output / "result.json", result)
        write_json(args.output / "status.json", {"status": "complete" if code == 0 else "requires_diagnosis", **result})
        receipt = {"receipt": str(receipt_path), "resource": "gpu", "elapsed_seconds": elapsed,
                   "exit_code": code, "ended_utc": datetime.now(timezone.utc).isoformat(),
                   "accounting": "main stage charged once; nested cells not charged again"}
        write_json(receipt_path, receipt)
        settle(ledger_path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--after-service")
    parser.add_argument("--after-result", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--work-deadline", type=float)
    args = parser.parse_args()
    args.config, args.output = args.config.resolve(), args.output.resolve()
    signal.signal(signal.SIGTERM, lambda number, frame: sys.exit(128 + number))
    if args.worker:
        worker(args)
        return
    config = read_json(args.config)
    if args.after_service:
        while True:
            check_wall_time(config)
            state = subprocess.run(["systemctl", "--user", "show", args.after_service,
                "--property=ActiveState", "--value"], check=True, text=True, capture_output=True, timeout=10)
            if state.stdout.strip() in {"inactive", "failed"}:
                break
            time.sleep(30)
        if args.after_result is None or read_json(args.after_result)["exit_code"] != 0:
            raise ValueError("preceding recovery failed; diagnose before main dispatch")
    template, datasets, prices, source = checked_inputs(config)
    available = available_main_seconds(config, args.output)
    allocation = allocate_lanes(template, datasets, prices, available)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "allocation.json", {"source": source, "prices": prices,
        "available_seconds": available, "allocation": allocation, "plan_file": PLAN})
    if not allocation["jobs"]:
        write_json(args.output / "status.json", {"status": "unfunded", "original_main_denominator": 32})
        return
    seconds = allocation["allocated_seconds"] + CLOSEOUT_SECONDS
    deadline = time.monotonic() + seconds - CLOSEOUT_SECONDS
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--service-type=exec",
        "--unit", args.service, "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={math.floor(seconds)}", "--property=TimeoutStopSec=30",
        f"--working-directory={REPO}", f"--setenv=CUDA_VISIBLE_DEVICES={args.gpu}",
        "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true", "--setenv=TF_NUM_INTRAOP_THREADS=2",
        "--setenv=TF_NUM_INTEROP_THREADS=2", "--setenv=OMP_NUM_THREADS=2",
        "--setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0", sys.executable, str(Path(__file__).resolve()),
        str(args.config), str(args.output), "--gpu", args.gpu, "--service", args.service,
        "--worker", "--seconds", str(seconds), "--work-deadline", str(deadline)]
    write_json(args.output / "launch.json", {"command": command, "plan_file": PLAN})
    confirmed = False
    try:
        result = subprocess.run(command, check=False, timeout=seconds + CLOSEOUT_SECONDS)
        confirmed = True
    finally:
        if not confirmed:
            confirmed = subprocess.run(["systemctl", "--user", "stop", args.service],
                check=False, timeout=CLOSEOUT_SECONDS).returncode == 0
        active = read_json(config["ledger"]).get("active_reservation", {})
        if confirmed and active.get("service") == args.service:
            receipt = {"receipt": active["receipt"], "resource": "gpu",
                "elapsed_seconds": seconds + CLOSEOUT_SECONDS, "exit_code": -1,
                "accounting": "conservative cap after confirmed main group shutdown"}
            write_json(active["receipt"], receipt)
            settle(Path(config["ledger"]), receipt)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
