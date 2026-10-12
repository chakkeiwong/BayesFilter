#!/usr/bin/env python3
"""Continue an authorized priced campaign through the unchanged public consumer.

This orchestration reserves a scoped MAMS feasibility allocation, never selects
members or changes a numerical policy. Polling time is separate from child cost.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.price_modern_hmc_frozen_design import digest, read, write
from scripts.report_modern_hmc_frozen_prices import build_report

METHODS = ("fixed_hmc", "randomized_hmc", "snaper", "mams")


def allocate(report, balance):
    """Use held reservations once; preserve cost uncertainty and full cohorts."""
    if report.get("status") != "audited_cost_diagnostic":
        raise ValueError("terminal audited prices required")
    original = balance["prior_unallocated_worker_seconds"]
    reservations = balance["reservations"]
    unallocated = balance["unallocated_worker_seconds"]
    numbers = [original, unallocated, *reservations.values(), report["actual_price_worker_seconds"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers):
        raise ValueError("invalid campaign balance")
    if not math.isclose(original-sum(reservations.values()), unallocated, abs_tol=1e-6, rel_tol=0):
        raise ValueError("campaign reservations do not reconcile")
    ceiling = max(0., unallocated-max(0., report["actual_price_worker_seconds"]-reservations["gpu_prices"]))
    if [row["method"] for row in report["rows"]] != list(METHODS):
        raise ValueError("complete ordered comparison denominator required")
    mams = report["rows"][-1]
    if mams["root_count"] != 4:
        raise ValueError("the declared complete adapted MAMS cohort is required")
    scenarios = {row["load_multiplier"]: row for row in mams["scenarios"]}
    if set(scenarios) != {1., 2., 4.}:
        raise ValueError("all declared load sensitivities required")
    minimum = scenarios[1.]["minimum_path_seconds"]
    maximum = scenarios[4.]["maximum_count_scenario_seconds"]
    if not (math.isfinite(minimum) and math.isfinite(maximum) and 0 < minimum <= maximum):
        raise ValueError("invalid cost scenarios")
    # Both scenarios include conservative cold-call and reserved-repair costs.
    # They forecast completion cost; they are not a veto on a bounded attempt.
    # The immutable child preserves every unfinished identity at its wall cap.
    funded = ceiling > 0
    limit = min(ceiling, maximum) if funded else 0.
    return dict(status="allocated" if funded else "budget_exhausted",
        scope="single_target_single_seed_bounded_mams_feasibility",
        planned_root_count=mams["root_count"],
        complete_frozen_cohort_preserved=True, full_completion_guaranteed=False,
        posterior_design=mams["design_file"], posterior_design_sha256=mams["design_sha256"],
        available_ceiling_seconds=ceiling, child_max_wall_seconds=limit,
        unfunded_methods=list(METHODS[:-1]), future_replicates_funded=False,
        all_member_minimum_path_seconds=minimum,
        all_member_load_four_maximum_count_seconds=maximum,
        minimum_count_scenario_fits=minimum <= ceiling,
        cost_forecast_is_continuation_veto=False,
        resource_censoring_possible=True, method_ranking=None,
        budget_interpretation=("budget_exhausted" if not funded else
            "cost_scenarios_fit" if maximum <= ceiling else "bounded_feasibility"),
        local_reservation_still_held_seconds=reservations["local_checks"],
        pricing_reservation_release_deferred=True,
        unallocated_after_child_reservation_seconds=ceiling-limit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-audit", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("memory growth must be set before any framework import")
    campaign, output = args.campaign.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(status="waiting_for_prices", started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], environment=sys.executable, driver_sha256=digest(__file__),
        plan="docs/plans/bayesfilter-modern-hmc-priced-comparison-2026-10-12.md",
        campaign=str(campaign), child_worker_seconds=0., method_ranking=None,
        gpu_trust_basis="trusted_escalated_gpu_execution", memory_growth_environment="true",
        cpu_threads={key: os.environ.get(key) for key in (
            "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS", "OMP_NUM_THREADS")})
    write(output / "manifest.json", manifest)
    try:
        while True:
            prices = read(campaign / "prices-01/result.json")
            if "supervisor_elapsed_seconds" in prices:
                break
            elapsed = (datetime.now(timezone.utc)-datetime.fromisoformat(prices["started_utc"])).total_seconds()
            if elapsed > prices["max_wall_seconds"]+3600:
                raise RuntimeError("pricing supervisor has no terminal settlement after its allocation")
            manifest.update(last_checked_utc=datetime.now(timezone.utc).isoformat(),
                            price_status=prices["status"], elapsed_seconds=time.monotonic()-started)
            write(output / "manifest.json", manifest)
            time.sleep(30)
        balance = read(campaign / "allocation-01.json")
        prior = read(args.deployment_audit)
        if prior["status"] != "passed":
            raise ValueError("overhead comparator is not the audited completed deployment")
        calls = prior["native_work"]["native_call_audit"]
        overhead = (prior["gpu_supervisor_seconds"]-sum(c["native_seconds"] for c in calls))/len(calls)
        report = build_report(campaign / "prices-01/result.json",
            [campaign / "designs-02" / f"{method}.json" for method in METHODS],
            available_seconds=balance["unallocated_worker_seconds"],
            trial_overhead_seconds=overhead, fixed_overhead_seconds=0.)
        report["overhead_comparator"] = dict(path=str(args.deployment_audit.resolve()),
            sha256=digest(args.deployment_audit), interpretation="amortized_all_non_native_deployment_cost")
        write(output / "price-audit.json", report)
        allocation = allocate(report, balance)
        write(output / "allocation.json", allocation)
        if allocation["status"] != "allocated":
            manifest.update(status=allocation["status"], allocation=allocation)
            return 0
        manifest.update(status="waiting_for_capacity", allocation=allocation)
        write(output / "manifest.json", manifest)
        # This established selector imports orchestration only. Actual numerical
        # work runs in a separate process from the declared committed snapshot.
        from scripts.price_modern_hmc_method_proposals import select_device
        capacity_started = time.monotonic()
        while True:
            try:
                gpu, probe = select_device("1")
                break
            except RuntimeError as exc:
                if "less than the declared 4096 MiB free-memory screen" not in str(exc):
                    raise
                manifest.update(capacity_error=str(exc), capacity_wait_seconds=time.monotonic()-capacity_started)
                write(output / "manifest.json", manifest)
                if time.monotonic()-capacity_started >= 3600:
                    manifest["status"] = "capacity_wait_exhausted"
                    return 0
                time.sleep(30)
        write(output / "trusted-gpu-probe.json", probe)
        source = Path(prices["source_root"])
        design = Path(allocation["posterior_design"])
        if digest(design) != allocation["posterior_design_sha256"]:
            raise ValueError("frozen posterior design changed before launch")
        command = [sys.executable, str(source / "scripts/run_modern_hmc_frozen_full_model.py"),
            "--design", str(design), "--output", str(output / "mams-run"),
            "--max-wall-seconds", str(allocation["child_max_wall_seconds"])]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TF_FORCE_GPU_ALLOW_GROWTH="true")
        manifest.update(status="running", child_command=command, cuda_visible_devices=gpu,
            source=prices["numerical_source"], design_sha256=allocation["posterior_design_sha256"],
            preparation_wait_seconds=time.monotonic()-started,
            capacity_wait_seconds=time.monotonic()-capacity_started)
        write(output / "manifest.json", manifest)
        tick = time.monotonic()
        with (output / "mams.log").open("x") as log:
            process = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest.update(child_exit_code=process.returncode, child_worker_seconds=time.monotonic()-tick)
        result = read(output / "mams-run/result.json")
        manifest.update(status=result["status"], child_result=str(output / "mams-run/result.json"))
        if process.returncode != 0 and manifest["status"] == "completed":
            raise RuntimeError("completed child report disagrees with process failure")
        return process.returncode
    except BaseException as exc:
        manifest.update(status="repair_required", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
        write(output / "manifest.json", manifest)


if __name__ == "__main__":
    sys.exit(main())
