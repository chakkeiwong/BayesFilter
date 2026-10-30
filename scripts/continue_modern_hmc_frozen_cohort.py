#!/usr/bin/env python3
"""Bounded four-method orchestration around an immutable numerical consumer.

Prices inform reservations only. Candidate membership and posterior decisions
remain with the unchanged full-model consumer; no failed member is rerun here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from scripts.price_modern_hmc_frozen_design import (
    audit_attempts, digest, planned_rows, read, resume_rows, write,
)
from scripts.report_modern_hmc_frozen_prices import build_report
from scripts.settle_modern_hmc_frozen_run import audit, service_state

METHODS = ("fixed_hmc", "randomized_hmc", "snaper", "mams")
INFRASTRUCTURE = {"repair_required", "worker_interrupted", "supervisor_interrupted"}


def validate_launcher_exit(code, supervisor):
    """Check the Python launcher's POSIX exit byte against its worker code.

    A negative launcher returncode means the launcher itself was signalled,
    unlike an orderly launcher exit carrying a worker signal.
    """
    worker_code = supervisor.get("exit_code")
    if (type(code) is not int or type(worker_code) is not int
            or not -255 <= worker_code <= 255 or not 0 <= code <= 255
            or code != worker_code % 256):
        raise ValueError("launcher exit and recorded worker exit disagree")
    return dict(launcher_exit_code=code, worker_exit_code=worker_code,
                convention="posix_sys_exit_low_byte")


def seconds(value, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or positive and value == 0):
        raise ValueError("finite " + ("positive" if positive else "nonnegative") + " seconds required")
    return float(value)


def inventory(designs):
    rows = planned_rows(designs)
    if tuple(row["method"] for row in rows) != METHODS:
        raise ValueError("complete ordered four-method inventory required")
    if len({(row["target"], row["seed"]) for row in rows}) != 1:
        raise ValueError("a common target and start seed are required")
    for row in rows:
        design = read(row["design_file"])
        roots = len(row["roots"])
        repairs = design["max_repairs_per_family"]
        if type(repairs) is not int or repairs < 0:
            raise ValueError("invalid repair allowance")
        if (design["search"]["max_candidates"] != roots * (repairs + 1)
                or design["execution"]["acceptance_policy"]["max_candidates"] < roots * (repairs + 1)
                or design["execution"]["use_xla"] is not True):
            raise ValueError("complete root/repair allowance and XLA required")
    return rows


def reconcile_preparation(preparation):
    initial = seconds(preparation["unallocated_after_preparation_and_pricing_hold_seconds"])
    hold = seconds(preparation["pricing_reserved_seconds"])
    total = seconds(preparation["unallocated_before_preparation_seconds"])
    spent = seconds(preparation["preparation_actual_seconds"])
    if (preparation.get("preparation_charged_once") is not True
            or not math.isclose(initial + hold + spent, total, abs_tol=1e-6, rel_tol=0)):
        raise ValueError("preparation balance does not reconcile")
    receipts = preparation["preparation_receipts"]
    if (len({item["path"] for item in receipts}) != len(receipts)
            or not math.isclose(math.fsum(seconds(item["seconds"]) for item in receipts),
                                spent, abs_tol=1e-6, rel_tol=0)):
        raise ValueError("preparation receipt charges do not reconcile")
    for item in receipts:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("preparation receipt changed")
    return initial + hold


def allocate_cohort(report, designs, *, after_preparation, engineering_hold):
    """Reserve all methods together, using the declared load-four cost weights."""
    rows = inventory(designs)
    if report.get("status") != "audited_cost_diagnostic":
        raise ValueError("complete audited prices required before allocation")
    if tuple(row["method"] for row in report["rows"]) != METHODS:
        raise ValueError("complete ordered price inventory required")
    available = (seconds(after_preparation) - seconds(engineering_hold)
                 - seconds(report["actual_price_worker_seconds"], positive=True))
    envelopes, minimums = [], []
    for planned, priced in zip(rows, report["rows"]):
        if any(priced.get(key) != planned[key] for key in ("method", "target", "design_file", "design_sha256")):
            raise ValueError("prices differ from the frozen design inventory")
        if priced["root_count"] != len(planned["roots"]):
            raise ValueError("priced root count differs from the complete design")
        scenarios = {item["load_multiplier"]: item for item in priced["scenarios"]}
        if len(scenarios) != 3 or len(priced["scenarios"]) != 3 or set(scenarios) != {1., 2., 4.}:
            raise ValueError("all declared load scenarios required")
        envelope = seconds(scenarios[4.]["maximum_count_scenario_seconds"], positive=True)
        minimum = seconds(scenarios[1.]["minimum_path_seconds"], positive=True)
        if minimum > envelope:
            raise ValueError("minimum path exceeds the cost envelope")
        envelopes.append(envelope)
        minimums.append(minimum)
    if available <= 0:
        raise ValueError("no numerical balance remains after prices and engineering hold")
    fraction = min(1., available / math.fsum(envelopes))
    caps = [seconds(fraction * value, positive=True) for value in envelopes]
    return dict(status="reserved", method_order=list(METHODS),
        available_numerical_seconds=available, pricing_charged_once_seconds=report["actual_price_worker_seconds"],
        engineering_reserved_seconds=engineering_hold, common_envelope_fraction=fraction,
        unallocated_after_all_method_reservations_seconds=max(0., available-math.fsum(caps)),
        rows=[dict(method=row["method"], root_count=len(row["roots"]),
            reserved_member_count=read(row["design_file"])["search"]["max_candidates"],
            initial_cap_seconds=cap, envelope_seconds=envelope,
            minimum_all_member_path_seconds=minimum, minimum_path_fits_cap=minimum <= cap)
            for row, cap, envelope, minimum in zip(rows, caps, envelopes, minimums)],
        full_completion_guaranteed=False, resource_censoring_possible=True,
        method_ranking=None, future_replications_funded=False)


def next_cap(allocation, index, balance):
    """Release settled unused caps without consuming later initial reservations."""
    balance = seconds(balance, positive=True)
    remaining = allocation["rows"][index:]
    if not remaining:
        raise ValueError("no pending method")
    total = math.fsum(seconds(row["initial_cap_seconds"], positive=True) for row in remaining)
    if balance < total:  # Inclusive supervisor overruns reduce every remaining share.
        return seconds(balance * remaining[0]["initial_cap_seconds"] / total, positive=True)
    released = balance - total
    return seconds(min(remaining[0]["envelope_seconds"],
                       remaining[0]["initial_cap_seconds"] + released), positive=True)


def continue_after(settlement):
    if (settlement.get("status") != "settled" or settlement.get("evidence_audit_passed") is not True
            or settlement.get("cohort_status") not in
            {"completed", "resource_stopped", "supervisor_budget_exhausted", *INFRASTRUCTURE}):
        raise ValueError("valid terminal settlement required")
    return settlement["cohort_status"] not in INFRASTRUCTURE


def measured_overhead(path):
    prior = read(path)
    if prior.get("status") != "passed":
        raise ValueError("audited deployment comparator required")
    calls = prior["native_work"]["native_call_audit"]
    if not calls:
        raise ValueError("deployment comparator has no native calls")
    overhead = (seconds(prior["gpu_supervisor_seconds"], positive=True)
                - math.fsum(seconds(call["native_seconds"], positive=True) for call in calls)) / len(calls)
    return seconds(overhead)


class EngineeringLedger:
    """Charge measured checks and exclusive orchestration, never sleeping/children."""
    def __init__(self, path, *, cap, initial_receipt, prior=None):
        self.path, self.cap = Path(path), seconds(cap, positive=True)
        receipt = read(initial_receipt)
        if receipt.get("status") != "passed":
            raise ValueError("focused engineering checks must pass")
        self.entries = [dict(role="focused_preparation", seconds=seconds(receipt["elapsed_seconds"]),
            receipt=str(Path(initial_receipt).resolve()), sha256=digest(initial_receipt))]
        if prior is not None:
            old = read(prior)
            if (old["cap_seconds"] != self.cap or
                    not math.isclose(math.fsum(seconds(e["seconds"]) for e in old["entries"]),
                                     seconds(old["charged_seconds"]), abs_tol=1e-6, rel_tol=0)):
                raise ValueError("prior engineering charges/cap do not reconcile")
            for entry in old["entries"]:
                if "receipt" in entry and digest(entry["receipt"]) != entry["sha256"]:
                    raise ValueError("prior engineering receipt changed")
            self.entries.insert(0, dict(role="prior_controller_inclusive_engineering",
                seconds=old["charged_seconds"], receipt=str(Path(prior).resolve()), sha256=digest(prior)))
        self.save()

    @property
    def spent(self):
        return math.fsum(item["seconds"] for item in self.entries)

    def save(self, *, terminal=False, enforce=True):
        spent = self.spent
        write(self.path, dict(status="within_cap" if spent <= self.cap else "cap_exhausted",
            cap_seconds=self.cap, charged_seconds=spent, entries=self.entries,
            unused_hold_seconds=max(0., self.cap-spent), children_and_sleep_excluded=True,
            unused_hold_released=terminal, shortfall_seconds=max(0., spent-self.cap)))
        if spent >= self.cap and enforce:
            raise RuntimeError("exclusive engineering allowance exhausted")

    @contextmanager
    def measure(self, role):
        started = time.monotonic()
        try:
            yield
        finally:
            self.entries.append(dict(role=role, seconds=time.monotonic()-started))
            self.save()


def child(command, *, output, env, cwd, progress, ledger, poll_seconds):
    """Observe one bounded supervisor; keep its inclusive charge authoritative."""
    output.mkdir(parents=True, exist_ok=False)
    with (output / "stdout.log").open("x") as log:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    with ledger.measure("child_progress_inspection"):
                        progress(process.pid)
            with ledger.measure("child_terminal_observation"):
                progress(process.pid)
        except Exception as exc:
            # A telemetry failure must not abandon a live numerical supervisor.
            # Its existing inclusive cap remains authoritative; do not kill or
            # restart numerical work because an observer failed.
            write(output / "observation-error.json", dict(error_type=type(exc).__name__,
                error=str(exc), child_pid=process.pid, action="wait_for_existing_bounded_supervisor"))
            process.wait()
            raise
        finally:
            if process.returncode is not None:
                write(output / "returncode.json", dict(launcher_exit_code=process.returncode))
    return code


def wait_prices(path, *, unit, progress, ledger, poll_seconds, grace_seconds):
    while True:
        with ledger.measure("original_price_observation"):
            prices = read(path)
            state = service_state(unit)
            settled = ("supervisor_elapsed_seconds" in prices and
                       state["ActiveState"] not in {"active", "activating", "deactivating", "reloading"})
            progress.update(status="waiting_for_original_prices", price_status=prices["status"],
                original_price_service=state, updated_utc=datetime.now(timezone.utc).isoformat())
            write(progress["status_file"], progress)
            age = (datetime.now(timezone.utc)-datetime.fromisoformat(prices["started_utc"])).total_seconds()
            if settled:
                return prices
            if age > prices["max_wall_seconds"] + grace_seconds:
                raise RuntimeError("original pricing has no terminal supervisor receipt within its observation allowance")
            if state["ActiveState"] in {"inactive", "failed"} and not settled:
                raise RuntimeError("original pricing exited without its inclusive receipt")
        time.sleep(poll_seconds)


def capacity(gpu, *, output, ledger, wait_seconds, poll_seconds):
    # This helper uses only the standard library until its trusted device probe.
    from scripts.resume_modern_hmc_bridge import recovery_gpu_inventory
    started = time.monotonic()
    while True:
        try:
            with ledger.measure("trusted_capacity_inspection"):
                probe = recovery_gpu_inventory(gpu)
                write(output, probe)
            return
        except RuntimeError as exc:
            if "less than the declared 4096 MiB free-memory screen" not in str(exc):
                raise
            if time.monotonic()-started >= wait_seconds:
                raise RuntimeError("capacity waiting allowance exhausted") from exc
        time.sleep(poll_seconds)


def recover_terminal_prefix(prior, output, *, rows, prices_path, allocation, progress):
    """Audit a stopped controller's prefix without changing or rerunning it."""
    old = read(prior / "status.json")
    old_inputs, inputs = read(prior / "inputs.json"), read(output / "inputs.json")
    if (old_inputs != inputs or read(prior / "cohort-allocation.json") != allocation
            or old.get("source_root") != progress["source_root"]
            or old.get("numerical_source") != progress["numerical_source"]
            or old.get("environment") != progress["environment"]
            or [r["method"] for r in old["methods"]] != list(METHODS)):
        raise ValueError("recovery changed prior inputs, source, device or reservations")
    balance = allocation["available_numerical_seconds"]
    charges, absent, phases = [], False, set()
    recovered = output / "recovered"
    recovered.mkdir(exist_ok=False)
    prices = read(prices_path)
    for index, method in enumerate(METHODS):
        prior_row = old["methods"][index]
        phase = prior / method
        if not phase.exists() and prior_row.get("evidence_phase"):
            phase = Path(prior_row["evidence_phase"])
        if not phase.exists():
            if prior_row["status"] != "not_started":
                raise ValueError("started method lacks recoverable terminal evidence")
            absent = True
            continue
        if absent or phase.resolve() in phases:
            raise ValueError("terminal recovery must be an ordered distinct prefix")
        phases.add(phase.resolve())
        launch, funds = read(phase / "launch.json"), read(phase / "allocation.json")
        row = rows[index]
        priced = prices["rows"][index]
        if (launch["design_file"] != row["design_file"]
                or launch["design_sha256"] != row["design_sha256"]
                or launch["source_commit"] != progress["numerical_source"]["git_commit"]
                or Path(launch["command"][1]).resolve() != Path(progress["source_root"]) / "scripts/run_modern_hmc_frozen_full_model.py"
                or launch["environment"] != progress["environment"]
                or launch["prices_file"] != str(prices_path)
                or launch["prices_sha256"] != digest(prices_path)
                or launch["expected_binding_hash"] != priced["binding_hash"]
                or launch["expected_data_sha256"] != priced["data_sha256"]
                or launch["expected_geometry_sha256"] != priced["geometry_sha256"]
                or not math.isclose(funds["available_after_existing_held_local_reserve_seconds"], balance, rel_tol=0, abs_tol=1e-6)
                or not math.isclose(launch["numerical_cap_seconds"], next_cap(allocation, index, balance), rel_tol=0, abs_tol=1e-6)):
            raise ValueError("recovered child differs from frozen cohort or budget sequence")
        settlement, hashes = audit(phase)
        supervisor = read(phase / "run/supervisor.json")
        exit_file = phase / "process/returncode.json"
        exit_check = (validate_launcher_exit(read(exit_file)["launcher_exit_code"], supervisor)
                      if exit_file.exists() else dict(launcher_exit_code=None,
                          worker_exit_code=supervisor["exit_code"],
                          limitation="historical_launcher_code_not_recorded; strict_terminal_evidence_audited"))
        if (phase / "settlement.json").exists() and read(phase / "settlement.json") != settlement:
            raise ValueError("prior settlement differs from independent terminal audit")
        write(recovered / (method + "-settlement.json"), settlement)
        write(recovered / (method + "-evidence.json"), dict(phase=str(phase.resolve()),
            sha256=hashes, exit_check=exit_check))
        charges.append(settlement["numerical_charged_seconds"])
        balance = settlement["signed_balance_after_existing_reserve_seconds"]
        progress["methods"][index].update(status="settled", evidence_phase=str(phase.resolve()),
            cohort_status=settlement["cohort_status"], disposition=settlement["disposition"],
            charged_seconds=settlement["numerical_charged_seconds"], recovered_without_rerun=True)
        if not continue_after(settlement):
            raise RuntimeError("terminal infrastructure failure requires localized diagnosis")
    old_accounting = read(prior / "accounting.json")
    recorded = seconds(old_accounting["numerical_inclusive_seconds"])
    declared = math.fsum(seconds(r["charged_seconds"]) for r in old["methods"] if r["status"] == "settled")
    if (not math.isclose(recorded, declared, rel_tol=0, abs_tol=1e-6)
            or recorded > math.fsum(charges) + 1e-6):
        raise ValueError("prior numerical charges do not reconcile with terminal prefix")
    write(output / "recovery.json", dict(prior_output=str(prior),
        recovered_methods=list(METHODS[:len(charges)]), numerical_charged_once_seconds=math.fsum(charges),
        remaining_numerical_seconds=balance, original_evidence_unchanged=True,
        prior_accounting_sha256=digest(prior / "accounting.json")))
    return charges, balance


def run(args):
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("memory growth must be established in the child environment before import")
    if not args.gpu or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("visible GPU must match the trusted capacity probe")
    seconds(args.recovery_cap_seconds)
    for name in ("engineering_cap_seconds", "poll_seconds", "capacity_wait_seconds"):
        seconds(getattr(args, name), positive=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    progress = dict(status="preparing", plan=args.plan, command=[sys.executable, *sys.argv],
        driver_sha256=digest(__file__), started_utc=datetime.now(timezone.utc).isoformat(),
        status_file=str(output / "status.json"), methods=[dict(method=m, status="not_started") for m in METHODS],
        gpu_trust_basis="trusted_escalated_gpu_execution", method_ranking=None)
    write(output / "status.json", progress)
    ledger = None
    prior = getattr(args, "resume_from", None)
    prior_unit = getattr(args, "resume_unit", None)
    if bool(prior) != bool(prior_unit):
        raise ValueError("recovery requires both prior output and its service unit")
    prior = prior.resolve() if prior else None
    after_preparation = pricing_charge = None
    numerical_charges = []
    terminal_children_audited = True
    try:
        if prior is not None:
            terminal_children_audited = False
            state = service_state(prior_unit)
            if state.get("ActiveState") not in {"inactive", "failed"}:
                raise ValueError("prior controller must be stopped before recovery")
            old_status = read(prior / "status.json")
            if not old_status.get("finished_utc") or old_status["status"] != "requires_inspection":
                raise ValueError("recovery requires a terminal interrupted controller")
            write(output / "prior-service.json", dict(unit=prior_unit, state=state))
        ledger = EngineeringLedger(output / "engineering.json", cap=args.engineering_cap_seconds,
                                   initial_receipt=args.engineering_receipt,
                                   prior=prior / "engineering.json" if prior else None)
        with ledger.measure("inventory_and_preparation_reconciliation"):
            designs = [p.resolve() for p in args.design]
            rows = inventory(designs)
            preparation = read(args.preparation_accounting)
            after_preparation = reconcile_preparation(preparation)
            source = args.source_root.resolve()
            provenance = read(source / ".validation-source.json")
            if not isinstance(provenance.get("git_commit"), str) or not provenance["git_commit"]:
                raise ValueError("numerical snapshot has no Git commit provenance")
            overhead = measured_overhead(args.deployment_audit)
            progress.update(source_root=str(source), numerical_source=provenance,
                environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES",
                    "TF_FORCE_GPU_ALLOW_GROWTH", "OMP_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")})
            write(output / "inputs.json", dict(designs=rows, after_preparation_seconds=after_preparation,
                preparation_accounting=dict(path=str(args.preparation_accounting), sha256=digest(args.preparation_accounting)),
                overhead_comparator=dict(path=str(args.deployment_audit), sha256=digest(args.deployment_audit),
                    trial_overhead_seconds=overhead, interpretation="transferred_amortized_non_native_cost_hypothesis")))
        prices_path = args.prices.resolve()
        prices = wait_prices(prices_path, unit=args.price_unit, progress=progress, ledger=ledger,
                             poll_seconds=args.poll_seconds, grace_seconds=args.capacity_wait_seconds)
        with ledger.measure("original_price_attempt_accounting"):
            if prices["source_root"] != str(source) or prices["numerical_source"] != provenance:
                raise ValueError("pricing changed the immutable numerical source")
            audit_attempts(prices, prices_path.parent)
            pricing_charge = seconds(prices["supervisor_elapsed_seconds"], positive=True)
        if prices["status"] == "supervisor_budget_exhausted":
            if prior is not None:
                raise ValueError("terminal-cohort recovery requires complete frozen prices")
            if args.recovery_cap_seconds == 0:
                raise RuntimeError("pricing incomplete and no recovery allowance remains; no numerical cohort launched")
            with ledger.measure("original_timeout_recovery_audit"):
                resume_rows(prices, designs, seed_namespace=args.recovery_namespace, prior_directory=prices_path.parent)
                recovery_cap = min(args.recovery_cap_seconds, after_preparation-args.engineering_cap_seconds
                                   - seconds(prices["supervisor_elapsed_seconds"], positive=True))
                seconds(recovery_cap, positive=True)
                recovery_phase = output / "pricing-recovery-01"
                command = [sys.executable, str(Path(__file__).with_name("price_modern_hmc_frozen_design.py")),
                    "--source-root", str(source), "--design", *(str(p) for p in designs),
                    "--output", str(recovery_phase / "run"), "--plan", args.plan,
                    "--max-wall-seconds", str(recovery_cap), "--seed-namespace", str(args.recovery_namespace),
                    "--resume-prices", str(prices_path)]
                progress.update(status="recovering_missing_prices", recovery_cap_seconds=recovery_cap)
                write(output / "status.json", progress)
                write(output / "pricing-recovery-launch.json", dict(command=command,
                    original_receipt=str(prices_path), original_sha256=digest(prices_path),
                    inclusive_cap_seconds=recovery_cap, returned_calls_reused_only=True))
            capacity(args.gpu, output=output / "recovery-capacity.json", ledger=ledger,
                     wait_seconds=args.capacity_wait_seconds, poll_seconds=args.poll_seconds)
            terminal_children_audited = False
            child(command, output=recovery_phase, env=os.environ.copy(), cwd=source,
                  progress=lambda pid: observe_child(progress, output, "pricing_recovery", recovery_phase / "run", pid),
                  ledger=ledger, poll_seconds=args.poll_seconds)
            prices_path = recovery_phase / "run/result.json"
            prices = read(prices_path)
            with ledger.measure("recovered_price_attempt_accounting"):
                audit_attempts(prices, prices_path.parent)
                pricing_charge = seconds(prices["supervisor_elapsed_seconds"], positive=True)
                terminal_children_audited = True
        with ledger.measure("complete_price_audit_and_all_method_reservation"):
            if prices["source_root"] != str(source) or prices["numerical_source"] != provenance:
                raise ValueError("pricing changed the immutable numerical source")
            report = build_report(prices_path, designs,
                                 available_seconds=after_preparation-args.engineering_cap_seconds-pricing_charge,
                                 trial_overhead_seconds=overhead, fixed_overhead_seconds=0.)
            write(output / "price-audit.json", report)
            allocation = allocate_cohort(report, designs, after_preparation=after_preparation,
                                         engineering_hold=args.engineering_cap_seconds)
            write(output / "cohort-allocation.json", allocation)
        balance = allocation["available_numerical_seconds"]
        if prior is not None:
            with ledger.measure("recover_terminal_prefix"):
                numerical_charges, balance = recover_terminal_prefix(prior, output, rows=rows,
                    prices_path=prices_path, allocation=allocation, progress=progress)
                terminal_children_audited = True
                progress.update(status="terminal_prefix_recovered", remaining_numerical_seconds=balance)
                write(output / "status.json", progress)
        recovered_count = len(numerical_charges)
        for index, method in enumerate(METHODS):
            if index < recovered_count:
                continue
            phase = output / method
            with ledger.measure("method_reservation_and_launch_record"):
                cap = next_cap(allocation, index, balance)
                phase.mkdir(exist_ok=False)
                row = rows[index]
                priced = prices["rows"][index]
                if digest(row["design_file"]) != row["design_sha256"]:
                    raise ValueError("frozen design changed before launch")
                command = [sys.executable, str(source / "scripts/run_modern_hmc_frozen_full_model.py"),
                    "--design", row["design_file"], "--output", str(phase / "run"), "--max-wall-seconds", str(cap)]
                write(phase / "allocation.json", dict(available_after_existing_held_local_reserve_seconds=balance,
                    numerical_cap_seconds=cap, unallocated_after_numerical_reservation_seconds=balance-cap,
                    existing_local_reserve_still_held_seconds=preparation["existing_held_reserve_seconds"],
                    prior_balance_carried_separately_seconds=preparation["prior_balance_carried_separately_seconds"],
                    funding_sources=[dict(allocation_id="sir_post_preparation_after_prices_and_engineering_hold",
                                         available_seconds=balance)],
                    all_method_reservations_sha256=digest(output / "cohort-allocation.json")))
                write(phase / "launch.json", dict(command=command, numerical_cap_seconds=cap,
                    source_commit=provenance["git_commit"], expected_binding_hash=priced["binding_hash"],
                    design_file=row["design_file"], design_sha256=row["design_sha256"],
                    expected_data_sha256=priced["data_sha256"], expected_geometry_sha256=priced["geometry_sha256"],
                    environment=progress["environment"], plan=args.plan,
                    prices_file=str(prices_path), prices_sha256=digest(prices_path)))
                progress["methods"][index].update(status="waiting_for_capacity", numerical_cap_seconds=cap)
                write(output / "status.json", progress)
            capacity(args.gpu, output=phase / "capacity.json", ledger=ledger,
                     wait_seconds=args.capacity_wait_seconds, poll_seconds=args.poll_seconds)
            terminal_children_audited = False
            code = child(command, output=phase / "process", env=os.environ.copy(), cwd=source,
                progress=lambda pid: observe_child(progress, output, method, phase / "run", pid),
                ledger=ledger, poll_seconds=args.poll_seconds)
            with ledger.measure("strict_terminal_evidence_and_budget_audit"):
                settlement, hashes = audit(phase)
                exit_check = validate_launcher_exit(code, read(phase / "run/supervisor.json"))
                write(phase / "launcher-exit-check.json", exit_check)
                write(phase / "settlement.json", settlement)
                write(phase / "terminal-evidence-sha256.json", hashes)
                numerical_charges.append(settlement["numerical_charged_seconds"])
                terminal_children_audited = True
                balance = settlement["signed_balance_after_existing_reserve_seconds"]
                progress["methods"][index].update(status="settled", cohort_status=settlement["cohort_status"],
                    disposition=settlement["disposition"], charged_seconds=settlement["numerical_charged_seconds"])
                progress.update(status="method_settled", remaining_numerical_seconds=balance)
                write(output / "status.json", progress)
                if not continue_after(settlement):
                    raise RuntimeError("terminal infrastructure failure requires localized diagnosis")
                if balance <= 0 and index < len(METHODS)-1:
                    raise RuntimeError("inclusive costs exhausted the remaining numerical balance")
        progress.update(status="completed_bounded_cohort" if balance >= 0 else "completed_with_budget_shortfall",
            remaining_numerical_seconds=balance, numerical_shortfall_seconds=max(0., -balance),
            full_matrix_complete=False, release_readiness=False, method_ranking=None)
        return 0 if balance >= 0 else 1
    except Exception as exc:
        progress.update(status="requires_inspection", error_type=type(exc).__name__, error=str(exc))
        return 1
    finally:
        if ledger is not None:
            ledger.save(terminal=terminal_children_audited, enforce=False)
            if after_preparation is not None and pricing_charge is not None:
                signed_balance = after_preparation-pricing_charge-math.fsum(numerical_charges)-ledger.spent
                write(output / "accounting.json", dict(
                    status="settled" if terminal_children_audited else "requires_terminal_child_accounting",
                    post_preparation_seconds=after_preparation, pricing_inclusive_seconds=pricing_charge,
                    numerical_inclusive_seconds=math.fsum(numerical_charges),
                    exclusive_engineering_seconds=ledger.spent,
                    signed_balance_seconds=signed_balance if terminal_children_audited else None,
                    unused_engineering_hold_released_seconds=max(0., ledger.cap-ledger.spent)
                        if terminal_children_audited else 0.,
                    shortfall_seconds=max(0., -signed_balance) if terminal_children_audited else None,
                    existing_reserves_excluded=True, overlapping_child_time_not_added=True))
        progress.update(finished_utc=datetime.now(timezone.utc).isoformat())
        write(output / "status.json", progress)


def observe_child(progress, output, method, run, pid):
    summary = {}
    for name in ("manifest.json", "result.json", "supervisor.json"):
        path = run / name
        if path.exists():
            value = read(path)
            summary[name] = {key: value[key] for key in ("status", "elapsed_seconds", "completion_status") if key in value}
    progress.update(status="running_" + method, child_pid=pid, child_progress=summary,
                    updated_utc=datetime.now(timezone.utc).isoformat())
    for row in progress["methods"]:
        if row["method"] == method:
            row.update(status="running", child_pid=pid, child_progress=summary)
    write(output / "status.json", progress)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "prices", "output", "preparation-accounting", "deployment-audit", "engineering-receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--design", type=Path, nargs=4, required=True)
    parser.add_argument("--price-unit", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--recovery-namespace", type=int, required=True)
    parser.add_argument("--recovery-cap-seconds", type=float, required=True,
        help="inclusive missing-price retry ceiling; zero disables automatic recovery")
    parser.add_argument("--engineering-cap-seconds", type=float, required=True)
    parser.add_argument("--capacity-wait-seconds", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--resume-unit")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
