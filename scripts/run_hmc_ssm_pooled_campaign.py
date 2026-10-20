"""Continue SSM pilots and main fits in one progress-aware, bounded pool."""
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
import shutil
import subprocess
import time

from bayesfilter.testing.inference_validation.campaign_pool import run_pool
from bayesfilter.testing.inference_validation.campaign_recovery import (
    checked_frozen_source, continue_frozen_fit, start_frozen_fit, _runtime_checked,
    check_checkpoint_device,
)
from bayesfilter.testing.inference_validation import ssm_campaign
from bayesfilter.testing.inference_validation.designs import ValidationDesign, digest
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation.references.intervals import report_saved_intervals
from scripts.run_hmc_c1_recovery_then_ssm import settle
from scripts.run_hmc_ssm_priced_main import numerical_workload

PLAN = "docs/plans/bayesfilter-hmc-ssm-progress-continuation-plan-2026-10-02.md"
ROOT = REPO / "docs/plans/artifacts/hmc-ssm-progress-continuation-2026-10-02"
PREVIOUS_RUN = REPO / "docs/plans/artifacts/hmc-ssm-pooled-repair-2026-09-30/r1"
PREPARED = REPO / "docs/plans/artifacts/hmc-shared-gpu-recovery-2026-09-29/prepared-r5"
OLD_MAIN = REPO / "docs/plans/artifacts/hmc-ssm-main-2026-09-29/r1"
LEDGER = ROOT / "grant-ledger.json"
PROGRESS = REPO / "docs/plans/artifacts/hmc-repair-master-2026-09-16/program-progress.json"
QUANTUM = 1775.  # Previously executed fairness quantum, not a full-fit forecast.
INITIAL_QUANTUM = 3600.  # Previous large-model bootstrap allowance.
CLOSEOUT = 30.
RESERVED_OTHER = 1200.
GRANT_CAP = 48 * 3600.  # New owner allowance, including service shutdown.
CAMPAIGN_ID = "hmc-ssm-progress-continuation-2026-10-02"


def fit_price(cell, source_identity):
    """Preserve all cumulative process cost, including unsuccessful attempts."""
    cell = Path(cell)
    design = read_json(cell / "isolated_design.json")
    parsed = ValidationDesign.from_payload(design)
    identity = read_json(cell / "isolation_identity.json")
    if identity != {"design_identity": parsed.identity, "source_identity": source_identity,
                    "reuse_leapfrog_graphs": True}:
        raise ValueError("price design or source changed")
    path = cell / "replication-0000"
    exits = sorted(path.glob("process-attempt-*-exit.json"))
    receipts = [read_json(p) for p in exits]
    launches = sorted(path.glob("process-attempt-*-launch.json"))
    if [p.name.replace("-launch.json", "-exit.json") for p in launches] != [p.name for p in exits]:
        raise ValueError("unreconciled price launch")
    costs = [max(r["elapsed_seconds"], r.get("continuation_invocation_seconds", 0.),
                 r.get("invocation_seconds", 0.)) for r in receipts]
    if any(not math.isfinite(c) or c < 0 for c in costs):
        raise ValueError("invalid cumulative price")
    row = {"cell": str(cell), "design": design, "status": "incomplete",
           "receipts": [str(p) for p in exits], "final_assessment": False,
           "seconds": sum(costs), "posterior_outcome_used_for_pricing": False}
    if cell.parent == PREPARED / "pricing":
        index = read_json(PREPARED / "pricing/run_index.json")
        if index["source"]["identity"] != source_identity:
            raise ValueError("pilot price index source changed")
        outer = sum(r["elapsed_seconds"] for r in index["jobs"][design["design_id"]]["attempts"])
        base = sum(c for c, r in zip(costs, receipts) if not r.get("campaign_continuation"))
        row["seconds"] += max(0., outer - base)
    if not receipts or receipts[-1]["status"] != "complete":
        return row
    assessment = path / "independent_assessment.json"
    if file_hash(assessment) != receipts[-1].get("assessment_sha256"):
        raise ValueError("completed fit assessment changed")
    fit = read_json(path / "fit_identity.json")
    if (fit["source_identity"] != source_identity or fit["design"] != parsed.identity
            or fit["fit_id"] != 0 or fit["data"] != digest(parsed.options.get("data"))
            or fit["reuse_leapfrog_graphs"] is not True):
        raise ValueError("price fit identity changed")
    for receipt_path, receipt in zip(exits, receipts):
        _runtime_checked(path, str(receipt_path).removesuffix("-exit.json"), parsed, identity)
        if receipt.get("campaign_continuation"):
            evidence = Path(receipt["allocation_file"]).parent / "preserved-evidence.json"
            if (not receipt.get("preserved_evidence_verified")
                    or any(file_hash(p) != sha for p, sha in read_json(evidence).items())):
                raise ValueError("continued price evidence changed")
    row["final_assessment"] = True
    pipeline = read_json(path / "pipeline.json")
    members = [m for m in pipeline["members"] if m.get("status") == "assessed"]
    if (pipeline["completion"] == "complete" and
            len(members) == design["options"]["posterior_member_count"] and
            all(m.get("recorded_retained_count", 0) >=
                design["options"]["posterior_settings"]["retained_min_results"] for m in members)):
        row["status"] = "complete_workload"
    return row


def make_job(job_id, case, cell, *, design=None, first_quantum=QUANTUM):
    path = Path(cell) / "replication-0000"
    prior = [read_json(p) for p in sorted(path.glob("process-attempt-*-exit.json"))]
    return {"job_id": job_id, "case": case, "cell": str(cell),
            "design": design, "first_quantum_seconds": first_quantum,
            "previous_seconds": sum(r["elapsed_seconds"] for r in prior),
            "previous_continuations": sum(bool(r.get("campaign_continuation")) for r in prior)}


def executor(source, output):
    def execute(job, allowance, deadline, attempt):
        cell = Path(job["cell"])
        if not (cell / "isolated_design.json").exists():
            if job.get("design") is None:
                raise ValueError("missing original fit")
            return start_frozen_fit(cell=cell, source=source,
                design=ValidationDesign.from_payload(job["design"]),
                quantum_seconds=allowance, deadline=deadline)
        # Authorize exactly the next turn. The pool's absolute deadline owns
        # the total allocation; earlier attempts are never reset or forgotten.
        prior = [read_json(p) for p in sorted(
            (cell / "replication-0000").glob("process-attempt-*-exit.json"))]
        cap = sum(r["elapsed_seconds"] for r in prior) + allowance
        return continue_frozen_fit(cell=cell, source=source, replication=0,
            output=output / job["job_id"] / f"attempt-{attempt:02d}",
            cumulative_cap_seconds=cap, quantum_seconds=allowance, deadline=deadline,
            max_additional_attempts=sum(bool(r.get("campaign_continuation")) for r in prior) + 1,
            repair_device_startup_failure=True)
    return execute


class ContinuationInventory:
    """Original slots plus pilot dependencies, refreshed from native receipts."""

    def __init__(self, root, previous, source_identity):
        self.root, self.previous = Path(root), Path(previous)
        self.source_identity = source_identity
        self.template = read_json(PREPARED / "main-unpriced.json")["designs"]
        prior = read_json(self.previous / "result.json")
        if prior.get("exit_code") != 0:
            raise ValueError("prior coordinator failure requires diagnosis")
        slots = prior["original_slots"]
        self.prior_slots = {s["design_id"]: s for s in slots}
        original_ids = {d["design_id"] for d in self.template}
        if (len(slots) != 32 or len(self.prior_slots) != 32
                or set(self.prior_slots) != original_ids):
            raise ValueError("continuation must preserve all original 32 slots")
        pilot_file = self.previous / "pilot-jobs.json"
        saved_jobs = read_json(pilot_file if pilot_file.exists() else self.previous / "repair-jobs.json")
        self.pilots = []
        for j in saved_jobs:
            if j["job_id"] in original_ids:
                continue
            actual = read_json(Path(j["cell"]) / "isolated_design.json")
            if j.get("design") is not None and digest(actual) != digest(j["design"]):
                raise ValueError("saved pilot design changed")
            self.pilots.append(make_job(j["job_id"], j["case"], j["cell"], design=actual))
        if len({j["job_id"] for j in self.pilots}) != len(self.pilots):
            raise ValueError("duplicate pilot continuation")
        self.price_cells = list(dict.fromkeys([
            *(PREPARED / "pricing" / ("price-K" + str(i)) for i in range(8)),
            *(Path(j["cell"]) for j in self.pilots)]))
        self.cells = {}
        self.preserved_assessments = {}
        for original in self.template:
            ident = original["design_id"]
            slot = self.prior_slots[ident]
            if slot["case"] != original["options"]["campaign_case"]:
                raise ValueError("original slot case changed")
            cell = Path(slot.get("cell", self.previous / "main" / ident))
            if (cell / "isolated_design.json").exists():
                actual = read_json(cell / "isolated_design.json")
                expected = self.main_design(original)
                if (digest(numerical_workload(actual)) != digest(numerical_workload(expected))
                        or actual["seed"] != expected["seed"]
                        or actual["options"].get("data") != expected["options"].get("data")):
                    raise ValueError("continuation main design changed")
                self.cells[ident] = cell
                price = fit_price(cell, source_identity)
                if price["final_assessment"]:
                    assessment = cell / "replication-0000/independent_assessment.json"
                    self.preserved_assessments[str(assessment)] = file_hash(assessment)
                elif slot["status"] in {"preserved_complete", "complete", "reused_final_assessment"}:
                    raise ValueError("prior completed assessment is no longer final")
            elif slot["status"] in {"preserved_complete", "complete", "reused_final_assessment"}:
                raise ValueError("prior completed cell is missing")
            else:
                self.cells[ident] = self.root / "main" / ident
        self.snapshot = {}

    @staticmethod
    def main_design(original):
        d = ValidationDesign.from_payload(original)
        if d.options["campaign_case"] in {"K5", "K6"}:
            d = replace(d, options={**d.options, "bootstrap_initialization_rounds": 20,
                                   "preparation_max_restarts": 3})
        return d.payload()

    def refresh(self, rows):
        """Return only newly eligible jobs; terminal outcomes never requeue."""
        if any(file_hash(p) != sha for p, sha in self.preserved_assessments.items()):
            raise ValueError("preserved final assessment changed")
        seen = {r["job_id"]: r for r in rows}
        prices = [fit_price(p, self.source_identity) for p in self.price_cells]
        by_cell = {p["cell"]: p for p in prices}
        jobs, pilots, slots = [], [], []
        for job in self.pilots:
            price = by_cell[job["cell"]]
            status = ("preserved_complete" if price["final_assessment"] else
                      seen.get(job["job_id"], {}).get("status", "queued"))
            pilots.append({"job_id": job["job_id"], "cell": job["cell"],
                           "case": job["case"], "status": status,
                           "price_status": price["status"], "seconds": price["seconds"]})
            if not price["final_assessment"] and job["job_id"] not in seen:
                jobs.append(job)
        main_jobs_by_case = {}
        for original in self.template:
            ident, case = original["design_id"], original["options"]["campaign_case"]
            cell, design = self.cells[ident], self.main_design(original)
            price = fit_price(cell, self.source_identity) if (cell / "isolated_design.json").exists() else None
            slot = {"design_id": ident, "case": case, "cell": str(cell),
                    "original_design_identity": ValidationDesign.from_payload(original).identity,
                    "executed_design_identity": ValidationDesign.from_payload(
                        price["design"] if price else design).identity,
                    "observed_fit_seconds": price["seconds"] if price else 0.}
            if price and price["final_assessment"]:
                slot["status"] = "complete" if ident in seen else "preserved_complete"
            elif ident in seen:
                slot["status"] = seen[ident]["status"]
            else:
                matching = [p for p in prices if p["status"] == "complete_workload"
                            and digest(numerical_workload(p["design"])) == digest(numerical_workload(design))]
                # Existing numerical work remains eligible without re-pricing;
                # fresh main fits still require a complete development workload.
                if price or matching:
                    quantum = QUANTUM if price else min(INITIAL_QUANTUM,
                        max(QUANTUM, math.ceil(max(p["seconds"] for p in matching))))
                    job = make_job(ident, case, cell, design=None if price else design,
                                   first_quantum=quantum)
                    main_jobs_by_case.setdefault(case, []).append(job)
                    slot.update(status="queued", price_cells=[p["cell"] for p in matching])
                else:
                    slot["status"] = "unpriced_workload"
            slots.append(slot)
        # Each pilot gets a turn; newly eligible main slots are interleaved by
        # model when appended to the shared queue.
        order = ("K0", "K2", "K1", "K3", "K5", "K6", "K4", "K7")
        jobs.extend(main_jobs_by_case[c][i] for i in range(4) for c in order
                    if i < len(main_jobs_by_case.get(c, [])))
        self.snapshot = {"original_slots": slots, "pilots": pilots, "prices": prices,
                         "posterior_outcomes_used_for_pricing": False}
        return jobs

    def check_devices(self):
        for cell in dict.fromkeys([*(j["cell"] for j in self.pilots), *self.cells.values()]):
            check_checkpoint_device(cell)


def check_inputs(source):
    """Check the prepared numerical inputs and all original main identities."""
    frozen = checked_frozen_source(source)
    for path, sha in read_json(PREPARED / "prepared-inputs.json").items():
        if file_hash(PREPARED / path) != sha:
            raise ValueError("frozen prepared input changed")
    ssm_campaign.check_preflight(PREPARED)
    for original in read_json(PREPARED / "pricing.json")["designs"]:
        if read_json(PREPARED / "pricing" / original["design_id"] / "isolated_design.json") != original:
            raise ValueError("original pilot design changed")
    template = read_json(PREPARED / "main-unpriced.json")["designs"]
    ids = [r["design_id"] for r in template]
    if len(ids) != 32 or len(set(ids)) != 32:
        raise ValueError("original 32-slot inventory changed")
    by_id = {r["design_id"]: r for r in template}
    for row in read_json(OLD_MAIN / "status.json")["outcomes"]:
        original = by_id[row["design_id"]]
        if row.get("receipt"):
            price = fit_price(Path(row["receipt"]).parents[1], frozen["identity"])
            actual = price["design"]
            if (digest(numerical_workload(actual)) != digest(numerical_workload(original))
                    or actual["seed"] != original["seed"]
                    or actual["options"]["data"] != original["options"]["data"]):
                raise ValueError("original main workload changed")
    return frozen


def available_budget(ledger):
    if ledger.get("active_reservation"):
        raise ValueError("another campaign is active")
    receipts = [r["receipt"] for r in ledger["records"]]
    costs = [r["elapsed_seconds"] for r in ledger["records"]]
    if len(set(receipts)) != len(receipts):
        raise ValueError("duplicate enclosing receipt charge")
    values = [*costs, ledger["grant_gpu_seconds"], ledger["remaining_gpu_seconds"]]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("invalid grant accounting")
    if not math.isclose(ledger["grant_gpu_seconds"] - sum(costs),
                        ledger["remaining_gpu_seconds"], rel_tol=0., abs_tol=1e-6):
        raise ValueError("grant balance and receipts disagree")
    campaign_spent = sum(r["elapsed_seconds"] for r in ledger["records"]
                         if r.get("campaign_id") == CAMPAIGN_ID)
    return max(0., min(ledger["remaining_gpu_seconds"] - RESERVED_OTHER,
                       GRANT_CAP - campaign_spent))


def coordinator_manifest(root, args, cap):
    files = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(REPO) and path.suffix == ".py":
                files.add(path)
    for relative in ("engines/pipeline.py", "references/analytic.py", "references/ssm.py"):
        files.add(REPO / "bayesfilter/testing/inference_validation" / relative)
    for path in files:
        saved = root / "coordinator-source" / path.relative_to(REPO)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
    return {"command": [sys.executable, *sys.argv], "environment": sys.executable,
        "gpu_uuid": args.gpu, "memory_growth_required": True,
        "TF_FORCE_GPU_ALLOW_GROWTH": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "cpu_count": os.cpu_count(), "threads_per_worker": 2,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "plan_file": PLAN, "result_file": str(root / "result.json"),
        "previous_run": str(args.previous_run), "grant_ledger": str(args.ledger),
        "maximum_seconds_including_shutdown": cap + CLOSEOUT,
        "coordinator_sources": {str(p.relative_to(REPO)): file_hash(p) for p in files},
        "original_main_denominator": 32, "fixed_attempt_limit": None,
        "quantum_seconds": QUANTUM, "maximum_first_quantum_seconds": INITIAL_QUANTUM}


def publish_master(args, status, slots, current=None):
    path = getattr(args, "progress_file", None)
    if path is None:
        return
    progress = read_json(path)
    ledger = read_json(args.ledger)
    complete = sum(s["status"] in {"preserved_complete", "complete", "reused_final_assessment"}
                   for s in slots)
    progress.update(active_phase="SSM_progress_continuation_20261002", status=status,
        active_service=args.service if status == "running" else None,
        active_gpu_ledger=str(args.ledger), active_campaign_root=str(args.output),
        active_campaign_plan=PLAN, original_main_complete_assessments=complete,
        original_main_denominator=32, current_job=current,
        next_phase="continue_shared_pool" if status == "running" else "SSM_terminal_evidence_review",
        updated_utc=datetime.now(timezone.utc).isoformat())
    progress.setdefault("remaining_budget_seconds", {})["gpu"] = ledger["remaining_gpu_seconds"]
    progress["remaining_budget_seconds_scope"] = (
        "Settled grant balance; subtract active reservation and protected 1200 seconds before allocation. "
        "CPU reference/test worker seconds are recorded separately in the active CPU ledger.")
    progress["active_gpu_reservation"] = ledger.get("active_reservation")
    write_json(path, progress)


def worker(args):
    root, source = args.output, PREPARED / "source"
    started = time.monotonic()
    if (os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu):
        raise ValueError("recorded GPU and memory growth required before import")
    ledger = read_json(args.ledger)
    if not math.isfinite(args.seconds) or not math.isfinite(args.deadline):
        raise ValueError("finite launch budget and deadline required")
    cap = min(args.seconds, GRANT_CAP - CLOSEOUT, available_budget(ledger) - CLOSEOUT)
    deadline = min(args.deadline, started + cap - CLOSEOUT)
    if cap <= CLOSEOUT * 2 or deadline <= started:
        raise ValueError("continuation budget exhausted")
    root.mkdir(parents=True, exist_ok=False)
    receipt_path = root / "execution.json"
    ledger["active_reservation"] = {"service": args.service, "receipt": str(receipt_path),
        "campaign_id": CAMPAIGN_ID,
        "maximum_gpu_seconds_including_shutdown": cap + CLOSEOUT}
    write_json(args.ledger, ledger)
    code, results, inventory = 1, {}, None
    latest_rows = []
    try:
        manifest = coordinator_manifest(root, args, cap)
        write_json(root / "manifest.json", manifest)
        frozen = check_inputs(source)
        write_json(root / "manifest.json", {**manifest, "source": frozen})
        inventory = ContinuationInventory(root, args.previous_run, frozen["identity"])
        inventory.check_devices()
        write_json(root / "pilot-jobs.json", inventory.pilots)
        write_json(root / "preserved-assessments.json", inventory.preserved_assessments)

        def refresh(rows):
            jobs = inventory.refresh(rows)
            write_json(root / "phase-refresh.json", {**inventory.snapshot,
                "newly_eligible_jobs": jobs, "elapsed_seconds": time.monotonic() - started,
                "remaining_seconds": max(0., deadline - time.monotonic()),
                "next_phase": "continue_eligible_work" if jobs else "continue_progress_queue"})
            return jobs

        def checkpoint(state):
            nonlocal latest_rows
            latest_rows = state["rows"]
            write_json(root / "status.json", {"status": "running", "phase": "shared_pool",
                "original_main_denominator": 32, **inventory.snapshot, **state,
                "elapsed_seconds": time.monotonic() - started,
                "remaining_seconds": max(0., deadline - time.monotonic())})
            publish_master(args, "running", inventory.snapshot["original_slots"], state["current"])

        results["pool"] = run_pool([], executor(source, root / "attempts"),
            deadline=deadline, quantum_seconds=QUANTUM, max_attempts=None,
            discover=refresh, checkpoint=checkpoint)
        inventory.refresh(results["pool"]["rows"])
        results.update(inventory.snapshot)
        reports = []
        for row in results["original_slots"]:
            cell = Path(row["cell"])
            if row["status"] in {"preserved_complete", "complete", "reused_final_assessment"}:
                output = root / "interval-reports" / (row["design_id"] + ".json")
                report_saved_intervals(cell, output)
                reports.append(str(output))
        results["interval_reports"] = reports
        code = 0
    except BaseException as exc:
        write_json(root / "failure.json", {"exception": type(exc).__name__, "reason": str(exc)})
        raise
    finally:
        elapsed = time.monotonic() - started
        if inventory is not None:
            results.setdefault("original_slots", inventory.snapshot.get("original_slots", []))
            results.setdefault("pilots", inventory.snapshot.get("pilots", []))
        results.setdefault("pool", {"rows": latest_rows, "queue_closed": False})
        complete = sum(s["status"] in {"preserved_complete", "complete", "reused_final_assessment"}
                       for s in results.get("original_slots", []))
        results.update(exit_code=code, elapsed_seconds=elapsed, original_main_denominator=32,
            original_main_complete_assessments=complete, all_original_slots_assessed=complete == 32,
            posterior_calibration_claim=False)
        # Settle the enclosing process even if terminal reporting fails. All
        # nested numerical attempts remain explanatory receipts, not extra charges.
        receipt = {"receipt": str(receipt_path), "resource": "gpu",
            "campaign_id": CAMPAIGN_ID, "elapsed_seconds": elapsed + CLOSEOUT,
            "measured_seconds_before_closeout": elapsed, "closeout_reserve_seconds": CLOSEOUT,
            "exit_code": code, "ended_utc": datetime.now(timezone.utc).isoformat(),
            "accounting": "enclosing elapsed plus conservative closeout reserve, charged once; nested fits not additional"}
        write_json(receipt_path, receipt)
        settle(args.ledger, receipt)
        write_json(root / "result.json", results)
        write_json(root / "status.json", {"status": "closed" if code == 0 else "requires_diagnosis", **results})
        publish_master(args, "closed" if code == 0 else "requires_diagnosis",
                       results.get("original_slots", []))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--gpu", required=True)
    p.add_argument("--service", required=True)
    p.add_argument("--previous-run", type=Path, default=PREVIOUS_RUN)
    p.add_argument("--ledger", type=Path, default=LEDGER)
    p.add_argument("--progress-file", type=Path, default=PROGRESS)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--seconds", type=float, default=GRANT_CAP)
    p.add_argument("--deadline", type=float)
    args = p.parse_args()
    args.output, args.previous_run, args.ledger = (x.resolve() for x in
        (args.output, args.previous_run, args.ledger))
    signal.signal(signal.SIGTERM, lambda n, f: sys.exit(128 + n))
    if args.worker:
        return worker(args)
    if args.output.exists():
        raise ValueError("used output directory")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("finite positive launch budget required")
    total = min(args.seconds, GRANT_CAP, available_budget(read_json(args.ledger)))
    seconds = total - CLOSEOUT  # RuntimeMaxSec plus TimeoutStopSec stays inside total.
    if seconds <= 2 * CLOSEOUT:
        raise ValueError("no remaining campaign time")
    deadline = time.monotonic() + seconds - CLOSEOUT
    command = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--service-type=exec",
        "--unit", args.service, "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={math.floor(seconds)}", "--property=TimeoutStopSec=30",
        f"--working-directory={REPO}", f"--setenv=CUDA_VISIBLE_DEVICES={args.gpu}",
        "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true", "--setenv=TF_NUM_INTRAOP_THREADS=2",
        "--setenv=TF_NUM_INTEROP_THREADS=2", "--setenv=OMP_NUM_THREADS=2",
        "--setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0", sys.executable, str(Path(__file__).resolve()),
        str(args.output), "--gpu", args.gpu, "--service", args.service,
        "--previous-run", str(args.previous_run), "--ledger", str(args.ledger),
        "--progress-file", str(args.progress_file.resolve()),
        "--worker", "--seconds", str(seconds), "--deadline", str(deadline)]
    write_json(args.output.parent / (args.output.name + "-launch.json"),
        {"command": command, "plan_file": PLAN, "maximum_seconds_including_shutdown": total,
         "started_utc": datetime.now(timezone.utc).isoformat()})
    confirmed, result = False, None
    try:
        result = subprocess.run(command, check=False, timeout=total)
        confirmed = True
    finally:
        if not confirmed:
            confirmed = subprocess.run(["systemctl", "--user", "stop", args.service],
                check=False, timeout=CLOSEOUT).returncode == 0
        active = read_json(args.ledger).get("active_reservation", {})
        if confirmed and active.get("service") == args.service:
            receipt = {"receipt": active["receipt"], "resource": "gpu", "elapsed_seconds": total,
                "campaign_id": CAMPAIGN_ID,
                "exit_code": -1, "accounting": "conservative cap after confirmed group shutdown"}
            write_json(active["receipt"], receipt)
            settle(args.ledger, receipt)
    raise SystemExit(result.returncode if result is not None else 1)


if __name__ == "__main__":
    main()
