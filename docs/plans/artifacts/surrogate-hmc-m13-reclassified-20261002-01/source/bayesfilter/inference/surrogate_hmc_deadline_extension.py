"""Bounded deadline extension of an already running M13 master.

Wait for its ordinary resource stop, then resume the same saved sequence.
No source loaded by the existing worker is changed by this companion launcher.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

from . import surrogate_hmc_resume_program as previous
from .surrogate_hmc_program import ROOT, file_hash, run_job, write_json
from .surrogate_hmc_worker_guard import process_identity

PLAN = "docs/plans/surrogate-hmc-midnight-extension-2026-10-01.md"
PARENT = previous.ARTIFACTS / "surrogate-hmc-m13-resume-20261001-01"
DEVELOPMENT = previous.ARTIFACTS / "surrogate-hmc-m13-midnight-development-01"
SCRIPT = ROOT / "scripts/run_surrogate_hmc_deadline_extension.py"
DEADLINE = "2026-10-01T16:00:00+00:00"
ALLOWANCE = 73338.
POST_WARMUP_MAXIMUM = 20000
EXTRA_SOURCES = (
    "bayesfilter/inference/surrogate_hmc_deadline_extension.py",
    "bayesfilter/testing/surrogate_hmc_m13_retained_resume.py",
    "scripts/run_surrogate_hmc_deadline_extension.py",
    "bayesfilter/testing/surrogate_hmc_m13_cumulative.py",
)
ACCOUNTING_MIGRATION = {
    "bayesfilter/inference/surrogate_hmc_resume_program.py",
    "bayesfilter/inference/surrogate_hmc_deadline_extension.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def utc_now():
    return datetime.now(timezone.utc)


def remaining_seconds():
    return (datetime.fromisoformat(DEADLINE) - utc_now()).total_seconds()


def source_hashes(parent):
    expected = read(parent / "run-manifest.json")["source_hashes"]
    for name, digest in expected.items():
        if file_hash(ROOT/name) != digest:
            if name not in ACCOUNTING_MIGRATION or file_hash(parent/"source"/name) != digest:
                raise ValueError("parent numerical source closure changed during deadline extension")
    return {name: file_hash(ROOT/name) for name in set(expected) | set(EXTRA_SOURCES)}



def continuation_action(result):
    """A deadline extension never overrides a scientific or integrity stop."""
    receipt = result.get("execution", {})
    if (result.get("source_unchanged") is not True
            or receipt.get("accounting_complete") is not True):
        return "stop_invalid_parent"
    if receipt.get("exit_code") != 0 or receipt.get("status") != "COMPLETED":
        recovery = result.get("infrastructure_recovery", {})
        if (receipt.get("exit_code") == 1 and receipt.get("status") == "FAILED"
                and recovery.get("kind") == "saved_manifest_path_reporting_failure"
                and result["worker"]["post_warmup_draws_per_chain"] < POST_WARMUP_MAXIMUM):
            return recovery["action"]
        return "stop_invalid_parent"
    worker = result.get("worker", {})
    if result.get("qualified") is True:
        return "already_qualified"
    if worker.get("post_warmup_draws_per_chain", POST_WARMUP_MAXIMUM) >= POST_WARMUP_MAXIMUM:
        return "stop_sample_cap"
    if result.get("status") == "M13_RESUMED_BUDGET_EXHAUSTED":
        return "resume_verification"
    if (result.get("status") == "M13_RETAINED_BUDGET_EXHAUSTED"
            and worker.get("verification_passed") is True):
        return "resume_retained"
    if result.get("status") in ("M13_POSTERIOR_BUDGET_EXHAUSTED", "M13_POSTERIOR_MIXING_INCONCLUSIVE"):
        return "resume_retained" if worker.get("verification_passed") is True else "resume_verification"
    return "preserve_terminal_result"


def parent_input(parent):
    """Recover only the inspected reporting failure, keeping the failed receipt."""
    result = read(parent/"report.json")
    if continuation_action(result) != "stop_invalid_parent":
        return result
    receipt = result.get("execution", {})
    if (result.get("source_unchanged") is not True or receipt.get("accounting_complete") is not True
            or receipt.get("exit_code") != 1 or receipt.get("status") != "FAILED"):
        return result
    manifest = parent/"worker/tuning/manifest.json"
    log_path = parent/"worker/command.log"
    if not manifest.exists() or not log_path.exists():
        return result
    with log_path.open("rb") as stream:
        stream.seek(max(0, log_path.stat().st_size-16384))
        tail = stream.read().decode(errors="replace")
    if ("surrogate_hmc_resume_program.py" not in tail
            or "tuning_manifest_sha256=file_hash(tuning.artifact_manifest_path)" not in tail
            or not tail.rstrip().endswith("AttributeError: 'str' object has no attribute 'read_bytes'")):
        return result
    saved = read(manifest)
    sequence = saved.get("sequence", {})
    events = sequence.get("events", [])
    state = sequence.get("status")
    if (saved.get("artifact_authority") is not True or sequence.get("warmup_ready") is not True
            or not events or not sequence.get("chunks")
            or any(not event.get("checks") or not all(event["checks"].values()) for event in events)
            or state not in ("BUDGET_EXHAUSTED", "VERIFIED")
            or saved.get("passed") is not (state == "VERIFIED")
            or saved.get("handoff_eligible") is not (state == "VERIFIED")):
        return result
    action = "resume_retained" if saved["passed"] else "resume_verification"
    result["worker"].update(tuning_manifest=str(manifest), tuning_manifest_sha256=file_hash(manifest),
        verification_passed=saved["passed"], counts=sequence["counts"],
        post_warmup_draws_per_chain=sequence["counts"]["verification"], retained_draws_per_chain=0)
    result["infrastructure_recovery"] = dict(kind="saved_manifest_path_reporting_failure",
        action=action, parent_failure_preserved=True, command_log_sha256=file_hash(log_path),
        checkpoint_sha256=file_hash(manifest), full_public_artifact_validation_required=True,
        repeats_tuning=False, repeats_warmup=False)
    return result


def resource_ledger(parent_result):
    charges = dict(parent_result["ledger"]["charged"])
    receipts = []
    for path in sorted(DEVELOPMENT.glob("*/guard-receipt.json")):
        receipt = read(path)
        if receipt.get("accounting_complete") is not True:
            raise ValueError("extension preparation accounting unresolved")
        for key in ("cpu_worker_seconds", "gpu_seconds"):
            charges[key] += receipt[key]
        receipts.append(str(path))
    own = resource.getrusage(resource.RUSAGE_SELF)
    charges["cpu_worker_seconds"] += own.ru_utime + own.ru_stime
    elapsed = max(0., (utc_now()-datetime.fromisoformat(previous.START)).total_seconds())
    charges = {key: max(value, elapsed) for key, value in charges.items()}
    return dict(limits=dict(cpu_worker_seconds=ALLOWANCE, gpu_seconds=ALLOWANCE),
        charged=charges, jobs=[], prior_receipts=receipts,
        parent_ledger=parent_result["ledger"], conservative_elapsed_charge_floor=True)


def worker(output):
    """Both phases use the unchanged public tuning and frozen-runner APIs."""
    from . import tune_hmc_kernel, load_verified_tensorflow_hmc_result
    from .hmc_tensorflow_verified import VerifiedTensorFlowHMCConfig
    output = Path(output)
    started = time.monotonic()
    parent_result = parent_input(PARENT)
    action = continuation_action(parent_result)
    if action not in ("resume_verification", "resume_retained"):
        raise ValueError("parent is not a healthy resource-interrupted sequence")
    parent_worker = parent_result["worker"]
    parent_tuning = Path(parent_worker["tuning_manifest"])
    if file_hash(parent_tuning) != parent_worker["tuning_manifest_sha256"]:
        raise ValueError("parent tuning manifest changed")
    adapter, binding, fixture, provenance = previous.components()
    reference, calibration = previous.prerequisites(fixture)
    report = dict(status="RESUME_VALIDATING", qualified=False, plan=PLAN,
        deadline=DEADLINE, parent=str(PARENT), parent_report_sha256=file_hash(PARENT/"report.json"),
        action=action, infrastructure_recovery=parent_result.get("infrastructure_recovery"),
        scientific_attempt=2, post_warmup_maximum=POST_WARMUP_MAXIMUM,
        all_models_complete=False, ranking_supported=False, **provenance)
    write_json(output/"report.json", report)
    if action == "resume_verification":
        cfg = replace(VerifiedTensorFlowHMCConfig.from_payload(read(parent_tuning)["config"]),
            resume_manifest=str(parent_tuning), verification_max_results=POST_WARMUP_MAXIMUM)
        write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
            phase="verification", action="continue_saved_frozen_chains", deadline=DEADLINE))
        tuning = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=cfg,
            parameter_scales=None, output_dir=output/"tuning", runner_binding=binding)
    else:
        tuning = load_verified_tensorflow_hmc_result(parent_tuning, adapter=adapter, runner_binding=binding)
    saved = read(tuning.artifact_manifest_path)
    count = saved["sequence"]["counts"]["verification"]
    report.update(status="M13_RESUMED_"+saved["sequence"]["status"],
        verification_passed=tuning.passed, counts=saved["sequence"]["counts"],
        tuning_manifest=str(tuning.artifact_manifest_path),
        tuning_manifest_sha256=file_hash(Path(tuning.artifact_manifest_path)), checks=saved["checks"],
        final_verification_rhat=next((e["rhat"] for e in reversed(saved["sequence"]["events"]) if e.get("rhat")), None),
        post_warmup_draws_per_chain=count, retained_draws_per_chain=0)
    write_json(output/"report.json", report)
    prefix = PARENT/"worker/posterior/report.json" if action == "resume_retained" else None
    prefix = prefix if prefix is not None and prefix.exists() else None
    if prefix is not None and read(prefix) != parent_worker.get("posterior"):
        raise ValueError("parent posterior report changed")
    from bayesfilter.testing.surrogate_hmc_m13_cumulative import consume_chunks
    write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
        phase="posterior_sampling", action="assess_all_saved_draws_before_sampling", deadline=DEADLINE,
        post_warmup_maximum=POST_WARMUP_MAXIMUM, independent_validation_required=False))
    last_cost = saved["sequence"]["resume_timing"]["last_chunk_wall_seconds"]
    posterior = consume_chunks(tuning, binding, output/"posterior",
        max_results=POST_WARMUP_MAXIMUM, resume_report=prefix,
        allow_sampling=tuning.passed and previous.first_retained_chunk_fits(last_cost))
    report.update(status=posterior["status"], qualified=posterior["sampling_qualified"],
        qualification_scope=posterior["qualification_scope"], posterior=posterior,
        sampling_qualified=posterior["sampling_qualified"], posterior_reference_qualified=False,
        retained_draws_per_chain=posterior["draws_per_chain"],
        post_warmup_draws_per_chain=posterior["draws_per_chain"],
        continuation_draws_per_chain=posterior["continuation_draws_per_chain"],
        independent_validation_required=False)
    write_json(output/"next-phase-plan.json", posterior["next_phase_plan"])
    if report["post_warmup_draws_per_chain"] > POST_WARMUP_MAXIMUM:
        raise ValueError("post-warmup sample cap violated")
    report.update(wall_seconds=time.monotonic()-started,
        reference_sha256=file_hash(previous.REFERENCE), coverage_sha256=file_hash(previous.EXTENDED_COVERAGE),
        baseline_sha256=file_hash(previous.BASELINE))
    write_json(output/"report.json", report)
    return 0


def publish(output, status, progress):
    payload = dict(status=status, deadline=DEADLINE, deadline_hong_kong="2026-10-02T00:00:00+08:00",
        post_warmup_maximum=POST_WARMUP_MAXIMUM, part_of_existing_master=True,
        parent=str(PARENT), progress=progress, updated_at=utc_now().isoformat())
    write_json(output/"progress.json", payload)
    action = ("wait_then_resume_saved_chains" if status == "WAITING_FOR_PARENT" else
              "continue_same_frozen_kernel" if status == "RUNNING" else "report_bounded_result")
    write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True, status=status,
        action=action, deadline=DEADLINE, post_warmup_maximum=POST_WARMUP_MAXIMUM))
    checkpoint = (
        "# Active midnight extension of surrogate-HMC master\n\n"
        f"Question: same frozen M13 sampler, same posterior criteria. Status: {status}.\n"
        f"Deadline: {DEADLINE}, midnight ending October 1 in Hong Kong. Cap: 20,000 post-warmup draws/chain.\n"
        f"Plan: {PLAN}. Parent: {PARENT}. Current progress: {json.dumps(progress, sort_keys=True)}\n"
        f"NEXT: {action}. This is part of the existing master. "
        "Do not launch a duplicate or change the running numerical source. No extension beyond midnight is authorized.\n")
    (output/"checkpoint.md").write_text(checkpoint)
    previous.CHECKPOINT.write_text(checkpoint)


def collect_progress(directory):
    progress = {}
    for name in ("next-phase-plan.json", "tuning/fresh-progress.json", "posterior/report.json"):
        path = directory/name
        if path.exists():
            value = read(path)
            if name.endswith("fresh-progress.json"):
                progress["sequence"] = dict(status=value["status"], counts=value["counts"],
                    latest_rhat=next((e["rhat"] for e in reversed(value["events"]) if e.get("rhat")), None))
            elif name.startswith("posterior"):
                progress["posterior"] = {key: value.get(key) for key in ("status", "draws_per_chain", "qualified")}
            else:
                progress["next_phase_plan"] = value
    return progress


def execute(output):
    output = Path(output).resolve()
    if remaining_seconds() <= 120:
        raise ValueError("midnight deadline has already exhausted the available time")
    for path in previous.ARTIFACTS.glob("surrogate-hmc-m13-midnight-*/run-manifest.json"):
        candidate = read(path)
        owner = candidate.get("owner")
        if owner and process_identity(owner["pid"]) == owner and candidate.get("status") != "TERMINAL":
            raise ValueError("deadline extension already active")
    sources = source_hashes(PARENT)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(status="WAITING_FOR_PARENT", plan=PLAN, command=[sys.executable, *sys.argv],
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        cwd=str(ROOT), environment=sys.executable, owner=process_identity(os.getpid()),
        launched_at=utc_now().isoformat(), deadline=DEADLINE, original_start=previous.START,
        limits=dict(cpu_worker_seconds=ALLOWANCE, gpu_seconds=ALLOWANCE),
        post_warmup_maximum=POST_WARMUP_MAXIMUM, parent=str(PARENT), source_hashes=sources,
        data_seed=1704, seeds=read(previous.PARENT)["config"]["fresh_seed"], device="gpu",
        result_file=str(output/"report.json"), waiting_process_uses_gpu=False)
    write_json(output/"run-manifest.json", manifest)
    for name in sources:
        dest = output/"source"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    write_json(PARENT/"deadline-extension.json", dict(deadline=DEADLINE, extension_root=str(output),
        execution="automatic_saved_state_handoff_after_existing_worker_exit", plan=PLAN,
        post_warmup_maximum=POST_WARMUP_MAXIMUM))
    terminal = None
    try:
        while read(PARENT/"run-manifest.json")["status"] != "TERMINAL":
            parent_owner = read(PARENT/"run-manifest.json")["owner"]
            if process_identity(parent_owner["pid"]) != parent_owner:
                raise ValueError("parent supervisor vanished before saving its terminal result")
            if remaining_seconds() <= 120:
                raise ValueError("midnight reached while waiting for parent")
            publish(output, "WAITING_FOR_PARENT", collect_progress(PARENT/"worker"))
            time.sleep(5.)
        parent_result = parent_input(PARENT)
        action = continuation_action(parent_result)
        manifest.update(parent_report_sha256=file_hash(PARENT/"report.json"), action=action,
            infrastructure_recovery=parent_result.get("infrastructure_recovery"))
        if source_hashes(PARENT) != sources:
            raise ValueError("source drift before deadline continuation")
        if action not in ("resume_verification", "resume_retained"):
            terminal = dict(status=("PARENT_VALIDITY_OR_EXECUTION_VETO" if action == "stop_invalid_parent"
                                     else parent_result["status"]), qualified=action == "already_qualified",
                action=action, parent_result=str(PARENT/"report.json"), no_new_sampling=True,
                source_unchanged=parent_result.get("source_unchanged") is True, ledger=resource_ledger(parent_result))
        else:
            ledger = resource_ledger(parent_result)
            timeout = min(remaining_seconds(), ALLOWANCE-ledger["charged"]["gpu_seconds"])-120.
            cpu_allowance = ALLOWANCE-ledger["charged"]["cpu_worker_seconds"]-120.
            if min(timeout, cpu_allowance) <= 60:
                terminal = dict(status="MIDNIGHT_BUDGET_EXHAUSTED", qualified=False, ledger=ledger)
            else:
                manifest.update(status="RUNNING", ledger=ledger)
                write_json(output/"run-manifest.json", manifest)
                last = {}
                def progress(directory):
                    current = collect_progress(directory)
                    if current != last:
                        last.clear()
                        last.update(current)
                        publish(output, "RUNNING", current)
                receipt = run_job([sys.executable, str(SCRIPT), "worker", "--output", str(output/"worker")],
                    output/"worker", ledger, device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
                    max_timeout_seconds=ALLOWANCE, progress_callback=progress)
                progress(output/"worker")
                result = read(output/"worker/report.json") if (output/"worker/report.json").exists() else {}
                unchanged = (source_hashes(PARENT) == sources
                    and file_hash(PARENT/"report.json") == manifest["parent_report_sha256"])
                status = result.get("status", "WORKER_INCOMPLETE")
                if receipt.get("status") != "COMPLETED" or receipt.get("exit_code") != 0:
                    status = "WORKER_RESOURCE_OR_INFRASTRUCTURE_STOP"
                if not unchanged:
                    status = "SOURCE_DRIFT_VETO"
                terminal = dict(status=status, qualified=unchanged and result.get("qualified") is True
                    and status == "M13_POSTERIOR_SAMPLING_COMPLETE", worker=result, execution=receipt,
                    ledger=ledger, source_unchanged=unchanged, final_progress=last)
    except Exception as error:
        terminal = dict(status="EXTENSION_REPAIR_REQUIRED", qualified=False,
            error_type=type(error).__name__, error=str(error), source_unchanged=False)
    terminal.update(plan=PLAN, deadline=DEADLINE, all_models_complete=False, ranking_supported=False)
    write_json(output/"report.json", terminal)
    manifest.update(status="TERMINAL", terminal_status=terminal["status"], completed_at=utc_now().isoformat())
    if "ledger" in terminal:
        manifest["ledger"] = terminal["ledger"]
    write_json(output/"run-manifest.json", manifest)
    publish(output, terminal["status"], terminal.get("final_progress", {}))
    (output/"result.md").write_text(
        f"# M13 midnight continuation result\n\nStatus: {terminal['status']}. Qualified: {terminal['qualified']}.\n\n"
        "| Decision | Primary criterion | Veto diagnostics | Uncertainty | Next action | Not concluded |\n"
        "|---|---|---|---|---|---|\n"
        f"| {terminal['status']} | cumulative R-hat/health qualification {terminal['qualified']} | see worker and receipt | finite chains and endpoint approximation | report bounded result | exact likelihood, all models, efficiency |\n\n"
        "| Inference status | Finding |\n|---|---|\n"
        "| Hard veto screen | See saved health, source and execution checks |\n"
        "| Statistically supported ranking | None |\n"
        "| Descriptive-only differences | R-hat, acceptance and runtime |\n"
        "| Default-readiness | Not established |\n"
        "| Next evidence needed | Determined by cumulative posterior checks |\n\n"
        "Alternative explanations for an inconclusive result include finite-chain mixing and endpoint approximation. "
        "A longer deadline alone does not establish convergence or posterior agreement.\n")
    return 0 if terminal["status"] != "EXTENSION_REPAIR_REQUIRED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "worker", "reclassify"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input", type=Path, help="Saved master run for archive-only reclassification")
    args = parser.parse_args(argv)
    if args.mode == "reclassify":
        if args.input is None:
            parser.error("reclassify requires --input")
        os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
        from bayesfilter.testing.surrogate_hmc_m13_cumulative import reclassify_saved_run
        result = reclassify_saved_run(args.input, args.output)
        print(json.dumps(dict(status=result["status"], draws_per_chain=result["draws_per_chain"],
            rhat=result["diagnostics"]["rhat"]["maximum"], new_hmc_transitions=0)))
        return 0
    return execute(args.output) if args.mode == "execute" else worker(args.output)
