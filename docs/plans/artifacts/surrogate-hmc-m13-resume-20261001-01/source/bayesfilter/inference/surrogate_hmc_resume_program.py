"""Resume M13 attempt 2, sharing one bounded allowance across sequential phases.

The public tuner validates the saved sequence and owns tuning admission.
This module never adapts a kernel or evaluates a different scientific target.
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

from .surrogate_hmc_program import ROOT, file_hash, run_job, write_json
from .surrogate_hmc_worker_guard import process_identity
from .surrogate_hmc_recovery_program import REFERENCE, EXTENDED_COVERAGE, EXTENDED_LOOKS, BASELINE

PLAN = "docs/plans/surrogate-hmc-m13-resume-2026-10-01.md"
SCRIPT = ROOT / "scripts/run_surrogate_hmc_resume.py"
START = "2026-09-30T19:37:42+00:00"
DEADLINE = "2026-10-01T10:37:42+00:00"
ALLOWANCE = 54000.
POST_WARMUP_MAXIMUM = 20000
ARTIFACTS = ROOT / "docs/plans/artifacts"
PARENT = ARTIFACTS / "surrogate-hmc-m13-sequential-48h-03/m13-public-tuning-02/tuning/manifest.json"
DEVELOPMENT = ARTIFACTS / "surrogate-hmc-m13-resume-20261001-development-01"
CHECKPOINT = ARTIFACTS / "surrogate-hmc-all-models-20260919-review-tests/checkpoint.md"
SOURCES = (
    "bayesfilter/inference/hmc_tensorflow_resume.py",
    "bayesfilter/inference/hmc_tensorflow_verified.py",
    "bayesfilter/inference/hmc_tensorflow_tuning.py",
    "bayesfilter/inference/hmc_tuning_dispatch.py",
    "bayesfilter/inference/hmc_sequential_verification.py",
    "bayesfilter/inference/neural_force_hmc.py",
    "bayesfilter/inference/surrogate_hmc_program.py",
    "bayesfilter/inference/surrogate_hmc_worker_guard.py",
    "bayesfilter/inference/surrogate_hmc_resume_program.py",
    "bayesfilter/inference/surrogate_hmc_selection_compatibility.py",
    "bayesfilter/inference/surrogate_hmc_recovery_program.py",
    "bayesfilter/testing/surrogate_hmc_m13_public_tuning.py",
    "bayesfilter/testing/surrogate_hmc_m13_retained.py",
    "bayesfilter/testing/surrogate_hmc_m13_staged_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_reference.py",
    "scripts/run_surrogate_hmc_resume.py",
)

def read(path):
    return json.loads(Path(path).read_text())

def retained_capacity(verification_count):
    if type(verification_count) is not int or not 0 <= verification_count <= POST_WARMUP_MAXIMUM:
        raise ValueError("invalid cumulative verification count")
    return min(max(EXTENDED_LOOKS), POST_WARMUP_MAXIMUM - verification_count)

def source_hashes():
    names = set(SOURCES)
    names.update("bayesfilter/inference/"+name for name in read(PARENT).get("source_hashes", {}))
    selection = ARTIFACTS / "surrogate-hmc-closure-20260922-m13-materiality-repair-01/m13-materiality-repair/selection.json"
    if selection.exists():
        names.update(read(selection)["source_hashes"])
    return {name: file_hash(ROOT/name) for name in sorted(names)}

def make_config():
    from .hmc_tensorflow_verified import VerifiedTensorFlowHMCConfig
    old = VerifiedTensorFlowHMCConfig.from_payload(read(PARENT)["config"])
    return replace(old, resume_manifest=str(PARENT), verification_max_results=POST_WARMUP_MAXIMUM)

def components():
    os.environ.update(TF_FORCE_GPU_ALLOW_GROWTH="true", TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components
    adapter, binding, target, fixture, selection, selection_path = selected_components(step_multipliers=(.125, 1.))
    provenance = dict(memory_policy=memory, device="gpu", tensorflow=tf.__version__,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(), jit_compile=True,
        gpu_trust_basis="escalated_local_gpu_execution",
        gpu_devices=[d.name for d in tf.config.list_physical_devices("GPU")],
        binding_hash=binding.binding_hash, data_seed=1704, observation_checksum=fixture.observation_checksum,
        selection_path=str(selection_path), selection_sha256=file_hash(selection_path),
        sequence_base_seed=list(make_config().fresh_seed), retained_seed=[20260926,2910],
        target_scope=make_config().nomination.target_scope,
        force_dtype="float32", endpoint_dtype="float64", exact_likelihood_claim=False,
        untouched_confirmation_seed_3701_used=False, repeats_tuning=False, repeats_warmup=False,
        classical_comparator_qualified=read(BASELINE)["qualified"])
    return adapter, binding, fixture, provenance

def prerequisites(fixture):
    from bayesfilter.testing.surrogate_hmc_m13_posterior_reference import load_reference
    from bayesfilter.testing.surrogate_hmc_m13_staged_assessment import assessment_scope
    reference, _, _ = load_reference(REFERENCE)
    calibration, baseline = read(EXTENDED_COVERAGE), read(BASELINE)
    if (calibration.get("qualified") is not True or baseline.get("health_passed") is not True
            or baseline.get("reference_sha256") != reference["reference_sha256"]
            or calibration.get("reference_sha256") != reference["reference_sha256"]
            or calibration.get("method") != assessment_scope(EXTENDED_LOOKS)
            or calibration.get("looks") != list(EXTENDED_LOOKS)
            or fixture.observation_checksum != reference["observation_checksum"]):
        raise ValueError("resume reference, calibration, comparator or data mismatch")
    return reference, calibration

def first_retained_chunk_fits(seconds):
    deadline = float(os.environ["BAYESFILTER_WORKER_WALL_DEADLINE"])
    cap = float(os.environ["BAYESFILTER_WORKER_CPU_ALLOWANCE"])
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    used = own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime
    return deadline-time.monotonic() > max(60., 1.5*seconds) and cap-used > max(60., 2.*seconds)

def worker(output, *, preflight=False):
    output = Path(output)
    started = time.monotonic()
    adapter, binding, fixture, provenance = components()
    reference, calibration = prerequisites(fixture)
    from .hmc_tensorflow_resume import prepare_resume
    from . import tune_hmc_kernel, load_verified_tensorflow_hmc_result
    cfg = make_config()
    report = dict(status="RESUME_VALIDATING", qualified=False, plan=PLAN, parent_manifest=str(PARENT),
        parent_sha256=file_hash(PARENT), deadline=DEADLINE, scientific_attempt=2,
        post_warmup_maximum=POST_WARMUP_MAXIMUM, ranking_supported=False, all_models_complete=False,
        **provenance)
    write_json(output/"report.json", report)
    if preflight:
        _, payload, _, migration = prepare_resume(adapter=adapter, config=cfg, runner_binding=binding)
        report.update(status="RESUME_PREFLIGHT_PASSED", passed=True,
            counts=payload["sequence"]["counts"], archived_chunks=len(payload["sequence"]["chunks"]),
            source_migration=migration, numerical_transitions_generated=0,
            wall_seconds=time.monotonic()-started)
        write_json(output/"report.json", report)
        return 0
    write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
        phase="verification", action="replay_archive_then_continue_frozen_kernel",
        deadline=DEADLINE, post_warmup_maximum=POST_WARMUP_MAXIMUM))
    tuning = tune_hmc_kernel(adapter=adapter, initial_position=(0., 0.), config=cfg,
        parameter_scales=None,
        output_dir=output/"tuning", runner_binding=binding)
    saved = read(tuning.artifact_manifest_path)
    count = saved["sequence"]["counts"]["verification"]
    report.update(status="M13_RESUMED_" + saved["sequence"]["status"],
        verification_passed=tuning.passed, counts=saved["sequence"]["counts"],
        tuning_manifest=tuning.artifact_manifest_path,
        tuning_manifest_sha256=file_hash(tuning.artifact_manifest_path),
        checks=saved["checks"], final_verification_rhat=next((e["rhat"] for e in reversed(saved["sequence"]["events"]) if e.get("rhat")), None),
        post_warmup_draws_per_chain=count, retained_draws_per_chain=0)
    write_json(output/"report.json", report)
    remaining = retained_capacity(count)
    if tuning.passed and remaining >= 250:
        # Seed/history and the artifact are rechecked by the public retained factory.
        last_cost = saved["sequence"]["resume_timing"]["last_chunk_wall_seconds"]
        if first_retained_chunk_fits(last_cost):
            write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
                phase="retained", action="continue_same_kernel_with_cumulative_posterior_checks",
                maximum_per_chain=remaining, post_warmup_maximum=POST_WARMUP_MAXIMUM,
                deadline=DEADLINE, independent_replication=False))
            tuning = load_verified_tensorflow_hmc_result(tuning.artifact_manifest_path,
                adapter=adapter, runner_binding=binding)
            from bayesfilter.testing.surrogate_hmc_m13_retained import consume_chunks
            retained = consume_chunks(tuning, binding, output/"posterior", reference, calibration,
                                      looks=EXTENDED_LOOKS, max_results=remaining)
            report.update(status=retained["status"], qualified=retained["qualified"],
                posterior=retained, retained_draws_per_chain=retained["draws_per_chain"],
                post_warmup_draws_per_chain=count+retained["draws_per_chain"])
        else:
            report["status"] = "M13_RETAINED_BUDGET_EXHAUSTED"
    elif tuning.passed:
        report["status"] = "M13_POST_WARMUP_SAMPLE_CAP_INCONCLUSIVE"
    if report["post_warmup_draws_per_chain"] > POST_WARMUP_MAXIMUM:
        raise ValueError("post-warmup cap violated")
    report.update(wall_seconds=time.monotonic()-started, reference_sha256=file_hash(REFERENCE),
        coverage_sha256=file_hash(EXTENDED_COVERAGE), baseline_sha256=file_hash(BASELINE))
    write_json(output/"report.json", report)
    write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
        phase="terminal_review", action="preserve_and_report_bounded_continuation",
        status=report["status"], deadline=DEADLINE, automatic_extension=False))
    return 0

def update_checkpoint(output, status, progress):
    CHECKPOINT.write_text(
        "# Active surrogate-HMC checkpoint — October 1 continuation\n\n"
        "Question: testing frozen surrogate HMC on M13/data1704/T20. LG1 accepted; all-model campaign incomplete.\n"
        f"Status: {status}. Evidence: {output}.\n"
        f"Progress: {json.dumps(progress, sort_keys=True)}\n"
        f"Plan: {PLAN}. Deadline: {DEADLINE} (18:37:42 HKT).\n"
        "User authorized 15 hours from 2026-09-30T19:37:42Z and at most 20000 total post-warmup draws per chain.\n"
        "Includes saved 2250 verification draws; excludes 9000 discarded warmup. Same scientific attempt 2.\n"
        "No retuning or repeated warmup. Verification and retained share the remaining allowance without a fixed reserve.\n"
        "Frozen kernel/target, FP32/TF32 force, FP64 approximate endpoint, GPU/XLA. Data3701 untouched.\n"
        f"Parent: {PARENT}. Historical accounting remains preserved separately.\n"
        "NEXT: follow next-phase-plan.json in this run. Continuation is part of the existing master.\n"
        "Do not restart an active worker or extend the deadline/cap. Preserve unrelated dirty edits.\n")

def execute(output):
    output = Path(output).resolve()
    # Same wall deadline and historical receipts apply to any infrastructure retry.
    elapsed = max(0., (datetime.now(timezone.utc)-datetime.fromisoformat(START)).total_seconds())
    remaining = max(0., (datetime.fromisoformat(DEADLINE)-datetime.now(timezone.utc)).total_seconds())
    charges = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    prior = []
    for path in sorted(ARTIFACTS.glob("surrogate-hmc-m13-resume-20261001-*/*/guard-receipt.json")):
        receipt = read(path)
        if receipt.get("accounting_complete") is not True:
            raise ValueError("unresolved continuation accounting")
        for key in charges:
            charges[key] += receipt[key]
        prior.append(str(path))
    for path in ARTIFACTS.glob("surrogate-hmc-m13-resume-20261001-*/run-manifest.json"):
        previous = read(path)
        owner = previous.get("owner")
        if owner and process_identity(owner["pid"]) == owner and previous.get("status") == "RUNNING":
            raise ValueError("continuation supervisor already active")
    # Elapsed preparation wall time is a conservative charge floor, covering
    # short unreceipted file operations; explicit CPU receipts can exceed it.
    charges = {k:max(v, elapsed) for k,v in charges.items()}
    ledger = dict(limits=dict(cpu_worker_seconds=ALLOWANCE, gpu_seconds=ALLOWANCE),
        charged=charges, jobs=[], prior_receipts=prior, conservative_preparation_charge=True)
    timeout = min(remaining-120., ALLOWANCE-charges["gpu_seconds"]-120.)
    cpu_allowance = ALLOWANCE-charges["cpu_worker_seconds"]-120.
    if min(timeout, cpu_allowance) <= 60.:
        raise ValueError("no remaining continuation budget")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    manifest = dict(status="RUNNING", plan=PLAN, command=[sys.executable, *sys.argv],
        cwd=str(ROOT), git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        owner=process_identity(os.getpid()), start=START, deadline=DEADLINE,
        launched_at=datetime.now(timezone.utc).isoformat(), parent=str(PARENT),
        parent_sha256=file_hash(PARENT), scientific_attempt=2, source_hashes=sources,
        post_warmup_maximum=POST_WARMUP_MAXIMUM, seeds=read(PARENT)["config"]["fresh_seed"],
        environment=sys.executable, device="gpu", ledger=ledger,
        data_seed=1704, result_file=str(output/"report.json"))
    write_json(output/"run-manifest.json", manifest)
    for name in sources:
        dest = output/"source"/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
    last = {}
    def progress(directory):
        selected = {}
        for name in ("next-phase-plan.json", "tuning/fresh-progress.json", "posterior/report.json"):
            path = directory/name
            if not path.exists():
                continue
            value = read(path)
            if name.endswith("fresh-progress.json"):
                selected["sequence"] = dict(status=value["status"], counts=value["counts"],
                    latest_rhat=next((e["rhat"] for e in reversed(value["events"]) if e.get("rhat")), None))
            elif name.startswith("posterior"):
                selected["posterior"] = {k:value.get(k) for k in ("status","draws_per_chain","qualified")}
                selected["next_phase_plan"] = value.get("next_phase_plan", {})
            else:
                selected["next_phase_plan"] = value
        if selected and selected != last:
            last.clear()
            last.update(selected)
            write_json(output/"progress.json", selected)
            write_json(output/"next-phase-plan.json", selected.get("next_phase_plan", {}))
            update_checkpoint(output, "RUNNING", selected)
    update_checkpoint(output, "RUNNING", dict(action="validate_parent_and_resume"))
    record = run_job([sys.executable, str(SCRIPT), "worker", "--output", str(output/"worker")],
        output/"worker", ledger, device="gpu", timeout=timeout, cpu_allowance=cpu_allowance,
        max_timeout_seconds=ALLOWANCE, progress_callback=progress)
    progress(output/"worker")
    result = read(output/"worker/report.json") if (output/"worker/report.json").exists() else {}
    unchanged = source_hashes() == sources and file_hash(PARENT) == manifest["parent_sha256"]
    status = result.get("status", "WORKER_INCOMPLETE")
    if record.get("exit_code") != 0 or record.get("status") != "COMPLETED":
        status = "WORKER_RESOURCE_OR_INFRASTRUCTURE_STOP"
    if not unchanged:
        status = "SOURCE_DRIFT_VETO"
    qualified = result.get("qualified") is True and unchanged and status == "M13_RETAINED_PILOT_QUALIFIED"
    terminal = dict(status=status, qualified=qualified, all_models_complete=False,
        source_unchanged=unchanged, ledger=ledger, worker=result, execution=record,
        final_progress=last,
        plan=PLAN, deadline=DEADLINE, ranking_supported=False)
    write_json(output/"report.json", terminal)
    manifest.update(status="TERMINAL", terminal_status=status, ledger=ledger,
                    completed_at=datetime.now(timezone.utc).isoformat())
    write_json(output/"run-manifest.json", manifest)
    write_json(output/"next-phase-plan.json", dict(part_of_existing_master=True,
        phase="terminal", action="report_result", status=status, automatic_extension=False))
    (output/"result.md").write_text(
        "# M13 saved-state continuation result\n\n"
        f"Status: {status}. Posterior pilot qualified: {qualified}. All-model campaign incomplete.\n\n"
        "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |\n"
        "|---|---|---|---|---|---|\n"
        f"| {status} | posterior qualification {qualified} | source unchanged {unchanged}; see worker checks | finite-chain/reference uncertainty | report saved result; no automatic extension | exact-likelihood, all-model or efficiency promotion |\n\n"
        "| Inference status | Finding |\n|---|---|\n"
        "| Hard veto screen | See worker checks and execution receipt |\n"
        "| Statistically supported ranking | None |\n"
        "| Descriptive-only differences | Rhat, acceptance, runtime and raw posterior differences |\n"
        f"| Default-readiness | Not established; exploratory posterior pilot {qualified} |\n"
        "| Next evidence needed | Determined by saved convergence and calibrated posterior assessment |\n\n"
        "Alternative explanation: remaining mixing or finite endpoint approximation can explain an inconclusive or discrepant result. "
        "Only a qualified calibrated posterior comparison supports this exploratory same-data pilot.\n")
    update_checkpoint(output, status, last)
    print(json.dumps(dict(status=status, qualified=qualified, output=str(output))), flush=True)
    return 0

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("execute", "worker", "preflight"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "execute":
        return execute(args.output)
    return worker(args.output, preflight=args.mode == "preflight")
