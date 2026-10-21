"""Executable recovery phases for the active all-model surrogate-HMC master.

No framework imports occur in the supervisor. Each worker is isolated, bounded
and receipted. This continuation tests the existing M13 surrogate HMC; other
model gaps remain in the all-model register.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from .surrogate_hmc_program import ROOT, file_hash, run_job, write_json
from .surrogate_hmc_worker_guard import process_identity

PLAN = "docs/plans/surrogate-hmc-sequential-recovery-master.md"
SCRIPT = ROOT/"scripts/run_surrogate_hmc_recovery.py"
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
ARTIFACTS = ROOT/"docs/plans/artifacts"
REFERENCE = ARTIFACTS/"surrogate-hmc-m13-assessment-reference-20260926-02/worker/reference/reference.json"
COVERAGE = REFERENCE.parents[1]/"coverage.json"
EXTENDED_COVERAGE = ARTIFACTS/"surrogate-hmc-m13-sequential-48h-development-01/coverage-extended-04/coverage.json"
EXTENDED_LOOKS = (1000, 2000, 4000, 6000, 8000, 10000)
BASELINE = ARTIFACTS/"surrogate-hmc-m13-assessment-rwm-20260926-06/worker/rwm/report.json"
LEGACY = ARTIFACTS/"surrogate-hmc-m13-initialization-repair-20260927-01"
ALLOCATION_ID = "surrogate_hmc_sequential_recovery_20260928"
LIMITS = dict(cpu_worker_seconds=60000., gpu_seconds=60000.)
CONTINUATION_PLAN = "docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md"
CONTINUATION_ID = "surrogate_hmc_m13_48h_20260929"
CONTINUATION_DEADLINE = "2026-09-30T16:27:39+00:00"


def allocation_profile(name="legacy"):
    if name == "legacy":
        return dict(name=name, id=ALLOCATION_ID, plan=PLAN, limits=LIMITS,
                    deadline=None, retained_reserve=12000., max_attempts=1)
    if name != "48h":
        raise ValueError("unknown allocation")
    return dict(name=name, id=CONTINUATION_ID, plan=CONTINUATION_PLAN,
                limits=dict(cpu_worker_seconds=172800., gpu_seconds=172800.),
                deadline=CONTINUATION_DEADLINE, retained_reserve=36000.,
                tuning_limit=115200., first_attempt_limit=72000., max_attempts=2)


def seconds_to_deadline(profile):
    if profile["deadline"] is None:
        return math.inf
    return max(0., (datetime.fromisoformat(profile["deadline"])
                    - datetime.now(timezone.utc)).total_seconds())


def continuation_charges(output):
    """Carry this allocation's durable receipts; never reset costs on a retry."""
    profile = allocation_profile("48h")
    charges = dict(cpu_worker_seconds=0., gpu_seconds=0.)
    tuning = dict(charges)
    receipts = []
    attempts = 0
    roots = {ARTIFACTS / "surrogate-hmc-m13-sequential-48h-development-01"}
    for path in sorted(ARTIFACTS.glob("surrogate-hmc-*sequential*/run-manifest.json")):
        if path.parent == output:
            continue
        manifest = read(path)
        if manifest.get("allocation_id") != profile["id"]:
            continue
        owner = manifest.get("owner")
        if owner and process_identity(owner["pid"]) == owner and manifest.get("status") == "RUNNING":
            raise ValueError("another continuation supervisor is still running")
        roots.add(path.parent)
    for root in sorted(roots):
        if not root.exists():
            continue
        for job in sorted(root.glob("*/run-manifest.json")):
            receipt_path = job.parent / "guard-receipt.json"
            if not receipt_path.exists():
                raise ValueError("unresolved continuation worker accounting: " + str(job))
            receipt = read(receipt_path)
            if receipt.get("accounting_complete") is not True:
                raise ValueError("incomplete continuation receipt: " + str(receipt_path))
            is_tuning = job.parent.name.startswith("m13-public-tuning")
            for key in charges:
                value = float(receipt[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid continuation charge")
                charges[key] += value
                if is_tuning:
                    tuning[key] += value
            if is_tuning and scientific_attempt_started(job.parent):
                attempts += 1
            receipts.append(dict(path=str(receipt_path), sha256=file_hash(receipt_path),
                                 kind="worker receipt"))
    if attempts >= profile["max_attempts"]:
        raise ValueError("both scientific attempts in the additional48-hour allocation have been used")
    receipts.append(dict(kind="allocation history", scientific_attempts_started=attempts,
                         tuning_charged=tuning))
    return charges, receipts

TESTS = (
    "tests/test_hmc_sequential_verification.py", "tests/test_hmc_tensorflow_verified.py",
    "tests/test_surrogate_hmc_worker_guard.py", "tests/test_surrogate_hmc_program.py",
    "tests/test_surrogate_hmc_recovery_program.py", "tests/test_surrogate_hmc_recovery_48h.py",
    "tests/test_surrogate_hmc_selection_compatibility.py",
    "tests/test_surrogate_hmc_m13_public_tuning.py", "tests/test_surrogate_hmc_m13_mixing_config.py",
    "tests/test_surrogate_hmc_m13_step_mixture.py",
    "tests/test_hmc_nomination_energy_guard.py", "tests/test_surrogate_hmc_m13_energy_repair.py",
    "tests/test_surrogate_hmc_m13_initialization_repair.py",
    "tests/test_surrogate_hmc_m13_retained.py",
    "tests/test_surrogate_hmc_m13_posterior_assessment.py",
)
NEW_SOURCES = (str(SCRIPT.relative_to(ROOT)), PLAN, CONTINUATION_PLAN,
    "scripts/diagnose_surrogate_hmc_m13_mixing.py",
    "scripts/plot_surrogate_hmc_m13_warmup_diagnostic.py",
    "bayesfilter/testing/surrogate_hmc_m13_mixing_repair.py",
    "bayesfilter/testing/surrogate_hmc_m13_mixing_config.py",
    "bayesfilter/testing/surrogate_hmc_m13_energy_repair.py",
    "bayesfilter/testing/surrogate_hmc_m13_step_mixture.py",
    "bayesfilter/testing/surrogate_hmc_m13_mixture_probe.py",
    "bayesfilter/testing/surrogate_hmc_m13_staged_assessment.py",
    "bayesfilter/testing/surrogate_hmc_m13_posterior_coverage.py",
    "scripts/qualify_hmc_tensorflow_verified.py",
    "bayesfilter/inference/surrogate_hmc_recovery_program.py",
    "bayesfilter/inference/surrogate_hmc_selection_compatibility.py",
    "bayesfilter/inference/surrogate_hmc_worker_guard.py",
    "bayesfilter/inference/hmc_sequential_verification.py",
    "bayesfilter/inference/hmc_tensorflow_verified.py",
    "bayesfilter/testing/surrogate_hmc_m13_retained.py",
    "bayesfilter/testing/surrogate_hmc_m13_public_tuning.py",
    "bayesfilter/testing/surrogate_hmc_m13_initialization_repair.py",
    "docs/reference/hmc-tuning-interface.md", *TESTS)

def read(path):
    return json.loads(Path(path).read_text())

def now():
    return datetime.now(timezone.utc).isoformat()

def remaining(ledger):
    return {key: max(0., ledger["limits"][key]-ledger["charged"][key]) for key in ledger["limits"]}

def prerequisite_check(allocation="legacy"):
    coverage_path = EXTENDED_COVERAGE if allocation == "48h" else COVERAGE
    looks = list(EXTENDED_LOOKS) if allocation == "48h" else [1000, 2000]
    reference, coverage, baseline = map(read, (REFERENCE, coverage_path, BASELINE))
    if reference["data_seed"] != 1704 or coverage.get("qualified") is not True:
        raise ValueError("reference/coverage prerequisite failed")
    if coverage.get("looks") != looks or baseline.get("health_passed") is not True:
        raise ValueError("coverage or classical health prerequisite failed")
    if not all(x.get("reference_sha256")==reference["reference_sha256"] for x in (coverage,baseline)):
        raise ValueError("prerequisite reference identity mismatch")
    extra_paths = []
    if allocation == "48h":
        receipt_path = coverage_path.with_name("guard-receipt.json")
        receipt = read(receipt_path)
        calibration_plan = coverage_path.parents[1]/"coverage-extended-plan.json"
        declared = read(calibration_plan)
        if (receipt.get("status") != "COMPLETED" or not receipt.get("accounting_complete")
                or declared.get("result") != str(coverage_path)
                or coverage.get("method") != "m13_ten_functions_guarded_lugsail_six_looks_v2"):
            raise ValueError("extended assessment calibration is not qualified")
        for name, digest in declared["sources"].items():
            if file_hash(ROOT/name) != digest:
                raise ValueError("extended assessment source changed")
        extra_paths = [receipt_path, calibration_plan]
    return dict(paths={str(p):file_hash(p) for p in (REFERENCE,coverage_path,BASELINE,*extra_paths)},
        retained_looks=looks,
        data_seed=1704, confirmation_data_used=False,
        rwm_healthy=True, rwm_formal_equivalence=baseline.get("qualified",False),
        legacy_missing_receipt=dict(directory=str(LEGACY),
            accounting_status="UNKNOWN" if not (LEGACY/"accounting.json").exists() else "RECEIPT_PRESENT",
            included_in_new_allocation=False))

def scientific_attempt_started(directory):
    """Preparation failures consume compute, not a scientific nomination."""
    directory = Path(directory)
    if not (directory / "run-manifest.json").exists():
        return False
    receipt = directory / "guard-receipt.json"
    if not receipt.exists() or read(receipt).get("accounting_complete") is not True:
        raise ValueError("unresolved recovery worker accounting: " + str(directory))
    if (directory / "report.json").exists() or (directory / "tuning").exists():
        return True  # Conservative for incomplete or older stage records.
    boundary = directory / "sampling-state.json"
    if boundary.exists():
        state = read(boundary)
        return not (state.get("status") == "PREPARING" and state.get("inference_launched") is False)
    # Recovery01 predates explicit boundaries. Its reviewed traceback and source
    # snapshot establish that it stopped at selection loading, before sampling.
    review = directory / "pre-sampling-review.json"
    if not review.exists():
        return True
    proof = read(review)
    required = {"command.log", "guard-receipt.json",
        "../source_snapshot/bayesfilter/testing/surrogate_hmc_m13_public_tuning.py"}
    evidence = proof.get("evidence", {})
    if (proof.get("classification") != "SELECTION_LOADING_BEFORE_PUBLIC_TUNER"
            or proof.get("inference_launched") is not False or set(evidence) != required
            or any(file_hash(directory/name) != digest for name,digest in evidence.items())):
        raise ValueError("invalid pre-sampling failure review")
    log = (directory / "command.log").read_text()
    if ("ValueError: wrong or stale materiality selection" not in log
            or "selected_components" not in log
            or "M13 public nomination and fresh verification started" in log):
        raise ValueError("pre-sampling failure evidence does not match")
    return False


def prior_recovery_charges(output, *, research, allocation="legacy"):
    if allocation == "48h":
        return continuation_charges(output)
    charges=dict(cpu_worker_seconds=0.,gpu_seconds=0.)
    receipts=[]
    if not research:
        return charges,receipts
    development=ARTIFACTS/"surrogate-hmc-sequential-development-01"
    for path in sorted(development.glob("*-tests.json")):
        payload=read(path)
        for key in charges:
            charges[key]+=float(payload[key])
        receipts.append(dict(path=str(path),sha256=file_hash(path),kind="development tests"))
    for path in sorted(ARTIFACTS.glob("surrogate-hmc-*sequential*/run-manifest.json")):
        if path.parent==output:
            continue
        manifest=read(path)
        if manifest.get("allocation_id")!=ALLOCATION_ID:
            continue
        owner=manifest.get("owner")
        if owner and process_identity(owner["pid"])==owner and manifest.get("status")=="RUNNING":
            raise ValueError("another recovery supervisor is still running")
        # A fresh scientific attempt cannot be hidden by changing output root.
        if manifest.get("research") and scientific_attempt_started(path.parent/"m13-public-tuning"):
            raise ValueError("this allocation has already launched its one scientific attempt; inspect preserved progress")
        accounting=path.parent/"accounting.json"
        jobs=sorted(path.parent.glob("*/run-manifest.json"))
        records=[]
        for job in jobs:
            receipt=job.parent/"guard-receipt.json"
            if not receipt.exists() or not read(receipt).get("accounting_complete"):
                raise ValueError("unresolved recovery worker accounting: "+str(job))
            records.append(read(receipt))
            receipts.append(dict(path=str(receipt),sha256=file_hash(receipt),kind="worker receipt"))
        if jobs:
            # The supervisor's summary can predate a guard's crash-final receipt.
            payload={key:sum(record[key] for record in records) for key in charges}
        elif accounting.exists():
            payload=read(accounting)["charged_this_run"]
        else:
            payload=dict(cpu_worker_seconds=0.,gpu_seconds=0.)  # No worker launched.
        for key in charges:
            value=float(payload[key])
            if not math.isfinite(value) or value<0:
                raise ValueError("invalid prior recovery charge")
            charges[key]+=value
        receipts.append(dict(path=str(path),sha256=file_hash(path),charged_this_run=payload))
    return charges,receipts

def snapshot(output):
    from .surrogate_hmc_closure_program import snapshot_sources
    hashes = snapshot_sources(output)
    for name in NEW_SOURCES:
        path=ROOT/name
        hashes[name]=file_hash(path)
        target=output/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    return hashes

def check_snapshot(hashes):
    changed=[name for name,digest in hashes.items() if not (ROOT/name).is_file() or file_hash(ROOT/name)!=digest]
    if changed:
        raise ValueError("source changed during execution: "+str(changed))

def gpu_qualification_passed(result):
    """Consume the real qualification report, including terminal arm checks."""
    arms = result.get("arms", {})
    if (result.get("jit_compile") is not True
            or set(arms) != {"exact", "imperfect_rotational"}
            or result.get("memory_policy", {}).get("all_physical_devices_memory_growth") is not True):
        return False
    for arm in arms.values():
        checks = arm.get("checks", {})
        parity = arm.get("parity_max")
        if (arm.get("passed") is not True or not checks
                or any(value is not True for value in checks.values())
                or arm.get("artifact_and_continuation_checks") is not True
                or arm.get("endpoint_flag_archive_wiring") != "PASS"
                or not isinstance(parity, (int,float)) or not math.isfinite(parity)
                or not 0 <= parity < 1e-12):
            return False
    return True


def phase_decision(phase, execution, result, available, *, allocation="legacy", attempts=1, tuning_charged=None):
    """A scientific non-pass is distinct from a failed worker."""
    if execution!="COMPLETED":
        return dict(phase="repair_execution", automatic=False, terminal=True,
                    reason=execution, candidate_rejected=False)
    if phase in ("tests","gpu-smoke","m13-preflight"):
        if ((phase == "gpu-smoke" and not gpu_qualification_passed(result))
                or (phase == "m13-preflight" and result.get("status") != "PASS")):
            return dict(phase="repair_execution", automatic=False, terminal=True,
                        reason="required preflight did not pass", candidate_rejected=False)
        following = {"tests":"gpu-smoke", "gpu-smoke":"m13-preflight",
                     "m13-preflight":"m13-public-tuning"}[phase]
        return dict(phase=following, automatic=True, terminal=False,
                    reason="mechanics and prerequisite checks passed")
    if phase=="m13-public-tuning":
        if result.get("qualified") is True:
            if min(available.values()) < 60:
                return dict(phase="budget_review",automatic=False,terminal=True,
                            reason="no safe retained reservation remains")
            return dict(phase="m13-retained",automatic=True,terminal=False,
                        reason="fresh warm-up and cumulative verification passed")
        sequence_status = result.get("tuning", {}).get("sequential_status")
        repairable = sequence_status in {"BUDGET_EXHAUSTED", "WARMUP_MAXIMUM_INCONCLUSIVE",
                                         "VERIFICATION_MAXIMUM_INCONCLUSIVE"}
        if allocation == "48h" and attempts < 2 and repairable:
            try:
                _, wall, cpu = phase_budget(phase, available, allocation=allocation,
                    attempt=attempts+1, tuning_charged=tuning_charged)
                if min(wall, cpu) >= 36000.:
                    return dict(phase=phase, automatic=True, terminal=False,
                        reason="healthy mixing attempt inconclusive; execute the second predeclared tuning repair",
                        candidate_rejected=True, research_direction_rejected=False)
            except ValueError:
                pass  # The allocation, deadline or retained reserve prevents another attempt.
        return dict(phase="review_candidate",automatic=False,terminal=True,
            reason=result.get("status","missing scientific decision"),
            candidate_rejected="BUDGET" not in result.get("status",""), research_direction_rejected=False)
    if phase=="m13-retained":
        return dict(phase="all-model-register", automatic=False,terminal=True,
            reason=result.get("status","missing retained assessment"),
            pilot_qualified=result.get("posterior_qualified",False),
            all_model_complete=False, research_direction_rejected=False)
    raise ValueError("unknown phase")

def refresh(output, report, ledger, decision, *, progress=None):
    report["next_phase_plan"]=dict(decision, part_of_existing_master=True,
        refreshed_at=now(), remaining_budget=remaining(ledger),
        allocation=ledger.get("allocation_id", ALLOCATION_ID),
        deadline=ledger.get("deadline"),
        scientific_goal="test surrogate HMC on the frozen M13 target",
        warmup_policy=">=2000, latest1000 Rhat<=1.05, max10000 per chain",
        verification_policy="cumulative >=1000 Rhat<=1.01, max10000 per chain",
        retained_policy=("calibrated cumulative looks1000/2000/4000/6000/8000/10000, ESS and posterior equivalence"
            if ledger.get("allocation_id") == allocation_profile("48h")["id"] else
            "calibrated cumulative looks1000/2000, ESS and posterior equivalence"))
    if progress is not None:
        report["next_phase_plan"]["worker_progress"]=progress
    write_json(output/"next-phase-plan.json", report["next_phase_plan"])
    (output/"next-phase-plan.md").write_text(
        "# Next phase in the surrogate-HMC master\n\n"
        f"Phase: {decision['phase']}. {decision['reason']}.\n\n"
        f"Automatic continuation: {decision['automatic']}. Remaining allocation: {remaining(ledger)}.\n"
        "Warm-up readiness precedes cumulative fresh verification; retained assessment follows only a qualified handoff.\n")
    write_json(output/"status.json", dict(status=report["status"],active_phase=report.get("active_phase"),
        last_phase=report.get("last_phase"),
        next_phase_plan=report["next_phase_plan"], master_complete=False,
        pilot_qualified=report.get("pilot_qualified",False)))
    write_json(output/"report.json",report)
    write_json(output/"accounting.json",dict(
        charged_this_run={k:ledger["charged"][k]-ledger.get("prior_charges",{}).get(k,0.) for k in ledger["charged"]},
        cumulative_recovery_charged=ledger["charged"],jobs=ledger["jobs"],
        limits=ledger["limits"], allocation_id=ledger.get("allocation_id", ALLOCATION_ID),
        deadline=ledger.get("deadline"), historical_cpu_unknown=True))

def phase_command(phase, output, *, allocation="legacy", attempt=1):
    if phase=="tests":
        return [PYTHON,"-m","pytest","-q",*TESTS]
    if phase=="gpu-smoke":
        return [PYTHON,str(ROOT/"scripts/qualify_hmc_tensorflow_verified.py"),"--output",str(output/"qualification")]
    return [PYTHON,str(SCRIPT),"worker","--phase",phase,"--output",str(output),
            "--allocation",allocation,"--attempt",str(max(1,attempt))]

def phase_budget(phase, available, *, allocation="legacy", attempt=1, tuning_charged=None):
    if allocation == "48h":
        profile = allocation_profile(allocation)
        deadline = seconds_to_deadline(profile) - 20.
        if phase in ("tests", "gpu-smoke", "m13-preflight"):
            device, wall, cpu = phase_budget(phase, available)
            wall = min(wall, deadline)
        else:
            reserve = profile["retained_reserve"] if phase == "m13-public-tuning" else 0.
            cpu = available["cpu_worker_seconds"] - reserve - 20.
            wall = min(available["gpu_seconds"] - reserve - 20., cpu, deadline-reserve)
            if phase == "m13-public-tuning":
                spent = tuning_charged or dict(cpu_worker_seconds=0., gpu_seconds=0.)
                cap = profile["first_attempt_limit"] if attempt == 1 else profile["tuning_limit"]
                cpu = min(cpu, cap, profile["tuning_limit"]-spent["cpu_worker_seconds"])
                wall = min(wall, cpu, cap, profile["tuning_limit"]-spent["gpu_seconds"])
            device = "gpu"
        # The48-hour allocation spans several workers; the shared guard permits
        # at most24 hours for one worker. Preserve that existing guard boundary.
        wall = min(wall, 24 * 3600.)
        if min(wall, cpu) <= 60.:
            raise ValueError("additional allocation or deadline cannot safely reserve this phase")
        return device, wall, cpu
    if phase=="tests":
        return "cpu", min(240.,available["cpu_worker_seconds"]/2), min(600.,available["cpu_worker_seconds"])
    if phase=="gpu-smoke":
        return "gpu", min(240.,available["gpu_seconds"]), min(600.,available["cpu_worker_seconds"])
    if phase=="m13-preflight":
        return "gpu", min(120.,available["gpu_seconds"]), min(240.,available["cpu_worker_seconds"])
    reserve=12000. if phase=="m13-public-tuning" else 0.
    # Leave room for supervisor work, signal cleanup and final receipts.
    cpu=available["cpu_worker_seconds"]-reserve-20.
    wall=min(available["gpu_seconds"]-reserve-20.,cpu)
    if min(cpu,wall)<=60:
        raise ValueError("remaining allocation cannot safely reserve this phase")
    return "gpu", wall, cpu

def worker(phase, output, *, allocation="legacy", attempt=1):
    os.environ.update(TF_FORCE_GPU_ALLOW_GROWTH="true",TF_NUM_INTRAOP_THREADS="2",
                      TF_NUM_INTEROP_THREADS="1",OMP_NUM_THREADS="2",BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if phase=="m13-preflight":
        from bayesfilter.testing.surrogate_hmc_m13_public_tuning import selected_components_preflight
        result=selected_components_preflight(output/"report.json",
            mixing_repair_attempt=max(1, attempt) if allocation == "48h" else None)
    elif phase=="m13-public-tuning":
        from bayesfilter.testing.surrogate_hmc_m13_public_tuning import public_tuning_stage
        options = dict(initialization_repair=True, sequential_recovery=True)
        if allocation == "48h":
            options["mixing_repair_attempt"] = attempt
        result=public_tuning_stage(output/"report.json", **options)
    elif phase=="m13-retained":
        from bayesfilter.testing.surrogate_hmc_m13_retained import retained_stage
        result=retained_stage(output.parent,output/"posterior",REFERENCE,
            EXTENDED_COVERAGE if allocation == "48h" else COVERAGE, BASELINE,
            looks=EXTENDED_LOOKS if allocation == "48h" else (1000, 2000))
    else:
        raise ValueError("unsupported research phase")
    result.update(plan=allocation_profile(allocation)["plan"],memory_policy=memory,device="gpu",tensorflow=tf.__version__,
        gpu_trust_basis="owner_designated_managed_session_visible_gpu_trusted")
    write_json(output/"report.json", result)
    return 0  # execution completion; the scientific verdict is in result

def execute(output, *, research, limits=None, allocation="legacy"):
    profile = allocation_profile(allocation)
    output=output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    prior,prior_receipts=prior_recovery_charges(output,research=research,allocation=allocation)
    allocation_history = next((r for r in prior_receipts if r.get("kind") == "allocation history"), {})
    attempts = allocation_history.get("scientific_attempts_started", 0)
    tuning_charged = allocation_history.get("tuning_charged", dict(cpu_worker_seconds=0., gpu_seconds=0.))
    ledger=dict(limits=limits or (profile["limits"] if allocation == "48h" else dict(cpu_worker_seconds=1500.,gpu_seconds=300.)),
                charged=dict(prior),prior_charges=prior,jobs=[],allocation_id=profile["id"],deadline=profile["deadline"])
    if any(ledger["charged"][k]>=ledger["limits"][k] for k in prior):
        raise ValueError("prior recovery work exhausted the additional allocation")
    start=time.monotonic()
    before=resource.getrusage(resource.RUSAGE_SELF)
    self_start=before.ru_utime+before.ru_stime
    report=dict(status="RUNNING",plan=profile["plan"],stages={},attempts=[],master_complete=False,
                scientific_goal="testing surrogate HMC",pilot_qualified=False,source_unchanged=False)
    manifest=dict(schema="bayesfilter.surrogate_hmc_recovery.v1",started=now(),
        owner=process_identity(os.getpid()),command=sys.argv, cwd=str(ROOT),
        python=sys.executable, conda_env="tftwogpu", execution_device="GPU workers; supervisor CPU",
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        seeds=(dict(attempts=[
            dict(attempt=1, nomination=[20260929,3101], fresh=[20260929,3102]),
            dict(attempt=2, nomination=[20260929,3201], fresh=[20260929,3202])],
            retained=[20260926,2910]) if allocation == "48h" else
            dict(nomination=[20260928,1101],fresh=[20260928,1102],retained=[20260926,2910])),
        data_seed=1704,plan=profile["plan"],result_file=str(output/"result.md"),output=str(output),
        limits=ledger["limits"],research=research,allocation_id=profile["id"],deadline=profile["deadline"],
        prior_charges=prior,prior_receipts=prior_receipts,
        cpu_accounting="guard plus worker and reaped descendants; supervisor CPU reported separately",
        budget_basis="separately approved incremental recovery" if research else "routine mechanics checks",
        status="RUNNING")
    write_json(output/"run-manifest.json",manifest)
    history=[]
    decision=dict(phase="preflight",automatic=True,terminal=False,reason="checking saved prerequisites")
    def charge_supervisor():
        usage=resource.getrusage(resource.RUSAGE_SELF)
        ledger["charged"]["cpu_worker_seconds"]=prior["cpu_worker_seconds"]+sum(j.get("cpu_worker_seconds",0.) for j in ledger["jobs"])
        manifest["supervisor_cpu_seconds"]=usage.ru_utime+usage.ru_stime-self_start
    try:
        prerequisites=prerequisite_check(allocation)
        hashes=snapshot(output)
        manifest.update(prerequisites=prerequisites,source_hashes=hashes)
        write_json(output/"run-manifest.json",manifest)
        refresh(output,report,ledger,decision)
        phases = ["tests", "gpu-smoke", "m13-preflight", "m13-public-tuning", "m13-retained"]
        while phases:
            phase = phases.pop(0)
            if phase=="m13-public-tuning" and not research:
                report["status"]="REPAIR_CHECKS_PASSED_RESEARCH_PENDING_BUDGET"
                decision=dict(phase=phase,automatic=False,terminal=True,
                    reason=("mechanics-only invocation completed; authorized research uses start --allocation 48h"
                            if allocation == "48h" else
                            "additional compute approval is required because historical CPU accounting is unknown"))
                break
            check_snapshot(hashes)
            charge_supervisor()
            if phase == "m13-public-tuning":
                attempts += 1
            directory = output / (phase + (f"-{attempts:02d}" if phase == "m13-public-tuning" and attempts > 1 else ""))
            device,wall,cpu=phase_budget(phase,remaining(ledger),allocation=allocation,
                attempt=attempts, tuning_charged=tuning_charged)
            report["active_phase"]=phase
            decision=dict(phase=phase,automatic=True,terminal=False,reason="executing planned phase",
                          reserved_cpu_seconds=cpu,reserved_wall_seconds=wall)
            history.append(dict(phase=phase,directory=str(directory),attempt=attempts,started=now(),reservation=decision))
            write_json(output/"phase-history.json",history)
            refresh(output,report,ledger,decision)
            signatures={}
            def progress(directory):
                candidates=[directory/"tuning/fresh-progress.json",directory/"posterior/report.json"]
                for path in candidates:
                    if path.exists() and signatures.get(str(path))!=path.stat().st_mtime_ns:
                        signatures[str(path)]=path.stat().st_mtime_ns
                        data=read(path)
                        summary={k:data[k] for k in ("status","counts","warmup_ready","draws_per_chain","next_phase_plan") if k in data}
                        if data.get("events"):
                            summary["last_check"]=data["events"][-1]
                        heartbeat=directory/"heartbeat.json"
                        if heartbeat.exists():
                            beat=read(heartbeat)
                            summary["live_usage_lower_bound"]={k:beat[k] for k in ("cpu_worker_seconds_lower_bound","gpu_seconds","wall_seconds")}
                        refresh(output,report,ledger,decision,progress=summary)
            # A resumed preflight must inspect the next candidate, before its
            # attempt counter is incremented at the actual sampling boundary.
            command_attempt = attempts + int(phase == "m13-preflight")
            job=run_job(phase_command(phase,directory,allocation=allocation,attempt=command_attempt),directory,ledger,
                device=device,timeout=wall,cpu_allowance=cpu,max_timeout_seconds=min(24 * 3600., max(profile["limits"].values())),
                progress_callback=progress)
            charge_supervisor()
            check_snapshot(hashes)
            result_path=directory/"report.json"
            if phase=="gpu-smoke":
                result_path=directory/"qualification/report.json"
            result=read(result_path) if result_path.exists() else {}
            execution=job["status"]
            if phase not in ("tests",) and execution=="COMPLETED" and not result:
                execution="MISSING_REQUIRED_REPORT"
            report["source_unchanged"]=True
            report["stages"][phase]=dict(execution=execution,result=result,directory=str(directory))
            if phase == "m13-public-tuning":
                report["attempts"].append(dict(attempt=attempts, **report["stages"][phase]))
                for key in tuning_charged:
                    tuning_charged[key] += float(job.get(key, 0.))
            history[-1].update(finished=now(),execution=execution,result_status=result.get("status"))
            write_json(output/"phase-history.json",history)
            decision=phase_decision(phase,execution,result,remaining(ledger),allocation=allocation,
                attempts=attempts, tuning_charged=tuning_charged)
            if phase == "m13-public-tuning" and decision["automatic"] and decision["phase"] == phase:
                phases.insert(0, phase)
            if decision["terminal"]:
                report["status"]=("EXECUTION_REPAIR_REQUIRED" if decision["phase"]=="repair_execution" else
                    "M13_PILOT_QUALIFIED" if decision.get("pilot_qualified") else
                    "BOUNDED_PHASES_COMPLETE_CANDIDATE_NOT_QUALIFIED")
                report["pilot_qualified"]=decision.get("pilot_qualified",False)
            refresh(output,report,ledger,decision)
            if decision["terminal"]:
                break
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status="EXECUTION_REPAIR_REQUIRED",error=f"{type(exc).__name__}: {exc}")
        decision=dict(phase="repair_execution",automatic=False,terminal=True,reason=report["error"])
    finally:
        report["last_phase"]=report.get("active_phase")
        report["active_phase"]=None
        charge_supervisor()
        manifest.update(status=report["status"],wall_seconds=time.monotonic()-start,
                        charged=ledger["charged"],finished=now())
        refresh(output,report,ledger,decision)
        write_json(output/"run-manifest.json",manifest)
        write_result(output,report,ledger)
    print(json.dumps(dict(status=report["status"],output=str(output),charged=ledger["charged"])),flush=True)
    return 0 if report["status"] in ("M13_PILOT_QUALIFIED","REPAIR_CHECKS_PASSED_RESEARCH_PENDING_BUDGET") else 1

def write_result(output,report,ledger):
    passed=report.get("pilot_qualified",False)
    decision=report["next_phase_plan"]
    text=f"""# Surrogate-HMC recovery result

Goal: test surrogate HMC on the unchanged frozen M13/data1704 target.
Status: {report['status']}. All-model campaign complete: false.
Measured charges in this allocation: {ledger['charged']}.
Prior interrupted CPU usage remains unknown and is not silently charged as zero.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| {'Pilot qualified' if passed else 'No posterior promotion'} | {'Retained criteria passed' if passed else 'See stage verdicts'} | See preserved chunk and worker checks | Finite endpoint/reference; exploratory data | {decision['phase']}: {decision['reason']} | All-model success or efficiency superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Per-chunk diagnostics and watchdog receipts preserved |
| Statistically supported ranking | None |
| Descriptive differences | R-hat, ESS, acceptance and runtime are diagnostics |
| Default readiness | Not established |
| Next evidence needed | Qualified retained assessment and unresolved model scopes |

Engineering, sampler validity and scientific qualification are separate in report.json.
A failed early R-hat check extends the same kernel. A maximum or resource cap
ends this attempt without rejecting the research direction. The strongest
alternative explanation for nonqualification is inadequate mixing or budget;
the retained reference comparison is still required to assess the endpoint.
"""
    (output/"result.md").write_text(text)

def status(output):
    manifest=read(output/"run-manifest.json")
    result=read(output/"status.json") if (output/"status.json").exists() else {}
    live=process_identity(manifest["owner"]["pid"])==manifest["owner"]
    receipts=[read(p) for p in output.glob("*/guard-receipt.json")]
    return dict(output=str(output),supervisor_alive=live,manifest_status=manifest["status"],
        status=result,completed_worker_receipts=len(receipts),
        receipted_cpu=sum(p["cpu_worker_seconds"] for p in receipts),
        receipted_gpu=sum(p["gpu_seconds"] for p in receipts),
        recovery_required=not live and manifest["status"]=="RUNNING")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","execute","start","status","worker"))
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--phase",choices=("m13-preflight","m13-public-tuning","m13-retained"))
    parser.add_argument("--cpu-budget",type=float)
    parser.add_argument("--gpu-budget",type=float)
    parser.add_argument("--allocation",choices=("legacy","48h"),default="legacy")
    parser.add_argument("--attempt",type=int,choices=(1,2),default=1)
    args=parser.parse_args(argv)
    if args.action=="worker":
        return worker(args.phase,args.output.resolve(),allocation=args.allocation,attempt=args.attempt)
    if args.action=="status":
        print(json.dumps(status(args.output.resolve()),indent=2)); return 0
    if args.output.resolve().parent!=ARTIFACTS or "sequential" not in args.output.name or not args.output.name.startswith("surrogate-hmc-"):
        parser.error("use a fresh surrogate-hmc-*sequential* directory under docs/plans/artifacts")
    if args.action=="check":
        return execute(args.output,research=False,allocation=args.allocation)
    limits=dict(cpu_worker_seconds=args.cpu_budget,gpu_seconds=args.gpu_budget)
    profile = allocation_profile(args.allocation)
    if any(v is None or not math.isfinite(v) or not 0<v<=profile["limits"][k] for k,v in limits.items()):
        parser.error("execution requires explicit budgets within the selected authorized allocation")
    if seconds_to_deadline(profile) <= 60:
        parser.error("the selected allocation deadline has expired")
    if args.action=="start":
        output=args.output.resolve()
        if output.exists():
            raise FileExistsError("fresh versioned output required")
        output.parent.mkdir(parents=True,exist_ok=True)
        command=[PYTHON,str(SCRIPT),"execute","--output",str(output),
                 "--cpu-budget",str(args.cpu_budget),"--gpu-budget",str(args.gpu_budget),
                 "--allocation",args.allocation]
        with output.with_name(output.name+"-supervisor.log").open("x") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,output=str(output),command=command)))
        return 0
    return execute(args.output,research=True,limits=limits,allocation=args.allocation)
