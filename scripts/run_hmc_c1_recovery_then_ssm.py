"""Bounded recovery of C1's ten timed-out seeds, then the prepared SSM stages.

The parent uses the revised framework-free supervisor. Numerical C1 children
run from their unchanged original source. Original results are never edited.
"""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from datetime import datetime, timezone
import os
import signal
import subprocess
import time

from bayesfilter.testing.inference_validation.fit_supervision import supervise_fit
from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
from bayesfilter.testing.inference_validation.timeout_policy import TimeoutPolicy, wait_for_gpu_admission
from bayesfilter.testing.inference_validation import ssm_campaign
from bayesfilter.testing.inference_validation.designs import digest

PLAN = "docs/plans/bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md"
INDICES = (78, 79, 80, 81, 82, 83, 84, 85, 86, 198)
RECOVERY_SECONDS = 4800.


def original_paths(repo):
    original = repo / "docs/plans/artifacts/hmc-budget-debug-2026-09-25"
    return (original, original / "source-confirmation-r1",
            original / "confirmation-r1/suite/closure-i2-c1-beta_binomial")


def checked_original_source(source):
    snapshot = read_json(source/"source_snapshot.json")
    expected = snapshot.get("source", snapshot)["files"]
    actual = {str(p.relative_to(source)): file_hash(p)
              for p in sorted((source/"bayesfilter").rglob("*.py"))}
    if actual != expected:
        raise ValueError("original C1 numerical source changed")
    return {"commit": snapshot["git_commit"], "files": actual, "identity": digest(actual)}


def selected_failures(cell):
    receipts = sorted(cell.glob("replication-*/process-attempt-001-exit.json"))
    if len(receipts) != 256:
        raise ValueError("original C1 beta-binomial inventory is incomplete")
    selected = tuple(int(p.parent.name.split("-")[1]) for p in receipts
                     if read_json(p)["status"] == "timed_out")
    if selected != INDICES:
        raise ValueError("original timeout selection changed")
    return selected


def completed_recoveries(previous_cell, original_cell):
    """Reuse clean prior recovery exits without looking at posterior outcomes."""
    if previous_cell is None:
        return []
    if read_json(previous_cell/"isolated_design.json") != read_json(original_cell/"isolated_design.json"):
        raise ValueError("prior recovery design changed")
    rows = []
    for index in INDICES:
        path = previous_cell/f"replication-{index:04d}"
        receipt_path = path/"process-attempt-001-exit.json"
        if not receipt_path.exists():
            continue
        receipt = read_json(receipt_path)
        if receipt["status"] in {"timed_out", "budget_exhausted"}:
            continue
        if receipt["status"] != "complete" or receipt["exit_code"] != 0:
            raise ValueError("prior recovery has a non-budget infrastructure failure")
        if read_json(path/"fit_identity.json") != read_json(original_cell/path.name/"fit_identity.json"):
            raise ValueError("prior recovery identity changed")
        if file_hash(path/"independent_assessment.json") != receipt["assessment_sha256"]:
            raise ValueError("prior recovery assessment changed")
        runtime = read_json(path/"process-attempt-001-manifest.json")["runtime"]
        memory = runtime["memory_policy"]
        if (runtime.get("jit_compile") is not True or "GPU" not in runtime.get("gpu_tensor_device", "")
                or not memory.get("configured_before_logical_device_initialization")
                or not memory.get("all_physical_devices_memory_growth")
                or not memory.get("physical_devices")):
            raise ValueError("prior recovery GPU provenance is invalid")
        rows.append({"replication":index, "status":"complete", "receipt":str(receipt_path),
                     "reused_prior_completion":True, "new_elapsed_seconds":0.})
    return rows


def settle(ledger_path, receipt):
    ledger = read_json(ledger_path)
    reservation = ledger.get("active_reservation", {})
    existing = [r for r in ledger["records"] if r["receipt"] == receipt["receipt"]]
    if not reservation and existing == [receipt]:
        return  # A completed enclosing charge is idempotent.
    if reservation.get("receipt") != receipt["receipt"]:
        raise ValueError("recovery reservation changed; reconcile before continuing")
    if not any(r["receipt"] == receipt["receipt"] for r in ledger["records"]):
        ledger["records"].append(receipt)
    ledger["charged_gpu_seconds"] = sum(r["elapsed_seconds"] for r in ledger["records"])
    ledger["remaining_gpu_seconds"] = ledger["grant_gpu_seconds"]-ledger["charged_gpu_seconds"]
    ledger.pop("active_reservation")
    ledger["updated_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(ledger_path, ledger)


def recover(*, repo, root, ledger_path, service, previous_cell=None):
    original, source, old_cell = original_paths(repo)
    all_indices = selected_failures(old_cell)
    reused = completed_recoveries(previous_cell, old_cell)
    indices = tuple(i for i in all_indices if i not in {r["replication"] for r in reused})
    source_record = checked_original_source(source)
    preserved = {str(p):file_hash(p) for p in (
        original/"confirmation-terminal-result.json", original/"confirmation-r1/execution.json",
        old_cell/"isolated_design.json")}
    for index in all_indices:
        prior = old_cell/f"replication-{index:04d}"
        identity = read_json(prior/"fit_identity.json")
        if identity["source_identity"] != source_record["identity"]:
            raise ValueError("C1 fit/source identity mismatch")
        for name in ("fit_identity.json", "process-attempt-001-exit.json", "tuning/tuning_checkpoint.json"):
            preserved[str(prior/name)] = file_hash(prior/name)
    for row in reused:
        prior = Path(row["receipt"]).parent
        for name in ("fit_identity.json", "process-attempt-001-exit.json", "independent_assessment.json"):
            preserved[str(prior/name)] = file_hash(prior/name)
    ledger = read_json(ledger_path)
    available = ssm_campaign.free_gpu_seconds(ledger, c1_active=ssm_campaign.c1_is_active())
    if available < RECOVERY_SECONDS+30:
        raise ValueError("recovery reservation exceeds settled grant")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("GPU memory growth required before launch")
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("GPU-") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise ValueError("one GPU UUID required")
    root.mkdir(parents=True, exist_ok=False)
    cell = root/"recovery/closure-i2-c1-beta_binomial"
    design_path = write_json(cell/"isolated_design.json", read_json(old_cell/"isolated_design.json"))
    policy = TimeoutPolicy(extension_mode="observed_intervals", max_extension_seconds=750.,
                           gpu_admission_wait_seconds=600., shutdown_grace_seconds=5.)
    receipt_path = root/"recovery-execution.json"
    ledger["active_reservation"] = {"service":service, "receipt":str(receipt_path),
        "maximum_gpu_seconds_including_shutdown":RECOVERY_SECONDS+30}
    write_json(ledger_path, ledger)
    manifest = {"plan_file":PLAN,"command":[sys.executable,*sys.argv], "environment":sys.executable,
        "original_source":source_record, "supervisor_source":str(SOURCE_ROOT),
        "supervisor_file_sha256":file_hash(SOURCE_ROOT/"bayesfilter/testing/inference_validation/fit_supervision.py"),
        "selected_indices":indices,"reused_completions":reused,"gpu_uuid":os.environ["CUDA_VISIBLE_DEVICES"],
        "TF_FORCE_GPU_ALLOW_GROWTH":os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
        "policy":policy.payload(),"maximum_gpu_seconds":RECOVERY_SECONDS+30,
        "data_and_seed_authority":str(design_path),"result_file":str(root/"recovery-result.json"),
        "started_utc":datetime.now(timezone.utc).isoformat()}
    write_json(root/"recovery-manifest.json", manifest)
    started = time.monotonic()
    deadline = started+RECOVERY_SECONDS
    outcomes = list(reused)
    code = 1
    try:
        for index in indices:
            if time.monotonic() >= deadline:
                break
            path = cell/f"replication-{index:04d}"
            admission = wait_for_gpu_admission(device="gpu",policy=policy,deadline=deadline)
            write_json(path/"admission.json",admission)
            if not admission["admitted"]:
                outcomes.append({"replication":index,"status":"resource_deferred"})
                break
            seconds = min(750.,deadline-time.monotonic())
            if seconds <= 0:
                break
            prefix = path/"process-attempt-001"
            command = [sys.executable,"-m","bayesfilter.testing.inference_validation","_pipeline_fit",
                str(design_path),str(cell),str(index),str(seconds),"1","--reuse-leapfrog-graphs"]
            write_json(str(prefix)+"-launch.json",{"command":command,"cwd":str(source),
                "original_fit":str(old_cell/path.name),"recovery_rule":"one fresh rerun of each timeout; same source/design/seeds",
                "additional_budget_seconds":seconds})
            observed = supervise_fit(command,str(prefix)+".log",seconds,"gpu",progress_root=path,
                cell_deadline=deadline,timeout_policy=policy,allowance_path=str(prefix)+"-allowance.json",
                budget_receipt_path=str(prefix)+"-budget.json",cwd=source)
            observed["allocation_limiter"] = "fit" if seconds == 750 else "recovery_enclosure"
            write_json(str(prefix)+"-exit.json",observed)
            assessment = path/"independent_assessment.json"
            if observed["status"] == "complete":
                if not assessment.is_file():
                    raise ValueError("completed recovery missing assessment")
                if read_json(path/"fit_identity.json") != read_json(old_cell/path.name/"fit_identity.json"):
                    raise ValueError("recovery changed numerical source, data, seed or fit identity")
                runtime = read_json(path/"process-attempt-001-manifest.json")["runtime"]
                memory = runtime["memory_policy"]
                if (not runtime.get("jit_compile") or "GPU" not in runtime.get("gpu_tensor_device", "")
                        or not memory.get("configured_before_logical_device_initialization")
                        or not memory.get("all_physical_devices_memory_growth")
                        or not memory.get("physical_devices")):
                    raise ValueError("recovery lacks verified GPU/XLA/memory-growth provenance")
                observed["assessment_sha256"] = file_hash(assessment)
            write_json(str(prefix)+"-exit.json",observed)
            outcomes.append({"replication":index,"status":observed["status"],
                "elapsed_seconds":observed["elapsed_seconds"],"receipt":str(prefix)+"-exit.json"})
            write_json(root/"recovery-progress.json",{"planned":10,"outcomes":outcomes,
                "remaining_enclosing_seconds":max(0.,deadline-time.monotonic())})
            print(outcomes[-1],flush=True)
            if observed["status"] not in {"complete","timed_out","budget_exhausted"}:
                raise RuntimeError("recovery worker infrastructure failure; inspect original receipt")
        if not all(file_hash(Path(p)) == sha for p,sha in preserved.items()):
            raise ValueError("original C1 evidence changed during recovery")
        write_json(root/"original-evidence-checksums.json",preserved)
        attempted = {r["replication"] for r in outcomes}
        outcomes.extend({"replication":i,"status":"unstarted"} for i in indices if i not in attempted)
        write_json(root/"recovery-result.json",{"planned":10,"outcomes":outcomes,
            "completed":sum(r["status"]=="complete" for r in outcomes),"original_confirmation_unchanged":True,
            "interpretation":"post-hoc execution recovery; original coverage claims remain unchanged",
            "numerical_source_identity":source_record["identity"]})
        code = 0
    except BaseException as exc:
        write_json(root/"recovery-failure.json", {"type":type(exc).__name__, "message":str(exc),
            "outcomes":outcomes, "remaining_indices":[i for i in indices
                if i not in {r["replication"] for r in outcomes}]})
        raise
    finally:
        receipt = {"receipt":str(receipt_path),"resource":"gpu","elapsed_seconds":time.monotonic()-started,
            "exit_code":code,"ended_utc":datetime.now(timezone.utc).isoformat(),
            "accounting":"one enclosing recovery charge; original C1 and nested fits not charged again"}
        write_json(receipt_path,receipt)
        settle(ledger_path,receipt)
    return old_cell,cell,source


def run_recovery(args):
    """Independent systemd ceiling bounds the whole recovery process group."""
    unit = args.service.removesuffix(".service") + "-recovery"
    command = [sys.executable, str(Path(__file__).resolve()), "--repo", str(args.repo),
        "--output", str(args.output), "--ssm-root", str(args.ssm_root),
        "--ledger", str(args.ledger), "--service", unit, "--recovery-only"]
    if getattr(args, "previous_cell", None) is not None:
        command.extend(["--previous-cell",str(args.previous_cell)])
    launch = ["systemd-run", "--user", "--wait", "--collect", "--pipe", "--service-type=exec",
        "--unit", unit, "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={RECOVERY_SECONDS}", "--property=TimeoutStopSec=10",
        f"--working-directory={SOURCE_ROOT}",
        f"--setenv=CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}",
        "--setenv=TF_FORCE_GPU_ALLOW_GROWTH=true", "--setenv=TF_NUM_INTRAOP_THREADS=2",
        "--setenv=TF_NUM_INTEROP_THREADS=2", "--setenv=OMP_NUM_THREADS=2",
        "--setenv=BAYESFILTER_PRELOAD_CUSTOM_OP=0", *command]
    confirmed = False
    try:
        result = subprocess.run(launch, check=False, timeout=RECOVERY_SECONDS+30)
        confirmed = True
    except BaseException:
        stop = subprocess.run(["systemctl", "--user", "stop", unit], check=False, timeout=30)
        confirmed = stop.returncode == 0
        raise
    finally:
        # Only after group shutdown may a killed writer's reservation settle.
        active = read_json(args.ledger).get("active_reservation", {})
        if confirmed and active.get("service") == unit:
            receipt = {"receipt":active["receipt"], "resource":"gpu",
                "elapsed_seconds":RECOVERY_SECONDS+30, "exit_code":-1,
                "accounting":"conservative enclosing cap after confirmed group shutdown"}
            write_json(active["receipt"],receipt)
            settle(args.ledger,receipt)
    if result.returncode != 0:
        raise RuntimeError("recovery stopped; inspect preserved recovery failure and service log")
    _, source, old = original_paths(args.repo)
    return old, args.output/"recovery/closure-i2-c1-beta_binomial", source


SUMMARY_CODE = '''
import sys
from pathlib import Path
from bayesfilter.testing.inference_validation.designs import ValidationDesign
from bayesfilter.testing.inference_validation.fit_process import _record
from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications
from bayesfilter.testing.inference_validation.storage import read_json,write_json,file_hash
old,new,output=map(Path,sys.argv[1:4])
previous=Path(sys.argv[4]) if len(sys.argv)>4 else None
design=ValidationDesign.from_payload(read_json(old/'isolated_design.json'))
records=[]; origins=[]
for index in range(256):
    old_path=old/f'replication-{index:04d}'
    recovery=new/old_path.name
    original=read_json(old_path/'process-attempt-001-exit.json')
    prior=previous/old_path.name if previous is not None else old_path
    path=(recovery if (recovery/'process-attempt-001-exit.json').exists() else
          prior if (prior/'process-attempt-001-exit.json').exists() else old_path)
    receipt=read_json(path/'process-attempt-001-exit.json')
    if path!=old_path and original['status']!='timed_out':raise ValueError('replaced a completed original fit')
    assessment=path/'independent_assessment.json'
    if assessment.exists() and file_hash(assessment)!=receipt.get('assessment_sha256'):raise ValueError('assessment changed')
    records.append(_record(path,index,receipt,design))
    origins.append({'replication':index,'origin':'recovery' if path==recovery else 'prior_recovery' if path!=old_path else 'original','status':receipt['status']})
report=summarize_replications(design,records)
write_json(output,{'interpretation':'post-hoc recovery, not a new prospective confirmation',
    'original_denominator':256,'origins':origins,'assessment':report})
'''


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ssm-root",type=Path,required=True)
    parser.add_argument("--ledger",type=Path,required=True)
    parser.add_argument("--service",required=True)
    parser.add_argument("--recovery-only",action="store_true")
    parser.add_argument("--previous-cell",type=Path)
    args = parser.parse_args()
    args.repo, args.output = args.repo.resolve(),args.output.resolve()
    args.ledger, args.ssm_root = args.ledger.resolve(),args.ssm_root.resolve()
    def interrupted(signum,frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    if args.recovery_only:
        recover(repo=args.repo,root=args.output,ledger_path=args.ledger,service=args.service,
                previous_cell=args.previous_cell)
        return
    old,new,source = run_recovery(args)
    summary_command = [sys.executable,"-c",SUMMARY_CODE,str(old),str(new),str(args.output/"recovered-beta-summary.json")]
    if args.previous_cell is not None:
        summary_command.append(str(args.previous_cell))
    subprocess.run(summary_command,
        cwd=source,env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1"},check=True,timeout=60)
    try:
        for stage in ("preflight-mechanics","preflight-pipeline","pricing","main"):
            write_json(args.output/"sequence-status.json",{"stage":stage,"state":"running","ssm_root":str(args.ssm_root)})
            receipt = ssm_campaign.run_stage(args.ssm_root,stage,args.ledger)
            print({"stage":stage,**receipt},flush=True)
            if receipt.get("exit_code",0) != 0:
                write_json(args.output/"sequence-status.json",{"stage":stage,"state":"requires_diagnosis","receipt":receipt})
                raise SystemExit(1)
        write_json(args.output/"sequence-status.json",{"stage":"terminal","state":"complete; inspect per-slot results"})
    finally:
        ssm_campaign.summarize(args.ssm_root)


if __name__ == "__main__":
    main()
