#!/usr/bin/env python3
"""Serial selected-consumer FP64 check under the existing HMC repair budget."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_surrogate_hmc_performance_repair as master

CAMPAIGN_SUBDIR = "selected-consumer-precision-01"
PASS = "SELECTED_CONSUMER_PRECISION_PASS"
PAIRED_TERMINAL = {"PAIRED_CONSUMER_CHECKS_COMPLETE", "LOCAL_REPAIR_REQUIRED",
                   "BUDGET_OR_ATTEMPT_LIMIT_REACHED"}


def read(path):
    return json.loads(Path(path).read_text())


def paired_finished(campaign):
    path = campaign / "consumer-paired-followups-01" / "next-phase-plan.json"
    if not path.is_file() or read(path).get("status") not in PAIRED_TERMINAL:
        return False
    return not any(a.get("status") == "RUNNING"
                   for a in read(campaign / "budget.json")["attempts"])


def worker(name, campaign, selection, scope, output):
    """GPU child. Set allocator policy before TensorFlow is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.testing import surrogate_hmc_five_model as runtime
    from bayesfilter.testing.surrogate_hmc_selected_precision_diagnostic import run
    case = runtime.load_case(name)
    case = runtime.configure_precision(case, scope["runtime_precision"],
                                       scope["tuning_seed_offset"])
    from dataclasses import replace
    case.controls = replace(case.controls,
        shared_gaussian_contraction=scope["contraction"])
    start = time.monotonic()
    try:
        result = run(case, selection, output)
        result.update(memory_policy=memory_policy, tensorflow=tf.__version__,
            gpu_devices=[tf.config.experimental.get_device_details(d)
                         for d in tf.config.list_physical_devices("GPU")],
            tf32=False, cpu_only=False, command=sys.argv,
            conda_env=sys.prefix, git_commit=read(campaign/"budget.json")["git_commit"],
            elapsed_worker_seconds=time.monotonic()-start,
            source_snapshot=str(Path(output)/"source"))
        closure=[ROOT/relative for relative in result["scope"]["source_sha256"]]
        for source in (Path(__file__),
                       ROOT/"bayesfilter/testing/surrogate_hmc_selected_precision_diagnostic.py",
                       *closure):
            target=Path(output)/"source"/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        runtime.save(Path(output)/"selected-consumer-precision.json",result)
        return 0 if result["status"] == PASS else 1
    except Exception:
        Path(output).mkdir(parents=True,exist_ok=True)
        (Path(output)/"failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1


def execute(campaign, output, wait_seconds=36000, per_model_seconds=3600):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if any(not math.isfinite(v) or v <= 0 for v in (wait_seconds,per_model_seconds)):
        raise ValueError("finite positive wait and model caps required")
    output.mkdir(parents=True,exist_ok=False)
    state = dict(status="WAITING_FOR_PAIRED_CONSUMER", campaign=str(campaign),
        command=sys.argv, python=sys.executable,
        created_utc=datetime.now(timezone.utc).isoformat(),
        evidence_contract=str(campaign/"selected-consumer-precision-contract.md"),
        models=[], posterior_sampling_complete=False)
    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        master.save(output/"next-phase-plan.json",state)
    save()
    deadline=time.monotonic()+wait_seconds
    while not paired_finished(campaign):
        left=deadline-time.monotonic()
        if left<=0:
            state.update(status="WAIT_LIMIT_REACHED",
                next="inspect paired controller, then resume this independent check")
            save(); return 1
        time.sleep(min(30,left))
    scopes=master.execution_scopes(campaign/"final-scope-execution.json")
    for name in master.MODELS:
        ledger,left=master.campaign_remaining(campaign/"budget.json")
        if left<=0 or master.campaign_attempts_remaining(ledger)<=0:
            state.update(status="BUDGET_OR_ATTEMPT_LIMIT_REACHED",
                next="report completed checks and remaining arithmetic risk")
            save(); return 1
        if any(a.get("status")=="RUNNING" for a in ledger["attempts"]):
            raise RuntimeError("campaign GPU worker overlap")
        selection=campaign/"final-scope-advance-01"/name/"filter-tuning"/name/"filter-selection.json"
        if not selection.is_file() or read(selection).get("selected") is None:
            state["models"].append(dict(model=name,status="FRESH_SELECTION_UNAVAILABLE"))
            save(); continue
        timeout=min(left,per_model_seconds)
        child_output=output/name
        command=[sys.executable,str(Path(__file__).resolve()),"worker",
            "--model",name,"--campaign",str(campaign),"--selection",str(selection),
            "--scope-plan",str(campaign/"final-scope-execution.json"),
            "--output",str(child_output)]
        state.update(status="RUNNING",active_model=name,
            next="finish selected arithmetic check; continue independent scopes")
        save()
        started=time.monotonic()
        entry=dict(output=str(child_output),model=name,command=command,status="RUNNING",
            started_utc=datetime.now(timezone.utc).isoformat(),reserved_seconds=timeout)
        ledger["attempts"].append(entry)
        ledger["next_action"]="running selected-consumer precision "+name
        master.save(campaign/"budget.json",ledger)
        try:
            with (output/(name+".log")).open("w") as log:
                child=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     timeout=timeout)
            code=child.returncode
            status=read(child_output/"selected-consumer-precision.json").get("status","MISSING_RESULT") if (
                child_output/"selected-consumer-precision.json").is_file() else "MISSING_RESULT"
        except subprocess.TimeoutExpired:
            code=None; status="TIMEOUT"
        entry.update(wall_seconds=time.monotonic()-started,exit_code=code,status=status,
            result=str(child_output/"selected-consumer-precision.json"))
        ledger["used_seconds"]=sum(float(a.get("wall_seconds",a.get("reserved_seconds",0)))
                                   for a in ledger["attempts"])
        ledger["remaining_seconds"]=max(0.,ledger["authorized_gpu_wall_seconds"]-ledger["used_seconds"])
        ledger["next_action"]="inspect selected arithmetic and continue planned repair"
        master.save(campaign/"budget.json",ledger)
        state["models"].append(dict(model=name,status=status,returncode=code,
            wall_seconds=entry["wall_seconds"],artifact=entry["result"],command=command))
        save()
    all_pass=len(state["models"])==len(master.MODELS) and all(
        row["status"]==PASS and row.get("returncode")==0 for row in state["models"])
    state.update(status="SELECTED_CONSUMER_PRECISION_COMPLETE" if all_pass else
                 "SELECTED_CONSUMER_PRECISION_REPAIR_REQUIRED",active_model=None,
        remaining_campaign_seconds=master.campaign_remaining(campaign/"budget.json")[1],
        next="assess arithmetic, cost, posterior target/admission and remaining master phases")
    save()
    return 0 if all_pass else 1


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("execute","worker"))
    parser.add_argument("--campaign",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",choices=master.MODELS)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--scope-plan",type=Path)
    parser.add_argument("--wait-seconds",type=float,default=36000)
    parser.add_argument("--per-model-seconds",type=float,default=3600)
    args=parser.parse_args()
    if args.mode=="worker":
        if not (args.model and args.selection and args.scope_plan):
            parser.error("worker requires model, selection and scope plan")
        scopes=master.execution_scopes(args.scope_plan)
        sys.exit(worker(args.model,args.campaign,args.selection,scopes[args.model],args.output))
    if any((args.model,args.selection,args.scope_plan)):
        parser.error("execute derives model scopes from campaign")
    sys.exit(execute(args.campaign,args.output,args.wait_seconds,args.per_model_seconds))
