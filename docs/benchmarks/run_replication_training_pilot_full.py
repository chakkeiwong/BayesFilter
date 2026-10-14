"""Bounded repaired KDM complete-training and common-author evaluation supervisor."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE=BASE/"repaired-training-20261011"
PLAN="docs/plans/controlled-smc-kdm-repaired-training-pilot-20261011.md"
RESULT="docs/plans/controlled-smc-kdm-mechanism-result-20261011.md"
TF="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
SOURCE="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
CAP={"gpu":28800.,"cpu":1800.}
PIN="b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)

def main(out,preflight=False):
    from run_replication_gap_localization import campaign
    if out.parent!=PHASE or out.name!="training-pilot-02":
        raise ValueError("Use the reviewed versioned full-pilot output")
    previous=PHASE/"training-pilot-01"
    prior=list(PHASE.glob("training-pilot-*/manifest.json"))
    if len(prior)!=1 or prior[0].parent!=previous:
        raise RuntimeError("Full pilot requires exactly the completed readiness attempt")
    readiness=read(previous/"manifest.json")
    cost=read(previous/"decision.json")
    if readiness["status"]!="COMPLETED" or cost["status"]!="READY_FOR_FULL":
        raise RuntimeError("Cost readiness has not passed")
    if readiness["decision_sha256"]!=sha(previous/"decision.json"):
        raise RuntimeError("Cost decision changed")
    if cost["projected_seconds"]>cost["threshold_seconds"]:
        raise RuntimeError("Cost projection exceeds the reviewed cap")
    for name,digest in readiness["source_hashes"].items():
        if name.startswith("bayesfilter/replication/") or Path(name).name in (
                "diagnose_kdm_source_epoch_tf.py","diagnose_kdm_fresh_training_cost_tf.py"):
            if sha(ROOT/name)!=digest:
                raise RuntimeError("Runtime changed since cost readiness: "+name)
    for name,digest in readiness["input_hashes"].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError("Input changed since cost readiness: "+name)
    from run_replication_gap_localization import campaign
    balance=campaign()
    spent={r:sum(read(p)["resource_wall_seconds"].get(r,0.) for p in prior) for r in CAP}
    remaining={r:min(CAP[r]-spent[r],balance["remaining_seconds"][r],
                     balance["tranche_remaining_seconds"][r]) for r in CAP}
    if remaining["gpu"]<cost["projected_seconds"] or remaining["cpu"]<300:
        raise RuntimeError("Insufficient existing pilot allocation")
    stage=BASE/"stage-refresh-20261011/stage-refresh-01"
    proof=stage/"verification/diagnostic/result.json"
    reference=BASE/"kdm-fresh-training-cost-20261007-01/diagnostic/result.json"
    source=BASE/"kdm-complete-training-20261007-03/diagnostic/result.json"
    source_result=read(source)
    fresh=read(reference)
    if (read(proof)["status"]!="PASS" or source_result["status"]!="PASS" or
            source_result["source_pin"]!=fresh["source_pin"] or
            fresh["source_pin"]!=PIN or not source_result["source_training_protocol_complete"] or
            source_result["source_training_configs"]!=fresh["source_training_configs"]):
        raise RuntimeError("Source/reference/current proof mismatch")
    for section in ("datasets","fixtures"):
        if set(source_result[section])!=set(fresh[section]):
            raise RuntimeError(section+" identities mismatch")
        for name,item in fresh[section].items():
            other=source_result[section][name]
            if item["sha256"]!=other["sha256"] or sha(item["path"])!=item["sha256"]:
                raise RuntimeError(section+"/"+name+" checksum mismatch")
    runtime=[ROOT/"docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py",
             ROOT/"docs/benchmarks/diagnose_kdm_source_epoch_tf.py",
             ROOT/"bayesfilter/runtime/gpu_memory_policy.py",proof]
    runtime+=sorted((ROOT/"bayesfilter/replication").glob("kdm*.py"))
    code=[ROOT/PLAN,ROOT/"docs/benchmarks/run_replication_gap_localization.py",
          ROOT/"docs/benchmarks/run_replication_training_pilot_full.py",
          ROOT/"docs/benchmarks/reference_kdm_training_terminal_audit.py",
          ROOT/"docs/benchmarks/reference_kdm_development_evaluation.py",
          ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py"]
    paths=list(dict.fromkeys(runtime+code))
    if preflight:
        print(json.dumps(dict(status="PREFLIGHT_PASS",output=str(out),
            remaining_seconds=remaining,source_pin=PIN,
            projected_gpu_seconds=cost["projected_seconds"],
            checked_code_files=len(paths))))
        return 0
    out.mkdir(parents=True,exist_ok=False)
    for path in paths:
        target=out/"sources"/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,TF_FORCE_GPU_ALLOW_GROWTH="true",
        TF_CPP_MIN_LOG_LEVEL="2",OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",MPLBACKEND="Agg")
    gpu_env.pop("LD_LIBRARY_PATH",None)
    cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",
        LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    manifest=dict(schema="repaired_kdm_development_training_pilot_v1",status="RUNNING",
        command=[sys.executable,*sys.argv],cwd=str(ROOT),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan_file=PLAN,result_file=RESULT,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        supervisor_pid=os.getpid(),phase_cap_seconds=CAP,prior_phase_seconds=spent,
        campaign_remaining_before=remaining,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (reference,source,proof,previous/"decision.json")},
        source_pin=PIN,source_training_run=str(source.parent.parent),tf_training_run=str(out/"training"),
        seeds=dict(data=202610071,constructors=[202610072,202610073,202610074],
                   tf_noise_base=202610082,evaluation_data=202610081,
                   evaluation_filter=[202610082,202610083]),
        environment_gpu={k:gpu_env.get(k) for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH",
             "TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS","OMP_NUM_THREADS")},
        environment_cpu={k:cpu_env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH",
             "OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        cpu_gpu_status="TF training on escalated RTX 5080; independent author evaluation CPU only",
        resource_wall_seconds={"cpu":0.,"gpu":0.},steps=[],replication="INCOMPLETE")
    def save(): write(out/"manifest.json",manifest)
    def run(label,command,resource,env,progress=None):
        allowance=CAP[resource]-spent[resource]-manifest["resource_wall_seconds"][resource]
        allowance=min(allowance,remaining[resource]-manifest["resource_wall_seconds"][resource])
        if allowance<=0: raise RuntimeError(resource+" pilot cap exhausted")
        step=dict(label=label,command=command,resource=resource,status="RUNNING",
                  timeout_seconds=allowance,log=label+".log")
        manifest["steps"].append(step);save()
        tick=time.monotonic();child=None
        try:
            with (out/step["log"]).open("x") as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                step["pid"]=child.pid;save()
                while True:
                    try:
                        code=child.wait(timeout=max(.1,min(30.,allowance-(time.monotonic()-tick))))
                        break
                    except subprocess.TimeoutExpired:
                        elapsed=time.monotonic()-tick
                        age=(time.time()-progress.stat().st_mtime if progress is not None and progress.exists()
                             else elapsed)
                        step.update(elapsed_seconds=elapsed,progress_age_seconds=age);save()
                        if elapsed>=allowance or age>1800:
                            raise TimeoutError(label+(" PHASE_TIME_LIMIT" if elapsed>=allowance else " HEARTBEAT_STALL"))
            step.update(status="COMPLETED" if code==0 else "FAILED",returncode=code)
            if code: raise RuntimeError(label+" exit "+str(code)+"; see "+step["log"])
        except BaseException:
            step["status"]="FAILED"
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
        finally:
            step["wall_seconds"]=time.monotonic()-tick
            manifest["resource_wall_seconds"][resource]+=step["wall_seconds"]
            save()
        return step
    save()
    train=out/"training"
    train.mkdir()
    training_manifest=dict(schema="kdm_complete_training_manifest_v1",mode="tf",
        epoch_mode="full",status="RUNNING",cwd=str(ROOT),git_commit=manifest["git_commit"],
        plan=PLAN,result=RESULT,reference=str(reference),reference_sha256=sha(reference),
        verification=str(proof),verification_sha256=sha(proof),
        hashes={str(p.relative_to(ROOT)):sha(p) for p in runtime},
        seeds=manifest["seeds"],environment=manifest["environment_gpu"],
        cpu_gpu_status=manifest["cpu_gpu_status"],source_pin=PIN,replication="INCOMPLETE")
    write(train/"manifest.json",training_manifest)
    decision=dict(status="ERROR",scope="HARNESS_OR_EXECUTION_FAILURE",replication="INCOMPLETE")
    try:
        run("nvidia",["nvidia-smi","--query-gpu=uuid,name,memory.used,memory.total",
            "--format=csv"],"gpu",gpu_env)
        driver=ROOT/"docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py"
        training_command=[TF,"-I","-B",str(driver),"--epochs","full","--output",
            str(train/"diagnostic"),"--reference",str(reference),"--verification",str(proof)]
        training_manifest["command"]=training_command
        write(train/"manifest.json",training_manifest)
        try:
            step=run("full-training",training_command,"gpu",gpu_env,
                     train/"diagnostic/progress.json")
            result=read(train/"diagnostic/result.json")
            if (result["status"]!="PASS" or result["checks_failed"]!=0 or
                not result["source_training_protocol_complete"] or
                [x["stage"] for x in result["completed_stages"]]!=["weights","mdpf","am_dpf"] or
                result["reference_sha256"]!=sha(reference) or
                result["verification"]["sha256"]!=sha(proof) or
                result["tf32"] is not False or result["jit_compile"] is not True or
                result["visible_device"]!=GPU or
                not result["memory_policy"]["all_physical_devices_memory_growth"]):
                raise RuntimeError("Full training result violates source/GPU/XLA contract")
            training_manifest.update(status="COMPLETED",returncode=0,
                 wall_seconds=step["wall_seconds"],result_sha256=sha(train/"diagnostic/result.json"),
                 memory_policy=result["memory_policy"],jit_compile=result["jit_compile"],
                 tf32=result["tf32"],visible_device=result["visible_device"])
        except BaseException:
            training_manifest.update(status="FAILED",returncode=1)
            raise
        finally:
            write(train/"manifest.json",training_manifest)
        audit=out/"audit.json"
        run("terminal-audit",[SOURCE,"-I","-B",
            str(ROOT/"docs/benchmarks/reference_kdm_training_terminal_audit.py"),
            "--run",str(train),"--output",str(audit)],"cpu",cpu_env)
        audited=read(audit)
        if (audited["verdict"]!="PASS" or audited["failures"] or
            audited["run_manifest_sha256"]!=sha(train/"manifest.json")):
            raise RuntimeError("Independent TF training terminal audit failed")
        evaluation=out/"evaluation"
        evaluation.mkdir()
        run("common-author-evaluation",[SOURCE,"-I","-B",
            str(ROOT/"docs/benchmarks/reference_kdm_development_evaluation.py"),
            "--worker","--output",str(evaluation/"diagnostic"),
            "--tf-run",str(train),"--tf-audit",str(audit)],"cpu",cpu_env)
        report=read(evaluation/"diagnostic/result.json")
        if (report["decision"]["status"]!="PASS" or report["failures"] or
            report["tf_training_run"]!=str(train) or
            report["tf_terminal_audit_sha256"]!=sha(audit) or
            len(report["comparisons"])!=4):
            raise RuntimeError("Common-author evaluation integrity failed")
        for name,digest in manifest["source_hashes"].items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Code changed during pilot: "+name)
        for name,digest in manifest["input_hashes"].items():
            if sha(ROOT/name)!=digest:raise RuntimeError("Input changed during pilot: "+name)
        manifest.update(status="COMPLETED",training_result_sha256=sha(train/"diagnostic/result.json"),
            training_manifest_sha256=sha(train/"manifest.json"),audit_sha256=sha(audit),
            evaluation_sha256=sha(evaluation/"diagnostic/result.json"))
        decision=dict(status="DEVELOPMENT_PAIR_COMPLETED",scope="ONE_REPAIRED_TF_UNIT",
            replication="INCOMPLETE",training_complete=True,
            source_training_protocol_complete=True,terminal_audit="PASS",
            comparisons=[{k:x[k] for k in ("variant","seed","nll_difference","rmse_ratio")}
                         for x in report["comparisons"]],
            inference="DESCRIPTIVE_ONLY_ONE_INDEPENDENT_TRAINING_PAIR",
            next_action="Review native TF evaluation and design multi-unit qualification if justified")
    except BaseException:
        manifest["status"]="FAILED"
        (out/"error.txt").write_text(traceback.format_exc())
    finally:
        write(out/"decision.json",decision)
        manifest["decision_sha256"]=sha(out/"decision.json")
        save()
        ledger=BASE/"reference-diagnostics.json"
        records=read(ledger)
        for resource in CAP:
            record=dict(id=out.name+"_"+resource,resource=resource,
                charged_seconds=manifest["resource_wall_seconds"][resource],
                summary=str((out/"decision.json").relative_to(ROOT)),
                summary_sha256=sha(out/"decision.json"),
                manifest=str((out/"manifest.json").relative_to(ROOT)),
                plan_file=PLAN,result_file=RESULT,stage="REPAIRED_KDM_COMPLETE_DEVELOPMENT_PILOT",
                next_action="Inspect completed run and resolve remaining replication gaps",
                after_attempt=balance["attempts"],
                executions=[dict(path=str(out.relative_to(ROOT)),resource=resource,
                    wall_seconds=manifest["resource_wall_seconds"][resource])])
            if any(x["id"]==record["id"] for x in records):
                raise RuntimeError("Duplicate budget ledger ID")
            records.append(record)
        write(ledger,records)
    print(json.dumps(dict(status=manifest["status"],decision=decision,
        resource_wall_seconds=manifest["resource_wall_seconds"],output=str(out))))
    return 0 if manifest["status"]=="COMPLETED" else 1

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--preflight",action="store_true")
    args=parser.parse_args()
    raise SystemExit(main(args.output.resolve(),preflight=args.preflight))
