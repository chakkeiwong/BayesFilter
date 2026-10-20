"""Bounded complete development-training supervisor; replication not qualified."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PLAN="docs/plans/controlled-smc-kdm-complete-training-20261007.md"
RESULT="docs/plans/controlled-smc-kdm-complete-training-result-20261007.md"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("source","tf"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--verification",type=Path)
    p.add_argument("--epochs",choices=("smoke","full"),required=True)
    args=p.parse_args()
    if args.mode=="tf" and args.reference is None: p.error("TF needs --reference")
    if args.mode=="tf" and args.verification is None: p.error("TF needs current --verification")
    out=args.output.resolve()
    if out.parent!=BASE or not out.name.startswith("kdm-complete-training-20261007-"):
        raise ValueError("Output outside phase root")
    manifests=[json.loads(p.read_text()) for p in BASE.glob("kdm-complete-training-20261007-*/manifest.json")]
    if len(manifests)>=6: raise RuntimeError("Six-attempt phase cap exhausted")
    resource="cpu" if args.mode=="source" else "gpu"
    if any(m["status"]=="RUNNING" and m["resource"]==resource for m in manifests):
        raise RuntimeError("A phase run on this resource is already active")
    used=sum(m.get("wall_seconds",0) for m in manifests if m["resource"]==resource)
    timeout=int((64800 if resource=="cpu" else 57600)-used)
    if timeout<30: raise RuntimeError("Phase resource budget exhausted")
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",MPLBACKEND="Agg")
    if resource=="cpu":
        interpreter="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
        env.update(CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
        driver="docs/benchmarks/reference_kdm_fresh_training_cost.py"
    else:
        interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
        env["CUDA_VISIBLE_DEVICES"]=GPU
        driver="docs/benchmarks/diagnose_kdm_fresh_training_cost_tf.py"
    command=[interpreter,"-I","-B",str(ROOT/driver),"--output",str(out/"diagnostic"),"--epochs",args.epochs]
    if args.reference: command += ["--reference",str(args.reference.resolve())]
    if args.mode=="tf": command += ["--verification",str(args.verification.resolve())]
    files=[Path(__file__),ROOT/driver,ROOT/PLAN]
    files+=sorted((ROOT/"bayesfilter/replication").glob("kdm*.py"))
    files+=[ROOT/"docs/benchmarks/diagnose_kdm_source_epoch_tf.py"]
    if resource=="cpu":
        old=ROOT/".localresources/worktrees/ledh-iapf-execution-20260922/docs/benchmarks"
        files += [old/"diagnose_replication_training.py",old/"diagnose_replication_stage0.py"]
    if args.verification: files.append(args.verification.resolve())
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if args.epochs=="full":
        smokes=[m for m in manifests if m["mode"]==args.mode and m.get("epoch_mode")=="smoke" and m["status"]=="COMPLETED"]
        if not smokes: raise RuntimeError("Passing same-backend two-epoch smoke required")
        proof=smokes[-1]
        if proof["hashes"]!=hashes: raise RuntimeError("Code or plan changed after smoke")
        if proof.get("reference_sha256")!=hashlib.sha256(args.reference.read_bytes()).hexdigest():
            raise RuntimeError("Fresh reference changed after smoke")
    manifest=dict(schema="kdm_complete_training_manifest_v1",resource=resource,mode=args.mode,epoch_mode=args.epochs,
        command=command,launcher_command=[sys.executable]+sys.argv,cwd=str(ROOT),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan=PLAN,result=RESULT,timeout_seconds=timeout,status="RUNNING",started_unix_seconds=time.time(),
        seeds=dict(data=202610071,constructors=[202610072,202610073,202610074],tf_noise_base=202610082),
        data="fresh source development train5000/validation1000",
        cpu_gpu_status="CPU reference; GPU intentionally hidden" if resource=="cpu" else "trusted GPU/XLA",
        environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","TF_FORCE_GPU_ALLOW_GROWTH",
            "OMP_NUM_THREADS","TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS")},
        hashes=hashes,supervisor_pid=os.getpid(),heartbeat_timeout_seconds=1800)
    if args.reference:
        manifest.update(reference=str(args.reference.resolve()),reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    def save():
        temporary=out/"manifest.json.tmp"
        temporary.write_text(json.dumps(manifest,indent=2)+"\n")
        temporary.replace(out/"manifest.json")
    save(); started=time.monotonic()
    with (out/"command.log").open("w") as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        manifest["child_pid"]=process.pid; save()
        while True:
            try:
                code=process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-started
                progress=out/"diagnostic/progress.json"
                last=progress.stat().st_mtime if progress.exists() else manifest["started_unix_seconds"]
                stalled=time.time()-last>1800
                manifest.update(elapsed_running_seconds=elapsed,last_progress_unix_seconds=last)
                save()
                if elapsed>=timeout or stalled:
                    manifest["termination_reason"]="HEARTBEAT_STALL" if stalled else "PHASE_TIME_LIMIT"
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                    code=124
                    break
    manifest.update(returncode=code,wall_seconds=time.monotonic()-started,status="COMPLETED" if code==0 else "FAILED")
    result=out/"diagnostic/result.json"
    if result.exists():
        manifest["result_sha256"]=hashlib.sha256(result.read_bytes()).hexdigest()
        payload=json.loads(result.read_text())
        for key in ("status","source_training_protocol_complete","memory_policy","visible_device","jit_compile","tf32"):
            if key in payload: manifest["result_"+key]=payload[key]
        if code==0 and payload.get("status")!="PASS":
            code=1
            manifest.update(status="FAILED",returncode=code)
    elif code==0:
        code=1
        manifest.update(status="FAILED",returncode=code,termination_reason="MISSING_RESULT")
    save()
    print(json.dumps({k:manifest[k] for k in ("status","returncode","wall_seconds","mode")}))
    print(out)
    return code
if __name__=="__main__": raise SystemExit(main())
