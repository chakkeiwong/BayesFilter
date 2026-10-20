"""Bounded launcher for source-stage diagnostics; no arbitrary command execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
PLAN="docs/plans/controlled-smc-kdm-source-stages-20261007.md"
RESULT="docs/plans/controlled-smc-kdm-source-stages-result-20261007.md"
GPU="GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("export","verify","epoch"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--reference",type=Path)
    p.add_argument("--verification",type=Path)
    args=p.parse_args()
    if args.mode!="export" and args.reference is None: p.error("--reference required")
    out=args.output.resolve()
    if out.parent!=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928":
        raise ValueError("Output must be within the reviewed campaign root")
    attempts=list(out.parent.glob("kdm-source-stages-20261007-??/manifest.json"))
    if len(attempts)>=8: raise RuntimeError("Eight-attempt repaired phase cap exhausted")
    resource="cpu" if args.mode=="export" else "gpu"
    used=sum(json.loads(f.read_text()).get("wall_seconds",0) for f in attempts
             if json.loads(f.read_text()).get("resource")==resource)
    timeout=int((1800 if resource=="cpu" else 3600)-used)
    if timeout<30: raise RuntimeError("Phase resource budget exhausted")
    out.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ)
    env.update(OMP_NUM_THREADS="1",TF_NUM_INTEROP_THREADS="1",TF_NUM_INTRAOP_THREADS="1",
               TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2")
    if resource=="cpu":
        interpreter="/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
        env.update(CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
        driver="docs/benchmarks/reference_kdm_source_stages.py"
    else:
        interpreter="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
        env["CUDA_VISIBLE_DEVICES"]=GPU
        driver="docs/benchmarks/diagnose_kdm_source_stages_tf.py"
    command=[interpreter,"-B",str(ROOT/driver),"--output",str(out/"diagnostic")]
    if args.mode!="export":
        command+=["--mode",args.mode,"--reference",str(args.reference.resolve())]
    if args.verification:
        command += ["--verification",str(args.verification.resolve())]
    files=[Path(__file__),ROOT/driver,ROOT/PLAN,ROOT/"docs/benchmarks/diagnose_kdm_source_epoch_tf.py"]
    files+=sorted((ROOT/"bayesfilter/replication").glob("kdm*.py"))
    files+=[ROOT/"docs/benchmarks/reference_kdm_neural.py", ROOT/"docs/benchmarks/reference_kdm_stage_transfer.py"]
    manifest=dict(schema="kdm_source_stages_manifest_v1",resource=resource,mode=args.mode,
        command=command,launcher_command=[sys.executable]+sys.argv,cwd=str(ROOT),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        plan=PLAN,result=RESULT,timeout_seconds=timeout,status="RUNNING",
        seeds=[20261007,20261008,811,731],data="source saved train5000/validation1000",
        cpu_gpu_status="CPU reference, GPU intentionally hidden" if resource=="cpu" else "trusted GPU/XLA",
        environment={k:env.get(k) for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH",
            "TF_FORCE_GPU_ALLOW_GROWTH","OMP_NUM_THREADS","TF_NUM_INTEROP_THREADS","TF_NUM_INTRAOP_THREADS")},
        hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    if args.reference:
        manifest["reference"]=str(args.reference.resolve())
        manifest["reference_sha256"]=hashlib.sha256(args.reference.read_bytes()).hexdigest()
    def save(): (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    save()
    start=time.monotonic()
    with (out/"command.log").open("w") as log:
        try:
            run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
            code=run.returncode
        except subprocess.TimeoutExpired:
            code=124
    manifest.update(returncode=code,wall_seconds=time.monotonic()-start,status="COMPLETED" if code==0 else "FAILED")
    result=out/"diagnostic/result.json"
    if result.exists(): manifest["result_sha256"]=hashlib.sha256(result.read_bytes()).hexdigest()
    save()
    print(json.dumps({k:manifest[k] for k in ("status","returncode","wall_seconds","mode")}))
    print(out)
    return code
if __name__=="__main__": raise SystemExit(main())
