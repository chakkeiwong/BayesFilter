"""Bounded local launcher for the reviewed adaptive-consumer reference campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/"docs/plans/artifacts/iapf-adaptive-consumer-20260922-01"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--attempt",required=True)
    p.add_argument("--mode",choices=("preflight","consumer","r_replay"),required=True)
    p.add_argument("--device",choices=("cpu","gpu"),default="cpu")
    p.add_argument("--cases")
    p.add_argument("--source-attempt")
    a=p.parse_args()
    if not a.attempt.isidentifier(): raise ValueError("simple attempt identifier required")
    manifest=EVIDENCE/(a.attempt+"-launch.json")
    if manifest.exists(): raise FileExistsError(manifest)
    old=list(EVIDENCE.glob("*-launch.json"))
    records=[json.loads(x.read_text()) for x in old]
    if len(records)>=12: raise RuntimeError("attempt budget exhausted")
    if any(r.get("status")=="running" for r in records): raise RuntimeError("inspect existing running attempt first")
    used_cpu=sum(r.get("cpu_seconds",0) for r in records)
    used_gpu=sum(r.get("gpu_seconds",0) for r in records)
    if used_cpu>=14400 or used_gpu>=7200: raise RuntimeError("suballocation exhausted")
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1" if a.device=="cpu" else "GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
         "TF_FORCE_GPU_ALLOW_GROWTH":"true","OPENBLAS_NUM_THREADS":"1","OMP_NUM_THREADS":"1",
         "MKL_NUM_THREADS":"1","TF_NUM_INTEROP_THREADS":"1","TF_NUM_INTRAOP_THREADS":"1",
         "BAYESFILTER_PRELOAD_CUSTOM_OP":"0"}
    command=["/home/chakwong/anaconda3/envs/tftwogpu/bin/python",str(ROOT/"docs/benchmarks/diagnose_iapf_adaptive_consumer.py"),
             "--mode",a.mode,"--device",a.device,"--output",str(EVIDENCE/a.attempt)]
    if a.cases: command += ["--cases",a.cases]
    if a.source_attempt:
        if not a.source_attempt.isidentifier(): raise ValueError("simple source attempt identifier required")
        command += ["--source-attempt",str(EVIDENCE/a.source_attempt)]
    sources=["docs/benchmarks/diagnose_iapf_adaptive_consumer.py","docs/benchmarks/reference_iapf_tf_adaptive.R",
        "docs/benchmarks/check_iapf_tf_adaptive.R","docs/benchmarks/run_iapf_adaptive_attempt.py",
        "docs/benchmarks/reference_iapf_paper.R","bayesfilter/score_study/iapf_adapter.py",
        "bayesfilter/score_study/iapf_fit_tf.py","bayesfilter/score_study/fitted_twist_tf.py",
        "bayesfilter/score_study/gaussian_tf.py","bayesfilter/score_study/conditional_means_tf.py",
        "docs/plans/iapf-adaptive-consumer-parity-2026-09-22.md"]
    hashes={}
    for name in sources:
        source=ROOT/name; dest=EVIDENCE/(a.attempt+"-source")/name
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest)
        hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
    record=dict(status="running",command=command,device=a.device,sources=hashes,
        plan="docs/plans/iapf-adaptive-consumer-parity-2026-09-22.md",timeout_seconds=900)
    manifest.write_text(json.dumps(record,indent=2)+"\n"); started=time.monotonic()
    try:
        with (EVIDENCE/(a.attempt+".log")).open("w") as log:
            run=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
        record["exit_code"]=run.returncode; record["status"]="finished"
    except subprocess.TimeoutExpired:
        record.update(exit_code=124,status="timeout")
    elapsed=time.monotonic()-started
    rseconds=0.
    for path in (EVIDENCE/a.attempt).glob("summary.json"):
        data=json.loads(path.read_text())
        rseconds=data.get("r",{}).get("wall_seconds",0)+sum(x["r"]["wall_seconds"] for x in data.get("cases",[]))
    record.update(wall_seconds=elapsed,cpu_seconds=elapsed if a.device=="cpu" else rseconds,
                  gpu_seconds=elapsed if a.device=="gpu" else 0.)
    manifest.write_text(json.dumps(record,indent=2)+"\n")
    records.append(record)
    cpu=sum(r.get("cpu_seconds",0) for r in records); gpu=sum(r.get("gpu_seconds",0) for r in records)
    (EVIDENCE/"budget.json").write_text(json.dumps(dict(prior_cpu_seconds=7458.68546110776,
        prior_gpu_seconds=296.8395259230165,phase_cpu_seconds=cpu,phase_gpu_seconds=gpu,
        remaining_cpu_seconds=165341.31453889224-cpu,remaining_gpu_seconds=172503.16047407698-gpu,
        launches=len(records),allocation_cpu_seconds=172800,allocation_gpu_seconds=172800),indent=2)+"\n")
    print(json.dumps({k:record[k] for k in ("status","exit_code","wall_seconds","cpu_seconds","gpu_seconds")}))


if __name__=="__main__": main()
