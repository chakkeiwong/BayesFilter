"""Fresh d2 diagnostic using the shared consumer and full contemporaneous traces."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT="docs/plans/artifacts/iapf-tf32-recorded-paths-20260923-01"
PLAN="docs/plans/iapf-tf32-recorded-paths-2026-09-23.md"
SCRIPT="docs/benchmarks/diagnose_iapf_tf32_recorded_paths.py"
BASE="docs/benchmarks/diagnose_iapf_tf32_revalidation.py"
REPORT="docs/benchmarks/summarize_iapf_tf32_revalidation.R"
PYTHON="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU="GPU-68251639-fe82-8f81-3ccc-2953c32e805b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def supervise():
    started=time.monotonic()
    repo=Path(__file__).resolve().parents[2];root=repo/ROOT;root.mkdir(exist_ok=True)
    previous=repo/"docs/plans/artifacts/iapf-tf32-revalidation-20260923-01"
    prior=json.loads((previous/"completed.json").read_text())
    opening=json.loads((repo/"docs/plans/artifacts/iapf-tf32-tail-replay-20260923-01/budget.json").read_text())
    ledger=root/"budget.json"
    budget=json.loads(ledger.read_text()) if ledger.exists() else dict(
        cpu_charged_seconds=0.,gpu_charged_seconds=0.,cpu_reserved_seconds=0.,gpu_reserved_seconds=0.,attempts=0,
        phase_cpu_cap_seconds=120.,phase_gpu_cap_seconds=1800.,
        **{k:v for k,v in opening.items() if k.startswith("remaining_campaign_")})
    assert budget["cpu_reserved_seconds"]==budget["gpu_reserved_seconds"]==0
    assert shutil.disk_usage(root).free>4*1024**3,"Less than4GiB free for diagnostic traces"
    fixture=Path(prior["fixtures"])
    for name,digest in prior["fixture_hashes"].items():assert sha(fixture/name)==digest
    source=root/f"source-{len(list(root.glob('source-*')))+1:02d}";source.mkdir()
    files=[SCRIPT,BASE,REPORT,PLAN,"bayesfilter/__init__.py","bayesfilter/runtime/__init__.py",
           "bayesfilter/runtime/gpu_memory_policy.py"]
    files += [str(x.relative_to(repo)) for x in (repo/"bayesfilter/score_study").glob("*.py")]
    old=json.loads((Path(prior["fp64"]).parent/"manifest.json").read_text())
    for name in files:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,path)
        if name.startswith("bayesfilter/"):assert sha(path)==old["source_sha256"][name]
    hashes={name:sha(source/name) for name in files}
    git=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()
    env=dict(os.environ,PYTHONPATH=str(source),TF_FORCE_GPU_ALLOW_GROWTH="true",BAYESFILTER_PRELOAD_CUSTOM_OP="0",
        OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="2")
    if env.get("XLA_FLAGS",""):raise RuntimeError("Unexpected compiler override")
    completed_path=root/"completed.json"
    completed=json.loads(completed_path.read_text()) if completed_path.exists() else dict(fixtures=str(fixture))
    gpu_open=budget["gpu_charged_seconds"];cpu_open=budget["cpu_charged_seconds"]

    def run(label,command,gpu,cap):
        kind="gpu" if gpu else "cpu"
        assert budget["attempts"]<6
        assert budget[f"{kind}_charged_seconds"]+cap<=budget[f"phase_{kind}_cap_seconds"]
        assert cap<=budget[f"remaining_campaign_{kind}_seconds"]
        budget["attempts"]+=1;budget[f"{kind}_reserved_seconds"]=cap;dump(ledger,budget)
        job=root/f"attempt{budget['attempts']:02d}-{label}";job.mkdir()
        command=[str(x).replace("{OUT}",str(job/"outputs")) for x in command]
        runenv=dict(env,CUDA_VISIBLE_DEVICES=GPU if gpu else "-1")
        manifest=dict(status="running",command=command,git_commit=git,source=str(source),source_sha256=hashes,
            plan=PLAN,result_file=str(root/"result.md"),fixture=str(fixture),fixture_sha256=prior["fixture_hashes"],
            data_version="phase18 fixed d2 datasets, fresh1024 particle streams",
            seeds="cells.csv particle_seed+10000000+replicate[0,1023]",cpu_only=not gpu,gpu_intentionally_hidden=not gpu,
            environment={k:runenv.get(k,"") for k in ("CUDA_VISIBLE_DEVICES","TF_FORCE_GPU_ALLOW_GROWTH","OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS","MKL_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","XLA_FLAGS")},
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),cap_seconds=cap)
        dump(job/"manifest.json",manifest);t=time.monotonic()
        try:
            with (job/"command.log").open("w") as log:
                r=subprocess.run(command,cwd=source,env=runenv,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
            manifest.update(status="complete" if r.returncode==0 else "failed",returncode=r.returncode)
        except BaseException as error:manifest.update(status="failed",error=repr(error))
        finally:
            seconds=time.monotonic()-t;manifest["wall_seconds"]=seconds
            budget[f"{kind}_reserved_seconds"]=0.;budget[f"{kind}_charged_seconds"]+=seconds
            budget[f"remaining_campaign_{kind}_seconds"]-=seconds
            manifest["output_sha256"]={str(x.relative_to(job)):sha(x) for x in job.rglob("*") if x.is_file() and x.name!="manifest.json"}
            dump(job/"manifest.json",manifest);dump(ledger,budget)
        print(json.dumps(dict(label=label,status=manifest["status"],seconds=seconds)),flush=True)
        if manifest["status"]!="complete":raise RuntimeError(f"Repair required at {job}")
        return str(job/"outputs")

    try:
        for mode in ("fp64","fp32","tf32"):
            if mode not in completed:
                completed[mode]=run(mode,[PYTHON,source/SCRIPT,"--mode",mode,"--fixture",fixture,"--output","{OUT}"],True,600)
                dump(completed_path,completed)
        if "summary" not in completed:
            completed["summary"]=run("summary",["/usr/bin/Rscript","--vanilla",source/REPORT,fixture,
                completed["fp64"],completed["fp32"],completed["tf32"],"{OUT}","12","1024"],False,30)
            dump(completed_path,completed)
    finally:
        overhead=max(0.,time.monotonic()-started-(budget["gpu_charged_seconds"]-gpu_open)-(budget["cpu_charged_seconds"]-cpu_open))
        budget["cpu_charged_seconds"]+=overhead;budget["remaining_campaign_cpu_seconds"]-=overhead;dump(ledger,budget)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--execute",action="store_true")
    p.add_argument("--mode",choices=("fp64","fp32","tf32"));p.add_argument("--fixture",type=Path);p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.execute:
        os.environ["CUDA_VISIBLE_DEVICES"]="-1";supervise()
    else:
        if not a.mode or not a.fixture or not a.output:p.error("worker requires mode, fixture and output")
        spec=importlib.util.spec_from_file_location("phase18_diagnostic",Path(__file__).with_name(Path(BASE).name))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.worker(a.mode,a.fixture,a.output,replicates=1024,dimensions=(2,),particle_seed_offset=10000000,
            trace_all=True,compress_traces=True)
