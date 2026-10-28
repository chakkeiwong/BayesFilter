"""Execute the reviewed native Stan/Walnutpie integration validation program.

GPU execution requires trusted launch permission. Outputs are always fresh.
This is a bridge-fidelity/overhead study, not a posterior-ranking campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[2]
PLAN=ROOT/"docs/plans/bayesfilter-native-stan-walnutpie-master-2026-10-28.md"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","gpu","both"],default="gpu")
    p.add_argument("--gpu",default="0")
    p.add_argument("--skip-build",action="store_true")
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    manifest={"plan":str(PLAN),"command":sys.argv,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "attempts":[],"interpretation":"exact native parity is primary; timings descriptive",
        "sampling_wall_budget_seconds":1200,"source_hashes":{}}
    snapshot=out/"source";snapshot.mkdir()
    for path in [*SOURCE.glob("*.py"),*SOURCE.glob("*.cc"),*SOURCE.glob("*.h"),PLAN]:
        shutil.copy2(path,snapshot/path.name)
        manifest["source_hashes"][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def save():
        manifest["wall_seconds"]=time.monotonic()-started
        (out/"master_manifest.json").write_text(json.dumps(manifest,indent=2))
    def run(name,script,arguments,device="cpu",limit=300):
        cmd=[sys.executable,str(SOURCE/script),*map(str,arguments)]
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]="-1" if device=="cpu" else args.gpu
        if device=="gpu":env["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
        attempt={"name":name,"command":cmd,"device":device,"timeout_seconds":limit,"log":str(out/(name+".log"))}
        manifest["attempts"].append(attempt);save();begin=time.monotonic()
        print("Running "+name,flush=True)
        try:
            with (out/(name+".log")).open("w") as log:
                result=subprocess.run(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=limit)
            attempt["returncode"]=result.returncode
            if result.returncode:raise RuntimeError("failed "+name+"; inspect "+attempt["log"])
        except Exception as error:
            attempt["error"]=repr(error);raise
        finally:
            attempt["wall_seconds"]=time.monotonic()-begin;save()
    try:
        if not args.skip_build:run("build","build.py",[],limit=600)
        else:
            hashes=json.loads((ROOT/".localresources/native_sampler_bridge/source_hashes.json").read_text())
            for name,expected in hashes.items():
                path=ROOT/name
                # Build manifest includes harnesses; additions are snapshotted.
                if path.suffix in (".cc",".hpp",".h",".so",".stan"):
                    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                        raise RuntimeError("stale native build: "+name)
        run("contract-tests","test_contract.py",[],limit=120)
        run("cpu-smoke","validate.py",["--device","cpu","--output",out/"cpu-smoke"],limit=180)
        routes=["cpu","gpu"] if args.device=="both" else [args.device]
        if "gpu" in routes:
            run("gpu-composition","test_gpu_composition.py",["--output",out/"gpu-composition"],device="gpu",limit=120)
        for route in routes:
            run(route+"-benchmark","validate.py",["--device",route,"--benchmark","--output",out/(route+"-benchmark")],device=route)
        if len(routes)==2:
            results=[json.loads((out/(r+"-benchmark")/"results.json").read_text()) for r in routes]
            a,b=[{(f["engine"],f["kind"]):f["native_output_sha256"] for f in r["fits"]} for r in results]
            if a!=b:raise RuntimeError("cross-device native-output mismatch")
            manifest["cross_device_output_identity"]=True
        run("posterior-diagnostics","diagnose.py",[out/(routes[0]+"-benchmark")],limit=120)
        manifest["status"]="completed; no sampler ranking or default promotion"
    except Exception as error:
        manifest["status"]="failed";manifest["error"]=repr(error);raise
    finally:save()
    print(json.dumps({"output":str(out),"status":manifest["status"],"seconds":manifest["wall_seconds"]}),flush=True)

if __name__=="__main__":main()
