"""Metered CPU-only M10 checks, with a hard timeout and preserved logs."""
import argparse, datetime, json, os, pathlib, subprocess, sys, time
p=argparse.ArgumentParser(); p.add_argument("--label",required=True); p.add_argument("--seconds",type=float,required=True); p.add_argument("command",nargs=argparse.REMAINDER); a=p.parse_args()
r=pathlib.Path(__file__).resolve().parent
assert not (r/(a.label+"-run.json")).exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",TF_CPP_MIN_LOG_LEVEL="2",TF_NUM_INTRAOP_THREADS="2",TF_NUM_INTEROP_THREADS="1")
start=time.monotonic(); when=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/(a.label+".log")).open("x") as log:
 try: code=subprocess.run(a.command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.seconds).returncode
 except subprocess.TimeoutExpired: code=124
record=dict(command=a.command,elapsed_seconds=time.monotonic()-start,started_utc=when,returncode=code,gpu_intentionally_hidden=True,device="cpu_reference",environment=sys.executable,plan_file="docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md")
(r/(a.label+"-run.json")).write_text(json.dumps(record,indent=2)+"\n"); print(json.dumps(record)); sys.exit(code)
