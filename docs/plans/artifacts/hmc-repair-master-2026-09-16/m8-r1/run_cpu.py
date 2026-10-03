"""Bounded CPU-only checks for the M7 continuation; writes measured attempts."""
import argparse, datetime, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--seconds", type=float, default=600.)
p.add_argument("command", nargs=argparse.REMAINDER)
a = p.parse_args()
command = a.command[1:] if a.command and a.command[0] == "--" else a.command
if not command: raise ValueError("command required")
record = ROOT/(a.label+"-run.json")
if record.exists(): raise FileExistsError(record)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_FORCE_GPU_ALLOW_GROWTH="true", TF_CPP_MIN_LOG_LEVEL="2", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OPENBLAS_NUM_THREADS="1")
started = time.monotonic()
when = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (ROOT/(a.label+".log")).open("x") as log:
    try:
        completed = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.seconds)
        code = completed.returncode
    except subprocess.TimeoutExpired:
        code = 124
elapsed = time.monotonic()-started
payload = {"command":command,"cwd":str(REPO),"started_utc":when,"elapsed_seconds":elapsed,"returncode":code,"device":"cpu_reference","gpu_intentionally_hidden":True,"environment":sys.executable,"plan_file":"docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),"log":str(ROOT/(a.label+".log"))}
record.write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
print((ROOT/(a.label+".log")).read_text()[-6000:])
sys.exit(code)
