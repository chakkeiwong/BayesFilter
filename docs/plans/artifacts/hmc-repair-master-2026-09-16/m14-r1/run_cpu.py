"""Metered CPU reference checks for the bounded M14 stage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--seconds", type=float, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
root = Path(__file__).resolve().parent
record_path = root / (args.label + "-run.json")
if record_path.exists():
    raise FileExistsError(record_path)
spent = sum(json.loads(p.read_text())["elapsed_seconds"] for p in root.glob("*-run.json"))
if spent + args.seconds + 900 > 12000:
    raise RuntimeError("M14 CPU allowance exhausted")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2",
           TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1")
started = time.monotonic()
when = datetime.now(timezone.utc).isoformat()
with (root / (args.label + ".log")).open("x") as log:
    try:
        code = subprocess.run(args.command, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=args.seconds).returncode
    except subprocess.TimeoutExpired:
        code = 124
record = dict(command=args.command, started_utc=when, elapsed_seconds=time.monotonic()-started,
    returncode=code, gpu_intentionally_hidden=True, device="cpu_reference", environment=sys.executable,
    plan_file="docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
record_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
sys.exit(code)
