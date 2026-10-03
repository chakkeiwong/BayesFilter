"""The four predeclared M18 GPU arms, at most two concurrent workers."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
jobs = [[sys.executable, str(ROOT / "run_hash_parity.py"), "--source", str(ROOT / "source-r2"),
         "--case", case, "--arm", arm, "--seconds", "400"]
        for case in ("gaussian", "beta_binomial") for arm in ("generic", "native")]


def run(command):
    code = subprocess.run(command).returncode
    return {"command": command, "returncode": code}


with ThreadPoolExecutor(max_workers=2) as pool:
    records = list(pool.map(run, jobs))
with (ROOT / "parity-matrix-r1.json").open("x") as handle:
    json.dump(records, handle, indent=2)
raise SystemExit(int(any(r["returncode"] for r in records)))
