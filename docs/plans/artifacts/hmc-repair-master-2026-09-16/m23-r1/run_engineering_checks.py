"""Run the reviewed training checks once a numerical worker slot is free."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
index = ROOT.parent / "m22-r1/full-fit-pilot-cpu-r1/run_index.json"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
waited = time.monotonic()
while True:
    jobs = json.loads(index.read_text())["jobs"] if index.exists() else {}
    if len(jobs) == 3 and all(jobs["m22-full-fit-pilot-" + arm].get("attempts")
                            and jobs["m22-full-fit-pilot-" + arm]["status"] != "running"
                            for arm in ("baseline", "noop")):
        break
    if time.monotonic() - waited > 3000:
        raise TimeoutError("preceding pilots did not release a worker")
    time.sleep(5)
command = [sys.executable, "-m", "pytest", "-q",
    "tests/test_neutra_training_graph_contract.py",
    "tests/test_neutra_reverse_kl_training.py",
    "tests/test_neutra_quadratic_anchor.py",
    "tests/test_neutra_whitening.py",
    "tests/test_neutra_dsge_procedure_parity.py",
    "--junitxml=" + str(ROOT / "engineering-tests-r1.xml")]
started = time.monotonic()
with (ROOT / "engineering-tests-r1.log").open("x") as log:
    try:
        code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, timeout=580).returncode
    except subprocess.TimeoutExpired:
        code = 124
record = {"command": command, "returncode": code, "elapsed_seconds": time.monotonic() - started,
    "device": "cpu_reference", "gpu_intentionally_hidden": True,
    "plan_file": "docs/plans/bayesfilter-hmc-repair-m23-training-engineering-2026-09-22.md",
    "sources": {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in (
        "bayesfilter/inference/neutra_training.py", "bayesfilter/inference/neutra_training_graphs.py",
        "tests/test_neutra_training_graph_contract.py")}}
(ROOT / "engineering-tests-r1-attempt.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
raise SystemExit(code)
