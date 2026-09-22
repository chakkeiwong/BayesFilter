"""Queue audited M21 reporting and M22 pilots with a shared two-worker ceiling."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "m21-r1"
REPO = ROOT.parents[4]
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
wait_start = time.monotonic()
while not (PREVIOUS / "queued-controls-execution.json").is_file():
    if time.monotonic() - wait_start > 1800:
        raise TimeoutError("preceding public-control suite did not finish")
    time.sleep(5)
previous = json.loads((PREVIOUS / "queued-controls-execution.json").read_text())
assert all(r["returncode"] == 0 for r in previous["steps"]), "inspect previous failure first"

def run(name, command, cwd, seconds, accounting):
    start = time.monotonic()
    with (ROOT / (name + ".log")).open("x") as log:
        try:
            code = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=seconds).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = {"command": command, "cwd": str(cwd), "elapsed_seconds": time.monotonic() - start,
           "returncode": code, "timeout_seconds": seconds, "accounting_phase": accounting,
           "device": "cpu_reference", "suite_worker_charges_replace_coordinator_time": "-m" in command and "run" in command}
    (ROOT / (name + "-attempt.json")).write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({"attempt": name, **row}), flush=True)
    return code

assert run("m21-report-tests-r1", [sys.executable, "-m", "pytest", "-q",
    "tests/inference_validation/test_controller_stopping_report.py",
    "tests/inference_validation/test_ar1_batch_reference.py",
    "--junitxml=" + str(ROOT / "m21-report-tests-r1.xml")], REPO, 120, "M21") == 0
assert run("m21-report-r1", [sys.executable, str(PREVIOUS / "summarize_controller.py")], REPO, 120, "M21") == 0

with ThreadPoolExecutor(max_workers=2) as pool:
    diagnostic = pool.submit(run, "m21-estimator-diagnosis-r1",
        [sys.executable, str(PREVIOUS / "diagnose_estimator.py")], REPO, 870, "M21")
    null = pool.submit(run, "null-pilot-r1", [sys.executable, "-m",
        "bayesfilter.testing.inference_validation", "run", str(ROOT / "null-pilot.json"),
        "--output", str(ROOT / "null-pilot-cpu-r1"), "--max-workers", "1"],
        ROOT / "source-r1", 240, "M22")
    codes = (diagnostic.result(), null.result())
assert codes == (0, 0), "inspect diagnostic or null pilot failure before further execution"
code = run("full-fit-pilot-r1", [sys.executable, "-m", "bayesfilter.testing.inference_validation",
    "run", str(ROOT / "full-fit-pilot.json"), "--output", str(ROOT / "full-fit-pilot-cpu-r1"),
    "--max-workers", "2"], ROOT / "source-r1", 2700, "M22")
(ROOT / "pilot-coordinator-complete.json").write_text(json.dumps({"returncode": code}) + "\n")
raise SystemExit(code)
