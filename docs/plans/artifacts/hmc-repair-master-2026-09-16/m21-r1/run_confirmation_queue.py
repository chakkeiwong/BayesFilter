"""Execute frozen independent studies without exceeding two numerical workers."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
NEXT = ROOT.parent / "m22-r1"
TRAINING = ROOT.parent / "m23-r1"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") == "true"
queue_path = ROOT / "confirmation-queue.json"
queue = json.loads(queue_path.read_text())
assert queue["max_workers"] == 2
state = {"queue_sha256": hashlib.sha256(queue_path.read_bytes()).hexdigest(),
    "status": "waiting_for_pilots_and_engineering_checks", "tasks": {},
    "reserved_cpu_seconds": sum(r["worker_seconds"] for r in queue["tasks"]),
    "gpu_seconds": 0, "pid": os.getpid()}
progress = ROOT / "confirmation-queue-progress.json"
lock = threading.Lock()

def save():
    temporary = progress.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(progress)

save()
waited = time.monotonic()
while not ((NEXT / "pilot-coordinator-complete.json").exists()
           and (TRAINING / "engineering-tests-r1-attempt.json").exists()):
    if time.monotonic() - waited > 3600:
        raise TimeoutError("preceding work did not complete within the queue ceiling")
    time.sleep(5)
assert json.loads((NEXT / "pilot-coordinator-complete.json").read_text())["returncode"] == 0
assert json.loads((TRAINING / "engineering-tests-r1-attempt.json").read_text())["returncode"] == 0
state.update(status="running", started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()

def execute(task):
    output = Path(task["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "bayesfilter.testing.inference_validation", "run", task["suite"],
               "--output", task["output"], "--max-workers", "1"]
    start = time.monotonic()
    with lock:
        state["tasks"][task["name"]] = {"status": "running", "phase": task["phase"],
            "command": command, "source": task["source"], "output": task["output"]}
        save()
    with (ROOT / (task["name"] + "-coordinator.log")).open("x") as log:
        # The suite enforces the worker budget and kills expired workers.
        code = subprocess.run(command, cwd=task["source"], stdout=log, stderr=subprocess.STDOUT).returncode
    index_path = output / "run_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    charges = [a["elapsed_seconds"] for j in index.get("jobs", {}).values() for a in j.get("attempts", ())]
    with lock:
        state["tasks"][task["name"]].update(status="complete" if code == 0 else "needs_repair",
            returncode=code, coordinator_elapsed_seconds=time.monotonic() - start,
            cpu_worker_seconds=sum(charges) if charges else None,
            worker_charges_replace_coordinator_time=True,
            index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest() if index_path.exists() else None)
        save()
    print(json.dumps({"task": task["name"], **state["tasks"][task["name"]]}), flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(execute, task) for task in queue["tasks"]]
    for future in as_completed(futures):
        future.result()
state.update(status="execution_finished_review_required", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             scientific_gaps_closed=False)
save()
