"""Execute the resolved M8 suites sequentially on the authorized GPU.

This coordinator never edits source, changes scientific criteria, or replaces
failed fits. The validation executor owns per-worker hard time limits.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"


def read(path):
    return json.loads(path.read_text())


def audit_suite(root, *, expected_preparation_failure=False):
    index = read(root/"run_index.json")
    elapsed = 0.
    rows = []
    for planned in index["plan"]["jobs"]:
        design = planned["design"]
        job = index["jobs"].get(design["design_id"], {})
        elapsed += sum(a.get("elapsed_seconds", 0.) for a in job.get("attempts", []))
        if job.get("status") == "complete":
            result_path = Path(job["result"])
            assert hashlib.sha256(result_path.read_bytes()).hexdigest() == job["result_sha256"]
            result = read(result_path)
            runtime = result["runtime"]
            assert runtime["device_scope"] == "gpu" and runtime["jit_compile"]
            assert runtime["memory_policy"]["all_physical_devices_memory_growth"]
            assert runtime["memory_policy"]["configured_before_logical_device_initialization"]
            rows.append((design, job["attempts"][-1]["elapsed_seconds"]))
        elif expected_preparation_failure and job.get("status") == "failed":
            failure = read(root/design["design_id"]/"attempt-001-failure.json")
            if "HMCPreparationFailure" not in json.dumps(failure):
                raise RuntimeError("unexpected funnel failure: " + str(failure))
            # Expected candidate/preparation rejection remains in the denominator.
            rows.append((design, job["attempts"][-1]["elapsed_seconds"]))
        else:
            raise RuntimeError("incomplete or invalid launch: " + design["design_id"])
    return elapsed, rows


def run(name, source, *, expected_preparation_failure=False):
    output = ROOT/(name+"-r1")
    suite = source/"suites"/("m8-"+name+".json")
    if output.exists():
        raise FileExistsError(output)
    command = [sys.executable,"-m","bayesfilter.testing.inference_validation","run",
               str(suite),"--output",str(output),"--max-workers","1"]
    started = time.monotonic()
    when = datetime.now(timezone.utc).isoformat()
    print("START", name, when, flush=True)
    with (ROOT/(name+"-coordinator.log")).open("x") as log:
        result = subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT)
    record = {"command":command,"cwd":str(source),"started_utc":when,
        "coordinator_wall_seconds":time.monotonic()-started,"returncode":result.returncode,
        "device":"gpu","plan_file":PLAN,"result":str(output/"run_index.json"),
        "accounting":"charge run_index worker attempts; do not double count this coordinator"}
    (ROOT/(name+"-coordinator.json")).write_text(json.dumps(record,indent=2)+"\n")
    cost, rows = audit_suite(output,expected_preparation_failure=expected_preparation_failure)
    print("FINISHED",name,"worker_seconds",cost,flush=True)
    return cost, rows


def affordable(rows, multiplier, limit):
    projection = sum(seconds for _,seconds in rows)*multiplier*1.5
    if projection > limit:
        raise RuntimeError(f"pilot projection {projection} exceeds remaining reservation {limit}")
    return projection


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    source = ROOT/"source-gpu-r4"
    screen_cost,screen_rows = audit_suite(ROOT/"screen-pilot-gpu-r1")
    affordable(screen_rows,16.,18000.-screen_cost)
    funnel_cost,funnel_rows = run("funnel-pilot-gpu",source,expected_preparation_failure=True)
    stopping_cost,stopping_rows = run("stopping-pilot-gpu",source)
    affordable(funnel_rows,3.,12000.-funnel_cost-1800.)
    affordable(stopping_rows,32.,36000.-stopping_cost)
    run("funnel-fresh-gpu",source,expected_preparation_failure=True)
    run("screen-fresh-gpu",ROOT/"source-gpu-r3")
    power_cost,power_rows = run("power-16k-pilot-gpu",source)
    affordable(power_rows,8.,6000.-power_cost)
    run("power-16k-confirmation-gpu",source)
    run("stopping-fresh-gpu",source)
    print("ALL RESOLVED SUITES FINISHED",flush=True)


if __name__ == "__main__":
    main()
