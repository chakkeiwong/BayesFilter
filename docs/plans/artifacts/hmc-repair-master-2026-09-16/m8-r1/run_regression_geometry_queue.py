"""Bounded sequential pilot/confirmation, using unused matched-reference funds."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
RESERVATION = 16000.


def read(path):
    return json.loads(path.read_text())


def spent():
    return sum(read(p)["elapsed_seconds"] for p in ROOT.glob("posteriordb-*-gpu-r1/gpu-diagnostic-run.json"))


def run(phase, replication, cap):
    command = [sys.executable, str(ROOT/"run_posteriordb_geometry.py"),
        "--source", str(ROOT/"source-gpu-r6"), "--case", "sblrc-blr",
        "--phase", phase, "--replication", str(replication), "--seconds", str(cap)]
    result = subprocess.run(command)
    output = ROOT/f"posteriordb-geometry-hint-regression-{phase}-{replication}-gpu-r1"
    if result.returncode:
        failure = read(output/"failure.json") if (output/"failure.json").exists() else {}
        if failure.get("exception") != "HMCPreparationFailure":
            raise RuntimeError("unresolved geometry diagnostic failure: " + str(output))
    else:
        read(output/"assessment.json")
    return read(output/"gpu-diagnostic-run.json")["elapsed_seconds"]


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    assert read(ROOT/"regression-geometry-check.json")["passed"]
    opening = spent()
    if opening + 4*1800 > RESERVATION:
        raise RuntimeError("four diagnostic ceilings exceed the reference reservation")
    pilot = run("pilot", 0, 1800)
    cap = max(1800, math.ceil(1.5*pilot/100)*100)
    record = {"opening_seconds": opening, "pilot_seconds": pilot, "fresh_cap_seconds": cap,
        "worst_case_total_seconds": spent()+3*cap, "reservation_seconds": RESERVATION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"}
    with (ROOT/"regression-geometry-pilot-cost-review.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    if record["worst_case_total_seconds"] > RESERVATION:
        raise RuntimeError("measured pilot requires resource revision before confirmation")
    for replication in range(3):
        run("fresh", replication, cap)
    print("REGRESSION GEOMETRY QUEUE COMPLETE; reference seconds", spent(), flush=True)


if __name__ == "__main__":
    main()
