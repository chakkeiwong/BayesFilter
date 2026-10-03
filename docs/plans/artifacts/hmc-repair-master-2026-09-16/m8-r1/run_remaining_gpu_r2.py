"""Continue the unchanged scientific M8 queue after the measured resource repair."""
import math
import os
from pathlib import Path
import subprocess
import sys

from run_remaining_gpu import ROOT,audit_suite,run,read


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower() == "true"
    stopping_cost,_ = audit_suite(ROOT/"stopping-pilot-gpu-r1")
    assert stopping_cost*32*1.5 < 70000-stopping_cost
    source = ROOT/"source-gpu-r4"
    subprocess.run([sys.executable,str(ROOT/"extend_saved_funnel_evidence.py")],check=True)
    power_cost,power_rows = run("power-16k-pilot-gpu",source)
    assert sum(seconds for _,seconds in power_rows)*8*1.5 < 6000-power_cost
    run("power-16k-confirmation-gpu",source)
    run("funnel-fresh-gpu",source,expected_preparation_failure=True)
    run("screen-fresh-gpu",ROOT/"source-gpu-r3")
    cases = ("eight_schools-eight_schools_noncentered","sblrc-blr")
    pilot_costs = []
    for case in cases:
        command = [sys.executable,str(ROOT/"run_posteriordb.py"),"--source",str(ROOT/"source-gpu-r6"),
            "--case",case,"--phase","pilot","--replication","0","--seconds","1800"]
        result = subprocess.run(command)
        tag = "schools" if case.startswith("eight_schools") else "regression"
        output = ROOT/f"posteriordb-{tag}-pilot-0-gpu-r1"
        record = read(output/"gpu-diagnostic-run.json")
        if result.returncode:
            failure = read(output/"failure.json") if (output/"failure.json").exists() else {}
            if failure.get("exception") != "HMCPreparationFailure":
                raise RuntimeError("invalid matched-reference pilot: "+str(failure))
        else:
            read(output/"assessment.json")
        pilot_costs.append(record["elapsed_seconds"])
    remaining = 16000-sum(pilot_costs)
    assert sum(pilot_costs)*3*1.5 < remaining
    # Measured pilot costs plus 50% margin, with a 600-second minimum for a
    # preparation-rejected pilot. This bounds resources, not statistical power.
    caps = [max(600,math.ceil(seconds*1.5/100)*100) for seconds in pilot_costs]
    assert sum(caps)*3 <= remaining
    for case,seconds in zip(cases,caps):
        for replication in range(3):
            result = subprocess.run([sys.executable,str(ROOT/"run_posteriordb.py"),"--source",str(ROOT/"source-gpu-r6"),
                "--case",case,"--phase","fresh","--replication",str(replication),"--seconds",str(seconds)])
            if result.returncode:
                tag = "schools" if case.startswith("eight_schools") else "regression"
                output = ROOT/f"posteriordb-{tag}-fresh-{replication}-gpu-r1"
                failure = read(output/"failure.json") if (output/"failure.json").exists() else {}
                if failure.get("exception") != "HMCPreparationFailure":
                    raise RuntimeError("invalid matched-reference fit: "+str(failure))
    run("stopping-fresh-gpu",ROOT/"source-gpu-r6")
    print("M8 RESOLVED NUMERICAL QUEUE COMPLETE",flush=True)


if __name__ == "__main__":
    main()
