"""Execute the three predeclared M9 fits sequentially within their recorded caps."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    pilot = json.loads((ROOT/"regression-pilot-0-gpu-r1/gpu-diagnostic-run.json").read_text())
    if pilot["returncode"] or pilot["elapsed_seconds"] * 1.5 > 1500.:
        raise RuntimeError("pilot has not completed within the predeclared margin")
    for replication in range(3):
        command = [sys.executable, str(ROOT/"run_regression_combined.py"), "--source", str(ROOT/"source-gpu-r2"),
                   "--case", "sblrc-blr", "--phase", "fresh", "--replication", str(replication), "--seconds", "1500"]
        result = subprocess.run(command)
        if result.returncode:
            raise RuntimeError("fresh fit requires failure diagnosis: " + str(replication))
        record = json.loads((ROOT/f"regression-combined-fresh-{replication}-gpu-r1/assessment.json").read_text())
        print(json.dumps({"replication": replication, "verified": record["verified_members"],
                          "posterior_available": record["posterior_available"],
                          "full_screen_passed": record["accuracy_screen_passed"]}), flush=True)


if __name__ == "__main__":
    main()
