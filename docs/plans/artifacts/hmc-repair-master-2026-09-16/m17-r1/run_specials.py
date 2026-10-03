"""Execute the predeclared M17 field and matched-reference cells, two at a time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "m16-r1/source-r1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("gpu", "cpu_reference"), required=True)
    args = parser.parse_args()
    if args.device == "gpu":
        assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    jobs = []
    for case in ("gaussian", "beta_binomial", *(("lgssm_location", "banana") if args.device == "cpu_reference" else ())):
        jobs.append([sys.executable, str(ROOT / "run_position_field.py"), "--source", str(SOURCE),
                     "--device", args.device, "--case", case, "--seconds", "200" if args.device == "gpu" else "100"])
    for case in ("sblrc-blr", "eight_schools-eight_schools_noncentered"):
        jobs.append([sys.executable, str(ROOT / "run_posteriordb.py"), "--source", str(SOURCE),
                     "--device", args.device, "--case", case, "--phase", "fresh", "--replication", "0",
                     "--seconds", "1000" if args.device == "gpu" else "900"])
    def run(command):
        code = subprocess.run(command).returncode
        return {"command": command, "returncode": code}
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, jobs))
    with (ROOT / ("specials-" + args.device + "-r1.json")).open("x") as handle:
        json.dump(records, handle, indent=2)


if __name__ == "__main__":
    main()
