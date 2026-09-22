"""Finite M20 diagnostic inventory; preserves attempts and meters worker time."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu_reference", "gpu"), required=True)
    parser.add_argument("--label", default="r1")
    parser.add_argument("--cases", nargs="+", default=["saved_funnel", "fresh_funnel", "rotated", "noncentered"])
    args = parser.parse_args()
    if args.device == "gpu":
        assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "", "-1")
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert len(set(args.cases)) == len(args.cases)
    assert set(args.cases) <= {"saved_funnel", "fresh_funnel", "rotated", "noncentered"}
    destination = ROOT / f"{args.device}-{args.label}"
    destination.mkdir(exist_ok=False)
    def run(case):
        seconds = 1800 if args.device == "gpu" else 1200
        source = ROOT.parent / "m16-r1/source-r1" if case == "saved_funnel" else ROOT / "source-r1"
        command = [sys.executable, str(ROOT / "run_geometry_probe.py"), "--source", str(source),
            "--device", args.device, "--case", case, "--seconds", str(seconds),
            "--seed", "2026092201" if case == "saved_funnel" else "2026092202",
            "--output", str(destination / case)]
        started = time.monotonic()
        with (destination / (case + ".log")).open("x") as log:
            try:
                outcome = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=seconds)
                code = outcome.returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = {"command": command, "device": args.device, "returncode": code,
            "elapsed_seconds": time.monotonic()-started, "case": case,
            "gpu_intentionally_hidden": args.device != "gpu",
            "plan_file": "docs/plans/bayesfilter-hmc-repair-m20-design-2026-09-22.md"}
        with (destination / (case + "-attempt.json")).open("x") as handle:
            json.dump(record, handle, indent=2);handle.write("\n")
        print(json.dumps(record), flush=True)
        return record
    with ThreadPoolExecutor(max_workers=1 if args.device == "gpu" else 2) as pool:
        records = list(pool.map(run, args.cases))
    with (destination / "execution.json").open("x") as handle:
        json.dump({"attempts": records, "charged_worker_seconds": sum(r["elapsed_seconds"] for r in records)}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
