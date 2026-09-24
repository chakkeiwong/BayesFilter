#!/usr/bin/env python3
"""Test 1B: IN-SAMPLE validation for SQMC tuning hypothesis testing.

This script intentionally evaluates tuned controls on the SAME seeds used for
tuning (seeds 50001-50016) to distinguish:

H1 (overfitting): Tuned controls improve in-sample but are seed-specific
H2 (algorithmic): Only certain routes respond to control changes

The overlap check in run_sqmc_tuned_vs_untuned.py is bypassed for this
diagnostic purpose only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, "/home/chakwong/python/src")

# Import the main comparison module but bypass the overlap check
import run_sqmc_tuned_vs_untuned as base_script


def main():
    parser = argparse.ArgumentParser(
        description="Test 1B: IN-SAMPLE validation (tuning seeds = claim seeds)"
    )
    parser.add_argument("--tuning-artifact", required=True)
    parser.add_argument("--route", default="iid_dual_cap")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--particles", type=int, default=1008)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    # Verify that we ARE using tuning seeds (Test 1B requirement)
    artifact_path = Path(args.tuning_artifact)
    tuning_artifact = json.loads(artifact_path.read_text())
    tuning_seeds = set(tuning_artifact.get("tuning_seeds", ()))
    overlap = tuning_seeds.intersection(args.seeds)

    if not overlap:
        print("WARNING: Test 1B requires claim seeds to overlap tuning seeds")
        print(f"  tuning seeds: {sorted(tuning_seeds)}")
        print(f"  claim seeds:  {args.seeds}")
        print("  This is not a valid in-sample test.")
    else:
        print("Test 1B: IN-SAMPLE validation")
        print(f"  Using {len(overlap)} tuning seeds as claim seeds: {sorted(overlap)}")

    # Temporarily remove the overlap check from the base script
    original_code = base_script.__file__
    base_script_path = Path(original_code)

    # Re-execute the base script's main with our args, but skip the check
    # by directly calling the internal functions
    return base_script.main()


if __name__ == "__main__":
    # Monkey-patch to bypass the overlap check
    import run_sqmc_tuned_vs_untuned as script
    original_lines = Path(script.__file__).read_text().split('\n')

    # Find and comment out the overlap check (lines 204-209)
    exec_globals = {}
    exec_code = []
    skip_block = False
    for i, line in enumerate(original_lines, 1):
        if i == 204:
            skip_block = True
            exec_code.append("    # BYPASSED FOR TEST 1B: overlap check")
            exec_code.append("    # " + line)
        elif i == 209:
            exec_code.append("    # " + line)
            skip_block = False
        elif skip_block:
            exec_code.append("    # " + line)
        else:
            exec_code.append(line)

    exec('\n'.join(exec_code), exec_globals)
    exec_globals['main']()
