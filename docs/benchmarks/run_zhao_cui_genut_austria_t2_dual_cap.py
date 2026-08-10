#!/usr/bin/env python3
"""Run the reviewed dual-cap T2 calibration/validation campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name(
    "run_zhao_cui_bounded_genut_austria_t2_crossed_validation.py"
)
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/"
    "zhao_cui_genut_austria_t2_dual_cap_20260806"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError("attempt must be positive")
    root = ARTIFACT_ROOT / f"attempt{args.attempt:02d}"
    teachers = ARTIFACT_ROOT.parent / (
        "zhao_cui_bounded_genut_austria_t2_crossed_validation_20260806"
    )
    command = [
        sys.executable,
        RUNNER.as_posix(),
        "--calibration-teacher-dir",
        (teachers / "teacher-calibration-n128-seeds98541-98542").as_posix(),
        "--validation-teacher-dir",
        (teachers / "teacher-validation01-n128-seeds98611-98612").as_posix(),
        "--validation-teacher-dir",
        (teachers / "teacher-validation02-n128-seeds98621-98622").as_posix(),
        "--validation-teacher-dir",
        (teachers / "teacher-validation03-n128-seeds98631-98632").as_posix(),
        "--output-root",
        root.as_posix(),
    ]
    root.parent.mkdir(parents=True, exist_ok=True)
    (root.parent / f"attempt{args.attempt:02d}-command.txt").write_text(
        " ".join(command) + "\n"
    )
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
