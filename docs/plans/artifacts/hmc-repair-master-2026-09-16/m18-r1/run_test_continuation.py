"""Continue the bounded audit, preserving per-test outcomes and child cleanup."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 2400:
        raise ValueError("outside the M18 test allowance")
    completed = set()
    for path in ROOT.glob("affected-tests-*.xml"):
        for case in ET.parse(path).getroot().iter("testcase"):
            if case.attrib.get("classname") and not any(c.tag in {"failure", "error", "skipped"} for c in case):
                completed.add(case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"])
    for path in ROOT.glob("affected-tests-*-reports.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["when"] == "call" and row["outcome"] == "passed":
                completed.add(row["nodeid"])
    completed_path = ROOT / (args.label + "-completed.json")
    with completed_path.open("x") as handle:
        json.dump(sorted(completed), handle, indent=2)
    sources = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((REPO / "bayesfilter").rglob("*.py"))}
    command = [sys.executable, str(ROOT / "run_affected_tests.py"), "--label", args.label]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1",
               HMC_TEST_REPORTS=str(ROOT / (args.label + "-reports.jsonl")),
               HMC_TEST_COMPLETED=str(completed_path),
               PYTHONPATH=str(ROOT) + os.pathsep + str(REPO),
               PYTEST_ADDOPTS="-p stream_pytest")
    started, when = time.monotonic(), datetime.now(timezone.utc).isoformat()
    capped = False
    with (ROOT / (args.label + ".log")).open("x") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=args.seconds - 15)
        except subprocess.TimeoutExpired:
            capped = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    record = {"command": command, "started_utc": when, "elapsed_seconds": time.monotonic()-started,
              "returncode": code, "capped": capped, "device": "cpu_reference",
              "gpu_intentionally_hidden": True, "environment": sys.executable,
              "plan_file": "docs/plans/bayesfilter-hmc-repair-m18-design-2026-09-21.md",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_files_at_launch": sources, "completed_nodes_excluded": len(completed)}
    with (ROOT / (args.label + "-run.json")).open("x") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != "source_files_at_launch"}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
