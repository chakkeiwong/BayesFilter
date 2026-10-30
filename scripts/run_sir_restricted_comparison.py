"""Run the planned restricted comparison after the neural study releases its GPU."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def remaining_after_attempt(neural, remaining, attempt, visited=None):
    visited = set() if visited is None else set(visited)
    attempt = attempt.resolve()
    if attempt in visited:
        raise ValueError("Cyclic retry accounting")
    visited.add(attempt)
    prior = json.loads((attempt / "ledger.json").read_text())
    if "previous_attempt" in prior:
        remaining = remaining_after_attempt(neural, remaining, Path(prior["previous_attempt"]), visited)
    if Path(prior["neural_study"]).resolve() != neural.resolve() or prior["status"] != "failed":
        raise ValueError("Retry requires a failed attempt of this neural comparison")
    if prior["allowance_seconds"] != remaining or prior["charged_seconds"] <= 0 \
            or prior["remaining_seconds"] != remaining - prior["charged_seconds"]:
        raise ValueError("Retry accounting is inconsistent")
    return prior["remaining_seconds"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=6000.)
    parser.add_argument("--repair-reserve", type=float, default=300.)
    parser.add_argument("--previous-attempt", type=Path)
    arguments = parser.parse_args()
    if arguments.cap_seconds <= 0. or arguments.repair_reserve < 0.:
        raise ValueError("Invalid comparison budget")
    repository = Path(__file__).resolve().parents[1]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    neural = arguments.neural_study.resolve()
    ledger = dict(status="waiting_for_neural_study", neural_study=str(neural),
        command=sys.argv, python=sys.executable,
        supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol="docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-01/restricted-comparator-protocol.md")
    save(output / "ledger.json", ledger)
    while not (neural / "result.json").exists():
        if (neural / "failure.json").exists():
            ledger.update(status="neural_study_failed", failure=str(neural / "failure.json"))
            save(output / "ledger.json", ledger)
            raise RuntimeError("Inspect the neural failure before launching a dependent comparison")
        time.sleep(5.)
    result = json.loads((neural / "result.json").read_text())
    if result["status"] != "complete" or not result["holdout"]["source_check"]["unchanged"]:
        raise ValueError("Neural study is not complete with unchanged source")
    remaining = result["remaining_seconds"]
    if arguments.previous_attempt is not None:
        remaining = remaining_after_attempt(neural, remaining, arguments.previous_attempt)
        ledger["previous_attempt"] = str(arguments.previous_attempt.resolve())
    cap = min(arguments.cap_seconds, remaining - arguments.repair_reserve)
    if cap <= 15.:
        raise TimeoutError("No remaining allowance for the restricted comparison")
    command = [sys.executable, "scripts/refit_sir_moment_baselines.py",
               "--neural-study", str(neural), "--output", str(output / "study"),
               "--budget-seconds", str(cap - 15.)]
    ledger.update(status="running", allowance_seconds=remaining, cap_seconds=cap,
        repair_reserve=arguments.repair_reserve, child_command=command,
        start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save(output / "ledger.json", ledger)
    begin = time.monotonic()
    with (output / "study.log").open("w") as log:
        process = subprocess.Popen(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        ledger["pid"] = process.pid
        save(output / "ledger.json", ledger)
        try:
            returncode = process.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
    elapsed = time.monotonic() - begin
    charge = math.ceil(elapsed)
    ledger.update(status="complete" if returncode == 0 else "failed", returncode=returncode,
                  wall_seconds=elapsed, charged_seconds=charge, remaining_seconds=remaining - charge)
    save(output / "ledger.json", ledger)
    print(json.dumps(ledger), flush=True)
    if returncode != 0:
        raise RuntimeError("Restricted comparison failed; inspect the preserved study.log")


if __name__ == "__main__":
    main()
