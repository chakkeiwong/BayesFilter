"""Execute the three predeclared M11 fresh fits sequentially, preserving failures."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    started = time.monotonic()
    rows = []
    for replication in range(3):
        command = [sys.executable, str(ROOT / "run_regression.py"), "--source",
                   str(ROOT / "source-gpu-r2"), "--case", "sblrc-blr", "--phase", "fresh",
                   "--replication", str(replication), "--seconds", "1500"]
        code = subprocess.run(command, stdout=subprocess.DEVNULL).returncode
        output = ROOT / f"regression-bounded-boundary-fresh-{replication}-gpu-r1"
        cost = read(output / "gpu-diagnostic-run.json")
        if cost["returncode"] != code:
            raise RuntimeError("worker exit/record disagreement")
        memory = (cost.get("runtime") or {}).get("memory_policy", {})
        if not (memory.get("all_physical_devices_memory_growth") is True
                and memory.get("configured_before_logical_device_initialization") is True):
            raise RuntimeError("launch-invalid memory policy")
        classification = "completed"
        if code:
            progress = read(output / "preparation/preparation_progress.json")
            failure = read(output / "failure.json")
            messages = [e.get("details", {}).get("diagnostics", {}).get("hmc_error_message", "")
                        for e in progress.get("events", [])]
            if (failure.get("exception") != "HMCPreparationFailure"
                    or not any(str(m).startswith("operational warmup produced a nonfinite trace") for m in messages)):
                raise RuntimeError("unclassified failure; inspect before continuing")
            classification = "preparation_numerical_veto"
        row = {"replication": replication, "returncode": code,
               "classification": classification, "elapsed_seconds": cost["elapsed_seconds"],
               "output": str(output)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "rows": rows, "supervisor_elapsed_seconds": time.monotonic() - started,
              "accounting": "Worker charges are recorded once in each gpu-diagnostic-run.json.",
              "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md"}
    with (ROOT / "fresh-queue-summary.json").open("x") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
