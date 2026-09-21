"""Run M12's two reserved checks after the three-fit queue terminates."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def read(path):
    return json.loads(path.read_text())


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    assert os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() == "true"
    deadline = time.monotonic() + 1600
    while not (ROOT / "fresh-queue-summary.json").exists():
        if time.monotonic() > deadline:
            raise RuntimeError("fresh queue did not finish; diagnose before final checks")
        time.sleep(5)
    queue = read(ROOT / "fresh-queue-summary.json")
    if len(queue["rows"]) != 3:
        raise ValueError("incomplete fresh-fit denominator")
    checks = [
        (ROOT / "source-gpu-r1", ROOT / "regression-sequential-probe-fresh-0-gpu-r1/manifest.json",
         ROOT / "fresh-0-trajectory-replay-gpu-r1", ["--trajectory-diagnostics"]),
        (ROOT / "source-final-r2", ROOT.parent / "m10-r1/regression-finite-window-fresh-0-gpu-r1/manifest.json",
         ROOT / "inactive-option-parity-gpu-r1", []),
    ]
    rows = []
    for source, manifest, output, options in checks:
        command = [sys.executable, str(REPO / "scripts/diagnose_hmc_metric_preparation.py"),
                   "--source", str(source), "--recorded-manifest", str(manifest), "--output", str(output),
                   "--campaign-stage", "m12", "--seconds", "900", *options]
        code = subprocess.run(command, stdout=subprocess.DEVNULL).returncode
        cost = read(output / "gpu-diagnostic-run.json")
        if code != cost["returncode"]:
            raise ValueError("exit/record mismatch")
        if options:
            windows = read(output / "trajectory_diagnostics/windows.json")
            if code != 1 or not windows[-1]["failed_indices"]:
                raise RuntimeError("failed-fit replay did not reproduce a numerical failure")
        elif code:
            raise RuntimeError("inactive-option parity preparation failed")
        row = {"output": str(output), "returncode": code, "elapsed_seconds": cost["elapsed_seconds"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (ROOT / "final-checks-summary.json").open("x") as stream:
        json.dump({"rows": rows, "command": sys.argv, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "accounting": "Each child GPU worker is charged once; waiting is not numerical execution."}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
