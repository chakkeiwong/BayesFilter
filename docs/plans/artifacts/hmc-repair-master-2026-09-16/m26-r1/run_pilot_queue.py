"""Execute the six frozen M26 development designs, two isolated workers at most."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = root.parents[4]
source = root / "source-r2"
launcher = root / "run_attempt.py"
python = sys.executable
rows = []
for target in ("gaussian", "beta_binomial", "rotated_gaussian"):
    def run(method):
        name = target + "-" + method + "-pilot-r1"
        command = [python, str(launcher), "--name", name, "--seconds", "900",
            "--source-root", str(source), "--", python,
            "docs/benchmarks/run_hmc_fit_lifetime_2026_09_22.py", "--design",
            str(root / (target + "-" + method + "-pilot-design.json")), "--output",
            str(root / name / "fits"), "--mode", "isolated"]
        with (root / (name + "-launcher.log")).open("x") as log:
            result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        return {"target": target, "method": method, "returncode": result.returncode, "command": command}
    with ThreadPoolExecutor(max_workers=2) as pool:
        batch = list(pool.map(run, ("lugsail", "autocorrelation")))
    rows.extend(batch)
    (root / "pilot-queue.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(batch), flush=True)
    if any(row["returncode"] != 0 for row in batch):
        raise SystemExit(1)
