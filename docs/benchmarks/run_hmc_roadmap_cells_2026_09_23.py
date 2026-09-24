"""Sequential funded M32/M34 cells; each child has its own budget and receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", type=int, choices=(32, 34), required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--attempt", default="r1")
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
root = repo / "docs/plans/artifacts/hmc-repair-master-2026-09-16" / f"m{args.phase}-r1"
cells = ([(target, strategy) for target in ("rotated_gaussian", "dirichlet", "residual")
          for strategy in ("static", "dynamic")] if args.phase == 32 else
         [(arm, "dynamic") for arm in ("baseline", "noop", "quarter", "half")])
cap = 800 if args.phase == 32 else 700
for target, strategy in cells:
    name = f"{target}-{strategy}-gpu-{args.attempt}"
    attempt = root / name
    command = [sys.executable, str(repo / "docs/benchmarks/run_hmc_roadmap_attempt_2026_09_23.py"),
        "--phase-root", str(root), "--name", name, "--seconds", str(cap),
        "--source-root", str(args.source_root.resolve()), "--device", "gpu", "--gpu", str(args.gpu),
        "--", sys.executable, str(repo / "docs/benchmarks/fit_hmc_roadmap_2026_09_23.py"),
        "--design", str(root / (target+"-design.json")), "--output", str(attempt / "fit"),
        "--strategy", strategy, "--seconds", str(cap-20)]
    with (root / (name + ".driver.log")).open("x") as log:
        result = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    receipt = json.loads((attempt / "execution.json").read_text())
    print(json.dumps({"cell": name, "exit_code": result.returncode,
                      "seconds": receipt["elapsed_seconds"]}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
