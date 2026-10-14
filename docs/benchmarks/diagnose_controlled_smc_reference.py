"""CPU-only independent Octave reference and MAT-to-JSON diagnostic export.

SciPy/NumPy arrays exist only inside this reference adapter; no BayesFilter
runtime imports this module. Numerical author code stays external and unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items() if not key.startswith("__")}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["pilot", "demo"], required=True)
    args = parser.parse_args()
    output = Path(args.output)
    root = Path("/home/chakwong/BayesFilter")
    script = output / "sources/reference_controlled_smc.m"
    command = ["/usr/bin/octave-cli", "--no-gui", "--quiet", str(script), str(root), str(output), args.mode]
    print(json.dumps({"octave_command": command}), flush=True)
    code = subprocess.call(command, cwd=root)
    if code:
        return code
    from scipy.io import loadmat
    data = loadmat(output / "export.mat", simplify_cells=True)
    summary = plain(data["summary"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if args.mode == "pilot":
        (output / "fixture.json").write_text(json.dumps(plain(data["fixture"]), allow_nan=False) + "\n")
    print(json.dumps(summary), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
