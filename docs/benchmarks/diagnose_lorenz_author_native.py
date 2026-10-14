"""Independent CPU source-execution diagnostic; no BayesFilter numerical kernel."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k,v in value.items() if not k.startswith("__")}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, complex):
        return dict(real=plain(value.real), imag=plain(value.imag))
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path.cwd()
    command = ["octave", "--no-gui", "--quiet",
               str(a.output/"sources/reference_lorenz_author_native.m"),
               str(root), str(a.output)]
    subprocess.run(command, cwd=root, check=True)
    path = a.output/"summary.json"
    from scipy.io import loadmat  # Independent reference MAT serialization only.
    summary = plain(loadmat(a.output/"summary.mat", simplify_cells=True))["summary"]
    if isinstance(summary["checks"], dict):
        summary["checks"] = [summary["checks"]]
    progress = plain(loadmat(a.output/"progress.mat", simplify_cells=True))["progress"]
    (a.output/"progress.json").write_text(json.dumps(progress, allow_nan=False)+"\n")

    summary["reference_command"] = command
    summary["dependency_sha256"] = {
        key: hashlib.sha256(Path(summary[key]).read_bytes()).hexdigest()
        for key in ("mvnrnd_entry", "x2fx_entry", "squareform_entry")
    }
    path.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"], cases=len(summary["cases"]))))
    if summary["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
