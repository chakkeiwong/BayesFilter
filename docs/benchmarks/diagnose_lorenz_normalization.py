"""Independent CPU Octave diagnostic; SciPy only reads the MAT result boundary."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import json
from pathlib import Path
import subprocess


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items() if not key.startswith("__")}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    root = Path("/home/chakwong/BayesFilter")
    command = ["/usr/bin/octave-cli", "--no-gui", "--quiet",
               str(output / "sources/reference_lorenz_normalization.m"), str(root), str(output)]
    subprocess.run(command, cwd=root, check=True)
    from scipy.io import loadmat
    summary = plain(loadmat(output / "export.mat", simplify_cells=True)["summary"])
    summary["reference_command"] = command
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "scope", "algorithm_seconds")}))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
