"""Exact saved-evidence hash parity and descriptive host timings, no sampler."""
from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO))
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_path = ROOT/"source-r1/bayesfilter/inference/hmc_candidate_set_tuning.py"
    tree = ast.parse(old_path.read_text())
    names = {"_stable_payload", "_sha256"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    import math
    namespace = {"Mapping": Mapping, "Any": Any, "math": math, "json": json, "hashlib": hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(old_path), "exec"), namespace)
    baseline = namespace["_sha256"]
    evidence = ROOT/"pilots-gpu-r1/m7-pilot-beta_binomial/replication-0000/tuning/numerical_evidence"
    files = sorted(evidence.glob("*.json"))
    assert files, "completed pilot evidence required"
    payloads = [json.loads(p.read_text()) for p in files]
    expected = [p.stem for p in files]
    for algorithm in (baseline, _sha256):
        assert [algorithm(p) for p in payloads] == expected
    timings = []
    for repeat in range(5):
        order = (("baseline", baseline), ("current", _sha256))
        if repeat % 2:
            order = tuple(reversed(order))
        record = {"repeat": repeat, "order": [name for name, _ in order]}
        for name, algorithm in order:
            started = time.perf_counter()
            observed = [algorithm(p) for p in payloads]
            elapsed = time.perf_counter() - started
            assert observed == expected
            record[name+"_seconds"] = elapsed
        timings.append(record)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "plan_file": "docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "baseline_file": str(old_path), "baseline_file_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        "current_file_sha256": hashlib.sha256((REPO/"bayesfilter/inference/hmc_candidate_set_tuning.py").read_bytes()).hexdigest(),
        "all_payload_hashes_identical": True, "payload_count": len(files),
        "payload_files": [str(p) for p in files], "expected_hashes": expected,
        "timing_repeats": timings,
        "median_seconds": {name: statistics.median(row[name+"_seconds"] for row in timings)
                           for name in ("baseline", "current")},
        "timing_role": "descriptive host cost on saved evidence; no end-to-end speedup or stochastic ranking claim",
        "gpu_intentionally_hidden": True, "sampler_executed": False}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("all_payload_hashes_identical", "payload_count", "median_seconds")}, indent=2))


if __name__ == "__main__":
    main()
