"""Diagnostic before/after comparison; incomplete fixture coverage fails closed."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from run_filter_repair_campaign import FIXTURES, records


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def compare_values(left, right):
    a, b = list(flatten(left)), list(flatten(right))
    if len(a) != len(b):
        raise ValueError("Output length changed")
    for x, y in zip(a, b, strict=True):
        if isinstance(x, (bool, int)):
            if x != y:
                raise ValueError("Discrete output changed")
        elif not (math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError(f"Numerical output mismatch: {x}, {y}")
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0)


def summarize(measurement):
    warm = measurement["warm"]
    snapshots = list(measurement["stages"].values()) + warm
    currents = [row.get("gpu", {}).get("current", 0) for row in warm]
    return {"trace_seconds": measurement["trace_seconds"], "cold_seconds": measurement["cold"]["synchronized_seconds"], "warm_median_seconds": statistics.median(row["synchronized_seconds"] for row in warm), "host_peak_bytes": max(row.get("VmHWM", 0) for row in snapshots), "device_peak_bytes": max(row.get("gpu", {}).get("peak", 0) for row in snapshots), "warm_current_range_bytes": max(currents) - min(currents), "graph_nodes": measurement["graph"]["nodes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurements = {}
    for row in records():
        action, _, arm, name, jit, size, repeat, device = row["key"]
        if action == "measure" and row["state"] == "passed":
            measurements[arm, name, jit, size, repeat, device] = (row, json.loads(Path(row["result"]).read_text()))
    result = {"schema": "filter_repair_comparison.v1", "pairs": [], "missing": [], "passed": False}
    for name in FIXTURES:
        for jit in ("off", "on"):
            for repeat in range(3):
                keys = [(arm, name, jit, 1, repeat, "GPU") for arm in ("before", "after")]
                if not all(key in measurements for key in keys):
                    result["missing"].append([name, jit, repeat])
                    continue
                (before_run, before), (after_run, after) = (measurements[key] for key in keys)
                for field in ("input_sha256", "input_shapes", "dimensions", "output_shapes", "tensorflow", "tf32", "hardware", "environment", "worker_sha256"):
                    if before[field] != after[field]:
                        raise ValueError(f"Unmatched comparison field: {field}")
                error = compare_values(before["values"], after["values"])
                result["pairs"].append({"fixture": name, "jit": jit, "repeat": repeat, "before": summarize(before), "after": summarize(after), "max_absolute_error": error, "before_run": before_run["result"], "after_run": after_run["result"]})
    result["passed"] = not result["missing"]
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"paired_measurements": len(result["pairs"]), "missing": result["missing"], "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
