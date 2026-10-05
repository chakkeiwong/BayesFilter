"""Diagnostic-only comparison of fresh-process memory and numeric outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten(item)
    else:
        yield float(value)


def metrics(result):
    stages = result["stages"]
    cold = stages["after_cold_outputs_live"]
    warm = stages["after_warm_outputs_released"]
    observations = [cold, *result["warm"]]
    warm_current = [item["gpu_allocator_bytes"]["current"] for item in result["warm"]]
    return {
        "trace_seconds": result["trace_seconds"],
        "cold_seconds_after_trace": result["cold_seconds_after_trace"],
        "cold_rss_mib": cold["VmRSS_kib"] / 1024,
        "cold_incremental_rss_mib": (cold["VmRSS_kib"] - stages["after_fixture_and_inputs"]["VmRSS_kib"]) / 1024,
        "gpu_peak_bytes": max(item["gpu_allocator_bytes"]["peak"] for item in observations),
        "warm_current_bytes": warm["gpu_allocator_bytes"]["current"],
        "warm_current_range_bytes": max(warm_current) - min(warm_current),
        "warm_rss_growth_kib": warm["VmRSS_kib"] - stages["after_cold_outputs_released"]["VmRSS_kib"],
        "trace_count": result["trace_count"],
        "warm_iterations": len(result["warm"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    comparisons = []
    for name in ("rectangular", "factor", "covariance", "sinkhorn_jvp"):
        paths = [args.input / f"{name}-{mode}.json" for mode in ("off", "on")]
        off, on = [json.loads(path.read_text()) for path in paths]
        for field in ("fixture", "dtype", "git_head", "output_shapes", "tensorflow", "environment"):
            if off[field] != on[field]:
                raise AssertionError(f"Unmatched {field} for {name}")
        old_sources, new_sources = (item["imported_repository_sources_sha256"] for item in (off, on))
        differing_sources = [path for path in old_sources.keys() | new_sources.keys() if old_sources.get(path) != new_sources.get(path)]
        if any(path != "scripts/measure_filter_xla_memory.py" for path in differing_sources):
            raise AssertionError(f"Runtime source drift: {differing_sources}")
        a, b = list(flatten(off["values"])), list(flatten(on["values"]))
        if len(a) != len(b) or not all(math.isfinite(item) for item in a + b):
            raise AssertionError(f"Invalid output for {name}")
        differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
        passed = all(error <= 1e-10 + 1e-10 * abs(y) for error, y in zip(differences, b, strict=True))
        if not passed or not off["finite"] or not on["finite"]:
            raise AssertionError(f"Parity failure for {name}")
        if off["graph"]["outer_xla_must_compile"] or any(off["graph"][field] for field in ("nested_xla_functions", "nested_xla_nodes", "explicit_xla_ops", "callbacks")):
            raise AssertionError("Invalid non-XLA arm")
        if not on["graph"]["outer_xla_must_compile"] or not Path(on["hlo"]["path"]).is_file() or on["graph"]["callbacks"]:
            raise AssertionError("Missing XLA evidence")
        off_stats, on_stats = metrics(off), metrics(on)
        if any(item["trace_count"] != 1 for item in (off_stats, on_stats)):
            raise AssertionError("Retracing")
        comparisons.append({
            "fixture": name, "artifacts": [str(path) for path in paths],
            "elementwise_parity_pass": passed, "max_abs_difference": max(differences),
            "elements": len(a), "non_xla": off_stats, "xla": on_stats,
            "gpu_peak_ratio_xla_over_non_xla": on_stats["gpu_peak_bytes"] / off_stats["gpu_peak_bytes"],
            "source_differences": differing_sources,
            "source_note": "First rectangular off/on worker differs only in import formatting; numerical sources match" if differing_sources else "All imported repository sources match within this pair",
        })
    report = {"schema": "filter_xla_memory_comparison.v1", "parity_tolerance": {"absolute": 1e-10, "relative": 1e-10}, "comparisons": comparisons, "limits": ["Four engineering fixtures, one process per arm, 20 warm invocations; not a repository-wide memory qualification", "GPU allocator bytes exclude driver/compiler allocations; process RSS includes host/compiler memory", "Non-XLA is ordinary tf.function graph execution, not eager", "No throughput ranking or proof of absence of long-run leaks"]}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for item in comparisons:
        print(json.dumps(item))


if __name__ == "__main__":
    main()
