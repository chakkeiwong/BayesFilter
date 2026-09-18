"""Matched execution diagnostics with current-source and completeness gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from run_filter_repair_campaign import (
    BASELINE,
    BASELINE_PARENT_PACKAGES,
    BASELINE_ROOT,
    FIXTURES,
    ROOT,
    campaign_output_root,
    measurement_harness,
    records,
)

ONE_SIZE = frozenset(("rectangular", "factor", "covariance", "sinkhorn_jvp"))
MATCHED_FIELDS = ("input_sha256", "input_shapes", "input_dtypes", "dimensions", "tensorflow",
                  "tf32", "hardware", "environment", "harness_sha256", "device")
REVIEW_PATH = ROOT / "docs/plans/filter_gradient_repair_investigation_reviews_20260917.json"


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    else:
        yield value


def compare_values(left, right):
    if isinstance(left, list) != isinstance(right, list):
        raise ValueError("Output structure changed")
    if isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("Output shape changed")
        return max((compare_values(x, y) for x, y in zip(left, right, strict=True)), default=0)
    if type(left) is not type(right):
        raise ValueError("Output dtype changed")
    if isinstance(left, (bool, int)):
        if left != right:
            raise ValueError("Discrete output changed")
    elif not (isinstance(left, float) and math.isfinite(left) and math.isfinite(right)
              and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)):
        raise ValueError(f"Numerical output mismatch: {left}, {right}")
    return abs(left - right)


def summarize(measurement):
    warm = measurement["warm"]
    if len(warm) < 20:
        raise ValueError("Twenty synchronized warm calls are required")
    snapshots = list(measurement["stages"].values()) + warm
    current = [row.get("gpu", {}).get("current", 0) for row in warm]
    rss = [row["VmRSS"] for row in warm]
    return {"preparation_seconds": measurement["preparation_seconds"],
            "trace_seconds": measurement["trace_seconds"],
            "cold_seconds": measurement["cold"]["synchronized_seconds"],
            "warm_median_seconds": statistics.median(row["synchronized_seconds"] for row in warm),
            "output_copy_median_seconds": statistics.median(row["output_copy_seconds"] for row in warm),
            "host_peak_bytes": max(row.get("VmHWM", 0) for row in snapshots),
            "device_peak_bytes": max(row.get("gpu", {}).get("peak", 0) for row in snapshots),
            "warm_current_range_bytes": max(current) - min(current),
            "late_device_growth_bytes": max(current[-5:]) - max(current[5:10]),
            "late_host_growth_bytes": max(rss[-5:]) - max(rss[5:10]),
            "graph_nodes": measurement["graph"]["nodes"]}


def baseline_compilation_failure(measurement):
    """Only explicit graph/XLA incompatibility may use a valid reference arm."""
    phase, error = measurement.get("phase"), measurement.get("error", "")
    graph = measurement.get("graph", {})
    if (phase == "trace" and measurement.get("jit") == "off"
            and measurement.get("error_type") == "RuntimeError"
            and "Invalid callback/compilation boundary" in error
            and not graph.get("callbacks") and (graph.get("nested_xla") or graph.get("xla_ops"))):
        return "baseline_forced_nested_xla_requires_eager_reference"
    if (phase == "trace" and measurement.get("error_type") == "ValueError"
            and measurement.get("fixture") in (
                "simulation_sv", "simulation_sir", "simulation_predator_prey")
            and "Generator.from_seed" in error
            and "tf.function only supports singleton tf.Variables" in error):
        return "baseline_generator_variable_created_during_trace"
    if phase == "trace" and measurement.get("error_type") in (
            "AttributeError", "TypeError", "NotImplementedError", "OperatorNotAllowedInGraphError"):
        if any(text in error for text in ("SymbolicTensor", "symbolic tf.Tensor",
                                         "symbolic `tf.Tensor`", "symbolic Tensor")):
            return "baseline_host_operation_during_trace"
        if (measurement.get("error_type") == "TypeError"
                and "Could not generate a generic TraceType" in error
                and "tf.ensure_shape" in error and "batch_finite_value_score" in error):
            return "baseline_forward_accumulator_shape_tracing_failure"
    if phase == "first_execution" and "XLA" in error and any(text in error for text in (
            "unsupported operations", "not supported", "must be a compile-time constant", "TensorList")):
        return "baseline_xla_compilation_failure"
    return None


def current_provenance(run, measurement, arm, current_hashes, baseline_hashes):
    if measurement.get("schema") != "filter_repair_measurement.v2":
        raise ValueError("Outdated measurement harness")
    if measurement.get("harness_sha256") != current_hashes:
        raise ValueError("Stale measurement harness")
    imported = measurement.get("imported_source_sha256", {})
    if not imported:
        raise ValueError("Missing imported source provenance")
    if arm == "before":
        if Path(measurement["source_root"]).resolve() != BASELINE_ROOT.resolve():
            raise ValueError("Baseline source root mismatch")
        if any(baseline_hashes.get(path) != digest for path, digest in imported.items()):
            raise ValueError("Baseline source contamination")
    else:
        measured_root = Path(measurement["source_root"]).resolve()
        if measured_root != Path(run.get("cwd", "")).resolve():
            raise ValueError("Candidate source root mismatch")
        if (measured_root != ROOT.resolve()
                and campaign_output_root(measured_root) != campaign_output_root(ROOT)):
            raise ValueError("Candidate source belongs to a different campaign repository")
        for path, digest in imported.items():
            # Older snapshots omit the two inert parent package markers.
            launched_digest = run.get("source_sha256", {}).get(path)
            if launched_digest is None and path in BASELINE_PARENT_PACKAGES:
                launched_digest = baseline_hashes.get(path)
            if launched_digest != digest:
                raise ValueError(f"Candidate source changed during measurement: {path}")
            file = ROOT / path
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Stale candidate source: {path}")
    if measurement["status"] == "passed":
        graph = measurement["graph"]
        if graph["callbacks"] or (measurement["jit"] != "eager" and measurement["trace_count"] != 1):
            raise ValueError("Callback or unbounded tracing")
        if measurement["jit"] == "off" and (graph["nested_xla"] or graph["xla_ops"]):
            raise ValueError("Non-XLA reference contains XLA")
        if measurement["jit"] == "on" and not (graph["must_compile"] and measurement.get("hlo_sha256")):
            raise ValueError("Missing complete XLA evidence")
        if measurement["device"] == "GPU" and any("GPU:0" not in device for device in measurement["output_devices"]):
            raise ValueError("GPU outputs escaped requested device")
        if measurement["device"] == "GPU" and not measurement.get("memory_policy"):
            raise ValueError("Missing verified memory policy")
    elif run["state"] != "failed":
        raise ValueError("Inconsistent failure record")


def compare_pair(before, after):
    for field in (*MATCHED_FIELDS, "output_shapes", "output_dtypes"):
        if field not in before or field not in after or before[field] != after[field]:
            raise ValueError(f"Unmatched comparison field: {field}")
    return compare_values(before["values"], after["values"])


def validate_repeat_hardware(pairs):
    scopes = {json.dumps(pair["hardware_scope"], sort_keys=True) for pair in pairs}
    if len(scopes) != 1:
        raise ValueError("Fresh-process repeats must use the same physical GPU and environment")


def regression_reasons(before, after):
    reasons = []
    if after["warm_median_seconds"] > 1.2 * before["warm_median_seconds"]:
        reasons.append("warm_time_over_20_percent")
    if after["device_peak_bytes"] > 2 * before["device_peak_bytes"]:
        reasons.append("device_peak_over_2x")
    if after["host_peak_bytes"] - before["host_peak_bytes"] > 256 * 2**20:
        reasons.append("host_peak_over_256_MiB")
    if after["late_device_growth_bytes"] > 0:
        reasons.append("continuing_device_allocation_growth")
    if after["late_host_growth_bytes"] > 2**20:
        reasons.append("continuing_host_RSS_growth_over_1_MiB")
    return reasons


def reviewed_investigations(investigations, reviews, root=ROOT):
    """Resolve only documented explanatory tradeoffs for the actual run set.

    Numerical failures, missing comparisons, and ongoing allocation growth
    cannot be waived. A review is stale whenever its measured run set changes.
    """
    unresolved, resolved = [], []
    for investigation in investigations:
        reasons = investigation.get("reasons", [investigation.get("reason")])
        eligible = not any(reason.startswith("continuing_") for reason in reasons)
        matches = [review for review in reviews if all(review.get(key) == investigation.get(key)
            for key in ("fixture", "size", "jit")) and review.get("reasons") == reasons
            and sorted(review.get("runs", [])) == sorted(investigation.get("runs", []))]
        review = matches[0] if len(matches) == 1 else None
        paths = [] if review is None else [Path(path) for path in review.get("evidence", [])]
        valid_paths = bool(paths) and all(not path.is_absolute() and ".." not in path.parts
            and (root / path).is_file() for path in paths)
        if (eligible and review and investigation.get("runs") and valid_paths
                and review.get("disposition") == "accept_documented_tradeoff"
                and all(isinstance(review.get(field), str) and review[field].strip()
                        for field in ("mechanism", "alternatives_checked", "reason_for_acceptance", "limitations"))):
            resolved.append({**investigation, "review": review})
        else:
            unresolved.append(investigation)
    return unresolved, resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    marker = json.loads((BASELINE_ROOT / "source-manifest.json").read_text())
    if marker["commit"] != BASELINE:
        raise ValueError("Wrong pinned baseline")
    measurements, excluded = {}, []
    for run in records():
        action, _, arm, name, jit, size, repeat, device = run["key"]
        if action != "measure" or not Path(run["result"]).is_file():
            continue
        value = json.loads(Path(run["result"]).read_text())
        try:
            current_provenance(run, value, arm, measurement_harness(name), marker["files"])
        except (ValueError, KeyError) as exc:
            excluded.append({"run": run["result"], "reason": str(exc)})
            continue
        measurements[arm, name, jit, size, repeat, device] = run, value
    result = {"schema": "filter_repair_comparison.v2", "pairs": [], "missing": [],
              "failures": [], "investigations": [], "excluded": excluded, "aggregates": [], "passed": False}
    for name in FIXTURES:
        device = "CPU" if name == "cpu_pool" else "GPU"
        for size in ((1,) if name in ONE_SIZE else (1, 2)):
            for jit in ("off", "on"):
                for repeat in range(3):
                    keys = [(arm, name, jit, size, repeat, device) for arm in ("before", "after")]
                    identity = {"fixture": name, "size": size, "jit": jit, "repeat": repeat}
                    if not all(key in measurements for key in keys):
                        result["missing"].append(identity)
                        continue
                    (before_run, before), (after_run, after) = (measurements[key] for key in keys)
                    if after["status"] != "passed":
                        result["failures"].append({**identity, "reason": "candidate_failed", "run": after_run["result"]})
                        continue
                    original_before = before_run["result"]
                    reason = None
                    if before["status"] != "passed":
                        reason = baseline_compilation_failure(before)
                        reference = next((measurements[key] for mode in ("off","eager")
                            if (key := ("before",name,mode,size,repeat,device)) in measurements
                            and measurements[key][1]["status"] == "passed"),None)
                        if not reason or reference is None:
                            result["failures"].append({**identity, "reason": "baseline_has_no_valid_reference", "run": original_before})
                            continue
                        before_run, before = reference
                    try:
                        error = compare_pair(before, after)
                        pair = {**identity, "before": summarize(before), "after": summarize(after),
                                "hardware_scope": {field: after[field] for field in
                                                   ("hardware", "environment", "tensorflow", "tf32", "device")},
                                "max_absolute_error": error, "before_run": before_run["result"], "after_run": after_run["result"],
                                "baseline_compilation_failure": reason, "baseline_attempt": original_before}
                        pair["before_mode"] = before["jit"]
                        result["pairs"].append(pair)
                    except (ValueError, KeyError) as exc:
                        result["failures"].append({**identity, "reason": str(exc)})
    groups = {}
    for pair in result["pairs"]:
        groups.setdefault((pair["fixture"], pair["size"], pair["jit"]), []).append(pair)
    for (name, size, jit), pairs in groups.items():
        if len(pairs) != 3:
            continue
        try:
            validate_repeat_hardware(pairs)
        except ValueError as exc:
            result["failures"].append({"fixture": name, "size": size, "jit": jit, "reason": str(exc)})
            continue
        summaries = {arm: {field: (max if field.startswith("late_") else statistics.median)(pair[arm][field] for pair in pairs)
                           for field in pairs[0][arm]} for arm in ("before", "after")}
        reasons = regression_reasons(summaries["before"], summaries["after"])
        aggregate = dict(fixture=name, size=size, jit=jit, **summaries,
            warm_ratio=summaries["after"]["warm_median_seconds"]/summaries["before"]["warm_median_seconds"],
            repeat_warm_ranges={arm: [min(p[arm]["warm_median_seconds"] for p in pairs),
                max(p[arm]["warm_median_seconds"] for p in pairs)] for arm in ("before", "after")},
            descriptive_only=True)
        result["aggregates"].append(aggregate)
        if reasons:
            result["investigations"].append({"fixture": name, "size": size, "jit": jit, "reasons": reasons,
                "runs": [pair["after_run"] for pair in pairs]})
    for name in FIXTURES:
        if name in ONE_SIZE:
            continue
        for jit in ("off", "on"):
            small = groups.get((name,1,jit),[])
            large = groups.get((name,2,jit),[])
            if len(small) == len(large) == 3:
                delta = statistics.median(p["after"]["graph_nodes"] for p in large)-statistics.median(p["after"]["graph_nodes"] for p in small)
                if delta > 50:
                    result["investigations"].append({"fixture": name,"jit": jit,"reason": "graph_growth_with_extent","additional_nodes": delta,
                        "runs": [pair["after_run"] for pair in (*small,*large)]})
    reviews = json.loads(REVIEW_PATH.read_text())["reviews"] if REVIEW_PATH.is_file() else []
    result["investigations"], result["resolved_investigations"] = reviewed_investigations(result["investigations"], reviews)
    result["passed"] = not (result["missing"] or result["failures"] or result["investigations"])
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"paired_measurements": len(result["pairs"]), "missing": len(result["missing"]),
        "failures": result["failures"], "investigations": result["investigations"], "passed": result["passed"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
