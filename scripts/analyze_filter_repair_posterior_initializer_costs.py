"""Post-run diagnostic analysis of complete initializer costs and numerical vetoes."""

import argparse
import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)


def compare_records(actual, expected, *, tolerance=1e-10, path="result"):
    """Compare all fields except explicitly boolean execution-setting metadata."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            if key == "jit_compile":
                assert type(actual[key]) is bool and type(expected[key]) is bool, path
            else:
                compare_records(actual[key], expected[key], tolerance=tolerance, path=f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_records(left, right, tolerance=tolerance, path=f"{path}[{i}]")
    elif type(expected) in (float, int) and type(actual) in (float, int):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), path
    else:
        assert type(actual) is type(expected) and actual == expected, path


def validate_numerics(result):
    arm = result["arm"]
    assert result["schema"] == "filter_posterior_initializer_cost.v1"
    assert result["reference_revision"] == "031692a0b" and result["reference_entire_module"] is True
    assert result["dimension"] in (1, 3) and arm in ("prior", "graph", "xla")
    assert result["jit_compile"] is (arm != "graph")
    assert len(result["samples"]) == len(result["warm_records"]) == 3
    assert all(record == result["result"] for record in result["warm_records"]), "Warm replay differs"
    assert len(result["original_records"]) == 2
    for actual, expected in zip((result["result"], result["changed_result"]), result["original_records"], strict=True):
        compare_records(actual, expected)
        assert actual["accepted"] is True
        assert actual["diagnostics"]["config"]["locator_config"]["jit_compile"] is (arm != "graph")
        compare_records(actual["center"], result["independent_gaussian"]["mean"], tolerance=1e-7)
        compare_records(actual["covariance_theta"], result["independent_gaussian"]["covariance"], tolerance=1e-7)
    if arm != "prior":
        program = result["program"]
        assert program["trace_count"] == 1 and not program["host_callback_ops"]
        assert program["jit_compile"] is (arm == "xla")
        if arm == "xla":
            assert program["hlo_bytes"] > 0 and program["hlo_unchanged"] is True
    else:
        assert result["program"] is None


def metric_row(number, directory, result):
    snapshots = [*result["stages"].values(), result["cold"]["memory"], result["changed_cost"]["memory"],
        *(sample["memory"] for sample in result["samples"])]
    times = [result["cold"]["seconds"], result["changed_cost"]["seconds"],
        *(sample["seconds"] for sample in result["samples"])]
    assert all(math.isfinite(value) and value > 0 for value in times)
    assert all(row["rollup"]["Rss"] > 0 and row["map_count"] > 0 for row in snapshots)
    return {"run": number,
        "result_sha256": hashlib.sha256((directory / "posterior-initializer-cost.json").read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((directory / "run.json").read_bytes()).hexdigest(),
        "cold_seconds": result["cold"]["seconds"],
        "warm_seconds": statistics.median(sample["seconds"] for sample in result["samples"]),
        "changed_seconds": result["changed_cost"]["seconds"],
        "observed_rss_mib": max(row["rollup"]["Rss"] for row in snapshots) / 1024**2,
        "host_hwm_mib": max(row["ru_maxrss_bytes"] for row in snapshots) / 1024**2,
        "max_map_count": max(row["map_count"] for row in snapshots),
        "gpu_allocator_peak_bytes": max(row["gpu"]["peak"] for row in snapshots) if result["gpu"] else None,
        "warm_rss_growth_mib": (result["stages"]["warm"]["rollup"]["Rss"]
            - result["stages"]["cold"]["rollup"]["Rss"]) / 1024**2}


def summarize_pair(prior, candidate):
    cold = statistics.median(row["cold_seconds"] for row in candidate) / statistics.median(row["cold_seconds"] for row in prior)
    warm = statistics.median(row["warm_seconds"] for row in candidate) / statistics.median(row["warm_seconds"] for row in prior)
    prior_rss, new_rss = max(row["observed_rss_mib"] for row in prior), max(row["observed_rss_mib"] for row in candidate)
    gpu_ratio = None
    if prior[0]["gpu_allocator_peak_bytes"] is not None:
        baseline = max(row["gpu_allocator_peak_bytes"] for row in prior)
        peak = max(row["gpu_allocator_peak_bytes"] for row in candidate)
        gpu_ratio = peak / baseline if baseline else None
    triggers = []
    if cold > 2:
        triggers.append("cold_above_2x")
    if warm > 1.2:
        triggers.append("warm_above_20_percent")
    if new_rss - prior_rss > 256 or new_rss > 2 * prior_rss:
        triggers.append("host_rss_above_256MiB_or_2x")
    if gpu_ratio is not None and gpu_ratio > 2:
        triggers.append("gpu_allocator_peak_above_2x")
    return {"cold_ratio": cold, "warm_ratio": warm, "extra_observed_rss_mib": new_rss - prior_rss,
        "gpu_allocator_peak_ratio": gpu_ratio, "investigation_triggers": triggers}


def analyze(root, first, last, devices):
    groups, identities, numerical_inputs, uuids = {}, {}, {}, []
    global_source = None
    for number in range(first, last + 1):
        directory = root / f"run-{number:05d}"
        manifest = directory / "run.json"
        if not manifest.is_file():
            continue
        run = json.loads(manifest.read_text())
        if run["device"] not in devices or not run["key"][1].startswith("posterior_initializer_cost_"):
            continue
        device, repeat = run["device"], run["key"][6]
        assert repeat in (0, 1, 2)
        result_path = directory / "posterior-initializer-cost.json"
        assert result_path.is_file(), f"Run {number} has no completed cost record; inspect its worker log"
        result = json.loads(result_path.read_text())
        dimension, arm = result["dimension"], result["arm"]
        key = device, dimension, arm
        rows = groups.setdefault(key, {})
        assert repeat not in rows, "Duplicate arm/repeat; select the reviewed cohort interval"
        provenance, = [json.loads(line) for line in (directory / "process.log").read_text().splitlines()
            if line.startswith('{"tensorflow_version":')]
        if global_source is None:
            global_source = run["source_sha256"]
        assert run["source_sha256"] == global_source, "Cross-device source drift"
        identity = (run["source_sha256"], provenance["tensorflow_version"], provenance["tf32_enabled"],
            run["environment"], result["original_source_sha256"], result["hardware"])
        identities.setdefault(device, identity)
        assert identities[device] == identity, "Source or execution environment drift"
        inputs = tuple(result[field] for field in ("input_sha256", "changed_input_sha256", "target_sha256",
            "config", "movement_config", "thresholds"))
        numerical_inputs.setdefault(dimension, inputs)
        assert numerical_inputs[dimension] == inputs, "Numerical fixture drift"
        assert result["gpu"] is (device == "GPU")
        cases = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
        errors = []
        if (run["state"] != "passed" or run.get("exit_code") != 0 or len(cases) != 1
                or any(case.findall(tag) for case in cases for tag in ("failure", "error", "skipped"))):
            errors.append("worker_or_junit_failed")
        try:
            validate_numerics(result)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            errors.append(f"numerical_veto: {error}")
        if not errors:
            try:
                uuids.append(validate_cost_device(run, provenance, result["gpu_process_observation"]))
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                errors.append(f"device_veto: {error}")
        rows[repeat] = {"valid": not errors, "vetoes": errors, **metric_row(number, directory, result)}
    require_same_physical_gpu(uuids)
    assert set(groups) == {(device, dimension, arm) for device in devices for dimension in (1, 3)
        for arm in ("prior", "graph", "xla")}, "Incomplete device/dimension/arm matrix"
    assert all(set(rows) == {0, 1, 2} for rows in groups.values()), "Three fresh processes per arm required"
    comparisons = []
    for device in devices:
        for dimension in (1, 3):
            arms = {arm: [groups[device, dimension, arm][repeat] for repeat in (0, 1, 2)]
                for arm in ("prior", "graph", "xla")}
            comparisons.append({"device": device, "dimension": dimension, "arms": arms,
                "prior_pairs": {arm: summarize_pair(arms["prior"], arms[arm])
                    if all(row["valid"] for row in arms["prior"] + arms[arm]) else None
                    for arm in ("graph", "xla")}})
    return {"schema": "filter_posterior_initializer_cost_comparison.v1", "comparisons": comparisons,
        "source_sha256_by_device": {device: item[0] for device, item in identities.items()},
        "nonclaims": ["Three fresh-process repeats are descriptive costs, not a statistical performance ranking.",
            "The immediate reference contains earlier repairs; oldest-original campaign gates remain open.",
            "The graph arm is a non-default diagnostic and can retain compiled numerical dependencies.",
            "Host snapshots and TensorFlow allocator peaks do not prove total native peak bounds or eviction.",
            "These Gaussian fixtures do not qualify actual DZ5 callers or broad filtering workloads."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-run", type=int, required=True)
    parser.add_argument("--last-run", type=int, required=True)
    parser.add_argument("--devices", nargs="+", choices=("CPU", "GPU"), default=["CPU", "GPU"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.first_run, args.last_run, args.devices)
    report["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{key: row[key] for key in ("device", "dimension", "prior_pairs")}
        for row in report["comparisons"]], indent=2))
