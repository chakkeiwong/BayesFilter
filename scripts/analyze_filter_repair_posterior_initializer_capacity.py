"""Diagnostic validation and descriptive initializer capacity trajectories."""

import argparse
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from analyze_filter_repair_posterior_initializer_costs import compare_records
from filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

REPORTING_OR_DIAGNOSTIC_SOURCES = frozenset({
    "scripts/run_filter_repair_campaign.py",
    "scripts/analyze_filter_repair_posterior_initializer_capacity.py",
    "tests/test_filter_repair_campaign.py",
    "tests/test_filter_repair_posterior_initializer_capacity_analysis.py",
    "tests/test_filter_repair_posterior_capacity_crash_diagnostic.py",
})


def source_differences(first, second):
    differences = {path: {"first": first.get(path), "current": second.get(path)}
        for path in first.keys() | second.keys() if first.get(path) != second.get(path)}
    assert differences.keys() <= REPORTING_OR_DIAGNOSTIC_SOURCES, "numerical source drift"
    return differences


def worker_vetoes(directory, run):
    errors = []
    if run["state"] != "passed" or run.get("exit_code") != 0:
        errors.append("worker_failed")
    try:
        cases = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
        if len(cases) != 1 or any(c.findall(t) for c in cases for t in ("failure", "error", "skipped")):
            errors.append("junit_failed")
    except (OSError, ET.ParseError):
        errors.append("missing_or_invalid_junit")
    return errors


def group_identity(group, device):
    match = re.fullmatch(r"posterior_initializer_capacity_(?:reuse_(prior|graph|xla)_(1|3)|owner_(1|3))_(cpu|gpu)", group)
    assert match is not None and match[4].upper() == device, "unknown capacity group/device"
    return {"kind": "reuse" if match[1] else "owner_replacement",
        "arm": match[1] or "xla", "dimension": int(match[2] or match[3]), "device": device}


def validate_capacity(result):
    assert result["schema"] == "filter_posterior_initializer_capacity.v1"
    assert result["reference_revision"] == "031692a0b"
    assert result["dimension"] in (1, 3)
    assert result["kind"] in ("reuse", "owner_replacement")
    records, originals = result["records"], result["original_records"]
    assert len(records) == len(result["elapsed_seconds"])
    assert all(math.isfinite(t) and t > 0 for t in result["elapsed_seconds"])
    if result["kind"] == "reuse":
        assert result["warm_calls"] == 20 and len(records) == 22 and len(originals) == 2
        assert result["arm"] in ("prior", "graph", "xla")
        assert set(result["stages"]) == {"prepared", "cold", "changed", *(f"warm_{n}" for n in (1, 5, 10, 15, 20))}
        for index, record in enumerate(records):
            compare_records(record, originals[index % 2])
            assert record == records[index % 2]
        if result["arm"] != "prior":
            program, reuse = result["program"], result["reuse"]
            assert program["jit_compile"] is (result["arm"] == "xla") and program["trace_count"] == 1
            if result["arm"] == "xla":
                assert program["hlo_unchanged"] and program["hlo_bytes"] > 0
            else:
                assert program["nested_xla_function_count"] > 0
            assert not program["host_callback_ops"]
            assert len(reuse) == 20 and all(r["owner_reused"] and r["traces"] == 1 for r in reuse)
            assert len({r["dependency_count"] for r in reuse}) == 1
        else:
            assert result["program"] is None and result["reuse"] == []
    else:
        assert result["arm"] == "xla" and result["owner_count"] == 4 and len(records) == 4 and len(originals) == 1
        assert len(result["superseded_collection"]) == 4 and len(result["final_collection"]) == 4
        for index, step in enumerate(result["superseded_collection"]):
            assert len(step) == index and all(len(row) == 4 and all(row) for row in step)
        assert all(len(row) == 4 and all(row) for row in result["final_collection"])
        programs = result["programs"]
        assert len(programs) == 4 and all(r["jit_compile"] and r["traces"] == 1 for r in programs)
        assert len({r["dependency_count"] for r in programs}) == 1
        for record in records:
            assert record == records[0]
            compare_records(record, originals[0])
    for record in records:
        assert record["accepted"] is True
        compare_records(record["center"], result["independent_gaussian"]["mean"], tolerance=1e-7)
        compare_records(record["covariance_theta"], result["independent_gaussian"]["covariance"], tolerance=1e-7)


def memory_trajectory(result):
    rows = []
    for label, value in result["stages"].items():
        rss = value["rollup"]["Rss"] / 1024**2
        assert math.isfinite(rss) and rss > 0
        allocator = value["gpu"]
        assert type(value["map_count"]) is int and value["map_count"] > 0
        if allocator is not None:
            assert all(type(allocator[key]) is int for key in ("current", "peak"))
            assert 0 <= allocator["current"] <= allocator["peak"]
        rows.append({"stage": label, "rss_mib": rss, "map_count": value["map_count"],
            "gpu_current_bytes": None if allocator is None else allocator["current"],
            "gpu_peak_bytes": None if allocator is None else allocator["peak"]})
    return rows


def validate_measurements(result):
    """Require complete, ordered observations before interpreting retention."""
    stages = result["stages"]
    if result["kind"] == "owner_replacement":
        assert set(stages) == {"prepared", "all_python_owners_released",
            *(f"owner_{n}_{label}" for n in range(1, 5) for label in ("completed", "collected"))}
    else:
        assert len(result["call_memory"]) == 22
        expected_returns = 0 if result["arm"] == "prior" else 21
        assert len(result["owner_returns"]) == expected_returns
        measurements = [*result["call_memory"], *(r["memory"] for r in result["owner_returns"])]
        for memory in measurements:
            assert math.isfinite(memory["rollup"]["Rss"]) and memory["rollup"]["Rss"] > 0
            assert (memory["gpu"] is not None) is result["gpu"]
            if result["gpu"]:
                assert 0 <= memory["gpu"]["current"] <= memory["gpu"]["peak"]
        for row in result["owner_returns"]:
            assert type(row["output_bytes"]) is int and row["output_bytes"] > 0
    assert all((memory["gpu"] is not None) is result["gpu"] for memory in stages.values())
    memory_trajectory(result)


def analyze(root, first, last, devices):
    results, controls, uuids, source, fixtures, environments, hardware = [], {}, [], None, {}, {}, {}
    drift, diagnostics, reference_sources = {}, [], None
    for number in range(first, last + 1):
        directory = root / f"run-{number:05d}"
        manifest = directory / "run.json"
        if not manifest.exists():
            continue
        run = json.loads(manifest.read_text())
        if run["device"] not in devices or not run["key"][1].startswith("posterior_initializer_capacity_"):
            continue
        if run["key"][1] == "posterior_initializer_capacity_analysis":
            continue
        if run["key"][1] == "posterior_initializer_capacity_prior_stack_cpu":
            diagnostics.append({"run": number, "state": run["state"], "exit_code": run.get("exit_code"),
                "role": "native_crash_localization_only"})
            continue
        assert run["state"] != "running", "unfinished capacity worker"
        source = run["source_sha256"] if source is None else source
        differences = source_differences(source, run["source_sha256"])
        if differences:
            drift[str(number)] = differences
        is_control = "observer" in run["key"][1]
        path = directory / ("posterior-initializer-capacity-observer.json" if is_control else "posterior-initializer-capacity.json")
        provenance, = [json.loads(line) for line in (directory / "process.log").read_text().splitlines()
            if line.startswith('{"tensorflow_version":')]
        environment = (run["environment"], provenance["tensorflow_version"],
            provenance["tf32_enabled"])
        environments.setdefault(run["device"], environment)
        assert environments[run["device"]] == environment, "execution environment drift"
        errors = worker_vetoes(directory, run)
        if not path.is_file():
            errors.append("missing_complete_capacity_record")
            assert not is_control, "missing observer control"
            results.append({"run": number, **group_identity(run["key"][1], run["device"]),
                "valid": False, "vetoes": errors, "memory": None, "seconds": None,
                "warm_median_seconds": None,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
            continue
        result = json.loads(path.read_text())
        hardware.setdefault(run["device"], result["hardware"])
        assert hardware[run["device"]] == result["hardware"], "hardware drift"
        if result["gpu"] is not (run["device"] == "GPU"):
            errors.append("device_mismatch")
        if not errors:
            try:
                uuids.append(validate_cost_device(run, provenance, result["gpu_process_observation"]))
            except (ValueError, KeyError, TypeError, AssertionError) as error:
                errors.append(f"device_veto: {error}")
        if is_control:
            assert not errors, "invalid observer control"
            assert result["schema"] == "filter_posterior_initializer_capacity_observer.v1"
            assert result["calls_between_snapshots"] == 0 and result["completed_records_retained"] == 22
            assert run["device"] not in controls
            assert result["owner_records_retained"] == 4
            controls[run["device"]] = {"run": number, "memory": memory_trajectory(result),
                "owner_memory": memory_trajectory({"stages": result["owner_stages"]})}
            continue
        identity = group_identity(run["key"][1], run["device"])
        assert all(result[key] == identity[key] for key in ("kind", "arm", "dimension"))
        assert result["original_source_sha256"], "missing original source identity"
        reference_sources = result["original_source_sha256"] if reference_sources is None else reference_sources
        assert reference_sources == result["original_source_sha256"], "original source drift"
        try:
            validate_capacity(result)
            validate_measurements(result)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            errors.append(f"numerical_veto: {error}")
        fixtures.setdefault(result["dimension"], result["fixture"])
        assert fixtures[result["dimension"]] == result["fixture"]
        trajectory = memory_trajectory(result)
        results.append({"run": number, **identity, "valid": not errors, "vetoes": errors, "memory": trajectory,
            "seconds": result["elapsed_seconds"],
            "call_memory": result.get("call_memory"), "owner_returns": result.get("owner_returns"),
            "warm_median_seconds": statistics.median(result["elapsed_seconds"][2:]) if result["kind"] == "reuse" and not errors else None,
            "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()})
    expected = {(d, n, "reuse", arm) for d in devices for n in (1, 3) for arm in ("prior", "graph", "xla")}
    expected |= {(d, n, "owner_replacement", "xla") for d in devices for n in (1, 3)}
    keys = [(r["device"], r["dimension"], r["kind"], r["arm"]) for r in results]
    assert len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete or duplicated capacity matrix"
    assert set(controls) == set(devices)
    require_same_physical_gpu(uuids)
    return {"schema": "filter_posterior_initializer_capacity_analysis.v3", "results": results,
        "observer_controls": controls, "source_sha256": source,
        "original_source_sha256": reference_sources,
        "reporting_or_diagnostic_source_differences": drift, "explanatory_runs": diagnostics,
        "nonclaims": ["Single-process descriptive trajectories, not statistical performance ranking.",
            "Python owner collection does not establish native executable eviction.",
            "No factor_max=2, D23/DZ5 or whole-repository capacity qualification."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--first-run", type=int, required=True)
    parser.add_argument("--last-run", type=int, required=True)
    parser.add_argument("--devices", nargs="+", choices=("CPU", "GPU"), default=["CPU", "GPU"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root, args.first_run, args.last_run, args.devices)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"workers": len(report["results"]), "controls": len(report["observer_controls"]),
        "output": str(args.output)}))
