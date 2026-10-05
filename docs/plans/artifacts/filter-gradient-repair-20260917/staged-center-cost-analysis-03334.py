"""Post-run CPU diagnostic; failed graph arms cannot support matched costs."""
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, "/tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts")
from filter_repair_cost_provenance import validate_cost_device

ROOT = Path(__file__).resolve().parent


def differences(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return [item for key in expected for item in differences(actual[key], expected[key], path + "." + key)]
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        return [item for index, (a, b) in enumerate(zip(actual, expected, strict=True))
                for item in differences(a, b, f"{path}[{index}]")]
    equal = (abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected)
             if isinstance(expected, float) else actual == expected)
    return [] if equal else [{"path": path, "actual": actual, "expected": expected}]


rows, source, environment, inputs, references = {}, None, None, {}, {}
for number in range(3329, 3335):
    directory = ROOT / f"run-{number:05d}"
    run = json.loads((directory / "run.json").read_text())
    data = json.loads((directory / "staged-center-cost.json").read_text())
    provenance, = [json.loads(line) for line in (directory / "process.log").read_text().splitlines()
                   if line.startswith('{"tensorflow_version":')]
    cases = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
    assert len(cases) == 1 and not cases[0].findall("error") and not cases[0].findall("skipped")
    assert run["device"] == "CPU" and data["gpu"] is False
    assert data["schema"] == "filter_staged_center_cost.v1"
    assert data["numerical_authority"] == data["mechanism_baseline"] == "3582b4ac"
    assert data["candidate_installed_publicly"] is False
    common = (run["environment"], provenance, data["original_source_sha256"])
    if source is None:
        source, environment = run["source_sha256"], common
    assert source == run["source_sha256"] and common == environment
    arm, dimension = data["arm"], data["dimension"]
    assert arm in ("prior", "graph", "xla") and dimension in (1, 3)
    assert data["jit_compile"] is (arm != "graph")
    identity = (data["input_sha256"], data["changed_input_sha256"], data["config"])
    inputs.setdefault(dimension, identity)
    assert inputs[dimension] == identity
    original = (data["original_result"], data["original_changed_result"])
    references.setdefault(dimension, original)
    assert not differences(list(original), list(references[dimension]))
    mismatches = []
    for label, expected in zip(("result", "changed_result"), original, strict=True):
        mismatches += differences(data[label], {**expected, "jit_compile": arm != "graph"}, label)
        assert data[label]["endpoint_accepted"] and data[label]["continuation_started"]
    passed = run["state"] == "passed"
    assert passed is (not mismatches)
    assert bool(cases[0].findall("failure")) is (not passed)
    if passed:
        validate_cost_device(run, provenance, data["gpu_process_observation"])
    else:
        assert arm == "graph" and run["exit_code"] == 1
    snapshots = [*data["stages"].values(), data["cold"]["memory"], data["changed_cost"]["memory"],
                 *(sample["memory"] for sample in data["samples"])]
    rss = {key: value["rollup"]["Rss"] / 2**20 for key, value in data["stages"].items()}
    row = {"run": number, "state": run["state"], "eligible_for_matched_costs": passed,
        "mismatches": mismatches, "build_seconds": data["build_seconds"],
        "cold_total_seconds": data["build_seconds"] + data["cold"]["seconds"],
        "warm_median_ms": 1000 * statistics.median(s["seconds"] for s in data["samples"]),
        "warm_sample_rss_mib": [s["memory"]["rollup"]["Rss"] / 2**20 for s in data["samples"]],
        "rss_mib": rss, "warm_growth_mib": rss["warm"] - rss["cold"],
        "observed_max_rss_mib": max(s["rollup"]["Rss"] for s in snapshots) / 2**20,
        "pss_mib": {k: s["rollup"]["Pss"] / 2**20 for k, s in data["stages"].items()},
        "map_counts": {k: s["map_count"] for k, s in data["stages"].items()},
        "initial_rusage_exceeds_process_hwm": data["stages"]["prepared"]["ru_maxrss_bytes"] >
            data["stages"]["prepared"]["status"]["VmHWM"],
        "programs": data["programs"],
        "artifact_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("run.json", "junit.xml", "staged-center-cost.json", "process.log")}}
    assert (dimension, arm) not in rows
    rows[dimension, arm] = row

assert set(rows) == {(dimension, arm) for dimension in (1, 3) for arm in ("prior", "graph", "xla")}
comparisons = []
for dimension in (1, 3):
    prior, candidate = rows[dimension, "prior"], rows[dimension, "xla"]
    assert prior["eligible_for_matched_costs"] and candidate["eligible_for_matched_costs"]
    cold_ratio = candidate["cold_total_seconds"] / prior["cold_total_seconds"]
    warm_ratio = candidate["warm_median_ms"] / prior["warm_median_ms"]
    extra_rss = candidate["observed_max_rss_mib"] - prior["observed_max_rss_mib"]
    triggers = []
    if cold_ratio > 2:
        triggers.append("cold_above_2x")
    if warm_ratio > 1.2:
        triggers.append("warm_above_1.2x")
    if extra_rss > 256 or candidate["observed_max_rss_mib"] > 2 * prior["observed_max_rss_mib"]:
        triggers.append("host_rss_above_256MiB_or_2x")
    comparisons.append({"dimension": dimension,
        "arms": {arm: rows[dimension, arm] for arm in ("prior", "graph", "xla")},
        "xla_vs_prior_cold_ratio": cold_ratio, "xla_vs_prior_warm_ratio": warm_ratio,
        "xla_extra_observed_rss_mib": extra_rss, "cost_triggers": triggers})
result = {"schema": "filter_staged_center_cost_analysis.v1", "device": "CPU",
    "source_sha256": source, "environment": environment, "comparisons": comparisons,
    "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "limits": ["One process per arm/extent and three warm calls: descriptive localization only.",
        "Both graph arms fail original-XLA full-record comparisons and are excluded from matched-cost conclusions.",
        "Original rebuilds stages each call; this is not an identical-graph compiler ablation.",
        "Use smaps rollup observations; initial rusage exceeds current-process VmHWM in all arms.",
        "Small warm RSS changes cannot establish a plateau, long-term bound or native eviction.",
        "CPU reference only; GPU, public integration, terminal costs and repository completion remain open."]}
output = ROOT / "staged-center-costs-cpu-03334.json"
with output.open("x") as handle:
    handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "comparisons": [
    {k: v for k, v in row.items() if k != "arms"} for row in comparisons]}, indent=2))
