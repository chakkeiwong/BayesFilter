"""Diagnostic analysis of fixed-source posterior and observer residency runs."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIB = 2**20


def read_run(index, filename):
    directory = ROOT / f"run-{index:05d}"
    manifest = json.loads((directory / "run.json").read_text())
    assert manifest["state"] == "passed" and manifest["test_evidence"]["passed"]
    path = directory / filename
    result = json.loads(path.read_text())
    metadata = next(json.loads(line) for line in (directory / "process.log").read_text().splitlines()
                    if line.startswith('{"tensorflow_version":'))
    if manifest["device"] == "GPU":
        assert manifest["environment"]["BAYESFILTER_TEST_DEVICE_SCOPE"] == "visible"
        assert metadata["gpu_memory_policy"]["all_physical_devices_memory_growth"]
        assert metadata["cuda_visible_devices"] == manifest["gpu_uuid"]
    else:
        assert manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    return manifest, result, hashlib.sha256(path.read_bytes()).hexdigest()


def resident(result, stage):
    return result["stages"][stage]["rollup"]["Rss"] / MIB


def executable_maps(result, stage):
    return result["mappings"][stage].get("anonymous_executable", {}).get("mappings", 0)


rows, sources, devices = [], [], set()
for index in range(2874, 2882):
    manifest, result, checksum = read_run(index, "posterior-public-residency.json")
    sources.append(manifest["source_sha256"])
    if result["gpu"]:
        devices.add(manifest["gpu_uuid"])
    row = {"run": index, "sha256": checksum, "device": manifest["device"],
        "dimension": result["dimension"], "minimal_xla_prewarm": result["minimal_xla_prewarm"],
        "rss_mib": {stage: resident(result, stage) for stage in result["stages"]},
        "endpoint_cold_after_trace_mib": resident(result, "cold") - resident(result, "traced"),
        "reuse_growth_mib": resident(result, "reuse_3000") - resident(result, "changed"),
        "executable_mappings_cold": executable_maps(result, "cold"),
        "executable_mappings_reuse": executable_maps(result, "reuse_3000"),
        "python_graph_released": result["python_graph_released"],
        "allocator_changed": result["stages"]["changed"]["gpu"],
        "allocator_reuse": result["stages"]["reuse_3000"]["gpu"],
        "seconds": result["seconds"]}
    if result["minimal_xla_prewarm"]:
        row["minimal_xla_growth_mib"] = resident(result, "minimal_xla") - resident(result, "prepared")
    rows.append(row)
assert all(source == sources[0] for source in sources)
assert len(devices) == 1

controls = []
for index in (2882, 2883):
    manifest, result, checksum = read_run(index, "posterior-residency-observer.json")
    assert result["calls_between_snapshots"] == 0
    controls.append({"run": index, "sha256": checksum, "device": manifest["device"],
        "rss_mib": {stage: resident(result, stage) for stage in result["stages"]},
        "seven_observation_intervals_growth_mib": resident(result, "7") - resident(result, "0"),
        "executable_mappings_before": executable_maps(result, "0"),
        "executable_mappings_after": executable_maps(result, "7")})

report = {"schema": "filter_posterior_residency_analysis.v1", "runs": rows,
    "observer_controls": controls, "gpu_uuid": next(iter(devices)),
    "matched_source_for_eight_residency_workers": True,
    "total_reuse_calls": 24000,
    "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "interpretation": [
        "Minimal XLA startup explains only part of endpoint cold host residency.",
        "Retained observation records themselves consume memory without numerical calls.",
        "Python graphs release, while native mappings and host residency persist.",
        "GPU allocator current/peak remain flat through the observed reuse interval."],
    "limitations": [
        "One process per condition; no timing ranking or cross-device comparison.",
        "Observer controls use a separate recorded harness source cohort.",
        "RSS/mappings do not identify every retained native allocation or prove leak freedom.",
        "These fixed-signature observations do not waive the preserved signature-churn failures."]}
destination = ROOT / "posterior-residency-analysis-02883.json"
with destination.open("x") as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write("\n")
print(destination)
print(json.dumps({"residency": [{key: row[key] for key in ("run", "device", "dimension",
    "minimal_xla_prewarm", "endpoint_cold_after_trace_mib", "reuse_growth_mib")} for row in rows],
    "controls": controls}, indent=2))
