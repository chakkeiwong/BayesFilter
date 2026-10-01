"""Independent standard-library analysis of the matched LEDH flow cost unit."""

import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def analyze(root):
    repo = Path(__file__).resolve().parents[4]
    validator_path = repo / "scripts/filter_repair_cost_provenance.py"
    spec = importlib.util.spec_from_file_location("cost_provenance", validator_path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    rows, closure, uuids = [], None, []
    for number in range(3837, 3855):
        directory = root / f"run-{number:05d}"
        run = json.loads((directory / "run.json").read_text())
        payload = json.loads((directory / "flow-cost.json").read_text())
        provenance = json.loads(next(line for line in (directory / "process.log").read_text().splitlines()
            if line.startswith('{"tensorflow_version"')))
        cases = list(ET.parse(directory / "junit.xml").getroot().iter("testcase"))
        assert len(cases) == 1 and not list(cases[0])
        assert payload["valid"] and payload["trace_count"] == 1
        assert payload["shape"] == [32, 2, 2] and payload["substeps"] == 24
        assert payload["dtype"] == "float64" and payload["seeds"] == [13, 37]
        assert len(payload["warm_seconds"]) == 15 and payload["rss_sample_count"] > 0
        assert payload["baseline"] == "9d8202b77"
        if closure is None:
            closure = run["source_sha256"]
        assert closure == run["source_sha256"]
        assert closure["scripts/filter_repair_cost_provenance.py"] == hashlib.sha256(validator_path.read_bytes()).hexdigest()
        uuids.append(validator.validate_cost_device(run, provenance, payload["cost_provenance"]))
        allocator = payload["after_warm"]["allocator"]
        if run["device"] == "GPU":
            assert allocator is not None
        rows.append({"run": number, "device": run["device"], "arm": payload["arm"],
            "repeat": run["key"][6], "cold_seconds": payload["cold_seconds"],
            "warm_median_seconds": statistics.median(payload["warm_seconds"]),
            "sampled_peak_rss_bytes": payload["sampled_peak_rss_bytes"],
            "incremental_peak_rss_bytes": payload["sampled_peak_rss_bytes"] - payload["before"]["rss_bytes"],
            "allocator_peak_bytes": None if allocator is None else allocator["peak"],
            "graph_node_count": payload["graph_node_count"],
            "result_sha256": hashlib.sha256((directory / "flow-cost.json").read_bytes()).hexdigest()})
    validator.require_same_physical_gpu(uuids)
    summary = {}
    for device in ("CPU", "GPU"):
        arms = {}
        for arm in ("prior_graph", "native_graph", "native_xla"):
            selected = [row for row in rows if row["device"] == device and row["arm"] == arm]
            assert sorted(row["repeat"] for row in selected) == [0, 1, 2]
            arms[arm] = {key: statistics.median(row[key] for row in selected)
                for key in ("cold_seconds", "warm_median_seconds", "sampled_peak_rss_bytes",
                            "incremental_peak_rss_bytes", "graph_node_count")}
            arms[arm]["allocator_peak_bytes"] = None if device == "CPU" else statistics.median(row["allocator_peak_bytes"] for row in selected)
        prior = arms["prior_graph"]
        for arm in ("native_graph", "native_xla"):
            candidate = arms[arm]
            candidate["warm_ratio_to_prior_graph"] = candidate["warm_median_seconds"] / prior["warm_median_seconds"]
            candidate["cold_ratio_to_prior_graph"] = candidate["cold_seconds"] / prior["cold_seconds"]
            candidate["triggers"] = {
                "warm_slowdown": candidate["warm_ratio_to_prior_graph"] > 1.2,
                "cold_slowdown": candidate["cold_ratio_to_prior_graph"] > 2.,
                "host_peak_increase": (candidate["incremental_peak_rss_bytes"] > 2 * prior["incremental_peak_rss_bytes"]
                    and candidate["incremental_peak_rss_bytes"] - prior["incremental_peak_rss_bytes"] > 100 * 2**20),
                "allocator_peak_increase": device == "GPU" and candidate["allocator_peak_bytes"] > 2 * prior["allocator_peak_bytes"]
                    and candidate["allocator_peak_bytes"] - prior["allocator_peak_bytes"] > 16 * 2**20,
            }
        summary[device] = arms
    return {"schema": "filter_repair_ledh_flow_cost_analysis.v1", "numerical_and_provenance_pass": True,
        "rows": rows, "descriptive_medians": summary, "source_count": len(closure),
        "physical_gpu_uuid": next(value for value in uuids if value is not None),
        "nonclaims": ["No population ranking from three repeats.", "No full filter/score or main promotion.",
                      "No native executable eviction or attribution of CPU compiler RSS is proved."],
        "preserved_initial_runs": [3832, 3833, 3834, 3835, 3836]}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
