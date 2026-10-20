"""Independent standard-library inspection of callback qualification records."""

import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def analyze(root):
    runs = {n: json.loads((root / f"run-{n:05d}" / "run.json").read_text())
            for n in range(3812, 3816)}
    assert runs[3812]["state"] == "failed"
    for n, count in ((3813, 8), (3814, 8), (3815, 129)):
        assert runs[n]["state"] == "passed"
        cases = list(ET.parse(root / f"run-{n:05d}" / "junit.xml").getroot().iter("testcase"))
        assert len(cases) == count
        assert not any(list(case) for case in cases)
    assert runs[3813]["source_sha256"] == runs[3814]["source_sha256"] == runs[3815]["source_sha256"]
    metadata = {}
    for n in (3813, 3814):
        line = next(line for line in (root / f"run-{n:05d}" / "process.log").read_text().splitlines()
                    if line.startswith('{"tensorflow_version"'))
        metadata[n] = json.loads(line)
    assert metadata[3813]["cuda_visible_devices"] == "-1"
    assert metadata[3814]["gpu_memory_policy"]["all_physical_devices_memory_growth"]
    assert metadata[3814]["gpu_memory_policy"]["physical_devices"]

    def numbers(value):
        if isinstance(value, list):
            return [x for child in value for x in numbers(child)]
        assert isinstance(value, (float, int)) and math.isfinite(value)
        return [value]

    summary = []
    files = sorted((root / "run-03813").glob("ledh-model-*.json"))
    assert len(files) == 6
    for path in files:
        cpu = json.loads(path.read_text())
        gpu = json.loads((root / "run-03814" / path.name).read_text())
        assert cpu["passed"] and gpu["passed"]
        assert cpu["model"] == gpu["model"] and cpu["dtype"] == gpu["dtype"]
        assert cpu["baseline_module_sha256"] == gpu["baseline_module_sha256"]
        tolerance = 1e-12 if cpu["dtype"] == "float64" else 1e-5
        cross_errors = []
        for left, right in zip(cpu["comparisons"], gpu["comparisons"], strict=True):
            assert left["label"] == right["label"]
            assert left["outputs"].keys() == right["outputs"].keys()
            for key in left["outputs"]:
                for a, b in zip(numbers(left["outputs"][key]), numbers(right["outputs"][key]), strict=True):
                    cross_errors.append(abs(a - b) / (tolerance * (1. + abs(a))))
        summary.append({"model": cpu["model"], "dtype": cpu["dtype"],
            "cross_device_max_scaled_error": max(cross_errors),
            "original_max_scaled_error": max(max(row["original_scaled_errors"].values())
                for report in (cpu, gpu) for row in report["comparisons"]),
            "graph_max_scaled_error": max(max(row["graph_scaled_errors"].values())
                for report in (cpu, gpu) for row in report["comparisons"]),
            "derivative_max_scaled_error": max(max(row["scaled_errors"].values())
                for report in (cpu, gpu) for row in report["independent_tangents"]),
            "native_loop_count": cpu["native_loop_count"], "trace_count": cpu["trace_count"],
        })
        assert max(summary[-1][key] for key in ("cross_device_max_scaled_error",
            "original_max_scaled_error", "graph_max_scaled_error", "derivative_max_scaled_error")) <= 1.
    return {"schema": "filter_repair_merged_ledh_models_analysis.v1", "passed": True,
        "through_run": 3815, "checks": {"CPU": 8, "GPU": 8, "policy": 129},
        "preserved_failures": [3812], "source_files": len(runs[3813]["source_sha256"]),
        "gpu_uuid": runs[3814]["gpu_uuid"], "metadata": metadata,
        "models": summary, "elapsed_seconds": {n: row["elapsed_seconds"] for n, row in runs.items()},
        "nonclaims": ["Full filter and score, default readiness, source equivalence, performance and memory attribution."]}


if __name__ == "__main__":
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
