"""Independent diagnostic analysis of the frozen KDM cost cohort.

Reads saved outputs only; never selects or admits an inference algorithm.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import median

from scripts.filter_repair_cost_provenance import validate_cost_device


def _equal(left, right, tolerance=2.e-10):
    if type(left) is not type(right):
        raise ValueError("Output type changed")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("Output schema changed")
        for key in left:
            _equal(left[key], right[key], tolerance)
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError("Output shape changed")
        for a, b in zip(left, right, strict=True):
            _equal(a, b, tolerance)
    elif isinstance(left, float):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Nonfinite accepted output")
        if abs(left - right) > tolerance * (1. + abs(right)):
            raise ValueError("Numerical output changed")
    elif left != right:
        raise ValueError("Output status changed")


def _public(numerical, reference):
    result = {**numerical}
    if result.pop("model_valid") is not True:
        raise ValueError("Model invalid")
    steps = result.pop("steps")
    result["steps"] = [{"time_index": i, **{key: value[i] for key, value in steps.items()}}
                       for i in range(2)]
    for key in ("route_id", "route_role", "canonical_target_label", "auxiliary_target_label"):
        result[key] = reference[key]
    return result


def analyze(root):
    root = Path(root)
    records, files, sources = [], [], None
    for number in range(3929, 3947):
        directory = root / f"run-{number:05d}"
        run = json.loads((directory / "run.json").read_text())
        value = json.loads((directory / "kdm-cost.json").read_text())
        if run["state"] != "passed" or run["exit_code"] != 0 or not run["test_evidence"]["passed"]:
            raise ValueError("Unqualified worker")
        junit = ET.parse(directory / "junit.xml").getroot()
        cases = list(junit.iter("testcase"))
        if len(cases) != 1 or any(list(c) for c in cases):
            raise ValueError("Missing or failed numerical check")
        if sources is None:
            sources = run["source_sha256"]
        elif sources != run["source_sha256"]:
            raise ValueError("Cost cohort source drift")
        provenance = [json.loads(line) for line in (directory / "process.log").read_text().splitlines()
                      if line.startswith('{"tensorflow_version":')]
        if len(provenance) != 1:
            raise ValueError("Missing device provenance")
        validate_cost_device(run, provenance[0], value["cost_device_observations"])
        arm, device = value["arm"], run["device"]
        if run["key"][1] != f"kdm_auxiliary_cost_{arm}_{device.lower()}":
            raise ValueError("Mismatched arm/device")
        if (value["shape"] != [2, 4, 2] or value["seed"] != 131
                or value["reset_sinkhorn_steps"] != 8 or value["reset_balance_steps"] != 8
                or value["enclosing_xla"] != (arm == "xla")
                or value["jit_compile"] != (arm != "graph")):
            raise ValueError("Fixture or execution mode changed")
        times = value["public_seconds"]
        if len(times) != 3 or any(not math.isfinite(t) or t <= 0 for t in times):
            raise ValueError("Invalid public timing")
        reference = value["public_results"][0]
        if (reference["valid"] is not True or reference["canonical_value_unchanged"] is not True
                or reference["kdm_feedback_into_canonical"] is not False):
            raise ValueError("Invalid diagnostic trajectory")
        for result in value["public_results"]:
            _equal(result, reference)
        if arm != "original":
            if (value["retained_owner_traces"] != 1
                    or len(value["retained_owner_warm_seconds"]) != 5
                    or any(not math.isfinite(t) or t <= 0 for t in value["retained_owner_warm_seconds"])):
                raise ValueError("Invalid retained-owner timing or retracing")
            _equal(_public(value["retained_result"], reference), reference)
            _equal(value["retained_result"], value["retained_replay"], tolerance=0.)
        if value["sampled_peak_rss_bytes"] < value["memory_after_public"]["rss_bytes"]:
            raise ValueError("Invalid memory observation")
        if f"device:{device}:0" not in value["value_device"]:
            raise ValueError("Result device mismatch")
        records.append((number, run, value))
        for name in ("run.json", "process.log", "junit.xml", "kdm-cost.json"):
            path = directory / name
            files.append({"path": str(path.relative_to(root)),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    reference = records[0][2]
    for _, _, value in records:
        if (value["reference_source_sha256"] != reference["reference_source_sha256"]
                or value["reference_reset_sha256"] != reference["reference_reset_sha256"]):
            raise ValueError("Frozen reference drift")
        _equal(value["public_results"][0], reference["public_results"][0])
    summary = []
    for device in ("CPU", "GPU"):
        for arm in ("original", "graph", "xla"):
            cohort = [(n, r, v) for n, r, v in records if r["device"] == device and v["arm"] == arm]
            if len(cohort) != 3 or {r["key"][6] for _, r, _ in cohort} != {0, 1, 2}:
                raise ValueError("Missing or duplicate repeat")
            values = [v for _, _, v in cohort]
            summary.append({"device": device, "arm": arm, "runs": [n for n, _, _ in cohort],
                "public_cold_seconds_median": median(v["public_seconds"][0] for v in values),
                "public_repeat_seconds_median": median(median(v["public_seconds"][1:]) for v in values),
                "retained_warm_seconds_median": None if arm == "original" else
                    median(median(v["retained_owner_warm_seconds"]) for v in values),
                "public_rss_bytes_median": median(v["memory_after_public"]["rss_bytes"] for v in values),
                "whole_phase_peak_rss_bytes_median": median(v["sampled_peak_rss_bytes"] for v in values),
                "device_allocator_peak_bytes_median": None if device == "CPU" else
                    median(v["memory_after_owner"]["allocator"]["peak"] for v in values)})
    return {"schema": "filter_repair_kdm_cost_analysis.v1", "passed": True,
            "scope": "Qualified tiny FP64 fixture; descriptive costs, no ranking or capacity admission",
            "summary": summary, "files": files,
            "qualification_limits": ["FP32 cross-mode reset comparison remains open",
                "Native full-phase memory includes an additional retained owner",
                "Repeated-owner graph retention remains open", "No canonical LEDH or HMC admission"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": result["passed"], "output": str(args.output)}))
