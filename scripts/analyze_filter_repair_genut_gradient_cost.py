"""Independent diagnostic analysis of saved GenUT gradients and costs.

NumPy is a post-run comparator only. This module is never a runtime dependency.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

CONTRACT = "genut_bounded_gradient_frozen_coefficients_v2"
REPORT_FIELD = "fraction_coordinatewise_cap_active"
FIELDS = {"particles", "valid", "mean_residual", "covariance_residual",
          "maximum_pairwise_pre_cap_particle_rms", "maximum_pairwise_post_cap_particle_rms",
          "minimum_pairwise_particle_cap_scale", "maximum_coordinatewise_pre_cap_absolute",
          "maximum_coordinatewise_post_cap_absolute", "mean_coordinatewise_cap_displacement",
          REPORT_FIELD, "minimum_coordinatewise_cap_derivative", "loss",
          "source_gradient", "weight_gradient", "reset_gradient", "coefficients"}


def compare_arrays(actual, reference, count, dimension):
    assert set(actual) == set(reference) == FIELDS, "Incomplete output fields"
    fields = {}
    matrix = {"particles", "source_gradient", "reset_gradient", "coefficients"}
    for name in sorted(FIELDS):
        observed, expected = np.asarray(actual[name]), np.asarray(reference[name])
        shape = (count, dimension) if name in matrix else ((count,) if name == "weight_gradient" else ())
        assert observed.shape == expected.shape == shape, f"Unexpected {name} shape"
        assert np.isfinite(observed).all() and np.isfinite(expected).all(), f"Nonfinite {name}"
        error = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        bound = 2e-5 * (1 + np.abs(expected.astype(np.float64)))
        fields[name] = {"passed": bool(np.all(error <= bound)), "max_abs_error": float(np.max(error)),
                        "max_tolerance_units": float(np.max(error / bound)),
                        "failing_elements": int(np.count_nonzero(error > bound))}
    assert np.array_equal(actual["coefficients"].astype(np.float64), reference["coefficients"]), "Changed objective"
    assert bool(actual["valid"]) and bool(reference["valid"]), "Invalid particle result"
    return {"fields": fields,
            "smooth_passed": all(row["passed"] for name, row in fields.items() if name != REPORT_FIELD),
            "complete_passed": all(row["passed"] for row in fields.values())}


def load_run(path):
    run = json.loads((path / "run.json").read_text())
    assert run["state"] == "passed", f"Incomplete worker: {path.name}"
    is_reference = (path / "genut-bounded-gradient-fp64.json").exists()
    filename = "genut-bounded-gradient-fp64.json" if is_reference else "genut-bounded-gradient-cost.json"
    record = json.loads((path / filename).read_text())
    assert record["contract"] == CONTRACT, "Superseded coefficient contract"
    assert record["reference_commit"] == "28cbdb536" and record["trace_count"] == 1
    output = path / record["output_file"]
    assert output.name == "genut-bounded-gradient-output.npz"
    assert hashlib.sha256(output.read_bytes()).hexdigest() == record["output_sha256"], "Output hash mismatch"
    with np.load(output, allow_pickle=False) as saved:
        arrays = dict(saved)
    assert hashlib.sha256(arrays["coefficients"].astype(np.float32).tobytes()).hexdigest() == record["operand_sha256"][3]
    if is_reference:
        assert run["device"] == "CPU" and run["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert record["finite"] and record["exact_replay"]
        wanted = {(name, step) for name in ("source_gradient", "weight_gradient", "reset_gradient") for step in (1e-4, 5e-5)}
        checks = record["finite_differences"]
        assert len(checks) == len(wanted) and {(row["gradient"], row["step"]) for row in checks} == wanted
        assert all(row["passed"] and np.isclose(row["projection"], row["finite_difference"], rtol=1e-6, atol=1e-6)
                   for row in checks), "Invalid reference derivatives"
    else:
        assert record["first_record_finite"] and record["valid"] and record["python_owner_collected"]
        assert record["tf32_enabled"] and record["jit_compile"] == (record["arm"] == "after_xla")
        assert len(record["records"]) == 10 and record["cold_seconds"] > 0
        assert all(row["exact"] and np.isfinite(row["seconds"]) and row["seconds"] > 0 for row in record["records"])
        if record["jit_compile"]:
            hlo = path / "genut-bounded-gradient-optimized.hlo"
            assert hashlib.sha256(hlo.read_bytes()).hexdigest() == record["hlo"]["sha256"]
        provenance = [json.loads(line) for line in (path / "process.log").read_text().splitlines()
                      if line.startswith('{"tensorflow_version":')]
        assert len(provenance) == 1 and provenance[0]["tf32_enabled"]
        record["validated_gpu_uuid"] = validate_cost_device(run, provenance[0], record["device_provenance"])
    return run, record, arrays, is_reference


def analyze(root, runs):
    loaded = [(number, *load_run(root / f"run-{number:05d}")) for number in runs]
    closure = loaded[0][1]["source_sha256"]
    assert all(run["source_sha256"] == closure for _, run, *_ in loaded), "Source drift"
    references = {}
    for number, _, record, arrays, is_reference in loaded:
        if is_reference:
            key = record["count"], record["dimension"]
            assert key not in references, "Duplicate reference"
            references[key] = number, record, arrays
    assert set(references) == {(1000, 3), (10000, 18)}
    rows, arrays_by_arm = [], {}
    for number, run, record, arrays, is_reference in loaded:
        if is_reference:
            continue
        count, dimension = record["count"], record["dimension"]
        reference_number, reference, expected = references[count, dimension]
        assert record["operand_sha256"] == reference["operand_sha256"]
        assert record["saved_input_sha256"] == reference["saved_input_sha256"]
        assert record["reference_source_sha256"] == reference["reference_source_sha256"]
        key = run["device"], count, dimension, record["arm"]
        assert key not in arrays_by_arm, "Duplicate cost arm"
        arrays_by_arm[key] = arrays
        rows.append({"run": number, "reference_run": reference_number, "device": run["device"],
                     "count": count, "dimension": dimension, "arm": record["arm"],
                     "numerics": compare_arrays(arrays, expected, count, dimension),
                     "cold_seconds": record["cold_seconds"],
                     "median_warm_seconds": statistics.median(row["seconds"] for row in record["records"]),
                     "memory": {name: record[name] for name in ("before", "after_first", "after_replay", "before_hlo", "after_hlo", "after_collection")},
                     "gpu_uuid": record["validated_gpu_uuid"]})
    assert set(arrays_by_arm) == {(device, count, dim, arm) for device in ("CPU", "GPU")
                                  for count, dim in references for arm in ("before_graph", "after_graph", "after_xla")}
    require_same_physical_gpu(row["gpu_uuid"] for row in rows)
    graph_identity = []
    for device in ("CPU", "GPU"):
        for count, dim in references:
            before = arrays_by_arm[device, count, dim, "before_graph"]
            after = arrays_by_arm[device, count, dim, "after_graph"]
            graph_identity.append({"device": device, "count": count, "dimension": dim,
                                   "bitwise_equal": all(np.array_equal(before[name], after[name]) for name in FIELDS)})
    return {"contract": CONTRACT, "role": "descriptive_pilot_no_runtime_or_canonical_score_admission",
            "rows": rows, "same_mode_graph_identity": graph_identity,
            "scientific_speed_ranking": None,
            "all_smooth_passed": all(row["numerics"]["smooth_passed"] for row in rows),
            "complete_record_gate_passed": all(row["numerics"]["complete_passed"] for row in rows),
            "source_closure_sha256": hashlib.sha256(json.dumps(closure, sort_keys=True).encode()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    result = analyze(options.root, options.runs)
    with options.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(options.output), "rows": len(result["rows"]),
                      "all_smooth_passed": result["all_smooth_passed"],
                      "complete_record_gate_passed": result["complete_record_gate_passed"]}))
