"""Read-only numerical-result audit; no runtime selection or admission authority."""

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    rows = [json.loads(line) for line in (directory / "evaluations.jsonl").read_text().splitlines()]
    arms = {name: (family, beta) for name, family, beta in manifest["arms"]}
    conditions = manifest["conditions"]
    expected = set(itertools.product(conditions, range(8), arms))
    actual = [(row["condition"], row["replicate"], row["arm"]) for row in rows]
    assert len(rows) == len(expected) == 192 and set(actual) == expected
    assert len(set(actual)) == len(actual)
    assert result["rows_completed"] == len(rows)
    assert result["decision"]["status"] == "engineering_screen_passed"
    assert manifest["target_id"] == "sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1"
    assert manifest["jit_compile"] and not manifest["tf32_enabled"]
    assert manifest["dtype"] == "float64" and manifest["transport_chunk"] == 32
    assert manifest["controls"]["reset_steps"] == 5
    assert manifest["controls"]["reset_epsilon"] == 1.0
    assert manifest["gpu_memory_policy"]["all_physical_devices_memory_growth"]
    traces = []
    cloud_hashes = defaultdict(set)
    for row in rows:
        family, beta = arms[row["arm"]]
        assert row["guide_family"] == family and row["beta"] == beta
        assert row["seed"] == manifest["seeds"][row["condition"]][row["replicate"]]
        cloud_hashes[row["condition"], row["replicate"]].add(row["cloud_sha256"])
        assert row["valid"] and all(row["direction_valid"])
        assert len(row["direction_valid"]) == len(row["score"]) == 2
        assert "GPU" in row["value_device"]
        assert all(math.isfinite(value) for value in [row["value"], *row["score"]])
        assert len(row["trace"]) == len(row["direction_trace"]) == 2
        assert all(len(direction) == 2 for direction in row["direction_trace"])
        traces.extend(row["trace"])
        traces.extend(itertools.chain.from_iterable(row["direction_trace"]))
    assert len(cloud_hashes) == 24 and all(len(hashes) == 1 for hashes in cloud_hashes.values())
    assert len(set(next(iter(hashes)) for hashes in cloud_hashes.values())) == 24
    fields = manifest["trace_fields"]
    for trace in traces:
        assert len(trace) == len(fields) and all(math.isfinite(value) for value in trace)
        named = dict(zip(fields, trace))
        for field in ("valid", "moment_valid", "correction_cap_valid"):
            assert named[field] == 1.0
        assert named["reset_mean_error"] <= named["mean_error_limit"]
        assert named["reset_covariance_relative_error"] <= named["covariance_relative_error_limit"]
    differences = result["finite_differences"]
    expected_differences = set(itertools.product(conditions,
        ("gaussian_future_global", "mixture_future_global"), range(2), (1e-4, 5e-5)))
    actual_differences = {(record["condition"], record["arm"], record["coordinate"], record["step"])
                          for record in differences}
    assert len(differences) == 24 and actual_differences == expected_differences
    for record in differences:
        tolerance = 2e-7 + 2e-5 * abs(record["finite_difference"])
        error = abs(record["analytical"] - record["finite_difference"])
        assert record["pass"] and error <= tolerance
        assert math.isclose(error, record["difference"], abs_tol=1e-15)
    reference_errors = [result["single_observation_reference_check"]]
    assert set(result["references"]) == set(conditions)
    for reference in result["references"].values():
        reference_errors.append(reference)
    assert all(record["value_difference"] <= 1e-6
               and record["score_difference"] <= 1e-5 for record in reference_errors)
    computed = {}
    for condition in conditions:
        reference = result["references"][condition]
        for arm in arms:
            selected = [row for row in rows if row["condition"] == condition and row["arm"] == arm]
            assert len(selected) == 8
            computed[condition, arm] = [statistics.fmean(
                (row["score"][coordinate] - reference["score"][coordinate]) ** 2
                for row in selected) for coordinate in range(2)]
        computed[condition, "gaussian_observation_approximation"] = [
            (reference["gaussian_approximation"]["score"][coordinate] - reference["score"][coordinate]) ** 2
            for coordinate in range(2)]
    for metric in result["summary"]["conditional_metrics"]:
        assert metric["score_mse"] == computed[metric["condition"], metric["arm"]]
    heuristics = [name for name, (family, _) in arms.items() if family == "none"]
    heuristics.append("gaussian_observation_approximation")
    guided = [name for name, (family, _) in arms.items() if family != "none"]
    expected_comparisons = set(itertools.product(conditions, guided, range(2), heuristics))
    losses = Counter()
    comparisons = result["summary"]["heuristic_dominance"]["comparisons"]
    assert len(comparisons) == 120
    assert {(row["condition"], row["arm"], row["coordinate"], row["heuristic"])
            for row in comparisons} == expected_comparisons
    for row in comparisons:
        condition, coordinate = row["condition"], row["coordinate"]
        loss = computed[condition, row["arm"]][coordinate] > computed[condition, row["heuristic"]][coordinate]
        assert row["promotion_veto"] == loss
        losses[row["arm"]] += int(loss)
    assert sum(losses.values()) == result["summary"]["heuristic_dominance"]["loss_count"]
    before = manifest["source_hashes_before"]
    after = result["source_hashes_after"]
    for path, checksum in before.items():
        assert after[path] == checksum
    for path, checksum in after.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == checksum, path
    return {"status": "pass", "rows": len(rows), "paired_clouds": len(cloud_hashes),
            "finite_difference_checks": len(differences),
            "maximum_derivative_absolute_error": max(row["difference"] for row in differences),
            "maximum_derivative_tolerance_fraction": max(row["difference"] / row["tolerance"]
                                                          for row in differences),
            "maximum_reference_value_difference": max(row["value_difference"] for row in reference_errors),
            "maximum_reference_score_difference": max(row["score_difference"] for row in reference_errors),
            "source_hashes_verified": len(after), "heuristic_losses_by_arm": dict(losses),
            "roundoff_warning_count_across_value_and_directions": sum(
                trace[fields.index("moment_roundoff_warning")] != 0.0 for trace in traces),
            "statistically_supported_ranking": False,
            "method": "stdlib independent post-run inspection; no TensorFlow or GPU work"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    report = audit(args.attempt)
    report["wall_seconds"] = time.monotonic() - started
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
