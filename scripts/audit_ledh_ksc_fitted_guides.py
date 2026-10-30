"""Independent standard-library terminal audit; no runtime admission authority."""

import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("ordinary", "negative_future_tail", "positive_future_tail")
BETAS = {"identity": [1.0, 0.0, 0.0], "global": [0.1, 0.9, 0.0],
         "local": [0.1, 0.0, 0.9], "equal": [1.0 / 3.0] * 3}
ARMS = ("unguided_identity", "unguided_global", "unguided_local", "unguided_equal",
        "gaussian_future_global", "gaussian_future_equal", "mixture_future_global",
        "mixture_future_equal", "fitted_quadratic_global", "fitted_quadratic_equal",
        "fitted_kernel_global", "fitted_kernel_equal")


def read_json(path):
    return json.loads(path.read_text())


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left, right):
    assert math.isclose(left, right, rel_tol=2e-12, abs_tol=1e-14), (left, right)


def mean(values):
    return math.fsum(values) / len(values)


def trace_fields():
    tree = ast.parse((ROOT / "bayesfilter/highdim/covariance_proposal_tf.py").read_text())
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TRACE_FIELDS" for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise AssertionError("missing source trace schema")


def audit(campaign):
    development = campaign / "development-01"
    final = campaign / "final-02"
    development_result = read_json(development / "result.json")
    result = read_json(final / "result.json")
    manifest = read_json(final / "manifest.json")
    frozen = read_json(development / "frozen-selection.json")
    seeds = read_json(campaign / "seed-design.json")
    analysis = read_json(final / "analysis.json")
    assert result["status"] == development_result["status"] == manifest["status"] == "complete"
    assert seeds == manifest["seed_design"] == frozen["seed_design"]
    assert manifest["frozen_selection_sha256"] == checksum(development / "frozen-selection.json")
    assert manifest["branch"] == "ledh-refactor-with-policy-fix"
    assert manifest["data_version"] == "sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1"
    assert manifest["dtype"] == "float64" and manifest["jit_compile"] and not manifest["tf32_enabled"]
    assert manifest["particle_count"] == manifest["chunk_extent"] == 32
    assert manifest["horizon"] == 2
    assert manifest["controls"] == {"flow_steps": 16, "reset_steps": 5, "reset_epsilon": 1.0,
                                     "correction_steps": 2, "correction_basis": 4,
                                     "correction_rate": 0.1, "correction_radius": 0.25}
    assert manifest["gpu_memory_policy"]["all_physical_devices_memory_growth"]
    assert manifest["gpu_memory_policy"]["configured_before_logical_device_initialization"]
    assert manifest["gpu_trust"] == "escalated_local_gpu_access"
    assert manifest["score_route"] == "analytical_recursive_frozen_guide_total_derivative"
    for artifact_name, expected in analysis["input_hashes"].items():
        assert checksum(final / artifact_name) == expected
    for directory in (development, final):
        recorded = read_json(directory / "manifest.json")
        completed = read_json(directory / "result.json")
        assert recorded["wall_seconds"] == completed["wall_seconds"] <= 450.0
        for path, expected in recorded["source_hashes"].items():
            assert completed["source_hashes_after"][path] == expected
        for path, expected in completed["source_hashes_after"].items():
            assert checksum(ROOT / path) == expected, path
    for path, expected in frozen["source_hashes"].items():
        assert checksum(ROOT / path) == expected, path
    development_rows = [json.loads(line) for line in (development / "fits.jsonl").read_text().splitlines()]
    assert len(development_rows) == development_result["fits_completed"] == frozen["fit_count"] == 120
    setting_counts = Counter((row["condition"], row["config"]["id"]) for row in development_rows)
    assert len(setting_counts) == 30 and set(setting_counts.values()) == {4}
    ineligible_reasons = Counter()
    eligible_settings = {}
    for row in development_rows:
        config = row["config"]
        assert row["seed"] == seeds["calibration"][row["condition"]][config["id"]][row["replicate"]]
        assert row["eligible"] == row["fit"]["density_eligible"]
        eligible_settings.setdefault(config["id"], []).append(row)
        if not row["eligible"]:
            ineligible_reasons["invalid_fit" if not row["fit"]["fit_valid"] else "nonpositive_guide_precision"] += 1
    selection_loss = {}
    for family in ("quadratic", "kernel"):
        candidates = []
        for setting, group in eligible_settings.items():
            if group[0]["config"]["family"] == family and all(row["eligible"] for row in group):
                selection_loss[setting] = mean([row["validation"]["excess_cross_entropy"] for row in group])
                candidates.append((selection_loss[setting], group[0]["config"]))
        if candidates:
            chosen = min(candidates, key=lambda candidate: candidate[0])[1]
            assert frozen["selection"][family] == chosen
        else:
            assert frozen["selection"][family] is None
    rows = [json.loads(line) for line in (final / "rows.jsonl").read_text().splitlines()]
    indexed = {(row["condition"], row["cluster"], row["replicate"], row["arm"]): row for row in rows}
    expected = set(itertools.product(CONDITIONS, range(8), range(8), ARMS))
    assert len(rows) == len(indexed) == result["rows_completed"] == 2304 and set(indexed) == expected
    fields = trace_fields()
    counts = Counter()
    cloud_hashes = {}
    valid_trace_rows = 0
    for key, row in indexed.items():
        condition, cluster, replicate, arm = key
        assert row["seed"] == seeds["final_clouds"][condition][cluster][replicate]
        assert row["beta"] == BETAS[arm.rsplit("_", 1)[1]]
        cloud_hashes.setdefault(key[:3], set()).add(row["cloud_sha256"])
        counts[row["status"]] += 1
        if arm.startswith("fitted_"):
            family = arm.split("_")[1]
            assert row["fit_seed"] == seeds["final_fits"][condition][family][cluster]
            assert row["config"] == frozen["selection"][family]
            if row["config"] is None:
                assert row["status"] == "candidate_ineligible" and "score" not in row
                continue
            fitted = read_json(final / f"fit-{condition}-{family}-{cluster}.json")
            assert fitted["seed"] == row["fit_seed"] and fitted["config"] == row["config"]
            if row["status"] == "candidate_ineligible":
                assert not fitted["fit"]["guide_eligible"]
                continue
            assert fitted["fit"]["fit_valid"] and fitted["fit"]["guide_eligible"]
            if family == "kernel":
                assert fitted["fit"]["active"] == [True, False]
                close(math.fsum(math.exp(weight) for weight in fitted["fit"]["log_weights"]), 1.0)
                assert fitted["fit"]["bandwidth"] > 0.0
        assert row["status"] == "valid" and row["valid"] and all(row["direction_valid"])
        assert "GPU" in row["value_device"] and len(row["score"]) == 2
        assert all(math.isfinite(value) for value in [row["value"], *row["score"]])
        traces = row["trace"] + list(itertools.chain.from_iterable(row["direction_trace"]))
        assert len(traces) == 6
        for trace in traces:
            assert len(trace) == len(fields) and all(math.isfinite(value) for value in trace)
            named = dict(zip(fields, trace))
            assert all(named[name] == 1.0 for name in ("valid", "moment_valid", "correction_cap_valid"))
            assert named["reset_mean_error"] <= named["mean_error_limit"]
            assert named["reset_covariance_relative_error"] <= named["covariance_relative_error_limit"]
            valid_trace_rows += 1
    assert len(cloud_hashes) == 192 and all(len(hashes) == 1 for hashes in cloud_hashes.values())
    assert len(set(next(iter(hashes)) for hashes in cloud_hashes.values())) == 192
    expected_fd = {(condition, f"fitted_{family}_global", coordinate, step)
                   for condition in CONDITIONS for family in ("quadratic", "kernel")
                   if indexed[(condition, 0, 0, f"fitted_{family}_global")]["status"] == "valid"
                   for coordinate in range(2) for step in (1e-4, 5e-5)}
    assert {(item["condition"], item["arm"], item["coordinate"], item["step"])
            for item in result["finite_differences"]} == expected_fd
    for item in result["finite_differences"]:
        analytical = indexed[(item["condition"], 0, 0, item["arm"])]["score"][item["coordinate"]]
        close(item["error"], abs(item["difference"] - analytical))
        close(item["tolerance"], 2e-7 + 2e-5 * abs(item["difference"]))
        assert item["passed"] and item["error"] <= item["tolerance"]
    computed_mse = {}
    for condition in CONDITIONS:
        reference = result["references"][condition]
        assert reference["value_difference"] <= 1e-6 and reference["score_difference"] <= 1e-5
        for coordinate in range(2):
            for arm in ARMS:
                group = [indexed[(condition, cluster, replicate, arm)]
                         for cluster in range(8) for replicate in range(8)]
                if all(row["status"] == "valid" for row in group):
                    computed_mse[(condition, coordinate, arm)] = mean([
                        (row["score"][coordinate] - reference["score"][coordinate]) ** 2 for row in group])
            computed_mse[(condition, coordinate, "gaussian_observation_approximation")] = (
                reference["gaussian_approximation"]["score"][coordinate] - reference["score"][coordinate]) ** 2
    for metric in analysis["metrics"]:
        if metric["complete"]:
            close(metric["mse"], computed_mse[(metric["condition"], metric["coordinate"], metric["arm"])])
    assert len(analysis["contrasts"]) == 24
    critical = analysis["uncertainty"]["critical_value"]
    assert analysis["uncertainty"]["degrees_of_freedom"] == 7
    for contrast in analysis["contrasts"]:
        condition, coordinate, arm = contrast["condition"], contrast["coordinate"], contrast["arm"]
        if not contrast["available"]:
            assert (condition, coordinate, arm) not in computed_mse
            continue
        reference = result["references"][condition]["score"][coordinate]
        clusters = []
        for cluster in range(8):
            differences = []
            for replicate in range(8):
                candidate = indexed[(condition, cluster, replicate, arm)]["score"][coordinate] - reference
                baseline = indexed[(condition, cluster, replicate, "unguided_global")]["score"][coordinate] - reference
                differences.append(candidate ** 2 - baseline ** 2)
            clusters.append(mean(differences))
        average = mean(clusters)
        standard_error = math.sqrt(math.fsum((value - average) ** 2 for value in clusters) / 56.0)
        close(contrast["mean_mse_difference"], average)
        close(contrast["standard_error"], standard_error)
        close(contrast["lower"], average - critical * standard_error)
        close(contrast["upper"], average + critical * standard_error)
        assert contrast["supported_improvement"] == (contrast["upper"] < 0)
        assert contrast["supported_harm"] == (contrast["lower"] > 0)
    heuristic_table = []
    for condition, coordinate, arm, heuristic in itertools.product(
            CONDITIONS, range(2), ARMS[4:], (*ARMS[:4], "gaussian_observation_approximation")):
        eligible = (condition, coordinate, arm) in computed_mse
        entry = {"condition": condition, "coordinate": coordinate, "arm": arm,
                 "heuristic": heuristic, "available": eligible}
        if eligible:
            entry["candidate_mse"] = computed_mse[(condition, coordinate, arm)]
            entry["heuristic_mse"] = computed_mse[(condition, coordinate, heuristic)]
            entry["promotion_veto"] = entry["candidate_mse"] > entry["heuristic_mse"]
        heuristic_table.append(entry)
    expected_losses = {(item["condition"], item["coordinate"], item["arm"], item["heuristic"])
                       for item in heuristic_table if item.get("promotion_veto")}
    assert {(item["condition"], item["coordinate"], item["arm"], item["heuristic"])
            for item in analysis["heuristic_losses"]} == expected_losses
    return {"status": "pass", "role": "independent numerical/provenance audit, not canonical admission",
            "rows": len(rows), "row_statuses": dict(counts), "trace_checks": valid_trace_rows,
            "finite_difference_checks": len(expected_fd), "source_hashes": len(manifest["source_hashes"]),
            "development_ineligible_reasons": dict(ineligible_reasons),
            "selection": frozen["selection"], "selection_loss": selection_loss,
            "heuristic_table": heuristic_table, "heuristic_losses": len(expected_losses),
            "supported_harm_contrasts": [item for item in analysis["contrasts"] if item.get("supported_harm")],
            "supported_improvement_contrasts": [item for item in analysis["contrasts"] if item.get("supported_improvement")],
            "gpu_process_wall_seconds": development_result["wall_seconds"] + result["wall_seconds"],
            "launches_used": 2, "launches_allowed": 2, "trace_fields": fields,
            "input_hashes": {str(path.relative_to(campaign)): checksum(path) for path in (
                development / "manifest.json", development / "result.json", development / "fits.jsonl",
                development / "frozen-selection.json", final / "manifest.json", final / "result.json",
                final / "rows.jsonl", final / "analysis.json")}}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    result["wall_seconds"] = time.monotonic() - started
    result["command"] = __import__("sys").argv
    path = args.campaign / "terminal-audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "rows", "row_statuses",
          "trace_checks", "finite_difference_checks", "source_hashes", "heuristic_losses",
          "gpu_process_wall_seconds", "wall_seconds")}))


if __name__ == "__main__":
    main()
