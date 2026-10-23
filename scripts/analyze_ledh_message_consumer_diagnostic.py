"""Independent diagnostic audit and conditional message/consumer attribution."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("ordinary", "negative_future_tail", "positive_future_tail")
HEURISTICS = ("unguided_identity", "unguided_global", "unguided_local", "unguided_equal",
              "gaussian_future_global", "gaussian_future_equal")
CANDIDATES = ("product_kernel_global", "product_kernel_equal",
              "exact_product_global", "exact_product_equal")
ARMS = HEURISTICS + ("mixture_future_global", "mixture_future_equal",
                     "fitted_kernel_global", "fitted_kernel_equal") + CANDIDATES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return True


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run / "analysis.json"
    require(not destination.exists(), "analysis already exists; preserve prior evidence")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "analysis must explicitly hide GPU devices")
    from scipy.stats import t

    manifest = json.loads((args.run / "manifest.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    require(manifest["status"] == result["status"] == "complete", "incomplete comparison")
    require(manifest["jit_compile"] and not manifest["tf32_enabled"]
            and manifest["dtype"] == "float64", "wrong execution scope")
    require((manifest["particle_count"], manifest["chunk_extent"], manifest["horizon"])
            == (32, 32, 2), "wrong numerical scope")
    require(manifest["controls"]["global_guide_conditioning"] == "positive_information"
            and manifest["product_controls"]["global_guide_conditioning"]
            == "mixture_product_moments", "consumer identity mismatch")
    require(manifest["score_route"] == "analytical_recursive_frozen_guide_total_derivative",
            "wrong score route")
    require(sha256(Path(manifest["plan"])) == manifest["plan_sha256"], "plan changed")
    for name, expected in manifest["source_hashes"].items():
        require(sha256(ROOT / name) == expected, f"source changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        require(sha256(Path(name)) == expected, f"input changed: {name}")
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    require(len(rows) == result["rows_completed"] == 2688, "row coverage mismatch")
    require(result["fits_loaded"] == len(manifest["fit_inputs"]) == 24, "fit coverage mismatch")
    syntax = ast.parse((ROOT / "bayesfilter/highdim/covariance_proposal_tf.py").read_text())
    fields = next(ast.literal_eval(node.value) for node in syntax.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "TRACE_FIELDS"
                          for target in node.targets))
    indexed = {}
    trace_checks = 0
    for row in rows:
        key = (row["condition"], row["cluster"], row["replicate"], row["arm"])
        require(key not in indexed, "duplicate paired row")
        require(key[0] in CONDITIONS and key[1] in range(8) and key[2] in range(8)
                and key[3] in ARMS, "unexpected design row")
        indexed[key] = row
        require(row["status"] == "valid" and row["valid"]
                and len(row["direction_valid"]) == 2 and all(row["direction_valid"]),
                "consumer validity veto")
        require("GPU" in row["value_device"], "CPU consumer fallback")
        require(len(row["score"]) == len(row["direction_trace"]) == 2, "wrong score dimension")
        for name in ("value", "score", "trace", "direction_trace", "ancestor_history"):
            require(finite_tree(row[name]), f"nonfinite {name}")
        for trace in [row["trace"], *row["direction_trace"]]:
            require(len(trace) == 2, "wrong trace horizon")
            for step in trace:
                require(len(step) == len(fields), "trace schema mismatch")
                for flag in ("valid", "moment_valid", "correction_cap_valid"):
                    require(step[fields.index(flag)] == 1., f"trace veto: {flag}")
                require(step[fields.index("guide_margin")] > 0, "invalid guide covariance")
                trace_checks += 1
    all_seeds = []
    for condition_index, condition in enumerate(CONDITIONS):
        reference = result["references"][condition]
        require(finite_tree(reference), "nonfinite reference")
        require(abs(reference["value"] - reference["grid"]["value"]) <= 1e-6
                and max(abs(left - right) for left, right in
                        zip(reference["score"], reference["grid"]["score"])) <= 1e-5,
                "independent reference disagreement")
        exact_hashes = set()
        for cluster in range(8):
            name = f"fit-{condition}-{cluster}.json"
            fit_identity = manifest["fit_inputs"][name]
            fit = json.loads((args.run / name).read_text())
            require(sha256(args.run / name) == fit_identity["sha256"], "copied fit changed")
            expected_seed = manifest["seed_design"]["fits"][condition_index][cluster]
            require(fit["seed"] == fit_identity["seed"] == expected_seed, "fit seed mismatch")
            require(fit["fit"]["guide_eligible"], "ineligible fit")
            fit_hash = digest([fit["fit"][key] for key in
                               ("logits", "centers", "covariances", "active")])
            for replicate in range(8):
                paired = [indexed[(condition, cluster, replicate, arm)] for arm in ARMS]
                seed = manifest["seed_design"]["clouds"][condition_index][cluster][replicate]
                all_seeds.append(seed)
                require({row["seed"] for row in paired} == {seed}, "cloud seed mismatch")
                require(len({row["cloud_sha256"] for row in paired}) == 1, "unpaired clouds")
                for row in paired:
                    arm = row["arm"]
                    require(row["fit_seed"] == expected_seed, "row fit seed mismatch")
                    if arm.startswith(("product_", "fitted_")):
                        require(row["guide_sha256"] == fit_hash, "fitted guide mismatch")
                    if arm.startswith(("exact_product_", "mixture_future_")):
                        exact_hashes.add(row["guide_sha256"])
                    expected_beta = ((1., 0., 0.) if arm.endswith("identity") else
                                     (.1, 0., .9) if arm.endswith("local") else
                                     (.1, .9, 0.) if arm.endswith("global") else (1. / 3.,) * 3)
                    require(tuple(row["beta"]) == expected_beta, "proposal mixture mismatch")
                    consumer = ("mixture_product_moments" if arm in CANDIDATES
                                else "positive_information")
                    require(row["global_guide_conditioning"] == consumer, "row consumer mismatch")
        require(len(exact_hashes) == 1, "exact message not frozen across consumer/cluster")
    require(len(set(all_seeds)) == len(all_seeds) == 192, "fresh cloud seeds overlap")
    require(manifest["cloud_seeds_disjoint_from_prior"], "prior seed overlap")
    expected_fd = {(condition, arm, coordinate, step) for condition in CONDITIONS
                   for arm in CANDIDATES for coordinate in range(2) for step in (1e-4, 5e-5)}
    observed_fd = {(check["condition"], check["arm"], check["coordinate"], check["step"])
                   for check in result["finite_differences"]}
    require(observed_fd == expected_fd and len(result["finite_differences"]) == 48,
            "finite-difference coverage mismatch")
    require(all(check["passed"] and math.isfinite(check["error"])
                and check["error"] <= check["tolerance"]
                for check in result["finite_differences"]), "derivative mismatch")
    summaries = []
    summary_index = {}
    squared_errors = {}
    for condition in CONDITIONS:
        for coordinate in range(2):
            reference = result["references"][condition]["score"][coordinate]
            for arm in ARMS:
                errors = [[indexed[(condition, cluster, replicate, arm)]["score"][coordinate]
                           - reference for replicate in range(8)] for cluster in range(8)]
                flat = [error for cluster in errors for error in cluster]
                mse = statistics.mean(error * error for error in flat)
                bias = statistics.mean(flat)
                variance = statistics.pvariance(flat)
                require(math.isclose(mse, bias * bias + variance, rel_tol=1e-12, abs_tol=1e-14),
                        "bias/variance accounting mismatch")
                summary = {"condition": condition, "coordinate": coordinate, "arm": arm,
                           "mse": mse, "bias": bias, "bias_squared": bias * bias,
                           "variance": variance}
                summaries.append(summary)
                summary_index[(condition, coordinate, arm)] = summary
                squared_errors[(condition, coordinate, arm)] = [
                    statistics.mean(error * error for error in cluster) for cluster in errors]
    contrasts = []
    critical = float(t.ppf(1. - .05 / (2 * 36), 7))
    comparisons = [(arm, "unguided_global") for arm in CANDIDATES] + [
        (f"exact_product_{mixture}", f"product_kernel_{mixture}") for mixture in ("global", "equal")]
    for condition in CONDITIONS:
        for coordinate in range(2):
            for candidate, comparator in comparisons:
                paired = [left - right for left, right in zip(
                    squared_errors[(condition, coordinate, candidate)],
                    squared_errors[(condition, coordinate, comparator)])]
                mean = statistics.mean(paired)
                half_width = critical * statistics.stdev(paired) / math.sqrt(8)
                lower, upper = mean - half_width, mean + half_width
                contrasts.append({"condition": condition, "coordinate": coordinate,
                                  "candidate": candidate, "comparator": comparator,
                                  "cluster_mse_differences": paired, "mean_difference": mean,
                                  "lower": lower, "upper": upper,
                                  "verdict": "supported_gain" if upper < 0 else
                                  "supported_harm" if lower > 0 else "unresolved"})
    require(len(contrasts) == 36, "contrast family mismatch")
    heuristic_table = []
    for condition in CONDITIONS:
        for coordinate in range(2):
            for candidate in CANDIDATES:
                for comparator in HEURISTICS:
                    left = summary_index[(condition, coordinate, candidate)]["mse"]
                    right = summary_index[(condition, coordinate, comparator)]["mse"]
                    heuristic_table.append({"condition": condition, "coordinate": coordinate,
                                            "candidate": candidate, "comparator": comparator,
                                            "mse_difference": left - right,
                                            "observed_loss": left > right})
    decisions = []
    for candidate in CANDIDATES:
        primary = [row for row in contrasts if row["candidate"] == candidate
                   and row["comparator"] == "unguided_global"]
        losses = [row for row in heuristic_table if row["candidate"] == candidate and row["observed_loss"]]
        decisions.append({"candidate": candidate, "numerically_viable": True,
                          "primary_supported_gains": sum(row["verdict"] == "supported_gain" for row in primary),
                          "primary_supported_harms": sum(row["verdict"] == "supported_harm" for row in primary),
                          "heuristic_losses": len(losses),
                          "promotion_veto": bool(losses) or any(row["verdict"] == "supported_harm" for row in primary),
                          "promoted": False})
    attribution = []
    for condition in CONDITIONS:
        for coordinate in range(2):
            for mixture in ("global", "equal"):
                old_exact = summary_index[(condition, coordinate, f"mixture_future_{mixture}")]["mse"]
                old_fit = summary_index[(condition, coordinate, f"fitted_kernel_{mixture}")]["mse"]
                new_exact = summary_index[(condition, coordinate, f"exact_product_{mixture}")]["mse"]
                new_fit = summary_index[(condition, coordinate, f"product_kernel_{mixture}")]["mse"]
                attribution.append({"condition": condition, "coordinate": coordinate, "mixture": mixture,
                                    "message_effect_old_consumer": old_exact - old_fit,
                                    "message_effect_product_consumer": new_exact - new_fit,
                                    "consumer_effect_exact_message": new_exact - old_exact,
                                    "consumer_effect_fitted_message": new_fit - old_fit,
                                    "message_consumer_interaction": (new_exact - new_fit) - (old_exact - old_fit),
                                    "inference_status": "descriptive attribution only"})
    analysis = {"schema": "message_consumer_analysis_v1", "terminal_audit": "PASS",
                "rows": len(rows), "trace_checks": trace_checks, "fits": result["fits_loaded"],
                "source_hashes_checked": len(manifest["source_hashes"]),
                "input_hashes_checked": len(manifest["input_sha256"]),
                "finite_difference_checks": 48,
                "max_finite_difference_error": max(check["error"] for check in result["finite_differences"]),
                "inference": {"method": "paired cluster t intervals, Bonferroni 36 contrasts, simultaneous 95%",
                              "clusters": 8, "degrees_of_freedom": 7, "critical": critical,
                              "limitation": "small-cluster approximation conditional on saved fits and fixed known datasets"},
                "summaries": summaries, "contrasts": contrasts, "heuristic_table": heuristic_table,
                "decisions": decisions, "attribution": attribution,
                "heuristic_dominance_verdict": "FAIL" if any(row["observed_loss"] for row in heuristic_table) else "PASS",
                "continuation_veto": False, "default_ready": False,
                "nonclaims": manifest["nonclaims"],
                "provenance": {name: sha256(args.run / name) for name in ("manifest.json", "result.json", "rows.jsonl")},
                "analysis_sha256": sha256(Path(__file__)),
                "execution_role": "CPU-only post-run diagnostic; GPU devices intentionally hidden",
                "wall_seconds": time.monotonic() - started}
    with destination.open("x") as stream:
        json.dump(analysis, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"terminal_audit": analysis["terminal_audit"], "rows": len(rows),
                      "trace_checks": trace_checks, "heuristic_dominance": analysis["heuristic_dominance_verdict"],
                      "decisions": decisions, "analysis": str(destination),
                      "wall_seconds": analysis["wall_seconds"]}, allow_nan=False))


if __name__ == "__main__":
    main()
