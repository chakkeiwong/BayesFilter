"""Independent CPU diagnostic audit and paired fixed-fit particle-count analysis."""

import argparse
import ast
import json
import math
import os
from pathlib import Path
import statistics
import time

from analyze_ledh_message_consumer_diagnostic import HEURISTICS, digest, finite_tree, require, sha256

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("product_kernel_global", "exact_product_global")
ARMS = HEURISTICS + CANDIDATES


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "analysis must explicitly hide GPU")
    require(not (args.run / "analysis.json").exists(), "analysis artifact already exists")
    manifest = json.loads((args.run / "manifest.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    require(manifest["status"] == result["status"] == "complete", "run incomplete")
    designs = {"positive_tail_n32_n64_r6_v1": ((32, 64), 6, 861800),
               "positive_tail_n64_n128_r12_v1": ((64, 128), 12, 862800)}
    design_id = manifest.get("design_id", "positive_tail_n32_n64_r6_v1")
    require(design_id in designs, "unreviewed design")
    counts, replicates, seed_base = designs[design_id]
    require(manifest["particle_counts"] == manifest["chunk_extents"] == list(counts), "wrong scope")
    require(manifest.get("cloud_replicates", 6) == replicates
            and manifest.get("cloud_seed_base", 861800) == seed_base, "wrong seed design")
    require(manifest["horizon"] == 2 and manifest["observations"] == [[.2], [5.]], "wrong data")
    require(manifest["jit_compile"] and manifest["dtype"] == "float64"
            and not manifest["tf32_enabled"], "wrong execution policy")
    require(manifest["score_route"] == "analytical_recursive_frozen_guide_total_derivative", "wrong score")
    require(not manifest["canonical_ledh"] and not manifest["default_ready"], "diagnostic scope changed")
    controls = manifest["controls"]
    for key, expected in (("reset_steps", 5), ("reset_epsilon", 1.), ("correction_steps", 2),
                          ("correction_basis", 4), ("correction_rate", .1),
                          ("correction_radius", .25)):
        require(controls[key] == expected, f"control changed: {key}")
    require(controls["global_guide_conditioning"] == "positive_information", "wrong baseline consumer")
    require(manifest["product_controls"] == {**controls, "global_guide_conditioning": "mixture_product_moments"},
            "product controls differ")
    require(sha256(ROOT / manifest["plan"]) == manifest["plan_sha256"], "plan changed")
    for name, expected in manifest["source_hashes"].items():
        require(sha256(ROOT / name) == expected, f"source changed: {name}")
    for name, expected in manifest["input_sha256"].items():
        require(sha256(Path(name)) == expected, f"input changed: {name}")
    require(result["fits_loaded"] == len(manifest["fit_inputs"]) == 8, "fit coverage mismatch")
    fit_digests = {}
    for cluster in range(8):
        name = f"fit-positive_future_tail-{cluster}.json"
        require(sha256(args.run / name) == manifest["fit_inputs"][name]["sha256"], "copied fit changed")
        fit = json.loads((args.run / name).read_text())
        require(fit["seed"] == 822000 + cluster and fit["fit"]["guide_eligible"], "fit mismatch")
        fit_digests[cluster] = digest([fit["fit"][key] for key in ("logits", "centers", "covariances", "active")])
    syntax = ast.parse((ROOT / "bayesfilter/highdim/covariance_proposal_tf.py").read_text())
    fields = next(ast.literal_eval(node.value) for node in syntax.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "TRACE_FIELDS" for target in node.targets))
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    require(len(rows) == result["rows_completed"] == 8 * replicates * len(counts) * len(ARMS), "row coverage mismatch")
    indexed = {}
    hashes = {}
    parent_hashes = {}
    exact_hashes = set()
    trace_checks = 0
    beta_by_arm = {"unguided_identity": [1., 0., 0.], "unguided_local": [.1, 0., .9],
                   "unguided_equal": [1 / 3] * 3, "gaussian_future_equal": [1 / 3] * 3}
    for row in rows:
        count, cluster, replicate, arm = (row[key] for key in ("particle_count", "cluster", "replicate", "arm"))
        key = (count, cluster, replicate, arm)
        require(key not in indexed and count in counts and cluster in range(8)
                and replicate in range(replicates) and arm in ARMS, "invalid or duplicate row")
        require(row["seed"] == seed_base + 100 * cluster + replicate and row["fit_seed"] == 822000 + cluster,
                "seed mismatch")
        require(row["chunk_extent"] == count and row["condition"] == "positive_future_tail", "row scope")
        require(row["beta"] == beta_by_arm.get(arm, [.1, .9, 0.]), "beta changed")
        expected_consumer = "mixture_product_moments" if arm in CANDIDATES else "positive_information"
        require(row["global_guide_conditioning"] == expected_consumer, "consumer mismatch")
        require(row["status"] == "valid" and row["valid"] and all(row["direction_valid"]), "invalid row")
        require("GPU" in row["value_device"], "CPU kernel")
        for field in ("value", "score", "trace", "direction_trace", "ancestor_history"):
            require(finite_tree(row[field]), f"nonfinite {field}")
        require(len(row["score"]) == len(row["direction_trace"]) == len(row["direction_valid"]) == 2,
                "score shape")
        for trace in [row["trace"], *row["direction_trace"]]:
            require(len(trace) == 2, "trace horizon")
            for step in trace:
                require(len(step) == len(fields), "trace field coverage")
                values = dict(zip(fields, step))
                require(all(values[name] == 1 for name in ("valid", "moment_valid", "correction_cap_valid")),
                        "invalid numerical trace")
                require(values["guide_margin"] > 0, "invalid guide factorization")
                trace_checks += 1
        hashes.setdefault((count, cluster, replicate), set()).add(row["cloud_sha256"])
        parent_hashes.setdefault((cluster, replicate), set()).add(row["parent_cloud_sha256"])
        if count == counts[-1]:
            require(row["cloud_sha256"] == row["parent_cloud_sha256"], "parent cloud mismatch")
        if arm == "product_kernel_global":
            require(row["guide_sha256"] == fit_digests[cluster], "fit consumer mismatch")
        if arm == "exact_product_global":
            exact_hashes.add(row["guide_sha256"])
        indexed[key] = row
    require(len(hashes) == len(counts) * 8 * replicates and len(parent_hashes) == 8 * replicates
            and all(len(value) == 1 for value in [*hashes.values(), *parent_hashes.values()]), "broken pairing")
    require(len(exact_hashes) == 1, "exact message changed")
    checks = result["finite_differences"]
    require(len(checks) == 16, "finite difference coverage")
    checked_keys = set()
    for check in checks:
        key = tuple(check[name] for name in ("particle_count", "arm", "coordinate", "step"))
        require(key not in checked_keys, "duplicate derivative check")
        checked_keys.add(key)
        require(key[0] in counts and key[1] in CANDIDATES and key[2] in (0, 1)
                and key[3] in (1e-4, 5e-5), "unexpected derivative check")
        analytical = indexed[(key[0], 0, 0, key[1])]["score"][key[2]]
        error = abs(check["difference"] - analytical)
        tolerance = 2e-7 + 2e-5 * abs(check["difference"])
        require(check["passed"] and math.isclose(error, check["error"], abs_tol=1e-15)
                and error <= tolerance, "finite-program derivative mismatch")
    reference = result["references"]["positive_future_tail"]
    require(finite_tree(reference), "nonfinite reference")
    require(abs(reference["value"] - reference["grid"]["value"]) <= 1e-6
            and max(abs(left - right) for left, right in zip(reference["score"], reference["grid"]["score"])) <= 1e-5,
            "reference disagreement")
    summaries = []
    errors = {}
    for count in counts:
        for coordinate in range(2):
            for arm in ARMS:
                residuals = [indexed[(count, cluster, replicate, arm)]["score"][coordinate] - reference["score"][coordinate]
                             for cluster in range(8) for replicate in range(replicates)]
                squared = [value * value for value in residuals]
                bias, variance, mse = statistics.mean(residuals), statistics.pvariance(residuals), statistics.mean(squared)
                require(math.isclose(mse, bias * bias + variance, rel_tol=1e-12, abs_tol=1e-14), "bias variance identity")
                summaries.append({"particle_count": count, "coordinate": coordinate, "arm": arm,
                                  "bias": bias, "variance": variance, "mse": mse})
                for cluster in range(8):
                    errors[(count, coordinate, arm, cluster)] = statistics.mean(
                        squared[replicates * cluster:replicates * (cluster + 1)])
    from scipy.stats import t
    critical = float(t.ppf(1 - .05 / (2 * 14), 7))
    contrasts = []
    for candidate in CANDIDATES:
        comparators = [(counts[0], candidate)] + [(counts[-1], heuristic) for heuristic in HEURISTICS]
        for baseline_count, baseline in comparators:
            differences = [errors[(counts[-1], 1, candidate, cluster)] - errors[(baseline_count, 1, baseline, cluster)]
                           for cluster in range(8)]
            mean = statistics.mean(differences)
            radius = critical * statistics.stdev(differences) / math.sqrt(8)
            contrasts.append({"candidate": candidate, "candidate_count": counts[-1], "baseline": baseline,
                              "baseline_count": baseline_count, "coordinate": 1, "difference": mean,
                              "interval": [mean - radius, mean + radius], "cluster_differences": differences,
                              "verdict": "gain" if mean + radius < 0 else "harm" if mean - radius > 0 else "unresolved"})
    heuristic_table = []
    for count in counts:
        for coordinate in range(2):
            for candidate in CANDIDATES:
                for heuristic in HEURISTICS:
                    candidate_mse = statistics.mean(errors[(count, coordinate, candidate, cluster)] for cluster in range(8))
                    heuristic_mse = statistics.mean(errors[(count, coordinate, heuristic, cluster)] for cluster in range(8))
                    heuristic_table.append({"particle_count": count, "coordinate": coordinate, "candidate": candidate,
                                            "heuristic": heuristic, "candidate_mse": candidate_mse,
                                            "heuristic_mse": heuristic_mse, "descriptive_loss": candidate_mse > heuristic_mse})
    losses = [row for row in heuristic_table if row["particle_count"] == counts[-1] and row["descriptive_loss"]]
    analysis = {"schema": "particle_count_stress_analysis_v1", "terminal_audit": "PASS", "rows": len(rows),
                "trace_checks": trace_checks, "fits_checked": 8, "finite_differences": len(checks),
                "finite_difference_max_error": max(check["error"] for check in checks),
                "source_hashes_checked": len(manifest["source_hashes"]),
                "input_hashes_checked": len(manifest["input_sha256"]), "summaries": summaries,
                "contrasts": contrasts, "heuristic_table": heuristic_table,
                "heuristic_dominance": {"verdict": "FAIL" if losses else "PASS", "losses_at_larger_count": len(losses),
                                        "larger_particle_count": counts[-1],
                                        "role": "promotion veto only; observed MSE comparisons are descriptive"},
                "inference": {"scope": f"eight fixed fits, {replicates} paired clouds per fit; conditional on fixed observations",
                              "method": "paired cluster means, t7, Bonferroni 95% over 14 contrasts",
                              "gain": sum(row["verdict"] == "gain" for row in contrasts),
                              "harm": sum(row["verdict"] == "harm" for row in contrasts),
                              "unresolved": sum(row["verdict"] == "unresolved" for row in contrasts)},
                "continuation_veto": False, "default_ready": False, "canonical_ledh": False,
                "next_action": "interpret particle-count contrast; persistent error triggers consumer/reset diagnosis",
                "nonclaims": ["canonical LEDH conformance", "default readiness", "fresh-fit or population ranking",
                              "particle-count convergence", "original KDM replication", "HMC readiness"],
                "gpu_intentionally_hidden_for_analysis": True,
                "provenance": {name: sha256(args.run / name) for name in ("manifest.json", "result.json", "rows.jsonl")},
                "wall_seconds": time.monotonic() - started}
    require(finite_tree(analysis), "nonfinite analysis")
    (args.run / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: analysis[key] for key in ("terminal_audit", "rows", "trace_checks", "finite_differences",
                                                   "finite_difference_max_error", "inference", "heuristic_dominance", "wall_seconds")}))


if __name__ == "__main__":
    main()
