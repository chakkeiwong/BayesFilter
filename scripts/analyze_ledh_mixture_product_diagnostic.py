"""Independent post-run audit and paired statistical diagnostic; no runtime use."""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
HEURISTICS = ("unguided_identity", "unguided_global", "unguided_local", "unguided_equal",
              "gaussian_future_global", "gaussian_future_equal")
CANDIDATES = ("product_kernel_global", "product_kernel_equal")
ARMS = HEURISTICS + ("mixture_future_global", "mixture_future_equal",
                     "fitted_kernel_global", "fitted_kernel_equal") + CANDIDATES
CONDITIONS = ("ordinary", "negative_future_tail", "positive_future_tail")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value):
    if isinstance(value, list):
        return all(finite_tree(child) for child in value)
    return isinstance(value, (float, int, bool)) and math.isfinite(value)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run / "analysis.json"
    if destination.exists():
        raise FileExistsError(destination)
    from scipy.stats import t
    manifest = json.loads((args.run / "manifest.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    require(manifest["status"] == result["status"] == "complete", "incomplete comparison")
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    require(len(rows) == result["rows_completed"] == 2304, "row coverage mismatch")
    require(result["fits_completed"] == 24, "fit count mismatch")
    require(manifest["jit_compile"] and not manifest["tf32_enabled"], "wrong backend scope")
    require(manifest["repaired_controls"]["global_guide_conditioning"] == "mixture_product_moments",
            "wrong repaired consumer")
    for path, expected in manifest["source_hashes"].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, f"source drift: {path}")
    parsed = ast.parse((ROOT / "bayesfilter/highdim/covariance_proposal_tf.py").read_text())
    fields = next(ast.literal_eval(node.value) for node in parsed.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id == "TRACE_FIELDS" for target in node.targets))
    indexed = {}
    trace_checks = 0
    for row in rows:
        key = (row["condition"], row["cluster"], row["replicate"], row["arm"])
        require(key not in indexed, "duplicate row")
        indexed[key] = row
        require(row["status"] == "valid" and row["valid"] and all(row["direction_valid"]), "invalid row")
        require("GPU" in row["value_device"], "non-GPU row")
        require(len(row["score"]) == len(row["direction_trace"]) == 2, "score direction mismatch")
        for field in ("value", "score", "trace", "direction_trace", "ancestor_history"):
            require(finite_tree(row[field]), f"nonfinite {field}")
        for trace in [row["trace"], *row["direction_trace"]]:
            require(len(trace) == 2, "wrong horizon")
            for step in trace:
                require(len(step) == len(fields), "trace schema mismatch")
                for flag in ("valid", "moment_valid", "correction_cap_valid"):
                    require(step[fields.index(flag)] == 1., f"trace veto: {flag}")
                require(step[fields.index("guide_margin")] > 0, "invalid guide covariance")
                trace_checks += 1
    for condition_index, condition in enumerate(CONDITIONS):
        reference = result["references"][condition]
        require(reference["value_difference"] <= 1e-6 and reference["score_difference"] <= 1e-5,
                "reference disagreement")
        for cluster in range(8):
            fit = json.loads((args.run / f"fit-{condition}-{cluster}.json").read_text())
            require(fit["seed"] == manifest["seed_design"]["fits"][condition_index][cluster], "fit seed mismatch")
            require(fit["fit"]["guide_eligible"], "ineligible fit")
            for replicate in range(8):
                paired = [indexed[(condition, cluster, replicate, arm)] for arm in ARMS]
                require(len({row["cloud_sha256"] for row in paired}) == 1, "unpaired clouds")
                require({row["seed"] for row in paired} == {
                    manifest["seed_design"]["clouds"][condition_index][cluster][replicate]}, "cloud seed mismatch")
                fitted = [row for row in paired if row["arm"].startswith(("fitted_", "product_"))]
                require(len({row["guide_sha256"] for row in fitted}) == 1, "unpaired fitted guides")
                require(all(row["fit_seed"] == fit["seed"] for row in fitted), "row fit seed mismatch")
    expected_fd = {(condition, arm, coordinate, step) for condition in CONDITIONS
                   for arm in CANDIDATES for coordinate in range(2) for step in (1e-4, 5e-5)}
    observed_fd = {(row["condition"], row["arm"], row["coordinate"], row["step"])
                   for row in result["finite_differences"]}
    require(observed_fd == expected_fd and len(observed_fd) == len(result["finite_differences"]),
            "finite-difference coverage mismatch")
    require(all(row["passed"] and row["error"] <= row["tolerance"] for row in result["finite_differences"]),
            "derivative disagreement")
    critical = float(t.ppf(1 - .05 / 48, 7))
    summaries, contrasts, heuristic_table = [], [], []
    for condition in CONDITIONS:
        for coordinate in range(2):
            reference = result["references"][condition]["score"][coordinate]
            errors = {arm: [[indexed[(condition, cluster, replicate, arm)]["score"][coordinate] - reference
                             for replicate in range(8)] for cluster in range(8)] for arm in ARMS}
            mse = {}
            for arm in ARMS:
                flat = [error for cluster in errors[arm] for error in cluster]
                mse[arm] = statistics.mean(error * error for error in flat)
                summaries.append({"condition": condition, "coordinate": coordinate, "arm": arm,
                                  "mse": mse[arm], "bias": statistics.mean(flat),
                                  "variance": statistics.pvariance(flat)})
            for arm in CANDIDATES:
                old = arm.replace("product_kernel", "fitted_kernel")
                for baseline in (old, "unguided_global"):
                    paired = [statistics.mean(candidate * candidate - comparator * comparator
                                              for candidate, comparator in zip(errors[arm][cluster], errors[baseline][cluster]))
                              for cluster in range(8)]
                    mean = statistics.mean(paired)
                    standard_error = statistics.stdev(paired) / math.sqrt(8)
                    lower, upper = mean - critical * standard_error, mean + critical * standard_error
                    contrasts.append({"condition": condition, "coordinate": coordinate, "arm": arm,
                                      "baseline": baseline, "cluster_differences": paired,
                                      "difference": mean, "standard_error": standard_error,
                                      "interval": [lower, upper], "supported_improvement": upper < 0,
                                      "supported_harm": lower > 0})
                for heuristic in HEURISTICS:
                    heuristic_table.append({"condition": condition, "coordinate": coordinate, "arm": arm,
                                            "heuristic": heuristic, "candidate_mse": mse[arm],
                                            "heuristic_mse": mse[heuristic], "promotion_veto": mse[arm] > mse[heuristic]})
    decisions = {}
    for arm in CANDIDATES:
        subset = [row for row in contrasts if row["arm"] == arm]
        losses = sum(row["promotion_veto"] for row in heuristic_table if row["arm"] == arm)
        decisions[arm] = {"supported_improvements": sum(row["supported_improvement"] for row in subset),
                          "supported_harms": sum(row["supported_harm"] for row in subset),
                          "heuristic_losses": losses,
                          "all_primary_cells_pass": all(row["supported_improvement"] for row in subset),
                          "promoted": all(row["supported_improvement"] for row in subset) and not losses,
                          "numerically_viable": True}
    analyzed = {"terminal_audit": "pass", "rows": len(rows), "trace_checks": trace_checks,
                "source_hashes_checked": len(manifest["source_hashes"]), "finite_difference_checks": len(observed_fd),
                "maximum_fd_error": max(row["error"] for row in result["finite_differences"]),
                "interval_method": "paired cluster Student t, df7, Bonferroni24, simultaneous95%",
                "critical_value": critical, "conditional_summaries": summaries, "contrasts": contrasts,
                "conditional_heuristic_table": heuristic_table, "decisions": decisions,
                "heuristic_dominance_verdict": "promotion_veto" if any(row["promotion_veto"] for row in heuristic_table) else "no_observed_loss",
                "interpretation_limit": "eight clusters and fixed previously inspected datasets; t approximation, no population claim",
                "input_hashes": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                                 for name in ("manifest.json", "result.json", "rows.jsonl")},
                "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cpu_only": "post-run standard-library calculations and SciPy reference t quantile; no TF/GPU import",
                "wall_seconds": time.monotonic() - started}
    destination.write_text(json.dumps(analyzed, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": analyzed["terminal_audit"], "decisions": decisions,
                      "max_fd_error": analyzed["maximum_fd_error"]}))


if __name__ == "__main__":
    main()
