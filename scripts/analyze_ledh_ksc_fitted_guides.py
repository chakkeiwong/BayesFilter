"""Post-run diagnostic analysis; SciPy supplies only the reference t quantile."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

CONDITIONS = ("ordinary", "negative_future_tail", "positive_future_tail")
BASELINE = "unguided_global"
HEURISTICS = ("unguided_identity", "unguided_global", "unguided_local", "unguided_equal",
              "gaussian_observation_approximation")
FITTED = tuple(f"fitted_{family}_{mixture}" for family in ("quadratic", "kernel")
               for mixture in ("global", "equal"))
CONTROLS = ("unguided_identity", "unguided_global", "unguided_local", "unguided_equal",
            "gaussian_future_global", "gaussian_future_equal",
            "mixture_future_global", "mixture_future_equal")


def analyze(rows, references, critical_value):
    indexed = {}
    cloud_hashes = {}
    for row in rows:
        key = (row["condition"], row["cluster"], row["replicate"], row["arm"])
        if key in indexed:
            raise ValueError(f"duplicate comparison cell {key}")
        if row["status"] not in ("valid", "candidate_ineligible"):
            raise ValueError(f"unrecognized row status {key}")
        if row["status"] == "valid":
            if len(row["score"]) != 2 or not all(math.isfinite(value) for value in row["score"]):
                raise ValueError(f"invalid score {key}")
        elif row["arm"] not in FITTED:
            raise ValueError("baseline or exact control cannot be ineligible")
        indexed[key] = row
        cloud_key = key[:3]
        if cloud_key in cloud_hashes and cloud_hashes[cloud_key] != row["cloud_sha256"]:
            raise ValueError("unpaired particle clouds")
        cloud_hashes[cloud_key] = row["cloud_sha256"]
    expected = {(condition, cluster, replicate, arm) for condition in CONDITIONS
                for cluster in range(8) for replicate in range(8) for arm in CONTROLS + FITTED}
    if set(indexed) != expected:
        raise ValueError(f"comparison grid mismatch: missing={len(expected - set(indexed))}, extra={len(set(indexed) - expected)}")
    if len(set(cloud_hashes.values())) != 192:
        raise ValueError("final cloud hashes are not unique")
    metrics = []
    contrasts = []
    heuristic_losses = []
    for condition in CONDITIONS:
        for coordinate in range(2):
            reference = references[condition]["score"][coordinate]
            errors = {}
            for arm in CONTROLS + FITTED:
                arm_rows = [indexed[(condition, cluster, replicate, arm)]
                            for cluster in range(8) for replicate in range(8)]
                failures = sum(row["status"] != "valid" for row in arm_rows)
                metric = {"condition": condition, "coordinate": coordinate, "arm": arm,
                          "ineligible_rows": failures, "complete": failures == 0}
                if failures == 0:
                    signed = [row["score"][coordinate] - reference for row in arm_rows]
                    errors[arm] = [value * value for value in signed]
                    metric.update({"mse": statistics.mean(errors[arm]),
                                   "bias": statistics.mean(signed),
                                   "rmse": math.sqrt(statistics.mean(errors[arm]))})
                metrics.append(metric)
            gaussian_error = references[condition]["gaussian_approximation"]["score"][coordinate] - reference
            errors["gaussian_observation_approximation"] = [gaussian_error ** 2] * 64
            metrics.append({"condition": condition, "coordinate": coordinate,
                            "arm": "gaussian_observation_approximation",
                            "complete": True, "ineligible_rows": 0,
                            "mse": gaussian_error ** 2, "bias": gaussian_error, "rmse": abs(gaussian_error),
                            "target_warning": "different moment-matched observation target; heuristic only"})
            for arm in FITTED:
                contrast = {"condition": condition, "coordinate": coordinate, "arm": arm,
                            "comparator": BASELINE, "available": arm in errors}
                if arm in errors:
                    differences = [candidate - baseline for candidate, baseline in zip(errors[arm], errors[BASELINE])]
                    clusters = [statistics.mean(differences[8 * cluster:8 * cluster + 8]) for cluster in range(8)]
                    mean = statistics.mean(clusters)
                    standard_error = statistics.stdev(clusters) / math.sqrt(8)
                    margin = critical_value * standard_error
                    contrast.update({"cluster_means": clusters, "mean_mse_difference": mean,
                                     "standard_error": standard_error, "lower": mean - margin,
                                     "upper": mean + margin,
                                     "supported_improvement": mean + margin < 0.0,
                                     "supported_harm": mean - margin > 0.0})
                contrasts.append(contrast)
            for arm in CONTROLS[4:] + FITTED:
                if arm in errors:
                    for heuristic in HEURISTICS:
                        candidate_mse = statistics.mean(errors[arm])
                        heuristic_mse = statistics.mean(errors[heuristic])
                        if candidate_mse > heuristic_mse:
                            heuristic_losses.append({"condition": condition, "coordinate": coordinate,
                                                     "arm": arm, "heuristic": heuristic,
                                                     "candidate_mse": candidate_mse,
                                                     "heuristic_mse": heuristic_mse,
                                                     "evidence_class": "descriptive promotion veto, not established harm"})
    arm_decisions = {}
    for arm in FITTED:
        selected = [item for item in contrasts if item["arm"] == arm]
        failures = sum(item["ineligible_rows"] for item in metrics if item["arm"] == arm) // 2
        losses = [item for item in heuristic_losses if item["arm"] == arm]
        across_conditions = all(item.get("supported_improvement", False) for item in selected)
        arm_decisions[arm] = {"ineligible_rows": failures, "heuristic_losses": len(losses),
                              "all_six_intervals_favor_arm": across_conditions,
                              "promotion": across_conditions and not failures and not losses,
                              "status": "candidate_ineligible" if failures else (
                                  "heuristic_promotion_veto" if losses else "viable_no_default_claim")}
    return {"metrics": metrics, "contrasts": contrasts, "arm_decisions": arm_decisions,
            "heuristic_losses": heuristic_losses,
            "heuristic_dominance_verdict": "promotion_veto" if heuristic_losses else "no_observed_loss",
            "uncertainty": {"method": "paired cluster Student-t with Bonferroni simultaneous 95% intervals",
                            "family_size": 24, "clusters": 8, "clouds_per_cluster": 8,
                            "degrees_of_freedom": 7, "critical_value": critical_value,
                            "limitations": "Conditional on three fixed datasets and frozen tuning; eight clusters and approximate normality of cluster means. Other contrasts descriptive."}}


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    from scipy.stats import t
    result = json.loads((args.run / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("cannot interpret an incomplete or invalid run")
    rows = [json.loads(line) for line in (args.run / "rows.jsonl").read_text().splitlines()]
    analyzed = analyze(rows, result["references"], float(t.ppf(1.0 - 0.05 / 48.0, 7)))
    analyzed["input_hashes"] = {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                                 for name in ("result.json", "manifest.json", "rows.jsonl")}
    analyzed["cpu_only"] = "Post-run standard-library analysis; SciPy reference quantile; no GPU or TensorFlow import"
    analyzed["wall_seconds"] = time.monotonic() - started
    destination = args.run / "analysis.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(analyzed, indent=2, allow_nan=False) + "\n")
    supported = [item for item in analyzed["contrasts"] if item.get("supported_improvement") or item.get("supported_harm")]
    lines = ["# KSC fitted-guide final comparison", "",
             "All 2,304 planned comparison cells are preserved. Ineligible fits remain failures, not replacement seeds.",
             "", "## Decision table", "",
             "| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |",
             "|---|---|---|---|---|---|"]
    for arm, decision in analyzed["arm_decisions"].items():
        lines.append(f"| {arm}: {decision['status']} | All six upper bounds < 0: {decision['all_six_intervals_favor_arm']} | {decision['heuristic_losses']} descriptive heuristic losses; {decision['ineligible_rows']} ineligible rows | Eight fit clusters, fixed datasets | Diagnose fitting versus consumer behavior; no default change | Canonical admission, HMC readiness, universal benefit |")
    lines += ["", "## Inference status", "", "| Evidence | Status |", "|---|---|",
              "| Hard veto screen | Completed runner validates references, finite outputs, analytical derivatives and source stability; ineligible fits are candidate failures. |",
              f"| Statistically supported ranking | {len(supported)} of 24 simultaneous contrasts exclude zero; full condition/coordinate details in analysis.json. |",
              "| Descriptive-only differences | All heuristic, exact-message, loss, ESS and runtime comparisons lack a predeclared inferential ranking. |",
              "| Default-readiness | Not evaluated; untuned diagnostic consumer only. |",
              "| Next evidence needed | A repaired mechanism must pass fresh conditional score comparisons and independent canonical validation. |",
              "", "## Post-run red team", "",
              "The strongest alternative explanation is consumer or reset error rather than poor message fitting. Compare each fitted arm with its exact-message control before attributing failure to fitting capacity. Eight clusters make normal-approximation intervals fragile; these intervals are conditional on the frozen selection and the three fixed observation datasets. A fresh valid comparison reversing the conditional errors would overturn a local rejection. Candidate rejection is not rejection of the research direction."]
    (args.run / "analysis.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"analysis": str(destination), "decisions": analyzed["arm_decisions"],
                      "supported_contrasts": len(supported), "wall_seconds": analyzed["wall_seconds"]}))


if __name__ == "__main__":
    main()
