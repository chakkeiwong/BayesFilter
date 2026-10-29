"""Post-run diagnostic reporting, never a runtime selector or admission issuer."""

import argparse
import json
import random
import statistics
from pathlib import Path


ARMS = ("identity", "global", "local", "equal", "current", "fitted", "exact")
HEURISTICS = ("identity", "global", "local", "equal")
COORDINATES = ("drift_0", "drift_1", "log_process_sd", "log_observation_sd")


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def analyze(rows, *, resamples=10000):
    if resamples < 2:
        raise ValueError("at least two bootstrap resamples required")
    expected = {(condition, partition, arm): count
                for condition in range(4)
                for partition, count in (("validation", 2), ("final", 12))
                for arm in ARMS}
    groups = {key: {} for key in expected}
    for row in rows:
        key = (row["condition"], row["partition"], row["arm"])
        if key not in groups or row["seed"] in groups[key]:
            raise ValueError("unexpected or duplicate evaluation row")
        groups[key][row["seed"]] = row
        if row["trace_min"]["valid"] != 1.:
            raise ValueError("invalid numerical row cannot support comparison")
    if any(len(groups[key]) != count for key, count in expected.items()):
        raise ValueError("incomplete comparison; no missing rows may be dropped")
    summaries, pairs = [], []
    losses = []
    for condition in range(4):
        for partition in ("validation", "final"):
            seeds = set(groups[(condition, partition, "current")])
            if any(set(groups[(condition, partition, arm)]) != seeds for arm in ARMS):
                raise ValueError("unpaired arm seeds")
            for seed in seeds:
                oracle = groups[(condition, partition, "current")][seed]["reference_score"]
                if any(groups[(condition, partition, arm)][seed]["reference_score"] != oracle
                       for arm in ARMS):
                    raise ValueError("arms do not share the same exact reference")
        seeds = sorted(groups[(condition, "final", "current")])
        generator = random.Random(107000 + condition)
        draws = [[generator.randrange(len(seeds)) for _ in seeds] for _ in range(resamples)]
        for arm in ARMS:
            selected = [groups[(condition, "final", arm)][seed] for seed in seeds]
            summaries.append({
                "condition": condition, "arm": arm, "datasets": len(seeds),
                "mean_squared_error": [statistics.mean(row["squared_error"][coordinate] for row in selected)
                                       for coordinate in range(4)],
                "mean_value_squared_error": statistics.mean(row["value_error"] ** 2 for row in selected),
                "total_evaluation_seconds": sum(row["evaluation_seconds"] for row in selected),
            })
        for candidate in ("fitted", "exact"):
            for comparator in (*HEURISTICS, "current"):
                differences = [[groups[(condition, "final", candidate)][seed]["squared_error"][coordinate]
                                - groups[(condition, "final", comparator)][seed]["squared_error"][coordinate]
                                for seed in seeds] for coordinate in range(4)]
                intervals = []
                for values in differences:
                    bootstrap = [statistics.mean(values[index] for index in draw) for draw in draws]
                    intervals.append([quantile(bootstrap, .025), quantile(bootstrap, .975)])
                means = [statistics.mean(values) for values in differences]
                pairs.append({"condition": condition, "candidate": candidate,
                              "comparator": comparator, "mean_error_difference": means,
                              "paired_percentile_interval_95": intervals})
                if comparator in HEURISTICS and means[2] > 0.:
                    losses.append({"condition": condition, "candidate": candidate,
                                   "heuristic": comparator, "mean_difference": means[2],
                                   "interval": intervals[2]})
    primary = [pair for pair in pairs if pair["candidate"] == "fitted"]
    nominated = all(pair["paired_percentile_interval_95"][2][1] < 0. for pair in primary)
    return {
        "status": "complete_diagnostic", "coordinate_order": COORDINATES,
        "rows": len(rows), "resamples": resamples,
        "resampling_unit": "independent dataset plus independent fit/evaluation randomness; arms paired",
        "bootstrap_seed_rule": "107000+condition",
        "summaries": summaries, "paired_comparisons": pairs,
        "heuristic_losses": losses,
        "heuristic_dominance_verdict": "VETO" if any(item["candidate"] == "fitted" for item in losses) else "NO_DESCRIPTIVE_LOSS",
        "nomination_criterion_passed": nominated,
        "default_readiness": False, "canonical_admission": False,
        "inference_limit": "Twelve clusters per condition; bootstrap intervals are approximate diagnostic uncertainty, not broad superiority or bias/variance separation.",
        "cost_limit": "Evaluation costs are arm-specific; recorded fit time combines fitted and exact guides and calibration time combines three arms. No arm-specific total-cost ranking.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("do not overwrite diagnostic evidence")
    manifest = json.loads((arguments.input / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("campaign incomplete; inspect its failure before analysis")
    rows = [json.loads(line) for line in (arguments.input / "rows.jsonl").read_text().splitlines()]
    result = analyze(rows)
    result["campaign_wall_seconds"] = manifest["wall_seconds"]
    result["campaign_timings"] = manifest["timings"]
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "rows", "heuristic_dominance_verdict", "nomination_criterion_passed")}))


if __name__ == "__main__":
    main()
