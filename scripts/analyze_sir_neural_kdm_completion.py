"""Post-run diagnostic analysis of the frozen SIR neural and restricted studies."""

import argparse
import json
import math
from pathlib import Path

from bayesfilter.score_study.neural_moment_evidence import (
    HEURISTIC_ARMS, PARTICLE_ARMS, paired_interval, validate_holdout,
)


RESTRICTED_ARMS = ("csmc_unadapted", "csmc_adapted", "rbf_unadapted", "rbf_adapted")


def analyze_completion(neural_rows, restricted_rows, datasets):
    indexed, regimes = validate_holdout(neural_rows, datasets)
    if any(row.get("arm") not in RESTRICTED_ARMS for row in restricted_rows):
        raise ValueError("Unexpected restricted arm")
    for arm in RESTRICTED_ARMS:
        replacement = [dict(row, arm="trained") for row in restricted_rows if row["arm"] == arm]
        checked, _ = validate_holdout(
            [row for row in neural_rows if row["arm"] != "trained"] + replacement, datasets)
        indexed.update({(dataset, arm, replicate): checked[dataset, "trained", replicate]
                        for dataset in datasets for replicate in range(4)})
    subsets = dict(all=datasets)
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][0] == name]
                    for name in ("low", "high")})
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][1] == name]
                    for name in ("nearer_boundary", "interior")})
    if any(not subset for subset in subsets.values()):
        raise ValueError("A required comparison regime is absent")
    arms = PARTICLE_ARMS + HEURISTIC_ARMS + RESTRICTED_ARMS
    averaged = {}
    for dataset in datasets:
        for arm in arms:
            replicates = [None] if arm in HEURISTIC_ARMS else range(4)
            records = [indexed[dataset, arm, replicate] for replicate in replicates]
            averaged[dataset, arm] = [math.fsum(row["loss"] for row in records) / len(records)]
            averaged[dataset, arm] += [math.fsum(row["blocks"][block] for row in records) / len(records)
                                      for block in range(2)]
    summaries, comparisons = [], []
    pairs = [("trained", arm) for arm in RESTRICTED_ARMS]
    pairs += [(f"{family}_adapted", f"{family}_unadapted") for family in ("csmc", "rbf")]
    pairs += [(arm, comparator) for arm in RESTRICTED_ARMS
              for comparator in ("plain", "conditional") + HEURISTIC_ARMS]
    for regime, subset in subsets.items():
        for arm in arms:
            means = [math.fsum(averaged[dataset, arm][block] for dataset in subset) / len(subset)
                     for block in range(3)]
            summaries.append(dict(regime=regime, datasets=len(subset), arm=arm,
                                  loss=means[0], blocks=means[1:]))
        for candidate, comparator in pairs:
            differences = [averaged[dataset, candidate][0] - averaged[dataset, comparator][0]
                           for dataset in subset]
            comparisons.append(dict(regime=regime, datasets=len(subset), candidate=candidate,
                comparator=comparator, mean_difference=math.fsum(differences) / len(subset),
                paired_cluster_interval95=paired_interval(differences), role="explanatory only"))
    vetoes = {arm: [row for row in comparisons if row["candidate"] == arm
                   and row["comparator"] in ("conditional",) + HEURISTIC_ARMS
                   and row["mean_difference"] > 0.] for arm in RESTRICTED_ARMS}
    return dict(status="complete_paired_diagnostic", rows=len(neural_rows) + len(restricted_rows),
        regime_counts={name: len(subset) for name, subset in subsets.items()},
        summaries=summaries, comparisons=comparisons,
        heuristic_dominance={arm: dict(passed=not failures, failures=failures)
                             for arm, failures in vetoes.items()},
        neural_primary_decision_unchanged=True, default_ready=False,
        model_score_accuracy_evaluated=False,
        bootstrap=dict(draws=5000, seed=263045001, unit="dataset with all four particle replications",
            limitation="conditional on frozen fits; exploratory intervals without multiplicity adjustment"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-study", type=Path, required=True)
    parser.add_argument("--restricted-study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    for directory in (arguments.neural_study, arguments.restricted_study):
        result = json.loads((directory / "result.json").read_text())
        if result.get("source_check", {}).get("unchanged") is not True:
            raise ValueError("Completed study must verify unchanged package source")
        if (directory / "failure.json").exists():
            raise ValueError("Failed study cannot support completed comparison")
    neural = [json.loads(line) for line in (arguments.neural_study / "holdout.jsonl").read_text().splitlines()]
    restricted = [json.loads(line) for line in (arguments.restricted_study / "holdout.jsonl").read_text().splitlines()]
    result = analyze_completion(neural, restricted, list(range(263043001, 263043065)))
    result["inputs"] = dict(neural=str(arguments.neural_study), restricted=str(arguments.restricted_study))
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
