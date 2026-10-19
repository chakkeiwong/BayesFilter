"""Post-run diagnostic reporting for supervised guidance pilots."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def summarize(rows, selection):
    test_rows = [row for row in rows if row["partition"] == "test"]
    methods = sorted({row["method"] for row in rows})
    failures = {}
    for method in methods:
        failed = [row for row in rows if row["method"] == method and not row["valid"]]
        failures[method] = dict(rows=len(failed), stages=dict(Counter(
            key for row in failed for key in {key for check in row["checks"]
                                            for key, valid in check.items() if not valid})))
    paired = {}
    for method in methods:
        datasets = []
        for seed in sorted({row["observation_seed"] for row in test_rows}):
            candidate = {row["particle_seed"]:row for row in test_rows
                if row["observation_seed"] == seed and row["method"] == method}
            baseline = {row["particle_seed"]:row for row in test_rows
                if row["observation_seed"] == seed and row["method"] == "unguided"}
            if not candidate or candidate.keys() != baseline.keys():
                raise ValueError("Incomplete paired test rows")
            valid = all(row["valid"] for row in list(candidate.values()) + list(baseline.values()))
            item = dict(observation_seed=seed, valid=valid)
            if valid:
                item["component_loss_difference"] = [statistics.mean(
                    candidate[particle]["loss"][coordinate]-baseline[particle]["loss"][coordinate]
                    for particle in candidate) for coordinate in range(3)]
                item["total_loss_difference"] = sum(item["component_loss_difference"])
                item["terminal_raw_moment_loss_difference"] = statistics.mean(
                    sum(candidate[particle]["terminal_moment_squared_error"])
                    -sum(baseline[particle]["terminal_moment_squared_error"]) for particle in candidate)
            datasets.append(item)
        entry = dict(datasets=datasets, all_valid=all(item["valid"] for item in datasets))
        if entry["all_valid"]:
            differences = [item["total_loss_difference"] for item in datasets]
            entry.update(mean_total_loss_difference=statistics.mean(differences),
                         range=[min(differences), max(differences)],
                         mean_component_loss_difference=[statistics.mean(
                             item["component_loss_difference"][coordinate] for item in datasets) for coordinate in range(3)])
        paired[method] = entry
    return dict(failures=failures, paired_against_unguided=paired, selection=selection,
        inference_status=dict(hard_veto="invalid candidate rows rejected",
            ranking="none: pilot differences descriptive only",
            default_readiness=False, heuristic_dominance="not_established",
            next_evidence="fresh scope-specific validation and adequately replicated untouched datasets"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    rows = [json.loads(line) for line in (arguments.root / "rows.jsonl").read_text().splitlines()]
    selection = json.loads((arguments.root / "selection.json").read_text())
    result = summarize(rows, selection)
    (arguments.root / "paired-analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: {key:value for key,value in entry.items() if key != "datasets"}
                      for name,entry in result["paired_against_unguided"].items()}, indent=2))


if __name__ == "__main__":
    main()
