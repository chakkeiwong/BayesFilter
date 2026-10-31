"""Post-run diagnostic analysis only; never imported by the training runtime."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


HEURISTICS = ("unguided", "csmc_half", "csmc_original", "kernel_original")
LEARNED = ("csmc_trained", "kernel_trained")
METHODS = HEURISTICS + LEARNED


def source_hash_checks(source_hashes, source_files=None):
    source_files = {} if source_files is None else source_files
    if set(source_files) - set(source_hashes):
        raise ValueError("Source mapping contains paths absent from the manifest")
    return {name: hashlib.sha256(Path(source_files.get(name, name)).read_bytes()).hexdigest() == expected
            for name, expected in source_hashes.items()}


def paired_interval(differences, *, draws=10000, seed=262026001):
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("Paired uncertainty needs at least two finite independent datasets")
    indices = np.random.default_rng(seed).integers(len(differences), size=(draws, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    interval = np.quantile(bootstrap, [.00625, .99375]).tolist()
    return dict(mean=float(differences.mean()), interval=interval,
                confidence=.9875, datasets=len(differences), draws=draws,
                supported_reduction=interval[1] < 0)


def analyze_records(rows, dataset_seeds, *, draws=10000):
    expected = {(seed, 274000000 + 2 * (seed % 1000) + replicate, method)
                for seed in dataset_seeds for replicate in range(2) for method in METHODS}
    keys = [(row["dataset_seed"], row["particle_seed"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Test grid is incomplete, duplicated, or differs from the frozen scope")
    lookup = dict(zip(keys, rows))
    regimes = {}
    for seed in dataset_seeds:
        names = {row["regime"] for row in rows if row["dataset_seed"] == seed}
        if len(names) != 1 or not names <= {"low", "high"}:
            raise ValueError("Inconsistent observation regime")
        regimes[seed] = names.pop()
    invalid = {method: sum(not row["valid"] for row in rows if row["method"] == method)
               for method in METHODS}
    paired, summaries = {}, {}
    for method in METHODS:
        if invalid[method]:
            summaries[method] = dict(invalid_rows=invalid[method], means=None)
            continue
        paired[method] = {}
        for seed in dataset_seeds:
            replicas = [lookup[seed, 274000000 + 2 * (seed % 1000) + replicate, method]
                        for replicate in range(2)]
            fields = {name: np.mean([row[name] for row in replicas], axis=0)
                      for name in ("loss", "blocks", "raw_blocks", "score_loss")}
            if not all(np.isfinite(value).all() for value in fields.values()):
                raise ValueError("A row marked valid contains a nonfinite reported metric")
            paired[method][seed] = fields
        summaries[method] = dict(invalid_rows=0, means={name: np.mean(
            [paired[method][seed][name] for seed in dataset_seeds], axis=0).tolist()
            for name in ("loss", "blocks", "raw_blocks", "score_loss")})
    contrasts = {}
    for learned in LEARNED:
        original = learned.replace("_trained", "_original")
        for comparator in (original, "unguided"):
            name = learned + " - " + comparator
            contrasts[name] = None
            if learned in paired and comparator in paired:
                differences = [paired[learned][seed]["loss"] - paired[comparator][seed]["loss"]
                               for seed in dataset_seeds]
                contrasts[name] = paired_interval(differences, draws=draws)
    conditional, failures = {}, []
    for regime in ("low", "high"):
        seeds = [seed for seed in dataset_seeds if regimes[seed] == regime]
        conditional[regime] = dict(datasets=len(seeds), means={})
        if not seeds:
            continue
        conditional[regime]["means"] = {method: float(np.mean(
            [values[seed]["loss"] for seed in seeds])) for method, values in paired.items()}
        for learned in LEARNED:
            if learned not in paired:
                continue
            for heuristic in HEURISTICS:
                if heuristic not in paired:
                    continue
                difference = conditional[regime]["means"][learned] - conditional[regime]["means"][heuristic]
                if difference > 0:
                    failures.append(dict(method=learned, heuristic=heuristic,
                                         regime=regime, mean_difference=difference))
    heuristic_complete = all(conditional[regime]["datasets"] for regime in ("low", "high"))
    decisions = {}
    for learned in LEARNED:
        relevant = [value for name, value in contrasts.items() if name.startswith(learned)]
        primary = all(value is not None and value["supported_reduction"] for value in relevant)
        valid = not any(invalid.values())
        heuristic_pass = heuristic_complete and not any(row["method"] == learned for row in failures)
        decisions[learned] = dict(primary_pass=primary, validity_pass=valid,
            heuristic_pass=heuristic_pass, scoped_promotion=primary and valid and heuristic_pass)
    dataset_table = [{"dataset_seed": seed, "regime": regimes[seed], "methods": {
        method: {name: np.asarray(value).tolist() for name, value in values[seed].items()}
        for method, values in paired.items()}} for seed in dataset_seeds]
    return dict(summary=summaries, contrasts=contrasts, conditional=conditional,
        heuristic_dominance=dict(verdict="veto" if failures or not heuristic_complete or any(invalid.values())
                                 else "pass", failures=failures),
        decisions=decisions, datasets=dataset_table,
        inference_unit="independent simulated dataset; two particle replicates averaged first",
        bootstrap_seed=262026001, bootstrap_draws=draws, familywise_confidence=.95,
        score_metrics="explanatory only; not used for fitting or selection",
        default_ready=False, original_neural_kdm_training=False)


def render_report(result):
    invalid = [method for method, row in result["summary"].items() if row["invalid_rows"]]
    vetoed = [method for method, row in result["decisions"].items() if not row["heuristic_pass"]]
    passed = [method for method, row in result["decisions"].items() if row["scoped_promotion"]]
    if invalid:
        headline = "Invalid test rows block promotion; affected methods: " + ", ".join(invalid) + "."
    elif vetoed:
        headline = "Conditional heuristic comparisons veto promotion of " + ", ".join(vetoed) + "."
    elif passed:
        headline = ", ".join(passed) + " meets the predeclared moment-improvement criterion in this scope."
    else:
        headline = "The study does not establish the required moment-improvement comparisons."
    lines = ["# Simulated terminal moment training: heldout results", "", headline, "",
        "This evaluates the quadratic cSMC and KDM-inspired RBF guide adapters at SIR T=5, N=64.",
        "All losses use fresh simulated labels. Smaller population label loss implies smaller",
        "moment-estimation MSE under the fixed calibration; individual posterior moments remain unknown.", "",
        "| Method | Moment loss | First-feature loss | Second-feature loss | Invalid rows |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for method, row in result["summary"].items():
        means = row["means"]
        values = "unavailable | unavailable | unavailable" if means is None else (
            f"{means['loss']:.6g} | {means['blocks'][0]:.6g} | {means['blocks'][1]:.6g}")
        lines.append(f"| {method} | {values} | {row['invalid_rows']} |")
    lines += ["", "The unscaled moment and analytical-score label losses below are explanatory only.",
        "Their means have no uncertainty analysis and establish no ranking. The score column sums",
        "the three calibration-scaled coordinate losses; score labels never select coefficients.", "",
        "| Method | Raw first-moment label loss | Raw second-moment label loss | Score-label loss |",
        "| --- | ---: | ---: | ---: |"]
    for method, row in result["summary"].items():
        means = row["means"]
        values = "unavailable | unavailable | unavailable" if means is None else (
            f"{means['raw_blocks'][0]:.6g} | {means['raw_blocks'][1]:.6g} | {sum(means['score_loss']):.6g}")
        lines.append(f"| {method} | {values} |")
    lines += ["", "Means alone do not establish a ranking. The predeclared paired intervals are:", "",
        "| Trained minus comparator | Mean difference | 98.75% paired interval | Reduction supported |",
        "| --- | ---: | --- | --- |"]
    for name, row in result["contrasts"].items():
        if row is None:
            lines.append(f"| {name} | unavailable | validity veto | no |")
        else:
            lines.append(f"| {name} | {row['mean']:.6g} | [{row['interval'][0]:.6g}, "
                         f"{row['interval'][1]:.6g}] | {row['supported_reduction']} |")
    lines += ["", "The four intervals use 10,000 dataset-paired bootstrap draws and a Bonferroni",
              "family-wise nominal 95% level. They describe this fixed simulation scope only.", "",
              "| Observation regime | Datasets | " + " | ".join(METHODS) + " |",
              "| --- | ---: | " + " | ".join(["---:"] * len(METHODS)) + " |"]
    for regime, row in result["conditional"].items():
        values = [f"{row['means'][method]:.6g}" if method in row["means"] else "unavailable"
                  for method in METHODS]
        lines.append(f"| {regime} | {row['datasets']} | " + " | ".join(values) + " |")
    lines += ["", "Heuristic-dominance verdict: **" + result["heuristic_dominance"]["verdict"] + "**.",
        "Conditional mean underperformance is a predeclared promotion veto, not a statistical ranking.", ""]
    for row in result["heuristic_dominance"]["failures"]:
        lines.append(f"- {row['method']} loses descriptively to {row['heuristic']} in the "
                     f"{row['regime']} regime by {row['mean_difference']:.6g}.")
    lines += ["", "| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |"]
    for method, row in result["decisions"].items():
        lines.append(f"| {method}: {'scoped pass' if row['scoped_promotion'] else 'no promotion'} | "
            f"{'pass' if row['primary_pass'] else 'not met'} | validity={row['validity_pass']}, "
            f"heuristics={row['heuristic_pass']} | finite datasets and adapter capacity | "
            "inspect training and heldout discrepancies before any fresh study | broad score/HMC/default benefit |")
    supported = [name for name, row in result["contrasts"].items() if row and row["supported_reduction"]]
    lines += ["", "| Inference status | Finding |", "| --- | --- |",
        "| Hard veto screen | " + ("all test rows valid" if all(row["invalid_rows"] == 0
             for row in result["summary"].values()) else "invalid rows retained; no silent deletion") + " |",
        "| Statistically supported ranking | " + ("; ".join(supported) if supported else "none") + " |",
        "| Descriptive-only differences | raw moments, score-label losses and conditional means |",
        "| Default readiness | not established |",
        "| Next evidence needed | fresh datasets/scope and independent uncertainty after any new training protocol |", "",
        "The score-label metrics in result.json are explanatory and were excluded from training and selection.",
        "Raw second moments are reported separately from first moments; these are not covariance MSEs.", "",
        "The strongest alternative explanation for a negative result is limited adaptation capacity or noisy",
        "optimization. A positive result could remain specific to this short horizon and small particle count.",
        "This study neither rejects the population risk identity nor tests original neural KDM retraining.", ""]
    if "training" in result:
        lines += ["| Training run | Updates | Accepted | Bound hits | Invalid probe pairs | Invalid trials |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in result["training"].items():
            lines.append(f"| {name} | {row['updates']} | {row['accepted']} | {row['bound_hits']} | "
                         f"{row['invalid_probe_pairs']} | {row['invalid_trials']} |")
        lines += ["", "Frozen validation choices:", ""]
        for family, row in result["selection"]["selected"].items():
            lines.append(f"- {family}: {row['run']}, update {row['update']}, "
                         f"validation moment loss {row['validation_loss']:.6g}.")
        lines += ["", "Exact coefficients and all candidate checkpoints are in selection.json;",
                  "test.jsonl preserves all test rows. Runtime source hashes match the launch manifest.", ""]
    return "\n".join(lines)


def render_figures(directory, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(name, row) for name, row in result["contrasts"].items() if row is not None]
    if rows:
        figure, axes = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
        for index, (name, row) in enumerate(rows):
            axes.plot(row["interval"], [index, index], color="black", linewidth=2)
            axes.scatter([row["mean"]], [index], color="black", s=35)
        axes.axvline(0, color="grey", linestyle="--", linewidth=1)
        axes.set_yticks(range(len(rows)), [name.replace("_", " ") for name, _ in rows])
        axes.invert_yaxis()
        axes.set_xlabel("Paired terminal moment-loss difference (negative favors training)")
        axes.set_title("64 fresh SIR datasets: 98.75% paired bootstrap intervals")
        figure.savefig(directory / "heldout-moment-differences.pdf")
        figure.savefig(directory / "heldout-moment-differences.png", dpi=180)
        plt.close(figure)
    validation = [json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines()]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, family in zip(axes, ("csmc", "kernel")):
        runs = sorted({row["run"] for row in validation if row["family"] == family})
        for run in runs:
            checkpoints = [row for row in validation if row["run"] == run and row["validation_loss"] is not None]
            if not checkpoints:
                continue
            initial = checkpoints[0]["validation_loss"]
            label = run.replace(family + "-", "").replace("26202500", "seed ")
            axis.plot([row["update"] for row in checkpoints],
                      [row["validation_loss"] - initial for row in checkpoints],
                      marker="o", markersize=3, linestyle="--" if "repair" in run else "-", label=label)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set_title("Quadratic cSMC" if family == "csmc" else "KDM-inspired RBF adapter")
        axis.set_xlabel("Optimizer update")
        axis.legend(fontsize=6)
    axes[0].set_ylabel("Validation moment loss minus initialization")
    figure.savefig(directory / "validation-learning-curves.pdf")
    figure.savefig(directory / "validation-learning-curves.png", dpi=180)
    plt.close(figure)


def interrupted_summary(rows, dataset_seeds, failure):
    expected = {(seed, 274000000 + 2 * (seed % 1000) + replicate, method)
                for seed in dataset_seeds for replicate in range(2) for method in METHODS}
    keys = [(row["dataset_seed"], row["particle_seed"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Interrupted test has duplicated or out-of-scope keys")
    complete = [seed for seed in dataset_seeds
                if sum(key[0] == seed for key in keys) == 2 * len(METHODS)]
    return dict(status="interrupted_continuation_veto", failure=failure,
        expected_rows=len(expected), recorded_rows=len(rows),
        valid_rows=sum(row["valid"] for row in rows),
        invalid_keys=[{name: row[name] for name in ("dataset_seed", "particle_seed", "method")}
                      for row in rows if not row["valid"]],
        complete_datasets=len(complete), missing_rows=len(expected - set(keys)),
        contrasts=None, ranking_supported=False, default_ready=False,
        interpretation="No survivor-only ranking; incomplete heldout comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--source-map", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.directory / "manifest.json").read_text())
    selection = json.loads((args.directory / "selection.json").read_text())
    completion_path = args.directory / "completion.json"
    completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
    if selection["criterion"] != "terminal_moment_loss":
        raise ValueError("The moment-only campaign is required")
    if completion is not None and completion["status"] != "complete":
        raise ValueError("Unknown campaign completion status")
    if not selection["frozen_before_test"] or selection["source_hashes"] != manifest["source_hashes"]:
        raise ValueError("Selection provenance differs from the executed manifest")
    source_files = json.loads(args.source_map.read_text()) if args.source_map else {}
    source_match = source_hash_checks(manifest["source_hashes"], source_files)
    if not all(source_match.values()):
        raise ValueError("Campaign runtime source changed after launch")
    histories = [json.loads(line) for line in (args.directory / "training.jsonl").read_text().splitlines()]
    training = {}
    for row in histories:
        run = training.setdefault(row["run"], dict(updates=0, accepted=0, bound_hits=0,
            invalid_probe_pairs=0, invalid_trials=0, last_eta=None))
        run.update(updates=row["update"], accepted=row["accepted_count"], last_eta=row["eta"])
        run["bound_hits"] += row["bound_hits"]
        run["invalid_probe_pairs"] += row["plus_loss"] is None or row["minus_loss"] is None
        run["invalid_trials"] += sum(trial["loss"] is None for trial in row["trials"])
    for family, chosen in selection["selected"].items():
        candidates = [row for row in selection["candidates"] if row["family"] == family]
        if chosen != min(candidates, key=lambda row: row["validation_loss"]):
            raise ValueError("Selected coefficients do not minimize the recorded validation moment loss")
    partitions = manifest["partitions"]
    sets = [set(partitions[name]) for name in ("training", "validation", "test")]
    if any(sets[left] & sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise ValueError("Training, validation and test datasets overlap")
    rows = [json.loads(line) for line in (args.directory / "test.jsonl").read_text().splitlines()]
    if completion is None:
        failure = json.loads((args.directory / "failure.json").read_text())
        result = interrupted_summary(rows, partitions["test"], failure)
        result.update(selection=selection, training=training, source_hash_checks=source_match,
                      source_file_mapping=source_files,
                      analysis_role="CPU post-run diagnostic; no partial comparison ranking")
        (args.directory / "interrupted-analysis.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({name: result[name] for name in
            ("status", "recorded_rows", "complete_datasets", "invalid_keys", "ranking_supported")}))
        return
    result = analyze_records(rows, partitions["test"])
    result.update(selection=selection, training=training, source_hash_checks=source_match,
                  source_file_mapping=source_files,
                  wall_seconds=completion["wall_seconds"], calls=completion["calls"],
                  analysis_role="CPU post-run diagnostic; NumPy never imported by runtime")
    (args.directory / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.directory / "result.md").write_text(render_report(result))
    render_figures(args.directory, result)
    print(json.dumps(dict(decisions=result["decisions"],
                         heuristic_dominance=result["heuristic_dominance"]["verdict"])))


if __name__ == "__main__":
    main()
