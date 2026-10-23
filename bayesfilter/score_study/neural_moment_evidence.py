"""Paired, dataset-cluster moment evidence for the scoped neural SIR study."""

import math
import random


PARTICLE_ARMS = ("trained", "untrained", "plain", "conditional", "physical")
HEURISTIC_ARMS = ("prediction", "observation_only", "recursive_gaussian")
PARTICLE_SEED_BASE = 283030000


def validate_holdout(rows, datasets):
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Expected unique nonempty datasets")
    expected = {(dataset, arm, replicate)
                for dataset in datasets for arm in PARTICLE_ARMS for replicate in range(4)}
    expected |= {(dataset, arm, None) for dataset in datasets for arm in HEURISTIC_ARMS}
    indexed, regimes = {}, {}
    for row in rows:
        key = (row.get("dataset"), row.get("arm"), row.get("replicate"))
        if key not in expected or key in indexed:
            raise ValueError("Unexpected or duplicate holdout row")
        dataset, arm, replicate = key
        if arm in PARTICLE_ARMS:
            if type(replicate) is not int or type(row.get("particle_seed")) is not int:
                raise ValueError("Invalid particle replication identity")
            seed = PARTICLE_SEED_BASE + 4 * datasets.index(dataset) + replicate
            if row["particle_seed"] != seed:
                raise ValueError("Unpaired holdout particle randomness")
        elif row.get("particle_seed") is not None:
            raise ValueError("Deterministic comparator has a particle seed")
        if row.get("valid") is not True:
            raise ValueError("Invalid holdout computation")
        blocks = row.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != 2:
            raise ValueError("Missing first/second moment losses")
        values = [row.get("loss"), *blocks]
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0.
               for value in values):
            raise ValueError("Nonfinite or negative moment loss")
        if not math.isclose(row["loss"], sum(blocks) / 2., rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("Moment blocks and total loss disagree")
        regime = (row.get("observation_regime"), row.get("compartment_regime"))
        if regime[0] not in ("low", "high") or regime[1] not in ("nearer_boundary", "interior"):
            raise ValueError("Missing holdout regime")
        if dataset in regimes and regimes[dataset] != regime:
            raise ValueError("Inconsistent dataset regime")
        regimes[dataset] = regime
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Incomplete paired holdout grid")
    return indexed, regimes


def paired_interval(differences, *, draws=5000, seed=263045001):
    if len(differences) < 2:
        return None
    generator = random.Random(seed)
    count = len(differences)
    samples = sorted(math.fsum(differences[generator.randrange(count)] for _ in range(count))
                     / count for _ in range(draws))
    return [samples[int(.025 * (draws - 1))], samples[int(.975 * (draws - 1))]]


def analyze_holdout(rows, datasets):
    indexed, regimes = validate_holdout(rows, datasets)
    arms = PARTICLE_ARMS + HEURISTIC_ARMS
    averaged = {}
    for dataset in datasets:
        for arm in arms:
            records = [indexed[dataset, arm, replicate] for replicate in range(4)] \
                if arm in PARTICLE_ARMS else [indexed[dataset, arm, None]]
            averaged[dataset, arm] = [math.fsum(record["loss"] for record in records) / len(records)]
            averaged[dataset, arm] += [math.fsum(record["blocks"][block] for record in records)
                                      / len(records) for block in range(2)]
    subsets = {"all": datasets}
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][0] == name]
                    for name in ("low", "high")})
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][1] == name]
                    for name in ("nearer_boundary", "interior")})
    summaries, comparisons = [], []
    for regime, selected in subsets.items():
        if not selected:
            continue
        for arm in arms:
            means = [math.fsum(averaged[dataset, arm][block] for dataset in selected) / len(selected)
                     for block in range(3)]
            summaries.append(dict(regime=regime, datasets=len(selected), arm=arm,
                                  loss=means[0], blocks=means[1:]))
        for comparator in arms[1:]:
            differences = [averaged[dataset, "trained"][0] - averaged[dataset, comparator][0]
                           for dataset in selected]
            baseline = math.fsum(averaged[dataset, comparator][0] for dataset in selected) / len(selected)
            difference = math.fsum(differences) / len(selected)
            interval = paired_interval(differences)
            comparisons.append(dict(regime=regime, datasets=len(selected), comparator=comparator,
                mean_difference=difference, relative_reduction=-difference / baseline if baseline > 0. else None,
                paired_cluster_interval95=interval,
                interval_excludes_zero=interval is not None and (interval[1] < 0. or interval[0] > 0.),
                descriptive_only=regime != "all" or comparator not in ("plain", "untrained")))
    primary = [row for row in comparisons if row["regime"] == "all"
               and row["comparator"] in ("plain", "untrained")]
    primary_pass = all(row["relative_reduction"] is not None and row["relative_reduction"] >= .1
                       and row["paired_cluster_interval95"] is not None
                       and row["paired_cluster_interval95"][1] < 0. for row in primary)
    heuristic_failures = [row for row in comparisons if row["comparator"] in HEURISTIC_ARMS + ("conditional",)
                          and row["mean_difference"] > 0.]
    return dict(status="holdout_complete", integrity=dict(complete=True, paired=True,
        rows=len(rows), datasets=len(datasets), particle_replications=4), summaries=summaries,
        comparisons=comparisons, regime_counts={name: len(selected) for name, selected in subsets.items()},
        bootstrap=dict(draws=5000, seed=263045001, unit="dataset including all particle replications",
                       conditional_intervals="exploratory; no multiplicity-adjusted ranking"),
        primary_pass=primary_pass, heuristic_dominance=dict(passed=not heuristic_failures,
            failures=heuristic_failures, role="promotion veto; not a tuning criterion"),
        decision="optional_diagnostic_candidate" if primary_pass and not heuristic_failures
                 else "candidate_not_promoted",
        model_score_accuracy_evaluated=False, default_ready=False,
        missing_comparators=["freshly refitted cSMC restricted adapter", "freshly refitted RBF restricted adapter"])
