"""Post-run diagnostic analysis of the independent-beta SIR comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from analyze_ledh_sir_guidance import average, quantile, squared_error


CONTRASTS = {"quadratic_matched": ("csmc_quadratic", "plain_quadratic_beta"),
             "kernel_matched": ("kernel", "plain_kernel_beta"),
             "quadratic_calibrated": ("csmc_quadratic", "plain"),
             "kernel_calibrated": ("kernel", "plain")}


def analyse(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["design"] != "independent_beta_matched_guidance_v1":
        raise ValueError("Unexpected design")
    rows_path = directory / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    keys = [(row["data_seed"], row["seed"], row["arm"]) for row in rows]
    expected = {(int(dataset), seed, arm) for dataset, seeds in manifest["seeds"].items()
                for seed in seeds for arm in manifest["arms"]}
    if len(keys) != len(set(keys)) or len(rows) != manifest["completed_rows"]:
        raise ValueError("Duplicate rows or manifest count mismatch")
    if len(expected) != 112 or not set(keys).issubset(expected):
        raise ValueError("Unexpected paired design")
    status_indices = [manifest["trace_fields"].index(field) for field in ("valid", "moment_valid", "correction_cap_valid")]
    failures, valid_rows = [], []
    for row in rows:
        if row["reset_epsilon"] != 1. or row["reset_steps"] != 5:
            raise ValueError("OT policy changed")
        if row["beta"] != manifest["arms_by_dataset"][str(row["data_seed"])][row["arm"]]:
            raise ValueError("Frozen beta changed")
        valid = row["valid"] and row["steps_completed"] == [50] * 3
        valid &= len(row["score"]) == 3 and len(row["traces"]) == 3
        valid &= all(isinstance(value, (int, float)) and math.isfinite(value) for value in [row["value"], *row["score"]])
        valid &= all(len(trace) == 50 for trace in row["traces"])
        valid &= all(len(step) == len(manifest["trace_fields"]) for trace in row["traces"] for step in trace)
        valid &= all(isinstance(value, (int, float)) and math.isfinite(value) for trace in row["traces"] for step in trace for value in step)
        if valid:
            valid &= all(step[index] == 1. for trace in row["traces"] for step in trace for index in status_indices)
        if valid:
            valid_rows.append(row)
        else:
            steps = min(row["steps_completed"])
            failures.append(dict(data_seed=row["data_seed"], seed=row["seed"], arm=row["arm"],
                steps_completed=row["steps_completed"], terminal_trace=dict(zip(manifest["trace_fields"], row["traces"][0][steps - 1])) if steps else {}))
    smoke = json.loads((directory / "smoke.json").read_text())
    if {(entry["data_seed"], entry["arm"]) for entry in smoke} != {(int(dataset), arm) for dataset in manifest["seeds"] for arm in manifest["arms"]} or len(smoke) != 14:
        raise ValueError("Incomplete derivative arms")
    checks = [check for entry in smoke for check in entry["derivative_checks"]]
    extrapolations = [check for entry in smoke for check in entry["richardson"]]
    if len(checks) != 84 or len(extrapolations) != 42 or not all(entry["valid"] for entry in smoke):
        raise ValueError("Incomplete derivative checks")
    if not all(check["passed"] for check in checks + extrapolations):
        raise ValueError("Failed derivative check")
    rng = random.Random(26101998)
    datasets = []
    for identity in manifest["seeds"]:
        data_seed = int(identity)
        reference = manifest["reference"]["datasets"][identity]
        groups = {arm: {row["seed"]: row for row in valid_rows if row["data_seed"] == data_seed and row["arm"] == arm} for arm in manifest["arms"]}
        summaries = {}
        for arm, group in groups.items():
            count = len(group)
            failed = sum(row["data_seed"] == data_seed and row["arm"] == arm for row in failures)
            summary = dict(count=count, failed_count=failed, beta=manifest["arms_by_dataset"][identity][arm])
            if count:
                mse = [average([(row["score"][coordinate] - reference["score"][coordinate]) ** 2 for row in group.values()]) for coordinate in range(3)]
                summary.update(mean_score=[average([row["score"][coordinate] for row in group.values()]) for coordinate in range(3)],
                    score_se=[statistics.stdev([row["score"][coordinate] for row in group.values()]) / math.sqrt(count) if count > 1 else None for coordinate in range(3)],
                    coordinate_mse=mse, rmse=math.sqrt(sum(mse)),
                    mean_seconds=average([row["seconds"] for row in group.values()]))
            summaries[arm] = summary
        contrasts = {}
        for label, (candidate, comparator) in CONTRASTS.items():
            seeds = sorted(groups[candidate].keys() & groups[comparator].keys())
            differences = [squared_error(groups[candidate][seed], reference["score"]) - squared_error(groups[comparator][seed], reference["score"]) for seed in seeds]
            interval = None
            if len(seeds) == 8:
                bootstrap = sorted(average([differences[rng.randrange(8)] for _ in range(8)]) for _ in range(20000))
                interval = [quantile(bootstrap, .003125), quantile(bootstrap, .996875)]
            contrasts[label] = dict(candidate=candidate, comparator=comparator, pairs=len(seeds),
                mse_difference=average(differences) if differences else None, interval_99_375=interval,
                conditional_gain=interval is not None and interval[1] < 0,
                conditional_harm=interval is not None and interval[0] > 0)
        losses = []
        for candidate in ("csmc_quadratic", "kernel"):
            for comparator in ("plain", "prior", "local"):
                if summaries[candidate]["count"] != 8 or summaries[comparator]["count"] != 8:
                    continue
                for coordinate in range(3):
                    if summaries[candidate]["coordinate_mse"][coordinate] > summaries[comparator]["coordinate_mse"][coordinate]:
                        losses.append(dict(candidate=candidate, comparator=comparator, coordinate=coordinate))
        datasets.append(dict(data_seed=data_seed, reference=reference, arms=summaries, contrasts=contrasts, heuristic_losses=losses))
    complete = manifest["status"] == "complete" and set(keys) == expected and not failures
    gains = {arm: all(dataset["contrasts"][contrast]["conditional_gain"] for dataset in datasets for contrast in labels)
             for arm, labels in (("csmc_quadratic", ("quadratic_matched", "quadratic_calibrated")), ("kernel", ("kernel_matched", "kernel_calibrated")))}
    decision = dict(complete=complete, failures=failures, conditional_gain_both_datasets=gains,
        heuristic_dominance={arm: "fail" if any(loss["candidate"] == arm for dataset in datasets for loss in dataset["heuristic_losses"]) else "pass" if complete else "incomplete" for arm in gains},
        default_ready=False, reference_uncertainty_propagated=False)
    analysis = dict(datasets=datasets, decision=decision, rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        derivative_checks=len(checks), richardson_checks=len(extrapolations),
        bootstrap=dict(replicates=20000, seed=26101998, confidence=.99375, family_contrasts=8),
        cpu_only=True, tensorflow_imported=False)
    (directory / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    report = ["# Independently calibrated SIR guidance", "",
              "OT remains (1,5). These are conditional comparisons of the optional three-density consumer, not canonical LEDH or original neural KDM claims.", ""]
    for dataset in datasets:
        report += [f"## Data {dataset['data_seed']}", "", "| Arm | Valid / failed | Beta | Mean score | Score RMSE |", "|---|---:|---|---|---:|"]
        for arm, summary in dataset["arms"].items():
            score = ', '.join(f"{value:.4f}" for value in summary.get("mean_score", []))
            rmse = f"{summary['rmse']:.4f}" if summary['count'] == 8 else "incomplete; no ranking"
            report.append(f"| {arm} | {summary['count']} / {summary['failed_count']} | {summary['beta']} | {score} | {rmse} |")
        report += ["", "| Contrast | Pairs | Difference in squared score error | 99.375% interval |", "|---|---:|---:|---|"]
        for label, contrast in dataset["contrasts"].items():
            report.append(f"| {label} | {contrast['pairs']} | {contrast['mse_difference']} | {contrast['interval_99_375']} |")
        report += ["", f"Component-wise heuristic losses: {dataset['heuristic_losses']}", ""]
    report += ["## Decision and uncertainty", "", "| Field | Finding |", "|---|---|",
        "| Decision | No canonical/default promotion; inspect conditional contrasts |",
        f"| Primary criterion | Joint conditional gains: {gains} |",
        f"| Veto status | {len(failures)} invalid rows; heuristic verdict {decision['heuristic_dominance']} |",
        "| Main uncertainty | Eight seeds, one guide fit per data set, approximate reference not propagated |",
        "| Next justified action | Resolve remaining message/consumer mismatch before claiming improved LEDH |",
        "| Not concluded | Canonical LEDH correctness, original KDM performance, equal-compute or fresh-data superiority |", "",
        "| Inference status | Finding |", "|---|---|",
        f"| Hard veto screen | {len(failures)} failures; full valid design={complete} |",
        "| Statistically supported ranking | Only contrasts with intervals excluding zero, conditional on the frozen fits and references |",
        "| Descriptive-only differences | Scores, RMSE, timings and heuristic component comparisons |",
        "| Default-readiness | Not established |",
        "| Next evidence needed | Independent guide fits, reference uncertainty, fresh data and canonical conformance |", "",
        "Lower pilot J is not a score-accuracy result. Matched-beta comparisons isolate guidance from mixture selection. A remaining loss could reflect the positive-information projection or limited kernel activity, rather than failure of controlled SMC/KDM as research directions. The weakest evidence remains the single fitted guide and approximate reference; fresh fits or a canonical consumer can overturn these conditional findings.", ""]
    (directory / "result.md").write_text("\n".join(report))
    print(json.dumps(dict(rows=len(rows), decision=decision)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    analyse(parser.parse_args().directory)
