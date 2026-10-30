"""Post-run diagnostic analysis; never selects runtime numerical controls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time


GUIDED = ("csmc_quadratic", "kernel")
HEURISTICS = ("prior", "global", "local")


def average(values):
    return math.fsum(values) / len(values)


def quantile(sorted_values, probability):
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def squared_error(row, reference):
    return math.fsum((value - target) ** 2 for value, target in zip(row["score"], reference))


def analyse(directory):
    started = time.monotonic()
    manifest = json.loads((directory / "manifest.json").read_text())
    rows_path = directory / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line]
    keys = [(row["data_seed"], row["seed"], row["arm"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation rows")
    if len(rows) != manifest["completed_rows"]:
        raise ValueError("Manifest/row count mismatch")
    expected_keys = {(int(data_seed), seed, arm)
                     for data_seed, seeds in manifest["seeds"].items()
                     for seed in seeds for arm in manifest["arms"]}
    if len(expected_keys) != 96 or not set(keys).issubset(expected_keys):
        raise ValueError("Unexpected paired evaluation design")
    trace_fields = manifest["trace_fields"]
    status_indices = [trace_fields.index(name) for name in ("valid", "moment_valid", "correction_cap_valid")]
    invalid = []
    for row in rows:
        if row["reset_steps"] != 5 or row["reset_epsilon"] != 1.:
            raise ValueError("The fixed OT setting changed")
        if row["beta"] != manifest["arms"][row["arm"]]:
            raise ValueError("The proposal mixture setting changed")
        if row["seed"] not in manifest["seeds"][str(row["data_seed"])]:
            raise ValueError("Unexpected evaluation seed")
        valid = row["valid"] and row["steps_completed"] == [50] * 3
        valid &= isinstance(row["value"], (float, int)) and math.isfinite(row["value"])
        valid &= len(row["score"]) == 3
        valid &= all(isinstance(value, (float, int)) and math.isfinite(value) for value in row["score"])
        valid &= len(row["traces"]) == 3 and all(len(trace) == 50 for trace in row["traces"])
        valid &= all(len(step) == len(trace_fields) for trace in row["traces"] for step in trace)
        valid &= all(isinstance(value, (float, int)) and math.isfinite(value)
                     for trace in row["traces"] for step in trace for value in step)
        if valid:
            valid &= all(step[index] == 1. for trace in row["traces"] for step in trace for index in status_indices)
        if not valid:
            invalid.append(list((row["data_seed"], row["seed"], row["arm"])))
    rng = random.Random(26101999)
    datasets = []
    for data_seed in (26100831, 26100832):
        reference = manifest["reference"]["datasets"][str(data_seed)]
        grouped = {arm: sorted([row for row in rows if row["data_seed"] == data_seed and row["arm"] == arm
                                and [row["data_seed"], row["seed"], row["arm"]] not in invalid],
                               key=lambda row: row["seed"]) for arm in manifest["arms"]}
        summaries = {}
        for arm, group in grouped.items():
            failed_count = sum(row["data_seed"] == data_seed and row["arm"] == arm
                               and [row["data_seed"], row["seed"], row["arm"]] in invalid for row in rows)
            if not group:
                summaries[arm] = dict(count=0, failed_count=failed_count)
                continue
            coordinate_mse = [average([(row["score"][coordinate] - reference["score"][coordinate]) ** 2
                                       for row in group]) for coordinate in range(3)]
            summaries[arm] = dict(
                count=len(group), failed_count=failed_count, mean_score=[average([row["score"][coordinate] for row in group])
                                             for coordinate in range(3)],
                score_se=[statistics.stdev([row["score"][coordinate] for row in group]) / math.sqrt(len(group))
                          if len(group) > 1 else None for coordinate in range(3)],
                coordinate_mse=coordinate_mse, coordinate_rmse=[math.sqrt(value) for value in coordinate_mse],
                mse=math.fsum(coordinate_mse), rmse=math.sqrt(math.fsum(coordinate_mse)),
                mean_value=average([row["value"] for row in group]),
                mean_seconds=average([row["seconds"] for row in group]),
            )
            explanatory = {}
            for field in ("ess", "max_weight", "reset_covariance_relative_error", "reset_margin", "sinkhorn_row_error"):
                if field in trace_fields:
                    index = trace_fields.index(field)
                    values = [step[index] for row in group for step in row["traces"][0]]
                    explanatory[field] = dict(minimum=min(values), median=statistics.median(values), maximum=max(values))
            summaries[arm]["explanatory_diagnostics"] = explanatory
        common_seeds = set.intersection(*({row["seed"] for row in group} for group in grouped.values()))
        balanced = {}
        for arm, group in grouped.items():
            paired_group = [row for row in group if row["seed"] in common_seeds]
            if paired_group:
                balanced[arm] = dict(count=len(paired_group),
                    rmse=math.sqrt(average([squared_error(row, reference["score"]) for row in paired_group])),
                    mean_score=[average([row["score"][coordinate] for row in paired_group])
                                for coordinate in range(3)])
        comparisons = {}
        for arm in GUIDED:
            baseline = {row["seed"]: row for row in grouped["plain"]}
            paired = [row for row in grouped[arm] if row["seed"] in baseline]
            differences = [squared_error(row, reference["score"]) -
                           squared_error(baseline[row["seed"]], reference["score"]) for row in paired]
            complete = len(paired) == 8
            comparison = dict(pairs=len(paired), paired_mse_difference=average(differences) if differences else None,
                              interval_98_75=None, conditional_gain=False, conditional_harm=False)
            if complete:
                samples = sorted(average([differences[rng.randrange(8)] for _ in range(8)])
                                 for _ in range(20000))
                interval = [quantile(samples, .00625), quantile(samples, .99375)]
                comparison.update(interval_98_75=interval, conditional_gain=interval[1] < 0,
                                  conditional_harm=interval[0] > 0)
            comparisons[arm] = comparison
        heuristic_losses = []
        for arm in ("plain", *GUIDED):
            for heuristic in HEURISTICS:
                if summaries[arm]["count"] != 8 or summaries[heuristic]["count"] != 8:
                    continue
                for coordinate in range(3):
                    candidate_mse = summaries[arm]["coordinate_mse"][coordinate]
                    heuristic_mse = summaries[heuristic]["coordinate_mse"][coordinate]
                    if candidate_mse > heuristic_mse:
                        heuristic_losses.append(dict(arm=arm, comparator=heuristic, coordinate=coordinate,
                            candidate_mse=candidate_mse, heuristic_mse=heuristic_mse))
        datasets.append(dict(data_seed=data_seed, reference=reference, arms=summaries,
                             comparisons=comparisons, heuristic_losses=heuristic_losses,
                             balanced_descriptive=dict(seeds=sorted(common_seeds), arms=balanced)))
    evaluation_complete = set(keys) == expected_keys and manifest["status"] == "complete"
    complete = evaluation_complete and not invalid
    failures = []
    for row in rows:
        if [row["data_seed"], row["seed"], row["arm"]] in invalid:
            steps = min(row["steps_completed"])
            failures.append(dict(data_seed=row["data_seed"], seed=row["seed"], arm=row["arm"],
                                 steps_completed=row["steps_completed"],
                                 terminal_trace=dict(zip(trace_fields, row["traces"][0][steps - 1])) if steps else {}))
    decision = dict(
        hard_veto_rows=invalid, complete=complete, evaluation_complete=evaluation_complete, failures=failures,
        conditional_gain_both_datasets={arm: all(
            dataset["comparisons"][arm]["pairs"] == 8 and dataset["comparisons"][arm]["conditional_gain"]
            for dataset in datasets) for arm in GUIDED},
        heuristic_dominance={arm: ("fail" if any(
            loss["arm"] == arm for dataset in datasets for loss in dataset["heuristic_losses"])
            else "incomplete" if not complete else "pass") for arm in ("plain", *GUIDED)},
        default_ready=False,
        reference_uncertainty_propagated=False,
        evidence_role="conditional_diagnostic_single_fit_per_dataset_not_canonical_admission",
    )
    fits = {}
    for data_seed in (26100831, 26100832):
        fit_path = directory / f"guides-{data_seed}.json"
        if hashlib.sha256(fit_path.read_bytes()).hexdigest() != manifest["fits"][str(data_seed)]["sha256"]:
            raise ValueError("Frozen guide checksum changed")
        fit = json.loads(fit_path.read_text())
        projection_ratios = [projected / raw for projected, raw in
                             zip(fit["projected_error"], fit["quadratic_error"]) if raw > 0]
        fits[data_seed] = dict(
            projected_times=sum(fit["projected"]), kernel_active_times=sum(fit["active"]),
            kernel_active_indices=[index + 1 for index, active in enumerate(fit["active"]) if active],
            max_projection_mse=max(fit["projected_error"]), max_raw_quadratic_mse=max(fit["quadratic_error"]),
            median_projection_mse_ratio=statistics.median(projection_ratios) if projection_ratios else None,
            maximum_projection_mse_ratio=max(projection_ratios) if projection_ratios else None,
            min_qr_condition=min(fit["qr_condition"]), min_design_margin=min(fit["design_margin"]),
            terminal_quadratic_zero=all(value == 0 for row in fit["precision"][-1] for value in row)
                and all(value == 0 for value in fit["linear"][-1]),
            terminal_kernel_inactive=not fit["active"][-1],
        )
    smoke = json.loads((directory / "smoke.json").read_text())
    derivative_checks = [check for arm in smoke for check in arm["derivative_checks"] if check["step"] <= 2.5e-5]
    richardson_checks = [check for arm in smoke for check in arm["richardson"]]
    if {arm["arm"] for arm in smoke} != {"plain", *GUIDED} or len(smoke) != 3:
        raise ValueError("Missing fitted-guide derivative arm")
    if len(derivative_checks) != 18 or len(richardson_checks) != 9:
        raise ValueError("Incomplete derivative evidence")
    if not all(arm["valid"] for arm in smoke) or not all(
            check["passed"] for check in derivative_checks + richardson_checks):
        raise ValueError("Invalid fitted-guide derivative evidence")
    analysis = dict(
        datasets=datasets, decision=decision, fits=fits,
        derivative_checks=len(derivative_checks), derivative_pass=all(check["passed"] for check in derivative_checks),
        maximum_derivative_relative_error=max(check["relative_error"] for check in derivative_checks),
        richardson_checks=len(richardson_checks), richardson_pass=all(check["passed"] for check in richardson_checks),
        maximum_richardson_relative_error=max(check["relative_error"] for check in richardson_checks),
        traces_checked=sum(len(trace) for row in rows for trace in row["traces"]),
        rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpu_only=True, tensorflow_imported=False, analysis_seconds=time.monotonic() - started,
        bootstrap=dict(replicates=20000, seed=26101999, family_contrasts=4, confidence=.9875,
                       sampling_unit="paired_evaluation_seed_conditional_on_one_fit_and_one_dataset"),
    )
    (directory / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    report = ["# SIR guidance with fixed OT (1,5)", "",
              "This comparison evaluates frozen cSMC quadratic and KDM-inspired kernel guidance in the existing optional three-density consumer. It does not run the original KDM neural learner or establish canonical LEDH admission.", "",
              f"Completed {len(rows)}/96 rows; {len(invalid)} invalid rows. The primary criterion is the paired change in squared L2 score error versus the equal-mixture unguided arm, separately for each dataset.", ""]
    for dataset in datasets:
        report += [f"## Data {dataset['data_seed']}", "",
                   "| Arm | Valid / failed seeds | Mean log κ score | Mean log ν score | Mean log noise score | Score RMSE |",
                   "|---|---:|---:|---:|---:|---:|"]
        reference = dataset["reference"]
        report.append("| Zhao–Cui reference | 2 fits | " + " | ".join(f"{value:.6f}" for value in reference["score"]) + " | — |")
        for arm, summary in dataset["arms"].items():
            if summary["count"]:
                reported_rmse = f"{summary['rmse']:.6f}" if not summary["failed_count"] else "not comparable: failed seed"
                report.append(f"| {arm} | {summary['count']} / {summary['failed_count']} | " + " | ".join(f"{value:.6f}" for value in summary["mean_score"])
                              + f" | {reported_rmse} |")
        if not evaluation_complete:
            balanced = dataset["balanced_descriptive"]
            report += ["", "Incomplete design: the table above preserves every available row and may have unequal counts. The following balanced subset uses only seeds completed by all six arms. It is descriptive, not a replacement for the predeclared eight-pair test.",
                       f"Common seeds: {balanced['seeds']}", "",
                       "| Arm | Common seeds | Descriptive score RMSE |", "|---|---:|---:|"]
            for arm, summary in balanced["arms"].items():
                report.append(f"| {arm} | {summary['count']} | {summary['rmse']:.6f} |")
        report += ["", "RMSE is the square root of the mean per-run squared L2 error; it includes bias and seed variability.", "",
                   "| Guided minus plain | Paired MSE difference | 98.75% paired bootstrap interval | Status |",
                   "|---|---:|---|---|"]
        for arm, comparison in dataset["comparisons"].items():
            interval = comparison["interval_98_75"]
            status = "conditional gain" if comparison["conditional_gain"] else "conditional harm" if comparison["conditional_harm"] else "no supported ranking"
            report.append(f"| {arm} | {comparison['paired_mse_difference']} | {interval} | {status} |")
        report += ["", "| Arm | κ RMSE | ν RMSE | noise RMSE |", "|---|---:|---:|---:|"]
        for arm, summary in dataset["arms"].items():
            if summary["count"]:
                report.append(f"| {arm} | " + " | ".join(f"{value:.6f}" for value in summary["coordinate_rmse"]) + " |")
        report += ["", f"Heuristic losses: {len(dataset['heuristic_losses'])} coordinate-specific descriptive losses across plain and guided arms. All are retained in analysis.json; any guided loss is a promotion veto, not a continuation veto.", ""]
    report += ["## Decision", "",
               "| Decision field | Result |", "|---|---|",
               f"| Primary criterion | Conditional gain on both datasets: {decision['conditional_gain_both_datasets']} |",
               f"| Veto diagnostics | Invalid rows: {len(invalid)}; heuristic screen: {decision['heuristic_dominance']} |",
               f"| Execution completeness | All 96 keys accounted for: {evaluation_complete}; all numerically valid: {complete} |",
               "| Decision | No canonical/default/HMC promotion from this diagnostic |",
               "| Main uncertainty | Up to eight evaluation seeds per dataset (counts above); one offline fit each; approximate reference uncertainty is not propagated |",
               "| Next justified action | Inspect message approximation and consumer mechanism before a fresh, independently fitted canonical comparison |",
               "| Not concluded | Exact model-score correctness, original KDM performance, cross-dataset superiority, or equal-compute advantage |", "",
               "| Inference status | Evidence |", "|---|---|",
               f"| Hard veto screen | {len(invalid)} invalid rows; complete={complete} |",
               "| Statistically supported ranking | Only the conditional contrasts explicitly labeled gain/harm above; bootstrap uncertainty is approximate with eight pairs |",
               "| Descriptive-only differences | Mean scores, component RMSE, timings and heuristic comparisons |",
               "| Default-readiness | Not established; scope tuning and canonical conformance are absent |",
               "| Next evidence needed | Independent offline fits, fresh data, reference uncertainty and canonical implementation comparison |", "",
               "## Interpretation and audit limits", "",
               "Engineering correctness is supported by CPU regression tests and fitted-guide GPU/XLA finite differences. Numerical validity is checked independently from statistical comparisons. MathDevMCP could not encode the three derivation obligations. Its single-file implementation scan also reported a missing solve; inspection found the checked Cholesky solve in the actual imported call chain. The full reports and dispositions are in the parent audit.md; neither automatic report is proof certification. The manuscript gives the completing-square, projection and chain-rule derivations explicitly.", "",
               "The strongest alternative explanation for any gain or loss is the approximation imposed on the fitted future message: many quadratic Hessians required positive-information projection, and the kernel activates at only a subset of times. A weak or misleading message can produce a numerically correct but statistically poor proposal. These are candidate failures, not rejection of controlled SMC or KDM as research directions.", "",
               "A claim of general improvement would be overturned by independently fitted repetitions or a canonical consumer showing no advantage. The weakest evidence is one fitted guide per dataset and the approximate Zhao–Cui reference. The source audit distinguishes the local corrected log-mean-exp path calculation from the original author's displayed mean-log-weight statistic.", "",
               "Artifacts: manifest.json, rows.jsonl, analysis.json, guides-*.json and smoke.json in this attempt directory. The parent directory contains CPU tests, GPU preflight and MathDevMCP audit results. GPU FP64/XLA is an explicit reference exception; TF32 is off and memory growth is verified. OT is (1,5) throughout.", ""]
    (directory / "result.md").write_text("\n".join(report))
    print(json.dumps(dict(completed=len(rows), decision=decision,
        summaries={dataset["data_seed"]: {arm: summary.get("rmse") for arm, summary in dataset["arms"].items()}
                   for dataset in datasets}, analysis_seconds=analysis["analysis_seconds"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    analyse(parser.parse_args().directory)
