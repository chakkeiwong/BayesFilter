"""Diagnostic analysis of fixed-input, alternating GPU consumer timings."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_sir_postmerge_replay import compare_numeric, read_rows, verify_pair


def compare_consumer_outputs(root):
    lanes = {}
    for lane in ("before", "after"):
        grouped = {}
        for line in (root / lane / "rows.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["dataset"], row["arm"], row["direction"])
            grouped.setdefault(key, []).append(row["output"])
        lanes[lane] = grouped
    if lanes["before"].keys() != lanes["after"].keys() or len(lanes["before"]) != 24:
        raise ValueError("Consumer output coverage differs")
    comparisons = []
    for key, baseline in lanes["before"].items():
        candidate = lanes["after"][key]
        if len(baseline) != 9 or len(candidate) != 9:
            raise ValueError("Expected one warm call and eight timed calls")
        for left, right in zip(baseline, candidate):
            if not (left["valid"] and right["valid"]) or left["checks"] != right["checks"]:
                raise ValueError("Consumer validity flags differ")
            for field in ("loss", "blocks", "raw_blocks", "estimate", "raw_estimate", "value", "score"):
                comparisons.append(compare_numeric(left[field], right[field]))
    if not all(value["passed"] for value in comparisons):
        raise ValueError("Direct consumer-output parity failed")
    return dict(paired_calls=216, exact_flags=True,
        max_absolute_difference=max(value["max_absolute"] for value in comparisons),
        max_tolerance_ratio=max(value["max_tolerance_ratio"] for value in comparisons))


def ratio_interval(log_ratios):
    if len(log_ratios) != 8:
        raise ValueError("The preregistered interval requires eight blocks")
    center = statistics.mean(log_ratios)
    margin = 2.364624251 * statistics.stdev(log_ratios) / math.sqrt(8)
    return dict(ratio=math.exp(center), lower=math.exp(center-margin),
                upper=math.exp(center+margin), block_log_ratios=log_ratios)


def verify_repaired_replay(baseline, candidate, *, baseline_harness=None, candidate_harness=None):
    report = verify_pair(baseline, candidate, baseline_lane="after",
                         baseline_harness=baseline_harness, candidate_harness=candidate_harness)
    strict = {}
    for filename, label in (("holdout.jsonl", "holdout"), ("scores.jsonl", "directional")):
        before = read_rows(baseline / filename, label == "directional")
        after = read_rows(candidate / filename, label == "directional")
        comparisons = []
        for key, left in before.items():
            for field in ("raw_blocks", "raw_estimate"):
                if field in left:
                    comparisons.append(compare_numeric(left[field], after[key][field]))
        strict[label] = dict(passed=all(value["passed"] for value in comparisons),
            exact_flags=not report[label]["changed_checks"],
            max_absolute_difference=max(value["max_absolute"] for value in comparisons),
            max_tolerance_ratio=max(value["max_tolerance_ratio"] for value in comparisons),
            scalars=sum(value["scalars"] for value in comparisons))
    report["repair_raw_moment_checks"] = strict
    report["repair_tolerance"] = dict(absolute=1e-8, relative=1e-8, includes_raw_moments=True)
    report["repair_acceptance"] = report["equivalence_passed"] and all(
        value["passed"] and value["exact_flags"] for value in strict.values())
    report["baseline_role"] = "frozen_replay_comparator"
    report["candidate_role"] = "frozen_repaired_candidate"
    report["baseline_replay"] = str(baseline)
    report["candidate_replay"] = str(candidate)
    return report


def analyze(root, *, baseline_factorization="qr"):
    if baseline_factorization not in ("qr", "checked_elimination"):
        raise ValueError("Unknown baseline flow factorization")
    result = json.loads((root / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Incomplete timing comparison")
    blocks = {(block["lane"], block["block"]): block["rows"] for block in result["blocks"]}
    if len(blocks) != 16:
        raise ValueError("Missing paired blocks")
    datasets = sorted({row["dataset"] for row in blocks[("before", 0)]})
    if len(datasets) != 4:
        raise ValueError("Expected four preselected regimes")
    timings = []
    for arm in ("plain", "conditional", "physical"):
        for direction in (-1, 0):
            ratios, seconds = {dataset: [] for dataset in datasets}, {"before": [], "after": []}
            for block in range(8):
                for dataset in datasets:
                    pair = {}
                    for lane in seconds:
                        matches = [row for row in blocks[(lane, block)]
                                   if (row["arm"], row["direction"], row["dataset"])
                                   == (arm, direction, dataset)]
                        if len(matches) != 1:
                            raise ValueError("Missing or duplicate consumer call")
                        row = matches[0]
                        if not all(value["passed"] for value in row["comparisons"].values()):
                            raise ValueError("Numerical parity failed")
                        pair[lane] = row["synchronized_seconds"]
                        seconds[lane].append(pair[lane])
                    ratios[dataset].append(math.log(pair["after"] / pair["before"]))
            pooled = ratio_interval([statistics.mean(ratios[dataset][block] for dataset in datasets)
                                     for block in range(8)])
            regimes = {dataset: ratio_interval(values) for dataset, values in ratios.items()}
            threshold = 1.10 if arm == "physical" else 1.
            timings.append(dict(arm=arm, direction=direction, pooled=pooled, per_dataset=regimes,
                mean_seconds={lane: statistics.mean(values) for lane, values in seconds.items()},
                acceptance_threshold=threshold, passed=pooled["upper"] < threshold,
                any_descriptive_regression=any(value["ratio"] > 1 for value in regimes.values())))
    compiled = {}
    for lane in ("before", "after"):
        terminal = result[lane]["terminal"]
        if not terminal["source_unchanged"] or set(terminal["trace_counts"].values()) != {1}:
            raise ValueError("Source mutation or retracing")
        compiled[lane] = {}
        for arm in ("plain", "conditional", "physical"):
            path = root / lane / f"{arm}.optimized-hlo.txt"
            hlo = path.read_text()
            compiled[lane][arm] = dict(live_qr='op_type="Qr"' in hlo,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            expected_qr = lane == "before" and baseline_factorization == "qr"
            if compiled[lane][arm]["live_qr"] != expected_qr:
                raise ValueError("Consumer HLO does not match the proposed repair")
    comparisons = [value for rows in blocks.values() for row in rows
                   for value in row["comparisons"].values()]
    summary = dict(status="complete", timings=timings, compiled=compiled,
        baseline_factorization=baseline_factorization,
        direct_output_parity=compare_consumer_outputs(root),
        timing_acceptance=all(row["passed"] for row in timings),
        paired_interval="approximate Student-t, 7 df, eight block means of paired log ratios",
        interval_scope="conditional on four exposed inputs and this hardware; block independence assumed",
        synchronized_calls=sum(len(rows) for rows in blocks.values()),
        max_absolute_difference=max(value["max_absolute"] for value in comparisons),
        max_tolerance_ratio=max(value["max_tolerance_ratio"] for value in comparisons),
        charge_seconds=result["charged_gpu_process_seconds"])
    (root / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--baseline-replay", type=Path)
    parser.add_argument("--candidate-replay", type=Path)
    parser.add_argument("--baseline-harness", type=Path)
    parser.add_argument("--candidate-harness", type=Path)
    parser.add_argument("--baseline-factorization", choices=("qr", "checked_elimination"), default="qr")
    arguments = parser.parse_args()
    if bool(arguments.baseline_replay) != bool(arguments.candidate_replay):
        parser.error("Both replay paths are required for a full repair comparison")
    if bool(arguments.baseline_harness) != bool(arguments.candidate_harness):
        parser.error("Both frozen replay drivers are required")
    if arguments.baseline_harness and not arguments.baseline_replay:
        parser.error("Frozen drivers require replay paths")
    summary = analyze(arguments.attempt, baseline_factorization=arguments.baseline_factorization)
    print(json.dumps(dict(accepted=summary["timing_acceptance"],
        timings=[dict(arm=row["arm"], direction=row["direction"],
                      ratio=row["pooled"]["ratio"], interval=[row["pooled"]["lower"], row["pooled"]["upper"]])
                 for row in summary["timings"]])))
    if arguments.baseline_replay:
        replay = verify_repaired_replay(arguments.baseline_replay, arguments.candidate_replay,
            baseline_harness=arguments.baseline_harness,
            candidate_harness=arguments.candidate_harness)
        (arguments.attempt.parent / "comparison.json").write_text(
            json.dumps(replay, indent=2, allow_nan=False)+"\n")
        print(json.dumps(dict(replay_accepted=replay["repair_acceptance"],
            rows=replay["holdout"]["rows"], scores=replay["directional"]["rows"])))
        if not replay["repair_acceptance"]:
            raise SystemExit(2)
    if not summary["timing_acceptance"]:
        raise SystemExit(2)
