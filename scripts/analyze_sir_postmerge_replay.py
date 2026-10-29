"""Independent standard-library reporting for the frozen SIR merge diagnostic."""

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys


PARTICLE_ARMS = ("trained", "untrained", "plain", "conditional", "physical",
                 "csmc_unadapted", "csmc_adapted", "rbf_unadapted", "rbf_adapted")
HEURISTIC_ARMS = ("prediction", "observation_only", "recursive_gaussian")
FIELDS = ("loss", "blocks", "estimate", "raw_blocks", "raw_estimate", "value", "score")
MEASUREMENT_FLAGS = {"guide_conditioning_margin_measured"}
TRACE_FIELDS = ("prior_observation_log_normalizer", "d_prior_observation_log_normalizer",
    "higher_moment_maximum_diagonal_scaled_system_condition",
    "higher_moment_minimum_pairwise_particle_cap_scale",
    "higher_moment_fraction_coordinatewise_cap_active",
    "higher_moment_affine_restoration_relative_covariance_residual",
    "higher_moment_affine_restoration_normalized_mean_residual")
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 263050001


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(row, directional=False):
    key = (row["dataset"], row["arm"], row["replicate"])
    return key + (row["direction"],) if directional else key


def read_rows(path, directional=False):
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row_key(row, directional)
        if key in result:
            raise ValueError(f"Duplicate row in {path}: {key}")
        result[key] = row
    return result


def numeric_leaves(value, prefix=()):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(numeric_leaves(item, prefix + (key,)))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(numeric_leaves(item, prefix + (index,)))
        return result
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Non-numeric leaf: {prefix}")
    if not math.isfinite(value):
        raise ValueError(f"Non-finite leaf: {prefix}")
    return {prefix: value}


def compare_numeric(before, after, absolute=1e-8, relative=1e-8):
    left, right = numeric_leaves(before), numeric_leaves(after)
    if left.keys() != right.keys():
        raise ValueError("Numerical shapes differ")
    differences = [abs(right[key] - value) for key, value in left.items()]
    ratios = [difference / (absolute + relative * abs(value))
              for value, difference in zip(left.values(), differences)]
    return {"passed": max(ratios, default=0.) <= 1.,
            "max_absolute": max(differences, default=0.),
            "max_tolerance_ratio": max(ratios, default=0.), "scalars": len(left)}


def expected_keys(manifest, directional=False):
    datasets = manifest["score_datasets"] if directional else manifest["dataset_seeds"]
    keys = {(dataset, arm, replicate) for dataset in datasets
            for arm in PARTICLE_ARMS for replicate in range(4)}
    if directional:
        return {key + (direction,) for key in keys for direction in range(3)}
    return keys | {(dataset, arm, None) for dataset in datasets for arm in HEURISTIC_ARMS}


def is_validity_flag(name):
    return (name.endswith("_valid") or name.endswith("_finite")
            or name == "higher_moment_materialized_shape_available")


def validity(row):
    failures = [name for name, value in row.get("checks", {}).items()
                if is_validity_flag(name) and value is not True]
    return row["valid"] is True and not failures, failures


def compare_rows(before, after):
    summaries, changed_checks, measurement_flags, invalid = {}, [], Counter(), []
    by_arm = {}
    missing_validity_checks = []
    compared = 0
    for key, left in before.items():
        right = after[key]
        metadata = ("particle_seed", "input_sha256", "observation_regime", "compartment_regime")
        if any(left.get(name) != right.get(name) for name in metadata):
            raise ValueError(f"Pairing or input mismatch: {key}")
        left_valid, left_failures = validity(left)
        right_valid, right_failures = validity(right)
        if not left_valid or not right_valid:
            invalid.append({"key": key, "before": left_failures, "after": right_failures})
        if left.get("checks") != right.get("checks"):
            changed_checks.append(key)
        missing = [name for name in left.get("checks", {})
                   if is_validity_flag(name) and name not in right.get("checks", {})]
        if missing:
            missing_validity_checks.append({"key": key, "checks": missing})
        for name in MEASUREMENT_FLAGS:
            if name in right.get("checks", {}):
                measurement_flags[f"{name}={right['checks'][name]}"] += 1
        if "direction" in left:
            for row in (left, right):
                if not all(check["passed"] for check in row["direction_invariance"].values()):
                    raise ValueError(f"Directional value invariance failed: {key}")
        fields = ("loss", "blocks") if left["arm"] in HEURISTIC_ARMS else FIELDS
        for field in fields:
            result = compare_numeric(left[field], right[field],
                                     absolute=1e-5 if field.startswith("raw_") else 1e-8)
            for collection in (summaries, by_arm.setdefault(left["arm"], {})):
                aggregate = collection.setdefault(field, {"passed": True, "max_absolute": 0.,
                    "max_tolerance_ratio": 0., "scalars": 0, "worst_key": None,
                    "max_absolute_key": None, "failed_rows": 0, "rows": 0})
                aggregate["passed"] &= result["passed"]
                if result["max_tolerance_ratio"] > aggregate["max_tolerance_ratio"]:
                    aggregate["worst_key"] = key
                if result["max_absolute"] > aggregate["max_absolute"]:
                    aggregate["max_absolute_key"] = key
                for name in ("max_absolute", "max_tolerance_ratio"):
                    aggregate[name] = max(aggregate[name], result[name])
                aggregate["scalars"] += result["scalars"]
                aggregate["failed_rows"] += int(not result["passed"])
                aggregate["rows"] += 1
        compared += 1
    return {"rows": compared, "fields": summaries, "by_arm": by_arm, "invalid_rows": invalid,
            "changed_checks": changed_checks, "measurement_flags": dict(measurement_flags),
            "missing_validity_checks": missing_validity_checks,
            "passed": not invalid and not missing_validity_checks
            and all(value["passed"] for value in summaries.values())}


def explanatory_trace(before, after, key):
    if key is None:
        return None
    left, right = before[key], after[key]
    return {"key": key, "selection": "largest absolute version difference; explanation only",
        "quantities": {name: {"before": left[name], "after": right[name]}
                       for name in ("loss", "value", "score") if name in left and name in right},
        "traces": {name: {"before": left["diagnostics"][name],
                          "after": right["diagnostics"][name]}
                   for name in TRACE_FIELDS if name in left.get("diagnostics", {})
                   and name in right.get("diagnostics", {})}}


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def paired_summary(before, after):
    if not before or len(before) != len(after):
        raise ValueError("Empty or unpaired summary")
    differences = [new - old for old, new in zip(before, after)]
    generator = random.Random(BOOTSTRAP_SEED)
    draws = [statistics.fmean(generator.choices(differences, k=len(differences)))
             for _ in range(BOOTSTRAP_DRAWS)]
    return {"datasets": len(before), "before_mean": statistics.fmean(before),
            "after_mean": statistics.fmean(after), "mean_difference": statistics.fmean(differences),
            "difference_percentile_95": [percentile(draws, .025), percentile(draws, .975)],
            "ratio_of_means": statistics.fmean(after) / statistics.fmean(before)}


def runtime_report(before, after):
    grouped = defaultdict(lambda: defaultdict(lambda: ([], [])))
    cold = {lane: [] for lane in ("before", "after")}
    for key, left in before.items():
        right = after[key]
        if "evaluation_seconds" not in left:
            continue
        for lane, row in (("before", left), ("after", right)):
            if row.get("cold", False):
                cold[lane].append({"key": key, "seconds": row["evaluation_seconds"],
                                   "origin": row.get("timing_origin")})
        if left.get("cold", False) or right.get("cold", False):
            continue
        values = grouped[left["arm"]][left["dataset"]]
        values[0].append(left["evaluation_seconds"])
        values[1].append(right["evaluation_seconds"])
    summaries = {}
    for arm, datasets in grouped.items():
        left = [statistics.fmean(values[0]) for values in datasets.values()]
        right = [statistics.fmean(values[1]) for values in datasets.values()]
        summaries[arm] = paired_summary(left, right)
    return {"warm_paired_dataset_summaries": summaries, "cold_calls": cold,
            "bootstrap_draws": BOOTSTRAP_DRAWS, "bootstrap_seed": BOOTSTRAP_SEED,
            "interpretation": "Descriptive sequential-process timings; no causal speed claim."}


def loss_report(rows):
    grouped, regimes = defaultdict(lambda: defaultdict(list)), {}
    for row in rows.values():
        grouped[row["arm"]][row["dataset"]].append(row["loss"])
        regimes[row["dataset"]] = (row["observation_regime"], row["compartment_regime"])
    means = {arm: {dataset: statistics.fmean(values) for dataset, values in datasets.items()}
             for arm, datasets in grouped.items()}
    selections = {"all": set(regimes)}
    for observation in ("low", "high"):
        selections[f"observation_{observation}"] = {
            dataset for dataset, regime in regimes.items() if regime[0] == observation}
    for compartment in ("nearer_boundary", "interior"):
        selections[f"compartment_{compartment}"] = {
            dataset for dataset, regime in regimes.items() if regime[1] == compartment}
    for observation, compartment in sorted(set(regimes.values())):
        selections[f"{observation}_{compartment}"] = {
            dataset for dataset, regime in regimes.items() if regime == (observation, compartment)}
    conditional = {}
    for regime, datasets in selections.items():
        if not datasets:
            raise ValueError(f"Missing required loss-report regime: {regime}")
        losses = {arm: statistics.fmean(values[dataset] for dataset in datasets)
                  for arm, values in means.items()}
        vetoes = {arm: [heuristic for heuristic in HEURISTIC_ARMS
                       if losses[arm] > losses[heuristic]] for arm in PARTICLE_ARMS}
        conditional[regime] = {"datasets": len(datasets), "losses": losses,
                               "observed_heuristic_promotion_vetoes": vetoes}
    return {"conditional": conditional,
            "heuristic_dominance_verdict": "promotion_veto" if any(
                value for group in conditional.values()
                for value in group["observed_heuristic_promotion_vetoes"].values()) else "screen_pass",
            "interpretation": "Exposed-data replay; descriptive loss comparisons, no fresh ranking."}


def normalized_input_hashes(input_hashes, repository):
    normalized = {}
    for name, expected in input_hashes.items():
        path = Path(name)
        resolved = str((path if path.is_absolute() else repository / path).resolve())
        if resolved in normalized:
            raise ValueError(f"Duplicate input path alias: {name}")
        normalized[resolved] = expected
    return normalized


class PlanMetadataNormalizer(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "parser" and call.func.attr == "add_argument"
                and len(call.args) == 1 and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "--plan" and len(call.keywords) == 1
                and call.keywords[0].arg == "default"
                and isinstance(call.keywords[0].value, ast.Constant)
                and isinstance(call.keywords[0].value.value, str)):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "manifest" and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict"):
            for keyword in node.value.keywords:
                if keyword.arg != "plan":
                    continue
                value = keyword.value
                if not ((isinstance(value, ast.Constant) and isinstance(value.value, str))
                        or (isinstance(value, ast.Attribute) and value.attr == "plan"
                            and isinstance(value.value, ast.Name)
                            and value.value.id == "arguments")):
                    raise ValueError("Unsupported plan metadata expression")
                keyword.value = ast.Constant(value="normalized_plan_metadata")
        return self.generic_visit(node)


def verify_replay_drivers(manifests, baseline_harness=None, candidate_harness=None):
    if (baseline_harness is None) != (candidate_harness is None):
        raise ValueError("Both frozen replay drivers are required")
    hashes = {lane: manifest["script_sha256"] for lane, manifest in manifests.items()}
    if baseline_harness is None:
        if hashes["before"] != hashes["after"]:
            raise ValueError("Manifest mismatch: script_sha256; provide both frozen drivers")
        return dict(mode="identical_driver_hashes", hashes=hashes)
    normalized = {}
    for lane, path in (("before", baseline_harness), ("after", candidate_harness)):
        if digest(path) != hashes[lane]:
            raise ValueError(f"Frozen replay driver digest mismatch: {lane}")
        normalized[lane] = ast.dump(PlanMetadataNormalizer().visit(ast.parse(path.read_text())))
    if normalized["before"] != normalized["after"]:
        raise ValueError("Replay drivers differ beyond plan metadata")
    return dict(mode="identical_computation_after_plan_metadata_normalization", hashes=hashes)


def verify_pair(before_path, after_path, baseline_lane="before", *, repository=None,
                baseline_harness=None, candidate_harness=None):
    if baseline_lane not in ("before", "after"):
        raise ValueError("Unknown baseline lane")
    manifests = {"before": read_json(before_path / "manifest.json"),
                 "after": read_json(after_path / "manifest.json")}
    results = {"before": read_json(before_path / "result.json"),
               "after": read_json(after_path / "result.json")}
    repository = Path(repository) if repository is not None else Path(__file__).resolve().parents[1]
    input_hashes = {lane: normalized_input_hashes(manifest["input_hashes"], repository)
                   for lane, manifest in manifests.items()}
    if input_hashes["before"] != input_hashes["after"]:
        raise ValueError("Manifest mismatch: input_hashes")
    driver_check = verify_replay_drivers(manifests, baseline_harness, candidate_harness)
    equal_fields = ("custom_op_sha256", "controls", "configuration",
        "neural_updates", "data_version", "dataset_seeds", "particle_seed_base", "replicates",
        "reset_seed", "score_datasets", "dtype", "tf32", "jit_compile", "chunk", "horizon",
        "particles", "tensorflow", "visible_devices", "gpu_details",
        "xla_flags", "memory_policy")
    for name in equal_fields:
        if manifests["before"][name] != manifests["after"][name]:
            raise ValueError(f"Manifest mismatch: {name}")
    for lane, result in results.items():
        expected_lane = baseline_lane if lane == "before" else "after"
        if manifests[lane]["lane"] != expected_lane or result["status"] != "complete":
            raise ValueError(f"Incomplete or mislabeled lane: {lane}")
        if not all((result["input_hashes_unchanged"], result["isolated_imports"],
                    result["source_check"]["unchanged"])):
            raise ValueError(f"Provenance failure: {lane}")
        manifest = manifests[lane]
        if manifest["cpu_only"] or not manifest["gpu_devices"]:
            raise ValueError(f"Expected GPU execution: {lane}")
        if not manifest["memory_policy"]["all_physical_devices_memory_growth"]:
            raise ValueError(f"GPU memory policy failed: {lane}")
    for path, expected in input_hashes["before"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed after replay: {path}")
    report = {"status": "complete", "source_commits": {
        lane: manifest["git_commit"] for lane, manifest in manifests.items()},
        "inputs_and_controls_equal": True, "replay_driver_check": driver_check,
        "resolved_input_count": len(input_hashes["before"]), "results": results,
        "input_artifact_hashes": {}, "analyzer_sha256": digest(Path(__file__))}
    for filename, label in (("holdout.jsonl", "holdout"), ("scores.jsonl", "directional")):
        left = read_rows(before_path / filename, label == "directional")
        right = read_rows(after_path / filename, label == "directional")
        expected = expected_keys(manifests["before"], label == "directional")
        if left.keys() != expected or right.keys() != expected:
            raise ValueError(f"Incomplete or unexpected {label} grid: {len(left)}, {len(right)}")
        for lane, path in (("before", before_path), ("after", after_path)):
            count_field = "score_rows" if label == "directional" else "holdout_rows"
            if results[lane][count_field] != len(expected):
                raise ValueError(f"Manifest row count disagrees: {lane} {label}")
            report["input_artifact_hashes"][str(path / filename)] = digest(path / filename)
        report[label] = compare_rows(left, right)
        report[label]["explanatory_traces"] = {
            field: explanatory_trace(left, right, report[label]["fields"][field]["max_absolute_key"])
            for field in ("loss", "value", "score")}
        report[f"{label}_runtime"] = runtime_report(left, right)
        if label == "holdout":
            report["before_losses"] = loss_report(left)
            report["after_losses"] = loss_report(right)
    report["equivalence_passed"] = report["holdout"]["passed"] and report["directional"]["passed"]
    return report


def markdown(report):
    passed = report["equivalence_passed"]
    heuristic_veto = report["after_losses"]["heuristic_dominance_verdict"] == "promotion_veto"
    boundary_datasets = report["after_losses"]["conditional"]["compartment_nearer_boundary"]["datasets"]
    lines = ["# Frozen SIR post-merge replay", "",
        "The frozen SIR replay " + ("passes" if passed else "fails") +
        " the combined predeclared equivalence criterion. "
        f"Invalid rows: {len(report['holdout']['invalid_rows']) + len(report['directional']['invalid_rows'])}. "
        "Numerical-tolerance failures are reported separately below.", "",
        "| Quantity | Rows | Maximum absolute difference | Tolerance status |",
        "| --- | ---: | ---: | --- |"]
    for label in ("holdout", "directional"):
        for field, value in report[label]["fields"].items():
            lines.append(f"| {label}: {field} | {value['rows']} | "
                f"{value['max_absolute']:.6g} | {'pass' if value['passed'] else 'FAIL'} |")
    lines += ["", "Absolute-plus-relative tolerances are 1e-8; raw quantities use "
        "1e-5 + 1e-8 times the reference magnitude. Directional derivatives use three "
        "unit parameter directions on four prespecified datasets, four particle seeds "
        "and nine arms. Historical zero-direction fields are not score evidence.", "",
        "## Numerical differences by method", "",
        "| Arm | Maximum loss difference | Maximum value difference | Maximum directional-score difference | All fields within tolerance |",
        "| --- | ---: | ---: | ---: | --- |"]
    for arm in PARTICLE_ARMS:
        holdout = report["holdout"]["by_arm"][arm]
        directional = report["directional"]["by_arm"][arm]
        arm_passed = all(value["passed"] for fields in (holdout, directional)
                         for value in fields.values())
        lines.append(f"| {arm} | {holdout['loss']['max_absolute']:.6g} | "
            f"{max(holdout['value']['max_absolute'], directional['value']['max_absolute']):.6g} | "
            f"{directional['score']['max_absolute']:.6g} | {'pass' if arm_passed else 'FAIL'} |")
    lines += ["", "## Moment losses", "", "| Arm | Before | After |", "| --- | ---: | ---: |"]
    for arm in PARTICLE_ARMS + HEURISTIC_ARMS:
        before = report["before_losses"]["conditional"]["all"]["losses"][arm]
        after = report["after_losses"]["conditional"]["all"]["losses"][arm]
        lines.append(f"| {arm} | {before:.9f} | {after:.9f} |")
    lines += ["", "All losses use equal dataset weight after averaging particle replicates. "
        "These are previously exposed data; the table supplies no new generalization "
        "or superiority evidence.", "", "## Conditional heuristic comparisons", "",
        "| Regime | Datasets | Trained neural | Plain LEDH | Prediction | Observation only | Recursive Gaussian |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for regime, group in report["after_losses"]["conditional"].items():
        values = [group["losses"][arm] for arm in
                  ("trained", "plain", *HEURISTIC_ARMS)]
        lines.append(f"| {regime} | {group['datasets']} | " +
                     " | ".join(f"{value:.6f}" for value in values) + " |")
    lines += ["", "The JSON retains every arm in every marginal and joint regime. "
        f"Observed losses to a heuristic remain promotion vetoes. The {boundary_datasets} boundary "
        "datasets cannot establish a population ranking; latent-defined strata "
        "also do not inherit the conditional moment-MSE decomposition.", "",
        "## Runtime", "", "| Arm | Warm datasets | Before seconds/call | After seconds/call | After/before | 95% interval for difference |",
        "| --- | ---: | ---: | ---: | ---: | --- |"]
    for arm, value in report["holdout_runtime"]["warm_paired_dataset_summaries"].items():
        interval = value["difference_percentile_95"]
        lines.append(f"| {arm} | {value['datasets']} | {value['before_mean']:.6f} | "
            f"{value['after_mean']:.6f} | {value['ratio_of_means']:.4f} | "
            f"[{interval[0]:.6f}, {interval[1]:.6f}] |")
    lines += ["", "Cold calls are excluded in both lanes if either member is cold, and "
        "are retained separately in JSON with process origins. Intervals resample paired "
        "datasets 5,000 times, seed263050001. Sequential execution, process restarts and "
        "uncontrolled GPU contention prevent a causal speedup claim.", "",
        "## Decision", "", "| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {'Accept scoped replay equivalence' if passed else 'Investigate version discrepancy'} | "
        f"{'Pass' if passed else 'Fail'} | "
        f"{len(report['holdout']['invalid_rows']) + len(report['directional']['invalid_rows'])} invalid rows; "
        f"heuristic promotion veto {'retained' if heuristic_veto else 'not observed'} | Exposed FP64 T5/N64 scope only | "
        f"{'Preserve scoped equivalence evidence' if passed else 'Localize version differences before claiming equivalence'}; "
        "any later training study uses a fresh protocol | "
        "No true-score accuracy, candidate promotion, TF32 or long-horizon claim |", "",
        "| Inference status | Finding |", "| --- | --- |",
        f"| Hard veto screen | Replay {'passes' if passed else 'fails'}; see numerical and validity JSON |",
        "| Statistically supported ranking | None newly established by this replay |",
        "| Descriptive differences | Moment losses and sequential-process timings |",
        "| Default-readiness | Not evaluated; this replay cannot promote a guide |",
        "| Next evidence needed | Fresh independently evaluated training protocol for guide quality; "
        "interleaved isolated timing repetitions for causal performance claims |", "",
        "## Limitations and red-team check", "",
        "The strongest alternative explanation for timing differences is resource contention "
        "or process compilation rather than the merge. Numerical parity does not detect a "
        "bug shared by both snapshots. Source and frozen-input hashes, complete pairing, "
        "archive reproduction, nonzero-direction checks and validity records address "
        "replay integrity, not independent algorithm correctness. The original campaign "
        "did not archive the custom-op binary hash; both replay lanes share the current "
        "hashed binary. A discrepancy at a different scope could overturn scope transfer, "
        "but does not contradict the measured result here.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    try:
        report = verify_pair(arguments.before, arguments.after)
    except (ValueError, KeyError, OSError) as error:
        report = {"status": "incomplete_or_invalid", "equivalence_passed": False,
                  "error": repr(error)}
    report["command"] = sys.argv
    (arguments.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if report["status"] == "complete":
        (arguments.output / "comparison.md").write_text(markdown(report))
    print(json.dumps({key: report[key] for key in ("status", "equivalence_passed")}))
    return 0 if report["equivalence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
