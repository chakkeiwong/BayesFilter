"""Diagnostic-only NumPy reporting; never runtime selection or admission."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from bayesfilter.score_study.guidance_factorial_design import ARMS, HEURISTICS, POLICIES, SEED_BASE


def analyze(rows, *, resamples=10000):
    if resamples < 2:
        raise ValueError("at least two resamples required")
    expected = {(condition, partition, SEED_BASE + 1000 * condition + 100 * number + index, arm)
                for condition in range(4)
                for number, partition, count in ((1, "validation", 2), (2, "final", 12))
                for index in range(count) for arm in ARMS}
    keyed = {}
    for row in rows:
        key = (row["condition"], row["partition"], row["seed"], row["arm"])
        if key not in expected or key in keyed:
            raise ValueError("unexpected or duplicate row")
        if row["trace_min"]["valid"] != 1.:
            raise ValueError("invalid numerical row")
        if any(len(row[field]) != length for field, length in
               (("score", 4), ("reference_score", 4), ("squared_error", 4), ("beta", 3))):
            raise ValueError("wrong coordinate or beta shape")
        scalars = row["score"] + row["reference_score"] + row["squared_error"] + row["beta"]
        if not all(math.isfinite(value) for value in scalars + [row["value_error"]]):
            raise ValueError("nonfinite result")
        if min(row["beta"]) < 0. or not math.isclose(sum(row["beta"]), 1., abs_tol=1e-12):
            raise ValueError("invalid probabilities")
        for score, reference, error in zip(row["score"], row["reference_score"], row["squared_error"]):
            if not math.isclose((score - reference) ** 2, error, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("score error disagrees with oracle")
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("incomplete comparison")
    summaries, contrasts, losses = [], [], []
    for condition in range(4):
        frozen = {}
        for number, partition, count in ((1, "validation", 2), (2, "final", 12)):
            for index in range(count):
                seed = SEED_BASE + 1000 * condition + 100 * number + index
                matched = {arm: keyed[(condition, partition, seed, arm)] for arm in ARMS}
                reference = matched["identity"]["reference_score"]
                fingerprint = matched["identity"]["paired_input_sha256"]
                if not fingerprint or any(row["reference_score"] != reference or
                                          row["paired_input_sha256"] != fingerprint
                                          for row in matched.values()):
                    raise ValueError("unpaired data, randomness or reference")
                for policy in POLICIES:
                    current = matched[f"current_{policy}"]["beta"]
                    if matched[f"future_{policy}"]["beta"] != current:
                        raise ValueError("guidance comparison changes beta")
                    if policy in frozen and frozen[policy] != current:
                        raise ValueError("beta changes across evaluation datasets")
                    frozen[policy] = current
                for policy, probabilities in (("global", [.1, .9, 0.]),
                                              ("local", [.1, 0., .9]),
                                              ("equal", [1. / 3.] * 3)):
                    if matched[f"current_{policy}"]["beta"] != probabilities:
                        raise ValueError("fixed heuristic changed")
                if matched["identity"]["beta"] != [1., 0., 0.]:
                    raise ValueError("identity heuristic changed")
        seeds = [SEED_BASE + 1000 * condition + 200 + index for index in range(12)]
        errors = {}
        for arm in ARMS:
            selected = [keyed[(condition, "final", seed, arm)] for seed in seeds]
            errors[arm] = np.asarray([row["squared_error"] for row in selected])
            summaries.append({"condition": condition, "arm": arm, "datasets": 12,
                              "mean_squared_error": errors[arm].mean(axis=0).tolist(),
                              "mean_value_squared_error": sum(row["value_error"] ** 2 for row in selected) / 12,
                              "beta": selected[0]["beta"]})
        draws = np.random.default_rng(227000 + condition).integers(0, 12, size=(resamples, 12))

        def contrast(name, terms, role):
            differences = sum(coefficient * errors[arm] for arm, coefficient in terms.items())
            interval = np.quantile(differences[draws].mean(axis=1), [.025, .975], axis=0).T
            result = {"condition": condition, "contrast": name, "role": role, "terms": terms,
                      "mean_error_difference": differences.mean(axis=0).tolist(),
                      "paired_percentile_interval_95": interval.tolist()}
            contrasts.append(result)
            return result

        for policy in POLICIES:
            contrast(f"guidance_at_{policy}", {f"future_{policy}": 1, f"current_{policy}": -1},
                     "primary" if policy == "global" else "explanatory")
        for guide in ("current", "future"):
            contrast(f"current_calibration_at_{guide}",
                     {f"{guide}_calibrated_current": 1, f"{guide}_global": -1}, "explanatory")
            contrast(f"future_vs_current_calibration_at_{guide}",
                     {f"{guide}_calibrated_future": 1, f"{guide}_calibrated_current": -1}, "explanatory")
        contrast("guidance_calibration_interaction",
                 {"future_calibrated_future": 1, "current_calibrated_future": -1,
                  "future_calibrated_current": -1, "current_calibrated_current": 1}, "explanatory")
        for policy in POLICIES:
            candidate = f"future_{policy}"
            for heuristic in HEURISTICS:
                result = contrast(f"{candidate}_versus_{heuristic}",
                                  {candidate: 1, heuristic: -1}, "promotion_veto")
                if result["mean_error_difference"][2] > 0.:
                    losses.append({"condition": condition, "candidate": candidate,
                                   "heuristic": heuristic,
                                   "mean_difference": result["mean_error_difference"][2],
                                   "interval": result["paired_percentile_interval_95"][2]})
    primary = [row for row in contrasts if row["role"] == "primary"]
    primary_losses = [row for row in losses if row["candidate"] == "future_global"]
    return {"status": "complete_diagnostic", "rows": len(rows), "resamples": resamples,
            "coordinate_order": ["drift_0", "drift_1", "log_process_sd", "log_observation_sd"],
            "bootstrap_seed_rule": "227000+condition",
            "resampling_unit": "paired independent dataset and evaluation randomness",
            "summaries": summaries, "contrasts": contrasts, "heuristic_losses": losses,
            "heuristic_dominance_verdict": "VETO" if primary_losses else "NO_DESCRIPTIVE_LOSS",
            "nomination_criterion_passed": not primary_losses and all(
                row["paired_percentile_interval_95"][2][1] < 0. for row in primary),
            "default_ready": False, "canonical_admission": False,
            "uncertainty_limit": "12 clusters per regime; pointwise exploratory intervals, no multiplicity correction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    manifest = json.loads((arguments.input / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest.get("guidance_factorial"):
        raise ValueError("not a completed factorial run")
    rows = [json.loads(line) for line in (arguments.input / "rows.jsonl").read_text().splitlines()]
    result = analyze(rows)
    result["manifest"] = str(arguments.input / "manifest.json")
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "rows", "nomination_criterion_passed")}))


if __name__ == "__main__":
    main()
