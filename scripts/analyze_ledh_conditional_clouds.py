"""Diagnostic-only NumPy analysis of fixed datasets; never runtime admission."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from bayesfilter.score_study.conditional_cloud_design import (
    ARMS, FIXED_BETAS, HEURISTICS, PARTICLE_COUNTS, REPLICATES,
    cloud_seed, data_seed, evaluation_cases)


def conditional_moments(errors):
    errors = np.asarray(errors, dtype=float)
    count = len(errors)
    if count < 2:
        raise ValueError("conditional variance needs repeated clouds")
    mean = errors.mean(axis=0)
    variance = errors.var(axis=0, ddof=1)
    return {"mean_error": mean.tolist(), "variance": variance.tolist(),
            "mean_error_mcse": np.sqrt(variance / count).tolist(),
            "unbiased_squared_bias_estimate": (mean ** 2 - variance / count).tolist(),
            "mean_squared_error": (errors ** 2).mean(axis=0).tolist()}


def analyze(rows, *, resamples=10000):
    if resamples < 2:
        raise ValueError("at least two resamples required")
    expected = {(particles, *case, arm) for particles in PARTICLE_COUNTS
                for case in evaluation_cases() for arm in ARMS}
    keyed, data_seen, inputs_seen, frozen = {}, {}, {}, {}
    for row in rows:
        particles, regime, partition, dataset, replicate, arm = key = tuple(
            row[field] for field in ("particles", "regime", "partition", "dataset", "replicate", "arm"))
        if key not in expected or key in keyed:
            raise ValueError("unexpected or duplicate row")
        if row["data_seed"] != data_seed(regime, partition, dataset):
            raise ValueError("wrong data partition or seed")
        if row["cloud_seed"] != cloud_seed(regime, partition, dataset, replicate, particles):
            raise ValueError("wrong cloud seed")
        if row["trace_min"]["valid"] != 1.:
            raise ValueError("invalid numerical row")
        for field, length in (("score", 4), ("reference_score", 4), ("squared_error", 4), ("beta", 3)):
            if len(row[field]) != length or not all(math.isfinite(value) for value in row[field]):
                raise ValueError("nonfinite or wrong-shaped row")
        if not math.isfinite(row["value_error"]):
            raise ValueError("nonfinite likelihood error")
        if min(row["beta"]) < 0. or not math.isclose(sum(row["beta"]), 1., abs_tol=1e-12):
            raise ValueError("invalid mixture probabilities")
        if not np.allclose((np.asarray(row["score"]) - row["reference_score"]) ** 2,
                           row["squared_error"], rtol=1e-12, atol=1e-14):
            raise ValueError("score error disagrees with reference")
        data_key = regime, partition, dataset
        data_identity = row["data_sha256"], row["reference_score"]
        if data_key in data_seen and data_seen[data_key] != data_identity:
            raise ValueError("fixed data or reference changes across clouds/counts")
        data_seen[data_key] = data_identity
        input_key = particles, regime, partition, dataset, replicate
        identity = row["paired_input_sha256"], row["prefix32_sha256"]
        if input_key in inputs_seen and inputs_seen[input_key] != identity:
            raise ValueError("arms are not paired on cloud inputs")
        inputs_seen[input_key] = identity
        beta_key = particles, regime, arm
        if beta_key in frozen and frozen[beta_key] != row["beta"]:
            raise ValueError("beta changes after calibration")
        frozen[beta_key] = row["beta"]
        keyed[key] = row
    if set(keyed) != expected:
        raise ValueError("incomplete comparison")
    for particles in PARTICLE_COUNTS:
        for regime in range(2):
            for dataset in range(2):
                hashes = {inputs_seen[(particles, regime, 2, dataset, replicate)][0]
                          for replicate in range(REPLICATES)}
                if len(hashes) != REPLICATES:
                    raise ValueError("repeated cloud inputs masquerade as independent replicates")
    for regime, partition, dataset, replicate in evaluation_cases():
        lower = inputs_seen[(32, regime, partition, dataset, replicate)]
        upper = inputs_seen[(64, regime, partition, dataset, replicate)]
        if lower[0] != lower[1] or lower[1] != upper[1]:
            raise ValueError("particle counts do not share the 32-particle prefix")
    for particles in PARTICLE_COUNTS:
        for regime in range(2):
            for current, future in (("current_global", "future_global"),
                                    ("current_calibrated_future", "future_calibrated_future")):
                if frozen[(particles, regime, current)] != frozen[(particles, regime, future)]:
                    raise ValueError("guidance contrast changes beta")
            for arm in HEURISTICS:
                policy = arm.removeprefix("current_")
                if frozen[(particles, regime, arm)] != FIXED_BETAS[policy]:
                    raise ValueError("fixed heuristic changed")
    summaries, contrasts, losses = [], [], []
    for regime in range(2):
        for dataset in range(2):
            errors = {}
            draws = np.random.default_rng(517000 + 100 * regime + dataset).integers(
                0, REPLICATES, size=(resamples, REPLICATES))
            for particles in PARTICLE_COUNTS:
                for arm in ARMS:
                    selected = [keyed[(particles, regime, 2, dataset, replicate, arm)]
                                for replicate in range(REPLICATES)]
                    signed = np.asarray([row["score"] for row in selected]) - selected[0]["reference_score"]
                    errors[(particles, arm)] = signed ** 2
                    summaries.append({"particles": particles, "regime": regime, "dataset": dataset,
                        "data_seed": data_seed(regime, 2, dataset), "arm": arm, "clouds": REPLICATES,
                        "beta": selected[0]["beta"], **conditional_moments(signed),
                        "min_ess": min(row["trace_min"]["ess"] for row in selected),
                        "max_weight": max(row["trace_max"]["max_weight"] for row in selected),
                        "cloud_squared_errors": errors[(particles, arm)].tolist()})

            def contrast(name, terms, role, particles=None):
                differences = sum(coefficient * errors[(count, arm)] for count, arm, coefficient in terms)
                interval = np.quantile(differences[draws].mean(axis=1), [.025, .975], axis=0).T
                record = {"regime": regime, "dataset": dataset, "particles": particles,
                          "contrast": name, "role": role, "terms": terms,
                          "mean_squared_error_difference": differences.mean(axis=0).tolist(),
                          "paired_cloud_interval_95": interval.tolist()}
                contrasts.append(record)
                return record

            for particles in PARTICLE_COUNTS:
                for policy in ("global", "calibrated_future"):
                    contrast(f"guidance_at_{policy}", [(particles, f"future_{policy}", 1),
                             (particles, f"current_{policy}", -1)],
                             "primary" if policy == "global" else "secondary", particles)
                contrast("own_calibration_guidance", [(particles, "future_calibrated_future", 1),
                         (particles, "current_calibrated_current", -1)], "explanatory", particles)
                for candidate in ("future_global", "future_calibrated_future"):
                    for heuristic in HEURISTICS:
                        record = contrast(f"{candidate}_versus_{heuristic}",
                            [(particles, candidate, 1), (particles, heuristic, -1)],
                            "heuristic_promotion_veto", particles)
                        if record["mean_squared_error_difference"][2] > 0.:
                            losses.append({"particles": particles, "regime": regime, "dataset": dataset,
                                "candidate": candidate, "heuristic": heuristic,
                                "log_process_sd_mse_difference": record["mean_squared_error_difference"][2]})
            for arm in ARMS:
                contrast(f"particle_count_{arm}", [(64, arm, 1), (32, arm, -1)],
                         "primary" if arm in ("current_global", "future_global") else "explanatory")
            for policy in ("global", "calibrated_future"):
                contrast(f"particle_count_guidance_interaction_{policy}",
                    [(64, f"future_{policy}", 1), (64, f"current_{policy}", -1),
                     (32, f"future_{policy}", -1), (32, f"current_{policy}", 1)],
                    "primary" if policy == "global" else "explanatory")
    return {"status": "complete_conditional_diagnostic", "rows": len(rows),
            "coordinate_order": ["alpha0", "alpha1", "log_process_sd", "log_observation_sd"],
            "resampling_unit": "paired particle cloud, conditional on each fixed dataset",
            "resamples": resamples, "bootstrap_seed_rule": "517000+100*regime+dataset",
            "uncertainty_limit": "pointwise exploratory intervals; twelve clouds and two datasets per regime; no population ranking",
            "summaries": summaries, "contrasts": contrasts, "heuristic_losses": losses,
            "heuristic_dominance_verdict": "VETO" if losses else "PASS_DESCRIPTIVE_SCREEN_ONLY",
            "canonical_admission": False, "default_ready": False, "population_ranking_supported": False}


def load_scope(directory, particles):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["particles"] != particles or manifest["row_count"] != 416:
        raise ValueError("completed matching scope required")
    calibrations = json.loads((directory / "calibration.json").read_text())
    calibrated = {(row["regime"], row["guide"]): row for row in calibrations}
    if len(calibrations) != 4 or set(calibrated) != {(regime, guide) for regime in range(2) for guide in ("current", "exact")}:
        raise ValueError("missing calibration scope")
    for row in calibrations:
        if not row["converged"] or row["particles"] != particles:
            raise ValueError("invalid or transferred calibration")
        if not all(math.isfinite(row[field]) for field in ("gap", "objective")):
            raise ValueError("nonfinite calibration certificate")
        if row["gap"] > 1e-7 * max(1., abs(row["objective"])):
            raise ValueError("calibration certificate exceeds tolerance")
    datasets = {}
    for line in (directory / "datasets.jsonl").read_text().splitlines():
        record = json.loads(line)
        digest = hashlib.sha256(json.dumps([record["observations"]], allow_nan=False).encode()).hexdigest()
        if digest != record["data_sha256"]:
            raise ValueError("stored observations disagree with data hash")
        key = record["regime"], record["partition"], record["dataset"]
        if key in datasets:
            raise ValueError("duplicate dataset record")
        datasets[key] = record
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines()]
    for row in rows:
        if row["particles"] != particles:
            raise ValueError("row outside manifest scope")
        record = datasets[(row["regime"], row["partition"], row["dataset"])]
        if row["data_sha256"] != record["data_sha256"] or row["reference_score"] != record["reference_score"]:
            raise ValueError("row disagrees with stored dataset/reference")
        guide = "current" if row["arm"] == "current_calibrated_current" else "exact"
        if "calibrated" in row["arm"] and row["beta"] != calibrated[(row["regime"], guide)]["beta"]:
            raise ValueError("evaluation beta differs from frozen calibration")
    return manifest, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n32", type=Path, required=True)
    parser.add_argument("--n64", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    lower_manifest, lower_rows = load_scope(arguments.n32, 32)
    upper_manifest, upper_rows = load_scope(arguments.n64, 64)
    for field in ("git_commit", "source_sha256", "controls", "data_version", "theta", "horizon", "dtype", "tf32", "jit_compile"):
        if lower_manifest[field] != upper_manifest[field]:
            raise ValueError(f"scope mismatch: {field}")
    decision = analyze(lower_rows + upper_rows)
    decision["manifests"] = [str(arguments.n32 / "manifest.json"), str(arguments.n64 / "manifest.json")]
    decision["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with arguments.output.open("x") as stream:
        stream.write(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: decision[key] for key in ("status", "rows", "heuristic_dominance_verdict")}))


if __name__ == "__main__":
    main()
