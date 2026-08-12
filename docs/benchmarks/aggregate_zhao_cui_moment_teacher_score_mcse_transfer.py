#!/usr/bin/env python3
"""Aggregate paired TF32 score drift against target-specific reference MCSE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_NAMES = ("phi1", "phi2", "phi3", "q_scale", "r_scale")
SCHEMA = "bayesfilter.zhao_cui_moment_teacher_score_mcse_transfer_result.v1"
RATIO_THRESHOLD = 0.1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _mcse(values: list[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values))


def _exact_two_sided_sign_p(differences: list[float]) -> float:
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    count = positive + negative
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, index) for index in range(min(positive, negative) + 1))
    return min(1.0, 2.0 * tail / (2**count))


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf32-result", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    tf32_payload = _load(args.tf32_result.resolve())
    reference_payload = _load(args.reference_result.resolve())
    tf32 = tf32_payload["result"]
    reference = reference_payload["result"]
    seeds = list(reference["estimator_seeds"])
    identity_checks = {
        "campaign": tf32_payload["campaign_id"] == reference_payload["campaign_id"],
        "arm_labels": (
            tf32_payload["precision_arm"] == "tf32"
            and reference_payload["precision_arm"] == "fp32_no_tf32_reference"
        ),
        "current_git_commit": (
            tf32_payload["git_commit"] == reference_payload["git_commit"]
        ),
        "source_sha256": tf32_payload["source_sha256"] == reference_payload["source_sha256"],
        "seeds": tf32["estimator_seeds"] == seeds,
        "seed_count_at_least_eight": len(seeds) >= 8,
        "prepared_inputs": (
            tf32["preparation_identity"] == reference["preparation_identity"]
        ),
        "scope": all(
            tf32[name] == reference[name]
            for name in (
                "arm",
                "time_steps",
                "num_particles",
                "balance_steps",
                "sinkhorn_steps",
            )
        ),
        "dtype": (
            tf32["device"]["dtype"] == "float32"
            and reference["device"]["dtype"] == "float32"
        ),
        "tf32_modes": (
            tf32["device"]["tf32_enabled"] is True
            and reference["device"]["tf32_enabled"] is False
        ),
    }
    validity_checks = {
        "tf32_hard_valid": bool(tf32["hard_valid"]),
        "reference_hard_valid": bool(reference["hard_valid"]),
        "tf32_finite": bool(tf32["finite"]),
        "reference_finite": bool(reference["finite"]),
    }
    interpretable = all(identity_checks.values()) and all(validity_checks.values())

    rows = []
    value_diagnostic = None
    if interpretable:
        tf32_scores = tf32["per_seed_physical_score"]
        reference_scores = reference["per_seed_physical_score"]
        candidate_values = [float(value) for value in tf32["per_seed_value"]]
        baseline_values = [float(value) for value in reference["per_seed_value"]]
        value_differences = [
            left - right
            for left, right in zip(candidate_values, baseline_values, strict=True)
        ]
        value_reference_mcse = _mcse(baseline_values)
        value_mean_difference = _mean(value_differences)
        value_diagnostic = {
            "role": "explanatory_only_not_part_of_score_promotion_criterion",
            "tf32_mean_value": _mean(candidate_values),
            "reference_mean_value": _mean(baseline_values),
            "mean_paired_numerical_difference": value_mean_difference,
            "maximum_absolute_per_seed_difference": max(
                abs(value) for value in value_differences
            ),
            "reference_value_sd": statistics.stdev(baseline_values),
            "reference_mean_mcse": value_reference_mcse,
            "paired_difference_mcse": _mcse(value_differences),
            "absolute_mean_difference_over_reference_mcse": (
                abs(value_mean_difference) / value_reference_mcse
                if value_reference_mcse > 0.0
                else (0.0 if value_mean_difference == 0.0 else math.inf)
            ),
        }
        for index, name in enumerate(PARAMETER_NAMES):
            candidate = [float(row[index]) for row in tf32_scores]
            baseline = [float(row[index]) for row in reference_scores]
            differences = [
                left - right for left, right in zip(candidate, baseline, strict=True)
            ]
            reference_mcse = _mcse(baseline)
            mean_difference = _mean(differences)
            paired_difference_mcse = _mcse(differences)
            sign_p = _exact_two_sided_sign_p(differences)
            ratio = (
                abs(mean_difference) / reference_mcse
                if reference_mcse > 0.0
                else (0.0 if mean_difference == 0.0 else math.inf)
            )
            rows.append(
                {
                    "parameter": name,
                    "tf32_mean_score": _mean(candidate),
                    "reference_mean_score": _mean(baseline),
                    "mean_paired_numerical_difference": mean_difference,
                    "maximum_absolute_per_seed_difference": max(
                        abs(value) for value in differences
                    ),
                    "reference_score_sd": statistics.stdev(baseline),
                    "reference_mean_mcse": reference_mcse,
                    "paired_difference_mcse": paired_difference_mcse,
                    "positive_difference_count": sum(
                        value > 0.0 for value in differences
                    ),
                    "negative_difference_count": sum(
                        value < 0.0 for value in differences
                    ),
                    "zero_difference_count": sum(
                        value == 0.0 for value in differences
                    ),
                    "exact_two_sided_sign_test_p": sign_p,
                    "systematic_displacement_supported": (
                        sign_p <= 0.05
                        and abs(mean_difference) > 2.0 * paired_difference_mcse
                    ),
                    "absolute_mean_difference_over_reference_mcse": ratio,
                    "passed": ratio <= RATIO_THRESHOLD,
                    "passed_practical_half_mcse_screen": ratio <= 0.5,
                }
            )
    maximum_ratio = (
        max(row["absolute_mean_difference_over_reference_mcse"] for row in rows)
        if rows
        else None
    )
    passed = interpretable and all(row["passed"] for row in rows)
    practical_half_mcse_passed = interpretable and all(
        row["passed_practical_half_mcse_screen"] for row in rows
    )
    payload = {
        "schema": SCHEMA,
        "classification": "canonical_lgssm_transfer_evidence_not_moment_teacher_score_evidence",
        "question": "Is paired TF32 final-score drift at most 0.1 reference MCSE in every coordinate?",
        "moment_teacher_final_score_tested": False,
        "criterion": {
            "ratio": "abs(mean(tf32-reference))/mcse(reference mean)",
            "maximum_allowed": RATIO_THRESHOLD,
            "all_coordinates_must_pass": True,
        },
        "identity_checks": identity_checks,
        "validity_checks": validity_checks,
        "interpretable": interpretable,
        "rows": rows,
        "value_diagnostic": value_diagnostic,
        "maximum_ratio": maximum_ratio,
        "passed": passed,
        "practical_half_mcse_screen": {
            "maximum_allowed": 0.5,
            "all_coordinates_passed": practical_half_mcse_passed,
        },
        "verdict": (
            "pass_for_exact_canonical_lgssm_scope_transfer_only"
            if passed
            else (
                "fail_score_drift_not_small_relative_to_mcse"
                if interpretable
                else "not_interpretable_due_to_identity_or_validity_veto"
            )
        ),
        "inputs": {
            "tf32_result": str(args.tf32_result),
            "tf32_sha256": _sha256(args.tf32_result.resolve()),
            "reference_result": str(args.reference_result),
            "reference_sha256": _sha256(args.reference_result.resolve()),
        },
        "nonclaims": [
            "the Zhao-Cui moment-teacher final score is not implemented or tested",
            "no long-horizon, nonlinear, HMC, or default-readiness conclusion",
            "no claim that every intermediate near-zero derivative can be ignored",
        ],
    }
    result_path = output / "result.json"
    _write_json(result_path, payload)
    manifest = {
        "schema": "bayesfilter.run_manifest.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": " ".join(sys.argv),
        "environment": "CPU standard-library aggregation of trusted GPU node artifacts",
        "device": "CPU aggregation; GPU provenance is in both node manifests",
        "data_version": "paired prepared-input identity embedded in node results",
        "seeds": seeds,
        "wall_time_seconds": "N/A_negligible_offline_aggregation",
        "output_artifact": str(output.relative_to(ROOT)),
        "plan": "docs/plans/bayesfilter-zhao-cui-moment-teacher-score-mcse-transfer-plan-2026-07-30.md",
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
    }
    _write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"verdict": payload["verdict"], "maximum_ratio": maximum_ratio}, indent=2))


if __name__ == "__main__":
    main()
