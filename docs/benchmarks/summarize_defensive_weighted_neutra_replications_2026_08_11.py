#!/usr/bin/env python3
"""Summarize eight independent two-mode weighted-NeuTra replications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PLAN = "docs/plans/bayesfilter-defensive-weighted-neutra-validation-plan-2026-08-11.md"
T_CRITICAL_DF7_975 = 2.364624251
TARGET_MINOR_PROBABILITY = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--replication-zero-name", default="unequal-080-020-budget2000-v1"
    )
    parser.add_argument(
        "--replication-prefix", default="unequal-080-020-budget2000-replication"
    )
    parser.add_argument("--expected-hidden-width", type=int, default=32)
    parser.add_argument("--expected-stages", type=int, default=3)
    parser.add_argument(
        "--replication-names",
        nargs=8,
        help="Explicit artifact directory names for replications 0 through 7",
    )
    parser.add_argument(
        "--expected-replications",
        nargs=8,
        type=int,
        help="Expected replication IDs corresponding to --replication-names",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    roots = _replication_roots(input_root, args)
    expected_replications = _expected_replication_ids(args)
    rows = [
        _read_replication(
            path,
            expected_replications[index],
            expected_hidden_width=int(args.expected_hidden_width),
            expected_stages=int(args.expected_stages),
        )
        for index, path in enumerate(roots)
    ]
    weighted = [float(row["weighted_minor_probability"]) for row in rows]
    reverse = [float(row["reverse_minor_probability"]) for row in rows]
    weighted_interval = _interval(weighted)
    reverse_interval = _interval(reverse)
    all_finite = all(bool(row["finite"]) for row in rows)
    all_components = all(bool(row["both_components_observed"]) for row in rows)
    truth_in_weighted_interval = (
        weighted_interval["lower"]
        <= TARGET_MINOR_PROBABILITY
        <= weighted_interval["upper"]
    )
    hard_veto = not all_finite or not all_components
    passed = not hard_veto and truth_in_weighted_interval
    result = {
        "schema": "bayesfilter.defensive_weighted_neutra_replication_summary.v1",
        "question": (
            "Across eight independent training and audit replications, does the "
            "weighted transport recover the analytic minority probability 0.2?"
        ),
        "replications": rows,
        "weighted_minor_probability_interval": weighted_interval,
        "reverse_kl_minor_probability_interval": reverse_interval,
        "decision_table": {
            "decision": "r1_replication_gate_passed" if passed else "r1_repair_required",
            "primary_criterion_status": (
                "pass_truth_inside_weighted_95pct_t_interval"
                if truth_in_weighted_interval
                else "fail_truth_outside_weighted_95pct_t_interval"
            ),
            "veto_diagnostic_status": (
                "pass_finite_and_all_components_observed"
                if not hard_veto
                else "fail_nonfinite_or_missing_component"
            ),
            "main_uncertainty": (
                "eight training replications quantify seed sensitivity on one target; "
                "they do not establish transfer to other targets"
            ),
            "next_justified_action": (
                "run the remaining predeclared r1 target variants"
                if passed
                else "repair capacity or componentwise structure before later rungs"
            ),
            "not_concluded": (
                "no HMC, posterior-correctness, cross-target, SSL-LSTM, or default claim"
            ),
        },
        "inference_status": {
            "hard_veto_screen": "pass" if not hard_veto else "fail",
            "statistically_supported_ranking": "none_not_tested",
            "descriptive_only_differences": (
                "weighted versus reverse-KL means are unpaired descriptive comparators"
            ),
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "equal-weight, unequal-covariance, rare-mode, and four-mode analytic rungs"
            ),
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "the fixed architecture and 2,000-update budget may be adequate only for "
                "this known, separated, defensively proposed two-mode target"
            ),
            "overturning_result": (
                "truth outside an independent replication interval or any missing mode"
            ),
            "weakest_evidence": (
                "only one target geometry and one hyperparameter setting are replicated"
            ),
        },
        "passed": passed,
        "numeric_provenance": {
            "replication_count": "predeclared plan minimum eight",
            "critical_value": (
                "Student-t 97.5th percentile for seven degrees of freedom, inherited "
                "from the reviewed analytic helper"
            ),
            "target_probability": "analytic normalized mixture weight",
        },
        "manifest": {
            "schema": "bayesfilter.defensive_weighted_neutra_replication_summary_manifest.v1",
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), check=True, capture_output=True, text=True
            ).stdout.strip(),
            "command": " ".join(sys.argv),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_file": PLAN,
            "input_roots": [str(path) for path in roots],
            "input_result_sha256": {
                str(path): _sha256(path / "result.json") for path in roots
            },
            "output_root": str(output_root),
            "backend": "python_standard_library_diagnostic_reporting_only",
            "cpu_gpu_status": "post_run_read_only_cpu_aggregation",
        },
    }
    _write(output_root / "result.json", result)
    _write(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.defensive_weighted_neutra_replication_summary_hashes.v1",
            "artifacts": {"result.json": _sha256(output_root / "result.json")},
        },
    )
    print(json.dumps({"completed": True, "passed": passed, "output_root": str(output_root)}))
    return 0


def _replication_roots(input_root: Path, args: argparse.Namespace) -> list[Path]:
    if args.replication_names:
        return [input_root / name for name in args.replication_names]
    return [input_root / args.replication_zero_name] + [
        input_root / f"{args.replication_prefix}-{index}-v1"
        for index in range(1, 8)
    ]


def _expected_replication_ids(args: argparse.Namespace) -> list[int]:
    if args.expected_replications is not None:
        if args.replication_names is None:
            raise ValueError("--expected-replications requires --replication-names")
        values = list(args.expected_replications)
    else:
        values = list(range(8))
    if len(set(values)) != 8:
        raise ValueError("expected replication IDs must be distinct")
    return values


def _read_replication(
    path: Path,
    expected_replication: int,
    *,
    expected_hidden_width: int,
    expected_stages: int,
) -> Mapping[str, Any]:
    result_path = path / "result.json"
    hashes_path = path / "artifact_hashes.json"
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    with hashes_path.open("r", encoding="utf-8") as handle:
        declared = json.load(handle)["artifacts"]["result.json"]
    if _sha256(result_path) != declared:
        raise ValueError(f"result hash mismatch: {result_path}")
    replication = int(result.get("replication", 0))
    if replication != expected_replication:
        raise ValueError(
            f"replication identity mismatch: expected {expected_replication}, got {replication}"
        )
    if result.get("mode") != "two-mode-canary":
        raise ValueError(f"wrong result mode: {result_path}")
    config = result["config"]
    if config["hidden_layers"] != [expected_hidden_width, expected_hidden_width]:
        raise ValueError(f"hidden-width identity mismatch: {result_path}")
    if int(config["stages"]) != expected_stages:
        raise ValueError(f"stage identity mismatch: {result_path}")
    coverage = result["audit"]["base_component_coverage"]
    weighted = coverage["weighted"]
    reverse = coverage["reverse_kl"]
    return {
        "replication": replication,
        "result_path": str(result_path),
        "result_sha256": declared,
        "weighted_minor_probability": float(
            weighted["soft_responsibility_component_probabilities"][1]
        ),
        "reverse_minor_probability": float(
            reverse["soft_responsibility_component_probabilities"][1]
        ),
        "finite": bool(weighted["all_finite"] and reverse["all_finite"]),
        "both_components_observed": bool(weighted["both_components_observed"]),
        "weighted_audit_nll": float(result["audit"]["weighted"]["weighted_nll"]),
        "reverse_audit_nll": float(result["audit"]["reverse_kl"]["weighted_nll"]),
        "weighted_selected_update": int(result["checkpoint_selection"]["weighted_update"]),
        "importance_ess_fraction": float(
            result["importance_audit"]["effective_sample_size_fraction"]
        ),
        "wall_time_seconds": float(result["run_manifest"]["wall_time_seconds"]),
    }


def _interval(values: list[float]) -> Mapping[str, Any]:
    if len(values) != 8:
        raise ValueError("exactly eight independent replications are required")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = T_CRITICAL_DF7_975 * standard_error
    return {
        "batch_count": len(values),
        "values": values,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_value": T_CRITICAL_DF7_975,
        "confidence_level": 0.95,
        "lower": mean - half_width,
        "upper": mean + half_width,
        "truth": TARGET_MINOR_PROBABILITY,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
