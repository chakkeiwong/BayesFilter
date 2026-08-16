#!/usr/bin/env python3
"""Verify and summarize four frozen-kernel analytic HMC replications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-defensive-weighted-neutra-analytic-hmc-replication-plan-2026-08-12.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _summary_stats(values: list[float]) -> Mapping[str, Any]:
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("exactly four finite replication values are required")
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "role": "descriptive_between_root_seed_variability_only",
    }


def _verify_archive(result: Mapping[str, Any]) -> Mapping[str, Any]:
    sequential = result["sequential"]
    manifest_path = Path(sequential["archive"]["manifest_path"])
    if _sha256(manifest_path) != sequential["archive"]["manifest_sha256"]:
        raise RuntimeError(f"archive manifest receipt mismatch: {manifest_path}")
    manifest = _read(manifest_path)
    receipt_count = 0
    for phase in ("warmup_chunks", "retained_chunks"):
        for row in manifest.get(phase, ()):
            receipts = (
                row["sample_receipt"],
                *row["trace_receipts"].values(),
                row["receipt"],
            )
            for receipt in receipts:
                path = Path(receipt["path"])
                if not path.is_file() or _sha256(path) != receipt["sha256"]:
                    raise RuntimeError(f"archive receipt mismatch: {path}")
                receipt_count += 1
    return {
        "archive_manifest_path": manifest_path.as_posix(),
        "archive_manifest_sha256": sequential["archive"]["manifest_sha256"],
        "verified_receipt_count": receipt_count,
        "all_passed": True,
    }


def _row(root: Path, expected_replication: int) -> Mapping[str, Any]:
    result_path = root / "result.json"
    hashes = _read(root / "artifact_hashes.json")
    declared = hashes["artifacts"]["result.json"]
    if _sha256(result_path) != declared:
        raise RuntimeError(f"result receipt mismatch: {result_path}")
    result = _read(result_path)
    for name in ("run_manifest.json", "sequential_result.json"):
        if _sha256(root / name) != hashes["artifacts"][name]:
            raise RuntimeError(f"artifact receipt mismatch: {root / name}")
    if int(result.get("replication", -1)) != expected_replication:
        raise RuntimeError(f"replication identity mismatch: {result_path}")
    if result.get("root_seed") != [20260812, 91011 + expected_replication]:
        raise RuntimeError(f"root seed mismatch: {result_path}")
    sequential = result["sequential"]
    retained = sequential["diagnostics"]["retained"]
    analytic = result["retained_analytic_diagnostics"]
    moments = analytic.get("moment_diagnostics", {})
    transitions = result["mode_transition_diagnostics"]
    archive_verification = _verify_archive(result)
    return {
        "replication": expected_replication,
        "root_seed": result["root_seed"],
        "result_path": result_path.as_posix(),
        "result_sha256": declared,
        "archive_verification": archive_verification,
        "passed": result["adjudication"]["status"] == "passed_frozen_seed_replication",
        "hard_vetoes": sequential["diagnostics"]["hard_vetoes"],
        "stop_reason": sequential["stop_reason"],
        "warmup_results_per_chain": sequential["warmup_results_per_chain"],
        "retained_results_per_chain": sequential["retained_results_per_chain"],
        "max_rhat": retained["max_rhat"],
        "min_bulk_ess": retained["min_bulk_ess"],
        "min_tail_ess": retained["min_tail_ess"],
        "minority_mass": analytic["minority_mass"],
        "minority_mass_99pct_interval": analytic["minority_mass_interval"],
        "minority_interval_contains_truth": analytic["gates"]["minority_mass_99pct_interval_contains_truth"],
        "both_modes_overall": analytic["gates"]["both_modes_observed_overall"],
        "both_modes_per_chain": analytic["gates"]["both_hard_modes_observed_per_chain"],
        "per_chain_soft_minority_mass": analytic["per_chain_soft_minority_mass"],
        "transition_count_by_chain": transitions["hard_assignment_transition_count_by_chain"],
        "all_chains_transitioned": transitions["all_chains_transitioned"],
        "mean_interval_pass_count": moments.get("mean_interval_pass_count"),
        "mean_interval_total_count": moments.get("mean_interval_total_count"),
        "covariance_interval_pass_count": moments.get("covariance_interval_pass_count"),
        "covariance_interval_total_count": moments.get("covariance_interval_total_count"),
        "wall_seconds": result["wall_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    rows = [_row(campaign / f"replication-{index}", index) for index in range(4)]
    all_passed = all(bool(row["passed"]) for row in rows)
    all_transitioned = all(bool(row["all_chains_transitioned"]) for row in rows)
    payload = {
        "schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_replication_summary.v1",
        "plan": PLAN.as_posix(),
        "replications": rows,
        "aggregate_descriptive_diagnostics": {
            "minority_mass": _summary_stats([float(row["minority_mass"]) for row in rows]),
            "max_rhat": _summary_stats([float(row["max_rhat"]) for row in rows]),
            "min_bulk_ess": _summary_stats([float(row["min_bulk_ess"]) for row in rows]),
            "min_tail_ess": _summary_stats([float(row["min_tail_ess"]) for row in rows]),
            "wall_seconds": _summary_stats([float(row["wall_seconds"]) for row in rows]),
        },
        "decision_table": {
            "decision": "all_frozen_root_seed_replications_passed" if all_passed else "one_or_more_frozen_root_seed_replications_rejected",
            "primary_criterion_status": "pass_all_four" if all_passed else "fail_not_all_four",
            "veto_diagnostic_status": "pass_no_seed_veto" if all_passed else "fail_seed_level_veto_or_primary_gate",
            "main_uncertainty": "four HMC roots condition on one frozen trained transport and one analytic target",
            "next_justified_action": "replicate on neutrally selected fresh frozen transports" if all_passed else "diagnose failed root without selecting it away",
            "not_concluded": "no equality, stationarity, sampler-ranking, cross-target, SSL-LSTM, or default-readiness claim",
        },
        "inference_status": {
            "hard_veto_screen": "pass" if all_passed else "fail",
            "statistically_supported_ranking": "none_no_method_or_seed_ranking_tested",
            "descriptive_only_differences": "all continuous between-seed differences",
            "default_readiness": "not_assessed",
            "next_evidence_needed": "fresh transport-seed HMC replications selected without posterior peeking",
        },
        "mode_transition_summary": {
            "all_replications_all_chains_transitioned": all_transitioned,
            "role": "explanatory_only_not_a_promotion_gate",
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": "mode-aware starts and one favorable trained transport may hide failures under other frozen transports",
            "overturning_result": "an independently seeded root or neutrally selected transport fails a hard or analytic primary gate",
            "weakest_evidence": "only four roots and one four-dimensional analytic target",
        },
        "git": {
            "commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_dirty": bool(subprocess.run(("git", "status", "--short"), cwd=ROOT, check=True, capture_output=True, text=True).stdout),
        },
    }
    _write(campaign / "summary.json", payload)
    _write(campaign / "artifact_hashes.json", {"schema": "bayesfilter.defensive_weighted_neutra_analytic_hmc_replication_summary_hashes.v1", "artifacts": {"summary.json": _sha256(campaign / "summary.json")}})
    print(json.dumps({"completed": True, "all_passed": all_passed, "campaign_root": campaign.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
