#!/usr/bin/env python3
"""Aggregate repeated TF32 versus no-TF32 warm execution timings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/bayesfilter-zhao-cui-moment-teacher-tf32-performance-decision-2026-07-30.md"
SCHEMA = "bayesfilter.zhao_cui_moment_teacher_tf32_performance.v1"
TIME_REDUCTION_THRESHOLD = 0.20


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timing_summary(payload: dict[str, object]) -> dict[str, object]:
    result = payload["result"]
    if int(result["batch_count"]) != 1:
        raise ValueError("performance node must contain exactly one batch")
    times = [
        float(value)
        for value in result["batches"][0]["timing_seconds"][
            "warm_execution_repetitions"
        ]
    ]
    return {
        "times_seconds": times,
        "repetitions": len(times),
        "median_seconds": statistics.median(times),
        "mean_seconds": statistics.fmean(times),
        "minimum_seconds": min(times),
        "maximum_seconds": max(times),
        "gpu_allocator_peak_bytes": int(result["gpu_allocator_bytes"]["peak"]),
    }


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
    identity_checks = {
        "campaign": tf32_payload["campaign_id"] == reference_payload["campaign_id"],
        "commit": tf32_payload["git_commit"] == reference_payload["git_commit"],
        "source_sha256": tf32_payload["source_sha256"] == reference_payload["source_sha256"],
        "prepared_inputs": tf32["preparation_identity"] == reference["preparation_identity"],
        "seeds": tf32["estimator_seeds"] == reference["estimator_seeds"],
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
        "tf32_modes": (
            tf32["device"]["tf32_enabled"] is True
            and reference["device"]["tf32_enabled"] is False
        ),
        "both_fp32": (
            tf32["device"]["dtype"] == "float32"
            and reference["device"]["dtype"] == "float32"
        ),
    }
    validity_checks = {
        "tf32_hard_valid": bool(tf32["hard_valid"]),
        "reference_hard_valid": bool(reference["hard_valid"]),
    }
    tf32_timing = _timing_summary(tf32_payload)
    reference_timing = _timing_summary(reference_payload)
    repetition_valid = (
        tf32_timing["repetitions"] >= 5
        and reference_timing["repetitions"] >= 5
    )
    interpretable = (
        all(identity_checks.values())
        and all(validity_checks.values())
        and repetition_valid
    )
    tf32_median = float(tf32_timing["median_seconds"])
    reference_median = float(reference_timing["median_seconds"])
    time_reduction = 1.0 - tf32_median / reference_median
    throughput_gain = reference_median / tf32_median - 1.0
    passed = interpretable and time_reduction >= TIME_REDUCTION_THRESHOLD
    payload = {
        "schema": SCHEMA,
        "question": "Is TF32 at least 20% faster in median warm execution time?",
        "identity_checks": identity_checks,
        "validity_checks": validity_checks,
        "repetition_valid": repetition_valid,
        "interpretable": interpretable,
        "tf32": tf32_timing,
        "fp32_no_tf32": reference_timing,
        "median_time_reduction": time_reduction,
        "median_throughput_gain": throughput_gain,
        "criterion": {
            "minimum_time_reduction": TIME_REDUCTION_THRESHOLD,
            "passed": passed,
        },
        "decision": (
            "speed_condition_met"
            if passed
            else (
                "speed_condition_not_met"
                if interpretable
                else "not_interpretable"
            )
        ),
        "hmc_contract": {
            "current_wrapper_uses_one_log_prob_and_grad_call_for_value_and_score": True,
            "mh_independently_recomputes_higher_precision_acceptance_energy": False,
            "interpretation": (
                "MH corrects proposal integration error relative to the finite target "
                "value returned by the adapter; it does not replace that value with "
                "the FP32-no-TF32 comparator."
            ),
        },
        "inputs": {
            "tf32_result": str(args.tf32_result),
            "tf32_sha256": _sha256(args.tf32_result.resolve()),
            "reference_result": str(args.reference_result),
            "reference_sha256": _sha256(args.reference_result.resolve()),
        },
        "nonclaims": [
            "no moment-teacher HMC runtime was benchmarked",
            "no posterior-correctness or HMC-readiness conclusion",
            "the 20% engineering threshold is not a scientific constant",
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
        "environment": "CPU standard-library aggregation of trusted GPU timings",
        "device": "CPU aggregation; GPU provenance is in both input manifests",
        "data_version": "paired prepared-input identity embedded in node results",
        "seeds": reference["estimator_seeds"],
        "wall_time_seconds": "N/A_negligible_offline_aggregation",
        "output_artifact": str(output.relative_to(ROOT)),
        "plan": PLAN,
        "result": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
    }
    _write_json(output / "run_manifest.json", manifest)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "median_time_reduction": time_reduction,
                "median_throughput_gain": throughput_gain,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

