#!/usr/bin/env python3
"""Aggregate the Contract E--TP Phase 8 one-factor refinement artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / (
    "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15"
)
INPUT_ROOT = ARTIFACT_ROOT / "phase8_refinement_20260715"
PLAN = "docs/plans/bayesfilter-contract-e-tp-phase8-one-factor-refinement-plan-2026-07-15.md"
REFERENCE_RESULTS = {
    ("actual_sv", 1): "phase7_actual_sv_t1_bound_target_result_20260715.json",
    ("actual_sv", 2): "phase7_actual_sv_t2_order41_lookahead1_result_20260715.json",
    ("ksc_sv", 1): "phase7_ksc_sv_t1_order41_current_target_result_20260715.json",
    ("ksc_sv", 2): "phase7_ksc_sv_t2_order41_lookahead1_result_20260715.json",
    ("generalized_sv", 1): "phase5_generalized_sv_t1_order25_timeorderfix_result_20260715.json",
}
ROW_PREFIX = {
    "actual_sv": "actual_sv_",
    "ksc_sv": "ksc_sv_",
    "generalized_sv": "generalized_sv_",
}


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _row_name(path: Path) -> str:
    matches = [row for row, prefix in ROW_PREFIX.items() if path.name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"cannot identify row from {path.name}")
    return matches[0]


def _reference(row: str, horizon: int) -> tuple[dict[str, Any], Path]:
    path = ARTIFACT_ROOT / REFERENCE_RESULTS[(row, horizon)]
    payload = _load(path)
    return payload["dense_references"][-1], path


def _arm(path: Path) -> dict[str, Any]:
    payload = _load(path)
    row = _row_name(path)
    horizon = int(payload["horizon"])
    reference, reference_path = _reference(row, horizon)
    value = float(payload["value"])
    score = [float(item) for item in payload["score"]]
    reference_value = float(reference["value"])
    reference_score = [float(item) for item in reference["score"]]
    if not math.isfinite(value) or not _finite(score):
        raise ValueError(f"nonfinite arm: {path}")
    score_difference = [left - right for left, right in zip(score, reference_score)]
    relative = [
        abs(delta) / max(abs(left), abs(right), 1.0e-12)
        for left, right, delta in zip(score, reference_score, score_difference)
    ]
    fit_steps = payload["finite_program"]["steps"]
    hard = payload["hard_vetoes"]
    preparation = payload["target"].get("preparation")
    if preparation is None:
        raise ValueError(f"Phase 8 arm does not bind a target preparation: {path}")
    preparation_path = ROOT / preparation["path"]
    if _sha256(preparation_path) != preparation["sha256"]:
        raise ValueError(f"preparation hash mismatch: {path}")
    engineering_pass = bool(
        payload["status"] == "certified_extension_or_invention"
        and payload["own_scalar_fd"]["status"] == "pass"
        and hard["first_step_time_order_valid"]
        and hard["carried_marginal_mass_valid"]
        and hard["finite_value_and_score"]
    )
    if not engineering_pass:
        raise ValueError(f"Phase 8 engineering veto failed: {path}")
    return {
        "row": row,
        "horizon": horizon,
        "artifact": {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)},
        "reference": {
            "path": str(reference_path.relative_to(ROOT)),
            "sha256": _sha256(reference_path),
            "value": reference_value,
            "score": reference_score,
        },
        "target_observations_sha256": payload["target"]["target_observations_sha256"],
        "config": payload["config"],
        "value": value,
        "score": score,
        "value_difference_to_reference": value - reference_value,
        "score_difference_to_reference": score_difference,
        "componentwise_relative_error": relative,
        "maximum_componentwise_relative_error": max(relative),
        "sign_reversal": [
            (left < 0.0 < right) or (right < 0.0 < left)
            for left, right in zip(score, reference_score)
        ],
        "maximum_fit_residual": max(float(step["fit_residual"]) for step in fit_steps),
        "maximum_scaled_condition": max(
            max(float(item) for item in step["fit_condition_numbers"])
            for step in fit_steps
        ),
        "own_scalar_fd_status": "pass",
        "engineering_status": "pass",
        "comparison_classification": "descriptive_only_margin_unavailable",
    }


def build() -> dict[str, Any]:
    paths = sorted(INPUT_ROOT.glob("*_result.json"))
    if len(paths) != 25:
        raise ValueError(f"expected 25 Phase 8 arms, found {len(paths)}")
    arms = [_arm(path) for path in paths]
    return {
        "schema": "bayesfilter.contract_e_tp.phase8_refinement.v1",
        "metadata_date": "2026-07-15",
        "status": "PHASE8_COMPLETE_PARTIAL_REPAIR_EXTENSION_NOT_PROMOTED",
        "plan": PLAN,
        "arms": arms,
        "arm_count": len(arms),
        "decisions": {
            "actual_sv": {
                "nomination": "degree12_order17_rank2_width8",
                "t2_status": "partial_repair_not_equivalence",
                "t2_worst_score_error_before": 0.1369785337669442,
                "t2_worst_score_error_after": 0.021629875635920497,
            },
            "ksc_sv": {
                "nomination": "degree8_order17_rank2_width6",
                "t2_status": "partial_repair_not_equivalence",
                "t2_worst_score_error_before": 0.3753265535876129,
                "t2_worst_score_error_after": 0.13400159211847323,
            },
            "generalized_sv": {
                "nomination": None,
                "status": "rank_nonmonotone_score_negative_result",
                "reason": "rank6_and_rank8_improve_value_and_fit_but_overshoot_gamma_score",
            },
            "quadrature": "not_limiting_at_tested_t1_rungs",
            "fit_residual": "explanatory_only_not_selection_metric",
        },
        "inference_status": {
            "hard_veto_screen": "all_25_arms_pass",
            "statistically_supported_ranking": False,
            "descriptive_only_differences": True,
            "default_readiness": False,
            "next_evidence": "gpu_xla_scaling_for_contract_e_tp_correctness_eligible_rows",
        },
        "nonclaims": [
            "no equivalence or superiority",
            "no extension promotion",
            "no Zhao-Cui parameter-learning comparator",
            "no default, HMC, leaderboard, or GPU readiness",
        ],
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "environment": "artifact_only_standard_library",
            "cpu_gpu_status": "no numerical framework initialized",
        },
    }


def main() -> None:
    args = _parse()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    payload = build()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["run_manifest"]["output"] = str(output.relative_to(ROOT))
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "arm_count": payload["arm_count"]}))


if __name__ == "__main__":
    main()
