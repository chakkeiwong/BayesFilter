#!/usr/bin/env python3
"""Fail-closed aggregate for the four-model GenUT NeuTra readiness campaign."""

from __future__ import annotations

import json
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT / "docs/benchmarks/artifacts/genut_four_model_neutra_readiness_20260804"
PLAN = "docs/plans/bayesfilter-genut-four-model-neutra-readiness-plan-2026-08-04.md"
RESULT = "docs/plans/bayesfilter-genut-four-model-neutra-readiness-result-2026-08-04.md"

FINAL = {
    "lgssm": {
        "tuning": "lgssm_no_tf32_tuning_attempt01/result.json",
        "claims": (
            "lgssm_no_tf32_final_claim_attempt01/result.json",
            "lgssm_no_tf32_final_claim_attempt02/result.json",
        ),
        "capacity": "lgssm_no_tf32_final_capacity_b4_attempt01/result.json",
        "parity": "lgssm_no_tf32_scalar_parity_attempt02/result.json",
        "kalman": "lgssm_no_tf32_kalman_attempt02/result.json",
        "training": "lgssm_no_tf32_final_training_attempt01/result.json",
        "tf32_enabled": False,
    },
    "ksc_sv": {
        "tuning": "ksc_tuning_attempt02/result.json",
        "claims": (
            "ksc_final_claim_attempt02/result.json",
            "ksc_final_claim_attempt03/result.json",
        ),
        "capacity": "ksc_final_capacity_b4_attempt01/result.json",
        "parity": "ksc_scalar_parity_attempt01/result.json",
        "training": "ksc_final_training_attempt02/result.json",
        "tf32_enabled": True,
    },
    "predator_prey": {
        "tuning": "predator_prey_tuning_attempt02/result.json",
        "claims": (
            "predator_prey_final_claim_attempt02/result.json",
            "predator_prey_final_claim_attempt03/result.json",
        ),
        "capacity": "predator_prey_final_capacity_b4_attempt01/result.json",
        "parity": "predator_prey_scalar_parity_attempt01/result.json",
        "training": "predator_prey_final_training_attempt02/result.json",
        "tf32_enabled": True,
    },
}
AUSTRIA = "austria_final_claim_attempt01/result.json"


def _load(relative: str) -> dict:
    path = ARTIFACT_ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(relative: str) -> dict:
    return _load(str(Path(relative).with_name("run_manifest.json")))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _cross_process_replay(first: dict, second: dict) -> dict:
    _require(first["target_signature"] == second["target_signature"], "claim signatures differ")
    value_error = max(
        _relative_error(float(left), float(right))
        for left, right in zip(first["value"], second["value"])
    )
    score_error = max(
        _relative_error(float(left), float(right))
        for left_row, right_row in zip(first["score"], second["score"])
        for left, right in zip(left_row, right_row)
    )
    same_validity = (
        first["status"]["valid_pre_regularized_score"]
        == second["status"]["valid_pre_regularized_score"]
    )
    passed = value_error <= 2.0e-6 and score_error <= 2.0e-5 and same_validity
    return {
        "passed": passed,
        "value_max_relative_error": value_error,
        "score_max_relative_error": score_error,
        "same_validity_decision": same_validity,
        "thresholds": {"value": 2.0e-6, "score": 2.0e-5},
    }


def _model_result(model: str, specification: dict) -> dict:
    tuning = _load(specification["tuning"])
    first, second = (_load(path) for path in specification["claims"])
    capacity = _load(specification["capacity"])
    parity = _load(specification["parity"])
    training = _load(specification["training"])
    signature = first["target_signature"]
    paths = tuple(specification["claims"]) + (
        specification["capacity"],
        specification["parity"],
        specification["training"],
    )
    payloads = (first, second, capacity, parity, training)
    manifests = tuple(_manifest(path) for path in paths)
    tuning_manifest = _manifest(specification["tuning"])
    _require(tuning["passed"], f"{model}: tuning failed")
    _require(tuning["deterministic_ops_enabled"], f"{model}: tuning is not deterministic")
    _require(
        tuning["tf32_enabled"] is specification["tf32_enabled"],
        f"{model}: tuning arithmetic mismatch",
    )
    _require(tuning_manifest.get("memory_policy", {}).get("all_physical_devices_memory_growth") is True, f"{model}: tuning manifest memory growth missing")
    _require(tuning_manifest.get("deterministic_ops_enabled") is True, f"{model}: tuning manifest determinism missing")
    _require(tuning_manifest.get("tf32_enabled") is specification["tf32_enabled"], f"{model}: tuning manifest arithmetic mismatch")
    _require(all(item["target_signature"] == signature for item in payloads), f"{model}: stale target identity")
    _require(all(item.get("control_status") == "repository_tuning_artifact_bound" for item in payloads), f"{model}: tuning artifact is not repository-bound")
    _require(all(item.get("gpu_allocator") is not None for item in payloads), f"{model}: allocator telemetry missing")
    _require(all(item.get("memory_policy", {}).get("all_physical_devices_memory_growth") is True for item in manifests), f"{model}: manifest memory growth missing")
    _require(all(item.get("deterministic_ops_enabled") is True for item in manifests), f"{model}: manifest determinism missing")
    _require(all(item.get("target_signature") == signature for item in manifests), f"{model}: manifest target identity mismatch")
    _require(all(item.get("tf32_enabled") is specification["tf32_enabled"] for item in manifests), f"{model}: manifest arithmetic mismatch")
    _require(all(item.get("tf32_enabled") is specification["tf32_enabled"] for item in (first, second, capacity, training)), f"{model}: claim arithmetic mismatch")
    _require(first["passed_capacity_replay_endpoint_gate"] and second["passed_capacity_replay_endpoint_gate"], f"{model}: claim failed")
    _require(capacity["passed_capacity_replay_endpoint_gate"] and capacity["batch_size"] == 4, f"{model}: B=4 failed")
    _require(parity["passed"], f"{model}: real-scope scalar parity failed")
    _require(parity.get("tf32_enabled") is specification["tf32_enabled"], f"{model}: parity result arithmetic mismatch")
    _require(training["passed"] and training["completed_steps"] == 1, f"{model}: optimizer update failed")
    _require(training["record"]["target_condition_estimate_available"] is False, f"{model}: condition availability is wrong")
    _require(training["record"]["target_min_innovation_eigenvalue_available"] is False, f"{model}: eigenvalue availability is wrong")
    binding = training["runtime_metadata"]["batch_native_target"]
    _require(not binding["scalar_fallback_used"], f"{model}: scalar fallback used")
    _require(not binding["sample_axis_python_loop_used"], f"{model}: sample loop used")
    _require(not binding["row_mapped_scalar_target_used"], f"{model}: row-mapped scalar target used")
    replay = _cross_process_replay(first, second)
    _require(replay["passed"], f"{model}: cross-process replay failed")
    finite_differences = tuple(item["finite_difference"] for item in (first, second))
    _require(all(item["stencil_valid"] and item["center_valid"] for item in finite_differences), f"{model}: FD validity changed")
    _require(all(float(item["maximum_relative_error"]) <= 0.05 for item in finite_differences), f"{model}: FD failed")
    controls = first["controls"]
    _require(controls["tuning_artifact"].endswith(specification["tuning"]), f"{model}: wrong tuning path")
    result = {
        "model": model,
        "ready_for_target_specific_serious_neutra_training": True,
        "ready_for_hmc": False,
        "target_signature": signature,
        "tf32_enabled": specification["tf32_enabled"],
        "particle_count": first["particle_count"],
        "controls": controls,
        "tuning_artifact": specification["tuning"],
        "claim_artifacts": list(specification["claims"]),
        "capacity_artifact": specification["capacity"],
        "scalar_parity_artifact": specification["parity"],
        "training_artifact": specification["training"],
        "finite_difference_steps": [item["step"] for item in finite_differences],
        "finite_difference_max_relative_errors": [item["maximum_relative_error"] for item in finite_differences],
        "endpoint_value_max_relative_error": max(first["value_endpoint_max_relative_error"], second["value_endpoint_max_relative_error"]),
        "scalar_parity_value_relative_error": parity["posterior_value_relative_error"],
        "scalar_parity_score_max_relative_error": parity["posterior_score_max_relative_error"],
        "cross_process_replay": replay,
        "b4_peak_allocator_bytes": capacity["gpu_allocator"]["peak_bytes"],
        "one_step_training_peak_allocator_bytes": training["gpu_allocator"]["peak_bytes"],
        "evidence_paths": list(paths),
    }
    if model == "lgssm":
        kalman = _load(specification["kalman"])
        kalman_manifest = _manifest(specification["kalman"])
        _require(kalman["passed"] and kalman["target_signature"] == signature, "lgssm: Kalman gate failed")
        _require(kalman.get("tf32_enabled") is False, "lgssm: Kalman result arithmetic mismatch")
        _require(kalman_manifest.get("target_signature") == signature, "lgssm: Kalman manifest identity mismatch")
        _require(kalman_manifest.get("tf32_enabled") is False, "lgssm: Kalman manifest arithmetic mismatch")
        _require(kalman_manifest.get("memory_policy", {}).get("all_physical_devices_memory_growth") is True, "lgssm: Kalman manifest memory growth missing")
        result["kalman_artifact"] = specification["kalman"]
        result["kalman_gross_error"] = {
            "posterior_value_error": kalman["posterior_value_error"],
            "posterior_score_max_absolute_error": kalman["posterior_score_max_absolute_error"],
            "posterior_score_direction_cosine": kalman["posterior_score_direction_cosine"],
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if output_root.exists():
        raise RuntimeError("aggregate output root must be fresh")
    output_root.mkdir(parents=True)
    output = output_root / "result.json"
    manifest = output_root / "run_manifest.json"
    if output.exists() or manifest.exists():
        raise RuntimeError("aggregate outputs must be fresh")
    started = time.monotonic()
    models = {model: _model_result(model, specification) for model, specification in FINAL.items()}
    austria = _load(AUSTRIA)
    _require(not austria["passed_capacity_replay_endpoint_gate"], "Austria unexpectedly passed")
    _require(austria["value_endpoint_max_relative_error"] > 2.0e-4, "Austria veto reason changed")
    models["austria_sir"] = {
        "model": "austria_sir",
        "ready_for_target_specific_serious_neutra_training": False,
        "ready_for_hmc": False,
        "target_signature": austria["target_signature"],
        "tf32_enabled": austria["tf32_enabled"],
        "blocking_gate": "value_score_route_and_tangent_free_endpoint_are_not_the_same_finite_program",
        "endpoint_value_max_relative_error": austria["value_endpoint_max_relative_error"],
        "threshold": 2.0e-4,
        "claim_artifact": AUSTRIA,
    }
    result = {
        "schema": "bayesfilter.genut_four_model_neutra_readiness_aggregate.v1",
        "passed": True,
        "campaign_question_answered": True,
        "models": models,
        "serious_neutra_training_eligible_models": ["lgssm", "ksc_sv", "predator_prey"],
        "blocked_models": ["austria_sir"],
        "hmc_ready_models": [],
        "default_ready_models": [],
        "statistical_ranking_supported": False,
        "decision": "proceed_to_target_specific_serious_neutra_training_for_three_models_only",
        "wall_time_seconds": time.monotonic() - started,
        "plan": PLAN,
        "result_note": RESULT,
        "nonclaims": [
            "no serious NeuTra training has been run",
            "no learned transport has been selected",
            "no HMC chain, convergence, posterior correctness, or default readiness claim",
            "no stochastic method ranking",
        ],
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commit = subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    manifest.write_text(
        json.dumps(
            {
                "schema": "bayesfilter.genut_four_model_neutra_readiness_aggregate_manifest.v1",
                "git_commit": commit,
                "command": list(sys.argv),
                "environment": "standard_library_artifact_aggregation",
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "gpu_status": "no GPU computation; aggregation of trusted GPU artifacts",
                "seeds": {"target": 140000, "tuning_calibration": 141001, "tuning_validation": 141002, "training": [20260804, 701]},
                "wall_time_seconds": time.monotonic() - started,
                "output_root": str(output_root.relative_to(ROOT)),
                "plan": PLAN,
                "result": str(output.relative_to(ROOT)),
                "result_note": RESULT,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": True, "result": str(output.relative_to(ROOT))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
