#!/usr/bin/env python3
"""Compare pairwise GenUT with b=.98 and radial RMS cap 2 across four models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
from docs.benchmarks.genut_fd_regression import (
    FD_REGRESSION_STEPS,
    evaluate_regression_derivative,
    fit_quadratic_step_regression,
)

PLAN = Path("docs/plans/bayesfilter-genut-b098-radial2-four-model-plan-2026-08-07.md")
PRIOR = Path("docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json")
AUSTRIA_T20 = Path("docs/benchmarks/artifacts/zhao_cui_genut_austria_t20_dual_cap_20260807/attempt01/result.json")
MODEL_IDS = ("lgssm_T50", "ksc_sv_T10", "predator_prey_T20", "austria_sir_T20")
N = base.N
CLAIM_SEEDS = tuple(range(98201, 98217))
TUNING_SEEDS = (98401, 98402)
AUSTRIA_TUNING_SEEDS = (98301, 98302)
CAP = 0.98
POWER = 8
RADIAL = 2.0
RESIDUAL_TOL = 5.0e-4
FD_STEPS = FD_REGRESSION_STEPS

# The current checkout regenerates the LGSSM and KSC fixtures with different
# source hashes than the 2026-08-07 warm-start artifact.  These are explicitly
# admitted as a new data scope; the prior controls remain warm starts only.
CURRENT_REGENERATED_SCOPE_HASHES = {
    "lgssm_T50": "8aa2e8102ef25d6accf5d30b9c341621af26fce151ac85133c5a0a6a44671e17",
    "ksc_sv_T10": "b223a99639b95a1955bc13167bff2999bcf626f22df8d880b339420acd4c13e9",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _prior_row(row_id: str) -> dict[str, Any]:
    rows = json.loads((ROOT / PRIOR).read_text(encoding="utf-8"))["rows"]
    matches = [row for row in rows if row["row_id"] == row_id and row["method"] == "genut"]
    if len(matches) != 1:
        raise RuntimeError(f"expected one prior GenUT row for {row_id}")
    return matches[0]


def _baseline(row_id: str, target: dict[str, Any]) -> dict[str, Any]:
    if row_id == "austria_sir_T20":
        payload = json.loads((ROOT / AUSTRIA_T20).read_text(encoding="utf-8"))
        scope = payload["target"]
        if scope["source_observation_sha256"] != target["source_observation_sha256"]:
            raise RuntimeError("Austria T20 baseline observation hash mismatch")
        return dict(payload["claim"]["diagonal_only"]["controls"])
    row = _prior_row(row_id)
    scope = row["scope"]
    if scope["source_observation_sha256"] != target["source_observation_sha256"]:
        expected_current = CURRENT_REGENERATED_SCOPE_HASHES.get(row_id)
        if expected_current != target["source_observation_sha256"]:
            raise RuntimeError(f"{row_id} prior observation hash mismatch")
        # This is a deliberate current-scope warm start, not a scope transfer
        # claim.  Calibration rows below provide the only current-scope checks.
    if scope["event_order"] != target["event_order"] or scope["particle_count"] != N:
        raise RuntimeError(f"{row_id} prior scope mismatch")
    return dict(row["tuning"]["selected_controls"])


def _arm_controls(baseline: dict[str, Any], arm: str, row_id: str) -> dict[str, Any]:
    controls = dict(baseline)
    pair_strength = 0.05 if row_id == "predator_prey_T20" else 0.02
    controls.update({
        "pairwise_moment_correction_steps": 0,
        "pairwise_moment_strength": 0.0,
        "pairwise_moment_floor": 1.0e-5,
        "pairwise_particle_rms_cap": 0.0,
        "coordinatewise_standardized_cap": 0.0,
        "coordinatewise_standardized_cap_power": POWER,
    })
    if arm in ("pairwise", "coordinate_cap", "dual_cap"):
        controls["pairwise_moment_correction_steps"] = 4
        controls["pairwise_moment_strength"] = pair_strength
    if arm in ("coordinate_cap", "dual_cap"):
        controls["coordinatewise_standardized_cap"] = CAP
    if arm == "dual_cap":
        controls["pairwise_particle_rms_cap"] = RADIAL
    return controls


def _evaluator(target: dict[str, Any], controls: dict[str, Any]):
    return base._make_evaluator(
        adapter=target["adapter"], horizon=int(target["observations"].shape[0]),
        observation_dim=target["observation_dim"], state_dim=target["state_dim"],
        parameter_dim=target["parameter_dim"],
        transition_before_first_observation=target["transition_before"], controls=controls,
    )


def _row(evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int) -> dict[str, Any]:
    return base._evaluate(evaluator, target["theta"], tf.cast(observations, tf.float32), seed, target["design"])


def _valid(row: dict[str, Any], controls: dict[str, Any]) -> bool:
    if not base._valid(row):
        return False
    if max(row["max_mean_residual"], row["max_row_residual"], row["max_col_residual"], row["score_increment_sum_residual"]) > RESIDUAL_TOL:
        return False
    if controls["coordinatewise_standardized_cap"] > 0.0 and row["maximum_coordinatewise_post_cap_absolute"] >= 1.000001:
        return False
    return True


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{i}" for i in range(len(rows[0]["score"]))])
    vectors = [[row["value"], *row["score"]] for row in rows]
    out: dict[str, Any] = {"count": len(rows), "all_finite": all(row["finite"] for row in rows)}
    for index, label in enumerate(labels):
        values = [vector[index] for vector in vectors]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 2.131449545559323 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
        out[label] = {"mean": mean, "sample_sd": sd, "mcse": sd / math.sqrt(len(values)) if values else None,
                      "ci95_lower": mean - half, "ci95_upper": mean + half}
    out["max_cap_active_fraction"] = max(row["fraction_coordinatewise_cap_active"] for row in rows)
    out["mean_cap_displacement"] = statistics.mean(row["mean_coordinatewise_cap_displacement"] for row in rows)
    out["max_pre_cap_absolute"] = max(row["maximum_coordinatewise_pre_cap_absolute"] for row in rows)
    out["max_post_cap_absolute"] = max(row["maximum_coordinatewise_post_cap_absolute"] for row in rows)
    out["max_pairwise_pre_cap_rms"] = max(row["maximum_pairwise_pre_cap_particle_rms"] for row in rows)
    out["min_pairwise_cap_scale"] = min(row["minimum_pairwise_particle_cap_scale"] for row in rows)
    return out


def _paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{i}" for i in range(len(left[0]["score"]))])
    result = {}
    for index, label in enumerate(labels):
        diffs = [[row["value"], *row["score"]][index] - [base_row["value"], *base_row["score"]][index] for base_row, row in zip(left, right)]
        result[label] = {"mean_candidate_minus_baseline": statistics.mean(diffs),
                         "sample_sd": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
                         "negative_count": sum(x < 0 for x in diffs), "positive_count": sum(x > 0 for x in diffs)}
    return result


def _fd(evaluator: Any, target: dict[str, Any], seed: int) -> dict[str, Any]:
    initial, process = base._noise(seed, int(target["observations"].shape[0]), target["state_dim"])
    theta = target["theta"]
    value, score, _ = evaluator(theta, target["observations"], initial, process, target["design"])
    rows = []
    for index in range(target["parameter_dim"]):
        direction = tf.one_hot(index, target["parameter_dim"], dtype=tf.float32)
        finite_differences = []
        endpoint_valid = True
        for step in FD_STEPS:
            plus_value, _, plus_status = evaluator(
                theta + step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )
            minus_value, _, minus_status = evaluator(
                theta - step * direction,
                target["observations"],
                initial,
                process,
                target["design"],
            )
            plus_valid = bool(plus_status["program_valid"].numpy())
            minus_valid = bool(minus_status["program_valid"].numpy())
            endpoint_valid = endpoint_valid and plus_valid and minus_valid
            finite_differences.append(
                float(((plus_value - minus_value) / (2.0 * step)).numpy())
            )
        regression = (
            fit_quadratic_step_regression(FD_STEPS, finite_differences)
            if endpoint_valid and all(math.isfinite(value) for value in finite_differences)
            else None
        )
        diagnostic = (
            evaluate_regression_derivative(float(score[index].numpy()), regression)
            if regression is not None
            else {"diagnostic_pass": False, "reason": "invalid_fd_endpoint"}
        )
        rows.append({
            "parameter": index,
            "manual_score": float(score[index].numpy()),
            "finite_difference_ladder": finite_differences,
            "endpoint_valid": endpoint_valid,
            "regression": regression,
            "regression_diagnostic": diagnostic,
        })
    return {
        "steps": list(FD_STEPS),
        "rows": rows,
        "all_pass": all(row["regression_diagnostic"]["diagnostic_pass"] for row in rows),
        "role": "internal same-program regression diagnostic, not an external score authority",
    }


def _references(row_id: str, target: dict[str, Any]) -> list[dict[str, Any]]:
    if row_id == "lgssm_T50":
        from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score
        value, score = _kalman_value_score(target["theta"], target["observations"])
        return [{"method": "Kalman", "value": float(value.numpy()), "score": [float(x) for x in score.numpy()], "role": "exact affine oracle"}]
    if row_id == "ksc_sv_T10":
        ref = base._prior_comparators(base._build_targets())
        return [{"method": x["method"], "value": x.get("value"), "score": x.get("score"), "role": "same-target diagnostic"} for x in ref[row_id]]
    if row_id == "predator_prey_T20":
        ref = base._prior_comparators(base._build_targets())[row_id]
        return [{"method": x["method"], "value": x.get("value"), "score": x.get("score"), "role": "same-target diagnostic"} for x in ref]
    payload = json.loads((ROOT / "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02/result.json").read_text())
    ukf = json.loads((ROOT / "docs/plans/artifacts/bayesfilter-neutra-remaining-models-20260730/sir-ukf-parity-gpu-attempt-02/result.json").read_text())
    return [{"method": "SGQF", "value": payload["compiled_value"][0], "score": payload["compiled_score"][0], "role": "same-target approximate Gaussian closure"},
            {"method": "UKF", "value": ukf["value"][0], "score": ukf["score"][0], "role": "same-target approximate Gaussian closure"}]


def _render(payload: dict[str, Any]) -> str:
    lines = ["# GenUT b=.98 + Radial-2 Four-Model Result", "", "| Model | Arm | Valid | Value mean (SD) | Score means | Score SDs |", "|---|---|---|---:|---|---|"]
    for model in payload["models"]:
        for arm_id, arm in model["claim"].items():
            summary = arm["summary"]
            means = [summary[f"score_{i}"]["mean"] for i in range(model["scope"]["parameter_dimension"])]
            sds = [summary[f"score_{i}"]["sample_sd"] for i in range(model["scope"]["parameter_dimension"])]
            lines.append(f"| {model['row_id']} | {arm_id} | {arm['hard_valid']} | {summary['value']['mean']:.8g} ({summary['value']['sample_sd']:.5g}) | `{means}` | `{sds}` |")
    lines += ["", "All differences are descriptive. Approximate comparators are not nonlinear truth oracles.", ""]
    return "\n".join(lines)


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("four-model campaign requires a logical GPU")
    # Fixture generation performs setup-only Cholesky/SVD work.  Keep it on
    # CPU after GPU memory policy verification so target construction cannot
    # consume or initialize the GPU solver path; claim evaluators remain
    # explicitly pinned to GPU:0 in ``_evaluator``.
    with tf.device("/CPU:0"):
        targets = base._build_targets()
    models = []
    for model_index, row_id in enumerate(MODEL_IDS):
        target = targets[row_id]
        baseline = _baseline(row_id, target)
        seeds = CLAIM_SEEDS
        tuning_seeds = AUSTRIA_TUNING_SEEDS if row_id == "austria_sir_T20" else TUNING_SEEDS
        claim = {}
        for arm_id in ("diagonal", "pairwise", "coordinate_cap", "dual_cap"):
            controls = _arm_controls(baseline, arm_id, row_id)
            evaluator = _evaluator(target, controls)
            calibration_rows = [_row(evaluator, target, data, seed) for data in target["calibration"] for seed in tuning_seeds]
            calibration_valid = all(_valid(row, controls) for row in calibration_rows)
            claim_rows = [_row(evaluator, target, target["observations"], seed) for seed in seeds]
            claim_valid = all(_valid(row, controls) for row in claim_rows)
            hard_valid = calibration_valid and claim_valid
            claim[arm_id] = {"controls": controls, "calibration_valid": calibration_valid, "claim_valid": claim_valid,
                             "hard_valid": hard_valid, "calibration_rows": calibration_rows, "rows": claim_rows,
                             "summary": _summary(claim_rows),
                             "fd": _fd(evaluator, target, CLAIM_SEEDS[0]) if claim_valid else None}
            (output_root / f"checkpoint_{row_id}_{arm_id}.json").write_text(json.dumps(_safe(claim[arm_id]), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not claim["diagonal"]["hard_valid"]:
            raise RuntimeError(f"{row_id} diagonal baseline failed hard validity")
        for arm_id in ("pairwise", "coordinate_cap", "dual_cap"):
            if claim[arm_id]["claim_valid"]:
                claim[arm_id]["paired_vs_diagonal"] = _paired(claim["diagonal"]["rows"], claim[arm_id]["rows"])
        models.append({"row_id": row_id, "scope": {"row_id": row_id, "horizon": int(target["observations"].shape[0]), "state_dimension": target["state_dim"], "observation_dimension": target["observation_dim"], "parameter_dimension": target["parameter_dim"], "particle_count": N, "event_order": target["event_order"], "source_observation_sha256": target["source_observation_sha256"], "runtime_fp32_observation_sha256": base._tensor_hash(target["observations"], tf.float32)}, "claim": claim, "references": _references(row_id, target), "classification": "extension_or_invention_for_caps"})
    result_json = output_root / "result.json"
    payload = {"schema_version": "bayesfilter.genut_b098_radial2_four_model.v1", "status": "COMPLETE_FOUR_MODEL_COMPARISON", "plan": PLAN.as_posix(), "models": models,
               "configuration": {"particle_count": N, "claim_seeds": CLAIM_SEEDS, "tuning_seeds": TUNING_SEEDS, "austria_tuning_seeds": AUSTRIA_TUNING_SEEDS, "coordinatewise_standardized_cap": CAP, "coordinatewise_standardized_cap_power": POWER, "pairwise_particle_rms_cap": RADIAL, "dtype": "float32", "tf32": True, "jit_compile": True},
               "device": {"logical_devices": [d.name for d in logical], "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"}, "memory_policy": dict(memory_policy), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "wall_time_seconds": time.perf_counter() - started,
               "run_manifest": {"command": [sys.executable, *sys.argv], "environment": sys.prefix, "host": platform.node(), "python": platform.python_version(), "tensorflow": tf.__version__, "plan": PLAN.as_posix(), "output_json": str(result_json.relative_to(ROOT)), "source_sha256": {PLAN.as_posix(): _sha256(ROOT / PLAN), Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__))}},
               "nonclaims": ["no exact nonlinear score or likelihood claim", "no lower-bias claim from comparator proximity", "no statistically supported ranking", "no Zhao-Cui source-faithfulness claim", "no default/HMC/NeuTra readiness"]}
    result_json.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "result.md").write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output_root.resolve())
    print(json.dumps({"status": payload["status"], "output": str(args.output_root.resolve()), "wall_time_seconds": payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
