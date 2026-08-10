#!/usr/bin/env python3
"""Compare diagonal, pairwise, and dual-cap GenUT on Austria SIR T20.

This is a finite-program comparison.  No T20 bounded Zhao-Cui teacher is
available, so it deliberately does not claim score accuracy.
"""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base

PLAN = Path("docs/plans/bayesfilter-zhao-cui-genut-austria-t20-dual-cap-plan-2026-08-07.md")
N = 1008
TUNING_SEEDS = (98301, 98302)
CLAIM_SEEDS = tuple(range(98201, 98217))
BASE = {
    "epsilon": 8.0, "sinkhorn_steps": 16, "balance_steps": 16,
    "ridge": 1.0e-5, "higher_moment_correction_steps": 4,
    "higher_moment_strength": 0.2, "higher_moment_floor": 1.0e-5,
    "pairwise_moment_correction_steps": 0, "pairwise_moment_strength": 0.0,
    "pairwise_moment_floor": 1.0e-5, "pairwise_particle_rms_cap": 0.0,
    "coordinatewise_standardized_cap": 0.0,
    "coordinatewise_standardized_cap_power": 8,
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


def _controls() -> dict[str, dict[str, Any]]:
    pair = {**BASE, "pairwise_moment_correction_steps": 4, "pairwise_moment_strength": 0.02}
    return {
        "diagonal_only": dict(BASE),
        "pairwise_only": pair,
        "dual_cap_b090": {**pair, "coordinatewise_standardized_cap": 0.90},
        "dual_cap_b095": {**pair, "coordinatewise_standardized_cap": 0.95},
        "dual_cap_b098": {**pair, "coordinatewise_standardized_cap": 0.98},
        "dual_cap_b098_radial2": {
            **pair, "coordinatewise_standardized_cap": 0.98,
            "pairwise_particle_rms_cap": 2.0,
        },
    }


def _row(evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int) -> dict[str, Any]:
    return base._evaluate(evaluator, target["theta"], tf.cast(observations, tf.float32), seed, target["design"])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", "score_0", "score_1", "score_2")
    vectors = [[r["value"], *r["score"]] for r in rows]
    out = {"count": len(rows), "all_valid": all(base._valid(r) for r in rows)}
    for i, label in enumerate(labels):
        sample = [v[i] for v in vectors]
        sd = statistics.stdev(sample) if len(sample) > 1 else 0.0
        half = 2.131449545559323 * sd / math.sqrt(len(sample)) if len(sample) > 1 else 0.0
        out[label] = {"mean": statistics.mean(sample), "sample_sd": sd,
                      "ci95_lower": statistics.mean(sample) - half,
                      "ci95_upper": statistics.mean(sample) + half}
    out["mean_shape_objective"] = statistics.mean(r["mean_normalized_shape_residual_objective"] for r in rows)
    out["mean_pairwise_shape_objective"] = statistics.mean(r["mean_normalized_pairwise_shape_residual_objective"] for r in rows)
    out["max_coordinate_cap_active_fraction"] = max(r["fraction_coordinatewise_cap_active"] for r in rows)
    out["mean_coordinate_cap_displacement"] = statistics.mean(r["mean_coordinatewise_cap_displacement"] for r in rows)
    out["max_inverse_derivative"] = max(r["maximum_coordinatewise_inverse_derivative"] for r in rows)
    return out


def _arm_valid(rows: list[dict[str, Any]], controls: dict[str, Any]) -> bool:
    if not all(base._valid(row) for row in rows):
        return False
    if controls["coordinatewise_standardized_cap"] > 0.0:
        return all(row["maximum_coordinatewise_post_cap_absolute"] < 1.0 for row in rows)
    return True


def _paired(candidate: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", "score_0", "score_1", "score_2")
    out = {}
    for i, label in enumerate(labels):
        values = [candidate[k]["value"] - baseline[k]["value"] if i == 0 else candidate[k]["score"][i-1] - baseline[k]["score"][i-1] for k in range(len(candidate))]
        out[label] = {"mean_difference": statistics.mean(values),
                      "sample_sd_difference": statistics.stdev(values) if len(values) > 1 else 0.0,
                      "negative_count": sum(v < 0 for v in values),
                      "positive_count": sum(v > 0 for v in values),
                      "zero_count": sum(v == 0 for v in values)}
    return out


def _fd(evaluator: Any, target: dict[str, Any], observations: tf.Tensor, seed: int, h: float = 1.0e-3) -> dict[str, Any]:
    center = _row(evaluator, target, observations, seed)
    residuals = []
    for j in range(3):
        plus_theta = tf.tensor_scatter_nd_add(target["theta"], [[j]], [tf.constant(h, tf.float32)])
        minus_theta = tf.tensor_scatter_nd_add(target["theta"], [[j]], [tf.constant(-h, tf.float32)])
        plus = base._evaluate(evaluator, plus_theta, observations, seed, target["design"])
        minus = base._evaluate(evaluator, minus_theta, observations, seed, target["design"])
        value_fd = (plus["value"] - minus["value"]) / (2.0 * h)
        residuals.append(abs(value_fd - center["score"][j]))
    return {"step": h, "max_absolute_residual": max(residuals), "residuals": residuals,
            "role": "internal same-program finite-difference diagnostic; not an external score authority"}


def run(output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("T20 campaign requires a logical GPU")
    target = base._build_targets()["austria_sir_T20"]
    controls = _controls()
    # Calibration is disjoint from the claim observations and claim seeds.
    calibration = target["calibration"]
    tuning_rows = {}
    for name, cfg in controls.items():
        evaluator = base._make_evaluator(adapter=target["adapter"], horizon=20,
            observation_dim=9, state_dim=18, parameter_dim=3,
            transition_before_first_observation=True, controls=cfg)
        rows = [_row(evaluator, target, data, seed) for data in calibration for seed in TUNING_SEEDS]
        valid = _arm_valid(rows, cfg)
        tuning_rows[name] = {"controls": cfg, "rows": rows,
                             "hard_valid": valid,
                             "summary": _summary(rows) if valid else None}
    claim = {}
    for name, cfg in controls.items():
        evaluator = base._make_evaluator(adapter=target["adapter"], horizon=20,
            observation_dim=9, state_dim=18, parameter_dim=3,
            transition_before_first_observation=True, controls=cfg)
        rows = [_row(evaluator, target, target["observations"], seed) for seed in CLAIM_SEEDS]
        valid = _arm_valid(rows, cfg)
        claim[name] = {"controls": cfg, "rows": rows, "hard_valid": valid,
                       "summary": _summary(rows) if valid else None}
        (output_root / f"checkpoint_{name}.json").write_text(
            json.dumps(_safe(claim[name]), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if not claim["diagonal_only"]["hard_valid"]:
        raise RuntimeError("diagonal-only baseline failed hard validity gates")
    baseline = claim["diagonal_only"]["rows"]
    for name in claim:
        if name != "diagonal_only" and claim[name]["hard_valid"]:
            claim[name]["paired_minus_diagonal"] = _paired(claim[name]["rows"], baseline)
    if claim["pairwise_only"]["hard_valid"]:
        claim["pairwise_only"]["fd_check"] = _fd(
        base._make_evaluator(adapter=target["adapter"], horizon=20, observation_dim=9,
            state_dim=18, parameter_dim=3, transition_before_first_observation=True,
            controls=controls["pairwise_only"]), target, target["observations"], CLAIM_SEEDS[0])
    if claim["dual_cap_b098"]["hard_valid"]:
        claim["dual_cap_b098"]["fd_check"] = _fd(
        base._make_evaluator(adapter=target["adapter"], horizon=20, observation_dim=9,
            state_dim=18, parameter_dim=3, transition_before_first_observation=True,
            controls=controls["dual_cap_b098"]), target, target["observations"], CLAIM_SEEDS[0])
    result_json = output_root / "result.json"
    payload = {
        "schema_version": "bayesfilter.zhao_cui_genut_austria_t20_dual_cap.v1",
        "status": "PASS_T20_FINITE_PROGRAM_COMPARISON",
        "plan": PLAN.as_posix(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "target": {"row_id": "austria_sir_T20", "horizon": 20, "particle_count": N,
                   "state_dimension": 18, "observation_dimension": 9, "parameter_dimension": 3,
                   "event_order": target["event_order"], "source_observation_sha256": target["source_observation_sha256"],
                   "runtime_fp32_observation_sha256": base._tensor_hash(target["observations"], tf.float32),
                   "theta": [float(x) for x in target["theta"].numpy()]},
        "tuning": tuning_rows, "claim": claim,
        "configuration": {"tuning_seeds": TUNING_SEEDS, "claim_seeds": CLAIM_SEEDS,
                           "dtype": "float32", "tf32": True, "jit_compile": True,
                           "score_policy": "recursive forward sensitivity of the same finite program"},
        "device": {"logical_devices": [d.name for d in logical], "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"},
        "memory_policy": dict(memory_policy), "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {"command": [sys.executable, *sys.argv], "environment": sys.prefix,
                         "host": platform.node(), "python": platform.python_version(), "tensorflow": tf.__version__,
                         "plan": PLAN.as_posix(), "output_json": str(result_json.relative_to(ROOT)),
                         "random_seeds": {"tuning": TUNING_SEEDS, "claim": CLAIM_SEEDS},
                         "source_sha256": {PLAN.as_posix(): _sha256(ROOT / PLAN), Path(__file__).relative_to(ROOT).as_posix(): _sha256(Path(__file__))}},
        "nonclaims": ["no valid T20 bounded Zhao-Cui teacher was available", "no score accuracy or bias claim", "no superiority/default/HMC readiness"],
    }
    result_json.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.output_root.resolve())
    print(json.dumps({"status": payload["status"], "output": str(args.output_root.resolve()), "wall_time_seconds": payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
