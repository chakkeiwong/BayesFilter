#!/usr/bin/env python3
"""Bounded code/math diagnostics for the Austria-SIR GenUT j0 discrepancy."""

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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.cubature_genut_adapters import parameterized_austria_sir_candidate_adapter
from bayesfilter.highdim.cubature_genut_filter import finite_value_score
from bayesfilter.highdim.ledh_contract_e_reset_tf import contract_e_chol_cloud_forward_tf, contract_e_chol_cloud_jvp_tf
from bayesfilter.independent_score import sir_observation_simulator_tf as sir
from docs.benchmarks import run_moment_retuned_genut_whole_leaderboard as base
from docs.benchmarks.genut_fd_regression import fit_quadratic_step_regression, evaluate_regression_derivative

PLAN = ROOT / "docs/plans/bayesfilter-genut-sir-root-cause-hypothesis-plan-2026-08-17.md"
N = 1008
SEEDS = (98201, 98202, 98203)
HORIZONS = (2, 5, 20)
FD_STEPS_LOCAL = (3.0e-3, 1.0e-3, 3.0e-4)
EXPECTED_OBS_HASH = "cd794ad6e90a74f7cf6dc06b33550bff4bef6fbf66bb0917846d0691b5910f07"
_EVALUATOR_CACHE = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def safe(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def controls(kind: str) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "epsilon": 8.0,
        "sinkhorn_steps": 16,
        "balance_steps": 16,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": 0,
        "higher_moment_strength": 0.0,
        "higher_moment_floor": 1.0e-5,
        "pairwise_moment_correction_steps": 0,
        "pairwise_moment_strength": 0.0,
        "pairwise_moment_floor": 1.0e-5,
        "pairwise_particle_rms_cap": 0.0,
        "coordinatewise_standardized_cap": 0.0,
        "coordinatewise_standardized_cap_power": 8,
    }
    if kind in ("diagonal", "pairwise", "dual_cap"):
        result.update({"higher_moment_correction_steps": 4, "higher_moment_strength": 0.2})
    if kind in ("pairwise", "dual_cap"):
        result.update({"pairwise_moment_correction_steps": 4, "pairwise_moment_strength": 0.02})
    if kind == "dual_cap":
        result.update({"pairwise_particle_rms_cap": 2.0, "coordinatewise_standardized_cap": 0.98})
    return result


def evaluator(adapter, horizon: int, cfg):
    key = (int(horizon), tuple(sorted(cfg.items())))
    if key in _EVALUATOR_CACHE:
        return _EVALUATOR_CACHE[key]

    @tf.function(jit_compile=True, reduce_retracing=True)
    def run(theta, observations, initial, process, design):
        return finite_value_score(
            adapter,
            theta,
            observations,
            initial,
            process,
            design,
            **cfg,
        )

    _EVALUATOR_CACHE[key] = run
    return run


def noise(seed: int, horizon: int):
    return (
        tf.random.stateless_normal([N, 18], [seed, 101], dtype=tf.float32),
        tf.random.stateless_normal([horizon, N, 18], [seed, 102], dtype=tf.float32),
    )


def transition_observation_tangent_audit(adapter):
    theta = tf.zeros([3], tf.float32)
    states = tf.cast(sir.INITIAL_MEAN[None, :] + tf.random.stateless_normal([32, 18], [731, 1], dtype=tf.float64), tf.float32)
    process_noise = tf.random.stateless_normal([32, 18], [731, 2], dtype=tf.float32)
    tangent = tf.zeros([32, 18, 3], tf.float32)
    transition = adapter.transition_value(theta, states, process_noise, tf.constant(0))
    tangent_value = adapter.transition_tangent(theta, states, process_noise, tangent, tf.constant(0))
    h = tf.constant(1.0e-3, tf.float32)
    fd_cols = []
    for j in range(3):
        direction = tf.one_hot(j, 3, dtype=tf.float32)
        plus = adapter.transition_value(theta + h * direction, states, process_noise, tf.constant(0))
        minus = adapter.transition_value(theta - h * direction, states, process_noise, tf.constant(0))
        fd_cols.append((plus - minus) / (2.0 * h))
    transition_fd = tf.stack(fd_cols, axis=-1)
    transition_error = tf.reduce_max(tf.abs(tangent_value - transition_fd))
    observations = tf.random.stateless_normal([32, 9], [731, 4], dtype=tf.float32)
    obs_tangent = adapter.observation_tangent(theta, states, tangent, observations[0], tf.constant(0))
    obs_fd_cols = []
    for j in range(3):
        direction = tf.one_hot(j, 3, dtype=tf.float32)
        plus = adapter.observation_value(theta + h * direction, states, observations[0], tf.constant(0))
        minus = adapter.observation_value(theta - h * direction, states, observations[0], tf.constant(0))
        obs_fd_cols.append((plus - minus) / (2.0 * h))
    obs_fd = tf.stack(obs_fd_cols, axis=-1)
    obs_error = tf.reduce_max(tf.abs(obs_tangent - obs_fd))
    return {"transition_max_abs_error": float(transition_error.numpy()), "observation_max_abs_error": float(obs_error.numpy()), "pass": bool(transition_error < 2.0e-2 and obs_error < 2.0e-2), "step": float(h.numpy())}


def reset_jvp_audit():
    tf.random.set_seed(7317)
    source = tf.random.normal([1, 36, 18], dtype=tf.float32)
    # Scale the transported cloud so target_cov - transported_cov is positive
    # definite; arbitrary clouds can make the reset's gap Cholesky undefined.
    transported = 0.5 * source
    weights = tf.fill([1, 36], tf.constant(1.0 / 36.0, tf.float32))
    design = tf.random.normal([1, 36, 18], dtype=tf.float32)
    ridge = tf.constant([1.0e-5], tf.float32)
    source_tangent = tf.random.stateless_normal([1, 36, 18], [7317, 1], dtype=tf.float32)
    transported_tangent = tf.random.stateless_normal([1, 36, 18], [7317, 2], dtype=tf.float32)
    weights_tangent = tf.zeros([1, 36], tf.float32)
    design_tangent = tf.zeros_like(design)
    ridge_tangent = tf.zeros_like(ridge)
    actual = contract_e_chol_cloud_jvp_tf(source, weights, transported, design, ridge, source_tangent, weights_tangent, transported_tangent, design_tangent, ridge_tangent)
    h = tf.constant(2.0e-3, tf.float32)
    plus = contract_e_chol_cloud_forward_tf(source + h * source_tangent, weights, transported + h * transported_tangent, design, ridge)["particles"]
    minus = contract_e_chol_cloud_forward_tf(source - h * source_tangent, weights, transported - h * transported_tangent, design, ridge)["particles"]
    fd = (plus - minus) / (2.0 * h)
    error = tf.reduce_max(tf.abs(actual - fd))
    return {"max_abs_error": float(error.numpy()), "pass": bool(tf.math.is_finite(error) and error < 2.0e-2), "step": float(h.numpy())}


def score_ladder(adapter, observations, seed: int, horizon: int, cfg):
    initial, process = noise(seed, horizon)
    design = base._genut_design(18)
    theta = tf.zeros([3], tf.float32)
    run = evaluator(adapter, horizon, cfg)
    value, score, diagnostics = run(theta, observations[:horizon], initial, process, design)
    rows = []
    for j in range(3):
        direction = tf.one_hot(j, 3, dtype=tf.float32)
        fd_values = []
        endpoint_valid = True
        for h in FD_STEPS_LOCAL:
            plus, plus_score, plus_d = run(theta + h * direction, observations[:horizon], initial, process, design)
            minus, minus_score, minus_d = run(theta - h * direction, observations[:horizon], initial, process, design)
            endpoint_valid = endpoint_valid and bool(plus_d["program_valid"].numpy()) and bool(minus_d["program_valid"].numpy())
            fd_values.append(float(((plus - minus) / (2.0 * h)).numpy()))
        regression = fit_quadratic_step_regression(FD_STEPS_LOCAL, fd_values) if endpoint_valid else None
        comparison = evaluate_regression_derivative(float(score[j].numpy()), regression) if regression else {"diagnostic_pass": False, "reason": "invalid_endpoint"}
        rows.append({"parameter": j, "score": float(score[j].numpy()), "finite_difference": fd_values, "endpoint_valid": endpoint_valid, "comparison": comparison})
    increments = diagnostics["score_increments"]
    return {"seed": seed, "horizon": horizon, "value": float(value.numpy()), "score": [float(x) for x in score.numpy()], "score_increments": increments.numpy().tolist(), "increment_j0_abs_fraction": float((tf.reduce_max(tf.abs(increments[:, 0])) / tf.reduce_sum(tf.abs(increments[:, 0]))).numpy()), "increment_j0_l1": float(tf.reduce_sum(tf.abs(increments[:, 0])).numpy()), "fd": {"rows": rows, "all_pass": all(row["comparison"].get("diagnostic_pass", False) for row in rows)}, "program_valid": bool(diagnostics["program_valid"].numpy()), "max_row_residual": float(diagnostics["max_row_residual"].numpy()), "max_col_residual": float(diagnostics["max_col_residual"].numpy()), "minimum_gap_eigenvalue": float(diagnostics["minimum_covariance_gap_eigenvalue"].numpy()), "max_shape_displacement": float(diagnostics["maximum_shape_displacement"].numpy()), "max_normalized_shape_displacement": float(diagnostics["maximum_normalized_shape_displacement"].numpy())}


def summarize(rows, key):
    values = [float(row[key]) for row in rows]
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"count": len(values), "mean": statistics.mean(values), "sample_sd": sd, "mcse": sd / math.sqrt(len(values)), "min": min(values), "max": max(values)}


def run(output: Path):
    started = time.perf_counter()
    memory = GPU_MEMORY_POLICY
    tf.config.set_soft_device_placement(False)
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("GPU required")
    adapter = parameterized_austria_sir_candidate_adapter()
    with tf.device("/CPU:0"):
        observations_cpu = sir.fixed_observed_path(81120, 20)
    observations_gpu = sir.fixed_observed_path(81120, 20)
    if tensor_sha(observations_cpu) != EXPECTED_OBS_HASH:
        raise RuntimeError("fixed observation hash mismatch")
    observations = tf.cast(observations_cpu, tf.float32)
    law = {"observation_hash_cpu": tensor_sha(observations_cpu), "observation_hash_gpu": tensor_sha(observations_gpu), "observation_gpu_cpu_max_absolute_difference": float(tf.reduce_max(tf.abs(observations_gpu - observations_cpu)).numpy()), "transition_observation_tangents": transition_observation_tangent_audit(adapter), "reset_jvp": reset_jvp_audit()}
    controls_by_arm = {name: controls(name) for name in ("none", "diagonal", "pairwise", "dual_cap")}
    rows_by_arm = {}
    for arm, cfg in controls_by_arm.items():
        rows = [score_ladder(adapter, observations, seed, 20, cfg) for seed in SEEDS]
        rows_by_arm[arm] = {"controls": cfg, "rows": rows, "summary_j0": summarize(rows, "score") if False else {"score_j0": summarize([{"score": row["score"][0]} for row in rows], "score"), "value": summarize(rows, "value")}, "fd_pass_count": sum(row["fd"]["all_pass"] for row in rows), "mean_increment_concentration": statistics.mean(row["increment_j0_abs_fraction"] for row in rows)}
    horizon_rows = {}
    for horizon in HORIZONS:
        rows = [score_ladder(adapter, observations, seed, horizon, controls("diagonal")) for seed in SEEDS]
        horizon_rows[str(horizon)] = {"rows": rows, "score_j0": summarize([{"score": row["score"][0]} for row in rows], "score"), "fd_pass_count": sum(row["fd"]["all_pass"] for row in rows)}
    source_paths = (Path(__file__).resolve(), PLAN.resolve(), Path("bayesfilter/highdim/cubature_genut_filter.py").resolve(), Path("bayesfilter/highdim/cubature_genut_adapters.py").resolve(), Path("bayesfilter/highdim/cubature_genut_batch_tf.py").resolve(), Path("bayesfilter/highdim/higher_moment_contract_e.py").resolve(), Path("bayesfilter/highdim/ledh_contract_e_reset_tf.py").resolve())
    payload = {"schema": "bayesfilter.genut_sir_root_cause_hypotheses.v1", "status": "COMPLETE", "plan": str(PLAN.relative_to(ROOT)), "target": {"model": "Austria-SIR", "horizons": HORIZONS, "particle_count": N, "theta": [0.0, 0.0, 0.0], "observation_hash": tensor_sha(observations)}, "law_and_mechanics": law, "arms": rows_by_arm, "horizon_ladder": horizon_rows, "upstream_vetoes": {"classifier_gaussian_oracle": "FAILED_UPSTREAM", "genut_lgssm_oracle": "FAILED_UPSTREAM"}, "decision": {"H1_callback_equations": "pass" if law["transition_observation_tangents"]["pass"] else "fail", "H2_score_jvp": "pass" if all(row["fd"]["all_pass"] for arm in rows_by_arm.values() for row in arm["rows"]) else "diagnostic_failure_present", "H3_variance_localization": "descriptive", "H4_shape_transport_ablation": "descriptive", "H5_reset_jvp": "pass" if law["reset_jvp"]["pass"] else "fail", "root_cause_classification": "finite_program_or_transport_shape_variance_candidate", "ranking_supported": False}, "manifest": {"command": sys.argv, "environment": "/home/chakwong/anaconda3/envs/tftwogpu", "python": sys.executable, "tensorflow": tf.__version__, "physical_gpus": [{"device": d.name, "details": tf.config.experimental.get_device_details(d)} for d in physical], "logical_gpus": [d.name for d in logical], "memory_policy": dict(memory), "seeds": SEEDS, "dtype": "float32", "jit_compile": True, "wall_time_seconds": time.perf_counter() - started, "source_hashes": {str(path.relative_to(ROOT)): sha(path) for path in source_paths}}, "nonclaims": ["no exact SIR score", "no classifier oracle", "no ranking", "no HMC/default readiness"]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(safe(payload), indent=2, sort_keys=True) + "\n")
    lines = ["# GenUT Austria-SIR Root-Cause Hypotheses", "", f"- status: `{payload['status']}`", f"- H1 callbacks: `{payload['decision']['H1_callback_equations']}`", f"- H5 reset JVP: `{payload['decision']['H5_reset_jvp']}`", "", "| Arm | j0 mean | j0 SD | FD-pass rows | Increment concentration |", "|---|---:|---:|---:|---:|"]
    for arm, result in rows_by_arm.items():
        lines.append(f"| {arm} | {result['summary_j0']['score_j0']['mean']:.6g} | {result['summary_j0']['score_j0']['sample_sd']:.6g} | {result['fd_pass_count']}/{len(SEEDS)} | {result['mean_increment_concentration']:.6g} |")
    lines += ["", f"- classification: `{payload['decision']['root_cause_classification']}`", "- no exact score or ranking claim."]
    (output / "result.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(output), "classification": payload["decision"]["root_cause_classification"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
