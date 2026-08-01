#!/usr/bin/env python3
"""Diagnostic FP32/TF32/XLA feasibility run for the reduced SIR GenUT path."""

from __future__ import annotations

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

ARTIFACT_ROOT = ROOT / "docs/benchmarks/artifacts/genut_sir_feasibility_20260722/attempt06"
T_VALUES = (2, 5, 10)
SEEDS = (97001, 97002, 97003, 97004, 97005, 97006, 97007, 97008)
N_PARTICLES = 96
FD_RELATIVE_STEP = 8.0e-3
FD_RELATIVE_TOLERANCE = 5.0e-2


def _theta() -> tf.Tensor:
    return tf.constant([0.0, 0.0, 0.0], tf.float32)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configure_devices() -> dict[str, object]:
    physical = tf.config.list_physical_devices("GPU")
    growth = []
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
        growth.append({"device": device.name, "memory_growth": True})
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = [device.name for device in tf.config.list_logical_devices("GPU")]
    if not logical:
        raise RuntimeError("GPU is required for the claim-bearing feasibility run")
    return {
        "physical_gpu_count": len(physical),
        "physical_devices": growth,
        "logical_gpu_devices": logical,
        "tf32_enabled": True,
        "memory_policy": "memory_growth",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _rk4_truth(state: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    kappa = tf.constant(0.1, tf.float32) * tf.exp(theta[0])
    nu = tf.constant(1.0, tf.float32) * tf.exp(theta[1])
    step = tf.constant(0.005, tf.float32)
    current = state
    for _ in range(4):
        def rhs(value: tf.Tensor) -> tf.Tensor:
            s, i = value[0], value[1]
            force = kappa * s * i
            return tf.stack([-force, force - nu * i])
        k1 = rhs(current)
        k2 = rhs(current + 0.5 * step * k1)
        k3 = rhs(current + 0.5 * step * k2)
        k4 = rhs(current + step * k3)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def _dataset(horizon: int, seed: int) -> dict[str, tf.Tensor]:
    theta = _theta()
    initial_noise = tf.random.stateless_normal([2], [seed, 1], dtype=tf.float32)
    process_noise = tf.random.stateless_normal([horizon, 2], [seed, 2], dtype=tf.float32)
    observation_noise = tf.random.stateless_normal([horizon], [seed, 3], dtype=tf.float32)
    state = tf.constant([0.3, 0.2], tf.float32) + initial_noise * tf.constant([0.5, 0.4], tf.float32)
    states = []
    observations = []
    for index in range(horizon):
        physical = tf.where(
            tf.equal(index, 0), state, tf.stack([tf.maximum(state[0], 0.0), state[1]])
        )
        state = _rk4_truth(physical, theta) + process_noise[index] * tf.constant([0.5, 0.4], tf.float32)
        states.append(state)
        observations.append(state[1] + tf.constant(0.4, tf.float32) * observation_noise[index])
    return {
        "states": tf.stack(states),
        "observations": tf.reshape(tf.stack(observations), [horizon, 1]),
        "initial_noise": tf.random.stateless_normal([N_PARTICLES, 2], [seed, 101], dtype=tf.float32),
        "process_particle_noise": tf.random.stateless_normal(
            [horizon, N_PARTICLES, 2], [seed, 102], dtype=tf.float32
        ),
    }


def _make_evaluator(horizon: int, controls: dict[str, object]):
    from bayesfilter.highdim.cubature_genut_adapters import reduced_sir_candidate_adapter
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = reduced_sir_candidate_adapter(mechanics_fixture_only=True)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [3])
        observations = tf.ensure_shape(observations, [horizon, 2])
        initial_noise = tf.ensure_shape(initial_noise, [N_PARTICLES, 2])
        process_noise = tf.ensure_shape(process_noise, [horizon, N_PARTICLES, 2])
        design = tf.ensure_shape(design, [N_PARTICLES, 2])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter, theta, observations, initial_noise, process_noise, design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
            )

    return evaluate


def _run_value_score(evaluate, dataset, design, theta=None):
    if theta is None:
        theta = _theta()
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, dataset["observations"], dataset["initial_noise"],
        dataset["process_particle_noise"], design,
    )
    return {
        "value": float(value.numpy()),
        "score": [float(v) for v in score.numpy()],
        "finite": bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_sum_residual": float(tf.reduce_max(tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)).numpy()),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(value.device),
    }


def _fd_check(evaluate, dataset, design, analytic: list[float]) -> dict[str, object]:
    finite_difference = []
    for index in range(3):
        step = FD_RELATIVE_STEP
        plus = _theta() + tf.one_hot(index, 3, dtype=tf.float32) * step
        minus = _theta() - tf.one_hot(index, 3, dtype=tf.float32) * step
        vp = float(_run_value_score(evaluate, dataset, design, plus)["value"])
        vm = float(_run_value_score(evaluate, dataset, design, minus)["value"])
        finite_difference.append((vp - vm) / (2.0 * step))
    errors = [abs(a - f) / max(1.0, abs(f)) for a, f in zip(analytic, finite_difference)]
    return {
        "finite_difference_score": finite_difference,
        "relative_errors": errors,
        "pass": all(error <= FD_RELATIVE_TOLERANCE for error in errors),
        "diagnostic_only": True,
    }


def main() -> None:
    started = time.perf_counter()
    device_info = _configure_devices()
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    design = replicate_positive_genut(gaussian_genut_design(dim=2), num_particles=N_PARTICLES)
    controls = {"epsilon": 2.0, "sinkhorn_steps": 8, "ridge": 1.0e-5}
    all_rows = []
    for horizon in T_VALUES:
        evaluate = _make_evaluator(horizon, controls)
        for seed in SEEDS:
            dataset = _dataset(horizon, seed)
            # Candidate core expects observations with state width; pad the scalar
            # infectious observation into a zero-valued susceptible observation.
            observations = tf.concat([tf.zeros_like(dataset["observations"]), dataset["observations"]], axis=1)
            dataset["observations"] = observations
            result = _run_value_score(evaluate, dataset, design)
            result["fd_check"] = _fd_check(evaluate, dataset, design, result["score"])
            result.update({"horizon": horizon, "seed": seed})
            all_rows.append(result)
            if not result["finite"] or result["max_mean_residual"] > 5.0e-4:
                raise RuntimeError(f"feasibility veto at T={horizon}, seed={seed}: {result}")
    summary = {}
    for horizon in T_VALUES:
        rows = [row for row in all_rows if row["horizon"] == horizon]
        summary[str(horizon)] = {
            "count": len(rows),
            "value_mean": statistics.mean(row["value"] for row in rows),
            "value_sd": statistics.stdev(row["value"] for row in rows),
            "score_mean": [statistics.mean(row["score"][j] for row in rows) for j in range(3)],
            "score_sd": [statistics.stdev(row["score"][j] for row in rows) for j in range(3)],
            "all_fd_checks_pass": all(row["fd_check"]["pass"] for row in rows),
        }
    result = {
        "schema_version": "bayesfilter.genut_sir_feasibility.v1",
        "campaign_id": "genut-sir-feasibility-20260722-attempt01",
        "status": "historical_mechanics_fixture_only_not_model_feasibility",
        "target_id": "artificial_reduced_preclip_sir_j1_mechanics_fixture_v1",
        "suite_eligibility": "ineligible_actual_model_suite",
        "target_measure_note": "preclip filtering state with physical infectious observation; not clipped Austria source measure",
        "parameter_order": ["log_kappa_scale", "log_nu_scale", "log_obs_noise_scale"],
        "truth_theta": [0.0, 0.0, 0.0],
        "horizons": list(T_VALUES),
        "particle_count": N_PARTICLES,
        "seeds": list(SEEDS),
        "dtype": "float32",
        "tf32": True,
        "jit_compile": True,
        "controls": controls,
        "design": "gaussian_genut_dim2_replicated_positive_equal_mass",
        "device": device_info,
        "summary": summary,
        "rows": all_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "git_commit": _git_commit(),
        "nonclaims": ["exact likelihood", "exact score", "Austria SIR leaderboard admission", "HMC readiness", "default promotion"],
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_ROOT / "result.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    md_lines = [
        "# GenUT SIR Feasibility Diagnostic",
        "",
        "Status: `diagnostic_only_feasible` if this file exists.",
        "",
        "| T | value mean | value SD | score means | FD checks |",
        "|---:|---:|---:|---|---|",
    ]
    for horizon in T_VALUES:
        row = summary[str(horizon)]
        md_lines.append(f"| {horizon} | {row['value_mean']:.7g} | {row['value_sd']:.7g} | {row['score_mean']} | {row['all_fd_checks_pass']} |")
    md_lines += [
        "", "This is a feasibility result for the reduced continuous preclip target.",
        "It is not evidence for the clipped Austria SIR measure or leaderboard admission.",
        "", f"JSON artifact: `{json_path}`", "",
    ]
    md_path = ARTIFACT_ROOT / "result.md"
    md_path.write_text("\n".join(md_lines))
    manifest = {
        "command": "python docs/benchmarks/run_genut_sir_feasibility.py",
        "plan": "docs/plans/bayesfilter-genut-sir-feasibility-plan-2026-07-22.md",
        "git_commit": result["git_commit"],
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__},
        "result_json": str(json_path),
        "result_sha256": _sha256(json_path),
    }
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"artifact": str(json_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
