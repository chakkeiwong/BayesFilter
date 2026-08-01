#!/usr/bin/env python3
"""One-seed GenUT feasibility run for the canonical predator-prey T20 row."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

ARTIFACT_ROOT = ROOT / "docs/benchmarks/artifacts/genut_predator_prey_one_seed_20260722/attempt03"
SEED = 81104
HORIZON = 20
N_PARTICLES = 96
THETA_VALUES = (0.6, 114.0, 25.0, 0.3, 0.5, 0.5)


def _configure_gpu() -> dict[str, object]:
    physical = tf.config.list_physical_devices("GPU")
    rows = []
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
        rows.append({"device": device.name, "memory_growth": True})
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = [device.name for device in tf.config.list_logical_devices("GPU")]
    if not logical:
        raise RuntimeError("GPU is required for the GenUT feasibility run")
    return {
        "physical_devices": rows,
        "logical_devices": logical,
        "memory_policy": "memory_growth",
        "tf32_enabled": True,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _genut(dataset: dict[str, tf.Tensor], controls: dict[str, object]) -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_adapters import predator_prey_candidate_adapter
    from bayesfilter.highdim.cubature_genut_candidate import gaussian_genut_design, replicate_positive_genut
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    adapter = predator_prey_candidate_adapter()
    design = replicate_positive_genut(gaussian_genut_design(dim=2), num_particles=N_PARTICLES)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, residual_design):
        theta = tf.ensure_shape(theta, [6])
        observations = tf.ensure_shape(observations, [HORIZON, 2])
        initial_noise = tf.ensure_shape(initial_noise, [N_PARTICLES, 2])
        process_noise = tf.ensure_shape(process_noise, [HORIZON, N_PARTICLES, 2])
        residual_design = tf.ensure_shape(residual_design, [N_PARTICLES, 2])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter, theta, observations, initial_noise, process_noise,
                residual_design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=False,
            )

    theta = tf.constant(THETA_VALUES, tf.float32)
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, dataset["observations"],
        dataset["initial_particle_noise"], dataset["process_particle_noise"], design,
    )
    fd_steps = tf.constant([2.0e-3, 2.0e-1, 5.0e-2, 2.0e-3, 2.0e-3, 2.0e-3], tf.float32)
    fd_score = []
    for index in range(6):
        direction = tf.one_hot(index, 6, dtype=tf.float32)
        plus, _, _ = evaluate(
            theta + fd_steps[index] * direction, dataset["observations"],
            dataset["initial_particle_noise"], dataset["process_particle_noise"], design,
        )
        minus, _, _ = evaluate(
            theta - fd_steps[index] * direction, dataset["observations"],
            dataset["initial_particle_noise"], dataset["process_particle_noise"], design,
        )
        fd_score.append(float(((plus - minus) / (2.0 * fd_steps[index])).numpy()))
    analytic_score = [float(v) for v in score.numpy()]
    fd_relative_errors = [
        abs(a - b) / max(1.0, abs(b)) for a, b in zip(analytic_score, fd_score)
    ]
    return {
        "value": float(value.numpy()),
        "score": analytic_score,
        "same_scalar_fd": {
            "steps": [float(v) for v in fd_steps.numpy()],
            "score": fd_score,
            "relative_errors": fd_relative_errors,
            "maximum_relative_error": max(fd_relative_errors),
            "diagnostic_only": True,
        },
        "finite": bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_sum_residual": float(tf.reduce_max(tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)).numpy()),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(value.device),
        "route_id": "cubature_genut_nonfused_positive_ot_candidate_v1",
    }


def _zhao_cui(observations: tf.Tensor) -> dict[str, object]:
    # The existing T20 route is a fixed-design diagnostic/historical route.
    import bayesfilter.highdim as highdim
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import (
        _zhao_cui_predator_prey_multistate_tt_config,
    )

    model = highdim.p30_predator_prey_fixture_model()
    derivative = highdim.FixedBranchDerivativeConfig(
        parameter_indices=tuple(range(model.parameter_dim())),
        finite_difference_h=(),
        solve_condition_number_veto=1e30,
    )
    started = time.perf_counter()
    result = highdim.multistate_nonlinear_fixed_design_tt_score_path(
        model,
        model.true_parameters(),
        tf.cast(observations, tf.float64),
        _zhao_cui_predator_prey_multistate_tt_config("genut-one-seed-zhaocui"),
        derivative,
        fixture_id="genut.zhaocui.predator-prey-t20.one-seed.v1",
        initial_target_id="genut.zhaocui.predator-prey-t20.initial.v1",
        transition_target_id="genut.zhaocui.predator-prey-t20.transition.v1",
        branch_seed_prefix="genut-one-seed-zhaocui-predator-prey-t20",
    )
    return {
        "value": float(result.log_likelihood.numpy()),
        "score": [float(v) for v in tf.reshape(result.score, [-1]).numpy()],
        "finite": bool(tf.math.is_finite(result.log_likelihood).numpy()) and bool(tf.reduce_all(tf.math.is_finite(result.score)).numpy()),
        "elapsed_seconds": time.perf_counter() - started,
        "route_id": "multistate_nonlinear_fixed_design_tt_score_path",
        "route_role": result.diagnostics["route_role"],
        "admission": result.diagnostics["leaderboard_admission"],
    }


def main() -> None:
    device = _configure_gpu()
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _predator_prey_dataset

    raw = _predator_prey_dataset(SEED)
    observations = tf.cast(tf.convert_to_tensor(raw["observations"][:HORIZON]), tf.float32)
    model = {
        "family": "PredatorPreySSM",
        "state_dimension": 2,
        "observation_dimension": 2,
        "parameter_order": ["r", "K", "a", "s", "u", "v"],
        "delta": 2.0,
        "rk4_internal_step": 0.1,
        "process_covariance": "4I",
        "observation_covariance": "4I",
        "initial_covariance": "I",
    }
    # Use the same canonical DGP observations but independent particle noise for GenUT.
    dataset = {
        "observations": observations,
        "initial_particle_noise": tf.random.stateless_normal([N_PARTICLES, 2], [SEED, 101], dtype=tf.float32),
        "process_particle_noise": tf.random.stateless_normal([HORIZON, N_PARTICLES, 2], [SEED, 102], dtype=tf.float32),
    }
    controls = {"epsilon": 2.0, "sinkhorn_steps": 8, "ridge": 1.0e-5}
    genut = _genut(dataset, controls)
    zhao = _zhao_cui(tf.cast(observations, tf.float64))
    result = {
        "schema_version": "bayesfilter.genut_predator_prey_one_seed.v1",
        "campaign_id": "genut-predator-prey-one-seed-20260722-attempt02",
        "status": "diagnostic_only_feasible",
        "row_id": "zhao_cui_predator_prey_T20",
        "seed": SEED,
        "horizon": HORIZON,
        "particle_count": N_PARTICLES,
        "theta": list(THETA_VALUES),
        "parameter_order": ["r", "K", "a", "s", "u", "v"],
        "dtype": "float32",
        "tf32": True,
        "jit_compile": True,
        "controls": controls,
        "design": "gaussian_genut_dim2_replicated_positive_equal_mass",
        "dgp_model_manifest": model,
        "observation_sha256": hashlib.sha256(tf.io.serialize_tensor(observations).numpy()).hexdigest(),
        "device": device,
        "genut": genut,
        "zhao_cui": zhao,
        "value_difference_genut_minus_zhao_cui": genut["value"] - zhao["value"],
        "score_difference_genut_minus_zhao_cui": [a - b for a, b in zip(genut["score"], zhao["score"])],
        "nonclaims": ["exact likelihood", "exact score", "Zhao-Cui oracle", "superiority", "leaderboard admission", "default promotion"],
        "git_commit": _git_commit(),
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# GenUT Predator-Prey T20 One-Seed Feasibility",
        "", "Status: `diagnostic_only_feasible`.", "",
        "| Route | value | score `(r,K,a,s,u,v)` | finite |",
        "|---|---:|---|---|",
        f"| GenUT | {genut['value']:.9g} | {genut['score']} | {genut['finite']} |",
        f"| Zhao-Cui diagnostic | {zhao['value']:.9g} | {zhao['score']} | {zhao['finite']} |",
        "",
        f"Value difference (GenUT - Zhao-Cui): `{result['value_difference_genut_minus_zhao_cui']:.9g}`",
        f"Score difference: `{result['score_difference_genut_minus_zhao_cui']}`",
        "",
        "Zhao-Cui is the local fixed-design retained-grid diagnostic, not an oracle.",
        f"JSON: `{path}`",
    ]
    (ARTIFACT_ROOT / "result.md").write_text("\n".join(lines) + "\n")
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps({"plan": "docs/plans/bayesfilter-genut-predator-prey-one-seed-plan-2026-07-22.md", "git_commit": result["git_commit"], "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "environment": {"python": platform.python_version(), "tensorflow": tf.__version__}}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
