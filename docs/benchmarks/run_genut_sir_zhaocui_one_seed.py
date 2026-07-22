#!/usr/bin/env python3
"""Paired one-seed reduced SIR GenUT/fixed-design Zhao-Cui diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.cubature_genut_adapters import reduced_sir_candidate_adapter
from bayesfilter.highdim.cubature_genut_candidate import gaussian_genut_design, replicate_positive_genut
from bayesfilter.highdim.cubature_genut_filter import finite_value_score
from bayesfilter.highdim.sir_latent_preclip_tf import LatentPreclipSIRSSM


ARTIFACT_ROOT = ROOT / "docs/benchmarks/artifacts/genut_sir_zhaocui_one_seed_20260722/attempt02"
SEED = 97001
HORIZONS = (2, 5, 10)
N_PARTICLES = 96
THETA = tf.constant([0.0, 0.0, 0.0], tf.float32)
DTYPE = tf.float64


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _model() -> highdim.LatentPreclipSIRSSM:
    base = highdim.SpatialSIRSSM(
        kappa=tf.constant([0.1], DTYPE),
        nu=tf.constant([1.0], DTYPE),
        initial_mean=tf.constant([0.3, 0.2], DTYPE),
        neighbor_sets=((),),
        delta=0.02,
        rk4_internal_step=0.005,
        process_covariance=tf.constant([[0.25, 0.0], [0.0, 0.16]], DTYPE),
        observation_covariance=tf.constant([[0.16]], DTYPE),
        initial_covariance=tf.constant([[0.25, 0.0], [0.0, 0.16]], DTYPE),
        process_noise_policy="clip_susceptible_after_noise",
    )
    return LatentPreclipSIRSSM(highdim.ParameterizedZhaoCuiSIRSSM(base))


def _rk4(state: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    kappa = tf.constant(0.1, tf.float32) * tf.exp(theta[0])
    nu = tf.constant(1.0, tf.float32) * tf.exp(theta[1])
    step = tf.constant(0.005, tf.float32)
    value = state
    for _ in range(4):
        def rhs(x):
            force = kappa * x[0] * x[1]
            return tf.stack([-force, force - nu * x[1]])
        k1 = rhs(value)
        k2 = rhs(value + 0.5 * step * k1)
        k3 = rhs(value + 0.5 * step * k2)
        k4 = rhs(value + step * k3)
        value = value + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return value


def _dataset(horizon: int) -> dict[str, tf.Tensor]:
    initial_noise = tf.random.stateless_normal([2], [SEED, 1], dtype=tf.float32)
    process_noise = tf.random.stateless_normal([horizon, 2], [SEED, 2], dtype=tf.float32)
    observation_noise = tf.random.stateless_normal([horizon], [SEED, 3], dtype=tf.float32)
    state = tf.constant([0.3, 0.2], tf.float32) + initial_noise * tf.constant([0.5, 0.4], tf.float32)
    observations = [state[1] + 0.4 * observation_noise[0]]
    for index in range(1, horizon):
        previous = tf.stack([tf.maximum(state[0], 0.0), state[1]])
        state = _rk4(previous, THETA) + process_noise[index] * tf.constant([0.5, 0.4], tf.float32)
        observations.append(state[1] + 0.4 * observation_noise[index])
    return {
        "observations_f32": tf.stack(observations)[:, None],
        "initial_noise": tf.random.stateless_normal([N_PARTICLES, 2], [SEED, 101], dtype=tf.float32),
        "process_particle_noise": tf.random.stateless_normal([horizon, N_PARTICLES, 2], [SEED, 102], dtype=tf.float32),
    }


def _genut(horizon: int, dataset: dict[str, tf.Tensor]) -> dict[str, object]:
    adapter = reduced_sir_candidate_adapter(
        transition_before_first_observation=False,
        mechanics_fixture_only=True,
    )
    design = replicate_positive_genut(gaussian_genut_design(dim=2), num_particles=N_PARTICLES)

    @tf.function(jit_compile=False)
    def evaluate(theta, observations, initial_noise, process_noise, residual_design):
        return finite_value_score(
            adapter, theta, observations, initial_noise, process_noise, residual_design,
            epsilon=2.0, sinkhorn_steps=8, ridge=1e-5,
        )

    result = evaluate(
        THETA,
        tf.concat([tf.zeros_like(dataset["observations_f32"]), dataset["observations_f32"]], axis=1),
        dataset["initial_noise"], dataset["process_particle_noise"], design,
    )
    value, score, diagnostics = result
    return {
        "value": float(value.numpy()),
        "score": [float(v) for v in score.numpy()],
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "finite": bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
    }


def _zhao_config(horizon: int) -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    degree = 6
    basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree) for _ in range(2)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 2, 1), ridge=1e-10, max_sweeps=3, sweep_order=(0, 1, 1, 0),
            row_budget=800, column_budget=128, dense_matrix_byte_budget=4_000_000,
            normal_matrix_byte_budget=500_000, condition_number_warning=1e12,
            condition_number_veto=1e16, holdout_tolerance=1.0,
        ),
        density_tau=0.0, normalizer_floor=1e-14, denominator_floor=1e-14,
        retained_storage_byte_budget=20_000_000,
        coordinate_maps=(highdim.AffineCoordinateMap(offset=tf.constant([0.3, 0.2], DTYPE), matrix=tf.linalg.diag(tf.constant([1.5, 1.2], DTYPE))),),
        measure_convention=convention, deterministic_seed=f"sir-zhaocui-one-seed-t{horizon}",
        product_basis=basis,
        initial_cores=highdim.norm_balanced_initial_cores(basis, (1, 2, 1)),
        fit_quadrature_order=7,
    )


def _zhaocui(horizon: int, observations: tf.Tensor) -> dict[str, object]:
    model = _model()
    config = _zhao_config(horizon)
    derivative = highdim.FixedBranchDerivativeConfig(
        parameter_indices=(0, 1, 2), finite_difference_h=(1e-3,), solve_condition_number_veto=1e16,
    )
    # This local route is explicitly diagnostic/historical; it is not the
    # production fixed-variant source route and has a different first-observation timing.
    result = highdim.multistate_nonlinear_fixed_design_tt_score_path(
        model, tf.cast(THETA, DTYPE), tf.cast(observations, DTYPE), config, derivative,
        fixture_id=f"sir-zhaocui-one-seed-t{horizon}",
        branch_seed_prefix=f"sir-zhaocui-one-seed-t{horizon}",
    )
    return {
        "value": float(result.log_likelihood.numpy()),
        "score": [float(v) for v in tf.reshape(result.score, [-1]).numpy()],
        "finite": bool(tf.math.is_finite(result.log_likelihood).numpy()) and bool(tf.reduce_all(tf.math.is_finite(result.score)).numpy()),
        "route_id": result.diagnostics["score_path"],
        "route_role": result.diagnostics["route_role"],
        "timing": "initial_observation_first",
    }


def _dense_reference(horizon: int, observations: tf.Tensor) -> dict[str, object]:
    from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
        dense_latent_sir_value_and_manual_score,
        prepare_reduced_dense_grids,
    )

    model = _model()
    grids = prepare_reduced_dense_grids(
        model, tf.cast(THETA, DTYPE), time_steps=horizon - 1,
        order=29, radius=7.0, integration_rule="split_gauss_legendre",
    )
    result = dense_latent_sir_value_and_manual_score(
        model, tf.cast(THETA, DTYPE), tf.cast(observations, DTYPE), grids
    )
    return {
        "value": float(result["objective"].numpy()),
        "score": [float(v) for v in result["score"].numpy()],
        "maximum_boundary_mass": float(tf.reduce_max(result["boundary_mass_history"]).numpy()),
        "configuration": {"order": 29, "radius": 7.0, "rule": "split_gauss_legendre"},
    }


def main() -> None:
    rows = []
    for horizon in HORIZONS:
        dataset = _dataset(horizon)
        genut = _genut(horizon, dataset)
        zhao = _zhaocui(horizon, dataset["observations_f32"])
        dense = _dense_reference(horizon, dataset["observations_f32"])
        rows.append({
            "horizon": horizon,
            "seed": SEED,
            "observations": [float(v) for v in dataset["observations_f32"][:, 0].numpy()],
            "genut": genut,
            "zhao_cui": zhao,
            "dense_reference": dense,
            "value_difference_genut_minus_zhao_cui": genut["value"] - zhao["value"],
            "score_difference_genut_minus_zhao_cui": [a - b for a, b in zip(genut["score"], zhao["score"])],
            "genut_error_vs_dense": {
                "value": genut["value"] - dense["value"],
                "score": [a - b for a, b in zip(genut["score"], dense["score"])],
            },
            "zhao_cui_error_vs_dense": {
                "value": zhao["value"] - dense["value"],
                "score": [a - b for a, b in zip(zhao["score"], dense["score"])],
            },
        })
    result = {
        "schema_version": "bayesfilter.genut_sir_zhaocui_one_seed.v1",
        "status": "historical_mechanics_fixture_only_not_model_comparison",
        "target_id": "artificial_reduced_preclip_sir_j1_mechanics_fixture_v1",
        "suite_eligibility": "ineligible_actual_model_suite",
        "seed": SEED,
        "theta": [0.0, 0.0, 0.0],
        "particle_count": N_PARTICLES,
        "genut_route": "cubature_genut_nonfused_positive_ot_candidate_v1",
        "zhao_cui_route": "multistate_nonlinear_fixed_design_tt_score_path",
        "comparability": "same_target_same_initial_observation_timing_different_finite_approximations",
        "rows": rows,
        "nonclaims": ["oracle", "accuracy", "superiority", "leaderboard", "source_faithfulness"],
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_ROOT / "result.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = ["# One-Seed GenUT/Zhao-Cui Reduced SIR Diagnostic", "", "Status: `diagnostic_only_same_target_one_seed`.", "", "| T | Dense value | GenUT value | Zhao-Cui value | GenUT score | Zhao-Cui score | Dense score |", "|---:|---:|---:|---:|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['horizon']} | {row['dense_reference']['value']:.8g} | {row['genut']['value']:.8g} | {row['zhao_cui']['value']:.8g} | {row['genut']['score']} | {row['zhao_cui']['score']} | {row['dense_reference']['score']} |")
    lines += ["", "Both methods use the same initial-observation-first target. Zhao-Cui is a diagnostic retained-grid approximation, not an oracle; the dense grid is the accuracy anchor.", "", f"JSON: `{path}`"]
    (ARTIFACT_ROOT / "result.md").write_text("\n".join(lines) + "\n")
    (ARTIFACT_ROOT / "manifest.json").write_text(json.dumps({"plan": "docs/plans/bayesfilter-genut-sir-zhaocui-one-seed-plan-2026-07-22.md", "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
