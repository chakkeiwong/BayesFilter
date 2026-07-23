#!/usr/bin/env python3
"""One-seed descriptive UKF/SGQF/Zhao-Cui/GenUT feasibility matrix.

This is deliberately not a leaderboard runner.  It reuses existing analytical
comparators and the repaired GenUT finite value/recursive-score program, and
records unavailable cells rather than substituting historical routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
import tensorflow_probability as tfp

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
N = 1002
CONTROLS = {"epsilon": 2.0, "sinkhorn_steps": 8, "balance_steps": 8, "ridge": 1.0e-5}
METHODS = ("ukf", "sgqf", "zhao_cui", "genut")


def _hash(value: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _finite(value: Any) -> bool:
    value = tf.convert_to_tensor(value)
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def _cell(method: str, *, status: str, reason: str | None = None, **values: Any) -> dict[str, Any]:
    result = {"method": method, "status": status, "reason": reason}
    result.update(values)
    return result


def _genut(
    *,
    model_id: str,
    theta: tf.Tensor,
    observations: tf.Tensor,
    state_dim: int,
    transition_before_first_observation: bool,
    seed: int,
) -> dict[str, Any]:
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    if model_id == "ksc_sv":
        from bayesfilter.highdim.cubature_genut_adapters import ksc_mixture_sv_candidate_adapter

        adapter = ksc_mixture_sv_candidate_adapter()
    elif model_id == "exact_sv":
        from bayesfilter.highdim.cubature_genut_adapters import exact_transformed_sv_candidate_adapter

        adapter = exact_transformed_sv_candidate_adapter()
    elif model_id == "generalized_sv":
        from bayesfilter.highdim.cubature_genut_adapters import generalized_sv_prior_mean_candidate_adapter

        adapter = generalized_sv_prior_mean_candidate_adapter()
    elif model_id == "predator_prey":
        from bayesfilter.highdim.cubature_genut_adapters import predator_prey_candidate_adapter

        adapter = predator_prey_candidate_adapter()
    else:
        raise ValueError(model_id)

    design = replicate_positive_genut(
        gaussian_genut_design(dim=state_dim), num_particles=N
    )
    horizon = int(observations.shape[0])
    parameter_dim = int(theta.shape[0])

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta_value, observations_value, initial_noise, process_noise, design_value):
        theta_value = tf.ensure_shape(theta_value, [parameter_dim])
        observations_value = tf.ensure_shape(observations_value, [horizon, int(observations.shape[1])])
        initial_noise = tf.ensure_shape(initial_noise, [N, state_dim])
        process_noise = tf.ensure_shape(process_noise, [horizon, N, state_dim])
        design_value = tf.ensure_shape(design_value, [N, state_dim])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta_value,
                observations_value,
                initial_noise,
                process_noise,
                design_value,
                epsilon=CONTROLS["epsilon"],
                sinkhorn_steps=CONTROLS["sinkhorn_steps"],
                balance_steps=CONTROLS["balance_steps"],
                ridge=CONTROLS["ridge"],
                transition_before_first_observation=transition_before_first_observation,
            )

    initial = tf.random.stateless_normal(
        [N, state_dim], [seed, 101], dtype=tf.float32
    )
    process = tf.random.stateless_normal(
        [horizon, N, state_dim], [seed, 102], dtype=tf.float32
    )
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        tf.cast(theta, tf.float32),
        tf.cast(observations, tf.float32),
        initial,
        process,
        design,
    )
    elapsed = time.perf_counter() - started
    score_sum_residual = tf.reduce_max(
        tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score)
    )
    maximum_residual = tf.reduce_max(
        tf.stack(
            [
                diagnostics["max_mean_residual"],
                diagnostics["max_row_residual"],
                diagnostics["max_col_residual"],
                score_sum_residual,
            ]
        )
    )
    finite = _finite(value) and _finite(score) and bool(diagnostics["program_valid"].numpy())
    return _cell(
        "genut",
        status="executed" if finite else "vetoed",
        reason=None if finite else "GenUT finite/program-valid gate failed",
        value=float(value.numpy()) if finite else None,
        score=[float(x) for x in score.numpy()] if finite else None,
        score_provenance="recursive_forward_sensitivity_same_finite_value_program",
        route_id="cubature_genut_nonfused_positive_ot_row_quotient_candidate_v2",
        dtype="float32",
        tf32=True,
        jit_compile=True,
        device=str(value.device),
        particle_count=N,
        particle_seed=seed,
        controls=CONTROLS,
        transition_before_first_observation=transition_before_first_observation,
        elapsed_seconds=elapsed,
        max_mean_residual=float(diagnostics["max_mean_residual"].numpy()),
        max_row_residual=float(diagnostics["max_row_residual"].numpy()),
        max_col_residual=float(diagnostics["max_col_residual"].numpy()),
        score_increment_sum_residual=float(score_sum_residual.numpy()),
        maximum_residual=float(maximum_residual.numpy()),
        minimum_row_mass=float(diagnostics["minimum_row_mass"].numpy()),
        maximum_post_quotient_column_tv_error=float(
            diagnostics["maximum_post_quotient_column_tv_error"].numpy()
        ),
    )


def _ksc_row() -> dict[str, Any]:
    import bayesfilter.highdim as highdim
    from bayesfilter.highdim.sv_mixture_cut4 import transformed_sv_observations
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import _zhao_cui_scalar_tt_config
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset

    raw_payload = _sv_dataset(81101)
    raw = tf.cast(raw_payload["observations"][:10], tf.float64)
    theta = tf.cast(raw_payload["truth_theta"], tf.float64)
    transformed = transformed_sv_observations(raw, offset=1.0e-8)
    normal = tfp.distributions.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64))
    gamma = normal.cdf(theta[0])
    beta = tf.exp(theta[1])
    cells: list[dict[str, Any]] = []
    started = time.perf_counter()
    ukf = highdim.independent_panel_sv_mixture_ukf_score(raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64))
    cells.append(_cell("ukf", status="executed", value=float(ukf.log_likelihood.numpy()), score=[float(x) for x in ukf.score.numpy()], score_provenance="principal_sqrt_ukf_manual_recursive_score", route_id="principal_sqrt_ukf_independent_panel_ksc_mixture", dtype="float64", elapsed_seconds=time.perf_counter() - started))
    started = time.perf_counter()
    sgqf = highdim.independent_panel_sv_mixture_fixed_sgqf_score(raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), sparse_level=2)
    cells.append(_cell("sgqf", status="executed", value=float(sgqf.log_likelihood.numpy()), score=[float(x) for x in sgqf.score.numpy()], score_provenance="fixed_sgqf_analytical_recursive_score", route_id="fixed_sgqf_independent_panel_ksc_mixture", dtype="float64", elapsed_seconds=time.perf_counter() - started))
    started = time.perf_counter()
    zc = highdim.independent_panel_sv_mixture_zhaocui_tt_score(raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), config=_zhao_cui_scalar_tt_config("one-seed-ksc"), derivative_config=highdim.FixedBranchDerivativeConfig(parameter_indices=(0, 1), finite_difference_h=(), solve_condition_number_veto=1e14))
    cells.append(_cell("zhao_cui", status="executed", value=float(zc.log_likelihood.numpy()), score=[float(x) for x in zc.score.numpy()], score_provenance="zhao_cui_fixed_variant_manual_parameter_score_methods_only", route_id="zhao_cui_ksc_mixture_fixed_branch_tt", dtype="float64", elapsed_seconds=time.perf_counter() - started, route_role="fixed_variant_diagnostic"))
    genut = _genut(model_id="ksc_sv", theta=theta, observations=transformed, state_dim=1, transition_before_first_observation=False, seed=81101)
    cells.append(genut)
    return {"model_id": "ksc_sv", "target_id": "amended_initial_observation_first_ksc_sv_prefix10_seed81101", "seed": 81101, "horizon": 10, "theta": [float(x) for x in theta.numpy()], "parameter_order": ["z_gamma", "log_beta"], "raw_observation_sha256": _hash(raw), "transformed_observation_sha256": _hash(transformed), "target_observation_policy": "log(y^2+1e-8) seven-component KSC mixture", "time_order": "stationary_initial_draw_then_observe_y0_to_y9_before_transitions", "cells": cells}


def _exact_sv_row() -> dict[str, Any]:
    import bayesfilter.highdim as highdim
    from bayesfilter.highdim.sv_mixture_cut4 import exact_transformed_sv_observations
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import _zhao_cui_scalar_tt_config
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset

    raw_payload = _sv_dataset(81101)
    raw = tf.cast(raw_payload["observations"][:10], tf.float64)
    theta = tf.cast(raw_payload["truth_theta"], tf.float64)
    transformed = exact_transformed_sv_observations(raw)
    gamma = tfp.distributions.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)).cdf(theta[0])
    beta = tf.exp(theta[1])
    cells: list[dict[str, Any]] = [_cell("ukf", status="not_comparable", reason="available UKF is an augmented-noise raw-observation Gaussian closure, not this exact transformed target")]
    started = time.perf_counter()
    sgqf = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), sparse_level=2)
    cells.append(_cell("sgqf", status="executed", value=float(sgqf.log_likelihood.numpy()), score=[float(x) for x in sgqf.score.numpy()], score_provenance="fixed_sgqf_analytical_recursive_score", route_id="fixed_sgqf_direct_exact_transformed_sv", dtype="float64", elapsed_seconds=time.perf_counter() - started))
    started = time.perf_counter()
    zc = highdim.exact_transformed_sv_independent_panel_zhaocui_tt_score(raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), config=_zhao_cui_scalar_tt_config("one-seed-exact-sv"), derivative_config=highdim.FixedBranchDerivativeConfig(parameter_indices=(0, 1), finite_difference_h=(), solve_condition_number_veto=1e14))
    cells.append(_cell("zhao_cui", status="executed", value=float(zc.log_likelihood.numpy()), score=[float(x) for x in zc.score.numpy()], score_provenance="zhao_cui_fixed_variant_manual_parameter_score_methods_only", route_id="zhao_cui_exact_transformed_sv_fixed_branch_tt", dtype="float64", elapsed_seconds=time.perf_counter() - started, route_role="fixed_variant_diagnostic"))
    cells.append(_genut(model_id="exact_sv", theta=theta, observations=transformed, state_dim=1, transition_before_first_observation=False, seed=81101))
    return {"model_id": "exact_transformed_sv", "target_id": "amended_initial_observation_first_exact_sv_prefix10_seed81101", "seed": 81101, "horizon": 10, "theta": [float(x) for x in theta.numpy()], "parameter_order": ["z_gamma", "log_beta"], "raw_observation_sha256": _hash(raw), "transformed_observation_sha256": _hash(transformed), "target_observation_policy": "exact log(y^2) log-chi-square observation", "time_order": "stationary_initial_draw_then_observe_y0_to_y9_before_transitions", "cells": cells}


def _generalized_sv_row() -> dict[str, Any]:
    import bayesfilter.highdim as highdim
    from bayesfilter.highdim.generalized_sv_sgqf_tf import (
        generalized_sv_sgqf_value_score_status,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _generalized_sv_prior_mean_dataset
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import _zhao_cui_scalar_tt_config

    raw = _generalized_sv_prior_mean_dataset(81105)
    observations = tf.cast(raw["observations"][:10], tf.float64)
    theta = tf.cast(raw["truth_theta"], tf.float64)
    model = highdim.GeneralizedSVPriorMeanSSM()
    deriv = highdim.FixedBranchDerivativeConfig(parameter_indices=(0, 1, 2), finite_difference_h=(), solve_condition_number_veto=1e14)
    started = time.perf_counter()
    sgqf_value, sgqf_score, status = generalized_sv_sgqf_value_score_status(
        theta, observations, sparse_level=3
    )
    cells = [_cell("sgqf", status="executed" if int(status["status_code"].numpy()) == 0 else "vetoed", value=float(sgqf_value.numpy()), score=[float(x) for x in sgqf_score.numpy()], score_provenance="generalized_sv_sgqf_analytical_recursive_score", route_id="fixed_sgqf_generalized_sv_prior_mean_raw_y_level3_gaussian_projection_manual_score_v1", dtype="float64", elapsed_seconds=time.perf_counter() - started)]
    started = time.perf_counter()
    zc = highdim.scalar_nonlinear_fixed_design_tt_score_path(model, theta, observations, _zhao_cui_scalar_tt_config("one-seed-generalized-sv", basis_degree=16, quadrature_order=41), deriv, fixture_id="one-seed.generalized-sv", initial_target_id="one-seed.generalized-sv.initial", transition_target_id="one-seed.generalized-sv.transition", branch_seed_prefix="one-seed-generalized-sv")
    cells.append(_cell("zhao_cui", status="executed", value=float(zc.log_likelihood.numpy()), score=[float(x) for x in tf.reshape(zc.score, [-1]).numpy()], score_provenance="zhao_cui_fixed_variant_manual_parameter_score_methods_only", route_id="zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt", dtype="float64", elapsed_seconds=time.perf_counter() - started, route_role="fixed_variant_diagnostic"))
    cells.append(_cell("ukf", status="not_comparable", reason="no reviewed same-target generalized-SV UKF route is implemented"))
    cells.append(_genut(model_id="generalized_sv", theta=theta, observations=observations, state_dim=1, transition_before_first_observation=True, seed=81105))
    return {"model_id": "generalized_sv", "target_id": "zhao_cui_generalized_sv_synthetic_from_estimated_values_prefix10", "seed": 81105, "horizon": 10, "theta": [float(x) for x in theta.numpy()], "parameter_order": ["z_gamma", "log_tau", "mu_over_tau"], "observation_sha256": _hash(observations), "target_observation_policy": "raw Gaussian SV observation y=sqrt(exp(tau*x))*epsilon", "time_order": "stationary_initial_draw_then_transition_before_every_observation", "cells": cells}


def _predator_prey_row() -> dict[str, Any]:
    from bayesfilter.highdim.cubature_genut_adapters import predator_prey_candidate_adapter
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import make_predator_prey_source_sgqf_route
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    route = make_predator_prey_source_sgqf_route()
    observations = route.observations
    theta = tf.constant([0.6, 114.0, 25.0, 0.3, 0.5, 0.5], tf.float64)
    started = time.perf_counter()
    sgqf_value, sgqf_score, status = route.physical_value_score_status(theta[None, :])
    cells = [_cell("sgqf", status="executed" if int(status["status_code"][0].numpy()) == 0 else "vetoed", value=float(sgqf_value[0].numpy()), score=[float(x) for x in sgqf_score[0].numpy()], score_provenance="fixed_sgqf_analytical_recursive_score", dtype="float64", elapsed_seconds=time.perf_counter() - started, route_id=route.manifest["route_id"], route_identity=route.route_identity)]
    cells.append(_cell("ukf", status="not_comparable", reason="available UKF uses initial-observation-first timing, while this canonical source row is transition-then-observe"))
    cells.append(_cell("zhao_cui", status="not_implemented_or_not_comparable", reason="fixed-variant Zhao-Cui predator-prey source-route evaluator is not implemented; historical retained-grid route is demoted"))
    cells.append(_genut(model_id="predator_prey", theta=theta, observations=observations, state_dim=2, transition_before_first_observation=True, seed=81104))
    return {"model_id": "predator_prey", "target_id": route.manifest["target_id"], "seed": 81104, "horizon": 20, "theta": [float(x) for x in theta.numpy()], "parameter_order": ["r", "K", "a", "s", "u", "v"], "observation_sha256": _hash(observations), "target_observation_policy": "additive Gaussian observation of source-order RK4 trajectory", "time_order": route.manifest["time_order"], "cells": cells}


def _render(payload: dict[str, Any]) -> str:
    lines = ["# One-Seed Four-Filter Feasibility", "", "All values are descriptive one-seed diagnostics; no method ranking is supported.", "", "| Model | Method | Status | Value | Score | Dtype | Reason |", "|---|---|---|---:|---|---|---|"]
    for row in payload["models"]:
        for cell in row["cells"]:
            value = "n/a" if cell.get("value") is None else f"{cell['value']:.9g}"
            score = "n/a" if cell.get("score") is None else str(cell["score"])
            lines.append(f"| {row['model_id']} | {cell['method']} | {cell['status']} | {value} | `{score}` | {cell.get('dtype', 'n/a')} | {cell.get('reason') or ''} |")
    lines += ["", "## Inference status", "", "| Item | Status |", "|---|---|", "| Hard veto screen | finite executed cells and explicit unavailable reasons recorded |", "| Statistically supported ranking | none; one seed |", "| Descriptive differences | all value/score differences only |", "| Default/leaderboard readiness | not evaluated |", "| Next evidence | target-specific tuning and multi-seed uncertainty on rows with complete coverage |", "", f"JSON: `{payload['artifact_json']}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("one-seed feasibility requires a visible logical GPU")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    models = [_ksc_row(), _exact_sv_row(), _generalized_sv_row(), _predator_prey_row()]
    payload: dict[str, Any] = {
        "schema_version": "bayesfilter.one_seed_four_filter_feasibility.v1",
        "plan": "docs/plans/bayesfilter-one-seed-four-filter-feasibility-plan-2026-07-22.md",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "models": models,
        "methods": METHODS,
        "configuration": {"genut_particle_count": N, "genut_controls": CONTROLS, "dtype_genut": "float32", "tf32": True, "jit_compile": True, "score_policy": "analytical/manual/recursive only"},
        "device": {"logical_devices": [x.name for x in logical], "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"},
        "memory_policy": MEMORY_POLICY,
        "gpu_allocator": {k: int(v) for k, v in tf.config.experimental.get_memory_info("GPU:0").items()},
        "wall_time_seconds": time.perf_counter() - started,
        "nonclaims": ["no method ranking", "no uncertainty interval", "no exact nonlinear oracle", "no default or leaderboard promotion", "missing cells are not numerical failures"],
        "run_manifest": {"command": [sys.executable, *sys.argv], "environment": sys.prefix, "python": platform.python_version(), "tensorflow": tf.__version__, "host": platform.node(), "cpu_gpu_status": "trusted_gpu_xla_genut_with_cpu_fp64_comparators", "data_version": "repository_frozen_seeded_fixtures_with_tensor_hashes", "random_seeds": [81101, 81104, 81105], "wall_time_seconds": None, "output_json": None, "output_markdown": None, "plan": "docs/plans/bayesfilter-one-seed-four-filter-feasibility-plan-2026-07-22.md", "result": "docs/plans/bayesfilter-one-seed-four-filter-feasibility-result-2026-07-22.md"},
    }
    json_path = output_root / "result.json"
    md_path = output_root / "result.md"
    payload["artifact_json"] = str(json_path.relative_to(ROOT))
    payload["run_manifest"]["wall_time_seconds"] = payload["wall_time_seconds"]
    payload["run_manifest"]["output_json"] = str(json_path.relative_to(ROOT))
    payload["run_manifest"]["output_markdown"] = str(md_path.relative_to(ROOT))
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_render(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output_root), "wall_time_seconds": payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
