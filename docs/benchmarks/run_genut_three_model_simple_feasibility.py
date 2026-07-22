#!/usr/bin/env python3
"""Retired historical runner with an invalid reduced-SIR suite entry.

The reduced SIR phase was an artificial boundary-stress fixture, not an actual
model.  This script is preserved for provenance but must not be relaunched.
Retain the already-tested Chapter 18b structural target in future GenUT
comparison planning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp


PLAN_PATH = "docs/plans/bayesfilter-genut-three-model-simple-feasibility-plan-2026-07-22.md"
ARTIFACT_ROOT = ROOT / "docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt02_n1008"
HORIZON = 10
N_PARTICLES = 1008
CONTROLS = {"epsilon": 2.0, "sinkhorn_steps": 8, "ridge": 1.0e-5}
ROUTE_ID = "cubature_genut_nonfused_positive_ot_candidate_v1"


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tensor_sha256(tensor: tf.Tensor) -> str:
    return hashlib.sha256(tf.io.serialize_tensor(tensor).numpy()).hexdigest()


def _allocator() -> dict[str, int]:
    memory = tf.config.experimental.get_memory_info("GPU:0")
    return {"current_bytes": int(memory["current"]), "peak_bytes": int(memory["peak"])}


def _score_difference(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _comparison(genut: dict[str, object], comparator: dict[str, object]) -> dict[str, object]:
    left = list(genut["score"])
    right = list(comparator["score"])
    if len(left) != len(right):
        raise ValueError("score coordinate count mismatch")
    return {
        "value_difference_genut_minus_comparator": float(genut["value"]) - float(comparator["value"]),
        "score_difference_genut_minus_comparator": _score_difference(left, right),
    }


def _run_genut(
    *,
    adapter,
    theta: tf.Tensor,
    observations: tf.Tensor,
    initial_noise: tf.Tensor,
    process_noise: tf.Tensor,
    design: tf.Tensor,
    fd_steps: tf.Tensor,
) -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    parameter_count = int(theta.shape[0])
    state_dimension = int(initial_noise.shape[1])
    observation_dimension = int(observations.shape[1])

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta_value, observations_value, initial_noise_value, process_noise_value, design_value):
        theta_value = tf.ensure_shape(theta_value, [parameter_count])
        observations_value = tf.ensure_shape(
            observations_value, [HORIZON, observation_dimension]
        )
        initial_noise_value = tf.ensure_shape(
            initial_noise_value, [N_PARTICLES, state_dimension]
        )
        process_noise_value = tf.ensure_shape(
            process_noise_value, [HORIZON, N_PARTICLES, state_dimension]
        )
        design_value = tf.ensure_shape(design_value, [N_PARTICLES, state_dimension])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta_value,
                observations_value,
                initial_noise_value,
                process_noise_value,
                design_value,
                epsilon=float(CONTROLS["epsilon"]),
                sinkhorn_steps=int(CONTROLS["sinkhorn_steps"]),
                ridge=float(CONTROLS["ridge"]),
                transition_before_first_observation=False,
            )

    tf.config.experimental.reset_memory_stats("GPU:0")
    started = time.perf_counter()
    value, score, diagnostics = evaluate(
        theta, observations, initial_noise, process_noise, design
    )
    main_elapsed = time.perf_counter() - started
    analytic = [float(item) for item in score.numpy()]

    fd_started = time.perf_counter()
    fd_score = []
    for index in range(parameter_count):
        direction = tf.one_hot(index, parameter_count, dtype=tf.float32)
        plus, _, _ = evaluate(
            theta + fd_steps[index] * direction,
            observations,
            initial_noise,
            process_noise,
            design,
        )
        minus, _, _ = evaluate(
            theta - fd_steps[index] * direction,
            observations,
            initial_noise,
            process_noise,
            design,
        )
        fd_score.append(float(((plus - minus) / (2.0 * fd_steps[index])).numpy()))
    fd_elapsed = time.perf_counter() - fd_started
    absolute_errors = [abs(a - b) for a, b in zip(analytic, fd_score)]
    scaled_errors = [error / max(1.0, abs(fd)) for error, fd in zip(absolute_errors, fd_score)]
    memory = _allocator()
    finite = bool(tf.math.is_finite(value).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    max_mean = float(diagnostics["max_mean_residual"].numpy())
    max_row = float(diagnostics["max_row_residual"].numpy())
    max_col = float(diagnostics["max_col_residual"].numpy())
    fd_max = max(scaled_errors)
    return {
        "route_id": ROUTE_ID,
        "value": float(value.numpy()),
        "score": analytic,
        "finite": finite,
        "value_increments": [float(item) for item in diagnostics["value_increments"].numpy()],
        "score_increments": [
            [float(item) for item in row] for row in diagnostics["score_increments"].numpy()
        ],
        "max_mean_residual": max_mean,
        "max_row_residual": max_row,
        "max_col_residual": max_col,
        "same_scalar_fd": {
            "role": "diagnostic_only_not_runtime_score",
            "steps": [float(item) for item in fd_steps.numpy()],
            "score": fd_score,
            "absolute_errors": absolute_errors,
            "scaled_errors": scaled_errors,
            "maximum_scaled_error": fd_max,
        },
        "timing_seconds": {
            "compile_and_main_evaluation": main_elapsed,
            "finite_difference_audit": fd_elapsed,
        },
        "allocator": memory,
        "device": str(value.device),
        "jit_compile": True,
        "dtype": "float32",
        "tf32": True,
        "transition_before_first_observation": False,
        "diagnostic_gate": {
            "finite": finite,
            "transport_residual_at_most_1e-4": max(max_mean, max_row, max_col) <= 1.0e-4,
            "same_scalar_fd_scaled_error_at_most_5pct": fd_max <= 0.05,
        },
    }


def _genut_noise(seed: int, state_dimension: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal(
            [N_PARTICLES, state_dimension], [seed, 101], dtype=tf.float32
        ),
        tf.random.stateless_normal(
            [HORIZON, N_PARTICLES, state_dimension], [seed, 102], dtype=tf.float32
        ),
    )


def _run_sir() -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_adapters import reduced_sir_candidate_adapter
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    from bayesfilter.highdim.sir_latent_preclip_reference_tf import (
        dense_latent_sir_value_and_manual_score,
        prepare_reduced_dense_grids,
        reduced_latent_preclip_sir_model,
    )

    seed = 97001
    theta = tf.zeros([3], tf.float32)
    model = reduced_latent_preclip_sir_model()
    simulation = model.simulate_from_standard_normals(
        tf.cast(theta, tf.float64),
        tf.random.stateless_normal([2], [seed, 1], dtype=tf.float64),
        tf.random.stateless_normal(
            [HORIZON - 1, 2], [seed, 2], dtype=tf.float64
        ),
        tf.random.stateless_normal(
            [HORIZON, 1], [seed, 3], dtype=tf.float64
        ),
    )
    observations64 = tf.cast(simulation["observations"], tf.float64)
    observations = tf.cast(observations64, tf.float32)
    padded_observations = tf.concat(
        [tf.zeros_like(observations), observations], axis=1
    )
    initial_noise, process_noise = _genut_noise(seed, 2)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=2), num_particles=N_PARTICLES
    )
    genut = _run_genut(
        adapter=reduced_sir_candidate_adapter(
            transition_before_first_observation=False
        ),
        theta=theta,
        observations=padded_observations,
        initial_noise=initial_noise,
        process_noise=process_noise,
        design=design,
        fd_steps=tf.constant([2.0e-3, 2.0e-3, 2.0e-3], tf.float32),
    )
    comparator_started = time.perf_counter()
    grids = prepare_reduced_dense_grids(
        model,
        tf.cast(theta, tf.float64),
        time_steps=HORIZON - 1,
        order=29,
        radius=7.0,
        integration_rule="split_gauss_legendre",
    )
    dense_result = dense_latent_sir_value_and_manual_score(
        model, tf.cast(theta, tf.float64), observations64, grids
    )
    dense = {
        "value": float(dense_result["objective"].numpy()),
        "score": [float(item) for item in dense_result["score"].numpy()],
        "maximum_boundary_mass": float(
            tf.reduce_max(dense_result["boundary_mass_history"]).numpy()
        ),
        "configuration": {
            "order": 29,
            "radius": 7.0,
            "rule": "split_gauss_legendre",
        },
    }
    dense["elapsed_seconds"] = time.perf_counter() - comparator_started
    dense["route_id"] = "dense_latent_sir_split_gauss_legendre_manual_score"
    dense["role"] = "same_target_accuracy_anchor"
    dense["dtype"] = "float64"
    comparison = _comparison(genut, dense)
    dense_valid = float(dense["maximum_boundary_mass"]) <= 1.0e-8
    phase_pass = all(bool(value) for value in genut["diagnostic_gate"].values()) and dense_valid
    return {
        "phase": 1,
        "model_id": "reduced_continuous_preclip_sir_j1_v1",
        "status": "diagnostic_feasibility_pass" if phase_pass else "diagnostic_feasibility_veto",
        "canonical_leaderboard_row": False,
        "explicit_non_substitution": "not_spatial_sir_austria_j9_T20",
        "seed": seed,
        "horizon": HORIZON,
        "particle_count": N_PARTICLES,
        "parameter_order": [
            "log_kappa_scale",
            "log_nu_scale",
            "log_obs_noise_scale",
        ],
        "theta": [float(item) for item in theta.numpy()],
        "observations_sha256": _tensor_sha256(observations),
        "target_timing": "stationary_initial_draw_then_y0_before_first_transition",
        "genut": genut,
        "comparators": {"dense_manual_score": dense},
        "comparisons": {"dense_manual_score": comparison},
        "phase_gate": {
            "genut_gate_passed": all(
                bool(value) for value in genut["diagnostic_gate"].values()
            ),
            "dense_boundary_mass_at_most_1e-8": dense_valid,
        },
        "nonclaims": [
            "canonical Austria SIR",
            "leaderboard admission",
            "method ranking",
            "default promotion",
        ],
    }


def _generalized_sv_comparator(
    theta: tf.Tensor, observations: tf.Tensor
) -> dict[str, object]:
    import bayesfilter.highdim as highdim
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import (
        _zhao_cui_scalar_tt_config,
    )

    model = highdim.GeneralizedSVPriorMeanSSM()
    derivative = highdim.FixedBranchDerivativeConfig(
        parameter_indices=(0, 1, 2),
        finite_difference_h=(),
        solve_condition_number_veto=1e14,
    )
    started = time.perf_counter()
    result = highdim.scalar_nonlinear_fixed_design_tt_score_path(
        model,
        tf.cast(theta, tf.float64),
        tf.cast(observations, tf.float64),
        _zhao_cui_scalar_tt_config(
            "genut-generalized-sv-t10-analytical-score-diagnostic",
            basis_degree=16,
            quadrature_order=41,
        ),
        derivative,
        fixture_id="genut.generalized-sv-prior-mean.t10.score.v1",
        initial_target_id="genut.generalized-sv-prior-mean.t10.initial.v1",
        transition_target_id="genut.generalized-sv-prior-mean.t10.transition.v1",
        branch_seed_prefix="genut-generalized-sv-prior-mean-t10-score",
        retained_moment_order=65,
        retained_propagation_order=81,
    )
    return {
        "route_id": "zhao_cui_generalized_sv_prior_mean_scalar_fixed_design_tt",
        "role": "same_target_diagnostic_not_oracle",
        "value": float(result.log_likelihood.numpy()),
        "score": [float(item) for item in tf.reshape(result.score, [-1]).numpy()],
        "finite": bool(tf.math.is_finite(result.log_likelihood).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(result.score)).numpy()),
        "elapsed_seconds": time.perf_counter() - started,
        "dtype": "float64",
        "score_backend": "analytical_model_parameter_score_methods_only",
    }


def _run_generalized_sv() -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_adapters import (
        generalized_sv_prior_mean_candidate_adapter,
    )
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _generalized_sv_prior_mean_dataset,
    )

    seed = 81105
    raw = _generalized_sv_prior_mean_dataset(seed)
    theta = tf.constant(raw["truth_theta"], tf.float32)
    observations = tf.cast(raw["observations"][:HORIZON], tf.float32)
    initial_noise, process_noise = _genut_noise(seed, 1)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=1), num_particles=N_PARTICLES
    )
    genut = _run_genut(
        adapter=generalized_sv_prior_mean_candidate_adapter(),
        theta=theta,
        observations=observations,
        initial_noise=initial_noise,
        process_noise=process_noise,
        design=design,
        fd_steps=tf.constant([2.0e-3, 2.0e-3, 2.0e-3], tf.float32),
    )
    comparator = _generalized_sv_comparator(theta, observations)
    comparison = _comparison(genut, comparator)
    phase_pass = all(
        bool(value) for value in genut["diagnostic_gate"].values()
    ) and bool(comparator["finite"])
    return {
        "phase": 2,
        "model_id": "zhao_cui_generalized_sv_synthetic_from_estimated_values",
        "status": "diagnostic_feasibility_pass" if phase_pass else "diagnostic_feasibility_veto",
        "canonical_dataset_prefix": True,
        "seed": seed,
        "horizon": HORIZON,
        "particle_count": N_PARTICLES,
        "parameter_order": ["z_gamma", "log_tau", "mu_over_tau"],
        "theta": [float(item) for item in theta.numpy()],
        "observations_sha256": _tensor_sha256(observations),
        "target_timing": "stationary_initial_draw_then_y0_before_first_transition",
        "genut": genut,
        "comparators": {"zhao_cui_fixed_branch": comparator},
        "comparisons": {"zhao_cui_fixed_branch": comparison},
        "phase_gate": {
            "genut_gate_passed": all(
                bool(value) for value in genut["diagnostic_gate"].values()
            ),
            "comparator_finite": bool(comparator["finite"]),
        },
        "nonclaims": [
            "Zhao-Cui oracle",
            "full T1008 validity",
            "leaderboard admission",
            "method ranking",
            "default promotion",
        ],
    }


def _ksc_comparators(raw_observations: tf.Tensor, theta: tf.Tensor) -> dict[str, dict[str, object]]:
    import bayesfilter.highdim as highdim

    normal = tfp.distributions.Normal(
        loc=tf.constant(0.0, tf.float64), scale=tf.constant(1.0, tf.float64)
    )
    theta64 = tf.cast(theta, tf.float64)
    gamma = normal.cdf(theta64[0])
    beta = tf.exp(theta64[1])
    common = {
        "gamma": gamma,
        "beta": beta,
        "sigma": tf.constant(1.0, tf.float64),
    }
    comparators = {}
    for name, function, route_id in (
        (
            "fixed_sgqf",
            highdim.independent_panel_sv_mixture_fixed_sgqf_score,
            "fixed_sgqf_independent_panel_ksc_mixture",
        ),
        (
            "principal_sqrt_ukf",
            highdim.independent_panel_sv_mixture_ukf_score,
            "principal_sqrt_ukf_independent_panel_ksc_mixture",
        ),
    ):
        started = time.perf_counter()
        keyword_arguments = dict(common)
        if name == "fixed_sgqf":
            keyword_arguments["sparse_level"] = 2
        result = function(
            tf.cast(raw_observations, tf.float64), **keyword_arguments
        )
        if result.score is None or result.log_likelihood is None:
            raise RuntimeError(f"{name} did not emit value and score")
        comparators[name] = {
            "route_id": route_id,
            "role": "same_KSC_surrogate_diagnostic_not_exact_SV_oracle",
            "value": float(result.log_likelihood.numpy()),
            "score": [float(item) for item in result.score.numpy()],
            "finite": bool(tf.math.is_finite(result.log_likelihood).numpy())
            and bool(tf.reduce_all(tf.math.is_finite(result.score)).numpy()),
            "elapsed_seconds": time.perf_counter() - started,
            "dtype": "float64",
            "score_backend": "analytical_recursive_score",
        }
    return comparators


def _run_ksc() -> dict[str, object]:
    from bayesfilter.highdim.cubature_genut_adapters import (
        ksc_mixture_sv_candidate_adapter,
    )
    from bayesfilter.highdim.cubature_genut_candidate import (
        gaussian_genut_design,
        replicate_positive_genut,
    )
    from bayesfilter.highdim.sv_mixture_cut4 import transformed_sv_observations
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset

    seed = 81101
    raw = _sv_dataset(seed)
    theta = tf.constant(raw["truth_theta"], tf.float32)
    raw_observations = tf.cast(raw["observations"][:HORIZON], tf.float32)
    transformed = tf.cast(
        transformed_sv_observations(tf.cast(raw_observations, tf.float64), offset=1.0e-8),
        tf.float32,
    )
    initial_noise, process_noise = _genut_noise(seed, 1)
    design = replicate_positive_genut(
        gaussian_genut_design(dim=1), num_particles=N_PARTICLES
    )
    genut = _run_genut(
        adapter=ksc_mixture_sv_candidate_adapter(),
        theta=theta,
        observations=transformed,
        initial_noise=initial_noise,
        process_noise=process_noise,
        design=design,
        fd_steps=tf.constant([2.0e-3, 2.0e-3], tf.float32),
    )
    comparators = _ksc_comparators(raw_observations, theta)
    comparisons = {
        name: _comparison(genut, comparator)
        for name, comparator in comparators.items()
    }
    comparators_finite = all(bool(item["finite"]) for item in comparators.values())
    phase_pass = all(
        bool(value) for value in genut["diagnostic_gate"].values()
    ) and comparators_finite
    return {
        "phase": 3,
        "model_id": "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
        "status": "diagnostic_feasibility_pass" if phase_pass else "diagnostic_feasibility_veto",
        "canonical_dataset_prefix": True,
        "seed": seed,
        "horizon": HORIZON,
        "particle_count": N_PARTICLES,
        "parameter_order": ["z_gamma", "log_beta"],
        "theta": [float(item) for item in theta.numpy()],
        "raw_observations_sha256": _tensor_sha256(raw_observations),
        "transformed_observations_sha256": _tensor_sha256(transformed),
        "target_timing": "stationary_initial_draw_then_y0_before_first_transition",
        "target_observation_policy": "log_y_squared_plus_1e-8_seven_component_KSC_mixture",
        "genut": genut,
        "comparators": comparators,
        "comparisons": comparisons,
        "phase_gate": {
            "genut_gate_passed": all(
                bool(value) for value in genut["diagnostic_gate"].values()
            ),
            "comparators_finite": comparators_finite,
        },
        "nonclaims": [
            "exact native SV likelihood",
            "KSC importance reweighting",
            "full T1000 validity",
            "leaderboard admission",
            "method ranking",
            "default promotion",
        ],
    }


def _write_phase(name: str, payload: dict[str, object]) -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _phase_failure(index: int, name: str, exc: BaseException) -> dict[str, object]:
    return {
        "phase": index,
        "model_id": name,
        "status": "execution_failure",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def _render_markdown(result: dict[str, object]) -> str:
    lines = [
        "# GenUT Three-Model Simple Feasibility Result",
        "",
        f"Status: `{result['status']}`.",
        "",
        "All differences are descriptive one-seed diagnostics. They do not rank methods.",
        "",
        "| Model | Route | Value | Score | Role/status |",
        "|---|---|---:|---|---|",
    ]
    for phase in result["phases"]:
        if "genut" not in phase:
            lines.append(
                f"| {phase['model_id']} | phase | N/A | N/A | {phase['status']}: {phase.get('error', '')} |"
            )
            continue
        lines.append(
            f"| {phase['model_id']} | GenUT | {phase['genut']['value']:.9g} | "
            f"`{phase['genut']['score']}` | {phase['status']} |"
        )
        for name, comparator in phase["comparators"].items():
            lines.append(
                f"| {phase['model_id']} | {name} | {comparator['value']:.9g} | "
                f"`{comparator['score']}` | {comparator['role']} |"
            )
    lines += [
        "",
        "## Inference status",
        "",
        "| Question | Verdict |",
        "|---|---|",
        f"| Hard veto screen | `{result['hard_veto_screen']}` |",
        "| Statistically supported ranking | None; one seed and one short prefix per model |",
        "| Descriptive-only differences | All GenUT-minus-comparator value and score differences |",
        "| Default readiness | Not evaluated |",
        "| Next evidence needed | Target-specific tuning, particle/seed ladders, uncertainty intervals, then untouched full-horizon runs |",
        "",
        f"JSON: `{ARTIFACT_ROOT / 'result.json'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    raise RuntimeError(
        "retired planning-error runner: reduced SIR is mechanics-only and cannot "
        "occupy an actual-model suite slot; retain the existing, already-tested "
        "Chapter 18b STR-UKF target in future GenUT comparisons"
    )

    # Historical code below is intentionally unreachable and retained only to
    # explain the already-preserved attempt01/attempt02 artifacts.
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    if N_PARTICLES <= 1000:
        raise RuntimeError("GenUT numerical feasibility tests require N > 1000")
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical_gpus = [item.name for item in tf.config.list_logical_devices("GPU")]
    if not logical_gpus:
        raise RuntimeError("GPU is required")

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    campaign_started = time.perf_counter()
    phases = []
    phase_specs: tuple[tuple[str, str, Callable[[], dict[str, object]]], ...] = (
        ("phase1_sir", "reduced_continuous_preclip_sir_j1_v1", _run_sir),
        (
            "phase2_generalized_sv",
            "zhao_cui_generalized_sv_synthetic_from_estimated_values",
            _run_generalized_sv,
        ),
        (
            "phase3_ksc_sv",
            "zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000",
            _run_ksc,
        ),
    )
    for index, (artifact_name, model_id, function) in enumerate(phase_specs, start=1):
        try:
            phase = function()
        except Exception as exc:  # Preserve earlier phases; aggregate the exact failure.
            phase = _phase_failure(index, model_id, exc)
        phases.append(phase)
        _write_phase(artifact_name, phase)

    completed = [phase for phase in phases if "genut" in phase]
    phase_passes = [phase["status"] == "diagnostic_feasibility_pass" for phase in phases]
    if all(phase_passes):
        status = "diagnostic_feasibility_pass_all_three"
        hard_veto = "passed for all three short-prefix phases"
    elif completed:
        status = "diagnostic_feasibility_partial_with_veto_or_failure"
        hard_veto = "at least one phase vetoed or failed; inspect per-phase gate"
    else:
        status = "campaign_execution_failure"
        hard_veto = "no phase completed"
    result = {
        "schema_version": "bayesfilter.genut_three_model_simple_feasibility.v1",
        "campaign_id": "genut-three-model-simple-feasibility-20260722-attempt02-n1008",
        "status": status,
        "hard_veto_screen": hard_veto,
        "research_question": "short-prefix GenUT value/recursive-score feasibility and same-target comparison",
        "phase_order": [item[1] for item in phase_specs],
        "horizon": HORIZON,
        "particle_count": N_PARTICLES,
        "controls": CONTROLS,
        "design": "Gaussian_GenUT_replicated_positive_equal_mass",
        "phases": phases,
        "statistical_interpretation": {
            "hard_veto_evidence": hard_veto,
            "viable_candidates": [
                phase["model_id"]
                for phase in phases
                if phase["status"] == "diagnostic_feasibility_pass"
            ],
            "statistically_supported_ranking": None,
            "descriptive_only": "all observed value, score, and runtime differences",
            "next_evidence": "target-specific tuning and multi-seed/particle ladders before full-horizon untouched runs",
        },
        "device": {
            "logical_gpus": logical_gpus,
            "memory_policy": memory_policy,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "run_manifest": {
            "git_commit": _git_commit(),
            "git_worktree_clean": False,
            "command": "TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_genut_three_model_simple_feasibility.py",
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "hardware": logical_gpus,
            "dtype": "float32 GenUT; float64 diagnostic comparators",
            "tf32": True,
            "xla": True,
            "seeds": [97001, 81105, 81101],
            "wall_time_seconds": time.perf_counter() - campaign_started,
            "plan": PLAN_PATH,
            "result": str(ARTIFACT_ROOT / "result.json"),
        },
        "nonclaims": [
            "statistical superiority",
            "accuracy certification",
            "leaderboard admission",
            "default promotion",
            "HMC readiness",
            "full-horizon validity",
        ],
    }
    result_path = ARTIFACT_ROOT / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ARTIFACT_ROOT / "result.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    manifest = {
        "schema_version": "bayesfilter.genut_three_model_simple_feasibility.manifest.v1",
        "plan": PLAN_PATH,
        "git_commit": result["run_manifest"]["git_commit"],
        "command": result["run_manifest"]["command"],
        "environment": result["run_manifest"]["environment"],
        "gpu": result["device"],
        "seeds": result["run_manifest"]["seeds"],
        "wall_time_seconds": result["run_manifest"]["wall_time_seconds"],
        "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "artifacts": [str(path) for path in sorted(ARTIFACT_ROOT.iterdir())],
    }
    (ARTIFACT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
