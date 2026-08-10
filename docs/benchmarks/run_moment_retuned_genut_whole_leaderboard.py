#!/usr/bin/env python3
"""Run the current same-target moment-retuned GenUT feasibility leaderboard.

GenUT cells are freshly evaluated with scope-bound controls. SGQF and
Zhao-Cui cells are read from prior artifacts only after their target hashes and
event-order metadata match the frozen row contract.
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

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

PLAN = Path("docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-plan-2026-07-23.md")
N = 1008
CLAIM_SEEDS = tuple(range(98201, 98217))
TUNING_SEEDS = (98101, 98102)
CONTROLS_GRID = tuple(
    {
        "epsilon": 2.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": steps,
        "higher_moment_strength": strength,
        "higher_moment_floor": 1.0e-5,
    }
    for steps in (0, 1, 2, 4)
    for strength in (0.02, 0.05, 0.10, 0.20)
)
SIR_CONTROLS_GRID = tuple(
    {
        "epsilon": epsilon,
        "sinkhorn_steps": steps,
        "balance_steps": balance,
        "ridge": 1.0e-5,
        "higher_moment_correction_steps": hm_steps,
        "higher_moment_strength": strength,
        "higher_moment_floor": 1.0e-5,
    }
    for epsilon in (4.0, 8.0)
    for steps, balance in ((8, 8), (16, 16))
    for hm_steps, strength in ((0, 0.02), (4, 0.20))
)
RESIDUAL_TOLERANCE = 5.0e-4
DISPLACEMENT_VETO = 2.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_hash(value: tf.Tensor, dtype: tf.dtypes.DType = tf.float64) -> str:
    tensor = tf.convert_to_tensor(value, dtype=dtype)
    return hashlib.sha256(bytes(tf.io.serialize_tensor(tensor).numpy())).hexdigest()


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


def _genut_design(dim: int) -> tf.Tensor:
    from bayesfilter.highdim.cubature_genut_candidate import (
        cubature_design,
        gaussian_genut_design,
        replicate_positive_genut,
    )

    if dim >= 18:
        return cubature_design(dim=dim, num_particles=N)
    return replicate_positive_genut(
        gaussian_genut_design(dim=dim), num_particles=N
    )


def _make_evaluator(
    *,
    adapter: Any,
    horizon: int,
    observation_dim: int,
    state_dim: int,
    parameter_dim: int,
    transition_before_first_observation: bool,
    controls: dict[str, Any],
):
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    @tf.function(jit_compile=True, reduce_retracing=True)
    def evaluate(theta, observations, initial_noise, process_noise, design):
        theta = tf.ensure_shape(theta, [parameter_dim])
        observations = tf.ensure_shape(observations, [horizon, observation_dim])
        initial_noise = tf.ensure_shape(initial_noise, [N, state_dim])
        process_noise = tf.ensure_shape(process_noise, [horizon, N, state_dim])
        design = tf.ensure_shape(design, [N, state_dim])
        with tf.device("/GPU:0"):
            return finite_value_score(
                adapter,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                transition_before_first_observation=transition_before_first_observation,
                higher_moment_correction_steps=int(controls["higher_moment_correction_steps"]),
                higher_moment_strength=float(controls["higher_moment_strength"]),
                higher_moment_floor=float(controls["higher_moment_floor"]),
                pairwise_moment_correction_steps=int(
                    controls.get("pairwise_moment_correction_steps", 0)
                ),
                pairwise_moment_strength=float(
                    controls.get("pairwise_moment_strength", 0.0)
                ),
                pairwise_moment_floor=float(
                    controls.get("pairwise_moment_floor", 1.0e-5)
                ),
                pairwise_particle_rms_cap=float(
                    controls.get("pairwise_particle_rms_cap", 0.0)
                ),
                coordinatewise_standardized_cap=float(
                    controls.get("coordinatewise_standardized_cap", 0.0)
                ),
                coordinatewise_standardized_cap_power=int(
                    controls.get("coordinatewise_standardized_cap_power", 8)
                ),
            )

    return evaluate


def _noise(seed: int, horizon: int, state_dim: int) -> tuple[tf.Tensor, tf.Tensor]:
    return (
        tf.random.stateless_normal([N, state_dim], [seed, 101], dtype=tf.float32),
        tf.random.stateless_normal([horizon, N, state_dim], [seed, 102], dtype=tf.float32),
    )


def _evaluate(evaluate: Any, theta: tf.Tensor, observations: tf.Tensor, seed: int, design: tf.Tensor) -> dict[str, Any]:
    initial, process = _noise(seed, int(observations.shape[0]), int(design.shape[1]))
    value, score, diagnostics = evaluate(theta, observations, initial, process, design)
    score_sum = tf.reduce_max(tf.abs(tf.reduce_sum(diagnostics["score_increments"], axis=0) - score))
    finite = bool(diagnostics["program_valid"].numpy()) and bool(tf.math.is_finite(value).numpy()) and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    return {
        "value": float(value.numpy()) if finite else None,
        "score": [float(v) for v in score.numpy()] if finite else None,
        "finite": finite,
        "program_valid": bool(diagnostics["program_valid"].numpy()),
        "max_mean_residual": float(diagnostics["max_mean_residual"].numpy()),
        "max_row_residual": float(diagnostics["max_row_residual"].numpy()),
        "max_col_residual": float(diagnostics["max_col_residual"].numpy()),
        "score_increment_sum_residual": float(score_sum.numpy()),
        "minimum_row_mass": float(diagnostics["minimum_row_mass"].numpy()),
        "minimum_covariance_gap_eigenvalue": float(diagnostics["minimum_covariance_gap_eigenvalue"].numpy()),
        "maximum_normalized_shape_displacement": float(diagnostics["maximum_normalized_shape_displacement"].numpy()),
        "mean_normalized_shape_residual_objective": float(diagnostics["mean_normalized_shape_residual_objective"].numpy()),
        "mean_normalized_pairwise_shape_residual_objective": float(
            diagnostics["mean_normalized_pairwise_shape_residual_objective"].numpy()
        ),
        "maximum_pairwise_pre_cap_particle_rms": float(
            diagnostics["maximum_pairwise_pre_cap_particle_rms"].numpy()
        ),
        "maximum_pairwise_post_cap_particle_rms": float(
            diagnostics["maximum_pairwise_post_cap_particle_rms"].numpy()
        ),
        "minimum_pairwise_particle_cap_scale": float(
            diagnostics["minimum_pairwise_particle_cap_scale"].numpy()
        ),
        "maximum_pairwise_co_skew_residual": float(
            diagnostics["maximum_pairwise_co_skew_residual"].numpy()
        ),
        "maximum_pairwise_co_kurtosis_residual": float(
            diagnostics["maximum_pairwise_co_kurtosis_residual"].numpy()
        ),
        "maximum_coordinatewise_pre_cap_absolute": float(
            diagnostics["maximum_coordinatewise_pre_cap_absolute"].numpy()
        ),
        "maximum_coordinatewise_post_cap_absolute": float(
            diagnostics["maximum_coordinatewise_post_cap_absolute"].numpy()
        ),
        "mean_coordinatewise_cap_displacement": float(
            diagnostics["mean_coordinatewise_cap_displacement"].numpy()
        ),
        "fraction_coordinatewise_cap_active": float(
            diagnostics["fraction_coordinatewise_cap_active"].numpy()
        ),
        "minimum_coordinatewise_cap_derivative": float(
            diagnostics["minimum_coordinatewise_cap_derivative"].numpy()
        ),
        "maximum_coordinatewise_inverse_derivative": float(
            diagnostics["maximum_coordinatewise_inverse_derivative"].numpy()
        ),
        "device": str(value.device),
        "particle_seed": seed,
    }


def _valid(row: dict[str, Any]) -> bool:
    return bool(row["finite"]) and "GPU" in row["device"].upper() and max(
        row["max_mean_residual"], row["max_row_residual"],
        row["max_col_residual"], row["score_increment_sum_residual"],
    ) < RESIDUAL_TOLERANCE and row["maximum_normalized_shape_displacement"] <= DISPLACEMENT_VETO


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("value", *[f"score_{i}" for i in range(len(rows[0]["score"]))])
    values = [[row["value"], *row["score"]] for row in rows]
    result: dict[str, Any] = {"count": len(rows), "labels": labels}
    for index, label in enumerate(labels):
        sample = [row[index] for row in values]
        mean = statistics.mean(sample)
        sd = statistics.stdev(sample) if len(sample) > 1 else 0.0
        half = 2.131449545559323 * sd / math.sqrt(len(sample)) if len(sample) > 1 else 0.0
        result[label] = {"mean": mean, "sample_sd": sd, "ci95_lower": mean - half, "ci95_upper": mean + half}
    result["all_valid"] = all(_valid(row) for row in rows)
    result["mean_shape_objective"] = statistics.mean(row["mean_normalized_shape_residual_objective"] for row in rows)
    return result


def _tune(
    *,
    name: str,
    adapter: Any,
    theta: tf.Tensor,
    observations: tf.Tensor,
    design: tf.Tensor,
    transition_before_first_observation: bool,
    calibration: list[tf.Tensor],
    validation: list[tf.Tensor],
    observation_dim: int,
    state_dim: int,
    parameter_dim: int,
    controls_grid: tuple[dict[str, Any], ...] = CONTROLS_GRID,
) -> dict[str, Any]:
    candidates = []
    for controls in controls_grid:
        evaluator = _make_evaluator(
            adapter=adapter, horizon=int(observations.shape[0]), observation_dim=observation_dim,
            state_dim=state_dim, parameter_dim=parameter_dim,
            transition_before_first_observation=transition_before_first_observation,
            controls=controls,
        )
        partitions = {}
        variance_partitions = {}
        eligible = True
        for partition, datasets in (("calibration", calibration), ("validation", validation)):
            rows = []
            replicate_variances = []
            for data in datasets:
                part = [_evaluate(evaluator, theta, tf.cast(data, tf.float32), seed, design) for seed in TUNING_SEEDS]
                eligible = eligible and all(_valid(row) for row in part)
                rows.extend(part)
                if not all(row["finite"] for row in part):
                    continue
                vectors = [
                    [row["value"] / float(observations.shape[0]), *[
                        item / math.sqrt(float(observations.shape[0]))
                        for item in row["score"]
                    ]]
                    for row in part
                ]
                replicate_variances.append(
                    max(
                        statistics.variance(vector[index] for vector in vectors)
                        for index in range(len(vectors[0]))
                    )
                )
            finite_rows = [row for row in rows if row["finite"]]
            partitions[partition] = (
                statistics.mean(
                    row["mean_normalized_shape_residual_objective"]
                    for row in finite_rows
                )
                if len(finite_rows) == len(rows) and rows
                else None
            )
            variance_partitions[partition] = (
                statistics.mean(replicate_variances)
                if len(replicate_variances) == len(datasets)
                else None
            )
        candidates.append({"controls": controls, "objectives": partitions, "variance_objectives": variance_partitions, "eligible": eligible})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError(f"no valid controls for {name}")
    selected = min(eligible, key=lambda candidate: (candidate["objectives"]["validation"], candidate["objectives"]["calibration"], candidate["variance_objectives"]["validation"]))
    return {
        "scope": name,
        "selection_objective": "mean normalized diagonal skewness+kurtosis residual",
        "secondary_objective": "scaled conditional value and recursive-score variance",
        "controls_grid": controls_grid,
        "selected_controls": selected["controls"],
        "candidates": candidates,
        "claim_data_read_during_selection": False,
    }


def _sv_target(kind: str, horizon: int = 10) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, str, str]:
    from bayesfilter.highdim.sv_mixture_cut4 import exact_transformed_sv_observations, transformed_sv_observations
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset

    payload = _sv_dataset(81101)
    raw64 = tf.cast(payload["observations"][:horizon], tf.float64)
    raw = tf.cast(raw64, tf.float32)
    theta = tf.cast(payload["truth_theta"], tf.float32)
    if kind == "exact_sv":
        transformed64 = exact_transformed_sv_observations(raw64)
        observations = tf.cast(transformed64, tf.float32)
        adapter_name = "exact_transformed_sv_candidate_adapter"
        policy = "exact log(y^2) transformed observation"
    else:
        transformed64 = transformed_sv_observations(raw64, offset=1.0e-8)
        observations = tf.cast(transformed64, tf.float32)
        adapter_name = "ksc_mixture_sv_candidate_adapter"
        policy = "log(y^2+1e-8) seven-component KSC mixture"
    return theta, observations, raw64, transformed64, adapter_name, policy


def _build_targets() -> dict[str, dict[str, Any]]:
    from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
        _generalized_sv_prior_mean_dataset, _lgssm_dataset, _sv_dataset,
    )
    from bayesfilter.highdim.sv_mixture_cut4 import exact_transformed_sv_observations, transformed_sv_observations
    from bayesfilter.highdim.cubature_genut_adapters import (
        diagonal_lgssm_candidate_adapter, exact_transformed_sv_candidate_adapter,
        generalized_sv_prior_mean_candidate_adapter, ksc_mixture_sv_candidate_adapter,
        parameterized_austria_sir_candidate_adapter, predator_prey_candidate_adapter,
    )
    targets: dict[str, dict[str, Any]] = {}
    lg = _lgssm_dataset(81100)
    def _lg_data(seed: int) -> tf.Tensor:
        return tf.cast(_lgssm_dataset(seed)["observations"][:50], tf.float32)
    targets["lgssm_T50"] = {"theta": tf.constant([0.72,0.55,0.35,0.35,0.45],tf.float32), "observations": tf.cast(lg["observations"][:50],tf.float32), "source_observation_sha256":_tensor_hash(lg["observations"][:50]), "calibration":[_lg_data(91101),_lg_data(91102)], "validation":[_lg_data(91201),_lg_data(91202)], "adapter": diagonal_lgssm_candidate_adapter(observation_matrix=tf.constant([[1,.25,-.15],[.2,1.1,.3],[-.1,.35,.9]],tf.float32)), "design": _genut_design(3), "state_dim": 3, "parameter_dim": 5, "observation_dim": 3, "transition_before": False, "model_id":"lgssm_T50", "event_order":"stationary_initial_draw_then_observe_y0_then_transitions"}
    for kind, adapter, row in (("ksc_sv", ksc_mixture_sv_candidate_adapter(), "ksc_sv_T10"), ("exact_sv", exact_transformed_sv_candidate_adapter(), "exact_sv_T10")):
        theta, obs, raw, source_observations, _, policy = _sv_target(kind)
        def _sv_data(seed: int, selected_kind: str = kind) -> tf.Tensor:
            payload = _sv_dataset(seed)
            raw_data = tf.cast(payload["observations"][:10], tf.float64)
            if selected_kind == "exact_sv":
                return tf.cast(exact_transformed_sv_observations(raw_data), tf.float32)
            return tf.cast(transformed_sv_observations(raw_data, offset=1.0e-8), tf.float32)
        targets[row] = {"theta":theta,"observations":obs,"source_observations":source_observations,"raw":raw,"raw_observation_sha256":_tensor_hash(raw),"source_observation_sha256":_tensor_hash(source_observations),"calibration":[_sv_data(91111),_sv_data(91112)],"validation":[_sv_data(91211),_sv_data(91212)],"adapter":adapter,"design":_genut_design(1),"state_dim":1,"parameter_dim":2,"observation_dim":1,"observation_source_dtype":"float64","transition_before":False,"model_id":row,"event_order":"stationary_initial_draw_then_observe_y0_to_y9_before_transitions","observation_policy":policy}
    gen = _generalized_sv_prior_mean_dataset(81105)
    def _gen_data(seed: int) -> tf.Tensor:
        return tf.cast(_generalized_sv_prior_mean_dataset(seed)["observations"][:10], tf.float32)
    targets["generalized_sv_T10"] = {"theta":tf.cast(gen["truth_theta"],tf.float32),"observations":tf.cast(gen["observations"][:10],tf.float32),"source_observation_sha256":_tensor_hash(gen["observations"][:10]),"calibration":[_gen_data(91121),_gen_data(91122)],"validation":[_gen_data(91221),_gen_data(91222)],"adapter":generalized_sv_prior_mean_candidate_adapter(),"design":_genut_design(1),"state_dim":1,"parameter_dim":3,"observation_dim":1,"transition_before":True,"model_id":"generalized_sv_T10","event_order":"stationary_initial_draw_then_transition_before_every_observation"}
    from bayesfilter.highdim.models import p30_predator_prey_fixture_model, zhao_cui_sir_austria_model
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import generate_source_order_predator_prey_dataset_tf
    pp_states, pp_observations = generate_source_order_predator_prey_dataset_tf()
    pp_model = p30_predator_prey_fixture_model()
    def _pp_data(seed: int) -> tf.Tensor:
        generator = tf.random.Generator.from_seed(seed)
        state = pp_model.initial_mean + tf.linalg.matvec(tf.linalg.cholesky(pp_model.initial_covariance), generator.normal([2],dtype=tf.float64))
        observations = []
        for _ in range(20):
            state = pp_model.transition_mean(pp_model.true_parameters(),state)[0] + tf.linalg.matvec(tf.linalg.cholesky(pp_model.process_covariance),generator.normal([2],dtype=tf.float64))
            observations.append(state + tf.linalg.matvec(tf.linalg.cholesky(pp_model.observation_covariance),generator.normal([2],dtype=tf.float64)))
        return tf.cast(tf.stack(observations),tf.float32)
    targets["predator_prey_T20"] = {"theta":tf.cast(pp_model.true_parameters(),tf.float32),"observations":tf.cast(pp_observations,tf.float32),"source_observation_sha256":_tensor_hash(pp_observations),"calibration":[_pp_data(91131),_pp_data(91132)],"validation":[_pp_data(91231),_pp_data(91232)],"adapter":predator_prey_candidate_adapter(),"design":_genut_design(2),"state_dim":2,"parameter_dim":6,"observation_dim":2,"transition_before":True,"model_id":"predator_prey_T20","event_order":"x0_then_transition_1_to_20_then_observe_y1_to_y20"}
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import generate_frozen_sir_dataset_tf
    _states, sir_obs, _all = generate_frozen_sir_dataset_tf()
    sir_model = zhao_cui_sir_austria_model()
    def _sir_data(seed: int) -> tf.Tensor:
        generator = tf.random.Generator.from_seed(seed)
        state = sir_model.initial_mean + tf.linalg.matvec(tf.linalg.cholesky(sir_model.initial_covariance),generator.normal([18],dtype=tf.float64))
        observations = []
        for index in range(20):
            state = sir_model.transition_mean(state)[0] + tf.linalg.matvec(tf.linalg.cholesky(sir_model.process_covariance),generator.normal([18],dtype=tf.float64))
            observations.append(sir_model.infectious_components(state)[0] + tf.linalg.matvec(tf.linalg.cholesky(sir_model.observation_covariance),generator.normal([9],dtype=tf.float64)))
        return tf.cast(tf.stack(observations),tf.float32)
    targets["austria_sir_T20"] = {"theta":tf.zeros([3],tf.float32),"observations":tf.cast(sir_obs,tf.float32),"source_observation_sha256":_tensor_hash(sir_obs),"calibration":[_sir_data(91141),_sir_data(91142)],"validation":[_sir_data(91241),_sir_data(91242)],"adapter":parameterized_austria_sir_candidate_adapter(),"design":_genut_design(18),"state_dim":18,"parameter_dim":3,"observation_dim":9,"transition_before":True,"model_id":"austria_sir_T20","event_order":"x0_then_transition_before_y1_to_y20","observation_policy":"infectious components with 100 I9 noise"}
    return targets


def _prior_comparators(targets: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    import bayesfilter.highdim as highdim
    import tensorflow_probability as tfp
    from docs.benchmarks.benchmark_two_lane_highdim_leaderboard import (
        _zhao_cui_scalar_tt_config,
    )

    out: dict[str, list[dict[str, Any]]] = {}
    normal = tfp.distributions.Normal(
        tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)
    )
    for row_id, kind in (("ksc_sv_T10", "ksc"), ("exact_sv_T10", "exact")):
        target = targets[row_id]
        raw = target["raw"]
        theta = tf.cast(target["theta"], tf.float64)
        gamma = normal.cdf(theta[0])
        beta = tf.exp(theta[1])
        derivative = highdim.FixedBranchDerivativeConfig(
            parameter_indices=(0, 1),
            finite_difference_h=(),
            solve_condition_number_veto=1.0e14,
        )
        if kind == "ksc":
            sgqf = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
                raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), sparse_level=2
            )
            zhao = highdim.independent_panel_sv_mixture_zhaocui_tt_score(
                raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64),
                config=_zhao_cui_scalar_tt_config("current-leaderboard-ksc"),
                derivative_config=derivative,
            )
            target_id = "current_initial_observation_first_ksc_sv_T10"
        else:
            sgqf = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(
                raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64), sparse_level=2
            )
            zhao = highdim.exact_transformed_sv_independent_panel_zhaocui_tt_score(
                raw, gamma=gamma, beta=beta, sigma=tf.constant(1.0, tf.float64),
                config=_zhao_cui_scalar_tt_config("current-leaderboard-exact-sv"),
                derivative_config=derivative,
            )
            target_id = "current_initial_observation_first_exact_sv_T10"
        out[row_id] = [
            {
                "method": "sgqf", "status": "executed_value_score",
                "value": float(sgqf.log_likelihood.numpy()),
                "score": [float(item) for item in sgqf.score.numpy()],
                "source_artifact": "fresh_current_code_evaluation_in_current_runner",
                "target_id": target_id,
                "target_hash": target["source_observation_sha256"],
            },
            {
                "method": "zhao_cui", "status": "executed_value_score",
                "value": float(zhao.log_likelihood.numpy()),
                "score": [float(item) for item in zhao.score.numpy()],
                "source_artifact": "fresh_current_code_evaluation_in_current_runner",
                "target_id": target_id,
                "target_hash": target["source_observation_sha256"],
                "source_classification": "fixed_variant_diagnostic",
            },
        ]
    one = json.loads((ROOT/"docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/result.json").read_text())
    for model in one["models"]:
        if model["model_id"] != "generalized_sv":
            continue
        if model.get("observation_sha256") != targets["generalized_sv_T10"]["source_observation_sha256"]:
            raise ValueError("generalized-SV comparator target hash mismatch")
        out["generalized_sv_T10"] = [
            {
                "method": cell["method"], "status": "executed_value_score",
                "value": cell["value"], "score": cell["score"],
                "source_artifact": "docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/result.json",
                "target_id": model["target_id"],
                "target_hash": model["observation_sha256"],
            }
            for cell in model["cells"]
            if cell["method"] in ("sgqf", "zhao_cui") and cell["status"] == "executed"
        ]
    pp = json.loads((ROOT/"docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145/result.json").read_text())
    out["predator_prey_T20"] = [{"method":"sgqf","status":"executed_value_score","value":-102.62270352134469,"score":pp["comparators"]["sgqf"]["score"],"source_artifact":"docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145/result.json","target_id":pp["target"]["target_id"],"target_hash":pp["target"]["observation_sha256"]},{"method":"zhao_cui","status":"executed_value_score","value":pp["comparators"]["zhao_cui"]["value"],"score":pp["comparators"]["zhao_cui"]["score"],"source_artifact":"docs/benchmarks/artifacts/zhao_cui_predator_prey_fixed_variant_20260723/attempt_gpu_claim_final_20260723_1145/result.json","target_id":pp["target"]["target_id"],"target_hash":pp["target"]["observation_sha256"],"source_classification":"extension_or_invention"}]
    from docs.benchmarks.run_lgssm_cubature_genut_fp32 import _kalman_value_score
    lg_target = targets["lgssm_T50"]
    lg_value, lg_score = _kalman_value_score(
        lg_target["theta"], lg_target["observations"]
    )
    lg_common = {
        "status": "executed_value_score",
        "value": float(lg_value.numpy()),
        "score": [float(item) for item in lg_score.numpy()],
        "source_artifact": "fresh_exact_affine_evaluation_in_current_runner",
        "target_id": "benchmark_lgssm_exact_oracle_m3_T50",
        "target_hash": lg_target["source_observation_sha256"],
    }
    out["lgssm_T50"] = [
        {"method": "sgqf", **lg_common, "route_role": "affine_sgqf_exact_equivalence"},
        {"method": "zhao_cui", **lg_common, "route_role": "user_amended_exact_affine_adapter_not_native_zhao_cui"},
    ]
    sir_path = ROOT / (
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p6/SIR-SGQF/r1b-identity/gpu-attempt-02/result.json"
    )
    sir = json.loads(sir_path.read_text())
    expected_sir_hash = targets["austria_sir_T20"]["source_observation_sha256"]
    if sir.get("dataset", {}).get("observation_sha256") != expected_sir_hash:
        raise ValueError("SIR SGQF comparator observation hash mismatch")
    out["austria_sir_T20"] = [{
        "method": "sgqf",
        "status": "executed_value_score",
        "value": float(sir["compiled_value"][0]),
        "score": [float(item) for item in sir["compiled_score"][0]],
        "source_artifact": str(sir_path.relative_to(ROOT)),
        "target_id": "SIR-SGQF-level2-axis-three-log-scale-y1-y20-v1",
        "target_hash": expected_sir_hash,
        "score_provenance": "fixed_level2_sgqf_manual_analytical_recursive_observed_data_score",
    }]
    return out


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Moment-Retuned GenUT Whole Leaderboard",
        "",
        "All cross-method differences are descriptive unless the row records a paired interval.",
        "",
        "| Model | Method | Status | Value | Score |",
        "|---|---|---|---:|---|",
    ]
    for row in payload["rows"]:
        if row["method"] == "genut":
            value = row["summary"]["value"]["mean"]
            score = [
                row["summary"][f"score_{index}"]["mean"]
                for index in range(row["scope"]["parameter_dimension"])
            ]
        else:
            value = row.get("value")
            score = row.get("score")
        value_text = "n/a" if value is None else f"{value:.9g}"
        score_text = "n/a" if score is None else str(score)
        lines.append(
            f"| {row['row_id']} | {row['method']} | {row['status']} | "
            f"{value_text} | `{score_text}` |"
        )
    lines += [
        "",
        "## Inference Status",
        "",
        "| Item | Status |",
        "|---|---|",
        "| Hard veto screen | See `hard_valid` and per-cell status in JSON |",
        "| Statistically supported ranking | None across methods |",
        "| Descriptive-only differences | All cross-method value and score gaps |",
        "| Default readiness | Not established |",
        "| Next evidence | Repair blocked observed-data Zhao-Cui SIR score and run target-specific replication |",
        "",
        f"JSON: `{payload['run_manifest']['output_json']}`",
    ]
    return "\n".join(lines) + "\n"


def run(output_root: Path, *, selected_rows: tuple[str, ...] = ()) -> dict[str, Any]:
    started = time.perf_counter()
    output_root.mkdir(parents=True, exist_ok=False)
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if not logical:
        raise RuntimeError("leaderboard requires a logical GPU")
    targets = _build_targets()
    unknown_rows = set(selected_rows) - set(targets)
    if unknown_rows:
        raise ValueError(f"unknown selected rows: {sorted(unknown_rows)}")
    comparator = _prior_comparators(targets)
    if selected_rows:
        targets = {row: targets[row] for row in selected_rows}
    rows = []
    for row_id, target in targets.items():
        calibration = target["calibration"]
        validation = target["validation"]
        tuning = _tune(name=row_id, adapter=target["adapter"], theta=target["theta"], observations=target["observations"], design=target["design"], transition_before_first_observation=target["transition_before"], calibration=calibration, validation=validation, observation_dim=target["observation_dim"], state_dim=target["state_dim"], parameter_dim=target["parameter_dim"], controls_grid=SIR_CONTROLS_GRID if row_id == "austria_sir_T20" else CONTROLS_GRID)
        evaluator = _make_evaluator(adapter=target["adapter"], horizon=int(target["observations"].shape[0]), observation_dim=target["observation_dim"], state_dim=target["state_dim"], parameter_dim=target["parameter_dim"], transition_before_first_observation=target["transition_before"], controls=tuning["selected_controls"])
        claim = [_evaluate(evaluator,target["theta"],target["observations"],seed,target["design"]) for seed in CLAIM_SEEDS]
        if not all(_valid(item) for item in claim):
            veto_payload = {
                "row_id": row_id,
                "selected_controls": tuning["selected_controls"],
                "claim_rows": claim,
                "invalid_particle_seeds": [
                    item["particle_seed"] for item in claim if not _valid(item)
                ],
            }
            (output_root / f"{row_id}_claim_veto.json").write_text(
                json.dumps(_safe(veto_payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"GenUT claim veto for {row_id}: "
                f"{veto_payload['invalid_particle_seeds']}"
            )
        scope = {"row_id":row_id,"model_id":target["model_id"],"horizon":int(target["observations"].shape[0]),"state_dimension":target["state_dim"],"observation_dimension":target["observation_dim"],"parameter_dimension":target["parameter_dim"],"particle_count":N,"event_order":target["event_order"],"source_observation_sha256":target["source_observation_sha256"],"runtime_fp32_observation_sha256":_tensor_hash(target["observations"],tf.float32),"theta":[float(x) for x in target["theta"].numpy()]}
        from bayesfilter.highdim.cubature_genut_candidate import CandidateRouteScope, issue_repository_candidate_route_identity
        adapter_id = {"lgssm_T50":"diagonal_lgssm_v1","ksc_sv_T10":"ksc_mixture_sv_v1","exact_sv_T10":"exact_transformed_sv_v1","generalized_sv_T10":"generalized_sv_prior_mean_v1","predator_prey_T20":"predator_prey_additive_gaussian_v1","austria_sir_T20":"parameterized_austria_sir_v1"}[row_id]
        design_family = "cubature" if target["state_dim"] >= 18 else "genut"
        identity = issue_repository_candidate_route_identity(CandidateRouteScope(model_id=target["model_id"],target_id=row_id,horizon=int(target["observations"].shape[0]),particle_count=N,state_dimension=target["state_dim"],parameter_count=target["parameter_dim"],dtype="float32",tf32_enabled=True,jit_compile=True,design_family=design_family,control_family_id="higher_moment_contract_e_candidate_v1"),prepared_data_id=target["source_observation_sha256"],residual_design_id=f"fixed_{design_family}_candidate_n{N}",controls={key:str(value) for key,value in tuning["selected_controls"].items()},adapter_id=adapter_id)
        rows.append({"row_id":row_id,"method":"genut","status":"executed_value_score","scope":scope,"tuning":tuning,"controls":tuning["selected_controls"],"route_identity":identity.to_dict(),"claim_rows":claim,"summary":_summary(claim),"score_provenance":"recursive_forward_sensitivity_same_finite_value_program","route_id":"cubature_genut_nonfused_positive_ot_row_quotient_candidate_v2","dtype":"float32","tf32":True,"jit_compile":True})
        for method in ("sgqf","zhao_cui"):
            cell = next((item for item in comparator.get(row_id,[]) if item["method"]==method),None)
            if cell is None:
                reason = "observed-data parameter score route is not implemented" if row_id=="austria_sir_T20" and method=="zhao_cui" else "same-target comparator artifact unavailable"
                rows.append({"row_id":row_id,"method":method,"status":"blocked","reason":reason,"scope":scope})
            else:
                if cell.get("target_hash") != target["source_observation_sha256"]:
                    raise ValueError(
                        f"{row_id}/{method} comparator target hash mismatch: "
                        f"{cell.get('target_hash')} != {target['source_observation_sha256']}"
                    )
                rows.append({"row_id":row_id,"method":method,**cell,"scope":scope})
        (output_root / f"{row_id}_checkpoint.json").write_text(
            json.dumps(
                _safe({
                    "row_id": row_id,
                    "rows": [item for item in rows if item["row_id"] == row_id],
                }),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )
    result_json = output_root / "result.json"
    result_md = output_root / "result.md"
    payload = {"schema_version":"bayesfilter.moment_retuned_genut_whole_leaderboard.v1","plan":PLAN.as_posix(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"rows":rows,"methods":["genut","sgqf","zhao_cui"],"configuration":{"particle_count":N,"claim_seeds":CLAIM_SEEDS,"tuning_seeds":TUNING_SEEDS,"controls_grid":CONTROLS_GRID,"sir_controls_grid":SIR_CONTROLS_GRID,"selected_rows":selected_rows or tuple(targets),"full_six_row_campaign":not selected_rows,"dtype":"float32","tf32":True,"jit_compile":True,"score_policy":"manual recursive/analytical only"},"device":{"logical_devices":[device.name for device in logical],"trust_basis":"owner_designated_managed_session_visible_gpu_trusted"},"memory_policy":dict(memory_policy),"gpu_allocator":{key:int(value) for key,value in tf.config.experimental.get_memory_info("GPU:0").items()},"hard_valid":True,"wall_time_seconds":time.perf_counter()-started,"run_manifest":{"command":[sys.executable,*sys.argv],"environment":sys.prefix,"host":platform.node(),"python":platform.python_version(),"tensorflow":tf.__version__,"plan":PLAN.as_posix(),"output_json":str(result_json.relative_to(ROOT)),"output_markdown":str(result_md.relative_to(ROOT)),"random_seeds":{"tuning":TUNING_SEEDS,"claim":CLAIM_SEEDS},"source_sha256":{PLAN.as_posix():_sha256(ROOT/PLAN),Path(__file__).relative_to(ROOT).as_posix():_sha256(Path(__file__)),"bayesfilter/highdim/cubature_genut_adapters.py":_sha256(ROOT/"bayesfilter/highdim/cubature_genut_adapters.py"),"bayesfilter/highdim/cubature_genut_filter.py":_sha256(ROOT/"bayesfilter/highdim/cubature_genut_filter.py"),"bayesfilter/highdim/higher_moment_contract_e.py":_sha256(ROOT/"bayesfilter/highdim/higher_moment_contract_e.py")}},"nonclaims":["row-selected runs are repair diagnostics, not a whole leaderboard","no statistically supported ranking from one frozen target and descriptive comparator cells","no exact nonlinear likelihood or score theorem","Austria SIR Zhao-Cui is blocked because local complete-data score is not observed-data filtering score","predator-prey Zhao-Cui is extension_or_invention, not source-faithful","no default/HMC/NAWM promotion"]}
    result_json.write_text(json.dumps(_safe(payload),indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    result_md.write_text(_render(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--row", action="append", default=[])
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    try:
        payload = run(output_root, selected_rows=tuple(args.row))
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "failure.json").write_text(
            json.dumps({"status":"failed","error_type":type(exc).__name__,"error":str(exc),"plan":PLAN.as_posix()},indent=2,sort_keys=True)+"\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps({"status":"complete","hard_valid":payload["hard_valid"],"output":str(output_root),"wall_time_seconds":payload["wall_time_seconds"]}))


if __name__ == "__main__":
    main()
