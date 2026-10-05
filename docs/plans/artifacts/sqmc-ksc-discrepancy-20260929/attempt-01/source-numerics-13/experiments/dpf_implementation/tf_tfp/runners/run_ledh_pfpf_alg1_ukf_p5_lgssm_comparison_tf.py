"""Run P5 exact-LGSSM diagnostics for Algorithm 1 UKF LEDH-PFPF."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import statistics
import sys
import time
from collections.abc import Callable
from typing import Any

import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import (
    run_bootstrap_particle_filter_tf,
)
from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    METHOD_GENERATION,
    algorithm1_route_identifiers,
    run_ledh_pfpf_alg1_ukf_tf,
    validate_algorithm1_route_identifiers,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


DTYPE = tf.float64
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_ledh_pfpf_alg1_ukf_p5_lgssm_comparison_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-li-coates-alg1-ukf-source-faithful-p5-rerun-comparison-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-li-coates-alg1-ukf-source-faithful-p5-rerun-comparison-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-li-coates-alg1-ukf-source-faithful-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_p5_lgssm_comparison_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-alg1-ukf-p5-lgssm-comparison-2026-06-10.md"

VALUE_SEEDS = [101, 202, 303]
VALUE_PARTICLE_COUNTS = [8, 16]
GRADIENT_SEED = 101
GRADIENT_NUM_PARTICLES = 4
GRADIENT_HORIZON = 3
PSEUDO_TIME_STEPS = [0.5, 0.5]
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0
COVARIANCE_FLOOR = 1e-10
RANK_TOLERANCE = 1e-12
FD_STEP = 1e-5
METHOD_IDS = (
    "bootstrap_pf_no_resampling_tf",
    "ledh_pfpf_alg1_ukf_no_resampling_tf",
)


class P5ValidationError(ValueError):
    """Raised when the P5 payload violates the comparison contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P5_LEDHPFPF_ALG1_UKF_LGSSM_VALIDATED")
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    markdown = _markdown(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    base = _base_lgssm()
    theta0 = tf.constant([1.0, 1.0], dtype=DTYPE)
    kalman = _kalman_reference(theta0, base)
    fd_score = _finite_difference_score(lambda x: _kalman_value(x, base), theta0)
    fd_delta = tf.constant(kalman["score"], dtype=DTYPE) - fd_score
    route = algorithm1_route_identifiers(resampling_route="none")
    validate_algorithm1_route_identifiers(route)

    value_rows = []
    for num_particles in VALUE_PARTICLE_COUNTS:
        for seed in VALUE_SEEDS:
            for method_id in METHOD_IDS:
                value_rows.append(
                    _value_row(
                        method_id=method_id,
                        theta=theta0,
                        base=base,
                        seed=seed,
                        num_particles=num_particles,
                        kalman=kalman,
                    )
                )

    value_summaries = {
        method_id: _value_summary(method_id, value_rows)
        for method_id in METHOD_IDS
    }
    gradient_smoke = _gradient_smoke(theta0, base)
    applicability = _applicability_matrix()
    veto = _veto_diagnostics(kalman, fd_delta, value_rows, value_summaries, gradient_smoke, applicability)
    decision = (
        "PASS_P5_LGSSM_ALG1_UKF_BOUNDED_DIAGNOSTICS_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P5_LGSSM_ALG1_UKF_BOUNDED_DIAGNOSTICS_VETO_PENDING_REVIEW"
    )
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m " + MODULE_PATH,
        pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
    )
    manifest.update(
        {
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "review_ledger_path": REVIEW_LEDGER_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "value_seed_list": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "gradient_smoke_seed": GRADIENT_SEED,
            "gradient_smoke_particle_count": GRADIENT_NUM_PARTICLES,
            "gradient_smoke_horizon": GRADIENT_HORIZON,
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "route_identifiers": route,
            "data_version": "fixed local LGSSM fixture generated by P5 runner",
        }
    )
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P5",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "comparison_scope": {
            "scope_id": "bounded_initial_p5_lgssm_lane",
            "value_lane": (
                "full 25-observation LGSSM fixture, 3 seeds, particle counts 8/16, "
                "no AD gradient tape"
            ),
            "gradient_lane": (
                "short 3-observation fixed-branch smoke for Algorithm 1 only, "
                "seed 101, 4 particles"
            ),
            "full_lgssm_gradient_ladder_deferred": True,
            "deferred_seed_list": [101, 202, 303, 404, 505],
            "deferred_particle_counts": [32, 64, 128],
        },
        "evidence_contract": {
            "baseline_comparator": (
                "tf_qr_sqrt_differentiated_kalman exact LGSSM value and analytic score"
            ),
            "primary_criterion": (
                "finite full-horizon value rows with Monte Carlo uncertainty, "
                "finite short-horizon fixed-branch gradient smoke, and source route identifiers"
            ),
            "veto_diagnostics": list(veto.keys()),
            "not_concluded": _nonclaims(),
        },
        "target": {
            "target_id": "p5_lgssm_2d_h25_rich",
            "model_definition": _model_definition(base),
            "theta": tensor_to_json(theta0),
            "gradient_parameterization": {
                "transition_matrix_scale": "physical scalar multiplying A",
                "observation_noise_scale": "physical scalar multiplying R",
            },
        },
        "kalman_reference": {
            **kalman,
            "finite_difference_score": tensor_to_json(fd_score),
            "max_abs_score_delta_vs_finite_difference": scalar(tf.reduce_max(tf.abs(fd_delta))),
            "finite_difference_policy": "diagnostic_only_not_a_promotion_gate",
            "score_status": "analytic_qr_kalman_score_hessian_reference",
        },
        "routes": {
            "kalman_exact": {
                "route_id": "tf_qr_sqrt_differentiated_kalman",
                "claim_class": "exact_lgssm_oracle",
            },
            "bootstrap_pf_no_resampling_tf": {
                "route_id": "bootstrap_pf_no_resampling_tf",
                "claim_class": "naive_stochastic_value_baseline",
                "resampling_route": "none",
            },
            "ledh_pfpf_alg1_ukf_no_resampling_tf": {
                "route_id": "ledh_pfpf_alg1_ukf_no_resampling_tf",
                "claim_class": "source_faithful_algorithm1_core_no_resampling",
                "route_identifiers": route,
            },
            "previous_ledh_pfpf_ot": {
                "route_id": "dpf_ledh_pfpf_ot",
                "status": "quarantined_historical_only",
                "used_as_evidence": False,
            },
        },
        "stochastic_contract": {
            "value_seeds": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "confidence_interval": "normal 95% CI over seeds by method/particle count",
            "gradient_object": "short_horizon_fixed_branch_smoke",
            "stochastic_score_claim": "not_claimed",
            "resampling_policy": "none for this first P5 source-faithful evidence lane",
        },
        "algorithm1_parameters": {
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
            "resampling_route": "none",
            "method_generation": METHOD_GENERATION,
        },
        "applicability_matrix": applicability,
        "value_rows": value_rows,
        "value_summaries": value_summaries,
        "gradient_smoke": gradient_smoke,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "value_agreement_is_not_gradient_promotion": True,
            "finite_difference_is_diagnostic_only": True,
            "old_ledh_pfpf_ot_used_as_oracle": False,
            "old_ledh_pfpf_ot_used_as_algorithm1_evidence": False,
            "broader_p44_matrix_deferred_pending_adapters": True,
            "full_lgssm_gradient_ladder_deferred": True,
            "monte_carlo_value_uncertainty_reported": True,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _value_row(
    *,
    method_id: str,
    theta: tf.Tensor,
    base: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
    kalman: dict[str, Any],
) -> dict[str, Any]:
    value, diagnostics = _method_value(
        method_id=method_id,
        theta=theta,
        base=base,
        seed=seed,
        num_particles=num_particles,
    )
    value_error = value - tf.constant(kalman["log_likelihood"], dtype=DTYPE)
    return {
        "method_id": method_id,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "value": scalar(value),
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(value_error / tf.constant(25.0, dtype=DTYPE)),
        "finite": bool(tf.math.is_finite(value).numpy()),
        "gradient_object": "not_evaluated_in_value_lane",
        "stochastic_score_claim": "not_claimed",
        "diagnostics": diagnostics,
    }


def _method_value(
    *,
    method_id: str,
    theta: tf.Tensor,
    base: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    params = _parameters(theta, base)
    return _run_method_with_params(
        method_id=method_id,
        params=params,
        seed=seed,
        num_particles=num_particles,
    )


def _run_method_with_params(
    *,
    method_id: str,
    params: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    if method_id == "bootstrap_pf_no_resampling_tf":
        result = run_bootstrap_particle_filter_tf(
            observations=params["observations"],
            initial_sample=lambda n, s: _sample_initial(params, n, s),
            transition_sample=lambda particles, s, t: _transition_sample(params, particles, s, t),
            observation_log_density=lambda particles, observation, _t: _observation_log_density(
                params,
                particles,
                observation,
            ),
            seed=seed,
            num_particles=num_particles,
            ess_threshold_ratio=-1.0,
            method_id=method_id,
        )
        return result.log_likelihood_estimate, _bootstrap_diagnostics(result)
    if method_id == "ledh_pfpf_alg1_ukf_no_resampling_tf":
        route = algorithm1_route_identifiers(resampling_route="none")
        validate_algorithm1_route_identifiers(route)
        result = run_ledh_pfpf_alg1_ukf_tf(
            observations=params["observations"],
            initial_sample=lambda n, s: _sample_initial(params, n, s),
            initial_covariance=params["P0"],
            transition_sample=lambda ancestors, s, t: _transition_sample(params, ancestors, s, t),
            transition_mean_fn=lambda points, _t: tf.linalg.matmul(
                tf.cast(points, DTYPE),
                params["A"],
                transpose_b=True,
            ),
            transition_log_density_fn=lambda x_next, x_prev, _t: _transition_log_density(
                params,
                x_prev,
                x_next,
            ),
            observation_mean_fn=lambda points, _t: tf.linalg.matmul(
                tf.cast(points, DTYPE),
                params["C"],
                transpose_b=True,
            ),
            observation_jacobian_fn=lambda _point, _t: params["C"],
            observation_log_density_fn=lambda particles, observation, _t: _observation_log_density(
                params,
                particles,
                observation,
            ),
            process_noise_covariance_fn=lambda _x_prev, _t: params["Q"],
            observation_covariance_fn=lambda _t: params["R"],
            seed=seed,
            num_particles=num_particles,
            pseudo_time_steps=tf.constant(PSEUDO_TIME_STEPS, dtype=DTYPE),
            resampling_route="none",
            alpha=UKF_ALPHA,
            beta=UKF_BETA,
            kappa=UKF_KAPPA,
            covariance_floor=COVARIANCE_FLOOR,
            rank_tolerance=RANK_TOLERANCE,
            method_id=method_id,
        )
        return result.log_likelihood_estimate, _alg1_diagnostics(result)
    raise ValueError(f"unknown P5 method_id: {method_id}")


def _gradient_smoke(theta: tf.Tensor, base: dict[str, tf.Tensor]) -> dict[str, Any]:
    short_base = {
        **base,
        "observations": base["observations"][:GRADIENT_HORIZON],
    }
    kalman = _kalman_reference(theta, short_base)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        params = _parameters(theta, short_base)
        value, diagnostics = _run_method_with_params(
            method_id="ledh_pfpf_alg1_ukf_no_resampling_tf",
            params=params,
            seed=GRADIENT_SEED,
            num_particles=GRADIENT_NUM_PARTICLES,
        )
    gradient = tape.gradient(value, theta)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
    reference = tf.constant(kalman["score"], dtype=DTYPE)
    error = tf.cast(gradient, DTYPE) - reference
    value_error = value - tf.constant(kalman["log_likelihood"], dtype=DTYPE)
    finite = bool(
        tf.math.is_finite(value).numpy()
        and tf.reduce_all(tf.math.is_finite(gradient)).numpy()
    )
    return {
        "method_id": "ledh_pfpf_alg1_ukf_no_resampling_tf",
        "seed": GRADIENT_SEED,
        "num_particles": GRADIENT_NUM_PARTICLES,
        "horizon": GRADIENT_HORIZON,
        "value": scalar(value),
        "kalman_value": kalman["log_likelihood"],
        "value_error": scalar(value_error),
        "fixed_branch_gradient": tensor_to_json(gradient),
        "kalman_score": kalman["score"],
        "gradient_error": tensor_to_json(error),
        "gradient_error_norm": scalar(tf.linalg.norm(error)),
        "relative_score_error": scalar(
            tf.linalg.norm(error)
            / tf.maximum(tf.constant(1.0, dtype=DTYPE), tf.linalg.norm(reference))
        ),
        "finite": finite,
        "gradient_object": "short_horizon_fixed_branch_smoke",
        "stochastic_score_claim": "not_claimed",
        "diagnostics": diagnostics,
    }


def _bootstrap_diagnostics(result: Any) -> dict[str, Any]:
    ess_values = [float(v) for v in tensor_to_json(result.ess_by_time)]
    return {
        "route_id": result.method_id,
        "resampling_route": "none",
        "resampling_count": int(result.resampling_count),
        "ess_min": min(ess_values),
        "ess_mean": statistics.fmean(ess_values),
        "finite": bool(result.finite),
        "old_ledh_pfpf_ot_used": False,
    }


def _alg1_diagnostics(result: Any) -> dict[str, Any]:
    validate_algorithm1_route_identifiers(result.route_identifiers)
    if any("ledh_pfpf_ot" in str(value) for value in result.route_identifiers.values()):
        raise P5ValidationError("old LEDH-PFPF-OT route appeared in Algorithm 1 diagnostics")
    ess_values = [float(v) for v in tensor_to_json(result.ess_by_time)]
    step_diagnostics = result.resampling_diagnostics
    return {
        "route_id": result.method_id,
        "route_identifiers": dict(result.route_identifiers),
        "resampling_route": result.route_identifiers["resampling_route"],
        "resampling_count": int(result.resampling_count),
        "ess_min": min(ess_values),
        "ess_mean": statistics.fmean(ess_values),
        "finite": bool(result.finite),
        "pseudo_time_step_count": len(PSEUDO_TIME_STEPS),
        "pseudo_time_sum_min": min(float(diag["pseudo_time_sum"]) for diag in step_diagnostics),
        "pseudo_time_sum_max": max(float(diag["pseudo_time_sum"]) for diag in step_diagnostics),
        "min_predicted_covariance_eigenvalue": min(
            float(diag["min_predicted_covariance_eigenvalue"]) for diag in step_diagnostics
        ),
        "max_prediction_floor_count": max(
            int(diag["max_prediction_floor_count"]) for diag in step_diagnostics
        ),
        "min_forward_log_det": min(float(diag["min_forward_log_det"]) for diag in step_diagnostics),
        "max_forward_log_det": max(float(diag["max_forward_log_det"]) for diag in step_diagnostics),
        "finite_forward_log_det": all(bool(diag["finite_forward_log_det"]) for diag in step_diagnostics),
        "finite_updated_covariances": all(bool(diag["finite_updated_covariances"]) for diag in step_diagnostics),
        "old_ledh_pfpf_ot_used": False,
    }


def _value_summary(method_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for num_particles in VALUE_PARTICLE_COUNTS:
        subset = [
            row
            for row in rows
            if row["method_id"] == method_id and row["num_particles"] == num_particles
        ]
        value_errors = [float(row["value_error"]) for row in subset]
        out[str(num_particles)] = {
            "seed_count": len(subset),
            "mean_value_error": statistics.fmean(value_errors),
            "value_error_standard_error": _standard_error(value_errors),
            "value_error_ci95": _ci95(value_errors),
            "value_error_rmse": _rmse(value_errors),
            "max_abs_value_error": max(abs(value) for value in value_errors),
            "min_ess": min(float(row["diagnostics"]["ess_min"]) for row in subset),
            "mean_resampling_count": statistics.fmean(
                float(row["diagnostics"]["resampling_count"]) for row in subset
            ),
            "all_rows_finite": all(bool(row["finite"]) for row in subset),
            "diagnostic_ranges": _diagnostic_ranges(method_id, subset),
        }
    counts = [str(count) for count in VALUE_PARTICLE_COUNTS]
    out["particle_ladder"] = {
        "value_rmse_reduction_ratio": _safe_ratio(
            out[counts[-1]]["value_error_rmse"],
            out[counts[0]]["value_error_rmse"],
        ),
        "trend_interpretation": "diagnostic_only_not_convergence_proof",
    }
    return out


def _diagnostic_ranges(method_id: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
    if method_id != "ledh_pfpf_alg1_ukf_no_resampling_tf":
        return {}
    return {
        "min_forward_log_det": min(float(row["diagnostics"]["min_forward_log_det"]) for row in subset),
        "max_forward_log_det": max(float(row["diagnostics"]["max_forward_log_det"]) for row in subset),
        "min_predicted_covariance_eigenvalue": min(
            float(row["diagnostics"]["min_predicted_covariance_eigenvalue"]) for row in subset
        ),
        "max_prediction_floor_count": max(
            int(row["diagnostics"]["max_prediction_floor_count"]) for row in subset
        ),
    }


def _veto_diagnostics(
    kalman: dict[str, Any],
    fd_delta: tf.Tensor,
    value_rows: list[dict[str, Any]],
    value_summaries: dict[str, Any],
    gradient_smoke: dict[str, Any],
    applicability: list[dict[str, Any]],
) -> dict[str, bool]:
    return {
        "kalman_reference_nonfinite": not bool(kalman["finite"]),
        "kalman_score_fd_diagnostic_large": scalar(tf.reduce_max(tf.abs(fd_delta))) > 1e-3,
        "value_row_nonfinite": any(not bool(row["finite"]) for row in value_rows),
        "gradient_smoke_nonfinite": not bool(gradient_smoke["finite"]),
        "missing_value_monte_carlo_uncertainty": any(
            summary[str(count)]["seed_count"] < len(VALUE_SEEDS)
            or not math.isfinite(float(summary[str(count)]["value_error_standard_error"]))
            for summary in value_summaries.values()
            for count in VALUE_PARTICLE_COUNTS
        ),
        "gradient_smoke_mislabeled_stochastic_score": (
            gradient_smoke["gradient_object"] != "short_horizon_fixed_branch_smoke"
            or gradient_smoke["stochastic_score_claim"] != "not_claimed"
        ),
        "old_ledh_pfpf_ot_used_as_algorithm1_evidence": any(
            "ledh_pfpf_ot" in str(row["method_id"])
            or bool(row["diagnostics"].get("old_ledh_pfpf_ot_used", False))
            for row in value_rows
        )
        or bool(gradient_smoke["diagnostics"].get("old_ledh_pfpf_ot_used", False)),
        "missing_algorithm1_route_identifiers": any(
            row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf"
            and row["diagnostics"].get("route_identifiers", {}).get("method_generation")
            != METHOD_GENERATION
            for row in value_rows
        )
        or gradient_smoke["diagnostics"].get("route_identifiers", {}).get("method_generation")
        != METHOD_GENERATION,
        "unsupported_pair_ranked": any(
            item["status"] == "DEFERRED_PENDING_EXPLICIT_ADAPTERS"
            and item.get("ranked", False)
            for item in applicability
        ),
        "value_pass_used_to_excuse_gradient_failure": False,
        "particle_ladder_promoted_to_convergence_proof": False,
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "kalman_reference",
        "routes",
        "value_rows",
        "value_summaries",
        "gradient_smoke",
        "applicability_matrix",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
    }
    missing = required.difference(payload)
    if missing:
        raise P5ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P5_LGSSM_ALG1_UKF_BOUNDED_DIAGNOSTICS_PENDING_CLAUDE_REVIEW",
        "P5_LGSSM_ALG1_UKF_BOUNDED_DIAGNOSTICS_VETO_PENDING_REVIEW",
    }:
        raise P5ValidationError(f"invalid P5 decision {payload['decision']}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "review_ledger_path",
        "value_seed_list",
        "value_particle_counts",
        "gradient_smoke_seed",
        "gradient_smoke_particle_count",
        "gradient_smoke_horizon",
        "pseudo_time_steps",
        "ukf_parameters",
        "route_identifiers",
    ):
        if field not in manifest:
            raise P5ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P5ValidationError("TensorFlow was not forced CPU-only before import")
    if payload["routes"]["previous_ledh_pfpf_ot"]["used_as_evidence"]:
        raise P5ValidationError("old LEDH-PFPF-OT was used as evidence")
    if len(payload["value_rows"]) != len(METHOD_IDS) * len(VALUE_SEEDS) * len(VALUE_PARTICLE_COUNTS):
        raise P5ValidationError("unexpected value-row count")
    for row in payload["value_rows"]:
        if "ledh_pfpf_ot" in str(row["method_id"]):
            raise P5ValidationError("old LEDH-PFPF-OT method id appeared in P5 rows")
        if not row["finite"]:
            raise P5ValidationError("nonfinite value row")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P5ValidationError("stochastic score claim appeared")
        if row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf":
            validate_algorithm1_route_identifiers(row["diagnostics"]["route_identifiers"])
    validate_algorithm1_route_identifiers(
        payload["gradient_smoke"]["diagnostics"]["route_identifiers"]
    )
    if not payload["gradient_smoke"]["finite"]:
        raise P5ValidationError("gradient smoke is nonfinite")
    if payload["veto_diagnostics"]["missing_value_monte_carlo_uncertainty"]:
        raise P5ValidationError("value Monte Carlo uncertainty summary is missing")
    if payload["veto_diagnostics"]["old_ledh_pfpf_ot_used_as_algorithm1_evidence"]:
        raise P5ValidationError("old LEDH-PFPF-OT evidence leaked into P5")
    if payload["veto_diagnostics"]["unsupported_pair_ranked"]:
        raise P5ValidationError("unsupported model/filter pair was ranked")
    if "reproducibility_digest" not in payload:
        raise P5ValidationError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P5 Result: Algorithm 1 UKF LEDH-PFPF LGSSM Diagnostics",
        "",
        "metadata_date: 2026-06-10",
        "phase: P5",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        "Status: `PASS_FOR_BOUNDED_LGSSM_DIAGNOSTICS_PENDING_CLAUDE_REVIEW`.",
        "",
        "Wrong-baseline risk is controlled by using exact QR Kalman as the LGSSM oracle. The previous `dpf_ledh_pfpf_ot` lineage is quarantined historical context only and is not used as Algorithm 1 evidence.",
        "",
        "Proxy-promotion risk is controlled by separating full-horizon value uncertainty from a short-horizon fixed-branch gradient smoke. ESS, determinant ranges, covariance floors, finite differences, and particle-ladder trends remain explanatory or veto diagnostics only.",
        "",
        "Unfair-comparison risk is controlled by ranking only applicable LGSSM value rows. The broader P44/filter matrix is deferred pending reviewed adapters and is not a comparison result here.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        "| Question | Record bounded full-horizon value diagnostics and short-horizon fixed-branch gradient smoke for source-faithful Algorithm 1 UKF LEDH-PFPF against exact LGSSM references. |",
        "| Baseline/comparator | `tf_qr_sqrt_differentiated_kalman` in `(transition_matrix_scale, observation_noise_scale)`. |",
        "| Primary criterion | Finite full-horizon value rows with uncertainty, finite short-horizon gradient smoke, source route identifiers. |",
        "| Veto diagnostics | Nonfinite rows, old-route reuse, missing value uncertainty, stochastic-score overclaim, unsupported pairs ranked. |",
        "| Not concluded | No full gradient ladder, no production default, no stochastic-score correctness, no nonlinear/P44 conclusion, no OT-as-source-Algorithm-1 claim. |",
        "",
        "## Veto Diagnostics",
        "",
        "| Diagnostic | Status |",
        "| --- | --- |",
        *[
            f"| `{key}` | `{value}` |"
            for key, value in payload["veto_diagnostics"].items()
        ],
        "",
        "## Applicability Matrix",
        "",
        "| Model | Filter | Status | Reason |",
        "| --- | --- | --- | --- |",
        *[
            f"| `{item['model']}` | `{item['filter']}` | `{item['status']}` | {item['reason']} |"
            for item in payload["applicability_matrix"]
        ],
        "",
        "## Kalman Reference",
        "",
        f"- full-horizon log likelihood: `{payload['kalman_reference']['log_likelihood']}`",
        f"- full-horizon score: `{payload['kalman_reference']['score']}`",
        f"- max score delta vs diagnostic finite difference: `{payload['kalman_reference']['max_abs_score_delta_vs_finite_difference']}`",
        "",
        "## Comparison Scope",
        "",
        f"- scope: `{payload['comparison_scope']['scope_id']}`",
        f"- value lane: {payload['comparison_scope']['value_lane']}",
        f"- gradient lane: {payload['comparison_scope']['gradient_lane']}",
        f"- full gradient ladder deferred: `{payload['comparison_scope']['full_lgssm_gradient_ladder_deferred']}`",
        f"- deferred seeds: `{payload['comparison_scope']['deferred_seed_list']}`",
        f"- deferred particle counts: `{payload['comparison_scope']['deferred_particle_counts']}`",
        "",
        "## Value Summaries",
        "",
    ]
    for method_id, summary in payload["value_summaries"].items():
        lines.append(f"### {method_id}")
        lines.append("")
        lines.append("| Particles | Mean value error | Value SE | Value RMSE | Min ESS |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for count in VALUE_PARTICLE_COUNTS:
            cell = summary[str(count)]
            lines.append(
                f"| {count} | `{cell['mean_value_error']}` | "
                f"`{cell['value_error_standard_error']}` | `{cell['value_error_rmse']}` | "
                f"`{cell['min_ess']}` |"
            )
        final = summary[str(VALUE_PARTICLE_COUNTS[-1])]
        lines.append(f"- diagnostic ranges: `{final['diagnostic_ranges']}`")
        lines.append(f"- particle ladder: `{summary['particle_ladder']}`")
        lines.append("")
    smoke = payload["gradient_smoke"]
    lines.extend(
        [
            "## Gradient Smoke",
            "",
            f"- method: `{smoke['method_id']}`",
            f"- horizon: `{smoke['horizon']}`",
            f"- particles: `{smoke['num_particles']}`",
            f"- value error: `{smoke['value_error']}`",
            f"- fixed-branch gradient: `{smoke['fixed_branch_gradient']}`",
            f"- Kalman score: `{smoke['kalman_score']}`",
            f"- gradient error norm: `{smoke['gradient_error_norm']}`",
            f"- relative score error: `{smoke['relative_score_error']}`",
            f"- gradient object: `{smoke['gradient_object']}`",
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | finite value uncertainty and finite gradient smoke | "
                f"`{payload['veto_diagnostics']}` | gradient evidence is short-horizon smoke only | "
                "run Claude read-only P5 review; then decide whether to add full gradient ladder or P44 adapters | "
                "no production, nonlinear/P44, full-gradient-ladder, stochastic-score, or OT-source claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: the Algorithm 1 row may reflect this no-resampling LGSSM fixture and the chosen pseudo-time/UKF settings rather than general LEDH-PFPF behavior.",
            "",
            "Result that would overturn the local interpretation: a reviewed rerun finds route leakage from old LEDH-PFPF-OT, non-finite covariance/determinant diagnostics, or a parameterization mismatch against the Kalman score.",
            "",
            "Weakest part of the evidence: P5 currently separates full-horizon value evidence from a short-horizon gradient smoke. The full gradient ladder and broader P44 rows remain future work.",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- dirty state summary: `{payload['run_manifest']['dirty_state_summary']}`",
            f"- environment/packages: `{payload['run_manifest']['package_versions']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- value seeds: `{payload['run_manifest']['value_seed_list']}`",
            f"- value particle counts: `{payload['run_manifest']['value_particle_counts']}`",
            f"- gradient smoke: seed `{payload['run_manifest']['gradient_smoke_seed']}`, particles `{payload['run_manifest']['gradient_smoke_particle_count']}`, horizon `{payload['run_manifest']['gradient_smoke_horizon']}`",
            f"- pseudo-time steps: `{payload['run_manifest']['pseudo_time_steps']}`",
            f"- UKF parameters: `{payload['run_manifest']['ukf_parameters']}`",
            f"- route identifiers: `{payload['run_manifest']['route_identifiers']}`",
            f"- data version: `{payload['run_manifest']['data_version']}`",
            f"- plan: `{payload['run_manifest']['plan_path']}`",
            f"- review ledger: `{payload['run_manifest']['review_ledger_path']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
            "## Gate Status",
            "",
            "P5 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _base_lgssm() -> dict[str, tf.Tensor]:
    a = tf.constant([[0.92, 0.18], [-0.04, 0.86]], dtype=DTYPE)
    c = tf.constant([[1.0, 0.35]], dtype=DTYPE)
    q = tf.constant([[0.08, 0.015], [0.015, 0.05]], dtype=DTYPE)
    r = tf.constant([[0.18]], dtype=DTYPE)
    m0 = tf.constant([0.25, -0.15], dtype=DTYPE)
    p0 = tf.constant([[0.45, 0.04], [0.04, 0.32]], dtype=DTYPE)
    states, observations = _simulate_lgssm(
        a=a,
        c=c,
        q=q,
        r=r,
        m0=m0,
        p0=p0,
        horizon=25,
        seed=2026052801,
    )
    return {
        "A": a,
        "C": c,
        "Q": q,
        "R": r,
        "m0": m0,
        "P0": p0,
        "states": states,
        "observations": observations,
        "fixture_generation_seed": tf.constant(2026052801, dtype=tf.int64),
    }


def _parameters(theta: tf.Tensor, base: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
    transition_scale, observation_scale = tf.unstack(tf.cast(theta, DTYPE))
    return {
        **base,
        "A": base["A"] * transition_scale,
        "R": base["R"] * observation_scale,
    }


def _kalman_reference(theta: tf.Tensor, base: dict[str, tf.Tensor]) -> dict[str, Any]:
    params = _parameters(theta, base)
    horizon = int(params["observations"].shape[0])
    model = TFLinearGaussianStateSpace(
        initial_mean=params["m0"],
        initial_covariance=params["P0"],
        transition_offset=tf.zeros([2], dtype=DTYPE),
        transition_matrix=params["A"],
        transition_covariance=params["Q"],
        observation_offset=tf.zeros([1], dtype=DTYPE),
        observation_matrix=params["C"],
        observation_covariance=params["R"],
    )
    zero_n = tf.zeros([2, 2], dtype=DTYPE)
    zero_m = tf.zeros([2, 1], dtype=DTYPE)
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=zero_n,
        d_initial_covariance=tf.zeros([2, 2, 2], dtype=DTYPE),
        d_transition_offset=zero_n,
        d_transition_matrix=tf.stack([params["A"] / theta[0], tf.zeros_like(params["A"])], axis=0),
        d_transition_covariance=tf.zeros([2, 2, 2], dtype=DTYPE),
        d_observation_offset=zero_m,
        d_observation_matrix=tf.zeros([2, 1, 2], dtype=DTYPE),
        d_observation_covariance=tf.stack([tf.zeros_like(params["R"]), params["R"] / theta[1]], axis=0),
        d2_initial_mean=tf.zeros([2, 2, 2], dtype=DTYPE),
        d2_initial_covariance=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_transition_offset=tf.zeros([2, 2, 2], dtype=DTYPE),
        d2_transition_matrix=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_transition_covariance=tf.zeros([2, 2, 2, 2], dtype=DTYPE),
        d2_observation_offset=tf.zeros([2, 2, 1], dtype=DTYPE),
        d2_observation_matrix=tf.zeros([2, 2, 1, 2], dtype=DTYPE),
        d2_observation_covariance=tf.zeros([2, 2, 1, 1], dtype=DTYPE),
    )
    result = tf_qr_linear_gaussian_score_hessian(
        params["observations"],
        model,
        derivatives,
        backend="tf_qr_sqrt",
        jitter=tf.constant(1e-9, dtype=DTYPE),
    )
    score = tf.cast(result.score, DTYPE)
    hessian = tf.cast(result.hessian, DTYPE)
    return {
        "reference_id": "tf_qr_sqrt_differentiated_kalman",
        "horizon": horizon,
        "log_likelihood": scalar(result.log_likelihood),
        "score": tensor_to_json(score),
        "hessian": tensor_to_json(hessian),
        "finite": bool(
            tf.math.is_finite(result.log_likelihood).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
            and tf.reduce_all(tf.math.is_finite(hessian)).numpy()
        ),
        "metadata": {
            "filter_name": result.metadata.filter_name,
            "differentiability_status": result.metadata.differentiability_status,
            "jitter": 1e-9,
        },
    }


def _kalman_value(theta: tf.Tensor, base: dict[str, tf.Tensor]) -> tf.Tensor:
    params = _parameters(theta, base)
    mean = params["m0"]
    covariance = params["P0"]
    value = tf.constant(0.0, dtype=DTYPE)
    for observation in tf.unstack(params["observations"], axis=0):
        pred_mean = tf.linalg.matvec(params["A"], mean)
        pred_cov = params["A"] @ covariance @ tf.transpose(params["A"]) + params["Q"]
        obs_mean = tf.linalg.matvec(params["C"], pred_mean)
        innovation_cov = params["C"] @ pred_cov @ tf.transpose(params["C"]) + params["R"]
        residual = tf.reshape(observation, [-1]) - obs_mean
        value = value + _gaussian_logpdf_zero_mean(tf.reshape(residual, [1, -1]), innovation_cov)[0]
        gain = tf.transpose(tf.linalg.solve(innovation_cov, params["C"] @ pred_cov))
        mean = pred_mean + tf.linalg.matvec(gain, residual)
        covariance = _symmetrize((tf.eye(2, dtype=DTYPE) - gain @ params["C"]) @ pred_cov)
    return value


def _simulate_lgssm(
    *,
    a: tf.Tensor,
    c: tf.Tensor,
    q: tf.Tensor,
    r: tf.Tensor,
    m0: tf.Tensor,
    p0: tf.Tensor,
    horizon: int,
    seed: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    x0 = _mvn_sample(m0, p0, seed, 1, ())
    states = [x0]
    observations = []
    for t in range(horizon):
        state = tf.linalg.matvec(a, states[-1]) + _mvn_sample(
            tf.zeros([2], dtype=DTYPE),
            q,
            seed,
            10 + t,
            (),
        )
        obs = tf.linalg.matvec(c, state) + _mvn_sample(
            tf.zeros([1], dtype=DTYPE),
            r,
            seed,
            100 + t,
            (),
        )
        states.append(state)
        observations.append(obs)
    return tf.stack(states, axis=0), tf.stack(observations, axis=0)


def _sample_initial(params: dict[str, tf.Tensor], num_particles: int, seed: int) -> tf.Tensor:
    return _mvn_sample(params["m0"], params["P0"], seed, 11, (num_particles,))


def _transition_sample(
    params: dict[str, tf.Tensor],
    particles: tf.Tensor,
    seed: int,
    time_index: int,
) -> tf.Tensor:
    mean = tf.linalg.matmul(tf.cast(particles, DTYPE), params["A"], transpose_b=True)
    noise = _mvn_sample(
        tf.zeros([2], dtype=DTYPE),
        params["Q"],
        seed,
        1000 + int(time_index),
        (int(particles.shape[0]),),
    )
    return mean + noise


def _mvn_sample(
    loc: tf.Tensor,
    covariance: tf.Tensor,
    seed: int,
    salt: int,
    sample_shape: tuple[int, ...],
) -> tf.Tensor:
    loc = tf.cast(loc, DTYPE)
    chol = tf.linalg.cholesky(tf.cast(covariance, DTYPE))
    shape = list(sample_shape) + [int(loc.shape[0])]
    normal = tf.random.stateless_normal(shape, seed=_seed_pair(seed, salt), dtype=DTYPE)
    if not sample_shape:
        return loc + tf.linalg.matvec(chol, normal)
    return loc + tf.linalg.matmul(normal, chol, transpose_b=True)


def _transition_log_density(
    params: dict[str, tf.Tensor],
    previous: tf.Tensor,
    current: tf.Tensor,
) -> tf.Tensor:
    mean = tf.linalg.matmul(tf.cast(previous, DTYPE), params["A"], transpose_b=True)
    return _gaussian_logpdf_zero_mean(tf.cast(current, DTYPE) - mean, params["Q"])


def _observation_log_density(
    params: dict[str, tf.Tensor],
    particles: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    loc = tf.linalg.matmul(tf.cast(particles, DTYPE), params["C"], transpose_b=True)
    residual = tf.reshape(tf.cast(observation, DTYPE), [1, 1]) - loc
    return _gaussian_logpdf_zero_mean(residual, params["R"])


def _gaussian_logpdf_zero_mean(residuals: tf.Tensor, covariance: tf.Tensor) -> tf.Tensor:
    residuals = tf.cast(residuals, DTYPE)
    covariance = _symmetrize(tf.cast(covariance, DTYPE))
    dim = tf.cast(tf.shape(residuals)[-1], DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.triangular_solve(chol, tf.transpose(residuals), lower=True)
    quadratic = tf.reduce_sum(solved * solved, axis=0)
    log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))
    return -0.5 * (dim * tf.math.log(tf.constant(2.0 * math.pi, dtype=DTYPE)) + log_det + quadratic)


def _applicability_matrix() -> list[dict[str, Any]]:
    return [
        {
            "model": "p5_lgssm_2d_h25_rich",
            "filter": "kalman_exact",
            "status": "APPLICABLE_ORACLE",
            "reason": "linear Gaussian state-space model",
        },
        {
            "model": "p5_lgssm_2d_h25_rich",
            "filter": "bootstrap_pf_no_resampling_tf",
            "status": "APPLICABLE_VALUE_BASELINE",
            "reason": "transition and observation density callbacks are exact",
        },
        {
            "model": "p5_lgssm_2d_h25_rich",
            "filter": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "status": "APPLICABLE_SOURCE_FAITHFUL_CORE",
            "reason": "all Algorithm 1 callbacks supplied with linear observation Jacobian and exact Gaussian densities",
        },
        {
            "model": "P44 broader model/filter matrix",
            "filter": "UKF/SVD/CUT4/Zhao-Cui/new Algorithm 1 DPF",
            "status": "DEFERRED_PENDING_EXPLICIT_ADAPTERS",
            "reason": (
                "not ranked in this bounded P5 lane; adapters and applicability "
                "contracts must be reviewed before broader rows are evidence"
            ),
        },
    ]


def _model_definition(base: dict[str, tf.Tensor]) -> dict[str, Any]:
    return {
        "model_id": "p5_lgssm_2d_h25_rich",
        "state_equation": "x_t = transition_matrix_scale * A x_{t-1} + q_t",
        "observation_equation": "y_t = C x_t + r_t",
        "q_t": "Normal(0, Q)",
        "r_t": "Normal(0, observation_noise_scale * R)",
        "initial_distribution": "Normal(m0, P0)",
        "A": tensor_to_json(base["A"]),
        "C": tensor_to_json(base["C"]),
        "Q": tensor_to_json(base["Q"]),
        "R": tensor_to_json(base["R"]),
        "m0": tensor_to_json(base["m0"]),
        "P0": tensor_to_json(base["P0"]),
        "horizon": int(base["observations"].shape[0]),
        "fixture_generation_seed": int(base["fixture_generation_seed"].numpy()),
        "model_checksum": stable_digest(
            {
                "A": tensor_to_json(base["A"]),
                "C": tensor_to_json(base["C"]),
                "Q": tensor_to_json(base["Q"]),
                "R": tensor_to_json(base["R"]),
                "m0": tensor_to_json(base["m0"]),
                "P0": tensor_to_json(base["P0"]),
            }
        ),
        "observation_checksum": stable_digest(tensor_to_json(base["observations"])),
    }


def _ukf_parameters() -> dict[str, float]:
    return {
        "alpha": UKF_ALPHA,
        "beta": UKF_BETA,
        "kappa": UKF_KAPPA,
    }


def _nonclaims() -> list[str]:
    return [
        "P5 full-horizon value rows do not include fixed-branch gradients.",
        "P5 gradient evidence is a short-horizon smoke, not a full LGSSM gradient ladder.",
        "P5 no-resampling rows do not establish classical-resampling or OT-extension correctness.",
        "Finite differences are diagnostic only and are not a promotion gate.",
        "The previous LEDH-PFPF-OT lineage remains quarantined and is not used as Algorithm 1 evidence.",
        "The broader P44/filter matrix is deferred pending reviewed adapters.",
        "P5 does not establish HMC readiness, production readiness, or GPU performance.",
    ]


def _finite_difference_score(
    value_fn: Callable[[tf.Tensor], tf.Tensor],
    theta: tf.Tensor,
) -> tf.Tensor:
    values = []
    for index in range(int(theta.shape[0])):
        basis = tf.one_hot(index, int(theta.shape[0]), dtype=DTYPE)
        plus = theta + tf.constant(FD_STEP, DTYPE) * basis
        minus = theta - tf.constant(FD_STEP, DTYPE) * basis
        values.append((value_fn(plus) - value_fn(minus)) / tf.constant(2.0 * FD_STEP, DTYPE))
    return tf.stack(values)


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return statistics.stdev(values) / math.sqrt(len(values))


def _ci95(values: list[float]) -> list[float]:
    mean = statistics.fmean(values)
    se = _standard_error(values)
    return [mean - 1.96 * se, mean + 1.96 * se]


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean([value * value for value in values]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return float("inf") if numerator != 0.0 else 0.0
    return numerator / denominator


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
