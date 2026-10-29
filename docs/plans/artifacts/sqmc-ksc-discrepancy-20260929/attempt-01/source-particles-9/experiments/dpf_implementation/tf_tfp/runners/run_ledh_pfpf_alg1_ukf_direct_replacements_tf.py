"""Run P1 direct replacements for quarantined LEDH-PFPF-OT lanes.

This runner intentionally reuses the reviewed Algorithm 1 UKF LEDH-PFPF core
from the P5 LGSSM diagnostic lane.  It does not import the old
``ledh_pfpf_ot_tf`` implementation.
"""

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
from typing import Any

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    METHOD_GENERATION,
    algorithm1_route_identifiers,
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
from experiments.dpf_implementation.tf_tfp.runners.run_ledh_pfpf_alg1_ukf_p5_lgssm_comparison_tf import (
    COVARIANCE_FLOOR,
    DTYPE,
    PSEUDO_TIME_STEPS,
    RANK_TOLERANCE,
    _base_lgssm,
    _ci95,
    _finite_difference_score,
    _kalman_reference,
    _kalman_value,
    _method_value,
    _model_definition,
    _parameters,
    _rmse,
    _run_method_with_params,
    _safe_ratio,
    _standard_error,
    _ukf_parameters,
)


MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_ledh_pfpf_alg1_ukf_direct_replacements_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p1-direct-lgssm-range-bearing-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p1-direct-lgssm-range-bearing-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
REGISTRY_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-registry-2026-06-10.json"
)
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_alg1_ukf_direct_replacements_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-alg1-ukf-direct-replacements-2026-06-10.md"

METHOD_IDS = (
    "bootstrap_pf_no_resampling_tf",
    "ledh_pfpf_alg1_ukf_no_resampling_tf",
)
VALUE_SEEDS = [101, 202, 303, 404, 505]
VALUE_PARTICLE_COUNTS = [8, 16, 32]
GRADIENT_SEEDS = [101, 202, 303]
GRADIENT_PARTICLE_COUNTS = [4, 8, 16]
GRADIENT_HORIZON = 3


class P1ValidationError(ValueError):
    """Raised when the P1 payload violates the direct-replacement contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P1_LEDHPFPF_ALG1_UKF_DIRECT_REPLACEMENTS_VALIDATED")
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
    registry = load_json(REPO_ROOT / REGISTRY_PATH)
    route = _augmented_algorithm1_route(resampling_route="none")
    base = _base_lgssm()
    theta0 = tf.constant([1.0, 1.0], dtype=DTYPE)
    kalman = _kalman_reference(theta0, base)
    fd_score = _finite_difference_score(lambda x: _kalman_value(x, base), theta0)
    fd_delta = tf.constant(kalman["score"], dtype=DTYPE) - fd_score

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
                        route=route,
                    )
                )

    value_summaries = {
        method_id: _value_summary(method_id, value_rows)
        for method_id in METHOD_IDS
    }
    gradient_rows = []
    for num_particles in GRADIENT_PARTICLE_COUNTS:
        for seed in GRADIENT_SEEDS:
            gradient_rows.append(
                _gradient_row(
                    theta=theta0,
                    base=base,
                    seed=seed,
                    num_particles=num_particles,
                    route=route,
                )
            )
    gradient_summary = _gradient_summary(gradient_rows)
    lane_statuses = _lane_statuses(value_summaries, gradient_summary)
    blocked_adapters = _blocked_adapters(route)
    veto = _veto_diagnostics(
        kalman=kalman,
        fd_delta=fd_delta,
        value_rows=value_rows,
        value_summaries=value_summaries,
        gradient_rows=gradient_rows,
        lane_statuses=lane_statuses,
    )
    decision = (
        "PASS_P1_DIRECT_REPLACEMENTS_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P1_DIRECT_REPLACEMENTS_VETO_PENDING_REVIEW"
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
            "registry_path": REGISTRY_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "value_seed_list": list(VALUE_SEEDS),
            "value_particle_counts": list(VALUE_PARTICLE_COUNTS),
            "gradient_seed_list": list(GRADIENT_SEEDS),
            "gradient_particle_counts": list(GRADIENT_PARTICLE_COUNTS),
            "gradient_horizon": GRADIENT_HORIZON,
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "algorithm1_route_identifiers": route,
            "data_version": "fixed local LGSSM fixture generated by reviewed P5 runner",
            "old_lane_registry_rows": _registry_lane_ids(registry, replacement_phase="P1"),
        }
    )
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P1",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "skeptical_plan_audit": {
            "status": "PASS_FOR_P1_DIRECT_DIAGNOSTIC_RERUN",
            "wrong_baseline_control": (
                "LGSSM rows use exact differentiated Kalman as oracle; bootstrap is "
                "a no-flow baseline, not a correctness oracle."
            ),
            "proxy_metric_control": (
                "Range-bearing UKF proxy and finite-only value agreement are not "
                "promotion criteria in this phase."
            ),
            "stop_conditions": (
                "P1 stops at Claude AGREE or max 5 review iterations; nonfinite rows, "
                "old-route leakage, missing route fields, or unsupported rankings veto."
            ),
            "environment_control": (
                "TensorFlow is forced CPU-only before import and recorded in manifest."
            ),
        },
        "evidence_contract": {
            "question": (
                "Replace the old direct LGSSM, LGSSM multiseed, range-bearing, "
                "range-bearing stress, and gradient LEDH-PFPF-OT lanes with "
                "Algorithm 1 UKF rerun statuses."
            ),
            "baseline_comparator": (
                "Exact QR Kalman value/score for LGSSM plus bootstrap no-flow PF as "
                "a baseline comparator."
            ),
            "primary_criterion": (
                "P1 promotes only route-faithful rerun/classification evidence. "
                "Numerical closeness rows are diagnostic-only until a calibrated "
                "P5/P6 threshold contract is reviewed."
            ),
            "veto_diagnostics": list(veto.keys()),
            "explanatory_only": [
                "ESS",
                "determinant ranges",
                "covariance eigenvalues",
                "particle-count trends",
                "finite-difference Kalman score check",
                "value and gradient error magnitudes",
            ],
            "not_concluded": _nonclaims(),
        },
        "threshold_policy": {
            "value_promotion_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "gradient_promotion_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "reason": (
                "P1 reruns and classifies the old direct lanes.  It does not create "
                "a calibrated numerical acceptance band from one fixture, noise "
                "scale, horizon, or particle ladder."
            ),
            "finite_only_promotion_allowed": False,
            "future_certification_phase": "P5/P6 exact-oracle/calibration phases",
        },
        "target": {
            "target_id": "p1_lgssm_2d_h25_rich_direct_replacement",
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
            "gradient_seeds": list(GRADIENT_SEEDS),
            "gradient_particle_counts": list(GRADIENT_PARTICLE_COUNTS),
            "confidence_interval": "normal 95% CI over seeds by method/particle count",
            "gradient_object": "short_horizon_fixed_branch_gradient_diagnostic",
            "stochastic_score_claim": "not_claimed",
            "resampling_policy": "none for Algorithm 1 core rows in P1",
        },
        "algorithm1_parameters": {
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "covariance_floor": COVARIANCE_FLOOR,
            "rank_tolerance": RANK_TOLERANCE,
            "route_identifiers": route,
        },
        "lane_statuses": lane_statuses,
        "blocked_adapters": blocked_adapters,
        "value_rows": value_rows,
        "value_summaries": value_summaries,
        "gradient_rows": gradient_rows,
        "gradient_summary": gradient_summary,
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "value_agreement_is_not_gradient_promotion": True,
            "finite_difference_is_diagnostic_only": True,
            "old_ledh_pfpf_ot_used_as_oracle": False,
            "old_ledh_pfpf_ot_used_as_algorithm1_evidence": False,
            "monte_carlo_value_uncertainty_reported": True,
            "range_bearing_rows_ranked": False,
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
    route: dict[str, str],
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
        "old_lane_ids": ["direct_lgssm_value", "direct_lgssm_multiseed"],
        "method_id": method_id,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "value": scalar(value),
        "kalman_value": kalman["log_likelihood"],
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(
            value_error / tf.constant(float(kalman["horizon"]), dtype=DTYPE)
        ),
        "finite": bool(tf.math.is_finite(value).numpy()),
        "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
        "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        "gradient_tolerance": "N/A",
        "gradient_object": "not_evaluated_in_value_lane",
        "stochastic_score_claim": "not_claimed",
        "route_fields": _row_route_fields(method_id, route),
        "diagnostics": diagnostics,
    }


def _gradient_row(
    *,
    theta: tf.Tensor,
    base: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
    route: dict[str, str],
) -> dict[str, Any]:
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
            seed=seed,
            num_particles=num_particles,
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
        "old_lane_ids": ["direct_gradient_checks", "direct_lgssm_multiseed"],
        "method_id": "ledh_pfpf_alg1_ukf_no_resampling_tf",
        "seed": int(seed),
        "num_particles": int(num_particles),
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
        "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
        "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        "gradient_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        "gradient_object": "short_horizon_fixed_branch_gradient_diagnostic",
        "stochastic_score_claim": "not_claimed",
        "route_fields": route,
        "diagnostics": diagnostics,
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
        per_obs_errors = [float(row["per_observation_value_error"]) for row in subset]
        out[str(num_particles)] = {
            "seed_count": len(subset),
            "mean_value_error": statistics.fmean(value_errors),
            "value_error_standard_error": _standard_error(value_errors),
            "value_error_ci95": _ci95(value_errors),
            "value_error_rmse": _rmse(value_errors),
            "mean_per_observation_value_error": statistics.fmean(per_obs_errors),
            "per_observation_value_error_standard_error": _standard_error(per_obs_errors),
            "max_abs_value_error": max(abs(value) for value in value_errors),
            "min_ess": min(float(row["diagnostics"]["ess_min"]) for row in subset),
            "mean_resampling_count": statistics.fmean(
                float(row["diagnostics"]["resampling_count"]) for row in subset
            ),
            "all_rows_finite": all(bool(row["finite"]) for row in subset),
            "diagnostic_ranges": _diagnostic_ranges(method_id, subset),
            "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
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


def _gradient_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for num_particles in GRADIENT_PARTICLE_COUNTS:
        subset = [row for row in rows if row["num_particles"] == num_particles]
        errors = [float(row["gradient_error_norm"]) for row in subset]
        rel_errors = [float(row["relative_score_error"]) for row in subset]
        out[str(num_particles)] = {
            "seed_count": len(subset),
            "mean_gradient_error_norm": statistics.fmean(errors),
            "gradient_error_norm_standard_error": _standard_error(errors),
            "gradient_error_norm_ci95": _ci95(errors),
            "gradient_error_norm_rmse": _rmse(errors),
            "mean_relative_score_error": statistics.fmean(rel_errors),
            "relative_score_error_standard_error": _standard_error(rel_errors),
            "max_gradient_error_norm": max(errors),
            "all_rows_finite": all(bool(row["finite"]) for row in subset),
            "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
        }
    counts = [str(count) for count in GRADIENT_PARTICLE_COUNTS]
    out["particle_ladder"] = {
        "gradient_error_rmse_ratio": _safe_ratio(
            out[counts[-1]]["gradient_error_norm_rmse"],
            out[counts[0]]["gradient_error_norm_rmse"],
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


def _lane_statuses(
    value_summaries: dict[str, Any],
    gradient_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "old_lane_id": "direct_lgssm_value",
            "status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "replacement_method": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "comparator_route": "exact_kalman_and_bootstrap_no_flow",
            "seed_count": len(VALUE_SEEDS),
            "particle_ladder": list(VALUE_PARTICLE_COUNTS),
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "gradient_tolerance": "N/A",
            "result_pointer": {
                "json": str(JSON_PATH.relative_to(REPO_ROOT)),
                "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "phase_result": RESULT_PATH,
            },
            "summary": value_summaries["ledh_pfpf_alg1_ukf_no_resampling_tf"],
            "reason": "LGSSM value rerun completed, but P1 has no calibrated promotion band.",
        },
        {
            "old_lane_id": "direct_lgssm_multiseed",
            "status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "replacement_method": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "comparator_route": "exact_kalman_and_bootstrap_no_flow",
            "seed_count": len(VALUE_SEEDS),
            "particle_ladder": list(VALUE_PARTICLE_COUNTS),
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "gradient_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "result_pointer": {
                "json": str(JSON_PATH.relative_to(REPO_ROOT)),
                "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "phase_result": RESULT_PATH,
            },
            "summary": {
                "value_summary": value_summaries["ledh_pfpf_alg1_ukf_no_resampling_tf"],
                "gradient_summary": gradient_summary,
            },
            "reason": "Multiseed rerun completed; trends are recorded but not promoted.",
        },
        {
            "old_lane_id": "direct_range_bearing_value",
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "replacement_method": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "comparator_route": "range_bearing_reference_must_be_declared",
            "seed_count": "N/A",
            "particle_ladder": "N/A",
            "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "value_tolerance": "N/A until range-bearing reference route is reviewed",
            "gradient_tolerance": "N/A",
            "result_pointer": {
                "json": str(JSON_PATH.relative_to(REPO_ROOT)),
                "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "phase_result": RESULT_PATH,
            },
            "reason": "Algorithm 1 nonlinear callbacks/reference contract are not reviewed for range-bearing in P1.",
        },
        {
            "old_lane_id": "direct_range_bearing_stress",
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "replacement_method": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "comparator_route": "diagnostic_stress_only_until_reference_declared",
            "seed_count": "N/A",
            "particle_ladder": "N/A",
            "primary_promote_statistic": "N/A_BLOCKED_REQUIRES_ADAPTER",
            "value_tolerance": "N/A until non-stress range-bearing adapter is reviewed",
            "gradient_tolerance": "N/A",
            "result_pointer": {
                "json": str(JSON_PATH.relative_to(REPO_ROOT)),
                "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "phase_result": RESULT_PATH,
            },
            "reason": "Stress lane depends on the blocked range-bearing Algorithm 1 adapter/reference contract.",
        },
        {
            "old_lane_id": "direct_gradient_checks",
            "status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
            "replacement_method": "ledh_pfpf_alg1_ukf_no_resampling_tf",
            "comparator_route": "exact_kalman_for_lgssm_gradient",
            "seed_count": len(GRADIENT_SEEDS),
            "particle_ladder": list(GRADIENT_PARTICLE_COUNTS),
            "primary_promote_statistic": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "value_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "gradient_tolerance": "N/A_DIAGNOSTIC_ONLY_IN_P1",
            "result_pointer": {
                "json": str(JSON_PATH.relative_to(REPO_ROOT)),
                "markdown_report": str(REPORT_PATH.relative_to(REPO_ROOT)),
                "phase_result": RESULT_PATH,
            },
            "summary": gradient_summary,
            "reason": "Same-scalar fixed-branch gradient rerun completed, but P1 does not promote stochastic gradient correctness.",
        },
    ]


def _blocked_adapters(route: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "old_lane_id": "direct_range_bearing_value",
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "missing_items": [
                "reviewed Algorithm 1 transition_mean_fn for nonlinear range-bearing fixture",
                "reviewed per-particle observation_jacobian_fn convention for Algorithm 1 local linearization",
                "reviewed transition_log_density_fn and observation_log_density_fn matching the target scalar",
                "declared comparator route with explicit oracle/proxy status and thresholds",
                "pre-run Monte Carlo uncertainty and veto contract for nonlinear value rows",
            ],
            "route_fields_if_unblocked": route,
            "not_a_valid_stop_reason": (
                "This is a phase classification, not an execution stop.  Later phases "
                "must either implement these adapters or keep the lane blocked."
            ),
        },
        {
            "old_lane_id": "direct_range_bearing_stress",
            "status": "BLOCKED_REQUIRES_ADAPTER",
            "missing_items": [
                "all non-stress range-bearing Algorithm 1 adapter requirements",
                "separate stress scalar and stop condition",
                "explicit statement that stress pass is diagnostic only",
            ],
            "route_fields_if_unblocked": route,
            "not_a_valid_stop_reason": (
                "This is a phase classification, not an execution stop.  Stress cannot "
                "run as Algorithm 1 evidence before the base nonlinear adapter is reviewed."
            ),
        },
    ]


def _veto_diagnostics(
    *,
    kalman: dict[str, Any],
    fd_delta: tf.Tensor,
    value_rows: list[dict[str, Any]],
    value_summaries: dict[str, Any],
    gradient_rows: list[dict[str, Any]],
    lane_statuses: list[dict[str, Any]],
) -> dict[str, bool]:
    loaded_old_modules = [
        name
        for name in sys.modules
        if name.endswith(".ledh_pfpf_ot_tf")
        or name.endswith(".run_lgssm_ledh_pfpf_ot_tf")
        or name.endswith(".run_range_bearing_ledh_pfpf_ot_tf")
        or name.endswith(".run_range_bearing_stress_ledh_pfpf_ot_tf")
        or name.endswith(".run_ledh_pfpf_gradient_checks_tf")
    ]
    return {
        "kalman_reference_nonfinite": not bool(kalman["finite"]),
        "kalman_score_fd_diagnostic_large": scalar(tf.reduce_max(tf.abs(fd_delta))) > 1e-3,
        "old_ledh_pfpf_ot_runtime_module_imported": bool(loaded_old_modules),
        "value_row_nonfinite": any(not bool(row["finite"]) for row in value_rows),
        "gradient_row_nonfinite": any(not bool(row["finite"]) for row in gradient_rows),
        "missing_value_monte_carlo_uncertainty": any(
            summary[str(count)]["seed_count"] < len(VALUE_SEEDS)
            or not math.isfinite(float(summary[str(count)]["value_error_standard_error"]))
            for summary in value_summaries.values()
            for count in VALUE_PARTICLE_COUNTS
        ),
        "missing_gradient_monte_carlo_uncertainty": any(
            "gradient_summary" not in lane.get("summary", {})
            and lane["old_lane_id"] == "direct_lgssm_multiseed"
            for lane in lane_statuses
        ),
        "algorithm1_route_fields_missing_or_wrong": any(
            row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf"
            and not _is_augmented_algorithm1_route(row["route_fields"])
            for row in value_rows + gradient_rows
        ),
        "old_ledh_pfpf_ot_used_as_algorithm1_evidence": any(
            "ledh_pfpf_ot" in str(row["method_id"])
            or bool(row["diagnostics"].get("old_ledh_pfpf_ot_used", False))
            for row in value_rows + gradient_rows
        ),
        "unsupported_range_bearing_pair_ranked": any(
            lane["old_lane_id"] in {"direct_range_bearing_value", "direct_range_bearing_stress"}
            and lane["status"] != "BLOCKED_REQUIRES_ADAPTER"
            for lane in lane_statuses
        ),
        "value_or_gradient_numerical_closeness_promoted_in_p1": any(
            lane["status"] == "RERUN_ALG1"
            for lane in lane_statuses
        ),
        "value_agreement_used_to_promote_gradient_correctness": False,
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "skeptical_plan_audit",
        "evidence_contract",
        "threshold_policy",
        "routes",
        "lane_statuses",
        "blocked_adapters",
        "value_rows",
        "value_summaries",
        "gradient_rows",
        "gradient_summary",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P1ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != "PASS_P1_DIRECT_REPLACEMENTS_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW":
        raise P1ValidationError(f"P1 decision is not pass: {payload['decision']}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "review_ledger_path",
        "registry_path",
        "value_seed_list",
        "value_particle_counts",
        "gradient_seed_list",
        "gradient_particle_counts",
        "gradient_horizon",
        "pseudo_time_steps",
        "ukf_parameters",
        "algorithm1_route_identifiers",
    ):
        if field not in manifest:
            raise P1ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P1ValidationError("TensorFlow was not forced CPU-only before import")
    if payload["routes"]["previous_ledh_pfpf_ot"]["used_as_evidence"]:
        raise P1ValidationError("old LEDH-PFPF-OT was used as evidence")
    if len(payload["value_rows"]) != len(METHOD_IDS) * len(VALUE_SEEDS) * len(VALUE_PARTICLE_COUNTS):
        raise P1ValidationError("unexpected value-row count")
    if len(payload["gradient_rows"]) != len(GRADIENT_SEEDS) * len(GRADIENT_PARTICLE_COUNTS):
        raise P1ValidationError("unexpected gradient-row count")
    expected_lanes = {
        "direct_lgssm_value",
        "direct_lgssm_multiseed",
        "direct_range_bearing_value",
        "direct_range_bearing_stress",
        "direct_gradient_checks",
    }
    actual_lanes = {row["old_lane_id"] for row in payload["lane_statuses"]}
    if actual_lanes != expected_lanes:
        raise P1ValidationError(f"wrong lane set {sorted(actual_lanes)}")
    for row in payload["value_rows"]:
        if "ledh_pfpf_ot" in str(row["method_id"]):
            raise P1ValidationError("old LEDH-PFPF-OT method id appeared in P1 rows")
        if not row["finite"]:
            raise P1ValidationError("nonfinite value row")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P1ValidationError("stochastic score claim appeared in value row")
        if row["method_id"] == "ledh_pfpf_alg1_ukf_no_resampling_tf":
            if not _is_augmented_algorithm1_route(row["route_fields"]):
                raise P1ValidationError("Algorithm 1 value row route fields are missing")
            validate_algorithm1_route_identifiers(row["diagnostics"]["route_identifiers"])
    for row in payload["gradient_rows"]:
        if not row["finite"]:
            raise P1ValidationError("nonfinite gradient row")
        if row["row_status"] != "RERUN_ALG1_DIAGNOSTIC_ONLY":
            raise P1ValidationError("P1 gradient row was promoted")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P1ValidationError("stochastic score claim appeared in gradient row")
        if not _is_augmented_algorithm1_route(row["route_fields"]):
            raise P1ValidationError("Algorithm 1 gradient row route fields are missing")
        validate_algorithm1_route_identifiers(row["diagnostics"]["route_identifiers"])
    if payload["veto_diagnostics"]["old_ledh_pfpf_ot_runtime_module_imported"]:
        raise P1ValidationError("old LEDH-PFPF-OT runtime module was imported")
    if payload["veto_diagnostics"]["old_ledh_pfpf_ot_used_as_algorithm1_evidence"]:
        raise P1ValidationError("old LEDH-PFPF-OT evidence leaked into P1")
    if payload["veto_diagnostics"]["unsupported_range_bearing_pair_ranked"]:
        raise P1ValidationError("range-bearing was ranked without adapter")
    if payload["veto_diagnostics"]["value_or_gradient_numerical_closeness_promoted_in_p1"]:
        raise P1ValidationError("P1 promoted numerical closeness without calibrated threshold")
    if payload["threshold_policy"]["finite_only_promotion_allowed"]:
        raise P1ValidationError("finite-only promotion was allowed")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P1 Result: Direct Algorithm 1 UKF Replacements For Old LEDH-PFPF-OT Lanes",
        "",
        "metadata_date: 2026-06-10",
        "phase: P1",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        f"Status: `{payload['skeptical_plan_audit']['status']}`.",
        "",
        payload["skeptical_plan_audit"]["wrong_baseline_control"],
        "",
        payload["skeptical_plan_audit"]["proxy_metric_control"],
        "",
        payload["skeptical_plan_audit"]["stop_conditions"],
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['evidence_contract']['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Threshold policy | {payload['threshold_policy']['reason']} |",
        f"| Not concluded | {'; '.join(payload['evidence_contract']['not_concluded'])} |",
        "",
        "## Lane Statuses",
        "",
        "| Old lane | Status | Comparator | Seeds | Particles | Reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for lane in payload["lane_statuses"]:
        lines.append(
            f"| `{lane['old_lane_id']}` | `{lane['status']}` | "
            f"`{lane['comparator_route']}` | `{lane['seed_count']}` | "
            f"`{lane['particle_ladder']}` | {lane['reason']} |"
        )
    lines.extend(
        [
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
            "## Kalman Reference",
            "",
            f"- full-horizon log likelihood: `{payload['kalman_reference']['log_likelihood']}`",
            f"- full-horizon score: `{payload['kalman_reference']['score']}`",
            f"- max score delta vs diagnostic finite difference: `{payload['kalman_reference']['max_abs_score_delta_vs_finite_difference']}`",
            "",
            "## Value Summaries",
            "",
        ]
    )
    for method_id, summary in payload["value_summaries"].items():
        lines.append(f"### {method_id}")
        lines.append("")
        lines.append("| Particles | Seeds | Mean value error | Value SE | Per-observation mean error | Value RMSE | Min ESS |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for count in VALUE_PARTICLE_COUNTS:
            cell = summary[str(count)]
            lines.append(
                f"| {count} | `{cell['seed_count']}` | `{cell['mean_value_error']}` | "
                f"`{cell['value_error_standard_error']}` | "
                f"`{cell['mean_per_observation_value_error']}` | "
                f"`{cell['value_error_rmse']}` | `{cell['min_ess']}` |"
            )
        lines.append(f"- particle ladder: `{summary['particle_ladder']}`")
        lines.append("")
    lines.extend(
        [
            "## Gradient Summary",
            "",
            "| Particles | Seeds | Mean gradient error norm | SE | RMSE | Mean relative score error |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for count in GRADIENT_PARTICLE_COUNTS:
        cell = payload["gradient_summary"][str(count)]
        lines.append(
            f"| {count} | `{cell['seed_count']}` | `{cell['mean_gradient_error_norm']}` | "
            f"`{cell['gradient_error_norm_standard_error']}` | "
            f"`{cell['gradient_error_norm_rmse']}` | "
            f"`{cell['mean_relative_score_error']}` |"
        )
    lines.extend(
        [
            f"- particle ladder: `{payload['gradient_summary']['particle_ladder']}`",
            "",
            "## Blocked Adapter Details",
            "",
            "| Old lane | Status | Missing items |",
            "| --- | --- | --- |",
        ]
    )
    for blocker in payload["blocked_adapters"]:
        lines.append(
            f"| `{blocker['old_lane_id']}` | `{blocker['status']}` | "
            f"{'; '.join(blocker['missing_items'])} |"
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `{payload['decision']}` | old direct LGSSM and gradient lanes rerun as "
                "Algorithm 1 diagnostic rows; range-bearing lanes classified as adapter blockers | "
                f"`{payload['veto_diagnostics']}` | numerical closeness thresholds are deliberately not promoted in P1 | "
                "Claude read-only P1 review, then P2 contract freeze | "
                "no nonlinear, production, stochastic-score, OT-extension, or calibrated numerical-performance claim |"
            ),
            "",
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: the finite LGSSM rows only show that the reviewed Algorithm 1 core executes on this fixture and particle ladder, not that it is numerically superior or calibrated.",
            "",
            "Result that would overturn the local interpretation: a reviewed rerun finds old-route import leakage, missing Algorithm 1 route fields, nonfinite covariance/determinant diagnostics, or range-bearing rows ranked without adapters.",
            "",
            "Weakest part of the evidence: P1 intentionally does not set a numerical acceptance band from one fixture; exact-oracle calibration remains in later phases.",
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
            f"- gradient seeds: `{payload['run_manifest']['gradient_seed_list']}`",
            f"- gradient particle counts: `{payload['run_manifest']['gradient_particle_counts']}`",
            f"- gradient horizon: `{payload['run_manifest']['gradient_horizon']}`",
            f"- pseudo-time steps: `{payload['run_manifest']['pseudo_time_steps']}`",
            f"- UKF parameters: `{payload['run_manifest']['ukf_parameters']}`",
            f"- route identifiers: `{payload['run_manifest']['algorithm1_route_identifiers']}`",
            f"- data version: `{payload['run_manifest']['data_version']}`",
            f"- plan: `{payload['run_manifest']['plan_path']}`",
            f"- registry: `{payload['run_manifest']['registry_path']}`",
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
            "P1 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _augmented_algorithm1_route(*, resampling_route: str) -> dict[str, str]:
    route = algorithm1_route_identifiers(resampling_route=resampling_route)
    validate_algorithm1_route_identifiers(route)
    route.update(
        {
            "prediction_covariance_route": "ukf_prediction_per_particle_covariance",
            "update_covariance_route": "ukf_update_per_particle_covariance",
            "core_resampling_route": "none",
            "extension_resampling_route": "none",
            "evidence_route_class": "SOURCE_ALGORITHM1_CORE",
        }
    )
    return route


def _row_route_fields(method_id: str, route: dict[str, str]) -> dict[str, str]:
    if method_id == "ledh_pfpf_alg1_ukf_no_resampling_tf":
        return dict(route)
    return {
        "method_generation": "N/A_BASELINE_NOT_ALGORITHM1",
        "flow_source_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "prediction_covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "update_covariance_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "flow_anchor_route": "N/A_BASELINE_NOT_ALGORITHM1",
        "resampling_route": "none",
        "core_resampling_route": "none",
        "extension_resampling_route": "none",
        "evidence_route_class": "BASELINE_NOT_ALGORITHM1",
        "previous_ledh_pfpf_ot_evidence_status": "quarantined",
    }


def _is_augmented_algorithm1_route(route: dict[str, str]) -> bool:
    expected = _augmented_algorithm1_route(resampling_route=route.get("resampling_route", "none"))
    return all(route.get(key) == value for key, value in expected.items())


def _registry_lane_ids(registry: dict[str, Any], *, replacement_phase: str) -> list[str]:
    return [
        row["old_lane_id"]
        for row in registry.get("registry_rows", [])
        if row.get("replacement_phase") == replacement_phase
    ]


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"created_at_utc", "run_manifest", "reproducibility_digest"}
    }
    return stable_digest(stable)


def _nonclaims() -> list[str]:
    return [
        "P1 does not set or pass a calibrated numerical closeness threshold.",
        "P1 value rows do not imply gradient correctness.",
        "P1 fixed-branch gradient rows do not imply stochastic-resampling or HMC-score correctness.",
        "P1 no-resampling rows do not establish classical-resampling or OT-extension correctness.",
        "Range-bearing and range-bearing stress rows are classified as adapter blockers, not negative scientific evidence.",
        "The previous LEDH-PFPF-OT lineage remains quarantined and is not used as Algorithm 1 evidence.",
        "P1 does not establish production readiness, default policy, GPU performance, or broad P44 model coverage.",
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
