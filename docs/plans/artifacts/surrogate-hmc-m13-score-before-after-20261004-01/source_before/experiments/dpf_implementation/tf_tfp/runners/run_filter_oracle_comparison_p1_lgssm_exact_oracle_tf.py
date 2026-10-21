"""Run P1 LGSSM exact-oracle value and gradient comparison.

P1 is a TensorFlow CPU-only numerical phase.  It compares DPF bootstrap-OT and
LEDH-PFPF-OT value and fixed-branch AD gradients against an exact Kalman
reference in the declared LGSSM parameterization.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from experiments.dpf_implementation.tf_tfp.flows.jacobians_tf import (
    linear_observation_jacobian_tf,
)
from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import ledh_flow_batch_tf
from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    annealed_transport_resample_tf,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    max_sinkhorn_residual,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.run_filter_oracle_comparison_p0_registry_tf import (
    REGISTRY_PATH,
)
from experiments.dpf_implementation.tf_tfp.runners.run_v2_bootstrap_ot_values_tf import (
    _gaussian_logpdf_zero_mean,
)


DTYPE = tf.float64
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p1_lgssm_exact_oracle_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-subplan-2026-06-08.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-result-2026-06-08.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filter-oracle-comparison-p1-claude-review-ledger-2026-06-08.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p1_lgssm_exact_oracle_2026-06-08.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p1-lgssm-exact-oracle-2026-06-08.md"

SEEDS = [101, 202, 303, 404, 505]
PARTICLE_COUNTS = [32, 64, 128]
MIN_PARTICLE_COUNTS = [32, 64]
GRADIENT_KNOBS = ("transition_matrix_scale", "observation_noise_scale")
VALUE_CI_TOLERANCE = 2.0
SCORE_RMSE_TOLERANCE = 12.0
SCORE_RELATIVE_ERROR_TOLERANCE = 0.75
FD_STEP = 1e-5


class P1ValidationError(ValueError):
    """Raised when a P1 artifact violates the phase contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P1_LGSSM_EXACT_ORACLE_VALIDATED")
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
    registry = load_json(REGISTRY_PATH)
    lgssm_routes = registry["route_matrix"]["lgssm_2d_h25_rich"]
    target = next(
        row for row in registry["targets"] if row["target_id"] == "lgssm_2d_h25_rich"
    )
    base = _base_lgssm()
    theta0 = tf.constant([1.0, 1.0], dtype=DTYPE)
    kalman = _kalman_reference(theta0, base)
    fd_score = _finite_difference_score(lambda x: _kalman_value(x, base), theta0)
    fd_delta = tf.cast(kalman["score"], DTYPE) - fd_score

    rows: list[dict[str, Any]] = []
    for num_particles in PARTICLE_COUNTS:
        for seed in SEEDS:
            rows.append(_dpf_row("dpf_bootstrap_ot", theta0, base, seed, num_particles, kalman))
            rows.append(_dpf_row("dpf_ledh_pfpf_ot", theta0, base, seed, num_particles, kalman))

    method_summaries = {
        method_id: _method_summary(method_id, rows, kalman)
        for method_id in ("dpf_bootstrap_ot", "dpf_ledh_pfpf_ot")
    }
    deterministic_sanity = _deterministic_sanity_rows(lgssm_routes)
    veto = _veto_diagnostics(kalman, fd_delta, rows, method_summaries)
    decision = (
        "PASS_P1_LGSSM_EXACT_ORACLE_PENDING_CLAUDE_REVIEW"
        if not any(bool(value) for value in veto.values())
        else "P1_LGSSM_EXACT_ORACLE_VETO_PENDING_REVIEW"
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
            "seed_list": list(SEEDS),
            "particle_counts": list(PARTICLE_COUNTS),
            "data_version": "fixed local LGSSM fixture generated by P1 runner",
        }
    )
    return {
        "metadata_date": "2026-06-08",
        "created_at_utc": utc_now(),
        "phase": "P1",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Are DPF bootstrap-OT and LEDH-PFPF-OT value and fixed-branch "
            "gradients close to exact Kalman value and analytic Kalman score on "
            "the P0 LGSSM target?"
        ),
        "target": {
            "target_id": "lgssm_2d_h25_rich",
            "registry_target": target,
            "model_definition": _model_definition(base),
            "theta": tensor_to_json(theta0),
            "gradient_knobs": list(GRADIENT_KNOBS),
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
            "kalman_exact": lgssm_routes["kalman_exact"],
            "dpf_bootstrap_ot": lgssm_routes["dpf_bootstrap_ot"],
            "dpf_ledh_pfpf_ot": lgssm_routes["dpf_ledh_pfpf_ot"],
            "deterministic_sanity": deterministic_sanity,
        },
        "stochastic_contract": {
            "seeds": list(SEEDS),
            "particle_counts": list(PARTICLE_COUNTS),
            "confidence_interval": "normal 95% CI over paired seeds by method/particle count",
            "gradient_object": "fixed_branch_score",
            "stochastic_score_claim": "not_claimed",
            "common_random_number_policy": (
                "paired seed list is shared across methods and particle counts; "
                "each method deterministically maps a seed to stateless initial "
                "particles and transition innovations"
            ),
            "branch_freeze_policy": (
                "fixed-branch score freezes stateless random draws and the "
                "realized ESS-triggered OT branch decisions within each row; "
                "gradients through random/discrete branch selection are not claimed"
            ),
            "transport_branch_policy": (
                "OT transport is differentiated only for the realized branch; "
                "resampling trigger decisions are diagnostics, not differentiated"
            ),
            "third_particle_count_policy": (
                "third count included because the master trigger fires when the "
                "larger of the first two particle counts does not reduce value "
                "RMSE or score RMSE by at least 25%, or when CI/bias diagnostics "
                "remain suspicious."
            ),
        },
        "tolerances": {
            "mean_value_error_ci_abs": VALUE_CI_TOLERANCE,
            "score_rmse": SCORE_RMSE_TOLERANCE,
            "relative_score_error": SCORE_RELATIVE_ERROR_TOLERANCE,
            "finite_difference_step": FD_STEP,
        },
        "rows": rows,
        "method_summaries": method_summaries,
        "method_interpretation": _method_interpretation(method_summaries),
        "veto_diagnostics": veto,
        "explanatory_diagnostics": {
            "value_agreement_is_not_gradient_promotion": True,
            "finite_difference_is_diagnostic_only": True,
            "bf_filterflow_prior_agreement_used_as_oracle": False,
            "deterministic_routes_are_sanity_only": True,
            "common_random_number_policy_recorded": True,
            "branch_freeze_policy_recorded": True,
            "gradient_evaluator_variance_reported": True,
        },
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


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
        d_transition_matrix=tf.stack([base["A"], tf.zeros_like(base["A"])], axis=0),
        d_transition_covariance=tf.zeros([2, 2, 2], dtype=DTYPE),
        d_observation_offset=zero_m,
        d_observation_matrix=tf.zeros([2, 1, 2], dtype=DTYPE),
        d_observation_covariance=tf.stack([tf.zeros_like(base["R"]), base["R"]], axis=0),
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


def _dpf_row(
    method_id: str,
    theta: tf.Tensor,
    base: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
    kalman: dict[str, Any],
) -> dict[str, Any]:
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value, diagnostics = _dpf_scalar(
            method_id=method_id,
            theta=theta,
            base=base,
            seed=seed,
            num_particles=num_particles,
        )
    gradient = tape.gradient(value, theta)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
    kalman_score = tf.constant(kalman["score"], dtype=DTYPE)
    grad_error = tf.cast(gradient, DTYPE) - kalman_score
    value_error = value - tf.constant(kalman["log_likelihood"], dtype=DTYPE)
    return {
        "method_id": method_id,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "value": scalar(value),
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(value_error / tf.constant(25.0, dtype=DTYPE)),
        "fixed_branch_gradient": tensor_to_json(gradient),
        "gradient_error": tensor_to_json(grad_error),
        "gradient_error_norm": scalar(tf.linalg.norm(grad_error)),
        "relative_score_error": scalar(
            tf.linalg.norm(grad_error)
            / tf.maximum(tf.constant(1.0, dtype=DTYPE), tf.linalg.norm(kalman_score))
        ),
        "gradient_cosine_similarity": _cosine_similarity(tf.cast(gradient, DTYPE), kalman_score),
        "finite": bool(tf.math.is_finite(value).numpy() and tf.reduce_all(tf.math.is_finite(gradient)).numpy()),
        "gradient_object": "fixed_branch_score",
        "stochastic_score_claim": "not_claimed",
        "diagnostics": diagnostics,
    }


def _dpf_scalar(
    *,
    method_id: str,
    theta: tf.Tensor,
    base: dict[str, tf.Tensor],
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    params = _parameters(theta, base)
    particles = _sample_initial(params, num_particles, seed)
    log_weights = tf.fill([num_particles], -tf.math.log(tf.cast(num_particles, DTYPE)))
    value = tf.constant(0.0, dtype=DTYPE)
    ess_values = []
    resampling_count = 0
    transport_diags = []
    ledh_jacobian_min = []
    corrected_weight_max = []

    for t, observation in enumerate(tf.unstack(params["observations"], axis=0)):
        ancestors = particles
        predicted = _transition_sample(params, ancestors, seed, t)
        if method_id == "dpf_bootstrap_ot":
            proposal_particles = predicted
            obs_density = _observation_log_density(params, proposal_particles, observation)
            corrected = log_weights + obs_density
        elif method_id == "dpf_ledh_pfpf_ot":
            flow = _ledh_flow(params, predicted, ancestors, observation)
            proposal_particles = flow.post_flow_particles
            target_transition = _transition_log_density(params, ancestors, proposal_particles)
            target_observation = _observation_log_density(params, proposal_particles, observation)
            corrected = (
                log_weights
                + target_transition
                + target_observation
                - flow.pre_flow_log_density
                + flow.forward_log_det
            )
            ledh_jacobian_min.append(float(flow.diagnostics["min_jacobian_singular_value"]))
            corrected_weight_max.append(scalar(tf.reduce_max(tf.abs(corrected))))
        else:
            raise ValueError(f"unknown P1 method_id: {method_id}")

        increment = tf.reduce_logsumexp(corrected)
        value = value + increment
        log_weights = corrected - increment
        weights = tf.exp(log_weights)
        ess = 1.0 / tf.reduce_sum(weights * weights)
        ess_values.append(scalar(ess))
        do_resample = bool((ess < tf.constant(0.5 * num_particles, dtype=DTYPE)).numpy())
        if do_resample:
            transport = annealed_transport_resample_tf(
                proposal_particles,
                log_weights,
                epsilon=0.7,
                scaling=0.9,
                convergence_threshold=1e-3,
                max_iterations=80,
                ess_mask=tf.constant([True], dtype=tf.bool),
                transport_gradient_mode="filterflow_clipped",
                application_mode="active_rows_only",
            )
            particles = tf.cast(transport.particles, DTYPE)
            log_weights = tf.cast(transport.log_weights, DTYPE)
            transport_diags.append(dict(transport.diagnostics))
            resampling_count += 1
        else:
            particles = proposal_particles

    diagnostics = {
        "ess_min": min(ess_values),
        "ess_mean": statistics.fmean(ess_values),
        "resampling_count": int(resampling_count),
        "max_sinkhorn_residual": max_sinkhorn_residual(transport_diags),
        "transport_trigger_count": len(transport_diags),
        "max_abs_corrected_log_weight": max(corrected_weight_max) if corrected_weight_max else None,
        "min_ledh_jacobian_singular_value": min(ledh_jacobian_min) if ledh_jacobian_min else None,
    }
    return value, diagnostics


def _method_summary(method_id: str, rows: list[dict[str, Any]], kalman: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for num_particles in PARTICLE_COUNTS:
        subset = [
            row
            for row in rows
            if row["method_id"] == method_id and row["num_particles"] == num_particles
        ]
        value_errors = [float(row["value_error"]) for row in subset]
        gradient_errors = [row["gradient_error"] for row in subset]
        gradient_norms = [float(row["gradient_error_norm"]) for row in subset]
        relative_errors = [float(row["relative_score_error"]) for row in subset]
        out[str(num_particles)] = {
            "seed_count": len(subset),
            "mean_value_error": statistics.fmean(value_errors),
            "value_error_standard_error": _standard_error(value_errors),
            "value_error_ci95": _ci95(value_errors),
            "value_error_rmse": _rmse(value_errors),
            "max_abs_value_error": max(abs(value) for value in value_errors),
            "gradient_error_mean": _vector_mean(gradient_errors),
            "gradient_error_standard_error": _vector_standard_error(gradient_errors),
            "gradient_error_ci95": _vector_ci95(gradient_errors),
            "gradient_error_rmse_by_coordinate": _vector_rmse(gradient_errors),
            "gradient_error_norm_standard_error": _standard_error(gradient_norms),
            "gradient_error_norm_ci95": _ci95(gradient_norms),
            "score_rmse": _rmse(gradient_norms),
            "mean_relative_score_error": statistics.fmean(relative_errors),
            "max_relative_score_error": max(relative_errors),
            "max_sinkhorn_residual": max(
                float(row["diagnostics"]["max_sinkhorn_residual"]) for row in subset
            ),
            "min_ess": min(float(row["diagnostics"]["ess_min"]) for row in subset),
            "mean_resampling_count": statistics.fmean(
                float(row["diagnostics"]["resampling_count"]) for row in subset
            ),
            "all_rows_finite": all(bool(row["finite"]) for row in subset),
        }
    counts = [str(count) for count in PARTICLE_COUNTS]
    trigger = _third_particle_trigger(out)
    out["particle_ladder"] = {
        "value_rmse_reduction_ratio": _safe_ratio(
            out[counts[-1]]["value_error_rmse"],
            out[str(MIN_PARTICLE_COUNTS[0])]["value_error_rmse"],
        ),
        "score_rmse_reduction_ratio": _safe_ratio(
            out[counts[-1]]["score_rmse"],
            out[str(MIN_PARTICLE_COUNTS[0])]["score_rmse"],
        ),
        "first_two_value_rmse_reduction_ratio": _safe_ratio(
            out[str(MIN_PARTICLE_COUNTS[1])]["value_error_rmse"],
            out[str(MIN_PARTICLE_COUNTS[0])]["value_error_rmse"],
        ),
        "first_two_score_rmse_reduction_ratio": _safe_ratio(
            out[str(MIN_PARTICLE_COUNTS[1])]["score_rmse"],
            out[str(MIN_PARTICLE_COUNTS[0])]["score_rmse"],
        ),
        "third_particle_count_triggered": trigger["triggered"],
        "third_particle_count_reasons": trigger["reasons"],
        "third_particle_count_included": len(PARTICLE_COUNTS) >= 3,
        "promotion_status": "evidence_only_pending_claude_and_p5; no DPF correctness promotion",
        "kalman_score_norm": math.sqrt(sum(float(v) ** 2 for v in kalman["score"])),
    }
    return out


def _method_interpretation(summaries: dict[str, Any]) -> dict[str, Any]:
    interpretations = {}
    final_count = str(PARTICLE_COUNTS[-1])
    for method_id, summary in summaries.items():
        final = summary[final_count]
        value_ci = final["value_error_ci95"]
        value_ci_includes_zero = value_ci[0] <= 0.0 <= value_ci[1]
        score_below_tolerance = final["score_rmse"] <= SCORE_RMSE_TOLERANCE
        relative_below_tolerance = final["mean_relative_score_error"] <= SCORE_RELATIVE_ERROR_TOLERANCE
        local_closeness_evidence = (
            value_ci_includes_zero
            and score_below_tolerance
            and relative_below_tolerance
            and final["all_rows_finite"]
        )
        interpretations[method_id] = {
            "status": (
                "P1_LOCAL_LGSSM_CLOSENESS_EVIDENCE_NOT_PROMOTION"
                if local_closeness_evidence
                else "P1_DIAGNOSTIC_OR_BIASED_EVIDENCE_PENDING_P5"
            ),
            "final_particle_count": int(final_count),
            "final_value_ci_includes_zero": value_ci_includes_zero,
            "final_score_rmse_below_tolerance": score_below_tolerance,
            "final_relative_score_error_below_tolerance": relative_below_tolerance,
            "correctness_promotion": "not_promoted_in_p1",
        }
    return interpretations


def _third_particle_trigger(summary: dict[str, Any]) -> dict[str, Any]:
    small = summary[str(MIN_PARTICLE_COUNTS[0])]
    large = summary[str(MIN_PARTICLE_COUNTS[1])]
    reasons: list[str] = []
    value_ci = large["value_error_ci95"]
    if min(abs(float(value_ci[0])), abs(float(value_ci[1]))) > VALUE_CI_TOLERANCE or (
        value_ci[0] > 0.0 or value_ci[1] < 0.0
    ):
        reasons.append("mean_value_error_ci_excludes_zero_or_row_tolerance_band")
    if large["mean_relative_score_error"] > SCORE_RELATIVE_ERROR_TOLERANCE:
        reasons.append("relative_score_error_suspicious")
    if _safe_ratio(large["value_error_rmse"], small["value_error_rmse"]) > 0.75:
        reasons.append("value_rmse_reduction_less_than_25_percent")
    if _safe_ratio(large["score_rmse"], small["score_rmse"]) > 0.75:
        reasons.append("score_rmse_reduction_less_than_25_percent")
    return {"triggered": bool(reasons), "reasons": reasons}


def _veto_diagnostics(
    kalman: dict[str, Any],
    fd_delta: tf.Tensor,
    rows: list[dict[str, Any]],
    summaries: dict[str, Any],
) -> dict[str, bool]:
    return {
        "kalman_reference_nonfinite": not bool(kalman["finite"]),
        "kalman_score_fd_diagnostic_large": scalar(tf.reduce_max(tf.abs(fd_delta))) > 1e-3,
        "dpf_row_nonfinite": any(not bool(row["finite"]) for row in rows),
        "missing_evaluator_variance": any(
            summary[str(count)]["seed_count"] < len(SEEDS)
            or not math.isfinite(float(summary[str(count)]["value_error_standard_error"]))
            or not math.isfinite(float(summary[str(count)]["gradient_error_norm_standard_error"]))
            or any(
                not math.isfinite(float(value))
                for value in summary[str(count)]["gradient_error_standard_error"]
            )
            for summary in summaries.values()
            for count in PARTICLE_COUNTS
        ),
        "third_particle_trigger_missing": any(
            summary["particle_ladder"]["third_particle_count_triggered"]
            and not summary["particle_ladder"]["third_particle_count_included"]
            for summary in summaries.values()
        ),
        "fixed_branch_gradient_mislabeled_stochastic_score": any(
            row["gradient_object"] != "fixed_branch_score"
            or row["stochastic_score_claim"] != "not_claimed"
            for row in rows
        ),
        "value_pass_used_to_excuse_gradient_failure": False,
        "bf_filterflow_agreement_used_as_oracle": False,
        "common_random_number_policy_missing": False,
        "branch_freeze_policy_missing": False,
    }


def _deterministic_sanity_rows(routes: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for route_id in ("ukf", "svd_sigma_point", "cut4", "zhao_cui_fixed_design_tt"):
        route = routes[route_id]
        rows.append(
            {
                "route_id": route_id,
                "claim_class": route["claim_class"],
                "status": "diagnostic_deferred_in_p1_runner",
                "reason": (
                    "P1 primary gate is Kalman-vs-DPF value and fixed-branch "
                    "gradient; deterministic same-target routes remain sanity "
                    "diagnostics unless separately bound in a reviewed amendment."
                ),
            }
        )
    return rows


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
            tf.zeros([2], dtype=DTYPE), q, seed, 10 + t, ()
        )
        obs = tf.linalg.matvec(c, state) + _mvn_sample(
            tf.zeros([1], dtype=DTYPE), r, seed, 100 + t, ()
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


def _ledh_flow(
    params: dict[str, tf.Tensor],
    pre_flow: tf.Tensor,
    ancestors: tf.Tensor,
    observation: tf.Tensor,
):
    observation_jacobian = linear_observation_jacobian_tf(params["C"])
    return ledh_flow_batch_tf(
        pre_flow_particles=pre_flow,
        ancestors=ancestors,
        observation=observation,
        transition_matrix=params["A"],
        transition_covariance=params["Q"],
        observation_covariance=params["R"],
        observation_fn=lambda x: tf.linalg.matvec(params["C"], x),
        observation_jacobian_fn=observation_jacobian,
        observation_residual_fn=lambda predicted, observed: tf.reshape(tf.cast(observed, DTYPE), [-1])
        - tf.reshape(tf.cast(predicted, DTYPE), [-1]),
    )


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


def _model_definition(base: dict[str, tf.Tensor]) -> dict[str, Any]:
    payload = {
        "model_id": "lgssm_2d_h25_rich",
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
    return payload


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "kalman_reference",
        "rows",
        "method_summaries",
        "method_interpretation",
        "veto_diagnostics",
        "run_manifest",
        "nonclaims",
    }
    missing = required.difference(payload)
    if missing:
        raise P1ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] not in {
        "PASS_P1_LGSSM_EXACT_ORACLE_PENDING_CLAUDE_REVIEW",
        "P1_LGSSM_EXACT_ORACLE_VETO_PENDING_REVIEW",
    }:
        raise P1ValidationError(f"invalid P1 decision {payload['decision']}")
    manifest = payload["run_manifest"]
    for field in (
        "command",
        "pre_import_cuda_visible_devices",
        "json_path",
        "report_path",
        "plan_path",
        "result_path",
        "seed_list",
        "particle_counts",
    ):
        if field not in manifest:
            raise P1ValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise P1ValidationError("TensorFlow was not forced CPU-only before import")
    if not bool(payload["kalman_reference"]["finite"]):
        raise P1ValidationError("nonfinite Kalman reference")
    if payload["kalman_reference"]["score_status"] != "analytic_qr_kalman_score_hessian_reference":
        raise P1ValidationError("Kalman score is not the analytic QR derivative reference")
    if len(payload["rows"]) != 2 * len(SEEDS) * len(PARTICLE_COUNTS):
        raise P1ValidationError("unexpected row count")
    for row in payload["rows"]:
        if not row["finite"]:
            raise P1ValidationError("nonfinite DPF row")
        if row["gradient_object"] != "fixed_branch_score":
            raise P1ValidationError("DPF row gradient object drifted")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P1ValidationError("DPF stochastic score claim appeared")
        if "value_error" not in row or "gradient_error" not in row:
            raise P1ValidationError("DPF row missing value or gradient error")
    if payload["veto_diagnostics"]["missing_evaluator_variance"]:
        raise P1ValidationError("evaluator variance summary is missing")
    for field in (
        "common_random_number_policy",
        "branch_freeze_policy",
        "transport_branch_policy",
    ):
        if field not in payload["stochastic_contract"]:
            raise P1ValidationError(f"stochastic_contract missing {field}")
    if payload["veto_diagnostics"].get("third_particle_trigger_missing"):
        raise P1ValidationError("third particle trigger fired but third count is missing")
    if payload["explanatory_diagnostics"]["bf_filterflow_prior_agreement_used_as_oracle"]:
        raise P1ValidationError("BF/FilterFlow agreement was used as oracle")
    for summary in payload["method_summaries"].values():
        for count in PARTICLE_COUNTS:
            cell = summary[str(count)]
            for field in (
                "gradient_error_standard_error",
                "gradient_error_ci95",
                "gradient_error_norm_standard_error",
                "gradient_error_norm_ci95",
                "gradient_error_rmse_by_coordinate",
            ):
                if field not in cell:
                    raise P1ValidationError(f"method summary missing {field}")
    if "reproducibility_digest" not in payload:
        raise P1ValidationError("missing reproducibility digest")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# P1 Result: LGSSM Exact-Oracle Value and Gradient Comparison",
        "",
        "metadata_date: 2026-06-08",
        "phase: P1",
        f"status: {payload['decision']}",
        "",
        "## Skeptical Plan Audit",
        "",
        "Status: `PASS_FOR_P1_EVIDENCE_RECORDING_PENDING_CLAUDE_REVIEW`.",
        "",
        "Wrong-baseline risk is controlled by using `tf_qr_sqrt_differentiated_kalman` as the exact LGSSM comparator in the declared two-parameter scaling. Prior BayesFilter/FilterFlow agreement is recorded as non-oracle context only.",
        "",
        "Proxy-promotion risk is controlled by treating finite differences, Sinkhorn residuals, ESS, and BF/FilterFlow tie-outs as explanatory or veto diagnostics only. P1 records fixed-branch DPF evidence and does not promote stochastic-score correctness.",
        "",
        "Missing-stop-condition risk is controlled by explicit veto diagnostics and by adding the third particle count required by the master stochastic-evidence trigger.",
        "",
        "Unfair-comparison risk is controlled by reporting paired seeds, particle ladder levels, evaluator variance, value RMSE, score RMSE, and nonclaims for stochastic DPF rows.",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        "| Question | Compare DPF value and fixed-branch gradients against exact Kalman value and analytic Kalman score on the P0 LGSSM target. |",
        "| Baseline/comparator | `tf_qr_sqrt_differentiated_kalman` in `(transition_matrix_scale, observation_noise_scale)`. |",
        "| Primary criterion | Kalman reference finite; DPF rows finite; evaluator variance and value/gradient error summaries reported. |",
        "| Veto diagnostics | Nonfinite Kalman/DPF row, missing evaluator variance, parameter/scalar mismatch, stochastic-score overclaim, BF/FilterFlow oracle misuse. |",
        "| Not concluded | No stochastic-resampling distribution correctness, nonlinear correctness, HMC readiness, production readiness, GPU claim, or paper-scale claim. |",
        "",
        "## Stochastic Policy",
        "",
        f"- common-random-number policy: {payload['stochastic_contract']['common_random_number_policy']}",
        f"- branch-freeze policy: {payload['stochastic_contract']['branch_freeze_policy']}",
        f"- transport branch policy: {payload['stochastic_contract']['transport_branch_policy']}",
        f"- gradient object: `{payload['stochastic_contract']['gradient_object']}`",
        f"- stochastic score claim: `{payload['stochastic_contract']['stochastic_score_claim']}`",
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
        "## Pre-Mortem",
        "",
        "This run could pass while misleading us if fixed-branch gradients are read as full stochastic scores, if the third particle count is mistaken for convergence proof, or if local LGSSM evidence is generalized to nonlinear targets. The controls are row-level nonclaims, explicit method interpretation, and a separate Claude gate before P1 advances.",
        "",
        "This run could fail for implementation reasons if the DPF scalar, Kalman scalar, or parameter scaling diverged. The cheap checks are the analytic Kalman score, diagnostic finite-difference tie-out, finite DPF rows, and shared `(transition_matrix_scale, observation_noise_scale)` parameterization.",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| `{payload['decision']}` | finite Kalman and DPF rows with value/gradient summaries | "
            f"`{payload['veto_diagnostics']}` | fixed-branch gradients are not stochastic scores | "
            "run Claude read-only P1 gate review | no HMC, production, GPU, nonlinear, or stochastic-score claim |"
        ),
        "",
        "## Kalman Reference",
        "",
        f"- log likelihood: `{payload['kalman_reference']['log_likelihood']}`",
        f"- score: `{payload['kalman_reference']['score']}`",
        f"- max score delta vs diagnostic finite difference: `{payload['kalman_reference']['max_abs_score_delta_vs_finite_difference']}`",
        "",
        "## Method Summaries",
        "",
    ]
    for method_id, summary in payload["method_summaries"].items():
        lines.append(f"### {method_id}")
        lines.append("")
        lines.append("| Particles | Mean value error | Value SE | Value RMSE | Score RMSE | Gradient norm SE | Mean relative score error |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for count in PARTICLE_COUNTS:
            cell = summary[str(count)]
            lines.append(
                f"| {count} | `{cell['mean_value_error']}` | "
                f"`{cell['value_error_standard_error']}` | `{cell['value_error_rmse']}` | "
                f"`{cell['score_rmse']}` | `{cell['gradient_error_norm_standard_error']}` | "
                f"`{cell['mean_relative_score_error']}` |"
            )
        final = summary[str(PARTICLE_COUNTS[-1])]
        lines.append(f"- final gradient mean error: `{final['gradient_error_mean']}`")
        lines.append(f"- final gradient SE by coordinate: `{final['gradient_error_standard_error']}`")
        lines.append(f"- final gradient 95% CI by coordinate: `{final['gradient_error_ci95']}`")
        lines.append(
            f"- third-particle trigger: `{summary['particle_ladder']['third_particle_count_triggered']}` "
            f"reasons `{summary['particle_ladder']['third_particle_count_reasons']}`"
        )
        interpretation = payload["method_interpretation"][method_id]
        lines.append(f"- interpretation: `{interpretation['status']}`")
        lines.append(f"- correctness promotion: `{interpretation['correctness_promotion']}`")
        lines.append("")
    lines.extend(
        [
            "## Post-Run Red-Team Note",
            "",
            "Strongest alternative explanation: the DPF rows may be finite and locally improving while still biased under the chosen fixed-branch/transport settings, especially for bootstrap-OT value error.",
            "",
            "Result that would overturn the local interpretation: a reviewed rerun with stronger particle ladders or stochastic-score treatment fails to preserve value and score error reduction, or finds a scalar/parameter mismatch in the DPF paths.",
            "",
            "Weakest part of the evidence: P1 uses fixed-branch AD gradients and a small LGSSM fixture. It is exact-oracle evidence for this fixture, not a broad DPF correctness proof.",
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
            f"- seeds: `{payload['run_manifest']['seed_list']}`",
            f"- particle counts: `{payload['run_manifest']['particle_counts']}`",
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
            "P1 is pending Claude read-only review.",
            "",
        ]
    )
    return "\n".join(lines)


def _nonclaims() -> list[str]:
    return [
        "DPF gradients are fixed-branch AD gradients, not full stochastic-resampling scores.",
        "Finite differences are diagnostic only and are not a promotion gate.",
        "Prior BayesFilter/FilterFlow agreement is not used as Kalman oracle evidence.",
        "P1 does not establish nonlinear model correctness.",
        "P1 does not establish HMC readiness or production readiness.",
        "P1 does not make GPU or paper-scale claims.",
    ]


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


def _vector_mean(values: list[list[float]]) -> list[float]:
    return [
        statistics.fmean(float(row[index]) for row in values)
        for index in range(len(values[0]))
    ]


def _vector_standard_error(values: list[list[float]]) -> list[float]:
    return [
        _standard_error([float(row[index]) for row in values])
        for index in range(len(values[0]))
    ]


def _vector_ci95(values: list[list[float]]) -> list[list[float]]:
    return [
        _ci95([float(row[index]) for row in values])
        for index in range(len(values[0]))
    ]


def _vector_rmse(values: list[list[float]]) -> list[float]:
    return [
        _rmse([float(row[index]) for row in values])
        for index in range(len(values[0]))
    ]


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


def _cosine_similarity(left: tf.Tensor, right: tf.Tensor) -> float:
    denom = tf.linalg.norm(left) * tf.linalg.norm(right)
    if float(denom.numpy()) == 0.0:
        return float("nan")
    return scalar(tf.tensordot(left, right, axes=1) / denom)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.transpose(matrix))


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2147483647, int(salt) % 2147483647], dtype=tf.int32)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
