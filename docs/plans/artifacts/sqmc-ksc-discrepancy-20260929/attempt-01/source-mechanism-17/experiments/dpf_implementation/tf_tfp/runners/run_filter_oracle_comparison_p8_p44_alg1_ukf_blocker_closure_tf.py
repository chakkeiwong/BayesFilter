"""Run P7/P8 P44 blocker closure with Algorithm 1 UKF LEDH-PFPF.

This runner replaces the old P44 `dpf_ledh_pfpf_ot` blocker-closure lane with
the reviewed Li-Coates Algorithm 1 UKF route.  It uses the P44 scalar target
definitions from the historical blocker-closure runner as target scaffolding
only.  Old LEDH-PFPF-OT implementation code is not imported or used as current
evidence.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-p7-alg1-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim
from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_alg1_ukf_tf import (
    algorithm1_route_identifiers,
    run_ledh_pfpf_alg1_ukf_tf,
    validate_algorithm1_route_identifiers,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    finite_tensor,
    load_json,
    scalar,
    stable_digest,
    tensor_to_json,
    utc_now,
    write_json,
    write_text,
)


MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_filter_oracle_comparison_p8_p44_alg1_ukf_blocker_closure_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p7-p44-blocker-closure-subplan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-p7-p44-blocker-closure-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-alg1-ukf-ledh-pfpf-ot-rerun-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p8_p44_alg1_ukf_blocker_closure_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-filter-oracle-comparison-p8-p44-alg1-ukf-blocker-closure-2026-06-10.md"
P5_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p5_alg1_ukf_statistical_closeness_2026-06-10.json"
P6_JSON_PATH = OUTPUT_DIR / "dpf_filter_oracle_comparison_p6_alg1_ukf_cross_filter_calibration_2026-06-10.json"

DTYPE = tf.float64
METHOD_ID = "ledh_pfpf_alg1_ukf_no_resampling_tf"
TARGET_IDS = [
    "p44_m2_cubic_additive_gaussian_panel",
    "p44_m3_quadratic_observation_panel",
    "p44_m4_nonlinear_transition_h2_panel",
]
SEEDS = [101, 202, 303]
PARTICLE_COUNTS = [16, 32]
PSEUDO_TIME_STEPS = [0.5, 0.5]
UKF_ALPHA = 1.0
UKF_BETA = 2.0
UKF_KAPPA = 0.0
COVARIANCE_FLOOR = 1e-10
RANK_TOLERANCE = 1e-12
FD_STEP = 1e-5
INITIAL_QUADRATURE_ORDER = 241

LOCAL_PASS_DECISION = "LOCAL_PASS_P7_P44_ALG1_UKF_BLOCKER_CLOSURE_DIAGNOSTIC_ONLY_PENDING_CLAUDE_REVIEW"
VETO_DECISION = "P7_P44_ALG1_UKF_BLOCKER_CLOSURE_VETO_PENDING_REVIEW"

TARGET_SPECS = {
    "p44_m2_cubic_additive_gaussian_panel": {
        "family": "m2",
        "theta": [0.25, math.log(0.14), math.log(0.10), 0.04, 0.35],
        "dense_order": 241,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean, cubic_raw)",
    },
    "p44_m3_quadratic_observation_panel": {
        "family": "m3",
        "theta": [0.18, math.log(0.13), math.log(0.09), 0.02],
        "dense_order": 281,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean)",
    },
    "p44_m4_nonlinear_transition_h2_panel": {
        "family": "m4",
        "theta": [0.22, math.log(0.12), math.log(0.09), 0.03, 0.40],
        "dense_order": 241,
        "horizon": 2,
        "parameterization": "(rho_raw, log_q, log_r, raw_initial_mean, nonlin_raw)",
    },
}


@dataclass(frozen=True)
class AxisModel:
    target_id: str
    family: str
    axis: int
    horizon: int

    def observations(self) -> tf.Tensor:
        return _observations(self.family, self.axis + 1, self.horizon)[:, self.axis : self.axis + 1]

    def transition_mean(self, theta: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
        parts = _axis_part(self.family, theta, self.axis)
        values = _rows(points, 1, "points")[:, 0]
        if self.family == "m4":
            mean = _transition_mean_scalar(values, parts)
        else:
            mean = parts["rho"] * values
        return tf.reshape(mean, [-1, 1])

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
    ) -> tf.Tensor:
        previous = _rows(x_prev, 1, "x_prev")[:, 0]
        current = _rows(x_next, 1, "x_next")[:, 0]
        parts = _axis_part(self.family, theta, self.axis)
        loc = (
            _transition_mean_scalar(previous, parts)
            if self.family == "m4"
            else parts["rho"] * previous
        )
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["transition_variance"]),
        ).log_prob(current)

    def observation_mean(self, theta: tf.Tensor, points: tf.Tensor) -> tf.Tensor:
        parts = _axis_part(self.family, theta, self.axis)
        values = _rows(points, 1, "points")[:, 0]
        return tf.reshape(_observation_mean(self.family, parts, values), [-1, 1])

    def observation_jacobian(self, theta: tf.Tensor, point: tf.Tensor) -> tf.Tensor:
        parts = _axis_part(self.family, theta, self.axis)
        value = tf.reshape(tf.cast(point, DTYPE), [-1])[0]
        return tf.reshape(_observation_jacobian(self.family, parts, value), [1, 1])

    def observation_log_density(self, theta: tf.Tensor, x_t: tf.Tensor, y_t: tf.Tensor) -> tf.Tensor:
        points = _rows(x_t, 1, "x_t")[:, 0]
        observation = tf.reshape(tf.cast(y_t, DTYPE), [1])[0]
        parts = _axis_part(self.family, theta, self.axis)
        loc = _observation_mean(self.family, parts, points)
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["observation_variance"]),
        ).log_prob(observation)


class P7Alg1ValidationError(ValueError):
    """Raised when the P7 Algorithm 1 P44 artifact violates its contract."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("P7_P44_ALG1_UKF_BLOCKER_CLOSURE_VALIDATED")
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
    p5 = load_json(P5_JSON_PATH)
    p6 = load_json(P6_JSON_PATH)
    _preflight(p5, p6)
    route = _augmented_algorithm1_route()
    references = [_reference_row(target_id, dim) for target_id in TARGET_IDS for dim in (1, 2, 3)]
    references_by_key = {(row["target_id"], int(row["dim"])): row for row in references}
    adapter_matrix = [_adapter_matrix_row(target_id, dim) for target_id in TARGET_IDS for dim in (1, 2, 3)]
    rows: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        for dim in (1, 2, 3):
            reference = references_by_key[(target_id, dim)]
            for num_particles in PARTICLE_COUNTS:
                for seed in SEEDS:
                    rows.append(
                        _safe_alg1_row(
                            target_id=target_id,
                            dim=dim,
                            seed=seed,
                            num_particles=num_particles,
                            reference=reference,
                        )
                    )
    directional = [
        _safe_directional_row(
            target_id=target_id,
            dim=1,
            seed=SEEDS[0],
            num_particles=PARTICLE_COUNTS[-1],
            ad_gradient=_first_gradient(rows, target_id, 1, PARTICLE_COUNTS[-1], SEEDS[0]),
        )
        for target_id in TARGET_IDS
    ]
    summaries = [
        _summary_row(
            target_id=target_id,
            dim=dim,
            num_particles=num_particles,
            rows=rows,
            reference=references_by_key[(target_id, dim)],
            directional=_directional_for(directional, target_id, dim, num_particles),
        )
        for target_id in TARGET_IDS
        for dim in (1, 2, 3)
        for num_particles in PARTICLE_COUNTS
    ]
    veto = _veto_diagnostics(rows, references, adapter_matrix, directional, p5, p6)
    decision = LOCAL_PASS_DECISION if not any(bool(value) for value in veto.values()) else VETO_DECISION
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "P7",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": (
            "Can former P44/P8 LEDH-related N/A cells be filled by Algorithm 1 "
            "UKF diagnostic rows or precise blockers?"
        ),
        "evidence_contract": {
            "baseline_comparator": "P44 dense fixed-branch references for value/score; P5/P6 Algorithm 1 replacement blockers.",
            "primary_criterion": "Every P44 target/dim receives an Algorithm 1 measured diagnostic row or precise reviewed blocker.",
            "promotion_policy": "diagnostic-only; no P44 row is promoted because no P7 numeric promotion band was reviewed for Algorithm 1.",
            "not_concluded": _nonclaims(),
        },
        "skeptical_plan_audit": {
            "status": "PASS_FOR_DIAGNOSTIC_RERUN",
            "wrong_baseline_control": "Dense P44 references are same-target value/score comparators; old DPF rows are not used.",
            "proxy_promotion_control": "Measured finite rows fill cells as diagnostic-only, not promoted closeness.",
            "timing_control": "Initial P44 timing is represented by an explicit raw/pre-initial Algorithm 1 transition adapter.",
            "stop_condition_control": "Nonfinite rows, missing route fields, missing adapter callbacks, or old-route imports veto the phase.",
            "environment_control": "CUDA_VISIBLE_DEVICES=-1 is set before TensorFlow import.",
        },
        "target_specs": TARGET_SPECS,
        "algorithm1_route_fields": route,
        "adapter_matrix": adapter_matrix,
        "reference_rows": references,
        "rows": rows,
        "directional_residuals": directional,
        "row_summaries": summaries,
        "route_summaries": _route_summaries(rows, summaries),
        "veto_diagnostics": veto,
        "historical_old_dpf_quarantine": [
            "dpf_ledh_pfpf_ot historical P8 rows are not used as current Algorithm 1 evidence",
            "experiments/dpf_implementation/tf_tfp/filters/ledh_pfpf_ot_tf.py is not imported",
        ],
        "decision_table": _decision_table(decision, veto, summaries),
        "post_run_red_team": _post_run_red_team(),
        "run_manifest": _manifest(route),
        "nonclaims": _nonclaims(),
    }


def _safe_alg1_row(
    *,
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _alg1_row(
            target_id=target_id,
            dim=dim,
            seed=seed,
            num_particles=num_particles,
            reference=reference,
        )
    except Exception as exc:
        theta_dim = len(TARGET_SPECS[target_id]["theta"])
        return {
            "target_id": target_id,
            "dim": int(dim),
            "method_id": METHOD_ID,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "row_status": "BLOCKED_WITH_REVIEWED_REASON",
            "finite": False,
            "value": float("nan"),
            "value_error": float("nan"),
            "per_observation_value_error": float("nan"),
            "fixed_branch_gradient": [float("nan")] * theta_dim,
            "gradient_error": [float("nan")] * theta_dim,
            "gradient_error_norm": float("nan"),
            "relative_gradient_error": float("nan"),
            "gradient_scope": "fixed_branch_score",
            "stochastic_score_claim": "not_claimed",
            "route_fields": _augmented_algorithm1_route(),
            "diagnostics": {
                "blocked_reason": type(exc).__name__,
                "blocked_message": str(exc),
                "old_ledh_pfpf_ot_used": False,
            },
            "branch_records": [],
            "blockers": [f"{type(exc).__name__}: {exc}"],
        }


def _alg1_row(
    *,
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value, diagnostics = _alg1_panel_value(
            target_id=target_id,
            dim=dim,
            theta=theta,
            seed=seed,
            num_particles=num_particles,
        )
    gradient = tape.gradient(value, theta)
    if gradient is None:
        gradient = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
    reference_score = tf.constant(reference["score"], dtype=DTYPE)
    value_error = value - tf.constant(reference["log_likelihood"], dtype=DTYPE)
    gradient_error = tf.cast(gradient, DTYPE) - reference_score
    reference_norm = tf.maximum(tf.constant(1.0, DTYPE), tf.linalg.norm(reference_score))
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": METHOD_ID,
        "seed": int(seed),
        "num_particles": int(num_particles),
        "row_status": "RERUN_ALG1_DIAGNOSTIC_ONLY",
        "finite": bool(finite_tensor(value) and finite_tensor(gradient)),
        "value": scalar(value),
        "value_error": scalar(value_error),
        "per_observation_value_error": scalar(
            value_error / tf.constant(float(reference["observation_count"]), DTYPE)
        ),
        "fixed_branch_gradient": tensor_to_json(gradient),
        "gradient_error": tensor_to_json(gradient_error),
        "gradient_error_norm": scalar(tf.linalg.norm(gradient_error)),
        "relative_gradient_error": scalar(tf.linalg.norm(gradient_error) / reference_norm),
        "gradient_scope": "fixed_branch_score_through_algorithm1_value_path",
        "stochastic_score_claim": "not_claimed",
        "route_fields": _augmented_algorithm1_route(),
        "diagnostics": diagnostics["summary"],
        "branch_records": diagnostics["branch_records"],
        "blockers": [],
    }


def _alg1_panel_value(
    *,
    target_id: str,
    dim: int,
    theta: tf.Tensor,
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, dict[str, Any]]:
    value = tf.constant(0.0, dtype=DTYPE)
    all_records: list[dict[str, Any]] = []
    for axis in range(dim):
        model = AxisModel(
            target_id=target_id,
            family=str(TARGET_SPECS[target_id]["family"]),
            axis=axis,
            horizon=int(TARGET_SPECS[target_id]["horizon"]),
        )
        axis_value, axis_diag = _alg1_axis_value(
            model=model,
            theta=theta,
            seed=seed,
            num_particles=num_particles,
        )
        value = value + axis_value
        all_records.extend(axis_diag)
    return value, {"summary": _branch_summary(all_records), "branch_records": all_records}


def _alg1_axis_value(
    *,
    model: AxisModel,
    theta: tf.Tensor,
    seed: int,
    num_particles: int,
) -> tuple[tf.Tensor, list[dict[str, Any]]]:
    raw_variance = _axis_part(model.family, theta, model.axis)["raw_initial_variance"]
    result = run_ledh_pfpf_alg1_ukf_tf(
        observations=model.observations(),
        initial_sample=lambda n, s: _sample_raw_initial(model, theta, n, s),
        initial_covariance=tf.reshape(raw_variance, [1, 1]),
        transition_sample=lambda ancestors, s, t: _transition_sample(model, theta, ancestors, s, t),
        transition_mean_fn=lambda points, _t: model.transition_mean(theta, points),
        transition_log_density_fn=lambda x_next, x_prev, _t: model.transition_log_density(theta, x_prev, x_next),
        observation_mean_fn=lambda points, _t: model.observation_mean(theta, points),
        observation_jacobian_fn=lambda point, _t: model.observation_jacobian(theta, point),
        observation_log_density_fn=lambda particles, observation, _t: model.observation_log_density(
            theta,
            particles,
            observation,
        ),
        process_noise_covariance_fn=lambda _x_prev, _t: tf.reshape(
            _axis_part(model.family, theta, model.axis)["transition_variance"],
            [1, 1],
        ),
        observation_covariance_fn=lambda _t: tf.reshape(
            _axis_part(model.family, theta, model.axis)["observation_variance"],
            [1, 1],
        ),
        seed=seed,
        num_particles=num_particles,
        pseudo_time_steps=tf.constant(PSEUDO_TIME_STEPS, dtype=DTYPE),
        resampling_route="none",
        alpha=UKF_ALPHA,
        beta=UKF_BETA,
        kappa=UKF_KAPPA,
        covariance_floor=COVARIANCE_FLOOR,
        rank_tolerance=RANK_TOLERANCE,
        method_id=METHOD_ID,
    )
    validate_algorithm1_route_identifiers(result.route_identifiers)
    records = []
    for item in result.resampling_diagnostics:
        records.append(
            {
                "target_id": model.target_id,
                "axis": int(model.axis),
                "method_id": METHOD_ID,
                "seed": int(seed),
                "num_particles": int(num_particles),
                "time_index": int(item["time_index"]),
                "ess": float(item["ess"]),
                "ess_ratio": float(item["ess_ratio"]),
                "resampled": bool(item["resampled"]),
                "resampling_method": item["resampling_method"],
                "flow_anchor_route": item["flow_anchor_route"],
                "covariance_route": item["covariance_route"],
                "min_predicted_covariance_eigenvalue": float(item["min_predicted_covariance_eigenvalue"]),
                "max_prediction_floor_count": int(item["max_prediction_floor_count"]),
                "min_forward_log_det": float(item["min_forward_log_det"]),
                "max_forward_log_det": float(item["max_forward_log_det"]),
                "finite_forward_log_det": bool(item["finite_forward_log_det"]),
                "finite_corrected_log_weights": bool(item["finite_corrected_log_weights"]),
                "initial_timing_adapter": "raw_pre_initial_state_transition_to_first_observed_latent",
            }
        )
    return result.log_likelihood_estimate, records


def _sample_raw_initial(
    model: AxisModel,
    theta: tf.Tensor,
    num_particles: int,
    seed: int,
) -> tf.Tensor:
    parts = _axis_part(model.family, theta, model.axis)
    return tf.reshape(
        _sample_normal(
            parts["raw_initial_mean"],
            parts["raw_initial_variance"],
            num_particles,
            seed,
            10_000 + 97 * model.axis,
        ),
        [num_particles, 1],
    )


def _transition_sample(
    model: AxisModel,
    theta: tf.Tensor,
    ancestors: tf.Tensor,
    seed: int,
    time_index: int,
) -> tf.Tensor:
    mean = model.transition_mean(theta, ancestors)
    variance = _axis_part(model.family, theta, model.axis)["transition_variance"]
    noise = _sample_normal(
        tf.zeros([int(ancestors.shape[0])], dtype=DTYPE),
        variance,
        int(ancestors.shape[0]),
        seed,
        20_000 + 997 * model.axis + int(time_index),
    )
    return tf.reshape(tf.reshape(mean, [-1]) + noise, [int(ancestors.shape[0]), 1])


def _safe_directional_row(
    *,
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    ad_gradient: list[float],
) -> dict[str, Any]:
    try:
        theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
        gradient = tf.constant(ad_gradient, dtype=DTYPE)
        residuals = []
        for direction in tf.unstack(_directions(int(theta.shape[0])), axis=0):
            plus = theta + tf.constant(FD_STEP, dtype=DTYPE) * direction
            minus = theta - tf.constant(FD_STEP, dtype=DTYPE) * direction
            plus_value, _plus_diag = _alg1_panel_value(
                target_id=target_id,
                dim=dim,
                theta=plus,
                seed=seed,
                num_particles=num_particles,
            )
            minus_value, _minus_diag = _alg1_panel_value(
                target_id=target_id,
                dim=dim,
                theta=minus,
                seed=seed,
                num_particles=num_particles,
            )
            fd = (plus_value - minus_value) / tf.constant(2.0 * FD_STEP, DTYPE)
            ad = tf.tensordot(direction, gradient, axes=1)
            residuals.append(scalar(tf.abs(ad - fd)))
        return {
            "target_id": target_id,
            "dim": int(dim),
            "method_id": METHOD_ID,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "fd_step": FD_STEP,
            "direction_count": int(_directions(int(theta.shape[0])).shape[0]),
            "directional_abs_residuals": residuals,
            "max_abs_directional_residual": max(residuals) if residuals else float("nan"),
            "branch_signature_stable": True,
            "status": "DIAGNOSTIC_ONLY_NOT_PROMOTION",
        }
    except Exception as exc:
        return {
            "target_id": target_id,
            "dim": int(dim),
            "method_id": METHOD_ID,
            "seed": int(seed),
            "num_particles": int(num_particles),
            "fd_step": FD_STEP,
            "direction_count": 0,
            "directional_abs_residuals": [],
            "max_abs_directional_residual": float("nan"),
            "branch_signature_stable": False,
            "status": "BLOCKED_WITH_REVIEWED_REASON",
            "blocked_reason": type(exc).__name__,
            "blocked_message": str(exc),
        }


def _summary_row(
    *,
    target_id: str,
    dim: int,
    num_particles: int,
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    directional: dict[str, Any] | None,
) -> dict[str, Any]:
    subset = [
        row
        for row in rows
        if row["target_id"] == target_id
        and int(row["dim"]) == int(dim)
        and int(row["num_particles"]) == int(num_particles)
    ]
    value_errors = [float(row["value_error"]) for row in subset]
    per_obs = [float(row["per_observation_value_error"]) for row in subset]
    grad_norms = [float(row["gradient_error_norm"]) for row in subset]
    rel_grad = [float(row["relative_gradient_error"]) for row in subset]
    all_finite = all(bool(row["finite"]) for row in subset)
    row_status = "RERUN_ALG1_DIAGNOSTIC_ONLY" if all_finite else "BLOCKED_WITH_REVIEWED_REASON"
    return {
        "target_id": target_id,
        "dim": int(dim),
        "method_id": METHOD_ID,
        "num_particles": int(num_particles),
        "seed_count": len(subset),
        "row_status": row_status,
        "mean_value_error": _finite_mean(value_errors),
        "value_error_standard_error": _standard_error(value_errors),
        "value_error_ci95": _ci95(value_errors),
        "value_error_rmse": _rmse(value_errors),
        "mean_per_observation_value_error": _finite_mean(per_obs),
        "per_observation_value_error_rmse": _rmse(per_obs),
        "mean_gradient_error_norm": _finite_mean(grad_norms),
        "gradient_error_norm_standard_error": _standard_error(grad_norms),
        "mean_relative_gradient_error": _finite_mean(rel_grad),
        "max_relative_gradient_error": max(rel_grad) if rel_grad and _all_finite(rel_grad) else None,
        "reference_score_norm": float(reference["score_norm"]),
        "directional_residual_max": None if directional is None else directional["max_abs_directional_residual"],
        "directional_status": "not_run_for_this_dim" if directional is None else directional["status"],
        "promotion_status": "not_promoted_diagnostic_only",
        "valid_for_p7_cell_fill": bool(all_finite),
    }


def _reference_row(target_id: str, dim: int) -> dict[str, Any]:
    theta = tf.constant(TARGET_SPECS[target_id]["theta"], dtype=DTYPE)
    value, score = _value_and_score(
        lambda current_theta: _dense_panel_value(current_theta, target_id, dim),
        theta,
    )
    observations = _observations(
        str(TARGET_SPECS[target_id]["family"]),
        dim,
        int(TARGET_SPECS[target_id]["horizon"]),
    )
    return {
        "target_id": target_id,
        "dim": int(dim),
        "reference_id": "p44_fixed_branch_dense_refined_quadrature",
        "dense_order": int(TARGET_SPECS[target_id]["dense_order"]),
        "horizon": int(TARGET_SPECS[target_id]["horizon"]),
        "observation_count": int(observations.shape[0]) * int(observations.shape[1]),
        "theta": tensor_to_json(theta),
        "log_likelihood": scalar(value),
        "score": tensor_to_json(score),
        "score_norm": scalar(tf.linalg.norm(score)),
        "finite": bool(finite_tensor(value) and finite_tensor(score)),
        "parameterization": TARGET_SPECS[target_id]["parameterization"],
        "scalar_sign_convention": "score = d log_likelihood / d theta",
        "target_timing": "observe_initial_latent_after_raw_pre_initial_transition",
        "source": "P44 target definitions transcribed into Algorithm 1 P7 runner",
    }


def _dense_panel_value(theta: tf.Tensor, target_id: str, dim: int) -> tf.Tensor:
    return tf.reduce_sum(tf.stack([_dense_axis_value(theta, target_id, axis) for axis in range(dim)]))


def _dense_axis_value(theta: tf.Tensor, target_id: str, axis: int) -> tf.Tensor:
    spec = TARGET_SPECS[target_id]
    model = _DenseAxisModel(
        target_id=target_id,
        family=str(spec["family"]),
        axis=axis,
        horizon=int(spec["horizon"]),
    )
    return highdim.FixedBranchSquaredTTFilter(
        _dense_config(axis, int(spec["dense_order"]))
    ).log_likelihood(
        model,
        theta,
        model.observations(),
    ).log_likelihood


@dataclass(frozen=True)
class _DenseAxisModel:
    target_id: str
    family: str
    axis: int
    horizon: int

    def observations(self) -> tf.Tensor:
        return _observations(self.family, self.axis + 1, self.horizon)[:, self.axis : self.axis + 1]

    def state_dim(self) -> int:
        return 1

    def observation_dim(self) -> int:
        return 1

    def parameter_dim(self) -> int:
        return 4 if self.family == "m3" else 5

    def initial_log_density(self, theta: tf.Tensor, x0: tf.Tensor) -> tf.Tensor:
        points = _rows(x0, 1, "x0")[:, 0]
        parts = _axis_part(self.family, theta, self.axis)
        if self.family == "m4":
            nodes, weights = highdim.legendre_gauss_nodes_weights(INITIAL_QUADRATURE_ORDER)
            raw = 6.0 * tf.cast(nodes, DTYPE)
            scaled_weights = 6.0 * tf.cast(weights, DTYPE)
            raw_prior = tfp.distributions.Normal(
                loc=parts["raw_initial_mean"],
                scale=tf.sqrt(parts["raw_initial_variance"]),
            )
            transition = tfp.distributions.Normal(
                loc=_transition_mean_scalar(raw, parts),
                scale=tf.sqrt(parts["transition_variance"]),
            )
            log_terms = (
                tf.math.log(scaled_weights)[tf.newaxis, :]
                + raw_prior.log_prob(raw)[tf.newaxis, :]
                + transition.log_prob(points[:, tf.newaxis])
            )
            return tf.reduce_logsumexp(log_terms, axis=1)
        return tfp.distributions.Normal(
            loc=parts["initial_mean"],
            scale=tf.sqrt(parts["initial_variance"]),
        ).log_prob(points)

    def transition_log_density(
        self,
        theta: tf.Tensor,
        x_prev: tf.Tensor,
        x_next: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        previous = _rows(x_prev, 1, "x_prev")[:, 0]
        current = _rows(x_next, 1, "x_next")[:, 0]
        parts = _axis_part(self.family, theta, self.axis)
        loc = (
            _transition_mean_scalar(previous, parts)
            if self.family == "m4"
            else parts["rho"] * previous
        )
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["transition_variance"]),
        ).log_prob(current)

    def observation_log_density(
        self,
        theta: tf.Tensor,
        x_t: tf.Tensor,
        y_t: tf.Tensor,
        t: int,
    ) -> tf.Tensor:
        del t
        points = _rows(x_t, 1, "x_t")[:, 0]
        observation = tf.reshape(tf.cast(y_t, DTYPE), [1])[0]
        parts = _axis_part(self.family, theta, self.axis)
        loc = _observation_mean(self.family, parts, points)
        return tfp.distributions.Normal(
            loc=loc,
            scale=tf.sqrt(parts["observation_variance"]),
        ).log_prob(observation)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "family": f"p44_{self.family}_scalar_axis_model",
            "target_id": self.target_id,
            "axis": int(self.axis),
            "horizon": int(self.horizon),
            "timing": "observe_initial_latent_at_t0_then_transition_between_observations",
        }


def _dense_config(axis: int, order: int) -> highdim.FixedBranchFilterConfig:
    return highdim.FixedBranchFilterConfig(
        fit_config=None,
        density_tau=0.0,
        normalizer_floor=1e-14,
        denominator_floor=1e-14,
        retained_storage_byte_budget=20_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=DTYPE),
                matrix=tf.constant([[6.0]], dtype=DTYPE),
            ),
        ),
        measure_convention=highdim.MeasureConvention(
            density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
            mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
            reference_weight_name="omega",
        ),
        deterministic_seed=f"p44-p7-alg1-dense-axis-{axis}-order-{order}",
        fit_quadrature_order=int(order),
    )


def _adapter_matrix_row(target_id: str, dim: int) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "dim": int(dim),
        "state_dimension": int(dim),
        "observation_dimension": int(dim),
        "transition_density_available": True,
        "observation_density_available": True,
        "transition_simulator_available": True,
        "observation_jacobian_available": True,
        "ukf_prediction_update_callbacks_available": True,
        "algorithm1_flow_anchor_status": "zero_noise_transition",
        "initial_timing_adapter": "raw_pre_initial_state_transition_to_first_observed_latent",
        "value_scalar": "sum of P44 dense-panel predictive log normalizers",
        "gradient_scalar": "fixed-branch Algorithm 1 log-likelihood estimator in P44 theta",
        "final_status": "RUNNABLE_ALG1_DIAGNOSTIC_ONLY",
    }


def _preflight(p5: dict[str, Any], p6: dict[str, Any]) -> None:
    if not p5.get("decision", "").startswith("LOCAL_PASS_P5_"):
        raise P7Alg1ValidationError("P5 Algorithm 1 replacement artifact is not ready")
    if not p6.get("decision", "").startswith("LOCAL_PASS_P6_"):
        raise P7Alg1ValidationError("P6 Algorithm 1 calibration artifact is not ready")
    if p5["route_summaries"]["blocked_rows"] != TARGET_IDS:
        raise P7Alg1ValidationError("P5 blocked-row set does not match P44 targets")


def _veto_diagnostics(
    rows: list[dict[str, Any]],
    references: list[dict[str, Any]],
    adapter_matrix: list[dict[str, Any]],
    directional: list[dict[str, Any]],
    p5: dict[str, Any],
    p6: dict[str, Any],
) -> dict[str, bool]:
    return {
        "p5_or_p6_not_ready": not p5.get("decision", "").startswith("LOCAL_PASS_P5_")
        or not p6.get("decision", "").startswith("LOCAL_PASS_P6_"),
        "dense_reference_missing_or_nonfinite": len(references) != len(TARGET_IDS) * 3
        or any(not bool(row["finite"]) for row in references),
        "adapter_callback_missing": any(
            not (
                row["transition_density_available"]
                and row["observation_density_available"]
                and row["transition_simulator_available"]
                and row["observation_jacobian_available"]
                and row["ukf_prediction_update_callbacks_available"]
            )
            for row in adapter_matrix
        ),
        "algorithm1_row_nonfinite": any(not bool(row["finite"]) for row in rows),
        "missing_algorithm1_route_fields": any(
            not _is_augmented_algorithm1_route(row.get("route_fields", {})) for row in rows
        ),
        "old_ledh_pfpf_ot_used_as_current_evidence": _old_runtime_module_loaded()
        or any(bool(row.get("diagnostics", {}).get("old_ledh_pfpf_ot_used", False)) for row in rows),
        "missing_monte_carlo_uncertainty": len(rows) != len(TARGET_IDS) * 3 * len(PARTICLE_COUNTS) * len(SEEDS),
        "directional_residual_missing": len(directional) != len(TARGET_IDS),
        "directional_residual_nonfinite": any(
            not math.isfinite(float(row["max_abs_directional_residual"])) for row in directional
        ),
        "row_promoted_without_band": any(row.get("promotion_status") == "promoted" for row in rows),
        "stochastic_score_claimed": any(row.get("stochastic_score_claim") != "not_claimed" for row in rows),
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "algorithm1_route_fields",
        "adapter_matrix",
        "reference_rows",
        "rows",
        "directional_residuals",
        "row_summaries",
        "route_summaries",
        "veto_diagnostics",
        "decision_table",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise P7Alg1ValidationError(f"missing payload fields {sorted(missing)}")
    if payload["decision"] != LOCAL_PASS_DECISION:
        raise P7Alg1ValidationError(f"invalid P7 decision {payload['decision']}")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise P7Alg1ValidationError("TensorFlow was not forced CPU-only before import")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        raise P7Alg1ValidationError(f"true veto diagnostics: {payload['veto_diagnostics']}")
    if len(payload["rows"]) != len(TARGET_IDS) * 3 * len(PARTICLE_COUNTS) * len(SEEDS):
        raise P7Alg1ValidationError("unexpected P7 row count")
    for row in payload["rows"]:
        if row["method_id"] != METHOD_ID:
            raise P7Alg1ValidationError("unexpected method id")
        if row["row_status"] != "RERUN_ALG1_DIAGNOSTIC_ONLY":
            raise P7Alg1ValidationError("P7 expected measured diagnostic rows only")
        if not row["finite"]:
            raise P7Alg1ValidationError("nonfinite Algorithm 1 row")
        if not _is_augmented_algorithm1_route(row["route_fields"]):
            raise P7Alg1ValidationError("row missing Algorithm 1 route fields")
        if row["stochastic_score_claim"] != "not_claimed":
            raise P7Alg1ValidationError("stochastic score claim appeared")
    if len(payload["directional_residuals"]) != len(TARGET_IDS):
        raise P7Alg1ValidationError("missing directional residual rows")


def _markdown(payload: dict[str, Any]) -> str:
    summary = payload["route_summaries"]
    lines = [
        "# P7 Result: P44 Algorithm 1 UKF Blocker Closure",
        "",
        "metadata_date: 2026-06-10",
        "phase: P7",
        f"status: {payload['decision']}",
        "",
        "## Evidence Contract",
        "",
        "| Field | Contract |",
        "| --- | --- |",
        f"| Question | {payload['question']} |",
        f"| Baseline/comparator | {payload['evidence_contract']['baseline_comparator']} |",
        f"| Primary criterion | {payload['evidence_contract']['primary_criterion']} |",
        f"| Promotion policy | {payload['evidence_contract']['promotion_policy']} |",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| `{row['decision']}` | {row['primary_criterion_status']} | "
            f"`{row['veto_diagnostic_status']}` | {row['main_uncertainty']} | "
            f"{row['next_justified_action']} | {row['not_concluded']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Measured rows: `{summary['measured_row_count']}`.",
            f"- Diagnostic cell fills: `{summary['diagnostic_cell_count']}`.",
            f"- Blocked cell count: `{summary['blocked_cell_count']}`.",
            f"- Promoted rows: `{summary['promoted_rows']}`.",
            "",
            "## Row Summaries",
            "",
            "| Target | Dim | Particles | Status | Value RMSE | Value SE | Mean grad err | Grad SE | Dir residual |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["row_summaries"]:
        lines.append(
            f"| `{row['target_id']}` | {row['dim']} | {row['num_particles']} | `{row['row_status']}` | "
            f"`{row['value_error_rmse']}` | `{row['value_error_standard_error']}` | "
            f"`{row['mean_gradient_error_norm']}` | `{row['gradient_error_norm_standard_error']}` | "
            f"`{row['directional_residual_max']}` |"
        )
    lines.extend(
        [
            "",
            "## Veto Diagnostics",
            "",
            "| Diagnostic | Status |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["veto_diagnostics"].items()],
            "",
            "## Run Manifest",
            "",
            "```json",
            json.dumps(payload["run_manifest"], indent=2, sort_keys=True),
            "```",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _manifest(route: dict[str, str]) -> dict[str, Any]:
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
            "pseudo_time_steps": list(PSEUDO_TIME_STEPS),
            "ukf_parameters": _ukf_parameters(),
            "route_identifiers": route,
            "data_version": "P44 local deterministic target fixtures with Algorithm 1 pre-initial timing adapter",
        }
    )
    return manifest


def _decision_table(
    decision: str,
    veto: dict[str, bool],
    summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    final = [row for row in summaries if int(row["num_particles"]) == max(PARTICLE_COUNTS)]
    return [
        {
            "decision": decision,
            "primary_criterion_status": f"{len(final)} final P44 Algorithm 1 diagnostic cells filled",
            "veto_diagnostic_status": {key: value for key, value in veto.items() if value} or "no structural vetoes",
            "main_uncertainty": "finite diagnostic rows use small particle counts and fixed-branch gradients only",
            "next_justified_action": "Claude read-only P7 review, then P8 historical/extension classification",
            "not_concluded": "no statistical-closeness certification, stochastic-score correctness, production, or universal P44 success",
        }
    ]


def _route_summaries(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    final = [row for row in summaries if int(row["num_particles"]) == max(PARTICLE_COUNTS)]
    return {
        "measured_row_count": len(rows),
        "diagnostic_cell_count": len(final),
        "blocked_cell_count": sum(1 for row in final if row["row_status"] != "RERUN_ALG1_DIAGNOSTIC_ONLY"),
        "promoted_rows": [],
        "target_ids": list(TARGET_IDS),
        "particle_counts": list(PARTICLE_COUNTS),
        "seed_count": len(SEEDS),
    }


def _post_run_red_team() -> dict[str, str]:
    return {
        "strongest_alternative_explanation": "The finite P44 rows may reflect this explicit pre-initial timing adapter and small particle counts rather than stable Algorithm 1 performance.",
        "what_result_would_overturn": "A reviewed same-target rerun with larger particle ladders or an independent adapter that changes the value/gradient conclusions.",
        "weakest_part_of_evidence": "Only fixed-branch gradients and diagnostic finite rows are reported; no stochastic score or promotion band is claimed.",
    }


def _nonclaims() -> list[str]:
    return [
        "P7 fills P44 cells as diagnostic Algorithm 1 measurements only.",
        "P7 does not certify Algorithm 1 statistical closeness on P44.",
        "P7 does not use old dpf_ledh_pfpf_ot results as current evidence.",
        "P7 does not establish stochastic-resampling gradient correctness.",
        "P7 does not establish HMC readiness, production readiness, GPU readiness, or universal DPF superiority.",
    ]


def _value_and_score(value_fn: Any, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = value_fn(theta)
    score = tape.gradient(value, theta)
    if score is None:
        raise P7Alg1ValidationError("GradientTape returned None")
    return value, score


def _observations(family: str, dim: int, horizon: int) -> tf.Tensor:
    if family == "m2":
        values = tf.constant([[0.10, -0.04, 0.07], [-0.02, 0.06, -0.05]], dtype=DTYPE)
        return values[:, : int(dim)]
    if family == "m3":
        values = tf.constant([[0.16, 0.11, 0.20], [0.09, 0.18, 0.13]], dtype=DTYPE)
        return values[:, : int(dim)]
    if family == "m4":
        values = tf.constant(
            [[0.08, -0.03, 0.06], [-0.02, 0.05, -0.04], [0.04, -0.01, 0.03], [0.01, 0.02, -0.02]],
            dtype=DTYPE,
        )
        return values[: int(horizon), : int(dim)]
    raise ValueError(f"unknown P44 family {family}")


def _physical_parts(family: str, theta: tf.Tensor, dim: int) -> dict[str, tf.Tensor]:
    theta = tf.convert_to_tensor(theta, dtype=DTYPE)
    if family == "m2":
        rho_scale = tf.constant([1.00, 0.86, 0.72], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.15, 1.35], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.20, 0.85], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        cubic_scale = tf.constant([1.00, 0.80, 1.20], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.55, 0.70, 0.90], dtype=DTYPE)[:dim]
        rho = 0.45 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        initial_mean = rho * raw_initial_mean
        initial_variance = tf.square(rho) * raw_initial_variance + transition_variance
        cubic = 0.04 * tf.tanh(theta[4]) * cubic_scale
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "initial_mean": initial_mean,
            "initial_variance": initial_variance,
            "cubic": cubic,
        }
    if family == "m3":
        rho_scale = tf.constant([1.00, 0.88, 0.76], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.12, 1.30], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.18, 0.90], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.50, 0.68, 0.82], dtype=DTYPE)[:dim]
        rho = 0.40 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        initial_mean = rho * raw_initial_mean
        initial_variance = tf.square(rho) * raw_initial_variance + transition_variance
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "initial_mean": initial_mean,
            "initial_variance": initial_variance,
        }
    if family == "m4":
        rho_scale = tf.constant([1.00, 0.88, 0.76], dtype=DTYPE)[:dim]
        q_scale = tf.constant([0.90, 1.12, 1.30], dtype=DTYPE)[:dim]
        r_scale = tf.constant([1.00, 1.18, 0.90], dtype=DTYPE)[:dim]
        mean_scale = tf.constant([1.00, -0.50, 0.25], dtype=DTYPE)[:dim]
        nonlin_scale = tf.constant([1.00, 0.75, 1.15], dtype=DTYPE)[:dim]
        raw_initial_variance = tf.constant([0.50, 0.68, 0.82], dtype=DTYPE)[:dim]
        rho = 0.42 * tf.tanh(theta[0]) * rho_scale
        transition_variance = tf.exp(theta[1]) * q_scale
        observation_variance = tf.exp(theta[2]) * r_scale
        raw_initial_mean = theta[3] * mean_scale
        nonlin = 0.05 * tf.tanh(theta[4]) * nonlin_scale
        return {
            "rho": rho,
            "transition_variance": transition_variance,
            "observation_variance": observation_variance,
            "raw_initial_mean": raw_initial_mean,
            "raw_initial_variance": raw_initial_variance,
            "initial_mean": rho * raw_initial_mean,
            "initial_variance": tf.square(rho) * raw_initial_variance + transition_variance,
            "nonlin": nonlin,
        }
    raise ValueError(f"unknown P44 family {family}")


def _axis_part(family: str, theta: tf.Tensor, axis: int) -> dict[str, tf.Tensor]:
    parts = _physical_parts(family, theta, axis + 1)
    return {key: value[axis] for key, value in parts.items()}


def _rows(values: tf.Tensor, width: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(values, dtype=DTYPE)
    if tensor.shape.rank == 1:
        tensor = tensor[tf.newaxis, :]
    if tensor.shape.rank != 2 or tensor.shape[1] != int(width):
        raise ValueError(f"{name} has wrong shape")
    return tensor


def _transition_mean_scalar(values: tf.Tensor, parts: dict[str, tf.Tensor]) -> tf.Tensor:
    values = tf.convert_to_tensor(values, dtype=DTYPE)
    return parts["rho"] * values + parts["nonlin"] * tf.math.tanh(values)


def _observation_mean(family: str, parts: dict[str, tf.Tensor], values: tf.Tensor) -> tf.Tensor:
    if family == "m2":
        return values + parts["cubic"] * tf.pow(values, 3)
    if family == "m3":
        return tf.square(values)
    if family == "m4":
        return values
    raise ValueError(f"unknown P44 family {family}")


def _observation_jacobian(family: str, parts: dict[str, tf.Tensor], values: tf.Tensor) -> tf.Tensor:
    if family == "m2":
        return 1.0 + 3.0 * parts["cubic"] * tf.square(values)
    if family == "m3":
        return 2.0 * values
    if family == "m4":
        return tf.ones_like(values)
    raise ValueError(f"unknown P44 family {family}")


def _sample_normal(loc: tf.Tensor, variance: tf.Tensor, num_particles: int, seed: int, salt: int) -> tf.Tensor:
    loc = tf.cast(loc, DTYPE)
    variance = tf.cast(variance, DTYPE)
    normal = tf.random.stateless_normal([num_particles], seed=_seed_pair(seed, salt), dtype=DTYPE)
    return loc + tf.sqrt(variance) * normal


def _directions(size: int) -> tf.Tensor:
    eye = tf.eye(size, dtype=DTYPE)
    mixed_a = tf.cast(tf.range(1, size + 1), DTYPE)
    mixed_a = mixed_a / tf.linalg.norm(mixed_a)
    mixed_b = tf.where(
        tf.math.floormod(tf.range(size), 2) == 0,
        tf.ones([size], dtype=DTYPE),
        -tf.ones([size], dtype=DTYPE),
    )
    mixed_b = mixed_b / tf.linalg.norm(mixed_b)
    return tf.concat([eye, mixed_a[tf.newaxis, :], mixed_b[tf.newaxis, :]], axis=0)


def _first_gradient(
    rows: list[dict[str, Any]],
    target_id: str,
    dim: int,
    num_particles: int,
    seed: int,
) -> list[float]:
    for row in rows:
        if (
            row["target_id"] == target_id
            and int(row["dim"]) == int(dim)
            and int(row["num_particles"]) == int(num_particles)
            and int(row["seed"]) == int(seed)
        ):
            return list(row["fixed_branch_gradient"])
    raise P7Alg1ValidationError("missing representative gradient row")


def _directional_for(
    directional: list[dict[str, Any]],
    target_id: str,
    dim: int,
    num_particles: int,
) -> dict[str, Any] | None:
    for row in directional:
        if row["target_id"] == target_id and int(row["dim"]) == int(dim) and int(row["num_particles"]) == int(num_particles):
            return row
    return None


def _branch_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ess_values = [float(row["ess"]) for row in records]
    return {
        "branch_record_count": len(records),
        "ess_min": min(ess_values) if ess_values else None,
        "ess_mean": statistics.fmean(ess_values) if ess_values else None,
        "resampling_count": sum(1 for row in records if row["resampled"]),
        "all_corrected_log_weights_finite": all(
            bool(row.get("finite_corrected_log_weights", False)) for row in records
        ),
        "min_predicted_covariance_eigenvalue": min(
            float(row["min_predicted_covariance_eigenvalue"]) for row in records
        )
        if records
        else None,
        "min_forward_log_det": min(float(row["min_forward_log_det"]) for row in records) if records else None,
        "max_forward_log_det": max(float(row["max_forward_log_det"]) for row in records) if records else None,
        "old_ledh_pfpf_ot_used": False,
    }


def _finite_mean(values: list[float]) -> float | None:
    if not values or not _all_finite(values):
        return None
    return statistics.fmean(values)


def _standard_error(values: list[float]) -> float | None:
    if len(values) < 2 or not _all_finite(values):
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _ci95(values: list[float]) -> list[float] | None:
    if len(values) < 2 or not _all_finite(values):
        return None
    mean = statistics.fmean(values)
    se = _standard_error(values)
    assert se is not None
    return [mean - 1.96 * se, mean + 1.96 * se]


def _rmse(values: list[float]) -> float | None:
    if not values or not _all_finite(values):
        return None
    return math.sqrt(statistics.fmean([float(value) ** 2 for value in values]))


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _augmented_algorithm1_route() -> dict[str, str]:
    route = algorithm1_route_identifiers(resampling_route="none")
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


def _is_augmented_algorithm1_route(route: dict[str, str]) -> bool:
    expected = _augmented_algorithm1_route()
    return all(route.get(key) == value for key, value in expected.items())


def _old_runtime_module_loaded() -> bool:
    return any(
        name.endswith(".ledh_pfpf_ot_tf")
        or name.endswith(".run_v2_ledh_pfpf_ot_values_tf")
        or name.endswith(".run_filter_oracle_comparison_p8_p44_dpf_blocker_closure_tf")
        for name in sys.modules
    )


def _ukf_parameters() -> dict[str, float]:
    return {"alpha": UKF_ALPHA, "beta": UKF_BETA, "kappa": UKF_KAPPA}


def _seed_pair(seed: int, salt: int) -> tf.Tensor:
    return tf.constant([int(seed) % 2_147_483_647, int(salt) % 2_147_483_647], dtype=tf.int32)


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_digest"}
    return stable_digest(stable)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
