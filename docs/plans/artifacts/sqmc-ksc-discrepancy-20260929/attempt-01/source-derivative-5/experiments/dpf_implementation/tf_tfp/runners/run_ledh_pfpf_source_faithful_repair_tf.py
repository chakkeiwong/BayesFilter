"""Run the source-faithful LEDH/PF-PF repair for P44 M3 rows."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-ledh-repair-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import math
import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filter_oracle_comparison_p8_p44_dpf_blocker_closure_tf as p8,
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


DTYPE = tf.float64
MODULE_PATH = (
    "experiments.dpf_implementation.tf_tfp.runners."
    "run_ledh_pfpf_source_faithful_repair_tf"
)
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-source-faithful-repair-plan-2026-06-10.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-source-faithful-repair-result-2026-06-10.md"
)
REVIEW_LEDGER_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-ledh-pfpf-source-faithful-repair-claude-review-ledger-2026-06-10.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_source_faithful_repair_2026-06-10.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-source-faithful-repair-2026-06-10.md"
TARGET_ID = "p44_m3_quadratic_observation_panel"
METHOD_ID = "dpf_ledh_pfpf_ot"
SEEDS = list(p8.SEEDS)
PARTICLE_COUNTS = [128, 256, 512]
EXTRA_PARTICLE_COUNTS = [1024, 2048]
PSEUDO_TIME_STEP_COUNT = 29
PSEUDO_TIME_RATIO = 1.2


class RepairValidationError(ValueError):
    """Raised when the repair artifact violates the reviewed contract."""


def _pseudo_time_steps() -> tf.Tensor:
    ratio = tf.constant(PSEUDO_TIME_RATIO, dtype=DTYPE)
    count = PSEUDO_TIME_STEP_COUNT
    first = (1.0 - ratio) / (1.0 - tf.pow(ratio, tf.cast(count, DTYPE)))
    powers = tf.pow(ratio, tf.cast(tf.range(count), DTYPE))
    steps = tf.cast(first, DTYPE) * powers
    return steps / tf.reduce_sum(steps)


PSEUDO_TIME_STEPS = _pseudo_time_steps()


def _source_faithful_m3_ledh_scalar_flow(
    *,
    model: p8.AxisModel,
    theta: tf.Tensor,
    pre_flow_particles: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    observation: tf.Tensor,
) -> dict[str, Any]:
    if model.family != "m3":
        return p8._ledh_scalar_flow(
            model=model,
            theta=theta,
            pre_flow_particles=pre_flow_particles,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            observation=observation,
        )

    eta0 = p8._rows(pre_flow_particles, 1, "pre_flow_particles")[:, 0]
    prior_mean_vec = tf.reshape(tf.cast(prior_mean, DTYPE), [-1])
    if int(prior_mean_vec.shape[0]) == 1:
        prior_mean_vec = tf.fill(tf.shape(eta0), prior_mean_vec[0])
    prior_variance = tf.maximum(tf.cast(prior_variance, DTYPE), tf.constant(1e-10, dtype=DTYPE))
    observation_value = tf.reshape(tf.cast(observation, DTYPE), [1])[0]
    parts = p8._axis_part(model.family, theta, model.axis)
    observation_variance = tf.maximum(parts["observation_variance"], tf.constant(1e-12, dtype=DTYPE))

    eta = eta0
    bar_eta = prior_mean_vec
    log_abs_det = tf.zeros_like(eta0)
    min_abs_step_det = tf.fill(tf.shape(eta0), tf.constant(float("inf"), dtype=DTYPE))
    max_abs_step_logdet = tf.zeros_like(eta0)
    min_abs_aux_jacobian = tf.fill(tf.shape(eta0), tf.constant(float("inf"), dtype=DTYPE))
    max_abs_a = tf.zeros_like(eta0)

    lam = tf.constant(0.0, dtype=DTYPE)
    for step in tf.unstack(PSEUDO_TIME_STEPS):
        lam = lam + step
        # The scalar Li-Coates LEDH step computes coefficients from the
        # auxiliary predicted state, then applies the same affine step to eta.
        h_jac = 2.0 * bar_eta
        e = tf.square(bar_eta) - h_jac * bar_eta
        denominator = lam * tf.square(h_jac) * prior_variance + observation_variance
        a = -0.5 * prior_variance * tf.square(h_jac) / denominator
        local_info = prior_variance * h_jac * (observation_value - e) / observation_variance
        b = (1.0 + 2.0 * lam * a) * ((1.0 + lam * a) * local_info + a * prior_mean_vec)
        step_det = 1.0 + step * a
        abs_step_det = tf.abs(step_det)
        if not bool(finite_tensor(abs_step_det)):
            raise FloatingPointError("source-faithful LEDH step determinant is non-finite")
        if bool(tf.reduce_any(abs_step_det <= tf.constant(1e-12, dtype=DTYPE)).numpy()):
            raise FloatingPointError("source-faithful LEDH step determinant is singular")
        bar_eta = bar_eta + step * (a * bar_eta + b)
        eta = eta + step * (a * eta + b)
        log_step = tf.math.log(abs_step_det)
        log_abs_det = log_abs_det + log_step
        min_abs_step_det = tf.minimum(min_abs_step_det, abs_step_det)
        max_abs_step_logdet = tf.maximum(max_abs_step_logdet, tf.abs(log_step))
        min_abs_aux_jacobian = tf.minimum(min_abs_aux_jacobian, tf.abs(h_jac))
        max_abs_a = tf.maximum(max_abs_a, tf.abs(a))

    pre_flow_log_density = tfp.distributions.Normal(
        loc=prior_mean_vec,
        scale=tf.sqrt(prior_variance),
    ).log_prob(eta0)
    diagnostics = {
        "ledh_component_id": "source_faithful_scalar_auxiliary_ledh_pfpf",
        "ledh_source_route_identifier": "li_coates_2017_algorithm_pf_pf_ledh_auxiliary_flow",
        "determinant_route_identifier": "sum_log_abs_det_one_plus_epsilon_a_i_j",
        "coefficient_source": "auxiliary_state_bar_eta",
        "actual_particle_role": "eta0_sample_migrated_by_auxiliary_affine_steps",
        "collapsed_shortcut_used": False,
        "exact_jacobian_of_collapsed_shortcut_used": False,
        "pseudo_time_step_count": PSEUDO_TIME_STEP_COUNT,
        "pseudo_time_step_ratio": PSEUDO_TIME_RATIO,
        "pseudo_time_step_sum": scalar(tf.reduce_sum(PSEUDO_TIME_STEPS)),
        "ledh_local_linearization": "per_particle_auxiliary_state_observation_jacobian",
        "pfpf_correction": "log_exact_target_prior_or_transition_plus_observation_minus_q0_plus_forward_logdet",
        "finite_pre_flow": bool(finite_tensor(eta0)),
        "finite_post_flow": bool(finite_tensor(eta)),
        "finite_forward_log_det": bool(finite_tensor(log_abs_det)),
        "finite_pre_flow_log_density": bool(finite_tensor(pre_flow_log_density)),
        "min_abs_auxiliary_observation_jacobian": scalar(tf.reduce_min(min_abs_aux_jacobian)),
        "min_abs_affine_step_determinant": scalar(tf.reduce_min(min_abs_step_det)),
        "min_forward_log_det": scalar(tf.reduce_min(log_abs_det)),
        "max_forward_log_det": scalar(tf.reduce_max(log_abs_det)),
        "max_abs_forward_log_det": scalar(tf.reduce_max(tf.abs(log_abs_det))),
        "max_abs_step_log_det": scalar(tf.reduce_max(max_abs_step_logdet)),
        "max_abs_a_coefficient": scalar(tf.reduce_max(max_abs_a)),
        "min_jacobian_singular_value": scalar(tf.reduce_min(tf.exp(log_abs_det))),
        "max_jacobian_singular_value": scalar(tf.reduce_max(tf.exp(log_abs_det))),
    }
    return {
        "post_flow_particles": tf.reshape(eta, [-1, 1]),
        "pre_flow_log_density": pre_flow_log_density,
        "forward_log_det": log_abs_det,
        "diagnostics": diagnostics,
    }


def _with_source_faithful_flow(fn):
    old = p8._ledh_scalar_flow
    p8._ledh_scalar_flow = _source_faithful_m3_ledh_scalar_flow
    try:
        return fn()
    finally:
        p8._ledh_scalar_flow = old


def _row(target_id: str, dim: int, seed: int, num_particles: int, reference: dict[str, Any]) -> dict[str, Any]:
    return _with_source_faithful_flow(
        lambda: p8._safe_dpf_row(
            method_id=METHOD_ID,
            target_id=target_id,
            dim=dim,
            seed=seed,
            num_particles=num_particles,
            reference=reference,
        )
    )


def _directional(
    target_id: str,
    dim: int,
    seed: int,
    num_particles: int,
    ad_gradient: list[float],
) -> dict[str, Any]:
    return _with_source_faithful_flow(
        lambda: p8._safe_directional_residual(
            target_id=target_id,
            dim=dim,
            method_id=METHOD_ID,
            seed=seed,
            num_particles=num_particles,
            ad_gradient=ad_gradient,
        )
    )


def _old_shortcut_flow_for_probe(
    model: p8.AxisModel,
    theta: tf.Tensor,
    x0_values: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    observation: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    flow = p8._ledh_scalar_flow(
        model=model,
        theta=theta,
        pre_flow_particles=tf.reshape(x0_values, [-1, 1]),
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        observation=observation,
    )
    return flow["post_flow_particles"][:, 0], flow["forward_log_det"]


def _source_faithful_flow_for_probe(
    model: p8.AxisModel,
    theta: tf.Tensor,
    x0_values: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_variance: tf.Tensor,
    observation: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    flow = _source_faithful_m3_ledh_scalar_flow(
        model=model,
        theta=theta,
        pre_flow_particles=tf.reshape(x0_values, [-1, 1]),
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        observation=observation,
    )
    return flow["post_flow_particles"][:, 0], flow["forward_log_det"]


def _determinant_diagnostics() -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    theta = tf.constant(p8.TARGET_SPECS[TARGET_ID]["theta"], dtype=DTYPE)
    for dim in (1, 2, 3):
        for axis in range(dim):
            model = p8.AxisModel(target_id=TARGET_ID, family="m3", axis=axis, horizon=2)
            observations = model.observations()
            observation = observations[0]
            prior_mean, prior_variance = p8._initial_ledh_gaussian_moments(model, theta)
            center = tf.reshape(tf.cast(prior_mean, DTYPE), [1])[0]
            spread = tf.sqrt(tf.cast(prior_variance, DTYPE))
            x0_values = center + spread * tf.constant([-1.25, -0.25, 0.25, 1.25], dtype=DTYPE)
            with tf.GradientTape() as tape:
                tape.watch(x0_values)
                old_post, old_logdet = _old_shortcut_flow_for_probe(
                    model,
                    theta,
                    x0_values,
                    prior_mean,
                    prior_variance,
                    observation,
                )
            old_true_diag = tf.linalg.diag_part(tape.jacobian(old_post, x0_values))
            source_post, source_logdet = _source_faithful_flow_for_probe(
                model,
                theta,
                x0_values,
                prior_mean,
                prior_variance,
                observation,
            )
            eps = tf.constant(1e-5, dtype=DTYPE)
            source_plus, _ = _source_faithful_flow_for_probe(
                model,
                theta,
                x0_values + eps,
                prior_mean,
                prior_variance,
                observation,
            )
            source_minus, _ = _source_faithful_flow_for_probe(
                model,
                theta,
                x0_values - eps,
                prior_mean,
                prior_variance,
                observation,
            )
            source_fd_diag = (source_plus - source_minus) / (2.0 * eps)
            diagnostics.append(
                {
                    "target_id": TARGET_ID,
                    "dim": int(dim),
                    "axis": int(axis),
                    "probe_x0": tensor_to_json(x0_values),
                    "old_shortcut_frozen_logdet": tensor_to_json(old_logdet),
                    "old_shortcut_true_log_abs_derivative": tensor_to_json(
                        tf.math.log(tf.abs(old_true_diag))
                    ),
                    "old_shortcut_max_logdet_discrepancy": scalar(
                        tf.reduce_max(tf.abs(old_logdet - tf.math.log(tf.abs(old_true_diag))))
                    ),
                    "source_faithful_accumulated_logdet": tensor_to_json(source_logdet),
                    "source_faithful_fd_log_abs_derivative": tensor_to_json(
                        tf.math.log(tf.abs(source_fd_diag))
                    ),
                    "source_faithful_max_logdet_discrepancy": scalar(
                        tf.reduce_max(tf.abs(source_logdet - tf.math.log(tf.abs(source_fd_diag))))
                    ),
                    "source_faithful_post_flow": tensor_to_json(source_post),
                    "old_shortcut_post_flow": tensor_to_json(old_post),
                    "diagnostic_role": "explanatory_only",
                }
            )
    return diagnostics


def _run() -> dict[str, Any]:
    references = [p8._reference_row(TARGET_ID, dim) for dim in (1, 2, 3)]
    reference_by_dim = {int(row["dim"]): row for row in references}
    rows: list[dict[str, Any]] = []
    for dim in (1, 2, 3):
        reference = reference_by_dim[dim]
        for num_particles in PARTICLE_COUNTS:
            for seed in SEEDS:
                rows.append(_row(TARGET_ID, dim, seed, num_particles, reference))

    extra_rows: list[dict[str, Any]] = []
    for num_particles in EXTRA_PARTICLE_COUNTS:
        for seed in SEEDS:
            extra_rows.append(_row(TARGET_ID, 1, seed, num_particles, reference_by_dim[1]))

    directional_residuals: list[dict[str, Any]] = []
    for dim in (1, 2, 3):
        representative = next(
            row
            for row in rows
            if int(row["dim"]) == dim
            and int(row["seed"]) == SEEDS[0]
            and int(row["num_particles"]) == PARTICLE_COUNTS[-1]
        )
        directional_residuals.append(
            _directional(
                TARGET_ID,
                dim,
                SEEDS[0],
                PARTICLE_COUNTS[-1],
                representative["fixed_branch_gradient"],
            )
        )

    summaries = []
    for dim in (1, 2, 3):
        reference = reference_by_dim[dim]
        for num_particles in PARTICLE_COUNTS:
            summaries.append(
                p8._row_summary(
                    target_id=TARGET_ID,
                    dim=dim,
                    method_id=METHOD_ID,
                    num_particles=num_particles,
                    rows=rows,
                    reference=reference,
                    directional=next(
                        (
                            row
                            for row in directional_residuals
                            if int(row["dim"]) == dim and int(row["num_particles"]) == num_particles
                        ),
                        None,
                    ),
                )
            )
    extra_summaries = [
        p8._row_summary(
            target_id=TARGET_ID,
            dim=1,
            method_id=METHOD_ID,
            num_particles=num_particles,
            rows=extra_rows,
            reference=reference_by_dim[1],
            directional=None,
        )
        for num_particles in EXTRA_PARTICLE_COUNTS
    ]
    determinant_diagnostics = _determinant_diagnostics()
    veto_diagnostics = _veto_diagnostics(rows, extra_rows, determinant_diagnostics, directional_residuals)
    decision = (
        "PASS_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR_READY_FOR_REVIEW"
        if not any(veto_diagnostics.values())
        else "BLOCKED_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR"
    )
    return _payload(
        decision=decision,
        references=references,
        rows=rows,
        extra_rows=extra_rows,
        row_summaries=summaries,
        extra_row_summaries=extra_summaries,
        directional_residuals=directional_residuals,
        determinant_diagnostics=determinant_diagnostics,
        veto_diagnostics=veto_diagnostics,
    )


def _veto_diagnostics(
    rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    determinant_diagnostics: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
) -> dict[str, bool]:
    branch_records = _flatten_branch_records(rows + extra_rows)
    source_route_ok = all(
        record.get("ledh_source_route_identifier")
        == "li_coates_2017_algorithm_pf_pf_ledh_auxiliary_flow"
        for record in branch_records
        if record.get("method_id") == METHOD_ID
    )
    determinant_route_ok = all(
        record.get("determinant_route_identifier") == "sum_log_abs_det_one_plus_epsilon_a_i_j"
        for record in branch_records
        if record.get("method_id") == METHOD_ID
    )
    no_collapsed = all(not bool(record.get("collapsed_shortcut_used", True)) for record in branch_records)
    no_exact_wrong = all(
        not bool(record.get("exact_jacobian_of_collapsed_shortcut_used", True))
        for record in branch_records
    )
    min_step_det = [
        float(record.get("min_abs_affine_step_determinant", float("nan")))
        for record in branch_records
    ]
    return {
        "missing_or_wrong_source_route": not source_route_ok,
        "missing_or_wrong_determinant_route": not determinant_route_ok,
        "collapsed_shortcut_used": not no_collapsed,
        "exact_jacobian_of_collapsed_shortcut_used": not no_exact_wrong,
        "nonfinite_rows": any(not bool(row["finite"]) for row in rows + extra_rows),
        "missing_branch_records": not branch_records,
        "singular_affine_step_determinant": any(
            (not math.isfinite(value)) or value <= 1e-12 for value in min_step_det
        ),
        "directional_residual_nonfinite": any(
            not math.isfinite(float(row.get("max_abs_directional_residual", float("nan"))))
            for row in directional_residuals
        ),
        "source_faithful_logdet_fd_mismatch": any(
            float(row["source_faithful_max_logdet_discrepancy"]) > 1e-4
            for row in determinant_diagnostics
        ),
    }


def _flatten_branch_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        if not row.get("branch_records"):
            continue
        for branch in row["branch_records"]:
            records.append(
                {
                    "target_id": row["target_id"],
                    "dim": int(row["dim"]),
                    "method_id": row["method_id"],
                    "seed": int(row["seed"]),
                    "num_particles": int(row["num_particles"]),
                    **branch,
                }
            )
    return records


def _payload(
    *,
    decision: str,
    references: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    extra_rows: list[dict[str, Any]],
    row_summaries: list[dict[str, Any]],
    extra_row_summaries: list[dict[str, Any]],
    directional_residuals: list[dict[str, Any]],
    determinant_diagnostics: list[dict[str, Any]],
    veto_diagnostics: dict[str, bool],
) -> dict[str, Any]:
    manifest = environment_manifest(
        command="CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/bayesfilter-dpf-ledh-repair-mpl python -m "
        + MODULE_PATH,
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
            "extra_particle_counts": list(EXTRA_PARTICLE_COUNTS),
            "pseudo_time_step_count": PSEUDO_TIME_STEP_COUNT,
            "pseudo_time_step_ratio": PSEUDO_TIME_RATIO,
            "pseudo_time_schedule": tensor_to_json(PSEUDO_TIME_STEPS),
            "source_route_identifier": "li_coates_2017_algorithm_pf_pf_ledh_auxiliary_flow",
            "determinant_route_identifier": "sum_log_abs_det_one_plus_epsilon_a_i_j",
            "data_version": "P44 M3 local deterministic target fixtures",
        }
    )
    branch_records = _flatten_branch_records(rows + extra_rows)
    return {
        "metadata_date": "2026-06-10",
        "created_at_utc": utc_now(),
        "phase": "LEDF_PFPF_SOURCE_FAITHFUL_REPAIR",
        "execution_route": "VISIBLE_IN_DIALOGUE",
        "decision": decision,
        "question": "Can BayesFilter replace the collapsed M3 LEDH shortcut with a Li-Coates auxiliary-flow PF-PF route and rerun P44 M3 rows?",
        "review_gate": {
            "plan_review_status": "PLAN_REVIEW_CONVERGED_VERDICT_AGREE_ITERATION_2",
            "plan_path": PLAN_PATH,
            "ledger_path": REVIEW_LEDGER_PATH,
        },
        "target": TARGET_ID,
        "method": METHOD_ID,
        "reference_rows": references,
        "rows": rows,
        "extra_rows": extra_rows,
        "row_summaries": row_summaries,
        "extra_row_summaries": extra_row_summaries,
        "directional_residuals": directional_residuals,
        "determinant_diagnostics": determinant_diagnostics,
        "branch_records": branch_records,
        "veto_diagnostics": veto_diagnostics,
        "decision_table": _decision_table(decision, veto_diagnostics, row_summaries),
        "post_run_red_team": _post_run_red_team(decision),
        "run_manifest": manifest,
        "nonclaims": _nonclaims(),
    }


def _decision_table(
    decision: str,
    veto_diagnostics: dict[str, bool],
    row_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    promoted = [
        row for row in row_summaries if row.get("row_decision") == "PROMOTED_EXACT_TARGET_CLOSENESS"
    ]
    diagnostic = [
        row for row in row_summaries if row.get("row_decision") == "DIAGNOSTIC_ONLY_MEASURED"
    ]
    failed = [row for row in row_summaries if row.get("row_decision") == "FAILED_NUMERIC_BANDS"]
    return [
        {
            "decision": decision,
            "primary_criterion_status": (
                f"{len(row_summaries)} repaired M3 rows; {len(promoted)} promoted, "
                f"{len(diagnostic)} diagnostic, {len(failed)} failed numeric bands"
            ),
            "veto_diagnostic_status": {
                key: value for key, value in veto_diagnostics.items() if bool(value)
            }
            or "no structural vetoes",
            "main_uncertainty": "Auxiliary-flow LEDH/PF-PF is source-faithful to Li-Coates but still a finite-particle nonlinear approximation.",
            "next_justified_action": "run Claude read-only result review and then decide whether to amend P8/P6 tables",
            "not_concluded": "no production, HMC, universal superiority, or exact nonlinear filtering claim",
        }
    ]


def _post_run_red_team(decision: str) -> dict[str, Any]:
    if decision != "PASS_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR_READY_FOR_REVIEW":
        return {
            "strongest_alternative_explanation": "Failure may reflect pseudo-time stiffness or an implementation mistake rather than evidence against LEDH/PF-PF.",
            "what_result_would_overturn": "A repaired rerun with source-route fields, finite step determinants, and determinant product matching finite differences.",
            "weakest_part_of_evidence": "The run stopped at source-faithful repair diagnostics rather than broader filter comparison.",
        }
    return {
        "strongest_alternative_explanation": "Finite M3 rows may still be noisy finite-particle approximations, not stable stochastic-score evidence.",
        "what_result_would_overturn": "Independent implementation or larger-particle ladder showing non-finite determinants or materially different value/score behavior.",
        "weakest_part_of_evidence": "Only P44 M3 and fixed-branch AD gradients are covered.",
    }


def _nonclaims() -> list[str]:
    return [
        "Historical P8 artifacts are not overwritten.",
        "This repair invalidates the old collapsed-shortcut M3 LEDH-PFPF interpretation but does not prove universal LEDH-PFPF superiority.",
        "Rows use fixed-branch AD gradients, not full stochastic-resampling scores.",
        "No HMC, production, public API, GPU, or default-policy readiness is concluded.",
        "Li-Coates auxiliary-flow LEDH/PF-PF remains an approximation for nonlinear observations.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "review_gate",
        "rows",
        "extra_rows",
        "reference_rows",
        "row_summaries",
        "directional_residuals",
        "determinant_diagnostics",
        "branch_records",
        "veto_diagnostics",
        "decision_table",
        "run_manifest",
        "nonclaims",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise RepairValidationError(f"missing payload fields {sorted(missing)}")
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
        "pseudo_time_schedule",
        "source_route_identifier",
        "determinant_route_identifier",
        "wall_time_seconds",
    ):
        if field not in manifest:
            raise RepairValidationError(f"run_manifest missing {field}")
    if manifest["pre_import_cuda_visible_devices"] != "-1":
        raise RepairValidationError("TensorFlow was not forced CPU-only before import")
    if manifest["source_route_identifier"] != "li_coates_2017_algorithm_pf_pf_ledh_auxiliary_flow":
        raise RepairValidationError("wrong source route")
    if manifest["determinant_route_identifier"] != "sum_log_abs_det_one_plus_epsilon_a_i_j":
        raise RepairValidationError("wrong determinant route")
    if payload["review_gate"]["plan_review_status"] != "PLAN_REVIEW_CONVERGED_VERDICT_AGREE_ITERATION_2":
        raise RepairValidationError("plan review gate is not closed")
    if payload["decision"] not in {
        "PASS_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR_READY_FOR_REVIEW",
        "BLOCKED_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR",
    }:
        raise RepairValidationError(f"invalid decision {payload['decision']}")
    if not payload["branch_records"]:
        raise RepairValidationError("missing branch records")
    if any(bool(value) for value in payload["veto_diagnostics"].values()):
        if payload["decision"] == "PASS_LEDH_PFPF_SOURCE_FAITHFUL_REPAIR_READY_FOR_REVIEW":
            raise RepairValidationError("pass decision has active veto diagnostics")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# LEDH-PFPF Source-Faithful Repair Result",
        "",
        "metadata_date: 2026-06-10",
        f"status: {payload['decision']}",
        "",
        "## Review Gate",
        "",
        f"- plan review: `{payload['review_gate']['plan_review_status']}`",
        f"- plan: `{payload['review_gate']['plan_path']}`",
        f"- ledger: `{payload['review_gate']['ledger_path']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            f"| `{row['decision']}` | {row['primary_criterion_status']} | "
            f"{row['veto_diagnostic_status']} | {row['main_uncertainty']} | "
            f"{row['next_justified_action']} | {row['not_concluded']} |"
        )
    lines.extend(
        [
            "",
            "## Source Route",
            "",
            f"- source route: `{payload['run_manifest']['source_route_identifier']}`",
            f"- determinant route: `{payload['run_manifest']['determinant_route_identifier']}`",
            "- collapsed shortcut used: `False`",
            "- exact Jacobian of collapsed shortcut used: `False`",
            "",
            "## P44 M3 Repaired Rows",
            "",
            "| Dim | Particles | Seeds | Value RMSE/obs | Mean relative score error | Row decision |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["row_summaries"]:
        lines.append(
            f"| {row['dim']} | {row['num_particles']} | {row['seed_count']} | "
            f"{row['per_observation_value_error_rmse']:.6g} | "
            f"{row['mean_relative_score_error']:.6g} | `{row['row_decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Extra Dim-1 Ladder",
            "",
            "| Particles | Seeds | Value RMSE/obs | Mean relative score error | Row decision |",
            "| ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["extra_row_summaries"]:
        lines.append(
            f"| {row['num_particles']} | {row['seed_count']} | "
            f"{row['per_observation_value_error_rmse']:.6g} | "
            f"{row['mean_relative_score_error']:.6g} | `{row['row_decision']}` |"
        )
    max_old_delta = max(
        float(row["old_shortcut_max_logdet_discrepancy"])
        for row in payload["determinant_diagnostics"]
    )
    max_source_delta = max(
        float(row["source_faithful_max_logdet_discrepancy"])
        for row in payload["determinant_diagnostics"]
    )
    lines.extend(
        [
            "",
            "## Determinant Diagnostics",
            "",
            f"- max old collapsed-shortcut frozen-vs-true logdet discrepancy: `{max_old_delta:.6g}`",
            f"- max source-faithful accumulated-vs-finite-difference logdet discrepancy: `{max_source_delta:.6g}`",
            "",
            "## Run Manifest",
            "",
            f"- command: `{payload['run_manifest']['command']}`",
            f"- git branch: `{payload['run_manifest']['branch']}`",
            f"- git commit: `{payload['run_manifest']['commit']}`",
            f"- CPU/GPU status: CPU-only, `CUDA_VISIBLE_DEVICES={payload['run_manifest']['pre_import_cuda_visible_devices']}` before TensorFlow import",
            f"- visible GPU devices: `{payload['run_manifest']['gpu_devices_visible']}`",
            f"- seeds: `{payload['run_manifest']['seed_list']}`",
            f"- particle counts: `{payload['run_manifest']['particle_counts']}`",
            f"- extra particle counts: `{payload['run_manifest']['extra_particle_counts']}`",
            f"- pseudo-time step count: `{payload['run_manifest']['pseudo_time_step_count']}`",
            f"- pseudo-time ratio: `{payload['run_manifest']['pseudo_time_step_ratio']}`",
            f"- JSON: `{payload['run_manifest']['json_path']}`",
            f"- report: `{payload['run_manifest']['report_path']}`",
            f"- result: `{payload['run_manifest']['result_path']}`",
            f"- wall time seconds: `{payload['run_manifest']['wall_time_seconds']}`",
            "",
            "## Nonclaims",
            "",
            *[f"- {item}" for item in payload["nonclaims"]],
            "",
        ]
    )
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_digest"}
    return stable_digest(stable)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        print("LEDF_PFPF_SOURCE_FAITHFUL_REPAIR_VALIDATED")
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


if __name__ == "__main__":
    raise SystemExit(main())
