"""Factorize row-173 BayesFilter log-weight carryover residuals."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import time
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import samplers

from experiments.dpf_implementation.tf_tfp.resampling import annealed_transport_tf
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_row_173_vjp_decomposition_tf as vjp,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_smoothness_gradient_localization_tf as localizer,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    reference_policy,
    validate_filterflow_reference_status,
)


tfd = tfp.distributions

PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-log-weight-edge-factorization-plan-2026-06-05.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-log-weight-edge-factorization-result-2026-06-05.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-log-weight-edge-factorization-review-loop-2026-06-05.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_log_weight_edge_factorization_2026-06-05.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-log-weight-edge-factorization-2026-06-05.md"
)
PRIOR_CARRYOVER_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_bayesfilter_carryover_split_2026-06-05.json"
)

TARGET_TIME_INDEX = 93
PREVIOUS_TIME_INDEX = TARGET_TIME_INDEX - 1
TAG = "row-173-log-weight-edge-factorization"
DTYPE = vjp.DTYPE
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
MODES = (
    "raw",
    "target_transport_log_weights_stop_gradient",
    "previous_carry_log_weights_stop_gradient",
    "previous_and_target_log_weights_stop_gradient",
    "all_times_transport_log_weights_stop_gradient",
)
EDGE_FIELDS = (
    "target_to_pre_log_weights",
    "pre_log_weights_to_pre_particles",
    "same_tape_pre_log_weights_carryover_vjp",
    "same_tape_full_recorded_state_residual",
    "same_tape_identity_residual",
    "same_tape_post_state_identity_residual",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0

    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    markdown = _markdown(payload)
    write_text(REPORT_PATH, markdown)
    write_text(REPO_ROOT / RESULT_PATH, markdown)
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    reference_status = r3._filterflow_status()
    validate_filterflow_reference_status(
        reference_status,
        marker_path=vjp.FILTERFLOW_MARKER_PATH,
    )
    initial_fingerprint = continuation._filterflow_fingerprint()
    config = vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    filterflow = vjp._filterflow_vjp_subprocess(config)
    if filterflow.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_log_weight_edge_factorization_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
            None,
        )
    raw_bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if raw_bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_log_weight_edge_factorization_bayesfilter_blocker",
            raw_bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            raw_bayesfilter,
            None,
        )
    mode_results = [
        _bayesfilter_factor_mode(filterflow, config, mode)
        for mode in MODES
    ]
    base_comparison = vjp._compare(filterflow, raw_bayesfilter)
    prior = _prior_carryover_split()
    edge_comparison = _edge_comparison(
        filterflow,
        raw_bayesfilter,
        mode_results,
        base_comparison,
        prior,
    )
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(
        initial_fingerprint,
        final_fingerprint,
    )
    veto_status = _veto_status(
        base_comparison,
        filterflow,
        raw_bayesfilter,
        mode_results,
        reference_status,
        comparator_drift,
    )
    decision = _decision(edge_comparison["classification"], veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_log_weight_edge_factorization_difference_audit",
        "decision": decision,
        "hypothesis_classification": edge_comparison["classification"],
        "hypothesis_reason": edge_comparison["classification_reason"],
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "json_path": str(JSON_PATH),
        "report_path": str(REPORT_PATH),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_vjp": vjp._compact_side(filterflow),
        "raw_bayesfilter_vjp": vjp._compact_side(raw_bayesfilter),
        "bayesfilter_mode_vjps": mode_results,
        "base_vjp_comparison": base_comparison,
        "log_weight_edge_comparison": edge_comparison,
        "prior_carryover_split": prior,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, edge_comparison, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_log_weight_edge_factorization_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH),
            "report_path": str(REPORT_PATH),
            "target_time_index": TARGET_TIME_INDEX,
            "previous_time_index": PREVIOUS_TIME_INDEX,
            "modes": list(MODES),
        },
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    reference_status: dict[str, Any],
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any] | None,
    raw_bayesfilter: dict[str, Any] | None,
    mode_results: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {
        "status": "blocked",
        "classification": "h1_blocked_or_vetoed",
        "classification_reason": blocker,
    }
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_log_weight_edge_factorization_difference_audit",
        "decision": decision,
        "hypothesis_classification": "h1_blocked_or_vetoed",
        "hypothesis_reason": blocker,
        "blocker": blocker,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "json_path": str(JSON_PATH),
        "report_path": str(REPORT_PATH),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(
            initial_fingerprint,
            final_fingerprint,
        ),
        "model_contract": _model_contract(),
        "filterflow_vjp": vjp._compact_side(filterflow),
        "raw_bayesfilter_vjp": vjp._compact_side(raw_bayesfilter),
        "bayesfilter_mode_vjps": mode_results,
        "log_weight_edge_comparison": comparison,
        "veto_status_table": {
            "all_vetoes_clear": False,
            "blocker": blocker,
        },
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(
            decision,
            comparison,
            {"all_vetoes_clear": False},
        ),
        "non_implications": _non_implications(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_log_weight_edge_factorization_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
    }


def _bayesfilter_factor_mode(
    filterflow: dict[str, Any],
    config: vjp.RunConfig,
    mode: str,
) -> dict[str, Any]:
    original_dtype = annealed_transport_tf.DTYPE
    annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
        model = vjp._model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = _factor_target_bundle(theta_variable, model, config, mode)
            target = bundle["target"]
        total_gradient = vjp._safe_gradient(tape, target, theta_variable)
        gradients = {
            name: vjp._safe_gradient(tape, target, tensor)
            for name, tensor in bundle.items()
            if name not in vjp.TARGET_FIELD_EXCLUSIONS
        }
        edge_tensors = _edge_tensors(tape, bundle, gradients)
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "mode": mode,
            "mode_description": _mode_description(mode),
            "target_scalar": vjp._float(target),
            "total_gradient_diag": r3._json(total_gradient),
            "finite_scalar": bool(tf.math.is_finite(target).numpy()),
            "finite_gradient": bool(
                tf.reduce_all(tf.math.is_finite(total_gradient)).numpy()
            ),
            "resampling_flag": [
                bool(value) for value in tf.reshape(bundle["flags"], [-1]).numpy().tolist()
            ],
            "edge_summaries": {
                name: vjp._field(tensor) for name, tensor in edge_tensors.items()
            },
            "edge_tensors": {
                name: r3._json(tensor) for name, tensor in edge_tensors.items()
            },
            "cpu_only_manifest": vjp._parent_cpu_manifest(),
        }
    finally:
        annealed_transport_tf.DTYPE = original_dtype


def _factor_target_bundle(
    theta: tf.Tensor,
    model: dict[str, Any],
    config: vjp.RunConfig,
    mode: str,
) -> dict[str, tf.Tensor]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    transition_matrix = localizer._transition_matrix(theta)
    observation_matrix = tf.constant(model["observation_matrix"], dtype=DTYPE)
    transition_chol = tf.constant(model["transition_covariance_chol"], dtype=DTYPE)
    observation_chol = tf.constant(model["observation_covariance_chol"], dtype=DTYPE)
    observations = tf.constant(model["observations"], dtype=DTYPE)
    particles = tf.constant(model["initial_particles"], dtype=DTYPE)
    num_particles = vjp.NUM_PARTICLES
    log_weights = tf.fill(
        [vjp.BATCH_SIZE, num_particles],
        -tf.math.log(tf.cast(num_particles, DTYPE)),
    )
    log_likelihoods = tf.zeros([vjp.BATCH_SIZE], dtype=DTYPE)
    transition_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([2], dtype=DTYPE),
        scale_tril=transition_chol,
    )
    observation_noise = tfd.MultivariateNormalTriL(
        loc=tf.zeros([1], dtype=DTYPE),
        scale_tril=observation_chol,
    )
    transition_cov_inv = tf.linalg.cholesky_solve(
        transition_chol,
        tf.eye(2, dtype=DTYPE),
    )
    observation_cov_inv = tf.linalg.cholesky_solve(
        observation_chol,
        tf.eye(1, dtype=DTYPE),
    )
    sigma_inv = transition_cov_inv + tf.linalg.matmul(
        tf.linalg.matmul(observation_matrix, observation_cov_inv, transpose_a=True),
        observation_matrix,
    )
    sigma = tf.linalg.inv(sigma_inv)
    sigma_chol = tf.linalg.cholesky(sigma)
    seed = tf.constant(vjp.FILTER_SEED, dtype=tf.int32)
    paddings = tf.stack([[0, 0], [0, 2 - tf.size(seed)]])
    seed = tf.squeeze(tf.pad(tf.reshape(seed, [1, -1]), paddings))

    for time_index in range(vjp.T):
        seed, _seed1, seed2 = samplers.split_seed(seed, n=3, salt="update")
        ess_log = r3._ess(log_weights)
        flags = ess_log <= tf.math.log(
            tf.cast(num_particles, DTYPE) * tf.constant(vjp.RESAMPLING_NEFF, DTYPE)
        )
        transport_log_weights = log_weights
        if mode == "all_times_transport_log_weights_stop_gradient" or (
            mode in {
                "target_transport_log_weights_stop_gradient",
                "previous_and_target_log_weights_stop_gradient",
            }
            and time_index == config.target_time_index
        ):
            transport_log_weights = tf.stop_gradient(transport_log_weights)
        transported = annealed_transport_tf.annealed_transport_resample_tf(
            particles,
            transport_log_weights,
            epsilon=vjp.EPSILON,
            scaling=vjp.SCALING,
            convergence_threshold=vjp.CONVERGENCE_THRESHOLD,
            max_iterations=vjp.MAX_ITERATIONS,
            ess_mask=tf.reshape(flags, [-1]),
            transport_gradient_mode="filterflow_clipped",
            application_mode="filterflow_all_rows",
        )
        pre_particles = particles
        pre_log_weights = log_weights
        transport_matrix = tf.cast(transported.transport_matrix, DTYPE)
        particles = transported.particles
        log_weights = transported.log_weights
        observation = observations[time_index]
        proposal_mean = localizer._optimal_proposal_mean(
            particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
        )
        proposal_dist = tfd.MultivariateNormalTriL(proposal_mean, sigma_chol)
        proposed_particles = proposal_dist.sample(seed=seed2)
        proposal_ll = r3._optimal_proposal_log_prob(
            particles,
            proposed_particles,
            observation,
            transition_matrix,
            observation_matrix,
            transition_cov_inv,
            observation_cov_inv,
            sigma,
            sigma_chol,
        )
        observation_ll = r3._observation_log_prob(
            proposed_particles,
            observation,
            observation_matrix,
            observation_noise,
        )
        transition_ll = r3._transition_log_prob(
            particles,
            proposed_particles,
            transition_matrix,
            transition_noise,
        )
        unnormalized = transition_ll + observation_ll - proposal_ll + log_weights
        increment = tf.reduce_logsumexp(unnormalized, axis=1)
        pre_current_log_likelihoods = log_likelihoods
        log_likelihoods = log_likelihoods + increment
        normalized = r3._filterflow_normalize(unnormalized, num_particles)
        log_weights = normalized
        if mode in {
            "previous_carry_log_weights_stop_gradient",
            "previous_and_target_log_weights_stop_gradient",
        } and time_index == PREVIOUS_TIME_INDEX:
            log_weights = tf.stop_gradient(log_weights)
        particles = proposed_particles
        if time_index == config.target_time_index:
            target = tf.reduce_mean(log_likelihoods)
            return {
                "target": target,
                "post_update_mean": target,
                "sum_pre_current_plus_increment_mean": tf.reduce_mean(
                    pre_current_log_likelihoods + increment
                ),
                "pre_current_mean": tf.reduce_mean(pre_current_log_likelihoods),
                "increment_mean": tf.reduce_mean(increment),
                "flags": flags,
                "log_ess": ess_log,
                "pre_particles": pre_particles,
                "pre_log_weights": pre_log_weights,
                "transport_matrix": transport_matrix,
                "post_particles": transported.particles,
                "post_log_weights": transported.log_weights,
                "increment": increment,
                "pre_current_log_likelihoods": pre_current_log_likelihoods,
            }
    raise RuntimeError("target time not reached")


def _edge_tensors(
    tape: tf.GradientTape,
    bundle: dict[str, tf.Tensor],
    gradients: dict[str, tf.Tensor],
) -> dict[str, tf.Tensor]:
    post_particles_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["post_particles"],
        bundle["pre_particles"],
        gradients["post_particles"],
    )
    post_log_weights_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["post_log_weights"],
        bundle["pre_particles"],
        gradients["post_log_weights"],
    )
    post_state_vjp = post_particles_vjp + post_log_weights_vjp
    pre_log_weights_to_pre_particles_unit = vjp._safe_gradient_with_upstream(
        tape,
        bundle["pre_log_weights"],
        bundle["pre_particles"],
        tf.ones_like(bundle["pre_log_weights"]),
    )
    pre_log_weights_carryover_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["pre_log_weights"],
        bundle["pre_particles"],
        gradients["pre_log_weights"],
    )
    pre_current_ll_to_pre_particles = vjp._safe_gradient_with_upstream(
        tape,
        bundle["pre_current_log_likelihoods"],
        bundle["pre_particles"],
        gradients["pre_current_log_likelihoods"],
    )
    log_ess_to_pre_particles = vjp._safe_gradient_with_upstream(
        tape,
        bundle["log_ess"],
        bundle["pre_particles"],
        gradients["log_ess"],
    )
    full_recorded_state_vjp = (
        post_state_vjp
        + pre_log_weights_carryover_vjp
        + pre_current_ll_to_pre_particles
        + log_ess_to_pre_particles
    )
    direct_pre_particle_adjoint = tf.linalg.matmul(
        bundle["transport_matrix"],
        gradients["post_particles"],
        transpose_a=True,
    )
    transport_matrix_to_pre_particles = vjp._safe_gradient_with_upstream(
        tape,
        bundle["transport_matrix"],
        bundle["pre_particles"],
        gradients["transport_matrix"],
    )
    return {
        "target_to_pre_log_weights": gradients["pre_log_weights"],
        "pre_log_weights_to_pre_particles": pre_log_weights_to_pre_particles_unit,
        "same_tape_pre_log_weights_carryover_vjp": pre_log_weights_carryover_vjp,
        "same_tape_full_recorded_state_residual": (
            gradients["pre_particles"] - full_recorded_state_vjp
        ),
        "same_tape_identity_residual": (
            gradients["pre_particles"]
            - (direct_pre_particle_adjoint + transport_matrix_to_pre_particles)
        ),
        "same_tape_post_state_identity_residual": (
            gradients["pre_particles"] - post_state_vjp
        ),
    }


def _edge_comparison(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
    base_comparison: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    mode_rows = {}
    raw_mode = _mode_by_name(mode_results, "raw")
    filterflow_edges = _filterflow_edge_summaries(filterflow)
    for mode_result in mode_results:
        mode = mode_result["mode"]
        scalar_delta = abs(
            float(mode_result["target_scalar"]) - float(filterflow["target_scalar"])
        )
        gradient_delta = [
            float(bf) - float(ff)
            for bf, ff in zip(
                mode_result["total_gradient_diag"],
                filterflow["total_gradient_diag"],
                strict=True,
            )
        ]
        value_valid = bool(
            mode_result["status"] == "executed"
            and mode_result["finite_scalar"]
            and mode_result["finite_gradient"]
            and scalar_delta <= VALUE_TOLERANCE
            and mode_result["resampling_flag"] == filterflow["resampling_flag"]
        )
        field_rows = {
            field: _field_row(
                field,
                filterflow_edges,
                raw_mode["edge_summaries"],
                mode_result["edge_summaries"],
                mode_result["edge_tensors"],
                filterflow_edges["tensors"],
                value_valid,
            )
            for field in EDGE_FIELDS
        }
        mode_rows[mode] = {
            "mode": mode,
            "mode_description": mode_result["mode_description"],
            "value_valid": value_valid,
            "scalar_delta": scalar_delta,
            "resampling_flag": mode_result["resampling_flag"],
            "resampling_flags_match_filterflow": (
                mode_result["resampling_flag"] == filterflow["resampling_flag"]
            ),
            "gradient_delta": gradient_delta,
            "max_abs_gradient_delta": max(abs(value) for value in gradient_delta),
            "gradient_within_tolerance": (
                max(abs(value) for value in gradient_delta) <= GRADIENT_TOLERANCE
            ),
            "field_rows": field_rows,
        }
    classification, reason, evidence = _classify_edges(mode_rows)
    return {
        "status": "compared",
        "classification": classification,
        "classification_reason": reason,
        "classification_evidence": evidence,
        "mode_rows": mode_rows,
        "raw_bayesfilter_edge_max_abs": {
            field: raw_mode["edge_summaries"][field]["max_abs"]
            for field in EDGE_FIELDS
        },
        "filterflow_edge_max_abs": {
            field: filterflow_edges["summaries"][field]["max_abs"]
            for field in EDGE_FIELDS
        },
        "total_gradient_delta": base_comparison["total_gradient_delta"],
        "max_abs_total_gradient_delta": base_comparison[
            "max_abs_total_gradient_delta"
        ],
        "scalar_delta": base_comparison["scalar_delta"],
        "first_value_delta_over_tolerance": base_comparison[
            "first_value_delta_over_tolerance"
        ],
        "first_gradient_delta_over_tolerance": base_comparison[
            "first_gradient_delta_over_tolerance"
        ],
        "prior_carryover_split_decision": prior.get("decision"),
        "prior_carryover_split_classification": prior.get("hypothesis_classification"),
        "decision_precedence": [
            "h1 if missing/non-finite raw or mode edge tensors",
            "h2 if target-only and previous-only each collapse the composed VJP and reduce different factors",
            "h3 if target-only reduces target upstream and composed VJP but previous-only does not",
            "h4 if previous-only reduces previous carry Jacobian and composed VJP but target-only does not",
            "h5 if only all-times transport-log-weight stopping reduces the composed VJP",
            "h6 otherwise",
        ],
    }


def _field_row(
    field: str,
    filterflow_edges: dict[str, Any],
    raw_summaries: dict[str, Any],
    mode_summaries: dict[str, Any],
    mode_tensors: dict[str, Any],
    filterflow_tensors: dict[str, Any],
    value_valid: bool,
) -> dict[str, Any]:
    raw_max = float(raw_summaries[field]["max_abs"])
    mode_max = float(mode_summaries[field]["max_abs"])
    reduction = raw_max - mode_max
    filterflow_tensor = filterflow_tensors[field]
    filterflow_max_abs = filterflow_edges["summaries"][field]["max_abs"]
    if filterflow_tensor is None:
        cross_delta: float | None = None
        comparator_status = "bayesfilter_only_unit_upstream_probe"
    else:
        cross_delta = vjp._max_abs_nested_delta(
            mode_tensors[field],
            filterflow_tensor,
        )
        comparator_status = "compared"
    return {
        "field": field,
        "filterflow_max_abs": filterflow_max_abs,
        "raw_bayesfilter_max_abs": raw_max,
        "mode_max_abs": mode_max,
        "mode_minus_filterflow_max_abs_delta": cross_delta,
        "comparator_status": comparator_status,
        "reduction_from_raw": reduction,
        "material_reduction": bool(value_valid and reduction > GRADIENT_TOLERANCE),
        "collapse": bool(value_valid and mode_max <= GRADIENT_TOLERANCE),
        "raw_material": bool(raw_max > GRADIENT_TOLERANCE),
        "mode_material": bool(mode_max > GRADIENT_TOLERANCE),
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _classify_edges(
    mode_rows: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    invalid = [
        mode
        for mode, row in mode_rows.items()
        if not row["value_valid"]
    ]
    if invalid:
        return (
            "h1_blocked_or_vetoed",
            "at least one required mode failed value/finiteness/resampling gates",
            {"invalid_modes": invalid},
        )
    target = mode_rows["target_transport_log_weights_stop_gradient"]["field_rows"]
    previous = mode_rows["previous_carry_log_weights_stop_gradient"]["field_rows"]
    all_times = mode_rows["all_times_transport_log_weights_stop_gradient"][
        "field_rows"
    ]
    composed_field = "same_tape_pre_log_weights_carryover_vjp"
    target_factor = "target_to_pre_log_weights"
    previous_factor = "pre_log_weights_to_pre_particles"
    composition = (
        target[composed_field]["collapse"]
        and previous[composed_field]["collapse"]
        and target[target_factor]["material_reduction"]
        and previous[previous_factor]["material_reduction"]
    )
    if composition:
        return (
            "h2_composition_edge",
            (
                "target-only and previous-only modes each collapse the composed "
                "log-weight carryover VJP while reducing different factors"
            ),
            {
                "target_mode": _evidence_row(
                    "target_transport_log_weights_stop_gradient",
                    target,
                    [target_factor, composed_field],
                ),
                "previous_mode": _evidence_row(
                    "previous_carry_log_weights_stop_gradient",
                    previous,
                    [previous_factor, composed_field],
                ),
            },
        )
    if (
        target[target_factor]["material_reduction"]
        and target[composed_field]["material_reduction"]
    ):
        return (
            "h3_target_upstream_edge",
            "target-time transport log-weight input reduces target upstream and composed VJP",
            _evidence_row(
                "target_transport_log_weights_stop_gradient",
                target,
                [target_factor, composed_field],
            ),
        )
    if (
        previous[previous_factor]["material_reduction"]
        and previous[composed_field]["material_reduction"]
    ):
        return (
            "h4_previous_carry_jacobian_edge",
            "previous carry log-weight boundary reduces previous carry Jacobian and composed VJP",
            _evidence_row(
                "previous_carry_log_weights_stop_gradient",
                previous,
                [previous_factor, composed_field],
            ),
        )
    only_all_times = (
        all_times[composed_field]["material_reduction"]
        and not target[composed_field]["material_reduction"]
        and not previous[composed_field]["material_reduction"]
    )
    if only_all_times:
        return (
            "h5_all_times_transport_only",
            "only all-times transport-log-weight stopping reduces the composed VJP",
            _evidence_row(
                "all_times_transport_log_weights_stop_gradient",
                all_times,
                [composed_field],
            ),
        )
    return (
        "h6_unresolved_log_weight_edge",
        "finite value-valid evidence did not isolate the log-weight edge factorization",
        {},
    )


def _evidence_row(
    mode: str,
    rows: dict[str, dict[str, Any]],
    fields: list[str],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "fields": {
            field: {
                "raw_bayesfilter_max_abs": rows[field]["raw_bayesfilter_max_abs"],
                "mode_max_abs": rows[field]["mode_max_abs"],
                "filterflow_max_abs": rows[field]["filterflow_max_abs"],
                "comparator_status": rows[field]["comparator_status"],
                "reduction_from_raw": rows[field]["reduction_from_raw"],
                "material_reduction": rows[field]["material_reduction"],
                "collapse": rows[field]["collapse"],
            }
            for field in fields
        },
    }


def _filterflow_edge_summaries(filterflow: dict[str, Any]) -> dict[str, Any]:
    decomp = filterflow["resampling_adjoint_decomposition"]
    summaries = {
        "target_to_pre_log_weights": filterflow["gradients"]["pre_log_weights"],
        "pre_log_weights_to_pre_particles": {
            "finite": True,
            "max_abs": None,
            "shape": None,
            "sum": None,
        },
        "same_tape_pre_log_weights_carryover_vjp": decomp[
            "same_tape_pre_log_weights_carryover_vjp"
        ],
        "same_tape_full_recorded_state_residual": decomp[
            "same_tape_full_recorded_state_residual"
        ],
        "same_tape_identity_residual": decomp["same_tape_identity_residual"],
        "same_tape_post_state_identity_residual": decomp[
            "same_tape_post_state_identity_residual"
        ],
    }
    tensors = {
        "target_to_pre_log_weights": filterflow["gradient_tensors"]["pre_log_weights"],
        "pre_log_weights_to_pre_particles": None,
        "same_tape_pre_log_weights_carryover_vjp": decomp[
            "same_tape_pre_log_weights_carryover_vjp_tensor"
        ],
        "same_tape_full_recorded_state_residual": decomp[
            "same_tape_full_recorded_state_residual_tensor"
        ],
        "same_tape_identity_residual": decomp["same_tape_identity_residual_tensor"],
        "same_tape_post_state_identity_residual": decomp[
            "same_tape_post_state_identity_residual_tensor"
        ],
    }
    return {"summaries": summaries, "tensors": tensors}


def _mode_by_name(
    modes: list[dict[str, Any]],
    name: str,
) -> dict[str, Any]:
    for mode in modes:
        if mode["mode"] == name:
            return mode
    raise KeyError(name)


def _veto_status(
    comparison: dict[str, Any],
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
    reference_status: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    cpu_rows = {
        "parent": _cpu_status(
            environment_manifest(
                command="parent-cpu-check",
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            )
        ),
        "filterflow": _cpu_status(filterflow["cpu_only_manifest"]),
        "raw_bayesfilter": _cpu_status(raw_bayesfilter["cpu_only_manifest"]),
    }
    for mode in mode_results:
        cpu_rows[f"mode:{mode['mode']}"] = _cpu_status(mode["cpu_only_manifest"])
    required_tensors_present = _required_edge_tensors_present(
        filterflow,
        raw_bayesfilter,
        mode_results,
    )
    edge_tensors_finite = _edge_tensors_finite(
        filterflow,
        raw_bayesfilter,
        mode_results,
    )
    mode_scalar_gates_pass = all(
        abs(float(mode["target_scalar"]) - float(filterflow["target_scalar"]))
        <= VALUE_TOLERANCE
        for mode in mode_results
    )
    mode_resampling_flags_match = all(
        mode["resampling_flag"] == filterflow["resampling_flag"]
        for mode in mode_results
    )
    status = {
        "all_vetoes_clear": False,
        "comparator_drift": comparator_drift,
        "reference_status_validated": bool(reference_status),
        "scalar_gate_pass": comparison["scalar_delta"] <= VALUE_TOLERANCE,
        "value_path_gate_pass": (
            comparison["first_value_delta_over_tolerance"]["status"] == "no_delta"
        ),
        "raw_resampling_flags_match": bool(comparison["resampling_flags_match"]),
        "mode_scalar_gates_pass": mode_scalar_gates_pass,
        "mode_resampling_flags_match": mode_resampling_flags_match,
        "required_edge_tensors_present": required_tensors_present,
        "edge_tensors_finite": edge_tensors_finite,
        "cpu_only_pass": all(row["pass"] for row in cpu_rows.values()),
        "cpu_rows": cpu_rows,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "path_boundary_manifest": path_boundary,
    }
    status["all_vetoes_clear"] = bool(
        not status["comparator_drift"]
        and status["scalar_gate_pass"]
        and status["value_path_gate_pass"]
        and status["raw_resampling_flags_match"]
        and status["mode_scalar_gates_pass"]
        and status["mode_resampling_flags_match"]
        and status["required_edge_tensors_present"]
        and status["edge_tensors_finite"]
        and status["cpu_only_pass"]
        and status["path_boundary_clean"]
    )
    return status


def _required_edge_tensors_present(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
) -> bool:
    ff_edges = _filterflow_edge_summaries(filterflow)
    for field in EDGE_FIELDS:
        if field not in ff_edges["summaries"] or field not in ff_edges["tensors"]:
            return False
        if field != "pre_log_weights_to_pre_particles" and ff_edges["tensors"][field] is None:
            return False
    if "pre_log_weights" not in raw_bayesfilter.get("gradients", {}):
        return False
    for mode in mode_results:
        if mode is None or mode.get("status") != "executed":
            return False
        for field in EDGE_FIELDS:
            if field not in mode.get("edge_summaries", {}):
                return False
            if field not in mode.get("edge_tensors", {}):
                return False
    return True


def _edge_tensors_finite(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
) -> bool:
    del raw_bayesfilter
    ff_edges = _filterflow_edge_summaries(filterflow)
    for field in EDGE_FIELDS:
        if field == "pre_log_weights_to_pre_particles":
            continue
        if not ff_edges["summaries"][field]["finite"]:
            return False
    for mode in mode_results:
        for field in EDGE_FIELDS:
            if not mode["edge_summaries"][field]["finite"]:
                return False
    return True


def _cpu_status(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        "pre_import_cuda_visible_devices": manifest.get("pre_import_cuda_visible_devices"),
        "gpu_devices_visible": manifest.get("gpu_devices_visible"),
        "pass": (
            manifest.get("cuda_visible_devices") == "-1"
            and manifest.get("pre_import_cuda_visible_devices") == "-1"
            and manifest.get("gpu_devices_visible") == []
        ),
    }


def _decision(classification: str, veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_log_weight_edge_factorization_blocked_or_vetoed"
    mapping = {
        "h1_blocked_or_vetoed": (
            "filterflow_float64_row_173_log_weight_edge_factorization_blocked_or_vetoed"
        ),
        "h2_composition_edge": (
            "filterflow_float64_row_173_log_weight_edge_factorization_composition_edge"
        ),
        "h3_target_upstream_edge": (
            "filterflow_float64_row_173_log_weight_edge_factorization_target_upstream_edge"
        ),
        "h4_previous_carry_jacobian_edge": (
            "filterflow_float64_row_173_log_weight_edge_factorization_previous_carry_jacobian_edge"
        ),
        "h5_all_times_transport_only": (
            "filterflow_float64_row_173_log_weight_edge_factorization_all_times_transport_only"
        ),
        "h6_unresolved_log_weight_edge": (
            "filterflow_float64_row_173_log_weight_edge_factorization_unresolved"
        ),
    }
    return mapping[classification]


def _prior_carryover_split() -> dict[str, Any]:
    if not PRIOR_CARRYOVER_JSON.exists():
        return {"status": "missing", "path": str(PRIOR_CARRYOVER_JSON)}
    payload = load_json(PRIOR_CARRYOVER_JSON)
    return {
        "status": "loaded",
        "path": str(PRIOR_CARRYOVER_JSON),
        "decision": payload.get("decision"),
        "hypothesis_classification": payload.get("hypothesis_classification"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
    }


def _mode_description(mode: str) -> str:
    descriptions = {
        "raw": "No extra BayesFilter graph boundary.",
        "target_transport_log_weights_stop_gradient": (
            "At target time 93 only, stop gradient through log weights as the "
            "transport input."
        ),
        "previous_carry_log_weights_stop_gradient": (
            "After previous time 92 update only, stop gradient through carried "
            "normalized log weights."
        ),
        "previous_and_target_log_weights_stop_gradient": (
            "Stop previous time 92 carried normalized log weights and target "
            "time 93 transport log-weight input."
        ),
        "all_times_transport_log_weights_stop_gradient": (
            "At every time, stop gradient through log weights as the transport input."
        ),
    }
    if mode not in descriptions:
        raise ValueError(f"unknown mode: {mode}")
    return descriptions[mode]


def _model_contract() -> dict[str, Any]:
    config = vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    return {
        **vjp._model_contract(config),
        "previous_time_index": PREVIOUS_TIME_INDEX,
        "factor_modes": list(MODES),
        "edge_fields": list(EDGE_FIELDS),
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Does row-173 target-time-93 BayesFilter log-weight carryover "
            "residual factorize into target upstream, previous carry Jacobian, "
            "their composition, or an unresolved route?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "primary_criterion": "classify log-weight edge factorization after vetoes clear",
        "vetoes": [
            "FilterFlow or BayesFilter execution blocker",
            "comparator drift",
            "CPU-only manifest violation",
            "scalar or value-path mismatch",
            "resampling flag mismatch",
            "missing or non-finite edge tensors",
            "path-boundary contamination",
        ],
        "explanatory_only": [
            "total-gradient collapse or worsening",
            "per-mode gradient delta against raw FilterFlow",
            "mode residual magnitude when no material reduction occurs",
        ],
        "not_concluded": _non_implications(),
    }


def _decision_table(
    decision: str,
    comparison: dict[str, Any],
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "Decision": decision,
            "Primary criterion status": comparison.get("classification", "blocked"),
            "Veto diagnostic status": json.dumps(
                {key: value for key, value in veto_status.items() if key != "cpu_rows"},
                sort_keys=True,
            ),
            "Main uncertainty": "single row and target time; no correctness claim",
            "Next justified action": _next_action(comparison.get("classification")),
            "Not concluded": "correctness, posterior correctness, production readiness, global agreement",
        }
    ]


def _next_action(classification: str | None) -> str:
    if classification == "h2_composition_edge":
        return "inspect exact BayesFilter previous-log-weight carry Jacobian and target transport-log-weight upstream against FilterFlow code"
    if classification == "h3_target_upstream_edge":
        return "inspect BayesFilter target-time transport log-weight upstream route"
    if classification == "h4_previous_carry_jacobian_edge":
        return "inspect BayesFilter previous normalized-log-weight carry Jacobian"
    if classification == "h5_all_times_transport_only":
        return "scan earlier transport-log-weight inputs for first residual time"
    if classification == "h6_unresolved_log_weight_edge":
        return "add tensor-entry localization for log-weight edge factors"
    return "repair blocker before interpreting log-weight edge evidence"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "created_at_utc",
        "question",
        "decision",
        "hypothesis_classification",
        "hypothesis_reason",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "filterflow_vjp",
        "raw_bayesfilter_vjp",
        "bayesfilter_mode_vjps",
        "base_vjp_comparison",
        "log_weight_edge_comparison",
        "veto_status_table",
        "path_boundary_manifest",
        "run_manifest",
        "decision_table",
        "non_implications",
        "reproducibility_digest",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    validate_filterflow_reference_status(
        payload["filterflow_status"],
        marker_path=vjp.FILTERFLOW_MARKER_PATH,
    )
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    if payload["decision"].endswith("blocked_or_vetoed"):
        return
    if not payload["veto_status_table"]["all_vetoes_clear"]:
        raise ValueError("non-blocked decision with uncleared vetoes")
    for label in ("filterflow_vjp", "raw_bayesfilter_vjp"):
        _validate_cpu(payload[label]["cpu_only_manifest"], label)
    for mode in payload["bayesfilter_mode_vjps"]:
        _validate_cpu(mode["cpu_only_manifest"], f"mode:{mode['mode']}")
    if payload["base_vjp_comparison"]["scalar_delta"] > VALUE_TOLERANCE:
        raise ValueError("scalar delta exceeded tolerance")
    if payload["base_vjp_comparison"]["first_value_delta_over_tolerance"]["status"] != "no_delta":
        raise ValueError("value path mismatch before edge interpretation")
    if not payload["veto_status_table"]["mode_scalar_gates_pass"]:
        raise ValueError("mode scalar gate failed")
    if not payload["veto_status_table"]["mode_resampling_flags_match"]:
        raise ValueError("mode resampling flags mismatch")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Result: Row 173 Log-Weight Edge Factorization Probe",
            "",
            "## Decision",
            "",
            f"`{payload['decision']}`",
            "",
            "## Hypothesis Classification",
            "",
            f"`{payload['hypothesis_classification']}`",
            "",
            payload["hypothesis_reason"],
            "",
            "## Decision Table",
            "",
            "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                "| {Decision} | {Primary criterion status} | {Veto diagnostic status} | {Main uncertainty} | {Next justified action} | {Not concluded} |".format(
                    **row
                )
                for row in payload["decision_table"]
            ],
            "",
            "## Log-Weight Edge Comparison",
            "",
            _json_block(_report_comparison(payload["log_weight_edge_comparison"])),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## Base VJP Comparison",
            "",
            _json_block(
                {
                    "total_gradient_delta": payload["base_vjp_comparison"][
                        "total_gradient_delta"
                    ],
                    "max_abs_total_gradient_delta": payload["base_vjp_comparison"][
                        "max_abs_total_gradient_delta"
                    ],
                    "scalar_delta": payload["base_vjp_comparison"]["scalar_delta"],
                    "first_value_delta_over_tolerance": payload["base_vjp_comparison"][
                        "first_value_delta_over_tolerance"
                    ],
                    "first_gradient_delta_over_tolerance": payload["base_vjp_comparison"][
                        "first_gradient_delta_over_tolerance"
                    ],
                    "resampling_flags_match": payload["base_vjp_comparison"][
                        "resampling_flags_match"
                    ],
                }
            ),
            "",
            "## Prior Carryover Split",
            "",
            _json_block(payload["prior_carryover_split"]),
            "",
            "## Model Contract",
            "",
            _json_block(payload["model_contract"]),
            "",
            "## Run Manifest",
            "",
            _json_block(payload["run_manifest"]),
            "",
            "## Non-Implications",
            "",
            "\n".join(f"- {item}" for item in payload["non_implications"]),
            "",
            "## Reproducibility Digest",
            "",
            f"`{payload.get('reproducibility_digest')}`",
            "",
        ]
    )


def _report_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
    mode_summary = {}
    for mode, row in comparison.get("mode_rows", {}).items():
        mode_summary[mode] = {
            "value_valid": row["value_valid"],
            "scalar_delta": row["scalar_delta"],
            "gradient_delta": row["gradient_delta"],
            "max_abs_gradient_delta": row["max_abs_gradient_delta"],
            "field_rows": {
                field: row["field_rows"][field] for field in EDGE_FIELDS
            },
        }
    return {
        "classification": comparison.get("classification"),
        "classification_reason": comparison.get("classification_reason"),
        "classification_evidence": comparison.get("classification_evidence"),
        "raw_bayesfilter_edge_max_abs": comparison.get(
            "raw_bayesfilter_edge_max_abs"
        ),
        "filterflow_edge_max_abs": comparison.get("filterflow_edge_max_abs"),
        "mode_summary": mode_summary,
        "decision_precedence": comparison.get("decision_precedence"),
    }


def _digest_payload(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("reproducibility_digest", None)
    return stable_digest(copy)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _non_implications() -> list[str]:
    return [
        "no correctness claim for FilterFlow or BayesFilter",
        "no analytic-gradient correctness claim",
        "no posterior correctness claim",
        "no global smoothness-surface agreement claim",
        "no claim that either implementation is mathematically authoritative",
        "no claim that any boundary mode is a code fix",
        "no production readiness or public API readiness",
        "no monograph, highdim, DSGE, NAWM, or banking/model-risk claim",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
