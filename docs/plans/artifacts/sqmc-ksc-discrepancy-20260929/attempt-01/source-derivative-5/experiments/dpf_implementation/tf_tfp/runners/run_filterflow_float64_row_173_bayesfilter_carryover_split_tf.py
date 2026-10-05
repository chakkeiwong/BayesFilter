"""Split row-173 BayesFilter carryover identity residuals by graph edge."""

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

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_float64_row_173_vjp_decomposition_tf as vjp,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_r3_float64_trace_replay_tf as r3,
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


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-bayesfilter-carryover-split-plan-2026-06-05.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-bayesfilter-carryover-split-result-2026-06-05.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-bayesfilter-carryover-split-review-loop-2026-06-05.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_bayesfilter_carryover_split_2026-06-05.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-bayesfilter-carryover-split-2026-06-05.md"
)
PRIOR_IDENTITY_ROUTE_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_state_update_identity_route_2026-06-05.json"
)

TARGET_TIME_INDEX = 93
TAG = "row-173-bayesfilter-carryover-split"
DTYPE = vjp.DTYPE
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
MODES = (
    "raw",
    "target_transport_log_weights_stop_gradient",
    "all_times_transport_log_weights_stop_gradient",
    "carry_log_weights_stop_gradient",
    "carry_log_likelihoods_stop_gradient",
    "carry_both_stop_gradient",
    "target_proposal_sample_filterflow_contract",
    "proposal_sample_noise_stop_gradient",
    "filterflow_custom_transport_gradient",
)
IDENTITY_FIELDS = (
    "same_tape_identity_residual",
    "same_tape_post_state_identity_residual",
    "same_tape_full_recorded_state_residual",
    "same_tape_pre_log_weights_carryover_vjp",
    "same_tape_pre_current_ll_carryover_vjp",
    "same_tape_post_particles_vjp",
    "same_tape_post_log_weights_vjp",
    "same_tape_transport_matrix_vjp",
    "same_tape_reconstructed_pre_particle_adjoint",
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
            "filterflow_float64_row_173_bayesfilter_carryover_split_filterflow_blocker",
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
            "filterflow_float64_row_173_bayesfilter_carryover_split_bayesfilter_blocker",
            raw_bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            raw_bayesfilter,
            None,
        )
    mode_results = [_mode_from_raw(raw_bayesfilter)]
    for mode in MODES:
        if mode == "raw":
            continue
        mode_results.append(_bayesfilter_mode_vjp(filterflow, config, mode))
    base_comparison = vjp._compare(filterflow, raw_bayesfilter)
    prior = _prior_identity_route()
    split_comparison = _split_comparison(
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
    decision = _decision(split_comparison["classification"], veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_bayesfilter_carryover_identity_residual_split",
        "decision": decision,
        "hypothesis_classification": split_comparison["classification"],
        "hypothesis_reason": split_comparison["classification_reason"],
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
        "bayesfilter_mode_vjps": [_compact_mode(mode) for mode in mode_results],
        "base_vjp_comparison": base_comparison,
        "carryover_split_comparison": split_comparison,
        "prior_identity_route": prior,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, split_comparison, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_bayesfilter_carryover_split_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH),
            "report_path": str(REPORT_PATH),
            "target_time_index": TARGET_TIME_INDEX,
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
        "question": "row_173_bayesfilter_carryover_identity_residual_split",
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
        "carryover_split_comparison": comparison,
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
                "run_filterflow_float64_row_173_bayesfilter_carryover_split_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
    }


def _bayesfilter_mode_vjp(
    filterflow: dict[str, Any],
    config: vjp.RunConfig,
    boundary_mode: str,
) -> dict[str, Any]:
    original_dtype = vjp.annealed_transport_tf.DTYPE
    vjp.annealed_transport_tf.DTYPE = DTYPE
    try:
        theta_variable = tf.Variable(vjp.THETA, dtype=DTYPE)
        model = vjp._model_from_filterflow(filterflow)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta_variable)
            bundle = vjp._bayesfilter_target_bundle(
                theta_variable,
                model,
                config,
                boundary_mode=boundary_mode,
            )
            target = bundle["target"]
        total_gradient = vjp._safe_gradient(tape, target, theta_variable)
        gradients = {
            name: vjp._safe_gradient(tape, target, tensor)
            for name, tensor in bundle.items()
            if name not in vjp.TARGET_FIELD_EXCLUSIONS
        }
        decomp = _identity_decomposition(tape, bundle, gradients)
        transport_upstream_clip_fraction = vjp._float(
            tf.reduce_mean(
                tf.cast(
                    tf.abs(gradients["transport_matrix"]) > tf.constant(1.0, DTYPE),
                    DTYPE,
                )
            )
        )
        del tape
        return {
            "status": "executed",
            "backend": "tensorflow_tensorflow_probability",
            "mode": boundary_mode,
            "mode_description": vjp._boundary_mode_description(boundary_mode),
            "settings": filterflow["settings"],
            "target_scalar": vjp._float(target),
            "total_gradient_diag": r3._json(total_gradient),
            "finite_scalar": bool(tf.math.is_finite(target).numpy()),
            "finite_gradient": bool(
                tf.reduce_all(tf.math.is_finite(total_gradient)).numpy()
            ),
            "resampling_flag": [
                bool(value) for value in tf.reshape(bundle["flags"], [-1]).numpy().tolist()
            ],
            "resampling_adjoint_decomposition": decomp,
            "transport_upstream_clip_fraction": transport_upstream_clip_fraction,
            "cpu_only_manifest": vjp._parent_cpu_manifest(),
        }
    finally:
        vjp.annealed_transport_tf.DTYPE = original_dtype


def _mode_from_raw(raw_bayesfilter: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": raw_bayesfilter["status"],
        "backend": raw_bayesfilter["backend"],
        "mode": "raw",
        "mode_description": vjp._boundary_mode_description("raw"),
        "settings": raw_bayesfilter["settings"],
        "target_scalar": raw_bayesfilter["target_scalar"],
        "total_gradient_diag": raw_bayesfilter["total_gradient_diag"],
        "finite_scalar": True,
        "finite_gradient": True,
        "resampling_flag": raw_bayesfilter["resampling_flag"],
        "resampling_adjoint_decomposition": raw_bayesfilter[
            "resampling_adjoint_decomposition"
        ],
        "transport_upstream_clip_fraction": raw_bayesfilter[
            "transport_upstream_clip_fraction"
        ],
        "cpu_only_manifest": raw_bayesfilter["cpu_only_manifest"],
    }


def _identity_decomposition(
    tape: tf.GradientTape,
    bundle: dict[str, tf.Tensor],
    gradients: dict[str, tf.Tensor],
) -> dict[str, Any]:
    direct_pre_particle_adjoint = tf.linalg.matmul(
        bundle["transport_matrix"],
        gradients["post_particles"],
        transpose_a=True,
    )
    same_tape_post_particles_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["post_particles"],
        bundle["pre_particles"],
        gradients["post_particles"],
    )
    same_tape_post_log_weights_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["post_log_weights"],
        bundle["pre_particles"],
        gradients["post_log_weights"],
    )
    same_tape_post_state_vjp = (
        same_tape_post_particles_vjp + same_tape_post_log_weights_vjp
    )
    same_tape_post_state_identity_residual = (
        gradients["pre_particles"] - same_tape_post_state_vjp
    )
    same_tape_pre_log_weights_carryover_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["pre_log_weights"],
        bundle["pre_particles"],
        gradients["pre_log_weights"],
    )
    same_tape_pre_current_ll_carryover_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["pre_current_log_likelihoods"],
        bundle["pre_particles"],
        gradients["pre_current_log_likelihoods"],
    )
    same_tape_log_ess_carryover_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["log_ess"],
        bundle["pre_particles"],
        gradients["log_ess"],
    )
    same_tape_full_recorded_state_vjp = (
        same_tape_post_state_vjp
        + same_tape_pre_log_weights_carryover_vjp
        + same_tape_pre_current_ll_carryover_vjp
        + same_tape_log_ess_carryover_vjp
    )
    same_tape_full_recorded_state_residual = (
        gradients["pre_particles"] - same_tape_full_recorded_state_vjp
    )
    same_tape_transport_matrix_vjp = vjp._safe_gradient_with_upstream(
        tape,
        bundle["transport_matrix"],
        bundle["pre_particles"],
        gradients["transport_matrix"],
    )
    same_tape_reconstructed_pre_particle_adjoint = (
        direct_pre_particle_adjoint + same_tape_transport_matrix_vjp
    )
    same_tape_identity_residual = (
        gradients["pre_particles"] - same_tape_reconstructed_pre_particle_adjoint
    )
    tensors = {
        "direct_pre_particle_adjoint": direct_pre_particle_adjoint,
        "same_tape_post_particles_vjp": same_tape_post_particles_vjp,
        "same_tape_post_log_weights_vjp": same_tape_post_log_weights_vjp,
        "same_tape_post_state_vjp": same_tape_post_state_vjp,
        "same_tape_post_state_identity_residual": (
            same_tape_post_state_identity_residual
        ),
        "same_tape_pre_log_weights_carryover_vjp": (
            same_tape_pre_log_weights_carryover_vjp
        ),
        "same_tape_pre_current_ll_carryover_vjp": (
            same_tape_pre_current_ll_carryover_vjp
        ),
        "same_tape_log_ess_carryover_vjp": same_tape_log_ess_carryover_vjp,
        "same_tape_full_recorded_state_vjp": same_tape_full_recorded_state_vjp,
        "same_tape_full_recorded_state_residual": (
            same_tape_full_recorded_state_residual
        ),
        "same_tape_transport_matrix_vjp": same_tape_transport_matrix_vjp,
        "same_tape_reconstructed_pre_particle_adjoint": (
            same_tape_reconstructed_pre_particle_adjoint
        ),
        "same_tape_identity_residual": same_tape_identity_residual,
    }
    decomp = {name: vjp._field(tensor) for name, tensor in tensors.items()}
    decomp.update({f"{name}_tensor": r3._json(tensor) for name, tensor in tensors.items()})
    return decomp


def _split_comparison(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
    base_comparison: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    raw_decomp = raw_bayesfilter["resampling_adjoint_decomposition"]
    ff_decomp = filterflow["resampling_adjoint_decomposition"]
    mode_rows = {}
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
            field: _mode_field_row(
                field,
                ff_decomp,
                raw_decomp,
                mode_result["resampling_adjoint_decomposition"],
                value_valid,
            )
            for field in IDENTITY_FIELDS
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
    classification, reason, evidence = _classify_split(mode_rows)
    precedence_audit = _precedence_audit(mode_rows)
    return {
        "status": "compared",
        "classification": classification,
        "classification_reason": reason,
        "classification_evidence": evidence,
        "ordered_rule_audit": precedence_audit,
        "mode_rows": mode_rows,
        "raw_bayesfilter_identity_max_abs": {
            field: raw_decomp[field]["max_abs"] for field in IDENTITY_FIELDS
        },
        "filterflow_identity_max_abs": {
            field: ff_decomp[field]["max_abs"] for field in IDENTITY_FIELDS
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
        "prior_identity_route_decision": prior.get("decision"),
        "prior_identity_route_classification": prior.get("hypothesis_classification"),
        "decision_precedence": [
            "h1 if missing/non-finite raw or mode identity tensors",
            "h2 if target transport-log-weight input mode reduces raw identity residuals",
            "h3 if carried log-weight mode reduces raw full-state/log-weight residuals",
            "h4 if carried log-likelihood mode reduces raw likelihood residuals",
            "h5 if proposal sample/post-particle modes reduce raw identity residuals",
            "h6 if custom transport gradient leaves residuals material",
            "h7 otherwise",
        ],
    }


def _mode_field_row(
    field: str,
    ff_decomp: dict[str, Any],
    raw_decomp: dict[str, Any],
    mode_decomp: dict[str, Any],
    value_valid: bool,
) -> dict[str, Any]:
    raw_max = float(raw_decomp[field]["max_abs"])
    mode_max = float(mode_decomp[field]["max_abs"])
    reduction = raw_max - mode_max
    return {
        "field": field,
        "filterflow_max_abs": ff_decomp[field]["max_abs"],
        "raw_bayesfilter_max_abs": raw_max,
        "mode_max_abs": mode_max,
        "mode_minus_filterflow_max_abs_delta": vjp._max_abs_nested_delta(
            mode_decomp[f"{field}_tensor"],
            ff_decomp[f"{field}_tensor"],
        ),
        "reduction_from_raw": reduction,
        "material_reduction": bool(value_valid and reduction > GRADIENT_TOLERANCE),
        "collapse": bool(value_valid and mode_max <= GRADIENT_TOLERANCE),
        "raw_material": bool(raw_max > GRADIENT_TOLERANCE),
        "mode_material": bool(mode_max > GRADIENT_TOLERANCE),
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _classify_split(
    mode_rows: dict[str, dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    missing_or_nonfinite = [
        mode
        for mode, row in mode_rows.items()
        if not row["value_valid"]
    ]
    if missing_or_nonfinite:
        return (
            "h1_blocked_or_vetoed",
            "at least one required mode failed value/finiteness/resampling gates",
            {"invalid_modes": missing_or_nonfinite},
        )

    h2 = _first_material_reduction(
        mode_rows,
        ["target_transport_log_weights_stop_gradient"],
        ["same_tape_identity_residual", "same_tape_post_state_identity_residual"],
    )
    if h2 is not None:
        return (
            "h2_target_transport_log_weight_edge",
            "target-time transport log-weight input materially reduces raw identity residual",
            h2,
        )
    h3 = _first_material_reduction(
        mode_rows,
        ["carry_log_weights_stop_gradient", "all_times_transport_log_weights_stop_gradient"],
        [
            "same_tape_full_recorded_state_residual",
            "same_tape_pre_log_weights_carryover_vjp",
        ],
    )
    if h3 is not None:
        return (
            "h3_carried_log_weight_edge",
            "carried log-weight boundary materially reduces raw full-state/log-weight residual",
            h3,
        )
    h4 = _first_material_reduction(
        mode_rows,
        ["carry_log_likelihoods_stop_gradient", "carry_both_stop_gradient"],
        [
            "same_tape_identity_residual",
            "same_tape_post_state_identity_residual",
            "same_tape_pre_current_ll_carryover_vjp",
        ],
    )
    if h4 is not None:
        return (
            "h4_carried_log_likelihood_edge",
            "carried log-likelihood boundary materially reduces raw likelihood residual",
            h4,
        )
    h5 = _first_material_reduction(
        mode_rows,
        [
            "target_proposal_sample_filterflow_contract",
            "proposal_sample_noise_stop_gradient",
        ],
        [
            "same_tape_identity_residual",
            "same_tape_post_state_identity_residual",
            "same_tape_post_particles_vjp",
        ],
    )
    if h5 is not None:
        return (
            "h5_post_particle_sample_edge",
            "proposal sample/post-particle boundary materially reduces raw residual",
            h5,
        )
    custom = mode_rows["filterflow_custom_transport_gradient"]["field_rows"]
    custom_material = [
        field
        for field in (
            "same_tape_identity_residual",
            "same_tape_post_state_identity_residual",
            "same_tape_full_recorded_state_residual",
        )
        if custom[field]["mode_material"]
    ]
    if custom_material:
        return (
            "h6_transport_custom_gradient_not_edge",
            "whole-transport custom-gradient mode leaves identity residuals material",
            {"mode": "filterflow_custom_transport_gradient", "fields": custom_material},
        )
    return (
        "h7_unresolved_split",
        "finite value-valid split did not isolate a single BayesFilter edge",
        {},
    )


def _precedence_audit(mode_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target_transport = mode_rows["target_transport_log_weights_stop_gradient"][
        "field_rows"
    ]
    h2_fields = [
        "same_tape_identity_residual",
        "same_tape_post_state_identity_residual",
    ]
    h3_log_weight_fields = [
        "same_tape_full_recorded_state_residual",
        "same_tape_pre_log_weights_carryover_vjp",
    ]
    return {
        "h2_target_transport_fields": {
            field: {
                "mode_max_abs": target_transport[field]["mode_max_abs"],
                "raw_bayesfilter_max_abs": target_transport[field][
                    "raw_bayesfilter_max_abs"
                ],
                "reduction_from_raw": target_transport[field]["reduction_from_raw"],
                "material_reduction": target_transport[field][
                    "material_reduction"
                ],
                "collapse": target_transport[field]["collapse"],
            }
            for field in h2_fields
        },
        "target_transport_log_weight_fields_explanatory_only": {
            field: {
                "mode_max_abs": target_transport[field]["mode_max_abs"],
                "raw_bayesfilter_max_abs": target_transport[field][
                    "raw_bayesfilter_max_abs"
                ],
                "reduction_from_raw": target_transport[field]["reduction_from_raw"],
                "material_reduction": target_transport[field][
                    "material_reduction"
                ],
                "collapse": target_transport[field]["collapse"],
            }
            for field in h3_log_weight_fields
        },
        "h2_did_not_fire_reason": (
            "target_transport_log_weights_stop_gradient did not materially "
            "reduce same_tape_identity_residual or "
            "same_tape_post_state_identity_residual under the accepted ordered "
            "rule, even though it collapses the full-recorded-state/log-weight "
            "fields as explanatory evidence"
        ),
    }


def _first_material_reduction(
    mode_rows: dict[str, dict[str, Any]],
    modes: list[str],
    fields: list[str],
) -> dict[str, Any] | None:
    for mode in modes:
        rows = mode_rows[mode]["field_rows"]
        for field in fields:
            if rows[field]["material_reduction"]:
                return {
                    "mode": mode,
                    "field": field,
                    "raw_bayesfilter_max_abs": rows[field]["raw_bayesfilter_max_abs"],
                    "mode_max_abs": rows[field]["mode_max_abs"],
                    "filterflow_max_abs": rows[field]["filterflow_max_abs"],
                    "reduction_from_raw": rows[field]["reduction_from_raw"],
                    "collapse": rows[field]["collapse"],
                    "mode_gradient_delta": mode_rows[mode]["gradient_delta"],
                    "mode_max_abs_gradient_delta": mode_rows[mode][
                        "max_abs_gradient_delta"
                    ],
                }
    return None


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
    required_tensors_present = _required_identity_tensors_present(
        filterflow,
        raw_bayesfilter,
        mode_results,
    )
    identity_tensors_finite = _identity_tensors_finite(
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
        "required_identity_tensors_present": required_tensors_present,
        "identity_tensors_finite": identity_tensors_finite,
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
        and status["required_identity_tensors_present"]
        and status["identity_tensors_finite"]
        and status["cpu_only_pass"]
        and status["path_boundary_clean"]
    )
    return status


def _required_identity_tensors_present(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
) -> bool:
    sides = [filterflow, raw_bayesfilter, *mode_results]
    for side in sides:
        if side is None or side.get("status") != "executed":
            return False
        decomp = side.get("resampling_adjoint_decomposition", {})
        for field in IDENTITY_FIELDS:
            if field not in decomp or f"{field}_tensor" not in decomp:
                return False
    return True


def _identity_tensors_finite(
    filterflow: dict[str, Any],
    raw_bayesfilter: dict[str, Any],
    mode_results: list[dict[str, Any]],
) -> bool:
    for side in [filterflow, raw_bayesfilter, *mode_results]:
        decomp = side["resampling_adjoint_decomposition"]
        for field in IDENTITY_FIELDS:
            if not decomp[field]["finite"]:
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
        return "filterflow_float64_row_173_bayesfilter_carryover_split_blocked_or_vetoed"
    mapping = {
        "h1_blocked_or_vetoed": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_blocked_or_vetoed"
        ),
        "h2_target_transport_log_weight_edge": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_target_transport_log_weight_edge"
        ),
        "h3_carried_log_weight_edge": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_carried_log_weight_edge"
        ),
        "h4_carried_log_likelihood_edge": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_carried_log_likelihood_edge"
        ),
        "h5_post_particle_sample_edge": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_post_particle_sample_edge"
        ),
        "h6_transport_custom_gradient_not_edge": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_transport_custom_gradient_not_edge"
        ),
        "h7_unresolved_split": (
            "filterflow_float64_row_173_bayesfilter_carryover_split_unresolved"
        ),
    }
    return mapping[classification]


def _prior_identity_route() -> dict[str, Any]:
    if not PRIOR_IDENTITY_ROUTE_JSON.exists():
        return {"status": "missing", "path": str(PRIOR_IDENTITY_ROUTE_JSON)}
    payload = load_json(PRIOR_IDENTITY_ROUTE_JSON)
    return {
        "status": "loaded",
        "path": str(PRIOR_IDENTITY_ROUTE_JSON),
        "decision": payload.get("decision"),
        "hypothesis_classification": payload.get("hypothesis_classification"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
    }


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
        "split_modes": list(MODES),
        "identity_fields": list(IDENTITY_FIELDS),
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Which BayesFilter graph edge carries the row-173 target-time-93 "
            "identity residual relative to local executable float64 FilterFlow?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "primary_criterion": "classify identity-residual reduction by named BayesFilter boundary mode",
        "vetoes": [
            "FilterFlow or BayesFilter execution blocker",
            "comparator drift",
            "CPU-only manifest violation",
            "scalar or value-path mismatch",
            "resampling flag mismatch",
            "missing or non-finite identity tensors",
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
    if classification == "h2_target_transport_log_weight_edge":
        return "inspect BayesFilter target-time transport log-weight input gradient route"
    if classification == "h3_carried_log_weight_edge":
        return "inspect BayesFilter carried log-weight update boundary and previous-time carryover"
    if classification == "h4_carried_log_likelihood_edge":
        return "inspect BayesFilter carried cumulative log-likelihood boundary"
    if classification == "h5_post_particle_sample_edge":
        return "inspect BayesFilter post-particle proposal sample gradient path"
    if classification == "h6_transport_custom_gradient_not_edge":
        return "deprioritize whole-transport custom-gradient mismatch and split state/update carryover further"
    if classification == "h7_unresolved_split":
        return "add tensor-level row/particle localization for residual entries"
    return "repair blocker before interpreting carryover-split evidence"


def _compact_mode(mode: dict[str, Any] | None) -> dict[str, Any] | None:
    if mode is None or mode.get("status") != "executed":
        return mode
    return {
        "status": mode["status"],
        "backend": mode["backend"],
        "mode": mode["mode"],
        "mode_description": mode["mode_description"],
        "target_scalar": mode["target_scalar"],
        "total_gradient_diag": mode["total_gradient_diag"],
        "resampling_flag": mode["resampling_flag"],
        "finite_scalar": mode["finite_scalar"],
        "finite_gradient": mode["finite_gradient"],
        "transport_upstream_clip_fraction": mode["transport_upstream_clip_fraction"],
        "resampling_adjoint_decomposition": mode["resampling_adjoint_decomposition"],
        "cpu_only_manifest": mode["cpu_only_manifest"],
    }


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
        "carryover_split_comparison",
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
        raise ValueError("value path mismatch before split interpretation")
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
            "# Result: Row 173 BayesFilter Carryover Split Probe",
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
            "## Carryover Split Comparison",
            "",
            _json_block(_report_comparison(payload["carryover_split_comparison"])),
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
            "## Prior Identity Route",
            "",
            _json_block(payload["prior_identity_route"]),
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
                field: row["field_rows"][field]
                for field in (
                    "same_tape_identity_residual",
                    "same_tape_post_state_identity_residual",
                    "same_tape_full_recorded_state_residual",
                    "same_tape_pre_log_weights_carryover_vjp",
                    "same_tape_pre_current_ll_carryover_vjp",
                )
            },
        }
    return {
        "classification": comparison.get("classification"),
        "classification_reason": comparison.get("classification_reason"),
        "classification_evidence": comparison.get("classification_evidence"),
        "ordered_rule_audit": comparison.get("ordered_rule_audit"),
        "raw_bayesfilter_identity_max_abs": comparison.get(
            "raw_bayesfilter_identity_max_abs"
        ),
        "filterflow_identity_max_abs": comparison.get("filterflow_identity_max_abs"),
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
