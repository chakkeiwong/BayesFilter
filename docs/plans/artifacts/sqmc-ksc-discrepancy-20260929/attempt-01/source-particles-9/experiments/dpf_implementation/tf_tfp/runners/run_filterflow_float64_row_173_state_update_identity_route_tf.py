"""Probe row-173 downstream state/update identity-route residuals."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import time
from typing import Any

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
    "bayesfilter-dpf-filterflow-float64-row-173-state-update-identity-route-plan-2026-06-05.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-state-update-identity-route-result-2026-06-05.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-state-update-identity-route-review-loop-2026-06-05.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_state_update_identity_route_2026-06-05.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-state-update-identity-route-2026-06-05.md"
)
DOWNSTREAM_ROUTE_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_downstream_adjoint_route_2026-06-05.json"
)

TARGET_TIME_INDEX = 93
TAG = "row-173-state-update-identity-route"
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
IDENTITY_KEYS = (
    "same_tape_identity",
    "same_tape_post_state_identity",
    "same_tape_full_recorded_state_identity",
)
IDENTITY_RESIDUAL_FIELDS = {
    "same_tape_identity": "same_tape_identity_residual",
    "same_tape_post_state_identity": "same_tape_post_state_identity_residual",
    "same_tape_full_recorded_state_identity": "same_tape_full_recorded_state_residual",
}
IDENTITY_VJP_FIELDS = (
    "direct_pre_particle_adjoint",
    "same_tape_post_particles_vjp",
    "same_tape_post_log_weights_vjp",
    "same_tape_post_state_vjp",
    "same_tape_pre_log_weights_carryover_vjp",
    "same_tape_pre_current_ll_carryover_vjp",
    "same_tape_log_ess_carryover_vjp",
    "same_tape_full_recorded_state_vjp",
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
            "filterflow_float64_row_173_state_update_identity_route_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )
    bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_state_update_identity_route_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            bayesfilter,
        )
    bayesfilter_boundary_modes = vjp._bayesfilter_boundary_modes(filterflow, config)
    base_comparison = vjp._compare(filterflow, bayesfilter)
    boundary_mode_comparison = vjp._compare_boundary_modes(
        filterflow,
        bayesfilter_boundary_modes,
    )
    prior = _downstream_prior()
    identity_comparison = _identity_comparison(
        filterflow,
        bayesfilter,
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
        bayesfilter,
        reference_status,
        comparator_drift,
    )
    decision = _decision(identity_comparison["classification"], veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_state_update_identity_route_difference_audit",
        "decision": decision,
        "hypothesis_classification": identity_comparison["classification"],
        "hypothesis_reason": identity_comparison["classification_reason"],
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
        "bayesfilter_vjp": vjp._compact_side(bayesfilter),
        "base_vjp_comparison": base_comparison,
        "identity_route_comparison": identity_comparison,
        "bayesfilter_boundary_modes": bayesfilter_boundary_modes,
        "boundary_mode_comparison": boundary_mode_comparison,
        "downstream_prior": prior,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, identity_comparison, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_state_update_identity_route_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH),
            "report_path": str(REPORT_PATH),
            "target_time_index": TARGET_TIME_INDEX,
        },
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    reference_status: dict[str, Any],
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any] | None,
    bayesfilter: dict[str, Any] | None,
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {
        "status": "blocked",
        "classification": "h1_blocked_or_vetoed",
        "classification_reason": blocker,
    }
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_state_update_identity_route_difference_audit",
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
        "filterflow_vjp": filterflow,
        "bayesfilter_vjp": bayesfilter,
        "identity_route_comparison": comparison,
        "veto_status_table": {
            "all_vetoes_clear": False,
            "blocker": blocker,
        },
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, comparison, {"all_vetoes_clear": False}),
        "non_implications": _non_implications(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_state_update_identity_route_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
    }


def _identity_comparison(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    base_comparison: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    ff_decomp = filterflow["resampling_adjoint_decomposition"]
    bf_decomp = bayesfilter["resampling_adjoint_decomposition"]
    within_side = {
        "filterflow": _within_side_identity_rows(ff_decomp),
        "bayesfilter": _within_side_identity_rows(bf_decomp),
    }
    cross_side = _cross_side_identity_rows(base_comparison["adjoint_decomposition"])
    filterflow_material = [
        row for row in within_side["filterflow"].values() if row["material"]
    ]
    bayesfilter_material = [
        row for row in within_side["bayesfilter"].values() if row["material"]
    ]
    cross_side_material = [
        row for row in cross_side.values() if row["material_cross_delta"]
    ]
    if filterflow_material and bayesfilter_material:
        classification = "h3_both_identity_residuals_material"
        reason = "both sides have material within-side identity residuals"
    elif not filterflow_material and bayesfilter_material:
        classification = "h2_bayesfilter_identity_residual_filterflow_identity_clean"
        reason = (
            "FilterFlow identity residuals are clean while BayesFilter identity "
            "residuals are material"
        )
    elif not filterflow_material and not bayesfilter_material and cross_side_material:
        classification = "h4_identity_residual_not_material_cross_side_vjp_diff"
        reason = "within-side identities are clean but cross-side VJP rows differ"
    else:
        classification = "h5_unresolved_identity_route"
        reason = "identity residuals and cross-side rows do not isolate a route"
    return {
        "status": "compared",
        "classification": classification,
        "classification_reason": reason,
        "within_side_identity_rows": within_side,
        "cross_side_identity_rows": cross_side,
        "material_filterflow_identities": [row["identity"] for row in filterflow_material],
        "material_bayesfilter_identities": [row["identity"] for row in bayesfilter_material],
        "material_cross_side_vjp_rows": [row["field"] for row in cross_side_material],
        "total_gradient_delta": base_comparison["total_gradient_delta"],
        "max_abs_total_gradient_delta": base_comparison["max_abs_total_gradient_delta"],
        "scalar_delta": base_comparison["scalar_delta"],
        "first_value_delta_over_tolerance": base_comparison[
            "first_value_delta_over_tolerance"
        ],
        "first_gradient_delta_over_tolerance": base_comparison[
            "first_gradient_delta_over_tolerance"
        ],
        "downstream_prior_decision": prior.get("decision"),
        "downstream_prior_classification": prior.get("hypothesis_classification"),
        "decision_precedence": [
            "h3 if both sides have material identity residuals",
            "h2 if FilterFlow clean and BayesFilter material",
            "h4 if identities clean but cross-side VJPs differ",
            "h5 otherwise",
        ],
    }


def _within_side_identity_rows(decomp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {}
    for identity in IDENTITY_KEYS:
        residual_field = IDENTITY_RESIDUAL_FIELDS[identity]
        residual = decomp[residual_field]
        rows[identity] = {
            "identity": identity,
            "residual_field": residual_field,
            "max_abs": residual["max_abs"],
            "sum": residual["sum"],
            "finite": residual["finite"],
            "clean": bool(residual["finite"] and residual["max_abs"] <= GRADIENT_TOLERANCE),
            "material": bool(residual["finite"] and residual["max_abs"] > GRADIENT_TOLERANCE),
            "gradient_tolerance": GRADIENT_TOLERANCE,
        }
    return rows


def _cross_side_identity_rows(
    adjoint_comparison: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {}
    source_rows = adjoint_comparison["rows"]
    for field in (*IDENTITY_VJP_FIELDS, *IDENTITY_RESIDUAL_FIELDS.values()):
        row = source_rows[field]
        rows[field] = {
            "field": field,
            "status": row["status"],
            "filterflow_max_abs": row.get("filterflow_max_abs"),
            "bayesfilter_max_abs": row.get("bayesfilter_max_abs"),
            "max_abs_delta": row.get("max_abs_delta"),
            "material_cross_delta": bool(
                row.get("status") == "compared"
                and row.get("max_abs_delta") is not None
                and row["max_abs_delta"] > GRADIENT_TOLERANCE
            ),
            "gradient_tolerance": GRADIENT_TOLERANCE,
        }
    return rows


def _veto_status(
    comparison: dict[str, Any],
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    reference_status: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    required_tensors_present = _required_identity_tensors_present(filterflow, bayesfilter)
    finite_identities = _identity_tensors_finite(filterflow, bayesfilter)
    cpu_rows = {
        "filterflow": _cpu_status(filterflow["cpu_only_manifest"]),
        "bayesfilter": _cpu_status(bayesfilter["cpu_only_manifest"]),
    }
    status = {
        "all_vetoes_clear": False,
        "comparator_drift": comparator_drift,
        "reference_status_validated": bool(reference_status),
        "scalar_gate_pass": comparison["scalar_delta"] <= VALUE_TOLERANCE,
        "value_path_gate_pass": (
            comparison["first_value_delta_over_tolerance"]["status"] == "no_delta"
        ),
        "resampling_flags_match": bool(comparison["resampling_flags_match"]),
        "required_identity_tensors_present": required_tensors_present,
        "identity_tensors_finite": finite_identities,
        "cpu_only_pass": all(row["pass"] for row in cpu_rows.values()),
        "cpu_rows": cpu_rows,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "path_boundary_manifest": path_boundary,
    }
    status["all_vetoes_clear"] = bool(
        not status["comparator_drift"]
        and status["scalar_gate_pass"]
        and status["value_path_gate_pass"]
        and status["resampling_flags_match"]
        and status["required_identity_tensors_present"]
        and status["identity_tensors_finite"]
        and status["cpu_only_pass"]
        and status["path_boundary_clean"]
    )
    return status


def _required_identity_tensors_present(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> bool:
    for side in (filterflow, bayesfilter):
        decomp = side.get("resampling_adjoint_decomposition", {})
        for field in (*IDENTITY_VJP_FIELDS, *IDENTITY_RESIDUAL_FIELDS.values()):
            if field not in decomp:
                return False
            if f"{field}_tensor" not in decomp:
                return False
    return True


def _identity_tensors_finite(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> bool:
    for side in (filterflow, bayesfilter):
        decomp = side["resampling_adjoint_decomposition"]
        for field in (*IDENTITY_VJP_FIELDS, *IDENTITY_RESIDUAL_FIELDS.values()):
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
        return "filterflow_float64_row_173_state_update_identity_route_blocked_or_vetoed"
    mapping = {
        "h1_blocked_or_vetoed": (
            "filterflow_float64_row_173_state_update_identity_route_blocked_or_vetoed"
        ),
        "h2_bayesfilter_identity_residual_filterflow_identity_clean": (
            "filterflow_float64_row_173_state_update_identity_route_bayesfilter_identity_residual"
        ),
        "h3_both_identity_residuals_material": (
            "filterflow_float64_row_173_state_update_identity_route_both_identity_residuals"
        ),
        "h4_identity_residual_not_material_cross_side_vjp_diff": (
            "filterflow_float64_row_173_state_update_identity_route_cross_side_vjp_only"
        ),
        "h5_unresolved_identity_route": (
            "filterflow_float64_row_173_state_update_identity_route_unresolved"
        ),
    }
    return mapping[classification]


def _downstream_prior() -> dict[str, Any]:
    if not DOWNSTREAM_ROUTE_JSON.exists():
        return {"status": "missing", "path": str(DOWNSTREAM_ROUTE_JSON)}
    payload = load_json(DOWNSTREAM_ROUTE_JSON)
    return {
        "status": "loaded",
        "path": str(DOWNSTREAM_ROUTE_JSON),
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
        "identity_value_tolerance": VALUE_TOLERANCE,
        "identity_gradient_tolerance": GRADIENT_TOLERANCE,
        "identity_keys": list(IDENTITY_KEYS),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Do within-side state/update identity residuals explain the row-173 "
            "BayesFilter-vs-local-float64-FilterFlow gradient mismatch?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "primary_criterion": "classify identity residual route after vetoes clear",
        "vetoes": [
            "FilterFlow or BayesFilter VJP execution blocker",
            "comparator drift",
            "CPU-only manifest violation",
            "scalar or value-path mismatch",
            "resampling flag mismatch",
            "missing or non-finite identity tensors",
            "path-boundary contamination",
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
    if classification == "h2_bayesfilter_identity_residual_filterflow_identity_clean":
        return "inspect BayesFilter tape boundaries for carried state/log-weight identity routing"
    if classification == "h3_both_identity_residuals_material":
        return "avoid BayesFilter-only patch; instrument both implementations more narrowly"
    if classification == "h4_identity_residual_not_material_cross_side_vjp_diff":
        return "localize cross-side VJP expression differences inside clean identities"
    if classification == "h5_unresolved_identity_route":
        return "add a smaller identity subcomponent probe"
    return "repair blocker before interpreting identity-route evidence"


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
        "bayesfilter_vjp",
        "base_vjp_comparison",
        "identity_route_comparison",
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
    for label in ("filterflow_vjp", "bayesfilter_vjp"):
        _validate_cpu(payload[label]["cpu_only_manifest"], label)
    if payload["base_vjp_comparison"]["scalar_delta"] > VALUE_TOLERANCE:
        raise ValueError("scalar delta exceeded tolerance")
    if payload["base_vjp_comparison"]["first_value_delta_over_tolerance"]["status"] != "no_delta":
        raise ValueError("value path mismatch before identity interpretation")


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
            "# Result: Row 173 State/Update Identity-Route Probe",
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
            "## Identity Route Comparison",
            "",
            _json_block(payload["identity_route_comparison"]),
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
            "## Boundary Mode Summary",
            "",
            _json_block(payload["boundary_mode_comparison"]),
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
        "no claim that the mismatch is fixed",
        "no production readiness or public API readiness",
        "no monograph, highdim, DSGE, NAWM, or banking/model-risk claim",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
