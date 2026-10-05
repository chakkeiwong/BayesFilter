"""Compare direct scalar-additivity gradients for row 173."""

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
    FILTERFLOW_BRANCH_MARKER,
    reference_policy,
    validate_filterflow_reference_status,
)


PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-scalar-additivity-route-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-scalar-additivity-route-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_scalar_additivity_route_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-scalar-additivity-route-2026-06-04.md"
)
FILTERFLOW_MARKER_PATH = vjp.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
TAG = "row-173-scalar-additivity-route"
TARGET_TIME_INDEX = 93
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE
OBJECTIVES = (
    "post_update_mean",
    "sum_pre_current_plus_increment_mean",
    "pre_current_mean",
    "increment_mean",
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
    validate_filterflow_reference_status(reference_status, marker_path=FILTERFLOW_MARKER_PATH)
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
            "filterflow_float64_row_173_scalar_additivity_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )
    bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_scalar_additivity_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            bayesfilter,
        )

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    comparison = _compare_scalar_additivity(filterflow, bayesfilter)
    veto_status = _veto_status(filterflow, bayesfilter, comparison, comparator_drift)
    decision = _decision(comparison, veto_status)
    payload = {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_direct_scalar_additivity_route_difference_audit",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "comparison": comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_scalar_additivity_route_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "data_seed": vjp.DATA_SEED,
            "filter_seed": vjp.FILTER_SEED,
            "target_time_index": TARGET_TIME_INDEX,
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        },
        "decision_table": _decision_table(decision, comparison, veto_status),
        "non_implications": _non_implications(),
    }
    return payload


def _blocked_payload(
    decision: str,
    blocker: str,
    reference_status: dict[str, Any],
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any] | None,
    bayesfilter: dict[str, Any] | None,
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_direct_scalar_additivity_route_difference_audit",
        "blocker": blocker,
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "comparison": comparison,
        "veto_status_table": {"status": "blocked", "blocker": blocker},
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_scalar_additivity_route_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, comparison, {"status": "blocked"}),
        "non_implications": _non_implications(),
    }


def _compare_scalar_additivity(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_probe = filterflow["scalar_additivity_probe"]
    bf_probe = bayesfilter["scalar_additivity_probe"]
    ff_gradients = _gradient_vectors(ff_probe)
    bf_gradients = _gradient_vectors(bf_probe)
    scalar_value_rows = {}
    gradient_rows = {}
    for name in OBJECTIVES:
        scalar_value_rows[name] = {
            "filterflow": ff_probe["scalar_values"][name],
            "bayesfilter": bf_probe["scalar_values"][name],
            "delta": float(bf_probe["scalar_values"][name]) - float(ff_probe["scalar_values"][name]),
            "abs_delta": abs(
                float(bf_probe["scalar_values"][name]) - float(ff_probe["scalar_values"][name])
            ),
        }
        gradient_rows[name] = _gradient_row(ff_gradients[name], bf_gradients[name])

    ff_additivity = _within_side_additivity(ff_probe["scalar_values"], ff_gradients)
    bf_additivity = _within_side_additivity(bf_probe["scalar_values"], bf_gradients)
    total_gradient_consistency = {
        "filterflow": _gradient_row(
            filterflow["total_gradient_diag"],
            ff_gradients["post_update_mean"],
            labels=("recorded_total_gradient", "scalar_post_update_gradient"),
        ),
        "bayesfilter": _gradient_row(
            bayesfilter["total_gradient_diag"],
            bf_gradients["post_update_mean"],
            labels=("recorded_total_gradient", "scalar_post_update_gradient"),
        ),
    }
    return {
        "status": "compared",
        "scalar_value_rows": scalar_value_rows,
        "gradient_rows": gradient_rows,
        "filterflow_within_side_additivity": ff_additivity,
        "bayesfilter_within_side_additivity": bf_additivity,
        "total_gradient_consistency": total_gradient_consistency,
        "max_abs_scalar_value_delta": max(row["abs_delta"] for row in scalar_value_rows.values()),
        "max_abs_gradient_delta": max(row["max_abs_delta"] for row in gradient_rows.values()),
        "first_gradient_delta_over_tolerance": _first_gradient_delta(gradient_rows),
        "interpretation": _interpret_scalar_additivity(gradient_rows, ff_additivity, bf_additivity),
    }


def _gradient_vectors(probe: dict[str, Any]) -> dict[str, list[float]]:
    tensors = probe.get("gradient_diag_tensors", probe.get("gradient_tensors", {}))
    return {name: [float(value) for value in tensors[name]] for name in OBJECTIVES}


def _gradient_row(
    filterflow_values: list[float],
    bayesfilter_values: list[float],
    labels: tuple[str, str] = ("filterflow", "bayesfilter"),
) -> dict[str, Any]:
    deltas = [
        float(bf) - float(ff)
        for ff, bf in zip(filterflow_values, bayesfilter_values, strict=True)
    ]
    return {
        labels[0]: [float(value) for value in filterflow_values],
        labels[1]: [float(value) for value in bayesfilter_values],
        "delta": deltas,
        "max_abs_delta": max(abs(value) for value in deltas),
        "finite": all(tf.math.is_finite(tf.constant(filterflow_values + bayesfilter_values, tf.float64)).numpy()),
    }


def _within_side_additivity(
    values: dict[str, float],
    gradients: dict[str, list[float]],
) -> dict[str, Any]:
    post = gradients["post_update_mean"]
    sum_route = gradients["sum_pre_current_plus_increment_mean"]
    pre = gradients["pre_current_mean"]
    increment = gradients["increment_mean"]
    value_post_minus_sum = float(values["post_update_mean"]) - float(
        values["sum_pre_current_plus_increment_mean"]
    )
    value_sum_minus_components = float(values["sum_pre_current_plus_increment_mean"]) - (
        float(values["pre_current_mean"]) + float(values["increment_mean"])
    )
    post_minus_sum = _vector_sub(post, sum_route)
    sum_minus_components = _vector_sub(sum_route, _vector_add(pre, increment))
    post_minus_components = _vector_sub(post, _vector_add(pre, increment))
    return {
        "value_post_minus_sum": value_post_minus_sum,
        "value_sum_minus_components": value_sum_minus_components,
        "post_gradient_minus_sum_gradient": post_minus_sum,
        "sum_gradient_minus_component_gradients": sum_minus_components,
        "post_gradient_minus_component_gradients": post_minus_components,
        "max_abs_gradient_additivity_gap": max(
            _max_abs(post_minus_sum),
            _max_abs(sum_minus_components),
            _max_abs(post_minus_components),
        ),
        "additivity_pass": (
            abs(value_post_minus_sum) <= VALUE_TOLERANCE
            and abs(value_sum_minus_components) <= VALUE_TOLERANCE
            and _max_abs(post_minus_sum) <= GRADIENT_TOLERANCE
            and _max_abs(sum_minus_components) <= GRADIENT_TOLERANCE
            and _max_abs(post_minus_components) <= GRADIENT_TOLERANCE
        ),
    }


def _vector_add(left: list[float], right: list[float]) -> list[float]:
    return [float(lhs) + float(rhs) for lhs, rhs in zip(left, right, strict=True)]


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    return [float(lhs) - float(rhs) for lhs, rhs in zip(left, right, strict=True)]


def _max_abs(values: list[float]) -> float:
    return max(abs(float(value)) for value in values)


def _first_gradient_delta(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for name, row in rows.items():
        if not row["finite"] or row["max_abs_delta"] > GRADIENT_TOLERANCE:
            return {"status": "delta", "field": name, "row": row, "tolerance": GRADIENT_TOLERANCE}
    return {"status": "no_delta", "tolerance": GRADIENT_TOLERANCE}


def _interpret_scalar_additivity(
    gradient_rows: dict[str, dict[str, Any]],
    filterflow_additivity: dict[str, Any],
    bayesfilter_additivity: dict[str, Any],
) -> str:
    post_delta = gradient_rows["post_update_mean"]["max_abs_delta"]
    sum_delta = gradient_rows["sum_pre_current_plus_increment_mean"]["max_abs_delta"]
    pre_delta = gradient_rows["pre_current_mean"]["max_abs_delta"]
    increment_delta = gradient_rows["increment_mean"]["max_abs_delta"]
    component_delta = max(pre_delta, increment_delta)
    if post_delta <= GRADIENT_TOLERANCE and sum_delta <= GRADIENT_TOLERANCE:
        return "direct_scalar_gradients_match"
    if (
        component_delta <= GRADIENT_TOLERANCE
        and sum_delta <= GRADIENT_TOLERANCE
        and post_delta > GRADIENT_TOLERANCE
    ):
        return "post_update_direct_route_differs_from_recomputed_sum_route"
    if (
        component_delta <= GRADIENT_TOLERANCE
        and sum_delta > GRADIENT_TOLERANCE
        and post_delta > GRADIENT_TOLERANCE
    ):
        return "summed_direct_scalar_route_differs_despite_component_gradient_match"
    if not filterflow_additivity["additivity_pass"] or not bayesfilter_additivity["additivity_pass"]:
        return "within_side_direct_scalar_additivity_failure"
    return "direct_scalar_component_gradient_difference"


def _veto_status(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparison: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    gradient_rows = comparison.get("gradient_rows", {})
    all_finite = all(row.get("finite", False) for row in gradient_rows.values())
    return {
        "all_vetoes_clear": (
            not comparator_drift
            and not any(bool(value) for value in path_boundary.values())
            and filterflow["resampling_flag"] == bayesfilter["resampling_flag"]
            and comparison["max_abs_scalar_value_delta"] <= VALUE_TOLERANCE
            and all_finite
        ),
        "comparator_drift": comparator_drift,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "resampling_flags_match": filterflow["resampling_flag"] == bayesfilter["resampling_flag"],
        "scalar_value_gate_pass": comparison["max_abs_scalar_value_delta"] <= VALUE_TOLERANCE,
        "gradient_rows_finite": all_finite,
        "cpu_only_parent": PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1",
    }


def _decision(comparison: dict[str, Any], veto_status: dict[str, Any]) -> str:
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_scalar_additivity_blocked"
    if not veto_status["all_vetoes_clear"]:
        return "filterflow_float64_row_173_scalar_additivity_vetoed"
    interpretation = comparison["interpretation"]
    if interpretation == "direct_scalar_gradients_match":
        return "filterflow_float64_row_173_scalar_additivity_direct_gradients_match"
    if interpretation == "post_update_direct_route_differs_from_recomputed_sum_route":
        return "filterflow_float64_row_173_scalar_additivity_post_update_route_residual"
    if interpretation == "summed_direct_scalar_route_differs_despite_component_gradient_match":
        return "filterflow_float64_row_173_scalar_additivity_sum_route_residual"
    if interpretation == "within_side_direct_scalar_additivity_failure":
        return "filterflow_float64_row_173_scalar_additivity_internal_additivity_failure"
    return "filterflow_float64_row_173_scalar_additivity_component_gradient_difference"


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None or side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "resampling_flag": side["resampling_flag"],
        "scalar_additivity_probe": side.get("scalar_additivity_probe"),
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _model_contract() -> dict[str, Any]:
    return {
        "mesh_index": vjp.MESH_INDEX,
        "target_time_index": TARGET_TIME_INDEX,
        "theta": vjp.THETA,
        "num_particles": vjp.NUM_PARTICLES,
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "dtype": "float64",
        "comparison_scope": "direct scalar gradients only; no correctness claim",
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Do direct scalar gradients for post_update, recomputed pre+increment, "
            "pre_current, and increment agree between BayesFilter and local "
            "float64 FilterFlow?"
        ),
        "baseline": "local canonical executable float64 FilterFlow reference",
        "primary_criterion": "fieldwise direct scalar-gradient delta",
        "veto_diagnostics": [
            "comparator drift",
            "resampling flag mismatch",
            "scalar value mismatch",
            "non-finite gradients",
            "CPU-only policy failure",
            "path boundary contamination",
        ],
        "explanatory_diagnostics": [
            "within-side direct scalar additivity",
            "post_update direct-gradient consistency with recorded total gradient",
        ],
        "not_concluded": _non_implications(),
        "artifact": str(JSON_PATH.relative_to(REPO_ROOT)),
    }


def _decision_table(
    decision: str,
    comparison: dict[str, Any],
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    first_delta = comparison.get("first_gradient_delta_over_tolerance", {})
    if decision.endswith("direct_gradients_match"):
        primary = "all direct scalar-gradient rows within tolerance"
        next_action = "return to intermediate VJP-row interpretation; scalar path is not the source"
    elif decision.endswith("post_update_route_residual"):
        primary = "component direct gradients match, but stored post_update scalar gradient differs"
        next_action = "trace state.log_likelihoods assignment/carry route inside the post-update state"
    elif decision.endswith("sum_route_residual"):
        primary = "components match but recomputed sum route differs"
        next_action = "inspect recomputed-sum graph construction and tape persistence"
    elif decision.endswith("component_gradient_difference"):
        primary = f"first component direct-gradient delta: {first_delta}"
        next_action = "debug the first direct scalar component over tolerance"
    else:
        primary = str(first_delta or comparison.get("blocker", decision))
        next_action = "repair veto or blocker before interpretation"
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": str(veto_status),
            "main_uncertainty": "single row/time diagnostic; no global gradient claim",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation or production readiness",
        }
    ]


def _non_implications() -> list[str]:
    return [
        "No correctness claim is made for either implementation.",
        "No global smoothness-gradient agreement is concluded.",
        "No analytic Kalman-gradient agreement is concluded.",
        "No production readiness or public API readiness is concluded.",
        "No posterior correctness or nonlinear SSM validity is concluded.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "filterflow_probe",
        "bayesfilter_probe",
        "comparison",
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
    validate_filterflow_reference_status(payload["filterflow_status"], marker_path=FILTERFLOW_MARKER_PATH)
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("pre-import CUDA_VISIBLE_DEVICES was not -1")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    allowed = {
        "filterflow_float64_row_173_scalar_additivity_filterflow_blocker",
        "filterflow_float64_row_173_scalar_additivity_bayesfilter_blocker",
        "filterflow_float64_row_173_scalar_additivity_blocked",
        "filterflow_float64_row_173_scalar_additivity_vetoed",
        "filterflow_float64_row_173_scalar_additivity_direct_gradients_match",
        "filterflow_float64_row_173_scalar_additivity_post_update_route_residual",
        "filterflow_float64_row_173_scalar_additivity_sum_route_residual",
        "filterflow_float64_row_173_scalar_additivity_internal_additivity_failure",
        "filterflow_float64_row_173_scalar_additivity_component_gradient_difference",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Scalar-Additivity Route Probe",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Decision Table",
        "",
        "| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| {decision} | {primary_criterion_status} | {veto_diagnostic_status} | "
            "{main_uncertainty} | {next_justified_action} | {not_concluded} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## FilterFlow Probe",
            "",
            _json_block(payload["filterflow_probe"]),
            "",
            "## BayesFilter Probe",
            "",
            _json_block(payload["bayesfilter_probe"]),
            "",
            "## Non-Implications",
            "",
            *[f"- {item}" for item in payload["non_implications"]],
            "",
        ]
    )
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = dict(payload)
    clone.pop("run_manifest", None)
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


if __name__ == "__main__":
    raise SystemExit(main())
