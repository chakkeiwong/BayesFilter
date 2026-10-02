"""Classify row-173 target-time transport-upstream clipping hypotheses."""

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
    "bayesfilter-dpf-filterflow-float64-row-173-transport-upstream-clipping-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-transport-upstream-clipping-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_transport_upstream_clipping_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-transport-upstream-clipping-2026-06-04.md"
)
FILTERFLOW_MARKER_PATH = vjp.FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
TAG = "row-173-transport-upstream-clipping"
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
            "filterflow_float64_row_173_transport_upstream_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            None,
        )
    bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_transport_upstream_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            reference_status,
            initial_fingerprint,
            filterflow,
            bayesfilter,
        )

    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    comparison = _compare_transport_clipping(filterflow, bayesfilter)
    scalar_comparison = _compare_scalar_additivity(filterflow, bayesfilter)
    veto_status = _veto_status(
        filterflow,
        bayesfilter,
        comparison,
        scalar_comparison,
        comparator_drift,
    )
    classification = _classify(comparison, scalar_comparison, veto_status)
    decision = _decision(classification, veto_status)
    return {
        "decision": decision,
        "hypothesis_classification": classification["classification"],
        "hypothesis_reason": classification["reason"],
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_target_transport_upstream_clipping_difference_audit",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "transport_clipping_comparison": comparison,
        "scalar_additivity_comparison": scalar_comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_transport_upstream_clipping_probe_tf"
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
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
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
    comparison = {"status": "blocked", "blocker": blocker}
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
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
        "transport_clipping_comparison": comparison,
        "scalar_additivity_comparison": comparison,
        "veto_status_table": {"status": "blocked", "blocker": blocker},
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_transport_upstream_clipping_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(
            decision,
            {"classification": "blocked_or_vetoed", "reason": blocker},
            {"status": "blocked"},
        ),
        "non_implications": _non_implications(),
    }


def _compare_transport_clipping(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_probe = filterflow["transport_clipping_probe"]
    bf_probe = bayesfilter["transport_clipping_probe"]
    rows = {}
    for name in OBJECTIVES:
        rows[name] = {
            "raw_upstream": _tensor_row(
                ff_probe["raw_upstream_tensors"][name],
                bf_probe["raw_upstream_tensors"][name],
            ),
            "clipped_upstream": _tensor_row(
                ff_probe["clipped_upstream_tensors"][name],
                bf_probe["clipped_upstream_tensors"][name],
            ),
            "clip_mask": _tensor_row(
                ff_probe["clip_mask_tensors"][name],
                bf_probe["clip_mask_tensors"][name],
            ),
            "target_time_vjp": _vector_row(
                _vjp_vector(ff_probe, name, manual=False),
                _vjp_vector(bf_probe, name, manual=False),
            ),
            "manual_clipped_target_time_vjp": _vector_row(
                _vjp_vector(ff_probe, name, manual=True),
                _vjp_vector(bf_probe, name, manual=True),
            ),
        }
        rows[name]["clip_count_filterflow"] = _sum_nested(ff_probe["clip_mask_tensors"][name])
        rows[name]["clip_count_bayesfilter"] = _sum_nested(bf_probe["clip_mask_tensors"][name])

    within_filterflow = _within_side_transport(ff_probe)
    within_bayesfilter = _within_side_transport(bf_probe)
    max_raw_delta = max(row["raw_upstream"]["max_abs_delta"] for row in rows.values())
    max_clipped_delta = max(row["clipped_upstream"]["max_abs_delta"] for row in rows.values())
    max_mask_delta = max(row["clip_mask"]["max_abs_delta"] for row in rows.values())
    max_target_vjp_delta = max(row["target_time_vjp"]["max_abs_delta"] for row in rows.values())
    return {
        "status": "compared",
        "rows": rows,
        "within_filterflow": within_filterflow,
        "within_bayesfilter": within_bayesfilter,
        "max_raw_upstream_delta": max_raw_delta,
        "max_clipped_upstream_delta": max_clipped_delta,
        "max_clip_mask_delta": max_mask_delta,
        "max_target_time_vjp_delta": max_target_vjp_delta,
        "first_delta_over_tolerance": _first_transport_delta(rows),
        "interpretation": _interpret_transport(
            rows,
            within_filterflow,
            within_bayesfilter,
            max_raw_delta,
            max_clipped_delta,
            max_mask_delta,
            max_target_vjp_delta,
        ),
    }


def _vjp_vector(probe: dict[str, Any], name: str, *, manual: bool) -> list[float]:
    if manual:
        if "manual_clipped_target_time_vjp_diag_tensors" in probe:
            return [float(value) for value in probe["manual_clipped_target_time_vjp_diag_tensors"][name]]
        return [float(value) for value in probe["manual_clipped_target_time_vjp_tensors"][name]]
    if "target_time_vjp_diag_tensors" in probe:
        return [float(value) for value in probe["target_time_vjp_diag_tensors"][name]]
    return [float(value) for value in probe["target_time_vjp_tensors"][name]]


def _within_side_transport(probe: dict[str, Any]) -> dict[str, Any]:
    raw = {name: probe["raw_upstream_tensors"][name] for name in OBJECTIVES}
    clipped = {name: probe["clipped_upstream_tensors"][name] for name in OBJECTIVES}
    vjps = {name: _vjp_vector(probe, name, manual=False) for name in OBJECTIVES}
    manual_vjps = {name: _vjp_vector(probe, name, manual=True) for name in OBJECTIVES}
    return {
        "raw_upstream_additivity_gap": _nested_sub(
            raw["post_update_mean"],
            _nested_add(raw["pre_current_mean"], raw["increment_mean"]),
        ),
        "clipped_upstream_additivity_gap": _nested_sub(
            clipped["post_update_mean"],
            _nested_add(clipped["pre_current_mean"], clipped["increment_mean"]),
        ),
        "target_time_vjp_additivity_gap": _vector_sub(
            vjps["post_update_mean"],
            _vector_add(vjps["pre_current_mean"], vjps["increment_mean"]),
        ),
        "manual_clipped_target_time_vjp_additivity_gap": _vector_sub(
            manual_vjps["post_update_mean"],
            _vector_add(manual_vjps["pre_current_mean"], manual_vjps["increment_mean"]),
        ),
        "raw_upstream_additive": _max_abs_nested(
            _nested_sub(
                raw["post_update_mean"],
                _nested_add(raw["pre_current_mean"], raw["increment_mean"]),
            )
        )
        <= GRADIENT_TOLERANCE,
        "clipped_upstream_additive": _max_abs_nested(
            _nested_sub(
                clipped["post_update_mean"],
                _nested_add(clipped["pre_current_mean"], clipped["increment_mean"]),
            )
        )
        <= GRADIENT_TOLERANCE,
        "target_time_vjp_additive": _max_abs(
            _vector_sub(vjps["post_update_mean"], _vector_add(vjps["pre_current_mean"], vjps["increment_mean"]))
        )
        <= GRADIENT_TOLERANCE,
    }


def _compare_scalar_additivity(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_probe = filterflow["scalar_additivity_probe"]
    bf_probe = bayesfilter["scalar_additivity_probe"]
    ff_gradients = _scalar_gradient_vectors(ff_probe)
    bf_gradients = _scalar_gradient_vectors(bf_probe)
    rows = {
        name: _vector_row(ff_gradients[name], bf_gradients[name])
        for name in OBJECTIVES
    }
    scalar_value_delta = max(
        abs(float(bf_probe["scalar_values"][name]) - float(ff_probe["scalar_values"][name]))
        for name in OBJECTIVES
    )
    return {
        "status": "compared",
        "rows": rows,
        "max_abs_scalar_value_delta": scalar_value_delta,
        "max_abs_gradient_delta": max(row["max_abs_delta"] for row in rows.values()),
        "component_gradients_match": (
            rows["pre_current_mean"]["max_abs_delta"] <= GRADIENT_TOLERANCE
            and rows["increment_mean"]["max_abs_delta"] <= GRADIENT_TOLERANCE
        ),
        "post_gradient_matches": rows["post_update_mean"]["max_abs_delta"] <= GRADIENT_TOLERANCE,
    }


def _scalar_gradient_vectors(probe: dict[str, Any]) -> dict[str, list[float]]:
    tensors = probe.get("gradient_diag_tensors", probe.get("gradient_tensors", {}))
    return {name: [float(value) for value in tensors[name]] for name in OBJECTIVES}


def _tensor_row(filterflow_tensor: Any, bayesfilter_tensor: Any) -> dict[str, Any]:
    return {
        "filterflow_summary": _summary_nested(filterflow_tensor),
        "bayesfilter_summary": _summary_nested(bayesfilter_tensor),
        "max_abs_delta": _max_abs_nested_delta(bayesfilter_tensor, filterflow_tensor),
        "sum_delta": _sum_nested_delta(bayesfilter_tensor, filterflow_tensor),
        "finite": _finite_nested(filterflow_tensor) and _finite_nested(bayesfilter_tensor),
    }


def _vector_row(filterflow_values: list[float], bayesfilter_values: list[float]) -> dict[str, Any]:
    delta = [float(bf) - float(ff) for ff, bf in zip(filterflow_values, bayesfilter_values, strict=True)]
    return {
        "filterflow": [float(value) for value in filterflow_values],
        "bayesfilter": [float(value) for value in bayesfilter_values],
        "delta": delta,
        "max_abs_delta": _max_abs(delta),
        "finite": all(tf.math.is_finite(tf.constant(filterflow_values + bayesfilter_values, tf.float64)).numpy()),
    }


def _classify(
    comparison: dict[str, Any],
    scalar_comparison: dict[str, Any],
    veto_status: dict[str, Any],
) -> dict[str, str]:
    if not veto_status.get("all_vetoes_clear", False):
        return {"classification": "blocked_or_vetoed", "reason": str(veto_status)}
    interp = comparison["interpretation"]
    if interp == "expected_clipping_nonlinearity":
        return {
            "classification": "h1_expected_clipping_nonlinearity",
            "reason": "raw upstreams are additive but clipped upstreams/VJPs are non-additive",
        }
    if interp == "cross_impl_clip_mask_or_upstream_mismatch":
        return {
            "classification": "h2_cross_impl_clip_mask_or_upstream_mismatch",
            "reason": "raw/clipped upstream or clip-mask deltas exceed tolerance",
        }
    if interp == "transport_custom_vjp_rule_mismatch":
        return {
            "classification": "h3_transport_custom_vjp_rule_mismatch",
            "reason": "upstreams and masks match but target-time transport VJP differs",
        }
    if interp == "residual_outside_target_transport_node":
        return {
            "classification": "h4_residual_outside_target_transport_node",
            "reason": "target-time transport upstreams, masks, and VJPs match within tolerance",
        }
    return {
        "classification": "blocked_or_vetoed",
        "reason": f"unclassified interpretation {interp}; scalar={scalar_comparison}",
    }


def _interpret_transport(
    rows: dict[str, Any],
    within_filterflow: dict[str, Any],
    within_bayesfilter: dict[str, Any],
    max_raw_delta: float,
    max_clipped_delta: float,
    max_mask_delta: float,
    max_target_vjp_delta: float,
) -> str:
    raw_additive = within_filterflow["raw_upstream_additive"] and within_bayesfilter["raw_upstream_additive"]
    clipped_nonadditive = (
        not within_filterflow["clipped_upstream_additive"]
        or not within_bayesfilter["clipped_upstream_additive"]
        or not within_filterflow["target_time_vjp_additive"]
        or not within_bayesfilter["target_time_vjp_additive"]
    )
    if (
        max_raw_delta > GRADIENT_TOLERANCE
        or max_clipped_delta > GRADIENT_TOLERANCE
        or max_mask_delta > 0.0
    ):
        return "cross_impl_clip_mask_or_upstream_mismatch"
    if raw_additive and clipped_nonadditive:
        return "expected_clipping_nonlinearity"
    if max_target_vjp_delta > GRADIENT_TOLERANCE:
        return "transport_custom_vjp_rule_mismatch"
    del rows
    return "residual_outside_target_transport_node"


def _first_transport_delta(rows: dict[str, Any]) -> dict[str, Any]:
    for objective, row in rows.items():
        for field in (
            "raw_upstream",
            "clipped_upstream",
            "clip_mask",
            "target_time_vjp",
            "manual_clipped_target_time_vjp",
        ):
            tolerance = 0.0 if field == "clip_mask" else GRADIENT_TOLERANCE
            if not row[field]["finite"] or row[field]["max_abs_delta"] > tolerance:
                return {
                    "status": "delta",
                    "objective": objective,
                    "field": field,
                    "row": row[field],
                    "tolerance": tolerance,
                }
    return {"status": "no_delta", "tolerance": GRADIENT_TOLERANCE}


def _veto_status(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparison: dict[str, Any],
    scalar_comparison: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    path_boundary = continuation._path_boundary_manifest()
    finite_transport = all(
        field["finite"]
        for row in comparison.get("rows", {}).values()
        for field in (
            row["raw_upstream"],
            row["clipped_upstream"],
            row["clip_mask"],
            row["target_time_vjp"],
            row["manual_clipped_target_time_vjp"],
        )
    )
    return {
        "all_vetoes_clear": (
            not comparator_drift
            and not any(bool(value) for value in path_boundary.values())
            and filterflow["resampling_flag"] == bayesfilter["resampling_flag"]
            and scalar_comparison["max_abs_scalar_value_delta"] <= VALUE_TOLERANCE
            and finite_transport
            and PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1"
        ),
        "comparator_drift": comparator_drift,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "resampling_flags_match": filterflow["resampling_flag"] == bayesfilter["resampling_flag"],
        "scalar_value_gate_pass": scalar_comparison["max_abs_scalar_value_delta"] <= VALUE_TOLERANCE,
        "transport_rows_finite": finite_transport,
        "cpu_only_parent": PRE_IMPORT_CUDA_VISIBLE_DEVICES == "-1",
    }


def _decision(classification: dict[str, str], veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_transport_upstream_blocked_or_vetoed"
    mapping = {
        "h1_expected_clipping_nonlinearity": (
            "filterflow_float64_row_173_transport_upstream_h1_expected_clipping_nonlinearity"
        ),
        "h2_cross_impl_clip_mask_or_upstream_mismatch": (
            "filterflow_float64_row_173_transport_upstream_h2_cross_impl_clip_mask_or_upstream_mismatch"
        ),
        "h3_transport_custom_vjp_rule_mismatch": (
            "filterflow_float64_row_173_transport_upstream_h3_transport_custom_vjp_rule_mismatch"
        ),
        "h4_residual_outside_target_transport_node": (
            "filterflow_float64_row_173_transport_upstream_h4_residual_outside_target_transport_node"
        ),
    }
    return mapping.get(classification["classification"], "filterflow_float64_row_173_transport_upstream_blocked_or_vetoed")


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
        "transport_clipping_probe": side.get("transport_clipping_probe"),
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
        "comparison_scope": "target-time transport upstream clipping only",
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": "Classify row-173 target-time transport upstream clipping hypothesis.",
        "baseline": "local canonical executable float64 FilterFlow reference",
        "primary_criterion": "raw/clipped upstream, clip-mask, and target-time transport VJP deltas",
        "veto_diagnostics": [
            "comparator drift",
            "resampling flag mismatch",
            "scalar value mismatch",
            "non-finite upstream or VJP tensors",
            "CPU-only policy failure",
            "path boundary contamination",
        ],
        "explanatory_diagnostics": [
            "within-side raw and clipped additivity",
            "clip counts",
            "target-time VJP additivity",
        ],
        "artifact": str(JSON_PATH.relative_to(REPO_ROOT)),
        "not_concluded": _non_implications(),
    }


def _decision_table(
    decision: str,
    classification: dict[str, str],
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "decision": decision,
            "primary_criterion_status": classification["classification"],
            "veto_diagnostic_status": str(veto_status),
            "main_uncertainty": "single target-time transport node; historical transport nodes may still matter",
            "next_justified_action": _next_action(classification["classification"]),
            "not_concluded": "correctness, posterior correctness, production readiness, global gradient agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h1_expected_clipping_nonlinearity":
        return "compare clipped upstream sensitivity near threshold and decide whether row residual is expected under nonlinear custom VJP"
    if classification == "h2_cross_impl_clip_mask_or_upstream_mismatch":
        return "inspect the first raw/clipped upstream or mask delta entry"
    if classification == "h3_transport_custom_vjp_rule_mismatch":
        return "compare target-time transport VJP rule implementation field-by-field"
    if classification == "h4_residual_outside_target_transport_node":
        return "extend probe to historical transport nodes and carryover topology"
    return "repair blocker or veto"


def _non_implications() -> list[str]:
    return [
        "No correctness claim is made for either implementation.",
        "No analytic gradient correctness is concluded.",
        "No global gradient agreement is concluded.",
        "No full mesh or surface agreement is concluded.",
        "No posterior correctness is concluded.",
        "No production readiness or public API readiness is concluded.",
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "hypothesis_classification",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "transport_clipping_comparison",
        "scalar_additivity_comparison",
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
        "filterflow_float64_row_173_transport_upstream_filterflow_blocker",
        "filterflow_float64_row_173_transport_upstream_bayesfilter_blocker",
        "filterflow_float64_row_173_transport_upstream_blocked_or_vetoed",
        "filterflow_float64_row_173_transport_upstream_h1_expected_clipping_nonlinearity",
        "filterflow_float64_row_173_transport_upstream_h2_cross_impl_clip_mask_or_upstream_mismatch",
        "filterflow_float64_row_173_transport_upstream_h3_transport_custom_vjp_rule_mismatch",
        "filterflow_float64_row_173_transport_upstream_h4_residual_outside_target_transport_node",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Transport-Upstream Clipping Probe",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Hypothesis Classification",
        "",
        f"`{payload['hypothesis_classification']}`",
        "",
        payload.get("hypothesis_reason", ""),
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
            "## Transport Clipping Comparison",
            "",
            _json_block(payload["transport_clipping_comparison"]),
            "",
            "## Scalar Additivity Comparison",
            "",
            _json_block(payload["scalar_additivity_comparison"]),
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


def _summary_nested(value: Any) -> dict[str, Any]:
    flat = _flatten(value)
    return {
        "count": len(flat),
        "finite": all(tf.math.is_finite(tf.constant(flat, tf.float64)).numpy()),
        "max_abs": max(abs(item) for item in flat) if flat else 0.0,
        "sum": sum(flat),
    }


def _finite_nested(value: Any) -> bool:
    flat = _flatten(value)
    return bool(all(tf.math.is_finite(tf.constant(flat, tf.float64)).numpy()))


def _max_abs_nested_delta(left: Any, right: Any) -> float:
    return _max_abs(_nested_sub(left, right))


def _sum_nested_delta(left: Any, right: Any) -> float:
    return _sum_nested(_nested_sub(left, right))


def _sum_nested(value: Any) -> float:
    return sum(_flatten(value))


def _max_abs_nested(value: Any) -> float:
    return _max_abs(_flatten(value))


def _nested_add(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_nested_add(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)]
    return float(left) + float(right)


def _nested_sub(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_nested_sub(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)]
    return float(left) - float(right)


def _flatten(value: Any) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [float(value)]


def _vector_add(left: list[float], right: list[float]) -> list[float]:
    return [float(lhs) + float(rhs) for lhs, rhs in zip(left, right, strict=True)]


def _vector_sub(left: list[float], right: list[float]) -> list[float]:
    return [float(lhs) - float(rhs) for lhs, rhs in zip(left, right, strict=True)]


def _max_abs(values: Any) -> float:
    flat = _flatten(values)
    return max(abs(float(value)) for value in flat) if flat else 0.0


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
