"""Probe the row-173 post-update log-likelihood gradient route."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
import time
from pathlib import Path
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


DTYPE = tf.float64
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-post-update-route-hypotheses-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-post-update-route-hypotheses-result-2026-06-04.md"
)
JSON_PATH = (
    OUTPUT_DIR
    / "dpf_filterflow_float64_row_173_post_update_route_hypotheses_2026-06-04.json"
)
REPORT_PATH = (
    REPORT_DIR
    / "dpf-filterflow-float64-row-173-post-update-route-hypotheses-2026-06-04.md"
)
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER
ADJACENT_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_adjacent_boundary_2026-06-04.json"
)

TAG = "row-173-post-update-route"
TARGET_TIME_INDEX = 93
VALUE_TOLERANCE = 5e-8
GRADIENT_TOLERANCE = 2e-4
POST_UPDATE_FIELDS = (
    "post_update_log_likelihoods",
    "pre_current_log_likelihoods",
    "increment",
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
    adjacent = _adjacent_baseline()
    current_comparator_fingerprint = _comparator_fingerprint(reference_status, initial_fingerprint)
    baseline_comparator_comparison = _baseline_comparator_comparison(
        adjacent,
        reference_status,
        initial_fingerprint,
        current_comparator_fingerprint,
    )
    if not baseline_comparator_comparison["pass"]:
        return _blocked_payload(
            "filterflow_float64_row_173_post_update_baseline_comparator_drift",
            "baseline-vs-current comparator identity mismatch",
            initial_fingerprint,
            reference_status,
            adjacent,
            None,
            None,
            baseline_comparator_comparison,
            current_comparator_fingerprint,
        )
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
            "filterflow_float64_row_173_post_update_filterflow_blocker",
            filterflow.get("blocker", "unknown FilterFlow blocker"),
            initial_fingerprint,
            reference_status,
            adjacent,
            filterflow,
            None,
            baseline_comparator_comparison,
            current_comparator_fingerprint,
        )
    bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_post_update_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            initial_fingerprint,
            reference_status,
            adjacent,
            filterflow,
            bayesfilter,
            baseline_comparator_comparison,
            current_comparator_fingerprint,
        )
    boundary_modes = vjp._bayesfilter_boundary_modes(filterflow, config)
    boundary_summary = vjp._compare_boundary_modes(filterflow, boundary_modes)
    same_tape_summary = vjp._compare_adjoint_decomposition(filterflow, bayesfilter)
    route = _post_update_route(filterflow, bayesfilter, adjacent)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    veto_status = _veto_status(route, filterflow, bayesfilter, comparator_drift)
    classification = _classify_hypothesis(route, veto_status, boundary_summary, same_tape_summary)
    decision = _decision(classification, veto_status)
    payload = {
        "decision": decision,
        "hypothesis_classification": classification["classification"],
        "hypothesis_reason": classification["reason"],
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_post_update_log_likelihood_route_hypothesis_probe",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "comparator_fingerprint": current_comparator_fingerprint,
        "baseline_comparator_comparison": baseline_comparator_comparison,
        "input_manifest": _input_manifest(filterflow, adjacent, initial_fingerprint),
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "post_update_route": route,
        "scalar_resampling_gates": _scalar_resampling_gates(filterflow, bayesfilter),
        "veto_status_table": veto_status,
        "boundary_mode_summary_table": boundary_summary,
        "same_tape_state_adjoint_summary_table": same_tape_summary,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_post_update_route_probe_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "data_seed": vjp.DATA_SEED,
            "filter_seed": vjp.FILTER_SEED,
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
            "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
            "gpu_visibility_note": (
                "CPU-only was intentionally forced before TensorFlow import; "
                "visible GPU list is informational only, not machine setup evidence."
            ),
        },
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
    }
    return payload


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    reference_status: dict[str, Any],
    adjacent: dict[str, Any],
    filterflow: dict[str, Any] | None,
    bayesfilter: dict[str, Any] | None,
    baseline_comparator_comparison: dict[str, Any] | None = None,
    current_comparator_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_fingerprint = continuation._filterflow_fingerprint()
    classification = {
        "classification": "blocked",
        "reason": blocker,
        "promoted": False,
        "next_smallest_probe": "repair execution blocker before interpreting route hypotheses",
    }
    veto_status = {
        "status": "blocked",
        "blocker": blocker,
        "all_vetoes_clear": False,
    }
    return {
        "decision": decision,
        "hypothesis_classification": classification["classification"],
        "hypothesis_reason": classification["reason"],
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint),
        "comparator_fingerprint": current_comparator_fingerprint
        or _comparator_fingerprint(reference_status, initial_fingerprint),
        "baseline_comparator_comparison": baseline_comparator_comparison
        or {"status": "not_available_before_blocker", "pass": False},
        "input_manifest": {"adjacent_baseline": adjacent, "status": "blocked_before_full_inputs"},
        "model_contract": _model_contract(),
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "post_update_route": {"status": "blocked", "blocker": blocker},
        "scalar_resampling_gates": {"status": "blocked", "blocker": blocker},
        "veto_status_table": veto_status,
        "boundary_mode_summary_table": {"status": "not_run"},
        "same_tape_state_adjoint_summary_table": {"status": "not_run"},
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_post_update_route_probe_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
    }


def _adjacent_baseline() -> dict[str, Any]:
    if not ADJACENT_JSON.exists():
        return {
            "status": "missing",
            "path": str(ADJACENT_JSON.relative_to(REPO_ROOT)),
            "observed_residual": [5.302734403676368, -0.1337765252068337],
        }
    payload = load_json(ADJACENT_JSON)
    best = payload["comparison"]["best_parameter_path_residual_match"]
    return {
        "status": "loaded",
        "path": str(ADJACENT_JSON.relative_to(REPO_ROOT)),
        "decision": payload["decision"],
        "interpretation": payload["comparison"].get("interpretation"),
        "field": best["field"],
        "observed_residual": payload["comparison"]["observed_total_gradient_delta"],
        "post_update_delta": best["delta"],
        "filterflow_status": payload.get("filterflow_status"),
        "filterflow_fingerprint_initial": payload.get("filterflow_fingerprint_initial"),
        "filterflow_fingerprint_final": payload.get("filterflow_fingerprint_final"),
        "comparator_fingerprint": payload.get("comparator_fingerprint"),
        "digest": stable_digest(payload),
    }


def _baseline_comparator_comparison(
    adjacent: dict[str, Any],
    current_status: dict[str, Any],
    current_fingerprint: dict[str, Any],
    current_comparator_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    if adjacent.get("status") != "loaded":
        return {
            "status": "blocked",
            "pass": False,
            "blocker": "adjacent-boundary baseline artifact not loaded",
        }
    baseline_status = adjacent.get("filterflow_status") or {}
    baseline_fingerprint = adjacent.get("filterflow_fingerprint_initial") or {}
    baseline_comparator = adjacent.get("comparator_fingerprint") or {}
    rows = {
        "checkout_path": _compare_values(
            baseline_status.get("path") or baseline_fingerprint.get("path"),
            current_status.get("path") or current_fingerprint.get("path"),
        ),
        "commit_sha": _compare_values(
            baseline_status.get("commit") or baseline_fingerprint.get("head_commit"),
            current_status.get("commit") or current_fingerprint.get("head_commit"),
        ),
        "dirty_status_summary": _compare_values(
            baseline_status.get("status_short") or baseline_fingerprint.get("status_short"),
            current_status.get("status_short") or current_fingerprint.get("status_short"),
        ),
        "status_branch": _compare_values(
            baseline_status.get("status_branch") or baseline_fingerprint.get("status_branch"),
            current_status.get("status_branch") or current_fingerprint.get("status_branch"),
        ),
        "symbolic_head_or_branch_marker": _compare_values(
            baseline_status.get("branch") or baseline_fingerprint.get("symbolic_head"),
            current_status.get("branch") or current_fingerprint.get("symbolic_head"),
        ),
        "diff_digest": _compare_values(
            baseline_fingerprint.get("diff_digest"),
            current_fingerprint.get("diff_digest"),
        ),
        "package_manifest_digest": _compare_values(
            baseline_fingerprint.get("package_manifest_digest"),
            current_fingerprint.get("package_manifest_digest"),
        ),
        "python_version": _compare_values(
            baseline_fingerprint.get("python_version"),
            current_fingerprint.get("python_version"),
        ),
    }
    baseline_hashes = baseline_comparator.get("entrypoint_hashes")
    current_hashes = current_comparator_fingerprint.get("entrypoint_hashes")
    if baseline_hashes is None:
        entrypoint_hash_comparison = {
            "status": "baseline_hashes_not_recorded",
            "pass": rows["commit_sha"]["pass"]
            and rows["dirty_status_summary"]["pass"]
            and rows["diff_digest"]["pass"],
            "equivalence_basis": (
                "Prior adjacent-boundary artifact predates per-entrypoint hashes. "
                "Matching commit SHA, clean dirty-status summary, and diff digest "
                "are used as the recorded baseline-vs-current entrypoint identity "
                "control for tracked comparator files."
            ),
            "baseline_entrypoint_hashes": None,
            "current_entrypoint_hashes": current_hashes,
        }
    else:
        entrypoint_hash_comparison = {
            "status": "compared",
            "pass": baseline_hashes == current_hashes,
            "baseline_entrypoint_hashes": baseline_hashes,
            "current_entrypoint_hashes": current_hashes,
        }
    pass_rows = all(row["pass"] for row in rows.values())
    overall_pass = pass_rows and bool(entrypoint_hash_comparison["pass"])
    return {
        "status": "compared",
        "pass": overall_pass,
        "rows": rows,
        "entrypoint_hash_or_mtime_comparison": entrypoint_hash_comparison,
        "baseline_artifact": adjacent.get("path"),
        "baseline_artifact_digest": adjacent.get("digest"),
    }


def _compare_values(baseline: Any, current: Any) -> dict[str, Any]:
    return {
        "baseline": baseline,
        "current": current,
        "pass": baseline is not None and current is not None and baseline == current,
    }


def _post_update_route(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    adjacent: dict[str, Any],
) -> dict[str, Any]:
    values = {
        field: {
            "filterflow": filterflow["value_tensors"][field],
            "bayesfilter": bayesfilter["value_tensors"][field],
            "max_abs_value_delta": _max_abs_delta(
                bayesfilter["value_tensors"][field],
                filterflow["value_tensors"][field],
            ),
        }
        for field in POST_UPDATE_FIELDS
    }
    gradient_rows = {
        field: {
            "filterflow": filterflow["parameter_path_adjoint_tensors"][field],
            "bayesfilter": bayesfilter["parameter_path_adjoint_tensors"][field],
            "delta": _vector_delta(
                bayesfilter["parameter_path_adjoint_tensors"][field],
                filterflow["parameter_path_adjoint_tensors"][field],
            ),
            "finite": _finite_nested(filterflow["parameter_path_adjoint_tensors"][field])
            and _finite_nested(bayesfilter["parameter_path_adjoint_tensors"][field]),
        }
        for field in POST_UPDATE_FIELDS
    }
    observed = [float(value) for value in adjacent["observed_residual"]]
    post_update_delta = gradient_rows["post_update_log_likelihoods"]["delta"]
    pre_delta = gradient_rows["pre_current_log_likelihoods"]["delta"]
    inc_delta = gradient_rows["increment"]["delta"]
    component_sum_delta = [pre_delta[i] + inc_delta[i] for i in range(len(observed))]
    post_residual = [post_update_delta[i] - observed[i] for i in range(len(observed))]
    component_residual = [component_sum_delta[i] - observed[i] for i in range(len(observed))]
    additivity = _additivity_gaps(filterflow, bayesfilter)
    return {
        "status": "compared",
        "fields": list(POST_UPDATE_FIELDS),
        "values": values,
        "gradient_rows": gradient_rows,
        "observed_residual": observed,
        "post_update_delta": post_update_delta,
        "pre_current_delta": pre_delta,
        "increment_delta": inc_delta,
        "component_sum_delta": component_sum_delta,
        "post_update_reconstruction_residual": post_residual,
        "component_sum_reconstruction_residual": component_residual,
        "max_abs_post_update_reconstruction_residual": _max_abs_vector(post_residual),
        "max_abs_component_sum_reconstruction_residual": _max_abs_vector(component_residual),
        "within_side_additivity_gaps": additivity,
        "value_additivity_pass": (
            additivity["filterflow"]["max_abs_gap"] <= VALUE_TOLERANCE
            and additivity["bayesfilter"]["max_abs_gap"] <= VALUE_TOLERANCE
            and additivity["cross_impl_gap_delta_max_abs"] <= VALUE_TOLERANCE
        ),
        "component_sum_pass": _max_abs_vector(component_residual) <= GRADIENT_TOLERANCE,
        "post_update_pass": _max_abs_vector(post_residual) <= GRADIENT_TOLERANCE,
    }


def _additivity_gaps(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    gap_values = {}
    for label, side in (("filterflow", filterflow), ("bayesfilter", bayesfilter)):
        post = _flatten(side["value_tensors"]["post_update_log_likelihoods"])
        pre = _flatten(side["value_tensors"]["pre_current_log_likelihoods"])
        inc = _flatten(side["value_tensors"]["increment"])
        gap = [post[i] - pre[i] - inc[i] for i in range(len(post))]
        gap_values[label] = gap
        rows[label] = {
            "gap": gap,
            "max_abs_gap": _max_abs_vector(gap),
        }
    delta = [gap_values["bayesfilter"][i] - gap_values["filterflow"][i] for i in range(len(gap_values["filterflow"]))]
    return {
        **rows,
        "cross_impl_gap_delta": delta,
        "cross_impl_gap_delta_max_abs": _max_abs_vector(delta),
    }


def _veto_status(
    route: dict[str, Any],
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    comparator_drift: bool,
) -> dict[str, Any]:
    scalar_delta = abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"]))
    rows_finite = all(row["finite"] for row in route["gradient_rows"].values())
    gates = {
        "comparator_drift": comparator_drift,
        "scalar_gate_pass": scalar_delta <= VALUE_TOLERANCE,
        "resampling_flags_match": bayesfilter["resampling_flag"] == filterflow["resampling_flag"],
        "value_additivity_pass": route["value_additivity_pass"],
        "gradient_rows_finite": rows_finite,
        "path_boundary_clean": not any(bool(value) for value in continuation._path_boundary_manifest().values()),
    }
    gates["all_vetoes_clear"] = (
        not gates["comparator_drift"]
        and gates["scalar_gate_pass"]
        and gates["resampling_flags_match"]
        and gates["value_additivity_pass"]
        and gates["gradient_rows_finite"]
        and gates["path_boundary_clean"]
    )
    gates["scalar_delta"] = scalar_delta
    return gates


def _classify_hypothesis(
    route: dict[str, Any],
    veto_status: dict[str, Any],
    boundary_summary: dict[str, Any],
    same_tape_summary: dict[str, Any],
) -> dict[str, Any]:
    if not veto_status.get("all_vetoes_clear", False):
        if not veto_status.get("value_additivity_pass", False):
            return {
                "classification": "h1_value_additivity_veto",
                "reason": "post_update != pre_current + increment before gradient interpretation",
                "promoted": False,
                "next_smallest_probe": "repair value replay/additivity before VJP interpretation",
            }
        return {
            "classification": "blocked_by_veto",
            "reason": f"veto status failed: {veto_status}",
            "promoted": False,
            "next_smallest_probe": "repair veto before interpreting route hypotheses",
        }
    if route["component_sum_pass"]:
        return {
            "classification": "h2_component_sum_reconstructs",
            "reason": (
                "pre_current_delta + increment_delta reconstructs observed residual "
                f"within {GRADIENT_TOLERANCE}"
            ),
            "promoted": True,
            "next_smallest_probe": (
                "split the component-sum route into earlier carried-likelihood and "
                "current-increment subgraphs only if needed"
            ),
        }
    if route["post_update_pass"]:
        return {
            "classification": "h3_post_update_route_residual",
            "reason": (
                "post_update_delta reconstructs the observed residual but the "
                "component-sum reconstruction fails tolerance"
            ),
            "promoted": True,
            "next_smallest_probe": (
                "instrument the post_update tape route to explain why its "
                "parameter-path adjoint is not equal to the component-sum route"
            ),
        }
    nomination = _h4_nomination(boundary_summary, same_tape_summary)
    if nomination is not None:
        return {
            "classification": "inconclusive_h4_next_boundary_nominated",
            "reason": nomination,
            "promoted": False,
            "next_smallest_probe": nomination,
        }
    return {
        "classification": "inconclusive_no_unique_hypothesis",
        "reason": "H1-H3 did not satisfy the ordered criteria and no narrower boundary was nominated",
        "promoted": False,
        "next_smallest_probe": "add a narrower tensor-level route probe upstream of post_update_log_likelihoods",
    }


def _h4_nomination(
    boundary_summary: dict[str, Any],
    same_tape_summary: dict[str, Any],
) -> str | None:
    best = boundary_summary.get("best_value_valid_mode")
    if best is not None and float(best.get("max_abs_gradient_delta", 999.0)) < 1.0:
        return f"boundary mode {best['mode']} materially reduces the row-gradient delta"
    if not same_tape_summary.get("bayesfilter_full_recorded_state_identity_holds", True):
        return "probe BayesFilter full recorded-state same-tape identity around carried state"
    if not same_tape_summary.get("bayesfilter_same_tape_identity_holds", True):
        return "probe BayesFilter transport same-tape identity around the transport matrix"
    return None


def _decision(classification: dict[str, Any], veto_status: dict[str, Any]) -> str:
    name = classification["classification"]
    if name == "h1_value_additivity_veto":
        return "filterflow_float64_row_173_post_update_h1_value_additivity_veto"
    if name == "h2_component_sum_reconstructs":
        return "filterflow_float64_row_173_post_update_h2_component_sum_reconstructs"
    if name == "h3_post_update_route_residual":
        return "filterflow_float64_row_173_post_update_h3_route_residual"
    if name == "inconclusive_h4_next_boundary_nominated":
        return "filterflow_float64_row_173_post_update_inconclusive_h4_nominated"
    if veto_status.get("status") == "blocked" or name == "blocked_by_veto":
        return "filterflow_float64_row_173_post_update_blocked_by_veto"
    return "filterflow_float64_row_173_post_update_inconclusive_no_unique_hypothesis"


def _comparator_fingerprint(
    reference_status: dict[str, Any],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    paths = [
        FILTERFLOW_PATH / "scripts" / "simple_linear_smoothness.py",
        FILTERFLOW_PATH / "filterflow" / "resampling" / "differentiable" / "regularized_transport" / "plan.py",
        FILTERFLOW_PATH / "filterflow" / "resampling" / "differentiable" / "biased.py",
        FILTERFLOW_PATH / "filterflow" / "models" / "simple_linear_gaussian.py",
    ]
    return {
        "checkout_path": str(FILTERFLOW_PATH),
        "reference_status": reference_status,
        "fingerprint": fingerprint,
        "entrypoint_hashes": {
            str(path.relative_to(REPO_ROOT)): _file_digest_or_mtime(path) for path in paths
        },
    }


def _file_digest_or_mtime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing"}
    return {
        "status": "present",
        "sha256": stable_digest(path.read_text(encoding="utf-8", errors="replace")),
        "mtime_ns": path.stat().st_mtime_ns,
    }


def _input_manifest(
    filterflow: dict[str, Any],
    adjacent: dict[str, Any],
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    model = filterflow["model"]
    transport = filterflow["value_tensors"]["transport_matrix"]
    observations = model["observations"]
    particles = model["initial_particles"]
    return {
        "row_index": vjp.MESH_INDEX,
        "target_time_index": TARGET_TIME_INDEX,
        "theta": vjp.THETA,
        "theta_digest": stable_digest(vjp.THETA),
        "observations_checksum": filterflow.get("observation_checksum"),
        "observations_digest": stable_digest(observations),
        "initial_particles_checksum": filterflow.get("initial_particles_checksum"),
        "initial_particles_digest": stable_digest(particles),
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "resampling_flag": filterflow["resampling_flag"],
        "transport_matrix_shape": _shape_nested(transport),
        "transport_matrix_digest": stable_digest(transport),
        "dtype": "float64",
        "covariance_convention": {
            "transition_covariance": [[1.0 / 3.0, 0.5], [0.5, 1.0]],
            "observation_covariance": [[0.01]],
            "source": "executable FilterFlow constant-velocity smoothness helper",
        },
        "comparator_checkout": str(FILTERFLOW_PATH),
        "comparator_fingerprint": fingerprint,
        "adjacent_boundary_baseline": adjacent,
    }


def _shape_nested(value: Any) -> list[int]:
    shape = []
    cursor = value
    while isinstance(cursor, list):
        shape.append(len(cursor))
        cursor = cursor[0] if cursor else None
    return shape


def _scalar_resampling_gates(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    scalar_delta = abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"]))
    return {
        "filterflow_target_scalar": filterflow["target_scalar"],
        "bayesfilter_target_scalar": bayesfilter["target_scalar"],
        "scalar_delta": scalar_delta,
        "scalar_gate_pass": scalar_delta <= VALUE_TOLERANCE,
        "filterflow_resampling_flag": filterflow["resampling_flag"],
        "bayesfilter_resampling_flag": bayesfilter["resampling_flag"],
        "resampling_flags_match": filterflow["resampling_flag"] == bayesfilter["resampling_flag"],
    }


def _model_contract() -> dict[str, Any]:
    return {
        "model": "filterflow_simple_linear_smoothness_constant_velocity_lgssm",
        "mesh_index": vjp.MESH_INDEX,
        "theta": vjp.THETA,
        "target_time_index": TARGET_TIME_INDEX,
        "T": vjp.T,
        "batch_size": vjp.BATCH_SIZE,
        "num_particles": vjp.NUM_PARTICLES,
        "data_seed": vjp.DATA_SEED,
        "filter_seed": vjp.FILTER_SEED,
        "epsilon": vjp.EPSILON,
        "scaling": vjp.SCALING,
        "convergence_threshold": vjp.CONVERGENCE_THRESHOLD,
        "max_iter": vjp.MAX_ITERATIONS,
        "resampling_neff": vjp.RESAMPLING_NEFF,
        "dtype": "float64",
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 FilterFlow checkout",
        "primary_question": "which post_update route hypothesis explains row 173 time 93",
        "ordered_decision_rule": [
            "H1 value additivity veto",
            "H2 component sum reconstructs",
            "H3 post update route residual",
            "H4 nomination only or inconclusive",
        ],
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "mathematical_correctness": "not_concluded",
    }


def _compact_side(side: dict[str, Any] | None) -> dict[str, Any] | None:
    if side is None:
        return None
    if side.get("status") != "executed":
        return side
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "resampling_flag": side["resampling_flag"],
        "post_update_values": {
            field: side["values"][field] for field in POST_UPDATE_FIELDS
        },
        "post_update_parameter_path_probe": {
            field: side["parameter_path_adjoint_probe"][field] for field in POST_UPDATE_FIELDS
        },
        "resampling_adjoint_decomposition": side.get("resampling_adjoint_decomposition"),
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _decision_table(
    decision: str,
    classification: dict[str, Any],
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    if not veto_status.get("all_vetoes_clear", False) and classification["classification"] != "h1_value_additivity_veto":
        veto = "veto failed"
    else:
        veto = "none" if veto_status.get("all_vetoes_clear", False) else "value additivity veto"
    return [
        {
            "decision": decision,
            "primary_criterion_status": classification["classification"],
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single row/time post-update route probe; no correctness claim",
            "next_justified_action": classification.get("next_smallest_probe", "N/A"),
            "not_concluded": "correctness of either implementation, global smoothness-gradient agreement, production readiness",
        }
    ]


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "decision",
        "hypothesis_classification",
        "hypothesis_reason",
        "plan_path",
        "result_path",
        "evidence_contract",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "comparator_fingerprint",
        "baseline_comparator_comparison",
        "input_manifest",
        "model_contract",
        "filterflow_probe",
        "bayesfilter_probe",
        "post_update_route",
        "scalar_resampling_gates",
        "veto_status_table",
        "boundary_mode_summary_table",
        "same_tape_state_adjoint_summary_table",
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
    allowed = {
        "filterflow_float64_row_173_post_update_filterflow_blocker",
        "filterflow_float64_row_173_post_update_bayesfilter_blocker",
        "filterflow_float64_row_173_post_update_baseline_comparator_drift",
        "filterflow_float64_row_173_post_update_blocked_by_veto",
        "filterflow_float64_row_173_post_update_h1_value_additivity_veto",
        "filterflow_float64_row_173_post_update_h2_component_sum_reconstructs",
        "filterflow_float64_row_173_post_update_h3_route_residual",
        "filterflow_float64_row_173_post_update_inconclusive_h4_nominated",
        "filterflow_float64_row_173_post_update_inconclusive_no_unique_hypothesis",
    }
    if payload["decision"] not in allowed:
        raise ValueError(f"unexpected decision: {payload['decision']}")
    _validate_cpu(payload["run_manifest"], "parent")
    if any(bool(value) for value in payload["path_boundary_manifest"].values()):
        raise ValueError(f"path boundary violation: {payload['path_boundary_manifest']}")
    for label in ("filterflow_probe", "bayesfilter_probe"):
        side = payload.get(label)
        if side is not None and side.get("status") == "executed":
            _validate_cpu(side["cpu_only_manifest"], label)
    if payload["post_update_route"].get("status") == "compared":
        finite = all(
            row["finite"]
            for row in payload["post_update_route"]["gradient_rows"].values()
        )
        if not finite and "blocked" not in payload["decision"]:
            raise ValueError("nonfinite gradient rows without blocked decision")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Post-Update Route Hypothesis Probe",
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
    ]
    for row in payload["decision_table"]:
        lines.append(
            "| "
            + " | ".join(
                str(row[key]).replace("\n", " ")
                for key in (
                    "decision",
                    "primary_criterion_status",
                    "veto_diagnostic_status",
                    "main_uncertainty",
                    "next_justified_action",
                    "not_concluded",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## Post-Update Route",
            "",
            _json_block(payload["post_update_route"]),
            "",
            "## Scalar And Resampling Gates",
            "",
            _json_block(payload["scalar_resampling_gates"]),
            "",
            "## Input Manifest",
            "",
            _json_block(payload["input_manifest"]),
            "",
            "## Comparator Fingerprint",
            "",
            _json_block(payload["comparator_fingerprint"]),
            "",
            "## Baseline Comparator Comparison",
            "",
            _json_block(payload["baseline_comparator_comparison"]),
            "",
            "## Boundary Mode Summary",
            "",
            _json_block(payload["boundary_mode_summary_table"]),
            "",
            "## Same-Tape State-Adjoint Summary",
            "",
            _json_block(payload["same_tape_state_adjoint_summary_table"]),
            "",
            "## Run Manifest",
            "",
            _json_block(payload["run_manifest"]),
            "",
            "## Non-Implications",
            "",
        ]
    )
    for item in payload["non_implications"]:
        lines.append(f"- {item}")
    lines.extend(["", "## JSON Output", "", f"`{payload['json_path']}`", ""])
    return "\n".join(lines)


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _vector_delta(left: Any, right: Any) -> list[float]:
    return [
        float(lhs) - float(rhs)
        for lhs, rhs in zip(_flatten(left), _flatten(right), strict=True)
    ]


def _max_abs_delta(left: Any, right: Any) -> float:
    return _max_abs_vector(_vector_delta(left, right))


def _max_abs_vector(values: list[float]) -> float:
    return max(abs(float(value)) for value in values) if values else 0.0


def _flatten(value: Any) -> list[float]:
    out: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            out.append(float(item))

    visit(value)
    return out


def _finite_nested(value: Any) -> bool:
    return bool(
        tf.reduce_all(
            tf.math.is_finite(tf.constant(_flatten(value), dtype=DTYPE))
        ).numpy()
    )


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    if "run_manifest" in clone:
        clone["run_manifest"].pop("wall_time_seconds", None)
    return stable_digest(clone)


def _non_implications() -> list[str]:
    return vjp._non_implications() + [
        "This post-update route probe is a row-173/time-93 difference audit only.",
        "No correctness claim is made for either implementation.",
        "No global smoothness-gradient correctness is concluded.",
        "No claim is made about other rows, times, horizons, parameter settings, or models.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
