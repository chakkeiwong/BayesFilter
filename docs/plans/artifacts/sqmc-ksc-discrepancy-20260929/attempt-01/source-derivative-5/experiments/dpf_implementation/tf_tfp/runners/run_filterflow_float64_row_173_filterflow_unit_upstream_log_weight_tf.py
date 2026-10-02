"""Compare row-173 FilterFlow and BayesFilter unit log-weight carry VJPs."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import subprocess
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
    "bayesfilter-dpf-filterflow-float64-row-173-filterflow-unit-upstream-log-weight-plan-2026-06-06.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-filterflow-unit-upstream-log-weight-result-2026-06-06.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-filterflow-unit-upstream-log-weight-review-loop-2026-06-06.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_filterflow_unit_upstream_log_weight_2026-06-06.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-filterflow-unit-upstream-log-weight-2026-06-06.md"
)
PRIOR_FACTOR_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_log_weight_edge_factorization_2026-06-05.json"
)

TARGET_TIME_INDEX = 93
TAG = "row-173-filterflow-unit-upstream-log-weight"
VALUE_TOLERANCE = vjp.VALUE_TOLERANCE
GRADIENT_TOLERANCE = vjp.GRADIENT_TOLERANCE

UNIT_FIELD = "pre_log_weights_to_pre_particles_unit_upstream"
TARGET_FIELD = "target_to_pre_log_weights"
COMPOSED_FIELD = "same_tape_pre_log_weights_carryover_vjp"


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
    prior = _load_prior_factorization()
    config = vjp.RunConfig(
        target_time_index=TARGET_TIME_INDEX,
        tag=TAG,
        plan_path=PLAN_PATH,
        result_path=RESULT_PATH,
        json_path=JSON_PATH,
        report_path=REPORT_PATH,
    )
    filterflow = _filterflow_unit_upstream_subprocess(config)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(
        initial_fingerprint,
        final_fingerprint,
    )
    if filterflow.get("status") != "executed":
        comparison = {"status": "blocked", "blocker": filterflow.get("blocker")}
    else:
        comparison = _compare_unit_edges(filterflow, prior)
    veto_status = _veto_status(
        filterflow,
        prior,
        comparison,
        reference_status,
        comparator_drift,
    )
    classification, reason = _classify(comparison, veto_status)
    decision = _decision(classification, veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_filterflow_unit_upstream_log_weight_difference_audit",
        "decision": decision,
        "hypothesis_classification": classification,
        "hypothesis_reason": reason,
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "filterflow_unit_upstream": _compact_filterflow(filterflow),
        "prior_factorization": _compact_prior(prior),
        "comparison": comparison,
        "veto_status_table": veto_status,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "decision_table": _decision_table(decision, classification, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners."
                "run_filterflow_float64_row_173_filterflow_unit_upstream_log_weight_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
    }


def _filterflow_unit_upstream_subprocess(config: vjp.RunConfig) -> dict[str, Any]:
    if not vjp.FILTERFLOW_ENV_PYTHON.exists():
        return {
            "status": "blocked",
            "blocker": f"missing filterflow env python: {vjp.FILTERFLOW_ENV_PYTHON}",
        }
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = str(vjp.FILTERFLOW_PATH)
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    completed = subprocess.run(
        [str(vjp.FILTERFLOW_ENV_PYTHON), "-c", _filterflow_unit_upstream_script(config)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        return {
            "status": "blocked",
            "blocker": "filterflow unit-upstream subprocess failed",
            "returncode": completed.returncode,
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_ROW_173_UNIT_UPSTREAM_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_ROW_173_UNIT_UPSTREAM_JSON_END")
    if start < 0 or end < 0 or end <= start:
        return {
            "status": "blocked",
            "blocker": "filterflow unit-upstream JSON sentinels missing",
            "stdout_excerpt": completed.stdout[-4000:],
            "stderr_excerpt": completed.stderr[-4000:],
        }
    payload = json.loads(
        completed.stdout[
            start + len("FILTERFLOW_ROW_173_UNIT_UPSTREAM_JSON_BEGIN") : end
        ].strip()
    )
    payload["stderr_excerpt"] = completed.stderr[-2000:]
    return payload


def _filterflow_unit_upstream_script(config: vjp.RunConfig) -> str:
    script = vjp._filterflow_vjp_script(config)
    script = _replace_once(
        script,
        """same_tape_pre_log_weights_carryover_vjp = watched_grad_with_upstream(
    tape,
    target_bundle["pre_log_weights"],
    target_bundle["pre_particles"],
    gradients["pre_log_weights"],
)
""",
        """same_tape_pre_log_weights_carryover_vjp = watched_grad_with_upstream(
    tape,
    target_bundle["pre_log_weights"],
    target_bundle["pre_particles"],
    gradients["pre_log_weights"],
)
same_tape_pre_log_weights_unit_upstream_vjp = watched_grad_with_upstream(
    tape,
    target_bundle["pre_log_weights"],
    target_bundle["pre_particles"],
    tf.ones_like(target_bundle["pre_log_weights"]),
)
""",
    )
    script = _replace_once(
        script,
        """        "same_tape_pre_log_weights_carryover_vjp": field(
            same_tape_pre_log_weights_carryover_vjp
        ),
""",
        """        "same_tape_pre_log_weights_carryover_vjp": field(
            same_tape_pre_log_weights_carryover_vjp
        ),
        "same_tape_pre_log_weights_unit_upstream_vjp": field(
            same_tape_pre_log_weights_unit_upstream_vjp
        ),
""",
    )
    script = _replace_once(
        script,
        """        "same_tape_pre_log_weights_carryover_vjp_tensor": to_json(
            same_tape_pre_log_weights_carryover_vjp
        ),
""",
        """        "same_tape_pre_log_weights_carryover_vjp_tensor": to_json(
            same_tape_pre_log_weights_carryover_vjp
        ),
        "same_tape_pre_log_weights_unit_upstream_vjp_tensor": to_json(
            same_tape_pre_log_weights_unit_upstream_vjp
        ),
""",
    )
    script = _replace_once(
        script,
        'print("FILTERFLOW_ROW_173_VJP_JSON_BEGIN")',
        'print("FILTERFLOW_ROW_173_UNIT_UPSTREAM_JSON_BEGIN")',
    )
    script = _replace_once(
        script,
        'print("FILTERFLOW_ROW_173_VJP_JSON_END")',
        'print("FILTERFLOW_ROW_173_UNIT_UPSTREAM_JSON_END")',
    )
    return script


def _replace_once(script: str, old: str, new: str) -> str:
    count = script.count(old)
    if count != 1:
        raise ValueError(f"expected exactly one script replacement target, found {count}")
    return script.replace(old, new, 1)


def _load_prior_factorization() -> dict[str, Any]:
    if not PRIOR_FACTOR_JSON.exists():
        return {"status": "missing", "path": str(PRIOR_FACTOR_JSON)}
    payload = load_json(PRIOR_FACTOR_JSON)
    return {"status": "loaded", "path": str(PRIOR_FACTOR_JSON), "payload": payload}


def _compare_unit_edges(filterflow: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    if prior.get("status") != "loaded":
        return {"status": "blocked", "blocker": "prior factorization artifact missing"}
    prior_payload = prior["payload"]
    raw_mode = _bayesfilter_raw_mode(prior_payload)
    ff_edges = _filterflow_edges(filterflow)
    bf_edges = _bayesfilter_edges(raw_mode)
    rows = {
        field: _delta_row(ff_edges[field], bf_edges[field])
        for field in (TARGET_FIELD, UNIT_FIELD, COMPOSED_FIELD)
    }
    return {
        "status": "compared",
        "rows": rows,
        "target_to_pre_log_weights_delta": rows[TARGET_FIELD]["max_abs_delta"],
        "unit_upstream_delta": rows[UNIT_FIELD]["max_abs_delta"],
        "composed_carryover_delta": rows[COMPOSED_FIELD]["max_abs_delta"],
        "unit_upstream_within_tolerance": rows[UNIT_FIELD]["max_abs_delta"]
        <= GRADIENT_TOLERANCE,
        "composed_within_tolerance": rows[COMPOSED_FIELD]["max_abs_delta"]
        <= GRADIENT_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "value_tolerance": VALUE_TOLERANCE,
        "filterflow_target_scalar": filterflow["target_scalar"],
        "prior_filterflow_target_scalar": prior_payload["filterflow_vjp"][
            "target_scalar"
        ],
        "bayesfilter_target_scalar": raw_mode["target_scalar"],
        "current_vs_prior_filterflow_scalar_delta": abs(
            float(filterflow["target_scalar"])
            - float(prior_payload["filterflow_vjp"]["target_scalar"])
        ),
        "current_filterflow_vs_bayesfilter_scalar_delta": abs(
            float(filterflow["target_scalar"]) - float(raw_mode["target_scalar"])
        ),
        "filterflow_resampling_flag": filterflow["resampling_flag"],
        "prior_filterflow_resampling_flag": prior_payload["filterflow_vjp"][
            "resampling_flag"
        ],
        "bayesfilter_resampling_flag": raw_mode["resampling_flag"],
        "prior_factorization_decision": prior_payload.get("decision"),
        "prior_factorization_classification": prior_payload.get(
            "hypothesis_classification"
        ),
        "prior_factorization_digest": prior_payload.get("reproducibility_digest"),
    }


def _bayesfilter_raw_mode(prior_payload: dict[str, Any]) -> dict[str, Any]:
    for mode in prior_payload.get("bayesfilter_mode_vjps", []):
        if mode.get("mode") == "raw":
            return mode
    raise KeyError("prior factorization artifact has no raw BayesFilter mode")


def _filterflow_edges(filterflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decomp = filterflow["resampling_adjoint_decomposition"]
    return {
        TARGET_FIELD: {
            "summary": filterflow["gradients"]["pre_log_weights"],
            "tensor": filterflow["gradient_tensors"]["pre_log_weights"],
        },
        UNIT_FIELD: {
            "summary": decomp["same_tape_pre_log_weights_unit_upstream_vjp"],
            "tensor": decomp["same_tape_pre_log_weights_unit_upstream_vjp_tensor"],
        },
        COMPOSED_FIELD: {
            "summary": decomp["same_tape_pre_log_weights_carryover_vjp"],
            "tensor": decomp["same_tape_pre_log_weights_carryover_vjp_tensor"],
        },
    }


def _bayesfilter_edges(raw_mode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        TARGET_FIELD: {
            "summary": raw_mode["edge_summaries"][TARGET_FIELD],
            "tensor": raw_mode["edge_tensors"][TARGET_FIELD],
        },
        UNIT_FIELD: {
            "summary": raw_mode["edge_summaries"]["pre_log_weights_to_pre_particles"],
            "tensor": raw_mode["edge_tensors"]["pre_log_weights_to_pre_particles"],
        },
        COMPOSED_FIELD: {
            "summary": raw_mode["edge_summaries"][COMPOSED_FIELD],
            "tensor": raw_mode["edge_tensors"][COMPOSED_FIELD],
        },
    }


def _delta_row(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_summary = filterflow["summary"]
    bf_summary = bayesfilter["summary"]
    return {
        "status": "compared",
        "shape_match": ff_summary["shape"] == bf_summary["shape"],
        "finite": bool(ff_summary["finite"] and bf_summary["finite"]),
        "filterflow_max_abs": ff_summary["max_abs"],
        "bayesfilter_max_abs": bf_summary["max_abs"],
        "filterflow_sum": ff_summary["sum"],
        "bayesfilter_sum": bf_summary["sum"],
        "max_abs_delta": vjp._max_abs_nested_delta(
            bayesfilter["tensor"],
            filterflow["tensor"],
        ),
        "sum_delta": _sum_nested_delta(
            bayesfilter["tensor"],
            filterflow["tensor"],
        ),
    }


def _sum_nested_delta(left: Any, right: Any) -> float:
    deltas: list[float] = []

    def visit(lhs: Any, rhs: Any) -> None:
        if isinstance(lhs, list) and isinstance(rhs, list):
            for lhs_item, rhs_item in zip(lhs, rhs, strict=True):
                visit(lhs_item, rhs_item)
        else:
            deltas.append(float(lhs) - float(rhs))

    visit(left, right)
    return sum(deltas)


def _veto_status(
    filterflow: dict[str, Any],
    prior: dict[str, Any],
    comparison: dict[str, Any],
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
    }
    if filterflow.get("status") == "executed":
        cpu_rows["filterflow"] = _cpu_status(filterflow["cpu_only_manifest"])
    if prior.get("status") == "loaded":
        prior_payload = prior["payload"]
        cpu_rows["prior_run_manifest"] = _cpu_status(prior_payload["run_manifest"])
        raw_mode = _bayesfilter_raw_mode(prior_payload)
        cpu_rows["prior_bayesfilter_raw_mode"] = _cpu_status(raw_mode["cpu_only_manifest"])
    required_tensors_present = _required_tensors_present(filterflow, prior, comparison)
    edge_tensors_finite = _edge_tensors_finite(comparison)
    scalar_gate_pass = (
        comparison.get("status") == "compared"
        and comparison["current_vs_prior_filterflow_scalar_delta"] <= VALUE_TOLERANCE
        and comparison["current_filterflow_vs_bayesfilter_scalar_delta"] <= VALUE_TOLERANCE
    )
    resampling_flags_match = (
        comparison.get("status") == "compared"
        and comparison["filterflow_resampling_flag"]
        == comparison["prior_filterflow_resampling_flag"]
        == comparison["bayesfilter_resampling_flag"]
    )
    prior_artifact_valid = (
        prior.get("status") == "loaded"
        and prior["payload"].get("decision")
        == "filterflow_float64_row_173_log_weight_edge_factorization_composition_edge"
        and prior["payload"].get("hypothesis_classification") == "h2_composition_edge"
    )
    status = {
        "all_vetoes_clear": False,
        "filterflow_executed": filterflow.get("status") == "executed",
        "prior_artifact_loaded": prior.get("status") == "loaded",
        "prior_artifact_valid": prior_artifact_valid,
        "comparator_drift": comparator_drift,
        "reference_status_validated": bool(reference_status),
        "scalar_gate_pass": scalar_gate_pass,
        "resampling_flags_match": resampling_flags_match,
        "required_tensors_present": required_tensors_present,
        "edge_tensors_finite": edge_tensors_finite,
        "cpu_only_pass": all(row["pass"] for row in cpu_rows.values()),
        "cpu_rows": cpu_rows,
        "path_boundary_clean": not any(bool(value) for value in path_boundary.values()),
        "path_boundary_manifest": path_boundary,
    }
    status["all_vetoes_clear"] = bool(
        status["filterflow_executed"]
        and status["prior_artifact_loaded"]
        and status["prior_artifact_valid"]
        and not status["comparator_drift"]
        and status["scalar_gate_pass"]
        and status["resampling_flags_match"]
        and status["required_tensors_present"]
        and status["edge_tensors_finite"]
        and status["cpu_only_pass"]
        and status["path_boundary_clean"]
    )
    return status


def _required_tensors_present(
    filterflow: dict[str, Any],
    prior: dict[str, Any],
    comparison: dict[str, Any],
) -> bool:
    if filterflow.get("status") != "executed" or prior.get("status") != "loaded":
        return False
    if comparison.get("status") != "compared":
        return False
    try:
        _filterflow_edges(filterflow)
        _bayesfilter_edges(_bayesfilter_raw_mode(prior["payload"]))
    except (KeyError, TypeError):
        return False
    return all(row["shape_match"] for row in comparison.get("rows", {}).values())


def _edge_tensors_finite(comparison: dict[str, Any]) -> bool:
    if comparison.get("status") != "compared":
        return False
    return all(bool(row["finite"]) for row in comparison["rows"].values())


def _cpu_status(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        "pre_import_cuda_visible_devices": manifest.get(
            "pre_import_cuda_visible_devices"
        ),
        "gpu_devices_visible": manifest.get("gpu_devices_visible"),
        "pass": (
            manifest.get("cuda_visible_devices") == "-1"
            and manifest.get("pre_import_cuda_visible_devices") == "-1"
            and manifest.get("gpu_devices_visible") == []
        ),
    }


def _classify(
    comparison: dict[str, Any],
    veto_status: dict[str, Any],
) -> tuple[str, str]:
    if not veto_status.get("all_vetoes_clear", False):
        return "h1_blocked_or_vetoed", "one or more required veto gates failed"
    unit_delta = float(comparison["unit_upstream_delta"])
    composed_delta = float(comparison["composed_carryover_delta"])
    if unit_delta > GRADIENT_TOLERANCE:
        return (
            "h2_unit_upstream_factor_differs",
            "the executable FilterFlow unit-upstream previous-carry VJP differs materially from BayesFilter",
        )
    if composed_delta > GRADIENT_TOLERANCE:
        return (
            "h3_unit_upstream_matches_composed_differs",
            "the unit-upstream VJP matches but the target-upstream-composed carryover VJP differs materially",
        )
    if unit_delta <= GRADIENT_TOLERANCE and composed_delta <= GRADIENT_TOLERANCE:
        return (
            "h4_log_weight_unit_probe_matches",
            "the unit-upstream and composed log-weight carryover VJPs both match within tolerance",
        )
    return (
        "h5_unresolved_unit_probe",
        "finite value-valid evidence did not isolate the unit-upstream factor",
    )


def _decision(classification: str, veto_status: dict[str, Any]) -> str:
    if not veto_status.get("all_vetoes_clear", False):
        return "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_blocked_or_vetoed"
    mapping = {
        "h2_unit_upstream_factor_differs": (
            "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_unit_factor_differs"
        ),
        "h3_unit_upstream_matches_composed_differs": (
            "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_composition_only_differs"
        ),
        "h4_log_weight_unit_probe_matches": (
            "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_matches"
        ),
        "h5_unresolved_unit_probe": (
            "filterflow_float64_row_173_filterflow_unit_upstream_log_weight_unresolved"
        ),
    }
    return mapping[classification]


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
        "unit_upstream_probe": (
            "VJP(pre_log_weights wrt pre_particles, ones_like(pre_log_weights))"
        ),
        "prior_factorization_artifact": str(PRIOR_FACTOR_JSON.relative_to(REPO_ROOT)),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "At row 173 and target time 93, does executable float64 FilterFlow's "
            "unit-upstream VJP(pre_log_weights wrt pre_particles, ones_like(pre_log_weights)) "
            "match BayesFilter's unit-upstream previous-carry factor?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "primary_criterion": (
            "classify unit-upstream previous-carry VJP comparison after vetoes clear"
        ),
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "vetoes": [
            "FilterFlow unit-upstream subprocess cannot execute",
            "accepted BayesFilter factorization artifact missing or invalid",
            "comparator fingerprint changes during the run",
            "CPU-only manifest violation",
            "scalar or resampling flag mismatch",
            "required unit/composed tensors missing or non-finite",
            "path-boundary contamination",
        ],
        "explanatory_only": [
            "target-to-pre-log-weight upstream delta",
            "composed log-weight carryover VJP delta",
            "raw total-gradient delta from the prior factorization artifact",
        ],
        "not_concluded": _non_implications(),
    }


def _decision_table(
    decision: str,
    classification: str,
    veto_status: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "Decision": decision,
            "Primary criterion status": classification,
            "Veto diagnostic status": json.dumps(
                {key: value for key, value in veto_status.items() if key != "cpu_rows"},
                sort_keys=True,
            ),
            "Main uncertainty": "single row and target time; no correctness claim",
            "Next justified action": _next_action(classification),
            "Not concluded": "correctness, posterior correctness, production readiness, global agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h2_unit_upstream_factor_differs":
        return "localize why FilterFlow and BayesFilter differ in the previous-log-weight carry Jacobian"
    if classification == "h3_unit_upstream_matches_composed_differs":
        return "localize composition of target upstream with a matching previous-carry Jacobian"
    if classification == "h4_log_weight_unit_probe_matches":
        return "move to the next non-log-weight route in the row-173 decomposition"
    if classification == "h5_unresolved_unit_probe":
        return "add tensor-entry localization for the unit-upstream log-weight route"
    return "repair blocker before interpreting unit-upstream evidence"


def _compact_filterflow(filterflow: dict[str, Any]) -> dict[str, Any]:
    if filterflow.get("status") != "executed":
        return filterflow
    edges = _filterflow_edges(filterflow)
    return {
        "status": filterflow["status"],
        "backend": filterflow["backend"],
        "settings": filterflow["settings"],
        "target_scalar": filterflow["target_scalar"],
        "resampling_flag": filterflow["resampling_flag"],
        "edge_summaries": {
            name: edge["summary"] for name, edge in edges.items()
        },
        "cpu_only_manifest": filterflow["cpu_only_manifest"],
        "stderr_excerpt": filterflow.get("stderr_excerpt", ""),
    }


def _compact_prior(prior: dict[str, Any]) -> dict[str, Any]:
    if prior.get("status") != "loaded":
        return prior
    payload = prior["payload"]
    raw_mode = _bayesfilter_raw_mode(payload)
    return {
        "status": "loaded",
        "path": prior["path"],
        "decision": payload.get("decision"),
        "hypothesis_classification": payload.get("hypothesis_classification"),
        "reproducibility_digest": payload.get("reproducibility_digest"),
        "filterflow_target_scalar": payload["filterflow_vjp"]["target_scalar"],
        "bayesfilter_target_scalar": raw_mode["target_scalar"],
        "filterflow_resampling_flag": payload["filterflow_vjp"]["resampling_flag"],
        "bayesfilter_resampling_flag": raw_mode["resampling_flag"],
        "bayesfilter_edge_summaries": {
            TARGET_FIELD: raw_mode["edge_summaries"][TARGET_FIELD],
            UNIT_FIELD: raw_mode["edge_summaries"][
                "pre_log_weights_to_pre_particles"
            ],
            COMPOSED_FIELD: raw_mode["edge_summaries"][COMPOSED_FIELD],
        },
        "graph_embedding_composed_consistency": _graph_embedding_composed_consistency(payload),
    }


def _graph_embedding_composed_consistency(payload: dict[str, Any]) -> dict[str, Any]:
    raw_mode = _bayesfilter_raw_mode(payload)
    graph = payload["raw_bayesfilter_vjp"]["graph_embedding_probe"]
    graph_tensor = graph["tensors"]["pre_log_weights_to_pre_particles_vjp"]
    raw_tensor = raw_mode["edge_tensors"][COMPOSED_FIELD]
    return {
        "raw_mode_composed_vs_graph_embedding_max_abs_delta": vjp._max_abs_nested_delta(
            raw_tensor,
            graph_tensor,
        ),
        "raw_mode_composed_summary": raw_mode["edge_summaries"][COMPOSED_FIELD],
        "graph_embedding_summary": graph["summaries"][
            "pre_log_weights_to_pre_particles_vjp"
        ],
        "status": "explanatory_composed_vjp_consistency_check",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "created_at_utc",
        "question",
        "decision",
        "hypothesis_classification",
        "hypothesis_reason",
        "plan_path",
        "result_path",
        "review_loop_path",
        "reference_policy",
        "filterflow_status",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "model_contract",
        "filterflow_unit_upstream",
        "prior_factorization",
        "comparison",
        "veto_status_table",
        "path_boundary_manifest",
        "evidence_contract",
        "decision_table",
        "non_implications",
        "run_manifest",
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
    _validate_cpu(
        payload["filterflow_unit_upstream"]["cpu_only_manifest"],
        "filterflow",
    )
    comparison = payload["comparison"]
    if comparison["current_vs_prior_filterflow_scalar_delta"] > VALUE_TOLERANCE:
        raise ValueError("current FilterFlow scalar differs from prior FilterFlow scalar")
    if comparison["current_filterflow_vs_bayesfilter_scalar_delta"] > VALUE_TOLERANCE:
        raise ValueError("current FilterFlow scalar differs from BayesFilter scalar")
    if not (
        comparison["filterflow_resampling_flag"]
        == comparison["prior_filterflow_resampling_flag"]
        == comparison["bayesfilter_resampling_flag"]
    ):
        raise ValueError("resampling flags differ")
    if payload["hypothesis_classification"] == "h2_unit_upstream_factor_differs":
        if comparison["unit_upstream_delta"] <= GRADIENT_TOLERANCE:
            raise ValueError("h2 classification without material unit delta")


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
            "# Result: Row 173 FilterFlow Unit-Upstream Log-Weight Probe",
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
            "## Comparison",
            "",
            _json_block(payload["comparison"]),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## FilterFlow Unit-Upstream Probe",
            "",
            _json_block(payload["filterflow_unit_upstream"]),
            "",
            "## Prior Factorization",
            "",
            _json_block(payload["prior_factorization"]),
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


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True, default=str) + "\n```"


def _digest_payload(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    copy.pop("reproducibility_digest", None)
    return stable_digest(copy)


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
