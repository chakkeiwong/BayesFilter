"""Probe the row-173 adjacent upstream-adjoint boundary."""

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
    run_filterflow_float64_row_173_direct_theta_hypothesis_tf as direct_theta,
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
    "bayesfilter-dpf-filterflow-float64-row-173-adjacent-boundary-plan-2026-06-04.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-adjacent-boundary-result-2026-06-04.md"
)
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_float64_row_173_adjacent_boundary_2026-06-04.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-float64-row-173-adjacent-boundary-2026-06-04.md"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

TARGET_TIME_INDEX = 93
TAG = "row-173-adjacent-boundary"
VALUE_TOLERANCE = 5e-8
GRADIENT_TOLERANCE = 2e-4
OBSERVED_TOTAL_GRADIENT_DELTA = [5.302734403676368, -0.1337765252068337]
DIRECT_THETA_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_direct_theta_hypothesis_2026-06-04.json"
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
            "filterflow_float64_row_173_adjacent_boundary_filterflow_blocker",
            filterflow.get("blocker", "unknown filterflow blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )
    bayesfilter = vjp._bayesfilter_vjp(filterflow, config)
    if bayesfilter.get("status") != "executed":
        return _blocked_payload(
            "filterflow_float64_row_173_adjacent_boundary_bayesfilter_blocker",
            bayesfilter.get("blocker", "unknown BayesFilter blocker"),
            initial_fingerprint,
            filterflow,
            reference_status,
        )

    direct_gate = _direct_theta_gate()
    comparison = _compare_adjacent_boundary(filterflow, bayesfilter, direct_gate)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = continuation._fingerprints_drifted(initial_fingerprint, final_fingerprint)
    decision = _decision(comparison, comparator_drift)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "report_path": str(REPORT_PATH.relative_to(REPO_ROOT)),
        "json_path": str(JSON_PATH.relative_to(REPO_ROOT)),
        "question": "row_173_time_93_adjacent_upstream_adjoint_boundary_probe",
        "evidence_contract": _evidence_contract(),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": initial_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "model_contract": _model_contract(),
        "direct_theta_gate": direct_gate,
        "filterflow_probe": _compact_side(filterflow),
        "bayesfilter_probe": _compact_side(bayesfilter),
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_adjacent_boundary_probe_tf"
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
        },
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _blocked_payload(
    decision: str,
    blocker: str,
    initial_fingerprint: dict[str, Any],
    filterflow: dict[str, Any],
    reference_status: dict[str, Any],
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
        "direct_theta_gate": _direct_theta_gate(required=False),
        "filterflow_probe": filterflow,
        "bayesfilter_probe": None,
        "comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_adjacent_boundary_probe_tf"
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
        },
        "decision_table": _decision_table(decision, comparison),
        "non_implications": _non_implications(),
    }


def _direct_theta_gate(required: bool = True) -> dict[str, Any]:
    if not DIRECT_THETA_JSON.exists():
        if required:
            raise FileNotFoundError(f"missing direct-theta gate JSON: {DIRECT_THETA_JSON}")
        return {"status": "missing", "path": str(DIRECT_THETA_JSON.relative_to(REPO_ROOT))}
    payload = load_json(DIRECT_THETA_JSON)
    best_increment = payload["comparison"]["best_increment_row"]
    best_core = payload["comparison"]["best_core_row"]
    gate_pass = (
        payload["decision"] == "filterflow_float64_row_173_direct_theta_not_source"
        and best_increment["max_abs_gradient_delta"] <= GRADIENT_TOLERANCE
        and best_core["max_abs_gradient_delta"] <= GRADIENT_TOLERANCE
    )
    return {
        "status": "passed" if gate_pass else "failed",
        "path": str(DIRECT_THETA_JSON.relative_to(REPO_ROOT)),
        "decision": payload["decision"],
        "interpretation": payload["comparison"]["interpretation"],
        "best_increment_row": best_increment,
        "best_core_row": best_core,
    }


def _compare_adjacent_boundary(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
    direct_gate: dict[str, Any],
) -> dict[str, Any]:
    scalar_delta = abs(float(bayesfilter["target_scalar"]) - float(filterflow["target_scalar"]))
    total_gradient_delta = _vector_delta(
        bayesfilter["total_gradient_diag"],
        filterflow["total_gradient_diag"],
    )
    parameter_path_rows = _parameter_path_rows(filterflow, bayesfilter)
    best_parameter_path = _best_residual_match(parameter_path_rows)
    adjoint_summary = _adjoint_decomposition_summary(filterflow, bayesfilter)
    boundary_rows = _boundary_rows(filterflow, bayesfilter)
    value_gate_pass = scalar_delta <= VALUE_TOLERANCE and direct_gate["status"] == "passed"
    finite_gate_pass = all(row["finite"] for row in parameter_path_rows.values())
    interpretation = _interpret(
        value_gate_pass,
        finite_gate_pass,
        best_parameter_path,
        adjoint_summary,
    )
    return {
        "status": "compared",
        "scalar_delta": scalar_delta,
        "value_gate_pass": value_gate_pass,
        "finite_gate_pass": finite_gate_pass,
        "total_gradient_delta": total_gradient_delta,
        "observed_total_gradient_delta": OBSERVED_TOTAL_GRADIENT_DELTA,
        "parameter_path_rows": parameter_path_rows,
        "best_parameter_path_residual_match": best_parameter_path,
        "adjoint_decomposition_summary": adjoint_summary,
        "boundary_rows": boundary_rows,
        "interpretation": interpretation,
    }


def _parameter_path_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    ff_rows = filterflow["parameter_path_adjoint_tensors"]
    bf_rows = bayesfilter["parameter_path_adjoint_tensors"]
    summaries = {}
    for name in sorted(ff_rows):
        if name not in bf_rows:
            continue
        delta = _vector_delta(bf_rows[name], ff_rows[name])
        residual_after = [
            OBSERVED_TOTAL_GRADIENT_DELTA[i] - delta[i]
            for i in range(len(OBSERVED_TOTAL_GRADIENT_DELTA))
        ]
        summaries[name] = {
            "filterflow": ff_rows[name],
            "bayesfilter": bf_rows[name],
            "delta": delta,
            "max_abs_delta": max(abs(value) for value in delta),
            "residual_after_subtracting_from_observed": residual_after,
            "max_abs_residual_after_subtracting_from_observed": max(
                abs(value) for value in residual_after
            ),
            "finite": _finite_vector(ff_rows[name]) and _finite_vector(bf_rows[name]),
        }
    return summaries


def _best_residual_match(rows: dict[str, Any]) -> dict[str, Any] | None:
    if not rows:
        return None
    name, row = min(
        rows.items(),
        key=lambda item: item[1]["max_abs_residual_after_subtracting_from_observed"],
    )
    return {"field": name, **row}


def _adjoint_decomposition_summary(
    filterflow: dict[str, Any],
    bayesfilter: dict[str, Any],
) -> dict[str, Any]:
    ff_rows = filterflow["resampling_adjoint_decomposition"]
    bf_rows = bayesfilter["resampling_adjoint_decomposition"]
    keys = [
        "same_tape_post_particles_vjp",
        "same_tape_pre_current_ll_carryover_vjp",
        "same_tape_transport_matrix_vjp",
        "same_tape_full_recorded_state_residual",
        "same_tape_identity_residual",
        "current_increment_pre_particle_adjoint",
        "carryover_pre_particle_adjoint",
    ]
    out = {}
    for key in keys:
        ff = ff_rows.get(key, {})
        bf = bf_rows.get(key, {})
        if not ff or not bf:
            out[key] = {"status": "missing"}
            continue
        out[key] = {
            "status": "compared",
            "filterflow_max_abs": ff.get("max_abs"),
            "bayesfilter_max_abs": bf.get("max_abs"),
            "max_abs_delta": None
            if ff.get("max_abs") is None or bf.get("max_abs") is None
            else abs(float(bf["max_abs"]) - float(ff["max_abs"])),
            "filterflow_sum": ff.get("sum"),
            "bayesfilter_sum": bf.get("sum"),
        }
    return out


def _boundary_rows(filterflow: dict[str, Any], bayesfilter: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "post_particles",
        "proposed_particles",
        "proposal_mean",
        "manual_proposal_mean",
        "fresh_proposal_mean",
        "proposal_ll",
        "increment",
        "pre_current_log_likelihoods",
        "pre_particles",
        "transport_matrix",
    ]
    rows = {}
    for field in fields:
        ff_gradient = filterflow["gradients"].get(field)
        bf_gradient = bayesfilter["gradients"].get(field)
        ff_param = filterflow["parameter_path_adjoint_tensors"].get(field)
        bf_param = bayesfilter["parameter_path_adjoint_tensors"].get(field)
        rows[field] = {
            "filterflow_gradient_summary": ff_gradient,
            "bayesfilter_gradient_summary": bf_gradient,
            "parameter_path_delta": None
            if ff_param is None or bf_param is None
            else _vector_delta(bf_param, ff_param),
        }
    return rows


def _interpret(
    value_gate_pass: bool,
    finite_gate_pass: bool,
    best_parameter_path: dict[str, Any] | None,
    adjoint_summary: dict[str, Any],
) -> str:
    if not value_gate_pass:
        return "value_gate_blocks_adjacent_boundary_interpretation"
    if not finite_gate_pass:
        return "nonfinite_gate_blocks_adjacent_boundary_interpretation"
    if (
        best_parameter_path is not None
        and best_parameter_path["max_abs_residual_after_subtracting_from_observed"]
        <= GRADIENT_TOLERANCE
    ):
        return f"adjacent_boundary_residual_reconstructed_by_{best_parameter_path['field']}"
    identity_delta = adjoint_summary.get("same_tape_identity_residual", {}).get("max_abs_delta")
    full_delta = adjoint_summary.get("same_tape_full_recorded_state_residual", {}).get("max_abs_delta")
    if (
        identity_delta is not None
        and identity_delta > GRADIENT_TOLERANCE
        or full_delta is not None
        and full_delta > GRADIENT_TOLERANCE
    ):
        return "adjacent_boundary_points_to_state_adjoint_identity_break"
    return "adjacent_boundary_not_explained_by_parameter_path_rows"


def _decision(comparison: dict[str, Any], comparator_drift: bool) -> str:
    if comparator_drift:
        return "filterflow_float64_row_173_adjacent_boundary_blocked_by_comparator_drift"
    if comparison.get("status") != "compared":
        return "filterflow_float64_row_173_adjacent_boundary_blocked"
    if not comparison["value_gate_pass"]:
        return "filterflow_float64_row_173_adjacent_boundary_value_veto"
    if not comparison["finite_gate_pass"]:
        return "filterflow_float64_row_173_adjacent_boundary_nonfinite_veto"
    interpretation = comparison["interpretation"]
    if interpretation.startswith("adjacent_boundary_residual_reconstructed_by_"):
        return "filterflow_float64_row_173_adjacent_boundary_reconstructed"
    if interpretation == "adjacent_boundary_points_to_state_adjoint_identity_break":
        return "filterflow_float64_row_173_adjacent_boundary_state_adjoint_identity"
    return "filterflow_float64_row_173_adjacent_boundary_unresolved"


def _evidence_contract() -> dict[str, Any]:
    return {
        "primary_comparator": "local executable float64 FilterFlow checkout",
        "primary_question": "does adjacent upstream-adjoint routing explain row 173 time 93",
        "primary_pass": "reconstruct observed gradient residual from parameter-path or state-adjoint rows",
        "artifact": {
            "result": RESULT_PATH,
            "json": str(JSON_PATH.relative_to(REPO_ROOT)),
        },
        "value_tolerance": VALUE_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE,
        "mathematical_correctness": "not_concluded",
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


def _compact_side(side: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": side["status"],
        "backend": side["backend"],
        "settings": side["settings"],
        "target_scalar": side["target_scalar"],
        "total_gradient_diag": side["total_gradient_diag"],
        "resampling_flag": side["resampling_flag"],
        "transport_upstream_clip_fraction": side["transport_upstream_clip_fraction"],
        "parameter_path_adjoint_probe": side.get("parameter_path_adjoint_probe"),
        "local_post_particle_adjoint_probe": side.get("local_post_particle_adjoint_probe"),
        "resampling_adjoint_decomposition": side.get("resampling_adjoint_decomposition"),
        "gradient_summaries": side.get("gradients"),
        "cpu_only_manifest": side["cpu_only_manifest"],
        "stderr_excerpt": side.get("stderr_excerpt", ""),
    }


def _decision_table(decision: str, comparison: dict[str, Any]) -> list[dict[str, str]]:
    if comparison.get("status") != "compared":
        primary = comparison.get("blocker", decision)
        veto = decision
        next_action = "repair blocker before interpreting adjacent-boundary evidence"
    elif not comparison["value_gate_pass"]:
        primary = "scalar or direct-theta value gate failed"
        veto = "value gate"
        next_action = "repair value/direct-theta gate before interpreting adjoint routing"
    elif not comparison["finite_gate_pass"]:
        primary = "nonfinite parameter-path row"
        veto = "nonfinite gate"
        next_action = "repair nonfinite adjoint row before interpretation"
    else:
        primary = comparison["interpretation"]
        veto = "none"
        next_action = _next_action(comparison["interpretation"])
    return [
        {
            "decision": decision,
            "primary_criterion_status": primary,
            "veto_diagnostic_status": veto,
            "main_uncertainty": "single row/time adjacent-boundary probe; no correctness claim",
            "next_justified_action": next_action,
            "not_concluded": "correctness of either implementation, production readiness, analytic gradient correctness",
        }
    ]


def _next_action(interpretation: str) -> str:
    if interpretation.startswith("adjacent_boundary_residual_reconstructed_by_"):
        return "build a minimal patch/control for the reconstructed adjoint-routing field"
    if interpretation == "adjacent_boundary_points_to_state_adjoint_identity_break":
        return "inspect same-tape state-adjoint identity around transport/custom-gradient embedding"
    return "move the cut-set upstream and add a narrower state-adjoint identity probe"


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
        "direct_theta_gate",
        "filterflow_probe",
        "bayesfilter_probe",
        "comparison",
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
        "filterflow_float64_row_173_adjacent_boundary_filterflow_blocker",
        "filterflow_float64_row_173_adjacent_boundary_bayesfilter_blocker",
        "filterflow_float64_row_173_adjacent_boundary_blocked_by_comparator_drift",
        "filterflow_float64_row_173_adjacent_boundary_blocked",
        "filterflow_float64_row_173_adjacent_boundary_value_veto",
        "filterflow_float64_row_173_adjacent_boundary_nonfinite_veto",
        "filterflow_float64_row_173_adjacent_boundary_reconstructed",
        "filterflow_float64_row_173_adjacent_boundary_state_adjoint_identity",
        "filterflow_float64_row_173_adjacent_boundary_unresolved",
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
    comparison = payload["comparison"]
    if comparison.get("status") == "compared":
        if not comparison["value_gate_pass"] and "value_veto" not in payload["decision"]:
            raise ValueError("value gate failed without value-veto decision")
        if not comparison["finite_gate_pass"] and "nonfinite_veto" not in payload["decision"]:
            raise ValueError("finite gate failed without nonfinite-veto decision")


def _validate_cpu(manifest: dict[str, Any], label: str) -> None:
    if manifest.get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: pre-import CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("cuda_visible_devices") != "-1":
        raise ValueError(f"{label}: CUDA_VISIBLE_DEVICES is not -1")
    if manifest.get("gpu_devices_visible") != []:
        raise ValueError(f"{label}: GPU devices visible {manifest.get('gpu_devices_visible')}")


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Result: Row 173 Adjacent-Boundary Gradient Probe",
        "",
        "## Decision",
        "",
        f"`{payload['decision']}`",
        "",
        "## Interpretation",
        "",
        f"`{payload['comparison'].get('interpretation')}`",
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
            "## Gates",
            "",
            _json_block(
                {
                    "direct_theta_gate": payload["direct_theta_gate"],
                    "scalar_delta": payload["comparison"].get("scalar_delta"),
                    "value_gate_pass": payload["comparison"].get("value_gate_pass"),
                    "finite_gate_pass": payload["comparison"].get("finite_gate_pass"),
                }
            ),
            "",
            "## Best Parameter-Path Residual Match",
            "",
            _json_block(payload["comparison"].get("best_parameter_path_residual_match")),
            "",
            "## Adjoint Decomposition Summary",
            "",
            _json_block(payload["comparison"].get("adjoint_decomposition_summary")),
            "",
            "## Parameter Path Rows",
            "",
            _json_block(payload["comparison"].get("parameter_path_rows")),
            "",
            "## Boundary Rows",
            "",
            _json_block(payload["comparison"].get("boundary_rows")),
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


def _flatten(value: Any) -> list[float]:
    values: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            values.append(float(item))

    visit(value)
    return values


def _finite_vector(value: Any) -> bool:
    return all(tf.math.is_finite(tf.constant(item, dtype=DTYPE)).numpy() for item in _flatten(value))


def _digest_payload(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload, sort_keys=True, default=str))
    clone.pop("created_at_utc", None)
    clone.pop("reproducibility_digest", None)
    return stable_digest(clone)


def _non_implications() -> list[str]:
    return direct_theta._non_implications() + [
        "This adjacent-boundary probe is a difference audit only.",
        "No claim is made that either implementation is mathematically correct.",
        "No global smoothness-gradient correctness is concluded.",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
