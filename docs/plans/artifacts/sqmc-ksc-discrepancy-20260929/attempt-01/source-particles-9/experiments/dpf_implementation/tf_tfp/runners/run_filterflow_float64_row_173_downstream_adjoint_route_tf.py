"""Classify the row-173 downstream adjoint route from reviewed evidence."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-dpf-mpl")
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import time
from pathlib import Path
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_1d_to_smoothness_ladder_tf as continuation,
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
    "bayesfilter-dpf-filterflow-float64-row-173-downstream-adjoint-route-plan-2026-06-05.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-downstream-adjoint-route-result-2026-06-05.md"
)
JSON_PATH = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_downstream_adjoint_route_2026-06-05.json"
)
REPORT_PATH = (
    REPORT_DIR / "dpf-filterflow-float64-row-173-downstream-adjoint-route-2026-06-05.md"
)
REVIEW_LOOP_PATH = (
    "docs/plans/"
    "bayesfilter-dpf-filterflow-float64-row-173-downstream-adjoint-route-review-loop-2026-06-05.md"
)
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_MARKER_PATH = FILTERFLOW_PATH / FILTERFLOW_BRANCH_MARKER

DIRECT_THETA_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_direct_theta_hypothesis_2026-06-04.json"
)
ADJACENT_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_adjacent_boundary_2026-06-04.json"
)
POST_UPDATE_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_post_update_route_hypotheses_2026-06-04.json"
)
OFFICIAL_TOPOLOGY_JSON = (
    OUTPUT_DIR / "dpf_filterflow_float64_row_173_official_proposal_topology_2026-06-05.json"
)

FINGERPRINT_KEYS = (
    "head_commit",
    "status_short",
    "status_branch",
    "diff_digest",
    "python_version",
    "package_manifest_digest",
)
GRADIENT_TOLERANCE = 2e-4


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
    current_fingerprint = continuation._filterflow_fingerprint()
    artifacts = _load_artifacts()
    artifact_digests = {
        name: _file_digest(path) for name, path in _artifact_paths().items()
    }
    artifact_vetoes = _artifact_vetoes(artifacts, current_fingerprint)
    comparison = _classify_route(artifacts, artifact_vetoes)
    final_fingerprint = continuation._filterflow_fingerprint()
    comparator_drift = _fingerprints_drifted(current_fingerprint, final_fingerprint)
    veto_status = _veto_status(artifact_vetoes, comparator_drift)
    decision = _decision(comparison["classification"], veto_status)
    return {
        "created_at_utc": utc_now(),
        "question": "row_173_downstream_adjoint_route_difference_audit",
        "decision": decision,
        "hypothesis_classification": comparison["classification"],
        "hypothesis_reason": comparison["classification_reason"],
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "review_loop_path": REVIEW_LOOP_PATH,
        "json_path": str(JSON_PATH),
        "report_path": str(REPORT_PATH),
        "reference_policy": reference_policy(),
        "filterflow_status": reference_status,
        "filterflow_fingerprint_initial": current_fingerprint,
        "filterflow_fingerprint_final": final_fingerprint,
        "comparator_drift": comparator_drift,
        "required_input_artifacts": _input_manifest(artifacts, artifact_digests),
        "veto_status_table": veto_status,
        "route_comparison": comparison,
        "path_boundary_manifest": continuation._path_boundary_manifest(),
        "evidence_contract": _evidence_contract(),
        "model_contract": _model_contract(artifacts),
        "decision_table": _decision_table(decision, comparison, veto_status),
        "non_implications": _non_implications(),
        "run_manifest": {
            **environment_manifest(
                command=(
                    "CUDA_VISIBLE_DEVICES=-1 python -m "
                    "experiments.dpf_implementation.tf_tfp.runners."
                    "run_filterflow_float64_row_173_downstream_adjoint_route_tf"
                ),
                pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
            ),
            "plan_path": PLAN_PATH,
            "result_path": RESULT_PATH,
            "json_path": str(JSON_PATH),
            "report_path": str(REPORT_PATH),
            "required_artifact_paths": {
                name: str(path) for name, path in _artifact_paths().items()
            },
        },
    }


def _artifact_paths() -> dict[str, Path]:
    return {
        "official_proposal_topology": OFFICIAL_TOPOLOGY_JSON,
        "post_update_route": POST_UPDATE_JSON,
        "adjacent_boundary": ADJACENT_JSON,
        "direct_theta": DIRECT_THETA_JSON,
    }


def _load_artifacts() -> dict[str, dict[str, Any]]:
    artifacts = {}
    for name, path in _artifact_paths().items():
        artifacts[name] = load_json(path)
    return artifacts


def _artifact_vetoes(
    artifacts: dict[str, dict[str, Any]],
    current_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    fingerprint_rows = _fingerprint_rows(artifacts, current_fingerprint)
    cpu_rows = {
        name: _cpu_manifest_status(artifact.get("run_manifest", {}))
        for name, artifact in artifacts.items()
    }
    scalar_rows = _scalar_gate_rows(artifacts)
    finite_rows = _finite_gate_rows(artifacts)
    path_rows = {
        name: artifact.get("path_boundary_manifest", {})
        for name, artifact in artifacts.items()
    }
    path_clean = all(
        not any(bool(value) for value in row.values())
        for row in path_rows.values()
    )
    return {
        "fingerprint_rows": fingerprint_rows,
        "fingerprints_match": all(row["matches_reference"] for row in fingerprint_rows.values()),
        "cpu_rows": cpu_rows,
        "cpu_only_pass": all(row["pass"] for row in cpu_rows.values()),
        "scalar_rows": scalar_rows,
        "scalar_gates_pass": all(row["pass"] for row in scalar_rows.values()),
        "finite_rows": finite_rows,
        "finite_gates_pass": all(row["pass"] for row in finite_rows.values()),
        "path_boundary_rows": path_rows,
        "path_boundary_clean": path_clean,
        "all_required_loaded": set(artifacts) == set(_artifact_paths()),
    }


def _fingerprint_rows(
    artifacts: dict[str, dict[str, Any]],
    current_fingerprint: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows = {}
    reference = artifacts["official_proposal_topology"]["filterflow_fingerprint_initial"]
    for name, artifact in artifacts.items():
        artifact_fingerprint = artifact.get("filterflow_fingerprint_initial", {})
        rows[name] = {
            "matches_reference": not _fingerprints_drifted(reference, artifact_fingerprint),
            "matches_current": not _fingerprints_drifted(current_fingerprint, artifact_fingerprint),
            "fingerprint_excerpt": {
                key: artifact_fingerprint.get(key) for key in FINGERPRINT_KEYS
            },
        }
    rows["current_checkout"] = {
        "matches_reference": not _fingerprints_drifted(reference, current_fingerprint),
        "matches_current": True,
        "fingerprint_excerpt": {
            key: current_fingerprint.get(key) for key in FINGERPRINT_KEYS
        },
    }
    return rows


def _cpu_manifest_status(manifest: dict[str, Any]) -> dict[str, Any]:
    visible = manifest.get("cuda_visible_devices")
    pre_import = manifest.get("pre_import_cuda_visible_devices")
    gpu_devices = manifest.get("gpu_devices_visible")
    return {
        "cuda_visible_devices": visible,
        "pre_import_cuda_visible_devices": pre_import,
        "gpu_devices_visible": gpu_devices,
        "pass": visible == "-1" and pre_import == "-1" and gpu_devices == [],
    }


def _scalar_gate_rows(artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    official_veto = artifacts["official_proposal_topology"].get("veto_status_table", {})
    post_veto = artifacts["post_update_route"].get("veto_status_table", {})
    adjacent = artifacts["adjacent_boundary"].get("comparison", {})
    direct = artifacts["direct_theta"].get("comparison", {})
    direct_value_vetoes = direct.get("value_vetoes", [])
    return {
        "official_proposal_topology": {
            "pass": bool(official_veto.get("scalar_value_gate_pass")),
            "source": "veto_status_table.scalar_value_gate_pass",
        },
        "post_update_route": {
            "pass": bool(post_veto.get("scalar_gate_pass")),
            "source": "veto_status_table.scalar_gate_pass",
        },
        "adjacent_boundary": {
            "pass": bool(adjacent.get("value_gate_pass")),
            "scalar_delta": adjacent.get("scalar_delta"),
            "source": "comparison.value_gate_pass",
        },
        "direct_theta": {
            "pass": len(direct_value_vetoes) == 0,
            "value_vetoes": direct_value_vetoes,
            "source": "comparison.value_vetoes",
        },
    }


def _finite_gate_rows(artifacts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    official_veto = artifacts["official_proposal_topology"].get("veto_status_table", {})
    post_veto = artifacts["post_update_route"].get("veto_status_table", {})
    adjacent = artifacts["adjacent_boundary"].get("comparison", {})
    direct = artifacts["direct_theta"].get("comparison", {})
    direct_nonfinite = direct.get("gradient_nonfinite_rows", [])
    return {
        "official_proposal_topology": {
            "pass": bool(
                official_veto.get("all_forward_finite")
                and official_veto.get("all_adjoints_finite")
                and official_veto.get("all_local_gradients_finite")
                and official_veto.get("all_variant_gradients_finite")
            ),
            "source": "official finite veto fields",
        },
        "post_update_route": {
            "pass": bool(post_veto.get("gradient_rows_finite")),
            "source": "veto_status_table.gradient_rows_finite",
        },
        "adjacent_boundary": {
            "pass": bool(adjacent.get("finite_gate_pass")),
            "source": "comparison.finite_gate_pass",
        },
        "direct_theta": {
            "pass": len(direct_nonfinite) == 0,
            "gradient_nonfinite_rows": direct_nonfinite,
            "source": "comparison.gradient_nonfinite_rows",
        },
    }


def _classify_route(
    artifacts: dict[str, dict[str, Any]],
    vetoes: dict[str, Any],
) -> dict[str, Any]:
    official = artifacts["official_proposal_topology"]["source_comparison"]
    post = artifacts["post_update_route"]["post_update_route"]
    adjacent = artifacts["adjacent_boundary"]["comparison"]
    direct = artifacts["direct_theta"]["comparison"]
    direct_variant = official["variant_comparisons"]["direct_sampled_distribution"]
    official_variant_name = official["best_primary_match_variant"]
    official_variant = official["variant_comparisons"][official_variant_name]
    direct_row = direct_variant["rows"][0]
    official_row = official_variant["rows"][0]
    adjacent_decomp = adjacent["adjoint_decomposition_summary"]
    route_evidence = {
        "direct_total_gradient_delta": direct_variant["total_gradient_delta"],
        "direct_max_abs_total_gradient_delta": direct_variant["max_abs_total_gradient_delta"],
        "official_local_match_variant": official_variant_name,
        "official_local_match_total_gradient_delta": official_variant["total_gradient_delta"],
        "official_local_match_max_abs_total_gradient_delta": (
            official_variant["max_abs_total_gradient_delta"]
        ),
        "best_variant_by_gradient_delta": official["best_variant_by_gradient_delta"],
        "best_variant_max_abs_total_gradient_delta": official[
            "best_variant_max_abs_total_gradient_delta"
        ],
        "primary_matching_variants": official["primary_matching_variants"],
        "material_global_reduction": official["material_global_reduction"],
        "global_gap_remaining": official["global_gap_remaining"],
        "direct_first_adjoint_delta_node": direct_row["first_adjoint_delta_node"],
        "direct_max_adjoint_delta_node": direct_row["max_adjoint_delta_node"],
        "official_match_first_adjoint_delta_node": official_row["first_adjoint_delta_node"],
        "official_match_max_adjoint_delta_node": official_row["max_adjoint_delta_node"],
        "post_update_observed_residual": post["observed_residual"],
        "post_update_delta": post["post_update_delta"],
        "post_update_reconstruction_residual": post["post_update_reconstruction_residual"],
        "component_sum_reconstruction_residual": (
            post["component_sum_reconstruction_residual"]
        ),
        "direct_theta_decision": artifacts["direct_theta"]["decision"],
        "direct_theta_best_increment_delta": direct["best_increment_row"][
            "gradient_delta"
        ],
        "direct_theta_best_core_delta": direct["best_core_row"]["gradient_delta"],
        "adjacent_best_parameter_path_field": adjacent[
            "best_parameter_path_residual_match"
        ]["field"],
        "adjacent_best_parameter_path_delta": adjacent[
            "best_parameter_path_residual_match"
        ]["delta"],
        "adjacent_decomposition_summary": adjacent_decomp,
        "official_local_match_worsening_factor": _safe_ratio(
            official_variant["max_abs_total_gradient_delta"],
            direct_variant["max_abs_total_gradient_delta"],
        ),
    }
    if not _all_vetoes_clear(vetoes):
        classification = "h1_artifact_or_reference_drift"
        reason = "required artifact, comparator, CPU, scalar, finite, or boundary gate failed"
    elif official_variant["max_abs_total_gradient_delta"] > direct_variant[
        "max_abs_total_gradient_delta"
    ] + GRADIENT_TOLERANCE:
        classification = "h2_local_official_match_not_global_route"
        reason = (
            "official local proposal-likelihood VJP match is value-valid but worsens "
            "the full gradient gap"
        )
    elif post["post_update_pass"] and post["max_abs_post_update_reconstruction_residual"] <= GRADIENT_TOLERANCE:
        classification = "h3_post_update_parameter_path_residual"
        reason = "post_update_log_likelihoods parameter path exactly reconstructs the residual"
    elif _adjacent_route_material(adjacent_decomp):
        classification = "h4_downstream_state_update_identity_route"
        reason = "adjacent-boundary state/update identity adjoints carry material route deltas"
    else:
        classification = "h5_unresolved_after_synthesis"
        reason = "reviewed evidence remains finite and value-valid but does not isolate a route"
    return {
        "status": "compared",
        "classification": classification,
        "classification_reason": reason,
        "route_evidence": route_evidence,
        "classification_order_note": (
            "h2 is selected before h3/h4 when the locally matched official "
            "proposal topology worsens the global route, because that directly "
            "answers why the local fix does not close the full gradient gap."
        ),
    }


def _all_vetoes_clear(vetoes: dict[str, Any]) -> bool:
    return bool(
        vetoes["all_required_loaded"]
        and vetoes["fingerprints_match"]
        and vetoes["cpu_only_pass"]
        and vetoes["scalar_gates_pass"]
        and vetoes["finite_gates_pass"]
        and vetoes["path_boundary_clean"]
    )


def _adjacent_route_material(decomposition: dict[str, Any]) -> bool:
    material_keys = (
        "same_tape_post_particles_vjp",
        "same_tape_transport_matrix_vjp",
        "same_tape_full_recorded_state_residual",
        "same_tape_identity_residual",
    )
    return any(
        float(decomposition[key]["max_abs_delta"]) > GRADIENT_TOLERANCE
        for key in material_keys
    )


def _veto_status(artifact_vetoes: dict[str, Any], comparator_drift: bool) -> dict[str, Any]:
    all_vetoes_clear = _all_vetoes_clear(artifact_vetoes) and not comparator_drift
    return {
        "all_vetoes_clear": all_vetoes_clear,
        "all_required_loaded": artifact_vetoes["all_required_loaded"],
        "comparator_drift_during_run": comparator_drift,
        "fingerprints_match": artifact_vetoes["fingerprints_match"],
        "cpu_only_pass": artifact_vetoes["cpu_only_pass"],
        "scalar_gates_pass": artifact_vetoes["scalar_gates_pass"],
        "finite_gates_pass": artifact_vetoes["finite_gates_pass"],
        "path_boundary_clean": artifact_vetoes["path_boundary_clean"],
        "artifact_veto_details": artifact_vetoes,
    }


def _decision(classification: str, veto_status: dict[str, Any]) -> str:
    if not veto_status["all_vetoes_clear"]:
        return "filterflow_float64_row_173_downstream_adjoint_route_blocked_or_vetoed"
    mapping = {
        "h1_artifact_or_reference_drift": (
            "filterflow_float64_row_173_downstream_adjoint_route_artifact_or_reference_drift"
        ),
        "h2_local_official_match_not_global_route": (
            "filterflow_float64_row_173_downstream_adjoint_route_local_match_not_global"
        ),
        "h3_post_update_parameter_path_residual": (
            "filterflow_float64_row_173_downstream_adjoint_route_post_update_parameter_path"
        ),
        "h4_downstream_state_update_identity_route": (
            "filterflow_float64_row_173_downstream_adjoint_route_state_update_identity"
        ),
        "h5_unresolved_after_synthesis": (
            "filterflow_float64_row_173_downstream_adjoint_route_unresolved"
        ),
    }
    return mapping[classification]


def _input_manifest(
    artifacts: dict[str, dict[str, Any]],
    digests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = {}
    for name, artifact in artifacts.items():
        rows[name] = {
            "path": str(_artifact_paths()[name]),
            "digest": digests[name],
            "decision": artifact.get("decision"),
            "result_path": artifact.get("result_path"),
            "plan_path": artifact.get("plan_path"),
            "reproducibility_digest": artifact.get("reproducibility_digest"),
        }
    return rows


def _model_contract(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    official = artifacts["official_proposal_topology"]
    return {
        "row_index": 173,
        "target_time_index": 93,
        "probe_time_index": 43,
        "comparator": "local executable float64 FilterFlow checkout",
        "official_model_contract": official.get("model_contract", {}),
        "official_canonical_input_manifest": official.get("canonical_input_manifest", {}),
    }


def _evidence_contract() -> dict[str, Any]:
    return {
        "question": (
            "Which downstream adjoint route explains why the full row-173 "
            "BayesFilter gradient still differs after local official proposal "
            "likelihood VJPs are identified?"
        ),
        "comparator": "local executable float64 FilterFlow reference",
        "primary_criterion": "classify downstream route from accepted JSON artifacts",
        "vetoes": [
            "missing or unparsable required JSON",
            "comparator fingerprint mismatch",
            "CPU-only manifest failure",
            "scalar gate failure",
            "finite gate failure",
            "path-boundary contamination",
        ],
        "explanatory_only": [
            "transport residuals",
            "finite gradients",
            "variant ranking",
            "adjoint node magnitudes",
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
            "Primary criterion status": comparison["classification"],
            "Veto diagnostic status": json.dumps(
                {
                    key: value
                    for key, value in veto_status.items()
                    if key != "artifact_veto_details"
                },
                sort_keys=True,
            ),
            "Main uncertainty": "single-row synthesis of reviewed artifacts; no global claim",
            "Next justified action": _next_action(comparison["classification"]),
            "Not concluded": "correctness, posterior correctness, production readiness, global agreement",
        }
    ]


def _next_action(classification: str) -> str:
    if classification == "h2_local_official_match_not_global_route":
        return (
            "test the downstream state/update identity route directly before "
            "adopting the local official proposal topology as a BayesFilter fix"
        )
    if classification == "h3_post_update_parameter_path_residual":
        return "instrument post_update_log_likelihoods parameter-path adjoints"
    if classification == "h4_downstream_state_update_identity_route":
        return "build a minimal post-state identity-route patch/control"
    if classification == "h1_artifact_or_reference_drift":
        return "refresh or rerun required reviewed input artifacts"
    return "design one smaller discriminating route probe"


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "created_at_utc",
        "question",
        "decision",
        "hypothesis_classification",
        "hypothesis_reason",
        "required_input_artifacts",
        "veto_status_table",
        "route_comparison",
        "filterflow_fingerprint_initial",
        "filterflow_fingerprint_final",
        "run_manifest",
        "decision_table",
        "non_implications",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"missing payload keys: {sorted(missing)}")
    if payload["run_manifest"].get("pre_import_cuda_visible_devices") != "-1":
        raise ValueError("CPU-only pre-import CUDA manifest failed")
    if payload["run_manifest"].get("cuda_visible_devices") != "-1":
        raise ValueError("CPU-only CUDA manifest failed")
    if payload["run_manifest"].get("gpu_devices_visible") != []:
        raise ValueError("GPU devices visible in CPU-only run")
    if payload["decision"] == "filterflow_float64_row_173_downstream_adjoint_route_blocked_or_vetoed":
        return
    if not payload["veto_status_table"]["all_vetoes_clear"]:
        raise ValueError("non-blocked decision with uncleared vetoes")
    if payload["route_comparison"]["status"] != "compared":
        raise ValueError("route comparison did not complete")


def _markdown(payload: dict[str, Any]) -> str:
    route = payload["route_comparison"]["route_evidence"]
    return "\n".join(
        [
            "# Result: Row 173 Downstream Adjoint-Route Classifier",
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
            "## Route Evidence",
            "",
            _json_block(route),
            "",
            "## Veto Status",
            "",
            _json_block(payload["veto_status_table"]),
            "",
            "## Required Input Artifacts",
            "",
            _json_block(payload["required_input_artifacts"]),
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


def _file_digest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "digest": None}
    return {
        "path": str(path),
        "exists": True,
        "digest": stable_digest(path.read_text(encoding="utf-8")),
    }


def _fingerprints_drifted(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return any(left.get(key) != right.get(key) for key in FINGERPRINT_KEYS)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


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
