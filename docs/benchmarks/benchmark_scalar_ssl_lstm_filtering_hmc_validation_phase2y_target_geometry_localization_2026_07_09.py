"""Phase 2Y scalar SSL-LSTM target geometry localization diagnostic.

This diagnostic localizes the Phase 2W/2X importance-reference ESS failures by
replaying target values, scores, proposal log densities, affine orientation
checks, and MAP-local quadratic residuals at predeclared anchors and rays in
``u_new``.  It does not run HMC, does not create a new reference, and does not
claim posterior correctness, HMC readiness, convergence, zero divergences,
GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_"
    "target_geometry_localization_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2y_target_geometry_localization.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md"
)
DEFAULT_PHASE2S_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2T_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2U_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2V_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2W_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json"
)
DEFAULT_PHASE2X_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md"
)
PHASE2U_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
PHASE2W_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py"
)
PHASE2X_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py"
)
TOP_ANCHOR_COUNT = 8
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
NONCLAIMS = (
    "Phase 2Y target-geometry localization diagnostic only",
    "not a new valid importance reference",
    "not HMC-vs-reference agreement evidence",
    "not an HMC run",
    "not HMC readiness evidence",
    "not HMC convergence evidence",
    "not posterior correctness evidence",
    "not a zero-divergence claim when native divergence is unavailable",
    "not sampler superiority evidence",
    "not statistically supported ranking evidence",
    "not GPU/XLA production-readiness evidence",
    "not default-readiness evidence",
    "not Zhao-Cui source-faithfulness evidence",
)


def load_module(module_path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase2u = load_module(
    PHASE2U_MODULE_PATH,
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_for_phase2y",
)
phase2w = load_module(
    PHASE2W_MODULE_PATH,
    "scalar_ssl_lstm_filtering_hmc_validation_phase2w_for_phase2y",
)
phase2x = load_module(
    PHASE2X_MODULE_PATH,
    "scalar_ssl_lstm_filtering_hmc_validation_phase2x_for_phase2y",
)


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2y_target_geometry_localization(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_phase2y_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        phase2w_payload,
        phase2x_payload,
    )
    vetoes = list(precondition.get("vetoes", ()))
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    anchor_data: Mapping[str, Any] = {"built": False, "vetoes": ("phase2y_not_run",)}
    anchor_eval: Mapping[str, Any] = {"computed": False, "vetoes": ("phase2y_not_run",)}
    ray_eval: Mapping[str, Any] = {"computed": False, "vetoes": ("phase2y_not_run",)}
    orientation_eval: Mapping[str, Any] = {"computed": False, "vetoes": ("phase2y_not_run",)}
    proposal_replay: Mapping[str, Any] = {"computed": False, "vetoes": ("phase2y_not_run",)}
    hypothesis_assessment: Mapping[str, Any] = {"computed": False}

    if not vetoes:
        adapter, adapter_audit = phase2u.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2y_adapter_not_built")

    if adapter is not None and not adapter_audit.get("vetoes"):
        anchor_data = build_anchor_set(
            phase2w_payload,
            phase2x_payload,
            top_count=TOP_ANCHOR_COUNT,
        )
        vetoes.extend(anchor_data.get("vetoes", ()))
        if not anchor_data.get("vetoes"):
            anchor_eval = evaluate_anchors(adapter, anchor_data, phase2s_payload)
            vetoes.extend(anchor_eval.get("vetoes", ()))
            ray_eval = evaluate_ray_profiles(adapter, anchor_data, phase2s_payload)
            vetoes.extend(ray_eval.get("vetoes", ()))
            orientation_eval = evaluate_orientation_diagnostic(
                adapter,
                anchor_data,
                phase2s_payload,
            )
            vetoes.extend(orientation_eval.get("vetoes", ()))
            proposal_replay = replay_proposal_log_densities(
                anchor_data,
                phase2w_payload,
                phase2x_payload,
            )
            vetoes.extend(proposal_replay.get("vetoes", ()))
            hypothesis_assessment = assess_hypotheses(
                phase2s_payload,
                anchor_data,
                anchor_eval,
                ray_eval,
                orientation_eval,
                proposal_replay,
            )
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    passed = bool(not unique_vetoes)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2y_target_geometry_localization",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": None if adapter is None else adapter.target_scope,
        "settings": {
            "coordinate": "phase2s_phase2u_map_local_u_new",
            "top_anchor_count_per_source": TOP_ANCHOR_COUNT,
            "alpha_grid": ALPHA_GRID,
            "quadratic_comparator": "target(center) - 0.5 * ||u_new||^2",
            "uses_hmc_moments_for_anchor_or_proposal_repair": False,
            "runs_hmc": False,
            "cpu_hidden": True,
            "use_xla": False,
        },
        "source_artifacts": {
            "phase2s_json": str(DEFAULT_PHASE2S_PATH.relative_to(ROOT)),
            "phase2t_json": str(DEFAULT_PHASE2T_PATH.relative_to(ROOT)),
            "phase2u_json": str(DEFAULT_PHASE2U_PATH.relative_to(ROOT)),
            "phase2v_json": str(DEFAULT_PHASE2V_PATH.relative_to(ROOT)),
            "phase2w_json": str(DEFAULT_PHASE2W_PATH.relative_to(ROOT)),
            "phase2x_json": str(DEFAULT_PHASE2X_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "anchors": anchor_data,
        "anchor_evaluation": anchor_eval,
        "ray_profiles": ray_eval,
        "orientation_diagnostic": orientation_eval,
        "proposal_log_density_replay": proposal_replay,
        "hypothesis_assessment": hypothesis_assessment,
        "review_record": {
            "claude_review_attempted": True,
            "claude_status": "blocked_by_approval_layer_external_data_transfer_risk",
            "reviewer": "Codex local substitute review",
            "review_strength": "weaker_than_full_claude_material_review",
            "verdict": "AGREE_FOR_DIAGNOSTIC_RUNTIME",
            "review_bundle": (
                "docs/reviews/"
                "bayesfilter-scalar-filtering-hmc-validation-phase2y-hypothesis-plan-review-bundle-2026-07-09.md"
            ),
        },
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2y_target_geometry_localization_passed": passed,
            "vetoes": unique_vetoes,
            "artifact_bug_indicated": (
                bool((orientation_eval.get("summary") or {}).get("artifact_bug_indicated"))
                or bool((proposal_replay.get("summary") or {}).get("artifact_bug_indicated"))
            ),
            "proposal_family_mismatch_indicated": bool(
                (hypothesis_assessment.get("summary") or {}).get(
                    "proposal_family_mismatch_indicated"
                )
            ),
            "zero_divergence_claim_made": False,
            "viable_for_phase3_gpu_xla_subplan": False,
            "next_justified_action": (
                "write Phase 2Y result and draft/review proposal redesign or local-reference-abandonment subplan"
                if passed
                else "write Phase 2Y target/artifact blocker result before any further reference attempt"
            ),
        },
        "metric_roles": {
            "phase2y_target_geometry_localization_passed": "primary_phase2y_artifact_validity_pass_fail",
            "input_artifacts_valid": "hard_veto_evidence",
            "target_value_and_score_finiteness": "hard_veto_evidence",
            "orientation_consistency": "bug_localization_diagnostic",
            "proposal_log_density_replay": "bug_localization_diagnostic",
            "top_weight_anchor_norms": "explanatory_only",
            "quadratic_residuals": "explanatory_only",
            "ray_profile_asymmetry": "explanatory_only",
            "score_directional_components": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "reference_validity": "not assessed; Phase 2Y does not build a reference",
            "hmc_reference_agreement": "not assessed",
            "native_divergence": "not assessed; no HMC run",
            "zero_divergence_claim": "not made",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "anchor norms, target values, scores, proposal log densities, "
                "ray profiles, and quadratic residuals"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "reviewed proposal redesign, transport/local mixture, or local-reference-abandonment subplan"
                if passed
                else "bug or target-validity repair before any new reference attempt"
            ),
        },
        "decision_table": {
            "decision": "Phase 2Y target geometry localization diagnostic",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "The diagnostic can localize proposal mismatch or artifact bugs, "
                "but it is not a posterior oracle and cannot certify HMC."
            ),
            "next_justified_action": (
                "draft/review proposal redesign or local-reference-abandonment subplan"
                if passed
                else "write blocker before more reference attempts"
            ),
            "what_is_not_being_concluded": (
                "No valid reference, posterior correctness, HMC readiness, convergence, "
                "zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA "
                "readiness, default readiness, or Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 300 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": "N/A; replays deterministic saved proposal artifacts only",
            "wall_time_seconds": float(time.perf_counter() - start),
            "output_artifacts": (
                str(DEFAULT_JSON_PATH.relative_to(ROOT)),
                str(DEFAULT_MARKDOWN_PATH.relative_to(ROOT)),
            ),
            "plan_file": PLAN_PATH,
            "subplan_file": SUBPLAN_PATH,
            "result_file": RESULT_PATH,
        },
        "post_run_red_team": {
            "strongest_alternative_explanation": (
                "Finite ray diagnostics may still miss target regions not represented "
                "by the Phase 2W/2X failed proposals."
            ),
            "what_would_overturn": (
                "A replay showing proposal log-density or affine orientation mismatch, "
                "or a later valid reference with adequate ESS under a reviewed proposal."
            ),
            "weakest_evidence": (
                "Diagnostics are anchored to two failed proposal artifacts and fixed rays, "
                "not exhaustive posterior exploration."
            ),
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_phase2y_handoff(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    expected_schemas = {
        "phase2s": "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1",
        "phase2t": "scalar_ssl_lstm.filtering_hmc_validation_phase2t_map_local_reference_handoff.v1",
        "phase2u": "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1",
        "phase2v": "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1",
        "phase2w": "scalar_ssl_lstm.filtering_hmc_validation_phase2w_importance_reference_agreement.v1",
        "phase2x": "scalar_ssl_lstm.filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.v1",
    }
    payloads = {
        "phase2s": phase2s_payload,
        "phase2t": phase2t_payload,
        "phase2u": phase2u_payload,
        "phase2v": phase2v_payload,
        "phase2w": phase2w_payload,
        "phase2x": phase2x_payload,
    }
    for name, expected in expected_schemas.items():
        if payloads[name].get("schema_version") != expected:
            vetoes.append(f"{name}_schema_mismatch")
    if phase2s_payload.get("decision", {}).get("phase2s_geometry_centering_repair_passed") is not True:
        vetoes.append("phase2s_not_passed")
    if phase2t_payload.get("decision", {}).get("phase2t_map_local_reference_handoff_passed") is not True:
        vetoes.append("phase2t_not_passed")
    if phase2u_payload.get("decision", {}).get("phase2u_retuned_map_local_hmc_screen_passed") is not True:
        vetoes.append("phase2u_not_passed")
    if phase2v_payload.get("decision", {}).get("phase2v_longer_selected_map_local_screen_passed") is not True:
        vetoes.append("phase2v_not_passed")
    expected_reference_vetoes = {
        "reference_ess_below_threshold",
        "reference_ess_ratio_below_threshold",
    }
    for name, payload in (("phase2w", phase2w_payload), ("phase2x", phase2x_payload)):
        decision = payload.get("decision", {})
        veto_set = set(str(item) for item in decision.get("vetoes", ()))
        if decision.get(f"{name}_importance_reference_agreement_passed") is True:
            vetoes.append(f"{name}_unexpectedly_passed_reference_agreement")
        if name == "phase2x" and decision.get("phase2x_shifted_mixture_reference_repair_passed") is True:
            vetoes.append("phase2x_unexpectedly_passed_reference_repair")
        if decision.get("reference_valid") is not False:
            vetoes.append(f"{name}_reference_validity_boundary_not_failed")
        if veto_set != expected_reference_vetoes:
            vetoes.append(f"{name}_failure_not_limited_to_ess_vetoes")
        if payload.get("hmc_reference_agreement", {}).get("evaluated") is not False:
            vetoes.append(f"{name}_hmc_reference_agreement_was_evaluated")
        reference = payload.get("importance_reference", {})
        target = np.asarray(reference.get("target_log_prob"), dtype=float)
        proposal_log_prob = np.asarray(reference.get("proposal_log_prob"), dtype=float)
        samples = np.asarray(payload.get("proposal", {}).get("samples"), dtype=float)
        if target.ndim != 1 or proposal_log_prob.shape != target.shape:
            vetoes.append(f"{name}_target_or_proposal_log_prob_shape_invalid")
        if samples.ndim != 2 or samples.shape[1] != 4 or samples.shape[0] != target.shape[0]:
            vetoes.append(f"{name}_proposal_samples_shape_invalid")
        if not np.all(np.isfinite(target)):
            vetoes.append(f"{name}_target_log_prob_nonfinite")
        if not np.all(np.isfinite(proposal_log_prob)):
            vetoes.append(f"{name}_proposal_log_prob_nonfinite")
        if not np.all(np.isfinite(samples)):
            vetoes.append(f"{name}_proposal_samples_nonfinite")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2w_decision": phase2w_payload.get("decision", {}),
        "phase2x_decision": phase2x_payload.get("decision", {}),
    }


def build_anchor_set(
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    *,
    top_count: int,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    rows: list[dict[str, Any]] = [
        {
            "anchor_id": "center",
            "source": "phase2s_center",
            "source_index": None,
            "u_new": np.zeros(4, dtype=float),
            "source_normalized_weight": None,
            "source_target_log_prob": None,
            "source_proposal_log_prob": None,
            "source_log_weight": None,
            "source_component": "center",
            "relation": "center",
        }
    ]
    for label, payload in (("phase2w", phase2w_payload), ("phase2x", phase2x_payload)):
        try:
            rows.extend(top_weight_rows(label, payload, top_count=top_count))
        except ValueError as exc:
            vetoes.append(f"{label}_top_weight_rows_invalid_{type(exc).__name__}")
    unique_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int | None, str]] = set()
    for row in rows:
        key = (str(row["source"]), row["source_index"], str(row["relation"]))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_rows.append(row)
    if len(unique_rows) < 1 + 2 * top_count:
        vetoes.append("insufficient_anchor_rows")
    u = np.asarray([row["u_new"] for row in unique_rows], dtype=float)
    if u.ndim != 2 or u.shape[1] != 4 or not np.all(np.isfinite(u)):
        vetoes.append("anchor_u_new_invalid")
    return json_ready(
        {
            "built": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "top_count_per_source": top_count,
            "anchor_count": len(unique_rows),
            "rows": unique_rows,
            "summary": summarize_anchor_rows(unique_rows),
        }
    )


def top_weight_rows(
    label: str,
    payload: Mapping[str, Any],
    *,
    top_count: int,
) -> list[dict[str, Any]]:
    proposal = payload.get("proposal", {})
    reference = payload.get("importance_reference", {})
    samples = np.asarray(proposal.get("samples"), dtype=float)
    target = np.asarray(reference.get("target_log_prob"), dtype=float)
    proposal_log_prob = np.asarray(reference.get("proposal_log_prob"), dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("proposal samples must have shape (n, 4)")
    if target.shape != (samples.shape[0],) or proposal_log_prob.shape != target.shape:
        raise ValueError("target/proposal log-probability arrays must match sample count")
    log_weight = target - proposal_log_prob
    shifted = np.exp(log_weight - float(np.max(log_weight)))
    normalized = shifted / float(np.sum(shifted))
    order = np.argsort(normalized)[-top_count:][::-1]
    components = proposal.get("component")
    rows: list[dict[str, Any]] = []
    for rank, index in enumerate(order):
        component = None
        if components is not None:
            component = str(components[int(index)])
        rows.append(
            {
                "anchor_id": f"{label}_top_{rank:02d}",
                "source": label,
                "source_index": int(index),
                "u_new": samples[int(index)],
                "source_rank": int(rank),
                "source_normalized_weight": float(normalized[int(index)]),
                "source_target_log_prob": float(target[int(index)]),
                "source_proposal_log_prob": float(proposal_log_prob[int(index)]),
                "source_log_weight": float(log_weight[int(index)]),
                "source_component": component,
                "relation": "top_weight",
                "antithetic_partner_index": antithetic_partner_index(label, payload, int(index)),
            }
        )
        partner = rows[-1]["antithetic_partner_index"]
        if partner is not None:
            rows.append(
                {
                    "anchor_id": f"{label}_top_{rank:02d}_partner",
                    "source": label,
                    "source_index": int(partner),
                    "u_new": samples[int(partner)],
                    "source_rank": int(rank),
                    "source_normalized_weight": float(normalized[int(partner)]),
                    "source_target_log_prob": float(target[int(partner)]),
                    "source_proposal_log_prob": float(proposal_log_prob[int(partner)]),
                    "source_log_weight": float(log_weight[int(partner)]),
                    "source_component": (
                        str(components[int(partner)]) if components is not None else None
                    ),
                    "relation": "antithetic_partner",
                    "antithetic_partner_index": int(index),
                }
            )
    return rows


def antithetic_partner_index(
    label: str,
    payload: Mapping[str, Any],
    index: int,
) -> int | None:
    proposal = payload.get("proposal", {})
    sample_count = int(proposal.get("sample_count", 0))
    if label == "phase2w":
        half = sample_count // 2
        if 0 <= index < half:
            return index + half
        if half <= index < sample_count:
            return index - half
        return None
    if label == "phase2x":
        counts = proposal.get("component_counts", {})
        standard_count = int(counts.get("standard", 0))
        shifted_count = int(counts.get("shifted", 0))
        standard_half = standard_count // 2
        shifted_half = shifted_count // 2
        if 0 <= index < standard_half:
            return index + standard_half
        if standard_half <= index < standard_count:
            return index - standard_half
        shifted_start = standard_count
        rel = index - shifted_start
        if 0 <= rel < shifted_half:
            return shifted_start + rel + shifted_half
        if shifted_half <= rel < shifted_count:
            return shifted_start + rel - shifted_half
    return None


def summarize_anchor_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    u = np.asarray([row["u_new"] for row in rows], dtype=float)
    norms = np.linalg.norm(u, axis=1)
    top_rows = [row for row in rows if row.get("relation") == "top_weight"]
    return {
        "norm_summary": finite_summary(norms),
        "max_abs_summary": finite_summary(np.max(np.abs(u), axis=1)),
        "top_weight_count": len(top_rows),
        "max_top_source_weight": (
            max(float(row["source_normalized_weight"]) for row in top_rows)
            if top_rows
            else None
        ),
        "source_counts": {
            source: sum(1 for row in rows if row.get("source") == source)
            for source in sorted({str(row.get("source")) for row in rows})
        },
    }


def evaluate_anchors(
    adapter: Any,
    anchor_data: Mapping[str, Any],
    phase2s_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    center_value = center_target_value(phase2s_payload)
    rows = []
    for anchor in anchor_data.get("rows", ()):
        u = np.asarray(anchor["u_new"], dtype=float)
        value, score = evaluate_target(adapter, u)
        norm = float(np.linalg.norm(u))
        rows.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source": anchor["source"],
                "source_index": anchor["source_index"],
                "relation": anchor["relation"],
                "source_component": anchor.get("source_component"),
                "u_new": u,
                "norm_u_new": norm,
                "max_abs_u_new": float(np.max(np.abs(u))),
                "target_log_prob": value,
                "target_delta_from_center": value - center_value,
                "score_u_new": score,
                "score_norm": float(np.linalg.norm(score)),
                "radial_score_component": radial_score_component(u, score),
                "quadratic_log_prob": quadratic_log_prob(center_value, u),
                "target_minus_quadratic": value - quadratic_log_prob(center_value, u),
            }
        )
        if not np.isfinite(value) or not np.all(np.isfinite(score)):
            vetoes.append(f"anchor_{anchor['anchor_id']}_target_or_score_nonfinite")
    return json_ready(
        {
            "computed": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "center_target_log_prob": center_value,
            "rows": rows,
            "summary": summarize_evaluation_rows(rows),
        }
    )


def evaluate_ray_profiles(
    adapter: Any,
    anchor_data: Mapping[str, Any],
    phase2s_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    center_value = center_target_value(phase2s_payload)
    profiles = []
    top_rows = [row for row in anchor_data.get("rows", ()) if row.get("relation") == "top_weight"]
    for anchor in top_rows:
        v = np.asarray(anchor["u_new"], dtype=float)
        for sign_label, sign in (("positive", 1.0), ("negative", -1.0)):
            points = []
            for alpha in ALPHA_GRID:
                u = sign * float(alpha) * v
                value, score = evaluate_target(adapter, u)
                points.append(
                    {
                        "alpha": float(alpha),
                        "u_new": u,
                        "norm_u_new": float(np.linalg.norm(u)),
                        "target_log_prob": value,
                        "target_delta_from_center": value - center_value,
                        "score_norm": float(np.linalg.norm(score)),
                        "radial_score_component": radial_score_component(u, score),
                        "quadratic_log_prob": quadratic_log_prob(center_value, u),
                        "target_minus_quadratic": value - quadratic_log_prob(center_value, u),
                    }
                )
                if not np.isfinite(value) or not np.all(np.isfinite(score)):
                    vetoes.append(f"ray_{anchor['anchor_id']}_{sign_label}_{alpha}_nonfinite")
            profiles.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "source": anchor["source"],
                    "source_index": anchor["source_index"],
                    "sign": sign_label,
                    "anchor_norm": float(np.linalg.norm(v)),
                    "points": points,
                }
            )
    return json_ready(
        {
            "computed": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "alpha_grid": ALPHA_GRID,
            "profile_count": len(profiles),
            "profiles": profiles,
            "summary": summarize_ray_profiles(profiles),
        }
    )


def evaluate_orientation_diagnostic(
    adapter: Any,
    anchor_data: Mapping[str, Any],
    phase2s_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    handoff = phase2s_payload.get("map_local_handoff", {})
    center = np.asarray(handoff.get("center_free_parameter_values"), dtype=float)
    scale = np.asarray(handoff.get("scale"), dtype=float)
    factor_z = np.asarray(handoff.get("factor_z"), dtype=float)
    adapter_factor = np.diag(scale) @ factor_z
    rows = []
    for anchor in anchor_data.get("rows", ()):
        u = np.asarray(anchor["u_new"], dtype=float)
        adapter_free = np.asarray(adapter.latent_to_position(tf.constant(u, dtype=tf.float64)).numpy())
        row_formula_free = center + u @ adapter_factor.T
        wrong_column_free = center + scale * (factor_z @ u)
        wrong_right_no_transpose_free = center + u @ adapter_factor
        rows.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source": anchor["source"],
                "source_index": anchor["source_index"],
                "relation": anchor["relation"],
                "adapter_vs_row_formula_max_abs": float(
                    np.max(np.abs(adapter_free - row_formula_free))
                ),
                "wrong_column_vs_adapter_max_abs": float(
                    np.max(np.abs(wrong_column_free - adapter_free))
                ),
                "wrong_right_no_transpose_vs_adapter_max_abs": float(
                    np.max(np.abs(wrong_right_no_transpose_free - adapter_free))
                ),
                "adapter_free_parameter_values": adapter_free,
                "row_formula_free_parameter_values": row_formula_free,
                "wrong_column_free_parameter_values": wrong_column_free,
            }
        )
        if not np.all(np.isfinite(adapter_free)):
            vetoes.append(f"orientation_{anchor['anchor_id']}_adapter_free_nonfinite")
    adapter_errors = np.asarray(
        [row["adapter_vs_row_formula_max_abs"] for row in rows],
        dtype=float,
    )
    wrong_column_errors = np.asarray(
        [row["wrong_column_vs_adapter_max_abs"] for row in rows],
        dtype=float,
    )
    artifact_bug_indicated = bool(
        np.any(~np.isfinite(adapter_errors)) or np.max(adapter_errors) > 1.0e-10
    )
    if artifact_bug_indicated:
        vetoes.append("adapter_row_formula_orientation_mismatch")
    return json_ready(
        {
            "computed": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "recorded_phase2s_coordinate_formula": handoff.get("coordinate_formula"),
            "phase2u_adapter_coordinate_formula": (
                "free = center_free_parameter_values + u_new @ (diag(scale) @ factor_z).T"
            ),
            "rows": rows,
            "summary": {
                "adapter_vs_row_formula_max_abs": (
                    float(np.max(adapter_errors)) if adapter_errors.size else None
                ),
                "wrong_column_vs_adapter_min_abs": (
                    float(np.min(wrong_column_errors)) if wrong_column_errors.size else None
                ),
                "wrong_column_vs_adapter_max_abs": (
                    float(np.max(wrong_column_errors)) if wrong_column_errors.size else None
                ),
                "artifact_bug_indicated": artifact_bug_indicated,
                "display_string_ambiguous": True,
            },
        }
    )


def replay_proposal_log_densities(
    anchor_data: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    phase2x_settings = phase2x.Phase2XReferenceSettings()
    rows = []
    source_payloads = {"phase2w": phase2w_payload, "phase2x": phase2x_payload}
    for anchor in anchor_data.get("rows", ()):
        u = np.asarray(anchor["u_new"], dtype=float).reshape(1, 4)
        standard_logq = float(phase2w.standard_normal_log_prob(u)[0])
        shifted_logq = float(phase2x.shifted_mixture_log_prob(u, phase2x_settings)[0])
        saved_delta = None
        if anchor.get("source") in source_payloads and anchor.get("source_index") is not None:
            payload = source_payloads[str(anchor["source"])]
            source_index = int(anchor["source_index"])
            saved = np.asarray(
                payload.get("proposal", {}).get("proposal_log_prob"),
                dtype=float,
            )
            if saved.shape[0] > source_index:
                replayed = standard_logq if anchor["source"] == "phase2w" else shifted_logq
                saved_delta = float(abs(replayed - float(saved[source_index])))
                if saved_delta > 1.0e-10:
                    vetoes.append(f"{anchor['anchor_id']}_proposal_log_density_replay_mismatch")
        rows.append(
            {
                "anchor_id": anchor["anchor_id"],
                "source": anchor["source"],
                "source_index": anchor["source_index"],
                "relation": anchor["relation"],
                "standard_normal_log_prob": standard_logq,
                "phase2x_shifted_mixture_log_prob": shifted_logq,
                "source_saved_replay_abs_delta": saved_delta,
            }
        )
    saved_deltas = np.asarray(
        [
            row["source_saved_replay_abs_delta"]
            for row in rows
            if row["source_saved_replay_abs_delta"] is not None
        ],
        dtype=float,
    )
    artifact_bug_indicated = bool(saved_deltas.size and np.max(saved_deltas) > 1.0e-10)
    return json_ready(
        {
            "computed": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "rows": rows,
            "summary": {
                "source_saved_replay_max_abs_delta": (
                    float(np.max(saved_deltas)) if saved_deltas.size else None
                ),
                "artifact_bug_indicated": artifact_bug_indicated,
                "standard_log_prob_summary": finite_summary(
                    [row["standard_normal_log_prob"] for row in rows]
                ),
                "shifted_mixture_log_prob_summary": finite_summary(
                    [row["phase2x_shifted_mixture_log_prob"] for row in rows]
                ),
            },
        }
    )


def assess_hypotheses(
    phase2s_payload: Mapping[str, Any],
    anchor_data: Mapping[str, Any],
    anchor_eval: Mapping[str, Any],
    ray_eval: Mapping[str, Any],
    orientation_eval: Mapping[str, Any],
    proposal_replay: Mapping[str, Any],
) -> Mapping[str, Any]:
    trust_radius = float(
        (
            phase2s_payload.get("initializer", {})
            .get("geometry", {})
            .get("diagnostics", {})
            .get("config", {})
            .get("trust_radius", 0.6)
        )
    )
    top_rows = [row for row in anchor_eval.get("rows", ()) if row.get("relation") == "top_weight"]
    top_norms = np.asarray([row["norm_u_new"] for row in top_rows], dtype=float)
    residual_abs = ray_abs_values(ray_eval, "target_minus_quadratic")
    radial_score_abs = ray_abs_values(ray_eval, "radial_score_component")
    orientation_summary = orientation_eval.get("summary") or {}
    proposal_summary = proposal_replay.get("summary") or {}
    all_top_outside = bool(top_norms.size and np.all(top_norms > trust_radius))
    residual_large = bool(residual_abs.size and np.max(residual_abs) > 1.0)
    radial_score_large = bool(radial_score_abs.size and np.max(radial_score_abs) > 1.0)
    artifact_bug = bool(orientation_summary.get("artifact_bug_indicated")) or bool(
        proposal_summary.get("artifact_bug_indicated")
    )
    assessments = {
        "H1_local_quadratic_trust_region_exceeded": {
            "status": "supported_descriptively" if all_top_outside else "not_supported_by_this_diagnostic",
            "top_anchor_norm_summary": finite_summary(top_norms),
            "trust_radius": trust_radius,
            "evidence_role": "explanatory_only",
        },
        "H2_tail_or_ridge_undercoverage": {
            "status": (
                "supported_descriptively"
                if (all_top_outside and radial_score_large and not artifact_bug)
                else "inconclusive"
            ),
            "radial_score_abs_summary": finite_summary(radial_score_abs),
            "evidence_role": "explanatory_only",
        },
        "H3_orientation_or_scaling_mismatch": {
            "status": "bug_indicated" if orientation_summary.get("artifact_bug_indicated") else "not_supported",
            "orientation_summary": orientation_summary,
            "evidence_role": "bug_localization_diagnostic",
        },
        "H4_proposal_log_density_correct_family_poor": {
            "status": (
                "proposal_density_replay_passed_family_mismatch_plausible"
                if (not proposal_summary.get("artifact_bug_indicated") and all_top_outside)
                else "inconclusive_or_bug_indicated"
            ),
            "proposal_log_density_summary": proposal_summary,
            "evidence_role": "bug_localization_plus_explanatory",
        },
        "H5_local_not_global_map_locator": {
            "status": "plausible_not_certified",
            "top_anchor_target_delta_summary": finite_summary(
                [row["target_delta_from_center"] for row in top_rows]
            ),
            "evidence_role": "explanatory_only",
        },
        "H6_quadratic_extrapolation_failure": {
            "status": "supported_descriptively" if residual_large else "not_supported_by_this_diagnostic",
            "ray_abs_target_minus_quadratic_summary": finite_summary(residual_abs),
            "evidence_role": "explanatory_only",
        },
    }
    return json_ready(
        {
            "computed": True,
            "trust_radius": trust_radius,
            "assessments": assessments,
            "summary": {
                "artifact_bug_indicated": artifact_bug,
                "proposal_family_mismatch_indicated": bool(
                    not artifact_bug and (all_top_outside or residual_large)
                ),
                "next_repair_hint": (
                    "repair affine/proposal replay bug before more references"
                    if artifact_bug
                    else "consider non-diagonal, heavy-tail, ridge/local mixture, transport proposal, or abandon SNIS reference branch"
                ),
            },
        }
    )


def evaluate_target(adapter: Any, u: Any) -> tuple[float, np.ndarray]:
    value, score = adapter.log_prob_and_grad(tf.constant(u, dtype=tf.float64))
    value_float = float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())
    score_array = np.asarray(
        tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1]).numpy(),
        dtype=float,
    )
    if score_array.shape != (4,):
        raise ValueError("target score must have shape (4,)")
    return value_float, score_array


def center_target_value(phase2s_payload: Mapping[str, Any]) -> float:
    value = (
        phase2s_payload.get("initializer", {})
        .get("geometry", {})
        .get("diagnostics", {})
        .get("center_log_prob")
    )
    if value is None:
        value = (
            phase2s_payload.get("adapter_audit", {})
            .get("target_replay", {})
            .get("map_local_center_value")
        )
    return float(value)


def quadratic_log_prob(center_value: float, u: Any) -> float:
    array = np.asarray(u, dtype=float)
    return float(center_value - 0.5 * np.sum(np.square(array)))


def radial_score_component(u: Any, score: Any) -> float | None:
    u_array = np.asarray(u, dtype=float)
    score_array = np.asarray(score, dtype=float)
    norm = float(np.linalg.norm(u_array))
    if norm <= 0.0:
        return None
    return float(np.dot(score_array, u_array / norm))


def summarize_evaluation_rows(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return {
        "target_log_prob_summary": finite_summary([row["target_log_prob"] for row in rows]),
        "target_delta_from_center_summary": finite_summary(
            [row["target_delta_from_center"] for row in rows]
        ),
        "score_norm_summary": finite_summary([row["score_norm"] for row in rows]),
        "norm_u_new_summary": finite_summary([row["norm_u_new"] for row in rows]),
        "target_minus_quadratic_summary": finite_summary(
            [row["target_minus_quadratic"] for row in rows]
        ),
    }


def summarize_ray_profiles(profiles: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    values = []
    residuals = []
    radial_scores = []
    endpoint_deltas = []
    for profile in profiles:
        points = profile.get("points", ())
        values.extend(point["target_log_prob"] for point in points)
        residuals.extend(point["target_minus_quadratic"] for point in points)
        radial_scores.extend(
            point["radial_score_component"]
            for point in points
            if point["radial_score_component"] is not None
        )
        if points:
            endpoint_deltas.append(points[-1]["target_delta_from_center"])
    return {
        "target_log_prob_summary": finite_summary(values),
        "target_minus_quadratic_summary": finite_summary(residuals),
        "radial_score_component_summary": finite_summary(radial_scores),
        "endpoint_delta_from_center_summary": finite_summary(endpoint_deltas),
    }


def ray_abs_values(ray_eval: Mapping[str, Any], field: str) -> np.ndarray:
    values = []
    for profile in ray_eval.get("profiles", ()):
        for point in profile.get("points", ()):
            value = point.get(field)
            if value is not None:
                values.append(abs(float(value)))
    return np.asarray(values, dtype=float)


def finite_summary(values: Any) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    return {
        "shape": array.shape,
        "finite_count": int(np.sum(finite)),
        "nonfinite_count": int(np.sum(~finite)),
        "min": float(np.min(array[finite])) if np.any(finite) else None,
        "max": float(np.max(array[finite])) if np.any(finite) else None,
        "mean": float(np.mean(array[finite])) if np.any(finite) else None,
        "max_abs": float(np.max(np.abs(array[finite]))) if np.any(finite) else None,
    }


def environment_payload() -> Mapping[str, Any]:
    return {
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "tf_physical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_physical_devices()
        ],
        "tf_logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
    }


def git_payload() -> Mapping[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    except Exception:  # noqa: BLE001
        status = ""
    lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "dirty": bool(lines),
        "dirty_line_count": len(lines),
        "dirty_preview": lines[:20],
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    anchors = payload.get("anchors", {})
    anchor_eval = payload.get("anchor_evaluation", {})
    rays = payload.get("ray_profiles", {})
    orientation = payload.get("orientation_diagnostic", {})
    replay = payload.get("proposal_log_density_replay", {})
    hypotheses = payload.get("hypothesis_assessment", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2Y - Target Geometry Localization",
        "",
        "## Decision",
        "",
        f"- phase2y_target_geometry_localization_passed: `{decision['phase2y_target_geometry_localization_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- artifact_bug_indicated: `{decision['artifact_bug_indicated']}`",
        f"- proposal_family_mismatch_indicated: `{decision['proposal_family_mismatch_indicated']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Anchor Summary",
        "",
        f"- anchor_count: `{anchors.get('anchor_count')}`",
        f"- summary: `{anchors.get('summary')}`",
        f"- anchor_evaluation_summary: `{anchor_eval.get('summary')}`",
        "",
        "## Ray Summary",
        "",
        f"- profile_count: `{rays.get('profile_count')}`",
        f"- summary: `{rays.get('summary')}`",
        "",
        "## Orientation Diagnostic",
        "",
        f"- summary: `{orientation.get('summary')}`",
        f"- recorded_phase2s_coordinate_formula: `{orientation.get('recorded_phase2s_coordinate_formula')}`",
        f"- phase2u_adapter_coordinate_formula: `{orientation.get('phase2u_adapter_coordinate_formula')}`",
        "",
        "## Proposal Log-Density Replay",
        "",
        f"- summary: `{replay.get('summary')}`",
        "",
        "## Hypothesis Assessment",
        "",
    ]
    assessments = hypotheses.get("assessments", {})
    lines.extend(["| hypothesis | status | evidence role |", "| --- | --- | --- |"])
    for key, row in assessments.items():
        lines.append(f"| {key} | {row.get('status')} | {row.get('evidence_role')} |")
    lines.extend(
        [
            "",
            f"- summary: `{hypotheses.get('summary')}`",
            "",
            "## Inference Status",
            "",
            "| field | value |",
            "| --- | --- |",
        ]
    )
    for key, value in payload["inference_status"].items():
        lines.append(f"| {key} | {value} |")
    manifest = payload.get("run_manifest", {})
    lines.extend(
        [
            "",
            "## Run Manifest",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| command | `{manifest.get('command')}` |",
            f"| git | `{manifest.get('git')}` |",
            f"| environment | `{manifest.get('environment')}` |",
            f"| conda_env | `{manifest.get('conda_env')}` |",
            f"| cpu_gpu_status | {manifest.get('cpu_gpu_status')} |",
            f"| jit_compile | `{manifest.get('jit_compile')}` |",
            f"| tf32_mode | {manifest.get('tf32_mode')} |",
            f"| random_seeds | `{manifest.get('random_seeds')}` |",
            f"| wall_time_seconds | `{manifest.get('wall_time_seconds')}` |",
            f"| output_artifacts | `{manifest.get('output_artifacts')}` |",
            f"| plan_file | `{manifest.get('plan_file')}` |",
            f"| subplan_file | `{manifest.get('subplan_file')}` |",
            f"| result_file | `{manifest.get('result_file')}` |",
            "",
            "## Review Record",
            "",
            f"- reviewer: {payload.get('review_record', {}).get('reviewer')}",
            f"- review_strength: {payload.get('review_record', {}).get('review_strength')}",
            f"- claude_status: {payload.get('review_record', {}).get('claude_status')}",
            "",
            "## Nonclaims",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if tf.is_tensor(value):
        return json_ready(value.numpy())
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase2s-json", type=Path, default=DEFAULT_PHASE2S_PATH)
    parser.add_argument("--phase2t-json", type=Path, default=DEFAULT_PHASE2T_PATH)
    parser.add_argument("--phase2u-json", type=Path, default=DEFAULT_PHASE2U_PATH)
    parser.add_argument("--phase2v-json", type=Path, default=DEFAULT_PHASE2V_PATH)
    parser.add_argument("--phase2w-json", type=Path, default=DEFAULT_PHASE2W_PATH)
    parser.add_argument("--phase2x-json", type=Path, default=DEFAULT_PHASE2X_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    payload = run_phase2y_target_geometry_localization(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
        load_json(args.phase2v_json),
        load_json(args.phase2w_json),
        load_json(args.phase2x_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
        "phase2v_json": str(args.phase2v_json),
        "phase2w_json": str(args.phase2w_json),
        "phase2x_json": str(args.phase2x_json),
    }
    payload["run_manifest"]["wall_time_seconds"] = float(time.perf_counter() - start)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
