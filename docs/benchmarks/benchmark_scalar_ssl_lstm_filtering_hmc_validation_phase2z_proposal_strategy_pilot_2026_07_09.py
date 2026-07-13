"""Phase 2Z scalar SSL-LSTM proposal strategy pilot.

This diagnostic pilots predeclared Student-t proposal families after Phase 2Y
localized Phase 2W/2X failures to proposal-family mismatch rather than affine
or proposal-log-density replay bugs.  It may nominate a proposal for a later
independent replication phase.  It does not build a valid reference by itself,
does not run HMC, and does not claim posterior correctness, HMC readiness,
convergence, zero divergences, GPU/XLA readiness, default readiness, or
Zhao-Cui source faithfulness.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_"
    "proposal_strategy_pilot_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2z_proposal_strategy_pilot.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md"
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
DEFAULT_PHASE2Y_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json"
)
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md"
)
PHASE2U_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
PHASE2Y_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_2026_07_09.py"
)
PILOT_SAMPLE_COUNT = 1024
DIMENSION = 4
DEGREES_OF_FREEDOM = 4.0
NOMINATION_ESS_MIN = 256.0
NOMINATION_ESS_RATIO_MIN = 0.05
NOMINATION_MAX_WEIGHT_MAX = 0.05
PHASE2Z_CANDIDATES = (
    "student_t_centered",
    "student_t_shifted",
    "anchor_mixture_student_t",
    "ridge_line_student_t",
)
NONCLAIMS = (
    "Phase 2Z proposal-strategy pilot only",
    "candidate nomination is not an independent valid reference",
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
    "scalar_ssl_lstm_filtering_hmc_validation_phase2u_for_phase2z",
)
phase2y = load_module(
    PHASE2Y_MODULE_PATH,
    "scalar_ssl_lstm_filtering_hmc_validation_phase2y_for_phase2z",
)


@dataclass(frozen=True)
class StudentTMixtureProposal:
    name: str
    centers: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    df: float
    seed: tuple[int, int]
    sample_count: int = PILOT_SAMPLE_COUNT

    def __post_init__(self) -> None:
        centers = np.asarray(self.centers, dtype=float)
        scales = np.asarray(self.scales, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != DIMENSION:
            raise ValueError("proposal centers must have shape (k, 4)")
        if scales.shape != centers.shape:
            raise ValueError("proposal scales must match centers")
        if weights.shape != (centers.shape[0],):
            raise ValueError("proposal weights must have shape (k,)")
        if not np.all(np.isfinite(centers)):
            raise ValueError("proposal centers must be finite")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("proposal scales must be positive finite")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("proposal weights must be positive finite")
        if abs(float(np.sum(weights)) - 1.0) > 1.0e-12:
            raise ValueError("proposal weights must sum to one")
        if not np.isfinite(self.df) or float(self.df) <= 0.0:
            raise ValueError("proposal degrees of freedom must be positive finite")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("proposal seed must contain two integers")
        count = int(self.sample_count)
        if count <= 0:
            raise ValueError("proposal sample_count must be positive")
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "df", float(self.df))
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "sample_count", count)

    @property
    def component_count(self) -> int:
        return int(self.weights.shape[0])

    def component_counts(self) -> np.ndarray:
        return deterministic_counts(self.weights, self.sample_count)

    def payload(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "component_count": self.component_count,
            "sample_count": self.sample_count,
            "seed": self.seed,
            "degrees_of_freedom": self.df,
            "weights": self.weights,
            "component_counts": self.component_counts(),
            "centers": self.centers,
            "scales": self.scales,
            "log_density_formula": "logsumexp_k(log weight_k + independent diagonal Student-t log density)",
        }


def deterministic_counts(weights: Any, sample_count: int) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    raw = weights_array * int(sample_count)
    counts = np.floor(raw).astype(int)
    remainder = int(sample_count) - int(np.sum(counts))
    if remainder > 0:
        order = sorted(
            range(weights_array.shape[0]),
            key=lambda index: (-(raw[index] - counts[index]), index),
        )
        for index in order[:remainder]:
            counts[index] += 1
    if np.any(counts <= 0):
        raise ValueError("deterministic allocation produced empty component")
    return counts


def independent_student_t_log_prob(samples: Any, center: Any, scale: Any, df: float) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    center_array = np.asarray(center, dtype=float)
    scale_array = np.asarray(scale, dtype=float)
    z = (array - center_array) / scale_array
    df_value = float(df)
    normalizer = (
        math.lgamma((df_value + 1.0) / 2.0)
        - math.lgamma(df_value / 2.0)
        - 0.5 * math.log(df_value * math.pi)
    )
    return np.sum(
        normalizer
        - np.log(scale_array)
        - 0.5 * (df_value + 1.0) * np.log1p(np.square(z) / df_value),
        axis=-1,
    )


def student_t_mixture_log_prob(samples: Any, proposal: StudentTMixtureProposal) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != DIMENSION:
        raise ValueError("samples must have shape (n, 4)")
    log_terms = []
    for weight, center, scale in zip(proposal.weights, proposal.centers, proposal.scales, strict=True):
        log_terms.append(
            np.log(weight) + independent_student_t_log_prob(array, center, scale, proposal.df)
        )
    stacked = np.stack(log_terms, axis=1)
    max_log = np.max(stacked, axis=1)
    return max_log + np.log(np.sum(np.exp(stacked - max_log[:, np.newaxis]), axis=1))


def sample_student_t_mixture(proposal: StudentTMixtureProposal) -> Mapping[str, Any]:
    rng = np.random.default_rng(proposal.seed)
    counts = proposal.component_counts()
    samples = []
    component_labels = []
    for component_index, count in enumerate(counts):
        normal = rng.standard_normal(size=(int(count), DIMENSION))
        chi_square = rng.chisquare(proposal.df, size=(int(count), 1))
        draws = normal / np.sqrt(chi_square / proposal.df)
        component_samples = proposal.centers[component_index] + draws * proposal.scales[component_index]
        samples.append(component_samples)
        component_labels.extend([int(component_index)] * int(count))
    sample_array = np.vstack(samples)
    labels_array = np.asarray(component_labels, dtype=int)
    if sample_array.shape != (proposal.sample_count, DIMENSION):
        raise ValueError("sampled proposal shape mismatch")
    log_prob = student_t_mixture_log_prob(sample_array, proposal)
    return json_ready(
        {
            "generated": True,
            "samples": sample_array,
            "component_index": labels_array,
            "proposal_log_prob": log_prob,
            "component_counts": counts,
            "sample_summary": summarize_samples(sample_array),
            "proposal_log_prob_summary": finite_summary(log_prob),
        }
    )


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2z_proposal_strategy_pilot(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    phase2y_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    start = time.perf_counter()
    precondition = validate_phase2z_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        phase2w_payload,
        phase2x_payload,
        phase2y_payload,
    )
    vetoes = list(precondition.get("vetoes", ()))
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    candidate_rows: list[Mapping[str, Any]] = []
    proposal_specs: list[StudentTMixtureProposal] = []
    if not vetoes:
        adapter, adapter_audit = phase2u.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2z_adapter_not_built")
    if adapter is not None and not adapter_audit.get("vetoes"):
        try:
            proposal_specs = build_phase2z_proposals(phase2w_payload, phase2x_payload, phase2y_payload)
        except ValueError as exc:
            vetoes.append(f"phase2z_proposal_build_failed_{type(exc).__name__}")
        for proposal in proposal_specs:
            row = evaluate_candidate(adapter, proposal)
            candidate_rows.append(row)
            vetoes.extend(row.get("hard_vetoes", ()))

    candidate_gate = evaluate_candidate_gate(candidate_rows)
    unique_vetoes = tuple(dict.fromkeys(vetoes))
    pilot_artifact_valid = bool(not unique_vetoes and candidate_rows)
    nomination_count = int(candidate_gate.get("nominated_candidate_count", 0))
    passed = bool(pilot_artifact_valid)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2z_proposal_strategy_pilot",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": None if adapter is None else adapter.target_scope,
        "settings": {
            "candidate_names": PHASE2Z_CANDIDATES,
            "pilot_sample_count": PILOT_SAMPLE_COUNT,
            "degrees_of_freedom": DEGREES_OF_FREEDOM,
            "nomination_ess_min": NOMINATION_ESS_MIN,
            "nomination_ess_ratio_min": NOMINATION_ESS_RATIO_MIN,
            "nomination_max_weight_max": NOMINATION_MAX_WEIGHT_MAX,
            "uses_hmc_moments_for_tuning": False,
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
            "phase2y_json": str(DEFAULT_PHASE2Y_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "proposal_specs": [proposal.payload() for proposal in proposal_specs],
        "candidate_rows": candidate_rows,
        "candidate_gate": candidate_gate,
        "review_record": {
            "claude_review_attempted": False,
            "claude_status": "not_attempted_this_turn_prior_gate_blocked_external_repo_context_transfer",
            "reviewer": "Codex local substitute review",
            "review_strength": "weaker_than_full_claude_material_review",
            "verdict": "AGREE_FOR_PHASE2Z_IMPLEMENTATION_AFTER_PATCH",
            "review_path": (
                "docs/reviews/"
                "bayesfilter-scalar-filtering-hmc-validation-phase2z-subplan-codex-substitute-review-2026-07-09.md"
            ),
        },
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2z_proposal_strategy_pilot_passed": passed,
            "pilot_artifact_valid": pilot_artifact_valid,
            "candidate_nominated": nomination_count > 0,
            "nominated_candidates": candidate_gate.get("nominated_candidates", ()),
            "nominated_candidate_count": nomination_count,
            "vetoes": unique_vetoes,
            "zero_divergence_claim_made": False,
            "viable_for_phase3_gpu_xla_subplan": False,
            "next_justified_action": (
                "write Phase 2Z result and draft/review Phase 2ZA independent replication subplan"
                if nomination_count > 0
                else "write Phase 2Z result and draft/review SNIS-abandonment or transport/sequential-reference decision subplan"
            ),
        },
        "metric_roles": {
            "phase2z_proposal_strategy_pilot_passed": "primary_phase2z_artifact_validity_pass_fail",
            "candidate_nomination": "pilot_nomination_only_not_reference_validity",
            "candidate_ess": "pilot_nomination_screen",
            "candidate_ess_ratio": "pilot_nomination_screen",
            "max_normalized_weight": "pilot_nomination_screen",
            "target_proposal_log_weight_finiteness": "hard_veto_evidence",
            "weighted_moments": "descriptive_only",
            "top_weight_coordinates": "explanatory_only",
            "runtime": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "reference_validity": "not assessed; Phase 2Z is pilot nomination only",
            "hmc_reference_agreement": "not assessed",
            "native_divergence": "not assessed; no HMC run",
            "zero_divergence_claim": "not made",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": (
                "per-candidate ESS, ESS ratio, max weight, weighted moments, top weights, and runtime"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed",
            "gpu_xla_readiness": "blocked",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "independent reference replication with fresh seeds"
                if nomination_count > 0
                else "reviewed decision to abandon SNIS branch or move to transport/sequential reference"
            ),
        },
        "decision_table": {
            "decision": "Phase 2Z proposal strategy pilot",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "A passing pilot nomination can be overfit to Phase 2Y diagnostics "
                "and requires independent replication."
            ),
            "next_justified_action": (
                "draft/review Phase 2ZA independent replication subplan"
                if nomination_count > 0
                else "draft/review SNIS-abandonment or transport/sequential-reference decision subplan"
            ),
            "what_is_not_being_concluded": (
                "No valid reference, posterior correctness, HMC readiness, convergence, "
                "zero-divergence claim, sampler superiority, statistical ranking, GPU/XLA "
                "readiness, default readiness, or Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": [proposal.seed for proposal in proposal_specs],
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
                "A nominated candidate may be overfit to Phase 2Y top-weight anchors "
                "or pilot seeds rather than being a robust reference proposal."
            ),
            "what_would_overturn": (
                "Independent replication with fresh seeds failing ESS/weight screens, "
                "or evidence that SNIS remains unstable despite richer proposals."
            ),
            "weakest_evidence": (
                "Smaller timeout-repaired pilot per candidate, no uncertainty analysis, and anchor/ridge "
                "candidates partially informed by failed-proposal diagnostics."
            ),
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_phase2z_handoff(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    phase2y_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    expected = {
        "phase2s": "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1",
        "phase2t": "scalar_ssl_lstm.filtering_hmc_validation_phase2t_map_local_reference_handoff.v1",
        "phase2u": "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1",
        "phase2v": "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1",
        "phase2w": "scalar_ssl_lstm.filtering_hmc_validation_phase2w_importance_reference_agreement.v1",
        "phase2x": "scalar_ssl_lstm.filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.v1",
        "phase2y": "scalar_ssl_lstm.filtering_hmc_validation_phase2y_target_geometry_localization.v1",
    }
    payloads = {
        "phase2s": phase2s_payload,
        "phase2t": phase2t_payload,
        "phase2u": phase2u_payload,
        "phase2v": phase2v_payload,
        "phase2w": phase2w_payload,
        "phase2x": phase2x_payload,
        "phase2y": phase2y_payload,
    }
    for name, schema in expected.items():
        if payloads[name].get("schema_version") != schema:
            vetoes.append(f"{name}_schema_mismatch")
    if phase2s_payload.get("decision", {}).get("phase2s_geometry_centering_repair_passed") is not True:
        vetoes.append("phase2s_not_passed")
    if phase2t_payload.get("decision", {}).get("phase2t_map_local_reference_handoff_passed") is not True:
        vetoes.append("phase2t_not_passed")
    if phase2u_payload.get("decision", {}).get("phase2u_retuned_map_local_hmc_screen_passed") is not True:
        vetoes.append("phase2u_not_passed")
    if phase2v_payload.get("decision", {}).get("phase2v_longer_selected_map_local_screen_passed") is not True:
        vetoes.append("phase2v_not_passed")
    expected_vetoes = {"reference_ess_below_threshold", "reference_ess_ratio_below_threshold"}
    for name, payload in (("phase2w", phase2w_payload), ("phase2x", phase2x_payload)):
        if payload.get("decision", {}).get("reference_valid") is not False:
            vetoes.append(f"{name}_reference_validity_boundary_not_failed")
        actual_vetoes = set(str(item) for item in payload.get("decision", {}).get("vetoes", ()))
        if actual_vetoes != expected_vetoes:
            vetoes.append(f"{name}_failure_not_limited_to_ess_vetoes")
        if payload.get("hmc_reference_agreement", {}).get("evaluated") is not False:
            vetoes.append(f"{name}_hmc_reference_agreement_was_evaluated")
    decision_y = phase2y_payload.get("decision", {})
    if decision_y.get("phase2y_target_geometry_localization_passed") is not True:
        vetoes.append("phase2y_not_passed")
    if decision_y.get("artifact_bug_indicated") is not False:
        vetoes.append("phase2y_artifact_bug_indicated")
    if decision_y.get("proposal_family_mismatch_indicated") is not True:
        vetoes.append("phase2y_proposal_family_mismatch_not_indicated")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2y_decision": decision_y,
        "phase2w_decision": phase2w_payload.get("decision", {}),
        "phase2x_decision": phase2x_payload.get("decision", {}),
        "uses_hmc_moments_for_tuning": False,
    }


def build_phase2z_proposals(
    phase2w_payload: Mapping[str, Any],
    phase2x_payload: Mapping[str, Any],
    phase2y_payload: Mapping[str, Any],
) -> list[StudentTMixtureProposal]:
    pilot_mean = np.asarray(
        phase2w_payload.get("importance_reference", {}).get("mean_u_new"),
        dtype=float,
    )
    phase2x_std = np.asarray(
        phase2x_payload.get("importance_reference", {}).get("std_u_new"),
        dtype=float,
    )
    if pilot_mean.shape != (DIMENSION,) or not np.all(np.isfinite(pilot_mean)):
        raise ValueError("phase2w pilot mean missing or nonfinite")
    if phase2x_std.shape != (DIMENSION,) or not np.all(np.isfinite(phase2x_std)):
        raise ValueError("phase2x std missing or nonfinite")
    centered_scale = np.clip(phase2x_std, 1.5, 4.0)
    local_scale = np.clip(0.75 * phase2x_std, 1.0, 3.0)
    zero = np.zeros(DIMENSION, dtype=float)
    top_anchors = unique_top_anchors(phase2y_payload)
    ridge_centers = ridge_line_centers(top_anchors, radii=(0.0, 2.5, 5.0))
    anchor_centers = np.vstack([zero, pilot_mean, top_anchors])
    anchor_weights = np.concatenate(
        [
            np.asarray([0.25, 0.25], dtype=float),
            np.full((top_anchors.shape[0],), 0.50 / top_anchors.shape[0], dtype=float),
        ]
    )
    ridge_weights = np.full((ridge_centers.shape[0],), 1.0 / ridge_centers.shape[0], dtype=float)
    return [
        StudentTMixtureProposal(
            name="student_t_centered",
            centers=zero.reshape(1, DIMENSION),
            scales=centered_scale.reshape(1, DIMENSION),
            weights=np.ones(1, dtype=float),
            df=DEGREES_OF_FREEDOM,
            seed=(20260709, 6701),
        ),
        StudentTMixtureProposal(
            name="student_t_shifted",
            centers=pilot_mean.reshape(1, DIMENSION),
            scales=centered_scale.reshape(1, DIMENSION),
            weights=np.ones(1, dtype=float),
            df=DEGREES_OF_FREEDOM,
            seed=(20260709, 6702),
        ),
        StudentTMixtureProposal(
            name="anchor_mixture_student_t",
            centers=anchor_centers,
            scales=np.repeat(local_scale.reshape(1, DIMENSION), anchor_centers.shape[0], axis=0),
            weights=anchor_weights,
            df=DEGREES_OF_FREEDOM,
            seed=(20260709, 6703),
        ),
        StudentTMixtureProposal(
            name="ridge_line_student_t",
            centers=ridge_centers,
            scales=np.repeat(local_scale.reshape(1, DIMENSION), ridge_centers.shape[0], axis=0),
            weights=ridge_weights,
            df=DEGREES_OF_FREEDOM,
            seed=(20260709, 6704),
        ),
    ]


def unique_top_anchors(phase2y_payload: Mapping[str, Any]) -> np.ndarray:
    rows = phase2y_payload.get("anchor_evaluation", {}).get("rows", ())
    anchors = []
    seen: set[tuple[float, ...]] = set()
    for row in rows:
        if row.get("relation") != "top_weight":
            continue
        value = np.asarray(row.get("u_new"), dtype=float)
        if value.shape != (DIMENSION,) or not np.all(np.isfinite(value)):
            continue
        key = tuple(float(round(item, 12)) for item in value)
        if key not in seen:
            seen.add(key)
            anchors.append(value)
    if not anchors:
        raise ValueError("no finite Phase 2Y top anchors found")
    return np.asarray(anchors, dtype=float)


def ridge_line_centers(anchors: Any, *, radii: Sequence[float]) -> np.ndarray:
    anchor_array = np.asarray(anchors, dtype=float)
    directions = []
    seen: set[tuple[float, ...]] = set()
    for row in anchor_array:
        norm = float(np.linalg.norm(row))
        if norm <= 0.0:
            continue
        direction = row / norm
        key = tuple(float(round(item, 8)) for item in direction)
        if key not in seen:
            seen.add(key)
            directions.append(direction)
    if not directions:
        raise ValueError("no nonzero ridge directions found")
    centers = []
    for direction in directions:
        for radius in radii:
            centers.append(float(radius) * direction)
    return np.asarray(centers, dtype=float)


def evaluate_candidate(adapter: Any, proposal: StudentTMixtureProposal) -> Mapping[str, Any]:
    start = time.perf_counter()
    hard_vetoes: list[str] = []
    generated = sample_student_t_mixture(proposal)
    samples = np.asarray(generated["samples"], dtype=float)
    proposal_log_prob = np.asarray(generated["proposal_log_prob"], dtype=float)
    replay_log_prob = student_t_mixture_log_prob(samples, proposal)
    replay_delta = float(np.max(np.abs(replay_log_prob - proposal_log_prob)))
    if replay_delta > 1.0e-10:
        hard_vetoes.append("proposal_log_density_replay_mismatch")
    target_log_prob = np.full((proposal.sample_count,), np.nan, dtype=float)
    target_score_norm = np.full((proposal.sample_count,), np.nan, dtype=float)
    first_error = None
    for index, sample in enumerate(samples):
        try:
            value, score = phase2y.evaluate_target(adapter, sample)
            target_log_prob[index] = value
            target_score_norm[index] = float(np.linalg.norm(score))
        except Exception as exc:  # noqa: BLE001 - fail-closed diagnostic.
            first_error = f"{type(exc).__name__}: {exc}"
            hard_vetoes.append("target_evaluation_exception")
            break
    if not np.all(np.isfinite(samples)):
        hard_vetoes.append("proposal_samples_nonfinite")
    if not np.all(np.isfinite(proposal_log_prob)):
        hard_vetoes.append("proposal_log_prob_nonfinite")
    if not np.all(np.isfinite(target_log_prob)):
        hard_vetoes.append("target_log_prob_nonfinite")
    if not np.all(np.isfinite(target_score_norm)):
        hard_vetoes.append("target_score_norm_nonfinite")
    log_weights = target_log_prob - proposal_log_prob
    reference = importance_summary(samples, target_log_prob, proposal_log_prob, hard_vetoes)
    nominated = bool(
        not hard_vetoes
        and reference["ess"] >= NOMINATION_ESS_MIN
        and reference["ess_ratio"] >= NOMINATION_ESS_RATIO_MIN
        and reference["weight_summary"]["max"] <= NOMINATION_MAX_WEIGHT_MAX
    )
    nomination_failures = []
    if reference["ess"] < NOMINATION_ESS_MIN:
        nomination_failures.append("ess_below_nomination_screen")
    if reference["ess_ratio"] < NOMINATION_ESS_RATIO_MIN:
        nomination_failures.append("ess_ratio_below_nomination_screen")
    if reference["weight_summary"]["max"] > NOMINATION_MAX_WEIGHT_MAX:
        nomination_failures.append("max_weight_above_nomination_screen")
    return json_ready(
        {
            "candidate_name": proposal.name,
            "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
            "first_error": first_error,
            "proposal": proposal.payload(),
            "generated_proposal": {
                key: value
                for key, value in generated.items()
                if key not in {"samples", "proposal_log_prob"}
            },
            "proposal_log_density_replay_max_abs_delta": replay_delta,
            "target_log_prob_summary": finite_summary(target_log_prob),
            "target_score_norm_summary": finite_summary(target_score_norm),
            "log_weight_summary": finite_summary(log_weights),
            "importance_summary": reference,
            "nominated_for_independent_replication": nominated,
            "nomination_failures": tuple(nomination_failures),
            "runtime_seconds": float(time.perf_counter() - start),
            "nonclaims": (
                "pilot candidate only",
                "not a valid independent reference",
                "not HMC readiness evidence",
            ),
        }
    )


def importance_summary(
    samples: np.ndarray,
    target_log_prob: np.ndarray,
    proposal_log_prob: np.ndarray,
    hard_vetoes: Sequence[str],
) -> Mapping[str, Any]:
    if hard_vetoes:
        return {
            "computed": False,
            "ess": float("nan"),
            "ess_ratio": float("nan"),
            "weight_summary": {"max": float("nan")},
        }
    log_weights = target_log_prob - proposal_log_prob
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    normalized = shifted / float(np.sum(shifted))
    ess = float(1.0 / np.sum(np.square(normalized)))
    mean = np.sum(normalized[:, np.newaxis] * samples, axis=0)
    centered = samples - mean
    variance = np.sum(normalized[:, np.newaxis] * np.square(centered), axis=0)
    top_order = np.argsort(normalized)[-8:][::-1]
    return json_ready(
        {
            "computed": True,
            "ess": ess,
            "ess_ratio": float(ess / samples.shape[0]),
            "mean_u_new": mean,
            "std_u_new": np.sqrt(variance),
            "mean_mcse_u_new": np.sqrt(variance / ess),
            "weight_summary": {
                "min": float(np.min(normalized)),
                "max": float(np.max(normalized)),
                "sum": float(np.sum(normalized)),
                "nonzero_count": int(np.sum(normalized > 0.0)),
            },
            "top_weight_rows": [
                {
                    "rank": int(rank),
                    "sample_index": int(index),
                    "normalized_weight": float(normalized[int(index)]),
                    "u_new": samples[int(index)],
                    "target_log_prob": float(target_log_prob[int(index)]),
                    "proposal_log_prob": float(proposal_log_prob[int(index)]),
                    "log_weight": float(log_weights[int(index)]),
                }
                for rank, index in enumerate(top_order)
            ],
        }
    )


def evaluate_candidate_gate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    nominated = []
    for row in rows:
        if row.get("nominated_for_independent_replication") is True:
            summary = row.get("importance_summary", {})
            nominated.append(
                {
                    "candidate_name": row["candidate_name"],
                    "ess": summary.get("ess"),
                    "ess_ratio": summary.get("ess_ratio"),
                    "max_weight": (summary.get("weight_summary") or {}).get("max"),
                }
            )
    return {
        "candidate_count": len(rows),
        "nominated_candidate_count": len(nominated),
        "nominated_candidates": nominated,
        "candidate_names": [row.get("candidate_name") for row in rows],
        "interpretation": (
            "pilot nomination only; independent replication required"
            if nominated
            else "no candidate passed the pilot nomination screen"
        ),
    }


def summarize_samples(samples: Any) -> Mapping[str, Any]:
    array = np.asarray(samples, dtype=float)
    return {
        "shape": array.shape,
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "max_abs": float(np.max(np.abs(array))),
    }


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
    gate = payload.get("candidate_gate", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2Z - Proposal Strategy Pilot",
        "",
        "## Decision",
        "",
        f"- phase2z_proposal_strategy_pilot_passed: `{decision['phase2z_proposal_strategy_pilot_passed']}`",
        f"- pilot_artifact_valid: `{decision['pilot_artifact_valid']}`",
        f"- candidate_nominated: `{decision['candidate_nominated']}`",
        f"- nominated_candidates: `{decision['nominated_candidates']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Candidate Gate",
        "",
        f"- summary: `{gate}`",
        "",
        "## Candidate Rows",
        "",
        "| candidate | nominated | ESS | ESS ratio | max weight | failures | hard vetoes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("candidate_rows", ()):
        summary = row.get("importance_summary", {})
        weight_summary = summary.get("weight_summary") or {}
        lines.append(
            "| {name} | `{nominated}` | `{ess}` | `{ratio}` | `{max_weight}` | `{failures}` | `{vetoes}` |".format(
                name=row.get("candidate_name"),
                nominated=row.get("nominated_for_independent_replication"),
                ess=summary.get("ess"),
                ratio=summary.get("ess_ratio"),
                max_weight=weight_summary.get("max"),
                failures=row.get("nomination_failures"),
                vetoes=row.get("hard_vetoes"),
            )
        )
    lines.extend(
        [
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
    parser.add_argument("--phase2y-json", type=Path, default=DEFAULT_PHASE2Y_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    payload = run_phase2z_proposal_strategy_pilot(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
        load_json(args.phase2v_json),
        load_json(args.phase2w_json),
        load_json(args.phase2x_json),
        load_json(args.phase2y_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
        "phase2v_json": str(args.phase2v_json),
        "phase2w_json": str(args.phase2w_json),
        "phase2x_json": str(args.phase2x_json),
        "phase2y_json": str(args.phase2y_json),
    }
    payload["run_manifest"]["wall_time_seconds"] = float(time.perf_counter() - start)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
