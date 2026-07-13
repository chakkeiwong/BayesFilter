"""Phase 2W MAP-local scalar SSL-LSTM importance-reference agreement.

This diagnostic builds a fixed CPU-hidden self-normalized importance reference
in the Phase 2S/2U MAP-local ``u_new`` coordinate and compares Phase 2V HMC
moment summaries only if the reference validity gates pass.  It does not claim
posterior correctness, HMC readiness, convergence, zero divergences, GPU/XLA
readiness, default readiness, or Zhao-Cui source faithfulness.
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
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_"
    "importance_reference_agreement_2026_07_09.py"
)
SCHEMA_VERSION = (
    "scalar_ssl_lstm.filtering_hmc_validation_phase2w_importance_reference_agreement.v1"
)
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md"
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
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md"
)
PHASE2U_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)
PHASE2V_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_2026_07_09.py"
)
NONCLAIMS = (
    "Phase 2W MAP-local self-normalized importance-reference agreement diagnostic only",
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


@dataclass(frozen=True)
class Phase2WReferenceSettings:
    """Fixed Phase 2W importance-reference settings."""

    proposal_sample_count: int = 1024
    seed: tuple[int, int] = (20260709, 6501)
    reference_ess_min: float = 128.0
    reference_ess_ratio_min: float = 0.125
    mean_abs_floor: float = 0.75
    mean_mcse_multiplier: float = 4.0
    std_ratio_lower: float = 0.5
    std_ratio_upper: float = 2.0

    def __post_init__(self) -> None:
        count = int(self.proposal_sample_count)
        if count <= 0 or count % 2 != 0:
            raise ValueError("proposal_sample_count must be a positive even integer")
        object.__setattr__(self, "proposal_sample_count", count)
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        object.__setattr__(self, "seed", seed)
        for name in (
            "reference_ess_min",
            "reference_ess_ratio_min",
            "mean_abs_floor",
            "mean_mcse_multiplier",
            "std_ratio_lower",
            "std_ratio_upper",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.reference_ess_min <= 0.0:
            raise ValueError("reference_ess_min must be positive")
        if not (0.0 < self.reference_ess_ratio_min <= 1.0):
            raise ValueError("reference_ess_ratio_min must be in (0, 1]")
        if self.mean_abs_floor < 0.0 or self.mean_mcse_multiplier < 0.0:
            raise ValueError("mean thresholds must be nonnegative")
        if not (0.0 < self.std_ratio_lower <= self.std_ratio_upper):
            raise ValueError("std ratio interval must satisfy 0 < lower <= upper")

    def payload(self) -> Mapping[str, Any]:
        return {
            "proposal_coordinate": "phase2s_phase2u_map_local_u_new",
            "proposal_distribution": "standard_normal_N_0_I4",
            "proposal_sample_count": self.proposal_sample_count,
            "antithetic_base_sample_count": self.proposal_sample_count // 2,
            "seed": self.seed,
            "reference_ess_min": self.reference_ess_min,
            "reference_ess_ratio_min": self.reference_ess_ratio_min,
            "mean_abs_floor": self.mean_abs_floor,
            "mean_mcse_multiplier": self.mean_mcse_multiplier,
            "mean_threshold_formula": "max(mean_abs_floor, mean_mcse_multiplier * reference_mean_mcse)",
            "std_ratio_interval": (self.std_ratio_lower, self.std_ratio_upper),
            "target_log_prob_route": "Phase 2U MAP-local adapter log_prob_and_grad; gradients ignored",
            "proposal_log_density": "standard normal including normalizing constant",
            "constant_affine_jacobian": "omitted consistently from normalized weights",
            "cpu_hidden": True,
            "use_xla": False,
        }


def load_phase2u_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2u_for_phase2w",
        PHASE2U_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 2U harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_phase2v_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2v_for_phase2w",
        PHASE2V_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 2V harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2w_importance_reference_agreement(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    settings: Phase2WReferenceSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2WReferenceSettings() if settings is None else settings
    start = time.perf_counter()
    phase2u = load_phase2u_module()
    phase2v = load_phase2v_module()
    precondition = validate_phase2w_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        cfg,
        phase2u=phase2u,
        phase2v=phase2v,
    )
    vetoes = list(precondition.get("vetoes", ()))
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    proposal: Mapping[str, Any] = {
        "generated": False,
        "vetoes": ("phase2w_reference_not_run",),
    }
    reference: Mapping[str, Any] = {
        "computed": False,
        "vetoes": ("phase2w_reference_not_run",),
    }
    agreement: Mapping[str, Any] = {
        "evaluated": False,
        "vetoes": ("phase2w_agreement_not_evaluated",),
    }

    if not vetoes:
        adapter, adapter_audit = phase2u.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2w_adapter_not_built")

    if adapter is not None and not adapter_audit.get("vetoes"):
        proposal = generate_antithetic_standard_normal_proposal(cfg)
        vetoes.extend(proposal.get("vetoes", ()))
        if not proposal.get("vetoes"):
            reference = compute_importance_reference(adapter, proposal, cfg)
            vetoes.extend(reference.get("vetoes", ()))
        if not reference.get("vetoes"):
            agreement = compare_hmc_to_reference(phase2v_payload, reference, cfg)
            vetoes.extend(agreement.get("vetoes", ()))

    unique_vetoes = tuple(dict.fromkeys(vetoes))
    reference_valid = bool(reference.get("reference_valid") is True and not reference.get("vetoes"))
    agreement_passed = bool(agreement.get("passed") is True and not agreement.get("vetoes"))
    passed = bool(not unique_vetoes and reference_valid and agreement_passed)
    telemetry = telemetry_policy_payload(phase2v_payload)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2w_importance_reference_agreement",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "script": f"docs/benchmarks/{SCRIPT_NAME}",
        "plan_path": PLAN_PATH,
        "subplan_path": SUBPLAN_PATH,
        "result_path": RESULT_PATH,
        "classification": "extension_or_invention",
        "target_scope": None if adapter is None else adapter.target_scope,
        "settings": cfg.payload(),
        "source_artifacts": {
            "phase2s_json": str(DEFAULT_PHASE2S_PATH.relative_to(ROOT)),
            "phase2t_json": str(DEFAULT_PHASE2T_PATH.relative_to(ROOT)),
            "phase2u_json": str(DEFAULT_PHASE2U_PATH.relative_to(ROOT)),
            "phase2v_json": str(DEFAULT_PHASE2V_PATH.relative_to(ROOT)),
        },
        "precondition": precondition,
        "adapter_audit": adapter_audit,
        "proposal": proposal,
        "importance_reference": reference,
        "hmc_reference_agreement": agreement,
        "telemetry_policy": telemetry,
        "environment": environment_payload(),
        "git": git_payload(),
        "decision": {
            "phase2w_importance_reference_agreement_passed": passed,
            "reference_valid": reference_valid,
            "agreement_passed": agreement_passed,
            "vetoes": unique_vetoes,
            "reference_ess": reference.get("ess"),
            "reference_ess_ratio": reference.get("ess_ratio"),
            "zero_divergence_claim_made": False,
            "viable_for_phase3_gpu_xla_subplan": passed,
            "next_justified_action": (
                "write Phase 2W result and draft/review narrowly scoped GPU/XLA reproduction subplan"
                if passed
                else "write Phase 2W blocker or narrower reference/tuning/localization repair result"
            ),
        },
        "metric_roles": {
            "phase2w_importance_reference_agreement_passed": "primary_phase2w_pass_fail",
            "phase2s_phase2u_phase2v_artifact_validity": "hard_veto_evidence",
            "reference_ess": "reference_validity_veto",
            "reference_ess_ratio": "reference_validity_veto",
            "proposal_target_log_prob_finiteness": "hard_veto_evidence",
            "proposal_log_weight_finiteness": "hard_veto_evidence",
            "hmc_mean_delta_threshold": "phase2w_agreement_gate_after_reference_validity",
            "hmc_std_ratio_interval": "phase2w_agreement_gate_after_reference_validity",
            "native_divergence": "hard_veto_if_available_positive; unavailable is not zero divergences",
            "acceptance_value": "descriptive_from_phase2v_after_prior_screen",
            "runtime": "explanatory_only",
        },
        "inference_status": {
            "hard_veto_screen": "passed" if passed else "failed",
            "reference_validity": "passed" if reference_valid else "failed",
            "hmc_reference_agreement": (
                "passed" if agreement_passed else "failed or not interpreted"
            ),
            "native_divergence": telemetry["native_divergence_interpretation"],
            "zero_divergence_claim": "not made",
            "statistically_supported_ranking": (
                "none; one HMC chain compared to one fixed importance reference with no ranking"
            ),
            "descriptive_only_differences": (
                "reference ESS, weighted moments, HMC-reference deltas, std ratios, and runtime"
            ),
            "posterior_correctness": "not assessed",
            "hmc_readiness": "not assessed; Phase 2W is a narrow reference-agreement diagnostic",
            "gpu_xla_readiness": "blocked until a later reviewed GPU/XLA reproduction phase",
            "default_readiness": "not assessed",
            "next_evidence_needed": (
                "reviewed GPU/XLA reproduction subplan"
                if passed
                else "reviewed reference/tuning/localization repair"
            ),
        },
        "decision_table": {
            "decision": "Phase 2W MAP-local importance-reference agreement diagnostic",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
            "main_uncertainty": (
                "A fixed standard-normal importance reference can be a useful local "
                "diagnostic only if ESS gates pass; it is not an exact posterior oracle."
            ),
            "next_justified_action": (
                "draft/review narrowly scoped GPU/XLA reproduction subplan"
                if passed
                else "write blocker or narrower repair"
            ),
            "what_is_not_being_concluded": (
                "No posterior correctness, HMC readiness, convergence, zero-divergence "
                "claim when native divergence is unavailable, sampler superiority, "
                "statistical ranking, GPU/XLA readiness, default readiness, or "
                "Zhao-Cui source faithfulness."
            ),
        },
        "run_manifest": {
            "command": (
                "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python "
                "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py "
                "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json "
                "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md"
            ),
            "git": git_payload(),
            "environment": environment_payload(),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "cpu_gpu_status": "CPU-hidden debug/reference exception",
            "jit_compile": False,
            "tf32_mode": "disabled_by_cpu_hidden_debug_contract",
            "data_version": "stateless_simulated_scalar_ssl_lstm_filtering_path_v1",
            "random_seeds": (cfg.seed,),
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
                "Agreement may reflect a broad local moment screen around the MAP-local "
                "coordinate rather than convergence or exact posterior validity."
            ),
            "what_would_overturn": (
                "Invalid importance ESS, failed repeated reference checks, longer-chain "
                "disagreement, positive native divergence when available, or GPU/XLA mismatch."
            ),
            "weakest_evidence": (
                "One CPU-hidden fixed proposal reference and one 128-draw selected HMC chain."
            ),
        },
        "review_record": {
            "reviewer": "Codex substitute reviewer",
            "review_strength": "weaker_than_full_claude_material_review",
            "claude_status": "unavailable_for_repo_context_material_review_per_prior_handoff",
            "rounds": (
                {
                    "round": 1,
                    "verdict": "REVISE",
                    "findings": (
                        "reference_mean_mcse needed square-root definition",
                        "Phase 2U artifact/selected-kernel handoff needed explicit validity veto",
                    ),
                },
                {
                    "round": 2,
                    "verdict": "AGREE",
                    "findings": ("prior blockers resolved",),
                },
            ),
        },
        "nonclaims": NONCLAIMS,
    }
    return json_ready(payload)


def validate_phase2w_handoff(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    settings: Phase2WReferenceSettings,
    *,
    phase2u: Any | None = None,
    phase2v: Any | None = None,
) -> Mapping[str, Any]:
    phase2u_module = load_phase2u_module() if phase2u is None else phase2u
    phase2v_module = load_phase2v_module() if phase2v is None else phase2v
    vetoes: list[str] = []
    phase2u_settings = phase2u_module.Phase2UScreenSettings()
    phase2v_settings = phase2v_module.Phase2VScreenSettings()
    phase2u_precondition = phase2u_module.validate_handoff_artifacts(
        phase2s_payload,
        phase2t_payload,
        phase2u_settings,
    )
    vetoes.extend(f"phase2u_precondition_{item}" for item in phase2u_precondition.get("vetoes", ()))
    phase2u_handoff = phase2v_module.validate_phase2u_handoff(
        phase2u_payload,
        phase2v_settings,
    )
    vetoes.extend(f"phase2u_handoff_{item}" for item in phase2u_handoff.get("vetoes", ()))
    phase2v_validity = validate_phase2v_payload(phase2v_payload, phase2v_settings)
    vetoes.extend(f"phase2v_{item}" for item in phase2v_validity.get("vetoes", ()))
    if int(settings.proposal_sample_count) != 1024:
        vetoes.append("proposal_sample_count_not_predeclared_1024")
    if tuple(settings.seed) != (20260709, 6501):
        vetoes.append("proposal_seed_not_predeclared")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2u_precondition": phase2u_precondition,
        "phase2u_handoff": phase2u_handoff,
        "phase2v_validity": phase2v_validity,
        "settings_contract": settings.payload(),
    }


def validate_phase2v_payload(
    phase2v_payload: Mapping[str, Any],
    settings: Any,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if (
        phase2v_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1"
    ):
        vetoes.append("schema_mismatch")
    decision = phase2v_payload.get("decision", {})
    if decision.get("phase2v_longer_selected_map_local_screen_passed") is not True:
        vetoes.append("decision_not_passed")
    if decision.get("vetoes"):
        vetoes.append("vetoes_present")
    if decision.get("viable_for_phase3_gpu_xla_subplan") is True:
        vetoes.append("gpu_xla_viability_unexpected")
    selected = decision.get("selected_kernel") or {}
    if int(selected.get("num_leapfrog_steps", -1)) != int(settings.num_leapfrog_steps):
        vetoes.append("selected_leapfrog_mismatch")
    if abs(float(selected.get("step_size", np.nan)) - float(settings.step_size)) > 1.0e-12:
        vetoes.append("selected_step_size_mismatch")
    if (
        abs(
            float(selected.get("trajectory_length_L_times_epsilon", np.nan))
            - float(settings.trajectory_length)
        )
        > 1.0e-12
    ):
        vetoes.append("selected_trajectory_length_mismatch")
    row = phase2v_payload.get("selected_kernel_row") or {}
    if row.get("status") != "passed_hard_vetoes":
        vetoes.append("selected_kernel_row_hard_veto_screen_failed")
    if row.get("hard_vetoes"):
        vetoes.append("selected_kernel_row_hard_vetoes_present")
    samples = row.get("samples_summary") or {}
    hmc_mean = _vector_or_none(samples.get("mean_u_new"), 4)
    hmc_std = _vector_or_none(samples.get("std_u_new"), 4)
    if hmc_mean is None:
        vetoes.append("hmc_mean_missing_or_nonfinite")
    if hmc_std is None or np.any(hmc_std <= 0.0):
        vetoes.append("hmc_std_missing_nonfinite_or_nonpositive")
    finite_count = int(samples.get("finite_sample_count", -1))
    nonfinite_count = int(samples.get("nonfinite_sample_count", -1))
    if finite_count != int(settings.num_results) or nonfinite_count != 0:
        vetoes.append("hmc_sample_finiteness_mismatch")
    if not _is_zero_initial_state(row.get("initial", {}).get("u_new")):
        vetoes.append("hmc_initial_state_not_zero")
    acceptance = row.get("acceptance_rate")
    if acceptance is None or not np.isfinite(float(acceptance)):
        vetoes.append("acceptance_missing_or_nonfinite")
    elif not (
        float(settings.acceptance_lower_exclusive)
        < float(acceptance)
        < float(settings.acceptance_upper_exclusive)
    ):
        vetoes.append("acceptance_outside_envelope")
    native = row.get("trace_summary", {}).get("native_divergence", {})
    if isinstance(native, Mapping) and native.get("available") is True and int(native.get("count", 0)) > 0:
        vetoes.append("native_divergence_detected")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "decision": decision,
        "selected_kernel": selected,
        "hmc_moments": {
            "mean_u_new": hmc_mean,
            "std_u_new": hmc_std,
            "finite_sample_count": finite_count,
            "nonfinite_sample_count": nonfinite_count,
        },
        "acceptance_rate": acceptance,
        "native_divergence": native,
    }


def generate_antithetic_standard_normal_proposal(
    settings: Phase2WReferenceSettings,
    *,
    dim: int = 4,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    rng = np.random.default_rng(settings.seed)
    base = rng.standard_normal(size=(settings.proposal_sample_count // 2, dim))
    samples = np.vstack([base, -base])
    if samples.shape != (settings.proposal_sample_count, dim):
        vetoes.append("proposal_shape_mismatch")
    if not np.all(np.isfinite(samples)):
        vetoes.append("proposal_nonfinite_values")
    antithetic_sum_max_abs = float(np.max(np.abs(samples[: base.shape[0]] + samples[base.shape[0] :])))
    if antithetic_sum_max_abs > 0.0:
        vetoes.append("proposal_antithetic_pairing_failed")
    log_proposal = standard_normal_log_prob(samples)
    if log_proposal.shape != (settings.proposal_sample_count,):
        vetoes.append("proposal_log_density_shape_mismatch")
    if not np.all(np.isfinite(log_proposal)):
        vetoes.append("proposal_log_density_nonfinite")
    return json_ready(
        {
            "generated": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "seed": settings.seed,
            "sample_count": int(settings.proposal_sample_count),
            "dimension": int(dim),
            "samples": samples,
            "proposal_log_prob": log_proposal,
            "antithetic_pairing": {
                "base_sample_count": int(base.shape[0]),
                "max_abs_pair_sum": antithetic_sum_max_abs,
            },
            "sample_summary": {
                "mean": np.mean(samples, axis=0),
                "std": np.std(samples, axis=0),
                "min": np.min(samples, axis=0),
                "max": np.max(samples, axis=0),
                "max_abs": float(np.max(np.abs(samples))),
            },
        }
    )


def standard_normal_log_prob(samples: Any) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError("samples must be a rank-2 array")
    dim = int(array.shape[1])
    return -0.5 * np.sum(np.square(array), axis=1) - 0.5 * dim * np.log(2.0 * np.pi)


def compute_importance_reference(
    adapter: Any,
    proposal: Mapping[str, Any],
    settings: Phase2WReferenceSettings,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    samples = np.asarray(proposal.get("samples"), dtype=float)
    proposal_log_prob = np.asarray(proposal.get("proposal_log_prob"), dtype=float)
    if samples.shape != (settings.proposal_sample_count, 4):
        vetoes.append("proposal_samples_shape_mismatch")
    if proposal_log_prob.shape != (settings.proposal_sample_count,):
        vetoes.append("proposal_log_prob_shape_mismatch")
    if not np.all(np.isfinite(samples)):
        vetoes.append("proposal_samples_nonfinite")
    if not np.all(np.isfinite(proposal_log_prob)):
        vetoes.append("proposal_log_prob_nonfinite")
    target_log_prob = np.full((settings.proposal_sample_count,), np.nan, dtype=float)
    target_score_norm = np.full((settings.proposal_sample_count,), np.nan, dtype=float)
    first_error = None
    if not vetoes:
        for index, sample in enumerate(samples):
            try:
                value, score = adapter.log_prob_and_grad(tf.constant(sample, dtype=tf.float64))
                target_log_prob[index] = float(tf.convert_to_tensor(value, dtype=tf.float64).numpy())
                score_np = np.asarray(
                    tf.reshape(tf.convert_to_tensor(score, dtype=tf.float64), [-1]).numpy(),
                    dtype=float,
                )
                if score_np.shape != (4,):
                    vetoes.append("target_score_shape_mismatch")
                    break
                target_score_norm[index] = float(np.linalg.norm(score_np))
            except Exception as exc:  # noqa: BLE001 - fail-closed reference diagnostic.
                first_error = f"{type(exc).__name__}: {exc}"
                vetoes.append("target_evaluation_exception")
                break
    if not np.all(np.isfinite(target_log_prob)):
        vetoes.append("target_log_prob_nonfinite")
    if not np.all(np.isfinite(target_score_norm)):
        vetoes.append("target_score_norm_nonfinite")
    if vetoes:
        return json_ready(
            {
                "computed": False,
                "reference_valid": False,
                "vetoes": tuple(dict.fromkeys(vetoes)),
                "first_error": first_error,
                "target_log_prob": target_log_prob,
                "proposal_log_prob": proposal_log_prob,
            }
        )

    log_weights = target_log_prob - proposal_log_prob
    if not np.all(np.isfinite(log_weights)):
        vetoes.append("log_weight_nonfinite")
    log_weight_shift = float(np.max(log_weights))
    shifted = np.exp(log_weights - log_weight_shift)
    weight_sum = float(np.sum(shifted))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        vetoes.append("normalized_weight_degeneracy")
        normalized = np.full_like(shifted, np.nan)
    else:
        normalized = shifted / weight_sum
    if not np.all(np.isfinite(normalized)) or np.any(normalized < 0.0):
        vetoes.append("normalized_weight_invalid")
    if not np.isclose(float(np.sum(normalized)), 1.0, rtol=1.0e-10, atol=1.0e-10):
        vetoes.append("normalized_weight_sum_invalid")
    ess = float(1.0 / np.sum(np.square(normalized))) if not vetoes else float("nan")
    ess_ratio = float(ess / settings.proposal_sample_count) if np.isfinite(ess) else float("nan")
    if not np.isfinite(ess) or ess < settings.reference_ess_min:
        vetoes.append("reference_ess_below_threshold")
    if not np.isfinite(ess_ratio) or ess_ratio < settings.reference_ess_ratio_min:
        vetoes.append("reference_ess_ratio_below_threshold")

    mean = np.sum(normalized[:, np.newaxis] * samples, axis=0)
    centered = samples - mean
    second_moment_variance = np.sum(normalized[:, np.newaxis] * np.square(centered), axis=0)
    std = np.sqrt(second_moment_variance)
    mean_mcse = np.sqrt(second_moment_variance / ess)
    reference_valid = bool(not vetoes)
    return json_ready(
        {
            "computed": True,
            "reference_valid": reference_valid,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "target_log_prob_route": "adapter.log_prob_and_grad_values_only",
            "target_log_prob": target_log_prob,
            "target_score_norm_summary": _finite_summary(target_score_norm),
            "proposal_log_prob": proposal_log_prob,
            "log_weight_summary": _finite_summary(log_weights),
            "log_weight_shift": log_weight_shift,
            "weight_summary": {
                "min": float(np.min(normalized)),
                "max": float(np.max(normalized)),
                "sum": float(np.sum(normalized)),
                "nonzero_count": int(np.sum(normalized > 0.0)),
            },
            "ess": ess,
            "ess_ratio": ess_ratio,
            "mean_u_new": mean,
            "std_u_new": std,
            "second_moment_variance_u_new": second_moment_variance,
            "mean_mcse_u_new": mean_mcse,
            "sample_count": int(settings.proposal_sample_count),
            "dimension": 4,
            "nonclaims": (
                "self-normalized importance reference only",
                "not exact posterior truth",
                "not HMC convergence evidence",
            ),
        }
    )


def compare_hmc_to_reference(
    phase2v_payload: Mapping[str, Any],
    reference: Mapping[str, Any],
    settings: Phase2WReferenceSettings,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    if reference.get("reference_valid") is not True:
        return {
            "evaluated": False,
            "passed": False,
            "vetoes": ("reference_invalid_agreement_not_interpreted",),
        }
    samples = (phase2v_payload.get("selected_kernel_row") or {}).get("samples_summary") or {}
    hmc_mean = _vector_or_none(samples.get("mean_u_new"), 4)
    hmc_std = _vector_or_none(samples.get("std_u_new"), 4)
    reference_mean = _vector_or_none(reference.get("mean_u_new"), 4)
    reference_std = _vector_or_none(reference.get("std_u_new"), 4)
    reference_mean_mcse = _vector_or_none(reference.get("mean_mcse_u_new"), 4)
    if hmc_mean is None:
        vetoes.append("hmc_mean_missing_or_nonfinite")
    if hmc_std is None or np.any(hmc_std <= 0.0):
        vetoes.append("hmc_std_missing_nonfinite_or_nonpositive")
    if reference_mean is None:
        vetoes.append("reference_mean_missing_or_nonfinite")
    if reference_std is None or np.any(reference_std <= 0.0):
        vetoes.append("reference_std_missing_nonfinite_or_nonpositive")
    if reference_mean_mcse is None or np.any(reference_mean_mcse < 0.0):
        vetoes.append("reference_mean_mcse_missing_nonfinite_or_negative")
    if vetoes:
        return {
            "evaluated": False,
            "passed": False,
            "vetoes": tuple(dict.fromkeys(vetoes)),
        }
    mean_abs_delta = np.abs(hmc_mean - reference_mean)
    mean_threshold = np.maximum(
        settings.mean_abs_floor,
        settings.mean_mcse_multiplier * reference_mean_mcse,
    )
    mean_pass = mean_abs_delta <= mean_threshold
    std_ratio = hmc_std / reference_std
    std_pass = (settings.std_ratio_lower <= std_ratio) & (std_ratio <= settings.std_ratio_upper)
    for index, passed in enumerate(mean_pass):
        if not bool(passed):
            vetoes.append(f"hmc_mean_component_{index}_outside_threshold")
    for index, passed in enumerate(std_pass):
        if not bool(passed):
            vetoes.append(f"hmc_std_component_{index}_ratio_outside_interval")
    return json_ready(
        {
            "evaluated": True,
            "passed": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "hmc_mean_u_new": hmc_mean,
            "reference_mean_u_new": reference_mean,
            "mean_abs_delta": mean_abs_delta,
            "reference_mean_mcse_u_new": reference_mean_mcse,
            "mean_threshold": mean_threshold,
            "mean_component_pass": mean_pass,
            "hmc_std_u_new": hmc_std,
            "reference_std_u_new": reference_std,
            "std_ratio": std_ratio,
            "std_ratio_interval": (settings.std_ratio_lower, settings.std_ratio_upper),
            "std_component_pass": std_pass,
            "interpretation_boundary": (
                "agreement screen is interpreted only because reference_valid is true; "
                "it remains a local diagnostic, not posterior certification"
            ),
        }
    )


def telemetry_policy_payload(phase2v_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    row = phase2v_payload.get("selected_kernel_row") or {}
    native = row.get("trace_summary", {}).get("native_divergence", {})
    if isinstance(native, Mapping) and native.get("available") is True:
        positive_count = int(native.get("count", 0))
        available_count = 1
        unavailable_count = 0
        statuses = ("available",)
    else:
        positive_count = 0
        available_count = 0
        unavailable_count = 1 if row else 0
        statuses = (str(native.get("status", "unavailable")) if isinstance(native, Mapping) else "unavailable",)
    if positive_count > 0:
        interpretation = "positive native divergence detected"
    elif available_count > 0:
        interpretation = "native divergence available with zero positive indicators"
    else:
        interpretation = "native divergence unavailable; unavailable is not zero divergences"
    return {
        "native_divergence_statuses": statuses,
        "native_divergence_available_count": available_count,
        "native_divergence_unavailable_count": unavailable_count,
        "native_divergence_positive_count": positive_count,
        "native_divergence_interpretation": interpretation,
        "zero_divergence_claim_made": False,
        "unavailable_native_divergence_is_zero_divergence": False,
        "log_accept_threshold_used_as_native_divergence": False,
    }


def _vector_or_none(value: Any, dim: int) -> np.ndarray | None:
    array = np.asarray(value, dtype=float)
    if array.shape != (dim,) or not np.all(np.isfinite(array)):
        return None
    return array


def _is_zero_initial_state(value: Any) -> bool:
    array = np.asarray(value, dtype=float)
    return bool(array.shape == (4,) and np.all(np.isfinite(array)) and np.all(array == 0.0))


def _finite_summary(values: Any) -> Mapping[str, Any]:
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
    reference = payload.get("importance_reference", {})
    agreement = payload.get("hmc_reference_agreement", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2W - Importance Reference Agreement",
        "",
        "## Decision",
        "",
        f"- phase2w_importance_reference_agreement_passed: `{decision['phase2w_importance_reference_agreement_passed']}`",
        f"- reference_valid: `{decision['reference_valid']}`",
        f"- agreement_passed: `{decision['agreement_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- reference_ess: `{decision['reference_ess']}`",
        f"- reference_ess_ratio: `{decision['reference_ess_ratio']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Reference",
        "",
        f"- valid: `{reference.get('reference_valid')}`",
        f"- vetoes: `{reference.get('vetoes')}`",
        f"- mean_u_new: `{reference.get('mean_u_new')}`",
        f"- std_u_new: `{reference.get('std_u_new')}`",
        f"- mean_mcse_u_new: `{reference.get('mean_mcse_u_new')}`",
        f"- weight_summary: `{reference.get('weight_summary')}`",
        f"- log_weight_summary: `{reference.get('log_weight_summary')}`",
        "",
        "## HMC Agreement",
        "",
        f"- evaluated: `{agreement.get('evaluated')}`",
        f"- passed: `{agreement.get('passed')}`",
        f"- vetoes: `{agreement.get('vetoes')}`",
        f"- mean_abs_delta: `{agreement.get('mean_abs_delta')}`",
        f"- mean_threshold: `{agreement.get('mean_threshold')}`",
        f"- std_ratio: `{agreement.get('std_ratio')}`",
        "",
        "## Inference Status",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
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
        ]
    )
    lines.extend(["", "## Review Record", ""])
    review = payload.get("review_record", {})
    lines.append(f"- reviewer: {review.get('reviewer')}")
    lines.append(f"- review_strength: {review.get('review_strength')}")
    lines.append(f"- claude_status: {review.get('claude_status')}")
    lines.extend(["", "## Nonclaims", ""])
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
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_phase2w_importance_reference_agreement(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
        load_json(args.phase2v_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
        "phase2v_json": str(args.phase2v_json),
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
