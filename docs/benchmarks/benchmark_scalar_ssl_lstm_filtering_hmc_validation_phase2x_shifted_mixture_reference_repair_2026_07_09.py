"""Phase 2X shifted-mixture MAP-local importance-reference repair.

This diagnostic repairs the Phase 2W standard-normal reference failure by using
a predeclared shifted diagonal mixture proposal in the Phase 2S/2U MAP-local
``u_new`` coordinate.  Proposal parameters come from Phase 2W pilot reference
diagnostics only, not Phase 2V HMC moments.  HMC agreement is interpreted only
if the repaired reference validity gates pass.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PHASE2W_MODULE_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py"
)
SCRIPT_NAME = (
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_"
    "shifted_mixture_reference_repair_2026_07_09.py"
)
SCHEMA_VERSION = "scalar_ssl_lstm.filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.v1"
PLAN_PATH = "docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md"
SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md"
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
DEFAULT_JSON_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json"
)
DEFAULT_MARKDOWN_PATH = (
    ROOT
    / "docs/benchmarks/"
    "scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md"
)
PHASE2W_REFERENCE_MEAN = np.asarray(
    [
        0.16900152112527375,
        0.34590014590251295,
        0.47216707577215133,
        -0.3362900480743778,
    ],
    dtype=float,
)
PHASE2W_REFERENCE_STD = np.asarray(
    [
        1.1289232726542155,
        1.3947178163994365,
        1.7877962561383989,
        1.7764811837333756,
    ],
    dtype=float,
)
SHIFTED_SCALE = np.clip(1.25 * PHASE2W_REFERENCE_STD, 0.75, 3.0)
NONCLAIMS = (
    "Phase 2X shifted-mixture importance-reference repair diagnostic only",
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


def load_phase2w_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2w_for_phase2x",
        PHASE2W_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load Phase 2W harness module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase2w = load_phase2w_module()


class Phase2XReferenceSettings(phase2w.Phase2WReferenceSettings):
    """Fixed Phase 2X shifted-mixture reference settings."""

    def __init__(
        self,
        *,
        proposal_sample_count: int = 2048,
        seed: tuple[int, int] = (20260709, 6601),
        reference_ess_min: float = 256.0,
        reference_ess_ratio_min: float = 0.125,
        mean_abs_floor: float = 0.75,
        mean_mcse_multiplier: float = 4.0,
        std_ratio_lower: float = 0.5,
        std_ratio_upper: float = 2.0,
        standard_component_weight: float = 0.25,
        shifted_component_weight: float = 0.75,
        scale_multiplier: float = 1.25,
        scale_clip_lower: float = 0.75,
        scale_clip_upper: float = 3.0,
    ) -> None:
        super().__init__(
            proposal_sample_count=proposal_sample_count,
            seed=seed,
            reference_ess_min=reference_ess_min,
            reference_ess_ratio_min=reference_ess_ratio_min,
            mean_abs_floor=mean_abs_floor,
            mean_mcse_multiplier=mean_mcse_multiplier,
            std_ratio_lower=std_ratio_lower,
            std_ratio_upper=std_ratio_upper,
        )
        weights = (float(standard_component_weight), float(shifted_component_weight))
        if not np.all(np.isfinite(weights)) or any(weight <= 0.0 for weight in weights):
            raise ValueError("mixture weights must be positive finite")
        if abs(sum(weights) - 1.0) > 1.0e-12:
            raise ValueError("mixture weights must sum to one")
        for name, value in (
            ("standard_component_weight", weights[0]),
            ("shifted_component_weight", weights[1]),
            ("scale_multiplier", scale_multiplier),
            ("scale_clip_lower", scale_clip_lower),
            ("scale_clip_upper", scale_clip_upper),
        ):
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not (0.0 < self.scale_clip_lower <= self.scale_clip_upper):
            raise ValueError("scale clipping bounds must satisfy 0 < lower <= upper")
        if int(self.proposal_sample_count * self.standard_component_weight) % 2 != 0:
            raise ValueError("standard component sample count must be even")
        if int(self.proposal_sample_count * self.shifted_component_weight) % 2 != 0:
            raise ValueError("shifted component sample count must be even")

    @property
    def standard_component_count(self) -> int:
        return int(round(self.proposal_sample_count * self.standard_component_weight))

    @property
    def shifted_component_count(self) -> int:
        return int(self.proposal_sample_count - self.standard_component_count)

    def payload(self) -> Mapping[str, Any]:
        base = dict(super().payload())
        base.update(
            {
                "proposal_distribution": (
                    "0.25 * N(0, I4) + 0.75 * N(phase2w_pilot_center, diag(shifted_scale^2))"
                ),
                "proposal_sample_count": self.proposal_sample_count,
                "standard_component_weight": self.standard_component_weight,
                "shifted_component_weight": self.shifted_component_weight,
                "standard_component_count": self.standard_component_count,
                "shifted_component_count": self.shifted_component_count,
                "seed": self.seed,
                "phase2w_pilot_center": PHASE2W_REFERENCE_MEAN,
                "phase2w_pilot_std": PHASE2W_REFERENCE_STD,
                "scale_multiplier": self.scale_multiplier,
                "scale_clip_lower": self.scale_clip_lower,
                "scale_clip_upper": self.scale_clip_upper,
                "shifted_scale": shifted_scale_from_settings(self),
                "proposal_parameter_source": "phase2w_importance_reference_pilot_only_not_hmc_moments",
                "reference_ess_min": self.reference_ess_min,
                "reference_ess_ratio_min": self.reference_ess_ratio_min,
            }
        )
        return phase2w.json_ready(base)


def shifted_scale_from_settings(settings: Phase2XReferenceSettings) -> np.ndarray:
    return np.clip(
        float(settings.scale_multiplier) * PHASE2W_REFERENCE_STD,
        float(settings.scale_clip_lower),
        float(settings.scale_clip_upper),
    )


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase2x_shifted_mixture_reference_repair(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    settings: Phase2XReferenceSettings | None = None,
) -> Mapping[str, Any]:
    cfg = Phase2XReferenceSettings() if settings is None else settings
    start = time.perf_counter()
    precondition = validate_phase2x_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        phase2w_payload,
        cfg,
    )
    adapter = None
    adapter_audit: Mapping[str, Any] = {"built": False, "vetoes": ()}
    proposal: Mapping[str, Any] = {
        "generated": False,
        "vetoes": ("phase2x_reference_not_run",),
    }
    reference: Mapping[str, Any] = {
        "computed": False,
        "vetoes": ("phase2x_reference_not_run",),
    }
    agreement: Mapping[str, Any] = {
        "evaluated": False,
        "vetoes": ("phase2x_agreement_not_evaluated",),
    }
    vetoes = list(precondition.get("vetoes", ()))
    if not vetoes:
        phase2u_module = phase2w.load_phase2u_module()
        adapter, adapter_audit = phase2u_module.build_phase2u_adapter(phase2s_payload)
        vetoes.extend(adapter_audit.get("vetoes", ()))
        if adapter is None:
            vetoes.append("phase2x_adapter_not_built")

    if adapter is not None and not adapter_audit.get("vetoes"):
        proposal = generate_shifted_mixture_proposal(cfg)
        vetoes.extend(proposal.get("vetoes", ()))
        if not proposal.get("vetoes"):
            reference = phase2w.compute_importance_reference(adapter, proposal, cfg)
            vetoes.extend(reference.get("vetoes", ()))
        if not reference.get("vetoes"):
            agreement = phase2w.compare_hmc_to_reference(phase2v_payload, reference, cfg)
            vetoes.extend(agreement.get("vetoes", ()))

    unique_vetoes = tuple(dict.fromkeys(vetoes))
    reference_valid = bool(reference.get("reference_valid") is True and not reference.get("vetoes"))
    agreement_passed = bool(agreement.get("passed") is True and not agreement.get("vetoes"))
    passed = bool(not unique_vetoes and reference_valid and agreement_passed)
    payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_role": "cpu_hidden_scalar_filtering_hmc_phase2x_shifted_mixture_reference_repair",
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
                "phase2w_json": str(DEFAULT_PHASE2W_PATH.relative_to(ROOT)),
            },
            "precondition": precondition,
            "adapter_audit": adapter_audit,
            "proposal": proposal,
            "importance_reference": reference,
            "hmc_reference_agreement": agreement,
            "telemetry_policy": phase2w.telemetry_policy_payload(phase2v_payload),
            "decision": {
                "phase2x_shifted_mixture_reference_repair_passed": passed,
                "reference_valid": reference_valid,
                "agreement_passed": agreement_passed,
                "vetoes": unique_vetoes,
                "reference_ess": reference.get("ess"),
                "reference_ess_ratio": reference.get("ess_ratio"),
                "zero_divergence_claim_made": False,
                "viable_for_reference_replication_subplan": passed,
                "viable_for_phase3_gpu_xla_subplan": False,
                "next_justified_action": (
                    "write Phase 2X result and draft/review independent shifted-mixture reference replication subplan"
                    if passed
                    else "write Phase 2X blocker or narrower reference/tuning/localization repair result"
                ),
            },
            "metric_roles": {
                "phase2x_shifted_mixture_reference_repair_passed": "primary_phase2x_pass_fail",
                "phase2w_reference_failure_boundary": "hard_veto_evidence",
                "proposal_parameter_source": "hard_veto_evidence_no_hmc_moment_tuning",
                "reference_ess": "reference_validity_veto",
                "reference_ess_ratio": "reference_validity_veto",
                "hmc_mean_delta_threshold": "phase2x_agreement_gate_after_reference_validity",
                "hmc_std_ratio_interval": "phase2x_agreement_gate_after_reference_validity",
                "native_divergence": "hard_veto_if_available_positive; unavailable is not zero divergences",
                "runtime": "explanatory_only",
            },
            "inference_status": {
                "hard_veto_screen": "passed" if passed else "failed",
                "reference_validity": "passed" if reference_valid else "failed",
                "hmc_reference_agreement": (
                    "passed" if agreement_passed else "failed or not interpreted"
                ),
                "native_divergence": phase2w.telemetry_policy_payload(phase2v_payload)[
                    "native_divergence_interpretation"
                ],
                "zero_divergence_claim": "not made",
                "statistically_supported_ranking": (
                    "none; one HMC chain compared to one repaired importance reference with no ranking"
                ),
                "descriptive_only_differences": (
                    "reference ESS, weighted moments, HMC-reference deltas, std ratios, and runtime"
                ),
                "posterior_correctness": "not assessed",
                "hmc_readiness": "not assessed; Phase 2X is a narrow reference-repair diagnostic",
                "gpu_xla_readiness": "blocked",
                "default_readiness": "not assessed",
                "next_evidence_needed": (
                    "reviewed independent shifted-mixture reference replication"
                    if passed
                    else "reviewed reference/tuning/localization repair"
                ),
            },
            "decision_table": {
                "decision": "Phase 2X shifted-mixture importance-reference repair diagnostic",
                "primary_criterion_status": "passed" if passed else "failed",
                "veto_diagnostic_status": "no vetoes" if passed else f"vetoes: {unique_vetoes}",
                "main_uncertainty": (
                    "A repaired proposal can validate local moment agreement only after "
                    "fresh ESS gates pass; it remains a diagnostic, not posterior certification."
                ),
                "next_justified_action": (
                    "draft/review independent shifted-mixture reference replication subplan"
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
                    "CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 600 python "
                    "docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py "
                    "--json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json "
                    "--markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md"
                ),
                "git": phase2w.git_payload(),
                "environment": phase2w.environment_payload(),
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
            "review_record": {
                "reviewer": "Codex substitute reviewer",
                "review_strength": "weaker_than_full_claude_material_review",
                "claude_status": "unavailable_for_repo_context_material_review_per_prior_handoff",
                "rounds": (
                    {
                        "round": 1,
                        "verdict": "AGREE",
                        "findings": ("no blocking findings",),
                    },
                ),
            },
            "nonclaims": NONCLAIMS,
        }
    return phase2w.json_ready(payload)


def validate_phase2x_handoff(
    phase2s_payload: Mapping[str, Any],
    phase2t_payload: Mapping[str, Any],
    phase2u_payload: Mapping[str, Any],
    phase2v_payload: Mapping[str, Any],
    phase2w_payload: Mapping[str, Any],
    settings: Phase2XReferenceSettings,
) -> Mapping[str, Any]:
    vetoes: list[str] = []
    phase2w_precondition = phase2w.validate_phase2w_handoff(
        phase2s_payload,
        phase2t_payload,
        phase2u_payload,
        phase2v_payload,
        phase2w.Phase2WReferenceSettings(),
    )
    vetoes.extend(f"phase2w_precondition_{item}" for item in phase2w_precondition.get("vetoes", ()))
    if (
        phase2w_payload.get("schema_version")
        != "scalar_ssl_lstm.filtering_hmc_validation_phase2w_importance_reference_agreement.v1"
    ):
        vetoes.append("phase2w_schema_mismatch")
    decision = phase2w_payload.get("decision", {})
    expected_vetoes = {
        "reference_ess_below_threshold",
        "reference_ess_ratio_below_threshold",
    }
    actual_vetoes = set(str(item) for item in decision.get("vetoes", ()))
    if decision.get("phase2w_importance_reference_agreement_passed") is not False:
        vetoes.append("phase2w_decision_not_failed")
    if actual_vetoes != expected_vetoes:
        vetoes.append("phase2w_failure_not_limited_to_ess_thresholds")
    reference = phase2w_payload.get("importance_reference", {})
    if reference.get("reference_valid") is not False:
        vetoes.append("phase2w_reference_unexpectedly_valid")
    if int((reference.get("log_weight_summary") or {}).get("finite_count", -1)) != 1024:
        vetoes.append("phase2w_log_weight_finite_count_mismatch")
    if int((reference.get("log_weight_summary") or {}).get("nonfinite_count", -1)) != 0:
        vetoes.append("phase2w_log_weight_nonfinite_count_nonzero")
    if int((reference.get("target_score_norm_summary") or {}).get("nonfinite_count", -1)) != 0:
        vetoes.append("phase2w_target_score_nonfinite_count_nonzero")
    agreement = phase2w_payload.get("hmc_reference_agreement", {})
    if agreement.get("evaluated") is not False:
        vetoes.append("phase2w_agreement_was_interpreted")
    pilot_mean = phase2w._vector_or_none(reference.get("mean_u_new"), 4)
    pilot_std = phase2w._vector_or_none(reference.get("std_u_new"), 4)
    if pilot_mean is None or not np.allclose(pilot_mean, PHASE2W_REFERENCE_MEAN, rtol=0.0, atol=1.0e-12):
        vetoes.append("phase2w_pilot_mean_mismatch")
    if pilot_std is None or not np.allclose(pilot_std, PHASE2W_REFERENCE_STD, rtol=0.0, atol=1.0e-12):
        vetoes.append("phase2w_pilot_std_mismatch")
    if int(settings.proposal_sample_count) != 2048:
        vetoes.append("proposal_sample_count_not_predeclared_2048")
    if tuple(settings.seed) != (20260709, 6601):
        vetoes.append("proposal_seed_not_predeclared")
    scale = shifted_scale_from_settings(settings)
    if not np.allclose(scale, SHIFTED_SCALE, rtol=0.0, atol=1.0e-12):
        vetoes.append("shifted_scale_mismatch")
    return {
        "passed": not vetoes,
        "vetoes": tuple(dict.fromkeys(vetoes)),
        "phase2w_precondition": phase2w_precondition,
        "phase2w_decision": decision,
        "phase2w_reference_summary": {
            "ess": reference.get("ess"),
            "ess_ratio": reference.get("ess_ratio"),
            "vetoes": reference.get("vetoes"),
            "mean_u_new": pilot_mean,
            "std_u_new": pilot_std,
        },
        "proposal_parameter_contract": {
            "uses_hmc_moments": False,
            "pilot_source": "phase2w_importance_reference_weighted_moments_only",
            "pilot_center": PHASE2W_REFERENCE_MEAN,
            "pilot_std": PHASE2W_REFERENCE_STD,
            "shifted_scale": scale,
        },
        "settings_contract": settings.payload(),
    }


def generate_shifted_mixture_proposal(settings: Phase2XReferenceSettings) -> Mapping[str, Any]:
    vetoes: list[str] = []
    rng = np.random.default_rng(settings.seed)
    dim = 4
    standard_count = settings.standard_component_count
    shifted_count = settings.shifted_component_count
    standard_base = rng.standard_normal(size=(standard_count // 2, dim))
    shifted_base = rng.standard_normal(size=(shifted_count // 2, dim))
    standard_samples = np.vstack([standard_base, -standard_base])
    shifted_scale = shifted_scale_from_settings(settings)
    shifted_offsets = np.vstack([shifted_base, -shifted_base])
    shifted_samples = PHASE2W_REFERENCE_MEAN + shifted_offsets * shifted_scale
    samples = np.vstack([standard_samples, shifted_samples])
    component = np.array(["standard"] * standard_count + ["shifted"] * shifted_count)
    if samples.shape != (settings.proposal_sample_count, dim):
        vetoes.append("proposal_shape_mismatch")
    if not np.all(np.isfinite(samples)):
        vetoes.append("proposal_nonfinite_values")
    if not np.all(np.isfinite(shifted_scale)) or np.any(shifted_scale <= 0.0):
        vetoes.append("shifted_scale_invalid")
    standard_pair_max = float(np.max(np.abs(standard_samples[: standard_count // 2] + standard_samples[standard_count // 2 :])))
    shifted_pair_max = float(
        np.max(
            np.abs(
                (shifted_samples[: shifted_count // 2] - PHASE2W_REFERENCE_MEAN)
                + (shifted_samples[shifted_count // 2 :] - PHASE2W_REFERENCE_MEAN)
            )
        )
    )
    if standard_pair_max > 0.0:
        vetoes.append("standard_component_antithetic_pairing_failed")
    if shifted_pair_max > 1.0e-12:
        vetoes.append("shifted_component_antithetic_pairing_failed")
    proposal_log_prob = shifted_mixture_log_prob(samples, settings)
    if proposal_log_prob.shape != (settings.proposal_sample_count,):
        vetoes.append("proposal_log_density_shape_mismatch")
    if not np.all(np.isfinite(proposal_log_prob)):
        vetoes.append("proposal_log_density_nonfinite")
    return phase2w.json_ready(
        {
            "generated": not vetoes,
            "vetoes": tuple(dict.fromkeys(vetoes)),
            "seed": settings.seed,
            "sample_count": int(settings.proposal_sample_count),
            "dimension": dim,
            "component_counts": {
                "standard": int(standard_count),
                "shifted": int(shifted_count),
            },
            "component_weights": {
                "standard": settings.standard_component_weight,
                "shifted": settings.shifted_component_weight,
            },
            "component": component,
            "samples": samples,
            "proposal_log_prob": proposal_log_prob,
            "pilot_center": PHASE2W_REFERENCE_MEAN,
            "pilot_std": PHASE2W_REFERENCE_STD,
            "shifted_scale": shifted_scale,
            "antithetic_pairing": {
                "standard_max_abs_pair_sum": standard_pair_max,
                "shifted_centered_max_abs_pair_sum": shifted_pair_max,
            },
            "sample_summary": {
                "mean": np.mean(samples, axis=0),
                "std": np.std(samples, axis=0),
                "min": np.min(samples, axis=0),
                "max": np.max(samples, axis=0),
                "max_abs": float(np.max(np.abs(samples))),
            },
            "proposal_log_prob_summary": phase2w._finite_summary(proposal_log_prob),
            "proposal_parameter_source": "phase2w_importance_reference_pilot_only_not_hmc_moments",
        }
    )


def shifted_mixture_log_prob(samples: Any, settings: Phase2XReferenceSettings) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("samples must have shape (n, 4)")
    log_standard_weight = np.log(settings.standard_component_weight)
    log_shifted_weight = np.log(settings.shifted_component_weight)
    log_standard = phase2w.standard_normal_log_prob(array)
    shifted_scale = shifted_scale_from_settings(settings)
    shifted_z = (array - PHASE2W_REFERENCE_MEAN) / shifted_scale
    log_shifted = (
        -0.5 * np.sum(np.square(shifted_z), axis=1)
        - np.sum(np.log(shifted_scale))
        - 0.5 * 4 * np.log(2.0 * np.pi)
    )
    a = log_standard_weight + log_standard
    b = log_shifted_weight + log_shifted
    max_ab = np.maximum(a, b)
    return max_ab + np.log(np.exp(a - max_ab) + np.exp(b - max_ab))


def render_markdown(payload: Mapping[str, Any]) -> str:
    decision = payload["decision"]
    reference = payload.get("importance_reference", {})
    agreement = payload.get("hmc_reference_agreement", {})
    proposal = payload.get("proposal", {})
    lines = [
        "# Scalar SSL-LSTM Filtering HMC Validation Phase 2X - Shifted-Mixture Reference Repair",
        "",
        "## Decision",
        "",
        f"- phase2x_shifted_mixture_reference_repair_passed: `{decision['phase2x_shifted_mixture_reference_repair_passed']}`",
        f"- reference_valid: `{decision['reference_valid']}`",
        f"- agreement_passed: `{decision['agreement_passed']}`",
        f"- vetoes: `{decision['vetoes']}`",
        f"- reference_ess: `{decision['reference_ess']}`",
        f"- reference_ess_ratio: `{decision['reference_ess_ratio']}`",
        f"- zero_divergence_claim_made: `{decision['zero_divergence_claim_made']}`",
        f"- next_justified_action: {decision['next_justified_action']}",
        "",
        "## Proposal",
        "",
        f"- component_counts: `{proposal.get('component_counts')}`",
        f"- component_weights: `{proposal.get('component_weights')}`",
        f"- pilot_center: `{proposal.get('pilot_center')}`",
        f"- shifted_scale: `{proposal.get('shifted_scale')}`",
        f"- proposal_log_prob_summary: `{proposal.get('proposal_log_prob_summary')}`",
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
    lines.extend(["", "## Nonclaims", ""])
    lines.extend(f"- {item}" for item in payload["nonclaims"])
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase2s-json", type=Path, default=DEFAULT_PHASE2S_PATH)
    parser.add_argument("--phase2t-json", type=Path, default=DEFAULT_PHASE2T_PATH)
    parser.add_argument("--phase2u-json", type=Path, default=DEFAULT_PHASE2U_PATH)
    parser.add_argument("--phase2v-json", type=Path, default=DEFAULT_PHASE2V_PATH)
    parser.add_argument("--phase2w-json", type=Path, default=DEFAULT_PHASE2W_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--markdown-path", type=Path, default=DEFAULT_MARKDOWN_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    payload = run_phase2x_shifted_mixture_reference_repair(
        load_json(args.phase2s_json),
        load_json(args.phase2t_json),
        load_json(args.phase2u_json),
        load_json(args.phase2v_json),
        load_json(args.phase2w_json),
    )
    payload["source_artifacts"] = {
        "phase2s_json": str(args.phase2s_json),
        "phase2t_json": str(args.phase2t_json),
        "phase2u_json": str(args.phase2u_json),
        "phase2v_json": str(args.phase2v_json),
        "phase2w_json": str(args.phase2w_json),
    }
    payload["run_manifest"]["wall_time_seconds"] = float(time.perf_counter() - start)
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(phase2w.json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
