"""Windowed HMC preparation and frozen geometry/start-bank handoff.

This module owns the operational and historical diagnostic windowed stages.
Preparation does not issue candidate-set tuning authority. Public and historical
imports share these definitions; numerical policies and seed-site IDs are stable.
"""
from __future__ import annotations

from bayesfilter.inference.hmc_preparation_recovery import validate_preparation_max_restarts

import dataclasses
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping
from bayesfilter.inference.hmc_preparation_common import (
    _validate_seed,
    _string_tuple,
    _mass_artifact_signature,
    _seed_from_mapping,
    _int_or_none,
    _json_ready,
    _runtime_seconds_or_none,
)
from bayesfilter.inference.hmc_bootstrap import (
    RunFullChainFn,
    PrivateTuningDiagnosticCallback,
    HMCBootstrapScreenResult,
    _BootstrapFixedMassLatentValueScoreAdapter,
    _build_bootstrap_fixed_mass_adapter,
)
from bayesfilter.inference.hmc_geometry import (
    HMCGeometryInitializationResult,
    _GEOMETRY_MAX_LEAPFROG,
    _derive_seed,
)
from bayesfilter.hmc_route_contract import (
    HMC_WINDOWED_MASS_STAGE,
    LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID,
    ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID,
    OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
    HMCAlgorithmRouteDecision,
    require_hmc_algorithm_route,
)
from bayesfilter.hmc_budget_contract import (
    BROAD_FIXED_METRIC_OPERATIONAL_ROUTE,
    HMCOperationalStatisticalWorkPolicy,
    build_public_hmc_work_manifest,
    reconcile_executed_hmc_work,
)
from bayesfilter.inference.hmc import (
    FixedSizeHMCChunkConfig,
    FixedSizeHMCChunkRunResult,
    FullChainHMCConfig,
    FullChainHMCRunResult,
    PrecomputedMassArtifact,
    SequentialRHatCheckpointWriterConfig,
    assert_sequential_rhat_checkpoint_public_reference_safe,
    build_fixed_size_hmc_chunk_runner,
    build_reusable_full_chain_tfp_hmc_runner,
    run_full_chain_tfp_hmc,
    stable_adapter_signature,
    write_sequential_rhat_boundary_handoff_checkpoint,
)
from bayesfilter.inference.hmc_budget_ladder import _build_fixed_mass_hmc_adapter
from bayesfilter.inference.hmc_tuning import (
    HMCTuningPolicy,
    WindowedMassAdaptationConfig,
    WindowedMassAdaptationResult,
    build_windowed_warmup_schedule,
    run_windowed_mass_adaptation_diagnostic,
    validate_windowed_shrinkage_target,
    _validate_metric_evidence_policy,
    _validate_metric_probe_num_results,
    welford_covariance,
)
from bayesfilter.inference.hmc_coordinates import WarmupTrajectoryPolicy, transform_from_precomputed_mass_artifact
from bayesfilter.inference.hmc_warmup import (
    G2PreboundarySeedUseRegistry,
    OPERATIONAL_WARMUP_NONCLAIMS,
    OperationalWindowedWarmupCloseout,
    OperationalWindowedWarmupResult,
    PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID,
    Phase7EngineeringProbeBankConfig,
    _G2P4BoundaryActionTracker,
    _G2SeedRegistryError,
    _G2_PREBOUNDARY_SHARED_INVALIDITY_ATTRIBUTE,
    _G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS_CONTRACT,
    _PHASE7_ENGINEERING_PROBE_DIAGNOSTIC_ATTRIBUTE,
    _compose_base_transform_with_nested_estimate,
    _phase7_engineering_probe_target_signature,
    compose_operational_transform_in_base_coordinates,
    engineering_probe_bank_qualification_payload_from_exception,
    g2_preboundary_shared_invalidity_exception,
    g2_preboundary_shared_invalidity_payload_from_exception,
    g2_seed_private_evidence_from_exception,
    run_operational_windowed_warmup,
    start_bank_qualification_payload_from_exception,
)
from bayesfilter.inference.hmc_verification import (
    _all_close,
    _all_finite,
    _float64_tensor,
    _is_boolean_scalar,
)
from bayesfilter.inference.hmc_kernel_selection import private_start_bank_content_signature
from bayesfilter.inference.posterior_adapter import value_score_capability
from bayesfilter.runtime import stable_config_hash

if TYPE_CHECKING:
    from bayesfilter.inference.hmc_configuration import _HMCAttemptBudgetPolicy
    from bayesfilter.inference.hmc_kernel_tuning import _HMCPhaseAttemptState


_G2_WINDOWED_STAGE_SEED_DERIVATION_SITE_ID = (
    "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_derivation.v1"
)

_G2_WINDOWED_STAGE_SEED_GATE_SITE_ID = (
    "hmc_kernel_tuning._run_p4_windowed_boundary_attempt.stage_seed_gate.v1"
)

_G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS = (
    _G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS_CONTRACT
)

_OPERATIONAL_WARMUP_DEFAULT_REUSABLE_RUNNER_BUILDER = (
    build_reusable_full_chain_tfp_hmc_runner
)

_METRIC_UPDATE_REQUIREMENTS = frozenset(
    {"allow_valid_incumbent", "require_operational_update"}
)

_NO_OPERATIONAL_METRIC_UPDATE_REPAIR_TRIGGER = (
    "windowed_mass_no_operational_metric_update"
)

_ENGINEERING_PROBE_BOUNDARY_PUBLIC_KEYS = (
    "schema",
    "policy_id",
    "evidence_role",
    "promotion_role",
    "adaptation_policy",
    "config_signature",
    "position_covariance_estimate_signature",
    "covariance_signature",
    "estimate_covariance_signature_equal",
    "transform_signature",
    "p4_transform_signature",
    "transform_p4_signature_equal",
    "metric_signature",
    "adaptation_generation",
    "applied_metric_update_count",
    "generation_update_count_equal",
    "target_signature",
    "dimension",
    "candidate_count",
    "derived_seed_signature",
    "source_coverage_artifact_sha256",
    "seed_registry_evidence_kind",
    "seed_registry_schema",
    "seed_registry_evidence_signature",
    "seed_preboundary_consumed_count",
    "registered_entry_count",
    "consumed_entry_count",
    "p4_distinct_from_preboundary_seeds",
    "p4_seed_consumed",
    "post_boundary_registry_call_count",
    "target_health_callback_invocation_count",
    "target_health_callback_batch_row_count",
    "target_health_callback_batch_dimension",
    "content_signature",
    "endpoint_round_trip_passed",
    "bank_round_trip_passed",
    "pairwise_distinct",
    "candidate_data_invalidity_present",
    "target_value_finite_count",
    "target_score_finite_count",
    "target_status_failure_count",
    "evaluated_candidate_count",
    "outcome",
    "failure_code",
    "stage",
    "p4_boundary_stage",
    "p4_builder_entered",
    "p4_rng_batch_invoked",
    "final_lineage_available",
    "raw_values_exposed",
    "paths_exposed",
    "seed_values_exposed",
    "covariance_multiplier_exposed",
    "pairwise_distances_exposed",
    "reports_posterior_convergence",
    "nonclaims",
)

WINDOWED_MASS_STAGE_NONCLAIMS = (
    "windowed mass-stage diagnostic only",
    "retained fixed-kernel samples are adaptation inputs only",
    "real acceptance telemetry required for step handoff",
    "no fixed-mass step tuning claim",
    "no trajectory tuning claim",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no GPU or XLA readiness claim",
)

FIXED_MASS_STEP_STAGE_NONCLAIMS = (
    "fixed-mass per-L epsilon tuning stage only",
    "phase 4 adapted mass is frozen during leapfrog-count selection",
    "each candidate leapfrog count gets its own epsilon tuning ladder",
    "fresh fixed-kernel screen required for step handoff",
    "selected pair is a kernel handoff only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no GPU or XLA readiness claim",
)

TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS = (
    "BayesFilter HMC tune-verify-repair loop only",
    "fresh fixed-kernel verification is a kernel handoff screen only",
    "no posterior convergence claim",
    "no sampler superiority claim",
    "no default-readiness claim",
    "no external-client scientific claim",
    "no GPU or XLA readiness claim",
)

STAGED_TIMEOUT_POLICY_STAGE_NAMES = (
    "geometry_and_bootstrap",
    "phase7_pre_windowed",
    "windowed_mass",
    "fixed_mass_step",
    "frozen_step_trajectory",
    "fresh_fixed_kernel_verification",
)

_WINDOWED_STAGE_API_DISCARD_STEPS = 1

_FIXED_MASS_STAGE_TUNE_NUM_RESULTS = 4

_WINDOWED_MASS_PUBLIC_TIMEOUT_RESERVE_S = 60.0

_WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_STATUS = "budget_exhausted"

_WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_ROLE = (
    "windowed_mass_resource_timeout_non_promoting"
)

_WINDOWED_MASS_PUBLIC_TIMEOUT_REPAIR_TRIGGER = (
    "windowed_mass_public_timeout_closeout_before_hmc_call"
)

_WINDOWED_MASS_SEGMENT_SIZE = 4

_OPERATIONAL_WARMUP_SEGMENT_SIZE = 64

_WINDOWED_MASS_SEGMENT_SOFT_DEADLINE_SAFETY_MULTIPLIER = 1.25

_WINDOWED_MASS_SEGMENT_SOFT_DEADLINE_RECENT_WINDOW = 3

LoopProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _validate_metric_update_requirement(value: Any) -> str:
    requirement = str(value)
    if requirement not in _METRIC_UPDATE_REQUIREMENTS:
        allowed = ", ".join(sorted(_METRIC_UPDATE_REQUIREMENTS))
        raise ValueError(f"metric_update_requirement must be one of: {allowed}")
    return requirement


def _validate_engineering_probe_covariance_multiplier(
    value: Any,
) -> float | None:
    if value is None:
        return None
    if _is_boolean_scalar(value):
        raise ValueError(
            "engineering_probe_covariance_multiplier must be positive and finite"
        )
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "engineering_probe_covariance_multiplier must be positive and finite"
        )
    return multiplier


def _engineering_probe_config_public_payload(
    multiplier: float | None,
) -> Mapping[str, Any]:
    configured = multiplier is not None
    return {
        "policy_id": PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID if configured else None,
        "configured": configured,
        "private_config_signature": None
        if multiplier is None
        else stable_config_hash(
            {
                "policy_id": PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID,
                "covariance_multiplier": multiplier,
            }
        ),
        "covariance_multiplier_exposed": False,
        "seed_values_exposed": False,
        "raw_values_exposed": False,
    }


def _engineering_probe_seed_signature(seed: Sequence[int]) -> str:
    """Hash a P4 seed lineage value without exposing its two integers."""

    return stable_config_hash(
        {
            "policy_id": PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID,
            "seed": tuple(int(item) for item in seed),
        }
    )


def _engineering_probe_seed_public_payload(
    seed: Sequence[int],
    *,
    configured: bool,
) -> Mapping[str, Any]:
    """Expose legacy seeds unchanged and bind P4 seeds without revealing them."""

    if not configured:
        return {"seed": tuple(int(item) for item in seed)}
    return {
        "seed": None,
        "seed_signature": _engineering_probe_seed_signature(seed),
        "seed_values_exposed": False,
    }


def _engineering_probe_seed_report_public_payload(
    seed_report: Mapping[str, Any],
    *,
    configured: bool,
) -> Mapping[str, Any]:
    if not configured:
        return dict(seed_report)
    return {
        "seed_values_exposed": False,
        "seed_lineage_signature": stable_config_hash(
            {
                "policy_id": PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID,
                "seed_report": dict(seed_report),
            }
        ),
        "seed_owner": "BayesFilter",
    }


def _engineering_probe_diagnostic_config_public_payload(
    payload: Mapping[str, Any] | None,
    *,
    configured: bool,
) -> Mapping[str, Any] | None:
    """Hide the derived HMC diagnostic seed from a P4 public stage payload."""

    if payload is None or not configured:
        return None if payload is None else dict(payload)
    public = dict(payload)
    seed = public.pop("seed", None)
    if seed is None:
        raise ValueError("configured P4 diagnostic payload must carry a seed")
    public.update(
        {
            "seed": None,
            "seed_signature": _engineering_probe_seed_signature(seed),
            "seed_values_exposed": False,
        }
    )
    return public


def _engineering_probe_windowed_config_public_payload(
    payload: Mapping[str, Any],
    *,
    configured: bool,
) -> Mapping[str, Any]:
    """Keep a configured P4 compatibility config private behind its digest."""

    if not configured:
        return dict(payload)
    return {
        "schema": "bayesfilter.hmc_p4e_windowed_config_summary.v1",
        "private_config_signature": stable_config_hash(dict(payload)),
        "seed_values_exposed": False,
        "raw_values_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_windowed_mass_result_public_payload(
    result: WindowedMassAdaptationResult | None,
    *,
    configured: bool,
) -> Mapping[str, Any] | None:
    """Allowlist compatibility evidence without exposing traces or artifacts."""

    if result is None:
        return None
    if not configured:
        return result.payload()
    return {
        "schema": "bayesfilter.hmc_p4e_windowed_mass_compatibility_summary.v1",
        "passed": bool(result.passed),
        "initial_mass_artifact_signature": result.initial_mass_artifact_signature,
        "shrinkage_target_signature": result.shrinkage_target_signature,
        "final_mass_artifact_signature": result.final_mass_artifact_signature,
        "window_count": len(result.windows),
        "mass_update_count": len(result.mass_updates),
        "step_size_count": len(result.step_size_trace),
        "acceptance_count": len(result.acceptance_trace),
        "final_step_size": result.final_step_size,
        "target_failure_present": result.target_failure_classification is not None,
        "seed_values_exposed": False,
        "raw_values_exposed": False,
        "array_values_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_operational_closeout_public_payload(
    result: OperationalWindowedWarmupCloseout,
    *,
    configured: bool,
) -> Mapping[str, Any]:
    """Publish only counters and fixed stop metadata for a P4 closeout."""

    if not configured:
        return result.public_payload()
    expected_stop_reasons = {
        "before_first_window": "public_timeout_budget_exhausted_at_window_boundary",
        "before_next_window": "public_timeout_budget_exhausted_at_window_boundary",
        "before_next_segment": "public_timeout_budget_exhausted_before_next_segment",
    }
    stop_source = result.boundary_payload.get("stop_source")
    stop_reason = result.boundary_payload.get("stop_reason")
    supervision_baseline = result.boundary_payload.get(
        "supervision_counter_baseline"
    )
    if (
        result.boundary not in expected_stop_reasons
        or stop_source != "bayesfilter_public_timeout_budget"
        or stop_reason != expected_stop_reasons[result.boundary]
        or type(supervision_baseline) is not int
        or supervision_baseline < 0
    ):
        raise ValueError("configured P4 closeout metadata is not closed")
    return {
        "schema": "bayesfilter.hmc_p4e_operational_warmup_closeout.v1",
        "status": result.status,
        "algorithm_id": result.algorithm_id,
        "route_contract_version": result.route_contract_version,
        "boundary": result.boundary,
        "completed_window_count": len(result.completed_windows),
        "planned_window_count": result.planned_window_count,
        "completed_transition_count": result.completed_transition_count,
        "planned_transition_count": result.planned_transition_count,
        "remaining_transition_count": (
            result.planned_transition_count - result.completed_transition_count
        ),
        "completed_segment_count": result.completed_segment_count,
        "planned_segment_count": result.planned_segment_count,
        "stop_source": stop_source,
        "stop_reason": stop_reason,
        "supervision_counter_baseline": supervision_baseline,
        "elapsed_s": result.elapsed_s,
        "completed_warmup_result": False,
        "private_start_bank_exposed": False,
        "seed_values_exposed": False,
        "raw_values_exposed": False,
        "array_values_exposed": False,
        "paths_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_timeout_diagnostics_public_payload(
    payload: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Bind a timeout receipt without republishing either legacy nested copy."""

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("configured P4 timeout diagnostics must be a mapping")
    counts: dict[str, int | None] = {}
    for name in (
        "completed_window_count",
        "planned_window_count",
        "completed_transition_count",
        "planned_transition_count",
        "remaining_transition_count",
        "completed_segment_count",
        "planned_segment_count",
    ):
        value = payload.get(name)
        counts[name] = value if type(value) is int and value >= 0 else None
    return {
        "schema": "bayesfilter.hmc_p4e_timeout_diagnostics_summary.v1",
        **counts,
        "private_closeout_signature": stable_config_hash(dict(payload)),
        "seed_values_exposed": False,
        "raw_values_exposed": False,
        "array_values_exposed": False,
        "paths_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_stage_diagnostics_public_payload(
    payload: Mapping[str, Any],
    *,
    configured: bool,
) -> Mapping[str, Any]:
    """Publish a closed scalar P4 diagnostic summary; retain legacy behavior."""

    if not configured:
        return payload
    boundary = payload.get("engineering_probe_boundary")
    if boundary is not None and not isinstance(boundary, Mapping):
        raise ValueError("configured P4 boundary diagnostics must be a mapping")
    public_boundary = (
        None
        if boundary is None
        else {
            key: boundary[key]
            for key in _ENGINEERING_PROBE_BOUNDARY_PUBLIC_KEYS
            if key in boundary
        }
    )

    def strict_scalar(name: str, allowed_types: tuple[type, ...]) -> Any:
        value = payload.get(name)
        return value if value is None or type(value) in allowed_types else None

    return {
        "schema": "bayesfilter.hmc_p4e_windowed_stage_diagnostics.v1",
        "passed": strict_scalar("passed", (bool,)),
        "runtime_s": strict_scalar("runtime_s", (int, float)),
        "runtime_finite": strict_scalar("runtime_finite", (bool,)),
        "finite_sample_count": strict_scalar("finite_sample_count", (int,)),
        "nonfinite_sample_count": strict_scalar(
            "nonfinite_sample_count", (int,)
        ),
        "windowed_mass_passed": strict_scalar(
            "windowed_mass_passed", (bool,)
        ),
        "candidate_step_size": strict_scalar(
            "candidate_step_size", (int, float)
        ),
        "engineering_probe_boundary": public_boundary,
        "p4_private_seed_evidence_available": strict_scalar(
            "p4_private_seed_evidence_available", (bool,)
        ),
        "required_operational_metric_update_missing": strict_scalar(
            "required_operational_metric_update_missing", (bool,)
        ),
        "public_timeout_closeout": (
            _engineering_probe_timeout_diagnostics_public_payload(
                payload.get("public_timeout_closeout")
            )
        ),
        "hmc_error_type": None,
        "hmc_error_message": None,
        "windowed_mass_error_type": None,
        "windowed_mass_error_message": None,
        "private_diagnostics_signature": stable_config_hash(dict(payload)),
        "seed_values_exposed": False,
        "raw_values_exposed": False,
        "array_values_exposed": False,
        "paths_exposed": False,
        "exception_details_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
    }


def _engineering_probe_reasonable_epsilon_public_payload(
    result: Any,
) -> Mapping[str, Any]:
    """Summarize a P4 epsilon bracket without publishing its RNG leaves."""

    return {
        "status": result.status,
        "selected_step_size": result.selected_step_size,
        "attempt_count": len(result.attempts),
        "qualification_source": result.qualification_source,
        "diagnostic_role": "reasonable_epsilon_engineering_bracket",
        "seed_values_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_operational_window_public_payload(
    window: Any,
) -> Mapping[str, Any]:
    """Allowlist one P4 warmup-window summary and omit private error reports."""

    metric_decision = window.metric_decision
    return {
        "window": window.window.payload(),
        "transition_count_before_window": window.transition_count_before_window,
        "transition_count_after_window": window.transition_count_after_window,
        "coordinate_signature_used": window.coordinate_signature_used,
        "metric_signature_used": window.metric_signature_used,
        "epsilon_start": window.epsilon_start,
        "epsilon_end": window.epsilon_end,
        "mean_acceptance_probability": window.mean_acceptance_probability,
        "binary_acceptance_rate": window.binary_acceptance_rate,
        "native_divergence_status": window.native_divergence_status,
        "native_divergence_count": window.native_divergence_count,
        "target_status_trace_policy": window.target_status_trace_policy,
        "target_status_failure_count": window.target_status_failure_count,
        "max_abs_log_accept_energy_proxy": window.max_abs_log_accept_energy_proxy,
        "step_size_upper_bound": window.step_size_upper_bound,
        "metric_decision": None
        if metric_decision is None
        else {
            "outcome": metric_decision.outcome,
            "estimator_family": metric_decision.estimator_family,
            "update_applied": metric_decision.update_applied,
            "exception_details_exposed": False,
        },
        "next_coordinate_signature": window.next_coordinate_signature,
        "next_metric_signature": window.next_metric_signature,
        "state_map_residual": window.state_map_residual,
        "target_value_map_residual": window.target_value_map_residual,
        "target_score_map_residual": window.target_score_map_residual,
        "next_reasonable_epsilon": None
        if window.next_reasonable_epsilon is None
        else _engineering_probe_reasonable_epsilon_public_payload(
            window.next_reasonable_epsilon
        ),
        "dual_averaging_generation": window.dual_averaging_generation,
        "runner_generation": window.runner_generation,
        "runner_trace_count": window.runner_trace_count,
        "runtime_s": window.runtime_s,
        "raw_states_exposed": False,
        "seed_values_exposed": False,
        "exception_details_exposed": False,
    }


def _engineering_probe_operational_warmup_public_payload(
    result: OperationalWindowedWarmupResult,
    *,
    configured: bool,
) -> Mapping[str, Any]:
    """Keep legacy output unchanged and make the P4 tree recursively private."""

    if not configured:
        return result.public_payload()
    qualification = result.engineering_probe_bank_qualification
    return {
        "schema": "bayesfilter.hmc_operational_windowed_warmup.v2",
        "status": result.status,
        "metric_adaptation_status": result.metric_adaptation_status,
        "algorithm_id": result.algorithm_id,
        "route_contract_version": result.route_contract_version,
        "config": {
            "schema": "bayesfilter.hmc_p4e_private_warmup_config.v1",
            "signature": stable_config_hash(result.config.payload()),
            "seed_values_exposed": False,
            "raw_values_exposed": False,
        },
        "initial_coordinate_signature": result.initial_coordinate_signature,
        "final_coordinate_signature": result.final_kernel_state.transform.signature,
        "final_metric_signature": result.final_kernel_state.momentum_metric.signature,
        "final_epsilon": result.final_kernel_state.epsilon,
        "trajectory_policy_signature": (
            result.final_kernel_state.trajectory_policy.signature
        ),
        "reasonable_epsilon": (
            _engineering_probe_reasonable_epsilon_public_payload(
                result.reasonable_epsilon
            )
        ),
        "windows": tuple(
            _engineering_probe_operational_window_public_payload(window)
            for window in result.windows
        ),
        "operational_metric_update_count": result.operational_metric_update_count,
        "every_update_used_by_later_transition": (
            result.every_update_used_by_later_transition
        ),
        "private_start_bank": {
            "schema": "bayesfilter.hmc_private_start_bank.v2",
            "policy_id": result.private_start_bank_policy_id,
            "signature": result.private_start_bank_signature,
            "count": 4,
            "engineering_probe_qualification": None
            if qualification is None
            else qualification.public_payload(),
            "raw_values_exposed": False,
            "paths_exposed": False,
            "seed_values_exposed": False,
        },
        "seed_root": None,
        "target_scope": result.target_scope,
        "target_status_trace_policy": result.target_status_trace_policy,
        "elapsed_s": result.elapsed_s,
        "seed_values_exposed": False,
        "exception_details_exposed": False,
        "reports_posterior_convergence": False,
        "nonclaims": OPERATIONAL_WARMUP_NONCLAIMS,
    }


@dataclass(frozen=True)
class HMCStagedTimeoutPolicy:
    """Opt-in public-safe staged timeout accounting policy."""

    policy_id: str = "bayesfilter_hmc_emergency_stage_caps_v2"
    stage_budgets_s: Mapping[str, float] | None = None
    stage_budget_provenance: Mapping[str, str] | None = None
    global_cap_s: float = 86400.0
    reserve_s: float = 600.0
    max_enlargement_rounds_per_stage: int = 1
    enlargement_multiplier: float = 1.5
    enabled: bool = True
    source: str = "bayesfilter.inference.hmc_kernel_tuning.staged_timeout_policy"

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id)
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        stage_budgets = dict(
            _default_staged_timeout_policy_stage_budgets()
            if self.stage_budgets_s is None
            else self.stage_budgets_s
        )
        if set(stage_budgets) != set(STAGED_TIMEOUT_POLICY_STAGE_NAMES):
            raise ValueError("stage_budgets_s must match the allowed stage names")
        budgets: dict[str, float] = {}
        for stage, value in stage_budgets.items():
            budget = float(value)
            if not math.isfinite(budget) or budget <= 0.0:
                raise ValueError("stage budgets must be positive and finite")
            budgets[str(stage)] = budget
        provenance_source = (
            _default_staged_timeout_policy_stage_budget_provenance()
            if self.stage_budget_provenance is None
            else self.stage_budget_provenance
        )
        provenance = {
            str(stage): str(value) for stage, value in dict(provenance_source).items()
        }
        if set(provenance) != set(budgets):
            raise ValueError("stage_budget_provenance must cover every stage budget")
        cap = float(self.global_cap_s)
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("global_cap_s must be positive and finite")
        reserve = float(self.reserve_s)
        if not math.isfinite(reserve) or reserve < 0.0:
            raise ValueError("reserve_s must be finite and non-negative")
        if reserve >= cap:
            raise ValueError("reserve_s must be smaller than global_cap_s")
        max_rounds = int(self.max_enlargement_rounds_per_stage)
        if max_rounds < 0:
            raise ValueError("max_enlargement_rounds_per_stage must be non-negative")
        multiplier = float(self.enlargement_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 1.0:
            raise ValueError("enlargement_multiplier must be finite and greater than 1")
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "policy_id", policy_id)
        object.__setattr__(self, "stage_budgets_s", budgets)
        object.__setattr__(self, "stage_budget_provenance", provenance)
        object.__setattr__(self, "global_cap_s", cap)
        object.__setattr__(self, "reserve_s", reserve)
        object.__setattr__(self, "max_enlargement_rounds_per_stage", max_rounds)
        object.__setattr__(self, "enlargement_multiplier", multiplier)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy_id": self.policy_id,
            "stage_budgets_s": dict(self.stage_budgets_s),
            "stage_budget_provenance": dict(self.stage_budget_provenance),
            "global_cap_s": self.global_cap_s,
            "reserve_s": self.reserve_s,
            "max_enlargement_rounds_per_stage": self.max_enlargement_rounds_per_stage,
            "enlargement_multiplier": self.enlargement_multiplier,
            "enabled": self.enabled,
            "source": self.source,
        }


def _default_staged_timeout_policy_stage_budgets() -> Mapping[str, float]:
    return {
        "geometry_and_bootstrap": 3600.0,
        "phase7_pre_windowed": 3600.0,
        "windowed_mass": 3600.0,
        "fixed_mass_step": 3600.0,
        "frozen_step_trajectory": 3600.0,
        "fresh_fixed_kernel_verification": 3600.0,
    }


def _default_staged_timeout_policy_stage_budget_provenance() -> Mapping[str, str]:
    return {
        "geometry_and_bootstrap": "emergency_cap_machine_protection_not_progress_gate",
        "phase7_pre_windowed": "emergency_cap_machine_protection_not_progress_gate",
        "windowed_mass": "emergency_cap_machine_protection_not_progress_gate",
        "fixed_mass_step": "emergency_cap_machine_protection_not_progress_gate",
        "frozen_step_trajectory": "emergency_cap_machine_protection_not_progress_gate",
        "fresh_fixed_kernel_verification": "emergency_cap_machine_protection_not_progress_gate",
    }


def _validate_staged_timeout_policy_or_none(
    value: Any,
) -> HMCStagedTimeoutPolicy | None:
    if value is None:
        return None
    if not isinstance(value, HMCStagedTimeoutPolicy):
        raise TypeError("staged_timeout_policy must be HMCStagedTimeoutPolicy or None")
    return value


def _validate_nonnegative_perf_counter_or_none(
    value: Any,
    *,
    name: str,
) -> float | None:
    if value is None:
        return None
    perf_counter = float(value)
    if not math.isfinite(perf_counter) or perf_counter < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return perf_counter


def _validate_staged_timeout_enlargement_rounds(
    value: Any,
) -> Mapping[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("staged_timeout_enlargement_rounds must be a mapping or None")
    rounds: dict[str, int] = {}
    for key, raw in value.items():
        round_index = int(raw)
        if round_index < 0:
            raise ValueError("staged_timeout_enlargement_rounds must be non-negative")
        rounds[str(key)] = round_index
    return rounds


def _bootstrap_geometry_preflight_kernel_payload(
    *,
    geometry: "HMCGeometryInitializationResult",
    bootstrap: "HMCBootstrapScreenResult",
    nonclaims: tuple[str, ...],
) -> Mapping[str, Any]:
    """Return a non-promoting kernel seed when bootstrap only passed preflight.

    A short bootstrap screen is allowed to answer the mechanics question
    "can this target run finitely?" without also selecting the step/L pair used
    by later tuning stages.  When no bootstrap round is inside the acceptance
    band, Phase 4 starts from the posterior/geometry seed instead of pretending
    the bootstrap selected a tuned kernel.
    """

    seed = _seed_from_mapping(geometry.seed_report, "geometry_seed")
    return {
        "runtime": "bayesfilter.inference.initialize_hmc_kernel_geometry",
        "schema": "bayesfilter.hmc_bootstrap_preflight_fallback_kernel.v1",
        "sample_space": "latent_fixed_mass",
        "handoff_role": "bootstrap_preflight_fallback_non_promoting",
        "step_size": geometry.initial_step_size,
        "num_leapfrog_steps": geometry.initial_num_leapfrog_steps,
        "target_trajectory_length": geometry.target_trajectory_length,
        "target_accept_prob": bootstrap.config.target_accept_prob,
        "acceptance_band": bootstrap.config.acceptance_band,
        "repair_band": bootstrap.config.repair_band,
        "adapter_signature": bootstrap.adapter_signature,
        "hmc_adapter_signature": bootstrap.hmc_adapter_signature,
        "mass_artifact_signature": bootstrap.mass_artifact_signature,
        "geometry_artifact_hash": geometry.artifact_hash,
        "bootstrap_artifact_hash": bootstrap.artifact_hash,
        "seed": seed,
        "screen_config_payload": None,
        "bootstrap_selected_round_index": bootstrap.selected_round_index,
        "bootstrap_final_status": bootstrap.final_status,
        "bootstrap_hard_veto_present": bool(_bootstrap_hard_vetoes(bootstrap)),
        "bootstrap_acceptance_promoted": False,
        "reports_bootstrap_tuning_success": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": nonclaims,
    }


def _bootstrap_hard_vetoes(bootstrap: "HMCBootstrapScreenResult") -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(veto)
            for round_result in bootstrap.rounds
            # A discarded, narrowly attributed proposal failure can be
            # repaired only by a later completed bootstrap screen. Its raw
            # veto and first-failure record remain in the round artifact.
            if not (
                getattr(bootstrap, "passed", False)
                and round_result.round_index < bootstrap.selected_round_index
                and round_result.classification == "repair"
                and round_result.diagnostics.get("proposal_domain_retry_eligible") is True
            )
            for veto in round_result.hard_vetoes
        )
    )


def _bootstrap_preflight_passed(bootstrap: "HMCBootstrapScreenResult") -> bool:
    return not _bootstrap_hard_vetoes(bootstrap)


def _active_bootstrap_handoff_kernel_payload(
    *,
    geometry: "HMCGeometryInitializationResult",
    bootstrap: "HMCBootstrapScreenResult",
) -> Mapping[str, Any]:
    selected = bootstrap.selected_kernel_payload
    if selected is not None:
        return selected
    if not _bootstrap_preflight_passed(bootstrap):
        raise ValueError("bootstrap hard veto cannot provide active handoff kernel")
    return _bootstrap_geometry_preflight_kernel_payload(
        geometry=geometry,
        bootstrap=bootstrap,
        nonclaims=bootstrap.nonclaims,
    )


def _active_bootstrap_handoff_kernel_hash(
    *,
    geometry: "HMCGeometryInitializationResult",
    bootstrap: "HMCBootstrapScreenResult",
) -> str:
    return stable_config_hash(
        _active_bootstrap_handoff_kernel_payload(
            geometry=geometry,
            bootstrap=bootstrap,
        )
    )


def _phase7_windowed_mass_seed_kernel_payload(
    *,
    geometry: "HMCGeometryInitializationResult",
    bootstrap: "HMCBootstrapScreenResult",
    attempt_state: "_HMCPhaseAttemptState | None",
) -> Mapping[str, Any]:
    """Choose the private kernel used to collect Phase 4 mass-window draws.

    The first attempt uses the bootstrap/geometry handoff.  Repair attempts
    should instead collect the next mass window at the previous selected or
    repaired fixed kernel, matching the old robust tuning loop: select a
    promising ``(L, epsilon)`` pair, update the mass moderately at that pair,
    then rerun the joint grid.
    """

    bootstrap_payload = _active_bootstrap_handoff_kernel_payload(
        geometry=geometry,
        bootstrap=bootstrap,
    )
    if attempt_state is None or attempt_state.selected_step_size is None:
        return bootstrap_payload
    if (
        attempt_state.verification_repair_applied
        and attempt_state.verification_repair_step_size is not None
    ):
        step = float(attempt_state.verification_repair_step_size)
        step_hash = attempt_state.verification_repair_step_hash
        source = "phase7_private_repair_step"
    else:
        step = float(attempt_state.selected_step_size)
        step_hash = attempt_state.selected_step_hash
        source = "phase7_private_selected_step"
    leapfrog = attempt_state.selected_num_leapfrog_steps
    if leapfrog is None:
        leapfrog = attempt_state.phase6_retry_num_leapfrog_steps
    if leapfrog is None:
        return bootstrap_payload
    return {
        **dict(bootstrap_payload),
        "runtime": "bayesfilter.inference.run_hmc_windowed_mass_stage",
        "handoff_role": "phase7_private_mass_window_seed_kernel",
        "step_size": step,
        "num_leapfrog_steps": int(leapfrog),
        "private_step_hash": step_hash,
        "private_kernel_source": source,
        "bootstrap_kernel_hash": _active_bootstrap_handoff_kernel_hash(
            geometry=geometry,
            bootstrap=bootstrap,
        ),
        "bootstrap_kernel_is_lineage_not_active_mass_window_seed": True,
        "phase6_retry_num_leapfrog_steps": attempt_state.phase6_retry_num_leapfrog_steps,
        "phase6_retry_anchor_source": attempt_state.phase6_retry_anchor_source,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_gpu_or_xla_readiness": False,
    }


@dataclass(frozen=True)
class HMCWindowedMassStageConfig:
    """Policy-level config for the Phase 4 windowed mass stage.

    The caller does not supply warmup windows, draw counts, step size, leapfrog
    counts, grids, mass schedules, or budget schedules.  Phase 4 owns those
    mechanics internally and records the chosen draw-capture policy in the
    result artifact.
    """

    algorithm_id: str = OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    target_accept_prob: float = 0.70
    seed: tuple[int, int] = (20260621, 4)
    chain_execution_mode: str = "tf_function"
    use_xla: bool = False
    target_scope: str | None = None
    target_status_trace_policy: str = "none"
    mass_policy: str = "windowed_adaptive"
    metric_update_requirement: str = "allow_valid_incumbent"
    metric_evidence_policy: str = "temporal_information"
    metric_probe_num_results: int = 1
    preparation_max_restarts: int = 0
    engineering_probe_covariance_multiplier: float | None = None
    public_timeout_budget_s: float | None = None
    public_timeout_started_perf_counter_s: float | None = None
    staged_timeout_policy: HMCStagedTimeoutPolicy | None = None
    staged_timeout_global_started_perf_counter_s: float | None = None
    staged_timeout_stage_started_perf_counter_s: float | None = None
    staged_timeout_enlargement_rounds: Mapping[str, int] | None = None
    source: str = "bayesfilter.inference.hmc_kernel_tuning.windowed_mass_stage"

    def __post_init__(self) -> None:
        algorithm_id = str(self.algorithm_id)
        if not algorithm_id:
            raise ValueError("algorithm_id must be non-empty")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        object.__setattr__(self, "seed", _validate_seed(self.seed))
        mode = str(self.chain_execution_mode)
        if mode not in {"tf_function", "eager"}:
            raise ValueError("chain_execution_mode must be 'tf_function' or 'eager'")
        if self.use_xla and mode != "tf_function":
            raise ValueError("XLA HMC requires chain_execution_mode='tf_function'")
        object.__setattr__(self, "chain_execution_mode", mode)
        object.__setattr__(self, "use_xla", bool(self.use_xla))
        if self.target_scope is not None:
            scope = str(self.target_scope)
            if not scope:
                raise ValueError("target_scope must be non-empty when provided")
            object.__setattr__(self, "target_scope", scope)
        target_status_policy = str(self.target_status_trace_policy)
        if target_status_policy not in {"none", "per_chain_step"}:
            raise ValueError(
                "target_status_trace_policy must be 'none' or 'per_chain_step'"
            )
        object.__setattr__(self, "target_status_trace_policy", target_status_policy)
        mass_policy = str(self.mass_policy)
        if mass_policy not in {"windowed_adaptive", "fixed_identity"}:
            raise ValueError(
                "mass_policy must be 'windowed_adaptive' or 'fixed_identity'"
            )
        object.__setattr__(self, "mass_policy", mass_policy)
        object.__setattr__(self, "metric_evidence_policy", _validate_metric_evidence_policy(
            self.metric_evidence_policy, mass_policy=mass_policy))
        object.__setattr__(self, "metric_probe_num_results", _validate_metric_probe_num_results(
            self.metric_probe_num_results, mass_policy=mass_policy))
        object.__setattr__(self, "preparation_max_restarts", validate_preparation_max_restarts(
            self.preparation_max_restarts, mass_policy=mass_policy))
        if self.metric_probe_num_results != 1 and self.algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID:
            raise ValueError("longer metric probes require operational windowed warmup")
        if self.preparation_max_restarts and (self.algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
                                              or self.engineering_probe_covariance_multiplier is not None):
            raise ValueError("preparation recovery requires the ordinary operational route")
        if self.metric_evidence_policy != "temporal_information" and self.algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID:
            raise ValueError("finite_window metric evidence requires operational windowed warmup")
        metric_requirement = _validate_metric_update_requirement(
            self.metric_update_requirement
        )
        if (
            metric_requirement == "require_operational_update"
            and algorithm_id != OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
        ):
            raise ValueError(
                "require_operational_update requires the operational windowed route"
            )
        if (
            metric_requirement == "require_operational_update"
            and mass_policy == "fixed_identity"
        ):
            raise ValueError(
                "require_operational_update is incompatible with fixed_identity"
            )
        object.__setattr__(
            self,
            "metric_update_requirement",
            metric_requirement,
        )
        object.__setattr__(
            self,
            "engineering_probe_covariance_multiplier",
            _validate_engineering_probe_covariance_multiplier(
                self.engineering_probe_covariance_multiplier
            ),
        )
        timeout_budget = (
            None
            if self.public_timeout_budget_s is None
            else float(self.public_timeout_budget_s)
        )
        if timeout_budget is not None and (
            not math.isfinite(timeout_budget) or timeout_budget <= 0.0
        ):
            raise ValueError("public_timeout_budget_s must be positive and finite")
        object.__setattr__(self, "public_timeout_budget_s", timeout_budget)
        timeout_started = (
            None
            if self.public_timeout_started_perf_counter_s is None
            else float(self.public_timeout_started_perf_counter_s)
        )
        if timeout_started is not None and (
            not math.isfinite(timeout_started) or timeout_started < 0.0
        ):
            raise ValueError(
                "public_timeout_started_perf_counter_s must be finite and non-negative"
            )
        object.__setattr__(
            self,
            "public_timeout_started_perf_counter_s",
            timeout_started,
        )
        staged_policy = _validate_staged_timeout_policy_or_none(
            self.staged_timeout_policy
        )
        object.__setattr__(self, "staged_timeout_policy", staged_policy)
        object.__setattr__(
            self,
            "staged_timeout_global_started_perf_counter_s",
            _validate_nonnegative_perf_counter_or_none(
                self.staged_timeout_global_started_perf_counter_s,
                name="staged_timeout_global_started_perf_counter_s",
            ),
        )
        object.__setattr__(
            self,
            "staged_timeout_stage_started_perf_counter_s",
            _validate_nonnegative_perf_counter_or_none(
                self.staged_timeout_stage_started_perf_counter_s,
                name="staged_timeout_stage_started_perf_counter_s",
            ),
        )
        object.__setattr__(
            self,
            "staged_timeout_enlargement_rounds",
            _validate_staged_timeout_enlargement_rounds(
                self.staged_timeout_enlargement_rounds,
            ),
        )
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "target_accept_prob": self.target_accept_prob,
            **_engineering_probe_seed_public_payload(
                self.seed,
                configured=self.engineering_probe_covariance_multiplier is not None,
            ),
            "chain_execution_mode": self.chain_execution_mode,
            "use_xla": self.use_xla,
            "target_scope": self.target_scope,
            "target_status_trace_policy": self.target_status_trace_policy,
            "mass_policy": self.mass_policy,
            "metric_update_requirement": self.metric_update_requirement,
            "metric_evidence_policy": self.metric_evidence_policy,
            "metric_probe_num_results": self.metric_probe_num_results,
            "preparation_max_restarts": self.preparation_max_restarts,
            "engineering_probe_bank": _engineering_probe_config_public_payload(
                self.engineering_probe_covariance_multiplier
            ),
            "public_timeout_budget_s": self.public_timeout_budget_s,
            "public_timeout_started_perf_counter_s": (
                self.public_timeout_started_perf_counter_s
            ),
            "staged_timeout_policy": None
            if self.staged_timeout_policy is None
            else self.staged_timeout_policy.payload(),
            "staged_timeout_global_started_perf_counter_s": (
                self.staged_timeout_global_started_perf_counter_s
            ),
            "staged_timeout_stage_started_perf_counter_s": (
                self.staged_timeout_stage_started_perf_counter_s
            ),
            "staged_timeout_enlargement_rounds": None
            if self.staged_timeout_enlargement_rounds is None
            else dict(self.staged_timeout_enlargement_rounds),
            "source": self.source,
        }


@dataclass(frozen=True)
class HMCWindowedMassStageResult:
    """Phase 4 adapted-mass handoff; not posterior or final tuning evidence."""

    config: HMCWindowedMassStageConfig
    geometry_artifact_hash: str
    bootstrap_artifact_hash: str
    selected_bootstrap_kernel_hash: str
    adapter_signature: str
    hmc_adapter_signature: str
    initial_mass_artifact_signature: str
    target_dimension: int
    final_status: str
    diagnostic_role: str
    hard_vetoes: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    draw_capture_policy: Mapping[str, Any]
    warmup_draw_provenance: Mapping[str, Any]
    acceptance_telemetry_provenance: Mapping[str, Any]
    diagnostic_run_config_payload: Mapping[str, Any] | None
    windowed_config_payload: Mapping[str, Any]
    windowed_mass_result: WindowedMassAdaptationResult | None
    seed_report: Mapping[str, Any]
    diagnostic_roles: Mapping[str, str]
    repair_triggers: tuple[str, ...] = ()
    operational_warmup_result: OperationalWindowedWarmupResult | None = None
    operational_warmup_closeout: OperationalWindowedWarmupCloseout | None = None
    operational_mass_artifact: PrecomputedMassArtifact | None = dataclasses.field(
        default=None,
        repr=False,
    )
    nonclaims: tuple[str, ...] = WINDOWED_MASS_STAGE_NONCLAIMS

    def __post_init__(self) -> None:
        for name in (
            "geometry_artifact_hash",
            "bootstrap_artifact_hash",
            "selected_bootstrap_kernel_hash",
            "adapter_signature",
            "hmc_adapter_signature",
            "initial_mass_artifact_signature",
            "final_status",
            "diagnostic_role",
        ):
            value = str(getattr(self, name))
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        dimension = int(self.target_dimension)
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        object.__setattr__(self, "target_dimension", dimension)
        hard_vetoes = _string_tuple(self.hard_vetoes)
        object.__setattr__(self, "hard_vetoes", hard_vetoes)
        repair_triggers = _string_tuple(self.repair_triggers)
        object.__setattr__(self, "repair_triggers", repair_triggers)
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "draw_capture_policy", dict(self.draw_capture_policy))
        object.__setattr__(
            self,
            "warmup_draw_provenance",
            dict(self.warmup_draw_provenance),
        )
        object.__setattr__(
            self,
            "acceptance_telemetry_provenance",
            dict(self.acceptance_telemetry_provenance),
        )
        payload = (
            None
            if self.diagnostic_run_config_payload is None
            else dict(self.diagnostic_run_config_payload)
        )
        object.__setattr__(self, "diagnostic_run_config_payload", payload)
        object.__setattr__(self, "windowed_config_payload", dict(self.windowed_config_payload))
        if self.config.mass_policy == "fixed_identity" and self.windowed_mass_result is not None:
            if self.adapted_mass_artifact_signature != self.initial_mass_artifact_signature:
                raise ValueError("fixed identity Phase 4 mass signature mutated")
            if self.windowed_mass_result is not None:
                checks = self.windowed_mass_result.semantic_checks()
                if checks.get("fixed_identity_signature_unchanged") is not True:
                    raise ValueError("fixed identity Phase 4 semantic invariant failed")
        if self.operational_warmup_result is not None and not isinstance(
            self.operational_warmup_result,
            OperationalWindowedWarmupResult,
        ):
            raise TypeError(
                "operational_warmup_result must be OperationalWindowedWarmupResult"
            )
        if self.operational_warmup_closeout is not None and not isinstance(
            self.operational_warmup_closeout,
            OperationalWindowedWarmupCloseout,
        ):
            raise TypeError(
                "operational_warmup_closeout must be OperationalWindowedWarmupCloseout"
            )
        if (
            self.operational_warmup_result is not None
            and self.operational_warmup_closeout is not None
        ):
            raise ValueError("completed operational warmup cannot also be a closeout")
        if self.operational_mass_artifact is not None and not isinstance(
            self.operational_mass_artifact, PrecomputedMassArtifact
        ):
            raise TypeError("operational_mass_artifact must be PrecomputedMassArtifact")
        object.__setattr__(self, "seed_report", dict(self.seed_report))
        object.__setattr__(self, "diagnostic_roles", dict(self.diagnostic_roles))
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "nonclaims", nonclaims)
        if self.final_status == "passed":
            if hard_vetoes:
                raise ValueError("passed windowed mass stage cannot have hard vetoes")
            operational_pass = (
                self.operational_warmup_result is not None
                and self.operational_mass_artifact is not None
            )
            compatibility_pass = (
                self.windowed_mass_result is not None
                and self.windowed_mass_result.passed
            )
            if not operational_pass and not compatibility_pass:
                raise ValueError("passed windowed mass stage requires authoritative mass")
            if (
                self.config.metric_update_requirement == "require_operational_update"
                and self.operational_warmup_result is not None
                and self.operational_warmup_result.operational_metric_update_count == 0
            ):
                raise ValueError(
                    "required operational metric update cannot be reported as passed"
                )
        if self.final_status == "passed_no_metric_update":
            if hard_vetoes:
                raise ValueError("no-update windowed stage cannot have hard vetoes")
            if self.config.metric_update_requirement != "require_operational_update":
                raise ValueError(
                    "passed_no_metric_update requires the operational-update policy"
                )
            if (
                self.operational_warmup_result is None
                or self.operational_warmup_result.operational_metric_update_count != 0
            ):
                raise ValueError(
                    "passed_no_metric_update requires a completed zero-update warmup"
                )
            if self.diagnostic_role != "metric_adaptation_not_observed":
                raise ValueError("no-update windowed stage diagnostic role is invalid")
            if _NO_OPERATIONAL_METRIC_UPDATE_REPAIR_TRIGGER not in repair_triggers:
                raise ValueError("no-update windowed stage lacks its repair trigger")
        boundary = self.diagnostics.get("engineering_probe_boundary")
        if self.final_status == "candidate_rejected":
            if self.diagnostic_role != "p4e_candidate_boundary_rejection":
                raise ValueError("P4-E candidate rejection role is invalid")
            if hard_vetoes or repair_triggers:
                raise ValueError("P4-E candidate rejection cannot carry vetoes")
            if not isinstance(boundary, Mapping) or boundary.get("outcome") not in {
                "candidate_generation_invalid",
                "candidate_policy_instance_invalid",
            }:
                raise ValueError("P4-E candidate rejection lacks its carrier")
            if any(
                boundary.get(name) is not True
                for name in (
                    "estimate_covariance_signature_equal",
                    "transform_p4_signature_equal",
                    "generation_update_count_equal",
                )
            ):
                raise ValueError("P4-E candidate rejection has invalid lineage")
            if (
                self.operational_warmup_result is not None
                or self.operational_warmup_closeout is not None
                or self.operational_mass_artifact is not None
                or self.windowed_mass_result is not None
            ):
                raise ValueError("P4-E candidate rejection cannot retain an operation")
            if (
                self.diagnostics.get("hmc_error_type") is not None
                or self.diagnostics.get("hmc_error_message") is not None
                or self.diagnostics.get("passed") is not False
            ):
                raise ValueError("P4-E candidate rejection entered a generic path")
        elif self.diagnostic_role == "p4e_candidate_boundary_rejection":
            raise ValueError("P4-E candidate role requires candidate_rejected")

        if self.diagnostic_role == "shared_implementation_invalid":
            if self.final_status != "hard_veto":
                raise ValueError("shared invalidity must be a hard veto")
            if not isinstance(boundary, Mapping) or boundary.get("outcome") != (
                "shared_implementation_invalid"
            ):
                raise ValueError("shared invalidity lacks its scalar carrier")
            failure_code = boundary.get("failure_code")
            if type(failure_code) is not str or hard_vetoes != (failure_code,):
                raise ValueError("shared invalidity hard-veto code is inconsistent")
            if (
                self.operational_warmup_result is not None
                or self.operational_warmup_closeout is not None
                or self.operational_mass_artifact is not None
                or self.windowed_mass_result is not None
            ):
                raise ValueError("shared invalidity cannot retain an operation")
            if (
                self.diagnostics.get("hmc_error_type") is not None
                or self.diagnostics.get("hmc_error_message") is not None
                or self.diagnostics.get("passed") is not False
            ):
                raise ValueError("shared invalidity entered a generic error path")

    @property
    def passed(self) -> bool:
        return self.final_status == "passed"

    @property
    def adapted_mass_artifact_payload(self) -> Mapping[str, Any] | None:
        if self.operational_mass_artifact is not None:
            return self.operational_mass_artifact.signature_payload()
        if self.windowed_mass_result is None:
            return None
        return self.windowed_mass_result.final_mass_artifact_payload

    @property
    def adapted_mass_artifact_signature(self) -> str | None:
        if self.operational_mass_artifact is not None:
            return _mass_artifact_signature(self.operational_mass_artifact)
        if self.windowed_mass_result is None:
            return None
        return self.windowed_mass_result.final_mass_artifact_signature

    @property
    def candidate_step_size(self) -> float | None:
        if self.operational_warmup_result is not None:
            return float(self.operational_warmup_result.final_kernel_state.epsilon)
        if self.windowed_mass_result is None:
            return None
        return self.windowed_mass_result.final_step_size

    @property
    def artifact_hash(self) -> str:
        return stable_config_hash(self.payload())

    def payload(self) -> Mapping[str, Any]:
        engineering_probe_configured = (
            self.config.engineering_probe_covariance_multiplier is not None
        )
        return {
            "schema": "bayesfilter.hmc_windowed_mass_stage.v1",
            "config": self.config.payload(),
            "geometry_artifact_hash": self.geometry_artifact_hash,
            "bootstrap_artifact_hash": self.bootstrap_artifact_hash,
            "selected_bootstrap_kernel_hash": self.selected_bootstrap_kernel_hash,
            "adapter_signature": self.adapter_signature,
            "hmc_adapter_signature": self.hmc_adapter_signature,
            "initial_mass_artifact_signature": self.initial_mass_artifact_signature,
            "mass_policy": self.config.mass_policy,
            "target_dimension": self.target_dimension,
            "final_status": self.final_status,
            "diagnostic_role": self.diagnostic_role,
            "hard_vetoes": self.hard_vetoes,
            "repair_triggers": self.repair_triggers,
            "diagnostics": _engineering_probe_stage_diagnostics_public_payload(
                self.diagnostics,
                configured=engineering_probe_configured,
            ),
            "draw_capture_policy": self.draw_capture_policy,
            "warmup_draw_provenance": self.warmup_draw_provenance,
            "acceptance_telemetry_provenance": self.acceptance_telemetry_provenance,
            "diagnostic_run_config_payload": (
                _engineering_probe_diagnostic_config_public_payload(
                    self.diagnostic_run_config_payload,
                    configured=engineering_probe_configured,
                )
            ),
            "windowed_config_payload": (
                _engineering_probe_windowed_config_public_payload(
                    self.windowed_config_payload,
                    configured=engineering_probe_configured,
                )
            ),
            "windowed_mass_result": (
                _engineering_probe_windowed_mass_result_public_payload(
                    self.windowed_mass_result,
                    configured=engineering_probe_configured,
                )
            ),
            "operational_warmup_result": None
            if self.operational_warmup_result is None
            else _engineering_probe_operational_warmup_public_payload(
                self.operational_warmup_result,
                configured=engineering_probe_configured,
            ),
            "operational_warmup_closeout": None
            if self.operational_warmup_closeout is None
            else _engineering_probe_operational_closeout_public_payload(
                self.operational_warmup_closeout,
                configured=engineering_probe_configured,
            ),
            "operational_mass_artifact_available": (
                self.operational_mass_artifact is not None
            ),
            "adapted_mass_artifact_payload": (
                None
                if engineering_probe_configured
                else self.adapted_mass_artifact_payload
            ),
            "adapted_mass_artifact_signature": self.adapted_mass_artifact_signature,
            "candidate_step_size": self.candidate_step_size,
            "seed_report": _engineering_probe_seed_report_public_payload(
                self.seed_report,
                configured=engineering_probe_configured,
            ),
            "diagnostic_roles": self.diagnostic_roles,
            "passed": self.passed,
            "reports_fixed_mass_step_tuning": False,
            "reports_trajectory_tuning": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_gpu_or_xla_readiness": False,
            "nonclaims": self.nonclaims,
        }


def _operational_windowed_mass_capture(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    hmc_adapter_signature: str,
    stage_mass_artifact: PrecomputedMassArtifact,
    mass_window_seed_kernel: Mapping[str, Any],
    windowed_config: WindowedMassAdaptationConfig,
    config: HMCWindowedMassStageConfig,
    stage_seed: tuple[int, int],
    target_scope: str,
    attempt_state: "_HMCPhaseAttemptState | None",
    route_decision: HMCAlgorithmRouteDecision,
    progress_callback: LoopProgressCallback | None,
    attempt_index: int | None,
    g2_seed_use_registry: G2PreboundarySeedUseRegistry | None = None,
    g2_p4_action_tracker: _G2P4BoundaryActionTracker | None = None,
) -> tuple[
    OperationalWindowedWarmupResult | None,
    WindowedMassAdaptationResult | None,
    Mapping[str, Any],
    OperationalWindowedWarmupCloseout | None,
    PrecomputedMassArtifact | None,
]:
    """Run R3 and build a deliberately non-operational v1 compatibility view."""

    import tensorflow as tf

    _base_estimate, base_transform = transform_from_precomputed_mass_artifact(
        geometry.mass_artifact,
        source_coordinate_signature=geometry.mass_artifact_signature,
        estimator_family="geometry_position_covariance",
    )
    active_estimate, active_transform = _compose_base_transform_with_nested_estimate(
        base_transform=base_transform,
        nested_artifact=stage_mass_artifact,
        source_coordinate_signature=_mass_artifact_signature(stage_mass_artifact),
    )
    initial_theta = (
        _float64_tensor(active_transform.center)
        if attempt_state is None or attempt_state.canonical_theta_state is None
        else _float64_tensor(attempt_state.canonical_theta_state)
    )
    trajectory_policy = WarmupTrajectoryPolicy(
        int(mass_window_seed_kernel["num_leapfrog_steps"]),
        _GEOMETRY_MAX_LEAPFROG,
    )
    reported_window_count = 0
    preparation_attempt_index = 0

    def recovery_callback(event: str, payload: Mapping[str, Any]) -> None:
        nonlocal reported_window_count, preparation_attempt_index
        if event == "attempt_start":
            reported_window_count = 0
            preparation_attempt_index = int(payload["attempt_index"])
        if progress_callback is not None:
            progress_callback("preparation_recovery_" + event, {
                **payload, "raw_samples_retained": event == "attempt_discarded",
                "diagnostic_role": "discarded_preparation_recovery", "artifact_authority": False,
            })

    def report_completed_windows(windows: tuple[Mapping[str, Any], ...]) -> None:
        nonlocal reported_window_count
        # P4-E retains its existing separately redacted public interface.
        if progress_callback is not None and config.engineering_probe_covariance_multiplier is None:
            for window in windows[reported_window_count:]:
                progress_callback("windowed_mass_metric_decision", {
                    "preparation_attempt_index": preparation_attempt_index,
                    "algorithm_id": route_decision.algorithm_id,
                    "route_contract_version": route_decision.route_contract_version,
                    "route_category": route_decision.algorithm_id,
                    "window": dict(window),
                    "diagnostic_role": "preparation_metric_decision",
                    "artifact_authority": False,
                    "reports_posterior_convergence": False,
                    "raw_states_exposed": False,
                })
        reported_window_count = len(windows)

    def boundary_callback(
        boundary: str,
        _completed_windows: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any] | None:
        report_completed_windows(_completed_windows)
        timeout = _windowed_mass_public_timeout_preflight(
            config,
            stage=f"operational_{boundary}",
            attempt_index=attempt_index,
        )
        if timeout is None:
            return None
        return {
            **dict(timeout),
            "stop_source": "bayesfilter_public_timeout_budget",
            "stop_reason": "public_timeout_budget_exhausted_at_window_boundary",
            "supervision_counter_baseline": sum(
                int(item.get("transition_count_after_window", 0))
                for item in _completed_windows[-1:]
            ),
        }

    def segment_callback(
        event: str,
        segment: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        completed = int(segment["completed_transition_count"])
        if event == "segment_start":
            timeout = _windowed_mass_public_timeout_preflight(
                config,
                stage="operational_segment_start",
                attempt_index=attempt_index,
            )
            _emit_windowed_mass_progress(
                progress_callback,
                "windowed_mass_operational_segment_start",
                attempt_index=attempt_index,
                route_category=route_decision.algorithm_id,
                route_decision=route_decision,
                started=True,
                elapsed_s=0.0,
                started_perf_counter_s=time.perf_counter(),
                operational_progress=segment,
            )
            if timeout is None:
                return None
            return {
                **dict(timeout),
                "stop_source": "bayesfilter_public_timeout_budget",
                "stop_reason": "public_timeout_budget_exhausted_before_next_segment",
                "supervision_counter_baseline": completed,
            }
        _emit_windowed_mass_progress(
            progress_callback,
            "windowed_mass_operational_segment_complete",
            attempt_index=attempt_index,
            route_category=route_decision.algorithm_id,
            route_decision=route_decision,
            completed=True,
            elapsed_s=float(segment.get("segment_elapsed_s", 0.0)),
            operational_progress=segment,
        )
        return None

    _emit_windowed_mass_progress(
        progress_callback,
        "windowed_mass_operational_warmup_start",
        attempt_index=attempt_index,
        route_category=route_decision.algorithm_id,
        route_decision=route_decision,
        started=True,
        elapsed_s=0.0,
        started_perf_counter_s=time.perf_counter(),
    )
    operational_start = time.perf_counter()
    screened_seed = bool(
        attempt_state is None
        and isinstance(mass_window_seed_kernel.get("screen_config_payload"), Mapping)
        and mass_window_seed_kernel.get("acceptance_role", "fixed_kernel_screen") != "warmup_startup_only"
    )
    engineering_probe_config = (
        None
        if config.engineering_probe_covariance_multiplier is None
        else Phase7EngineeringProbeBankConfig(
            chain_count=4,
            covariance_multiplier=config.engineering_probe_covariance_multiplier,
            root_seed=stage_seed,
        )
    )
    if progress_callback is not None and engineering_probe_config is None:
        from bayesfilter.inference.hmc_warmup import metric_schedule_capacity, normalize_operational_warmup_config
        progress_callback("windowed_mass_metric_schedule", {
            "algorithm_id": route_decision.algorithm_id,
            "route_contract_version": route_decision.route_contract_version,
            "route_category": route_decision.algorithm_id,
            **metric_schedule_capacity(normalize_operational_warmup_config(windowed_config),
                                       active_transform.dimension),
        })
    operational_outcome = run_operational_windowed_warmup(
        adapter=adapter,
        initial_transform=active_transform,
        initial_canonical_theta=initial_theta,
        initial_step_size=float(mass_window_seed_kernel["step_size"]),
        initial_step_size_upper_bound=(
            float(mass_window_seed_kernel["step_size"])
            if screened_seed
            else None
        ),
        initial_step_qualification_source=(
            "bayesfilter_fixed_kernel_screen_handoff"
            if screened_seed
            else None
        ),
        trajectory_policy=trajectory_policy,
        config=windowed_config,
        target_accept_prob=config.target_accept_prob,
        seed=stage_seed,
        target_scope=target_scope,
        engineering_probe_config=engineering_probe_config,
        initial_position_covariance_estimate_signature=active_estimate.signature,
        _g2_seed_use_registry=g2_seed_use_registry,
        _g2_p4_action_tracker=g2_p4_action_tracker,
        chain_execution_mode=config.chain_execution_mode,
        jit_compile=config.use_xla,
        target_status_trace_policy=config.target_status_trace_policy,
        algorithm_id=route_decision.algorithm_id,
        route_contract_version=route_decision.route_contract_version,
        boundary_callback=boundary_callback,
        execution_segment_size=_OPERATIONAL_WARMUP_SEGMENT_SIZE,
        segment_callback=segment_callback,
        recovery_callback=recovery_callback,
    )
    if isinstance(operational_outcome, OperationalWindowedWarmupCloseout):
        closeout_payload = _engineering_probe_operational_closeout_public_payload(
            operational_outcome,
            configured=config.engineering_probe_covariance_multiplier is not None,
        )
        _emit_windowed_mass_progress(
            progress_callback,
            "windowed_mass_public_timeout_closeout",
            attempt_index=attempt_index,
            route_category=route_decision.algorithm_id,
            route_decision=route_decision,
            completed=True,
            elapsed_s=operational_outcome.elapsed_s,
            timeout_closeout=closeout_payload,
        )
        capture = dict(_windowed_stage_public_timeout_capture(closeout_payload))
        capture["runtime_metadata"] = {
            "windowed_stage_route_category": route_decision.algorithm_id,
            "operational_windowed_warmup_partial_closeout": True,
            "completed_window_count": len(operational_outcome.completed_windows),
            "planned_window_count": operational_outcome.planned_window_count,
            "completed_transition_count": operational_outcome.completed_transition_count,
            "planned_transition_count": operational_outcome.planned_transition_count,
            "completed_segment_count": operational_outcome.completed_segment_count,
            "planned_segment_count": operational_outcome.planned_segment_count,
            "algorithm_route": route_decision.payload(),
        }
        return None, None, capture, operational_outcome, None
    operational_result = operational_outcome
    report_completed_windows(tuple(window.public_payload() for window in operational_result.windows))
    _emit_windowed_mass_progress(
        progress_callback,
        "windowed_mass_operational_warmup_complete",
        attempt_index=attempt_index,
        route_category=route_decision.algorithm_id,
        route_decision=route_decision,
        completed=True,
        elapsed_s=time.perf_counter() - operational_start,
    )
    effective_config = operational_result.config
    compatibility_mass = compose_operational_transform_in_base_coordinates(
        base_transform=base_transform,
        final_transform=operational_result.final_kernel_state.transform,
        adapter_signature=hmc_adapter_signature,
    )
    authoritative_mass = (
        stage_mass_artifact
        if effective_config.mass_policy == "fixed_identity"
        else compatibility_mass
    )
    canonical_draws = tf.concat(
        [
            tf.reshape(_float64_tensor(window.adaptation_canonical_states),
                (-1, geometry.target_dimension)
            )
            for window in operational_result.windows
        ],
        axis=0,
    )
    original_latent_draws = base_transform.theta_to_latent(canonical_draws)
    log_accept_ratio = tf.concat(
        [_float64_tensor(window.log_accept_ratio) for window in operational_result.windows], axis=0,
    )
    acceptance_probability = tf.exp(tf.minimum(log_accept_ratio, 0.0))
    binary_acceptance = tf.concat(
        [tf.convert_to_tensor(window.is_accepted) for window in operational_result.windows], axis=0,
    )
    target_log_prob = tf.concat(
        [_float64_tensor(window.target_log_prob) for window in operational_result.windows], axis=0,
    )
    policy = HMCTuningPolicy.windowed_mass_adaptation(
        num_adaptation_steps=effective_config.warmup_steps,
        target_accept_prob=config.target_accept_prob,
        source=config.source,
    )
    compatibility_status: Mapping[str, Any]
    try:
        legacy_result = run_windowed_mass_adaptation_diagnostic(
            policy,
            config=effective_config,
            initial_mass_artifact=stage_mass_artifact,
            warmup_draws=original_latent_draws,
            initial_step_size=float(
                operational_result.reasonable_epsilon.selected_step_size
            ),
            acceptance_trace=acceptance_probability,
            expected_adapter_signature=hmc_adapter_signature,
            target_failure_classification={
                "classification": "tuning_diagnostic_passed_not_convergence",
                "diagnostic_role": "diagnostic_only",
                "projection_role": "legacy_v1_compatibility_only",
                "operational_metric_evidence": False,
                "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
            },
        )
        projected_updates = tuple(
            dataclasses.replace(
                update,
                reset_event={
                    **dict(update.reset_event),
                    "diagnostic_role": "legacy_v1_nonoperational_projection",
                    "operational_metric_evidence": False,
                    "replay_as_operational_update_forbidden": True,
                },
            )
            for update in legacy_result.mass_updates
        )
        windowed_result = dataclasses.replace(
            legacy_result,
            mass_updates=projected_updates,
            final_mass_artifact_payload=authoritative_mass.signature_payload(),
            final_mass_artifact_signature=_mass_artifact_signature(authoritative_mass),
            step_size_trace=tuple(
                float(operational_result.final_kernel_state.epsilon)
                for _ in legacy_result.step_size_trace
            ),
            final_mass_artifact=authoritative_mass,
        )
        compatibility_status = {
            "status": "available",
            "authoritative": False,
            "error_type": None,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001 - compatibility is non-authoritative.
        windowed_result = None
        compatibility_status = {
            "status": "unavailable_error",
            "authoritative": False,
            "failure_code": "legacy_v1_compatibility_projection_unavailable",
            "error_type": type(exc).__name__,
            "error_message": None,
            "exception_details_exposed": False,
        }
    binary_trace = tf.cast(binary_acceptance, tf.float64)
    capture = {
        "warmup_draws": original_latent_draws,
        "acceptance_trace": binary_trace,
        "mean_acceptance_probability_trace": acceptance_probability,
        "log_accept_ratio": log_accept_ratio,
        "target_log_prob": target_log_prob,
        "runtime_s": operational_result.elapsed_s,
        "runtime_finite": math.isfinite(operational_result.elapsed_s),
        "samples_shape": tuple(original_latent_draws.shape),
        "acceptance_shape": tuple(binary_trace.shape),
        "log_accept_shape": tuple(log_accept_ratio.shape),
        "target_log_prob_shape": tuple(target_log_prob.shape),
        "expected_steps": effective_config.warmup_steps,
        "target_dimension": geometry.target_dimension,
        "finite_sample_count": int(original_latent_draws.shape[0]),
        "nonfinite_sample_count": 0,
        "raw_diagnostics": {
            "accepted_decision_count": int(tf.reduce_sum(tf.cast(binary_acceptance, tf.int32)).numpy()),
            "acceptance_decision_count": int(tf.size(binary_acceptance).numpy()),
            "acceptance_trace_decision_count": int(binary_acceptance.shape[0]),
            "acceptance_raw_chain_count": 1,
            "raw_acceptance_shape": tuple(binary_acceptance.shape),
            "acceptance_decision_source": "operational_window_binary_trace",
            "mean_acceptance_probability": float(tf.reduce_mean(acceptance_probability).numpy()),
            "operational_metric_update_count": operational_result.operational_metric_update_count,
            "target_status_trace_policy": operational_result.target_status_trace_policy,
            "target_status_failure_count": sum(
                int(window.target_status_failure_count or 0)
                for window in operational_result.windows
            )
            if operational_result.target_status_trace_policy == "per_chain_step"
            else None,
            "consumed_coordinate_signature": active_transform.signature,
            "consumed_step_source": mass_window_seed_kernel.get(
                "private_kernel_source", "bootstrap_or_geometry_handoff"
            ),
            "consumed_num_leapfrog_steps": int(
                mass_window_seed_kernel["num_leapfrog_steps"]
            ),
            "start_bank_qualification": (
                None
                if operational_result.start_bank_qualification is None
                else operational_result.start_bank_qualification.public_payload()
            ),
            "engineering_probe_bank_qualification": None
            if operational_result.engineering_probe_bank_qualification is None
            else (
                operational_result.engineering_probe_bank_qualification.public_payload()
            ),
        },
        "runtime_metadata": {
            "runtime": "tfp.mcmc.operational_interleaved_windowed_warmup",
            "fixture_or_synthetic": False,
            "operational_warmup": True,
            "windowed_stage_route_category": "operational_windowed_warmup",
            "legacy_v1_mass_updates_operational": False,
            "legacy_v1_compatibility_projection": compatibility_status,
            "target_status_trace_policy": operational_result.target_status_trace_policy,
        },
        "runtime_evidence": "tfp_hmc_runtime",
        "fixture_or_synthetic": False,
        "acceptance_trace_key_present": True,
        "acceptance_policy_filled_or_default": False,
        "trace_summary": {
            "trace_keys": (
                "is_accepted",
                "mean_acceptance_probability",
                "log_accept_ratio",
                "target_log_prob",
                *(
                    ("target_status_telemetry",)
                    if operational_result.target_status_trace_policy == "per_chain_step"
                    else ()
                ),
            ),
            "trace_unavailability": {},
        },
    }
    return operational_result, windowed_result, capture, None, authoritative_mass


def _run_p4_windowed_boundary_attempt(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    hmc_adapter_signature: str,
    stage_mass_artifact: PrecomputedMassArtifact,
    mass_window_seed_kernel: Mapping[str, Any],
    windowed_config: WindowedMassAdaptationConfig,
    config: HMCWindowedMassStageConfig,
    target_scope: str,
    attempt_state: "_HMCPhaseAttemptState | None",
    route_decision: HMCAlgorithmRouteDecision,
    progress_callback: LoopProgressCallback | None,
    attempt_index: int | None,
    registry: G2PreboundarySeedUseRegistry,
) -> tuple[
    FullChainHMCConfig | None,
    Mapping[str, Any] | None,
    OperationalWindowedWarmupResult | None,
    WindowedMassAdaptationResult | None,
    Mapping[str, Any],
    OperationalWindowedWarmupCloseout | None,
    PrecomputedMassArtifact | None,
    str | None,
    str | None,
    Mapping[str, Any] | None,
]:
    """Execute the P4 boundary attempt under one carrier-aware failure scope."""

    diagnostic_config: FullChainHMCConfig | None = None
    diagnostic_payload: Mapping[str, Any] | None = None
    operational_result: OperationalWindowedWarmupResult | None = None
    windowed_result: WindowedMassAdaptationResult | None = None
    operational_closeout: OperationalWindowedWarmupCloseout | None = None
    operational_mass: PrecomputedMassArtifact | None = None
    action_tracker = _G2P4BoundaryActionTracker()
    failure_stage = "windowed_stage_seed_derivation"
    try:
        stage_seed = _derive_seed(config.seed, stage_index=0)
        failure_stage = _G2_WINDOWED_STAGE_SEED_GATE_SITE_ID
        stage_seed = registry.consume(
            derivation_site_id=_G2_WINDOWED_STAGE_SEED_DERIVATION_SITE_ID,
            terminal_gate_site_id=_G2_WINDOWED_STAGE_SEED_GATE_SITE_ID,
            key="phase4/stage",
            owner_file="hmc_mass_adaptation.py",
            owner_qualname="_run_p4_windowed_boundary_attempt",
            terminal_consumer="hmc_runner_interface",
            derivation={
                "kind": "derive_stage",
                "base_key": "windowed_stage_config.seed",
                "stage_index": 0,
            },
            indices=(),
            seed=stage_seed,
            interface_hop_site_ids=_G2_WINDOWED_STAGE_SEED_INTERFACE_HOPS,
        )
        failure_stage = "windowed_stage_diagnostic_config"
        diagnostic_config = _windowed_stage_diagnostic_run_config(
            config,
            windowed_config=windowed_config,
            selected_kernel=mass_window_seed_kernel,
            seed=stage_seed,
            target_scope=target_scope,
        )
        failure_stage = "windowed_stage_diagnostic_config_signature"
        diagnostic_payload = diagnostic_config.signature_payload()
        failure_stage = "windowed_stage_operational_capture_entry"
        (
            operational_result,
            windowed_result,
            capture,
            operational_closeout,
            operational_mass,
        ) = _operational_windowed_mass_capture(
            adapter=adapter,
            geometry=geometry,
            hmc_adapter_signature=hmc_adapter_signature,
            stage_mass_artifact=stage_mass_artifact,
            mass_window_seed_kernel=mass_window_seed_kernel,
            windowed_config=windowed_config,
            config=config,
            stage_seed=stage_seed,
            target_scope=target_scope,
            attempt_state=attempt_state,
            route_decision=route_decision,
            progress_callback=progress_callback,
            attempt_index=attempt_index,
            g2_seed_use_registry=registry,
            g2_p4_action_tracker=action_tracker,
        )
        return (
            diagnostic_config,
            diagnostic_payload,
            operational_result,
            windowed_result,
            capture,
            operational_closeout,
            operational_mass,
            None,
            None,
            None,
        )
    except Exception as exc:  # noqa: BLE001 - carrier-first closed P4 route.
        terminal = engineering_probe_bank_qualification_payload_from_exception(exc)
        early = g2_preboundary_shared_invalidity_payload_from_exception(exc)
        if terminal is None and early is None and not registry.p4_seed_consumed:
            wrapped = g2_preboundary_shared_invalidity_exception(
                registry,
                stage=failure_stage,
                cause=exc if isinstance(exc, _G2SeedRegistryError) else None,
            )
            exc = wrapped
        capture, classification, failure_code, private = (
            _p4_boundary_capture_from_exception(
                exc,
                registry=registry,
                action_tracker=action_tracker,
            )
        )
        return (
            diagnostic_config,
            diagnostic_payload,
            None,
            None,
            capture,
            None,
            None,
            classification,
            failure_code,
            private,
        )


def run_hmc_windowed_mass_stage(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    bootstrap: HMCBootstrapScreenResult,
    config: HMCWindowedMassStageConfig | None = None,
    run_full_chain: RunFullChainFn = run_full_chain_tfp_hmc,
    _attempt_budget_policy: "_HMCAttemptBudgetPolicy" | None = None,
    _attempt_state: "_HMCPhaseAttemptState" | None = None,
    _progress_callback: LoopProgressCallback | None = None,
    _attempt_index: int | None = None,
    _checkpoint_writer_config: SequentialRHatCheckpointWriterConfig | None = None,
    _private_diagnostic_callback: PrivateTuningDiagnosticCallback | None = None,
    _g2_seed_use_registry: G2PreboundarySeedUseRegistry | None = None,
    _windowed_config: WindowedMassAdaptationConfig | None = None,
) -> HMCWindowedMassStageResult:
    """Capture retained diagnostic draws and run windowed mass adaptation.

    The retained samples from the fixed-kernel run are adaptation inputs only.
    They are not posterior samples and cannot establish convergence.
    """

    cfg = HMCWindowedMassStageConfig() if config is None else config
    if not isinstance(cfg, HMCWindowedMassStageConfig):
        raise TypeError("config must be HMCWindowedMassStageConfig")
    p4_route = cfg.engineering_probe_covariance_multiplier is not None
    if p4_route and not isinstance(
        _g2_seed_use_registry,
        G2PreboundarySeedUseRegistry,
    ):
        raise TypeError("P4-E requires a caller-owned G2 seed-use registry")
    runner_identity = (
        "default"
        if run_full_chain is run_full_chain_tfp_hmc
        and build_reusable_full_chain_tfp_hmc_runner
        is _OPERATIONAL_WARMUP_DEFAULT_REUSABLE_RUNNER_BUILDER
        else "injected"
    )
    route_decision = require_hmc_algorithm_route(
        algorithm_id=cfg.algorithm_id,
        stage=HMC_WINDOWED_MASS_STAGE,
        runtime_backend="tensorflow",
        chain_execution_mode=cfg.chain_execution_mode,
        use_xla=cfg.use_xla,
        timeout_enabled=cfg.public_timeout_budget_s is not None,
        checkpointing_enabled=_checkpoint_writer_config is not None,
        runner_identity=runner_identity,
    )
    if not isinstance(geometry, HMCGeometryInitializationResult):
        raise TypeError("geometry must be HMCGeometryInitializationResult")
    if not isinstance(bootstrap, HMCBootstrapScreenResult):
        raise TypeError("bootstrap must be HMCBootstrapScreenResult")
    _validate_windowed_stage_inputs(adapter=adapter, geometry=geometry, bootstrap=bootstrap)
    target_scope = _resolve_windowed_stage_target_scope(adapter, cfg)
    mass_signature = _mass_artifact_signature(geometry.mass_artifact)
    hmc_adapter = _build_bootstrap_fixed_mass_adapter(
        adapter=adapter,
        mass_artifact=geometry.mass_artifact,
        mass_signature=mass_signature,
        target_scope=target_scope,
        nonclaims=WINDOWED_MASS_STAGE_NONCLAIMS,
    )
    hmc_adapter_signature = stable_adapter_signature(hmc_adapter)
    if hmc_adapter_signature != bootstrap.hmc_adapter_signature:
        raise ValueError("windowed stage HMC adapter signature must match bootstrap")
    stage_mass_artifact = _windowed_stage_initial_mass_artifact(
        adapter_signature=hmc_adapter_signature,
        target_dimension=geometry.target_dimension,
        attempt_state=_attempt_state,
    )
    stage_mass_signature = _mass_artifact_signature(stage_mass_artifact)
    selected_bootstrap = _active_bootstrap_handoff_kernel_payload(
        geometry=geometry,
        bootstrap=bootstrap,
    )
    selected_hash = _active_bootstrap_handoff_kernel_hash(
        geometry=geometry,
        bootstrap=bootstrap,
    )
    mass_window_seed_kernel = _phase7_windowed_mass_seed_kernel_payload(
        geometry=geometry,
        bootstrap=bootstrap,
        attempt_state=_attempt_state,
    )

    expected_windowed_config = _windowed_mass_stage_internal_config(
        _attempt_budget_policy,
        mass_policy=cfg.mass_policy,
        metric_evidence_policy=cfg.metric_evidence_policy,
        metric_probe_num_results=cfg.metric_probe_num_results,
        preparation_max_restarts=cfg.preparation_max_restarts,
    )
    windowed_config = expected_windowed_config if _windowed_config is None else _windowed_config
    if not isinstance(windowed_config, WindowedMassAdaptationConfig):
        raise TypeError("windowed schedule must be WindowedMassAdaptationConfig")
    for name in ("mass_policy", "metric_evidence_policy", "metric_probe_num_results",
                 "preparation_max_restarts"):
        if getattr(windowed_config, name) != getattr(expected_windowed_config, name):
            raise ValueError(f"windowed schedule {name} mismatch")
    if (_windowed_config is not None and _attempt_budget_policy is not None
            and windowed_config.warmup_steps != _attempt_budget_policy.phase4_warmup_steps):
        raise ValueError("windowed schedule and warmup budget disagree")
    draw_capture_policy = _windowed_stage_draw_capture_policy(windowed_config)
    stage_seed: tuple[int, int] | None = None
    diagnostic_config: FullChainHMCConfig | None = None
    if not p4_route:
        stage_seed = _derive_seed(cfg.seed, stage_index=0)
        diagnostic_config = _windowed_stage_diagnostic_run_config(
            cfg,
            windowed_config=windowed_config,
            selected_kernel=mass_window_seed_kernel,
            seed=stage_seed,
            target_scope=target_scope,
        )
    progress_attempt_index = (
        int(_attempt_budget_policy.attempt_index)
        if _attempt_index is None and _attempt_budget_policy is not None
        else None if _attempt_index is None else int(_attempt_index)
    )
    use_reusable_runner = (
        run_full_chain is run_full_chain_tfp_hmc
        and cfg.chain_execution_mode == "tf_function"
    )
    use_segmented_runner = bool(
        route_decision.algorithm_id == LEGACY_SEGMENTED_WINDOWED_MASS_ALGORITHM_ID
        and run_full_chain is run_full_chain_tfp_hmc
    )
    route_category = (
        "segmented_windowed_mass_runner"
        if use_segmented_runner
        else "reusable_runner" if use_reusable_runner else "injected_runner"
    )

    diagnostics: Mapping[str, Any]
    run_error: Exception | None = None
    diagnostic_run_config_payload: Mapping[str, Any] | None = (
        None if diagnostic_config is None else diagnostic_config.signature_payload()
    )
    windowed_result: WindowedMassAdaptationResult | None = None
    operational_warmup_result: OperationalWindowedWarmupResult | None = None
    operational_warmup_closeout: OperationalWindowedWarmupCloseout | None = None
    operational_mass_artifact: PrecomputedMassArtifact | None = None
    use_operational_warmup = bool(
        route_decision.algorithm_id == OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    )
    p4_boundary_classification: str | None = None
    p4_boundary_failure_code: str | None = None
    p4_private_seed_evidence: Mapping[str, Any] | None = None
    timeout_closeout = (
        None
        if p4_route
        else _windowed_mass_public_timeout_preflight(
            cfg,
            stage="windowed_mass_runner_build_start",
            attempt_index=progress_attempt_index,
        )
    )
    if p4_route:
        if not use_operational_warmup:
            raise ValueError("P4-E requires the operational windowed-warmup route")
        assert _g2_seed_use_registry is not None
        (
            diagnostic_config,
            diagnostic_run_config_payload,
            operational_warmup_result,
            windowed_result,
            capture,
            operational_warmup_closeout,
            operational_mass_artifact,
            p4_boundary_classification,
            p4_boundary_failure_code,
            p4_private_seed_evidence,
        ) = _run_p4_windowed_boundary_attempt(
            adapter=adapter,
            geometry=geometry,
            hmc_adapter_signature=hmc_adapter_signature,
            stage_mass_artifact=stage_mass_artifact,
            mass_window_seed_kernel=mass_window_seed_kernel,
            windowed_config=windowed_config,
            config=cfg,
            target_scope=target_scope,
            attempt_state=_attempt_state,
            route_decision=route_decision,
            progress_callback=_progress_callback,
            attempt_index=progress_attempt_index,
            registry=_g2_seed_use_registry,
        )
        if diagnostic_config is not None:
            stage_seed = diagnostic_config.seed
        route_category = route_decision.algorithm_id
    elif timeout_closeout is not None:
        _emit_windowed_mass_progress(
            _progress_callback,
            "windowed_mass_public_timeout_closeout",
            attempt_index=progress_attempt_index,
            route_category=route_category,
            route_decision=route_decision,
            completed=True,
            elapsed_s=0.0,
            timeout_closeout=timeout_closeout,
        )
        capture = _windowed_stage_public_timeout_capture(timeout_closeout)
    else:
        try:
            if use_operational_warmup:
                (
                    operational_warmup_result,
                    windowed_result,
                    capture,
                    operational_warmup_closeout,
                    operational_mass_artifact,
                ) = _operational_windowed_mass_capture(
                    adapter=adapter,
                    geometry=geometry,
                    hmc_adapter_signature=hmc_adapter_signature,
                    stage_mass_artifact=stage_mass_artifact,
                    mass_window_seed_kernel=mass_window_seed_kernel,
                    windowed_config=windowed_config,
                    config=cfg,
                    stage_seed=stage_seed,
                    target_scope=target_scope,
                    attempt_state=_attempt_state,
                    route_decision=route_decision,
                    progress_callback=_progress_callback,
                    attempt_index=progress_attempt_index,
                )
                route_category = route_decision.algorithm_id
            else:
                _emit_windowed_mass_progress(
                    _progress_callback,
                    "windowed_mass_runner_build_start",
                    attempt_index=progress_attempt_index,
                    route_category=route_category,
                    route_decision=route_decision,
                    started=True,
                    elapsed_s=0.0,
                    started_perf_counter_s=time.perf_counter(),
                )
                runner_build_start = time.perf_counter()
                runner_build_s = 0.0
                if use_segmented_runner:
                    segment_size = max(
                        1,
                        min(_WINDOWED_MASS_SEGMENT_SIZE, int(windowed_config.warmup_steps)),
                    )
                    initial_chunk_config = _windowed_stage_chunk_run_config(
                        cfg,
                        diagnostic_config=diagnostic_config,
                        max_results=segment_size,
                        num_burnin_steps=int(diagnostic_config.num_burnin_steps),
                    )
                    continuation_chunk_config = _windowed_stage_chunk_run_config(
                        cfg,
                        diagnostic_config=diagnostic_config,
                        max_results=segment_size,
                        num_burnin_steps=0,
                    )
                    segmented_initial_runner = build_fixed_size_hmc_chunk_runner(
                        hmc_adapter,
                        hmc_adapter.initial_position(),
                        initial_chunk_config,
                    )
                    segmented_continuation_runner = (
                        segmented_initial_runner
                        if int(diagnostic_config.num_burnin_steps) == 0
                        else build_fixed_size_hmc_chunk_runner(
                            hmc_adapter,
                            hmc_adapter.initial_position(),
                            continuation_chunk_config,
                        )
                    )
                elif use_reusable_runner:
                    reusable_runner = build_reusable_full_chain_tfp_hmc_runner(
                        hmc_adapter,
                        hmc_adapter.initial_position(),
                        diagnostic_config,
                    )
                runner_build_s = time.perf_counter() - runner_build_start
                _emit_windowed_mass_progress(
                    _progress_callback,
                    "windowed_mass_runner_build_complete",
                    attempt_index=progress_attempt_index,
                    route_category=route_category,
                    route_decision=route_decision,
                    completed=True,
                    elapsed_s=runner_build_s,
                )
                timeout_closeout = _windowed_mass_public_timeout_preflight(
                    cfg,
                    stage="windowed_mass_runner_execute_start",
                    attempt_index=progress_attempt_index,
                )
                if timeout_closeout is not None:
                    _emit_windowed_mass_progress(
                        _progress_callback,
                        "windowed_mass_public_timeout_closeout",
                        attempt_index=progress_attempt_index,
                        route_category=route_category,
                        route_decision=route_decision,
                        completed=True,
                        elapsed_s=0.0,
                        timeout_closeout=timeout_closeout,
                    )
                    capture = _windowed_stage_public_timeout_capture(timeout_closeout)
                else:
                    checkpoint_reference = None
                    checkpoint_reference_public_safe = None
                    if _checkpoint_writer_config is not None:
                        checkpoint_reference = write_sequential_rhat_boundary_handoff_checkpoint(
                            writer_config=_checkpoint_writer_config,
                            adapter=hmc_adapter,
                            config_private_payload={
                                "schema": "bayesfilter.phase7_boundary_config_private.v1",
                                "source": "run_hmc_windowed_mass_stage",
                                "stage": "windowed_mass_runner_execute_start",
                                "target_scope": target_scope,
                                "target_dimension": int(geometry.target_dimension),
                                "attempt_index": progress_attempt_index,
                                "route_category": route_category,
                                "chain_execution_mode": cfg.chain_execution_mode,
                                "use_xla": bool(cfg.use_xla),
                                "budget_payload": None
                                if _attempt_budget_policy is None
                                else _attempt_budget_policy.payload(),
                            },
                            boundary_private_payload={
                                "schema": "bayesfilter.phase7_boundary_handoff_private.v1",
                                "stage": "windowed_mass_runner_execute_start",
                                "target_scope": target_scope,
                                "target_dimension": int(geometry.target_dimension),
                                "attempt_index": progress_attempt_index,
                                "route_category": route_category,
                                "hmc_adapter_signature": hmc_adapter_signature,
                                "bootstrap_artifact_hash": bootstrap.artifact_hash,
                                "geometry_artifact_hash": geometry.artifact_hash,
                                "windowed_config_hash": stable_config_hash(windowed_config.payload()),
                                "diagnostic_config_hash": stable_config_hash(
                                    diagnostic_config.signature_payload()
                                ),
                                "budget_payload": None
                                if _attempt_budget_policy is None
                                else _attempt_budget_policy.payload(),
                                "private_raw_state_allowed": False,
                                "private_raw_samples_allowed": False,
                                "private_hmc_mechanics_allowed": False,
                                "milestone": "target_forced_stop_observability_checkpoint",
                                "reports_verifier_entry_durability": False,
                            },
                            state_summary_private_payload={
                                "schema": "bayesfilter.phase7_boundary_state_summary_private.v1",
                                "summary_only": True,
                                "initial_state_shape": tuple(
                                    int(dim) for dim in _float64_tensor(hmc_adapter.initial_position()).shape
                                ),
                                "raw_state_included": False,
                                "raw_samples_included": False,
                                "tensor_payload_included": False,
                            },
                        )
                        assert_sequential_rhat_checkpoint_public_reference_safe(
                            checkpoint_reference
                        )
                        checkpoint_reference_public_safe = True
                    _emit_windowed_mass_progress(
                        _progress_callback,
                        "windowed_mass_runner_execute_start",
                        attempt_index=progress_attempt_index,
                        route_category=route_category,
                        route_decision=route_decision,
                        started=True,
                        elapsed_s=0.0,
                        started_perf_counter_s=time.perf_counter(),
                        checkpoint_reference=checkpoint_reference,
                        checkpoint_reference_public_safe=checkpoint_reference_public_safe,
                    )
                    runner_execute_start = time.perf_counter()
                    segmented_timeout_closeout = False
                    if use_segmented_runner:
                        capture = _windowed_stage_segmented_capture_payload(
                            config=cfg,
                            hmc_adapter=hmc_adapter,
                            initial_runner=segmented_initial_runner,
                            continuation_runner=segmented_continuation_runner,
                            diagnostic_config=diagnostic_config,
                            windowed_config=windowed_config,
                            target_dimension=geometry.target_dimension,
                            progress_callback=_progress_callback,
                            attempt_index=progress_attempt_index,
                            route_category=route_category,
                            route_decision=route_decision,
                        )
                        segmented_timeout_closeout = (
                            capture.get("public_timeout_closeout") is not None
                        )
                        run_result = None
                    elif use_reusable_runner:
                        run_result = reusable_runner.run(
                            current_state=hmc_adapter.initial_position(),
                            seed=diagnostic_config.seed,
                            step_size=diagnostic_config.step_size,
                        )
                    else:
                        run_result = run_full_chain(
                            hmc_adapter,
                            hmc_adapter.initial_position(),
                            diagnostic_config,
                        )
                    runner_execute_s = time.perf_counter() - runner_execute_start
                    if segmented_timeout_closeout:
                        capture = _with_windowed_stage_timing_metadata(
                            capture,
                            runner_build_s=runner_build_s,
                            runner_execute_s=runner_execute_s,
                            capture_s=0.0,
                            route_category=route_category,
                        )
                    else:
                        _emit_windowed_mass_progress(
                            _progress_callback,
                            "windowed_mass_runner_execute_complete",
                            attempt_index=progress_attempt_index,
                            route_category=route_category,
                            route_decision=route_decision,
                            completed=True,
                            elapsed_s=runner_execute_s,
                        )
                        _emit_windowed_mass_progress(
                            _progress_callback,
                            "windowed_mass_capture_start",
                            attempt_index=progress_attempt_index,
                            route_category=route_category,
                            route_decision=route_decision,
                            started=True,
                            elapsed_s=0.0,
                            started_perf_counter_s=time.perf_counter(),
                        )
                        capture_start = time.perf_counter()
                        if not use_segmented_runner:
                            capture = _windowed_stage_capture_payload(
                                run_result,
                                expected_steps=windowed_config.warmup_steps,
                                target_dimension=geometry.target_dimension,
                            )
                        capture_s = time.perf_counter() - capture_start
                        capture = _with_windowed_stage_timing_metadata(
                            capture,
                            runner_build_s=runner_build_s,
                            runner_execute_s=runner_execute_s,
                            capture_s=capture_s,
                            route_category=route_category,
                        )
                        _emit_windowed_mass_progress(
                            _progress_callback,
                            "windowed_mass_capture_complete",
                            attempt_index=progress_attempt_index,
                            route_category=route_category,
                            route_decision=route_decision,
                            completed=True,
                            elapsed_s=capture_s,
                        )
        except Exception as exc:  # noqa: BLE001 - return fail-closed artifact.
            run_error = exc
            capture = _windowed_stage_error_capture(exc)
    start_bank_qualification = capture.get("raw_diagnostics", {}).get(
        "start_bank_qualification"
    )
    if _private_diagnostic_callback is not None and isinstance(
        start_bank_qualification,
        Mapping,
    ):
        _private_diagnostic_callback(
            "start_bank_qualification",
            {
                "stage": "windowed_mass_start_bank_boundary",
                "attempt_index": progress_attempt_index,
                "start_bank_qualification": _json_ready(
                    start_bank_qualification
                ),
                "private_hmc_mechanics": False,
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
            },
        )
    engineering_probe_qualification = capture.get("raw_diagnostics", {}).get(
        "engineering_probe_bank_qualification",
        capture.get("raw_diagnostics", {}).get("engineering_probe_boundary"),
    )
    if _private_diagnostic_callback is not None and isinstance(
        engineering_probe_qualification,
        Mapping,
    ):
        _private_diagnostic_callback(
            "engineering_probe_bank_qualification",
            {
                "stage": "phase7_engineering_probe_bank_boundary",
                "attempt_index": progress_attempt_index,
                "engineering_probe_bank_qualification": _json_ready(
                    engineering_probe_qualification
                ),
                "private_hmc_mechanics": False,
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
            },
        )
    if p4_boundary_classification == "candidate_rejected":
        hard_vetoes: list[str] = []
    elif p4_boundary_classification == "shared_implementation_invalid":
        hard_vetoes = [str(p4_boundary_failure_code)]
    else:
        hard_vetoes = list(
            _classify_windowed_stage_capture(
                capture,
                run_error=run_error,
            )
        )
    if (
        not hard_vetoes
        and capture.get("public_timeout_closeout") is None
        and not use_operational_warmup
    ):
        _emit_windowed_mass_progress(
            _progress_callback,
            "windowed_mass_semantic_diagnostic_start",
            attempt_index=progress_attempt_index,
            route_category=route_category,
            route_decision=route_decision,
            started=True,
            elapsed_s=0.0,
            started_perf_counter_s=time.perf_counter(),
        )
        semantic_diagnostic_start = time.perf_counter()
        try:
            policy = HMCTuningPolicy.windowed_mass_adaptation(
                num_adaptation_steps=windowed_config.warmup_steps,
                target_accept_prob=cfg.target_accept_prob,
                source=cfg.source,
            )
            semantic_config = dataclasses.replace(
                windowed_config,
                mass_policy=cfg.mass_policy,
            )
            windowed_result = run_windowed_mass_adaptation_diagnostic(
                policy,
                config=semantic_config,
                initial_mass_artifact=stage_mass_artifact,
                warmup_draws=capture["warmup_draws"],
                initial_step_size=float(mass_window_seed_kernel["step_size"]),
                acceptance_trace=capture["acceptance_trace"],
                expected_adapter_signature=hmc_adapter_signature,
                target_failure_classification={
                    "classification": "tuning_diagnostic_passed_not_convergence",
                    "diagnostic_role": "diagnostic_only",
                    "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
                },
            )
            if not windowed_result.passed:
                hard_vetoes.append("windowed_mass_diagnostic_hard_veto")
        except Exception as exc:  # noqa: BLE001 - return fail-closed artifact.
            hard_vetoes.append("windowed_mass_diagnostic_error")
            capture = {
                **dict(capture),
                "windowed_mass_error_type": type(exc).__name__,
                "windowed_mass_error_message": str(exc),
            }
        _emit_windowed_mass_progress(
            _progress_callback,
            "windowed_mass_semantic_diagnostic_complete",
            attempt_index=progress_attempt_index,
            route_category=route_category,
            route_decision=route_decision,
            completed=True,
            elapsed_s=time.perf_counter() - semantic_diagnostic_start,
        )
    resource_closeout = capture.get("public_timeout_closeout")
    repair_triggers: tuple[str, ...] = ()
    required_update_missing = bool(
        cfg.metric_update_requirement == "require_operational_update"
        and operational_warmup_result is not None
        and operational_warmup_result.operational_metric_update_count == 0
    )
    if p4_boundary_classification == "candidate_rejected":
        final_status = "candidate_rejected"
        diagnostic_role = "p4e_candidate_boundary_rejection"
    elif p4_boundary_classification == "shared_implementation_invalid":
        final_status = "hard_veto"
        diagnostic_role = "shared_implementation_invalid"
    elif hard_vetoes:
        final_status = "hard_veto"
        diagnostic_role = "hard_veto"
    elif isinstance(resource_closeout, Mapping):
        final_status = _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_STATUS
        diagnostic_role = _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_ROLE
        repair_triggers = (_WINDOWED_MASS_PUBLIC_TIMEOUT_REPAIR_TRIGGER,)
    elif required_update_missing:
        final_status = "passed_no_metric_update"
        diagnostic_role = "metric_adaptation_not_observed"
        repair_triggers = (_NO_OPERATIONAL_METRIC_UPDATE_REPAIR_TRIGGER,)
    else:
        final_status = "passed"
        diagnostic_role = "windowed_mass_stage_handoff_only"
    diagnostics = dict(_windowed_stage_diagnostics(
        capture,
        windowed_result=windowed_result,
        hard_vetoes=tuple(hard_vetoes),
        mass_window_seed_kernel=mass_window_seed_kernel,
        bootstrap_kernel=selected_bootstrap,
    ))
    engineering_probe_boundary = capture.get("raw_diagnostics", {}).get(
        "engineering_probe_boundary"
    )
    if isinstance(engineering_probe_boundary, Mapping):
        diagnostics["engineering_probe_boundary"] = dict(
            engineering_probe_boundary
        )
        diagnostics["passed"] = False
        diagnostics["p4_private_seed_evidence_available"] = (
            p4_private_seed_evidence is not None
        )
    diagnostics.update(
        {
            "algorithm_id": route_decision.algorithm_id,
            "route_contract_version": route_decision.route_contract_version,
            "algorithm_route": route_decision.payload(),
            "evidence_role": route_decision.evidence_role,
            "promotion_role": route_decision.promotion_role,
            "stopping_rule_role": route_decision.stopping_rule_role,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "metric_update_requirement": cfg.metric_update_requirement,
            "metric_adaptation_status": None
            if operational_warmup_result is None
            else operational_warmup_result.metric_adaptation_status,
            "required_operational_metric_update_missing": required_update_missing,
        }
    )
    operational_work_manifest_summary = None
    if (
        use_operational_warmup
        and _attempt_budget_policy is not None
        and p4_boundary_classification is None
    ):
        operational_policy = HMCOperationalStatisticalWorkPolicy(
            initial_candidate_results=int(
                _attempt_budget_policy.operational_screen_num_results
            ),
            candidate_burnin_steps=int(
                _attempt_budget_policy.operational_screen_num_burnin_steps
            ),
            evidence_extension_checkpoints=tuple(
                int(item)
                for item in _attempt_budget_policy.operational_evidence_extension_checkpoints
            ),
            exact_l_tune_adaptation_steps=int(
                _attempt_budget_policy.operational_exact_l_tune_adaptation_steps
            ),
            fresh_verification_results=int(
                _attempt_budget_policy.operational_verification_num_results
            ),
            fresh_verification_burnin_steps=int(
                _attempt_budget_policy.operational_verification_num_burnin_steps
            ),
            fresh_verification_starts_per_outer_attempt=int(
                _attempt_budget_policy.operational_verification_starts_per_outer_attempt
            ),
            policy_id=str(_attempt_budget_policy.operational_budget_policy_id),
        )
        public_manifest = build_public_hmc_work_manifest(
            target_dimension=geometry.target_dimension,
            metric_adaptation_steps=(int(_attempt_budget_policy.phase4_warmup_steps),),
            selection_attempts_per_outer_attempt=(1,),
            max_leapfrog_steps=_GEOMETRY_MAX_LEAPFROG,
            policy=operational_policy,
            algorithm_id=ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID,
            run_class="serious",
            route_marker=BROAD_FIXED_METRIC_OPERATIONAL_ROUTE,
            per_l_tune_budget_schedule=tuple(
                _attempt_budget_policy.phase5_tune_budgets
            ),
            per_l_tune_num_results=_FIXED_MASS_STAGE_TUNE_NUM_RESULTS,
            per_l_screen_num_results=int(
                _attempt_budget_policy.phase5_screen_num_results
            ),
            per_l_screen_num_burnin_steps=int(
                _attempt_budget_policy.phase5_screen_burnin_steps
            ),
        )
        completed_metric_transitions = (
            0
            if operational_warmup_closeout is None
            else int(operational_warmup_closeout.completed_transition_count)
        )
        executed_work = {
            "metric_adaptation_batched_transitions": completed_metric_transitions,
            "initial_candidate_batched_transitions": 0,
            "extension_candidate_batched_transitions": 0,
            "exact_l_tune_batched_transitions": 0,
            "fresh_verification_batched_transitions": 0,
            "total_batched_transitions": completed_metric_transitions,
        }
        reconciliation = reconcile_executed_hmc_work(
            public_manifest=public_manifest,
            executed_work=executed_work,
        )
        operational_work_manifest_summary = {
            "schema": "bayesfilter.hmc_operational_partial_work_summary.v1",
            "policy_id": public_manifest["policy_id"],
            "policy_hash": public_manifest["policy_hash"],
            "public_manifest_hash": public_manifest["manifest_hash"],
            "executed_work_reconciliation": reconciliation,
            "accounting_scope": "partial_metric_adaptation_before_selection",
            "fresh_verification_accounted": False,
            "partial_resource_closeout": isinstance(resource_closeout, Mapping),
            "stop_source": None
            if not isinstance(resource_closeout, Mapping)
            else resource_closeout.get("stop_source"),
            "stop_reason": None
            if not isinstance(resource_closeout, Mapping)
            else resource_closeout.get("stop_reason"),
            "supervision_counter_baseline": None
            if not isinstance(resource_closeout, Mapping)
            else resource_closeout.get("supervision_counter_baseline"),
            "private_hmc_mechanics_exposed": False,
            "aggregate_counts_public_safe": True,
        }
        diagnostics["operational_work_manifest_summary"] = (
            operational_work_manifest_summary
        )
    if (
        _private_diagnostic_callback is not None
        and windowed_result is not None
        and windowed_result.final_mass_artifact is not None
    ):
        _private_diagnostic_callback(
            "windowed_mass_matrix_change",
            {
                "stage": "windowed_mass_complete",
                "attempt_index": progress_attempt_index,
                "final_status": final_status,
                "diagnostic_role": diagnostic_role,
                "initial_mass_artifact_signature": stage_mass_signature,
                "adapted_mass_artifact_signature": (
                    windowed_result.final_mass_artifact_signature
                ),
                "candidate_step_size": windowed_result.final_step_size,
                "step_size_trace": windowed_result.step_size_trace,
                "acceptance_trace": windowed_result.acceptance_trace,
                "mass_artifact": windowed_result.final_mass_artifact,
                "private_hmc_mechanics": True,
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
            },
        )

    return HMCWindowedMassStageResult(
        config=cfg,
        geometry_artifact_hash=geometry.artifact_hash,
        bootstrap_artifact_hash=bootstrap.artifact_hash,
        selected_bootstrap_kernel_hash=selected_hash,
        adapter_signature=geometry.adapter_signature,
        hmc_adapter_signature=hmc_adapter_signature,
        initial_mass_artifact_signature=stage_mass_signature,
        target_dimension=geometry.target_dimension,
        final_status=final_status,
        diagnostic_role=diagnostic_role,
        hard_vetoes=tuple(hard_vetoes),
        repair_triggers=repair_triggers,
        diagnostics=diagnostics,
        draw_capture_policy=draw_capture_policy,
        warmup_draw_provenance=_warmup_draw_provenance(capture, draw_capture_policy),
        acceptance_telemetry_provenance=_acceptance_telemetry_provenance(capture),
        diagnostic_run_config_payload=diagnostic_run_config_payload,
        windowed_config_payload=(
            operational_warmup_result.config.payload()
            if operational_warmup_result is not None
            else windowed_config.payload()
        ),
        windowed_mass_result=windowed_result,
        operational_warmup_result=operational_warmup_result,
        operational_warmup_closeout=operational_warmup_closeout,
        operational_mass_artifact=operational_mass_artifact,
        seed_report={
            "geometry_root_seed": geometry.seed_report.get("root_seed"),
            "bootstrap_root_seed": bootstrap.seed_report.get("bootstrap_root_seed"),
            "windowed_stage_root_seed": cfg.seed,
            "windowed_stage_seed": stage_seed,
            "seed_owner": "BayesFilter",
        },
        diagnostic_roles={
            "retained_samples": "adaptation_input_only",
            "acceptance_trace": "hard_veto_and_step_handoff_input",
            "log_accept_ratio": "hard_veto",
            "target_log_prob": "hard_veto",
            "runtime": "hard_veto",
            "windowed_mass_artifact": "phase5_handoff_only",
        },
    )


def _resolve_windowed_stage_target_scope(
    adapter: Any,
    config: HMCWindowedMassStageConfig,
) -> str:
    capability = value_score_capability(adapter)
    capability_scope = capability.target_scope
    if config.target_scope is not None:
        target_scope = str(config.target_scope)
        if not target_scope:
            raise ValueError("target_scope must be non-empty when provided")
        if capability_scope is not None and target_scope != capability_scope:
            raise ValueError("value/score target_scope mismatch")
        return target_scope
    if capability_scope is None or not str(capability_scope):
        raise ValueError(
            "windowed mass stage requires config.target_scope or an adapter "
            "value_score_capability target_scope"
        )
    return str(capability_scope)


def _validate_windowed_stage_inputs(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    bootstrap: HMCBootstrapScreenResult,
) -> None:
    adapter_signature = stable_adapter_signature(adapter)
    if adapter_signature != geometry.adapter_signature:
        raise ValueError("windowed stage adapter signature must match geometry")
    if adapter_signature != bootstrap.adapter_signature:
        raise ValueError("windowed stage adapter signature must match bootstrap")
    if not _bootstrap_preflight_passed(bootstrap):
        raise ValueError("windowed stage requires bootstrap preflight without hard veto")
    geometry.mass_artifact.validate_for_adapter(
        adapter,
        expected_dim=geometry.target_dimension,
    )
    mass_signature = _mass_artifact_signature(geometry.mass_artifact)
    if mass_signature != geometry.mass_artifact_signature:
        raise ValueError("geometry mass artifact signature mismatch")
    if mass_signature != bootstrap.mass_artifact_signature:
        raise ValueError("windowed stage mass artifact signature must match bootstrap")
    if bootstrap.geometry_artifact_hash != geometry.artifact_hash:
        raise ValueError("windowed stage bootstrap must match geometry artifact")
    if geometry.target_dimension != bootstrap.target_dimension:
        raise ValueError("windowed stage target dimension must match bootstrap")


def _emit_windowed_mass_progress(
    callback: LoopProgressCallback | None,
    stage: str,
    *,
    attempt_index: int | None,
    route_category: str,
    route_decision: HMCAlgorithmRouteDecision,
    started: bool = False,
    completed: bool = False,
    elapsed_s: float | None = None,
    started_perf_counter_s: float | None = None,
    checkpoint_reference: Mapping[str, Any] | None = None,
    checkpoint_reference_public_safe: bool | None = None,
    timeout_closeout: Mapping[str, Any] | None = None,
    segment_index: int | None = None,
    segment_count: int | None = None,
    segment_active_results: int | None = None,
    operational_progress: Mapping[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    payload: dict[str, Any] = {
        "stage": str(stage),
        "started": bool(started),
        "completed": bool(completed),
        "progress_only": True,
        "hmc_mechanics_exposed": False,
        "route_category": str(route_category),
        "algorithm_id": route_decision.algorithm_id,
        "route_contract_version": route_decision.route_contract_version,
        "algorithm_route": route_decision.payload(),
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_external_client_scientific_claim": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
    }
    if attempt_index is not None:
        payload["attempt_index"] = int(attempt_index)
    if started_perf_counter_s is not None:
        payload["started_perf_counter_s"] = float(started_perf_counter_s)
        payload["timing_anchor_role"] = "process_local_monotonic_debug_only"
    if elapsed_s is not None:
        payload["elapsed_s"] = float(elapsed_s)
    if checkpoint_reference is not None:
        payload["checkpoint_reference"] = dict(checkpoint_reference)
    if checkpoint_reference_public_safe is not None:
        payload["checkpoint_reference_public_safe"] = bool(checkpoint_reference_public_safe)
    if timeout_closeout is not None:
        payload["public_timeout_closeout"] = dict(timeout_closeout)
    if segment_index is not None:
        payload["segment_index"] = int(segment_index)
    if segment_count is not None:
        payload["segment_count"] = int(segment_count)
    if segment_active_results is not None:
        payload["segment_active_results"] = int(segment_active_results)
    if operational_progress is not None:
        payload["operational_progress"] = dict(operational_progress)
    callback(str(stage), payload)


def _staged_timeout_round(
    rounds: Mapping[str, int] | None,
    stage: str,
) -> int:
    if rounds is None:
        return 0
    return int(rounds.get(str(stage), 0))


def _staged_timeout_public_state(
    *,
    policy: HMCStagedTimeoutPolicy,
    stage: str,
    attempt_index: int | None,
    global_started_perf_counter_s: float | None,
    stage_started_perf_counter_s: float | None,
    enlargement_rounds: Mapping[str, int] | None,
) -> Mapping[str, Any]:
    now = time.perf_counter()
    global_anchor = (
        now
        if global_started_perf_counter_s is None
        else float(global_started_perf_counter_s)
    )
    stage_anchor = (
        now
        if stage_started_perf_counter_s is None
        else float(stage_started_perf_counter_s)
    )
    stage_name = str(stage)
    stage_budget = float(policy.stage_budgets_s[stage_name])
    stage_elapsed = max(0.0, now - stage_anchor)
    global_elapsed = max(0.0, now - global_anchor)
    stage_remaining = stage_budget - stage_elapsed
    global_remaining = float(policy.global_cap_s) - global_elapsed
    enlargement_round = _staged_timeout_round(enlargement_rounds, stage_name)
    cap_hit = bool(
        global_remaining <= float(policy.reserve_s)
        or stage_remaining <= float(policy.reserve_s)
    )
    payload: dict[str, Any] = {
        "policy_id": policy.policy_id,
        "stage": stage_name,
        "attempt_index": None if attempt_index is None else int(attempt_index),
        "stage_timeout_budget_s": stage_budget,
        "stage_elapsed_s": stage_elapsed,
        "stage_remaining_s": stage_remaining,
        "stage_budget_provenance": policy.stage_budget_provenance[stage_name],
        "global_cap_s": policy.global_cap_s,
        "global_elapsed_s": global_elapsed,
        "global_remaining_s": global_remaining,
        "timeout_role": "emergency_cap_machine_protection_only",
        "progress_monitor_role": "meaningful_progress_decides_stall_separately",
        "no_progress_timeout_is_separate": True,
        "enlargement_round": enlargement_round,
        "max_enlargement_rounds": policy.max_enlargement_rounds_per_stage,
        "cap_hit": cap_hit,
        "hard_veto": None,
        "repair_trigger": stage_name,
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "nonclaims": TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS,
    }
    if enlargement_round < int(policy.max_enlargement_rounds_per_stage):
        proposed = min(
            stage_budget * float(policy.enlargement_multiplier),
            max(0.0, global_remaining),
        )
        payload["proposed_stage_timeout_budget_s"] = proposed
        payload["repair_loop_available"] = bool(
            proposed > stage_budget + float(policy.reserve_s)
        )
    else:
        payload["repair_loop_available"] = False
    return payload


def _staged_timeout_public_payload_for_config(
    config: HMCWindowedMassStageConfig,
    *,
    stage: str,
    attempt_index: int | None,
) -> Mapping[str, Any] | None:
    policy = config.staged_timeout_policy
    if policy is None or not policy.enabled:
        return None
    return _staged_timeout_public_state(
        policy=policy,
        stage=stage,
        attempt_index=attempt_index,
        global_started_perf_counter_s=(
            config.staged_timeout_global_started_perf_counter_s
        ),
        stage_started_perf_counter_s=(
            config.staged_timeout_stage_started_perf_counter_s
        ),
        enlargement_rounds=config.staged_timeout_enlargement_rounds,
    )


def _windowed_mass_public_timeout_state(
    config: HMCWindowedMassStageConfig,
) -> Mapping[str, Any]:
    staged = _staged_timeout_public_payload_for_config(
        config,
        stage="windowed_mass",
        attempt_index=None,
    )
    if staged is not None:
        return {
            "enabled": True,
            "timeout_budget_s": staged["stage_timeout_budget_s"],
            "reserve_s": config.staged_timeout_policy.reserve_s,
            "elapsed_s": staged["stage_elapsed_s"],
            "remaining_s": staged["stage_remaining_s"],
            "within_closeout_window": staged["cap_hit"],
            "deadline_clock_scope": "staged_timeout_windowed_mass_stage_local",
            "staged_timeout": staged,
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
        }
    if config.public_timeout_budget_s is None:
        return {
            "enabled": False,
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
        }
    budget = float(config.public_timeout_budget_s)
    reserve = max(0.0, min(_WINDOWED_MASS_PUBLIC_TIMEOUT_RESERVE_S, budget * 0.5))
    now = time.perf_counter()
    anchor = (
        now
        if config.public_timeout_started_perf_counter_s is None
        else float(config.public_timeout_started_perf_counter_s)
    )
    elapsed = max(0.0, now - anchor)
    remaining = budget - elapsed
    return {
        "enabled": True,
        "timeout_budget_s": budget,
        "reserve_s": reserve,
        "elapsed_s": elapsed,
        "remaining_s": remaining,
        "within_closeout_window": remaining <= reserve,
        "deadline_clock_scope": (
            "windowed_mass_stage_local"
            if config.public_timeout_started_perf_counter_s is None
            else "public_one_call_global"
        ),
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
    }


def _windowed_mass_public_timeout_preflight(
    config: HMCWindowedMassStageConfig,
    *,
    stage: str,
    attempt_index: int | None,
) -> Mapping[str, Any] | None:
    if config.public_timeout_budget_s is None:
        return None
    state = dict(_windowed_mass_public_timeout_state(config))
    closeout_required = bool(
        state["within_closeout_window"]
        or float(state["remaining_s"]) <= float(state["reserve_s"])
    )
    if not closeout_required:
        return None
    payload: dict[str, Any] = {
        **state,
        "schema": "bayesfilter.windowed_mass_public_timeout_closeout.v1",
        "stage": str(stage),
        "closeout_required_before_hmc_call": True,
        "stop_source": "bayesfilter_public_timeout_budget",
        "stop_reason": "public_timeout_budget_exhausted_before_hmc_call",
        "supervision_counter_baseline": 0,
        "final_status": _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_STATUS,
        "diagnostic_role": _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_ROLE,
        "hard_veto": None,
        "repair_trigger": _WINDOWED_MASS_PUBLIC_TIMEOUT_REPAIR_TRIGGER,
        "progress_only": True,
        "public_closeout_artifact_expected": True,
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_external_client_scientific_claim": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
    }
    if attempt_index is not None:
        payload["attempt_index"] = int(attempt_index)
    return payload


def _windowed_mass_next_segment_soft_deadline_preflight(
    config: HMCWindowedMassStageConfig,
    *,
    stage: str,
    attempt_index: int | None,
    completed_segment_elapsed_s: Sequence[float],
) -> Mapping[str, Any] | None:
    if config.public_timeout_budget_s is None:
        return None
    state = dict(_windowed_mass_public_timeout_state(config))
    completed = tuple(float(value) for value in completed_segment_elapsed_s)
    if not completed:
        estimated_next_s = min(
            _WINDOWED_MASS_PUBLIC_TIMEOUT_RESERVE_S,
            max(1.0, float(state["timeout_budget_s"]) * 0.25),
        )
        estimator = "fallback_min_reserve_or_quarter_budget"
        recent_window_count = 0
    else:
        recent_window = max(
            1,
            int(_WINDOWED_MASS_SEGMENT_SOFT_DEADLINE_RECENT_WINDOW),
        )
        recent_completed = tuple(completed[-recent_window:])
        estimated_next_s = (
            max(recent_completed)
            * _WINDOWED_MASS_SEGMENT_SOFT_DEADLINE_SAFETY_MULTIPLIER
        )
        estimator = "recent_max_times_safety_multiplier"
        recent_window_count = min(len(completed), recent_window)
    remaining_values = [float(state["remaining_s"])]
    staged = state.get("staged_timeout")
    if isinstance(staged, Mapping):
        remaining_values.append(float(staged["global_remaining_s"]))
    effective_remaining_s = min(remaining_values)
    stage_enlargement_available = bool(
        isinstance(staged, Mapping)
        and staged.get("repair_loop_available") is True
        and float(staged.get("global_remaining_s", 0.0))
        > float(state["reserve_s"]) + float(estimated_next_s)
    )
    closeout_required = bool(
        state["within_closeout_window"]
        or effective_remaining_s <= float(state["reserve_s"]) + float(estimated_next_s)
    )
    if not closeout_required or stage_enlargement_available:
        return None
    payload: dict[str, Any] = {
        **state,
        "schema": "bayesfilter.windowed_mass_public_timeout_closeout.v1",
        "stage": str(stage),
        "effective_remaining_s": float(effective_remaining_s),
        "estimated_next_segment_s": float(estimated_next_s),
        "stage_enlargement_available": False,
        "completed_segment_elapsed_count": len(completed),
        "completed_segment_elapsed_estimator": estimator,
        "completed_segment_elapsed_recent_window": recent_window_count,
        "closeout_required_before_hmc_call": True,
        "closeout_required_before_next_segment": True,
        "stop_source": "bayesfilter_public_timeout_budget",
        "stop_reason": "public_timeout_budget_exhausted_before_next_segment",
        "supervision_counter_baseline": 0,
        "final_status": _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_STATUS,
        "diagnostic_role": _WINDOWED_MASS_PUBLIC_TIMEOUT_RESOURCE_ROLE,
        "hard_veto": None,
        "repair_trigger": _WINDOWED_MASS_PUBLIC_TIMEOUT_REPAIR_TRIGGER,
        "progress_only": True,
        "public_closeout_artifact_expected": True,
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
        "reports_sampler_superiority": False,
        "reports_default_readiness": False,
        "reports_external_client_scientific_claim": False,
        "reports_gpu_or_xla_readiness": False,
        "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
    }
    if attempt_index is not None:
        payload["attempt_index"] = int(attempt_index)
    return payload


def _phase4_latent_adapter_for_step_stage(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    windowed_stage: HMCWindowedMassStageResult,
    target_scope: str,
) -> _BootstrapFixedMassLatentValueScoreAdapter:
    mass_signature = _mass_artifact_signature(geometry.mass_artifact)
    if mass_signature != geometry.mass_artifact_signature:
        raise ValueError("geometry mass artifact signature mismatch")
    phase4_initial_mass = _phase4_adapted_mass_artifact(windowed_stage)
    if not isinstance(phase4_initial_mass, PrecomputedMassArtifact):
        raise TypeError("Phase 4 final mass artifact must be PrecomputedMassArtifact")
    if phase4_initial_mass.adapter_signature != windowed_stage.hmc_adapter_signature:
        raise ValueError("Phase 4 initial latent mass signature mismatch")
    hmc_adapter = _build_bootstrap_fixed_mass_adapter(
        adapter=adapter,
        mass_artifact=geometry.mass_artifact,
        mass_signature=mass_signature,
        target_scope=target_scope,
        nonclaims=FIXED_MASS_STEP_STAGE_NONCLAIMS,
    )
    if stable_adapter_signature(hmc_adapter) != windowed_stage.hmc_adapter_signature:
        raise ValueError("rebuilt Phase 4 HMC adapter signature mismatch")
    return hmc_adapter


def _phase4_adapted_mass_artifact(
    windowed_stage: HMCWindowedMassStageResult,
) -> PrecomputedMassArtifact:
    if windowed_stage.operational_mass_artifact is not None:
        return windowed_stage.operational_mass_artifact
    windowed_result = windowed_stage.windowed_mass_result
    if windowed_result is None or windowed_result.final_mass_artifact is None:
        raise ValueError("Phase 4 result does not carry final mass artifact")
    artifact = windowed_result.final_mass_artifact
    if not isinstance(artifact, PrecomputedMassArtifact):
        raise TypeError("Phase 4 final mass artifact must be PrecomputedMassArtifact")
    return artifact


def _phase7_verification_runtime_context(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    windowed_stage: HMCWindowedMassStageResult,
    target_scope: str,
) -> tuple[PrecomputedMassArtifact, str, Any, Any, str]:
    """Rebuild the shared Phase 4 and fixed-mass adapters for verification."""

    scope = str(target_scope)
    if not scope:
        raise ValueError("Phase 7 verification target scope must be non-empty")
    adapter_signature = stable_adapter_signature(adapter)
    if adapter_signature != geometry.adapter_signature:
        raise ValueError("Phase 7 verification adapter must match geometry")
    if adapter_signature != windowed_stage.adapter_signature:
        raise ValueError("Phase 7 verification adapter must match Phase 4")
    if geometry.artifact_hash != windowed_stage.geometry_artifact_hash:
        raise ValueError("Phase 7 verification geometry lineage mismatch")
    if geometry.target_dimension != windowed_stage.target_dimension:
        raise ValueError("Phase 7 verification target dimension mismatch")
    if not windowed_stage.passed:
        raise ValueError("Phase 7 verification requires passed Phase 4")
    adapted_mass = _phase4_adapted_mass_artifact(windowed_stage)
    mass_signature = _mass_artifact_signature(adapted_mass)
    if mass_signature != windowed_stage.adapted_mass_artifact_signature:
        raise ValueError("Phase 7 verification Phase 4 mass signature mismatch")
    if adapted_mass.dimension != geometry.target_dimension:
        raise ValueError("Phase 7 verification adapted mass dimension mismatch")
    phase4_adapter = _phase4_latent_adapter_for_step_stage(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed_stage,
        target_scope=scope,
    )
    phase4_signature = stable_adapter_signature(phase4_adapter)
    if phase4_signature != windowed_stage.hmc_adapter_signature:
        raise ValueError("Phase 7 verification Phase 4 adapter signature mismatch")
    adapted_mass.validate_for_adapter(
        phase4_adapter,
        expected_dim=geometry.target_dimension,
    )
    verification_adapter = _build_fixed_mass_hmc_adapter(
        adapter=phase4_adapter,
        mass_artifact=adapted_mass,
        mass_signature=mass_signature,
        target_scope=scope,
    )
    verification_signature = stable_adapter_signature(verification_adapter)
    return (
        adapted_mass,
        mass_signature,
        phase4_adapter,
        verification_adapter,
        verification_signature,
    )


def _phase7_verification_initial_state(
    *,
    windowed_stage: HMCWindowedMassStageResult,
    phase4_adapter: Any,
    verification_adapter: Any,
    verification_hmc_signature: str,
) -> tuple[Any, Mapping[str, Any]]:
    """Map the frozen canonical start bank through both active affine layers."""

    import tensorflow as tf

    operational = windowed_stage.operational_warmup_result
    if operational is None:
        return tf.zeros(windowed_stage.target_dimension, dtype=tf.float64).numpy(), {
            "source": "historical_compatibility_zero_template",
            "count": 4,
            "frozen_post_warmup_bank_consumed": False,
            "raw_values_exposed": False,
            "reports_operational_start_lineage": False,
        }
    policy_id = str(getattr(operational, "private_start_bank_policy_id", ""))
    if policy_id != PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID:
        stage_config = getattr(windowed_stage, "config", None)
        if getattr(stage_config, "engineering_probe_covariance_multiplier", None) is not None:
            raise ValueError("configured P4-E stage requires its explicit engineering probe bank")
        # The ordinary broad route uses the checked post-warmup bank produced
        # by operational warmup. Map that bank through both frozen affine
        # layers and verify its identity before any per-L epsilon ladder runs.
        canonical = _float64_tensor(operational.private_start_bank_theta)
        if canonical.shape != (4, windowed_stage.target_dimension):
            raise ValueError("operational verification start bank shape mismatch")
        if not _all_finite(canonical):
            raise ValueError("operational verification start bank must be finite")
        source_signature = str(
            getattr(operational, "private_start_bank_signature", "")
        )
        if not source_signature:
            raise ValueError("operational verification start bank signature is missing")
        phase4_latent = _float64_tensor(
            phase4_adapter.transform.position_to_latent(canonical)
        )
        verification_latent = _float64_tensor(
            verification_adapter.transform.position_to_latent(phase4_latent),
        )
        round_trip_phase4 = _float64_tensor(
            verification_adapter.latent_to_position(verification_latent)
        )
        round_trip_theta = _float64_tensor(
            phase4_adapter.latent_to_position(round_trip_phase4)
        )
        operational_latent = operational.final_kernel_state.transform.theta_to_latent(canonical)
        if not _all_close(
            round_trip_phase4, phase4_latent, rtol=1.0e-10, atol=1.0e-10
        ):
            raise ValueError("verification nested start-bank transform did not round trip")
        if not _all_close(
            round_trip_theta, canonical, rtol=1.0e-10, atol=1.0e-10
        ):
            raise ValueError("verification canonical start bank did not round trip")
        if not _all_close(
            verification_latent,
            operational_latent,
            rtol=1.0e-10,
            atol=1.0e-10,
        ):
            raise ValueError(
                "verification start bank does not match final warmup coordinates"
            )
        active_signature = private_start_bank_content_signature(
            verification_latent,
            operational.final_kernel_state.transform.signature,
        )
        return verification_latent.numpy(), {
            "source": "operational_post_warmup_start_bank_v2",
            "policy_id": policy_id,
            "source_signature": source_signature,
            "active_signature": active_signature,
            "target_scope": operational.target_scope,
            "final_transform_signature": operational.final_kernel_state.transform.signature,
            "phase4_adapter_signature": stable_adapter_signature(phase4_adapter),
            "verification_adapter_signature": verification_hmc_signature,
            "count": 4,
            "frozen_post_warmup_bank_consumed": True,
            "canonical_round_trip_passed": True,
            "final_coordinate_match_passed": True,
            "raw_values_exposed": False,
            "reports_operational_start_lineage": True,
            "evidence_role": "tuning_handoff",
            "promotion_role": "phase5_candidate_start_only",
            "reports_posterior_convergence": False,
        }
    engineering_qualification = operational.engineering_probe_bank_qualification
    if engineering_qualification is None or not engineering_qualification.passed:
        raise ValueError("operational Phase 7 P4-E qualification is missing or failed")
    active_target = getattr(phase4_adapter, "base_adapter", None)
    if active_target is None:
        raise ValueError("operational Phase 7 P4-E base target identity is missing")
    expected_target_signature = _phase7_engineering_probe_target_signature(
        active_target
    )
    if engineering_qualification.target_signature != expected_target_signature:
        raise ValueError("operational Phase 7 P4-E target identity mismatch")
    active_scope = str(getattr(phase4_adapter, "target_scope", ""))
    operational_scope = str(getattr(operational, "target_scope", ""))
    if not active_scope or not operational_scope or active_scope != operational_scope:
        raise ValueError("operational Phase 7 P4-E target scope mismatch")
    stage_config = getattr(windowed_stage, "config", None)
    multiplier = getattr(stage_config, "engineering_probe_covariance_multiplier", None)
    if multiplier is None:
        raise ValueError("operational Phase 7 P4-E configuration is missing")
    configured_scope = getattr(stage_config, "target_scope", None)
    if configured_scope is not None and str(configured_scope) != active_scope:
        raise ValueError("operational Phase 7 P4-E configured target scope mismatch")
    stage_seed = _derive_seed(
        _validate_seed(getattr(stage_config, "seed", None)),
        stage_index=0,
    )
    operational_seed = _validate_seed(getattr(operational, "seed_root", None))
    if stage_seed != operational_seed:
        raise ValueError("operational Phase 7 P4-E stage seed lineage mismatch")
    expected_probe_config = Phase7EngineeringProbeBankConfig(
        chain_count=4,
        covariance_multiplier=multiplier,
        root_seed=operational_seed,
    )
    if engineering_qualification.config_signature != expected_probe_config.config_signature:
        raise ValueError("operational Phase 7 P4-E configuration lineage mismatch")
    if (
        engineering_qualification.derived_seed_signature
        != expected_probe_config.derived_seed_signature
    ):
        raise ValueError("operational Phase 7 P4-E derived seed lineage mismatch")
    if (
        engineering_qualification.transform_signature
        != operational.final_kernel_state.transform.signature
    ):
        raise ValueError("operational Phase 7 P4-E transform lineage mismatch")
    canonical = _float64_tensor(operational.private_start_bank_theta)
    if canonical.shape != (4, windowed_stage.target_dimension):
        raise ValueError("operational verification start bank shape mismatch")
    source_signature = operational.private_start_bank_signature
    if not source_signature:
        raise ValueError("operational verification start bank signature is missing")
    phase4_latent = _float64_tensor(
        phase4_adapter.transform.position_to_latent(canonical)
    )
    verification_latent = _float64_tensor(
        verification_adapter.transform.position_to_latent(phase4_latent)
    )
    round_trip_phase4 = _float64_tensor(
        verification_adapter.latent_to_position(verification_latent)
    )
    round_trip_theta = _float64_tensor(
        phase4_adapter.latent_to_position(round_trip_phase4)
    )
    operational_latent = operational.final_kernel_state.transform.theta_to_latent(canonical)
    if not _all_close(round_trip_phase4, phase4_latent, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("verification nested start-bank transform did not round trip")
    if not _all_close(round_trip_theta, canonical, rtol=1.0e-10, atol=1.0e-10):
        raise ValueError("verification canonical start bank did not round trip")
    if not _all_close(
        verification_latent,
        operational_latent,
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError("verification start bank does not match final warmup coordinates")
    active_signature = private_start_bank_content_signature(
        verification_latent,
        operational.final_kernel_state.transform.signature,
    )
    return verification_latent.numpy(), {
        "source": "phase7_engineering_probe_bank_v1",
        "policy_id": PHASE7_ENGINEERING_PROBE_BANK_POLICY_ID,
        "source_signature": source_signature,
        "active_signature": active_signature,
        "qualification_content_signature": (
            engineering_qualification.content_signature
        ),
        "qualification_target_signature": engineering_qualification.target_signature,
        "qualification_config_signature": engineering_qualification.config_signature,
        "qualification_derived_seed_signature": (
            engineering_qualification.derived_seed_signature
        ),
        "target_scope": operational.target_scope,
        "final_transform_signature": (
            operational.final_kernel_state.transform.signature
        ),
        "phase4_adapter_signature": stable_adapter_signature(phase4_adapter),
        "verification_adapter_signature": verification_hmc_signature,
        "count": 4,
        "frozen_post_warmup_bank_consumed": True,
        "canonical_round_trip_passed": True,
        "final_coordinate_match_passed": True,
        "raw_values_exposed": False,
        "reports_operational_start_lineage": True,
        "evidence_role": "engineering_only",
        "promotion_role": "non_promoting",
        "reports_posterior_convergence": False,
    }


def build_operational_fixed_mass_hmc_adapter(
    *,
    adapter: Any,
    geometry: HMCGeometryInitializationResult,
    windowed_stage: HMCWindowedMassStageResult,
    target_scope: str,
) -> Mapping[str, Any]:
    """Return the exact frozen-mass adapter and post-warmup start bank.

    This is the public handoff boundary for callers that need to run a custom
    fixed-kernel qualification after BayesFilter geometry and windowed-mass
    preparation.  The nested Phase-4/final adapter signatures and the
    operational four-chain start bank are reconstructed by BayesFilter's
    lineage checks; callers must not rebuild either layer from a mass payload.
    No transitions are executed and no samples are retained by this helper.
    """

    if not isinstance(geometry, HMCGeometryInitializationResult):
        raise TypeError("geometry must be HMCGeometryInitializationResult")
    if not isinstance(windowed_stage, HMCWindowedMassStageResult):
        raise TypeError("windowed_stage must be HMCWindowedMassStageResult")
    scope = str(target_scope)
    if not scope:
        raise ValueError("target_scope must be non-empty")
    (
        adapted_mass,
        mass_signature,
        phase4_adapter,
        final_adapter,
        final_signature,
    ) = _phase7_verification_runtime_context(
        adapter=adapter,
        geometry=geometry,
        windowed_stage=windowed_stage,
        target_scope=scope,
    )
    starts, start_lineage = _phase7_verification_initial_state(
        windowed_stage=windowed_stage,
        phase4_adapter=phase4_adapter,
        verification_adapter=final_adapter,
        verification_hmc_signature=final_signature,
    )
    if not start_lineage.get("frozen_post_warmup_bank_consumed"):
        raise ValueError("operational fixed-mass handoff requires the post-warmup start bank")
    return {
        "adapted_mass_artifact": adapted_mass,
        "adapted_mass_artifact_signature": mass_signature,
        "phase4_adapter": phase4_adapter,
        "final_adapter": final_adapter,
        "final_adapter_signature": final_signature,
        "initial_position": starts,
        "start_lineage": start_lineage,
        "target_scope": scope,
        "hmc_or_tuning_invoked": False,
        "raw_samples_retained": False,
    }


def _windowed_mass_stage_internal_config(
    attempt_budget_policy: _HMCAttemptBudgetPolicy | None = None,
    *,
    mass_policy: str = "windowed_adaptive",
    metric_evidence_policy: str = "temporal_information",
    metric_probe_num_results: int = 1,
    preparation_max_restarts: int = 0,
) -> WindowedMassAdaptationConfig:
    if attempt_budget_policy is None:
        warmup_steps = 12
        initial_buffer = 2
        final_buffer = 2
        first_window_size = 3
    else:
        warmup_steps = int(attempt_budget_policy.phase4_warmup_steps)
        initial_buffer = max(2, min(warmup_steps // 10, warmup_steps // 4))
        final_buffer = max(2, min(warmup_steps // 10, warmup_steps // 4))
        if initial_buffer + final_buffer >= warmup_steps:
            initial_buffer = max(1, warmup_steps // 4)
            final_buffer = max(1, warmup_steps // 4)
        slow_steps = warmup_steps - initial_buffer - final_buffer
        first_window_size = max(2, min(max(2, slow_steps // 4), slow_steps))
    min_window_samples = 2
    if (attempt_budget_policy is not None and mass_policy == "windowed_adaptive"
            and metric_evidence_policy == "finite_window"):
        from bayesfilter.inference.hmc_warmup import metric_covariance_state_requirements

        dense_floor, diagonal_floor = metric_covariance_state_requirements(
            attempt_budget_policy.target_dimension)
        # Preserve the declared transition budget. A final remainder shorter
        # than this floor is merged by the common schedule builder.
        slow_steps = warmup_steps - initial_buffer - final_buffer
        count_floor = (dense_floor if slow_steps >= dense_floor else
                       diagonal_floor if slow_steps >= diagonal_floor else 2)
        min_window_samples = count_floor
        first_window_size = max(first_window_size, count_floor)
    return WindowedMassAdaptationConfig(
        warmup_steps=warmup_steps,
        initial_buffer=initial_buffer,
        final_buffer=final_buffer,
        first_window_size=first_window_size,
        min_window_samples=min_window_samples,
        mass_shrinkage=0.25,
        covariance_jitter=1.0e-6,
        eigenvalue_floor=1.0e-9,
        step_adaptation_rate=0.03,
        mass_policy=mass_policy,
        metric_evidence_policy=metric_evidence_policy,
        metric_probe_num_results=metric_probe_num_results,
        preparation_max_restarts=preparation_max_restarts,
    )


def _windowed_stage_initial_mass_artifact(
    *,
    adapter_signature: str,
    target_dimension: int,
    attempt_state: _HMCPhaseAttemptState | None = None,
) -> PrecomputedMassArtifact:
    import tensorflow as tf

    dimension = int(target_dimension)
    if dimension <= 0:
        raise ValueError("target_dimension must be positive")
    if attempt_state is not None and attempt_state.mass_artifact_payload is not None:
        artifact = PrecomputedMassArtifact.from_payload(
            attempt_state.mass_artifact_payload,
            expected_adapter_signature=str(adapter_signature),
            expected_dim=dimension,
        )
        if attempt_state.mass_artifact_signature is not None:
            observed = _mass_artifact_signature(artifact)
            if observed != attempt_state.mass_artifact_signature:
                raise ValueError("Phase 7 carried mass artifact signature mismatch")
        return artifact
    return PrecomputedMassArtifact.from_covariance(
        position=tf.zeros(dimension, dtype=tf.float64),
        covariance=tf.eye(dimension, dtype=tf.float64),
        adapter_signature=str(adapter_signature),
        position_role="latent_fixed_mass_origin",
        covariance_source="latent_identity_initial_mass",
        matrix_used_for_square_root="latent_identity",
        source="windowed_mass_stage_initial_latent_mass",
        jitter=0.0,
        regularization_report={
            "method": "latent_identity",
            "coordinate_system": "latent_fixed_mass",
            "source": "run_hmc_windowed_mass_stage",
        },
        nonclaims=WINDOWED_MASS_STAGE_NONCLAIMS,
    )


def _windowed_stage_draw_capture_policy(
    config: WindowedMassAdaptationConfig,
) -> Mapping[str, Any]:
    return {
        "route": "retained_fixed_kernel_samples",
        "num_results": config.warmup_steps,
        "warmup_steps": config.warmup_steps,
        "num_burnin_steps": _WINDOWED_STAGE_API_DISCARD_STEPS,
        "api_discarded_burnin_count": _WINDOWED_STAGE_API_DISCARD_STEPS,
        "api_discarded_burnin_counted_as_adaptation_input": False,
        "retained_samples_are_adaptation_inputs_only": True,
        "assumes_discarded_burnin_state_capture": False,
        "sample_space": "latent_fixed_mass",
        "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
    }


def _windowed_stage_diagnostic_run_config(
    config: HMCWindowedMassStageConfig,
    *,
    windowed_config: WindowedMassAdaptationConfig,
    selected_kernel: Mapping[str, Any],
    seed: tuple[int, int],
    target_scope: str,
) -> FullChainHMCConfig:
    return FullChainHMCConfig(
        num_results=windowed_config.warmup_steps,
        num_burnin_steps=_WINDOWED_STAGE_API_DISCARD_STEPS,
        step_size=float(selected_kernel["step_size"]),
        num_leapfrog_steps=int(selected_kernel["num_leapfrog_steps"]),
        seed=seed,
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy=config.target_status_trace_policy,
        target_scope=target_scope,
        chain_execution_mode=config.chain_execution_mode,
        capture_first_failure=config.chain_execution_mode == "tf_function" and not config.use_xla,
        failure_capture_role="sampling",
    )


def _windowed_stage_chunk_run_config(
    config: HMCWindowedMassStageConfig,
    *,
    diagnostic_config: FullChainHMCConfig,
    max_results: int,
    num_burnin_steps: int,
) -> FixedSizeHMCChunkConfig:
    return FixedSizeHMCChunkConfig(
        max_results=max_results,
        num_burnin_steps=num_burnin_steps,
        step_size=diagnostic_config.step_size,
        num_leapfrog_steps=diagnostic_config.num_leapfrog_steps,
        seed=diagnostic_config.seed,
        use_xla=config.use_xla,
        trace_policy="standard",
        target_status_trace_policy="none",
        target_scope=diagnostic_config.target_scope,
        chain_execution_mode=config.chain_execution_mode,
    )


def _windowed_stage_valid_rows(chunk: FixedSizeHMCChunkRunResult, key: str) -> Any:
    import tensorflow as tf

    mask = tf.convert_to_tensor(chunk.valid_mask)
    if mask.dtype != tf.bool:
        raise ValueError("windowed chunk valid mask must be boolean")
    return tf.boolean_mask(tf.convert_to_tensor(chunk.trace[key]), mask)


def _windowed_stage_acceptance_capture(
    value: Any,
    *,
    expected_steps: int,
) -> Mapping[str, Any]:
    """Return per-draw acceptance plus raw runtime decision counts.

    TFP emits one accept/reject decision per chain at each draw.  Phase 4's
    mass/step diagnostic consumes a rank-1 per-draw acceptance series, so
    batched-chain traces are averaged across chain axes while raw binary
    decisions remain available for provenance checks.
    """

    import tensorflow as tf

    raw = _float64_tensor(value)
    if raw.shape[:1] != (int(expected_steps),) or int(tf.size(raw).numpy()) == 0:
        return {
            "acceptance_trace": None,
            "raw_shape": tuple(int(dim) for dim in raw.shape),
            "decision_count": None,
            "accepted_decision_count": None,
            "binary_trace": False,
        }
    if raw.shape.rank == 1:
        acceptance = raw
    else:
        acceptance = tf.reduce_mean(tf.reshape(raw, (int(expected_steps), -1)), axis=1)
    finite = _all_finite(raw) and _all_finite(acceptance)
    binary = bool(tf.reduce_all((raw == 0.0) | (raw == 1.0)).numpy()) if finite else False
    decision_count = int(tf.size(raw).numpy())
    accepted_count = int(tf.reduce_sum(raw).numpy()) if binary else None
    return {
        "acceptance_trace": acceptance,
        "raw_shape": tuple(int(dim) for dim in raw.shape),
        "decision_count": decision_count,
        "accepted_decision_count": accepted_count,
        "binary_trace": binary,
    }


def _windowed_stage_per_draw_trace(
    value: Any,
    *,
    expected_steps: int,
) -> Any | None:
    """Reduce scalar or per-chain draw telemetry to a finite per-draw vector."""

    import tensorflow as tf

    raw = _float64_tensor(value)
    if raw.shape[:1] != (int(expected_steps),) or int(tf.size(raw).numpy()) == 0:
        return None
    if raw.shape.rank == 1:
        reduced = raw
    else:
        reduced = tf.reduce_mean(tf.reshape(raw, (int(expected_steps), -1)), axis=1)
    if not _all_finite(raw) or not _all_finite(reduced):
        return None
    return reduced


def _windowed_stage_segmented_capture_payload(
    *,
    config: HMCWindowedMassStageConfig,
    hmc_adapter: Any,
    initial_runner: Any,
    continuation_runner: Any,
    diagnostic_config: FullChainHMCConfig,
    windowed_config: WindowedMassAdaptationConfig,
    target_dimension: int,
    progress_callback: LoopProgressCallback | None,
    attempt_index: int | None,
    route_category: str,
    route_decision: HMCAlgorithmRouteDecision,
) -> Mapping[str, Any]:
    """Run windowed-mass diagnostic draws as small state-carrying HMC chunks."""

    import tensorflow as tf

    total_steps = int(windowed_config.warmup_steps)
    segment_size = max(1, min(_WINDOWED_MASS_SEGMENT_SIZE, total_steps))
    segment_count = _ceil_div(total_steps, segment_size)
    current_state = hmc_adapter.initial_position()
    sample_segments: list[Any] = []
    acceptance_segments: list[Any] = []
    log_accept_segments: list[Any] = []
    target_log_prob_segments: list[Any] = []
    finite_sample_count = 0
    nonfinite_sample_count = 0
    accepted_decision_count = 0
    acceptance_decision_count = 0
    runtime_supported_segment_count = 0
    segment_elapsed: list[float] = []
    run_start = time.perf_counter()

    for segment_index in range(segment_count):
        completed = int(sum(segment.shape[0] for segment in sample_segments))
        active = min(segment_size, total_steps - completed)
        timeout_closeout = _windowed_mass_next_segment_soft_deadline_preflight(
            config,
            stage="windowed_mass_segment_start",
            attempt_index=attempt_index,
            completed_segment_elapsed_s=segment_elapsed,
        )
        if timeout_closeout is not None:
            _emit_windowed_mass_progress(
                progress_callback,
                "windowed_mass_public_timeout_closeout",
                attempt_index=attempt_index,
                route_category=route_category,
                route_decision=route_decision,
                completed=True,
                elapsed_s=time.perf_counter() - run_start,
                timeout_closeout={
                    **dict(timeout_closeout),
                    "completed_segment_count": int(segment_index),
                    "planned_segment_count": int(segment_count),
                },
            )
            capture = dict(_windowed_stage_public_timeout_capture(timeout_closeout))
            closeout = {
                **dict(timeout_closeout),
                "completed_segment_count": int(segment_index),
                "planned_segment_count": int(segment_count),
            }
            metadata = dict(capture.get("runtime_metadata", {}))
            metadata.update(
                {
                    "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop",
                    "windowed_stage_segmented_chunk_runner": True,
                    "completed_segment_count": int(segment_index),
                    "planned_segment_count": int(segment_count),
                    "hmc_mechanics_exposed": False,
                    "public_timeout_closeout": closeout,
                }
            )
            capture["runtime_metadata"] = metadata
            capture["public_timeout_closeout"] = closeout
            return capture

        _emit_windowed_mass_progress(
            progress_callback,
            "windowed_mass_segment_start",
            attempt_index=attempt_index,
            route_category=route_category,
            route_decision=route_decision,
            started=True,
            elapsed_s=0.0,
            started_perf_counter_s=time.perf_counter(),
            segment_index=segment_index,
            segment_count=segment_count,
            segment_active_results=active,
        )
        segment_start = time.perf_counter()
        seed = (
            int(diagnostic_config.seed[0]),
            int(diagnostic_config.seed[1]) + 1009 * (segment_index + 1),
        )
        runner = initial_runner if segment_index == 0 else continuation_runner
        chunk = runner.run(
            active_results=active,
            current_state=current_state,
            seed=seed,
            step_size=diagnostic_config.step_size,
        )
        current_state = chunk.final_state
        elapsed = time.perf_counter() - segment_start
        segment_elapsed.append(elapsed)
        mask = tf.cast(tf.convert_to_tensor(_tensor_to_numpy(chunk.valid_mask)), tf.bool)
        samples = tf.boolean_mask(_float64_tensor(_tensor_to_numpy(chunk.samples)), mask)
        sample_segments.append(samples)
        acceptance_capture = _windowed_stage_acceptance_capture(
            _windowed_stage_valid_rows(chunk, "is_accepted"),
            expected_steps=active,
        )
        acceptance_rows = acceptance_capture["acceptance_trace"]
        if acceptance_rows is not None:
            acceptance_segments.append(_float64_tensor(acceptance_rows))
        segment_decision_count = _int_or_none(acceptance_capture["decision_count"])
        segment_accepted_count = _int_or_none(
            acceptance_capture["accepted_decision_count"]
        )
        if segment_decision_count is not None:
            acceptance_decision_count += int(segment_decision_count)
        if segment_accepted_count is not None:
            accepted_decision_count += int(segment_accepted_count)
        chunk_metadata = dict(chunk.metadata)
        if (
            chunk_metadata.get("fixed_size_chunk_runner") is True
            and chunk_metadata.get("runtime")
            == "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop"
        ):
            runtime_supported_segment_count += 1
        log_accept_rows = _windowed_stage_per_draw_trace(
            _windowed_stage_valid_rows(chunk, "log_accept_ratio"),
            expected_steps=active,
        )
        if log_accept_rows is not None:
            log_accept_segments.append(log_accept_rows)
        target_log_prob_rows = _windowed_stage_per_draw_trace(
            _windowed_stage_valid_rows(chunk, "target_log_prob"),
            expected_steps=active,
        )
        if target_log_prob_rows is not None:
            target_log_prob_segments.append(target_log_prob_rows)
        finite_rows = tf.reduce_all(tf.math.is_finite(samples), axis=-1)
        finite_sample_count += int(tf.reduce_sum(tf.cast(finite_rows, tf.int32)))
        nonfinite_sample_count += int(tf.reduce_sum(tf.cast(tf.logical_not(finite_rows), tf.int32)))
        _emit_windowed_mass_progress(
            progress_callback,
            "windowed_mass_segment_complete",
            attempt_index=attempt_index,
            route_category=route_category,
            route_decision=route_decision,
            completed=True,
            elapsed_s=elapsed,
            segment_index=segment_index,
            segment_count=segment_count,
            segment_active_results=active,
        )

    warmup_draws = (
        tf.concat(sample_segments, axis=0)
        if sample_segments
        else tf.zeros((0, int(target_dimension)), dtype=tf.float64)
    )
    acceptance = (
        tf.concat(acceptance_segments, axis=0)
        if acceptance_segments
        else tf.zeros((0,), dtype=tf.float64)
    )
    log_accept = (
        tf.concat(log_accept_segments, axis=0)
        if log_accept_segments
        else tf.zeros((0,), dtype=tf.float64)
    )
    target_log_prob = (
        tf.concat(target_log_prob_segments, axis=0)
        if target_log_prob_segments
        else tf.zeros((0,), dtype=tf.float64)
    )
    runtime_s = time.perf_counter() - run_start
    acceptance_rate = (
        None
        if acceptance_decision_count <= 0
        else float(accepted_decision_count) / float(acceptance_decision_count)
    )
    runtime_decision_supported = (
        runtime_supported_segment_count == segment_count
        and acceptance.shape == (total_steps,)
        and acceptance_decision_count >= total_steps
        and acceptance_decision_count % total_steps == 0
    )
    raw_diagnostics = {
        "valid_sample_count": int(warmup_draws.shape[0]),
        "finite_sample_count": int(finite_sample_count),
        "nonfinite_sample_count": int(nonfinite_sample_count),
        "accepted_decision_count": int(accepted_decision_count),
        "acceptance_decision_count": int(acceptance_decision_count),
        "acceptance_trace_decision_count": int(acceptance.shape[0]),
        "acceptance_raw_chain_count": None
        if acceptance_decision_count <= 0 or total_steps <= 0
        else int(acceptance_decision_count // total_steps),
        "acceptance_rate": acceptance_rate,
        "acceptance_decision_source": (
            "fixed_size_chunk_runner_trace_counts"
            if runtime_decision_supported
            else "unavailable"
        ),
        "runtime_supported_segment_count": int(runtime_supported_segment_count),
        "segment_count": int(segment_count),
        "completed_segment_count": int(segment_count),
        "hmc_mechanics_exposed": False,
        "reports_posterior_convergence": False,
    }
    payload = {
        "warmup_draws": warmup_draws,
        "acceptance_trace": acceptance,
        "log_accept_ratio": log_accept,
        "target_log_prob": target_log_prob,
        "runtime_s": runtime_s,
        "runtime_finite": bool(math.isfinite(runtime_s)),
        "samples_shape": tuple(int(dim) for dim in warmup_draws.shape),
        "acceptance_shape": tuple(int(dim) for dim in acceptance.shape),
        "log_accept_shape": tuple(int(dim) for dim in log_accept.shape),
        "target_log_prob_shape": tuple(int(dim) for dim in target_log_prob.shape),
        "expected_steps": total_steps,
        "target_dimension": int(target_dimension),
        "finite_sample_count": finite_sample_count,
        "nonfinite_sample_count": nonfinite_sample_count,
        "raw_diagnostics": raw_diagnostics,
        "runtime_metadata": {
            "runtime": "tfp.mcmc.HamiltonianMonteCarlo.one_step_tf_while_loop",
            "windowed_stage_segmented_chunk_runner": True,
            "uses_sample_chain": False,
            "segment_count": int(segment_count),
            "completed_segment_count": int(segment_count),
            "segment_elapsed_s": tuple(float(item) for item in segment_elapsed),
            "windowed_stage_segmented_execute_s": float(runtime_s),
            "hmc_mechanics_exposed": False,
            "timing_buckets": {
                "windowed_stage_segmented_execute_s": (
                    "explanatory_only_segmented_windowed_mass_execute"
                ),
                "segment_elapsed_s": (
                    "explanatory_only_segment_elapsed_without_hmc_mechanics"
                ),
            },
            "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
        },
        "runtime_evidence": "tfp_hmc_runtime",
        "fixture_or_synthetic": False,
        "acceptance_trace_key_present": True,
        "trace_summary": {
            "trace_keys": ("is_accepted", "log_accept_ratio", "target_log_prob"),
            "trace_unavailability": {},
        },
    }
    payload["acceptance_policy_filled_or_default"] = (
        _windowed_stage_acceptance_policy_filled_or_default(payload)
    )
    return payload


def _windowed_stage_capture_payload(
    run_result: FullChainHMCRunResult,
    *,
    expected_steps: int,
    target_dimension: int,
) -> Mapping[str, Any]:
    diagnostics = dict(run_result.diagnostics)
    trace = dict(run_result.trace)
    metadata = dict(run_result.metadata)
    samples = _float64_tensor(_tensor_to_numpy(run_result.samples))
    raw_acceptance = _trace_array_or_none(trace, "is_accepted")
    acceptance_capture = (
        None
        if raw_acceptance is None
        else _windowed_stage_acceptance_capture(
            raw_acceptance,
            expected_steps=int(expected_steps),
        )
    )
    acceptance = (
        None
        if acceptance_capture is None
        else acceptance_capture["acceptance_trace"]
    )
    raw_log_accept = _trace_array_or_none(trace, "log_accept_ratio")
    log_accept = (
        None
        if raw_log_accept is None
        else _windowed_stage_per_draw_trace(
            raw_log_accept,
            expected_steps=int(expected_steps),
        )
    )
    raw_target_log_prob = _trace_array_or_none(trace, "target_log_prob")
    target_log_prob = (
        None
        if raw_target_log_prob is None
        else _windowed_stage_per_draw_trace(
            raw_target_log_prob,
            expected_steps=int(expected_steps),
        )
    )
    runtime_s = _runtime_seconds_or_none(metadata)
    finite_acceptance = _valid_trace_vector(
        acceptance,
        expected_steps,
        bounds=(0.0, 1.0),
    )
    runtime_evidence = _windowed_stage_runtime_evidence(metadata)
    fixture_or_synthetic = _metadata_marks_fixture_or_synthetic(metadata)
    payload = {
        "warmup_draws": samples,
        "acceptance_trace": acceptance,
        "log_accept_ratio": log_accept,
        "target_log_prob": target_log_prob,
        "runtime_s": runtime_s,
        "runtime_finite": runtime_s is not None and bool(math.isfinite(runtime_s)),
        "samples_shape": tuple(int(dim) for dim in samples.shape),
        "acceptance_shape": None
        if acceptance is None
        else tuple(int(dim) for dim in acceptance.shape),
        "log_accept_shape": None
        if log_accept is None
        else tuple(int(dim) for dim in log_accept.shape),
        "target_log_prob_shape": None
        if target_log_prob is None
        else tuple(int(dim) for dim in target_log_prob.shape),
        "expected_steps": int(expected_steps),
        "target_dimension": int(target_dimension),
        "finite_sample_count": _int_or_none(diagnostics.get("finite_sample_count")),
        "nonfinite_sample_count": _int_or_none(
            diagnostics.get("nonfinite_sample_count")
        ),
        "raw_diagnostics": _json_ready(
            {
                **diagnostics,
                "acceptance_decision_source": (
                    "sample_chain_trace_counts"
                    if acceptance_capture is not None
                    and acceptance_capture.get("binary_trace") is True
                    else "unavailable"
                ),
                "accepted_decision_count": None
                if acceptance_capture is None
                else acceptance_capture.get("accepted_decision_count"),
                "acceptance_decision_count": None
                if acceptance_capture is None
                else acceptance_capture.get("decision_count"),
                "acceptance_trace_decision_count": None
                if acceptance is None
                else int(acceptance.shape[0]),
                "acceptance_raw_chain_count": None
                if acceptance_capture is None
                or acceptance_capture.get("decision_count") is None
                else int(
                    int(acceptance_capture["decision_count"]) // int(expected_steps)
                ),
                "raw_acceptance_shape": None
                if acceptance_capture is None
                else acceptance_capture.get("raw_shape"),
            }
        ),
        "runtime_metadata": _json_ready(metadata),
        "runtime_evidence": runtime_evidence,
        "fixture_or_synthetic": fixture_or_synthetic,
        "acceptance_trace_key_present": "is_accepted" in trace,
        "trace_summary": {
            "trace_keys": tuple(sorted(trace.keys())),
            "trace_unavailability": metadata.get("trace_unavailability"),
        },
    }
    payload["acceptance_policy_filled_or_default"] = (
        _windowed_stage_acceptance_policy_filled_or_default(payload)
        if finite_acceptance
        else True
    )
    return payload


def _with_windowed_stage_timing_metadata(
    capture: Mapping[str, Any],
    *,
    runner_build_s: float,
    runner_execute_s: float,
    capture_s: float,
    route_category: str,
) -> Mapping[str, Any]:
    payload = dict(capture)
    metadata = dict(payload.get("runtime_metadata", {}))
    timing_buckets = dict(metadata.get("timing_buckets", {}))
    timing_buckets.update(
        {
            "windowed_stage_runner_build_s": (
                "explanatory_only_windowed_stage_runner_build"
            ),
            "windowed_stage_runner_execute_s": (
                "explanatory_only_windowed_stage_sample_chain_call"
            ),
            "windowed_stage_capture_s": (
                "explanatory_only_windowed_stage_public_safe_capture"
            ),
        }
    )
    metadata.update(
        {
            "windowed_stage_route_category": str(route_category),
            "windowed_stage_runner_build_s": float(runner_build_s),
            "windowed_stage_runner_execute_s": float(runner_execute_s),
            "windowed_stage_capture_s": float(capture_s),
            "windowed_stage_timing_scope": (
                "public_safe_runner_build_execute_and_capture_timing"
            ),
            "timing_buckets": timing_buckets,
        }
    )
    payload["runtime_metadata"] = metadata
    return payload


def _windowed_stage_public_timeout_capture(
    timeout_closeout: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "error_type": None,
        "error_message": None,
        "warmup_draws": None,
        "acceptance_trace": None,
        "log_accept_ratio": None,
        "target_log_prob": None,
        "runtime_s": None,
        "runtime_finite": False,
        "samples_shape": None,
        "acceptance_shape": None,
        "log_accept_shape": None,
        "target_log_prob_shape": None,
        "expected_steps": None,
        "target_dimension": None,
        "finite_sample_count": None,
        "nonfinite_sample_count": None,
        "raw_diagnostics": {},
        "runtime_metadata": {
            "public_timeout_closeout": dict(timeout_closeout),
            "timing_scope": "windowed_mass_public_timeout_closeout_before_hmc_call",
        },
        "runtime_evidence": "public_timeout_closeout",
        "fixture_or_synthetic": False,
        "acceptance_trace_key_present": False,
        "acceptance_policy_filled_or_default": True,
        "public_timeout_closeout": dict(timeout_closeout),
        "trace_summary": {"trace_keys": (), "trace_unavailability": {}},
    }


def _windowed_stage_error_capture(exc: Exception) -> Mapping[str, Any]:
    start_bank_qualification = start_bank_qualification_payload_from_exception(exc)
    engineering_probe_qualification = (
        engineering_probe_bank_qualification_payload_from_exception(exc)
    )
    raw_diagnostics = {}
    if hasattr(exc, "recovery_summary"):
        raw_diagnostics["preparation_recovery"] = exc.recovery_summary
    if start_bank_qualification is not None:
        raw_diagnostics["start_bank_qualification"] = start_bank_qualification
    if engineering_probe_qualification is not None:
        raw_diagnostics["engineering_probe_bank_qualification"] = (
            engineering_probe_qualification
        )
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "warmup_draws": None,
        "acceptance_trace": None,
        "log_accept_ratio": None,
        "target_log_prob": None,
        "runtime_s": None,
        "runtime_finite": False,
        "samples_shape": None,
        "acceptance_shape": None,
        "log_accept_shape": None,
        "target_log_prob_shape": None,
        "expected_steps": None,
        "target_dimension": None,
        "finite_sample_count": None,
        "nonfinite_sample_count": None,
        "raw_diagnostics": raw_diagnostics,
        "runtime_metadata": {},
        "runtime_evidence": "error",
        "fixture_or_synthetic": False,
        "acceptance_trace_key_present": False,
        "acceptance_policy_filled_or_default": True,
        "trace_summary": {"trace_keys": (), "trace_unavailability": {}},
    }


def _p4_boundary_scalar_capture(
    boundary: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build a redacted P4 capture with no generic exception channel."""

    return {
        "error_type": None,
        "error_message": None,
        "warmup_draws": None,
        "acceptance_trace": None,
        "log_accept_ratio": None,
        "target_log_prob": None,
        "runtime_s": None,
        "runtime_finite": False,
        "samples_shape": None,
        "acceptance_shape": None,
        "log_accept_shape": None,
        "target_log_prob_shape": None,
        "expected_steps": None,
        "target_dimension": None,
        "finite_sample_count": None,
        "nonfinite_sample_count": None,
        "raw_diagnostics": {"engineering_probe_boundary": dict(boundary)},
        "runtime_metadata": {},
        "runtime_evidence": "p4_boundary",
        "fixture_or_synthetic": False,
        "acceptance_trace_key_present": False,
        "acceptance_policy_filled_or_default": True,
        "trace_summary": {"trace_keys": (), "trace_unavailability": {}},
    }


def _p4_unavailable_invalidity_payload(
    *,
    failure_code: str,
    p4_boundary_stage: str,
    p4_builder_entered: bool,
    p4_seed_consumed: bool,
    p4_rng_batch_invoked: bool,
) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.hmc_p4e_unavailable_invalidity.v1",
        "outcome": "shared_implementation_invalid",
        "failure_code": failure_code,
        "stage": "windowed_mass_p4_carrier_validation",
        "source_coverage_artifact_sha256": None,
        "seed_registry_evidence_kind": "unavailable_invalid_carrier",
        "seed_registry_schema": None,
        "seed_registry_evidence_signature": None,
        "registered_entry_count": None,
        "consumed_entry_count": None,
        "p4_boundary_stage": p4_boundary_stage,
        "p4_builder_entered": p4_builder_entered,
        "p4_seed_consumed": p4_seed_consumed,
        "p4_rng_batch_invoked": p4_rng_batch_invoked,
        "final_lineage_available": False,
    }


def _p4_boundary_capture_from_exception(
    exc: Exception,
    *,
    registry: G2PreboundarySeedUseRegistry,
    action_tracker: _G2P4BoundaryActionTracker | None = None,
) -> tuple[Mapping[str, Any], str, str, Mapping[str, Any] | None]:
    """Classify concrete carriers before any generic error capture is allowed."""

    terminal = engineering_probe_bank_qualification_payload_from_exception(exc)
    early = g2_preboundary_shared_invalidity_payload_from_exception(exc)
    private = registry.validated_private_evidence(
        g2_seed_private_evidence_from_exception(exc)
    )

    tracker_payload = (
        None if action_tracker is None else action_tracker.scalar_payload()
    )
    boundary = terminal if terminal is not None else early
    if boundary is not None and private is not None:
        private_signature = private.get(
            "seed_use_registry_signature",
            private.get("seed_use_registry_snapshot_signature"),
        )
        private_entries = private.get("entries")
        private_count = (
            None
            if not isinstance(private_entries, (tuple, list))
            else len(private_entries)
        )
        boundary_count = boundary.get(
            "seed_preboundary_consumed_count",
            boundary.get("registered_entry_count"),
        )
        if (
            boundary.get("seed_registry_evidence_signature") != private_signature
            or boundary.get("seed_registry_schema") != private.get("schema")
            or boundary.get("source_coverage_artifact_sha256")
            != registry.source_coverage_artifact_sha256
            or private.get("source_coverage_artifact_sha256")
            != registry.source_coverage_artifact_sha256
            or boundary_count != private.get("preboundary_consumed_seed_count")
            or (
                terminal is not None
                and (
                    boundary.get("p4_distinct_from_preboundary_seeds")
                    != private.get("p4_distinct_from_preboundary_seeds")
                    or boundary.get("p4_seed_consumed")
                    != (
                        private.get("schema")
                        == "bayesfilter.hmc_g2_preboundary_seed_use_registry.v1"
                    )
                    or boundary.get("post_boundary_registry_call_count")
                    != private.get("post_boundary_registry_call_count")
                )
            )
            or (
                early is not None
                and (
                    boundary.get("failure_code") != private.get("failure_code")
                    or boundary.get("registered_entry_count") != private_count
                    or boundary.get("consumed_entry_count") != private_count
                )
            )
        ):
            boundary = None
    elif boundary is not None:
        boundary = None

    if boundary is not None and tracker_payload is not None:
        action_names = (
            "p4_boundary_stage",
            "p4_builder_entered",
            "p4_seed_consumed",
            "p4_rng_batch_invoked",
        )
        if any(
            boundary.get(name) != tracker_payload[name]
            for name in action_names
        ):
            boundary = None
        elif boundary.get("final_lineage_available") is True:
            callback_names = (
                "target_health_callback_invocation_count",
                "target_health_callback_batch_row_count",
                "target_health_callback_batch_dimension",
            )
            if any(
                boundary.get(name) != tracker_payload[name]
                for name in callback_names
            ):
                boundary = None
        elif any(
            tracker_payload[name] not in {0, None}
            for name in (
                "target_health_callback_invocation_count",
                "target_health_callback_batch_row_count",
                "target_health_callback_batch_dimension",
            )
        ):
            boundary = None

    if boundary is not None:
        outcome = str(boundary.get("outcome"))
        failure_code = str(boundary.get("failure_code"))
        if outcome in {
            "candidate_generation_invalid",
            "candidate_policy_instance_invalid",
        }:
            return (
                _p4_boundary_scalar_capture(boundary),
                "candidate_rejected",
                failure_code,
                private,
            )
        if outcome == "shared_implementation_invalid":
            return (
                _p4_boundary_scalar_capture(boundary),
                "shared_implementation_invalid",
                failure_code,
                private,
            )

    try:
        carrier_attribute_present = bool(
            hasattr(exc, _PHASE7_ENGINEERING_PROBE_DIAGNOSTIC_ATTRIBUTE)
            or hasattr(exc, _G2_PREBOUNDARY_SHARED_INVALIDITY_ATTRIBUTE)
        )
    except Exception:  # noqa: BLE001 - hostile carrier access remains redacted.
        carrier_attribute_present = True
    failure_code = (
        "qualification_carrier_invalid"
        if carrier_attribute_present
        else "unexpected_builder_exception"
    )
    if tracker_payload is None:
        p4_consumed = bool(registry.p4_seed_consumed)
        action_payload = {
            "p4_boundary_stage": (
                "seed_consumed_pre_rng" if p4_consumed else "not_entered"
            ),
            "p4_builder_entered": p4_consumed,
            "p4_seed_consumed": p4_consumed,
            "p4_rng_batch_invoked": False,
        }
    else:
        action_payload = tracker_payload
    boundary = _p4_unavailable_invalidity_payload(
        failure_code=failure_code,
        p4_boundary_stage=str(action_payload["p4_boundary_stage"]),
        p4_builder_entered=bool(action_payload["p4_builder_entered"]),
        p4_seed_consumed=bool(action_payload["p4_seed_consumed"]),
        p4_rng_batch_invoked=bool(action_payload["p4_rng_batch_invoked"]),
    )
    return (
        _p4_boundary_scalar_capture(boundary),
        "shared_implementation_invalid",
        failure_code,
        None,
    )


def _classify_windowed_stage_capture(
    capture: Mapping[str, Any],
    *,
    run_error: Exception | None,
) -> tuple[str, ...]:
    hard_vetoes: list[str] = []
    if capture.get("public_timeout_closeout") is not None:
        return ()
    if run_error is not None:
        hard_vetoes.append("windowed_stage_hmc_error")
    expected_steps = capture.get("expected_steps")
    target_dimension = capture.get("target_dimension")
    draws = capture.get("warmup_draws")
    acceptance = capture.get("acceptance_trace")
    log_accept = capture.get("log_accept_ratio")
    target_log_prob = capture.get("target_log_prob")
    if capture.get("runtime_finite") is not True:
        hard_vetoes.append("windowed_stage_runtime_missing_or_nonfinite")
    if capture.get("runtime_evidence") != "tfp_hmc_runtime":
        hard_vetoes.append("windowed_stage_fixture_or_nonruntime_telemetry")
    if capture.get("fixture_or_synthetic") is True:
        hard_vetoes.append("windowed_stage_fixture_or_nonruntime_telemetry")
    if not _valid_draw_matrix(draws, expected_steps, target_dimension):
        hard_vetoes.append("windowed_stage_warmup_draws_invalid")
    if (
        not _valid_trace_vector(acceptance, expected_steps, bounds=(0.0, 1.0))
        or capture.get("acceptance_policy_filled_or_default") is True
    ):
        hard_vetoes.append("windowed_stage_acceptance_telemetry_invalid_or_default")
    if not _valid_trace_vector(log_accept, expected_steps):
        hard_vetoes.append("windowed_stage_log_accept_invalid")
    if not _valid_trace_vector(target_log_prob, expected_steps):
        hard_vetoes.append("windowed_stage_target_log_prob_invalid")
    return tuple(dict.fromkeys(hard_vetoes))


def _valid_draw_matrix(value: Any, expected_steps: Any, target_dimension: Any) -> bool:
    if value is None or expected_steps is None or target_dimension is None:
        return False
    array = _float64_tensor(value)
    if int(expected_steps) <= 0 or int(target_dimension) <= 0 or array.shape != (int(expected_steps), int(target_dimension)):
        return False
    return _all_finite(array)


def _valid_trace_vector(
    value: Any,
    expected_steps: Any,
    *,
    bounds: tuple[float, float] | None = None,
) -> bool:
    if value is None or expected_steps is None:
        return False
    import tensorflow as tf

    array = _float64_tensor(value)
    if int(expected_steps) <= 0 or array.shape != (int(expected_steps),):
        return False
    if not _all_finite(array):
        return False
    if bounds is not None:
        lower, upper = bounds
        if bool(tf.reduce_any((array < lower) | (array > upper)).numpy()):
            return False
    return True


def _acceptance_trace_is_default_like(value: Any) -> bool:
    import tensorflow as tf

    if value is None:
        return True
    array = _float64_tensor(value)
    if int(tf.size(array).numpy()) == 0:
        return True
    return _all_close(array, tf.reshape(array, [-1])[0], rtol=1.0e-5, atol=1.0e-8)


def _windowed_stage_acceptance_has_runtime_decision_support(
    capture: Mapping[str, Any],
) -> bool:
    import tensorflow as tf

    expected_steps = capture.get("expected_steps")
    acceptance = capture.get("acceptance_trace")
    if not _valid_trace_vector(acceptance, expected_steps, bounds=(0.0, 1.0)):
        return False
    if capture.get("runtime_evidence") != "tfp_hmc_runtime":
        return False
    if capture.get("fixture_or_synthetic") is True:
        return False
    raw = capture.get("raw_diagnostics")
    if not isinstance(raw, Mapping):
        return False
    decision_source = raw.get("acceptance_decision_source")
    if decision_source not in {
        "fixed_size_chunk_runner_trace_counts",
        "operational_window_binary_trace",
    }:
        if decision_source != "sample_chain_trace_counts":
            return False
        metadata = capture.get("runtime_metadata")
        if not isinstance(metadata, Mapping):
            return False
        if metadata.get("runtime") != "tfp.mcmc.sample_chain":
            return False
        invocation_count = _int_or_none(metadata.get("sample_chain_invocation_count"))
        if invocation_count is None or invocation_count <= 0:
            return False
        if metadata.get("program_signature") is None:
            return False
    if decision_source == "operational_window_binary_trace":
        metadata = capture.get("runtime_metadata")
        if (
            not isinstance(metadata, Mapping)
            or metadata.get("runtime")
            != "tfp.mcmc.operational_interleaved_windowed_warmup"
            or metadata.get("operational_warmup") is not True
        ):
            return False
    decision_count = _int_or_none(raw.get("acceptance_decision_count"))
    accepted_count = _int_or_none(raw.get("accepted_decision_count"))
    trace_decision_count = _int_or_none(raw.get("acceptance_trace_decision_count"))
    if decision_count is None or accepted_count is None or expected_steps is None:
        return False
    if trace_decision_count is None or trace_decision_count != int(expected_steps):
        return False
    if decision_count < int(expected_steps):
        return False
    if decision_count % int(expected_steps) != 0:
        return False
    if accepted_count < 0 or accepted_count > decision_count:
        return False
    array = _float64_tensor(acceptance)
    raw_shape = raw.get("raw_acceptance_shape")
    if raw_shape is None:
        chain_count = _int_or_none(raw.get("acceptance_raw_chain_count"))
        if chain_count is None:
            chain_count = max(1, decision_count // int(expected_steps))
    else:
        try:
            normalized_shape = tuple(int(item) for item in raw_shape)
        except (TypeError, ValueError):
            return False
        if not normalized_shape or normalized_shape[0] != int(expected_steps):
            return False
        chain_count = math.prod(normalized_shape[1:])
    if chain_count <= 0 or decision_count != int(expected_steps) * int(chain_count):
        return False
    return _all_close(
        tf.reduce_sum(array) * float(chain_count), float(accepted_count),
        rtol=1.0e-5, atol=1.0e-8,
    )


def _windowed_stage_acceptance_policy_filled_or_default(
    capture: Mapping[str, Any],
) -> bool:
    expected_steps = capture.get("expected_steps")
    acceptance = capture.get("acceptance_trace")
    if not _valid_trace_vector(acceptance, expected_steps, bounds=(0.0, 1.0)):
        return True
    if not _acceptance_trace_is_default_like(acceptance):
        return False
    return not _windowed_stage_acceptance_has_runtime_decision_support(capture)


def _windowed_stage_runtime_evidence(metadata: Mapping[str, Any]) -> str:
    if _metadata_marks_fixture_or_synthetic(metadata):
        return "fixture_or_synthetic"
    if metadata.get("runtime") != "tfp.mcmc.sample_chain":
        return "missing_or_unknown_runtime"
    invocation_count = _int_or_none(metadata.get("sample_chain_invocation_count"))
    if invocation_count is None or invocation_count <= 0:
        return "missing_or_unknown_runtime"
    return "tfp_hmc_runtime"


def _metadata_marks_fixture_or_synthetic(metadata: Mapping[str, Any]) -> bool:
    for key in ("fixture_or_synthetic", "test_fixture", "synthetic"):
        if metadata.get(key) is True:
            return True
    nonclaims = metadata.get("nonclaims", ())
    if isinstance(nonclaims, str):
        candidates = (nonclaims,)
    else:
        try:
            candidates = tuple(nonclaims)
        except TypeError:
            candidates = ()
    markers = ("fake", "fixture", "synthetic", "test-only", "unit test")
    return any(
        any(marker in str(item).lower() for marker in markers)
        for item in candidates
    )


def _windowed_stage_diagnostics(
    capture: Mapping[str, Any],
    *,
    windowed_result: WindowedMassAdaptationResult | None,
    hard_vetoes: tuple[str, ...],
    mass_window_seed_kernel: Mapping[str, Any],
    bootstrap_kernel: Mapping[str, Any],
) -> Mapping[str, Any]:
    seed_step = float(mass_window_seed_kernel["step_size"])
    seed_l = int(mass_window_seed_kernel["num_leapfrog_steps"])
    bootstrap_step = float(bootstrap_kernel["step_size"])
    bootstrap_l = int(bootstrap_kernel["num_leapfrog_steps"])
    return {
        "passed": not hard_vetoes,
        "hard_vetoes": hard_vetoes,
        "runtime_s": capture.get("runtime_s"),
        "runtime_finite": capture.get("runtime_finite"),
        "samples_shape": capture.get("samples_shape"),
        "acceptance_shape": capture.get("acceptance_shape"),
        "log_accept_shape": capture.get("log_accept_shape"),
        "target_log_prob_shape": capture.get("target_log_prob_shape"),
        "finite_sample_count": capture.get("finite_sample_count"),
        "nonfinite_sample_count": capture.get("nonfinite_sample_count"),
        "windowed_mass_passed": None
        if windowed_result is None
        else bool(windowed_result.passed),
        "adapted_mass_artifact_signature": None
        if windowed_result is None
        else windowed_result.final_mass_artifact_signature,
        "candidate_step_size": None
        if windowed_result is None
        else windowed_result.final_step_size,
        "mass_window_seed_kernel": {
            "step_size": seed_step,
            "num_leapfrog_steps": seed_l,
            "seed_kernel_source": mass_window_seed_kernel.get(
                "private_kernel_source",
                "bootstrap_or_geometry_handoff",
            ),
            "uses_private_retry_pair": bool(
                mass_window_seed_kernel.get(
                    "bootstrap_kernel_is_lineage_not_active_mass_window_seed",
                    False,
                )
            ),
            "bootstrap_step_size": bootstrap_step,
            "bootstrap_num_leapfrog_steps": bootstrap_l,
            "bootstrap_kernel_is_lineage_not_active_mass_window_seed": bool(
                mass_window_seed_kernel.get(
                    "bootstrap_kernel_is_lineage_not_active_mass_window_seed",
                    False,
                )
            ),
            "hmc_mechanics_publicized": False,
            "reports_posterior_convergence": False,
            "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
        },
        "reports_posterior_convergence": False,
        "raw_diagnostics": capture.get("raw_diagnostics", {}),
        "runtime_metadata": capture.get("runtime_metadata", {}),
        "public_timeout_closeout": capture.get("public_timeout_closeout"),
        "trace_summary": capture.get("trace_summary", {}),
        "windowed_mass_error_type": capture.get("windowed_mass_error_type"),
        "windowed_mass_error_message": capture.get("windowed_mass_error_message"),
        "hmc_error_type": capture.get("error_type"),
        "hmc_error_message": capture.get("error_message"),
        "nonclaims": WINDOWED_MASS_STAGE_NONCLAIMS,
    }


def _warmup_draw_provenance(
    capture: Mapping[str, Any],
    draw_capture_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    if capture.get("runtime_evidence") == "p4_boundary":
        return {
            "source": "none_p4_boundary",
            "sample_space": None,
            "samples_shape": None,
            "expected_steps": None,
            "target_dimension": None,
            "draw_capture_policy_hash": stable_config_hash(draw_capture_policy),
            "fixture_or_synthetic": False,
            "runtime_evidence": "p4_boundary",
            "adaptation_input_only": False,
            "posterior_samples": False,
        }
    return {
        "source": "run_full_chain_tfp_hmc_retained_samples",
        "sample_space": "latent_fixed_mass",
        "samples_shape": capture.get("samples_shape"),
        "expected_steps": capture.get("expected_steps"),
        "target_dimension": capture.get("target_dimension"),
        "draw_capture_policy_hash": stable_config_hash(draw_capture_policy),
        "fixture_or_synthetic": bool(capture.get("fixture_or_synthetic")),
        "runtime_evidence": capture.get("runtime_evidence"),
        "adaptation_input_only": True,
        "posterior_samples": False,
    }


def _acceptance_telemetry_provenance(
    capture: Mapping[str, Any],
) -> Mapping[str, Any]:
    if capture.get("runtime_evidence") == "p4_boundary":
        return {
            "source": "none_p4_boundary",
            "shape": None,
            "expected_steps": None,
            "trace_key_present": False,
            "fixture_or_synthetic": False,
            "policy_filled_or_default": True,
            "constant_trace": False,
            "runtime_decision_count_supported": False,
            "accepted_decision_count": None,
            "acceptance_decision_count": None,
            "runtime_evidence": "p4_boundary",
            "finite_and_aligned": False,
        }
    acceptance = capture.get("acceptance_trace")
    return {
        "source": "run_full_chain_tfp_hmc_trace.is_accepted",
        "shape": capture.get("acceptance_shape"),
        "expected_steps": capture.get("expected_steps"),
        "trace_key_present": bool(capture.get("acceptance_trace_key_present")),
        "fixture_or_synthetic": bool(capture.get("fixture_or_synthetic")),
        "policy_filled_or_default": bool(
            capture.get("acceptance_policy_filled_or_default")
        ),
        "constant_trace": _acceptance_trace_is_default_like(acceptance),
        "runtime_decision_count_supported": (
            _windowed_stage_acceptance_has_runtime_decision_support(capture)
        ),
        "accepted_decision_count": _int_or_none(
            capture.get("raw_diagnostics", {}).get("accepted_decision_count")
            if isinstance(capture.get("raw_diagnostics"), Mapping)
            else None
        ),
        "acceptance_decision_count": _int_or_none(
            capture.get("raw_diagnostics", {}).get("acceptance_decision_count")
            if isinstance(capture.get("raw_diagnostics"), Mapping)
            else None
        ),
        "runtime_evidence": capture.get("runtime_evidence"),
        "finite_and_aligned": _valid_trace_vector(
            acceptance,
            capture.get("expected_steps"),
            bounds=(0.0, 1.0),
        ),
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    numerator = int(numerator)
    denominator = int(denominator)
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def _tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _trace_array_or_none(trace: Mapping[str, Any], key: str) -> Any | None:
    import tensorflow as tf

    if key not in trace:
        return None
    return tf.convert_to_tensor(_tensor_to_numpy(trace[key]))


__all__ = [
    "HMCStagedTimeoutPolicy",
    "HMCWindowedMassStageConfig",
    "HMCWindowedMassStageResult",
    "WINDOWED_MASS_STAGE_NONCLAIMS",
    "WindowedMassAdaptationConfig",
    "WindowedMassAdaptationResult",
    "build_operational_fixed_mass_hmc_adapter",
    "build_windowed_warmup_schedule",
    "run_hmc_windowed_mass_stage",
    "run_windowed_mass_adaptation_diagnostic",
    "validate_windowed_shrinkage_target",
    "welford_covariance",
]
