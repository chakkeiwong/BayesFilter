"""HMC preparation presets, translation and geometry-scaled budget policy.

These definitions preserve the serialized public and historical policies. They
configure preparation; candidate-set authority belongs to the shared controller.
Historical loop configuration remains readable without importing its executor.
"""
from __future__ import annotations

from bayesfilter.inference.hmc_preparation_recovery import validate_preparation_max_restarts

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping
from bayesfilter.inference.hmc_preparation_common import (
    _validate_band,
    _validate_seed,
    _validate_step_repair_multiplier,
)
from bayesfilter.inference.hmc_bootstrap import HMCBootstrapScreenConfig
from bayesfilter.inference.hmc_geometry import (
    HMCGeometryInitializationConfig,
    HMCGeometryInitializationResult,
    _GEOMETRY_MAX_LEAPFROG,
    _derive_seed,
    _validate_max_leapfrog_steps,
)
from bayesfilter.hmc_route_contract import (
    LEGACY_JOINT_L_EPSILON_ALGORITHM_ID,
    LEGACY_OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID,
    ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID,
    windowed_algorithm_for_selection_algorithm,
)
from bayesfilter.hmc_ordinary_selection_policy import (
    ORDINARY_BROAD_FIXED_METRIC_POLICY_ID,
    ORDINARY_BROAD_MAX_LEAPFROG_STEPS,
    ORDINARY_BROAD_PRIMARY_L_GRID,
)
from bayesfilter.hmc_budget_contract import HMCOperationalStatisticalWorkPolicy, OPERATIONAL_HMC_BUDGET_POLICY_ID
from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.hmc_kernel_selection import candidate_handoff_policy_payload
from bayesfilter.inference.hmc_tuning import _validate_metric_evidence_policy, _validate_metric_probe_num_results
from bayesfilter.inference.hmc_mass_adaptation import (
    TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS,
    STAGED_TIMEOUT_POLICY_STAGE_NAMES,
    _validate_metric_update_requirement,
    _validate_engineering_probe_covariance_multiplier,
    _engineering_probe_config_public_payload,
    _engineering_probe_seed_public_payload,
    HMCStagedTimeoutPolicy,
    _validate_staged_timeout_policy_or_none,
    _validate_nonnegative_perf_counter_or_none,
    _validate_staged_timeout_enlargement_rounds,
    HMCWindowedMassStageConfig,
    HMCWindowedMassStageResult,
    _ceil_div,
)

_OPERATIONAL_EVIDENCE_POLICY_INITIAL_ONLY = "initial_only"


_OPERATIONAL_EVIDENCE_POLICY_ONE_DOUBLING = "one_doubling"


_OPERATIONAL_EVIDENCE_POLICIES = {
    _OPERATIONAL_EVIDENCE_POLICY_INITIAL_ONLY,
    _OPERATIONAL_EVIDENCE_POLICY_ONE_DOUBLING,
}


_OPERATIONAL_CANDIDATE_HANDOFF_POLICY_STRICT = "strict"


_OPERATIONAL_CANDIDATE_HANDOFF_POLICY_SCHEMA = (
    "bayesfilter.hmc_operational_candidate_handoff_policy.v1"
)


_OPERATIONAL_VERIFICATION_BRACKET_POLICY_SINGLE_REPAIR = "single_repair"


_OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT = (
    "one_verified_log_midpoint"
)


_OPERATIONAL_VERIFICATION_BRACKET_POLICIES = {
    _OPERATIONAL_VERIFICATION_BRACKET_POLICY_SINGLE_REPAIR,
    _OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT,
}


ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID = (
    "ordinary_shared_epsilon_screen_v3"
)


ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID = (
    "ordinary_legacy_joint_l_epsilon_grid_v1"
)


ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID = (
    "ordinary_engineering_joint_l_epsilon_grid_v1"
)


_ORDINARY_RUNTIME_BACKEND_POLICY_ID = "ordinary_tf_tfp_runtime_v1"


def resolve_ordinary_hmc_selection_policy(
    algorithm_id: str,
    *,
    engineering_probe_covariance_multiplier_configured: bool = False,
) -> Mapping[str, Any]:
    """Describe the observed ordinary epsilon/L policy without running HMC.

    This resolver is deliberately descriptive.  It does not select an
    algorithm, invent candidate values, or grant scientific authority.  The
    returned policy is suitable for configuration/result provenance and for
    fail-closed authority checks.
    """

    selected = str(algorithm_id)
    if selected == LEGACY_JOINT_L_EPSILON_ALGORITHM_ID:
        return {
            "policy_id": ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID,
            "algorithm_id": selected,
            "epsilon_l_treatment": "per_l_epsilon_ladder_with_bounded_edge_repair",
            "candidate_construction": "legacy_internal_joint_l_epsilon_grid",
            "mass_signature_frozen_during_selection": True,
            "seed_separation": "fresh_candidate_and_replication_seeds",
            "authority_status": "diagnostic_only_non_promoting",
            "claim_bearing_blockers": (
                "legacy_joint_grid_not_owner_promoted",
            ),
        }
    if engineering_probe_covariance_multiplier_configured:
        return {
            "policy_id": ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID,
            "algorithm_id": selected,
            "epsilon_l_treatment": "per_l_epsilon_ladder_with_bounded_edge_repair",
            "candidate_construction": "engineering_probe_joint_l_epsilon_grid",
            "mass_signature_frozen_during_selection": True,
            "seed_separation": "fresh_candidate_and_replication_seeds",
            "authority_status": "engineering_only_non_promoting",
            "claim_bearing_blockers": (
                "engineering_probe_route_not_claim_bearing",
            ),
        }
    if selected == ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID:
        return {
            "policy_id": ORDINARY_BROAD_FIXED_METRIC_POLICY_ID,
            "algorithm_id": selected,
            "epsilon_l_treatment": "independent_epsilon_ladder_for_every_l",
            "candidate_construction": (
                "broad_primary_grid_then_survivor_midpoint_refinement"
            ),
            "primary_l_grid": ORDINARY_BROAD_PRIMARY_L_GRID,
            "refinement_policy": (
                "all_untested_midpoints_adjacent_to_every_survivor"
            ),
            "refinement_rounds": 1,
            "mass_signature_frozen_during_selection": True,
            "seed_separation": "fresh_ladder_and_screen_seeds_for_every_l",
            "authority_status": "artifact_authoritative_stage_handoff",
            "claim_bearing_blockers": (),
        }
    if selected == LEGACY_OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID:
        return {
            "policy_id": ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID,
            "algorithm_id": selected,
            "epsilon_l_treatment": (
                "shared_frozen_epsilon_screen_then_exact_l_retune"
            ),
            "candidate_construction": "floor_anchor_double_trajectory_screen",
            "mass_signature_frozen_during_selection": True,
            "seed_separation": "three_replications_with_fresh_screen_seeds",
            "authority_status": "historical_only_non_promoting",
            "claim_bearing_blockers": (
                "shared_epsilon_screen_not_joint_pair_selection",
            ),
        }
    raise ValueError(f"unknown ordinary HMC selection algorithm_id: {selected}")


def _ordinary_selection_policy_payload(
    config: Any,
) -> Mapping[str, Any]:
    return resolve_ordinary_hmc_selection_policy(
        str(config.algorithm_id),
        engineering_probe_covariance_multiplier_configured=(
            getattr(config, "engineering_probe_covariance_multiplier", None)
            is not None
        ),
    )


def _validated_operational_verification_bracket_policy(value: Any) -> str:
    policy = str(value)
    if policy not in _OPERATIONAL_VERIFICATION_BRACKET_POLICIES:
        allowed = ", ".join(sorted(_OPERATIONAL_VERIFICATION_BRACKET_POLICIES))
        raise ValueError(
            "operational_verification_bracket_policy must be one of: "
            f"{allowed}"
        )
    return policy


def _operational_verification_starts_per_outer_attempt(policy: str) -> int:
    validated = _validated_operational_verification_bracket_policy(policy)
    return (
        3
        if validated
        == _OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT
        else 2
    )


def _validated_operational_candidate_handoff_policy(value: Any) -> str:
    contract = candidate_handoff_policy_payload(value)
    if contract.get("schema") != _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_SCHEMA:
        raise ValueError("operational candidate handoff policy schema mismatch")
    return str(contract["policy"])


@dataclass(frozen=True)
class HMCGeometryScaledBudgetTimingPolicy:
    """Public-safe policy tying HMC tuning budgets to dimension and geometry.

    The policy does not expose sampled states, mass arrays, step sizes,
    leapfrog counts, or candidate grids.  It records why a draw budget was
    chosen: target dimension, covariance condition pressure, effective
    dimension/anisotropy pressure, and SPD-regularization pressure.  Emergency
    clock limits are machine-protection caps only; meaningful progress remains
    a separate monitor decision.
    """

    policy_id: str = "bayesfilter_hmc_geometry_scaled_budget_timing_v1"
    dimension_factor: float = 20.0
    min_initial_budget: int = 1000
    max_initial_budget: int = 5000
    max_tune_budget: int = 10000
    min_geometry_multiplier: float = 1.0
    max_geometry_multiplier: float = 4.0
    condition_log10_weight: float = 0.25
    anisotropy_sqrt_weight: float = 0.50
    regularization_clip_weight: float = 0.05
    regularization_nonpositive_weight: float = 0.25
    diagonal_fallback_multiplier: float = 1.50
    bootstrap_sqrt_dimension_factor: float = 4.0
    bootstrap_min_results: int = 32
    bootstrap_max_results: int = 1024
    bootstrap_burnin_fraction: float = 0.25
    emergency_min_stage_s: float = 3600.0
    emergency_max_stage_s: float = 21600.0
    emergency_global_cap_s: float = 86400.0
    emergency_reserve_s: float = 600.0
    stage_time_budget_multiplier: Mapping[str, float] | None = None
    source: str = "bayesfilter.inference.hmc_kernel_tuning.geometry_scaled_budget_timing_policy"

    def __post_init__(self) -> None:
        policy_id = str(self.policy_id)
        if not policy_id:
            raise ValueError("policy_id must be non-empty")
        object.__setattr__(self, "policy_id", policy_id)
        for name in (
            "dimension_factor",
            "min_geometry_multiplier",
            "max_geometry_multiplier",
            "condition_log10_weight",
            "anisotropy_sqrt_weight",
            "regularization_clip_weight",
            "regularization_nonpositive_weight",
            "diagonal_fallback_multiplier",
            "bootstrap_sqrt_dimension_factor",
            "bootstrap_burnin_fraction",
            "emergency_min_stage_s",
            "emergency_max_stage_s",
            "emergency_global_cap_s",
            "emergency_reserve_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.dimension_factor <= 0.0:
            raise ValueError("dimension_factor must be positive")
        if self.min_geometry_multiplier <= 0.0:
            raise ValueError("min_geometry_multiplier must be positive")
        if self.max_geometry_multiplier < self.min_geometry_multiplier:
            raise ValueError("max_geometry_multiplier must not be smaller than minimum")
        if self.diagonal_fallback_multiplier < 1.0:
            raise ValueError("diagonal_fallback_multiplier must be at least 1")
        if self.bootstrap_sqrt_dimension_factor <= 0.0:
            raise ValueError("bootstrap_sqrt_dimension_factor must be positive")
        if self.bootstrap_burnin_fraction <= 0.0:
            raise ValueError("bootstrap_burnin_fraction must be positive")
        for name in (
            "min_initial_budget",
            "max_initial_budget",
            "max_tune_budget",
            "bootstrap_min_results",
            "bootstrap_max_results",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.max_initial_budget < self.min_initial_budget:
            raise ValueError("max_initial_budget must be at least min_initial_budget")
        if self.max_tune_budget < self.max_initial_budget:
            raise ValueError("max_tune_budget must be at least max_initial_budget")
        if self.bootstrap_max_results < self.bootstrap_min_results:
            raise ValueError("bootstrap_max_results must be at least bootstrap_min_results")
        if self.emergency_max_stage_s < self.emergency_min_stage_s:
            raise ValueError("emergency_max_stage_s must be at least emergency_min_stage_s")
        if self.emergency_global_cap_s <= self.emergency_reserve_s:
            raise ValueError("emergency_global_cap_s must exceed emergency_reserve_s")
        multipliers = (
            self._default_stage_time_budget_multiplier()
            if self.stage_time_budget_multiplier is None
            else dict(self.stage_time_budget_multiplier)
        )
        if set(multipliers) != set(STAGED_TIMEOUT_POLICY_STAGE_NAMES):
            raise ValueError("stage_time_budget_multiplier must cover every stage")
        normalized: dict[str, float] = {}
        for stage, value in multipliers.items():
            multiplier = float(value)
            if not math.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError("stage time multipliers must be positive and finite")
            normalized[str(stage)] = multiplier
        object.__setattr__(self, "stage_time_budget_multiplier", normalized)
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    @staticmethod
    def _default_stage_time_budget_multiplier() -> Mapping[str, float]:
        return {
            "geometry_and_bootstrap": 1.0,
            "phase7_pre_windowed": 1.0,
            "windowed_mass": 1.0,
            "fixed_mass_step": 1.0,
            "frozen_step_trajectory": 1.0,
            "fresh_fixed_kernel_verification": 1.0,
        }

    def geometry_summary(
        self,
        *,
        target_dimension: int | None = None,
        mass_artifact: PrecomputedMassArtifact | None = None,
    ) -> Mapping[str, Any]:
        dimension = (
            int(mass_artifact.dimension)
            if mass_artifact is not None
            else int(target_dimension or 0)
        )
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        eigen_summary = (
            {}
            if mass_artifact is None
            else dict(mass_artifact.eigen_summary)
        )
        eigenvalues = _geometry_policy_eigenvalues(
            dimension=dimension,
            mass_artifact=mass_artifact,
            eigen_summary=eigen_summary,
        )
        condition_number = _geometry_policy_condition_number(
            eigenvalues=eigenvalues,
            eigen_summary=eigen_summary,
        )
        effective_dimension = _geometry_policy_effective_dimension(eigenvalues)
        anisotropy_ratio = float(dimension) / max(1.0, effective_dimension)
        regularization = (
            {}
            if mass_artifact is None
            else dict(mass_artifact.regularization_report)
        )
        regularization_counts = _geometry_policy_regularization_counts(
            regularization
        )
        multiplier = self.geometry_multiplier(
            dimension=dimension,
            condition_number=condition_number,
            effective_dimension=effective_dimension,
            regularization_counts=regularization_counts,
        )
        return {
            "schema": "bayesfilter.hmc_geometry_scaled_budget_summary.v1",
            "dimension": dimension,
            "geometry_source": (
                "mass_artifact_metadata" if mass_artifact is not None else "dimension_only"
            ),
            "condition_number": condition_number,
            "condition_log10": (
                None
                if condition_number is None
                else float(math.log10(max(1.0, condition_number)))
            ),
            "effective_dimension": effective_dimension,
            "effective_dimension_ratio": effective_dimension / float(dimension),
            "anisotropy_ratio": anisotropy_ratio,
            "regularization_pressure": regularization_counts,
            "geometry_multiplier": multiplier,
            "raw_eigenvalues_exposed": False,
            "mass_arrays_exposed": False,
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
        }

    def geometry_multiplier(
        self,
        *,
        dimension: int,
        condition_number: float | None,
        effective_dimension: float,
        regularization_counts: Mapping[str, Any],
    ) -> float:
        condition_log10 = (
            0.0
            if condition_number is None
            else max(0.0, math.log10(max(1.0, float(condition_number))))
        )
        condition_pressure = 1.0 + self.condition_log10_weight * condition_log10
        anisotropy_ratio = float(dimension) / max(1.0, float(effective_dimension))
        if not math.isfinite(anisotropy_ratio) or anisotropy_ratio < 1.0:
            anisotropy_ratio = 1.0
        anisotropy_pressure = 1.0 + self.anisotropy_sqrt_weight * (
            math.sqrt(anisotropy_ratio) - 1.0
        )
        clipped = int(regularization_counts.get("clipped_eigenvalue_count", 0))
        nonpositive = int(
            regularization_counts.get("raw_nonpositive_eigenvalue_count", 0)
        )
        regularization_pressure = (
            1.0
            + self.regularization_clip_weight * max(0, clipped)
            + self.regularization_nonpositive_weight * max(0, nonpositive)
        )
        if bool(regularization_counts.get("diagonal_fallback_used", False)):
            regularization_pressure *= self.diagonal_fallback_multiplier
        multiplier = (
            float(condition_pressure)
            * float(anisotropy_pressure)
            * float(regularization_pressure)
        )
        return float(
            min(
                max(multiplier, self.min_geometry_multiplier),
                self.max_geometry_multiplier,
            )
        )

    def attempt_budget_payload(
        self,
        *,
        target_dimension: int,
        attempt_index: int,
        mass_artifact: PrecomputedMassArtifact | None = None,
    ) -> Mapping[str, Any]:
        dimension = int(target_dimension)
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        index = int(attempt_index)
        if index < 0:
            raise ValueError("attempt_index must be non-negative")
        summary = self.geometry_summary(
            target_dimension=dimension,
            mass_artifact=mass_artifact,
        )
        base_uncapped = int(
            math.ceil(
                self.dimension_factor
                * float(dimension)
                * float(summary["geometry_multiplier"])
            )
        )
        budget0 = int(
            min(
                self.max_initial_budget,
                max(self.min_initial_budget, base_uncapped),
            )
        )
        budget = int(min(self.max_tune_budget, budget0 * (2 ** index)))
        phase5_screen = max(32, _ceil_div(budget, 4))
        phase6_screen = max(32, _ceil_div(budget, 4))
        verification_results = max(64, _ceil_div(budget, 2))
        return {
            "target_dimension": dimension,
            "attempt_index": index,
            "budget": budget,
            "phase4_warmup_steps": budget,
            "phase5_tune_budgets": (
                _ceil_div(budget, 4),
                _ceil_div(budget, 2),
                budget,
            ),
            "phase5_screen_num_results": phase5_screen,
            "phase5_screen_burnin_steps": max(8, _ceil_div(phase5_screen, 4)),
            "phase6_screen_num_results": phase6_screen,
            "phase6_screen_burnin_steps": max(8, _ceil_div(phase6_screen, 4)),
            "verification_num_results": verification_results,
            "verification_num_burnin_steps": max(
                16,
                _ceil_div(verification_results, 4),
            ),
            "budget_formula": (
                "budget0=clamp(ceil(dimension_factor*d*geometry_multiplier), "
                "min_initial_budget, max_initial_budget); "
                "budget_k=min(max_tune_budget, budget0*2**attempt_index)"
            ),
            "budget_formula_parameters": self.budget_formula_parameters(),
            "geometry_budget_summary": summary,
            "budget0_uncapped": base_uncapped,
            "budget0_after_floor_and_cap": budget0,
            "initial_budget_cap_active": budget0 < max(
                self.min_initial_budget,
                base_uncapped,
            ),
            "tune_budget_cap_active": budget < budget0 * (2 ** index),
            "budget_claim": (
                "dimension/geometry-scaled tuning work budget; not posterior "
                "convergence or sampler-validity evidence"
            ),
        }

    def bootstrap_screen_counts(
        self,
        *,
        target_dimension: int,
        mass_artifact: PrecomputedMassArtifact | None = None,
    ) -> Mapping[str, Any]:
        summary = self.geometry_summary(
            target_dimension=int(target_dimension),
            mass_artifact=mass_artifact,
        )
        raw_results = int(
            math.ceil(
                self.bootstrap_sqrt_dimension_factor
                * math.sqrt(float(summary["dimension"]))
                * float(summary["geometry_multiplier"])
            )
        )
        results = int(
            min(
                self.bootstrap_max_results,
                max(self.bootstrap_min_results, raw_results),
            )
        )
        burnin = max(1, int(math.ceil(results * self.bootstrap_burnin_fraction)))
        return {
            "screen_num_results": results,
            "screen_num_burnin_steps": burnin,
            "raw_screen_num_results": raw_results,
            "geometry_budget_summary": summary,
            "bootstrap_formula": (
                "screen_results=clamp(ceil(sqrt_dimension_factor*sqrt(d)*"
                "geometry_multiplier), min_results, max_results); "
                "burnin=ceil(results*burnin_fraction)"
            ),
            "bootstrap_formula_parameters": {
                "sqrt_dimension_factor": self.bootstrap_sqrt_dimension_factor,
                "min_results": self.bootstrap_min_results,
                "max_results": self.bootstrap_max_results,
                "burnin_fraction": self.bootstrap_burnin_fraction,
            },
            "bootstrap_claim": (
                "public-safe mechanics and finite-runtime screen only; not a "
                "reasonable-posterior or tuning-success gate"
            ),
        }

    def stage_budgets_s(
        self,
        *,
        target_dimension: int | None = None,
        mass_artifact: PrecomputedMassArtifact | None = None,
    ) -> Mapping[str, float]:
        summary = self.geometry_summary(
            target_dimension=1 if target_dimension is None else int(target_dimension),
            mass_artifact=mass_artifact,
        )
        stage_floor = float(self.emergency_min_stage_s) * float(
            summary["geometry_multiplier"]
        )
        stage_budgets: dict[str, float] = {}
        for stage, multiplier in self.stage_time_budget_multiplier.items():
            budget = min(
                float(self.emergency_max_stage_s),
                max(float(self.emergency_min_stage_s), stage_floor * float(multiplier)),
            )
            stage_budgets[str(stage)] = float(budget)
        return stage_budgets

    def stage_budget_provenance(self) -> Mapping[str, str]:
        return {
            stage: "geometry_scaled_emergency_cap_machine_protection_not_progress_gate"
            for stage in STAGED_TIMEOUT_POLICY_STAGE_NAMES
        }

    def staged_timeout_policy(self) -> "HMCStagedTimeoutPolicy":
        return HMCStagedTimeoutPolicy(
            policy_id=f"{self.policy_id}.emergency_caps",
            stage_budgets_s=self.stage_budgets_s(),
            stage_budget_provenance=self.stage_budget_provenance(),
            global_cap_s=self.emergency_global_cap_s,
            reserve_s=self.emergency_reserve_s,
            source=f"{self.source}.staged_timeout_policy",
        )

    def budget_formula_parameters(self) -> Mapping[str, Any]:
        return {
            "dimension_factor": self.dimension_factor,
            "min_initial_budget": self.min_initial_budget,
            "max_initial_budget": self.max_initial_budget,
            "max_tune_budget": self.max_tune_budget,
            "min_geometry_multiplier": self.min_geometry_multiplier,
            "max_geometry_multiplier": self.max_geometry_multiplier,
            "condition_log10_weight": self.condition_log10_weight,
            "anisotropy_sqrt_weight": self.anisotropy_sqrt_weight,
            "regularization_clip_weight": self.regularization_clip_weight,
            "regularization_nonpositive_weight": (
                self.regularization_nonpositive_weight
            ),
            "diagonal_fallback_multiplier": self.diagonal_fallback_multiplier,
        }

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_geometry_scaled_budget_timing_policy.v1",
            "policy_id": self.policy_id,
            "budget_formula_parameters": self.budget_formula_parameters(),
            "bootstrap_formula_parameters": {
                "sqrt_dimension_factor": self.bootstrap_sqrt_dimension_factor,
                "min_results": self.bootstrap_min_results,
                "max_results": self.bootstrap_max_results,
                "burnin_fraction": self.bootstrap_burnin_fraction,
            },
            "emergency_timing_policy": {
                "role": "machine_protection_only_not_scientific_stop_rule",
                "min_stage_s": self.emergency_min_stage_s,
                "max_stage_s": self.emergency_max_stage_s,
                "global_cap_s": self.emergency_global_cap_s,
                "reserve_s": self.emergency_reserve_s,
                "stage_time_budget_multiplier": dict(
                    self.stage_time_budget_multiplier
                ),
                "progress_monitor_is_separate": True,
            },
            "source": self.source,
            "public_safe": True,
            "raw_eigenvalues_exposed": False,
            "mass_arrays_exposed": False,
            "hmc_mechanics_exposed": False,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
        }


def _geometry_scaled_budget_timing_policy() -> HMCGeometryScaledBudgetTimingPolicy:
    return HMCGeometryScaledBudgetTimingPolicy()


def _geometry_policy_eigenvalues(
    *,
    dimension: int,
    mass_artifact: PrecomputedMassArtifact | None,
    eigen_summary: Mapping[str, Any],
) -> Any:
    import tensorflow as tf

    raw = eigen_summary.get("eigenvalues")
    if raw is not None:
        try:
            values = tf.reshape(
                tf.convert_to_tensor(tuple(raw), dtype=tf.float64), [-1]
            )
        except TypeError:
            values = tf.zeros((0,), dtype=tf.float64)
        if values.shape == (int(dimension),) and bool(
            tf.reduce_all(tf.math.is_finite(values)).numpy()
        ):
            return tf.maximum(values, tf.constant(1.0e-300, dtype=tf.float64))
    if mass_artifact is not None:
        covariance = tf.convert_to_tensor(mass_artifact.covariance, dtype=tf.float64)
        if covariance.shape == (int(dimension), int(dimension)):
            values = tf.linalg.eigvalsh(
                0.5 * (covariance + tf.transpose(covariance))
            )
            if bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
                return tf.maximum(values, tf.constant(1.0e-300, dtype=tf.float64))
    return tf.ones((int(dimension),), dtype=tf.float64)


def _geometry_policy_condition_number(
    *,
    eigenvalues: Any,
    eigen_summary: Mapping[str, Any],
) -> float | None:
    import tensorflow as tf

    raw_condition = eigen_summary.get("condition_number")
    if raw_condition is not None:
        try:
            condition = float(raw_condition)
        except (TypeError, ValueError):
            condition = float("nan")
        if math.isfinite(condition) and condition >= 1.0:
            return condition
    values = tf.reshape(tf.convert_to_tensor(eigenvalues, dtype=tf.float64), [-1])
    positive = tf.boolean_mask(
        values, tf.logical_and(tf.math.is_finite(values), values > 0.0)
    )
    if int(tf.size(positive).numpy()) == 0:
        return None
    condition = float((tf.reduce_max(positive) / tf.reduce_min(positive)).numpy())
    return condition if math.isfinite(condition) and condition >= 1.0 else None


def _geometry_policy_effective_dimension(eigenvalues: Any) -> float:
    import tensorflow as tf

    values = tf.reshape(tf.convert_to_tensor(eigenvalues, dtype=tf.float64), [-1])
    values = tf.boolean_mask(
        values, tf.logical_and(tf.math.is_finite(values), values > 0.0)
    )
    if int(tf.size(values).numpy()) == 0:
        return 1.0
    total = float(tf.reduce_sum(values).numpy())
    squared_total = float(tf.reduce_sum(tf.square(values)).numpy())
    if not math.isfinite(total) or not math.isfinite(squared_total) or squared_total <= 0.0:
        return 1.0
    return float(
        min(max((total * total) / squared_total, 1.0), int(tf.size(values).numpy()))
    )


def _geometry_policy_regularization_counts(
    regularization_report: Mapping[str, Any],
) -> Mapping[str, Any]:
    def int_field(*names: str) -> int:
        for name in names:
            if name not in regularization_report:
                continue
            try:
                return max(0, int(regularization_report[name]))
            except (TypeError, ValueError):
                return 0
        return 0

    clipped = int_field("clipped_eigenvalue_count", "covariance_clipped_eigenvalue_count")
    nonpositive = int_field(
        "raw_nonpositive_eigenvalue_count",
        "covariance_raw_nonpositive_eigenvalue_count",
    )
    fallback = bool(regularization_report.get("diagonal_fallback_used", False))
    return {
        "clipped_eigenvalue_count": clipped,
        "raw_nonpositive_eigenvalue_count": nonpositive,
        "diagonal_fallback_used": fallback,
        "regularization_fields_used": (
            "clipped_eigenvalue_count",
            "raw_nonpositive_eigenvalue_count",
            "diagonal_fallback_used",
        ),
    }


def _attempt_budget_policy_from_payload(
    payload: Mapping[str, Any],
    *,
    serious_policy: bool,
    public_budget_class: str | None = None,
    public_budget_cap: int | None = None,
    public_max_attempts: int | None = None,
    public_diagnostic_preset: str | None = None,
) -> "_HMCAttemptBudgetPolicy":
    operational_keys = {
        "operational_screen_num_results",
        "operational_screen_num_burnin_steps",
        "operational_evidence_extension_checkpoints",
        "operational_exact_l_tune_adaptation_steps",
        "operational_verification_num_results",
        "operational_verification_num_burnin_steps",
        "operational_budget_policy_id",
        "operational_budget_policy_hash",
    }
    present_operational_keys = operational_keys.intersection(payload)
    if present_operational_keys and present_operational_keys != operational_keys:
        missing = ", ".join(sorted(operational_keys - present_operational_keys))
        raise ValueError(f"serialized operational budget policy is incomplete: {missing}")
    if present_operational_keys:
        operational = HMCOperationalStatisticalWorkPolicy(
            initial_candidate_results=int(payload["operational_screen_num_results"]),
            candidate_burnin_steps=int(
                payload["operational_screen_num_burnin_steps"]
            ),
            evidence_extension_checkpoints=tuple(
                int(item)
                for item in payload["operational_evidence_extension_checkpoints"]
            ),
            exact_l_tune_adaptation_steps=int(
                payload["operational_exact_l_tune_adaptation_steps"]
            ),
            fresh_verification_results=int(
                payload["operational_verification_num_results"]
            ),
            fresh_verification_burnin_steps=int(
                payload["operational_verification_num_burnin_steps"]
            ),
            fresh_verification_starts_per_outer_attempt=int(
                payload.get("operational_verification_starts_per_outer_attempt", 2)
            ),
            policy_id=str(payload["operational_budget_policy_id"]),
        )
        if str(payload["operational_budget_policy_hash"]) != operational.policy_hash:
            raise ValueError("serialized operational budget policy hash mismatch")
    else:
        # Compatibility for artifacts written before operational budgets were
        # serialized separately from the legacy geometry-scaled fields.
        operational = HMCOperationalStatisticalWorkPolicy()
    return _HMCAttemptBudgetPolicy(
        target_dimension=int(payload["target_dimension"]),
        attempt_index=int(payload["attempt_index"]),
        budget=int(payload["budget"]),
        phase4_warmup_steps=int(payload["phase4_warmup_steps"]),
        phase5_tune_budgets=tuple(int(item) for item in payload["phase5_tune_budgets"]),
        phase5_screen_num_results=int(payload["phase5_screen_num_results"]),
        phase5_screen_burnin_steps=int(payload["phase5_screen_burnin_steps"]),
        phase6_screen_num_results=int(payload["phase6_screen_num_results"]),
        phase6_screen_burnin_steps=int(payload["phase6_screen_burnin_steps"]),
        verification_num_results=int(payload["verification_num_results"]),
        verification_num_burnin_steps=int(payload["verification_num_burnin_steps"]),
        operational_screen_num_results=operational.initial_candidate_results,
        operational_screen_num_burnin_steps=operational.candidate_burnin_steps,
        operational_evidence_extension_checkpoints=(
            operational.evidence_extension_checkpoints
        ),
        operational_exact_l_tune_adaptation_steps=(
            operational.exact_l_tune_adaptation_steps
        ),
        operational_verification_num_results=(
            operational.fresh_verification_results
        ),
        operational_verification_num_burnin_steps=(
            operational.fresh_verification_burnin_steps
        ),
        operational_verification_starts_per_outer_attempt=(
            operational.fresh_verification_starts_per_outer_attempt
        ),
        operational_budget_policy_id=operational.policy_id,
        operational_budget_policy_hash=operational.policy_hash,
        serious_policy=serious_policy,
        public_budget_class=public_budget_class,
        public_budget_cap=public_budget_cap,
        public_max_attempts=public_max_attempts,
        public_diagnostic_preset=public_diagnostic_preset,
        budget_formula=str(payload["budget_formula"]),
        budget_formula_parameters=dict(payload["budget_formula_parameters"]),
        geometry_budget_summary=dict(payload["geometry_budget_summary"]),
        budget0_uncapped=int(payload["budget0_uncapped"]),
        budget0_after_floor_and_cap=int(payload["budget0_after_floor_and_cap"]),
        budget_claim=str(payload["budget_claim"]),
    )


_PHASE7_MAX_ATTEMPTS_CAP = 10


_HANDOFF_SCREEN_POLICY_PHASE22_HEURISTIC_GATE = "phase22_heuristic_viability_gate"


_HANDOFF_SCREEN_POLICY_PHASE23_NOMINATION_ONLY = "phase23_nomination_only"


_HANDOFF_SCREEN_POLICIES = {
    _HANDOFF_SCREEN_POLICY_PHASE22_HEURISTIC_GATE,
    _HANDOFF_SCREEN_POLICY_PHASE23_NOMINATION_ONLY,
}


_PUBLIC_TERMINAL_PHASE6_REPAIR_SCREEN_MAX_RESULTS = 128


def _validate_handoff_screen_policy(value: Any) -> str:
    policy = str(value)
    if policy not in _HANDOFF_SCREEN_POLICIES:
        allowed = ", ".join(sorted(_HANDOFF_SCREEN_POLICIES))
        raise ValueError(f"handoff_screen_policy must be one of: {allowed}")
    return policy


def _validate_trajectory_window_multiplier(
    value: Any,
    *,
    name: str,
) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return multiplier


def _validate_trajectory_window_multipliers(
    lower: Any,
    upper: Any,
) -> tuple[float, float]:
    lower_value = _validate_trajectory_window_multiplier(
        lower,
        name="trajectory_window_lower_multiplier",
    )
    upper_value = _validate_trajectory_window_multiplier(
        upper,
        name="trajectory_window_upper_multiplier",
    )
    if lower_value > 1.0:
        raise ValueError("trajectory_window_lower_multiplier must be <= 1")
    if upper_value < 1.0:
        raise ValueError("trajectory_window_upper_multiplier must be >= 1")
    if lower_value > upper_value:
        raise ValueError(
            "trajectory_window_lower_multiplier must not exceed "
            "trajectory_window_upper_multiplier"
        )
    return lower_value, upper_value


def _validate_positive_int_or_none(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return integer


@dataclass(frozen=True)
class HMCTuneVerifyRepairLoopConfig:
    """Policy-level config for the Phase 7 outer tuning loop.

    The config intentionally hides raw HMC mechanics, stage budgets, candidate
    grids, and final verification draw counts from model clients. Those are
    derived from target dimension by BayesFilter-owned policy.
    """

    algorithm_id: str = ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID
    operational_budget_policy_id: str = OPERATIONAL_HMC_BUDGET_POLICY_ID
    operational_candidate_handoff_policy: str = (
        _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_STRICT
    )
    operational_verification_bracket_policy: str = (
        _OPERATIONAL_VERIFICATION_BRACKET_POLICY_SINGLE_REPAIR
    )
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    max_leapfrog_steps: int = _GEOMETRY_MAX_LEAPFROG
    step_repair_factor: float = 2.0
    step_repair_min_directional_factor: float = 1.25
    step_repair_high_acceptance_directional_factor: float | None = None
    step_repair_high_acceptance_ladder_max_factor: float | None = None
    repair_nonfinite_proposal_screen: bool = False
    trajectory_window_lower_multiplier: float = 0.3
    trajectory_window_upper_multiplier: float = 3.0
    handoff_screen_policy: str = _HANDOFF_SCREEN_POLICY_PHASE22_HEURISTIC_GATE
    max_attempts: int = 5
    terminal_phase6_repair_extra_attempts: int = 0
    seed: tuple[int, int] = (20260621, 7)
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
    verification_chunk_max_results: int | None = None
    verification_min_retained_results_for_pass: int | None = None
    staged_timeout_policy: HMCStagedTimeoutPolicy | None = None
    staged_timeout_global_started_perf_counter_s: float | None = None
    staged_timeout_stage_started_perf_counter_s: float | None = None
    staged_timeout_enlargement_rounds: Mapping[str, int] | None = None
    incall_progress_heartbeat_s: float | None = None
    source: str = "bayesfilter.inference.hmc_kernel_tuning.tune_verify_repair_loop"

    def __post_init__(self) -> None:
        algorithm_id = str(self.algorithm_id)
        if not algorithm_id:
            raise ValueError("algorithm_id must be non-empty")
        object.__setattr__(self, "algorithm_id", algorithm_id)
        if (
            algorithm_id == ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID
            and self.engineering_probe_covariance_multiplier is None
            and int(self.max_leapfrog_steps)
            != ORDINARY_BROAD_MAX_LEAPFROG_STEPS
        ):
            raise ValueError(
                "canonical ordinary tuning fixes max_leapfrog_steps at "
                f"{ORDINARY_BROAD_MAX_LEAPFROG_STEPS}"
            )
        budget_policy_id = str(self.operational_budget_policy_id)
        if budget_policy_id != OPERATIONAL_HMC_BUDGET_POLICY_ID:
            raise ValueError("operational_budget_policy_id is fixed by BayesFilter")
        object.__setattr__(
            self,
            "operational_budget_policy_id",
            budget_policy_id,
        )
        candidate_handoff_policy = (
            _validated_operational_candidate_handoff_policy(
                self.operational_candidate_handoff_policy
            )
        )
        object.__setattr__(
            self,
            "operational_candidate_handoff_policy",
            candidate_handoff_policy,
        )
        verification_bracket_policy = (
            _validated_operational_verification_bracket_policy(
                self.operational_verification_bracket_policy
            )
        )
        if (
            verification_bracket_policy
            == _OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT
            and algorithm_id != LEGACY_OPERATIONAL_FIXED_TRAJECTORY_ALGORITHM_ID
        ):
            raise ValueError(
                "one_verified_log_midpoint is historical compatibility logic and "
                "requires the explicit legacy shared-epsilon algorithm"
            )
        object.__setattr__(
            self,
            "operational_verification_bracket_policy",
            verification_bracket_policy,
        )
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        acceptance_band = _validate_band(self.acceptance_band, name="acceptance_band")
        repair_band = _validate_band(self.repair_band, name="repair_band")
        if repair_band[0] > acceptance_band[0] or repair_band[1] < acceptance_band[1]:
            raise ValueError("repair_band must contain acceptance_band")
        object.__setattr__(self, "acceptance_band", acceptance_band)
        object.__setattr__(self, "repair_band", repair_band)
        object.__setattr__(
            self,
            "max_leapfrog_steps",
            _validate_max_leapfrog_steps(self.max_leapfrog_steps),
        )
        lower, upper = _validate_trajectory_window_multipliers(
            self.trajectory_window_lower_multiplier,
            self.trajectory_window_upper_multiplier,
        )
        object.__setattr__(
            self,
            "step_repair_factor",
            _validate_step_repair_multiplier(
                self.step_repair_factor,
                name="step_repair_factor",
            ),
        )
        min_directional_factor = _validate_step_repair_multiplier(
            self.step_repair_min_directional_factor,
            name="step_repair_min_directional_factor",
        )
        if min_directional_factor > 2.0:
            raise ValueError("step_repair_min_directional_factor must be at most 2")
        object.__setattr__(
            self,
            "step_repair_min_directional_factor",
            min_directional_factor,
        )
        high_factor = (
            self.step_repair_factor
            if self.step_repair_high_acceptance_directional_factor is None
            else self.step_repair_high_acceptance_directional_factor
        )
        object.__setattr__(
            self,
            "step_repair_high_acceptance_directional_factor",
            _validate_step_repair_multiplier(
                high_factor,
                name="step_repair_high_acceptance_directional_factor",
            ),
        )
        high_ladder_max = (
            high_factor
            if self.step_repair_high_acceptance_ladder_max_factor is None
            else self.step_repair_high_acceptance_ladder_max_factor
        )
        high_ladder_max = _validate_step_repair_multiplier(
            high_ladder_max,
            name="step_repair_high_acceptance_ladder_max_factor",
        )
        if high_ladder_max < high_factor:
            raise ValueError(
                "step_repair_high_acceptance_ladder_max_factor must be at least "
                "step_repair_high_acceptance_directional_factor"
            )
        object.__setattr__(
            self,
            "step_repair_high_acceptance_ladder_max_factor",
            high_ladder_max,
        )
        object.__setattr__(
            self,
            "repair_nonfinite_proposal_screen",
            bool(self.repair_nonfinite_proposal_screen),
        )
        object.__setattr__(self, "trajectory_window_lower_multiplier", lower)
        object.__setattr__(self, "trajectory_window_upper_multiplier", upper)
        object.__setattr__(
            self,
            "handoff_screen_policy",
            _validate_handoff_screen_policy(self.handoff_screen_policy),
        )
        attempts = int(self.max_attempts)
        if attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if attempts > _PHASE7_MAX_ATTEMPTS_CAP:
            raise ValueError(
                f"Phase 7 max_attempts is hard-capped at {_PHASE7_MAX_ATTEMPTS_CAP}"
            )
        if (
            verification_bracket_policy
            == _OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT
            and attempts < 3
        ):
            raise ValueError(
                "one_verified_log_midpoint requires at least three attempt slots"
            )
        object.__setattr__(self, "max_attempts", attempts)
        terminal_extra = int(self.terminal_phase6_repair_extra_attempts)
        if terminal_extra < 0 or terminal_extra > 1:
            raise ValueError("terminal_phase6_repair_extra_attempts must be 0 or 1")
        object.__setattr__(
            self,
            "terminal_phase6_repair_extra_attempts",
            terminal_extra,
        )
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
        object.__setattr__(
            self,
            "metric_update_requirement",
            _validate_metric_update_requirement(self.metric_update_requirement),
        )
        object.__setattr__(
            self,
            "engineering_probe_covariance_multiplier",
            _validate_engineering_probe_covariance_multiplier(
                self.engineering_probe_covariance_multiplier
            ),
        )
        if (
            self.metric_update_requirement == "require_operational_update"
            and mass_policy == "fixed_identity"
        ):
            raise ValueError(
                "require_operational_update is incompatible with fixed_identity"
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
        object.__setattr__(
            self,
            "verification_chunk_max_results",
            _validate_positive_int_or_none(
                self.verification_chunk_max_results,
                name="verification_chunk_max_results",
            ),
        )
        object.__setattr__(
            self,
            "verification_min_retained_results_for_pass",
            _validate_positive_int_or_none(
                self.verification_min_retained_results_for_pass,
                name="verification_min_retained_results_for_pass",
            ),
        )
        object.__setattr__(
            self,
            "staged_timeout_policy",
            _validate_staged_timeout_policy_or_none(self.staged_timeout_policy),
        )
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
        heartbeat = (
            None
            if self.incall_progress_heartbeat_s is None
            else float(self.incall_progress_heartbeat_s)
        )
        if heartbeat is not None and (not math.isfinite(heartbeat) or heartbeat <= 0.0):
            raise ValueError("incall_progress_heartbeat_s must be positive and finite")
        object.__setattr__(self, "incall_progress_heartbeat_s", heartbeat)
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    def payload(self) -> Mapping[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "ordinary_selection_policy": _ordinary_selection_policy_payload(self),
            "runtime_backend_policy": _ORDINARY_RUNTIME_BACKEND_POLICY_ID,
            "operational_budget_policy_id": self.operational_budget_policy_id,
            "operational_candidate_handoff_policy": (
                self.operational_candidate_handoff_policy
            ),
            "operational_candidate_handoff_policy_contract": (
                candidate_handoff_policy_payload(
                    self.operational_candidate_handoff_policy
                )
            ),
            "operational_verification_bracket_policy": (
                self.operational_verification_bracket_policy
            ),
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "step_repair_factor": self.step_repair_factor,
            "step_repair_min_directional_factor": (
                self.step_repair_min_directional_factor
            ),
            "step_repair_high_acceptance_directional_factor": (
                self.step_repair_high_acceptance_directional_factor
            ),
            "step_repair_high_acceptance_ladder_max_factor": (
                self.step_repair_high_acceptance_ladder_max_factor
            ),
            "repair_nonfinite_proposal_screen": self.repair_nonfinite_proposal_screen,
            "trajectory_window_lower_multiplier": self.trajectory_window_lower_multiplier,
            "trajectory_window_upper_multiplier": self.trajectory_window_upper_multiplier,
            "handoff_screen_policy": self.handoff_screen_policy,
            "max_attempts": self.max_attempts,
            "terminal_phase6_repair_extra_attempts": (
                self.terminal_phase6_repair_extra_attempts
            ),
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
            "public_timeout_started_perf_counter_s": self.public_timeout_started_perf_counter_s,
            "verification_chunk_max_results": self.verification_chunk_max_results,
            "verification_min_retained_results_for_pass": (
                self.verification_min_retained_results_for_pass
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
            "incall_progress_heartbeat_s": self.incall_progress_heartbeat_s,
            "source": self.source,
        }


@dataclass(frozen=True)
class HMCKernelTuningConfig:
    """Public one-call HMC kernel tuning policy.

    This config is intentionally model-facing.  It exposes presets, acceptance
    policy, geometry policy, seeds, and target metadata; it does not expose
    caller-chosen step sizes, leapfrog counts, candidate grids, mass windows,
    warmup budgets, draw counts, or budget schedules.
    """

    algorithm_id: str = ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID
    operational_budget_policy_id: str = OPERATIONAL_HMC_BUDGET_POLICY_ID
    operational_evidence_policy: str = _OPERATIONAL_EVIDENCE_POLICY_INITIAL_ONLY
    operational_candidate_handoff_policy: str = (
        _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_STRICT
    )
    operational_candidate_handoff_policy_schema_version: str = (
        _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_SCHEMA
    )
    operational_verification_bracket_policy: str = (
        _OPERATIONAL_VERIFICATION_BRACKET_POLICY_SINGLE_REPAIR
    )
    preset: str = "standard"
    target_accept_prob: float = 0.70
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    repair_band: tuple[float, float] = (0.55, 0.85)
    max_leapfrog_steps: int = _GEOMETRY_MAX_LEAPFROG
    step_repair_factor: float = 2.0
    step_repair_min_directional_factor: float = 1.25
    step_repair_high_acceptance_directional_factor: float | None = None
    step_repair_high_acceptance_ladder_max_factor: float | None = None
    repair_nonfinite_proposal_screen: bool = False
    trajectory_window_lower_multiplier: float = 0.3
    trajectory_window_upper_multiplier: float = 3.0
    handoff_screen_policy: str = _HANDOFF_SCREEN_POLICY_PHASE22_HEURISTIC_GATE
    bootstrap_max_repairs: int = 5
    bootstrap_initialization_rounds: int = 0
    max_attempts: int = 5
    candidate_search_bound_expansion_steps: int = 0
    terminal_phase6_repair_extra_attempts: int = 0
    seed: tuple[int, int] = (20260621, 8)
    chain_execution_mode: str = "tf_function"
    use_xla: bool = True
    target_scope: str | None = None
    target_status_trace_policy: str = "none"
    mass_policy: str = "windowed_adaptive"
    metric_update_requirement: str = "allow_valid_incumbent"
    metric_evidence_policy: str = "temporal_information"
    metric_probe_num_results: int = 1
    preparation_max_restarts: int = 0
    engineering_probe_covariance_multiplier: float | None = None
    geometry_scaling_c: float = 0.5
    stability_guard: float = 0.8
    covariance_jitter: float = 1.0e-9
    eigenvalue_floor: float | None = 1.0e-9
    max_condition_number: float | None = None
    allow_geometry_fallback: bool = False
    geometry_position_role: str = "initial_position"
    negative_hessian_source: str = "negative_hessian"
    public_timeout_budget_s: float | None = None
    bootstrap_diagnostic_screen_num_results: int | None = None
    bootstrap_diagnostic_screen_num_burnin_steps: int | None = None
    verification_chunk_max_results: int | None = None
    verification_min_retained_results_for_pass: int | None = None
    staged_timeout_policy: HMCStagedTimeoutPolicy | None = None
    staged_timeout_global_started_perf_counter_s: float | None = None
    staged_timeout_stage_started_perf_counter_s: float | None = None
    staged_timeout_enlargement_rounds: Mapping[str, int] | None = None
    incall_progress_heartbeat_s: float | None = None
    source: str = "bayesfilter.inference.tune_hmc_kernel"

    def __post_init__(self) -> None:
        for name in ("max_leapfrog_steps", "bootstrap_max_repairs", "max_attempts",
                     "terminal_phase6_repair_extra_attempts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        expansion_steps = self.candidate_search_bound_expansion_steps
        if type(expansion_steps) is not int or not 0 <= expansion_steps <= self.max_attempts:
            raise ValueError("candidate_search_bound_expansion_steps must be an integer in [0, max_attempts]")
        if expansion_steps and self.mass_policy != "windowed_adaptive":
            raise ValueError("preparation bound expansion requires windowed_adaptive mass")
        rounds = self.bootstrap_initialization_rounds
        if type(rounds) is not int or rounds < 0:
            raise ValueError("bootstrap_initialization_rounds must be a non-negative integer")
        if rounds and self.mass_policy != "windowed_adaptive":
            raise ValueError("bootstrap initialization requires windowed_adaptive mass")
        algorithm_id = str(self.algorithm_id)
        if algorithm_id != ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID:
            raise ValueError(
                "tune_hmc_kernel exposes one ordinary tuning policy: "
                f"{ORDINARY_BROAD_FIXED_METRIC_ALGORITHM_ID!r}; compatibility "
                "algorithms are available only through lower-level diagnostic APIs"
            )
        object.__setattr__(self, "algorithm_id", algorithm_id)
        budget_policy_id = str(self.operational_budget_policy_id)
        if budget_policy_id != OPERATIONAL_HMC_BUDGET_POLICY_ID:
            raise ValueError("operational_budget_policy_id is fixed by BayesFilter")
        object.__setattr__(
            self,
            "operational_budget_policy_id",
            budget_policy_id,
        )
        evidence_policy = str(self.operational_evidence_policy)
        if evidence_policy not in _OPERATIONAL_EVIDENCE_POLICIES:
            allowed = ", ".join(sorted(_OPERATIONAL_EVIDENCE_POLICIES))
            raise ValueError(
                f"operational_evidence_policy must be one of: {allowed}"
            )
        object.__setattr__(
            self,
            "operational_evidence_policy",
            evidence_policy,
        )
        candidate_handoff_policy = (
            _validated_operational_candidate_handoff_policy(
                self.operational_candidate_handoff_policy
            )
        )
        policy_schema = str(
            self.operational_candidate_handoff_policy_schema_version
        )
        if policy_schema != _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_SCHEMA:
            raise ValueError(
                "operational_candidate_handoff_policy_schema_version is fixed"
            )
        object.__setattr__(
            self,
            "operational_candidate_handoff_policy",
            candidate_handoff_policy,
        )
        object.__setattr__(
            self,
            "operational_candidate_handoff_policy_schema_version",
            policy_schema,
        )
        verification_bracket_policy = (
            _validated_operational_verification_bracket_policy(
                self.operational_verification_bracket_policy
            )
        )
        if verification_bracket_policy != (
            _OPERATIONAL_VERIFICATION_BRACKET_POLICY_SINGLE_REPAIR
        ):
            raise ValueError(
                "tune_hmc_kernel fixes operational_verification_bracket_policy "
                "at 'single_repair'; the verified-log-midpoint policy belongs "
                "only to the lower-level historical shared-epsilon route"
            )
        object.__setattr__(
            self,
            "operational_verification_bracket_policy",
            verification_bracket_policy,
        )
        preset = str(self.preset)
        if preset not in {"smoke", "diagnostic", "diagnostic_plus", "standard", "serious"}:
            raise ValueError(
                "preset must be 'smoke', 'diagnostic', 'diagnostic_plus', "
                "'standard', or 'serious'"
            )
        if (
            evidence_policy != _OPERATIONAL_EVIDENCE_POLICY_INITIAL_ONLY
            and preset != "serious"
        ):
            raise ValueError(
                "non-default operational_evidence_policy requires preset='serious'"
            )
        if candidate_handoff_policy != _OPERATIONAL_CANDIDATE_HANDOFF_POLICY_STRICT:
            if preset != "serious":
                raise ValueError(
                    "non-default operational_candidate_handoff_policy requires "
                    "preset='serious'"
                )
            if evidence_policy != _OPERATIONAL_EVIDENCE_POLICY_ONE_DOUBLING:
                raise ValueError(
                    "mixed operational_candidate_handoff_policy requires "
                    "operational_evidence_policy='one_doubling'"
                )
        object.__setattr__(self, "preset", preset)
        target = float(self.target_accept_prob)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_accept_prob must be finite and in (0, 1)")
        object.__setattr__(self, "target_accept_prob", target)
        acceptance_band = _validate_band(self.acceptance_band, name="acceptance_band")
        repair_band = _validate_band(self.repair_band, name="repair_band")
        if repair_band[0] > acceptance_band[0] or repair_band[1] < acceptance_band[1]:
            raise ValueError("repair_band must contain acceptance_band")
        object.__setattr__(self, "acceptance_band", acceptance_band)
        object.__setattr__(self, "repair_band", repair_band)
        max_leapfrog_steps = _validate_max_leapfrog_steps(
            self.max_leapfrog_steps
        )
        if max_leapfrog_steps != ORDINARY_BROAD_MAX_LEAPFROG_STEPS:
            raise ValueError(
                "tune_hmc_kernel fixes max_leapfrog_steps at "
                f"{ORDINARY_BROAD_MAX_LEAPFROG_STEPS} so the ordinary broad L "
                "grid is not silently truncated"
            )
        object.__setattr__(
            self,
            "max_leapfrog_steps",
            max_leapfrog_steps,
        )
        lower, upper = _validate_trajectory_window_multipliers(
            self.trajectory_window_lower_multiplier,
            self.trajectory_window_upper_multiplier,
        )
        object.__setattr__(
            self,
            "step_repair_factor",
            _validate_step_repair_multiplier(
                self.step_repair_factor,
                name="step_repair_factor",
            ),
        )
        min_directional_factor = _validate_step_repair_multiplier(
            self.step_repair_min_directional_factor,
            name="step_repair_min_directional_factor",
        )
        if min_directional_factor > 2.0:
            raise ValueError("step_repair_min_directional_factor must be at most 2")
        object.__setattr__(
            self,
            "step_repair_min_directional_factor",
            min_directional_factor,
        )
        high_factor = (
            self.step_repair_factor
            if self.step_repair_high_acceptance_directional_factor is None
            else self.step_repair_high_acceptance_directional_factor
        )
        object.__setattr__(
            self,
            "step_repair_high_acceptance_directional_factor",
            _validate_step_repair_multiplier(
                high_factor,
                name="step_repair_high_acceptance_directional_factor",
            ),
        )
        high_ladder_max = (
            high_factor
            if self.step_repair_high_acceptance_ladder_max_factor is None
            else self.step_repair_high_acceptance_ladder_max_factor
        )
        high_ladder_max = _validate_step_repair_multiplier(
            high_ladder_max,
            name="step_repair_high_acceptance_ladder_max_factor",
        )
        if high_ladder_max < high_factor:
            raise ValueError(
                "step_repair_high_acceptance_ladder_max_factor must be at least "
                "step_repair_high_acceptance_directional_factor"
            )
        object.__setattr__(
            self,
            "step_repair_high_acceptance_ladder_max_factor",
            high_ladder_max,
        )
        object.__setattr__(
            self,
            "repair_nonfinite_proposal_screen",
            bool(self.repair_nonfinite_proposal_screen),
        )
        object.__setattr__(self, "trajectory_window_lower_multiplier", lower)
        object.__setattr__(self, "trajectory_window_upper_multiplier", upper)
        object.__setattr__(
            self,
            "handoff_screen_policy",
            _validate_handoff_screen_policy(self.handoff_screen_policy),
        )
        repairs = int(self.bootstrap_max_repairs)
        if repairs < 0:
            raise ValueError("bootstrap_max_repairs must be non-negative")
        object.__setattr__(self, "bootstrap_max_repairs", repairs)
        attempts = int(self.max_attempts)
        if attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if attempts > _PHASE7_MAX_ATTEMPTS_CAP:
            raise ValueError(
                f"max_attempts is hard-capped at {_PHASE7_MAX_ATTEMPTS_CAP}"
            )
        if (
            verification_bracket_policy
            == _OPERATIONAL_VERIFICATION_BRACKET_POLICY_ONE_LOG_MIDPOINT
            and attempts < 3
        ):
            raise ValueError(
                "one_verified_log_midpoint requires at least three attempt slots"
            )
        object.__setattr__(self, "max_attempts", attempts)
        terminal_extra = int(self.terminal_phase6_repair_extra_attempts)
        if terminal_extra < 0 or terminal_extra > 1:
            raise ValueError("terminal_phase6_repair_extra_attempts must be 0 or 1")
        object.__setattr__(
            self,
            "terminal_phase6_repair_extra_attempts",
            terminal_extra,
        )
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
        object.__setattr__(
            self,
            "metric_update_requirement",
            _validate_metric_update_requirement(self.metric_update_requirement),
        )
        engineering_probe_multiplier = (
            _validate_engineering_probe_covariance_multiplier(
                self.engineering_probe_covariance_multiplier
            )
        )
        if engineering_probe_multiplier is not None:
            raise ValueError(
                "engineering_probe_covariance_multiplier is a lower-level P4-E "
                "diagnostic compatibility control, not a public ordinary tuning mode"
            )
        object.__setattr__(
            self,
            "engineering_probe_covariance_multiplier",
            None,
        )
        if (
            self.metric_update_requirement == "require_operational_update"
            and mass_policy == "fixed_identity"
        ):
            raise ValueError(
                "require_operational_update is incompatible with fixed_identity"
            )
        scaling = float(self.geometry_scaling_c)
        if not math.isfinite(scaling) or scaling <= 0.0:
            raise ValueError("geometry_scaling_c must be positive and finite")
        guard = float(self.stability_guard)
        if not math.isfinite(guard) or guard <= 0.0:
            raise ValueError("stability_guard must be positive and finite")
        jitter = float(self.covariance_jitter)
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("covariance_jitter must be finite and non-negative")
        floor = (
            None
            if self.eigenvalue_floor is None
            else float(self.eigenvalue_floor)
        )
        if floor is not None and (not math.isfinite(floor) or floor < 0.0):
            raise ValueError("eigenvalue_floor must be finite and non-negative")
        condition = (
            None
            if self.max_condition_number is None
            else float(self.max_condition_number)
        )
        if condition is not None and (
            not math.isfinite(condition) or condition <= 1.0
        ):
            raise ValueError("max_condition_number must be finite and greater than 1")
        object.__setattr__(self, "geometry_scaling_c", scaling)
        object.__setattr__(self, "stability_guard", guard)
        object.__setattr__(self, "covariance_jitter", jitter)
        object.__setattr__(self, "eigenvalue_floor", floor)
        object.__setattr__(self, "max_condition_number", condition)
        object.__setattr__(self, "allow_geometry_fallback", bool(self.allow_geometry_fallback))
        geometry_position_role = str(self.geometry_position_role)
        if not geometry_position_role:
            raise ValueError("geometry_position_role must be non-empty")
        object.__setattr__(self, "geometry_position_role", geometry_position_role)
        negative_hessian_source = str(self.negative_hessian_source)
        if not negative_hessian_source:
            raise ValueError("negative_hessian_source must be non-empty")
        object.__setattr__(self, "negative_hessian_source", negative_hessian_source)
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
        object.__setattr__(
            self,
            "bootstrap_diagnostic_screen_num_results",
            _validate_positive_int_or_none(
                self.bootstrap_diagnostic_screen_num_results,
                name="bootstrap_diagnostic_screen_num_results",
            ),
        )
        object.__setattr__(
            self,
            "bootstrap_diagnostic_screen_num_burnin_steps",
            _validate_positive_int_or_none(
                self.bootstrap_diagnostic_screen_num_burnin_steps,
                name="bootstrap_diagnostic_screen_num_burnin_steps",
            ),
        )
        object.__setattr__(
            self,
            "verification_chunk_max_results",
            _validate_positive_int_or_none(
                self.verification_chunk_max_results,
                name="verification_chunk_max_results",
            ),
        )
        object.__setattr__(
            self,
            "verification_min_retained_results_for_pass",
            _validate_positive_int_or_none(
                self.verification_min_retained_results_for_pass,
                name="verification_min_retained_results_for_pass",
            ),
        )
        object.__setattr__(
            self,
            "staged_timeout_policy",
            _validate_staged_timeout_policy_or_none(self.staged_timeout_policy),
        )
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
                self.staged_timeout_enlargement_rounds
            ),
        )
        heartbeat = (
            None
            if self.incall_progress_heartbeat_s is None
            else float(self.incall_progress_heartbeat_s)
        )
        if heartbeat is not None and (not math.isfinite(heartbeat) or heartbeat <= 0.0):
            raise ValueError("incall_progress_heartbeat_s must be positive and finite")
        object.__setattr__(self, "incall_progress_heartbeat_s", heartbeat)
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "source", source)

    @classmethod
    def smoke(cls, **overrides: Any) -> "HMCKernelTuningConfig":
        payload: dict[str, Any] = {
            "preset": "smoke",
            "max_attempts": 1,
            "chain_execution_mode": "tf_function",
            "use_xla": False,
            "source": "bayesfilter.inference.tune_hmc_kernel.smoke",
        }
        payload.update(overrides)
        return cls(**payload)

    @classmethod
    def standard(cls, **overrides: Any) -> "HMCKernelTuningConfig":
        payload: dict[str, Any] = {
            "preset": "standard",
            "max_attempts": 3,
            "source": "bayesfilter.inference.tune_hmc_kernel.standard",
        }
        payload.update(overrides)
        return cls(**payload)

    @classmethod
    def diagnostic(cls, **overrides: Any) -> "HMCKernelTuningConfig":
        payload: dict[str, Any] = {
            "preset": "diagnostic",
            "max_attempts": 2,
            "source": "bayesfilter.inference.tune_hmc_kernel.diagnostic",
        }
        payload.update(overrides)
        return cls(**payload)

    @classmethod
    def diagnostic_plus(cls, **overrides: Any) -> "HMCKernelTuningConfig":
        payload: dict[str, Any] = {
            "preset": "diagnostic_plus",
            "max_attempts": 2,
            "source": "bayesfilter.inference.tune_hmc_kernel.diagnostic_plus",
        }
        payload.update(overrides)
        return cls(**payload)

    @classmethod
    def serious(cls, **overrides: Any) -> "HMCKernelTuningConfig":
        payload: dict[str, Any] = {
            "preset": "serious",
            "max_attempts": 5,
            "source": "bayesfilter.inference.tune_hmc_kernel.serious",
        }
        payload.update(overrides)
        return cls(**payload)

    @property
    def is_smoke(self) -> bool:
        return self.preset == "smoke"

    @property
    def uses_serious_budget_policy(self) -> bool:
        return self.preset == "serious"

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_kernel_tuning_config.v1",
            "algorithm_id": self.algorithm_id,
            "ordinary_selection_policy": _ordinary_selection_policy_payload(self),
            "runtime_backend_policy": _ORDINARY_RUNTIME_BACKEND_POLICY_ID,
            "operational_budget_policy_id": self.operational_budget_policy_id,
            "operational_evidence_policy": self.operational_evidence_policy,
            "operational_candidate_handoff_policy": (
                self.operational_candidate_handoff_policy
            ),
            "operational_candidate_handoff_policy_schema_version": (
                self.operational_candidate_handoff_policy_schema_version
            ),
            "operational_candidate_handoff_policy_contract": (
                candidate_handoff_policy_payload(
                    self.operational_candidate_handoff_policy
                )
            ),
            "operational_verification_bracket_policy": (
                self.operational_verification_bracket_policy
            ),
            "preset": self.preset,
            "target_accept_prob": self.target_accept_prob,
            "acceptance_band": self.acceptance_band,
            "repair_band": self.repair_band,
            "max_leapfrog_steps": self.max_leapfrog_steps,
            "step_repair_factor": self.step_repair_factor,
            "step_repair_min_directional_factor": (
                self.step_repair_min_directional_factor
            ),
            "step_repair_high_acceptance_directional_factor": (
                self.step_repair_high_acceptance_directional_factor
            ),
            "step_repair_high_acceptance_ladder_max_factor": (
                self.step_repair_high_acceptance_ladder_max_factor
            ),
            "repair_nonfinite_proposal_screen": self.repair_nonfinite_proposal_screen,
            "trajectory_window_lower_multiplier": self.trajectory_window_lower_multiplier,
            "trajectory_window_upper_multiplier": self.trajectory_window_upper_multiplier,
            "handoff_screen_policy": self.handoff_screen_policy,
            "bootstrap_max_repairs": self.bootstrap_max_repairs,
            "bootstrap_initialization_rounds": self.bootstrap_initialization_rounds,
            "max_attempts": self.max_attempts,
            "candidate_search_bound_expansion_steps": self.candidate_search_bound_expansion_steps,
            "terminal_phase6_repair_extra_attempts": (
                self.terminal_phase6_repair_extra_attempts
            ),
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
            "geometry_scaling_c": self.geometry_scaling_c,
            "stability_guard": self.stability_guard,
            "covariance_jitter": self.covariance_jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "allow_geometry_fallback": self.allow_geometry_fallback,
            "geometry_position_role": self.geometry_position_role,
            "negative_hessian_source": self.negative_hessian_source,
            "public_timeout_budget_s": self.public_timeout_budget_s,
            "bootstrap_diagnostic_screen_num_results": (
                self.bootstrap_diagnostic_screen_num_results
            ),
            "bootstrap_diagnostic_screen_num_burnin_steps": (
                self.bootstrap_diagnostic_screen_num_burnin_steps
            ),
            "bootstrap_diagnostic_sizing_claim": (
                "public observability diagnostic only; not sampler promotion"
            ),
            "verification_chunk_max_results": self.verification_chunk_max_results,
            "verification_min_retained_results_for_pass": (
                self.verification_min_retained_results_for_pass
            ),
            "verification_compile_sizing_claim": (
                "XLA compile-shape policy only; not a reduced verification evidence gate"
            ),
            "geometry_scaled_budget_timing_policy": (
                _geometry_scaled_budget_timing_policy().payload()
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
            "incall_progress_heartbeat_s": self.incall_progress_heartbeat_s,
            "source": self.source,
            "preset_role": _public_tuning_preset_role(self.preset),
            "hmc_mechanics_owned_by_bayesfilter": True,
            "forbidden_public_fields": _public_tuning_forbidden_fields(),
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_default_readiness": False,
            "reports_gpu_or_xla_readiness": False,
        }


def _public_tuning_forbidden_fields() -> tuple[str, ...]:
    return (
        "step_size",
        "initial_step_size",
        "num_leapfrog_steps",
        "leapfrog_count",
        "min_leapfrog",
        "max_leapfrog",
        "candidate_l_grid",
        "trajectory_grid",
        "mass_window_schedule",
        "warmup_budget_schedule",
        "tuning_budget_schedule",
        "tune_num_results",
        "screen_num_results",
        "verification_num_results",
        "verification_num_burnin_steps",
        "operational_evidence_extension_checkpoints",
    )


def _public_tuning_preset_role(preset: str) -> str:
    if preset == "smoke":
        return "contract_scale_only"
    if preset == "diagnostic":
        return "bounded_public_timeout_diagnostic_only"
    if preset == "diagnostic_plus":
        return "bounded_public_verification_diagnostic_only"
    if preset == "standard":
        return "moderate_local_diagnostic_only"
    if preset == "serious":
        return "dimension_scaled_tuning_policy_only"
    raise ValueError("unknown HMC kernel tuning preset")


def _public_geometry_config(config: HMCKernelTuningConfig) -> HMCGeometryInitializationConfig:
    return HMCGeometryInitializationConfig(
        geometry_scaling_c=config.geometry_scaling_c,
        stability_guard=config.stability_guard,
        covariance_jitter=config.covariance_jitter,
        eigenvalue_floor=config.eigenvalue_floor,
        max_condition_number=config.max_condition_number,
        max_leapfrog_steps=config.max_leapfrog_steps,
        allow_geometry_fallback=config.allow_geometry_fallback,
        position_role=config.geometry_position_role,
        negative_hessian_source=config.negative_hessian_source,
        mass_policy=config.mass_policy,
        seed=_derive_seed(config.seed, stage_index=2),
        source=f"{config.source}.geometry",
    )


def _public_bootstrap_config(
    config: HMCKernelTuningConfig,
    geometry: HMCGeometryInitializationResult | None = None,
) -> HMCBootstrapScreenConfig:
    if config.preset == "smoke":
        screen_num_results = 4
        screen_num_burnin_steps = 1
    elif config.preset == "standard":
        screen_num_results = 16
        screen_num_burnin_steps = 4
    elif config.preset == "serious":
        counts = _geometry_scaled_budget_timing_policy().bootstrap_screen_counts(
            target_dimension=1 if geometry is None else geometry.target_dimension,
            mass_artifact=None if geometry is None else geometry.mass_artifact,
        )
        screen_num_results = int(counts["screen_num_results"])
        screen_num_burnin_steps = int(counts["screen_num_burnin_steps"])
    else:
        screen_num_results = 32
        screen_num_burnin_steps = 8
    if config.bootstrap_diagnostic_screen_num_results is not None:
        screen_num_results = config.bootstrap_diagnostic_screen_num_results
    if config.bootstrap_diagnostic_screen_num_burnin_steps is not None:
        screen_num_burnin_steps = config.bootstrap_diagnostic_screen_num_burnin_steps
    return HMCBootstrapScreenConfig(
        target_accept_prob=config.target_accept_prob,
        acceptance_band=config.acceptance_band,
        repair_band=config.repair_band,
        step_repair_factor=config.step_repair_factor,
        max_leapfrog_steps=config.max_leapfrog_steps,
        max_repairs=config.bootstrap_max_repairs,
        acceptance_role=("warmup_startup_only" if config.bootstrap_initialization_rounds
                         else "fixed_kernel_screen"),
        screen_num_results=screen_num_results,
        screen_num_burnin_steps=screen_num_burnin_steps,
        seed=_derive_seed(config.seed, stage_index=3),
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        source=f"{config.source}.bootstrap",
    )


def _public_loop_config(
    config: HMCKernelTuningConfig,
    *,
    public_timeout_started_perf_counter_s: float | None = None,
) -> HMCTuneVerifyRepairLoopConfig:
    return HMCTuneVerifyRepairLoopConfig(
        algorithm_id=config.algorithm_id,
        operational_budget_policy_id=config.operational_budget_policy_id,
        operational_candidate_handoff_policy=(
            config.operational_candidate_handoff_policy
        ),
        operational_verification_bracket_policy=(
            config.operational_verification_bracket_policy
        ),
        target_accept_prob=config.target_accept_prob,
        acceptance_band=config.acceptance_band,
        repair_band=config.repair_band,
        max_leapfrog_steps=config.max_leapfrog_steps,
        step_repair_factor=config.step_repair_factor,
        step_repair_min_directional_factor=config.step_repair_min_directional_factor,
        step_repair_high_acceptance_directional_factor=(
            config.step_repair_high_acceptance_directional_factor
        ),
        step_repair_high_acceptance_ladder_max_factor=(
            config.step_repair_high_acceptance_ladder_max_factor
        ),
        repair_nonfinite_proposal_screen=config.repair_nonfinite_proposal_screen,
        trajectory_window_lower_multiplier=config.trajectory_window_lower_multiplier,
        trajectory_window_upper_multiplier=config.trajectory_window_upper_multiplier,
        handoff_screen_policy=config.handoff_screen_policy,
        max_attempts=config.max_attempts,
        terminal_phase6_repair_extra_attempts=(
            config.terminal_phase6_repair_extra_attempts
        ),
        seed=_derive_seed(config.seed, stage_index=7),
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        mass_policy=config.mass_policy,
        metric_update_requirement=config.metric_update_requirement,
        metric_evidence_policy=config.metric_evidence_policy,
        metric_probe_num_results=config.metric_probe_num_results,
        preparation_max_restarts=config.preparation_max_restarts,
        engineering_probe_covariance_multiplier=(
            config.engineering_probe_covariance_multiplier
        ),
        public_timeout_budget_s=config.public_timeout_budget_s,
        public_timeout_started_perf_counter_s=public_timeout_started_perf_counter_s,
        verification_chunk_max_results=config.verification_chunk_max_results,
        verification_min_retained_results_for_pass=(
            config.verification_min_retained_results_for_pass
        ),
        staged_timeout_policy=config.staged_timeout_policy,
        staged_timeout_global_started_perf_counter_s=(
            config.staged_timeout_global_started_perf_counter_s
            if config.staged_timeout_global_started_perf_counter_s is not None
            else public_timeout_started_perf_counter_s
        ),
        staged_timeout_stage_started_perf_counter_s=(
            config.staged_timeout_stage_started_perf_counter_s
        ),
        staged_timeout_enlargement_rounds=config.staged_timeout_enlargement_rounds,
        incall_progress_heartbeat_s=config.incall_progress_heartbeat_s,
        source=f"{config.source}.tune_verify_repair_loop",
    )


def _public_budget_policy_factory(
    config: HMCKernelTuningConfig,
    geometry: HMCGeometryInitializationResult | None = None,
) -> Callable[[int, int], _HMCAttemptBudgetPolicy] | None:
    if config.preset == "serious":
        central = _geometry_scaled_budget_timing_policy()
        initial_operational = HMCOperationalStatisticalWorkPolicy(
            policy_id=config.operational_budget_policy_id,
        )
        evidence_checkpoints = (
            ()
            if config.operational_evidence_policy
            == _OPERATIONAL_EVIDENCE_POLICY_INITIAL_ONLY
            else (2 * initial_operational.initial_candidate_results,)
        )
        operational = HMCOperationalStatisticalWorkPolicy(
            initial_candidate_results=initial_operational.initial_candidate_results,
            candidate_burnin_steps=initial_operational.candidate_burnin_steps,
            evidence_extension_checkpoints=evidence_checkpoints,
            exact_l_tune_adaptation_steps=(
                initial_operational.exact_l_tune_adaptation_steps
            ),
            fresh_verification_results=initial_operational.fresh_verification_results,
            fresh_verification_burnin_steps=(
                initial_operational.fresh_verification_burnin_steps
            ),
            candidate_count_upper_bound=(
                initial_operational.candidate_count_upper_bound
            ),
            replications_per_candidate=initial_operational.replications_per_candidate,
            exact_l_tune_result_steps=initial_operational.exact_l_tune_result_steps,
            fresh_verification_starts_per_outer_attempt=(
                _operational_verification_starts_per_outer_attempt(
                    config.operational_verification_bracket_policy
                )
            ),
            chain_count=initial_operational.chain_count,
            policy_id=config.operational_budget_policy_id,
        )

        def serious_factory(
            target_dimension: int,
            attempt_index: int,
        ) -> _HMCAttemptBudgetPolicy:
            policy = _default_attempt_budget_policy(
                target_dimension,
                attempt_index,
                mass_artifact=None if geometry is None else geometry.mass_artifact,
                policy=central,
                operational_policy=operational,
            )
            if policy.operational_budget_policy_hash != operational.policy_hash:
                raise ValueError("serious operational budget policy hash mismatch")
            return policy

        return serious_factory

    def factory(target_dimension: int, attempt_index: int) -> _HMCAttemptBudgetPolicy:
        dimension = int(target_dimension)
        if dimension <= 0:
            raise ValueError("target_dimension must be positive")
        index = int(attempt_index)
        if index < 0:
            raise ValueError("attempt_index must be non-negative")
        if config.preset == "smoke":
            base = min(max(8, 4 * dimension), 64)
            warmup = max(12, base)
            screen_floor = 4
            verification_floor = 4
            burnin_floor = 1
            budget_class = "smoke_contract"
            budget_cap = 64
        elif config.preset == "diagnostic":
            budget_cap = 64
            base = min(max(16, 8 * dimension), 32)
            warmup = max(16, base)
            screen_floor = 8
            verification_floor = 8
            burnin_floor = 2
            budget_class = "bounded_public_diagnostic"
        elif config.preset == "diagnostic_plus":
            budget_cap = 256
            base = min(max(64, 32 * dimension), 128)
            warmup = max(64, base)
            screen_floor = 16
            verification_floor = 32
            burnin_floor = 4
            budget_class = "bounded_public_diagnostic_plus"
        else:
            base = max(128, 25 * dimension)
            warmup = base
            screen_floor = 16
            verification_floor = 32
            burnin_floor = 4
            budget_class = "standard_public_diagnostic"
            budget_cap = None
        budget = int(base * (2 ** index))
        if budget_cap is not None:
            budget = min(budget, budget_cap)
        phase5_screen = max(screen_floor, _ceil_div(budget, 4))
        phase6_screen = max(screen_floor, _ceil_div(budget, 4))
        if (
            config.preset == "standard"
            and int(config.terminal_phase6_repair_extra_attempts) > 0
            and index >= int(config.max_attempts)
        ):
            phase6_screen = min(
                phase6_screen,
                _PUBLIC_TERMINAL_PHASE6_REPAIR_SCREEN_MAX_RESULTS,
            )
        verification_results = max(verification_floor, _ceil_div(budget, 2))
        # The operational candidate selector uses the reviewed R0-R8
        # acceptance policy: four chains, four blocks, and at least sixteen
        # decisions per block.  Keep the small public diagnostic budgets above
        # as mechanics-only evidence, but never let their derived operational
        # fields produce an under-sized acceptance screen.
        operational_screen_results = max(64, phase5_screen)
        operational_screen_burnin = max(16, _ceil_div(operational_screen_results, 4))
        operational_tune_steps = max(64, budget)
        operational_verification_results = max(64, verification_results)
        operational_verification_burnin = max(
            16,
            _ceil_div(operational_verification_results, 4),
        )
        return _HMCAttemptBudgetPolicy(
            target_dimension=dimension,
            attempt_index=index,
            budget=budget,
            phase4_warmup_steps=warmup,
            phase5_tune_budgets=(
                max(2, _ceil_div(budget, 4)),
                max(4, _ceil_div(budget, 2)),
                max(8, budget),
            ),
            phase5_screen_num_results=phase5_screen,
            phase5_screen_burnin_steps=max(burnin_floor, _ceil_div(phase5_screen, 4)),
            phase6_screen_num_results=phase6_screen,
            phase6_screen_burnin_steps=max(burnin_floor, _ceil_div(phase6_screen, 4)),
            verification_num_results=verification_results,
            verification_num_burnin_steps=max(
                burnin_floor,
                _ceil_div(verification_results, 4),
            ),
            operational_screen_num_results=operational_screen_results,
            operational_screen_num_burnin_steps=operational_screen_burnin,
            operational_exact_l_tune_adaptation_steps=operational_tune_steps,
            operational_verification_num_results=operational_verification_results,
            operational_verification_num_burnin_steps=operational_verification_burnin,
            operational_budget_policy_id=config.operational_budget_policy_id,
            serious_policy=False,
            public_budget_class=budget_class,
            public_budget_cap=budget_cap,
            public_max_attempts=config.max_attempts,
            public_diagnostic_preset=config.preset,
            budget_formula=(
                "bounded public diagnostic budget: budget_k=min(cap, "
                "base(dimension,preset)*2**attempt_index)"
            ),
            budget_formula_parameters={
                "preset": config.preset,
                "base": base,
                "budget_cap": budget_cap,
                "screen_floor": screen_floor,
                "verification_floor": verification_floor,
                "burnin_floor": burnin_floor,
                "operational_acceptance_screen_floor": 64,
                "operational_acceptance_burnin_floor": 16,
                "non_promoting_diagnostic": True,
            },
            geometry_budget_summary=_geometry_scaled_budget_timing_policy().geometry_summary(
                target_dimension=dimension
            ),
            budget0_uncapped=base,
            budget0_after_floor_and_cap=(
                min(base, budget_cap) if budget_cap is not None else base
            ),
            budget_claim=(
                f"{config.preset} public diagnostic budget only; not posterior "
                "convergence or sampler-validity evidence"
            ),
        )

    return factory


@dataclass(frozen=True)
class _HMCAttemptBudgetPolicy:
    target_dimension: int
    attempt_index: int
    budget: int
    phase4_warmup_steps: int
    phase5_tune_budgets: tuple[int, int, int]
    phase5_screen_num_results: int
    phase5_screen_burnin_steps: int
    phase6_screen_num_results: int
    phase6_screen_burnin_steps: int
    verification_num_results: int
    verification_num_burnin_steps: int
    operational_screen_num_results: int | None = None
    operational_screen_num_burnin_steps: int | None = None
    operational_evidence_extension_checkpoints: tuple[int, ...] = ()
    operational_exact_l_tune_adaptation_steps: int | None = None
    operational_verification_num_results: int | None = None
    operational_verification_num_burnin_steps: int | None = None
    operational_verification_starts_per_outer_attempt: int = 2
    operational_budget_policy_id: str | None = None
    operational_budget_policy_hash: str | None = None
    serious_policy: bool = True
    public_budget_class: str | None = None
    public_budget_cap: int | None = None
    public_max_attempts: int | None = None
    public_diagnostic_preset: str | None = None
    budget_formula: str | None = None
    budget_formula_parameters: Mapping[str, Any] | None = None
    geometry_budget_summary: Mapping[str, Any] | None = None
    budget0_uncapped: int | None = None
    budget0_after_floor_and_cap: int | None = None
    budget_claim: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "target_dimension",
            "attempt_index",
            "budget",
            "phase4_warmup_steps",
            "phase5_screen_num_results",
            "phase5_screen_burnin_steps",
            "phase6_screen_num_results",
            "phase6_screen_burnin_steps",
            "verification_num_results",
            "verification_num_burnin_steps",
        ):
            value = int(getattr(self, name))
            if name == "attempt_index":
                if value < 0:
                    raise ValueError("attempt_index must be non-negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        budgets = tuple(int(item) for item in self.phase5_tune_budgets)
        if len(budgets) != 3 or any(item <= 0 for item in budgets):
            raise ValueError("phase5_tune_budgets must contain three positive values")
        object.__setattr__(self, "phase5_tune_budgets", budgets)
        operational_fields = {
            "operational_screen_num_results": self.phase5_screen_num_results,
            "operational_screen_num_burnin_steps": self.phase5_screen_burnin_steps,
            "operational_exact_l_tune_adaptation_steps": budgets[-1],
            "operational_verification_num_results": self.verification_num_results,
            "operational_verification_num_burnin_steps": (
                self.verification_num_burnin_steps
            ),
        }
        for name, fallback in operational_fields.items():
            raw_value = getattr(self, name)
            value = int(fallback if raw_value is None else raw_value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        checkpoints = tuple(
            int(item) for item in self.operational_evidence_extension_checkpoints
        )
        if any(item <= self.operational_screen_num_results for item in checkpoints):
            raise ValueError(
                "operational evidence extension checkpoints must exceed the screen"
            )
        if any(left >= right for left, right in zip(checkpoints, checkpoints[1:])):
            raise ValueError(
                "operational evidence extension checkpoints must strictly increase"
            )
        if len(checkpoints) > 2:
            raise ValueError("operational evidence extensions are capped at two")
        object.__setattr__(
            self,
            "operational_evidence_extension_checkpoints",
            checkpoints,
        )
        verification_starts = int(
            self.operational_verification_starts_per_outer_attempt
        )
        if verification_starts not in {2, 3}:
            raise ValueError(
                "operational_verification_starts_per_outer_attempt must be 2 or 3"
            )
        object.__setattr__(
            self,
            "operational_verification_starts_per_outer_attempt",
            verification_starts,
        )
        policy_id = (
            OPERATIONAL_HMC_BUDGET_POLICY_ID
            if self.operational_budget_policy_id is None
            else str(self.operational_budget_policy_id)
        )
        if not policy_id:
            raise ValueError("operational_budget_policy_id must be non-empty")
        object.__setattr__(self, "operational_budget_policy_id", policy_id)
        expected_policy_hash = HMCOperationalStatisticalWorkPolicy(
            initial_candidate_results=self.operational_screen_num_results,
            candidate_burnin_steps=self.operational_screen_num_burnin_steps,
            evidence_extension_checkpoints=checkpoints,
            exact_l_tune_adaptation_steps=(
                self.operational_exact_l_tune_adaptation_steps
            ),
            fresh_verification_results=(
                self.operational_verification_num_results
            ),
            fresh_verification_burnin_steps=(
                self.operational_verification_num_burnin_steps
            ),
            fresh_verification_starts_per_outer_attempt=verification_starts,
            policy_id=policy_id,
        ).policy_hash
        supplied_policy_hash = (
            None
            if self.operational_budget_policy_hash is None
            else str(self.operational_budget_policy_hash)
        )
        if supplied_policy_hash is not None and supplied_policy_hash != expected_policy_hash:
            raise ValueError("operational_budget_policy_hash does not match its fields")
        object.__setattr__(
            self,
            "operational_budget_policy_hash",
            expected_policy_hash,
        )
        object.__setattr__(self, "serious_policy", bool(self.serious_policy))
        budget_class = (
            None
            if self.public_budget_class is None
            else str(self.public_budget_class)
        )
        object.__setattr__(self, "public_budget_class", budget_class)
        budget_cap = (
            None
            if self.public_budget_cap is None
            else int(self.public_budget_cap)
        )
        if budget_cap is not None and budget_cap <= 0:
            raise ValueError("public_budget_cap must be positive when provided")
        object.__setattr__(self, "public_budget_cap", budget_cap)
        max_attempts = (
            None
            if self.public_max_attempts is None
            else int(self.public_max_attempts)
        )
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("public_max_attempts must be positive when provided")
        object.__setattr__(self, "public_max_attempts", max_attempts)
        preset = (
            None
            if self.public_diagnostic_preset is None
            else str(self.public_diagnostic_preset)
        )
        object.__setattr__(self, "public_diagnostic_preset", preset)
        budget_formula = (
            "budget0=clamp(ceil(dimension_factor*d*geometry_multiplier), "
            "min_initial_budget, max_initial_budget); "
            "budget_k=min(max_tune_budget, budget0*2**attempt_index)"
            if self.budget_formula is None
            else str(self.budget_formula)
        )
        if not budget_formula:
            raise ValueError("budget_formula must be non-empty")
        object.__setattr__(self, "budget_formula", budget_formula)
        formula_parameters = (
            _geometry_scaled_budget_timing_policy().budget_formula_parameters()
            if self.budget_formula_parameters is None
            else dict(self.budget_formula_parameters)
        )
        object.__setattr__(
            self,
            "budget_formula_parameters",
            formula_parameters,
        )
        geometry_summary = (
            _geometry_scaled_budget_timing_policy().geometry_summary(
                target_dimension=self.target_dimension
            )
            if self.geometry_budget_summary is None
            else dict(self.geometry_budget_summary)
        )
        object.__setattr__(self, "geometry_budget_summary", geometry_summary)
        uncapped = (
            int(
                math.ceil(
                    float(formula_parameters.get("dimension_factor", 20.0))
                    * float(self.target_dimension)
                    * float(geometry_summary.get("geometry_multiplier", 1.0))
                )
            )
            if self.budget0_uncapped is None
            else int(self.budget0_uncapped)
        )
        if uncapped <= 0:
            raise ValueError("budget0_uncapped must be positive")
        object.__setattr__(self, "budget0_uncapped", uncapped)
        after_cap = self.budget if self.budget0_after_floor_and_cap is None else int(
            self.budget0_after_floor_and_cap
        )
        if after_cap <= 0:
            raise ValueError("budget0_after_floor_and_cap must be positive")
        object.__setattr__(self, "budget0_after_floor_and_cap", after_cap)
        budget_claim = (
            "dimension/geometry-scaled tuning work budget; not posterior "
            "convergence or sampler-validity evidence"
            if self.budget_claim is None
            else str(self.budget_claim)
        )
        if not budget_claim:
            raise ValueError("budget_claim must be non-empty")
        object.__setattr__(self, "budget_claim", budget_claim)

    def payload(self) -> Mapping[str, Any]:
        return {
            "target_dimension": self.target_dimension,
            "attempt_index": self.attempt_index,
            "budget": self.budget,
            "phase4_warmup_steps": self.phase4_warmup_steps,
            "phase5_tune_budgets": self.phase5_tune_budgets,
            "phase5_screen_num_results": self.phase5_screen_num_results,
            "phase5_screen_burnin_steps": self.phase5_screen_burnin_steps,
            "phase6_screen_num_results": self.phase6_screen_num_results,
            "phase6_screen_burnin_steps": self.phase6_screen_burnin_steps,
            "verification_num_results": self.verification_num_results,
            "verification_num_burnin_steps": self.verification_num_burnin_steps,
            "operational_screen_num_results": (
                self.operational_screen_num_results
            ),
            "operational_screen_num_burnin_steps": (
                self.operational_screen_num_burnin_steps
            ),
            "operational_evidence_extension_checkpoints": (
                self.operational_evidence_extension_checkpoints
            ),
            "operational_exact_l_tune_adaptation_steps": (
                self.operational_exact_l_tune_adaptation_steps
            ),
            "operational_verification_num_results": (
                self.operational_verification_num_results
            ),
            "operational_verification_num_burnin_steps": (
                self.operational_verification_num_burnin_steps
            ),
            "operational_verification_starts_per_outer_attempt": (
                self.operational_verification_starts_per_outer_attempt
            ),
            "operational_budget_policy_id": self.operational_budget_policy_id,
            "operational_budget_policy_hash": self.operational_budget_policy_hash,
            "budget_formula": self.budget_formula,
            "budget_formula_parameters": dict(self.budget_formula_parameters),
            "geometry_budget_summary": dict(self.geometry_budget_summary),
            "budget0_uncapped": self.budget0_uncapped,
            "budget0_after_floor_and_cap": self.budget0_after_floor_and_cap,
            "budget_claim": self.budget_claim,
            "subbudget_formula": {
                "phase4_warmup_steps": "budget_k",
                "phase5_tune_budgets": "(ceil(budget_k/4), ceil(budget_k/2), budget_k)",
                "phase5_screen": "results=max(32, ceil(budget_k/4)); burnin=max(8, ceil(results/4))",
                "phase6_screen": "results=max(32, ceil(budget_k/4)); burnin=max(8, ceil(results/4))",
                "verification": "results=max(64, ceil(budget_k/2)); burnin=max(16, ceil(results/4))",
                "operational": (
                    "independent named policy; no field is derived from budget_k"
                ),
            },
            "serious_policy": self.serious_policy,
            "public_budget_class": self.public_budget_class,
            "public_budget_cap": self.public_budget_cap,
            "public_max_attempts": self.public_max_attempts,
            "public_diagnostic_preset": self.public_diagnostic_preset,
            "diagnostic_role": (
                "public_bounded_timeout_diagnostic"
                if self.public_diagnostic_preset == "diagnostic"
                else (
                    "public_bounded_verification_diagnostic_plus"
                    if self.public_diagnostic_preset == "diagnostic_plus"
                    else "public_kernel_tuning_budget"
                )
            ),
            "nonclaims": TUNE_VERIFY_REPAIR_LOOP_NONCLAIMS,
            "internal_policy_only": True,
        }


def _default_attempt_budget_policy(
    target_dimension: int,
    attempt_index: int,
    mass_artifact: PrecomputedMassArtifact | None = None,
    policy: HMCGeometryScaledBudgetTimingPolicy | None = None,
    operational_policy: HMCOperationalStatisticalWorkPolicy | None = None,
) -> _HMCAttemptBudgetPolicy:
    central = _geometry_scaled_budget_timing_policy() if policy is None else policy
    payload = dict(
        central.attempt_budget_payload(
            target_dimension=int(target_dimension),
            attempt_index=int(attempt_index),
            mass_artifact=mass_artifact,
        )
    )
    if operational_policy is not None:
        payload.update(
            {
                "operational_screen_num_results": (
                    operational_policy.initial_candidate_results
                ),
                "operational_screen_num_burnin_steps": (
                    operational_policy.candidate_burnin_steps
                ),
                "operational_evidence_extension_checkpoints": (
                    operational_policy.evidence_extension_checkpoints
                ),
                "operational_exact_l_tune_adaptation_steps": (
                    operational_policy.exact_l_tune_adaptation_steps
                ),
                "operational_verification_num_results": (
                    operational_policy.fresh_verification_results
                ),
                "operational_verification_num_burnin_steps": (
                    operational_policy.fresh_verification_burnin_steps
                ),
                "operational_verification_starts_per_outer_attempt": (
                    operational_policy.fresh_verification_starts_per_outer_attempt
                ),
                "operational_budget_policy_id": operational_policy.policy_id,
                "operational_budget_policy_hash": operational_policy.policy_hash,
            }
        )
    return _attempt_budget_policy_from_payload(
        payload,
        serious_policy=True,
    )


def _phase7_attempt_seed(root_seed: tuple[int, int], attempt_index: int) -> tuple[int, int]:
    return _derive_seed(root_seed, stage_index=10 + int(attempt_index))


def _staged_timeout_stage_budget(
    policy: HMCStagedTimeoutPolicy | None,
    stage: str,
    fallback: float | None,
) -> float | None:
    if policy is None or not policy.enabled:
        return fallback
    return float(policy.stage_budgets_s[str(stage)])


def _phase7_windowed_stage_config(
    config: HMCTuneVerifyRepairLoopConfig,
    *,
    attempt_index: int,
) -> HMCWindowedMassStageConfig:
    return HMCWindowedMassStageConfig(
        algorithm_id=windowed_algorithm_for_selection_algorithm(config.algorithm_id),
        target_accept_prob=config.target_accept_prob,
        seed=_derive_seed(_phase7_attempt_seed(config.seed, attempt_index), stage_index=0),
        chain_execution_mode=config.chain_execution_mode,
        use_xla=config.use_xla,
        target_scope=config.target_scope,
        target_status_trace_policy=config.target_status_trace_policy,
        mass_policy=config.mass_policy,
        metric_update_requirement=config.metric_update_requirement,
        metric_evidence_policy=config.metric_evidence_policy,
        metric_probe_num_results=config.metric_probe_num_results,
        preparation_max_restarts=config.preparation_max_restarts,
        engineering_probe_covariance_multiplier=(
            config.engineering_probe_covariance_multiplier
        ),
        public_timeout_budget_s=_staged_timeout_stage_budget(
            config.staged_timeout_policy,
            "windowed_mass",
            config.public_timeout_budget_s,
        ),
        public_timeout_started_perf_counter_s=None
        if config.staged_timeout_policy is not None
        else config.public_timeout_started_perf_counter_s,
        staged_timeout_policy=config.staged_timeout_policy,
        staged_timeout_global_started_perf_counter_s=(
            config.staged_timeout_global_started_perf_counter_s
        ),
        staged_timeout_stage_started_perf_counter_s=(
            None
            if config.staged_timeout_policy is None
            else time.perf_counter()
        ),
        staged_timeout_enlargement_rounds=config.staged_timeout_enlargement_rounds,
        source=config.source,
    )


def _fixed_mass_step_upper_bound(windowed_stage: HMCWindowedMassStageResult) -> float | None:
    """Recover the qualified bound in the final, not historical, metric."""
    operational = windowed_stage.operational_warmup_result
    if operational is None:
        return None
    if not operational.windows:
        raise ValueError("qualified step bound requires a final warmup window")
    window = operational.windows[-1]
    final = operational.final_kernel_state
    metric_changed = (
        window.metric_decision is not None and window.metric_decision.update_applied
    )
    coordinate_signature = (
        window.next_coordinate_signature if metric_changed
        else window.coordinate_signature_used
    )
    metric_signature = (
        window.next_metric_signature if metric_changed else window.metric_signature_used
    )
    if (
        coordinate_signature != final.transform.signature
        or metric_signature != final.momentum_metric.signature
    ):
        raise ValueError("qualified step bound has stale final metric coordinates")
    if metric_changed:
        qualification = window.next_reasonable_epsilon
        if qualification is None or not qualification.passed:
            raise ValueError("changed metric requires a fresh qualified step bound")
        bound = qualification.selected_step_size
    else:
        bound = window.step_size_upper_bound
    if bound is None or not 0.0 < float(bound) < float("inf"):
        raise ValueError("qualified step bound must be finite and positive")
    if final.epsilon is None or not 0.0 < float(final.epsilon) <= float(bound):
        raise ValueError("final epsilon exceeds its qualified step bound")
    return float(bound)


__all__ = [
    "HMCKernelTuningConfig",
    "HMCTuneVerifyRepairLoopConfig",
    "HMCGeometryScaledBudgetTimingPolicy",
    "resolve_ordinary_hmc_selection_policy",
    "ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID",
    "ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID",
    "ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID",
]
