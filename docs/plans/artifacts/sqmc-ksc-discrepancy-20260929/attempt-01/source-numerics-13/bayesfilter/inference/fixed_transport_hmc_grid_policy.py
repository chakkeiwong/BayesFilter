"""Generic fixed-transport HMC candidate-grid policy contracts.

This module owns the BayesFilter side of fixed-NeuTra HMC candidate generation.
Model-specific repositories should consume the policy-spec payload and execute
the candidates; they should not rebuild generic HMC step/leapfrog grids.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bayesfilter.inference.generic_hmc_tuning import (
    GenericHMCFixedGridScaleSelection,
    select_hmc_fixed_grid_scale,
)
from bayesfilter.runtime import stable_config_hash


FIXED_TRANSPORT_HMC_GRID_POLICY_NONCLAIMS: tuple[str, ...] = (
    "generic fixed-transport HMC candidate-grid policy only",
    "does not run HMC",
    "does not adapt a trained transport",
    "does not claim posterior convergence",
    "does not claim sampler superiority",
    "does not claim model-specific scientific validity",
)

FIXED_TRANSPORT_HMC_PREPARED_GRID_NONCLAIMS: tuple[str, ...] = (
    "fixed-transport HMC grid preparation only",
    "requires real pilot acceptance diagnostics",
    "does not train or adapt a transport",
    "does not run the final HMC candidate grid",
    "does not claim posterior convergence",
    "does not claim sampler superiority",
    "does not claim model-specific scientific validity",
)

FIXED_TRANSPORT_HMC_JOINT_PREPARED_GRID_NONCLAIMS: tuple[str, ...] = (
    "fixed-transport HMC joint-grid preparation only",
    "requires real joint pilot rows over step size and leapfrog count",
    "selected scale is descriptive only when present",
    "does not train or adapt a transport",
    "does not run the final HMC candidate grid",
    "does not claim posterior convergence",
    "does not claim sampler superiority",
    "does not claim model-specific scientific validity",
)

FIXED_TRANSPORT_HMC_ADAPTIVE_JOINT_PREPARED_GRID_NONCLAIMS: tuple[str, ...] = (
    "fixed-transport HMC adaptive joint-grid preparation only",
    "requires real joint pilot rows over step size and leapfrog count",
    "next pilot tuples are requests for client execution, not readiness evidence",
    "selected scale is descriptive only when present",
    "does not train or adapt a transport",
    "does not execute pilot or final HMC candidates",
    "does not claim posterior convergence",
    "does not claim sampler superiority",
    "does not claim model-specific scientific validity",
)


def _finite_positive(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _finite_probability(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        return None
    return number


def _rounded(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True)
class FixedTransportHMCGridPolicyConfig:
    """Configuration for a generic fixed-transport HMC candidate policy."""

    base_step_size_candidates: tuple[float, ...] = (
        0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5)
    num_leapfrog_step_candidates: tuple[int, ...] = (2, 3, 5, 8, 10, 12, 16, 20, 25)
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    high_acceptance_refinement_max: float = 0.85
    max_num_leapfrog_steps: int = 25
    min_num_leapfrog_steps: int = 1
    min_trajectory_time: float = 0.0
    refinement_leapfrog_step: int = 2
    refinement_lower_leapfrog_margin: int = 4
    refinement_step_size_increment: float = 0.05
    refinement_step_size_count: int = 5
    policy_source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy"
    schema_version: int = 1

    def __post_init__(self) -> None:
        base_steps = tuple(
            _finite_positive(item, name="base_step_size_candidates")
            for item in self.base_step_size_candidates
        )
        if not base_steps:
            raise ValueError("base_step_size_candidates must be non-empty")
        leapfrogs = tuple(int(item) for item in self.num_leapfrog_step_candidates)
        if not leapfrogs or any(item <= 0 for item in leapfrogs):
            raise ValueError("num_leapfrog_step_candidates must be positive")
        lower, upper = (float(self.acceptance_band[0]), float(self.acceptance_band[1]))
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower <= upper < 1.0):
            raise ValueError("acceptance_band must satisfy 0 < lower <= upper < 1")
        high_max = float(self.high_acceptance_refinement_max)
        if not (math.isfinite(high_max) and upper <= high_max < 1.0):
            raise ValueError("high_acceptance_refinement_max must be in [upper, 1)")
        max_l = int(self.max_num_leapfrog_steps)
        if max_l <= 0:
            raise ValueError("max_num_leapfrog_steps must be positive")
        min_l = int(self.min_num_leapfrog_steps)
        if min_l <= 0 or min_l > max_l:
            raise ValueError(
                "min_num_leapfrog_steps must be positive and no larger than "
                "max_num_leapfrog_steps"
            )
        min_tau = float(self.min_trajectory_time)
        if not math.isfinite(min_tau) or min_tau < 0.0:
            raise ValueError("min_trajectory_time must be finite and nonnegative")
        step = int(self.refinement_leapfrog_step)
        margin = int(self.refinement_lower_leapfrog_margin)
        count = int(self.refinement_step_size_count)
        if step <= 0 or margin < 0 or count <= 0:
            raise ValueError("refinement step, margin, and count must be valid")
        increment = _finite_positive(
            self.refinement_step_size_increment,
            name="refinement_step_size_increment",
        )
        source = str(self.policy_source)
        if not source:
            raise ValueError("policy_source must be non-empty")
        schema = int(self.schema_version)
        if schema <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "base_step_size_candidates", base_steps)
        object.__setattr__(self, "num_leapfrog_step_candidates", tuple(dict.fromkeys(leapfrogs)))
        object.__setattr__(self, "acceptance_band", (lower, upper))
        object.__setattr__(self, "high_acceptance_refinement_max", high_max)
        object.__setattr__(self, "max_num_leapfrog_steps", max_l)
        object.__setattr__(self, "min_num_leapfrog_steps", min_l)
        object.__setattr__(self, "min_trajectory_time", min_tau)
        object.__setattr__(self, "refinement_leapfrog_step", step)
        object.__setattr__(self, "refinement_lower_leapfrog_margin", margin)
        object.__setattr__(self, "refinement_step_size_increment", increment)
        object.__setattr__(self, "refinement_step_size_count", count)
        object.__setattr__(self, "policy_source", source)
        object.__setattr__(self, "schema_version", schema)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_step_size_candidates": self.base_step_size_candidates,
            "num_leapfrog_step_candidates": self.num_leapfrog_step_candidates,
            "acceptance_band": self.acceptance_band,
            "high_acceptance_refinement_max": self.high_acceptance_refinement_max,
            "max_num_leapfrog_steps": self.max_num_leapfrog_steps,
            "min_num_leapfrog_steps": self.min_num_leapfrog_steps,
            "min_trajectory_time": self.min_trajectory_time,
            "refinement_leapfrog_step": self.refinement_leapfrog_step,
            "refinement_lower_leapfrog_margin": self.refinement_lower_leapfrog_margin,
            "refinement_step_size_increment": self.refinement_step_size_increment,
            "refinement_step_size_count": self.refinement_step_size_count,
            "policy_source": self.policy_source,
        }


@dataclass(frozen=True)
class FixedTransportHMCGridCandidateSpec:
    """One executable HMC candidate emitted by a policy spec."""

    candidate_index: int
    step_size: float
    num_leapfrog_steps: int
    grid_stage: str
    base_step_size: float | None = None
    step_size_scale: float | None = None
    refinement_reason: str | None = None

    def __post_init__(self) -> None:
        index = int(self.candidate_index)
        if index < 0:
            raise ValueError("candidate_index must be non-negative")
        step = _finite_positive(self.step_size, name="step_size")
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        stage = str(self.grid_stage)
        if not stage:
            raise ValueError("grid_stage must be non-empty")
        base = (
            None if self.base_step_size is None
            else _finite_positive(self.base_step_size, name="base_step_size")
        )
        scale = (
            None if self.step_size_scale is None
            else _finite_positive(self.step_size_scale, name="step_size_scale")
        )
        reason = None if self.refinement_reason is None else str(self.refinement_reason)
        object.__setattr__(self, "candidate_index", index)
        object.__setattr__(self, "step_size", _rounded(step))
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "grid_stage", stage)
        object.__setattr__(self, "base_step_size", None if base is None else _rounded(base))
        object.__setattr__(self, "step_size_scale", None if scale is None else _rounded(scale))
        object.__setattr__(self, "refinement_reason", reason)

    def identity(self) -> tuple[float, int]:
        return (self.step_size, self.num_leapfrog_steps)

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "trajectory_time": _rounded(self.step_size * self.num_leapfrog_steps),
            "grid_stage": self.grid_stage,
            "base_step_size": self.base_step_size,
            "step_size_scale": self.step_size_scale,
            "refinement_reason": self.refinement_reason,
        }


@dataclass(frozen=True)
class FixedTransportHMCGridPolicySpec:
    """Stable BayesFilter-owned policy artifact for model-specific executors."""

    config: FixedTransportHMCGridPolicyConfig
    candidate_specs: tuple[FixedTransportHMCGridCandidateSpec, ...]
    status: str
    refinement_reasons: tuple[str, ...] = ()
    selection_rule: str = "eligible_trajectory_acceptance_in_band_then_rhat_convergence_then_ess"
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_GRID_POLICY_NONCLAIMS

    def __post_init__(self) -> None:
        candidates = tuple(self.candidate_specs)
        if not candidates:
            raise ValueError("candidate_specs must be non-empty")
        identities = [candidate.identity() for candidate in candidates]
        if len(set(identities)) != len(identities):
            raise ValueError("candidate_specs must be unique by step_size and L")
        status = str(self.status)
        if not status:
            raise ValueError("status must be non-empty")
        selection = str(self.selection_rule)
        if not selection:
            raise ValueError("selection_rule must be non-empty")
        object.__setattr__(self, "candidate_specs", candidates)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "refinement_reasons",
            tuple(str(item) for item in self.refinement_reasons),
        )
        object.__setattr__(self, "selection_rule", selection)
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def payload(self) -> Mapping[str, Any]:
        payload = {
            "artifact_type": "bayesfilter_fixed_transport_hmc_grid_policy_spec",
            "schema_version": self.config.schema_version,
            "policy_source": self.config.policy_source,
            "config": self.config.payload(),
            "candidate_specs": tuple(candidate.payload() for candidate in self.candidate_specs),
            "candidate_count": len(self.candidate_specs),
            "status": self.status,
            "refinement_reasons": self.refinement_reasons,
            "selection_rule": self.selection_rule,
            "nonclaims": self.nonclaims,
        }
        return {
            **payload,
            "policy_hash": stable_config_hash(payload),
            "hash_function": "bayesfilter.runtime.stable_config_hash",
        }

    @property
    def policy_hash(self) -> str:
        return str(self.payload()["policy_hash"])


@dataclass(frozen=True)
class FixedTransportHMCPreparedGridConfig:
    """Configuration for BayesFilter-owned fixed-transport grid preparation."""

    grid_policy_config: FixedTransportHMCGridPolicyConfig
    scale_candidates: tuple[float, ...]
    pilot_acceptance_rates: tuple[float | None, ...]
    pilot_base_step_size: float
    pilot_num_leapfrog_steps: int
    target_acceptance: float = 0.70
    fallback_acceptance_max: float = 0.85
    source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy"
    schema_version: int = 1

    def __post_init__(self) -> None:
        scales = tuple(
            _finite_positive(item, name="scale_candidates")
            for item in self.scale_candidates
        )
        if not scales:
            raise ValueError("scale_candidates must be non-empty")
        if tuple(sorted(scales)) != scales:
            raise ValueError("scale_candidates must be sorted ascending")
        acceptances = tuple(
            None if item is None else _finite_probability(item)
            for item in self.pilot_acceptance_rates
        )
        if not acceptances:
            raise ValueError("pilot_acceptance_rates must contain real attempts")
        if len(acceptances) > len(scales):
            raise ValueError("pilot_acceptance_rates cannot exceed scale_candidates")
        pilot_base = _finite_positive(
            self.pilot_base_step_size,
            name="pilot_base_step_size",
        )
        pilot_l = int(self.pilot_num_leapfrog_steps)
        if pilot_l <= 0:
            raise ValueError("pilot_num_leapfrog_steps must be positive")
        target = float(self.target_acceptance)
        if not math.isfinite(target) or not 0.0 < target < 1.0:
            raise ValueError("target_acceptance must be finite and in (0, 1)")
        fallback = float(self.fallback_acceptance_max)
        upper = float(self.grid_policy_config.acceptance_band[1])
        if not math.isfinite(fallback) or fallback < upper or fallback >= 1.0:
            raise ValueError(
                "fallback_acceptance_max must be finite and in [upper, 1)"
            )
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        schema = int(self.schema_version)
        if schema <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "scale_candidates", scales)
        object.__setattr__(self, "pilot_acceptance_rates", acceptances)
        object.__setattr__(self, "pilot_base_step_size", pilot_base)
        object.__setattr__(self, "pilot_num_leapfrog_steps", pilot_l)
        object.__setattr__(self, "target_acceptance", target)
        object.__setattr__(self, "fallback_acceptance_max", fallback)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "schema_version", schema)

    @property
    def attempted_scale_candidates(self) -> tuple[float, ...]:
        return self.scale_candidates[: len(self.pilot_acceptance_rates)]

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grid_policy_config": self.grid_policy_config.payload(),
            "scale_candidates": self.scale_candidates,
            "attempted_scale_candidates": self.attempted_scale_candidates,
            "pilot_acceptance_rates": self.pilot_acceptance_rates,
            "pilot_base_step_size": self.pilot_base_step_size,
            "pilot_num_leapfrog_steps": self.pilot_num_leapfrog_steps,
            "target_acceptance": self.target_acceptance,
            "fallback_acceptance_max": self.fallback_acceptance_max,
            "source": self.source,
        }


@dataclass(frozen=True)
class FixedTransportHMCPreparedGrid:
    """Unified BayesFilter artifact for fixed-NeuTra HMC launch preparation."""

    config: FixedTransportHMCPreparedGridConfig
    scale_selection: GenericHMCFixedGridScaleSelection
    policy_spec: FixedTransportHMCGridPolicySpec | None
    status: str
    hard_vetoes: tuple[str, ...] = ()
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_PREPARED_GRID_NONCLAIMS

    def __post_init__(self) -> None:
        status = str(self.status)
        if not status:
            raise ValueError("status must be non-empty")
        vetoes = tuple(str(item) for item in self.hard_vetoes)
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        if self.launch_ready and self.policy_spec is None:
            raise ValueError("launch-ready prepared grid requires policy_spec")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "hard_vetoes", vetoes)
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def launch_ready(self) -> bool:
        return (
            self.status == "prepared_grid_ready"
            and self.policy_spec is not None
            and self.scale_selection.selected_scale is not None
            and not self.hard_vetoes
        )

    @property
    def prepared_policy_hash(self) -> str | None:
        return None if self.policy_spec is None else self.policy_spec.policy_hash

    def payload(self) -> Mapping[str, Any]:
        policy_payload = (
            None if self.policy_spec is None else self.policy_spec.payload()
        )
        payload = {
            "artifact_type": "bayesfilter_fixed_transport_hmc_prepared_grid",
            "schema_version": self.config.schema_version,
            "source": self.config.source,
            "config": self.config.payload(),
            "scale_selection": self.scale_selection.payload(),
            "selected_scale": self.scale_selection.selected_scale,
            "tuning_attempt_count": len(self.scale_selection.probes),
            "policy_spec": policy_payload,
            "prepared_policy_hash": self.prepared_policy_hash,
            "prepared_candidate_count": (
                None if policy_payload is None else policy_payload["candidate_count"]
            ),
            "status": self.status,
            "launch_ready": self.launch_ready,
            "hard_vetoes": self.hard_vetoes,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_model_scientific_validity": False,
            "nonclaims": self.nonclaims,
        }
        return {
            **payload,
            "prepared_grid_hash": stable_config_hash(payload),
            "hash_function": "bayesfilter.runtime.stable_config_hash",
        }

    @property
    def prepared_grid_hash(self) -> str:
        return str(self.payload()["prepared_grid_hash"])


@dataclass(frozen=True)
class FixedTransportHMCJointPilotRow:
    """One real pilot HMC diagnostic over a concrete step-size/L tuple."""

    candidate_index: int
    step_size: float
    num_leapfrog_steps: int
    acceptance_rate: float | None
    acceptance_class: str
    finite: bool
    status: str
    base_step_size: float | None = None
    step_size_scale: float | None = None
    hard_vetoes: tuple[str, ...] = ()
    launch_eligibility_reasons: tuple[str, ...] = ()
    adaptive_round_index: int | None = None

    def __post_init__(self) -> None:
        index = int(self.candidate_index)
        if index < 0:
            raise ValueError("candidate_index must be non-negative")
        step = _finite_positive(self.step_size, name="step_size")
        leapfrog = int(self.num_leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("num_leapfrog_steps must be positive")
        acceptance = _finite_probability(self.acceptance_rate)
        klass = str(self.acceptance_class)
        status = str(self.status)
        if not klass or not status:
            raise ValueError("acceptance_class and status must be non-empty")
        base = (
            None if self.base_step_size is None
            else _finite_positive(self.base_step_size, name="base_step_size")
        )
        scale = (
            None if self.step_size_scale is None
            else _finite_positive(self.step_size_scale, name="step_size_scale")
        )
        round_index = (
            None if self.adaptive_round_index is None
            else int(self.adaptive_round_index)
        )
        if round_index is not None and round_index < 0:
            raise ValueError("adaptive_round_index must be non-negative")
        object.__setattr__(self, "candidate_index", index)
        object.__setattr__(self, "step_size", _rounded(step))
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "acceptance_rate", acceptance)
        object.__setattr__(self, "acceptance_class", klass)
        object.__setattr__(self, "finite", bool(self.finite))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "base_step_size", None if base is None else _rounded(base))
        object.__setattr__(self, "step_size_scale", None if scale is None else _rounded(scale))
        object.__setattr__(self, "hard_vetoes", tuple(str(item) for item in self.hard_vetoes))
        object.__setattr__(
            self,
            "launch_eligibility_reasons",
            tuple(str(item) for item in self.launch_eligibility_reasons),
        )
        object.__setattr__(self, "adaptive_round_index", round_index)

    def identity(self) -> tuple[float, int]:
        return (self.step_size, self.num_leapfrog_steps)

    def payload(self) -> Mapping[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "trajectory_time": _rounded(self.step_size * self.num_leapfrog_steps),
            "acceptance_rate": self.acceptance_rate,
            "acceptance_class": self.acceptance_class,
            "finite": self.finite,
            "status": self.status,
            "base_step_size": self.base_step_size,
            "step_size_scale": self.step_size_scale,
            "hard_vetoes": self.hard_vetoes,
            "launch_eligible": bool(not self.launch_eligibility_reasons),
            "launch_eligibility_reasons": self.launch_eligibility_reasons,
            "adaptive_round_index": self.adaptive_round_index,
        }


@dataclass(frozen=True)
class FixedTransportHMCJointPreparedGridConfig:
    """Configuration for BayesFilter-owned joint finite-path preparation."""

    grid_policy_config: FixedTransportHMCGridPolicyConfig
    target_acceptance: float = 0.70
    fallback_acceptance_max: float = 0.85
    source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy"
    schema_version: int = 1

    def __post_init__(self) -> None:
        target = float(self.target_acceptance)
        lower, upper = self.grid_policy_config.acceptance_band
        if not math.isfinite(target) or not lower <= target <= upper:
            raise ValueError("target_acceptance must lie inside acceptance_band")
        fallback = float(self.fallback_acceptance_max)
        if not math.isfinite(fallback) or fallback < upper or fallback >= 1.0:
            raise ValueError(
                "fallback_acceptance_max must be finite and in [upper, 1)"
            )
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        schema = int(self.schema_version)
        if schema <= 0:
            raise ValueError("schema_version must be positive")
        object.__setattr__(self, "target_acceptance", target)
        object.__setattr__(self, "fallback_acceptance_max", fallback)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "schema_version", schema)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grid_policy_config": self.grid_policy_config.payload(),
            "target_acceptance": self.target_acceptance,
            "fallback_acceptance_max": self.fallback_acceptance_max,
            "source": self.source,
        }


@dataclass(frozen=True)
class FixedTransportHMCJointPreparedGrid:
    """Unified launch-prep artifact selected from joint pilot HMC rows."""

    config: FixedTransportHMCJointPreparedGridConfig
    joint_pilot_rows: tuple[FixedTransportHMCJointPilotRow, ...]
    selected_joint_pilot_row: FixedTransportHMCJointPilotRow | None
    policy_spec: FixedTransportHMCGridPolicySpec | None
    status: str
    selected_joint_pilot_candidate_match: bool = False
    hard_vetoes: tuple[str, ...] = ()
    nonclaims: tuple[str, ...] = FIXED_TRANSPORT_HMC_JOINT_PREPARED_GRID_NONCLAIMS
    adaptive: bool = False
    adaptive_round_count: int = 1
    max_adaptive_rounds: int | None = None
    seen_joint_pilot_tuples: tuple[Mapping[str, Any], ...] = ()
    next_joint_pilot_tuples: tuple[Mapping[str, Any], ...] = ()
    adaptive_scale_request: float | None = None
    adaptive_request_status: str | None = None
    adaptive_request_reason: str | None = None
    adaptive_round_summaries: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        rows = tuple(self.joint_pilot_rows)
        if not rows:
            raise ValueError("joint_pilot_rows must contain real attempts")
        status = str(self.status)
        if not status:
            raise ValueError("status must be non-empty")
        vetoes = tuple(str(item) for item in self.hard_vetoes)
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        adaptive_round_count = int(self.adaptive_round_count)
        if adaptive_round_count <= 0:
            raise ValueError("adaptive_round_count must be positive")
        max_rounds = (
            None if self.max_adaptive_rounds is None
            else int(self.max_adaptive_rounds)
        )
        if max_rounds is not None and max_rounds <= 0:
            raise ValueError("max_adaptive_rounds must be positive")
        next_tuples = tuple(dict(item) for item in self.next_joint_pilot_tuples)
        if next_tuples and self.launch_ready:
            raise ValueError("launch-ready joint prepared grid cannot request next pilots")
        seen_tuples = tuple(dict(item) for item in self.seen_joint_pilot_tuples)
        round_summaries = tuple(dict(item) for item in self.adaptive_round_summaries)
        request_status = (
            None if self.adaptive_request_status is None
            else str(self.adaptive_request_status)
        )
        request_reason = (
            None if self.adaptive_request_reason is None
            else str(self.adaptive_request_reason)
        )
        scale_request = (
            None if self.adaptive_scale_request is None
            else _finite_positive(
                self.adaptive_scale_request,
                name="adaptive_scale_request",
            )
        )
        if self.launch_ready and self.policy_spec is None:
            raise ValueError("launch-ready joint prepared grid requires policy_spec")
        if self.launch_ready and self.selected_joint_pilot_row is None:
            raise ValueError("launch-ready joint prepared grid requires selected row")
        object.__setattr__(self, "joint_pilot_rows", rows)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "hard_vetoes", vetoes)
        object.__setattr__(self, "nonclaims", nonclaims)
        object.__setattr__(self, "adaptive_round_count", adaptive_round_count)
        object.__setattr__(self, "max_adaptive_rounds", max_rounds)
        object.__setattr__(self, "seen_joint_pilot_tuples", seen_tuples)
        object.__setattr__(self, "next_joint_pilot_tuples", next_tuples)
        object.__setattr__(self, "adaptive_scale_request", scale_request)
        object.__setattr__(self, "adaptive_request_status", request_status)
        object.__setattr__(self, "adaptive_request_reason", request_reason)
        object.__setattr__(self, "adaptive_round_summaries", round_summaries)

    @property
    def launch_ready(self) -> bool:
        return (
            self.status == "joint_prepared_grid_ready"
            and self.policy_spec is not None
            and self.selected_joint_pilot_row is not None
            and bool(self.selected_joint_pilot_candidate_match)
            and not self.hard_vetoes
        )

    @property
    def prepared_policy_hash(self) -> str | None:
        return None if self.policy_spec is None else self.policy_spec.policy_hash

    def payload(self) -> Mapping[str, Any]:
        policy_payload = (
            None if self.policy_spec is None else self.policy_spec.payload()
        )
        selected_payload = (
            None
            if self.selected_joint_pilot_row is None
            else self.selected_joint_pilot_row.payload()
        )
        selected_tuple = (
            None if selected_payload is None
            else {
                "step_size": selected_payload["step_size"],
                "num_leapfrog_steps": selected_payload["num_leapfrog_steps"],
            }
        )
        payload = {
            "artifact_type": "bayesfilter_fixed_transport_hmc_joint_prepared_grid",
            "schema_version": self.config.schema_version,
            "source": self.config.source,
            "config": self.config.payload(),
            "joint_pilot_rows": tuple(row.payload() for row in self.joint_pilot_rows),
            "selected_joint_pilot_row": selected_payload,
            "selected_joint_pilot_tuple": selected_tuple,
            "selected_step_size": None if selected_payload is None else selected_payload["step_size"],
            "selected_num_leapfrog_steps": (
                None if selected_payload is None else selected_payload["num_leapfrog_steps"]
            ),
            "selected_scale": None if selected_payload is None else selected_payload["step_size_scale"],
            "selected_base_step_size": (
                None if selected_payload is None else selected_payload["base_step_size"]
            ),
            "tuning_attempt_count": len(self.joint_pilot_rows),
            "policy_spec": policy_payload,
            "prepared_policy_hash": self.prepared_policy_hash,
            "prepared_candidate_count": (
                None if policy_payload is None else policy_payload["candidate_count"]
            ),
            "selected_joint_pilot_candidate_match": bool(
                self.selected_joint_pilot_candidate_match
            ),
            "status": self.status,
            "launch_ready": self.launch_ready,
            "hard_vetoes": self.hard_vetoes,
            "adaptive": bool(self.adaptive),
            "adaptive_round_count": self.adaptive_round_count,
            "max_adaptive_rounds": self.max_adaptive_rounds,
            "seen_joint_pilot_tuples": self.seen_joint_pilot_tuples,
            "next_joint_pilot_tuples": self.next_joint_pilot_tuples,
            "next_joint_pilot_candidate_count": len(self.next_joint_pilot_tuples),
            "adaptive_scale_request": self.adaptive_scale_request,
            "adaptive_request_status": self.adaptive_request_status,
            "adaptive_request_reason": self.adaptive_request_reason,
            "adaptive_round_summaries": self.adaptive_round_summaries,
            "reports_posterior_convergence": False,
            "reports_sampler_superiority": False,
            "reports_model_scientific_validity": False,
            "nonclaims": self.nonclaims,
        }
        selected_hash_payload = {
            "selected_joint_pilot_row": selected_payload,
            "policy_hash": None if policy_payload is None else policy_payload["policy_hash"],
        }
        return {
            **payload,
            "selected_joint_pilot_row_hash": (
                None if selected_payload is None
                else stable_config_hash(selected_hash_payload)
            ),
            "prepared_grid_hash": stable_config_hash(payload),
            "hash_function": "bayesfilter.runtime.stable_config_hash",
        }

    @property
    def prepared_grid_hash(self) -> str:
        return str(self.payload()["prepared_grid_hash"])


def build_fixed_transport_hmc_grid_policy_spec(
    *,
    step_size_scale: float,
    config: FixedTransportHMCGridPolicyConfig | None = None,
    observed_candidate_rows: Sequence[Mapping[str, Any]] = (),
) -> FixedTransportHMCGridPolicySpec:
    """Build a candidate-grid policy spec with optional boundary refinement."""

    policy = FixedTransportHMCGridPolicyConfig() if config is None else config
    scale = _finite_positive(step_size_scale, name="step_size_scale")
    candidates: list[FixedTransportHMCGridCandidateSpec] = []
    seen: set[tuple[float, int]] = set()

    def add_candidate(
        *,
        step_size: float,
        leapfrog: int,
        stage: str,
        base_step_size: float | None = None,
        reason: str | None = None,
    ) -> None:
        if int(leapfrog) > policy.max_num_leapfrog_steps:
            return
        if int(leapfrog) < policy.min_num_leapfrog_steps:
            return
        if float(step_size) * int(leapfrog) < policy.min_trajectory_time:
            return
        candidate = FixedTransportHMCGridCandidateSpec(
            candidate_index=len(candidates),
            step_size=step_size,
            num_leapfrog_steps=int(leapfrog),
            grid_stage=stage,
            base_step_size=base_step_size,
            step_size_scale=scale,
            refinement_reason=reason,
        )
        identity = candidate.identity()
        if identity in seen:
            return
        seen.add(identity)
        candidates.append(candidate)

    for base_step in policy.base_step_size_candidates:
        for leapfrog in policy.num_leapfrog_step_candidates:
            add_candidate(
                step_size=base_step * scale,
                leapfrog=leapfrog,
                stage="coarse_grid",
                base_step_size=base_step,
            )

    reasons = []
    boundary = _boundary_refinement_request(policy, observed_candidate_rows)
    if boundary is not None:
        step_size, start_l, stop_l = boundary
        reason = "high_acceptance_shorter_L_and_in_band_longer_L_boundary"
        reasons.append(reason)
        for refined_step in _refined_step_sizes(policy, step_size):
            for leapfrog in _refined_leapfrogs(policy, start_l, stop_l):
                add_candidate(
                    step_size=refined_step,
                    leapfrog=leapfrog,
                    stage="local_refinement_grid",
                    reason=reason,
                )

    ceiling_refinement = _finite_domain_ceiling_refinement_requests(
        policy,
        observed_candidate_rows,
    )
    if ceiling_refinement:
        reason = "finite_domain_ceiling_bracket_refinement"
        reasons.append(reason)
        for refined_step, leapfrog in ceiling_refinement:
            add_candidate(
                step_size=refined_step,
                leapfrog=leapfrog,
                stage="finite_domain_ceiling_refinement_grid",
                reason=reason,
            )

    return FixedTransportHMCGridPolicySpec(
        config=policy,
        candidate_specs=tuple(
            FixedTransportHMCGridCandidateSpec(
                candidate_index=index,
                step_size=candidate.step_size,
                num_leapfrog_steps=candidate.num_leapfrog_steps,
                grid_stage=candidate.grid_stage,
                base_step_size=candidate.base_step_size,
                step_size_scale=candidate.step_size_scale,
                refinement_reason=candidate.refinement_reason,
            )
            for index, candidate in enumerate(candidates)
        ),
        status="refinement_added" if reasons else "coarse_grid_only",
        refinement_reasons=tuple(reasons),
    )


def prepare_fixed_transport_hmc_grid_policy(
    *,
    base_step_size_candidates: Sequence[float],
    num_leapfrog_step_candidates: Sequence[int],
    scale_candidates: Sequence[float],
    pilot_acceptance_rates: Sequence[float | None],
    acceptance_band: tuple[float, float] = (0.65, 0.75),
    fallback_acceptance_max: float = 0.85,
    target_acceptance: float = 0.70,
    pilot_base_step_size: float | None = None,
    pilot_num_leapfrog_steps: int | None = None,
    observed_candidate_rows: Sequence[Mapping[str, Any]] = (),
    policy_source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy",
) -> FixedTransportHMCPreparedGrid:
    """Build a launch-safe fixed-transport HMC grid from real pilot attempts.

    This is the public preparation contract model-specific NeuTra launchers
    should consume.  Candidate identity is emitted only after BayesFilter has
    seen nonempty pilot acceptance diagnostics and selected an in-band scale.
    Identity-only policy specs remain available for diagnostics, but they are
    not launch-readiness evidence.
    """

    base_steps = tuple(
        _finite_positive(item, name="base_step_size_candidates")
        for item in base_step_size_candidates
    )
    leapfrogs = tuple(int(item) for item in num_leapfrog_step_candidates)
    if not leapfrogs or any(item <= 0 for item in leapfrogs):
        raise ValueError("num_leapfrog_step_candidates must be positive")
    pilot_base = (
        max(base_steps)
        if pilot_base_step_size is None
        else _finite_positive(pilot_base_step_size, name="pilot_base_step_size")
    )
    pilot_l = max(leapfrogs) if pilot_num_leapfrog_steps is None else int(
        pilot_num_leapfrog_steps
    )
    grid_config = FixedTransportHMCGridPolicyConfig(
        base_step_size_candidates=base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        acceptance_band=acceptance_band,
        high_acceptance_refinement_max=fallback_acceptance_max,
        max_num_leapfrog_steps=max(leapfrogs),
        min_num_leapfrog_steps=min(leapfrogs),
        min_trajectory_time=0.0,
        policy_source=policy_source,
    )
    config = FixedTransportHMCPreparedGridConfig(
        grid_policy_config=grid_config,
        scale_candidates=tuple(float(item) for item in scale_candidates),
        pilot_acceptance_rates=tuple(pilot_acceptance_rates),
        pilot_base_step_size=pilot_base,
        pilot_num_leapfrog_steps=pilot_l,
        target_acceptance=target_acceptance,
        fallback_acceptance_max=fallback_acceptance_max,
        source=policy_source,
    )
    selection = select_hmc_fixed_grid_scale(
        base_step_size_candidates=base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        scale_candidates=config.attempted_scale_candidates,
        pilot_acceptance_rates=config.pilot_acceptance_rates,
        acceptance_band=grid_config.acceptance_band,
        fallback_acceptance_max=fallback_acceptance_max,
        target_acceptance=target_acceptance,
        pilot_base_step_size=pilot_base,
        pilot_num_leapfrog_steps=pilot_l,
        policy_source=f"{policy_source}.prepare_fixed_transport_hmc_grid_policy",
    )
    if selection.status != "scale_selected_in_band":
        return FixedTransportHMCPreparedGrid(
            config=config,
            scale_selection=selection,
            policy_spec=None,
            status=str(selection.status),
            hard_vetoes=tuple(selection.vetoes or ("scale_not_in_acceptance_band",)),
        )
    policy_spec = build_fixed_transport_hmc_grid_policy_spec(
        step_size_scale=float(selection.selected_scale),
        config=grid_config,
        observed_candidate_rows=observed_candidate_rows,
    )
    return FixedTransportHMCPreparedGrid(
        config=config,
        scale_selection=selection,
        policy_spec=policy_spec,
        status="prepared_grid_ready",
        hard_vetoes=(),
    )


def prepare_fixed_transport_hmc_joint_grid_policy(
    *,
    base_step_size_candidates: Sequence[float],
    num_leapfrog_step_candidates: Sequence[int],
    joint_pilot_rows: Sequence[Mapping[str, Any]],
    acceptance_band: tuple[float, float] = (0.65, 0.75),
    fallback_acceptance_max: float = 0.85,
    target_acceptance: float = 0.70,
    min_num_leapfrog_steps: int | None = None,
    min_trajectory_time: float = 0.0,
    observed_candidate_rows: Sequence[Mapping[str, Any]] = (),
    policy_source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy",
) -> FixedTransportHMCJointPreparedGrid:
    """Build a launch-safe fixed-transport grid from joint HMC pilot rows.

    This is the preferred serious NeuTra preparation contract.  It never treats
    a policy hash or a scale-only pilot as launch-readiness evidence.  A
    launch-ready artifact requires at least one finite, in-band pilot row for a
    concrete `(step_size, num_leapfrog_steps)` tuple, and the emitted policy
    must contain that selected tuple.
    """

    base_steps = tuple(
        _finite_positive(item, name="base_step_size_candidates")
        for item in base_step_size_candidates
    )
    leapfrogs = tuple(int(item) for item in num_leapfrog_step_candidates)
    if not leapfrogs or any(item <= 0 for item in leapfrogs):
        raise ValueError("num_leapfrog_step_candidates must be positive")
    initial_grid_config = FixedTransportHMCGridPolicyConfig(
        base_step_size_candidates=base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        acceptance_band=acceptance_band,
        high_acceptance_refinement_max=fallback_acceptance_max,
        max_num_leapfrog_steps=max(leapfrogs),
        min_num_leapfrog_steps=(
            min(leapfrogs)
            if min_num_leapfrog_steps is None
            else int(min_num_leapfrog_steps)
        ),
        min_trajectory_time=float(min_trajectory_time),
        policy_source=policy_source,
    )
    rows = tuple(
        _joint_pilot_row_from_mapping(
            row,
            index=index,
            base_step_size_candidates=base_steps,
            min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
            min_trajectory_time=initial_grid_config.min_trajectory_time,
            acceptance_band=initial_grid_config.acceptance_band,
            fallback_acceptance_max=fallback_acceptance_max,
        )
        for index, row in enumerate(joint_pilot_rows)
    )
    if not rows:
        raise ValueError("joint_pilot_rows must contain real attempts")
    lower, upper = initial_grid_config.acceptance_band
    viable = [
        row for row in rows
        if row.finite
        and not row.hard_vetoes
        and row.acceptance_rate is not None
        and lower <= row.acceptance_rate <= upper
        and row.acceptance_class == "in_band"
        and not row.launch_eligibility_reasons
    ]
    if not viable:
        config = FixedTransportHMCJointPreparedGridConfig(
            grid_policy_config=initial_grid_config,
            target_acceptance=target_acceptance,
            fallback_acceptance_max=fallback_acceptance_max,
            source=policy_source,
        )
        return FixedTransportHMCJointPreparedGrid(
            config=config,
            joint_pilot_rows=rows,
            selected_joint_pilot_row=None,
            policy_spec=None,
            status=_joint_prep_failure_status(rows),
            selected_joint_pilot_candidate_match=False,
            hard_vetoes=_joint_prep_failure_vetoes(rows),
        )
    midpoint = float(target_acceptance)
    selected = min(
        viable,
        key=lambda row: (
            abs(float(row.acceptance_rate) - midpoint),
            row.step_size * row.num_leapfrog_steps,
            row.num_leapfrog_steps,
            row.step_size,
            row.candidate_index,
        ),
    )
    policy_base_steps = tuple(sorted({
        *(_rounded(step) for step in base_steps),
        _rounded(selected.step_size),
    }))
    grid_config = FixedTransportHMCGridPolicyConfig(
        base_step_size_candidates=policy_base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        acceptance_band=acceptance_band,
        high_acceptance_refinement_max=fallback_acceptance_max,
        max_num_leapfrog_steps=max(leapfrogs),
        min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
        min_trajectory_time=initial_grid_config.min_trajectory_time,
        policy_source=policy_source,
    )
    config = FixedTransportHMCJointPreparedGridConfig(
        grid_policy_config=grid_config,
        target_acceptance=target_acceptance,
        fallback_acceptance_max=fallback_acceptance_max,
        source=policy_source,
    )
    policy_spec = build_fixed_transport_hmc_grid_policy_spec(
        step_size_scale=1.0,
        config=grid_config,
        observed_candidate_rows=tuple(observed_candidate_rows) + tuple(
            row.payload() for row in rows
        ),
    )
    policy_identities = {
        (
            candidate.step_size,
            candidate.num_leapfrog_steps,
        )
        for candidate in policy_spec.candidate_specs
    }
    selected_match = selected.identity() in policy_identities
    hard_vetoes: tuple[str, ...] = (
        () if selected_match else ("selected_joint_pilot_tuple_missing_from_policy",)
    )
    return FixedTransportHMCJointPreparedGrid(
        config=config,
        joint_pilot_rows=rows,
        selected_joint_pilot_row=selected,
        policy_spec=None if hard_vetoes else policy_spec,
        status=(
            "joint_prepared_grid_ready"
            if not hard_vetoes
            else "joint_prepared_grid_policy_mismatch"
        ),
        selected_joint_pilot_candidate_match=selected_match,
        hard_vetoes=hard_vetoes,
    )


def prepare_fixed_transport_hmc_adaptive_joint_grid_policy(
    *,
    base_step_size_candidates: Sequence[float],
    num_leapfrog_step_candidates: Sequence[int],
    joint_pilot_rows: Sequence[Mapping[str, Any]],
    acceptance_band: tuple[float, float] = (0.65, 0.75),
    fallback_acceptance_max: float = 0.85,
    target_acceptance: float = 0.70,
    min_num_leapfrog_steps: int | None = None,
    min_trajectory_time: float = 0.0,
    observed_candidate_rows: Sequence[Mapping[str, Any]] = (),
    max_adaptive_rounds: int = 5,
    policy_source: str = "bayesfilter.inference.fixed_transport_hmc_grid_policy",
) -> FixedTransportHMCJointPreparedGrid:
    """Build or request a bounded adaptive fixed-transport HMC joint grid.

    Client code supplies real pilot rows and executes any returned
    ``next_joint_pilot_tuples``.  BayesFilter owns the row classification,
    bounded bracketing rule, final selected tuple, and policy/prepared hashes;
    it never treats a requested next tuple set as launch readiness.
    """

    base_steps = tuple(
        _finite_positive(item, name="base_step_size_candidates")
        for item in base_step_size_candidates
    )
    leapfrogs = tuple(int(item) for item in num_leapfrog_step_candidates)
    if not leapfrogs or any(item <= 0 for item in leapfrogs):
        raise ValueError("num_leapfrog_step_candidates must be positive")
    max_rounds = int(max_adaptive_rounds)
    if max_rounds <= 0:
        raise ValueError("max_adaptive_rounds must be positive")
    initial_grid_config = FixedTransportHMCGridPolicyConfig(
        base_step_size_candidates=base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        acceptance_band=acceptance_band,
        high_acceptance_refinement_max=fallback_acceptance_max,
        max_num_leapfrog_steps=max(leapfrogs),
        min_num_leapfrog_steps=(
            min(leapfrogs)
            if min_num_leapfrog_steps is None
            else int(min_num_leapfrog_steps)
        ),
        min_trajectory_time=float(min_trajectory_time),
        policy_source=policy_source,
    )
    rows = tuple(
        _joint_pilot_row_from_mapping(
            row,
            index=index,
            base_step_size_candidates=base_steps,
            min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
            min_trajectory_time=initial_grid_config.min_trajectory_time,
            acceptance_band=initial_grid_config.acceptance_band,
            fallback_acceptance_max=fallback_acceptance_max,
        )
        for index, row in enumerate(joint_pilot_rows)
    )
    if not rows:
        raise ValueError("joint_pilot_rows must contain real attempts")

    lower, upper = initial_grid_config.acceptance_band
    viable = [
        row for row in rows
        if row.finite
        and not row.hard_vetoes
        and row.acceptance_rate is not None
        and lower <= row.acceptance_rate <= upper
        and row.acceptance_class == "in_band"
        and not row.launch_eligibility_reasons
    ]
    round_count = _adaptive_round_count(rows)
    if viable:
        return _adaptive_ready_joint_grid(
            base_steps=base_steps,
            leapfrogs=leapfrogs,
            rows=rows,
            viable=viable,
            acceptance_band=acceptance_band,
            fallback_acceptance_max=fallback_acceptance_max,
            target_acceptance=target_acceptance,
            min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
            min_trajectory_time=initial_grid_config.min_trajectory_time,
            observed_candidate_rows=observed_candidate_rows,
            policy_source=policy_source,
            max_adaptive_rounds=max_rounds,
        )

    config = FixedTransportHMCJointPreparedGridConfig(
        grid_policy_config=initial_grid_config,
        target_acceptance=target_acceptance,
        fallback_acceptance_max=fallback_acceptance_max,
        source=policy_source,
    )
    if round_count >= max_rounds:
        return FixedTransportHMCJointPreparedGrid(
            config=config,
            joint_pilot_rows=rows,
            selected_joint_pilot_row=None,
            policy_spec=None,
            status="adaptive_joint_grid_max_rounds_exhausted",
            selected_joint_pilot_candidate_match=False,
            hard_vetoes=("adaptive_joint_pilot_max_rounds_exhausted",),
            nonclaims=FIXED_TRANSPORT_HMC_ADAPTIVE_JOINT_PREPARED_GRID_NONCLAIMS,
            adaptive=True,
            adaptive_round_count=round_count,
            max_adaptive_rounds=max_rounds,
            seen_joint_pilot_tuples=_seen_joint_pilot_tuples(rows),
            next_joint_pilot_tuples=(),
            adaptive_request_status="max_rounds_exhausted",
            adaptive_request_reason=_adaptive_request_reason(rows),
            adaptive_round_summaries=_adaptive_round_summaries(rows),
        )

    scale, reason = _adaptive_next_scale_request(
        rows,
        acceptance_band=initial_grid_config.acceptance_band,
        fallback_acceptance_max=fallback_acceptance_max,
    )
    local_tuples = _adaptive_local_high_warning_tuples(
        rows=rows,
        base_step_size_candidates=base_steps,
        num_leapfrog_step_candidates=leapfrogs,
        acceptance_band=initial_grid_config.acceptance_band,
        fallback_acceptance_max=fallback_acceptance_max,
        min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
        min_trajectory_time=initial_grid_config.min_trajectory_time,
    )
    if local_tuples:
        next_tuples = local_tuples
        reason = "latest_high_warning_candidate_local_scale_refinement"
    else:
        too_high_tuples = _adaptive_local_too_high_tuples(
            rows=rows,
            acceptance_band=initial_grid_config.acceptance_band,
            fallback_acceptance_max=fallback_acceptance_max,
            min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
            min_trajectory_time=initial_grid_config.min_trajectory_time,
        )
        if too_high_tuples:
            next_tuples = too_high_tuples
            reason = "latest_too_high_candidate_same_leapfrog_local_step_refinement"
        else:
            next_tuples = _adaptive_next_joint_pilot_tuples(
                base_step_size_candidates=base_steps,
                num_leapfrog_step_candidates=leapfrogs,
                scale=scale,
                rows=rows,
                min_num_leapfrog_steps=initial_grid_config.min_num_leapfrog_steps,
                min_trajectory_time=initial_grid_config.min_trajectory_time,
            )
    if not next_tuples:
        return FixedTransportHMCJointPreparedGrid(
            config=config,
            joint_pilot_rows=rows,
            selected_joint_pilot_row=None,
            policy_spec=None,
            status="adaptive_joint_grid_no_new_tuples",
            selected_joint_pilot_candidate_match=False,
            hard_vetoes=("adaptive_joint_pilot_no_new_tuples",),
            nonclaims=FIXED_TRANSPORT_HMC_ADAPTIVE_JOINT_PREPARED_GRID_NONCLAIMS,
            adaptive=True,
            adaptive_round_count=round_count,
            max_adaptive_rounds=max_rounds,
            seen_joint_pilot_tuples=_seen_joint_pilot_tuples(rows),
            next_joint_pilot_tuples=(),
            adaptive_scale_request=scale,
            adaptive_request_status="max_rounds_exhausted",
            adaptive_request_reason=reason,
            adaptive_round_summaries=_adaptive_round_summaries(rows),
        )
    return FixedTransportHMCJointPreparedGrid(
        config=config,
        joint_pilot_rows=rows,
        selected_joint_pilot_row=None,
        policy_spec=None,
        status="adaptive_joint_grid_next_round_requested",
        selected_joint_pilot_candidate_match=False,
        hard_vetoes=(),
        nonclaims=FIXED_TRANSPORT_HMC_ADAPTIVE_JOINT_PREPARED_GRID_NONCLAIMS,
        adaptive=True,
        adaptive_round_count=round_count,
        max_adaptive_rounds=max_rounds,
        seen_joint_pilot_tuples=_seen_joint_pilot_tuples(rows),
        next_joint_pilot_tuples=next_tuples,
        adaptive_scale_request=scale,
        adaptive_request_status="next_round_requested",
        adaptive_request_reason=reason,
        adaptive_round_summaries=_adaptive_round_summaries(rows),
    )


def _boundary_refinement_request(
    config: FixedTransportHMCGridPolicyConfig,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, int, int] | None:
    lower, upper = config.acceptance_band
    del lower
    by_step: dict[float, list[tuple[int, float]]] = {}
    for row in rows:
        step = row.get("step_size")
        leapfrog = row.get("num_leapfrog_steps", row.get("leapfrog"))
        acceptance = _finite_probability(row.get("acceptance_rate"))
        if step is None or leapfrog is None or acceptance is None:
            continue
        step_value = _rounded(float(step))
        leapfrog_value = int(leapfrog)
        by_step.setdefault(step_value, []).append((leapfrog_value, acceptance))

    for step_value, diagnostics in sorted(by_step.items()):
        high = [
            (leapfrog, acceptance)
            for leapfrog, acceptance in diagnostics
            if upper < acceptance <= config.high_acceptance_refinement_max
        ]
        in_band = [
            (leapfrog, acceptance)
            for leapfrog, acceptance in diagnostics
            if config.acceptance_band[0] <= acceptance <= upper
        ]
        if not high or not in_band:
            continue
        shorter_high = min(high, key=lambda item: item[0])
        longer_in_band = min(
            (item for item in in_band if item[0] > shorter_high[0]),
            key=lambda item: item[0],
            default=None,
        )
        if longer_in_band is None:
            continue
        start = max(1, shorter_high[0] - config.refinement_lower_leapfrog_margin)
        stop = min(config.max_num_leapfrog_steps, longer_in_band[0])
        return (step_value, start, stop)
    return None


def _finite_domain_vetoes() -> set[str]:
    return {
        "candidate_finite_domain_error",
        "bayesfilter_principal_sqrt_nonpositive_covariance",
        "bayesfilter_symmetric_sylvester_nonpositive_factor",
    }


def _finite_domain_ceiling_refinement_requests(
    config: FixedTransportHMCGridPolicyConfig,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, int], ...]:
    usable: list[tuple[float, int, float]] = []
    ceiling_steps: list[float] = []
    finite_domain_vetoes = _finite_domain_vetoes()
    for row in rows:
        step = row.get("step_size", row.get("epsilon"))
        leapfrog = row.get(
            "num_leapfrog_steps",
            row.get("num_leapfrog", row.get("leapfrog")),
        )
        if step is None or leapfrog is None:
            continue
        try:
            step_value = _finite_positive(float(step), name="step_size")
            leapfrog_value = int(leapfrog)
        except (TypeError, ValueError):
            continue
        if _joint_launch_eligibility_reasons(
            step_size=step_value,
            num_leapfrog_steps=leapfrog_value,
            min_num_leapfrog_steps=config.min_num_leapfrog_steps,
            min_trajectory_time=config.min_trajectory_time,
        ):
            continue
        hard_vetoes = _row_hard_vetoes(row)
        if finite_domain_vetoes.intersection(hard_vetoes):
            ceiling_steps.append(_rounded(step_value))
            continue
        acceptance = _finite_probability(
            row.get("acceptance_rate", row.get("accept_rate"))
        )
        finite = bool(row.get("finite", not hard_vetoes))
        if (
            finite
            and not hard_vetoes
            and acceptance is not None
            and acceptance > config.acceptance_band[1]
        ):
            usable.append((_rounded(step_value), leapfrog_value, acceptance))
    if not usable or not ceiling_steps:
        return ()

    leapfrogs = tuple(
        item for item in config.num_leapfrog_step_candidates
        if item >= config.min_num_leapfrog_steps
    )
    out: list[tuple[float, int]] = []
    seen: set[tuple[float, int]] = set()
    for lower_step in sorted({step for step, _, _ in usable}):
        upper_candidates = [
            step for step in ceiling_steps if step > lower_step
        ]
        if not upper_candidates:
            continue
        upper_step = min(upper_candidates)
        gap = upper_step - lower_step
        if gap <= 0.0:
            continue
        for step_value in (
            lower_step,
            _rounded(lower_step + 0.25 * gap),
            _rounded(lower_step + 0.50 * gap),
            _rounded(lower_step + 0.75 * gap),
        ):
            for leapfrog in leapfrogs:
                if _joint_launch_eligibility_reasons(
                    step_size=step_value,
                    num_leapfrog_steps=leapfrog,
                    min_num_leapfrog_steps=config.min_num_leapfrog_steps,
                    min_trajectory_time=config.min_trajectory_time,
                ):
                    continue
                identity = (_rounded(step_value), int(leapfrog))
                if identity in seen:
                    continue
                seen.add(identity)
                out.append(identity)
    return tuple(out)


def _joint_pilot_row_from_mapping(
    row: Mapping[str, Any],
    *,
    index: int,
    base_step_size_candidates: Sequence[float],
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
) -> FixedTransportHMCJointPilotRow:
    step = row.get("step_size", row.get("epsilon"))
    leapfrog = row.get("num_leapfrog_steps", row.get("num_leapfrog", row.get("leapfrog")))
    if step is None or leapfrog is None:
        raise ValueError("joint pilot rows require step_size and num_leapfrog_steps")
    acceptance = _finite_probability(
        row.get("acceptance_rate", row.get("accept_rate"))
    )
    hard_vetoes = _row_hard_vetoes(row)
    finite = bool(row.get("finite", not hard_vetoes))
    if acceptance is None or hard_vetoes or not finite:
        acceptance_class = "invalid"
        status = "pilot_joint_invalid"
    else:
        acceptance_class = _classify_joint_acceptance(
            acceptance,
            acceptance_band=acceptance_band,
            fallback_acceptance_max=fallback_acceptance_max,
        )
        status = "pilot_joint_usable"
    step_value = float(step)
    leapfrog_value = int(leapfrog)
    base_step = _infer_base_step_size(row, step_value, base_step_size_candidates)
    scale = (
        None
        if base_step is None
        else _rounded(step_value / float(base_step))
    )
    candidate_index = int(row.get("candidate_index", index))
    adaptive_round_index = row.get("adaptive_round_index")
    eligibility_reasons = _joint_launch_eligibility_reasons(
        step_size=step_value,
        num_leapfrog_steps=leapfrog_value,
        min_num_leapfrog_steps=min_num_leapfrog_steps,
        min_trajectory_time=min_trajectory_time,
    )
    return FixedTransportHMCJointPilotRow(
        candidate_index=candidate_index,
        step_size=step_value,
        num_leapfrog_steps=leapfrog_value,
        acceptance_rate=acceptance,
        acceptance_class=acceptance_class,
        finite=finite,
        status=status,
        base_step_size=base_step,
        step_size_scale=scale,
        hard_vetoes=tuple(hard_vetoes),
        launch_eligibility_reasons=tuple(eligibility_reasons),
        adaptive_round_index=adaptive_round_index,
    )


def _joint_launch_eligibility_reasons(
    *,
    step_size: float,
    num_leapfrog_steps: int,
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    leapfrog = int(num_leapfrog_steps)
    if leapfrog < int(min_num_leapfrog_steps):
        reasons.append("short_leapfrog")
    trajectory_time = float(step_size) * leapfrog
    if trajectory_time < float(min_trajectory_time):
        reasons.append("short_trajectory_time")
    return tuple(reasons)


def _adaptive_ready_joint_grid(
    *,
    base_steps: Sequence[float],
    leapfrogs: Sequence[int],
    rows: Sequence[FixedTransportHMCJointPilotRow],
    viable: Sequence[FixedTransportHMCJointPilotRow],
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
    target_acceptance: float,
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
    observed_candidate_rows: Sequence[Mapping[str, Any]],
    policy_source: str,
    max_adaptive_rounds: int,
) -> FixedTransportHMCJointPreparedGrid:
    midpoint = float(target_acceptance)
    selected = min(
        viable,
        key=lambda row: (
            abs(float(row.acceptance_rate) - midpoint),
            row.step_size * row.num_leapfrog_steps,
            row.num_leapfrog_steps,
            row.step_size,
            row.candidate_index,
        ),
    )
    policy_base_steps = tuple(sorted({
        *(_rounded(step) for step in base_steps),
        _rounded(selected.step_size),
    }))
    grid_config = FixedTransportHMCGridPolicyConfig(
        base_step_size_candidates=policy_base_steps,
        num_leapfrog_step_candidates=tuple(leapfrogs),
        acceptance_band=acceptance_band,
        high_acceptance_refinement_max=fallback_acceptance_max,
        max_num_leapfrog_steps=max(tuple(leapfrogs)),
        min_num_leapfrog_steps=int(min_num_leapfrog_steps),
        min_trajectory_time=float(min_trajectory_time),
        policy_source=policy_source,
    )
    config = FixedTransportHMCJointPreparedGridConfig(
        grid_policy_config=grid_config,
        target_acceptance=target_acceptance,
        fallback_acceptance_max=fallback_acceptance_max,
        source=policy_source,
    )
    policy_spec = build_fixed_transport_hmc_grid_policy_spec(
        step_size_scale=1.0,
        config=grid_config,
        observed_candidate_rows=tuple(observed_candidate_rows) + tuple(
            row.payload() for row in rows
        ),
    )
    policy_identities = {
        (
            candidate.step_size,
            candidate.num_leapfrog_steps,
        )
        for candidate in policy_spec.candidate_specs
    }
    selected_match = selected.identity() in policy_identities
    hard_vetoes: tuple[str, ...] = (
        () if selected_match else ("selected_joint_pilot_tuple_missing_from_policy",)
    )
    return FixedTransportHMCJointPreparedGrid(
        config=config,
        joint_pilot_rows=tuple(rows),
        selected_joint_pilot_row=selected,
        policy_spec=None if hard_vetoes else policy_spec,
        status=(
            "joint_prepared_grid_ready"
            if not hard_vetoes
            else "joint_prepared_grid_policy_mismatch"
        ),
        selected_joint_pilot_candidate_match=selected_match,
        hard_vetoes=hard_vetoes,
        nonclaims=FIXED_TRANSPORT_HMC_ADAPTIVE_JOINT_PREPARED_GRID_NONCLAIMS,
        adaptive=True,
        adaptive_round_count=_adaptive_round_count(rows),
        max_adaptive_rounds=int(max_adaptive_rounds),
        seen_joint_pilot_tuples=_seen_joint_pilot_tuples(rows),
        next_joint_pilot_tuples=(),
        adaptive_scale_request=selected.step_size_scale,
        adaptive_request_status="ready",
        adaptive_request_reason="finite_in_band_joint_pilot_tuple_selected",
        adaptive_round_summaries=_adaptive_round_summaries(rows),
    )


def _adaptive_round_count(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> int:
    round_indices = [
        int(row.adaptive_round_index)
        for row in rows
        if row.adaptive_round_index is not None
    ]
    if not round_indices:
        return 1
    return max(round_indices) + 1


def _seen_joint_pilot_tuples(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "step_size": row.step_size,
            "num_leapfrog_steps": row.num_leapfrog_steps,
            "adaptive_round_index": row.adaptive_round_index,
        }
        for row in rows
    )


def _adaptive_next_scale_request(
    rows: Sequence[FixedTransportHMCJointPilotRow],
    *,
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
) -> tuple[float, str]:
    lower, upper = acceptance_band
    latest_rows = _latest_adaptive_round_rows(rows)
    valid = [
        row for row in latest_rows
        if row.finite
        and not row.hard_vetoes
        and row.acceptance_rate is not None
    ]
    current_scale = _latest_adaptive_scale(latest_rows)
    if not valid:
        return (
            _rounded(current_scale * 0.1),
            "latest_joint_pilot_rows_missing_or_invalid_factor_0.1",
        )
    max_acceptance = max(float(row.acceptance_rate) for row in valid)
    min_acceptance = min(float(row.acceptance_rate) for row in valid)
    finite_domain_ceiling = _latest_has_finite_domain_ceiling(latest_rows)
    if all(float(row.acceptance_rate) > upper for row in valid):
        if finite_domain_ceiling:
            if min_acceptance <= fallback_acceptance_max:
                return (
                    _rounded(current_scale * 1.05),
                    "latest_valid_acceptance_high_near_band_with_finite_domain_ceiling_factor_1.05",
                )
            return (
                _rounded(current_scale * 0.9),
                "latest_valid_acceptance_high_with_finite_domain_ceiling_factor_0.9",
            )
        largest_step = max(row.step_size for row in valid)
        largest_step_acceptance = max(
            float(row.acceptance_rate)
            for row in valid
            if row.step_size == largest_step
        )
        if largest_step_acceptance < fallback_acceptance_max:
            return (
                _rounded(current_scale * 5.0),
                "latest_valid_acceptance_high_largest_step_below_fallback_factor_5",
            )
        return (
            _rounded(current_scale * 9.0),
            "latest_valid_acceptance_too_high_factor_9",
        )
    if all(float(row.acceptance_rate) < lower for row in valid):
        if max_acceptance > 0.4:
            return (
                _rounded(current_scale * 0.2),
                "latest_valid_acceptance_below_band_above_point_four_factor_0.2",
            )
        return (
            _rounded(current_scale * 0.1),
            "latest_valid_acceptance_below_point_four_factor_0.1",
        )
    if any(float(row.acceptance_rate) > upper for row in valid):
        return (
            _rounded(current_scale * 5.0),
            "latest_mixed_valid_acceptance_no_in_band_high_present_factor_5",
        )
    return (
        _rounded(current_scale * 0.2),
        "latest_mixed_valid_acceptance_no_in_band_low_present_factor_0.2",
    )


def _latest_has_finite_domain_ceiling(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> bool:
    finite_domain_vetoes = _finite_domain_vetoes()
    for row in rows:
        if finite_domain_vetoes.intersection(row.hard_vetoes):
            return True
    return False


def _latest_adaptive_round_rows(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> tuple[FixedTransportHMCJointPilotRow, ...]:
    latest = max(
        (0 if row.adaptive_round_index is None else int(row.adaptive_round_index))
        for row in rows
    )
    return tuple(
        row for row in rows
        if (0 if row.adaptive_round_index is None else int(row.adaptive_round_index)) == latest
    )


def _latest_adaptive_scale(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> float:
    scales = [
        float(row.step_size_scale)
        for row in rows
        if row.step_size_scale is not None and math.isfinite(float(row.step_size_scale))
    ]
    if not scales:
        return 1.0
    return max(scales)


def _adaptive_request_reason(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> str:
    classes = sorted({row.acceptance_class for row in rows})
    if not classes:
        return "no_joint_pilot_rows"
    return "joint_pilot_classes_" + "_".join(classes)


def _adaptive_next_joint_pilot_tuples(
    *,
    base_step_size_candidates: Sequence[float],
    num_leapfrog_step_candidates: Sequence[int],
    scale: float,
    rows: Sequence[FixedTransportHMCJointPilotRow],
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
) -> tuple[Mapping[str, Any], ...]:
    seen = {row.identity() for row in rows}
    out: list[Mapping[str, Any]] = []
    for base_step in base_step_size_candidates:
        step_size = _rounded(float(base_step) * float(scale))
        for leapfrog in num_leapfrog_step_candidates:
            if _joint_launch_eligibility_reasons(
                step_size=step_size,
                num_leapfrog_steps=int(leapfrog),
                min_num_leapfrog_steps=min_num_leapfrog_steps,
                min_trajectory_time=min_trajectory_time,
            ):
                continue
            identity = (step_size, int(leapfrog))
            if identity in seen:
                continue
            out.append({
                "candidate_index": len(out),
                "step_size": step_size,
                "num_leapfrog_steps": int(leapfrog),
                "base_step_size": _rounded(float(base_step)),
                "step_size_scale": _rounded(float(scale)),
            })
    return tuple(out)


def _adaptive_local_high_warning_tuples(
    *,
    rows: Sequence[FixedTransportHMCJointPilotRow],
    base_step_size_candidates: Sequence[float],
    num_leapfrog_step_candidates: Sequence[int],
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
) -> tuple[Mapping[str, Any], ...]:
    lower, upper = acceptance_band
    del lower
    latest_rows = _latest_adaptive_round_rows(rows)
    finite_domain_ceiling = _latest_has_finite_domain_ceiling(latest_rows)
    valid = [
        row for row in latest_rows
        if row.finite
        and not row.hard_vetoes
        and row.acceptance_rate is not None
        and not row.launch_eligibility_reasons
    ]
    high_warning = [
        row for row in valid
        if upper < float(row.acceptance_rate) <= float(fallback_acceptance_max)
    ]
    if not high_warning or not finite_domain_ceiling:
        return ()
    selected = min(
        high_warning,
        key=lambda row: (
            abs(float(row.acceptance_rate) - upper),
            row.step_size * row.num_leapfrog_steps,
            row.num_leapfrog_steps,
            row.step_size,
            row.candidate_index,
        ),
    )
    if selected.base_step_size is None:
        return ()
    base_step = float(selected.base_step_size)
    selected_scale = float(selected.step_size) / base_step
    seen = {row.identity() for row in rows}
    selected_acceptance = float(selected.acceptance_rate)
    if selected_acceptance <= upper + 0.05:
        scale_factors = (0.9, 0.95, 0.975, 1.025)
    else:
        scale_factors = (
            0.9,
            0.95,
            0.975,
            1.025,
            1.1,
            1.2,
            1.333333333333,
            1.5,
        )
    ordered_base_steps = tuple(_rounded(float(item)) for item in base_step_size_candidates)
    if _rounded(base_step) in ordered_base_steps:
        base_index = ordered_base_steps.index(_rounded(base_step))
        base_steps = ordered_base_steps[max(0, base_index - 1): base_index + 2]
    else:
        base_steps = (_rounded(base_step),)
    out: list[Mapping[str, Any]] = []
    for factor in scale_factors:
        scale = _rounded(selected_scale * factor)
        for candidate_base in base_steps:
            step_size = _rounded(float(candidate_base) * scale)
            for leapfrog in num_leapfrog_step_candidates:
                if _joint_launch_eligibility_reasons(
                    step_size=step_size,
                    num_leapfrog_steps=int(leapfrog),
                    min_num_leapfrog_steps=int(min_num_leapfrog_steps),
                    min_trajectory_time=float(min_trajectory_time),
                ):
                    continue
                identity = (step_size, int(leapfrog))
                if identity in seen:
                    continue
                seen.add(identity)
                out.append({
                    "candidate_index": len(out),
                    "step_size": step_size,
                    "num_leapfrog_steps": int(leapfrog),
                    "base_step_size": _rounded(float(candidate_base)),
                    "step_size_scale": scale,
                    "refinement_reason": (
                        "finite_domain_ceiling_high_warning_local_scale_refinement"
                    ),
                })
    return tuple(out)


def _adaptive_local_too_high_tuples(
    *,
    rows: Sequence[FixedTransportHMCJointPilotRow],
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
    min_num_leapfrog_steps: int,
    min_trajectory_time: float,
) -> tuple[Mapping[str, Any], ...]:
    lower, upper = acceptance_band
    del lower
    latest_rows = _latest_adaptive_round_rows(rows)
    valid = [
        row for row in latest_rows
        if row.finite
        and not row.hard_vetoes
        and row.acceptance_rate is not None
        and not row.launch_eligibility_reasons
        and float(row.acceptance_rate) > upper
    ]
    if not valid:
        return ()
    if len({row.num_leapfrog_steps for row in valid}) < 2:
        return ()
    if not _latest_has_finite_domain_ceiling(latest_rows):
        return ()
    if not all(float(row.acceptance_rate) > fallback_acceptance_max for row in valid):
        return ()
    seen = {row.identity() for row in rows}
    finite_domain_vetoes = _finite_domain_vetoes()
    ceiling_by_leapfrog: dict[int, float] = {}
    for row in latest_rows:
        if not finite_domain_vetoes.intersection(row.hard_vetoes):
            continue
        current = ceiling_by_leapfrog.get(row.num_leapfrog_steps)
        if current is None or row.step_size < current:
            ceiling_by_leapfrog[row.num_leapfrog_steps] = row.step_size
    selected_by_leapfrog: dict[int, FixedTransportHMCJointPilotRow] = {}
    for row in valid:
        current = selected_by_leapfrog.get(row.num_leapfrog_steps)
        if current is None or row.step_size > current.step_size:
            selected_by_leapfrog[row.num_leapfrog_steps] = row
    out: list[Mapping[str, Any]] = []
    for leapfrog in sorted(selected_by_leapfrog):
        row = selected_by_leapfrog[leapfrog]
        ceiling = ceiling_by_leapfrog.get(leapfrog)
        for factor in (1.2, 1.4, 1.6, 1.8, 2.0):
            step_size = _rounded(row.step_size * factor)
            if ceiling is not None and step_size >= ceiling:
                continue
            if _joint_launch_eligibility_reasons(
                step_size=step_size,
                num_leapfrog_steps=leapfrog,
                min_num_leapfrog_steps=int(min_num_leapfrog_steps),
                min_trajectory_time=float(min_trajectory_time),
            ):
                continue
            identity = (step_size, leapfrog)
            if identity in seen:
                continue
            seen.add(identity)
            base_step = row.base_step_size
            scale = None
            if base_step is not None:
                scale = _rounded(step_size / float(base_step))
            out.append({
                "candidate_index": len(out),
                "step_size": step_size,
                "num_leapfrog_steps": leapfrog,
                "base_step_size": base_step,
                "step_size_scale": scale,
                "refinement_reason": (
                    "too_high_acceptance_same_leapfrog_local_step_refinement"
                ),
            })
    return tuple(out)


def _adaptive_round_summaries(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> tuple[Mapping[str, Any], ...]:
    grouped: dict[int, list[FixedTransportHMCJointPilotRow]] = {}
    for row in rows:
        round_index = 0 if row.adaptive_round_index is None else int(
            row.adaptive_round_index
        )
        grouped.setdefault(round_index, []).append(row)
    summaries = []
    for round_index in sorted(grouped):
        round_rows = grouped[round_index]
        finite_acceptances = [
            float(row.acceptance_rate)
            for row in round_rows
            if row.finite and row.acceptance_rate is not None and not row.hard_vetoes
        ]
        class_counts: dict[str, int] = {}
        for row in round_rows:
            class_counts[row.acceptance_class] = (
                class_counts.get(row.acceptance_class, 0) + 1
            )
        summaries.append({
            "round_index": round_index,
            "row_count": len(round_rows),
            "class_counts": class_counts,
            "finite_valid_count": len(finite_acceptances),
            "acceptance_min": (
                None if not finite_acceptances else min(finite_acceptances)
            ),
            "acceptance_max": (
                None if not finite_acceptances else max(finite_acceptances)
            ),
        })
    return tuple(summaries)


def _row_hard_vetoes(row: Mapping[str, Any]) -> tuple[str, ...]:
    hard = row.get("hard_vetoes", row.get("hard_veto_reasons", ()))
    if hard is None:
        return ()
    if isinstance(hard, str):
        return (hard,)
    return tuple(str(item) for item in hard)


def _classify_joint_acceptance(
    acceptance: float,
    *,
    acceptance_band: tuple[float, float],
    fallback_acceptance_max: float,
) -> str:
    lower, upper = acceptance_band
    value = float(acceptance)
    if lower <= value <= upper:
        return "in_band"
    if upper < value <= fallback_acceptance_max:
        return "high_warning_band"
    if value < lower:
        return "below_band"
    return "too_high"


def _infer_base_step_size(
    row: Mapping[str, Any],
    step_size: float,
    base_step_size_candidates: Sequence[float],
) -> float | None:
    explicit = row.get("base_step_size")
    if explicit is not None:
        return _finite_positive(float(explicit), name="base_step_size")
    candidates = sorted(float(item) for item in base_step_size_candidates)
    for candidate in candidates:
        if abs(float(step_size) - candidate) < 1.0e-12:
            return _rounded(candidate)
    candidates = sorted(candidates, reverse=True)
    for candidate in candidates:
        if candidate > 0.0:
            ratio = float(step_size) / candidate
            if math.isfinite(ratio) and abs(ratio - round(ratio, 12)) < 1.0e-9:
                return _rounded(candidate)
    return None


def _joint_prep_failure_status(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> str:
    classes = {row.acceptance_class for row in rows}
    if classes == {"invalid"}:
        return "joint_grid_search_failed_all_invalid"
    if any(row.acceptance_class == "in_band" for row in rows):
        return "joint_grid_search_failed_no_eligible_in_band"
    if any(row.acceptance_class == "too_high" for row in rows):
        return "joint_grid_search_failed_high_acceptance"
    if any(row.acceptance_class == "high_warning_band" for row in rows):
        return "joint_grid_selected_warning_band"
    if any(row.acceptance_class == "below_band" for row in rows):
        return "joint_grid_selected_below_band"
    return "joint_grid_search_failed_no_in_band"


def _joint_prep_failure_vetoes(
    rows: Sequence[FixedTransportHMCJointPilotRow],
) -> tuple[str, ...]:
    status = _joint_prep_failure_status(rows)
    if status == "joint_grid_search_failed_all_invalid":
        return ("joint_pilot_rows_all_invalid",)
    if status == "joint_grid_search_failed_no_eligible_in_band":
        return ("joint_pilot_no_eligible_in_band_candidate",)
    if status == "joint_grid_search_failed_high_acceptance":
        return ("joint_pilot_acceptance_above_band",)
    if status == "joint_grid_selected_warning_band":
        return ("joint_pilot_warning_band_not_launch_ready",)
    if status == "joint_grid_selected_below_band":
        return ("joint_pilot_acceptance_below_band",)
    return ("joint_pilot_no_in_band_candidate",)


def _refined_step_sizes(
    config: FixedTransportHMCGridPolicyConfig,
    boundary_step_size: float,
) -> tuple[float, ...]:
    return tuple(
        _rounded(boundary_step_size + index * config.refinement_step_size_increment)
        for index in range(config.refinement_step_size_count)
    )


def _refined_leapfrogs(
    config: FixedTransportHMCGridPolicyConfig,
    start_l: int,
    stop_l: int,
) -> tuple[int, ...]:
    values = list(range(int(start_l), int(stop_l), config.refinement_leapfrog_step))
    values.append(int(stop_l))
    return tuple(dict.fromkeys(item for item in values if item <= config.max_num_leapfrog_steps))
