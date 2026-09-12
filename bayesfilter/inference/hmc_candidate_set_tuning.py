"""Pure controller for the shared HMC candidate-set tuning lifecycle.

The controller deliberately does not run TensorFlow or a sampler.  It owns the
candidate identity, cohort ordering, repair lineage, reserve accounting, and
result authority.  Numerical adapters provide one deterministic observation for
each dispatched work item.  Keeping this state machine independent makes the
candidate-set contract testable without treating a short numerical run as
evidence of convergence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal


HMC_CANDIDATE_SET_TUNING_SCHEMA = "bayesfilter.hmc_candidate_set_tuning.v1"
HMC_CANDIDATE_SET_RESULT_SCHEMA = "bayesfilter.hmc_candidate_set_result.v1"
REPAIR_REASONS = frozenset(
    {
        "repair_budget_exhausted",
        "repair_not_scheduled",
        "repair_infrastructure_failure",
        "repair_verification_failed",
        "shared_invalidity",
        "repair_limit_exhausted",
    }
)
_DIRECTIONAL = {"repair_step_higher", "repair_step_lower"}
_STAGE_ORDER = {"measurement": 0, "verification": 1}


def _finite_positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _stable_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _stable_payload(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_stable_payload(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("hash payload contains a non-finite float")
        return value
    return str(value)


def _sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _stable_payload(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tuple_ints(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain distinct values")
    return result


@dataclass(frozen=True)
class HMCCandidateSetScope:
    """Immutable preparation identity shared by every candidate in one call."""

    scope_id: str
    search_id: str
    target_signature: str
    mass_signature: str
    coordinate_system: str
    start_bank_signature: str
    warmup_protocol: str
    backend: str = "tensorflow_probability"
    dtype: str = "float64"
    execution_mode: str = "tf_function"
    adapter_signature: str = "unbound-adapter"
    source_dependency_hash: str = "unbound-source-closure"
    target_preparation_identity: str = "unbound-target-preparation"
    transition_identity: str = "unbound-transition"
    epsilon_domain: tuple[float, float] = (1.0e-6, 2.0)
    repair_factor: float = 2.0
    max_repairs_per_family: int = 2

    def __post_init__(self) -> None:
        for name in (
            "scope_id",
            "search_id",
            "target_signature",
            "mass_signature",
            "coordinate_system",
            "start_bank_signature",
            "warmup_protocol",
            "backend",
            "dtype",
            "execution_mode",
            "adapter_signature",
            "source_dependency_hash",
            "target_preparation_identity",
            "transition_identity",
        ):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must be non-empty")
        lo, hi = (float(value) for value in self.epsilon_domain)
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0.0 < lo < hi:
            raise ValueError("epsilon_domain must be finite and 0 < low < high")
        object.__setattr__(self, "epsilon_domain", (lo, hi))
        object.__setattr__(
            self, "repair_factor", _finite_positive(self.repair_factor, "repair_factor")
        )
        if self.repair_factor == 1.0:
            raise ValueError("repair_factor must change epsilon")
        repairs = int(self.max_repairs_per_family)
        if repairs < 0:
            raise ValueError("max_repairs_per_family must be non-negative")
        object.__setattr__(self, "max_repairs_per_family", repairs)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_tuning_scope.v1",
            "scope_id": self.scope_id,
            "search_id": self.search_id,
            "target_signature": self.target_signature,
            "mass_signature": self.mass_signature,
            "coordinate_system": self.coordinate_system,
            "start_bank_signature": self.start_bank_signature,
            "warmup_protocol": self.warmup_protocol,
            "backend": self.backend,
            "dtype": self.dtype,
            "execution_mode": self.execution_mode,
            "adapter_signature": self.adapter_signature,
            "source_dependency_hash": self.source_dependency_hash,
            "target_preparation_identity": self.target_preparation_identity,
            "transition_identity": self.transition_identity,
            "epsilon_domain": self.epsilon_domain,
            "repair_factor": self.repair_factor,
            "max_repairs_per_family": self.max_repairs_per_family,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCCandidateSetScope":
        if payload.get("schema") != "bayesfilter.hmc_tuning_scope.v1":
            raise ValueError("unsupported HMC tuning scope schema")
        return cls(
            scope_id=payload["scope_id"],
            search_id=payload["search_id"],
            target_signature=payload["target_signature"],
            mass_signature=payload["mass_signature"],
            coordinate_system=payload["coordinate_system"],
            start_bank_signature=payload["start_bank_signature"],
            warmup_protocol=payload["warmup_protocol"],
            backend=payload.get("backend", "tensorflow_probability"),
            dtype=payload.get("dtype", "float64"),
            execution_mode=payload.get("execution_mode", "tf_function"),
            adapter_signature=payload.get("adapter_signature", "unbound-adapter"),
            source_dependency_hash=payload.get(
                "source_dependency_hash", "unbound-source-closure"
            ),
            target_preparation_identity=payload.get(
                "target_preparation_identity", "unbound-target-preparation"
            ),
            transition_identity=payload.get(
                "transition_identity", "unbound-transition"
            ),
            epsilon_domain=tuple(payload.get("epsilon_domain", (1.0e-6, 2.0))),
            repair_factor=payload.get("repair_factor", 2.0),
            max_repairs_per_family=payload.get("max_repairs_per_family", 2),
        )


@dataclass(frozen=True)
class HMCTuningCandidateRecord:
    """One exact immutable ``(L, epsilon)`` setting."""

    candidate_id: str
    candidate_family_id: str
    parent_candidate_id: str | None
    creation_ordinal: int
    scope_id: str
    search_id: str
    leapfrog_steps: int
    epsilon: float
    target_signature: str
    mass_signature: str
    coordinate_system: str
    start_bank_signature: str
    warmup_protocol: str
    backend: str
    dtype: str
    execution_mode: str
    adapter_signature: str
    source_dependency_hash: str
    target_preparation_identity: str
    transition_identity: str
    candidate_record_hash: str

    @classmethod
    def create(
        cls,
        scope: HMCCandidateSetScope,
        *,
        leapfrog_steps: int,
        epsilon: float,
        creation_ordinal: int,
        candidate_family_id: str | None = None,
        parent_candidate_id: str | None = None,
    ) -> "HMCTuningCandidateRecord":
        steps = int(leapfrog_steps)
        if steps <= 0:
            raise ValueError("leapfrog_steps must be positive")
        value = _finite_positive(epsilon, "epsilon")
        if not scope.epsilon_domain[0] <= value <= scope.epsilon_domain[1]:
            raise ValueError("epsilon is outside the scope domain")
        ordinal = int(creation_ordinal)
        if ordinal <= 0:
            raise ValueError("creation_ordinal must be positive")
        family = candidate_family_id or f"{scope.scope_id}:{scope.search_id}:family:{ordinal:06d}"
        candidate_id = f"{scope.scope_id}:{scope.search_id}:candidate:{ordinal:06d}"
        payload = {
            "schema": "bayesfilter.hmc_candidate_record.v1",
            "candidate_id": candidate_id,
            "candidate_family_id": family,
            "parent_candidate_id": parent_candidate_id,
            "creation_ordinal": ordinal,
            "scope_id": scope.scope_id,
            "search_id": scope.search_id,
            "leapfrog_steps": steps,
            "epsilon": value,
            "target_signature": scope.target_signature,
            "mass_signature": scope.mass_signature,
            "coordinate_system": scope.coordinate_system,
            "start_bank_signature": scope.start_bank_signature,
            "warmup_protocol": scope.warmup_protocol,
            "backend": scope.backend,
            "dtype": scope.dtype,
            "execution_mode": scope.execution_mode,
            "adapter_signature": scope.adapter_signature,
            "source_dependency_hash": scope.source_dependency_hash,
            "target_preparation_identity": scope.target_preparation_identity,
            "transition_identity": scope.transition_identity,
        }
        return cls(
            candidate_id=candidate_id,
            candidate_family_id=family,
            parent_candidate_id=parent_candidate_id,
            creation_ordinal=ordinal,
            scope_id=scope.scope_id,
            search_id=scope.search_id,
            leapfrog_steps=steps,
            epsilon=value,
            target_signature=scope.target_signature,
            mass_signature=scope.mass_signature,
            coordinate_system=scope.coordinate_system,
            start_bank_signature=scope.start_bank_signature,
            warmup_protocol=scope.warmup_protocol,
            backend=scope.backend,
            dtype=scope.dtype,
            execution_mode=scope.execution_mode,
            adapter_signature=scope.adapter_signature,
            source_dependency_hash=scope.source_dependency_hash,
            target_preparation_identity=scope.target_preparation_identity,
            transition_identity=scope.transition_identity,
            candidate_record_hash=_sha256(payload),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_candidate_record.v1",
            "candidate_id": self.candidate_id,
            "candidate_family_id": self.candidate_family_id,
            "parent_candidate_id": self.parent_candidate_id,
            "creation_ordinal": self.creation_ordinal,
            "scope_id": self.scope_id,
            "search_id": self.search_id,
            "leapfrog_steps": self.leapfrog_steps,
            "epsilon": self.epsilon,
            "target_signature": self.target_signature,
            "mass_signature": self.mass_signature,
            "coordinate_system": self.coordinate_system,
            "start_bank_signature": self.start_bank_signature,
            "warmup_protocol": self.warmup_protocol,
            "backend": self.backend,
            "dtype": self.dtype,
            "execution_mode": self.execution_mode,
            "adapter_signature": self.adapter_signature,
            "source_dependency_hash": self.source_dependency_hash,
            "target_preparation_identity": self.target_preparation_identity,
            "transition_identity": self.transition_identity,
            "candidate_record_hash": self.candidate_record_hash,
        }

    @classmethod
    def from_payload(
        cls,
        scope: HMCCandidateSetScope,
        payload: Mapping[str, Any],
    ) -> "HMCTuningCandidateRecord":
        candidate = cls.create(
            scope,
            leapfrog_steps=payload["leapfrog_steps"],
            epsilon=payload["epsilon"],
            creation_ordinal=payload["creation_ordinal"],
            candidate_family_id=payload.get("candidate_family_id"),
            parent_candidate_id=payload.get("parent_candidate_id"),
        )
        if candidate.payload() != dict(payload):
            raise ValueError("candidate record payload does not match its issued identity")
        return candidate


@dataclass(frozen=True)
class HMCVerificationReceipt:
    verification_attempt_id: str
    candidate_id: str
    candidate_record_hash: str
    exact_l: int
    epsilon: float
    mass_signature: str
    stream_id: str
    draw_range: tuple[int, int]
    seed_lineage: tuple[int, ...]
    decision: str
    acceptance: float | None
    hard_vetoes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.draw_range[0] < 0 or self.draw_range[1] < self.draw_range[0]:
            raise ValueError("draw_range must be ordered and non-negative")
        if self.acceptance is not None and not math.isfinite(float(self.acceptance)):
            raise ValueError("acceptance must be finite when present")

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_verification_receipt.v1",
            "verification_attempt_id": self.verification_attempt_id,
            "candidate_id": self.candidate_id,
            "candidate_record_hash": self.candidate_record_hash,
            "exact_l": self.exact_l,
            "epsilon": self.epsilon,
            "mass_signature": self.mass_signature,
            "stream_id": self.stream_id,
            "draw_range": self.draw_range,
            "seed_lineage": self.seed_lineage,
            "decision": self.decision,
            "acceptance": self.acceptance,
            "hard_vetoes": self.hard_vetoes,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCVerificationReceipt":
        if payload.get("schema") != "bayesfilter.hmc_verification_receipt.v1":
            raise ValueError("unsupported HMC verification receipt schema")
        return cls(
            verification_attempt_id=payload["verification_attempt_id"],
            candidate_id=payload["candidate_id"],
            candidate_record_hash=payload["candidate_record_hash"],
            exact_l=payload["exact_l"],
            epsilon=payload["epsilon"],
            mass_signature=payload["mass_signature"],
            stream_id=payload["stream_id"],
            draw_range=tuple(payload["draw_range"]),
            seed_lineage=tuple(payload["seed_lineage"]),
            decision=payload["decision"],
            acceptance=payload.get("acceptance"),
            hard_vetoes=tuple(payload.get("hard_vetoes", ())),
        )


@dataclass(frozen=True)
class HMCRepairAction:
    repair_action_id: str
    parent_candidate_id: str
    child_candidate_id: str
    candidate_family_id: str
    source_verification_hash: str
    old_epsilon: float
    new_epsilon: float
    exact_l: int
    mass_signature: str
    direction: Literal["repair_step_higher", "repair_step_lower"]
    execution_status: Literal["not_started", "running", "executed"]
    verification_status: Literal["pending", "passed", "failed", "skipped"]
    qualified_repair_status: Literal[
        "executed_and_verified", "not_executed_with_reason"
    ]
    not_executed_reason: str | None = None
    allocation_source: str | None = None

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_repair_action.v1",
            "repair_action_id": self.repair_action_id,
            "parent_candidate_id": self.parent_candidate_id,
            "child_candidate_id": self.child_candidate_id,
            "candidate_family_id": self.candidate_family_id,
            "source_verification_hash": self.source_verification_hash,
            "old_epsilon": self.old_epsilon,
            "new_epsilon": self.new_epsilon,
            "exact_l": self.exact_l,
            "mass_signature": self.mass_signature,
            "direction": self.direction,
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "qualified_repair_status": self.qualified_repair_status,
            "not_executed_reason": self.not_executed_reason,
            "allocation_source": self.allocation_source,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCRepairAction":
        if payload.get("schema") != "bayesfilter.hmc_repair_action.v1":
            raise ValueError("unsupported HMC repair action schema")
        return cls(
            repair_action_id=payload["repair_action_id"],
            parent_candidate_id=payload["parent_candidate_id"],
            child_candidate_id=payload["child_candidate_id"],
            candidate_family_id=payload["candidate_family_id"],
            source_verification_hash=payload["source_verification_hash"],
            old_epsilon=payload["old_epsilon"],
            new_epsilon=payload["new_epsilon"],
            exact_l=payload["exact_l"],
            mass_signature=payload["mass_signature"],
            direction=payload["direction"],
            execution_status=payload["execution_status"],
            verification_status=payload["verification_status"],
            qualified_repair_status=payload["qualified_repair_status"],
            not_executed_reason=payload.get("not_executed_reason"),
            allocation_source=payload.get("allocation_source"),
        )


@dataclass(frozen=True)
class HMCWorkItem:
    work_item_id: str
    candidate_id: str
    candidate_record_hash: str
    candidate_family_id: str
    stage: Literal["measurement", "verification"]
    cohort_id: str
    ordinal: int
    priority: int
    repair_action_id: str | None = None
    verification_attempt_id: str | None = None
    status: Literal["pending", "running", "completed", "interrupted"] = "pending"
    reservation_units: int = 1

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_work_item.v1",
            "work_item_id": self.work_item_id,
            "candidate_id": self.candidate_id,
            "candidate_record_hash": self.candidate_record_hash,
            "candidate_family_id": self.candidate_family_id,
            "stage": self.stage,
            "cohort_id": self.cohort_id,
            "ordinal": self.ordinal,
            "priority": self.priority,
            "repair_action_id": self.repair_action_id,
            "verification_attempt_id": self.verification_attempt_id,
            "status": self.status,
            "reservation_units": self.reservation_units,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCWorkItem":
        if payload.get("schema") != "bayesfilter.hmc_work_item.v1":
            raise ValueError("unsupported HMC work-item schema")
        return cls(
            work_item_id=payload["work_item_id"],
            candidate_id=payload["candidate_id"],
            candidate_record_hash=payload["candidate_record_hash"],
            candidate_family_id=payload["candidate_family_id"],
            stage=payload["stage"],
            cohort_id=payload["cohort_id"],
            ordinal=payload["ordinal"],
            priority=payload["priority"],
            repair_action_id=payload.get("repair_action_id"),
            verification_attempt_id=payload.get("verification_attempt_id"),
            status=payload.get("status", "pending"),
            reservation_units=payload.get("reservation_units", 1),
        )


@dataclass(frozen=True)
class HMCControllerConfig:
    """Bounded deterministic scheduler settings."""

    primary_l_grid: tuple[int, ...]
    epsilon_by_l: tuple[tuple[int, tuple[float, ...]], ...]
    total_budget_units: int = 100
    repair_reserve_units: int = 20
    candidate_reserve_units: int = 3
    allow_repair_from_free_pool: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "primary_l_grid", _tuple_ints(self.primary_l_grid, "primary_l_grid"))
        normalized: list[tuple[int, tuple[float, ...]]] = []
        seen: set[int] = set()
        for raw_l, raw_epsilons in self.epsilon_by_l:
            leapfrog = int(raw_l)
            if leapfrog not in self.primary_l_grid or leapfrog in seen:
                raise ValueError("epsilon_by_l must cover each L exactly once")
            values = tuple(_finite_positive(value, "epsilon") for value in raw_epsilons)
            if not values or len(set(values)) != len(values):
                raise ValueError("each L requires distinct positive epsilon proposals")
            normalized.append((leapfrog, values))
            seen.add(leapfrog)
        if seen != set(self.primary_l_grid):
            raise ValueError("epsilon_by_l must cover each L exactly once")
        object.__setattr__(self, "epsilon_by_l", tuple(normalized))
        for name in ("total_budget_units", "repair_reserve_units", "candidate_reserve_units"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if self.repair_reserve_units > self.total_budget_units:
            raise ValueError("repair_reserve_units cannot exceed total_budget_units")
        object.__setattr__(self, "allow_repair_from_free_pool", bool(self.allow_repair_from_free_pool))

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_controller_config.v1",
            "primary_l_grid": self.primary_l_grid,
            "epsilon_by_l": self.epsilon_by_l,
            "total_budget_units": self.total_budget_units,
            "repair_reserve_units": self.repair_reserve_units,
            "candidate_reserve_units": self.candidate_reserve_units,
            "allow_repair_from_free_pool": self.allow_repair_from_free_pool,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCControllerConfig":
        if payload.get("schema") != "bayesfilter.hmc_controller_config.v1":
            raise ValueError("unsupported HMC controller config schema")
        return cls(
            primary_l_grid=tuple(payload["primary_l_grid"]),
            epsilon_by_l=tuple(
                (item[0], tuple(item[1])) for item in payload["epsilon_by_l"]
            ),
            total_budget_units=payload["total_budget_units"],
            repair_reserve_units=payload["repair_reserve_units"],
            candidate_reserve_units=payload["candidate_reserve_units"],
            allow_repair_from_free_pool=payload.get("allow_repair_from_free_pool", True),
        )


@dataclass(frozen=True)
class HMCTuningCandidateSetResult:
    scope: HMCCandidateSetScope
    config: HMCControllerConfig
    candidates: tuple[HMCTuningCandidateRecord, ...]
    candidate_states: Mapping[str, str]
    work_items: tuple[HMCWorkItem, ...]
    verification_receipts: tuple[HMCVerificationReceipt, ...]
    repair_actions: tuple[HMCRepairAction, ...]
    accounting_events: tuple[Mapping[str, Any], ...]
    verified_candidate_ids: tuple[str, ...]
    viable_candidate_ids: tuple[str, ...]
    final_status: str
    completion_status: str
    remaining_budget_units: int
    resume_pending_work_item_ids: tuple[str, ...]
    budget_used_units: int = 0
    reserved_budget_units: int = 0

    def replay_candidate(self, candidate_id: str) -> HMCTuningCandidateRecord:
        if self.completion_status == "shared_invalidity":
            raise ValueError("shared-invalidity result cannot be replayed")
        if candidate_id not in self.verified_candidate_ids:
            raise ValueError("replay requires an explicitly verified candidate ID")
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                if candidate.parent_candidate_id is not None:
                    action = next(
                        (
                            item
                            for item in self.repair_actions
                            if item.child_candidate_id == candidate_id
                        ),
                        None,
                    )
                    if action is None or action.qualified_repair_status != (
                        "executed_and_verified"
                    ):
                        raise ValueError("candidate repair is not qualified for replay")
                return candidate
        raise ValueError("verified candidate is missing from immutable records")

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": HMC_CANDIDATE_SET_RESULT_SCHEMA,
            "scope": self.scope.payload(),
            "config": self.config.payload(),
            "candidates": tuple(item.payload() for item in self.candidates),
            "candidate_states": dict(self.candidate_states),
            "work_items": tuple(item.payload() for item in self.work_items),
            "verification_receipts": tuple(item.payload() for item in self.verification_receipts),
            "repair_actions": tuple(item.payload() for item in self.repair_actions),
            "accounting_events": self.accounting_events,
            "verified_candidate_ids": self.verified_candidate_ids,
            "viable_candidate_ids": self.viable_candidate_ids,
            "final_status": self.final_status,
            "completion_status": self.completion_status,
            "remaining_budget_units": self.remaining_budget_units,
            "resume_pending_work_item_ids": self.resume_pending_work_item_ids,
            "budget_used_units": self.budget_used_units,
            "reserved_budget_units": self.reserved_budget_units,
            "artifact_authority": False,
            "replay_authority": True,
            "numerical_handoff_authority": False,
            "nominee_id": None,
            "nonclaims": (
                "no statistical superiority claim",
                "no posterior convergence claim from tuning evidence",
                "no implicit candidate selection",
            ),
        }


@dataclass(frozen=True)
class HMCTuningScopeCollection:
    """Read-only reporting view over independent scope results.

    The collection intentionally has no outcome provider, queue, or mutation
    method. It cannot merge chain states or issue a new kernel handoff.
    """

    results: tuple[HMCTuningCandidateSetResult, ...]
    expected_scope_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        results = tuple(self.results)
        if any(not isinstance(result, HMCTuningCandidateSetResult) for result in results):
            raise TypeError("scope collection requires candidate-set results")
        keys = tuple((result.scope.scope_id, result.scope.search_id) for result in results)
        if len(keys) != len(set(keys)):
            raise ValueError("scope collection contains duplicate scope/search results")
        expected = tuple(str(value) for value in self.expected_scope_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected scope IDs must be unique")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "expected_scope_ids", expected)

    @property
    def complete(self) -> bool:
        if self.expected_scope_ids:
            by_scope = {result.scope.scope_id: result for result in self.results}
            return all(
                scope_id in by_scope and by_scope[scope_id].completion_status == "complete"
                for scope_id in self.expected_scope_ids
            )
        return all(result.completion_status == "complete" for result in self.results)

    def member(self, scope_id: str, candidate_id: str) -> HMCTuningCandidateRecord:
        for result in self.results:
            if result.scope.scope_id == scope_id:
                return result.replay_candidate(candidate_id)
        raise ValueError("scope is absent from the reporting collection")

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_tuning_scope_collection.v1",
            "scope_ids": tuple(result.scope.scope_id for result in self.results),
            "search_ids": tuple(result.scope.search_id for result in self.results),
            "expected_scope_ids": self.expected_scope_ids,
            "complete": self.complete,
            "results": tuple(result.payload() for result in self.results),
            "artifact_authority": False,
            "nominee": None,
            "nonclaims": (
                "read-only reporting collection",
                "no cross-scope chain-state pooling",
                "no new tuning or repair scheduling",
                "no statistical superiority claim",
            ),
        }


class HMCInfrastructureFailure(RuntimeError):
    """Adapter could not complete a work item and retry is disallowed."""


class HMCSharedInvalidity(RuntimeError):
    """The frozen scope is invalid for every remaining candidate."""


OutcomeProvider = Callable[[HMCWorkItem, HMCTuningCandidateRecord], Mapping[str, Any]]


def tune_hmc_candidate_set(
    scope: HMCCandidateSetScope,
    config: HMCControllerConfig,
    outcome_provider: OutcomeProvider,
    *,
    max_work_items: int | None = None,
) -> HMCTuningCandidateSetResult:
    """Run one bounded candidate-set controller invocation.

    Numerical adapters should use this boundary while the existing public
    numerical entry points are migrated.  It intentionally accepts an outcome
    provider rather than a raw sampler callback so the controller cannot create
    a second hidden validation lifecycle.
    """
    controller = HMCTuningCandidateSetController(scope, config)
    return controller.run(outcome_provider, max_work_items=max_work_items)


class HMCTuningCandidateSetController:
    """Deterministic single-scope controller used by numerical adapters."""

    def __init__(self, scope: HMCCandidateSetScope, config: HMCControllerConfig) -> None:
        self.scope = scope
        self.config = config
        self._candidates: dict[str, HMCTuningCandidateRecord] = {}
        self._states: dict[str, str] = {}
        self._work_items: dict[str, HMCWorkItem] = {}
        self._receipts: list[HMCVerificationReceipt] = []
        self._repairs: dict[str, HMCRepairAction] = {}
        self._deferred_repairs: list[str] = []
        self._accounting: list[Mapping[str, Any]] = []
        self._candidate_reserves: dict[str, int] = {}
        self._candidate_reserve_sources: dict[str, str] = {}
        self._repair_counts: dict[str, int] = {}
        self._creation_ordinal = 0
        self._work_ordinal = 0
        self._verification_ordinal: dict[str, int] = {}
        self._active_cohort: tuple[str, ...] = ()
        self._cohort_ordinal = 0
        self._available_units = config.total_budget_units
        self._spent_units = 0
        self._repair_reserve_remaining = config.repair_reserve_units
        self._scope_invalid = False
        self._infrastructure_paused = False
        self._repair_budget_blocked = False
        self._stop_reason: str | None = None
        self._admit_primary_candidates()

    @classmethod
    def from_result_payload(
        cls, payload: Mapping[str, Any]
    ) -> "HMCTuningCandidateSetController":
        """Reconstruct a controller without rebuilding or reordering work.

        The artifact loader performs the checksum and semantic validation before
        calling this method.  Reconstruction still re-issues every immutable
        record through its repository-owned constructor so a serialized caller
        cannot replace a candidate identity while resuming.
        """

        if payload.get("schema") != HMC_CANDIDATE_SET_RESULT_SCHEMA:
            raise ValueError("unsupported HMC candidate-set result schema")
        scope_payload = payload.get("scope")
        config_payload = payload.get("config")
        if not isinstance(scope_payload, Mapping) or not isinstance(config_payload, Mapping):
            raise ValueError("candidate-set result is missing scope or config")
        scope = HMCCandidateSetScope.from_payload(scope_payload)
        config = HMCControllerConfig.from_payload(config_payload)
        controller = object.__new__(cls)
        controller.scope = scope
        controller.config = config
        controller._candidates = {}
        controller._states = {
            str(key): str(value)
            for key, value in dict(payload.get("candidate_states", {})).items()
        }
        controller._work_items = {}
        controller._receipts = []
        controller._repairs = {}
        controller._deferred_repairs = []
        controller._accounting = [
            dict(event) for event in payload.get("accounting_events", ())
        ]
        controller._candidate_reserves = {}
        controller._candidate_reserve_sources = {}
        controller._repair_counts = {}
        controller._creation_ordinal = 0
        controller._work_ordinal = 0
        controller._verification_ordinal = {}
        controller._active_cohort = ()
        controller._cohort_ordinal = 0
        controller._available_units = int(payload.get("remaining_budget_units", 0))
        controller._spent_units = int(payload.get("budget_used_units", 0))
        controller._repair_reserve_remaining = config.repair_reserve_units
        controller._scope_invalid = payload.get("completion_status") == "shared_invalidity"
        controller._infrastructure_paused = payload.get("completion_status") == "paused_infrastructure"
        controller._repair_budget_blocked = payload.get("final_status") == "repair_budget_exhausted"
        controller._stop_reason = (
            "budget_bound"
            if payload.get("completion_status") == "partial_budget"
            else None
        )

        candidate_payloads = payload.get("candidates", ())
        if not isinstance(candidate_payloads, (tuple, list)):
            raise ValueError("candidate-set result candidates must be a sequence")
        for item in candidate_payloads:
            if not isinstance(item, Mapping):
                raise ValueError("candidate-set result has an invalid candidate")
            candidate = HMCTuningCandidateRecord.from_payload(scope, item)
            controller._candidates[candidate.candidate_id] = candidate
            controller._creation_ordinal = max(
                controller._creation_ordinal, candidate.creation_ordinal
            )
            controller._repair_counts.setdefault(candidate.candidate_family_id, 0)
            if candidate.parent_candidate_id is not None:
                controller._repair_counts[candidate.candidate_family_id] += 1

        work_payloads = payload.get("work_items", ())
        if not isinstance(work_payloads, (tuple, list)):
            raise ValueError("candidate-set result work_items must be a sequence")
        for item in work_payloads:
            if not isinstance(item, Mapping):
                raise ValueError("candidate-set result has an invalid work item")
            work = HMCWorkItem.from_payload(item)
            candidate = controller._candidates.get(work.candidate_id)
            if candidate is None or work.candidate_record_hash != candidate.candidate_record_hash:
                raise ValueError("work item candidate identity mismatch")
            controller._work_items[work.work_item_id] = work
            controller._work_ordinal = max(controller._work_ordinal, work.ordinal)
            if work.verification_attempt_id is not None:
                try:
                    attempt_number = int(str(work.verification_attempt_id).rsplit(":", 1)[-1])
                except ValueError:
                    attempt_number = 1
                controller._verification_ordinal[work.candidate_id] = max(
                    controller._verification_ordinal.get(work.candidate_id, 0),
                    attempt_number,
                )
            if work.cohort_id != f"{scope.scope_id}:cohort:pending":
                try:
                    cohort_number = int(str(work.cohort_id).rsplit(":", 1)[-1])
                except ValueError:
                    cohort_number = 0
                controller._cohort_ordinal = max(controller._cohort_ordinal, cohort_number)

        for item in payload.get("verification_receipts", ()):
            if not isinstance(item, Mapping):
                raise ValueError("candidate-set result has an invalid receipt")
            controller._receipts.append(HMCVerificationReceipt.from_payload(item))
        for item in payload.get("repair_actions", ()):
            if not isinstance(item, Mapping):
                raise ValueError("candidate-set result has an invalid repair action")
            action = HMCRepairAction.from_payload(item)
            controller._repairs[action.repair_action_id] = action
            if action.execution_status == "not_started" and action.not_executed_reason in (
                None,
                "",
            ):
                controller._deferred_repairs.append(action.repair_action_id)

        # Fold the append-only accounting ledger back into the reservation
        # oracle. This keeps resume independent of object identity or process
        # lifetime and detects inconsistent budget fields early.
        for event in controller._accounting:
            event_name = event.get("event")
            candidate_id = event.get("candidate_id")
            if not candidate_id:
                continue
            units = int(event.get("units", 0))
            if event_name == "reserve_allocated":
                controller._candidate_reserves[candidate_id] = (
                    controller._candidate_reserves.get(candidate_id, 0) + units
                )
                controller._candidate_reserve_sources[candidate_id] = str(
                    event.get("source", "initial_reserve")
                )
                if event.get("source") == "repair_reserve":
                    controller._repair_reserve_remaining -= units
            elif event_name == "work_charged":
                controller._candidate_reserves[candidate_id] = (
                    controller._candidate_reserves.get(candidate_id, 0) - units
                )
            elif event_name == "reserve_released":
                source = controller._candidate_reserve_sources.pop(candidate_id, None)
                controller._candidate_reserves[candidate_id] = max(
                    0,
                    controller._candidate_reserves.get(candidate_id, 0) - units,
                )
                if source == "repair_reserve":
                    controller._repair_reserve_remaining = min(
                        config.repair_reserve_units,
                        controller._repair_reserve_remaining + units,
                    )
        controller._candidate_reserves = {
            key: value for key, value in controller._candidate_reserves.items() if value
        }
        reserved = sum(controller._candidate_reserves.values())
        expected_remaining = controller._available_units
        if expected_remaining != config.total_budget_units - controller._spent_units:
            # Older artifacts did not record a complete budget ledger. They are
            # readable historical evidence but cannot be resumed as authority.
            raise ValueError("candidate-set result has inconsistent budget accounting")
        controller._available_units = expected_remaining - reserved

        pending = [
            work
            for work in controller._work_items.values()
            if work.status in {"pending", "interrupted"}
            and work.cohort_id != f"{scope.scope_id}:cohort:pending"
        ]
        if pending:
            earliest = min(work.ordinal for work in pending)
            cohort_id = next(
                work.cohort_id for work in sorted(pending, key=lambda item: item.ordinal)
                if work.ordinal == earliest
            )
            controller._active_cohort = tuple(
                work.work_item_id
                for work in sorted(pending, key=lambda item: item.ordinal)
                if work.cohort_id == cohort_id
            )
        return controller

    @property
    def candidates(self) -> tuple[HMCTuningCandidateRecord, ...]:
        return tuple(sorted(self._candidates.values(), key=lambda item: item.creation_ordinal))

    def _admit_primary_candidates(self) -> None:
        expected = sum(len(epsilons) for _, epsilons in self.config.epsilon_by_l)
        if expected * self.config.candidate_reserve_units > self.config.total_budget_units:
            raise ValueError(
                "declared primary candidate cohort does not fit total budget"
            )
        for leapfrog in self.config.primary_l_grid:
            epsilons = dict(self.config.epsilon_by_l)[leapfrog]
            for epsilon in epsilons:
                candidate = self._new_candidate(leapfrog, epsilon)
                self._enqueue(candidate, "measurement", priority=1, repair_action_id=None)

    def add_exploration_candidate(self, leapfrog_steps: int, epsilon: float) -> str:
        """Admit a separately identified candidate after the primary cohort."""
        candidate = self._new_candidate(leapfrog_steps, epsilon)
        if self._states[candidate.candidate_id] == "proposed":
            self._enqueue(candidate, "measurement", priority=1, repair_action_id=None)
        else:
            self._accounting.append(
                {
                    "event": "exploration_candidate_pending",
                    "candidate_id": candidate.candidate_id,
                    "reason": "candidate_reserve_exhausted",
                }
            )
        return candidate.candidate_id

    def _new_candidate(
        self,
        leapfrog_steps: int,
        epsilon: float,
        *,
        candidate_family_id: str | None = None,
        parent_candidate_id: str | None = None,
    ) -> HMCTuningCandidateRecord:
        self._creation_ordinal += 1
        candidate = HMCTuningCandidateRecord.create(
            self.scope,
            leapfrog_steps=leapfrog_steps,
            epsilon=epsilon,
            creation_ordinal=self._creation_ordinal,
            candidate_family_id=candidate_family_id,
            parent_candidate_id=parent_candidate_id,
        )
        # Repair children are proposed now but receive their fresh reservation
        # only when the active cohort closes. This prevents a child from being
        # allocated once at proposal time and again at deferred dispatch.
        allocated = (
            self._allocate_candidate_reserve(candidate.candidate_id, "initial")
            if parent_candidate_id is None
            else False
        )
        self._candidates[candidate.candidate_id] = candidate
        self._states[candidate.candidate_id] = (
            "proposed" if allocated else "proposal_budget_pending"
        )
        self._repair_counts.setdefault(candidate.candidate_family_id, 0)
        return candidate

    def _allocate_candidate_reserve(self, candidate_id: str, source: str) -> bool:
        units = self.config.candidate_reserve_units
        if units > self._available_units:
            self._accounting.append({"event": "reserve_allocation_failed", "candidate_id": candidate_id, "units": units, "source": source})
            return False
        if source == "repair":
            if units <= self._repair_reserve_remaining:
                allocation_source = "repair_reserve"
                self._repair_reserve_remaining -= units
            elif not self.config.allow_repair_from_free_pool:
                self._accounting.append(
                    {
                        "event": "reserve_allocation_failed",
                        "candidate_id": candidate_id,
                        "units": units,
                        "source": source,
                        "reason": "repair_reserve_exhausted",
                    }
                )
                return False
            else:
                allocation_source = "free_pool"
        else:
            allocation_source = "initial_reserve"
        self._available_units -= units
        self._candidate_reserves[candidate_id] = units
        self._candidate_reserve_sources[candidate_id] = allocation_source
        self._accounting.append({"event": "reserve_allocated", "candidate_id": candidate_id, "units": units, "source": allocation_source})
        return True

    def _charge_work(self, work: HMCWorkItem, candidate: HMCTuningCandidateRecord) -> bool:
        """Consume one unit from the candidate's reservation at dispatch.

        Candidate reservations are held at admission so an active cohort cannot
        steal another member's budget.  Charging at dispatch makes the result's
        remaining budget equal to total budget minus work actually executed.
        """

        units = int(work.reservation_units)
        reserved = int(self._candidate_reserves.get(candidate.candidate_id, 0))
        if units <= 0 or reserved < units:
            self._accounting.append(
                {
                    "event": "work_charge_failed",
                    "work_item_id": work.work_item_id,
                    "candidate_id": candidate.candidate_id,
                    "units": units,
                    "reserved_units": reserved,
                    "reason": "candidate_reserve_exhausted",
                }
            )
            return False
        self._candidate_reserves[candidate.candidate_id] = reserved - units
        self._spent_units += units
        self._accounting.append(
            {
                "event": "work_charged",
                "work_item_id": work.work_item_id,
                "candidate_id": candidate.candidate_id,
                "units": units,
                "remaining_candidate_reserve": reserved - units,
            }
        )
        return True

    def _release_candidate_reserve(self, candidate_id: str, reason: str) -> None:
        units = self._candidate_reserves.pop(candidate_id, 0)
        source = self._candidate_reserve_sources.pop(candidate_id, None)
        if units:
            self._available_units += units
            if source == "repair_reserve":
                self._repair_reserve_remaining = min(
                    self.config.repair_reserve_units,
                    self._repair_reserve_remaining + units,
                )
            self._accounting.append({"event": "reserve_released", "candidate_id": candidate_id, "units": units, "reason": reason})

    def _enqueue(
        self,
        candidate: HMCTuningCandidateRecord,
        stage: Literal["measurement", "verification"],
        *,
        priority: int,
        repair_action_id: str | None,
    ) -> HMCWorkItem | None:
        self._work_ordinal += 1
        attempt_id = None
        if stage == "verification":
            next_attempt = self._verification_ordinal.get(candidate.candidate_id, 0) + 1
            self._verification_ordinal[candidate.candidate_id] = next_attempt
            attempt_id = f"{candidate.candidate_id}:verification:{next_attempt:03d}"
        work = HMCWorkItem(
            work_item_id=f"{self.scope.scope_id}:{self.scope.search_id}:work:{self._work_ordinal:06d}",
            candidate_id=candidate.candidate_id,
            candidate_record_hash=candidate.candidate_record_hash,
            candidate_family_id=candidate.candidate_family_id,
            stage=stage,
            cohort_id=f"{self.scope.scope_id}:cohort:pending",
            ordinal=self._work_ordinal,
            priority=priority,
            repair_action_id=repair_action_id,
            verification_attempt_id=attempt_id,
        )
        self._work_items[work.work_item_id] = work
        self._accounting.append({"event": "work_reserved", "work_item_id": work.work_item_id, "candidate_id": candidate.candidate_id, "units": work.reservation_units})
        return work

    def _set_work(self, work: HMCWorkItem, **changes: Any) -> HMCWorkItem:
        updated = replace(work, **changes)
        self._work_items[work.work_item_id] = updated
        return updated

    def _next_cohort(self) -> tuple[str, ...]:
        if self._active_cohort:
            pending = tuple(
                work_id
                for work_id in self._active_cohort
                if self._work_items[work_id].status in {"pending", "interrupted"}
            )
            if pending:
                return pending
            self._active_cohort = ()
        ready = [
            work
            for work in self._work_items.values()
            if work.status in {"pending", "interrupted"}
        ]
        if not ready:
            return ()
        min_stage = min(_STAGE_ORDER[work.stage] for work in ready)
        stage_ready = [work for work in ready if _STAGE_ORDER[work.stage] == min_stage]
        min_priority = min(work.priority for work in stage_ready)
        selected = sorted(
            (work for work in stage_ready if work.priority == min_priority),
            key=lambda item: item.ordinal,
        )
        self._cohort_ordinal += 1
        cohort_id = f"{self.scope.scope_id}:cohort:{self._cohort_ordinal:04d}"
        ids = []
        for work in selected:
            self._work_items[work.work_item_id] = replace(work, cohort_id=cohort_id)
            ids.append(work.work_item_id)
        self._active_cohort = tuple(ids)
        return self._active_cohort

    def _cohort_closed(self) -> bool:
        return bool(self._active_cohort) and all(
            self._work_items[work_id].status == "completed" for work_id in self._active_cohort
        )

    def _materialize_deferred_repairs(self) -> None:
        if not self._cohort_closed():
            return
        self._active_cohort = ()
        for action_id in tuple(self._deferred_repairs):
            action = self._repairs[action_id]
            if action.execution_status != "not_started":
                self._deferred_repairs.remove(action_id)
                continue
            child = self._candidates[action.child_candidate_id]
            if not self._allocate_candidate_reserve(child.candidate_id, "repair"):
                self._repair_budget_blocked = True
                self._states[child.candidate_id] = "proposal_budget_pending"
                self._repairs[action_id] = replace(
                    action,
                    not_executed_reason="repair_budget_exhausted",
                )
                self._deferred_repairs.remove(action_id)
                self._release_candidate_reserve(child.candidate_id, "unfunded_repair")
                continue
            allocation_source = next(
                (
                    str(event["source"])
                    for event in reversed(self._accounting)
                    if event.get("event") == "reserve_allocated"
                    and event.get("candidate_id") == child.candidate_id
                ),
                None,
            )
            self._repairs[action_id] = replace(
                action,
                allocation_source=allocation_source,
            )
            self._states[child.candidate_id] = "proposed"
            self._enqueue(child, "measurement", priority=0, repair_action_id=action_id)
            self._deferred_repairs.remove(action_id)

    def _parse_observation(self, raw: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(raw, Mapping):
            raise TypeError("outcome provider must return a mapping")
        decision = str(raw.get("decision", "passed"))
        if decision not in {
            "passed",
            "acceptance_in_band",
            "inconclusive_evidence",
            "inconclusive_conflict",
            "repair_step_higher",
            "repair_step_lower",
            "promotion_failed",
            "failed",
        }:
            raise ValueError(f"unknown tuning decision: {decision}")
        return {
            **dict(raw),
            "decision": decision,
            "hard_vetoes": tuple(str(item) for item in raw.get("hard_vetoes", ())),
        }

    def _receipt(self, work: HMCWorkItem, candidate: HMCTuningCandidateRecord, observation: Mapping[str, Any]) -> HMCVerificationReceipt:
        attempt = work.verification_attempt_id or f"{work.work_item_id}:attempt"
        stream = str(observation.get("stream_id", f"{candidate.candidate_id}:{work.stage}:stream"))
        draw = tuple(int(item) for item in observation.get("draw_range", (0, 0)))
        seed = tuple(int(item) for item in observation.get("seed_lineage", (candidate.creation_ordinal, work.ordinal)))
        return HMCVerificationReceipt(
            verification_attempt_id=attempt,
            candidate_id=candidate.candidate_id,
            candidate_record_hash=candidate.candidate_record_hash,
            exact_l=candidate.leapfrog_steps,
            epsilon=candidate.epsilon,
            mass_signature=candidate.mass_signature,
            stream_id=stream,
            draw_range=draw,
            seed_lineage=seed,
            decision=str(observation["decision"]),
            acceptance=None if observation.get("acceptance") is None else float(observation["acceptance"]),
            hard_vetoes=tuple(observation["hard_vetoes"]),
        )

    def _request_repair(self, candidate: HMCTuningCandidateRecord, receipt: HMCVerificationReceipt) -> None:
        direction = receipt.decision
        count = self._repair_counts[candidate.candidate_family_id]
        if count >= self.scope.max_repairs_per_family:
            self._states[candidate.candidate_id] = "promotion_failed"
            self._accounting.append({"event": "repair_limit_exhausted", "candidate_id": candidate.candidate_id, "candidate_family_id": candidate.candidate_family_id})
            return
        multiplier = self.scope.repair_factor if direction == "repair_step_higher" else 1.0 / self.scope.repair_factor
        new_epsilon = candidate.epsilon * multiplier
        lo, hi = self.scope.epsilon_domain
        if not lo <= new_epsilon <= hi or new_epsilon == candidate.epsilon:
            self._states[candidate.candidate_id] = "promotion_failed"
            self._accounting.append({"event": "repair_limit_exhausted", "candidate_id": candidate.candidate_id, "reason": "epsilon_domain"})
            return
        self._creation_ordinal += 1
        child = HMCTuningCandidateRecord.create(
            self.scope,
            leapfrog_steps=candidate.leapfrog_steps,
            epsilon=new_epsilon,
            creation_ordinal=self._creation_ordinal,
            candidate_family_id=candidate.candidate_family_id,
            parent_candidate_id=candidate.candidate_id,
        )
        self._candidates[child.candidate_id] = child
        self._states[child.candidate_id] = "proposal_budget_pending"
        self._repair_counts[candidate.candidate_family_id] = count + 1
        action_id = f"{self.scope.scope_id}:{self.scope.search_id}:repair:{len(self._repairs) + 1:04d}"
        action = HMCRepairAction(
            repair_action_id=action_id,
            parent_candidate_id=candidate.candidate_id,
            child_candidate_id=child.candidate_id,
            candidate_family_id=candidate.candidate_family_id,
            source_verification_hash=_sha256(receipt.payload()),
            old_epsilon=candidate.epsilon,
            new_epsilon=new_epsilon,
            exact_l=candidate.leapfrog_steps,
            mass_signature=candidate.mass_signature,
            direction=direction,  # type: ignore[arg-type]
            execution_status="not_started",
            verification_status="pending",
            qualified_repair_status="not_executed_with_reason",
        )
        self._repairs[action_id] = action
        self._deferred_repairs.append(action_id)
        self._states[candidate.candidate_id] = "promotion_failed"
        self._accounting.append({"event": "repair_proposed", "repair_action_id": action_id, "parent_candidate_id": candidate.candidate_id, "child_candidate_id": child.candidate_id})

    def _apply_observation(self, work: HMCWorkItem, observation: Mapping[str, Any]) -> None:
        candidate = self._candidates[work.candidate_id]
        parsed = self._parse_observation(observation)
        if work.stage == "measurement":
            action_id = work.repair_action_id
            if parsed["hard_vetoes"] or parsed["decision"] in {"failed", "promotion_failed"}:
                self._states[candidate.candidate_id] = "promotion_failed"
                if action_id is not None:
                    action = self._repairs[action_id]
                    self._repairs[action_id] = replace(
                        action,
                        execution_status="executed",
                        verification_status="skipped",
                        qualified_repair_status="not_executed_with_reason",
                        not_executed_reason="repair_verification_failed",
                    )
                self._release_candidate_reserve(candidate.candidate_id, "measurement_failure")
            else:
                self._states[candidate.candidate_id] = "screened"
                if action_id is not None:
                    action = self._repairs[action_id]
                    self._repairs[action_id] = replace(
                        action,
                        execution_status="executed",
                        verification_status="pending",
                    )
                self._enqueue(candidate, "verification", priority=work.priority, repair_action_id=work.repair_action_id)
            return
        receipt = self._receipt(work, candidate, parsed)
        self._receipts.append(receipt)
        action_id = work.repair_action_id
        if action_id is not None:
            action = self._repairs[action_id]
            if parsed["hard_vetoes"] or parsed["decision"] in {"failed", "promotion_failed"}:
                self._repairs[action_id] = replace(action, execution_status="executed", verification_status="failed", qualified_repair_status="not_executed_with_reason", not_executed_reason="repair_verification_failed")
            elif parsed["decision"] in _DIRECTIONAL:
                self._repairs[action_id] = replace(action, execution_status="executed", verification_status="pending", qualified_repair_status="not_executed_with_reason", not_executed_reason="repair_not_scheduled")
                self._request_repair(candidate, receipt)
            else:
                self._repairs[action_id] = replace(action, execution_status="executed", verification_status="passed", qualified_repair_status="executed_and_verified", not_executed_reason=None)
                self._states[candidate.candidate_id] = "verified"
            self._release_candidate_reserve(candidate.candidate_id, "repair_verification_terminal")
            return
        if parsed["hard_vetoes"] or parsed["decision"] in {"failed", "promotion_failed"}:
            self._states[candidate.candidate_id] = "promotion_failed"
            self._release_candidate_reserve(candidate.candidate_id, "verification_failure")
        elif parsed["decision"] in _DIRECTIONAL:
            self._request_repair(candidate, receipt)
            self._release_candidate_reserve(candidate.candidate_id, "directional_repair")
        elif parsed["decision"] in {"inconclusive_evidence", "inconclusive_conflict"}:
            self._states[candidate.candidate_id] = "validating"
            self._release_candidate_reserve(candidate.candidate_id, "inconclusive_terminal")
        else:
            self._states[candidate.candidate_id] = "verified"
            self._release_candidate_reserve(candidate.candidate_id, "verified")

    def run(self, outcome_provider: OutcomeProvider, *, max_work_items: int | None = None) -> HMCTuningCandidateSetResult:
        """Dispatch deterministic cohorts until complete or a declared bound."""
        if not callable(outcome_provider):
            raise TypeError("outcome_provider must be callable")
        if self._scope_invalid:
            return self.result()
        dispatched = 0
        self._infrastructure_paused = False
        self._stop_reason = None
        while True:
            cohort = self._next_cohort()
            if not cohort:
                break
            for work_id in cohort:
                work = self._work_items[work_id]
                if work.status == "completed":
                    continue
                if max_work_items is not None and dispatched >= int(max_work_items):
                    self._stop_reason = "budget_bound"
                    return self.result()
                candidate = self._candidates[work.candidate_id]
                if not self._charge_work(work, candidate):
                    self._stop_reason = "budget_bound"
                    self._set_work(self._work_items[work_id], status="interrupted")
                    return self.result()
                self._set_work(work, status="running")
                if work.repair_action_id is not None:
                    action = self._repairs[work.repair_action_id]
                    self._repairs[work.repair_action_id] = replace(action, execution_status="running")
                dispatched += 1
                try:
                    observation = outcome_provider(work, candidate)
                except HMCSharedInvalidity:
                    self._scope_invalid = True
                    self._stop_reason = "shared_invalidity"
                    self._set_work(self._work_items[work_id], status="interrupted")
                    return self.result()
                except HMCInfrastructureFailure:
                    self._infrastructure_paused = True
                    self._stop_reason = "repair_infrastructure_failure" if work.repair_action_id else "paused_infrastructure"
                    self._set_work(self._work_items[work_id], status="interrupted")
                    return self.result()
                self._apply_observation(work, observation)
                self._set_work(self._work_items[work_id], status="completed")
            self._materialize_deferred_repairs()
        return self.result()

    def result(self) -> HMCTuningCandidateSetResult:
        pending = tuple(
            work.work_item_id
            for work in sorted(self._work_items.values(), key=lambda item: item.ordinal)
            if work.status in {"pending", "running", "interrupted"}
        )
        if self._scope_invalid:
            completion = "shared_invalidity"
            final = "shared_invalidity"
        elif self._infrastructure_paused:
            completion = "paused_infrastructure"
            final = "paused_infrastructure"
        elif (
            pending
            or self._deferred_repairs
            or self._repair_budget_blocked
            or self._stop_reason == "budget_bound"
            or any(state == "proposal_budget_pending" for state in self._states.values())
        ):
            completion = "partial_budget"
            final = "repair_budget_exhausted" if self._repair_budget_blocked else "partial_budget"
        else:
            completion = "complete"
            final = "complete"
        verified = () if self._scope_invalid else tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if self._states.get(candidate.candidate_id) == "verified"
        )
        viable = () if self._scope_invalid else tuple(
            candidate.candidate_id
            for candidate in self.candidates
            if self._states.get(candidate.candidate_id)
            in {"screened", "validating", "verified"}
        )
        return HMCTuningCandidateSetResult(
            scope=self.scope,
            config=self.config,
            candidates=self.candidates,
            candidate_states=dict(self._states),
            work_items=tuple(sorted(self._work_items.values(), key=lambda item: item.ordinal)),
            verification_receipts=tuple(self._receipts),
            repair_actions=tuple(self._repairs.values()),
            accounting_events=tuple(self._accounting),
            verified_candidate_ids=verified,
            viable_candidate_ids=viable,
            final_status=final,
            completion_status=completion,
            remaining_budget_units=(
                self._available_units + sum(self._candidate_reserves.values())
            ),
            resume_pending_work_item_ids=pending,
            budget_used_units=self._spent_units,
            reserved_budget_units=sum(self._candidate_reserves.values()),
        )


__all__ = [
    "HMC_CANDIDATE_SET_RESULT_SCHEMA",
    "HMC_CANDIDATE_SET_TUNING_SCHEMA",
    "HMCControllerConfig",
    "HMCInfrastructureFailure",
    "HMCRepairAction",
    "HMCSharedInvalidity",
    "HMCTuningCandidateRecord",
    "HMCTuningCandidateSetController",
    "HMCTuningCandidateSetResult",
    "HMCTuningScopeCollection",
    "HMCCandidateSetScope",
    "HMCVerificationReceipt",
    "HMCWorkItem",
    "tune_hmc_candidate_set",
]
