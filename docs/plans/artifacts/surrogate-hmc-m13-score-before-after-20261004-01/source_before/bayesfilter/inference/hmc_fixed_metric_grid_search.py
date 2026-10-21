"""Opt-in broad fixed-metric HMC trajectory/step-size search policy.

This module does not implement HMC transitions.  Caller-supplied TensorFlow /
TensorFlow Probability callbacks own fixed-mass dual averaging and fixed-kernel
screens.  The module owns the non-directional ``L`` grid, independent seed and
step-size lineage, failure scope, survivor preservation, and one midpoint
refinement round.

All draws are tuning draws and must be discarded by the callbacks.  A survivor
is viable under the bounded :class:`HMCAcceptancePolicy` heuristic only; it is
not evidence of convergence, superiority, or retained-sampling readiness.
Fresh candidate confirmation is deliberately outside ``run_fixed_metric_grid_search``;
callers must invoke ``confirm_fixed_metric_candidate`` explicitly when that
separate phase is authorized.  The optional evidence extension is still tuning
evidence, not confirmation.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import math
import multiprocessing
import numbers
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

from bayesfilter.inference.hmc_verification import (
    HMCAcceptanceEvidence,
    HMCAcceptancePolicy,
    hmc_acceptance_evidence_from_payload,
)


DEFAULT_L_GRID = (3, 5, 9, 13, 18, 25)
MIN_LEAPFROG_STEPS = 1
MAX_LEAPFROG_STEPS = 25
REPLICATION_COUNT = 3
GRID_EXECUTION_MODES = ("serial", "process_parallel")
AGGREGATE_EVIDENCE_PHASES = ("nomination", "confirmation")
AGGREGATE_EVIDENCE_DISPOSITIONS = (
    "hard_rejected",
    "needs_lower_epsilon",
    "needs_higher_epsilon",
    "confirmation_required",
    "provisional_viable",
    "unresolved_budget",
)
DEFAULT_CONFIRMATION_RESULTS = 256
DEFAULT_WORKING_T_CRITICAL_90_DF11 = 1.7958848187036691

GRID_SEARCH_NONCLAIMS = (
    "bounded fixed-metric HMC tuning search only",
    "survival is robust heuristic viability, not a confidence-test result",
    "no stochastic candidate ranking",
    "no posterior convergence claim",
    "no mass-matrix adequacy claim",
    "no retained-sampling readiness claim",
    "no sampler superiority or default-readiness claim",
)

_CANDIDATE_TUNE_REASONS = frozenset(
    {
        "nonfinite_candidate_state",
        "nonfinite_log_accept_ratio",
        "nonfinite_target_log_prob",
        "nonfinite_target_score",
        "nonfinite_adapted_step_size",
        "target_finite_reject",
        "target_status_telemetry_failure",
    }
)
_CANDIDATE_SCREEN_REASONS = frozenset(
    {
        "nonfinite_candidate_state",
        "nonfinite_log_accept_ratio",
        "nonfinite_target_log_prob",
        "nonfinite_target_score",
        "target_finite_reject",
        "target_status_telemetry_failure",
        "candidate_screen_execution_failed",
    }
)
_SHARED_REASONS = frozenset(
    {
        "shared_schema_invalid",
        "shared_callback_invalid",
        "shared_adapter_invalid",
        "required_standard_trace_missing",
        "required_target_status_telemetry_missing",
        "coordinate_signature_mismatch",
        "metric_signature_mismatch",
        "start_bank_signature_mismatch",
        "common_state_signature_mismatch",
        "candidate_identity_mismatch",
        "seed_lineage_mismatch",
        "step_size_lineage_mismatch",
        "acceptance_policy_mismatch",
        "untyped_callback_failure",
    }
)


def _strict_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer scalar")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_seed(value: Any, *, name: str) -> tuple[int, int]:
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two integers") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two integers")
    return tuple(
        _strict_integer(item, name=f"{name} item", minimum=0) for item in items
    )


def _nonempty(value: Any, *, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _reason_tuple(
    values: Sequence[str], *, allowed: frozenset[str], name: str
) -> tuple[str, ...]:
    reasons = tuple(dict.fromkeys(str(item) for item in values))
    if not reasons or not set(reasons).issubset(allowed):
        raise ValueError(f"{name} contains an unsupported reason code")
    return reasons


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _signature(label: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"label": label, "payload": _json_ready(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FixedMetricSearchLineage:
    """Content signatures frozen across every tune and screen callback."""

    coordinate_signature: str
    metric_signature: str
    private_start_bank_content_signature: str
    common_state_signature: str

    def __post_init__(self) -> None:
        for name in (
            "coordinate_signature",
            "metric_signature",
            "private_start_bank_content_signature",
            "common_state_signature",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(getattr(self, name), name=name),
            )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.hmc_fixed_metric_search_lineage.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "coordinate_signature": self.coordinate_signature,
            "metric_signature": self.metric_signature,
            "private_start_bank_content_signature": (
                self.private_start_bank_content_signature
            ),
            "common_state_signature": self.common_state_signature,
        }


@dataclass(frozen=True)
class FixedMetricGridSearchConfig:
    """Reviewed broad-grid controls with optional one-round refinement."""

    l_grid: tuple[int, ...] = DEFAULT_L_GRID
    root_seed: tuple[int, int] = (20260719, 7300)
    initial_step_size: float = 0.05
    screen_num_results: int = 64
    extension_num_results: int = 128
    refinement_rounds: int = 1

    def __post_init__(self) -> None:
        grid = tuple(
            _strict_integer(item, name="l_grid item", minimum=MIN_LEAPFROG_STEPS)
            for item in self.l_grid
        )
        if len(grid) < 3:
            raise ValueError("l_grid must contain at least three distinct values")
        if len(set(grid)) != len(grid):
            raise ValueError("l_grid must contain distinct values")
        if any(item > MAX_LEAPFROG_STEPS for item in grid):
            raise ValueError("l_grid cannot exceed the reviewed L=25 bound")
        root_seed = _strict_seed(self.root_seed, name="root_seed")
        initial_step = float(self.initial_step_size)
        if not math.isfinite(initial_step) or initial_step <= 0.0:
            raise ValueError("initial_step_size must be positive and finite")
        screen_results = _strict_integer(
            self.screen_num_results,
            name="screen_num_results",
            minimum=1,
        )
        extension_results = _strict_integer(
            self.extension_num_results,
            name="extension_num_results",
            minimum=1,
        )
        if extension_results <= screen_results:
            raise ValueError("extension_num_results must exceed screen_num_results")
        rounds = _strict_integer(
            self.refinement_rounds,
            name="refinement_rounds",
            minimum=0,
        )
        if rounds not in {0, 1}:
            raise ValueError("the reviewed search authorizes zero or one refinement round")
        object.__setattr__(self, "l_grid", grid)
        object.__setattr__(self, "root_seed", root_seed)
        object.__setattr__(self, "initial_step_size", initial_step)
        object.__setattr__(self, "screen_num_results", screen_results)
        object.__setattr__(self, "extension_num_results", extension_results)
        object.__setattr__(self, "refinement_rounds", rounds)

    def payload(self) -> Mapping[str, Any]:
        return {
            "l_grid": self.l_grid,
            "root_seed": self.root_seed,
            "initial_step_size": self.initial_step_size,
            "screen_num_results": self.screen_num_results,
            "extension_num_results": self.extension_num_results,
            "replication_count": REPLICATION_COUNT,
            "refinement_rounds": self.refinement_rounds,
            "minimum_leapfrog_steps": MIN_LEAPFROG_STEPS,
            "maximum_leapfrog_steps": MAX_LEAPFROG_STEPS,
        }


@dataclass(frozen=True)
class FixedMetricGridExecutionConfig:
    """Execution topology for a fixed-metric candidate grid.

    Process mode uses fresh ``spawn`` workers.  The application factory is
    imported inside each worker after the declared environment is installed;
    it must construct and return that candidate's tune and screen callbacks.
    """

    mode: str = "serial"
    max_workers: int | None = None
    worker_factory_locator: str | None = None
    worker_environment: tuple[tuple[str, str], ...] = ()
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        mode = str(self.mode)
        if mode not in GRID_EXECUTION_MODES:
            raise ValueError(f"mode must be one of {GRID_EXECUTION_MODES}")
        object.__setattr__(self, "mode", mode)
        if str(self.start_method) != "spawn":
            raise ValueError("fixed-metric process execution requires spawn")
        object.__setattr__(self, "start_method", "spawn")

        workers = 1 if self.max_workers is None and mode == "serial" else self.max_workers
        workers = len(DEFAULT_L_GRID) if workers is None else _strict_integer(
            workers,
            name="max_workers",
            minimum=1,
        )
        if mode == "serial" and workers != 1:
            raise ValueError("serial fixed-metric execution requires max_workers=1")
        object.__setattr__(self, "max_workers", workers)

        locator = (
            None
            if self.worker_factory_locator is None
            else str(self.worker_factory_locator)
        )
        if mode == "serial":
            if locator is not None or self.worker_environment:
                raise ValueError(
                    "serial execution cannot declare a process worker factory or environment"
                )
            object.__setattr__(self, "worker_factory_locator", None)
            object.__setattr__(self, "worker_environment", ())
            return
        if not locator or ":" not in locator:
            raise ValueError(
                "process_parallel execution requires a module:factory locator"
            )
        module_name, attribute_path = locator.split(":", 1)
        if not module_name or not attribute_path:
            raise ValueError("worker_factory_locator must be module:factory")
        object.__setattr__(self, "worker_factory_locator", locator)

        try:
            environment_items = tuple(
                (str(key), str(value)) for key, value in self.worker_environment
            )
        except (TypeError, ValueError) as error:
            raise ValueError("worker_environment must contain key/value pairs") from error
        if any(not key for key, _ in environment_items):
            raise ValueError("worker_environment keys must be nonempty")
        if len({key for key, _ in environment_items}) != len(environment_items):
            raise ValueError("worker_environment keys must be unique")
        environment = dict(environment_items)
        cuda_visible = environment.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is None or not cuda_visible:
            raise ValueError(
                "process_parallel execution requires explicit CUDA_VISIBLE_DEVICES"
            )
        if (
            cuda_visible != "-1"
            and environment.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true"
        ):
            raise ValueError(
                "GPU process workers require TF_FORCE_GPU_ALLOW_GROWTH=true"
            )
        object.__setattr__(
            self,
            "worker_environment",
            tuple(sorted(environment_items)),
        )

    def payload(self) -> Mapping[str, Any]:
        environment = dict(self.worker_environment)
        return {
            "mode": self.mode,
            "max_workers": self.max_workers,
            "start_method": self.start_method,
            "worker_factory_locator": self.worker_factory_locator,
            "worker_environment_keys": tuple(sorted(environment)),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "tf_force_gpu_allow_growth": environment.get(
                "TF_FORCE_GPU_ALLOW_GROWTH"
            ),
        }


@dataclass(frozen=True)
class FixedMetricCandidateRunners:
    """Application-owned tune/screen callbacks constructed inside one worker."""

    tune_runner: Any
    screen_runner: Any

    def __post_init__(self) -> None:
        if not callable(self.tune_runner) or not callable(self.screen_runner):
            raise TypeError("candidate runners must both be callable")


@dataclass(frozen=True)
class FixedMetricCandidateWorkerRequest:
    """Complete immutable input supplied to an application worker factory."""

    round_index: int
    num_leapfrog_steps: int
    config: FixedMetricGridSearchConfig
    lineage: FixedMetricSearchLineage
    acceptance_policy: HMCAcceptancePolicy


@dataclass(frozen=True)
class FixedMetricCandidateWorkerOutcome:
    """Typed cross-process envelope; raw samples and states are never included."""

    status: str
    candidate: FixedMetricCandidateRecord | None = None
    shared_invalidity_reasons: tuple[str, ...] = ()
    message: str | None = None

    def __post_init__(self) -> None:
        allowed = {
            "candidate_complete",
            "shared_execution_invalid",
            "target_veto",
            "resource_closeout",
        }
        status = str(self.status)
        if status not in allowed:
            raise ValueError("worker outcome status is invalid")
        object.__setattr__(self, "status", status)
        if status == "candidate_complete":
            if not isinstance(self.candidate, FixedMetricCandidateRecord):
                raise ValueError("completed worker outcome requires a candidate")
            if self.shared_invalidity_reasons or self.message is not None:
                raise ValueError("completed worker outcome cannot carry failure fields")
            return
        if self.candidate is not None:
            raise ValueError("failed worker outcome cannot carry a candidate")
        if status == "shared_execution_invalid":
            reasons = _reason_tuple(
                self.shared_invalidity_reasons,
                allowed=_SHARED_REASONS,
                name="worker shared invalidity",
            )
            object.__setattr__(self, "shared_invalidity_reasons", reasons)
        elif self.shared_invalidity_reasons:
            raise ValueError("target/resource outcome cannot carry shared reasons")
        message = "" if self.message is None else str(self.message)
        if status in {"target_veto", "resource_closeout"} and not message:
            raise ValueError("target/resource outcome requires a message")
        object.__setattr__(self, "message", message or None)


def fixed_metric_search_seed(
    root_seed: tuple[int, int],
    *,
    domain: str,
    num_leapfrog_steps: int,
    replication_index: int = 0,
) -> tuple[int, int]:
    """Derive an order-independent seed from the complete candidate identity."""

    root = _strict_seed(root_seed, name="root_seed")
    lane = _nonempty(domain, name="domain")
    leapfrog = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=MIN_LEAPFROG_STEPS,
    )
    replication = _strict_integer(
        replication_index,
        name="replication_index",
        minimum=0,
    )
    digest = hashlib.sha256(
        f"{root[0]}:{root[1]}:{lane}:{leapfrog}:{replication}".encode("ascii")
    ).digest()
    modulus = 2**31 - 1
    seed = (
        (root[0] + int.from_bytes(digest[:8], "big")) % modulus,
        (root[1] + int.from_bytes(digest[8:16], "big")) % modulus,
    )
    return (0, 1) if seed == (0, 0) else seed


@dataclass(frozen=True)
class FixedMetricTuneRequest:
    round_index: int
    num_leapfrog_steps: int
    seed: tuple[int, int]
    initial_step_size: float
    lineage: FixedMetricSearchLineage


@dataclass(frozen=True)
class FixedMetricTuneOutcome:
    num_leapfrog_steps: int
    seed: tuple[int, int]
    tuned_step_size: float
    lineage: FixedMetricSearchLineage


@dataclass(frozen=True)
class FixedMetricScreenRequest:
    round_index: int
    stage: str
    num_leapfrog_steps: int
    replication_index: int
    seed: tuple[int, int]
    tuned_step_size: float
    num_results: int
    lineage: FixedMetricSearchLineage


@dataclass(frozen=True)
class FixedMetricScreenOutcome:
    num_leapfrog_steps: int
    replication_index: int
    seed: tuple[int, int]
    tuned_step_size: float
    lineage: FixedMetricSearchLineage
    acceptance_evidence_payload: Mapping[str, Any]


class CandidateTuneRejected(RuntimeError):
    """Typed candidate-local tune failure; other grid candidates remain valid."""

    def __init__(self, *reasons: str) -> None:
        self.reasons = _reason_tuple(
            reasons,
            allowed=_CANDIDATE_TUNE_REASONS,
            name="candidate tune rejection",
        )
        super().__init__("candidate tune rejected")


class CandidateScreenRejected(RuntimeError):
    """Typed candidate-local screen failure; the shared harness remains valid."""

    def __init__(self, *reasons: str) -> None:
        self.reasons = _reason_tuple(
            reasons,
            allowed=_CANDIDATE_SCREEN_REASONS,
            name="candidate screen rejection",
        )
        super().__init__("candidate screen rejected")


class SharedGridSearchInvalidity(RuntimeError):
    """Typed shared contract failure that stops the complete search barrier."""

    def __init__(self, *reasons: str) -> None:
        self.reasons = _reason_tuple(
            reasons,
            allowed=_SHARED_REASONS,
            name="shared grid-search invalidity",
        )
        super().__init__("shared grid-search execution is invalid")


class GridSearchResourceCloseout(TimeoutError):
    """Typed resource stop that preserves completed candidate evidence."""

    def __init__(self, reason: str) -> None:
        self.reason = _nonempty(reason, name="resource closeout reason")
        super().__init__(self.reason)


class GridSearchTargetVeto(RuntimeError):
    """Typed shared target-health veto preserved for caller-owned closeout."""

    def __init__(self, reason: str) -> None:
        self.reason = _nonempty(reason, name="target veto reason")
        super().__init__(self.reason)


def _validate_lineage(
    actual: FixedMetricSearchLineage,
    expected: FixedMetricSearchLineage,
) -> None:
    if not isinstance(actual, FixedMetricSearchLineage):
        raise SharedGridSearchInvalidity("shared_schema_invalid")
    fields = (
        ("coordinate_signature", "coordinate_signature_mismatch"),
        ("metric_signature", "metric_signature_mismatch"),
        (
            "private_start_bank_content_signature",
            "start_bank_signature_mismatch",
        ),
        ("common_state_signature", "common_state_signature_mismatch"),
    )
    mismatches = tuple(
        reason
        for field, reason in fields
        if getattr(actual, field) != getattr(expected, field)
    )
    if mismatches:
        raise SharedGridSearchInvalidity(*mismatches)


@dataclass(frozen=True)
class FixedMetricScreenRecord:
    request: FixedMetricScreenRequest
    evidence_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        evidence = hmc_acceptance_evidence_from_payload(self.evidence_payload)
        object.__setattr__(self, "evidence_payload", evidence.payload())

    @property
    def evidence(self) -> HMCAcceptanceEvidence:
        return hmc_acceptance_evidence_from_payload(self.evidence_payload)

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.hmc_fixed_metric_screen_record.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "stage": self.request.stage,
            "round_index": self.request.round_index,
            "num_leapfrog_steps": self.request.num_leapfrog_steps,
            "replication_index": self.request.replication_index,
            "seed": self.request.seed,
            "tuned_step_size": self.request.tuned_step_size,
            "num_results": self.request.num_results,
            "lineage": self.request.lineage.payload(),
            "acceptance_evidence": self.evidence_payload,
        }


@dataclass(frozen=True)
class FixedMetricEvidenceExtensionRecord:
    replication_index: int
    prior_screen_signature: str
    replacement: FixedMetricScreenRecord

    def payload(self) -> Mapping[str, Any]:
        return {
            "replication_index": self.replication_index,
            "prior_screen_signature": self.prior_screen_signature,
            "replacement": self.replacement.payload(),
            "fresh_trace_replaces_prior_evidence": True,
            "traces_concatenated": False,
        }


@dataclass(frozen=True)
class FixedMetricCandidateRecord:
    round_index: int
    num_leapfrog_steps: int
    tune_seed: tuple[int, int]
    tuned_step_size: float | None
    screens: tuple[FixedMetricScreenRecord, ...] = ()
    evidence_extensions: tuple[FixedMetricEvidenceExtensionRecord, ...] = ()
    rejection_stage: str | None = None
    rejection_reasons: tuple[str, ...] = ()
    rejection_replication_index: int | None = None

    @property
    def survivor(self) -> bool:
        return (
            self.tuned_step_size is not None
            and self.rejection_stage is None
            and len(self.screens) == REPLICATION_COUNT
            and all(item.evidence.promotion_eligible for item in self.screens)
        )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.hmc_fixed_metric_candidate_record.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "round_index": self.round_index,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "tune_seed": self.tune_seed,
            "tuned_step_size": self.tuned_step_size,
            "screens": tuple(item.payload() for item in self.screens),
            "evidence_extensions": tuple(
                item.payload() for item in self.evidence_extensions
            ),
            "rejection_stage": self.rejection_stage,
            "rejection_reasons": self.rejection_reasons,
            "rejection_replication_index": self.rejection_replication_index,
            "survivor": self.survivor,
        }


@dataclass(frozen=True)
class FixedMetricCandidateEvidencePolicy:
    """Opt-in uncertainty-aware candidate confirmation policy.

    The Student-t interval is a bounded tuning heuristic over independently
    randomized chain-run means. It is not a posterior confidence interval and
    does not establish convergence or retained-sampling readiness.
    """

    confirmation_num_results: int = DEFAULT_CONFIRMATION_RESULTS
    working_interval_level: float = 0.90
    working_t_critical: float = DEFAULT_WORKING_T_CRITICAL_90_DF11

    def __post_init__(self) -> None:
        results = _strict_integer(
            self.confirmation_num_results,
            name="confirmation_num_results",
            minimum=1,
        )
        if results <= 64:
            raise ValueError("confirmation_num_results must exceed the 64-draw nomination")
        level = float(self.working_interval_level)
        critical = float(self.working_t_critical)
        if not 0.0 < level < 1.0:
            raise ValueError("working_interval_level must lie inside (0, 1)")
        if not math.isfinite(critical) or critical <= 0.0:
            raise ValueError("working_t_critical must be positive and finite")
        object.__setattr__(self, "confirmation_num_results", results)
        object.__setattr__(self, "working_interval_level", level)
        object.__setattr__(self, "working_t_critical", critical)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_fixed_metric_candidate_evidence_policy.v1",
            "confirmation_num_results": self.confirmation_num_results,
            "working_interval_level": self.working_interval_level,
            "working_t_critical": self.working_t_critical,
            "working_interval_unit": "freshly_seeded_chain_run_mean_with_shared_start",
            "working_interval_role": "bounded_tuning_heuristic_not_confidence_guarantee",
            "working_interval_dependence_limitations": (
                "shared_initial_position",
                "within_chain_mcmc_autocorrelation",
                "chain_run_means_not_proven_independent_or_gaussian",
            ),
            "fresh_confirmation_required_for_provisional_viability": True,
        }


@dataclass(frozen=True)
class FixedMetricAggregateEvidence:
    """Candidate-level working evidence across replications and chains."""

    phase: str
    num_leapfrog_steps: int
    tuned_step_size: float
    replication_count: int
    chain_count: int
    chain_run_means: tuple[float, ...]
    grand_mean: float
    sample_standard_deviation: float
    standard_error: float
    working_interval: tuple[float, float]
    disposition: str
    hard_rejection_reasons: tuple[str, ...]
    policy: FixedMetricCandidateEvidencePolicy

    @property
    def provisional_viable(self) -> bool:
        return self.phase == "confirmation" and self.disposition == "provisional_viable"

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_fixed_metric_aggregate_evidence.v1",
            "phase": self.phase,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "tuned_step_size": self.tuned_step_size,
            "replication_count": self.replication_count,
            "chain_count": self.chain_count,
            "chain_run_means": self.chain_run_means,
            "grand_mean": self.grand_mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "standard_error": self.standard_error,
            "working_interval": self.working_interval,
            "disposition": self.disposition,
            "hard_rejection_reasons": self.hard_rejection_reasons,
            "policy": self.policy.payload(),
            "uses_fresh_evidence": self.phase == "confirmation",
            "reports_posterior_convergence": False,
            "retained_sampling_authorized": False,
        }


@dataclass(frozen=True)
class FixedMetricCandidateConfirmationRecord:
    """Fresh fixed-epsilon confirmation linked to one immutable nomination."""

    source_candidate_signature: str
    num_leapfrog_steps: int
    tuned_step_size: float
    nomination: FixedMetricAggregateEvidence
    confirmation_screens: tuple[FixedMetricScreenRecord, ...]
    confirmation: FixedMetricAggregateEvidence | None
    rejection_reasons: tuple[str, ...] = ()
    rejection_replication_index: int | None = None

    @property
    def disposition(self) -> str:
        if self.rejection_reasons:
            return "hard_rejected"
        if self.confirmation is None:
            return self.nomination.disposition
        return self.confirmation.disposition

    @property
    def provisional_viable(self) -> bool:
        return self.confirmation is not None and self.confirmation.provisional_viable

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_fixed_metric_candidate_confirmation.v1",
            "source_candidate_signature": self.source_candidate_signature,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "tuned_step_size": self.tuned_step_size,
            "nomination": self.nomination.payload(),
            "confirmation_screens": tuple(
                item.payload() for item in self.confirmation_screens
            ),
            "confirmation": (
                None if self.confirmation is None else self.confirmation.payload()
            ),
            "rejection_reasons": self.rejection_reasons,
            "rejection_replication_index": self.rejection_replication_index,
            "disposition": self.disposition,
            "provisional_viable": self.provisional_viable,
            "original_nomination_can_promote": False,
            "retained_sampling_authorized": False,
        }


def refinement_l_values(
    initial_grid: Sequence[int], survivor_l_values: Sequence[int]
) -> tuple[int, ...]:
    """Return untested integer midpoints adjacent to every initial survivor."""

    grid = tuple(sorted(int(item) for item in initial_grid))
    survivors = set(int(item) for item in survivor_l_values)
    if len(grid) < 3 or len(set(grid)) != len(grid):
        raise ValueError("refinement requires at least three distinct grid values")
    if any(
        item < MIN_LEAPFROG_STEPS or item > MAX_LEAPFROG_STEPS for item in grid
    ):
        raise ValueError("refinement grid values exceed the reviewed bounds")
    if not survivors.issubset(grid):
        raise ValueError("refinement survivors must come from the initial grid")
    additions: set[int] = set()
    for survivor in survivors:
        index = grid.index(survivor)
        intervals = []
        if index > 0:
            intervals.append((grid[index - 1], survivor))
        if index + 1 < len(grid):
            intervals.append((survivor, grid[index + 1]))
        for lower, upper in intervals:
            floor_midpoint = (lower + upper) // 2
            ceil_midpoint = (lower + upper + 1) // 2
            additions.update(
                item
                for item in (floor_midpoint, ceil_midpoint)
                if MIN_LEAPFROG_STEPS <= item <= MAX_LEAPFROG_STEPS
                and item not in grid
            )
    return tuple(sorted(additions))


@dataclass(frozen=True)
class FixedMetricGridSearchResult:
    config: FixedMetricGridSearchConfig
    lineage: FixedMetricSearchLineage
    acceptance_policy: HMCAcceptancePolicy
    round0_candidates: tuple[FixedMetricCandidateRecord, ...]
    refinement_candidates: tuple[FixedMetricCandidateRecord, ...]
    disposition: str
    shared_invalidity_reasons: tuple[str, ...] = ()
    execution: FixedMetricGridExecutionConfig = field(
        default_factory=FixedMetricGridExecutionConfig
    )

    @property
    def candidates(self) -> tuple[FixedMetricCandidateRecord, ...]:
        return self.round0_candidates + self.refinement_candidates

    @property
    def survivors(self) -> tuple[FixedMetricCandidateRecord, ...]:
        if self.shared_invalidity_reasons:
            return ()
        return tuple(
            sorted(
                (item for item in self.candidates if item.survivor),
                key=lambda item: item.num_leapfrog_steps,
            )
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_fixed_metric_grid_search.private.v1",
            "config": self.config.payload(),
            "lineage": self.lineage.payload(),
            "acceptance_policy": self.acceptance_policy.payload(),
            "round0_candidates": tuple(
                item.payload() for item in self.round0_candidates
            ),
            "refinement_candidates": tuple(
                item.payload() for item in self.refinement_candidates
            ),
            "survivor_pairs": tuple(
                {
                    "num_leapfrog_steps": item.num_leapfrog_steps,
                    "tuned_step_size": item.tuned_step_size,
                }
                for item in self.survivors
            ),
            "disposition": self.disposition,
            "shared_invalidity_reasons": self.shared_invalidity_reasons,
            "execution": self.execution.payload(),
            "representative": None,
            "stochastic_ranking_performed": False,
            "all_tuning_draws_discarded": True,
            "private_handoff_only": True,
            "raw_samples_exposed": False,
            "raw_states_exposed": False,
            "metric_matrix_exposed": False,
            "nonclaims": GRID_SEARCH_NONCLAIMS,
        }

    def public_summary(self) -> Mapping[str, Any]:
        """Return aggregate, non-replayable status without HMC mechanics."""

        planned_refinement_count = (
            len(
                refinement_l_values(
                    self.config.l_grid,
                    tuple(
                        item.num_leapfrog_steps
                        for item in self.round0_candidates
                        if item.survivor
                    ),
                )
            )
            if self.config.refinement_rounds == 1
            and len(self.round0_candidates) == len(self.config.l_grid)
            else 0
        )
        return {
            "schema": "bayesfilter.hmc_fixed_metric_grid_search.public.v1",
            "disposition": self.disposition,
            "planned_initial_count": len(self.config.l_grid),
            "completed_initial_count": len(self.round0_candidates),
            "planned_refinement_count": planned_refinement_count,
            "completed_refinement_count": len(self.refinement_candidates),
            "completed_total_count": len(self.candidates),
            "tune_rejected_count": sum(
                item.rejection_stage == "tune" for item in self.candidates
            ),
            "screen_rejected_count": sum(
                item.rejection_stage in {"screen", "evidence_extension"}
                for item in self.candidates
            ),
            "screened_count": sum(bool(item.screens) for item in self.candidates),
            "surviving_count": len(self.survivors),
            "shared_invalidity_fired": bool(self.shared_invalidity_reasons),
            "execution_mode": self.execution.mode,
            "execution_worker_count": self.execution.max_workers,
            "stochastic_ranking_performed": False,
            "retained_sampling_authorized": False,
            "raw_samples_exposed": False,
            "raw_states_exposed": False,
            "replayable_mechanics_exposed": False,
            "nonclaims": GRID_SEARCH_NONCLAIMS,
        }


TuneRunner = Callable[[FixedMetricTuneRequest], FixedMetricTuneOutcome]
ScreenRunner = Callable[[FixedMetricScreenRequest], FixedMetricScreenOutcome]
CandidateStartCallback = Callable[[int, int, int], None]
CandidateCompleteCallback = Callable[[FixedMetricCandidateRecord, int, int], None]


def _callback_shared_failure(error: BaseException) -> SharedGridSearchInvalidity:
    if isinstance(error, SharedGridSearchInvalidity):
        return error
    return SharedGridSearchInvalidity("untyped_callback_failure")


def _run_tune(
    *,
    round_index: int,
    leapfrog: int,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    tune_runner: TuneRunner,
) -> tuple[FixedMetricTuneRequest, FixedMetricTuneOutcome]:
    seed = fixed_metric_search_seed(
        config.root_seed,
        domain=f"round_{round_index}_tune",
        num_leapfrog_steps=leapfrog,
    )
    request = FixedMetricTuneRequest(
        round_index=round_index,
        num_leapfrog_steps=leapfrog,
        seed=seed,
        initial_step_size=config.initial_step_size,
        lineage=lineage,
    )
    try:
        outcome = tune_runner(request)
    except CandidateTuneRejected:
        raise
    except (GridSearchResourceCloseout, GridSearchTargetVeto):
        raise
    except Exception as error:  # noqa: BLE001 - unknown callback failures are shared.
        raise _callback_shared_failure(error) from error
    if not isinstance(outcome, FixedMetricTuneOutcome):
        raise SharedGridSearchInvalidity("shared_schema_invalid")
    _validate_lineage(outcome.lineage, lineage)
    if outcome.num_leapfrog_steps != leapfrog:
        raise SharedGridSearchInvalidity("candidate_identity_mismatch")
    if tuple(outcome.seed) != seed:
        raise SharedGridSearchInvalidity("seed_lineage_mismatch")
    step = float(outcome.tuned_step_size)
    if not math.isfinite(step) or step <= 0.0:
        raise CandidateTuneRejected("nonfinite_adapted_step_size")
    return request, replace(outcome, tuned_step_size=step)


def _run_screen(
    *,
    round_index: int,
    stage: str,
    leapfrog: int,
    replication_index: int,
    tuned_step_size: float,
    num_results: int,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    screen_runner: ScreenRunner,
) -> FixedMetricScreenRecord:
    domain = f"round_{round_index}_{stage}_{num_results}"
    seed = fixed_metric_search_seed(
        config.root_seed,
        domain=domain,
        num_leapfrog_steps=leapfrog,
        replication_index=replication_index,
    )
    request = FixedMetricScreenRequest(
        round_index=round_index,
        stage=stage,
        num_leapfrog_steps=leapfrog,
        replication_index=replication_index,
        seed=seed,
        tuned_step_size=tuned_step_size,
        num_results=num_results,
        lineage=lineage,
    )
    try:
        outcome = screen_runner(request)
    except CandidateScreenRejected:
        raise
    except (GridSearchResourceCloseout, GridSearchTargetVeto):
        raise
    except Exception as error:  # noqa: BLE001 - unknown callback failures are shared.
        raise _callback_shared_failure(error) from error
    if not isinstance(outcome, FixedMetricScreenOutcome):
        raise SharedGridSearchInvalidity("shared_schema_invalid")
    _validate_lineage(outcome.lineage, lineage)
    if (
        outcome.num_leapfrog_steps != leapfrog
        or outcome.replication_index != replication_index
    ):
        raise SharedGridSearchInvalidity("candidate_identity_mismatch")
    if tuple(outcome.seed) != seed:
        raise SharedGridSearchInvalidity("seed_lineage_mismatch")
    if not math.isclose(
        float(outcome.tuned_step_size),
        tuned_step_size,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise SharedGridSearchInvalidity("step_size_lineage_mismatch")
    try:
        evidence = hmc_acceptance_evidence_from_payload(
            outcome.acceptance_evidence_payload
        )
    except (TypeError, ValueError) as error:
        raise SharedGridSearchInvalidity("shared_schema_invalid") from error
    if evidence.policy.payload() != acceptance_policy.payload():
        raise SharedGridSearchInvalidity("acceptance_policy_mismatch")
    if evidence.evidence_validity == "shared_execution_invalid":
        raise SharedGridSearchInvalidity("shared_schema_invalid")
    return FixedMetricScreenRecord(request=request, evidence_payload=evidence.payload())


def _extension_eligible(screens: Sequence[FixedMetricScreenRecord]) -> bool:
    evidence = tuple(item.evidence for item in screens)
    decisions = tuple(item.acceptance_decision for item in evidence)
    return (
        len(evidence) == REPLICATION_COUNT
        and all(item.evidence_validity == "valid" for item in evidence)
        and all(not item.candidate_promotion_vetoes for item in evidence)
        and all(not item.cost_stop_reasons for item in evidence)
        and all(
            decision in {"passed", "inconclusive_evidence"}
            for decision in decisions
        )
        and "inconclusive_evidence" in decisions
    )


def aggregate_fixed_metric_candidate_evidence(
    *,
    phase: str,
    num_leapfrog_steps: int,
    tuned_step_size: float,
    screens: Sequence[FixedMetricScreenRecord],
    acceptance_policy: HMCAcceptancePolicy,
    evidence_policy: FixedMetricCandidateEvidencePolicy,
) -> FixedMetricAggregateEvidence:
    """Summarize one complete evidence phase without changing screen decisions."""

    phase_name = str(phase)
    if phase_name not in AGGREGATE_EVIDENCE_PHASES:
        raise ValueError(f"phase must be one of {AGGREGATE_EVIDENCE_PHASES}")
    leapfrog = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=MIN_LEAPFROG_STEPS,
    )
    step = float(tuned_step_size)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("tuned_step_size must be positive and finite")
    if not isinstance(acceptance_policy, HMCAcceptancePolicy):
        raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
    if not isinstance(evidence_policy, FixedMetricCandidateEvidencePolicy):
        raise TypeError("evidence_policy must be FixedMetricCandidateEvidencePolicy")
    records = tuple(screens)
    if len(records) != REPLICATION_COUNT:
        raise ValueError("aggregate evidence requires exactly three replications")

    hard_reasons: list[str] = []
    chain_means: list[float] = []
    expected_results = (
        evidence_policy.confirmation_num_results if phase_name == "confirmation" else None
    )
    for replication_index, record in enumerate(records):
        if not isinstance(record, FixedMetricScreenRecord):
            raise TypeError("screens must contain FixedMetricScreenRecord values")
        request = record.request
        evidence = record.evidence
        if (
            request.num_leapfrog_steps != leapfrog
            or request.replication_index != replication_index
            or not math.isclose(
                request.tuned_step_size,
                step,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ):
            raise ValueError("aggregate evidence screen identity mismatch")
        if phase_name == "confirmation":
            if request.stage != "confirmation" or request.num_results != expected_results:
                raise ValueError("confirmation screen budget or stage mismatch")
        if evidence.policy.payload() != acceptance_policy.payload():
            raise ValueError("aggregate evidence acceptance policy mismatch")
        if evidence.evidence_validity != "valid":
            hard_reasons.extend(evidence.engineering_invalidity_reasons)
        hard_reasons.extend(evidence.candidate_promotion_vetoes)
        hard_reasons.extend(evidence.cost_stop_reasons)
        if len(evidence.chain_means) != acceptance_policy.chain_count:
            raise ValueError("aggregate evidence requires complete chain means")
        chain_means.extend(float(item) for item in evidence.chain_means)

    values = tuple(chain_means)
    expected_units = REPLICATION_COUNT * acceptance_policy.chain_count
    if len(values) != expected_units or any(
        not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values
    ):
        raise ValueError("aggregate chain-run means are incomplete or invalid")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((item - mean) ** 2 for item in values) / (len(values) - 1)
    standard_deviation = math.sqrt(max(0.0, variance))
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = evidence_policy.working_t_critical * standard_error
    interval = (max(0.0, mean - half_width), min(1.0, mean + half_width))
    practical_low, practical_high = acceptance_policy.practical_region
    repair_low, repair_high = acceptance_policy.repair_region
    reasons = tuple(dict.fromkeys(hard_reasons))
    if reasons:
        disposition = "hard_rejected"
    elif interval[1] < practical_low:
        disposition = "needs_lower_epsilon"
    elif interval[0] > practical_high:
        disposition = "needs_higher_epsilon"
    elif phase_name == "nomination":
        disposition = "confirmation_required"
    elif interval[0] >= repair_low and interval[1] <= repair_high:
        disposition = "provisional_viable"
    else:
        disposition = "unresolved_budget"
    return FixedMetricAggregateEvidence(
        phase=phase_name,
        num_leapfrog_steps=leapfrog,
        tuned_step_size=step,
        replication_count=REPLICATION_COUNT,
        chain_count=acceptance_policy.chain_count,
        chain_run_means=values,
        grand_mean=mean,
        sample_standard_deviation=standard_deviation,
        standard_error=standard_error,
        working_interval=interval,
        disposition=disposition,
        hard_rejection_reasons=reasons,
        policy=evidence_policy,
    )


def confirm_fixed_metric_candidate(
    *,
    candidate: FixedMetricCandidateRecord,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    evidence_policy: FixedMetricCandidateEvidencePolicy,
    screen_runner: ScreenRunner,
) -> FixedMetricCandidateConfirmationRecord:
    """Run fresh confirmation at one immutable candidate's frozen epsilon."""

    if not isinstance(candidate, FixedMetricCandidateRecord):
        raise TypeError("candidate must be FixedMetricCandidateRecord")
    if candidate.tuned_step_size is None or candidate.rejection_stage is not None:
        raise ValueError("confirmation requires a tuned candidate without hard rejection")
    nomination = aggregate_fixed_metric_candidate_evidence(
        phase="nomination",
        num_leapfrog_steps=candidate.num_leapfrog_steps,
        tuned_step_size=candidate.tuned_step_size,
        screens=candidate.screens,
        acceptance_policy=acceptance_policy,
        evidence_policy=evidence_policy,
    )
    if nomination.disposition != "confirmation_required":
        return FixedMetricCandidateConfirmationRecord(
            source_candidate_signature=candidate.signature,
            num_leapfrog_steps=candidate.num_leapfrog_steps,
            tuned_step_size=candidate.tuned_step_size,
            nomination=nomination,
            confirmation_screens=(),
            confirmation=None,
            rejection_reasons=nomination.hard_rejection_reasons,
        )

    screens: list[FixedMetricScreenRecord] = []
    for replication_index in range(REPLICATION_COUNT):
        try:
            screen = _run_screen(
                round_index=candidate.round_index,
                stage="confirmation",
                leapfrog=candidate.num_leapfrog_steps,
                replication_index=replication_index,
                tuned_step_size=candidate.tuned_step_size,
                num_results=evidence_policy.confirmation_num_results,
                config=config,
                lineage=lineage,
                acceptance_policy=acceptance_policy,
                screen_runner=screen_runner,
            )
        except CandidateScreenRejected as failure:
            return FixedMetricCandidateConfirmationRecord(
                source_candidate_signature=candidate.signature,
                num_leapfrog_steps=candidate.num_leapfrog_steps,
                tuned_step_size=candidate.tuned_step_size,
                nomination=nomination,
                confirmation_screens=tuple(screens),
                confirmation=None,
                rejection_reasons=failure.reasons,
                rejection_replication_index=replication_index,
            )
        screens.append(screen)
        if screen.evidence.evidence_validity == "candidate_data_invalid":
            return FixedMetricCandidateConfirmationRecord(
                source_candidate_signature=candidate.signature,
                num_leapfrog_steps=candidate.num_leapfrog_steps,
                tuned_step_size=candidate.tuned_step_size,
                nomination=nomination,
                confirmation_screens=tuple(screens),
                confirmation=None,
                rejection_reasons=screen.evidence.engineering_invalidity_reasons,
                rejection_replication_index=replication_index,
            )
    confirmation = aggregate_fixed_metric_candidate_evidence(
        phase="confirmation",
        num_leapfrog_steps=candidate.num_leapfrog_steps,
        tuned_step_size=candidate.tuned_step_size,
        screens=screens,
        acceptance_policy=acceptance_policy,
        evidence_policy=evidence_policy,
    )
    return FixedMetricCandidateConfirmationRecord(
        source_candidate_signature=candidate.signature,
        num_leapfrog_steps=candidate.num_leapfrog_steps,
        tuned_step_size=candidate.tuned_step_size,
        nomination=nomination,
        confirmation_screens=tuple(screens),
        confirmation=confirmation,
        rejection_reasons=confirmation.hard_rejection_reasons,
    )


def run_fixed_metric_confirmation_screen(
    *,
    round_index: int,
    num_leapfrog_steps: int,
    replication_index: int,
    tuned_step_size: float,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    evidence_policy: FixedMetricCandidateEvidencePolicy,
    screen_runner: ScreenRunner,
) -> FixedMetricScreenRecord:
    """Run one fresh confirmation replication through the public API."""

    if not isinstance(config, FixedMetricGridSearchConfig):
        raise TypeError("config must be FixedMetricGridSearchConfig")
    if not isinstance(lineage, FixedMetricSearchLineage):
        raise TypeError("lineage must be FixedMetricSearchLineage")
    if not isinstance(acceptance_policy, HMCAcceptancePolicy):
        raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
    if not isinstance(evidence_policy, FixedMetricCandidateEvidencePolicy):
        raise TypeError("evidence_policy must be FixedMetricCandidateEvidencePolicy")
    if not callable(screen_runner):
        raise TypeError("screen_runner must be callable")
    round_value = _strict_integer(round_index, name="round_index", minimum=0)
    if round_value not in {0, 1}:
        raise ValueError("fixed-metric confirmation supports only rounds zero and one")
    leapfrog = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=MIN_LEAPFROG_STEPS,
    )
    if leapfrog > MAX_LEAPFROG_STEPS:
        raise ValueError("confirmation L exceeds the reviewed bound")
    replication = _strict_integer(
        replication_index,
        name="replication_index",
        minimum=0,
    )
    if replication >= REPLICATION_COUNT:
        raise ValueError("replication_index exceeds the reviewed replication count")
    step = float(tuned_step_size)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("tuned_step_size must be positive and finite")
    return _run_screen(
        round_index=round_value,
        stage="confirmation",
        leapfrog=leapfrog,
        replication_index=replication,
        tuned_step_size=step,
        num_results=evidence_policy.confirmation_num_results,
        config=config,
        lineage=lineage,
        acceptance_policy=acceptance_policy,
        screen_runner=screen_runner,
    )


def _run_candidate(
    *,
    round_index: int,
    leapfrog: int,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    tune_runner: TuneRunner,
    screen_runner: ScreenRunner,
) -> FixedMetricCandidateRecord:
    tune_seed = fixed_metric_search_seed(
        config.root_seed,
        domain=f"round_{round_index}_tune",
        num_leapfrog_steps=leapfrog,
    )
    try:
        _, tune = _run_tune(
            round_index=round_index,
            leapfrog=leapfrog,
            config=config,
            lineage=lineage,
            tune_runner=tune_runner,
        )
    except CandidateTuneRejected as failure:
        return FixedMetricCandidateRecord(
            round_index=round_index,
            num_leapfrog_steps=leapfrog,
            tune_seed=tune_seed,
            tuned_step_size=None,
            rejection_stage="tune",
            rejection_reasons=failure.reasons,
        )

    screens: list[FixedMetricScreenRecord] = []
    try:
        for replication_index in range(REPLICATION_COUNT):
            screen = _run_screen(
                round_index=round_index,
                stage="screen",
                leapfrog=leapfrog,
                replication_index=replication_index,
                tuned_step_size=tune.tuned_step_size,
                num_results=config.screen_num_results,
                config=config,
                lineage=lineage,
                acceptance_policy=acceptance_policy,
                screen_runner=screen_runner,
            )
            screens.append(screen)
            if screen.evidence.evidence_validity == "candidate_data_invalid":
                return FixedMetricCandidateRecord(
                    round_index=round_index,
                    num_leapfrog_steps=leapfrog,
                    tune_seed=tune_seed,
                    tuned_step_size=tune.tuned_step_size,
                    screens=tuple(screens),
                    rejection_stage="screen",
                    rejection_reasons=screen.evidence.engineering_invalidity_reasons,
                    rejection_replication_index=replication_index,
                )
    except CandidateScreenRejected as failure:
        return FixedMetricCandidateRecord(
            round_index=round_index,
            num_leapfrog_steps=leapfrog,
            tune_seed=tune_seed,
            tuned_step_size=tune.tuned_step_size,
            screens=tuple(screens),
            rejection_stage="screen",
            rejection_reasons=failure.reasons,
            rejection_replication_index=len(screens),
        )

    extensions: list[FixedMetricEvidenceExtensionRecord] = []
    if _extension_eligible(screens):
        updated = list(screens)
        for index, prior in enumerate(tuple(screens)):
            if prior.evidence.acceptance_decision != "inconclusive_evidence":
                continue
            try:
                replacement = _run_screen(
                    round_index=round_index,
                    stage="evidence_extension",
                    leapfrog=leapfrog,
                    replication_index=index,
                    tuned_step_size=tune.tuned_step_size,
                    num_results=config.extension_num_results,
                    config=config,
                    lineage=lineage,
                    acceptance_policy=acceptance_policy,
                    screen_runner=screen_runner,
                )
            except CandidateScreenRejected as failure:
                return FixedMetricCandidateRecord(
                    round_index=round_index,
                    num_leapfrog_steps=leapfrog,
                    tune_seed=tune_seed,
                    tuned_step_size=tune.tuned_step_size,
                    screens=tuple(updated),
                    evidence_extensions=tuple(extensions),
                    rejection_stage="evidence_extension",
                    rejection_reasons=failure.reasons,
                )
            extension = FixedMetricEvidenceExtensionRecord(
                replication_index=index,
                prior_screen_signature=prior.signature,
                replacement=replacement,
            )
            extensions.append(extension)
            updated[index] = replacement
            if replacement.evidence.evidence_validity == "candidate_data_invalid":
                return FixedMetricCandidateRecord(
                    round_index=round_index,
                    num_leapfrog_steps=leapfrog,
                    tune_seed=tune_seed,
                    tuned_step_size=tune.tuned_step_size,
                    screens=tuple(updated),
                    evidence_extensions=tuple(extensions),
                    rejection_stage="evidence_extension",
                    rejection_reasons=(
                        replacement.evidence.engineering_invalidity_reasons
                    ),
                    rejection_replication_index=index,
                )
        screens = updated

    return FixedMetricCandidateRecord(
        round_index=round_index,
        num_leapfrog_steps=leapfrog,
        tune_seed=tune_seed,
        tuned_step_size=tune.tuned_step_size,
        screens=tuple(screens),
        evidence_extensions=tuple(extensions),
    )


def run_fixed_metric_candidate(
    *,
    round_index: int,
    num_leapfrog_steps: int,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    tune_runner: TuneRunner,
    screen_runner: ScreenRunner,
) -> FixedMetricCandidateRecord:
    """Run one complete candidate using the same semantics as grid execution."""

    if not isinstance(config, FixedMetricGridSearchConfig):
        raise TypeError("config must be FixedMetricGridSearchConfig")
    if not isinstance(lineage, FixedMetricSearchLineage):
        raise TypeError("lineage must be FixedMetricSearchLineage")
    if not isinstance(acceptance_policy, HMCAcceptancePolicy):
        raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
    if not callable(tune_runner) or not callable(screen_runner):
        raise TypeError("tune_runner and screen_runner must be callable")
    round_value = _strict_integer(round_index, name="round_index", minimum=0)
    if round_value not in {0, 1}:
        raise ValueError("fixed-metric candidates support only rounds zero and one")
    leapfrog = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=MIN_LEAPFROG_STEPS,
    )
    allowed = (
        set(config.l_grid)
        if round_value == 0
        else set(range(MIN_LEAPFROG_STEPS, MAX_LEAPFROG_STEPS + 1))
    )
    if leapfrog not in allowed or leapfrog > MAX_LEAPFROG_STEPS:
        raise ValueError("candidate L is outside its reviewed round")
    return _run_candidate(
        round_index=round_value,
        leapfrog=leapfrog,
        config=config,
        lineage=lineage,
        acceptance_policy=acceptance_policy,
        tune_runner=tune_runner,
        screen_runner=screen_runner,
    )


def _resolve_worker_factory(locator: str) -> Callable[..., Any]:
    module_name, attribute_path = locator.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError("worker factory locator does not resolve to a callable")
    return value


def _run_candidate_process_worker(
    request: FixedMetricCandidateWorkerRequest,
    factory_locator: str,
) -> FixedMetricCandidateWorkerOutcome:
    """Spawn-worker entry point; environment is inherited before module import."""

    try:
        factory = _resolve_worker_factory(factory_locator)
        runners = factory(request)
        if not isinstance(runners, FixedMetricCandidateRunners):
            raise TypeError("worker factory must return FixedMetricCandidateRunners")
        candidate = run_fixed_metric_candidate(
            round_index=request.round_index,
            num_leapfrog_steps=request.num_leapfrog_steps,
            config=request.config,
            lineage=request.lineage,
            acceptance_policy=request.acceptance_policy,
            tune_runner=runners.tune_runner,
            screen_runner=runners.screen_runner,
        )
        return FixedMetricCandidateWorkerOutcome(
            status="candidate_complete",
            candidate=candidate,
        )
    except GridSearchTargetVeto as error:
        return FixedMetricCandidateWorkerOutcome(
            status="target_veto",
            message=str(error),
        )
    except GridSearchResourceCloseout as error:
        return FixedMetricCandidateWorkerOutcome(
            status="resource_closeout",
            message=str(error),
        )
    except SharedGridSearchInvalidity as error:
        return FixedMetricCandidateWorkerOutcome(
            status="shared_execution_invalid",
            shared_invalidity_reasons=error.reasons,
        )
    except Exception:
        return FixedMetricCandidateWorkerOutcome(
            status="shared_execution_invalid",
            shared_invalidity_reasons=("untyped_callback_failure",),
        )


@contextmanager
def _temporary_worker_environment(
    environment_items: Sequence[tuple[str, str]],
):
    previous = {key: os.environ.get(key) for key, _ in environment_items}
    os.environ.update(dict(environment_items))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_process_candidate_barrier(
    *,
    round_index: int,
    leapfrog_values: Sequence[int],
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    execution: FixedMetricGridExecutionConfig,
    before_candidate: CandidateStartCallback | None,
    after_candidate: CandidateCompleteCallback | None,
) -> tuple[tuple[FixedMetricCandidateRecord, ...], tuple[str, ...]]:
    values = tuple(int(item) for item in leapfrog_values)
    for candidate_index, leapfrog in enumerate(values):
        if before_candidate is not None:
            before_candidate(round_index, candidate_index, leapfrog)
    requests = tuple(
        FixedMetricCandidateWorkerRequest(
            round_index=round_index,
            num_leapfrog_steps=leapfrog,
            config=config,
            lineage=lineage,
            acceptance_policy=acceptance_policy,
        )
        for leapfrog in values
    )
    context = multiprocessing.get_context(execution.start_method)
    candidates: list[FixedMetricCandidateRecord] = []
    shared_reasons: list[str] = []
    target_veto: str | None = None
    resource_closeout: str | None = None
    completed_outcomes: dict[int, FixedMetricCandidateWorkerOutcome | None] = {}
    with _temporary_worker_environment(execution.worker_environment):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(execution.max_workers, len(requests)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    _run_candidate_process_worker,
                    request,
                    str(execution.worker_factory_locator),
                ): (candidate_index, request)
                for candidate_index, request in enumerate(requests)
            }
            for future in concurrent.futures.as_completed(futures):
                candidate_index, _ = futures[future]
                try:
                    completed_outcomes[candidate_index] = future.result()
                except Exception:
                    completed_outcomes[candidate_index] = None

    # Interpret completed work in declared order so callbacks and failure
    # precedence do not depend on operating-system process completion timing.
    for candidate_index, request in enumerate(requests):
        outcome = completed_outcomes.get(candidate_index)
        if outcome is None:
            shared_reasons.append("untyped_callback_failure")
        elif not isinstance(outcome, FixedMetricCandidateWorkerOutcome):
            shared_reasons.append("shared_schema_invalid")
        elif outcome.status == "candidate_complete":
            candidate = outcome.candidate
            if (
                candidate is None
                or candidate.round_index != request.round_index
                or candidate.num_leapfrog_steps != request.num_leapfrog_steps
            ):
                shared_reasons.append("candidate_identity_mismatch")
                continue
            candidates.append(candidate)
            if after_candidate is not None:
                after_candidate(candidate, candidate_index, len(values))
        elif outcome.status == "target_veto":
            if target_veto is None:
                target_veto = outcome.message
        elif outcome.status == "resource_closeout":
            if resource_closeout is None:
                resource_closeout = outcome.message
        else:
            shared_reasons.extend(outcome.shared_invalidity_reasons)
    if target_veto is not None:
        raise GridSearchTargetVeto(target_veto)
    if resource_closeout is not None:
        raise GridSearchResourceCloseout(resource_closeout)
    return (
        tuple(sorted(candidates, key=lambda item: item.num_leapfrog_steps)),
        tuple(dict.fromkeys(shared_reasons)),
    )


def run_fixed_metric_grid_search(
    *,
    config: FixedMetricGridSearchConfig,
    lineage: FixedMetricSearchLineage,
    acceptance_policy: HMCAcceptancePolicy,
    tune_runner: TuneRunner | None = None,
    screen_runner: ScreenRunner | None = None,
    execution: FixedMetricGridExecutionConfig | None = None,
    before_candidate: CandidateStartCallback | None = None,
    after_candidate: CandidateCompleteCallback | None = None,
) -> FixedMetricGridSearchResult:
    """Run the complete broad barrier and the configured refinement phase."""

    if not isinstance(config, FixedMetricGridSearchConfig):
        raise TypeError("config must be FixedMetricGridSearchConfig")
    if not isinstance(lineage, FixedMetricSearchLineage):
        raise TypeError("lineage must be FixedMetricSearchLineage")
    if not isinstance(acceptance_policy, HMCAcceptancePolicy):
        raise TypeError("acceptance_policy must be HMCAcceptancePolicy")
    execution_config = (
        FixedMetricGridExecutionConfig() if execution is None else execution
    )
    if not isinstance(execution_config, FixedMetricGridExecutionConfig):
        raise TypeError("execution must be FixedMetricGridExecutionConfig")
    if execution_config.mode == "serial":
        if not callable(tune_runner) or not callable(screen_runner):
            raise TypeError("serial tune_runner and screen_runner must be callable")
    elif tune_runner is not None or screen_runner is not None:
        raise ValueError(
            "process_parallel constructs callbacks inside workers; do not pass parent callbacks"
        )
    if before_candidate is not None and not callable(before_candidate):
        raise TypeError("before_candidate must be callable when provided")
    if after_candidate is not None and not callable(after_candidate):
        raise TypeError("after_candidate must be callable when provided")

    round0: list[FixedMetricCandidateRecord] = []
    refinement: list[FixedMetricCandidateRecord] = []
    shared_reasons: tuple[str, ...] = ()
    try:
        if execution_config.mode == "process_parallel":
            parallel_round0, round0_shared = _run_process_candidate_barrier(
                round_index=0,
                leapfrog_values=config.l_grid,
                config=config,
                lineage=lineage,
                acceptance_policy=acceptance_policy,
                execution=execution_config,
                before_candidate=before_candidate,
                after_candidate=after_candidate,
            )
            round0.extend(parallel_round0)
            shared_reasons = round0_shared
        else:
            assert tune_runner is not None and screen_runner is not None
            for candidate_index, leapfrog in enumerate(config.l_grid):
                if before_candidate is not None:
                    before_candidate(0, candidate_index, leapfrog)
                candidate = run_fixed_metric_candidate(
                    round_index=0,
                    num_leapfrog_steps=leapfrog,
                    config=config,
                    lineage=lineage,
                    acceptance_policy=acceptance_policy,
                    tune_runner=tune_runner,
                    screen_runner=screen_runner,
                )
                round0.append(candidate)
                if after_candidate is not None:
                    after_candidate(candidate, candidate_index, len(config.l_grid))
        if config.refinement_rounds == 1 and not shared_reasons:
            round0_survivors = tuple(
                item.num_leapfrog_steps for item in round0 if item.survivor
            )
            refinement_values = refinement_l_values(config.l_grid, round0_survivors)
            if execution_config.mode == "process_parallel" and refinement_values:
                parallel_refinement, refinement_shared = (
                    _run_process_candidate_barrier(
                        round_index=1,
                        leapfrog_values=refinement_values,
                        config=config,
                        lineage=lineage,
                        acceptance_policy=acceptance_policy,
                        execution=execution_config,
                        before_candidate=before_candidate,
                        after_candidate=after_candidate,
                    )
                )
                refinement.extend(parallel_refinement)
                shared_reasons = refinement_shared
            elif execution_config.mode == "serial":
                assert tune_runner is not None and screen_runner is not None
                for candidate_index, leapfrog in enumerate(refinement_values):
                    if before_candidate is not None:
                        before_candidate(1, candidate_index, leapfrog)
                    candidate = run_fixed_metric_candidate(
                        round_index=1,
                        num_leapfrog_steps=leapfrog,
                        config=config,
                        lineage=lineage,
                        acceptance_policy=acceptance_policy,
                        tune_runner=tune_runner,
                        screen_runner=screen_runner,
                    )
                    refinement.append(candidate)
                    if after_candidate is not None:
                        after_candidate(
                            candidate,
                            candidate_index,
                            len(refinement_values),
                        )
    except SharedGridSearchInvalidity as failure:
        shared_reasons = failure.reasons

    completed = tuple(round0 + refinement)
    if shared_reasons:
        disposition = "shared_execution_invalid"
    elif any(item.survivor for item in completed):
        disposition = "survivor_set"
    else:
        disposition = "no_survivor"
    return FixedMetricGridSearchResult(
        config=config,
        lineage=lineage,
        acceptance_policy=acceptance_policy,
        round0_candidates=tuple(
            sorted(round0, key=lambda item: item.num_leapfrog_steps)
        ),
        refinement_candidates=tuple(
            sorted(refinement, key=lambda item: item.num_leapfrog_steps)
        ),
        disposition=disposition,
        shared_invalidity_reasons=shared_reasons,
        execution=execution_config,
    )
