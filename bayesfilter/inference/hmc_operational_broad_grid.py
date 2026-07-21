"""Operational broad fixed-mass HMC grid and neighbor-guard contracts.

TensorFlow/TFP callbacks owned by the application perform HMC transitions,
dual averaging, and fixed-kernel screens.  This module owns only immutable
lineage, uncertainty-aware screen classification, the non-directional primary
``L`` grid, exact-epsilon one-hop guards, and complete execution barriers.

All evidence handled here is discarded tuning evidence.  A viable pair is a
candidate for later frozen-kernel validation, not a retained sampler, a ranked
winner, or evidence of posterior convergence.
"""

from __future__ import annotations

import hashlib
import concurrent.futures
import importlib
import json
import math
import multiprocessing
import numbers
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


ROUTE_ID = "operational_broad_fixed_mass_l_epsilon_grid_v1"
CLASSIFICATION_POLICY_ID = "replication_mean_t90_band_compatibility_v1"
WORKING_INTERVAL_LEVEL = 0.90
WORKING_T_CRITICAL_DF2 = 2.919985580355516
PRIMARY_L_GRID = (3, 5, 9, 13, 18, 25)
MIN_GUARD_L = 2
MAX_L = 25
PRIMARY_ROLE = "independently_tuned_primary"
GUARD_ROLE = "same_epsilon_neighbor_guard"
MASS_UPDATE_DISPOSITIONS = (
    "dense_update",
    "fixed_identity",
    "diagonal_fallback",
    "no_update_insufficient_metric_evidence",
    "candidate_metric_rejected",
)
PAIR_DISPOSITIONS = (
    "hard_rejected",
    "needs_lower_epsilon",
    "needs_higher_epsilon",
    "provisional_viable",
    "unresolved_budget",
)
RESULT_DISPOSITIONS = (
    "mass_repair_required",
    "shared_execution_invalid",
    "viable_pair_set",
    "no_viable_pair",
    "inconclusive_evidence",
)
NONCLAIMS = (
    "discarded fixed-mass HMC tuning evidence only",
    "no stochastic candidate ranking",
    "no posterior convergence claim",
    "no retained-sampling readiness claim",
    "no sampler superiority or default-readiness claim",
    "no identification or scientific claim",
)
EXECUTION_MODES = ("serial", "process_parallel")


def _strict_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer scalar")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _nonempty(value: Any, *, name: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _strict_seed(value: Any, *, name: str) -> tuple[int, int]:
    try:
        items = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain two integer scalars") from error
    if len(items) != 2:
        raise ValueError(f"{name} must contain two integer scalars")
    return tuple(
        _strict_integer(item, name=f"{name} item", minimum=0) for item in items
    )


def _finite_step(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    return result


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


def operational_broad_seed(
    root_seed: tuple[int, int],
    *,
    domain: str,
    num_leapfrog_steps: int,
    epsilon: float | None = None,
    replication_index: int = 0,
) -> tuple[int, int]:
    """Derive an order-independent seed from the full role and pair identity."""

    root = _strict_seed(root_seed, name="root_seed")
    role_domain = _nonempty(domain, name="domain")
    leapfrog = _strict_integer(
        num_leapfrog_steps,
        name="num_leapfrog_steps",
        minimum=MIN_GUARD_L,
    )
    if leapfrog > MAX_L:
        raise ValueError("num_leapfrog_steps exceeds the reviewed L=25 bound")
    replication = _strict_integer(
        replication_index,
        name="replication_index",
        minimum=0,
    )
    epsilon_identity = "none" if epsilon is None else _finite_step(epsilon).hex()
    digest = hashlib.sha256(
        (
            f"{ROUTE_ID}:{root[0]}:{root[1]}:{role_domain}:"
            f"{leapfrog}:{epsilon_identity}:{replication}"
        ).encode("ascii")
    ).digest()
    modulus = 2**31 - 1
    seed = (
        (root[0] + int.from_bytes(digest[:8], "big")) % modulus,
        (root[1] + int.from_bytes(digest[8:16], "big")) % modulus,
    )
    return (0, 1) if seed == (0, 0) else seed


@dataclass(frozen=True)
class OperationalBroadGridPolicy:
    """Reviewed route controls; no directional L refinement is permitted."""

    root_seed: tuple[int, int]
    confirmation_num_results: int
    chain_count: int = 4
    replication_count: int = 3
    working_interval_level: float = WORKING_INTERVAL_LEVEL
    working_t_critical: float = WORKING_T_CRITICAL_DF2
    practical_region: tuple[float, float] = (0.65, 0.75)
    repair_region: tuple[float, float] = (0.55, 0.85)
    primary_l_grid: tuple[int, ...] = PRIMARY_L_GRID

    def __post_init__(self) -> None:
        root = _strict_seed(self.root_seed, name="root_seed")
        grid = tuple(
            _strict_integer(item, name="primary_l_grid item", minimum=3)
            for item in self.primary_l_grid
        )
        if grid != PRIMARY_L_GRID:
            raise ValueError(f"primary_l_grid must equal {PRIMARY_L_GRID}")
        results = _strict_integer(
            self.confirmation_num_results,
            name="confirmation_num_results",
            minimum=1,
        )
        if results <= 64:
            raise ValueError("confirmation_num_results must exceed the 64-draw nomination")
        chains = _strict_integer(self.chain_count, name="chain_count", minimum=1)
        replications = _strict_integer(
            self.replication_count,
            name="replication_count",
            minimum=1,
        )
        if replications != 3:
            raise ValueError("the reviewed broad route requires three replications")
        interval_level = float(self.working_interval_level)
        critical = float(self.working_t_critical)
        if not 0.0 < interval_level < 1.0:
            raise ValueError("working_interval_level must lie inside (0, 1)")
        if not math.isfinite(critical) or critical <= 0.0:
            raise ValueError("working_t_critical must be positive and finite")
        if (
            interval_level != WORKING_INTERVAL_LEVEL
            or critical != WORKING_T_CRITICAL_DF2
        ):
            raise ValueError(
                "classification policy requires the frozen 90% df=2 interval"
            )
        practical = tuple(float(item) for item in self.practical_region)
        repair = tuple(float(item) for item in self.repair_region)
        if not (
            len(practical) == 2
            and len(repair) == 2
            and 0.0 < repair[0] <= practical[0] <= practical[1] <= repair[1] < 1.0
        ):
            raise ValueError("acceptance regions are inconsistent")
        object.__setattr__(self, "root_seed", root)
        object.__setattr__(self, "primary_l_grid", grid)
        object.__setattr__(self, "confirmation_num_results", results)
        object.__setattr__(self, "chain_count", chains)
        object.__setattr__(self, "replication_count", replications)
        object.__setattr__(self, "working_interval_level", interval_level)
        object.__setattr__(self, "working_t_critical", critical)
        object.__setattr__(self, "practical_region", practical)
        object.__setattr__(self, "repair_region", repair)

    @property
    def evidence_unit_count(self) -> int:
        return self.chain_count * self.replication_count

    def payload(self) -> Mapping[str, Any]:
        return {
            "route": ROUTE_ID,
            "root_seed": self.root_seed,
            "primary_l_grid": self.primary_l_grid,
            "minimum_guard_l": MIN_GUARD_L,
            "maximum_l": MAX_L,
            "confirmation_num_results": self.confirmation_num_results,
            "chain_count": self.chain_count,
            "replication_count": self.replication_count,
            "classification_policy_id": CLASSIFICATION_POLICY_ID,
            "working_interval_level": self.working_interval_level,
            "working_t_critical": self.working_t_critical,
            "working_interval_unit": "fresh_seeded_replication_mean_across_chains",
            "practical_region": self.practical_region,
            "repair_region": self.repair_region,
            "guard_expansion": "one_hop_nonrecursive",
            "guard_epsilon_policy": "inherit_exact_primary_epsilon_no_retuning",
            "stochastic_ranking_performed": False,
        }


@dataclass(frozen=True)
class OperationalBroadGridExecutionConfig:
    """Opt-in spawn topology for independent primary and guard processes."""

    mode: str = "serial"
    primary_max_workers: int = 1
    guard_max_workers: int = 1
    primary_worker_factory_locator: str | None = None
    guard_worker_factory_locator: str | None = None
    worker_environment: tuple[tuple[str, str], ...] = ()
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        mode = str(self.mode)
        if mode not in EXECUTION_MODES:
            raise ValueError(f"mode must be one of {EXECUTION_MODES}")
        if str(self.start_method) != "spawn":
            raise ValueError("operational broad-grid process execution requires spawn")
        primary_workers = _strict_integer(
            self.primary_max_workers, name="primary_max_workers", minimum=1
        )
        guard_workers = _strict_integer(
            self.guard_max_workers, name="guard_max_workers", minimum=1
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "start_method", "spawn")
        object.__setattr__(self, "primary_max_workers", primary_workers)
        object.__setattr__(self, "guard_max_workers", guard_workers)
        if mode == "serial":
            if primary_workers != 1 or guard_workers != 1:
                raise ValueError("serial execution requires one worker per barrier")
            if self.primary_worker_factory_locator or self.guard_worker_factory_locator:
                raise ValueError("serial execution cannot declare worker factories")
            if self.worker_environment:
                raise ValueError("serial execution cannot declare a worker environment")
            return
        locators = (
            str(self.primary_worker_factory_locator or ""),
            str(self.guard_worker_factory_locator or ""),
        )
        if any(":" not in item for item in locators):
            raise ValueError("process execution requires module:factory locators")
        object.__setattr__(self, "primary_worker_factory_locator", locators[0])
        object.__setattr__(self, "guard_worker_factory_locator", locators[1])
        try:
            environment_items = tuple(
                (str(key), str(value)) for key, value in self.worker_environment
            )
        except (TypeError, ValueError) as error:
            raise ValueError("worker_environment must contain key/value pairs") from error
        if any(not key for key, _ in environment_items) or len(
            {key for key, _ in environment_items}
        ) != len(environment_items):
            raise ValueError("worker_environment keys must be non-empty and unique")
        environment = dict(environment_items)
        cuda_visible = environment.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is None or not cuda_visible:
            raise ValueError("process execution requires explicit CUDA_VISIBLE_DEVICES")
        if (
            cuda_visible != "-1"
            and environment.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true"
        ):
            raise ValueError("GPU workers require TF_FORCE_GPU_ALLOW_GROWTH=true")
        object.__setattr__(self, "worker_environment", tuple(sorted(environment_items)))

    def payload(self) -> Mapping[str, Any]:
        environment = dict(self.worker_environment)
        return {
            "mode": self.mode,
            "primary_max_workers": self.primary_max_workers,
            "guard_max_workers": self.guard_max_workers,
            "start_method": self.start_method,
            "primary_worker_factory_locator": self.primary_worker_factory_locator,
            "guard_worker_factory_locator": self.guard_worker_factory_locator,
            "worker_environment_keys": tuple(sorted(environment)),
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "tf_force_gpu_allow_growth": environment.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        }


@dataclass(frozen=True)
class OperationalMassHandoff:
    """Qualified dense-metric handoff from modern operational warmup."""

    update_disposition: str
    prior_metric_signature: str
    frozen_metric_signature: str
    coordinate_signature: str
    adapter_signature: str
    target_signature: str
    lineage_signature: str
    canonical_covariance_signature: str
    latent_metric_signature: str
    metric_evidence_signature: str
    retained_prior_metric: bool
    latent_identity_equivalence_proven: bool

    def __post_init__(self) -> None:
        disposition = str(self.update_disposition)
        if disposition not in MASS_UPDATE_DISPOSITIONS:
            raise ValueError("unsupported mass update disposition")
        object.__setattr__(self, "update_disposition", disposition)
        for name in (
            "prior_metric_signature",
            "frozen_metric_signature",
            "coordinate_signature",
            "adapter_signature",
            "target_signature",
            "lineage_signature",
            "canonical_covariance_signature",
            "latent_metric_signature",
            "metric_evidence_signature",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))
        retained = bool(self.retained_prior_metric)
        equivalence = bool(self.latent_identity_equivalence_proven)
        object.__setattr__(self, "retained_prior_metric", retained)
        object.__setattr__(self, "latent_identity_equivalence_proven", equivalence)
        if disposition == "dense_update":
            if retained:
                raise ValueError("dense_update cannot be labeled retained prior metric")
            if not equivalence:
                raise ValueError("dense_update requires canonical/latent equivalence")
        elif disposition == "fixed_identity":
            if not retained:
                raise ValueError("fixed_identity must retain the reviewed identity metric")
            if not equivalence:
                raise ValueError("fixed_identity requires canonical/latent equivalence")
            if self.frozen_metric_signature != self.prior_metric_signature:
                raise ValueError("fixed_identity metric signature changed")
        else:
            if not retained:
                raise ValueError("non-dense update must retain the qualified prior metric")
            if self.frozen_metric_signature != self.prior_metric_signature:
                raise ValueError("retained prior metric signature changed")

    @property
    def grid_ready(self) -> bool:
        dense_ready = (
            self.update_disposition == "dense_update"
            and not self.retained_prior_metric
        )
        fixed_identity_ready = (
            self.update_disposition == "fixed_identity"
            and self.retained_prior_metric
            and self.frozen_metric_signature == self.prior_metric_signature
        )
        return bool(
            (dense_ready or fixed_identity_ready)
            and self.latent_identity_equivalence_proven
        )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.operational_mass_handoff.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "update_disposition": self.update_disposition,
            "prior_metric_signature": self.prior_metric_signature,
            "frozen_metric_signature": self.frozen_metric_signature,
            "coordinate_signature": self.coordinate_signature,
            "adapter_signature": self.adapter_signature,
            "target_signature": self.target_signature,
            "lineage_signature": self.lineage_signature,
            "canonical_covariance_signature": self.canonical_covariance_signature,
            "latent_metric_signature": self.latent_metric_signature,
            "metric_evidence_signature": self.metric_evidence_signature,
            "retained_prior_metric": self.retained_prior_metric,
            "latent_identity_equivalence_proven": (
                self.latent_identity_equivalence_proven
            ),
            "grid_ready": self.grid_ready,
            "fixed_identity_metric_preserved": (
                self.update_disposition == "fixed_identity"
            ),
            "identity_metric_substituted": False,
        }


@dataclass(frozen=True)
class OperationalPairEvidence:
    """Uncertainty-aware tuning heuristic over fresh replication means."""

    chain_run_means: tuple[float, ...]
    replication_means: tuple[float, ...]
    grand_mean: float
    sample_standard_deviation: float
    standard_error: float
    working_interval: tuple[float, float]
    disposition: str
    hard_rejection_reasons: tuple[str, ...]
    evidence_signature: str

    @property
    def viable(self) -> bool:
        return self.disposition == "provisional_viable"

    def payload(self) -> Mapping[str, Any]:
        return {
            "chain_run_means": self.chain_run_means,
            "replication_means": self.replication_means,
            "grand_mean": self.grand_mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "standard_error": self.standard_error,
            "working_interval": self.working_interval,
            "disposition": self.disposition,
            "hard_rejection_reasons": self.hard_rejection_reasons,
            "evidence_signature": self.evidence_signature,
            "classification_policy_id": CLASSIFICATION_POLICY_ID,
            "working_interval_level": WORKING_INTERVAL_LEVEL,
            "working_interval_unit": "fresh_seeded_replication_mean_across_chains",
            "working_interval_role": (
                "statistical_compatibility_heuristic_not_in_band_proof"
            ),
            "working_interval_limitations": (
                "three_replications_only",
                "shared_calibrated_start",
                "student_t_working_model",
                "no_convergence_claim",
            ),
            "retained_sampling_authorized": False,
        }


def classify_operational_pair_evidence(
    *,
    chain_run_means: Sequence[float],
    evidence_signature: str,
    policy: OperationalBroadGridPolicy,
    hard_rejection_reasons: Sequence[str] = (),
) -> OperationalPairEvidence:
    """Classify whether replicated tuning evidence is compatible with the band."""

    if not isinstance(policy, OperationalBroadGridPolicy):
        raise TypeError("policy must be OperationalBroadGridPolicy")
    values = tuple(float(item) for item in chain_run_means)
    if len(values) != policy.evidence_unit_count or any(
        not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values
    ):
        raise ValueError("chain_run_means are incomplete or invalid")
    reasons = tuple(dict.fromkeys(str(item) for item in hard_rejection_reasons))
    if any(not item for item in reasons):
        raise ValueError("hard rejection reasons must be non-empty")
    replication_means = tuple(
        math.fsum(values[start : start + policy.chain_count]) / policy.chain_count
        for start in range(0, len(values), policy.chain_count)
    )
    if len(replication_means) != policy.replication_count:
        raise ValueError("replication means are incomplete")
    mean = math.fsum(replication_means) / len(replication_means)
    variance = math.fsum(
        (item - mean) ** 2 for item in replication_means
    ) / (len(replication_means) - 1)
    standard_deviation = math.sqrt(max(0.0, variance))
    standard_error = standard_deviation / math.sqrt(len(replication_means))
    half_width = policy.working_t_critical * standard_error
    interval = (max(0.0, mean - half_width), min(1.0, mean + half_width))
    practical_low, practical_high = policy.practical_region
    if reasons:
        disposition = "hard_rejected"
    elif interval[1] < practical_low:
        disposition = "needs_lower_epsilon"
    elif interval[0] > practical_high:
        disposition = "needs_higher_epsilon"
    else:
        disposition = "provisional_viable"
    return OperationalPairEvidence(
        chain_run_means=values,
        replication_means=replication_means,
        grand_mean=mean,
        sample_standard_deviation=standard_deviation,
        standard_error=standard_error,
        working_interval=interval,
        disposition=disposition,
        hard_rejection_reasons=reasons,
        evidence_signature=_nonempty(
            evidence_signature,
            name="evidence_signature",
        ),
    )


@dataclass(frozen=True)
class OperationalPrimaryRequest:
    num_leapfrog_steps: int
    tune_seed: tuple[int, int]
    mass_handoff_signature: str

    def __post_init__(self) -> None:
        leapfrog = _strict_integer(
            self.num_leapfrog_steps,
            name="num_leapfrog_steps",
            minimum=3,
        )
        if leapfrog not in PRIMARY_L_GRID:
            raise ValueError("primary L is outside the reviewed broad grid")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "tune_seed", _strict_seed(self.tune_seed, name="tune_seed"))
        object.__setattr__(
            self,
            "mass_handoff_signature",
            _nonempty(self.mass_handoff_signature, name="mass_handoff_signature"),
        )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.operational_primary_request.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "route": ROUTE_ID,
            "role": PRIMARY_ROLE,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "tune_seed": self.tune_seed,
            "mass_handoff_signature": self.mass_handoff_signature,
        }


@dataclass(frozen=True)
class OperationalPrimaryCandidate:
    request: OperationalPrimaryRequest
    tuned_step_size: float
    evidence: OperationalPairEvidence
    metric_signature: str
    coordinate_signature: str
    lineage_signature: str
    tune_evidence_signature: str

    def __post_init__(self) -> None:
        if not isinstance(self.request, OperationalPrimaryRequest):
            raise TypeError("request must be OperationalPrimaryRequest")
        if not isinstance(self.evidence, OperationalPairEvidence):
            raise TypeError("evidence must be OperationalPairEvidence")
        object.__setattr__(self, "tuned_step_size", _finite_step(self.tuned_step_size))
        for name in (
            "metric_signature",
            "coordinate_signature",
            "lineage_signature",
            "tune_evidence_signature",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

    @property
    def viable(self) -> bool:
        return self.evidence.viable

    @property
    def pair_identity(self) -> tuple[Any, ...]:
        return (
            ROUTE_ID,
            PRIMARY_ROLE,
            self.request.num_leapfrog_steps,
            self.tuned_step_size.hex(),
            self.metric_signature,
            self.coordinate_signature,
            self.lineage_signature,
        )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.operational_primary_candidate.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "request": self.request.payload(),
            "tuned_step_size": self.tuned_step_size,
            "evidence": self.evidence.payload(),
            "metric_signature": self.metric_signature,
            "coordinate_signature": self.coordinate_signature,
            "lineage_signature": self.lineage_signature,
            "tune_evidence_signature": self.tune_evidence_signature,
            "independent_epsilon_tuning_performed": True,
            "viable": self.viable,
        }


@dataclass(frozen=True)
class SameEpsilonNeighborGuardRequest:
    num_leapfrog_steps: int
    inherited_step_size: float
    parent_candidate_signatures: tuple[str, ...]
    parent_l_values: tuple[int, ...]
    screen_seeds: tuple[tuple[int, int], ...]
    mass_handoff_signature: str
    metric_signature: str
    coordinate_signature: str
    lineage_signature: str

    def __post_init__(self) -> None:
        leapfrog = _strict_integer(
            self.num_leapfrog_steps,
            name="num_leapfrog_steps",
            minimum=MIN_GUARD_L,
        )
        if leapfrog > MAX_L or leapfrog == 1:
            raise ValueError("guard L is outside the reviewed [2, 25] domain")
        parents = tuple(
            _strict_integer(item, name="parent_l_values item", minimum=3)
            for item in self.parent_l_values
        )
        if not parents or any(
            parent not in PRIMARY_L_GRID or abs(parent - leapfrog) != 1
            for parent in parents
        ):
            raise ValueError("guard must be bound one hop from a primary L")
        signatures = tuple(
            dict.fromkeys(
                _nonempty(item, name="parent candidate signature")
                for item in self.parent_candidate_signatures
            )
        )
        if len(signatures) != len(set(parents)):
            raise ValueError("guard parent signatures and L values must align")
        seeds = tuple(_strict_seed(item, name="screen seed") for item in self.screen_seeds)
        if len(seeds) != 3 or len(set(seeds)) != len(seeds):
            raise ValueError("guard requires three distinct screen seeds")
        object.__setattr__(self, "num_leapfrog_steps", leapfrog)
        object.__setattr__(self, "inherited_step_size", _finite_step(self.inherited_step_size))
        object.__setattr__(self, "parent_candidate_signatures", signatures)
        object.__setattr__(self, "parent_l_values", tuple(sorted(set(parents))))
        object.__setattr__(self, "screen_seeds", seeds)
        for name in (
            "mass_handoff_signature",
            "metric_signature",
            "coordinate_signature",
            "lineage_signature",
        ):
            object.__setattr__(self, name, _nonempty(getattr(self, name), name=name))

    @property
    def pair_identity(self) -> tuple[Any, ...]:
        return (
            ROUTE_ID,
            GUARD_ROLE,
            self.num_leapfrog_steps,
            self.inherited_step_size.hex(),
            self.metric_signature,
            self.coordinate_signature,
            self.lineage_signature,
        )

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.same_epsilon_neighbor_guard_request.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "route": ROUTE_ID,
            "role": GUARD_ROLE,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "inherited_step_size": self.inherited_step_size,
            "parent_candidate_signatures": self.parent_candidate_signatures,
            "parent_l_values": self.parent_l_values,
            "screen_seeds": self.screen_seeds,
            "mass_handoff_signature": self.mass_handoff_signature,
            "metric_signature": self.metric_signature,
            "coordinate_signature": self.coordinate_signature,
            "lineage_signature": self.lineage_signature,
            "epsilon_retuned": False,
            "recursive_expansion_allowed": False,
        }


@dataclass(frozen=True)
class SameEpsilonNeighborGuard:
    request: SameEpsilonNeighborGuardRequest
    evidence: OperationalPairEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.request, SameEpsilonNeighborGuardRequest):
            raise TypeError("request must be SameEpsilonNeighborGuardRequest")
        if not isinstance(self.evidence, OperationalPairEvidence):
            raise TypeError("evidence must be OperationalPairEvidence")

    @property
    def viable(self) -> bool:
        return self.evidence.viable

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.same_epsilon_neighbor_guard.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "request": self.request.payload(),
            "evidence": self.evidence.payload(),
            "viable": self.viable,
            "independently_tuned": False,
        }


def primary_requests(
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> tuple[OperationalPrimaryRequest, ...]:
    if not isinstance(policy, OperationalBroadGridPolicy):
        raise TypeError("policy must be OperationalBroadGridPolicy")
    if not isinstance(handoff, OperationalMassHandoff):
        raise TypeError("handoff must be OperationalMassHandoff")
    if not handoff.grid_ready:
        return ()
    return tuple(
        OperationalPrimaryRequest(
            num_leapfrog_steps=leapfrog,
            tune_seed=operational_broad_seed(
                policy.root_seed,
                domain="primary_independent_epsilon_tune",
                num_leapfrog_steps=leapfrog,
            ),
            mass_handoff_signature=handoff.signature,
        )
        for leapfrog in policy.primary_l_grid
    )


def expand_same_epsilon_neighbor_guards(
    primary_candidates: Sequence[OperationalPrimaryCandidate],
    *,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> tuple[SameEpsilonNeighborGuardRequest, ...]:
    """Expand viable primaries once and deduplicate by complete guard pair."""

    if not handoff.grid_ready:
        raise ValueError("neighbor guards require a grid-ready mass handoff")
    grouped: dict[tuple[Any, ...], list[OperationalPrimaryCandidate]] = {}
    for candidate in primary_candidates:
        if not isinstance(candidate, OperationalPrimaryCandidate):
            raise TypeError("primary_candidates must contain primary records")
        if not candidate.viable:
            continue
        if (
            candidate.metric_signature != handoff.frozen_metric_signature
            or candidate.coordinate_signature != handoff.coordinate_signature
            or candidate.lineage_signature != handoff.lineage_signature
            or candidate.request.mass_handoff_signature != handoff.signature
        ):
            raise ValueError("primary candidate drifted from the frozen mass handoff")
        parent_l = candidate.request.num_leapfrog_steps
        for neighbor_l in (parent_l - 1, parent_l + 1):
            if MIN_GUARD_L <= neighbor_l <= MAX_L:
                identity = (
                    neighbor_l,
                    candidate.tuned_step_size.hex(),
                    candidate.metric_signature,
                    candidate.coordinate_signature,
                    candidate.lineage_signature,
                )
                grouped.setdefault(identity, []).append(candidate)
    requests: list[SameEpsilonNeighborGuardRequest] = []
    for identity, parents in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        neighbor_l = int(identity[0])
        epsilon = parents[0].tuned_step_size
        requests.append(
            SameEpsilonNeighborGuardRequest(
                num_leapfrog_steps=neighbor_l,
                inherited_step_size=epsilon,
                parent_candidate_signatures=tuple(
                    sorted(candidate.signature for candidate in parents)
                ),
                parent_l_values=tuple(
                    sorted(candidate.request.num_leapfrog_steps for candidate in parents)
                ),
                screen_seeds=tuple(
                    operational_broad_seed(
                        policy.root_seed,
                        domain="same_epsilon_neighbor_guard_screen",
                        num_leapfrog_steps=neighbor_l,
                        epsilon=epsilon,
                        replication_index=index,
                    )
                    for index in range(policy.replication_count)
                ),
                mass_handoff_signature=handoff.signature,
                metric_signature=handoff.frozen_metric_signature,
                coordinate_signature=handoff.coordinate_signature,
                lineage_signature=handoff.lineage_signature,
            )
        )
    return tuple(requests)


@dataclass(frozen=True)
class OperationalBarrier:
    stage: str
    planned_signatures: tuple[str, ...]
    completed_signatures: tuple[str, ...]
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        stage = _nonempty(self.stage, name="stage")
        planned = tuple(_nonempty(item, name="planned signature") for item in self.planned_signatures)
        completed = tuple(
            _nonempty(item, name="completed signature") for item in self.completed_signatures
        )
        failures = tuple(dict.fromkeys(str(item) for item in self.failure_reasons))
        if len(set(planned)) != len(planned) or len(set(completed)) != len(completed):
            raise ValueError("barrier signatures must be unique")
        if not set(completed).issubset(planned):
            raise ValueError("barrier completed an unplanned request")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "planned_signatures", planned)
        object.__setattr__(self, "completed_signatures", completed)
        object.__setattr__(self, "failure_reasons", failures)

    @property
    def complete(self) -> bool:
        return (
            not self.failure_reasons
            and set(self.completed_signatures) == set(self.planned_signatures)
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "stage": self.stage,
            "planned_count": len(self.planned_signatures),
            "completed_count": len(self.completed_signatures),
            "failure_reasons": self.failure_reasons,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class OperationalBroadGridResult:
    policy: OperationalBroadGridPolicy
    mass_handoff: OperationalMassHandoff
    primary_candidates: tuple[OperationalPrimaryCandidate, ...]
    guard_candidates: tuple[SameEpsilonNeighborGuard, ...]
    primary_barrier: OperationalBarrier
    guard_barrier: OperationalBarrier
    disposition: str
    execution: OperationalBroadGridExecutionConfig = OperationalBroadGridExecutionConfig()

    @property
    def viable_primary_candidates(self) -> tuple[OperationalPrimaryCandidate, ...]:
        return tuple(item for item in self.primary_candidates if item.viable)

    @property
    def viable_guard_candidates(self) -> tuple[SameEpsilonNeighborGuard, ...]:
        return tuple(item for item in self.guard_candidates if item.viable)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.operational_broad_grid.private.v1",
            "route": ROUTE_ID,
            "policy": self.policy.payload(),
            "mass_handoff": self.mass_handoff.payload(),
            "primary_candidates": tuple(item.payload() for item in self.primary_candidates),
            "guard_candidates": tuple(item.payload() for item in self.guard_candidates),
            "primary_barrier": self.primary_barrier.payload(),
            "guard_barrier": self.guard_barrier.payload(),
            "disposition": self.disposition,
            "execution": self.execution.payload(),
            "viable_primary_count": len(self.viable_primary_candidates),
            "viable_guard_count": len(self.viable_guard_candidates),
            "all_viable_pairs_preserved": True,
            "representative": None,
            "stochastic_ranking_performed": False,
            "all_tuning_draws_discarded": True,
            "retained_sampling_authorized": False,
            "nonclaims": NONCLAIMS,
        }

    def public_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.operational_broad_grid.public.v1",
            "route": ROUTE_ID,
            "disposition": self.disposition,
            "execution": self.execution.payload(),
            "mass_handoff_ready": self.mass_handoff.grid_ready,
            "planned_primary_count": len(self.primary_barrier.planned_signatures),
            "completed_primary_count": len(self.primary_barrier.completed_signatures),
            "planned_guard_count": len(self.guard_barrier.planned_signatures),
            "completed_guard_count": len(self.guard_barrier.completed_signatures),
            "viable_primary_count": len(self.viable_primary_candidates),
            "viable_guard_count": len(self.viable_guard_candidates),
            "stochastic_ranking_performed": False,
            "retained_sampling_authorized": False,
            "raw_samples_exposed": False,
            "raw_states_exposed": False,
            "epsilon_values_exposed": False,
            "metric_arrays_exposed": False,
            "nonclaims": NONCLAIMS,
        }


def assemble_operational_broad_grid_result(
    *,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
    primary_candidates: Sequence[OperationalPrimaryCandidate] = (),
    guard_candidates: Sequence[SameEpsilonNeighborGuard] = (),
    primary_failure_reasons: Sequence[str] = (),
    guard_failure_reasons: Sequence[str] = (),
    execution: OperationalBroadGridExecutionConfig | None = None,
) -> OperationalBroadGridResult:
    """Validate serial or process-parallel records against both barriers."""

    execution_config = OperationalBroadGridExecutionConfig() if execution is None else execution
    if not isinstance(execution_config, OperationalBroadGridExecutionConfig):
        raise TypeError("execution must be OperationalBroadGridExecutionConfig")
    planned_primary = primary_requests(policy, handoff)
    primary_by_request: dict[str, OperationalPrimaryCandidate] = {}
    for candidate in primary_candidates:
        if candidate.request.signature in primary_by_request:
            raise ValueError("duplicate primary candidate request")
        primary_by_request[candidate.request.signature] = candidate
    primary_barrier = OperationalBarrier(
        stage="primary_independent_epsilon_grid",
        planned_signatures=tuple(item.signature for item in planned_primary),
        completed_signatures=tuple(primary_by_request),
        failure_reasons=tuple(primary_failure_reasons),
    )
    ordered_primary = tuple(
        primary_by_request[item.signature]
        for item in planned_primary
        if item.signature in primary_by_request
    )
    expected_guards = (
        expand_same_epsilon_neighbor_guards(
            ordered_primary,
            policy=policy,
            handoff=handoff,
        )
        if primary_barrier.complete
        else ()
    )
    guard_by_request: dict[str, SameEpsilonNeighborGuard] = {}
    for guard in guard_candidates:
        if guard.request.signature in guard_by_request:
            raise ValueError("duplicate neighbor-guard request")
        guard_by_request[guard.request.signature] = guard
    guard_barrier = OperationalBarrier(
        stage="same_epsilon_neighbor_guards",
        planned_signatures=tuple(item.signature for item in expected_guards),
        completed_signatures=tuple(guard_by_request),
        failure_reasons=tuple(guard_failure_reasons),
    )
    ordered_guards = tuple(
        guard_by_request[item.signature]
        for item in expected_guards
        if item.signature in guard_by_request
    )
    if not handoff.grid_ready:
        disposition = "mass_repair_required"
    elif not primary_barrier.complete or not guard_barrier.complete:
        disposition = "shared_execution_invalid"
    elif any(item.viable for item in ordered_primary + ordered_guards):
        disposition = "viable_pair_set"
    elif any(
        item.evidence.disposition == "unresolved_budget"
        for item in ordered_primary + ordered_guards
    ):
        disposition = "inconclusive_evidence"
    else:
        disposition = "no_viable_pair"
    return OperationalBroadGridResult(
        policy=policy,
        mass_handoff=handoff,
        primary_candidates=ordered_primary,
        guard_candidates=ordered_guards,
        primary_barrier=primary_barrier,
        guard_barrier=guard_barrier,
        disposition=disposition,
        execution=execution_config,
    )


PrimaryRunner = Callable[[OperationalPrimaryRequest], OperationalPrimaryCandidate]
GuardRunner = Callable[[SameEpsilonNeighborGuardRequest], SameEpsilonNeighborGuard]


def run_operational_broad_grid(
    *,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
    primary_runner: PrimaryRunner,
    guard_runner: GuardRunner,
    execution: OperationalBroadGridExecutionConfig | None = None,
) -> OperationalBroadGridResult:
    """Run the serial reference orchestration used to verify parallel semantics."""

    execution_config = OperationalBroadGridExecutionConfig() if execution is None else execution
    if execution_config.mode != "serial":
        raise ValueError("callback execution is serial; use run_operational_broad_grid_process_parallel")
    if not callable(primary_runner) or not callable(guard_runner):
        raise TypeError("primary_runner and guard_runner must be callable")
    if not handoff.grid_ready:
        return assemble_operational_broad_grid_result(
            policy=policy, handoff=handoff, execution=execution_config
        )
    primary: list[OperationalPrimaryCandidate] = []
    primary_failures: list[str] = []
    for request in primary_requests(policy, handoff):
        try:
            candidate = primary_runner(request)
            if not isinstance(candidate, OperationalPrimaryCandidate):
                raise TypeError("primary runner returned an invalid record")
            if candidate.request.signature != request.signature:
                raise ValueError("primary runner changed request identity")
            primary.append(candidate)
        except Exception as error:  # noqa: BLE001 - recorded barrier invalidity.
            primary_failures.append(f"{type(error).__name__}: {error}")
            break
    partial = assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primary,
        primary_failure_reasons=primary_failures,
        execution=execution_config,
    )
    if not partial.primary_barrier.complete:
        return partial
    guards: list[SameEpsilonNeighborGuard] = []
    guard_failures: list[str] = []
    requests = expand_same_epsilon_neighbor_guards(
        primary,
        policy=policy,
        handoff=handoff,
    )
    for request in requests:
        try:
            guard = guard_runner(request)
            if not isinstance(guard, SameEpsilonNeighborGuard):
                raise TypeError("guard runner returned an invalid record")
            if guard.request.signature != request.signature:
                raise ValueError("guard runner changed request identity")
            guards.append(guard)
        except Exception as error:  # noqa: BLE001 - recorded barrier invalidity.
            guard_failures.append(f"{type(error).__name__}: {error}")
            break
    return assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primary,
        guard_candidates=guards,
        guard_failure_reasons=guard_failures,
        execution=execution_config,
    )


def _resolve_factory(locator: str) -> Callable[..., Any]:
    module_name, attribute_path = locator.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for component in attribute_path.split("."):
        value = getattr(value, component)
    if not callable(value):
        raise TypeError("worker factory locator did not resolve to a callable")
    return value


def _primary_process_worker(
    locator: str,
    request: OperationalPrimaryRequest,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> tuple[str, OperationalPrimaryCandidate | None, str | None]:
    try:
        candidate = _resolve_factory(locator)(request, policy, handoff)
        if not isinstance(candidate, OperationalPrimaryCandidate):
            raise TypeError("primary worker returned an invalid record")
        if candidate.request.signature != request.signature:
            raise ValueError("primary worker changed request identity")
        return ("complete", candidate, None)
    except Exception as error:  # noqa: BLE001 - typed barrier closeout.
        return ("failed", None, f"{type(error).__name__}: {error}")


def _guard_process_worker(
    locator: str,
    request: SameEpsilonNeighborGuardRequest,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
) -> tuple[str, SameEpsilonNeighborGuard | None, str | None]:
    try:
        guard = _resolve_factory(locator)(request, policy, handoff)
        if not isinstance(guard, SameEpsilonNeighborGuard):
            raise TypeError("guard worker returned an invalid record")
        if guard.request.signature != request.signature:
            raise ValueError("guard worker changed request identity")
        return ("complete", guard, None)
    except Exception as error:  # noqa: BLE001 - typed barrier closeout.
        return ("failed", None, f"{type(error).__name__}: {error}")


@contextmanager
def _temporary_worker_environment(items: Sequence[tuple[str, str]]):
    prior = {key: os.environ.get(key) for key, _ in items}
    try:
        for key, value in items:
            os.environ[key] = value
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _process_barrier(
    worker: Callable[..., tuple[str, Any, str | None]],
    locator: str,
    requests: Sequence[Any],
    *,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
    max_workers: int,
    environment: Sequence[tuple[str, str]],
) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    if not requests:
        return (), ()
    outcomes: dict[str, tuple[str, Any, str | None]] = {}
    context = multiprocessing.get_context("spawn")
    with _temporary_worker_environment(environment):
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=context,
        ) as executor:
            future_to_request = {
                executor.submit(worker, locator, request, policy, handoff): request
                for request in requests
            }
            for future, request in future_to_request.items():
                try:
                    outcomes[request.signature] = future.result()
                except BaseException as error:  # noqa: BLE001 - broken process barrier.
                    outcomes[request.signature] = (
                        "failed",
                        None,
                        f"{type(error).__name__}: {error}",
                    )
    completed: list[Any] = []
    failures: list[str] = []
    for request in requests:
        status, record, message = outcomes[request.signature]
        if status == "complete":
            completed.append(record)
        else:
            failures.append(str(message))
    return tuple(completed), tuple(failures)


def run_operational_broad_grid_process_parallel(
    *,
    policy: OperationalBroadGridPolicy,
    handoff: OperationalMassHandoff,
    execution: OperationalBroadGridExecutionConfig,
) -> OperationalBroadGridResult:
    """Run two complete spawn barriers, with guards dependent on primaries."""

    if not isinstance(execution, OperationalBroadGridExecutionConfig):
        raise TypeError("execution must be OperationalBroadGridExecutionConfig")
    if execution.mode != "process_parallel":
        raise ValueError("process-parallel runner requires process_parallel mode")
    if not handoff.grid_ready:
        return assemble_operational_broad_grid_result(
            policy=policy, handoff=handoff, execution=execution
        )
    primary, primary_failures = _process_barrier(
        _primary_process_worker,
        str(execution.primary_worker_factory_locator),
        primary_requests(policy, handoff),
        policy=policy,
        handoff=handoff,
        max_workers=execution.primary_max_workers,
        environment=execution.worker_environment,
    )
    partial = assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primary,
        primary_failure_reasons=primary_failures,
        execution=execution,
    )
    if not partial.primary_barrier.complete:
        return partial
    guard_requests = expand_same_epsilon_neighbor_guards(
        primary, policy=policy, handoff=handoff
    )
    guards, guard_failures = _process_barrier(
        _guard_process_worker,
        str(execution.guard_worker_factory_locator),
        guard_requests,
        policy=policy,
        handoff=handoff,
        max_workers=execution.guard_max_workers,
        environment=execution.worker_environment,
    )
    return assemble_operational_broad_grid_result(
        policy=policy,
        handoff=handoff,
        primary_candidates=primary,
        guard_candidates=guards,
        guard_failure_reasons=guard_failures,
        execution=execution,
    )
