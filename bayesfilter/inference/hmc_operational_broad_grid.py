"""Operational broad fixed-mass HMC grid and one-hop coverage contracts.

TensorFlow/TFP callbacks owned by the application perform HMC transitions,
dual averaging, and fixed-kernel screens.  This module owns only immutable
lineage, uncertainty-aware screen classification, the non-directional primary
``L`` grid, exact-epsilon one-hop coverage probes, and complete execution
barriers.

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
WORKING_T_CRITICAL_DF4 = 2.1318467863266495
PRIMARY_L_GRID = (3, 5, 9, 13, 18, 25)
MIN_GUARD_L = 2
MAX_L = 25
PRIMARY_ROLE = "independently_tuned_primary"
NEIGHBOR_COVERAGE_ROLE = "same_epsilon_neighbor_coverage"
# Preserve the v1 serialized request role for historical lineage.  The active
# scientific role is exposed separately as coverage, not parent promotion.
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
STATISTICAL_EPSILON_REPAIR_POLICY_ID = (
    "replication_mean_t90_df4_candidate_nomination_epsilon_repair_v2"
)
STATISTICAL_EPSILON_REPAIR_TERMINAL_DISPOSITIONS = (
    "freeze_for_qualification",
    "repair_epsilon",
    "tuning_unresolved",
    "hard_rejected",
    "attempt_budget_exhausted",
    "bracket_conflict",
    "stalled_repeated_or_negligible",
)


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
class OperationalStatisticalEpsilonRepairPolicy:
    """Prospective CCMA epsilon-repair controls over replication means."""

    tuning_root_seed: tuple[int, int]
    qualification_root_seed: tuple[int, int]
    tuning_num_results: int = 64
    qualification_num_results: int = 128
    chain_count: int = 4
    replication_count: int = 5
    working_interval_level: float = WORKING_INTERVAL_LEVEL
    working_t_critical: float = WORKING_T_CRITICAL_DF4
    practical_region: tuple[float, float] = (0.65, 0.75)
    repair_region: tuple[float, float] = (0.55, 0.85)
    maximum_attempts: int = 5
    repair_factor: float = 1.25
    minimum_relative_step_change: float = 0.01

    def __post_init__(self) -> None:
        tuning_root = _strict_seed(self.tuning_root_seed, name="tuning_root_seed")
        qualification_root = _strict_seed(self.qualification_root_seed, name="qualification_root_seed")
        if tuning_root == qualification_root:
            raise ValueError("tuning and qualification root seeds must be disjoint")
        tuning_results = _strict_integer(self.tuning_num_results, name="tuning_num_results", minimum=64)
        qualification_results = _strict_integer(self.qualification_num_results, name="qualification_num_results", minimum=tuning_results)
        chains = _strict_integer(self.chain_count, name="chain_count", minimum=1)
        replications = _strict_integer(self.replication_count, name="replication_count", minimum=2)
        if chains != 4 or replications != 5:
            raise ValueError("statistical epsilon repair requires four chains and five replications")
        interval_level = float(self.working_interval_level)
        critical = float(self.working_t_critical)
        if interval_level != WORKING_INTERVAL_LEVEL or critical != WORKING_T_CRITICAL_DF4:
            raise ValueError("statistical epsilon repair requires the frozen 90% df=4 interval")
        practical = tuple(float(item) for item in self.practical_region)
        if not (len(practical) == 2 and 0.0 < practical[0] < practical[1] < 1.0):
            raise ValueError("practical_region must be ordered inside (0, 1)")
        repair = tuple(float(item) for item in self.repair_region)
        if not (len(repair) == 2 and 0.0 < repair[0] < repair[1] < 1.0):
            raise ValueError("repair_region must be ordered inside (0, 1)")
        if not (repair[0] < practical[0] < practical[1] < repair[1]):
            raise ValueError("repair_region must contain practical_region")
        attempts = _strict_integer(self.maximum_attempts, name="maximum_attempts", minimum=1)
        if attempts > 5:
            raise ValueError("maximum_attempts must not exceed five")
        factor = float(self.repair_factor)
        if not math.isfinite(factor) or not 1.0 < factor <= 2.0:
            raise ValueError("repair_factor must lie inside (1, 2]")
        minimum_change = float(self.minimum_relative_step_change)
        if not math.isfinite(minimum_change) or not 0.0 < minimum_change < 1.0:
            raise ValueError("minimum_relative_step_change must lie inside (0, 1)")
        object.__setattr__(self, "tuning_root_seed", tuning_root)
        object.__setattr__(self, "qualification_root_seed", qualification_root)
        object.__setattr__(self, "tuning_num_results", tuning_results)
        object.__setattr__(self, "qualification_num_results", qualification_results)
        object.__setattr__(self, "chain_count", chains)
        object.__setattr__(self, "replication_count", replications)
        object.__setattr__(self, "working_interval_level", interval_level)
        object.__setattr__(self, "working_t_critical", critical)
        object.__setattr__(self, "practical_region", practical)
        object.__setattr__(self, "repair_region", repair)
        object.__setattr__(self, "maximum_attempts", attempts)
        object.__setattr__(self, "repair_factor", factor)
        object.__setattr__(self, "minimum_relative_step_change", minimum_change)

    @property
    def evidence_unit_count(self) -> int:
        return self.chain_count * self.replication_count

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy_id": STATISTICAL_EPSILON_REPAIR_POLICY_ID,
            "tuning_root_seed": self.tuning_root_seed,
            "qualification_root_seed": self.qualification_root_seed,
            "tuning_num_results": self.tuning_num_results,
            "qualification_num_results": self.qualification_num_results,
            "chain_count": self.chain_count,
            "replication_count": self.replication_count,
            "working_interval_level": self.working_interval_level,
            "working_t_critical": self.working_t_critical,
            "working_interval_unit": "fresh_seeded_replication_mean_across_four_chains",
            "practical_region": self.practical_region,
            "repair_region": self.repair_region,
            "maximum_attempts": self.maximum_attempts,
            "repair_factor": self.repair_factor,
            "minimum_relative_step_change": self.minimum_relative_step_change,
            "tuning_evidence_role": "adaptive_diagnostic_only",
            "qualification_evidence_role": "single_use_admission_screen",
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
    grand_mean: float | None
    sample_standard_deviation: float | None
    standard_error: float | None
    working_interval: tuple[float, float] | None
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
    """Classify replicated tuning evidence without promoting partial overlap."""

    if not isinstance(policy, OperationalBroadGridPolicy):
        raise TypeError("policy must be OperationalBroadGridPolicy")
    reasons = tuple(dict.fromkeys(str(item) for item in hard_rejection_reasons))
    if any(not item for item in reasons):
        raise ValueError("hard rejection reasons must be non-empty")
    values = tuple(float(item) for item in chain_run_means)
    if not values:
        if not reasons:
            raise ValueError("empty chain_run_means require a hard rejection reason")
        return OperationalPairEvidence(
            chain_run_means=(),
            replication_means=(),
            grand_mean=None,
            sample_standard_deviation=None,
            standard_error=None,
            working_interval=None,
            disposition="hard_rejected",
            hard_rejection_reasons=reasons,
            evidence_signature=_nonempty(
                evidence_signature,
                name="evidence_signature",
            ),
        )
    if len(values) != policy.evidence_unit_count or any(
        not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values
    ):
        raise ValueError("chain_run_means are incomplete or invalid")
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
    elif practical_low <= interval[0] and interval[1] <= practical_high:
        disposition = "provisional_viable"
    else:
        disposition = "unresolved_budget"
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
class OperationalStatisticalEpsilonEvidence:
    """Five-replication working evidence for tuning or qualification."""

    chain_run_means: tuple[float, ...]
    replication_means: tuple[float, ...]
    grand_mean: float | None
    sample_standard_deviation: float | None
    standard_error: float | None
    working_interval: tuple[float, float] | None
    disposition: str
    hard_rejection_reasons: tuple[str, ...]
    evidence_signature: str

    @property
    def admitted(self) -> bool:
        return self.disposition == "candidate_nominated"

    @property
    def candidate_nominated(self) -> bool:
        return self.disposition == "candidate_nominated"

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
            "classification_policy_id": STATISTICAL_EPSILON_REPAIR_POLICY_ID,
            "working_interval_level": WORKING_INTERVAL_LEVEL,
            "working_interval_unit": "fresh_seeded_replication_mean_across_four_chains",
            "working_interval_role": "candidate_nomination_compatibility_diagnostic_not_equivalence_proof",
            "working_interval_limitations": (
                "five_replications_only",
                "shared_calibrated_start",
                "student_t_working_model",
                "no_familywise_claim",
                "no_convergence_claim",
            ),
            "retained_sampling_authorized": False,
        }


def classify_operational_statistical_epsilon_evidence(
    *,
    chain_run_means: Sequence[float],
    evidence_signature: str,
    policy: OperationalStatisticalEpsilonRepairPolicy,
    hard_rejection_reasons: Sequence[str] = (),
) -> OperationalStatisticalEpsilonEvidence:
    """Classify five fresh replication means under the reviewed working model."""

    if not isinstance(policy, OperationalStatisticalEpsilonRepairPolicy):
        raise TypeError("policy must be OperationalStatisticalEpsilonRepairPolicy")
    reasons = tuple(dict.fromkeys(str(item) for item in hard_rejection_reasons))
    if any(not item for item in reasons):
        raise ValueError("hard rejection reasons must be non-empty")
    values = tuple(float(item) for item in chain_run_means)
    if not values:
        if not reasons:
            raise ValueError("empty chain_run_means require a hard rejection reason")
        return OperationalStatisticalEpsilonEvidence(
            chain_run_means=(),
            replication_means=(),
            grand_mean=None,
            sample_standard_deviation=None,
            standard_error=None,
            working_interval=None,
            disposition="hard_rejected",
            hard_rejection_reasons=reasons,
            evidence_signature=_nonempty(evidence_signature, name="evidence_signature"),
        )
    if len(values) != policy.evidence_unit_count or any(
        not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in values
    ):
        raise ValueError("chain_run_means are incomplete or invalid")
    replication_means = tuple(
        math.fsum(values[start : start + policy.chain_count]) / policy.chain_count
        for start in range(0, len(values), policy.chain_count)
    )
    mean = math.fsum(replication_means) / len(replication_means)
    variance = math.fsum((item - mean) ** 2 for item in replication_means) / (len(replication_means) - 1)
    standard_deviation = math.sqrt(max(0.0, variance))
    standard_error = standard_deviation / math.sqrt(len(replication_means))
    half_width = policy.working_t_critical * standard_error
    interval = (max(0.0, mean - half_width), min(1.0, mean + half_width))
    practical_low, practical_high = policy.practical_region
    repair_low, repair_high = policy.repair_region
    if reasons:
        disposition = "hard_rejected"
    elif interval[1] < practical_low:
        disposition = "needs_lower_epsilon"
    elif interval[0] > practical_high:
        disposition = "needs_higher_epsilon"
    elif repair_low <= interval[0] and interval[1] <= repair_high and (
        interval[1] >= practical_low and interval[0] <= practical_high
    ):
        disposition = "candidate_nominated"
    else:
        disposition = "unresolved_budget"
    return OperationalStatisticalEpsilonEvidence(
        chain_run_means=values,
        replication_means=replication_means,
        grand_mean=mean,
        sample_standard_deviation=standard_deviation,
        standard_error=standard_error,
        working_interval=interval,
        disposition=disposition,
        hard_rejection_reasons=reasons,
        evidence_signature=_nonempty(evidence_signature, name="evidence_signature"),
    )


@dataclass(frozen=True)
class OperationalStatisticalEpsilonRepairDecision:
    """One append-only controller transition after a complete tuning attempt."""

    attempt_index: int
    current_epsilon: float
    evidence_disposition: str
    terminal_disposition: str
    bracket_before: tuple[float | None, float | None]
    bracket_after: tuple[float | None, float | None]
    direction: str | None
    next_epsilon: float | None

    def __post_init__(self) -> None:
        index = _strict_integer(self.attempt_index, name="attempt_index", minimum=0)
        current = _finite_step(self.current_epsilon)
        terminal = str(self.terminal_disposition)
        if terminal not in STATISTICAL_EPSILON_REPAIR_TERMINAL_DISPOSITIONS:
            raise ValueError("invalid statistical epsilon-repair disposition")
        before = _validated_epsilon_bracket(self.bracket_before)
        after = _validated_epsilon_bracket(self.bracket_after)
        direction = self.direction
        if direction is not None and direction not in {"higher_epsilon", "lower_epsilon"}:
            raise ValueError("invalid epsilon-repair direction")
        next_epsilon = None if self.next_epsilon is None else _finite_step(self.next_epsilon)
        if terminal == "repair_epsilon":
            if direction is None or next_epsilon is None:
                raise ValueError("repair_epsilon requires direction and next epsilon")
        elif next_epsilon is not None:
            raise ValueError("terminal epsilon decision cannot carry a next epsilon")
        object.__setattr__(self, "attempt_index", index)
        object.__setattr__(self, "current_epsilon", current)
        object.__setattr__(self, "evidence_disposition", str(self.evidence_disposition))
        object.__setattr__(self, "terminal_disposition", terminal)
        object.__setattr__(self, "bracket_before", before)
        object.__setattr__(self, "bracket_after", after)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "next_epsilon", next_epsilon)

    def payload(self) -> Mapping[str, Any]:
        return {
            "attempt_index": self.attempt_index,
            "current_epsilon": self.current_epsilon,
            "evidence_disposition": self.evidence_disposition,
            "terminal_disposition": self.terminal_disposition,
            "bracket_before": self.bracket_before,
            "bracket_after": self.bracket_after,
            "direction": self.direction,
            "next_epsilon": self.next_epsilon,
            "tuning_evidence_role": "adaptive_diagnostic_only",
            "qualification_reuse_allowed": False,
        }


def _validated_epsilon_bracket(
    bracket: tuple[float | None, float | None],
) -> tuple[float | None, float | None]:
    try:
        lower, upper = tuple(bracket)
    except (TypeError, ValueError) as error:
        raise ValueError("epsilon bracket must contain two bounds") from error
    lower = None if lower is None else _finite_step(lower)
    upper = None if upper is None else _finite_step(upper)
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("epsilon bracket is inverted or empty")
    return lower, upper


def advance_operational_statistical_epsilon_repair(
    *,
    evidence: OperationalStatisticalEpsilonEvidence,
    current_epsilon: float,
    attempt_index: int,
    bracket_before: tuple[float | None, float | None],
    attempted_epsilons: Sequence[float],
    policy: OperationalStatisticalEpsilonRepairPolicy,
) -> OperationalStatisticalEpsilonRepairDecision:
    """Advance a bounded tuning-only epsilon bracket from interval evidence."""

    if not isinstance(evidence, OperationalStatisticalEpsilonEvidence):
        raise TypeError("evidence must be OperationalStatisticalEpsilonEvidence")
    if not isinstance(policy, OperationalStatisticalEpsilonRepairPolicy):
        raise TypeError("policy must be OperationalStatisticalEpsilonRepairPolicy")
    current = _finite_step(current_epsilon)
    index = _strict_integer(attempt_index, name="attempt_index", minimum=0)
    history = tuple(_finite_step(item) for item in attempted_epsilons)
    if len(history) != index + 1 or not math.isclose(history[-1], current, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError("attempted_epsilons must end at the indexed current epsilon")
    lower, upper = _validated_epsilon_bracket(bracket_before)
    disposition = evidence.disposition
    if disposition == "hard_rejected":
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "hard_rejected", (lower, upper), (lower, upper), None, None)
    if disposition == "unresolved_budget":
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "tuning_unresolved", (lower, upper), (lower, upper), None, None)
    if disposition == "candidate_nominated":
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "freeze_for_qualification", (lower, upper), (lower, upper), None, None)
    if disposition == "needs_higher_epsilon":
        direction = "higher_epsilon"
        lower = current if lower is None else max(lower, current)
    elif disposition == "needs_lower_epsilon":
        direction = "lower_epsilon"
        upper = current if upper is None else min(upper, current)
    else:
        raise ValueError("evidence disposition cannot drive epsilon repair")
    bracket_after = (lower, upper)
    if lower is not None and upper is not None and lower >= upper:
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "bracket_conflict", (lower, upper), (lower, upper), None, None)
    if index + 1 >= policy.maximum_attempts:
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "attempt_budget_exhausted", (lower, upper), bracket_after, direction, None)
    proposed = math.sqrt(lower * upper) if lower is not None and upper is not None else (
        current * policy.repair_factor if direction == "higher_epsilon" else current / policy.repair_factor
    )
    repeated = any(math.isclose(proposed, item, rel_tol=1.0e-12, abs_tol=0.0) for item in history)
    negligible = abs(proposed / current - 1.0) < policy.minimum_relative_step_change
    if repeated or negligible:
        return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "stalled_repeated_or_negligible", (lower, upper), bracket_after, None, None)
    return OperationalStatisticalEpsilonRepairDecision(index, current, disposition, "repair_epsilon", (lower, upper), bracket_after, direction, proposed)


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
        return _signature(
            "bayesfilter.same_epsilon_neighbor_guard_request.v1",
            self.identity_payload(),
        )

    def identity_payload(self) -> Mapping[str, Any]:
        """The unchanged v1 request identity payload."""

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

    def payload(self) -> Mapping[str, Any]:
        return {
            **self.identity_payload(),
            "scientific_role": NEIGHBOR_COVERAGE_ROLE,
            "coverage_derived": True,
            "parent_promotion_veto": False,
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
        return _signature(
            "bayesfilter.same_epsilon_neighbor_guard.v1",
            self.identity_payload(),
        )

    def identity_payload(self) -> Mapping[str, Any]:
        """The unchanged v1 guard identity payload."""

        return {
            "request": self.request.identity_payload(),
            "evidence": self.evidence.payload(),
            "viable": self.viable,
            "independently_tuned": False,
        }

    def payload(self) -> Mapping[str, Any]:
        return {
            **self.identity_payload(),
            "coverage_derived": True,
            "parent_promotion_veto": False,
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
    """Expand viable primaries once into exact-epsilon coverage probes.

    These probes fill holes in the primary ``L`` grid.  Their result is
    admitted independently into the next-round union when compatible; a
    failed probe does not veto its compatible parent primary.
    """

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

    @property
    def viable_coverage_candidates(self) -> tuple[SameEpsilonNeighborGuard, ...]:
        """Compatible one-hop probes; failures do not veto parent primaries."""

        return self.viable_guard_candidates

    @property
    def coverage_barrier(self) -> OperationalBarrier:
        """Compatibility alias naming the barrier's active scientific role."""

        return self.guard_barrier

    @property
    def next_round_candidates(
        self,
    ) -> tuple[OperationalPrimaryCandidate | SameEpsilonNeighborGuard, ...]:
        """Return the complete unranked primary-plus-coverage union.

        A next-round set is exposed only after both execution barriers pass.
        The one-hop probes contribute themselves when compatible; their
        compatibility is never used to remove a compatible primary.
        """

        if (
            not self.mass_handoff.grid_ready
            or not self.primary_barrier.complete
            or not self.guard_barrier.complete
        ):
            return ()
        candidates = self.viable_primary_candidates + self.viable_coverage_candidates
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    item.request.num_leapfrog_steps,
                    0 if isinstance(item, OperationalPrimaryCandidate) else 1,
                    item.tuned_step_size.hex()
                    if isinstance(item, OperationalPrimaryCandidate)
                    else item.request.inherited_step_size.hex(),
                ),
            )
        )

    @property
    def next_round_l_values(self) -> tuple[int, ...]:
        """Sorted unique ``L`` values for the unranked next-round union."""

        return tuple(
            sorted(
                {
                    item.request.num_leapfrog_steps
                    for item in self.next_round_candidates
                }
            )
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.operational_broad_grid.private.v2",
            "route": ROUTE_ID,
            "policy": self.policy.payload(),
            "mass_handoff": self.mass_handoff.payload(),
            "primary_candidates": tuple(item.payload() for item in self.primary_candidates),
            "guard_candidates": tuple(item.payload() for item in self.guard_candidates),
            "coverage_candidates": tuple(
                item.payload() for item in self.guard_candidates
            ),
            "primary_barrier": self.primary_barrier.payload(),
            "guard_barrier": self.guard_barrier.payload(),
            "coverage_barrier": self.coverage_barrier.payload(),
            "disposition": self.disposition,
            "execution": self.execution.payload(),
            "viable_primary_count": len(self.viable_primary_candidates),
            "viable_guard_count": len(self.viable_guard_candidates),
            "viable_coverage_count": len(self.viable_coverage_candidates),
            "next_round_candidates": tuple(
                item.payload() for item in self.next_round_candidates
            ),
            "next_round_l_values": self.next_round_l_values,
            "next_round_candidate_count": len(self.next_round_candidates),
            "all_viable_pairs_preserved": True,
            "representative": None,
            "stochastic_ranking_performed": False,
            "all_tuning_draws_discarded": True,
            "retained_sampling_authorized": False,
            "nonclaims": NONCLAIMS,
        }

    def public_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.operational_broad_grid.public.v2",
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
            "viable_coverage_count": len(self.viable_coverage_candidates),
            "next_round_l_values": self.next_round_l_values,
            "next_round_candidate_count": len(self.next_round_candidates),
            "next_round_roles": tuple(
                {
                    "num_leapfrog_steps": item.request.num_leapfrog_steps,
                    "role": (
                        PRIMARY_ROLE
                        if isinstance(item, OperationalPrimaryCandidate)
                        else NEIGHBOR_COVERAGE_ROLE
                    ),
                    "parent_l_values": (
                        ()
                        if isinstance(item, OperationalPrimaryCandidate)
                        else item.request.parent_l_values
                    ),
                    "epsilon_policy": (
                        "independently_tuned"
                        if isinstance(item, OperationalPrimaryCandidate)
                        else "inherit_exact_primary_epsilon_no_retuning"
                    ),
                }
                for item in self.next_round_candidates
            ),
            "stochastic_ranking_performed": False,
            "retained_sampling_authorized": False,
            "raw_samples_exposed": False,
            "raw_states_exposed": False,
            "epsilon_values_exposed": False,
            "metric_arrays_exposed": False,
            "nonclaims": NONCLAIMS,
        }


@dataclass(frozen=True)
class OperationalCandidateUnionSelection:
    """Deterministic policy selection over a complete viable ``(L, epsilon)`` union.

    This boundary intentionally consumes candidate summaries rather than HMC
    tensors.  It preserves the broad-grid uncertainty contract: viability is
    decided by each candidate's typed evidence, while the representative is a
    policy tie-break and never a ranking by acceptance, ESJD, or runtime.
    """

    anchor_l: int
    candidate_records: tuple[Mapping[str, Any], ...]
    selected_index: int | None
    disposition: str
    selection_order: tuple[int, ...]
    stochastic_ranking_performed: bool = False

    def __post_init__(self) -> None:
        anchor = _strict_integer(self.anchor_l, name="anchor_l", minimum=1)
        records = tuple(dict(record) for record in self.candidate_records)
        if not records:
            raise ValueError("candidate union must not be empty")
        if self.selected_index is not None:
            index = _strict_integer(
                self.selected_index,
                name="selected_index",
                minimum=0,
            )
            if index >= len(records):
                raise ValueError("selected_index is outside candidate union")
            object.__setattr__(self, "selected_index", index)
        order = tuple(
            _strict_integer(item, name="selection_order item", minimum=0)
            for item in self.selection_order
        )
        if order != tuple(sorted(order, key=lambda item: self._sort_key(records[item]))):
            raise ValueError("selection_order does not match deterministic policy")
        if set(order) != set(range(len(records))):
            raise ValueError("selection_order must cover the complete candidate union")
        disposition = str(self.disposition)
        if disposition not in {"representative_selected", "no_viable_candidate"}:
            raise ValueError("invalid candidate-union selection disposition")
        if disposition == "representative_selected" and self.selected_index is None:
            raise ValueError("selected disposition requires a representative")
        if disposition == "no_viable_candidate" and self.selected_index is not None:
            raise ValueError("no-viable disposition cannot carry a representative")
        if bool(self.stochastic_ranking_performed):
            raise ValueError("candidate-union selection cannot use stochastic ranking")
        object.__setattr__(self, "anchor_l", anchor)
        object.__setattr__(self, "candidate_records", records)
        object.__setattr__(self, "selection_order", order)
        object.__setattr__(self, "stochastic_ranking_performed", False)

    @staticmethod
    def _sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            abs(int(record["num_leapfrog_steps"]) - int(record["anchor_l"])),
            int(record["num_leapfrog_steps"]),
            str(record.get("content_signature", "")),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.operational_candidate_union_selection.v1",
            "anchor_l": self.anchor_l,
            "candidate_records": self.candidate_records,
            "selected_index": self.selected_index,
            "disposition": self.disposition,
            "selection_order": self.selection_order,
            "selection_rule": "closest_to_anchor_then_lower_l_then_content_signature",
            "stochastic_ranking_performed": False,
            "nonclaims": NONCLAIMS,
        }


def select_operational_candidate_union(
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    anchor_l: int,
    expected_lineage: Mapping[str, Any],
) -> OperationalCandidateUnionSelection:
    """Validate and select a complete, viable candidate union by policy only.

    Each record must expose ``num_leapfrog_steps``, ``tuned_step_size``,
    ``viable``, ``anchor_l``, and the shared ``metric_signature``,
    ``coordinate_signature``, and ``lineage_signature``.  The function never
    inspects acceptance magnitudes, runtime, or efficiency metrics.
    """

    anchor = _strict_integer(anchor_l, name="anchor_l", minimum=1)
    expected = {str(key): str(value) for key, value in expected_lineage.items()}
    required = {
        "metric_signature",
        "coordinate_signature",
        "lineage_signature",
    }
    if set(expected) != required or any(not value for value in expected.values()):
        raise ValueError("expected_lineage must contain the three shared signatures")
    records = tuple(dict(record) for record in candidate_records)
    if not records:
        raise ValueError("candidate union must not be empty")
    seen_l: set[int] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        try:
            leapfrog = _strict_integer(
                record["num_leapfrog_steps"],
                name="candidate num_leapfrog_steps",
                minimum=2,
            )
            epsilon = _finite_step(record["tuned_step_size"])
            record_anchor = _strict_integer(record["anchor_l"], name="candidate anchor_l")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("candidate union record is malformed") from error
        if record_anchor != anchor:
            raise ValueError("candidate anchor does not match selection anchor")
        if leapfrog in seen_l:
            raise ValueError("candidate union contains duplicate L")
        seen_l.add(leapfrog)
        for name in required:
            if str(record.get(name, "")) != expected[name]:
                raise ValueError("candidate union lineage mismatch")
        if "viable" not in record or not isinstance(record["viable"], bool):
            raise ValueError("candidate viability must be boolean")
        record["num_leapfrog_steps"] = leapfrog
        record["tuned_step_size"] = epsilon
        record["anchor_l"] = anchor
        record["content_signature"] = str(record.get("content_signature", ""))
        normalized.append(record)
    order = tuple(
        sorted(
            range(len(normalized)),
            key=lambda index: OperationalCandidateUnionSelection._sort_key(
                normalized[index]
            ),
        )
    )
    selected = next(
        (index for index in order if normalized[index]["viable"]),
        None,
    )
    return OperationalCandidateUnionSelection(
        anchor_l=anchor,
        candidate_records=tuple(normalized),
        selected_index=selected,
        disposition=(
            "representative_selected" if selected is not None else "no_viable_candidate"
        ),
        selection_order=order,
    )


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
