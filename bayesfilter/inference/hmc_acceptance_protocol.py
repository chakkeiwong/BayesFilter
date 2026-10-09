"""Framework-free contracts for experimental replicated acceptance evidence.

The independent unit is a complete frozen-start-bank trial, not a transition,
temporal block, or differently initialized chain. This module grants no tuning
artifact authority and does not change the v5/v6 default policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


PROTOCOL_SCHEMA = "bayesfilter.hmc_replicated_acceptance_policy.v7"
EVIDENCE_UNIT = "independent_fixed_horizon_trial"


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _probability(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between zero and one")
    return float(value)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class HMCReplicatedAcceptancePolicy:
    """Explicit finite-trial statistical contract, pending calibration.

    Counts, horizons and error budgets have no inferred scientific defaults.
    The wider qualification band retains the old broad compatibility intent;
    the preferred band guides step-size hypotheses. Neither is convergence.
    """

    trial_num_results: int
    discarded_prefix: int
    base_repetitions: int
    max_repetitions: int
    max_candidates: int
    search_family_alpha: float
    verification_family_alpha: float
    diagnostic_family_alpha: float
    temporal_tolerance: float
    preferred_region: tuple[float, float] = (.65, .75)
    qualification_region: tuple[float, float] = (.55, .85)
    start_weights: tuple[float, ...] = (.25, .25, .25, .25)
    method: str = "bounded_betting_mixture_v1"
    bet_fractions: tuple[float, ...] = (1., .5, .25, .125)
    min_movement_rate: float = .05
    max_repeated_state_fraction: float = .95
    min_normalized_return_displacement: float = 1.e-4
    max_abs_log_accept_energy_proxy: float = 1000.

    def __post_init__(self) -> None:
        _integer(self.trial_num_results, "trial_num_results", 64)
        _integer(self.discarded_prefix, "discarded_prefix", 0)
        for name in ("base_repetitions", "max_repetitions", "max_candidates"):
            _integer(getattr(self, name), name)
        if self.base_repetitions > self.max_repetitions:
            raise ValueError("base repetitions exceed their cap")
        for name in ("search_family_alpha", "verification_family_alpha",
                     "diagnostic_family_alpha", "temporal_tolerance"):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        for name in ("preferred_region", "qualification_region"):
            value = tuple(getattr(self, name))
            if len(value) != 2:
                raise ValueError(f"{name} requires two endpoints")
            value = tuple(_probability(v, name) for v in value)
            if value[0] >= value[1]:
                raise ValueError(f"{name} must be ordered")
            object.__setattr__(self, name, value)
        if not (self.qualification_region[0] <= self.preferred_region[0]
                < self.preferred_region[1] <= self.qualification_region[1]):
            raise ValueError("qualification region must contain the preferred region")
        weights = tuple(self.start_weights)
        if len(weights) != 4 or any(type(w) not in (int, float) or not math.isfinite(w)
                                    or w <= 0 for w in weights):
            raise ValueError("four positive start weights are required")
        # Do not silently renormalize or omit a start from the estimand.
        if math.fsum(weights) != 1.:
            raise ValueError("start weights must sum to one")
        object.__setattr__(self, "start_weights", tuple(float(w) for w in weights))
        if self.method not in {"bounded_betting_mixture_v1", "bounded_hoeffding_rungs_v1"}:
            raise ValueError("unsupported replicated uncertainty method")
        bets = tuple(self.bet_fractions)
        if not bets or len(set(bets)) != len(bets) or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 1 for v in bets
        ):
            raise ValueError("distinct fixed bets in (0,1] are required")
        object.__setattr__(self, "bet_fractions", tuple(float(v) for v in bets))
        for name in ("min_movement_rate", "max_repeated_state_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {name}")
        for name in ("min_normalized_return_displacement", "max_abs_log_accept_energy_proxy"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name}")

    @property
    def chain_count(self) -> int:
        return len(self.start_weights)

    @property
    def min_decisions_per_chain(self) -> int:
        return self.trial_num_results

    @property
    def practical_region(self) -> tuple[float, float]:
        return self.preferred_region

    @property
    def repair_region(self) -> tuple[float, float]:
        return self.qualification_region

    @property
    def target(self) -> float:
        return .5 * math.fsum(self.preferred_region)

    @property
    def identity(self) -> str:
        return hashlib.sha256(_canonical(self.payload()).encode()).hexdigest()

    def payload(self) -> Mapping[str, Any]:
        return {"schema": PROTOCOL_SCHEMA, **asdict(self),
                "evidence_unit": EVIDENCE_UNIT,
                "estimand": "conditional_frozen_start_bank_fixed_horizon_mean_metropolis_probability",
                "temporal_diagnostic_role": "preparation_investigation_not_admission",
                "rhat_role": "reporting_only",
                "independent_unit": "complete_start_bank_trial",
                "default_promotion_status": "experimental_pending_validation"}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "HMCReplicatedAcceptancePolicy":
        from dataclasses import fields
        if not isinstance(payload, Mapping) or payload.get("schema") != PROTOCOL_SCHEMA:
            raise ValueError("replicated acceptance policy schema mismatch")
        values = {f.name: payload[f.name] for f in fields(cls)}
        policy = cls(**values)
        if _canonical(policy.payload()) != _canonical(payload):
            raise ValueError("replicated acceptance policy metadata mismatch")
        return policy

    def repetition_target(self, multiplier: int) -> int:
        target = self.base_repetitions * _integer(multiplier, "evidence multiplier")
        if target > self.max_repetitions:
            raise ValueError("evidence rung exceeds the declared repetition cap")
        return target

    def family_alpha(self, stage: str) -> float:
        if stage == "verification":
            return self.verification_family_alpha
        if stage in {"measurement", "pilot"}:
            return self.search_family_alpha
        raise ValueError("unknown acceptance evidence stage")

    def one_sided_alpha(self, stage: str) -> float:
        # Pooled mean plus each start mean. No independence between them is used.
        # Pilot and measurement can each make a directional assertion. Their
        # fresh streams do not entitle them to spend the search budget twice.
        stages = 1 if stage == "verification" else 2
        try:
            value = self.family_alpha(stage) / (2 * stages * self.max_candidates * (self.chain_count + 1))
        except OverflowError as exc:
            raise ValueError("error allocation is below floating-point resolution") from exc
        if not 0 < value < 1:
            raise ValueError("error allocation is below floating-point resolution")
        return value


def batch_information_preflight(*, draws: int, batch_size: int,
                                min_batches: int) -> Mapping[str, Any]:
    """Count information structurally, without importing a framework or sampling."""
    _integer(draws, "draws", 0)
    _integer(batch_size, "batch_size")
    _integer(min_batches, "min_batches", 2)
    counts = tuple((draws // 4) // b for b in (batch_size, 2 * batch_size))
    return {"feasible": min(counts) >= min_batches,
            "window_complete_batches": counts, "minimum_draws_fixed_batch": 8 * batch_size * min_batches,
            "statistical_sufficiency": "not_established_by_batch_counts"}


def replicated_evidence_preflight(policy: HMCReplicatedAcceptancePolicy, *,
                                  evidence_rungs: tuple[int, ...], candidate_cap: int,
                                  leapfrog_steps: tuple[int, ...],
                                  max_gradient_work: int | None = None,
                                  pilot_enabled: bool = False) -> Mapping[str, Any]:
    """Check the full evidence design before target evaluation or compilation."""
    if not isinstance(policy, HMCReplicatedAcceptancePolicy):
        raise TypeError("a replicated acceptance policy is required")
    rungs = tuple(evidence_rungs)
    if not rungs or rungs[0] != 1 or any(type(r) is not int or r < 1 for r in rungs):
        raise ValueError("evidence rungs must be integer multipliers starting at one")
    if tuple(sorted(set(rungs))) != rungs:
        raise ValueError("evidence rungs must be strictly increasing")
    _integer(candidate_cap, "candidate cap")
    if candidate_cap > policy.max_candidates:
        raise ValueError("search candidate cap exceeds the statistical error allocation")
    steps = tuple(leapfrog_steps)
    if not steps:
        raise ValueError("at least one leapfrog setting is required")
    if len(steps) > candidate_cap:
        raise ValueError("initial cohort exceeds the declared candidate cap")
    for l in steps:
        _integer(l, "leapfrog steps")
    if max_gradient_work is not None:
        _integer(max_gradient_work, "max_gradient_work")
    if type(pilot_enabled) is not bool:
        raise ValueError("pilot_enabled must be boolean")
    targets = tuple(policy.repetition_target(r) for r in rungs)
    per_trial = policy.chain_count * (policy.discarded_prefix + policy.trial_num_results)
    first_stage = tuple(per_trial * targets[0] * (l + 1) for l in steps)
    mandatory = (2 + int(pilot_enabled)) * sum(first_stage)
    return {"schema": "bayesfilter.hmc_replicated_evidence_preflight.v1",
            "policy_id": policy.identity, "evidence_unit": EVIDENCE_UNIT,
            "replication_targets": targets,
            "incremental_repetitions": (targets[0], *(b-a for a, b in zip(targets, targets[1:]))),
            "trial_transitions": per_trial, "initial_cohort_mandatory_gradient_work": mandatory,
            "initial_measurement_gradient_work_by_candidate": first_stage,
            "maximum_stage_gradient_work_by_candidate": tuple(per_trial * targets[-1] * (l+1) for l in steps),
            "candidate_cap": candidate_cap, "planned_looks": len(rungs),
            "pilot_enabled": pilot_enabled,
            "search_one_sided_alpha": policy.one_sided_alpha("measurement"),
            "verification_one_sided_alpha": policy.one_sided_alpha("verification"),
            "first_cohort_fundable": max_gradient_work is None or mandatory <= max_gradient_work,
            "precision_sufficiency": "not_established; requires statistical and cost validation",
            "cost_basis": "conservative transitions times L+1; includes every discarded prefix"}
