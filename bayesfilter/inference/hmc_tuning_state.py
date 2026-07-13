"""Typed state and repair aggregation for the private HMC tuning loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bayesfilter.inference.hmc_verification import HMCAcceptanceEvidence


HMC_TUNING_STATES = (
    "initialized",
    "warming",
    "metric_updated",
    "metric_frozen",
    "step_tuned",
    "trajectory_nominated",
    "verifying",
    "repair_required",
    "step_repaired",
    "inconclusive",
    "evidence_extended",
    "passed",
    "budget_exhausted_valid",
    "candidate_set_exhausted",
    "shared_invalidity",
    "architecture_blocked",
    "hard_timeout",
    "cancelled",
)


# v3 deliberately has no reason-only step repair. Epsilon may change only from
# valid acceptance evidence with a typed directional trigger.
RECOVERABLE_STEP_VETO_REASONS = frozenset()


SANITIZED_HEALTH_FAILURE_REASONS = frozenset(
    {
        *RECOVERABLE_STEP_VETO_REASONS,
        "nonfinite_candidate_state",
        "nonfinite_log_accept_ratio",
        "nonfinite_target_log_prob",
        "nonfinite_target_score",
        "target_status_telemetry_failure",
        "required_target_status_telemetry_missing",
        "target_value_score_shape_invalid",
        "shared_schema_invalid",
        "shared_adapter_invalid",
        "shared_callback_invalid",
        "nonfinite_adapted_step_size",
        "log_accept_energy_proxy_exceeded",
        "native_divergence_positive",
        "native_divergence_count_missing",
        "nonfinite_retained_samples",
        "nonfinite_final_state",
        "nonfinite_private_acceptance_log_value",
        "nonfinite_private_target_value",
        "required_standard_acceptance_trace_missing",
        "unrecognized_health_failure",
    }
)


def sanitize_health_failure_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    """Return only fixed public-safe reason codes, failing unknowns closed."""

    return tuple(
        sorted(
            {
                reason
                if reason in SANITIZED_HEALTH_FAILURE_REASONS
                else "unrecognized_health_failure"
                for reason in (str(item) for item in reasons)
            }
        )
    )


LEGAL_TUNING_TRANSITIONS = {
    "initialized": {"warming", "shared_invalidity", "architecture_blocked", "cancelled"},
    "warming": {"metric_updated", "metric_frozen", "shared_invalidity", "hard_timeout", "cancelled"},
    "metric_updated": {"warming", "shared_invalidity", "hard_timeout", "cancelled"},
    "metric_frozen": {"step_tuned", "shared_invalidity", "hard_timeout", "cancelled"},
    "step_tuned": {"trajectory_nominated", "verifying", "shared_invalidity", "hard_timeout", "cancelled"},
    "trajectory_nominated": {"step_tuned", "verifying", "candidate_set_exhausted", "inconclusive", "shared_invalidity", "hard_timeout", "cancelled"},
    "verifying": {"passed", "repair_required", "inconclusive", "candidate_set_exhausted", "shared_invalidity", "budget_exhausted_valid", "hard_timeout", "cancelled"},
    "repair_required": {"step_repaired", "warming", "trajectory_nominated", "initialized", "budget_exhausted_valid", "shared_invalidity", "cancelled"},
    "step_repaired": {"verifying", "budget_exhausted_valid", "shared_invalidity", "cancelled"},
    "inconclusive": {"evidence_extended", "budget_exhausted_valid", "shared_invalidity", "cancelled"},
    "evidence_extended": {"verifying", "shared_invalidity", "hard_timeout", "cancelled"},
}


TERMINAL_TUNING_STATES = frozenset(
    {
        "passed",
        "budget_exhausted_valid",
        "candidate_set_exhausted",
        "shared_invalidity",
        "architecture_blocked",
        "hard_timeout",
        "cancelled",
    }
)


@dataclass(frozen=True)
class HMCTuningTransition:
    source: str
    target: str
    reason: str
    coordinate_signature: str | None = None
    metric_signature: str | None = None
    trajectory_signature: str | None = None

    def __post_init__(self) -> None:
        source = str(self.source)
        target = str(self.target)
        if source not in HMC_TUNING_STATES or target not in HMC_TUNING_STATES:
            raise ValueError("unknown HMC tuning state")
        if target not in LEGAL_TUNING_TRANSITIONS.get(source, set()):
            raise ValueError(f"illegal HMC tuning transition: {source} -> {target}")
        if not str(self.reason):
            raise ValueError("transition reason must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "reason", str(self.reason))
        for name in (
            "coordinate_signature",
            "metric_signature",
            "trajectory_signature",
        ):
            value = getattr(self, name)
            object.__setattr__(self, name, None if value is None else str(value))

    def payload(self):
        return {
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "coordinate_signature": self.coordinate_signature,
            "metric_signature": self.metric_signature,
            "trajectory_signature": self.trajectory_signature,
        }


@dataclass(frozen=True)
class HMCStepRepair:
    disposition: str
    direction: str | None
    base_step_size: float
    repaired_step_size: float | None
    factor: float | None
    bracket: tuple[float | None, float | None]
    source_decisions: tuple[str, ...]
    verification_reserved: bool
    source_health_failures: tuple[tuple[str, ...], ...] = ()
    lower_bound_source_attempt_index: int | None = None

    def __post_init__(self) -> None:
        allowed = {
            "repair_step",
            "inconclusive_conflict",
            "inconclusive_evidence",
            "inconclusive_resonance",
            "inconclusive_trajectory",
            "inconclusive_stalled_or_oscillating",
            "candidate_data_invalid",
            "shared_invalidity",
        }
        disposition = str(self.disposition)
        if disposition not in allowed:
            raise ValueError("invalid step-repair disposition")
        base = float(self.base_step_size)
        if not np.isfinite(base) or base <= 0.0:
            raise ValueError("base_step_size must be positive and finite")
        lower, upper = self.bracket
        lower = None if lower is None else float(lower)
        upper = None if upper is None else float(upper)
        if lower is not None and (not np.isfinite(lower) or lower <= 0.0):
            raise ValueError("step bracket lower bound is invalid")
        if upper is not None and (not np.isfinite(upper) or upper <= 0.0):
            raise ValueError("step bracket upper bound is invalid")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("step bracket is inverted")
        if disposition == "repair_step":
            if self.direction not in {"lower_epsilon", "higher_epsilon"}:
                raise ValueError("repair_step requires a directional repair")
            repaired = float(self.repaired_step_size)
            factor = float(self.factor)
            if not np.isfinite(repaired) or repaired <= 0.0:
                raise ValueError("repaired_step_size must be positive and finite")
            expected_factor = (
                repaired / base
                if self.direction == "higher_epsilon"
                else base / repaired
            )
            if (
                not np.isfinite(factor)
                or factor <= 1.0
                or factor > 2.0 + 1.0e-12
                or not np.isclose(factor, expected_factor, rtol=1.0e-12, atol=0.0)
            ):
                raise ValueError("repair factor must equal the realized directional ratio")
            if lower is not None and repaired < lower:
                raise ValueError("repaired step is below its bracket")
            if upper is not None and repaired > upper:
                raise ValueError("repaired step is above its bracket")
            if not self.verification_reserved:
                raise ValueError("an applied repair requires reserved verification")
            object.__setattr__(self, "repaired_step_size", repaired)
            object.__setattr__(self, "factor", factor)
        elif self.repaired_step_size is not None or self.factor is not None:
            raise ValueError("non-repair disposition cannot carry a repaired step")
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "base_step_size", base)
        object.__setattr__(self, "bracket", (lower, upper))
        object.__setattr__(self, "source_decisions", tuple(self.source_decisions))
        health_failures = tuple(
            sanitize_health_failure_reasons(reasons)
            for reasons in self.source_health_failures
        )
        if any(not reasons or any(not reason for reason in reasons) for reasons in health_failures):
            raise ValueError("source health failure sets must be nonempty")
        if health_failures and len(health_failures) != len(self.source_decisions):
            raise ValueError("source health failures must align with source decisions")
        object.__setattr__(self, "source_health_failures", health_failures)
        lower_source = self.lower_bound_source_attempt_index
        if lower_source is not None:
            lower_source = int(lower_source)
            if (
                lower_source < 0
                or disposition != "repair_step"
                or self.direction != "lower_epsilon"
            ):
                raise ValueError("lower-bound source is invalid for this repair")
        object.__setattr__(self, "lower_bound_source_attempt_index", lower_source)

    def payload(self):
        return {
            "disposition": self.disposition,
            "direction": self.direction,
            "base_step_size": self.base_step_size,
            "repaired_step_size": self.repaired_step_size,
            "factor": self.factor,
            "bracket": self.bracket,
            "source_decisions": self.source_decisions,
            "source_health_failures": self.source_health_failures,
            "lower_bound_source_attempt_index": (
                self.lower_bound_source_attempt_index
            ),
            "verification_reserved": self.verification_reserved,
        }


def aggregate_step_repair(
    evidence: Sequence[HMCAcceptanceEvidence],
    *,
    base_step_size: float,
    repair_factor: float,
    bracket: tuple[float | None, float | None] = (None, None),
    direction_history: Sequence[str] = (),
    repaired_step_history: Sequence[float] = (),
    verification_reserved: bool,
) -> HMCStepRepair:
    """Aggregate a completed batch without depending on candidate order."""

    records = tuple(sorted(evidence, key=lambda item: repr(item.payload())))
    if not records:
        raise ValueError("repair aggregation requires evidence")
    decisions = tuple(item.acceptance_decision for item in records)
    disposition, direction = _step_repair_disposition(records)
    if direction is not None:
        history = tuple(str(item) for item in direction_history)
        alternations = sum(
            left != right for left, right in zip(history, history[1:])
        )
        repeated_direction = bool(history and history[-1] == direction)
        configured_factor = float(np.clip(repair_factor, 1.25, 2.0))
        multiplier = (
            1.0 / configured_factor
            if direction == "lower_epsilon"
            else configured_factor
        )
        repaired = float(base_step_size) * multiplier
        lower, upper = bracket
        if lower is not None:
            repaired = max(repaired, float(lower))
        if upper is not None:
            repaired = min(repaired, float(upper))
        history_steps = tuple(float(item) for item in repaired_step_history)
        negligible = abs(repaired / float(base_step_size) - 1.0) < 0.01
        identical = any(
            np.isclose(repaired, item, rtol=1.0e-12, atol=0.0)
            for item in history_steps
        )
        if alternations >= 2 or (repeated_direction and identical) or negligible:
            disposition = "inconclusive_stalled_or_oscillating"
        elif not verification_reserved:
            disposition = "inconclusive_evidence"
        else:
            return HMCStepRepair(
                disposition="repair_step",
                direction=direction,
                base_step_size=base_step_size,
                repaired_step_size=repaired,
                factor=_realized_directional_factor(
                    base_step_size,
                    repaired,
                    direction,
                ),
                bracket=bracket,
                source_decisions=decisions,
                verification_reserved=True,
            )
    return HMCStepRepair(
        disposition=disposition,
        direction=None,
        base_step_size=base_step_size,
        repaired_step_size=None,
        factor=None,
        bracket=bracket,
        source_decisions=decisions,
        verification_reserved=verification_reserved,
    )


def aggregate_bracketed_step_repair(
    evidence: Sequence[HMCAcceptanceEvidence],
    *,
    base_step_size: float,
    repair_factor: float,
    empirical_bracket: tuple[float | None, float | None] = (None, None),
    direction_history: Sequence[str] = (),
    repaired_step_history: Sequence[float] = (),
    verification_reserved: bool,
) -> HMCStepRepair:
    """Aggregate a batch and retain an empirical acceptance bracket.

    The lower endpoint is a step size observed in the high-acceptance regime;
    the upper endpoint is a step size observed in the low-acceptance regime.
    Once both exist, the next screen uses their log-scale midpoint. Alternating
    directions are expected while narrowing a valid bracket, so only repeated
    or negligible proposed steps are classified as stalled.
    """

    records = tuple(sorted(evidence, key=lambda item: repr(item.payload())))
    if not records:
        raise ValueError("repair aggregation requires evidence")
    base = float(base_step_size)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_step_size must be positive and finite")
    lower, upper = empirical_bracket
    lower = _positive_bound_or_none(lower, name="empirical bracket lower bound")
    upper = _positive_bound_or_none(upper, name="empirical bracket upper bound")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("empirical acceptance bracket is inverted")

    decisions = tuple(item.acceptance_decision for item in records)
    disposition, direction = _step_repair_disposition(records)
    if direction is None:
        return HMCStepRepair(
            disposition=disposition,
            direction=None,
            base_step_size=base,
            repaired_step_size=None,
            factor=None,
            bracket=(lower, upper),
            source_decisions=decisions,
            verification_reserved=verification_reserved,
        )

    expected_direction_decision = (
        "repair_step_higher"
        if direction == "higher_epsilon"
        else "repair_step_lower"
    )
    complete_direction_support = all(
        item.evidence_validity == "valid"
        and item.acceptance_decision == expected_direction_decision
        for item in records
    )
    if complete_direction_support:
        if direction == "higher_epsilon":
            lower = base if lower is None else max(lower, base)
        else:
            upper = base if upper is None else min(upper, base)
    if lower is not None and upper is not None and lower >= upper:
        return HMCStepRepair(
            disposition="inconclusive_conflict",
            direction=None,
            base_step_size=base,
            repaired_step_size=None,
            factor=None,
            bracket=empirical_bracket,
            source_decisions=decisions,
            verification_reserved=verification_reserved,
        )

    configured_factor = float(np.clip(repair_factor, 1.25, 2.0))
    if complete_direction_support and lower is not None and upper is not None:
        repaired = float(np.exp(0.5 * (np.log(lower) + np.log(upper))))
    elif direction == "higher_epsilon":
        repaired = base * configured_factor
    else:
        repaired = base / configured_factor
    if lower is not None:
        repaired = max(repaired, lower)
    if upper is not None:
        repaired = min(repaired, upper)

    history_steps = tuple(float(item) for item in repaired_step_history)
    if any(not np.isfinite(item) or item <= 0.0 for item in history_steps):
        raise ValueError("repaired_step_history must be positive and finite")
    history_directions = tuple(str(item) for item in direction_history)
    if any(
        item not in {"lower_epsilon", "higher_epsilon"}
        for item in history_directions
    ):
        raise ValueError("direction_history contains an invalid direction")
    if len(history_directions) != len(history_steps):
        raise ValueError("repair direction and step histories must have equal length")
    negligible = abs(repaired / base - 1.0) < 0.01
    repeated = any(
        np.isclose(repaired, item, rtol=1.0e-12, atol=0.0)
        for item in history_steps
    )
    if negligible or repeated:
        disposition = "inconclusive_stalled_or_oscillating"
    elif not verification_reserved:
        disposition = "inconclusive_evidence"
    else:
        return HMCStepRepair(
            disposition="repair_step",
            direction=direction,
            base_step_size=base,
            repaired_step_size=repaired,
            factor=_realized_directional_factor(base, repaired, direction),
            bracket=(lower, upper),
            source_decisions=decisions,
            verification_reserved=True,
        )
    return HMCStepRepair(
        disposition=disposition,
        direction=None,
        base_step_size=base,
        repaired_step_size=None,
        factor=None,
        bracket=(lower, upper),
        source_decisions=decisions,
        verification_reserved=verification_reserved,
    )


def aggregate_step_veto_bracket_repair(
    evidence: Sequence[HMCAcceptanceEvidence],
    *,
    base_step_size: float,
    repair_factor: float,
    empirical_bracket: tuple[float | None, float | None],
    repaired_step_history: Sequence[float] = (),
    verification_reserved: bool,
    lower_bound_source_attempt_index: int | None = None,
) -> HMCStepRepair:
    """Reject reason-only epsilon repair under the v3 evidence contract.

    This migration boundary remains callable, but invalid data, divergences,
    and alerts cannot establish a directional step bound.
    """

    records = tuple(sorted(evidence, key=lambda item: repr(item.payload())))
    if not records:
        raise ValueError("step-veto repair requires evidence")
    decisions = tuple(item.acceptance_decision for item in records)
    health_failures = tuple(
        tuple(sorted(item.engineering_invalidity_reasons)) for item in records
    )
    base = float(base_step_size)
    if not np.isfinite(base) or base <= 0.0:
        raise ValueError("base_step_size must be positive and finite")
    lower, upper = empirical_bracket
    lower = _positive_bound_or_none(lower, name="empirical bracket lower bound")
    upper = _positive_bound_or_none(upper, name="empirical bracket upper bound")
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("empirical acceptance bracket is inverted")

    if any(item.evidence_validity == "shared_execution_invalid" for item in records):
        disposition = "shared_invalidity"
    elif all(item.evidence_validity == "candidate_data_invalid" for item in records):
        disposition = "candidate_data_invalid"
    else:
        disposition = "inconclusive_evidence"
    return HMCStepRepair(
        disposition=disposition,
        direction=None,
        base_step_size=base,
        repaired_step_size=None,
        factor=None,
        bracket=(lower, upper),
        source_decisions=decisions,
        source_health_failures=health_failures,
        verification_reserved=verification_reserved,
    )


def _step_repair_disposition(
    records: Sequence[HMCAcceptanceEvidence],
) -> tuple[str, str | None]:
    """Return fail-closed batch scope and its unique supported direction."""

    if any(item.evidence_validity == "shared_execution_invalid" for item in records):
        return "shared_invalidity", None
    if all(item.evidence_validity == "candidate_data_invalid" for item in records):
        return "candidate_data_invalid", None
    usable = tuple(
        item for item in records if item.evidence_validity == "valid"
    )
    if any(
        item.acceptance_decision == "repair_trajectory"
        and "path_return_resonance_detected" in item.candidate_promotion_vetoes
        for item in usable
    ):
        return "inconclusive_resonance", None
    if any(item.acceptance_decision == "repair_trajectory" for item in usable):
        return "inconclusive_trajectory", None
    directions = {
        item.repair_direction
        for item in usable
        if item.repair_direction is not None
    }
    if len(directions) > 1:
        return "inconclusive_conflict", None
    if not directions:
        return "inconclusive_evidence", None
    return "repair_step", next(iter(directions))


def _positive_bound_or_none(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    bound = float(value)
    if not np.isfinite(bound) or bound <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return bound


def _realized_directional_factor(
    base_step_size: float,
    repaired_step_size: float,
    direction: str,
) -> float:
    base = float(base_step_size)
    repaired = float(repaired_step_size)
    return repaired / base if direction == "higher_epsilon" else base / repaired
