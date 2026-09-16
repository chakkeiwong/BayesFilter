"""Shared, host-side interpretation of candidate evidence.

Acceptance compatibility and eligibility for promotion are separate facts.
This module imports no numerical backend and grants no execution authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


PASSING_DECISIONS = frozenset({"passed", "acceptance_in_band"})
DIRECTIONAL_DECISIONS = frozenset({"repair_step_higher", "repair_step_lower"})
DECISIONS = PASSING_DECISIONS | DIRECTIONAL_DECISIONS | frozenset({
    "inconclusive_evidence", "inconclusive_conflict", "repair_trajectory",
    "promotion_failed", "failed", "unavailable",
})
VALIDITIES = frozenset({"valid", "candidate_data_invalid", "shared_execution_invalid"})


@dataclass(frozen=True)
class HMCCandidateDecision:
    acceptance_decision: str
    evidence_validity: str = "valid"
    hard_vetoes: tuple[str, ...] = ()
    promotion_vetoes: tuple[str, ...] = ()
    repair_eligible: bool | None = None

    def __post_init__(self) -> None:
        if self.acceptance_decision not in DECISIONS:
            raise ValueError(f"unknown tuning decision: {self.acceptance_decision}")
        if self.evidence_validity not in VALIDITIES:
            raise ValueError(f"unknown evidence validity: {self.evidence_validity}")
        for name in ("hard_vetoes", "promotion_vetoes"):
            values = getattr(self, name)
            if isinstance(values, str):
                raise TypeError(f"{name} must be a sequence of reasons")
            object.__setattr__(self, name, tuple(str(value) for value in values))
        possible = (self.acceptance_decision in DIRECTIONAL_DECISIONS
                    and self.evidence_validity == "valid" and not self.hard_vetoes)
        if self.repair_eligible is not None and type(self.repair_eligible) is not bool:
            raise TypeError("repair_eligible must be boolean")
        if self.repair_eligible and not possible:
            raise ValueError("repair eligibility contradicts candidate evidence")
        object.__setattr__(self, "repair_eligible", possible if self.repair_eligible is None
                           else self.repair_eligible)

    @property
    def promotion_eligible(self) -> bool:
        return (self.acceptance_decision in PASSING_DECISIONS
                and self.evidence_validity == "valid"
                and not self.hard_vetoes and not self.promotion_vetoes)

    def payload(self) -> dict[str, Any]:
        return {"decision": self.acceptance_decision,
                "acceptance_decision": self.acceptance_decision,
                "evidence_validity": self.evidence_validity,
                "hard_vetoes": self.hard_vetoes,
                "promotion_vetoes": self.promotion_vetoes,
                "repair_eligible": self.repair_eligible,
                "promotion_eligible": self.promotion_eligible}

    @classmethod
    def from_observation(cls, value: Mapping[str, Any]) -> "HMCCandidateDecision":
        if not isinstance(value, Mapping):
            raise TypeError("candidate observation must be a mapping")
        decision = value.get("acceptance_decision", value.get("decision", "passed"))
        if "decision" in value and value["decision"] != decision:
            raise ValueError("acceptance decision aliases disagree")
        result = cls(acceptance_decision=decision,
            evidence_validity=value.get("evidence_validity", "valid"),
            hard_vetoes=value.get("hard_vetoes", ()),
            promotion_vetoes=value.get("promotion_vetoes", ()),
            repair_eligible=value.get("repair_eligible"))
        if "promotion_eligible" in value and value["promotion_eligible"] != result.promotion_eligible:
            raise ValueError("promotion eligibility contradicts candidate evidence")
        return result

    @classmethod
    def from_evidence(cls, evidence: Any, *, hard_vetoes=(), shared_invalidity=False):
        hard = tuple(dict.fromkeys((*hard_vetoes, *evidence.engineering_invalidity_reasons)))
        return cls(acceptance_decision=evidence.acceptance_decision,
            evidence_validity="shared_execution_invalid" if shared_invalidity else evidence.evidence_validity,
            hard_vetoes=hard, promotion_vetoes=evidence.candidate_promotion_vetoes)
