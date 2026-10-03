"""Generic staged selection among fixed-transport HMC candidates.

The numerical HMC runner remains caller-supplied.  This module owns the
candidate-set protocol: candidate-local rungs, hard validity screens,
retention of every viable candidate, and caller-supplied nomination ordering.
Training receipts and domain-specific quality metrics remain outside this
module.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


FIXED_TRANSPORT_CANDIDATE_SELECTION_SCHEMA = (
    "bayesfilter.fixed_transport_candidate_selection.v1"
)
CandidateValidationFn = Callable[[Mapping[str, Any], int], Mapping[str, Any]]
CandidateScoreFn = Callable[
    [Mapping[str, Any], Mapping[str, Any]], Sequence[float]
]


@dataclass(frozen=True)
class FixedTransportCandidateSelectionConfig:
    """Experimental hard-screen settings, not universal convergence defaults."""

    rungs: int = 2
    rhat_max: float = 1.01
    min_bulk_ess: float = 400.0
    min_tail_ess: float = 200.0
    max_mcse_sd_ratio: float = 0.10
    require_all_chain_movement: bool = True
    require_target_status: bool = True
    require_finite: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.rungs, bool)
            or not _finite(self.rungs)
            or not float(self.rungs).is_integer()
            or float(self.rungs) <= 0.0
        ):
            raise ValueError("rungs must be a positive integer")
        object.__setattr__(self, "rungs", int(self.rungs))
        for name in ("rhat_max", "min_bulk_ess", "min_tail_ess", "max_mcse_sd_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": FIXED_TRANSPORT_CANDIDATE_SELECTION_SCHEMA,
            "rungs": self.rungs,
            "rhat_max": self.rhat_max,
            "min_bulk_ess": self.min_bulk_ess,
            "min_tail_ess": self.min_tail_ess,
            "max_mcse_sd_ratio": self.max_mcse_sd_ratio,
            "require_all_chain_movement": self.require_all_chain_movement,
            "require_target_status": self.require_target_status,
            "require_finite": self.require_finite,
        }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _hard_vetoes(
    observation: Mapping[str, Any],
    config: FixedTransportCandidateSelectionConfig,
) -> list[str]:
    vetoes = [str(item) for item in observation.get("hard_vetoes", ())]
    if config.require_finite and observation.get("all_finite") is not True:
        vetoes.append("nonfinite_candidate_observation")
    if config.require_target_status and observation.get("target_status_valid") is not True:
        vetoes.append("target_status_invalid")
    if config.require_all_chain_movement and observation.get("all_chain_movement") is not True:
        vetoes.append("chain_movement_failed")
    if observation.get("native_divergence_status") == "available" and observation.get("native_divergence_count") is None:
        vetoes.append("native_divergence_count_missing")
    if observation.get("native_divergence_count") is not None:
        if not _finite(observation["native_divergence_count"]):
            vetoes.append("native_divergence_count_nonfinite")
        elif (
            isinstance(observation["native_divergence_count"], bool)
            or float(observation["native_divergence_count"]) < 0.0
            or not float(observation["native_divergence_count"]).is_integer()
        ):
            vetoes.append("native_divergence_count_invalid")
        elif float(observation["native_divergence_count"]) > 0.0:
            vetoes.append("native_divergence_detected")
    for name, threshold, relation, reason in (
        ("max_rhat", config.rhat_max, lambda value, limit: value > limit, "rhat_screen_failed"),
        ("min_bulk_ess", config.min_bulk_ess, lambda value, limit: value < limit, "bulk_ess_screen_failed"),
        ("min_tail_ess", config.min_tail_ess, lambda value, limit: value < limit, "tail_ess_screen_failed"),
        (
            "max_mcse_sd_ratio",
            config.max_mcse_sd_ratio,
            lambda value, limit: value > limit,
            "mcse_sd_screen_failed",
        ),
    ):
        value = observation.get(name)
        if not _finite(value):
            vetoes.append(f"{name}_nonfinite_or_missing")
        elif float(value) < 0.0:
            vetoes.append(f"{name}_negative")
        elif relation(float(value), threshold):
            vetoes.append(reason)
    return list(dict.fromkeys(vetoes))


def select_fixed_transport_candidate_set(
    candidates: Sequence[Mapping[str, Any]],
    *,
    validate_rung: CandidateValidationFn,
    score_candidate: CandidateScoreFn | None = None,
    config: FixedTransportCandidateSelectionConfig | None = None,
    score_metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate, retain, and nominate a set of fixed-transport candidates.

    ``validate_rung`` executes one fresh fixed-kernel validation rung and
    returns scalar diagnostics in the observation contract.  It is called at
    most once per candidate and rung, and the next rung is skipped after a
    hard veto.  ``score_candidate`` supplies a lower-is-better lexicographic
    score; it may use caller-owned transport or efficiency receipts but must
    return finite scalar values. Without it the viable set is returned without
    a nominee. ``score_metadata`` preserves the caller's ordering, provenance
    and uncertainty description. This function makes no superiority claim and
    does not manage, discard or promote validation draws.
    """

    policy = config or FixedTransportCandidateSelectionConfig()
    identifiers = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError("candidates must contain mappings")
        identifiers.append(candidate.get("candidate_id", candidate.get("id", index)))
    if any(identifier in identifiers[:index] for index, identifier in enumerate(identifiers)):
        raise ValueError("candidate identifiers must be unique")
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_id = identifiers[index]
        rung_rows: list[dict[str, Any]] = []
        hard_vetoes: list[str] = []
        for rung in range(policy.rungs):
            observation = dict(validate_rung(candidate, rung))
            rung_vetoes = _hard_vetoes(observation, policy)
            rung_rows.append(
                {
                    "rung": rung + 1,
                    "status": "passed" if not rung_vetoes else "rejected",
                    "observation": observation,
                    "hard_vetoes": rung_vetoes,
                }
            )
            hard_vetoes.extend(rung_vetoes)
            if rung_vetoes:
                break
        passed = not hard_vetoes and len(rung_rows) == policy.rungs
        score = None
        if passed and score_candidate is not None:
            score_values = tuple(float(value) for value in score_candidate(candidate, {
                "candidate_id": candidate_id,
                "rungs": rung_rows,
            }))
            if not score_values or any(not math.isfinite(value) for value in score_values):
                raise ValueError("score_candidate must return finite nonempty values")
            score = score_values
        candidate_rows.append(
            {
                "candidate_index": index,
                "candidate_id": candidate_id,
                "passed": passed,
                "rungs": rung_rows,
                "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
                "score": score,
                "candidate": dict(candidate),
            }
        )

    viable = [row for row in candidate_rows if row["passed"]]
    ordered = sorted(
        viable, key=lambda row: (tuple(row["score"]), row["candidate_index"])
    ) if score_candidate is not None else []
    selected = ordered[0] if ordered else None
    return {
        "schema": FIXED_TRANSPORT_CANDIDATE_SELECTION_SCHEMA,
        "interface_kind": "diagnostic_helper",
        "replacement": "tune_hmc_kernel",
        "artifact_authority": False,
        "replay_authority": False,
        "numerical_handoff_authority": False,
        "config": policy.payload(),
        "candidate_rows": tuple(candidate_rows),
        "viable_candidates": tuple(viable),
        "continuation_candidates": tuple(viable),
        "selected_candidate": selected,
        "selected_candidate_index": None if selected is None else selected["candidate_index"],
        "nomination_status": (
            "descriptive_nomination" if selected is not None
            else "nomination_not_requested" if viable else "no_viable_candidate"
        ),
        "score_metadata": dict(score_metadata or {"provenance": "unspecified", "uncertainty_status": "not_provided"}),
        "statistical_ranking_supported": False,
        "validation_draws_policy": "caller_managed_validation_evidence_not_posterior_estimation",
        "nonclaims": (
            "no posterior convergence claim",
            "no statistical superiority claim",
            "no default-readiness claim",
        ),
    }


__all__ = [
    "FIXED_TRANSPORT_CANDIDATE_SELECTION_SCHEMA",
    "FixedTransportCandidateSelectionConfig",
    "select_fixed_transport_candidate_set",
]
