"""Summary-level uncertainty diagnostics for fixed-metric HMC retuning.

This module deliberately does not run HMC.  It consumes means from independently
seeded chain runs, computes descriptive spread, and can nominate a candidate for
a fresh retuning screen.  A nomination never changes BayesFilter's existing
``HMCAcceptanceEvidence.promotion_eligible`` decision and is not a convergence
interval or posterior uncertainty statement.
"""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from bayesfilter.inference.hmc_fixed_metric_grid_search import (
    REPLICATION_COUNT,
    FixedMetricCandidateRecord,
    FixedMetricEvidenceExtensionRecord,
    FixedMetricScreenRecord,
    FixedMetricScreenRequest,
    FixedMetricSearchLineage,
)


UNCERTAINTY_RETUNING_DISPOSITIONS = (
    "provisional_nomination",
    "outside_practical_region",
    "repair_region_violation",
    "hard_veto",
    "invalid_summary",
)


def _canonical(value: Any) -> Any:
    """Normalize JSON list/tuple differences for strict payload comparison."""

    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_canonical(item) for item in value)
    return value


def _signature(label: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"label": label, "payload": _canonical(payload)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_integer(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


@dataclass(frozen=True)
class HMCUncertaintyRetuningPolicy:
    """Explicit, opt-in nomination policy for noisy fixed-metric screens.

    ``interval_multiplier`` multiplies the sample standard deviation across
    independent chain-run means.  The interval is descriptive and is used only
    to nominate a fresh epsilon retuning screen.  It cannot authorize
    confirmation or retained sampling.
    """

    interval_multiplier: float = 1.0
    practical_region: tuple[float, float] = (0.65, 0.75)
    repair_region: tuple[float, float] = (0.55, 0.85)
    chain_count: int = 4
    replication_count: int = 3
    role: str = "retuning_nomination_only"

    def __post_init__(self) -> None:
        multiplier = float(self.interval_multiplier)
        if not math.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("interval_multiplier must be positive and finite")
        practical = tuple(float(item) for item in self.practical_region)
        repair = tuple(float(item) for item in self.repair_region)
        if len(practical) != 2 or not 0.0 < practical[0] < practical[1] < 1.0:
            raise ValueError("practical_region must be ordered inside (0, 1)")
        if len(repair) != 2 or not 0.0 < repair[0] < repair[1] < 1.0:
            raise ValueError("repair_region must be ordered inside (0, 1)")
        if not repair[0] <= practical[0] <= practical[1] <= repair[1]:
            raise ValueError("repair_region must contain practical_region")
        chain_count = _strict_integer(self.chain_count, name="chain_count")
        replication_count = _strict_integer(
            self.replication_count, name="replication_count"
        )
        if not isinstance(self.role, str) or not self.role:
            raise ValueError("role must be a nonempty string")
        object.__setattr__(self, "interval_multiplier", multiplier)
        object.__setattr__(self, "practical_region", practical)
        object.__setattr__(self, "repair_region", repair)
        object.__setattr__(self, "chain_count", chain_count)
        object.__setattr__(self, "replication_count", replication_count)

    @property
    def expected_chain_run_count(self) -> int:
        return self.chain_count * self.replication_count

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_uncertainty_retuning_policy.v1",
            "interval_multiplier": self.interval_multiplier,
            "practical_region": self.practical_region,
            "repair_region": self.repair_region,
            "chain_count": self.chain_count,
            "replication_count": self.replication_count,
            "expected_chain_run_count": self.expected_chain_run_count,
            "role": self.role,
            "interval_definition": "mean +/- interval_multiplier * sample_sd_of_chain_run_means",
            "sd_definition": "sample_standard_deviation_ddof_1_across_independent_seeded_chain_run_means",
            "limitations": (
                "chain_run_means_are_not_proven_iid",
                "within_chain_mcmc_autocorrelation_is_not_removed",
                "shared_start_state_can_induce_dependence",
                "nomination_is_not_convergence_or_posterior_uncertainty",
            ),
            "promotion_effect": "none",
        }


@dataclass(frozen=True)
class HMCUncertaintyRetuningSummary:
    """Descriptive candidate-level spread and nomination result."""

    disposition: str
    chain_run_count: int
    grand_mean: float | None
    sample_standard_deviation: float | None
    standard_error: float | None
    interval: tuple[float, float] | None
    chain_run_means: tuple[float, ...]
    hard_vetoes: tuple[str, ...]
    policy: HMCUncertaintyRetuningPolicy

    @property
    def nominated(self) -> bool:
        return self.disposition == "provisional_nomination"

    @property
    def signature(self) -> str:
        return _signature("bayesfilter.hmc_uncertainty_retuning_summary.v1", self.payload())

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_uncertainty_retuning_summary.v1",
            "disposition": self.disposition,
            "nominated": self.nominated,
            "chain_run_count": self.chain_run_count,
            "grand_mean": self.grand_mean,
            "sample_standard_deviation": self.sample_standard_deviation,
            "standard_error": self.standard_error,
            "interval": self.interval,
            "chain_run_means": self.chain_run_means,
            "hard_vetoes": self.hard_vetoes,
            "policy": self.policy.payload(),
            "nonclaims": (
                "no posterior convergence claim",
                "no confidence or credible interval claim",
                "no fixed-epsilon promotion",
                "no candidate ranking",
            ),
        }


@dataclass(frozen=True)
class HMCUncertaintyConfirmationAdmission:
    """Lineage-bound admission of one uncertainty nominee to confirmation.

    Admission does not change the original screen decisions. It only proves
    that a complete fixed candidate and its descriptive uncertainty summary
    refer to the same twelve chain-run means and contain no hard veto.
    """

    candidate: FixedMetricCandidateRecord
    nomination: HMCUncertaintyRetuningSummary
    lineage: FixedMetricSearchLineage
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, FixedMetricCandidateRecord):
            raise TypeError("candidate must be FixedMetricCandidateRecord")
        if not isinstance(self.nomination, HMCUncertaintyRetuningSummary):
            raise TypeError("nomination must be HMCUncertaintyRetuningSummary")
        if not isinstance(self.lineage, FixedMetricSearchLineage):
            raise TypeError("lineage must be FixedMetricSearchLineage")
        source_hash = str(self.source_artifact_sha256)
        if len(source_hash) != 64 or any(item not in "0123456789abcdef" for item in source_hash):
            raise ValueError("source_artifact_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "source_artifact_sha256", source_hash)

        candidate = self.candidate
        if (
            candidate.tuned_step_size is None
            or candidate.rejection_stage is not None
            or candidate.rejection_reasons
            or len(candidate.screens) != REPLICATION_COUNT
        ):
            raise ValueError("confirmation admission requires one complete fixed candidate")
        if not self.nomination.nominated or self.nomination.hard_vetoes:
            raise ValueError("confirmation admission requires a veto-free provisional nomination")

        means: list[float] = []
        for replication_index, screen in enumerate(candidate.screens):
            if screen.request.replication_index != replication_index:
                raise ValueError("candidate screen replication order is invalid")
            if screen.request.lineage != self.lineage:
                raise ValueError("candidate screen lineage mismatch")
            evidence = screen.evidence
            if evidence.evidence_validity != "valid":
                raise ValueError("candidate evidence is invalid")
            if evidence.candidate_promotion_vetoes or evidence.cost_stop_reasons:
                raise ValueError("candidate contains a confirmation veto")
            means.extend(float(item) for item in evidence.chain_means)
        if tuple(means) != self.nomination.chain_run_means:
            raise ValueError("nomination chain means do not match the source candidate")

    @property
    def signature(self) -> str:
        return _signature(
            "bayesfilter.hmc_uncertainty_confirmation_admission.v1", self.payload()
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.hmc_uncertainty_confirmation_admission.v1",
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_candidate_signature": self.candidate.signature,
            "nomination_signature": self.nomination.signature,
            "num_leapfrog_steps": self.candidate.num_leapfrog_steps,
            "fixed_step_size": self.candidate.tuned_step_size,
            "lineage": self.lineage.payload(),
            "lineage_signature": self.lineage.signature,
            "nomination": self.nomination.payload(),
            "original_candidate_promotion_unchanged": True,
            "confirmation_uses_fresh_evidence": True,
            "retained_sampling_authorized": False,
            "nonclaims": (
                "admission is not confirmation",
                "no candidate ranking",
                "no posterior convergence claim",
                "no retained-sampling readiness claim",
            ),
        }


def fixed_metric_search_lineage_from_payload(
    payload: Mapping[str, Any],
) -> FixedMetricSearchLineage:
    """Strictly reconstruct one fixed-metric lineage payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("lineage payload must be a mapping")
    lineage = FixedMetricSearchLineage(
        coordinate_signature=payload.get("coordinate_signature"),
        metric_signature=payload.get("metric_signature"),
        private_start_bank_content_signature=payload.get(
            "private_start_bank_content_signature"
        ),
        common_state_signature=payload.get("common_state_signature"),
    )
    if _canonical(lineage.payload()) != _canonical(payload):
        raise ValueError("fixed-metric lineage payload is inconsistent")
    return lineage


def fixed_metric_screen_record_from_payload(
    payload: Mapping[str, Any],
) -> FixedMetricScreenRecord:
    """Strictly reconstruct one fixed-metric screen record payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("screen payload must be a mapping")
    lineage = fixed_metric_search_lineage_from_payload(payload.get("lineage"))
    request = FixedMetricScreenRequest(
        round_index=_strict_integer(payload.get("round_index"), name="round_index", minimum=0),
        stage=str(payload.get("stage")),
        num_leapfrog_steps=_strict_integer(
            payload.get("num_leapfrog_steps"), name="num_leapfrog_steps"
        ),
        replication_index=_strict_integer(
            payload.get("replication_index"), name="replication_index", minimum=0
        ),
        seed=tuple(payload.get("seed", ())),
        tuned_step_size=float(payload.get("tuned_step_size")),
        num_results=_strict_integer(payload.get("num_results"), name="num_results"),
        lineage=lineage,
    )
    if request.stage not in {"screen", "evidence_extension", "confirmation"}:
        raise ValueError("fixed-metric screen stage is invalid")
    if len(request.seed) != 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 0
        for item in request.seed
    ):
        raise ValueError("fixed-metric screen seed is invalid")
    request = FixedMetricScreenRequest(
        **{**request.__dict__, "seed": tuple(int(item) for item in request.seed)}
    )
    if not math.isfinite(request.tuned_step_size) or request.tuned_step_size <= 0.0:
        raise ValueError("fixed-metric screen step size is invalid")
    record = FixedMetricScreenRecord(
        request=request,
        evidence_payload=payload.get("acceptance_evidence"),
    )
    if _canonical(record.payload()) != _canonical(payload):
        raise ValueError("fixed-metric screen payload is inconsistent")
    return record


def fixed_metric_candidate_record_from_payload(
    payload: Mapping[str, Any],
) -> FixedMetricCandidateRecord:
    """Strictly reconstruct one fixed-metric candidate payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("candidate payload must be a mapping")
    screens = tuple(
        fixed_metric_screen_record_from_payload(item)
        for item in payload.get("screens", ())
    )
    extensions: list[FixedMetricEvidenceExtensionRecord] = []
    for item in payload.get("evidence_extensions", ()):
        if not isinstance(item, Mapping):
            raise TypeError("evidence extension payload must be a mapping")
        replacement = fixed_metric_screen_record_from_payload(item.get("replacement"))
        extension = FixedMetricEvidenceExtensionRecord(
            replication_index=_strict_integer(
                item.get("replication_index"), name="replication_index", minimum=0
            ),
            prior_screen_signature=str(item.get("prior_screen_signature")),
            replacement=replacement,
        )
        if _canonical(extension.payload()) != _canonical(item):
            raise ValueError("fixed-metric evidence extension payload is inconsistent")
        extensions.append(extension)
    tuned = payload.get("tuned_step_size")
    candidate = FixedMetricCandidateRecord(
        round_index=_strict_integer(payload.get("round_index"), name="round_index", minimum=0),
        num_leapfrog_steps=_strict_integer(
            payload.get("num_leapfrog_steps"), name="num_leapfrog_steps"
        ),
        tune_seed=tuple(payload.get("tune_seed", ())),
        tuned_step_size=None if tuned is None else float(tuned),
        screens=screens,
        evidence_extensions=tuple(extensions),
        rejection_stage=payload.get("rejection_stage"),
        rejection_reasons=tuple(payload.get("rejection_reasons", ())),
        rejection_replication_index=payload.get("rejection_replication_index"),
    )
    if len(candidate.tune_seed) != 2 or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 0
        for item in candidate.tune_seed
    ):
        raise ValueError("fixed-metric tune seed is invalid")
    if _canonical(candidate.payload()) != _canonical(payload):
        raise ValueError("fixed-metric candidate payload is inconsistent")
    return candidate


def admit_hmc_uncertainty_nomination_for_confirmation(
    candidate_payload: Mapping[str, Any],
    nomination_payload: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
) -> HMCUncertaintyConfirmationAdmission:
    """Admit one exact uncertainty nominee to a separate fresh confirmation."""

    candidate = fixed_metric_candidate_record_from_payload(candidate_payload)
    nomination = hmc_uncertainty_retuning_summary_from_payload(nomination_payload)
    if not candidate.screens:
        raise ValueError("confirmation admission requires source screens")
    lineage = candidate.screens[0].request.lineage
    return HMCUncertaintyConfirmationAdmission(
        candidate=candidate,
        nomination=nomination,
        lineage=lineage,
        source_artifact_sha256=source_artifact_sha256,
    )


def hmc_uncertainty_retuning_policy_from_payload(
    payload: Mapping[str, Any],
) -> HMCUncertaintyRetuningPolicy:
    """Reconstruct and validate one uncertainty-retuning policy payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("policy payload must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_uncertainty_retuning_policy.v1":
        raise ValueError("uncertainty-retuning policy schema mismatch")
    policy = HMCUncertaintyRetuningPolicy(
        interval_multiplier=payload.get("interval_multiplier"),
        practical_region=tuple(payload.get("practical_region", ())),
        repair_region=tuple(payload.get("repair_region", ())),
        chain_count=payload.get("chain_count"),
        replication_count=payload.get("replication_count"),
        role=payload.get("role"),
    )
    if payload.get("expected_chain_run_count") != policy.expected_chain_run_count:
        raise ValueError("uncertainty-retuning chain count is inconsistent")
    return policy


def hmc_uncertainty_retuning_summary_from_payload(
    payload: Mapping[str, Any],
) -> HMCUncertaintyRetuningSummary:
    """Reconstruct and validate one uncertainty-retuning summary payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("summary payload must be a mapping")
    if payload.get("schema") != "bayesfilter.hmc_uncertainty_retuning_summary.v1":
        raise ValueError("uncertainty-retuning summary schema mismatch")
    policy = hmc_uncertainty_retuning_policy_from_payload(payload.get("policy"))
    summary = summarize_hmc_uncertainty_for_retuning(
        tuple(payload.get("chain_run_means", ())),
        policy=policy,
        hard_vetoes=tuple(payload.get("hard_vetoes", ())),
    )
    if _canonical(summary.payload()) != _canonical(payload):
        raise ValueError("uncertainty-retuning summary payload is inconsistent")
    return summary


def summarize_hmc_uncertainty_for_retuning(
    chain_run_means: Sequence[float],
    *,
    policy: HMCUncertaintyRetuningPolicy | None = None,
    hard_vetoes: Sequence[str] = (),
) -> HMCUncertaintyRetuningSummary:
    """Compute descriptive spread and an opt-in fresh-retuning nomination.

    A candidate is nominated when its mean +/- the configured SD multiplier
    intersects the practical acceptance region and every chain-run mean lies in
    the existing repair region.  Hard vetoes always win.  The function requires
    exactly ``replication_count * chain_count`` values so accidental pooling of
    raw serial draws cannot masquerade as independent evidence.
    """

    selected = HMCUncertaintyRetuningPolicy() if policy is None else policy
    if not isinstance(selected, HMCUncertaintyRetuningPolicy):
        raise TypeError("policy must be HMCUncertaintyRetuningPolicy")
    reasons = tuple(dict.fromkeys(str(item) for item in hard_vetoes))
    if any(not item for item in reasons):
        raise ValueError("hard_vetoes must contain nonempty strings")
    values = tuple(float(item) for item in chain_run_means)
    expected = selected.expected_chain_run_count
    if len(values) != expected or any(not math.isfinite(item) for item in values):
        return HMCUncertaintyRetuningSummary(
            disposition="invalid_summary",
            chain_run_count=len(values),
            grand_mean=None,
            sample_standard_deviation=None,
            standard_error=None,
            interval=None,
            chain_run_means=values,
            hard_vetoes=reasons,
            policy=selected,
        )
    if any(item < 0.0 or item > 1.0 for item in values):
        return HMCUncertaintyRetuningSummary(
            disposition="invalid_summary",
            chain_run_count=len(values),
            grand_mean=None,
            sample_standard_deviation=None,
            standard_error=None,
            interval=None,
            chain_run_means=values,
            hard_vetoes=reasons,
            policy=selected,
        )
    mean = float(np.mean(np.asarray(values, dtype=np.float64)))
    standard_deviation = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    standard_error = standard_deviation / math.sqrt(len(values))
    half_width = selected.interval_multiplier * standard_deviation
    interval = (
        max(0.0, mean - half_width),
        min(1.0, mean + half_width),
    )
    if reasons:
        disposition = "hard_veto"
    elif any(
        item < selected.repair_region[0] or item > selected.repair_region[1]
        for item in values
    ):
        disposition = "repair_region_violation"
    elif interval[1] < selected.practical_region[0] or interval[0] > selected.practical_region[1]:
        disposition = "outside_practical_region"
    else:
        disposition = "provisional_nomination"
    return HMCUncertaintyRetuningSummary(
        disposition=disposition,
        chain_run_count=len(values),
        grand_mean=mean,
        sample_standard_deviation=standard_deviation,
        standard_error=standard_error,
        interval=interval,
        chain_run_means=values,
        hard_vetoes=reasons,
        policy=selected,
    )


__all__ = [
    "HMCUncertaintyConfirmationAdmission",
    "HMCUncertaintyRetuningPolicy",
    "HMCUncertaintyRetuningSummary",
    "UNCERTAINTY_RETUNING_DISPOSITIONS",
    "admit_hmc_uncertainty_nomination_for_confirmation",
    "fixed_metric_candidate_record_from_payload",
    "fixed_metric_screen_record_from_payload",
    "fixed_metric_search_lineage_from_payload",
    "hmc_uncertainty_retuning_policy_from_payload",
    "hmc_uncertainty_retuning_summary_from_payload",
    "summarize_hmc_uncertainty_for_retuning",
]
