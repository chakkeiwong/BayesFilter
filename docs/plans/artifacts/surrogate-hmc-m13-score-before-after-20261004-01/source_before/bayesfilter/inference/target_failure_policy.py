"""Finite invalid-region target policy with fail-loud exception boundaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_FAILURE_LABELS = frozenset(
    {
        "prior_support",
        "stationarity",
        "covariance_positive_definite",
        "factorization_failure",
        "solve_residual",
        "nonfinite_value_gradient",
    }
)
TARGET_BRANCH_LABELS = frozenset(
    {
        "valid",
        "fallback_prior_support",
        "fallback_stationarity",
        "fallback_covariance_positive_definite",
        "fallback_factorization_failure",
        "fallback_solve_residual",
        "fallback_nonfinite_value_gradient",
    }
)
_TARGET_BOUNDARY_LABELS = frozenset(
    {
        "prior_support",
        "stationarity",
        "covariance_positive_definite",
    }
)
_BACKEND_BREAKDOWN_LABELS = frozenset({"factorization_failure", "solve_residual"})


class TargetRegionError(Exception):
    """Base class for declared target-region failures eligible for fallback."""

    failure_label = "prior_support"

    def __init__(self, message: str = "", *, details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = {} if details is None else dict(details)


class PriorSupportError(TargetRegionError):
    failure_label = "prior_support"


class StationarityError(TargetRegionError):
    failure_label = "stationarity"


class CovariancePositiveDefiniteError(TargetRegionError):
    failure_label = "covariance_positive_definite"


class FactorizationFailure(TargetRegionError):
    failure_label = "factorization_failure"


class SolveResidualError(TargetRegionError):
    failure_label = "solve_residual"


@dataclass(frozen=True)
class TargetFailurePolicy:
    """Opt-in finite fallback policy for declared target-region failures."""

    target_scope: str
    fallback_log_prob: float = -1.0e100
    fallback_gradient_value: float = 0.0
    catch_nonfinite_output: bool = True
    allowed_failure_labels: tuple[str, ...] = tuple(sorted(TARGET_FAILURE_LABELS))
    allowed_branch_labels: tuple[str, ...] = tuple(sorted(TARGET_BRANCH_LABELS))
    nonclaims: tuple[str, ...] = (
        "finite invalid-region fallback only",
        "no posterior convergence claim",
        "no sampler robustness claim",
        "no default target policy claim",
    )

    def __post_init__(self) -> None:
        target_scope = str(self.target_scope)
        if not target_scope:
            raise ValueError("target_scope must be non-empty")
        fallback = float(self.fallback_log_prob)
        gradient_value = float(self.fallback_gradient_value)
        if not np.isfinite(fallback):
            raise ValueError("fallback_log_prob must be finite")
        if not np.isfinite(gradient_value):
            raise ValueError("fallback_gradient_value must be finite")
        failure_labels = tuple(str(label) for label in self.allowed_failure_labels)
        branch_labels = tuple(str(label) for label in self.allowed_branch_labels)
        unknown_failures = sorted(set(failure_labels) - TARGET_FAILURE_LABELS)
        unknown_branches = sorted(set(branch_labels) - TARGET_BRANCH_LABELS)
        if unknown_failures:
            raise ValueError(f"unknown target failure labels: {unknown_failures}")
        if unknown_branches:
            raise ValueError(f"unknown target branch labels: {unknown_branches}")
        if not failure_labels:
            raise ValueError("allowed_failure_labels must be non-empty")
        if not branch_labels:
            raise ValueError("allowed_branch_labels must be non-empty")
        object.__setattr__(self, "target_scope", target_scope)
        object.__setattr__(self, "fallback_log_prob", fallback)
        object.__setattr__(self, "fallback_gradient_value", gradient_value)
        object.__setattr__(self, "catch_nonfinite_output", bool(self.catch_nonfinite_output))
        object.__setattr__(self, "allowed_failure_labels", failure_labels)
        object.__setattr__(self, "allowed_branch_labels", branch_labels)
        object.__setattr__(self, "nonclaims", tuple(str(item) for item in self.nonclaims))

    def branch_for_failure(self, failure_label: str) -> str:
        label = str(failure_label)
        if label not in self.allowed_failure_labels:
            raise ValueError(f"target failure label is not allowed: {label!r}")
        branch = f"fallback_{label}"
        if branch not in self.allowed_branch_labels:
            raise ValueError(f"target branch label is not allowed: {branch!r}")
        return branch

    def fallback_score(self, reference: Any) -> np.ndarray:
        array = np.asarray(reference, dtype=float)
        return np.full_like(array, self.fallback_gradient_value, dtype=float)


@dataclass(frozen=True)
class TargetPolicyEvaluation:
    """Structured value/score result from a target failure policy."""

    value: float
    score: np.ndarray
    fallback_used: bool
    branch_label: str
    failure_label: str | None
    classification: str
    target_scope: str
    diagnostics: Mapping[str, Any]
    nonclaims: tuple[str, ...]

    @property
    def value_finite(self) -> bool:
        return bool(np.isfinite(self.value))

    @property
    def score_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.score)))

    def payload(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "score_shape": tuple(int(dim) for dim in self.score.shape),
            "fallback_used": self.fallback_used,
            "branch_label": self.branch_label,
            "failure_label": self.failure_label,
            "classification": self.classification,
            "target_scope": self.target_scope,
            "value_finite": self.value_finite,
            "score_finite": self.score_finite,
            "diagnostics": dict(self.diagnostics),
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class TargetFailureClassification:
    """Coarse cause label for target/sampler failure evidence."""

    classification: str
    diagnostic_role: str
    reason: str
    target_scope: str | None = None
    nonclaims: tuple[str, ...] = (
        "classification is local diagnostic evidence only",
        "no posterior convergence claim",
        "no sampler promotion claim",
    )

    def payload(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "diagnostic_role": self.diagnostic_role,
            "reason": self.reason,
            "target_scope": self.target_scope,
            "nonclaims": self.nonclaims,
        }


def evaluate_target_with_failure_policy(
    value_score_fn: Callable[[Any], tuple[Any, Any]],
    position: Any,
    policy: TargetFailurePolicy,
) -> TargetPolicyEvaluation:
    """Evaluate a value/score target with declared finite fallback semantics.

    Only ``TargetRegionError`` subclasses are caught. Shape errors,
    programmer errors, TensorFlow shape/type errors, generic ``ValueError`` and
    generic ``RuntimeError`` remain loud by construction.
    """

    try:
        value, score = value_score_fn(position)
    except TargetRegionError as exc:
        return _fallback_evaluation(
            position,
            policy,
            failure_label=exc.failure_label,
            exception_type=type(exc).__name__,
            details=exc.details,
        )

    value_array = np.asarray(value, dtype=float)
    score_array = np.asarray(score, dtype=float)
    if value_array.shape != ():
        raise ValueError("target value must be scalar")
    if score_array.shape != np.asarray(position, dtype=float).shape:
        raise ValueError("target score shape must match position shape")
    if not np.all(np.isfinite(value_array)) or not np.all(np.isfinite(score_array)):
        if not policy.catch_nonfinite_output:
            raise FloatingPointError("target value/score is nonfinite")
        return _fallback_evaluation(
            position,
            policy,
            failure_label="nonfinite_value_gradient",
            exception_type=None,
            details={
                "value_finite": bool(np.all(np.isfinite(value_array))),
                "score_finite": bool(np.all(np.isfinite(score_array))),
            },
        )
    return TargetPolicyEvaluation(
        value=float(value_array),
        score=score_array,
        fallback_used=False,
        branch_label="valid",
        failure_label=None,
        classification="valid_target_region",
        target_scope=policy.target_scope,
        diagnostics={
            "fallback_reason": None,
            "exception_type": None,
            "value_finite": True,
            "score_finite": True,
        },
        nonclaims=policy.nonclaims,
    )


def classify_target_failure_mode(
    evaluation: TargetPolicyEvaluation | None = None,
    *,
    sampler_diagnostics: Mapping[str, Any] | None = None,
    target_scope: str | None = None,
) -> TargetFailureClassification:
    """Classify local target/sampler evidence without promoting ambiguity."""

    if evaluation is not None and evaluation.fallback_used:
        label = str(evaluation.failure_label)
        if label in _TARGET_BOUNDARY_LABELS:
            return TargetFailureClassification(
                classification="target_boundary",
                diagnostic_role="repair_trigger",
                reason=f"declared target-region fallback: {label}",
                target_scope=evaluation.target_scope,
            )
        if label in _BACKEND_BREAKDOWN_LABELS:
            return TargetFailureClassification(
                classification="backend_numerical_breakdown",
                diagnostic_role="repair_trigger",
                reason=f"declared backend numerical fallback: {label}",
                target_scope=evaluation.target_scope,
            )
        return TargetFailureClassification(
            classification="ambiguous_nonfinite_target_or_backend",
            diagnostic_role="repair_trigger",
            reason=f"nonfinite or ambiguous target evidence: {label}",
            target_scope=evaluation.target_scope,
        )
    if evaluation is not None and evaluation.classification == "valid_target_region":
        sampler = {} if sampler_diagnostics is None else dict(sampler_diagnostics)
        hard_sampler_failure = any(
            sampler.get(key) is False
            for key in (
                "required_arrays_finite",
                "zero_divergences",
                "log_accept_nonfinite_count_zero",
            )
        )
        if hard_sampler_failure:
            return TargetFailureClassification(
                classification="sampler_energy_error",
                diagnostic_role="repair_trigger",
                reason="target evaluation is finite but sampler hard-veto diagnostics failed",
                target_scope=evaluation.target_scope,
            )
        return TargetFailureClassification(
            classification="valid_target_region",
            diagnostic_role="explanatory",
            reason="target value and score are finite without fallback",
            target_scope=evaluation.target_scope,
        )
    return TargetFailureClassification(
        classification="ambiguous",
        diagnostic_role="repair_trigger",
        reason="insufficient target evaluation evidence",
        target_scope=target_scope,
    )


def _fallback_evaluation(
    position: Any,
    policy: TargetFailurePolicy,
    *,
    failure_label: str,
    exception_type: str | None,
    details: Mapping[str, Any] | None,
) -> TargetPolicyEvaluation:
    branch = policy.branch_for_failure(failure_label)
    score = policy.fallback_score(position)
    classification = (
        "target_region_fallback"
        if failure_label in _TARGET_BOUNDARY_LABELS
        else (
            "backend_numerical_fallback"
            if failure_label in _BACKEND_BREAKDOWN_LABELS
            else "ambiguous_nonfinite_fallback"
        )
    )
    return TargetPolicyEvaluation(
        value=policy.fallback_log_prob,
        score=score,
        fallback_used=True,
        branch_label=branch,
        failure_label=str(failure_label),
        classification=classification,
        target_scope=policy.target_scope,
        diagnostics={
            "fallback_reason": str(failure_label),
            "exception_type": exception_type,
            "details": {} if details is None else dict(details),
            "value_finite": True,
            "score_finite": bool(np.all(np.isfinite(score))),
        },
        nonclaims=policy.nonclaims,
    )
