"""Reviewed HMC tuning-policy contracts with diagnostic-only adaptation helpers."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bayesfilter.inference.hmc_diagnostics import (
    HMCFailureClassification,
    HMCScreenResult,
    classify_hmc_screen,
)
from bayesfilter.inference.hmc_artifact_identity import mass_artifact_signature


HMC_TUNING_POLICY_LABELS = (
    "fixed_kernel_screen",
    "dual_averaging_step_size",
    "fixed_mass_dual_averaging",
    "windowed_mass_adaptation",
    "fixed_trajectory_tuning",
    "windowed_mass_adaptation_future",
    "manual_ladder_diagnostic",
)

_DUAL_AVERAGING_LABELS = frozenset(
    {"dual_averaging_step_size", "fixed_mass_dual_averaging"}
)

_WINDOWED_MASS_LABELS = frozenset({"windowed_mass_adaptation"})

_FIXED_TRAJECTORY_LABELS = frozenset({"fixed_trajectory_tuning"})


@dataclass(frozen=True)
class HMCTuningPolicy:
    """Bounded HMC tuning-policy metadata.

    A policy is executable only when it is explicitly implemented and reviewed.
    Raw strings remain fail-closed for adaptive behavior; callers must pass a
    policy object to opt into any adaptation. The labels are deliberately more
    specific than TFP kernel names so result artifacts can distinguish fixed
    diagnostic screens, fixed-mass step-size adaptation, and future mass
    adaptation that is named but not executable in this phase.
    """

    label: str
    adaptation_policy: str
    num_adaptation_steps: int = 0
    target_accept_prob: float | None = None
    source: str = "reviewed_stage_7_phase_3"
    enabled: bool = False
    implemented: bool = True
    diagnostic_role: str = "bounded HMC tuning diagnostic"
    nonclaims: tuple[str, ...] = (
        "tuning policy metadata only",
        "no posterior convergence claim",
        "no sampler superiority claim",
        "no default adaptation readiness claim",
    )

    def __post_init__(self) -> None:
        label = str(self.label)
        if label not in HMC_TUNING_POLICY_LABELS:
            raise ValueError(
                "HMC tuning policy label must be one of "
                + ", ".join(HMC_TUNING_POLICY_LABELS)
            )
        adaptation_policy = str(self.adaptation_policy)
        if not adaptation_policy:
            raise ValueError("adaptation_policy must be non-empty")
        steps = int(self.num_adaptation_steps)
        if steps < 0:
            raise ValueError("num_adaptation_steps must be non-negative")
        target_accept = (
            None
            if self.target_accept_prob is None
            else float(self.target_accept_prob)
        )
        if target_accept is not None:
            if not np.isfinite(target_accept) or not (0.0 < target_accept < 1.0):
                raise ValueError("target_accept_prob must be finite and in (0, 1)")
        source = str(self.source)
        if not source:
            raise ValueError("source must be non-empty")
        role = str(self.diagnostic_role)
        if not role:
            raise ValueError("diagnostic_role must be non-empty")
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "adaptation_policy", adaptation_policy)
        object.__setattr__(self, "num_adaptation_steps", steps)
        object.__setattr__(self, "target_accept_prob", target_accept)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "implemented", bool(self.implemented))
        object.__setattr__(self, "diagnostic_role", role)
        object.__setattr__(self, "nonclaims", nonclaims)

    @classmethod
    def fixed_kernel_screen(cls) -> "HMCTuningPolicy":
        """Current fail-closed BayesFilter behavior: no adaptive kernel."""

        return cls(
            label="fixed_kernel_screen",
            adaptation_policy="fixed_kernel_no_adaptation",
            num_adaptation_steps=0,
            target_accept_prob=None,
            enabled=False,
            implemented=True,
            diagnostic_role="fixed-kernel bounded HMC screen",
            nonclaims=(
                "fixed-kernel screen only",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def dual_averaging_step_size(
        cls,
        *,
        num_adaptation_steps: int,
        target_accept_prob: float,
        source: str,
    ) -> "HMCTuningPolicy":
        """Reviewed diagnostic-only step-size adaptation with fixed mass."""

        return cls(
            label="dual_averaging_step_size",
            adaptation_policy="dual_averaging_step_size",
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
            source=source,
            enabled=True,
            implemented=True,
            diagnostic_role="dual-averaging step-size diagnostic only",
            nonclaims=(
                "dual-averaging step-size diagnostic only",
                "fixed mass matrix; no mass adaptation claim",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def fixed_mass_dual_averaging(
        cls,
        *,
        num_adaptation_steps: int,
        target_accept_prob: float,
        source: str,
    ) -> "HMCTuningPolicy":
        """Alias-like reviewed policy for fixed-mass step-size adaptation."""

        return cls(
            label="fixed_mass_dual_averaging",
            adaptation_policy="dual_averaging_step_size",
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
            source=source,
            enabled=True,
            implemented=True,
            diagnostic_role="fixed-mass dual-averaging diagnostic only",
            nonclaims=(
                "fixed-mass dual-averaging diagnostic only",
                "mass matrix is supplied/fixed; no windowed mass adaptation claim",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def windowed_mass_adaptation_future(cls) -> "HMCTuningPolicy":
        """Named future policy that is intentionally not executable."""

        return cls(
            label="windowed_mass_adaptation_future",
            adaptation_policy="windowed_mass_adaptation_future",
            num_adaptation_steps=0,
            target_accept_prob=None,
            enabled=True,
            implemented=False,
            diagnostic_role="future policy placeholder; not executable",
            nonclaims=(
                "future mass-adaptation label only",
                "not implemented or executable in Stage 7 / accepted Phase 3",
                "no posterior convergence claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def windowed_mass_adaptation(
        cls,
        *,
        num_adaptation_steps: int,
        target_accept_prob: float,
        source: str,
    ) -> "HMCTuningPolicy":
        """Reviewed non-default policy for Phase 4 windowed mass diagnostics."""

        return cls(
            label="windowed_mass_adaptation",
            adaptation_policy="windowed_mass_adaptation",
            num_adaptation_steps=num_adaptation_steps,
            target_accept_prob=target_accept_prob,
            source=source,
            enabled=True,
            implemented=True,
            diagnostic_role="windowed mass adaptation diagnostic only",
            nonclaims=(
                "windowed mass adaptation diagnostic only",
                "non-default experimental policy",
                "not exact Stan equivalence",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def fixed_trajectory_tuning(
        cls,
        *,
        target_accept_prob: float,
        source: str,
    ) -> "HMCTuningPolicy":
        """Reviewed non-default policy for Phase 5 fixed-trajectory diagnostics."""

        return cls(
            label="fixed_trajectory_tuning",
            adaptation_policy="fixed_trajectory_no_adaptation",
            num_adaptation_steps=0,
            target_accept_prob=target_accept_prob,
            source=source,
            enabled=True,
            implemented=True,
            diagnostic_role="fixed-trajectory tuning diagnostic only",
            nonclaims=(
                "fixed-trajectory tuning diagnostic only",
                "consumes frozen Phase 4 mass and step artifacts",
                "no posterior convergence claim",
                "no sampler superiority claim",
                "no default adaptation readiness claim",
            ),
        )

    @classmethod
    def manual_ladder_diagnostic(cls) -> "HMCTuningPolicy":
        """Metadata for externally reviewed fixed-kernel ladders."""

        return cls(
            label="manual_ladder_diagnostic",
            adaptation_policy="fixed_kernel_manual_ladder",
            num_adaptation_steps=0,
            target_accept_prob=None,
            enabled=False,
            implemented=False,
            diagnostic_role="manual ladder metadata only; not an adaptive kernel",
            nonclaims=(
                "manual fixed-kernel ladder metadata only",
                "not executable by the BayesFilter HMC runner in this phase",
                "no posterior convergence claim",
                "no sampler superiority claim",
            ),
        )

    @property
    def uses_dual_averaging(self) -> bool:
        return self.label in _DUAL_AVERAGING_LABELS

    @property
    def uses_windowed_mass_adaptation(self) -> bool:
        return self.label in _WINDOWED_MASS_LABELS

    @property
    def uses_fixed_trajectory_tuning(self) -> bool:
        return self.label in _FIXED_TRAJECTORY_LABELS

    def payload(self) -> Mapping[str, Any]:
        return {
            "label": self.label,
            "adaptation_policy": self.adaptation_policy,
            "num_adaptation_steps": self.num_adaptation_steps,
            "target_accept_prob": self.target_accept_prob,
            "source": self.source,
            "enabled": self.enabled,
            "implemented": self.implemented,
            "diagnostic_role": self.diagnostic_role,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class HMCTuningDiagnosticResult:
    """Tiny adaptation diagnostic result; not posterior evidence."""

    policy: HMCTuningPolicy
    diagnostics: Mapping[str, Any]
    trace: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy": self.policy.payload(),
            "diagnostics": self.diagnostics,
            "trace": self.trace,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class InitialStepBracketAttempt:
    """One finite-step bracketing attempt for fixed-mass HMC tuning."""

    step_size: float
    finite: bool
    reason: str

    def __post_init__(self) -> None:
        step_size = float(self.step_size)
        if not np.isfinite(step_size) or step_size <= 0.0:
            raise ValueError("bracket attempt step_size must be positive and finite")
        reason = str(self.reason)
        if not reason:
            raise ValueError("bracket attempt reason must be non-empty")
        object.__setattr__(self, "step_size", step_size)
        object.__setattr__(self, "finite", bool(self.finite))
        object.__setattr__(self, "reason", reason)

    def payload(self) -> Mapping[str, Any]:
        return {
            "step_size": self.step_size,
            "finite": self.finite,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class InitialStepBracketResult:
    """Recorded initial-step bracket, not posterior or convergence evidence."""

    selected_step_size: float | None
    attempts: tuple[InitialStepBracketAttempt, ...]
    passed: bool
    nonclaims: tuple[str, ...] = (
        "initial step bracket only",
        "fixed mass matrix; no mass adaptation claim",
        "no posterior convergence claim",
        "no sampler superiority claim",
    )

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        if not attempts:
            raise ValueError("initial step bracketing requires at least one attempt")
        selected = (
            None
            if self.selected_step_size is None
            else float(self.selected_step_size)
        )
        if selected is not None and (not np.isfinite(selected) or selected <= 0.0):
            raise ValueError("selected_step_size must be positive and finite")
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "selected_step_size", selected)
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "nonclaims", nonclaims)

    def payload(self) -> Mapping[str, Any]:
        return {
            "selected_step_size": self.selected_step_size,
            "attempts": tuple(attempt.payload() for attempt in self.attempts),
            "passed": self.passed,
            "nonclaims": self.nonclaims,
        }


@dataclass(frozen=True)
class FixedMassStepTuningResult:
    """Fixed-mass step-size tuning evidence with frozen-artifact invariants."""

    policy: HMCTuningPolicy
    mass_artifact_payload: Mapping[str, Any]
    mass_artifact_signature: str
    initial_step_bracket: InitialStepBracketResult
    diagnostic: HMCTuningDiagnosticResult
    frozen_mass_invariant: Mapping[str, Any]
    target_failure_classification: Mapping[str, Any] | None = None
    passed: bool = True
    nonclaims: tuple[str, ...] = (
        "fixed-mass step tuning only",
        "mass artifact is fingerprinted but not adapted",
        "no posterior convergence claim",
        "no sampler superiority claim",
        "no default adaptation readiness claim",
    )

    def __post_init__(self) -> None:
        signature = str(self.mass_artifact_signature)
        if not signature:
            raise ValueError("mass_artifact_signature must be non-empty")
        invariant = dict(self.frozen_mass_invariant)
        passed = bool(self.passed)
        if passed and not bool(invariant.get("passed")):
            raise ValueError("frozen mass invariant must pass")
        target_failure = _validate_target_failure_classification(
            self.target_failure_classification
        )
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "mass_artifact_payload", dict(self.mass_artifact_payload))
        object.__setattr__(self, "mass_artifact_signature", signature)
        object.__setattr__(self, "frozen_mass_invariant", invariant)
        object.__setattr__(self, "target_failure_classification", target_failure)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "nonclaims", nonclaims)

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy": self.policy.payload(),
            "mass_artifact_payload": self.mass_artifact_payload,
            "mass_artifact_signature": self.mass_artifact_signature,
            "initial_step_bracket": self.initial_step_bracket.payload(),
            "diagnostics": self.diagnostic.diagnostics,
            "trace": self.diagnostic.trace,
            "metadata": {
                "passed": self.passed,
                **dict(self.diagnostic.metadata),
                "frozen_mass_invariant": self.frozen_mass_invariant,
                "target_failure_classification": self.target_failure_classification,
                "minimum_phase4_telemetry_present": self.minimum_phase4_telemetry_present(),
                "nonclaims": self.nonclaims,
            },
        }

    def minimum_phase4_telemetry_present(self) -> Mapping[str, bool]:
        diagnostics = self.diagnostic.diagnostics
        trace = self.diagnostic.trace
        return {
            "final_finite_step_size": bool(diagnostics.get("final_step_size_finite")),
            "step_size_trace_or_bracket": bool(trace.get("step_size"))
            or bool(self.initial_step_bracket.attempts),
            "acceptance_log_accept_tuning_only": (
                "acceptance_rate" in diagnostics
                and "log_accept_ratio" in trace
                and bool(diagnostics.get("reports_posterior_convergence") is False)
            ),
            "adaptation_step_count": "num_adaptation_steps" in diagnostics,
            "target_accept_probability": "target_accept_prob" in diagnostics,
            "frozen_mass_artifact_signature": bool(self.mass_artifact_signature),
            "frozen_mass_invariant": bool(self.frozen_mass_invariant.get("passed")),
            "target_invalidity_classification": self.target_failure_classification
            is not None
            and "classification" in self.target_failure_classification
            and "diagnostic_role" in self.target_failure_classification,
            "explicit_nonclaims": bool(self.nonclaims),
        }


@dataclass(frozen=True)
class WindowedMassAdaptationConfig:
    """Non-default warmup-window semantics for mass adaptation diagnostics."""

    warmup_steps: int
    initial_buffer: int
    final_buffer: int
    first_window_size: int
    min_window_samples: int = 2
    mass_shrinkage: float = 0.1
    covariance_jitter: float = 1.0e-9
    eigenvalue_floor: float | None = None
    max_condition_number: float | None = None
    step_size_floor: float = 1.0e-6
    step_size_ceiling: float = 10.0
    step_adaptation_rate: float = 0.05
    mass_policy: str = "windowed_adaptive"

    def __post_init__(self) -> None:
        for name in (
            "warmup_steps",
            "initial_buffer",
            "final_buffer",
            "first_window_size",
            "min_window_samples",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")
        if self.first_window_size <= 0:
            raise ValueError("first_window_size must be positive")
        if self.min_window_samples <= 0:
            raise ValueError("min_window_samples must be positive")
        if self.initial_buffer + self.final_buffer >= self.warmup_steps:
            raise ValueError("buffers must leave at least one slow-window sample")
        slow_steps = self.warmup_steps - self.initial_buffer - self.final_buffer
        if slow_steps < self.min_window_samples:
            raise ValueError("slow-window span must contain at least min_window_samples")
        if self.first_window_size < self.min_window_samples:
            raise ValueError("first_window_size must be at least min_window_samples")
        shrinkage = float(self.mass_shrinkage)
        if not np.isfinite(shrinkage) or shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError("mass_shrinkage must be finite and in [0, 1]")
        jitter = float(self.covariance_jitter)
        if not np.isfinite(jitter) or jitter < 0.0:
            raise ValueError("covariance_jitter must be finite and non-negative")
        floor = (
            None
            if self.eigenvalue_floor is None
            else float(self.eigenvalue_floor)
        )
        if floor is not None and (not np.isfinite(floor) or floor < 0.0):
            raise ValueError("eigenvalue_floor must be finite and non-negative")
        max_condition = (
            None
            if self.max_condition_number is None
            else float(self.max_condition_number)
        )
        if max_condition is not None and (
            not np.isfinite(max_condition) or max_condition <= 1.0
        ):
            raise ValueError("max_condition_number must be finite and greater than 1")
        step_floor = float(self.step_size_floor)
        step_ceiling = float(self.step_size_ceiling)
        if not np.isfinite(step_floor) or step_floor <= 0.0:
            raise ValueError("step_size_floor must be positive and finite")
        if not np.isfinite(step_ceiling) or step_ceiling <= step_floor:
            raise ValueError("step_size_ceiling must be finite and greater than floor")
        step_rate = float(self.step_adaptation_rate)
        if not np.isfinite(step_rate) or step_rate < 0.0:
            raise ValueError("step_adaptation_rate must be finite and non-negative")
        mass_policy = str(self.mass_policy)
        if mass_policy not in {"windowed_adaptive", "fixed_identity"}:
            raise ValueError(
                "mass_policy must be 'windowed_adaptive' or 'fixed_identity'"
            )
        object.__setattr__(self, "mass_shrinkage", shrinkage)
        object.__setattr__(self, "covariance_jitter", jitter)
        object.__setattr__(self, "eigenvalue_floor", floor)
        object.__setattr__(self, "max_condition_number", max_condition)
        object.__setattr__(self, "step_size_floor", step_floor)
        object.__setattr__(self, "step_size_ceiling", step_ceiling)
        object.__setattr__(self, "step_adaptation_rate", step_rate)
        object.__setattr__(self, "mass_policy", mass_policy)

    def payload(self) -> Mapping[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "initial_buffer": self.initial_buffer,
            "final_buffer": self.final_buffer,
            "first_window_size": self.first_window_size,
            "min_window_samples": self.min_window_samples,
            "mass_shrinkage": self.mass_shrinkage,
            "covariance_jitter": self.covariance_jitter,
            "eigenvalue_floor": self.eigenvalue_floor,
            "max_condition_number": self.max_condition_number,
            "step_size_floor": self.step_size_floor,
            "step_size_ceiling": self.step_size_ceiling,
            "step_adaptation_rate": self.step_adaptation_rate,
            "mass_policy": self.mass_policy,
        }


@dataclass(frozen=True)
class WindowedWarmupWindow:
    """One contiguous fast/slow/final warmup window."""

    index: int
    kind: str
    start: int
    end: int
    update_mass: bool

    def __post_init__(self) -> None:
        index = int(self.index)
        start = int(self.start)
        end = int(self.end)
        kind = str(self.kind)
        if index < 0:
            raise ValueError("window index must be non-negative")
        if kind not in {"initial_fast", "slow", "final_fast"}:
            raise ValueError("window kind must be initial_fast, slow, or final_fast")
        if start < 0 or end <= start:
            raise ValueError("window bounds must satisfy 0 <= start < end")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "update_mass", bool(self.update_mass))

    @property
    def length(self) -> int:
        return self.end - self.start

    def payload(self) -> Mapping[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "update_mass": self.update_mass,
        }


@dataclass(frozen=True)
class WelfordCovarianceResult:
    """Online covariance result for one mass-update window."""

    count: int
    mean: Any
    covariance: Any
    finite: bool

    def __post_init__(self) -> None:
        count = int(self.count)
        if count <= 1:
            raise ValueError("Welford covariance requires at least two samples")
        mean = np.asarray(self.mean, dtype=float).copy()
        covariance = np.asarray(self.covariance, dtype=float).copy()
        if mean.ndim != 1:
            raise ValueError("Welford mean must be one-dimensional")
        if covariance.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("Welford covariance shape must match mean dimension")
        finite = bool(self.finite and np.all(np.isfinite(mean)) and np.all(np.isfinite(covariance)))
        mean.setflags(write=False)
        covariance.setflags(write=False)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "finite", finite)

    def payload(self) -> Mapping[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "finite": self.finite,
        }


@dataclass(frozen=True)
class WindowedMassUpdate:
    """One empirical covariance, shrinkage, mass rebuild, and reset event."""

    window: WindowedWarmupWindow
    welford: WelfordCovarianceResult
    shrinkage: float
    mass_artifact_payload: Mapping[str, Any]
    mass_artifact_signature: str
    reset_event: Mapping[str, Any]

    def __post_init__(self) -> None:
        shrinkage = float(self.shrinkage)
        if not np.isfinite(shrinkage) or shrinkage < 0.0 or shrinkage > 1.0:
            raise ValueError("shrinkage must be finite and in [0, 1]")
        signature = str(self.mass_artifact_signature)
        if not signature:
            raise ValueError("mass_artifact_signature must be non-empty")
        reset = dict(self.reset_event)
        if reset.get("event") != "dual_averaging_reset":
            raise ValueError("reset_event must record dual_averaging_reset")
        object.__setattr__(self, "shrinkage", shrinkage)
        object.__setattr__(self, "mass_artifact_payload", dict(self.mass_artifact_payload))
        object.__setattr__(self, "mass_artifact_signature", signature)
        object.__setattr__(self, "reset_event", reset)

    def payload(self) -> Mapping[str, Any]:
        return {
            "window": self.window.payload(),
            "welford": self.welford.payload(),
            "shrinkage": self.shrinkage,
            "mass_artifact_payload": self.mass_artifact_payload,
            "mass_artifact_signature": self.mass_artifact_signature,
            "reset_event": self.reset_event,
        }


@dataclass(frozen=True)
class WindowedMassAdaptationResult:
    """Windowed mass adaptation evidence; not posterior convergence evidence."""

    policy: HMCTuningPolicy
    config: WindowedMassAdaptationConfig
    initial_mass_artifact_payload: Mapping[str, Any]
    initial_mass_artifact_signature: str
    shrinkage_target_signature: str
    compatibility: Mapping[str, Any]
    windows: tuple[WindowedWarmupWindow, ...]
    mass_updates: tuple[WindowedMassUpdate, ...]
    final_mass_artifact_payload: Mapping[str, Any]
    final_mass_artifact_signature: str
    step_size_trace: tuple[float, ...]
    acceptance_trace: tuple[float, ...]
    final_mass_artifact: Any | None = None
    target_failure_classification: Mapping[str, Any] | None = None
    passed: bool = True
    nonclaims: tuple[str, ...] = (
        "windowed mass adaptation diagnostic only",
        "non-default experimental policy",
        "not exact Stan equivalence",
        "no posterior convergence claim",
        "no sampler superiority claim",
        "no default adaptation readiness claim",
    )

    def __post_init__(self) -> None:
        initial_signature = str(self.initial_mass_artifact_signature)
        target_signature = str(self.shrinkage_target_signature)
        final_signature = str(self.final_mass_artifact_signature)
        if not initial_signature or not target_signature or not final_signature:
            raise ValueError("mass artifact signatures must be non-empty")
        windows = tuple(self.windows)
        if not windows:
            raise ValueError("windowed mass adaptation requires a non-empty schedule")
        updates = tuple(self.mass_updates)
        step_trace = tuple(float(item) for item in self.step_size_trace)
        accept_trace = tuple(float(item) for item in self.acceptance_trace)
        if not step_trace or not np.all(np.isfinite(step_trace)):
            raise ValueError("step_size_trace must be non-empty and finite")
        if any(step <= 0.0 for step in step_trace):
            raise ValueError("step_size_trace must be positive")
        if not accept_trace or not np.all(np.isfinite(accept_trace)):
            raise ValueError("acceptance_trace must be non-empty and finite")
        if any(accept < 0.0 or accept > 1.0 for accept in accept_trace):
            raise ValueError("acceptance_trace must lie in [0, 1]")
        target_failure = _validate_target_failure_classification(
            self.target_failure_classification
        )
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(self, "initial_mass_artifact_payload", dict(self.initial_mass_artifact_payload))
        object.__setattr__(self, "initial_mass_artifact_signature", initial_signature)
        object.__setattr__(self, "shrinkage_target_signature", target_signature)
        object.__setattr__(self, "compatibility", dict(self.compatibility))
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "mass_updates", updates)
        object.__setattr__(self, "final_mass_artifact_payload", dict(self.final_mass_artifact_payload))
        object.__setattr__(self, "final_mass_artifact_signature", final_signature)
        final_artifact = self.final_mass_artifact
        if final_artifact is not None:
            final_payload = _mass_artifact_payload(final_artifact)
            final_artifact_signature = _mass_artifact_signature(final_artifact)
            if dict(final_payload) != dict(self.final_mass_artifact_payload):
                raise ValueError("final_mass_artifact payload mismatch")
            if final_artifact_signature != final_signature:
                raise ValueError("final_mass_artifact signature mismatch")
        object.__setattr__(self, "final_mass_artifact", final_artifact)
        object.__setattr__(self, "step_size_trace", step_trace)
        object.__setattr__(self, "acceptance_trace", accept_trace)
        object.__setattr__(self, "target_failure_classification", target_failure)
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def final_step_size(self) -> float:
        return self.step_size_trace[-1]

    def semantic_checks(self) -> Mapping[str, bool]:
        slow_update_count = sum(1 for window in self.windows if window.update_mass)
        reset_count = sum(
            1
            for update in self.mass_updates
            if update.reset_event.get("event") == "dual_averaging_reset"
        )
        return {
            "window_schedule_contiguous": _windows_are_contiguous(self.windows),
            "mass_update_count_matches_slow_windows": (
                slow_update_count == len(self.mass_updates)
                if self.config.mass_policy == "windowed_adaptive"
                else len(self.mass_updates) == 0
            ),
            "every_update_has_dual_averaging_reset": reset_count
            == len(self.mass_updates),
            "final_mass_artifact_frozen_payload": bool(self.final_mass_artifact_signature),
            "fixed_identity_signature_unchanged": (
                self.config.mass_policy != "fixed_identity"
                or self.final_mass_artifact_signature
                == self.initial_mass_artifact_signature
            ),
            "shrinkage_target_compatible": bool(self.compatibility.get("compatible")),
            "does_not_report_posterior_convergence": True,
        }

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy": self.policy.payload(),
            "config": self.config.payload(),
            "initial_mass_artifact_payload": self.initial_mass_artifact_payload,
            "initial_mass_artifact_signature": self.initial_mass_artifact_signature,
            "shrinkage_target_signature": self.shrinkage_target_signature,
            "compatibility": self.compatibility,
            "windows": tuple(window.payload() for window in self.windows),
            "mass_updates": tuple(update.payload() for update in self.mass_updates),
            "final_mass_artifact_payload": self.final_mass_artifact_payload,
            "final_mass_artifact_signature": self.final_mass_artifact_signature,
            "trace": {
                "step_size": self.step_size_trace,
                "acceptance": self.acceptance_trace,
            },
            "diagnostics": {
                "passed": self.passed,
                "final_step_size": self.final_step_size,
                "final_step_size_finite": bool(np.isfinite(self.final_step_size)),
                "mass_update_count": len(self.mass_updates),
                "semantic_checks": self.semantic_checks(),
                "target_failure_classification": self.target_failure_classification,
                "reports_posterior_convergence": False,
                "nonclaims": self.nonclaims,
            },
        }


@dataclass(frozen=True)
class FixedTrajectoryTuningConfig:
    """Closed-grid fixed-trajectory tuning screen for a frozen HMC kernel."""

    num_leapfrog_step_candidates: tuple[int, ...]
    acceptance_band: tuple[float, float] = (0.65, 0.75)
    num_results: int = 16
    num_burnin_steps: int = 0
    seed: tuple[int, int] = (20260613, 5)
    target_trajectory_length: float = float(np.pi)

    def __post_init__(self) -> None:
        candidates = tuple(int(item) for item in self.num_leapfrog_step_candidates)
        if not candidates:
            raise ValueError("num_leapfrog_step_candidates must be non-empty")
        if any(item <= 0 for item in candidates):
            raise ValueError("num_leapfrog_step_candidates must be positive")
        lower, upper = _validate_acceptance_band(self.acceptance_band)
        draws = int(self.num_results)
        burnin = int(self.num_burnin_steps)
        if draws <= 0:
            raise ValueError("num_results must be positive")
        if burnin < 0:
            raise ValueError("num_burnin_steps must be non-negative")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain exactly two integers")
        target = float(self.target_trajectory_length)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_trajectory_length must be positive and finite")
        object.__setattr__(self, "num_leapfrog_step_candidates", candidates)
        object.__setattr__(self, "acceptance_band", (lower, upper))
        object.__setattr__(self, "num_results", draws)
        object.__setattr__(self, "num_burnin_steps", burnin)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "target_trajectory_length", target)

    def payload(self) -> Mapping[str, Any]:
        return {
            "num_leapfrog_step_candidates": self.num_leapfrog_step_candidates,
            "acceptance_band": self.acceptance_band,
            "num_results": self.num_results,
            "num_burnin_steps": self.num_burnin_steps,
            "seed": self.seed,
            "target_trajectory_length": self.target_trajectory_length,
        }


@dataclass(frozen=True)
class FixedTrajectoryCandidateResult:
    """One fixed leapfrog-count candidate for a frozen step size and mass."""

    step_size: float
    num_leapfrog_steps: int
    trajectory_length: float
    acceptance_rate: float | None
    log_accept_ratio_finite: bool
    finite_sample_count: int
    nonfinite_sample_count: int
    outcome: str
    grid_index: int
    vetoes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        step = float(self.step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("candidate step_size must be positive and finite")
        leapfrogs = int(self.num_leapfrog_steps)
        if leapfrogs <= 0:
            raise ValueError("candidate num_leapfrog_steps must be positive")
        trajectory = float(self.trajectory_length)
        if not np.isfinite(trajectory) or trajectory <= 0.0:
            raise ValueError("candidate trajectory_length must be positive and finite")
        acceptance = (
            None if self.acceptance_rate is None else float(self.acceptance_rate)
        )
        if acceptance is not None and (
            not np.isfinite(acceptance) or acceptance < 0.0 or acceptance > 1.0
        ):
            raise ValueError("candidate acceptance_rate must be in [0, 1]")
        outcome = str(self.outcome)
        if not outcome:
            raise ValueError("candidate outcome must be non-empty")
        object.__setattr__(self, "step_size", step)
        object.__setattr__(self, "num_leapfrog_steps", leapfrogs)
        object.__setattr__(self, "trajectory_length", trajectory)
        object.__setattr__(self, "acceptance_rate", acceptance)
        object.__setattr__(self, "log_accept_ratio_finite", bool(self.log_accept_ratio_finite))
        object.__setattr__(self, "finite_sample_count", int(self.finite_sample_count))
        object.__setattr__(self, "nonfinite_sample_count", int(self.nonfinite_sample_count))
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "grid_index", int(self.grid_index))
        object.__setattr__(self, "vetoes", tuple(str(item) for item in self.vetoes))

    def payload(self) -> Mapping[str, Any]:
        return {
            "step_size": self.step_size,
            "num_leapfrog_steps": self.num_leapfrog_steps,
            "trajectory_length": self.trajectory_length,
            "acceptance_rate": self.acceptance_rate,
            "log_accept_ratio_finite": self.log_accept_ratio_finite,
            "finite_sample_count": self.finite_sample_count,
            "nonfinite_sample_count": self.nonfinite_sample_count,
            "outcome": self.outcome,
            "grid_index": self.grid_index,
            "vetoes": self.vetoes,
        }


@dataclass(frozen=True)
class FixedTrajectoryTuningResult:
    """Frozen-kernel trajectory tuning evidence, not convergence evidence."""

    policy: HMCTuningPolicy
    config: FixedTrajectoryTuningConfig
    frozen_mass_artifact_payload: Mapping[str, Any]
    frozen_mass_artifact_signature: str
    frozen_step_size: float
    candidate_results: tuple[FixedTrajectoryCandidateResult, ...]
    selected_num_leapfrog_steps: int | None
    selected_trajectory_length: float | None
    production_leapfrog_rule: Mapping[str, Any]
    blocker_reason: str | None = None
    nonclaims: tuple[str, ...] = (
        "fixed-trajectory tuning diagnostic only",
        "acceptance band is a tuning promotion screen only",
        "no posterior convergence claim",
        "no sampler superiority claim",
        "no default adaptation readiness claim",
        "no GPU/XLA readiness claim",
        "no MacroFinance model success claim",
    )

    def __post_init__(self) -> None:
        signature = str(self.frozen_mass_artifact_signature)
        if not signature:
            raise ValueError("frozen_mass_artifact_signature must be non-empty")
        step = float(self.frozen_step_size)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("frozen_step_size must be positive and finite")
        candidates = tuple(self.candidate_results)
        if not candidates:
            raise ValueError("fixed-trajectory tuning requires candidate results")
        selected_steps = (
            None
            if self.selected_num_leapfrog_steps is None
            else int(self.selected_num_leapfrog_steps)
        )
        selected_length = (
            None
            if self.selected_trajectory_length is None
            else float(self.selected_trajectory_length)
        )
        if selected_steps is not None and selected_steps <= 0:
            raise ValueError("selected_num_leapfrog_steps must be positive")
        if selected_length is not None and (
            not np.isfinite(selected_length) or selected_length <= 0.0
        ):
            raise ValueError("selected_trajectory_length must be positive and finite")
        blocker = None if self.blocker_reason is None else str(self.blocker_reason)
        nonclaims = tuple(str(item) for item in self.nonclaims)
        if not nonclaims:
            raise ValueError("nonclaims must be non-empty")
        object.__setattr__(
            self,
            "frozen_mass_artifact_payload",
            dict(self.frozen_mass_artifact_payload),
        )
        object.__setattr__(self, "frozen_mass_artifact_signature", signature)
        object.__setattr__(self, "frozen_step_size", step)
        object.__setattr__(self, "candidate_results", candidates)
        object.__setattr__(self, "selected_num_leapfrog_steps", selected_steps)
        object.__setattr__(self, "selected_trajectory_length", selected_length)
        object.__setattr__(
            self,
            "production_leapfrog_rule",
            dict(self.production_leapfrog_rule),
        )
        object.__setattr__(self, "blocker_reason", blocker)
        object.__setattr__(self, "nonclaims", nonclaims)

    @property
    def selected_candidate(self) -> FixedTrajectoryCandidateResult | None:
        for candidate in self.candidate_results:
            if (
                candidate.outcome == "passed_screen"
                and candidate.num_leapfrog_steps == self.selected_num_leapfrog_steps
                and candidate.trajectory_length == self.selected_trajectory_length
            ):
                return candidate
        return None

    @property
    def passed(self) -> bool:
        return self.selected_candidate is not None and self.blocker_reason is None

    def payload(self) -> Mapping[str, Any]:
        return {
            "policy": self.policy.payload(),
            "config": self.config.payload(),
            "frozen_mass_artifact_payload": self.frozen_mass_artifact_payload,
            "frozen_mass_artifact_signature": self.frozen_mass_artifact_signature,
            "frozen_step_size": self.frozen_step_size,
            "candidate_results": tuple(
                candidate.payload() for candidate in self.candidate_results
            ),
            "selected_configuration": None
            if self.selected_candidate is None
            else self.selected_candidate.payload(),
            "selected_num_leapfrog_steps": self.selected_num_leapfrog_steps,
            "selected_trajectory_length": self.selected_trajectory_length,
            "no_further_adaptation": self.selected_candidate is not None,
            "production_leapfrog_rule": self.production_leapfrog_rule,
            "blocker_reason": self.blocker_reason,
            "diagnostics": {
                "passed": self.passed,
                "acceptance_band_role": "closed tuning promotion screen only",
                "selection_rule": (
                    "closest_acceptance_to_closed_band_midpoint_then_shorter_"
                    "trajectory_then_declared_grid_order"
                ),
                "candidate_count": len(self.candidate_results),
                "passed_candidate_count": sum(
                    candidate.outcome == "passed_screen"
                    for candidate in self.candidate_results
                ),
                "reports_posterior_convergence": False,
                "reports_sampler_superiority": False,
                "reports_default_readiness": False,
                "reports_gpu_xla_readiness": False,
                "reports_macrofinance_model_success": False,
                "nonclaims": self.nonclaims,
            },
        }


def normalize_hmc_tuning_policy(
    tuning_policy: str | HMCTuningPolicy | None,
) -> HMCTuningPolicy:
    """Normalize reviewed tuning policies while keeping raw adaptation fail-closed."""

    if tuning_policy is None:
        return HMCTuningPolicy.fixed_kernel_screen()
    if isinstance(tuning_policy, HMCTuningPolicy):
        return tuning_policy
    label = str(tuning_policy)
    if label in {"fixed_kernel_no_adaptation", "fixed_kernel_screen"}:
        return HMCTuningPolicy.fixed_kernel_screen()
    raise ValueError(
        "HMC adaptation is fail-closed: use 'fixed_kernel_no_adaptation' "
        "or pass a reviewed HMCTuningPolicy object"
    )


def require_executable_tuning_policy(policy: HMCTuningPolicy) -> HMCTuningPolicy:
    if not policy.implemented:
        raise ValueError(f"HMC tuning policy {policy.label!r} is not implemented")
    if policy.label == "fixed_kernel_screen":
        return policy
    if policy.uses_dual_averaging:
        if not policy.enabled:
            raise ValueError("dual-averaging tuning policy must be enabled")
        if policy.num_adaptation_steps <= 0:
            raise ValueError("dual-averaging tuning policy requires adaptation steps")
        if policy.target_accept_prob is None:
            raise ValueError("dual-averaging tuning policy requires target_accept_prob")
        return policy
    if policy.uses_windowed_mass_adaptation:
        if not policy.enabled:
            raise ValueError("windowed mass adaptation policy must be enabled")
        if policy.num_adaptation_steps <= 0:
            raise ValueError("windowed mass adaptation requires adaptation steps")
        if policy.target_accept_prob is None:
            raise ValueError("windowed mass adaptation requires target_accept_prob")
        return policy
    if policy.uses_fixed_trajectory_tuning:
        if not policy.enabled:
            raise ValueError("fixed-trajectory tuning policy must be enabled")
        if policy.num_adaptation_steps != 0:
            raise ValueError("fixed-trajectory tuning must not adapt mass or step size")
        if policy.target_accept_prob is None:
            raise ValueError("fixed-trajectory tuning requires target_accept_prob")
        return policy
    raise ValueError(f"HMC tuning policy {policy.label!r} is not executable")


def classify_fixed_kernel_screen_with_tuning_policy(
    policy: HMCTuningPolicy,
    screen: HMCScreenResult,
    *,
    acceptance_rate_by_chain: Any,
    step_size: float | None = None,
    num_leapfrog_steps: int | None = None,
    max_abs_log_accept_ratio: float | None = None,
) -> HMCFailureClassification:
    """Classify a fixed-kernel screen under the explicit Stage 7 policy label."""

    policy = require_executable_tuning_policy(policy)
    if policy.label != "fixed_kernel_screen":
        raise ValueError("fixed-kernel screen classification requires fixed_kernel_screen")
    return classify_hmc_screen(
        screen,
        acceptance_rate_by_chain=acceptance_rate_by_chain,
        fixed_kernel_used=True,
        num_adaptation_steps_zero=True,
        step_size=step_size,
        num_leapfrog_steps=num_leapfrog_steps,
        max_abs_log_accept_ratio=max_abs_log_accept_ratio,
    )


def classify_hmc_tuning_diagnostic(
    policy: HMCTuningPolicy,
    *,
    target_evaluation: Any | None = None,
    required_arrays_finite: bool = True,
    hmc_error: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Separate tuning telemetry from target validity and convergence claims."""

    require_executable_tuning_policy(policy)
    if hmc_error is not None:
        return {
            "passed": False,
            "classification": "hmc_execution_error",
            "diagnostic_role": "hard_veto",
            "nonclaims": policy.nonclaims,
        }
    if getattr(target_evaluation, "fallback_used", False):
        return {
            "passed": False,
            "classification": "target_invalidity_not_tuning_success",
            "diagnostic_role": "hard_veto",
            "failure_label": getattr(target_evaluation, "failure_label", None),
            "nonclaims": policy.nonclaims,
        }
    if not bool(required_arrays_finite):
        return {
            "passed": False,
            "classification": "nonfinite_tuning_diagnostic",
            "diagnostic_role": "hard_veto",
            "nonclaims": policy.nonclaims,
        }
    return {
        "passed": True,
        "classification": "tuning_diagnostic_passed_not_convergence",
        "diagnostic_role": "diagnostic_only",
        "reports_posterior_convergence": False,
        "nonclaims": policy.nonclaims,
    }


def bracket_initial_step_size(
    *,
    target_probe_fn: Any,
    initial_step_size: float = 1.0,
    max_attempts: int = 8,
    contraction: float = 0.5,
) -> InitialStepBracketResult:
    """Find a finite initial step and record every attempt.

    ``target_probe_fn(step_size)`` must return a truthy value only when the
    target-side transition/probe is finite.  Exceptions are recorded as
    fail-closed reasons instead of being misclassified as convergence evidence.
    """

    step = float(initial_step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("initial_step_size must be positive and finite")
    attempts_count = int(max_attempts)
    if attempts_count <= 0:
        raise ValueError("max_attempts must be positive")
    contraction_value = float(contraction)
    if (
        not np.isfinite(contraction_value)
        or contraction_value <= 0.0
        or contraction_value >= 1.0
    ):
        raise ValueError("contraction must be finite and in (0, 1)")

    attempts: list[InitialStepBracketAttempt] = []
    selected: float | None = None
    for _index in range(attempts_count):
        try:
            finite = bool(target_probe_fn(step))
            reason = "finite_probe" if finite else "nonfinite_probe"
        except Exception as exc:  # noqa: BLE001 - provenance records fail-closed probe.
            finite = False
            reason = f"{exc.__class__.__name__}: {exc}"
        attempts.append(
            InitialStepBracketAttempt(
                step_size=step,
                finite=finite,
                reason=reason,
            )
        )
        if finite:
            selected = step
            break
        step *= contraction_value
    return InitialStepBracketResult(
        selected_step_size=selected,
        attempts=tuple(attempts),
        passed=selected is not None,
    )


def run_fixed_mass_step_tuning_diagnostic(
    policy: HMCTuningPolicy,
    *,
    mass_artifact: Any,
    initial_state: Any,
    target_probe_fn: Any,
    target_failure_classification: Mapping[str, Any] | None,
    num_results: int = 4,
    num_burnin_steps: int = 4,
    step_size: float = 0.05,
    num_leapfrog_steps: int = 2,
    seed: tuple[int, int] = (20260613, 3),
) -> FixedMassStepTuningResult:
    """Run tiny fixed-mass step-size telemetry without mutating the mass artifact."""

    policy = require_executable_tuning_policy(policy)
    if policy.label != "fixed_mass_dual_averaging":
        raise ValueError("fixed-mass step tuning requires fixed_mass_dual_averaging")
    payload_before = _mass_artifact_payload(mass_artifact)
    signature_before = _mass_artifact_signature(mass_artifact)
    bracket = bracket_initial_step_size(
        target_probe_fn=target_probe_fn,
        initial_step_size=step_size,
    )
    if not bracket.passed:
        signature_after = _mass_artifact_signature(mass_artifact)
        invariant = _frozen_mass_invariant(signature_before, signature_after)
        return FixedMassStepTuningResult(
            policy=policy,
            mass_artifact_payload=payload_before,
            mass_artifact_signature=signature_before,
            initial_step_bracket=bracket,
            diagnostic=_failed_fixed_mass_diagnostic(
                policy,
                classification="initial_step_bracketing_failed",
            ),
            frozen_mass_invariant=invariant,
            target_failure_classification=target_failure_classification,
            passed=False,
        )
    try:
        diagnostic = run_gaussian_dual_averaging_diagnostic(
            policy,
            initial_state=initial_state,
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            step_size=float(bracket.selected_step_size),
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed,
        )
        passed = True
    except Exception as exc:  # noqa: BLE001 - return evidence artifact with failure cause.
        diagnostic = _failed_fixed_mass_diagnostic(
            policy,
            classification="hmc_execution_error",
            error=exc,
        )
        passed = False
    signature_after = _mass_artifact_signature(mass_artifact)
    invariant = _frozen_mass_invariant(signature_before, signature_after)
    return FixedMassStepTuningResult(
        policy=policy,
        mass_artifact_payload=payload_before,
        mass_artifact_signature=signature_before,
        initial_step_bracket=bracket,
        diagnostic=diagnostic,
        frozen_mass_invariant=invariant,
        target_failure_classification=target_failure_classification,
        passed=passed,
    )


def build_windowed_warmup_schedule(
    config: WindowedMassAdaptationConfig,
) -> tuple[WindowedWarmupWindow, ...]:
    """Build a contiguous fast/slow/final warmup schedule."""

    windows: list[WindowedWarmupWindow] = []
    index = 0
    if config.initial_buffer > 0:
        windows.append(
            WindowedWarmupWindow(
                index=index,
                kind="initial_fast",
                start=0,
                end=config.initial_buffer,
                update_mass=False,
            )
        )
        index += 1
    slow_start = config.initial_buffer
    slow_end = config.warmup_steps - config.final_buffer
    window_size = config.first_window_size
    cursor = slow_start
    while cursor < slow_end:
        remaining = slow_end - cursor
        size = min(window_size, remaining)
        if 0 < remaining - size < config.min_window_samples:
            size = remaining
        windows.append(
            WindowedWarmupWindow(
                index=index,
                kind="slow",
                start=cursor,
                end=cursor + size,
                update_mass=config.mass_policy == "windowed_adaptive",
            )
        )
        index += 1
        cursor += size
        window_size *= 2
    if config.final_buffer > 0:
        windows.append(
            WindowedWarmupWindow(
                index=index,
                kind="final_fast",
                start=slow_end,
                end=config.warmup_steps,
                update_mass=False,
            )
        )
    if not _windows_are_contiguous(windows):
        raise ValueError("windowed warmup schedule must be contiguous")
    return tuple(windows)


def welford_covariance(samples: Any) -> WelfordCovarianceResult:
    """Compute sample covariance with Welford's online recursion."""

    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError("Welford samples must be a rank-2 array")
    if array.shape[0] <= 1:
        raise ValueError("Welford covariance requires at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError("Welford samples must be finite")
    mean = np.zeros(array.shape[1], dtype=float)
    m2 = np.zeros((array.shape[1], array.shape[1]), dtype=float)
    count = 0
    for row in array:
        count += 1
        delta = row - mean
        mean = mean + delta / count
        delta2 = row - mean
        m2 = m2 + np.outer(delta, delta2)
    covariance = m2 / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    return WelfordCovarianceResult(
        count=count,
        mean=mean,
        covariance=covariance,
        finite=True,
    )


def validate_windowed_shrinkage_target(
    *,
    initial_mass_artifact: Any,
    shrinkage_target_mass_artifact: Any,
    expected_adapter_signature: str | None = None,
) -> Mapping[str, Any]:
    """Fail closed unless the empirical covariance and target share coordinates."""

    initial_payload = _mass_artifact_payload(initial_mass_artifact)
    target_payload = _mass_artifact_payload(shrinkage_target_mass_artifact)
    initial_signature = _mass_artifact_signature(initial_mass_artifact)
    target_signature = _mass_artifact_signature(shrinkage_target_mass_artifact)
    checks = {
        "initial_dimension": int(initial_payload["dimension"]),
        "target_dimension": int(target_payload["dimension"]),
        "dimension_match": int(initial_payload["dimension"])
        == int(target_payload["dimension"]),
        "adapter_signature_match": initial_payload["adapter_signature"]
        == target_payload["adapter_signature"],
        "covariance_shape_match": np.asarray(initial_mass_artifact.covariance).shape
        == np.asarray(shrinkage_target_mass_artifact.covariance).shape,
        "factor_shape_match": np.asarray(initial_mass_artifact.factor).shape
        == np.asarray(shrinkage_target_mass_artifact.factor).shape,
        "coordinate_order_source": "adapter_signature",
        "initial_signature": initial_signature,
        "target_signature": target_signature,
    }
    if expected_adapter_signature is not None:
        checks["expected_adapter_signature_match"] = (
            str(expected_adapter_signature) == initial_payload["adapter_signature"]
            == target_payload["adapter_signature"]
        )
    compatible = all(
        bool(checks[key])
        for key in (
            "dimension_match",
            "adapter_signature_match",
            "covariance_shape_match",
            "factor_shape_match",
        )
    ) and bool(checks.get("expected_adapter_signature_match", True))
    checks["compatible"] = compatible
    if not compatible:
        raise ValueError(
            "windowed mass shrinkage target is stale or coordinate-incompatible"
        )
    return checks


def run_windowed_mass_adaptation_diagnostic(
    policy: HMCTuningPolicy | str,
    *,
    config: WindowedMassAdaptationConfig,
    initial_mass_artifact: Any,
    warmup_draws: Any,
    initial_step_size: float,
    acceptance_trace: Any | None = None,
    shrinkage_target_mass_artifact: Any | None = None,
    expected_adapter_signature: str | None = None,
    target_failure_classification: Mapping[str, Any] | None = None,
) -> WindowedMassAdaptationResult:
    """Run a bounded windowed-mass semantic diagnostic on supplied warmup draws.

    The supplied draws are diagnostic warmup states. They are used to exercise
    Welford covariance, shrinkage, mass rebuilds, and reset telemetry without
    claiming posterior convergence or exact Stan equivalence.
    """

    if not isinstance(policy, HMCTuningPolicy):
        raise ValueError(
            "windowed mass adaptation requires a reviewed HMCTuningPolicy object"
        )
    policy = require_executable_tuning_policy(policy)
    if policy.label != "windowed_mass_adaptation":
        raise ValueError("windowed mass adaptation requires windowed_mass_adaptation")
    if policy.num_adaptation_steps != config.warmup_steps:
        raise ValueError(
            "windowed mass adaptation policy steps must equal config.warmup_steps"
        )
    target_failure = _validate_target_failure_classification(
        target_failure_classification
    )
    draws = np.asarray(warmup_draws, dtype=float)
    if draws.ndim != 2:
        raise ValueError("warmup_draws must be a rank-2 array")
    if draws.shape[0] != config.warmup_steps:
        raise ValueError("warmup_draws row count must equal config.warmup_steps")
    if draws.shape[1] != int(_mass_artifact_payload(initial_mass_artifact)["dimension"]):
        raise ValueError("warmup_draws dimension must match mass artifact dimension")
    if not np.all(np.isfinite(draws)):
        raise ValueError("warmup_draws must be finite")
    step = float(initial_step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("initial_step_size must be positive and finite")
    if acceptance_trace is None:
        acceptance = np.full(config.warmup_steps, policy.target_accept_prob, dtype=float)
    else:
        acceptance = np.asarray(acceptance_trace, dtype=float)
    if acceptance.shape != (config.warmup_steps,):
        raise ValueError("acceptance_trace length must equal config.warmup_steps")
    if not np.all(np.isfinite(acceptance)):
        raise ValueError("acceptance_trace must be finite")
    if np.any((acceptance < 0.0) | (acceptance > 1.0)):
        raise ValueError("acceptance_trace must lie in [0, 1]")

    target_artifact = (
        initial_mass_artifact
        if shrinkage_target_mass_artifact is None
        else shrinkage_target_mass_artifact
    )
    compatibility = validate_windowed_shrinkage_target(
        initial_mass_artifact=initial_mass_artifact,
        shrinkage_target_mass_artifact=target_artifact,
        expected_adapter_signature=expected_adapter_signature,
    )
    windows = build_windowed_warmup_schedule(config)
    initial_signature = _mass_artifact_signature(initial_mass_artifact)
    target_signature = _mass_artifact_signature(target_artifact)
    step_trace: list[float] = []
    updates: list[WindowedMassUpdate] = []
    current_mass_artifact = initial_mass_artifact
    current_signature = initial_signature
    for window in windows:
        for accept in acceptance[window.start : window.end]:
            step = _adapt_windowed_step_size(
                step,
                accept=float(accept),
                target_accept=float(policy.target_accept_prob),
                config=config,
            )
            step_trace.append(step)
        if config.mass_policy == "fixed_identity" or not window.update_mass:
            continue
        welford = welford_covariance(draws[window.start : window.end])
        covariance = _shrink_covariance(
            empirical_covariance=welford.covariance,
            target_covariance=np.asarray(target_artifact.covariance, dtype=float),
            shrinkage=config.mass_shrinkage,
        )
        previous_signature = current_signature
        current_mass_artifact = _rebuild_windowed_mass_artifact(
            base_mass_artifact=target_artifact,
            covariance=covariance,
            config=config,
            window=window,
        )
        current_signature = _mass_artifact_signature(current_mass_artifact)
        updates.append(
            WindowedMassUpdate(
                window=window,
                welford=welford,
                shrinkage=config.mass_shrinkage,
                mass_artifact_payload=_mass_artifact_payload(current_mass_artifact),
                mass_artifact_signature=current_signature,
                reset_event={
                    "event": "dual_averaging_reset",
                    "window_index": window.index,
                    "previous_mass_signature": previous_signature,
                    "new_mass_signature": current_signature,
                    "step_size_after_reset": step,
                    "diagnostic_role": "metric_boundary_reset_semantics",
                    "nonclaims": policy.nonclaims,
                },
            )
        )
    return WindowedMassAdaptationResult(
        policy=policy,
        config=config,
        initial_mass_artifact_payload=_mass_artifact_payload(initial_mass_artifact),
        initial_mass_artifact_signature=initial_signature,
        shrinkage_target_signature=target_signature,
        compatibility=compatibility,
        windows=windows,
        mass_updates=tuple(updates),
        final_mass_artifact_payload=_mass_artifact_payload(current_mass_artifact),
        final_mass_artifact_signature=current_signature,
        step_size_trace=tuple(step_trace),
        acceptance_trace=tuple(float(item) for item in acceptance),
        final_mass_artifact=current_mass_artifact,
        target_failure_classification=target_failure,
        passed=not (
            target_failure is not None
            and target_failure.get("diagnostic_role") == "hard_veto"
        ),
    )


def production_leapfrog_count(
    step_size: float,
    max_leapfrog: int,
    min_leapfrog: int,
    *,
    target_traj: float = float(np.pi),
    epsilon_guard: float = 1.0e-12,
) -> tuple[int, int]:
    """Return production and theory leapfrog counts for a fixed trajectory."""

    step = float(step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be positive and finite")
    max_l = int(max_leapfrog)
    min_l = int(min_leapfrog)
    if max_l <= 0 or min_l <= 0:
        raise ValueError("leapfrog bounds must be positive")
    if min_l > max_l:
        raise ValueError("min_leapfrog must not exceed max_leapfrog")
    target = float(target_traj)
    guard = float(epsilon_guard)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_traj must be positive and finite")
    if not np.isfinite(guard) or guard <= 0.0:
        raise ValueError("epsilon_guard must be positive and finite")
    theory = int(np.ceil(target / max(step, guard)))
    production = min(theory, max_l)
    production = max(production, min_l)
    return int(production), int(theory)


def run_fixed_trajectory_tuning_diagnostic(
    policy: HMCTuningPolicy | str,
    *,
    config: FixedTrajectoryTuningConfig,
    windowed_mass_result: WindowedMassAdaptationResult | None = None,
) -> FixedTrajectoryTuningResult:
    """Select a fixed leapfrog count from a frozen Phase 4 mass/step artifact."""

    if not isinstance(policy, HMCTuningPolicy):
        raise ValueError(
            "fixed-trajectory tuning requires a reviewed HMCTuningPolicy object"
        )
    policy = require_executable_tuning_policy(policy)
    if policy.label != "fixed_trajectory_tuning":
        raise ValueError("fixed-trajectory tuning requires fixed_trajectory_tuning")
    if not isinstance(windowed_mass_result, WindowedMassAdaptationResult):
        raise ValueError(
            "fixed-trajectory tuning requires a Phase 4 WindowedMassAdaptationResult"
        )
    if not bool(windowed_mass_result.semantic_checks().get("final_mass_artifact_frozen_payload")):
        raise ValueError("Phase 4 windowed mass result must expose a frozen final mass")
    mass_payload = windowed_mass_result.final_mass_artifact_payload
    mass_signature = windowed_mass_result.final_mass_artifact_signature
    step = windowed_mass_result.final_step_size
    if not mass_signature:
        raise ValueError("frozen mass signature must be non-empty")
    required_payload_keys = (
        "dimension",
        "adapter_signature",
        "covariance_source",
        "source",
        "nonclaims",
    )
    missing = tuple(key for key in required_payload_keys if key not in mass_payload)
    if missing:
        raise ValueError("frozen mass payload missing required fields: " + ", ".join(missing))
    if str(mass_payload["source"]) != "windowed_mass_adaptation":
        raise ValueError(
            "fixed-trajectory tuning requires a Phase 4 windowed mass artifact"
        )
    target_dimension = int(mass_payload["dimension"])
    if target_dimension <= 0:
        raise ValueError("frozen mass payload dimension must be positive")
    step = float(step)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("frozen step size must be positive and finite")

    lower, upper = config.acceptance_band
    midpoint = 0.5 * (lower + upper)
    start = time.perf_counter()
    candidates = tuple(
        _run_fixed_trajectory_candidate(
            step_size=step,
            num_leapfrog_steps=leapfrog_count,
            grid_index=index,
            config=config,
            acceptance_band=(lower, upper),
            target_dimension=target_dimension,
        )
        for index, leapfrog_count in enumerate(config.num_leapfrog_step_candidates)
    )
    selected = _select_fixed_trajectory_candidate(
        candidates,
        midpoint=midpoint,
    )
    blocker = None if selected is not None else "no_candidate_in_closed_acceptance_promotion_band"
    production_l, theory_l = production_leapfrog_count(
        step,
        max(config.num_leapfrog_step_candidates),
        min(config.num_leapfrog_step_candidates),
        target_traj=config.target_trajectory_length,
    )
    return FixedTrajectoryTuningResult(
        policy=policy,
        config=config,
        frozen_mass_artifact_payload=mass_payload,
        frozen_mass_artifact_signature=mass_signature,
        frozen_step_size=step,
        candidate_results=candidates,
        selected_num_leapfrog_steps=None if selected is None else selected.num_leapfrog_steps,
        selected_trajectory_length=None if selected is None else selected.trajectory_length,
        production_leapfrog_rule={
            "target_trajectory_length": config.target_trajectory_length,
            "production_leapfrog_count": production_l,
            "theory_leapfrog_count": theory_l,
            "min_leapfrog": min(config.num_leapfrog_step_candidates),
            "max_leapfrog": max(config.num_leapfrog_step_candidates),
            "target_dimension": target_dimension,
            "elapsed_s": float(time.perf_counter() - start),
        },
        blocker_reason=blocker,
    )


def run_gaussian_dual_averaging_diagnostic(
    policy: HMCTuningPolicy,
    *,
    initial_state: Any,
    num_results: int = 4,
    num_burnin_steps: int = 4,
    step_size: float = 0.05,
    num_leapfrog_steps: int = 2,
    seed: tuple[int, int] = (20260609, 3),
) -> HMCTuningDiagnosticResult:
    """Run a tiny Gaussian TFP diagnostic for reviewed dual averaging.

    This helper exists to prove telemetry extraction and policy gating. It is
    not a posterior validation run and intentionally uses only a standard
    Gaussian target.
    """

    policy = require_executable_tuning_policy(policy)
    if not policy.uses_dual_averaging:
        raise ValueError("Gaussian dual-averaging diagnostic requires a dual policy")
    for name, value in {
        "num_results": num_results,
        "num_burnin_steps": num_burnin_steps,
        "num_leapfrog_steps": num_leapfrog_steps,
    }.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if policy.num_adaptation_steps > int(num_burnin_steps):
        raise ValueError("num_adaptation_steps must not exceed num_burnin_steps")
    step = float(step_size)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be positive and finite")

    import tensorflow as tf
    import tensorflow_probability as tfp

    state = tf.convert_to_tensor(initial_state, dtype=tf.float64)
    tfm = tfp.mcmc

    def target_log_prob(x: Any) -> Any:
        values = tf.convert_to_tensor(x, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1)

    base_kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(step, dtype=tf.float64),
        num_leapfrog_steps=int(num_leapfrog_steps),
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=base_kernel,
        num_adaptation_steps=policy.num_adaptation_steps,
        target_accept_prob=tf.constant(policy.target_accept_prob, dtype=tf.float64),
    )

    def trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
        return {
            "is_accepted": kernel_results.inner_results.is_accepted,
            "log_accept_ratio": kernel_results.inner_results.log_accept_ratio,
            "step_size": kernel_results.new_step_size,
            "target_accept_prob": kernel_results.target_accept_prob,
            "num_adaptation_steps": kernel_results.num_adaptation_steps,
        }

    start = time.perf_counter()
    samples, trace = tfm.sample_chain(
        num_results=int(num_results),
        num_burnin_steps=int(num_burnin_steps),
        current_state=state,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=tf.constant(tuple(int(item) for item in seed), dtype=tf.int32),
    )
    elapsed = time.perf_counter() - start

    sample_array = np.asarray(samples.numpy(), dtype=float)
    step_sizes = np.asarray(trace["step_size"].numpy(), dtype=float)
    accepted = np.asarray(trace["is_accepted"].numpy(), dtype=bool)
    log_accept_ratio = np.asarray(trace["log_accept_ratio"].numpy(), dtype=float)
    target_accept = float(np.asarray(trace["target_accept_prob"].numpy()).ravel()[-1])
    adaptation_steps = int(
        np.asarray(trace["num_adaptation_steps"].numpy()).ravel()[-1]
    )
    diagnostics = {
        "policy_label": policy.label,
        "adaptation_policy": policy.adaptation_policy,
        "num_adaptation_steps": adaptation_steps,
        "target_accept_prob": target_accept,
        "initial_step_size": step,
        "final_step_size": float(np.asarray(step_sizes).ravel()[-1]),
        "final_step_size_finite": bool(np.all(np.isfinite(step_sizes))),
        "acceptance_rate": float(np.mean(accepted)) if accepted.size else float("nan"),
        "log_accept_ratio_finite": bool(np.all(np.isfinite(log_accept_ratio))),
        "finite_sample_count": int(np.sum(np.all(np.isfinite(sample_array), axis=-1))),
        "nonfinite_sample_count": int(
            np.sum(~np.all(np.isfinite(sample_array), axis=-1))
        ),
        "reports_posterior_convergence": False,
    }
    metadata = {
        "runtime": "tfp.mcmc.DualAveragingStepSizeAdaptation",
        "diagnostic_role": policy.diagnostic_role,
        "elapsed_s": elapsed,
        "nonclaims": policy.nonclaims,
    }
    trace_payload = {
        "is_accepted": accepted.tolist(),
        "log_accept_ratio": log_accept_ratio.tolist(),
        "step_size": step_sizes.tolist(),
    }
    return HMCTuningDiagnosticResult(
        policy=policy,
        diagnostics=diagnostics,
        trace=trace_payload,
        metadata=metadata,
    )


def _mass_artifact_payload(mass_artifact: Any) -> Mapping[str, Any]:
    payload = mass_artifact.signature_payload()
    required = (
        "dimension",
        "adapter_signature",
        "covariance_source",
        "eigen_summary",
        "precision_eigen_summary",
        "regularization_report",
        "nonclaims",
    )
    missing = tuple(key for key in required if key not in payload)
    if missing:
        raise ValueError(
            "mass artifact payload missing required fields: " + ", ".join(missing)
        )
    return payload


def _mass_artifact_signature(mass_artifact: Any) -> str:
    _mass_artifact_payload(mass_artifact)
    return mass_artifact_signature(mass_artifact)


def _adapt_windowed_step_size(
    step_size: float,
    *,
    accept: float,
    target_accept: float,
    config: WindowedMassAdaptationConfig,
) -> float:
    log_step = np.log(float(step_size))
    log_step += config.step_adaptation_rate * (float(accept) - float(target_accept))
    clipped = np.clip(np.exp(log_step), config.step_size_floor, config.step_size_ceiling)
    return float(clipped)


def _validate_acceptance_band(band: tuple[float, float]) -> tuple[float, float]:
    if len(tuple(band)) != 2:
        raise ValueError("acceptance_band must contain exactly two values")
    lower, upper = (float(band[0]), float(band[1]))
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("acceptance_band values must be finite")
    if not (0.0 < lower <= upper < 1.0):
        raise ValueError("acceptance_band must satisfy 0 < lower <= upper < 1")
    return lower, upper


def _shrink_covariance(
    *,
    empirical_covariance: Any,
    target_covariance: Any,
    shrinkage: float,
) -> np.ndarray:
    empirical = np.asarray(empirical_covariance, dtype=float)
    target = np.asarray(target_covariance, dtype=float)
    if empirical.shape != target.shape:
        raise ValueError("empirical and target covariance shapes must match")
    if not np.all(np.isfinite(empirical)) or not np.all(np.isfinite(target)):
        raise ValueError("empirical and target covariance must be finite")
    value = (1.0 - float(shrinkage)) * empirical + float(shrinkage) * target
    return 0.5 * (value + value.T)


def _run_fixed_trajectory_candidate(
    *,
    step_size: float,
    num_leapfrog_steps: int,
    grid_index: int,
    config: FixedTrajectoryTuningConfig,
    acceptance_band: tuple[float, float],
    target_dimension: int,
) -> FixedTrajectoryCandidateResult:
    import tensorflow as tf
    import tensorflow_probability as tfp

    lower, upper = acceptance_band
    tfm = tfp.mcmc
    state = tf.zeros([int(target_dimension)], dtype=tf.float64)

    def target_log_prob(x: Any) -> Any:
        values = tf.convert_to_tensor(x, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(values), axis=-1)

    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob,
        step_size=tf.constant(float(step_size), dtype=tf.float64),
        num_leapfrog_steps=int(num_leapfrog_steps),
    )

    def trace_fn(_state: Any, kernel_results: Any) -> Mapping[str, Any]:
        return {
            "is_accepted": kernel_results.is_accepted,
            "log_accept_ratio": kernel_results.log_accept_ratio,
        }

    try:
        samples, trace = tfm.sample_chain(
            num_results=config.num_results,
            num_burnin_steps=config.num_burnin_steps,
            current_state=state,
            kernel=kernel,
            trace_fn=trace_fn,
            seed=tf.constant(
                tuple(int(item) for item in config.seed),
                dtype=tf.int32,
            ),
        )
        sample_array = np.asarray(samples.numpy(), dtype=float)
        accepted = np.asarray(trace["is_accepted"].numpy(), dtype=bool)
        log_accept_ratio = np.asarray(trace["log_accept_ratio"].numpy(), dtype=float)
        finite_rows = np.all(np.isfinite(sample_array), axis=-1)
        finite_count = int(np.sum(finite_rows))
        nonfinite_count = int(np.sum(~finite_rows))
        log_accept_finite = bool(np.all(np.isfinite(log_accept_ratio)))
        acceptance_rate = float(np.mean(accepted)) if accepted.size else float("nan")
        if nonfinite_count:
            outcome = "rejected_nonfinite_sample"
            vetoes = ("nonfinite_sample",)
        elif not log_accept_finite:
            outcome = "rejected_nonfinite_log_accept_ratio"
            vetoes = ("nonfinite_log_accept_ratio",)
        elif not np.isfinite(acceptance_rate):
            outcome = "blocked_missing_acceptance_diagnostic"
            vetoes = ("acceptance_rate_nonfinite",)
        elif acceptance_rate < lower:
            outcome = "rejected_accept_low"
            vetoes = ("acceptance_below_closed_promotion_band",)
        elif acceptance_rate > upper:
            outcome = "rejected_accept_high"
            vetoes = ("acceptance_above_closed_promotion_band",)
        else:
            outcome = "passed_screen"
            vetoes = ()
    except Exception as exc:  # pragma: no cover - TFP runtime fault path.
        acceptance_rate = None
        log_accept_finite = False
        finite_count = 0
        nonfinite_count = 0
        outcome = "blocked_hmc_execution_error"
        vetoes = (f"hmc_execution_error:{exc.__class__.__name__}",)
    return FixedTrajectoryCandidateResult(
        step_size=float(step_size),
        num_leapfrog_steps=int(num_leapfrog_steps),
        trajectory_length=float(step_size) * int(num_leapfrog_steps),
        acceptance_rate=acceptance_rate,
        log_accept_ratio_finite=log_accept_finite,
        finite_sample_count=finite_count,
        nonfinite_sample_count=nonfinite_count,
        outcome=outcome,
        grid_index=int(grid_index),
        vetoes=tuple(vetoes),
    )


def _select_fixed_trajectory_candidate(
    candidates: tuple[FixedTrajectoryCandidateResult, ...],
    *,
    midpoint: float,
) -> FixedTrajectoryCandidateResult | None:
    passing = [
        candidate for candidate in candidates if candidate.outcome == "passed_screen"
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda candidate: (
            abs(float(candidate.acceptance_rate) - float(midpoint)),
            candidate.trajectory_length,
            candidate.grid_index,
        ),
    )


def _rebuild_windowed_mass_artifact(
    *,
    base_mass_artifact: Any,
    covariance: Any,
    config: WindowedMassAdaptationConfig,
    window: WindowedWarmupWindow,
) -> Any:
    from bayesfilter.inference.hmc import PrecomputedMassArtifact

    regularized_covariance, regularization_report = _regularize_windowed_covariance(
        covariance,
        jitter=config.covariance_jitter,
        eigenvalue_floor=config.eigenvalue_floor,
        max_condition_number=config.max_condition_number,
    )
    return PrecomputedMassArtifact.from_covariance(
        position=np.asarray(base_mass_artifact.position, dtype=float),
        covariance=regularized_covariance,
        adapter_signature=base_mass_artifact.adapter_signature,
        position_role=base_mass_artifact.position_role,
        covariance_source=f"windowed_welford_shrinkage_window_{window.index}",
        matrix_used_for_square_root="windowed_shrunk_covariance",
        source="windowed_mass_adaptation",
        jitter=0.0,
        regularization_report={
            **regularization_report,
            "method": "welford_covariance_shrinkage",
            "window_index": window.index,
            "window_start": window.start,
            "window_end": window.end,
            "mass_shrinkage": config.mass_shrinkage,
            "dual_averaging_reset_required": True,
            "silent_eigenvalue_reflection": False,
        },
    )


def _regularize_windowed_covariance(
    covariance: Any,
    *,
    jitter: float,
    eigenvalue_floor: float | None,
    max_condition_number: float | None,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("windowed covariance must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("windowed covariance must be finite")
    jitter_value = float(jitter)
    if not np.isfinite(jitter_value) or jitter_value < 0.0:
        raise ValueError("covariance jitter must be finite and non-negative")
    floor = 0.0 if eigenvalue_floor is None else float(eigenvalue_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("covariance eigenvalue_floor must be finite and non-negative")
    max_condition = (
        None if max_condition_number is None else float(max_condition_number)
    )
    if max_condition is not None and (
        not np.isfinite(max_condition) or max_condition <= 1.0
    ):
        raise ValueError("covariance max_condition_number must be finite and greater than 1")
    symmetric = 0.5 * (matrix + matrix.T)
    jittered = symmetric + jitter_value * np.eye(symmetric.shape[0])
    raw_eigvals, eigvecs = np.linalg.eigh(jittered)
    if not np.all(np.isfinite(raw_eigvals)):
        raise ValueError("windowed covariance eigenvalues must be finite")
    positive_raw = raw_eigvals[raw_eigvals > 0.0]
    if floor == 0.0:
        if positive_raw.size == 0:
            raise ValueError(
                "windowed covariance must have a positive eigenvalue; "
                "pass eigenvalue_floor"
            )
        floor = max(floor, np.finfo(float).eps * max(1.0, float(np.max(positive_raw))))
    if max_condition is not None:
        floor = max(floor, float(np.max(raw_eigvals)) / max_condition)
    if floor <= 0.0:
        floor = np.finfo(float).eps
    regularized_eigvals = np.maximum(raw_eigvals, floor)
    regularized = (eigvecs * regularized_eigvals) @ eigvecs.T
    regularized = 0.5 * (regularized + regularized.T)
    return regularized, {
        "covariance_regularization_method": "symmetric_eigendecomposition_floor",
        "covariance_jitter": jitter_value,
        "requested_covariance_eigenvalue_floor": (
            None if eigenvalue_floor is None else float(eigenvalue_floor)
        ),
        "effective_covariance_eigenvalue_floor": float(floor),
        "covariance_max_condition_number": max_condition,
        "raw_covariance_min_eigenvalue": float(np.min(raw_eigvals)),
        "raw_covariance_max_eigenvalue": float(np.max(raw_eigvals)),
        "regularized_covariance_min_eigenvalue": float(np.min(regularized_eigvals)),
        "regularized_covariance_max_eigenvalue": float(np.max(regularized_eigvals)),
        "covariance_clipped_eigenvalue_count": int(
            np.sum(regularized_eigvals > raw_eigvals)
        ),
    }


def _windows_are_contiguous(windows: tuple[WindowedWarmupWindow, ...] | list[WindowedWarmupWindow]) -> bool:
    if not windows:
        return False
    if windows[0].start != 0:
        return False
    return all(left.end == right.start for left, right in zip(windows, windows[1:]))


def _frozen_mass_invariant(before: str, after: str) -> Mapping[str, Any]:
    return {
        "passed": before == after,
        "before_signature": before,
        "after_signature": after,
        "signature_includes_arrays": True,
        "mass_update_allowed": False,
        "role": "hard_veto",
    }


def _failed_fixed_mass_diagnostic(
    policy: HMCTuningPolicy,
    *,
    classification: str,
    error: Exception | None = None,
) -> HMCTuningDiagnosticResult:
    diagnostics: dict[str, Any] = {
        "policy_label": policy.label,
        "adaptation_policy": policy.adaptation_policy,
        "classification": classification,
        "num_adaptation_steps": policy.num_adaptation_steps,
        "target_accept_prob": policy.target_accept_prob,
        "initial_step_size": None,
        "final_step_size": None,
        "final_step_size_finite": False,
        "acceptance_rate": None,
        "log_accept_ratio_finite": False,
        "finite_sample_count": 0,
        "nonfinite_sample_count": None,
        "reports_posterior_convergence": False,
    }
    if error is not None:
        diagnostics["error_type"] = error.__class__.__name__
        diagnostics["error_message"] = str(error)
    return HMCTuningDiagnosticResult(
        policy=policy,
        diagnostics=diagnostics,
        trace={
            "is_accepted": [],
            "log_accept_ratio": [],
            "step_size": [],
        },
        metadata={
            "runtime": "not_run",
            "diagnostic_role": "hard_veto",
            "nonclaims": policy.nonclaims,
        },
    )


def _validate_target_failure_classification(
    classification: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if classification is None:
        return None
    result = dict(classification)
    required = ("classification", "diagnostic_role", "nonclaims")
    missing = tuple(key for key in required if key not in result)
    if missing:
        raise ValueError(
            "target_failure_classification missing required fields: "
            + ", ".join(missing)
        )
    class_label = str(result["classification"])
    allowed_classes = {
        "tuning_diagnostic_passed_not_convergence",
        "target_invalidity_not_tuning_success",
        "nonfinite_tuning_diagnostic",
        "hmc_execution_error",
    }
    if class_label not in allowed_classes:
        raise ValueError(
            "target_failure_classification has unsupported classification: "
            f"{class_label!r}"
        )
    role = str(result["diagnostic_role"])
    if role not in {"diagnostic_only", "hard_veto"}:
        raise ValueError(
            "target_failure_classification diagnostic_role must be "
            "'diagnostic_only' or 'hard_veto'"
        )
    nonclaims = tuple(str(item) for item in result["nonclaims"])
    if not nonclaims:
        raise ValueError("target_failure_classification nonclaims must be non-empty")
    result["classification"] = class_label
    result["diagnostic_role"] = role
    result["nonclaims"] = nonclaims
    return result


def _stable_payload_signature(payload: Mapping[str, Any]) -> str:
    import hashlib
    import json

    blob = json.dumps(_normalize_payload(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _normalize_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_payload(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _normalize_payload(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
