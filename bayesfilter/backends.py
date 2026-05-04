"""Backend and derivative readiness metadata gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FactorBackendAuditResult:
    """Classification for a value or derivative factor backend."""

    backend_name: str
    value_status: str
    derivative_status: str
    hmc_status: str
    compiled_status: str
    blockers: tuple[str, ...]
    approximation_label: str | None = None
    min_spectral_gap: float | None = None
    source: str = "factor_backend_audit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_name", str(self.backend_name))
        object.__setattr__(self, "value_status", str(self.value_status))
        object.__setattr__(self, "derivative_status", str(self.derivative_status))
        object.__setattr__(self, "hmc_status", str(self.hmc_status))
        object.__setattr__(self, "compiled_status", str(self.compiled_status))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        if self.approximation_label is not None:
            object.__setattr__(self, "approximation_label", str(self.approximation_label))
        if self.min_spectral_gap is not None:
            object.__setattr__(self, "min_spectral_gap", float(self.min_spectral_gap))


@dataclass(frozen=True)
class SpectralDerivativeCertificationResult:
    """Numerical gate for spectral derivative promotion."""

    object_name: str
    singular_or_eigen_values: np.ndarray
    min_gap: float
    gap_tolerance: float
    derivative_certified: bool
    hmc_eligible: bool
    blockers: tuple[str, ...]
    warning_label: str | None = None
    source: str = "spectral_derivative_certification"

    def __post_init__(self) -> None:
        values = np.asarray(self.singular_or_eigen_values, dtype=float).copy()
        values.setflags(write=False)
        object.__setattr__(self, "object_name", str(self.object_name))
        object.__setattr__(self, "singular_or_eigen_values", values)
        object.__setattr__(self, "min_gap", float(self.min_gap))
        object.__setattr__(self, "gap_tolerance", float(self.gap_tolerance))
        object.__setattr__(self, "derivative_certified", bool(self.derivative_certified))
        object.__setattr__(self, "hmc_eligible", bool(self.hmc_eligible))
        object.__setattr__(self, "blockers", tuple(str(blocker) for blocker in self.blockers))
        if self.warning_label is not None:
            object.__setattr__(self, "warning_label", str(self.warning_label))


def audit_factor_backend(
    backend_name: str,
    *,
    value_exact: bool,
    derivative_checked: bool = False,
    compiled_supported: bool = False,
    approximation_label: str | None = None,
    min_spectral_gap: float | None = None,
) -> FactorBackendAuditResult:
    """Classify a backend without promoting unsupported derivative/HMC claims."""

    blockers: list[str] = []
    if not value_exact and approximation_label is None:
        blockers.append("non-exact backend requires an approximation label")
    if derivative_checked and min_spectral_gap is not None and min_spectral_gap <= 0.0:
        blockers.append("derivative-checked spectral backend reports nonpositive gap")
    value_status = "exact_value" if value_exact else "approximation_only"
    derivative_status = "derivative_checked" if derivative_checked else "not_certified"
    compiled_status = "compiled_supported" if compiled_supported else "eager_only"
    hmc_status = (
        "target_candidate"
        if value_exact and derivative_checked and compiled_supported and not blockers
        else "blocked"
    )
    return FactorBackendAuditResult(
        backend_name=backend_name,
        value_status=value_status,
        derivative_status=derivative_status,
        hmc_status=hmc_status,
        compiled_status=compiled_status,
        blockers=tuple(blockers),
        approximation_label=approximation_label,
        min_spectral_gap=min_spectral_gap,
    )


def certify_spectral_derivative_region(
    values: Any,
    *,
    object_name: str = "spectral_factor",
    gap_tolerance: float = 1e-6,
    finite_difference_checked: bool = False,
    jvp_vjp_checked: bool = False,
) -> SpectralDerivativeCertificationResult:
    """Gate spectral derivative claims on gap telemetry and numerical evidence."""

    spectral_values = np.sort(np.asarray(values, dtype=float).ravel())
    if spectral_values.size == 0:
        raise ValueError("spectral values must not be empty")
    if spectral_values.size == 1:
        min_gap = float("inf")
    else:
        min_gap = float(np.min(np.diff(spectral_values)))
    blockers: list[str] = []
    warning_label: str | None = None
    if not np.all(np.isfinite(spectral_values)):
        blockers.append("spectral values are nonfinite")
    if min_gap <= float(gap_tolerance):
        blockers.append(
            f"minimum spectral gap {min_gap:.3e} is at or below tolerance {float(gap_tolerance):.3e}"
        )
        warning_label = "spectral_gap_too_small"
    if not finite_difference_checked:
        blockers.append("finite-difference derivative check missing")
    if not jvp_vjp_checked:
        blockers.append("JVP/VJP parity check missing")
    certified = not blockers
    return SpectralDerivativeCertificationResult(
        object_name=object_name,
        singular_or_eigen_values=spectral_values,
        min_gap=min_gap,
        gap_tolerance=float(gap_tolerance),
        derivative_certified=certified,
        hmc_eligible=certified,
        blockers=tuple(blockers),
        warning_label=warning_label,
    )
