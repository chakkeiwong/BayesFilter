"""BayesFilter core contracts and reference backends."""

from bayesfilter.backends import (
    FactorBackendAuditResult,
    SpectralDerivativeCertificationResult,
    audit_factor_backend,
    certify_spectral_derivative_region,
)
from bayesfilter.structural import (
    FilterRunMetadata,
    StatePartition,
    StructuralFilterConfig,
    validate_filter_config,
)

__all__ = [
    "FactorBackendAuditResult",
    "FilterRunMetadata",
    "SpectralDerivativeCertificationResult",
    "StatePartition",
    "StructuralFilterConfig",
    "audit_factor_backend",
    "certify_spectral_derivative_region",
    "validate_filter_config",
]
