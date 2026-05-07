"""BayesFilter core contracts and reference backends."""

from bayesfilter.backends import (
    FactorBackendAuditResult,
    SpectralDerivativeCertificationResult,
    audit_factor_backend,
    certify_spectral_derivative_region,
)
from bayesfilter.linear import (
    LinearGaussianStateSpace,
    LinearGaussianStateSpaceDerivatives,
)
from bayesfilter.linear.kalman_derivatives_numpy import solve_kalman_score_hessian
from bayesfilter.results import FilterDerivativeResult, FilterValueResult
from bayesfilter.structural import (
    FilterRunMetadata,
    StatePartition,
    StructuralFilterConfig,
    validate_filter_config,
)

__all__ = [
    "FactorBackendAuditResult",
    "FilterDerivativeResult",
    "FilterRunMetadata",
    "FilterValueResult",
    "LinearGaussianStateSpace",
    "LinearGaussianStateSpaceDerivatives",
    "SpectralDerivativeCertificationResult",
    "StatePartition",
    "StructuralFilterConfig",
    "audit_factor_backend",
    "certify_spectral_derivative_region",
    "solve_kalman_score_hessian",
    "validate_filter_config",
]
