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
    TFLinearValueBackend,
    TFQRLinearDerivativeBackend,
    TFQRLinearValueBackend,
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
    tf_kalman_filter,
    tf_kalman_log_likelihood,
    tf_linear_gaussian_log_likelihood,
    tf_masked_kalman_filter,
    tf_masked_kalman_log_likelihood,
    tf_qr_linear_gaussian_log_likelihood,
    tf_qr_linear_gaussian_score_hessian,
    tf_qr_sqrt_kalman_filter,
    tf_qr_sqrt_kalman_log_likelihood,
    tf_qr_sqrt_kalman_score_hessian,
    tf_qr_sqrt_masked_kalman_filter,
    tf_qr_sqrt_masked_kalman_log_likelihood,
)
from bayesfilter.linear.kalman_derivatives_numpy import solve_kalman_score_hessian
from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.results import FilterDerivativeResult, FilterValueResult
from bayesfilter.results_tf import TFFilterDerivativeResult, TFFilterValueResult
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
    "TFFilterDerivativeResult",
    "TFFilterDiagnostics",
    "TFFilterValueResult",
    "TFLinearGaussianStateSpace",
    "TFLinearGaussianStateSpaceDerivatives",
    "TFLinearValueBackend",
    "TFQRLinearDerivativeBackend",
    "TFQRLinearValueBackend",
    "TFRegularizationDiagnostics",
    "StructuralFilterConfig",
    "audit_factor_backend",
    "certify_spectral_derivative_region",
    "solve_kalman_score_hessian",
    "tf_kalman_filter",
    "tf_kalman_log_likelihood",
    "tf_linear_gaussian_log_likelihood",
    "tf_masked_kalman_filter",
    "tf_masked_kalman_log_likelihood",
    "tf_qr_linear_gaussian_log_likelihood",
    "tf_qr_linear_gaussian_score_hessian",
    "tf_qr_sqrt_kalman_filter",
    "tf_qr_sqrt_kalman_log_likelihood",
    "tf_qr_sqrt_kalman_score_hessian",
    "tf_qr_sqrt_masked_kalman_filter",
    "tf_qr_sqrt_masked_kalman_log_likelihood",
    "validate_filter_config",
]
