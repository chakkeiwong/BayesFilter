"""Linear Gaussian filtering contracts and backends."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "LinearGaussianStateSpace",
    "LinearGaussianStateSpaceDerivatives",
    "TFQRLinearValueBackend",
    "TFQRLinearDerivativeBackend",
    "TFLinearValueBackend",
    "TFSVDLinearValueBackend",
    "TFLinearGaussianStateSpaceFirstDerivatives",
    "SVD_LINEAR_SCORE_STATUS_BLOCKED_ACTIVE_FLOOR",
    "SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT",
    "SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED",
    "StationaryLGSSMFirstDerivativeCoverage",
    "continuous_lyapunov_first_derivatives_tf",
    "continuous_lyapunov_solution_tf",
    "diffusion_from_cholesky_first_derivatives_tf",
    "first_to_full_linear_gaussian_derivatives",
    "matrix_exponential_frechet_tf",
    "stationary_lgssm_first_derivative_coverage",
    "stationary_lgssm_from_continuous_first_derivatives_tf",
    "tf_qr_linear_gaussian_log_likelihood",
    "tf_qr_linear_gaussian_score",
    "tf_qr_linear_gaussian_score_hessian",
    "tf_qr_sqrt_factorized_kalman_log_likelihood",
    "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments",
    "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments_graph",
    "tf_qr_sqrt_kalman_filter",
    "tf_qr_sqrt_kalman_log_likelihood",
    "tf_qr_sqrt_masked_kalman_log_likelihood_batched_static",
    "tf_qr_sqrt_kalman_score_hessian",
    "tf_qr_sqrt_masked_kalman_filter",
    "tf_qr_sqrt_masked_kalman_log_likelihood",
    "tf_qr_sqrt_masked_kalman_score_hessian",
    "tf_kalman_filter",
    "tf_kalman_log_likelihood",
    "tf_correlated_kalman_filter",
    "tf_correlated_kalman_log_likelihood",
    "tf_correlated_kalman_filter_batched_time_varying",
    "tf_correlated_kalman_log_likelihood_batched_time_varying",
    "tf_linear_gaussian_log_likelihood",
    "tf_masked_correlated_kalman_filter",
    "tf_masked_correlated_kalman_log_likelihood",
    "tf_masked_correlated_kalman_filter_batched_time_varying",
    "tf_masked_correlated_kalman_log_likelihood_batched_time_varying",
    "tf_masked_kalman_filter",
    "tf_masked_kalman_log_likelihood",
    "tf_svd_kalman_log_likelihood",
    "tf_svd_linear_gaussian_score_first_order",
    "tf_svd_linear_gaussian_score_first_order_graph_status",
    "tf_svd_linear_gaussian_score_hessian",
    "tf_svd_linear_gaussian_score_hessian_graph_status",
    "tf_svd_linear_gaussian_log_likelihood",
    "tf_svd_masked_kalman_log_likelihood",
    "cholesky_factor",
    "cholesky_factor_derivatives",
    "factor_covariance_derivatives",
    "factor_derivative_reconstruction_errors",
    "factor_solve",
    "lower_factor_from_horizontal_stack",
    "PrincipalSqrtFirstDerivativeDiagnostics",
    "principal_sqrt_frechet_derivative_from_eigh",
    "strict_spd_principal_sqrt_first_derivatives",
    "lower_factor_from_horizontal_stack",
    "qr_factor_derivatives",
    "qr_factor_full_derivatives",
    "qr_factor_second_derivatives",
    "qr_positive",
    "stack_covariance_derivatives",
    "stack_qr_lower_factor_derivatives",
    "symmetrize",
    "trace_factor_solve",
    "TFLinearGaussianStateSpace",
    "TFLinearGaussianStateSpaceDerivatives",
]

_EXPORT_MODULES = {
    "LinearGaussianStateSpace": "bayesfilter.linear.types",
    "LinearGaussianStateSpaceDerivatives": "bayesfilter.linear.types",
    "TFQRLinearValueBackend": "bayesfilter.linear.kalman_qr_tf",
    "TFQRLinearDerivativeBackend": "bayesfilter.linear.kalman_qr_derivatives_tf",
    "TFLinearValueBackend": "bayesfilter.linear.kalman_tf",
    "TFSVDLinearValueBackend": "bayesfilter.linear.kalman_svd_tf",
    "TFLinearGaussianStateSpaceFirstDerivatives": "bayesfilter.linear.types_tf",
    "SVD_LINEAR_SCORE_STATUS_BLOCKED_ACTIVE_FLOOR": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "SVD_LINEAR_SCORE_STATUS_INVALID_EIGENSOLVER_INPUT": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "SVD_LINEAR_SCORE_STATUS_VALID_PRE_REGULARIZED": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "StationaryLGSSMFirstDerivativeCoverage": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "continuous_lyapunov_first_derivatives_tf": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "continuous_lyapunov_solution_tf": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "diffusion_from_cholesky_first_derivatives_tf": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "first_to_full_linear_gaussian_derivatives": "bayesfilter.linear.types_tf",
    "matrix_exponential_frechet_tf": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "stationary_lgssm_first_derivative_coverage": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "stationary_lgssm_from_continuous_first_derivatives_tf": (
        "bayesfilter.linear.stationary_lgssm_derivatives_tf"
    ),
    "tf_qr_linear_gaussian_log_likelihood": "bayesfilter.linear.kalman_qr_tf",
    "tf_qr_linear_gaussian_score": (
        "bayesfilter.linear.kalman_qr_derivatives_tf"
    ),
    "tf_qr_linear_gaussian_score_hessian": (
        "bayesfilter.linear.kalman_qr_derivatives_tf"
    ),
    "tf_qr_sqrt_factorized_kalman_log_likelihood": (
        "bayesfilter.linear.kalman_qr_tf"
    ),
    "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments": (
        "bayesfilter.linear.kalman_qr_tf"
    ),
    "tf_qr_sqrt_factorized_kalman_log_likelihood_with_increments_graph": (
        "bayesfilter.linear.kalman_qr_tf"
    ),
    "tf_qr_sqrt_kalman_filter": "bayesfilter.linear.kalman_qr_tf",
    "tf_qr_sqrt_kalman_log_likelihood": "bayesfilter.linear.kalman_qr_tf",
    "tf_qr_sqrt_masked_kalman_log_likelihood_batched_static": (
        "bayesfilter.linear.kalman_qr_tf"
    ),
    "tf_qr_sqrt_kalman_score_hessian": (
        "bayesfilter.linear.kalman_qr_derivatives_tf"
    ),
    "tf_qr_sqrt_masked_kalman_filter": "bayesfilter.linear.kalman_qr_tf",
    "tf_qr_sqrt_masked_kalman_log_likelihood": (
        "bayesfilter.linear.kalman_qr_tf"
    ),
    "tf_qr_sqrt_masked_kalman_score_hessian": (
        "bayesfilter.linear.kalman_qr_derivatives_tf"
    ),
    "tf_kalman_filter": "bayesfilter.linear.kalman_tf",
    "tf_kalman_log_likelihood": "bayesfilter.linear.kalman_tf",
    "tf_correlated_kalman_filter": "bayesfilter.linear.correlated_kalman_tf",
    "tf_correlated_kalman_log_likelihood": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_correlated_kalman_filter_batched_time_varying": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_correlated_kalman_log_likelihood_batched_time_varying": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_linear_gaussian_log_likelihood": "bayesfilter.linear.kalman_tf",
    "tf_masked_correlated_kalman_filter": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_masked_correlated_kalman_log_likelihood": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_masked_correlated_kalman_filter_batched_time_varying": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_masked_correlated_kalman_log_likelihood_batched_time_varying": (
        "bayesfilter.linear.correlated_kalman_tf"
    ),
    "tf_masked_kalman_filter": "bayesfilter.linear.kalman_tf",
    "tf_masked_kalman_log_likelihood": "bayesfilter.linear.kalman_tf",
    "tf_svd_kalman_log_likelihood": "bayesfilter.linear.kalman_svd_tf",
    "tf_svd_linear_gaussian_score_first_order": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "tf_svd_linear_gaussian_score_first_order_graph_status": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "tf_svd_linear_gaussian_score_hessian": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "tf_svd_linear_gaussian_score_hessian_graph_status": (
        "bayesfilter.linear.kalman_svd_derivatives_tf"
    ),
    "tf_svd_linear_gaussian_log_likelihood": "bayesfilter.linear.kalman_svd_tf",
    "tf_svd_masked_kalman_log_likelihood": "bayesfilter.linear.kalman_svd_tf",
    "PrincipalSqrtFirstDerivativeDiagnostics": "bayesfilter.linear.svd_factor_tf",
    "principal_sqrt_frechet_derivative_from_eigh": "bayesfilter.linear.svd_factor_tf",
    "strict_spd_principal_sqrt_first_derivatives": "bayesfilter.linear.svd_factor_tf",
    "cholesky_factor": "bayesfilter.linear.qr_factor_tf",
    "cholesky_factor_derivatives": "bayesfilter.linear.qr_factor_tf",
    "factor_covariance_derivatives": "bayesfilter.linear.qr_factor_tf",
    "factor_derivative_reconstruction_errors": "bayesfilter.linear.qr_factor_tf",
    "factor_solve": "bayesfilter.linear.qr_factor_tf",
    "lower_factor_from_horizontal_stack": "bayesfilter.linear.qr_factor_tf",
    "qr_factor_derivatives": "bayesfilter.linear.qr_factor_tf",
    "qr_factor_full_derivatives": "bayesfilter.linear.qr_factor_tf",
    "qr_factor_second_derivatives": "bayesfilter.linear.qr_factor_tf",
    "qr_positive": "bayesfilter.linear.qr_factor_tf",
    "stack_covariance_derivatives": "bayesfilter.linear.qr_factor_tf",
    "stack_qr_lower_factor_derivatives": "bayesfilter.linear.qr_factor_tf",
    "symmetrize": "bayesfilter.linear.qr_factor_tf",
    "trace_factor_solve": "bayesfilter.linear.qr_factor_tf",
    "TFLinearGaussianStateSpace": "bayesfilter.linear.types_tf",
    "TFLinearGaussianStateSpaceDerivatives": "bayesfilter.linear.types_tf",
}


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
