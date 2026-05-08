"""Linear Gaussian filtering contracts and backends."""

from bayesfilter.linear.kalman_tf import (
    TFLinearValueBackend,
    tf_kalman_filter,
    tf_kalman_log_likelihood,
    tf_linear_gaussian_log_likelihood,
    tf_masked_kalman_filter,
    tf_masked_kalman_log_likelihood,
)
from bayesfilter.linear.kalman_qr_tf import (
    TFQRLinearValueBackend,
    tf_qr_linear_gaussian_log_likelihood,
    tf_qr_sqrt_kalman_filter,
    tf_qr_sqrt_kalman_log_likelihood,
    tf_qr_sqrt_masked_kalman_filter,
    tf_qr_sqrt_masked_kalman_log_likelihood,
)
from bayesfilter.linear.kalman_qr_derivatives_tf import (
    TFQRLinearDerivativeBackend,
    tf_qr_linear_gaussian_score_hessian,
    tf_qr_sqrt_kalman_score_hessian,
)
from bayesfilter.linear.qr_factor_tf import (
    cholesky_factor,
    cholesky_factor_derivatives,
    factor_covariance_derivatives,
    factor_derivative_reconstruction_errors,
    factor_solve,
    lower_factor_from_horizontal_stack,
    qr_factor_derivatives,
    qr_factor_full_derivatives,
    qr_factor_second_derivatives,
    qr_positive,
    stack_covariance_derivatives,
    stack_qr_lower_factor_derivatives,
    symmetrize,
    trace_factor_solve,
)
from bayesfilter.linear.types import (
    LinearGaussianStateSpace,
    LinearGaussianStateSpaceDerivatives,
)
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)

__all__ = [
    "LinearGaussianStateSpace",
    "LinearGaussianStateSpaceDerivatives",
    "TFQRLinearValueBackend",
    "TFQRLinearDerivativeBackend",
    "TFLinearValueBackend",
    "tf_qr_linear_gaussian_log_likelihood",
    "tf_qr_linear_gaussian_score_hessian",
    "tf_qr_sqrt_kalman_filter",
    "tf_qr_sqrt_kalman_log_likelihood",
    "tf_qr_sqrt_kalman_score_hessian",
    "tf_qr_sqrt_masked_kalman_filter",
    "tf_qr_sqrt_masked_kalman_log_likelihood",
    "tf_kalman_filter",
    "tf_kalman_log_likelihood",
    "tf_linear_gaussian_log_likelihood",
    "tf_masked_kalman_filter",
    "tf_masked_kalman_log_likelihood",
    "cholesky_factor",
    "cholesky_factor_derivatives",
    "factor_covariance_derivatives",
    "factor_derivative_reconstruction_errors",
    "factor_solve",
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
