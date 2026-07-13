"""TensorFlow nonlinear filtering backends and sigma-point rules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CompiledValuePathClassification",
    "CompiledValuePathMode",
    "InvalidCompiledValuePathContract",
    "NonlinearFilterValuePathContract",
    "NonlinearFilterValueStaticShape",
    "TFFixedSGQF1DLevelRule",
    "TFFixedSGQFAffineModel",
    "TFFixedSGQFBranchConfig",
    "TFFixedSGQFBranchHash",
    "TFFixedSGQFBranchIdentity",
    "TFFixedSGQFBranchManifest",
    "TFFixedSGQFCloud",
    "TFFixedSGQFDerivatives",
    "TFFixedSGQFNonlinearModel",
    "TFFixedSGQFOneStepOracle",
    "TFFixedSGQFScoreResult",
    "TFFixedSGQFStepFailure",
    "TFFixedSGQFStepResult",
    "TFFixedSGQFValueResult",
    "TFStructuralFirstDerivatives",
    "TFSmoothEighFactorFirstDerivatives",
    "TFSigmaPointDiagnostics",
    "TFSigmaPointRule",
    "TFSigmaPointValueBackend",
    "TFSRUKFSigmaPointRule",
    "TFSRUKFStepDerivatives",
    "TFSRUKFStepResult",
    "SRUKFRouteGuardViolation",
    "SSLLSTMParameterMask",
    "SSLLSTMForecastConfig",
    "SSLLSTMForecastPaths",
    "SSLLSTMForecastProvenance",
    "SSLLSTMInnovationBank",
    "SSLLSTMPosteriorConfig",
    "SSLLSTMPosteriorTarget",
    "SSLLSTMTerminalState",
    "assert_no_forbidden_srukf_routes",
    "find_forbidden_compiled_value_tokens",
    "find_forbidden_srukf_routes",
    "locked_ssl_lstm_posterior_target",
    "stable_nonlinear_filter_value_path_signature",
    "tf_fixed_sgqf_active_multi_indices",
    "tf_fixed_sgqf_branch_identity",
    "tf_fixed_sgqf_cloud",
    "tf_fixed_sgqf_combination_coefficient",
    "tf_fixed_sgqf_filter",
    "tf_fixed_sgqf_p47_one_step_oracle",
    "tf_fixed_sgqf_same_branch_signature",
    "tf_fixed_sgqf_score",
    "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
    "tf_svd_sigma_point_filter",
    "tf_svd_sigma_point_log_likelihood",
    "tf_svd_sigma_point_log_likelihood_with_rule",
    "tf_svd_sigma_point_placement",
    "tf_cut4g_sigma_point_rule",
    "tf_svd_cubature_score",
    "tf_svd_cut4_filter",
    "tf_svd_cut4_log_likelihood",
    "tf_svd_cut4_score",
    "tf_srukf_factor_score_step",
    "tf_srukf_unit_sigma_point_rule",
    "tf_unit_sigma_point_rule",
    "tf_svd_sigma_point_score_with_rule",
    "tf_svd_ukf_score",
    "tf_principal_sqrt_ukf_score",
    "tf_standard_normal_ghq_level_rule",
    "tensorflow_nonlinear_value_path_contract",
]

_EXPORT_MODULES = {
    "CompiledValuePathClassification": "bayesfilter.nonlinear.compiled_value_paths",
    "CompiledValuePathMode": "bayesfilter.nonlinear.compiled_value_paths",
    "InvalidCompiledValuePathContract": "bayesfilter.nonlinear.compiled_value_paths",
    "NonlinearFilterValuePathContract": "bayesfilter.nonlinear.compiled_value_paths",
    "NonlinearFilterValueStaticShape": "bayesfilter.nonlinear.compiled_value_paths",
    "TFFixedSGQF1DLevelRule": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFAffineModel": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFBranchConfig": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFBranchHash": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFBranchIdentity": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFBranchManifest": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFCloud": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFDerivatives": "bayesfilter.nonlinear.fixed_sgqf_derivatives_tf",
    "TFFixedSGQFNonlinearModel": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFOneStepOracle": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFScoreResult": "bayesfilter.nonlinear.fixed_sgqf_derivatives_tf",
    "TFFixedSGQFStepFailure": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFStepResult": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFFixedSGQFValueResult": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "TFStructuralFirstDerivatives": (
        "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf"
    ),
    "TFSmoothEighFactorFirstDerivatives": (
        "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf"
    ),
    "TFSigmaPointDiagnostics": "bayesfilter.nonlinear.sigma_points_tf",
    "TFSigmaPointRule": "bayesfilter.nonlinear.sigma_points_tf",
    "TFSigmaPointValueBackend": "bayesfilter.nonlinear.sigma_points_tf",
    "TFSRUKFSigmaPointRule": "bayesfilter.nonlinear.srukf_factor_tf",
    "TFSRUKFStepDerivatives": "bayesfilter.nonlinear.srukf_factor_tf",
    "TFSRUKFStepResult": "bayesfilter.nonlinear.srukf_factor_tf",
    "SRUKFRouteGuardViolation": "bayesfilter.nonlinear.srukf_route_guard",
    "SSLLSTMParameterMask": "bayesfilter.nonlinear.ssl_lstm_posterior_tf",
    "SSLLSTMForecastConfig": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",
    "SSLLSTMForecastPaths": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",
    "SSLLSTMForecastProvenance": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",
    "SSLLSTMInnovationBank": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",
    "SSLLSTMPosteriorConfig": "bayesfilter.nonlinear.ssl_lstm_posterior_tf",
    "SSLLSTMPosteriorTarget": "bayesfilter.nonlinear.ssl_lstm_posterior_tf",
    "SSLLSTMTerminalState": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",
    "assert_no_forbidden_srukf_routes": "bayesfilter.nonlinear.srukf_route_guard",
    "find_forbidden_compiled_value_tokens": "bayesfilter.nonlinear.compiled_value_paths",
    "find_forbidden_srukf_routes": "bayesfilter.nonlinear.srukf_route_guard",
    "locked_ssl_lstm_posterior_target": (
        "bayesfilter.nonlinear.ssl_lstm_posterior_tf"
    ),
    "stable_nonlinear_filter_value_path_signature": (
        "bayesfilter.nonlinear.compiled_value_paths"
    ),
    "tf_fixed_sgqf_active_multi_indices": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_branch_identity": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_cloud": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_combination_coefficient": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_filter": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_p47_one_step_oracle": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tf_fixed_sgqf_same_branch_signature": (
        "bayesfilter.nonlinear.fixed_sgqf_derivatives_tf"
    ),
    "tf_fixed_sgqf_score": "bayesfilter.nonlinear.fixed_sgqf_derivatives_tf",
    "tf_batched_svd_sigma_point_value_and_score_custom_gradient": (
        "bayesfilter.nonlinear.batched_svd_sigma_point_tf"
    ),
    "tf_svd_sigma_point_filter": "bayesfilter.nonlinear.sigma_points_tf",
    "tf_svd_sigma_point_log_likelihood": "bayesfilter.nonlinear.sigma_points_tf",
    "tf_svd_sigma_point_log_likelihood_with_rule": (
        "bayesfilter.nonlinear.sigma_points_tf"
    ),
    "tf_svd_sigma_point_placement": "bayesfilter.nonlinear.sigma_points_tf",
    "tf_cut4g_sigma_point_rule": "bayesfilter.nonlinear.cut_tf",
    "tf_svd_cubature_score": (
        "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf"
    ),
    "tf_svd_cut4_filter": "bayesfilter.nonlinear.svd_cut_tf",
    "tf_svd_cut4_log_likelihood": "bayesfilter.nonlinear.svd_cut_tf",
    "tf_svd_cut4_score": "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf",
    "tf_srukf_factor_score_step": "bayesfilter.nonlinear.srukf_factor_tf",
    "tf_srukf_unit_sigma_point_rule": "bayesfilter.nonlinear.srukf_factor_tf",
    "tf_unit_sigma_point_rule": "bayesfilter.nonlinear.sigma_points_tf",
    "tf_svd_sigma_point_score_with_rule": (
        "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf"
    ),
    "tf_svd_ukf_score": "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf",
    "tf_principal_sqrt_ukf_score": "bayesfilter.nonlinear.svd_sigma_point_derivatives_tf",
    "tf_standard_normal_ghq_level_rule": "bayesfilter.nonlinear.fixed_sgqf_tf",
    "tensorflow_nonlinear_value_path_contract": (
        "bayesfilter.nonlinear.compiled_value_paths"
    ),
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
