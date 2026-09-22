"""Testing fixtures with lazy public exports.

The package must not import unrelated diagnostic/reference modules merely to
load a TensorFlow-native target or training harness.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DenseProjectionStep": "bayesfilter.testing.nonlinear_models_tf",
    "FixedSGQFBranchSummary": "bayesfilter.testing.fixed_sgqf_diagnostics_tf",
    "FixedSGQFDiagnosticSnapshot": "bayesfilter.testing.fixed_sgqf_diagnostics_tf",
    "NonlinearSigmaPointBranchSummary": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "NonlinearSigmaPointDiagnosticSnapshot": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "ModelBNonlinearSVDTarget": "bayesfilter.testing.tf_hmc_readiness",
    "QRStaticLGSSMTarget": "bayesfilter.testing.tf_hmc_readiness",
    "SVDCUTBranchSummary": "bayesfilter.testing.tf_svd_cut_branch_diagnostics",
    "TFNonlinearBranchMode": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "TFNonlinearSigmaPointBackend": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "dense_gaussian_projection_step": "bayesfilter.testing.nonlinear_models_tf",
    "dense_projection_first_step": "bayesfilter.testing.nonlinear_models_tf",
    "fixed_sgqf_branch_summary": "bayesfilter.testing.fixed_sgqf_diagnostics_tf",
    "fixed_sgqf_diagnostic_snapshot": "bayesfilter.testing.fixed_sgqf_diagnostics_tf",
    "fixed_sgqf_failure_label": "bayesfilter.testing.fixed_sgqf_diagnostics_tf",
    "make_affine_gaussian_structural_oracle_tf": "bayesfilter.testing.nonlinear_models_tf",
    "make_nonlinear_accumulation_first_derivatives_tf": "bayesfilter.testing.nonlinear_models_tf",
    "make_nonlinear_accumulation_model_tf": "bayesfilter.testing.nonlinear_models_tf",
    "make_univariate_nonlinear_growth_first_derivatives_tf": "bayesfilter.testing.nonlinear_models_tf",
    "make_univariate_nonlinear_growth_model_tf": "bayesfilter.testing.nonlinear_models_tf",
    "model_a_observations_tf": "bayesfilter.testing.nonlinear_models_tf",
    "model_b_observations_tf": "bayesfilter.testing.nonlinear_models_tf",
    "model_c_observations_tf": "bayesfilter.testing.nonlinear_models_tf",
    "nonlinear_sigma_point_diagnostic_snapshot": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "nonlinear_sigma_point_score_branch_summary": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "nonlinear_sigma_point_value_branch_summary": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "run_model_b_nonlinear_svd_cut4_hmc_smoke": "bayesfilter.testing.tf_hmc_readiness",
    "run_qr_static_lgssm_hmc_smoke": "bayesfilter.testing.tf_hmc_readiness",
    "sigma_point_projection_first_step": "bayesfilter.testing.nonlinear_models_tf",
    "svd_cut_branch_frequency_summary": "bayesfilter.testing.tf_svd_cut_branch_diagnostics",
    "tf_nonlinear_sigma_point_score": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "tf_nonlinear_sigma_point_value_filter": "bayesfilter.testing.nonlinear_diagnostics_tf",
    "tf_svd_cut4_score_hessian_autodiff_oracle": "bayesfilter.testing.tf_svd_cut_autodiff_oracle",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module = import_module(_EXPORTS[name])
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
