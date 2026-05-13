"""Testing fixtures for BayesFilter contracts."""

from bayesfilter.testing.nonlinear_models_tf import (
    DenseProjectionStep,
    dense_gaussian_projection_step,
    dense_projection_first_step,
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_first_derivatives_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_first_derivatives_tf,
    make_univariate_nonlinear_growth_model_tf,
    model_a_observations_tf,
    model_b_observations_tf,
    model_c_observations_tf,
    sigma_point_projection_first_step,
)
from bayesfilter.testing.nonlinear_diagnostics_tf import (
    NonlinearSigmaPointBranchSummary,
    NonlinearSigmaPointDiagnosticSnapshot,
    TFNonlinearBranchMode,
    TFNonlinearSigmaPointBackend,
    nonlinear_sigma_point_diagnostic_snapshot,
    nonlinear_sigma_point_score_branch_summary,
    nonlinear_sigma_point_value_branch_summary,
    tf_nonlinear_sigma_point_score,
    tf_nonlinear_sigma_point_value_filter,
)
from bayesfilter.testing.tf_hmc_readiness import (
    ModelBNonlinearSVDTarget,
    QRStaticLGSSMTarget,
    run_model_b_nonlinear_svd_cut4_hmc_smoke,
    run_qr_static_lgssm_hmc_smoke,
)
from bayesfilter.testing.tf_svd_cut_branch_diagnostics import (
    SVDCUTBranchSummary,
    svd_cut_branch_frequency_summary,
)
from bayesfilter.testing.tf_svd_cut_autodiff_oracle import (
    tf_svd_cut4_score_hessian_autodiff_oracle,
)

__all__ = [
    "DenseProjectionStep",
    "NonlinearSigmaPointBranchSummary",
    "NonlinearSigmaPointDiagnosticSnapshot",
    "ModelBNonlinearSVDTarget",
    "QRStaticLGSSMTarget",
    "SVDCUTBranchSummary",
    "TFNonlinearBranchMode",
    "TFNonlinearSigmaPointBackend",
    "dense_gaussian_projection_step",
    "dense_projection_first_step",
    "make_affine_gaussian_structural_oracle_tf",
    "make_nonlinear_accumulation_first_derivatives_tf",
    "make_nonlinear_accumulation_model_tf",
    "make_univariate_nonlinear_growth_first_derivatives_tf",
    "make_univariate_nonlinear_growth_model_tf",
    "model_a_observations_tf",
    "model_b_observations_tf",
    "model_c_observations_tf",
    "nonlinear_sigma_point_diagnostic_snapshot",
    "nonlinear_sigma_point_score_branch_summary",
    "nonlinear_sigma_point_value_branch_summary",
    "run_model_b_nonlinear_svd_cut4_hmc_smoke",
    "run_qr_static_lgssm_hmc_smoke",
    "sigma_point_projection_first_step",
    "svd_cut_branch_frequency_summary",
    "tf_nonlinear_sigma_point_score",
    "tf_nonlinear_sigma_point_value_filter",
    "tf_svd_cut4_score_hessian_autodiff_oracle",
]
