import sys

import bayesfilter


V1_PUBLIC_SYMBOLS = {
    "TFLinearGaussianStateSpace",
    "TFLinearGaussianStateSpaceDerivatives",
    "tf_linear_gaussian_log_likelihood",
    "tf_kalman_log_likelihood",
    "tf_masked_kalman_log_likelihood",
    "tf_qr_linear_gaussian_log_likelihood",
    "tf_qr_sqrt_kalman_log_likelihood",
    "tf_qr_sqrt_masked_kalman_log_likelihood",
    "tf_qr_linear_gaussian_score_hessian",
    "tf_qr_sqrt_kalman_score_hessian",
    "tf_qr_sqrt_masked_kalman_score_hessian",
    "tf_svd_linear_gaussian_log_likelihood",
    "tf_svd_kalman_log_likelihood",
    "tf_svd_masked_kalman_log_likelihood",
    "TFStructuralStateSpace",
    "structural_block_metadata",
    "structural_filter_diagnostics",
    "structural_filter_metadata",
    "pointwise_deterministic_residuals",
    "affine_structural_to_linear_gaussian_tf",
    "make_affine_structural_tf",
    "tf_svd_sigma_point_log_likelihood",
    "tf_svd_sigma_point_log_likelihood_with_rule",
    "tf_svd_sigma_point_filter",
    "tf_svd_sigma_point_placement",
    "tf_svd_sigma_point_score_with_rule",
    "tf_svd_cubature_score",
    "tf_svd_ukf_score",
    "tf_unit_sigma_point_rule",
    "tf_cut4g_sigma_point_rule",
    "tf_svd_cut4_log_likelihood",
    "tf_svd_cut4_filter",
    "tf_svd_cut4_score",
    "TFStructuralFirstDerivatives",
}


def test_v1_public_api_symbols_are_top_level_importable() -> None:
    missing = sorted(name for name in V1_PUBLIC_SYMBOLS if not hasattr(bayesfilter, name))

    assert missing == []
    assert V1_PUBLIC_SYMBOLS.issubset(set(bayesfilter.__all__))


def test_v1_public_api_import_does_not_import_external_clients() -> None:
    forbidden_prefixes = ("dsge_hmc", "MacroFinance", "macrofinance")
    imported_clients = sorted(
        name for name in sys.modules if name.startswith(forbidden_prefixes)
    )

    assert imported_clients == []
