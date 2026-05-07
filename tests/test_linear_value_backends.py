import numpy as np
import pytest

from bayesfilter.filters import (
    kalman_log_likelihood,
    linear_gaussian_log_likelihood,
    solve_kalman_log_likelihood,
    svd_kalman_log_likelihood,
)
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def test_solve_and_svd_linear_backends_match_covariance_reference():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.2], [0.1], [-0.1], [0.05]], dtype=float)

    covariance = kalman_log_likelihood(observations, model, return_filtered=True)
    solve = solve_kalman_log_likelihood(observations, model, return_filtered=True)
    svd = svd_kalman_log_likelihood(observations, model, return_filtered=True)

    np.testing.assert_allclose(solve.log_likelihood, covariance.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(svd.log_likelihood, covariance.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(solve.filtered_means, covariance.filtered_means, atol=1e-10)
    np.testing.assert_allclose(svd.filtered_means, covariance.filtered_means, atol=1e-10)
    assert solve.metadata.filter_name == "solve_kalman"
    assert svd.metadata.filter_name == "svd_kalman"
    assert solve.diagnostics["mask_convention"] == "row_selection"
    assert svd.diagnostics["singular_floor"] == 1e-12


def test_linear_backend_selector_dispatches_and_rejects_unknown_backend():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.2], [0.1]], dtype=float)

    direct = solve_kalman_log_likelihood(observations, model)
    selected = linear_gaussian_log_likelihood(observations, model, backend="solve")

    np.testing.assert_allclose(selected.log_likelihood, direct.log_likelihood)
    with pytest.raises(ValueError, match="unknown linear Gaussian backend"):
        linear_gaussian_log_likelihood(observations, model, backend="not_a_backend")


def test_svd_backend_reports_floor_activity_for_near_singular_innovation():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.0]], dtype=float)

    result = svd_kalman_log_likelihood(
        observations,
        model,
        singular_floor=10.0,
    )

    assert np.isfinite(result.log_likelihood)
    assert result.diagnostics["max_floor_count"] == 1
    assert result.diagnostics["min_innovation_eigenvalue"] < 10.0


def test_solve_and_svd_backends_preserve_all_missing_prediction_only_step():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.2], [0.0], [0.1]], dtype=float)
    mask = np.array([[True], [False], [True]])

    covariance = kalman_log_likelihood(observations, model, mask=mask)
    solve = solve_kalman_log_likelihood(observations, model, mask=mask)
    svd = svd_kalman_log_likelihood(observations, model, mask=mask)

    np.testing.assert_allclose(solve.log_likelihood, covariance.log_likelihood, atol=1e-10)
    np.testing.assert_allclose(svd.log_likelihood, covariance.log_likelihood, atol=1e-10)
