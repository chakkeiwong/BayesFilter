import numpy as np

from bayesfilter.filters import kalman_log_likelihood
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def test_kalman_accepts_structurally_singular_process_covariance():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.2], [0.1], [-0.1], [0.05]], dtype=float)

    result = kalman_log_likelihood(observations, model, return_filtered=True)

    assert np.isfinite(result.log_likelihood)
    assert result.filtered_means.shape == (4, 2)
    assert np.linalg.matrix_rank(model.transition_covariance) == 1


def test_kalman_all_missing_step_contributes_prediction_only():
    model = AR2StructuralModel().as_lgssm()
    observations = np.array([[0.2], [0.0], [0.1]], dtype=float)
    mask = np.array([[True], [False], [True]])

    with_missing = kalman_log_likelihood(observations, model, mask=mask)
    compressed = kalman_log_likelihood(observations[[0, 2]], model)

    assert np.isfinite(with_missing.log_likelihood)
    assert with_missing.log_likelihood != compressed.log_likelihood
