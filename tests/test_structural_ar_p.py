import numpy as np

from bayesfilter.filters import StructuralSVDSigmaPointFilter, kalman_log_likelihood
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def test_structural_sigma_points_recover_ar2_linear_kalman_likelihood():
    model = AR2StructuralModel(phi1=0.35, phi2=-0.10, sigma=0.25, observation_sigma=0.15)
    observations = np.array([[0.2], [0.05], [-0.1], [0.15], [0.0]], dtype=float)

    kalman = kalman_log_likelihood(observations, model.as_lgssm())
    sigma = StructuralSVDSigmaPointFilter().filter(model, observations)

    np.testing.assert_allclose(sigma.log_likelihood, kalman.log_likelihood, rtol=1e-10, atol=1e-10)


def test_ar2_transition_preserves_lag_shift_identity_pointwise():
    model = AR2StructuralModel()
    previous = np.array([0.7, -0.2])
    innovation = np.array([1.1])

    next_state = model.transition(previous, innovation)

    assert next_state[1] == previous[0]
