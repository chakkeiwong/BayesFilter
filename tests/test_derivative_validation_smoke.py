import numpy as np

from bayesfilter.filters import kalman_log_likelihood
from bayesfilter.testing.structural_fixtures import AR2StructuralModel


def _finite_difference_phi1(phi1):
    model = AR2StructuralModel(phi1=phi1).as_lgssm()
    observations = np.array([[0.2], [0.05], [-0.1], [0.15]], dtype=float)
    return kalman_log_likelihood(observations, model).log_likelihood


def test_finite_difference_derivative_smoke_for_reference_kalman_value():
    phi = 0.4
    step = 1e-5
    grad = (_finite_difference_phi1(phi + step) - _finite_difference_phi1(phi - step)) / (2.0 * step)

    assert np.isfinite(grad)
