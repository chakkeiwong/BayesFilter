import numpy as np

from bayesfilter.filters import StructuralSVDSigmaPointFilter
from bayesfilter.testing.structural_fixtures import NonlinearAccumulationModel


def _dense_one_step_gaussian_observation_ll(model, observation, order=7):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    weights = weights / np.sqrt(np.pi)
    mean = model.initial_mean()
    cov = model.initial_cov()
    innovation_cov = model.innovation_cov()
    aug_cov = np.zeros((3, 3))
    aug_cov[:2, :2] = cov
    aug_cov[2:, 2:] = innovation_cov
    factor = np.linalg.cholesky(aug_cov + 1e-15 * np.eye(3))

    y_values = []
    w_values = []
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            for k, wk in enumerate(weights):
                z = np.sqrt(2.0) * np.array([nodes[i], nodes[j], nodes[k]])
                point = np.concatenate([mean, [0.0]]) + factor @ z
                state = model.transition(point[:2], point[2:])
                y_values.append(model.observe(state[None, :])[0, 0])
                w_values.append(wi * wj * wk)
    y_values = np.asarray(y_values)
    w_values = np.asarray(w_values)
    y_mean = float(w_values @ y_values)
    centered = y_values - y_mean
    y_var = float(w_values @ (centered**2) + model.observation_cov()[0, 0])
    innovation = float(observation - y_mean)
    return -0.5 * (np.log(2.0 * np.pi * y_var) + innovation**2 / y_var)


def test_nonlinear_structural_sigma_likelihood_is_finite_and_near_dense_reference():
    model = NonlinearAccumulationModel()
    observation = np.array([[0.15]], dtype=float)

    result = StructuralSVDSigmaPointFilter().filter(model, observation)
    reference = _dense_one_step_gaussian_observation_ll(model, observation[0, 0])

    assert np.isfinite(result.log_likelihood)
    assert abs(result.log_likelihood - reference) < 0.15


def test_nonlinear_transition_completes_deterministic_coordinate():
    model = NonlinearAccumulationModel(alpha=0.4, beta=0.9)
    previous = np.array([0.2, -0.1])
    innovation = np.array([0.5])

    next_state = model.transition(previous, innovation)
    expected_m = model.rho * previous[0] + model.sigma * innovation[0]
    expected_k = model.alpha * previous[1] + model.beta * np.tanh(expected_m)

    np.testing.assert_allclose(next_state, np.array([expected_m, expected_k]))
