import numpy as np

from bayesfilter.filters import StructuralSVDSigmaPointFilter, UnscentedRule
from bayesfilter.testing.structural_fixtures import (
    NonlinearAccumulationModel,
    WorkedStructuralUKFModel,
)


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


def _worked_ukf_one_step_moments(model):
    mean = model.initial_mean()
    covariance = model.initial_cov()
    innovation_cov = model.innovation_cov()
    augmented_mean = np.concatenate([mean, np.zeros(model.partition.innovation_dim)])
    augmented_covariance = np.zeros((3, 3), dtype=float)
    augmented_covariance[:2, :2] = covariance
    augmented_covariance[2:, 2:] = innovation_cov
    points, mean_weights, covariance_weights, _ = UnscentedRule(3).sigma_points(
        augmented_mean,
        augmented_covariance,
    )
    predicted_points = np.asarray(
        [model.transition(point[:2], point[2:]) for point in points],
        dtype=float,
    )
    predicted_mean = mean_weights @ predicted_points
    centered_x = predicted_points - predicted_mean[None, :]
    predicted_covariance = centered_x.T @ (centered_x * covariance_weights[:, None])
    observation_points = model.observe(predicted_points)
    observation_mean = mean_weights @ observation_points
    centered_y = observation_points - observation_mean[None, :]
    innovation_covariance = (
        centered_y.T @ (centered_y * covariance_weights[:, None]) + model.observation_cov()
    )
    cross_covariance = centered_x.T @ (centered_y * covariance_weights[:, None])
    innovation = np.array([0.30]) - observation_mean
    gain = np.linalg.solve(innovation_covariance.T, cross_covariance.T).T
    posterior_mean = predicted_mean + gain @ innovation
    posterior_covariance = predicted_covariance - gain @ innovation_covariance @ gain.T
    log_likelihood = -0.5 * (
        np.log(2.0 * np.pi * innovation_covariance[0, 0])
        + innovation[0] ** 2 / innovation_covariance[0, 0]
    )
    return {
        "points": points,
        "predicted_points": predicted_points,
        "predicted_mean": predicted_mean,
        "predicted_covariance": predicted_covariance,
        "observation_mean": observation_mean,
        "innovation_covariance": innovation_covariance,
        "cross_covariance": cross_covariance,
        "innovation": innovation,
        "gain": gain,
        "posterior_mean": posterior_mean,
        "posterior_covariance": posterior_covariance,
        "log_likelihood": float(log_likelihood),
    }


def test_worked_structural_ukf_example_reproduces_chapter_numbers():
    model = WorkedStructuralUKFModel()
    moments = _worked_ukf_one_step_moments(model)

    np.testing.assert_allclose(
        moments["predicted_mean"],
        np.array([0.0, 0.11024]),
        atol=5e-6,
    )
    np.testing.assert_allclose(
        moments["predicted_covariance"],
        np.array([[0.27560, 0.0], [0.0, 0.08656743]]),
        atol=5e-6,
    )
    np.testing.assert_allclose(moments["observation_mean"], np.array([0.11024]), atol=5e-6)
    np.testing.assert_allclose(
        moments["innovation_covariance"],
        np.array([[0.61216743]]),
        atol=5e-6,
    )
    np.testing.assert_allclose(
        moments["cross_covariance"],
        np.array([[0.27560], [0.08656743]]),
        atol=5e-6,
    )
    np.testing.assert_allclose(moments["innovation"], np.array([0.18976]), atol=5e-6)
    np.testing.assert_allclose(
        moments["gain"].ravel(),
        np.array([0.45020363, 0.14141136]),
        atol=5e-6,
    )
    np.testing.assert_allclose(
        moments["posterior_mean"],
        np.array([0.08543064, 0.13707422]),
        atol=5e-6,
    )
    np.testing.assert_allclose(
        moments["posterior_covariance"],
        np.array([[0.15152388, -0.03897297], [-0.03897297, 0.07432581]]),
        atol=5e-6,
    )
    assert abs(moments["log_likelihood"] - -0.7029747608892933) < 5e-6

    artificial_s = moments["innovation_covariance"][0, 0] + 0.04
    artificial_log_likelihood = -0.5 * (
        np.log(2.0 * np.pi * artificial_s) + moments["innovation"][0] ** 2 / artificial_s
    )
    assert abs(artificial_s - 0.6521674304) < 5e-6
    assert abs(artificial_log_likelihood - -0.7328186209822024) < 5e-6


def test_worked_structural_ukf_points_preserve_deterministic_identity():
    model = WorkedStructuralUKFModel()
    moments = _worked_ukf_one_step_moments(model)

    for point, predicted in zip(
        moments["points"],
        moments["predicted_points"],
        strict=True,
    ):
        assert abs(model.deterministic_residual(point[:2], predicted)) < 1e-14
