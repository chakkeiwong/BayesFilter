import numpy as np

from bayesfilter.filters import kalman_log_likelihood, solve_kalman_score_hessian
from bayesfilter.linear import LinearGaussianStateSpace, LinearGaussianStateSpaceDerivatives


def _model(theta):
    phi, log_q, obs_offset = theta
    q = np.exp(log_q)
    return LinearGaussianStateSpace(
        initial_mean=np.array([0.1]),
        initial_covariance=np.array([[0.7]]),
        transition_offset=np.array([0.05]),
        transition_matrix=np.array([[phi]]),
        transition_covariance=np.array([[q]]),
        observation_offset=np.array([obs_offset]),
        observation_matrix=np.array([[1.2]]),
        observation_covariance=np.array([[0.4]]),
    )


def _derivatives(theta):
    _, log_q, _ = theta
    q = np.exp(log_q)
    p, n, m = 3, 1, 1
    zeros_pn = np.zeros((p, n))
    zeros_pnn = np.zeros((p, n, n))
    zeros_pm = np.zeros((p, m))
    zeros_pmn = np.zeros((p, m, n))
    zeros_pmm = np.zeros((p, m, m))
    zeros_ppn = np.zeros((p, p, n))
    zeros_ppnn = np.zeros((p, p, n, n))
    zeros_ppm = np.zeros((p, p, m))
    zeros_ppmn = np.zeros((p, p, m, n))
    zeros_ppmm = np.zeros((p, p, m, m))

    d_transition_matrix = zeros_pnn.copy()
    d_transition_matrix[0, 0, 0] = 1.0
    d_transition_covariance = zeros_pnn.copy()
    d_transition_covariance[1, 0, 0] = q
    d_observation_offset = zeros_pm.copy()
    d_observation_offset[2, 0] = 1.0
    d2_transition_covariance = zeros_ppnn.copy()
    d2_transition_covariance[1, 1, 0, 0] = q

    return LinearGaussianStateSpaceDerivatives(
        d_initial_mean=zeros_pn.copy(),
        d_initial_covariance=zeros_pnn.copy(),
        d_transition_offset=zeros_pn.copy(),
        d_transition_matrix=d_transition_matrix,
        d_transition_covariance=d_transition_covariance,
        d_observation_offset=d_observation_offset,
        d_observation_matrix=zeros_pmn.copy(),
        d_observation_covariance=zeros_pmm.copy(),
        d2_initial_mean=zeros_ppn.copy(),
        d2_initial_covariance=zeros_ppnn.copy(),
        d2_transition_offset=zeros_ppn.copy(),
        d2_transition_matrix=zeros_ppnn.copy(),
        d2_transition_covariance=d2_transition_covariance,
        d2_observation_offset=zeros_ppm.copy(),
        d2_observation_matrix=zeros_ppmn.copy(),
        d2_observation_covariance=zeros_ppmm.copy(),
    )


def _loglik(theta, observations):
    return kalman_log_likelihood(observations, _model(theta)).log_likelihood


def _finite_difference_grad_hess(theta, observations, step=1e-5):
    p = theta.size
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    base = _loglik(theta, observations)
    for i in range(p):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (_loglik(plus, observations) - _loglik(minus, observations)) / (
            2.0 * step
        )
        hess[i, i] = (
            _loglik(plus, observations)
            - 2.0 * base
            + _loglik(minus, observations)
        ) / (step**2)
        for j in range(i + 1, p):
            pp = theta.copy()
            pm = theta.copy()
            mp = theta.copy()
            mm = theta.copy()
            pp[[i, j]] += step
            pm[i] += step
            pm[j] -= step
            mp[i] -= step
            mp[j] += step
            mm[[i, j]] -= step
            value = (
                _loglik(pp, observations)
                - _loglik(pm, observations)
                - _loglik(mp, observations)
                + _loglik(mm, observations)
            ) / (4.0 * step**2)
            hess[i, j] = value
            hess[j, i] = value
    return grad, hess


def test_solve_kalman_score_hessian_matches_finite_differences():
    theta = np.array([0.55, np.log(0.2), 0.03])
    observations = np.array([[0.2], [0.1], [-0.05], [0.15]], dtype=float)

    result = solve_kalman_score_hessian(
        observations,
        _model(theta),
        _derivatives(theta),
        return_trace=True,
    )
    fd_grad, fd_hess = _finite_difference_grad_hess(theta, observations)

    np.testing.assert_allclose(result.log_likelihood, _loglik(theta, observations), atol=1e-10)
    np.testing.assert_allclose(result.score, fd_grad, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(result.hessian, fd_hess, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=1e-12)
    assert result.metadata.filter_name == "solve_kalman_score_hessian"
    assert len(result.trace) == observations.shape[0]
    assert result.diagnostics["mask_convention"] == "dense_only"
