"""Independent-reference numerical tests; no sampler or promotion claims."""
import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from bayesfilter.testing.inference_validation.references import modern_macrofinance as ref


def points():
    return np.random.default_rng(20261029).normal(0., .25, (4, 9))


def dense_observation_law(f, q, r, initial_mean, initial_covariance, horizon):
    means, covariances = [], []
    mean, covariance = initial_mean.copy(), initial_covariance.copy()
    offset = ref.STATE_MEAN - f @ ref.STATE_MEAN
    for _ in range(horizon):
        mean = offset + f @ mean
        covariance = f @ covariance @ f.T + q
        means.append(mean + ref.OBSERVATION_OFFSET)
        covariances.append(covariance)
    joint = np.empty((2*horizon, 2*horizon))
    for i in range(horizon):
        for j in range(i+1):
            block = np.linalg.matrix_power(f, i-j) @ covariances[j]
            if i == j:
                block = block + r
            joint[2*i:2*i+2, 2*j:2*j+2] = block
            joint[2*j:2*j+2, 2*i:2*i+2] = block.T
    return np.concatenate(means), joint


@pytest.mark.parametrize("horizon", [1, 2, 7])
@pytest.mark.parametrize("stationary_start", [True, False])
def test_prediction_first_recursion_matches_dense_joint_gaussian(horizon, stationary_start):
    y = np.random.default_rng(44).normal(size=(horizon, 2))
    f, q, r, p = ref.decode_chart(points()[:1])
    m0 = ref.STATE_MEAN if stationary_start else np.array([2., -1.])
    p0 = p[0] if stationary_start else np.diag([.3, 1.7])
    actual = ref.covariance_log_likelihood(y, f, q, r, p, initial_mean=m0, initial_covariance=p0)[0]
    mean, covariance = dense_observation_law(f[0], q[0], r[0], m0, p0, horizon)
    expected = multivariate_normal.logpdf(y.ravel(), mean=mean, cov=covariance)
    np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-12)


def test_direct_lyapunov_and_row_order_independence():
    theta = points()
    f, q, _, p = ref.decode_chart(theta)
    for i in range(len(theta)):
        np.testing.assert_allclose(p[i], solve_discrete_lyapunov(f[i], q[i]), rtol=1.e-12, atol=1.e-12)
    y = np.random.default_rng(17).normal(size=(48, 2))
    actual = ref.log_density(theta, y)
    scalar = np.array([ref.log_density(row[None], y)[0] for row in theta])
    np.testing.assert_allclose(actual, scalar, rtol=1.e-12, atol=1.e-12)
    np.testing.assert_allclose(ref.log_density(theta[[2,0,2,1]], y), actual[[2,0,2,1]], rtol=1.e-12, atol=1.e-12)


def test_standardized_prior_measure_and_quantity_order():
    z = points() / ref.PRIOR_SCALE
    y = np.zeros((3, 2))
    expected = ref.log_density(z*ref.PRIOR_SCALE, y) - 4.5*np.log(2*np.pi)
    np.testing.assert_allclose(ref.log_posterior_standardized(z,y),expected)
    values = ref.quantities_standardized(z)
    assert values.shape == (4,31) and len(ref.QUANTITY_NAMES)==31
    np.testing.assert_allclose(values[:,:9],z*ref.PRIOR_SCALE)
    np.testing.assert_allclose(values[:,9:18],z*z)
    assert np.all((values[:,18:27]>0)&(values[:,18:27]<1))
    assert np.all(values[:,27:29]>0) and np.all(np.abs(values[:,29:])<1)


@pytest.mark.parametrize("bad", [np.zeros(9), np.zeros((2,8)), np.empty((0,9)), np.full((1,9),np.nan)])
def test_invalid_chart_inputs_fail(bad):
    with pytest.raises(ValueError, match="chart coordinates"):
        ref.decode_chart(bad)


@pytest.mark.parametrize("bad", [np.zeros(4), np.zeros((3,3)), np.empty((0,2)), np.full((2,2),np.inf)])
def test_invalid_data_fail(bad):
    with pytest.raises(ValueError, match="observations"):
        ref.log_density(points(),bad)
