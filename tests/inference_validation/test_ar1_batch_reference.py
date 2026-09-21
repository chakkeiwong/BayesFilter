"""Dense independent Gaussian quadratic forms check the analytical diagnostic."""
import numpy as np
import pytest

from bayesfilter.testing.inference_validation.engines.ar1_batch_reference import (
    expected_batch_lrv, expected_lugsail_mean_variance,
)


@pytest.mark.parametrize("rho", [0., -.4, .8, .995])
@pytest.mark.parametrize("stationary", [False, True])
def test_expected_batch_estimate_matches_dense_quadratic_form(rho, stationary):
    warmup, draws, batch = 3, 25, 4
    starts = (-3., 0., 2.)
    count = draws // batch
    t = np.arange(warmup + 1, warmup + count * batch + 1)
    covariance = rho**np.abs(t[:, None] - t[None, :])
    transient = rho**t
    if not stationary:
        covariance -= np.outer(transient, transient)
    average = np.kron(np.eye(count), np.ones((1, batch)) / batch)
    centering = np.eye(count) - np.ones((count, count)) / count
    quadratic = batch / (count - 1) * average.T @ centering @ average
    expected = [np.trace(quadratic @ covariance) + (
        0. if stationary else x * x * transient @ quadratic @ transient) for x in starts]
    actual = expected_batch_lrv(rho, starts, warmup=warmup, draws=draws,
                                batch=batch, stationary_start=stationary)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)


def test_iid_lugsail_mean_variance_is_unbiased_with_unused_terminal_draws():
    assert expected_lugsail_mean_variance(0., (1., 2., 3., 4.), warmup=3,
        draws=25, batch=6, stationary_start=False) == pytest.approx(1 / 100)


def test_persistent_gaussian_sqrt_batch_bias_is_detected():
    small = expected_lugsail_mean_variance(.995, (-8., -4., 4., 8.), warmup=10000,
        draws=10000, batch=100, stationary_start=False)
    large = expected_lugsail_mean_variance(.995, (-8., -4., 4., 8.), warmup=10000,
        draws=10000, batch=500, stationary_start=False)
    # The exact asymptotic mean variance is (1+rho)/(1-rho) / 40000.
    assert 0 < small < large < 399 / 40000
