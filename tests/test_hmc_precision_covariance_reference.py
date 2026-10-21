"""Exact diagnostic covariance oracles; no simulated sampler or coverage claim."""
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_precision import _batch_program, mean_precision
from scripts.calibrate_hmc_stopped_mcse import finite_mean_variance


def covariance_lrv(n, batch, rho):
    """Expectation from the full covariance of complete consecutive batch means."""
    count = n // batch
    means_covariance = [
        [math.fsum(rho ** abs(i * batch + u - j * batch - v)
                   for u in range(batch) for v in range(batch)) / batch**2
         for j in range(count)]
        for i in range(count)
    ]
    trace = math.fsum(means_covariance[i][i] for i in range(count))
    all_covariances = math.fsum(value for row in means_covariance for value in row)
    return batch / (count - 1) * (trace - all_covariances / count)


@pytest.mark.parametrize("rho", [0., .9, -.9])
@pytest.mark.parametrize("n,batch", [(12, 3), (14, 4), (23, 5), (36, 6)])
def test_tensorflow_batch_formula_matches_exact_covariance(rho, n, batch):
    covariance = rho ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    # These columns are deterministic quadrature points with mean second
    # moment C, not independent draws/chains for convergence assessment.
    points = math.sqrt(n) * np.linalg.cholesky(covariance)
    small = batch // 3
    plain, lugsail = _batch_program((n, n, 1), batch, small, .5, True)(
        tf.constant(points[:, :, None], tf.float64))
    expected_plain = covariance_lrv(n, batch, rho)
    expected_small = covariance_lrv(n, small, rho)
    assert float(tf.reduce_mean(plain)) == pytest.approx(expected_plain, abs=2e-12, rel=2e-11)
    assert float(tf.reduce_mean(lugsail)) == pytest.approx(
        2 * expected_plain - expected_small, abs=2e-12, rel=2e-11)


def test_default_lugsail_is_not_universally_conservative_for_antithetic_means():
    n, rho = 10000, -.9

    def expected_batch_lrv(batch):
        count = n // batch
        return batch * count / (count - 1) * (
            finite_mean_variance(batch, rho, chains=1)
            - finite_mean_variance(count * batch, rho, chains=1))

    exact_mean_variance = finite_mean_variance(n, rho, chains=1)
    plain = expected_batch_lrv(100)
    lugsail = 2 * plain - expected_batch_lrv(33)
    assert plain / (n * exact_mean_variance) > 1
    assert lugsail / (n * exact_mean_variance) == pytest.approx(.8935868831421041, rel=2e-11)


def test_positive_pool_cannot_hide_a_negative_chain_variance():
    alternating_blocks = np.tile(np.repeat([1., -1.], 10), 12)
    drift = np.arange(240, dtype=float)
    values = np.stack((alternating_blocks, drift), axis=1)[:, :, None]
    result = mean_precision(values, method="lugsail", batch_size=20,
                            min_batches=2, jit_compile=True)
    per_chain = result["per_chain_long_run_variance"][:, 0]
    assert float(per_chain[0]) < 0
    assert float(tf.reduce_sum(per_chain)) > 0
    assert not bool(result["valid"][0])
    assert math.isnan(float(result["mcse"][0]))
