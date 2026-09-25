"""Independent finite-count expectations for the Gaussian AR(1) diagnostic.

This is a mathematical reference for post-run estimator diagnosis, never a
bandwidth selector or an inference implementation.
"""
from __future__ import annotations

import math


def expected_batch_lrv(rho, starts, *, warmup, draws, batch, stationary_start):
    """Expected ordinary per-chain batch-means LRV, including transient means.

    Writing batch averages as Y, E[sum(Y-Ybar)^2] is tr(H Cov(Y)) + mu'Hmu.
    Conditional AR(1) covariance subtracts uu', where u averages rho**t
    over each batch; a deterministic start adds starts[c]**2 * uu'.
    """
    if not math.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be finite and strictly between -1 and 1")
    if (type(warmup) is not int or warmup < 0 or type(draws) is not int
            or draws < 4 or type(batch) is not int or batch < 1 or draws // batch < 2):
        raise ValueError("nonnegative warmup and at least two complete batches required")
    starts = tuple(float(v) for v in starts)
    if not starts or any(not math.isfinite(v) for v in starts):
        raise ValueError("finite starting values required")
    count = draws // batch
    used = count * batch

    def sum_variance(n):
        return n + 2 * math.fsum((n - k) * rho**k for k in range(1, n))

    expectation = (sum_variance(batch) - sum_variance(used) / count**2) / (
        batch * (1 - 1 / count))
    if stationary_start:
        return tuple(expectation for _ in starts)
    u = [math.fsum(rho**t for t in range(warmup + i * batch + 1,
                                        warmup + (i + 1) * batch + 1)) / batch
         for i in range(count)]
    center = math.fsum(u) / count
    correction = batch * math.fsum((v - center)**2 for v in u) / (count - 1)
    return tuple(expectation + (x * x - 1) * correction for x in starts)


def expected_lugsail_mean_variance(rho, starts, *, warmup, draws, batch,
                                    stationary_start, r=3, c=.5):
    """Expectation of the untruncated pooled lugsail variance estimator."""
    if type(r) is not int or r < 1 or not math.isfinite(c) or not 0 <= c < 1:
        raise ValueError("invalid lugsail constants")
    starts = tuple(starts)
    large = expected_batch_lrv(rho, starts, warmup=warmup, draws=draws,
                               batch=batch, stationary_start=stationary_start)
    small = expected_batch_lrv(rho, starts, warmup=warmup, draws=draws,
                               batch=batch // r, stationary_start=stationary_start)
    return math.fsum((a - c * b) / (1 - c) for a, b in zip(large, small)) / (
        len(starts)**2 * draws)
