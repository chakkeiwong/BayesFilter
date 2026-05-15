from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LinearGaussianFixture:
    a: float
    h: float
    q: float
    r: float
    prior_mean: float
    prior_variance: float
    observation: float
    predictive_mean: float
    predictive_variance: float
    posterior_mean: float
    posterior_variance: float
    log_likelihood: float


def build_linear_gaussian_fixture() -> LinearGaussianFixture:
    a = 0.9
    h = 1.25
    q = 0.4
    r = 0.7
    prior_mean = 0.35
    prior_variance = 1.1
    observation = -0.2

    predictive_mean = a * prior_mean
    predictive_variance = (a * a) * prior_variance + q
    innovation_variance = (h * h) * predictive_variance + r
    innovation = observation - h * predictive_mean
    kalman_gain = predictive_variance * h / innovation_variance
    posterior_mean = predictive_mean + kalman_gain * innovation
    posterior_variance = predictive_variance - kalman_gain * h * predictive_variance
    log_likelihood = -0.5 * (
        math.log(2.0 * math.pi * innovation_variance)
        + (innovation * innovation) / innovation_variance
    )

    return LinearGaussianFixture(
        a=a,
        h=h,
        q=q,
        r=r,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        observation=observation,
        predictive_mean=predictive_mean,
        predictive_variance=predictive_variance,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        log_likelihood=log_likelihood,
    )
