"""Analytic oscillation controls, not sampler performance comparisons."""
import math

import numpy as np  # Independent known-law fixture.
import tensorflow as tf

from bayesfilter.inference.hmc_pilot_diagnostics import (
    pilot_diagnostic_program, summarize_pilot_window)


def test_antithetic_means_do_not_hide_slow_squared_quantities():
    # Stationary AR(1): tau(x)=1/19, tau(x^2)=181/19 at rho=-.9.
    rho = -.9
    assert (1+rho)/(1-rho) < .06
    assert (1+rho*rho)/(1-rho*rho) > 9.
    rng = np.random.default_rng(582)
    draws = rng.normal(size=(4096, 4, 1))
    for i in range(1, len(draws)):
        draws[i] = rho*draws[i-1] + math.sqrt(1-rho*rho)*draws[i]
    samples = tf.constant(draws)
    raw = pilot_diagnostic_program(tuple(draws.shape), 0, 64, True)(
        samples, tf.ones((1,), tf.float64), tf.zeros((0, 1), tf.float64))
    # Wide bounds from the exact lag-one correlations -.9 and .81. They
    # distinguish signed-mean correlation from squared-quantity persistence.
    assert bool(tf.reduce_all(raw["rho"][1, :, 0] < -.7))
    assert bool(tf.reduce_all(raw["rho"][1, :, 1] > .6))
    report = summarize_pilot_window(samples, frozen_scale=(1.,), max_lag=64)
    mean_tau, square_tau, _ = report["iact_initial_positive_pairs"]
    assert mean_tau is None or mean_tau < .5
    assert square_tau is None or square_tau > 2.
    assert square_tau is not None or report["iact_lag_limited"][1]
    assert not report["posterior_convergence_authority"]


def test_exact_sign_flip_does_not_invent_square_precision():
    values = np.where(np.arange(256) % 2, -1., 1.)[:, None, None]
    values = values*np.array([1., 1.2, 1.4, 1.6])[None, :, None]
    report = summarize_pilot_window(tf.constant(values), frozen_scale=(1.,), max_lag=32)
    assert report["longer_lag_squared_displacement_by_coordinate"] == [0.]
    assert report["constant_or_nonfinite_observable"] == [False, True, True]
    assert report["iact_initial_positive_pairs"] == [None, None, None]
    assert not report["source_estimate_usable"]


def test_randomized_hmc_includes_periodic_returns_and_real_movement():
    from tests.test_modern_hmc_kernels import runner
    # epsilon=1 gives Verlet phase pi/3. Six steps return to the start;
    # the uniform 1..6 policy also includes five nonreturning durations.
    chains = 128
    q = tf.constant(np.random.default_rng(583).normal(size=(chains, 2)))
    native = runner(family="hmc", trajectory="uniform_duration", count=1, chains=chains, jit=True)
    _, trace = native.one_step(q, tf.constant([584, 21]), tf.constant(1., tf.float64), tf.constant(6))
    periodic = trace["trajectory_steps"] == 6
    assert bool(tf.reduce_any(periodic)) and bool(tf.reduce_any(~periodic))
    np.testing.assert_allclose(tf.boolean_mask(trace["proposed_state"], periodic),
                               tf.boolean_mask(q, periodic), rtol=0, atol=2e-13)
    assert float(tf.reduce_max(tf.abs(trace["proposed_state"]-q))) > 1.
    assert bool(tf.reduce_all(tf.math.is_finite(trace["log_accept_ratio"])))
    # A returning endpoint is part of this valid randomized policy. It is
    # not evidence that the entire frozen policy is numerically invalid.
