"""Focused mechanics tests for the scalar piecewise-density kernel."""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.piecewise_density_hmc_1d import (
    PiecewiseDensity1DTransitionKernel,
    piecewise_density_leapfrog_proposal_1d,
    scalar_event_drift,
    scalar_reflection_or_refraction,
)


def _step_target(log_weight: float):
    def target(x):
        x = tf.cast(x, tf.float64)
        return -0.5 * tf.square(x) + tf.where(x >= 0.0, tf.constant(log_weight, tf.float64), 0.0)

    def score(x):
        x = tf.cast(x, tf.float64)
        return -x

    return target, score


def test_reflection_and_refraction_conserve_scalar_event_energy():
    momentum, reflected, refracted = scalar_reflection_or_refraction(
        tf.constant([0.5, 1.0, -1.0], tf.float64),
        tf.constant([0.5, 0.5, -0.5], tf.float64),
    )
    np.testing.assert_allclose(momentum.numpy(), [-0.5, 0.0, -math.sqrt(2.0)])
    np.testing.assert_array_equal(reflected.numpy(), [True, False, False])
    np.testing.assert_array_equal(refracted.numpy(), [False, True, True])


def test_kernel_is_calibrated_and_reports_event_telemetry():
    target, score = _step_target(math.log(0.25))
    kernel = PiecewiseDensity1DTransitionKernel(
        target_log_prob_fn=target,
        score_fn=score,
        boundary=0.0,
        potential_jump=math.log(4.0),
        step_size=0.25,
    )
    assert kernel.is_calibrated
    state = tf.constant([-0.01], tf.float64)
    results = kernel.bootstrap_results(state)
    expected_log_prob = target(state)
    np.testing.assert_allclose(
        results.accepted_results.numpy(), expected_log_prob.numpy(), rtol=0.0, atol=0.0
    )
    proposed, next_results = kernel.one_step(
        state, results, seed=tf.constant([7, 11], tf.int32)
    )
    assert proposed.shape == (1,)
    assert next_results.event_count.shape == (1,)
    assert bool(tf.reduce_all(next_results.finite_status).numpy())


def test_scalar_event_drift_replays_under_momentum_reversal():
    for position, momentum, jump, step_size in (
        (-0.01, 1.0, math.log(4.0), 0.03),
        (-0.01, 8.0, math.log(4.0), 0.03),
    ):
        q0 = tf.constant([position], tf.float64)
        p0 = tf.constant([momentum], tf.float64)
        q1, p1, crossed, _reflected, _refracted = scalar_event_drift(
            q0, p0, boundary=0.0, potential_jump=jump, step_size=step_size
        )
        q2, p2, _crossed2, _reflected2, _refracted2 = scalar_event_drift(
            q1, -p1, boundary=0.0, potential_jump=jump, step_size=step_size
        )
        assert bool(crossed.numpy()[0])
        np.testing.assert_allclose(q2.numpy(), q0.numpy(), atol=2.0e-14, rtol=0.0)
        np.testing.assert_allclose(p2.numpy(), -p0.numpy(), atol=2.0e-14, rtol=0.0)


def test_full_kick_event_drift_kick_proposal_replays_under_reversal():
    _target, score = _step_target(math.log(0.25))
    for position, momentum in ((-0.01, 1.0), (-0.01, 8.0)):
        q0 = tf.constant([position], tf.float64)
        p0 = tf.constant([momentum], tf.float64)
        q1, p1, crossed, _reflected, _refracted = (
            piecewise_density_leapfrog_proposal_1d(
                q0,
                p0,
                score_fn=score,
                boundary=0.0,
                potential_jump=math.log(4.0),
                step_size=0.03,
            )
        )
        q2, p2, _crossed2, _reflected2, _refracted2 = (
            piecewise_density_leapfrog_proposal_1d(
                q1,
                -p1,
                score_fn=score,
                boundary=0.0,
                potential_jump=math.log(4.0),
                step_size=0.03,
            )
        )
        assert bool(crossed.numpy()[0])
        np.testing.assert_allclose(q2.numpy(), q0.numpy(), atol=2.0e-14, rtol=0.0)
        np.testing.assert_allclose(p2.numpy(), -p0.numpy(), atol=2.0e-14, rtol=0.0)


def test_kernel_runs_through_tfp_sample_chain_with_fixed_signature():
    target, score = _step_target(math.log(0.25))
    kernel = PiecewiseDensity1DTransitionKernel(
        target_log_prob_fn=target,
        score_fn=score,
        boundary=0.0,
        potential_jump=math.log(4.0),
        step_size=0.1,
    )
    samples, trace = tfp.mcmc.sample_chain(
        num_results=4,
        current_state=tf.constant([-0.5], tf.float64),
        kernel=kernel,
        num_burnin_steps=1,
        trace_fn=lambda _state, results: results,
        seed=[13, 17],
    )
    assert samples.shape == (4, 1)
    assert trace.log_accept_ratio.shape == (4, 1)
    assert trace.event_count.shape == (4, 1)
    assert bool(tf.reduce_all(tf.math.is_finite(samples)).numpy())
