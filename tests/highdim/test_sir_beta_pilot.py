"""Tiny CPU diagnostic proving pilot wiring to shared proposal densities."""

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_tf import (
    ProposalControls, build_maps, component_densities, draw_from_maps, observation_density,
)
from bayesfilter.highdim.ledh_backward_guide_tf import FrozenQuadraticBackwardGuide
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
from bayesfilter.score_study.sir_beta_calibration_diagnostic_tf import make_context_pilot


def test_pilot_matches_shared_map_and_density_calls():
    dtype = tf.float64
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(dtype)
    initial = tf.random.stateless_normal([32, 18], [991, 1], dtype=dtype)
    states = spec.initial_cloud(theta, initial, tf.zeros_like(theta))[0]
    ancestors = tf.stack([states, states])
    model, _ = spec.model(theta, tf.zeros_like(theta))
    observations = model.observation_fn(tf.reduce_mean(ancestors, axis=1))
    noise = tf.random.stateless_normal([2, 32, 18], [991, 2], dtype=dtype)
    uniforms = tf.random.stateless_uniform([2, 32, 2], [991, 3], dtype=dtype)
    controls = ProposalControls(proposal_policy="original_three_branch", reset_epsilon=1., reset_steps=5)
    guide = FrozenQuadraticBackwardGuide(tf.stack([.001 * tf.eye(18, dtype=dtype), tf.zeros([18, 18], dtype)]),
                                         tf.zeros([2, 18], dtype))
    result = make_context_pilot(spec, 32, 2, controls, guide, dtype, jit_compile=False)(
        theta, observations, ancestors, noise, uniforms)
    assert bool(tf.reduce_all(result["valid"]))
    for time in range(2):
        maps = build_maps(model, theta, ancestors[time], tf.zeros_like(states), observations[time],
                          controls, backward_guide=guide, time=tf.constant(time))
        child, tangent = draw_from_maps(maps, tf.constant([1 / 3] * 3, dtype), uniforms[time], noise[time])
        densities = component_densities(child, tangent, *maps[:4])[0]
        target = observation_density(model, theta, child, tangent, observations[time])[0] + densities[:, 0]
        np.testing.assert_allclose(result["log_components"][time], densities, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(result["log_target"][time], target, rtol=2e-12, atol=2e-12)
