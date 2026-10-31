"""CPU-only reference checks for the diagnostic shared-consumer call chain."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import covariance_proposal_tf as core
from bayesfilter.highdim.ledh_backward_guide_tf import FrozenQuadraticBackwardGuide
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs


def test_wrappers_use_shared_guided_filter_and_density(monkeypatch):
    dtype = tf.float64
    controls = core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5, correction_steps=2)
    theta = tf.constant([.2, .2, .3, .5], dtype)
    observations = tf.constant([[.1, -.2], [.3, .1]], dtype)
    initial = tf.random.stateless_normal([12, 2], [962, 1], dtype=dtype)
    normals = tf.random.stateless_normal([2, 12, 2], [962, 2], dtype=dtype)
    uniforms = tf.random.stateless_uniform([2, 12, 2], [962, 3], dtype=dtype)
    beta = tf.constant([.2, .3, .5], dtype)
    precision = tf.constant([[[.2, 0.], [0., .3]], [[0., 0.], [0., 0.]]], dtype)
    linear = tf.constant([[.1, -.1], [0., 0.]], dtype)
    calls = []
    original = core.make_filter

    def observed(*args, **kwargs):
        calls.append(kwargs.get("backward_guide"))
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "make_filter", observed)
    evaluate, pilot = make_diagnostic_programs(12, 2, controls, jit_compile=False)
    arguments = (theta, observations, initial, normals, uniforms, beta, precision, linear)
    result = evaluate(*arguments)
    guide = FrozenQuadraticBackwardGuide(precision, linear)
    spec = LGSSMSpec("diagonal_ar", 2)
    direct = original(spec, 12, 2, controls, dtype, jit_compile=False, backward_guide=guide)
    assert bool(result["valid"])
    for coordinate in range(4):
        reference = direct(theta, tf.one_hot(coordinate, 4, dtype=dtype), observations,
                           initial, normals, uniforms, beta)
        np.testing.assert_allclose(result["score"][coordinate], reference["score"], atol=1e-11)
        np.testing.assert_allclose(result["value"], reference["value"], atol=1e-12)
        np.testing.assert_allclose(result["direction_trace"][coordinate], reference["trace"], atol=1e-11)
        assert bool(result["direction_valid"][coordinate]) == bool(reference["valid"])
    pilot_result = pilot(*arguments, normals, uniforms)
    assert bool(pilot_result["valid"])
    assert len(calls) >= 2 and all(isinstance(value, FrozenQuadraticBackwardGuide) for value in calls)
    model, _ = spec.model(theta, tf.zeros_like(theta))
    maps = core.build_maps(model, theta, initial, tf.zeros_like(initial), observations[0],
                           controls, backward_guide=guide, time=0)
    child, tangent = core.draw_from_maps(maps, tf.fill([3], tf.constant(1. / 3., dtype)),
                                        uniforms[0], normals[0])
    densities, _, _ = core.component_densities(child, tangent, *maps[:4])
    likelihood, _ = core.observation_density(model, theta, child, tangent, observations[0])
    np.testing.assert_allclose(pilot_result["log_components"][0], densities, atol=1e-12)
    np.testing.assert_allclose(pilot_result["log_target"][0], likelihood + densities[:, 0], atol=1e-12)
    assert bool(tf.reduce_all(result["trace"][:, core.TRACE_FIELDS.index("moment_valid")] == 1.))
    assert bool(tf.reduce_all(result["trace"][:, core.TRACE_FIELDS.index("correction_cap_valid")] == 1.))


@pytest.mark.parametrize("error,stretch,accepted", [
    (2e-13, False, True), (2e-13, True, True),
    (1e-4, False, False), (1e-4, True, False),
    (float("nan"), False, False),
])
def test_consumer_distinguishes_roundoff_from_broken_reset(monkeypatch, error, stretch, accepted):
    original = core.reset_cloud

    def perturbed_reset(*arguments):
        result = list(original(*arguments))
        particles, mean, covariance = result[0], result[2], result[4]
        if stretch:
            result[0] = mean + (particles - mean) * tf.constant(1. + error, particles.dtype)
        else:
            scale = tf.maximum(tf.reduce_max(tf.abs(mean)), tf.sqrt(tf.reduce_max(tf.linalg.diag_part(covariance))))
            result[0] = particles + tf.constant(error, particles.dtype) * scale
        return tuple(result)

    monkeypatch.setattr(core, "reset_cloud", perturbed_reset)
    dtype = tf.float64
    program = core.make_filter(LGSSMSpec("diagonal_ar", 2), 12, 1,
                               core.ProposalControls(proposal_policy="original_three_branch", reset_steps=5), dtype, jit_compile=False)
    theta = tf.constant([.2, .2, .3, .5], dtype)
    result = program(theta, tf.one_hot(2, 4, dtype=dtype),
                     tf.constant([[.1, -.2]], dtype),
                     tf.random.stateless_normal([12, 2], [963, 1], dtype=dtype),
                     tf.random.stateless_normal([1, 12, 2], [963, 2], dtype=dtype),
                     tf.random.stateless_uniform([1, 12, 2], [963, 3], dtype=dtype),
                     tf.constant([.2, .3, .5], dtype))
    assert bool(result["valid"]) is accepted
    assert bool(result["trace"][0, core.TRACE_FIELDS.index("moment_valid")]) is accepted
    if accepted:
        assert bool(result["trace"][0, core.TRACE_FIELDS.index("moment_roundoff_warning")])
