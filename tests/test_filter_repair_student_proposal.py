"""Pinned stream, Gaussian conditioning and compiled Student proposal checks."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import (
    c2_transformed_observation_student_proposal_tf as candidate,
)
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _proposal(module, nu, dimension):
    return module.build_c2_transformed_observation_student_proposal(
        transition_matrix=tf.eye(dimension, dtype=D) * .6 + .02,
        process_covariance=tf.eye(dimension, dtype=D) * .3 + .01,
        observation=tf.linspace(tf.constant(.1, D), .3, dimension),
        theta_reference=tf.constant([.6, -.91], D), nu=nu, time_index=1)


@pytest.mark.parametrize("seed,nu,dimension", [((193, 17), 8., 2), ((-17, 41), 3.5, 4),
    ((2**36 + 3, -2**35), 5., 3)])
@pytest.mark.parametrize("jit", [False, True])
def test_complete_seeded_student_matches_pinned_values_and_density(seed, nu, dimension, jit):
    before = _proposal(_original("c2_transformed_observation_student_proposal_tf"), nu, dimension)
    after = _proposal(candidate, nu, dimension)
    count = 11
    parents = tf.reshape(tf.linspace(tf.constant(-.4, D), .6, count * dimension), [count, dimension])
    expected = before.sample_with_seed(parents, count, seed, jit_compile=jit)
    actual = after.sample_with_seed(parents, count, seed, jit_compile=jit)
    for field in ("gain", "posterior_covariance", "scale", "chol", "transformed_observation"):
        np.testing.assert_allclose(getattr(after, field), getattr(before, field), atol=1e-10, rtol=1e-10)
    for field in ("physical_points", "physical_log_density"):
        np.testing.assert_allclose(actual[field], expected[field], atol=1e-10, rtol=1e-10)
    assert bool(actual["finite"]) == bool(expected["finite"]) is True
    np.testing.assert_allclose(after.log_density(actual["physical_points"], parents),
                               actual["physical_log_density"], atol=1e-10, rtol=1e-10)
    call = after.compiled_seeded_sampler(count, jit_compile=jit)
    assert call.experimental_get_tracing_count() == 1
    _graph(call)
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(parents, tf.constant(seed, tf.int64))(stage="hlo")
    for program, inputs in ((candidate._geometry_program(dimension, nu), (after.process_covariance,)),
        (after._compiled_evaluation(count, "mean"), (parents,)),
        (after._compiled_evaluation(count, "density"), (actual["physical_points"], parents))):
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
        _graph(program)
