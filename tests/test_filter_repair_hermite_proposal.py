"""Pinned implementation and independent checks for the C2 proposal repair."""

from dataclasses import replace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as candidate
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _proposal(module, dimension):
    ranks = (1, *([2] * (dimension - 1)), 3)
    cores = tuple(tf.reshape(.2 + .1 * tf.sin(tf.cast(tf.range(ranks[i] * 3 * ranks[i+1]), D)),
                             [ranks[i], 3, ranks[i+1]]) for i in range(dimension))
    gram = tf.linalg.diag(tf.constant([1.2, .8, 1.1], D))
    authority = _original("c2_gaussian_hermite_proposal_tf")
    z_h = authority._paired_right_environments(cores, gram)[0][0, 0]
    return module.GaussianHermiteRetainedProposal(cores, gram, z_h, tf.constant(.04, D),
        tf.linspace(tf.constant(-.1, D), .2, dimension), tf.eye(dimension, dtype=D),
        5., 1, "a" * 64)


@pytest.mark.parametrize("degree", [0, 2, 6])
@pytest.mark.parametrize("jit", [False, True])
def test_gram_preserves_pinned_sum(degree, jit):
    before = _original("c2_gaussian_hermite_proposal_tf")
    points = tf.constant([-2.1, -.35, 0., 1.6, 24.], D)
    signature = [tf.TensorSpec(points.shape, D)]
    after_call = tf.function(lambda x: candidate.normalized_hermite_incomplete_gram(x, degree),
                            input_signature=signature, jit_compile=jit, autograph=False)
    before_call = tf.function(lambda x: before.normalized_hermite_incomplete_gram(x, degree),
                             input_signature=signature, jit_compile=jit, autograph=False)
    np.testing.assert_allclose(after_call(points), before_call(points), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(after_call(points), before.normalized_hermite_incomplete_gram(points, degree),
                               atol=1e-10, rtol=1e-10)
    if jit:
        assert "HloModule" in after_call.experimental_get_compiler_ir(points)(stage="hlo")
    _graph(after_call)


@pytest.mark.parametrize("dimension", [2, 4])
@pytest.mark.parametrize("jit", [False, True])
def test_complete_heterogeneous_rank_proposal_preserves_baseline(dimension, jit):
    before = _proposal(_original("c2_gaussian_hermite_proposal_tf"), dimension)
    after = _proposal(candidate, dimension)
    count = 8
    inputs = (tf.linspace(tf.constant(.1, D), .9, count),
              tf.reshape(tf.linspace(tf.constant(.03, D), .97, count * dimension), [count, dimension]),
              tf.reshape(tf.sin(tf.cast(tf.range(count * dimension), D)), [count, dimension]))
    before_call, after_call = before.compiled_sampler(count, jit_compile=jit), after.compiled_sampler(count, jit_compile=jit)
    expected, actual = before_call(*inputs), after_call(*inputs)
    assert bool(actual["cdf_bracket_valid"]) and bool(actual["finite"])
    for name in expected:
        if expected[name].dtype == tf.bool:
            np.testing.assert_array_equal(actual[name], expected[name])
        else:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-10, rtol=1e-10)
    if jit:
        assert "HloModule" in after_call.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(after_call)


def test_polynomial_recurrence_graph_does_not_grow_with_degree():
    counts = []
    for degree in (2, 6):
        call = candidate._gram_program((5,), degree)
        counts.append(len(_graph(call)))
    assert counts[0] == counts[1]


def test_default_sampler_has_one_fixed_signature():
    proposal = _proposal(candidate, 2)
    inputs = (tf.constant([.2, .8], D), tf.constant([[.25, .75], [.4, .6]], D), tf.zeros([2, 2], D))
    program = proposal.compiled_sampler(2)
    assert program._jit_compile is True
    actual = proposal.sample_physical(*inputs)
    reference = proposal.sample_reference(*inputs)
    np.testing.assert_allclose(actual["reference_points"], reference["reference_points"], atol=1e-10, rtol=1e-10)
    proposal.sample_physical(*inputs)
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("dimension", [2, 4])
def test_default_density_and_environment_programs_preserve_pinned_values(dimension):
    before = _proposal(_original("c2_gaussian_hermite_proposal_tf"), dimension)
    after = _proposal(candidate, dimension)
    points = tf.reshape(tf.linspace(tf.constant(-.3, D), .4, 3 * dimension), [3, dimension])
    for name in ("reference_quadratic_form", "reference_log_density", "physical_log_density"):
        np.testing.assert_allclose(getattr(after, name)(points), getattr(before, name)(points),
                                   rtol=1e-10, atol=1e-10)
        call = after._compiled_density(points.shape, name)
        assert "HloModule" in call.experimental_get_compiler_ir(points)(stage="hlo")
        _graph(call)
    values = (*after.prefix_core_values, after.suffix_gram)
    call = candidate._environments_program(tuple(tf.TensorSpec(x.shape, D) for x in values))
    assert "HloModule" in call.experimental_get_compiler_ir(*values)(stage="hlo")
    _graph(call)


@pytest.mark.parametrize("nu", [None, 1., 2., 5.])
@pytest.mark.parametrize("seed,count", [((713, 29), 8), ((-17, 83), 31), ((2**36 + 3, -2**35), 4)])
def test_complete_proposal_random_inputs_preserve_existing_draws_and_hlo(nu, seed, count):
    proposal = replace(_proposal(candidate, 3), defensive_nu=nu)
    expected = _original("c2_gaussian_hermite_proposal_tf").stateless_proposal_random_inputs(
        proposal, count, seed)
    actual = candidate.stateless_proposal_random_inputs(proposal, count, seed)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-10, rtol=1e-10)
    call = candidate._random_inputs_program(count, 3, nu)
    assert "HloModule" in call.experimental_get_compiler_ir(tf.constant(seed, tf.int64))(stage="hlo")
    _graph(call)
    changed = candidate.stateless_proposal_random_inputs(proposal, count, (seed[0] + 1, seed[1]))
    assert float(tf.reduce_max(tf.abs(changed[2] - actual[2]))) > 1e-5
    assert call.experimental_get_tracing_count() == 1
