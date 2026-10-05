"""Diagnostic parity for UKF projection and deterministic TT channel packing."""

import inspect

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ukf_initializer as initializer
from bayesfilter.highdim.bases import (
    BoundedInterval,
    LagrangePiecewiseBasis1D,
    LegendreBasis1D,
    ProductBasis,
)
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.ukf_initializer_tf import (
    embedding_program,
    initializer_program,
    projection_program,
)
from tests.highdim.test_p76_ukf_initializer import _scout
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _basis(dimension, lebesgue=False):
    convention = MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_LEBESGUE if lebesgue else DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_LEBESGUE if lebesgue else MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    return ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1. + .04 * axis, 1. + .07 * axis),
                                            2 + axis % 3) for axis in range(dimension)), convention)


@pytest.mark.parametrize("dimension", [2, 5])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("lebesgue", [False, True])
def test_projection_matches_original_and_preserves_live_frames(dimension, jit, lebesgue):
    basis = _basis(dimension, lebesgue)
    kwargs = {"gamma": 2.3, "quadrature_order": 13,
              "center": tf.linspace(tf.constant(-0.1, D), .2, dimension),
              "linear_map": tf.eye(dimension, dtype=D) + .03,
              "reference_offset": tf.linspace(tf.constant(.2, D), -.1, dimension),
              "reference_matrix": .9 * tf.eye(dimension, dtype=D) + .02}
    expected = _original("ukf_initializer").p76_gaussian_sqrt_projection_coefficients(basis, **kwargs)
    actual = initializer.p76_gaussian_sqrt_projection_coefficients(basis, jit_compile=jit, **kwargs)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, atol=1e-10, rtol=1e-10)
    program = projection_program(basis, 13, jit_compile=jit)
    moved = initializer.p76_gaussian_sqrt_projection_coefficients(
        basis, jit_compile=jit, **{**kwargs, "center": kwargs["center"] + .1})
    assert np.max(np.abs(moved[0] - actual[0])) > 1e-5
    assert program.experimental_get_tracing_count() == 1
    _graph(program)
    if jit:
        inputs = (tf.constant(kwargs["gamma"], D), kwargs["center"], kwargs["linear_map"],
                  kwargs["reference_offset"], kwargs["reference_matrix"])
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")


@pytest.mark.parametrize("widths,ranks", [((1,), (1, 1)), ((3, 5), (1, 3, 1)),
                                       ((1, 4, 2, 5), (1, 4, 4, 4, 1))])
@pytest.mark.parametrize("jit", [False, True])
def test_seeded_channel_embedding_preserves_exact_coefficients(widths, ranks, jit):
    coefficients = tuple(tf.constant(np.linspace(-.3, .5, width) + .01 * axis, D)
                         for axis, width in enumerate(widths))
    kwargs = {"ranks": ranks, "seed_epsilon": 0.0031}
    expected = _original("ukf_initializer").p76_embed_rank_one_with_seeded_channels(coefficients, **kwargs)
    actual = initializer.p76_embed_rank_one_with_seeded_channels(coefficients, jit_compile=jit, **kwargs)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result.values, reference.values)
    program = embedding_program(widths, ranks, jit_compile=jit)
    _graph(program)
    if jit:
        packed = tf.stack([tf.pad(value, [[0, max(widths) - value.shape[0]]]) for value in coefficients])
        assert "HloModule" in program.experimental_get_compiler_ir(
            packed, tf.constant(kwargs["seed_epsilon"] / max(max(ranks) - 1, 1), D))(stage="hlo")


def test_initializer_defaults_and_projection_graph_size():
    assert inspect.signature(initializer.p76_gaussian_sqrt_projection_coefficients).parameters["jit_compile"].default
    assert inspect.signature(initializer.p76_embed_rank_one_with_seeded_channels).parameters["jit_compile"].default
    counts = [len(_graph(projection_program(_basis(size), 13))) for size in (3, 6)]
    assert counts[0] == counts[1]


@pytest.mark.parametrize("jit", [False, True])
def test_heterogeneous_projection_preserves_lagrange_basis_dispatch(jit):
    basis = _basis(2)
    basis = ProductBasis((basis.bases[0], LagrangePiecewiseBasis1D(BoundedInterval(-.8, 1.2), 2, 2)), basis.convention)
    kwargs = {"gamma": 1.3, "quadrature_order": 16}
    expected = _original("ukf_initializer").p76_gaussian_sqrt_projection_coefficients(basis, **kwargs)
    actual = initializer.p76_gaussian_sqrt_projection_coefficients(basis, jit_compile=jit, **kwargs)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("jit", [False, True])
def test_complete_initializer_preserves_frame_covariance_cores_and_manifest(jit):
    before = _original("ukf_initializer")
    basis, ranks = _basis(4), (1, 3, 3, 3, 1)
    kwargs = {"product_basis": basis, "ranks": ranks, "quadrature_order": 16}
    config = initializer.P76UKFInitializerConfig(**kwargs)
    expected = before.p76_build_ukf_initializer(_scout(), before.P76UKFInitializerConfig(**kwargs))
    actual = initializer.p76_build_ukf_initializer(_scout(), config, jit_compile=jit)
    for field in ("center", "stabilized_covariance", "raw_eigenvalues", "floored_eigenvalues"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=1e-10, atol=1e-10)
    # SelfAdjointEigV2 does not specify eigenvector signs across backends.
    signs = tf.sign(tf.reduce_sum(actual.linear_map * expected.linear_map, axis=0))
    np.testing.assert_array_equal(tf.abs(signs), tf.ones_like(signs))
    np.testing.assert_allclose(actual.linear_map * signs, expected.linear_map, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual.linear_map @ tf.transpose(actual.linear_map),
                               expected.linear_map @ tf.transpose(expected.linear_map), rtol=1e-10, atol=1e-10)
    for result, reference in zip(actual.cores, expected.cores, strict=True):
        np.testing.assert_allclose(result.values, reference.values, rtol=1e-10, atol=1e-10)
    for result, reference in zip(actual.projection_coefficients, expected.projection_coefficients, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-10)
    assert actual.manifest.keys() == expected.manifest.keys()
    assert actual.status == expected.status and actual.nonclaims == expected.nonclaims
    program = initializer_program(basis, ranks, 16, jit_compile=jit)
    _graph(program)
    if jit:
        moments = initializer.p76_adjacent_moments_from_scout(_scout(), time_index=config.time_index)
        inputs = (moments.center, moments.covariance, tf.constant(config.gamma, D),
                  tf.constant(config.covariance_abs_floor, D), tf.constant(config.covariance_rel_floor, D),
                  tf.zeros([4], D), tf.eye(4, dtype=D), tf.constant(config.seed_epsilon / 2, D))
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
