"""Independent seeded-stream and enclosing-XLA preparation checks."""

import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.tt_native_control_tf import random_core_start_program
from bayesfilter.ops.stateless_random_tf import philox_normal_float64


@pytest.mark.parametrize("shape", [(7,), (4, 8), (1, 3, 2)])
def test_normal_conversion_preserves_non_xla_seeded_draws(shape):
    call = tf.function(lambda seed: philox_normal_float64(shape, seed),
        input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    for seed in ([71, 7000], [7781, 7033], [-1, 17]):
        seed = tf.constant(seed, tf.int32)
        with tf.device("/CPU:0"):
            expected = tf.random.stateless_normal(shape, seed, dtype=tf.float64, alg="philox")
        np.testing.assert_allclose(call(seed), expected, atol=2e-14, rtol=2e-14)
    assert call.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("count", [1, 4])
def test_core_start_preparation_preserves_each_core_and_date(count):
    shapes = ((1, 3, 2), (2, 2, 3), (3, 5, 1))
    dates, seed = tf.range(1, count + 1), tf.constant(7781)
    call = random_core_start_program(shapes, count)
    actual = call(dates, seed)
    maximum = max(math.prod(shape) for shape in shapes)
    for index, date in enumerate(dates):
        for axis, shape in enumerate(shapes):
            expected = .3 * tf.random.stateless_normal(shape,
                tf.stack([seed, 7000 + 31 * date + axis]), dtype=tf.float64)
            expected = tf.pad(tf.reshape(expected, [-1]), [[0, maximum - math.prod(shape)]])
            np.testing.assert_allclose(actual[index, axis], expected, atol=1e-14, rtol=1e-14)
    assert "HloModule" in call.experimental_get_compiler_ir(dates, seed)(stage="hlo")
    assert call.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("row_design", ["mc", "sobol"])
@pytest.mark.parametrize("dimension,degree", [(2, None), (3, 2), (7, 3)])
def test_design_bank_matches_original_rows_weights_and_status(row_design, dimension, degree):
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig, _design_rows
    from bayesfilter.highdim.squared_tt_engine_gaussian_tf import _christoffel_rows_tensor
    from bayesfilter.highdim.tt_preparation_tf import design_row_program

    config = EngineConfig(2, 2, 64, 2, 1e-10, 1e-6, 3., 7781, row_design=row_design)
    dates, seed = tf.constant([0, 3, 9]), tf.constant(config.seed)
    call = design_row_program(row_design, 64, dimension, 3, 17, gaussian_degree=degree)
    actual = call(seed, dates)
    expected = []
    for date in (0, 3, 9):
        if degree is None:
            expected.append(_design_rows(config, 64, dimension, (config.seed, 17 + date)))
        else:
            expected.append(_christoffel_rows_tensor(config, 64, dimension,
                (config.seed, 17 + date), degree))
    expected = tf.nest.map_structure(lambda *values: tf.stack(values), *expected)
    for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        if left.dtype == tf.bool or degree is None:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=1e-11, rtol=1e-12)
    assert "HloModule" in call.experimental_get_compiler_ir(seed, dates)(stage="hlo")
    call(seed + 1, dates + 2)
    assert call.experimental_get_tracing_count() == 1


def test_preparation_graph_size_is_independent_of_dates():
    from bayesfilter.highdim.tt_preparation_tf import design_row_program

    def nodes(call):
        graph = call.get_concrete_function().graph.as_graph_def()
        return len(graph.node) + sum(len(fn.node_def) for fn in graph.library.function)

    for factory in (
        lambda count: design_row_program("sobol", 32, 3, count, 100),
        lambda count: random_core_start_program(((1, 3, 2), (2, 3, 1)), count),
    ):
        assert nodes(factory(2)) == nodes(factory(5))


@pytest.mark.parametrize("jit", [False, True])
def test_heterogeneous_basis_preparation_is_compiled_and_reusable(jit):
    from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
    from bayesfilter.highdim.diagnostics import DensityMeasure, MassMeasure, MeasureConvention
    from bayesfilter.highdim.tt import TTCore
    from bayesfilter.highdim.tt_native_control_tf import (
        fixed_basis_program, fixed_basis_rows, make_fixed_squared_marginal,
    )

    basis = ProductBasis([LegendreBasis1D(BoundedInterval(-1., 1.), 2),
        LegendreBasis1D(BoundedInterval(-1., 1.), 3)], MeasureConvention(
            DensityMeasure.REFERENCE_MEASURE, MassMeasure.REFERENCE_MEASURE, "omega"))
    shapes = ((1, 3, 2), (2, 4, 1))
    cores = tuple(TTCore(tf.ones(shape, tf.float64)) for shape in shapes)
    points = tf.constant([[-.7, -.3], [-.1, .2], [.4, .8]], tf.float64)
    call = fixed_basis_program(basis, points.shape, shapes, jit_compile=jit)
    actual = fixed_basis_rows(basis, points, cores, jit_compile=jit)
    expected = np.stack([np.pad(np.polynomial.legendre.legvander(points[:, axis], degree)
        * np.sqrt(2 * np.arange(degree + 1) + 1), ((0, 0), (0, 3 - degree)))
        for axis, degree in enumerate((2, 3))])
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert call._jit_compile == jit and call.experimental_get_tracing_count() == 1
    assert call is fixed_basis_program(basis, points.shape, shapes, jit_compile=jit)
    shifted = call(points + .03)
    assert not np.array_equal(actual, shifted)
    assert call.experimental_get_tracing_count() == 1
    if jit:
        assert "HloModule" in call.experimental_get_compiler_ir(points)(stage="hlo")
    marginal = make_fixed_squared_marginal(basis, cores, (0,), points[:, :1], jit_compile=jit)
    preparation = marginal.preparation_program
    assert preparation._jit_compile == jit and preparation.experimental_get_tracing_count() == 1
    expected_phi = expected[0, :, :3]
    mass = basis.bases[1].mass_matrix(MassMeasure.REFERENCE_MEASURE)
    np.testing.assert_allclose(marginal(cores),
        np.square(np.sum(expected_phi, axis=1)) * 4 * np.sum(mass), atol=1e-12, rtol=1e-12)
    if jit:
        assert "HloModule" in preparation.experimental_get_compiler_ir(points[:, :1])(stage="hlo")

    outer = tf.function(lambda rows: fixed_basis_rows(basis, rows, cores),
        input_signature=[tf.TensorSpec(points.shape, points.dtype)], jit_compile=False, autograph=False)
    graph = outer.get_concrete_function().graph.as_graph_def()
    assert not any(fn.attr.get("_XlaMustCompile") and fn.attr["_XlaMustCompile"].b
                   for fn in graph.library.function)
