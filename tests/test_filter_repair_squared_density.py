"""Frozen-source parity for compiled public squared-TT numerical programs."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.squared_tt_density_native_tf import density_program


@pytest.fixture(scope="module")
def original():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.check_output([
        "git", "show", "3582b4ac:bayesfilter/highdim/squared_tt.py"], cwd=root, text=True)
    spec = importlib.util.spec_from_loader("squared_density_original_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, "frozen_squared_density_reference.py", "exec"), module.__dict__)
    return module


def _density(dimension, lebesgue=False):
    convention = highdim.MeasureConvention(
        density_measure=(highdim.DensityMeasure.REFERENCE_LEBESGUE if lebesgue else highdim.DensityMeasure.REFERENCE_MEASURE),
        mass_measure=(highdim.MassMeasure.REFERENCE_LEBESGUE if lebesgue else highdim.MassMeasure.REFERENCE_MEASURE),
        reference_weight_name="omega")
    product = highdim.ProductBasis([
        highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0 + i), 1+i % 2)
        for i in range(dimension)], convention)
    ranks = [1] + [2] * (dimension-1) + [1]
    cores = [highdim.TTCore(tf.random.stateless_normal(
        [ranks[i], 2+i % 2, ranks[i+1]], [872, i], dtype=tf.float64)) for i in range(dimension)]
    ftt = highdim.FunctionalTT(cores, product, convention)
    defensive = highdim.TensorProductReferenceDensity(product, convention, tf.constant(0.15, tf.float64))
    arguments = dict(sqrt_tt=ftt, defensive_density=defensive,
        tau=tf.constant(0.02, tf.float64), normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64), measure_convention=convention)
    return highdim.SquaredTTDensity(**arguments,
        branch_identity=highdim.SquaredTTDensity.expected_branch_identity(**arguments))


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("lebesgue", [False, True])
def test_complete_density_marginal_and_grid_conditional_preserve_frozen_source(original, dimension, lebesgue):
    actual = _density(dimension, lebesgue)
    arguments = dict(actual.__dict__)
    arguments["defensive_density"] = original.TensorProductReferenceDensity(
        actual.sqrt_tt.product_basis, actual.measure_convention, actual.defensive_density.floor)
    expected = original.SquaredTTDensity(**arguments)
    points = tf.random.stateless_uniform([7, dimension], [489, 1], minval=-0.7, maxval=0.8, dtype=tf.float64)
    tf.debugging.assert_near(actual.normalizer(), expected.normalizer(), atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(actual.log_density(points), expected.log_density(points), atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(actual.normalized_retained_density_values(tuple(range(dimension)), points),
        expected.normalized_retained_density_values(tuple(range(dimension)), points), atol=1e-10, rtol=1e-10)
    for axes in ((), (0,), (dimension-1,), tuple(range(dimension))):
        selected = tf.gather(points, tf.constant(axes, tf.int32), axis=1)
        if not axes:
            for density in (actual, expected):
                with pytest.raises(ValueError, match="ProductBasis requires"):
                    density.normalized_marginal_density_values(axes, selected)
            continue
        tf.debugging.assert_near(actual.normalized_marginal_density_values(axes, selected),
            expected.normalized_marginal_density_values(axes, selected), atol=1e-10, rtol=1e-10)
    for axis in range(dimension):
        prefix = points[:1, :axis]
        grid = tf.linspace(tf.constant(-1., tf.float64), tf.constant(1.+axis, tf.float64), 7)
        baseline = expected.conditional_density(axis, prefix, grid)
        for jit in (False, True):
            value = actual.conditional_density(axis, prefix, grid, jit_compile=jit)
            tf.debugging.assert_near(value, baseline, atol=1e-10, rtol=1e-10)


def test_conditional_enclosing_hlo_and_graph_size_are_bounded():
    density = _density(3)
    counts = []
    for size in (7, 19):
        program, inputs = density_program(density, "conditional", axis=0,
            prefix=tf.zeros([1, 0], tf.float64), grid=tf.linspace(tf.constant(-1., tf.float64), tf.constant(1., tf.float64), size))
        outputs = program(*inputs)
        assert bool(tf.reduce_all(tf.math.is_finite(outputs[0])))
        assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        assert not any(node.op in ("PyFunc", "EagerPyFunc") for node in nodes)
        counts.append(len(nodes))
        changed = (tuple(core * 1.02 for core in inputs[0]), *inputs[1:])
        assert abs(float(program(*changed)[2] - outputs[2])) > 1e-8
        assert program.experimental_get_tracing_count() == 1
    assert counts[0] == counts[1]


@pytest.mark.parametrize("operation,axes", [("normalized_retained", (0, 1)), ("normalized_marginal", (0,))])
def test_complete_normalized_result_is_inside_compiled_boundary(original, operation, axes):
    density = _density(2, lebesgue=True)
    arguments = dict(density.__dict__)
    arguments["defensive_density"] = original.TensorProductReferenceDensity(
        density.sqrt_tt.product_basis, density.measure_convention, density.defensive_density.floor)
    reference = original.SquaredTTDensity(**arguments)
    points = tf.fill([7, len(axes)], tf.constant(0.3, tf.float64))
    program, inputs = density_program(density, operation, points=points, keep_axes=axes)
    actual = program(*inputs)[0]
    expected = (reference.normalized_retained_density_values(axes, points)
                if operation == "normalized_retained"
                else reference.normalized_marginal_density_values(axes, points))
    tf.debugging.assert_near(actual, expected, rtol=1e-10, atol=1e-10)
    assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert program.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("lebesgue", [False, True])
def test_defensive_marginal_helper_preserves_frozen_source(original, dimension, lebesgue):
    actual = _density(dimension, lebesgue)
    arguments = dict(actual.__dict__)
    arguments["defensive_density"] = original.TensorProductReferenceDensity(
        actual.sqrt_tt.product_basis, actual.measure_convention, actual.defensive_density.floor)
    expected = original.SquaredTTDensity(**arguments)
    points = tf.random.stateless_uniform([7, dimension], [914, 2], dtype=tf.float64)
    for axes in ((), (0,), (dimension - 1,), tuple(range(dimension))):
        selected = tf.gather(points, tf.constant(axes, tf.int32), axis=1)
        if not axes:
            for density in (actual, expected):
                with pytest.raises(ValueError, match="ProductBasis requires"):
                    density._defensive_marginal_values(axes, selected)
            continue
        tf.debugging.assert_near(actual._defensive_marginal_values(axes, selected),
            expected._defensive_marginal_values(axes, selected), rtol=1e-10, atol=1e-10)
    program, inputs = density_program(actual, "defensive_marginal",
        keep_axes=(0,), points=points[:, :1])
    assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")


def test_custom_defensive_marginal_does_not_evaluate_unneeded_normalizer(original):
    class CustomDefensive:
        def log_density(self, points):
            return -tf.reduce_sum(tf.square(points), axis=1)

        def normalizer(self, measure):
            raise AssertionError("defensive marginal must not calculate the normalizer")

        def manifest_payload(self):
            return {"family": "test_custom_defensive"}

    arguments = dict(_density(2).__dict__)
    arguments["defensive_density"] = CustomDefensive()
    arguments.pop("branch_identity")
    arguments["branch_identity"] = highdim.SquaredTTDensity.expected_branch_identity(**arguments)
    actual = highdim.SquaredTTDensity(**arguments)
    expected = original.SquaredTTDensity(**arguments)
    points = tf.constant([[0.2, -0.4], [0.1, 0.3]], tf.float64)
    tf.debugging.assert_near(actual._defensive_marginal_values((0, 1), points),
        expected._defensive_marginal_values((0, 1), points), rtol=1e-10, atol=1e-10)
    for density in (actual, expected):
        with pytest.raises(NotImplementedError, match="requires tensor-product reference density"):
            density._defensive_marginal_values((0,), points[:, :1])


@pytest.mark.parametrize("jit", [False, True])
def test_retained_chunks_preserve_partial_block_and_bounded_graph(original, jit):
    density = _density(2)
    arguments = dict(density.__dict__)
    arguments["defensive_density"] = original.TensorProductReferenceDensity(
        density.sqrt_tt.product_basis, density.measure_convention, density.defensive_density.floor)
    expected = original.SquaredTTDensity(**arguments)
    counts = []
    for size in (7, 19):
        points = tf.reshape(tf.linspace(tf.constant(-0.7, tf.float64),
            tf.constant(0.8, tf.float64), size * 2), [size, 2])
        program, inputs = density_program(density, "retained_chunked", points=points,
            chunk_size=3, jit_compile=jit)
        actual, z = program(*inputs)
        tf.debugging.assert_near(actual, expected.normalized_retained_density_values((0, 1), points),
                                rtol=1e-10, atol=1e-10)
        tf.debugging.assert_near(z, expected.normalizer(), rtol=1e-10, atol=1e-10)
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        counts.append(len(nodes))
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        if jit:
            assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert counts[0] == counts[1]
