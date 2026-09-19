"""Shared coordinate execution against the pre-sharing transport and marginals."""

import importlib.util
import subprocess
from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim.squared_tt_density_native_tf import density_program
from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
from bayesfilter.highdim.ttsirt_coordinate_tf import masked_marginal_program
from bayesfilter.highdim.ttsirt_native_tf import transport_arguments, transport_program
from tests.highdim.test_zhao_cui_frozen_ttsirt_apf_compiler import _constant_transport
from tests.test_filter_repair_squared_density import _density

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "8bd5b30c:bayesfilter/highdim/ttsirt_native_tf.py"], text=True)
    spec = importlib.util.spec_from_loader("ttsirt_before_coordinate_sharing", loader=None)
    module = importlib.util.module_from_spec(spec)
    exec(compile(source, "ttsirt_before_coordinate_sharing.py", "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize("lebesgue", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_masked_marginal_preserves_heterogeneous_values_and_all_pullbacks(lebesgue, jit):
    density = _density(3, lebesgue)
    query = tf.reshape(tf.linspace(tf.constant(-.63, D), tf.constant(.71, D), 9), [3, 3])
    cores = tuple(core.values for core in density.sqrt_tt.cores)
    candidate = masked_marginal_program(density, 3, jit_compile=jit)
    for axes in ((0,), (1, 2), (0, 2), (0, 1, 2)):
        keep = tf.constant(tuple(axis in axes for axis in range(3)))
        authority = density_program(density, "marginal", keep_axes=axes,
            points=tf.TensorSpec([3, len(axes)], D), jit_compile=jit)[0]

        def scored(use_candidate, mask=keep, selected_axes=axes, reference_program=authority):
            @tf.function(input_signature=[tf.TensorSpec(query.shape, D),
                tuple(tf.TensorSpec(core.shape, D) for core in cores), tf.TensorSpec([], D)],
                jit_compile=jit, autograph=False)
            def evaluate(points, values, tau):
                with tf.GradientTape() as tape:
                    tape.watch((points, values, tau))
                    if use_candidate:
                        result = candidate.python_function(*values, tau, points, mask)
                    else:
                        selected = tf.gather(points, selected_axes, axis=1)
                        result = reference_program.python_function(values, tau, selected)[0]
                    loss = tf.reduce_sum(result * tf.constant([.3, -.2, .7], D))
                return result, tape.gradient(loss, (points, values, tau))

            return evaluate(query, cores, density.tau)

        actual, expected = scored(True), scored(False)
        for value, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
            assert value is not None and reference is not None
            tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
        if len(axes) < 3:
            omitted = tuple(axis for axis in range(3) if axis not in axes)
            tf.debugging.assert_equal(tf.gather(actual[1][0], omitted, axis=1),
                                      tf.zeros([3, len(omitted)], D))
    assert candidate.experimental_get_tracing_count() <= 1


@pytest.mark.parametrize("suffix", [False, True])
@pytest.mark.parametrize("jit", [False, True])
def test_shared_coordinate_preserves_pinned_transport_and_complete_score(previous, suffix, jit):
    transport = FixedTTSIRTTransport(_density(3, lebesgue=True),
        KRCDFConfig(grid_size=5, bisection_steps=4, monotonicity_tolerance=1e-10,
                    bracket_tolerance=1e-10, denominator_floor=1e-12, max_floor_count=0))
    query = tf.constant([[-.23, .42], [.37, -.16]], D)
    condition = tf.constant([[.21, -.13]], D)
    arguments = (*transport_arguments(transport), condition, query)
    results = []
    for create in (previous.transport_program, transport_program):
        program = create(transport, "log_jacobian_suffix" if suffix else "log_jacobian",
                         2, conditioning_dimension=1, jit_compile=jit)

        def scored(bound):
            @tf.function(input_signature=bound.input_signature, jit_compile=jit, autograph=False)
            def evaluate(cores, tau, normalizer_floor, denominator_floor, known, values):
                with tf.GradientTape() as tape:
                    tape.watch((cores, tau, known, values))
                    output, code = bound.python_function(cores, tau, normalizer_floor,
                                                        denominator_floor, known, values)
                    loss = tf.reduce_sum(output * tf.constant([.3, .7], D))
                return output, code, tape.gradient(loss, (cores, tau, known, values))

            return evaluate

        evaluated = scored(program)(*arguments)
        assert int(evaluated[1]) == 0
        results.append(evaluated)
    for actual, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0]), strict=True):
        assert actual is not None and expected is not None
        if actual.dtype.is_floating:
            tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)
        else:
            tf.debugging.assert_equal(actual, expected)


def test_shared_coordinate_graph_scales_with_dimension_and_preserves_uniform_cdf(record_property):
    counts = []
    for dimension in (4, 8):
        transport = _constant_transport(dimension)
        transport = replace(transport, cdf_config=replace(transport.cdf_config, grid_size=5, bisection_steps=4))
        program = transport_program(transport, "forward", 2)
        points = tf.reshape(tf.linspace(tf.constant(-.7, D), tf.constant(.8, D), dimension*2),
                            [dimension, 2])
        arguments = (*transport_arguments(transport), tf.zeros([0, 2], D), points)
        result, code = program(*arguments)
        assert int(code) == 0
        tf.debugging.assert_near(result, .5*(points+1.), atol=1e-12, rtol=1e-12)
        definition = program.get_concrete_function().graph.as_graph_def()
        nodes = [*definition.node, *(node for function in definition.library.function for node in function.node_def)]
        assert not {node.op for node in nodes} & {"PyFunc", "EagerPyFunc", "PyFuncStateless"}
        counts.append(len(nodes))
        assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        program(*arguments[:-1], points*.9)
        assert program.experimental_get_tracing_count() == 1
    record_property("dimension_4_8_graph_nodes", counts)
    assert counts[1] < 2.5*counts[0]
