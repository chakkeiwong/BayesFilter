"""Pinned source-route endpoint and independent enclosing-gradient checks."""

from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_runtime_tf as native
from tests.highdim.test_p57_m6_sequential_fixed_hmc_source_loop import (
    ShiftedGaussianTransport,
    _convention,
)
from tests.highdim.test_zhao_cui_frozen_ttsirt_apf_compiler import _correlated_transport
from tests.test_filter_repair_remaining_routes import _original

D = tf.float64


@pytest.mark.parametrize("variable", [False, True])
def test_compiled_route_pullback_preserves_captured_numerical_inputs(variable):
    coefficient = tf.constant([.3, -.4], D)
    if variable:
        coefficient = tf.Variable(coefficient)
    query = tf.constant([[.2, .5], [.3, .7]], D)

    def numerical(points):
        value = tf.foldl(lambda state, row: state + tf.reduce_sum(coefficient * row)**2,
                         points, initializer=tf.zeros([], D), parallel_iterations=1)
        return value, coefficient * tf.reduce_sum(points, axis=0)

    compiled = native._query_program(numerical, query.shape, True)
    gradients = []
    for function in (numerical, compiled):
        with tf.GradientTape() as tape:
            tape.watch((query, coefficient))
            values = function(query)
            loss = tf.add_n([tf.reduce_sum(value) for value in values])
        gradients.append(tape.gradient(loss, (query, coefficient)))
    for actual, expected in zip(gradients[1], gradients[0], strict=True):
        assert actual is not None
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)


def _route(module, family):
    frame = module.SourceRouteCoordinateFrame(mu=tf.constant([.1, -.2], D),
        matrix=tf.constant([[1.2, .1], [0., .7]], D), expansion_factor=1.)
    target = module.build_source_route_target(
        negative_log_physical_density_fn=lambda x: .4 * tf.reduce_sum(x*x, axis=0),
        coordinate_frame=frame, shift_constant=tf.constant(.2, D), time_index=1)
    if family == "ttsirt":
        transport = _correlated_transport()
        transport = replace(transport, cdf_config=replace(transport.cdf_config,
                                                        grid_size=9, bisection_steps=8))
    else:
        transport = ShiftedGaussianTransport(tf.constant([.2, -.1], D), .35)
    return target, module.SourceRouteTransportProtocol(transport)


def _retained_outputs(value):
    return (value.retained_batch.samples, value.proposal_log_density,
        value.target_log_density, value.correction_log_weights, value.retained_batch.log_weights,
        value.diagnostics.effective_sample_size, value.normalizer.log_transport_normalizer)


@pytest.mark.parametrize("variable", [False, True])
def test_complete_retained_pullback_preserves_query_frame_and_callback_coefficients(variable):
    before = _original("source_route")
    target, transport = _route(candidate, "gaussian")
    coefficient = tf.constant(.4, D)
    if variable:
        coefficient = tf.Variable(coefficient)
    target = replace(target, negative_log_physical_density_fn=lambda x:
                     coefficient * tf.reduce_sum(x*x, axis=0))
    reference = before.SourceRouteTarget(**{
        **target.__dict__,
        "coordinate_frame": before.SourceRouteCoordinateFrame(**target.coordinate_frame.__dict__),
    })
    reference_transport = before.SourceRouteTransportProtocol(transport.transport_object)
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], D)
    inputs = (points, target.coordinate_frame.mu, target.coordinate_frame.matrix, coefficient)
    results = []
    for module, route, proposal in ((before, reference, reference_transport), (candidate, target, transport)):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            outputs = _retained_outputs(module.source_route_generate_retained_samples(
                target=route, transport=proposal, reference_samples=points, time_index=1))
            loss = tf.add_n([tf.reduce_sum(value) for value in outputs])
        results.append(tape.gradient(loss, inputs))
    for actual, expected in zip(results[1], results[0], strict=True):
        assert actual is not None and expected is not None
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("family", ["gaussian", "ttsirt"])
def test_retained_complete_default_and_enclosing_xla_preserve_pinned_values(family):
    before = _original("source_route")
    target, transport = _route(candidate, family)
    reference_target, reference_transport = _route(before, family)
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], D)

    def evaluate(module, target, transport, query):
        return _retained_outputs(module.source_route_generate_retained_samples(
            target=target, transport=transport, reference_samples=query, time_index=1))

    expected = evaluate(before, reference_target, reference_transport, points)
    graph = tf.function(lambda query: evaluate(candidate, target, transport, query),
        input_signature=[tf.TensorSpec(points.shape, D)], jit_compile=False, autograph=False)
    xla = tf.function(lambda query: evaluate(candidate, target, transport, query),
        input_signature=[tf.TensorSpec(points.shape, D)], jit_compile=True, autograph=False)
    for actual in (evaluate(candidate, target, transport, points), graph(points), xla(points)):
        for value, reference in zip(actual, expected, strict=True):
            tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    compiled = native.retained_program(target, transport, points.shape)
    assert "HloModule" in compiled.experimental_get_compiler_ir(points)(stage="hlo")
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    assert compiled.experimental_get_tracing_count() == 1
    invalid = tf.fill(points.shape, tf.constant(float("nan"), D))
    with pytest.raises(ValueError, match="NONFINITE"):
        evaluate(candidate, target, transport, invalid)
    assert not bool(tf.reduce_all(tf.math.is_finite(xla(invalid)[4])))


@pytest.mark.parametrize("family", ["gaussian", "ttsirt"])
def test_previous_marginal_first_trace_preserves_full_query_derivative(family):
    target, transport = _route(candidate, family)
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], D)
    components = candidate.SourceRouteSequentialDensityComponents(parameter_dim=0, state_dim=1,
        transition_log_density_fn=lambda x, t: -.1 * x[0]**2,
        likelihood_log_density_fn=lambda x, t: -.2 * x[0]**2,
        prior_log_density_fn=lambda x: -.1 * x[0]**2)
    spec = candidate.SourceRouteSequentialStepSpec(target=target, transport=transport,
        reference_samples=points, measure_convention=_convention(), density_components=components)
    previous = candidate._p59_retained_object_from_spec(spec)
    query = tf.constant([[-.3, .1, .6]], D)
    # Force the public boundary to prepare immutable marginal metadata on its
    # first enclosing trace, before any eager warm-up of that boundary.
    native._MARGINALS.clear()
    native._PROGRAMS.clear()

    def evaluate(points):
        with tf.GradientTape() as tape:
            tape.watch(points)
            value = candidate.source_route_previous_marginal_log_density(
                previous_retained_object=previous, physical_points=points, keep_axes=(0,))
            loss = tf.reduce_sum(value.log_density)
        return value.local_points, value.log_density, tape.gradient(loss, points)

    xla = tf.function(evaluate, input_signature=[tf.TensorSpec(query.shape, D)],
                      jit_compile=True, autograph=False)
    actual = xla(query)
    expected = evaluate(query)
    for value, reference in zip(actual, expected, strict=True):
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    step = 1e-5
    difference = (evaluate(query + step)[1] - evaluate(query - step)[1]) / (2 * step)
    tf.debugging.assert_near(actual[2][0], difference, atol=1e-8, rtol=1e-7)
    assert "HloModule" in xla.experimental_get_compiler_ir(query)(stage="hlo")
    assert xla.experimental_get_tracing_count() == 1
    invalid = tf.fill(query.shape, tf.constant(float("nan"), D))
    with pytest.raises(ValueError, match="NONFINITE"):
        evaluate(invalid)
    assert not bool(tf.reduce_all(tf.math.is_finite(xla(invalid)[1])))
