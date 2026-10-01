"""Pinned complete-date replay checks for the frozen source-route adaptation."""

from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_sequential_tf as native
from tests.highdim.test_p57_m6_sequential_fixed_hmc_source_loop import (
    _convention,
    _specs,
)
from tests.test_filter_repair_remaining_routes import _original
from tests.test_filter_repair_source_runtime import _retained_outputs, _route

D = tf.float64


def _clone(module, specs):
    def clone(spec):
        target = module.SourceRouteTarget(**{
            **spec.target.__dict__,
            "coordinate_frame": module.SourceRouteCoordinateFrame(**spec.target.coordinate_frame.__dict__),
        })
        return module.SourceRouteSequentialStepSpec(**{
            **spec.__dict__, "target": target,
            "transport": module.SourceRouteTransportProtocol(spec.transport.transport_object),
            "density_components": module.SourceRouteSequentialDensityComponents(**spec.density_components.__dict__),
        })
    return tuple(clone(spec) for spec in specs)


def _outputs(result):
    return tuple((*_retained_outputs(step.retained_samples),
        *((step.previous_marginal_density.physical_points,
           step.previous_marginal_density.local_points,
           step.previous_marginal_density.log_density)
          if step.previous_marginal_density is not None else ())) for step in result.steps)


@pytest.mark.parametrize("dates,heterogeneous", [(2, False), (4, False), (2, True)])
def test_complete_source_date_loop_preserves_pinned_values_and_query_gradients(dates, heterogeneous):
    first, second = _specs()
    specs = (first, *(replace(second, target=replace(second.target, time_index=index + 1),
        reference_samples=second.reference_samples + .03 * (index - 1)) for index in range(1, dates)))
    if heterogeneous:
        specs = (*specs[:-1], replace(specs[-1], reference_samples=specs[-1].reference_samples[:, :2]))
    before = _original("source_route")
    original_specs = _clone(before, specs)
    queries = tuple(spec.reference_samples for spec in specs)
    with tf.GradientTape() as tape:
        tape.watch(queries)
        expected = before.source_route_run_sequential_fixed_hmc(step_specs=original_specs)
        loss = tf.add_n([tf.reduce_sum(step.retained_samples.target_log_density) for step in expected.steps])
    expected_gradient = tape.gradient(loss, queries)
    with tf.GradientTape() as tape:
        tape.watch(queries)
        actual = candidate.source_route_run_sequential_fixed_hmc(step_specs=specs)
        loss = tf.add_n([tf.reduce_sum(step.retained_samples.target_log_density) for step in actual.steps])
    actual_gradient = tape.gradient(loss, queries)
    for value, reference in zip(tf.nest.flatten(_outputs(actual)), tf.nest.flatten(_outputs(expected)), strict=True):
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    for value, reference in zip(actual_gradient, expected_gradient, strict=True):
        assert value is not None and reference is not None
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(actual.log_marginal_likelihood, expected.log_marginal_likelihood,
                             atol=1e-10, rtol=1e-10)
    for index in range(1, dates):
        assert actual.steps[index].previous_retained_object is actual.steps[index-1].retained_object
        points = actual.steps[index].retained_object.samples
        with tf.GradientTape() as tape:
            tape.watch(points)
            value = actual.steps[index].target.negative_log_physical_density_fn(points)
            loss = tf.reduce_sum(value)
        gradient = tape.gradient(loss, points)
        with tf.GradientTape() as tape:
            tape.watch(points)
            reference = expected.steps[index].target.negative_log_physical_density_fn(points)
            loss = tf.reduce_sum(reference)
        reference_gradient = tape.gradient(loss, points)
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
        tf.debugging.assert_near(gradient, reference_gradient, atol=1e-10, rtol=1e-10)
    program, _, _ = native.sequential_program(specs)
    assert "HloModule" in program.experimental_get_compiler_ir(*queries)(stage="hlo")
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    assert any(node.op in ("While", "StatelessWhile") for node in nodes)
    assert not any(node.op in ("PyFunc", "EagerPyFunc") for node in nodes)
    assert program.experimental_get_tracing_count() == 1
    invalid = (*queries[:-1], tf.fill(queries[-1].shape, tf.constant(float("nan"), D)))
    assert not bool(tf.reduce_all(tf.math.is_finite(program(*invalid)[-1])))


def test_complete_date_loop_executes_real_ttsirt_transports_and_previous_marginal():
    target, transport = _route(candidate, "ttsirt")
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], D)
    components = candidate.SourceRouteSequentialDensityComponents(parameter_dim=0, state_dim=1,
        transition_log_density_fn=lambda x, t: -.1 * (x[0] - x[1])**2,
        likelihood_log_density_fn=lambda x, t: -.2 * x[0]**2,
        prior_log_density_fn=lambda x: -.1 * x[0]**2)
    first = candidate.SourceRouteSequentialStepSpec(target=target, transport=transport,
        reference_samples=points, measure_convention=_convention(), density_components=components)
    second = replace(first, target=replace(target, time_index=2), reference_samples=1.-points,
        density_components=replace(components, prior_log_density_fn=None),
        previous_marginal_keep_axes=(0,), previous_marginal_input_axes=(1,))
    specs = first, second
    before = _original("source_route")
    expected = before.source_route_run_sequential_fixed_hmc(step_specs=_clone(before, specs))
    actual = candidate.source_route_run_sequential_fixed_hmc(step_specs=specs)
    for value, reference in zip(tf.nest.flatten(_outputs(actual)), tf.nest.flatten(_outputs(expected)), strict=True):
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    program, _, _ = native.sequential_program(specs)
    assert "HloModule" in program.experimental_get_compiler_ir(points, 1.-points)(stage="hlo")


def test_repeated_transport_keeps_all_captured_core_frame_and_callback_derivatives():
    target, transport = _route(candidate, "ttsirt")
    points = tf.constant([[.2, .5, .8], [.3, .7, .4]], D)
    coefficient = tf.Variable(.2, dtype=D)
    components = candidate.SourceRouteSequentialDensityComponents(parameter_dim=0, state_dim=1,
        transition_log_density_fn=lambda x, t: -.1 * t * (x[0] - x[1])**2,
        likelihood_log_density_fn=lambda x, t: -coefficient * x[0]**2,
        prior_log_density_fn=lambda x: -.1 * x[0]**2)
    first = candidate.SourceRouteSequentialStepSpec(target=target, transport=transport,
        reference_samples=points, measure_convention=_convention(), density_components=components)
    specs = (first, *(replace(first, target=replace(target, time_index=index+1),
        density_components=replace(components, prior_log_density_fn=None),
        previous_marginal_keep_axes=(0,), previous_marginal_input_axes=(1,)) for index in range(1, 4)))
    before = _original("source_route")
    sources = (points, target.coordinate_frame.mu, target.coordinate_frame.matrix, coefficient,
               *(core.values for core in transport.transport_object.density.sqrt_tt.cores))
    # Keep the entire comparator numerical chain pinned. Mixing the old host
    # loop with new compiled density boundaries is not an independent oracle.
    legacy_tt = _original("tt")
    legacy_density = _original("squared_tt")
    legacy_density.FunctionalTT = legacy_tt.FunctionalTT
    legacy_density.TTContractedRepresentation = legacy_tt.TTContractedRepresentation
    density = transport.transport_object.density
    original_tt = legacy_tt.FunctionalTT(
        tuple(legacy_tt.TTCore(core.values) for core in density.sqrt_tt.cores),
        density.sqrt_tt.product_basis, density.measure_convention)
    original_density = legacy_density.SquaredTTDensity(**{**density.__dict__,
        "sqrt_tt": original_tt,
        "defensive_density": legacy_density.TensorProductReferenceDensity(
            density.sqrt_tt.product_basis, density.measure_convention, density.defensive_density.floor)})
    legacy_transport = _original("transport")
    legacy_transport.SquaredTTDensity = legacy_density.SquaredTTDensity
    legacy_transport.SquaredTTMarginal = legacy_density.SquaredTTMarginal
    before.SquaredTTMarginal = legacy_density.SquaredTTMarginal
    pinned_transport = legacy_transport.FixedTTSIRTTransport(
        original_density,
        legacy_transport.KRCDFConfig(**transport.transport_object.cdf_config.__dict__))
    original_specs = tuple(replace(spec, transport=before.SourceRouteTransportProtocol(pinned_transport))
                           for spec in _clone(before, specs))
    results = []
    for module, steps in ((before, original_specs), (candidate, specs)):
        with tf.GradientTape() as tape:
            tape.watch(sources)
            result = module.source_route_run_sequential_fixed_hmc(step_specs=steps)
            # Include all numerical fields, not just the target log density.
            loss = tf.add_n([tf.reduce_sum(value) for value in tf.nest.flatten(_outputs(result))])
        # Fixed-count bisection returns endpoints selected by comparisons, so
        # its finite program is locally constant in the uniform query. The
        # legacy eager tape reports that zero derivative as unconnected.
        results.append(tape.gradient(loss, sources, unconnected_gradients=tf.UnconnectedGradients.ZERO))
    for expected in results[0][1:]:
        assert bool(tf.reduce_any(expected != 0.))
    for actual, expected in zip(results[1], results[0], strict=True):
        assert actual is not None and expected is not None
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)


def test_changed_query_values_reuse_the_frozen_date_program():
    specs = _specs()
    changed = tuple(replace(spec, reference_samples=spec.reference_samples + .04) for spec in specs)
    program, _, _ = native.sequential_program(specs)
    reused, _, _ = native.sequential_program(changed)
    assert reused is program
    before = _original("source_route")
    expected = before.source_route_run_sequential_fixed_hmc(step_specs=_clone(before, changed))
    actual = candidate.source_route_run_sequential_fixed_hmc(step_specs=changed)
    for value, reference in zip(tf.nest.flatten(_outputs(actual)), tf.nest.flatten(_outputs(expected)), strict=True):
        tf.debugging.assert_near(value, reference, atol=1e-10, rtol=1e-10)
    assert program.experimental_get_tracing_count() == 1


def test_overridden_component_formula_is_rejected_before_silent_substitution():
    class DifferentFormula(candidate.SourceRouteSequentialDensityComponents):
        def negative_log_physical_density(self, **kwargs):
            return super().negative_log_physical_density(**kwargs) + .25

    first, second = _specs()
    modified = replace(first, density_components=DifferentFormula(**first.density_components.__dict__))
    with pytest.raises(TypeError, match="declared density-component formula"):
        candidate.source_route_run_sequential_fixed_hmc(step_specs=(modified, second))
