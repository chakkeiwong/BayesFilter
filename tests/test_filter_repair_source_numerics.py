"""Pinned scalar/affine source-route checks, including enclosing XLA validity."""

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route as candidate
from bayesfilter.highdim import source_route_numerics_tf as native
from tests.test_filter_repair_remaining_routes import _graph, _original

D = tf.float64


def _weights(module, target, proposal, shift):
    correction = module.source_route_proposal_log_weights(log_target_density=target, log_proposal_density=proposal)
    return (correction, module.normalize_log_weights(correction),
        module.effective_sample_size_from_log_weights(correction),
        module.source_route_proposal_log_weights_from_negative_log_target(
            negative_log_target=-target, log_proposal_density=proposal),
        module.source_route_discrete_log_normalizer_from_correction(
            log_proposal_density=proposal, correction_log_weights=correction),
        module.source_route_equal_weight_log_normalizer_estimate(correction),
        module.source_route_shifted_negative_log_target(negative_log_target=-target, shift_constant=shift),
        module.source_route_log_normalizer_update(log_transport_normalizer=shift + 1., shift_constant=shift),
        module.source_route_residual_negative_log_target(full_negative_log_target=target,
            preconditioner_negative_log_target=proposal),
        module.source_route_preconditioned_target_identity_error(full_negative_log_target=target,
            preconditioner_negative_log_target=proposal, residual_negative_log_target=target - proposal))


@pytest.mark.parametrize("rows", [4, 8])
def test_public_weight_calculations_preserve_values_gradients_and_enclosing_graph(rows):
    before = _original("source_route")
    inputs = (tf.linspace(tf.constant(-.5, D), .3, rows),
              tf.linspace(tf.constant(.2, D), -.1, rows), tf.constant(.35, D))

    def endpoint(module, *values):
        with tf.GradientTape() as tape:
            tape.watch(values)
            outputs = _weights(module, *values)
            objective = tf.add_n([tf.reduce_sum(value) for value in outputs])
        return outputs, tape.gradient(objective, values)

    expected = endpoint(before, *inputs)
    specs = tuple(tf.TensorSpec(value.shape, D) for value in inputs)
    graph = tf.function(lambda *values: endpoint(candidate, *values), input_signature=specs,
                        jit_compile=False, autograph=False)
    xla = tf.function(lambda *values: endpoint(candidate, *values), input_signature=specs,
                      jit_compile=True, autograph=False)
    for actual in (endpoint(candidate, *inputs), graph(*inputs), xla(*inputs)):
        for value, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
            np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    assert xla.experimental_get_tracing_count() == 1
    assert "HloModule" in xla.experimental_get_compiler_ir(*inputs)(stage="hlo")
    _graph(xla)
    invalid = (tf.fill([rows], tf.constant(float("nan"), D)), *inputs[1:])
    with pytest.raises(ValueError, match="NONFINITE"):
        candidate.normalize_log_weights(invalid[0])
    assert not bool(tf.reduce_all(tf.math.is_finite(xla(*invalid)[0][1])))


@pytest.mark.parametrize("pivot", [False, True])
def test_affine_determinant_value_and_full_input_pullback(pivot):
    matrix = tf.constant([[1.3, .2], [-.1, .7]], D)
    if pivot:
        matrix = tf.reverse(matrix, [0])
    mu = tf.constant([.1, -.2], D)
    points = tf.constant([[-.5, .1, .2], [.7, -.6, .4]], D)
    inputs = matrix, mu, points

    def endpoint(matrix, mu, points, *, current):
        with tf.GradientTape() as tape:
            tape.watch((matrix, mu, points))
            if current:
                physical = native.physical_points(points, mu, matrix)
                log_density = native.reference_log_density(-tf.reduce_sum(physical**2, axis=0), matrix)
            else:
                physical = matrix @ points + mu[:, None]
                log_density = -tf.reduce_sum(physical**2, axis=0) + tf.math.log(tf.abs(tf.linalg.det(matrix)))
            value = tf.reduce_sum(log_density)
        return log_density, tape.gradient(value, (matrix, mu, points))

    expected = endpoint(*inputs, current=False)
    xla = tf.function(lambda *values: endpoint(*values, current=True),
        input_signature=tuple(tf.TensorSpec(value.shape, D) for value in inputs),
        jit_compile=True, autograph=False)
    for actual in (endpoint(*inputs, current=True), xla(*inputs)):
        for value, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected), strict=True):
            np.testing.assert_allclose(value, reference, atol=1e-10, rtol=1e-10)
    frame = candidate.SourceRouteCoordinateFrame(mu=mu, matrix=matrix, expansion_factor=1.)
    target = candidate.build_source_route_target(negative_log_physical_density_fn=lambda x: tf.reduce_sum(x*x, axis=0),
        coordinate_frame=frame, shift_constant=tf.zeros([], D), time_index=1)
    np.testing.assert_allclose(target.physical_points_from_reference(points), matrix @ points + mu[:, None])
    assert "HloModule" in xla.experimental_get_compiler_ir(*inputs)(stage="hlo")


def test_compiled_quantile_rejects_invalid_domain_and_preserves_derivative():
    point = tf.constant(.3, D)
    with tf.GradientTape() as tape:
        tape.watch(point)
        value = candidate.tfp_normal_quantile(point)
    expected = np.sqrt(2. * np.pi) * tf.exp(tf.square(value) / 2.)
    np.testing.assert_allclose(tape.gradient(value, point), expected, atol=1e-10, rtol=1e-10)
    for invalid in (0., 1., float("nan")):
        with pytest.raises(ValueError, match="NONFINITE"):
            candidate.tfp_normal_quantile(tf.constant(invalid, D))
    call = candidate.tfp_normal_quantile.compiled
    assert not bool(tf.math.is_finite(call(tf.constant(0., D))))
    assert "HloModule" in call.experimental_get_compiler_ir(point)(stage="hlo")


def test_complete_target_and_transport_methods_default_to_xla_and_keep_pullbacks():
    from tests.highdim.test_p57_m6_sequential_fixed_hmc_source_loop import _specs

    spec = _specs()[0]
    target, protocol, points = spec.target, spec.transport, spec.reference_samples
    methods = (
        (target, "negative_log_density", (points,), {}),
        (target, "log_target_density", (points,), {}),
        (protocol, "inverse_transport", (points,), {}),
        (protocol, "forward_transport", (points,), {}),
        (protocol, "log_reference_density", (points,), {}),
        (protocol, "eval_pdf", (points,), {}),
        (protocol, "potential", (points,), {}),
        (protocol, "conditional_inverse_transport", (points[:1], points), {}),
        (protocol, "proposal_log_density", (), {"local_points": points, "reference_points": points}),
        (protocol, "log_normalizer", (), {}),
    )
    for owner, name, args, kwargs in methods:
        method = getattr(type(owner), name)
        with tf.GradientTape() as tape:
            tape.watch(points)
            value = getattr(owner, name)(*args, **kwargs)
            loss = tf.reduce_sum(value)
        gradient = tape.gradient(loss, points, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        with tf.GradientTape() as tape:
            tape.watch(points)
            expected = method.__wrapped__(owner, *args, **kwargs)
            loss = tf.reduce_sum(expected)
        expected_gradient = tape.gradient(loss, points, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10, rtol=1e-10)
        compiled = method.program_for(owner, *args, **kwargs)
        inputs = (*args, *kwargs.values())
        assert "HloModule" in compiled.experimental_get_compiler_ir(*inputs)(stage="hlo")
        assert compiled.experimental_get_tracing_count() == 1

    def enclosing(query):
        with tf.GradientTape() as tape:
            tape.watch(query)
            local = protocol.inverse_transport(query)
            log_weights = target.log_target_density(local) - protocol.proposal_log_density(
                local_points=local, reference_points=query)
            value = tf.reduce_sum(log_weights)
        return log_weights, tape.gradient(value, query)

    specs = [tf.TensorSpec(points.shape, D)]
    graph = tf.function(enclosing, input_signature=specs, jit_compile=False, autograph=False)
    xla = tf.function(enclosing, input_signature=specs, jit_compile=True, autograph=False)
    for value, expected in zip(xla(points), graph(points), strict=True):
        np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10)
    definition = graph.get_concrete_function().graph.as_graph_def()
    assert not any(function.attr.get("_XlaMustCompile") and function.attr["_XlaMustCompile"].b
                   for function in definition.library.function)
    invalid = tf.fill(points.shape, tf.constant(float("nan"), D))
    with pytest.raises(ValueError, match="NONFINITE"):
        target.log_target_density(invalid)
    assert not bool(tf.reduce_all(tf.math.is_finite(xla(invalid)[0])))
