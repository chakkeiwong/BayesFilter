"""External tapes across public TT, density and transport XLA boundaries."""

import pytest
import tensorflow as tf

from bayesfilter.highdim import source_route
from tests.test_filter_repair_remaining_routes import _original
from tests.test_filter_repair_source_runtime import _route


def _pinned_transport(transport):
    tt = _original("tt")
    squared = _original("squared_tt")
    squared.FunctionalTT = tt.FunctionalTT
    squared.TTContractedRepresentation = tt.TTContractedRepresentation
    density = transport.density
    original_tt = tt.FunctionalTT(tuple(tt.TTCore(core.values) for core in density.sqrt_tt.cores),
                                 density.sqrt_tt.product_basis, density.measure_convention)
    original_density = squared.SquaredTTDensity(**{**density.__dict__, "sqrt_tt": original_tt,
        "defensive_density": squared.TensorProductReferenceDensity(density.sqrt_tt.product_basis,
            density.measure_convention, density.defensive_density.floor)})
    module = _original("transport")
    module.SquaredTTDensity = squared.SquaredTTDensity
    module.SquaredTTMarginal = squared.SquaredTTMarginal
    return module.FixedTTSIRTTransport(original_density,
                                      module.KRCDFConfig(**transport.cdf_config.__dict__))


@pytest.mark.parametrize("family", ["tt", "density", "transport"])
@pytest.mark.parametrize("jit_compile", [False, True])
def test_public_external_tape_preserves_pinned_query_and_core_derivatives(family, jit_compile):
    _, protocol = _route(source_route, "ttsirt")
    candidate = protocol.transport_object
    baseline = _pinned_transport(candidate)
    query = tf.constant([[-.23, .38], [.19, -.41], [.63, .12]], tf.float64)
    cores = tuple(core.values for core in candidate.density.sqrt_tt.cores)
    sources = (query, *cores)

    def outputs(transport, kwargs):
        density = transport.density
        if family == "tt":
            contracted = density.sqrt_tt.contract_axes((1,), **kwargs)
            return (density.sqrt_tt.evaluate(query, **kwargs),
                    density.sqrt_tt.integrate_all(**kwargs),
                    *(core.values for core in contracted.cores))
        if family == "density":
            return (density.log_density(query, **kwargs), density.normalizer(**kwargs),
                    density.normalized_marginal_density_values((0,), query[:, :1]))
        points = tf.transpose(query)
        return (transport.forward_transport(points, **kwargs),
                transport.forward_log_jacobian(points, **kwargs), transport.eval_pdf(points, **kwargs))

    results = []
    for transport, kwargs in ((baseline, {}), (candidate, {"jit_compile": jit_compile})):
        with tf.GradientTape() as tape:
            tape.watch(sources)
            values = outputs(transport, kwargs)
            loss = tf.add_n([tf.reduce_sum(value) for value in values])
        gradients = tape.gradient(loss, sources)
        assert all(value is not None for value in gradients)
        results.append((values, gradients))
    for expected in results[0][1]:
        assert bool(tf.reduce_any(expected != 0.))
    for actual, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0]), strict=True):
        tf.debugging.assert_near(actual, expected, atol=1e-10, rtol=1e-10)
