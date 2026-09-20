"""Diagnostic-only lifetime of a cached fitter's internal resource."""

import gc
import weakref

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_padded_factor import _data


@pytest.mark.parametrize("prebound", [False, True])
def test_factor_resource_survives_cache_eviction(prebound):
    factor._make_factor_program.cache_clear()
    arguments, _ = _data(3, 1, 4)
    config = factor.FactorCorrelationGeometryConfig(max_iterations=4)
    signature = [tf.TensorSpec(value.shape, value.dtype) for value in arguments]
    if prebound:
        factor._make_factor_program(3, 7, 6, config, True, factor._prediction_jacobian_diagnostics)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def enclosing(*inputs):
        program = factor._make_factor_program(3, 7, 6, config, True,
            factor._prediction_jacobian_diagnostics)
        return program(*inputs)

    before = enclosing(*arguments)
    initial_traces = enclosing.experimental_get_tracing_count()
    # TensorFlow traces twice only when the first call creates a variable.
    assert initial_traces == (1 if prebound else 2)
    concrete = enclosing.get_concrete_function()
    reference = weakref.ref(concrete.variables[0])
    graph_reference = weakref.ref(concrete.graph)
    print("FACTOR_RESOURCE_BEFORE " + str([(v.dtype.name, v.shape) for v in concrete.variables]), flush=True)
    factor._make_factor_program.cache_clear()
    gc.collect()
    print("FACTOR_RESOURCE_AFTER " + str([(v.dtype.name, v.shape) for v in concrete.variables]), flush=True)
    after = enclosing(*arguments)
    assert int(before["invalid_covariance_evaluations"]) == int(after["invalid_covariance_evaluations"])
    tf.debugging.assert_equal(before["precision"], after["precision"])
    assert enclosing.experimental_get_tracing_count() == initial_traces
    del concrete, enclosing
    gc.collect()
    print("FACTOR_RESOURCE_RELEASE " + str({"graph_released": graph_reference() is None,
        "variable_released": reference() is None}), flush=True)
    assert reference() is None
    assert graph_reference() is None
