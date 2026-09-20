"""Diagnostic localization of a nearly singular padded factor Jacobian."""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.compiler.tf2xla.ops.gen_xla_ops import (
    xla_remove_dynamic_dimension_size,
    xla_set_dynamic_dimension_size,
    xla_svd,
)

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_padded_factor import _data


def _observed_diagnosis():
    # Observe the exact current diagnostic operations, without a separate
    # derivative/factorization implementation in the comparison harness.
    source = inspect.getsource(factor._prediction_jacobian_diagnostics)
    assert source.count("return rank, condition") == 1
    source = source.replace("return rank, condition", "return rank, {"
        "'condition': condition, 'jacobian': jacobian, 'upper': upper, "
        "'singular': singular, 'tolerance': tolerance}")
    namespace = dict(factor._prediction_jacobian_diagnostics.__globals__)
    exec(compile(source, "padded_jacobian_observation", "exec"), namespace)  # noqa: S102
    return namespace["_prediction_jacobian_diagnostics"]


def test_partial_occupancy_jacobian_factorization_localization(request):
    observed = _observed_diagnosis()
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    compact, padded = _data(5, 1, 4)
    results = []
    for masked, arguments, rows in ((False, compact, 11), (True, padded, 14)):
        program = factor._make_factor_program(5, rows, 10, config, True,
            observed, padded_training=masked)
        result = program(*arguments)
        results.append(result)
    before, after = results
    first, second = (row["jacobian_condition"] for row in results)
    np.testing.assert_array_equal(before["optimizer"].position, after["optimizer"].position)
    np.testing.assert_array_equal(before["anchors"], after["anchors"])
    np.testing.assert_array_equal(second["jacobian"][55:], tf.zeros_like(second["jacobian"][55:]))

    # A separate call at the same raw state distinguishes optimizer/enclosure
    # effects from shape-dependent derivative or QR arithmetic.
    detached = []
    for offsets in (compact[1], tf.pad(compact[1], [[0, 3], [0, 0]])):
        @tf.function(input_signature=[tf.TensorSpec([14], tf.float64),
            tf.TensorSpec(offsets.shape, tf.float64), tf.TensorSpec([2], tf.int32)],
            jit_compile=True, autograph=False)
        def diagnose(raw, points, anchors):
            return observed(raw, points, dimension=5, anchors=tf.unstack(anchors),
                config=config, active_training_rows=tf.constant(11))

        detached.append(diagnose(before["optimizer"].position, offsets, before["anchors"])[1])

    def compare(left, right):
        return {"jacobian_max_error": float(tf.reduce_max(tf.abs(left["jacobian"] - right["jacobian"][:55]))),
            "upper_max_error": float(tf.reduce_max(tf.abs(left["upper"] - right["upper"]))),
            "singular_max_error": float(tf.reduce_max(tf.abs(left["singular"] - right["singular"]))),
            "condition": [float(left["condition"]), float(right["condition"])],
            "smallest_singular": [float(tf.reduce_min(left["singular"])), float(tf.reduce_min(right["singular"]))],
            "threshold": [float(left["tolerance"]), float(right["tolerance"])]}

    report = {"role": "explanatory_only", "optimizer_states_identical": True,
        "enclosed": compare(first, second), "detached": compare(*detached),
        "observations": [tf.nest.map_structure(lambda x: x.numpy().tolist(), row)
            for row in (first, second, *detached)]}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "padded-jacobian-factorization.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("PADDED_JACOBIAN_FACTORIZATION " + json.dumps({key: value for key, value in report.items()
        if key != "observations"}, sort_keys=True))


@pytest.mark.parametrize("clear_output_metadata", [False, True])
def test_bounded_dynamic_qr_localization(request, clear_output_metadata):
    """Try actual active dimensions on the same deterministically fitted Jacobian."""
    directory = Path(request.config.getoption("xmlpath")).parent
    compact, _ = _data(5, 1, 4)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    fitted = factor._make_factor_program(5, 11, 10, config, True, _observed_diagnosis())(*compact)
    reference = fitted["jacobian_condition"]
    matrix = reference["jacobian"]
    padded = tf.pad(matrix, [[0, 15], [0, 0]])

    @tf.function(input_signature=[tf.TensorSpec([70, 14], tf.float64), tf.TensorSpec([], tf.int32)],
        jit_compile=True, autograph=False)
    def factorize(values, rows):
        active = xla_set_dynamic_dimension_size(values, dim_index=0, size=rows)
        upper = tf.linalg.qr(active, full_matrices=False)[1]
        if clear_output_metadata:
            upper = xla_remove_dynamic_dimension_size(upper, dim_index=0)
        return upper, xla_svd(upper, max_iter=100, epsilon=sys.float_info.epsilon, precision_config="").s

    result = {"role": "explanatory_bounded_dynamic_qr_only", "source": "deterministic_compact_fit",
        "clear_output_metadata": clear_output_metadata}
    try:
        upper, singular = factorize(padded, tf.constant(55))
    except (tf.errors.OpError, ValueError) as error:
        # Unsupported compilation is an explicit diagnostic result, never an
        # execution fallback or fitter qualification.
        result.update(compiled=False, error_type=type(error).__name__, error=str(error))
    else:
        shapes_valid = upper.shape == (14, 14) and singular.shape == (14,)
        result.update(compiled=True, shapes_valid=shapes_valid,
            upper_shape=upper.shape.as_list(), singular_shape=singular.shape.as_list())
        if shapes_valid:
            result.update(upper_max_error=float(tf.reduce_max(tf.abs(upper - reference["upper"]))),
                singular=singular.numpy().tolist(),
                singular_max_error=float(tf.reduce_max(tf.abs(singular - reference["singular"]))))
        for stage in ("hlo", "optimized_hlo"):
            hlo = factorize.experimental_get_compiler_ir(padded, tf.constant(55))(stage=stage)
            with (directory / f"bounded-dynamic-qr-{int(clear_output_metadata)}-{stage}.txt").open("x") as handle:
                handle.write(hlo)
    with (directory / f"bounded-dynamic-qr-{int(clear_output_metadata)}.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print("BOUNDED_DYNAMIC_QR " + json.dumps(result, sort_keys=True))


def test_active_shape_qr_dispatch_localization(request):
    """Measure bounded static shape dispatch while numerical work stays in XLA."""
    observed = _observed_diagnosis()
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    compact, _ = _data(5, 1, 4)
    result = factor._make_factor_program(5, 11, 10, config, True, observed)(*compact)
    reference = result["jacobian_condition"]
    directory = Path(request.config.getoption("xmlpath")).parent
    reports = []
    for capacity in (14, 18):
        def factory(capacity):
            @tf.function(input_signature=[tf.TensorSpec([capacity * 5, 14], tf.float64),
                tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
            def dispatch(matrix, rows):
                def shape_case(active):
                    def qr():
                        return tf.linalg.qr(matrix[:active * 5], full_matrices=False)[1]
                    return qr
                # Only bind a finite set of static operand shapes. XLA executes
                # one selected QR; this is not a host iteration over observations.
                branches = tuple(shape_case(active) for active in range(10, capacity + 1))
                upper = tf.switch_case(rows - 10, branches)
                return upper, xla_svd(upper, max_iter=100, epsilon=sys.float_info.epsilon, precision_config="").s
            return dispatch

        program = factory(capacity)
        arguments = (tf.pad(reference["jacobian"], [[0, (capacity - 11) * 5], [0, 0]]), tf.constant(11))
        upper, singular = program(*arguments)
        np.testing.assert_array_equal(upper, reference["upper"])
        np.testing.assert_array_equal(singular, reference["singular"])
        hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        changed = (arguments[0], tf.constant(capacity))
        program(*changed)
        assert hlo == program.experimental_get_compiler_ir(*changed)(stage="hlo")
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        reports.append({"capacity": capacity, "branches": capacity - 9, "graph_nodes": len(nodes),
            "hlo_bytes": len(hlo.encode()), "same_compact_qr": True, "same_compact_singular": True,
            "same_hlo_on_active_count_change": True, "trace_count": program.experimental_get_tracing_count()})
    with (directory / "active-shape-qr-dispatch.json").open("x") as handle:
        json.dump(reports, handle, indent=2)
        handle.write("\n")
    print("ACTIVE_SHAPE_QR_DISPATCH " + json.dumps(reports, sort_keys=True))
