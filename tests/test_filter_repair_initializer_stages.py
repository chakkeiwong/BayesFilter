"""Diagnostic isolation of initializer rounding, never a runtime replacement."""

import json
from decimal import Decimal

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference.mass_matrix_tf import _eigenpairs
from bayesfilter.ops.qr_lstsq_tf import complete_orthogonal_lstsq
from tests.test_filter_repair_fixed_fitting import _inputs
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_initializer_rounding import _decimal_error, _decimal_lstsq

D = tf.float64


def _maximum_error(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _project(raw, *, jit):
    symmetric = .5 * (raw + tf.transpose(raw))
    values, vectors = _eigenpairs(symmetric, jit)
    floor = tf.maximum(tf.reduce_max(tf.abs(values)), 1.) / 1e8
    precision = (vectors * tf.maximum(values, floor)[None, :]) @ tf.transpose(vectors)
    return precision, {
        "symmetric": symmetric,
        "values": values,
        "vectors": vectors,
        "residual": symmetric @ vectors - vectors * values[None, :],
        "orthogonality": tf.transpose(vectors) @ vectors - tf.eye(3, dtype=D),
    }


def _state(module, covariance, config, *, native):
    arguments = {"factor_count": 1, "loading_margin": config.loading_margin}
    if native:
        arguments["jit_compile"] = True
    deviations, loadings, anchors = module._initial_factor_state(covariance, **arguments)
    raw = module._encode_state(deviations, loadings, anchors, config)
    return {"deviations": deviations, "loadings": loadings,
        "anchor": tf.convert_to_tensor(anchors, tf.int32), "raw": raw}


def _ready(value):
    return tf.nest.map_structure(lambda item: tf.convert_to_tensor(item).numpy().tolist(), value)


def test_identical_predecessors_separate_projection_inverse_state_and_encoding():
    original = _baseline()
    config = factor.FactorCorrelationGeometryConfig(max_condition_number=1e8)
    center, training, scores = (tf.constant(value, D) for value in _inputs(3)[1:4])
    weights = tf.fill([9], tf.constant(1 / 9, D))
    weights /= tf.reduce_sum(weights)
    square_signature = [tf.TensorSpec([3, 3], D)]
    project = tf.function(lambda raw: _project(raw, jit=True),
        input_signature=square_signature, jit_compile=True, autograph=False)
    inverse = tf.function(tf.linalg.inv, input_signature=square_signature,
        jit_compile=True, autograph=False)
    state = tf.function(lambda covariance: _state(factor, covariance, config, native=True),
        input_signature=square_signature, jit_compile=True, autograph=False)
    encode = tf.function(lambda std, load, anchor: factor._encode_state(std, load, (anchor,), config),
        input_signature=[tf.TensorSpec([3], D), tf.TensorSpec([3, 1], D), tf.TensorSpec([], tf.int32)],
        jit_compile=True, autograph=False)

    @tf.function(input_signature=[tf.TensorSpec([9, 3], D), tf.TensorSpec([9, 3], D)],
        jit_compile=True, autograph=False)
    def initialize(offsets, response):
        raw = complete_orthogonal_lstsq(offsets * tf.sqrt(weights)[:, None], response * tf.sqrt(weights)[:, None])
        precision = factor._weighted_dense_precision(offsets, response, weights, max_condition_number=1e8)
        covariance = tf.linalg.inv(precision)
        return {"raw": raw, "precision": precision, "covariance": covariance,
            "state": _state(factor, covariance, config, native=True)}

    report = {"role": "explanatory_initializer_stage_isolation_only", "runtime_modified": False,
        "baseline": "3582b4ac", "replicas": [],
        "limitation": "Exposing intermediate outputs can change compiler fusion; complete fit parity remains the acceptance gate."}
    for index in range(2):
        offsets, response = training[index], center[None, :] - scores[index]
        raw = tf.linalg.lstsq(offsets * tf.sqrt(weights)[:, None], response * tf.sqrt(weights)[:, None], fast=False)
        precision = original._weighted_dense_precision(offsets, response, weights, max_condition_number=1e8)
        covariance = tf.linalg.inv(precision)
        before = {"raw": raw, "precision": precision, "covariance": covariance,
            "state": _state(original, covariance, config, native=False)}
        after = initialize(offsets, response)
        one = {"replicate": index, "original": _ready(before), "native": _ready(after),
            "propagated_errors": {key: _maximum_error(after[key], before[key])
                for key in ("raw", "precision", "covariance")}, "identical_predecessors": {}}
        one["propagated_errors"]["encoded_raw"] = _maximum_error(after["state"]["raw"], before["state"]["raw"])
        np.testing.assert_array_equal(after["state"]["anchor"], before["state"]["anchor"])
        for label, values in (("original", before), ("native", after)):
            eager_projected, eager_spectrum = _project(values["raw"], jit=False)
            xla_projected, xla_spectrum = project(values["raw"])
            eager_inverse, xla_inverse = tf.linalg.inv(values["precision"]), inverse(values["precision"])
            eager_state = _state(original, values["covariance"], config, native=False)
            xla_state = state(values["covariance"])
            encoded = encode(eager_state["deviations"], eager_state["loadings"], eager_state["anchor"][0])
            reference = _decimal_lstsq(values["precision"].numpy(), np.eye(3), 90)
            repeated = _decimal_lstsq(values["precision"].numpy(), np.eye(3), 60)
            reference_gap = _decimal_error(reference, repeated)
            assert reference_gap < Decimal("1e-50")
            assert np.linalg.cond(values["precision"].numpy()) < 100.
            decimal_errors = {}
            for name, value in (("original_eager", eager_inverse), ("native_xla", xla_inverse)):
                decimals = [[Decimal.from_float(float(x)) for x in row] for row in value.numpy()]
                decimal_errors[name] = str(_decimal_error(decimals, reference))
            one["identical_predecessors"][label] = {
                "projection_error": _maximum_error(xla_projected, eager_projected),
                "projection_original_eager": _ready(eager_projected), "projection_native_xla": _ready(xla_projected),
                "eigen_original_eager": _ready(eager_spectrum), "eigen_native_xla": _ready(xla_spectrum),
                "inverse_error": _maximum_error(xla_inverse, eager_inverse),
                "inverse_decimal_errors": decimal_errors, "decimal_reference_agreement": str(reference_gap),
                "inverse_original_eager": _ready(eager_inverse), "inverse_native_xla": _ready(xla_inverse),
                "inverse_original_residual": _maximum_error(values["precision"] @ eager_inverse, tf.eye(3, dtype=D)),
                "inverse_native_residual": _maximum_error(values["precision"] @ xla_inverse, tf.eye(3, dtype=D)),
                "factor_state_original": _ready(eager_state), "factor_state_native": _ready(xla_state),
                "factor_raw_error": _maximum_error(xla_state["raw"], eager_state["raw"]),
                "encoding_only_error": _maximum_error(encoded, eager_state["raw"]),
                "encoding_only_native": _ready(encoded)}
            np.testing.assert_array_equal(xla_state["anchor"], eager_state["anchor"])
            assert np.all(np.isfinite(xla_state["raw"]))
            assert _maximum_error(xla_projected, eager_projected) < 1e-12
        report["replicas"].append(one)
    print("FIXED_FITTING_INITIALIZER_STAGES " + json.dumps(report, sort_keys=True))
