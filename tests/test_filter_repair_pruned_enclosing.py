"""Diagnostic proof of the public pruned SRUKF factory's enclosing XLA boundary."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear import make_pruned_srukf_value_and_score
from tests.test_pruned_srukf_tf import _covariance_reference, _model


def test_public_pruned_factory_encloses_value_and_autodiff_score(request):
    from bayesfilter.nonlinear.pruned_srukf_tf import (
        make_pruned_srukf_value_and_score as implementation,
    )

    assert make_pruned_srukf_value_and_score is implementation
    parameters = tf.constant([[.65, .3, -2.], [.45, .18, -1.5]], tf.float64)
    panel = tf.constant([[[.1], [-.2], [.15]], [[0.], [.12], [-.04]]], tf.float64)
    signature = (tf.TensorSpec([2, 3], tf.float64), tf.TensorSpec([2, 3, 1], tf.float64))
    compiled = make_pruned_srukf_value_and_score(_model, *signature)
    reference = make_pruned_srukf_value_and_score(_model, *signature, jit_compile=False)
    concrete = compiled.get_concrete_function()
    assert concrete.function_def.attr["_XlaMustCompile"].b
    hlo = compiled.experimental_get_compiler_ir(parameters, panel)(stage="hlo")
    comparisons, independent, first = [], [], None
    changed = (parameters + .01, panel - .02)
    for label, args in (("initial", (parameters, panel)), ("changed", changed),
                        ("replay", (parameters, panel))):
        value, score, diagnostics = compiled(*args)
        expected_value, expected_score, expected_diagnostics = reference(*args)
        np.testing.assert_allclose(value, expected_value, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(score, expected_score, atol=1e-10, rtol=1e-10)
        assert diagnostics.keys() == expected_diagnostics.keys()
        for key, tensor in diagnostics.items():
            if tensor.dtype.is_floating:
                np.testing.assert_allclose(tensor, expected_diagnostics[key], atol=1e-11, rtol=1e-11)
            else:
                np.testing.assert_array_equal(tensor, expected_diagnostics[key])
        assert diagnostics["valid"].numpy().tolist() == [True, True]
        assert all("device:GPU:" in tensor.device for tensor in tf.nest.flatten((value, score, diagnostics)))
        if first is None:
            first = value, score, diagnostics
        if label == "replay":
            for tensor, initial in zip(tf.nest.flatten((value, score, diagnostics)),
                                       tf.nest.flatten(first), strict=True):
                np.testing.assert_array_equal(tensor, initial)
        else:
            theta, observations = (operand.numpy() for operand in args)
            for row in range(2):
                expected = _covariance_reference(theta[row], observations[row, :, 0])[0]
                np.testing.assert_allclose(value[row], expected, atol=1e-11, rtol=1e-11)
                for step in (1e-4, 1e-5):
                    numeric = []
                    for parameter in range(3):
                        shift = np.zeros(3)
                        shift[parameter] = step
                        plus = _covariance_reference(theta[row] + shift, observations[row, :, 0])[0]
                        minus = _covariance_reference(theta[row] - shift, observations[row, :, 0])[0]
                        numeric.append((plus - minus) / (2. * step))
                    np.testing.assert_allclose(score[row], numeric, atol=2e-7, rtol=2e-6)
                    independent.append({"label": label, "row": row, "step": step,
                        "numeric_score": numeric, "score": score[row].numpy().tolist()})
        comparisons.append({"label": label, "value": value.numpy().tolist(),
            "score": score.numpy().tolist(),
            "diagnostics": {key: tensor.numpy().tolist() for key, tensor in diagnostics.items()}})

    invalid_panel = tf.tensor_scatter_nd_update(panel, [[1, 0, 0]],
                                               [tf.constant(float("nan"), tf.float64)])
    invalid = compiled(parameters, invalid_panel)
    graph_invalid = reference(parameters, invalid_panel)
    assert invalid[2]["valid"].numpy().tolist() == [True, False]
    assert np.isneginf(invalid[0][1]) and np.all(np.isnan(invalid[1][1]))
    for tensor, initial, graph_tensor in zip(tf.nest.flatten(invalid), tf.nest.flatten(first),
                                             tf.nest.flatten(graph_invalid), strict=True):
        np.testing.assert_array_equal(tensor[0], initial[0])
        np.testing.assert_allclose(tensor, graph_tensor, atol=1e-10, rtol=1e-10, equal_nan=True)
    assert compiled.experimental_get_tracing_count() == reference.experimental_get_tracing_count() == 1
    changed_hlo = compiled.experimental_get_compiler_ir(*changed)(stage="hlo")
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / "pruned-enclosing.hlo.txt").write_text(hlo)
    (directory / "pruned-enclosing-changed.hlo.txt").write_text(changed_hlo)
    # TF exporter op_name labels are diagnostic metadata, not program operands.
    # Preserve both raw exports and demand byte equality of everything else.
    def without_op_names(text):
        return re.sub(r' op_name="[^"\n]*"', '', text)

    assert without_op_names(hlo) == without_op_names(changed_hlo)
    definition = concrete.graph.as_graph_def()
    operations = {node.op for node in definition.node}
    operations.update(node.op for function in definition.library.function for node in function.node_def)
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc", "XlaHostCompute", "Svd", "Qr"}
    assert "while(" in hlo
    assert not any("pfor" in function.signature.name.lower() for function in definition.library.function)
    report = {"schema": "filter_repair_pruned_srukf_enclosing.v1", "passed": True,
        "public_endpoint": "bayesfilter.nonlinear.make_pruned_srukf_value_and_score",
        "jit_compile_default": True, "score_provenance": "autodiff_total_likelihood",
        "trace_count": 1, "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "changed_hlo_sha256": hashlib.sha256(changed_hlo.encode()).hexdigest(),
        "raw_hlo_equal": hlo == changed_hlo,
        "hlo_equal_excluding_op_name_metadata": True,
        "comparisons": comparisons, "independent_score": independent,
        "invalid_status": [True, False], "valid_row_bitwise_unchanged": True,
        "nonclaims": ["Analytical recursion, module-wide loop compliance, native retention or cost qualification."]}
    (directory / "pruned-enclosing.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
