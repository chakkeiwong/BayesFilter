"""Independent pinned public consensus reference and enclosing-XLA checks."""

import ast
import hashlib
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as runtime


@pytest.fixture(scope="module")
def original():
    source = subprocess.check_output([
        "git", "show", "3582b4ac:bayesfilter/inference/fixed_center_curvature.py"],
        cwd=Path(__file__).resolve().parents[1], text=True)
    names = {"consensus_shrunk_precision", "consensus_shrunk_precision_tf", "_symmetric_matrix"}
    nodes = [node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"np": np, "tf": tf, "Sequence": Sequence, "Any": Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "3582b4ac:consensus", "exec"), namespace)  # noqa: S102 - pinned independent reference
    namespace["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return namespace


@pytest.mark.parametrize("dimension,count", [(3, 2), (5, 4)])
def test_public_consensus_original_records_and_operand_reuse(original, request, dimension, count):
    rng = np.random.default_rng(370 + dimension)
    factors = rng.normal(size=(count, dimension, dimension))
    matrices = factors @ np.swapaxes(factors, -1, -2) + np.eye(dimension)
    # Within the original tolerance: the wrapper must retain symmetrization.
    matrices[0, 0, 1] += 1e-13
    target = np.diag(np.linspace(.8, 2., dimension))
    args = (tf.constant(matrices), tf.constant(target), tf.constant(.25, tf.float64))
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in args)
    program = runtime._compiled_kernel(runtime._checked_consensus_kernel, signature)
    reports = []
    first_hlo = None
    for delta, weight in ((0., .25), (.2, 0.), (-.1, 1.), (0., .25)):
        values = matrices + delta * np.eye(dimension)
        expected = original["consensus_shrunk_precision"](list(values), target=target, weight=weight)
        actual = runtime.consensus_shrunk_precision(tuple(values), target=target, weight=weight)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
        operands = (tf.constant(values), args[1], tf.constant(weight, tf.float64))
        hlo = program.experimental_get_compiler_ir(*operands)(stage="hlo")
        assert "eigh" in hlo.lower()
        if first_hlo is not None:
            assert hlo == first_hlo
        else:
            first_hlo = hlo
        reports.append({"delta": delta, "weight": weight, "original": expected.tolist(),
            "actual": actual.numpy().tolist(), "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest()})
    assert program.experimental_get_tracing_count() == 1
    assert program.function_spec.jit_compile
    # An indefinite individual input can have a positive consensus. Preserve
    # the actual candidate-SPD criterion, despite the legacy exception wording.
    indefinite = [np.diag([-1., 1.]), np.diag([3., 1.])]
    np.testing.assert_allclose(runtime.consensus_shrunk_precision(
        indefinite, target=np.eye(2), weight=0.), original["consensus_shrunk_precision"](
            indefinite, target=np.eye(2), weight=0.), atol=1e-10, rtol=1e-10)
    report = {"schema": "consensus_endpoint_parity.v1", "dimension": dimension,
        "count": count, "original_revision": "3582b4ac", "source_sha256": original["source_sha256"],
        "trace_count": program.experimental_get_tracing_count(), "records": reports}
    path = Path(request.config.getoption("xmlpath")).parent / f"consensus-{dimension}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("case", ["empty", "nonsquare", "mismatch", "target_shape",
    "asymmetric", "target_asymmetric", "nonfinite", "target_nonfinite", "weight_nan",
    "weight_low", "weight_high", "indefinite", "singular"])
def test_consensus_original_rejections(original, case):
    matrices = [np.eye(2), 2. * np.eye(2)]
    target, weight = np.eye(2), .25
    if case == "empty":
        matrices = []
    elif case == "nonsquare":
        matrices = [np.ones((2, 3))]
    elif case == "mismatch":
        matrices[1] = np.eye(3)
    elif case == "target_shape":
        target = np.eye(3)
    elif case == "asymmetric":
        matrices[0][0, 1] = 1e-8
    elif case == "target_asymmetric":
        target[0, 1] = 1e-8
    elif case == "nonfinite":
        matrices[0][0, 0] = np.nan
    elif case == "target_nonfinite":
        target[0, 0] = np.inf
    elif case.startswith("weight_"):
        weight = {"weight_nan": np.nan, "weight_low": -.1, "weight_high": 1.1}[case]
    else:
        matrices = [np.diag([-1. if case == "indefinite" else 0., 1.])]
        weight = 0.
    with pytest.raises(ValueError) as expected:
        original["consensus_shrunk_precision"](matrices, target=target, weight=weight)
    with pytest.raises(ValueError) as actual:
        runtime.consensus_shrunk_precision(matrices, target=target, weight=weight)
    assert str(actual.value) == str(expected.value)


def test_consensus_enclosing_gradient_matches_original_kernel(original):
    matrices = tf.constant([[[2., .2], [.2, 1.]], [[1., .1], [.1, 3.]]], tf.float64)
    target = tf.constant([[2., .3], [.3, 2.]], tf.float64)
    weight = tf.constant(.3, tf.float64)
    operands = (matrices, target, weight)
    signature = tuple(tf.TensorSpec(value.shape, value.dtype) for value in operands)

    @tf.function(input_signature=signature, autograph=False, jit_compile=True)
    def gradient(*args):
        with tf.GradientTape() as tape:
            tape.watch(args)
            candidate, *_ = runtime._checked_consensus_kernel(*args)
            objective = tf.reduce_sum(candidate ** 2)
        return tape.gradient(objective, args)

    with tf.GradientTape() as tape:
        tape.watch(operands)
        # The endpoint symmetrizes before its kernel; include that derivative.
        expected = original["consensus_shrunk_precision_tf"](
            .5 * (matrices + tf.linalg.matrix_transpose(matrices)),
            .5 * (target + tf.transpose(target)), weight)
        objective = tf.reduce_sum(expected ** 2)
    for actual, expected in zip(gradient(*operands), tape.gradient(objective, operands), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_consensus_rejects_nonfinite_arithmetic_without_regularizing(original):
    matrices = [np.diag([1.7e308, 1.])]
    target = np.eye(2)
    with np.errstate(over="ignore", invalid="ignore"):
        # Preserve the original unsafe result as the reason for this pure veto.
        previous = original["consensus_shrunk_precision"](matrices, target=target, weight=.5)
    assert not np.isfinite(previous).all()
    with pytest.raises(ValueError, match="requires SPD"):
        runtime.consensus_shrunk_precision(matrices, target=target, weight=.5)
