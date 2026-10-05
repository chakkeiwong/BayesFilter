"""Original-source and independent numerical checks for quadratic XLA kernels."""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import batched_quadratic_center as trust
from bayesfilter.inference import paired_score_pilot_tf as paired
from bayesfilter.inference import score_curvature_tf as dense
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint

D = tf.float64
KINDS = ("dense", "paired", "trust")


@lru_cache(maxsize=1)
def original():
    checkpoint = FrozenCheckpoint("3582b4ac", "quadratic_numerics_original")
    modules = {kind: checkpoint.load(module.__name__) for kind, module in
               zip(KINDS, (dense, paired, trust), strict=True)}
    return checkpoint, modules


def inputs(kind, dimension, case="regular"):
    rng = np.random.default_rng(20260921 + dimension)
    frame = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
    precision = frame @ np.diag(np.linspace(0.7, 3., dimension)) @ frame.T
    center = np.linspace(-.2, .3, dimension)
    if case in ("indefinite", "zero", "condition", "repeated"):
        diagonal = {"indefinite": -1., "zero": 0., "condition": 1e-11, "repeated": 1.}[case]
        precision = np.diag([diagonal] + [1.] * (dimension - 1))
    if case in ("tiny", "huge"):
        precision = np.diag(np.linspace(.7, 3., dimension)) * (1e-150 if case == "tiny" else 1e150)
        center = np.zeros(dimension)
    if case == "near_identity":
        precision = np.eye(dimension) + 1e-8 * precision
    if kind == "trust":
        radius = .03 if case != "interior" else 3.
        return tuple(tf.constant(value, D) for value in (precision, center, radius))
    if kind == "dense":
        offsets, checks = rng.normal(size=(2, 32, dimension)) * .1
        if case == "singular":
            offsets[:, -1] = offsets[:, 0]
        scores, check_scores = center - offsets @ precision, center - checks @ precision
        if case == "nonquadratic":
            scores -= .04 * offsets ** 3
            check_scores -= .04 * checks ** 3
        values = center, offsets, scores, checks, check_scores
    else:
        axes = np.concatenate((np.eye(dimension), -np.eye(dimension)))
        checks = .001 * np.concatenate((frame, -frame))
        first, second = center - .001 * axes @ precision, center - .0001 * axes @ precision
        if case == "inconsistent":
            first = center - 1.1 * .001 * axes @ precision
        elif case == "nonfinite":
            first[0, 0] = np.nan
        values = center, first, second, checks, center - checks @ precision
        if case.startswith("invalid_"):
            values[int(case.rsplit("_", 1)[-1])].flat[0] = np.inf
    return tuple(tf.constant(value, D) for value in values)


@lru_cache(maxsize=32)
def program(kind, dimension, *, source="current", jit=False):
    if kind == "paired" and source == "current":
        return paired.make_paired_score_precision_program(dimension, jit_compile=jit)
    module = original()[1][kind] if source == "original" else dict(zip(KINDS, (dense, paired, trust), strict=True))[kind]
    if kind == "trust":
        function = module.solve_spd_quadratic_trust_region_tf
        signature = [tf.TensorSpec([dimension, dimension], D), tf.TensorSpec([dimension], D), tf.TensorSpec([], D)]
    else:
        rows = 32 if kind == "dense" else 2 * dimension
        signature = [tf.TensorSpec([dimension], D)] + [tf.TensorSpec([rows, dimension], D)] * 4
        if kind == "dense":
            def function(center, offsets, scores, checks, check_scores):
                return module.fit_dense_score_precision_tf(center, offsets, scores,
                    selection_offsets=checks, selection_scores=check_scores)
        else:
            def function(center, first, second, checks, check_scores):
                return module.fit_paired_score_precision_tf(center, first, second, checks, check_scores)
    return tf.function(function, input_signature=signature, jit_compile=jit, autograph=False)


def equal_fields(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if value.dtype.is_floating:
            np.testing.assert_allclose(actual[key], value, atol=1e-10, rtol=1e-10, err_msg=key)
        else:
            np.testing.assert_array_equal(actual[key], value, err_msg=key)


def equal_dense_fields(actual, expected, dimension):
    """Keep the approved rank-policy definition separate from the raw archive."""
    if int(expected['design_rank']) < dimension:
        assert np.isposinf(float(actual['design_condition']))
        expected = {**expected, 'design_condition': tf.constant(float('inf'), D)}
    equal_fields(actual, expected)


def serializable(record):
    def clean(value):
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value
    return {key: clean(value.numpy().tolist()) for key, value in record.items()}


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("kind,case", [
    ("dense", "regular"), ("dense", "nonquadratic"), ("dense", "singular"),
    ("paired", "regular"), ("paired", "inconsistent"), ("paired", "nonfinite"),
    *(("paired", case) for case in ("indefinite", "zero", "condition", "repeated", "tiny", "huge", "near_identity")),
    *(("paired", f"invalid_{index}") for index in range(5)),
    ("trust", "regular"), ("trust", "interior"),
])
def test_all_fields_against_original_graph(kind, case, dimension, request):
    arguments = inputs(kind, dimension, case)
    reference = program(kind, dimension, source="original")(*arguments)
    results = {mode: program(kind, dimension, jit=mode == "xla")(*arguments) for mode in ("graph", "xla")}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"quadratic-numerics-{kind}-{case}-{dimension}.json").open("x") as handle:
        json.dump({"original": serializable(reference),
            **{mode: serializable(value) for mode, value in results.items()},
            "baseline": "3582b4ac", "source_sha256": original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    for result in results.values():
        if kind == 'dense':
            equal_dense_fields(result, reference, dimension)
        else:
            equal_fields(result, reference)
    result = results["xla"]
    if kind == "dense":
        coefficient = np.linalg.lstsq(arguments[1], arguments[0][None, :] - arguments[2], rcond=None)[0]
        np.testing.assert_allclose(result["raw_precision"], .5 * (coefficient + coefficient.T), atol=1e-10, rtol=1e-10)
    elif kind == "trust":
        step, multiplier = result["step"].numpy(), float(result["multiplier"])
        np.testing.assert_allclose((arguments[0].numpy() + multiplier * np.eye(dimension)) @ step,
                                  arguments[1], atol=1e-12, rtol=1e-12)
        assert np.linalg.norm(step) <= float(arguments[2]) * (1 + 1e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_changed_inputs_keep_runtime_operands_and_single_trace(kind):
    arguments = inputs(kind, 3)
    compiled = program(kind, 3, jit=True)
    first = compiled(*arguments)
    changed = tuple(value * tf.constant(1.1, D) for value in arguments)
    second = compiled(*changed)
    assert any(not np.array_equal(first[key], second[key]) for key in first)
    assert compiled.experimental_get_tracing_count() == 1
    concrete = compiled.get_concrete_function()
    assert not concrete.captured_inputs
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {node.op for node in nodes}
    hlo = compiled.experimental_get_compiler_ir(*arguments)(stage="hlo")
    changed_hlo = compiled.experimental_get_compiler_ir(*changed)(stage="hlo")
    assert hlo == changed_hlo
    assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == len(arguments)


def test_paired_raw_precision_pullback_matches_original():
    arguments = inputs("paired", 5)
    cotangent = tf.reshape(tf.range(25, dtype=D), [5, 5]) / 25.

    def differentiated(source, jit):
        kernel = program("paired", 5, source=source, jit=jit)

        @tf.function(input_signature=kernel.input_signature, jit_compile=jit, autograph=False)
        def value_and_gradient(*values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                raw = kernel(*values)["raw_precision"]
                scalar = tf.reduce_sum(raw * cotangent)
            return scalar, tape.gradient(scalar, values, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return value_and_gradient(*arguments)

    expected, observed = differentiated("original", False), differentiated("current", True)
    for left, right in zip(tf.nest.flatten(observed), tf.nest.flatten(expected), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dimension", [3, 5])
@pytest.mark.parametrize("case", ["centered", "move", "nonquadratic"])
def test_paired_public_complete_original_graph_records(dimension, case, request):
    original_module = original()[1]["trust"]
    matrix = inputs("trust", dimension)[0]
    quartic = .04 if case == "nonquadratic" else 0.

    def callback(points):
        score = -points @ matrix - quartic * points ** 3
        value = -.5 * tf.reduce_sum(points * (points @ matrix), axis=1)
        value -= .25 * quartic * tf.reduce_sum(points ** 4, axis=1)
        return value, score, tf.ones([4], tf.bool)

    center = tf.zeros([dimension], D) if case == "centered" else tf.linspace(tf.constant(-.4, D), tf.constant(.7, D), dimension)
    options = {"pilot_method": "paired_local", "max_fit_rounds": 4, "centeredness_cap": 1e-8}
    expected = original_module.refine_batched_quadratic_center(callback, center, tf.ones([dimension], D),
        config=original_module.BatchedQuadraticCenterConfig(**options, jit_compile_trust=False)).payload()
    actual = trust.refine_batched_quadratic_center(callback, center, tf.ones([dimension], D),
        config=trust.BatchedQuadraticCenterConfig(**options)).payload()
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"paired-public-{case}-{dimension}.json").open("x") as handle:
        json.dump({"original": expected, "current": actual, "baseline": "3582b4ac",
            "original_trust_jit_compile": False, "reason": "graph precision authority; original trust XLA has demonstrated errors",
            "source_sha256": original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write("\n")
    from tests.test_filter_repair_quadratic_rounds import compare_public_records

    assert expected["diagnostics"]["jit_compile_trust"] is False
    compare_public_records(actual, expected, jit=True)
