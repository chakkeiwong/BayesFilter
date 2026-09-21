"""Original-source, complete-record and enclosing-XLA evaluator diagnostics."""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import batched_quadratic_center as runtime
from bayesfilter.inference.quadratic_batch_evaluation_tf import (
    make_quadratic_batch_evaluator,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.filter_repair_quadratic_batch_reference import original_evaluator


def _target(batch, dimension, *, counter=None, invalid_call=0, tie=False):
    def callback(points):
        if counter is None:
            call = tf.constant(0, tf.int64)
        else:
            call = counter.assign_add(1)
        scores = -points
        values = -0.5 * tf.reduce_sum(points * points, axis=1)
        if tie:
            values = tf.ones([batch], tf.float64)
        eligible = tf.ones([batch], tf.bool)
        if invalid_call:
            eligible &= (call != invalid_call) | (tf.range(batch) != 1)
        return values, scores, eligible
    return callback


def _inputs(count, record=True):
    points = tf.reshape(tf.range(18, dtype=tf.float64), [9, 2]) / 9.0
    return (points, tf.constant(count), tf.constant(record), tf.zeros([2], tf.float64),
            tf.constant(-100., tf.float64), tf.zeros([2], tf.float64),
            tf.constant(2**31 + 5, tf.int64), tf.constant(-1, tf.int64))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("case", ["full", "partial", "replay", "tie", "invalid_first", "invalid_later",
                                 "nan_first", "nan_later", "all_invalid"])
def test_ordered_batches_match_original_and_stop_at_same_row(jit, case):
    counter = tf.Variable(0, dtype=tf.int64)
    invalid = {"invalid_first": 1, "invalid_later": 2}.get(case, 0)
    callback = _target(4, 2, counter=counter, invalid_call=invalid, tie=case == "tie")
    if case == "all_invalid":
        base = callback

        def callback(points):
            values, scores, _ = base(points)
            return values, scores, tf.zeros([4], tf.bool)

    args = _inputs(8 if case == "full" else 9, record=case != "replay")
    if case.startswith("nan_"):
        row = 0 if case == "nan_first" else 4
        args = (tf.tensor_scatter_nd_update(args[0], [[row, 0]], [tf.constant(float("nan"), tf.float64)]), *args[1:])
    reference = original_evaluator(callback, 2, 4)(*args)
    original_calls = int(counter)
    counter.assign(0)
    program = make_quadratic_batch_evaluator(callback, 2, 4, 9, jit_compile=jit)
    actual = program(*args)
    assert int(counter) == original_calls == int(actual["callback_batches"])
    for name, expected in reference.items():
        observed = actual[name]
        if name in ("positions", "values", "scores", "valid"):
            observed = observed[:original_calls]
        if tf.convert_to_tensor(expected).dtype.is_floating:
            np.testing.assert_allclose(observed, expected, atol=1e-10, rtol=1e-10, err_msg=name)
        else:
            np.testing.assert_array_equal(observed, expected, err_msg=name)
    assert program.experimental_get_tracing_count() == 1


def test_changed_active_rows_and_replay_share_one_enclosing_xla_program():
    program = make_quadratic_batch_evaluator(_target(4, 2), 2, 4, 9)
    initial = program(*_inputs(9))
    changed = _inputs(1, record=False)
    replay = program(*changed)
    assert int(initial["callback_batches"]) == 3
    assert int(replay["callback_batches"]) == 1
    np.testing.assert_array_equal(replay["center"], changed[3])
    np.testing.assert_array_equal(replay["positions"][0], tf.repeat(changed[0][:1], 4, axis=0))
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
    assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {node.op for node in nodes}
    assert {"While", "StatelessWhile"} & {node.op for node in nodes}
    hlo = program.experimental_get_compiler_ir(*changed)(stage="hlo")
    assert "ENTRY" in hlo and "while" in hlo


def _equal_records(actual, expected, path="result"):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for name in expected:
            _equal_records(actual[name], expected[name], path + "." + name)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _equal_records(left, right, f"{path}[{index}]")
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, err_msg=path)
    else:
        assert actual == expected, path


@pytest.mark.parametrize("method", ["uniform_cloud", "paired_local"])
@pytest.mark.parametrize("case", ["centered", "move", "invalid"])
def test_complete_public_records_against_original_source(method, case, request):
    checkpoint = FrozenCheckpoint("3582b4ac", "quadratic_batch_original")
    original = checkpoint.load("bayesfilter.inference.batched_quadratic_center")
    options = {"pilot_method": method, "max_fit_rounds": 4, "centeredness_cap": 1e-8}
    if method == "uniform_cloud":
        options["rows_per_cloud"] = 9
    center = [0., 0.] if case != "move" else [0.7, -0.4]
    callback = _target(4, 2)
    if case == "invalid":
        base = callback

        def callback(points):
            values, scores, eligible = base(points)
            return values, scores, eligible & tf.reduce_all(tf.abs(points) < 1e-8, axis=1)

    expected = original.refine_batched_quadratic_center(
        callback, center, [1., 1.], config=original.BatchedQuadraticCenterConfig(**options)).payload()
    actual = runtime.refine_batched_quadratic_center(
        callback, center, [1., 1.], config=runtime.BatchedQuadraticCenterConfig(**options)).payload()
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"quadratic-record-{method}-{case}.json").open("x") as handle:
        json.dump({"before": expected, "after": actual, "baseline": "3582b4ac",
                   "original_source_sha256": checkpoint.hashes()}, handle, allow_nan=False, indent=2)
        handle.write("\n")
    if method == "paired_local":
        from tests.test_filter_repair_quadratic_rounds import compare_public_records

        compare_public_records(actual, expected, jit=True)
    else:
        assert actual["diagnostics"].pop("jit_compile_fit") is False
        _equal_records(actual, expected)
