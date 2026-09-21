"""Frozen-source probe/partition checks; NumPy is an independent diagnostic only."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.paired_score_pilot_tf import make_paired_score_probe_program
from bayesfilter.inference.quadratic_probe_evaluation_tf import (
    make_paired_probe_evaluator,
)
from tests.filter_repair_quadratic_batch_reference import original_evaluator
from tests.test_filter_repair_quadratic_batches import _target
from tests.test_filter_repair_quadratic_numerics import (
    equal_fields,
    original,
    serializable,
)

D = tf.float64


def original_probes(callback, dimension, batch, *, steps=(.001, .0001)):
    """Reference composition retains original random generation and chunk code."""
    source = original()[1]["paired"]
    rows, batches = 2 * dimension, (2 * dimension + batch - 1) // batch
    evaluate = original_evaluator(callback, dimension, batch)

    def execute(seed, round_index, scale, center, center_value, center_score, first_index, selected_index):
        designs = source.paired_score_probe_designs_tf(dimension, seed=int(seed), round_index=int(round_index), steps=steps)
        state = {"partitions": tf.constant(0), "ok": tf.constant(True),
            "physical_rows": tf.constant(0, tf.int64), "padded_rows": tf.constant(0, tf.int64),
            "invalid_rows": tf.constant(0, tf.int64), "callback_batches": tf.constant(0),
            "center": center, "center_value": center_value, "center_score": center_score,
            "selected_index": selected_index, "partition_batches": tf.zeros([3], tf.int32),
            "partition_first_index": tf.zeros([3], tf.int64),
            "positions": tf.zeros([3, batches, batch, dimension], D),
            "values": tf.zeros([3, batches, batch], D),
            "scores": tf.zeros([3, batches, batch, dimension], D),
            "valid": tf.zeros([3, batches, batch], tf.bool), "scaled_scores": tf.zeros([3, rows, dimension], D)}
        for index, offsets in enumerate(designs):
            result = evaluate(center[None, :] + offsets * scale, rows, True, state["center"], state["center_value"],
                state["center_score"], first_index + state["physical_rows"], state["selected_index"])
            count = int(result["callback_batches"])
            state["partition_first_index"] = tf.tensor_scatter_nd_update(state["partition_first_index"], [[index]], [first_index + state["physical_rows"]])
            state["partition_batches"] = tf.tensor_scatter_nd_update(state["partition_batches"], [[index]], [count])
            for key in ("positions", "values", "scores", "valid"):
                pad = [[0, batches - count]] + [[0, 0]] * (result[key].shape.rank - 1)
                state[key] = tf.tensor_scatter_nd_update(state[key], [[index]], tf.pad(result[key], pad)[None])
            scores = tf.reshape(state["scores"][index], [-1, dimension])[:rows] * scale
            state["scaled_scores"] = tf.tensor_scatter_nd_update(state["scaled_scores"], [[index]], scores[None])
            for key in ("physical_rows", "padded_rows", "invalid_rows", "callback_batches"):
                state[key] += tf.cast(result[key], state[key].dtype)
            for key in ("ok", "center", "center_value", "center_score", "selected_index"):
                state[key] = tf.convert_to_tensor(result[key], dtype=state[key].dtype)
            state["partitions"] += 1
            if not bool(state["ok"]):
                break
        return {**state, "check_offsets": designs[2]}

    return execute


def arguments(dimension, *, seed=731, round_index=0):
    center = tf.linspace(tf.constant(-.3, D), tf.constant(.2, D), dimension)
    return (tf.constant(seed), tf.constant(round_index), tf.linspace(tf.constant(.7, D), tf.constant(1.3, D), dimension),
        center, -.5 * tf.reduce_sum(center * center), -center, tf.constant(2**31 + 9, tf.int64), tf.constant(-1, tf.int64))


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("steps", [(.001, .0001), (.1, .01)])
def test_original_seed_frame_and_axis_order(dimension, steps):
    reference = original()[1]["paired"].paired_score_probe_designs_tf
    for jit in (False, True):
        kernel = make_paired_score_probe_program(dimension, steps=steps, jit_compile=jit)
        for seed, round_index in ((731, 0), (20260910, 1), (11, 7), (-1729, 2)):
            expected = reference(dimension, seed=seed, round_index=round_index, steps=steps)
            actual = kernel(seed, round_index)
            for left, right in zip(actual, expected, strict=True):
                np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
                np.testing.assert_array_equal(left[dimension:], -left[:dimension])
            assert kernel.experimental_get_tracing_count() == 1


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("case", ["regular", "invalid_first", "invalid_second", "invalid_third", "nonfinite", "tie"])
def test_ordered_probe_partitions_preserve_original_complete_records(dimension, case, request):
    batches = (2 * dimension + 3) // 4
    bad = {"invalid_first": 1, "invalid_second": batches + 1, "invalid_third": 2 * batches + 1}.get(case, 0)
    counter = tf.Variable(0, dtype=tf.int64)
    callback = _target(4, dimension, counter=counter, invalid_call=bad, tie=case == "tie")
    args = arguments(dimension)
    if case == "nonfinite":
        args = (*args[:2], tf.fill([dimension], tf.constant(float("inf"), D)), *args[3:])
    expected = original_probes(callback, dimension, 4)(*args)
    original_calls = int(counter)
    result = {}
    for mode in ("graph", "xla"):
        counter.assign(0)
        kernel = make_paired_probe_evaluator(callback, dimension, 4, jit_compile=mode == "xla")
        actual = kernel(*args)
        assert int(counter) == original_calls == int(actual["callback_batches"])
        equal_fields(actual, expected)
        result[mode] = serializable(actual)
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"paired-probes-{dimension}-{case}.json").open("x") as handle:
        json.dump({"original": serializable(expected), **result, "source_sha256": original()[0].hashes()}, handle, indent=2, allow_nan=False)
        handle.write("\n")


def test_changed_probe_inputs_remain_runtime_operands(request):
    kernel = make_paired_probe_evaluator(_target(4, 5), 5, 4)
    args = arguments(5)
    first = kernel(*args)
    changed = arguments(5, seed=20260910, round_index=3)
    changed = (*changed[:2], changed[2] * 1.1, changed[3] + .02,
        changed[4] - .1, changed[5] + .05, changed[6] + 7, changed[7] + 2)
    second = kernel(*changed)
    equal_fields(second, original_probes(_target(4, 5), 5, 4)(*changed))
    assert not np.array_equal(first["check_offsets"], second["check_offsets"])
    assert kernel.experimental_get_tracing_count() == 1
    concrete = kernel.get_concrete_function()
    assert not concrete.captured_inputs
    graph = concrete.graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    assert not {"PyFunc", "EagerPyFunc", "PyFuncStateless"} & {node.op for node in nodes}
    hlo = kernel.experimental_get_compiler_ir(*args)(stage="hlo")
    changed_hlo = kernel.experimental_get_compiler_ir(*changed)(stage="hlo")
    directory = Path(request.config.getoption("xmlpath")).parent
    for filename, value in (("probe-initial.hlo", hlo), ("probe-changed.hlo", changed_hlo),
                            ("probe-graph.pbtxt", str(graph))):
        with (directory / filename).open("x") as handle:
            handle.write(value)
    assert hlo == changed_hlo
    assert len(re.findall(r"\bparameter\((\d+)\)", hlo[hlo.rfind("\nENTRY "):])) == 8


def test_public_paired_steps_accept_original_list_schema():
    from bayesfilter.inference.batched_quadratic_center import (
        BatchedQuadraticCenterConfig,
        refine_batched_quadratic_center,
    )
    from tests.test_filter_repair_quadratic_batches import _equal_records

    baseline = original()[1]["trust"]
    callback = _target(4, 3)
    options = {"pilot_method": "paired_local", "paired_steps": [.001, .0001]}
    expected = baseline.refine_batched_quadratic_center(callback, tf.zeros([3], D), tf.ones([3], D),
        config=baseline.BatchedQuadraticCenterConfig(**options)).payload()
    actual = refine_batched_quadratic_center(callback, tf.zeros([3], D), tf.ones([3], D),
        config=BatchedQuadraticCenterConfig(**options)).payload()
    assert actual["diagnostics"].pop("jit_compile_fit") is True
    _equal_records(actual, expected)
