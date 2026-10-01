"""Frozen compact preparation authority for native structured reused rows."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "f06fd505:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("structured_preparation_frozen_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - frozen diagnostic oracle
    return module


def _target(dimension):
    def batched(rows):
        precision = tf.linalg.diag(tf.linspace(tf.constant(1., D), tf.constant(2., D), dimension)) + .1
        scores = -(rows @ precision)
        return .5 * tf.reduce_sum(rows * scores, axis=1), scores

    def scalar(row):
        values, scores = batched(row[None, :])
        return values[0], scores[0]

    return scalar, batched


def _arguments(dimension, capacity, occupancy):
    center = tf.linspace(tf.constant(-.2, D), tf.constant(.3, D), dimension)
    scale = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), dimension)
    radius = tf.constant(.3, D)
    scalar, batched = _target(dimension)
    if occupancy == "boundary":
        center, scale = tf.zeros([dimension], D), tf.ones([dimension], D)
    center_score = scalar(center)[1]
    offsets = np.zeros([capacity, dimension])
    if occupancy == "full":
        offsets[:, 0] = np.linspace(.01, .2, capacity)
    elif occupancy == "partial":
        offsets[::2, 0] = np.linspace(.01, .2, len(offsets[::2]))
    elif occupancy == "three":
        offsets[:3, 0] = [.03, .11, .19]
    elif occupancy == "boundary":
        offsets[:4, 0] = [.3 * (1. + .5e-12), .3 * (1. + 2e-12), 1e-12, 2e-12]
        if capacity > 4:
            offsets[4, 0] = np.nan
            offsets[5, 0] = np.inf
            offsets[6, 0] = .1
    elif occupancy != "none":
        raise ValueError(occupancy)
    positions = center[None, :] + tf.constant(offsets, D) * scale[None, :]
    scores = batched(positions)[1]
    if occupancy == "boundary" and capacity > 4:
        scores = tf.tensor_scatter_nd_update(scores, [[6, 0]], [tf.constant(float("nan"), D)])
    return scalar, batched, (center, center_score, scale, radius, tf.constant([2026, 715]), positions, scores)


def _before(baseline, scalar, batched, args, fresh_count, reuse):
    center, center_score, scale, radius, seed, positions, scores = args
    return baseline._structured_factor_fit_data(scalar, center, center_score, scale,
        search_theta=positions, search_scores=scores, dimension=int(center.shape[0]),
        radius=float(radius), fresh_sample_count=fresh_count, seed=tuple(seed.numpy().tolist()),
        evaluations=7, batched_value_and_score_fn=batched, reuse_search_scores=reuse)


def _public_data(result):
    count = int(result["active_training_rows"])
    winner = int(result["best_index"]) >= 0
    return {"center_score_z": result["center_score_z"],
        "training_offsets_z": result["training_offsets_z"][:count],
        "training_scores_z": result["training_scores_z"][:count],
        "holdout_offsets_z": result["holdout_offsets_z"],
        "holdout_scores_z": result["holdout_scores_z"],
        "training_weights": result["training_weights"][:count],
        "fresh_training_count": int(result["fresh_training_count"]),
        "fresh_holdout_count": int(result["fresh_holdout_count"]),
        "reused_training_count": int(result["reused_training_count"]),
        "unique_fresh_evaluations": int(result["unique_fresh_evaluations"]),
        "best_exact_value": float(result["best_value"]) if winner else None,
        "best_exact_position": result["best_position"].numpy().tolist() if winner else None,
        "best_exact_score": result["best_score"].numpy().tolist() if winner else None,
        "best_exact_source": "structured_fit_cloud" if winner else None}


@pytest.mark.parametrize("dimension,capacity", [(3, 4), (5, 32)])
@pytest.mark.parametrize("occupancy", ["none", "partial", "three", "full", "boundary"])
@pytest.mark.parametrize("batched_enabled", [False, True])
def test_preparation_preserves_every_compact_field(baseline, dimension, capacity, occupancy, batched_enabled):
    scalar, batched, args = _arguments(dimension, capacity, occupancy)
    callback = batched if batched_enabled else None
    before, evaluations = _before(baseline, scalar, callback, args, 4 * dimension, True)
    program = structured_data_program(scalar, callback, dimension, 4 * dimension, capacity, True)
    after = program(*args)
    _compare(_public_data(after), before)
    assert evaluations == 7 + int(after["unique_fresh_evaluations"])
    active = int(after["active_training_rows"])
    np.testing.assert_array_equal(after["training_offsets_z"][active:], 0.)
    np.testing.assert_array_equal(after["training_scores_z"][active:], 0.)
    np.testing.assert_array_equal(after["training_weights"][active:], 0.)
    assert int(after["reused_training_count"]) == (capacity if occupancy == "full" else
        capacity // 2 if occupancy == "partial" else 3 if occupancy == "three" else
        2 if occupancy == "boundary" else 0)


def test_reuse_disabled_and_empty_capacity_match_compact(baseline):
    for capacity, reuse in ((4, False), (0, True)):
        scalar, batched, args = _arguments(3, capacity, "full")
        expected, _ = _before(baseline, scalar, batched, args, 12, reuse)
        actual = structured_data_program(scalar, batched, 3, 12, capacity, reuse)(*args)
        _compare(_public_data(actual), expected)
        assert actual["training_offsets_z"].shape == (6, 3)


def test_runtime_eligibility_keeps_one_trace_all_operands_and_same_hlo(request, baseline):
    scalar, batched, first = _arguments(3, 4, "none")
    program = structured_data_program(scalar, batched, 3, 12, 4, True)
    before = program(*first)
    hlo = program.experimental_get_compiler_ir(*first)(stage="hlo")
    _, _, changed = _arguments(3, 4, "full")
    after = program(*changed)
    assert int(before["reused_training_count"]) == 0
    assert int(after["reused_training_count"]) == 4
    assert program.experimental_get_tracing_count() == 1
    assert program.experimental_get_compiler_ir(*changed)(stage="hlo") == hlo
    changed = (changed[0] + .07, changed[1] - .03, changed[2] * .9, changed[3] * 1.1,
        changed[4] + [3, 11], changed[5] + .02, changed[6] - .04)
    expected, _ = _before(baseline, scalar, batched, changed, 12, True)
    _compare(_public_data(program(*changed)), expected)
    assert program.experimental_get_tracing_count() == 1
    assert program.experimental_get_compiler_ir(*changed)(stage="hlo") == hlo
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "structured-preparation-hlo.txt").open("x") as handle:
        handle.write(hlo)
    entry = hlo[hlo.rfind("\nENTRY "): ]
    assert sorted(int(number) for number in re.findall(r"\bparameter\((\d+)\)", entry)) == list(range(7))
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = [*graph.node, *(node for function in graph.library.function for node in function.node_def)]
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)


@pytest.mark.parametrize("batched_enabled", [False, True])
def test_fresh_callback_order_counts_and_first_finite_tie(baseline, batched_enabled):
    _, _, args = _arguments(3, 4, "none")
    calls = tf.Variable(0, dtype=tf.int64)
    visited = tf.Variable(tf.zeros([12, 3], D))

    def scalar(row):
        index = calls.read_value()
        visited.assign(tf.tensor_scatter_nd_update(visited, [[index]], [row]))
        calls.assign_add(1)
        return tf.constant(1., D), -row

    def batched(rows):
        visited.assign(rows)
        calls.assign_add(1)
        return tf.ones([12], D), -rows

    callback = batched if batched_enabled else None
    before, _ = _before(baseline, scalar, callback, args, 12, True)
    old_rows = visited.read_value()
    assert int(calls) == (1 if batched_enabled else 12)
    calls.assign(0)
    after = structured_data_program(scalar, callback, 3, 12, 4, True)(*args)
    assert int(calls) == (1 if batched_enabled else 12)
    _compare(visited, old_rows)
    _compare(_public_data(after), before)
    assert int(after["best_index"]) == 0


def test_invalid_fresh_candidates_and_nontraceable_callback(baseline):
    _, _, args = _arguments(3, 4, "none")

    def invalid(rows):
        return tf.fill([12], tf.constant(float("nan"), D)), -rows

    before, _ = _before(baseline, None, invalid, args, 12, True)
    after = structured_data_program(None, invalid, 3, 12, 4, True)(*args)
    _compare(_public_data(after), before)
    assert int(after["best_index"]) == -1

    def host_only(row):
        row.numpy()
        raise AssertionError("An eager fallback must never execute")

    with pytest.raises(AttributeError, match="numpy"):
        structured_data_program(host_only, None, 3, 12, 4, True)(*args)

def test_preparation_pullbacks_preserve_original_tensors(baseline):
    scalar, batched, inputs = _arguments(3, 4, 'partial')
    program = structured_data_program(scalar, batched, 3, 12, 4, True)
    indices = (0, 1, 2, 5, 6)
    watched = tuple(inputs[index] for index in indices)

    def objective(data):
        return (tf.reduce_sum(tf.square(data['center_score_z']))
            + .3 * tf.reduce_sum(tf.square(data['training_offsets_z']))
            + .7 * tf.reduce_sum(tf.square(data['training_scores_z']))
            + .2 * tf.reduce_sum(tf.square(data['holdout_offsets_z']))
            + .6 * tf.reduce_sum(tf.square(data['holdout_scores_z'])))

    with tf.GradientTape() as tape:
        tape.watch(watched)
        before, _ = _before(baseline, scalar, batched, inputs, 12, True)
        value = objective(before)
    expected = tape.gradient(value, watched)

    @tf.function(input_signature=program.input_signature, jit_compile=True, autograph=False)
    def differentiate(*args):
        watched_inputs = tuple(args[index] for index in indices)
        with tf.GradientTape() as tape:
            tape.watch(watched_inputs)
            value = objective(program.python_function(*args))
        return value, tape.gradient(value, watched_inputs)

    actual_value, actual = differentiate(*inputs)
    np.testing.assert_allclose(actual_value, value, atol=1e-10, rtol=1e-10)
    for before, after in zip(expected, actual, strict=True):
        np.testing.assert_allclose(tf.convert_to_tensor(after), tf.convert_to_tensor(before), atol=1e-10, rtol=1e-10)
