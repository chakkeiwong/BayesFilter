"""Pinned batched optimizer, scalar replay and buffered progress parity."""

import importlib.util
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from bayesfilter.inference.sequential_batched_locator_tf import (
    BufferedBatchedLocator,
    batched_locator_program,
)
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_sequential_locator import _target

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "dcfaa15d:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("batched_locator_pre_enclosure_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned reference
    return module


def _batch(kind):
    def target(points):
        delta = points - tf.constant([.15, -.2], D)
        scores = -delta @ tf.constant([[2., .3], [.3, 4.]], D)
        values = .5 * tf.reduce_sum(delta * scores, axis=1)
        if kind == "nonlinear":
            values -= .1 * tf.reduce_sum(delta**4, axis=1)
            scores -= .4 * delta**3
        if kind == "nonfinite":
            values = tf.where(points[:, 0] > .4, tf.constant(float("nan"), D), values)
        if kind == "ties":
            values, scores = tf.ones([points.shape[0]], D), tf.zeros_like(points)
        return values, scores
    return target


@pytest.mark.parametrize("kind", ["quadratic", "nonlinear", "nonfinite", "ties"])
@pytest.mark.parametrize("stopping", ["converged_all", "converged_any"])
def test_complete_records_and_every_objective_observation(previous, kind, stopping):
    records, events = [], []
    for module in (previous, candidate):
        seen = []
        config = module.SequentialMapCovarianceConfig(locator_max_iterations=4,
            locator_max_line_search_iterations=7, locator_stopping_condition=stopping,
            max_exact_evaluations=1, seed=(2026, 930))
        result = module.estimate_sequential_map_covariance(_target(kind),
            [[.5, -.6], [-.3, .2], [.1, -.1]], batched_locator_value_and_score_fn=_batch(kind),
            scale=[.7, 1.3], config=config, progress_callback=seen.append)
        for event in seen:
            if module is candidate and event["stage"] == "locator_objective_completed":
                assert event.pop("delivery_mode") == "buffered_after_compiled_locator"
        records.append(result.payload())
        events.append(seen)
    _compare(records[1], records[0])
    _compare(events[1], events[0])


@pytest.mark.parametrize("kind", ["quadratic", "nonlinear"])
def test_unbuffered_downstream_refinement_keeps_original_records(previous, kind):
    records = []
    for module in (previous, candidate):
        result = module.estimate_sequential_map_covariance(_target(kind), [[.4, -.5], [-.3, .2]],
            batched_locator_value_and_score_fn=_batch(kind), scale=[.7, 1.3],
            config=module.SequentialMapCovarianceConfig(locator_max_iterations=1,
                max_attempts=2, search_sample_count=8, regression_sample_count=16,
                terminal_sample_count=16, max_exact_evaluations=128,
                record_refinement_movement_diagnostics=True, seed=(2026, 931)))
        records.append(result.payload())
    _compare(records[1], records[0])


def test_exact_scalar_authority_target_order_and_invalid_endpoint_skip(previous):
    order, calls = tf.Variable(0, dtype=tf.int64), tf.Variable(0, dtype=tf.int64)

    def scalar(point):
        first = order.assign(order * 17 + tf.cast(point[0], tf.int64) + 1)
        second = calls.assign_add(1)
        with tf.control_dependencies([first, second]):
            return -point[0], tf.zeros([2], D)

    def batched(points):
        first, second = order.assign(order * 17 + 9), calls.assign_add(3)
        with tf.control_dependencies([first, second]):
            return tf.where(points[:, 0] == 2., tf.constant(float("nan"), D), points[:, 0]), tf.zeros_like(points)

    records, observations = [], []
    for module in (previous, candidate):
        order.assign(0)
        calls.assign(0)
        result = module.estimate_sequential_map_covariance(scalar, [[0., 0.], [1., 0.], [2., 0.]],
            batched_locator_value_and_score_fn=batched,
            config=module.SequentialMapCovarianceConfig(locator_max_iterations=1,
                locator_max_line_search_iterations=2, max_exact_evaluations=1))
        records.append(result.payload())
        observations.append((int(order), int(calls)))
    _compare(records[1], records[0])
    assert observations[1] == observations[0]
    np.testing.assert_array_equal(records[1]["map_candidate"], [0., 0.])


def test_optional_trace_is_reset_and_concurrent_calls_have_separate_snapshots():
    starts = tf.constant([[.4, -.5], [-.3, .2]], D)
    scale = tf.constant([.7, 1.3], D)
    program = BufferedBatchedLocator(_target("quadratic"), _batch("quadratic"),
        2, 2, 4., 1e-8, 4, 7, "converged_all", device=starts.device, capacity=128)
    assert program.calls.device == program.rows.device == starts.device
    before = program(starts, scale)
    frozen = before["trace"].numpy().copy()
    alternate = program(starts + .1, scale)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(program, points, scale) for points in (starts, starts + .1)]
        after, other = [future.result() for future in pending]
    _compare(after, before)
    _compare(other, alternate)
    np.testing.assert_array_equal(before["trace"], frozen)
    assert int(after["trace_count"]) == int(after["objective_calls"])


def test_optional_trace_overflow_is_explicit_without_numerical_shortcut():
    starts = tf.constant([[.4, -.5], [-.3, .2]], D)
    scale = tf.ones([2], D)
    program = BufferedBatchedLocator(_target("quadratic"), _batch("quadratic"),
        2, 2, 4., 1e-8, 4, 7, "converged_all", device=starts.device, capacity=1)
    with pytest.raises(RuntimeError, match="No partial progress was delivered"):
        program(starts, scale)
    result = program.compiled(starts, scale)
    plain = batched_locator_program(_target("quadratic"), _batch("quadratic"),
        2, 2, 4., 1e-8, 4, 7, "converged_all")(starts, scale)
    assert int(result["trace_count"]) == int(result["objective_calls"]) > 1
    _compare({key: result[key] for key in plain}, plain)


@pytest.mark.parametrize("buffered", [False, True])
def test_bounded_graphs_no_host_callbacks_and_no_hidden_xla_in_graph_mode(buffered):
    counts = []
    scalar, batched = _target("quadratic"), _batch("quadratic")
    for count in (2, 4):
        starts = tf.reshape(tf.linspace(tf.constant(-.4, D), tf.constant(.3, D), 2 * count), [count, 2])
        inputs = starts, tf.constant([.7, 1.3], D)
        pair, sizes = [], []
        for jit in (False, True):
            arguments = (scalar, batched, count, 2, 4., .05, 4, 7, "converged_all")
            if buffered:
                owner = BufferedBatchedLocator(*arguments, device=starts.device, jit_compile=jit, capacity=128)
                program = owner.compiled
            else:
                program = batched_locator_program(*arguments, jit_compile=jit)
            pair.append(program(*inputs))
            graph = program.get_concrete_function().graph.as_graph_def()
            nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
            assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)
            assert any(node.op in ("While", "StatelessWhile") for node in nodes)
            if jit:
                assert "HloModule" in program.experimental_get_compiler_ir(*inputs)(stage="hlo")
            else:
                assert not any(function.attr["_XlaMustCompile"].b for function in graph.library.function
                    if "_XlaMustCompile" in function.attr)
            assert program.experimental_get_tracing_count() == 1
            sizes.append(len(nodes))
        _compare(pair[1], pair[0])
        counts.append(sizes)
    assert counts[0] == counts[1]


def test_unsupported_batched_host_target_cannot_fall_back():
    def batched(points):
        points.numpy()
        return tf.ones([2], D), tf.zeros_like(points)

    with pytest.raises(AttributeError, match="numpy"):
        candidate.estimate_sequential_map_covariance(_target("ties"), [[0., 0.], [1., 0.]],
            batched_locator_value_and_score_fn=batched)
