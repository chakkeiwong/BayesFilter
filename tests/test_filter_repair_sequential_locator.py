"""Pinned scalar locator records, exact call order and enclosing compilation."""

import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from bayesfilter.inference.sequential_locator_tf import scalar_locator_program
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "d91a1268:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("scalar_locator_pre_enclosure_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102
    return module


def _target(kind):
    def target(point):
        delta = point - tf.constant([.15, -.2], D)
        score = -tf.linalg.matvec(tf.constant([[2., .3], [.3, 4.]], D), delta)
        value = .5 * tf.reduce_sum(delta * score)
        if kind == "nonlinear":
            value -= .1 * tf.reduce_sum(delta**4)
            score -= .4 * delta**3
        if kind == "nonfinite":
            value = tf.where(point[0] > .4, tf.constant(float("nan"), D), value)
        if kind == "ties":
            value, score = tf.constant(1., D), tf.zeros([2], D)
        return value, score
    return target


@pytest.mark.parametrize("kind", ["quadratic", "nonlinear", "nonfinite", "ties"])
@pytest.mark.parametrize("stopping", ["converged_all", "converged_any"])
def test_complete_budget_rejection_keeps_optimizer_and_candidate_records(previous, kind, stopping):
    target = _target(kind)
    starts = [[.5, -.6], [-.3, .2], [.1, -.1]]
    records = []
    for module in (previous, candidate):
        config = module.SequentialMapCovarianceConfig(locator_max_iterations=4,
            locator_max_line_search_iterations=7, locator_stopping_condition=stopping,
            max_exact_evaluations=1, seed=(2026, 930))
        result = module.estimate_sequential_map_covariance(target, starts,
            scale=[.7, 1.3], config=config)
        records.append(result.payload())
    _compare(records[1], records[0])


@pytest.mark.parametrize("kind", ["quadratic", "nonlinear"])
def test_qualified_refinement_consumer_keeps_complete_original_records(previous, kind):
    target = _target(kind)
    records = []
    for module in (previous, candidate):
        config = module.SequentialMapCovarianceConfig(locator_max_iterations=1,
            max_attempts=2, search_sample_count=8, regression_sample_count=16,
            terminal_sample_count=16, max_exact_evaluations=128,
            record_refinement_movement_diagnostics=True, seed=(2026, 931))
        result = module.estimate_sequential_map_covariance(target, [[.4, -.5], [-.3, .2]],
            scale=[.7, 1.3], config=config)
        records.append(result.payload())
    _compare(records[1], records[0])


def test_ordered_calls_skip_ineligible_endpoints_without_extra_target_rows(previous):
    order = tf.Variable(0, dtype=tf.int64)
    calls = tf.Variable(0, dtype=tf.int64)

    def target(point):
        visited = order.assign(order * 17 + tf.cast(point[0], tf.int64) + 1)
        counted = calls.assign_add(1)
        with tf.control_dependencies([visited, counted]):
            return tf.where(point[0] == 2., tf.constant(float("nan"), D), point[0]), tf.zeros([2], D)

    starts = tf.constant([[0., 0.], [1., 0.], [2., 0.]], D)
    original = previous.estimate_sequential_map_covariance(target, starts,
        config=previous.SequentialMapCovarianceConfig(locator_standardized_box_radius=8.,
            locator_gradient_tolerance=1e-3, locator_max_iterations=1,
            locator_max_line_search_iterations=2, max_exact_evaluations=1))
    before_order, before_calls = int(order), int(calls)
    order.assign(0)
    calls.assign(0)
    result = scalar_locator_program(target, 3, 2, 8., 1e-3, 1, 2, "converged_all")(
        starts, tf.ones([2], D))
    expected_order = 0
    # TFP rechecks the NaN start during its initial failed line search. Keep
    # that call as well as each endpoint and every eligible exact replay.
    expected_rows = [0, 0, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1]
    for row in expected_rows:
        expected_order = expected_order * 17 + row + 1
    assert int(order) == before_order == expected_order
    assert int(calls) == before_calls == len(expected_rows)
    assert int(result['objective_evaluations']) == sum(
        row['objective_evaluations'] for row in original.diagnostics['locator']) == 4
    assert int(result['exact_evaluations']) == 8
    assert int(result['selected']['finite_count']) == 4
    assert int(result['selected']['index']) == 1
    np.testing.assert_array_equal(result['endpoint_finite'], [True, True, False])


def test_empty_start_bank_does_not_trace_or_call_a_target():
    def unused(point):
        raise AssertionError("empty start bank must not trace a target")

    result = scalar_locator_program(unused, 0, 2, 8., 1e-3, 1, 2, "converged_all")(
        tf.zeros([0, 2], D), tf.ones([2], D))
    assert int(result['selected']['index']) == -1
    assert int(result['selected']['finite_count']) == 0
    assert int(result['exact_evaluations']) == int(result['objective_evaluations']) == 0


def test_complete_graph_is_bounded_and_graph_diagnostic_has_no_hidden_xla():
    target = _target('quadratic')
    node_counts = []
    for count in (2, 4):
        starts = tf.reshape(tf.linspace(tf.constant(-.4, D), tf.constant(.3, D), 2 * count), [count, 2])
        arguments = starts, tf.constant([.7, 1.3], D)
        programs = [scalar_locator_program(target, count, 2, 8., .05, 4, 7,
            "converged_all", jit_compile=jit) for jit in (False, True)]
        before, after = [program(*arguments) for program in programs]
        _compare(after, before)
        assert 'HloModule' in programs[1].experimental_get_compiler_ir(*arguments)(stage='hlo')
        counts = []
        for jit, program in zip((False, True), programs, strict=True):
            graph = program.get_concrete_function().graph.as_graph_def()
            nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
            assert not any(node.op in ('PyFunc', 'EagerPyFunc', 'PyFuncStateless') for node in nodes)
            if not jit:
                assert not any(function.attr['_XlaMustCompile'].b for function in graph.library.function
                    if '_XlaMustCompile' in function.attr)
            assert program.experimental_get_tracing_count() == 1
            counts.append(len(nodes))
        node_counts.append(counts)
    assert node_counts[0] == node_counts[1]


def test_unsupported_host_callback_cannot_fall_back_to_eager():
    def target(point):
        point.numpy()
        return tf.constant(1., D), tf.zeros([2], D)

    with pytest.raises(AttributeError, match='numpy'):
        candidate.estimate_sequential_map_covariance(target, [[.2, -.1]],
            config=candidate.SequentialMapCovarianceConfig(locator_max_iterations=1))
