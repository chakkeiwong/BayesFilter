"""Pinned block-center decisions and complete numerical callback compilation.

The reference wrapper shares the current sequential locator. These checks do
not certify the still host-controlled sweep or sequential refinement lifecycle.
"""

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import block_coordinate_center as candidate
from tests.test_block_coordinate_center import (
    _batched_quadratic_target,
    _quadratic_target,
    _sequential_config,
)

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/block_coordinate_center.py"], text=True)
    spec = importlib.util.spec_from_loader("block_center_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned diagnostic reference
    return module


def _assert_record(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_record(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_record(left, right)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("coupling,stop", [(0., True), (3., True), (3., False)])
def test_complete_public_and_private_records_match_original(baseline, coupling, stop, batched):
    precision = np.array([[4., coupling], [coupling, 4.]])
    mode = np.array([1., -1.])
    target = _quadratic_target(precision, mode)
    batch_target = _batched_quadratic_target(precision, mode) if batched else None

    def run(module):
        events = []
        blocks = (module.BlockCoordinateCenterBlock("first", 0, 1, _sequential_config()),
            module.BlockCoordinateCenterBlock("second", 1, 2, _sequential_config()))
        result = module.locate_block_coordinate_center(target, [0., 0.], blocks=blocks,
            scale=[1., 1.], batched_value_and_score_fn=batch_target,
            config=module.BlockCoordinateCenterConfig(max_physical_target_rows=300,
                stop_on_material_reversal=stop), progress_callback=events.append)
        return result, events

    expected, expected_events = run(baseline)
    actual, actual_events = run(candidate)
    assert actual.payload() == expected.payload()
    _assert_record(actual.private_payload(), expected.private_payload())
    _assert_record(actual_events, expected_events)


@pytest.mark.parametrize("centers,threshold", [([], 1e-6), ([[0.]], 1e-6),
    ([[0.], [1.]], 1e-6), ([[0.], [1.], [0.]], 1e-6),
    ([[0.], [1.], [1.]], 1e-6), ([[0., 1.], [2., 1.], [3., 2.], [0., 1.]], 1e-6),
    ([[0.], [1.], [0.]], 1.), ([[0.], [1.], [0.]], np.nextafter(1., 0.)),
    ([[0.], [1.], [0.]], np.nextafter(1., np.inf))])
def test_cycle_decisions_and_strict_edges_match_original(baseline, centers, threshold):
    assert candidate.classify_center_trace_cycles(centers, threshold) == baseline.classify_center_trace_cycles(centers, threshold)


@pytest.mark.parametrize("centers,threshold", [([[0.], [np.nan]], 1e-6),
    ([[0.], [1., 2.]], 1e-6), ([[0.]], 0.), ([], np.inf)])
def test_invalid_cycles_keep_rejections(baseline, centers, threshold):
    for module in (baseline, candidate):
        with pytest.raises(ValueError):
            module.classify_center_trace_cycles(centers, threshold)


def test_reversal_and_partial_partition_thresholds():
    epsilon = math.sqrt(math.ulp(1.))
    previous = np.array([0., 1., 4., 4., 4.])
    current = np.array([epsilon, 1.25, 5., np.nextafter(5., np.inf), 5.1])
    floors, material, any_material = candidate._numerical_call(candidate._reversal_core,
        tf.constant(current, D), tf.constant(previous, D), tf.constant(1.25, D))
    expected_floors = epsilon * np.maximum(1., previous)
    expected_material = (current - previous > expected_floors) & (current > 1.25 * previous)
    np.testing.assert_array_equal(floors, expected_floors)
    np.testing.assert_array_equal(material, expected_material)
    assert bool(any_material) == bool(np.any(expected_material))
    score = tf.constant([-4., 10., 2., 30., -3.], D)
    scale = tf.constant([2., 1., .5, 1., 3.], D)
    maxima = candidate._numerical_call(candidate._block_maxima_core, score, scale,
        tf.constant([[4, 5], [0, 1], [2, 3]], tf.int32))
    np.testing.assert_array_equal(maxima, [9., 8., 1.])


@pytest.mark.parametrize("rows", [None, 3, 5])
def test_complete_conditional_callback_has_stable_xla_and_keeps_captured_center(rows):
    precision = np.array([[2., .1, .2], [.1, 3., .3], [.2, .3, 4.]])
    mode = np.array([.5, -.4, .2])
    target = (_quadratic_target if rows is None else _batched_quadratic_target)(precision, mode)
    program = candidate._block_target_program(target, 3, 1, 3, rows)
    block = tf.constant([.1, -.2] if rows is None else [[.1, -.2]] * rows, D)
    for first in (1., 2.):
        center = tf.constant([first, 9., 10.], D)
        values, scores, finite = program(center, block)
        full = np.array([first, .1, -.2] if rows is None else [[first, .1, -.2]] * rows)
        delta = full - mode
        expected_scores = -np.einsum("ij,...j->...i", precision, delta)
        expected_values = .5 * np.sum(delta * expected_scores, axis=-1)
        np.testing.assert_allclose(values, expected_values, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scores, expected_scores[..., 1:], rtol=1e-10, atol=1e-10)
        assert bool(finite)
    assert program.experimental_get_tracing_count() == 1
    assert "HloModule" in program.experimental_get_compiler_ir(center, block)(stage="hlo")
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)


def test_nonfinite_callback_rejects_on_host_and_in_compiled_execution(baseline):
    def target(point):
        return tf.constant(float("nan"), D), tf.ones_like(point)

    point = tf.constant([0., 0.], D)
    for module in (baseline, candidate):
        with pytest.raises(ValueError, match="finite"):
            module._full_value_score(target, point, 2)
    values, scores, finite = candidate._block_target_program(target, 2, 0, 1, None)(point, point[:1])
    assert not bool(finite)
    assert math.isnan(float(values))
    assert bool(tf.reduce_all(tf.math.is_nan(scores)))


def test_result_snapshot_and_serialization_preserve_legacy_buffers(baseline):
    vector = np.arange(4, dtype=">f8")[::2]
    score = tf.Variable([1., 2.], dtype=D)
    arguments = {"completed": True, "status": "fixture", "initial_center": vector, "final_center": vector,
        "initial_score": score, "final_score": score, "initial_objective": 0., "final_objective": 1.,
        "initial_score_l2": 2., "final_score_l2": 1., "initial_score_max_abs": 2., "final_score_max_abs": 1.,
        "completed_block_count": 1, "accepted_block_count": 1, "transaction_rejection_count": 0,
        "sequential_exact_evaluations": 3, "physical_target_rows": 5, "maximum_physical_target_rows": 130,
        "material_reversal_detected": False, "repeat_cycle_detected": False, "two_step_return_cycle_detected": False,
        "scheduled_block_maxima_no_worse": True, "stop_on_material_reversal": True,
        "require_scheduled_block_maxima_no_worse": True, "private_block_records": ({"vector": vector},)}
    actual = candidate.BlockCoordinateCenterResult(**arguments)
    expected = baseline.BlockCoordinateCenterResult(**arguments)
    vector[:] = 9.
    score.assign([3., 4.])
    assert tf.is_tensor(actual.initial_center)
    _assert_record(actual.private_payload(), expected.private_payload())
