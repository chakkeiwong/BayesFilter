"""Pinned and independent references for complete XLA incumbent selection."""

import importlib.util
import math
import subprocess
import sys
from itertools import accumulate

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import _exact_incumbent as candidate


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/_exact_incumbent.py"], text=True)
    spec = importlib.util.spec_from_loader("exact_incumbent_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned independent reference
    return module


def _records(module, rows):
    return tuple(module.ExactCandidate(position=position, score=score, value=value,
        eligible=eligible, evaluation_index=100 - index, source_role=str(index))
        for index, (position, score, value, eligible) in enumerate(rows))


@pytest.mark.parametrize("rows", [
    [],
    [([], [], 1., True)],
    [([], [], 1., False)],
    [([0.], [0.], -0., True), ([], [], 0., True)],
    [([0.], [0.], 0., True), ([], [], -0., True)],
    [([math.nan], [0.], 100., True), ([0., 1.], [0., 0.], 1., True)],
    [([0.], [math.inf], 100., True), ([], [], -3., True)],
    [([], [], math.inf, True), ([], [], -math.inf, True), ([], [], math.nan, True)],
    [([0.], [0.], 100., False), ([], [], -5., True), ([1., 2., 3.], [0., 0., 0.], -5., True)],
    [([math.nan], [math.inf], 1., False)],
])
def test_selection_preserves_original_identity_ties_and_ragged_eligibility(baseline, rows):
    original, records = _records(baseline, rows), _records(candidate, rows)
    expected = baseline.select_exact_incumbent(original)
    actual = candidate.select_exact_incumbent(iter(records))
    assert (actual is None) == (expected is None)
    if expected is not None:
        index = int(expected.source_role)
        assert actual is records[index]
        assert actual.evaluation_index == expected.evaluation_index
        assert math.copysign(1., actual.value) == math.copysign(1., expected.value)
    for reference, record in zip(original, records, strict=True):
        assert record.strict_finite_eligible == reference.strict_finite_eligible


def test_tensor_selector_matches_independent_reference_with_one_trace_and_hlo():
    rng = np.random.default_rng(7013)  # Independent diagnostic inputs, not runtime RNG.
    for count in (0, 1, 7, 64):
        widths = rng.integers(0, 6, size=count)
        length = int(widths.sum())
        positions, scores = rng.normal(size=(2, length))
        values = rng.integers(-2, 3, size=count).astype(float)
        flags = rng.integers(0, 2, size=count).astype(bool)
        if length:
            positions[::7], scores[::11] = np.nan, np.inf
        if count:
            values[-1] = np.inf
        ends = list(accumulate(widths))
        expected, start = [], 0
        for value, flag, stop in zip(values, flags, ends, strict=True):
            expected.append(bool(flag and math.isfinite(value)
                and all(math.isfinite(x) for x in positions[start:stop])
                and all(math.isfinite(x) for x in scores[start:stop])))
            start = stop
        winner = max((i for i, finite in enumerate(expected) if finite),
            key=lambda i: values[i], default=-1)
        inputs = (tf.constant(positions, tf.float64), tf.constant(scores, tf.float64),
            tf.constant(values, tf.float64), tf.constant(flags), tf.constant(ends, tf.int32))
        mask, index = candidate._incumbent_selection(*inputs)
        np.testing.assert_array_equal(mask, expected)
        assert int(index) == winner
        assert "HloModule" in candidate._incumbent_selection.experimental_get_compiler_ir(*inputs)(stage="hlo")
    assert candidate._incumbent_selection.experimental_get_tracing_count() == 1
    graph = candidate._incumbent_selection.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)


def test_row_record_construction_preserves_eligibility_and_order(baseline):
    positions = np.arange(6.).reshape(3, 2)
    options = {"start_index": 7, "source_role": "cloud", "eligibility": [True, False, True]}
    actual = candidate.candidates_from_rows(positions, [1., 7., 2.], -positions, **options)
    expected = baseline.candidates_from_rows(positions, [1., 7., 2.], -positions, **options)
    positions[:] = -100.
    for record, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(record.position, reference.position)
        np.testing.assert_array_equal(record.score, reference.score)
        assert (record.value, record.evaluation_index, record.eligible, record.source_role) == (
            reference.value, reference.evaluation_index, reference.eligible, reference.source_role)
    assert candidate.select_exact_incumbent(actual) is actual[2]


def test_growing_histories_use_geometric_physical_compile_shapes(monkeypatch):
    original = candidate._incumbent_selection
    shapes = set()

    def record_shapes(*inputs):
        shapes.add((inputs[0].shape[0], inputs[2].shape[0]))
        return original(*inputs)

    monkeypatch.setattr(candidate, "_incumbent_selection", record_shapes)
    records = []
    for count in range(1, 18):
        records.append(candidate.ExactCandidate(position=[1., 2.], score=[0., 0.],
            value=float(count), evaluation_index=count, source_role="history"))
        assert candidate.select_exact_incumbent(records) is records[-1]
    assert shapes == {(2, 1), (4, 2), (8, 4), (16, 8), (32, 16), (64, 32)}


@pytest.mark.parametrize("count,width", [(0, 0), (0, 2), (1, 0), (3, 0)])
def test_empty_row_schemas_preserve_record_construction(baseline, count, width):
    values = np.arange(count, dtype=float)
    points = np.empty((count, width))
    options = {"start_index": 0, "source_role": "empty_schema"}
    expected = baseline.candidates_from_rows(points, values, points, **options)
    actual = candidate.candidates_from_rows(points, values, points, **options)
    assert len(actual) == len(expected)
    winner = candidate.select_exact_incumbent(actual)
    if count:
        assert winner is actual[-1]
    else:
        assert winner is None
