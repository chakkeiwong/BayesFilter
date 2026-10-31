"""Independent prefix-selection, frozen-authority and execution diagnostics."""

import dataclasses
import math

import pytest
import tensorflow as tf

from bayesfilter.inference.posterior_candidate_ledger_tf import candidate_ledger_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def fixture(dimension, case):
    values = [-0., 0., 2., 2., -1., 3., 3., 4., 4., 1., 5.]
    points = [[.1 * (row + 1) + .03 * col for col in range(dimension)] for row in range(len(values))]
    scores = [[.2 * (row - 2) - .07 * col for col in range(dimension)] for row in range(len(values))]
    eligible = [True] * len(values)
    if case == "adverse":
        values[0] = -1e30
        eligible[0] = False
        points[1][0] = math.nan
        scores[2][0] = math.inf
        values[3:6] = [math.nan, math.inf, -math.inf]
        eligible[10] = False
    if case == "ineligible":
        eligible = [False] * len(values)
    scale = [.5 + index for index in range(dimension)]
    return tuple(tf.constant(value, dtype) for value, dtype in
                 ((points, D), (values, D), (scores, D), (eligible, tf.bool), (scale, D)))


def reference(module, operands, count):
    positions, values, scores, eligibility, scale = (value.numpy().tolist() for value in operands)
    capacity, dimension = len(values), len(scale)
    valid_count = 0 <= count <= capacity
    expected = {"input_valid": valid_count, "recorded_count": count if valid_count else 0,
        "selected_index": -1, "selected_present": False, "position": [0.] * dimension,
        "value": 0., "score": [0.] * dimension, "candidate_eligible": [False] * capacity,
        "promoted": [False] * capacity, "selected_prefix_indices": [-1] * capacity,
        "scaled_score_l2": [0.] * capacity}
    records, independent, selected_records = [], [], []
    for index in range(count if valid_count else 0):
        candidate = module.ExactCandidate(position=tf.constant(positions[index], D),
            value=values[index], score=tf.constant(scores[index], D), evaluation_index=index,
            source_role="diagnostic_row", eligible=eligibility[index])
        records.append(candidate)
        selected = module.select_exact_incumbent(records)
        valid = eligibility[index] and all(math.isfinite(x) for x in
            [values[index], *positions[index], *scores[index]])
        assert candidate.strict_finite_eligible is valid
        if valid:
            independent.append(index)
        winner = max(independent, key=lambda row: values[row]) if independent else -1
        assert (-1 if selected is None else selected.evaluation_index) == winner
        expected["candidate_eligible"][index] = valid
        expected["promoted"][index] = winner == index
        expected["selected_prefix_indices"][index] = winner
        expected["scaled_score_l2"][index] = math.sqrt(sum((x * unit) ** 2 for x, unit in
                                                          zip(scores[index], scale, strict=True)))
        if selected is not None:
            expected.update(selected_index=winner, selected_present=True, position=positions[winner],
                            value=values[winner], score=scores[winner])
            selected_records.append(clean(dataclasses.asdict(selected)))
        else:
            selected_records.append(None)
    return clean(expected), selected_records


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("case", ["healthy", "adverse", "ineligible"])
def test_complete_prefix_ledger(dimension, case, request):
    frozen = FrozenCheckpoint("031692a0b", "posterior_ledger_reference")
    module = frozen.load("bayesfilter.inference._exact_incumbent")
    inputs = fixture(dimension, case)
    capacity = int(inputs[1].shape[0])
    compiled = candidate_ledger_program(capacity, dimension)
    graph = candidate_ledger_program(capacity, dimension, jit_compile=False)
    observations = []
    for count in (0, 1, capacity - 1, capacity, -1, capacity + 1):
        expected, selected = reference(module, inputs, count)
        args = (*inputs, tf.constant(count, tf.int32))
        actual = clean(compiled(*args))
        graph_record = clean(graph(*args))
        observations.append({"active_count": count, "expected": expected, "xla": actual,
                             "graph": graph_record, "selected_original_records": selected})
    save(request, f"posterior-ledger-{dimension}-{case}.json", {"reference": "031692a0b",
        "reference_sources": frozen.hashes(), "comparisons": observations})
    for observation in observations:
        _equal_records(observation["xla"], observation["expected"])
        _equal_records(observation["graph"], observation["expected"])
    assert compiled.experimental_get_tracing_count() == graph.experimental_get_tracing_count() == 1


def test_changed_inputs_hlo_replay_and_frozen_gradients(request):
    original = fixture(3, "healthy")
    changed = (original[0] + .17, -original[1] + .2, original[2] - .11, original[3], original[4] * 1.1)
    program = candidate_ledger_program(11, 3)
    full, partial = tf.constant(11, tf.int32), tf.constant(7, tf.int32)
    first = clean(program(*original, full))
    before = stable_hlo(program.experimental_get_compiler_ir(*original, full)(stage="hlo"))
    other = clean(program(*changed, partial))
    replay = clean(program(*original, full))
    assert first == replay and first["selected_index"] != other["selected_index"]
    assert before == stable_hlo(program.experimental_get_compiler_ir(*changed, partial)(stage="hlo"))
    assert program.experimental_get_tracing_count() == 1
    positions, values, scores = (tf.Variable(value) for value in original[:3])
    with tf.GradientTape() as tape:
        raw = program(positions, values, scores, *original[3:], full)
        total = raw["value"] + tf.reduce_sum(raw["position"] + raw["score"] + raw["scaled_score_l2"][:3])
    assert tape.gradient(total, (positions, values, scores)) == (None, None, None)
    graph = program.get_concrete_function().graph.as_graph_def()
    operations = {node.op for nodes in (graph.node, *(f.node_def for f in graph.library.function)) for node in nodes}
    assert not operations & {"PyFunc", "PyFuncStateless", "EagerPyFunc"}
    assert program.function_spec.jit_compile is True
    save(request, "posterior-ledger-ownership.json", {"one_trace": True, "stable_hlo": True,
        "same_backend_replay": True, "frozen_gradients": True, "operations": sorted(operations)})


@pytest.mark.parametrize("count", [-1, 0, 1])
def test_zero_capacity_and_configuration(count):
    inputs = (tf.zeros([0, 1], D), tf.zeros([0], D), tf.zeros([0, 1], D),
              tf.zeros([0], tf.bool), tf.ones([1], D), tf.constant(count, tf.int32))
    result = candidate_ledger_program(0, 1)(*inputs)
    assert bool(result["input_valid"]) is (count == 0)
    assert not bool(result["selected_present"]) and int(result["selected_index"]) == -1
    assert int(result["recorded_count"]) == 0 and result["promoted"].shape == (0,)
    with pytest.raises(ValueError):
        candidate_ledger_program(-1, 1)
    with pytest.raises(ValueError):
        candidate_ledger_program(1, 0)
