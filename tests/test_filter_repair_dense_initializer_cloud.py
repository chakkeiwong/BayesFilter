"""Diagnostic exact external-loop authority for native dense cloud evaluation."""

import ast
import contextlib
import difflib
import gc
import hashlib
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference.dense_initializer_cloud_tf import (
    make_dense_initializer_cloud_program,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64
SOURCE = Path(__file__).parent / "fixtures/filter_repair_dense_initializer_reference_2f386f75.py"
SOURCE_SHA = "7c4d5598959fcd9605bed01d16d416f2002fd442301e82264ff396b1aa050f75"


def original_partition_loop():
    source = SOURCE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_SHA
    tree = ast.parse(source)
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == "(partition_index, rows)")
    # Freeze identical prepared offsets, retaining every statement from positions
    # construction through target evaluation, validation, selection and storage.
    position_index = next(index for index, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "positions")
    loop.body = [ast.parse("offsets = clouds[partition_index][:rows]").body[0], *loop.body[position_index:]]

    class CompletedResult(ast.NodeTransformer):
        def visit_Return(self, node):
            assert ast.unparse(node.value) == "finish('dense_curvature_cloud_invalid')"
            return ast.parse("return False, candidate_center, candidate_value, exact_rows, partitions, archives").body[0]

    loop = CompletedResult().visit(loop)
    definition = ast.parse("def reference(combined, center, scale, center_value, partition_rows, clouds):\n"
        "    exact_rows, attempt = 0, 0\n"
        "    archives, partitions = {}, []\n"
        "    candidate_center, candidate_value = center, float(center_value)\n").body[0]
    definition.body.append(loop)
    definition.body.append(ast.parse("return True, candidate_center, candidate_value, exact_rows, partitions, archives").body[0])
    module = ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))
    excerpt = ast.unparse(module)
    namespace = {"tf": tf, "context": SimpleNamespace(boundary=lambda _: contextlib.nullcontext())}
    exec(compile(module, str(SOURCE) + ":frozen_cloud_loop", "exec"), namespace)  # noqa: S102
    return namespace["reference"], excerpt


@pytest.mark.parametrize("dimension", [1, 3])
@pytest.mark.parametrize("case", ["healthy", "improving", "ties", "invalid_first", "invalid_second",
    "nonfinite_value", "nonfinite_score"])
def test_complete_original_partition_loop(dimension, case, request):
    reference, excerpt = original_partition_loop()
    replicates = 2
    train, selection, audit = 3 * dimension, 2 * dimension, 2 * dimension + 1
    rows = [train] * replicates + [selection] * replicates + [audit]
    capacity, total = max(rows), sum(rows)
    clouds = tf.stack([tf.pad(.13 * tf.random.stateless_normal([n, dimension], [217, index], dtype=D),
        [[0, capacity - n], [0, 0]]) for index, n in enumerate(rows)])
    precision = tf.linalg.diag(tf.cast(tf.range(dimension), D) + 1.3) + .07
    mode = tf.fill([dimension], tf.constant(.8 if case == "improving" else 0., D))
    calls = tf.Variable(0, dtype=tf.int64)
    count = tf.Variable(0, dtype=tf.int64)
    locations = tf.Variable(tf.zeros([total, dimension], D))

    def target_body(points):
        batch_rows = points.shape[0]
        index = count.assign_add(batch_rows) - batch_rows
        call = calls.assign_add(1)
        updated = locations.scatter_nd_update((index + tf.range(batch_rows, dtype=tf.int64))[:, None], points)
        with tf.control_dependencies([updated, call]):
            delta = points - mode
            scores = -tf.linalg.matmul(delta, precision, transpose_b=True)
            values = .5 * tf.reduce_sum(delta * scores, axis=1)
        valid = tf.ones([batch_rows], tf.bool)
        if case == "ties":
            values, scores = tf.zeros_like(values), tf.zeros_like(scores)
        if case in ("invalid_first", "invalid_second"):
            valid &= call != (1 if case == "invalid_first" else 2)
        if case == "nonfinite_value":
            values = tf.where(call == 2, tf.fill([batch_rows], tf.constant(float("nan"), D)), values)
        if case == "nonfinite_score":
            scores = tf.where(call == 2, tf.fill([batch_rows, dimension], tf.constant(float("inf"), D)), scores)
        return values, scores, valid

    def callback(points):
        return target_body(points)

    program = make_dense_initializer_cloud_program(callback, dimension, replicates, train, selection, audit)
    records, hlos = [], []
    for shift in (0., .03, 0.):
        center = tf.fill([dimension], tf.constant(shift, D))
        scale = .8 + tf.cast(tf.range(dimension), D) * .09 + shift
        delta = center - mode
        value = .5 * tf.reduce_sum(delta * -tf.linalg.matvec(precision, delta)) if case != "ties" else tf.constant(0., D)
        operands = (center, scale, value, clouds + shift)
        calls.assign(0)
        count.assign(0)
        expected = reference(target_body, center, scale, value, rows, operands[3])
        original_calls = {"calls": int(calls), "rows": int(count), "positions": clean(locations[:int(count)])}
        calls.assign(0)
        count.assign(0)
        raw = program(*operands)
        actual_calls = {"calls": int(calls), "rows": int(count), "positions": clean(locations[:int(count)])}
        observed = clean(raw)
        archives = {}
        partitions = []
        for index in range(observed["partition_count"]):
            for name, key in (("positions", "positions"), ("values", "values"),
                ("scores", "scores"), ("valid", "row_validity")):
                archives[f"attempt_0_partition_{index}_{name}"] = observed[key][index][:rows[index]]
            if observed["valid"] or index < observed["partition_count"] - 1:
                partitions.append((clean(operands[3][index, :rows[index]]),
                    observed["scaled_scores"][index][:rows[index]]))
        actual = (observed["valid"], observed["candidate_center"], observed["candidate_value"],
            observed["exact_rows"], partitions, archives)
        records.append({"original": clean(expected), "actual": clean(actual),
            "original_calls": original_calls, "actual_calls": actual_calls})
        hlos.append(stable_hlo(program.experimental_get_compiler_ir(*operands)(stage="hlo")))
    graph = program.get_concrete_function().graph
    refs = {"program": weakref.ref(program), "graph": weakref.ref(graph), "callback": weakref.ref(callback)}
    traces = program.experimental_get_tracing_count()
    del program, graph, callback
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    report = {"original_source": str(SOURCE), "original_source_sha256": SOURCE_SHA,
        "reference_excerpt": excerpt, "reference_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "records": records, "trace_count": traces, "hlo_unchanged": len(set(hlos)) == 1,
        "hlo_differences": list(difflib.unified_diff(hlos[0].splitlines(), hlos[1].splitlines()))[:120],
        "python_released": released, "row_extents": rows,
        "nonclaims": ["Prepared identical offsets; no RNG migration, full initializer or actual DZ5 target qualification."]}
    save(request, f"dense-cloud-{dimension}-{case}.json", report)
    for row in records:
        _equal_records(row["actual"], row["original"])
        _equal_records(row["actual_calls"], row["original_calls"])
    assert traces == 1 and report["hlo_unchanged"] and all(released.values())
    expected_partitions = 1 if case == "invalid_first" else 2 if case in (
        "invalid_second", "nonfinite_value", "nonfinite_score") else len(rows)
    assert records[0]["actual_calls"]["calls"] == expected_partitions
    assert records[0]["actual_calls"]["rows"] == sum(rows[:expected_partitions])
