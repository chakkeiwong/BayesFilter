"""Frozen-reference eligibility and budget accounting inside one XLA sequence."""

import ast
import hashlib
import math
import subprocess
import symtable
from pathlib import Path
from types import SimpleNamespace

import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_local_initializer as current
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64
I = tf.int64


def frozen_tracker():
    path = "bayesfilter/inference/posterior_local_initializer.py"
    source = subprocess.check_output(["git", "show", "031692a0b:" + path],
        cwd=Path(__file__).resolve().parents[1], text=True)
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.ClassDef) and item.name == "_EligibilityTrackingEvaluator")
    excerpt = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
    code = "from __future__ import annotations\n" + excerpt
    symbols = symtable.symtable(code, "031692a0b:" + path, "exec")

    def global_names(table):
        return {symbol.get_name() for symbol in table.get_symbols() if symbol.is_global() and symbol.is_referenced()} | set().union(
            *(global_names(child) for child in table.get_children()))

    names = global_names(symbols)
    assert names <= {"tf", "math", "int", "bool", "ValueError"}, names
    namespace = {"tf": tf, "math": math, "__name__": "frozen_posterior_tracker"}
    exec(compile(code, "031692a0b:" + path, "exec"), namespace)  # noqa: S102 - exact frozen reference class
    authority = {"file": path, "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "class_sha256": hashlib.sha256(excerpt.encode()).hexdigest(), "globals": sorted(names),
        "classification": "Exact frozen class; only TensorFlow and standard-library numerical globals."}
    return SimpleNamespace(_EligibilityTrackingEvaluator=namespace["_EligibilityTrackingEvaluator"]), authority


def evaluator(module, case, budget):
    calls = tf.Variable(0, dtype=I)
    points = tf.Variable(tf.zeros([8, 2], D))

    def batch(positions):
        count = tf.cast(tf.shape(positions)[0], I)
        end = calls.assign_add(count)
        update = points.scatter_nd_update((tf.range(count, dtype=I) + end - count)[:, None], positions)
        with tf.control_dependencies([update]):
            scores = -positions
            values = -.5 * tf.reduce_sum(positions ** 2, axis=1)
        bad = positions[:, 0] > 0
        if case in ("nonfinite", "false_finite_status"):
            values = tf.where(bad, tf.constant(float("nan"), D), values)
        if case == "sentinel":
            values = tf.where(bad, tf.constant(-1e30, D), values)
            scores = tf.where(bad[:, None], tf.zeros_like(scores), scores)
        return values, scores

    def scalar(position):
        values, scores = batch(position[None])
        return values[0], scores[0]

    def eligible_batch(positions):
        return positions[:, 0] <= 0 if case == "sentinel" else tf.ones([tf.shape(positions)[0]], tf.bool)

    with_status = case in ("sentinel", "false_finite_status")
    tracker = module._EligibilityTrackingEvaluator(scalar, dimension=2, max_rows=budget,
        batched_fn=batch, eligibility_fn=(lambda position: eligible_batch(position[None])[0]) if with_status else None,
        batched_eligibility_fn=eligible_batch if with_status else None)
    resources = (tracker.evaluated_rows, tracker.invalid_rows, tracker.mismatch_rows,
                 tracker.budget_exhausted, calls, points)

    def sequence(start, cloud, end):
        resets = tuple(resource.assign(tf.zeros_like(resource)) for resource in resources)
        with tf.control_dependencies(resets):
            first = tracker.scalar(start)
        with tf.control_dependencies(first):
            middle = tracker.batched(cloud)
        with tf.control_dependencies(middle):
            last = tracker.scalar(end)
        with tf.control_dependencies(last):
            return {"first": first, "middle": middle, "last": last,
                "evaluated_rows": tracker.evaluated_rows.read_value(),
                "invalid_rows": tracker.invalid_rows.read_value(),
                "mismatch_rows": tracker.mismatch_rows.read_value(),
                "budget_exhausted": tracker.budget_exhausted.read_value(),
                "physical_rows": calls.read_value(), "ordered_positions": points.read_value()}

    return sequence


@pytest.mark.parametrize("case,budget,rows,invalid,mismatch,exhausted", [
    ("healthy", 5, 5, 0, 0, False),
    ("healthy", 1, 1, 0, 0, True),
    ("healthy", 3, 2, 0, 0, True),
    ("healthy", 4, 4, 0, 0, True),
    ("nonfinite", 5, 5, 3, 0, False),
    ("sentinel", 5, 5, 3, 3, False),
    ("false_finite_status", 5, 5, 3, 3, False),
])
def test_native_tracker_sequence(case, budget, rows, invalid, mismatch, exhausted, request):
    prior, authority = frozen_tracker()
    reference = evaluator(prior, case, budget)
    program = tf.function(evaluator(current, case, budget), input_signature=[
        tf.TensorSpec([2], D), tf.TensorSpec([3, 2], D), tf.TensorSpec([2], D)],
        jit_compile=True, autograph=False)
    inputs = (tf.constant([-1., .2], D), tf.constant([[-.5, .1], [.5, .2], [1., .3]], D),
              tf.constant([.8, -.2], D))
    observations = []
    hlo = None
    for multiplier in (1., .8, 1.):
        args = tuple(value * multiplier for value in inputs)
        expected, actual = clean(reference(*args)), clean(program(*args))
        observations.append({"expected": expected, "actual": actual})
        current_hlo = stable_hlo(program.experimental_get_compiler_ir(*args)(stage="hlo"))
        if hlo is None:
            hlo = current_hlo
        assert current_hlo == hlo
    save(request, f"posterior-tracker-{case}-{budget}.json", {"comparisons": observations,
        "reference": "031692a0b", "reference_authority": authority, "jit_compile": True,
        "scope": "Existing tracker in an enclosing scalar/batch/scalar sequence, not full initializer."})
    for observation in observations:
        actual = observation["actual"]
        _equal_records(actual, observation["expected"])
        assert actual["physical_rows"] == actual["evaluated_rows"] == rows
        assert actual["invalid_rows"] == invalid and actual["mismatch_rows"] == mismatch
        assert actual["budget_exhausted"] is exhausted
    assert observations[0] == observations[2]
    assert program.experimental_get_tracing_count() == 1
