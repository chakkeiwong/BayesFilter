"""Independent original input checks for compiled dense partition validation."""

import ast
import gc
import hashlib
import itertools
import subprocess
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as current
from bayesfilter.inference.dense_partition_validation_tf import (
    make_dense_partition_validation_program,
)
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save

D = tf.float64


def original_validation_source():
    path = "bayesfilter/inference/fixed_center_curvature.py"
    source = subprocess.check_output(["git", "show", f"3582b4ac:{path}"],
        cwd=Path(__file__).resolve().parents[1], text=True)
    names = {"_vector", "_cloud_partitions", "_cloud_pair", "_require_independent_partitions"}
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        and node.name in names]
    assert len(functions) == len(names)
    module = ast.fix_missing_locations(ast.Module(body=[
        ast.parse("from __future__ import annotations").body[0], *functions], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, f"3582b4ac:{path}:validation", "exec"), namespace)  # noqa: S102
    authority = {"revision": "3582b4ac", "source_path": path,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "excerpt": ast.unparse(module),
        "excerpt_sha256": hashlib.sha256(ast.unparse(module).encode()).hexdigest()}
    return SimpleNamespace(**{name: namespace[name] for name in names}), authority


def original_check(module, center, score, offsets, scores, rows):
    """Execute the actual public helpers in their public caller's order."""
    try:
        module._vector(center, "center")
        module._vector(score, "center_score_z")
        dimension = len(center)
        train = module._cloud_partitions(offsets[:2, :rows[0]].copy(),
            scores[:2, :rows[0]].copy(), dimension, "training")
        select = module._cloud_partitions(offsets[2:4, :rows[2]].copy(),
            scores[2:4, :rows[2]].copy(), dimension, "selection")
        audit, _ = module._cloud_pair(offsets[4, :rows[4]].copy(),
            scores[4, :rows[4]].copy(), dimension, "audit")
        named = [(f"training[{i}]", values[0]) for i, values in enumerate(train)]
        named += [(f"selection[{i}]", values[0]) for i, values in enumerate(select)]
        module._require_independent_partitions([*named, ("audit", audit)])
    except ValueError as error:
        return {"type": type(error).__name__, "message": str(error)}
    return None


def error_record(raw):
    code = int(raw["error_code"])
    if code == 0:
        return None
    messages = ("", "center must be a nonempty finite vector",
        "center_score_z must be a nonempty finite vector",
        "training offsets/scores must be finite", "selection offsets/scores must be finite",
        "audit offsets/scores must be finite")
    if code == 6:
        names = ("training[0]", "training[1]", "selection[0]", "selection[1]", "audit")
        left, right = raw["overlap_pair"].numpy().tolist()
        message = f"partition offsets contain copied rows: {names[left]} and {names[right]} overlap"
    else:
        message = messages[code]
    return {"type": "ValueError", "message": message}


@pytest.mark.parametrize("dimension", [1, 3])
def test_native_validation_preserves_original_error_order(dimension, request):
    original, authority = original_validation_source()
    rows = [3 * dimension] * 2 + [2 * dimension] * 2 + [2 * dimension + 1]
    capacity = max(rows)
    shape = (5, capacity, dimension)
    program = make_dense_partition_validation_program(dimension, 2, rows[0], rows[2], rows[4])

    def enclose(inner):
        @tf.function(input_signature=inner.input_signature, jit_compile=True, autograph=False)
        def outer(center, score, offsets, scores):
            # This status must be usable directly by the future fit/no-fit branch.
            result = inner(center, score, offsets, scores)
            return {**result, "fit_permitted": result["error_code"] == 0}
        return outer

    outer = enclose(program)
    cases = [("healthy", None), ("changed", None), ("return", None),
        ("within_partition", None), ("signed_zero", None), ("distinct_subnormal", None),
        ("zero_vs_subnormal", None), ("same_subnormal", None), ("nonfinite_padding", None),
        ("many_overlaps", None), ("finite_error_before_overlap", None)]
    cases += [("overlap", pair) for pair in itertools.combinations(range(5), 2)]
    cases += [("nonfinite", index) for index in range(5)]
    cases += [("nonfinite_score", index) for index in range(5)]
    cases += [("center_and_score_nonfinite", None), ("score_nonfinite", None),
        ("all_roles_nonfinite", None)]
    records, hlos = [], []
    for case, selector in cases:
        center = np.full(dimension, .2)
        score = np.full(dimension, -.3)
        offsets = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 16 + 1.
        scores = -offsets * 2.
        if case == "changed":
            center += .07
            score -= .03
            offsets += .25
        if case == "within_partition":
            offsets[0, 1] = offsets[0, 0]
        if case == "overlap":
            left, right = selector
            offsets[right, 0] = offsets[left, rows[left] - 1]
        if case in ("signed_zero", "distinct_subnormal", "zero_vs_subnormal", "same_subnormal"):
            encodings = {"signed_zero": (0, -(2**63)), "distinct_subnormal": (1, 2),
                "zero_vs_subnormal": (0, 1), "same_subnormal": (1, 1)}[case]
            offsets[0, 0] = np.full(dimension, encodings[0], dtype=np.int64).view(np.float64)
            offsets[4, 0] = np.full(dimension, encodings[1], dtype=np.int64).view(np.float64)
        if case == "nonfinite_padding":
            for index, n in enumerate(rows):
                offsets[index, n:] = np.nan
                scores[index, n:] = np.inf
        if case in ("many_overlaps", "finite_error_before_overlap", "all_roles_nonfinite"):
            offsets[1, 0] = offsets[0, 0]
            offsets[4, 0] = offsets[2, 0]
        if case == "nonfinite":
            offsets[selector, 0, 0] = np.nan
        if case == "nonfinite_score":
            scores[selector, 0, 0] = np.inf
        if case in ("center_and_score_nonfinite", "score_nonfinite"):
            score[0] = np.inf
        if case == "center_and_score_nonfinite":
            center[0] = np.nan
        if case in ("finite_error_before_overlap", "all_roles_nonfinite"):
            scores[4, 0, 0] = np.inf
        if case == "all_roles_nonfinite":
            offsets[0, 0, 0] = np.nan
            offsets[2, 0, 0] = np.inf
        expected = original_check(original, center, score, offsets, scores, rows)
        current_error = original_check(current, center, score, offsets, scores, rows)
        operands = tuple(tf.convert_to_tensor(value, D) for value in (center, score, offsets, scores))
        raw = outer(*operands)
        actual = error_record(raw)
        records.append({"case": case, "selector": selector, "original": expected,
            "current": current_error, "actual": actual, "native": clean(raw)})
        if case in ("healthy", "changed", "return", "signed_zero", "distinct_subnormal", "all_roles_nonfinite"):
            hlos.append(stable_hlo(outer.experimental_get_compiler_ir(*operands)(stage="hlo")))
    graph = outer.get_concrete_function().graph
    inner_graph = program.get_concrete_function().graph
    refs = {"program": weakref.ref(program), "outer": weakref.ref(outer),
        "graph": weakref.ref(graph), "inner_graph": weakref.ref(inner_graph)}
    traces = (program.experimental_get_tracing_count(), outer.experimental_get_tracing_count())
    del program, outer, graph, inner_graph
    gc.collect()
    released = {key: ref() is None for key, ref in refs.items()}
    report = {"records": records, "original_sources": authority, "trace_counts": traces,
        "hlo_unchanged": len(set(hlos)) == 1, "python_released": released,
        "nonclaims": ["Disjoint prepared logical partitions only; original caller-buffer validation remains at preparation.",
            "No full fitting, RNG, initializer, performance or actual DZ5 qualification."]}
    save(request, f"dense-partition-validation-{dimension}.json", report)
    for record in records:
        assert record["original"] == record["actual"], record
        assert record["current"] == record["actual"], record
        assert record["native"]["fit_permitted"] is (record["original"] is None)
    assert traces == (1, 1) and report["hlo_unchanged"] and all(released.values()), report
