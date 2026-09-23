"""Original complete records and actual no-use checks for enclosing dense fits."""

import dataclasses
import gc
import hashlib
import importlib.util
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import dense_validated_fit_tf as native
from bayesfilter.inference import fixed_center_curvature as current
from bayesfilter.inference import fixed_center_fitting_tf as fitting
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_dense_partition_validation import error_record
from tests.test_filter_repair_fixed_fitting import _inputs, _thresholds
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def original_fitter():
    repo = Path(__file__).resolve().parents[1]
    path = "bayesfilter/inference/fixed_center_curvature.py"
    source = subprocess.check_output(["git", "show", f"3582b4ac:{path}"], cwd=repo, text=True)
    spec = importlib.util.spec_from_loader("dense_validated_fit_original", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102
    factor = _baseline()
    module.fit_factor_correlation_score_geometry = factor.fit_factor_correlation_score_geometry
    module.FactorCorrelationGeometryConfig = factor.FactorCorrelationGeometryConfig
    factor_path = "bayesfilter/inference/factor_correlation_geometry.py"
    factor_source = subprocess.check_output(["git", "show", f"3582b4ac:{factor_path}"], cwd=repo)
    return module, {path: hashlib.sha256(source.encode()).hexdigest(),
        factor_path: hashlib.sha256(factor_source).hexdigest()}


def padded(inputs):
    center, score, train, train_score, select, select_score, audit, audit_score = inputs
    capacity = max(train.shape[1], select.shape[1], audit.shape[0])
    offsets = np.zeros((5, capacity, len(center)))
    scores = np.zeros_like(offsets)
    for index, (cloud, values) in enumerate([*zip(train, train_score, strict=True),
            *zip(select, select_score, strict=True), (audit, audit_score)]):
        offsets[index, :len(cloud)] = cloud
        scores[index, :len(cloud)] = values
    return tuple(tf.constant(value, D) for value in (center, score, offsets, scores))


def report_result(raw, operands, thresholds, dimension):
    invalid = error_record(raw["validation"])
    if invalid is not None:
        return {"error": invalid}
    try:
        result = current._fixed_center_result_from_native(raw["fit"], operands[0], operands[1],
            dimension=dimension, replicates=2, training_rows=3 * dimension,
            selection_rows=2 * dimension, audit_rows=2 * dimension, thresholds=thresholds,
            factor_max=2, weights=(0., .25, .5, .75, 1.), structured_target_family=None,
            lineage={"role": "dense_validated_fit_test"})
        return {"result": result.payload()}
    except ValueError as error:
        return {"error": {"type": type(error).__name__, "message": str(error)}}


def public_result(module, inputs, thresholds):
    try:
        result = module.fit_fixed_center_curvature(*inputs, thresholds=thresholds,
            factor_max=2, lineage={"role": "dense_validated_fit_test"}).payload()
        return {"result": result}
    except ValueError as error:
        return {"error": {"type": type(error).__name__, "message": str(error)}}


def normalized(record):
    record = clean(record)
    if "result" in record:
        for fit in record["result"]["fits"]:
            # This declared execution field was absent from original3582b4ac.
            fit["diagnostics"].pop("jit_compile", None)
    return record


@pytest.mark.parametrize("dimension,case", [(1, "healthy"), (3, "healthy"),
    (3, "audit"), (3, "incomplete"), (3, "rank")])
def test_validated_fitting_full_original_records(dimension, case, monkeypatch, request):
    original, hashes = original_fitter()
    thresholds = _thresholds(current, incomplete=case == "incomplete")
    if case == "rank":
        thresholds = dataclasses.replace(thresholds, principal_subspace_rank=dimension + 1)
    original_thresholds = original.FixedCenterCurvatureThresholds(**dataclasses.asdict(thresholds))
    fit_calls = tf.Variable(0, dtype=tf.int64)
    factory = fitting.fit_program.__wrapped__

    def counted_factory(*args, **kwargs):
        inner = factory(*args, **kwargs)

        @tf.function(input_signature=inner.input_signature, jit_compile=True, autograph=False)
        def counted(*operands):
            with tf.control_dependencies([fit_calls.assign_add(1)]):
                return inner(*operands)
        return counted

    monkeypatch.setattr(fitting.fit_program, "__wrapped__", counted_factory)
    program = native.make_dense_validated_fit_program(dimension, 2, 3 * dimension,
        2 * dimension, 2 * dimension, thresholds=thresholds)
    records, hlos = [], []
    baseline = _inputs(dimension, fault=case)
    variants = ("initial", "changed", "return", "invalid_center", "invalid_score", "overlap", "after_invalid")
    for variant in variants:
        inputs = tuple(value.copy() for value in baseline)
        if variant == "changed":
            inputs[0][:] += .07
            for index in (1, 3, 5, 7):
                inputs[index][:] += .125
        if variant == "invalid_center":
            inputs[0][0] = np.nan
        if variant == "invalid_score":
            inputs[7][0, 0] = np.inf
        if variant == "overlap":
            inputs[6][0] = inputs[2][0, -1]
        operands = padded(inputs)
        before_count = int(fit_calls)
        raw = program(*operands)
        actual = report_result(raw, operands, thresholds, dimension)
        expected = public_result(original, inputs, original_thresholds)
        public = public_result(current, inputs, thresholds)
        records.append({"variant": variant, "original": normalized(expected), "actual": normalized(actual),
            "public": normalized(public), "fit_calls": int(fit_calls) - before_count,
            "fit_ran": bool(raw["fit_ran"]), "fit_error_code": int(raw["fit_error_code"]),
            "usable": bool(raw["usable"]), "validation": clean(raw["validation"])})
        if variant in ("initial", "changed", "overlap", "after_invalid"):
            hlos.append(stable_hlo(program.experimental_get_compiler_ir(*operands)(stage="hlo")))
    graph = program.get_concrete_function().graph
    refs = {"program": weakref.ref(program), "graph": weakref.ref(graph),
        "scope": weakref.ref(program.dependency_scope)}
    traces = program.experimental_get_tracing_count()
    del program, graph
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    report = {"records": records, "original_sources": hashes, "trace_count": traces,
        "hlo_unchanged": len(set(hlos)) == 1, "python_released": released,
        "nonclaims": ["Prepared disjoint partitions only; full initializer and actual DZ5 integration remain.",
            "No performance, in-process native eviction or scientific admission claim."]}
    save(request, f"dense-validated-fit-{dimension}-{case}.json", report)
    for row in records:
        _equal_records(row["actual"], row["original"])
        _equal_records(row["public"], row["original"])
        assert row["fit_calls"] == int(row["fit_ran"])
        assert row["fit_ran"] is (row["validation"]["error_code"] == 0)
        assert row["usable"] is bool(row["actual"].get("result", {}).get("accepted", False))
        assert (row["fit_error_code"] != 0) is (case == "rank" and row["fit_ran"])
    assert traces == 1 and report["hlo_unchanged"] and all(released.values()), report


def test_fitting_error_precedence_is_tensor_native():
    @tf.function(input_signature=[tf.TensorSpec([8], tf.int32)], jit_compile=True, autograph=False)
    def decide(codes):
        return native.fit_error_code({"one_stability": {"complete": codes[0] != 0, "error": codes[1]},
            "selection": {"error": codes[7], "stability": {"complete": codes[2:5] != 0,
                "error": tf.stack([codes[5], codes[6], tf.constant(0)])}}})
    assert int(decide(tf.constant([1, 3, 1, 1, 1, 1, 2, 1]))) == 3
    assert int(decide(tf.constant([0, 3, 1, 1, 1, 1, 2, 1]))) == 1
    assert int(decide(tf.constant([0, 3, 0, 1, 1, 1, 2, 1]))) == 2
    assert int(decide(tf.constant([0, 3, 0, 0, 1, 1, 2, 1]))) == 4
    assert int(decide(tf.constant([0, 3, 0, 0, 1, 1, 2, 0]))) == 0


def test_enclosing_validation_recurrence_and_frozen_derivatives(request):
    original, hashes = original_fitter()
    thresholds = _thresholds(current)
    program = native.make_dense_validated_fit_program(3, 2, 9, 6, 6, thresholds=thresholds)

    def enclose(inner, freeze_result):
        template = inner.get_concrete_function().structured_outputs

        @tf.function(input_signature=inner.input_signature, jit_compile=True, autograph=False)
        def outer(center, score, offsets, scores):
            initial = tf.nest.map_structure(lambda value: tf.zeros(value.shape, value.dtype), template)

            def step(index, last, errors):
                del last
                # First reject an actual copied row, then fit the original cloud.
                bad = tf.tensor_scatter_nd_update(offsets, [[4, 0]], offsets[0, -1:])
                chosen = tf.where(index == 0, bad, offsets)
                result = inner(center, score, chosen, scores)
                return index + 1, result, tf.tensor_scatter_nd_update(errors, [[index]],
                    [result["validation"]["error_code"]])

            _, result, errors = tf.while_loop(lambda index, *_: index < 2, step,
                (tf.constant(0), initial, tf.zeros([2], tf.int32)), maximum_iterations=2,
                parallel_iterations=1)
            completed = (result, errors)
            return tf.nest.map_structure(tf.stop_gradient, completed) if freeze_result else completed
        return outer

    raw_outer = enclose(program, False)
    probe = padded(_inputs(3))
    with tf.GradientTape() as probe_tape:
        probe_tape.watch(probe)
        raw_probe, _ = raw_outer(*probe)
        probe_value = tf.reduce_sum(raw_probe["fit"]["covariance"])
    raw_derivatives = probe_tape.gradient(probe_value, probe)
    raw_derivative_maxima = [None if value is None else float(tf.reduce_max(tf.abs(value)))
        for value in raw_derivatives]
    outer = enclose(program, True)
    reports, hlos = [], []
    for shift in (0., .125, 0.):
        inputs = _inputs(3)
        inputs[0][:] += shift
        for index in (1, 3, 5, 7):
            inputs[index][:] += shift
        operands = padded(inputs)
        with tf.GradientTape() as tape:
            tape.watch(operands)
            raw, errors = outer(*operands)
            scalar = tf.reduce_sum(raw["fit"]["covariance"])
        derivatives = tape.gradient(scalar, operands)
        actual = normalized(report_result(raw, operands, thresholds, 3))
        expected = normalized(public_result(original, inputs, _thresholds(original)))
        reports.append({"actual": actual, "original": expected, "errors": errors.numpy().tolist(),
            "frozen_derivatives": [value is None for value in derivatives]})
        hlos.append(stable_hlo(outer.experimental_get_compiler_ir(*operands)(stage="hlo")))
    inner_graph, outer_graph = program.get_concrete_function().graph, outer.get_concrete_function().graph
    traces = [program.experimental_get_tracing_count(), outer.experimental_get_tracing_count()]
    refs = {"program": weakref.ref(program), "outer": weakref.ref(outer),
        "inner_graph": weakref.ref(inner_graph), "outer_graph": weakref.ref(outer_graph),
        "unfrozen_loop": weakref.ref(raw_outer), "unfrozen_loop_graph": weakref.ref(raw_outer.get_concrete_function().graph)}
    del program, outer, raw_outer, inner_graph, outer_graph, tape, probe_tape
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    report = {"records": reports, "original_sources": hashes, "traces": traces,
        "hlo_unchanged": len(set(hlos)) == 1, "python_released": released,
        "prior_failure_run": "03413", "unfrozen_loop_derivative_maxima": raw_derivative_maxima,
        "outer_boundary": "Explicit stop_gradient after the enclosing loop preserves original disconnected outputs."}
    save(request, "dense-validated-fit-enclosing.json", report)
    for row in reports:
        _equal_records(row["actual"], row["original"])
        assert row["errors"] == [6, 0] and all(row["frozen_derivatives"])
    assert all(value is None or value == 0. for value in raw_derivative_maxima)
    assert traces == [1, 1] and report["hlo_unchanged"] and all(released.values()), report
