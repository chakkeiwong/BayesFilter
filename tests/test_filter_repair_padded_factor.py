"""Padded/compact factor-fitting parity for enclosing structured preparation."""

import inspect
import json
import re
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64


def _data(dimension, reused, capacity):
    fresh = 2 * dimension
    active = fresh + reused
    rows = fresh + capacity
    offsets = tf.reshape(tf.cast(tf.range(rows * dimension), D), [rows, dimension])
    offsets = .2 * tf.sin(.17 * offsets ** 2 + .3)
    center = tf.linspace(tf.constant(-.2, D), tf.constant(.3, D), dimension)
    precision = tf.linalg.diag(tf.linspace(tf.constant(1., D), tf.constant(3., D), dimension)) + .13
    scores = center[None, :] - offsets @ precision
    heldout = tf.reverse(offsets[:fresh], [0]) * .91
    heldout_scores = center[None, :] - heldout @ precision
    if reused:
        weights = tf.concat([tf.fill([fresh], tf.constant(.5 / fresh, D)),
            tf.fill([reused], tf.constant(.5 / reused, D))], 0)
    else:
        weights = tf.fill([fresh], tf.constant(1. / fresh, D))
    compact = (center, offsets[:active], scores[:active], heldout, heldout_scores, weights)
    # Invalid placeholders expose accidental use of an inactive observation.
    padded = (center, tf.concat([offsets[:active], tf.fill([rows - active, dimension], tf.constant(float("nan"), D))], 0),
        tf.concat([scores[:active], tf.fill([rows - active, dimension], tf.constant(float("nan"), D))], 0),
        heldout, heldout_scores, tf.pad(weights, [[0, rows - active]]), tf.constant(active))
    return compact, padded


def _public_numerics(result):
    optimizer = result["optimizer"]
    return {**{name: value for name, value in result.items() if name != "optimizer"},
        "optimizer_converged": optimizer.converged, "optimizer_failed": optimizer.failed,
        "optimizer_iterations": optimizer.num_iterations,
        "optimizer_evaluations": optimizer.num_objective_evaluations,
        "final_loss": optimizer.objective_value}


@pytest.mark.parametrize("dimension,factors", [(3, 1), (5, 2)])
@pytest.mark.parametrize("reused", [0, 1, 4])
@pytest.mark.parametrize("capacity", [4, 32])
def test_padded_fit_keeps_complete_public_numerics(dimension, factors, reused, capacity):
    compact, padded = _data(dimension, reused, capacity)
    config = factor.FactorCorrelationGeometryConfig(factor_count=factors)
    before = factor._make_factor_program(dimension, 2 * dimension + reused, 2 * dimension,
        config, True, factor._prediction_jacobian_diagnostics)(*compact)
    after = factor._make_factor_program(dimension, 2 * dimension + capacity, 2 * dimension,
        config, True, factor._prediction_jacobian_diagnostics, padded_training=True)(*padded)
    print("PADDED_FACTOR_DIAGNOSTIC " + json.dumps({"dimension": dimension, "factors": factors,
        "reused": reused, "capacity": capacity, "raw_optimizer_discrepancy": float(tf.reduce_max(tf.abs(
            before["optimizer"].position - after["optimizer"].position))),
        "jacobian_condition": [float(before["jacobian_condition"]), float(after["jacobian_condition"])],
        "iterations": [int(before["optimizer"].num_iterations), int(after["optimizer"].num_iterations)],
        "evaluations": [int(before["optimizer"].num_objective_evaluations), int(after["optimizer"].num_objective_evaluations)]}))
    _compare(_public_numerics(after), _public_numerics(before))


def test_active_row_count_has_one_trace_and_no_external_optimizer_derivative():
    config = factor.FactorCorrelationGeometryConfig(max_iterations=4)
    program = factor._make_factor_program(3, 10, 6, config, True,
        factor._prediction_jacobian_diagnostics, padded_training=True)
    for reused in (0, 1, 4):
        _, arguments = _data(3, reused, 4)
        with tf.GradientTape() as tape:
            tape.watch(arguments[2])
            result = program(*arguments)
            value = tf.reduce_sum(result["precision"])
        assert tape.gradient(value, arguments[2]) is None
        assert bool(result["finite"])
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
    assert any(node.op in ("While", "StatelessWhile") for node in nodes)
    assert not any(node.op in ("PyFunc", "EagerPyFunc", "PyFuncStateless") for node in nodes)
    assert "HloModule" in program.experimental_get_compiler_ir(*arguments)(stage="hlo")


def test_active_equation_count_not_capacity_sets_jacobian_threshold():
    config = factor.FactorCorrelationGeometryConfig()
    raw = tf.constant([.1, .3, .5, -.7, .2, -.4], D)
    offsets = tf.reshape(tf.linspace(tf.constant(-.2, D), tf.constant(.3, D), 18), [6, 3])
    padded = tf.pad(offsets, [[0, 20], [0, 0]])

    @tf.function(input_signature=[tf.TensorSpec([26, 3], D), tf.TensorSpec([], tf.int32)],
        jit_compile=True, autograph=False)
    def inspect(points, active):
        return factor._prediction_jacobian_diagnostics(raw, points, dimension=3,
            anchors=(1,), config=config, active_training_rows=active)

    before = factor._prediction_jacobian_diagnostics(raw, offsets, dimension=3,
        anchors=(1,), config=config, jit_compile=False)
    after = inspect(padded, tf.constant(6))
    assert int(before[0]) == int(after[0])
    np.testing.assert_allclose(after[1], before[1], atol=1e-10, rtol=1e-10)


def test_full_occupancy_initializer_and_same_state_loss_localization(monkeypatch):
    from tests.test_filter_repair_initializer_rounding import _record_differences

    compact, padded = _data(5, 4, 4)
    for before, after in zip(compact, padded[:6], strict=True):
        np.testing.assert_array_equal(before, after)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    diagnostic_result = namedtuple("InitializationDiagnostic", (
        "position", "objective_value", "objective_gradient", "converged", "failed",
        "num_iterations", "num_objective_evaluations"))

    def observe(objective, *, initial_position, **_):
        value, gradient = objective(initial_position)
        return diagnostic_result(initial_position, value, gradient, tf.constant(False),
            tf.constant(False), tf.constant(0), tf.constant(1))

    def execute(masked):
        return factor._make_factor_program(5, 14, 10, config, True,
            factor._prediction_jacobian_diagnostics, padded_training=masked)(*(padded if masked else compact))

    report = {"role": "explanatory_full_occupancy_localization_only", "inputs_exact": True}
    frozen = None
    for fixed in (False, True):
        with monkeypatch.context() as context:
            context.setattr(factor.tfp.optimizer, "lbfgs_minimize", observe)
            if fixed:
                assert frozen is not None
                context.setattr(factor, "_encode_state", lambda *_, state=frozen: state)
            factor._make_factor_program.cache_clear()
            try:
                before, after = execute(False), execute(True)
            finally:
                factor._make_factor_program.cache_clear()
        if not fixed:
            frozen = tf.identity(before["optimizer"].position)
        first, second = before["optimizer"], after["optimizer"]
        report["same_initial_state" if fixed else "computed_initial_state"] = {
            "position": [first.position.numpy().tolist(), second.position.numpy().tolist()],
            "value": [float(first.objective_value), float(second.objective_value)],
            "gradient": [first.objective_gradient.numpy().tolist(), second.objective_gradient.numpy().tolist()],
            "position_max_error": float(tf.reduce_max(tf.abs(first.position - second.position))),
            "gradient_max_error": float(tf.reduce_max(tf.abs(first.objective_gradient - second.objective_gradient)))}
        assert bool(tf.reduce_all(tf.math.is_finite(first.objective_gradient)))
        assert bool(tf.reduce_all(tf.math.is_finite(second.objective_gradient)))
    with monkeypatch.context() as context:
        context.setattr(factor, "_encode_state", lambda *_: frozen)
        factor._make_factor_program.cache_clear()
        try:
            before, after = execute(False), execute(True)
        finally:
            factor._make_factor_program.cache_clear()
    first, second = (tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(row))
        for row in (before, after))
    report["same_initial_complete_records"] = [first, second]
    report["same_initial_record_differences"] = _record_differences(second, first)
    print("PADDED_FACTOR_INITIALIZATION " + json.dumps(report, sort_keys=True))


def test_full_occupancy_loss_arithmetic_localization(monkeypatch, request):
    compact, padded = _data(5, 4, 4)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    fields = ("position", "objective_value", "objective_gradient", "converged", "failed",
        "num_iterations", "num_objective_evaluations", "prepared", "stages", "pullbacks")
    diagnostic_result = namedtuple("LossArithmeticDiagnostic", fields)

    def observe(objective, *, initial_position, **_):
        value, gradient = objective(initial_position)
        loss = inspect.getclosurevars(objective.python_function).nonlocals["loss"]
        closure = inspect.getclosurevars(loss).nonlocals
        weights, offsets, response = (closure[name] for name in ("weights", "train_z", "train_response"))
        anchors = closure["anchors"]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(initial_position)
            covariance, _, _ = factor._decode_covariance(initial_position,
                dimension=5, anchors=anchors, config=config)
            precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance), tf.eye(5, dtype=D))
            prediction = tf.einsum("ij,bj->bi", precision, offsets)
            residual = prediction - response
            per_row = tf.reduce_mean(tf.square(residual), axis=1)
            weighted = weights * per_row
            observed_value = tf.reduce_sum(weighted)
        stages = {"covariance": covariance, "precision": precision, "prediction": prediction,
            "residual": residual, "per_row": per_row, "weighted": weighted, "value": observed_value}
        pullbacks = {name: tape.gradient(observed_value, tensor) for name, tensor in
            {"raw": initial_position, **stages}.items()}
        del tape
        return diagnostic_result(initial_position, value, gradient, tf.constant(False),
            tf.constant(False), tf.constant(0), tf.constant(1),
            {"weights": weights, "offsets": offsets, "response": response, "anchors": tf.stack(anchors)},
            stages, pullbacks)

    directory = Path(request.config.getoption("xmlpath")).parent
    observations = []
    compiler_files = []
    with monkeypatch.context() as context:
        context.setattr(factor.tfp.optimizer, "lbfgs_minimize", observe)
        factor._make_factor_program.cache_clear()
        try:
            for masked, arguments in ((False, compact), (True, padded)):
                program = factor._make_factor_program(5, 14, 10, config, True,
                    factor._prediction_jacobian_diagnostics, padded_training=masked)
                observations.append(program(*arguments)["optimizer"])
                compiler_arguments = arguments if masked else (*arguments, None)
                for stage in ("hlo", "optimized_hlo"):
                    path = directory / f"padded-loss-{int(masked)}-{stage}.txt"
                    with path.open("x") as handle:
                        handle.write(program.experimental_get_compiler_ir(*compiler_arguments)(stage=stage))
                    compiler_files.append(str(path))
        finally:
            factor._make_factor_program.cache_clear()

    def compare(first, second):
        return {name: {"max_error": float(tf.reduce_max(tf.abs(
                tf.cast(first[name], D) - tf.cast(second[name], D)))),
            "before": first[name].numpy().tolist(), "after": second[name].numpy().tolist()}
            for name in first}

    before, after = observations
    report = {"role": "explanatory_loss_arithmetic_only", "compiler_files": compiler_files,
        "prepared": compare(before.prepared, after.prepared),
        "stages": compare(before.stages, after.stages),
        "pullbacks": compare(before.pullbacks, after.pullbacks),
        "objective": compare({"value": before.objective_value, "gradient": before.objective_gradient},
            {"value": after.objective_value, "gradient": after.objective_gradient}),
        "instrumentation_gradient_errors": [float(tf.reduce_max(tf.abs(
            row.objective_gradient - row.pullbacks["raw"]))) for row in observations]}
    for row in observations:
        assert bool(tf.reduce_all(tf.math.is_finite(row.objective_gradient)))
    assert all(row["max_error"] == 0 for row in report["prepared"].values())
    print("PADDED_FACTOR_ARITHMETIC " + json.dumps(report, sort_keys=True))


def test_factor_compiler_input_specialization_localization(request, monkeypatch):
    directory = Path(request.config.getoption("xmlpath")).parent
    report = []
    diagnostic_result = namedtuple("SpecializationDiagnostic", ("position", "converged", "failed"))

    def skip_optimizer(_objective, *, initial_position, **_):
        return diagnostic_result(initial_position, tf.constant(False), tf.constant(False))

    def skip_diagnosis(*_, **__):
        return tf.constant(0), tf.constant(0., D)

    for dimension, factors, omitted in ((3, 1, "optimizer"), (3, 1, "diagnosis"), (3, 1, "both"),
            (5, 2, "optimizer"), (5, 2, "diagnosis"), (5, 2, "both")):
        arguments, _ = _data(dimension, 4, 4)
        config = factor.FactorCorrelationGeometryConfig(factor_count=factors, max_iterations=4)
        with monkeypatch.context() as context:
            if omitted in ("optimizer", "both"):
                context.setattr(factor.tfp.optimizer, "lbfgs_minimize", skip_optimizer)
            diagnosis = skip_diagnosis if omitted in ("diagnosis", "both") else factor._prediction_jacobian_diagnostics
            factor._make_factor_program.cache_clear()
            try:
                program = factor._make_factor_program(dimension, 2 * dimension + 4, 2 * dimension,
                    config, True, diagnosis)
                result = program(*arguments)
                hlo = program.experimental_get_compiler_ir(*arguments, None)(stage="hlo")
            finally:
                factor._make_factor_program.cache_clear()
        with (directory / f"factor-{factors}-without-{omitted}-specialization-hlo.txt").open("x") as handle:
            handle.write(hlo)
        entry = hlo[hlo.rfind("\nENTRY "):]
        parameter_lines = [line for line in entry.splitlines() if re.search(r"\bparameter\(\d+\)", line)]
        report.append({"dimension": dimension, "factors": factors, "omitted": omitted, "expected_parameters": 6,
            "compiler_parameters": len(parameter_lines), "parameter_lines": parameter_lines,
            "trace_count": program.experimental_get_tracing_count()})
        assert bool(result["finite"])
    print("FACTOR_INPUT_SPECIALIZATION " + json.dumps(report, sort_keys=True))


@pytest.mark.parametrize("dimension,factors", [(3, 1), (5, 2)])
@pytest.mark.parametrize("masked", [False, True])
def test_complete_fitter_retains_runtime_inputs(dimension, factors, masked):
    compact, padded = _data(dimension, 4, 4)
    arguments = padded if masked else (*compact, None)
    config = factor.FactorCorrelationGeometryConfig(factor_count=factors, max_iterations=4)
    program = factor._make_factor_program(dimension, 2 * dimension + 4, 2 * dimension,
        config, True, factor._prediction_jacobian_diagnostics, padded_training=masked)
    first = program(*arguments)
    hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo")
    entry = hlo[hlo.rfind("\nENTRY "):]
    parameters = [line for line in entry.splitlines() if re.search(r"\bparameter\(\d+\)", line)]
    assert len(parameters) == (7 if masked else 6)
    # Different training data must use the same executable and affect the fit.
    changed = (*arguments[:2], arguments[2] * 1.03, *arguments[3:])
    second = program(*changed)
    changed_hlo = program.experimental_get_compiler_ir(*changed)(stage="hlo")
    assert changed_hlo == hlo
    assert program.experimental_get_tracing_count() == 1
    assert bool(first["finite"]) and bool(second["finite"])
    assert float(tf.reduce_max(tf.abs(first["precision"] - second["precision"]))) > 1e-5
