"""Frozen public fit lifecycle parity and native enclosing control checks."""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as candidate
from bayesfilter.inference import fixed_center_fitting_tf as native
from tests.test_filter_repair_fixed_stability import _compare

D = tf.float64


@pytest.fixture(scope="module")
def previous():
    source = subprocess.check_output(["git", "show",
        "22094f76:bayesfilter/inference/fixed_center_curvature.py"], text=True)
    spec = importlib.util.spec_from_loader("fixed_fitting_pre_enclosure_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned diagnostic
    # Isolate enclosure from the separately evidenced eigensystem repair.
    module._dense_fit_kernel = candidate._dense_fit_kernel
    module._fit_structured_precision = candidate._fit_structured_precision
    return module


def _inputs(dimension=2, replicates=2, *, fault=None):
    precision = np.diag(np.linspace(1., 3., dimension)) + .13
    if fault == "raw":
        precision[0, 0] = -1.
    center = np.linspace(-.2, .3, dimension)
    # Frozen deterministic, disjoint clouds; no RNG stream changes.
    def cloud(rows, phase):
        return .2 * np.sin(.17 * np.arange(rows * dimension).reshape(rows, dimension) ** 2 + phase)
    training = np.stack([cloud(3 * dimension, .1 + index) for index in range(replicates)])
    selection = np.stack([cloud(2 * dimension, .4 + index) for index in range(replicates)])
    audit = cloud(2 * dimension, .8)
    return (np.zeros(dimension), center, training, center - training @ precision,
        selection, center - selection @ (precision * (3. if fault == "holdout" else 1.)),
        audit, center - audit @ (precision * (3. if fault == "audit" else 1.)))


def _thresholds(module, *, incomplete=False):
    return module.FixedCenterCurvatureThresholds(selection_holdout_relative_rmse_cap=.1,
        audit_relative_rmse_cap=.1, projection_relative_frobenius_cap=.1,
        **({} if incomplete else {"generalized_eigenvalue_spread_cap": 2., "trace_normalized_frobenius_cap": .1,
            "trace_normalized_operator_cap": .1, "principal_angle_degrees_cap": 90., "principal_subspace_rank": 1}))


@pytest.mark.parametrize("dimension,factor_max,fault", [(1, 2, None), (2, 2, None), (2, 1, "raw"),
    (2, 1, "holdout"), (2, 1, "audit"), (2, 1, "incomplete"), (3, 1, None),
    (5, 2, "holdout"), (5, 2, "explicit_second")])
def test_complete_fit_records_preserve_pre_enclosure(previous, dimension, factor_max, fault):
    inputs = _inputs(dimension, fault=fault)
    before, after = (module.fit_fixed_center_curvature(*inputs,
        thresholds=_thresholds(module, incomplete=fault == "incomplete"), factor_max=factor_max,
        structured_target_family="factor_2" if fault == "explicit_second" else None).payload()
        for module in (previous, candidate))
    _compare(after, before)


def test_audit_veto_preserves_selected_candidate(previous):
    healthy, failed = (candidate.fit_fixed_center_curvature(*_inputs(fault=fault),
        thresholds=_thresholds(candidate), factor_max=1) for fault in (None, "audit"))
    assert healthy.accepted
    assert failed.status == "audit_holdout_rejected"
    assert healthy.selected_family == failed.selected_family
    np.testing.assert_array_equal(healthy.selected_precision_z, failed.selected_precision_z)


@pytest.mark.parametrize("endpoint", ["fixed", "factor"])
def test_public_fitted_geometry_preserves_frozen_derivative_boundary(endpoint):
    from tests.test_filter_repair_fixed_fitting_localization import _baseline

    dimension = 2 if endpoint == "fixed" else 3
    inputs = _inputs(dimension)
    if endpoint == "fixed":
        source = subprocess.check_output(["git", "show",
            "3582b4ac:bayesfilter/inference/fixed_center_curvature.py"], text=True)
        spec = importlib.util.spec_from_loader("fixed_fitting_original_frozen_reference", loader=None)
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        exec(compile(source, spec.name, "exec"), original.__dict__)  # noqa: S102

        def fit(module, training_scores):
            arguments = (*inputs[:3], training_scores, *inputs[4:])
            return module.fit_fixed_center_curvature(*arguments,
                thresholds=_thresholds(module), factor_max=1).selected_precision_z

        current = candidate
    else:
        original = _baseline()
        current = native.factor

        def fit(module, training_scores):
            return module.fit_factor_correlation_score_geometry(inputs[1], inputs[2][0],
                training_scores[0], inputs[4][0], inputs[5][0]).precision_z

    # The public preparation records were frozen through host materialization.
    # Neither executing under a tape nor requesting its gradient should expose
    # an optimizer/eigensolver pullback or change that result into a derivative.
    for module in (original, current):
        training_scores = tf.constant(inputs[3], D)
        with tf.GradientTape() as tape:
            tape.watch(training_scores)
            precision = fit(module, training_scores)
            assert precision is not None
            objective = tf.reduce_sum(tf.convert_to_tensor(precision, D))
        assert tape.gradient(objective, training_scores) is None


def test_original_factor_and_dense_lifecycle_fields():
    from tests.test_filter_repair_fixed_fitting_localization import _baseline

    original_factor = _baseline()
    source = subprocess.check_output(["git", "show",
        "3582b4ac:bayesfilter/inference/fixed_center_curvature.py"], text=True)
    spec = importlib.util.spec_from_loader("fixed_fitting_original_complete_reference", loader=None)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    exec(compile(source, spec.name, "exec"), original.__dict__)  # noqa: S102
    original.fit_factor_correlation_score_geometry = original_factor.fit_factor_correlation_score_geometry
    original.FactorCorrelationGeometryConfig = original_factor.FactorCorrelationGeometryConfig
    inputs = _inputs(3)
    before, after = (module.fit_fixed_center_curvature(*inputs, thresholds=_thresholds(module), factor_max=1).payload()
        for module in (original, candidate))
    # Compilation provenance is a new declared field, not an old numerical report.
    for fit in after["fits"]:
        fit["diagnostics"].pop("jit_compile", None)
    differences = []

    def inspect(left, right, path):
        if isinstance(right, dict):
            for key in right:
                inspect(left[key], right[key], path + "." + key)
        elif isinstance(right, (list, tuple)):
            for index, (x, y) in enumerate(zip(left, right, strict=True)):
                inspect(x, y, path + f"[{index}]")
        elif isinstance(right, (str, bool, int)) or right is None:
            if left != right:
                differences.append((path, left, right))
        elif not np.isclose(left, right, atol=1e-10, rtol=1e-10, equal_nan=True):
            differences.append((path, left, right))

    inspect(after, before, "result")
    print("ORIGINAL_FULL_RECORD_DIFFERENCES " + json.dumps(differences, allow_nan=True))
    _compare(after, before)


def test_complete_lifecycle_graph_and_xla_are_bounded():
    sizes = []
    for count in (2, 4):
        inputs = _inputs(replicates=count)
        arguments = tuple(tf.constant(value, D) for value in inputs[1:]) + (
            tf.constant([2., .1, .1, 90.], D), tf.ones([4], tf.bool), tf.constant(1),
            tf.constant([0., .5, 1.], D), tf.constant(1e-8, D), tf.constant(.1, D),
            tf.constant(True), tf.constant(.1, D))
        programs = [native.fit_program(candidate._dense_fit_kernel, native._structured_fit,
            candidate._precision_geometry_kernel, candidate._score_error_kernel,
            2, count, 6, 4, 4, 2, 1e8, .1, None, 3, jit_compile=jit) for jit in (False, True)]
        graph, compiled = (program(*arguments) for program in programs)
        for left, right in zip(tf.nest.flatten(compiled), tf.nest.flatten(graph), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
        assert compiled["status"] == 1
        assert "HloModule" in programs[1].experimental_get_compiler_ir(*arguments)(stage="hlo")
        counts = []
        for jit, program in zip((False, True), programs, strict=True):
            definition = program.get_concrete_function().graph.as_graph_def()
            nodes = list(definition.node) + [node for function in definition.library.function for node in function.node_def]
            assert any(node.op in ("While", "StatelessWhile") for node in nodes)
            assert not any(node.op in ("PyFunc", "EagerPyFunc") for node in nodes)
            if not jit:
                assert not any(function.attr["_XlaMustCompile"].b for function in definition.library.function
                    if "_XlaMustCompile" in function.attr)
            assert program.experimental_get_tracing_count() == 1
            counts.append(len(nodes))
        sizes.append(counts)
    assert sizes[0] == sizes[1]
