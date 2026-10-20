"""Pinned fixed-center family records, optional gates and native pair loops."""

import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as candidate
from bayesfilter.inference import fixed_center_stability_tf as native

D = tf.float64


@pytest.fixture(scope="module")
def baseline():
    source = subprocess.check_output(["git", "show",
        "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf:bayesfilter/inference/fixed_center_curvature.py"], text=True)
    spec = importlib.util.spec_from_loader("fixed_stability_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - pinned diagnostic oracle
    assert module._family_stability.__globals__["compare_precision_geometry"].__module__ == spec.name
    return module


def _compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _compare(left, right)
    elif isinstance(expected, (str, bool, int)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


def _matrices(count=3, dimension=3):
    left = np.diag(np.linspace(.8, 3.3, dimension)) + .13
    matrices = []
    for index in range(count):
        rotation = np.eye(dimension)
        angle = .13 * index
        rotation[-2:, -2:] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        matrices.append(rotation @ left @ rotation.T * (1. + .04 * index))
    return matrices


def _fits(matrices, usable=None):
    flags = [True] * len(matrices) if usable is None else usable
    return [SimpleNamespace(precision_z=matrix, replicate_index=10 + 3 * index,
        accepted=False, diagnostics={"geometry_admissible": flag})
        for index, (matrix, flag) in enumerate(zip(matrices, flags, strict=True))]


def _thresholds(module, **options):
    return module.FixedCenterCurvatureThresholds(selection_holdout_relative_rmse_cap=.2,
        audit_relative_rmse_cap=.2, projection_relative_frobenius_cap=.2, **options)


@pytest.mark.parametrize("rank", [None, 1, 2])
@pytest.mark.parametrize("caps", [{}, {"principal_angle_degrees_cap": 1.},
    {"generalized_eigenvalue_spread_cap": 1.1}, {"trace_normalized_frobenius_cap": .01},
    {"trace_normalized_operator_cap": .01}, {"generalized_eigenvalue_spread_cap": 100.,
        "trace_normalized_frobenius_cap": 1., "trace_normalized_operator_cap": 1., "principal_angle_degrees_cap": 90.}])
def test_complete_family_records_preserve_optional_caps_and_pair_order(baseline, rank, caps):
    fits = _fits(_matrices())
    if (rank is None) != ("principal_angle_degrees_cap" not in caps):
        # The original configuration couples these two fields. The first run
        # incorrectly tried to use their invalid Cartesian-product combinations.
        for module in (baseline, candidate):
            with pytest.raises(ValueError, match="must be set together"):
                _thresholds(module, principal_subspace_rank=rank, **caps)
        return
    before, after = (module._family_stability(fits, _thresholds(module, principal_subspace_rank=rank, **caps))
        for module in (baseline, candidate))
    _compare(after, before)
    assert [(pair["left_replicate"], pair["right_replicate"]) for pair in after["comparisons"]] == [
        (10, 13), (10, 16), (13, 16)]
    if not caps:
        assert after["passed"]
        assert all(value is None for pair in after["comparisons"] for value in pair["checks"].values())


@pytest.mark.parametrize("count,missing,unusable", [(0, (), ()), (1, (), ()), (2, (0, 1), ()),
    (3, (1,), ()), (3, (), (1,)), (3, (1,), (2,))])
def test_incomplete_families_preserve_all_or_nothing_gate(baseline, count, missing, unusable):
    matrices = _matrices(count)
    for index in missing:
        matrices[index] = None
    fits = _fits(matrices, [index not in unusable for index in range(count)])
    before, after = (module._family_stability(fits, _thresholds(module)) for module in (baseline, candidate))
    assert not after["passed"] and after["comparisons"] == []
    _compare(after, before)


@pytest.mark.parametrize("matrix", [np.diag([-1., 0., 2.]), np.zeros((3, 3)), np.diag([-3., -2., -1.])])
@pytest.mark.parametrize("gated", [False, True])
def test_non_spd_comparisons_preserve_missing_metrics_and_optional_veto(baseline, matrix, gated):
    fits = _fits([matrix, np.diag([.7, 1.3, 3.])])
    options = {"generalized_eigenvalue_spread_cap": 2.} if gated else {}
    before, after = (module._family_stability(fits, _thresholds(module, **options)) for module in (baseline, candidate))
    _compare(after, before)
    assert after["passed"] is not gated
    assert after["comparisons"][0]["metrics"]["generalized_eigenvalues"] is None


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("fault", ["nan", "asymmetric"])
def test_invalid_matrix_errors_preserve_pair_order(baseline, index, fault):
    matrices = _matrices()
    matrices[index][0, 1] = np.nan if fault == "nan" else 3.
    fits = _fits(matrices)
    message = ("left" if index == 0 else "right") + " must be a finite symmetric square matrix"
    for module in (baseline, candidate):
        with pytest.raises(ValueError, match=message):
            module._family_stability(fits, _thresholds(module))


def test_unusable_family_does_not_validate_its_numerical_matrices(baseline):
    matrices = _matrices()
    matrices[0][0, 0] = np.nan
    fits = _fits(matrices, [True, False, True])
    before, after = (module._family_stability(fits, _thresholds(module)) for module in (baseline, candidate))
    _compare(after, before)


def test_rank_error_is_retained_only_for_complete_families(baseline):
    fits = _fits(_matrices())
    for module in (baseline, candidate):
        with pytest.raises(ValueError, match="subspace_rank must lie"):
            module._family_stability(fits, _thresholds(module, principal_subspace_rank=4, principal_angle_degrees_cap=90.))
        assert not module._family_stability(fits[:1], _thresholds(module,
            principal_subspace_rank=4, principal_angle_degrees_cap=90.))["passed"]


def test_complete_graph_and_xla_pair_loops_have_bounded_graph_size():
    sizes = []
    for count in (2, 5):
        arguments = (tf.constant(_matrices(count), D), tf.ones([count, 2], tf.bool),
            tf.constant([100., 1., 1., 90.], D), tf.ones([4], tf.bool), tf.constant(2))
        programs = [native.stability_program(candidate._precision_geometry_kernel, 3, count,
            jit_compile=jit) for jit in (False, True)]
        graph, compiled = (program(*arguments) for program in programs)
        for name in graph:
            np.testing.assert_allclose(compiled[name], graph[name], atol=1e-10, rtol=1e-10)
        assert compiled["passed"]
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
