"""Pinned complete fixed-center selector records and compiled numerical control."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as candidate
from bayesfilter.inference import fixed_center_selection_tf as native
from tests.test_filter_repair_fixed_stability import _compare, _thresholds

pytest_plugins = ["tests.test_filter_repair_fixed_stability"]

D = tf.float64


def _fit(family, index, matrix, *, accepted=True, admissible=True):
    return SimpleNamespace(family=family, replicate_index=7 + 3 * index,
        precision_z=matrix, accepted=accepted, diagnostics={"geometry_admissible": admissible})


def _case(kind):
    dense = np.array([[2., .4, .1], [.4, 3., .2], [.1, .2, 4.]])
    first, second = np.diag([1., 2., 3.]), np.diag([1.5, 2.5, 3.5])
    fits = [_fit(family, i, matrix, accepted=family != "dense")
        for family, matrix in (("dense", dense), ("factor_1", first), ("factor_2", second)) for i in range(3)]
    target, expected_precision = None, first
    if kind == "second":
        fits[3].accepted = fits[4].accepted = False
        expected_precision = second
    elif kind == "subset":
        fits[4].accepted = False
    elif kind.startswith("explicit"):
        target = "factor_2" if kind == "explicit_second" else "factor_1"
        expected_precision = .5 * (dense + (second if target == "factor_2" else first))
        if kind == "explicit_missing":
            fits[3].precision_z = None
        if kind == "explicit_rejected":
            for fit in fits[3:6]:
                fit.accepted = False
    elif kind in ("diagonal", "partial", "dense_missing", "no_dense", "holdout_reject", "non_spd"):
        fits = fits[:3]
        expected_precision = .5 * (dense + np.diag(np.diag(dense)))
        if kind == "partial":
            fits[1].diagnostics["geometry_admissible"] = False
        if kind == "dense_missing":
            fits[1].precision_z = None
        if kind == "no_dense":
            fits = [_fit("factor_1", i, first, accepted=False) for i in range(3)]
        if kind == "holdout_reject":
            expected_precision = np.eye(3) * 100.
        if kind == "non_spd":
            for fit in fits:
                fit.precision_z = -dense
    elif kind == "empty":
        fits = []
    center = np.array([.1, -.2, .5])
    offsets = (np.arange(12, dtype=float).reshape(4, 3) / 11., np.eye(3))
    partitions = tuple((rows, center - rows @ expected_precision.T) for rows in offsets)
    return fits, center, partitions, target


@pytest.mark.parametrize("kind", ["first", "second", "subset", "explicit_first", "explicit_second",
    "explicit_missing", "explicit_rejected", "diagonal", "partial", "dense_missing", "no_dense",
    "holdout_reject", "empty", "non_spd"])
@pytest.mark.parametrize("capped", [False, True])
def test_complete_selector_preserves_pinned_records(baseline, kind, capped):
    fits, center, partitions, target = _case(kind)
    caps = {"generalized_eigenvalue_spread_cap": 5., "trace_normalized_frobenius_cap": 1.,
        "trace_normalized_operator_cap": 1., "principal_angle_degrees_cap": 90., "principal_subspace_rank": 2} if capped else {}
    if kind == "non_spd" and not capped:
        for module in (baseline, candidate):
            with pytest.raises(ValueError, match="requires SPD"):
                module._select_candidate(fits, center, partitions, thresholds=_thresholds(module, **caps),
                    shrinkage_weights=(0., .5, 1.), structured_target_family=target)
        return
    before, after = (module._select_candidate(fits, center, partitions, thresholds=_thresholds(module, **caps),
        shrinkage_weights=(0., .5, 1.), structured_target_family=target) for module in (baseline, candidate))
    _compare(after[1], before[1])
    if before[0] is None:
        assert after[0] is None
    else:
        assert after[0]["family"] == before[0]["family"]
        np.testing.assert_allclose(after[0]["precision_z"], before[0]["precision_z"], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("fault", ["left", "right", "rank"])
def test_stability_errors_precede_early_family_selection(baseline, fault):
    fits, center, partitions, _ = _case("first")
    options = {}
    message = "left must" if fault == "left" else "right must"
    if fault == "rank":
        options = {"principal_subspace_rank": 4, "principal_angle_degrees_cap": 90.}
        message = "subspace_rank must"
    else:
        fits[0 if fault == "left" else 1].precision_z = np.full([3, 3], np.nan)
    for module in (baseline, candidate):
        with pytest.raises(ValueError, match=message):
            module._select_candidate(fits, center, partitions, thresholds=_thresholds(module, **options),
                shrinkage_weights=(0., 1.), structured_target_family=None)


@pytest.mark.parametrize("rows", [(3,), (4, 3, 7), ()])
def test_mean_selection_error_uses_equal_partition_weights(baseline, rows):
    precision, center = np.diag([1., 2., 3.]), np.array([.2, -.1, .3])
    partitions = tuple((np.arange(count * 3).reshape(count, 3) / 7.,
        np.ones([count, 3]) * (i + 1.)) for i, count in enumerate(rows))
    before = baseline._mean_selection_error(precision, center, partitions)
    after = candidate._mean_selection_error(precision, center, partitions)
    np.testing.assert_allclose(after, before, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_shrinkage_ties_use_lowest_weight_and_keep_candidate_order(baseline):
    precision = np.diag([1., 2., 4.])
    fits = [_fit("dense", i, precision) for i in range(3)]
    before, after = (module._select_candidate(fits, np.zeros(3), ((np.eye(3), -precision),),
        thresholds=_thresholds(module), shrinkage_weights=(1., .5, 0., .5), structured_target_family=None)
        for module in (baseline, candidate))
    _compare(after[1], before[1])
    assert after[1]["selected_weight"] == 0.
    assert [row["selected"] for row in after[1]["candidates"]] == [False, False, True, False]


@pytest.mark.parametrize("dimension", [1, 2, 4])
@pytest.mark.parametrize("cap", [0.5, np.nextafter(.5, 0.), np.nextafter(.5, 1.)])
def test_unequal_family_counts_and_holdout_cap_boundary(baseline, dimension, cap):
    precision = 3. * np.eye(dimension)
    fits = [_fit("dense", i, precision) for i in range(4)]
    fits += [_fit("factor_1", 0, precision, accepted=False)]
    fits += [_fit("factor_2", i, precision, accepted=False) for i in range(2)]
    before, after = (module._select_candidate(fits, np.zeros(dimension),
        ((np.eye(dimension), -2. * np.eye(dimension)),),
        thresholds=module.FixedCenterCurvatureThresholds(selection_holdout_relative_rmse_cap=cap,
            audit_relative_rmse_cap=.2, projection_relative_frobenius_cap=.2),
        shrinkage_weights=(0., .5, 1.), structured_target_family=None) for module in (baseline, candidate))
    _compare(after[1], before[1])
    assert (after[0] is None) == (before[0] is None)


def test_nan_error_order_and_asymmetric_input_normalization(baseline):
    precision = np.diag([1., 2., 4.])
    precision[0, 1] = 5e-13
    fits = [_fit("dense", i, precision) for i in range(2)]
    partitions = ((np.eye(3), np.full([3, 3], np.nan)),)
    before, after = (module._select_candidate(fits, np.zeros(3), partitions,
        thresholds=_thresholds(module), shrinkage_weights=(1., .5, 0.), structured_target_family=None)
        for module in (baseline, candidate))
    _compare(after[1], before[1])
    assert after[1]["selected_weight"] == 1.
    np.testing.assert_allclose(after[0]["precision_z"], before[0]["precision_z"], atol=1e-14, rtol=1e-14)


def test_complete_selection_graph_is_bounded_and_host_free():
    sizes = []
    for count, weights in ((2, [0., .5, 1.]), (5, [0., .25, .5, .75, 1.])):
        matrices = np.tile(np.array([[2., .3], [.3, 3.]])[None, None], [1, count, 1, 1])
        arguments = (tf.constant(matrices, D), tf.ones([1, count, 3], tf.bool), tf.zeros([2], D),
            tf.constant(np.tile(np.eye(2)[None], [count, 1, 1]), D),
            tf.constant(np.tile(-np.diag([2., 3.])[None], [count, 1, 1]), D),
            tf.zeros([4], D), tf.zeros([4], tf.bool), tf.constant(2),
            tf.constant(weights, D), tf.constant(.2, D))
        programs = [native.selection_program(candidate._precision_geometry_kernel, candidate._score_error_kernel,
            2, ("dense",), (count,), (2,) * count, len(weights), None, jit_compile=jit) for jit in (False, True)]
        graph, compiled = (program(*arguments) for program in programs)
        for left, right in zip(tf.nest.flatten(compiled), tf.nest.flatten(graph), strict=True):
            np.testing.assert_allclose(left, right, atol=1e-10, rtol=1e-10)
        assert compiled["weight"] == 1.
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
