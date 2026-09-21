"""Mutation checks for the owner-approved huge-fixture diagnostic comparison."""

from copy import deepcopy

import numpy as np
import pytest

from tests.filter_repair_dense_extreme_comparison import compare_huge_dense_records


def fixture(dimension=3):
    # Replay the declared boundary fixture's design, including its frame draw.
    rng = np.random.default_rng(20260921 + dimension)
    rng.normal(size=(dimension, dimension))
    x = rng.normal(size=(2, 32, dimension))[0] * .1
    precision = np.diag(np.linspace(.7, 3., dimension)) * 1e150
    eigenvalues = np.diag(precision).copy()
    singular_values = np.linalg.svd(x, compute_uv=False)
    record = {"raw_precision": precision, "raw_eigenvalues": eigenvalues,
        "design_condition": np.float64(singular_values[0] / singular_values[-1]),
        "design_rank": np.int32(dimension), "minimum_eigenvalue": eigenvalues[0],
        "maximum_eigenvalue": eigenvalues[-1], "precision_condition": np.float64(eigenvalues[-1] / eigenvalues[0]),
        "raw_spd": np.bool_(True), "selection_relative_rmse": np.float64(0.)}
    return record, {"dimension": dimension, "case": "huge", "offsets": x, "response": x @ precision}


@pytest.mark.parametrize("dimension", [3, 5])
def test_small_matrix_relative_roundoff_passes_but_larger_error_fails(dimension):
    expected, args = fixture(dimension)
    actual = deepcopy(expected)
    actual["raw_precision"][0, 1] += 1e137
    report = compare_huge_dense_records(actual, expected, **args)
    assert 0. < report["raw_precision_matrix_relative_error"] < 1e-12
    actual["raw_precision"][0, 1] += 1e140
    with pytest.raises(AssertionError):
        compare_huge_dense_records(actual, expected, **args)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "precision_shape", "eigenvalue_shape",
    "scalar_shape", "rank", "decision", "boolean_type", "dtype", "nonfinite", "reference", "case", "dimension", "response"])
def test_malformed_or_unapproved_comparisons_fail(mutation):
    expected, args = fixture()
    actual = deepcopy(expected)
    if mutation == "missing":
        del actual["raw_precision"]
    elif mutation == "unexpected":
        actual["unchecked"] = np.float64(0.)
    elif mutation == "precision_shape":
        actual["raw_precision"] = actual["raw_precision"][:1]
    elif mutation == "eigenvalue_shape":
        actual["raw_eigenvalues"] = actual["raw_eigenvalues"][:1]
    elif mutation == "scalar_shape":
        actual["selection_relative_rmse"] = np.array([0.])
    elif mutation == "rank":
        actual["design_rank"] = np.int32(2)
    elif mutation == "decision":
        actual["raw_spd"] = np.bool_(False)
    elif mutation == "boolean_type":
        actual["raw_spd"] = np.int32(1)
    elif mutation == "dtype":
        actual["design_rank"] = np.int64(3)
    elif mutation == "nonfinite":
        actual["raw_precision"][0, 1] = np.nan
    elif mutation == "reference":
        # Agreement between two equally corrupted solvers cannot pass.
        expected["raw_precision"][0, 0] += 1e141
        actual["raw_precision"][0, 0] += 1e141
    elif mutation == "case":
        args["case"] = "regular"
    elif mutation == "dimension":
        args["dimension"] = 4
    else:
        args["response"] = args["response"] * 1.1
    with pytest.raises(AssertionError):
        compare_huge_dense_records(actual, expected, **args)


@pytest.mark.parametrize("key", ["raw_eigenvalues", "design_condition", "minimum_eigenvalue",
    "maximum_eigenvalue", "precision_condition", "selection_relative_rmse"])
def test_every_other_float_field_keeps_strict_tolerance(key):
    expected, args = fixture()
    actual = deepcopy(expected)
    actual[key] = actual[key] * (1. + 1e-8) + 1e-8
    with pytest.raises(AssertionError):
        compare_huge_dense_records(actual, expected, **args)
