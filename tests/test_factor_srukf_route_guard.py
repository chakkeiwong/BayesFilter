from __future__ import annotations

from pathlib import Path
import inspect

import pytest

from bayesfilter.nonlinear.srukf_route_guard import (
    ADMITTED_DIRECT_FACTOR_SRUKF_FILES,
    assert_no_forbidden_srukf_routes,
    find_forbidden_srukf_routes,
)
from bayesfilter.linear import rectangular_factor_tf
from bayesfilter.nonlinear import rectangular_srukf_tf


ROOT = Path(__file__).resolve().parents[1]


def test_closed_admitted_file_list_passes() -> None:
    paths = [ROOT / path for path in ADMITTED_DIRECT_FACTOR_SRUKF_FILES]
    assert assert_no_forbidden_srukf_routes(paths) == ()


@pytest.mark.parametrize("snippet", ["TF.LINALG.SVD(x)", "tf.linalg.ChOlEsKy(x)", "Principal_Sqrt(x)", "covariance_to_factor(x)"])
def test_route_guard_is_case_insensitive(snippet: str) -> None:
    assert find_forbidden_srukf_routes(snippet)


def test_fixed_rectangular_score_primitive_closure_has_no_spectral_factorization() -> None:
    fixed_functions = (
        rectangular_factor_tf._batched_qr_with_derivative,
        rectangular_factor_tf._fixed_chart_decomposition,
        rectangular_factor_tf.batched_fixed_pivot_rectangular_qr,
        rectangular_factor_tf.batched_fixed_support_qr_likelihood,
        rectangular_factor_tf.batched_fixed_support_qr_conditional,
        rectangular_factor_tf.batched_fixed_support_qr_update,
        rectangular_srukf_tf.tf_rectangular_srukf_value_and_score,
    )
    source = "\n".join(inspect.getsource(function) for function in fixed_functions)
    assert "tf.linalg.svd" not in source.casefold()
    assert "tf.linalg.eigh" not in source.casefold()
    assert "covariance_to_factor" not in source.casefold()
