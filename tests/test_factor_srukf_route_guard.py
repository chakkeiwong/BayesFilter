from __future__ import annotations

from pathlib import Path

import pytest

from bayesfilter.nonlinear.srukf_route_guard import (
    ADMITTED_DIRECT_FACTOR_SRUKF_FILES,
    assert_no_forbidden_srukf_routes,
    find_forbidden_srukf_routes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_closed_admitted_file_list_passes() -> None:
    paths = [ROOT / path for path in ADMITTED_DIRECT_FACTOR_SRUKF_FILES]
    assert assert_no_forbidden_srukf_routes(paths) == ()


@pytest.mark.parametrize("snippet", ["TF.LINALG.SVD(x)", "tf.linalg.ChOlEsKy(x)", "Principal_Sqrt(x)", "covariance_to_factor(x)"])
def test_route_guard_is_case_insensitive(snippet: str) -> None:
    assert find_forbidden_srukf_routes(snippet)
