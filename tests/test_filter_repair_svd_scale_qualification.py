"""Verify the original scale-attribution fixture after the factorization repair."""

import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_svd_scale_audit import (
    test_actual_svd_endpoints_at_equal_condition_and_different_scale as run_probe,
)


def test_actual_svd_endpoint_numerical_qualification(request):
    run_probe(request)
    path = Path(request.config.getoption('xmlpath')).parent / 'actual-svd-scale-attribution.json'
    report = json.loads(path.read_text())
    for row in report['records']:
        np.testing.assert_allclose(row['telemetry'][0], [row['reference_telemetry'][0]], rtol=1e-10, atol=0.)
        np.testing.assert_allclose(row['telemetry'][1], [row['reference_telemetry'][1]], rtol=1e-10, atol=0.)
        assert row['factor_rank'] == row['conditional_rank'] == [3]
        assert row['on_support'] == [True]
        assert row['factor_relative_covariance_error'] < 1e-10
        assert row['gain_relative_error'] < 1e-10
        assert row['conditional_covariance_absolute_error'] < 1e-10
        np.testing.assert_allclose(row['conditional_value'], [row['reference_conditional_value']], rtol=1e-10, atol=1e-10)
