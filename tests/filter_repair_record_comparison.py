"""Owner-approved diagnostic comparison for finite iterative factor outputs.

The 1e-8 allowance applies only inside a factor-correlation result with the
repository schema. All other fields keep the 1e-10 campaign comparison.
"""

from collections.abc import Mapping

import numpy as np

FACTOR_SCHEMA = 'bayesfilter.factor_correlation_geometry.v1'
FACTOR_ARRAYS = frozenset(('covariance_z', 'precision_z', 'projected_precision_z',
                         'marginal_standard_deviations', 'loadings'))
FACTOR_DIAGNOSTICS = frozenset(('covariance_eigenvalues', 'condition_number',
    'prediction_jacobian_condition_number', 'loading_row_squared_norms',
    'train_score_rmse', 'holdout_score_rmse', 'holdout_score_relative_rmse'))


def compare_records(actual, expected, *, _path='result', _tolerance=1e-10, _factor_diagnostics=False):
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping), _path
        assert actual.keys() == expected.keys(), _path + ': fields changed'
        factor = expected.get('schema') == FACTOR_SCHEMA
        for key, value in expected.items():
            relaxed = (factor and key in FACTOR_ARRAYS) or (_factor_diagnostics and key in FACTOR_DIAGNOSTICS)
            compare_records(actual[key], value, _path=_path + '.' + key,
                _tolerance=1e-8 if relaxed else _tolerance,
                _factor_diagnostics=factor and key == 'diagnostics')
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)), _path
        assert len(actual) == len(expected), _path + ': shape changed'
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare_records(left, right, _path=_path + f'[{index}]', _tolerance=_tolerance)
    elif isinstance(expected, (str, bool, int)) or expected is None:
        assert type(actual) is type(expected), _path + ': discrete type changed'
        assert actual == expected, _path
    else:
        # Explicit shape checking prevents NumPy broadcasting from accepting a
        # changed matrix shape. Nonfinite semantics retain the original gate.
        assert np.shape(actual) == np.shape(expected), _path + ': shape changed'
        assert not isinstance(actual, (str, bool)) and actual is not None, _path
        np.testing.assert_allclose(actual, expected, atol=_tolerance, rtol=_tolerance,
                                   equal_nan=True, err_msg=_path)
