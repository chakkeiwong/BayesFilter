"""Approved iterative-factor equivalence with independent analytic geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_factor_sensitivity import (
    _analytic_geometry,
    _geometry_errors,
)
from tests.test_filter_repair_factor_trajectory import FIELDS
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import _compare_original


@pytest.mark.parametrize('data_name', ['original', 'current'])
def test_factor_complete_records_and_analytic_accuracy(data_name, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01826/lifecycle-factor-inputs.json'
    archived = json.loads(source.read_text())
    data = dict(archived['prepared'][data_name])
    data.update({key: tf.constant(data[key], tf.float64) for key in FIELDS})
    checkpoint = FrozenCheckpoint('3582b4ac', 'factor_equivalence_' + data_name)
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    expected = original._fit_factor_from_data(data, factor_count=2, config=cfg)
    actual = current._fit_factor_from_data({**data, '_native_factor_data': {
        **data, 'active_training_rows': tf.constant(10)}}, factor_count=2, config=cfg)
    truth = _analytic_geometry()
    report = {'role': 'approved_factor_equivalence_and_independent_geometry',
        'original': expected, 'current': actual, 'data': data_name,
        'strict_1e10_differences': _record_differences(actual, expected),
        'archive_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'frozen_source_sha256': checkpoint.hashes(),
        'analytic_errors': {name: _geometry_errors(row, data, truth)
                            for name, row in [('original', expected), ('current', actual)]}}
    with (directory / f'factor-equivalence-{data_name}.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    _compare_original(actual, expected)
    for row in (expected, actual):
        for field in ('covariance_z', 'precision_z', 'loadings'):
            np.testing.assert_allclose(row[field], truth[field], atol=1e-5, rtol=0., err_msg=field)
