"""Independent roundoff sensitivity of the frozen finite-iteration factor fit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_factor_trajectory import FIELDS
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import _without_new_execution_metadata

D = tf.float64


def _analytic_geometry():
    """Independent fixture algebra, without either fitted covariance helper."""
    deviations = tf.constant([.8, 1.1, .9, 1.2, .7], D)
    loadings = tf.constant([[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]], D)
    raw_covariance = deviations[:, None] * (tf.linalg.diag(1. - tf.reduce_sum(loadings ** 2, axis=1))
        + loadings @ tf.transpose(loadings)) * deviations[None, :]
    scale = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), 5) * 1.1
    covariance = raw_covariance / scale[:, None] / scale[None, :]
    return {'covariance_z': covariance, 'precision_z': tf.linalg.inv(covariance),
            'marginal_standard_deviations': deviations / scale, 'loadings': loadings}


def _geometry_errors(record, data, truth):
    result = {}
    for key, expected in truth.items():
        observed = tf.constant(record[key], D)
        result[key] = {'max_abs': float(tf.reduce_max(tf.abs(observed - expected))),
            'relative_l2': float(tf.linalg.norm(observed - expected) / tf.linalg.norm(expected))}
    for partition in ('training', 'holdout'):
        response = data['center_score_z'][None, :] - data[partition + '_scores_z']
        prediction = data[partition + '_offsets_z'] @ tf.transpose(tf.constant(record['precision_z'], D))
        residual = prediction - response
        result[partition + '_prediction'] = {
            'max_abs': float(tf.reduce_max(tf.abs(residual))),
            'relative_l2': float(tf.linalg.norm(residual) / tf.linalg.norm(response))}
    return result


def test_original_one_ulp_input_sensitivity(request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01826/lifecycle-factor-inputs.json'
    archive = json.loads(source.read_text())
    checkpoint = FrozenCheckpoint('3582b4ac', 'factor_sensitivity')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    prepared = dict(archive['prepared']['original'])
    prepared.update({key: tf.constant(prepared[key], D) for key in FIELDS})
    truth = _analytic_geometry()
    report = {'role': 'diagnostic_original_one_ulp_input_sensitivity',
        'archived_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'gpu_execution': bool(tf.config.list_logical_devices('GPU')),
        'truth': current._json_ready(truth), 'arms': {},
        'nonclaims': ['Roundoff sensitivity does not waive the current 1e-10 full-record gate.',
            'Analytic geometry is the exact target, not the finite stopped optimizer program.',
            'No optimizer, method, threshold, seed or runtime input changes.']}
    reference_response_error = {}
    for partition in ('training', 'holdout'):
        response = prepared['center_score_z'][None, :] - prepared[partition + '_scores_z']
        predicted = prepared[partition + '_offsets_z'] @ tf.transpose(truth['precision_z'])
        error = float(tf.reduce_max(tf.abs(response - predicted)))
        reference_response_error[partition] = error
        assert error < 1e-14
    report['analytic_response_max_abs_error'] = reference_response_error
    baseline = original._fit_factor_from_data(prepared, factor_count=2, config=cfg)
    if not report['gpu_execution']:
        assert not _record_differences(baseline, archive['records']['original_data_original_fit'])
    configurations = [('unmodified_original', original, prepared)]
    for keys in (('training_offsets_z',), ('training_scores_z',),
                 ('training_offsets_z', 'training_scores_z')):
        for direction in (-1, 1):
            changed = dict(prepared)
            for key in keys:
                # Independent diagnostic input perturbation, never runtime.
                changed[key] = tf.constant(np.nextafter(prepared[key].numpy(),
                    np.inf if direction > 0 else -np.inf), D)
            configurations.append(('_'.join(keys) + ('_up' if direction > 0 else '_down'), original, changed))
    configurations.append(('unmodified_current', current, prepared))
    for label, module, data in configurations:
        print('FACTOR_SENSITIVITY ' + label, flush=True)
        inputs = data if module is original else {**data,
            '_native_factor_data': {**data, 'active_training_rows': tf.constant(10)}}
        record = baseline if label == 'unmodified_original' else module._fit_factor_from_data(
            inputs, factor_count=2, config=cfg)
        record = _without_new_execution_metadata(record)
        report['arms'][label] = {'record': record,
            'input_sha256': {key: hashlib.sha256(tf.io.serialize_tensor(data[key]).numpy()).hexdigest()
                             for key in FIELDS},
            'input_max_abs_delta': {key: float(tf.reduce_max(tf.abs(data[key] - prepared[key]))) for key in FIELDS},
            'differences_from_original': _record_differences(record, baseline),
            'errors_from_analytic_geometry': _geometry_errors(record, data, truth)}
    report['frozen_source_sha256'] = checkpoint.hashes()
    with (directory / 'factor-input-sensitivity.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
