"""Diagnostic attribution of the original-source changed-scale factor mismatch."""

import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import (
    _compare_original,
    _without_new_execution_metadata,
)
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def test_changed_scale_factor_crossed_inputs(request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01821/lifecycle-original-runtime-comparisons-5.json'
    archived = json.loads(source.read_text())['comparisons'][2]
    checkpoint = FrozenCheckpoint('3582b4ac', 'changed_scale_factor_inputs')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    scalar, batched, _ = _inputs(5, 4)
    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), 5) * .8
    score = scalar(center)[1]
    scale = tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), 5) * 1.1
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    # The preserved failing event does not recenter and does not reuse rows.
    assert archived['expected']['diagnostics']['history'][0]['recentered'] is False
    data = {}
    for name, module in [('original', original), ('current', current)]:
        prepared, count = module._structured_factor_fit_data(scalar, center, score, scale,
            search_theta=tf.zeros([4, 5], D), search_scores=tf.zeros([4, 5], D),
            dimension=5, radius=.25, fresh_sample_count=20, seed=(2026, 10715),
            evaluations=11, batched_value_and_score_fn=batched, reuse_search_scores=False)
        assert count == 31 and int(prepared['reused_training_count']) == 0
        data[name] = {key: value for key, value in prepared.items() if key != '_native_factor_data'}
    numerical_fields = ('center_score_z', 'training_offsets_z', 'training_scores_z',
        'holdout_offsets_z', 'holdout_scores_z', 'training_weights')
    records = {}
    for data_name, prepared in data.items():
        for fit_name, module in [('original', original), ('current', current)]:
            inputs = prepared if fit_name == 'original' else {**prepared,
                '_native_factor_data': {**prepared, 'active_training_rows': tf.constant(10)}}
            record = module._fit_factor_from_data(inputs, factor_count=2, config=cfg)
            records[data_name + '_data_' + fit_name + '_fit'] = _without_new_execution_metadata(record)
    report = {'role': 'explanatory_crossed_preparation_and_fitter_diagnostic',
        'baseline': checkpoint.revision, 'frozen_source_sha256': checkpoint.hashes(),
        'archived_comparison_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'prepared': {name: current._json_ready(prepared) for name, prepared in data.items()},
        'prepared_max_abs_error': {name: float(tf.reduce_max(tf.abs(data['current'][name] - data['original'][name])))
            for name in numerical_fields}, 'records': records,
        'differences_from_original_data_original_fit': {name: _record_differences(record,
            records['original_data_original_fit']) for name, record in records.items()},
        'nonclaims': ['Passing attribution checks does not waive the full-record numerical veto.',
            'No optimizer, seed, acceptance threshold or runtime arithmetic changed.']}
    with (directory / 'lifecycle-factor-inputs.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    _compare_original(records['original_data_original_fit'],
        archived['expected']['diagnostics']['history'][0]['fit'])
    _compare_original(records['current_data_current_fit'],
        archived['actual']['diagnostics']['history'][0]['fit'])
