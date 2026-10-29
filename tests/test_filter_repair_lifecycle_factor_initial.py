"""Diagnostic frozen-initial-state intervention; never a runtime replacement."""

import hashlib
import inspect
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_structured_fit_tf import _cached_structured_fit
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import (
    _compare_original,
    _without_new_execution_metadata,
)

D = tf.float64


def _objective_program(anchors, factor_cfg, old_loss):
    @tf.function(input_signature=[tf.TensorSpec([14], D)], jit_compile=True, autograph=False)
    def objective(raw):
        with tf.GradientTape() as tape:
            tape.watch(raw)
            covariance, _, _ = factor._decode_covariance(raw, dimension=5,
                anchors=anchors, config=factor_cfg)
            precision = tf.linalg.cholesky_solve(tf.linalg.cholesky(covariance), tf.eye(5, dtype=D))
            prediction = tf.einsum('ij,bj->bi', precision, old_loss['train_z'])
            per_row = tf.reduce_mean(tf.square(prediction - old_loss['train_response']), axis=1)
            value = tf.reduce_sum(old_loss['weights'] * per_row)
        return value, tape.gradient(value, raw)

    return objective


def test_changed_scale_original_initial_state_intervention(monkeypatch, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01826/lifecycle-factor-inputs.json'
    archived = json.loads(source.read_text())
    checkpoint = FrozenCheckpoint('3582b4ac', 'changed_scale_factor_initial')
    original = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    factor_cfg = factor.FactorCorrelationGeometryConfig(factor_count=2,
        max_condition_number=cfg.max_condition_number,
        holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
    numerical_fields = ('center_score_z', 'training_offsets_z', 'training_scores_z',
        'holdout_offsets_z', 'holdout_scores_z', 'training_weights')
    minimize = factor.tfp.optimizer.lbfgs_minimize
    captured = []

    def capture(objective, *, initial_position, **kwargs):
        result = minimize(objective, initial_position=initial_position, **kwargs)
        captured.append((tf.identity(initial_position), result, objective))
        return result

    report = {'role': 'explanatory_frozen_initial_state_intervention',
        'archived_inputs_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'rows': [], 'nonclaims': ['Injected original states are diagnostic only.',
            'Instrumentation may affect compiler arithmetic; unmodified full-record gates remain mandatory.']}
    for data_name in ('original', 'current'):
        prepared = dict(archived['prepared'][data_name])
        prepared.update({key: tf.constant(prepared[key], D) for key in numerical_fields})
        with monkeypatch.context() as context:
            context.setattr(factor.tfp.optimizer, 'lbfgs_minimize', capture)
            expected = original._fit_factor_from_data(prepared, factor_count=2, config=cfg)
        initial, optimizer, objective = captured[-1]
        _compare_original(expected, archived['records'][data_name + '_data_original_fit'])
        original_arithmetic = [objective(point) for point in (initial, optimizer.position)]
        closure = inspect.getclosurevars(objective).nonlocals
        old_loss = inspect.getclosurevars(closure['loss']).nonlocals
        anchors = tuple(expected['anchor_indices'])

        current_objective = _objective_program(anchors, factor_cfg, old_loss)
        same_state = []
        for point, before in zip((initial, optimizer.position), original_arithmetic, strict=True):
            after = current_objective(point)
            same_state.append({'state': point.numpy().tolist(),
                'before': current._json_ready(before), 'after': current._json_ready(after),
                'max_value_error': float(tf.abs(after[0] - before[0])),
                'max_gradient_error': float(tf.reduce_max(tf.abs(after[1] - before[1])))})
        factor._make_factor_program.cache_clear()
        _cached_structured_fit.cache_clear()
        try:
            with monkeypatch.context() as context:
                context.setattr(factor, '_encode_state', lambda *_, initial=initial: initial)
                injected = current._fit_factor_from_data({**prepared,
                    '_native_factor_data': {**prepared, 'active_training_rows': tf.constant(10)}},
                    factor_count=2, config=cfg)
        finally:
            factor._make_factor_program.cache_clear()
            _cached_structured_fit.cache_clear()
        injected = _without_new_execution_metadata(injected)
        report['rows'].append({'data': data_name, 'original_initial': initial.numpy().tolist(),
            'same_state_arithmetic': same_state, 'injected_record': injected,
            'differences_from_original': _record_differences(injected, expected),
            'unmodified_current_differences': _record_differences(
                archived['records'][data_name + '_data_current_fit'], expected)})
    report['frozen_source_sha256'] = checkpoint.hashes()
    with (directory / 'lifecycle-factor-initial.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
