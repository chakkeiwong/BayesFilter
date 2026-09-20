"""Diagnostic native-row translation of the original two-factor decoder."""

import hashlib
import json
import math
from pathlib import Path
from types import FunctionType

import numpy as np
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_structured_fit_tf import _cached_structured_fit
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_lifecycle_original import _without_new_execution_metadata

D = tf.float64
_CURRENT_DECODE = factor._decode_covariance


def _row_decode(raw, *, dimension, anchors, config, domain_violations=None):
    if config.factor_count == 1:
        return _CURRENT_DECODE(raw, dimension=dimension, anchors=anchors,
            config=config, domain_violations=domain_violations)
    vector = tf.reshape(tf.convert_to_tensor(raw, D), [-1])
    deviations = config.standard_deviation_floor + tf.nn.softplus(vector[:dimension])
    bound = tf.sqrt(tf.constant(1. - config.loading_margin, D))
    remaining = vector[dimension:]
    anchor_a, anchor_b = anchors

    def row(index, rows):
        offset = 2 * index - tf.cast(index > anchor_a, tf.int32)

        def first_anchor():
            radius = bound * tf.math.sigmoid(tf.gather(remaining, offset))
            return tf.stack((radius, tf.constant(0., D)))

        def second_anchor():
            radius = bound * tf.math.sigmoid(tf.gather(remaining, offset))
            angle = tf.constant(math.pi, D) * tf.math.sigmoid(tf.gather(remaining, offset + 1))
            return radius * tf.stack((tf.cos(angle), tf.sin(angle)))

        def free_row():
            values = tf.gather(remaining, tf.stack((offset, offset + 1)))
            return bound * values / tf.sqrt(1. + tf.reduce_sum(tf.square(values)))

        loading = tf.cond(index == anchor_a, first_anchor,
            lambda: tf.cond(index == anchor_b, second_anchor, free_row))
        return index + 1, rows.write(index, loading)

    _, rows = tf.while_loop(lambda index, _: index < dimension, row,
        (tf.constant(0), tf.TensorArray(D, size=dimension, element_shape=[2])),
        maximum_iterations=dimension, parallel_iterations=1)
    loadings = rows.stack()
    if domain_violations is None:
        covariance = factor.factor_correlation_covariance(deviations, loadings,
            loading_margin=config.loading_margin)
    else:
        covariance, valid = factor._covariance_with_domain(deviations, loadings,
            tf.constant(config.loading_margin, D))
        update = domain_violations.assign_add(tf.cast(~valid, tf.int64))
        with tf.control_dependencies([update]):
            covariance = tf.identity(covariance)
    return covariance, deviations, loadings


def test_native_row_decoder_value_and_pullback(request):
    directory = Path(request.config.getoption('xmlpath')).parent
    saved = json.loads((directory.parent / 'run-01829/lifecycle-factor-initial.json').read_text())
    original = FrozenCheckpoint('3582b4ac', 'row_decoder').load(
        'bayesfilter.inference.factor_correlation_geometry')
    cfg = factor.FactorCorrelationGeometryConfig(factor_count=2)
    anchors = tuple(saved['rows'][0]['injected_record']['anchor_indices'])

    def evaluate(raw, decoder):
        with tf.GradientTape() as tape:
            tape.watch(raw)
            covariance, deviations, loadings = decoder(raw, dimension=5, anchors=anchors, config=cfg)
            objective = tf.reduce_sum(covariance ** 2) + tf.reduce_sum(deviations) + tf.reduce_sum(loadings ** 3)
        return covariance, deviations, loadings, tape.gradient(objective, raw)

    compiled = tf.function(lambda raw: evaluate(raw, _row_decode),
        input_signature=[tf.TensorSpec([14], D)], jit_compile=True, autograph=False)
    for row in saved['rows']:
        for state in row['same_state_arithmetic']:
            raw = tf.constant(state['state'], D)
            expected = evaluate(raw, original._decode_covariance)
            actual = compiled(raw)
            for before, after in zip(expected, actual, strict=True):
                np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)
    assert compiled.experimental_get_tracing_count() == 1


def test_original_row_arithmetic_in_native_loop(monkeypatch, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    source = directory.parent / 'run-01826/lifecycle-factor-inputs.json'
    archived = json.loads(source.read_text())
    cfg = current.SequentialMapCovarianceConfig(structured_holdout_score_relative_rmse=.001)
    numerical_fields = ('center_score_z', 'training_offsets_z', 'training_scores_z',
        'holdout_offsets_z', 'holdout_scores_z', 'training_weights')
    report = {'role': 'explanatory_native_row_decoder_trial',
        'archived_input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'rows': [],
        'decoder_scope': 'optimizer objective and final covariance only; existing vector decoder retained inside Jacobian diagnostic',
        'nonclaims': ['Diagnostic-only function injection; no runtime source replacement.',
            'Optimizer controls, data, random seeds and original full-record gates stay fixed.']}
    # ForwardAccumulator cannot trace the nested Cond in the row-loop trial.
    # Keep its already-qualified vector decoder for Jacobian reporting, so
    # this diagnostic isolates objective arithmetic without replacing that
    # derivative path. The independent test above checks identical values/VJP.
    original_diagnosis = factor._prediction_jacobian_diagnostics
    diagnosis = FunctionType(original_diagnosis.__code__,
        {**original_diagnosis.__globals__, '_decode_covariance': _CURRENT_DECODE},
        argdefs=original_diagnosis.__defaults__, closure=original_diagnosis.__closure__)
    diagnosis.__kwdefaults__ = original_diagnosis.__kwdefaults__
    for data_name in ('original', 'current'):
        prepared = dict(archived['prepared'][data_name])
        prepared.update({key: tf.constant(prepared[key], D) for key in numerical_fields})
        factor._make_factor_program.cache_clear()
        _cached_structured_fit.cache_clear()
        try:
            with monkeypatch.context() as context:
                context.setattr(factor, '_decode_covariance', _row_decode)
                context.setattr(factor, '_prediction_jacobian_diagnostics', diagnosis)
                result = current._fit_factor_from_data({**prepared,
                    '_native_factor_data': {**prepared, 'active_training_rows': tf.constant(10)}},
                    factor_count=2, config=cfg)
        finally:
            factor._make_factor_program.cache_clear()
            _cached_structured_fit.cache_clear()
        result = _without_new_execution_metadata(result)
        reference = archived['records'][data_name + '_data_original_fit']
        report['rows'].append({'data': data_name, 'record': result,
            'differences_from_original': _record_differences(result, reference),
            'differences_from_unmodified_current': _record_differences(result,
                archived['records'][data_name + '_data_current_fit'])})
        with (directory / ('factor-row-decode-' + data_name + '.json')).open('x') as handle:
            json.dump(report['rows'][-1], handle, indent=2)
            handle.write('\n')
    with (directory / 'factor-row-decode.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
