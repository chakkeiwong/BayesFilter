"""Explanatory attribution of the frozen D3 condition-diagnostic discrepancy."""

import hashlib
import json
from functools import partial
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import fixed_center_curvature as fixed
from tests.test_filter_repair_dense_validated_fit import _thresholds, normalized
from tests.test_filter_repair_fixed_fitting_localization import _baseline
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def test_dense_attempt_condition_identical_data_and_states(monkeypatch, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    path = directory.parent / 'run-03450/dense-attempt-3-centered.json'
    archive = json.loads(path.read_text())
    captured = archive['records'][0]
    partitions = captured['cloud']['original'][4]
    assert partitions == captured['cloud']['actual'][4]
    precision = tf.linalg.diag(tf.cast(tf.range(3), D) + 1.3) + .07
    center = tf.zeros([3], D)
    scale = .8 + tf.cast(tf.range(3), D) * .2
    score = -tf.linalg.matvec(precision, center - tf.fill([3], tf.constant(.001, D))) * scale
    train, select, audit = partitions[:2], partitions[2:4], partitions[4]
    inputs = (center, score, tf.constant([row[0] for row in train], D),
        tf.constant([row[1] for row in train], D), tf.constant([row[0] for row in select], D),
        tf.constant([row[1] for row in select], D), tf.constant(audit[0], D), tf.constant(audit[1], D))
    standalone = normalized({'result': fixed.fit_fixed_center_curvature(*inputs,
        thresholds=_thresholds(fixed), factor_max=2, lineage={'role': 'dense_attempt_composition'}).payload()})
    original = _baseline()
    cfg = factor.FactorCorrelationGeometryConfig(holdout_score_relative_rmse=.1)
    old_cfg = original.FactorCorrelationGeometryConfig(holdout_score_relative_rmse=.1)
    observations = []
    for replicate in range(2):
        args = (score, inputs[2][replicate], inputs[3][replicate], inputs[4][replicate], inputs[5][replicate])
        program = factor._make_factor_program(3, 9, 6, cfg, True, factor._prediction_jacobian_diagnostics)
        computed = program(*args, tf.fill([9], tf.constant(1. / 9, D)))
        # Capture the unmodified original optimizer's completed return value.
        states = []
        minimize = original.tfp.optimizer.lbfgs_minimize

        def recorded_minimize(*args, minimize=minimize, states=states, **kwargs):
            result = minimize(*args, **kwargs)
            states.append(result)
            return result

        with monkeypatch.context() as patch:
            patch.setattr(original.tfp.optimizer, 'lbfgs_minimize', recorded_minimize)
            old_result = original.fit_factor_correlation_score_geometry(*args, config=old_cfg)
        assert len(states) == 1
        anchors = old_result.anchor_indices
        current_anchors = tuple(int(value) for value in computed['anchors'])
        current_result = factor._factor_result_from_computed(computed, cfg, 3, 9, 6, True).payload()
        measurements = {}
        for label, state, state_anchors in [('original', states[0].position, anchors),
                ('current', computed['optimizer'].position, current_anchors)]:
            for algorithm, module, jit in [('original', original, False), ('current_graph', factor, False),
                    ('current_xla', factor, True)]:
                options = {'dimension': 3, 'anchors': state_anchors,
                    'config': old_cfg if algorithm == 'original' else cfg}
                if algorithm != 'original':
                    options['jit_compile'] = jit
                body = partial(module._prediction_jacobian_diagnostics, **options)
                check = (body if algorithm == 'original' else tf.function(body,
                    input_signature=[tf.TensorSpec([6], D), tf.TensorSpec([9, 3], D)],
                    jit_compile=jit, autograph=False))
                rank, condition = check(state, args[1])
                measurements[label + '_' + algorithm] = {'rank': int(rank), 'condition': float(condition)}
        observations.append({'replicate': replicate, 'original': clean(old_result.payload()),
            'current': clean(current_result), 'original_state': clean(states[0].position),
            'current_state': clean(computed['optimizer'].position), 'diagnostics': measurements})
    report = {'role': 'explanatory_only_identical_data_and_state_attribution',
        'source_artifact': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'standalone': standalone,
        'standalone_vs_composition': _record_differences(standalone, captured['fit']['actual']),
        'standalone_vs_original': _record_differences(standalone, captured['fit']['original']),
        'observations': observations,
        'nonclaims': ['No tolerance, rank, scientific threshold or fitted-result change.',
            'The failed complete-record gate is not waived by this diagnostic.']}
    save(request, 'dense-attempt-condition-attribution.json', report)
    _equal_records(standalone, captured['fit']['actual'])
