"""Actual enclosing lifecycle versus the exact frozen public recurrence.

The common final mass/reporting suffix is outside the boundary under test.
Progress event contents/order are checked, not live delivery timing.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_lifecycle_tf as native
from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference.sequential_refinement_tf import refinement_program
from bayesfilter.inference.sequential_terminal_tf import terminal_program
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_sequential_refinement import _record
from tests.test_filter_repair_sequential_terminal import _payload
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def _excerpts(checkpoint):
    frozen = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    source = checkpoint.sources['bayesfilter/inference/sequential_map_covariance.py']
    start = source.index('    radius = cfg.initial_radius\n')
    end = source.index('\n\ndef _refinement_movement_payload', start)
    finish = source.index('    if max_score > cfg.terminal_score_max_abs or terminal_fit is None', start)
    initial = source[source.index('    refinement_origin = center if'):start]
    initial = initial[:initial.index('    _emit_progress(')]
    before = ('def execute(value_and_score_fn, batched_value_and_score_fn, center, center_value, '
        'center_score, scale_tf, cfg, evaluations, progress_callback):\n'
        '    dimension, locator_rows = int(center.shape[0]), []\n'
        + initial + source[start:end])
    after = ('def finish(center, max_score, terminal_fit, terminal_fit_attempts, evaluations, '
        'history, scale_tf, cfg, refinement_movement_initial, progress_callback):\n'
        '    locator_rows = []\n' + source[finish:end])
    namespace = dict(vars(frozen))
    exec(compile(before, 'frozen_complete_lifecycle_exact_excerpt', 'exec'), namespace)  # noqa: S102 - frozen diagnostic
    original = namespace['execute']
    namespace = dict(vars(current))
    exec(compile(after, 'unchanged_lifecycle_final_report_excerpt', 'exec'), namespace)  # noqa: S102 - report suffix only
    return original, namespace['finish']


def _materialize(result, center, value, score, scale, cfg, finish, progress):
    history = []
    dimension = int(center.shape[0])
    movement_initial = ({'position_z': tf.zeros([dimension], D), 'value': float(value),
        'score_norm': float(tf.linalg.norm(scale * score))}
        if cfg.record_refinement_movement_diagnostics else None)
    # Copy each completed column once. Array indexing transports already
    # computed records; it performs no host-side numerical recurrence.
    columns = tf.nest.map_structure(lambda values: values.numpy(), result['history'])
    for index in range(int(result['attempt_count'])):
        row = tf.nest.map_structure(lambda values, index=index: tf.convert_to_tensor(values[index]), columns)
        if bool(row['terminal_called']):
            fitted = _payload(row['terminal'], cfg)
            progress.append({'stage': 'terminal_fit_started', 'attempt': index,
                'exact_evaluations': int(row['evaluations_before']),
                'max_abs_scaled_score': float(tf.reduce_max(tf.abs(scale * row['score_before']))),
                'radius': float(row['radius_before'])})
            progress.append({'stage': 'terminal_fit_completed', 'attempt': index,
                'exact_evaluations': int(row['evaluations_before']) + int(row['terminal']['evaluations']),
                'fit_status': fitted['status'],
                'projection_relative_frobenius': fitted.get('projection_relative_frobenius')})
        event = int(row['event'])
        if event == 1:
            history.append({'attempt': index, 'action': 'terminal_fit_cloud_recentered',
                'center_value': float(row['terminal']['best_value']), 'fit': fitted})
        elif event == 2:
            history.append({'attempt': index, 'action': 'terminal_fit_rejected', **fitted})
        elif event == 3:
            refined = row['refine']
            record = _record(refined, cfg, dimension, float(row['radius_before']), index)
            action = int(refined['action'])
            if cfg.record_refinement_movement_diagnostics and action != 0:
                proposals = refined['attempts']
                evaluated = action != 1 and bool(proposals['last_evaluated'])
                last = proposals['last']
                record['refinement_movement'] = current._refinement_movement_payload(
                    refinement_origin=center, scale=scale,
                    pre_center=row['center_before'], pre_value=float(row['value_before']),
                    pre_score=row['score_before'], search_center=refined['selected_center'],
                    search_value=float(refined['selected_value']), search_score=refined['selected_score'],
                    evaluated_proposal=last['position'] if evaluated else None,
                    evaluated_proposal_value=float(last['value']) if evaluated else None,
                    evaluated_proposal_score=last['score'] if evaluated else None,
                    terminal_center=refined['center'], terminal_value=float(refined['center_value']),
                    terminal_score=refined['center_score'], radius_before=float(row['radius_before']),
                    radius_after=float(refined['radius_after']), search_recentered=bool(refined['search_recentered']),
                    proposal_accepted=action == 2)
            history.append(record)
            if action in (2, 3):
                progress.append({'stage': 'refinement_attempt_completed', 'attempt': index,
                    'exact_evaluations': int(row['evaluations_before']) + int(refined['evaluations']),
                    'action': record['action'],
                    'max_abs_scaled_score': float(tf.reduce_max(tf.abs(scale * refined['center_score']))),
                    'radius': float(refined['radius_after'])})
    status = int(result['status'])
    if status in (1, 2):
        extra = {'history': history}
        if movement_initial is not None:
            extra['refinement_movement_initial'] = movement_initial
        return current._rejected('maximum_exact_evaluations_before_terminal_fit' if status == 1 else
            'maximum_exact_evaluations', int(result['evaluations']), [], cfg,
            map_candidate=result['center'].numpy(), extra=extra)
    terminal = _payload(result['terminal'], cfg) if bool(result['has_terminal']) else None
    return finish(result['center'], float(result['max_score']), terminal, int(result['terminal_attempts']),
        int(result['evaluations']), history, scale, cfg, movement_initial, progress.append)


CASES = ('terminal', 'terminal_reject', 'symmetric', 'recenter', 'fit_reject',
    'factor_one', 'factor_two', 'factor_two_reuse')


@pytest.mark.parametrize('case', CASES)
def test_actual_lifecycle_complete_records_and_calls(case, request):
    checkpoint = FrozenCheckpoint('cfbc32d2', 'lifecycle_actual')
    original, finish = _excerpts(checkpoint)
    dimension = 5 if 'two' in case else 3
    cfg = current.SequentialMapCovarianceConfig(locator_policy='center_first', max_attempts=2,
        search_sample_count=4, regression_sample_count=24, terminal_sample_count=24,
        max_exact_evaluations=256, initial_radius=.25, terminal_score_max_abs=1e-5,
        refinement_geometry_policy='factor_correlation' if case.startswith('factor') else 'full_symmetric',
        structured_max_factors=2 if 'two' in case else 1, reuse_search_scores=case.endswith('reuse'),
        structured_holdout_score_relative_rmse=.001,
        score_holdout_relative_rmse=1e-12 if case.endswith('reject') else .35,
        record_refinement_movement_diagnostics=True)
    if case.startswith('factor'):
        raw_scalar, raw_batched, _ = _inputs(dimension, 4)
    else:
        def raw_batched(rows):
            return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

        def raw_scalar(row):
            values, scores = raw_batched(row[None, :])
            return values[0], scores[0]

    calls = tf.Variable(0, dtype=tf.int64, trainable=False)
    points = tf.Variable(tf.zeros([256, dimension], D), trainable=False)

    def observe(rows):
        count = tf.shape(rows, out_type=tf.int64)[0]
        indices = (tf.range(count) + calls.read_value())[:, None]
        points.scatter_nd_update(indices, rows)
        calls.assign_add(count)

    def scalar(row):
        observe(row[None, :])
        return raw_scalar(row)

    def batched(rows):
        observe(rows)
        return raw_batched(rows)

    center = (tf.zeros([dimension], D) if case.startswith('terminal') else
        tf.linspace(tf.constant(-.5, D), tf.constant(.8, D), dimension) if case == 'recenter' else
        tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension))
    value, score = raw_scalar(center)
    scale = tf.ones([dimension], D)
    before_progress, after_progress = [], []
    before = original(scalar, batched, center, float(value), score, scale, cfg, 1, before_progress.append).payload()
    before_calls, before_points = int(calls), points.numpy().tolist()
    calls.assign(0)
    points.assign(tf.zeros_like(points))
    refine = refinement_program(scalar, batched, dimension, cfg, cfg.search_sample_count)
    terminal = terminal_program(scalar, batched, dimension, cfg)
    program = native.lifecycle_program(refine, terminal, dimension, cfg, cfg.search_sample_count)
    arguments = (center, value, score, scale, tf.constant(1, tf.int64))
    result = program(*arguments)
    after = _materialize(result, center, value, score, scale, cfg, finish, after_progress).payload()
    directory = Path(request.config.getoption('xmlpath')).parent
    report = {'case': case, 'before': before, 'after': after, 'before_progress': before_progress,
        'after_progress': after_progress, 'before_calls': before_calls, 'after_calls': int(calls),
        'before_points': before_points[:before_calls], 'after_points': points.numpy().tolist()[:int(calls)],
        'checkpoint': checkpoint.revision, 'frozen_source_sha256': checkpoint.hashes(),
        'scope': 'complete actual lifecycle and endpoint records; final mass/reporting suffix shared',
        'nonclaims': ['Buffered progress does not preserve live timing or interruptibility.']}
    with (directory / f'lifecycle-actual-{case}.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    _compare(after, before)
    _compare(after_progress, before_progress)
    assert int(calls) == before_calls == int(result['evaluations']) - 1
    np.testing.assert_allclose(points[:before_calls], before_points[:before_calls], atol=1e-10, rtol=1e-10)
    assert program.experimental_get_tracing_count() == 1
    if 'two' in case:
        assert any(len(row.get('proposal_attempts', [])) == 2 and row['proposal_attempts'][1]['proposal_evaluated']
            for row in after['diagnostics']['history'])
