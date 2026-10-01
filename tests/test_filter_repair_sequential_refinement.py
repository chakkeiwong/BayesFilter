"""Complete search/fit/attempt records from an exact frozen public-step excerpt."""

import dataclasses
import json
import textwrap
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_refinement_tf as native
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_sequential_proposal import _target
from tests.test_filter_repair_sequential_terminal import _payload
from tests.test_filter_repair_structured_memory import _inputs

D = tf.float64


def _old_step(frozen, source):
    start = source.index('        search = _search_exact_candidates')
    stop = source.index('        _emit_progress(', start)
    block = textwrap.dedent(source[start:stop])
    wrapper = ('def execute(value_and_score_fn, batched_value_and_score_fn, dimension, cfg, '
        'center, center_value, center_score, scale_tf, radius, stalled, index):\n'
        '    evaluations, history = 0, []\n'
        '    search_sample_count = cfg.search_sample_count\n'
        '    structured_fresh_count = cfg.structured_fresh_sample_multiplier * dimension\n'
        '    for attempt in (index,):\n' + textwrap.indent(block, '        ')
        + '    return {"center": center, "center_value": center_value, "center_score": center_score, '
        '"radius": radius, "stalled": stalled, "evaluations": evaluations, "history": history}\n')
    namespace = dict(vars(frozen))
    exec(compile(wrapper, 'frozen_refinement_exact_excerpt', 'exec'), namespace)  # noqa: S102 - exact diagnostic source
    return namespace['execute']


def _factor_payload(result, prepared, factor_count, cfg, dimension):
    factor_cfg = current.FactorCorrelationGeometryConfig(factor_count=factor_count,
        max_condition_number=cfg.max_condition_number,
        holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
    fitted = current._factor_result_from_native(result, factor_cfg, dimension,
        int(prepared['active_training_rows']), int(prepared['fresh_holdout_count']))
    winner = int(prepared['best_index']) >= 0
    metadata = {**prepared, 'best_exact_value': float(prepared['best_value']) if winner else None,
        'best_exact_position': prepared['best_position'] if winner else None,
        'best_exact_score': prepared['best_score'] if winner else None,
        'best_exact_source': 'structured_fit_cloud' if winner else None}
    return current._factor_fit_payload(fitted, metadata)


def _record(result, cfg, dimension, radius, index):
    structured = cfg.refinement_geometry_policy == 'factor_correlation'
    first = (_factor_payload(result['first_fit'], result['preparation'], 1, cfg, dimension)
        if structured else _payload(result['first_fit'],
            dataclasses.replace(cfg, terminal_sample_count=cfg.regression_sample_count)))
    action = int(result['action'])
    if action == 0:
        return {'attempt': index, 'action': 'fit_cloud_recentered', 'center_value': float(result['center_value']),
            'fit': first, 'radius_action': 'retain', 'radius_after': radius,
            'proposal_score_gate': {'policy': cfg.proposal_score_acceptance_policy,
                'active': cfg.require_proposal_score_reduction, 'passed': not cfg.require_proposal_score_reduction,
                'proposal_evaluated': False}}
    base = {'attempt': index, 'radius_before': radius, 'recentered': bool(result['search_recentered']),
        'center_value': float(result['selected_value']), 'fit': first}
    if action == 1:
        return {**base, 'action': 'fit_rejected_contract'}
    proposals = result['attempts']
    selected = (_factor_payload(proposals['second_fit']['computed'], result['preparation'], 2, cfg, dimension)
        if int(proposals['attempted']) == 2 else first)
    last = proposals['last']
    rows = []
    for i in range(int(proposals['attempted'])):
        fit = first if i == 0 else selected
        if not bool(proposals['evaluated'][i]):
            rows.append({'factor_count': i + 1, 'fit_status': fit['status'], 'proposal_evaluated': False})
            continue
        row = {key: value[i] for key, value in proposals['histories'].items()}
        rows.append({'factor_count': i + 1 if structured else None, 'fit_status': fit['status'],
            'proposal_evaluated': True, 'actual_improvement': float(row['actual']),
            'predicted_improvement': float(row['predicted']), 'rho': float(row['rho']),
            'score_norm_before': float(row['old_norm']), 'score_norm_after': float(row['new_norm']),
            'score_reduction_passed': bool(row['score_passed']),
            'proposal_score_gate': current._proposal_gate_payload(row, cfg), 'accepted': bool(row['accepted'])})
    gate = (current._proposal_gate_payload(last, cfg) if bool(proposals['last_evaluated']) else
        {'policy': cfg.proposal_score_acceptance_policy, 'active': cfg.require_proposal_score_reduction,
            'passed': not cfg.require_proposal_score_reduction})
    return {**base, 'fit': selected, 'action': 'proposal_accepted' if action == 2 else 'proposal_rejected',
        'actual_improvement': float(last['actual']), 'predicted_improvement': float(last['predicted']),
        'rho': float(last['rho']), 'score_norm_before': float(last['old_norm']), 'score_norm_after': float(last['new_norm']),
        'proposal_score_gate': gate, 'boundary_active': bool(last['boundary']),
        'radius_action': ('contract', 'expand', 'retain')[int(proposals['radius_action'])],
        'radius_after': float(result['radius_after']), 'proposal_attempts': rows,
        'exact_incumbent_promoted_without_model_acceptance': bool(proposals['promoted_without_acceptance'])}


@pytest.mark.parametrize('case', ['symmetric_proposal', 'symmetric_recenter', 'symmetric_reject',
    'factor_one', 'factor_two', 'factor_two_reuse'])
def test_complete_refinement_records(case, request):
    checkpoint = FrozenCheckpoint('cfbc32d2', 'refinement')
    frozen = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    dimension = 5 if 'two' in case else 3
    cfg = current.SequentialMapCovarianceConfig(search_sample_count=4, regression_sample_count=24,
        refinement_geometry_policy='factor_correlation' if case.startswith('factor') else 'full_symmetric',
        structured_max_factors=2 if 'two' in case else 1, reuse_search_scores=case.endswith('reuse'),
        structured_holdout_score_relative_rmse=.001,
        score_holdout_relative_rmse=1e-12 if case.endswith('reject') else .35)
    if case.startswith('factor'):
        scalar, batched, _ = _inputs(dimension, 4)
    else:
        scalar = _target

        def batched(rows):
            return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

    center = (tf.linspace(tf.constant(-.5, D), tf.constant(.8, D), dimension) if case.endswith('recenter')
        else tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension))
    value, score = scalar(center)
    scale, radius, stalled, index = tf.ones([dimension], D), tf.constant(.25, D), 2, 0
    expected = _old_step(frozen, checkpoint.sources['bayesfilter/inference/sequential_map_covariance.py'])(
        scalar, batched, dimension, cfg, center, float(value), score, scale, float(radius), stalled, index)
    program = native.refinement_program(scalar, batched, dimension, cfg, cfg.search_sample_count)
    result = program(center, value, score, scale, radius, tf.constant(stalled), tf.constant(index))
    actual = {'center': result['center'], 'center_value': result['center_value'], 'center_score': result['center_score'],
        'radius': result['radius_after'], 'stalled': result['stalled'], 'evaluations': result['evaluations'],
        'history': [_record(result, cfg, dimension, float(radius), index)]}
    before, after = current._json_ready(expected), current._json_ready(actual)
    path = Path(request.config.getoption('xmlpath')).parent / f'refinement-{case}.json'
    with path.open('x') as handle:
        json.dump({'case': case, 'before': before, 'after': after,
            'frozen_source_sha256': checkpoint.hashes()}, handle, indent=2)
        handle.write('\n')
    _compare(after, before)
    if 'two' in case:
        assert len(after['history'][0]['proposal_attempts']) == 2
        assert after['history'][0]['proposal_attempts'][1]['proposal_evaluated']
    assert program.experimental_get_tracing_count() == 1
