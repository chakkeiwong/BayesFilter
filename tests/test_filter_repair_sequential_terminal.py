"""Actual seeded terminal-fit records against a fully pinned public helper."""

import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_terminal_tf as native
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_sequential_proposal import _target

D = tf.float64


def _payload(result, cfg):
    computed, seed = result['record'], result['seed'].numpy().tolist()
    status = int(computed['status'])
    if status == -1:
        return {'status': 'insufficient_symmetric_support', 'seed': seed}
    winner = bool(result['has_best'])
    payload = {'status': ('rank_deficient_symmetric_fit', 'usable', 'score_holdout_failed')[status],
        'rank': int(computed['rank']), 'seed': seed,
        'best_exact_value': float(computed['best_value']) if winner else None,
        'best_exact_position': computed['best_position'] if winner else None,
        'best_exact_score': computed['best_score'] if winner else None,
        'best_exact_source': 'score_fit_cloud' if winner else None}
    if status != 0:
        payload.update(train_score_rmse=float(computed['train_score_rmse']),
            holdout_score_relative_rmse=float(computed['holdout_score_relative_rmse']),
            raw_eigenvalues=computed['raw_eigenvalues'], projected_eigenvalues=computed['projected_eigenvalues'],
            projection_relative_frobenius=float(computed['projection_relative_frobenius']),
            projected_precision_z=computed['projected_precision_z'])
        if cfg.pair_disjoint_score_holdout:
            training, holdout = current.partition_schema(cfg.terminal_sample_count, cfg.holdout_fraction, pair_disjoint=True)
            payload.update(pair_disjoint_score_holdout=True, training_sample_count=len(training),
                holdout_sample_count=len(holdout))
    return current._json_ready(payload)


@pytest.mark.parametrize('case', ['healthy3', 'healthy5', 'paired', 'insufficient', 'paired_insufficient', 'rank', 'holdout'])
def test_actual_terminal_seeded_records_and_enclosure(case, request):
    # The intermediate cfbc32d2 XLA eigensystem has demonstrated residual error.
    # The separate three-way test preserves it; precision is checked against
    # the original source with every field and the same tolerance retained.
    checkpoint = FrozenCheckpoint('3582b4ac', 'terminal')
    frozen = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
    dimension = 5 if case == 'healthy5' else 10 if case == 'insufficient' else 3
    count = 2 if case == 'insufficient' else 4 if case == 'paired_insufficient' else 24
    pair = case in ('paired', 'paired_insufficient')
    cfg = current.SequentialMapCovarianceConfig(terminal_sample_count=count,
        pair_disjoint_score_holdout=pair, score_holdout_relative_rmse=1e-12 if case == 'holdout' else .35)
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def scalar(row):
        calls.assign_add(1)
        return _target(row)

    def batched(rows):
        calls.assign_add(tf.shape(rows, out_type=tf.int64)[0])
        return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension)
    score, scale = _target(center)[1], tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), dimension)
    radius = tf.constant(0. if case == 'rank' else .25, D)
    seed = tf.constant([2026, 100718])
    before, evaluations = frozen._fit_score_curvature(scalar, center, score, scale,
        dimension=dimension, radius=float(radius), sample_count=count, seed=tuple(seed.numpy().tolist()),
        config=cfg, evaluations=11, batched_value_and_score_fn=batched)
    original = current._json_ready(before)
    expected_calls = evaluations - 11
    assert int(calls) == expected_calls
    program = native.terminal_program(scalar, batched, dimension, cfg)
    args = (center, score, scale, radius, seed)
    direct = program(*args)

    @tf.function(input_signature=program.input_signature, jit_compile=True, autograph=False)
    def enclosed(*args):
        return program.python_function(*args)

    enclosing = enclosed(*args)
    assert int(calls) == expected_calls * 3
    for result in (direct, enclosing):
        assert int(result['evaluations']) == expected_calls
        assert bool(result['usable']) == (before['status'] == 'usable')
        _compare(_payload(result, cfg), original)
    expected_status = ('insufficient_symmetric_support' if 'insufficient' in case else
        'rank_deficient_symmetric_fit' if case == 'rank' else 'score_holdout_failed' if case == 'holdout' else 'usable')
    assert before['status'] == expected_status
    assert program.experimental_get_tracing_count() == enclosed.experimental_get_tracing_count() == 1
    path = Path(request.config.getoption('xmlpath')).parent / f'terminal-{case}.json'
    with path.open('x') as handle:
        json.dump({'case': case, 'before': original, 'direct': _payload(direct, cfg),
            'enclosed': _payload(enclosing, cfg), 'exact_callback_rows': int(calls),
            'frozen_source_sha256': checkpoint.hashes()}, handle, indent=2)
        handle.write('\n')
