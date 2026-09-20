"""Three-way original/intermediate/repaired terminal eigenpair records."""

import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from bayesfilter.inference import sequential_terminal_tf as native
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_sequential_proposal import _target
from tests.test_filter_repair_sequential_terminal import _payload

D = tf.float64


@pytest.mark.parametrize('case', ['healthy3', 'healthy5', 'paired', 'insufficient', 'paired_insufficient', 'rank', 'holdout'])
def test_original_and_intermediate_complete_terminal_records(case, request):
    authorities = {name: FrozenCheckpoint(revision, 'terminal_original_' + name)
        for name, revision in [('original', '3582b4ac'), ('intermediate', 'cfbc32d2')]}
    dimension = 5 if case == 'healthy5' else 10 if case == 'insufficient' else 3
    count = 2 if case == 'insufficient' else 4 if case == 'paired_insufficient' else 24
    cfg = current.SequentialMapCovarianceConfig(terminal_sample_count=count,
        pair_disjoint_score_holdout=case in ('paired', 'paired_insufficient'),
        score_holdout_relative_rmse=1e-12 if case == 'holdout' else .35)
    calls = tf.Variable(0, dtype=tf.int64, trainable=False)

    def scalar(row):
        calls.assign_add(1)
        return _target(row)

    def batched(rows):
        calls.assign_add(tf.shape(rows, out_type=tf.int64)[0])
        return -.5 * tf.reduce_sum(rows ** 2, axis=1) - .1 * tf.reduce_sum(rows ** 4, axis=1), -rows - .4 * rows ** 3

    center = tf.linspace(tf.constant(.002, D), tf.constant(.004, D), dimension)
    score, scale = _target(center)[1], tf.linspace(tf.constant(.8, D), tf.constant(1.2, D), dimension)
    radius, seed = tf.constant(0. if case == 'rank' else .25, D), tf.constant([2026, 100718])
    records, counts = {}, {}
    for name, checkpoint in authorities.items():
        module = checkpoint.load('bayesfilter.inference.sequential_map_covariance')
        calls.assign(0)
        result, evaluations = module._fit_score_curvature(scalar, center, score, scale,
            dimension=dimension, radius=float(radius), sample_count=count, seed=tuple(seed.numpy().tolist()),
            config=cfg, evaluations=11, batched_value_and_score_fn=batched)
        records[name] = module._json_ready(result)
        counts[name] = int(calls)
        assert int(calls) == evaluations - 11
    program = native.terminal_program(scalar, batched, dimension, cfg)
    arguments = (center, score, scale, radius, seed)

    @tf.function(input_signature=program.input_signature, jit_compile=True, autograph=False)
    def enclosed(*args):
        return program.python_function(*args)

    for name, function in [('direct', program), ('enclosed', enclosed)]:
        calls.assign(0)
        computed = function(*arguments)
        records[name] = _payload(computed, cfg)
        counts[name] = int(calls)
        assert counts[name] == int(computed['evaluations']) == counts['original'] == counts['intermediate']
    report = {'case': case, 'records': records, 'target_rows': counts,
        'source_sha256': {name: checkpoint.hashes() for name, checkpoint in authorities.items()},
        'differences_from_original': {name: _record_differences(row, records['original'])
            for name, row in records.items()},
        'nonclaims': ['The intermediate eigenpair defect is preserved and compared, not a precision authority.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'terminal-original-{case}.json').open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    _compare(records['direct'], records['original'])
    _compare(records['enclosed'], records['original'])
    assert program.experimental_get_tracing_count() == enclosed.experimental_get_tracing_count() == 1
