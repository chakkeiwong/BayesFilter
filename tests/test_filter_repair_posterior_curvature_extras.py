"""Independent geometry, rejection, callback and ownership diagnostics."""

import copy
import dataclasses
import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_curvature_tf as runtime
from bayesfilter.inference.posterior_curvature_report import posterior_curvature_result
from tests.test_filter_repair_posterior_curvature import fixture, original, reference
from tests.test_filter_repair_quadratic_batches import _equal_records


def assert_ill_conditioned_rejection(actual, expected):
    """Check this rejected D3 fixture; its discarded precision is diagnostic.

    This comparator cannot admit any geometry or cover an accepted fit. Every
    other record field retains the original comparison, including rank, error
    status, condition diagnostics and exact callback/row accounting.
    """
    compared = []
    for record in (actual, expected):
        assert record['schema'] == 'bayesfilter.posterior_curvature_refinement.v1'
        assert record['accepted'] is False
        assert record['status'] == 'curvature_fit_rejected'
        assert all(record[name] is None for name in ('precision_z', 'refined_covariance', 'refined_factor'))
        diagnostics = record['diagnostics']
        assert diagnostics['failure_partition'] == 'fit'
        assert len(diagnostics['replicates']) == 1
        fit = diagnostics['replicates'][0]
        assert fit['accepted'] is False and fit['raw_spd'] is False
        assert type(fit['design_rank']) is int and fit['design_rank'] == 2
        assert fit['design_condition'] is None and fit['precision_condition'] is None
        matrix = fit['precision_z']
        assert isinstance(matrix, list) and len(matrix) == 3
        assert all(isinstance(row, list) and len(row) == 3 for row in matrix)
        assert all(value is None or type(value) is float for row in matrix for value in row)
        masked = copy.deepcopy(record)
        masked['diagnostics']['replicates'][0]['precision_z'] = 'unusable_rejected_diagnostic'
        compared.append(masked)
    _equal_records(*compared)


def assert_no_post_rejection_execution(raw):
    assert int(raw['status']) == runtime.STATUS.index('curvature_fit_rejected')
    assert int(raw['fit_count']) == 1
    assert not bool(raw['fit_accepted'][0])
    for flag in ('spread_done', 'audit_done', 'reconstruction_done', 'norm_done', 'proposal_done'):
        assert not bool(raw[flag]), flag
    for name in ('precision', 'covariance', 'refined_factor'):
        np.testing.assert_array_equal(raw[name], np.zeros([3, 3]))


class EqualCallable:
    """Equal but distinct two-output callbacks must not share a cached target."""

    def __init__(self, callback):
        self.callback = callback

    def __eq__(self, other):
        return isinstance(other, EqualCallable)

    def __call__(self, points):
        return self.callback(points)


def pure_fixture(dimension, *, design='uniform_box', replicates=2):
    # Stateless target for unlimited warm repetitions and independent geometry.
    from bayesfilter.inference.posterior_curvature_refinement import (
        PosteriorCurvatureRefinementConfig,
    )

    cfg = PosteriorCurvatureRefinementConfig(rows_per_partition=33, batch_size=8,
        replicate_count=replicates, fit_design=design, seed=-73)
    args = (tf.range(dimension, dtype=tf.float64) * .03,
        tf.eye(dimension, dtype=tf.float64) * .9 + tf.linalg.band_part(tf.ones([dimension, dimension], tf.float64) * .05, -1, 0),
        tf.constant(cfg.seed))

    def callback(theta):
        precision = tf.linalg.diag(tf.cast(tf.range(dimension) + 1, tf.float64)) + .05
        delta = theta - .2
        product = delta @ precision
        return -.5 * tf.reduce_sum(delta * product, axis=1), -product

    def eligibility(theta):
        return tf.ones([cfg.batch_size], tf.bool)

    return callback, eligibility, cfg, args


@pytest.mark.parametrize('design', ['uniform_box', 'uniform_ball'])
@pytest.mark.parametrize('dimension', [1, 3, 5])
def test_independent_gaussian_geometry(design, dimension):
    callback, eligible, cfg, args = pure_fixture(dimension, design=design, replicates=3)
    result = runtime.make_posterior_curvature_controller(callback, eligible, dimension, cfg)(*args)
    assert int(result['status']) == 1
    precision = np.diag(np.arange(dimension) + 1.) + .05
    covariance = np.linalg.inv(precision)
    np.testing.assert_allclose(result['covariance'], covariance, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['precision'], args[1].numpy().T @ precision @ args[1].numpy(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(result['refined_factor'], np.linalg.cholesky(covariance), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('design', ['zero', 'rank_one', 'ill_conditioned'])
def test_deficient_design_preserves_full_rejection(monkeypatch, design, request):
    callback, eligibility, cfg, args = pure_fixture(3)
    if design == 'zero':
        offsets = tf.zeros([33, 3], tf.float64)
    elif design == 'rank_one':
        offsets = tf.ones([33, 3], tf.float64)
    else:
        offsets = tf.reshape(tf.sin(tf.range(99, dtype=tf.float64)), [33, 3]) * tf.constant([1., 1e-8, 1e-10], tf.float64)
    _, baseline = original()
    monkeypatch.setattr(baseline, '_draw_offsets', lambda *args: offsets)
    expected = baseline.refine_posterior_local_curvature(callback, args[0], args[1],
        batched_eligibility_fn=eligibility, config=baseline.PosteriorCurvatureRefinementConfig(**dataclasses.asdict(cfg))).payload()
    for fit in expected['diagnostics'].get('replicates', []):
        if fit['design_rank'] < 3:
            fit['design_condition'] = None
    monkeypatch.setattr(runtime, 'draw_offsets', lambda *args: offsets)
    records, execution = {}, {}
    for jit in (False, True):
        raw = runtime.make_posterior_curvature_controller(callback, eligibility, 3, cfg, jit_compile=jit)(*args)
        actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
        records[str(jit)] = actual
        assert_no_post_rejection_execution(raw)
        if design == 'ill_conditioned':
            assert int(raw['fit_rank'][0]) == 2
            assert np.isposinf(float(raw['fit_metrics'][0, 0]))
        execution[str(jit)] = {name: bool(raw[name]) for name in
            ('spread_done', 'audit_done', 'reconstruction_done', 'norm_done', 'proposal_done')}
    comparison = 'full_original_record_equivalence'
    diagnostic_difference = {}
    if design == 'ill_conditioned':
        comparison = 'rejection_and_no_use; discarded_precision_values_are_explanatory_only'
        for mode, actual in records.items():
            left = np.asarray(actual['diagnostics']['replicates'][0]['precision_z'])
            right = np.asarray(expected['diagnostics']['replicates'][0]['precision_z'])
            diagnostic_difference[mode] = {
                'max_absolute_difference': float(np.max(np.abs(left - right))),
                'passes_historical_entrywise_comparison': bool(np.allclose(left, right, atol=1e-10, rtol=1e-10)),
                'role': 'explanatory_only; rejected_matrix_is_not_an_admissible_precision',
            }
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'posterior-design-{design}.json').open('x') as out:
        json.dump({'original': expected, 'candidate': records, 'offsets': offsets.numpy().tolist(),
            'comparison_criterion': comparison, 'diagnostic_precision_difference': diagnostic_difference,
            'post_rejection_execution': execution}, out, indent=2, allow_nan=False)
        out.write('\n')
    for actual in records.values():
        if design == 'ill_conditioned':
            assert_ill_conditioned_rejection(actual, expected)
        else:
            _equal_records(actual, expected)
        assert actual['status'] == 'curvature_fit_rejected'
        assert actual['refined_factor'] is None


def rejected_record_fixture():
    """Minimal diagnostic record for mutations of the comparison boundary."""
    return {'schema': 'bayesfilter.posterior_curvature_refinement.v1', 'accepted': False,
        'status': 'curvature_fit_rejected', 'precision_z': None, 'refined_covariance': None,
        'refined_factor': None, 'center': [0., .03, .06], 'diagnostics': {
            'failure_partition': 'fit', 'callback_batches': 21, 'target_rows': 168,
            'replicates': [{'precision_z': np.eye(3).tolist(), 'design_rank': 2,
                'design_condition': None, 'precision_condition': None, 'raw_spd': False,
                'accepted': False, 'selection_relative_rmse': .095}]}}


@pytest.mark.parametrize('field', ['accepted', 'status', 'precision_z', 'refined_covariance',
    'refined_factor', 'failure_partition', 'design_rank', 'design_condition', 'precision_condition',
    'fit_accepted', 'raw_spd', 'callback_batches', 'target_rows', 'selection_relative_rmse',
    'missing_matrix', 'matrix_shape', 'matrix_dtype', 'replicate_count', 'center'])
@pytest.mark.parametrize('side', ['actual', 'expected'])
def test_rejection_contract_rejects_mutations(field, side):
    records = {'actual': rejected_record_fixture(), 'expected': rejected_record_fixture()}
    record = records[side]
    diagnostics = record['diagnostics']
    fit = diagnostics['replicates'][0]
    if field == 'accepted':
        record[field] = True
    elif field == 'status':
        record[field] = 'eligible_for_local_position_factor'
    elif field in ('precision_z', 'refined_covariance', 'refined_factor'):
        record[field] = np.eye(3).tolist()
    elif field == 'failure_partition':
        diagnostics[field] = 'proposal'
    elif field in ('design_rank', 'design_condition', 'precision_condition'):
        fit[field] = {'design_rank': 3, 'design_condition': 1., 'precision_condition': 1.}[field]
    elif field in ('fit_accepted', 'raw_spd'):
        fit['accepted' if field == 'fit_accepted' else field] = True
    elif field in ('callback_batches', 'target_rows'):
        diagnostics[field] += 1
    elif field == 'selection_relative_rmse':
        fit[field] += .01
    elif field == 'missing_matrix':
        del fit['precision_z']
    elif field == 'matrix_shape':
        fit['precision_z'] = [[1.]]
    elif field == 'matrix_dtype':
        fit['precision_z'][0][0] = 'not a diagnostic number'
    elif field == 'replicate_count':
        diagnostics['replicates'].append(copy.deepcopy(fit))
    elif field == 'center':
        record['center'][0] += .01
    with pytest.raises((AssertionError, KeyError)):
        assert_ill_conditioned_rejection(records['actual'], records['expected'])


def test_rejection_contract_preserves_arbitrary_discarded_diagnostics():
    expected, actual = rejected_record_fixture(), rejected_record_fixture()
    actual['diagnostics']['replicates'][0]['precision_z'] = [[None, 1e200, -1e200]] * 3
    saved = copy.deepcopy(actual)
    assert_ill_conditioned_rejection(actual, expected)
    assert actual == saved  # Complete forensic diagnostics must not be overwritten.


def test_rejected_precision_cannot_reach_later_phases(monkeypatch, request):
    base, eligibility, cfg, args = pure_fixture(3)
    poison = tf.Variable(1., dtype=tf.float64)
    with tf.device(poison.device):
        calls = tf.Variable(0, dtype=tf.int64)

    def callback(points):
        calls.assign_add(1)
        return base(points)

    def rejected_fit(*_args, **_kwargs):
        return {'raw_precision': tf.eye(3, dtype=tf.float64) * poison,
            'raw_eigenvalues': tf.ones(3, tf.float64), 'raw_spd': tf.constant(True),
            'design_rank': tf.constant(2), 'design_condition': tf.constant(float('inf'), tf.float64),
            'precision_condition': tf.constant(1., tf.float64), 'selection_relative_rmse': tf.constant(0., tf.float64)}

    monkeypatch.setattr(runtime, 'fit_dense_score_precision_tf', rejected_fit)
    records = []
    for jit in (False, True):
        program = runtime.make_posterior_curvature_controller(callback, eligibility, 3, cfg, jit_compile=jit)
        for value in (1., 1e200, float('nan')):
            poison.assign(value)
            calls.assign(0)
            raw = program(*args)
            assert_no_post_rejection_execution(raw)
            result = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
            assert result['accepted'] is False and result['status'] == 'curvature_fit_rejected'
            assert all(result[name] is None for name in ('precision_z', 'refined_covariance', 'refined_factor'))
            assert int(calls) == result['diagnostics']['callback_batches'] == 21
            assert result['diagnostics']['failure_partition'] == 'fit'
            assert len(result['diagnostics']['replicates']) == 1
            records.append({'jit_compile': jit, 'poison': str(value), 'calls': int(calls), 'result': result})
        assert program.experimental_get_tracing_count() == 1
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-rejected-precision-no-use.json').open('x') as out:
        json.dump({'role': 'rejection isolation diagnostic; not numerical-fit correctness', 'runs': records},
            out, indent=2, allow_nan=False)
        out.write('\n')


@pytest.mark.parametrize('fault,expected', [
    ('eligibility_dtype', TypeError), ('eligibility_shape', ValueError),
    ('values_dtype', TypeError), ('scores_dtype', TypeError),
    ('values_shape', ValueError), ('scores_shape', ValueError), ('exception', RuntimeError),
])
def test_malformed_callbacks_raise(fault, expected):
    base, eligible, cfg, args = pure_fixture(3)

    def eligibility(theta):
        answer = eligible(theta)
        if fault == 'eligibility_dtype':
            answer = tf.cast(answer, tf.int32)
        if fault == 'eligibility_shape':
            answer = answer[:, None]
        return answer

    def callback(theta):
        if fault == 'exception':
            raise RuntimeError('callback programming error')
        values, scores = base(theta)
        if fault == 'values_dtype':
            values = tf.cast(values, tf.float32)
        if fault == 'scores_dtype':
            scores = tf.cast(scores, tf.float32)
        if fault == 'values_shape':
            values = values[:, None]
        if fault == 'scores_shape':
            scores = scores[:, 0]
        return values, scores

    with pytest.raises(expected):
        reference(callback, eligibility, cfg, args)
    with pytest.raises(expected):
        runtime.make_posterior_curvature_controller(callback, eligibility, 3, cfg)(*args)


def test_replicate_instability_preserves_all_fit_records():
    _unused, eligible, cfg, args, counters, _positions = fixture(3, replicates=3, rows=17, batch=7)
    counter = counters[1]
    args = (tf.zeros(3, tf.float64), tf.eye(3, dtype=tf.float64), args[2])

    def callback(theta):
        call = counter.assign_add(1)
        replicate = ((call - 2) // 3) % 3
        scale = tf.where(replicate == 0, tf.constant(1., tf.float64), tf.constant(2., tf.float64))
        return -.5 * scale * tf.reduce_sum(theta**2, axis=1), -scale * theta

    expected, _ = reference(callback, eligible, cfg, args)
    original_calls = int(counter)
    counter.assign(0)
    raw = runtime.make_posterior_curvature_controller(callback, eligible, 3, cfg)(*args)
    actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
    _equal_records(actual, expected)
    assert actual['status'] == 'replicate_instability'
    assert int(counter) == original_calls == 19
    assert len(actual['diagnostics']['replicates']) == 3


def scaled_target(base, strength):
    def callback(points):
        values, scores = base(points)
        return strength * values, strength * scores
    return callback


def test_cache_identity_changed_target_and_resource_release(request):
    runtime.clear_posterior_curvature_controller_cache()
    base, eligibility, cfg, args = pure_fixture(3)
    strength = tf.Variable(1., dtype=tf.float64)
    target = EqualCallable(scaled_target(base, strength))
    program = runtime.posterior_curvature_controller(target, eligibility, 3, cfg)
    assert runtime.posterior_curvature_controller(target, eligibility, 3,
        dataclasses.replace(cfg, seed=17, lineage={'changed': 'reporting only'})) is program
    records = []
    for value in (1., 2.):
        strength.assign(value)
        expected, _ = reference(target, eligibility, cfg, args)
        raw = program(*args)
        actual = posterior_curvature_result(raw, args[0], args[1], cfg).payload()
        _equal_records(actual, expected)
        records.append(actual)
    assert records[0]['refined_covariance'] != records[1]['refined_covariance']
    assert program.experimental_get_tracing_count() == 1
    concrete = program.get_concrete_function()
    watched = {name: weakref.ref(value) for name, value in
        (('graph', concrete.graph), ('target', target), ('strength', strength))}
    other = EqualCallable(base)
    assert other == target
    assert runtime.posterior_curvature_controller(other, eligibility, 3, cfg) is not program
    runtime.clear_posterior_curvature_controller_cache()
    del concrete, program, target, strength, raw
    gc.collect()
    released = {name: ref() is None for name, ref in watched.items()}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'posterior-resource-ownership.json').open('x') as out:
        json.dump({'released': released, 'changed_target_records': records, 'cache_capacity': 1,
            'nonclaim': 'Python collection does not prove native executable eviction.'}, out, indent=2, allow_nan=False)
        out.write('\n')
    assert all(released.values())
