"""Independent original-record qualification of runtime geometry row counts."""

import dataclasses
import re

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference.quadratic_geometry_fit_tf import make_geometry_fit_program
from bayesfilter.inference.quadratic_geometry_prepare_tf import (
    make_geometry_seeded_partition_program,
)
from bayesfilter.ops.geometry_permutation_tf import make_geometry_permutation_program
from bayesfilter.ops.geometry_random_tf import _draw_kernel
from tests.test_filter_repair_geometry_control import save, source, stable_hlo
from tests.test_filter_repair_geometry_fit import fixture, materialize, original_result
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def active_fixture(dimension, case, training, holdout, *, capacity=16, shift=0., counter=True):
    target, cfg, inputs, calls = fixture(dimension, case, shift=shift, counter=counter)
    cfg = dataclasses.replace(cfg, sample_count=training + holdout,
        holdout_fraction=holdout / (training + holdout))
    zt, yt, st = inputs[3][:training], inputs[4][:training], inputs[5][:training]
    zh, yh = inputs[6][:holdout], inputs[7][:holdout]
    offsets = tf.concat([zh, zt], axis=0)
    points = inputs[0][None] + offsets * inputs[1][None]
    values = tf.concat([yh, yt], axis=0)
    best = int(np.argmax(np.r_[float(inputs[8]), values.numpy()]))
    position = inputs[0] if best == 0 else points[best - 1]
    value, score = target(position)
    compact = (inputs[0], inputs[1], inputs[2], zt, yt, st, zh, yh, inputs[8], inputs[9],
        position, value, score, tf.constant(best, tf.int64), tf.constant(0 if best == 0 else 2),
        tf.constant(True), tf.constant(training + holdout + 1, tf.int64))

    def poison(rows, extent):
        padding = extent - rows.shape[0]
        return tf.concat([rows, tf.fill([padding, *rows.shape[1:]], tf.constant(float('nan'), D))], 0)

    padded = (*compact[:3], poison(zt, capacity), poison(yt, capacity), poison(st, capacity),
        poison(zh, 4), poison(yh, 4), *compact[8:], tf.constant(training), tf.constant(holdout))
    if calls is not None:
        calls.assign(0)
    return target, cfg, compact, padded, calls


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('case', ['gaussian', 'nonquadratic', 'zero_design', 'reject_holdout'])
def test_active_fit_preserves_complete_original_records(dimension, case, monkeypatch, request):
    rank = dimension - 1
    minimum = 2 + dimension + rank
    target, cfg, _, _, calls = active_fixture(dimension, case, 16, 4)
    programs = {jit: make_geometry_fit_program(target, dimension, rank, 16, 4, cfg,
        jit_compile=jit, active_rows=True) for jit in (False, True)}
    records, hlos = [], []
    for training, holdout, shift in ((minimum, 0, 0.), (minimum + 1, 1, .03), (16, 4, .06)):
        _, local_cfg, compact, padded, _ = active_fixture(dimension, case, training, holdout, shift=shift)
        calls.assign(0)
        expected, payload, hashes = original_result(target, local_cfg, compact, monkeypatch)
        expected_calls = int(calls)
        actual = {}
        for jit, program in programs.items():
            calls.assign(compact[-1])
            raw = program(*padded)
            actual[str(jit)] = materialize(raw, local_cfg, compact)
            _equal_records(actual[str(jit)], expected)
            assert int(calls) == expected_calls == actual[str(jit)]['exact_evaluation_count']
            if jit:
                hlos.append(program.experimental_get_compiler_ir(*padded)(stage='hlo'))
        records.append({'training': training, 'holdout': holdout, 'original': expected,
            'original_payload': payload, 'candidate': actual, 'original_source_sha256': hashes})
    assert programs[True].experimental_get_tracing_count() == 1
    assert len({stable_hlo(hlo) for hlo in hlos}) == 1
    entry = hlos[0][hlos[0].rfind('\nENTRY '):]
    # XLA removes the zero-element rank-zero basis. All nonempty inputs and
    # the target counter must remain runtime operands, including both counts.
    operands = sum(value.shape.num_elements() != 0 for value in padded) + 1
    assert len(re.findall(r'\bparameter\(\d+\)', entry)) == operands
    save(request, f'geometry-active-{dimension}-{case}.json', {'records': records,
        'runtime_count_operands': True, 'stable_hlo': True, 'trace_count': 1,
        'hlo_bytes': len(hlos[0].encode()), 'inactive_padding': 'nan'})


def test_invalid_counts_cannot_call_target(request):
    target, cfg, _, padded, calls = active_fixture(3, 'gaussian', 12, 2)
    program = make_geometry_fit_program(target, 3, 2, 16, 4, cfg, active_rows=True)
    records = []
    for training, holdout in ((0, 0), (-1, 0), (17, 0), (12, -1), (12, 5)):
        calls.assign(0)
        raw = program(*padded[:-2], tf.constant(training), tf.constant(holdout))
        assert not bool(raw['fit']['finite'])
        assert int(raw['status']) == 0 and int(calls) == 0
        assert int(raw['evaluation_count']) == int(padded[-3])
        records.append({'training': training, 'holdout': holdout, 'status': int(raw['status']),
            'target_calls': int(calls), 'evaluation_count': int(raw['evaluation_count'])})
    save(request, 'geometry-active-invalid.json', records)


@pytest.mark.parametrize('capacity', [0, 1, 7, 17, 33])
def test_active_permutation_preserves_cpu_stream(capacity, request):
    program = make_geometry_permutation_program(capacity)
    records = []
    for seed in ((173, 251), (739, 911), (0, 2147483647)):
        for count in sorted({0, min(1, capacity), capacity // 2, capacity}):
            actual = program(tf.constant(count), tf.constant(seed))
            with tf.device('/CPU:0'):
                expected = _draw_kernel('permutation', (count,))(tf.constant(seed))
            assert bool(actual['valid'])
            np.testing.assert_array_equal(actual['permutation'][:count], expected)
            np.testing.assert_array_equal(actual['permutation'][count:], np.arange(count, capacity))
            records.append({'count': count, 'seed': seed, 'permutation': actual['permutation']})
    for count in (-1, capacity + 1):
        result = program(tf.constant(count), tf.constant([1, 2]))
        assert not bool(result['valid'])
        np.testing.assert_array_equal(result['permutation'], np.arange(capacity))
    assert program.experimental_get_tracing_count() == 1
    save(request, f'geometry-active-permutation-{capacity}.json', records)


def test_seeded_partition_and_fit_enclose_runtime_counts(request):
    target, cfg, compact, _, _ = active_fixture(3, 'nonquadratic', 16, 4)
    offsets = tf.concat([compact[6], compact[3]], 0)
    points = compact[0][None] + offsets * compact[1][None]
    values = tf.concat([compact[7], compact[4]], 0)
    scores = tf.stack([target(point)[1] for point in points])
    partition = make_geometry_seeded_partition_program(3, 20, 7, .2)
    fit = make_geometry_fit_program(target, 3, 2, 20, 20, cfg, active_rows=True)

    @tf.function(input_signature=[tf.TensorSpec([20, 3], D), tf.TensorSpec([20], D),
        tf.TensorSpec([20, 3], D), tf.TensorSpec([2], tf.int32)], autograph=False, jit_compile=True)
    def enclosing(z, y, score, seed):
        split = partition(z, y, score, compact[1], seed)
        result = fit(*compact[:3], split['z_train'], split['y_train'], split['score_train'],
            split['z_holdout'], split['y_holdout'], *compact[8:], split['training_count'], split['holdout_count'])
        return split, result

    records, hlos = [], []
    for invalid in (0, 5):
        y = tf.where(tf.range(20) < invalid, tf.constant(float('nan'), D), values)
        args = (offsets, y, scores, tf.constant([311, 521]))
        split, result = enclosing(*args)
        count = 20 - invalid
        assert int(split['finite_count']) == count and bool(split['ready'])
        with tf.device('/CPU:0'):
            expected_order = _draw_kernel('permutation', (count,))(args[3]).numpy() + invalid
        holdout = min(int(.2 * count), count - 7)
        np.testing.assert_array_equal(split['holdout_indices'][:holdout], expected_order[:holdout])
        np.testing.assert_array_equal(split['train_indices'][:count - holdout], expected_order[holdout:])
        exact = make_geometry_fit_program(target, 3, 2, count - holdout, holdout, cfg)(
            *compact[:3], tf.gather(offsets, expected_order[holdout:]), tf.gather(y, expected_order[holdout:]),
            tf.gather(scores, expected_order[holdout:]) * compact[1], tf.gather(offsets, expected_order[:holdout]),
            tf.gather(y, expected_order[:holdout]), *compact[8:])
        report_inputs = (*compact[:6], tf.gather(offsets, expected_order[:holdout]), *compact[7:])
        _equal_records(materialize(result, cfg, report_inputs), materialize(exact, cfg, report_inputs))
        hlos.append(enclosing.experimental_get_compiler_ir(*args)(stage='hlo'))
        records.append({'count': count, 'partition': split, 'fit': result})
    assert enclosing.experimental_get_tracing_count() == 1
    assert stable_hlo(hlos[0]) == stable_hlo(hlos[1])
    save(request, 'geometry-active-enclosing.json', {'records': records, 'stable_hlo': True,
        'trace_count': 1, 'scope': 'seeded finite partition through fit/replay; initial callback clouds not enclosed'})


def test_single_row_and_rank_cutoff_use_active_extent(request):
    checkpoint, original = source('3582b4ac')
    prior_checkpoint, prior = source('ca920bac5')
    capacity = 16
    q = tf.constant([[1.], [0.]], D)
    center_score = tf.zeros([2], D)
    cfg = geometry.LowRankSPDQuadraticGeometryConfig(rank=1, eigenvalue_floor=1e-8)

    @tf.function(input_signature=[tf.TensorSpec([capacity, 2], D), tf.TensorSpec([capacity], D),
        tf.TensorSpec([capacity, 2], D), tf.TensorSpec([], tf.int32)], autograph=False, jit_compile=True)
    def fit(z, y, score, count):
        return geometry._quadratic_fit_kernel(z, y, score, q, center_score,
            tf.constant(cfg.eigenvalue_floor, D), tf.constant(cfg.max_condition_number, D), active_rows=count)

    records = []
    for count, small in ((1, 1.), (2, 1e-16), (2, 2e-15), (16, 2e-15)):
        z = np.tile([1., small], (count, 1))
        score = -z * [3., 2.]
        y = 5. + .5 * np.sum(z * score, axis=1)
        args = (tf.constant(np.pad(z, ((0, capacity - count), (0, 0)), constant_values=np.nan), D),
            tf.constant(np.pad(y, (0, capacity - count), constant_values=np.nan), D),
            tf.constant(np.pad(score, ((0, capacity - count), (0, 0)), constant_values=np.nan), D), tf.constant(count))
        actual = fit(*args)
        expected = original._fit_constrained_quadratic(z, y, score, q_basis=q.numpy(), cfg=cfg,
            dim=2, rank=1, center_score_z=center_score.numpy())
        prior_kernel = tf.function(prior._quadratic_fit_kernel, input_signature=[
            tf.TensorSpec([count, 2], D), tf.TensorSpec([count], D), tf.TensorSpec([count, 2], D),
            tf.TensorSpec([2, 1], D), tf.TensorSpec([2], D), tf.TensorSpec([], D), tf.TensorSpec([], D)],
            autograph=False, jit_compile=True)
        before = prior_kernel(tf.constant(z, D), tf.constant(y, D), tf.constant(score, D), q, center_score,
            tf.constant(cfg.eigenvalue_floor, D), tf.constant(cfg.max_condition_number, D))
        records.append({'count': count, 'small': small, 'candidate': actual, 'original': expected,
            'prior_compact_xla': before})
    save(request, 'geometry-active-rank.json', {'records': records, 'original_source_sha256': checkpoint.hashes(),
        'prior_source_sha256': prior_checkpoint.hashes()})
    for row in records:
        actual, expected = row['candidate'], row['original']
        assert int(actual['score_design_rank']) == expected['score_design_rank']
        for field, value in row['prior_compact_xla'].items():
            np.testing.assert_allclose(actual[field], value, rtol=1e-10, atol=1e-10)
        if row['count'] == 2 and row['small'] == 2e-15:
            assert not bool(actual['design_resolved'])
            assert float(actual['design_roundoff_indicator']) > float(actual['design_roundoff_limit'])
            continue
        assert bool(actual['design_resolved'])
        for field in ('precision', 'intercept', 'loss', 'score_rmse'):
            np.testing.assert_allclose(actual[field], expected[field], rtol=1e-10, atol=1e-10)


def test_unreliable_retained_solve_reports_error_without_target_use(request):
    dimension = 2
    target, cfg, inputs, calls = fixture(dimension, 'gaussian')
    cfg = dataclasses.replace(cfg, rank=1, eigenvalue_floor=1e-8)
    observations = []
    # For repeated [1, x] rows with this basis, the design condition is
    # 2/x + O(x). Four design rows put the guard crossing at 8*sqrt(eps).
    boundary = float(8 * np.sqrt(np.finfo(np.float64).eps))
    for small in (1e-6, boundary * 1.01, boundary * .99, 1e-8, 2e-15):
        z = tf.constant([[1., small], [1., small]], D)
        score = -z * tf.constant([3., 2.], D)
        y = 5. + .5 * tf.reduce_sum(z * score, axis=1)
        args = (*inputs[:3], z, y, score, inputs[6][:0], inputs[7][:0],
            inputs[8], tf.zeros([2], D), *inputs[10:])
        report_cfg = cfg
        for jit in (False, True):
            for active in (False, True):
                calls.assign(0)
                program = make_geometry_fit_program(target, dimension, 1, 2, 0, report_cfg,
                    jit_compile=jit, active_rows=active)
                raw = program(*args, *((tf.constant(2), tf.constant(0)) if active else ()))
                report = materialize(raw, report_cfg, args)
                rejected = small < boundary
                assert (report['status'] == 'fit_design_ill_conditioned') is rejected
                if rejected:
                    assert not report['accepted'] and report['precision'] is report['covariance'] is None
                    assert int(calls) == 0 and report['exact_evaluation_count'] == int(args[-1])
                    assert 'center_refinement' not in report and 'best_evaluated_replay' not in report
                    assert report['fit']['score_design_roundoff_indicator'] > report['fit']['score_design_roundoff_limit']
                else:
                    assert bool(raw['fit']['design_resolved']) and int(calls) > 0
                observations.append({'weak_direction': small, 'jit': jit, 'active': active,
                    'record': report, 'target_calls': int(calls)})
        direct = geometry._fit_constrained_quadratic(z, y, score, q_basis=inputs[2], cfg=cfg,
            dim=dimension, rank=1, center_score_z=tf.zeros([2], D))
        assert (direct['status'] == 'fit_design_ill_conditioned') is (small < boundary)
    save(request, 'geometry-active-condition-guard.json', {'observations': observations,
        'scope': 'relative retained-solve roundoff guard; no covariance/proposal/replay after rejected fit'})
