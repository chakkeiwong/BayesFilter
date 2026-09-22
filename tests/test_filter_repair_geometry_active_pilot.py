"""Exact original pilot records with runtime retained direction counts."""

import re

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from bayesfilter.inference.quadratic_geometry_pilot_report import geometry_pilot_report
from bayesfilter.inference.quadratic_geometry_pilot_tf import (
    make_geometry_pilot_program,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_geometry_control import clean, save, stable_hlo
from tests.test_filter_repair_geometry_pilot import record_pilot, reference
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def stable_pilot_hlo(hlo):
    # 02675 differs only in nested synthetic-zero node uniquifiers. Preserve
    # every instruction, constant, operand and all other metadata verbatim.
    return re.sub(r'(op_name="StatefulPartitionedCall/zeros(?:_\d+)?/_)\d+(" source_file="dummy_file_name")',
        r'\1_ID\2', stable_hlo(hlo))


def test_active_pilot_degenerate_basis_and_hlo_attribution(request):
    import difflib
    from pathlib import Path

    checkpoint = FrozenCheckpoint('40f169fcc', 'pilot_active_attribution')
    prior = checkpoint.load('bayesfilter.inference.quadratic_geometry_pilot_tf')
    records = []
    directory = Path(request.config.getoption('xmlpath')).parent
    for batched, case in ((False, 'gaussian'), (True, 'gaussian'), (True, 'nonfinite')):
        target, calls, extent, positions, reset = active_pilot_fixture(3, batched, case)
        cfg = LowRankSPDQuadraticGeometryConfig(rank=2, pilot_direction_count=9)
        programs = {jit: make_geometry_pilot_program(target, 3, 2, 9, batched=batched,
            jit_compile=jit, active_rows=True) for jit in (False, True)}
        directions = np.random.default_rng(256).normal(size=(9, 3))
        hlos = []
        for count in (0, 1, 4, 9):
            raw_directions = directions.copy()
            raw_directions[count:] = 0.
            compact = raw_directions[:count] / np.linalg.norm(raw_directions[:count], axis=1, keepdims=True)
            inputs = (tf.range(3, dtype=D) * .01 + count * .003,
                (tf.range(3, dtype=D) * .1 + .8) * (1. + count * .01),
                tf.constant(np.pad(compact, ((0, 9-count), (0, 0)), constant_values=np.nan), D),
                tf.constant(cfg.pilot_radius, D), tf.ones([3], D) * .3, tf.constant(count))
            reset()
            original, hashes = reference(target, cfg, inputs[:5], raw_directions, batched=batched)
            candidates = {}
            for jit, program in programs.items():
                reset()
                raw = program(*inputs)
                candidates[str(jit)] = {'raw': raw, 'report': compact_report(raw, 2, 9, batched),
                    'calls': int(calls), 'extent': int(extent), 'positions': positions.read_value()}
                reset()
                old = prior.make_geometry_pilot_program(target, 3, 2, count, batched=batched, jit_compile=jit)
                candidates[f'prior_{jit}'] = old(inputs[0], inputs[1], tf.constant(compact, D), *inputs[3:5])
                if jit:
                    hlos.append(program.experimental_get_compiler_ir(*inputs)(stage='hlo'))
            records.append({'batched': batched, 'case': case, 'count': count, 'original': original,
                'original_source_sha256': hashes, 'candidate': candidates})
        for index, hlo in enumerate(hlos):
            (directory / f'pilot-{batched}-{case}-{index}.hlo').write_text(hlo)
        difference = ''.join(difflib.unified_diff(stable_pilot_hlo(hlos[0]).splitlines(True),
            stable_pilot_hlo(hlos[-1]).splitlines(True), fromfile='count0', tofile='count9'))
        (directory / f'pilot-{batched}-{case}.diff').write_text(difference)
    save(request, 'geometry-active-pilot-attribution.json', {'records': records,
        'prior_source_sha256': checkpoint.hashes(), 'role': 'diagnostic_only; no equivalence waiver'})


def active_pilot_fixture(dimension, batched, case, capacity=9):
    with tf.device(tf.constant(0.).device):
        # TensorFlow pins int32 resources to CPU even in a GPU device scope.
        calls = tf.Variable(0, dtype=tf.int64)
        extent = tf.Variable(-1, dtype=tf.int64)
        positions = tf.Variable(tf.zeros([max(1, 2 * capacity), dimension], D))

    def callback(points):
        cloud = points if batched else points[None]
        size = cloud.shape[0]
        extent.assign(size)
        if batched:
            calls.assign_add(1)
            positions.assign(tf.pad(cloud, [[0, max(1, 2 * capacity) - size], [0, 0]]))
        else:
            row = calls.assign_add(1) - 1
            positions.scatter_nd_update(row[None, None], points[None])
        delta = cloud - .13
        score = -delta * (tf.cast(tf.range(dimension), D) + 2.)
        value = .5 * tf.reduce_sum(delta * score, axis=1)
        if case == 'shape_sensitive':
            # Coupling to the batch average makes padded calls numerically wrong.
            average = tf.reduce_sum(cloud, axis=0) / tf.cast(max(1, size), D)
            score -= .03 * average[None] ** 3
            value -= .03 * tf.reduce_sum(cloud * average[None] ** 3, axis=1)
        if case == 'nonfinite':
            value = tf.where(cloud[:, 0] > 0., tf.constant(float('nan'), D), value)
        if case == 'exception':
            raise ValueError('deliberate callback failure')
        return (value, score) if batched else (value[0], score[0])

    def reset():
        calls.assign(0)
        extent.assign(-1)
        positions.assign(tf.zeros_like(positions))

    return callback, calls, extent, positions, reset


def compact_report(raw, rank, capacity, batched):
    count = int(raw['direction_count'])
    compact = {**raw, **{name: raw[name][:2 * count] for name in ('positions', 'values', 'scores', 'valid')}}
    return record_pilot(geometry_pilot_report(compact, rank=rank,
        requested_direction_count=capacity, batched=batched))


@pytest.mark.parametrize('dimension', [3, 5])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['gaussian', 'shape_sensitive', 'nonfinite', 'exception'])
def test_active_pilot_complete_original_records(dimension, batched, case, request):
    capacity, rank = 9, dimension - 1
    target, calls, extent, positions, reset = active_pilot_fixture(dimension, batched, case, capacity)
    cfg = LowRankSPDQuadraticGeometryConfig(rank=rank, pilot_direction_count=capacity)
    programs = {jit: make_geometry_pilot_program(target, dimension, rank, capacity,
        batched=batched, jit_compile=jit, active_rows=True) for jit in (False, True)}
    directions = np.random.default_rng(253 + dimension).normal(size=(capacity, dimension))
    observations, hlos = [], []
    for count in (0, 1, 4, capacity):
        raw_directions = directions.copy()
        raw_directions[count:] = 0.
        compact = raw_directions[:count] / np.linalg.norm(raw_directions[:count], axis=1, keepdims=True)
        padded = np.pad(compact, ((0, capacity - count), (0, 0)), constant_values=np.nan)
        inputs = (tf.cast(tf.range(dimension), D) * .01 + count * .003,
            (tf.cast(tf.range(dimension), D) * .1 + .8) * (1. + count * .01),
            tf.constant(padded, D), tf.constant(cfg.pilot_radius, D), tf.ones([dimension], D) * .3,
            tf.constant(count))
        reset()
        original, hashes = reference(target, cfg, inputs[:5], raw_directions, batched=batched)
        original_calls, original_extent = int(calls), int(extent)
        original_positions = clean(positions.read_value())
        results = {}
        for jit, program in programs.items():
            reset()
            raw = program(*inputs)
            assert bool(raw['count_valid'])
            actual = compact_report(raw, rank, capacity, batched)
            unresolved = count == 1 and case in ('gaussian', 'shape_sensitive')
            if unresolved:
                assert not bool(raw['basis_resolved']) and actual['basis'] is None
                assert actual['diagnostics']['status'] == 'pilot_eigenbasis_ill_conditioned'
                assert float(raw['basis_roundoff_indicator']) > float(raw['basis_roundoff_limit'])
                _equal_records(actual['candidates'], original['candidates'])
                _equal_records({key: value for key, value in actual['diagnostics'].items()
                    if key not in ('status', 'basis_roundoff_indicator', 'basis_roundoff_limit',
                        'basis_minimum_eigen_gap')}, original['diagnostics'])
            else:
                assert bool(raw['basis_resolved'])
                _equal_records(actual, original)
            assert (int(calls), int(extent)) == (original_calls, original_extent)
            np.testing.assert_allclose(positions, original_positions, rtol=1e-10, atol=1e-10)
            results[str(jit)] = {'result': actual, 'calls': int(calls), 'extent': int(extent)}
            if jit:
                hlos.append(program.experimental_get_compiler_ir(*inputs)(stage='hlo'))
        observations.append({'count': count, 'original': original, 'original_calls': original_calls,
            'original_extent': original_extent, 'candidate': results, 'original_source_sha256': hashes})
    assert programs[True].experimental_get_tracing_count() == 1
    assert len({stable_pilot_hlo(hlo) for hlo in hlos}) == 1
    entry = hlos[0][hlos[0].rfind('\nENTRY '):]
    assert any('s32[]' in line and 'parameter(5)' in line for line in entry.splitlines())
    save(request, f'geometry-active-pilot-{case}-{dimension}-{batched}.json', {
        'observations': observations, 'hlo_bytes': len(hlos[0].encode()),
        'trace_count': 1, 'stable_hlo': True, 'inactive_padding': 'nan',
        'scope': 'Prepared normalized pilot through shared curvature/basis, original full records; full initializer pending.'})


@pytest.mark.parametrize('dimension', [1, 3])
@pytest.mark.parametrize('batched', [False, True])
def test_active_pilot_invalid_counts_never_call_target(dimension, batched, request):
    rank = dimension - 1
    capacity = 3 if rank else 0
    target, calls, extent, _positions, reset = active_pilot_fixture(dimension, batched, 'gaussian', capacity)
    program = make_geometry_pilot_program(target, dimension, rank, capacity, batched=batched, active_rows=True)
    observations = []
    for count in (-2147483648, -1, capacity + 1, 2147483647):
        reset()
        raw = program(tf.zeros([dimension], D), tf.ones([dimension], D),
            tf.fill([capacity, dimension], tf.constant(float('nan'), D)), tf.constant(.15, D),
            tf.ones([dimension], D), tf.constant(count))
        assert not bool(raw['count_valid']) and int(calls) == 0 and int(extent) == -1
        assert not bool(raw['basis_resolved'])
        assert geometry_pilot_report(raw, rank=rank, requested_direction_count=capacity, batched=batched) == (
            None, {'status': 'pilot_active_count_invalid'}, ())
        observations.append({'count': count, 'result': raw, 'calls': int(calls)})
    if rank == 0:
        reset()
        raw = program(tf.zeros([1], D), tf.ones([1], D), tf.zeros([0, 1], D),
            tf.constant(.15, D), tf.zeros([1], D), tf.constant(0))
        assert bool(raw['count_valid']) and int(calls) == 0
        assert compact_report(raw, 0, 0, batched)['diagnostics'] == {'positive_curvature_count': 0}
    save(request, f'geometry-active-pilot-invalid-{dimension}-{batched}.json', observations)


def test_pilot_basis_guard_relative_margin(request):
    strength = tf.Variable([3., 2.], dtype=D)

    def callback(point):
        score = -point * strength
        return .5 * tf.reduce_sum(point * score), score

    programs = {(jit, active): make_geometry_pilot_program(callback, 2, 1, 4 if active else 2,
        active_rows=active, jit_compile=jit) for jit in (False, True) for active in (False, True)}
    root_eps = float(np.sqrt(np.finfo(np.float64).eps))
    # Solve eps * 2 * (2+gap) / gap == sqrt(eps) for this diagonal sketch.
    boundary = 4 * root_eps / (1 - 2 * root_eps)
    observations = []
    for gap in (1., boundary * 1.01, boundary * .99, 0.):
        strength.assign([2. + gap, 2.])
        for (jit, active), program in programs.items():
            directions = tf.eye(2, dtype=D)
            if active:
                directions = tf.concat([directions, tf.fill([2, 2], tf.constant(float('nan'), D))], 0)
            raw = program(tf.zeros([2], D), tf.ones([2], D), directions, tf.constant(1., D),
                tf.zeros([2], D), *((tf.constant(2),) if active else ()))
            resolved = gap > boundary
            assert bool(raw['basis_resolved']) == resolved
            report = compact_report(raw, 1, 4, False) if active else record_pilot(
                geometry_pilot_report(raw, rank=1, requested_direction_count=2, batched=False))
            if not resolved:
                assert report['basis'] is None and report['diagnostics']['status'] == 'pilot_eigenbasis_ill_conditioned'
            observations.append({'gap': gap, 'jit': jit, 'active': active, 'raw': raw, 'report': report})
    save(request, 'geometry-active-pilot-guard-margin.json', {'boundary': boundary,
        'observations': observations, 'role': 'engineering eigenvector-sensitivity guard; no certified error bound'})
