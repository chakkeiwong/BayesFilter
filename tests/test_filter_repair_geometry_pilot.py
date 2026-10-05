"""Complete original pilot records and callback ordering on frozen directions."""

import dataclasses
import gc
import re
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import quadratic_geometry as geometry
from bayesfilter.inference.quadratic_geometry_pilot_report import geometry_pilot_report
from bayesfilter.inference.quadratic_geometry_pilot_tf import (
    make_geometry_cloud_program,
    make_geometry_pilot_program,
)
from tests.test_filter_repair_geometry_control import clean, save, source, stable_hlo
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def fixture(dimension, case, *, batched, counter=True):
    count = 0 if dimension == 1 or case == 'empty' else 9
    rank = dimension - 1
    raw = np.random.default_rng(135 + dimension).normal(size=(count, dimension))
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    cfg = geometry.LowRankSPDQuadraticGeometryConfig(rank=max(1, rank), pilot_direction_count=max(1, count))
    center = tf.cast(tf.range(dimension), D) * .01
    scale = tf.cast(tf.range(dimension), D) * .1 + .8
    args = (center, scale, tf.constant(normalized, D), tf.constant(.15, D), tf.ones([dimension], D) * .3)
    calls, recorded = None, None
    if counter:
        with tf.device(center.device):
            calls = tf.Variable(0, dtype=tf.int64)
            recorded = tf.Variable(tf.zeros([max(1, 2 * count), dimension], D))

    def callback(points):
        cloud = points if batched else points[None]
        if calls is not None:
            if batched:
                calls.assign_add(1)
                recorded.assign(tf.pad(cloud, [[0, max(1, 2 * count) - 2 * count], [0, 0]]))
            else:
                index = calls.assign_add(1) - 1
                recorded.scatter_nd_update(index[None, None], points[None])
        delta = cloud - .13
        score = -delta * (tf.cast(tf.range(dimension), D) + 2.)
        value = .5 * tf.reduce_sum(delta * score, axis=1)
        if case == 'nonquadratic':
            value -= .05 * tf.reduce_sum(delta ** 4, axis=1)
            score -= .2 * delta ** 3
        if case == 'negative':
            value, score = -value, -score
        if case == 'flat':
            value, score = tf.zeros_like(value), tf.zeros_like(score)
        if case == 'nan_value':
            value = tf.where(cloud[:, 0] > 0., tf.constant(float('nan'), D), value)
        if case == 'nan_score':
            score = tf.where(cloud[:, :1] > 0., tf.constant(float('nan'), D), score)
        if case == 'exception':
            raise ValueError('deliberate pilot callback error')
        return (value, score) if batched else (value[0], score[0])

    return callback, cfg, args, raw, calls, recorded


def record_pilot(result):
    basis, report, candidates = result
    return clean({'basis': basis, 'diagnostics': report,
        'candidates': [dataclasses.asdict(candidate) for candidate in candidates]})


def reference(target, cfg, args, raw, *, batched, revision='3582b4ac'):
    checkpoint, module = source(revision)
    result = module._pilot_q_basis(target,
        batched_value_and_score_fn=target if batched else None,
        center=args[0].numpy(), scale=args[1].numpy(), rank=args[0].shape[0] - 1,
        cfg=module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg)),
        rng=SimpleNamespace(normal=lambda **kw: raw.copy()), center_value=0.,
        center_score_z=args[4].numpy(), start_index=1)
    return record_pilot(result), checkpoint.hashes()


@pytest.mark.parametrize('dimension', [1, 3, 5])
@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('case', ['gaussian', 'nonquadratic', 'negative', 'flat', 'nan_value', 'nan_score', 'empty', 'exception'])
def test_original_pilot_records(dimension, batched, case, request):
    target, cfg, args, raw_directions, calls, positions = fixture(dimension, case, batched=batched)
    records, counters, recorded_positions, hashes = {}, {}, {}, {}
    for revision in ('3582b4ac', '3f3f07ee'):
        calls.assign(0)
        positions.assign(tf.zeros_like(positions))
        try:
            records[revision], hashes[revision] = reference(target, cfg, args, raw_directions, batched=batched, revision=revision)
        except Exception as error:  # Checkpoint errors are attribution, never original authority.
            if revision == '3582b4ac':
                raise
            records[revision] = {'error': type(error).__name__, 'message': str(error)}
            hashes[revision] = source(revision)[0].hashes()
        counters[revision] = int(calls)
        recorded_positions[revision] = clean(positions.read_value())
    for jit in (False, True):
        calls.assign(0)
        positions.assign(tf.zeros_like(positions))
        try:
            program = make_geometry_pilot_program(target, dimension, dimension - 1,
                args[2].shape[0], batched=batched, jit_compile=jit)
            raw = program(*args)
            records[str(jit)] = record_pilot(geometry_pilot_report(raw, rank=dimension - 1,
                requested_direction_count=cfg.pilot_direction_count, batched=batched))
        except Exception as error:  # noqa: BLE001 - retain failure before numerical assertions.
            records[str(jit)] = {'error': type(error).__name__, 'message': str(error)}
        counters[str(jit)] = int(calls)
        recorded_positions[str(jit)] = clean(positions.read_value())
    save(request, f'geometry-pilot-{case}-{dimension}-{batched}.json', {
        'records': records, 'callback_counts': counters, 'callback_positions': recorded_positions,
        'original_source_sha256': hashes, 'raw_directions': raw_directions, 'normalized_directions': args[2]})
    for jit in (False, True):
        _equal_records(records[str(jit)], records['3582b4ac'])
        assert counters[str(jit)] == counters['3582b4ac']
        np.testing.assert_allclose(recorded_positions[str(jit)], recorded_positions['3582b4ac'], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('batched', [False, True])
def test_pilot_changes_are_runtime_inputs(batched, request):
    callback, cfg, args, raw, _, _ = fixture(3, 'nonquadratic', batched=batched, counter=False)
    changed = (args[0] + .02, args[1] * 1.1, -args[2], args[3] * .9, args[4] * .8)
    program = make_geometry_pilot_program(callback, 3, 2, 9, batched=batched)
    records = []
    for inputs, directions in ((args, raw), (changed, -raw)):
        expected, _ = reference(callback, dataclasses.replace(cfg, pilot_radius=float(inputs[3])), inputs, directions, batched=batched)
        result = program(*inputs)
        actual = record_pilot(geometry_pilot_report(result, rank=2, requested_direction_count=9, batched=batched))
        _equal_records(actual, expected)
        records.append(actual)
    assert program.experimental_get_tracing_count() == 1
    hlo = program.experimental_get_compiler_ir(*args)(stage='hlo')
    assert stable_hlo(hlo) == stable_hlo(program.experimental_get_compiler_ir(*changed)(stage='hlo'))
    assert len(re.findall(r'\bparameter\((\d+)\)', hlo[hlo.rfind('\nENTRY '):])) == 5
    save(request, f'geometry-pilot-operands-{batched}.json', {'records': records, 'runtime_operands': 5, 'trace_count': 1})


def scaled_callback(callback, strength):
    def evaluate(point):
        value, score = callback(point)
        return strength * value, strength * score
    return evaluate


@pytest.mark.parametrize('batched', [False, True])
def test_pilot_target_resources_and_graph_release(batched, request):
    base, cfg, args, raw, _, _ = fixture(4, 'nonquadratic', batched=batched, counter=False)
    strength = tf.Variable(1., dtype=D)
    target = scaled_callback(base, strength)
    program = make_geometry_pilot_program(target, 4, 3, 9, batched=batched)
    records = []
    for factor in (1., 2.):
        strength.assign(factor)
        expected, _ = reference(target, cfg, args, raw, batched=batched)
        actual = record_pilot(geometry_pilot_report(program(*args), rank=3,
            requested_direction_count=9, batched=batched))
        _equal_records(actual, expected)
        records.append(actual)
    assert program.experimental_get_tracing_count() == 1
    graph = program.get_concrete_function().graph
    watched = {name: weakref.ref(value) for name, value in (
        ('target', target), ('strength', strength), ('graph', graph))}
    del target, strength, graph, program
    gc.collect()
    released = {name: ref() is None for name, ref in watched.items()}
    save(request, f'geometry-pilot-lifetime-{batched}.json', {'records': records, 'released': released})
    assert all(released.values())


@pytest.mark.parametrize('fault', ['arity', 'value_shape', 'score_shape', 'dtype', 'exception'])
def test_batched_cloud_failure_matches_original(fault, request):
    def callback(points):
        values, scores = -.5 * tf.reduce_sum(points ** 2, axis=1), -points
        if fault == 'arity':
            return values, scores, values
        if fault == 'value_shape':
            return values[:, None], scores
        if fault == 'score_shape':
            return values, scores[:, :1]
        if fault == 'dtype':
            return tf.cast(values, tf.float32), scores
        if fault == 'exception':
            raise ValueError('deliberate batch error')
        return values, scores

    points = tf.reshape(tf.range(18, dtype=D), [6, 3]) / 10.
    _, original = source('3582b4ac')
    expected = clean(original._evaluate_values_scores(callback, points.numpy(), batched_value_and_score_fn=callback))
    results = {}
    for jit in (False, True):
        raw = make_geometry_cloud_program(callback, 3, 6, batched=True, jit_compile=jit)(points)
        results[str(jit)] = clean((raw['values'], raw['scores']))
        _equal_records(results[str(jit)], expected)
    save(request, f'geometry-cloud-failure-{fault}.json', {'original': expected, 'candidate': results})
