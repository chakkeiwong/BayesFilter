"""Complete pinned batched-locator records across reusable runtime operands."""

import dataclasses
import gc
import json
import math
import weakref
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.batched_local_center import (
    BatchedLocalCenterConfig,
    BatchedLocalCenterResult,
)
from bayesfilter.inference.batched_local_center_tf import BatchedLocalCenterProgram
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64


def _diagnostic_json(value):
    """Preserve nonfinite callback positions as explicit diagnostic labels."""
    if isinstance(value, dict):
        return {key: _diagnostic_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_diagnostic_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': 'nan' if math.isnan(value) else ('+inf' if value > 0 else '-inf')}
    return value


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('case', ['quadratic', 'nonquadratic', 'flat', 'invalid_rows'])
def test_batched_locator_reuses_operand_starts(case, batch_size, request):
    counter = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([1200, 2], D))
    precision = tf.constant([[2., .4], [.4, 1.3]], D)
    mode = tf.constant([.2, -.15], D)

    def target(points):
        index = counter.assign_add(batch_size) - batch_size
        positions.scatter_nd_update((index + tf.range(batch_size, dtype=tf.int64))[:, None], points)
        delta = points - mode
        scores = -tf.linalg.matmul(delta, precision, transpose_b=True)
        values = .5 * tf.reduce_sum(delta * scores, axis=1)
        valid = tf.ones([batch_size], tf.bool)
        if case == 'nonquadratic':
            values -= .01 * tf.reduce_sum(delta ** 4, axis=1)
            scores -= .04 * delta ** 3
        if case == 'flat':
            values, scores = tf.zeros_like(values), tf.zeros_like(scores)
        if case == 'invalid_rows':
            valid = points[:, 0] > 0.
            values = tf.where(valid, values, tf.constant(1e200, D))
        return values, scores, valid

    cfg = BatchedLocalCenterConfig(max_iterations=8, max_optimizer_callback_batches_per_round=40)
    initial = tf.constant([[.8, -.7], [-.8, .7], [1., 1.2]][:batch_size], D)
    scale = tf.constant([.5, 2.], D)
    original = FrozenCheckpoint('d6a568384', 'batched_locator_reuse_original')
    public = original.load('bayesfilter.inference.batched_local_center')
    before_cfg = public.BatchedLocalCenterConfig(**dataclasses.asdict(cfg))
    owner = BatchedLocalCenterProgram(target, batch_size, 2, cfg)
    records, hlos = [], []
    for offset, multiplier in ((0., 1.), (.01, 1.05), (0., 1.)):
        starts, scales = initial + offset, scale * multiplier
        counter.assign(0)
        expected = public.locate_batched_local_center(target, starts, scales, config=before_cfg).payload()
        expected_count = int(counter)
        expected_positions = positions[:expected_count].numpy().tolist()
        counter.assign(0)
        raw = owner(starts, scales)
        result = BatchedLocalCenterResult(**raw, trace_count=owner.compiled.experimental_get_tracing_count()).payload()
        actual_count = int(counter)
        actual_positions = positions[:actual_count].numpy().tolist()
        records.append({'result': result, 'original': expected, 'actual_positions': actual_positions,
            'expected_positions': expected_positions, 'count': actual_count, 'expected_count': expected_count})
        hlos.append(owner.compiled.experimental_get_compiler_ir(starts, scales)(stage='hlo'))
    count = owner.compiled.experimental_get_tracing_count()
    concrete = owner.compiled.get_concrete_function()
    refs = {'owner': weakref.ref(owner), 'graph': weakref.ref(concrete.graph), 'callback': weakref.ref(target)}
    del owner, concrete, target
    gc.collect()
    released = {name: ref() is None for name, ref in refs.items()}
    report = {'schema': 'filter_batched_locator_reuse.v1', 'case': case, 'batch_size': batch_size,
        'records': records, 'trace_count': count, 'hlo_unchanged': len({stable_hlo(hlo) for hlo in hlos}) == 1,
        'python_released': released, 'original_source_sha256': original.hashes()}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / f'batched-locator-{case}-{batch_size}.json').open('x') as output:
        json.dump(_diagnostic_json(report), output, indent=2, allow_nan=False)
        output.write('\n')
    for row in records:
        _equal_records(row['result'], row['original'])
        _equal_records(row['actual_positions'], row['expected_positions'])
        assert row['count'] == row['expected_count'] == row['result']['physical_target_rows']
    assert count == 1 and report['hlo_unchanged']
    assert all(released.values()), released


def test_batched_locator_resets_inside_enclosing_xla_recurrence(request):
    """An invalid attempt must not contaminate the following valid attempt."""
    batch_size, dimension = 3, 2
    counter = tf.Variable(0, dtype=tf.int64)
    positions = tf.Variable(tf.zeros([1200, dimension], D))

    def target(points):
        index = counter.assign_add(batch_size) - batch_size
        positions.scatter_nd_update((index + tf.range(batch_size, dtype=tf.int64))[:, None], points)
        delta = points - tf.constant([.2, -.15], D)
        score = -delta * tf.constant([2., 1.3], D)
        return .5 * tf.reduce_sum(delta * score, axis=1), score, points[:, 0] > 0.

    cfg = BatchedLocalCenterConfig(max_iterations=8, max_optimizer_callback_batches_per_round=40)
    owner = BatchedLocalCenterProgram(target, batch_size, dimension, cfg)
    original = FrozenCheckpoint('d6a568384', 'batched_locator_enclosing_original')
    public = original.load('bayesfilter.inference.batched_local_center')
    before_cfg = public.BatchedLocalCenterConfig(**dataclasses.asdict(cfg))
    starts = tf.constant([[[-.8, .7], [-.7, .5], [-1., 1.2]],
        [[.8, -.7], [.7, -.5], [1., 1.2]]], D)
    scales = tf.constant([.5, 2.], D)
    signature = [tf.TensorSpec(starts.shape, D), tf.TensorSpec(scales.shape, D)]

    def enclosing(program):
        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def outer(points, scale):
            first = program(points[0], scale)
            rows = tf.nest.map_structure(lambda value: tf.stack([value, tf.zeros_like(value)]), first)

            def step(index, rows):
                value = program(points[index], scale)
                updated = tf.nest.map_structure(
                    lambda output, row: tf.tensor_scatter_nd_update(output, [[index]], [row]), rows, value)
                return index + 1, updated

            return tf.while_loop(lambda index, _: index < 2, step,
                (tf.constant(1), rows), maximum_iterations=1, parallel_iterations=1)[1]
        return outer

    outer = enclosing(owner.compiled)

    records, hlos = [], []
    for shift in (0., .01, 0.):
        inputs = starts + shift
        counter.assign(0)
        expected = [public.locate_batched_local_center(target, inputs[index], scales,
            config=before_cfg).payload() for index in range(2)]
        expected_count = int(counter)
        expected_positions = positions[:expected_count].numpy().tolist()
        counter.assign(0)
        output = outer(inputs, scales)
        actual = [BatchedLocalCenterResult(**tf.nest.map_structure(lambda rows, index=index: rows[index], output),
            trace_count=owner.compiled.experimental_get_tracing_count()).payload() for index in range(2)]
        actual_count = int(counter)
        actual_positions = positions[:actual_count].numpy().tolist()
        records.append({'result': actual, 'original': expected,
            'actual_positions': actual_positions, 'expected_positions': expected_positions,
            'count': actual_count, 'expected_count': expected_count})
        hlos.append(outer.experimental_get_compiler_ir(inputs, scales)(stage='hlo'))
    concrete = outer.get_concrete_function()
    references = {'outer': weakref.ref(outer), 'graph': weakref.ref(concrete.graph),
        'owner': weakref.ref(owner), 'callback': weakref.ref(target)}
    outer_traces, inner_traces = outer.experimental_get_tracing_count(), owner.compiled.experimental_get_tracing_count()
    del outer, concrete, owner, target
    gc.collect()
    released = {name: reference() is None for name, reference in references.items()}
    report = {'schema': 'filter_batched_locator_enclosing.v1', 'records': records,
        'outer_trace_count': outer_traces, 'inner_trace_count': inner_traces,
        'hlo_unchanged': len({stable_hlo(hlo) for hlo in hlos}) == 1,
        'python_released': released, 'original_source_sha256': original.hashes()}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'batched-locator-enclosing.json').open('x') as output_file:
        json.dump(_diagnostic_json(report), output_file, indent=2, allow_nan=False)
        output_file.write('\n')
    for row in records:
        _equal_records(row['result'], row['original'])
        _equal_records(row['actual_positions'], row['expected_positions'])
        assert row['count'] == row['expected_count']
        assert row['result'][0]['status'] == 'initial_target_invalid'
        assert row['result'][1]['accepted']
    assert outer_traces == inner_traces == 1 and report['hlo_unchanged']
    assert all(released.values()), released
