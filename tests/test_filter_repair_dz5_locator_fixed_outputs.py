"""Diagnostic locator replay with a frozen, call-indexed callback transcript.

This callback is not a value/gradient function of its input. It deliberately
removes target rounding as a variable and cannot qualify an optimizer or a
scientific target. NumPy only reads and compares archived diagnostic arrays.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from bayesfilter.inference.batched_local_center import (
    BatchedLocalCenterConfig,
    BatchedLocalCenterResult,
)
from bayesfilter.inference.batched_local_center_tf import BatchedLocalCenterProgram
from bayesfilter.inference.tensor_npz_archive import write_tensor_npz
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
D = tf.float64


def first_difference(left, right, *, strict):
    common = min(len(left), len(right))
    within = np.isclose(left[:common], right[:common], atol=1e-10, rtol=1e-10) if strict else left[:common].view(np.uint64) == right[:common].view(np.uint64)
    changed = np.flatnonzero(~np.all(within, axis=1))
    return int(changed[0]) if len(changed) else None


def test_original_and_current_with_identical_callback_outputs(request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    source = RAW / 'run-04584'
    manifest = json.loads((source / 'dz5-snapshot-import.json').read_text())
    path = source / 'locator-callbacks.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['callbacks_sha256']
    assert manifest['passed'] and manifest['callback_rows'] == 474
    with np.load(path, allow_pickle=False) as archive:
        bank = {key: archive[key].copy() for key in archive.files}
    oracle_path = RAW / 'run-04560/dz5-score-oracle.json'
    oracle = json.loads(oracle_path.read_text())
    assert oracle['snapshot_manifest_sha256'] == manifest['snapshot_manifest_sha256']
    scale = tf.constant(oracle['prior_scale'], D)
    initial = tf.constant(bank['positions'][:1], D)
    values, scores, valid = (tf.constant(bank[key]) for key in ('values', 'scores', 'valid'))
    settings = manifest['settings']
    capacity = settings['max_optimizer_callback_batches_per_round'] + 10
    checkpoint = FrozenCheckpoint('d6a568384', 'locator_original_fixed_outputs')
    original = checkpoint.load('bayesfilter.inference.batched_local_center')
    records, positions = {}, {}
    count = len(bank['values'])
    for arm in ('original', 'candidate'):
        calls = tf.Variable(0, dtype=tf.int64, trainable=False)
        visited = tf.Variable(tf.zeros([capacity, 23], D), trainable=False)

        def fixed_outputs(points, calls=calls, visited=visited):
            index = calls.assign_add(1) - 1
            slot = tf.minimum(index, tf.cast(count - 1, tf.int64))
            write = visited.scatter_nd_update(tf.reshape(tf.minimum(index, capacity - 1), [1, 1]), points)
            with tf.control_dependencies([write]):
                return (tf.reshape(tf.gather(values, slot), [1]),
                    tf.reshape(tf.gather(scores, slot), [1, 23]),
                    tf.reshape(tf.gather(valid, slot) & (index < count), [1]))

        started = time.monotonic()
        if arm == 'original':
            result = original.locate_batched_local_center(fixed_outputs, initial, scale,
                config=original.BatchedLocalCenterConfig(**settings))
        else:
            program = BatchedLocalCenterProgram(fixed_outputs, 1, 23, BatchedLocalCenterConfig(**settings))
            result = BatchedLocalCenterResult(**program(initial, scale),
                trace_count=program.compiled.experimental_get_tracing_count())
        used = int(calls)
        positions[arm] = visited[:min(used, capacity)].numpy()
        records[arm] = {'result': result.payload(), 'callback_rows': used,
            'transcript_exhausted': used > count, 'trace_capacity_exceeded': used > capacity,
            'elapsed_seconds': time.monotonic() - started,
            'first_exact_position_difference_from_original_CDF': first_difference(positions[arm], bank['positions'], strict=False),
            'first_bound_position_difference_from_original_CDF': first_difference(positions[arm], bank['positions'], strict=True)}
    directory = Path(request.config.getoption('xmlpath')).parent
    with (directory / 'fixed-output-locator-positions.npz').open('xb') as stream:
        write_tensor_npz(stream, {key: tf.constant(value) for key, value in positions.items()})
    result = {'schema': 'filter_dz5_locator_fixed_callback_outputs.v1',
        'role': 'call_indexed_transcript_diagnostic_not_a_target_function',
        'source_callbacks_sha256': manifest['callbacks_sha256'],
        'source_manifest_sha256': hashlib.sha256((source / 'dz5-snapshot-import.json').read_bytes()).hexdigest(),
        'original_source_hashes': checkpoint.hashes(), 'settings': settings, 'arms': records,
        'record_differences': differences(records['candidate']['result'], records['original']['result']),
        'first_exact_position_difference_between_controllers': first_difference(positions['candidate'], positions['original'], strict=False),
        'first_bound_position_difference_between_controllers': first_difference(positions['candidate'], positions['original'], strict=True),
        'nonclaims': ['Callbacks ignore input; agreement cannot qualify value/gradient correctness, optimizer convergence, target equivalence or admission.']}
    save(request, 'dz5-locator-fixed-outputs.json', result)
    assert all(not row['trace_capacity_exceeded'] for row in records.values())
    assert all(row['result']['trace_count'] == 1 for row in records.values())
