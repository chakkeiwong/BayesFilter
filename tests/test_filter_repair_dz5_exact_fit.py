"""Diagnostic exact-byte CPU/GPU replay; numerical failures remain recorded."""

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import dense_validated_fit_tf as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_dz5_initializer_fit_localization import differences
from tests.test_filter_repair_geometry_control import clean, save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
D = tf.float64


@pytest.mark.parametrize('arm', ['before', 'after'])
def test_exact_saved_fit_inputs(arm, request):
    tf.config.experimental.enable_tensor_float_32_execution(False)
    cpu_record = json.loads((RAW / 'run-04591/dz5-saved-fit-localization.json').read_text())
    assert cpu_record['arm'] == 'xla'
    assert json.loads((RAW / 'run-04591/run.json').read_text())['state'] == 'passed'
    path = RAW / 'run-04591/frozen-fit-inputs.npz'
    keys = ('center', 'center_score', 'offsets', 'scores')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in keys}
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
    assert hashes == cpu_record['operand_sha256']
    operands = tuple(tf.constant(arrays[key], D) for key in keys)
    assert all(hashlib.sha256(value.numpy().tobytes()).hexdigest() == hashes[key]
        for key, value in zip(keys, operands, strict=True))
    checkpoint = FrozenCheckpoint('5d398a45b', 'exact_saved_fit_before_anchor')
    module = checkpoint.load('bayesfilter.inference.dense_validated_fit_tf') if arm == 'before' else current
    recipe_path = RAW / 'dz5-initializer-accepted-inputs-20260928-r1/recipe.json'
    recipe = json.loads(recipe_path.read_text())
    assert hashlib.sha256(recipe_path.read_bytes()).hexdigest() == cpu_record['inputs']['source_sha256'][str(recipe_path)]
    config = recipe['initializer']
    thresholds = module.fixed.FixedCenterCurvatureThresholds(**recipe['curvature_thresholds'])
    dimensions = (23, config['replicate_count'], config['training_rows_per_replicate'],
        config['selection_rows_per_replicate'], config['audit_rows'])
    settings = {key: config[key] for key in ('factor_max', 'dense_eigenvalue_floor',
        'max_condition_number', 'shrinkage_weights', 'structured_target_family')}
    gpu = os.environ['CUDA_VISIBLE_DEVICES'] != '-1'
    if gpu:
        tf.config.experimental.reset_memory_stats('GPU:0')
    started = time.monotonic()
    owner = module.make_dense_validated_fit_program(*dimensions,
        thresholds=thresholds, jit_compile=True, **settings)
    raw = owner(*operands)
    tf.nest.map_structure(lambda value: value.numpy(), raw)
    result = {'schema': 'filter_dz5_exact_saved_fit.v1', 'arm': arm,
        'role': 'diagnostic_outcome_not_numerical_admission', 'device': 'GPU' if gpu else 'CPU',
        'operand_sha256': hashes, 'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'recipe_sha256': hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
        'baseline_sources': checkpoint.hashes() if arm == 'before' else {},
        'elapsed_seconds': time.monotonic() - started, 'raw': clean(raw),
        'fit_numerically_usable': bool(raw['usable']), 'fit_error_code': int(raw['fit_error_code']),
        'trace_count': owner.experimental_get_tracing_count(),
        'input_signature': str(owner.input_signature), 'jit_compile': True,
        'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'allocator': tf.config.experimental.get_memory_info('GPU:0') if gpu else None,
        'output_device': raw['fit']['status'].device,
        'nonclaims': ['A passing artifact check does not turn a rejected fit into a pass or qualify whole-consumer equivalence.']}
    save(request, 'dz5-exact-saved-fit-raw.json', result)
    assert bool(raw['fit_ran']) and int(raw['validation']['error_code']) == 0
    assert result['trace_count'] == 1
    assert raw['fit']['status'].device.endswith('device:GPU:0' if gpu else 'device:CPU:0')
    if int(raw['fit_error_code']) == 0:
        payload = module.fixed._fixed_center_result_from_native(raw['fit'], operands[0], operands[1],
            dimension=23, replicates=config['replicate_count'],
            training_rows=config['training_rows_per_replicate'], selection_rows=config['selection_rows_per_replicate'],
            audit_rows=config['audit_rows'], thresholds=thresholds, factor_max=config['factor_max'],
            weights=tuple(config['shrinkage_weights']), structured_target_family=config['structured_target_family'],
            lineage=cpu_record['result']['diagnostics']['lineage'], jit_compile=True).payload()
        result['result'] = clean(payload)
        result['differences_from_CPU04591'] = differences(result['result'], cpu_record['result'])
    save(request, 'dz5-exact-saved-fit.json', result)
