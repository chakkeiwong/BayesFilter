"""Matched fresh-process endpoint cost screen; no statistical ranking."""

import gc
import statistics
import time

import pytest
import tensorflow as tf

from tests.test_filter_repair_gaussian_binding_cost import memory
from tests.test_filter_repair_score_inputs import (
    BASELINE,
    case,
    hashes,
    numerical_record,
    reference,
    runtime_setup,
    save,
)


@pytest.mark.parametrize('arm', ['before', 'after'])
def test_endpoint_cost(arm, request, monkeypatch):
    from bayesfilter.score_study import adapters
    original, digest = reference('adapters')
    module = original if arm == 'before' else adapters
    monkeypatch.setattr(module, 'configure_runtime', runtime_setup)
    row, context, _ = case('gaussian', 'twist')
    device = context['study']['settings']['device']
    if device == 'GPU':
        tf.config.experimental.reset_memory_stats('GPU:0')
    before = memory()
    start = time.perf_counter()
    value = module.evaluate_gaussian(row, context)
    cold = time.perf_counter()-start
    after_cold = memory()
    for _ in range(2):
        assert numerical_record(module.evaluate_gaussian(row, context)) == numerical_record(value)
    warm = []
    for _ in range(30):
        start = time.perf_counter()
        replay = module.evaluate_gaussian(row, context)
        warm.append(time.perf_counter()-start)
        assert numerical_record(replay) == numerical_record(value)
    after_warm = memory()
    gc.collect()
    save(request, 'score-inputs-cost.json', {'baseline': BASELINE, 'arm': arm, 'device': device,
        'source_sha256': hashes(), 'baseline_adapter_sha256': digest, 'row': row,
        'settings': context['study']['settings'], 'seed': context['study']['seed'],
        'scope': 'Complete Gaussian twist endpoint including data/input/oracle/reporting; preconfigured runtime, import/initialization excluded',
        'cold_seconds': cold, 'warm_seconds': warm, 'warm_median_seconds': statistics.median(warm),
        'memory': {'before': before, 'after_cold': after_cold, 'after_warm': after_warm,
                   'after_python_collection': memory()},
        'numerical_result': numerical_record(value), 'runtime': value['runtime'], 'exact_replay': True,
        'nonclaims': ['One process per arm is descriptive; no statistical ranking or terminal memory acceptance.',
                      'Python collection does not prove native executable eviction.']})
