"""Fail-closed provenance checks for complete public block cost comparisons."""

import json
from types import SimpleNamespace

import pytest

from tests.test_filter_repair_campaign import load
from tests.test_filter_repair_sequential_cost_analysis import (
    matrix as sequential_matrix,
)


def matrix(root):
    last = sequential_matrix(root)
    for number in range(1, last + 1):
        directory = root / f'run-{number:05d}'
        source = directory / 'sequential-public-memory.json'
        record = json.loads(source.read_text())
        record.update(schema='filter_block_public_cost.v1', mechanism_baseline='aee3ad043',
            samples=record['samples'][:3])
        for key in ('result', 'changed_result', 'original_result', 'original_changed_result'):
            record[key]['public_summary'] = {'completed': True, 'accepted_block_count': 2}
        (directory / 'block-public-memory.json').write_text(json.dumps(record))
    return last


def test_complete_block_cost_matrix(tmp_path):
    last = matrix(tmp_path)
    result = load('analyze_filter_repair_block_public_costs').analyze(tmp_path, 1, last, ['CPU', 'GPU'])
    assert len(result['comparisons']) == 4
    assert all(len(row['arms']['xla']['repeats']) == 3 for row in result['comparisons'])


@pytest.mark.parametrize('fault', ['source', 'input', 'baseline', 'record', 'blocks', 'completed', 'repeat', 'samples'])
def test_block_cost_matrix_rejects_invalid_evidence(tmp_path, fault):
    last = matrix(tmp_path)
    directory = tmp_path / f'run-{last:05d}'
    path = directory / 'block-public-memory.json'
    result = json.loads(path.read_text())
    run_path = directory / 'run.json'
    run = json.loads(run_path.read_text())
    if fault == 'source':
        run['source_sha256']['different'] = 'changed'
    elif fault == 'input':
        result['input_sha256'] = ['different']
    elif fault == 'baseline':
        result['mechanism_baseline'] = 'wrong'
    elif fault == 'record':
        result['result']['count'] += 1
    elif fault in ('blocks', 'completed'):
        for key in ('result', 'changed_result', 'original_result', 'original_changed_result'):
            result[key]['public_summary']['accepted_block_count' if fault == 'blocks' else 'completed'] = 1 if fault == 'blocks' else False
    elif fault == 'repeat':
        run['key'][6] = 1
    else:
        result['samples'] = result['samples'][:2]
    path.write_text(json.dumps(result))
    run_path.write_text(json.dumps(run))
    with pytest.raises((AssertionError, ValueError)):
        load('analyze_filter_repair_block_public_costs').analyze(tmp_path, 1, last, ['CPU', 'GPU'])


def test_block_costs_reject_incomplete_cohort_and_shared_preflight(tmp_path, monkeypatch):
    last = matrix(tmp_path)
    with pytest.raises(AssertionError):
        load('analyze_filter_repair_block_public_costs').analyze(tmp_path, 1, last-1, ['CPU', 'GPU'])
    runner = load('run_filter_repair_campaign')
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    for group in runner.TEST_BATCHES['block_public_cost_gpu']:
        args = SimpleNamespace(action='test', device='GPU', group=group, gpu_uuid='GPU-device-2',
            gpu_preflight=[{'performance_preflight_uncontended': False}] * 2)
        with pytest.raises(RuntimeError, match='declined before launch'):
            runner.require_unshared_cost_preflight(args)
