"""Renew actual-consumer evidence with current sources; retain r1 diagnostics."""

import hashlib
import json
from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_initializer_admission as admission
from tests import test_filter_repair_dz5_initializer_consumer as consumer
from tests import test_filter_repair_dz5_initializer_target as target
from tests.test_filter_repair_geometry_control import save

SNAPSHOT = target.SNAPSHOT.with_name('dz5-initializer-adapter-20260928-r2')
ADMISSION = SNAPSHOT.parent / 'dz5-initializer-adapter-target-admission-20260928-r2.json'


@pytest.fixture(autouse=True)
def renewed_sources(monkeypatch):
    # Only these renewal tests select r2. Historical trajectory diagnostics
    # continue to consume their original frozen sources and admission.
    monkeypatch.setattr(target, 'SNAPSHOT', SNAPSHOT)
    monkeypatch.setattr(consumer, 'SNAPSHOT', SNAPSHOT)
    monkeypatch.setattr(consumer, 'ADMISSION', ADMISSION)


@pytest.mark.parametrize('batch', [1, 4, 46, 68])
def test_current_initializer_target(request, monkeypatch, batch):
    target.test_current_initializer_target(request, monkeypatch, batch)


def test_current_initializer_score_oracle(request, monkeypatch):
    target.test_current_initializer_score_oracle(request, monkeypatch)


@pytest.mark.parametrize('accepted', [False, True])
def test_declared_initializer_case(request, accepted):
    consumer.test_declared_initializer_case(request, accepted)


@pytest.mark.parametrize('workers', [2])
def test_actual_initializer_supervisor_lifetime(request, workers):
    consumer.test_actual_initializer_supervisor_lifetime(request, workers)


def test_renewed_target_admission():
    saved = json.loads(ADMISSION.read_text())
    numbers = tuple(saved['fresh_runs'].values())
    rebuilt = admission.gate.build(SNAPSHOT.parent, SNAPSHOT, min(numbers), max(numbers))
    assert rebuilt == saved
    assert saved['snapshot_manifest_sha256'] == hashlib.sha256(
        (SNAPSHOT / 'manifest.json').read_bytes()).hexdigest()


def test_complete_renewed_lifetime_evidence(request):
    manifest_hash = hashlib.sha256((SNAPSHOT / 'manifest.json').read_bytes()).hexdigest()
    found = {}
    for path in sorted(SNAPSHOT.parent.glob('run-*/dz5-initializer-lifetime.json')):
        first = path.parent / 'initializer-worker-0/dz5-snapshot-import.json'
        report = json.loads(first.read_text())
        if report['snapshot_manifest_sha256'] != manifest_hash:
            continue
        run = json.loads((path.parent / 'run.json').read_text())
        assert run['state'] == 'passed' and run['exit_code'] == 0
        device = run['device']
        assert device not in found, 'Duplicate successful lifetime requires explicit disposition'
        lifetime = json.loads(path.read_text())
        assert lifetime['initializer_accepted'] == [True, True]
        assert lifetime['all_children_reaped']
        assert all(not row['timed_out'] and row['returncode'] == 0 for row in lifetime['receipts'])
        records = []
        for index in range(2):
            directory = path.parent / f'initializer-worker-{index}'
            child = json.loads((directory / 'dz5-snapshot-import.json').read_text())
            assert child['snapshot_manifest_sha256'] == manifest_hash
            assert child['initializer_accepted']
            assert child['trace_count'] == 1 and not child['host_callbacks']
            assert child['memory_after_python_release']['live_observed_owners'] == 0
            assert hashlib.sha256((directory / 'dz5-initializer.hlo.txt').read_bytes()).hexdigest() == child['hlo_sha256']
            payload = json.loads((directory / 'initialize.json').read_text())
            for artifact, digest in payload['artifacts'].items():
                actual = directory / Path(artifact).name
                assert hashlib.sha256(actual.read_bytes()).hexdigest() == digest
            records.append(payload)
        parent = lifetime['parent_memory']
        growth = parent[-1]['rss_bytes'] - parent[0]['rss_bytes']
        assert growth < 256 * 1024**2
        found[device] = {'run': path.parent.name, 'parent_RSS_growth_bytes': growth,
            'parent_memory': parent, 'records_exactly_equal': records[0] == records[1],
            'child_initializer_seconds': [json.loads((path.parent / f'initializer-worker-{index}' /
                'dz5-snapshot-import.json').read_text())['initializer_seconds'] for index in range(2)]}
    save(request, 'dz5-renewed-lifetime-readback.json', {'schema': 'filter_dz5_renewed_lifetime_readback.v1',
        'snapshot_manifest_sha256': manifest_hash, 'devices': found,
        'nonclaims': ['Process-exit containment does not imply native in-process eviction or general memory bounds.']})
    assert set(found) == {'CPU', 'GPU'}
