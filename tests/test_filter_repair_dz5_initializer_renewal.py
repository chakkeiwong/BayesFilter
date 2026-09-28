"""Renew actual-consumer evidence with current sources; retain r1 diagnostics."""

import hashlib
import json

import pytest

from tests import test_filter_repair_dz5_initializer_admission as admission
from tests import test_filter_repair_dz5_initializer_consumer as consumer
from tests import test_filter_repair_dz5_initializer_target as target

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
