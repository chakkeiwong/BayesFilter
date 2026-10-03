"""Budget/retry/provenance checks for bounded campaign bookkeeping memory."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_filter_repair_campaign import load

ROOT = Path(__file__).resolve().parents[1]


def test_streamed_summary_keeps_exact_budget_and_source_retry_identity(tmp_path, monkeypatch):
    driver = load('run_filter_repair_campaign')
    monkeypatch.setattr(driver, 'OUTPUT', tmp_path)
    key = ['test', 'policy', 'after', 'covariance', 'on', 1, 0, 'CPU']
    sources = {'one.py': 'original', 'two.py': 'present'}
    versions = [sources, {'one.py': 'changed', 'two.py': 'present'}, {'one.py': 'original'}, sources]
    for number, source in enumerate(versions, 1):
        directory = tmp_path / f'run-{number:05d}'
        directory.mkdir()
        row = {'device': 'CPU', 'timeout_seconds': 300, 'key': key, 'source_sha256': source,
            'state': 'failed' if number == 1 else 'running'}
        if number != 4:
            row['elapsed_seconds'] = float(number)
        (directory / 'run.json').write_text(json.dumps(row))
    (tmp_path / 'supplemental-compute-reference.json').write_text(json.dumps({'device': 'CPU', 'charged_seconds': 120.}))
    rows, attempts = driver.history_summary(driver.records(), key, sources)
    assert len(rows) == 4 and attempts == 2
    assert driver.charged_seconds(rows, 'CPU') == 426.
    assert all(set(row) <= {'device', 'timeout_seconds', 'elapsed_seconds'} for row in rows)
    assert driver.latest_record()['source_sha256'] == sources
    # Parsing is lazy, and invalid metadata still fails rather than being skipped.
    iterator = driver.records()
    (tmp_path / 'run-00002/run.json').write_text('invalid')
    assert next(iterator)['source_sha256'] == sources
    with pytest.raises(json.JSONDecodeError):
        next(iterator)


def test_three_attempt_veto_survives_streaming_and_source_changes(tmp_path, monkeypatch):
    driver = load('run_filter_repair_campaign')
    monkeypatch.setattr(driver, 'OUTPUT', tmp_path)
    key = ['audit', 'policy', 'after', 'covariance', 'on', 1, 0, 'CPU']
    hashes = {'kernel.py': 'original'}
    monkeypatch.setattr(driver, 'source_hashes', lambda: hashes)
    for number in range(1, 4):
        directory = tmp_path / f'run-{number:05d}'
        directory.mkdir()
        (directory / 'run.json').write_text(json.dumps({'key': key, 'source_sha256': hashes,
            'device': 'CPU', 'elapsed_seconds': 1., 'timeout_seconds': 300, 'state': 'failed'}))
    args = argparse.Namespace(action='audit', group='policy', arm='after', fixture='covariance',
        jit='on', size=1, repeat=0, device='CPU')
    monkeypatch.setattr(driver.subprocess, 'Popen', lambda *a, **k: pytest.fail('A veto must precede launch'))
    with pytest.raises(RuntimeError, match='Three attempts consumed'):
        driver.run_job(args)
    hashes = {'kernel.py': 'repaired'}
    # The changed source is a different attempt identity; no prior is discarded.
    summary, attempts = driver.history_summary(driver.records(), key, hashes)
    assert len(summary) == 3 and attempts == 0


@pytest.mark.parametrize('arm', ['prior', 'streamed'])
def test_fresh_driver_history_memory(arm, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    output = directory / 'driver-history-memory.json'
    index = ROOT / 'docs/plans/artifacts/filter-gradient-repair-20260917/driver-history-fixture-03870.json'
    command = [sys.executable, str(ROOT / 'scripts/filter_repair_driver_memory_worker.py'),
        '--arm', arm, '--index', str(index), '--output', str(output)]
    with (directory / 'driver-history-memory.log').open('x') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            timeout=120, check=False)
    assert completed.returncode == 0, (directory / 'driver-history-memory.log').read_text()
    result = json.loads(output.read_text())
    assert result['run_count'] == 3870 and result['next_run_number'] == 3871
    assert result['exact_source_attempts'] == 1
    expected = json.loads((ROOT / 'docs/plans/artifacts/filter-gradient-repair-20260917/dz5-score-verification-03870.json').read_text())
    assert result['charged_seconds'] == expected['charged_seconds']
    assert result['rss_retained_bytes'] > 0 and result['seconds'] > 0
