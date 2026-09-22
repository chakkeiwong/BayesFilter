"""Diagnostic campaign controls: budget, repair, and candidate-veto semantics."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def driver(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'docs/benchmarks/diagnose_iapf_adaptive_replication_ladder.py'
    spec = importlib.util.spec_from_file_location('iapf_replication_test_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    module.save(tmp_path / 'budget.json', dict(remaining_cpu_seconds=162000,
        remaining_gpu_seconds=170000, phase_cpu_seconds=0, phase_gpu_seconds=0,
        reserved_cpu_seconds=0, launches=0))
    return module


def test_reserved_workers_cannot_exceed_phase_budget(driver):
    driver.PHASE_SECONDS = 100
    driver.reserve('first', 60)
    with pytest.raises(RuntimeError, match='budget'):
        driver.reserve('second', 60)
    driver.settle(dict(name='first', wall_seconds=20))
    driver.reserve('second', 60)
    budget = driver.read(driver.ROOT / 'budget.json')
    assert budget['phase_cpu_seconds'] == 20
    assert budget['reserved_cpu_seconds'] == 60
    assert budget['remaining_cpu_seconds'] == 161980
    with pytest.raises(RuntimeError, match='reservation'):
        driver.settle(dict(name='first', wall_seconds=20))


def test_timeout_retry_and_candidate_rejection_continue_all_stages(driver, monkeypatch):
    monkeypatch.setattr(driver, 'verify_prior', lambda: None)
    driver.save(driver.ROOT / 'preflight.json', dict(passed=True, current_sources={}))
    attempted = []
    candidate_failures = []
    stages = []

    def launch(name, script, arguments, cap):
        attempted.append((name, cap))
        timeout = name == 'd80_r0001-0010_attempt01'
        record = dict(name=name, exit_code=124 if timeout else 0,
                      wall_seconds=cap if timeout else 1, sources_unchanged=True, outputs={})
        driver.save(driver.ROOT / (name + '-launch.json'), record)
        return record

    def verify_batch(directory, d, first, last):
        # A valid experiment may reject every candidate; this is not an infrastructure veto.
        candidate_failures.append(directory.name)
        return [dict(status='iteration_cap')]

    def stage_summary(stage, completed):
        stages.append(stage)
        directory = driver.ROOT / f'stage{stage}'
        directory.mkdir()
        driver.save(directory / 'manifest.json', dict(stage=stage))
        assert len(completed) == stage // 10 * 5

    monkeypatch.setattr(driver, 'launch', launch)
    monkeypatch.setattr(driver, 'verify_batch', verify_batch)
    monkeypatch.setattr(driver, 'stage_summary', stage_summary)
    driver.run_ladder()
    assert stages == [100, 300, 1000]
    assert len(candidate_failures) == 500
    assert len(attempted) == 501
    assert ('d80_r0001-0010_attempt02', 1800) in attempted
    budget = driver.read(driver.ROOT / 'budget.json')
    assert budget['phase_cpu_seconds'] == 1400
    assert budget['reservations'] == {}
    assert driver.read(driver.ROOT / 'checkpoint.json')['status'] == 'complete_pending_interpretation'


def test_repeated_timeout_stops_and_preserves_charges(driver, monkeypatch):
    monkeypatch.setattr(driver, 'verify_prior', lambda: None)
    driver.save(driver.ROOT / 'preflight.json', dict(passed=True, current_sources={}))

    def launch(name, script, arguments, cap):
        record = dict(name=name, exit_code=124, wall_seconds=cap,
                      sources_unchanged=True, outputs={})
        driver.save(driver.ROOT / (name + '-launch.json'), record)
        return record

    monkeypatch.setattr(driver, 'launch', launch)
    with pytest.raises(RuntimeError, match='Invalid attempt'):
        driver.run_ladder()
    budget = driver.read(driver.ROOT / 'budget.json')
    assert budget['phase_cpu_seconds'] >= 2700
    assert budget['reservations'] == {}
    assert driver.read(driver.ROOT / 'checkpoint.json')['status'] == 'blocked'
    assert not (driver.ROOT / 'stage100').exists()
