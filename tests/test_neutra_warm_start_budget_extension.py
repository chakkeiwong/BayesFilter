"""CPU-only budget handover fixtures; no framework initialization or research run."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'repair_budget_extension', ROOT / 'scripts/apply_neutra_warm_start_budget_extension.py')
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


def prepare(root, status):
    before = {'cpu_core_seconds': 122400., 'gpu_process_seconds': 64800.}
    after = {'cpu_core_seconds': 237600., 'gpu_process_seconds': 122400.}
    directory = root / extension.ALLOCATION
    directory.mkdir()
    extension.write(directory / 'allocation.json', {
        'previous_ceilings': before, 'new_ceilings': after,
        'authorization': 'owner additional 48 compute-hours; CPU 32, GPU 16'})
    extension.write(root / 'config.json', {**before, 'seeds': [11, 23, 37],
                                          'job_wall_seconds': {'closure_fit': 600}})
    state = {'status': status, 'attempts': [{'cpu_core_seconds': 17, 'gpu_process_seconds': 9}]}
    extension.write(root / 'state.json', state)
    extension.write(root / 'repair-master-state.json', {'status': status})
    return state


def test_funding_is_applied_once_and_does_not_change_costs_or_science(tmp_path):
    prepare(tmp_path, 'ready')
    state_bytes = (tmp_path / 'state.json').read_bytes()
    extension.run(tmp_path)
    first = extension.load(tmp_path / 'config.json')
    extension.run(tmp_path)
    assert extension.load(tmp_path / 'config.json') == first
    assert (tmp_path / 'state.json').read_bytes() == state_bytes
    assert first['cpu_core_seconds'] == 237600 and first['gpu_process_seconds'] == 122400
    assert first['seeds'] == [11, 23, 37] and first['job_wall_seconds'] == {'closure_fit': 600}


@pytest.mark.parametrize('status,expected', [
    ('budget_exhausted', 1), ('declared_matrix_resolved', 0),
    ('failed', 0), ('attempt_cap', 0), ('ready', 0)])
def test_only_budget_exhaustion_resumes(tmp_path, status, expected):
    prepare(tmp_path, status)
    calls = []
    extension.run(tmp_path, resume=calls.append)
    assert len(calls) == expected
    if calls:
        assert calls == [tmp_path]


def test_different_allocation_is_preserved_for_reconciliation(tmp_path):
    prepare(tmp_path, 'budget_exhausted')
    config = extension.load(tmp_path / 'config.json')
    config['gpu_process_seconds'] += 1
    extension.write(tmp_path / 'config.json', config)
    with pytest.raises(ValueError, match='ceilings changed'):
        extension.run(tmp_path, resume=lambda root: pytest.fail('unexpected restart'))
    assert extension.load(tmp_path / 'config.json') == config


def test_completed_controller_does_not_resume_from_stale_budget_status(tmp_path):
    prepare(tmp_path, 'budget_exhausted')
    extension.write(tmp_path / 'repair-master-state.json', {'status': 'declared_matrix_resolved'})
    extension.run(tmp_path, resume=lambda root: pytest.fail('completed matrix restarted'))


def test_preserved_budget_exception_resumes(tmp_path):
    prepare(tmp_path, 'budget_exhausted')
    extension.write(tmp_path / 'repair-master-state.json', {
        'status': 'stopped_with_preserved_evidence',
        'decisions': [{'decision': 'budget_exhausted'}]})
    calls = []
    extension.run(tmp_path, resume=calls.append)
    assert calls == [tmp_path]
