"""Saved-evidence gate tests; mutations must prevent initializer admission."""

import copy
import importlib.util
from pathlib import Path

import pytest

from tests.test_filter_repair_dz5_initializer_target import SNAPSHOT

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('initializer_admission_gate',
    ROOT / 'scripts/prepare_filter_repair_dz5_initializer_admission.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.fixture
def cohort():
    rows = []
    manifest_hash = gate.digest(SNAPSHOT / 'manifest.json')
    for path in sorted(SNAPSHOT.parent.glob('run-*/run.json')):
        if int(path.parent.name[4:]) < 4559:
            continue
        run = gate.read(path)
        if run['key'][1] in gate.EXPECTED:
            report_path = path.parent / 'dz5-snapshot-import.json'
            if not report_path.is_file():
                continue
            if gate.read(report_path)['snapshot_manifest_sha256'] != manifest_hash:
                continue
            rows.append((int(path.parent.name[4:]), run['key'][1]))
    assert {name for _, name in rows} == gate.EXPECTED, 'Execute the fresh target cohort first'
    return min(n for n, _ in rows), max(n for n, _ in rows)


def test_complete_fresh_evidence_issues_only_initializer_target_admission(cohort):
    result = gate.build(SNAPSHOT.parent, SNAPSHOT, *cohort)
    assert result['passed'] and set(result['fresh_runs']) == gate.EXPECTED
    assert result['role'] == 'fresh_target_only_isolated_initializer_regression'
    assert not any(result[key] for key in ('old_admission_reused', 'training_authorized',
        'hmc_authorized', 'retained_authorized'))


@pytest.mark.parametrize('mutation', ['missing_group', 'failed_worker', 'stale_snapshot',
    'non_xla', 'host_callback', 'source_mismatch', 'unconfigured_growth',
    'invalid_row', 'failed_oracle', 'altered_stencil_values', 'wrong_batch',
    'nonfinite_value', 'replay_mismatch'])
def test_broken_saved_evidence_cannot_issue_admission(cohort, monkeypatch, mutation):
    original = gate.read

    def altered(path):
        result = copy.deepcopy(original(path))
        name = Path(path).name
        if mutation == 'missing_group' and name == 'run.json' and result['key'][1] == 'dz5_initializer_target_4_cpu':
            result['key'][1] = 'not_a_target_gate'
        elif mutation == 'failed_worker' and name == 'run.json':
            result['state'] = 'failed'
        elif name == 'dz5-snapshot-import.json':
            if mutation == 'stale_snapshot':
                result['snapshot_manifest_sha256'] = '0' * 64
            elif mutation == 'non_xla':
                result['jit_compile'] = False
            elif mutation == 'host_callback':
                result['host_callbacks'] = ['PyFunc']
            elif mutation == 'source_mismatch':
                next(iter(result['post_target_loaded_modules'].values()))['sha256'] = '0' * 64
            elif mutation == 'unconfigured_growth':
                result['gpu_memory_policy']['configured_before_logical_device_initialization'] = False
            elif mutation == 'invalid_row' and 'invalid_rows' in result:
                result['invalid_rows']['valid'][1] = True
            elif mutation == 'wrong_batch':
                result['batch'] += 1
            elif mutation == 'nonfinite_value' and 'comparisons' in result:
                result['comparisons'][0]['value'][0] = float('nan')
            elif mutation == 'replay_mismatch' and 'comparisons' in result:
                # Keep the recorded graph comparator equal so the exact replay
                # check, rather than the numerical comparison, detects this.
                result['comparisons'][-1]['value'][0] += 1.
                result['comparisons'][-1]['reference_value'][0] += 1.
        elif mutation == 'failed_oracle' and name == 'dz5-score-oracle.json':
            result['scores'][0][0] += 1.
        elif mutation == 'altered_stencil_values' and name == 'dz5-score-oracle.json':
            result['values'][1] += 1.
        return result

    monkeypatch.setattr(gate, 'read', altered)
    with pytest.raises(AssertionError):
        gate.build(SNAPSHOT.parent, SNAPSHOT, *cohort)
