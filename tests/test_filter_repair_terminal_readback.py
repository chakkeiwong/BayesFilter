"""Final scoped evidence applicability; no numerical implementation imports."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
EVIDENCE = ROOT / 'docs/plans/filter_gradient_terminal_evidence_20261001.json'
RECORD = json.loads(EVIDENCE.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('row', RECORD['rows'], ids=lambda row: row['finding'])
def test_current_endpoint_source_applicability(row):
    """Changed dependencies must have an explicitly qualified renewal."""
    original = json.loads((RAW / f"run-{row['source_boundary_run']:05d}/run.json").read_text())
    assert original['state'] == 'passed'
    closure = row['conservative_import_closure']
    assert closure and set(row['roots']) <= set(closure)
    changed = set()
    for path, expected in closure.items():
        assert digest(ROOT / path) == expected, (row['finding'], path)
        saved = original['source_sha256'].get(path)
        if saved != expected:
            changed.add(path)
            renewal = row['source_renewals'][path]
            manifest = json.loads((RAW / f"run-{renewal['run']:05d}/run.json").read_text())
            assert manifest['state'] == 'passed' and renewal['reason']
            assert manifest['source_sha256'][path] == expected
    assert changed == set(row['source_renewals'])
    assert row['terminal_reports'] and row['endpoints_and_roles']
    assert all(path in RECORD['evidence_sha256'] for path in row['terminal_reports'])


def test_selected_terminal_evidence_integrity(request):
    counts = {}
    for relative, expected in RECORD['evidence_sha256'].items():
        path = RAW / relative
        assert digest(path) == expected, relative
        if path.name == 'run.json':
            manifest = json.loads(path.read_text())
            assert manifest['state'] == 'passed'
            assert manifest['device'] == 'CPU'
            assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
        elif path.suffix == '.xml':
            cases = ET.parse(path).findall('.//testcase')
            assert cases and all(case.find(tag) is None for case in cases
                                 for tag in ('failure', 'error', 'skipped'))
            counts[path.parent.name] = len(cases)
        else:
            assert isinstance(json.loads(path.read_text()), dict)
    assert digest(RAW / 'run-05537/audit.json.gz') == RECORD['audit_sha256']
    assert digest(ROOT / 'scripts/filter_gradient_runtime_policy.json') == RECORD['policy_sha256']
    assert {row['finding'] for row in RECORD['rows']} == {f'F{i:02d}' for i in range(1, 21)}
    assert RECORD['preserved_failures'] and RECORD['nonclaims']
    output = Path(request.config.getoption('xmlpath')).parent
    (output / 'terminal-scope-readback.json').write_text(json.dumps({
        'schema': 'filter_repair.terminal_scope_readback.v1',
        'evidence_index_sha256': digest(EVIDENCE),
        'finding_count': len(RECORD['rows']),
        'current_source_count': len({p for row in RECORD['rows']
                                     for p in row['conservative_import_closure']}),
        'verified_terminal_tests': counts,
        'deferred': ['adaptive_iAPF', 'KDM'],
        'preserved_failures': RECORD['preserved_failures'],
        'nonclaims': RECORD['nonclaims'],
        'main_merge_complete': False,
    }, indent=2) + '\n')


@pytest.mark.parametrize('row', RECORD['changed_unguarded_dispositions'],
                         ids=lambda row: row['path'])
def test_reviewed_wrapper_scope_has_not_changed(row):
    assert digest(ROOT / row['path']) == row['sha256']
    assert row['disposition'] and 'functions_reviewed' in row
    if row['numpy_imports']:
        assert row['path'] == 'docs/benchmarks/analyze_sqmc_4route_comparison.py'
        assert 'historical diagnostic' in row['disposition']
