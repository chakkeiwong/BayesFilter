"""Read saved optional-pfor disposition; do not promote remaining F14 debt."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
PATHS = ('bayesfilter/highdim/ledh_canonical_batch_fused_tf.py',
         'bayesfilter/highdim/ledh_canonical_score_tf.py',
         'bayesfilter/highdim/ledh_canonical_reset_score_tf.py',
         'bayesfilter/highdim/ledh_unified_reset_tf.py',
         'bayesfilter/highdim/ledh_unified_correction_tf.py',
         'docs/benchmarks/ledh_k_batch_parity_and_timing.py',
         'docs/benchmarks/ledh_execution_mode_matrix.py')


def test_saved_optional_route_and_remaining_debt(request):
    failure = _load(4707, 'run.json')
    assert failure['state'] == 'failed' and failure['test_evidence']['failure'] == 2
    assert failure['test_evidence']['tests'] == 16
    errors = [case.find('failure').text for case in ET.parse(RAW / 'run-04707/junit.xml').iter('testcase')
              if case.find('failure') is not None]
    assert all("missing 1 required keyword-only argument: 'substeps'" in error for error in errors)
    reports = []
    baseline_source = subprocess.check_output(['git', 'show',
        '020d794be:bayesfilter/highdim/ledh_canonical_batch_fused_tf.py'], cwd=ROOT)
    for number, count in ((4708, 16), (4709, 1)):
        run = _load(number, 'run.json')
        assert run['state'] == 'passed'
        assert run['test_evidence'] == {'passed': True, 'tests': count, 'failure': 0, 'error': 0, 'skipped': 0}
        for relative in PATHS:
            assert run['source_sha256'][relative] == hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        record = _load(number, 'sequential-score.json')
        assert record['baseline'] == '020d794be'
        assert record['baseline_sha256'] == hashlib.sha256(baseline_source).hexdigest()
        assert record['trace_count'] == 1 and record['jit_compile'] and record['pfor_executed'] is False
        assert len(record['records']) == 2
        assert all(len(row['five_point']) == 4 for row in record['records'])
        hlo = (RAW / f'run-{number:05d}' / 'sequential-score.hlo.txt').read_bytes()
        assert hashlib.sha256(hlo).hexdigest() == record['hlo_sha256']
        provenance = _provenance(number)
        assert provenance['tf32_enabled'] is True
        assert provenance['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
        policy = provenance['gpu_memory_policy']
        assert policy['all_physical_devices_memory_growth'] and policy['configured_before_logical_device_initialization']
        if run['device'] == 'CPU':
            assert provenance['cuda_visible_devices'] == '-1'
        else:
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
            assert len(policy['physical_devices']) == 1
        reports.append({'run': number, 'tests': count, 'device': run['device']})
    for name in ('ledh_k_batch_parity_and_timing.py', 'ledh_execution_mode_matrix.py'):
        row = _load(4708, f'retired-{name}.json')
        assert row['returncode'] == 0 and row['framework_import_blocked']
    policy = json.loads((ROOT / 'scripts/filter_gradient_runtime_policy.json').read_text())
    assert 'bayesfilter/highdim/ledh_canonical_batch_fused_tf.py' in policy['sources']
    assert policy['sources']['bayesfilter/highdim/ledh_canonical_batch_fused_tf.py'] is None
    assert not [row for row in policy['exceptions'] if row['path'] == 'bayesfilter/highdim/ledh_canonical_batch_fused_tf.py']
    ledger = json.loads((ROOT / 'docs/plans/filter_gradient_repair_ledger_20260917.json').read_text())
    finding = next(row for row in ledger['findings'] if row['id'] == 'F14')
    assert finding['status'] == 'open'
    assert len(finding['additional_unclassified_pfor_sites_20260929']) == 5
    path = Path(request.config.getoption('xmlpath')).parent / 'pfor-disposition-readback.json'
    path.write_text(json.dumps({'runs': reports, 'optional_branch_repaired': True,
        'full_F14_closed': False, 'whole_master_complete': False,
        'remaining_sites': finding['additional_unclassified_pfor_sites_20260929']}, indent=2) + '\n')
