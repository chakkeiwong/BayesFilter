"""Source applicability and finite SVD owner-lifetime evidence readback."""

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.filter_repair_cost_provenance import validate_cost_device

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
DIRECT_SOURCES = ('bayesfilter/inference/block_score_geometry_tf.py',
    'bayesfilter/inference/quadratic_geometry.py', 'bayesfilter/inference/score_curvature_tf.py',
    'bayesfilter/inference/sequential_score_fit_tf.py', 'bayesfilter/ops/qr_lstsq_tf.py',
    'bayesfilter/ops/accurate_svd_tf.py', 'bayesfilter/ops/symmetric_matrix_tf.py')


def test_remaining_svd_resource_evidence(request):
    applicability = []
    for path in DIRECT_SOURCES:
        old = subprocess.check_output(['git', 'show', f'ea94aac96:{path}'], cwd=ROOT)
        current = (ROOT/path).read_bytes()
        assert old == current
        applicability.append({'path': path, 'sha256': hashlib.sha256(current).hexdigest()})
    rows = {}
    failed = []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 5227:
            continue
        run = json.loads(path.read_text())
        if run['key'][1] not in ('resource_acceptance_svd_reuse_cpu', 'resource_acceptance_svd_reuse_gpu'):
            continue
        if run['state'] != 'passed':
            failed.append({'run': path.parent.name, 'state': run['state']})
            continue
        report = json.loads((path.parent/'remaining-svd-resource-lifetime.json').read_text())
        provenance = next(json.loads(line) for line in (path.parent/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        validate_cost_device(run, provenance, report['device_observation'])
        for source, digest in run['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
        assert report['replacement_count'] == 12
        for lifetime in report['lifetime']:
            assert lifetime['calls'] == 256 and lifetime['exact_replay'] and lifetime['trace_count'] == 1
            assert all(lifetime['released'].values())
            assert lifetime['late_128_call_rss_growth_bytes'] <= 16*1024**2
        assert all(all(c['checks'].values()) and c['released'] for c in report['capacities'])
        assert all(p == {'trace_count': 1, 'captures': 0} for p in report['primitives'].values())
        exit_record = run['process_exit_observation']
        assert not exit_record['proc_entry_present'] and not exit_record['errors']
        assert not any(p['pid'] == exit_record['pid'] for p in exit_record['gpu_processes'])
        rows[run['device']] = {'run': path.parent.name, 'report': report, 'process_exit': exit_record}
    assert set(rows) == {'CPU', 'GPU'}
    receipt = ROOT/'docs/plans/artifacts/filter-gradient-repair-20260917/remaining-svd-cost-receipt-03760.json'
    accepted_cost = json.loads(receipt.read_text())
    assert accepted_cost['accepted_cost_cohort'] and accepted_cost['worker_count'] == 48
    for name, digest in accepted_cost['artifacts'].items():
        assert hashlib.sha256((receipt.parent/name).read_bytes()).hexdigest() == digest
    record = {'schema': 'filter_repair.remaining_svd_resource_readback.v1',
        'prior_cost_checkpoint': 'ea94aac96', 'direct_numerical_sources_unchanged': applicability,
        'prior_cost_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'lifetimes': rows, 'failed_workers_preserved': failed,
        'nonclaims': ['The original inaccurate XLA arm stays excluded from timing ratios.',
                     'Angle/subspace diagnostic costs are not part of the old fit-only composition.',
                     'No native eviction or unrestricted capacity claim.']}
    (Path(request.config.getoption('xmlpath')).parent/'remaining-svd-resource-readback.json').write_text(
        json.dumps(record, indent=2)+'\n')
