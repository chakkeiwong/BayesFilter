"""Inspect Python lifetime evidence separately from native executable residency."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
sources = None
workers = []
for number in range(2944, 2968):
    directory = root / f'run-{number:05d}'
    path = directory / 'run.json'
    run = json.loads(path.read_text())
    assert run['state'] == 'passed' and run['test_evidence']['passed']
    if sources is None:
        sources = run['source_sha256']
    assert run['source_sha256'] == sources, (number, 'source mismatch')
    workers.append({'run': number, 'group': run['key'][1], 'device': run['device'],
        'gpu_uuid': run.get('gpu_uuid'), 'tests': run['test_evidence']['tests'],
        'elapsed_seconds': run['elapsed_seconds'], 'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    for report in directory.glob('sequential-owner-*.json'):
        value = json.loads(report.read_text())
        final = value['stages']['old_handle_released']
        assert not any(final['root_refs'].values()) and not any(final['dependency_refs'].values())
        assert not any(value['successor_refs_alive_after_clear'])
        assert value['standalone_cache_sizes_before'] == value['standalone_cache_sizes_after']
        assert value['retained_result'] == value['successor_result'] == value['result']
        assert value['retained_trace_count'] == value['successor_trace_count'] == 1
        workers[-1]['lifetime'] = {'case': value['case'], 'released_dependency_programs': len(final['dependency_refs']),
            'evidence_sha256': hashlib.sha256(report.read_bytes()).hexdigest()}
assert len(workers) == 24
assert sum(w['tests'] for w in workers[:12]) == 17
assert sum(w['tests'] for w in workers[12:23]) == 15
assert workers[-1]['tests'] == 128
assert sum('lifetime' in w for w in workers) == 6
report = {'schema': 'filter_sequential_ownership_qualification.v1', 'workers': workers,
    'source_sha256': sources, 'charged_seconds': sum(w['elapsed_seconds'] for w in workers),
    'nonclaims': ['Python collection does not prove native TensorFlow executable eviction.',
        'Remaining callback owners, ordered blocks and terminal cost/integration gates remain separate.']}
with (root / 'sequential-ownership-qualification-02967.json').open('x') as out:
    json.dump(report, out, indent=2)
    out.write('\n')
print(json.dumps({'workers': len(workers), 'charged_seconds': report['charged_seconds'],
    'CPU_checks_including_scope': 17, 'GPU_checks': 15, 'policy_checks': workers[-1]['tests']}))
