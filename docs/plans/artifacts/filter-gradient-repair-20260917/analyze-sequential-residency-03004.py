"""Saved post-run diagnostic analysis; source/device cohorts remain explicit."""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, '/tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts')
from analyze_filter_repair_sequential_public_costs import same_records


def load(number, expected_device, observer=False):
    directory = ROOT / f'run-{number:05d}'
    run = json.loads((directory / 'run.json').read_text())
    filename = 'sequential-residency-observer.json' if observer else 'sequential-public-residency.json'
    path = directory / filename
    report = json.loads(path.read_text())
    assert run['device'] == expected_device and run['state'] == 'passed'
    assert report['gpu'] == (expected_device == 'GPU')
    tests = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
    assert len(tests) == 1 and not any(case.findall(tag) for case in tests for tag in ('error', 'failure', 'skipped'))
    provenance, = [json.loads(line) for line in (directory / 'process.log').read_text().splitlines()
        if line.startswith('{"tensorflow_version":')]
    policy = provenance['gpu_memory_policy']
    assert policy['mode'] == 'memory_growth' and policy['all_physical_devices_memory_growth']
    assert policy['configured_before_logical_device_initialization']
    assert run['environment']['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
    visible = run['environment']['CUDA_VISIBLE_DEVICES']
    assert visible == provenance['cuda_visible_devices']
    observed = report['gpu_process_observation']
    assert not observed['errors']
    if expected_device == 'CPU':
        assert visible == '-1' and not observed['enabled'] and not policy['physical_devices']
        assert provenance['trust_basis'] == 'explicit_cpu_reference'
    else:
        assert visible == run['gpu_uuid'] == observed['uuid'] and observed['enabled']
        assert run['environment']['BAYESFILTER_TEST_DEVICE_SCOPE'] == provenance['bayesfilter_test_device_scope'] == 'visible'
        assert len(policy['physical_devices']) == 1 and policy['physical_devices'][0]['memory_growth'] is True
        assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert len(observed['samples']) >= 2
    same_records(report['result'], report['original'])
    if not observer:
        same_records(report['changed_result'], report['original_changed'])
        assert report['reuse_calls'] == 1200 and report['trace_count'] == 1
        assert all(report['python_released'].values())
    else:
        assert report['calls_between_snapshots'] == 0
    stages = report['stages']
    rss = {key: value['rollup']['Rss'] / 2**20 for key, value in stages.items()}
    pss = {key: value['rollup']['Pss'] / 2**20 for key, value in stages.items()}
    executable_maps = {key: value.get('anonymous_executable', {}).get('mappings', 0)
        for key, value in report['mappings'].items()}
    categories = {key: {category: sum(item['Rss'] for name, item in value.items()
        if (name.strip() == category if category in ('[heap]', 'anonymous_other', 'anonymous_executable')
            else name.strip() not in ('[heap]', 'anonymous_other', 'anonymous_executable'))) / 2**20
        for category in ('[heap]', 'anonymous_other', 'anonymous_executable', 'other_files')}
        for key, value in report['mappings'].items()}
    foreign = sorted({item['pid'] for sample in observed['samples'] for item in sample['processes']
        if item['pid'] != observed['pid']})
    first, last = list(rss)[0], list(rss)[-1]
    row = {'run': number, 'device': expected_device, 'uuid': run.get('gpu_uuid'),
        'dimension': report.get('dimension'), 'primed': report.get('minimal_xla_prewarm'),
        'rss_mib': rss, 'pss_mib': pss, 'mapping_rss_categories_mib': categories,
        'executable_maps': executable_maps, 'foreign_compute_pids_observed': foreign,
        'allocator': {key: value['gpu'] for key, value in stages.items()},
        'python_released': report.get('python_released'), 'seconds': report.get('seconds'),
        'observer_growth_mib': rss[last] - rss[first] if observer else None,
        'reuse_growth_mib': None if observer else rss['reuse_1200'] - rss['cold'],
        'minimal_startup_growth_mib': rss.get('minimal_xla', rss[first]) - rss[first],
        'compile_after_trace_growth_mib': None if observer else rss['cold'] - rss['traced'],
        'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest()}
    return row, run


records, cohorts = [], {}
for device, numbers in (('CPU', range(2989, 2994)), ('GPU', range(3000, 3005))):
    for number in numbers:
        row, run = load(number, device, observer=number == numbers.stop - 1)
        records.append(row)
        cohorts.setdefault(device, run['source_sha256'])
        assert cohorts[device] == run['source_sha256'], 'Within-device source drift'
assert len({row['uuid'] for row in records if row['device'] == 'GPU'}) == 1
# Only the routing repair and its negative regression separate the cohorts.
changed = [path for path in set(cohorts['CPU']) | set(cohorts['GPU'])
    if cohorts['CPU'].get(path) != cohorts['GPU'].get(path)]
assert set(changed) == {'scripts/run_filter_repair_campaign.py', 'tests/test_filter_repair_campaign.py'}
report = {'schema': 'filter_sequential_residency_analysis.v1', 'records': records,
    'CPU_cohort': list(range(2989,2994)), 'GPU_cohort': list(range(3000,3005)),
    'cohort_changes': changed, 'accidental_duplicate_CPU_references': list(range(2994,2999)),
    'numerical_authority': '3582b4ac', 'actual_reuse_calls': 8 * 1200,
    'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'nonclaims': ['Fresh processes and sampled mappings do not establish exact peaks or leak freedom.',
        'Python collection does not establish native executable eviction.',
        'Shared-device residency results are not clean cost evidence or a performance ranking.',
        'Separate CPU/GPU source cohorts are explicit; no hidden harness mismatch.']}
output = ROOT / 'sequential-residency-analysis-03004.json'
with output.open('x') as out:
    json.dump(report, out, indent=2, allow_nan=False)
    out.write('\n')
print(json.dumps([{key: row[key] for key in ('run','device','dimension','primed',
    'observer_growth_mib','reuse_growth_mib','minimal_startup_growth_mib','compile_after_trace_growth_mib',
    'python_released','foreign_compute_pids_observed')} for row in records], indent=2))
