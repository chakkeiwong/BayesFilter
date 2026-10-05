"""Read back the bounded core qualification; never numerical or scientific admission."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(number, filename):
    return json.loads((RAW/f'run-{number:05d}'/filename).read_text())


def test_core_reference_failure_attribution():
    for number, enabled, expected in ((5152, True, (1153, 760)), (5153, False, (0, 0))):
        manifest = read(number, 'run.json')
        report = read(number, 'contract-e-rounding.json')
        assert manifest['state'] == 'passed' and manifest['device'] == 'GPU'
        assert report['tf32_enabled'] == enabled and report['jit_compile'] is False
        assert report['results']['original'] == report['results']['current']
        row = report['results']['current']
        assert (row['per_batch_ulps'], row['aggregate_ulps']) == expected
        assert row['original_cpu_ulp_gate_passed'] is (not enabled)
        for path, sha in report['current_sources'].items():
            assert digest(ROOT/path) == sha == manifest['source_sha256'][path]
    report = read(5155, 'contract-e-default-factory.json')
    assert report['jit_compile'] and report['trace_count'] == 1
    assert len(report['records']) == 2 and 'GPU:0' in report['device']
    assert all(max(row['maximum_absolute_errors'].values()) <= 1e-10 for row in report['records'])
    assert all(row['actual']['valid_chart'] == row['expected']['valid_chart'] == [True, True]
               for row in report['records'])


def test_current_core_qualification_evidence(request):
    manifests = {}
    for number in range(5139, 5200):
        path = RAW/f'run-{number:05d}'/'run.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['key'][1].startswith('core_execution_'):
                manifests[number] = manifest
    scopes = ('contract_e', 'tt_value', 'tt_maps', 'tt_adjoint', 'tt_actual',
              'tt_scalar', 'apf', 'preparation', 'signatures', 'sgqf', 'streaming_mask')
    required = [f'core_execution_{name}_cpu' for name in scopes]
    required += [f'core_execution_{name}_gpu' for name in scopes if name not in ('contract_e', 'tt_scalar')]
    required += ['core_execution_contract_e_fp32_reference_gpu', 'core_execution_contract_e_factory_gpu',
                 'core_execution_tt_scalar_operands_gpu', 'core_execution_pool_cpu', 'core_execution_imports_cpu']
    evidence = []
    for group in required:
        matches = [(number, m) for number, m in manifests.items()
                   if m['key'][1] == group and m['state'] == 'passed']
        assert matches, group
        number, manifest = matches[-1]
        directory = RAW/f'run-{number:05d}'
        suites = ET.parse(directory/'junit.xml').getroot().findall('testsuite')
        assert suites and sum(int(s.attrib['tests']) for s in suites) > 0
        assert all(int(s.attrib[k]) == 0 for s in suites for k in ('failures', 'errors', 'skipped'))
        provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        policy = provenance['gpu_memory_policy']
        assert policy['all_physical_devices_memory_growth']
        assert policy['configured_before_logical_device_initialization']
        expected_device = 'GPU' if group.endswith('_gpu') else 'CPU'
        assert manifest['device'] == expected_device
        if expected_device == 'GPU':
            assert policy['physical_devices']
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        # Only changed numerical sources invalidate this source-frozen cohort.
        # Harness fixes and unrelated documentation are preserved separately.
        changed = [path for path, sha in manifest['source_sha256'].items()
                   if path.startswith(('bayesfilter/', 'experiments/dpf_implementation/tf_tfp/'))
                   and (ROOT/path).is_file() and digest(ROOT/path) != sha]
        assert not changed, (group, changed)
        evidence.append({'group': group, 'run': number,
                         'tests': sum(int(s.attrib['tests']) for s in suites),
                         'manifest_sha256': digest(directory/'run.json'),
                         'junit_sha256': digest(directory/'junit.xml'),
                         'log_sha256': digest(directory/'process.log'),
                         'device_provenance': provenance})
    # These failed workers retain partial passes. Recovery adds only the failed
    # comparisons; it does not pretend that the original workers passed.
    for number, expected_passed in ((5150, 2), (5160, 7)):
        assert manifests[number]['state'] == 'failed'
        cases = ET.parse(RAW/f'run-{number:05d}'/'junit.xml').findall('.//testcase')
        passed = [c.attrib['name'] for c in cases if all(c.find(tag) is None
                  for tag in ('failure', 'error', 'skipped'))]
        assert len(passed) == expected_passed
        evidence.append({'run': number, 'state': 'failed_preserved', 'passing_cases': passed,
                         'manifest_sha256': digest(RAW/f'run-{number:05d}'/'run.json')})
    output = Path(request.config.getoption('xmlpath')).parent/'core-execution-readback.json'
    output.write_text(json.dumps({'schema': 'filter_repair.core_execution_readback.v1',
        'evidence': evidence, 'numerical_sources_unchanged': True,
        'nonclaims': ['Focused finite-fixture execution is not repository-wide compliance.',
                      'Resource acceptance, remote integration and deferred families remain separate.']}, indent=2)+'\n')
