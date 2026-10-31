"""Read back actual incoming-SQMC repair evidence, retaining failed workers."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_integration_completed_evidence(request):
    required = {5172: 6, 5173: 6, 5174: 14, 5175: 14, 5180: 1, 5181: 9,
                5182: 4, 5185: 1, 5186: 10, 5187: 8, 5188: 10, 5189: 11,
                5190: 13, 5191: 20, 5193: 3, 5194: 11, 5195: 160,
                5196: 1, 5197: 2, 5200: 1, 5201: 2}
    partial = {5176: 8, 5183: 21, 5184: 9, 5192: 17, 5198: 1, 5199: 1}
    failures = {5176, 5177, 5178, 5179, 5183, 5184, 5192, 5198, 5199}
    rows = []
    manifests = {}
    for number in range(5170, 5202):
        directory = RAW/f'run-{number:05d}'
        manifest = json.loads((directory/'run.json').read_text())
        manifests[number] = manifest
        cases = ET.parse(directory/'junit.xml').findall('.//testcase')
        passed = [c for c in cases if all(c.find(tag) is None for tag in ('failure', 'error', 'skipped'))]
        assert manifest['state'] == ('failed' if number in failures else 'passed')
        if number in required:
            assert len(cases) == len(passed) == required[number]
        if number in partial:
            assert len(passed) == partial[number]
        provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        policy = provenance['gpu_memory_policy']
        assert policy['all_physical_devices_memory_growth']
        assert policy['configured_before_logical_device_initialization']
        if manifest['device'] == 'GPU':
            assert policy['physical_devices']
            assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        else:
            assert provenance['cuda_visible_devices'] == '-1'
        rows.append({'run': number, 'group': manifest['key'][1], 'state': manifest['state'],
                     'passed_cases': len(passed), 'manifest_sha256': sha(directory/'run.json'),
                     'junit_sha256': sha(directory/'junit.xml'), 'provenance': provenance})

    # Bind each changed numerical owner to its final executed scope. Earlier
    # failed workers and partial passes never override these final witnesses.
    owners = {
        'bayesfilter/highdim/sqmc_campaign_tf.py': 5181,
        'bayesfilter/highdim/sqmc_tf.py': 5193,
        'bayesfilter/ops/halton_tf.py': 5201,
        'bayesfilter/ops/halton_primes.py': 5175,
        'bayesfilter/ops/stateless_random_tf.py': 5175,
        'bayesfilter/highdim/sqmc_full_lgssm_tf.py': 5186,
        'bayesfilter/highdim/sqmc_lgssm_tf.py': 5187,
        'bayesfilter/highdim/sqmc_ksc_tf.py': 5190,
        'bayesfilter/highdim/sqmc_ksc_gaussian_sum_reference_tf.py': 5190,
        'bayesfilter/highdim/ledh_canonical_score_tf.py': 5189,
        'bayesfilter/highdim/ledh_canonical_models_tf.py': 5190,
        'bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py': 5187,
        'bayesfilter/highdim/sqmc_campaign_tuning.py': 5187,
        'docs/benchmarks/run_sqmc_expanded_repair.py': 5186,
    }
    for path, number in owners.items():
        assert sha(ROOT/path) == manifests[number]['source_sha256'][path], (path, number)

    input_cases = 0
    for number in (5174, 5175):
        reports = sorted((RAW/f'run-{number:05d}').glob('sqmc-inputs-*.json'))
        assert len(reports) == 12
        for path in reports:
            report = json.loads(path.read_text())
            assert report['baseline'] == '023e10610'
            tolerance = 3e-6 if report['dtype'] == 'float32' else 1e-12
            for row in report['rows']:
                for actual, expected in zip(row['current'], row['original'], strict=True):
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)
                np.testing.assert_array_equal(row['current'][2], row['original'][2])
                input_cases += 1
    assert input_cases == 96

    for number in (5176, 5181):
        reports = sorted((RAW/f'run-{number:05d}').glob('sqmc-endpoint-*.json'))
        assert len(reports) == 8
        for path in reports:
            report = json.loads(path.read_text())
            assert report['jit_compile'] and report['trace_count'] == 1
            for row in report['rows']:
                assert row['original'][2] is True and row['current'][2] is True
                for a, b in zip(row['current'][:2], row['original'][:2], strict=True):
                    np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
    for number, pattern in ((5184, 'sqmc-trace-4-True.json'), (5185, 'sqmc-trace-2-False.json'),
                            (5186, 'sqmc-trace-*.json')):
        reports = sorted((RAW/f'run-{number:05d}').glob(pattern))
        assert len(reports) == (2 if number == 5186 else 1)
        for path in reports:
            report = json.loads(path.read_text())
            assert report['jit_compile'] and report['trace_count'] == 1
            for row in report['records']:
                assert row['original']['valid'] == row['current']['valid']
                assert row['original']['nomination_pass'] == row['current']['nomination_pass']
    boundary_cases = 0
    for number in (5197, 5201):
        for dtype in ('float32', 'float64'):
            report = json.loads((RAW/f'run-{number:05d}'/f'halton-boundaries-{dtype}.json').read_text())
            assert len(report['records']) == 7
            for row in report['records']:
                assert [c['context'] for c in row['comparisons']] == ['ordinary', 'xla']
                for comparison in row['comparisons']:
                    assert max(comparison['maximum_absolute_errors']) <= (2e-7 if dtype == 'float32' else 1e-14)
                    boundary_cases += 1
    attribution = json.loads((RAW/'run-05200/halton-gpu-radix-attribution.json').read_text())
    assert attribution['radix19_squared'] == {'cpu': 361., 'gpu': 361.00000000000006, 'xla': 361.}
    assert attribution['incorrect_gpu_digit_indices'] == [[360, 7, 2], [721, 7, 2]]
    assert attribution['cpu_and_xla_digits_equal_exact_integer_reference']
    assert attribution['maximum_repair_cpu_difference'] <= 1e-14
    assert attribution['maximum_enclosed_difference'] == 0.
    output = Path(request.config.getoption('xmlpath')).parent/'sqmc-integration-readback.json'
    output.write_text(json.dumps({'schema': 'filter_repair.sqmc_integration_readback.v1',
        'evidence': rows, 'final_source_witnesses': owners,
        'frozen_live_input_cases': input_cases, 'live_endpoint_parameter_seed_cases': 32,
        'halton_boundary_context_cases': boundary_cases, 'gpu_reference_attribution': attribution,
        'nonclaims': ['No memory/performance, canonical LEDH or whole-repository admission.',
                      'Failed workers remain failed; final scope witnesses qualify only declared call paths.']}, indent=2)+'\n')
