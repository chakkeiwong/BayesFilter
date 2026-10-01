"""Source-bound readback of Gaussian binding numerical and descriptive costs."""

import hashlib
import json
import math
from pathlib import Path

from tests.test_filter_repair_gaussian_binding import ROOT, compare, hashes, save

RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
FIRST_RUN, LAST_RUN = 4883, 4896


def saved():
    found = {}
    failures = []
    for number in range(FIRST_RUN, LAST_RUN + 1):
        path = RAW / f'run-{number:05d}' / 'run.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        group = manifest['key'][1]
        assert group.startswith('gaussian_binding_')
        if group == 'gaussian_binding_readback_cpu':
            continue
        if manifest['state'] != 'passed':
            failures.append({'run': number, 'group': group, 'state': manifest['state']})
            continue
        assert group not in found, 'Duplicate passed measurement needs a disposition'
        assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == '-1' if manifest['device'] == 'CPU' else (
            manifest['environment']['CUDA_VISIBLE_DEVICES'].startswith('GPU-'))
        found[group] = (path.parent, manifest)
    return found, failures


def read(directory, manifest, name):
    path = directory / name
    record = json.loads(path.read_text())
    assert record['source_sha256'] == hashes()
    assert all(manifest['source_sha256'][source] == digest for source, digest in hashes().items())
    return record, {'run': directory.name,
        'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
        'result': name, 'result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'worker_runtime': runtime_receipt(directory, manifest)}


def runtime_receipt(directory, manifest):
    log_path = directory / 'process.log'
    records = [json.loads(line) for line in log_path.read_text().splitlines()
               if line.startswith('{') and '"gpu_memory_policy"' in line]
    assert len(records) == 1
    record = records[0]
    assert record['cuda_visible_devices'] == manifest['environment']['CUDA_VISIBLE_DEVICES']
    policy = record['gpu_memory_policy']
    assert policy['mode'] == 'memory_growth' and policy['configured_before_logical_device_initialization']
    assert policy['all_physical_devices_memory_growth'] and policy['full_device_preallocation_disabled']
    if manifest['device'] == 'GPU':
        assert policy['physical_devices'] and all(item['memory_growth'] for item in policy['physical_devices'])
        assert record['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
    else:
        assert not policy['physical_devices'] and record['trust_basis'] == 'explicit_cpu_reference'
    for source in ('scripts/filter_repair_test_worker.py', 'bayesfilter/runtime/gpu_memory_policy.py'):
        assert manifest['source_sha256'][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    return {'record': record, 'process_log_sha256': hashlib.sha256(log_path.read_bytes()).hexdigest()}


def test_gaussian_binding_saved_evidence(request):
    found, failures = saved()
    evidence, results = {}, {}
    max_numerical = {'float64': 0., 'float32': 0.}
    max_fd = 0.
    for device in ('cpu', 'gpu'):
        for dtype in ('float64', 'float32'):
            group = f'gaussian_binding_{dtype}_{device}'
            directory, manifest = found[group]
            record, evidence[group] = read(directory, manifest, f'gaussian-binding-{dtype}.json')
            assert len(record['cases']) == 4
            if device == 'gpu':
                assert record['last_output_devices'] and all('GPU:0' in item for item in record['last_output_devices'])
            for case in record['cases']:
                assert case['nonfinite_score_preserved'] and case['graph']['trace_count'] == 1
                assert case['graph']['jit_compile'] and not case['graph']['host_callbacks'] and not case['graph']['pfor']
                graph_path = directory / f'gaussian-{case["dimension"]}-{case["observations"]}-{case["unscented"]}-{dtype}.hlo.txt'
                assert hashlib.sha256(graph_path.read_bytes()).hexdigest() == case['graph']['hlo_sha256']
                for values in case['outputs']:
                    error = compare(values['candidate'], values['original'], 1e-9 if dtype == 'float64' else 1e-5)
                    max_numerical[dtype] = max(max_numerical[dtype], error)
                for fd in case['finite_differences']:
                    max_fd = max(max_fd, compare(fd['score'], fd['finite_difference'], 1e-7))
                assert len(case['finite_differences']) == (4 if dtype == 'float64' else 0)
        group = f'gaussian_binding_public_{device}'
        directory, manifest = found[group]
        record, evidence[group] = read(directory, manifest, 'gaussian-binding-public.json')
        assert len(record['cases']) == 4 and record['live_dataset_seeds'] and record['single_trace']
        assert record['preserved_public_invalidity'] and record['deferred_iapf_not_executed']
        for case in record['cases']:
            compare({k: v for k, v in case['candidate'].items() if k != 'runtime'},
                    {k: v for k, v in case['original'].items() if k != 'runtime'}, 1e-9)
        group = f'gaussian_binding_regressions_{device}'
        directory, manifest = found[group]
        assert all(manifest['source_sha256'][source] == digest for source, digest in hashes().items())
        caller_evidence = {}
        for case in ('ledh', 'ledh_diagnostics', 'sgqf', 'kdm_covariance', 'integrated_kdm', 'resampling_kdm'):
            path = directory / f'directions-{case}.json'
            record = json.loads(path.read_text())
            assert record['nonfinite_reset_rejected'] and record['host_nonfinite_rejection_checked']
            assert record['graph']['no_pfor_or_host_callback'] and record['graph']['trace_count'] == 1
            assert max(record['five_point_max_absolute_errors']) <= 2e-6
            hlo_path = directory / f'directions-{case}.hlo.txt'
            assert hashlib.sha256(hlo_path.read_bytes()).hexdigest() == record['graph']['hlo_sha256']
            caller_evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = directory / 'fitted-apf-endpoint-gaussian.json'
        record = json.loads(path.read_text())
        assert record['passed'] and record['factory_identity_verified']
        caller_evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence[group] = {'run': directory.name,
            'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'caller_artifact_sha256': caller_evidence,
            'worker_runtime': runtime_receipt(directory, manifest)}
    for device in ('cpu', 'gpu'):
        arms = {}
        for arm in ('before', 'after'):
            group = f'gaussian_binding_cost_{arm}_{device}'
            if device == 'gpu' and group not in found:
                continue
            directory, manifest = found[group]
            record, evidence[group] = read(directory, manifest, 'gaussian-binding-cost.json')
            assert record['exact_replay'] and len(record['warm_seconds']) == 30
            assert all(math.isfinite(v) and v > 0 for v in record['warm_seconds'])
            assert record['cold_seconds'] > 0
            if device == 'gpu':
                assert manifest['gpu_performance_preflight_uncontended']
            arms[arm] = (record, manifest)
        if not arms:
            results[device] = {'status': 'uncontended_gpu_costs_not_run'}
            continue
        assert set(arms) == {'before', 'after'}
        a, b = (arms[arm][0] for arm in ('before', 'after'))
        for key in ('row', 'settings', 'seed', 'environment', 'cpu_affinity', 'baseline_adapter_sha256'):
            assert a[key] == b[key], key
        if device == 'gpu':
            assert arms['before'][1]['gpu_uuid'] == arms['after'][1]['gpu_uuid']
        numerical_error = compare(a['numerical_result'], b['numerical_result'], 1e-9)
        results[device] = {'status': 'descriptive_only', 'numerical_max_error': numerical_error,
            'cold_ratio': b['cold_seconds']/a['cold_seconds'],
            'warm_ratio': b['warm_median_seconds']/a['warm_median_seconds'],
            'before_warm_seconds': a['warm_median_seconds'], 'after_warm_seconds': b['warm_median_seconds'],
            'extra_warm_RSS_MiB': (b['memory']['after_warm']['VmRSS']-a['memory']['after_warm']['VmRSS'])/2**20,
            'before_memory': a['memory'], 'after_memory': b['memory']}
    save(request, 'gaussian-binding-readback.json', {'schema': 'filter_gaussian_binding_readback.v1',
        'source_sha256': hashes(), 'evidence': evidence, 'preserved_failed_runs': failures,
        'maximum_numerical_errors': max_numerical, 'maximum_finite_difference_error': max_fd,
        'costs': results, 'numerical_qualified': True, 'terminal_cost_accepted': False,
        'runtime_setup_scope': 'Trusted campaign worker preconfigures TensorFlow; identical adapter setup descriptor excludes import/configure_runtime overhead in both arms.',
        'nonclaims': ['Gaussian binding only; shared eager random/KDM-pilot preparation remains open.',
                      'No adaptive iAPF, whole-repository closure, statistical ranking or main merge.']})
