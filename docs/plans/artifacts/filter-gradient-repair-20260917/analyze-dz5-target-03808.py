"""Post-run diagnostic comparison of explicitly selected frozen CDF records."""

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
CANDIDATE = ROOT / 'dz5-candidate-source-merged-9d8202b77-r2'
ARCHIVED = ROOT / 'dz5-archived-cdf-source-31f0067f9-r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerical(actual, expected):
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        return max((numerical(a, b) for a, b in zip(actual, expected, strict=True)), default=0.)
    assert type(actual) in (int, float) and type(expected) in (int, float)
    assert math.isfinite(actual) and math.isfinite(expected)
    scaled = abs(actual - expected) / (1e-8 + 1e-7 * abs(expected))
    assert scaled <= 1., (actual, expected, scaled)
    return scaled


def load(number, snapshot):
    directory = ROOT / f'run-{number:05d}'
    run = json.loads((directory / 'run.json').read_text())
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    command = json.loads((directory / 'isolated-dz5-command.json').read_text())
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    assert run['state'] == 'passed' and run['exit_code'] == 0
    assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
    assert report['passed'] and report['target_evaluated'] and not report['adapter_admitted']
    assert report['snapshot_manifest_sha256'] == command['snapshot_manifest_sha256'] == sha(snapshot / 'manifest.json')
    assert command['child_sha256'] == sha(directory / 'isolated-dz5-import.py')
    assert report['hlo_sha256'] == sha(directory / 'dz5-target.hlo.txt')
    assert report['trace_count'] == 1 and report['hlo_changed_input_equal']
    assert not report['host_callbacks'] and report['tf32_enabled'] is False
    assert not report['unexpected_modules'] and not report['post_target_unexpected_modules']
    assert report['device'] == run['device']
    assert report['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
    assert all('ro' in options for options in report['source_mount_options'].values())
    memory = report['gpu_memory_policy']
    assert memory['mode'] == 'memory_growth'
    assert memory['configured_before_logical_device_initialization']
    assert memory['all_physical_devices_memory_growth']
    if run['device'] == 'GPU':
        assert len(report['physical_gpus']) == 1
        assert report['cuda_visible_devices'].startswith('GPU-')
        assert report['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert report['gpu_allocator']['peak'] >= report['gpu_allocator']['current'] >= 0
        assert run['gpu_uuid'] == report['cuda_visible_devices']
        assert all(row['result_device'].endswith('device:GPU:0') for row in report['comparisons'])
    else:
        assert report['cuda_visible_devices'] == '-1' and report['physical_gpus'] == []
    for canonical, row in manifest['sources'].items():
        assert sha(Path(row['saved'])) == row['sha256'], canonical
    forbidden_imports = []
    for loaded in report['post_target_loaded_modules'].values():
        row = manifest['sources'][loaded['path']]
        assert row['sha256'] == loaded['sha256']
        if loaded['path'].endswith('.py'):
            for node in ast.walk(ast.parse(Path(row['saved']).read_text())):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                if isinstance(node, ast.ImportFrom):
                    names = [node.module or '']
                if any(name == 'filters' or name.startswith(('filters.', 'inference.hmc',
                        'inference.mass_matrix', 'inference.posterior_adapter')) for name in names):
                    forbidden_imports.append((loaded['path'], node.lineno, names))
    assert not forbidden_imports, forbidden_imports
    return run, report


def compare(actual, expected):
    for key in ('batch', 'fixture_identity', 'filter_contract', 'observation_shape',
                'parameter_names', 'tfp_special_sha256', 'tensorflow'):
        assert actual[key] == expected[key], key
    assert [row['label'] for row in actual['comparisons']] == ['initial', 'changed', 'replay']
    assert [row['label'] for row in expected['comparisons']] == ['initial', 'changed', 'replay']
    errors = []
    for left, right in zip(actual['comparisons'], expected['comparisons'], strict=True):
        assert left['positions'] == right['positions']
        assert left['status'] == right['status']
        for field in ('value', 'score', 'reference_value', 'reference_score'):
            errors.append(numerical(left[field], right[field]))
        assert left['floating_diagnostics'].keys() == right['floating_diagnostics'].keys()
        for key in left['floating_diagnostics']:
            errors.append(numerical(left['floating_diagnostics'][key], right['floating_diagnostics'][key]))
    if actual['batch'] == 4:
        assert actual['invalid_row_isolation'] and expected['invalid_row_isolation']
        for key in ('valid', 'branch_status_code'):
            assert actual['invalid_rows'][key] == expected['invalid_rows'][key]
        for key in ('value', 'score'):
            errors.append(numerical(actual['invalid_rows'][key], expected['invalid_rows'][key]))
    return max(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', nargs='+', required=True, help='candidate:archived run numbers')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'schema': 'filter_repair_dz5_target_comparison.v1', 'passed': False,
        'role': 'frozen_actual_target_engineering_only', 'pairs': [], 'cross_device': [],
        'nonclaims': ['No admission, initializer qualification, HMC, cost ranking or main promotion.'],
        'analyzer_sha256': sha(Path(__file__))}
    reports = {}
    for pair in args.pairs:
        first, second = map(int, pair.split(':'))
        ra, a = load(first, CANDIDATE)
        rb, b = load(second, ARCHIVED)
        assert ra['device'] == rb['device']
        if ra['device'] == 'GPU':
            assert a['cuda_visible_devices'] == b['cuda_visible_devices']
        error = compare(a, b)
        result['pairs'].append({'candidate_run': first, 'archived_run': second,
            'batch': a['batch'], 'device': ra['device'], 'max_scaled_error': error,
            'candidate_report_sha256': sha(ROOT / f'run-{first:05d}/dz5-snapshot-import.json'),
            'archived_report_sha256': sha(ROOT / f'run-{second:05d}/dz5-snapshot-import.json'),
            'uuid': a['cuda_visible_devices'], 'module_import_audit_passed': True})
        assert (ra['device'], a['batch']) not in reports
        reports[(ra['device'], a['batch'])] = (a, b)
    for (device, batch), rows in reports.items():
        if device == 'GPU' and ('CPU', batch) in reports:
            errors = [compare(gpu, cpu) for gpu, cpu in zip(rows, reports[('CPU', batch)], strict=True)]
            result['cross_device'].append({'batch': batch, 'max_scaled_error': max(errors)})
    # Adverse mutations establish that the comparator enforces key vetoes.
    a, b = next(iter(reports.values()))
    mutations = []
    for label in ('score', 'nonfinite', 'status', 'position', 'fixture'):
        changed = copy.deepcopy(a)
        if label == 'score': changed['comparisons'][0]['score'][0][0] += 1.
        elif label == 'nonfinite': changed['comparisons'][0]['value'][0] = float('nan')
        elif label == 'status': changed['comparisons'][0]['status']['branch_status_code'][0] = 9001
        elif label == 'position': changed['comparisons'][0]['positions'][0][0] += .001
        else: changed['fixture_identity'] = 'wrong_fixture'
        try:
            compare(changed, b)
        except AssertionError:
            mutations.append(label)
        else:
            raise AssertionError(f'adverse mutation accepted: {label}')
    result.update(passed=True, rejected_mutations=mutations)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': True, 'pairs': len(result['pairs']),
        'cross_device_pairs': len(result['cross_device']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
