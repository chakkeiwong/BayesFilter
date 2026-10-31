"""Validate fresh saved target checks for isolated initializer regression use.

This issues a new target-only artifact consumed by the real external adapter.
It cannot renew old admissions or admit training, HMC, or posterior results.
All numerical evidence is from the TensorFlow/XLA workers; this is a standard
library readback of those records and their exact frozen dependency closure.
"""

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

EXPECTED = frozenset({
    'dz5_initializer_target_4_cpu', 'dz5_initializer_target_1_gpu',
    'dz5_initializer_target_4_gpu', 'dz5_initializer_target_46_gpu',
    'dz5_initializer_target_68_gpu', 'dz5_initializer_oracle_cpu',
    'dz5_initializer_oracle_gpu',
})
BF = '/home/ubuntu/workspace/BayesFilter'
MF = '/home/ubuntu/workspace/MacroFinance-dz5-neutra'
FIXTURE = 'e116fe853c8579369036ab2ce57724ba07544524d0920fa0a706d76714f75d8a'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(actual, expected, *, atol=1e-8, rtol=1e-7):
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            close(left, right, atol=atol, rtol=rtol)
    else:
        assert math.isfinite(actual) and math.isfinite(expected)
        assert abs(actual - expected) <= atol + rtol * abs(expected)


def validate_report(report, manifest_hash, sources, device):
    assert report['passed'] and report['target_evaluated']
    assert report['snapshot_manifest_sha256'] == manifest_hash
    assert report['fixture_identity'] == FIXTURE
    assert report['observation_shape'][0] == 96 and len(report['parameter_names']) == 23
    assert report['jit_compile'] is True and report['trace_count'] == 1
    assert not report['host_callbacks'] and not report['forbidden_local_runtime_imports']
    assert not report['unexpected_modules'] and not report['post_target_unexpected_modules']
    assert report['tf32_enabled'] is False and report['device'] == device
    assert report['filter_contract'] == 'rectangular_srukf_full_rank_identity_prepared_cir_v2'
    policy = report['gpu_memory_policy']
    assert policy['mode'] == 'memory_growth'
    assert policy['configured_before_logical_device_initialization']
    assert policy['full_device_preallocation_disabled']
    assert policy['tf_force_gpu_allow_growth'] == 'true'
    assert policy['all_physical_devices_memory_growth']
    assert len(policy['physical_devices']) == (1 if device == 'GPU' else 0)
    assert all(row['memory_growth'] is True for row in policy['physical_devices'])
    for path, options in report['source_mount_options'].items():
        assert path in (BF, MF) and 'ro' in options
    assert set(report['source_mount_options']) == {BF, MF}
    for group in ('loaded_modules', 'post_target_loaded_modules'):
        for row in report[group].values():
            assert sources[row['path']]['sha256'] == row['sha256']
    for path, value in report['source_closure'].items():
        assert sources[path]['sha256'] == value
    if device == 'GPU':
        assert report['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert len(report['physical_gpus']) == 1


def build(root, snapshot, first, last):
    manifest = read(snapshot / 'manifest.json')
    manifest_hash = digest(snapshot / 'manifest.json')
    sources = manifest['sources']
    assert all(digest(row['saved']) == row['sha256'] for row in sources.values())
    assert all(digest(row['saved']) == row['sha256'] for row in manifest['inputs'].values())
    completed, artifacts, reports = {}, {str(snapshot / 'manifest.json'): manifest_hash}, {}
    for number in range(first, last + 1):
        directory = root / f'run-{number:05d}'
        run = read(directory / 'run.json')
        group = run['key'][1]
        if group not in EXPECTED:
            continue
        assert group not in completed, 'Duplicate target evidence requires explicit disposition'
        assert run['state'] == 'passed' and run['exit_code'] == 0
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and not any(c.findall(t) for c in cases for t in ('failure', 'error', 'skipped'))
        report = read(directory / 'dz5-snapshot-import.json')
        device = 'GPU' if group.endswith('_gpu') else 'CPU'
        assert run['device'] == device
        assert run['environment']['TF_FORCE_GPU_ALLOW_GROWTH'] == 'true'
        validate_report(report, manifest_hash, sources, device)
        if 'oracle' in group:
            assert report['replay_exact'] and report['batch'] == 185
            oracle = read(directory / 'dz5-score-oracle.json')
            assert len(oracle['bank']) == len(oracle['values']) == len(oracle['scores']) == 185
            assert len(oracle['validity']) == len(oracle['branch_status']) == 185
            assert len(oracle['prior_scale']) == 23
            assert all(oracle['validity']) and all(code == 0 for code in oracle['branch_status'])
            assert len(oracle['checks']) == 2
            for index, check in enumerate(oracle['checks']):
                assert check['passed']
                step = (1e-3, 5e-4)[index]
                assert check['step_prior_sd'] == step
                values = oracle['values']
                start = 1 + 92 * index
                derivative = [(values[start + axis] - 8 * values[start + 23 + axis]
                    + 8 * values[start + 46 + axis] - values[start + 69 + axis])
                    / (12 * step * oracle['prior_scale'][axis]) for axis in range(23)]
                close(oracle['scores'][0], derivative)
                close(check['finite_difference'], derivative, atol=1e-12, rtol=1e-12)
                close(oracle['scores'][0], check['finite_difference'])
            hlo = directory / 'dz5-score-oracle.hlo.txt'
        else:
            batch = int(group.split('_')[-2])
            assert report['batch'] == batch
            comparisons = report['comparisons']
            assert [row['label'] for row in comparisons] == ['initial', 'changed', 'replay']
            assert report['hlo_changed_input_equal']
            for row in comparisons:
                assert len(row['value']) == len(row['score']) == len(row['positions']) == batch
                assert all(len(score) == 23 for score in row['score'])
                close(row['value'], row['reference_value'])
                close(row['score'], row['reference_score'])
                assert all(row['status']['valid_pre_regularized_score'])
                assert all(code == 0 for code in row['status']['branch_status_code'])
            assert comparisons[0]['value'] == comparisons[-1]['value']
            assert comparisons[0]['score'] == comparisons[-1]['score']
            if report['batch'] == 4:
                assert report['invalid_row_isolation']
                invalid = report['invalid_rows']
                assert invalid['valid'] == [True, False, False, True]
                assert invalid['value'][1:3] == [-1e100, -1e100]
                assert invalid['score'][1:3] == [[0.] * 23] * 2
            hlo = directory / 'dz5-target.hlo.txt'
        assert digest(hlo) == report['hlo_sha256']
        reports[group] = report
        completed[group] = number
        for path in directory.iterdir():
            if path.is_file():
                artifacts[str(path)] = digest(path)
    assert set(completed) == EXPECTED, 'Missing fresh target/score evidence'
    for suffix in ('target_4', 'oracle'):
        cpu, gpu = (reports[f'dz5_initializer_{suffix}_{device}'] for device in ('cpu', 'gpu'))
        if suffix == 'target_4':
            for left, right in zip(cpu['comparisons'], gpu['comparisons'], strict=True):
                assert left['positions'] == right['positions'] and left['status'] == right['status']
                close(left['value'], right['value'])
                close(left['score'], right['score'])
        else:
            cp = read(root / f"run-{completed['dz5_initializer_oracle_cpu']:05d}" / 'dz5-score-oracle.json')
            gp = read(root / f"run-{completed['dz5_initializer_oracle_gpu']:05d}" / 'dz5-score-oracle.json')
            assert cp['bank'] == gp['bank'] and cp['validity'] == gp['validity']
            close(cp['values'], gp['values'])
            close(cp['scores'], gp['scores'])
    return {'schema': 'filter_repair_dz5_initializer_target_admission.v1',
        'role': 'fresh_target_only_isolated_initializer_regression', 'passed': True,
        'stage': 'CDF', 'coordinate': 'sqrt_intensity_v1', 'fixture_identity': FIXTURE,
        'filter_contract': 'rectangular_srukf_full_rank_identity_prepared_cir_v2',
        'target_gpu_xla_passed': True, 'score_and_rejection_passed': True,
        'target_sources': {path: row['sha256'] for path, row in sources.items()},
        'artifacts': artifacts, 'fresh_runs': completed, 'snapshot_manifest_sha256': manifest_hash,
        'old_admission_reused': False, 'training_authorized': False, 'hmc_authorized': False,
        'retained_authorized': False, 'nonclaims': ['No full-chain XLA, tuning, transport quality, posterior or scientific admission.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--snapshot', required=True, type=Path)
    parser.add_argument('--first-run', required=True, type=int)
    parser.add_argument('--last-run', required=True, type=int)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = build(args.root, args.snapshot, args.first_run, args.last_run)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'fresh_runs': result['fresh_runs']}))
