"""Independent standard-library reconstruction of saved DZ5 score checks."""

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(left, right):
    assert len(left) == len(right)
    errors = []
    for a, b in zip(left, right, strict=True):
        assert math.isfinite(a) and math.isfinite(b)
        errors.append(abs(a - b) / (1e-8 + 1e-7 * abs(b)))
    assert max(errors, default=0.) <= 1., max(errors)
    return max(errors, default=0.)


def verify_oracle(oracle, replay, manifest_sha):
    assert oracle['snapshot_manifest_sha256'] == manifest_sha
    assert len(oracle['bank']) == len(oracle['values']) == len(oracle['scores']) == 185
    assert all(len(row) == 23 for row in oracle['bank'] + oracle['scores'])
    assert len(oracle['parameter_names']) == len(set(oracle['parameter_names'])) == 23
    assert oracle['validity'] == replay['validity'] == [True] * 185
    assert oracle['branch_status'] == replay['branch_status'] == [0] * 185
    assert oracle['values'] == replay['values']
    assert oracle['scores'] == replay['scores']
    assert replay['value_max_absolute_error'] == replay['score_max_absolute_error'] == 0.
    for row in oracle['scores']:
        assert all(math.isfinite(x) for x in row)
    assert all(math.isfinite(x) for x in oracle['values'])
    errors = []
    for index, step in enumerate((1e-3, 5e-4)):
        numerical = []
        for coordinate, scale in enumerate(oracle['prior_scale']):
            assert scale > 0 and math.isfinite(scale)
            for multiplier_index, multiplier in enumerate((-2, -1, 1, 2)):
                row_index = 1 + 92 * index + 23 * multiplier_index + coordinate
                expected = list(oracle['bank'][0])
                expected[coordinate] += multiplier * step * scale
                assert expected == oracle['bank'][row_index]
            offset = 1 + 92 * index + coordinate
            a, b, c, d = (oracle['values'][offset + shift * 23] for shift in range(4))
            numerical.append((a - 8 * b + 8 * c - d) / (12 * step * scale))
        saved = oracle['checks'][index]
        assert saved['step_prior_sd'] == step and saved['passed'] is True
        # The independent scalar reconstruction should match the diagnostic
        # array arithmetic exactly for this fixed sequence of operations.
        assert numerical == saved['finite_difference']
        absolute = [abs(a - b) for a, b in zip(oracle['scores'][0], numerical, strict=True)]
        allowed = [1e-8 + 1e-7 * abs(x) for x in numerical]
        assert absolute == saved['absolute_error'] and allowed == saved['allowed_error']
        error = compare(oracle['scores'][0], numerical)
        assert error == saved['max_scaled_error']
        errors.append(error)
    return errors


def verify_run(root, number):
    directory = root / f'run-{number:05d}'
    run = json.loads((directory / 'run.json').read_text())
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    oracle = json.loads((directory / 'dz5-score-oracle.json').read_text())
    replay = json.loads((directory / 'dz5-score-replay.json').read_text())
    command = json.loads((directory / 'isolated-dz5-command.json').read_text())
    cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
    assert len(cases) == 1 and not list(cases[0])
    assert run['state'] == 'passed' and run['exit_code'] == 0
    assert report['passed'] and report['target_evaluated'] and not report['adapter_admitted']
    assert report['role'] == 'fresh_DZ5_independent_five_point_score_oracle'
    assert report['op_determinism_enabled'] and not report['tf32_enabled']
    assert report['batch'] == 185 and report['observation_shape'][0] == 96
    assert report['trace_count'] == 1 and report['replay_exact'] and report['host_callbacks'] == []
    assert report['device'] == oracle['device'] == run['device']
    assert report['jit_compile'] == oracle['jit_compile']
    assert not report['unexpected_modules'] and not report['post_target_unexpected_modules']
    assert command['child_sha256'] == sha(directory / 'isolated-dz5-import.py')
    snapshot = Path(command['command'][command['command'].index('/tmp/dz5-source') - 1])
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    manifest_sha = sha(snapshot / 'manifest.json')
    assert report['snapshot_manifest_sha256'] == command['snapshot_manifest_sha256'] == manifest_sha
    assert oracle['frozen_qualifier_sha256'] == report['frozen_qualifier_sha256'] == manifest['frozen_qualifier_sha256']
    assert report['bayesfilter_commit'] == manifest['bayesfilter_commit']
    for path, entry in manifest['sources'].items():
        assert sha(snapshot / path.lstrip('/')) == entry['sha256']
    for module in report['post_target_loaded_modules'].values():
        assert manifest['sources'][module['path']]['sha256'] == module['sha256']
    for path, digest in report['source_closure'].items():
        assert manifest['sources'][path]['sha256'] == digest
    assert report['fixture_identity'] == 'e116fe853c8579369036ab2ce57724ba07544524d0920fa0a706d76714f75d8a'
    assert report['filter_contract'] == 'rectangular_srukf_full_rank_identity_prepared_cir_v2'
    assert set(report['source_mount_options']) == {
        '/home/ubuntu/workspace/BayesFilter', '/home/ubuntu/workspace/MacroFinance-dz5-neutra'}
    assert all('ro' in options for options in report['source_mount_options'].values())
    memory = report['gpu_memory_policy']
    assert memory['configured_before_logical_device_initialization']
    assert memory['mode'] == 'memory_growth' and memory['full_device_preallocation_disabled']
    assert memory['all_physical_devices_memory_growth']
    if run['device'] == 'GPU':
        assert report['cuda_visible_devices'] == run['gpu_uuid']
        assert report['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert len(memory['physical_devices']) == len(report['physical_gpus']) == 1
        assert 'DEVICE:GPU:0' in report['result_device'].upper()
    else:
        assert report['cuda_visible_devices'] == '-1' and not memory['physical_devices']
        assert 'DEVICE:CPU:0' in report['result_device'].upper()
    if report['jit_compile']:
        assert sha(directory / 'dz5-score-oracle.hlo.txt') == report['hlo_sha256']
    errors = verify_oracle(oracle, replay, manifest_sha)
    return run, report, oracle, replay, errors


def analyze(root, numbers):
    verified = [verify_run(root, number) for number in numbers]
    assert {item[0]['device'] for item in verified} == {'CPU', 'GPU'}
    assert all(item[1]['jit_compile'] for item in verified)
    baseline = verified[0][2]
    rows = []
    for number, (run, report, oracle, replay, errors) in zip(numbers, verified, strict=True):
        assert run['source_sha256'] == verified[0][0]['source_sha256']
        assert oracle['bank'] == baseline['bank']
        assert oracle['snapshot_manifest_sha256'] == baseline['snapshot_manifest_sha256']
        assert oracle['validity'] == baseline['validity'] and oracle['branch_status'] == baseline['branch_status']
        rows.append({'run': number, 'device': run['device'], 'jit_compile': report['jit_compile'],
            'max_scaled_five_point_errors': errors,
            'cross_device_value_scaled_error': compare(oracle['values'], baseline['values']),
            'cross_device_score_scaled_error': compare([x for row in oracle['scores'] for x in row],
                [x for row in baseline['scores'] for x in row]),
            'cold_seconds': report['cold_seconds'], 'replay_seconds': report['replay_seconds'],
            'rss_before_bytes': report['sampled_rss_before_bytes'], 'rss_peak_bytes': report['sampled_peak_rss_bytes'],
            'rss_after_bytes': report['sampled_rss_after_bytes'], 'allocator': report['allocator'],
            'oracle_sha256': sha(root / f'run-{number:05d}' / 'dz5-score-oracle.json')})
    rejected = []
    oracle, replay = verified[0][2:4]
    for mutation in ('score', 'value', 'false_pass', 'snapshot', 'replay', 'validity'):
        candidate, repeated = copy.deepcopy(oracle), copy.deepcopy(replay)
        if mutation == 'score':
            candidate['scores'][0][0] += 1.
            repeated['scores'][0][0] += 1.
        elif mutation == 'value':
            candidate['values'][1] += 1.
            repeated['values'][1] += 1.
        elif mutation == 'false_pass':
            candidate['checks'][0]['finite_difference'][0] += 1.
            candidate['checks'][0]['passed'] = True
        elif mutation == 'snapshot':
            candidate['snapshot_manifest_sha256'] = 'stale'
        elif mutation == 'replay':
            repeated['scores'][0][0] += 1e-12
        else:
            candidate['validity'][0] = repeated['validity'][0] = False
        try:
            verify_oracle(candidate, repeated, oracle['snapshot_manifest_sha256'])
        except (AssertionError, KeyError):
            rejected.append(mutation)
        else:
            raise AssertionError(f'Accepted corrupt record: {mutation}')
    return {'schema': 'filter_repair_dz5_score_analysis.v1', 'xla_oracle_passed': True,
        'rows': rows, 'rejected_mutations': rejected, 'snapshot_manifest_sha256': baseline['snapshot_manifest_sha256'],
        'nonclaims': ['CPU graph exact replay remains vetoed.', 'No adapter admission or E5 completion.',
            'Single-process timings/RSS are observations, not a repeated performance ranking or memory-cause diagnosis.']}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]), [int(x) for x in sys.argv[2:]]), indent=2, allow_nan=False))
