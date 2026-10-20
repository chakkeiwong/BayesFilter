"""Independent standard-library verification of the DZ5 replay investigation."""

import array
import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_array(archive, name):
    data = archive.read(name)
    assert data[:6] == b'\x93NUMPY'
    width = 2 if data[6] == 1 else 4
    size = int.from_bytes(data[8:8 + width], 'little')
    header = ast.literal_eval(data[8 + width:8 + width + size].decode('latin1'))
    assert not header['fortran_order']
    payload = data[8 + width + size:]
    types = {'<f8': 'd', '<i8': 'q', '<i4': 'i', '|b1': 'b'}
    assert header['descr'] in types and sys.byteorder == 'little'
    result = array.array(types[header['descr']])
    result.frombytes(payload)
    assert len(result) == math.prod(header['shape'])
    return header, result


def compare_archives(left_path, right_path, histories=False):
    report = {}
    with zipfile.ZipFile(left_path) as left, zipfile.ZipFile(right_path) as right:
        assert set(left.namelist()) == set(right.namelist())
        for filename in left.namelist():
            header, a = read_array(left, filename)
            other_header, b = read_array(right, filename)
            assert header == other_header
            assert all(math.isfinite(value) for value in a)
            assert all(math.isfinite(value) for value in b)
            name = filename.removesuffix('.npy')
            changed = 0
            maximum = 0.
            dates = [0] * header['shape'][0] if histories and name.startswith('history_') else None
            stride = math.prod(header['shape'][1:]) if dates is not None else None
            for index, (x, y) in enumerate(zip(a, b, strict=True)):
                if x != y:
                    changed += 1
                    maximum = max(maximum, abs(float(x) - float(y)))
                    if dates is not None:
                        dates[index // stride] += 1
            row = {'changed_elements': changed, 'max_absolute_error': maximum}
            if dates is not None:
                row.update(changed_by_time=dates,
                    observation=next((index + 1 for index, count in enumerate(dates) if count), None))
            if name in ('valid', 'history_valid'):
                assert all(a) and all(b)
            if name == 'branch_status':
                assert not any(a) and not any(b)
            report[name] = row
    return report


def read_run(root, number, passed=True):
    directory = root / f'run-{number:05d}'
    run = json.loads((directory / 'run.json').read_text())
    cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
    assert len(cases) == 1
    assert (run['state'] == 'passed') == passed
    assert (run['exit_code'] == 0) == passed
    assert bool(list(cases[0])) != passed
    return run


def analyze(root):
    spec = importlib.util.spec_from_file_location('prior_oracle_analysis',
        Path(__file__).with_name('analyze-dz5-score-oracle-20260925.py'))
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verified = {number: verifier.verify_run(root, number) for number in (3866, 3867, 3901)}
    _, reference, oracle, replay, errors = verified[3901]
    assert not reference['jit_compile']
    assert reference['optimizer_options']['arithmetic_optimization'] is False
    assert reference['reference_control'] == 'explicit_graph_arithmetic_optimizer_disabled'
    comparisons = {}
    for number in (3866, 3867):
        baseline = verified[number][2]
        assert oracle['bank'] == baseline['bank']
        comparisons[str(number)] = {
            'value_scaled_error': verifier.compare(oracle['values'], baseline['values']),
            'score_scaled_error': verifier.compare(
                [x for row in oracle['scores'] for x in row],
                [x for row in baseline['scores'] for x in row])}
    rejected = []
    for mutation in ('score', 'value', 'replay', 'validity', 'snapshot', 'false_pass'):
        a, b = copy.deepcopy(oracle), copy.deepcopy(replay)
        if mutation == 'score':
            a['scores'][0][0] += 1.
            b['scores'][0][0] += 1.
        elif mutation == 'value':
            a['values'][1] += 1.
            b['values'][1] += 1.
        elif mutation == 'replay':
            b['scores'][0][0] += 1e-8
        elif mutation == 'validity':
            a['validity'][0] = b['validity'][0] = False
        elif mutation == 'snapshot':
            a['snapshot_manifest_sha256'] = 'stale'
        else:
            a['checks'][0]['finite_difference'][0] += 1.
        try:
            verifier.verify_oracle(a, b, oracle['snapshot_manifest_sha256'])
        except (AssertionError, KeyError):
            rejected.append(mutation)
        else:
            raise AssertionError(f'Corrupt oracle accepted: {mutation}')

    read_run(root, 3896)
    directory = root / 'run-03896'
    saved = json.loads((directory / 'dz5-prefix-replay.json').read_text())
    trajectories = []
    for repeat in (1, 2):
        comparison = compare_archives(directory / 'prefix-replay-0.npz',
            directory / f'prefix-replay-{repeat}.npz', histories=True)
        for key, row in saved['replay_summaries'][repeat - 1].items():
            assert row == {field: comparison[key][field] for field in row}
        if repeat == 2:
            for key, row in saved['first_differing_time'].items():
                assert row == {field: comparison[key][field] for field in row}
        trajectories.append(comparison)
    assert all(row['changed_elements'] == 0 for row in trajectories[0].values())
    assert trajectories[1]['history_d_mean']['observation'] == 35
    assert trajectories[1]['history_d_factor']['observation'] == 36
    assert trajectories[1]['history_score']['observation'] == 36
    for key in ('history_mean', 'history_factor', 'history_value', 'history_valid'):
        assert trajectories[1][key]['changed_elements'] == 0
    read_run(root, 3894, passed=False)
    read_run(root, 3895)
    controls = []
    for threads in (2, 1):
        directory = root / 'run-03895' / f'threads-{threads}'
        record = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        assert record['passed'] and record['replay_exact']
        assert record['frozen_input_sha256'] == sha(directory / 'frozen-step-inputs.npz')
        assert len(record['call_seconds']) == 20
        for repeat in range(1, 20):
            comparison = compare_archives(directory / 'late-step-replay-00.npz',
                directory / f'late-step-replay-{repeat:02d}.npz')
            assert all(row['changed_elements'] == 0 for row in comparison.values())
        controls.append(record['frozen_input_sha256'])
    assert controls[0] == controls[1]
    read_run(root, 3900)
    directory = root / 'run-03900'
    graphs = json.loads((directory / 'dz5-optimized-additions.json').read_text())
    for row in graphs:
        assert sha(directory / row['path']) == row['sha256']
    primitive = json.loads((directory / 'addn-buffer-primitive.json').read_text())
    assert [row['expose'] for row in primitive] == ['none', 'first_two', 'all']
    for row, expected in zip(primitive, (1., 0., 1.), strict=True):
        assert row['expected_exact_sum'] == 1. and row['trace_count'] == 1
        assert len(row['observed']) == 4
        assert all(observed == [expected] * 257 for observed in row['observed'])
    return {'schema': 'filter_repair_dz5_replay_analysis.v1',
        'qualified_controlled_graph_run': 3901, 'full_score_scaled_errors': errors,
        'comparison_to_qualified_xla': comparisons, 'rejected_oracle_mutations': rejected,
        'trajectory_03896': trajectories, 'frozen_step_48_control_calls_per_thread': 20,
        'frozen_input_sha256': controls[0],
        'optimized_addn_counts': [len(row['addn']) for row in graphs],
        'primitive_addn_results': [1., 0., 1.],
        'timing_and_memory_observations': {key: reference[key] for key in (
            'cold_seconds', 'replay_seconds', 'sampled_rss_before_bytes',
            'sampled_peak_rss_bytes', 'sampled_rss_after_bytes', 'allocator')},
        'nonclaims': ['Default graph replay remains unresolved.',
            'Exact failing filter operator is not proved.',
            'Frozen MacroFinance autodiff/pfor is outside this repair scope.',
            'No admission, default change, performance ranking or compiler-memory attribution.']}


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2, allow_nan=False))
