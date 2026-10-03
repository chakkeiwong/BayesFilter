"""Named compiler-pass ablation on frozen historical CPU locator diagnostics."""

import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_filter_repair_locator_optimized_hlo import comparison, computations
from tests import test_filter_repair_dz5_locator_derived_replay as records
from tests import test_filter_repair_dz5_locator_optimized_hlo as probe
from tests.test_filter_repair_dz5_locator_context_readback import array_comparison
from tests.test_filter_repair_geometry_control import save

FIRST_RUN = 4877
LAST_RUN = 4882
CONTROLS = {'original': 4656, 'candidate': 4657}
ARMS = tuple((arm, mode) for mode in ('control', 'disabled') for arm in CONTROLS)
FLAG = '--xla_disable_hlo_passes=while-loop-constant-sinking'
RAW = probe.probe.context.trajectory.consumer.RAW


def child_source(arm, mode):
    child = probe.child_source(arm)
    anchor = '    import tensorflow as tf\n'
    assert child.count(anchor) == 1
    flags = FLAG if mode == 'disabled' else ''
    addition = (
        "    assert not os.environ.get('XLA_FLAGS', ''), 'Unexpected inherited compiler flags'\n"
        f"    os.environ['XLA_FLAGS'] = {flags!r}\n"
        "    report['compiler_intervention'] = {\n"
        f"        'arm': {arm!r}, 'mode': {mode!r}, 'xla_flags': {flags!r},\n"
        "        'role': 'historical_CPU_reference_named_pass_ablation',\n"
        "        'runtime_default_change': False}\n"
    )
    child = child.replace(anchor, addition + anchor)
    compile(child, '<constant_sinking_diagnostic>', 'exec')
    return child


def load_callbacks(directory):
    with np.load(directory / 'locator-callbacks.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {'positions', 'values', 'scores', 'valid'}
        return {key: archive[key].copy() for key in archive.files}


def callbacks_comparison(actual, expected):
    result = {}
    for key in actual:
        if actual[key].shape != expected[key].shape:
            result[key] = {'bitwise_equal': False,
                'actual_shape': list(actual[key].shape),
                'expected_shape': list(expected[key].shape)}
        else:
            result[key] = array_comparison(actual[key], expected[key])
    result['first_position_equal'] = actual['positions'][0].tobytes() == expected['positions'][0].tobytes()
    if result['first_position_equal']:
        result['first_score'] = array_comparison(actual['scores'][0], expected['scores'][0])
    result['identical_input_rows'] = [
        {'row': index, 'score': array_comparison(actual['scores'][index], expected['scores'][index]),
         'value': array_comparison(actual['values'][index:index + 1], expected['values'][index:index + 1])}
        for index in range(min(len(actual['positions']), len(expected['positions'])))
        if actual['positions'][index].tobytes() == expected['positions'][index].tobytes()]
    return result


def verify_saved(directory, arm, mode):
    report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
    old = RAW / f'run-{CONTROLS[arm]:05d}'
    previous = json.loads((old / 'dz5-snapshot-import.json').read_text())
    for key in ('snapshot_manifest_sha256', 'probe_dispatch_sha256',
            'original_locator_sha256', 'executed_original_source_sha256',
            'settings', 'fixture_identity', 'source_closure', 'loaded_modules',
            'tensorflow', 'cuda_visible_devices'):
        assert report[key] == previous[key], (arm, mode, key)
    command = json.loads((directory / 'isolated-dz5-command.json').read_text())
    child_digest = hashlib.sha256((directory / 'isolated-dz5-import.py').read_bytes()).hexdigest()
    assert command['child_sha256'] == child_digest == hashlib.sha256(child_source(arm, mode).encode()).hexdigest()
    assert command['snapshot_manifest_sha256'] == report['snapshot_manifest_sha256']
    manifest = json.loads((directory / 'run.json').read_text())
    baseline_manifest = json.loads((old / 'run.json').read_text())
    assert manifest['environment'] == baseline_manifest['environment']
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['compiler_intervention'] == {
        'arm': arm, 'mode': mode, 'xla_flags': FLAG if mode == 'disabled' else '',
        'role': 'historical_CPU_reference_named_pass_ablation',
        'runtime_default_change': False}
    for filename, field in (('locator-callbacks.npz', 'callbacks_sha256'),
            ('first-objective-context.hlo.txt', 'context_hlo_sha256'),
            ('optimized-context.hlo.txt', 'optimized_hlo_sha256')):
        assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == report[field]
    actual = load_callbacks(directory)
    expected = load_callbacks(old)
    differences = records.record_differences(
        json.loads((directory / 'locator-result.json').read_text()),
        json.loads((old / 'locator-result.json').read_text()))
    if mode == 'control':
        assert all(actual[key].tobytes() == expected[key].tobytes() for key in actual)
        assert not differences, differences
        assert report['optimizer_callback_batches'] == 3
    return report, actual, differences, callbacks_comparison(actual, expected)


@pytest.mark.parametrize('arm,mode', ARMS)
def test_named_pass_intervention(request, arm, mode):
    # Require both fresh controls before allowing an ablated execution.
    if mode == 'disabled':
        found = saved_runs()
        for control_arm in CONTROLS:
            verify_saved(found[(control_arm, 'control')], control_arm, 'control')
    report = probe.probe.context.trajectory.snapshot_test.run_isolated_snapshot(
        request, snapshot=probe.probe.context.trajectory.SNAPSHOT,
        child=child_source(arm, mode),
        scope=f'CPU_constant_sinking_{arm}_{mode}_historical_diagnostic_only',
        child_timeout_seconds=870, read_only_paths=(RAW,))
    assert report['target_evaluated'] and report['optimized_hlo_bytes'] > 0
    directory = Path(request.config.getoption('xmlpath')).parent
    verify_saved(directory, arm, mode)


def saved_runs():
    found = {}
    for number in range(FIRST_RUN, LAST_RUN + 1):
        path = RAW / f'run-{number:05d}' / 'run.json'
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        for arm, mode in ARMS:
            if run['key'][1] != f'dz5_locator_sinking_{arm}_{mode}_cpu' or run['state'] != 'passed':
                continue
            assert (arm, mode) not in found, 'Repeated successful arm requires a disposition'
            assert run['device'] == 'CPU' and run['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
            found[(arm, mode)] = path.parent
    return found


def test_saved_pass_intervention(request):
    found = saved_runs()
    assert set(found) == set(ARMS)
    historical_structure = RAW / 'terminal-locator-optimized-20260928-r1/comparison.json'
    old = json.loads(historical_structure.read_text())
    pair = old['comparisons']['candidate__replay_int32']
    signatures = {row['normalized_sha256']: row['instruction_count']
        for side in ('left_unmatched', 'right_unmatched') for row in pair[side]
        if row['name'].startswith('fused_computation.')}
    assert sorted(signatures.values()) == [214, 217]
    result = {'schema': 'filter_locator_constant_sinking.v1', 'arms': {}, 'comparisons': {},
        'historical_structure_sha256': hashlib.sha256(historical_structure.read_bytes()).hexdigest(),
        'nonclaims': [
            'Disabling the pass affects the whole module; no isolated four-fusion causal proof.',
            'Exported IR may compile separately; no executed-machine-code equivalence.',
            'No runtime flag change, convergence, production cost, GPU or current-source admission.']}
    arrays, short_records, parsed = {}, {}, {}
    for arm, mode in ARMS:
        directory = found[(arm, mode)]
        label = f'{arm}_{mode}'
        report, arrays[label], differences, callback_result = verify_saved(directory, arm, mode)
        short_records[label] = json.loads((directory / 'locator-result.json').read_text())
        with (directory / 'optimized-context.hlo.txt').open() as stream:
            parsed[label] = computations(stream)
        variants = collections.Counter(signatures[row['normalized_sha256']]
            for row in parsed[label] if row['normalized_sha256'] in signatures)
        result['arms'][label] = {'run': directory.name, 'control_run': CONTROLS[arm],
            'manifest_sha256': hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest(),
            'report_sha256': hashlib.sha256((directory / 'dz5-snapshot-import.json').read_bytes()).hexdigest(),
            'optimized_hlo_sha256': report['optimized_hlo_sha256'],
            'optimized_hlo_bytes': report['optimized_hlo_bytes'],
            'xla_flags': report['compiler_intervention']['xla_flags'],
            'optimizer_callback_batches': report['optimizer_callback_batches'],
            'historical_record_differences': differences,
            'historical_callback_comparison': callback_result,
            'identified_fusion_counts': dict(variants),
            'computation_count': len(parsed[label]),
            'opaque_constant_count': sum(len(row['opaque_constant_lines']) for row in parsed[label])}
    for left, right in (('original_control', 'candidate_control'),
            ('original_disabled', 'candidate_disabled'),
            ('original_disabled', 'candidate_control'),
            ('original_control', 'original_disabled'), ('candidate_control', 'candidate_disabled')):
        structural = comparison(parsed[left], parsed[right])
        result['comparisons'][f'{left}__{right}'] = {
            'record_differences': records.record_differences(short_records[left], short_records[right]),
            'callbacks': callbacks_comparison(arrays[left], arrays[right]),
            'structure': structural}
    save(request, 'dz5-constant-sinking-readback.json', result)
