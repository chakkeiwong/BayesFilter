"""Optimized IR diagnostic for unchanged one-iteration numerical controls."""

import hashlib
import json

import numpy as np
import pytest

from tests import test_filter_repair_dz5_locator_derived_replay as records
from tests import test_filter_repair_dz5_locator_one_iteration as probe
from tests import test_filter_repair_dz5_locator_reporting_storage as storage
from tests.test_filter_repair_geometry_control import save

CONTROLS = {'original': 4635, 'candidate': 4636, 'replay_int32': 4651}


def child_source(arm):
    child = storage.child_source('replays') if arm == 'replay_int32' else probe.child_source(arm)
    anchor = "    report['context_hlo_bytes'] = len(hlo.encode())\n"
    assert child.count(anchor) == 1
    addition = '''    report['optimized_export_peak_rss_before_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    export_tick = time.monotonic()
    optimized = compiled.experimental_get_compiler_ir(*arguments)(stage='optimized_hlo')
    report['optimized_export_seconds'] = time.monotonic() - export_tick
    (OUT / 'optimized-context.hlo.txt').write_text(optimized)
    report['optimized_hlo_sha256'] = hashlib.sha256(optimized.encode()).hexdigest()
    report['optimized_hlo_bytes'] = len(optimized.encode())
    report['optimized_export_peak_rss_after_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['optimized_export_nonclaim'] = 'IR export is diagnostic and may compile separately; no observed-machine-code equivalence or production memory/timing claim.'
'''
    child = child.replace(anchor, anchor + addition)
    compile(child, '<optimized_context>', 'exec')
    return child


@pytest.mark.parametrize('arm', tuple(CONTROLS))
def test_optimized_export(request, arm):
    report = probe.context.trajectory.snapshot_test.run_isolated_snapshot(request,
        snapshot=probe.context.trajectory.SNAPSHOT, child=child_source(arm),
        scope=f'CPU_optimized_HLO_{arm}_historical_context_only', child_timeout_seconds=870,
        read_only_paths=(probe.context.trajectory.consumer.RAW,))
    assert report['optimized_hlo_bytes'] > 0 and report['optimized_export_seconds'] >= 0
    assert report['trace_count'] == 1 and not report['host_callbacks']
    assert report['optimizer_callback_batches'] == 3


def test_saved_optimized_controls(request):
    root = probe.context.trajectory.consumer.RAW
    found = {}
    for path in sorted(root.glob('run-*/run.json')):
        run = json.loads(path.read_text())
        for arm in CONTROLS:
            if run['key'][1] == f'dz5_locator_optimized_{arm}_cpu' and run['state'] == 'passed':
                assert arm not in found, 'Repeated successful arm needs explicit disposition'
                assert run['device'] == 'CPU'
                assert run['environment']['CUDA_VISIBLE_DEVICES'] == '-1'
                found[arm] = path.parent
    assert found.keys() == CONTROLS.keys()
    results = {}
    for arm, directory in found.items():
        previous = root / f'run-{CONTROLS[arm]:05d}'
        report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        prior = json.loads((previous / 'dz5-snapshot-import.json').read_text())
        for key in ('snapshot_manifest_sha256', 'probe_dispatch_sha256',
                'original_locator_sha256', 'executed_original_source_sha256'):
            assert report[key] == prior[key]
        assert report['trace_count'] == 1 and not report['host_callbacks']
        assert report['optimizer_callback_batches'] == prior['optimizer_callback_batches'] == 3
        digests = {}
        for filename, field in (('locator-callbacks.npz', 'callbacks_sha256'),
                ('first-objective-context.hlo.txt', 'context_hlo_sha256'),
                ('optimized-context.hlo.txt', 'optimized_hlo_sha256')):
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            assert report[field] == digest
            digests[filename] = digest
        assert prior['callbacks_sha256'] == hashlib.sha256((previous / 'locator-callbacks.npz').read_bytes()).hexdigest()
        with np.load(directory / 'locator-callbacks.npz', allow_pickle=False) as current, np.load(
                previous / 'locator-callbacks.npz', allow_pickle=False) as old:
            assert set(current.files) == set(old.files) == {'positions', 'values', 'scores', 'valid'}
            for key in current.files:
                assert current[key].dtype == old[key].dtype and current[key].shape == old[key].shape
                assert current[key].tobytes() == old[key].tobytes(), (arm, key)
        actual_record = json.loads((directory / 'locator-result.json').read_text())
        prior_record = json.loads((previous / 'locator-result.json').read_text())
        assert not records.record_differences(actual_record, prior_record), arm
        if arm == 'replay_int32':
            assert report['counter_context_patch'] == prior['counter_context_patch']
        results[arm] = {'run': directory.name, 'control': previous.name,
            'exact_all_callbacks_and_records': True, 'digests': digests,
            'optimized_hlo_bytes': report['optimized_hlo_bytes'],
            'export_seconds': report['optimized_export_seconds'],
            'export_peak_rss_before_kib': report['optimized_export_peak_rss_before_kib'],
            'export_peak_rss_after_kib': report['optimized_export_peak_rss_after_kib']}
    short_records = {arm: json.loads((directory / 'locator-result.json').read_text())
        for arm, directory in found.items()}
    differences = {f'{a}__{b}': records.record_differences(short_records[a], short_records[b])
        for a, b in (('original', 'candidate'), ('original', 'replay_int32'),
            ('candidate', 'replay_int32'))}
    save(request, 'dz5-optimized-controls-readback.json', {
        'schema': 'filter_dz5_optimized_controls.v1', 'arms': results,
        'short_record_difference_paths': differences,
        'short_record_difference_counts': {key: len(paths) for key, paths in differences.items()},
        'nonclaims': ['Exported IR is explanatory; no runtime remedy, optimized-machine-code, GPU or full-trajectory equivalence.']})
