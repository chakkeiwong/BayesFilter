"""Independent readback of current streaming evidence and saved buffered scope."""

import hashlib
import json
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.filter_repair_cost_provenance import validate_cost_device
from tests.test_filter_repair_ledh_seeded_readback import (
    DEPENDENCIES,
    _load,
    _provenance,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
BUFFERED = "c7c0b88c2"


def _check(number, count, *, frozen=False):
    run = _load(number, 'run.json')
    assert run['state'] == 'passed'
    assert run['test_evidence'] == {'passed': True, 'tests': count, 'failure': 0, 'error': 0, 'skipped': 0}
    for relative in DEPENDENCIES:
        source = subprocess.check_output(['git', 'show', f'{BUFFERED}:{relative}'], cwd=ROOT) if frozen else (ROOT / relative).read_bytes()
        assert run['source_sha256'][relative] == hashlib.sha256(source).hexdigest(), relative
    provenance = _provenance(number)
    assert provenance['tf32_enabled'] is True
    assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
    assert provenance['gpu_memory_policy']['all_physical_devices_memory_growth']
    assert provenance['cuda_visible_devices'] == run['environment']['CUDA_VISIBLE_DEVICES']
    if run['device'] == 'GPU':
        assert provenance['trust_basis'] == 'owner_designated_managed_session_visible_gpu_trusted'
        assert len(provenance['gpu_memory_policy']['physical_devices']) == 1
    else:
        assert provenance['trust_basis'] == 'explicit_cpu_reference'
        assert provenance['cuda_visible_devices'] == '-1'
    return run, provenance


def _write(request, name, record):
    path = Path(request.config.getoption('xmlpath')).parent / name
    path.write_text(json.dumps(record, indent=2) + '\n')


def _exit_check(run):
    record = run['process_exit_observation']
    assert record['pid'] == run['worker_pid']
    assert not record['errors'] and not record['proc_entry_present']
    assert all(process['pid'] != record['pid'] for process in record['gpu_processes'])


def test_buffer_freeze_and_current_qualification(request):
    report = []
    for number in (4688, 4689):
        _check(number, 3, frozen=True)
        for case in ('composed', 'annealed', 'dual_trust'):
            row = _load(number, f'buffered-freeze-{case}.json')
            assert row['baseline_commit'] == BUFFERED and len(row['records']) == 2
            assert row['graph']['hlo']['optimized_hlo']['process_shape_occurrences'] > 0
            for relative, digest in row['source_sha256'].items():
                source = subprocess.check_output(['git', 'show', f'{BUFFERED}:{relative}'], cwd=ROOT)
                assert hashlib.sha256(source).hexdigest() == digest
    for number in (4690, 4691):
        run, _ = _check(number, 9)
        for case in ('composed', 'annealed', 'dual_trust', 'invalid_initial', 'invalid_prediction',
                     'first_invalid_prediction', 'time_dependent', 'large_seeds'):
            row = _load(number, f'streamed-{case}.json')
            graph = row['graph']
            assert graph['trace_count'] == 1 and graph['no_host_callbacks']
            assert not graph['process_buffer_shape_nodes']
            assert graph['hlo']['optimized_hlo']['process_shape_occurrences'] == 0
            assert len(row['records']) == 2
            for item in row['records']:
                assert item['exact_final_philox_state']
                assert item['actual']['process_draws_consumed'] == item['original_process_draws']
                assert item['actual']['program_valid'] == item['buffered']['program_valid']
                assert all(error in (0, None) for error in item['buffered_comparison']['max_absolute_errors'].values())
            report.append({'run': number, 'case': case, 'device': run['device'], 'exact_shared_record': True})
    _write(request, 'streaming-qualification-readback.json', {'rows': report,
        'scope': 'bounded execution and random-buffer mechanism; no canonical/scientific admission'})


def test_isolated_memory_and_cost_readback(request):
    reports = []
    for number, arm in ((4692, 'buffered_xla'), (4693, 'streaming_graph'), (4694, 'streaming_xla')):
        run, provenance = _check(number, 1)
        row = _load(number, 'seeded-cost.json')
        assert row['implementation_arm'] == arm and row['numerical_passed']
        assert row['all_three_owners_collected'] == [True, True, True]
        assert len(row['warm_seconds']) == 15 and len(row['one_shot_seconds']) == 2
        assert row['trace_count'] == 1 and row['validation_outside_cost_measurements']
        assert row['jit_compile'] == (arm != 'streaming_graph')
        validate_cost_device(run, provenance, row['cost_provenance'])
        _exit_check(run)
        reports.append({'run': number, 'arm': arm, 'cold_seconds': row['cold_seconds'],
            'warm_median_ms': statistics.median(row['warm_seconds']) * 1000,
            'one_shot_seconds': row['one_shot_seconds'],
            'rss_after_cold_mib': (row['after_cold']['rss_bytes'] - row['before']['rss_bytes']) / 2**20,
            'rss_after_warm_mib': (row['after_warm']['rss_bytes'] - row['before']['rss_bytes']) / 2**20,
            'rss_after_release_mib': (row['after_release']['rss_bytes'] - row['before']['rss_bytes']) / 2**20,
            'allocator': row['after_warm']['allocator']})
    _write(request, 'streaming-cost-readback.json', {'rows': reports,
        'role': 'single fresh process per arm; descriptive only',
        'GPU_costs': 'pending; unshared preflight declined',
        'owner_lifetime': 'Python owners collect; native/compiler residency persists until process exit',
        'nonclaim': 'No unbounded leak proof or statistical speed ranking.'})


def test_cpu_capacity_ladder_readback(request):
    reports = []
    for number, horizon, count, arm in (
        (4695, 3, 8, 'buffered'), (4696, 3, 8, 'streaming'),
        (4697, 3, 64, 'buffered'), (4698, 3, 64, 'streaming'),
        (4699, 32, 64, 'buffered'), (4700, 32, 64, 'streaming'),
        (4701, 128, 64, 'buffered'), (4702, 128, 64, 'streaming')):
        run, provenance = _check(number, 1)
        row = _load(number, f'capacity-{horizon}-{count}-{arm}.json')
        assert row['fresh_process_per_arm_and_point'] and row['owner_collected']
        assert row['arm'] == arm and row['shape'] == {'T': horizon, 'N': count, 'd': 2}
        assert row['buffered_process_bytes'] == horizon * count * 16
        assert row['streaming_draw_bytes'] == count * 16 and row['philox_state_bytes'] == 24
        assert bool(row['graph']['process_buffer_shape_nodes']) == (arm == 'buffered')
        assert (row['graph']['hlo']['optimized_hlo']['process_shape_occurrences'] > 0) == (arm == 'buffered')
        assert row['record']['program_valid'] == row['comparison']['healthy']
        validate_cost_device(run, provenance, row['cost_provenance'])
        _exit_check(run)
        reports.append({'run': number, 'shape': row['shape'], 'arm': arm,
            'healthy': row['record']['program_valid'], 'cold_seconds': row['cold_seconds'],
            'warm_median_ms': statistics.median(row['warm_seconds']) * 1000,
            'rss_after_warm_mib': (row['after_warm']['rss_bytes'] - row['before']['rss_bytes']) / 2**20,
            'allocator': row['after_warm']['allocator'],
            'buffered_process_bytes': row['buffered_process_bytes'],
            'streaming_draw_bytes': row['streaming_draw_bytes']})
    _write(request, 'streaming-capacity-readback.json', {'rows': reports,
        'scope': 'CPU reference only; GPU capacity remains pending',
        'memory_interpretation': 'HLO proves removal of the random buffer; RSS includes compiler/context overhead',
        'nonclaims': ['Rejected trajectories do not qualify numerical speed rankings.',
                      'No production/default capacity or scientific admission.']})


def test_current_numerical_regressions(request):
    failed = _load(4703, 'run.json')
    assert failed['state'] == 'failed' and failed['test_evidence']['failure'] == 1
    failures = [case.attrib['name'] for case in ET.parse(RAW / 'run-04703/junit.xml').iter('testcase')
                if case.find('failure') is not None]
    assert failures == ['test_full_value_fixed_inputs[annealed-3-3-True-0.8-False]']
    rows = []
    for number in (4704, 4705):
        run, _ = _check(number, 43)
        _exit_check(run)
        for case in ('one_step', 'composed', 'annealed', 'dual_trust', 'invalid_initial',
                     'invalid_prediction', 'invalid_observation'):
            directory = RAW / f'run-{number:05d}'
            hlo = (directory / f'value-native-{case}.hlo.txt').read_text()
            changed = (directory / f'value-native-{case}-changed.hlo.txt').read_text()
            row = _load(number, f'value-native-{case}.json')
            assert hashlib.sha256(hlo.encode()).hexdigest() == row['hlo_sha256']
            assert hashlib.sha256(changed.encode()).hexdigest() == row['changed_hlo_sha256']
            stripped = re.sub(r', metadata=\{[^}\n]*\}', '', hlo)
            assert stripped == re.sub(r', metadata=\{[^}\n]*\}', '', changed)
            assert hashlib.sha256(stripped.encode()).hexdigest() == row['semantic_hlo_sha256']
        rows.append({'run': number, 'count': 43, 'device': run['device']})
    _write(request, 'streaming-regression-readback.json', {'rows': rows,
        'preserved_HLO_metadata_failure': failures,
        'existing_F14_static_violation': 'preserved at04682; separate repair remains open',
        'whole_master_complete': False})
