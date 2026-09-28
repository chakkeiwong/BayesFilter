"""Check paired nonlinear cost records and report OS counter disagreements."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_directions import FIXTURE
from tests.test_filter_repair_nonlinear_scope import (
    ROOT,
    _latest,
    _nomination,
    _partition,
    _scope_fixture,
)


def test_nonlinear_direction_cost_readback(request):
    comparisons = []
    for case in ('ukf', 'ledh_diagnostics'):
        if case == 'ukf':
            fixture, nomination = FIXTURE, None
        else:
            nomination, count = _nomination('ledh')
            fixture = _scope_fixture(_partition('untouched')[0]['providers']['ledh'][0], count)
        digest = hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()
        rows = {}
        for arm in ('python_reference', 'enclosing'):
            number, run = _latest(f'nonlinear_direction_cost_{case}_{arm}_cpu')
            assert run['state'] == 'passed' and run['device'] == 'CPU'
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            result = _load(number, 'nonlinear-direction-cost.json')
            assert result['case'] == case and result['arm'] == arm
            assert result['fixture_sha256'] == digest and result['nomination_run'] == nomination
            assert result['trace_count'] == 1 and result['factory_reused_same_owner']
            assert len(result['warm_seconds']) == 30 and min(result['warm_seconds']) > 0
            assert result['warm_median_seconds'] == statistics.median(result['warm_seconds'])
            assert result['cold_total_seconds'] == result['setup_seconds']+result['cold_call_seconds']
            for name, source_hash in run['source_sha256'].items():
                if name.startswith(('bayesfilter/', 'tests/test_filter_repair_nonlinear')):
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == source_hash, (number, name)
            provenance = _provenance(number)
            assert provenance['cuda_visible_devices'] == '-1'
            assert provenance['trust_basis'] == 'explicit_cpu_reference'
            accounting = {}
            assert set(result['memory']) == {'before', 'after_setup', 'after_cold', 'after_warm', 'after_collection'}
            for stage, memory in result['memory'].items():
                status, smaps = memory['status'], memory['smaps_rollup']
                assert set(status) == {'VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem'}
                assert set(smaps) == {'Rss', 'Pss', 'Anonymous', 'Shared_Clean', 'Private_Clean', 'Private_Dirty'}
                assert all(x >= 0 for x in (*status.values(), *smaps.values()))
                assert min(status['VmRSS'], smaps['Rss'], memory['rusage_maxrss_bytes']) > 0
                assert memory['read_order'] == 'status, smaps_rollup, rusage; not atomic'
                assert memory['read_wall_seconds'] > 0
                # These are observations, not equality assertions: kernel RSS
                # counters and a page-table walk are different measurements.
                accounting[stage] = {
                    'status_minus_smaps_rss_bytes': status['VmRSS']-smaps['Rss'],
                    'rusage_minus_status_rss_bytes': memory['rusage_maxrss_bytes']-status['VmRSS'],
                    'rusage_minus_status_hwm_bytes': memory['rusage_maxrss_bytes']-status['VmHWM'],
                    'status_component_sum_error_bytes': status['VmRSS']-sum(status[k] for k in ('RssAnon', 'RssFile', 'RssShmem')),
                }
            rows[arm] = {'run': number, 'manifest': run, 'result': result, 'provenance': provenance,
                         'counter_differences': accounting}
        a, b = rows['python_reference'], rows['enclosing']
        assert a['manifest']['environment'] == b['manifest']['environment']
        assert a['manifest']['source_sha256'] == b['manifest']['source_sha256']
        assert a['provenance'] == b['provenance']
        left, right = a['result'], b['result']
        for key in ('fixture_sha256', 'nomination_run', 'cpu_affinity', 'extra_environment'):
            assert left[key] == right[key], key
        error = 0.
        for x, y in zip(left['flattened_output'], right['flattened_output'], strict=True):
            assert x['dtype'] == y['dtype'] and x['shape'] == y['shape']
            lhs, rhs = np.asarray(x['value']), np.asarray(y['value'])
            if x['dtype'].startswith('float'):
                assert np.isfinite(lhs).all() and np.isfinite(rhs).all()
                np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
                error = max(error, float(np.max(np.abs(lhs-rhs), initial=0)))
            else:
                np.testing.assert_array_equal(lhs, rhs)
        arms = {arm: {'run': row['run'], 'counter_differences': row['counter_differences'],
                     **{key: row['result'][key] for key in (
                         'setup_seconds', 'cold_call_seconds', 'cold_total_seconds', 'warm_median_seconds', 'memory')}}
                for arm, row in rows.items()}
        comparisons.append({'case': case, 'arms': arms, 'maximum_shared_output_error': error,
            'enclosing_over_prior_warm_ratio': right['warm_median_seconds']/left['warm_median_seconds'],
            'enclosing_over_prior_cold_ratio': right['cold_total_seconds']/left['cold_total_seconds'],
            'warm_status_rss_difference_bytes': right['memory']['after_warm']['status']['VmRSS']-left['memory']['after_warm']['status']['VmRSS'],
            'warm_smaps_rss_difference_bytes': right['memory']['after_warm']['smaps_rollup']['Rss']-left['memory']['after_warm']['smaps_rollup']['Rss'],
            'role': 'one_process_per_arm_descriptive_screen_no_statistical_ranking'})
    report = {'schema': 'filter_repair_nonlinear_direction_cost_summary.v1', 'comparisons': comparisons,
        'performance_accepted': False, 'gpu_cost_and_capacity_checked': False,
        'nonclaims': ['No ranking, atomic peak guarantee, live-allocator measurement, compiler eviction or master completion.']}
    (Path(request.config.getoption('xmlpath')).parent/'nonlinear-direction-cost-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
