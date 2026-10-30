"""Recompute matched fitted-APF cost summaries from complete saved outputs."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from tests.test_filter_repair_fitted_apf_fixed import FIXTURE, FIXTURES
from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import ROOT, _latest


def _compare(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        return max((_compare(left[key], right[key]) for key in left), default=0.)
    if isinstance(left, list):
        assert len(left) == len(right)
        return max((_compare(a, b) for a, b in zip(left, right)), default=0.)
    if isinstance(left, float):
        assert np.isfinite(left) and np.isfinite(right)
        np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-10)
        return abs(left-right)
    assert left == right
    return 0.


def test_fitted_apf_cost_readback(request):
    comparisons = []
    for model in ('gaussian', 'nonlinear_scalar'):
        rows = {}
        for arm in ('original', 'enclosing'):
            number, run = _latest(f'fitted_apf_cost_{model}_{arm}_cpu')
            assert run['state'] == 'passed' and run['device'] == 'CPU'
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            result = _load(number, 'fitted-apf-cost.json')
            assert result['model'] == model and result['arm'] == arm
            assert result['fixture_sha256'] == hashlib.sha256((FIXTURES/'fixture.json').read_bytes()).hexdigest()
            assert result['reference_sources'] == FIXTURE['baseline_sha256']
            assert result['factory_reused_same_owner'] and result['trace_count'] == 1
            assert result['subkernel_calls'] == 5
            assert len(result['warm_seconds']) == 30 and min(result['warm_seconds']) > 0
            assert result['warm_median_seconds'] == statistics.median(result['warm_seconds'])
            assert result['cold_total_seconds'] == result['setup_seconds'] + result['cold_call_seconds']
            for name, source_hash in run['source_sha256'].items():
                if name.startswith(('bayesfilter/', 'tests/test_filter_repair_fitted_apf')):
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == source_hash, (number, name)
            provenance = _provenance(number)
            assert provenance['cuda_visible_devices'] == '-1'
            assert provenance['trust_basis'] == 'explicit_cpu_reference'
            assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
            assert set(result['memory']) == {'before', 'after_setup', 'after_cold', 'after_warm', 'after_collection'}
            for memory in result['memory'].values():
                assert min(memory['status']['VmRSS'], memory['smaps_rollup']['Rss'], memory['rusage_maxrss_bytes']) > 0
                assert memory['read_order'] == 'status, smaps_rollup, rusage; not atomic'
                assert memory['read_wall_seconds'] > 0
            rows[arm] = {'run': number, 'manifest': run, 'provenance': provenance, 'result': result}
        a, b = rows['original'], rows['enclosing']
        assert a['manifest']['source_sha256'] == b['manifest']['source_sha256']
        assert a['manifest']['environment'] == b['manifest']['environment']
        assert a['provenance'] == b['provenance']
        left, right = a['result'], b['result']
        for key in ('fixture_sha256', 'cpu_affinity', 'extra_environment', 'platform', 'kernel_release'):
            assert left[key] == right[key]
        left_fields, right_fields = dict(left['diagnostics']), dict(right['diagnostics'])
        assert right_fields.pop('fit_execution') == 'enclosing_tensorflow_loop'
        assert right_fields.pop('fit_enclosing_calls') == 1
        from bayesfilter.score_study.contracts import digest
        for fields in (left_fields, right_fields):
            assert fields.pop('fit_digest') == digest(fields['fit'])
        error = max(_compare(left['final'], right['final']), _compare(left_fields, right_fields))
        warm_ratio = right['warm_median_seconds']/left['warm_median_seconds']
        cold_ratio = right['cold_total_seconds']/left['cold_total_seconds']
        rss = right['memory']['after_warm']['status']['VmRSS'] - left['memory']['after_warm']['status']['VmRSS']
        comparisons.append({'model': model, 'maximum_shared_output_error': error,
            'arms': {arm: {'run': row['run'], **{key: row['result'][key] for key in
                ('setup_seconds', 'cold_call_seconds', 'cold_total_seconds', 'warm_median_seconds', 'memory')}}
                for arm, row in rows.items()},
            'warm_ratio': warm_ratio, 'cold_ratio': cold_ratio, 'extra_warm_rss_bytes': rss,
            'attribution_triggers': {'warm_over_1_10': warm_ratio > 1.10, 'cold_over_1_25': cold_ratio > 1.25,
                                     'extra_rss_over_64_mib': rss > 64*1024**2}})
    summary = {'schema': 'filter_repair_fitted_apf_cost_summary.v1', 'comparisons': comparisons,
               'performance_accepted': False, 'gpu_cost_and_capacity_checked': False,
               'scope': 'One fresh CPU process per arm; descriptive accounting only, no statistical ranking.'}
    (Path(request.config.getoption('xmlpath')).parent/'fitted-apf-cost-readback.json').write_text(
        json.dumps(summary, indent=2)+'\n')
