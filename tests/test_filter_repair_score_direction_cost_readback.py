"""Matched-source owner cost readback; retain the descriptive-only boundary."""

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_score_directions_readback import ROOT, _latest


def test_score_direction_cost_readback(request):
    comparisons = []
    for case in ('ledh_diagnostics', 'resampling_kdm'):
        rows = {}
        for arm in ('python_reference', 'enclosing'):
            number, run = _latest(f'score_direction_cost_{case}_{arm}_cpu')
            assert run['state'] == 'passed' and run['device'] == 'CPU'
            assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
            result = _load(number, 'score-direction-cost.json')
            assert result['case'] == case and result['arm'] == arm
            assert result['trace_count'] == 1 and result['factory_reused_same_owner']
            assert len(result['warm_seconds']) == 30 and min(result['warm_seconds']) > 0
            assert result['cold_total_seconds'] == result['setup_seconds']+result['cold_call_seconds']
            for name, digest in run['source_sha256'].items():
                if name.startswith('bayesfilter/'):
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
            provenance = _provenance(number)
            assert provenance['cuda_visible_devices'] == '-1'
            assert provenance['trust_basis'] == 'explicit_cpu_reference'
            rows[arm] = {'run': number, 'manifest': run, 'result': result, 'provenance': provenance}
        a, b = rows['python_reference'], rows['enclosing']
        assert a['manifest']['environment'] == b['manifest']['environment']
        assert a['provenance'] == b['provenance']
        left, right = a['result'], b['result']
        for key in ('fixture_sha256', 'source_sha256', 'cpu_affinity', 'extra_environment'):
            assert left[key] == right[key], key
        error = 0.
        for x, y in zip(left['flattened_output'], right['flattened_output'], strict=True):
            assert x['dtype'] == y['dtype'] and x['shape'] == y['shape']
            lhs, rhs = np.asarray(x['value']), np.asarray(y['value'])
            if x['dtype'].startswith('float'):
                np.testing.assert_allclose(lhs, rhs, atol=1e-9, rtol=1e-9)
                error = max(error, float(np.max(np.abs(lhs-rhs), initial=0)))
            else:
                np.testing.assert_array_equal(lhs, rhs)
        arms = {arm: {'run': row['run'], **{key: row['result'][key] for key in (
            'setup_seconds', 'cold_call_seconds', 'cold_total_seconds', 'warm_median_seconds',
            'rss', 'primary_peak_rss_bytes')}} for arm, row in rows.items()}
        comparisons.append({'case': case, 'arms': arms, 'maximum_shared_output_error': error,
            'enclosing_over_prior_warm_ratio': right['warm_median_seconds']/left['warm_median_seconds'],
            'enclosing_over_prior_cold_ratio': right['cold_total_seconds']/left['cold_total_seconds'],
            'warm_rss_difference_bytes': right['rss']['after_warm']-left['rss']['after_warm'],
            'role': 'one_process_per_arm_descriptive_screen_no_statistical_ranking'})
    report = {'schema': 'filter_repair_score_direction_cost_summary.v1', 'comparisons': comparisons,
        'performance_accepted': False, 'gpu_cost_and_capacity_checked': False,
        'nonclaims': ['No statistically supported ranking, universal memory/eviction claim or master completion.']}
    (Path(request.config.getoption('xmlpath')).parent/'score-direction-cost-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
