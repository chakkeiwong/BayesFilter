"""Independent saved-output, provenance and paired-cost acceptance diagnostics."""

import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)
from tests.test_filter_repair_gaussian_binding import compare
from tests.test_filter_repair_resource_owners import CASES

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_saved_owner_resources(request):
    runs = []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 5202:
            continue
        run = json.loads(path.read_text())
        if run['key'][1].startswith(('resource_acceptance_cost_', 'resource_acceptance_reuse_')):
            runs.append((path.parent, run))
    selected = {}
    failures = []
    for directory, run in runs:
        if run['state'] != 'passed':
            failures.append({'run': directory.name, 'state': run['state'], 'group': run['key'][1]})
            continue
        selected[run['key'][1]] = (directory, run)
    expected_groups = {f'resource_acceptance_cost_{case}_{pair}_{arm}_gpu'
        for case in CASES for pair in range(3) for arm in ('before', 'after')}
    expected_groups |= {f'resource_acceptance_reuse_{case}_{device}'
        for case in CASES for device in ('cpu', 'gpu')}
    assert set(selected) == expected_groups
    records, uuids = {}, []
    for group, (directory, run) in selected.items():
        filename = 'resource-owner-cost.json' if '_cost_' in group else 'resource-owner-lifetime.json'
        record = json.loads((directory/filename).read_text())
        provenance = next(json.loads(line) for line in (directory/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        uuids.append(validate_cost_device(run, provenance, record['device_observation']))
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        for path, digest in run['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        exit_record = run['process_exit_observation']
        assert not exit_record['proc_entry_present'] and not exit_record['errors']
        assert not any(row['pid'] == exit_record['pid'] for row in exit_record['gpu_processes'])
        records[group] = {'run': directory.name, 'manifest': run, 'record': record}
    require_same_physical_gpu(uuids)
    summaries = []
    for case in CASES:
        pairs = []
        for pair in range(3):
            rows = [records[f'resource_acceptance_cost_{case}_{pair}_{arm}_gpu'] for arm in ('before', 'after')]
            before, after = [row['record'] for row in rows]
            assert rows[0]['manifest']['source_sha256'] == rows[1]['manifest']['source_sha256']
            assert rows[0]['manifest']['environment'] == rows[1]['manifest']['environment']
            for row in (before, after):
                assert len(row['warm_seconds']) == 30 and min(row['warm_seconds']) > 0
                assert statistics.median(row['warm_seconds']) == row['warm_median_seconds']
            if case.startswith('fitted_'):
                left, right = copy.deepcopy(before['diagnostics']), copy.deepcopy(after['diagnostics'])
                assert right.pop('fit_execution') == 'enclosing_tensorflow_loop'
                assert right.pop('fit_enclosing_calls') == 1
                from bayesfilter.score_study.contracts import digest
                for fields in (left, right):
                    assert fields.pop('fit_digest') == digest(fields['fit'])
                error = max(compare(before['final'], after['final'], 2e-10), compare(left, right, 2e-10))
                cold_key = 'cold_total_seconds'
            else:
                error = compare(before['numerical_result'], after['numerical_result'], 1e-10)
                cold_key = 'cold_seconds'
            warm_ratio = after['warm_median_seconds']/before['warm_median_seconds']
            pairs.append({'pair': pair, 'runs': [row['run'] for row in rows], 'output_error': error,
                'warm_ratio': warm_ratio, 'cold_ratio': after[cold_key]/before[cold_key],
                'before_cold_seconds': before[cold_key], 'after_cold_seconds': after[cold_key],
                'before_warm_seconds': before['warm_median_seconds'],
                'after_warm_seconds': after['warm_median_seconds'],
                'warm_rss_difference_bytes': after['memory']['after_warm']['VmRSS']-before['memory']['after_warm']['VmRSS'],
                'allocator_before': before['memory']['after_warm']['allocator'],
                'allocator_after': after['memory']['after_warm']['allocator']})
        logs = [math.log(row['warm_ratio']) for row in pairs]
        mean = statistics.mean(logs)
        half = 4.30265272975*statistics.stdev(logs)/math.sqrt(3)
        summaries.append({'case': case, 'pairs': pairs,
            'geometric_mean_warm_ratio': math.exp(mean),
            'paired_log_t_95_interval': [math.exp(mean-half), math.exp(mean+half)],
            'interval_limitation': 'Three independent process pairs; log-normal approximation with two degrees of freedom.'})
    lifetimes = []
    for case in CASES:
        for device in ('cpu', 'gpu'):
            item = records[f'resource_acceptance_reuse_{case}_{device}']
            row = item['record']
            assert row['calls'] == 1024 and row['same_input_exact_replay']
            assert row['owner_count'] == 1 and row['trace_counts'] == [1]
            assert row['cache_before']['currsize'] == row['cache_after']['currsize']
            assert row['changed_seed_and_theta_witness'][0] != row['changed_seed_and_theta_witness'][1]
            assert row['late_512_call_rss_growth_bytes'] <= 16*1024**2
            if device == 'gpu':
                a, b = [row['memory']['snapshots'][str(index)]['allocator'] for index in (512, 1024)]
                assert a['current'] == b['current'] and b['peak'] <= 2*1024**3
            lifetimes.append({'run': item['run'], 'case': case, 'device': device,
                'late_512_call_rss_growth_bytes': row['late_512_call_rss_growth_bytes'],
                'cache_before': row['cache_before'], 'cache_after': row['cache_after'],
                'memory': row['memory']})
    output = {'schema': 'filter_repair.resource_owner_readback.v1', 'comparisons': summaries,
        'lifetimes': lifetimes, 'preserved_failed_workers': failures,
        'nonclaims': ['Engineering interpretation requires review of cold/RSS tradeoffs and intervals.',
                     'No universal speed, memory eviction, canonical LEDH, posterior or HMC claim.']}
    (Path(request.config.getoption('xmlpath')).parent/'resource-owners-readback.json').write_text(
        json.dumps(output, indent=2, allow_nan=False)+'\n')
