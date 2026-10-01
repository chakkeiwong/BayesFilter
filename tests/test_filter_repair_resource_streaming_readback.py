"""Paired GPU streaming timing and finite lifetime readback diagnostics."""

import ast
import hashlib
import json
import math
import statistics
from pathlib import Path

from scripts.filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_saved_streaming_resources(request):
    historical = ROOT/'docs/plans/artifacts/filter-gradient-repair-20260917/streaming-resource-harness-05228-05241.py'
    harness = 'tests/test_filter_repair_resource_streaming.py'
    old_tree, current_tree = [ast.parse(path.read_text()) for path in (historical, ROOT/harness)]
    for name in ('_sample', 'test_matched_streaming_cost'):
        a, b = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                for tree in (old_tree, current_tree)]
        assert ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
    rows = {}
    failures = []
    uuids = []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 5227:
            continue
        run = json.loads(path.read_text())
        group = run['key'][1]
        if not group.startswith(('resource_acceptance_stream_cost_', 'resource_acceptance_stream_reuse_')):
            continue
        if run['state'] != 'passed':
            failures.append({'run': path.parent.name, 'group': group, 'state': run['state']})
            continue
        filename = 'streaming-reuse.json' if '_reuse_' in group else 'streaming-paired-cost.json'
        result = json.loads((path.parent/filename).read_text())
        provenance = next(json.loads(line) for line in (path.parent/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        uuids.append(validate_cost_device(run, provenance, result['device_observation']))
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        for name, digest in run['source_sha256'].items():
            if name in (harness, 'scripts/run_filter_repair_campaign.py', __file__.removeprefix(str(ROOT)+'/')):
                continue
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        if int(path.parent.name[4:]) <= 5241:
            assert run['source_sha256'][harness] == hashlib.sha256(historical.read_bytes()).hexdigest()
        else:
            assert run['source_sha256'][harness] == hashlib.sha256((ROOT/harness).read_bytes()).hexdigest()
        exited = run['process_exit_observation']
        assert not exited['proc_entry_present'] and not exited['errors']
        assert not any(p['pid'] == exited['pid'] for p in exited['gpu_processes'])
        rows[group] = {'run': path.parent.name, 'manifest': run, 'result': result}
    require_same_physical_gpu(uuids)
    expected = {f'resource_acceptance_stream_cost_{pair}_{horizon}_{arm}_gpu'
        for pair in range(3) for horizon in (32, 128) for arm in ('buffered', 'streaming')}
    reuse = ['resource_acceptance_stream_reuse_32_gpu', 'resource_acceptance_stream_reuse_128_gpu',
             'resource_acceptance_stream_reuse_128_cpu']
    assert set(rows) == expected | set(reuse)
    comparisons = []
    for horizon in (32, 128):
        pairs = []
        for pair in range(3):
            a, b = [rows[f'resource_acceptance_stream_cost_{pair}_{horizon}_{arm}_gpu']
                    for arm in ('buffered', 'streaming')]
            before, after = a['result'], b['result']
            assert a['manifest']['environment'] == b['manifest']['environment']
            for key in ('shape', 'seeds', 'input_sha256', 'controls', 'buffered_commit',
                        'buffered_source_sha256', 'shared_record', 'streaming_rng_diagnostics', 'affinity', 'threads'):
                assert before[key] == after[key], key
            for r in (before, after):
                assert r['exact_complete_record_agreement'] and r['comparison']['healthy']
                assert r['record']['program_valid'] and r['graph']['trace_count'] == 1
                assert len(r['warm_seconds']) == 30 and min(r['warm_seconds']) > 0
                assert statistics.median(r['warm_seconds']) == r['warm_median_seconds']
            pairs.append({'pair': pair, 'before_run': a['run'], 'after_run': b['run'],
                'before_cold_seconds': before['cold_seconds'], 'after_cold_seconds': after['cold_seconds'],
                'before_warm_seconds': before['warm_median_seconds'], 'after_warm_seconds': after['warm_median_seconds'],
                'warm_ratio': after['warm_median_seconds']/before['warm_median_seconds'],
                'extra_warm_rss_bytes': after['after_warm']['VmRSS']-before['after_warm']['VmRSS'],
                'before_memory': before['after_warm'], 'after_memory': after['after_warm']})
        logs = [math.log(p['warm_ratio']) for p in pairs]
        mean = statistics.mean(logs)
        half = 4.30265272975*statistics.stdev(logs)/math.sqrt(3)
        comparisons.append({'horizon': horizon, 'pairs': pairs,
            'geometric_mean_warm_ratio': math.exp(mean),
            'paired_log_t_95_interval': [math.exp(mean-half), math.exp(mean+half)]})
    lifetimes = []
    for group in reuse:
        row = rows[group]
        r = row['result']
        assert r['calls'] == 128 and r['same_input_exact_replay'] and r['owner_collected']
        assert r['trace_count'] == 1 and r['independent_reference_errors'] == [0., 0.]
        assert r['late_64_call_rss_growth_bytes'] <= 16*1024**2
        assert [c['particles'] for c in r['capacities']] == [8, 16, 32, 64]
        assert all(c['trace_count'] == 1 for c in r['capacities'])
        assert max(c['memory']['VmRSS'] for c in r['capacities'])-r['memory']['capacity_before']['VmRSS'] <= 2*1024**3
        lifetimes.append({'run': row['run'], 'result': r})
    record = {'schema': 'filter_repair.streaming_resource_readback.v1', 'comparisons': comparisons,
        'lifetimes': lifetimes, 'failed_workers_preserved': failures,
        'existing_cpu_stricter_study': {'ratios': [1.08239, 1.09678], 'upper_95': [1.11687, 1.15608],
                                       'threshold': 1.10, 'verdict': 'failed'},
        'nonclaims': ['GPU evidence does not convert the prior stricter CPU verdict to passed.',
                     'No universal timing/capacity, canonical LEDH or HMC claim.']}
    diagnosis = json.loads((RAW/'run-05242/streaming-changed-status.json').read_text())
    assert not diagnosis['comparison']['healthy']
    for result in diagnosis['results'].values():
        assert result['program_valid'] is False
    record['changed_long_horizon_refusal'] = {'run': 5242,
        'diagnostic_sha256': hashlib.sha256((RAW/'run-05242/streaming-changed-status.json').read_bytes()).hexdigest(),
        'baseline_and_candidate_invalid': True}
    (Path(request.config.getoption('xmlpath')).parent/'streaming-resource-readback.json').write_text(
        json.dumps(record, indent=2)+'\n')
