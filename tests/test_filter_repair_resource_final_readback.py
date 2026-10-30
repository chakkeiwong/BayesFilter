"""Independent saved-record comparison for final public-owner resource checks."""

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
CASES = ('angle3', 'angle23', 'sqmc_iid', 'sqmc_halton', 'trace_static', 'trace_dynamic')


def equal(a, b, bound):
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key], bound)
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y, bound)
    elif a is None or isinstance(a, (str, bool)):
        assert a == b
    else:
        assert math.isfinite(a) and math.isfinite(b)
        assert abs(a-b) <= bound+bound*abs(b), (a, b, bound)


def interval(pairs):
    logs = [math.log(p['warm_ratio']) for p in pairs]
    mean = statistics.mean(logs)
    half = 4.30265272975*statistics.stdev(logs)/math.sqrt(3)
    return {'geometric_mean_warm_ratio': math.exp(mean),
            'paired_log_t_95_interval': [math.exp(mean-half), math.exp(mean+half)]}


def test_final_owner_resource_readback(request):
    harness = 'tests/test_filter_repair_resource_final.py'
    old_path = ROOT/'docs/plans/artifacts/filter-gradient-repair-20260917/final-resource-harness-05249-05255.py'
    old_tree, new_tree = [ast.parse(p.read_text()) for p in (old_path, ROOT/harness)]
    for name in ('arms', 'synchronized', 'compare', 'rotation_fixture', 'angle_owner',
                 'sqmc_owner', 'build', 'owner_records', 'environment', 'test_public_cost', 'test_public_reuse'):
        nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for tree in (old_tree, new_tree)]
        assert ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False)
    costs, lifetimes, failures, uuids = {}, {}, [], []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 5248:
            continue
        run = json.loads(path.read_text())
        group = run['key'][1]
        if not group.startswith(('resource_acceptance_final_cost_', 'resource_acceptance_final_reuse_')):
            continue
        if run['state'] != 'passed':
            failures.append({'run': path.parent.name, 'state': run['state'], 'group': group})
            continue
        filename = 'final-resource-cost.json' if '_cost_' in group else 'final-resource-reuse.json'
        record = json.loads((path.parent/filename).read_text())
        provenance = next(json.loads(line) for line in (path.parent/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version":'))
        uuids.append(validate_cost_device(run, provenance, record['device_observation']))
        for source, digest in run['source_sha256'].items():
            if source in (harness, 'scripts/run_filter_repair_campaign.py', 'tests/test_filter_repair_resource_final_readback.py'):
                continue
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
        historical = int(path.parent.name[4:]) <= 5255
        assert run['source_sha256'][harness] == hashlib.sha256((old_path if historical else ROOT/harness).read_bytes()).hexdigest()
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        exited = run['process_exit_observation']
        assert not exited['errors'] and not exited['proc_entry_present']
        assert not any(p['pid'] == exited['pid'] for p in exited['gpu_processes'])
        assert record['exact_replay'] and record['owners']
        assert all(owner['trace_count'] == 1 and not owner['host_callbacks'] for owner in record['owners'])
        row = {'run': path.parent.name, 'record': record, 'manifest': run}
        if '_cost_' in group:
            costs[(record['case'], record['arm'], record['pair'])] = row
        else:
            lifetimes[(record['case'], record['device'])] = row
    require_same_physical_gpu(uuids)
    expected = {(case, arm, pair) for case in CASES
                for arm in (('pre_angle', 'pre_guard', 'current') if case.startswith('angle') else ('before', 'current'))
                for pair in range(3)}
    assert set(costs) == expected
    assert set(lifetimes) == {(case, device) for case in CASES for device in ('CPU', 'GPU')}
    comparisons = []
    for case in CASES:
        comparisons_for_case = [('pre_angle', 'pre_guard'), ('pre_guard', 'current'), ('pre_angle', 'current')] if case.startswith('angle') else [('before', 'current')]
        for first_arm, second_arm in comparisons_for_case:
            pairs = []
            for pair in range(3):
                before, after = [costs[(case, arm, pair)] for arm in (first_arm, second_arm)]
                a, b = before['record'], after['record']
                assert before['manifest']['environment'] == after['manifest']['environment']
                assert a['environment'] == b['environment']
                assert a['metadata']['fixture_sha256'] == b['metadata']['fixture_sha256']
                assert a['metadata']['bound'] == b['metadata']['bound']
                equal(a['initial'], b['initial'], a['metadata']['bound'])
                equal(a['changed'], b['changed'], a['metadata']['bound'])
                for result in (a, b):
                    assert len(result['warm_seconds']) == 30 and min(result['warm_seconds']) > 0
                    assert statistics.median(result['warm_seconds']) == result['warm_median_seconds']
                pairs.append({'pair': pair, 'before_run': before['run'], 'after_run': after['run'],
                    'warm_ratio': b['warm_median_seconds']/a['warm_median_seconds'],
                    'before_warm_seconds': a['warm_median_seconds'], 'after_warm_seconds': b['warm_median_seconds'],
                    'before_cold_seconds': a['cold_seconds'], 'after_cold_seconds': b['cold_seconds'],
                    'extra_rss_bytes': b['memory']['warm']['VmRSS']-a['memory']['warm']['VmRSS']})
            comparisons.append({'case': case, 'before_arm': first_arm, 'after_arm': second_arm,
                                'pairs': pairs, **interval(pairs)})
    reuse = []
    for (case, device), row in lifetimes.items():
        result = row['record']
        assert result['calls'] == 256 and result['late_128_call_rss_growth_bytes'] <= 16*1024**2
        if device == 'GPU':
            memory = result['memory']
            assert memory['128']['allocator']['current'] == memory['256']['allocator']['current']
            assert memory['256']['allocator']['peak'] <= 2*1024**3
        reuse.append({'case': case, 'device': device, 'run': row['run'],
            'late_128_call_rss_growth_bytes': result['late_128_call_rss_growth_bytes'],
            'cache_size': result['cache_size'], 'final_memory': result['memory']['256']})
    report = {'schema': 'filter_repair.final_resources_readback.v1', 'comparisons': comparisons,
        'lifetimes': reuse, 'failed_workers_preserved': failures,
        'nonclaims': ['No comparison with inaccurate saved D23 original angles.',
                     'No universal runtime, native eviction, arbitrary capacity or canonical LEDH claim.']}
    (Path(request.config.getoption('xmlpath')).parent/'final-resource-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')
