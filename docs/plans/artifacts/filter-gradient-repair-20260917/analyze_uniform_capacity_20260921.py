"""Post-run diagnostic analysis of uniform-controller capacity and warm memory."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def equal(actual, expected, path='record'):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            equal(actual[key], expected[key], path + '.' + key)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            equal(left, right, f'{path}[{i}]')
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert abs(actual - expected) <= 1e-10 + 1e-10 * abs(expected), path
    else:
        assert type(actual) is type(expected) and actual == expected, path


parser = argparse.ArgumentParser()
parser.add_argument('--runs', nargs='+', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source, rows, scopes = None, [], set()
for number in args.runs:
    directory = ROOT / f'run-{number:05d}'
    manifest = json.loads((directory / 'run.json').read_text())
    assert manifest['state'] == 'passed'
    assert manifest['key'][1].startswith('uniform_round_growth_')
    source = manifest['source_sha256'] if source is None else source
    assert manifest['source_sha256'] == source
    data = json.loads((directory / 'uniform-round-growth.json').read_text())
    device = manifest['device']
    assert data['gpu'] == (device == 'GPU')
    assert manifest['environment']['CUDA_VISIBLE_DEVICES'] == ('3' if device == 'GPU' else '-1')
    scope = (device, data['round_capacity'])
    assert scope not in scopes
    scopes.add(scope)
    equal(data['result'], data['original'])
    equal(data['changed_result'], data['original_changed'])
    observations = data['observations']
    assert [o['additional_calls'] for o in observations] == [0, 1000, 2000, 3000]
    rss = [o['memory']['host']['VmRSS'] for o in observations]
    stage_rss = {k: v['host']['VmRSS'] for k, v in data['stages'].items()}
    row = dict(run=number, device=device, dimension=data['dimension'],
        round_capacity=data['round_capacity'], graph_nodes=data['graph_nodes'],
        hlo_bytes=data['hlo_bytes'], runtime_operands=data['runtime_operands'],
        build_seconds=data['build_seconds'], trace_seconds=data['trace_seconds'],
        cold_seconds=data['cold_seconds'], stage_rss_bytes=stage_rss,
        warm_rss_bytes=rss, warm_rss_increments=[b-a for a, b in zip(rss, rss[1:])],
        warm_gpu_current_bytes=[o['memory']['gpu']['current'] for o in observations] if data['gpu'] else None,
        complete_original_records_equal=True)
    rows.append(row)
    print(json.dumps(row))
assert scopes == {(device, cap) for device in ('CPU', 'GPU') for cap in (1, 4, 8)}
report = dict(role='descriptive_capacity_and_fixed_target_warm_memory', rows=rows,
    source_sha256=source, analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    nonclaims=['No arbitrary target-turnover or native executable-eviction bound.',
        'Three sparse warm snapshots do not prove exact process peaks or general leak freedom.',
        'Capacity changes only the bound, not the target or algorithm; no statistical ranking.',
        'Whole-public and multistart/DZ5 measurements remain separate.'])
with args.output.open('x') as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write('\n')
