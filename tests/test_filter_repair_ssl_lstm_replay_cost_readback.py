"""Independent saved-evidence readback; no filtering or gradient kernel runs."""

import hashlib
import itertools
import json
import math
import statistics
import subprocess
from pathlib import Path

import pytest

from scripts.filter_repair_cost_provenance import (
    require_same_physical_gpu,
    validate_cost_device,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
BASELINE = 'c4950a827'
SOURCE = 'bayesfilter/nonlinear/ssl_lstm_zhaocui_fixed_adapter.py'


def numerical_comparison(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        errors = [numerical_comparison(actual[k], expected[k]) for k in expected]
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        errors = [numerical_comparison(a, b) for a, b in zip(actual, expected, strict=True)]
    else:
        assert math.isfinite(actual) and math.isfinite(expected)
        error = abs(actual-expected)
        assert error <= 1e-10+1e-10*abs(expected)
        return error
    return max(errors, default=0.)


def paired_ratio(values):
    logs = [math.log(v) for v in values]
    mean = statistics.mean(logs)
    radius = 4.302652729911275*statistics.stdev(logs)/math.sqrt(3)
    return {'ratios': values, 'geometric_mean': math.exp(mean),
            'two_sided_95_percent_interval': [math.exp(mean-radius), math.exp(mean+radius)]}


@pytest.mark.parametrize('device', ['CPU', 'GPU'])
def test_saved_complete_owner_costs(device, request):
    rows = {}
    witnesses = []
    baseline_digest = hashlib.sha256(subprocess.check_output(
        ['git', 'show', f'{BASELINE}:{SOURCE}'], cwd=ROOT)).hexdigest()
    for directory in sorted(RAW.glob('run-*')):
        if int(directory.name[4:]) <= 4940:
            continue
        payload_path = directory/'ssl-lstm-replay-cost.json'
        if not payload_path.exists():
            continue
        run = json.loads((directory/'run.json').read_text())
        if run['state'] != 'passed' or run['device'] != device:
            continue
        result = json.loads(payload_path.read_text())
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest
               for p, digest in result['source_sha256'].items()):
            continue
        assert result['baseline'] == BASELINE and result['baseline_sha256'] == baseline_digest
        assert result['device'] == device and result['tf32_enabled'] is False
        assert result['exact_replay'] is True
        assert len(result['warm_seconds']) == 30
        assert result['warm_median_seconds'] == statistics.median(result['warm_seconds'])
        assert all(math.isfinite(v) and v > 0 for v in [result['cold_seconds'], *result['warm_seconds']])
        physical = validate_cost_device(run, result['worker_provenance'], result['device_observation'])
        key = (result['mode'], result['arm'], result['horizon'], run['key'][6])
        expected_group = f'ssl_lstm_replay_cost_{key[0]}_{key[1]}_t{key[2]}_{device.lower()}'
        assert run['key'][1] == expected_group
        assert result['actual_jit_compile'] == (key[0] == 'xla' or key[:2] == ('default', 'after'))
        rows[key] = (directory, result, physical)
    expected_keys = set(itertools.product(('default', 'graph', 'xla'), ('before', 'after'), (2, 8), range(3)))
    assert rows.keys() == expected_keys, sorted(expected_keys-rows.keys())
    require_same_physical_gpu([v[2] for v in rows.values()])
    pairs = []
    for mode, horizon in itertools.product(('default', 'graph', 'xla'), (2, 8)):
        warm, cold, rss, peak, drift, errors = [], [], [], [], [], []
        for repeat in range(3):
            a_dir, after, _ = rows[(mode, 'after', horizon, repeat)]
            b_dir, before, _ = rows[(mode, 'before', horizon, repeat)]
            assert after['inputs'] == before['inputs'] and after['rng_stream'] == before['rng_stream']
            errors.append(numerical_comparison(after['numerical_result'], before['numerical_result']))
            warm.append(after['warm_median_seconds']/before['warm_median_seconds'])
            cold.append(after['cold_seconds']/before['cold_seconds'])
            rss.append(after['memory']['after_warm']['VmRSS']-before['memory']['after_warm']['VmRSS'])
            peak.append(after['memory']['after_warm']['VmHWM']-before['memory']['after_warm']['VmHWM'])
            drift.append(after['memory']['after_warm']['VmRSS']-after['memory']['after_cold']['VmRSS'])
            for directory in (a_dir, b_dir):
                witnesses.append({'run': directory.name,
                    'run_sha256': hashlib.sha256((directory/'run.json').read_bytes()).hexdigest(),
                    'result_sha256': hashlib.sha256((directory/'ssl-lstm-replay-cost.json').read_bytes()).hexdigest()})
        timing = paired_ratio(warm)
        pairs.append({'mode': mode, 'horizon': horizon, 'warm': timing,
            'cold': paired_ratio(cold), 'host_rss_difference_bytes': rss,
            'host_hwm_difference_bytes': peak, 'candidate_cold_to_warm_rss_bytes': drift,
            'maximum_numerical_absolute_error': max(errors),
            'warm_10_percent_trigger': timing['two_sided_95_percent_interval'][1] > 1.10,
            'host_64mib_trigger': max(rss) > 64*1024**2})
    for mode, repeat in itertools.product(('default', 'graph', 'xla'), range(3)):
        small = rows[(mode, 'after', 2, repeat)][1]['graph_node_count']
        large = rows[(mode, 'after', 8, repeat)][1]['graph_node_count']
        assert small == large, (mode, small, large)
    report = {'schema': 'filter_repair.ssl_lstm_replay_cost_readback.v1',
        'device': device, 'complete_device_cohort': True, 'workers': len(rows),
        'pairs': pairs, 'witnesses': witnesses,
        'timing_convention': 'Paired geometric means; Student-t interval on three log ratios, df=2; no multiple-comparison claim.',
        'terminal_resource_acceptance': False,
        'nonclaims': ['Independent numerical/device/integrity readback does not waive measured resource triggers.',
                      'This is one declared fixture family, not whole-program or scientific admission.']}
    output = Path(request.config.getoption('xmlpath')).parent/f'ssl-lstm-replay-cost-readback-{device.lower()}.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


@pytest.mark.parametrize('device', ['CPU', 'GPU'])
def test_saved_lifetime_result(device):
    witnesses = []
    for directory in sorted(RAW.glob('run-*')):
        path = directory/'ssl-lstm-replay-lifetime.json'
        if not path.exists():
            continue
        run = json.loads((directory/'run.json').read_text())
        result = json.loads(path.read_text())
        if run['state'] != 'passed' or run['device'] != device:
            continue
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest
               for p, digest in result['source_sha256'].items()):
            continue
        assert run['key'][1] == f'ssl_lstm_replay_lifetime_{device.lower()}'
        assert result['device'] == device and result['tf32_enabled'] is False
        validate_cost_device(run, result['worker_provenance'], result['device_observation'])
        assert result['calls'] == 2000 and result['exact_replay']
        assert result['trace_count'] == 1
        assert result['retained_owner'] == {'exact_replay': True, 'traces_before': 1, 'traces_after': 1}
        snapshots = result['memory']['snapshots']
        growth = snapshots['2000']['VmRSS'] - snapshots['1000']['VmRSS']
        assert growth == result['late_1000_call_rss_growth_bytes'] <= 16 * 1024**2
        capacity = result['capacity']
        assert len(capacity) == 20
        assert all(row['cache']['currsize'] <= 16 for row in capacity)
        assert capacity[-1]['cache']['currsize'] == 16
        capacity_growth = max(row['memory']['VmRSS'] for row in capacity) - result['memory']['capacity_before']['VmRSS']
        assert capacity_growth == result['capacity_growth_bytes'] <= 2 * 1024**3
        assert result['cache_after_clear']['currsize'] == 0
        if device == 'GPU':
            assert snapshots['1000']['allocator'] == snapshots['2000']['allocator']
            assert max(row['memory']['allocator']['peak'] for row in capacity) <= 256 * 1024**2
        witnesses.append(directory.name)
    assert witnesses, f'No qualified current-source {device} lifetime witness'
