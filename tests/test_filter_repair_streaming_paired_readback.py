"""Independent diagnostic analysis of all fresh-process streaming pairs."""

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from tests.test_filter_repair_ledh_seeded_readback import _provenance

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def test_paired_streaming_complete_cohort(request):
    order = json.loads((ROOT / 'docs/plans/filter_gradient_streaming_paired_order_20260929.json').read_text())
    rows = []
    excluded = []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 4727:
            continue
        run = json.loads(path.read_text())
        if not run['key'][1].startswith('streaming_paired_') or run['key'][1] == 'streaming_paired_readback_cpu':
            continue
        if int(path.parent.name[4:]) == 4728:
            assert run['state'] == 'failed' and run['test_evidence']['failure'] == 1
            failed = json.loads((path.parent / 'streaming-paired-cost.json').read_text())
            assert failed['comparison']['healthy']
            assert all(error == 0 for error in failed['comparison']['max_absolute_errors'].values())
            excluded.append({'run': 4728, 'reason': 'dictionary schema mismatch: candidate adds three RNG diagnostics; all shared errors zero'})
            continue
        assert run['state'] == 'passed', f'Failed measurement needs explicit disposition: {path}'
        assert run['test_evidence'] == {'passed': True, 'tests': 1, 'failure': 0, 'error': 0, 'skipped': 0}
        record = json.loads((path.parent / 'streaming-paired-cost.json').read_text())
        number = int(path.parent.name[4:])
        provenance = _provenance(number)
        assert provenance['cuda_visible_devices'] == '-1'
        assert provenance['trust_basis'] == 'explicit_cpu_reference'
        assert record['fresh_process_per_pair_horizon_arm'] and record['exact_complete_record_agreement']
        assert record['memory_sample_excludes_hlo_export_and_comparison_owner']
        assert record['graph']['trace_count'] == 1 and record['graph']['no_host_callbacks']
        assert record['record']['program_valid']
        assert len(record['warm_seconds']) == 30 and record['conditioning_calls'] == 3
        assert all(math.isfinite(v) and v > 0 for v in record['warm_seconds'])
        assert record['warm_median_seconds'] == statistics.median(record['warm_seconds'])
        assert bool(record['graph']['process_buffer_shape_nodes']) == (record['arm'] == 'buffered')
        rows.append((number, run, record))
    assert len(rows) == 20
    assert [[r['pair'], r['shape']['T'], r['arm']] for _, _, r in rows] == order['order']
    source = rows[0][1]['source_sha256']
    environment = rows[0][1]['environment']
    for _, run, record in rows:
        assert run['source_sha256'] == source
        assert run['environment'] == environment
        assert record['threads'] == {'intra': '2', 'inter': '1'}
        assert record['affinity'] == rows[0][2]['affinity']
    for name, expected in source.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    comparisons = []
    for horizon in (32, 128):
        pairs = []
        for pair in range(5):
            arms = {r['arm']: (number, r) for number, _, r in rows
                    if r['shape']['T'] == horizon and r['pair'] == pair}
            assert set(arms) == {'buffered', 'streaming'}
            before_run, before = arms['buffered']
            after_run, after = arms['streaming']
            for key in ('input_sha256', 'controls', 'seeds', 'shape', 'shared_record', 'streaming_rng_diagnostics', 'buffered_source_sha256'):
                assert before[key] == after[key], key
            ratio = after['warm_median_seconds'] / before['warm_median_seconds']
            pairs.append({'pair': pair, 'buffered_run': before_run, 'streaming_run': after_run,
                'buffered_ms': 1000 * before['warm_median_seconds'],
                'streaming_ms': 1000 * after['warm_median_seconds'], 'ratio': ratio,
                'buffered_cold_seconds': before['cold_seconds'], 'streaming_cold_seconds': after['cold_seconds'],
                'rss_after_warm_difference_bytes': after['after_warm']['rss_bytes'] - before['after_warm']['rss_bytes'],
                'peak_rss_difference_bytes': after['after_warm']['process_high_water_bytes'] - before['after_warm']['process_high_water_bytes']})
        logs = [math.log(row['ratio']) for row in pairs]
        mean = statistics.mean(logs)
        half = 2.776445105 * statistics.stdev(logs) / math.sqrt(5)
        interval = [math.exp(mean-half), math.exp(mean+half)]
        observed = abs(mean)
        pvalue = sum(abs(statistics.mean(sign * value for sign, value in zip(signs, logs, strict=True)))
            >= observed - 1e-15 for signs in itertools.product((-1, 1), repeat=5)) / 32
        median_ratio = statistics.median(row['ratio'] for row in pairs)
        comparisons.append({'horizon': horizon, 'pairs': pairs,
            'median_ratio': median_ratio, 'geometric_mean_ratio': math.exp(mean),
            'conditional_log_t_95_interval': interval, 'exact_two_sided_sign_flip_p': pvalue,
            'requires_runtime_profiling': median_ratio > 1.10 or interval[1] > 1.10,
            'statistical_caveat': 'Five independent pairs; t interval assumes approximately normal log effects; sign-flip minimum p=.0625.'})
    result = {'schema': 'filter_repair_streaming_paired_analysis.v1', 'seed': order['seed'],
        'order': order['order'], 'run_numbers': [number for number, _, _ in rows], 'excluded_runs': excluded,
        'all_complete_records_exact': True, 'all_sources_environment_matched': True,
        'comparisons': comparisons, 'source_sha256': source,
        'nonclaims': ['CPU reference, not GPU-default evidence.',
            'No general superiority, canonical LEDH, HMC, leak-freedom or master completion.']}
    directory = Path(request.config.getoption('xmlpath')).parent
    (directory / 'streaming-paired-analysis.json').write_text(json.dumps(result, indent=2) + '\n')
