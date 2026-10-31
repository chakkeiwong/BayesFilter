"""Bounded diagnostic renewal of missing core-filter resource comparisons."""

import argparse
import json
import math
import statistics
from pathlib import Path

CASES = {
    'contract_e': 'on', 'tt': 'eager', 'tt_adapted': 'eager',
    'tt_gaussian': 'eager', 'tt_adjoint': 'eager', 'tt_actual': 'on',
    'tt_scalar': 'eager', 'apf': 'on', 'dns': 'on', 'retained_moments': 'on',
}
GROUP = 'resource_acceptance_core_measure_gpu'


def validate(run, result, runner, baseline):
    from compare_filter_repair_campaign import current_provenance
    from filter_repair_cost_provenance import validate_cost_device

    if run['state'] != 'passed' or result['status'] != 'passed':
        raise ValueError(f"Failed core cost worker: {run['result']}")
    current_provenance(run, result, run['key'][2],
                       runner.measurement_harness(run['key'][3]), baseline)
    validate_cost_device(run, result['device_provenance'], result['device_observation'])
    observed = run['process_exit_observation']
    if (observed['errors'] or observed['proc_entry_present']
            or any(p['pid'] == observed['pid'] for p in observed['gpu_processes'])):
        raise ValueError('Missing core worker exit containment')
    if len(result['warm']) != 20:
        raise ValueError('Incomplete warm timing samples')
    if run['key'][2] == 'after' and result['jit'] == 'on':
        reuse = result['reuse']
        if not (reuse['calls'] == 128 and reuse['exact_replay']
                and reuse['late_64_call_rss_growth_bytes'] <= 16*1024**2):
            raise ValueError('Missing bounded current-XLA reuse')
        if (reuse['samples']['64']['gpu']['current'] != reuse['samples']['128']['gpu']['current']
                or reuse['samples']['128']['gpu']['peak'] > 2*1024**3):
            raise ValueError('Unbounded core allocator reuse')


def allocation(runner):
    data = json.loads((runner.ROOT/'docs/plans/filter_gradient_repair_ledger_20260917.json').read_text())
    return data['current_checkpoint']['core_resource_allocation']


def allocation_rows(runner):
    first = allocation(runner)['start_after_run']
    return [row for row in runner.records() if row['key'][1].startswith('resource_acceptance_core_')
            and int(Path(row['result']).parent.name[4:]) > first]


def cost_rows(runner):
    return [row for row in allocation_rows(runner) if row['key'][1] == GROUP]


def run_core_resources(args, runner):
    """Freeze sources, reserve bounded workers and stop at the first bad block."""
    from compare_filter_repair_campaign import compare_pair

    frozen = runner.source_hashes()
    runner.ensure_baseline()
    baseline = json.loads((runner.BASELINE_ROOT/'source-manifest.json').read_text())['files']
    runner.prepare_gpu(args, 'measurement_gpu_index')
    limits = allocation(runner)
    for case, before_mode in CASES.items():
        arms = (('before', before_mode), ('after', 'off'), ('after', 'on'))
        for pair in range(3):
            values = {}
            for arm, mode in arms[pair:]+arms[:pair]:
                runner.check_matrix_state(frozen)
                charged = allocation_rows(runner)
                prior = [row for row in charged if row['key'][1] == GROUP]
                matches = [row for row in prior if row['state'] == 'passed'
                    and row['source_sha256'] == frozen and runner.same_gpu(row, args)
                    and row['key'][2:] == [arm, case, mode, 1, pair, 'GPU']]
                if matches:
                    row = matches[-1]
                else:
                    spent = sum(row.get('elapsed_seconds', row['timeout_seconds']) for row in prior)
                    if len(charged) >= limits['max_workers']-1 or spent+300 > limits['max_seconds']['GPU']:
                        raise RuntimeError('Core resource allocation exhausted before launch')
                    job = argparse.Namespace(**vars(args))
                    job.action, job.group, job.arm = 'measure', GROUP, arm
                    job.fixture, job.jit, job.size, job.repeat = case, mode, 1, pair
                    job.device, job.gpu_preflight = 'GPU', None
                    runner.prepare_gpu(job, 'measurement_gpu_index')
                    code = runner.run_job(job)
                    row = runner.latest_record()
                    if code:
                        raise RuntimeError(f"Core resource worker failed: {row['log']}")
                value = json.loads(Path(row['result']).read_text())
                validate(row, value, runner, baseline)
                values[(arm, mode)] = value
            original, graph, xla = [values[arm] for arm in arms]
            errors = [compare_pair(original, graph), compare_pair(original, xla), compare_pair(graph, xla)]
            print(json.dumps({'core_resources': case, 'pair': pair,
                              'comparison': 'passed', 'maximum_absolute_errors': errors}), flush=True)
    job = argparse.Namespace(**vars(args))
    charged = allocation_rows(runner)
    cpu = sum(row.get('elapsed_seconds', row['timeout_seconds']) for row in charged if row['device'] == 'CPU')
    if len(charged) >= limits['max_workers'] or cpu+300 > limits['max_seconds']['CPU']:
        raise RuntimeError('Core terminal allocation exhausted before launch')
    job.action, job.group, job.arm, job.device = 'test', 'resource_acceptance_core_terminal_cpu', 'after', 'CPU'
    job.gpu_preflight, job.test_timeout_seconds = None, 300
    return runner.run_job(job)


def summarize(rows):
    """Report process-paired ratios; inner warm calls are not replicates."""
    indexed = {}
    for run in rows:
        if run['state'] == 'passed':
            indexed[tuple(run['key'][2:7])] = (run, json.loads(Path(run['result']).read_text()))
    expected = {(arm, case, mode, 1, pair) for case, old in CASES.items()
                for arm, mode in (('before', old), ('after', 'off'), ('after', 'on')) for pair in range(3)}
    if set(indexed) != expected:
        raise ValueError(f'Incomplete core cost cohort: missing {sorted(expected-set(indexed))}')
    comparisons = []
    for case, old in CASES.items():
        for first in (('before', old), ('after', 'off')):
            pairs = []
            for pair in range(3):
                row_a, a = indexed[(first[0], case, first[1], 1, pair)]
                row_b, b = indexed[('after', case, 'on', 1, pair)]
                def metrics(value):
                    return {
                        'warm_seconds': statistics.median(r['synchronized_seconds'] for r in value['warm']),
                        'total_cold_seconds': value['preparation_seconds']+value['trace_seconds']+value['cold']['synchronized_seconds'],
                        'host_peak_bytes': max(value['stages']['cold_outputs_live']['VmHWM'],
                                               *(r['VmHWM'] for r in value['warm'])),
                        'device_peak_bytes': max(value['stages']['cold_outputs_live']['gpu']['peak'],
                                                 *(r['gpu']['peak'] for r in value['warm'])),
                        'warm_rss_bytes': value['warm'][-1]['VmRSS'],
                    }
                before, after = metrics(a), metrics(b)
                pairs.append({'pair': pair, 'before_run': str(Path(row_a['result']).parent),
                    'after_run': str(Path(row_b['result']).parent), 'before': before, 'after': after,
                    'warm_ratio': after['warm_seconds']/before['warm_seconds'],
                    'extra_rss_bytes': after['warm_rss_bytes']-before['warm_rss_bytes']})
            logs = [math.log(p['warm_ratio']) for p in pairs]
            center, half = statistics.mean(logs), 4.30265272975*statistics.stdev(logs)/math.sqrt(3)
            comparisons.append({'fixture': case, 'before_arm': list(first), 'pairs': pairs,
                'geometric_mean_warm_ratio': math.exp(center),
                'paired_log_t_95_interval': [math.exp(center-half), math.exp(center+half)]})
    return comparisons
