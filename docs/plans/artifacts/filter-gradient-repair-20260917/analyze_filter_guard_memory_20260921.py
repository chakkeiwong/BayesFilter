"""Independent standard-library reporting of completed guard-cost diagnostics."""
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def compare(before, after, path='result'):
    if isinstance(before, dict):
        if set(before) != set(after):
            raise ValueError(f'{path}: field mismatch')
        return max((compare(before[k], after[k], f'{path}.{k}') for k in before), default=0.)
    if isinstance(before, list):
        if len(before) != len(after):
            raise ValueError(f'{path}: length mismatch')
        return max((compare(a, b, f'{path}[{i}]') for i, (a, b) in enumerate(zip(before, after))), default=0.)
    if isinstance(before, float):
        if not math.isfinite(before) or not math.isfinite(after) or not math.isclose(before, after, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'{path}: numerical mismatch: {before} / {after}')
        return abs(before-after)
    if type(before) is not type(after) or before != after:
        raise ValueError(f'{path}: discrete mismatch: {before} / {after}')
    return 0.


def summary(record):
    samples = record['samples']
    return {
        'cold_seconds': samples[0]['seconds'],
        'warm_median_seconds': statistics.median(row['seconds'] for row in samples[1:]),
        'host_peak_bytes': max([row['memory']['host']['VmHWM'] for row in samples]
                               + [stage['host']['VmHWM'] for stage in record['stages'].values()]),
        'gpu_peak_bytes': max(row['memory']['gpu']['peak'] for row in samples),
        'warm_gpu_current_bytes': sorted({row['memory']['gpu']['current'] for row in samples[1:]}),
        'warm_host_rss_first_last_bytes': [samples[1]['memory']['host']['VmRSS'], samples[-1]['memory']['host']['VmRSS']],
        'trace_count': record['trace_count'],
        'graph_nodes': record['graph_nodes'],
    }


def main():
    root = Path(sys.argv[1])
    run_numbers = [int(number) for number in sys.argv[2:]]
    reports = {}
    provenance = {}
    sources = set()
    for number in run_numbers:
        directory = root/f'run-{number:05d}'
        run = json.loads((directory/'run.json').read_text())
        if run['state'] != 'passed' or not run['test_evidence']['passed']:
            raise ValueError(f'{number}: required arm did not pass')
        report_path = directory/'factor-guard-memory.json'
        record = json.loads(report_path.read_text())
        key = record['arm'], record['jit_compile'], record['dimension']
        if key in reports:
            raise ValueError(f'{number}: duplicate arm')
        if len(record['samples']) != 21 or record['max_iterations'] != 200 or record['trace_count'] != 1:
            raise ValueError(f'{number}: invalid measurement contract')
        if key[0] == 'candidate' and any(row['invalid_evaluations'] != 0 for row in record['samples']):
            raise ValueError(f'{number}: invalid covariance encountered')
        reports[key] = record
        provenance[str(key)] = {'run': number, 'report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
                                'gpu_index': run['environment']['CUDA_VISIBLE_DEVICES']}
        sources.add(json.dumps(run['source_sha256'], sort_keys=True))
    expected = {(arm,jit,dimension) for arm in ('checkpoint','candidate') for jit in (False,True) for dimension in (3,5)}
    if set(reports) != expected or len(sources) != 1:
        raise ValueError('incomplete matrix or source changed')
    if len({row['gpu_index'] for row in provenance.values()}) != 1:
        raise ValueError('mixed GPU devices')
    comparisons = []
    for dimension in (3,5):
        for jit in (False,True):
            before, after = (reports[(arm,jit,dimension)] for arm in ('checkpoint','candidate'))
            if before['input_sha256'] != after['input_sha256']:
                raise ValueError('unmatched inputs')
            error = compare(before['result'], after['result'])
            old, new = summary(before), summary(after)
            comparisons.append({'dimension':dimension,'jit_compile':jit,'max_absolute_error':error,
                'checkpoint':old,'candidate':new,
                'host_peak_delta_bytes':new['host_peak_bytes']-old['host_peak_bytes'],
                'cold_ratio':new['cold_seconds']/old['cold_seconds'],
                'warm_ratio':new['warm_median_seconds']/old['warm_median_seconds'],
                'gpu_peak_ratio':new['gpu_peak_bytes']/old['gpu_peak_bytes']})
    cross_mode=[]
    for arm in ('checkpoint','candidate'):
        for dimension in (3,5):
            graph,xla=(reports[(arm,jit,dimension)] for jit in (False,True))
            if graph['input_sha256'] != xla['input_sha256']:
                raise ValueError('unmatched graph/XLA inputs')
            # Record disagreements rather than presenting different trajectories
            # as a same-program performance ratio.
            try:
                error=compare(graph['result'],xla['result'])
                agreement={'passed':True,'max_absolute_error':error}
            except ValueError as exc:
                agreement={'passed':False,'error':str(exc)}
            cross_mode.append({'arm':arm,'dimension':dimension,'complete_record_parity':agreement,
                               'graph':summary(graph),'xla':summary(xla)})
    result={'schema':'filter_repair_factor_guard_cost_comparison.v1','provenance':provenance,
            'same_mode_comparisons':comparisons,'graph_vs_xla':cross_mode,
            'analysis_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'nonclaims':['One fresh process per arm; descriptive cost only, not a statistical performance ranking.',
                         'Baseline 085baaaa isolates guard cost; original whole-campaign baseline and terminal repeats remain separate.',
                         'CPU executable-cache mapping limit is not changed or resolved by these GPU memory measurements.']}
    destination=root/f'factor-guard-cost-comparison-{max(run_numbers):05d}.json'
    with destination.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False)
        output.write('\n')
    print(json.dumps(result,indent=2))
    print(destination)


if __name__ == '__main__':
    main()
