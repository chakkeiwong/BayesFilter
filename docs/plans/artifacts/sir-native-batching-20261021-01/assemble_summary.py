"""Post-run diagnostic reporting only; no runtime or admission decisions."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'sir-fp32-batch-capacity-20261020-02'


def read(path):
    return json.loads(path.read_text())


def scaled_score(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b))) / max(
        1.0, math.sqrt(sum(x*x for x in b)))


def comparison(a, b):
    return {
        'value_relative_by_row': [abs(x-b['values'][i % 4]) /
            max(1.0, abs(b['values'][i % 4])) for i, x in enumerate(a['values'])],
        'score_relative_l2_by_row': [scaled_score(x, b['scores'][i % 4])
            for i, x in enumerate(a['scores'])],
    }


old32 = read(BASE / 'gpu-b4/result.json')
new32 = read(ROOT / 'gpu-density-order/gpu-b4/result.json')
old64 = read(ROOT / 'gpu-baseline-fp64/gpu-b4/result.json')
new64 = read(ROOT / 'gpu-native-fp64/gpu-b4/result.json')
reference = new64['measurements'][-1]
native4 = new32['measurements'][-1]

arms = []
budget = {}
for group in sorted(ROOT.glob('*/manifest.json')):
    manifest = read(group)
    budget[group.parent.name] = {
        'finished': 'finished_utc' in manifest,
        'supervisor_wall_seconds': manifest.get('wall_seconds'),
        'completed_arm_wall_seconds': sum(x['wall_seconds'] for x in manifest['results']),
    }
    for arm in manifest['results']:
        result_path = group.parent / arm['arm'] / 'result.json'
        result = read(result_path)
        measurements = result.get('measurements', [])
        warm = [x['seconds'] for x in measurements if x['kind'] == 'warmed_execution']
        allocator = [x.get('allocator', {}).get('peak', 0) for x in measurements]
        row = {
            'group': group.parent.name, 'arm': arm['arm'],
            'device': arm['device'], 'batch_size': arm['batch_size'],
            'dtype': result.get('dtype'), 'shard_size': result.get('native_shard_size'),
            'exit_code': arm['exit_code'], 'stop_reason': arm.get('stop_reason'),
            'worker_status': result['status'], 'last_phase': result.get('phase'),
            'exception': result.get('exception'), 'arm_wall_seconds': arm['wall_seconds'],
            'first_execution_seconds': measurements[0]['seconds'] if measurements else None,
            'warmed_seconds': warm,
            'median_warmed_seconds': statistics.median(warm) if warm else None,
            'peak_allocator_gib': max(allocator)/2**30 if allocator else None,
            'peak_host_rss_gib': arm.get('peak_rss_bytes_sampled', 0)/2**30,
            'completed_evaluations': len(measurements),
            'all_completed_evaluations_valid': all(x['full_horizon_valid'] for x in measurements) if measurements else None,
            'input_hashes': result.get('input_hashes'),
            'source_result': str(result_path.relative_to(ROOT)),
            'source_result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        }
        if measurements:
            row['against_fp64_reference'] = comparison(measurements[-1], reference)
            row['against_native_b4'] = comparison(measurements[-1], native4)
        arms.append(row)

test_seconds = {}
for log in sorted(ROOT.glob('*tests*.log')):
    matches = re.findall(r' in ([0-9.]+)s', log.read_text(errors='replace'))
    if matches:
        test_seconds[log.name] = float(matches[-1])
gpu_seconds = sum(row['arm_wall_seconds'] for row in arms if row['device'] == 'gpu')
cpu_seconds = sum(row['arm_wall_seconds'] for row in arms if row['device'] == 'cpu')

summary = {
    'schema': 'sir_native_batching_diagnostic_summary_v1',
    'purpose': 'Engineering and numerical diagnostics; no posterior or training admission',
    'reference_scope': 'Same finite filter and FP32-prepared data/design, cast to FP64; not an exact model-likelihood oracle',
    'score_relative_scale': 'L2(candidate-reference)/max(1,L2(reference)) per row',
    'value_relative_scale': 'abs(candidate-reference)/max(1,abs(reference)) per row',
    'tolerance': .01,
    'precision': {
        'old_fp32': old32['measurements'][-1],
        'native_fp32': native4,
        'old_fp64': old64['measurements'][-1],
        'native_fp64': reference,
        'native_vs_old_fp64': comparison(reference, old64['measurements'][-1]),
        'native_vs_old_fp32': comparison(native4, old32['measurements'][-1]),
        'old_fp32_vs_fp64': comparison(old32['measurements'][-1], reference),
        'native_fp32_vs_fp64': comparison(native4, reference),
    },
    'arms': arms, 'budget': budget,
    'budget_totals': {
        'gpu_arm_wall_seconds': gpu_seconds,
        'cpu_arm_wall_seconds': cpu_seconds,
        'test_elapsed_seconds_by_log': test_seconds,
        'cpu_arms_plus_recorded_test_seconds': cpu_seconds + sum(test_seconds.values()),
        'gpu_ceiling_seconds': 7200,
        'cpu_ceiling_seconds': 7200,
        'accounting_scope': 'All supervised arms and logged pytest durations, including failed attempts; excludes editing, reporting and small un-timed setup/import commands',
    },
    'interpretation': {
        'engineering_algebra': 'Supported by focused regressions and full-horizon FP64 parity',
        'numerical_promotion': 'BLOCKED: uniform FP32 score discrepancy exceeds 1%',
        'training_ready': False,
        'timing_ranking': 'Descriptive only: shared hardware, small repeat count, no uncertainty-based ranking',
        'remaining': ['FP32 score stability', 'NeuTra target adapter and complete optimizer-step parity/memory'],
    },
}
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print(json.dumps({'arms': len(arms), 'finished_groups': sum(v['finished'] for v in budget.values()),
                  'groups': len(budget), 'output': str(ROOT/'summary.json')}))
