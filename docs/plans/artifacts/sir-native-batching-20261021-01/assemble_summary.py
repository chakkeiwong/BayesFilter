"""Post-run diagnostic reporting only; no runtime or admission decisions."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'sir-fp32-batch-capacity-20261020-02'
REPO = ROOT.parents[3]
STARTED = time.monotonic()


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


def adapted_reference_comparison():
    """Describe saved estimates; do not infer equivalence from a noisy reference."""
    for key in ('data_sha256', 'input_hashes', 'parameter_rows', 'controls',
                'beta', 'target_id', 'fixed_design_seed', 'particles', 'horizon',
                'state_dimension', 'observation_dimension', 'dtype', 'tf32'):
        if old32[key] != new32[key]:
            raise ValueError(f'Old/new comparison mismatch: {key}')
    for run in (old32, new32):
        if not run['measurements'][-1]['full_horizon_valid']:
            raise ValueError('Comparison requires valid full-horizon evaluations')
    data_path = REPO / Path(new32['data_path']).relative_to(REPO)
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != new32['data_sha256']:
        raise ValueError('Dataset hash differs from the executed benchmark')
    dataset = read(data_path)
    reference_path = (ROOT.parent / 'ledh-independent-validation-20261008-01' /
                      'report-20261009-011148/comparison.json')
    rows = [row for row in read(reference_path)['datasets']
            if row['model'] == 'sir_d18' and row['data_seed'] == dataset['data_seed']]
    if len(rows) != 1 or dataset['target_id'] != new32['target_id']:
        raise ValueError('Reference model/data identity is not uniquely matched')
    row = rows[0]
    adapted = row['reference']['zhao_rank20']
    if not adapted['complete'] or adapted['replications'] != len(adapted['fits']):
        raise ValueError('Incomplete adapted reference')
    for fit in adapted['fits']:
        provenance = fit['provenance']
        if (provenance['source_observation_sha256'] != dataset['observation_sha256']
                or provenance['horizon'] != new32['horizon']):
            raise ValueError('Reference observations/horizon do not match')
    matched = [i for i, theta in enumerate(new32['parameter_rows']) if theta == row['theta']]
    if not matched:
        raise ValueError('No saved adapted reference at the benchmark parameter points')
    scores = adapted['score']
    score_se = adapted['score_se']
    matches = []
    for i in matched:
        old = old32['measurements'][-1]
        new = native4
        if not (len(old['scores'][i]) == len(new['scores'][i]) == len(scores)
                == len(score_se) == len(row['parameter_names'])):
            raise ValueError('Score coordinate count mismatch')
        change = [b-a for a, b in zip(old['scores'][i], new['scores'][i])]
        ratios = [abs(x)/se if se > 0 else None for x, se in zip(change, score_se)]
        value_change = new['values'][i] - old['values'][i]
        matches.append({
            'row': i, 'theta': row['theta'], 'parameter_names': row['parameter_names'],
            'old_score': old['scores'][i], 'new_score': new['scores'][i],
            'reference_score': scores, 'reference_between_fit_score_se': score_se,
            'new_minus_old_score': change,
            'absolute_change_over_reference_se_descriptive_only': ratios,
            'old_score_minus_reference': [a-b for a, b in zip(old['scores'][i], scores)],
            'new_score_minus_reference': [a-b for a, b in zip(new['scores'][i], scores)],
            'old_relative_l2_to_reference': scaled_score(old['scores'][i], scores),
            'new_relative_l2_to_reference': scaled_score(new['scores'][i], scores),
            'new_relative_l2_to_old': scaled_score(new['scores'][i], old['scores'][i]),
            'old_value': old['values'][i], 'new_value': new['values'][i],
            'reference_value': adapted['value'],
            'reference_between_fit_value_se': adapted['value_se'],
            'new_minus_old_value': value_change,
            'absolute_value_change_over_reference_se_descriptive_only':
                abs(value_change)/adapted['value_se'] if adapted['value_se'] > 0 else None,
            'paired_design_count': 1, 'paired_change_mcse': None,
            'statistical_equivalence': 'not_established',
        })
    rounding = max(abs(value-struct.unpack('f', struct.pack('f', value))[0])
                   for observation in dataset['observations'] for value in observation)
    return {
        'reference_path': str(reference_path.relative_to(REPO)),
        'reference_sha256': hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        'data_sha256': new32['data_sha256'], 'kind': adapted['kind'],
        'replications': adapted['replications'],
        'fits': [{k: f['provenance'][k] for k in ('fit_seed', 'rank', 'samples')}
                 for f in adapted['fits']],
        'uncertainty': adapted['uncertainty'],
        'reference_input_precision': 'Original FP64 observations; kernels use FP32-rounded observations',
        'maximum_absolute_observation_rounding': rounding,
        'reference_score_effect_of_rounding': 'not_measured',
        'matched_rows': matches,
        'unmatched_rows': [{'row': i, 'theta': theta, 'status': 'no_matched_adapted_reference'}
                           for i, theta in enumerate(new32['parameter_rows']) if i not in matched],
        'uncertainty_limits': [
            'Reference SE is not the MCSE of the paired old/new change',
            'Common reference cancels in the paired score difference',
            'Repeated fixed-design evaluations are not independent Monte Carlo replications',
            'Reference rank/support/local-fit bias is not bounded by between-fit SE',
            'Saved bootstrap reference disagrees; no exact SIR score oracle is established',
        ],
    }

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
    'schema': 'sir_native_batching_diagnostic_summary_v2',
    'purpose': 'Engineering diagnostics and reference-based materiality reclassification; no posterior or training admission',
    'reference_scope': 'Adapted Zhao-Cui reference for model-score comparison; same finite-filter FP64 for arithmetic diagnostics',
    'arithmetic_reference_scope': 'Same finite filter and FP32-prepared data/design, cast to FP64; not an exact model-likelihood oracle',
    'score_relative_scale': 'L2(candidate-reference)/max(1,L2(reference)) per row',
    'value_relative_scale': 'abs(candidate-reference)/max(1,abs(reference)) per row',
    'materiality_policy': {
        'plan': 'docs/plans/sir-batching-materiality-policy-2026-10-21.md',
        'comparator': 'Adapted Zhao-Cui numerical reference with estimation uncertainty',
        'historical_percentage_threshold': .01,
        'percentage_threshold_role': 'retired_as_scientific_veto',
        'fp64_role': 'arithmetic_diagnostic_only',
        'fixed_percentage_blocker': False,
        'nominal_continuation_decision': 'owner_accepts_batching_change_as_immaterial',
        'all_row_statistical_equivalence': 'not_established',
    },
    'adapted_reference_comparison': adapted_reference_comparison(),
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
        'numerical_promotion': 'Percentage-only blocker retired; nominal change accepted for continuation, broader accuracy unestablished',
        'training_ready': False,
        'timing_ranking': 'Descriptive only: shared hardware, small repeat count, no uncertainty-based ranking',
        'remaining': ['NeuTra target adapter and complete optimizer-step parity/memory',
                      'Paired independent-design and matched-reference uncertainty for broader accuracy claims'],
    },
}
summary['analysis_manifest'] = {
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'command': [sys.executable, *sys.argv], 'python': sys.version,
    'execution': 'CPU standard-library post-run analysis; no framework or GPU initialization',
    'new_stochastic_evaluations': 0,
    'fixed_design_seed': new32['fixed_design_seed'],
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'input_sha256': {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [BASE/'gpu-b4/result.json', ROOT/'gpu-density-order/gpu-b4/result.json',
                               ROOT/'summary.json']},
    'historical_summary': 'summary.json', 'output': 'summary-materiality.json',
    'plan': 'docs/plans/sir-batching-materiality-policy-2026-10-21.md',
    'result_note': 'docs/plans/sir-native-batching-results-2026-10-21.md',
    'elapsed_seconds': time.monotonic()-STARTED,
}
(ROOT / 'summary-materiality.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print(json.dumps({'arms': len(arms), 'finished_groups': sum(v['finished'] for v in budget.values()),
                  'groups': len(budget), 'output': str(ROOT/'summary-materiality.json')}))
