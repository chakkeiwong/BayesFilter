#!/usr/bin/env python3
"""Diagnostic reporting only; reads preserved capacity results without TensorFlow."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def summarize(root):
    manifest = read(root / 'manifest.json')
    rows = []
    invariant_reference = None
    invariant_mismatches = []
    common_source = None
    source_mismatches = []
    for arm in manifest.get('results', []):
        result = read(root / arm['arm'] / 'result.json')
        measurements = result.get('measurements', [])
        full = [m for m in measurements if m['full_horizon_valid']]
        warmed = [m for m in full if m['kind'] == 'warmed_execution']
        errors = result.get('exception', '')
        if 'input_hashes' in result:
            invariant = {k: result[k] for k in ('particles', 'horizon', 'controls',
                'beta', 'data_sha256', 'fixed_design_seed', 'parameter_bank_q',
                'dtype', 'precision_policy', 'tf32', 'jit_compile')}
            invariant['fixed_input_hashes'] = result['input_hashes'][1:]
            if invariant_reference is None:
                invariant_reference = invariant
            elif invariant != invariant_reference:
                invariant_mismatches.append(arm['arm'])
        sources = result.get('source_sha256', {})
        if sources:
            if common_source is None:
                common_source = sources
            else:
                source_mismatches.extend([dict(arm=arm['arm'], source=k)
                    for k in common_source.keys() & sources.keys() if common_source[k] != sources[k]])
        request = re.search(r'trying to allocate (\d+) bytes', errors)
        if full:
            fit = 'full_horizon_executed'
        elif 'HOST_RSS_GUARD' in str(arm.get('stop_reason')):
            fit = 'host_memory_guard_before_completion'
        elif 'SYSTEM_AVAILABLE_MEMORY_GUARD' in str(arm.get('stop_reason')):
            fit = 'system_memory_guard_before_completion'
        elif 'ResourceExhausted' in result.get('exception_type', '') or 'out of memory' in errors.lower():
            fit = 'allocation_failed'
        elif measurements:
            fit = 'executed_but_numerically_ineligible'
        elif 'TIME_CAP' in str(arm.get('stop_reason')):
            fit = 'unknown_time_limit'
        else:
            fit = 'unknown_execution_failure'
        allocator_peaks = [m['allocator']['peak'] for m in measurements
                           if 'peak' in m.get('allocator', {})]
        median = statistics.median(m['seconds'] for m in warmed) if warmed else None
        row = dict(
            arm=arm['arm'], device=arm['device'], batch_size=arm['batch_size'],
            capacity_status=fit, stop_reason=arm.get('stop_reason'),
            exit_code=arm['exit_code'], last_phase=arm.get('last_phase'),
            wall_seconds=arm['wall_seconds'],
            host_peak_rss_bytes=max(arm['peak_rss_bytes_sampled'],
                                    arm.get('peak_hwm_bytes_sampled', 0)),
            minimum_available_host_bytes=arm['minimum_available_host_bytes'],
            allocator_peak_bytes=max(allocator_peaks) if allocator_peaks else None,
            first_call_seconds=measurements[0]['seconds'] if measurements else None,
            full_horizon_calls=len(full), completed_calls=len(measurements),
            warmed_full_horizon_calls=len(warmed), median_warmed_seconds=median,
            warmed_range_seconds=([min(m['seconds'] for m in warmed),
                                   max(m['seconds'] for m in warmed)] if warmed else None),
            full_rows_per_second=arm['batch_size']/median if median else None,
            graph_nodes=result.get('graph', {}).get('nodes_including_functions'),
            trace_seconds=result.get('trace_seconds'),
            exception_type=result.get('exception_type'),
            failed_allocation_request_bytes=int(request.group(1)) if request else None,
            exception=errors or None,
            per_call_validity=[dict(valid=m['valid'], steps_completed=m['steps_completed'],
                                    failure_fields=m['failure_fields']) for m in measurements],
            evidence_paths=dict(result=arm['result'], log=arm['log']))
        if full:
            row['first_valid_values'] = full[0]['values']
            row['first_valid_scores'] = full[0]['scores']
            row['within_process_repeat_max_value_delta'] = max(
                abs(x-y) for m in full for x,y in zip(m['values'], full[0]['values']))
            row['within_process_repeat_max_score_delta'] = max(
                abs(x-y) for m in full for s,t in zip(m['scores'], full[0]['scores']) for x,y in zip(s,t))
        rows.append(row)
    reference = next((r for r in rows if r['arm'] == 'cpu-b4' and 'first_valid_values' in r), None)
    if reference:
        for row in rows:
            if 'first_valid_values' not in row:
                continue
            row['parity_vs_cpu_b4'] = dict(
                evidence_role='descriptive_finite_program_comparison_not_oracle_accuracy',
                maximum_absolute_value_difference=max(abs(v-reference['first_valid_values'][i % 4])
                    for i, v in enumerate(row['first_valid_values'])),
                maximum_absolute_score_difference=max(abs(v-reference['first_valid_scores'][i % 4][j])
                    for i, scores in enumerate(row['first_valid_scores']) for j, v in enumerate(scores)),
                maximum_relative_score_norm_difference=max(
                    math.sqrt(sum((x-y)**2 for x,y in zip(scores, reference['first_valid_scores'][i % 4]))) /
                    math.sqrt(sum(x*x for x in reference['first_valid_scores'][i % 4]))
                    for i, scores in enumerate(row['first_valid_scores'])))
    return dict(schema='bayesfilter.sir_fp32_capacity_summary.v1',
                source_manifest=str(root / 'manifest.json'),
                source_finished_utc=manifest.get('finished_utc'),
                measured_scope='SIR N4032 T50 d18 p3 uniform FP32 TF32 off XLA',
                timing_inference='descriptive_only_shared_hardware',
                fixed_workload_invariant_mismatches=invariant_mismatches,
                common_source_mismatches=source_mismatches,
                gpu_allocator_limit_mib=manifest.get('gpu_allocator_limit_mib'),
                wall_seconds=manifest.get('wall_seconds'), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root.resolve())
    if summary['source_finished_utc'] is None:
        parser.error('benchmark manifest is not finished; leave live evidence untouched')
    with args.output.open('x') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    for row in summary['rows']:
        print(json.dumps({k: row[k] for k in ('arm', 'capacity_status', 'host_peak_rss_bytes',
            'allocator_peak_bytes', 'first_call_seconds', 'median_warmed_seconds',
            'warmed_full_horizon_calls')}))


if __name__ == '__main__':
    main()
