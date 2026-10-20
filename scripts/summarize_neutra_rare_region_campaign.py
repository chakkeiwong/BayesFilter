#!/usr/bin/env python3
"""Read-only post-run diagnostic audit; never changes campaign decisions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(root, shared):
    state, cfg = read(root/'state.json'), read(root/'config.json')
    if state['status'] != 'completed' or state['active']:
        raise ValueError('terminal review requires a settled completed campaign')
    summary = read(root/'summary.json')
    jobs = [row for attempts in state['jobs'].values() for row in attempts]
    sources = sorted({row['source'] for row in jobs})
    failures, source_counts, inputs = [], {}, {}
    for source in sources:
        directory = Path(source)
        manifest = read(directory/'source.json')
        source_counts[source] = len(manifest['sha256'])
        for name, expected in manifest['sha256'].items():
            path = directory/name
            if not path.is_file() or digest(path) != expected:
                failures.append({'check': 'source_sha256', 'path': str(path)})
    gpu_count, cpu_count, peak = 0, 0, 0
    for row in jobs:
        directory = Path(row['output'])
        manifest = read(directory/'manifest.json')
        for path, expected in manifest.get('input_sha256', {}).items():
            if path in inputs and inputs[path] != expected:
                failures.append({'check': 'inconsistent_input_hash', 'path': path})
            inputs[path] = expected
        if row['status'] != 'complete':
            continue
        if manifest.get('status') != 'complete' or not (directory/'result.json').is_file():
            failures.append({'check': 'completed_output', 'path': str(directory)})
        if row['device'] == 'gpu':
            gpu_count += 1
            policy = manifest.get('memory_policy', {})
            if not (policy.get('all_physical_devices_memory_growth') is True
                    and policy.get('configured_before_logical_device_initialization') is True
                    and manifest.get('jit_compile') is True):
                failures.append({'check': 'gpu_policy', 'path': str(directory)})
            peak = max(peak, manifest.get('allocator', {}).get('peak', 0))
        else:
            cpu_count += 1
            if manifest.get('gpu_devices_intentionally_hidden') is not True:
                failures.append({'check': 'cpu_policy', 'path': str(directory)})
    for path, expected in inputs.items():
        if not Path(path).is_file() or digest(path) != expected:
            failures.append({'check': 'input_sha256', 'path': path})

    keys = ('gpu_process_seconds', 'cpu_core_seconds')
    consumed = {k: sum(r.get(k, 0.) for r in jobs) for k in keys}
    shared_cfg, shared_state = read(shared/'config.json'), read(shared/'state.json')
    for row in jobs:
        identifier = 'rare-region-20261002-'+row['name']+'-r'+str(row['attempt'])
        charges = [r for r in shared_state['attempts'] if r['job'] == identifier]
        if len(charges) != 1 or any(abs(charges[0][k]-row[k]) > 1e-8 for k in keys):
            failures.append({'check': 'shared_charge', 'job': identifier})
    for k in keys:
        if abs(cfg[k]-consumed[k]-state['remaining'][k]) > 1e-8 or consumed[k] > cfg[k]:
            failures.append({'check': 'budget', 'resource': k})

    methods = {}
    for name, row in summary['methods'].items():
        path = Path(row['output']); result = read(path/'result.json')
        if len(result['diagnostics']) != len(cfg['replication_seeds']):
            failures.append({'check': 'replication_count', 'method': name})
        methods[name] = {'output': str(path), 'screen': result['screen'],
            'teacher_replication': result['training_bank_replication'],
            'teacher_estimate': result['diagnostics'][0]['estimate'],
            'teacher_event_rows': result['diagnostics'][0]['event_rows'],
            'repair': result['repair']}
    training = []
    for name in summary['training']:
        path = Path(name); result = read(path/'result.json'); checkpoints = []
        for row in result['checkpoints']:
            frozen = read(row['frozen']); probe = read(row['probe'])
            directed = read(path/(row['name']+'-directed.json'))
            if (probe.get('rows') != 1000 or probe.get('valid_rows') != 1000
                    or probe.get('complete') is not True
                    or probe.get('transport_hash') != frozen['transport_hash']):
                failures.append({'check': 'probe_identity', 'path': row['probe']})
            checkpoints.append({'stage': row['name'], 'finite': row['finite'],
                'score_residual_norm': probe['score_residual_norm'],
                'log_ratio_mean': probe['r_log_target_over_gaussian_up_to_constant']['mean'],
                'directed_maximum_norm': directed['maximum_norm'],
                'roundtrip_error': directed['roundtrip_error']})
        training.append({'output': name, 'selected': result['selected'],
            'forward_clip_fraction': result['forward_clip_fraction'],
            'batch_native': result['batch_native'], 'jit_compile': result['jit_compile'],
            'checkpoints': checkpoints})
    expected_fits = len(cfg['targets'])*len(cfg['methods'])*len(cfg['training_seeds'])
    if len(training) != expected_fits:
        failures.append({'check': 'training_denominator', 'expected': expected_fits})
    qualifications = []
    for name in summary['qualifications']:
        result = read(Path(name)/'result.json')
        members = []
        for path in sorted(Path(name).glob('checkpoint-*/member-*/posterior.json')):
            posterior = read(path)
            stage = 'retained' if posterior['retained_results_per_chain'] else 'warmup'
            checks = posterior[stage+'_checks']
            last = checks[-1] if checks else {}
            if stage == 'retained':
                assessment = last.get('full_convergence') or {}
            else:
                assessment = (last.get('modern_rhat') or {}).get('assessment') or {}
            members.append({'path': str(path), 'passed': posterior['passed'],
                'warmup_results_per_chain': posterior['warmup_results_per_chain'],
                'retained_results_per_chain': posterior['retained_results_per_chain'],
                'warmup_cap_hit': posterior['warmup_cap_hit'],
                'retained_cap_hit': posterior['retained_cap_hit'],
                'hard_vetoes': posterior['hard_vetoes'],
                'last_assessed_stage': stage,
                'last_failed_checks': assessment.get('failed_checks', [])})
        qualifications.append({'output': name, **result, 'posterior_members': members,
            'hard_veto_counts': dict(Counter(x for m in members for x in m['hard_vetoes'])),
            'last_failed_check_counts': dict(Counter(x for m in members for x in m['last_failed_checks']))})
    controls = {}
    for target in cfg['targets']:
        rows = state['jobs']['prepare-'+target]
        result = read(Path(rows[-1]['output'])/'preparation.json')
        controls[target] = {key: result[key] for key in ('iid_baseline', 'laplace_baseline')}
        controls[target]['analytic_cdf_oracle'] = {
            'exact': result['iid_baseline']['exact'],
            'role': 'constructed independent erfc calculation; benchmark-specific oracle'}
    return {'schema': 'neutra.rare_region.terminal_diagnostic.v1',
        'integrity_passed': not failures, 'findings': failures,
        'source_files_checked': source_counts, 'unique_inputs_checked': len(inputs),
        'completed_gpu_jobs': gpu_count, 'completed_cpu_jobs': cpu_count,
        'maximum_recorded_tf_allocator_peak_bytes': peak,
        'attempts': len(jobs), 'failed_attempts': [r['output'] for r in jobs if r['status'] != 'complete'],
        'consumed': consumed, 'remaining': state['remaining'],
        'shared_remaining': {k: shared_cfg[k]-sum(r.get(k, 0.) for r in shared_state['attempts']) for k in keys},
        'methods': methods, 'training': training, 'qualifications': qualifications,
        'unmet_prerequisites': summary['omitted'], 'controls': controls,
        'heuristic_dominance': {
            'situations': ['broad valley', 'narrow valley', 'mode occupancy'],
            'adversaries': ['analytic CDF oracle', 'exact iid posterior', 'Laplace defensive IS'],
            'verdict': 'dominance_not_established_no_default_promotion',
            'reason': 'oracle is benchmark-specific; finite stochastic screens do not establish ranking'},
        'method_ranking_established': False, 'default_readiness': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root, args.shared)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('integrity_passed', 'attempts', 'consumed', 'remaining')}))
    if report['findings']:
        print(json.dumps(report['findings']))
    return 0 if report['integrity_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
