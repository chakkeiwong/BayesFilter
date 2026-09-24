"""Read-only M20 saved-state geometry diagnosis; never changes tuning records."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.funnel_geometry import model_to_latent, residual_geometry_program
    from bayesfilter.testing.inference_validation.storage import read_json, read_tensor, file_hash, write_json

    records = []
    for path in sorted(args.input.glob('fits-*/**/worker/result.json')):
        fit_result = read_json(path)
        if not fit_result.get('posterior'):
            continue
        root = path.parent / 'fit'
        tuning = read_json(root / 'tuning/candidate_set_result.json')
        candidates = {c['candidate_id']: c for c in tuning['candidates']}
        kind = fit_result['map']
        geometry = residual_geometry_program(kind, jit_compile=False)
        for selected in fit_result['posterior']:
            cid = selected['candidate_id']
            member_root = root / 'members' / cid
            result_path = member_root / 'result.json'
            member = read_json(result_path)
            record = {'input': str(path), 'map': kind, 'seed': fit_result['seed'],
                      'candidate_id': cid, 'epsilon': selected['epsilon'], 'L': selected['L'],
                      'search_completion': tuning['completion_status'],
                      'candidate_count': len(candidates),
                      'verified_siblings': [{k: candidates[c][k] for k in
                          ('candidate_id', 'epsilon', 'leapfrog_steps')}
                          for c in tuning['verified_candidate_ids']],
                      'unassessed_members': fit_result['unassessed_members'],
                      'outcome': selected, 'stages': [],
                      'input_hashes': {str(p): file_hash(p) for p in
                          (path, result_path, root / 'tuning/candidate_set_result.json')},
                      'proposal_failure_location': 'not identifiable from accepted model states; rejected leapfrog paths not archived'}
            for stage, filename, checks in (
                    ('warmup', 'warmup.tensor', member['posterior']['warmup_checks']),
                    ('retained', 'draws.tensor', member['posterior']['retained_checks'])):
                tensor_path = member_root / filename
                values = read_tensor(tensor_path)
                record['input_hashes'][str(tensor_path)] = file_hash(tensor_path)
                if values.shape[0] == 0:
                    continue
                # Only the model-coordinate accepted states are saved here.
                flat = tf.reshape(model_to_latent(values, kind), (-1, 3))
                report = geometry(flat)
                eigenvalues = tf.linalg.eigvalsh(report['hessian'])
                local_frequency = tf.sqrt(tf.maximum(eigenvalues[:, -1], 0.))
                positive_indicator = selected['epsilon'] * local_frequency
                failed_checks = [c for c in checks if not c['health']['health_passed']]
                record['stages'].append({
                    'stage': stage, 'draws_per_chain': int(values.shape[0]),
                    'accepted_states_finite': bool(tf.reduce_all(tf.math.is_finite(values))),
                    'analytic_potential_score_hessian_finite': all(bool(tf.reduce_all(tf.math.is_finite(report[k])))
                        for k in ('potential', 'score', 'hessian')),
                    'model_v_min': float(tf.reduce_min(report['v'])),
                    'model_v_max': float(tf.reduce_max(report['v'])),
                    'max_child_curvature': float(tf.reduce_max(report['child_curvature'])),
                    'max_positive_hessian_eigenvalue': float(tf.reduce_max(eigenvalues[:, -1])),
                    'max_local_harmonic_epsilon_frequency': float(tf.reduce_max(positive_indicator)),
                    'local_harmonic_indicator_above_two_count': int(tf.reduce_sum(tf.cast(positive_indicator > 2., tf.int32))),
                    'indicator_interpretation': 'explanatory local harmonic comparison; not a nonlinear stability veto',
                    'failed_chunks': [{k: c[k] for k in ('chunk_index', 'completed_results_per_chain', 'health', 'seed')}
                                      for c in failed_checks],
                })
            records.append(record)
    if len(records) != 6 or sum(r['map'] == 'exact' for r in records) != 2:
        raise ValueError('expected two exact and four partial saved posterior outcomes')
    summary = {
        'schema': 'bayesfilter.supplied_funnel_geometry_audit.v1',
        'plan_file': 'docs/plans/bayesfilter-hmc-gap-closure-continuation-2026-09-22.md',
        'command': [sys.executable, *sys.argv], 'records': records,
        'derivation': 'U=.5*z0^2+.5*exp(2s-v)*sum(z_child^2)-(2s-v), v=3*z0, k=2',
        'partial_tail_result': 's(v) is bounded, so exp(2s-v) diverges as v tends to negative infinity',
        'exact_tail_result': 's(v)=v/2 gives U=.5*sum(z^2) and identity Hessian',
        'tuning_records_changed': False, 'posterior_promoted': False,
        'ranking_supported': False, 'gpu_intentionally_hidden': True, 'jit_compile': False,
        'elapsed_seconds': time.monotonic() - started,
    }
    write_json(args.output / 'result.json', summary)
    for row in records:
        print(json.dumps({k: row[k] for k in ('map', 'seed', 'epsilon', 'L', 'stages')}), flush=True)


if __name__ == '__main__':
    main()
