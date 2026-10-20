"""CPU-only post-run comparison of saved RBF solves at matching tolerances."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from diagnose_rbf_coefficient_stability import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'fixture', 'diagnosis', 'candidate', 'stability'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    fixture = json.loads(args.fixture.read_text())
    stability = json.loads((args.stability/'summary.json').read_text())
    calibration = json.loads((args.candidate/'summary.json').read_text())
    if len(stability['reference_replay_checks']) != 2 or not all(row['passed'] for row in stability['reference_replay_checks']):
        raise ValueError('Independent reference replay must pass')
    candidates = tf.io.parse_tensor(tf.io.read_file(str(args.candidate/'frozen_coefficients.tensor')), tf.float64).numpy()
    if candidates.shape != (8, 512) or len(fixture['cases']) != 2:
        raise ValueError('Unexpected frozen input shape')
    rows = []
    for case_index, case in enumerate(fixture['cases']):
        frozen = json.loads((args.diagnosis/('frozen_seed%d.json'%case['seed'])).read_text())
        x = np.asarray(frozen['x'])
        target = np.asarray(frozen['target'])
        beta = frozen['beta']
        t = int(frozen['time'])
        previous = np.asarray(case['rounds'][0]['x'][t-1])
        parents = .5*previous+25.*previous/(1.+previous**2)+8.*np.cos(1.2*t)
        design = np.exp(-beta*(x[:, None]-x[None, :])**2)
        for factor in (.1, .01):
            index, arm = next((i, a) for i, a in enumerate(calibration['accuracy_calibration'])
                              if a['case'] == case_index and a['stopping_factor'] == factor)
            ref_record = next(r for r in stability['cases'][case_index]['svd_reference_candidates']
                              if r['stopping_factor'] == factor)
            fit_checks = [c for c in arm['checks'] if c['name'] in ('valid', 'prediction', 'objective')]
            if len(fit_checks) != 3 or not ref_record['valid'] or not all(c['passed'] for c in fit_checks):
                raise ValueError('Invalid fitted coefficients')
            reference = np.load(args.stability/('reference_seed%d_factor%s.npy'%(case['seed'], factor)))
            row = compare(candidates[index], reference, x, beta, design, target, parents)
            objective_error = abs(row['rss']-float(np.sum((design@reference-target)**2)))
            objective_pass = objective_error <= 1e-8+1e-6*float(np.sum((design@reference-target)**2))
            row.update(seed=case['seed'], time=t, stopping_factor=factor,
                       objective_error=objective_error, objective_pass=objective_pass,
                       matched_policy_pass=bool(row['prediction_pass'] and objective_pass and
                           all(row[k]['retained_pass'] for k in ('policy', 'integral_at_x', 'integral_at_parents'))))
            rows.append(row)
    passing = [factor for factor in (.1, .01) if all(r['matched_policy_pass'] for r in rows if r['stopping_factor'] == factor)]
    summary = dict(status='PASS', scope='RBF_MATCHED_STOPPING_DIAGNOSTIC',
                   cpu_only=True, gpu_intentionally_hidden=True, numpy_role='postrun_diagnostic_only',
                   replication='INCOMPLETE', full_retry_authorized_by_result=False,
                   matched_policy_passing_factors=passing, rows=rows,
                   baseline='saved fresh-SVD solve with the same stopping factor, design and target',
                   criteria=dict(prediction_and_retained_policy_atol=1e-7, rtol=1e-6, objective_atol=1e-8),
                   wall_seconds=time.monotonic()-start)
    (args.output/'summary.json').write_text(json.dumps(summary, allow_nan=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
