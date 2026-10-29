"""CPU-only post-run reference diagnostic; never a runtime policy selector."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import svdvals
from scipy.special import logsumexp
import tensorflow as tf
from reference_rbf_svd_nnls import svd_nnls


def close(actual, reference):
    return bool(np.all(np.abs(actual-reference) <= 1e-7+1e-6*np.abs(reference)))


def log_mixture(points, knots, beta, weights, variance=0.):
    denominator = 1.+2.*beta*variance
    with np.errstate(divide='ignore'):
        return (-.5*np.log(denominator)+logsumexp(
            np.log(weights)[None, :]-beta/denominator*(points[:, None]-knots[None, :])**2,
            axis=1))


def compare(coef, reference, x, beta, design, target, parents):
    own_ix = np.argsort(-coef, kind='stable')[:16]
    ref_ix = np.argsort(-reference, kind='stable')[:16]
    prediction = design@coef
    ref_prediction = design@reference
    row = dict(coefficient_error=float(np.max(np.abs(coef-reference))),
               prediction_error=float(np.max(np.abs(prediction-ref_prediction))),
               prediction_pass=close(prediction, ref_prediction),
               rss=float(np.sum((prediction-target)**2)),
               nonzero_coefficients=int(np.count_nonzero(coef)),
               top16_overlap=len(set(own_ix)&set(ref_ix)))
    for label, points, variance in (('policy', x, 0.), ('integral_at_x', x, 10.),
                                    ('integral_at_parents', parents, 10.)):
        full = log_mixture(points, x, beta, coef, variance)
        ref_full = log_mixture(points, x, beta, reference, variance)
        retained = log_mixture(points, x[own_ix], beta, coef[own_ix], variance)
        ref_retained = log_mixture(points, x[ref_ix], beta, reference[ref_ix], variance)
        row[label] = dict(full_log_error=float(np.max(np.abs(full-ref_full))),
                          retained_log_error=float(np.max(np.abs(retained-ref_retained))),
                          full_pass=close(full, ref_full), retained_pass=close(retained, ref_retained),
                          candidate_max_removed_mass_fraction=float(np.max(-np.expm1(retained-full))),
                          reference_max_removed_mass_fraction=float(np.max(-np.expm1(ref_retained-ref_full))))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'fixture', 'diagnosis', 'candidate'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    fixture = json.loads(args.fixture.read_text())
    candidate_summary = json.loads((args.candidate/'summary.json').read_text())
    candidates = tf.io.parse_tensor(tf.io.read_file(str(args.candidate/'frozen_coefficients.tensor')), tf.float64).numpy()
    arms = candidate_summary['accuracy_calibration']
    if candidates.shape != (8, 512) or len(arms) != 8:
        raise ValueError('Unexpected saved policy-calibration shape')
    cases = []
    replay_checks = []
    all_reference_valid = True
    for case_index, case in enumerate(fixture['cases']):
        frozen_path = args.diagnosis/('frozen_seed%d.json'%case['seed'])
        frozen = json.loads(frozen_path.read_text())
        x = np.asarray(frozen['x'])
        beta = frozen['beta']
        target = np.asarray(frozen['target'])
        reference = np.asarray(frozen['reference_coefficients'])
        t = int(frozen['time'])
        previous = np.asarray(case['rounds'][0]['x'][t-1])
        parents = .5*previous+25.*previous/(1.+previous**2)+8.*np.cos(1.2*t)
        design = np.exp(-beta*(x[:, None]-x[None, :])**2)
        singular = svdvals(design)
        rank_threshold = np.finfo(float).eps*max(design.shape)*singular[0]
        baseline_tolerance = 10.*np.finfo(float).eps*np.linalg.norm(design, 1)*len(x)
        gpu_rows = []
        reference_rows = []
        for index, arm in enumerate(arms):
            if arm['case'] != case_index:
                continue
            row = compare(candidates[index], reference, x, beta, design, target, parents)
            row.update(stopping_factor=arm['stopping_factor'])
            gpu_rows.append(row)
        for factor in (.1, .01, .001):
            try:
                coef, info = svd_nnls(design, target, stopping_factor=factor)
                gradient = design.T@(design@coef-target)
                dual = max(0., float(-np.min(gradient)))
                complementarity = float(np.max(np.abs(coef*gradient)))
                valid = bool(np.all(np.isfinite(coef)) and np.min(coef) >= 0.
                             and dual <= 20.*baseline_tolerance
                             and complementarity <= 20.*baseline_tolerance*max(1., float(np.max(coef))))
                row = compare(coef, reference, x, beta, design, target, parents)
                row.update(stopping_factor=factor, valid=valid, dual_error=dual,
                           complementarity=complementarity, solver=info)
                np.save(args.output/('reference_seed%d_factor%s.npy'%(case['seed'], factor)), coef)
                if factor == .01:
                    replay_checks.append(dict(seed=case['seed'], coefficient_error=row['coefficient_error'],
                        passed=bool(valid and row['prediction_pass']
                                    and all(row[key]['retained_pass'] for key in
                                            ('policy', 'integral_at_x', 'integral_at_parents')))))
            except RuntimeError as exc:
                row = dict(stopping_factor=factor, valid=False, failure=str(exc))
                if factor == .01:
                    replay_checks.append(dict(seed=case['seed'], passed=False, failure=str(exc)))
            all_reference_valid = all_reference_valid and row['valid']
            reference_rows.append(row)
        cases.append(dict(seed=case['seed'], time=t, beta=beta,
                          spectrum=singular.tolist(), largest_singular=float(singular[0]),
                          smallest_singular=float(singular[-1]),
                          effective_rank=int(np.sum(singular > rank_threshold)),
                          rank_threshold=float(rank_threshold), gpu_candidates=gpu_rows,
                          svd_reference_candidates=reference_rows))
    status = 'PASS' if len(replay_checks) == 2 and all(c['passed'] for c in replay_checks) and all_reference_valid else 'FAIL'
    summary = dict(status=status, scope='RBF_COEFFICIENT_STABILITY_DIAGNOSIS',
                   replication='INCOMPLETE', runtime_policy_unchanged=True,
                   cpu_only=True, gpu_intentionally_hidden=True,
                   numpy_role='postrun_diagnostic_and_independent_reference_only',
                   criteria=dict(atol=1e-7, rtol=1e-6, role='unchanged frozen-function parity; no promotion'),
                   reference_replay_checks=replay_checks, all_reference_valid=all_reference_valid,
                   cases=cases, wall_seconds=time.monotonic()-start)
    (args.output/'summary.json').write_text(json.dumps(summary, allow_nan=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}))
    if status != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
