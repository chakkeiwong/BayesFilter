"""Independent CPU reference and diagnostic analysis; never a runtime backend.

NumPy/SciPy are restricted to reference serialization, comparisons and reporting.
The original algorithm and the eligible TensorFlow implementation are untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from scipy.io import loadmat, savemat


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mode', choices=('form', 'solve', 'fallback'), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    fixture_path = args.reference/'fixture.json'
    fixture = json.loads(fixture_path.read_text())
    arrays = {}
    checks = []

    def flag(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise RuntimeError(name)

    def load(name):
        item = fixture['arrays'][name]
        path = args.reference/item['file']
        flag('hash_'+name, sha(path) == item['sha256'])
        return np.frombuffer(path.read_bytes(), dtype='<f8').reshape(item['shape'])

    if args.mode == 'form':
        flag('capture_schema', fixture['schema'] == 'lorenz_author_capture_v1')
        flag('source_unchanged', sha(args.output/'sources/cSMC_Resample.m') == fixture['source_sha256'])
        captured_summary = json.loads((args.reference/'summary.json').read_text())
        flag('capture_exact_source_replay', captured_summary['source_results_exactly_unchanged'])
        rows, xs, ys, betas = [], [], [], []
        for case in fixture['cases']:
            for stage in range(case['refinements']):
                prefix = 'case%d_stage%d_' % (case['case'], stage)
                x, y, beta = (load(prefix+key) for key in ('x', 'Value', 'BetaHat'))
                for t in (100, 99):
                    rows.append(dict(case=case['case'], forcing=case['forcing'],
                                     seed=case['seed'], refinement=stage+1, time=t))
                    xs.append(x[t]); ys.append(y[t]); betas.append(beta[t])
        arrays.update(x=np.stack(xs), target=np.stack(ys), captured_beta=np.stack(betas))
        flag('all_refinements_and_controls', len(rows) == 32)
    else:
        flag('cross_schema', fixture['schema'] == 'lorenz_matrix_crossover_v1')
        flag('gpu_stage_complete', fixture['stage'] == 'tensorflow')
        rows = fixture['rows']
        arrays.update({name: load(name) for name in fixture['arrays']})
    savemat(args.output/'input.mat', arrays)
    command = ['octave', '--no-gui', '--quiet',
               str(args.output/'sources/reference_lorenz_matrix_crossover.m'),
               str(Path.cwd()), str(args.output), args.mode]
    subprocess.run(command, check=True)
    result = loadmat(args.output/{'form':'reference.mat','solve':'crosssolve.mat','fallback':'fallback.mat'}[args.mode])
    index = dict(schema='lorenz_matrix_crossover_v1', stage=args.mode, rows=rows,
                 arrays={}, dtype='float64', byte_order='little',
                 input_fixture_sha256=sha(fixture_path), reference_root=str(args.reference))
    comparisons = []

    def save(name, value):
        array = np.ascontiguousarray(value, dtype='<f8')
        path = args.output/(name+'.bin')
        path.write_bytes(array.tobytes())
        index['arrays'][name] = dict(file=path.name, shape=list(array.shape), sha256=sha(path))

    def compare(name, actual, expected):
        actual = np.asarray(actual).reshape(len(rows), -1)
        expected = np.asarray(expected).reshape(len(rows), -1)
        delta = np.abs(actual-expected)
        finite = np.all(np.isfinite(actual), axis=1) & np.all(np.isfinite(expected), axis=1)
        scaled = np.max(delta/(1e-7+1e-9*np.abs(expected)), axis=1)
        record = dict(name=name, passed_rows=int(np.sum(finite & (scaled <= 1))),
                      finite_rows=int(np.sum(finite)), count=len(rows),
                      maximum_absolute=float(np.nanmax(delta)) if np.any(np.isfinite(delta)) else None,
                      per_row_maximum_scaled=[float(v) if np.isfinite(v) else None for v in scaled])
        comparisons.append(record)

    if args.mode == 'fallback':
        flag('exact_native_replay', np.array_equal(result['ordinary_beta'][0], arrays['native_beta']))
        for name in ('ordinary_beta','forced_singular_beta','gram_spectrum','gram_rcond'):
            flag('finite_'+name, np.all(np.isfinite(result[name])))
            save(name, result[name])
        exact = np.all(result['ordinary_beta'] == result['forced_singular_beta'], axis=2)
        for arm, label in enumerate(('nativeG_nativeR','nativeG_tfR','tfG_nativeR','tfG_tfR')):
            compare('forced_minimum_norm_vs_ordinary_'+label,
                    result['forced_singular_beta'][arm], result['ordinary_beta'][arm])
        diagnostics = dict(exact_equal_by_arm=exact.tolist(), exact_equal_count=int(np.sum(exact)),
            solve_count=int(exact.size),
            warning_ids=[[str(v[0]) if v.size else '' for v in row] for row in result['warning_ids']],
            cached_types=[[str(v[0]) if v.size else '' for v in row] for row in result['cached_types']],
            smallest_largest_gram_singular_ratio=(result['gram_spectrum'][:,:,-1]/result['gram_spectrum'][:,:,0]).tolist(),
            octave_source_sha256=sha(args.output/'sources/octave-6.4.0-dMatrix.cc'),
            octave_version=str(result['octave_version'][0]))
        verdict = ('OCTAVE_MINIMUM_NORM_FALLBACK_REPRODUCES_ALL_SOLVES_EXACTLY'
                   if np.all(exact) else 'OCTAVE_FALLBACK_PARTIAL_OR_MISMATCH_REVIEW_REQUIRED')
    elif args.mode == 'form':
        flag('exact_beta_replay', np.array_equal(result['native_beta'], arrays['captured_beta']))
        for key in ('native_design', 'native_gram', 'native_rhs', 'native_beta',
                    'design_singular_values', 'gram_rcond', 'direct_beta',
                    'centered_qr_beta', 'standardized_singular_values'):
            arrays[key] = result[key]
            flag('finite_'+key, np.all(np.isfinite(arrays[key])))
        native = json.loads((Path(fixture['reference_root'])/'summary.json').read_text())
        x2fx_path = Path(str(result['x2fx_path'][0]))
        expected_hash = next(value for key, value in native['dependency_sha256'].items()
                             if Path(native[key]).name == x2fx_path.name)
        flag('pinned_x2fx_dependency', sha(x2fx_path) == expected_hash)
        index['octave_version'] = str(result['octave_version'][0])
        index['x2fx_sha256'] = sha(x2fx_path)
        for name, value in arrays.items():
            save(name, value)
        verdict = 'EXACT_NATIVE_MATRIX_BASELINE_CAPTURED'
        diagnostics = {}
    else:
        for name in ('octave_cross_beta', 'lu_cross_beta', 'chol_cross_beta', 'chol_status'):
            save(name, result[name])
        flag('exact_native_cross_replay', np.array_equal(result['octave_cross_beta'][0], arrays['native_beta']))
        for arm, label in enumerate(('nativeG_nativeR', 'nativeG_tfR', 'tfG_nativeR', 'tfG_tfR')):
            beta = result['octave_cross_beta'][arm]
            compare('octave_'+label+'_vs_native', beta, arrays['native_beta'])
            compare('tf_vs_octave_'+label, arrays['tf_cross_beta'][arm], beta)
            compare('lu_vs_backslash_'+label, result['lu_cross_beta'][arm], beta)
            compare('chol_vs_backslash_'+label, result['chol_cross_beta'][arm], beta)
        compare('tf_raw_endpoint_vs_native', arrays['tf_raw_beta'], arrays['native_beta'])
        compare('tf_qr_endpoint_vs_native', arrays['tf_qr_beta'], arrays['native_beta'])
        compare('tf_qr_vs_independent_centered_qr', arrays['tf_qr_beta'], arrays['centered_qr_beta'])
        compare('extracted_tf_solve_vs_actual_endpoint', arrays['tf_cross_beta'][3], arrays['tf_raw_beta'])
        d, y = arrays['native_design'], arrays['target']
        rss = {}
        for key in ('native_beta','direct_beta','centered_qr_beta','tf_raw_beta','tf_qr_beta'):
            prediction = np.einsum('bnk,bk->bn', d, arrays[key])
            rss[key] = np.sum((prediction-y)**2, axis=1).tolist()
        diagnostics = dict(residual_sum_squares=rss,
            gram_max_absolute=np.max(np.abs(arrays['tf_gram']-arrays['native_gram']), axis=(1,2)).tolist(),
            rhs_max_absolute=np.max(np.abs(arrays['tf_rhs']-arrays['native_rhs']), axis=1).tolist(),
            design_exact_rows=np.all(arrays['tf_design']==d, axis=(1,2)).tolist(),
            design_singular_ratio=(arrays['design_singular_values'][:,-1]/arrays['design_singular_values'][:,0]).tolist(),
            chol_success_count=int(np.sum(result['chol_status']==0)))
        verdict = 'CROSSED_MATRIX_SOLVES_COMPLETE_REVIEW_REQUIRED'
    (args.output/'fixture.json').write_text(json.dumps(index, indent=2, allow_nan=False)+'\n')
    summary = dict(status='PASS', scope='LORENZ_MATRIX_CROSSOVER', replication='INCOMPLETE',
        scientific_verdict=verdict, mode=args.mode, cpu_only=True, gpu_intentionally_hidden=True,
        numpy_role='INDEPENDENT_REFERENCE_ONLY', rows=rows, checks=checks,
        comparisons=comparisons, diagnostics=diagnostics, command=command,
        input_fixture_sha256=sha(fixture_path), algorithm_seconds=time.monotonic()-started)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS', mode=args.mode, rows=len(rows),
                         comparisons=len(comparisons), scientific_verdict=verdict)))


if __name__ == '__main__':
    main()
