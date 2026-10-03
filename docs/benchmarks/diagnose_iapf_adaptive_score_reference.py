"""Bounded CPU-only R reference comparison of adaptive reconstructed fitters."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'docs/plans/artifacts/iapf-adaptive-score-reference-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-paper-score-transfer-20260922-01'
PARITY = REPO / 'docs/plans/artifacts/iapf-covariance-floor-factorial-20260922-01'
PLAN = 'docs/plans/iapf-adaptive-score-reference-2026-09-22.md'
SCRIPT = 'docs/benchmarks/run_iapf_adaptive_score_reference.R'
PREFLIGHT = 'docs/benchmarks/run_iapf_paper_score_transfer.R'
REFERENCES = {
    'docs/benchmarks/reference_iapf_paper.R': '979c84f9dbe906742a8101b94ab5391c3268b2b46c81a903b1d5529022916887',
    'docs/benchmarks/reference_iapf_author_choices.R': '420455b782efa415783eba9d0ae3fbd3483a7994c24bb76d4e126900b081d2b4',
    'docs/benchmarks/reference_iapf_constrained_diagnostic.R': '7ef3fd67dd198c6443634494ef4f2d85cf1abdfbafd1c2477ca294509a404da6',
}
DIMENSIONS = [5, 10, 20, 40, 80]
LEARNED = [f'{f}_{c}' for f in ['score', 'qr'] for c in ['after_k', 'first_full_window']]
HEURISTICS = ['bootstrap', 'fully_adapted', 'current_observation', 'full_oracle']


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_inputs():
    for path, digest in REFERENCES.items():
        if sha(REPO / path) != digest:
            raise RuntimeError('Frozen reference changed: ' + path)
    for prior in [PRIOR, PARITY]:
        for path, digest in read(prior / 'manifest.json')['outputs'].items():
            if sha(prior / path) != digest:
                raise RuntimeError('Prior evidence changed: ' + str(prior / path))
        if not read(prior / 'verification.json')['passed']:
            raise RuntimeError('Prior verification failed')


def reserve(count):
    budget = read(ROOT / 'budget.json')
    if (len(list(ROOT.glob('*-launch.json'))) + count > 12 or
            budget['phase_cpu_seconds'] + count * 600 > 10800 or
            budget['remaining_cpu_seconds'] < count * 600):
        raise RuntimeError('Phase launch or compute budget exhausted')


def charge(seconds):
    budget = read(ROOT / 'budget.json')
    budget['phase_cpu_seconds'] += seconds
    budget['remaining_cpu_seconds'] -= seconds
    save(ROOT / 'budget.json', budget)


def launch(name, script, arguments):
    directory = ROOT / name
    directory.mkdir(exist_ok=False)
    sources = [PLAN, SCRIPT, PREFLIGHT, 'docs/benchmarks/diagnostic_iapf_paper_score.R',
               'docs/benchmarks/diagnose_iapf_adaptive_score_reference.py', *REFERENCES]
    hashes = {}
    for relative in sources:
        destination = ROOT / (name + '-source') / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
        hashes[relative] = sha(destination)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = ['Rscript', '--vanilla', script, *map(str, arguments(directory))]
    record = dict(status='running', command=command, plan=PLAN,
                  result=str(ROOT / 'result.md'), sources=hashes,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                  cpu_only=True, gpu_intentionally_hidden=True,
                  R=subprocess.check_output(['Rscript', '--version'], text=True, stderr=subprocess.STDOUT).strip(),
                  data_version='GJL2017 section5.2; one fresh generated observation set per dimension',
                  seeds='data961000+d;learner962000+100*d+r;heuristic963000+100*d+r',
                  prior_manifests={str(p.relative_to(REPO)): sha(p / 'manifest.json') for p in [PRIOR, PARITY]})
    path = ROOT / (name + '-launch.json')
    save(path, record)
    started = time.monotonic()
    with (directory / 'R.log').open('x') as stream:
        try:
            code = subprocess.run(command, cwd=REPO, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=600).returncode
        except subprocess.TimeoutExpired:
            code = 124
    record.update(status='finished', exit_code=code, wall_seconds=time.monotonic() - started)
    # Source snapshots must describe the actual files consumed, including dirty code.
    record['sources_unchanged'] = all(sha(REPO / p) == h for p, h in hashes.items())
    save(path, record)
    print(json.dumps({k: record[k] for k in ['exit_code', 'wall_seconds', 'sources_unchanged']} | {'attempt': name}), flush=True)
    return record


def preflight():
    verify_inputs()
    reserve(1)
    result = launch('attempt01_preflight', PREFLIGHT,
                    lambda directory: ['preflight', directory, PARITY / 'attempt01_gpu_factorial/reference-input.R'])
    charge(result['wall_seconds'])
    if result['exit_code'] or not result['sources_unchanged']:
        raise RuntimeError('Preflight failed; inspect preserved log')


def cases():
    verify_inputs()
    pre = read(ROOT / 'attempt01_preflight-launch.json')
    if pre['exit_code'] or not pre['sources_unchanged'] or len(rows(ROOT / 'attempt01_preflight/checks.csv')) != 48:
        raise RuntimeError('Preflight required')
    reserve(10)
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(launch, f'case_d{d}_b{b}', SCRIPT,
                               lambda directory, d=d, b=b: [directory, d, b]): (d, b)
                   for d in DIMENSIONS for b in [1, 2]}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            result = future.result()
            charge(result['wall_seconds'])
            if result['exit_code'] or not result['sources_unchanged']:
                failures.append(dict(case=futures[future], exit_code=result['exit_code'],
                                     sources_unchanged=result['sources_unchanged']))
                for pending in futures:
                    pending.cancel()
    save(ROOT / 'case-launch-summary.json', dict(passed=not failures, failures=failures, planned_cases=10))
    if failures:
        raise RuntimeError('Case failure requires classification; queued jobs canceled')


def number(value):
    return None if value in ['NA', ''] else float(value)


def results():
    started = time.monotonic()
    verify_inputs()
    if not read(ROOT / 'case-launch-summary.json')['passed']:
        raise RuntimeError('Unresolved case failure')
    records, fit_summary, checks = [], [], {}
    numeric = ['d', 'batch', 'replicate', 'data_seed', 'run_seed', 'particles', 'stop_iteration',
               'filter_calls', 'fit_calls', 'particle_filter_work', 'log_likelihood', 'kalman',
               'log_error', 'ratio', 'resampling_count', 'min_ess', 'floor_probability_max', 'cpu_seconds']
    for d in DIMENSIONS:
        for b in [1, 2]:
            name = f'case_d{d}_b{b}'
            directory = ROOT / name
            case = rows(directory / 'records.csv')
            checks[name + '/rows'] = len(case) == 40
            expected = {(r, m) for r in (range(1, 6) if b == 1 else range(6, 11)) for m in LEARNED + HEURISTICS}
            checks[name + '/complete_labels'] = {(int(x['replicate']), x['method']) for x in case} == expected
            for row in case:
                for key in numeric:
                    row[key] = number(row[key])
                row['controller_wiring'] = row['controller_wiring'] == 'TRUE'
                checks[name + '/' + row['method'] + '/r' + str(int(row['replicate']))] = (
                    row['controller_wiring'] and row['status'] in ['complete', 'fit_rejection', 'particle_cap', 'iteration_cap'] and
                    (row['status'] != 'complete' or all(math.isfinite(row[k]) for k in ['log_error', 'ratio'])) and
                    row['cpu_seconds'] >= 0)
            records.extend(case)
            checks[name + '/oracle'] = all(abs(x['log_error']) <= 1e-8 for x in case if x['method'] == 'full_oracle')
            for path in directory.glob('fit_diagnostics_*.csv'):
                fits = rows(path)
                first = min(int(x['fit_call']) for x in fits)
                last = max(int(x['fit_call']) for x in fits)
                checks[name + '/' + path.stem + '/all_times'] = len(fits) == last * 100
                checks[name + '/' + path.stem + '/probabilities'] = all(
                    x['gaussian_probability_min'] == 'NA' or
                    0 <= float(x['gaussian_probability_min']) <= float(x['gaussian_probability_mean']) <= 1 + 1e-14 for x in fits)
                for stage, index in [('first', first), ('last', last)]:
                    selected = [x for x in fits if int(x['fit_call']) == index]
                    valid = all(x['actual_valid'] == x['negligible_valid'] == 'TRUE' for x in selected)
                    fit_summary.append(dict(d=d, replicate=int(selected[0]['replicate']), method=selected[0]['method'],
                        stage=stage, fit_call=index, N=int(selected[0]['N']), valid=valid,
                        coefficient_difference=max(float(x['coefficient_difference']) for x in selected) if valid else None,
                        minimum_gaussian_probability=min(float(x['gaussian_probability_min']) for x in selected) if valid else None,
                        mean_gaussian_probability=sum(float(x['gaussian_probability_mean']) for x in selected) / 100 if valid else None))
    conditional, heuristic = [], []
    for d in DIMENSIONS:
        by_method = {}
        for method in LEARNED + HEURISTICS:
            selected = [r for r in records if r['d'] == d and r['method'] == method]
            complete = len(selected) == 10 and all(r['status'] == 'complete' for r in selected)
            values = [r['ratio'] for r in selected] if complete else []
            mean = sum(values) / 10 if complete else None
            item = dict(d=d, method=method, n=len(selected), completed=sum(r['status'] == 'complete' for r in selected),
                complete=complete, ratio_mean=mean, ratio_sd=math.sqrt(sum((v-mean)**2 for v in values)/9) if complete else None,
                relative_mse=sum((v-1)**2 for v in values)/10 if complete else None,
                log_mse=sum(r['log_error']**2 for r in selected)/10 if complete else None,
                final_particles=sorted({int(r['particles']) for r in selected if r['particles'] is not None}),
                mean_particle_filter_work=sum(r['particle_filter_work'] for r in selected)/len(selected),
                cpu_seconds=sum(r['cpu_seconds'] for r in selected),
                statuses={s: sum(r['status'] == s for r in selected) for s in sorted({r['status'] for r in selected})})
            item['relative_rmse'] = math.sqrt(item['relative_mse']) if complete else None
            conditional.append(item)
            by_method[method] = item
        for method in LEARNED:
            value = by_method[method]['relative_mse']
            losses = [h for h in HEURISTICS[:3] if value is not None and value > by_method[h]['relative_mse']]
            heuristic.append(dict(d=d, method=method, losses=losses,
                promotion_veto=value is None or bool(losses), ranking_statistically_supported=False,
                exact_oracle_gap=value, basis='conditional observed relative MSE; ten independent complete learner seeds'))
    for path in ROOT.glob('*-launch.json'):
        record = read(path)
        checks[path.stem + '/exit'] = record['exit_code'] == 0 and record['sources_unchanged']
        for source, digest in record['sources'].items():
            checks[path.stem + '/' + source] = sha(ROOT / (path.stem.replace('-launch', '') + '-source') / source) == digest
    for filename, value in [('records.json', records), ('conditional-summary.json', conditional),
                            ('heuristic-dominance.json', heuristic), ('fit-summary.json', fit_summary)]:
        save(ROOT / filename, value)
    save(ROOT / 'verification.json', dict(passed=all(checks.values()), checks=checks))
    charge(time.monotonic() - started)
    print(json.dumps(dict(passed=all(checks.values()), checks=len(checks), records=len(records),
                          candidate_failures=sum(r['status'] != 'complete' for r in records))))
    if not all(checks.values()):
        raise RuntimeError('Terminal verification failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'cases', 'results'], required=True)
    arguments = parser.parse_args()
    {'preflight': preflight, 'cases': cases, 'results': results}[arguments.mode]()
