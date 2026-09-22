"""Independent CPU/R oracle-start and known-Gaussian population diagnosis."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from diagnose_iapf_representation import REPO, PRIOR, OPTIMIZER, read, save, sha, preserve

ROOT = REPO / 'docs/plans/artifacts/iapf-oracle-start-population-20260922-01'
PLAN = 'docs/plans/iapf-oracle-start-population-2026-09-22.md'


def launch(attempt):
    if not attempt.isidentifier():
        raise ValueError('Unique attempt identifier required')
    budget = read(ROOT / 'budget.json')
    if (len(list(ROOT.glob('*-launch.json'))) >= 4 or
            budget['phase_cpu_seconds'] + 600 > 3600 or budget['remaining_cpu_seconds'] < 600):
        raise RuntimeError('Budget/attempt cap')
    directory = ROOT / attempt
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    input_path = preserve(directory, PRIOR, 'attempt02_gpu_isolate/reference-input.R')
    preserve(directory, PRIOR, 'attempt02_gpu_isolate/results.json')
    control_path = preserve(directory, OPTIMIZER, 'attempt01_cpu_reference/controls.R')
    previous_path = preserve(directory, OPTIMIZER, 'attempt01_cpu_reference/results.csv')
    sources = [str(Path(__file__).resolve().relative_to(REPO)),
               'docs/benchmarks/diagnose_iapf_oracle_start.R',
               'docs/benchmarks/diagnose_iapf_representation.py', PLAN]
    hashes = {}
    for name in sources:
        source = REPO / name
        target = ROOT / (attempt + '-source') / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha(source)
    command = ['Rscript', '--vanilla', 'docs/benchmarks/diagnose_iapf_oracle_start.R',
               str(input_path), str(control_path), str(previous_path), str(directory)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    record = dict(status='running', command=command, wrapper_command=[
        'python3', 'docs/benchmarks/diagnose_iapf_oracle_start.py', '--attempt', attempt],
        environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS',
                                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        device='CPU independent R reference; GPU intentionally hidden', sources=hashes, plan=PLAN,
        prior_manifest=sha(PRIOR / 'manifest.json'), optimizer_manifest=sha(OPTIMIZER / 'manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        seeds={'oracle_start':'no new randomness','Gaussian_controls':list(range(9101, 9133)),
               'policy':'R set.seed per d/r cell; common standard normal draws across r and delta'},
        data='Hash-checked fitting inputs; analytically declared Gaussian controls',
        inputs={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()})
    record_path = ROOT / (attempt + '-launch.json')
    save(record_path, record)
    with (ROOT / (attempt + '.log')).open('x') as log:
        try:
            code = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=600).returncode
        except subprocess.TimeoutExpired:
            code = 124
    elapsed = time.monotonic() - started
    if code == 0:
        tables = {}
        for name in ['oracle-start', 'population-formulas', 'population-draws', 'narrow-limit']:
            with (directory / (name + '.csv')).open() as stream:
                tables[name] = list(csv.DictReader(stream))
        checks = dict(oracle_rows=len(tables['oracle-start'])==64,
            oracle_finite=all(r['finite']=='TRUE' for r in tables['oracle-start']),
            oracle_start_identity=max(float(r['initial_identity_error']) for r in tables['oracle-start'])<=1e-8,
            formulas_36=len(tables['population-formulas'])==36,
            formula_identity=max(float(r['formula_error']) for r in tables['population-formulas'])<=1e-10,
            samples_1152=len(tables['population-draws'])==1152,
            identical_guides=all(float(r['sample_shape'])<=1e-12 for r in tables['population-draws'] if r['delta']=='0'),
            weight_checks=all(r['finite']=='TRUE' and abs(float(r['weight_sum'])-1)<=1e-12
                              and 1-1e-10<=float(r['ESS'])<=1000+1e-10 for r in tables['population-draws']),
            narrow_18=len(tables['narrow-limit'])==18)
        summary = dict(passed=all(checks.values()), checks=checks,
                       R_version=(directory / 'R-version.txt').read_text().strip())
        save(directory / 'summary.json', summary)
        if not summary['passed']:
            code = 1
    record.update(status='finished', exit_code=code, wall_seconds=elapsed)
    save(record_path, record)
    budget = read(ROOT / 'budget.json')
    budget['phase_cpu_seconds'] += elapsed
    budget['remaining_cpu_seconds'] -= elapsed
    save(ROOT / 'budget.json', budget)
    print(json.dumps(dict(exit_code=code, wall_seconds=elapsed, artifact=str(directory))))
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True)
    raise SystemExit(launch(parser.parse_args().attempt))
