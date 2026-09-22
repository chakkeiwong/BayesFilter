"""CPU/base-R independent diagnostic of preserved iAPF fitting geometry."""
import argparse
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
ROOT = REPO / 'docs/plans/artifacts/iapf-representation-support-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-fit-input-isolation-20260922-01'
OPTIMIZER = REPO / 'docs/plans/artifacts/iapf-optimizer-isolation-20260922-01'
PLAN = 'docs/plans/iapf-representation-support-2026-09-22.md'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preserve(directory, source_root, name):
    source = source_root / name
    if sha(source) != read(source_root / 'manifest.json')['outputs'][name]:
        raise RuntimeError('Changed input ' + name)
    target = directory / 'inputs' / source_root.name / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def launch(attempt):
    if not attempt.isidentifier():
        raise ValueError('Unique attempt identifier required')
    record_path = ROOT / (attempt + '-launch.json')
    budget = read(ROOT / 'budget.json')
    if (len(list(ROOT.glob('*-launch.json'))) >= 4 or
            budget['phase_cpu_seconds'] + 300 > 3600 or
            budget['remaining_cpu_seconds'] < 300):
        raise RuntimeError('Budget/attempt cap')
    directory = ROOT / attempt
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    input_path = preserve(directory, PRIOR, 'attempt02_gpu_isolate/reference-input.R')
    preserve(directory, PRIOR, 'attempt02_gpu_isolate/results.json')
    controls_path = preserve(directory, OPTIMIZER, 'attempt01_cpu_reference/controls.R')
    previous_path = preserve(directory, OPTIMIZER, 'attempt01_cpu_reference/results.csv')
    sources = [str(Path(__file__).resolve().relative_to(REPO)),
               'docs/benchmarks/diagnose_iapf_representation.R', PLAN]
    hashes = {}
    for name in sources:
        source = REPO / name
        target = ROOT / (attempt + '-source') / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = sha(source)
    command = ['Rscript', '--vanilla', 'docs/benchmarks/diagnose_iapf_representation.R',
               str(input_path), str(controls_path), str(previous_path), str(directory)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    record = dict(status='running', command=command, wrapper_command=[
        'python3', 'docs/benchmarks/diagnose_iapf_representation.py', '--attempt', attempt],
        environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OPENBLAS_NUM_THREADS',
                                       'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        device='CPU independent R reference; GPU intentionally hidden', sources=hashes,
        plan=PLAN, prior_manifest=sha(PRIOR / 'manifest.json'),
        optimizer_manifest=sha(OPTIMIZER / 'manifest.json'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        seeds='No new randomness; preceding captured clouds and targets',
        data='Hash-checked phase7 inputs and phase8 controls; no new observations',
        inputs={str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()})
    save(record_path, record)
    with (ROOT / (attempt + '.log')).open('x') as log:
        try:
            code = subprocess.run(command, cwd=REPO, env=env, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=300).returncode
        except subprocess.TimeoutExpired:
            code = 124
    elapsed = time.monotonic() - started
    if code == 0:
        with (directory / 'results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        exact = [r for r in rows if r['target'] == 'exact']
        numeric_checks = ['projection_identity_error', 'qr_parity_error',
                          'curvature_check_error', 'whitening_error']
        maxima = {k: max(float(r[k]) for r in rows) for k in numeric_checks}
        summary = dict(rows=len(rows), exact_rows=len(exact), maxima=maxima,
            exact_log_residual=max(float(r['full_log_residual']) for r in exact),
            exact_center_error=max(float(r['full_center_error']) for r in exact),
            exact_covariance_error=max(float(r['full_covariance_error']) for r in exact),
            finite_checks=all(math.isfinite(float(r[k])) for r in rows for k in numeric_checks),
            R_version=(directory / 'R-version.txt').read_text().strip())
        summary['passed'] = (len(rows) == 32 and len(exact) == 16 and summary['finite_checks']
            and all(r['rank_pass'] == 'TRUE' for r in rows)
            and all(r['full_positive_definite'] == 'TRUE' for r in exact)
            and maxima['projection_identity_error'] <= 1e-8 and maxima['qr_parity_error'] <= 1e-8
            and maxima['curvature_check_error'] <= 1e-5 and maxima['whitening_error'] <= 1e-8
            and summary['exact_log_residual'] <= 1e-9 and summary['exact_center_error'] <= 1e-8
            and summary['exact_covariance_error'] <= 1e-8)
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
