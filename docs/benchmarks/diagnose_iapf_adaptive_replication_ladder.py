"""CPU-only diagnostic supervisor: complete adaptive R learners, bounded budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01'
PRIOR = REPO / 'docs/plans/artifacts/iapf-adaptive-score-reference-20260922-01'
PLAN = 'docs/plans/iapf-adaptive-replication-ladder-2026-09-22.md'
SCRIPT = 'docs/benchmarks/run_iapf_adaptive_score_reference.R'
REPLAY = 'docs/benchmarks/check_iapf_adaptive_reference_replay.R'
STATISTICS = 'docs/benchmarks/summarize_iapf_adaptive_replication.R'
DRIVER = 'docs/benchmarks/diagnose_iapf_adaptive_replication_ladder.py'
REFERENCES = {
    'docs/benchmarks/reference_iapf_paper.R': '979c84f9dbe906742a8101b94ab5391c3268b2b46c81a903b1d5529022916887',
    'docs/benchmarks/reference_iapf_author_choices.R': '420455b782efa415783eba9d0ae3fbd3483a7994c24bb76d4e126900b081d2b4',
    'docs/benchmarks/reference_iapf_constrained_diagnostic.R': '7ef3fd67dd198c6443634494ef4f2d85cf1abdfbafd1c2477ca294509a404da6',
}
SOURCES = [PLAN, SCRIPT, REPLAY, STATISTICS, DRIVER, 'docs/benchmarks/diagnostic_iapf_paper_score.R', *REFERENCES]
METHODS = ['score_after_k', 'qr_after_k', 'bootstrap', 'fully_adapted', 'current_observation', 'full_oracle']
DIMENSIONS = [5, 10, 20, 40, 80]
PHASE_SECONDS = 36 * 3600


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def verify_prior():
    for path, expected in REFERENCES.items():
        if sha(REPO / path) != expected:
            raise RuntimeError('Frozen reference changed: ' + path)
    if not read(PRIOR / 'verification.json')['passed']:
        raise RuntimeError('Prior phase did not pass')
    for relative, expected in read(PRIOR / 'manifest.json')['outputs'].items():
        if sha(PRIOR / relative) != expected:
            raise RuntimeError('Prior evidence changed: ' + relative)


def reserve(name, cap):
    budget = read(ROOT / 'budget.json')
    reservations = budget.setdefault('reservations', {})
    reserved = sum(reservations.values())
    if (budget['launches'] >= 520 or name in reservations or
            budget['phase_cpu_seconds'] + reserved + cap > PHASE_SECONDS or
            budget['remaining_cpu_seconds'] < reserved + cap):
        raise RuntimeError('CPU budget/attempt cap prevents another launch')
    reservations[name] = cap
    budget['reserved_cpu_seconds'] = reserved + cap
    budget['launches'] += 1
    save(ROOT / 'budget.json', budget)


def settle(record):
    budget = read(ROOT / 'budget.json')
    if record['name'] not in budget.get('reservations', {}):
        raise RuntimeError('Missing budget reservation for completed attempt')
    del budget['reservations'][record['name']]
    budget['reserved_cpu_seconds'] = sum(budget['reservations'].values())
    budget['phase_cpu_seconds'] += record['wall_seconds']
    budget['remaining_cpu_seconds'] -= record['wall_seconds']
    save(ROOT / 'budget.json', budget)
    record['budget_charged'] = True
    save(ROOT / (record['name'] + '-launch.json'), record)


def launch(name, script, arguments, cap):
    directory = ROOT / name
    directory.mkdir(exist_ok=False)
    hashes = {}
    for relative in SOURCES:
        destination = ROOT / (name + '-source') / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
        hashes[relative] = sha(destination)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    command = ['Rscript', '--vanilla', script, *map(str, arguments(directory))]
    path = ROOT / (name + '-launch.json')
    record = dict(name=name, status='running', started_utc=now(), cap_seconds=cap,
                  command=command, plan=PLAN, result=str(ROOT / 'result.md'), sources=hashes,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  environment={k: env[k] for k in ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']},
                  cpu_only=True, gpu_intentionally_hidden=True, budget_charged=False,
                  R=subprocess.check_output(['Rscript', '--version'], text=True, stderr=subprocess.STDOUT).strip(),
                  data_version='GJL2017 section5.2 independent generated observations',
                  seeds='data971000+d;learner972000+10000*d+r;heuristic973000+10000*d+r;preflight uses phase15 seeds',
                  prior_manifest_sha256=sha(PRIOR / 'manifest.json'))
    started = time.monotonic()
    with (directory / 'R.log').open('x') as stream:
        try:
            child = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
            record['pid'] = child.pid
            save(path, record)
            try:
                code = child.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
                code = 124
        except OSError as error:
            record['launch_error'] = str(error)
            code = 125
    record.update(status='finished', exit_code=code, finished_utc=now(), wall_seconds=time.monotonic() - started,
                  sources_unchanged=all(sha(REPO / p) == h for p, h in hashes.items()))
    record['outputs'] = {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*') if p.is_file()}
    save(path, record)
    print(json.dumps(dict(attempt=name, exit_code=code, wall_seconds=record['wall_seconds'],
                          sources_unchanged=record['sources_unchanged'])), flush=True)
    return record


def run_single(name, script, arguments, cap=900):
    reserve(name, cap)
    record = launch(name, script, arguments, cap)
    settle(record)
    if record['exit_code'] or not record['sources_unchanged']:
        raise RuntimeError('Launch failed: ' + name)
    return record


def verify_batch(directory, d, first, last, phase15=False):
    records = csv_rows(directory / 'records.csv')
    expected = {(r, m) for r in range(first, last + 1) for m in METHODS}
    checks = dict(complete_labels={(int(r['replicate']), r['method']) for r in records} == expected,
                  record_count=len(records) == len(expected), all_d=all(int(r['d']) == d for r in records))
    checks['statuses_and_wiring'] = all(r['status'] in ['complete', 'fit_rejection', 'particle_cap', 'iteration_cap']
        and r['controller_wiring'] == 'TRUE' for r in records)
    checks['finite_completed'] = all(r['status'] != 'complete' or
        all(math.isfinite(float(r[k])) for k in ['log_likelihood', 'log_error', 'ratio']) for r in records)
    checks['oracle_kalman'] = all(abs(float(r['log_error'])) <= 1e-8 for r in records if r['method'] == 'full_oracle')
    checks['seed_identity'] = all(int(r['data_seed']) == (961000 + d if phase15 else 971000 + d) and
        int(r['run_seed']) == ((962000 if r['method'] in METHODS[:2] else 963000) + 100*d + int(r['replicate'])
            if phase15 else (972000 if r['method'] in METHODS[:2] else 973000) + 10000*d + int(r['replicate'])) for r in records)
    for r in records:
        label = int(r['replicate'])
        prefix = 'learner' if r['method'] in METHODS[:2] else 'heuristic'
        checks[f'{prefix}_{r["method"]}_r{label:02}'] = (directory / f'{prefix}_{r["method"]}_r{label:02}.rds').is_file()
        if r['method'] == 'score_after_k':
            fits = csv_rows(directory / f'fit_diagnostics_score_after_k_r{label:02}.csv')
            checks[f'score_fits_r{label}'] = len(fits) == int(r['fit_calls']) * 100
            checks[f'score_probabilities_r{label}'] = all(x['gaussian_probability_min'] == 'NA' or
                0 <= float(x['gaussian_probability_min']) <= float(x['gaussian_probability_mean']) <= 1 + 1e-14 for x in fits)
    save(directory / 'verification.json', dict(passed=all(checks.values()), checks=checks))
    if not all(checks.values()):
        raise RuntimeError('Batch verification failed: ' + str(directory))
    return records


def preflight():
    verify_prior()
    replay_records = []
    def fresh(base):
        if not (ROOT/base).exists():
            return base
        attempt = 2
        while (ROOT/f'{base}_attempt{attempt:02}').exists():
            attempt += 1
        return f'{base}_attempt{attempt:02}'
    for d in [5, 80]:
        name = fresh(f'preflight_d{d}_replay')
        result = run_single(name, SCRIPT,
            lambda directory, d=d: [directory, d, 1, 1, 1, 961000+d, 962000+100*d, 963000+100*d, 'after_k'])
        verify_batch(ROOT / name, d, 1, 1, phase15=True)
        run_single(fresh(f'preflight_d{d}_verification'), REPLAY,
            lambda directory, name=name, d=d: [ROOT/name, PRIOR/f'case_d{d}_b1', directory/'exact-replay.csv'])
        replay_records.append(result['name'])
    # Verify the uncertainty helper's central statistics against independent
    # standard-library calculations on preserved data before the long ladder.
    records = []
    for d in DIMENSIONS:
        for batch in [1, 2]:
            records.extend(r for r in csv_rows(PRIOR/f'case_d{d}_b{batch}/records.csv') if r['method'] in METHODS)
    statistics_input = ROOT / (fresh('preflight_statistics') + '-input.csv')
    with statistics_input.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    statistics = run_single(fresh('preflight_statistics'), STATISTICS,
        lambda directory: [statistics_input, directory, 10])
    statistic_checks = {}
    for row in csv_rows(ROOT/statistics['name']/'conditional-summary.csv'):
        selected = [r for r in records if r['d'] == row['d'] and r['method'] == row['method']]
        ratios = [float(r['ratio']) for r in selected]
        mean = sum(ratios)/10
        expected = dict(ratio_mean=mean, ratio_sd=math.sqrt(sum((v-mean)**2 for v in ratios)/9),
                        relative_rmse=math.sqrt(sum((v-1)**2 for v in ratios)/10))
        for key, value in expected.items():
            statistic_checks[row['d']+'/'+row['method']+'/'+key] = math.isclose(float(row[key]), value, abs_tol=1e-12, rel_tol=1e-12)
        for prefix in ['mean', 'sd', 'rmse']:
            statistic_checks[row['d']+'/'+row['method']+'/'+prefix+'_interval'] = math.isfinite(float(row[prefix+'_lower'])) and float(row[prefix+'_lower']) <= float(row[prefix+'_upper'])
    save(ROOT/'preflight-statistics-verification.json', dict(passed=all(statistic_checks.values()), checks=statistic_checks))
    if not all(statistic_checks.values()):
        raise RuntimeError('Uncertainty helper verification failed')
    save(ROOT / 'preflight.json', dict(passed=True, same_seed_replays=replay_records,
        current_sources={p: sha(REPO / p) for p in SOURCES}, exact_history_guides_final_and_diagnostics=True,
        R_likelihood_and_controller_reference='frozen phase15 snapshots'))
    checkpoint('preflight_passed', 0, [], 'Ready to execute100/300/1000 without further approval.')


def checkpoint(status, stage, active, detail):
    budget = read(ROOT / 'budget.json')
    state = dict(status=status, pid=os.getpid(), updated_utc=now(), stage=stage,
                 active_attempts=active, detail=detail, budget=budget,
                 completed_batches=len(read(ROOT/'completed-batches.json')) if (ROOT/'completed-batches.json').exists() else 0)
    save(ROOT / 'checkpoint.json', state)
    (ROOT / 'checkpoint.md').write_text(
        f'# Adaptive R replication checkpoint\n\nStatus: {status}; target stage:{stage}; PID:{os.getpid()}.\n'
        f'Updated UTC:{state["updated_utc"]}. {detail}\n'
        f'Completed batches:{state["completed_batches"]}; active:{", ".join(active) or "none"}.\n'
        f'Remaining CPU:{budget["remaining_cpu_seconds"]/3600:.6f}h; GPU:{budget["remaining_gpu_seconds"]/3600:.6f}h.\n'
        f'Phase used:{budget["phase_cpu_seconds"]/3600:.6f} of36 worker hours; reserved:{budget["reserved_cpu_seconds"]:.0f}s.\n'
        'Question: conditional complete-learner likelihood accuracy and Monte Carlo uncertainty.\n'
        'Two CPU workers; GPU hidden. Both reconstructed fitters, after_k, fresh final APF.\n'
        'Candidate failures stay recorded and do not stop stages. Source/numerical/artifact\n'
        'veto, repeated infrastructure failure or budget exhaustion stops with evidence.\n'
        'Not Eq15, author code, full paper replication, nonlinear or default evidence.\n')


def verify_launch(record):
    if record['exit_code'] or not record['sources_unchanged']:
        raise RuntimeError('Invalid attempt: ' + record['name'])
    for path, digest in record['outputs'].items():
        if sha(ROOT / record['name'] / path) != digest:
            raise RuntimeError('Completed output changed: ' + path)


def stage_summary(stage, completed):
    directory = ROOT / f'stage{stage}'
    directory.mkdir(exist_ok=False)
    selected = [v for v in completed.values() if v['last'] <= stage]
    all_rows = []
    for item in selected:
        record = read(ROOT / (item['attempt'] + '-launch.json'))
        verify_launch(record)
        all_rows.extend(verify_batch(ROOT / item['attempt'], item['d'], item['first'], item['last']))
    expected = {(d, r, m) for d in DIMENSIONS for r in range(1, stage + 1) for m in METHODS}
    if len(all_rows) != len(expected) or {(int(r['d']), int(r['replicate']), r['method']) for r in all_rows} != expected:
        raise RuntimeError('Stage label coverage failed')
    with (directory / 'records.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    stats = run_single(f'stage{stage}_statistics', STATISTICS,
                       lambda result_dir: [directory/'records.csv', result_dir, stage])
    summary = csv_rows(ROOT / stats['name'] / 'conditional-summary.csv')
    comparisons = csv_rows(ROOT / stats['name'] / 'paired-comparisons.csv')
    if len(summary) != 30 or len(comparisons) != 30:
        raise RuntimeError('Incomplete uncertainty summaries')
    save(directory / 'heuristic-dominance.json', comparisons)
    save(directory / 'decision.json', dict(stage=stage, complete_labels=True, records=len(all_rows),
        candidate_failures=sum(r['status'] != 'complete' for r in all_rows),
        observed_heuristic_vetoes=[dict(d=int(r['d']), method=r['method'], comparator=r['comparator'])
            for r in comparisons if r['promotion_veto'] == 'TRUE'],
        interval_scope='pointwise conditional Monte Carlo; no simultaneous inference',
        ranking_or_default_claim=False, next_action='continue_frozen_ladder' if stage < 1000 else 'terminal_scientific_inspection',
        statistics_attempt=stats['name']))
    lines = [f'# Adaptive R replication stage{stage}', '',
        f'{len(all_rows)} records; {sum(r["status"] != "complete" for r in all_rows)} failed/capped learners. All scheduled labels retained.', '',
        'Pointwise95% empirical bootstrap intervals use2000 resamples and describe Monte Carlo uncertainty conditional on one data set per dimension.',
        'They do not identify author code or establish full-paper replication, population ranking, simultaneous inference or a new default.', '',
        '| d | Method | Complete | Ratio mean | Ratio SD [95% interval] | Relative RMSE [95% interval] | Mean final N |',
        '|---|---|---:|---:|---|---|---:|']
    for row in summary:
        if row['method'] not in METHODS[:2]:
            continue
        def fmt(key):
            return row[key] if row[key] == 'NA' else f'{float(row[key]):.5g}'
        lines.append(f'|{row["d"]}|{row["method"]}|{row["completed"]}/{row["n"]}|{fmt("ratio_mean")}|'
                     f'{fmt("ratio_sd")} [{fmt("sd_lower")}, {fmt("sd_upper")}]|'
                     f'{fmt("relative_rmse")} [{fmt("rmse_lower")}, {fmt("rmse_upper")}]|{fmt("mean_particles")}|')
    vetoes = [r for r in comparisons if r['promotion_veto'] == 'TRUE']
    lines += ['', 'Observed heuristic promotion vetoes: ' + ('; '.join(
        f'd{r["d"]} {r["method"]} vs {r["comparator"]}' for r in vetoes) or 'none in this sample') + '.', '',
        '| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |',
        '|---|---|---|---|---|---|',
        f'|Stage recorded|Complete labels and exact oracle checked|{len(vetoes)} observed heuristic comparisons veto promotion|Conditional Monte Carlo/tail uncertainty|'
        + ('Continue to next frozen stage' if stage < 1000 else 'Inspect terminal results') + '|Paper replication/default status|', '',
        '| Inference item | Status |', '|---|---|',
        '|Hard veto screen|Source, finite-state, oracle, wiring and label controls passed; candidate failures preserved|',
        '|Statistically supported ranking|No automatic ranking claim; intervals are pointwise only|',
        '|Descriptive-only differences|Observed moments, tails, counts and CPU times; bootstrap intervals conditional on the fixed data|',
        '|Default-readiness|Not evaluated; independent R extensions|',
        '|Next evidence needed|Review tail concentration and intervals; original implementation/data identity remains missing|', '',
        'Red-team: rare importance-weight events may be missing even when bootstrap intervals are narrow. Check maximum_loss_share, ratio tails and underflow counts.',
        'Particle counts differ between fitters; R times include explanatory score refits. An Eq15/source-faithfulness conclusion does not follow.']
    (directory / 'result.md').write_text('\n'.join(lines) + '\n')
    save(directory / 'manifest.json', dict(plan=PLAN, result=str(directory/'result.md'),
        outputs={str(p.relative_to(directory)): sha(p) for p in directory.iterdir() if p.is_file() and p.name != 'manifest.json'},
        attempts={v['attempt']: sha(ROOT/(v['attempt']+'-launch.json')) for v in selected},
        statistics_manifest_sha256=sha(ROOT/(stats['name']+'-launch.json'))))
    print(json.dumps(dict(stage_complete=stage, records=len(all_rows), candidate_failures=sum(r['status'] != 'complete' for r in all_rows))), flush=True)


def run_ladder():
    verify_prior()
    pre = read(ROOT / 'preflight.json')
    if not pre['passed'] or any(sha(REPO / p) != h for p, h in pre['current_sources'].items()):
        raise RuntimeError('Source-complete preflight required')
    old = read(ROOT/'checkpoint.json') if (ROOT/'checkpoint.json').exists() else {}
    if old.get('status') == 'running' and old.get('pid') != os.getpid():
        try:
            os.kill(old['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('An active supervisor already owns this campaign')
    if read(ROOT/'budget.json').get('reservations'):
        raise RuntimeError('Unresolved running/crashed launch reservations require inspection')
    completed = read(ROOT/'completed-batches.json') if (ROOT/'completed-batches.json').exists() else {}
    save(ROOT/'completed-batches.json', completed)
    try:
        for stage in [100, 300, 1000]:
            if (ROOT/f'stage{stage}/manifest.json').exists():
                continue
            tasks = [(d, first, first + 9, 1) for first in range(1, stage + 1, 10)
                     for d in [80, 40, 20, 10, 5] if f'd{d}_r{first:04}-{first+9:04}' not in completed]
            pending, failure = {}, None
            with ThreadPoolExecutor(max_workers=2) as pool:
                while tasks or pending:
                    while tasks and len(pending) < 2 and failure is None:
                        d, first, last, attempt = tasks.pop(0)
                        key = f'd{d}_r{first:04}-{last:04}'
                        name = key + f'_attempt{attempt:02}'
                        cap = 900 if attempt == 1 else 1800
                        try:
                            reserve(name, cap)
                        except Exception as error:
                            failure = str(error)
                            break
                        future = pool.submit(launch, name, SCRIPT,
                            lambda directory, d=d, first=first, last=last: [directory, d, (first-1)//10+1,
                                first, last, 971000+d, 972000+10000*d, 973000+10000*d, 'after_k'], cap)
                        pending[future] = (d, first, last, attempt, name)
                    checkpoint('running', stage, [v[4] for v in pending.values()],
                               failure or 'Executing frozen batches; candidate losses do not stop the ladder.')
                    if not pending:
                        if failure:
                            raise RuntimeError(failure)
                        break
                    done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        d, first, last, attempt, name = pending.pop(future)
                        try:
                            record = future.result()
                            settle(record)
                            if record['exit_code'] == 124 and record['sources_unchanged'] and attempt == 1:
                                tasks.insert(0, (d, first, last, 2))
                                save(ROOT / (name+'-repair.json'), dict(classification='infrastructure_time_cap',
                                    repair='fresh_directory_same_science_and_seeds_cap1800', prior_attempt=name,
                                    unchanged_contract=True, budget_charged=True))
                                continue
                            verify_launch(record)
                            verify_batch(ROOT / name, d, first, last)
                            completed[f'd{d}_r{first:04}-{last:04}'] = dict(d=d, first=first, last=last, attempt=name)
                            save(ROOT / 'completed-batches.json', completed)
                        except Exception as error:
                            failure = str(error)
                    if failure and not pending:
                        raise RuntimeError(failure)
            stage_summary(stage, completed)
        budget = read(ROOT/'budget.json')
        (ROOT/'result.md').write_text('# Adaptive R replication ladder finished\n\n'
            'All scheduled stages were executed. See [stage1000/result.md](stage1000/result.md) '
            'for the complete-label conditional statistics and pointwise intervals.\n'
            'No original-author, Eq15, full-paper-replication or default claim is issued. '
            'Terminal scientific inspection remains required before interpreting a ranking.\n'
            f'Charged CPU worker hours:{budget["phase_cpu_seconds"]/3600:.6f}; GPU hours:0.\n')
        save(ROOT/'manifest.json', dict(plan=PLAN, result=str(ROOT/'result.md'),
            stage_manifests={str(s): sha(ROOT/f'stage{s}/manifest.json') for s in [100,300,1000]},
            budget=budget, source_identity='independent_R_extensions', scientific_promotion=False))
        checkpoint('complete_pending_interpretation', 1000, [], 'All three stages completed; inspect terminal uncertainty and source gaps.')
    except Exception as error:
        (ROOT/'supervisor-error.log').write_text(traceback.format_exc())
        checkpoint('blocked', locals().get('stage', 0), [], str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['preflight', 'run'], required=True)
    arguments = parser.parse_args()
    {'preflight': preflight, 'run': run_ladder}[arguments.mode]()
