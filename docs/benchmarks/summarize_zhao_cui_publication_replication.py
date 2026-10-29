#!/usr/bin/env python3
"""Independent diagnostic/reporting of saved source-replication arrays (NumPy)."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def weighted_quantiles(values, weights, probabilities=(.05, .5, .95)):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return values[order][np.minimum(np.searchsorted(cumulative, probabilities), len(order)-1)]


def summarize(run):
    manifest = json.loads((run / 'manifest.json').read_text())
    settings = manifest['settings']
    with (run / 'smoothing-summary.csv').open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    entry = dict(run=run.name, status=manifest['status'], settings=settings, path_ess=rows,
                 complete_horizon=bool(rows and int(rows[-1]['time']) == settings['horizon'] and manifest['status'] == 'complete'))
    repetition_file = run / 'smoothing-repetitions.csv'
    quartiles = []
    if repetition_file.exists():
        with repetition_file.open() as stream:
            repetitions = [{k: float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        for t in sorted({int(row['time']) for row in repetitions}):
            group = [row for row in repetitions if int(row['time']) == t]
            values = np.asarray([row['ess_fraction'] for row in group])
            if not np.all(np.isfinite(values)) or np.any(values<=0) or np.any(values>1+1e-12):
                raise ValueError('invalid repeated ESS measurements')
            q25,median,q75 = np.quantile(values,[.25,.5,.75])
            quartiles.append(dict(time=t,completed_repetitions=len(group),
                requested_repetitions=(settings.get('smooth_repetitions',1)
                    if settings.get('repeat_times') is None or t in settings['repeat_times'] else 1),
                q25=float(q25),median=float(median),q75=float(q75),
                minimum=float(values.min()),maximum=float(values.max())))
    entry['conditional_smoothing_quartiles'] = quartiles
    entry['repetition_scope'] = 'fresh smoothing draws conditional on one fitted TT and dataset; excludes between-fit uncertainty'
    if not rows:
        return entry, None
    terminal = int(rows[-1]['time'])
    mat = loadmat(run / f'smoothing-t{terminal:02d}.mat')
    paths, weights = mat['sams'], mat['w'].reshape(-1)
    if not np.all(np.isfinite(paths)) or not np.all(np.isfinite(weights)) or abs(weights.sum()-1) > 1e-10:
        raise ValueError(f'invalid paths/weights in {run}')
    unweighted = np.quantile(paths, [.05, .5, .95], axis=1).transpose(1, 2, 0)
    weighted = np.empty_like(unweighted)
    for j in range(paths.shape[0]):
        for t in range(paths.shape[2]):
            weighted[j, t] = weighted_quantiles(paths[j, :, t], weights)
    truth = np.loadtxt(run / 'true-states.csv', delimiter=',', ndmin=2).T
    observations = np.loadtxt(run / 'observations.csv', delimiter=',', ndmin=2).T
    truth = truth[:, :terminal+1]
    entry.update(terminal=terminal, samples=len(weights), terminal_ess=float(1/np.sum(weights**2)),
                 max_weight=float(weights.max()), unweighted_quantiles=unweighted.tolist(), weighted_quantiles=weighted.tolist(),
                 interval_definition='unweighted NumPy linear quantiles; weighted inverse empirical CDF',
                 unweighted_truth_rmse_by_state=np.sqrt(np.mean((unweighted[:, :, 1]-truth)**2,axis=1)).tolist(),
                 weighted_truth_rmse_by_state=np.sqrt(np.mean((weighted[:, :, 1]-truth)**2,axis=1)).tolist(),
                 unweighted_truth_coverage_by_state=np.mean((truth>=unweighted[:, :, 0]) & (truth<=unweighted[:, :, 2]),axis=1).tolist(),
                 weighted_truth_coverage_by_state=np.mean((truth>=weighted[:, :, 0]) & (truth<=weighted[:, :, 2]),axis=1).tolist())
    return entry, (truth, observations, unweighted, weighted)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--published-ess', type=Path, help='Digitized Figure 15/17 CSV; no extrapolation')
    args = parser.parse_args()
    out = args.output_root.resolve(); out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    entries = []
    for run in args.run:
        entry, arrays = summarize(run.resolve()); entries.append(entry)
        settings = entry['settings']; pp = settings['model'] == 'pp'
        rows = entry['path_ess']
        label = f"{settings['route']}, rank {settings['rank']}, {settings['profile']}, seed {settings['fit_seed']}"
        if not entry['complete_horizon']: label += ' (partial)'
        axess=axes[0 if pp else 1]
        qs=[q for q in entry['conditional_smoothing_quartiles'] if q['completed_repetitions']>=2]
        if qs:
            first, = axess.plot([r['time'] for r in rows],[100*r['ess_fraction'] for r in rows],alpha=.4,linewidth=1)
            scope = 'medians + IQR' if len(qs)==len(rows) else 'first draw + terminal IQR'
            axess.errorbar([q['time'] for q in qs],[100*q['median'] for q in qs],
                yerr=[[100*(q['median']-q['q25']) for q in qs],[100*(q['q75']-q['median']) for q in qs]],
                marker='.',capsize=3,color=first.get_color(),
                label=f"Local {settings['route']}, rank {settings['rank']}: {scope}")
        else:
            axess.plot([r['time'] for r in rows], [100*r['ess_fraction'] for r in rows], marker='.',
                       label=f"Local {settings['route']}, rank {settings['rank']}: first draw")
        if arrays is None: continue
        truth, observations, unweighted, weighted = arrays
        selected = [0, 1] if pp else [0, 1, 8, 9]
        names = ['Prey', 'Predator'] if pp else ['Susceptible, region 1', 'Infectious, region 1', 'Susceptible, region 5', 'Infectious, region 5']
        step = 2 if pp else .02
        times = step*np.arange(entry['terminal']+1)
        pathfig, pathaxes = plt.subplots(len(selected)//2, 2, figsize=(10, 3*len(selected)//2), squeeze=False, constrained_layout=True)
        for ax, j, name in zip(pathaxes.flat, selected, names):
            ax.fill_between(times, unweighted[j,:,0], unweighted[j,:,2], color='#d5dde6', label='Unweighted 5–95%')
            ax.plot(times, unweighted[j,:,1], color='#202c3a', label='Unweighted median')
            ax.plot(times, weighted[j,:,1], '--', color='#3c79a0', alpha=.8, label='Importance-weighted median')
            ax.plot(times, truth[j], color='#a85d27', label='True state', linewidth=1.5)
            if pp or j%2:
                k = j if pp else j//2
                ax.scatter(times[1:], observations[k,:entry['terminal']], marker='x', color='#477fc5', s=18, label='Observation')
            ax.set(title=name, xlabel='Physical time', ylabel='Population')
        pathaxes.flat[0].legend(fontsize=8)
        pathfig.suptitle(f"{run.name}: smoothing through t={entry['terminal']}" + ('' if entry['complete_horizon'] else ' (partial)'))
        pathfig.savefig(out / f'{run.name}-trajectories.png', dpi=160)
        pathfig.savefig(out / f'{run.name}-trajectories.pdf')
        plt.close(pathfig)
    comparisons = []
    if args.published_ess:
        with args.published_ess.open() as stream:
            published = list(csv.DictReader(stream))
        for model, method in sorted({(r['model'], r['method']) for r in published}):
            group = [r for r in published if (r['model'],r['method']) == (model,method)]
            ax = axes[0 if model == 'pp' else 1]
            times = [int(r['time']) for r in group]
            line, = ax.plot(times, [100*float(r['median']) for r in group], '--', linewidth=1.5,
                            label=f'Paper {method}: median (digitized)')
            ax.fill_between(times, [100*float(r['q25']) for r in group], [100*float(r['q75']) for r in group],
                            color=line.get_color(), alpha=.12)
            for e in entries:
                settings = e['settings']
                own_model = 'pp' if settings['model'] == 'pp' else 'sir'
                own_method = settings['route'] if own_model == 'pp' else f"rank{settings['rank']}"
                if (own_model, own_method) != (model, method) or settings['profile'] != 'paper':
                    continue
                measured = e['conditional_smoothing_quartiles'] or [
                    dict(time=int(row['time']), completed_repetitions=1,
                         q25=row['ess_fraction'],median=row['ess_fraction'],q75=row['ess_fraction'])
                    for row in e['path_ess']]
                for q in measured:
                    source = next((r for r in group if int(r['time']) == q['time']), None)
                    if source is not None:
                        comparisons.append(dict(run=e['run'],time=q['time'],local_draws=q['completed_repetitions'],
                            local_q25=q['q25'],local_median=q['median'],local_q75=q['q75'],
                            published_q25=float(source['q25']),published_median=float(source['median']),
                            published_q75=float(source['q75']),
                            difference_percentage_points=100*(q['median']-float(source['median']))))
    if comparisons:
        with (out / 'published-comparison.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)
    for ax, name in zip(axes, ['Predator–prey', 'SIR, 18 states']):
        ax.set(title=name, xlabel='Observation update', ylabel='Joint-path ESS (%)', xlim=(0,20), ylim=(0,103))
        if ax.lines: ax.legend(fontsize=8)
    axes[0].axhline(40, color='#777777', linestyle=':', linewidth=1)
    axes[0].text(.5, 41.5, 'Paper text: approximately 40% at t=20', fontsize=8, color='#666666')
    fig.savefig(out / 'path-ess.png', dpi=160); fig.savefig(out / 'path-ess.pdf'); plt.close(fig)
    payload = dict(recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat(), runs=entries,
                   interpretation='descriptive numerical replication; different synthetic realization from paper',
                   statistically_supported_ranking=None, oracle_status='not_evaluated',
                   publication_source='https://jmlr.org/papers/v25/23-0743.html',
                   published_comparison=comparisons,
                   comparison_limitation='Different datasets; local repetitions condition on one fit. Differences are descriptive, not tests of the published results.')
    (out / 'summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    with (out / 'actual-values.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['run','time','ess','ess_fraction','max_weight','finite_fraction','smoothing_seconds'])
        for e in entries:
            for r in e['path_ess']:
                writer.writerow([e['run'], r['time'], r['ess'],r['ess_fraction'],r['max_weight'],r['finite_fraction'],r['seconds']])
    with (out / 'conditional-ess-quartiles.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['run','time','completed_repetitions','requested_repetitions','q25','median','q75','minimum','maximum'])
        for entry in entries:
            for row in entry['conditional_smoothing_quartiles']:
                writer.writerow([entry['run'],*[row[key] for key in ['time','completed_repetitions','requested_repetitions','q25','median','q75','minimum','maximum']]])
    print(json.dumps(dict(output=str(out), runs=len(entries), completed=sum(e['complete_horizon'] for e in entries))))


if __name__ == '__main__': main()
