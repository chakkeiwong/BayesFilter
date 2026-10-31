#!/usr/bin/env python3
"""Post-run diagnostic reporting only; no runtime tuning or oracle admission."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from collections import defaultdict
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', type=Path, required=True)
    parser.add_argument('--bootstrap', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bootstrap = read(args.bootstrap)
    scores, likelihoods, sources, groups, model_info = [], {}, [], defaultdict(list), {}
    for directory in args.run:
        result, manifest = read(directory/'result.json'), read(directory/'manifest.json')
        if result['status'] != 'complete' or result.get('diagnostic_only'):
            raise ValueError(f'not a complete full-path score run: {directory}')
        command = manifest['command']
        model = command[command.index('--model')+1]
        names = manifest['parameter_names']
        observations = {v['observation_sha256'] for v in result['proposals']}
        if len(observations) != 1:
            raise ValueError('mixed observations')
        observation = next(iter(observations))
        identity = (observation, tuple(names), tuple(manifest['theta0']))
        if model in model_info and model_info[model]['identity'] != identity:
            raise ValueError('different parameter/data targets under one model name')
        refs = [v for v in bootstrap if v['model']==model and v['observation_sha256']==observation]
        if not refs:
            raise ValueError('missing independent reference on identical observations')
        ref = max(refs, key=lambda v:v['particles'])
        model_info[model] = dict(identity=identity, names=names, reference=ref,
            theta_difference=[a-b for a,b in zip(manifest['theta0'],ref['theta'])])
        sources.append(dict(directory=str(directory), result_sha256=sha(directory/'result.json'),
                            manifest_sha256=sha(directory/'manifest.json')))
        for proposal in result['proposals']:
            likelihoods[proposal['directory']] = dict(model=model, proposal=proposal['directory'],
                rank=proposal['rank'], paths=proposal['evaluated_paths'], fit_seed=proposal['fit_seed'],
                log_likelihood=proposal['log_likelihood'],
                conditional_jackknife_se=proposal.get('conditional_log_likelihood_jackknife_se'),
                ess=proposal['ess'], max_weight=proposal['max_weight'],
                joint_density_parity_max=proposal['joint_density_parity_max'],
                bootstrap_log_likelihood=ref['log_likelihood'],
                bootstrap_mcse=ref['jackknife_mcse_log_likelihood'])
        for estimate in result['estimates']:
            proposal = result['proposals'][estimate['proposal']]
            for j,name in enumerate(names):
                se = estimate['conditional_path_jackknife'].get('standard_error')
                row = dict(model=model, parameter=name, rank=proposal['rank'],
                    paths=proposal['evaluated_paths'], fit_seed=proposal['fit_seed'],
                    design_seed=manifest['design_seed'], design_points=manifest['design_points'],
                    radius=estimate['radius'], score=estimate['score'][j],
                    conditional_jackknife_se=None if se is None else se[j],
                    central_difference=estimate['central_difference'][j],
                    prefix_quarter_score=estimate['prefix_scores'][0].get('score',[None]*len(names))[j],
                    prefix_half_score=estimate['prefix_scores'][1].get('score',[None]*len(names))[j],
                    heldout_rms=estimate['heldout_residual_rms'], minimum_ess=estimate['ess_min'],
                    bootstrap_score=ref['score'][j], bootstrap_mcse=ref['jackknife_mcse_score'][j])
                scores.append(row)
                key=tuple(row[k] for k in ('model','parameter','rank','paths','radius','design_seed','design_points'))
                groups[key].append(row)
    aggregate=[]
    for key, rows in sorted(groups.items()):
        if len({r['fit_seed'] for r in rows}) != len(rows):
            raise ValueError('duplicate fit seeds within a reporting group')
        n=len(rows); values=[r['score'] for r in rows]; mean=statistics.mean(values)
        mcse=statistics.stdev(values)/math.sqrt(n) if n>1 else None
        half=float(student_t.ppf(.975,n-1))*mcse if n>=3 else None
        jk=[r['conditional_jackknife_se'] for r in rows]
        row=dict(zip(('model','parameter','rank','paths','radius','design_seed','design_points'),key))
        row.update(fits=n,mean_score=mean,between_fit_mcse=mcse,between_fit_95_t_halfwidth=half,
            conditional_jackknife_se_of_mean=math.sqrt(sum(v*v for v in jk))/n if all(v is not None for v in jk) else None,
            central_minus_quadratic_mean=statistics.mean(r['central_difference']-r['score'] for r in rows),
            heldout_rms_max=max(r['heldout_rms'] for r in rows),minimum_ess=min(r['minimum_ess'] for r in rows),
            bootstrap_score=rows[0]['bootstrap_score'],bootstrap_mcse=rows[0]['bootstrap_mcse'],
            score_minus_bootstrap=mean-rows[0]['bootstrap_score'])
        aggregate.append(row)
    write_csv(out/'scores-by-fit.csv',scores)
    write_csv(out/'scores-aggregate.csv',aggregate)
    write_csv(out/'likelihoods-by-fit.csv',list(likelihoods.values()))
    for model,info in model_info.items():
        names=info['names']; fig,axes=plt.subplots(2 if len(names)>3 else 1,3,figsize=(14,7 if len(names)>3 else 3.8),squeeze=False)
        for j,(name,ax) in enumerate(zip(names,axes.flat)):
            selected=[r for r in aggregate if r['model']==model and r['parameter']==name]
            configurations=sorted({(r['rank'],r['paths']) for r in selected})
            for rank,paths in configurations:
                values=sorted([r for r in selected if r['rank']==rank and r['paths']==paths],key=lambda r:r['radius'])
                ax.errorbar([r['radius'] for r in values],[r['mean_score'] for r in values],
                    yerr=[r['between_fit_95_t_halfwidth'] or 0 for r in values],fmt='o-',capsize=3,
                    label=f'rank {rank}, N={paths:,}, fits={values[0]["fits"]}')
            ref=info['reference']; v=ref['score'][j]; se=ref['jackknife_mcse_score'][j]
            ax.axhline(v,color='black',ls='--',lw=1,label='Bootstrap mean and ±1 MCSE')
            ax.axhspan(v-se,v+se,color='black',alpha=.10)
            ax.set_xscale('log'); ax.set_title(name); ax.set_xlabel('Dimensionless radius h'); ax.set_ylabel('Score')
            ax.grid(alpha=.2)
        handles,labels=axes.flat[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9)
        fig.suptitle(model+': numerical score and radius sensitivity\nBars: 95% t intervals across fits; exclude radius and shared-support bias',fontsize=12)
        fig.tight_layout(rect=(0,.1,1,.93)); fig.savefig(out/(model+'-score-radii.png'),dpi=160); plt.close(fig)
    selected=[]
    for key in sorted({(r['model'],r['rank'],r['paths'],r['parameter']) for r in aggregate}):
        values=[r for r in aggregate if tuple(r[k] for k in ('model','rank','paths','parameter'))==key]
        selected.append(min(values,key=lambda r:r['radius']))
    lines=['# Numerical score reference: actual values','',
        'These are means of separately fitted finite-sample log-likelihood derivatives. The full CSV retains every radius and fit. The table uses the smallest measured radius at each rank and sample count; this is reporting, not a selection or oracle-admission rule.','',
        'Between-fit intervals use a Student t approximation with only three independent fits where available. They exclude shared support error and radius bias. Conditional jackknife errors measure path-sampling variation given the proposal. Neither error measure certifies accuracy. Bootstrap values have their own finite-particle bias; its displayed uncertainty is one MCSE, not a confidence interval.','',
        '| Model | Rank | Paths | Parameter | h | Mean score | Between-fit 95% t halfwidth | Conditional mean SE | Bootstrap | Bootstrap MCSE |','|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x): return '—' if x is None else f'{x:.8g}'
    for r in selected:
        lines.append('| '+' | '.join(str(r[k]) if k in ('model','parameter','rank','paths') else fmt(r[k]) for k in ('model','rank','paths','parameter','radius','mean_score','between_fit_95_t_halfwidth','conditional_jackknife_se_of_mean','bootstrap_score','bootstrap_mcse'))+' |')
    lines+=['','Parameter differences (TT minus bootstrap): '+json.dumps({m:i['theta_difference'] for m,i in model_info.items()})+'. The saved PP comparator rounds r and s to float32; the separately recorded common-path likelihood check measures that shift. A likelihood-shift check alone is not a score-shift bound.','',
        'No filter ranking or production-readiness claim is made. Interpret radius, rank, sample-count and independent-reference differences together.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    manifest=dict(schema='zhao_cui_score_diagnostic_report.v1',created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reporter_sha256=sha(Path(__file__)),bootstrap_sha256=sha(args.bootstrap),sources=sources,
        aggregation='arithmetic mean of independent-proposal finite-sample log-score estimates; no pooled-likelihood derivative claim',
        cpu_only=True,role='post-run reporting only; cannot tune or admit a runtime method')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),fits=len(likelihoods),score_rows=len(scores),aggregate_rows=len(aggregate))))

if __name__=='__main__':
    main()
